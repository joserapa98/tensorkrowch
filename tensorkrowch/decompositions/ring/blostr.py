"""
This script contains:

    Public classes:
        * BLOSTR
        * BLOSTRSVD

    Public functions:
        * tr_blostr
        * tr_blostr_svd

    Internal classes:
        * _FirstCore

    Internal functions:
        * _balanced_order
        * _spectral_order
        * _first_core
        * _recover_cores
"""

from dataclasses import dataclass
from math import isfinite, prod
from typing import Optional, Sequence, Tuple, Union
from warnings import warn

import torch

from tensorkrowch.utils import safe_divide

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.decompositions._utils import (_normalize_positive_int,
                                              _normalize_rank)
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                               ErrorRecord,
                                               TimingRecord)
from tensorkrowch.decompositions.observers import (_normalize_verbosity,
                                                 _resolve_observer,
                                                 DecompositionEvent)
from tensorkrowch.decompositions.results import (TensorDecomposition,
                                               TRDecomposition,
                                               _result_from_source)
from tensorkrowch.decompositions.ring.blocks import _split_block
from tensorkrowch.decompositions.sources.base import BaseSource
from tensorkrowch.decompositions.sources.factory import (SourceLike,
                                                       as_tensor_source)


_Rank = Optional[Union[int, Sequence[int]]]
_Device = Optional[Union[str, torch.device]]


@dataclass(frozen=True)
class _FirstCore:
    """Stores the first spectral core and the sampled projections."""

    core: torch.Tensor
    slices: Optional[Tuple[Tuple[int, ...], ...]]
    cluster_error: torch.Tensor


def _balanced_order(values: torch.Tensor,
                    cluster_size: int,
                    n_clusters: int,
                    n_iters: int,
                    n_restarts: int,
                    generator: Optional[torch.Generator]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Groups complex values into equal-size clusters by balanced k-means."""
    features = torch.view_as_real(values).detach().cpu()
    n_values = len(values)
    best_cost = None
    best_labels = None
    for _ in range(n_restarts):
        device = 'cpu' if generator is None else generator.device
        order = torch.randperm(n_values, generator=generator,
                               device=device).cpu()
        labels = torch.empty(n_values, dtype=torch.long)
        for cluster in range(n_clusters):
            labels[order[cluster * cluster_size:
                         (cluster + 1) * cluster_size]] = cluster
        for _ in range(n_iters):
            centers = torch.stack([
                features[labels == cluster].mean(0)
                for cluster in range(n_clusters)])
            distances = torch.cdist(features, centers)
            new_labels = torch.empty_like(labels)
            capacity = torch.zeros(n_clusters, dtype=torch.long)
            for index in distances.min(dim=1).values.argsort().tolist():
                for cluster in distances[index].argsort().tolist():
                    if capacity[cluster] < cluster_size:
                        new_labels[index] = cluster
                        capacity[cluster] += 1
                        break
            if torch.equal(new_labels, labels):
                break
            labels = new_labels
        centers = torch.stack([
            features[labels == cluster].mean(0)
            for cluster in range(n_clusters)])
        cost = (features - centers[labels]).square().sum()
        if (best_cost is None) or (cost < best_cost):
            best_cost, best_labels = cost, labels.clone()
    order = torch.cat([
        torch.where(best_labels == cluster)[0]
        for cluster in range(n_clusters)])
    return order.to(values.device), best_cost.to(values.device)


def _spectral_order(values: torch.Tensor,
                    ranks: Optional[Tuple[int, int]],
                    spectral_atol: float,
                    n_iters: int,
                    n_restarts: int,
                    generator: Optional[torch.Generator]
                    ) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
    """Finds balanced multiplicities, or orders prescribed spectral blocks."""
    if ranks is None:
        candidates = [(len(values) // count, count)
                      for count in range(1, len(values) + 1)
                      if len(values) % count == 0]
    else:
        candidates = [ranks]
    scale = values.abs().square().sum().clamp_min(
        torch.finfo(values.real.dtype).tiny)
    tolerance = max(
        spectral_atol, 10 * len(values) * torch.finfo(values.real.dtype).eps)
    for cyclic, adjacent in candidates:
        order, cost = _balanced_order(
            values, cyclic, adjacent, n_iters, n_restarts, generator)
        relative = (cost / scale).sqrt()
        if (ranks is not None) or (relative <= tolerance):
            # Interleave equal-eigenvalue blocks to expose TR bond indices.
            interleave = torch.cat([
                cyclic * torch.arange(adjacent, device=values.device) + offset
                for offset in range(cyclic)])
            return order[interleave], cyclic, adjacent, relative
    raise RuntimeError('BLOSTR could not identify balanced spectral blocks')


def _draw_slices(in_dim: Sequence[int],
                 generator: Optional[torch.Generator]
                 ) -> Tuple[Tuple[int, ...], ...]:
    """Draws distinct configurations for two independent slice ratios."""
    dimensions = tuple(in_dim[1:-1])
    n_configurations = prod(dimensions)
    if n_configurations < 3:
        raise ValueError(
            '`projection="slices"` requires three interior configurations')
    device = 'cpu' if generator is None else generator.device
    indices = torch.randperm(n_configurations, generator=generator,
                             device=device)[:4].cpu().tolist()
    configurations = []
    for index in indices:
        configuration = []
        for dimension in reversed(dimensions):
            configuration.append(index % dimension)
            index //= dimension
        configurations.append(tuple(reversed(configuration)))
    if len(configurations) == 3:
        configurations = [configurations[0], configurations[1],
                          configurations[0], configurations[2]]
    return tuple(configurations)


def _validate_slices(slices: Sequence[Sequence[int]],
                     dimensions: Sequence[int]
                     ) -> Tuple[Tuple[int, ...], ...]:
    """Validates four interior index configurations."""
    if isinstance(slices, (str, bytes)) or not isinstance(slices, Sequence):
        raise TypeError('`slices` should contain four index sequences')
    configurations = []
    for configuration in slices:
        if isinstance(configuration, (str, bytes)) or not isinstance(
                configuration, Sequence):
            raise TypeError('`slices` should contain four index sequences')
        configuration = tuple(configuration)
        if len(configuration) != len(dimensions) - 2:
            raise ValueError('Each entry of `slices` should index every '
                             'interior axis')
        for index, dimension in zip(configuration, dimensions[1:-1]):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError('Indices in `slices` should be integers')
            if (index < 0) or (index >= dimension):
                raise ValueError('An index in `slices` is out of bounds')
        configurations.append(configuration)
    if len(configurations) != 4:
        raise ValueError('`slices` should contain four index sequences')
    if (configurations[0] == configurations[1]) or \
            (configurations[2] == configurations[3]):
        raise ValueError(
            'Each ratio in `slices` needs distinct configurations')
    return tuple(configurations)


def _first_core(tensor: torch.Tensor,
                ranks: Optional[Sequence[int]],
                slices: Optional[Sequence[Sequence[int]]],
                projection: str,
                spectral_atol: float,
                n_iters: int,
                n_restarts: int,
                generator: Optional[torch.Generator]) -> _FirstCore:
    """Recovers the first core by simultaneous block diagonalization."""
    # Both projection modes leave the first and last physical axes open.
    if projection == 'slices':
        if slices is None:
            slices = _draw_slices(tensor.shape, generator)
        matrices = [tensor[(slice(None), *index, slice(None))]
                    for index in slices]
    else:
        device = tensor.device if generator is None else generator.device
        weights = torch.randn(4, prod(tensor.shape[1:-1]),
                              dtype=tensor.dtype, device=device,
                              generator=generator).to(tensor.device)
        matrices = torch.einsum(
            'amb,sm->sab', tensor.reshape(tensor.shape[0], -1,
                                         tensor.shape[-1]), weights)

    # Two independent ratios expose repeated spectral subspaces.
    ratios = (matrices[0] @ torch.linalg.pinv(matrices[1]),
              matrices[2] @ torch.linalg.pinv(matrices[3]))
    spaces = []
    pair = None if ranks is None else (ranks[-1], ranks[0])
    for matrix in ratios:
        values, vectors = torch.linalg.eig(matrix)
        selected = values.abs().argsort(descending=True)
        threshold = max(
            spectral_atol, max(matrix.shape) *
            torch.finfo(values.real.dtype).eps * values.abs().max())
        selected = selected[values[selected].abs() > threshold]
        if pair is not None:
            size = prod(pair)
            if len(selected) < size:
                raise RuntimeError('BLOSTR found too few nonzero eigenvalues '
                                   'for `rank`')
            selected = selected[:size]
        if not len(selected):
            raise RuntimeError('BLOSTR found no nonzero eigenvalues')
        order, cyclic, adjacent, error = _spectral_order(
            values[selected], pair, spectral_atol, n_iters, n_restarts,
            generator)
        pair = cyclic, adjacent
        spaces.append(vectors[:, selected[order]])
    cyclic, adjacent = pair
    if min(tensor.shape[0], tensor.shape[-1]) < cyclic * adjacent:
        raise ValueError('The first and last axes of `tensor` should contain '
                         'at least the product of the first two TR ranks')

    # Align the degenerate eigenspaces to separate cyclic and local indices.
    alignment = torch.linalg.pinv(spaces[0]) @ spaces[1]
    first_block = alignment[0::adjacent, 0::adjacent]
    transform = torch.zeros_like(alignment)
    for block in range(adjacent):
        current = alignment[block::adjacent, 0::adjacent]
        factor = torch.linalg.solve(current.mT, first_block.mT).mT
        transform[block::adjacent, block::adjacent] = torch.linalg.inv(factor)
    matrix = spaces[0] @ transform
    return _FirstCore(
        matrix.reshape(tensor.shape[0], cyclic, adjacent).permute(1, 0, 2),
        None if slices is None else tuple(tuple(s) for s in slices), error)


def _recover_cores(tensor: torch.Tensor,
                   first: torch.Tensor,
                   projection: str,
                   generator: Optional[torch.Generator]
                   ) -> Tuple[torch.Tensor, ...]:
    """Recovers cores with slices or separate dense past/future projections."""
    rank = first.shape[0]
    device = tensor.device if generator is None else generator.device
    if projection == 'slices':
        indices = tuple(int(torch.randint(
            size, (), device=device, generator=generator))
            for size in tensor.shape)
    cores = [first]
    prefix = torch.eye(rank, dtype=tensor.dtype, device=tensor.device)
    for site in range(1, tensor.ndim):
        if projection == 'slices':
            selection = list(indices)
            selection[site - 1] = selection[site] = slice(None)
            matrix = tensor[tuple(selection)].mT
            if site > 1:
                prefix = prefix @ cores[-2][:, indices[site - 2], :]
            product = prefix
        else:
            # Keep the known prefix separate from the unknown suffix, but
            # allow dense correlations between physical axes within each.
            weights = []
            for dimensions in (tensor.shape[:site - 1],
                               tensor.shape[site + 1:]):
                if dimensions:
                    weight = torch.randn(
                        prod(dimensions), dtype=tensor.dtype, device=device,
                        generator=generator).to(tensor.device)
                else:
                    weight = tensor.new_ones(1)
                weights.append(weight / weight.norm())
            left, right = weights
            matrix = torch.einsum(
                'a,aijb,b->ji', left, tensor.reshape(
                    left.numel(), tensor.shape[site - 1], tensor.shape[site],
                    right.numel()), right)
            if site > 1:
                prefix = torch.tensordot(prefix, cores[-2], dims=([-1], [0]))
            product = torch.einsum(
                'aib,i->ab', prefix.reshape(rank, -1, rank), left)
        previous = torch.einsum('ab,bic->iac', product, cores[-1])
        previous = previous.reshape(-1, rank * rank).mT
        current = (matrix @ torch.linalg.pinv(previous)).reshape(
            tensor.shape[site], rank, rank).permute(2, 0, 1)
        cores.append(current)
    return tuple(cores)


class BLOSTR:  # MARK: BLOSTR
    r"""
    Recovers a tensor ring from a finite source using spectral BLOSTR.

    Two pairs of interior projections identify the first core through blockwise
    simultaneous diagonalization. Each remaining core is recovered by a linear
    solve with its physical axis and the previous one left open. Slice
    projection uses discrete indices throughout, following Section 3 of `A
    Provably Efficient Method for Tensor Ring Decomposition and Its
    Applications <https://arxiv.org/abs/2512.01016>`_. Gaussian projection is
    an extension: the first step jointly projects all interior axes; later
    steps use independent dense projections of the known prefix and unknown
    suffix, preserving separability only between those two groups.

    The full construction targets exact tensors with uniform ranks, injective
    physical unfoldings at every site and nondegenerate projection ratios. Each
    physical dimension must therefore be at least the squared TR rank. With
    slices, the selected recovery environments must also be full rank;
    injectivity alone does not ensure this. Blocking physical sites, when
    needed, is left to the caller. :class:`BLOSTRSVD` instead requires
    injectivity only at its two external physical axes. No accuracy guarantee
    is made for approximate inputs or rank-deficient recovery environments.

    Parameters
    ----------
    source : BaseSource, TensorFormat1D, MPS, MPO or torch.Tensor
        Finite tensor source, format or dense tensor. The complete tensor is
        materialized once when constructing the algorithm. At least three sites
        are required; structural batches and continuous sources are
        unsupported. Matrix sources group input/output axes at each site.
    out_device : str or torch.device, optional
        Final core storage device. Default is ``"cpu"``; ``None`` keeps the
        input device.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(torch.ones(3, 3, 3))
    >>> result = tk.decompositions.BLOSTR(source).fit(rank=1)
    >>> result.rank
    [1, 1, 1]
    """

    _algorithm = 'blostr'

    def __init__(self, source: SourceLike, *,
                 out_device: _Device = 'cpu') -> None:
        self._source = as_tensor_source(source)
        if not self._source.is_discrete_tensor:
            raise ValueError('`source` should represent a discrete tensor')
        if self._source.n_batches:
            raise ValueError('BLOSTR does not support `n_batches`')
        tensor = self._source.contract_dense()
        if self._source.is_matrix:
            dimensions = tuple(a * b for a, b in zip(
                self._source.in_dim, self._source.out_dim))
            tensor = tensor.reshape(dimensions)
        if tensor.ndim < 3:
            raise ValueError('`source` should contain at least three sites')
        if any(size < 1 for size in tensor.shape):
            raise ValueError('`source` should have positive input dimensions')
        if not (tensor.is_floating_point() or tensor.is_complex()):
            raise TypeError('`source` should have floating or complex dtype')
        if not torch.isfinite(tensor).all():
            raise ValueError('`source` should contain finite values')
        dtype = torch.complex128 if tensor.dtype in (
            torch.float64, torch.complex128) else torch.complex64

        # Cache the materialization in the complex working dtype for repeated
        # fits without modifying the source's storage.
        self._tensor = tensor.to(dtype=dtype)
        self._runtime = _RuntimePolicy.from_tensor(
            self._tensor, out_device=out_device)

    @property
    def source(self) -> BaseSource:
        """Source describing the fixed decomposition input."""
        return self._source

    def _validate_rank(self, rank: Optional[Sequence[int]]) -> None:
        """Checks the uniform ranks and physical capacities of full BLOSTR."""
        if rank is None:
            return
        if len(set(rank)) != 1:
            warn('Full BLOSTR does not support different TR ranks; use '
                 'BLOSTR-SVD instead.', UserWarning, stacklevel=4)
            raise ValueError('Full BLOSTR requires uniform `rank`')
        if min(self._tensor.shape) < rank[0]**2:
            warn('Full BLOSTR cannot satisfy injectivity: a physical '
                 'dimension is smaller than the product of its TR ranks. '
                 'Block the TT format before TT2TR, or group the physical '
                 'axes of a dense input, to meet this dimension condition.',
                 UserWarning, stacklevel=4)
            raise RuntimeError('Full BLOSTR requires every physical '
                               'dimension to be at least rank**2')

    def _recover(self, first: torch.Tensor,
                 rank: Optional[Sequence[int]], projection: str,
                 spectral_atol: float, generator: Optional[torch.Generator],
                 truncation: Optional[_TruncationSpec],
                 metrics: Optional[DecompositionMetrics]
                 ) -> Tuple[torch.Tensor, ...]:
        """Recovers every remaining core by the full spectral construction."""
        cyclic, _, adjacent = first.shape
        if cyclic != adjacent:
            warn('Full BLOSTR does not support different TR ranks; use '
                 'BLOSTR-SVD instead.', UserWarning, stacklevel=4)
            raise ValueError('Full BLOSTR requires uniform discovered ranks')
        if rank is None:
            self._validate_rank((cyclic, adjacent))
        return _recover_cores(self._tensor, first, projection, generator)

    def _fit(self, rank: _Rank, *, projection: str,
             slices: Optional[Sequence[Sequence[int]]], spectral_atol: float,
             n_attempts: int, n_iters: int, n_restarts: int,
             generator: Optional[torch.Generator],
             truncation: Optional[_TruncationSpec] = None,
             collect_metrics: bool, verbose: Union[bool, int]
             ) -> TensorDecomposition:
        """Retains the smallest reconstruction error over spectral attempts."""
        ranks = None if rank is None else _normalize_rank(
            rank, self._tensor.ndim)
        self._validate_rank(ranks)
        if projection not in ('slices', 'gaussian'):
            raise ValueError('`projection` should be "slices" or "gaussian"')
        if (slices is not None) and (projection != 'slices'):
            raise ValueError('Explicit `slices` require `projection="slices"`')
        if slices is not None:
            slices = _validate_slices(slices, self._tensor.shape)
        if isinstance(spectral_atol, bool) or not isinstance(
                spectral_atol, (int, float)):
            raise TypeError('`spectral_atol` should be a non-negative number')
        if (spectral_atol < 0) or (not isfinite(spectral_atol)):
            raise ValueError(
                '`spectral_atol` should be finite and non-negative')
        for name, value in (('n_attempts', n_attempts), ('n_iters', n_iters),
                            ('n_restarts', n_restarts)):
            _normalize_positive_int(value, name)
        if generator is not None and not isinstance(
                generator, torch.Generator):
            raise TypeError(
                '`generator` should be torch.Generator type or None')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        phase = self._algorithm.upper().replace('_', '-')
        verbosity = _normalize_verbosity(verbose)
        observer = _resolve_observer(verbosity, None) if verbosity else None
        if observer is not None:
            observer.emit(DecompositionEvent(
                name='start', phase=phase,
                values={'rank': rank, 'attempts': n_attempts}))
        collect_metrics = collect_metrics or bool(verbosity)
        active = self._tensor
        metrics = DecompositionMetrics()
        failures = []
        best = None
        attempts = n_attempts if slices is None else 1
        with self._runtime.timer() as timer:

            # Reject degenerate projections and keep the best reconstruction.
            for attempt in range(attempts):
                local_metrics = DecompositionMetrics()
                try:
                    first = _first_core(active, ranks, slices, projection,
                                        spectral_atol, n_iters, n_restarts,
                                        generator)
                    cores = self._recover(
                        first.core, ranks, projection, spectral_atol,
                        generator, truncation,
                        local_metrics if collect_metrics else None)
                    candidate = TRDecomposition(cores)
                    approximation = candidate.contract_dense()
                    absolute = torch.linalg.vector_norm(
                        approximation - active)
                    denominator = torch.linalg.vector_norm(active)
                    relative = safe_divide(absolute, denominator,
                                           zero_value=absolute)
                    if (best is None) or (relative < best[0]):
                        best = (relative, candidate, first,
                                local_metrics.truncations, absolute)
                    if observer is not None:
                        observer.emit(DecompositionEvent(
                            name='attempt_complete', phase=phase, values={
                                'attempt': attempt + 1, 'attempts': attempts,
                                'relative_error': relative.detach()}))
                except (RuntimeError, ValueError) as error:
                    failures.append(str(error))
        if best is None:
            detail = '; '.join(dict.fromkeys(failures))
            raise RuntimeError(
                f'Every BLOSTR spectral attempt failed. {detail}')

        # Report the accepted attempt, not the final projection that was tried.
        relative, candidate, first, records, absolute = best
        if collect_metrics:
            metrics.errors.append(ErrorRecord(
                kind='reconstruction', absolute=absolute, relative=relative,
                denominator=torch.linalg.vector_norm(active)))
            metrics.truncations.extend(records)
            metrics.timings.append(TimingRecord(name='fit',
                                               elapsed=timer.elapsed))
            metrics.warnings.extend(failures)
        result = TRDecomposition(
            [self._runtime.finalize(core) for core in candidate.cores],
            metrics=metrics,
            metadata={
                'algorithm': self._algorithm,
                'requested_rank': None if ranks is None else list(ranks),
                'adaptive': ranks is None, 'projection': projection,
                'slices': first.slices, 'attempts': attempts,
                'cluster_error': first.cluster_error.detach(),
            })
        result = _result_from_source(result, self._source)
        if observer is not None:
            observer.emit(DecompositionEvent(
                name='summary', phase=phase, values={
                    'rank': result.rank, 'absolute_error': absolute,
                    'relative_error': relative, 'elapsed': timer.elapsed}))
            for site, core in enumerate(result.cores):
                observer.emit(DecompositionEvent(
                    name='core', phase=phase, level=3, site=site,
                    values={'shape': tuple(core.shape), 'tensor': core}))
            observer.close(metrics)
        return result

    def fit(self,
            rank: _Rank = None,
            *,
            projection: str = 'slices',
            slices: Optional[Sequence[Sequence[int]]] = None,
            spectral_atol: float = 1e-10,
            n_attempts: int = 8,
            n_iters: int = 50,
            n_restarts: int = 10,
            generator: Optional[torch.Generator] = None,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0) -> TensorDecomposition:
        r"""
        Runs spectral recovery with prescribed or discovered TR ranks.

        Parameters
        ----------
        rank : int or sequence[int], optional
            Prescribed right-link ranks, including the cyclic last entry. Full
            BLOSTR requires equal ranks. With ``None``, balanced k-means
            discovers the multiplicities of the nonzero spectrum and requires
            them to imply equal cyclic and adjacent ranks.
        slices : sequence of four index sequences, optional
            Configurations fixing every interior axis for the two spectral
            ratios. With ``None``, configurations are drawn using
            ``generator``.
        projection : {"slices", "gaussian"}
            Coordinate slices or dense Gaussian projections. The first step
            uses four interior projections. Full BLOSTR then uses discrete
            indices with ``"slices"``, or independent dense known/unknown
            contractions with ``"gaussian"``. BLOSTR-SVD instead splits the
            tail with SVDs. Explicit ``slices`` require ``"slices"``.
        spectral_atol : float
            Absolute eigenvalue cutoff and relative within-cluster tolerance
            used when discovering balanced spectral multiplicities. A
            machine-precision floor is applied to both tests. Default is 1e-10.
        n_attempts : int
            Number of independent spectral attempts. Default is 8; explicit
            ``slices`` use one attempt. The smallest reconstruction error is
            kept.
        n_iters, n_restarts : int
            Balanced k-means iteration and restart budgets. Defaults are 50 and
            10.
        generator : torch.Generator, optional
            Controls spectral projections, balanced clustering and recovery
            projections.
        collect_metrics : bool
            Whether to retain reconstruction, truncation and timing
            diagnostics.
        verbose : bool or int
            Console verbosity from 0 (silent) to 3.

        Returns
        -------
        TensorDecomposition
            Recovered TR cores and construction metadata. Spectral factors use
            a complex dtype even for real input tensors.

        Warns
        -----
        UserWarning
            Different prescribed or discovered TR ranks recommend BLOSTR-SVD.
            An insufficient physical dimension recommends blocking the input.
            Both conditions still raise an error; blocking meets a necessary
            dimension condition, but does not guarantee numerical injectivity.
        """
        return self._fit(
            rank, projection=projection, slices=slices,
            spectral_atol=spectral_atol, n_attempts=n_attempts,
            n_iters=n_iters, n_restarts=n_restarts, generator=generator,
            collect_metrics=collect_metrics, verbose=verbose)


class BLOSTRSVD(BLOSTR):  # MARK: BLOSTRSVD
    r"""
    Recovers the first TR core spectrally and splits the rest with TT-SVD.

    Unlike :class:`BLOSTR`, interior ranks may differ. The two ranks bounding
    the projected interior chain must coincide: ``rank[0] == rank[-2]``. The
    cyclic rank may differ. Gaussian joint projections can retain full matrix
    rank even when an intermediate bond makes each coordinate slice deficient.
    The physical first and last axes must both have dimension at least
    ``rank[-1] * rank[0]``. These conditions permit spectral recovery; they do
    not guarantee spectral separation for a particular projection.

    Parameters
    ----------
    source : BaseSource, TensorFormat1D, MPS, MPO or torch.Tensor
        Finite tensor source, format or dense tensor. The complete tensor is
        materialized once when constructing the algorithm. At least three sites
        are required; structural batches and continuous sources are
        unsupported. Matrix sources group input/output axes at each site.
    out_device : str or torch.device, optional
        Final core storage device. Default is ``"cpu"``; ``None`` keeps the
        input device.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(torch.ones(3, 3, 3))
    >>> result = tk.decompositions.BLOSTRSVD(source).fit(rank=1)
    >>> result.rank
    [1, 1, 1]
    """

    _algorithm = 'blostr_svd'

    def _validate_rank(self, rank: Optional[Sequence[int]]) -> None:
        """Checks the matching ranks bounding the projected interior chain."""
        if (rank is not None) and (rank[0] != rank[-2]):
            raise ValueError('BLOSTR-SVD requires `rank[0] == rank[-2]`')

    def _recover(self, first: torch.Tensor,
                 rank: Optional[Sequence[int]], projection: str,
                 spectral_atol: float, generator: Optional[torch.Generator],
                 truncation: Optional[_TruncationSpec],
                 metrics: Optional[DecompositionMetrics]
                 ) -> Tuple[torch.Tensor, ...]:
        """Removes the first factor and splits the remaining open chain."""
        cyclic, _, adjacent = first.shape
        first_matrix = first.permute(1, 0, 2).reshape(
            self._tensor.shape[0], -1)
        tail = torch.linalg.pinv(first_matrix) @ self._tensor.reshape(
            self._tensor.shape[0], -1)
        tail = tail.reshape(cyclic, adjacent, *self._tensor.shape[1:])
        tail = tail.movedim(0, -1)
        options = truncation.as_kwargs()
        if (rank is None) and all(
                options.get(name) is None for name in
                ('cutoff', 'atol', 'rtol', 'cum_percentage')):
            precision = (10 * max(self._tensor.shape) *
                         torch.finfo(self._tensor.real.dtype).eps *
                         torch.linalg.cond(first_matrix))
            options['rtol'] = min(max(spectral_atol, float(precision))**2, 1)
        tail_cores = _split_block(
            tail, self._tensor.shape[1:],
            rank=None if rank is None else rank[1:-1],
            cutoff=options.get('cutoff'), atol=options.get('atol'),
            rtol=options.get('rtol'),
            cum_percentage=options.get('cum_percentage'), metrics=metrics)
        return (first, *tail_cores)

    def fit(self,
            rank: _Rank = None,
            *,
            projection: str = 'slices',
            slices: Optional[Sequence[Sequence[int]]] = None,
            spectral_atol: float = 1e-10,
            n_attempts: int = 8,
            n_iters: int = 50,
            n_restarts: int = 10,
            generator: Optional[torch.Generator] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0) -> TensorDecomposition:
        r"""
        Recovers the first core spectrally and splits the tail with SVDs.

        Parameters
        ----------
        rank : int or sequence[int], optional
            Prescribed right-link ranks, including the cyclic last entry. With
            ``None``, balanced k-means discovers the first cyclic and adjacent
            ranks; TT-SVD discovers the remaining bonds numerically.
        slices : sequence of four index sequences, optional
            Configurations fixing every interior axis for the two spectral
            ratios. With ``None``, configurations are drawn using
            ``generator``.
        projection : {"slices", "gaussian"}
            Coordinate slices or dense Gaussian projections. The first step
            uses four interior projections. Full BLOSTR then uses discrete
            indices with ``"slices"``, or independent dense known/unknown
            contractions with ``"gaussian"``. BLOSTR-SVD instead splits the
            tail with SVDs. Explicit ``slices`` require ``"slices"``.
        spectral_atol : float
            Absolute eigenvalue cutoff and relative within-cluster tolerance
            used when discovering balanced spectral multiplicities. A
            machine-precision floor is applied to both tests. Default is 1e-10.
        n_attempts : int
            Number of independent spectral attempts. Default is 8; explicit
            ``slices`` use one attempt. The smallest reconstruction error is
            kept.
        n_iters, n_restarts : int
            Balanced k-means iteration and restart budgets. Defaults are 50 and
            10.
        generator : torch.Generator, optional
            Controls spectral projections, balanced clustering and recovery
            projections.
        cutoff : float, optional
            Singular values at or below this value are discarded in tail
            TT-SVD.
        atol, rtol, cum_percentage : float, optional
            Absolute or relative discarded-energy tolerance, or retained
            squared singular-value fraction, in tail TT-SVD. With no rank or
            tolerance, a numerical threshold accounts for the precision and
            conditioning of the recovered first core. A supplied rank is an
            upper bound; smaller retained ranks are returned without zero
            padding.
        collect_metrics : bool
            Whether to retain reconstruction, truncation and timing
            diagnostics.
        verbose : bool or int
            Console verbosity from 0 (silent) to 3.

        Returns
        -------
        TensorDecomposition
            Recovered TR cores and construction metadata. Spectral factors use
            a complex dtype even for real input tensors.
        """
        truncation = _TruncationSpec(cutoff=cutoff, atol=atol, rtol=rtol,
                                     cum_percentage=cum_percentage)
        return self._fit(
            rank, projection=projection, slices=slices,
            spectral_atol=spectral_atol, n_attempts=n_attempts,
            n_iters=n_iters, n_restarts=n_restarts, generator=generator,
            truncation=truncation,
            collect_metrics=collect_metrics, verbose=verbose)


def tr_blostr(source: SourceLike,
              rank: _Rank = None,
              *,
              slices: Optional[Sequence[Sequence[int]]] = None,
              projection: str = 'slices',
              spectral_atol: float = 1e-10,
              n_attempts: int = 8,
              n_iters: int = 50,
              n_restarts: int = 10,
              generator: Optional[torch.Generator] = None,
              out_device: _Device = 'cpu',
              collect_metrics: bool = False,
              verbose: Union[bool, int] = 0) -> TensorDecomposition:
    r"""
    Decomposes a finite source using :class:`BLOSTR`.

    This is equivalent to ``BLOSTR(source, out_device=...).fit(...)``. The
    source is materialized once; returned cores use complex dtype. See
    :meth:`BLOSTR.fit` for spectral options, restrictions and diagnostics.

    Parameters
    ----------
    source : BaseSource, TensorFormat1D, MPS, MPO or torch.Tensor
        Finite source, format or dense tensor to decompose.
    rank : int or sequence[int], optional
        Prescribed right-link ranks, including the cyclic last entry. Full
        BLOSTR requires equal ranks. With ``None``, balanced k-means discovers
        the multiplicities of the nonzero spectrum and requires them to imply
        equal cyclic and adjacent ranks.
    slices : sequence of four index sequences, optional
        Configurations fixing every interior axis for the two spectral ratios.
        With ``None``, configurations are drawn using ``generator``.
    projection : {"slices", "gaussian"}
        Coordinate slices or dense Gaussian projections. The first step uses
        four interior projections. Full BLOSTR then uses discrete indices with
        ``"slices"``, or independent dense known/unknown contractions with
        ``"gaussian"``. BLOSTR-SVD instead splits the tail with SVDs. Explicit
        ``slices`` require ``"slices"``.
    spectral_atol : float
        Absolute eigenvalue cutoff and relative within-cluster tolerance used
        when discovering balanced spectral multiplicities. A machine-precision
        floor is applied to both tests. Default is 1e-10.
    n_attempts : int
        Number of independent spectral attempts. Default is 8; explicit
        ``slices`` use one attempt. The smallest reconstruction error is kept.
    n_iters, n_restarts : int
        Balanced k-means iteration and restart budgets. Defaults are 50 and 10.
    generator : torch.Generator, optional
        Controls spectral projections, balanced clustering and recovery
        projections.
    out_device : str or torch.device, optional
        Final core storage device. ``None`` keeps the input device.
    collect_metrics : bool
        Whether to retain reconstruction, truncation and timing diagnostics.
    verbose : bool or int
        Console verbosity from 0 (silent) to 3.

    Returns
    -------
    TensorDecomposition
        Recovered TR cores and construction metadata. Spectral factors use a
        complex dtype even for real input tensors.

    Warns
    -----
    UserWarning
        Different prescribed or discovered TR ranks recommend BLOSTR-SVD. An
        insufficient physical dimension recommends blocking the input. Both
        conditions still raise an error; blocking meets a necessary dimension
        condition, but does not guarantee numerical injectivity.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(torch.ones(3, 3, 3))
    >>> result = tk.decompositions.tr_blostr(source, rank=1)
    >>> result.rank
    [1, 1, 1]
    """
    return BLOSTR(source, out_device=out_device).fit(
        rank, projection=projection, slices=slices,
        spectral_atol=spectral_atol, n_attempts=n_attempts, n_iters=n_iters,
        n_restarts=n_restarts, generator=generator,
        collect_metrics=collect_metrics, verbose=verbose)


def tr_blostr_svd(source: SourceLike,
                  rank: _Rank = None,
                  *,
                  slices: Optional[Sequence[Sequence[int]]] = None,
                  projection: str = 'slices',
                  spectral_atol: float = 1e-10,
                  n_attempts: int = 8,
                  n_iters: int = 50,
                  n_restarts: int = 10,
                  generator: Optional[torch.Generator] = None,
                  cutoff: Optional[float] = None,
                  atol: Optional[float] = None,
                  rtol: Optional[float] = None,
                  cum_percentage: Optional[float] = None,
                  out_device: _Device = 'cpu',
                  collect_metrics: bool = False,
                  verbose: Union[bool, int] = 0) -> TensorDecomposition:
    r"""
    Decomposes a finite source using :class:`BLOSTRSVD`.

    This is equivalent to ``BLOSTRSVD(source, out_device=...).fit(...)``. The
    source is materialized once; returned cores use complex dtype. See
    :meth:`BLOSTRSVD.fit` for spectral options, restrictions and diagnostics.

    Parameters
    ----------
    source : BaseSource, TensorFormat1D, MPS, MPO or torch.Tensor
        Finite source, format or dense tensor to decompose.
    rank : int or sequence[int], optional
        Prescribed right-link ranks, including the cyclic last entry. With
        ``None``, balanced k-means discovers the first cyclic and adjacent
        ranks; TT-SVD discovers the remaining bonds numerically.
    slices : sequence of four index sequences, optional
        Configurations fixing every interior axis for the two spectral ratios.
        With ``None``, configurations are drawn using ``generator``.
    projection : {"slices", "gaussian"}
        Coordinate slices or dense Gaussian projections. The first step uses
        four interior projections. Full BLOSTR then uses discrete indices with
        ``"slices"``, or independent dense known/unknown contractions with
        ``"gaussian"``. BLOSTR-SVD instead splits the tail with SVDs. Explicit
        ``slices`` require ``"slices"``.
    spectral_atol : float
        Absolute eigenvalue cutoff and relative within-cluster tolerance used
        when discovering balanced spectral multiplicities. A machine-precision
        floor is applied to both tests. Default is 1e-10.
    n_attempts : int
        Number of independent spectral attempts. Default is 8; explicit
        ``slices`` use one attempt. The smallest reconstruction error is kept.
    n_iters, n_restarts : int
        Balanced k-means iteration and restart budgets. Defaults are 50 and 10.
    generator : torch.Generator, optional
        Controls spectral projections, balanced clustering and recovery
        projections.
    cutoff : float, optional
        Singular values at or below this value are discarded in tail TT-SVD.
    atol, rtol, cum_percentage : float, optional
        Absolute or relative discarded-energy tolerance, or retained squared
        singular-value fraction, in tail TT-SVD. With no rank or tolerance, a
        numerical threshold accounts for the precision and conditioning of the
        recovered first core. A supplied rank is an upper bound; smaller
        retained ranks are returned without zero padding.
    out_device : str or torch.device, optional
        Final core storage device. ``None`` keeps the input device.
    collect_metrics : bool
        Whether to retain reconstruction, truncation and timing diagnostics.
    verbose : bool or int
        Console verbosity from 0 (silent) to 3.

    Returns
    -------
    TensorDecomposition
        Recovered TR cores and construction metadata. Spectral factors use a
        complex dtype even for real input tensors.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(torch.ones(3, 3, 3))
    >>> result = tk.decompositions.tr_blostr_svd(source, rank=1)
    >>> result.rank
    [1, 1, 1]
    """
    return BLOSTRSVD(source, out_device=out_device).fit(
        rank, projection=projection, slices=slices,
        spectral_atol=spectral_atol, n_attempts=n_attempts, n_iters=n_iters,
        n_restarts=n_restarts, generator=generator, cutoff=cutoff,
        atol=atol, rtol=rtol, cum_percentage=cum_percentage,
        collect_metrics=collect_metrics, verbose=verbose)


__all__ = ['BLOSTR', 'BLOSTRSVD', 'tr_blostr', 'tr_blostr_svd']
