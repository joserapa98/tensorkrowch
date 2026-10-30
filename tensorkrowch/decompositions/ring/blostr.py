"""
This script contains:

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
        * _fit_blostr
"""

from dataclasses import dataclass
from math import isfinite, prod
from typing import Optional, Sequence, Tuple, Union

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
from tensorkrowch.decompositions.results import TRDecomposition
from tensorkrowch.decompositions.ring.blocks import _split_block


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
                   generator: Optional[torch.Generator]
                   ) -> Tuple[torch.Tensor, ...]:
    """Recovers uniform-rank cores using product Gaussian projections."""
    rank = first.shape[0]
    device = tensor.device if generator is None else generator.device

    # Generic product projections avoid singular coordinate slices of otherwise
    # injective physical cores.
    vectors = [torch.randn(dimension, dtype=tensor.dtype, device=device,
                           generator=generator).to(tensor.device)
               for dimension in tensor.shape]
    vectors = [vector / vector.norm() for vector in vectors]
    cores = [first]
    for site in range(1, tensor.ndim):
        other_sites = [index for index in range(tensor.ndim)
                       if index not in (site, site - 1)]
        matrix = tensor.permute(site, site - 1, *other_sites)
        for index in reversed(other_sites):
            matrix = torch.tensordot(matrix, vectors[index], dims=([-1], [0]))
        product = torch.eye(rank, dtype=tensor.dtype, device=tensor.device)
        for index in range(site - 1):
            projected = torch.einsum('aib,i->ab', cores[index], vectors[index])
            product = product @ projected
        previous = torch.einsum('ab,bic->iac', product, cores[-1])
        previous = previous.reshape(-1, rank * rank).mT
        current = (matrix @ torch.linalg.pinv(previous)).reshape(
            tensor.shape[site], rank, rank).permute(2, 0, 1)
        cores.append(current)
    return tuple(cores)


def _fit_blostr(tensor: torch.Tensor,
                rank: _Rank,
                *,
                use_svd: bool,
                slices: Optional[Sequence[Sequence[int]]] = None,
                projection: str = 'slices',
                spectral_atol: float = 1e-10,
                n_attempts: int = 8,
                n_iters: int = 50,
                n_restarts: int = 10,
                generator: Optional[torch.Generator] = None,
                truncation: Optional[_TruncationSpec] = None,
                out_device: _Device = 'cpu',
                collect_metrics: bool = False,
                verbose: Union[bool, int] = 0) -> TRDecomposition:
    """Runs spectral attempts and retains the smallest reconstruction error."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('`tensor` should be torch.Tensor type')
    if tensor.ndim < 3:
        raise ValueError('`tensor` should contain at least three axes')
    if not tensor.is_floating_point() and not tensor.is_complex():
        raise TypeError('`tensor` should have floating or complex dtype')
    if not torch.isfinite(tensor).all():
        raise ValueError('`tensor` should contain finite values')
    ranks = None if rank is None else _normalize_rank(rank, tensor.ndim)
    if ranks is not None:
        if (not use_svd) and (len(set(ranks)) != 1):
            raise ValueError('Full BLOSTR requires uniform `rank`')
        if use_svd and (ranks[0] != ranks[-2]):
            raise ValueError('BLOSTR-SVD requires `rank[0] == rank[-2]`')
    if projection not in ('slices', 'gaussian'):
        raise ValueError('`projection` should be "slices" or "gaussian"')
    if (slices is not None) and (projection != 'slices'):
        raise ValueError('Explicit `slices` require `projection="slices"`')
    if slices is not None:
        slices = _validate_slices(slices, tensor.shape)
    if isinstance(spectral_atol, bool) or not isinstance(
            spectral_atol, (int, float)):
        raise TypeError('`spectral_atol` should be a non-negative number')
    if (spectral_atol < 0) or (not isfinite(spectral_atol)):
        raise ValueError('`spectral_atol` should be finite and non-negative')
    for name, value in (('n_attempts', n_attempts), ('n_iters', n_iters),
                        ('n_restarts', n_restarts)):
        _normalize_positive_int(value, name)
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError('`generator` should be torch.Generator type or None')
    if not isinstance(collect_metrics, bool):
        raise TypeError('`collect_metrics` should be bool type')
    verbosity = _normalize_verbosity(verbose)
    observer = _resolve_observer(verbosity, None) if verbosity else None
    if observer is not None:
        observer.emit(DecompositionEvent(
            name='start', phase='BLOSTR',
            values={'rank': rank, 'attempts': n_attempts}))
    collect_metrics = collect_metrics or bool(verbosity)
    runtime = _RuntimePolicy.from_tensor(tensor, out_device=out_device)
    dtype = torch.complex128 if tensor.dtype in (
        torch.float64, torch.complex128) else torch.complex64

    # Spectral factors may be complex even when the original tensor is real.
    active = tensor.to(dtype=dtype)
    metrics = DecompositionMetrics()
    failures = []
    best = None
    attempts = n_attempts if slices is None else 1
    with runtime.timer() as timer:

        # Reject degenerate projections and keep the best full reconstruction.
        for attempt in range(attempts):
            local_metrics = DecompositionMetrics()
            try:
                first = _first_core(active, ranks, slices, projection,
                                    spectral_atol, n_iters, n_restarts,
                                    generator)
                cyclic, _, adjacent = first.core.shape
                if use_svd:

                    # Remove the first factor and place its virtual axes at the
                    # TT ends.
                    first_matrix = first.core.permute(1, 0, 2).reshape(
                        active.shape[0], -1)
                    tail = torch.linalg.pinv(first_matrix) @ active.reshape(
                        active.shape[0], -1)
                    tail = tail.reshape(cyclic, adjacent, *active.shape[1:])
                    tail = tail.movedim(0, -1)
                    options = {} if truncation is None else \
                        truncation.as_kwargs()
                    if (ranks is None) and all(
                            options.get(name) is None for name in
                            ('cutoff', 'atol', 'rtol', 'cum_percentage')):
                        precision = (10 * max(active.shape) *
                                     torch.finfo(active.real.dtype).eps *
                                     torch.linalg.cond(first_matrix))
                        options['rtol'] = min(
                            max(spectral_atol, float(precision))**2, 1)
                    tail_cores = _split_block(
                        tail, active.shape[1:],
                        rank=None if ranks is None else ranks[1:-1],
                        cutoff=options.get('cutoff'),
                        atol=options.get('atol'), rtol=options.get('rtol'),
                        cum_percentage=options.get('cum_percentage'),
                        metrics=local_metrics if collect_metrics else None)
                    cores = (first.core, *tail_cores)
                else:
                    if min(active.shape) < cyclic * cyclic:
                        raise ValueError('Full BLOSTR requires every physical '
                                         'dimension to be at least rank**2')
                    if cyclic != adjacent:
                        raise ValueError('Full BLOSTR requires uniform '
                                         'discovered ranks')
                    cores = _recover_cores(active, first.core, generator)
                candidate = TRDecomposition(cores)
                approximation = candidate.contract_dense()
                absolute = torch.linalg.vector_norm(approximation - active)
                denominator = torch.linalg.vector_norm(active)
                relative = safe_divide(absolute, denominator,
                                       zero_value=absolute)
                if (best is None) or (relative < best[0]):
                    best = (relative, candidate, first,
                            local_metrics.truncations, absolute)
                if observer is not None:
                    observer.emit(DecompositionEvent(
                        name='attempt_complete', phase='BLOSTR', values={
                            'attempt': attempt + 1, 'attempts': attempts,
                            'relative_error': relative.detach()}))
            except (RuntimeError, ValueError) as error:
                failures.append(str(error))
    if best is None:
        detail = '; '.join(dict.fromkeys(failures))
        raise RuntimeError(f'Every BLOSTR spectral attempt failed. {detail}')

    # Report the accepted attempt, not the final projection that was tried.
    relative, candidate, first, records, absolute = best
    if collect_metrics:
        metrics.errors.append(ErrorRecord(
            kind='reconstruction', absolute=absolute, relative=relative,
            denominator=torch.linalg.vector_norm(active)))
        metrics.truncations.extend(records)
        metrics.timings.append(TimingRecord(name='fit', elapsed=timer.elapsed))
        metrics.warnings.extend(failures)
    result = TRDecomposition(
        [runtime.finalize(core) for core in candidate.cores], metrics=metrics,
        metadata={
            'algorithm': 'blostr_svd' if use_svd else 'blostr',
            'requested_rank': None if ranks is None else list(ranks),
            'adaptive': ranks is None, 'projection': projection,
            'slices': first.slices, 'attempts': attempts,
            'cluster_error': first.cluster_error.detach(),
        })
    if observer is not None:
        observer.emit(DecompositionEvent(
            name='summary', phase='BLOSTR', values={
                'rank': result.rank, 'absolute_error': absolute,
                'relative_error': relative, 'elapsed': timer.elapsed}))
        for site, core in enumerate(result.cores):
            observer.emit(DecompositionEvent(
                name='core', phase='BLOSTR', level=3, site=site,
                values={'shape': tuple(core.shape), 'tensor': core}))
        observer.close(metrics)
    return result


def tr_blostr(tensor: torch.Tensor,
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
              verbose: Union[bool, int] = 0) -> TRDecomposition:
    r"""
    Recovers a dense tensor ring using spectral BLOSTR at every site.

    Two pairs of interior slices identify the first core through blockwise
    simultaneous diagonalization. Product Gaussian projections of all but
    two physical axes then recover each remaining core by a linear solve.
    These avoid singular coordinate slices of otherwise injective cores.
    This is the uniform-rank construction of
    Algorithm 1 in `A Provably Efficient Method for Tensor Ring Decomposition
    and Its Applications <https://arxiv.org/abs/2512.01016>`_. It targets exact
    tensors with injective physical unfoldings at every site and nondegenerate
    projection ratios. Each physical dimension must therefore be at least the
    squared TR rank. This differs from :func:`tr_blostr_svd`, which needs
    injectivity only at its two external physical axes. No accuracy guarantee
    is made for approximate inputs or rank-deficient recovery environments.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor with one physical axis per future TR core.
    rank : int or sequence[int], optional
        Prescribed right-link ranks, including the cyclic last entry. Full
        BLOSTR requires equal ranks. With ``None``, balanced k-means discovers
        the multiplicities of the nonzero spectrum and requires them to imply
        equal cyclic and adjacent ranks.
    slices : sequence of four index sequences, optional
        Configurations fixing every interior axis for the two spectral ratios.
        With ``None``, configurations are drawn using ``generator``.
    projection : {"slices", "gaussian"}
        Coordinate slices or contractions against four independent dense
        Gaussian tensors over all interior axes. The latter can span matrix
        directions absent from any single coordinate slice. Explicit
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
    TRDecomposition
        Recovered TR cores and construction metadata. Spectral factors use a
        complex dtype even for real input tensors.

    Examples
    --------
    >>> cores = [torch.randn(1, 3, 1, dtype=torch.float64) for _ in range(3)]
    >>> tensor = tk.formats.TR(cores).contract_dense()
    >>> result = tk.decompositions.tr_blostr(tensor, rank=1)
    >>> torch.allclose(result.contract_dense(), tensor.to(torch.complex128))
    True
    """
    return _fit_blostr(
        tensor, rank, use_svd=False, slices=slices, projection=projection,
        spectral_atol=spectral_atol, n_attempts=n_attempts, n_iters=n_iters,
        n_restarts=n_restarts, generator=generator, out_device=out_device,
        collect_metrics=collect_metrics, verbose=verbose)


def tr_blostr_svd(tensor: torch.Tensor,
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
                  verbose: Union[bool, int] = 0) -> TRDecomposition:
    r"""
    Recovers the first TR core spectrally and splits the rest with TT-SVD.

    Unlike :func:`tr_blostr`, interior ranks may differ. The two ranks bounding
    the projected interior chain must coincide: ``rank[0] == rank[-2]``.
    The cyclic rank may differ. Gaussian joint projections can retain full
    matrix rank even when an intermediate bond makes each coordinate slice
    deficient. The physical first and last axes must both have dimension at
    least ``rank[-1] * rank[0]``. These conditions permit spectral recovery;
    they do not guarantee spectral separation for a particular projection.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor with one physical axis per future TR core.
    rank : int or sequence[int], optional
        Prescribed right-link ranks, including the cyclic last entry. With
        ``None``, balanced k-means discovers the first cyclic and adjacent
        ranks; TT-SVD discovers the remaining bonds numerically.
    slices : sequence of four index sequences, optional
        Configurations fixing every interior axis for the two spectral ratios.
        With ``None``, configurations are drawn using ``generator``.
    projection : {"slices", "gaussian"}
        Coordinate slices or contractions against four independent dense
        Gaussian tensors over all interior axes. The latter can span matrix
        directions absent from any single coordinate slice. Explicit
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
        singular-value fraction, in tail TT-SVD. With no rank or tolerance,
        a numerical threshold accounts for the precision and conditioning of
        the recovered first core. A supplied rank is an upper bound; smaller
        retained ranks are returned without zero padding.
    out_device : str or torch.device, optional
        Final core storage device. ``None`` keeps the input device.
    collect_metrics : bool
        Whether to retain reconstruction, truncation and timing diagnostics.
    verbose : bool or int
        Console verbosity from 0 (silent) to 3.

    Returns
    -------
    TRDecomposition
        Recovered TR cores and construction metadata. Spectral factors use a
        complex dtype even for real input tensors.

    Examples
    --------
    >>> cores = [torch.randn(1, 3, 1, dtype=torch.float64) for _ in range(3)]
    >>> tensor = tk.formats.TR(cores).contract_dense()
    >>> result = tk.decompositions.tr_blostr_svd(tensor, rank=1)
    >>> torch.allclose(result.contract_dense(), tensor.to(torch.complex128))
    True
    """
    truncation = _TruncationSpec(cutoff=cutoff, atol=atol, rtol=rtol,
                                 cum_percentage=cum_percentage)
    return _fit_blostr(
        tensor, rank, use_svd=True, slices=slices, projection=projection,
        spectral_atol=spectral_atol, n_attempts=n_attempts, n_iters=n_iters,
        n_restarts=n_restarts, generator=generator, truncation=truncation,
        out_device=out_device, collect_metrics=collect_metrics,
        verbose=verbose)


__all__ = ['tr_blostr', 'tr_blostr_svd']
