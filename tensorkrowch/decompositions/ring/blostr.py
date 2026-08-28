"""Experimental blockwise spectral tensor-ring decomposition."""

import time
import warnings
from dataclasses import dataclass
from math import isfinite, prod
from typing import (Any, Mapping, Optional, Sequence, Tuple, Union)

import torch

from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord,
                                                 TimingRecord,
                                                 TruncationRecord)
from tensorkrowch.decompositions.results import TRDecomposition
from tensorkrowch.decompositions.ring.gauges import ExperimentalWarning
from tensorkrowch.decompositions.ring.opening import (LoopOpenerCapabilities,
                                                      LoopOpening)
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 as_tensor_source)
from tensorkrowch.utils import truncated_svd


_Rank = Union[int, Sequence[int]]
_Device = Optional[Union[str, torch.device]]


@dataclass(frozen=True)
class _FirstCoreFactorization:
    """Stores the recovered first core and spectral diagnostics."""

    core: torch.Tensor
    slices: Tuple[Tuple[int, ...], ...]
    eigenvalues: Tuple[Tuple[complex, ...], Tuple[complex, ...]]
    eigenspace_residual: Tuple[float, float]


def _normalize_rank(rank: _Rank, n_sites: int) -> Tuple[int, ...]:
    """Normalizes the common spectral rank required by current BLOSTR."""
    if isinstance(rank, bool):
        raise TypeError('`rank` should be int or a sequence of ints')
    if isinstance(rank, int):
        ranks = (rank,) * n_sites
    else:
        if isinstance(rank, (str, bytes)):
            raise TypeError('`rank` should be int or a sequence of ints')
        try:
            ranks = tuple(rank)
        except TypeError as exc:
            raise TypeError(
                '`rank` should be int or a sequence of ints') from exc
        if len(ranks) != n_sites:
            raise ValueError(
                'A BLOSTR `rank` sequence should contain one right-link rank '
                'per site')
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in ranks):
        raise TypeError('BLOSTR ranks should be integers')
    if any(value < 1 for value in ranks):
        raise ValueError('BLOSTR ranks should be positive')
    if len(set(ranks)) != 1:
        raise ValueError(
            'The current BLOSTR implementation requires equal TR ranks')
    return ranks


def _normalize_positive_int(value: int, name: str) -> int:
    """Validates one positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'`{name}` should be int type')
    if value < 1:
        raise ValueError(f'`{name}` should be positive')
    return value


def _normalize_non_negative_float(value: float, name: str) -> float:
    """Validates one finite non-negative floating-point option."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'`{name}` should be a real scalar')
    value = float(value)
    if value < 0 or not isfinite(value):
        raise ValueError(f'`{name}` should be finite and non-negative')
    return value


def _random_permutation(
        size: int,
        generator: Optional[torch.Generator]) -> torch.Tensor:
    """Draws reproducibly and returns CPU indices for Python-side grouping."""
    device = 'cpu' if generator is None else generator.device
    return torch.randperm(
        size, generator=generator, device=device).detach().cpu()


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Returns the matching eigendecomposition dtype."""
    if dtype in (torch.float16, torch.bfloat16, torch.float32,
                 torch.complex64):
        return torch.complex64
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    raise TypeError('BLOSTR requires a floating or complex tensor')


def _flat_to_configuration(flat: int,
                           input_dim: Sequence[int]) -> Tuple[int, ...]:
    """Unravels one flat interior-slice id."""
    configuration = [0] * len(input_dim)
    for axis in range(len(input_dim) - 1, -1, -1):
        configuration[axis] = flat % input_dim[axis]
        flat //= input_dim[axis]
    return tuple(configuration)


def _draw_slices(input_dim: Sequence[int],
                 generator: Optional[torch.Generator]
                 ) -> Tuple[Tuple[int, ...], ...]:
    """Draws two distinct slice ratios from interior configurations."""
    interior = tuple(input_dim[1:-1])
    n_configurations = prod(interior)
    if n_configurations < 3:
        raise ValueError(
            'BLOSTR requires at least three interior slice configurations')
    draw = min(4, n_configurations)
    permutation = _random_permutation(
        n_configurations, generator)[:draw].tolist()
    configurations = tuple(
        _flat_to_configuration(index, interior) for index in permutation)
    if draw == 3:
        return (configurations[0], configurations[1],
                configurations[0], configurations[2])
    return configurations


def _validate_slices(
        slices: Sequence[Sequence[int]],
        input_dim: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    """Validates four user-provided interior slice configurations."""
    if isinstance(slices, (str, bytes)):
        raise TypeError('`slices` should contain four index sequences')
    try:
        slices = tuple(tuple(configuration) for configuration in slices)
    except TypeError as exc:
        raise TypeError(
            '`slices` should contain four index sequences') from exc
    if len(slices) != 4:
        raise ValueError('`slices` should contain alpha, beta and their primes')
    interior = tuple(input_dim[1:-1])
    for configuration in slices:
        if len(configuration) != len(interior):
            raise ValueError(
                'Every BLOSTR slice should index all interior dimensions')
        for index, dimension in zip(configuration, interior):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError('BLOSTR slice indices should be integers')
            if index < 0 or index >= dimension:
                raise ValueError('A BLOSTR slice index lies outside the tensor')
    if slices[0] == slices[1] or slices[2] == slices[3]:
        raise ValueError('Each BLOSTR slice ratio requires distinct indices')
    return slices


def _slice_matrix(tensor: torch.Tensor,
                  configuration: Sequence[int]) -> torch.Tensor:
    """Leaves the first and last tensor axes open in one spectral slice."""
    index = (slice(None), *configuration, slice(None))
    return tensor[index]


def _balanced_order(eigenvalues: torch.Tensor,
                    cluster_size: int,
                    n_clusters: int,
                    n_iters: int,
                    n_restarts: int,
                    generator: Optional[torch.Generator]) -> torch.Tensor:
    """Groups complex eigenvalues into equal-size clusters reproducibly."""
    features = torch.stack(
        (eigenvalues.real, eigenvalues.imag), dim=1).detach().cpu()
    n_values = features.shape[0]
    if n_values != cluster_size * n_clusters:
        raise ValueError('The selected spectrum does not match BLOSTR ranks')

    best_labels = None
    best_cost = None
    for _ in range(n_restarts):
        permutation = _random_permutation(n_values, generator)
        labels = torch.empty(n_values, dtype=torch.long)
        for cluster in range(n_clusters):
            labels[permutation[
                cluster * cluster_size:(cluster + 1) * cluster_size]] = cluster

        for _ in range(n_iters):
            centers = torch.stack([
                features[labels == cluster].mean(0)
                for cluster in range(n_clusters)
            ])
            distances = torch.cdist(features, centers)
            new_labels = torch.full_like(labels, -1)
            capacity = torch.zeros(n_clusters, dtype=torch.long)
            order = torch.argsort(distances.min(dim=1).values)
            for value in order.tolist():
                for cluster in torch.argsort(distances[value]).tolist():
                    if capacity[cluster] < cluster_size:
                        new_labels[value] = cluster
                        capacity[cluster] += 1
                        break
            if torch.equal(new_labels, labels):
                break
            labels = new_labels

        centers = torch.stack([
            features[labels == cluster].mean(0)
            for cluster in range(n_clusters)
        ])
        cost = ((features - centers[labels]).square()).sum()
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_labels = labels.clone()

    grouped = torch.cat([
        torch.where(best_labels == cluster)[0]
        for cluster in range(n_clusters)
    ])
    return grouped.to(eigenvalues.device)


def _selected_eigenspace(matrix: torch.Tensor,
                         n_vectors: int,
                         spectral_atol: float,
                         cluster_size: int,
                         n_clusters: int,
                         n_iters: int,
                         n_restarts: int,
                         generator: Optional[torch.Generator]
                         ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Selects, groups and interleaves one BLOSTR eigenspace."""
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    order = torch.argsort(eigenvalues.abs(), descending=True)
    order = order[eigenvalues[order].abs() > spectral_atol]
    if order.numel() < n_vectors:
        raise RuntimeError(
            'BLOSTR found too few nonzero eigenvalues for the requested ranks')
    order = order[:n_vectors]
    selected_values = eigenvalues.index_select(0, order)
    selected_vectors = eigenvectors.index_select(1, order)
    grouping = _balanced_order(
        selected_values,
        cluster_size=cluster_size,
        n_clusters=n_clusters,
        n_iters=n_iters,
        n_restarts=n_restarts,
        generator=generator)
    selected_values = selected_values.index_select(0, grouping)
    selected_vectors = selected_vectors.index_select(1, grouping)

    interleave = torch.cat([
        cluster_size * torch.arange(
            n_clusters, device=matrix.device) + offset
        for offset in range(cluster_size)
    ])
    selected_values = selected_values.index_select(0, interleave)
    selected_vectors = selected_vectors.index_select(1, interleave)
    reconstruction = (
        selected_vectors @ torch.diag(selected_values) @
        torch.linalg.pinv(selected_vectors))
    denominator = torch.linalg.vector_norm(matrix)
    residual = torch.linalg.vector_norm(reconstruction - matrix)
    residual = residual if denominator == 0 else residual / denominator
    return selected_vectors, selected_values, float(
        residual.detach().cpu().item())


def _first_core(tensor: torch.Tensor,
                ranks: Sequence[int],
                slices: Sequence[Sequence[int]],
                spectral_atol: float,
                n_iters: int,
                n_restarts: int,
                generator: Optional[torch.Generator]
                ) -> _FirstCoreFactorization:
    """Recovers the first TR core by blockwise spectral alignment."""
    cyclic_rank = ranks[-1]
    right_rank = ranks[0]
    spectral_rank = cyclic_rank * right_rank
    if tensor.shape[0] < spectral_rank:
        raise ValueError(
            'The first input dimension should be at least '
            '`rank[-1] * rank[0]` for BLOSTR')
    if tensor.shape[-1] < spectral_rank:
        raise ValueError(
            'The last input dimension should be at least '
            '`rank[-1] * rank[0]` for BLOSTR')

    alpha, beta, alpha_prime, beta_prime = slices
    x = _slice_matrix(tensor, alpha)
    y = _slice_matrix(tensor, beta)
    x_prime = _slice_matrix(tensor, alpha_prime)
    y_prime = _slice_matrix(tensor, beta_prime)
    matrix = x @ torch.linalg.pinv(y)
    matrix_prime = x_prime @ torch.linalg.pinv(y_prime)

    eigenspace, eigenvalues, residual = _selected_eigenspace(
        matrix,
        spectral_rank,
        spectral_atol,
        cluster_size=cyclic_rank,
        n_clusters=right_rank,
        n_iters=n_iters,
        n_restarts=n_restarts,
        generator=generator)
    eigenspace_prime, eigenvalues_prime, residual_prime = \
        _selected_eigenspace(
            matrix_prime,
            spectral_rank,
            spectral_atol,
            cluster_size=cyclic_rank,
            n_clusters=right_rank,
            n_iters=n_iters,
            n_restarts=n_restarts,
            generator=generator)

    alignment = torch.linalg.pinv(eigenspace) @ eigenspace_prime
    first_block = alignment[0::right_rank, 0::right_rank]
    transforms = [torch.eye(
        cyclic_rank, dtype=tensor.dtype, device=tensor.device)]
    try:
        for block in range(1, right_rank):
            current = alignment[block::right_rank, 0::right_rank]
            transform = torch.linalg.solve(
                current.mT, first_block.mT).mT
            transforms.append(transform)
    except RuntimeError as exc:
        raise RuntimeError(
            'BLOSTR could not align spectral blocks; the selected slices are '
            'singular or spectrally degenerate') from exc

    block_transform = torch.zeros_like(alignment)
    try:
        for block, transform in enumerate(transforms):
            identity = torch.eye(
                cyclic_rank, dtype=tensor.dtype, device=tensor.device)
            block_transform[
                block::right_rank, block::right_rank] = torch.linalg.solve(
                    transform, identity)
    except RuntimeError as exc:
        raise RuntimeError(
            'BLOSTR recovered a singular block gauge') from exc
    first_flat = eigenspace @ block_transform
    first_core = first_flat.reshape(
        tensor.shape[0], cyclic_rank, right_rank).permute(1, 0, 2)
    return _FirstCoreFactorization(
        core=first_core,
        slices=tuple(tuple(configuration) for configuration in slices),
        eigenvalues=(
            tuple(complex(value) for value in eigenvalues.detach().cpu()),
            tuple(complex(value)
                  for value in eigenvalues_prime.detach().cpu())),
        eigenspace_residual=(residual, residual_prime))


def _recover_tail(first: _FirstCoreFactorization,
                  tensor: torch.Tensor,
                  ranks: Sequence[int],
                  cutoff: Optional[float],
                  atol: Optional[float],
                  rtol: Optional[float],
                  cum_percentage: Optional[float]
                  ) -> Tuple[Tuple[torch.Tensor, ...],
                             Tuple[TruncationRecord, ...]]:
    """Recovers all remaining cores by removing Q1 and splitting the tail."""
    first_matrix = first.core.permute(1, 0, 2).reshape(tensor.shape[0], -1)
    tail = torch.linalg.pinv(first_matrix) @ tensor.reshape(tensor.shape[0], -1)
    tail = tail.reshape(ranks[-1], ranks[0], *tensor.shape[1:])
    tail = tail.movedim(0, -1)

    cores = [first.core]
    records = []
    left_rank = ranks[0]
    remaining_input = tuple(tensor.shape[1:])
    for offset, input_dimension in enumerate(remaining_input[:-1], start=1):
        matrix = tail.reshape(left_rank * input_dimension, -1)
        u, singular_values, vh, info = truncated_svd(
            matrix,
            rank=ranks[offset],
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            return_info=True)
        selected_rank = singular_values.shape[-1]
        cores.append(u.reshape(left_rank, input_dimension, selected_rank))
        tail = (singular_values.to(vh.dtype).unsqueeze(-1) * vh).reshape(
            selected_rank,
            *remaining_input[(offset):],
            ranks[-1])
        left_rank = selected_rank
        records.append(TruncationRecord.from_svd_info(info, site=offset))
    cores.append(tail.reshape(
        left_rank, remaining_input[-1], ranks[-1]))
    return tuple(cores), tuple(records)


def _fit_blostr(tensor: torch.Tensor,
                rank: _Rank,
                *,
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
                output_device: _Device = 'cpu') -> TRDecomposition:
    """Runs multiple spectral slice attempts and keeps the best recovery."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('`tensor` should be torch.Tensor type')
    if tensor.ndim < 3:
        raise ValueError('BLOSTR requires a tensor with at least three inputs')
    if not tensor.is_floating_point() and not tensor.is_complex():
        raise TypeError('BLOSTR requires a floating or complex tensor')
    ranks = _normalize_rank(rank, tensor.ndim)
    spectral_atol = _normalize_non_negative_float(
        spectral_atol, 'spectral_atol')
    n_attempts = _normalize_positive_int(n_attempts, 'n_attempts')
    n_iters = _normalize_positive_int(n_iters, 'n_iters')
    n_restarts = _normalize_positive_int(n_restarts, 'n_restarts')
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError('`generator` should be torch.Generator type or None')
    if output_device is not None:
        output_device = torch.device(output_device)

    active_tensor = tensor.to(dtype=_complex_dtype(tensor.dtype))
    fixed_slices = None if slices is None else _validate_slices(
        slices, tensor.shape)
    attempts = 1 if fixed_slices is not None else n_attempts
    failures = []
    best = None
    start = time.perf_counter()
    for _ in range(attempts):
        active_slices = fixed_slices
        if active_slices is None:
            active_slices = _draw_slices(tensor.shape, generator)
        try:
            first = _first_core(
                active_tensor,
                ranks,
                active_slices,
                spectral_atol,
                n_iters,
                n_restarts,
                generator)
            cores, records = _recover_tail(
                first,
                active_tensor,
                ranks,
                cutoff,
                atol,
                rtol,
                cum_percentage)
            candidate = TRDecomposition(cores)
            approximation = candidate.contract_dense()
            absolute = torch.linalg.vector_norm(approximation - active_tensor)
            denominator = torch.linalg.vector_norm(active_tensor)
            relative = absolute if denominator == 0 else absolute / denominator
            score = float(relative.detach().cpu().item())
            if best is None or score < best[0]:
                best = score, candidate, first, records, absolute, denominator
        except (RuntimeError, ValueError) as exc:
            failures.append(str(exc))

    if best is None:
        detail = '; '.join(dict.fromkeys(failures))
        raise RuntimeError(
            'Every BLOSTR spectral attempt failed. '
            f'{detail}')
    relative, candidate, first, records, absolute, denominator = best
    elapsed = time.perf_counter() - start
    metrics = DecompositionMetrics(
        truncations=list(records),
        errors=[ErrorRecord(
            kind='reconstruction',
            absolute=absolute,
            relative=relative,
            denominator=denominator)],
        timings=[TimingRecord(name='fit', elapsed=elapsed)])
    final_cores = [
        core if output_device is None else core.to(output_device)
        for core in candidate.cores
    ]
    return TRDecomposition(
        final_cores,
        metrics=metrics,
        metadata={
            'algorithm': 'blostr',
            'experimental': True,
            'requested_rank': list(ranks),
            'slices': first.slices,
            'eigenspace_residual': first.eigenspace_residual,
            'attempts': attempts,
            'failed_attempts': len(failures),
        })


def _materialize_target(target,
                        context: Mapping[str, Any]) -> torch.Tensor:
    """Materializes a scalar local source for the spectral algorithm."""
    source = as_tensor_source(
        target,
        input_dim=context.get('input_dim'),
        output_shape=(),
        dtype=context.get('dtype'),
        device=context.get('device', 'cpu'),
        batch_size=context.get('batch_size'))
    axes = [
        torch.arange(dimension, device=source.device)
        for dimension in source.input_dim
    ]
    indices = torch.cartesian_prod(*axes)
    if len(source.input_dim) == 1:
        indices = indices.unsqueeze(1)
    values = source.evaluate(ConfigurationBatch(indices, kind='indices'))
    if values.shape != (indices.shape[0],):
        raise ValueError('BLOSTR requires a scalar tensor source')
    return values.reshape(source.input_dim)


def _mirror_cores(cores: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """Restores cores from a reversed BLOSTR local problem."""
    return tuple(core.permute(2, 1, 0) for core in reversed(cores))


def _mirror_rank(rank: Sequence[int]) -> Tuple[int, ...]:
    """Maps right-link ranks through local reversal."""
    return (*reversed(rank[:-1]), rank[-1])


class BLOSTRLoopOpener:
    """Uses unrestricted BLOSTR as a local spectral loop initializer.

    ``fit_options`` accepts the keyword options of :func:`tr_blostr` except
    target, rank, output device and ``return_info``. BLOSTR does not support
    fixed gauges; combine it with :class:`CompositeLoopOpener` and an ALS
    refiner when constraints are required.
    """

    _capabilities = LoopOpenerCapabilities(supports_blocks=True)

    def __init__(self,
                 fit_options: Optional[Mapping[str, Any]] = None) -> None:
        if fit_options is None:
            fit_options = {}
        if not isinstance(fit_options, Mapping):
            raise TypeError('`fit_options` should be a mapping or None')
        reserved = {'rank', 'output_device', 'return_info'}
        overlap = reserved.intersection(fit_options)
        if overlap:
            raise ValueError(
                f'`fit_options` should not override {sorted(overlap)}')
        self.fit_options = dict(fit_options)
        warnings.warn(
            'BLOSTRLoopOpener is experimental and its numerical behavior may '
            'change.',
            ExperimentalWarning,
            stacklevel=2)

    @property
    def capabilities(self) -> LoopOpenerCapabilities:
        """Declares unrestricted block support and no fixed-gauge support."""
        return self._capabilities

    def open(self,
             target,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: Optional[Mapping[str, Any]] = None) -> LoopOpening:
        """Recovers a free local ring and exposes its environment gauges."""
        if orientation not in ('right', 'left'):
            raise ValueError("`orientation` should be 'right' or 'left'")
        if context is None:
            context = {}
        elif not isinstance(context, Mapping):
            raise TypeError('`context` should be a mapping or None')
        input_dim = context.get('input_dim')
        if input_dim is None:
            if not isinstance(target, torch.Tensor):
                raise ValueError(
                    '`context["input_dim"]` is required for a lazy target')
            input_dim = target.shape
        input_dim = tuple(input_dim)
        ranks = _normalize_rank(rank, len(input_dim))
        self.capabilities.require(
            fixed_left=fixed_left is not None,
            fixed_right=fixed_right is not None,
            block_size=len(input_dim) - 2)

        dense = _materialize_target(target, context)
        active_ranks = ranks
        fit_options = dict(self.fit_options)
        if orientation == 'left':
            dense = dense.permute(*reversed(range(dense.ndim)))
            active_ranks = _mirror_rank(ranks)
            if fit_options.get('slices') is not None:
                fit_options['slices'] = tuple(
                    tuple(reversed(configuration))
                    for configuration in fit_options['slices'])
        result = _fit_blostr(
            dense,
            active_ranks,
            output_device=None,
            **fit_options)
        cores = tuple(result.cores)
        if orientation == 'left':
            cores = _mirror_cores(cores)
        actual_rank = tuple(core.shape[-1] for core in cores)
        if actual_rank != ranks:
            raise ValueError(
                'BLOSTR initializer ranks do not match the requested local '
                f'ranks: {actual_rank} != {ranks}')
        return LoopOpening(
            left_gauge=cores[0],
            cores=cores[1:-1],
            right_gauge=cores[-1],
            rank=actual_rank,
            orientation=orientation,
            diagnostics={
                'algorithm': 'blostr',
                'metadata': dict(result.metadata),
                'error': {
                    'absolute': result.metrics.errors[0].absolute,
                    'relative': result.metrics.errors[0].relative,
                },
            })


def tr_blostr(tensor: torch.Tensor,
              rank: _Rank,
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
              output_device: _Device = 'cpu',
              return_info: bool = False):
    r"""Decomposes a dense tensor into a TR with experimental BLOSTR.

    This implements the blockwise simultaneous-diagonalization construction
    from Algorithm 1 of *A Provably Efficient Method for Tensor Ring
    Decomposition and Its Applications*, Han Chen, Sitan Chen and Anru R.
    Zhang (2025), available in this `paper
    <https://arxiv.org/abs/2512.01016>`_. TensorKrowch recovers the remaining
    cores with its standard truncated-SVD semantics and can try several
    reproducible spectral slices.

    The current implementation requires a common TR rank, passed either as a
    scalar or as an equal-valued sequence with one right-link rank per core.
    This is the uniform-rank setting covered by the spectral construction; a
    non-uniform sequence is rejected instead of returning an inaccurate
    decomposition. The first and last input dimensions must be at least
    ``rank ** 2``. BLOSTR is sensitive to slice degeneracy; failed attempts
    produce explicit diagnostics rather than silent zero padding. Spectral
    factors are complex-valued even when ``tensor`` is real, since valid
    complex gauges may be required to represent the same real tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor with one input dimension per future TR core.
    rank : int or sequence of int
        Common TR rank, or an equal-valued sequence containing one right-link
        rank per core. The last entry denotes the cyclic link as usual.
    slices : sequence of four index sequences, optional
        Explicit ``alpha``, ``beta``, ``alpha_prime`` and ``beta_prime``
        interior slices. If omitted, slices are drawn with ``generator``.
    spectral_atol : float
        Absolute threshold for retaining eigenvalues.
    n_attempts : int
        Number of slice selections tried when ``slices`` is omitted.
    n_iters, n_restarts : int
        Balanced clustering iterations and random restarts.
    generator : torch.Generator, optional
        Controls slice selection and balanced-clustering initialization.
    cutoff, atol, rtol, cum_percentage : float, optional
        Standard :func:`~tensorkrowch.truncated_svd` criteria used while
        recovering all cores after the first spectral core.
    output_device : str or torch.device, optional
        Final core storage device. ``None`` keeps the input device.
    return_info : bool
        If ``True``, returns ``(cores, info)``.

    Returns
    -------
    list[torch.Tensor] or tuple
        TR cores, optionally followed by structured result information.

    Examples
    --------
    >>> generator = torch.Generator().manual_seed(0)
    >>> cores = [torch.randn(2, 4, 2, generator=generator)
    ...          for _ in range(3)]
    >>> tensor = TRDecomposition(cores).contract_dense()
    >>> recovered = tr_blostr(
    ...     tensor, rank=2,
    ...     generator=torch.Generator().manual_seed(1))
    >>> [tuple(core.shape) for core in recovered]
    [(2, 4, 2), (2, 4, 2), (2, 4, 2)]
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    warnings.warn(
        '`tr_blostr` is experimental and its numerical behavior may change.',
        ExperimentalWarning,
        stacklevel=2)
    result = _fit_blostr(
        tensor,
        rank,
        slices=slices,
        spectral_atol=spectral_atol,
        n_attempts=n_attempts,
        n_iters=n_iters,
        n_restarts=n_restarts,
        generator=generator,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
        cum_percentage=cum_percentage,
        output_device=output_device)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


__all__ = ['BLOSTRLoopOpener', 'tr_blostr']
