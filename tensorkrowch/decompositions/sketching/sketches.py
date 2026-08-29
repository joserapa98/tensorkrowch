"""Sketch operators and core-determining systems for TT-RS."""

from dataclasses import dataclass, field
from math import prod, sqrt
from typing import (Any, Mapping, Optional, Protocol, Sequence, Tuple,
                    Union, runtime_checkable)

import torch

from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 TruncationRecord)
from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.sketching.sources import (
    SupportTensorSource,
    _iter_support,
)
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 TensorSource)
from tensorkrowch.utils import truncated_svd


_Rank = Union[int, Sequence[int]]


def _normalize_rank(rank: _Rank, n_sites: int) -> Tuple[int, ...]:
    """Normalizes one shared rank cap or one cap per open TT link."""
    if isinstance(rank, bool):
        raise TypeError('`rank` should be int or a sequence of ints')
    if isinstance(rank, int):
        ranks = (rank,) * (n_sites - 1)
    else:
        if isinstance(rank, (str, bytes)):
            raise TypeError('`rank` should be int or a sequence of ints')
        try:
            ranks = tuple(rank)
        except TypeError as exc:
            raise TypeError(
                '`rank` should be int or a sequence of ints') from exc
        if len(ranks) != n_sites - 1:
            raise ValueError(
                '`rank` should contain one value per open TT link')
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in ranks):
        raise TypeError('TT ranks should be integers')
    if any(value < 1 for value in ranks):
        raise ValueError('TT ranks should be positive')
    return ranks


def _ravel_subset(indices: torch.Tensor,
                  dimensions: Sequence[int]) -> torch.Tensor:
    """Ravels a non-empty consecutive subset of discrete indices."""
    strides = indices.new_tensor([
        prod(dimensions[site + 1:])
        for site in range(len(dimensions))
    ])
    return (indices * strides).sum(dim=1)


def _one_hot_ids(ids: torch.Tensor,
                 size: int,
                 dtype: torch.dtype) -> torch.Tensor:
    """Creates feature rows while mapping negative ids to zero."""
    result = torch.zeros(
        ids.shape[0], size, dtype=dtype, device=ids.device)
    valid = ids >= 0
    if torch.any(valid):
        result[valid, ids[valid]] = 1
    return result


def _selected_ids(values: torch.Tensor,
                  reference: torch.Tensor) -> torch.Tensor:
    """Maps sorted flat values to sorted reference positions or ``-1``."""
    positions = torch.searchsorted(reference, values)
    safe = positions.clamp(max=max(reference.numel() - 1, 0))
    result = torch.full_like(positions, -1)
    if reference.numel():
        matched = (positions < reference.numel()) & \
            (reference.index_select(0, safe) == values)
        result[matched] = positions[matched]
    return result


def _sample_states(values: torch.Tensor,
                   maximum: Optional[int],
                   generator: Optional[torch.Generator]) -> torch.Tensor:
    """Selects a deterministic sorted state set under an optional cap."""
    states = torch.unique(values, sorted=True)
    if maximum is None or states.numel() <= maximum:
        return states
    permutation = torch.randperm(
        states.numel(), generator=generator, device=states.device)[:maximum]
    return states.index_select(0, permutation).sort().values


@dataclass(frozen=True)
class _SketchWeights:
    """Evaluated left/right sketches and recursive left local blocks."""

    left: Sequence[torch.Tensor]
    right: Sequence[torch.Tensor]
    left_blocks: Sequence[torch.Tensor]
    value_scale: torch.Tensor
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        left = tuple(self.left)
        right = tuple(self.right)
        blocks = tuple(self.left_blocks)
        if len(left) != len(right) or len(blocks) != len(left):
            raise ValueError('Sketch weights should align on every TT cut')
        if not isinstance(self.value_scale, torch.Tensor) or \
                self.value_scale.ndim != 1:
            raise ValueError('`value_scale` should be a vector')
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)
        object.__setattr__(self, 'left_blocks', blocks)
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))


@runtime_checkable
class SketchOperator(Protocol):
    """Protocol for operators that construct a validated TT-RS system."""

    def builder(self, source: TensorSource) -> 'SketchSystemBuilder':
        """Returns the source-bound core-determining-system builder."""


@runtime_checkable
class SketchSystemBuilder(Protocol):
    """Protocol for constructing local TT-RS equations from one source."""

    def build(self,
              *,
              batch_size: Optional[int] = None,
              generator: Optional[torch.Generator] = None
              ) -> 'CoreDeterminingSystem':
        """Builds sketched local tensors and recursive left blocks."""


@dataclass(frozen=True)
class CoreDeterminingSystem:
    """Stores the TT-RS local tensors and solves their coupled equations.

    ``phi[k]`` is the source contracted with the recursive left sketch through
    site ``k - 1`` and the right sketch after site ``k``. ``left_blocks[k]``
    is the small tensor that recursively maps the left sketch at cut ``k - 1``
    to the sketch at cut ``k``. This pairing is the essential TT-RS contract;
    an arbitrary collection of unrelated random projections is not accepted.

    This is the discrete core-determining-equation construction from
    `Generative modeling via tensor train sketching
    <https://arxiv.org/abs/2202.11788>`_ by YoonHaeng Hur, Jeremy G. Hoskins,
    Michael Lindsey, E. M. Stoudenmire and Yuehaw Khoo (2022), Algorithms 1--4.
    TensorKrowch extends the source contraction to explicit sparse support and
    exposes the system independently from its final TT solve.
    """

    phi: Sequence[torch.Tensor]
    left_blocks: Sequence[torch.Tensor]
    input_dim: Sequence[int]
    operator: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phi = tuple(self.phi)
        blocks = tuple(self.left_blocks)
        input_dim = tuple(self.input_dim)
        if len(phi) != len(input_dim):
            raise ValueError('`phi` should contain one local tensor per site')
        if len(blocks) != len(input_dim) - 1:
            raise ValueError(
                '`left_blocks` should contain one recursive block per TT cut')
        previous = 1
        for site, tensor in enumerate(phi):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                raise ValueError('Every local Phi should have three axes')
            if tensor.shape[0] != previous or tensor.shape[1] != input_dim[site]:
                raise ValueError('Local Phi dimensions are not recursive')
            if site < len(blocks):
                block = blocks[site]
                if not isinstance(block, torch.Tensor) or block.ndim != 3:
                    raise ValueError(
                        'Every recursive left block should have three axes')
                if block.shape[1:] != (input_dim[site], previous):
                    raise ValueError(
                        'A recursive block does not match its previous sketch')
                previous = block.shape[0]
        if phi[-1].shape[-1] != 1:
            raise ValueError('The final local Phi should have unit right size')
        object.__setattr__(self, 'phi', phi)
        object.__setattr__(self, 'left_blocks', blocks)
        object.__setattr__(self, 'input_dim', input_dim)
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))

    def solve(self,
              rank: _Rank,
              *,
              cutoff: Optional[float] = None,
              atol: Optional[float] = None,
              rtol: Optional[float] = None,
              cum_percentage: Optional[float] = None,
              strict_system: bool = False,
              collect_metrics: bool = False) -> TTDecomposition:
        """Trims local ranges and solves the core-determining equations."""
        if not isinstance(strict_system, bool):
            raise TypeError('`strict_system` should be bool type')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        ranks = _normalize_rank(rank, len(self.input_dim))
        truncation = {
            'cutoff': cutoff,
            'atol': atol,
            'rtol': rtol,
            'cum_percentage': cum_percentage,
        }
        metrics = DecompositionMetrics()
        bases = []
        for site, tensor in enumerate(self.phi[:-1]):
            matrix = tensor.reshape(-1, tensor.shape[-1])
            options = {**truncation, 'rank': min(ranks[site], min(matrix.shape))}
            if collect_metrics:
                u, _, _, info = truncated_svd(
                    matrix, return_info=True, **options)
                metrics.truncations.append(TruncationRecord.from_svd_info(
                    info, site=site))
            else:
                u, _, _ = truncated_svd(matrix, **options)
            bases.append(u.reshape(
                tensor.shape[0], tensor.shape[1], u.shape[-1]))
        bases.append(self.phi[-1])

        coefficients = []
        coefficient_ranks = []
        for site, (block, basis) in enumerate(zip(
                self.left_blocks, bases[:-1])):
            coefficient = torch.einsum('anb,bnc->ac', block, basis)
            coefficients.append(coefficient)
            if strict_system or collect_metrics:
                numerical_rank = int(torch.linalg.matrix_rank(
                    coefficient).detach().cpu().item())
                coefficient_ranks.append(numerical_rank)
                if strict_system and numerical_rank < coefficient.shape[1]:
                    raise ValueError(
                        'Core-determining coefficient matrix is rank deficient '
                        f'at site {site}: rank={numerical_rank}/'
                        f'{coefficient.shape[1]}')

        cores = [bases[0]]
        solver = LeastSquaresSolver()
        for site in range(1, len(self.input_dim)):
            right = bases[site]
            solution, record = solver.solve(
                coefficients[site - 1],
                right.reshape(right.shape[0], -1),
                site=site,
                return_record=True)
            cores.append(solution.reshape(
                solution.shape[0], self.input_dim[site], right.shape[-1]))
            if collect_metrics:
                metrics.local_solves.append(record)

        compact = [cores[0].squeeze(0), *cores[1:-1], cores[-1].squeeze(-1)]
        return TTDecomposition(
            compact,
            metrics=metrics,
            metadata={
                'algorithm': 'tt_rs',
                'sketch_operator': self.operator,
                'coefficient_ranks': (
                    coefficient_ranks
                    if strict_system or collect_metrics else None),
                'system': dict(self.diagnostics),
            })


class _SupportSketchSystemBuilder:
    """Contracts finite support, enumerating only non-support sources."""

    def __init__(self,
                 source: TensorSource,
                 operator: SketchOperator) -> None:
        if not isinstance(source, TensorSource):
            raise TypeError('`source` should implement TensorSource')
        if tuple(source.output_shape) != ():
            raise ValueError('TT-RS currently requires a scalar source')
        if len(source.input_dim) < 2:
            raise ValueError('TT-RS requires at least two input sites')
        self.source = source
        self.operator = operator

    def build(self,
              *,
              batch_size: Optional[int] = None,
              generator: Optional[torch.Generator] = None
              ) -> CoreDeterminingSystem:
        """Builds all Phi tensors in support-linear memory when possible."""
        if isinstance(self.source, SupportTensorSource):
            batches = list(_iter_support(self.source, batch_size=batch_size))
            indices = torch.cat(
                [batch.as_tensor() for batch, _ in batches], dim=0)
            values = torch.cat([value for _, value in batches], dim=0)
            source_path = 'support'
        else:
            axes = [
                torch.arange(dimension, device=self.source.device)
                for dimension in self.source.input_dim
            ]
            indices = torch.cartesian_prod(*axes)
            if batch_size is None:
                batch_size = indices.shape[0]
            if isinstance(batch_size, bool) or not isinstance(batch_size, int):
                raise TypeError('`batch_size` should be int type or None')
            if batch_size < 1:
                raise ValueError('`batch_size` should be positive')
            values = []
            for start in range(0, indices.shape[0], batch_size):
                values.append(self.source.evaluate(ConfigurationBatch(
                    indices[start:start + batch_size], kind='indices')))
            values = torch.cat(values, dim=0)
            source_path = 'enumerated_grid'
        nonzero = values != 0
        indices = indices[nonzero]
        values = values[nonzero]
        if not values.numel():
            raise ValueError('TT-RS cannot decompose an empty sparse support')
        weights = self.operator._weights(
            indices,
            self.source.input_dim,
            values.dtype,
            generator)
        values = values * weights.value_scale.to(values.dtype)
        phis = []
        n_sites = len(self.source.input_dim)
        for site, dimension in enumerate(self.source.input_dim):
            left = values.new_ones(values.shape[0], 1) if site == 0 \
                else weights.left[site - 1].to(values.dtype)
            right = values.new_ones(values.shape[0], 1) \
                if site == n_sites - 1 \
                else weights.right[site].to(values.dtype)
            contributions = values[:, None, None] * \
                left[:, :, None] * right[:, None, :]
            phi = values.new_zeros(
                dimension, left.shape[1], right.shape[1])
            phi.index_add_(0, indices[:, site], contributions)
            phis.append(phi.permute(1, 0, 2))
        return CoreDeterminingSystem(
            phi=phis,
            left_blocks=weights.left_blocks,
            input_dim=self.source.input_dim,
            operator=type(self.operator).__name__,
            diagnostics={
                'support_size': values.shape[0],
                'source_path': source_path,
                'left_dimensions': tuple(item.shape[1] for item in weights.left),
                'right_dimensions': tuple(item.shape[1]
                                          for item in weights.right),
                **weights.diagnostics,
            })


class SampledSketch:
    """Recursive one-hot sketches induced by correlated configurations.

    For a sparse source the default configurations are its non-zero support.
    Prefix and suffix states are kept correlated; missing Cartesian-product
    entries are therefore never evaluated merely to recover their known zero.
    """

    def __init__(self,
                 samples: Optional[torch.Tensor] = None,
                 sketch_size: Optional[int] = None) -> None:
        if samples is not None and (
                not isinstance(samples, torch.Tensor) or samples.ndim != 2 or
                samples.dtype not in (torch.uint8, torch.int8, torch.int16,
                                      torch.int32, torch.int64)):
            raise TypeError('`samples` should be a two-dimensional integer tensor')
        if sketch_size is not None and (
                isinstance(sketch_size, bool) or
                not isinstance(sketch_size, int)):
            raise TypeError('`sketch_size` should be int type or None')
        if sketch_size is not None and sketch_size < 1:
            raise ValueError('`sketch_size` should be positive')
        self.samples = samples
        self.sketch_size = sketch_size

    def builder(self, source: TensorSource) -> SketchSystemBuilder:
        """Returns a finite-support sampled-system builder."""
        return _SupportSketchSystemBuilder(source, self)

    def _weights(self, indices, input_dim, dtype, generator):
        reference = indices if self.samples is None else self.samples.to(
            device=indices.device, dtype=torch.long)
        if reference.shape[1] != len(input_dim):
            raise ValueError('`samples` should contain one index per site')
        for site, dimension in enumerate(input_dim):
            if torch.any(reference[:, site] < 0) or \
                    torch.any(reference[:, site] >= dimension):
                raise ValueError(f'`samples` are out of bounds at site {site}')

        left = []
        right = []
        blocks = []
        prefix_states = []
        for site in range(len(input_dim) - 1):
            ref_ids = _ravel_subset(
                reference[:, :site + 1], input_dim[:site + 1])
            states = _sample_states(ref_ids, self.sketch_size, generator)
            prefix_states.append(states)
            ids = _ravel_subset(indices[:, :site + 1], input_dim[:site + 1])
            left.append(_one_hot_ids(
                _selected_ids(ids, states), states.numel(), dtype))

            suffix_dim = input_dim[site + 1:]
            ref_suffix = _ravel_subset(reference[:, site + 1:], suffix_dim)
            suffix_states = _sample_states(
                ref_suffix, self.sketch_size, generator)
            suffix_ids = _ravel_subset(indices[:, site + 1:], suffix_dim)
            right.append(_one_hot_ids(
                _selected_ids(suffix_ids, suffix_states),
                suffix_states.numel(),
                dtype))

            old_states = indices.new_tensor([0]) if site == 0 \
                else prefix_states[site - 1]
            block = torch.zeros(
                states.numel(), input_dim[site], old_states.numel(),
                dtype=dtype, device=indices.device)
            for new_position, state in enumerate(states):
                current = torch.remainder(state, input_dim[site])
                previous = torch.div(
                    state, input_dim[site], rounding_mode='floor')
                old_position = 0 if site == 0 else int(
                    _selected_ids(previous.reshape(1), old_states)[0].item())
                if old_position >= 0:
                    block[new_position, current, old_position] = 1
            blocks.append(block)
        return _SketchWeights(
            left, right, blocks,
            torch.ones(indices.shape[0], device=indices.device, dtype=dtype),
            diagnostics={'sampled_states': tuple(
                states.numel() for states in prefix_states)})


class MarginalSketch:
    """Marginal sketches that retain a local neighborhood at every cut.

    :meth:`markov` implements the neighbor-preserving sketches of Section 5.1
    of `Generative modeling via tensor train sketching
    <https://arxiv.org/abs/2202.11788>`_ (Hur, Hoskins, Lindsey, Stoudenmire and
    Khoo, 2022). Higher ``order`` values and an explicit contraction ``factor``
    are TensorKrowch extensions of the same recursive marginal structure.
    """

    def __init__(self,
                 order: int = 1,
                 factor: Optional[Sequence[torch.Tensor]] = None) -> None:
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError('`order` should be int type')
        if order < 1:
            raise ValueError('`order` should be positive')
        if factor is not None:
            factor = tuple(factor)
            if not all(isinstance(value, torch.Tensor) and value.ndim == 1
                       for value in factor):
                raise TypeError('`factor` should contain vectors')
        self.order = order
        self.factor = factor

    @classmethod
    def markov(cls,
               order: int = 1,
               factor: Optional[Sequence[torch.Tensor]] = None
               ) -> 'MarginalSketch':
        """Builds the standard neighbor-preserving Markov marginal sketch."""
        return cls(order=order, factor=factor)

    def builder(self, source: TensorSource) -> SketchSystemBuilder:
        """Returns a finite-support marginal-system builder."""
        return _SupportSketchSystemBuilder(source, self)

    def _weights(self, indices, input_dim, dtype, generator):
        del generator
        if self.factor is not None and len(self.factor) != len(input_dim):
            raise ValueError('`factor` should contain one vector per site')
        scale = torch.ones(indices.shape[0], dtype=dtype, device=indices.device)
        if self.factor is not None:
            for site, (factor, dimension) in enumerate(zip(
                    self.factor, input_dim)):
                if factor.shape != (dimension,):
                    raise ValueError(
                        f'`factor` at site {site} has an invalid shape')
                scale = scale * factor.to(
                    device=indices.device, dtype=dtype).index_select(
                        0, indices[:, site])

        left = []
        right = []
        blocks = []
        for site in range(len(input_dim) - 1):
            left_start = max(0, site + 1 - self.order)
            left_dim = input_dim[left_start:site + 1]
            left_ids = _ravel_subset(
                indices[:, left_start:site + 1], left_dim)
            left.append(_one_hot_ids(left_ids, prod(left_dim), dtype))

            right_stop = min(len(input_dim), site + 1 + self.order)
            right_dim = input_dim[site + 1:right_stop]
            right_ids = _ravel_subset(
                indices[:, site + 1:right_stop], right_dim)
            right.append(_one_hot_ids(right_ids, prod(right_dim), dtype))

            old_start = max(0, site - self.order)
            old_dim = input_dim[old_start:site]
            old_size = prod(old_dim) if old_dim else 1
            new_size = prod(left_dim)
            block = torch.zeros(
                new_size, input_dim[site], old_size,
                dtype=dtype, device=indices.device)
            old_ids = torch.arange(old_size, device=indices.device)
            for value in range(input_dim[site]):
                if self.order == 1:
                    new_ids = torch.full_like(old_ids, value)
                else:
                    tail_modulus = prod(old_dim[-(self.order - 1):]) \
                        if old_dim else 1
                    tail = torch.remainder(old_ids, tail_modulus)
                    new_ids = tail * input_dim[site] + value
                block[new_ids, value, old_ids] = 1
            blocks.append(block)
        return _SketchWeights(
            left, right, blocks, scale,
            diagnostics={
                'marginal_order': self.order,
                'weighted_marginal': self.factor is not None,
            })


def _gaussian(shape, dtype, device, generator, scale):
    """Draws real or complex Gaussian entries with explicit scale."""
    return torch.randn(
        *shape, dtype=dtype, device=device, generator=generator) * scale


def _left_orthogonal(core: torch.Tensor) -> torch.Tensor:
    """Orthogonalizes the outgoing columns of one TT core."""
    matrix = core.reshape(-1, core.shape[-1])
    if matrix.shape[0] >= matrix.shape[1]:
        matrix = torch.linalg.qr(matrix, mode='reduced').Q
    else:
        matrix = torch.linalg.qr(matrix.T, mode='reduced').Q.T
    return matrix.reshape(core.shape)


def _right_orthogonal(core: torch.Tensor) -> torch.Tensor:
    """Orthogonalizes the incoming rows of one TT core."""
    matrix = core.reshape(core.shape[0], -1)
    if matrix.shape[1] >= matrix.shape[0]:
        matrix = torch.linalg.qr(matrix.T, mode='reduced').Q.T
    else:
        matrix = torch.linalg.qr(matrix, mode='reduced').Q
    return matrix.reshape(core.shape)


class TTStackSketch:
    """Stacked Gaussian TT sketch with distinct stack and internal ranks.

    ``tt_rank=1`` is the separable Khatri-Rao limit. ``n_stacks=1`` is one
    Gaussian TT projection with ``tt_rank`` output rows at every partial cut.
    The full partial-sketch dimension is ``n_stacks * tt_rank``. Setting
    ``orthogonal=True`` selects a separately labelled orthogonalized variant;
    it does not claim the iid-Gaussian guarantees of the default construction.

    The random cores follow Definition 3.1 of `Linear-scaling Tensor Train
    Sketching <https://arxiv.org/abs/2603.11009>`_ by Paul Cazeaux, Mi-Song
    Dupuy and Rodrigo Figueroa Justiniano (2026). That paper establishes
    embedding guarantees for the global TTStack map. Using its partial suffix
    contractions inside TT-RS is an additional construction here, so the
    builder separately verifies the recursive left blocks and right-sketch
    dimensions required by :class:`CoreDeterminingSystem`.
    """

    def __init__(self,
                 tt_rank: int = 1,
                 n_stacks: int = 1,
                 orthogonal: bool = False) -> None:
        for name, value in (('tt_rank', tt_rank), ('n_stacks', n_stacks)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            if value < 1:
                raise ValueError(f'`{name}` should be positive')
        if not isinstance(orthogonal, bool):
            raise TypeError('`orthogonal` should be bool type')
        self.tt_rank = tt_rank
        self.n_stacks = n_stacks
        self.orthogonal = orthogonal

    def builder(self, source: TensorSource) -> SketchSystemBuilder:
        """Returns a finite-support Gaussian-TT-system builder."""
        return _SupportSketchSystemBuilder(source, self)

    def _weights(self, indices, input_dim, dtype, generator):
        feature_dim = self.tt_rank * self.n_stacks
        device = indices.device
        scale = 1 / sqrt(self.tt_rank)
        stack_scale = 1 / sqrt(self.n_stacks)

        left_cores = []
        right_cores = []
        for stack in range(self.n_stacks):
            stack_left = []
            stack_right = []
            for site, dimension in enumerate(input_dim):
                left_rank = 1 if site == 0 else self.tt_rank
                right_rank = 1 if site == len(input_dim) - 1 \
                    else self.tt_rank
                left_core = _gaussian(
                    (left_rank, dimension, right_rank),
                    dtype, device, generator, scale)
                right_core = _gaussian(
                    (left_rank, dimension, right_rank),
                    dtype, device, generator, scale)
                if self.orthogonal:
                    left_core = _left_orthogonal(left_core)
                    right_core = _right_orthogonal(right_core)
                stack_left.append(left_core)
                stack_right.append(right_core)
            left_cores.append(tuple(stack_left))
            right_cores.append(tuple(stack_right))

        blocks = []
        left = []
        state = torch.ones(
            indices.shape[0], self.n_stacks, 1,
            dtype=dtype, device=device)
        for site in range(len(input_dim) - 1):
            block = torch.zeros(
                feature_dim, input_dim[site],
                1 if site == 0 else feature_dim,
                dtype=dtype, device=device)
            next_state = []
            for stack in range(self.n_stacks):
                core = left_cores[stack][site]
                selected = core[:, indices[:, site], :].permute(1, 0, 2)
                value = torch.einsum(
                    'nba,nb->na', selected, state[:, stack])
                if site == 0:
                    value = value * stack_scale
                    block[
                        stack * self.tt_rank:(stack + 1) * self.tt_rank,
                        :,
                        0,
                    ] = core[0].T * stack_scale
                else:
                    start = stack * self.tt_rank
                    block[start:start + self.tt_rank, :,
                          start:start + self.tt_rank] = core.permute(2, 1, 0)
                next_state.append(value)
            state = torch.stack(next_state, dim=1)
            left.append(state.reshape(indices.shape[0], feature_dim))
            blocks.append(block)

        right = [None] * (len(input_dim) - 1)
        state = torch.ones(
            indices.shape[0], self.n_stacks, 1,
            dtype=dtype, device=device)
        for site in range(len(input_dim) - 1, 0, -1):
            next_state = []
            for stack in range(self.n_stacks):
                core = right_cores[stack][site]
                selected = core[:, indices[:, site], :].permute(1, 0, 2)
                value = torch.einsum(
                    'nba,na->nb', selected, state[:, stack])
                if site == len(input_dim) - 1:
                    value = value * stack_scale
                next_state.append(value)
            state = torch.stack(next_state, dim=1)
            right[site - 1] = state.reshape(indices.shape[0], feature_dim)

        return _SketchWeights(
            left, right, blocks,
            torch.ones(indices.shape[0], dtype=dtype, device=device),
            diagnostics={
                'tt_rank': self.tt_rank,
                'n_stacks': self.n_stacks,
                'sketch_dimension': feature_dim,
                'orthogonal': self.orthogonal,
                'distribution': (
                    'orthogonal_variant' if self.orthogonal
                    else 'gaussian_tt_stack'),
                'core_determining_gate': 'recursive_left_and_suffix_right',
            })


__all__ = [
    'SketchOperator',
    'SketchSystemBuilder',
    'CoreDeterminingSystem',
    'SampledSketch',
    'MarginalSketch',
    'TTStackSketch',
]
