"""Block selection, rank discovery and splitting for tensor rings."""

from dataclasses import dataclass, field
from math import ceil, prod, sqrt
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.decompositions.metrics import DecompositionMetrics
from tensorkrowch.decompositions.svd.tt import TTSVD


_Rank = Union[int, Sequence[int]]


def _normalize_positive_int(value: int, name: str) -> int:
    """Validates one positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'`{name}` should be int type')
    if value < 1:
        raise ValueError(f'`{name}` should be positive')
    return value


def _normalize_input_dim(provider: Any) -> Tuple[int, ...]:
    """Obtains input dimensions from a provider or a direct sequence."""
    input_dim = getattr(provider, 'input_dim', provider)
    if isinstance(input_dim, (str, bytes)):
        raise TypeError(
            '`provider` should expose input dimensions or be their sequence')
    try:
        input_dim = tuple(input_dim)
    except TypeError as exc:
        raise TypeError(
            '`provider` should expose input dimensions or be their sequence') \
            from exc
    if not input_dim:
        raise ValueError('At least one input dimension is required')
    for value in input_dim:
        _normalize_positive_int(value, 'input_dim')
    return input_dim


def _normalize_rank_spec(rank: _Rank, n_sites: int) -> Tuple[int, ...]:
    """Normalizes one cap per right link of a tensor ring."""
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
                '`rank` should contain one right-link cap per site')
    for value in ranks:
        _normalize_positive_int(value, 'rank')
    return ranks


@dataclass(frozen=True)
class BlockSelection:
    """Describes a contiguous block selected for a local ring operation."""

    sites: Sequence[int]
    input_dim: Sequence[int]
    left_rank_cap: int
    right_rank_cap: int
    input_capacity: int
    required_input_capacity: int
    feasible: bool
    reason: str
    boundary: Optional[str] = None
    growth: Sequence[Tuple[int, int]] = ()

    def __post_init__(self) -> None:
        sites = tuple(self.sites)
        input_dim = tuple(self.input_dim)
        growth = tuple(tuple(interval) for interval in self.growth)
        if not sites:
            raise ValueError('`sites` should contain at least one site')
        if any(isinstance(site, bool) or not isinstance(site, int)
               for site in sites):
            raise TypeError('`sites` should contain integer indices')
        if sites != tuple(range(sites[0], sites[-1] + 1)):
            raise ValueError('`sites` should describe one contiguous block')
        if len(input_dim) != len(sites):
            raise ValueError(
                '`input_dim` should contain one dimension per selected site')
        for value in input_dim:
            _normalize_positive_int(value, 'input_dim')
        for name in (
                'left_rank_cap',
                'right_rank_cap',
                'input_capacity',
                'required_input_capacity'):
            _normalize_positive_int(getattr(self, name), name)
        if not isinstance(self.feasible, bool):
            raise TypeError('`feasible` should be bool type')
        if not isinstance(self.reason, str):
            raise TypeError('`reason` should be str type')
        if self.boundary not in (None, 'left', 'right', 'both'):
            raise ValueError(
                "`boundary` should be None, 'left', 'right' or 'both'")
        if any(len(interval) != 2 for interval in growth):
            raise ValueError('Every growth interval should have two endpoints')
        if any(any(isinstance(site, bool) or not isinstance(site, int)
                   for site in interval)
               for interval in growth):
            raise TypeError('Growth intervals should contain integer indices')
        object.__setattr__(self, 'sites', sites)
        object.__setattr__(self, 'input_dim', input_dim)
        object.__setattr__(self, 'growth', growth)

    @property
    def left(self) -> int:
        """First site in the selected block."""
        return self.sites[0]

    @property
    def right(self) -> int:
        """Last site in the selected block."""
        return self.sites[-1]


class CentralBlockSelector:
    """Selects a refinable block by balanced growth around a seed site.

    The provider can be any object exposing ``input_dim`` or the input-dimension
    sequence itself. ``rank`` follows the standard TR convention: a scalar is
    shared by every link and a sequence stores the right-link cap of each site.
    """

    def select(self,
               provider: Any,
               rank: _Rank,
               center: Optional[int] = None,
               *,
               bounds: Optional[Tuple[int, int]] = None) -> BlockSelection:
        """Grows a contiguous block until input capacity exceeds rank caps."""
        input_dim = _normalize_input_dim(provider)
        n_sites = len(input_dim)
        rank_spec = _normalize_rank_spec(rank, n_sites)

        if bounds is None:
            lower, upper = 0, n_sites - 1
        else:
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise TypeError('`bounds` should be a pair of site indices')
            lower, upper = bounds
            if any(isinstance(value, bool) or not isinstance(value, int)
                   for value in bounds):
                raise TypeError('`bounds` should contain integer site indices')
            if lower < 0 or upper >= n_sites or lower > upper:
                raise ValueError('`bounds` should define a valid site interval')

        if center is None:
            center = (lower + upper) // 2
        elif isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int or None')
        if center < lower or center > upper:
            raise ValueError('`center` should lie inside `bounds`')

        left = right = center
        target_center = (lower + upper) / 2
        alternate_right = True
        growth = [(left, right)]

        while True:
            left_rank_cap = rank_spec[(left - 1) % n_sites]
            right_rank_cap = rank_spec[right]
            input_capacity = prod(input_dim[left:(right + 1)])
            required_input_capacity = left_rank_cap * right_rank_cap + 1
            if input_capacity >= required_input_capacity:
                feasible = True
                reason = 'input_capacity_exceeds_external_rank_caps'
                break

            can_grow_left = left > lower
            can_grow_right = right < upper
            if not can_grow_left and not can_grow_right:
                feasible = False
                reason = 'available_sites_exhausted'
                break

            block_center = (left + right) / 2
            if block_center < target_center:
                grow_right = True
            elif block_center > target_center:
                grow_right = False
            else:
                grow_right = alternate_right
                alternate_right = not alternate_right

            if grow_right and can_grow_right:
                right += 1
            elif not grow_right and can_grow_left:
                left -= 1
            elif can_grow_right:
                right += 1
            else:
                left -= 1
            growth.append((left, right))

        touches_left = left == 0
        touches_right = right == n_sites - 1
        if touches_left and touches_right:
            boundary = 'both'
        elif touches_left:
            boundary = 'left'
        elif touches_right:
            boundary = 'right'
        else:
            boundary = None

        return BlockSelection(
            sites=tuple(range(left, right + 1)),
            input_dim=input_dim[left:(right + 1)],
            left_rank_cap=left_rank_cap,
            right_rank_cap=right_rank_cap,
            input_capacity=input_capacity,
            required_input_capacity=required_input_capacity,
            feasible=feasible,
            reason=reason,
            boundary=boundary,
            growth=growth)


class PrescribedCentralBlockSelector(CentralBlockSelector):
    """Selects one fixed center without adaptive injectivity growth."""

    def select(self,
               provider: Any,
               rank: _Rank,
               center: Optional[int] = None,
               *,
               bounds: Optional[Tuple[int, int]] = None) -> BlockSelection:
        """Returns one internal site with its adjacent right-link caps."""
        input_dim = _normalize_input_dim(provider)
        rank_spec = _normalize_rank_spec(rank, len(input_dim))
        if center is None:
            center = len(input_dim) // 2
        if isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int type or None')
        if center <= 0 or center >= len(input_dim) - 1:
            raise ValueError(
                '`center` should be an internal TT site or TR site')
        if bounds is not None and not (bounds[0] <= center <= bounds[1]):
            raise ValueError('`center` should lie inside `bounds`')
        return BlockSelection(
            sites=(center,),
            input_dim=(input_dim[center],),
            left_rank_cap=rank_spec[center - 1],
            right_rank_cap=rank_spec[center],
            input_capacity=input_dim[center],
            required_input_capacity=1,
            feasible=True,
            reason='prescribed_fixed_rank_center',
            boundary=None,
            growth=((center, center),))


@dataclass(frozen=True)
class RingRankEstimate:
    """Stores a balanced ring-rank estimate and its cap diagnostics."""

    left_rank: int
    right_rank: int
    cyclic_rank: int
    cyclic_rank_estimate: float
    rank_caps: Tuple[int, int, int]
    feasible: bool
    limitations: Sequence[str] = ()

    def __post_init__(self) -> None:
        for name in ('left_rank', 'right_rank', 'cyclic_rank'):
            _normalize_positive_int(getattr(self, name), name)
        if not isinstance(self.cyclic_rank_estimate, float):
            raise TypeError('`cyclic_rank_estimate` should be float type')
        if self.cyclic_rank_estimate <= 0:
            raise ValueError('`cyclic_rank_estimate` should be positive')
        if len(self.rank_caps) != 3:
            raise ValueError('`rank_caps` should contain three values')
        for value in self.rank_caps:
            _normalize_positive_int(value, 'rank_caps')
        if not isinstance(self.feasible, bool):
            raise TypeError('`feasible` should be bool type')
        limitations = tuple(self.limitations)
        if not all(isinstance(value, str) for value in limitations):
            raise TypeError('`limitations` should contain strings')
        object.__setattr__(self, 'limitations', limitations)

    @property
    def rank(self) -> Tuple[int, int, int]:
        """Returns left, right and cyclic effective ranks."""
        return self.left_rank, self.right_rank, self.cyclic_rank


class RingRankEstimator:
    """Estimates balanced adjacent and cyclic ranks under explicit caps."""

    def estimate(self,
                 left_dim: int,
                 right_dim: int,
                 auxiliary_rank: int,
                 rank_caps: Sequence[int]) -> RingRankEstimate:
        """Balances three ranks and reports every unsatisfied capacity."""
        left_dim = _normalize_positive_int(left_dim, 'left_dim')
        right_dim = _normalize_positive_int(right_dim, 'right_dim')
        auxiliary_rank = _normalize_positive_int(
            auxiliary_rank, 'auxiliary_rank')
        try:
            rank_caps = tuple(rank_caps)
        except TypeError as exc:
            raise TypeError('`rank_caps` should be a sequence of ints') from exc
        if len(rank_caps) != 3:
            raise ValueError(
                '`rank_caps` should contain left, right and cyclic caps')
        for value in rank_caps:
            _normalize_positive_int(value, 'rank_caps')
        left_cap, right_cap, cyclic_cap = rank_caps

        cyclic_estimate = sqrt(
            float(left_dim * right_dim) / float(auxiliary_rank))
        cyclic_lower_bound = max(
            1,
            int(ceil(float(left_dim) / left_cap)),
            int(ceil(float(right_dim) / right_cap)))
        cyclic_rank = min(
            cyclic_cap,
            max(int(ceil(cyclic_estimate)), cyclic_lower_bound))

        left_rank = min(
            left_cap,
            max(
                1,
                int(ceil(float(left_dim) / cyclic_estimate)),
                int(ceil(float(left_dim) / cyclic_rank))))
        right_rank = min(
            right_cap,
            max(
                1,
                int(ceil(float(right_dim) / cyclic_estimate)),
                int(ceil(float(right_dim) / cyclic_rank))))

        while left_rank * right_rank < auxiliary_rank:
            can_left = left_rank < left_cap
            can_right = right_rank < right_cap
            if not can_left and not can_right:
                break
            if can_left and (
                    not can_right or
                    left_rank / left_cap <= right_rank / right_cap):
                left_rank += 1
            else:
                right_rank += 1

        limitations = []
        if cyclic_rank * left_rank < left_dim:
            limitations.append('left_dim_exceeds_rank_capacity')
        if cyclic_rank * right_rank < right_dim:
            limitations.append('right_dim_exceeds_rank_capacity')
        if left_rank * right_rank < auxiliary_rank:
            limitations.append('auxiliary_rank_exceeds_adjacent_rank_capacity')

        return RingRankEstimate(
            left_rank=left_rank,
            right_rank=right_rank,
            cyclic_rank=cyclic_rank,
            cyclic_rank_estimate=float(cyclic_estimate),
            rank_caps=rank_caps,
            feasible=not limitations,
            limitations=limitations)


@dataclass(frozen=True)
class BlockSplit:
    """Stores a TT-SVD split of a supercore and optional explicit padding."""

    cores: Sequence[torch.Tensor]
    input_dim: Sequence[int]
    effective_rank: Sequence[int]
    rank: Sequence[int]
    requested_rank: Optional[int]
    padding: Sequence[int]
    metrics: DecompositionMetrics = field(default_factory=DecompositionMetrics)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cores = tuple(self.cores)
        input_dim = tuple(self.input_dim)
        effective_rank = tuple(self.effective_rank)
        rank = tuple(self.rank)
        padding = tuple(self.padding)
        if not cores:
            raise ValueError('`cores` should contain at least one core')
        if len(cores) != len(input_dim):
            raise ValueError('`cores` and `input_dim` should have equal length')
        if any(not isinstance(core, torch.Tensor) or core.ndim != 3
               for core in cores):
            raise ValueError('Block cores should be three-dimensional tensors')
        if len(rank) != len(cores) - 1 or \
                len(effective_rank) != len(rank) or len(padding) != len(rank):
            raise ValueError(
                'Internal rank metadata should contain one value per cut')
        for name, values in (
                ('effective_rank', effective_rank), ('rank', rank)):
            for value in values:
                _normalize_positive_int(value, name)
        if self.requested_rank is not None:
            _normalize_positive_int(self.requested_rank, 'requested_rank')
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in padding):
            raise TypeError('`padding` should contain integers')
        if any(core.shape[1] != dim
               for core, dim in zip(cores, input_dim)):
            raise ValueError('Core input dimensions should match `input_dim`')
        if any(cores[k].shape[-1] != cores[k + 1].shape[0]
               for k in range(len(cores) - 1)):
            raise ValueError('Adjacent block ranks should match')
        if tuple(core.shape[-1] for core in cores[:-1]) != rank:
            raise ValueError('`rank` should contain actual internal ranks')
        if any(value < 0 for value in padding):
            raise ValueError('`padding` values should be non-negative')
        if any(actual != effective + added
               for actual, effective, added in zip(
                   rank, effective_rank, padding)):
            raise ValueError(
                '`padding` should record rank minus effective rank')
        if not isinstance(self.metrics, DecompositionMetrics):
            raise TypeError('`metrics` should be DecompositionMetrics type')
        if not isinstance(self.metadata, Mapping):
            raise TypeError('`metadata` should be a mapping')
        object.__setattr__(self, 'cores', cores)
        object.__setattr__(self, 'input_dim', input_dim)
        object.__setattr__(self, 'effective_rank', effective_rank)
        object.__setattr__(self, 'rank', rank)
        object.__setattr__(self, 'padding', padding)
        object.__setattr__(self, 'metadata', dict(self.metadata))

    @property
    def padded(self) -> bool:
        """Whether any internal link was enlarged with explicit zeros."""
        return any(self.padding)

    def contract_dense(self) -> torch.Tensor:
        """Contracts the block while preserving both external rank axes."""
        tensor = self.cores[0]
        for core in self.cores[1:]:
            tensor = torch.tensordot(tensor, core, dims=([-1], [0]))
        return tensor


def _pad_block_cores(cores: Sequence[torch.Tensor],
                     rank: int) -> Tuple[torch.Tensor, ...]:
    """Zero-pads every internal link to one explicitly requested rank."""
    padded = []
    for site, core in enumerate(cores):
        left_rank = core.shape[0] if site == 0 else rank
        right_rank = core.shape[-1] if site == len(cores) - 1 else rank
        padded_core = core.new_zeros(left_rank, core.shape[1], right_rank)
        padded_core[:core.shape[0], :, :core.shape[-1]] = core
        padded.append(padded_core)
    return tuple(padded)


def split_block_ttsvd(
        block: torch.Tensor,
        input_dim: Sequence[int],
        rank: Optional[int] = None,
        cutoff: Optional[float] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        cum_percentage: Optional[float] = None,
        renormalize: bool = False,
        pad_rank: bool = False,
        collect_metrics: bool = False,
        output_device: Optional[Union[str, torch.device]] = None) -> BlockSplit:
    """Splits a supercore while retaining its two external rank axes.

    ``block`` has shape ``(left_rank, *input_dim, right_rank)``. The external
    ranks are fused into the first and last TT-SVD inputs and restored after
    the split. ``rank`` is one shared upper bound for all internal cuts.
    Padding to that bound is performed only when ``pad_rank=True``.
    """
    if not isinstance(block, torch.Tensor):
        raise TypeError('`block` should be torch.Tensor type')
    input_dim = _normalize_input_dim(input_dim)
    if block.ndim != len(input_dim) + 2:
        raise ValueError(
            '`block` should have two external rank axes around `input_dim`')
    if tuple(block.shape[1:-1]) != input_dim:
        raise ValueError('The inner axes of `block` should match `input_dim`')
    if rank is not None:
        rank = _normalize_positive_int(rank, 'rank')
    if not isinstance(pad_rank, bool):
        raise TypeError('`pad_rank` should be bool type')
    if pad_rank and rank is None:
        raise ValueError('`rank` is required when `pad_rank=True`')
    if not isinstance(collect_metrics, bool):
        raise TypeError('`collect_metrics` should be bool type')

    if len(input_dim) == 1:
        core = block.reshape(block.shape[0], input_dim[0], block.shape[-1])
        if output_device is not None:
            core = core.to(device=output_device)
        cores = (core,)
        return BlockSplit(
            cores=cores,
            input_dim=input_dim,
            effective_rank=(),
            rank=(),
            requested_rank=rank,
            padding=(),
            metadata={
                'algorithm': 'block_ttsvd',
                'pad_rank': pad_rank,
                'left_rank': block.shape[0],
                'right_rank': block.shape[-1],
            })

    left_rank = block.shape[0]
    right_rank = block.shape[-1]
    fused_input_dim = (
        left_rank * input_dim[0],
        *input_dim[1:-1],
        input_dim[-1] * right_rank)
    tensor = block.reshape(fused_input_dim)
    result = TTSVD(tensor, out_device=output_device).fit(
        rank=rank,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
        cum_percentage=cum_percentage,
        renormalize=renormalize,
        collect_metrics=collect_metrics)

    cores = list(result.cores)
    cores[0] = cores[0].reshape(
        left_rank, input_dim[0], cores[0].shape[-1])
    cores[-1] = cores[-1].reshape(
        cores[-1].shape[0], input_dim[-1], right_rank)
    effective_rank = tuple(result.rank)
    padding = tuple(0 for _ in effective_rank)
    if pad_rank:
        padding = tuple(rank - value for value in effective_rank)
        cores = list(_pad_block_cores(cores, rank))
    actual_rank = tuple(core.shape[-1] for core in cores[:-1])

    return BlockSplit(
        cores=cores,
        input_dim=input_dim,
        effective_rank=effective_rank,
        rank=actual_rank,
        requested_rank=rank,
        padding=padding,
        metrics=result.metrics,
        metadata={
            'algorithm': 'block_ttsvd',
            'pad_rank': pad_rank,
            'left_rank': left_rank,
            'right_rank': right_rank,
        })


__all__ = [
    'BlockSelection',
    'CentralBlockSelector',
    'PrescribedCentralBlockSelector',
    'RingRankEstimate',
    'RingRankEstimator',
    'BlockSplit',
    'split_block_ttsvd',
]
