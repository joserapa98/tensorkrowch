"""
This script contains:

    Public classes:
        * BlockSelection
        * CentralBlockSelector
        * PrescribedCentralBlockSelector
        * RingRankEstimate
        * RingRankEstimator
        * BlockSplit

    Internal functions:
        * _normalize_positive_int
        * _normalize_in_dim
        * _normalize_rank_spec
        * _pad_block_cores

    Public functions:
        * split_block_ttsvd
"""

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


def _normalize_in_dim(provider: Any) -> Tuple[int, ...]:
    """Obtains input dimensions from a provider or a direct sequence."""
    in_dim = getattr(provider, 'in_dim', provider)
    if isinstance(in_dim, (str, bytes)):
        raise TypeError(
            '`provider` should expose input dimensions or be their sequence')
    try:
        in_dim = tuple(in_dim)
    except TypeError as exc:
        raise TypeError(
            '`provider` should expose input dimensions or be their sequence') \
            from exc
    if not in_dim:
        raise ValueError('At least one input dimension is required')
    for value in in_dim:
        _normalize_positive_int(value, 'in_dim')
    return in_dim


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

    sites: Sequence[int]  # Ordered zero-based sites represented by this object
    in_dim: Sequence[int]  # Input dimension at each represented site
    left_rank_cap: int  # Rank cap at the left block boundary
    right_rank_cap: int  # Rank cap at the right block boundary
    input_capacity: int  # Product of input dimensions in the selected block
    required_input_capacity: int  # Input capacity required by the external ranks
    feasible: bool  # Whether all requested capacities are satisfied
    reason: str  # Reason for the update or selection outcome
    boundary: Optional[str] = None  # Boundary touched by the selected block
    # Sequence of blocks visited during selection
    growth: Sequence[Tuple[int, int]] = ()

    def __post_init__(self) -> None:
        sites = tuple(self.sites)
        in_dim = tuple(self.in_dim)
        growth = tuple(tuple(interval) for interval in self.growth)
        if not sites:
            raise ValueError('`sites` should contain at least one site')
        if any(isinstance(site, bool) or not isinstance(site, int)
               for site in sites):
            raise TypeError('`sites` should contain integer indices')
        if sites != tuple(range(sites[0], sites[-1] + 1)):
            raise ValueError('`sites` should describe one contiguous block')
        if len(in_dim) != len(sites):
            raise ValueError(
                '`in_dim` should contain one dimension per selected site')
        for value in in_dim:
            _normalize_positive_int(value, 'in_dim')
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
        object.__setattr__(self, 'in_dim', in_dim)
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

    The provider can be any object exposing ``in_dim`` or the input-dimension
    sequence itself. ``rank`` follows the standard TR convention: a scalar is
    shared by every link and a sequence stores the right-link cap of each site.
    """

    def select(self,
               provider: Any,
               rank: _Rank,
               center: Optional[int] = None,
               *,
               bounds: Optional[Tuple[int, int]] = None) -> BlockSelection:
        """Grows a contiguous block until its input capacity exceeds the rank
        product.

        Parameters
        ----------
        provider : RingTargetProvider or sequence[int]
            Provider declaring ``in_dim``, or the dimensions themselves.
        rank : int or sequence[int]
            Right-link caps of the complete ring. The scalar form shares one
            cap.
        center : int, optional
            Initial site. Defaults to the middle of the allowed interval.
        bounds : tuple[int, int], optional
            Inclusive lower and upper site bounds. Defaults to the entire
            chain.

        Returns
        -------
        BlockSelection
            Selected sites, growth history and capacity checks. Exhausting the
            bounds returns an infeasible selection with a reason; it does not
            discard the block already reached. No tensors are contracted during
            selection.
        """
        in_dim = _normalize_in_dim(provider)
        n_sites = len(in_dim)
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
            input_capacity = prod(in_dim[left:(right + 1)])
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
            in_dim=in_dim[left:(right + 1)],
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
        """Selects one prescribed center without adaptive block growth.

        Uses the same arguments and returns as
        :meth:`CentralBlockSelector.select`. The fixed-rank TT-to-TR path
        deliberately opens one site, even if its input capacity does not
        satisfy the adaptive selector's strict rank-product test.
        """
        in_dim = _normalize_in_dim(provider)
        rank_spec = _normalize_rank_spec(rank, len(in_dim))
        if center is None:
            center = len(in_dim) // 2
        if isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int type or None')
        if center <= 0 or center >= len(in_dim) - 1:
            raise ValueError(
                '`center` should be an internal TT site or TR site')
        if bounds is not None and not (bounds[0] <= center <= bounds[1]):
            raise ValueError('`center` should lie inside `bounds`')
        return BlockSelection(
            sites=(center,),
            in_dim=(in_dim[center],),
            left_rank_cap=rank_spec[center - 1],
            right_rank_cap=rank_spec[center],
            input_capacity=in_dim[center],
            required_input_capacity=1,
            feasible=True,
            reason='prescribed_fixed_rank_center',
            boundary=None,
            growth=((center, center),))


@dataclass(frozen=True)
class RingRankEstimate:
    """Stores a balanced ring-rank estimate and its cap diagnostics."""

    left_rank: int  # Estimated left virtual rank
    right_rank: int  # Estimated right virtual rank
    cyclic_rank: int  # Estimated or prescribed closing rank
    cyclic_rank_estimate: float  # Unclipped estimate of the cyclic rank
    rank_caps: Tuple[int, int, int]  # Requested caps for the estimated virtual links
    feasible: bool  # Whether all requested capacities are satisfied
    limitations: Sequence[str] = ()  # Capacity constraints that remain unsatisfied

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
        """Estimates three compatible local ring ranks under explicit caps.

        Parameters
        ----------
        left_dim, right_dim : int
            Positive external dimensions of the local target.
        auxiliary_rank : int
            Positive auxiliary capacity used by the rank-balancing heuristic.
        rank_caps : sequence[int]
            Three positive caps in left, right and cyclic order.

        Returns
        -------
        RingRankEstimate
            Estimated ranks and all unsatisfied capacity constraints. This is a
            capacity heuristic, not a numerical rank determination from
            singular values.
        """
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

    cores: Sequence[torch.Tensor]  # Raw cores in site order
    in_dim: Sequence[int]  # Input dimension at each represented site
    effective_rank: Sequence[int]  # Ranks retained before optional zero padding
    rank: Sequence[int]  # Right-link ranks in core order
    requested_rank: Optional[int]  # Requested shared rank or right-link rank profile
    padding: Sequence[int]  # Number of zero channels added at each internal cut
    # Structured measurements collected during execution
    metrics: DecompositionMetrics = field(default_factory=DecompositionMetrics)
    # Small structural and algorithm configuration metadata
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cores = tuple(self.cores)
        in_dim = tuple(self.in_dim)
        effective_rank = tuple(self.effective_rank)
        rank = tuple(self.rank)
        padding = tuple(self.padding)
        if not cores:
            raise ValueError('`cores` should contain at least one core')
        if len(cores) != len(in_dim):
            raise ValueError('`cores` and `in_dim` should have equal length')
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
               for core, dim in zip(cores, in_dim)):
            raise ValueError('Core input dimensions should match `in_dim`')
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
        object.__setattr__(self, 'in_dim', in_dim)
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
        in_dim: Sequence[int],
        rank: Optional[int] = None,
        cutoff: Optional[float] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        cum_percentage: Optional[float] = None,
        renormalize: bool = False,
        pad_rank: bool = False,
        collect_metrics: bool = False,
        out_device: Optional[Union[str, torch.device]] = None) -> BlockSplit:
    r"""Splits a supercore while retaining its two external rank axes.

    ``block`` has shape ``(left_rank, *in_dim, right_rank)``. The external
    ranks are fused into the first and last TT-SVD inputs and restored after
    the split. ``rank`` is one shared upper bound for all internal cuts.
    Padding to that bound is performed only when ``pad_rank=True``.

    Parameters
    ----------
    block : torch.Tensor
        Supercore with shape ``(left_rank, *in_dim, right_rank)``.
    in_dim : sequence[int]
        Input dimensions of the sites to recover, in their original order.
    rank : int, optional
        Maximum rank allowed at every link. At each SVD cut, at most this many
        singular values are retained.
    cutoff : float, optional
        Minimum singular value to keep. It must be finite and non-negative.
        Singular values ``<= cutoff`` are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the accumulated sum of squares is ``<= atol``. It must be finite and
        non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the tail sum of squares divided by the total sum of squares is ``<=
        rtol``. It must be finite and in ``[0, 1]``.
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent to
        setting ``rtol = 1 - cum_percentage``. It must be finite and in ``[0,
        1]``.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
            cum\_percentage
    renormalize : bool
        Whether the TT-SVD subroutine extracts intermediate scales. Default is
        False.
    pad_rank : bool
        Zero-pad internal links to the shared cap after truncation. Default is
        False. Padding is recorded separately from the retained numerical
        ranks.
    collect_metrics : bool
        Whether to retain SVD truncation and timing records. Default is False.
    out_device : str or torch.device, optional
        Device for the returned cores. Default is None, retaining the active
        device.

    Returns
    -------
    BlockSplit
        Recovered cores, effective and stored ranks, optional padding and
        metrics.
    """
    if not isinstance(block, torch.Tensor):
        raise TypeError('`block` should be torch.Tensor type')
    in_dim = _normalize_in_dim(in_dim)
    if block.ndim != len(in_dim) + 2:
        raise ValueError(
            '`block` should have two external rank axes around `in_dim`')
    if tuple(block.shape[1:-1]) != in_dim:
        raise ValueError('The inner axes of `block` should match `in_dim`')
    if rank is not None:
        rank = _normalize_positive_int(rank, 'rank')
    if not isinstance(pad_rank, bool):
        raise TypeError('`pad_rank` should be bool type')
    if pad_rank and rank is None:
        raise ValueError('`rank` is required when `pad_rank=True`')
    if not isinstance(collect_metrics, bool):
        raise TypeError('`collect_metrics` should be bool type')

    if len(in_dim) == 1:
        core = block.reshape(block.shape[0], in_dim[0], block.shape[-1])
        if out_device is not None:
            core = core.to(device=out_device)
        cores = (core,)
        return BlockSplit(
            cores=cores,
            in_dim=in_dim,
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
    fused_in_dim = (
        left_rank * in_dim[0],
        *in_dim[1:-1],
        in_dim[-1] * right_rank)
    tensor = block.reshape(fused_in_dim)
    result = TTSVD(tensor, out_device=out_device).fit(
        rank=rank,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
        cum_percentage=cum_percentage,
        renormalize=renormalize,
        collect_metrics=collect_metrics)

    cores = list(result.cores)
    cores[0] = cores[0].reshape(
        left_rank, in_dim[0], cores[0].shape[-1])
    cores[-1] = cores[-1].reshape(
        cores[-1].shape[0], in_dim[-1], right_rank)
    effective_rank = tuple(result.rank)
    padding = tuple(0 for _ in effective_rank)
    if pad_rank:
        padding = tuple(rank - value for value in effective_rank)
        cores = list(_pad_block_cores(cores, rank))
    actual_rank = tuple(core.shape[-1] for core in cores[:-1])

    return BlockSplit(
        cores=cores,
        in_dim=in_dim,
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
