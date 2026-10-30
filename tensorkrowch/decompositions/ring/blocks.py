"""
This script contains:

    Internal classes:
        * _BlockSelection
        * _CentralBlockSelector
        * _PrescribedCentralBlockSelector
        * _RingRankEstimate
        * _RingRankEstimator

    Internal functions:
        * _split_block
        * _discovery_partition
        * _boundary_partition
        * _numerical_rank

"""

from dataclasses import dataclass
from math import ceil, prod, sqrt
from typing import Any, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.formats.formats1d import split_block
from tensorkrowch.utils import truncated_svd

from tensorkrowch.decompositions._utils import (_normalize_dims,
                                                _normalize_positive_int,
                                                _normalize_rank)
from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 TruncationRecord)


_Rank = Union[int, Sequence[int]]


@dataclass(frozen=True)
class _BlockSelection:  # MARK: _BlockSelection
    """Describes a contiguous block selected for a local ring operation."""

    sites: Sequence[int]  # Ordered zero-based sites represented by this object
    in_dim: Sequence[int]  # Input dimension at each represented site
    left_rank_cap: int  # Rank cap at the left block boundary
    right_rank_cap: int  # Rank cap at the right block boundary
    input_capacity: int  # Product of input dimensions in the selected block
    required_input_capacity: int  # Capacity required by the external ranks
    feasible: bool  # Whether all requested capacities are satisfied
    reason: str  # Reason for the update or selection outcome
    boundary: Optional[str] = None  # Boundary touched by the selected block
    # Sequence of blocks visited during selection
    growth: Sequence[Tuple[int, int]] = ()

    def __post_init__(self) -> None:
        sites = tuple(self.sites)
        in_dim = _normalize_dims(self.in_dim, 'in_dim')
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


class _CentralBlockSelector:  # MARK: _CentralBlockSelector
    """
    Selects a refinable block by balanced growth around a seed site.

    The provider can be any object exposing ``in_dim`` or the input-dimension
    sequence itself. ``rank`` follows the standard TR convention: a scalar is
    shared by every link and a sequence stores the right-link cap of each site.
    """

    def select(self,
               provider: Any,
               rank: _Rank,
               center: Optional[int] = None,
               *,
               bounds: Optional[Tuple[int, int]] = None) -> _BlockSelection:
        """
        Grows a contiguous block until its input capacity exceeds the ``rank``
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
            Inclusive lower and upper site ``bounds``. Defaults to the entire
            chain.

        Returns
        -------
        _BlockSelection
            Selected sites, growth history and capacity checks. Exhausting the
            ``bounds`` returns an infeasible selection with a reason, retaining
            the block already reached. No tensors are contracted during
            selection.
        """
        in_dim = _normalize_dims(
            getattr(provider, 'in_dim', provider), 'in_dim')
        n_sites = len(in_dim)
        rank_spec = _normalize_rank(rank, n_sites)

        if bounds is None:
            lower, upper = 0, n_sites - 1
        else:
            if not isinstance(bounds, tuple) or (len(bounds) != 2):
                raise TypeError('`bounds` should be a pair of site indices')
            lower, upper = bounds
            if any(isinstance(value, bool) or not isinstance(value, int)
                   for value in bounds):
                raise TypeError('`bounds` should contain integer site indices')
            if (lower < 0) or (upper >= n_sites) or (lower > upper):
                raise ValueError(
                    '`bounds` should define a valid site interval')

        if center is None:
            center = (lower + upper) // 2
        elif isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int or None')
        if (center < lower) or (center > upper):
            raise ValueError('`center` should lie inside `bounds`')

        # Grow the block until its physical space exceeds both external ranks.
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

        return _BlockSelection(
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


class _PrescribedCentralBlockSelector(_CentralBlockSelector):
    """Selects one fixed center without adaptive injectivity growth."""

    def select(self,
               provider: Any,
               rank: _Rank,
               center: Optional[int] = None,
               *,
               bounds: Optional[Tuple[int, int]] = None) -> _BlockSelection:
        """
        Selects one prescribed ``center`` without adaptive block growth.

        Uses the same arguments and returns as
        :meth:`_CentralBlockSelector.select`. The fixed-``rank`` TT-to-TR path
        deliberately opens one site, even if its input capacity does not
        satisfy the adaptive selector's strict ``rank``-product test.
        """
        in_dim = _normalize_dims(
            getattr(provider, 'in_dim', provider), 'in_dim')
        rank_spec = _normalize_rank(rank, len(in_dim))
        if center is None:
            center = len(in_dim) // 2
        if isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int type or None')
        if (center <= 0) or (center >= len(in_dim) - 1):
            raise ValueError(
                '`center` should be an internal TT site or TR site')
        if (bounds is not None) and not (bounds[0] <= center <= bounds[1]):
            raise ValueError('`center` should lie inside `bounds`')
        return _BlockSelection(
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
class _RingRankEstimate:  # MARK: _RingRankEstimate
    """Stores a balanced ring-rank estimate and its cap diagnostics."""

    left_rank: int  # Estimated left virtual rank
    right_rank: int  # Estimated right virtual rank
    cyclic_rank: int  # Estimated or prescribed closing rank
    cyclic_rank_estimate: float  # Unclipped estimate of the cyclic rank
    rank_caps: Tuple[int, int, int]  # Caps of the three estimated ranks
    feasible: bool  # Whether all requested capacities are satisfied
    limitations: Sequence[str] = ()  # Unsatisfied capacity constraints

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


class _RingRankEstimator:  # MARK: _RingRankEstimator
    """Estimates balanced adjacent and cyclic ranks under explicit caps."""

    def estimate(self,
                 left_dim: int,
                 right_dim: int,
                 auxiliary_rank: int,
                 rank_caps: Sequence[int]) -> _RingRankEstimate:
        """
        Estimates three compatible local ring ranks under explicit caps.

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
        _RingRankEstimate
            Estimated ranks and all unsatisfied capacity constraints. This is a
            capacity heuristic, not a numerical rank determination from
            singular values.
        """
        left_dim = _normalize_positive_int(left_dim, 'left_dim')
        right_dim = _normalize_positive_int(right_dim, 'right_dim')
        auxiliary_rank = _normalize_positive_int(
            auxiliary_rank, 'auxiliary_rank')
        rank_caps = _normalize_dims(rank_caps, 'rank_caps')
        if len(rank_caps) != 3:
            raise ValueError(
                '`rank_caps` should contain left, right and cyclic caps')
        left_cap, right_cap, cyclic_cap = rank_caps

        # Infer the closing rank from the three injective-block rank products.
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
                    (left_rank / left_cap <= right_rank / right_cap)):
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

        return _RingRankEstimate(
            left_rank=left_rank,
            right_rank=right_rank,
            cyclic_rank=cyclic_rank,
            cyclic_rank_estimate=float(cyclic_estimate),
            rank_caps=rank_caps,
            feasible=not limitations,
            limitations=limitations)


def _split_block(block: torch.Tensor,
                 in_dim: Sequence[int],
                 rank: Optional[_Rank] = None,
                 cutoff: Optional[float] = None,
                 atol: Optional[float] = None,
                 rtol: Optional[float] = None,
                 cum_percentage: Optional[float] = None,
                 *,
                 renormalize: bool = False,
                 metrics: Optional[DecompositionMetrics] = None
                 ) -> Tuple[torch.Tensor, ...]:
    """
    Splits ``(left, *in_dim, right)`` while preserving both external bonds.

    ``rank`` bounds the internal cuts only: an integer applies to every cut,
    a sequence has one entry per cut, and None leaves them unbounded. Each
    split delegates to :func:`~tensorkrowch.formats.split_block`; variable
    caps use successive two-site views. Singular values are absorbed on the
    right. Retained dimensions are returned without zero padding.
    """
    if not isinstance(block, torch.Tensor):
        raise TypeError('`block` should be torch.Tensor type')
    in_dim = _normalize_dims(in_dim, 'in_dim')
    if (block.ndim != len(in_dim) + 2) or \
            (tuple(block.shape[1:-1]) != in_dim):
        raise ValueError('`block` should have shape (left, *in_dim, right)')
    if (block.shape[0] < 1) or (block.shape[-1] < 1):
        raise ValueError('External dimensions of `block` should be positive')
    ranks = None if rank is None else _normalize_rank(rank, len(in_dim) - 1)
    truncation = _TruncationSpec(
        cutoff=cutoff, atol=atol, rtol=rtol, cum_percentage=cum_percentage)
    if not isinstance(renormalize, bool):
        raise TypeError('`renormalize` should be bool type')
    if (metrics is not None) and not isinstance(metrics, DecompositionMetrics):
        raise TypeError('`metrics` should be DecompositionMetrics or None')

    norm = block.norm() if metrics is not None else None
    offset = 0

    def record_cut(site: int, info: Any,
                   singular_values: torch.Tensor,
                   log_scale: torch.Tensor) -> None:
        metrics.truncations.append(TruncationRecord.from_svd_info(
            info, site=site + offset, log_scale=log_scale, global_norm=norm,
            singular_values=singular_values))

    options = truncation.as_kwargs()
    options.update(mode='right', renormalize=renormalize,
                   _svd_callback=record_cut if metrics is not None else None)

    # Uniform bounds need one split; profiles apply a different cap per cut.
    if (rank is None) or isinstance(rank, int):
        cores = split_block(block, in_dim, **{**options, 'rank': rank}).cores
    else:
        state = block
        cores = []
        for offset, dimension in enumerate(in_dim[:-1]):
            remaining = in_dim[offset + 1:]
            pair = state.reshape(
                state.shape[0], dimension, prod(remaining), block.shape[-1])
            split = split_block(
                pair, (dimension, prod(remaining)),
                **{**options, 'rank': ranks[offset]})
            cores.append(split.cores[0])
            state = split.cores[1].reshape(
                split.cores[1].shape[0], *remaining, block.shape[-1])
        cores.append(state.reshape(
            state.shape[0], in_dim[-1], block.shape[-1]))
    return tuple(cores)


def _numerical_rank(matrix: torch.Tensor, rtol: float) -> int:
    """Counts retained singular directions using standard SVD tail energy."""
    return len(truncated_svd(matrix, rtol=rtol)[1])


def _boundary_partition(provider: Any, sites: Sequence[int]
                        ) -> Tuple[Tuple[int, ...], ...]:
    """Selects end blocks and merges intersections with the central block."""
    cores, dimensions = provider.cores, provider.in_dim
    count = len(cores)

    # Find the smallest end blocks not saturated by their outgoing TT bond.
    left_end = 0
    while (left_end < count - 1) and (prod(dimensions[:left_end + 1]) <=
                                    cores[left_end].shape[-1]):
        left_end += 1
    right_start = count - 1
    while (right_start > 0) and (prod(dimensions[right_start:]) <=
                               cores[right_start].shape[0]):
        right_start -= 1

    # Merge intersections before construction so no site is resolved twice.
    left, right = sites[0], sites[-1]
    left_sites = tuple(range(left_end + 1))
    right_sites = tuple(range(right_start, count))
    if left <= left_end:
        left = 0
        left_sites = ()
    if right >= right_start:
        right = count - 1
        right_sites = ()
    if left_end >= right_start:
        left, right = 0, count - 1
        left_sites = right_sites = ()
    return tuple(range(left, right + 1)), left_sites, right_sites


def _discovery_partition(provider: Any, center: int,
                         rtol: float) -> Tuple[Any, ...]:
    """Discovers central ranks and partitions the chain into three blocks."""
    cores, dimensions = provider.cores, provider.in_dim
    count = len(cores)
    left = right = center
    while True:
        sites = tuple(range(left, right + 1))
        target = provider.local_target(sites, {})
        physical = prod(dimensions[left:right + 1])
        q = _numerical_rank(target.movedim(0, -2).reshape(physical, -1), rtol)
        if physical > q:
            break
        if (left == 0) and (right == count - 1):
            break
        left = max(0, left - 1)
        right = min(count - 1, right + 1)

    # Measure the two external supports and the physical-to-virtual rank.
    left_dim = _numerical_rank(target.reshape(target.shape[0], -1), rtol)
    right_dim = _numerical_rank(target.reshape(-1, target.shape[-1]), rtol)
    cyclic = max(1, round(sqrt(left_dim * right_dim / q)))
    ranks = [max(1, (core.shape[-1] + cyclic - 1) // cyclic)
             for core in cores[:-1]] + [cyclic]
    sites, left_sites, right_sites = _boundary_partition(
        provider, tuple(range(left, right + 1)))
    return (sites, left_sites, right_sites, tuple(ranks), {
        'left_dim': left_dim, 'right_dim': right_dim,
        'physical_rank': q, 'cyclic_rank': cyclic})
