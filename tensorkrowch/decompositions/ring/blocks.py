"""
This script contains:

    Functions:
        * _select_block
        * _spectral_block
        * _split_block
        * _numerical_rank
        * _boundary_partition
        * _discovery_partition
        * _ring_partition
"""

from math import prod, sqrt
from typing import Any, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.formats.formats1d import split_block
from tensorkrowch.utils import truncated_svd

from tensorkrowch.decompositions._utils import (_normalize_dims,
                                                _normalize_rank)
from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 TruncationRecord)


_Rank = Union[int, Sequence[int]]


def _spectral_block(tt_rank: Sequence[int], center: int) -> Tuple[int, ...]:
    """
    Selects the shortest interior block with equal external TT dimensions.

    ``tt_rank`` contains the internal bonds of an open chain. The block
    contains ``center``; ties favor a balanced interval, then its leftmost
    position. Without a matching interior pair, opens the complete chain.
    """
    count = len(tt_rank) + 1
    left_cuts = {}
    for cut in range(1, center + 1):
        left_cuts[tt_rank[cut - 1]] = cut

    # For each right cut, the nearest matching left cut gives the shortest
    # interval. Each cut is visited once; no physical cores are contracted.
    left, right = 0, count
    score = (count, count, 0)
    for cut in range(center + 1, count):
        start = left_cuts.get(tt_rank[cut - 1])
        if start is None:
            continue
        candidate = (cut - start, abs(2 * center - start - cut + 1), start)
        if candidate < score:
            left, right, score = start, cut, candidate
    return tuple(range(left, right))


def _select_block(
        provider: Any,
        rank: _Rank,
        center: Optional[int] = None,
        *,
        bounds: Optional[Tuple[int, int]] = None) -> dict:
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
    dict
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

    # Grow until the physical capacity exceeds the external rank product.
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

    return dict(
        sites=tuple(range(left, right + 1)),
        in_dim=in_dim[left:(right + 1)],
        left_rank_cap=left_rank_cap,
        right_rank_cap=right_rank_cap,
        input_capacity=input_capacity,
        required_input_capacity=required_input_capacity,
        feasible=feasible,
        reason=reason,
        boundary=boundary,
        growth=tuple(growth))


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
    truncation = _TruncationSpec(
        rank=rank, cutoff=cutoff, atol=atol, rtol=rtol,
        cum_percentage=cum_percentage)
    truncation.validate_rank(len(in_dim) - 1)
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

    options = dict(mode='right', renormalize=renormalize,
                   _svd_callback=record_cut if metrics is not None else None)

    # Uniform bounds need one split; profiles apply a different cap per cut.
    if not isinstance(truncation.rank, tuple):
        cores = split_block(block, in_dim,
                            **truncation.as_kwargs(), **options).cores
    else:
        state = block
        cores = []
        for offset, dimension in enumerate(in_dim[:-1]):
            remaining = in_dim[offset + 1:]
            pair = state.reshape(
                state.shape[0], dimension, prod(remaining), block.shape[-1])
            split = split_block(
                pair, (dimension, prod(remaining)),
                **truncation.as_kwargs(site=offset), **options)
            cores.append(split.cores[0])
            state = split.cores[1].reshape(
                split.cores[1].shape[0], *remaining, block.shape[-1])
        cores.append(state.reshape(
            state.shape[0], in_dim[-1], block.shape[-1]))
    return tuple(cores)


def _numerical_rank(matrix: torch.Tensor, rtol: float) -> int:
    """Counts retained singular directions using standard SVD tail energy."""
    return len(truncated_svd(matrix, rtol=rtol)[1])


def _boundary_partition(provider: Any, sites: Sequence[int], *,
                        discover: bool = False
                        ) -> Tuple[Tuple[int, ...], ...]:
    """Selects end blocks, rejecting discovery overlaps or merging them."""
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
    if discover:
        if (left <= left_end) and (right >= right_start):
            raise ValueError(
                'The chain is too small to discover ranks with three '
                'separate blocks; provide `rank` or use ALS directly')
        if (left <= left_end) or (right >= right_start):
            raise ValueError(
                'Cannot discover ranks because the central block overlaps '
                'a boundary block; choose a more interior `center` or '
                'provide `rank`')
        return tuple(sites), left_sites, right_sites
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
    """
    Selects a geometric block, then estimates its ranks from SVD supports.

    Selection uses the external TT bond dimensions. All three blocks must be
    disjoint before any SVD is computed. Full physical rank does not grow the
    block. The resulting profile is used by the ordinary prescribed-rank
    openings, including links inside the central block.
    """
    cores, dimensions = provider.cores, provider.in_dim
    if center in (0, len(cores) - 1):
        raise ValueError('Cannot discover ranks from an endpoint `center`; '
                         'choose an interior site or provide `rank`')
    tt_rank = tuple(core.shape[-1] for core in cores[:-1])
    sites = _select_block(provider, (*tt_rank, 1), center)['sites']
    sites, left_sites, right_sites = _boundary_partition(
        provider, sites, discover=True)
    target = provider.local_target(sites, {})
    physical = prod(dimensions[site] for site in sites)

    # Measure the two external supports and the physical-to-virtual rank.
    q = _numerical_rank(target.movedim(0, -2).reshape(physical, -1), rtol)
    left_dim = _numerical_rank(target.reshape(target.shape[0], -1), rtol)
    right_dim = _numerical_rank(target.reshape(-1, target.shape[-1]), rtol)
    cyclic = max(1, round(sqrt(left_dim * right_dim / q)))
    ranks = [max(1, (value + cyclic - 1) // cyclic)
             for value in tt_rank] + [cyclic]
    return (sites, left_sites, right_sites, tuple(ranks), {
        'left_dim': left_dim, 'right_dim': right_dim,
        'physical_rank': q, 'cyclic_rank': cyclic})


def _ring_partition(provider: Any, rank: Optional[_Rank], center: int,
                    block_boundaries: bool, rtol: Optional[float]
                    ) -> Tuple[Any, ...]:
    """Selects shared TT2TR/TR-RSS ranks, center and boundary intervals."""
    count = len(provider.in_dim)
    if rank is None:
        precision = torch.finfo(provider.cores[0].real.dtype).eps
        size = max(core.numel() for core in provider.cores)
        rtol = max(1e-12 if rtol is None else rtol,
                   (size * precision) ** 2)
        sites, left, right, ranks, discovery = _discovery_partition(
            provider, center, rtol)
    else:
        ranks = _normalize_rank(rank, count)
        bounds = (1, count - 2) if 0 < center < count - 1 else (center, center)
        sites = _select_block(provider, ranks, center, bounds=bounds)['sites']
        sites, left, right = (_boundary_partition(provider, sites)
                              if block_boundaries else (sites, (), ()))
        discovery = None
    if not block_boundaries:
        left = () if sites[0] == 0 else (0,)
        right = () if sites[-1] == count - 1 else (count - 1,)
    return sites, left, right, ranks, discovery, rtol
