"""
This script contains:

    Classes:
        * AlternatingRingDriver

    Functions:
        * _block_selection
        * _gauge_stability
"""

from dataclasses import replace
from math import prod
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
import warnings

import torch

from tensorkrowch.decompositions.metrics import DecompositionMetrics
from tensorkrowch.decompositions.ring.blocks import (BlockSelection,
                                                     CentralBlockSelector)
from tensorkrowch.decompositions.ring.driver import (BidirectionalRingDriver,
                                                     BidirectionalRingResult,
                                                     RingTargetProvider,
                                                     _normalize_context,
                                                     _validate_provider)
from tensorkrowch.decompositions.ring.gauges import (ExperimentalWarning,
                                                     GaugeMap,
                                                     GaugeRecursion)
from tensorkrowch.decompositions.ring.opening import (FixedGaugeCoreOpener,
                                                      LoopOpener,
                                                      LoopOpening)


def _block_selection(provider: RingTargetProvider,
                     rank: Union[int, Sequence[int]],
                     sites: Tuple[int, ...],
                     context: Mapping[str, Any]) -> BlockSelection:
    """Builds the result descriptor for the first independent anchor."""
    local_rank = tuple(provider.local_rank(sites, rank, context))
    in_dim = tuple(provider.in_dim[site] for site in sites)
    return BlockSelection(
        sites=sites,
        in_dim=in_dim,
        left_rank_cap=local_rank[0],
        right_rank_cap=local_rank[-2],
        input_capacity=prod(in_dim),
        required_input_capacity=1,
        feasible=True,
        reason='alternating_anchor',
        boundary=None,
        growth=((sites[0], sites[-1]),))


def _gauge_stability(opening: Any, sites: Tuple[int, ...]) -> Dict[str, Any]:
    """Returns rank-revealing SVD diagnostics for both outgoing gauges."""
    diagnostics = []
    for orientation, gauge in (
            ('left', opening.left_gauge),
            ('right', opening.right_gauge)):
        if gauge is None:
            continue
        matrix = GaugeMap(gauge, orientation=orientation).matrix
        singular_values = torch.linalg.svdvals(matrix)
        tolerance = max(matrix.shape) * torch.finfo(
            singular_values.dtype).eps * singular_values[0]
        numerical_rank = int(
            (singular_values > tolerance).sum().detach().cpu().item())
        smallest = singular_values[-1]
        condition = torch.where(
            smallest > tolerance,
            singular_values[0] / smallest,
            singular_values.new_tensor(torch.inf))
        diagnostics.append({
            'sites': sites,
            'orientation': orientation,
            'shape': tuple(matrix.shape),
            'numerical_rank': numerical_rank,
            'singular_values': singular_values.detach().cpu(),
            'condition_number': float(condition.detach().cpu().item()),
        })
    return tuple(diagnostics)


class AlternatingRingDriver:  # MARK: AlternatingRingDriver
    """
    Runs a serial anchor/fixed-block schedule for a tensor ring.

    Independent anchor blocks are opened first with two free gauges. Every
    intervening block is then solved with the gauges recursively propagated
    from its two neighboring anchors. The implementation is deliberately
    serial: it validates the task dependency pattern before phase 4 adds an
    execution backend.

    The exact checkerboard requires an even cyclic ring, or an odd number of
    sites for a provider with open target boundaries. Unsupported layouts and
    incompatible gauges fall back to :class:`BidirectionalRingDriver` when
    ``fallback=True``.
    """

    def fit(self,
            provider: RingTargetProvider,
            rank: Union[int, Sequence[int]],
            opener: LoopOpener,
            recursion: GaugeRecursion,
            *,
            fixed_opener: Optional[LoopOpener] = None,
            block_size: int = 1,
            anchor_offset: int = 0,
            stability_diagnostics: bool = True,
            fallback: bool = True,
            fallback_driver: Optional[BidirectionalRingDriver] = None,
            block_selector: Optional[CentralBlockSelector] = None,
            center: Optional[int] = None,
            context: Optional[Mapping[str, Any]] = None) -> BidirectionalRingResult:
        """Executes the experimental alternating schedule serially."""
        in_dim = _validate_provider(provider)
        if not isinstance(opener, LoopOpener):
            raise TypeError('`opener` should implement LoopOpener')
        if not isinstance(recursion, GaugeRecursion):
            raise TypeError('`recursion` should implement GaugeRecursion')
        if fixed_opener is None:
            fixed_opener = FixedGaugeCoreOpener()
        elif not isinstance(fixed_opener, LoopOpener):
            raise TypeError('`fixed_opener` should implement LoopOpener')
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise TypeError('`block_size` should be int type')
        if block_size < 1:
            raise ValueError('`block_size` should be positive')
        if isinstance(anchor_offset, bool) or not isinstance(anchor_offset, int):
            raise TypeError('`anchor_offset` should be int type')
        if not isinstance(stability_diagnostics, bool):
            raise TypeError('`stability_diagnostics` should be bool type')
        if not isinstance(fallback, bool):
            raise TypeError('`fallback` should be bool type')
        if fallback_driver is None:
            fallback_driver = BidirectionalRingDriver()
        elif not isinstance(fallback_driver, BidirectionalRingDriver):
            raise TypeError(
                '`fallback_driver` should be BidirectionalRingDriver type')
        context = _normalize_context(context)
        boundary_mode = getattr(provider, 'boundary_mode', 'cyclic')
        if boundary_mode not in ('cyclic', 'open'):
            raise ValueError(
                "`provider.boundary_mode` should be 'cyclic' or 'open'")

        warnings.warn(
            'AlternatingRingDriver is experimental and may fall back to the '
            'center-out schedule.',
            ExperimentalWarning,
            stacklevel=2)

        unsupported = self._unsupported_reason(
            len(in_dim), boundary_mode, block_size, anchor_offset)
        if unsupported is not None:
            return self._fallback(
                unsupported,
                fallback,
                fallback_driver,
                provider,
                rank,
                opener,
                recursion,
                block_selector,
                center,
                context)
        try:
            if boundary_mode == 'cyclic':
                return self._fit_cyclic(
                    provider, rank, opener, fixed_opener, recursion,
                    block_size, anchor_offset, stability_diagnostics, context)
            return self._fit_open(
                provider, rank, opener, fixed_opener, recursion,
                stability_diagnostics, context)
        except (RuntimeError, ValueError) as exc:
            return self._fallback(
                f'incompatible_alternating_gauges: {exc}',
                fallback,
                fallback_driver,
                provider,
                rank,
                opener,
                recursion,
                block_selector,
                center,
                context)

    @staticmethod
    def _unsupported_reason(n_sites: int,
                            boundary_mode: str,
                            block_size: int,
                            anchor_offset: int) -> Optional[str]:
        """Returns why an exact alternating partition cannot be formed."""
        if anchor_offset not in (0, 1):
            return 'anchor_offset should be zero or one'
        if boundary_mode == 'open':
            if block_size != 1:
                return 'open-boundary alternating blocks currently have size one'
            if n_sites % 2 == 0:
                return 'open-boundary checkerboard requires an odd site count'
            return None
        if n_sites % (2 * block_size):
            return 'cyclic sites should be divisible by twice block_size'
        return None

    @staticmethod
    def _fallback(reason: Any,
                  fallback: Any,
                  driver: Any,
                  provider: Any,
                  rank: Union[int, Sequence[int]],
                  opener: Any,
                  recursion: Any,
                  block_selector: Any,
                  center: Any,
                  context: Any) -> BidirectionalRingResult:
        """Runs ``center``-out with an explicit diagnostic, or raises."""
        if not fallback:
            raise ValueError(
                'Alternating ring schedule is unavailable: ' + reason)
        result = driver.fit(
            provider=provider,
            rank=rank,
            opener=opener,
            recursion=recursion,
            block_selector=block_selector,
            center=center,
            context=context)
        return replace(result, diagnostics={
            **result.diagnostics,
            'schedule': 'center_out',
            'requested_schedule': 'alternating',
            'fallback_reason': reason,
        })

    @staticmethod
    def _open_anchor(provider: Any,
                     rank: Union[int, Sequence[int]],
                     opener: Any,
                     sites: Sequence[int],
                     context: Any,
                     cores: Sequence[torch.Tensor],
                     openings: Any,
                     order: Sequence[int],
                     directions: Any,
                     metrics: Optional['DecompositionMetrics']) -> LoopOpening:
        """Opens and stores one independent anchor block."""
        target = provider.local_target(sites, context)
        opening = BidirectionalRingDriver._open(
            provider=provider,
            rank=rank,
            opener=opener,
            sites=sites,
            target=target,
            orientation='right',
            fixed_left=None,
            fixed_right=None,
            context=context)
        BidirectionalRingDriver._store_opening(
            sites, 'anchor', opening, cores, openings, order,
            directions, metrics)
        return opening

    @staticmethod
    def _open_fixed(provider: Any,
                    rank: Union[int, Sequence[int]],
                    opener: Any,
                    recursion: Any,
                    sites: Sequence[int],
                    left_sites: Any,
                    right_sites: Any,
                    context: Any,
                    cores: Sequence[torch.Tensor],
                    openings: Any,
                    order: Sequence[int],
                    directions: Any,
                    metrics: Optional['DecompositionMetrics'],
                    recursion_diagnostics: Any) -> None:
        """Solves one intervening block from its two anchor gauges."""
        target = provider.local_target(sites, context)
        fixed_left = BidirectionalRingDriver._advance(
            recursion=recursion,
            direction='right',
            opening=openings[left_sites],
            local_target=target,
            from_sites=left_sites,
            to_sites=sites,
            provider=provider,
            context=context,
            metrics=metrics,
            diagnostics=recursion_diagnostics)
        fixed_right = BidirectionalRingDriver._advance(
            recursion=recursion,
            direction='left',
            opening=openings[right_sites],
            local_target=target,
            from_sites=right_sites,
            to_sites=sites,
            provider=provider,
            context=context,
            metrics=metrics,
            diagnostics=recursion_diagnostics)
        opening = BidirectionalRingDriver._open(
            provider=provider,
            rank=rank,
            opener=opener,
            sites=sites,
            target=target,
            orientation='right',
            fixed_left=fixed_left,
            fixed_right=fixed_right,
            context=context)
        BidirectionalRingDriver._store_opening(
            sites, 'fixed', opening, cores, openings, order,
            directions, metrics)

    def _fit_cyclic(self,
                    provider: Any,
                    rank: Union[int, Sequence[int]],
                    opener: Any,
                    fixed_opener: Any,
                    recursion: Any,
                    block_size: Any,
                    anchor_offset: Any,
                    stability_diagnostics: Any,
                    context: Any) -> BidirectionalRingResult:
        """Runs alternating free/fixed blocks around a cyclic ``provider``."""
        n_sites = len(provider.in_dim)
        offset = anchor_offset * block_size
        ordered = tuple(range(offset, n_sites)) + tuple(range(offset))
        blocks = tuple(
            tuple(ordered[start:start + block_size])
            for start in range(0, n_sites, block_size))
        if any(block != tuple(range(block[0], block[-1] + 1))
               for block in blocks):
            raise ValueError('An anchor block cannot wrap across array order')
        anchors = blocks[::2]
        fixed_blocks = blocks[1::2]
        cores = [None] * n_sites
        openings = {}
        order = []
        directions = []
        metrics = DecompositionMetrics()
        recursion_diagnostics = []
        stability = []

        for sites in anchors:
            opening = self._open_anchor(
                provider, rank, opener, sites, context, cores, openings,
                order, directions, metrics)
            if stability_diagnostics:
                stability.extend(_gauge_stability(opening, sites))
        for index, sites in enumerate(fixed_blocks):
            self._open_fixed(
                provider, rank, fixed_opener, recursion, sites,
                anchors[index], anchors[(index + 1) % len(anchors)],
                context, cores, openings, order, directions, metrics,
                recursion_diagnostics)

        selection = _block_selection(provider, rank, anchors[0], context)
        return BidirectionalRingResult(
            cores=cores,
            central_block=selection,
            openings=openings,
            order=order,
            directions=directions,
            metrics=metrics,
            diagnostics={
                'schedule': 'alternating',
                'boundary_mode': 'cyclic',
                'anchor_blocks': anchors,
                'fixed_blocks': fixed_blocks,
                'gauge_stability': tuple(stability),
                'recursions': tuple(recursion_diagnostics),
            })

    def _fit_open(self,
                  provider: Any,
                  rank: Union[int, Sequence[int]],
                  opener: Any,
                  fixed_opener: Any,
                  recursion: Any,
                  stability_diagnostics: Any,
                  context: Any) -> BidirectionalRingResult:
        """Runs odd internal anchors and closes both open target boundaries."""
        n_sites = len(provider.in_dim)
        anchors = tuple((site,) for site in range(1, n_sites - 1, 2))
        fixed_blocks = tuple((site,) for site in range(2, n_sites - 1, 2))
        cores = [None] * n_sites
        openings = {}
        boundaries = {}
        order = []
        directions = []
        metrics = DecompositionMetrics()
        recursion_diagnostics = []
        stability = []

        for sites in anchors:
            opening = self._open_anchor(
                provider, rank, opener, sites, context, cores, openings,
                order, directions, metrics)
            if stability_diagnostics:
                stability.extend(_gauge_stability(opening, sites))
        for index, sites in enumerate(fixed_blocks):
            self._open_fixed(
                provider, rank, fixed_opener, recursion, sites,
                anchors[index], anchors[index + 1], context, cores, openings,
                order, directions, metrics, recursion_diagnostics)

        close_boundary = getattr(provider, 'close_boundary', None)
        if not callable(close_boundary):
            raise TypeError(
                'An open-boundary provider should implement `close_boundary`')
        for site, direction, anchor in (
                (n_sites - 1, 'right', anchors[-1]),
                (0, 'left', anchors[0])):
            opening = openings[anchor]
            closure = close_boundary(
                site=site,
                direction=direction,
                opening=opening,
                context=BidirectionalRingDriver._boundary_context(
                    recursion=recursion,
                    direction=direction,
                    opening=opening,
                    boundary_site=site,
                    provider=provider,
                    context=context,
                    openings=openings,
                    metrics=metrics,
                    diagnostics=recursion_diagnostics))
            BidirectionalRingDriver._store_boundary(
                closure, cores, boundaries, order, directions, metrics,
                observer=context.get('_observer'))

        selection = _block_selection(provider, rank, anchors[0], context)
        return BidirectionalRingResult(
            cores=cores,
            central_block=selection,
            openings=openings,
            boundaries=boundaries,
            order=order,
            directions=directions,
            metrics=metrics,
            diagnostics={
                'schedule': 'alternating',
                'boundary_mode': 'open',
                'anchor_blocks': anchors,
                'fixed_blocks': fixed_blocks,
                'gauge_stability': tuple(stability),
                'recursions': tuple(recursion_diagnostics),
            })


__all__ = [
    'AlternatingRingDriver',
]
