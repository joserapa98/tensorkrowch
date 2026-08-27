"""Bidirectional orchestration shared by tensor-ring constructions."""

from dataclasses import dataclass, field
from typing import (Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple,
                    runtime_checkable)

import torch

from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 GaugeRecord)
from tensorkrowch.decompositions.results import TRDecomposition
from tensorkrowch.decompositions.ring.blocks import (BlockSelection,
                                                     CentralBlockSelector)
from tensorkrowch.decompositions.ring.gauges import (GaugeRecursion,
                                                     GaugeRecursionStep)
from tensorkrowch.decompositions.ring.opening import (FixedGaugeCoreOpener,
                                                      LoopOpener,
                                                      LoopOpening)


@runtime_checkable
class RingTargetProvider(Protocol):
    """Provides local targets, ranks and contexts to the ring driver."""

    @property
    def input_dim(self) -> Sequence[int]:
        """Returns one input dimension per final TR site."""

    def local_target(self,
                     sites: Sequence[int],
                     context: Mapping[str, Any]) -> Any:
        """Returns the local object to open for contiguous ``sites``."""

    def local_rank(self,
                   sites: Sequence[int],
                   rank,
                   context: Mapping[str, Any]):
        """Returns right-link ranks for gauges and local sites."""

    def local_context(self,
                      sites: Sequence[int],
                      context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Returns opener-specific information for one local target."""


@dataclass(frozen=True)
class BoundaryClosure:
    """Stores one final core obtained by absorbing an open target boundary."""

    site: int
    direction: str
    core: torch.Tensor
    records: Sequence[GaugeRecord] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.site, bool) or not isinstance(self.site, int):
            raise TypeError('`site` should be int type')
        if self.site < 0:
            raise ValueError('`site` should be non-negative')
        if self.direction not in ('left', 'right'):
            raise ValueError("`direction` should be 'left' or 'right'")
        if not isinstance(self.core, torch.Tensor):
            raise TypeError('`core` should be torch.Tensor type')
        if self.core.ndim != 3:
            raise ValueError('A boundary core should be three-dimensional')
        records = tuple(self.records)
        if not all(isinstance(record, GaugeRecord) for record in records):
            raise TypeError('`records` should contain GaugeRecord objects')
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError('`diagnostics` should be a mapping')
        object.__setattr__(self, 'records', records)
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))


@dataclass(frozen=True)
class BidirectionalRingResult:
    """Stores ordered cores and structural diagnostics from a driver run."""

    cores: Sequence[torch.Tensor]
    central_block: BlockSelection
    openings: Mapping[Tuple[int, ...], LoopOpening]
    order: Sequence[Tuple[int, ...]]
    directions: Sequence[str]
    boundaries: Mapping[int, BoundaryClosure] = field(default_factory=dict)
    metrics: DecompositionMetrics = field(default_factory=DecompositionMetrics)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cores = tuple(self.cores)
        openings = dict(self.openings)
        boundaries = dict(self.boundaries)
        order = tuple(tuple(sites) for sites in self.order)
        directions = tuple(self.directions)
        if not isinstance(self.central_block, BlockSelection):
            raise TypeError('`central_block` should be BlockSelection type')
        if not all(isinstance(opening, LoopOpening)
                   for opening in openings.values()):
            raise TypeError('`openings` should contain LoopOpening objects')
        if not all(isinstance(site, int) and
                   isinstance(closure, BoundaryClosure) and
                   closure.site == site
                   for site, closure in boundaries.items()):
            raise TypeError(
                '`boundaries` should map sites to matching BoundaryClosure '
                'objects')
        opening_keys = set(openings)
        boundary_keys = {(site,) for site in boundaries}
        if opening_keys.intersection(boundary_keys):
            raise ValueError('A site cannot be both opened and boundary-closed')
        step_keys = opening_keys.union(boundary_keys)
        if len(step_keys) != len(order) or step_keys != set(order):
            raise ValueError('`order` should identify every stored opening')
        opened_sites = tuple(site for sites in order for site in sites)
        if sorted(opened_sites) != list(range(len(cores))):
            raise ValueError(
                '`order` should cover every assembled site exactly once')
        if any(len(opening.cores) != len(sites)
               for sites, opening in openings.items()):
            raise ValueError(
                'Every opening should contain one core per identified site')
        if len(directions) != len(order):
            raise ValueError('`directions` should align with `order`')
        if not all(direction in (
                'center', 'left', 'right', 'boundary',
                'left_boundary', 'right_boundary')
                   for direction in directions):
            raise ValueError('`directions` contains an unknown driver step')
        if not isinstance(self.metrics, DecompositionMetrics):
            raise TypeError('`metrics` should be DecompositionMetrics type')
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError('`diagnostics` should be a mapping')
        decomposition = TRDecomposition(cores, metrics=self.metrics)
        object.__setattr__(self, 'cores', tuple(decomposition.cores))
        object.__setattr__(self, 'openings', openings)
        object.__setattr__(self, 'boundaries', boundaries)
        object.__setattr__(self, 'order', order)
        object.__setattr__(self, 'directions', directions)
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))

    @property
    def rank(self) -> Tuple[int, ...]:
        """Returns one actual right-link rank per assembled core."""
        return tuple(core.shape[-1] for core in self.cores)

    def as_decomposition(self) -> TRDecomposition:
        """Returns the assembled lightweight TR decomposition."""
        return TRDecomposition(
            self.cores,
            metrics=self.metrics,
            metadata={
                'algorithm': 'bidirectional_ring_driver',
                'central_block': self.central_block.sites,
                'order': self.order,
                'boundaries': tuple(self.boundaries),
                **self.diagnostics,
            })

    def contract_dense(self) -> torch.Tensor:
        """Contracts the assembled ring without constructing a model."""
        return self.as_decomposition().contract_dense()


def _normalize_context(
        context: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Copies the shared driver context after validation."""
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise TypeError('`context` should be a mapping or None')
    return dict(context)


def _validate_provider(provider: RingTargetProvider) -> Tuple[int, ...]:
    """Validates the intentionally small provider protocol."""
    if not isinstance(provider, RingTargetProvider):
        raise TypeError('`provider` should implement RingTargetProvider')
    try:
        input_dim = tuple(provider.input_dim)
    except TypeError as exc:
        raise TypeError('`provider.input_dim` should be a sequence') from exc
    if len(input_dim) < 3:
        raise ValueError('Bidirectional ring construction requires three sites')
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
           for dim in input_dim):
        raise ValueError('Provider input dimensions should be positive integers')
    return input_dim


def _validate_opening(opening: LoopOpening,
                      sites: Tuple[int, ...],
                      orientation: str,
                      fixed_left: Optional[torch.Tensor],
                      fixed_right: Optional[torch.Tensor]) -> None:
    """Checks a strategy result before it enters the global assembly."""
    if not isinstance(opening, LoopOpening):
        raise TypeError('The loop opener should return a LoopOpening')
    if opening.orientation != orientation:
        raise ValueError('The loop opener changed the requested orientation')
    if len(opening.cores) != len(sites):
        raise ValueError(
            'The loop opener returned a different number of local cores')
    if fixed_left is not None and \
            (opening.left_gauge is None or
             not torch.equal(opening.left_gauge, fixed_left)):
        raise ValueError('The loop opener changed the fixed left gauge')
    if fixed_right is not None and \
            (opening.right_gauge is None or
             not torch.equal(opening.right_gauge, fixed_right)):
        raise ValueError('The loop opener changed the fixed right gauge')


class BidirectionalRingDriver:
    """Builds a ring from a center block and two alternating recursions.

    The unprocessed complement of the central block is one cyclic interval.
    Sites are opened alternately from its right and left ends with one fixed
    gauge. Its final site is opened with both gauges, explicitly reconciling
    the two fronts instead of leaving an unchecked cyclic interface.
    """

    def fit(self,
            provider: RingTargetProvider,
            rank,
            opener: LoopOpener,
            recursion: GaugeRecursion,
            block_selector: Optional[CentralBlockSelector] = None,
            *,
            center: Optional[int] = None,
            boundary_opener: Optional[LoopOpener] = None,
            context: Optional[
                Mapping[str, Any]] = None) -> BidirectionalRingResult:
        """Runs the isolated central/right/left/boundary driver workflow."""
        input_dim = _validate_provider(provider)
        if not isinstance(opener, LoopOpener):
            raise TypeError('`opener` should implement LoopOpener')
        if not isinstance(recursion, GaugeRecursion):
            raise TypeError('`recursion` should implement GaugeRecursion')
        if block_selector is None:
            block_selector = CentralBlockSelector()
        elif not isinstance(block_selector, CentralBlockSelector):
            raise TypeError(
                '`block_selector` should be CentralBlockSelector type')
        if boundary_opener is None:
            if opener.capabilities.supports_two_fixed_gauges:
                boundary_opener = opener
            else:
                boundary_opener = FixedGaugeCoreOpener()
        elif not isinstance(boundary_opener, LoopOpener):
            raise TypeError('`boundary_opener` should implement LoopOpener')
        context = _normalize_context(context)
        bounds = context.get('block_bounds')
        boundary_mode = getattr(provider, 'boundary_mode', 'cyclic')
        if boundary_mode not in ('cyclic', 'open'):
            raise ValueError(
                "`provider.boundary_mode` should be 'cyclic' or 'open'")

        selection = block_selector.select(
            provider, rank, center=center, bounds=bounds)
        if not selection.feasible:
            raise ValueError(
                'Could not select a refinable central block: '
                f'reason={selection.reason}, sites={selection.sites}, '
                f'input_capacity={selection.input_capacity}, '
                f'required_input_capacity={selection.required_input_capacity}')
        if len(selection.sites) == len(input_dim):
            raise ValueError(
                'The central block should leave at least one boundary site '
                'to reconcile its two outgoing gauges')

        cores: list = [None] * len(input_dim)
        openings: Dict[Tuple[int, ...], LoopOpening] = {}
        order = []
        directions = []
        metrics = DecompositionMetrics()
        recursion_diagnostics = []

        central_sites = selection.sites
        central_target = provider.local_target(central_sites, context)
        central_context = provider.local_context(central_sites, context)
        central_rank = provider.local_rank(central_sites, rank, context)
        central_opening = opener.open(
            central_target,
            central_rank,
            orientation='right',
            context=central_context)
        _validate_opening(
            central_opening, central_sites, 'right', None, None)
        self._store_opening(
            central_sites,
            'center',
            central_opening,
            cores,
            openings,
            order,
            directions,
            metrics)

        if boundary_mode == 'open':
            return self._fit_open_boundaries(
                provider=provider,
                rank=rank,
                opener=opener,
                recursion=recursion,
                selection=selection,
                central_opening=central_opening,
                input_dim=input_dim,
                context=context,
                cores=cores,
                openings=openings,
                order=order,
                directions=directions,
                metrics=metrics,
                recursion_diagnostics=recursion_diagnostics)

        remaining = len(input_dim) - len(central_sites)
        left_site = (selection.left - 1) % len(input_dim)
        right_site = (selection.right + 1) % len(input_dim)
        left_opening = central_opening
        right_opening = central_opening

        while remaining > 1:
            sites = (right_site,)
            target = provider.local_target(sites, context)
            fixed_left = self._advance(
                recursion=recursion,
                direction='right',
                opening=right_opening,
                local_target=target,
                from_sites=self._opening_sites(openings, right_opening),
                to_sites=sites,
                provider=provider,
                context=context,
                metrics=metrics,
                diagnostics=recursion_diagnostics)
            opening = self._open(
                provider=provider,
                rank=rank,
                opener=opener,
                sites=sites,
                target=target,
                orientation='right',
                fixed_left=fixed_left,
                fixed_right=None,
                context=context)
            self._store_opening(
                sites, 'right', opening, cores, openings, order,
                directions, metrics)
            right_opening = opening
            right_site = (right_site + 1) % len(input_dim)
            remaining -= 1
            if remaining <= 1:
                break

            sites = (left_site,)
            target = provider.local_target(sites, context)
            fixed_right = self._advance(
                recursion=recursion,
                direction='left',
                opening=left_opening,
                local_target=target,
                from_sites=self._opening_sites(openings, left_opening),
                to_sites=sites,
                provider=provider,
                context=context,
                metrics=metrics,
                diagnostics=recursion_diagnostics)
            opening = self._open(
                provider=provider,
                rank=rank,
                opener=opener,
                sites=sites,
                target=target,
                orientation='left',
                fixed_left=None,
                fixed_right=fixed_right,
                context=context)
            self._store_opening(
                sites, 'left', opening, cores, openings, order,
                directions, metrics)
            left_opening = opening
            left_site = (left_site - 1) % len(input_dim)
            remaining -= 1

        if remaining == 1:
            if left_site != right_site:
                raise RuntimeError(
                    'The two ring sweeps did not reach the same boundary site')
            sites = (right_site,)
            target = provider.local_target(sites, context)
            fixed_left = self._advance(
                recursion=recursion,
                direction='right',
                opening=right_opening,
                local_target=target,
                from_sites=self._opening_sites(openings, right_opening),
                to_sites=sites,
                provider=provider,
                context=context,
                metrics=metrics,
                diagnostics=recursion_diagnostics)
            fixed_right = self._advance(
                recursion=recursion,
                direction='left',
                opening=left_opening,
                local_target=target,
                from_sites=self._opening_sites(openings, left_opening),
                to_sites=sites,
                provider=provider,
                context=context,
                metrics=metrics,
                diagnostics=recursion_diagnostics)
            opening = self._open(
                provider=provider,
                rank=rank,
                opener=boundary_opener,
                sites=sites,
                target=target,
                orientation='right',
                fixed_left=fixed_left,
                fixed_right=fixed_right,
                context=context)
            self._store_opening(
                sites, 'boundary', opening, cores, openings, order,
                directions, metrics)

        if any(core is None for core in cores):
            raise RuntimeError('The ring driver did not assemble every site')
        return BidirectionalRingResult(
            cores=cores,
            central_block=selection,
            openings=openings,
            order=order,
            directions=directions,
            metrics=metrics,
            diagnostics={
                'recursions': tuple(recursion_diagnostics),
            })

    def _fit_open_boundaries(
            self,
            provider: RingTargetProvider,
            rank,
            opener: LoopOpener,
            recursion: GaugeRecursion,
            selection: BlockSelection,
            central_opening: LoopOpening,
            input_dim: Tuple[int, ...],
            context: Mapping[str, Any],
            cores: list,
            openings: Dict[Tuple[int, ...], LoopOpening],
            order: list,
            directions: list,
            metrics: DecompositionMetrics,
            recursion_diagnostics: list) -> BidirectionalRingResult:
        """Runs two independent sweeps and absorbs both open target edges."""
        if selection.left == 0 or selection.right == len(input_dim) - 1:
            raise ValueError(
                'An open-boundary provider requires an internal central block')
        close_boundary = getattr(provider, 'close_boundary', None)
        if not callable(close_boundary):
            raise TypeError(
                'An open-boundary provider should implement `close_boundary`')

        boundaries = {}
        right_opening = central_opening
        for site in range(selection.right + 1, len(input_dim) - 1):
            sites = (site,)
            target = provider.local_target(sites, context)
            fixed_left = self._advance(
                recursion=recursion,
                direction='right',
                opening=right_opening,
                local_target=target,
                from_sites=self._opening_sites(openings, right_opening),
                to_sites=sites,
                provider=provider,
                context=context,
                metrics=metrics,
                diagnostics=recursion_diagnostics)
            opening = self._open(
                provider=provider,
                rank=rank,
                opener=opener,
                sites=sites,
                target=target,
                orientation='right',
                fixed_left=fixed_left,
                fixed_right=None,
                context=context)
            self._store_opening(
                sites, 'right', opening, cores, openings, order,
                directions, metrics)
            right_opening = opening
        right_closure = close_boundary(
            site=len(input_dim) - 1,
            direction='right',
            opening=right_opening,
            context=context)
        self._store_boundary(
            right_closure, cores, boundaries, order, directions, metrics)

        left_opening = central_opening
        for site in range(selection.left - 1, 0, -1):
            sites = (site,)
            target = provider.local_target(sites, context)
            fixed_right = self._advance(
                recursion=recursion,
                direction='left',
                opening=left_opening,
                local_target=target,
                from_sites=self._opening_sites(openings, left_opening),
                to_sites=sites,
                provider=provider,
                context=context,
                metrics=metrics,
                diagnostics=recursion_diagnostics)
            opening = self._open(
                provider=provider,
                rank=rank,
                opener=opener,
                sites=sites,
                target=target,
                orientation='left',
                fixed_left=None,
                fixed_right=fixed_right,
                context=context)
            self._store_opening(
                sites, 'left', opening, cores, openings, order,
                directions, metrics)
            left_opening = opening
        left_closure = close_boundary(
            site=0,
            direction='left',
            opening=left_opening,
            context=context)
        self._store_boundary(
            left_closure, cores, boundaries, order, directions, metrics)

        if any(core is None for core in cores):
            raise RuntimeError('The ring driver did not assemble every site')
        return BidirectionalRingResult(
            cores=cores,
            central_block=selection,
            openings=openings,
            boundaries=boundaries,
            order=order,
            directions=directions,
            metrics=metrics,
            diagnostics={
                'boundary_mode': 'open',
                'recursions': tuple(recursion_diagnostics),
            })

    @staticmethod
    def _opening_sites(
            openings: Mapping[Tuple[int, ...], LoopOpening],
            opening: LoopOpening) -> Tuple[int, ...]:
        """Finds the already stored site key for one opening identity."""
        for sites, candidate in openings.items():
            if candidate is opening:
                return sites
        raise RuntimeError('The recursion source opening was not stored')

    @staticmethod
    def _advance(
            recursion: GaugeRecursion,
            direction: str,
            opening: LoopOpening,
            local_target: Any,
            from_sites: Tuple[int, ...],
            to_sites: Tuple[int, ...],
            provider: RingTargetProvider,
            context: Mapping[str, Any],
            metrics: DecompositionMetrics,
            diagnostics: list) -> torch.Tensor:
        """Runs and validates one direction-specific gauge recursion."""
        recursion_context = dict(context)
        recursion_context.update({
            'direction': direction,
            'from_sites': from_sites,
            'to_sites': to_sites,
            'provider': provider,
        })
        if direction == 'right':
            step = recursion.advance_right(
                opening, local_target, recursion_context)
        else:
            step = recursion.advance_left(
                opening, local_target, recursion_context)
        if not isinstance(step, GaugeRecursionStep):
            raise TypeError(
                'Gauge recursion methods should return GaugeRecursionStep')
        metrics.gauges.extend(step.records)
        diagnostics.append({
            'direction': direction,
            'from_sites': from_sites,
            'to_sites': to_sites,
            **step.diagnostics,
        })
        return step.gauge

    @staticmethod
    def _open(
            provider: RingTargetProvider,
            rank,
            opener: LoopOpener,
            sites: Tuple[int, ...],
            target: Any,
            orientation: str,
            fixed_left: Optional[torch.Tensor],
            fixed_right: Optional[torch.Tensor],
            context: Mapping[str, Any]) -> LoopOpening:
        """Builds and validates one constrained local opening."""
        local_context = provider.local_context(sites, context)
        local_rank = provider.local_rank(sites, rank, context)
        opening = opener.open(
            target,
            local_rank,
            fixed_left=fixed_left,
            fixed_right=fixed_right,
            orientation=orientation,
            context=local_context)
        _validate_opening(
            opening, sites, orientation, fixed_left, fixed_right)
        return opening

    @staticmethod
    def _store_opening(
            sites: Tuple[int, ...],
            direction: str,
            opening: LoopOpening,
            cores: list,
            openings: Dict[Tuple[int, ...], LoopOpening],
            order: list,
            directions: list,
            metrics: DecompositionMetrics) -> None:
        """Stores cores and records after checking sites are still empty."""
        if sites in openings or any(cores[site] is not None for site in sites):
            raise RuntimeError('A ring site was opened more than once')
        for site, core in zip(sites, opening.cores):
            cores[site] = core
        openings[sites] = opening
        order.append(sites)
        directions.append(direction)
        metrics.local_solves.extend(opening.local_records)

    @staticmethod
    def _store_boundary(
            closure: BoundaryClosure,
            cores: list,
            boundaries: Dict[int, BoundaryClosure],
            order: list,
            directions: list,
            metrics: DecompositionMetrics) -> None:
        """Stores one provider-specific open-boundary absorption."""
        if not isinstance(closure, BoundaryClosure):
            raise TypeError('`close_boundary` should return BoundaryClosure')
        if closure.site >= len(cores):
            raise ValueError('Boundary closure site lies outside the ring')
        if cores[closure.site] is not None or closure.site in boundaries:
            raise RuntimeError('A ring boundary was resolved more than once')
        cores[closure.site] = closure.core
        boundaries[closure.site] = closure
        order.append((closure.site,))
        directions.append(f'{closure.direction}_boundary')
        metrics.gauges.extend(closure.records)


__all__ = [
    'RingTargetProvider',
    'BoundaryClosure',
    'BidirectionalRingResult',
    'BidirectionalRingDriver',
]
