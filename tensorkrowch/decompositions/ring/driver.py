"""
This script contains:

    Classes:
        * RingTargetProvider
        * BidirectionalRingDriver

    Functions:
        * _validate_provider
        * _validate_opening
"""

from typing import (Any,
                    Dict,
                    Mapping,
                    Optional,
                    Protocol,
                    Sequence,
                    Tuple,
                    Union,
                    runtime_checkable)

import torch

from tensorkrowch.decompositions._utils import _normalize_dims
from tensorkrowch.decompositions.metrics import DecompositionMetrics
from tensorkrowch.decompositions.observers import DecompositionEvent
from tensorkrowch.decompositions.results import TRDecomposition
from tensorkrowch.decompositions.ring.blocks import _select_block
from tensorkrowch.decompositions.ring.gauges import (GaugeRecursion,
                                                     GaugeMap)
from tensorkrowch.decompositions.ring.opening import (_FixedGaugeCoreOpener,
                                                      LoopOpener,
                                                      LoopOpening,
                                                      _normalize_context)


@runtime_checkable
class RingTargetProvider(Protocol):  # MARK: RingTargetProvider
    """Provides local targets, ranks and contexts to the ring driver."""

    @property
    def in_dim(self) -> Sequence[int]:
        """Returns one input dimension per final TR site."""

    def local_target(self,
                     sites: Sequence[int],
                     context: Mapping[str, Any]) -> Any:
        """Returns the local object to open for contiguous ``sites``."""

    def local_rank(self,
                   sites: Sequence[int],
                   rank: Union[int, Sequence[int]],
                   context: Mapping[str, Any]) -> Tuple[int, ...]:
        """Returns right-link ranks for gauges and local ``sites``."""

    def local_context(self,
                      sites: Sequence[int],
                      context: Mapping[str, Any]) -> Mapping[str, Any]:
        """Returns opener-specific information for one local target."""


def _validate_provider(provider: RingTargetProvider) -> Tuple[int, ...]:
    """Validates the intentionally small ``provider`` protocol."""
    if not isinstance(provider, RingTargetProvider):
        raise TypeError('`provider` should implement RingTargetProvider')
    in_dim = _normalize_dims(provider.in_dim, 'provider.in_dim')
    return in_dim


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
    if (fixed_left is not None) and \
            ((opening.left_gauge is None) or
             not torch.equal(opening.left_gauge, fixed_left)):
        raise ValueError('The loop opener changed the fixed left gauge')
    if (fixed_right is not None) and \
            ((opening.right_gauge is None) or
             not torch.equal(opening.right_gauge, fixed_right)):
        raise ValueError('The loop opener changed the fixed right gauge')


class BidirectionalRingDriver:  # MARK: BidirectionalRingDriver
    """
    Builds a ring from one central opening and outward gauge transports.

    Open target providers run an independent rightward branch followed by a
    leftward branch. Discovery resolves a rank profile before local openings;
    every ALS problem keeps separate physical cores. Prescribed end blocks
    use constrained openings; discovery end blocks absorb gauges before SVD
    splitting. Single-site ends always absorb their gauges. Only prescribed
    central and boundary blocks may merge, including the complete chain.

    The low-level cyclic-provider path alternates the two fronts and solves
    their final common site with two fixed gauges. TT2TR and TR-RSS use the
    open-provider path.
    """

    def _fit_open_boundaries(self,
                             provider: RingTargetProvider,
                             rank: Union[int, Sequence[int]],
                             opener: LoopOpener,
                             recursion: GaugeRecursion,
                             selection: Mapping[str, Any],
                             central_opening: LoopOpening,
                             in_dim: Tuple[int, ...],
                             context: Mapping[str, Any],
                             cores: list,
                             openings: Dict[Tuple[int, ...], LoopOpening],
                             order: list,
                             directions: list,
                             metrics: DecompositionMetrics,
                             recursion_diagnostics: list
                             ) -> TRDecomposition:
        """Runs two sweeps with constrained openings or boundary absorption."""
        close_boundary = getattr(provider, 'close_boundary', None)
        if not callable(close_boundary):
            raise TypeError('An open provider should implement '
                            '`close_boundary`')
        boundaries = {}
        left_sites = tuple(context.get('left_boundary', (0,)))
        right_sites = tuple(context.get('right_boundary', (len(in_dim) - 1,)))

        # Each outward branch starts independently from the same central
        # opening.
        for direction, boundary_sites in (('right', right_sites),
                                           ('left', left_sites)):
            if not boundary_sites:
                continue
            if direction == 'right':
                sites_to_open = range(selection['sites'][-1] + 1,
                                      boundary_sites[0])
            else:
                sites_to_open = range(selection['sites'][0] - 1,
                                      boundary_sites[-1], -1)
            # Prescribed end blocks use the same constrained ALS opening as
            # interior sites; discovery and one-site ends absorb the gauge.
            open_boundary = (self.rank_discovery is None) and \
                (len(boundary_sites) > 1)
            groups = [(site,) for site in sites_to_open]
            if open_boundary:
                groups.append(boundary_sites)
            previous = central_opening
            for sites in groups:
                adjacent = (sites[0] if direction == 'right' else sites[-1],)
                fixed = self._advance(
                    recursion, direction, previous,
                    provider.local_target(adjacent, context),
                    self._opening_sites(openings, previous), adjacent,
                    provider, context, metrics, recursion_diagnostics)
                opening = self._open(
                    provider, rank, opener, sites,
                    provider.local_target(sites, context), direction,
                    fixed if direction == 'right' else None,
                    fixed if direction == 'left' else None, context)
                if sites == boundary_sites:
                    finalize = getattr(provider, 'finalize_center', None)
                    if callable(finalize):
                        opening = finalize(opening, sites)
                    boundaries[sites[0]] = opening.cores
                step_direction = direction + '_boundary' \
                    if sites == boundary_sites else direction
                self._store_opening(
                    sites, step_direction, opening, cores, openings, order,
                    directions, metrics)
                previous = opening
            if open_boundary:
                continue
            adjacent = (boundary_sites[0] if direction == 'right'
                        else boundary_sites[-1])
            boundary_cores = close_boundary(
                sites=boundary_sites, direction=direction, opening=previous,
                context=self._boundary_context(
                    recursion, direction, previous, adjacent, provider,
                    context, openings, metrics, recursion_diagnostics))
            self._store_boundary(
                boundary_sites, direction, boundary_cores, cores, boundaries,
                order, directions, observer=context.get('_observer'))

        if any(core is None for core in cores):
            raise RuntimeError('The ring driver did not assemble every site')
        return self._finish(
            cores=cores,
            central_block=selection['sites'],
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
    def _opening_sites(openings: Mapping[Tuple[int, ...], LoopOpening],
                       opening: LoopOpening) -> Tuple[int, ...]:
        """Finds the already stored site key for one ``opening`` identity."""
        for sites, candidate in openings.items():
            if candidate is opening:
                return sites
        raise RuntimeError('The recursion source opening was not stored')

    @staticmethod
    def _record_gauge(step: GaugeMap, context: Mapping[str, Any],
                      metrics: DecompositionMetrics,
                      diagnostics: list) -> None:
        """Validates and records shared recursive and boundary gauge data."""
        if not isinstance(step, GaugeMap):
            raise TypeError('Gauge transport should return GaugeMap')
        observer = context.get('_observer')
        if observer is not None:
            for record in step.records:
                observer.emit(DecompositionEvent(
                    name='gauge', phase='TT to TR', level=2,
                    site=record.site,
                    values={'orientation': record.orientation,
                            'rank': (record.numerical_rank,
                                     record.cancellable_rank),
                            'condition_number': record.condition_number,
                            'cancellation_error': record.cancellation_error,
                            'projective': record.projective}))
        metrics.gauges.extend(step.records)
        diagnostics.append({
            'direction': context['direction'],
            'from_sites': context['from_sites'],
            'to_sites': context['to_sites'],
            **step.metadata,
        })

    @classmethod
    def _boundary_context(cls,
                          recursion: GaugeRecursion,
                          direction: str,
                          opening: LoopOpening,
                          boundary_site: int,
                          provider: RingTargetProvider,
                          context: Mapping[str, Any],
                          openings: Mapping[Tuple[int, ...], LoopOpening],
                          metrics: DecompositionMetrics,
                          diagnostics: list) -> Mapping[str, Any]:
        """
        Lets the provider or recursion prepare a boundary-absorption gauge.
        """
        boundary_context = dict(context)
        prepare_boundary = getattr(provider, 'prepare_boundary', None)
        if not callable(prepare_boundary):
            prepare_boundary = getattr(recursion, 'prepare_boundary', None)
        if not callable(prepare_boundary):
            return boundary_context

        sites = (boundary_site,)
        recursion_context = dict(context)
        recursion_context.update({
            'direction': direction,
            'from_sites': cls._opening_sites(openings, opening),
            'to_sites': sites,
            'provider': provider,
            'boundary': True,
        })
        step = prepare_boundary(
            opening,
            provider.local_target(sites, context),
            recursion_context,
            direction)
        cls._record_gauge(step, recursion_context, metrics, diagnostics)
        boundary_context['boundary_gauge'] = step.core
        return boundary_context

    @classmethod
    def _advance(cls, recursion: GaugeRecursion,
                 direction: str,
                 opening: LoopOpening,
                 local_target: Any,
                 from_sites: Tuple[int, ...],
                 to_sites: Tuple[int, ...],
                 provider: RingTargetProvider,
                 context: Mapping[str, Any],
                 metrics: DecompositionMetrics,
                 diagnostics: list) -> torch.Tensor:
        """
        Runs and validates one ``direction``-specific gauge ``recursion``.
        """
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
        cls._record_gauge(step, recursion_context, metrics, diagnostics)
        return step.core

    def _open(self, provider: RingTargetProvider,
              rank: Union[int, Sequence[int]],
              opener: LoopOpener,
              sites: Tuple[int, ...],
              target: Any,
              orientation: str,
              fixed_left: Optional[torch.Tensor],
              fixed_right: Optional[torch.Tensor],
              context: Mapping[str, Any]) -> LoopOpening:
        """Builds and validates one constrained local opening."""
        local_context = {**context, **provider.local_context(sites, context),
                         'sites': sites}
        local_rank = provider.local_rank(sites, rank, context)
        opening = opener.open(
            target, local_rank, fixed_left=fixed_left,
            fixed_right=fixed_right, orientation=orientation,
            context=local_context)
        _validate_opening(
            opening, sites, orientation, fixed_left, fixed_right)
        observer = context.get('_observer')
        if observer is not None:
            observer.emit(DecompositionEvent(
                name='site_complete', phase='TT to TR', site=sites[0],
                values={'sites': sites, 'direction': orientation,
                        'total_sites': len(provider.in_dim)}))
        return opening

    @staticmethod
    def _store_opening(sites: Tuple[int, ...],
                       direction: str,
                       opening: LoopOpening,
                       cores: list,
                       openings: Dict[Tuple[int, ...], LoopOpening],
                       order: list,
                       directions: list,
                       metrics: DecompositionMetrics) -> None:
        """
        Stores ``cores`` and records after checking ``sites`` are still empty.
        """
        if (sites in openings) or any(
                cores[site] is not None for site in sites):
            raise RuntimeError('A ring site was opened more than once')
        for site, core in zip(sites, opening.cores):
            cores[site] = core
        openings[sites] = opening
        order.append(sites)
        directions.append(direction)
        metrics.local_solves.extend(opening.local_records)
        metrics.truncations.extend(opening.diagnostics.get('truncations', ()))
        metrics.errors.extend(opening.diagnostics.get('errors', ()))
        metrics.warnings.extend(opening.diagnostics.get('warnings', ()))

    @staticmethod
    def _store_boundary(sites: Sequence[int], direction: str,
                        recovered: Sequence[torch.Tensor], cores: list,
                        boundaries: dict, order: list, directions: list,
                        observer: Any = None) -> None:
        """Stores directly absorbed end cores in their global positions."""
        sites, recovered = tuple(sites), tuple(recovered)
        if not sites or any(isinstance(site, bool) or
                            not isinstance(site, int) for site in sites):
            raise TypeError('Boundary sites should be integer indices')
        if (sites[0] < 0) or (sites[-1] >= len(cores)) or \
                (sites != tuple(range(sites[0], sites[-1] + 1))):
            raise ValueError('Boundary sites should be contiguous '
                             'and in bounds')
        if len(sites) != len(recovered):
            raise ValueError('A boundary should return one core per site')
        if any(not isinstance(core, torch.Tensor) or (core.ndim != 3)
               for core in recovered):
            raise ValueError('Boundary cores should have three axes')
        if any(cores[site] is not None for site in sites):
            raise ValueError('A site cannot be assembled twice')
        for site, core in zip(sites, recovered):
            cores[site] = core
        boundaries[sites[0]] = recovered
        order.append(sites)
        directions.append(direction + '_boundary')
        if observer is not None:
            observer.emit(DecompositionEvent(
                name='site_complete', phase='TT to TR', site=sites[0],
                values={'sites': sites, 'direction': direction,
                        'total_sites': len(cores)}))

    def _finish(self, cores: Sequence[torch.Tensor],
                central_block: Sequence[int],
                openings: Mapping, order: Sequence, directions: Sequence[str],
                metrics: DecompositionMetrics, diagnostics: Mapping,
                boundaries: Optional[Mapping] = None) -> TRDecomposition:
        """Retains construction state and creates the numerical result once."""
        # Retain construction state independently of the numerical result.
        self.cores = tuple(cores)
        self.central_block = tuple(central_block)
        self.openings = dict(openings)
        self.order = tuple(tuple(sites) for sites in order)
        self.directions = tuple(directions)
        self.boundaries = {} if boundaries is None else dict(boundaries)
        self.metrics = metrics
        self.diagnostics = dict(diagnostics)
        return TRDecomposition(self.cores, metrics=metrics, metadata={
            'algorithm': 'bidirectional_ring_driver',
            'central_block': self.central_block,
            'center_block': self.central_block, 'order': self.order,
            'boundaries': tuple(self.boundaries),
            **{key: value for key, value in diagnostics.items()
               if key != 'recursions'},
            'boundary_blocks': self.boundary_blocks,
            'rank_discovery': self.rank_discovery})

    def fit(self,
            provider: RingTargetProvider,
            rank: Union[int, Sequence[int]],
            opener: LoopOpener,
            recursion: GaugeRecursion,
            *,
            center: Optional[int] = None,
            block_boundaries: bool = True,
            rtol: Optional[float] = None,
            boundary_opener: Optional[LoopOpener] = None,
            context: Optional[Mapping[str, Any]] = None
            ) -> TRDecomposition:
        """
        Opens the center, transports both branches and assembles the TR.

        Parameters
        ----------
        provider : RingTargetProvider
            Supplies local targets, local ranks and opener metadata. An open
            provider also implements boundary-block closure.
        rank : int or sequence[int] or None
            Right-link ranks supplied to the provider for local openings.
            Open providers support ``None`` for rank discovery.
        opener : LoopOpener
            Solves the central block and subsequent one-site problems.
        recursion : GaugeRecursion
            Builds the fixed gauge inherited by each subsequent opening.
        center : int, optional
            Seed site for automatic central-block selection.
        block_boundaries : bool
            Whether an open provider may combine endpoint sites into blocks.
            Defaults to ``True``.
        rtol : float, optional
            Relative discarded squared singular-value mass for discovery and
            block splitting.
        boundary_opener : LoopOpener, optional
            Solver for the final two-fixed-gauge site of a cyclic provider.
            Open providers fit prescribed boundary blocks or close discovery
            blocks by gauge absorption instead.
        context : mapping, optional
            Provider execution data. ``left_boundary`` and ``right_boundary``
            contain the open-provider boundary sites; an empty tuple denotes
            a boundary already included in the central block.

        Returns
        -------
        TRDecomposition
            Assembled cores and diagnostics. Construction state remains on
            this driver for inspection.
        """
        in_dim = _validate_provider(provider)
        if not isinstance(opener, LoopOpener):
            raise TypeError('`opener` should implement LoopOpener')
        if not isinstance(recursion, GaugeRecursion):
            raise TypeError('`recursion` should implement GaugeRecursion')
        if boundary_opener is None:
            if opener.capabilities.supports_two_fixed_gauges:
                boundary_opener = opener
            else:
                boundary_opener = _FixedGaugeCoreOpener()
        elif not isinstance(boundary_opener, LoopOpener):
            raise TypeError('`boundary_opener` should implement LoopOpener')
        context = _normalize_context(context)
        bounds = context.get('block_bounds')
        boundary_mode = getattr(provider, 'boundary_mode', 'cyclic')
        if boundary_mode not in ('cyclic', 'open'):
            raise ValueError(
                "`provider.boundary_mode` should be 'cyclic' or 'open'")

        self.rank_discovery = None
        self.boundary_blocks = ((), ())
        if boundary_mode == 'open':
            if not callable(getattr(provider, 'close_boundary', None)):
                raise TypeError('An open provider should implement '
                                '`close_boundary`')
            sites, left, right, rank, self.rank_discovery, rtol = \
                provider.partition(rank, center, block_boundaries, rtol)
            self.boundary_blocks = (left, right)
            context.update(left_boundary=left, right_boundary=right,
                           rank_caps=None if self.rank_discovery is not None
                           else tuple(rank),
                           rtol=rtol)
            configure = getattr(provider, 'configure_partition', None)
            if callable(configure):
                configure(sites, left, right)
            selection = {'sites': sites, 'feasible': True}
        else:
            selection = _select_block(provider, rank, center=center,
                                      bounds=bounds)
        if not selection['feasible']:
            raise ValueError(
                'Could not select a refinable central block: '
                f'reason={selection["reason"]}, sites={selection["sites"]}, '
                f'input_capacity={selection["input_capacity"]}, '
                f'required_input_capacity='
                f'{selection["required_input_capacity"]}')
        if (len(selection['sites']) == len(in_dim)) and (
                boundary_mode != 'open'):
            raise ValueError(
                'The central block should leave at least one boundary site '
                'to reconcile its two outgoing gauges')

        cores: list = [None] * len(in_dim)
        openings: Dict[Tuple[int, ...], LoopOpening] = {}
        order = []
        directions = []
        metrics = DecompositionMetrics()
        recursion_diagnostics = []
        context['_metrics'] = metrics

        central_sites = selection['sites']
        central_target = provider.local_target(central_sites, context)
        central_opening = self._open(
            provider=provider,
            rank=rank,
            opener=opener,
            sites=central_sites,
            target=central_target,
            orientation='right',
            fixed_left=None,
            fixed_right=None,
            context=context)

        # Merged end blocks absorb their dummy gauges before cores are stored.
        finalize = getattr(provider, 'finalize_center', None)
        if callable(finalize):
            central_opening = finalize(central_opening, central_sites)
        self._store_opening(
            central_sites,
            'center',
            central_opening,
            cores,
            openings,
            order,
            directions,
            metrics)

        # Open providers own boundary absorption; cyclic compatibility uses the
        # separate two-front path below.
        if boundary_mode == 'open':
            return self._fit_open_boundaries(
                provider=provider,
                rank=rank,
                opener=opener,
                recursion=recursion,
                selection=selection,
                central_opening=central_opening,
                in_dim=in_dim,
                context=context,
                cores=cores,
                openings=openings,
                order=order,
                directions=directions,
                metrics=metrics,
                recursion_diagnostics=recursion_diagnostics)

        # Compatibility path: reconcile both fronts at the final cyclic site.
        remaining = len(in_dim) - len(central_sites)
        left_site = (selection['sites'][0] - 1) % len(in_dim)
        right_site = (selection['sites'][-1] + 1) % len(in_dim)
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
            right_site = (right_site + 1) % len(in_dim)
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
            left_site = (left_site - 1) % len(in_dim)
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
        return self._finish(
            cores=cores,
            central_block=selection['sites'],
            openings=openings,
            order=order,
            directions=directions,
            metrics=metrics,
            diagnostics={
                'recursions': tuple(recursion_diagnostics),
            })


__all__ = [
    'RingTargetProvider',
    'BidirectionalRingDriver',
]
