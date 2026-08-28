"""Topology-neutral orchestration for recursive-sketching decompositions."""

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import (Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union)

import torch

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 TimingRecord,
                                                 TruncationRecord)
from tensorkrowch.decompositions.observers import (
    DecompositionEvent,
    DecompositionObserver,
    _resolve_observer,
)
from tensorkrowch.decompositions.sketching.fitting import (
    BasisFitter,
    FittedInputAxis,
    FixedEmbeddingFitter,
    InputFitter,
)
from tensorkrowch.decompositions.sketching.phi import PhiOperator, PhiView
from tensorkrowch.decompositions.sketching.projections import (
    IdentityRangeProjector,
    ProjectedRange,
    RandomizedRangeProjector,
    RangeProjector,
)
from tensorkrowch.decompositions.sketching.specs import (
    _DomainSpec,
    _EmbeddingSpec,
    _OutputSpec,
    _SketchingFitSpec,
)
from tensorkrowch.decompositions.sketching.transforms import (
    GlobalValueTransform,
    IdentityGlobalValueTransform,
    IdentityLocalValueTransform,
    LocalValueTransform,
    _apply_local_transform,
)
from tensorkrowch.decompositions.sources import TensorSource
from tensorkrowch.utils import truncated_svd


_SKETCHING_PHASES = (
    'source.prepare',
    'regions.build',
    'phi.plan',
    'source.evaluate',
    'values.global_transform',
    'values.local_transform',
    'input.fit',
    'range.project',
    'svd.trim',
    'recursion.apply',
    'core.solve',
    'result.validate',
)


@dataclass
class _SketchingFitContext:
    """Mutable state owned exclusively by one recursive-sketching fit."""

    spec: _SketchingFitSpec
    runtime: _RuntimePolicy
    projector: RangeProjector
    input_fitters: Sequence[InputFitter]
    global_transform: GlobalValueTransform
    local_transform: LocalValueTransform
    generator: Optional[torch.Generator] = None
    observer: Optional[DecompositionObserver] = None
    need_diagnostics: bool = False
    metrics: DecompositionMetrics = field(default_factory=DecompositionMetrics)
    regions: Dict[Any, Any] = field(default_factory=dict)
    phis: Dict[int, PhiOperator] = field(default_factory=dict)
    fitted_axes: Dict[int, FittedInputAxis] = field(default_factory=dict)
    projected_ranges: Dict[int, ProjectedRange] = field(default_factory=dict)
    cores: List[torch.Tensor] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, _SketchingFitSpec):
            raise TypeError('`spec` should be _SketchingFitSpec type')
        if not isinstance(self.runtime, _RuntimePolicy):
            raise TypeError('`runtime` should be _RuntimePolicy type')
        if not isinstance(self.projector, RangeProjector):
            raise TypeError('`projector` should implement RangeProjector')
        input_fitters = tuple(self.input_fitters)
        if not input_fitters or not all(
                isinstance(fitter, InputFitter) for fitter in input_fitters):
            raise TypeError(
                '`input_fitters` should contain InputFitter objects')
        object.__setattr__(self, 'input_fitters', input_fitters)
        if not isinstance(self.global_transform, GlobalValueTransform):
            raise TypeError(
                '`global_transform` should implement GlobalValueTransform')
        if not isinstance(self.local_transform, LocalValueTransform):
            raise TypeError(
                '`local_transform` should implement LocalValueTransform')
        if self.generator is not None and \
                not isinstance(self.generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator type or None')
        if self.observer is not None and (
                not callable(getattr(self.observer, 'emit', None)) or
                not callable(getattr(self.observer, 'close', None))):
            raise TypeError('`observer` should implement `emit` and `close`')
        if not isinstance(self.need_diagnostics, bool):
            raise TypeError('`need_diagnostics` should be bool type')

    @property
    def collect_metrics(self) -> bool:
        """Whether structured records should be retained in the result."""
        return self.spec.collect_metrics

    def emit(self,
             name: str,
             *,
             site: Optional[int] = None,
             level: int = 1,
             elapsed: Optional[float] = None,
             values: Optional[Dict[str, Any]] = None) -> None:
        """Emits an event only when a real observer consumes it."""
        if self.observer is None:
            return
        self.observer.emit(DecompositionEvent(
            name=name,
            phase='Recursive sketching',
            level=level,
            site=site,
            elapsed=elapsed,
            values={} if values is None else values))

    @contextmanager
    def phase(self,
              name: str,
              *,
              site: Optional[int] = None,
              values: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        """Times, records and emits one normalized sketching phase."""
        if name not in _SKETCHING_PHASES:
            raise ValueError('`name` should identify a sketching phase')
        timer_context = self.runtime.timer() \
            if self.need_diagnostics else nullcontext()
        with timer_context as timer:
            yield
        elapsed = timer.elapsed if self.need_diagnostics else None
        if self.collect_metrics:
            self.metrics.timings.append(TimingRecord(
                name=name,
                elapsed=0. if elapsed is None else elapsed,
                site=site))
        self.emit(
            name,
            site=site,
            elapsed=elapsed,
            values=values)

    def close(self) -> None:
        """Closes the observer once with this fit's metrics."""
        if self.observer is not None:
            self.observer.close(self.metrics)


class RecursiveSketching(ABC):
    """Base class composing shared recursive-sketching phases and strategies."""

    def __init__(
            self,
            source: TensorSource,
            domains: _DomainSpec,
            embeddings: _EmbeddingSpec,
            outputs: _OutputSpec,
            *,
            input_fitters: Optional[Sequence[InputFitter]] = None,
            range_projector: Optional[RangeProjector] = None,
            global_transform: Optional[GlobalValueTransform] = None,
            local_transform: Optional[LocalValueTransform] = None,
            output_device: Optional[Union[str, torch.device]] = 'cpu',
            synchronize_timers: bool = True) -> None:
        if not isinstance(source, TensorSource):
            raise TypeError('`source` should implement TensorSource')
        if not isinstance(domains, _DomainSpec):
            raise TypeError('`domains` should be _DomainSpec type')
        if not isinstance(embeddings, _EmbeddingSpec):
            raise TypeError('`embeddings` should be _EmbeddingSpec type')
        if not isinstance(outputs, _OutputSpec):
            raise TypeError('`outputs` should be _OutputSpec type')
        if domains.n_sites != len(source.input_dim) or \
                embeddings.n_sites != domains.n_sites or \
                outputs.n_input_sites != domains.n_sites:
            raise ValueError(
                'Source, domains, embeddings and outputs should share sites')
        if source.output_shape is not None:
            source_output = tuple(source.output_shape)
            expected_output = outputs.output_shape if not outputs.scalar else ()
            if source_output == (1,) and outputs.scalar:
                source_output = ()
            if source_output != expected_output:
                raise ValueError('Source and output specs should match')
        if not isinstance(synchronize_timers, bool):
            raise TypeError('`synchronize_timers` should be bool type')

        self.source = source
        self.domains = domains
        self.embeddings = embeddings
        self.outputs = outputs
        self._runtime = _RuntimePolicy(
            device=source.device,
            output_device=output_device,
            dtype=source.dtype,
            synchronize_timers=synchronize_timers)
        self._range_projector = range_projector
        if range_projector is not None and \
                not isinstance(range_projector, RangeProjector):
            raise TypeError(
                '`range_projector` should implement RangeProjector or be None')
        self.global_transform = IdentityGlobalValueTransform() \
            if global_transform is None else global_transform
        self.local_transform = IdentityLocalValueTransform() \
            if local_transform is None else local_transform
        if not isinstance(self.global_transform, GlobalValueTransform):
            raise TypeError(
                '`global_transform` should implement GlobalValueTransform')
        if not isinstance(self.local_transform, LocalValueTransform):
            raise TypeError(
                '`local_transform` should implement LocalValueTransform')

        if input_fitters is None:
            input_fitters = self._default_input_fitters()
        else:
            input_fitters = tuple(input_fitters)
        if len(input_fitters) != outputs.n_sites or not all(
                isinstance(fitter, InputFitter) for fitter in input_fitters):
            raise ValueError(
                '`input_fitters` should contain one fitter per final site')
        self.input_fitters = tuple(input_fitters)

    def _default_input_fitters(self) -> Tuple[InputFitter, ...]:
        """Builds cached fixed-embedding and exact output-basis fitters."""
        fitters = []
        for kind, axis in self.outputs.layout:
            if kind == 'input':
                fitters.append(FixedEmbeddingFitter(
                    self.embeddings.matrix(axis)))
            else:
                fitters.append(BasisFitter(
                    input_dim=self.outputs.output_shape[axis]))
        return tuple(fitters)

    def _new_context(
            self,
            *,
            rank: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            random_projection: bool = True,
            projection_dim: Optional[int] = None,
            projection_oversampling: int = 0,
            n_power_iter: int = 0,
            batch_size: int = 64,
            generator: Optional[torch.Generator] = None,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0,
            observer: Optional[DecompositionObserver] = None
            ) -> _SketchingFitContext:
        """Creates isolated mutable state and resolves fit-time strategies."""
        spec = _SketchingFitSpec(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            random_projection=random_projection,
            projection_dim=projection_dim,
            batch_size=batch_size,
            verbose=verbose,
            collect_metrics=collect_metrics)
        emit_events = bool(spec.verbosity) or (observer is not None)
        fit_observer = _resolve_observer(spec.verbosity, observer) \
            if emit_events else None
        need_diagnostics = collect_metrics or (observer is not None) or \
            (spec.verbosity >= 2)
        projector = self._range_projector
        if projector is None:
            projector = RandomizedRangeProjector(
                projection_dim=spec.effective_projection_dim,
                projection_oversampling=projection_oversampling,
                n_power_iter=n_power_iter,
                synchronize_timers=self._runtime.synchronize_timers) \
                if random_projection else IdentityRangeProjector(
                    synchronize_timers=self._runtime.synchronize_timers)
        return _SketchingFitContext(
            spec=spec,
            runtime=self._runtime,
            projector=projector,
            input_fitters=self.input_fitters,
            global_transform=self.global_transform,
            local_transform=self.local_transform,
            generator=generator,
            observer=fit_observer,
            need_diagnostics=need_diagnostics)

    def _execute(self, context: _SketchingFitContext) -> Any:
        """Runs the topology-neutral prefix/suffix around a concrete driver."""
        context.emit('start', values={'n_sites': self.outputs.n_sites})
        with context.phase('source.prepare'):
            self._prepare_source(context)
        with context.phase('regions.build'):
            regions = self._build_regions(context)
            if not isinstance(regions, dict):
                raise TypeError('`_build_regions` should return a dict')
            context.regions.update(regions)
        result = self._decompose(context)
        with context.phase('result.validate'):
            self._validate_result(result, context)
        context.emit('summary', values={
            'n_sites': self.outputs.n_sites,
            'n_cores': len(context.cores),
        })
        context.close()
        return result

    def _prepare_source(self, context: _SketchingFitContext) -> None:
        """Optional hook for fit-local source preparation."""

    def _plan_phi(self,
                  site: int,
                  context: _SketchingFitContext) -> PhiOperator:
        """Builds and stores one topology-specific Phi operator."""
        with context.phase('phi.plan', site=site):
            phi = self._build_phi(site, context.regions, context)
            if not isinstance(phi, PhiOperator):
                raise TypeError('`_build_phi` should return PhiOperator')
            context.phis[site] = phi
        return phi

    def _transform_local(self,
                         site: int,
                         phi_view: PhiView,
                         context: _SketchingFitContext) -> PhiView:
        """Applies the configured local value transform for one site."""
        with context.phase('values.local_transform', site=site):
            return _apply_local_transform(context.local_transform, phi_view)

    def _fit_input_axis(self,
                        site: int,
                        phi_view: PhiView,
                        axis: int,
                        context: _SketchingFitContext) -> FittedInputAxis:
        """Fits one final-chain site's sampled axis using its strategy."""
        kind, source_axis = self.outputs.layout[site]
        domain = self.domains.for_site(source_axis) if kind == 'input' \
            else torch.arange(
                self.outputs.output_shape[source_axis],
                device=self.source.device)
        fitter = context.input_fitters[site]
        with context.phase('input.fit', site=site):
            fitted = fitter.fit(
                phi_view,
                axis,
                domain,
                context=context,
                return_info=context.need_diagnostics)
            context.fitted_axes[site] = fitted
            if context.collect_metrics and fitted.record is not None:
                context.metrics.input_fits.append(fitted.record)
        return fitted

    def _project_range(self,
                       site: int,
                       matrix: torch.Tensor,
                       axis: int,
                       context: _SketchingFitContext) -> ProjectedRange:
        """Projects one fitted Phi on the axis compressed by the next SVD."""
        with context.phase('range.project', site=site):
            projected = context.projector.project(
                matrix,
                rank=context.spec.rank,
                generator=context.generator,
                axis=axis,
                return_info=context.need_diagnostics)
            context.projected_ranges[site] = projected
            if context.collect_metrics and projected.record is not None:
                context.metrics.range_projections.append(projected.record)
        return projected

    def _trim(self,
              site: int,
              projected: ProjectedRange,
              context: _SketchingFitContext):
        """Truncates a projected range and lifts its selected left vectors."""
        with context.phase('svd.trim', site=site):
            if context.need_diagnostics:
                u, s, vh, info = truncated_svd(
                    projected.small_matrix,
                    return_info=True,
                    **context.spec.truncation.as_kwargs())
                record = TruncationRecord.from_svd_info(info, site=site)
                if context.collect_metrics:
                    context.metrics.truncations.append(record)
            else:
                u, s, vh = truncated_svd(
                    projected.small_matrix,
                    **context.spec.truncation.as_kwargs())
                record = None
            u = projected.restore_left(u)
        return u, s, vh, record

    def _validate_result(self,
                         result: Any,
                         context: _SketchingFitContext) -> None:
        """Optional hook for topology-specific result invariants."""

    @abstractmethod
    def fit(self, *args, **kwargs):
        """Runs one concrete recursive-sketching decomposition."""

    @abstractmethod
    def _build_regions(self,
                       context: _SketchingFitContext) -> Dict[Any, Any]:
        """Builds topology-specific region sketches."""

    @abstractmethod
    def _build_phi(self,
                   site: int,
                   regions: Dict[Any, Any],
                   context: _SketchingFitContext) -> PhiOperator:
        """Builds one topology-specific lazy Phi."""

    @abstractmethod
    def _decompose(self, context: _SketchingFitContext) -> Any:
        """Runs topology-specific dependencies using the shared helpers."""

    @abstractmethod
    def _solve_local(self, *args, **kwargs):
        """Solves one topology-specific core equation."""

    @abstractmethod
    def _assemble_result(self, *args, **kwargs):
        """Assembles the lightweight decomposition result."""


__all__ = ['RecursiveSketching']
