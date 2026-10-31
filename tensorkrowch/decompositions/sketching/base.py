"""
This script contains:

    Classes:
        * _SketchingFitContext
        * RecursiveSketching
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import torch

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 TimingRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   _resolve_observer)
from tensorkrowch.decompositions.sketching.fitting import (
    BasisFitter, FixedEmbeddingFitter, InputFitter)
from tensorkrowch.decompositions.sketching.projections import RangeProjector
from tensorkrowch.decompositions.sketching.specs import (_DomainSpec,
                                                         _EmbeddingSpec,
                                                         _OutputSpec,
                                                         _SketchingFitSpec)
from tensorkrowch.decompositions.sources import BaseSource


@dataclass
class _SketchingFitContext:  # MARK: _SketchingFitContext
    """Stores diagnostics and numerical state for one independent fit."""

    spec: _SketchingFitSpec
    runtime: _RuntimePolicy
    generator: Optional[torch.Generator] = None
    observer: Optional[Any] = None
    metrics: DecompositionMetrics = field(default_factory=DecompositionMetrics)
    regions: Dict[str, Any] = field(default_factory=dict)
    cores: List[torch.Tensor] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)

    @property
    def collect_metrics(self) -> bool:
        """Whether structured measurements are retained in the result."""
        return self.spec.collect_metrics

    @property
    def need_diagnostics(self) -> bool:
        """Whether measurements are requested for storage or display."""
        return self.collect_metrics or (self.spec.verbosity >= 2)

    def emit(self, name: str, *, site: Optional[int] = None,
             level: int = 1, elapsed: Optional[float] = None,
             values: Optional[Dict[str, Any]] = None) -> None:
        """Emits progress without constructing events on the silent path."""
        if self.observer is not None:
            self.observer.emit(DecompositionEvent(
                name=name, phase='Recursive sketching', level=level,
                site=site, elapsed=elapsed,
                values={} if values is None else values))

    @contextmanager
    def phase(self, name: str, *, site: Optional[int] = None
              ) -> Iterator[None]:
        """Times a phase only when diagnostics need elapsed time."""
        timer_context = (self.runtime.timer() if self.need_diagnostics
                         else nullcontext())
        with timer_context as timer:
            yield
        elapsed = timer.elapsed if self.need_diagnostics else None
        if self.collect_metrics:
            self.metrics.timings.append(TimingRecord(
                name=name, elapsed=elapsed, site=site))
        self.emit(name, site=site, elapsed=elapsed)

    def complete(self, site: int, core: torch.Tensor, n_sites: int) -> None:
        """Reports a core when its solve finishes, in construction order."""
        self.emit('site_complete', site=site, values={
            'total_sites': n_sites, 'shape': tuple(core.shape)})
        self.emit('core', site=site, level=3, values={'tensor': core})


class RecursiveSketching(ABC):  # MARK: RecursiveSketching
    """
    Shares source, embedding and diagnostic preparation for sampled sketches.

    Concrete algorithms implement :meth:`fit` and the topology-specific core
    construction. User-facing source normalization belongs to :class:`TTRSS`;
    this base receives already normalized domains, embeddings and outputs.

    Parameters
    ----------
    source : BaseSource
        Evaluates the input configurations.
    domains, embeddings, outputs
        Internal normalized coordinate domains, embedding matrices and final
        input/output layout, prepared by the concrete algorithm.
    input_fitter : InputFitter or sequence of InputFitter, optional
        Shared fitter or one per final site; inferred when omitted.
    range_projector : RangeProjector, optional
        Complete range-projection strategy. ``None`` uses direct SVD.
    transform : callable, optional
        Called once with a :class:`ConfigurationMap` and returns transformed
        values with the same shape as its initial ``values`` table.
    out_device : str or torch.device, optional
        Device receiving completed cores; ``None`` retains the source device.
    """

    def __init__(self, source: BaseSource, domains: _DomainSpec,
                 embeddings: _EmbeddingSpec, outputs: _OutputSpec, *,
                 input_fitter: Optional[Union[
                     InputFitter, Sequence[InputFitter]]] = None,
                 range_projector: Optional[RangeProjector] = None,
                 transform: Optional[Any] = None,
                 out_device: Optional[Union[str, torch.device]] = 'cpu'
                 ) -> None:
        if not isinstance(source, BaseSource):
            raise TypeError('`source` should inherit from BaseSource')
        if not isinstance(domains, _DomainSpec):
            raise TypeError('`domains` should be _DomainSpec type')
        if not isinstance(embeddings, _EmbeddingSpec):
            raise TypeError('`embeddings` should be _EmbeddingSpec type')
        if not isinstance(outputs, _OutputSpec):
            raise TypeError('`outputs` should be _OutputSpec type')
        if (domains.n_sites != len(source.in_sites)) or \
                (embeddings.n_sites != domains.n_sites) or \
                (outputs.n_input_sites != domains.n_sites):
            raise ValueError('Source, domains and embeddings should agree')
        if (range_projector is not None) and not isinstance(
                range_projector, RangeProjector):
            raise TypeError(
                '`range_projector` should implement RangeProjector')
        if (transform is not None) and not callable(transform):
            raise TypeError('`transform` should be callable or None')

        self._source = source
        self.domains = domains
        self.embeddings = embeddings
        self.outputs = outputs
        self._runtime = _RuntimePolicy(
            device=source.device, out_device=out_device, dtype=source.dtype)
        self._range_projector = range_projector
        self.transform = transform

        if input_fitter is None:
            fitters = []
            for kind, axis in outputs.layout:
                fitters.append(FixedEmbeddingFitter(embeddings.matrix(axis))
                               if kind == 'input' else BasisFitter(
                                   in_dim=outputs.out_shape[axis]))
        elif isinstance(input_fitter, InputFitter):
            fitters = [input_fitter] * outputs.n_sites
        elif isinstance(input_fitter, Sequence) and not isinstance(
                input_fitter, (str, bytes)):
            fitters = tuple(input_fitter)
        else:
            raise TypeError('`input_fitter` should be a fitter or sequence')
        if (len(fitters) != outputs.n_sites) or not all(
                isinstance(fitter, InputFitter) for fitter in fitters):
            raise ValueError('`input_fitter` should cover every final site')
        self.input_fitter = tuple(fitters)

    @property
    def source(self) -> BaseSource:
        """Source supplying values for the fitted function."""
        return self._source

    def _new_context(self, *, rank: Optional[int] = None,
                     cutoff: Optional[float] = None,
                     atol: Optional[float] = None,
                     rtol: Optional[float] = None,
                     cum_percentage: Optional[float] = None,
                     batch_size: int = 64,
                     generator: Optional[torch.Generator] = None,
                     collect_metrics: bool = False,
                     verbose: Union[bool, int] = 0) -> _SketchingFitContext:
        """Creates isolated numerical and diagnostic state for one fit."""
        if (generator is not None) and not isinstance(
                generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator or None')
        spec = _SketchingFitSpec(
            rank=rank, cutoff=cutoff, atol=atol, rtol=rtol,
            cum_percentage=cum_percentage, batch_size=batch_size,
            verbose=verbose, collect_metrics=collect_metrics)
        observer = (_resolve_observer(spec.verbosity, None)
                    if spec.verbosity else None)
        return _SketchingFitContext(
            spec, self._runtime, generator=generator, observer=observer)

    def _execute(self, context: _SketchingFitContext) -> Any:
        """Executes shared preparation and a concrete serial construction."""
        context.emit('start', values={'n_sites': self.outputs.n_sites})
        timer_context = (context.runtime.timer() if context.need_diagnostics
                         else nullcontext())
        with timer_context as timer:
            with context.phase('source.prepare'):
                self._prepare_source(context)
            with context.phase('regions.build'):
                context.regions.update(self._build_regions(context))
            result = self._decompose(context)
            self._validate_result(result, context)
        elapsed = timer.elapsed if context.need_diagnostics else None
        if context.collect_metrics:
            context.metrics.timings.append(TimingRecord(
                name='total', elapsed=elapsed))
        context.emit('summary', elapsed=elapsed, values={
            'n_sites': self.outputs.n_sites, 'rank': result.rank})
        if context.observer is not None:
            context.observer.close(context.metrics)
        return result

    def _fit_input_axis(self, site: int, phi: Any, axis: int,
                        context: _SketchingFitContext) -> torch.Tensor:
        """Fits a physical axis and retains its optional diagnostics."""
        kind, source_axis = self.outputs.layout[site]
        domain = (self.domains.for_site(source_axis) if kind == 'input'
                  else torch.arange(self.outputs.out_shape[source_axis],
                                    device=self._source.device))
        with context.phase('input.fit', site=site):
            fitted = self.input_fitter[site].fit(
                phi, axis, domain, return_info=context.need_diagnostics)
        if context.collect_metrics and (fitted.record is not None):
            context.metrics.input_fits.append(fitted.record)
            threshold = torch.finfo(fitted.tensor.real.dtype).eps ** -0.5
            if fitted.record.condition_number > threshold:
                context.metrics.warnings.append(
                    f'Input fit at site {site} is ill-conditioned')
        return fitted.tensor

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Runs a concrete recursive-sketching decomposition."""

    @abstractmethod
    def _prepare_source(self, context: _SketchingFitContext) -> None:
        """Prepares sample-dependent output coordinates."""

    @abstractmethod
    def _build_regions(self, context: _SketchingFitContext) -> Dict[str, Any]:
        """Builds correlated prefix and suffix sketches."""

    @abstractmethod
    def _decompose(self, context: _SketchingFitContext) -> Any:
        """Constructs the cores and their result."""

    @abstractmethod
    def _validate_result(self, result: Any,
                         context: _SketchingFitContext) -> None:
        """Checks the result's topology and physical dimensions."""


__all__ = ['RecursiveSketching']
