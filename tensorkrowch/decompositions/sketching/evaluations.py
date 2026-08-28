"""Deduplicated evaluation plans shared by lazy Phi operators."""

from dataclasses import dataclass
from typing import (Optional, Sequence, Tuple)

import torch

from tensorkrowch.decompositions.metrics import EvaluationStats
from tensorkrowch.decompositions.sketching.regions import (SiteRegion,
                                                           _SamplePool)
from tensorkrowch.decompositions.sketching.specs import _OutputSpec
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 TensorSource)


def _site_values(
        configurations: ConfigurationBatch) -> Tuple[torch.Tensor, ...]:
    """Returns one tensor per configuration site."""
    if configurations.packed:
        return tuple(
            configurations.values[:, site]
            for site in range(configurations.n_sites))
    return tuple(configurations.values)


def _concatenate_configurations(
        batches: Sequence[ConfigurationBatch]) -> ConfigurationBatch:
    """Concatenates compatible packed or heterogeneous configuration batches."""
    batches = tuple(batches)
    if not batches:
        raise ValueError('At least one configuration batch is required')
    first = batches[0]
    for batch in batches:
        if not isinstance(batch, ConfigurationBatch):
            raise TypeError(
                '`batches` should contain ConfigurationBatch objects')
        if batch.n_sites != first.n_sites:
            raise ValueError(
                'Configuration batches should contain the same number of sites')
        if batch.kind != first.kind:
            raise ValueError(
                'Configuration batches should use the same value kind')
        if batch.site_shape != first.site_shape:
            raise ValueError(
                'Configuration batches should have matching site shapes')
        if batch.device != first.device:
            raise ValueError('Configuration batches should share a device')
        if batch.dtype != first.dtype:
            raise ValueError('Configuration batches should share a dtype')

    if all(batch.packed for batch in batches):
        values = torch.cat([batch.values for batch in batches], dim=0)
    else:
        split = [_site_values(batch) for batch in batches]
        values = tuple(torch.cat(
            [batch_values[site] for batch_values in split], dim=0)
            for site in range(first.n_sites))
    return ConfigurationBatch(values, kind=first.kind)


@dataclass(frozen=True)
class _EvaluationRequest:
    """One flattened Phi or closure request before global deduplication."""

    configurations: ConfigurationBatch
    result_shape: Sequence[int]
    output_spec: Optional[_OutputSpec] = None
    output_labels: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.configurations, ConfigurationBatch):
            raise TypeError(
                '`configurations` should be ConfigurationBatch type')
        try:
            result_shape = tuple(self.result_shape)
        except TypeError as exc:
            raise TypeError('`result_shape` should be a sequence of integers') \
                from exc
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
               for dim in result_shape):
            raise ValueError(
                '`result_shape` should contain non-negative integers')
        n_results = 1
        for dim in result_shape:
            n_results *= dim
        if n_results != self.configurations.batch_size:
            raise ValueError(
                '`result_shape` should contain one entry per configuration')
        if self.output_spec is not None and \
                not isinstance(self.output_spec, _OutputSpec):
            raise TypeError('`output_spec` should be _OutputSpec type or None')
        if self.output_spec is None or self.output_spec.scalar:
            if self.output_labels is not None:
                raise ValueError(
                    '`output_labels` requires a tensor-valued output spec')
        else:
            labels = self.output_spec._validate_flat_labels(
                self.output_labels)
            if labels.shape[0] != self.configurations.batch_size:
                raise ValueError(
                    '`output_labels` should match the configuration batch')
            object.__setattr__(self, 'output_labels', labels)
        object.__setattr__(self, 'result_shape', result_shape)


@dataclass(frozen=True)
class _IncidenceMap:
    """Maps one request's rows to globally unique source configurations."""

    unique_ids: torch.Tensor
    result_shape: Sequence[int]
    output_spec: Optional[_OutputSpec] = None
    output_labels: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.unique_ids, torch.Tensor) or \
                self.unique_ids.ndim != 1 or \
                self.unique_ids.dtype != torch.long:
            raise TypeError('`unique_ids` should be a one-dimensional long tensor')
        result_shape = tuple(self.result_shape)
        n_results = 1
        for dim in result_shape:
            if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
                raise ValueError(
                    '`result_shape` should contain non-negative integers')
            n_results *= dim
        if n_results != self.unique_ids.shape[0]:
            raise ValueError(
                '`result_shape` should contain one entry per incidence id')
        if self.output_spec is not None and \
                not isinstance(self.output_spec, _OutputSpec):
            raise TypeError('`output_spec` should be _OutputSpec type or None')
        if self.output_labels is not None:
            if not isinstance(self.output_labels, torch.Tensor) or \
                    self.output_labels.shape != self.unique_ids.shape or \
                    self.output_labels.dtype != torch.long:
                raise TypeError(
                    '`output_labels` should be a long vector matching ids')
        object.__setattr__(self, 'result_shape', result_shape)


@dataclass(frozen=True)
class EvaluationView:
    """Read-only configurations, values and incidences for advanced callbacks."""

    configurations: ConfigurationBatch
    values: Optional[torch.Tensor] = None
    incidences: Sequence[_IncidenceMap] = ()
    phase: str = 'collect'

    def __post_init__(self) -> None:
        if not isinstance(self.configurations, ConfigurationBatch):
            raise TypeError(
                '`configurations` should be ConfigurationBatch type')
        if self.values is not None:
            if not isinstance(self.values, torch.Tensor):
                raise TypeError('`values` should be torch.Tensor type or None')
            if self.values.shape[0] != self.configurations.batch_size:
                raise ValueError(
                    '`values` should match the configuration batch')
        incidences = tuple(self.incidences)
        if not all(isinstance(item, _IncidenceMap) for item in incidences):
            raise TypeError('`incidences` should contain _IncidenceMap objects')
        if self.phase not in ('collect', 'expand', 'frozen', 'evaluated'):
            raise ValueError('`phase` is not a valid evaluation phase')
        object.__setattr__(self, 'incidences', incidences)


@dataclass(frozen=True)
class _EvaluationPlan:
    """Frozen unique configurations and request incidence maps."""

    source: TensorSource
    configurations: ConfigurationBatch
    incidences: Sequence[_IncidenceMap]
    stats: EvaluationStats
    global_transform_prepared: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, TensorSource):
            raise TypeError('`source` should implement TensorSource')
        if not isinstance(self.configurations, ConfigurationBatch):
            raise TypeError(
                '`configurations` should be ConfigurationBatch type')
        incidences = tuple(self.incidences)
        if not incidences or \
                not all(isinstance(item, _IncidenceMap)
                        for item in incidences):
            raise TypeError(
                '`incidences` should contain _IncidenceMap objects')
        if not isinstance(self.stats, EvaluationStats):
            raise TypeError('`stats` should be EvaluationStats type')
        if not isinstance(self.global_transform_prepared, bool):
            raise TypeError('`global_transform_prepared` should be bool type')
        object.__setattr__(self, 'incidences', incidences)

    def view(self) -> EvaluationView:
        """Returns the frozen global evaluation table without values."""
        return EvaluationView(
            configurations=self.configurations,
            incidences=self.incidences,
            phase='frozen')


class _EvaluationRegistry:
    """Deduplicates request points and builds their gather incidence maps."""

    @staticmethod
    def build(
            source: TensorSource,
            requests: Sequence[_EvaluationRequest],
            global_transform_prepared: bool = False) -> _EvaluationPlan:
        """Freezes requests into one globally deduplicated evaluation plan."""
        requests = tuple(requests)
        configurations = _concatenate_configurations(
            [request.configurations for request in requests])
        pool = _SamplePool(configurations)
        unique = pool.restrict(SiteRegion(range(configurations.n_sites)))
        unique_configurations = configurations.index_select(
            unique.representative_row_ids)

        incidences = []
        start = 0
        for request in requests:
            stop = start + request.configurations.batch_size
            incidences.append(_IncidenceMap(
                unique_ids=unique.inverse_ids[start:stop],
                result_shape=request.result_shape,
                output_spec=request.output_spec,
                output_labels=request.output_labels))
            start = stop
        requested_points = configurations.batch_size
        unique_points = unique_configurations.batch_size
        stats = EvaluationStats(
            requested_points=requested_points,
            unique_points=unique_points,
            cache_hits=requested_points - unique_points)
        return _EvaluationPlan(
            source=source,
            configurations=unique_configurations,
            incidences=incidences,
            stats=stats,
            global_transform_prepared=global_transform_prepared)


class _EvaluationPlanBuilder:
    """Mutable collect/expand builder whose ``freeze`` result is immutable."""

    def __init__(self, source: TensorSource) -> None:
        if not isinstance(source, TensorSource):
            raise TypeError('`source` should implement TensorSource')
        self.source = source
        self._requests = []
        self._phase = 'collect'
        self._global_transform_prepared = False

    @property
    def phase(self) -> str:
        """Current lifecycle phase."""
        return self._phase

    def _validate_request(self, request: _EvaluationRequest) -> None:
        """Validates one request against the shared source input metadata."""
        if not isinstance(request, _EvaluationRequest):
            raise TypeError('`request` should be _EvaluationRequest type')
        if request.configurations.n_sites != len(self.source.input_dim):
            raise ValueError(
                'Configurations should contain one value per source input site')

    def collect(self, request: _EvaluationRequest) -> int:
        """Adds a Phi request during the initial collect phase."""
        if self._phase != 'collect':
            raise RuntimeError('Phi requests can only be collected before expand')
        if self._global_transform_prepared:
            raise RuntimeError(
                'Phi requests should be collected before global preparation')
        self._validate_request(request)
        handle = len(self._requests)
        self._requests.append(request)
        return handle

    def expand(self, configurations: ConfigurationBatch) -> int:
        """Adds closure points before freeze and returns their result handle."""
        if self._phase == 'frozen':
            raise RuntimeError('An evaluation plan cannot expand after freeze')
        if not isinstance(configurations, ConfigurationBatch):
            raise TypeError(
                '`configurations` should be ConfigurationBatch type')
        self._phase = 'expand'
        request = _EvaluationRequest(
            configurations=configurations,
            result_shape=(configurations.batch_size,))
        self._validate_request(request)
        handle = len(self._requests)
        self._requests.append(request)
        return handle

    def snapshot(self) -> EvaluationView:
        """Returns all requested rows without exposing mutable builder storage."""
        if not self._requests:
            raise ValueError('The evaluation plan should contain a request')
        return EvaluationView(
            configurations=_concatenate_configurations([
                request.configurations for request in self._requests]),
            phase=self._phase)

    def _mark_global_transform_prepared(self) -> None:
        """Marks that global closure requirements were handled before freeze."""
        if self._phase == 'frozen':
            raise RuntimeError(
                'A global transform cannot be prepared after freeze')
        if self._global_transform_prepared:
            raise RuntimeError('The global transform is already prepared')
        self._global_transform_prepared = True

    def freeze(self) -> _EvaluationPlan:
        """Deduplicates every request and permanently freezes this builder."""
        if self._phase == 'frozen':
            raise RuntimeError('The evaluation plan is already frozen')
        if not self._requests:
            raise ValueError('The evaluation plan should contain a request')
        plan = _EvaluationRegistry.build(
            self.source,
            self._requests,
            global_transform_prepared=self._global_transform_prepared)
        self._phase = 'frozen'
        return plan


class _EvaluationSession:
    """Evaluates one frozen plan once and scatters values to all requests."""

    def __init__(self,
                 plan: _EvaluationPlan,
                 global_transform=None,
                 context=None) -> None:
        if not isinstance(plan, _EvaluationPlan):
            raise TypeError('`plan` should be _EvaluationPlan type')
        self.plan = plan
        self.global_transform = global_transform
        self.context = context
        if global_transform is not None and \
                not getattr(global_transform, 'is_identity', False) and \
                not plan.global_transform_prepared:
            raise RuntimeError(
                'The global transform should be prepared before freeze')
        self._values = None
        self._raw_values = None
        self._results = None
        self._stats = None

    @property
    def stats(self) -> EvaluationStats:
        """Final session stats, or planned deduplication stats before evaluate."""
        return self.plan.stats if self._stats is None else self._stats

    def _evaluate_unique(self, batch_size: Optional[int]) -> torch.Tensor:
        """Evaluates unique configurations in deterministic contiguous batches."""
        n_points = self.plan.configurations.batch_size
        if batch_size is None:
            batch_size = n_points
        elif isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError('`batch_size` should be int type or None')
        elif batch_size < 1:
            raise ValueError('`batch_size` should be positive')

        before = getattr(self.plan.source, 'evaluation_stats', None)
        chunks = []
        n_calls = 0
        for start in range(0, n_points, batch_size):
            stop = min(start + batch_size, n_points)
            ids = torch.arange(
                start, stop, device=self.plan.configurations.device)
            chunks.append(self.plan.source.evaluate(
                self.plan.configurations.index_select(ids)))
            n_calls += 1
        values = torch.cat(chunks, dim=0)
        if values.shape[0] != n_points:
            raise ValueError(
                'The source should preserve the configuration batch dimension')

        after = getattr(self.plan.source, 'evaluation_stats', None)
        if isinstance(before, EvaluationStats) and \
                isinstance(after, EvaluationStats):
            delta = after.delta(before)
            batches = delta.batches
            source_calls = delta.source_calls
        else:
            batches = n_calls
            source_calls = n_calls
        self._stats = EvaluationStats(
            requested_points=self.plan.stats.requested_points,
            unique_points=self.plan.stats.unique_points,
            batches=batches,
            cache_hits=self.plan.stats.cache_hits,
            source_calls=source_calls)
        return values

    @staticmethod
    def _scatter(
            values: torch.Tensor,
            incidence: _IncidenceMap) -> torch.Tensor:
        """Scatters global values through one incidence and output selection."""
        unique_ids = incidence.unique_ids.to(values.device)
        selected = values.index_select(0, unique_ids)
        if incidence.output_spec is None:
            return selected.reshape(
                (*incidence.result_shape, *selected.shape[1:]))
        selected = incidence.output_spec.validate_values(selected)
        if incidence.output_spec.scalar:
            return selected.reshape(incidence.result_shape)
        labels = incidence.output_labels.to(values.device)
        selected = selected.reshape(selected.shape[0], -1).gather(
            1, labels.unsqueeze(1)).squeeze(1)
        return selected.reshape(incidence.result_shape)

    def evaluate(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor,
                                                                  ...]:
        """Evaluates and scatters the plan, reusing results on repeated calls."""
        if self._results is not None:
            return self._results
        self.evaluate_source(batch_size=batch_size)
        if self.global_transform is None:
            self._values = self._raw_values
        else:
            view = EvaluationView(
                configurations=self.plan.configurations,
                values=self._raw_values,
                incidences=self.plan.incidences,
                phase='evaluated')
            self._values = self.global_transform.apply(view, self.context)
            if not isinstance(self._values, torch.Tensor):
                raise TypeError(
                    'A global value transform should return a torch.Tensor')
            if self._values.shape != self._raw_values.shape:
                raise ValueError(
                    'A global value transform should preserve value shape')
            if self._values.device != self._raw_values.device:
                raise ValueError(
                    'A global value transform should preserve value device')
        self._results = tuple(
            self._scatter(self._values, incidence)
            for incidence in self.plan.incidences)
        return self._results

    def evaluate_source(
            self,
            batch_size: Optional[int] = None) -> torch.Tensor:
        """Evaluates and caches only the globally unique source values."""
        if self._raw_values is None:
            self._raw_values = self._evaluate_unique(batch_size)
        return self._raw_values

    def result(self,
               handle: int,
               batch_size: Optional[int] = None) -> torch.Tensor:
        """Returns one request result by its stable builder handle."""
        if isinstance(handle, bool) or not isinstance(handle, int):
            raise TypeError('`handle` should be int type')
        if handle < 0 or handle >= len(self.plan.incidences):
            raise ValueError('`handle` is outside the evaluation plan')
        return self.evaluate(batch_size=batch_size)[handle]

    def view(self, batch_size: Optional[int] = None) -> EvaluationView:
        """Returns the evaluated global table and immutable incidence maps."""
        self.evaluate(batch_size=batch_size)
        return EvaluationView(
            configurations=self.plan.configurations,
            values=self._values,
            incidences=self.plan.incidences,
            phase='evaluated')


__all__ = ['EvaluationView']
