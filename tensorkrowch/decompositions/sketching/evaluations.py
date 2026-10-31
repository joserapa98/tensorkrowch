"""
This script contains:

    Public classes:
        * ConfigurationMap

    Internal classes:
        * _EvaluationRequest
        * _IncidenceMap
        * _EvaluationPlan
        * _EvaluationPlanBuilder

    Internal functions:
        * _input_values
        * _concatenate_configurations
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.decompositions.metrics import EvaluationStats
from tensorkrowch.decompositions.sketching.regions import (SiteRegion,
                                                         _SamplePool)
from tensorkrowch.decompositions.sketching.specs import _OutputSpec
from tensorkrowch.decompositions.sources import (BaseSource,
                                               ConfigurationBatch)


def _input_values(configurations: ConfigurationBatch
                  ) -> Tuple[torch.Tensor, ...]:
    """Returns one tensor per configuration input."""
    if configurations.packed:
        return tuple(configurations.values[:, site]
                     for site in range(configurations.n_inputs))
    return tuple(configurations.values)


def _concatenate_configurations(batches: Sequence[ConfigurationBatch]
                                ) -> ConfigurationBatch:
    """Concatenates compatible configuration batches along their row axis."""
    batches = tuple(batches)
    if not batches:
        raise ValueError('`batches` should contain a configuration batch')
    first = batches[0]
    if not isinstance(first, ConfigurationBatch):
        raise TypeError('`batches` should contain ConfigurationBatch objects')
    for batch in batches[1:]:
        if not isinstance(batch, ConfigurationBatch):
            raise TypeError(
                '`batches` should contain ConfigurationBatch objects')
        if batch.n_inputs != first.n_inputs:
            raise ValueError('`batches` should have the same input count')
        if batch.kind != first.kind:
            raise ValueError('`batches` should use the same value kind')
        if batch.feature_shape != first.feature_shape:
            raise ValueError('`batches` should have matching feature shapes')
        if batch.device != first.device:
            raise ValueError('`batches` should share a device')
        if batch.dtype != first.dtype:
            raise ValueError('`batches` should share a dtype')

    if all(batch.packed for batch in batches):
        values = torch.cat([batch.values for batch in batches], dim=0)
    else:
        split = [_input_values(batch) for batch in batches]
        values = tuple(torch.cat([items[site] for items in split], dim=0)
                       for site in range(first.n_inputs))
    return ConfigurationBatch(values, kind=first.kind)


@dataclass(frozen=True)
class _EvaluationRequest:  # MARK: _EvaluationRequest
    """One flattened Phi request before global deduplication."""

    configurations: ConfigurationBatch
    result_shape: Sequence[int]
    output_spec: Optional[_OutputSpec] = None
    output_labels: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.configurations, ConfigurationBatch):
            raise TypeError('`configurations` should be ConfigurationBatch')
        if isinstance(self.result_shape, (str, bytes)) or \
                not isinstance(self.result_shape, Sequence):
            raise TypeError('`result_shape` should be a sequence of integers')
        shape = tuple(self.result_shape)
        n_results = 1
        for dim in shape:
            if isinstance(dim, bool) or not isinstance(dim, int) or (dim < 0):
                raise ValueError(
                    '`result_shape` should contain non-negative integers')
            n_results *= dim
        if n_results != self.configurations.batch_size:
            raise ValueError(
                '`result_shape` should contain one entry per configuration')
        if (self.output_spec is not None) and \
                not isinstance(self.output_spec, _OutputSpec):
            raise TypeError('`output_spec` should be _OutputSpec or None')
        if (self.output_spec is None) or self.output_spec.scalar:
            if self.output_labels is not None:
                raise ValueError(
                    '`output_labels` requires a tensor-valued output spec')
        else:
            labels = self.output_spec._validate_flat_labels(self.output_labels)
            if labels.shape[0] != self.configurations.batch_size:
                raise ValueError(
                    '`output_labels` should match the configuration batch')
            object.__setattr__(self, 'output_labels', labels)
        object.__setattr__(self, 'result_shape', shape)


@dataclass(frozen=True)
class _IncidenceMap:  # MARK: _IncidenceMap
    """Gather indices and output selection for one planned Phi."""

    unique_ids: torch.Tensor
    result_shape: Tuple[int, ...]
    output_spec: Optional[_OutputSpec] = None
    output_labels: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class _EvaluationPlan:  # MARK: _EvaluationPlan
    """Unique source configurations and the gathers for all planned Phis."""

    source: BaseSource
    configurations: ConfigurationBatch
    incidences: Tuple[_IncidenceMap, ...]
    stats: EvaluationStats


class _EvaluationPlanBuilder:  # MARK: _EvaluationPlanBuilder
    """Collects Phi requests and deduplicates their source configurations."""

    def __init__(self, source: BaseSource) -> None:
        if not isinstance(source, BaseSource):
            raise TypeError('`source` should inherit from BaseSource')
        self.source = source
        self._requests = []
        self._frozen = False

    def collect(self, request: _EvaluationRequest) -> int:
        """Adds a request and returns its stable result handle."""
        if self._frozen:
            raise RuntimeError('The evaluation plan is already frozen')
        if not isinstance(request, _EvaluationRequest):
            raise TypeError('`request` should be _EvaluationRequest type')
        if request.configurations.n_inputs != len(self.source.in_sites):
            raise ValueError(
                '`configurations` should contain one value per source input')
        handle = len(self._requests)
        self._requests.append(request)
        return handle

    def freeze(self) -> _EvaluationPlan:
        """Deduplicates the requests and permanently freezes this builder."""
        if self._frozen:
            raise RuntimeError('The evaluation plan is already frozen')
        if not self._requests:
            raise ValueError('The evaluation plan should contain a request')
        configurations = _concatenate_configurations(
            [request.configurations for request in self._requests])
        pool = _SamplePool(configurations)
        unique = pool.restrict(SiteRegion(range(configurations.n_inputs)))
        unique_configurations = configurations.index_select(
            unique.representative_row_ids)

        incidences = []
        start = 0
        for request in self._requests:
            stop = start + request.configurations.batch_size
            incidences.append(_IncidenceMap(
                unique_ids=unique.inverse_ids[start:stop],
                result_shape=request.result_shape,
                output_spec=request.output_spec,
                output_labels=request.output_labels))
            start = stop
        stats = EvaluationStats(
            requested_points=configurations.batch_size,
            unique_points=unique_configurations.batch_size,
            cache_hits=(configurations.batch_size -
                        unique_configurations.batch_size))
        self._frozen = True
        return _EvaluationPlan(self.source, unique_configurations,
                               tuple(incidences), stats)


class ConfigurationMap:  # MARK: ConfigurationMap
    """
    Source values shared by all RSS queries and a global transformation.

    RSS passes this object to ``transform(configuration_map)`` after evaluating
    the unique configurations needed by the local Phis. :attr:`configurations`
    contains these inputs directly, and :attr:`values` contains the aligned raw
    source values. Both tables keep their original shape and order while the
    callback runs. The callback returns transformed values with the same shape.

    :meth:`evaluate` can query additional configurations. It returns raw source
    values, preserving the requested order and repetitions, and evaluates only
    points absent from the shared cache. These auxiliary points do not change
    the initial tables or the sketch samples. Evaluation never recursively
    applies the transformation.

    Instances are created by RSS; callbacks normally only read these properties
    and call :meth:`evaluate`. Source input columns follow ``source.in_sites``.
    Treat the exposed tables as read-only so cached raw evaluations remain
    available throughout the callback.

    Examples
    --------
    A symmetry average can evaluate new configurations in the callback::

        def transform(configuration_map):
            inputs = configuration_map.configurations
            reflected = inputs.flip(1)
            return (configuration_map.values +
                    configuration_map.evaluate(reflected)) / 2
    """

    def __init__(self, plan: _EvaluationPlan,
                 batch_size: Optional[int] = 64,
                 transform: Optional[Callable] = None) -> None:
        if not isinstance(plan, _EvaluationPlan):
            raise TypeError('`plan` should be _EvaluationPlan type')
        if (batch_size is not None) and (
                isinstance(batch_size, bool) or
                not isinstance(batch_size, int)):
            raise TypeError('`batch_size` should be int type or None')
        if (batch_size is not None) and (batch_size < 1):
            raise ValueError('`batch_size` should be positive')
        if (transform is not None) and not callable(transform):
            raise TypeError('`transform` should be callable or None')
        self._plan = plan
        self._batch_size = batch_size
        self._transform = transform
        self._values = None
        self._transformed_values = None
        self._cache_configurations = plan.configurations
        self._cache_values = None
        self._stats = plan.stats

    @property
    def configurations(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Initial packed inputs or one tensor per input, in query order."""
        return self._plan.configurations.values

    @property
    def values(self) -> torch.Tensor:
        """Raw source values aligned with the initial configurations."""
        return self.evaluate_source()

    @property
    def stats(self) -> EvaluationStats:
        """Source calls and cache reuse, including auxiliary evaluations."""
        return self._stats

    def _evaluate_batches(self, configurations: ConfigurationBatch
                          ) -> torch.Tensor:
        """Evaluates missing points in contiguous batches."""
        batch_size = self._batch_size or configurations.batch_size
        before = self._plan.source.evaluation_stats
        chunks = []
        n_calls = 0
        for start in range(0, configurations.batch_size, batch_size):
            ids = torch.arange(start, min(start + batch_size,
                                          configurations.batch_size),
                               device=configurations.device)
            values = self._plan.source.evaluate(
                configurations.index_select(ids))
            if values.shape[0] != ids.numel():
                raise ValueError(
                    'Source values should preserve the configuration batch')
            chunks.append(values)
            n_calls += 1
        values = torch.cat(chunks, dim=0)
        delta = self._plan.source.evaluation_stats.delta(before)
        # Custom sources may not maintain counters, but each call still counts.
        self._stats = EvaluationStats(
            requested_points=self._stats.requested_points,
            unique_points=self._stats.unique_points,
            batches=self._stats.batches + (delta.batches or n_calls),
            cache_hits=self._stats.cache_hits,
            source_calls=(self._stats.source_calls +
                          (delta.source_calls or n_calls)))
        return values

    def evaluate_source(self) -> torch.Tensor:
        """Evaluates and retains the initial raw source table once."""
        if self._values is None:
            self._values = self._evaluate_batches(self._plan.configurations)
            self._cache_values = self._values
        return self._values

    def evaluate(self, configurations: Union[
            torch.Tensor, Sequence[torch.Tensor], ConfigurationBatch]
                 ) -> torch.Tensor:
        """
        Evaluates raw source values using the shared configuration cache.

        Parameters
        ----------
        configurations : torch.Tensor or sequence of torch.Tensor
            Packed inputs with shape ``(batch, n_inputs, *feature_shape)``, or
            one tensor per input with a common leading batch dimension. Use
            the same input order, feature shapes, device and dtype as
            :attr:`configurations`. Discrete inputs contain scalar indices.

        Returns
        -------
        torch.Tensor
            Raw source values with shape ``(batch, *out_shape)``, in requested
            order. Repeated configurations produce repeated rows.
        """
        self.evaluate_source()
        if not isinstance(configurations, ConfigurationBatch):
            configurations = ConfigurationBatch(
                configurations, kind=self._plan.configurations.kind)
        combined = _concatenate_configurations(
            (self._cache_configurations, configurations))
        if configurations.batch_size == 0:
            return self._cache_values[:0]

        pool = _SamplePool(combined)
        unique = pool.restrict(SiteRegion(range(combined.n_inputs)))
        n_cached = self._cache_configurations.batch_size
        representatives = unique.representative_row_ids
        missing_ids = representatives[representatives >= n_cached]
        n_missing = missing_ids.numel()
        lookup = torch.empty(unique.n_unique, dtype=torch.long,
                             device=combined.device)
        lookup[unique.inverse_ids[:n_cached]] = torch.arange(
            n_cached, device=combined.device)
        if n_missing:
            missing = combined.index_select(missing_ids)
            values = self._evaluate_batches(missing)
            self._cache_configurations = _concatenate_configurations(
                (self._cache_configurations, missing))
            self._cache_values = torch.cat((self._cache_values, values), dim=0)
            lookup[unique.inverse_ids[missing_ids]] = torch.arange(
                n_cached, n_cached + n_missing, device=combined.device)

        self._stats = EvaluationStats(
            requested_points=(self._stats.requested_points +
                              configurations.batch_size),
            unique_points=self._stats.unique_points + n_missing,
            batches=self._stats.batches,
            cache_hits=(self._stats.cache_hits +
                        configurations.batch_size - n_missing),
            source_calls=self._stats.source_calls)
        ids = lookup[unique.inverse_ids[n_cached:]]
        return self._cache_values.index_select(0, ids.to(
            self._cache_values.device))

    def prepare_values(self) -> torch.Tensor:
        """Applies the callback once to the stable initial raw table."""
        if self._transformed_values is None:
            raw = self.evaluate_source()
            values = raw if self._transform is None else self._transform(self)
            if not isinstance(values, torch.Tensor):
                raise TypeError('`transform` should return a torch.Tensor')
            if values.shape != raw.shape:
                raise ValueError('`transform` should preserve the value shape')
            if values.device != raw.device:
                raise ValueError(
                    '`transform` should preserve the value device')
            if not (values.is_floating_point() or values.is_complex()):
                raise TypeError(
                    '`transform` should return floating or complex values')
            if not torch.isfinite(values).all():
                raise ValueError('`transform` should return finite values')
            self._transformed_values = values
        return self._transformed_values

    def _result(self, handle: int,
               index_selection: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scatters one planned Phi without retaining another tensor cache."""
        if isinstance(handle, bool) or not isinstance(handle, int):
            raise TypeError('`handle` should be int type')
        if (handle < 0) or (handle >= len(self._plan.incidences)):
            raise ValueError('`handle` is outside the evaluation plan')
        incidence = self._plan.incidences[handle]
        values = self.prepare_values()
        ids = incidence.unique_ids
        labels = incidence.output_labels
        result_shape = incidence.result_shape
        if index_selection is not None:
            selection = index_selection.to(ids.device).reshape(
                -1, len(result_shape))
            indices = tuple(selection.unbind(-1))
            ids = ids.reshape(result_shape)[indices]
            if labels is not None:
                labels = labels.reshape(result_shape)[indices]
            result_shape = tuple(index_selection.shape[:-1])
        selected = values.index_select(0, ids.to(values.device))
        if incidence.output_spec is None:
            return selected.reshape(
                (*result_shape, *selected.shape[1:]))
        selected = incidence.output_spec.validate_values(selected)
        if incidence.output_spec.scalar:
            return selected.reshape(result_shape)
        labels = labels.to(values.device)
        selected = selected.reshape(selected.shape[0], -1).gather(
            1, labels.unsqueeze(1)).squeeze(1)
        return selected.reshape(result_shape)


__all__ = ['ConfigurationMap']
