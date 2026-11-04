"""
This script contains:

    Classes:
        * ConfigurationMap

    Functions:
        * _input_values
        * _concatenate_configurations
"""

from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.decompositions.metrics import EvaluationStats
from tensorkrowch.decompositions.sketching.regions import (SiteRegion,
                                                           _SamplePool)
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

    def __init__(self, source: BaseSource, batch_size: Optional[int] = 64,
                 transform: Optional[Callable] = None) -> None:
        if not isinstance(source, BaseSource):
            raise TypeError('`source` should inherit from BaseSource')
        if (batch_size is not None) and (
                isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or (batch_size < 1)):
            raise ValueError('`batch_size` should be positive or None')
        if (transform is not None) and not callable(transform):
            raise TypeError('`transform` should be callable or None')
        self.source = source
        self._batch_size = batch_size
        self._transform = transform
        # Requests and gather indices share one frozen configuration table.
        self._requests = []
        self._incidences = []
        self._configurations = None

        # Raw evaluations remain available after the global transformation.
        self._values = None
        self._transformed_values = None
        self._cache_configurations = None
        self._cache_values = None
        self._stats = EvaluationStats()

    def register(self, phi: object,
                 index_selection: Optional[torch.Tensor] = None) -> int:
        """
        Registers one Phi query before evaluation freezes the table.

        Parameters
        ----------
        phi : PhiOperator
            Local operator sharing this map's source.
        index_selection : torch.Tensor, optional
            Selected Phi entries; omitted to register the complete tensor.
        """
        from tensorkrowch.decompositions.sketching.phi import PhiOperator

        if not isinstance(phi, PhiOperator):
            raise TypeError('`phi` should be PhiOperator type')
        if self._configurations is not None:
            raise RuntimeError('The configuration map is already frozen')
        if phi.source is not self.source:
            raise ValueError('The map and Phi should share a source')
        configurations, shape, labels = phi._request(index_selection)
        handle = len(self._requests)
        self._requests.append((configurations, shape, phi.output_spec, labels))
        return handle

    def freeze(self) -> None:
        """Deduplicates all registered queries once before evaluation."""
        if self._configurations is not None:
            return
        if not self._requests:
            raise ValueError('The map should contain a registered Phi')
        configurations = _concatenate_configurations(
            [request[0] for request in self._requests])
        pool = _SamplePool(configurations)
        unique = pool.restrict(SiteRegion(range(configurations.n_inputs)))
        self._configurations = configurations.index_select(
            unique.representative_row_ids)
        self._cache_configurations = self._configurations
        offset = 0
        for batch, shape, output_spec, labels in self._requests:
            stop = offset + batch.batch_size
            self._incidences.append((unique.inverse_ids[offset:stop],
                                     tuple(shape), output_spec, labels))
            offset = stop
        self._stats = EvaluationStats(
            requested_points=configurations.batch_size,
            unique_points=self._configurations.batch_size,
            cache_hits=configurations.batch_size -
            self._configurations.batch_size)
        self._requests.clear()

    @property
    def configurations(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Initial packed inputs or one tensor per input, in query order."""
        self.freeze()
        return self._configurations.values

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
        before = self.source.evaluation_stats
        chunks = []
        n_calls = 0
        for start in range(0, configurations.batch_size, batch_size):
            ids = torch.arange(start, min(start + batch_size,
                                          configurations.batch_size),
                               device=configurations.device)
            values = self.source.evaluate(
                configurations.index_select(ids))
            if values.shape[0] != ids.numel():
                raise ValueError(
                    'Source values should preserve the configuration batch')
            chunks.append(values)
            n_calls += 1
        values = torch.cat(chunks, dim=0)
        delta = self.source.evaluation_stats.delta(before)
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
        self.freeze()
        if self._values is None:
            self._values = self._evaluate_batches(self._configurations)
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
                configurations, kind=self._configurations.kind)
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
                index_selection: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """Scatters one planned Phi without retaining another tensor cache."""
        if isinstance(handle, bool) or not isinstance(handle, int):
            raise TypeError('`handle` should be int type')
        self.freeze()
        if (handle < 0) or (handle >= len(self._incidences)):
            raise ValueError('`handle` is outside the evaluation plan')
        self.freeze()
        ids, result_shape, output_spec, labels = self._incidences[handle]
        values = self.prepare_values()
        if index_selection is not None:
            selection = index_selection.to(ids.device).reshape(
                -1, len(result_shape))
            indices = tuple(selection.unbind(-1))
            ids = ids.reshape(result_shape)[indices]
            if labels is not None:
                labels = labels.reshape(result_shape)[indices]
            result_shape = tuple(index_selection.shape[:-1])
        selected = values.index_select(0, ids.to(values.device))
        if output_spec is None:
            return selected.reshape(
                (*result_shape, *selected.shape[1:]))
        selected = output_spec.validate_values(selected)
        if output_spec.scalar:
            return selected.reshape(result_shape)
        labels = labels.to(values.device)
        selected = selected.reshape(selected.shape[0], -1).gather(
            1, labels.unsqueeze(1)).squeeze(1)
        return selected.reshape(result_shape)


__all__ = ['ConfigurationMap']
