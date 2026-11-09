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
    available throughout the callback. :attr:`phi_indices` reconstructs each
    registered Phi from a value table with its original shape.
    :meth:`sample_indices` matches original sketch samples to the initial
    table without evaluating missing samples.

    Parameters
    ----------
    source : BaseSource
        Source supplying the shared evaluations.
    batch_size : int or None, optional
        Maximum configurations per source call. Default is 64. ``None``
        evaluates all missing configurations in one call.
    transform : callable, optional
        Called once with this map and returns values of the same shape.
    sketch_samples : torch.Tensor, sequence or ConfigurationBatch, optional
        Original sketch samples, used only by :meth:`sample_indices`.
        Input columns follow ``source.in_sites``. They do not add queries.

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
                 transform: Optional[Callable] = None, *,
                 sketch_samples: Optional[Union[
                     torch.Tensor, Sequence[torch.Tensor],
                     ConfigurationBatch]] = None) -> None:
        if not isinstance(source, BaseSource):
            raise TypeError('`source` should inherit from BaseSource')
        if (batch_size is not None) and (
                isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or (batch_size < 1)):
            raise ValueError('`batch_size` should be positive or None')
        if (transform is not None) and not callable(transform):
            raise TypeError('`transform` should be callable or None')
        if (sketch_samples is not None) and not isinstance(
                sketch_samples, ConfigurationBatch):
            sketch_samples = ConfigurationBatch(
                sketch_samples, kind=source.input_kind)
        self.source = source
        self._batch_size = batch_size
        self._transform = transform
        # Requests and gather indices share one frozen configuration table.
        self._requests = []
        self._incidences = []
        self._configurations = None
        self._phi_indices = None
        self._sketch_samples = sketch_samples
        self._sample_indices = None

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
    def phi_indices(self) -> Tuple[Tuple[torch.Tensor, ...], ...]:
        """
        Index tuples reconstructing the registered Phis in registration order.

        ``values[phi_indices[k]]`` returns the registered Phi with its original
        query shape, including any output-component selection. The same
        indices apply to another value table of the same shape and device,
        such as trainable values inside ``transform``. Indices are cached and
        should be treated as read-only. They do not apply the transformation.

        Returns
        -------
        tuple[tuple[torch.Tensor, ...], ...]
            One PyTorch index tuple per registered Phi. Every index tensor has
            the registered result shape and is on the value table's device.
            A partial registration uses the selected entries' shape.
        """
        if self._phi_indices is None:
            values = self.evaluate_source()
            selectors = []
            validated_shapes = set()
            for ids, shape, output_spec, labels in self._incidences:
                rows = ids.to(values.device).reshape(shape)
                indices = (rows,)
                if output_spec is not None:
                    out_shape = tuple(output_spec.out_shape)
                    if out_shape not in validated_shapes:
                        output_spec.validate_values(values)
                        validated_shapes.add(out_shape)
                    if output_spec.scalar:
                        if values.ndim == 2:
                            indices += (torch.zeros_like(rows),)
                    else:
                        outputs = output_spec.unflatten_labels(
                            labels.to(values.device))
                        indices += tuple(output.reshape(shape)
                                         for output in outputs.unbind(-1))
                selectors.append(indices)
            self._phi_indices = tuple(selectors)
        return self._phi_indices

    def sample_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Matches original sketch samples to the initial configuration table.

        Only samples already present in :attr:`configurations` are returned.
        No source values are evaluated and auxiliary :meth:`evaluate` queries
        do not change this match. Repeated sketch samples retain separate
        positions and may select the same value row.

        Returns
        -------
        value_rows, sample_rows : torch.Tensor
            Integer vectors of equal length on the configuration device.
            ``values[value_rows]`` corresponds to
            ``sketch_samples[sample_rows]``. ``sample_rows`` is increasing.
            Both vectors are empty when no original sample is present.
            The cached indices should be treated as read-only.

        Raises
        ------
        RuntimeError
            If ``sketch_samples`` was not supplied to the map.
        """
        if self._sketch_samples is None:
            raise RuntimeError('`sketch_samples` was not supplied to the map')
        self.freeze()
        if self._sample_indices is None:
            combined = _concatenate_configurations(
                (self._configurations, self._sketch_samples))
            unique = _SamplePool(combined).restrict(
                SiteRegion(range(combined.n_inputs)))
            count = self._configurations.batch_size
            lookup = torch.full((unique.n_unique,), -1, dtype=torch.long,
                                device=combined.device)
            lookup[unique.inverse_ids[:count]] = torch.arange(
                count, device=combined.device)
            rows = lookup[unique.inverse_ids[count:]]
            sample_rows = torch.nonzero(rows >= 0, as_tuple=True)[0]
            self._sample_indices = rows[sample_rows], sample_rows
        return self._sample_indices

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
        values = self.prepare_values()
        indices = self.phi_indices[handle]
        if index_selection is not None:
            result_shape = self._incidences[handle][1]
            selection = index_selection.to(indices[0].device).reshape(
                -1, len(result_shape))
            selection = tuple(selection.unbind(-1))
            indices = tuple(index[selection].reshape(
                index_selection.shape[:-1]) for index in indices)
        return values[indices]


__all__ = ['ConfigurationMap']
