"""Lazy Phi operators assembled from regions, sampled axes and sources."""

from typing import (Hashable, Optional, Protocol, Sequence, Tuple, Union,
                    runtime_checkable)

import torch

from tensorkrowch.decompositions.metrics import EvaluationStats
from tensorkrowch.decompositions.sketching.evaluations import (
    _EvaluationPlanBuilder,
    _EvaluationRequest,
    _EvaluationSession,
)
from tensorkrowch.decompositions.sketching.regions import (RegionSketch, Site,
                                                           SiteRegion)
from tensorkrowch.decompositions.sketching.specs import _OutputSpec
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 FiberTensorSource,
                                                 TensorSource)


AxisComponent = Tuple[Site, torch.Tensor]
PhiComponent = Union[RegionSketch, AxisComponent]


def _normalize_selection(
        shape: Sequence[int],
        index_selection: Optional[torch.Tensor],
        device: torch.device) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Returns flattened component indices and their requested result shape."""
    shape = tuple(shape)
    n_components = len(shape)
    if index_selection is None:
        grids = torch.meshgrid(*[
            torch.arange(dim, device=device) for dim in shape
        ], indexing='ij')
        selection = torch.stack(grids, dim=-1)
        return selection.reshape(-1, n_components), shape
    if not isinstance(index_selection, torch.Tensor):
        raise TypeError('`index_selection` should be torch.Tensor type or None')
    if index_selection.ndim < 1 or \
            index_selection.shape[-1] != n_components:
        raise ValueError(
            'The last `index_selection` dimension should match Phi axes')
    if index_selection.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError('`index_selection` should contain integers')
    selection = index_selection.to(device=device, dtype=torch.long)
    flat = selection.reshape(-1, n_components)
    for axis, dim in enumerate(shape):
        if torch.any(flat[:, axis] < 0) or torch.any(flat[:, axis] >= dim):
            raise ValueError(
                f'`index_selection` is out of bounds at Phi axis {axis}')
    return flat, tuple(selection.shape[:-1])


def _fiber_selection(
        shape: Sequence[int],
        axis: int,
        fixed_indices: Optional[torch.Tensor],
        device: torch.device) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Builds selections varying one Phi axis over fixed remaining indices."""
    shape = tuple(shape)
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError('`axis` should be int type')
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError('`axis` is out of bounds for Phi')
    n_fixed = len(shape) - 1
    if fixed_indices is None:
        if n_fixed:
            raise ValueError(
                '`fixed_indices` is required when Phi has other axes')
        fixed = torch.empty((1, 0), device=device, dtype=torch.long)
        fixed_shape = ()
    else:
        if not isinstance(fixed_indices, torch.Tensor):
            raise TypeError(
                '`fixed_indices` should be torch.Tensor type or None')
        if fixed_indices.ndim < 1 or fixed_indices.shape[-1] != n_fixed:
            raise ValueError(
                'The last `fixed_indices` dimension should match fixed axes')
        if fixed_indices.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64):
            raise TypeError('`fixed_indices` should contain integers')
        fixed_indices = fixed_indices.to(device=device, dtype=torch.long)
        fixed_shape = tuple(fixed_indices.shape[:-1])
        fixed = fixed_indices.reshape(-1, n_fixed)

    fixed_axes = [other for other in range(len(shape)) if other != axis]
    for column, other in enumerate(fixed_axes):
        if torch.any(fixed[:, column] < 0) or \
                torch.any(fixed[:, column] >= shape[other]):
            raise ValueError(
                f'`fixed_indices` is out of bounds at Phi axis {other}')
    selection = torch.empty(
        fixed.shape[0], shape[axis], len(shape),
        device=device,
        dtype=torch.long)
    selection[:, :, axis] = torch.arange(
        shape[axis], device=device).unsqueeze(0)
    for column, other in enumerate(fixed_axes):
        selection[:, :, other] = fixed[:, column].unsqueeze(1)
    return selection.reshape(*fixed_shape, shape[axis], len(shape)), \
        (*fixed_shape, shape[axis])


@runtime_checkable
class PhiView(Protocol):
    """Minimal lazy/materialized Phi interface consumed by later fitters."""

    def evaluate(self, index_selection: torch.Tensor) -> torch.Tensor:
        """Evaluates selected Phi entries."""

    def fiber(self,
              axis: int,
              fixed_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates one varying Phi axis."""

    def materialize(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Returns the complete represented Phi tensor."""


class _MaterializedPhi:
    """Materialized Phi tensor retaining its component-axis layout."""

    def __init__(self,
                 tensor: torch.Tensor,
                 layout: Sequence[Hashable]) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        layout = tuple(layout)
        if len(layout) != tensor.ndim:
            raise ValueError('`layout` should contain one item per Phi axis')
        self.tensor = tensor
        self.layout = layout

    @property
    def shape(self) -> Tuple[int, ...]:
        """Materialized Phi shape."""
        return tuple(self.tensor.shape)

    def evaluate(self, index_selection: torch.Tensor) -> torch.Tensor:
        """Gathers arbitrary entries from the materialized Phi."""
        selection, result_shape = _normalize_selection(
            self.shape, index_selection, self.tensor.device)
        values = self.tensor[tuple(
            selection[:, axis] for axis in range(selection.shape[1]))]
        return values.reshape(result_shape)

    def fiber(self,
              axis: int,
              fixed_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns fibers from the materialized tensor."""
        selection, _ = _fiber_selection(
            self.shape, axis, fixed_indices, self.tensor.device)
        return self.evaluate(selection)

    def materialize(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Returns the already-materialized tensor without a copy."""
        if batch_size is not None and (
                isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or batch_size < 1):
            raise ValueError('`batch_size` should be a positive integer or None')
        return self.tensor


class _SelectedPhiView:
    """Lazy selection over a parent Phi operator."""

    def __init__(self,
                 phi: 'PhiOperator',
                 index_selection: torch.Tensor) -> None:
        self.phi = phi
        self.index_selection = index_selection

    def materialize(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Evaluates only the rows represented by this view."""
        return self.phi._evaluate_selection(
            self.index_selection, batch_size=batch_size)

    def evaluate(self, index_selection: torch.Tensor) -> torch.Tensor:
        """Indexes the selected result after evaluating it."""
        tensor = self.materialize()
        materialized = _MaterializedPhi(tensor, range(tensor.ndim))
        return materialized.evaluate(index_selection)

    def fiber(self,
              axis: int,
              fixed_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns a fiber of the selected result."""
        tensor = self.materialize()
        materialized = _MaterializedPhi(tensor, range(tensor.ndim))
        return materialized.fiber(axis, fixed_indices)


class PhiOperator:
    """Lazy Cartesian Phi assembled from regional states and sampled axes."""

    def __init__(
            self,
            source: TensorSource,
            components: Sequence[PhiComponent],
            output_spec: _OutputSpec,
            *,
            input_sites: Optional[Sequence[Site]] = None,
            output_sites: Optional[Sequence[Site]] = None,
            input_kind: Optional[str] = None) -> None:
        if not isinstance(source, TensorSource):
            raise TypeError('`source` should implement TensorSource')
        if not isinstance(output_spec, _OutputSpec):
            raise TypeError('`output_spec` should be _OutputSpec type')
        if input_kind not in (None, 'indices', 'coordinates'):
            raise ValueError(
                "`input_kind` should be 'indices', 'coordinates' or None")

        default_input_sites = input_sites is None
        if input_sites is None:
            input_sites = output_spec.input_positions
        input_region = SiteRegion(input_sites)
        if len(input_region) != output_spec.n_input_sites or \
                len(input_region) != len(source.input_dim):
            raise ValueError(
                '`input_sites` should contain one site per source input')
        if output_sites is None:
            if output_spec.n_output_sites and not default_input_sites:
                raise ValueError(
                    '`output_sites` is required with custom input sites')
            output_sites = output_spec.positions
        output_region = SiteRegion(output_sites)
        if len(output_region) != output_spec.n_output_sites:
            raise ValueError(
                '`output_sites` should contain one site per output axis')
        declared_region = input_region.union(output_region)
        if len(declared_region) != \
                (len(input_region) + len(output_region)):
            raise ValueError('Input and output sites should be distinct')

        source_output_shape = source.output_shape
        if source_output_shape is not None:
            source_output_shape = tuple(source_output_shape)
            if output_spec.scalar:
                if source_output_shape not in ((), (1,)):
                    raise ValueError(
                        'The source output should match the scalar output spec')
            elif source_output_shape != output_spec.output_shape:
                raise ValueError(
                    'The source and output spec should have matching shapes')

        try:
            components = tuple(components)
        except TypeError as exc:
            raise TypeError('`components` should be a sequence') from exc
        if not components:
            raise ValueError('`components` should contain at least one Phi axis')
        normalized = []
        covered = set()
        shape = []
        layout = []
        for component in components:
            if isinstance(component, RegionSketch):
                component_sites = component.region.sites
                size = component.n_unique
                normalized_component = component
                layout_item = component.region
            else:
                if not isinstance(component, tuple) or len(component) != 2 or \
                        not isinstance(component[1], torch.Tensor):
                    raise TypeError(
                        'Each component should be RegionSketch or (site, values)')
                site, values = component
                SiteRegion((site,))
                if values.ndim < 1 or values.shape[0] < 1:
                    raise ValueError(
                        'Sampled axis values should have a non-empty first axis')
                if (values.is_floating_point() or values.is_complex()) and \
                        not torch.isfinite(values).all():
                    raise ValueError('Sampled axis values should be finite')
                component_sites = (site,)
                size = values.shape[0]
                normalized_component = (site, values)
                layout_item = site
            for site in component_sites:
                if site in covered:
                    raise ValueError(
                        'Phi components should cover every site only once')
                if not declared_region.contains(site):
                    raise ValueError(
                        'Every Phi component site should be declared')
                covered.add(site)
            normalized.append(normalized_component)
            shape.append(size)
            layout.append(layout_item)
        if covered != set(declared_region.sites):
            raise ValueError(
                'Phi components should cover every input and output site')

        self.source = source
        self.components = tuple(normalized)
        self.output_spec = output_spec
        self.input_sites = input_region.sites
        self.output_sites = output_region.sites
        self.input_kind = input_kind
        self.shape = tuple(shape)
        self.layout = tuple(layout)
        self._last_stats = None

    @property
    def last_stats(self) -> Optional[EvaluationStats]:
        """Stats from the latest convenience evaluation, if any."""
        return self._last_stats

    def _values_from_selection(
            self,
            selection: torch.Tensor) -> dict:
        """Expands component row ids to one value tensor per declared site."""
        values_by_site = {}
        for axis, component in enumerate(self.components):
            ids = selection[:, axis]
            if isinstance(component, RegionSketch):
                for site, values in zip(component.region, component.values):
                    values_by_site[site] = values.index_select(
                        0, ids.to(values.device)).to(self.source.device)
            else:
                site, values = component
                values_by_site[site] = values.index_select(
                    0, ids.to(values.device)).to(self.source.device)
        return values_by_site

    def _configuration_and_labels(
            self,
            selection: torch.Tensor
            ) -> Tuple[ConfigurationBatch, Optional[torch.Tensor]]:
        """Builds source inputs and flattened output labels for selections."""
        values_by_site = self._values_from_selection(selection)
        input_values = tuple(values_by_site[site] for site in self.input_sites)
        if self.input_kind is None:
            integer_dtypes = (
                torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            kind = 'indices' if all(
                value.ndim == 1 and value.dtype in integer_dtypes
                for value in input_values) else 'coordinates'
        else:
            kind = self.input_kind

        same_shape = all(
            value.shape[1:] == input_values[0].shape[1:]
            for value in input_values[1:])
        same_dtype = all(
            value.dtype == input_values[0].dtype
            for value in input_values[1:])
        if same_shape and same_dtype:
            configurations = ConfigurationBatch(
                torch.stack(input_values, dim=1), kind=kind)
        else:
            configurations = ConfigurationBatch(input_values, kind=kind)

        if self.output_spec.scalar:
            labels = None
        else:
            output_values = tuple(
                values_by_site[site] for site in self.output_sites)
            if any(value.ndim != 1 for value in output_values):
                raise ValueError('Output sites should contain scalar indices')
            output_indices = torch.stack(output_values, dim=1)
            labels = self.output_spec.flatten_labels(output_indices)
        return configurations, labels

    def _request(
            self,
            index_selection: Optional[torch.Tensor]) -> _EvaluationRequest:
        """Builds one flattened request without evaluating the source."""
        selection, result_shape = _normalize_selection(
            self.shape, index_selection, self.source.device)
        configurations, labels = self._configuration_and_labels(selection)
        return _EvaluationRequest(
            configurations=configurations,
            result_shape=result_shape,
            output_spec=self.output_spec,
            output_labels=labels)

    def configuration_batch(
            self,
            index_selection: Optional[torch.Tensor] = None
            ) -> ConfigurationBatch:
        """Builds source configurations and removes all output sites."""
        return self._request(index_selection).configurations

    def collect(
            self,
            builder: _EvaluationPlanBuilder,
            index_selection: Optional[torch.Tensor] = None) -> int:
        """Collects this Phi request in a shared evaluation-plan builder."""
        if not isinstance(builder, _EvaluationPlanBuilder):
            raise TypeError('`builder` should be _EvaluationPlanBuilder type')
        if builder.source is not self.source:
            raise ValueError('The builder and Phi should share the same source')
        return builder.collect(self._request(index_selection))

    def _evaluate_selection(
            self,
            index_selection: Optional[torch.Tensor],
            batch_size: Optional[int] = None) -> torch.Tensor:
        """Evaluates one selection through a temporary deduplicated session."""
        builder = _EvaluationPlanBuilder(self.source)
        handle = self.collect(builder, index_selection)
        session = _EvaluationSession(builder.freeze())
        result = session.result(handle, batch_size=batch_size)
        self._last_stats = session.stats
        return result

    def evaluate(self, index_selection: torch.Tensor) -> torch.Tensor:
        """Evaluates only explicitly selected Phi entries."""
        return self._evaluate_selection(index_selection)

    def select(self, index_selection: torch.Tensor) -> PhiView:
        """Returns a lazy view over explicitly selected Phi entries."""
        _normalize_selection(self.shape, index_selection, self.source.device)
        return _SelectedPhiView(self, index_selection)

    def fiber(self,
              axis: int,
              fixed_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates one Phi fiber, using source partial contractions if valid."""
        selection, result_shape = _fiber_selection(
            self.shape, axis, fixed_indices, self.source.device)
        normalized_axis = axis if axis >= 0 else axis + len(self.shape)
        component = self.components[normalized_axis]
        if isinstance(component, tuple) and \
                component[0] in self.input_sites and \
                isinstance(self.source, FiberTensorSource):
            base_selection = selection[..., 0, :]
            base_request = self._request(base_selection)
            if base_request.configurations.packed:
                site, values = component
                source_axis = self.input_sites.index(site)
                before = getattr(self.source, 'evaluation_stats', None)
                fiber = self.source.fiber(
                    base_request.configurations,
                    source_axis,
                    values.to(self.source.device))
                expected = (
                    base_request.configurations.batch_size,
                    self.shape[normalized_axis])
                if fiber.shape[:2] != expected:
                    raise ValueError(
                        'The source fiber should preserve batch and value axes')
                after = getattr(self.source, 'evaluation_stats', None)
                if isinstance(before, EvaluationStats) and \
                        isinstance(after, EvaluationStats):
                    delta = after.delta(before)
                    batches = delta.batches
                    source_calls = delta.source_calls
                else:
                    batches = 1
                    source_calls = 1
                n_points = expected[0] * expected[1]
                self._last_stats = EvaluationStats(
                    requested_points=n_points,
                    unique_points=n_points,
                    batches=batches,
                    source_calls=source_calls)
                flat = fiber.reshape(
                    expected[0] * expected[1], *fiber.shape[2:])
                flat = self.output_spec.validate_values(flat)
                if self.output_spec.scalar:
                    return flat.reshape(result_shape)
                labels = base_request.output_labels.to(fiber.device)
                labels = labels.unsqueeze(1).expand(
                    -1, expected[1]).reshape(-1)
                flat = flat.reshape(flat.shape[0], -1).gather(
                    1, labels.unsqueeze(1)).squeeze(1)
                return flat.reshape(result_shape)
        return self._evaluate_selection(selection)

    def materialize(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Materializes the complete Cartesian Phi tensor."""
        result = self._evaluate_selection(None, batch_size=batch_size)
        return _MaterializedPhi(result, self.layout).materialize()


__all__ = ['PhiView', 'PhiOperator']
