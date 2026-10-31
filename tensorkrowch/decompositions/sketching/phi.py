"""
This script contains:

    Classes:
        * PhiView
        * _SelectedPhiView
        * PhiOperator

    Functions:
        * _normalize_selection
        * _fiber_selection
"""

from math import prod
from typing import (Any,
                    Optional,
                    Protocol,
                    Sequence,
                    TYPE_CHECKING,
                    Tuple,
                    Union,
                    runtime_checkable)

import torch

from tensorkrowch.utils import _INTEGER_DTYPES

from tensorkrowch.decompositions.metrics import EvaluationStats
from tensorkrowch.decompositions.sketching.evaluations import (
    ConfigurationMap, _EvaluationPlanBuilder, _EvaluationRequest)
from tensorkrowch.decompositions.sketching.regions import (RegionSketch,
                                                           SiteRegion)
from tensorkrowch.decompositions.sketching.specs import _OutputSpec
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 BaseSource)


AxisComponent = Tuple[int, torch.Tensor]
PhiComponent = Union[RegionSketch, AxisComponent]


if TYPE_CHECKING:
    from tensorkrowch.decompositions.sketching.fitting import FittedInputAxis


def _normalize_selection(shape: Sequence[int],
                         index_selection: Optional[torch.Tensor],
                         device: torch.device
                         ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Returns flattened component indices and their requested result ``shape``.
    """
    shape = tuple(shape)
    n_components = len(shape)
    if index_selection is None:
        if not n_components:
            return torch.empty(1, 0, device=device, dtype=torch.long), ()
        grids = torch.meshgrid(*[
            torch.arange(dim, device=device) for dim in shape
        ], indexing='ij')
        selection = torch.stack(grids, dim=-1)
        return selection.reshape(-1, n_components), shape
    if not isinstance(index_selection, torch.Tensor):
        raise TypeError(
            '`index_selection` should be torch.Tensor type or None')
    if (index_selection.ndim < 1) or \
            (index_selection.shape[-1] != n_components):
        raise ValueError(
            'The last `index_selection` dimension should match Phi axes')
    if index_selection.dtype not in _INTEGER_DTYPES:
        raise TypeError('`index_selection` should contain integers')
    selection = index_selection.to(device=device, dtype=torch.long)
    flat = selection.reshape(prod(selection.shape[:-1]), n_components)
    for axis, dim in enumerate(shape):
        if torch.any(flat[:, axis] < 0) or torch.any(flat[:, axis] >= dim):
            raise ValueError(
                f'`index_selection` is out of bounds at Phi axis {axis}')
    return flat, tuple(selection.shape[:-1])


def _fiber_selection(shape: Sequence[int],
                     axis: int,
                     fixed_indices: Optional[torch.Tensor],
                     device: torch.device
                     ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Builds selections varying one Phi ``axis`` over fixed remaining indices.
    """
    shape = tuple(shape)
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError('`axis` should be int type')
    if axis < 0:
        axis += len(shape)
    if (axis < 0) or (axis >= len(shape)):
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
        if (fixed_indices.ndim < 1) or \
                (fixed_indices.shape[-1] != n_fixed):
            raise ValueError(
                'The last `fixed_indices` dimension should match fixed axes')
        if fixed_indices.dtype not in _INTEGER_DTYPES:
            raise TypeError('`fixed_indices` should contain integers')
        fixed_indices = fixed_indices.to(device=device, dtype=torch.long)
        fixed_shape = tuple(fixed_indices.shape[:-1])
        fixed = fixed_indices.reshape(prod(fixed_shape), n_fixed)

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
class PhiView(Protocol):  # MARK: PhiView
    """Minimal lazy/materialized Phi interface consumed by later fitters."""

    @property
    def shape(self) -> Tuple[int, ...]:
        """Complete represented Phi shape."""

    def evaluate(self, index_selection: torch.Tensor) -> torch.Tensor:
        """Evaluates selected Phi entries."""

    def fiber(self,
              axis: int,
              fixed_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates one varying Phi ``axis``."""

    def materialize(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Returns the complete represented Phi tensor."""


class _SelectedPhiView:  # MARK: _SelectedPhiView
    """Lazy selection over a parent Phi operator."""

    def __init__(self,
                 phi: 'PhiOperator',
                 index_selection: torch.Tensor) -> None:
        self.phi = phi
        self.index_selection = index_selection
        self.shape = tuple(index_selection.shape[:-1])

    def materialize(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Evaluates only the rows represented by this view."""
        if batch_size is not None:
            return self.phi._evaluate_selection(
                self.index_selection, batch_size=batch_size)
        return self.phi.evaluate(self.index_selection)

    def evaluate(self, index_selection: torch.Tensor) -> torch.Tensor:
        """Maps selected indices to the original Phi without materializing."""
        selection, result_shape = _normalize_selection(
            self.shape, index_selection, self.index_selection.device)
        if self.shape:
            indices = self.index_selection[tuple(selection.unbind(-1))]
        else:
            indices = self.index_selection.expand(
                *result_shape, self.index_selection.shape[-1])
        return self.phi.evaluate(indices).reshape(result_shape)

    def fiber(self,
              axis: int,
              fixed_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Maps a selected fiber to the original Phi."""
        selection, _ = _fiber_selection(
            self.shape, axis, fixed_indices, self.index_selection.device)
        return self.evaluate(selection)


class PhiOperator:  # MARK: PhiOperator
    """
    Local source evaluations indexed by regional states and explicit axes.

    Each component supplies one Phi axis: a :class:`RegionSketch` contributes
    its unique regional configurations; ``(site, values)`` contributes the
    supplied values of one input or output variable. Components cover each
    represented variable exactly once. ``shape`` and ``layout`` describe this
    original indexing throughout the operator's lifetime.

    :meth:`evaluate` and :meth:`fiber` can query selected entries without
    constructing the complete tensor. :meth:`materialize` fills ``tensor``,
    initially ``None``, and later queries reuse it. Clearing ``tensor`` drops
    that cache. Fitting or truncating an axis produces separate tensors and
    does not change the operator's coordinates.

    Parameters
    ----------
    source : BaseSource
        Source supplying the values. Structural source batches are not used
        by the sketching algorithms.
    components : sequence of RegionSketch or tuple
        Regional configurations or ``(site, values)`` pairs, in Phi axis order.
        Explicit values have shape ``(n_values, *feature_shape)``; output
        variables contain scalar integer indices.
    output_spec : _OutputSpec
        Output shape and placement in the represented chain.
    input_sites : sequence of int, optional
        Represented variables supplied to the source, in source input order.
        Defaults to the input positions in ``output_spec``.
    output_sites : sequence of int, optional
        Represented output variables, in source output-axis order. Required
        with custom ``input_sites`` when the source has output axes.
    input_kind : {``"indices"``, ``"features"``}, optional
        Meaning of input values. Inferred from the source or component values
        when omitted.
    """

    def __init__(self,
                 source: BaseSource,
                 components: Sequence[PhiComponent],
                 output_spec: _OutputSpec,
                 *,
                 input_sites: Optional[Sequence[int]] = None,
                 output_sites: Optional[Sequence[int]] = None,
                 input_kind: Optional[str] = None) -> None:
        if not isinstance(source, BaseSource):
            raise TypeError('`source` should inherit from BaseSource')
        if not isinstance(output_spec, _OutputSpec):
            raise TypeError('`output_spec` should be _OutputSpec type')
        if input_kind not in (None, 'indices', 'features'):
            raise ValueError(
                "`input_kind` should be 'indices', 'features' or None")

        default_input_sites = input_sites is None
        if input_sites is None:
            input_sites = output_spec.input_positions
        input_region = SiteRegion(input_sites)
        if (len(input_region) != output_spec.n_input_sites) or \
                (len(input_region) != len(source.in_sites)):
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

        source_out_shape = source.out_shape
        if source_out_shape is not None:
            source_out_shape = tuple(source_out_shape)
            if output_spec.scalar:
                if source_out_shape not in ((), (1,)):
                    raise ValueError(
                        'The source output should match the scalar spec')
            elif source_out_shape != output_spec.out_shape:
                raise ValueError(
                    'The source and output spec should have matching shapes')

        if isinstance(components, (str, bytes)) or \
                not isinstance(components, Sequence):
            raise TypeError('`components` should be a sequence')
        components = tuple(components)
        if not components:
            raise ValueError('`components` should contain at least one axis')
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
                if not isinstance(component, tuple) or \
                        (len(component) != 2) or \
                        not isinstance(component[1], torch.Tensor):
                    raise TypeError(
                        'Each component should be RegionSketch or '
                        '(site, values)')
                site, values = component
                SiteRegion((site,))
                if (values.ndim < 1) or (values.shape[0] < 1):
                    raise ValueError(
                        'Axis values should have a non-empty first axis')
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
        self.input_kind = (getattr(source, 'input_kind', None)
                           if input_kind is None else input_kind)
        self.shape = tuple(shape)
        self.layout = tuple(layout)
        self.tensor = None
        self._configuration_map = None
        self._handle = None
        self._last_stats = None

    @property
    def last_stats(self) -> Optional[EvaluationStats]:
        """Stats from the latest convenience evaluation, if any."""
        return self._last_stats

    def _values_from_selection(self, selection: torch.Tensor) -> dict:
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

    def _configuration_and_labels(self,
                                  selection: torch.Tensor) -> Tuple[
            ConfigurationBatch,
            Optional[torch.Tensor]]:
        """Builds source inputs and flattened output labels for selections."""
        values_by_site = self._values_from_selection(selection)
        input_values = tuple(values_by_site[site] for site in self.input_sites)
        if self.input_kind is None:
            kind = 'indices' if all(
                (value.ndim == 1) and (value.dtype in _INTEGER_DTYPES)
                for value in input_values) else 'features'
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

    def _request(self,
                 index_selection: Optional[torch.Tensor]
                 ) -> _EvaluationRequest:
        """Builds one flattened request without evaluating the source."""
        selection, result_shape = _normalize_selection(
            self.shape, index_selection, self.source.device)
        configurations, labels = self._configuration_and_labels(selection)
        return _EvaluationRequest(
            configurations=configurations,
            result_shape=result_shape,
            output_spec=self.output_spec,
            output_labels=labels)

    def configuration_batch(self,
                            index_selection: Optional[torch.Tensor] = None
                            ) -> ConfigurationBatch:
        """
        Returns the source configurations underlying the selected Phi entries.

        Parameters
        ----------
        index_selection : torch.Tensor, optional
            Integer indices with shape ``(*batch, len(shape))``, in Phi axis
            order. Selects the complete Cartesian Phi when omitted. Output
            variables select source output components and are absent from the
            returned input configurations.
        """
        return self._request(index_selection).configurations

    def collect(self,
                builder: _EvaluationPlanBuilder,
                index_selection: Optional[torch.Tensor] = None) -> int:
        """
        Collects this Phi request in a shared evaluation-plan ``builder``.

        Parameters
        ----------
        builder : _EvaluationPlanBuilder
            Unfrozen planner using the same source as this Phi.
        index_selection : torch.Tensor, optional
            Integer indices with shape ``(*batch, len(shape))``. Collects the
            complete Phi when omitted. Returns the request's integer handle.
        """
        if not isinstance(builder, _EvaluationPlanBuilder):
            raise TypeError('`builder` should be _EvaluationPlanBuilder type')
        if builder.source is not self.source:
            raise ValueError('The builder and Phi should share a source')
        return builder.collect(self._request(index_selection))

    def _bind(self, configuration_map: ConfigurationMap, handle: int) -> None:
        """Reuses a shared source table and its planned full-Phi gather."""
        self._configuration_map = configuration_map
        self._handle = handle
        self.tensor = None

    def _evaluate_selection(self,
                            index_selection: Optional[torch.Tensor],
                            batch_size: Optional[int] = None) -> torch.Tensor:
        """Gathers a cached selection or evaluates only its source queries."""
        if self.tensor is not None:
            if index_selection is None:
                return self.tensor
            selection, result_shape = _normalize_selection(
                self.shape, index_selection, self.tensor.device)
            return self.tensor[tuple(selection.unbind(-1))].reshape(
                result_shape)
        if self._configuration_map is not None:
            if index_selection is not None:
                _normalize_selection(self.shape, index_selection,
                                     self.source.device)
            result = self._configuration_map._result(
                self._handle, index_selection=index_selection)
            self._last_stats = self._configuration_map.stats
            return result
        builder = _EvaluationPlanBuilder(self.source)
        handle = self.collect(builder, index_selection)
        configuration_map = ConfigurationMap(builder.freeze(),
                                             batch_size=batch_size)
        result = configuration_map._result(handle)
        self._last_stats = configuration_map.stats
        return result

    def evaluate(self, index_selection: torch.Tensor) -> torch.Tensor:
        """
        Evaluates selected entries, using ``tensor`` when materialized.

        Parameters
        ----------
        index_selection : torch.Tensor
            Integer indices with shape ``(*batch, len(shape))``, in Phi axis
            order. The returned tensor has shape ``(*batch,)``.
        """
        return self._evaluate_selection(index_selection)

    def select(self, index_selection: torch.Tensor) -> PhiView:
        """
        Returns a lazy view over explicitly selected Phi entries.

        Parameters
        ----------
        index_selection : torch.Tensor
            Integer indices with shape ``(*batch, len(shape))``, in Phi axis
            order. The view has shape ``(*batch,)``.
        """
        _normalize_selection(self.shape, index_selection, self.source.device)
        return _SelectedPhiView(self, index_selection)

    def with_axis_values(self,
                         axis: int,
                         values: torch.Tensor) -> 'PhiOperator':
        """
        Returns a lazy Phi with new ``values`` on one explicit input ``axis``.

        Regional and output axes cannot be replaced. The returned operator
        owns no evaluated ``values`` and can therefore be collected into a new
        evaluation plan before that plan is frozen.

        Parameters
        ----------
        axis : int
            Explicit input axis to replace.
        values : torch.Tensor
            New values with shape ``(n_values, *feature_shape)``.
        """
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError('`axis` should be int type')
        if axis < 0:
            axis += len(self.shape)
        if (axis < 0) or (axis >= len(self.shape)):
            raise ValueError('`axis` is out of bounds for Phi')
        if not isinstance(values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        if (values.ndim < 1) or (values.shape[0] < 1):
            raise ValueError('`values` should contain a non-empty first axis')
        component = self.components[axis]
        if isinstance(component, RegionSketch) or \
                (component[0] not in self.input_sites):
            raise ValueError('Only explicit input axes can receive new values')
        components = list(self.components)
        components[axis] = (component[0], values)
        return PhiOperator(
            self.source,
            components,
            self.output_spec,
            input_sites=self.input_sites,
            output_sites=self.output_sites,
            input_kind=self.input_kind)

    def fiber_at(self,
                 axis: int,
                 values: torch.Tensor,
                 fixed_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluates an explicit input axis at new values.

        Parameters
        ----------
        axis : int
            Explicit input axis to vary.
        values : torch.Tensor
            Values with shape ``(n_values, *feature_shape)``.
        fixed_indices : torch.Tensor, optional
            Integer coordinates of the remaining axes, with shape
            ``(*batch, len(shape) - 1)``. Required when another axis exists.
            Returns shape ``(*batch, n_values)``.
        """
        return self.with_axis_values(axis, values).fiber(
            axis, fixed_indices=fixed_indices)

    def fit(self,
            axis: int,
            fitter: Any,
            domain: torch.Tensor,
            return_info: bool = False) -> 'FittedInputAxis':
        """
        Fits one explicit input axis using an input fitter.

        Parameters
        ----------
        axis : int
            Input axis to fit.
        fitter : InputFitter
            Strategy defining the finite input basis and its fitting method.
        domain : torch.Tensor
            Evaluated input values with shape ``(n_values, *feature_shape)``.
        return_info : bool, optional
            Whether to include fitting diagnostics in the returned
            ``FittedInputAxis``. Defaults to ``False``.
        """
        from tensorkrowch.decompositions.sketching.fitting import InputFitter

        if not isinstance(fitter, InputFitter):
            raise TypeError('`fitter` should implement InputFitter')
        return fitter.fit(
            self,
            axis=axis,
            domain=domain,
            return_info=return_info)

    def fiber(self,
              axis: int,
              fixed_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluates one Phi fiber, using source partial contractions if valid.

        Parameters
        ----------
        axis : int
            Phi axis to vary over all its values.
        fixed_indices : torch.Tensor, optional
            Integer coordinates of the remaining axes, in their original
            order, with shape ``(*batch, len(shape) - 1)``. Required when
            another axis exists. Returns shape ``(*batch, shape[axis])``.
        """
        selection, result_shape = _fiber_selection(
            self.shape, axis, fixed_indices, self.source.device)
        if (self.tensor is not None) or (self._configuration_map is not None):
            return self._evaluate_selection(selection)
        normalized_axis = axis if axis >= 0 else axis + len(self.shape)
        component = self.components[normalized_axis]
        if isinstance(component, tuple) and (component[0] in self.input_sites):
            base_selection = selection[..., 0, :]
            base_request = self._request(base_selection)
            if base_request.configurations.packed:
                site, values = component
                source_axis = self.source.in_sites[
                    self.input_sites.index(site)]
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
                        'The source fiber should preserve batch and '
                        'value axes')
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
        """
        Evaluates and caches the complete Phi without changing its axes.

        Parameters
        ----------
        batch_size : int, optional
            Maximum configurations per source call. Uses all requested points
            when omitted. A bound shared configuration map uses its own batch
            size. Existing cached tensors are returned without a copy.
        """
        if (batch_size is not None) and (
                isinstance(batch_size, bool) or
                not isinstance(batch_size, int)):
            raise TypeError('`batch_size` should be int type or None')
        if (batch_size is not None) and (batch_size < 1):
            raise ValueError('`batch_size` should be positive')
        if self.tensor is None:
            self.tensor = self._evaluate_selection(None, batch_size=batch_size)
        return self.tensor


__all__ = [
    'PhiView',
    'PhiOperator',
]
