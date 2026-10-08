"""Quantized source adaptation; structural layouts and maps live in formats."""

from math import prod
from typing import Optional, Sequence, Tuple, Union
import torch
from tensorkrowch.decompositions.sources.base import (ConfigurationBatch, TensorSource,
    _discrete_indices, _fiber_configurations, _SourceEvaluationTracker)
from tensorkrowch.decompositions.sources.sparse import EmpiricalDistribution, SparseTensorSource
from tensorkrowch.formats.quantization import (
    QuantizedLayout,
    CoordinateMap,
    AffineCoordinateMap,
    FunctionalCoordinateMap,
    ExplicitGridMap,
    _CompositeCoordinateMap,
    Domain)

class QuantizedSourceAdapter(_SourceEvaluationTracker):
    """Presents a physical or variable-index source on quantized digit sites.

    ``source_space="physical"`` maps decoded grid indices to physical
    coordinates before calling a function or coordinate-aware ``TensorSource``.
    ``source_space="indices"`` sends one decoded integer per variable to a
    discrete source. ``source_space="digits"`` is reserved for an already
    quantized source and requires explicit compatible ``source_layout``
    metadata; digit columns are reordered when the two layouts differ.

    Physical sparse support and empirical datasets use the class methods
    :meth:`from_physical_support` and :meth:`from_physical_dataset`. They
    quantize before constructing the sparse source, so collisions are
    coalesced by the existing sparse/empirical contracts.
    """

    def __init__(
            self,
            source,
            layout: QuantizedLayout,
            coordinate_map: Optional[
                Union[CoordinateMap, Sequence[CoordinateMap]]] = None,
            domain: Domain = None,
            *,
            source_space: str = 'physical',
            source_layout: Optional[QuantizedLayout] = None,
            out_shape: Optional[Sequence[int]] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[Union[str, torch.device]] = None,
            computational_grid: Union[str, float] = 'left',
            out_of_domain: str = 'error') -> None:
        self._initialize_evaluation_stats()
        if not isinstance(layout, QuantizedLayout):
            raise TypeError('`layout` should be QuantizedLayout type')
        if coordinate_map is None:
            intervals = torch.tensor([0., 1.]) if domain is None else domain
            coordinate_map = AffineCoordinateMap(
                intervals, layout.grid_size, grid_offset=computational_grid,
                out_of_domain=out_of_domain)
        else:
            if domain is not None:
                raise ValueError('`domain` belongs in the supplied `coordinate_map`')
            if isinstance(coordinate_map, (list, tuple)):
                coordinate_map = _CompositeCoordinateMap(coordinate_map)
            if not isinstance(coordinate_map, CoordinateMap):
                raise TypeError('`coordinate_map` should be CoordinateMap type')
        if coordinate_map.grid_size != layout.grid_size:
            raise ValueError('`coordinate_map.grid_size` should match `layout.grid_size`')
        if source_space not in ('physical', 'indices', 'digits'):
            raise ValueError(
                "`source_space` should be 'physical', 'indices' or 'digits'")
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type or None')
        if out_shape is not None:
            out_shape = tuple(out_shape)
            if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
                   for dim in out_shape):
                raise ValueError(
                    '`out_shape` should contain positive integers')

        is_source = isinstance(source, TensorSource)
        if not is_source and not callable(source):
            raise TypeError('`source` should be TensorSource type or callable')
        if not is_source and source_space != 'physical':
            raise ValueError('A raw callable should use `source_space="physical"`')
        if source_space == 'digits':
            if not is_source or not isinstance(source_layout, QuantizedLayout):
                raise ValueError(
                    'Digit sources require explicit `source_layout` metadata')
            if source_layout.base != layout.base or \
                    source_layout.level != layout.level or \
                    source_layout.n_coordinates != layout.n_coordinates:
                raise ValueError('Source and adapter layouts are incompatible')
            if tuple(source.in_dim) != source_layout.in_dim:
                raise ValueError(
                    'Digit source dimensions do not match `source_layout`')
        elif source_layout is not None:
            raise ValueError(
                '`source_layout` is only valid with `source_space="digits"`')
        elif is_source and source_space == 'indices':
            if tuple(source.in_dim) != layout.grid_size:
                raise ValueError(
                    'Indexed source dimensions should match layout grid sizes')
        elif is_source and len(source.in_dim) != layout.n_coordinates:
            raise ValueError(
                'Domain source should contain one site per coordinate')

        if is_source:
            resolved_device = source.device
            if device is not None and \
                    torch.empty(0, device=device).device != resolved_device:
                raise ValueError('`source` and `device` should match')
            resolved_dtype = source.dtype if dtype is None else dtype
            if dtype is not None and source.dtype is not None and \
                    source.dtype != dtype:
                raise ValueError('`source` and `dtype` should match')
            if out_shape is None:
                out_shape = source.out_shape
        else:
            resolved_device = torch.empty(
                0, device='cpu' if device is None else device).device
            resolved_dtype = dtype

        self.source = source
        self.layout = layout
        self.coordinate_map = coordinate_map
        self.source_space = source_space
        self.source_layout = source_layout
        self._device = resolved_device
        self._dtype = resolved_dtype
        self._out_shape = out_shape

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Basis dimension of every scheduled digit site."""
        return self.layout.in_dim

    @property
    def out_shape(self) -> Optional[Tuple[int, ...]]:
        """Declared or inferred physical-source output shape."""
        return self._out_shape

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Declared or inferred physical-source dtype."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device used for mapping and source evaluation."""
        return self._device

    @property
    def domain(self) -> Domain:
        """Domain owned by the coordinate map."""
        return getattr(self.coordinate_map, 'domain', None)

    @property
    def computational_grid(self) -> Union[str, float]:
        """Uniform grid offset stored by the coordinate map."""
        return getattr(self.coordinate_map, 'grid_offset', 'left')

    @property
    def out_of_domain(self) -> str:
        """Out-of-domain policy stored by the coordinate map."""
        return getattr(self.coordinate_map, 'out_of_domain', 'error')

    def indices_to_physical(self, indices: torch.Tensor) -> torch.Tensor:
        """Maps integer grid indices to coordinates in the domain."""
        return self.coordinate_map.from_indices(indices)

    def physical_to_indices(self,
                            domain_coordinates: torch.Tensor) -> torch.Tensor:
        """Quantizes coordinates in the domain to grid indices."""
        return self.coordinate_map.to_indices(domain_coordinates)

    def physical_to_digits(self,
                           domain_coordinates: torch.Tensor) -> torch.Tensor:
        """Quantizes physical coordinates directly into scheduled digits."""
        return self.layout.encode_indices(
            self.physical_to_indices(domain_coordinates))

    def digits_to_physical(self, digits: torch.Tensor) -> torch.Tensor:
        """Decodes scheduled digits and maps them to physical coordinates."""
        return self.indices_to_physical(self.layout.decode_digits(digits))

    def _validate_values(self,
                         values: torch.Tensor,
                         batch_size: int) -> torch.Tensor:
        """Validates and infers output metadata after one source call."""
        if not isinstance(values, torch.Tensor):
            raise TypeError('`source` should return a torch.Tensor')
        if values.device != self._device:
            raise ValueError('`source` should return values on adapter device')
        if values.ndim < 1 or values.shape[0] != batch_size:
            raise ValueError(
                '`source` should preserve the leading batch dimension')
        if not (values.is_floating_point() or values.is_complex()):
            raise TypeError('`source` output should be floating or complex')
        out_shape = tuple(values.shape[1:])
        if self._out_shape is None:
            self._out_shape = out_shape
        elif out_shape != self._out_shape:
            raise ValueError('`source` output shape changed between calls')
        if self._dtype is None:
            self._dtype = values.dtype
        elif values.dtype != self._dtype:
            raise ValueError('`source` output dtype changed between calls')
        return values

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Evaluates scheduled digit configurations through the fixed adapter."""
        digits = _discrete_indices(
            configurations, self.in_dim, self._device)
        indices = self.layout.decode_digits(digits)
        if self.source_space == 'digits':
            source_digits = self.layout.reorder_configurations(
                digits, self.source_layout)
            values = self.source.evaluate(ConfigurationBatch(
                source_digits, kind='indices'))
        elif self.source_space == 'indices':
            values = self.source.evaluate(ConfigurationBatch(
                indices, kind='indices'))
        else:
            physical = self.indices_to_physical(indices)
            if isinstance(self.source, TensorSource):
                values = self.source.evaluate(ConfigurationBatch(
                    physical, kind='features'))
            else:
                values = self.source(physical)
        values = self._validate_values(values, digits.shape[0])
        self._record_evaluation(points=digits.shape[0])
        return values

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates one digit fiber through the generic adapter path."""
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if site < 0 or site >= len(self.in_dim):
            raise ValueError('`site` should identify a digit site')
        if values is None:
            values = torch.arange(
                self.in_dim[site], device=configurations.device)
        expanded, n_values = _fiber_configurations(
            configurations, site, values)
        result = self.evaluate(expanded)
        return result.reshape(
            configurations.batch_size, n_values, *result.shape[1:])

    @classmethod
    def from_physical_support(
            cls,
            coordinates: torch.Tensor,
            values: torch.Tensor,
            layout: QuantizedLayout,
            coordinate_map: Optional[
                Union[CoordinateMap, Sequence[CoordinateMap]]] = None,
            domain: Domain = None,
            **kwargs) -> 'QuantizedSourceAdapter':
        """Quantizes physical sparse support and coalesces grid collisions."""
        provisional = cls(
            lambda data: data.new_zeros(data.shape[0]),
            layout,
            coordinate_map,
            domain,
            dtype=values.dtype,
            device=coordinates.device,
            **kwargs)
        indices = provisional.physical_to_indices(coordinates)
        source = SparseTensorSource(indices, values, layout.grid_size)
        return cls(
            source,
            layout,
            provisional.coordinate_map,
            source_space='indices')

    @classmethod
    def from_physical_dataset(
            cls,
            dataset: torch.Tensor,
            layout: QuantizedLayout,
            coordinate_map: Optional[
                Union[CoordinateMap, Sequence[CoordinateMap]]] = None,
            domain: Domain = None,
            weights: Optional[torch.Tensor] = None,
            **kwargs) -> 'QuantizedSourceAdapter':
        """Quantizes a physical dataset into an empirical grid distribution."""
        dtype = torch.get_default_dtype() if weights is None else weights.dtype
        provisional = cls(
            lambda data: data.new_zeros(data.shape[0], dtype=dtype),
            layout,
            coordinate_map,
            domain,
            dtype=dtype,
            device=dataset.device,
            **kwargs)
        indices = provisional.physical_to_indices(dataset)
        source = EmpiricalDistribution(
            indices, in_dim=layout.grid_size, weights=weights)
        return cls(
            source,
            layout,
            provisional.coordinate_map,
            source_space='indices')


__all__ = [
    'QuantizedLayout',
    'CoordinateMap',
    'AffineCoordinateMap',
    'FunctionalCoordinateMap',
    'ExplicitGridMap',
    'QuantizedSourceAdapter',
]


def _quantize_tensor(tensor: torch.Tensor, layout: QuantizedLayout,
                     n_batches: int = 0,
                     in_features: Optional[Sequence[int]] = None) -> torch.Tensor:
    """Reshapes raw variable axes and orders digits without padding or fitting."""
    if not isinstance(layout, QuantizedLayout):
        raise TypeError('`quantization` should be QuantizedLayout type')
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('`tensor` should be torch.Tensor type')
    if isinstance(n_batches, bool) or not isinstance(n_batches, int):
        raise TypeError('`n_batches` should be int type')
    if n_batches < 0 or n_batches >= tensor.ndim:
        raise ValueError('n_batches should leave at least one variable axis')
    features = tuple(range(n_batches, tensor.ndim)) if in_features is None else tuple(in_features)
    if any(isinstance(axis, bool) or not isinstance(axis, int) or
           not n_batches <= axis < tensor.ndim for axis in features) or len(set(features)) != len(features):
        raise ValueError('in_features should select distinct non-batch tensor axes')
    if tuple(tensor.shape[axis] for axis in features) != layout.grid_size:
        raise ValueError('Raw variable dimensions should match quantization.grid_size')
    if len(features) != tensor.ndim - n_batches:
        raise ValueError(
            'Quantics vectors require a digit at every site; '
            '`in_features` should include every non-batch axis')
    tensor = tensor.permute([*range(n_batches), *features])
    sites = [(coordinate, digit) for coordinate in range(layout.n_coordinates)
             for digit in range(layout.level[coordinate])]
    shape = [layout.base[coordinate] for coordinate, _ in sites]
    tensor = tensor.reshape(*tensor.shape[:n_batches], *shape)
    permutation = [*range(n_batches), *[n_batches + sites.index(site)
                                      for site in layout.sites()]]
    return tensor.permute(permutation)


def _quantize_matrix(tensor, in_dim, out_dim, axis_layout, quantization,
                     family):
    """Tensorizes matrix coordinate axes before the existing fused SVD engine."""
    from tensorkrowch.decompositions.svd._matrix import (
        _normalize_dim, _prepare_matrix_input)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError('`tensor` should be torch.Tensor type')
    if not isinstance(quantization, tuple) or len(quantization) != 2 or not all(
            isinstance(layout, QuantizedLayout) for layout in quantization):
        raise TypeError(
            'Matrix `quantization` should contain input and output layouts')
    in_layout, out_layout = quantization
    if in_layout.n_sites != out_layout.n_sites:
        raise ValueError('Matrix digit schedules should have equal lengths')
    if (in_dim is None) != (out_dim is None):
        raise ValueError('`in_dim` and `out_dim` should be provided together')
    if in_dim is not None and (
            _normalize_dim(in_dim, 'in_dim') != in_layout.grid_size or
            _normalize_dim(out_dim, 'out_dim') != out_layout.grid_size):
        raise ValueError(
            'Raw matrix dimensions should match quantized coordinate grids')
    if axis_layout not in ('interleaved', 'grouped'):
        raise ValueError('`layout` should be "interleaved" or "grouped"')

    # Coordinate counts may differ; digit-site counts must agree.
    in_shape, out_shape = in_layout.grid_size, out_layout.grid_size
    grouped_shape = in_shape + out_shape
    if tensor.ndim == 2 and tuple(tensor.shape) == (
            prod(in_shape), prod(out_shape)):
        grouped = tensor.reshape(grouped_shape)
    elif axis_layout == 'grouped' and tuple(tensor.shape) == grouped_shape:
        grouped = tensor
    elif axis_layout == 'interleaved' and len(in_shape) == len(out_shape):
        interleaved_shape = tuple(
            dim for pair in zip(in_shape, out_shape) for dim in pair)
        if tuple(tensor.shape) != interleaved_shape:
            raise ValueError('`tensor` shape should match the coordinate grids')
        grouped = tensor.permute(
            *range(0, tensor.ndim, 2), *range(1, tensor.ndim, 2))
    else:
        raise ValueError(
            '`tensor` should be a matrix or match the coordinate grids; '
            'unequal coordinate counts require `layout="grouped"`')

    # Split each coordinate into digits, then pair the two site schedules.
    sites = [(side, coordinate, digit)
             for side, layout in enumerate(quantization)
             for coordinate in range(layout.n_coordinates)
             for digit in range(layout.level[coordinate])]
    shape = [quantization[side].base[coordinate]
             for side, coordinate, _ in sites]
    schedule = [site for in_site, out_site in zip(
        in_layout.sites(), out_layout.sites())
                for site in ((0, *in_site), (1, *out_site))]
    interleaved = grouped.reshape(shape).permute(
        [sites.index(site) for site in schedule])
    return _prepare_matrix_input(
        interleaved, None, None, 'interleaved', family)


def _prepare_quantized_source(source, layout, *, in_dim=None, dtype=None,
                              device='cpu', batch_size=None, source_space=None,
                              coordinate_map=None, domain=None,
                              computational_grid='left', out_of_domain='error'):
    """Presents raw discrete or physical variables through the shared digit source."""
    from tensorkrowch.decompositions.sources.factory import as_tensor_source
    from tensorkrowch.formats.quantics import _QuanticsVector
    if not isinstance(layout, QuantizedLayout):
        raise TypeError('`quantization` should be QuantizedLayout type')
    source_layout = None
    raw_callable = callable(source) and not isinstance(source, TensorSource)
    if isinstance(source, _QuanticsVector):
        source_layout = source.layout
        source_space = 'digits' if source_space is None else source_space
        if source_space != 'digits':
            raise ValueError('Quantics formats should use source_space="digits"')
        coordinate_map = source.coordinate_map if coordinate_map is None else coordinate_map
        source = as_tensor_source(source)
        raw_callable = False
    elif source_space is None:
        source_space = 'physical' if raw_callable else 'indices'
    if in_dim is not None and tuple(in_dim) != layout.grid_size:
        raise ValueError('Raw in_dim should match quantization.grid_size')
    if not raw_callable or source_space != 'physical':
        source = as_tensor_source(source, in_dim=layout.grid_size,
                                  out_shape=(), dtype=dtype,
                                  device=device, batch_size=batch_size)
    return QuantizedSourceAdapter(
        source, layout, coordinate_map, domain, source_space=source_space,
        source_layout=source_layout, out_shape=(), dtype=dtype,
        device=device if raw_callable else None,
        computational_grid=computational_grid, out_of_domain=out_of_domain)


def _quantized_observations(observations, values, in_dim, weights, layout, *,
                            sample_space='indices', coordinate_map=None,
                            domain=None, computational_grid='left',
                            out_of_domain='error'):
    """Encodes completion rows while keeping values and weights paired."""
    from tensorkrowch.decompositions.als.problem import ObservedEntries
    if not isinstance(layout, QuantizedLayout):
        raise TypeError('`quantization` should be QuantizedLayout type')
    if sample_space not in ('indices', 'physical', 'digits'):
        raise ValueError('Invalid completion sample_space')
    if isinstance(observations, ObservedEntries):
        if values is not None or in_dim is not None or weights is not None:
            raise ValueError('Values, dimensions and weights belong inside ObservedEntries')
        if sample_space == 'physical':
            raise ValueError('ObservedEntries contains indices, not physical points')
        values, weights, in_dim = observations.values, observations.weights, observations.in_dim
        observations = observations.indices
    expected = layout.in_dim if sample_space == 'digits' else layout.grid_size
    if in_dim is not None and tuple(in_dim) != expected:
        raise ValueError('Observation dimensions should match the selected quantized space')
    if sample_space == 'physical':
        adapter = QuantizedSourceAdapter(
            lambda points: points.new_zeros(points.shape[0]), layout,
            coordinate_map, domain, device=observations.device,
            computational_grid=computational_grid, out_of_domain=out_of_domain)
        digits = adapter.physical_to_digits(observations)
    else:
        adapter = None
        digits = observations if sample_space == 'digits' else layout.encode_indices(observations)
        layout.decode_digits(digits)
    return ObservedEntries(digits, values, layout.in_dim, weights), adapter
