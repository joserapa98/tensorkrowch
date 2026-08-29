"""Quantized layouts and coordinate maps for recursive sketching."""

from dataclasses import dataclass
from typing import (Callable, Optional, Protocol, Sequence, Tuple, Union,
                    runtime_checkable)

import torch

from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 EmpiricalDistribution,
                                                 SparseTensorSource,
                                                 TensorSource)
from tensorkrowch.decompositions.sources.base import (
    _discrete_indices,
    _fiber_configurations,
    _SourceEvaluationTracker,
)


IntegerSpec = Union[int, Sequence[int]]
DigitSite = Tuple[int, int]
Domain = Optional[Union[torch.Tensor, Sequence[torch.Tensor]]]


def _integer_spec(value: IntegerSpec,
                  n_variables: int,
                  name: str,
                  minimum: int) -> Tuple[int, ...]:
    """Broadcasts one integer or validates one value per variable."""
    if isinstance(value, bool):
        raise TypeError(f'`{name}` should be int or a sequence of ints')
    if isinstance(value, int):
        values = (value,) * n_variables
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError(f'`{name}` should be int or a sequence of ints')
        try:
            values = tuple(value)
        except TypeError as exc:
            raise TypeError(
                f'`{name}` should be int or a sequence of ints') from exc
        if len(values) != n_variables:
            raise ValueError(
                f'`{name}` should contain one value per variable')
    if any(isinstance(item, bool) or not isinstance(item, int)
           for item in values):
        raise TypeError(f'`{name}` values should be integers')
    if any(item < minimum for item in values):
        qualifier = 'at least two' if minimum == 2 else 'positive'
        raise ValueError(f'`{name}` values should be {qualifier}')
    return values


@dataclass(frozen=True)
class QuantizedLayout:
    """Defines how multivariable integer indices are expanded into digits.

    A digit site is identified by ``(variable, digit)``. The digit coordinate
    is canonical and always runs from zero at the most significant (coarse)
    digit to ``level[variable] - 1`` at the least significant (fine) digit.
    ``digit_order`` only changes the TT site schedule.

    ``ordering="grouped"`` places all digits of each variable together.
    ``ordering="interleaved"`` cycles over variables at every available digit
    depth and omits variables whose levels are exhausted. A custom layout uses
    ``ordering="custom"`` and supplies every canonical digit site exactly once
    through ``permutation``.

    Changing the layout reorders configurations, not fitted TT cores. Moving
    non-neighboring cores would require explicit tensor swaps and possible
    rank truncation.

    Parameters
    ----------
    n_variables : int
        Number of independently quantized variables.
    base : int or sequence of int
        Digit base shared by all variables or specified per variable.
    level : int or sequence of int
        Number of digits shared by all variables or specified per variable.
    ordering : {``"grouped"``, ``"interleaved"``, ``"custom"``}
        Final digit-site schedule.
    digit_order : {``"coarse_to_fine"``, ``"fine_to_coarse"``}
        Direction in which each variable contributes its canonical digits.
    permutation : sequence of tuple[int, int], optional
        Complete custom schedule of ``(variable, canonical_digit)`` pairs.

    Examples
    --------
    >>> layout = QuantizedLayout(
    ...     n_variables=2, base=2, level=3, ordering='interleaved')
    >>> digits = layout.encode_indices(torch.tensor([[3, 5]]))
    >>> torch.equal(layout.decode_digits(digits), torch.tensor([[3, 5]]))
    True
    """

    n_variables: int
    base: IntegerSpec = 2
    level: IntegerSpec = 1
    ordering: str = 'grouped'
    digit_order: str = 'coarse_to_fine'
    permutation: Optional[Sequence[DigitSite]] = None

    def __post_init__(self) -> None:
        if isinstance(self.n_variables, bool) or not isinstance(
                self.n_variables, int):
            raise TypeError('`n_variables` should be int type')
        if self.n_variables < 1:
            raise ValueError('`n_variables` should be positive')
        base = _integer_spec(
            self.base, self.n_variables, 'base', minimum=2)
        level = _integer_spec(
            self.level, self.n_variables, 'level', minimum=1)
        if self.ordering not in ('grouped', 'interleaved', 'custom'):
            raise ValueError(
                "`ordering` should be 'grouped', 'interleaved' or 'custom'")
        if self.digit_order not in ('coarse_to_fine', 'fine_to_coarse'):
            raise ValueError(
                "`digit_order` should be 'coarse_to_fine' or "
                "'fine_to_coarse'")

        maximum = torch.iinfo(torch.long).max
        for variable, (site_base, site_level) in enumerate(zip(base, level)):
            if site_base ** site_level > maximum:
                raise OverflowError(
                    f'Quantized variable {variable} exceeds int64 indices')

        canonical = tuple(
            (variable, digit)
            for variable, variable_level in enumerate(level)
            for digit in range(variable_level))
        if self.ordering == 'custom':
            if self.permutation is None:
                raise ValueError(
                    '`permutation` is required for custom ordering')
            try:
                permutation = tuple(tuple(site) for site in self.permutation)
            except TypeError as exc:
                raise TypeError(
                    '`permutation` should contain digit-site pairs') from exc
            if any(len(site) != 2 or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in site) for site in permutation):
                raise TypeError(
                    '`permutation` should contain integer digit-site pairs')
            if len(permutation) != len(canonical) or \
                    set(permutation) != set(canonical):
                raise ValueError(
                    '`permutation` should contain every digit site once')
        elif self.permutation is not None:
            raise ValueError(
                '`permutation` is only valid with custom ordering')
        else:
            permutation = None

        object.__setattr__(self, 'base', base)
        object.__setattr__(self, 'level', level)
        object.__setattr__(self, 'permutation', permutation)

    @property
    def n_sites(self) -> int:
        """Total number of digit sites."""
        return sum(self.level)

    @property
    def grid_size(self) -> Tuple[int, ...]:
        """Number of representable integer indices per variable."""
        return tuple(site_base ** site_level for site_base, site_level in zip(
            self.base, self.level))

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Basis input dimension of every digit site in schedule order."""
        return tuple(self.base[variable] for variable, _ in self.sites())

    def _variable_digits(self, variable: int) -> Tuple[int, ...]:
        """Canonical digit ids in the requested within-variable direction."""
        digits = tuple(range(self.level[variable]))
        return digits if self.digit_order == 'coarse_to_fine' \
            else tuple(reversed(digits))

    def sites(self) -> Tuple[DigitSite, ...]:
        """Returns the ordered ``(variable, canonical_digit)`` schedule."""
        if self.ordering == 'custom':
            return tuple(self.permutation)
        variable_digits = tuple(
            self._variable_digits(variable)
            for variable in range(self.n_variables))
        if self.ordering == 'grouped':
            return tuple(
                (variable, digit)
                for variable, digits in enumerate(variable_digits)
                for digit in digits)
        return tuple(
            (variable, digits[depth])
            for depth in range(max(self.level))
            for variable, digits in enumerate(variable_digits)
            if depth < len(digits))

    @staticmethod
    def _integer_tensor(values: torch.Tensor, name: str) -> torch.Tensor:
        """Validates one integer tensor without changing its device."""
        if not isinstance(values, torch.Tensor):
            raise TypeError(f'`{name}` should be torch.Tensor type')
        if values.ndim < 1 or values.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64):
            raise TypeError(f'`{name}` should be an integer tensor')
        return values.to(dtype=torch.long)

    def encode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Encodes integer variable indices into scheduled digit columns."""
        indices = self._integer_tensor(indices, 'indices')
        if indices.shape[-1] != self.n_variables:
            raise ValueError(
                'The last `indices` dimension should match `n_variables`')
        canonical = {}
        for variable, (site_base, site_level, size) in enumerate(zip(
                self.base, self.level, self.grid_size)):
            values = indices[..., variable]
            if torch.any(values < 0) or torch.any(values >= size):
                raise ValueError(
                    f'`indices` is out of bounds for variable {variable}')
            for digit in range(site_level):
                stride = site_base ** (site_level - 1 - digit)
                canonical[(variable, digit)] = torch.remainder(
                    torch.div(values, stride, rounding_mode='floor'),
                    site_base)
        return torch.stack(
            [canonical[site] for site in self.sites()], dim=-1)

    def decode_digits(self, digits: torch.Tensor) -> torch.Tensor:
        """Decodes scheduled digits into one integer index per variable."""
        digits = self._integer_tensor(digits, 'digits')
        if digits.shape[-1] != self.n_sites:
            raise ValueError(
                'The last `digits` dimension should match the layout sites')
        canonical = {}
        for column, site in enumerate(self.sites()):
            variable, _ = site
            values = digits[..., column]
            if torch.any(values < 0) or torch.any(values >= self.base[variable]):
                raise ValueError(
                    f'`digits` is out of bounds at site {column}')
            canonical[site] = values
        indices = []
        for variable, (site_base, site_level) in enumerate(zip(
                self.base, self.level)):
            value = digits.new_zeros(digits.shape[:-1])
            for digit in range(site_level):
                stride = site_base ** (site_level - 1 - digit)
                value = value + canonical[(variable, digit)] * stride
            indices.append(value)
        return torch.stack(indices, dim=-1)

    def reorder_configurations(
            self,
            digits: torch.Tensor,
            target_ordering: Union[str, 'QuantizedLayout']
            ) -> torch.Tensor:
        """Reorders digit columns into another compatible layout."""
        if isinstance(target_ordering, str):
            target = QuantizedLayout(
                n_variables=self.n_variables,
                base=self.base,
                level=self.level,
                ordering=target_ordering,
                digit_order=self.digit_order)
        elif isinstance(target_ordering, QuantizedLayout):
            target = target_ordering
        else:
            raise TypeError(
                '`target_ordering` should be str or QuantizedLayout type')
        if target.n_variables != self.n_variables or \
                target.base != self.base or target.level != self.level:
            raise ValueError('Source and target layouts should be compatible')
        return target.encode_indices(self.decode_digits(digits))


def _coordinate_tensor(values: torch.Tensor, name: str) -> torch.Tensor:
    """Validates floating coordinates with a final variable dimension."""
    if not isinstance(values, torch.Tensor):
        raise TypeError(f'`{name}` should be torch.Tensor type')
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError(
            f'`{name}` should contain a final variable dimension')
    if not values.is_floating_point():
        raise TypeError(f'`{name}` should be floating')
    if not torch.isfinite(values).all():
        raise ValueError(f'`{name}` should contain finite values')
    return values


def _domain_tensor(domain: Domain,
                   n_variables: int,
                   device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
    """Normalizes one shared interval or one interval per variable."""
    if domain is None:
        raise ValueError('`domain` is required by this coordinate map')
    if isinstance(domain, torch.Tensor):
        intervals = domain
        if intervals.shape == (2,):
            intervals = intervals.expand(n_variables, 2)
        elif intervals.shape != (n_variables, 2):
            raise ValueError(
                '`domain` should be one interval or one per variable')
    else:
        if isinstance(domain, (str, bytes)):
            raise TypeError('`domain` should contain interval tensors')
        try:
            values = tuple(domain)
        except TypeError as exc:
            raise TypeError('`domain` should contain interval tensors') \
                from exc
        if len(values) != n_variables:
            raise ValueError('`domain` should contain one interval per variable')
        intervals = torch.stack([
            value if isinstance(value, torch.Tensor)
            else torch.as_tensor(value)
            for value in values
        ])
        if intervals.shape != (n_variables, 2):
            raise ValueError('Every domain interval should contain two values')
    intervals = intervals.to(device=device, dtype=dtype)
    if not torch.isfinite(intervals).all():
        raise ValueError('`domain` should contain finite values')
    if torch.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError('Every domain interval should be strictly increasing')
    return intervals


def _out_of_domain(value: str) -> str:
    """Validates the explicit out-of-domain policy."""
    if value not in ('error', 'clip'):
        raise ValueError("`out_of_domain` should be 'error' or 'clip'")
    return value


def _indices_to_unit(indices: torch.Tensor,
                     grid_size: Sequence[int],
                     grid: str,
                     dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Maps integer grid indices to computational coordinates."""
    if not isinstance(indices, torch.Tensor) or indices.ndim < 1 or \
            indices.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64):
        raise TypeError('`indices` should be an integer tensor')
    grid_size = tuple(grid_size)
    if indices.shape[-1] != len(grid_size):
        raise ValueError('`grid_size` should match the index variables')
    if any(isinstance(size, bool) or not isinstance(size, int) or size < 2
           for size in grid_size):
        raise ValueError('`grid_size` should contain integers of at least two')
    if dtype is None:
        dtype = torch.get_default_dtype()
    sizes = torch.tensor(grid_size, device=indices.device, dtype=dtype)
    values = indices.to(dtype=sizes.dtype)
    if torch.any(values < 0) or torch.any(values >= sizes):
        raise ValueError('`indices` is out of bounds for `grid_size`')
    if grid == 'endpoints':
        return values / (sizes - 1)
    if grid == 'cell_centers':
        return (values + 0.5) / sizes
    raise ValueError("`grid` should be 'endpoints' or 'cell_centers'")


def _unit_to_indices(unit_coordinates: torch.Tensor,
                     grid_size: Sequence[int],
                     grid: str,
                     out_of_domain: str) -> torch.Tensor:
    """Quantizes computational coordinates with lower-index tie breaking."""
    unit_coordinates = _coordinate_tensor(
        unit_coordinates, 'unit_coordinates')
    out_of_domain = _out_of_domain(out_of_domain)
    grid_size = tuple(grid_size)
    if unit_coordinates.shape[-1] != len(grid_size):
        raise ValueError('`grid_size` should match the coordinate variables')
    sizes = unit_coordinates.new_tensor(grid_size)
    outside = (unit_coordinates < 0) | (unit_coordinates > 1)
    if out_of_domain == 'error' and torch.any(outside):
        raise ValueError('Coordinates lie outside the computational domain')
    unit = unit_coordinates.clamp(0, 1)
    if grid == 'endpoints':
        scaled = unit * (sizes - 1)
    elif grid == 'cell_centers':
        scaled = unit * sizes - 0.5
    else:
        raise ValueError("`grid` should be 'endpoints' or 'cell_centers'")
    # ceil(x - 1/2) implements nearest integer with exact half-way values
    # assigned to the smaller index.
    return torch.ceil(scaled - 0.5).clamp_min(0).minimum(
        sizes - 1).to(torch.long)


@runtime_checkable
class CoordinateMap(Protocol):
    """Maps computational coordinates to a physical coordinate space."""

    def forward(self,
                unit_coordinates: torch.Tensor,
                domain: Domain = None) -> torch.Tensor:
        """Maps ``(..., variables)`` unit coordinates to physical values."""


@dataclass(frozen=True)
class UniformCoordinateMap:
    """Affine map between a uniform computational grid and physical domains."""

    grid: str = 'endpoints'

    def __post_init__(self) -> None:
        if self.grid not in ('endpoints', 'cell_centers'):
            raise ValueError(
                "`grid` should be 'endpoints' or 'cell_centers'")

    def forward(self,
                unit_coordinates: torch.Tensor,
                domain: Domain = None) -> torch.Tensor:
        """Maps unit coordinates affinely into each physical interval."""
        unit = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
        intervals = _domain_tensor(
            domain, unit.shape[-1], unit.device, unit.dtype)
        return intervals[:, 0] + unit * (intervals[:, 1] - intervals[:, 0])

    def inverse(self,
                physical_coordinates: torch.Tensor,
                domain: Domain = None,
                out_of_domain: str = 'error') -> torch.Tensor:
        """Maps physical coordinates back to the unit computational domain."""
        physical = _coordinate_tensor(
            physical_coordinates, 'physical_coordinates')
        intervals = _domain_tensor(
            domain, physical.shape[-1], physical.device, physical.dtype)
        unit = (physical - intervals[:, 0]) / (
            intervals[:, 1] - intervals[:, 0])
        policy = _out_of_domain(out_of_domain)
        outside = (unit < 0) | (unit > 1)
        if policy == 'error' and torch.any(outside):
            raise ValueError('Coordinates lie outside the physical domain')
        return unit.clamp(0, 1) if policy == 'clip' else unit

    def from_indices(self,
                     indices: torch.Tensor,
                     grid_size: Sequence[int],
                     domain: Domain = None) -> torch.Tensor:
        """Maps integer grid indices directly to physical coordinates."""
        dtype = None
        if isinstance(domain, torch.Tensor) and domain.is_floating_point():
            dtype = domain.dtype
        elif isinstance(domain, (list, tuple)) and domain and \
                isinstance(domain[0], torch.Tensor) and \
                domain[0].is_floating_point():
            dtype = domain[0].dtype
        unit = _indices_to_unit(
            indices, grid_size, self.grid, dtype=dtype)
        return self.forward(unit, domain)

    def to_indices(self,
                   physical_coordinates: torch.Tensor,
                   grid_size: Sequence[int],
                   domain: Domain = None,
                   out_of_domain: str = 'error') -> torch.Tensor:
        """Quantizes physical coordinates to their nearest grid indices."""
        unit = self.inverse(
            physical_coordinates,
            domain,
            out_of_domain=out_of_domain)
        return _unit_to_indices(
            unit, grid_size, self.grid, out_of_domain=out_of_domain)


@dataclass(frozen=True)
class WarpedCoordinateMap:
    """User-defined separable or coupled computational-coordinate map.

    Callables receive ``(coordinates, domain)`` and should preserve the input
    shape. ``domain`` may be ``None`` when the callable already contains the
    complete physical geometry.
    """

    forward_function: Callable
    inverse_function: Optional[Callable] = None

    def __post_init__(self) -> None:
        if not callable(self.forward_function):
            raise TypeError('`forward_function` should be callable')
        if self.inverse_function is not None and not callable(
                self.inverse_function):
            raise TypeError('`inverse_function` should be callable or None')

    @staticmethod
    def _validate_result(result: torch.Tensor,
                         reference: torch.Tensor,
                         name: str) -> torch.Tensor:
        if not isinstance(result, torch.Tensor):
            raise TypeError(f'`{name}` should return a torch.Tensor')
        if result.shape != reference.shape:
            raise ValueError(f'`{name}` should preserve coordinate shape')
        if not result.is_floating_point() or not torch.isfinite(result).all():
            raise ValueError(f'`{name}` should return finite floating values')
        return result

    def forward(self,
                unit_coordinates: torch.Tensor,
                domain: Domain = None) -> torch.Tensor:
        """Applies the user-defined forward warp."""
        unit = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
        return self._validate_result(
            self.forward_function(unit, domain), unit, 'forward_function')

    def inverse(self,
                physical_coordinates: torch.Tensor,
                domain: Domain = None,
                out_of_domain: str = 'error') -> torch.Tensor:
        """Applies the optional inverse warp and validates the unit result."""
        if self.inverse_function is None:
            raise NotImplementedError(
                'This warped coordinate map does not define an inverse')
        physical = _coordinate_tensor(
            physical_coordinates, 'physical_coordinates')
        unit = self._validate_result(
            self.inverse_function(physical, domain),
            physical,
            'inverse_function')
        policy = _out_of_domain(out_of_domain)
        outside = (unit < 0) | (unit > 1)
        if policy == 'error' and torch.any(outside):
            raise ValueError('Inverse warp lies outside the unit domain')
        return unit.clamp(0, 1) if policy == 'clip' else unit


class ExplicitGridMap:
    """Maps unit coordinates through arbitrary monotonic point grids."""

    def __init__(self,
                 points: Union[torch.Tensor, Sequence[torch.Tensor]]) -> None:
        if isinstance(points, torch.Tensor):
            grids = (points,)
            self.shared = True
        else:
            if isinstance(points, (str, bytes)):
                raise TypeError('`points` should contain grid tensors')
            try:
                grids = tuple(points)
            except TypeError as exc:
                raise TypeError('`points` should contain grid tensors') from exc
            self.shared = False
        if not grids or not all(
                isinstance(grid, torch.Tensor) and grid.ndim == 1 and
                grid.shape[0] >= 2 and grid.is_floating_point() and
                torch.isfinite(grid).all()
                for grid in grids):
            raise ValueError(
                '`points` should contain finite floating vectors of size >= 2')
        for grid in grids:
            differences = grid[1:] - grid[:-1]
            if not (torch.all(differences > 0) or
                    torch.all(differences < 0)):
                raise ValueError('Every explicit grid should be monotonic')
        self.points = grids

    def _grids(self, n_variables: int) -> Tuple[torch.Tensor, ...]:
        """Broadcasts one shared grid or validates per-variable grids."""
        if self.shared:
            return self.points * n_variables
        if len(self.points) != n_variables:
            raise ValueError('`points` should contain one grid per variable')
        return self.points

    @property
    def grid_size(self) -> Tuple[int, ...]:
        """Stored point count per explicit grid before shared broadcasting."""
        return tuple(grid.shape[0] for grid in self.points)

    def forward(self,
                unit_coordinates: torch.Tensor,
                domain: Domain = None) -> torch.Tensor:
        """Linearly interpolates each explicit grid at unit coordinates."""
        if domain is not None:
            raise ValueError('`domain` is not used by ExplicitGridMap')
        unit = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
        if torch.any(unit < 0) or torch.any(unit > 1):
            raise ValueError('Unit coordinates should lie in [0, 1]')
        values = []
        for variable, grid in enumerate(self._grids(unit.shape[-1])):
            grid = grid.to(device=unit.device, dtype=unit.dtype)
            scaled = unit[..., variable] * (grid.shape[0] - 1)
            lower = scaled.floor().to(torch.long)
            upper = (lower + 1).clamp_max(grid.shape[0] - 1)
            fraction = scaled - lower
            values.append(
                grid[lower] * (1 - fraction) + grid[upper] * fraction)
        return torch.stack(values, dim=-1)

    def inverse(self,
                physical_coordinates: torch.Tensor,
                domain: Domain = None,
                out_of_domain: str = 'error') -> torch.Tensor:
        """Maps to the nearest explicit point with lower-index tie breaking."""
        if domain is not None:
            raise ValueError('`domain` is not used by ExplicitGridMap')
        physical = _coordinate_tensor(
            physical_coordinates, 'physical_coordinates')
        policy = _out_of_domain(out_of_domain)
        values = []
        for variable, grid in enumerate(self._grids(physical.shape[-1])):
            grid = grid.to(device=physical.device, dtype=physical.dtype)
            coordinate = physical[..., variable]
            lower_bound = grid.min()
            upper_bound = grid.max()
            outside = (coordinate < lower_bound) | (coordinate > upper_bound)
            if policy == 'error' and torch.any(outside):
                raise ValueError('Coordinate lies outside an explicit grid')
            coordinate = coordinate.clamp(lower_bound, upper_bound)
            distances = (coordinate.unsqueeze(-1) - grid).abs()
            index = distances.argmin(dim=-1)
            values.append(index.to(physical.dtype) / (grid.shape[0] - 1))
        return torch.stack(values, dim=-1)

    def from_indices(self,
                     indices: torch.Tensor,
                     grid_size: Optional[Sequence[int]] = None,
                     domain: Domain = None) -> torch.Tensor:
        """Selects explicit point values at integer grid indices."""
        if domain is not None:
            raise ValueError('`domain` is not used by ExplicitGridMap')
        if not isinstance(indices, torch.Tensor) or indices.ndim < 1 or \
                indices.dtype not in (
                    torch.uint8, torch.int8, torch.int16, torch.int32,
                    torch.int64):
            raise TypeError('`indices` should be an integer tensor')
        grids = self._grids(indices.shape[-1])
        if grid_size is not None and tuple(grid_size) != tuple(
                grid.shape[0] for grid in grids):
            raise ValueError('`grid_size` should match the explicit grids')
        values = []
        for variable, grid in enumerate(grids):
            index = indices[..., variable].to(torch.long)
            if torch.any(index < 0) or torch.any(index >= grid.shape[0]):
                raise ValueError('`indices` is out of bounds for explicit grid')
            values.append(grid.to(indices.device)[index])
        return torch.stack(values, dim=-1)

    def to_indices(self,
                   physical_coordinates: torch.Tensor,
                   grid_size: Optional[Sequence[int]] = None,
                   domain: Domain = None,
                   out_of_domain: str = 'error') -> torch.Tensor:
        """Selects nearest explicit point indices."""
        if domain is not None:
            raise ValueError('`domain` is not used by ExplicitGridMap')
        unit = self.inverse(
            physical_coordinates, out_of_domain=out_of_domain)
        sizes_tuple = tuple(
            grid.shape[0] for grid in self._grids(unit.shape[-1]))
        if grid_size is not None and tuple(grid_size) != sizes_tuple:
            raise ValueError('`grid_size` should match the explicit grids')
        sizes = unit.new_tensor(sizes_tuple)
        return torch.round(unit * (sizes - 1)).to(torch.long)


class _CompositeCoordinateMap:
    """Applies one independent coordinate map per physical variable."""

    def __init__(self, maps: Sequence[CoordinateMap]) -> None:
        self.maps = tuple(maps)
        if not self.maps or not all(
                isinstance(coordinate_map, CoordinateMap)
                for coordinate_map in self.maps):
            raise TypeError('Every coordinate map should implement CoordinateMap')

    @staticmethod
    def _domains(domain: Domain, n_variables: int) -> Tuple[Domain, ...]:
        if domain is None:
            return (None,) * n_variables
        if isinstance(domain, torch.Tensor):
            if domain.shape == (2,):
                return (domain,) * n_variables
            if domain.shape == (n_variables, 2):
                return tuple(domain[variable] for variable in range(n_variables))
            raise ValueError('`domain` should contain one interval per variable')
        values = tuple(domain)
        if len(values) != n_variables:
            raise ValueError('`domain` should contain one entry per variable')
        return values

    def forward(self,
                unit_coordinates: torch.Tensor,
                domain: Domain = None) -> torch.Tensor:
        unit = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
        if unit.shape[-1] != len(self.maps):
            raise ValueError('Coordinate variables should match coordinate maps')
        domains = self._domains(domain, len(self.maps))
        values = [
            coordinate_map.forward(
                unit[..., variable:variable + 1], domains[variable])
            for variable, coordinate_map in enumerate(self.maps)
        ]
        return torch.cat(values, dim=-1)

    def inverse(self,
                physical_coordinates: torch.Tensor,
                domain: Domain = None,
                out_of_domain: str = 'error') -> torch.Tensor:
        physical = _coordinate_tensor(
            physical_coordinates, 'physical_coordinates')
        if physical.shape[-1] != len(self.maps):
            raise ValueError('Coordinate variables should match coordinate maps')
        domains = self._domains(domain, len(self.maps))
        values = []
        for variable, coordinate_map in enumerate(self.maps):
            inverse = getattr(coordinate_map, 'inverse', None)
            if not callable(inverse):
                raise NotImplementedError(
                    f'Coordinate map {variable} does not define an inverse')
            values.append(inverse(
                physical[..., variable:variable + 1],
                domains[variable],
                out_of_domain=out_of_domain))
        return torch.cat(values, dim=-1)

    def from_indices(self,
                     indices: torch.Tensor,
                     grid_size: Sequence[int],
                     domain: Domain = None) -> torch.Tensor:
        domains = self._domains(domain, len(self.maps))
        values = []
        for variable, coordinate_map in enumerate(self.maps):
            kernel = getattr(coordinate_map, 'from_indices', None)
            if callable(kernel):
                value = kernel(
                    indices[..., variable:variable + 1],
                    (grid_size[variable],),
                    domains[variable])
            else:
                unit = _indices_to_unit(
                    indices[..., variable:variable + 1],
                    (grid_size[variable],),
                    'endpoints')
                value = coordinate_map.forward(unit, domains[variable])
            values.append(value)
        return torch.cat(values, dim=-1)

    def to_indices(self,
                   physical_coordinates: torch.Tensor,
                   grid_size: Sequence[int],
                   domain: Domain = None,
                   out_of_domain: str = 'error') -> torch.Tensor:
        domains = self._domains(domain, len(self.maps))
        values = []
        for variable, coordinate_map in enumerate(self.maps):
            kernel = getattr(coordinate_map, 'to_indices', None)
            if callable(kernel):
                value = kernel(
                    physical_coordinates[..., variable:variable + 1],
                    (grid_size[variable],),
                    domains[variable],
                    out_of_domain=out_of_domain)
            else:
                inverse = getattr(coordinate_map, 'inverse', None)
                if not callable(inverse):
                    raise NotImplementedError(
                        f'Coordinate map {variable} does not define an inverse')
                unit = inverse(
                    physical_coordinates[..., variable:variable + 1],
                    domains[variable],
                    out_of_domain=out_of_domain)
                value = _unit_to_indices(
                    unit,
                    (grid_size[variable],),
                    'endpoints',
                    out_of_domain)
            values.append(value)
        return torch.cat(values, dim=-1)


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
            output_shape: Optional[Sequence[int]] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[Union[str, torch.device]] = None,
            computational_grid: str = 'endpoints',
            out_of_domain: str = 'error') -> None:
        self._initialize_evaluation_stats()
        if not isinstance(layout, QuantizedLayout):
            raise TypeError('`layout` should be QuantizedLayout type')
        if coordinate_map is None:
            coordinate_map = UniformCoordinateMap(grid=computational_grid)
        elif not isinstance(coordinate_map, CoordinateMap):
            if isinstance(coordinate_map, (str, bytes)):
                raise TypeError(
                    '`coordinate_map` should implement CoordinateMap')
            coordinate_map = _CompositeCoordinateMap(tuple(coordinate_map))
        if source_space not in ('physical', 'indices', 'digits'):
            raise ValueError(
                "`source_space` should be 'physical', 'indices' or 'digits'")
        if computational_grid not in ('endpoints', 'cell_centers'):
            raise ValueError(
                "`computational_grid` should be 'endpoints' or "
                "'cell_centers'")
        out_of_domain = _out_of_domain(out_of_domain)
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type or None')
        if output_shape is not None:
            output_shape = tuple(output_shape)
            if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
                   for dim in output_shape):
                raise ValueError(
                    '`output_shape` should contain positive integers')

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
                    source_layout.n_variables != layout.n_variables:
                raise ValueError('Source and adapter layouts are incompatible')
            if tuple(source.input_dim) != source_layout.input_dim:
                raise ValueError(
                    'Digit source dimensions do not match `source_layout`')
        elif source_layout is not None:
            raise ValueError(
                '`source_layout` is only valid with `source_space="digits"`')
        elif is_source and source_space == 'indices':
            if tuple(source.input_dim) != layout.grid_size:
                raise ValueError(
                    'Indexed source dimensions should match layout grid sizes')
        elif is_source and len(source.input_dim) != layout.n_variables:
            raise ValueError(
                'Physical source should contain one site per variable')

        if is_source:
            resolved_device = source.device
            if device is not None and torch.device(device) != resolved_device:
                raise ValueError('`source` and `device` should match')
            resolved_dtype = source.dtype if dtype is None else dtype
            if dtype is not None and source.dtype is not None and \
                    source.dtype != dtype:
                raise ValueError('`source` and `dtype` should match')
            if output_shape is None:
                output_shape = source.output_shape
        else:
            resolved_device = torch.device('cpu' if device is None else device)
            resolved_dtype = dtype

        self.source = source
        self.layout = layout
        self.coordinate_map = coordinate_map
        self.domain = domain
        self.source_space = source_space
        self.source_layout = source_layout
        self.computational_grid = computational_grid
        self.out_of_domain = out_of_domain
        self._device = resolved_device
        self._dtype = resolved_dtype
        self._output_shape = output_shape

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Basis dimension of every scheduled digit site."""
        return self.layout.input_dim

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """Declared or inferred physical-source output shape."""
        return self._output_shape

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Declared or inferred physical-source dtype."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device used for mapping and source evaluation."""
        return self._device

    def indices_to_physical(self, indices: torch.Tensor) -> torch.Tensor:
        """Maps one integer grid index per variable to physical coordinates."""
        kernel = getattr(self.coordinate_map, 'from_indices', None)
        if callable(kernel):
            return kernel(indices, self.layout.grid_size, self.domain)
        unit = _indices_to_unit(
            indices,
            self.layout.grid_size,
            self.computational_grid,
            dtype=torch.get_default_dtype())
        return self.coordinate_map.forward(unit, self.domain)

    def physical_to_indices(self,
                            physical_coordinates: torch.Tensor) -> torch.Tensor:
        """Quantizes physical coordinates into one index per variable."""
        kernel = getattr(self.coordinate_map, 'to_indices', None)
        if callable(kernel):
            return kernel(
                physical_coordinates,
                self.layout.grid_size,
                self.domain,
                out_of_domain=self.out_of_domain)
        inverse = getattr(self.coordinate_map, 'inverse', None)
        if not callable(inverse):
            raise NotImplementedError(
                'Physical samples require a coordinate-map inverse')
        unit = inverse(
            physical_coordinates,
            self.domain,
            out_of_domain=self.out_of_domain)
        return _unit_to_indices(
            unit,
            self.layout.grid_size,
            self.computational_grid,
            self.out_of_domain)

    def physical_to_digits(self,
                           physical_coordinates: torch.Tensor) -> torch.Tensor:
        """Quantizes physical coordinates directly into scheduled digits."""
        return self.layout.encode_indices(
            self.physical_to_indices(physical_coordinates))

    def digits_to_physical(self, digits: torch.Tensor) -> torch.Tensor:
        """Decodes scheduled digits and maps them to physical coordinates."""
        return self.indices_to_physical(self.layout.decode_digits(digits))

    def _validate_values(self,
                         values: torch.Tensor,
                         batch_size: int) -> torch.Tensor:
        """Validates and infers output metadata after one source call."""
        if not isinstance(values, torch.Tensor):
            raise TypeError('`source` should return a torch.Tensor')
        if values.device != self.device:
            raise ValueError('`source` should return values on adapter device')
        if values.ndim < 1 or values.shape[0] != batch_size:
            raise ValueError(
                '`source` should preserve the leading batch dimension')
        if not (values.is_floating_point() or values.is_complex()):
            raise TypeError('`source` output should be floating or complex')
        output_shape = tuple(values.shape[1:])
        if self._output_shape is None:
            self._output_shape = output_shape
        elif output_shape != self._output_shape:
            raise ValueError('`source` output shape changed between calls')
        if self._dtype is None:
            self._dtype = values.dtype
        elif values.dtype != self._dtype:
            raise ValueError('`source` output dtype changed between calls')
        return values

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Evaluates scheduled digit configurations through the fixed adapter."""
        digits = _discrete_indices(
            configurations, self.input_dim, self.device)
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
                    physical, kind='coordinates'))
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
        if site < 0 or site >= len(self.input_dim):
            raise ValueError('`site` should identify a digit site')
        if values is None:
            values = torch.arange(
                self.input_dim[site], device=configurations.device)
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
            coordinate_map,
            domain,
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
            indices, input_dim=layout.grid_size, weights=weights)
        return cls(
            source,
            layout,
            coordinate_map,
            domain,
            source_space='indices')


__all__ = [
    'QuantizedLayout',
    'CoordinateMap',
    'UniformCoordinateMap',
    'WarpedCoordinateMap',
    'ExplicitGridMap',
    'QuantizedSourceAdapter',
]
