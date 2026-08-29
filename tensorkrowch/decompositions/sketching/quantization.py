"""Quantized layouts and coordinate maps for recursive sketching."""

from dataclasses import dataclass
from typing import (Callable, Optional, Protocol, Sequence, Tuple, Union,
                    runtime_checkable)

import torch


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

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Selects explicit point values at integer grid indices."""
        if not isinstance(indices, torch.Tensor) or indices.ndim < 1 or \
                indices.dtype not in (
                    torch.uint8, torch.int8, torch.int16, torch.int32,
                    torch.int64):
            raise TypeError('`indices` should be an integer tensor')
        values = []
        for variable, grid in enumerate(self._grids(indices.shape[-1])):
            index = indices[..., variable].to(torch.long)
            if torch.any(index < 0) or torch.any(index >= grid.shape[0]):
                raise ValueError('`indices` is out of bounds for explicit grid')
            values.append(grid.to(indices.device)[index])
        return torch.stack(values, dim=-1)

    def to_indices(self,
                   physical_coordinates: torch.Tensor,
                   out_of_domain: str = 'error') -> torch.Tensor:
        """Selects nearest explicit point indices."""
        unit = self.inverse(
            physical_coordinates, out_of_domain=out_of_domain)
        sizes = unit.new_tensor([
            grid.shape[0] for grid in self._grids(unit.shape[-1])])
        return torch.round(unit * (sizes - 1)).to(torch.long)


__all__ = [
    'QuantizedLayout',
    'CoordinateMap',
    'UniformCoordinateMap',
    'WarpedCoordinateMap',
    'ExplicitGridMap',
]
