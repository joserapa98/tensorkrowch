"""
This script contains:

    Internal classes:
        * _CompositeCoordinateMap

    Public classes:
        * QuantizedLayout
        * CoordinateMap
        * UniformCoordinateMap
        * WarpedCoordinateMap
        * ExplicitGridMap

    Internal functions:
        * _integer_spec
        * _coordinate_tensor
        * _domain_tensor
        * _out_of_domain
        * _indices_to_unit
        * _unit_to_indices

    Aliases:
        * IntegerSpec, DigitSite, Domain
"""

from dataclasses import dataclass
from typing import (Callable, Optional, Protocol, Sequence, Tuple, Union,
                    runtime_checkable)

import torch

from tensorkrowch.utils import _INTEGER_DTYPES


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
    >>> layout = tk.formats.QuantizedLayout(
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
        """Normalizes integer specifications and validates the digit schedule."""
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

        # Canonical digit identities stay fixed when the site order changes.
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

        # Freeze normalized specifications rather than the original containers.
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
    def in_dim(self) -> Tuple[int, ...]:
        """Basis input dimension of every digit site in schedule order."""
        return tuple(self.base[variable] for variable, _ in self.sites())


    def _variable_digits(self, variable: int) -> Tuple[int, ...]:
        """Canonical digit ids in the requested within-variable direction."""
        digits = tuple(range(self.level[variable]))
        return digits if self.digit_order == 'coarse_to_fine' \
            else tuple(reversed(digits))


    def sites(self) -> Tuple[DigitSite, ...]:
        """Returns the scheduled canonical digit-site pairs.

        Returns
        -------
        tuple[tuple[int, int], ...]
            Pairs (variable, canonical_digit). Canonical digit zero is most
            significant; digit_order changes only the site schedule.
        """
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
        if values.ndim < 1 or values.dtype not in _INTEGER_DTYPES:
            raise TypeError(f'`{name}` should be an integer tensor')
        return values.to(dtype=torch.long)


    def encode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Expands integer grid indices into scheduled digit columns.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape (*batch, n_variables), in [0,
            grid_size[variable] - 1].

        Returns
        -------
        torch.Tensor
            torch.long digits with shape (*batch, n_sites), on the input device.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(1, base=2, level=3)
        >>> layout.encode_indices(torch.tensor([[5]])).tolist()
        [[1, 0, 1]]
        """
        indices = self._integer_tensor(indices, 'indices')
        if indices.shape[-1] != self.n_variables:
            raise ValueError(
                'The last `indices` dimension should match `n_variables`')

        # Expand each index, then arrange its digits in the chosen site order.
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
        """Combines scheduled digits into original variable indices.

        Parameters
        ----------
        digits : torch.Tensor
            Integer digit configurations in layout schedule order, with shape
            (*data_batch, layout.n_sites). Every digit should lie within its
            site base.

        Returns
        -------
        torch.Tensor
            torch.long indices with shape (*batch, n_variables), on the input
            device.
        """
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
        # Combine digits by significance, independently of their site order.
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
        """Reorders digit columns without modifying any format cores.

        Parameters
        ----------
        digits : torch.Tensor
            Integer digit configurations in layout schedule order, with shape
            (*data_batch, layout.n_sites). Every digit should lie within its
            site base.
        target_ordering : str or QuantizedLayout
            Grouped/interleaved ordering name or a layout with the same
            variables, bases and levels. A custom ordering requires an explicit
            layout.

        Returns
        -------
        torch.Tensor
            Digits in the target schedule, preserving their represented variable
            indices.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(2, base=2, level=2)
        >>> digits = layout.encode_indices(torch.tensor([[1, 2]]))
        >>> reordered = layout.reorder_configurations(digits, 'interleaved')
        >>> target = tk.formats.QuantizedLayout(2, 2, 2, ordering='interleaved')
        >>> target.decode_digits(reordered).tolist()
        [[1, 2]]
        """
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
            indices.dtype not in _INTEGER_DTYPES:
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
        """Maps computational coordinates into physical space.

        Implementations preserve the coordinate shape. Inverse and grid-index
        methods are optional capabilities used for physical evaluation.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape (*batch, n_variables),
            expressed in the unit computational domain.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Domain metadata understood by the implementation. Interval-based
            maps use (2,) for a shared interval or (n_variables, 2) for separate
            intervals. The meaning of None depends on the concrete map.

        Returns
        -------
        torch.Tensor
            Physical coordinates with the same shape as the input.
        """


@dataclass(frozen=True)
class UniformCoordinateMap:
    """Affine map between a uniform computational grid and physical domains.

    Parameters
    ----------
    grid : {"endpoints", "cell_centers"}
        Grid convention used by from_indices and to_indices. Coordinate
        forward/inverse mappings remain affine for both conventions.
    """

    grid: str = 'endpoints'

    def __post_init__(self) -> None:
        """Validates the computational grid convention."""
        if self.grid not in ('endpoints', 'cell_centers'):
            raise ValueError(
                "`grid` should be 'endpoints' or 'cell_centers'")


    def forward(self,
                unit_coordinates: torch.Tensor,
                domain: Domain = None) -> torch.Tensor:
        """Maps computational coordinates into physical space.

        Uniform maps use an affine transformation of each physical interval.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape (*batch, n_variables),
            expressed in the unit computational domain.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Physical intervals as (2,) for a shared interval or (n_variables, 2)
            for separate intervals. Required for this affine map; None raises
            ValueError.

        Returns
        -------
        torch.Tensor
            Physical coordinates with the same shape as the input.
        """
        unit = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
        intervals = _domain_tensor(
            domain, unit.shape[-1], unit.device, unit.dtype)
        return intervals[:, 0] + unit * (intervals[:, 1] - intervals[:, 0])


    def inverse(self,
                physical_coordinates: torch.Tensor,
                domain: Domain = None,
                out_of_domain: str = 'error') -> torch.Tensor:
        """Maps physical coordinates back to the unit computational domain.

        Uses the inverse affine interval transformation.

        Parameters
        ----------
        physical_coordinates : torch.Tensor
            Finite floating physical coordinates with shape (*batch,
            n_variables).
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Physical intervals as (2,) for a shared interval or (n_variables, 2)
            for separate intervals. Required for this affine map; None raises
            ValueError.
        out_of_domain : {"error", "clip"}
            Whether coordinates outside the domain raise ValueError or are
            clipped to the domain boundary.

        Returns
        -------
        torch.Tensor
            Unit coordinates with the same shape and floating dtype as the
            input.
        """
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
        """Maps integer grid indices to physical point values.

        The grid policy chooses endpoint positions i / (size - 1) or cell
        centers (i + 0.5) / size.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape (*batch, n_variables), in [0,
            grid_size[variable] - 1].
        grid_size : sequence of int
            Number of grid points per physical variable.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Physical intervals as (2,) for a shared interval or (n_variables, 2)
            for separate intervals. Required for this affine map; None raises
            ValueError.

        Returns
        -------
        torch.Tensor
            Physical values with shape (*batch, n_variables), using the domain
            dtype when available and otherwise the default floating dtype.
        """
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
        """Quantizes physical coordinates to nearest grid indices.

        Parameters
        ----------
        physical_coordinates : torch.Tensor
            Finite floating physical coordinates with shape (*batch,
            n_variables).
        grid_size : sequence of int
            Number of grid points per physical variable.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Physical intervals as (2,) for a shared interval or (n_variables, 2)
            for separate intervals. Required for this affine map; None raises
            ValueError.
        out_of_domain : {"error", "clip"}
            Whether coordinates outside the domain raise ValueError or are
            clipped to the domain boundary.

        Returns
        -------
        torch.Tensor
            torch.long indices with shape (*batch, n_variables). Exact ties
            select the lower index.

        Examples
        --------
        >>> coordinate_map = tk.formats.UniformCoordinateMap()
        >>> domain = torch.tensor([0., 4.])
        >>> coordinate_map.to_indices(torch.tensor([[1.], [3.]]), (5,), domain).tolist()
        [[1], [3]]
        """
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

    Parameters
    ----------
    forward_function : callable
        Function (unit_coordinates, domain) returning finite floating
        physical coordinates of unchanged shape.
    inverse_function : callable, optional
        Function (physical_coordinates, domain) returning unit coordinates
        of unchanged shape. Without it, inverse evaluation raises
        NotImplementedError.

    Examples
    --------
    >>> coordinate_map = tk.formats.WarpedCoordinateMap(
    ...     lambda unit, domain: unit.square(),
    ...     lambda physical, domain: physical.sqrt())
    >>> unit = torch.tensor([[0.5]])
    >>> torch.equal(coordinate_map.inverse(coordinate_map.forward(unit)), unit)
    True
    """

    forward_function: Callable[[torch.Tensor, Domain], torch.Tensor]
    inverse_function: Optional[Callable[[torch.Tensor, Domain], torch.Tensor]] = None

    def __post_init__(self) -> None:
        """Checks that forward and optional inverse functions are callable."""
        if not callable(self.forward_function):
            raise TypeError('`forward_function` should be callable')
        if self.inverse_function is not None and not callable(
                self.inverse_function):
            raise TypeError('`inverse_function` should be callable or None')


    @staticmethod
    def _validate_result(result: torch.Tensor,
                         reference: torch.Tensor,
                         name: str) -> torch.Tensor:
        """Checks that a coordinate-map result preserves shape and finite values."""
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
        """Maps computational coordinates into physical space.

        Calls forward_function(coordinates, domain) and requires finite floating
        outputs of unchanged shape.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape (*batch, n_variables),
            expressed in the unit computational domain.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Domain metadata understood by the implementation. Warped maps pass
            it unchanged to the supplied callable; None may leave the physical
            geometry entirely within that callable.

        Returns
        -------
        torch.Tensor
            Physical coordinates with the same shape as the input.
        """
        unit = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
        return self._validate_result(
            self.forward_function(unit, domain), unit, 'forward_function')


    def inverse(self,
                physical_coordinates: torch.Tensor,
                domain: Domain = None,
                out_of_domain: str = 'error') -> torch.Tensor:
        """Maps physical coordinates back to the unit computational domain.

        Requires inverse_function; otherwise raises NotImplementedError. No
        numerical inverse is inferred.

        Parameters
        ----------
        physical_coordinates : torch.Tensor
            Finite floating physical coordinates with shape (*batch,
            n_variables).
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Domain metadata understood by the implementation. Warped maps pass
            it unchanged to the supplied callable; None may leave the physical
            geometry entirely within that callable.
        out_of_domain : {"error", "clip"}
            Whether coordinates outside the domain raise ValueError or are
            clipped to the domain boundary.

        Returns
        -------
        torch.Tensor
            Unit coordinates with the same shape and floating dtype as the
            input.
        """
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
    """Maps unit coordinates through arbitrary monotonic point grids.

    The forward map interpolates linearly between stored points. Inverse
    evaluation selects the nearest stored point, with ties resolved by its
    lower index. Increasing and decreasing grids are both supported.

    Parameters
    ----------
    points : torch.Tensor or sequence of torch.Tensor
        A shared strictly monotonic floating vector, or one per variable.
        Each grid should be finite and have at least two points. Physical
        intervals are specified by these points, so domain arguments should
        be None.
    """

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
        """Interpolates explicit grid points at unit coordinates.

        Explicit grids are linearly interpolated; unit values should lie in [0,
        1].

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape (*batch, n_variables),
            expressed in the unit computational domain.
        domain : None, optional
            Should be None: the physical grid points already define the domain.

        Returns
        -------
        torch.Tensor
            Physical coordinates with the same shape as the input.
        """
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
        """Finds nearest explicit grid points and returns their unit positions.

        Ties choose the lower stored index, also on descending grids.

        Parameters
        ----------
        physical_coordinates : torch.Tensor
            Finite floating physical coordinates with shape (*batch,
            n_variables).
        domain : None, optional
            Should be None: the physical grid points already define the domain.
        out_of_domain : {"error", "clip"}
            Whether coordinates outside the domain raise ValueError or are
            clipped to the domain boundary.

        Returns
        -------
        torch.Tensor
            Unit coordinates with the same shape and floating dtype as the
            input.
        """
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
        """Maps integer grid indices to physical point values.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape (*batch, n_variables), in [0,
            grid_size[variable] - 1].
        grid_size : sequence of int, optional
            Expected number of grid points per variable. None uses stored point
            counts; a supplied value should match them.
        domain : None, optional
            Should be None: the physical grid points already define the domain.

        Returns
        -------
        torch.Tensor
            Physical values with shape (*batch, n_variables), retaining the
            stored grid dtype.
        """
        if domain is not None:
            raise ValueError('`domain` is not used by ExplicitGridMap')
        if not isinstance(indices, torch.Tensor) or indices.ndim < 1 or \
                indices.dtype not in _INTEGER_DTYPES:
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
        """Quantizes physical coordinates to nearest grid indices.

        Parameters
        ----------
        physical_coordinates : torch.Tensor
            Finite floating physical coordinates with shape (*batch,
            n_variables).
        grid_size : sequence of int, optional
            Expected number of grid points per variable. None uses stored point
            counts; a supplied value should match them.
        domain : None, optional
            Should be None: the physical grid points already define the domain.
        out_of_domain : {"error", "clip"}
            Whether coordinates outside the domain raise ValueError or are
            clipped to the domain boundary.

        Returns
        -------
        torch.Tensor
            torch.long indices with shape (*batch, n_variables). Exact ties
            select the lower index.

        Examples
        --------
        >>> coordinate_map = tk.formats.ExplicitGridMap(torch.tensor([0., 1., 4.]))
        >>> coordinate_map.to_indices(torch.tensor([[2.5]])).tolist()
        [[1]]
        """
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
        """Normalizes one domain specification per physical variable."""
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
        """Maps each variable through its constituent forward coordinate map."""
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
        """Maps each physical variable back through its constituent inverse map."""
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
        """Maps each variable grid index through its constituent coordinate map."""
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
        """Quantizes each physical variable using its constituent coordinate map."""
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
