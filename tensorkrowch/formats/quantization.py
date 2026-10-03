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
        * _smallest_level
        * _smallest_base
        * _coordinate_tensor
        * _domain_tensor
        * _out_of_domain
        * _grid_offset
        * _indices_to_unit
        * _unit_to_indices
        * _validate_explicit_grid

    Aliases:
        * IntegerSpec, CoordinateDigit, Domain

Terminology:
    * Coordinate: a continuous input value along one dimension.
    * Grid: the discrete values available along that dimension.
    * Index: an integer position on its grid.
    * Digit: one component of an index in the chosen base, used as a TT site
      input.
    * Layout: the order in which coordinates' digits occupy TT sites.

For example, index ``5`` on a base-``2`` grid with three digits becomes
``(1, 0, 1)``. A coordinate is first mapped to a grid index, then expanded
into digits; evaluating coordinates selects entries of the format without
interpolating between them.
"""

from dataclasses import dataclass
from typing import (Callable, Optional, Protocol, Sequence, Tuple, Union,
                    runtime_checkable)
import warnings

import torch

from tensorkrowch.utils import _INTEGER_DTYPES


IntegerSpec = Union[int, Sequence[int]]
CoordinateDigit = Tuple[int, int]
Domain = Optional[Union[torch.Tensor, Sequence[torch.Tensor]]]


def _integer_spec(value: IntegerSpec,
                  n_coordinates: int,
                  name: str,
                  minimum: int) -> Tuple[int, ...]:
    """Broadcasts one integer or validates one value per coordinate."""
    if isinstance(value, bool):
        raise TypeError(f'`{name}` should be int or a sequence of ints')
    if isinstance(value, int):
        values = (value,) * n_coordinates
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError(f'`{name}` should be int or a sequence of ints')
        try:
            values = tuple(value)
        except TypeError as exc:
            raise TypeError(
                f'`{name}` should be int or a sequence of ints') from exc
        if len(values) != n_coordinates:
            raise ValueError(
                f'`{name}` should contain one value per coordinate')
    if any(isinstance(item, bool) or not isinstance(item, int)
           for item in values):
        raise TypeError(f'`{name}` values should be integers')
    if any(item < minimum for item in values):
        qualifier = 'at least two' if minimum == 2 else 'positive'
        raise ValueError(f'`{name}` values should be {qualifier}')
    return values


def _smallest_level(size: int, base: int) -> int:
    """Returns the fewest digits in ``base`` that cover ``size`` indices."""
    level, capacity = 1, base
    while capacity < size:
        level += 1
        capacity *= base
    return level


def _smallest_base(size: int, level: int) -> int:
    """Returns the smallest base whose ``level`` digits cover ``size`` indices."""
    lower, upper = 2, size
    while lower < upper:
        middle = (lower + upper) // 2
        if middle ** level < size:
            lower = middle + 1
        else:
            upper = middle
    return lower


@dataclass(frozen=True)
class QuantizedLayout:
    """
    Defines how multivariable integer indices are expanded into digits.

    A digit site is identified by ``(coordinate, digit)``: ``coordinate``
    numbers the input dimensions, and ``digit`` numbers positions in the base
    expansion of that coordinate's grid index. For example, index ``5``
    is ``101`` in binary. Its leftmost digit selects the broad range ``4–7``;
    the next narrows it to ``4–5``, and the rightmost selects ``5``. Thus
    positions from left to right go from coarse to fine. ``digit`` numbers them
    from ``0`` to ``level[coordinate] - 1``; ``digit_order`` only changes which
    position appears first in the TT site schedule.

    ``ordering="grouped"`` places all digits of each coordinate together.
    ``ordering="interleaved"`` cycles over coordinates at every available digit
    depth and omits coordinates whose levels are exhausted. With
    ``ordering="custom"``, ``permutation`` lists the ``(coordinate, digit)``
    pairs in TT site order, using each pair exactly once.

    Changing the layout reorders configurations, not fitted TT cores. Moving
    non-neighboring cores would require explicit tensor swaps and possible rank
    truncation.

    Parameters
    ----------
    n_coordinates : int
        Number of independently quantized coordinates.
    base : int or sequence of int
        Digit base shared by all coordinates or specified per coordinate.
    level : int or sequence of int
        Number of digits shared by all coordinates or specified per
        coordinate.
    ordering : {"grouped", "interleaved", "custom"}
        Final digit-site schedule.
    digit_order : {"coarse_to_fine", "fine_to_coarse"}
        Direction of digit positions within each coordinate.
    permutation : sequence of tuple[int, int], optional
        Complete site schedule of ``(coordinate, digit)`` pairs when
        ``ordering="custom"``.

    Examples
    --------
    >>> layout = tk.formats.QuantizedLayout(
    ...     n_coordinates=2, base=2, level=3, ordering='interleaved')
    >>> digits = layout.encode_indices(torch.tensor([[3, 5]]))
    >>> digits.tolist()
    [[0, 1, 1, 0, 1, 1]]
    >>> layout.decode_digits(digits).tolist()
    [[3, 5]]
    """

    n_coordinates: int
    base: IntegerSpec = 2
    level: IntegerSpec = 1
    ordering: str = 'grouped'
    digit_order: str = 'coarse_to_fine'
    permutation: Optional[Sequence[CoordinateDigit]] = None

    def __post_init__(self) -> None:
        """Normalizes integer specifications and validates the digit schedule."""
        if isinstance(self.n_coordinates, bool) or not isinstance(
                self.n_coordinates, int):
            raise TypeError('`n_coordinates` should be int type')
        if self.n_coordinates < 1:
            raise ValueError('`n_coordinates` should be positive')
        base = _integer_spec(
            self.base, self.n_coordinates, 'base', minimum=2)
        level = _integer_spec(
            self.level, self.n_coordinates, 'level', minimum=1)
        if self.ordering not in ('grouped', 'interleaved', 'custom'):
            raise ValueError(
                "`ordering` should be 'grouped', 'interleaved' or 'custom'")
        if self.digit_order not in ('coarse_to_fine', 'fine_to_coarse'):
            raise ValueError(
                "`digit_order` should be 'coarse_to_fine' or "
                "'fine_to_coarse'")

        maximum = torch.iinfo(torch.long).max
        for coordinate, (site_base, site_level) in enumerate(zip(base, level)):
            if site_base ** site_level > maximum:
                raise OverflowError(
                    f'Quantized coordinate {coordinate} exceeds int64 indices')

        # Digit identities stay fixed when the site order changes.
        digit_sites = tuple(
            (coordinate, digit)
            for coordinate, coordinate_level in enumerate(level)
            for digit in range(coordinate_level))
        if self.ordering == 'custom':
            if self.permutation is None:
                raise ValueError(
                    '`permutation` is required for custom ordering')
            try:
                permutation = tuple(tuple(site) for site in self.permutation)
            except TypeError as exc:
                raise TypeError(
                    '`permutation` should contain `(coordinate, digit)` pairs'
                    ) from exc
            if any((len(site) != 2) or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in site) for site in permutation):
                raise TypeError(
                    '`permutation` should contain `(coordinate, digit)` pairs')
            if (len(permutation) != len(digit_sites)) or (
                    set(permutation) != set(digit_sites)):
                raise ValueError(
                    '`permutation` should contain every `(coordinate, digit)`'
                    ' pair once')
        elif self.permutation is not None:
            raise ValueError(
                '`permutation` is only valid with custom ordering')
        else:
            permutation = None

        # Freeze normalized specifications rather than the original containers.
        object.__setattr__(self, 'base', base)
        object.__setattr__(self, 'level', level)
        object.__setattr__(self, 'permutation', permutation)

    @classmethod
    def from_grid(cls,
                  n_coordinates: int,
                  grid: Union[IntegerSpec, 'ExplicitGridMap'],
                  *,
                  base: Optional[IntegerSpec] = None,
                  level: Optional[IntegerSpec] = None,
                  ordering: str = 'grouped',
                  digit_order: str = 'coarse_to_fine',
                  permutation: Optional[Sequence[CoordinateDigit]] = None
                  ) -> 'QuantizedLayout':
        """
        Constructs a layout from grid sizes and the supplied digit resolution.

        An integer or sequence of integers specifies a uniform grid size. When
        only ``base`` or ``level`` is given, the other is chosen to cover each
        grid with the smallest possible ``base ** level``. A larger grid is
        used when necessary, and a warning reports the changed size. An
        :class:`ExplicitGridMap` supplies coordinate values, which cannot be
        extended automatically: its sizes must match exactly.

        Parameters
        ----------
        n_coordinates : int
            Number of independently quantized coordinates.
        grid : int, sequence of int or ExplicitGridMap
            Uniform grid sizes or an explicit grid of coordinate values.
        base : int or sequence of int
            Digit base shared by all coordinates or specified per coordinate.
        level : int or sequence of int
            Number of digits shared by all coordinates or specified per
            coordinate.
        ordering : {"grouped", "interleaved", "custom"}
            Final digit-site schedule.
        digit_order : {"coarse_to_fine", "fine_to_coarse"}
            Direction of digit positions within each coordinate.
        permutation : sequence of tuple[int, int], optional
            Complete site schedule of ``(coordinate, digit)`` pairs when
            ``ordering="custom"``.

        Returns
        -------
        QuantizedLayout
            Layout whose ``grid_size`` matches the resulting grid.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout.from_grid(1, 8, base=2)
        >>> layout.level, layout.grid_size
        ((3,), (8,))
        """
        if base is None and level is None:
            raise ValueError('Specify `base` or `level` with `grid`')
        if isinstance(n_coordinates, bool) or not isinstance(n_coordinates, int) \
                or (n_coordinates < 1):
            raise ValueError('`n_coordinates` should be a positive integer')

        explicit = isinstance(grid, ExplicitGridMap)
        sizes = (tuple(value.shape[0] for value in grid._grids(n_coordinates))
                 if explicit else _integer_spec(grid, n_coordinates, 'grid', 2))
        bases = (_integer_spec(base, n_coordinates, 'base', 2)
                 if base is not None else None)
        levels = (_integer_spec(level, n_coordinates, 'level', 1)
                  if level is not None else None)

        if bases is None:
            bases = tuple(_smallest_base(size, coordinate_level)
                          for size, coordinate_level in zip(sizes, levels))
        if levels is None:
            levels = tuple(_smallest_level(size, coordinate_base)
                           for size, coordinate_base in zip(sizes, bases))

        result = cls(n_coordinates, bases, levels,
                     ordering, digit_order, permutation)

        if explicit and (result.grid_size != sizes):
            raise ValueError(
                '`base ** level` should match the explicit grid size')
        if base is not None and level is not None and (
            result.grid_size != sizes):
            raise ValueError('`base ** level` should match `grid`')
        if result.grid_size != sizes:
            warnings.warn(
                f'Uniform grid size changed from {sizes} to {result.grid_size} '
                 'to match `base ** level`', UserWarning, stacklevel=2)
        return result

    @property
    def grid_size(self) -> Tuple[int, ...]:
        """Number of representable integer indices per coordinate."""
        return tuple(site_base ** site_level for site_base, site_level in zip(
            self.base, self.level))

    @property
    def n_sites(self) -> int:
        """Total number of digit sites."""
        return sum(self.level)

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Basis input dimension of every digit site in schedule order."""
        return tuple(self.base[coordinate] for coordinate, _ in self.sites())

    def sites(self) -> Tuple[CoordinateDigit, ...]:
        """
        Returns the ``(coordinate, digit)`` pairs in site order.

        Returns
        -------
        tuple[tuple[int, int], ...]
            Pairs ``(coordinate, digit)``. Digit position zero is the leftmost
            in the index's base expansion; ``digit_order`` changes only the
            site order.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(
        ...     2, base=2, level=2, ordering='interleaved')
        >>> layout.sites()
        ((0, 0), (1, 0), (0, 1), (1, 1))
        """
        if self.ordering == 'custom':
            return tuple(self.permutation)
        coordinate_digits = tuple(
            range(level) if self.digit_order == 'coarse_to_fine'
            else range(level - 1, -1, -1)
            for level in self.level)
        if self.ordering == 'grouped':
            return tuple(
                (coordinate, digit)
                for coordinate, digits in enumerate(coordinate_digits)
                for digit in digits)
        return tuple(
            (coordinate, digits[depth])
            for depth in range(max(self.level))
            for coordinate, digits in enumerate(coordinate_digits)
            if depth < len(digits))

    @staticmethod
    def _integer_tensor(values: torch.Tensor, name: str) -> torch.Tensor:
        """Validates one integer tensor without changing its device."""
        if not isinstance(values, torch.Tensor):
            raise TypeError(f'`{name}` should be torch.Tensor type')
        if (values.ndim < 1) or values.dtype not in _INTEGER_DTYPES:
            raise TypeError(f'`{name}` should be an integer tensor')
        return values.to(dtype=torch.long)

    def encode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Expands integer grid indices into scheduled digit columns.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape ``(*batch, n_coordinates)``; each
            value lies in ``[0, grid_size[coordinate] - 1]``.

        Returns
        -------
        torch.Tensor
            ``torch.long`` digits with shape ``(*batch, n_sites)``, on the
            input device.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(1, base=2, level=3)
        >>> layout.encode_indices(torch.tensor([[5]])).tolist()
        [[1, 0, 1]]
        """
        indices = self._integer_tensor(indices, 'indices')
        if indices.shape[-1] != self.n_coordinates:
            raise ValueError(
                'The last `indices` dimension should match `n_coordinates`')

        # Expand each index, then arrange its digits in the chosen site order.
        digit_values = {}
        for coordinate, (base, level, size) in enumerate(zip(
                self.base, self.level, self.grid_size)):
            values = indices[..., coordinate]
            if torch.any(values < 0) or torch.any(values >= size):
                raise ValueError(
                    f'`indices` is out of bounds for coordinate {coordinate}')
            for digit in range(level):
                stride = base ** (level - 1 - digit)
                digit_values[(coordinate, digit)] = torch.remainder(
                    torch.div(values, stride, rounding_mode='floor'),
                    base)
        return torch.stack(
            [digit_values[site] for site in self.sites()], dim=-1)

    def decode_digits(self, digits: torch.Tensor) -> torch.Tensor:
        """
        Combines scheduled digits into original coordinate indices.

        Parameters
        ----------
        digits : torch.Tensor
            Integer digit configurations in layout schedule order, with shape
            ``(*batch, n_sites)``. Every digit should lie within its site base.

        Returns
        -------
        torch.Tensor
            ``torch.long`` indices with shape ``(*batch, n_coordinates)``, on
            the input device.
        """
        digits = self._integer_tensor(digits, 'digits')
        if digits.shape[-1] != self.n_sites:
            raise ValueError(
                'The last `digits` dimension should match the layout sites '
                '(`n_sites`)')

        digit_values = {}
        for site, coordinate_digit in enumerate(self.sites()):
            coordinate, _ = coordinate_digit
            values = digits[..., site]
            if torch.any(values < 0) or torch.any(values >= self.base[coordinate]):
                raise ValueError(
                    f'`digits` is out of bounds at site {site}')
            digit_values[coordinate_digit] = values

        # Combine digits by significance, independently of their site order.
        indices = []
        for coordinate, (base, level) in enumerate(zip(self.base, self.level)):
            value = digits.new_zeros(digits.shape[:-1])
            for digit in range(level):
                stride = base ** (level - 1 - digit)
                value = value + digit_values[(coordinate, digit)] * stride
            indices.append(value)
        return torch.stack(indices, dim=-1)

    def reorder_configurations(
            self,
            digits: torch.Tensor,
            target_ordering: Union[str, 'QuantizedLayout']
    ) -> torch.Tensor:
        """
        Reorders digit columns without modifying any format cores.

        Parameters
        ----------
        digits : torch.Tensor
            Integer digit configurations in layout schedule order, with shape
            ``(*batch, n_sites)``. Every digit should lie within its site base.
        target_ordering : {"grouped", "interleaved"} or QuantizedLayout
            Grouped/interleaved ordering name or a layout with the same
            coordinates, bases and levels. A custom ordering requires an
            explicit layout.

        Returns
        -------
        torch.Tensor
            Digits in the target schedule, preserving their represented
            coordinate indices.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(2, base=2, level=2)
        >>> digits = layout.encode_indices(torch.tensor([[1, 2]]))
        >>> reordered = layout.reorder_configurations(digits, 'interleaved')
        >>> target = tk.formats.QuantizedLayout(2, 2, 2, ordering='interleaved')
        >>> target.decode_digits(reordered).tolist()
        [[1, 2]]
        """
        if isinstance(target_ordering, str) and target_ordering in (
                'grouped', 'interleaved'):
            target = QuantizedLayout(
                n_coordinates=self.n_coordinates,
                base=self.base,
                level=self.level,
                ordering=target_ordering,
                digit_order=self.digit_order)
        elif isinstance(target_ordering, QuantizedLayout):
            target = target_ordering
        else:
            raise TypeError(
                '`target_ordering` should be "grouped", "interleaved" or '
                'QuantizedLayout type')
        if (target.n_coordinates != self.n_coordinates) or (
                target.base != self.base) or (target.level != self.level):
            raise ValueError(
                '`target_ordering` should match the source coordinates, bases '
                'and levels')
        return target.encode_indices(self.decode_digits(digits))


def _coordinate_tensor(values: torch.Tensor, name: str) -> torch.Tensor:
    """Validates floating coordinates with a final coordinate dimension."""
    if not isinstance(values, torch.Tensor):
        raise TypeError(f'`{name}` should be torch.Tensor type')
    if values.ndim < 1 or values.shape[-1] < 1:
        raise ValueError(
            f'`{name}` should contain a final coordinate dimension')
    if not values.is_floating_point():
        raise TypeError(f'`{name}` should be floating')
    if not torch.isfinite(values).all():
        raise ValueError(f'`{name}` should contain finite values')
    return values


def _domain_tensor(domain: Domain,
                   n_coordinates: int,
                   device: torch.device,
                   dtype: torch.dtype) -> torch.Tensor:
    """Normalizes one shared interval or one interval per coordinate."""
    if domain is None:
        raise ValueError('`domain` is required by this coordinate map')
    if isinstance(domain, torch.Tensor):
        intervals = domain
        if intervals.shape == (2,):
            intervals = intervals.expand(n_coordinates, 2)
        elif intervals.shape != (n_coordinates, 2):
            raise ValueError(
                '`domain` should be one interval or one per coordinate')
    else:
        if isinstance(domain, (str, bytes)):
            raise TypeError('`domain` should contain interval tensors')
        try:
            values = tuple(domain)
        except TypeError as exc:
            raise TypeError('`domain` should contain interval tensors') \
                from exc
        if len(values) != n_coordinates:
            raise ValueError(
                '`domain` should contain one interval per coordinate')
        intervals = torch.stack([
            value if isinstance(value, torch.Tensor)
            else torch.as_tensor(value)
            for value in values
        ])
        if intervals.shape != (n_coordinates, 2):
            raise ValueError(
                'Every `domain` interval should contain two values')
    intervals = intervals.to(device=device, dtype=dtype)
    if not torch.isfinite(intervals).all():
        raise ValueError('`domain` should contain finite values')
    if torch.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError(
            'Every `domain` interval should be strictly increasing')
    return intervals


def _out_of_domain(value: str) -> str:
    """Validates the explicit out-of-domain policy."""
    if value not in ('error', 'clip'):
        raise ValueError("`out_of_domain` should be 'error' or 'clip'")
    return value


def _grid_offset(grid: Union[str, float]) -> Optional[float]:
    """Validates the grid convention and returns its within-cell offset."""
    if isinstance(grid, str):
        if grid == 'endpoints':
            return None
        offsets = {'left': 0., 'centers': 0.5, 'right': 1.}
        if grid in offsets:
            return offsets[grid]
        raise ValueError(
            "`grid` should be 'endpoints', 'left', 'centers', 'right' or "
            'a number between 0 and 1')

    if isinstance(grid, bool) or not isinstance(grid, (int, float)):
        raise TypeError('`grid` should be str or a number between 0 and 1')
    if not 0 <= grid <= 1:
        raise ValueError('A numeric `grid` should be between 0 and 1')
    return float(grid)


def _indices_to_unit(indices: torch.Tensor,
                     grid_size: Sequence[int],
                     grid: Union[str, float],
                     dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Maps integer grid indices to coordinates in the unit interval."""
    if not isinstance(indices, torch.Tensor) or (indices.ndim < 1) or \
            indices.dtype not in _INTEGER_DTYPES:
        raise TypeError('`indices` should be an integer tensor')
    grid_size = tuple(grid_size)
    if indices.shape[-1] != len(grid_size):
        raise ValueError('`grid_size` should match the index coordinates')
    if any(isinstance(size, bool) or not isinstance(size, int) or (size < 2)
           for size in grid_size):
        raise ValueError('`grid_size` should contain integers of at least two')

    if dtype is None:
        dtype = torch.get_default_dtype()
    sizes = torch.tensor(grid_size, device=indices.device, dtype=dtype)
    values = indices.to(dtype=sizes.dtype)
    if torch.any(values < 0) or torch.any(values >= sizes):
        raise ValueError('`indices` is out of bounds for `grid_size`')

    offset = _grid_offset(grid)
    if offset is None:
        return values / (sizes - 1)
    return (values + offset) / sizes


def _unit_to_indices(unit_coordinates: torch.Tensor,
                     grid_size: Sequence[int],
                     grid: Union[str, float],
                     out_of_domain: str) -> torch.Tensor:
    """Quantizes unit coordinates according to the grid convention."""
    unit_coordinates = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
    out_of_domain = _out_of_domain(out_of_domain)
    grid_size = tuple(grid_size)
    if unit_coordinates.shape[-1] != len(grid_size):
        raise ValueError('`grid_size` should contain one size per coordinate')

    sizes = unit_coordinates.new_tensor(grid_size)
    outside = (unit_coordinates < 0) | (unit_coordinates > 1)
    if (out_of_domain == 'error') and torch.any(outside):
        raise ValueError(
            '`unit_coordinates` lie outside the unit interval [0, 1]')

    unit = unit_coordinates.clamp(0, 1)
    offset = _grid_offset(grid)
    if offset == 0:
        indices = torch.floor(unit * sizes)
    elif offset == 1:
        indices = torch.ceil(unit * sizes) - 1
    else:
        scaled = (unit * (sizes - 1) if offset is None
                  else unit * sizes - offset)
        # Exact half-way values select the smaller index.
        indices = torch.ceil(scaled - 0.5)

    return indices.clamp_min(0).minimum(sizes - 1).to(torch.long)


@runtime_checkable
class CoordinateMap(Protocol):
    """Maps unit coordinates to coordinates in the domain."""

    def forward(self,
                unit_coordinates: torch.Tensor,
                domain: Domain = None) -> torch.Tensor:
        """
        Maps unit coordinates to coordinates in the domain.

        Implementations preserve the coordinate shape. Inverse and grid-index
        methods are optional capabilities used for evaluation at coordinates
        in the domain.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``,
            with each coordinate in ``[0, 1]``.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Domain metadata understood by the implementation. Interval-based
            maps use ``(2,)`` for a shared interval or ``(n_coordinates, 2)`` for
            separate intervals. The meaning of ``None`` depends on the concrete
            map.

        Returns
        -------
        torch.Tensor
            Coordinates in the domain with the same shape as the input.
        """


@dataclass(frozen=True)
class UniformCoordinateMap:
    """
    Affine map between a uniform grid and coordinates in the domain.

    With ``N`` grid coordinates, ``"endpoints"`` places them at
    ``i / (N - 1)``, including both domain boundaries. The other conventions
    use ``(i + offset) / N``: ``"left"``, ``"centers"`` and ``"right"``
    correspond to offsets ``0``, ``0.5`` and ``1``. A numeric ``grid`` selects
    any offset between ``0`` and ``1``.

    ``"left"`` assigns coordinates by interval using ``floor(N * x)``,
    with ``x=1`` assigned to the last interval, as in
    :func:`~tensorkrowch.embeddings.discretize`. ``"right"`` uses
    ``ceil(N * x) - 1``, with ``x=0`` assigned to the first interval.
    Numeric offsets ``0`` and ``1`` follow the same rules. For ``"endpoints"``
    and intermediate offsets, the nearest grid coordinate is selected;
    ties select the lower index. Here ``x`` is a unit coordinate
    in ``[0, 1]``.

    Parameters
    ----------
    grid : {"endpoints", "left", "centers", "right"} or float
        Grid convention used by :meth:`from_indices` and :meth:`to_indices`.
        Numeric offsets should be between ``0`` and ``1``. Coordinate
        forward/inverse mappings remain affine for all conventions.

    Examples
    --------
    >>> coordinate_map = tk.formats.UniformCoordinateMap(grid='left')
    >>> domain = torch.tensor([0., 1.])
    >>> coordinate_map.from_indices(
    ...     torch.arange(4).unsqueeze(-1), (4,), domain).tolist()
    [[0.0], [0.25], [0.5], [0.75]]
    >>> coordinate_map.to_indices(
    ...     torch.tensor([[0.2], [0.75], [1.]]), (4,), domain).tolist()
    [[0], [3], [3]]
    >>> coordinate_map = tk.formats.UniformCoordinateMap(grid=0.25)
    >>> coordinate_map.from_indices(
    ...     torch.arange(4).unsqueeze(-1), (4,), domain).tolist()
    [[0.0625], [0.3125], [0.5625], [0.8125]]
    """

    grid: Union[str, float] = 'endpoints'

    def __post_init__(self) -> None:
        """Validates the computational grid convention."""
        _grid_offset(self.grid)

    def forward(self,
                unit_coordinates: torch.Tensor,
                domain: Domain = None) -> torch.Tensor:
        """
        Maps unit coordinates to coordinates in the domain.

        Uniform maps use an affine transformation of each domain interval.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``,
            with each coordinate in ``[0, 1]``.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Domain intervals as ``(2,)`` for a shared interval or
            ``(n_coordinates, 2)`` for separate intervals. Required for this
            affine map; ``None`` raises ``ValueError``.

        Returns
        -------
        torch.Tensor
            Coordinates in the domain with the same shape as the input.

        Examples
        --------
        >>> coordinate_map = tk.formats.UniformCoordinateMap()
        >>> domain = torch.tensor([-2., 2.])
        >>> unit_coordinates = torch.tensor([[0.], [0.5], [1.]])
        >>> coordinate_map.forward(unit_coordinates, domain).tolist()
        [[-2.0], [0.0], [2.0]]
        """
        unit = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
        intervals = _domain_tensor(
            domain, unit.shape[-1], unit.device, unit.dtype)
        return intervals[:, 0] + unit * (intervals[:, 1] - intervals[:, 0])

    def inverse(self,
                domain_coordinates: torch.Tensor,
                domain: Domain = None,
                out_of_domain: str = 'error') -> torch.Tensor:
        """
        Maps coordinates in the domain back to unit coordinates.

        Uses the inverse affine interval transformation.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates in the domain with shape
            ``(*batch, n_coordinates)``.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Domain intervals as ``(2,)`` for a shared interval or
            ``(n_coordinates, 2)`` for separate intervals. Required for this
            affine map; ``None`` raises ``ValueError``.
        out_of_domain : {"error", "clip"}
            Whether coordinates outside the domain raise ``ValueError`` or are
            clipped to the domain boundary.

        Returns
        -------
        torch.Tensor
            Unit coordinates with the same shape and floating dtype as the
            input.

        Examples
        --------
        >>> coordinate_map = tk.formats.UniformCoordinateMap()
        >>> domain = torch.tensor([-2., 2.])
        >>> domain_coordinates = torch.tensor([[-2.], [0.], [2.]])
        >>> coordinate_map.inverse(domain_coordinates, domain).tolist()
        [[0.0], [0.5], [1.0]]
        >>> coordinate_map.inverse(torch.tensor([[-3.], [3.]]), domain,
        ...                        out_of_domain='clip').tolist()
        [[0.0], [1.0]]
        """
        coordinates = _coordinate_tensor(
            domain_coordinates, 'domain_coordinates')
        intervals = _domain_tensor(
            domain, coordinates.shape[-1],
            coordinates.device, coordinates.dtype)
        unit = (coordinates - intervals[:, 0]) / (
            intervals[:, 1] - intervals[:, 0])
        policy = _out_of_domain(out_of_domain)
        outside = (unit < 0) | (unit > 1)
        if (policy == 'error') and torch.any(outside):
            raise ValueError(
                '`domain_coordinates` lie outside the domain')
        return unit.clamp(0, 1) if policy == 'clip' else unit

    def from_indices(self,
                     indices: torch.Tensor,
                     grid_size: Sequence[int],
                     domain: Domain = None) -> torch.Tensor:
        """
        Maps integer grid indices to coordinate values in the domain.

        Positions follow the ``grid`` convention described in
        :class:`UniformCoordinateMap`, then are mapped into ``domain``.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape ``(*batch, n_coordinates)``; each
            value lies in ``[0, grid_size[coordinate] - 1]``.
        grid_size : sequence of int
            Number of grid coordinates per coordinate.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Domain intervals as ``(2,)`` for a shared interval or
            ``(n_coordinates, 2)`` for separate intervals. Required for this
            affine map; ``None`` raises ``ValueError``.

        Returns
        -------
        torch.Tensor
            Values in the domain with shape ``(*batch, n_coordinates)``,
            using the domain dtype when available and otherwise the
            default floating dtype.

        Examples
        --------
        >>> coordinate_map = tk.formats.UniformCoordinateMap(grid='centers')
        >>> domain = torch.tensor([-2., 2.])
        >>> indices = torch.tensor([[0], [1], [2], [3]])
        >>> coordinate_map.from_indices(indices, (4,), domain).tolist()
        [[-1.5], [-0.5], [0.5], [1.5]]
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
                   domain_coordinates: torch.Tensor,
                   grid_size: Sequence[int],
                   domain: Domain = None,
                   out_of_domain: str = 'error') -> torch.Tensor:
        """
        Quantizes coordinates in the domain to grid indices.

        ``"left"`` and ``"right"`` assign coordinates by interval. Other
        conventions select the nearest grid coordinate, with ties assigned
        to the lower index. See :class:`UniformCoordinateMap` for the rules.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates in the domain with shape
            ``(*batch, n_coordinates)``.
        grid_size : sequence of int
            Number of grid coordinates per coordinate.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Domain intervals as ``(2,)`` for a shared interval or
            ``(n_coordinates, 2)`` for separate intervals. Required for this
            affine map; ``None`` raises ``ValueError``.
        out_of_domain : {"error", "clip"}
            Whether coordinates outside the domain raise ``ValueError`` or are
            clipped to the domain boundary.

        Returns
        -------
        torch.Tensor
            ``torch.long`` indices with shape ``(*batch, n_coordinates)``.

        Examples
        --------
        >>> coordinate_map = tk.formats.UniformCoordinateMap()
        >>> domain = torch.tensor([0., 4.])
        >>> coordinate_map.to_indices(torch.tensor([[1.], [3.]]),
        ...                           (5,), domain).tolist()
        [[1], [3]]
        """
        unit = self.inverse(
            domain_coordinates,
            domain,
            out_of_domain=out_of_domain)
        return _unit_to_indices(
            unit, grid_size, self.grid, out_of_domain=out_of_domain)


@dataclass(frozen=True)
class WarpedCoordinateMap:
    """
    User-defined separable or coupled map from unit coordinates to the domain.

    Callables receive ``(coordinates, domain)`` and should preserve the input
    shape. ``domain`` may be ``None`` when the callable already contains the
    complete domain geometry.

    Parameters
    ----------
    forward_function : callable
        Function ``(unit_coordinates, domain)`` returning finite floating
        coordinates in the domain of unchanged shape.
    inverse_function : callable, optional
        Function ``(domain_coordinates, domain)`` returning unit coordinates
        of unchanged shape. Without it, inverse evaluation raises
        ``NotImplementedError``.

    Examples
    --------
    >>> coordinate_map = tk.formats.WarpedCoordinateMap(
    ...     lambda unit, domain: unit.square(),
    ...     lambda coordinates, domain: coordinates.sqrt())
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
        """
        Maps unit coordinates to coordinates in the domain.

        Calls ``forward_function(coordinates, domain)`` and requires finite
        floating outputs of unchanged shape.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``,
            with each coordinate in ``[0, 1]``.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Domain metadata understood by the implementation. Warped maps pass
            it unchanged to the supplied callable; ``None`` may leave the
            domain geometry entirely within that callable.

        Returns
        -------
        torch.Tensor
            Coordinates in the domain with the same shape as the input.
        """
        unit = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
        return self._validate_result(
            self.forward_function(unit, domain), unit, 'forward_function')

    def inverse(self,
                domain_coordinates: torch.Tensor,
                domain: Domain = None,
                out_of_domain: str = 'error') -> torch.Tensor:
        """
        Maps coordinates in the domain back to unit coordinates.

        Requires ``inverse_function``; otherwise raises
        ``NotImplementedError``. No numerical inverse is inferred.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates in the domain with shape
            ``(*batch, n_coordinates)``.
        domain : torch.Tensor or sequence of torch.Tensor, optional
            Domain metadata understood by the implementation. Warped maps pass
            it unchanged to the supplied callable; ``None`` may leave the
            domain geometry entirely within that callable.
        out_of_domain : {"error", "clip"}
            Whether coordinates outside the domain raise ``ValueError`` or are
            clipped to the domain boundary.

        Returns
        -------
        torch.Tensor
            Unit coordinates with the same shape and floating dtype as the
            input.
        """
        if self.inverse_function is None:
            raise NotImplementedError(
                '`inverse_function` is required for inverse mapping')
        coordinates = _coordinate_tensor(
            domain_coordinates, 'domain_coordinates')
        unit = self._validate_result(
            self.inverse_function(coordinates, domain),
            coordinates,
            'inverse_function')
        policy = _out_of_domain(out_of_domain)
        outside = (unit < 0) | (unit > 1)
        if policy == 'error' and torch.any(outside):
            raise ValueError(
                '`inverse_function` returned coordinates outside the unit '
                'domain')
        return unit.clamp(0, 1) if policy == 'clip' else unit


class ExplicitGridMap:
    """
    Maps unit coordinates through arbitrary monotonic coordinate grids.

    The forward map interpolates linearly between stored coordinates. Inverse
    evaluation selects the nearest stored coordinate, with ties resolved by its
    lower index. Increasing and decreasing grids are both supported.

    Parameters
    ----------
    coordinates : torch.Tensor or sequence of torch.Tensor
        A shared strictly monotonic floating vector, or one per coordinate. Each
        grid should be finite and have at least two coordinates. These
        coordinates define the domain interval, so ``domain`` should be
        ``None``.
    """

    def __init__(self,
                 coordinates: Union[torch.Tensor, Sequence[torch.Tensor]]) -> None:
        if isinstance(coordinates, torch.Tensor):
            grids = (coordinates,)
            self.shared = True
        else:
            if isinstance(coordinates, (str, bytes)):
                raise TypeError('`coordinates` should contain grid tensors')
            try:
                grids = tuple(coordinates)
            except TypeError as exc:
                raise TypeError(
                    '`coordinates` should contain grid tensors') from exc
            self.shared = False
        if not grids or not all(
                isinstance(grid, torch.Tensor) and grid.ndim == 1 and
                grid.shape[0] >= 2 and grid.is_floating_point() and
                torch.isfinite(grid).all()
                for grid in grids):
            raise ValueError(
                '`coordinates` should contain finite floating vectors of size >= 2')
        for grid in grids:
            differences = grid[1:] - grid[:-1]
            if not (torch.all(differences > 0) or
                    torch.all(differences < 0)):
                raise ValueError(
                    'Every grid in `coordinates` should be monotonic')
        self.coordinates = grids

    def _grids(self, n_coordinates: int) -> Tuple[torch.Tensor, ...]:
        """Broadcasts one shared grid or validates per-coordinate grids."""
        if self.shared:
            return self.coordinates * n_coordinates
        if len(self.coordinates) != n_coordinates:
            raise ValueError('`coordinates` should contain one grid per coordinate')
        return self.coordinates

    @property
    def grid_size(self) -> Tuple[int, ...]:
        """Stored coordinate count per explicit grid before shared broadcasting."""
        return tuple(grid.shape[0] for grid in self.coordinates)

    def forward(self,
                unit_coordinates: torch.Tensor,
                domain: Domain = None) -> torch.Tensor:
        """
        Interpolates explicit grid coordinates at unit coordinates.

        Explicit grids are linearly interpolated; unit values should lie in
        :math:`[0, 1]`.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``,
            with each coordinate in ``[0, 1]``.
        domain : None, optional
            Should be ``None``: the grid coordinates already define the
            domain.

        Returns
        -------
        torch.Tensor
            Coordinates in the domain with the same shape as the input.
        """
        if domain is not None:
            raise ValueError('`domain` is not used by ExplicitGridMap')
        unit = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
        if torch.any(unit < 0) or torch.any(unit > 1):
            raise ValueError('`unit_coordinates` should lie in [0, 1]')
        values = []
        for coordinate, grid in enumerate(self._grids(unit.shape[-1])):
            grid = grid.to(device=unit.device, dtype=unit.dtype)
            scaled = unit[..., coordinate] * (grid.shape[0] - 1)
            lower = scaled.floor().to(torch.long)
            upper = (lower + 1).clamp_max(grid.shape[0] - 1)
            fraction = scaled - lower
            values.append(
                grid[lower] * (1 - fraction) + grid[upper] * fraction)
        return torch.stack(values, dim=-1)

    def inverse(self,
                domain_coordinates: torch.Tensor,
                domain: Domain = None,
                out_of_domain: str = 'error') -> torch.Tensor:
        """
        Finds nearest explicit grid coordinates and returns their unit positions.

        Ties choose the lower stored index, also on descending grids.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates in the domain with shape
            ``(*batch, n_coordinates)``.
        domain : None, optional
            Should be ``None``: the grid coordinates already define the
            domain.
        out_of_domain : {"error", "clip"}
            Whether coordinates outside the domain raise ``ValueError`` or are
            clipped to the domain boundary.

        Returns
        -------
        torch.Tensor
            Unit coordinates with the same shape and floating dtype as the
            input.
        """
        if domain is not None:
            raise ValueError('`domain` is not used by ExplicitGridMap')
        coordinates = _coordinate_tensor(
            domain_coordinates, 'domain_coordinates')
        policy = _out_of_domain(out_of_domain)
        values = []
        for coordinate, grid in enumerate(self._grids(coordinates.shape[-1])):
            grid = grid.to(device=coordinates.device, dtype=coordinates.dtype)
            value = coordinates[..., coordinate]
            lower_bound = grid.min()
            upper_bound = grid.max()
            outside = (value < lower_bound) | (value > upper_bound)
            if policy == 'error' and torch.any(outside):
                raise ValueError(
                    '`domain_coordinates` lie outside an explicit grid')
            value = value.clamp(lower_bound, upper_bound)
            distances = (value.unsqueeze(-1) - grid).abs()
            index = distances.argmin(dim=-1)
            values.append(index.to(coordinates.dtype) / (grid.shape[0] - 1))
        return torch.stack(values, dim=-1)

    def from_indices(self,
                     indices: torch.Tensor,
                     grid_size: Optional[Sequence[int]] = None,
                     domain: Domain = None) -> torch.Tensor:
        """
        Maps integer grid indices to coordinate values in the domain.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape ``(*batch, n_coordinates)``; each
            value lies in ``[0, grid_size[coordinate] - 1]``.
        grid_size : sequence of int, optional
            Expected number of grid coordinates per coordinate. ``None`` uses stored
            coordinate counts; a supplied value should match them.
        domain : None, optional
            Should be ``None``: the grid coordinates already define the
            domain.

        Returns
        -------
        torch.Tensor
            Values in the domain with shape ``(*batch, n_coordinates)``,
            retaining the stored grid dtype.
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
        for coordinate, grid in enumerate(grids):
            index = indices[..., coordinate].to(torch.long)
            if torch.any(index < 0) or torch.any(index >= grid.shape[0]):
                raise ValueError(
                    '`indices` is out of bounds for explicit grid')
            values.append(grid.to(indices.device)[index])
        return torch.stack(values, dim=-1)

    def to_indices(self,
                   domain_coordinates: torch.Tensor,
                   grid_size: Optional[Sequence[int]] = None,
                   domain: Domain = None,
                   out_of_domain: str = 'error') -> torch.Tensor:
        """
        Quantizes coordinates in the domain to nearest grid indices.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates in the domain with shape
            ``(*batch, n_coordinates)``.
        grid_size : sequence of int, optional
            Expected number of grid coordinates per coordinate. ``None`` uses stored
            coordinate counts; a supplied value should match them.
        domain : None, optional
            Should be ``None``: the grid coordinates already define the
            domain.
        out_of_domain : {"error", "clip"}
            Whether coordinates outside the domain raise ``ValueError`` or are
            clipped to the domain boundary.

        Returns
        -------
        torch.Tensor
            ``torch.long`` indices with shape ``(*batch, n_coordinates)``. Exact
            ties select the lower index.

        Examples
        --------
        >>> coordinate_map = tk.formats.ExplicitGridMap(torch.tensor([0., 1., 4.]))
        >>> coordinate_map.to_indices(torch.tensor([[2.5]])).tolist()
        [[1]]
        """
        if domain is not None:
            raise ValueError('`domain` is not used by ExplicitGridMap')
        unit = self.inverse(
            domain_coordinates, out_of_domain=out_of_domain)
        sizes_tuple = tuple(
            grid.shape[0] for grid in self._grids(unit.shape[-1]))
        if grid_size is not None and tuple(grid_size) != sizes_tuple:
            raise ValueError('`grid_size` should match the explicit grids')
        sizes = unit.new_tensor(sizes_tuple)
        return torch.round(unit * (sizes - 1)).to(torch.long)


class _CompositeCoordinateMap:
    """Applies one independent coordinate map per coordinate."""

    def __init__(self, maps: Sequence[CoordinateMap]) -> None:
        self.maps = tuple(maps)
        if not self.maps or not all(
                isinstance(coordinate_map, CoordinateMap)
                for coordinate_map in self.maps):
            raise TypeError(
                'Every map in `maps` should implement CoordinateMap')

    @staticmethod
    def _domains(domain: Domain, n_coordinates: int) -> Tuple[Domain, ...]:
        """Normalizes one domain specification per coordinate."""
        if domain is None:
            return (None,) * n_coordinates
        if isinstance(domain, torch.Tensor):
            if domain.shape == (2,):
                return (domain,) * n_coordinates
            if domain.shape == (n_coordinates, 2):
                return tuple(domain[coordinate] for coordinate in range(n_coordinates))
            raise ValueError(
                '`domain` should contain one interval per coordinate')
        values = tuple(domain)
        if len(values) != n_coordinates:
            raise ValueError('`domain` should contain one entry per coordinate')
        return values

    def forward(self,
                unit_coordinates: torch.Tensor,
                domain: Domain = None) -> torch.Tensor:
        """Maps each coordinate through its constituent forward coordinate map."""
        unit = _coordinate_tensor(unit_coordinates, 'unit_coordinates')
        if unit.shape[-1] != len(self.maps):
            raise ValueError(
                'Coordinate coordinates should match coordinate maps')
        domains = self._domains(domain, len(self.maps))
        values = [
            coordinate_map.forward(
                unit[..., coordinate:coordinate + 1], domains[coordinate])
            for coordinate, coordinate_map in enumerate(self.maps)
        ]
        return torch.cat(values, dim=-1)


    def inverse(self,
                domain_coordinates: torch.Tensor,
                domain: Domain = None,
                out_of_domain: str = 'error') -> torch.Tensor:
        """Maps coordinates in the domain through each inverse map."""
        coordinates = _coordinate_tensor(
            domain_coordinates, 'domain_coordinates')
        if coordinates.shape[-1] != len(self.maps):
            raise ValueError(
                'Coordinate coordinates should match coordinate maps')
        domains = self._domains(domain, len(self.maps))
        values = []
        for coordinate, coordinate_map in enumerate(self.maps):
            inverse = getattr(coordinate_map, 'inverse', None)
            if not callable(inverse):
                raise NotImplementedError(
                    f'Coordinate map {coordinate} does not define an inverse')
            values.append(inverse(
                coordinates[..., coordinate:coordinate + 1],
                domains[coordinate],
                out_of_domain=out_of_domain))
        return torch.cat(values, dim=-1)

    def from_indices(self,
                     indices: torch.Tensor,
                     grid_size: Sequence[int],
                     domain: Domain = None) -> torch.Tensor:
        """Maps each coordinate grid index through its constituent coordinate map."""
        domains = self._domains(domain, len(self.maps))
        values = []
        for coordinate, coordinate_map in enumerate(self.maps):
            kernel = getattr(coordinate_map, 'from_indices', None)
            if callable(kernel):
                value = kernel(
                    indices[..., coordinate:coordinate + 1],
                    (grid_size[coordinate],),
                    domains[coordinate])
            else:
                unit = _indices_to_unit(
                    indices[..., coordinate:coordinate + 1],
                    (grid_size[coordinate],),
                    'endpoints')
                value = coordinate_map.forward(unit, domains[coordinate])
            values.append(value)
        return torch.cat(values, dim=-1)

    def to_indices(self,
                   domain_coordinates: torch.Tensor,
                   grid_size: Sequence[int],
                   domain: Domain = None,
                   out_of_domain: str = 'error') -> torch.Tensor:
        """Quantizes coordinates in the domain using each coordinate map."""
        domains = self._domains(domain, len(self.maps))
        values = []
        for coordinate, coordinate_map in enumerate(self.maps):
            kernel = getattr(coordinate_map, 'to_indices', None)
            if callable(kernel):
                value = kernel(
                    domain_coordinates[..., coordinate:coordinate + 1],
                    (grid_size[coordinate],),
                    domains[coordinate],
                    out_of_domain=out_of_domain)
            else:
                inverse = getattr(coordinate_map, 'inverse', None)
                if not callable(inverse):
                    raise NotImplementedError(
                        f'Coordinate map {coordinate} does not define an inverse')
                unit = inverse(
                    domain_coordinates[..., coordinate:coordinate + 1],
                    domains[coordinate],
                    out_of_domain=out_of_domain)
                value = _unit_to_indices(
                    unit,
                    (grid_size[coordinate],),
                    'endpoints',
                    out_of_domain)
            values.append(value)
        return torch.cat(values, dim=-1)


def _validate_explicit_grid(layout: QuantizedLayout,
                            coordinate_map: Optional[CoordinateMap]) -> None:
    """Checks that explicit coordinate grids cover every digit configuration."""
    if isinstance(coordinate_map, ExplicitGridMap):
        sizes = tuple(grid.shape[0] for grid in coordinate_map._grids(
            layout.n_coordinates))
        if sizes != layout.grid_size:
            raise ValueError('Explicit grid size should match `base ** level`')
    elif isinstance(coordinate_map, _CompositeCoordinateMap):
        for coordinate, item in enumerate(coordinate_map.maps):
            if isinstance(item, ExplicitGridMap):
                sizes = tuple(grid.shape[0] for grid in item._grids(1))
                if sizes != (layout.grid_size[coordinate],):
                    raise ValueError(
                        'Explicit grid size should match `base ** level`')
