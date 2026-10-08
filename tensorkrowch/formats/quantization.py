"""
This script contains:

    Public classes:
        * QuantizedLayout
        * CoordinateMap
            + AffineCoordinateMap
            + FunctionalCoordinateMap
            + ExplicitGridMap
            + _CompositeCoordinateMap

    Internal functions:
        * _integer_spec
        * _coordinate_tensor
        * _domain_tensor
        * _out_of_domain
        * _grid_offset
        * _indices_to_unit
        * _unit_to_indices

Terminology:
    * Coordinate: a continuous input value along one dimension.
    * Grid: the discrete values available along that dimension.
    * Index: an integer position on its grid.
    * Digit: one component of an index in the chosen base, used as a
      site input.
    * Layout: the order in which coordinates' digits occupy sites.

For example, index ``5`` on a base-``2`` grid with three digits becomes
``(1, 0, 1)``. A coordinate is first mapped to a grid index, then expanded
into digits; evaluating coordinates selects entries of the format without
interpolating between them.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

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


@dataclass(frozen=True)
class QuantizedLayout:  # MARK: QuantizedLayout
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

    ``ordering="interleaved"`` cycles over coordinates at every available digit
    depth and omits coordinates whose levels are exhausted.
    ``ordering="grouped"`` places all digits of each coordinate together. With
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
    ordering : {"interleaved", "grouped", "custom"}
        Final digit-site schedule. Defaults to ``"interleaved"``.
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
    ordering: str = 'interleaved'
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

    def _validate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Validates coordinate index dimensions and bounds, returning integers."""
        indices = self._integer_tensor(indices, 'indices')
        if indices.shape[-1] != self.n_coordinates:
            raise ValueError(
                'The last `indices` dimension should match `n_coordinates`')
        for coordinate, size in enumerate(self.grid_size):
            values = indices[..., coordinate]
            if torch.any(values < 0) or torch.any(values >= size):
                raise ValueError(
                    f'`indices` is out of bounds for coordinate {coordinate}')
        return indices

    def _validate_digits(self, digits: torch.Tensor) -> torch.Tensor:
        """Validates scheduled digit dimensions and bounds, returning integers."""
        digits = self._integer_tensor(digits, 'digits')
        if digits.shape[-1] != self.n_sites:
            raise ValueError(
                'The last `digits` dimension should match the layout sites '
                '(`n_sites`)')
        for site, base in enumerate(self.in_dim):
            values = digits[..., site]
            if torch.any(values < 0) or torch.any(values >= base):
                raise ValueError(f'`digits` is out of bounds at site {site}')
        return digits

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
        indices = self._validate_indices(indices)

        # Expand each index, then arrange its digits in the chosen site order.
        digit_values = {}
        for coordinate, (base, level) in enumerate(zip(self.base, self.level)):
            values = indices[..., coordinate]
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
        digits = self._validate_digits(digits)
        digit_values = {coordinate_digit: digits[..., site]
                        for site, coordinate_digit in enumerate(self.sites())}

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
        >>> layout = tk.formats.QuantizedLayout(
        ...     2, base=2, level=2, ordering='grouped')
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
                   device: Optional[torch.device],
                   dtype: Optional[torch.dtype]) -> torch.Tensor:
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
            domains = tuple(domain)
        except TypeError as exc:
            raise TypeError('`domain` should contain interval tensors') \
                from exc
        intervals = torch.stack([
            interval if isinstance(interval, torch.Tensor)
            else torch.as_tensor(interval)
            for interval in domains
        ])
        if intervals.shape == (2,):
            intervals = intervals.expand(n_coordinates, 2)
        elif intervals.shape != (n_coordinates, 2):
            raise ValueError(
                'Every `domain` interval should contain two values')
    intervals = intervals.to(device=device, dtype=dtype)
    if intervals.is_complex():
        raise TypeError('`domain` should contain real intervals')
    if not intervals.is_floating_point():
        intervals = intervals.to(torch.get_default_dtype())
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


def _grid_offset(grid_offset: Union[str, float]) -> Optional[float]:
    """Validates the grid convention and returns its within-cell offset."""
    if isinstance(grid_offset, str):
        if grid_offset == 'endpoints':
            return None
        offsets = {'left': 0., 'centers': 0.5, 'right': 1.}
        if grid_offset in offsets:
            return offsets[grid_offset]
        raise ValueError(
            "`grid_offset` should be 'endpoints', 'left', 'centers', 'right' or "
            'a number between 0 and 1')

    if isinstance(grid_offset, bool) or not isinstance(
            grid_offset, (int, float)):
        raise TypeError(
            '`grid_offset` should be str or a number between 0 and 1')
    if not 0 <= grid_offset <= 1:
        raise ValueError('A numeric `grid_offset` should be between 0 and 1')
    return float(grid_offset)


def _indices_to_unit(indices: torch.Tensor,
                     grid_size: Sequence[int],
                     grid_offset: Union[str, float],
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

    offset = _grid_offset(grid_offset)
    if offset is None:
        return values / (sizes - 1)
    return (values + offset) / sizes


def _unit_to_indices(unit_coordinates: torch.Tensor,
                     grid_size: Sequence[int],
                     grid_offset: Union[str, float],
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
    offset = _grid_offset(grid_offset)
    scaled = unit * (sizes - 1) if offset is None else unit * sizes
    if offset not in (0, 1):
        scaled = scaled - (0 if offset is None else offset) - 0.5

    # Snap to an integer boundary when it lies within the rounding interval
    tolerance = 4 * torch.finfo(unit.dtype).eps * scaled.abs().clamp_min(1)
    boundary = scaled.round()
    contains_boundary = (scaled - tolerance <= boundary) & \
        (boundary <= scaled + tolerance)
    scaled = torch.where(contains_boundary, boundary, scaled)

    if offset == 0:
        indices = torch.floor(scaled)
    elif offset == 1:
        indices = torch.ceil(scaled) - 1
    else:
        # Half-way values within numerical tolerance select the smaller index
        indices = torch.ceil(scaled)

    return indices.clamp_min(0).minimum(sizes - 1).to(torch.long)


class CoordinateMap(ABC):  # MARK: CoordinateMap
    """
    Converts between unit coordinates, domain coordinates and grid indices.

    Maps own their domain, grid sizes and out-of-domain policy. Unit coordinates
    lie in ``[0, 1]``; domain coordinates lie in the domain described by the map.
    :class:`AffineCoordinateMap` and :class:`FunctionalCoordinateMap` both use
    uniform grids in unit space. Their transformations determine where these
    grid points lie in the domain. :class:`ExplicitGridMap` stores the domain
    grid points directly.

    With ``N`` grid points, ``grid_offset="endpoints"`` uses ``i / (N - 1)``.
    Other uniform grids use ``(i + offset) / N``: ``"left"``, ``"centers"``
    and ``"right"`` correspond to offsets ``0``, ``0.5`` and ``1``. A numeric
    ``grid_offset`` selects any offset in ``[0, 1]``.

    ``"left"`` assigns unit coordinates using ``floor(N * x)``, with ``x=1``
    assigned to the last index, as in :func:`~tensorkrowch.embeddings.discretize`.
    ``"right"`` uses ``ceil(N * x) - 1``, with ``x=0`` assigned to the first
    index. Numeric offsets ``0`` and ``1`` follow the same rules. Other grids
    select the nearest unit grid point; ties select the lower index.
    """

    domain: Domain
    grid_size: Tuple[int, ...]
    grid_offset: Union[str, float]
    out_of_domain: str

    def _initialize_grid(self) -> None:
        """Normalizes uniform grid sizes and validates the grid policy."""
        if isinstance(self.grid_size, (str, bytes)):
            raise TypeError('`grid_size` should be a sequence of integers')
        sizes = tuple(self.grid_size)
        if not sizes:
            raise ValueError('`grid_size` should contain at least one size')
        sizes = _integer_spec(sizes, len(sizes), 'grid_size', 2)
        _grid_offset(self.grid_offset)
        _out_of_domain(self.out_of_domain)
        object.__setattr__(self, 'grid_size', sizes)

    def _coordinates(self, values: torch.Tensor, name: str) -> torch.Tensor:
        """Checks coordinate values and their final coordinate dimension."""
        values = _coordinate_tensor(values, name)
        if values.shape[-1] != len(self.grid_size):
            raise ValueError(
                f'`{name}` should contain one value per coordinate')
        return values

    def _unit_coordinates(self, values: torch.Tensor) -> torch.Tensor:
        """Checks or clips unit coordinates according to the stored policy."""
        unit = self._coordinates(values, 'unit_coordinates')
        outside = (unit < 0) | (unit > 1)
        if (self.out_of_domain == 'error') and torch.any(outside):
            raise ValueError('`unit_coordinates` should lie in [0, 1]')
        return unit.clamp(0, 1) if self.out_of_domain == 'clip' else unit

    @abstractmethod
    def forward(self, unit_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Maps unit coordinates to coordinates in the domain.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``.
            Each coordinate lies in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Coordinates in the domain with the same shape as the input.
        """

    @abstractmethod
    def inverse(self, domain_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Maps coordinates in the domain back to unit coordinates.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``.

        Returns
        -------
        torch.Tensor
            Unit coordinates with the same shape as the input.
        """

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Maps grid indices to coordinates in the domain.

        Indices select points of the uniform grid in unit space. The forward
        transformation maps these points into the domain.

        Parameters
        ----------
        indices : torch.Tensor
            Integer indices with shape ``(*batch, n_coordinates)``. Each index
            lies in ``[0, grid_size[coordinate] - 1]``.

        Returns
        -------
        torch.Tensor
            Coordinates in the domain with the same shape as ``indices``.
            Uses the domain dtype when available, otherwise the default
            floating dtype.

        Examples
        --------
        >>> coordinate_map = tk.formats.AffineCoordinateMap(
        ...     domain=torch.tensor([-2., 2.]),
        ...     grid_size=(4,),
        ...     grid_offset='centers')
        >>> coordinate_map.from_indices(torch.tensor([[0], [3]])).tolist()
        [[-1.5], [1.5]]
        """
        dtype = (self.domain.dtype if isinstance(self.domain, torch.Tensor) and
                 self.domain.is_floating_point() else None)
        unit = _indices_to_unit(
            indices, self.grid_size, self.grid_offset, dtype=dtype)
        return self.forward(unit)

    def to_indices(self, domain_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Quantizes coordinates in the domain to grid indices.

        Applies the inverse transformation, then selects indices according to
        ``grid_offset`` in unit space. No interpolation of format values is
        performed. Values within floating-point rounding tolerance of a
        quantization boundary are treated as lying exactly on that boundary.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``.

        Returns
        -------
        torch.Tensor
            ``torch.long`` indices with the same shape as the input.

        Examples
        --------
        >>> coordinate_map = tk.formats.AffineCoordinateMap(
        ...     domain=torch.tensor([0., 1.]), grid_size=(4,))
        >>> coordinate_map.to_indices(torch.tensor([[0.2], [0.75], [1.]])).tolist()
        [[0], [3], [3]]
        """
        unit = self.inverse(domain_coordinates)
        return _unit_to_indices(
            unit, self.grid_size, self.grid_offset, self.out_of_domain)


@dataclass(frozen=True)
class AffineCoordinateMap(CoordinateMap):  # MARK: AffineCoordinateMap
    """
    Affine transformation of a uniform unit grid into a coordinate domain.

    Grid points are uniform in unit space and remain uniform in each domain
    interval. ``forward`` and ``inverse`` apply affine transformations between
    unit and domain coordinates. ``from_indices`` and ``to_indices`` also use
    the stored uniform grid.

    Parameters
    ----------
    domain : torch.Tensor or sequence of torch.Tensor
        Domain intervals as ``(2,)`` for a shared interval or
        ``(n_coordinates, 2)`` for separate intervals.
    grid_size : sequence of int
        Number of grid points per coordinate, with each size at least ``2``.
    grid_offset : {"endpoints", "left", "centers", "right"} or float
        Uniform unit grid convention, as described in :class:`CoordinateMap`.
        The default ``"left"`` reproduces interval assignment in
        :func:`~tensorkrowch.embeddings.discretize`.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ``ValueError`` or are
        clipped to its boundaries.

    Examples
    --------
    >>> coordinate_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([0., 1.]), grid_size=(4,))
    >>> coordinate_map.from_indices(torch.arange(4).unsqueeze(-1)).tolist()
    [[0.0], [0.25], [0.5], [0.75]]
    """

    domain: Domain
    grid_size: Sequence[int]
    grid_offset: Union[str, float] = 'left'
    out_of_domain: str = 'error'

    def __post_init__(self) -> None:
        """Validates the grid and required affine domain."""
        if self.domain is None:
            raise ValueError('`domain` is required by AffineCoordinateMap')
        self._initialize_grid()
        intervals = _domain_tensor(self.domain, len(self.grid_size),
                                   None, None)
        object.__setattr__(self, 'domain', intervals)

    def forward(self, unit_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Maps unit coordinates into the domain using an affine transformation.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``.
            Each coordinate lies in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Coordinates in the domain with the same shape and dtype as input.

        Examples
        --------
        >>> coordinate_map = tk.formats.AffineCoordinateMap(
        ...     domain=torch.tensor([-2., 2.]), grid_size=(4,))
        >>> coordinate_map.forward(torch.tensor([[0.], [0.5], [1.]])).tolist()
        [[-2.0], [0.0], [2.0]]
        """
        unit = self._unit_coordinates(unit_coordinates)
        intervals = self.domain.to(device=unit.device, dtype=unit.dtype)
        return intervals[:, 0] + unit * (intervals[:, 1] - intervals[:, 0])

    def inverse(self, domain_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Maps coordinates in the domain back to unit coordinates.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``.

        Returns
        -------
        torch.Tensor
            Unit coordinates with the same shape and dtype as the input.

        Examples
        --------
        >>> coordinate_map = tk.formats.AffineCoordinateMap(
        ...     domain=torch.tensor([-2., 2.]), grid_size=(4,))
        >>> coordinate_map.inverse(torch.tensor([[-2.], [0.], [2.]])).tolist()
        [[0.0], [0.5], [1.0]]
        >>> coordinate_map = tk.formats.AffineCoordinateMap(
        ...     domain=torch.tensor([-2., 2.]),
        ...     grid_size=(4,),
        ...     out_of_domain='clip')
        >>> coordinate_map.inverse(torch.tensor([[-3.], [3.]])).tolist()
        [[0.0], [1.0]]
        """
        coordinates = self._coordinates(
            domain_coordinates, 'domain_coordinates')
        intervals = self.domain.to(
            device=coordinates.device, dtype=coordinates.dtype)
        unit = (coordinates - intervals[:, 0]) / (
            intervals[:, 1] - intervals[:, 0])
        return self._unit_coordinates(unit)


@dataclass(frozen=True, init=False)
class FunctionalCoordinateMap(CoordinateMap):  # MARK: FunctionalCoordinateMap
    """
    Function-defined transformation of a uniform unit grid into the domain.

    Like :class:`AffineCoordinateMap`, this map uses a uniform grid in unit
    space. The supplied transformation can make its grid nonuniform in the
    domain, or couple coordinates. ``to_indices`` quantizes in unit space after
    applying the supplied inverse; it does not search for the nearest domain
    grid point.

    Parameters
    ----------
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Domain metadata stored in the map and passed unchanged to the functions.
        ``None`` is allowed when the functions already define the domain.
    grid_size : sequence of int
        Number of uniform unit grid points per coordinate.
    grid_offset : {"endpoints", "left", "centers", "right"} or float
        Uniform unit grid convention, as described in :class:`CoordinateMap`.
    out_of_domain : {"error", "clip"}
        Whether unit coordinates outside ``[0, 1]`` raise ``ValueError`` or are
        clipped to its boundaries.
    forward_function : callable
        Function ``(unit_coordinates, domain)`` returning finite floating
        coordinates in the domain with unchanged shape.
    inverse_function : callable, optional
        Function ``(domain_coordinates, domain)`` returning unit coordinates
        with unchanged shape. Without it, inverse operations raise
        ``NotImplementedError``.

    Examples
    --------
    >>> coordinate_map = tk.formats.FunctionalCoordinateMap(
    ...     domain=None, grid_size=(4,),
    ...     forward_function=lambda u, domain: u.square(),
    ...     inverse_function=lambda x, domain: x.sqrt())
    >>> coordinate_map.from_indices(torch.arange(4).unsqueeze(-1)).tolist()
    [[0.0], [0.0625], [0.25], [0.5625]]
    >>> coordinate_map.to_indices(torch.tensor([[0.25]])).tolist()
    [[2]]
    """

    domain: Domain
    grid_size: Sequence[int]
    grid_offset: Union[str, float]
    out_of_domain: str
    forward_function: Callable[[torch.Tensor, Domain], torch.Tensor]
    inverse_function: Optional[Callable[[torch.Tensor, Domain], torch.Tensor]]

    def __init__(self,
                 domain: Domain = None,
                 grid_size: Optional[Sequence[int]] = None,
                 grid_offset: Union[str, float] = 'left',
                 out_of_domain: str = 'error',
                 *,
                 forward_function: Callable[[torch.Tensor, Domain],
                                            torch.Tensor],
                 inverse_function: Optional[Callable[[torch.Tensor, Domain],
                                                     torch.Tensor]] = None
                 ) -> None:
        if grid_size is None:
            raise TypeError(
                '`grid_size` is required by FunctionalCoordinateMap')

        object.__setattr__(self, 'domain', domain)
        object.__setattr__(self, 'grid_size', grid_size)
        object.__setattr__(self, 'grid_offset', grid_offset)
        object.__setattr__(self, 'out_of_domain', out_of_domain)
        object.__setattr__(self, 'forward_function', forward_function)
        object.__setattr__(self, 'inverse_function', inverse_function)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validates the grid and supplied coordinate functions."""
        self._initialize_grid()
        if not callable(self.forward_function):
            raise TypeError('`forward_function` should be callable')
        if self.inverse_function is not None and not callable(
                self.inverse_function):
            raise TypeError('`inverse_function` should be callable or None')

    def _result(self,
                result: torch.Tensor,
                reference: torch.Tensor,
                name: str) -> torch.Tensor:
        """Checks the shape and values returned by a coordinate function."""
        result = self._coordinates(result, name)
        if result.shape != reference.shape:
            raise ValueError(f'`{name}` should preserve coordinate shape')
        return result

    def forward(self, unit_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Transforms unit coordinates using ``forward_function``.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``.
            Each coordinate lies in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Coordinates in the domain with the same shape as the input.

        Examples
        --------
        >>> coordinate_map = tk.formats.FunctionalCoordinateMap(
        ...     domain=None, grid_size=(4,),
        ...     forward_function=lambda u, domain: u.square(),
        ...     inverse_function=lambda x, domain: x.sqrt())
        >>> coordinate_map.forward(torch.tensor([[0.5]])).tolist()
        [[0.25]]
        """
        unit = self._unit_coordinates(unit_coordinates)
        return self._result(self.forward_function(unit, self.domain),
                            unit, 'forward_function')

    def inverse(self, domain_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Transforms domain coordinates using ``inverse_function``.

        No numerical inverse is inferred. If ``inverse_function`` is absent,
        this method raises ``NotImplementedError``.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``.

        Returns
        -------
        torch.Tensor
            Unit coordinates with the same shape as the input.

        Examples
        --------
        >>> coordinate_map = tk.formats.FunctionalCoordinateMap(
        ...     domain=None, grid_size=(4,),
        ...     forward_function=lambda u, domain: u.square(),
        ...     inverse_function=lambda x, domain: x.sqrt())
        >>> coordinate_map.inverse(torch.tensor([[0.25]])).tolist()
        [[0.5]]
        """
        if self.inverse_function is None:
            raise NotImplementedError(
                '`inverse_function` is required for inverse mapping')
        coordinates = self._coordinates(
            domain_coordinates, 'domain_coordinates')
        unit = self._result(self.inverse_function(coordinates, self.domain),
                            coordinates, 'inverse_function')
        return self._unit_coordinates(unit)


@dataclass(frozen=True, init=False)
class ExplicitGridMap(CoordinateMap):  # MARK: ExplicitGridMap
    """
    Coordinate map with grid points supplied directly in the domain.

    A vector describes one coordinate. A matrix has one row per coordinate;
    a sequence of vectors also supports different grid sizes. Grids can be
    increasing or decreasing. ``from_indices`` selects stored grid points and
    ``to_indices`` selects the nearest point, with ties choosing the lower
    stored index. ``forward`` interpolates between the stored points;
    ``inverse`` reverses this interpolation to recover unit coordinates.

    Parameters
    ----------
    grid_coordinates : torch.Tensor or sequence of torch.Tensor
        Finite, strictly monotonic floating grid vectors of size at least ``2``.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the grid boundaries raise ``ValueError``
        or are clipped to those boundaries.

    Examples
    --------
    >>> coordinate_map = tk.formats.ExplicitGridMap(
    ...     torch.tensor([0., 0.125, 0.5, 1.]))
    >>> coordinate_map.from_indices(torch.tensor([[2]])).tolist()
    [[0.5]]
    """

    grid_coordinates: Tuple[torch.Tensor, ...]
    out_of_domain: str = 'error'

    def __init__(self,
                 grid_coordinates: Union[torch.Tensor, Sequence[torch.Tensor]],
                 out_of_domain: str = 'error') -> None:
        if isinstance(grid_coordinates, torch.Tensor):
            if grid_coordinates.ndim not in (1, 2):
                raise ValueError(
                    '`grid_coordinates` should be a vector or matrix')
            grids = ((grid_coordinates,) if grid_coordinates.ndim == 1
                     else tuple(grid_coordinates.unbind(0)))
        else:
            if isinstance(grid_coordinates, (str, bytes)):
                raise TypeError(
                    '`grid_coordinates` should contain grid tensors')
            try:
                grids = tuple(grid_coordinates)
            except TypeError as exc:
                raise TypeError(
                    '`grid_coordinates` should contain grid tensors') from exc
        if not grids or not all(
                isinstance(grid, torch.Tensor) and (grid.ndim == 1) and
                (grid.shape[0] >= 2) and grid.is_floating_point() and
                torch.isfinite(grid).all() for grid in grids):
            raise ValueError(
                '`grid_coordinates` should contain finite floating grid vectors')
        for grid in grids:
            differences = grid[1:] - grid[:-1]
            if not (torch.all(differences > 0) or torch.all(differences < 0)):
                raise ValueError(
                    'Every grid in `grid_coordinates` should be strictly '
                    'monotonic')
        object.__setattr__(self, 'grid_coordinates', grids)
        object.__setattr__(self, 'out_of_domain',
                           _out_of_domain(out_of_domain))

    @property
    def domain(self) -> Tuple[torch.Tensor, ...]:
        """Domain boundaries taken from the endpoints of each stored grid."""
        return tuple(grid[[0, -1]] for grid in self.grid_coordinates)

    @property
    def grid_size(self) -> Tuple[int, ...]:
        """Number of stored grid points per coordinate."""
        return tuple(grid.shape[0] for grid in self.grid_coordinates)

    def forward(self, unit_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Interpolates grid points at unit coordinates.

        Parameters
        ----------
        unit_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``.
            Each coordinate lies in ``[0, 1]``.

        Returns
        -------
        torch.Tensor
            Coordinates in the domain with the same shape and dtype as input.

        Examples
        --------
        >>> coordinate_map = tk.formats.ExplicitGridMap(
        ...     torch.tensor([0., 1., 4.]))
        >>> coordinate_map.forward(torch.tensor([[0.25]])).tolist()
        [[0.5]]
        """
        unit = self._unit_coordinates(unit_coordinates)
        values = []
        for coordinate, grid in enumerate(self.grid_coordinates):
            grid = grid.to(device=unit.device, dtype=unit.dtype)
            scaled = unit[..., coordinate] * (grid.shape[0] - 1)
            lower = scaled.floor().to(torch.long)
            upper = (lower + 1).clamp_max(grid.shape[0] - 1)
            fraction = scaled - lower
            values.append(grid[lower] * (1 - fraction) + grid[upper] * fraction)
        return torch.stack(values, dim=-1)

    def inverse(self, domain_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Maps domain coordinates to unit coordinates by inverse interpolation.

        Finds the two grid points surrounding each domain coordinate and its
        relative position between them. Uses that position between their unit
        coordinates, ``i / (N - 1)`` and ``(i + 1) / (N - 1)``, to reverse the
        interpolation performed by :meth:`forward`.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``.

        Returns
        -------
        torch.Tensor
            Unit coordinates with the same shape and dtype as the input.

        Examples
        --------
        >>> coordinate_map = tk.formats.ExplicitGridMap(
        ...     torch.tensor([0., 1., 4.]))
        >>> coordinate_map.inverse(torch.tensor([[2.5]])).tolist()
        [[0.75]]
        """
        coordinates = self._coordinates(
            domain_coordinates, 'domain_coordinates')
        values = []
        for coordinate, grid in enumerate(self.grid_coordinates):
            grid = grid.to(device=coordinates.device, dtype=coordinates.dtype)
            value = coordinates[..., coordinate]
            lower, upper = grid.min(), grid.max()
            outside = (value < lower) | (value > upper)
            if (self.out_of_domain == 'error') and torch.any(outside):
                raise ValueError(
                    '`domain_coordinates` lie outside an explicit grid')
            value = value.clamp(lower, upper)

            # Find the enclosing interval in increasing or decreasing grids.
            if grid[0] < grid[-1]:
                right = torch.searchsorted(grid, value.contiguous())
            else:
                right = torch.searchsorted(-grid, (-value).contiguous())
            right = right.clamp(1, grid.shape[0] - 1)
            left = right - 1
            fraction = (value - grid[left]) / (grid[right] - grid[left])
            values.append((left + fraction) / (grid.shape[0] - 1))

        return torch.stack(values, dim=-1)

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Selects stored grid points by index.

        Parameters
        ----------
        indices : torch.Tensor
            Integer indices with shape ``(*batch, n_coordinates)``. Each index
            lies in ``[0, grid_size[coordinate] - 1]``.

        Returns
        -------
        torch.Tensor
            Coordinates in the domain with the same shape as ``indices``,
            retaining the stored grid dtype.

        Examples
        --------
        >>> coordinate_map = tk.formats.ExplicitGridMap(
        ...     torch.tensor([0., 1., 4.]))
        >>> coordinate_map.from_indices(torch.tensor([[0], [2]])).tolist()
        [[0.0], [4.0]]
        """
        if not isinstance(indices, torch.Tensor) or (indices.ndim < 1) or \
                indices.dtype not in _INTEGER_DTYPES:
            raise TypeError('`indices` should be an integer tensor')
        if indices.shape[-1] != len(self.grid_coordinates):
            raise ValueError(
                '`indices` should contain one index per coordinate')
        values = []
        for coordinate, grid in enumerate(self.grid_coordinates):
            index = indices[..., coordinate].long()
            if torch.any(index < 0) or torch.any(index >= grid.shape[0]):
                raise ValueError(
                    '`indices` is out of bounds for explicit grid')
            values.append(grid.to(indices.device)[index])
        return torch.stack(values, dim=-1)

    def to_indices(self, domain_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Selects the nearest grid point for each coordinate in the domain.

        Parameters
        ----------
        domain_coordinates : torch.Tensor
            Finite floating coordinates with shape ``(*batch, n_coordinates)``.

        Returns
        -------
        torch.Tensor
            ``torch.long`` indices with the same shape as the input. Ties select
            the lower stored index, including on decreasing grids.

        Examples
        --------
        >>> coordinate_map = tk.formats.ExplicitGridMap(
        ...     torch.tensor([0., 1., 4.]))
        >>> coordinate_map.to_indices(torch.tensor([[2.5]])).tolist()
        [[1]]
        """
        coordinates = self._coordinates(
            domain_coordinates, 'domain_coordinates')
        values = []
        for coordinate, grid in enumerate(self.grid_coordinates):
            grid = grid.to(device=coordinates.device, dtype=coordinates.dtype)
            value = coordinates[..., coordinate]
            lower, upper = grid.min(), grid.max()
            outside = (value < lower) | (value > upper)
            if (self.out_of_domain == 'error') and torch.any(outside):
                raise ValueError(
                    '`domain_coordinates` lie outside an explicit grid')
            value = value.clamp(lower, upper)
            values.append((value.unsqueeze(-1) - grid).abs().argmin(dim=-1))
        return torch.stack(values, dim=-1)


@dataclass(frozen=True)
class _CompositeCoordinateMap(CoordinateMap):  # MARK: _CompositeCoordinateMap
    """Applies one independent coordinate map per coordinate."""

    maps: Sequence[CoordinateMap]

    def __post_init__(self) -> None:
        """Checks and stores maps for individual coordinates."""
        maps = tuple(self.maps)
        if not maps or not all(isinstance(item, CoordinateMap) and
                               (len(item.grid_size) == 1) for item in maps):
            raise ValueError(
                '`maps` should contain one single-coordinate map per coordinate')
        object.__setattr__(self, 'maps', maps)

    @property
    def grid_size(self) -> Tuple[int, ...]:
        """Grid size of each constituent coordinate map."""
        return tuple(item.grid_size[0] for item in self.maps)

    def _apply(self, method: str, values: torch.Tensor) -> torch.Tensor:
        """Applies each constituent map to its coordinate column."""
        if not isinstance(values, torch.Tensor) or (values.ndim < 1) or (
                values.shape[-1] != len(self.maps)):
            raise ValueError(
                '`values` should contain one column per coordinate map')
        return torch.cat([getattr(item, method)(values[..., site:site + 1])
                          for site, item in enumerate(self.maps)], dim=-1)

    def forward(self, unit_coordinates: torch.Tensor) -> torch.Tensor:
        """Maps each unit coordinate through its constituent map."""
        return self._apply('forward', unit_coordinates)

    def inverse(self, domain_coordinates: torch.Tensor) -> torch.Tensor:
        """Maps each domain coordinate through its constituent inverse map."""
        return self._apply('inverse', domain_coordinates)

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Selects grid points through each constituent map."""
        return self._apply('from_indices', indices)

    def to_indices(self, domain_coordinates: torch.Tensor) -> torch.Tensor:
        """Quantizes domain coordinates through each constituent map."""
        return self._apply('to_indices', domain_coordinates)
