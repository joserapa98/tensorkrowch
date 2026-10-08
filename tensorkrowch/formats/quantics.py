"""
This script contains:

    Internal classes:
        * _QuanticsFormat
        * _QuanticsVector
        * _QuanticsMatrix

    Public classes:
        * QTT, QTR, QTTM, QTRM

    Internal functions:
        * _map_structure
        * _equal_structure
        * _same_references
        * _resolve_quantization

Terminology:
    * Coordinates are continuous input values in their original domains.
    * A grid provides discrete coordinate values; an index selects one per
      coordinate.
    * Digits expand each index in its chosen base and select TT core inputs.
    * A layout fixes the order of these digit sites across coordinates.

Thus ``evaluate_coordinates`` maps coordinates to grid indices, and
``evaluate_indices`` encodes those indices as digits before evaluating the
cores. ``evaluate_digits`` accepts digits already in layout order.
"""

from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable, Optional, Sequence, Tuple, Type, Union

import torch

from tensorkrowch.formats.formats1d import (_restore_cores, TensorFormat1D,
                                            _VectorFormat1D,
                                            TT, TR, TTM, TRM)
from tensorkrowch.formats.quantization import (Domain, QuantizedLayout,
                                               CoordinateMap,
                                               AffineCoordinateMap,
                                               ExplicitGridMap)


def _map_structure(value: Any,
                   function: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Maps stored coordinate tensors without casting real grids to complex."""
    if isinstance(value, torch.Tensor):
        result = function(value)
        return (result.real if not value.is_complex() and result.is_complex()
                else result)
    if is_dataclass(value):
        mapped = {field.name: _map_structure(getattr(value, field.name), function)
                  for field in fields(value) if field.init}
        return replace(value, **mapped)
    if isinstance(value, tuple):
        return tuple(_map_structure(item, function) for item in value)
    if isinstance(value, list):
        return [_map_structure(item, function) for item in value]
    return value


def _equal_structure(first: Any, second: Any) -> bool:
    """Compares coordinate meaning without tensor truth-value ambiguities."""
    if first is second:
        return True
    if type(first) is not type(second):
        return False
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if is_dataclass(first):
        return all(_equal_structure(getattr(first, field.name),
                                    getattr(second, field.name))
                   for field in fields(first))
    if isinstance(first, (list, tuple)):
        return (len(first) == len(second)) and all(
            _equal_structure(left, right) for left, right in zip(first, second))
    if callable(first):
        return False
    return first == second


def _same_references(first: Any, second: Any) -> bool:
    """Compares storage references to decide whether a conversion was a no-op."""
    if isinstance(first, torch.Tensor):
        return first is second
    if is_dataclass(first):
        return all(_same_references(getattr(first, field.name),
                                    getattr(second, field.name))
                   for field in fields(first))
    if isinstance(first, (list, tuple)):
        return all(_same_references(left, right)
                   for left, right in zip(first, second))
    return True


def _resolve_quantization(
        n_coordinates: int,
        *,
        base: Optional[Union[int, Sequence[int]]],
        level: Optional[Union[int, Sequence[int]]],
        domain: Domain,
        grid_coordinates: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
        layout: Optional[QuantizedLayout],
        coordinate_map: Optional[CoordinateMap]
        ) -> Tuple[QuantizedLayout, CoordinateMap]:
    """Builds or accepts a layout and map, then checks their compatibility."""
    if isinstance(n_coordinates, bool) or not isinstance(n_coordinates, int):
        raise TypeError('`n_coordinates` should be int type')
    if n_coordinates < 1:
        raise ValueError('`n_coordinates` should be positive')

    if layout is not None or coordinate_map is not None:
        if any(value is not None for value in (
                base, level, domain, grid_coordinates)):
            raise ValueError(
                'Do not combine `layout` or `coordinate_map` with '
                'shorthand arguments')
        if not isinstance(layout, QuantizedLayout):
            raise TypeError('`layout` should be QuantizedLayout type')
        if not isinstance(coordinate_map, CoordinateMap):
            raise TypeError('`coordinate_map` should be CoordinateMap type')
    else:
        if base is None or level is None:
            raise ValueError(
                '`base` and `level` are required without `layout`')
        layout = QuantizedLayout(n_coordinates, base, level,
                                 ordering='interleaved',
                                 digit_order='coarse_to_fine')
        if grid_coordinates is not None:
            if domain is not None:
                raise ValueError(
                    '`domain` should be None with `grid_coordinates`')
            coordinate_map = ExplicitGridMap(grid_coordinates)
        else:
            if domain is None:
                raise ValueError(
                    '`domain` is required without `grid_coordinates`')
            coordinate_map = AffineCoordinateMap(domain, layout.grid_size)

    if layout.n_coordinates != n_coordinates:
        raise ValueError('`n_coordinates` should match `layout.n_coordinates`')
    if coordinate_map.grid_size != layout.grid_size:
        raise ValueError(
            '`coordinate_map.grid_size` should match `layout.grid_size`')
    return layout, coordinate_map


class _QuanticsFormat:  # MARK: _QuanticsFormat
    """Coordinate compatibility and direct construction of Quantics results."""

    _quantized = True
    _coordinate_names = ()

    def _map_tensors(
            self, function: Callable[[torch.Tensor], torch.Tensor]
        ) -> TensorFormat1D:
        """Maps cores, factors and tensors stored in coordinate metadata."""
        result = super()._map_tensors(function)
        for name in self._coordinate_names:
            setattr(result, name, _map_structure(getattr(self, name), function))
        return result

    def _same_aux_tensors(self, other: TensorFormat1D) -> bool:
        """Checks whether coordinate tensors retain their original references."""
        return all(_same_references(getattr(self, name), getattr(other, name))
                   for name in self._coordinate_names)

    def _to_format(self, cls: Type[TensorFormat1D]) -> TensorFormat1D:
        """Drops coordinate metadata while retaining cores and bond factors."""
        result = cls(self._cores, n_batches=self._n_batches)
        if self._bonds is not None:
            result._bonds = self._bonds._map_tensors(
                lambda tensor: tensor, result._on_bonds_changed)
        if self._out_dim is None:
            result._is_row = self._is_row
        return result

    def _check_semantics(self,
                         other: TensorFormat1D,
                         product: bool = False) -> None:
        """Checks layouts, coordinate maps and contracted coordinate spaces."""
        super()._check_semantics(other, product=product)
        if product:
            left = (self.in_layout, self.in_coordinate_map)
            right = ((other.out_layout, other.out_coordinate_map)
                     if isinstance(other, _QuanticsMatrix)
                     else (other.layout, other.coordinate_map))
            if not _equal_structure(left, right):
                raise ValueError(
                    'Contracted Quantics coordinate spaces should match')
        else:
            names = (('layout', 'coordinate_map')
                     if isinstance(self, _QuanticsVector)
                     else ('in_layout', 'out_layout',
                           'in_coordinate_map', 'out_coordinate_map'))
            if any(not _equal_structure(getattr(self, name),
                                        getattr(other, name, None))
                   for name in names):
                raise ValueError(
                    'Quantics layouts and coordinate maps should match')

    def _new_from_standard_cores(self,
                                 cores: Sequence[torch.Tensor],
                                 in_dim: Sequence[int],
                                 out_dim: Optional[Sequence[int]],
                                 n_batches: int,
                                 cyclic: bool,
                                 other: Optional[TensorFormat1D] = None,
                                 product: bool = False) -> TensorFormat1D:
        """Constructs algebra results with their input and output coordinate maps."""
        cores = _restore_cores(cores, in_dim, out_dim, n_batches, cyclic)
        if out_dim is not None:
            cls = QTRM if cyclic else QTTM
            input_format = other if product else self
            return cls(
                cores=cores,
                in_n_coordinates=input_format.in_layout.n_coordinates,
                out_n_coordinates=self.out_layout.n_coordinates,
                in_layout=input_format.in_layout,
                out_layout=self.out_layout,
                in_coordinate_map=input_format.in_coordinate_map,
                out_coordinate_map=self.out_coordinate_map,
                n_batches=n_batches)

        cls = QTR if cyclic else QTT
        layout = self.out_layout if product else self.layout
        coordinate_map = (self.out_coordinate_map if product
                          else self.coordinate_map)
        return cls(cores=cores,
                   n_coordinates=layout.n_coordinates,
                   layout=layout,
                   coordinate_map=coordinate_map,
                   n_batches=n_batches)


class _QuanticsVector(_QuanticsFormat):  # MARK: _QuanticsVector
    """Coordinate semantics shared by open and cyclic Quantics vectors."""

    _coordinate_names = ('coordinate_map',)

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 n_coordinates: int,
                 *,
                 base: Optional[Union[int, Sequence[int]]] = None,
                 level: Optional[Union[int, Sequence[int]]] = None,
                 domain: Domain = None,
                 grid_coordinates: Optional[Union[
                     torch.Tensor, Sequence[torch.Tensor]]] = None,
                 layout: Optional[QuantizedLayout] = None,
                 coordinate_map: Optional[CoordinateMap] = None,
                 n_batches: int = 0,
                 bonds: Optional[Sequence[Optional[torch.Tensor]]] = None) -> None:
        self.layout, self.coordinate_map = _resolve_quantization(
            n_coordinates, base=base, level=level, domain=domain,
            grid_coordinates=grid_coordinates, layout=layout,
            coordinate_map=coordinate_map)
        self.n_coordinates = n_coordinates
        super().__init__(cores, n_batches=n_batches, bonds=bonds)

    def _new_from_standard_cores(self,
                                 cores: Sequence[torch.Tensor],
                                 in_dim: Sequence[int],
                                 out_dim: Optional[Sequence[int]],
                                 n_batches: int,
                                 cyclic: bool,
                                 other: Optional[TensorFormat1D] = None,
                                 product: bool = False) -> TensorFormat1D:
        """Preserves Quantics vector orientation in algebra results."""
        result = super()._new_from_standard_cores(
            cores, in_dim, out_dim, n_batches, cyclic,
            other=other, product=product)
        if not product:
            result._is_row = self._is_row
        return result

    def _new_outer_product(self,
                           cores: Sequence[torch.Tensor],
                           other: _VectorFormat1D,
                           n_batches: int,
                           cyclic: bool) -> Union['QTTM', 'QTRM']:
        """Builds an operator with the row's inputs and this column's outputs."""
        cores = _restore_cores(
            cores, other._in_dim, self._in_dim, n_batches, cyclic)
        cls = QTRM if cyclic else QTTM
        return cls(cores=cores,
                   in_n_coordinates=other.layout.n_coordinates,
                   out_n_coordinates=self.layout.n_coordinates,
                   in_layout=other.layout,
                   out_layout=self.layout,
                   in_coordinate_map=other.coordinate_map,
                   out_coordinate_map=self.coordinate_map,
                   n_batches=n_batches)

    def validate(self) -> TensorFormat1D:
        """
        Validates cores and their Quantics digit-layout dimensions.

        Returns
        -------
        :class:`~tensorkrowch.formats.TensorFormat1D`
            The current format. Invalid controlled replacement restores prior
            metadata.
        """
        super().validate()
        if self._in_dim != self.layout.in_dim:
            raise ValueError(
                'Digit core dimensions should match `layout.in_dim`')
        return self

    def evaluate_digits(self, digits: torch.Tensor) -> torch.Tensor:
        """
        Evaluates scheduled digit configurations.

        Parameters
        ----------
        digits : torch.Tensor
            Integer digit configurations in layout schedule order, with shape
            ``(*data_batch, layout.n_sites)``. Every digit should lie within
            its site base.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *data_batch)``.

        Examples
        --------
        >>> format = tk.formats.QTT([torch.eye(2), torch.eye(2)], 1,
        ...     base=2, level=2, domain=torch.tensor([0., 1.]))
        >>> format.evaluate_digits(torch.tensor([[0, 0], [0, 1], [1, 1]])).tolist()
        [1.0, 0.0, 1.0]
        """
        digits = self.layout._validate_digits(digits).to(self.device)
        return self.evaluate(digits, n_batches=digits.ndim - 1)

    def evaluate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Evaluates original coordinate index configurations.

        Encodes indices through
        :meth:`~tensorkrowch.formats.QuantizedLayout.encode_indices` and calls
        :meth:`evaluate_digits`.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape ``(*data_batch, n_coordinates)``;
            each value lies in ``[0, grid_size[coordinate] - 1]``.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *data_batch)``.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(1, 2, 2)
        >>> coordinate_map = tk.formats.AffineCoordinateMap(
        ...     domain=torch.tensor([0., 1.]), grid_size=layout.grid_size)
        >>> format = tk.formats.QTT([torch.eye(2), torch.eye(2)], 1,
        ...     layout=layout, coordinate_map=coordinate_map)
        >>> format.evaluate_indices(torch.tensor([[0], [3]])).tolist()
        [1.0, 1.0]
        """
        return self.evaluate_digits(self.layout.encode_indices(indices))

    def evaluate_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Evaluates coordinate configurations in the domain.

        The coordinate map quantizes the inputs using its stored grid
        and out-of-domain policy.

        Parameters
        ----------
        coordinates : torch.Tensor
            Finite coordinates in the domain with shape
            ``(*data_batch, n_coordinates)``. A coordinate map is required.
            Coordinates are quantized to the computational grid; no
            interpolation of the represented function is performed.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *data_batch)``.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(1, 2, 2)
        >>> format = tk.formats.QTT([torch.eye(2), torch.eye(2)], 1,
        ...     layout=layout, coordinate_map=tk.formats.AffineCoordinateMap(
        ...         domain=torch.tensor([0., 3.]), grid_size=layout.grid_size))
        >>> format.evaluate_coordinates(torch.tensor([[0.], [3.]])).tolist()
        [1.0, 1.0]
        """
        indices = self.coordinate_map.to_indices(coordinates)
        return self.evaluate_indices(indices)

    def to_dense_grid(self) -> torch.Tensor:
        """
        Evaluates every original coordinate index on a small grid. Explicitly
        allocates and evaluates the full grid.

        Returns one axis per original coordinate, combining its digits and
        restoring coordinate order independently of the layout schedule.
        In contrast, :meth:`contract_dense` returns one axis per digit site
        in layout order.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *grid_size)``, in
            original coordinate order.
        """
        axes = [torch.arange(size, device=self.device)
                for size in self.layout.grid_size]
        indices = torch.cartesian_prod(*axes).reshape(-1, self.layout.n_coordinates)
        values = self.evaluate_indices(indices)
        return values.reshape(*self._batch_shape, *self.layout.grid_size)


class QTT(_QuanticsVector, TT):  # MARK: QTT
    """
    Quantics tensor train with a :class:`~tensorkrowch.formats.QuantizedLayout`
    and a :class:`~tensorkrowch.formats.CoordinateMap`.

    Every core represents one digit site; evaluation contracts all sites.

    Shorthand construction uses ``base``, ``level`` and either ``domain`` or
    ``grid_coordinates``. Its layout is ``interleaved`` and ``coarse_to_fine``;
    domain intervals produce an affine map with ``grid_offset="left"``. For other
    choices, supply ``layout`` and ``coordinate_map`` explicitly.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described in :class:`~tensorkrowch.formats.TT`.
        The container is copied; tensors retain storage and autograd.
    n_coordinates : int
        Number of original input coordinates.
    base : int or sequence of int, optional
        Digit bases for shorthand construction. Required together with ``level``
        and either ``domain`` or ``grid_coordinates``.
    level : int or sequence of int, optional
        Number of digits per coordinate for shorthand construction.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Intervals used to build an :class:`~tensorkrowch.formats.AffineCoordinateMap`
        with ``grid_offset="left"``. Cannot be combined with
        ``grid_coordinates`` or
        prebuilt objects.
    grid_coordinates : torch.Tensor or sequence of torch.Tensor, optional
        Grid points in the domain, used to build an
        :class:`~tensorkrowch.formats.ExplicitGridMap`. A vector describes one
        coordinate; a matrix or sequence of vectors describes multiple
        coordinates. Sizes must equal ``base ** level`` exactly.
    layout : :class:`~tensorkrowch.formats.QuantizedLayout`, optional
        Prebuilt digit layout, supplied together with ``coordinate_map``.
        Cannot be combined with shorthand construction arguments.
    coordinate_map : :class:`~tensorkrowch.formats.CoordinateMap`, optional
        Prebuilt map containing the domain, grid and conversion policy.
        Its ``grid_size`` must match ``layout.grid_size``.
    n_batches : int
        Number of leading structural batch axes shared by the cores.
        Independent of data batches during evaluation.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores, as in the corresponding plain format.

    Examples
    --------
    First, discretize domain coordinates into indices on a four-point grid:

    >>> coordinate_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([0., 1.]), grid_size=(4,))
    >>> coordinates = torch.tensor([[0.], [0.25], [0.75]])
    >>> indices = coordinate_map.to_indices(coordinates)
    >>> indices.tolist()
    [[0], [1], [3]]

    Then, expand each index into two binary digits, one per TT site:

    >>> layout = tk.formats.QuantizedLayout(1, base=2, level=2)
    >>> layout.encode_indices(indices).tolist()
    [[0, 0], [0, 1], [1, 1]]

    Finally, construct the QTT with these objects and evaluate the coordinates:

    >>> format = tk.formats.QTT([torch.eye(2), torch.eye(2)], 1,
    ...     layout=layout, coordinate_map=coordinate_map)
    >>> format.evaluate_coordinates(coordinates).tolist()
    [1.0, 0.0, 1.0]

    With two coordinates, interleave their digits and evaluate both together:

    >>> coordinate_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([0., 1.]), grid_size=(4, 4))
    >>> layout = tk.formats.QuantizedLayout(2, base=2, level=2)
    >>> cores = [torch.tensor([[1.], [2.]]), torch.ones(1, 2, 1),
    ...          torch.ones(1, 2, 1), torch.tensor([[3., 4.]])]
    >>> format = tk.formats.QTT(cores, 2,
    ...     layout=layout, coordinate_map=coordinate_map)
    >>> coordinates = torch.tensor([[0., 0.], [0.5, 0.25]])
    >>> layout.encode_indices(coordinate_map.to_indices(coordinates)).tolist()
    [[0, 0, 0, 0], [1, 0, 0, 1]]
    >>> format.evaluate_coordinates(coordinates).tolist()
    [3.0, 8.0]
    """

    def to_tt(self) -> TT:
        """
        Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        :class:`~tensorkrowch.formats.TT`
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        return self._to_format(TT)


class QTR(_QuanticsVector, TR):  # MARK: QTR
    """
    Quantics tensor ring with a :class:`~tensorkrowch.formats.QuantizedLayout`
    and a :class:`~tensorkrowch.formats.CoordinateMap`.

    Every core represents one digit site; evaluation contracts all sites.

    Shorthand construction uses ``base``, ``level`` and either ``domain`` or
    ``grid_coordinates``. Its layout is ``interleaved`` and ``coarse_to_fine``;
    domain intervals produce an affine map with ``grid_offset="left"``. For other
    choices, supply ``layout`` and ``coordinate_map`` explicitly.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described in :class:`~tensorkrowch.formats.TR`.
        The container is copied; tensors retain storage and autograd.
    n_coordinates : int
        Number of original input coordinates.
    base : int or sequence of int, optional
        Digit bases for shorthand construction. Required together with ``level``
        and either ``domain`` or ``grid_coordinates``.
    level : int or sequence of int, optional
        Number of digits per coordinate for shorthand construction.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Intervals used to build an :class:`~tensorkrowch.formats.AffineCoordinateMap`
        with ``grid_offset="left"``. Cannot be combined with
        ``grid_coordinates`` or
        prebuilt objects.
    grid_coordinates : torch.Tensor or sequence of torch.Tensor, optional
        Grid points in the domain, used to build an
        :class:`~tensorkrowch.formats.ExplicitGridMap`. A vector describes one
        coordinate; a matrix or sequence of vectors describes multiple
        coordinates. Sizes must equal ``base ** level`` exactly.
    layout : :class:`~tensorkrowch.formats.QuantizedLayout`, optional
        Prebuilt digit layout, supplied together with ``coordinate_map``.
        Cannot be combined with shorthand construction arguments.
    coordinate_map : :class:`~tensorkrowch.formats.CoordinateMap`, optional
        Prebuilt map containing the domain, grid and conversion policy.
        Its ``grid_size`` must match ``layout.grid_size``.
    n_batches : int
        Number of leading structural batch axes shared by the cores.
        Independent of data batches during evaluation.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores, as in the corresponding plain format.

    Examples
    --------
    First, discretize domain coordinates into indices on a four-point grid:

    >>> coordinate_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([0., 1.]), grid_size=(4,))
    >>> coordinates = torch.tensor([[0.], [0.25], [0.75]])
    >>> indices = coordinate_map.to_indices(coordinates)
    >>> indices.tolist()
    [[0], [1], [3]]

    Then, expand each index into two binary digits, one per ring site:

    >>> layout = tk.formats.QuantizedLayout(1, base=2, level=2)
    >>> layout.encode_indices(indices).tolist()
    [[0, 0], [0, 1], [1, 1]]

    Finally, construct a ring whose cores select matching digits:

    >>> core = torch.tensor([[[1., 0.], [0., 0.]],
    ...                      [[0., 0.], [0., 1.]]])
    >>> format = tk.formats.QTR([core, core], 1,
    ...     layout=layout, coordinate_map=coordinate_map)
    >>> format.evaluate_coordinates(coordinates).tolist()
    [1.0, 0.0, 1.0]

    With two coordinates, interleave their digits and evaluate both together:

    >>> coordinate_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([0., 1.]), grid_size=(4, 4))
    >>> layout = tk.formats.QuantizedLayout(2, base=2, level=2)
    >>> cores = [torch.tensor([1., 2.]).reshape(1, 2, 1),
    ...          torch.ones(1, 2, 1), torch.ones(1, 2, 1),
    ...          torch.tensor([3., 4.]).reshape(1, 2, 1)]
    >>> format = tk.formats.QTR(cores, 2,
    ...     layout=layout, coordinate_map=coordinate_map)
    >>> coordinates = torch.tensor([[0., 0.], [0.5, 0.25]])
    >>> layout.encode_indices(coordinate_map.to_indices(coordinates)).tolist()
    [[0, 0, 0, 0], [1, 0, 0, 1]]
    >>> format.evaluate_coordinates(coordinates).tolist()
    [3.0, 8.0]
    """

    def to_tr(self) -> TR:
        """
        Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        :class:`~tensorkrowch.formats.TR`
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        return self._to_format(TR)

    def to_qtt(self) -> QTT:
        """
        Opens the ring while preserving its digit layout and coordinate map.

        Carries the closing index through the chain, as in
        :meth:`~tensorkrowch.formats.TR.to_tt`, without truncation or dense
        reconstruction.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTT`
            Open Quantics format representing the same values, with structural
            batches and vector orientation preserved.
        """
        return super().to_tt()

    def to_tt(self) -> TT:
        """
        Requires an explicit choice about retaining Quantics information.

        Use :meth:`to_qtt` to open the ring while retaining Quantics, or
        :meth:`to_tr` to remove Quantics while retaining the ring. To obtain
        a plain TT, call ``to_qtt().to_tt()`` or ``to_tr().to_tt()``.

        Raises
        ------
        NotImplementedError
            Direct conversion is disabled to make the intended format explicit.
        """
        raise NotImplementedError(
            'Use `to_qtt()` to retain Quantics, or `to_qtt().to_tt()` '
            'or `to_tr().to_tt()` to obtain a plain TT')

    def rotate(self, first: int = 0) -> 'QTR':
        """
        Rotates ring sites and updates the Quantics digit schedules.

        Parameters
        ----------
        first : int
            Site that becomes index zero, in ``[0, n_sites - 1]``.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTR`
            Rotated format preserving evaluations at coordinates in the
            original domain. Dense digit axes rotate with the core order.
        """
        if isinstance(first, bool) or not isinstance(first, int):
            raise TypeError('`first` should be int type')
        if not 0 <= first < self.n_sites:
            raise ValueError('`first` should lie inside the format')

        cores = [*self._cores[first:], *self._cores[:first]]
        schedule = self.layout.sites()
        layout = self.layout if first == 0 else QuantizedLayout(
            n_coordinates=self.layout.n_coordinates,
            base=self.layout.base,
            level=self.layout.level,
            ordering='custom',
            digit_order=self.layout.digit_order,
            permutation=(*schedule[first:], *schedule[:first])
        )

        result = QTR(cores=cores,
                     n_coordinates=self.layout.n_coordinates,
                     layout=layout,
                     coordinate_map=self.coordinate_map,
                     n_batches=self._n_batches)
        result._is_row = self._is_row
        if self._bonds is not None:
            factors = self._bonds.factors
            result.bonds = [*factors[first:], *factors[:first]]
        return result


class _QuanticsMatrix(_QuanticsFormat):  # MARK: _QuanticsMatrix
    """Paired input/output digit layouts of a tensorized operator."""

    _coordinate_names = ('in_coordinate_map', 'out_coordinate_map')

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 in_n_coordinates: int,
                 out_n_coordinates: int,
                 *,
                 in_base: Optional[Union[int, Sequence[int]]] = None,
                 out_base: Optional[Union[int, Sequence[int]]] = None,
                 in_level: Optional[Union[int, Sequence[int]]] = None,
                 out_level: Optional[Union[int, Sequence[int]]] = None,
                 in_domain: Domain = None,
                 out_domain: Domain = None,
                 in_grid_coordinates: Optional[Union[
                     torch.Tensor, Sequence[torch.Tensor]]] = None,
                 out_grid_coordinates: Optional[Union[
                     torch.Tensor, Sequence[torch.Tensor]]] = None,
                 in_layout: Optional[QuantizedLayout] = None,
                 out_layout: Optional[QuantizedLayout] = None,
                 in_coordinate_map: Optional[CoordinateMap] = None,
                 out_coordinate_map: Optional[CoordinateMap] = None,
                 n_batches: int = 0,
                 bonds: Optional[Sequence[Optional[torch.Tensor]]] = None
                 ) -> None:
        self.in_layout, self.in_coordinate_map = _resolve_quantization(
            in_n_coordinates, base=in_base, level=in_level, domain=in_domain,
            grid_coordinates=in_grid_coordinates, layout=in_layout,
            coordinate_map=in_coordinate_map)
        self.out_layout, self.out_coordinate_map = _resolve_quantization(
            out_n_coordinates, base=out_base, level=out_level, domain=out_domain,
            grid_coordinates=out_grid_coordinates, layout=out_layout,
            coordinate_map=out_coordinate_map)
        if self.in_layout.n_sites != self.out_layout.n_sites:
            raise ValueError(
                '`in_layout` and `out_layout` should have equal `n_sites`')
        self.in_n_coordinates = in_n_coordinates
        self.out_n_coordinates = out_n_coordinates
        super().__init__(cores, n_batches=n_batches, bonds=bonds)

    def validate(self) -> TensorFormat1D:
        """
        Validates cores and their Quantics digit-layout dimensions.

        Returns
        -------
        :class:`~tensorkrowch.formats.TensorFormat1D`
            The current format. Invalid controlled replacement restores prior
            metadata.
        """
        super().validate()
        if ((self._in_dim != self.in_layout.in_dim) or
                (self._out_dim != self.out_layout.in_dim)):
            raise ValueError(
                'Matrix core dimensions should match paired digit layouts')
        return self

    def transpose(self) -> Union['QTTM', 'QTRM']:
        """
        Swaps local matrix input/output axes without reversing sites.

        Also available through the :attr:`T` property.

        The result keeps the concrete format class and its methods. Its core
        and bond containers are independent, sharing tensor storage.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTTM` or :class:`~tensorkrowch.formats.QTRM`
            Separate format sharing tensor storage. Sites retain their order.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(1, 2, 1)
        >>> core = torch.tensor([[1+2j, 3+4j], [5+6j, 7+8j]])
        >>> coordinate_map = tk.formats.AffineCoordinateMap(
        ...     domain=torch.tensor([0., 1.]), grid_size=layout.grid_size)
        >>> matrix = tk.formats.QTTM([core], 1, 1,
        ...     in_layout=layout, out_layout=layout,
        ...     in_coordinate_map=coordinate_map,
        ...     out_coordinate_map=coordinate_map)
        >>> matrix.transpose().contract_dense().tolist()
        [[(1+2j), (5+6j)], [(3+4j), (7+8j)]]
        >>> torch.equal(matrix.T.contract_dense(),
        ...             matrix.transpose().contract_dense())
        True
        """
        result = super().transpose()
        result.in_n_coordinates, result.out_n_coordinates = (
            self.out_n_coordinates, self.in_n_coordinates)
        result.in_layout, result.out_layout = self.out_layout, self.in_layout
        result.in_coordinate_map, result.out_coordinate_map = (
            self.out_coordinate_map, self.in_coordinate_map)
        return result

    def evaluate_digits(self,
                        in_digits: torch.Tensor,
                        out_digits: torch.Tensor) -> torch.Tensor:
        """
        Evaluates matrix entries using paired digits.

        Parameters
        ----------
        in_digits : torch.Tensor
            Input digits with shape ``(*data_batch, in_layout.n_sites)``.
            Values follow the input digit schedule and should lie within each
            site base.
        out_digits : torch.Tensor
            Output digits with shape ``(*data_batch, out_layout.n_sites)``,
            within the output site bases. Data batches match the inputs.

        Returns
        -------
        torch.Tensor
            Matrix entries with shape ``(*core_batch, *data_batch)``.
        """
        in_digits = self.in_layout._validate_digits(in_digits)
        out_digits = self.out_layout._validate_digits(out_digits)
        return self.evaluate(in_digits, out_digits, n_batches=in_digits.ndim - 1)

    def evaluate_indices(self,
                         in_indices: torch.Tensor,
                         out_indices: torch.Tensor) -> torch.Tensor:
        """
        Evaluates matrix entries using paired indices.

        Parameters
        ----------
        in_indices : torch.Tensor
            Input indices with shape ``(*data_batch, in_layout.n_coordinates)``.
            Indices lie within the grid bounds of each original input coordinate.
        out_indices : torch.Tensor
            Output indices with shape
            ``(*data_batch, out_layout.n_coordinates)``, within the output grid
            bounds. Data batches match the inputs.

        Returns
        -------
        torch.Tensor
            Matrix entries with shape ``(*core_batch, *data_batch)``.
        """
        return self.evaluate_digits(self.in_layout.encode_indices(in_indices),
                                    self.out_layout.encode_indices(out_indices))

    def evaluate_coordinates(self,
                             in_coordinates: torch.Tensor,
                             out_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Evaluates matrix entries using paired coordinates.

        Parameters
        ----------
        in_coordinates : torch.Tensor
            Finite floating input coordinates with shape
            ``(*data_batch, in_layout.n_coordinates)``. An input coordinate map
            with an inverse or grid lookup is required.
        out_coordinates : torch.Tensor
            Finite floating output coordinates with shape
            ``(*data_batch, out_layout.n_coordinates)``, with matching data
            batches. An output coordinate map with an inverse or grid lookup is
            required.

        Returns
        -------
        torch.Tensor
            Entries with shape ``(*core_batch, *data_batch)``. Evaluation
            in the domain quantizes both coordinate groups; it does not
            interpolate matrix entries.
        """
        in_indices = self.in_coordinate_map.to_indices(in_coordinates)
        out_indices = self.out_coordinate_map.to_indices(out_coordinates)
        return self.evaluate_indices(in_indices, out_indices)

    def to_dense_grid(self) -> torch.Tensor:
        """
        Evaluates every original coordinate index on a small grid.

        Returns one axis per original input and output coordinate, combining
        their digits and placing input coordinates before output coordinates.
        In contrast, :meth:`contract_dense` returns the interleaved input and
        output digit axes in layout order. Explicitly allocates and evaluates
        the full grid.

        Returns
        -------
        torch.Tensor
            Values with shape
            ``(*core_batch, *input_grid_size, *output_grid_size)``, placing all
            input coordinates before all output coordinates.
        """
        input_axes = [torch.arange(size, device=self.device)
                      for size in self.in_layout.grid_size]
        output_axes = [torch.arange(size, device=self.device)
                       for size in self.out_layout.grid_size]
        in_indices = torch.cartesian_prod(*input_axes).reshape(
            -1, self.in_layout.n_coordinates)
        out_indices = torch.cartesian_prod(*output_axes).reshape(
            -1, self.out_layout.n_coordinates)

        # Evaluate the Cartesian product of input and output configurations.
        values = self.evaluate_indices(
            in_indices.repeat_interleave(out_indices.shape[0], 0),
            out_indices.repeat(in_indices.shape[0], 1))
        return values.reshape(*self._batch_shape,
                              *self.in_layout.grid_size,
                              *self.out_layout.grid_size)


class QTTM(_QuanticsMatrix, TTM):  # MARK: QTTM
    """
    Quantics tensor train operator with a
    :class:`~tensorkrowch.formats.QuantizedLayout` and a
    :class:`~tensorkrowch.formats.CoordinateMap` for each input and output
    space.

    Each space independently accepts shorthand arguments or a prebuilt layout
    and map, as in :class:`QTT`. Shorthand layouts are ``interleaved`` and
    ``coarse_to_fine``; shorthand affine maps use ``grid_offset="left"``.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described in :class:`~tensorkrowch.formats.TTM`.
        The container is copied; tensors retain storage and autograd.
    in_n_coordinates, out_n_coordinates : int
        Number of original input and output coordinates.
    in_base, out_base : int or sequence of int, optional
        Digit bases for shorthand construction of each coordinate space.
        Each base requires its corresponding level and domain or grid points.
    in_level, out_level : int or sequence of int, optional
        Number of digits per coordinate in each space.
    in_domain, out_domain : torch.Tensor or sequence of torch.Tensor, optional
        Intervals used to build affine maps with ``grid_offset="left"``.
        Cannot be combined with grid points or prebuilt objects for that space.
    in_grid_coordinates, out_grid_coordinates : torch.Tensor or sequence, optional
        Domain grid points used to build explicit maps. A vector describes one
        coordinate; a matrix or sequence of vectors describes multiple
        coordinates. Grid sizes must match the corresponding ``base ** level``.
    in_layout, out_layout : :class:`~tensorkrowch.formats.QuantizedLayout`, optional
        Prebuilt digit layouts, each supplied with its corresponding map.
        Input and output layouts must have the same number of sites.
    in_coordinate_map, out_coordinate_map : :class:`~tensorkrowch.formats.CoordinateMap`, optional
        Prebuilt maps, each with grid sizes matching its layout. Cannot be
        combined with shorthand arguments for that coordinate space.
    n_batches : int
        Number of leading structural batch axes shared by the cores.
        Independent of data batches during evaluation.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores, as in the corresponding plain format.

    Examples
    --------
    First, discretize input and output coordinates on their respective grids:

    >>> in_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([0., 1.]), grid_size=(4,))
    >>> out_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([-1., 1.]), grid_size=(4,))
    >>> in_coordinates = torch.tensor([[0.], [0.5]])
    >>> out_coordinates = torch.tensor([[-1.], [0.5]])
    >>> in_indices = in_map.to_indices(in_coordinates)
    >>> out_indices = out_map.to_indices(out_coordinates)
    >>> in_indices.tolist(), out_indices.tolist()
    ([[0], [2]], [[0], [3]])

    Then, expand both sets of indices into the digits paired at each site:

    >>> in_layout = tk.formats.QuantizedLayout(1, base=2, level=2)
    >>> out_layout = tk.formats.QuantizedLayout(1, base=2, level=2)
    >>> in_layout.encode_indices(in_indices).tolist()
    [[0, 0], [1, 0]]
    >>> out_layout.encode_indices(out_indices).tolist()
    [[0, 0], [1, 1]]

    Finally, construct an identity matrix on the grid indices and evaluate
    entries:

    >>> cores = [torch.eye(2).unsqueeze(1), torch.eye(2).unsqueeze(0)]
    >>> matrix = tk.formats.QTTM(cores, 1, 1,
    ...     in_layout=in_layout, out_layout=out_layout,
    ...     in_coordinate_map=in_map, out_coordinate_map=out_map)
    >>> matrix.evaluate_coordinates(in_coordinates, out_coordinates).tolist()
    [1.0, 0.0]

    With two coordinates per space, interleave their digits in both layouts:

    >>> coordinate_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([0., 1.]), grid_size=(4, 4))
    >>> layout = tk.formats.QuantizedLayout(2, base=2, level=2)
    >>> cores = [torch.eye(2).unsqueeze(1),
    ...          torch.eye(2).reshape(1, 2, 1, 2),
    ...          torch.eye(2).reshape(1, 2, 1, 2), torch.eye(2).unsqueeze(0)]
    >>> matrix = tk.formats.QTTM(cores, 2, 2,
    ...     in_layout=layout, out_layout=layout,
    ...     in_coordinate_map=coordinate_map, out_coordinate_map=coordinate_map)
    >>> in_coordinates = torch.tensor([[0., 0.], [0.5, 0.25]])
    >>> out_coordinates = torch.tensor([[0., 0.], [0.5, 0.75]])
    >>> matrix.evaluate_coordinates(in_coordinates, out_coordinates).tolist()
    [1.0, 0.0]
    """

    def to_ttm(self) -> TTM:
        """
        Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        :class:`~tensorkrowch.formats.TTM`
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        return self._to_format(TTM)


class QTRM(_QuanticsMatrix, TRM):  # MARK: QTRM
    """
    Quantics tensor ring operator with a
    :class:`~tensorkrowch.formats.QuantizedLayout` and a
    :class:`~tensorkrowch.formats.CoordinateMap` for each input and output
    space.

    Each space independently accepts shorthand arguments or a prebuilt layout
    and map, as in :class:`QTT`. Shorthand layouts are ``interleaved`` and
    ``coarse_to_fine``; shorthand affine maps use ``grid_offset="left"``.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described in :class:`~tensorkrowch.formats.TRM`.
        The container is copied; tensors retain storage and autograd.
    in_n_coordinates, out_n_coordinates : int
        Number of original input and output coordinates.
    in_base, out_base : int or sequence of int, optional
        Digit bases for shorthand construction of each coordinate space.
        Each base requires its corresponding level and domain or grid points.
    in_level, out_level : int or sequence of int, optional
        Number of digits per coordinate in each space.
    in_domain, out_domain : torch.Tensor or sequence of torch.Tensor, optional
        Intervals used to build affine maps with ``grid_offset="left"``.
        Cannot be combined with grid points or prebuilt objects for that space.
    in_grid_coordinates, out_grid_coordinates : torch.Tensor or sequence, optional
        Domain grid points used to build explicit maps. A vector describes one
        coordinate; a matrix or sequence of vectors describes multiple
        coordinates. Grid sizes must match the corresponding ``base ** level``.
    in_layout, out_layout : :class:`~tensorkrowch.formats.QuantizedLayout`, optional
        Prebuilt digit layouts, each supplied with its corresponding map.
        Input and output layouts must have the same number of sites.
    in_coordinate_map, out_coordinate_map : :class:`~tensorkrowch.formats.CoordinateMap`, optional
        Prebuilt maps, each with grid sizes matching its layout. Cannot be
        combined with shorthand arguments for that coordinate space.
    n_batches : int
        Number of leading structural batch axes shared by the cores.
        Independent of data batches during evaluation.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores, as in the corresponding plain format.

    Examples
    --------
    First, discretize input and output coordinates on their respective grids:

    >>> in_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([0., 1.]), grid_size=(4,))
    >>> out_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([-1., 1.]), grid_size=(4,))
    >>> in_coordinates = torch.tensor([[0.], [0.5]])
    >>> out_coordinates = torch.tensor([[-1.], [0.5]])
    >>> in_indices = in_map.to_indices(in_coordinates)
    >>> out_indices = out_map.to_indices(out_coordinates)
    >>> in_indices.tolist(), out_indices.tolist()
    ([[0], [2]], [[0], [3]])

    Then, expand both sets of indices into the digits paired at each site:

    >>> in_layout = tk.formats.QuantizedLayout(1, base=2, level=2)
    >>> out_layout = tk.formats.QuantizedLayout(1, base=2, level=2)
    >>> in_layout.encode_indices(in_indices).tolist()
    [[0, 0], [1, 0]]
    >>> out_layout.encode_indices(out_indices).tolist()
    [[0, 0], [1, 1]]

    Finally, construct an identity matrix on the grid indices and evaluate entries:

    >>> cores = [torch.eye(2).reshape(1, 2, 1, 2)] * 2
    >>> matrix = tk.formats.QTRM(cores, 1, 1,
    ...     in_layout=in_layout, out_layout=out_layout,
    ...     in_coordinate_map=in_map, out_coordinate_map=out_map)
    >>> matrix.evaluate_coordinates(in_coordinates, out_coordinates).tolist()
    [1.0, 0.0]

    With two coordinates per space, interleave their digits in both layouts:

    >>> coordinate_map = tk.formats.AffineCoordinateMap(
    ...     domain=torch.tensor([0., 1.]), grid_size=(4, 4))
    >>> layout = tk.formats.QuantizedLayout(2, base=2, level=2)
    >>> cores = [torch.eye(2).reshape(1, 2, 1, 2)] * 4
    >>> matrix = tk.formats.QTRM(cores, 2, 2,
    ...     in_layout=layout, out_layout=layout,
    ...     in_coordinate_map=coordinate_map, out_coordinate_map=coordinate_map)
    >>> in_coordinates = torch.tensor([[0., 0.], [0.5, 0.25]])
    >>> out_coordinates = torch.tensor([[0., 0.], [0.5, 0.75]])
    >>> matrix.evaluate_coordinates(in_coordinates, out_coordinates).tolist()
    [1.0, 0.0]
    """

    def to_trm(self) -> TRM:
        """
        Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        :class:`~tensorkrowch.formats.TRM`
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        return self._to_format(TRM)

    def to_qttm(self) -> QTTM:
        """
        Opens the ring while preserving its input and output layouts and maps.

        Carries the closing index through the chain, as in
        :meth:`~tensorkrowch.formats.TRM.to_ttm`, without truncation or dense
        reconstruction.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTTM`
            Open Quantics matrix representing the same entries, with structural
            batches preserved.
        """
        return super().to_ttm()

    def to_ttm(self) -> TTM:
        """
        Requires an explicit choice about retaining Quantics information.

        Use :meth:`to_qttm` to open the ring while retaining Quantics, or
        :meth:`to_trm` to remove Quantics while retaining the ring. To obtain
        a plain TTM, call ``to_qttm().to_ttm()`` or ``to_trm().to_ttm()``.

        Raises
        ------
        NotImplementedError
            Direct conversion is disabled to make the intended format explicit.
        """
        raise NotImplementedError(
            'Use `to_qttm()` to retain Quantics, or `to_qttm().to_ttm()` '
            'or `to_trm().to_ttm()` to obtain a plain TTM')

    def rotate(self, first: int = 0) -> 'QTRM':
        """
        Rotates ring sites and updates the Quantics digit schedules.

        Parameters
        ----------
        first : int
            Site that becomes index zero, in ``[0, n_sites - 1]``.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTRM`
            Rotated format preserving evaluations at coordinates in the
            original domain. Dense digit axes rotate with the core order.
        """
        if isinstance(first, bool) or not isinstance(first, int):
            raise TypeError('`first` should be int type')
        if not 0 <= first < self.n_sites:
            raise ValueError('`first` should lie inside the format')

        cores = [*self._cores[first:], *self._cores[:first]]

        def rotated_layout(layout: QuantizedLayout) -> QuantizedLayout:
            """Rotates a digit schedule consistently with the ring cores."""
            schedule = layout.sites()
            return QuantizedLayout(n_coordinates=layout.n_coordinates,
                                   base=layout.base,
                                   level=layout.level,
                                   ordering='custom',
                                   digit_order=layout.digit_order,
                                   permutation=(*schedule[first:],
                                                *schedule[:first]))

        result = QTRM(cores=cores,
                      in_n_coordinates=self.in_layout.n_coordinates,
                      out_n_coordinates=self.out_layout.n_coordinates,
                      in_layout=rotated_layout(self.in_layout),
                      out_layout=rotated_layout(self.out_layout),
                      in_coordinate_map=self.in_coordinate_map,
                      out_coordinate_map=self.out_coordinate_map,
                      n_batches=self._n_batches)
        if self._bonds is not None:
            factors = self._bonds.factors
            result.bonds = [*factors[first:], *factors[:first]]
        return result
