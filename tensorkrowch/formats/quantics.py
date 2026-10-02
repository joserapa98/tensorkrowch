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
        * _points_to_indices
"""

from dataclasses import fields, is_dataclass, replace
from math import prod
from typing import Any, Callable, Optional, Sequence, Type, Union

import torch

from tensorkrowch.formats.formats1d import (TensorFormat1D, TT, TR, TTM, TRM,
                                         _VectorFormat1D,
                                         _restore_cores)
from tensorkrowch.formats.quantization import (QuantizedLayout, CoordinateMap,
                                             Domain,
                                             _CompositeCoordinateMap,
                                             _unit_to_indices)


def _map_structure(value: Any, function: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Maps stored coordinate tensors without casting real grids to complex."""
    if isinstance(value, torch.Tensor):
        result = function(value)
        return result.real if not value.is_complex() and result.is_complex() else result
    if isinstance(value, _CompositeCoordinateMap):
        return _CompositeCoordinateMap(
            [_map_structure(item, function) for item in value.maps])
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
    if isinstance(first, _CompositeCoordinateMap):
        return _equal_structure(first.maps, second.maps)
    if is_dataclass(first):
        return all(_equal_structure(getattr(first, field.name), getattr(second, field.name))
                   for field in fields(first))
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(
            _equal_structure(left, right) for left, right in zip(first, second))
    if callable(first):
        return False
    return first == second


def _same_references(first: Any, second: Any) -> bool:
    """Compares storage references to decide whether a conversion was a no-op."""
    if isinstance(first, torch.Tensor):
        return first is second
    if isinstance(first, _CompositeCoordinateMap):
        return _same_references(first.maps, second.maps)
    if is_dataclass(first):
        return all(_same_references(getattr(first, field.name), getattr(second, field.name))
                   for field in fields(first))
    if isinstance(first, (list, tuple)):
        return all(_same_references(left, right)
                   for left, right in zip(first, second))
    return True


def _points_to_indices(points: torch.Tensor,
                       layout: QuantizedLayout,
                       coordinate_map: Optional[CoordinateMap],
                       domain: Domain,
                       grid: str,
                       policy: str) -> torch.Tensor:
    """Converts physical points using the actual stored coordinate map."""
    if coordinate_map is None:
        raise ValueError('Physical evaluation requires a coordinate map')
    if not isinstance(points, torch.Tensor):
        raise TypeError('`points` should be torch.Tensor type')
    if points.ndim < 1 or points.shape[-1] != layout.n_variables:
        raise ValueError('Point coordinates should match the layout variables')
    direct = getattr(coordinate_map, 'to_indices', None)
    if callable(direct):
        return direct(points, layout.grid_size, domain, out_of_domain=policy)
    inverse = getattr(coordinate_map, 'inverse', None)
    if not callable(inverse):
        raise NotImplementedError(
            'Physical evaluation requires a coordinate-map inverse')
    return _unit_to_indices(inverse(points, domain, out_of_domain=policy),
                            layout.grid_size, grid, policy)


class _QuanticsFormat:
    """Coordinate compatibility and direct construction of Quantics results."""

    _quantized = True
    _coordinate_names = ()

    def _map_tensors(self,
                     function: Callable[[torch.Tensor], torch.Tensor]) -> TensorFormat1D:
        """Maps cores, factors and tensors stored in coordinate metadata."""
        result = super()._map_tensors(function)
        for name in self._coordinate_names:
            setattr(result, name, _map_structure(getattr(self, name), function))
        return result


    def _same_aux_tensors(self, other: TensorFormat1D) -> bool:
        """Checks whether coordinate tensors retain their original references."""
        return all(_same_references(getattr(self, name), getattr(other, name))
                   for name in self._coordinate_names)


    def _as_format(self, cls: Type[TensorFormat1D]) -> TensorFormat1D:
        """Drops coordinate metadata while retaining cores and bond factors."""
        result = cls(self._cores, n_batches=self._n_batches)
        if self._bonds is not None:
            result._bonds = self._bonds._map_tensors(
                lambda tensor: tensor, result._on_bonds_changed)
        if self._out_dim is None:
            result._is_row = self._is_row
        return result


    def _check_semantics(self, other: TensorFormat1D, product: bool = False) -> None:
        """Checks layouts, coordinate maps and contracted coordinate spaces."""
        super()._check_semantics(other, product=product)
        if product:
            # Operator products are constructed by the matrix operand.
            left = (self.in_layout, self.in_coordinate_map, self.in_domain)
            right = ((other.out_layout, other.out_coordinate_map, other.out_domain)
                     if isinstance(other, _QuanticsMatrix) else
                     (other.layout, other.coordinate_map, other.domain))
            if not _equal_structure(left, right):
                raise ValueError('Contracted Quantics coordinate spaces should match')
        else:
            names = (('layout', 'coordinate_map', 'domain', 'digit_positions')
                     if isinstance(self, _QuanticsVector) else
                     ('in_layout', 'out_layout', 'in_coordinate_map',
                      'out_coordinate_map', 'in_domain', 'out_domain'))
            if any(not _equal_structure(getattr(self, name), getattr(other, name, None))
                   for name in names):
                raise ValueError('Quantics layouts and coordinate maps should match')
        if (self.computational_grid, self.out_of_domain) != (
            other.computational_grid, other.out_of_domain):
            raise ValueError('Quantics coordinate policies should match')


    def _new_from_standard_cores(self,
                                 cores: Sequence[torch.Tensor],
                                 in_dim: Sequence[int],
                                 out_dim: Optional[Sequence[int]],
                                 n_batches: int,
                                 cyclic: bool,
                                 other: Optional[TensorFormat1D] = None,
                                 product: bool = False) -> TensorFormat1D:
        """Constructs the matching Quantics result directly from its cores."""
        cores = _restore_cores(cores, in_dim, out_dim, n_batches, cyclic)
        options = dict(n_batches=n_batches,
                       computational_grid=self.computational_grid,
                       out_of_domain=self.out_of_domain)
        if out_dim is not None:
            cls = QTRM if cyclic else QTTM
            input_format = other if product else self
            return cls(
                cores, input_format.in_layout, self.out_layout,
                input_format.in_coordinate_map, self.out_coordinate_map,
                input_format.in_domain, self.out_domain, **options)
        cls = QTR if cyclic else QTT
        if product:
            layout, coordinate_map, domain = (
                self.out_layout, self.out_coordinate_map, self.out_domain)
            positions = None
        else:
            layout, coordinate_map, domain = self.layout, self.coordinate_map, self.domain
            positions = self.digit_positions
        return cls(cores, layout, coordinate_map, domain,
                   digit_positions=positions, **options)


class _QuanticsVector(_QuanticsFormat):
    """Coordinate semantics shared by open and cyclic Quantics vectors."""

    _coordinate_names = ('coordinate_map', 'domain')

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
            cores, in_dim, out_dim, n_batches, cyclic, other=other,
            product=product)
        if not product:
            result._is_row = self._is_row
        return result

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 layout: QuantizedLayout,
                 coordinate_map: Optional[Union[CoordinateMap, Sequence[CoordinateMap]]] = None,
                 domain: Domain = None,
                 *,
                 digit_positions: Optional[Sequence[int]] = None,
                 n_batches: int = 0,
                 computational_grid: str = 'endpoints',
                 out_of_domain: str = 'error',
                 bonds: Optional[Sequence[Optional[torch.Tensor]]] = None) -> None:
        if not isinstance(layout, QuantizedLayout):
            raise TypeError('`layout` should be QuantizedLayout type')
        if isinstance(coordinate_map, (list, tuple)):
            coordinate_map = _CompositeCoordinateMap(coordinate_map)
        if coordinate_map is not None and not isinstance(coordinate_map, CoordinateMap):
            raise TypeError('`coordinate_map` should implement CoordinateMap')
        if computational_grid not in (
            'endpoints', 'cell_centers') or out_of_domain not in ('error', 'clip'):
            raise ValueError('Invalid computational grid or out-of-domain policy')
        self.layout = layout
        self.coordinate_map = coordinate_map
        self.domain = domain
        self.computational_grid = computational_grid
        self.out_of_domain = out_of_domain
        super().__init__(cores, n_batches=n_batches, bonds=bonds)

        # Digit sites carry inputs; any remaining physical sites stay open.
        positions = (tuple(range(self.n_sites)) if digit_positions is None
                     else tuple(digit_positions))
        if len(positions) != layout.n_sites or any(
                isinstance(site, bool) or not isinstance(
                    site, int) or not 0 <= site < self.n_sites
                for site in positions) or len(set(positions)) != len(positions):
            raise ValueError(
                'Digit positions should select every scheduled digit exactly once')
        if tuple(self._in_dim[site] for site in positions) != layout.in_dim:
            raise ValueError('Digit core dimensions should match the quantized layout')
        self.digit_positions = positions


    def validate(self) -> TensorFormat1D:
        """Validates cores and their Quantics digit-layout dimensions.

        Returns
        -------
        TensorFormat1D
            The current format. Invalid controlled replacement restores prior
            metadata.
        """
        super().validate()
        if 'digit_positions' in self.__dict__ and (
                any(site >= len(self._cores) for site in self.digit_positions) or
                tuple(self._in_dim[site] for site in self.digit_positions) != self.layout.in_dim):
            raise ValueError('Digit core dimensions should match the quantized layout')
        return self


    def _new_outer_product(self,
                           cores: Sequence[torch.Tensor],
                           other: _VectorFormat1D,
                           n_batches: int,
                           cyclic: bool) -> Union['QTTM', 'QTRM']:
        """Builds an operator with the row's inputs and this column's outputs."""
        if any(format.digit_positions != tuple(range(format.n_sites))
               for format in (self, other)):
            raise ValueError('Quantics outer products require only digit sites')
        if (self.computational_grid, self.out_of_domain) != (
                other.computational_grid, other.out_of_domain):
            raise ValueError('Quantics coordinate policies should match')

        cores = _restore_cores(
            cores, other._in_dim, self._in_dim, n_batches, cyclic)
        cls = QTRM if cyclic else QTTM
        return cls(
            cores, other.layout, self.layout,
            other.coordinate_map, self.coordinate_map,
            other.domain, self.domain, n_batches=n_batches,
            computational_grid=self.computational_grid,
            out_of_domain=self.out_of_domain)


    def evaluate_digits(self, digits: torch.Tensor) -> torch.Tensor:
        """Evaluates scheduled digit configurations.

        Parameters
        ----------
        digits : torch.Tensor
            Integer digit configurations in layout schedule order, with shape
            ``(*data_batch, layout.n_sites)``. Every digit should lie within its
            site base.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *data_batch, *output_sites)``. Sites
            outside ``digit_positions`` remain open.
        """
        digits = self.layout._integer_tensor(digits, 'digits').to(self.device)
        self.layout.decode_digits(digits)
        if self.digit_positions == tuple(range(self.n_sites)):
            return self.evaluate(digits, n_batches=digits.ndim - 1)

        # Contract digit sites while retaining output and structural batch axes.
        data_batch_shape = digits.shape[:-1]
        data_batch_size = prod(data_batch_shape)
        core_batch_size = prod(self._batch_shape)
        digits = digits.reshape(data_batch_size, -1)
        cores = self._effective_cores()
        closing = cores[0].shape[-3]
        state = torch.eye(closing, device=self.device, dtype=self.dtype)
        state = state.expand(core_batch_size, data_batch_size, closing, closing)
        columns = {site: column for column, site in enumerate(self.digit_positions)}
        output_shape = []
        for site, core in enumerate(cores):
            core = core.reshape(core_batch_size, *core.shape[-3:])
            if site in columns:
                local = core[:, :, digits[:, columns[site]], :].permute(0, 2, 1, 3)
                state = torch.einsum('cds...l,cdlr->cds...r', state, local)
            else:
                state = torch.einsum('cds...l,clpr->cds...pr', state, core)
                output_shape.append(core.shape[-2])
        value = state.diagonal(dim1=2, dim2=-1).sum(-1)
        return value.reshape(*self._batch_shape, *data_batch_shape, *output_shape)


    def evaluate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Evaluates original variable index configurations.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape ``(*batch, n_variables)``; each value lies in
            ``[0, grid_size[variable] - 1]``.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *data_batch, *output_sites)``. Sites
            outside ``digit_positions`` remain open.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(1, 2, 2)
        >>> format = tk.formats.QTT([torch.eye(2), torch.eye(2)], layout)
        >>> format.evaluate_indices(torch.tensor([[0], [3]])).tolist()
        [1.0, 1.0]
        """
        return self.evaluate_digits(self.layout.encode_indices(indices))


    def evaluate_points(self, points: torch.Tensor) -> torch.Tensor:
        """Evaluates physical coordinate configurations.

        Physical coordinates require a coordinate map with an inverse or
        grid-index lookup. The ``computational_grid`` and ``out_of_domain`` policies
        determine quantization.

        Parameters
        ----------
        points : torch.Tensor
            Finite physical coordinates with shape ``(*data_batch, n_variables)``. A
            coordinate map is required. Coordinates are quantized to the
            computational grid; no interpolation of the represented function is
            performed.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *data_batch, *output_sites)``. Sites
            outside ``digit_positions`` remain open.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(1, 2, 2)
        >>> format = tk.formats.QTT([torch.eye(2), torch.eye(2)], layout,
        ...     tk.formats.UniformCoordinateMap(), domain=torch.tensor([0., 3.]))
        >>> format.evaluate_points(torch.tensor([[0.], [3.]])).tolist()
        [1.0, 1.0]
        """
        return self.evaluate_indices(_points_to_indices(
            points, self.layout, self.coordinate_map, self.domain,
            self.computational_grid, self.out_of_domain))


    def to_dense_grid(self) -> torch.Tensor:
        """Evaluates every original variable index on a small grid.

        Explicitly allocates and evaluates the full grid, independent of
        digit-site scheduling.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *grid_size, *output_sites)``, in
            original variable order.
        """
        axes = [torch.arange(size, device=self.device)
                for size in self.layout.grid_size]
        indices = torch.cartesian_prod(*axes).reshape(-1, self.layout.n_variables)
        values = self.evaluate_indices(indices)
        output_shape = tuple(self._in_dim[site] for site in range(self.n_sites)
                        if site not in self.digit_positions)
        return values.reshape(*self._batch_shape, *self.layout.grid_size, *output_shape)


class QTT(_QuanticsVector, TT):
    """A tensor train plus the physical meaning of its digit sites.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described in :class:`TT`. The
        container is copied and tensors retain storage and autograd.
    layout : QuantizedLayout
        Digit bases, levels and site schedule. Scheduled core input
        dimensions should match the layout.
    coordinate_map : CoordinateMap, optional
        Physical-coordinate map. Evaluation at physical points requires an
        inverse or direct grid-index lookup.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Physical intervals as ``(2,)`` for a shared interval or ``(n_variables, 2)``
        for separate intervals. Interval-based maps require a domain; maps with
        their own physical grid or geometry can use ``None``.
    digit_positions : sequence of int, optional
        Core position for each scheduled digit column, in layout order. ``None``
        uses every core. Other sites are open tensor outputs.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores, as in the corresponding plain format.
    computational_grid : {"endpoints", "cell_centers"}
        Uniform computational positions used when the coordinate map does
        not provide direct index lookup.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ``ValueError`` or are
        clipped to the domain boundary.
    """

    def as_tt(self) -> TT:
        """Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        TT
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        return self._as_format(TT)


class QTR(_QuanticsVector, TR):
    """A tensor ring plus the physical meaning of its digit sites.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described in :class:`TR`. The
        container is copied and tensors retain storage and autograd.
    layout : QuantizedLayout
        Digit bases, levels and site schedule. Scheduled core input
        dimensions should match the layout.
    coordinate_map : CoordinateMap, optional
        Physical-coordinate map. Evaluation at physical points requires an
        inverse or direct grid-index lookup.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Physical intervals as ``(2,)`` for a shared interval or ``(n_variables, 2)``
        for separate intervals. Interval-based maps require a domain; maps with
        their own physical grid or geometry can use ``None``.
    digit_positions : sequence of int, optional
        Core position for each scheduled digit column, in layout order. ``None``
        uses every core. Other sites are open tensor outputs.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores, as in the corresponding plain format.
    computational_grid : {"endpoints", "cell_centers"}
        Uniform computational positions used when the coordinate map does
        not provide direct index lookup.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ``ValueError`` or are
        clipped to the domain boundary.
    """

    def as_tr(self) -> TR:
        """Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        TR
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        return self._as_format(TR)


    def rotate(self, first: int = 0) -> 'QTR':
        """Rotates ring sites and updates the Quantics digit schedules.

        Parameters
        ----------
        first : int
            Site that becomes index zero, in ``[0, n_sites - 1]``.

        Returns
        -------
        QTR
            Rotated format preserving evaluations in original physical-variable
            coordinates. Dense digit axes rotate with the core order.
        """
        if isinstance(first, bool) or not isinstance(first, int):
            raise TypeError('The first rotation site should be an integer')
        if not 0 <= first < self.n_sites:
            raise ValueError('The first rotation site should lie inside the format')
        cores = [*self._cores[first:], *self._cores[:first]]
        positions = tuple((site - first) %
                          self.n_sites for site in self.digit_positions)
        result = QTR(
            cores, self.layout, self.coordinate_map, self.domain,
            digit_positions=positions, n_batches=self._n_batches,
            computational_grid=self.computational_grid,
            out_of_domain=self.out_of_domain)
        result._is_row = self._is_row
        if self._bonds is not None:
            factors = self._bonds.factors
            result.bonds = [*factors[first:], *factors[:first]]
        return result


class _QuanticsMatrix(_QuanticsFormat):
    """Paired input/output digit layouts of a tensorized operator."""

    _coordinate_names = ('in_coordinate_map', 'out_coordinate_map',
                         'in_domain', 'out_domain')

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 in_layout: QuantizedLayout,
                 out_layout: QuantizedLayout,
                 in_coordinate_map: Optional[CoordinateMap] = None,
                 out_coordinate_map: Optional[CoordinateMap] = None,
                 in_domain: Domain = None,
                 out_domain: Domain = None,
                 *,
                 n_batches: int = 0,
                 computational_grid: str = 'endpoints',
                 out_of_domain: str = 'error',
                 bonds: Optional[Sequence[Optional[torch.Tensor]]] = None) -> None:
        if not isinstance(in_layout, QuantizedLayout) or not isinstance(
            out_layout, QuantizedLayout):
            raise TypeError('Matrix layouts should be QuantizedLayout objects')
        if in_layout.n_sites != out_layout.n_sites:
            raise ValueError('Matrix digit schedules should have equal lengths')
        for coordinate_map in (in_coordinate_map, out_coordinate_map):
            if coordinate_map is not None and not isinstance(
                coordinate_map, CoordinateMap):
                raise TypeError('Matrix coordinate maps should implement CoordinateMap')
        if computational_grid not in (
            'endpoints', 'cell_centers') or out_of_domain not in ('error', 'clip'):
            raise ValueError('Invalid computational grid or out-of-domain policy')
        self.in_layout = in_layout
        self.out_layout = out_layout
        self.in_coordinate_map = in_coordinate_map
        self.out_coordinate_map = out_coordinate_map
        self.in_domain = in_domain
        self.out_domain = out_domain
        self.computational_grid = computational_grid
        self.out_of_domain = out_of_domain

        super().__init__(cores, n_batches=n_batches, bonds=bonds)


    def validate(self) -> TensorFormat1D:
        """Validates cores and their Quantics digit-layout dimensions.

        Returns
        -------
        TensorFormat1D
            The current format. Invalid controlled replacement restores prior
            metadata.
        """
        super().validate()
        if self._in_dim != self.in_layout.in_dim or self._out_dim != self.out_layout.in_dim:
            raise ValueError('Matrix core dimensions should match paired digit layouts')
        return self


    def transpose(self) -> Union['QTTM', 'QTRM']:
        """Transposes matrix cores and exchanges input/output coordinate spaces.

        Returns
        -------
        QTTM or QTRM
            Separate format sharing tensor storage. Sites retain their order.
        """
        result = super().transpose()
        result.in_layout, result.out_layout = self.out_layout, self.in_layout
        result.in_coordinate_map, result.out_coordinate_map = (
            self.out_coordinate_map, self.in_coordinate_map)
        result.in_domain, result.out_domain = self.out_domain, self.in_domain
        return result


    def evaluate_digits(self, in_digits: torch.Tensor, out_digits: torch.Tensor) -> torch.Tensor:
        """Evaluates matrix entries using paired digits.

        Parameters
        ----------
        in_digits : torch.Tensor
            Input digits with shape ``(*data_batch, in_layout.n_sites)``. Values
            follow the input digit schedule and should lie within each site base.
        out_digits : torch.Tensor
            Output digits with shape ``(*data_batch, out_layout.n_sites)``, within
            the output site bases. Data batches match the inputs.

        Returns
        -------
        torch.Tensor
            Matrix entries with shape ``(*core_batch, *data_batch)``.
        """
        self.in_layout.decode_digits(in_digits)
        self.out_layout.decode_digits(out_digits)
        return self.evaluate(in_digits, out_digits, n_batches=in_digits.ndim - 1)


    def evaluate_indices(self,
                         in_indices: torch.Tensor,
                         out_indices: torch.Tensor) -> torch.Tensor:
        """Evaluates matrix entries using paired indices.

        Parameters
        ----------
        in_indices : torch.Tensor
            Input indices with shape ``(*data_batch, in_layout.n_variables)``.
            Indices lie within the grid bounds of each original input variable.
        out_indices : torch.Tensor
            Output indices with shape ``(*data_batch, out_layout.n_variables)``,
            within the output grid bounds. Data batches match the inputs.

        Returns
        -------
        torch.Tensor
            Matrix entries with shape ``(*core_batch, *data_batch)``.
        """
        return self.evaluate_digits(self.in_layout.encode_indices(in_indices),
                                    self.out_layout.encode_indices(out_indices))


    def evaluate_points(self, in_points: torch.Tensor, out_points: torch.Tensor) -> torch.Tensor:
        """Evaluates matrix entries using paired points.

        Parameters
        ----------
        in_points : torch.Tensor
            Finite floating input points with shape
            ``(*data_batch, in_layout.n_variables)``. An input coordinate map
            with an inverse or grid lookup is required.
        out_points : torch.Tensor
            Finite floating output points with shape
            ``(*data_batch, out_layout.n_variables)``, with matching data batches.
            An output coordinate map with an inverse or grid lookup is required.

        Returns
        -------
        torch.Tensor
            Entries with shape ``(*core_batch, *data_batch)``. Physical evaluation
            quantizes both coordinate groups; it does not interpolate matrix
            entries.
        """
        in_indices = _points_to_indices(
            in_points, self.in_layout, self.in_coordinate_map,
            self.in_domain, self.computational_grid, self.out_of_domain)
        out_indices = _points_to_indices(
            out_points, self.out_layout, self.out_coordinate_map,
            self.out_domain, self.computational_grid, self.out_of_domain)
        return self.evaluate_indices(in_indices, out_indices)


    def to_dense_grid(self) -> torch.Tensor:
        """Evaluates every original variable index on a small grid.

        Explicitly allocates and evaluates the full grid, independent of
        digit-site scheduling.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *input_grid_size, *output_grid_size)``,
            placing all input variables before all output variables.
        """
        input_axes = [torch.arange(size, device=self.device)
                      for size in self.in_layout.grid_size]
        output_axes = [torch.arange(size, device=self.device)
                       for size in self.out_layout.grid_size]
        in_indices = torch.cartesian_prod(*input_axes).reshape(-1, self.in_layout.n_variables)
        out_indices = torch.cartesian_prod(*output_axes).reshape(-1, self.out_layout.n_variables)

        # Evaluate the Cartesian product of input and output configurations.
        values = self.evaluate_indices(
            in_indices.repeat_interleave(out_indices.shape[0], 0),
            out_indices.repeat(in_indices.shape[0], 1))
        return values.reshape(*self._batch_shape,
                              *self.in_layout.grid_size, *self.out_layout.grid_size)


class QTTM(_QuanticsMatrix, TTM):
    """Open-chain operator with separate input/output Quantics layouts.

    QTTM requires unbatched cores.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described in :class:`TTM`. The
        container is copied and tensors retain storage and autograd.
    in_layout : QuantizedLayout
        Digit layout of input/column indices.
    out_layout : QuantizedLayout
        Digit layout of output/row indices. Input and output schedules
        should have the same number of sites.
    in_coordinate_map : CoordinateMap, optional
        Physical-coordinate map for the operator inputs.
    out_coordinate_map : CoordinateMap, optional
        Physical-coordinate map for the operator outputs.
    in_domain : torch.Tensor or sequence of torch.Tensor, optional
        Physical intervals for input coordinates.
    out_domain : torch.Tensor or sequence of torch.Tensor, optional
        Physical intervals for output coordinates.
    n_batches : int
        Number of leading structural batch axes. Only zero is supported.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores, as in the corresponding plain format.
    computational_grid : {"endpoints", "cell_centers"}
        Computational grid convention used when a coordinate map lacks
        direct index lookup.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ``ValueError`` or are
        clipped to the domain boundary.
    """

    def as_ttm(self) -> TTM:
        """Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        TTM
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        return self._as_format(TTM)


class QTRM(_QuanticsMatrix, TRM):
    """Cyclic operator with separate input/output Quantics layouts.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described in :class:`TRM`. The
        container is copied and tensors retain storage and autograd.
    in_layout : QuantizedLayout
        Digit layout of input/column indices.
    out_layout : QuantizedLayout
        Digit layout of output/row indices. Input and output schedules
        should have the same number of sites.
    in_coordinate_map : CoordinateMap, optional
        Physical-coordinate map for the operator inputs.
    out_coordinate_map : CoordinateMap, optional
        Physical-coordinate map for the operator outputs.
    in_domain : torch.Tensor or sequence of torch.Tensor, optional
        Physical intervals for input coordinates.
    out_domain : torch.Tensor or sequence of torch.Tensor, optional
        Physical intervals for output coordinates.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores, as in the corresponding plain format.
    computational_grid : {"endpoints", "cell_centers"}
        Computational grid convention used when a coordinate map lacks
        direct index lookup.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ``ValueError`` or are
        clipped to the domain boundary.
    """

    def as_trm(self) -> TRM:
        """Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        TRM
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        return self._as_format(TRM)


    def rotate(self, first: int = 0) -> 'QTRM':
        """Rotates ring sites and updates the Quantics digit schedules.

        Parameters
        ----------
        first : int
            Site that becomes index zero, in ``[0, n_sites - 1]``.

        Returns
        -------
        QTRM
            Rotated format preserving evaluations in original physical-variable
            coordinates. Dense digit axes rotate with the core order.
        """
        if isinstance(first, bool) or not isinstance(first, int):
            raise TypeError('The first rotation site should be an integer')
        if not 0 <= first < self.n_sites:
            raise ValueError('The first rotation site should lie inside the format')
        cores = [*self._cores[first:], *self._cores[:first]]

        def rotated_layout(layout: QuantizedLayout) -> QuantizedLayout:
            """Rotates a digit schedule consistently with the ring cores."""
            schedule = layout.sites()
            return QuantizedLayout(layout.n_variables, layout.base, layout.level,
                                   ordering='custom', digit_order=layout.digit_order,
                                   permutation=(*schedule[first:], *schedule[:first]))

        result = QTRM(
            cores, rotated_layout(self.in_layout), rotated_layout(self.out_layout),
            self.in_coordinate_map, self.out_coordinate_map,
            self.in_domain, self.out_domain, n_batches=self._n_batches,
            computational_grid=self.computational_grid,
            out_of_domain=self.out_of_domain)
        if self._bonds is not None:
            factors = self._bonds.factors
            result.bonds = [*factors[first:], *factors[:first]]
        return result
