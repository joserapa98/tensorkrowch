"""Quantics formats add coordinate meaning to ordinary compact tensor chains."""

from dataclasses import fields, is_dataclass, replace
from math import prod
from typing import Optional, Sequence

import torch

from tensorkrowch.formats.bonds import BondFactors1D
from tensorkrowch.formats.formats1d import TT, TR, TTM, TRM, _restore_cores
from tensorkrowch.formats.quantization import (QuantizedLayout, CoordinateMap,
                                             _CompositeCoordinateMap,
                                             _unit_to_indices)


def _map_structure(value, function):
    """Maps stored coordinate tensors without casting real grids to complex."""
    if isinstance(value, torch.Tensor):
        result = function(value)
        return result.real if not value.is_complex() and result.is_complex() else result
    if isinstance(value, _CompositeCoordinateMap):
        return _CompositeCoordinateMap(
            [_map_structure(item, function) for item in value.maps])
    if is_dataclass(value):
        return replace(value, **{field.name: _map_structure(getattr(value, field.name), function)
                                 for field in fields(value) if field.init})
    if isinstance(value, tuple):
        return tuple(_map_structure(item, function) for item in value)
    if isinstance(value, list):
        return [_map_structure(item, function) for item in value]
    return value


def _equal_structure(first, second):
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
        return len(first) == len(second) and all(_equal_structure(a, b)
                   for a, b in zip(first, second))
    if callable(first):
        return False
    return first == second


def _same_references(first, second):
    """Compares storage references to decide whether a conversion was a no-op."""
    if isinstance(first, torch.Tensor):
        return first is second
    if isinstance(first, _CompositeCoordinateMap):
        return _same_references(first.maps, second.maps)
    if is_dataclass(first):
        return all(_same_references(getattr(first, field.name), getattr(second, field.name))
                   for field in fields(first))
    if isinstance(first, (list, tuple)):
        return all(_same_references(a, b) for a, b in zip(first, second))
    return True


class _QuanticsFormat:
    """Coordinate compatibility and direct construction of Quantics results."""

    _quantized = True

    def _check_semantics(self, other, product=False):
        """Checks layouts, coordinate maps and contracted coordinate spaces."""
        super()._check_semantics(other, product=product)
        if product:
            if not isinstance(self, _QuanticsMatrix) and not isinstance(
                other, _QuanticsMatrix):
                raise TypeError('At least one Quantics @ operand should be a matrix')
            if isinstance(self, _QuanticsMatrix):
                left = (self.in_layout, self.in_coordinate_map, self.in_domain)
                right = ((other.out_layout, other.out_coordinate_map, other.out_domain)
                         if isinstance(other, _QuanticsMatrix) else
                         (other.layout, other.coordinate_map, other.domain))
            else:
                left = (self.layout, self.coordinate_map, self.domain)
                right = (other.out_layout, other.out_coordinate_map, other.out_domain)
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

    def _new_from_standard_cores(self, cores, in_dim, out_dim, n_batches,
                                 cyclic, other=None, product=False,
                                 transpose=False):
        """Constructs the matching Quantics result directly from its cores."""
        cores = _restore_cores(cores, in_dim, out_dim, n_batches, cyclic)
        options = dict(n_batches=n_batches,
                       computational_grid=self.computational_grid,
                       out_of_domain=self.out_of_domain)
        if out_dim is not None:
            cls = QTRM if cyclic else QTTM
            if transpose:
                return cls(
                    cores, self.out_layout, self.in_layout,
                    self.out_coordinate_map, self.in_coordinate_map,
                    self.out_domain, self.in_domain, **options)
            inputs = other if product else self
            return cls(
                cores, inputs.in_layout, self.out_layout,
                inputs.in_coordinate_map, self.out_coordinate_map,
                inputs.in_domain, self.out_domain, **options)
        cls = QTR if cyclic else QTT
        if product and isinstance(self, _QuanticsMatrix):
            layout, coordinate_map, domain = (
                self.out_layout, self.out_coordinate_map, self.out_domain)
            positions = None
        elif product:
            layout, coordinate_map, domain = (
                other.in_layout, other.in_coordinate_map, other.in_domain)
            positions = None
        else:
            layout, coordinate_map, domain = self.layout, self.coordinate_map, self.domain
            positions = self.digit_positions
        return cls(cores, layout, coordinate_map, domain,
                   digit_positions=positions, **options)


def _points_to_indices(points, layout, coordinate_map, domain, grid, policy):
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


class _QuanticsVector(_QuanticsFormat):
    """Coordinate semantics shared by open and cyclic Quantics vectors."""

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 layout,
                 coordinate_map=None,
                 domain=None,
                 *,
                 digit_positions: Optional[Sequence[int]] = None,
                 n_batches: int = 0,
                 computational_grid: str = 'endpoints',
                 out_of_domain: str = 'error') -> None:
        """Initializes the stored tensor references and validates construction."""
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
        super().__init__(cores, n_batches=n_batches)
        positions = tuple(range(self.n_sites)
                          ) if digit_positions is None else tuple(digit_positions)
        if len(positions) != layout.n_sites or any(
                isinstance(site, bool) or not isinstance(
                    site, int) or not 0 <= site < self.n_sites
                for site in positions) or len(set(positions)) != len(positions):
            raise ValueError(
                'Digit positions should select every scheduled digit exactly once')
        if tuple(self._in_dim[site] for site in positions) != layout.in_dim:
            raise ValueError('Digit core dimensions should match the quantized layout')
        self.digit_positions = positions

    def _map_tensors(self, function):
        """Maps stored tensors while preserving concrete container semantics."""
        result = super()._map_tensors(function)
        result.coordinate_map = _map_structure(self.coordinate_map, function)
        result.domain = _map_structure(self.domain, function)
        return result

    def _same_aux_tensors(self, other):
        """Checks whether auxiliary tensor references are unchanged."""
        return _same_references(self.coordinate_map, other.coordinate_map) and \
            _same_references(self.domain, other.domain)

    def validate(self):
        r"""Validates cores and their Quantics digit-layout dimensions.

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

    def evaluate_digits(self, digits) -> torch.Tensor:
        r"""Evaluates scheduled digit configurations.

        Parameters
        ----------
        digits : torch.Tensor
            Integer digit configurations in layout schedule order, with shape
            (*data_batch, layout.n_sites). Every digit should lie within its
            site base.

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch, *output_sites). Sites
            outside digit_positions remain open.
        """
        digits = self.layout._integer_tensor(digits, 'digits').to(self.device)
        self.layout.decode_digits(digits)
        if self.digit_positions == tuple(range(self.n_sites)):
            return self.evaluate(digits, n_batches=digits.ndim - 1)
        data_batch = digits.shape[:-1]
        count = prod(data_batch)
        core_count = prod(self._batch_shape)
        digits = digits.reshape(count, -1)
        cores = self._standard_cores()
        closing = cores[0].shape[-3]
        state = torch.eye(closing, device=self.device, dtype=self.dtype)
        state = state.expand(core_count, count, closing, closing)
        columns = {site: column for column, site in enumerate(self.digit_positions)}
        outputs = []
        for site, core in enumerate(cores):
            core = core.reshape(core_count, *core.shape[-3:])
            if site in columns:
                local = core[:, :, digits[:, columns[site]], :].permute(0, 2, 1, 3)
                state = torch.einsum('cds...l,cdlr->cds...r', state, local)
            else:
                state = torch.einsum('cds...l,clpr->cds...pr', state, core)
                outputs.append(core.shape[-2])
        value = state.diagonal(dim1=2, dim2=-1).sum(-1)
        return value.reshape(*self._batch_shape, *data_batch, *outputs)

    def evaluate_indices(self, indices) -> torch.Tensor:
        r"""Evaluates original variable index configurations.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape (*batch, n_variables), in [0,
            grid_size[variable] - 1].

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch, *output_sites). Sites
            outside digit_positions remain open.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(1, 2, 2)
        >>> format = tk.formats.QTT([torch.eye(2), torch.eye(2)], layout)
        >>> format.evaluate_indices(torch.tensor([[0], [3]])).tolist()
        [1.0, 1.0]
        """
        return self.evaluate_digits(self.layout.encode_indices(indices))

    def evaluate_points(self, points) -> torch.Tensor:
        r"""Evaluates physical coordinate configurations.

        Physical coordinates require a coordinate map with an inverse or
        grid-index lookup. The computational_grid and out_of_domain policies
        determine quantization.

        Parameters
        ----------
        points : torch.Tensor
            Finite physical coordinates with shape (*data_batch, n_variables). A
            coordinate map is required. Coordinates are quantized to the
            computational grid; no interpolation of the represented function is
            performed.

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch, *output_sites). Sites
            outside digit_positions remain open.

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
        r"""Evaluates every original variable index on a small grid.

        Explicitly allocates and evaluates the full grid, independent of
        digit-site scheduling.

        Returns
        -------
        torch.Tensor
            Dense values in original variable order, after structural batch
            axes. Vector output sites follow variable axes; matrix axes are all
            input variables followed by all output variables.
        """
        axes = [torch.arange(size, device=self.device)
                for size in self.layout.grid_size]
        indices = torch.cartesian_prod(*axes).reshape(-1, self.layout.n_variables)
        values = self.evaluate_indices(indices)
        outputs = tuple(self._in_dim[site] for site in range(self.n_sites)
                        if site not in self.digit_positions)
        return values.reshape(*self._batch_shape, *self.layout.grid_size, *outputs)

    def _as_vector(self, cls):
        """Drops coordinate metadata while retaining raw cores and factors."""
        result = cls(self.cores, n_batches=self._n_batches)
        result.bonds = self._bonds
        return result


class QTT(_QuanticsVector, TT):
    r"""A tensor train plus the physical meaning of its digit sites.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores in the endpoint layout of the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
    layout : QuantizedLayout
        Digit bases, levels and site schedule. Scheduled core input
        dimensions should match the layout.
    coordinate_map : CoordinateMap, optional
        Physical-coordinate map. Evaluation at physical points requires an
        inverse or direct grid-index lookup.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Physical intervals as (2,) for a shared interval or (n_variables, 2)
        for separate intervals. None uses [0, 1] for each variable.
    digit_positions : sequence of int, optional
        Core position for each scheduled digit column, in layout order. None
        uses every core. Other sites are open tensor outputs.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    computational_grid : {"endpoints", "cell_centers"}
        Uniform computational positions used when the coordinate map does
        not provide direct index lookup.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ValueError or are
        clipped to the domain boundary.
    """

    def as_tt(self):
        r"""Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        TT
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        return self._as_vector(TT)


class QTR(_QuanticsVector, TR):
    r"""A tensor ring plus the physical meaning of its digit sites.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores in the endpoint layout of the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
    layout : QuantizedLayout
        Digit bases, levels and site schedule. Scheduled core input
        dimensions should match the layout.
    coordinate_map : CoordinateMap, optional
        Physical-coordinate map. Evaluation at physical points requires an
        inverse or direct grid-index lookup.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Physical intervals as (2,) for a shared interval or (n_variables, 2)
        for separate intervals. None uses [0, 1] for each variable.
    digit_positions : sequence of int, optional
        Core position for each scheduled digit column, in layout order. None
        uses every core. Other sites are open tensor outputs.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    computational_grid : {"endpoints", "cell_centers"}
        Uniform computational positions used when the coordinate map does
        not provide direct index lookup.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ValueError or are
        clipped to the domain boundary.
    """

    def as_tr(self):
        r"""Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        TR
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        return self._as_vector(TR)


    def rotate(self, first=0):
        r"""Rotates ring sites and updates the Quantics digit schedules.

        Parameters
        ----------
        first : int
            Site that becomes index zero, in [0, n_sites - 1].

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
        result = QTR(cores, self.layout, self.coordinate_map,
                                    self.domain, digit_positions=positions,
                                    n_batches=self._n_batches,
                                    computational_grid=self.computational_grid,
                                    out_of_domain=self.out_of_domain)
        if self._bonds is not None:
            values = self._bonds.values
            result.bonds = BondFactors1D([*values[first:], *values[:first]])
        return result


class _QuanticsMatrix(_QuanticsFormat):
    """Paired input/output digit layouts of a tensorized operator."""

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 in_layout,
                 out_layout,
                 in_coordinate_map=None,
                 out_coordinate_map=None,
                 in_domain=None,
                 out_domain=None,
                 *,
                 n_batches: int = 0,
                 computational_grid: str = 'endpoints',
                 out_of_domain: str = 'error') -> None:
        """Initializes the stored tensor references and validates construction."""
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
        self.in_layout, self.out_layout = in_layout, out_layout
        self.in_coordinate_map, self.out_coordinate_map = in_coordinate_map, out_coordinate_map
        self.in_domain, self.out_domain = in_domain, out_domain
        self.computational_grid, self.out_of_domain = computational_grid, out_of_domain
        super().__init__(cores, n_batches=n_batches)
        if self._in_dim != in_layout.in_dim or self._out_dim != out_layout.in_dim:
            raise ValueError('Matrix core dimensions should match paired digit layouts')

    def _map_tensors(self, function):
        """Maps stored tensors while preserving concrete container semantics."""
        result = super()._map_tensors(function)
        for name in ('in_coordinate_map', 'out_coordinate_map',
                     'in_domain', 'out_domain'):
            setattr(result, name, _map_structure(getattr(self, name), function))
        return result

    def _same_aux_tensors(self, other):
        """Checks whether auxiliary tensor references are unchanged."""
        return all(_same_references(getattr(self, name), getattr(other, name)) for name in (
            'in_coordinate_map', 'out_coordinate_map', 'in_domain', 'out_domain'))

    def validate(self):
        r"""Validates cores and their Quantics digit-layout dimensions.

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

    def evaluate_digits(self, in_digits, out_digits) -> torch.Tensor:
        r"""Evaluates matrix entries using paired digits.

        Parameters
        ----------
        in_digits : torch.Tensor
            Input digits with shape (*data_batch, in_layout.n_sites). Values
            follow the input digit schedule or coordinate metadata.
        out_digits : torch.Tensor
            Output digits with matching data batches, following out_layout and
            its coordinate metadata.

        Returns
        -------
        torch.Tensor
            Entries with shape (*core_batch, *data_batch). Physical evaluation
            quantizes both coordinate groups; it does not interpolate matrix
            entries.
        """
        self.in_layout.decode_digits(in_digits)
        self.out_layout.decode_digits(out_digits)
        return self.evaluate(in_digits, out_digits, n_batches=in_digits.ndim - 1)

    def evaluate_indices(self, in_indices, out_indices) -> torch.Tensor:
        r"""Evaluates matrix entries using paired indices.

        Parameters
        ----------
        in_indices : torch.Tensor
            Input indices with shape (*data_batch, in_layout.n_variables).
            Values follow the input digit schedule or coordinate metadata.
        out_indices : torch.Tensor
            Output indices with matching data batches, following out_layout and
            its coordinate metadata.

        Returns
        -------
        torch.Tensor
            Entries with shape (*core_batch, *data_batch). Physical evaluation
            quantizes both coordinate groups; it does not interpolate matrix
            entries.
        """
        return self.evaluate_digits(self.in_layout.encode_indices(in_indices),
                                    self.out_layout.encode_indices(out_indices))

    def evaluate_points(self, in_points, out_points) -> torch.Tensor:
        r"""Evaluates matrix entries using paired points.

        Parameters
        ----------
        in_points : torch.Tensor
            Input points with shape (*data_batch, in_layout.n_variables). Values
            follow the input digit schedule or coordinate metadata.
        out_points : torch.Tensor
            Output points with matching data batches, following out_layout and
            its coordinate metadata.

        Returns
        -------
        torch.Tensor
            Entries with shape (*core_batch, *data_batch). Physical evaluation
            quantizes both coordinate groups; it does not interpolate matrix
            entries.
        """
        inputs = _points_to_indices(in_points, self.in_layout, self.in_coordinate_map,
                                    self.in_domain, self.computational_grid, self.out_of_domain)
        outputs = _points_to_indices(out_points, self.out_layout, self.out_coordinate_map,
                                     self.out_domain, self.computational_grid, self.out_of_domain)
        return self.evaluate_indices(inputs, outputs)

    def to_dense_grid(self) -> torch.Tensor:
        r"""Evaluates every original variable index on a small grid.

        Explicitly allocates and evaluates the full grid, independent of
        digit-site scheduling.

        Returns
        -------
        torch.Tensor
            Dense values in original variable order, after structural batch
            axes. Vector output sites follow variable axes; matrix axes are all
            input variables followed by all output variables.
        """
        inputs = torch.cartesian_prod(*[torch.arange(size, device=self.device)
                                        for size in self.in_layout.grid_size]).reshape(-1, self.in_layout.n_variables)
        outputs = torch.cartesian_prod(*[torch.arange(size, device=self.device)
                                         for size in self.out_layout.grid_size]).reshape(-1, self.out_layout.n_variables)
        values = self.evaluate_indices(inputs.repeat_interleave(outputs.shape[0], 0),
                                       outputs.repeat(inputs.shape[0], 1))
        return values.reshape(*self._batch_shape, * \
                              self.in_layout.grid_size, *self.out_layout.grid_size)


class QTTM(_QuanticsMatrix, TTM):
    r"""Open-chain operator with separate input/output Quantics layouts.

    QTTM requires unbatched cores.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores in the endpoint layout of the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
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
    computational_grid : {"endpoints", "cell_centers"}
        Computational grid convention used when a coordinate map lacks
        direct index lookup.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ValueError or are
        clipped to the domain boundary.
    """

    def as_ttm(self):
        r"""Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        TTM
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        result = TTM(self.cores, n_batches=self._n_batches)
        result.bonds = self._bonds
        return result


class QTRM(_QuanticsMatrix, TRM):
    r"""Cyclic operator with separate input/output Quantics layouts.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores in the endpoint layout of the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
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
    computational_grid : {"endpoints", "cell_centers"}
        Computational grid convention used when a coordinate map lacks
        direct index lookup.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ValueError or are
        clipped to the domain boundary.
    """

    def as_trm(self):
        r"""Drops Quantics metadata while retaining the represented tensor.

        Returns
        -------
        TRM
            Plain raw-tensor format sharing tensor storage. Coordinate maps,
            domains and digit-layout semantics are not retained.
        """
        result = TRM(self.cores, n_batches=self._n_batches)
        result.bonds = self._bonds
        return result


    def rotate(self, first=0):
        r"""Rotates ring sites and updates the Quantics digit schedules.

        Parameters
        ----------
        first : int
            Site that becomes index zero, in [0, n_sites - 1].

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

        def rotated_layout(layout):
            """Rotates a digit schedule consistently with the ring cores."""
            schedule = layout.sites()
            return QuantizedLayout(layout.n_variables, layout.base, layout.level,
                                   ordering='custom', digit_order=layout.digit_order,
                                   permutation=(*schedule[first:], *schedule[:first]))
        result = QTRM(cores, rotated_layout(self.in_layout),
                                          rotated_layout(self.out_layout),
                                          self.in_coordinate_map, self.out_coordinate_map,
                                          self.in_domain, self.out_domain,
                                          n_batches=self._n_batches,
                                          computational_grid=self.computational_grid,
                                          out_of_domain=self.out_of_domain)
        if self._bonds is not None:
            values = self._bonds.values
            result.bonds = BondFactors1D([*values[first:], *values[:first]])
        return result
