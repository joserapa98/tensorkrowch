"""Quantics formats add coordinate meaning to ordinary compact tensor chains."""

from dataclasses import fields, is_dataclass, replace
from math import prod

import torch

from .quantization import (QuantizedLayout, CoordinateMap, _CompositeCoordinateMap,
                           _unit_to_indices)
from .tt import TT
from .tr import TR
from .ttm import TTM
from .trm import TRM


def _map_structure(value, function):
    """Maps stored coordinate tensors without casting real grids to complex."""
    if isinstance(value, torch.Tensor):
        result = function(value)
        return result.real if not value.is_complex() and result.is_complex() else result
    if isinstance(value, _CompositeCoordinateMap):
        return _CompositeCoordinateMap([_map_structure(item, function) for item in value.maps])
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
        return len(first) == len(second) and all(_equal_structure(a, b) for a, b in zip(first, second))
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


def _check_semantics(first, second, product=False):
    """Checks physical compatibility before core algebra on Quantics networks."""
    quantics = (_QuanticsVector, _QuanticsMatrix)
    a, b = isinstance(first, quantics), isinstance(second, quantics)
    if not (a or b):
        return
    if not (a and b):
        raise ValueError('Quantics algebra requires compatible coordinate semantics; use as_tt/as_tr explicitly')
    if product:
        if not isinstance(first, _QuanticsMatrix) and not isinstance(second, _QuanticsMatrix):
            raise TypeError('At least one Quantics @ operand should be a matrix')
        if isinstance(first, _QuanticsMatrix):
            left = (first.in_layout, first.in_coordinate_map, first.in_domain)
            right = ((second.out_layout, second.out_coordinate_map, second.out_domain)
                     if isinstance(second, _QuanticsMatrix) else
                     (second.layout, second.coordinate_map, second.domain))
        else:
            left = (first.layout, first.coordinate_map, first.domain)
            right = (second.out_layout, second.out_coordinate_map, second.out_domain)
        if not _equal_structure(left, right):
            raise ValueError('Contracted Quantics coordinate spaces should match')
    else:
        names = ('layout', 'coordinate_map', 'domain', 'digit_positions') if isinstance(first, _QuanticsVector) else (
            'in_layout', 'out_layout', 'in_coordinate_map', 'out_coordinate_map', 'in_domain', 'out_domain')
        if any(not _equal_structure(getattr(first, name), getattr(second, name, None)) for name in names):
            raise ValueError('Quantics layouts and coordinate maps should match')
    if (first.computational_grid, first.out_of_domain) != (second.computational_grid, second.out_of_domain):
        raise ValueError('Quantics coordinate policies should match')


def _inherit_semantics(result, first, second=None, product=False):
    """Wraps an algebra result with the corresponding coordinate layouts."""
    if not isinstance(first, (_QuanticsVector, _QuanticsMatrix)):
        return result
    options = dict(n_batches=result.n_batches,
                   computational_grid=first.computational_grid,
                   out_of_domain=first.out_of_domain)
    cyclic = result._topology.startswith('tr')
    if result._out_dim is not None:
        cls = QTRM if cyclic else QTTM
        inputs = second if product else first
        wrapped = cls(result.cores, inputs.in_layout, first.out_layout,
                       inputs.in_coordinate_map, first.out_coordinate_map,
                       inputs.in_domain, first.out_domain, **options)
    else:
        cls = QTR if cyclic else QTT
        if product and isinstance(first, _QuanticsMatrix):
            layout, coordinate_map, domain = first.out_layout, first.out_coordinate_map, first.out_domain
            positions = None
        elif product:
            layout, coordinate_map, domain = second.in_layout, second.in_coordinate_map, second.in_domain
            positions = None
        else:
            layout, coordinate_map, domain = first.layout, first.coordinate_map, first.domain
            positions = first.digit_positions
        wrapped = cls(result.cores, layout, coordinate_map, domain,
                       digit_positions=positions, **options)
    wrapped._bonds = result.bonds
    return wrapped


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
        raise NotImplementedError('Physical evaluation requires a coordinate-map inverse')
    return _unit_to_indices(inverse(points, domain, out_of_domain=policy),
                            layout.grid_size, grid, policy)


class _QuanticsVector:
    """Coordinate semantics shared by open and cyclic Quantics vectors."""

    def __init__(self, cores, layout, coordinate_map=None, domain=None, *,
                 digit_positions=None, n_batches=0, computational_grid='endpoints',
                 out_of_domain='error'):
        if not isinstance(layout, QuantizedLayout):
            raise TypeError('`layout` should be QuantizedLayout type')
        if isinstance(coordinate_map, (list, tuple)):
            coordinate_map = _CompositeCoordinateMap(coordinate_map)
        if coordinate_map is not None and not isinstance(coordinate_map, CoordinateMap):
            raise TypeError('`coordinate_map` should implement CoordinateMap')
        if computational_grid not in ('endpoints', 'cell_centers') or out_of_domain not in ('error', 'clip'):
            raise ValueError('Invalid computational grid or out-of-domain policy')
        self.layout = layout
        self.coordinate_map = coordinate_map
        self.domain = domain
        self.computational_grid = computational_grid
        self.out_of_domain = out_of_domain
        super().__init__(cores, n_batches=n_batches)
        positions = tuple(range(self.n_sites)) if digit_positions is None else tuple(digit_positions)
        if len(positions) != layout.n_sites or any(
                isinstance(site, bool) or not isinstance(site, int) or not 0 <= site < self.n_sites
                for site in positions) or len(set(positions)) != len(positions):
            raise ValueError('Digit positions should select every scheduled digit exactly once')
        if tuple(self._in_dim[site] for site in positions) != layout.in_dim:
            raise ValueError('Digit core dimensions should match the quantized layout')
        self.digit_positions = positions

    def _map_tensors(self, function):
        result = super()._map_tensors(function)
        result.coordinate_map = _map_structure(self.coordinate_map, function)
        result.domain = _map_structure(self.domain, function)
        return result

    def _same_auxiliary_tensors(self, other):
        return _same_references(self.coordinate_map, other.coordinate_map) and \
            _same_references(self.domain, other.domain)

    def validate(self):
        super().validate()
        if 'digit_positions' in self.__dict__ and tuple(
                self._in_dim[site] for site in self.digit_positions) != self.layout.in_dim:
            self._dirty = True
            raise ValueError('Digit core dimensions should match the quantized layout')
        return self

    def evaluate_digits(self, digits):
        """Evaluates scheduled digits; nondigit physical output sites remain open."""
        self._ensure_valid()
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

    def evaluate_indices(self, indices):
        """Evaluates integer coordinates of the original raw grid."""
        return self.evaluate_digits(self.layout.encode_indices(indices))

    def evaluate_points(self, points):
        """Maps physical coordinates to grid indices and evaluates the format."""
        return self.evaluate_indices(_points_to_indices(
            points, self.layout, self.coordinate_map, self.domain,
            self.computational_grid, self.out_of_domain))

    def to_dense_grid(self):
        """Explicit small-grid oracle in original variable order, then output sites."""
        axes = [torch.arange(size, device=self.device) for size in self.layout.grid_size]
        indices = torch.cartesian_prod(*axes).reshape(-1, self.layout.n_variables)
        values = self.evaluate_indices(indices)
        outputs = tuple(self._in_dim[site] for site in range(self.n_sites)
                        if site not in self.digit_positions)
        return values.reshape(*self._batch_shape, *self.layout.grid_size, *outputs)

    def _as_vector(self, cls):
        result = cls(self.cores, n_batches=self._n_batches)
        result._bonds = self._bonds
        return result


class QTT(_QuanticsVector, TT):
    """A tensor train plus the physical meaning of its digit sites."""

    def as_tt(self):
        """Drops coordinate semantics deliberately, retaining the raw network."""
        return self._as_vector(TT)


class QTR(_QuanticsVector, TR):
    """A tensor ring plus the physical meaning of its digit sites."""

    def as_tr(self):
        """Drops coordinate semantics deliberately, retaining the raw network."""
        return self._as_vector(TR)

    def to_tt(self):
        base = self.as_tr().to_tt()
        return QTT(base.cores, self.layout, self.coordinate_map,
                                   self.domain, digit_positions=self.digit_positions,
                                   n_batches=self._n_batches,
                                   computational_grid=self.computational_grid,
                                   out_of_domain=self.out_of_domain)

    def rotate(self, first=0):
        base = self.as_tr().rotate(first)
        positions = tuple((site - first) % self.n_sites for site in self.digit_positions)
        result = QTR(base.cores, self.layout, self.coordinate_map,
                                    self.domain, digit_positions=positions,
                                    n_batches=self._n_batches,
                                    computational_grid=self.computational_grid,
                                    out_of_domain=self.out_of_domain)
        result._bonds = base.bonds
        return result


class _QuanticsMatrix:
    """Paired input/output digit layouts of a tensorized operator."""

    def __init__(self, cores, in_layout, out_layout, in_coordinate_map=None,
                 out_coordinate_map=None, in_domain=None, out_domain=None, *,
                 n_batches=0, computational_grid='endpoints', out_of_domain='error'):
        if not isinstance(in_layout, QuantizedLayout) or not isinstance(out_layout, QuantizedLayout):
            raise TypeError('Matrix layouts should be QuantizedLayout objects')
        if in_layout.n_sites != out_layout.n_sites:
            raise ValueError('Matrix digit schedules should have equal lengths')
        for coordinate_map in (in_coordinate_map, out_coordinate_map):
            if coordinate_map is not None and not isinstance(coordinate_map, CoordinateMap):
                raise TypeError('Matrix coordinate maps should implement CoordinateMap')
        if computational_grid not in ('endpoints', 'cell_centers') or out_of_domain not in ('error', 'clip'):
            raise ValueError('Invalid computational grid or out-of-domain policy')
        self.in_layout, self.out_layout = in_layout, out_layout
        self.in_coordinate_map, self.out_coordinate_map = in_coordinate_map, out_coordinate_map
        self.in_domain, self.out_domain = in_domain, out_domain
        self.computational_grid, self.out_of_domain = computational_grid, out_of_domain
        super().__init__(cores, n_batches=n_batches)
        if self._in_dim != in_layout.in_dim or self._out_dim != out_layout.in_dim:
            raise ValueError('Matrix core dimensions should match paired digit layouts')

    def _map_tensors(self, function):
        result = super()._map_tensors(function)
        for name in ('in_coordinate_map', 'out_coordinate_map', 'in_domain', 'out_domain'):
            setattr(result, name, _map_structure(getattr(self, name), function))
        return result

    def _same_auxiliary_tensors(self, other):
        return all(_same_references(getattr(self, name), getattr(other, name)) for name in (
            'in_coordinate_map', 'out_coordinate_map', 'in_domain', 'out_domain'))

    def validate(self):
        super().validate()
        if self._in_dim != self.in_layout.in_dim or self._out_dim != self.out_layout.in_dim:
            self._dirty = True
            raise ValueError('Matrix core dimensions should match paired digit layouts')
        return self

    def evaluate_digits(self, in_digits, out_digits):
        self.in_layout.decode_digits(in_digits)
        self.out_layout.decode_digits(out_digits)
        return self.evaluate(in_digits, out_digits, n_batches=in_digits.ndim - 1)

    def evaluate_indices(self, in_indices, out_indices):
        return self.evaluate_digits(self.in_layout.encode_indices(in_indices),
                                     self.out_layout.encode_indices(out_indices))

    def evaluate_points(self, in_points, out_points):
        inputs = _points_to_indices(in_points, self.in_layout, self.in_coordinate_map,
                                    self.in_domain, self.computational_grid, self.out_of_domain)
        outputs = _points_to_indices(out_points, self.out_layout, self.out_coordinate_map,
                                     self.out_domain, self.computational_grid, self.out_of_domain)
        return self.evaluate_indices(inputs, outputs)

    def to_dense_grid(self):
        """Explicit oracle with original input grid axes followed by output axes."""
        inputs = torch.cartesian_prod(*[torch.arange(size, device=self.device)
                                       for size in self.in_layout.grid_size]).reshape(-1, self.in_layout.n_variables)
        outputs = torch.cartesian_prod(*[torch.arange(size, device=self.device)
                                        for size in self.out_layout.grid_size]).reshape(-1, self.out_layout.n_variables)
        values = self.evaluate_indices(inputs.repeat_interleave(outputs.shape[0], 0),
                                        outputs.repeat(inputs.shape[0], 1))
        return values.reshape(*self._batch_shape, *self.in_layout.grid_size, *self.out_layout.grid_size)

    def transpose(self):
        base = super().transpose()
        cls = QTRM if self._topology.startswith('tr') else QTTM
        result = cls(base.cores, self.out_layout, self.in_layout,
                     self.out_coordinate_map, self.in_coordinate_map,
                     self.out_domain, self.in_domain, n_batches=self._n_batches,
                     computational_grid=self.computational_grid, out_of_domain=self.out_of_domain)
        result._bonds = base.bonds
        return result

    def apply(self, data, n_batches=1):
        from ._chain import TensorFormat1D
        result = super().apply(data, n_batches=n_batches)
        if isinstance(data, TensorFormat1D):
            return result
        return _inherit_semantics(result, self, product=True)


class QTTM(_QuanticsMatrix, TTM):
    """Open-chain operator with separate input/output Quantics layouts."""

    def as_ttm(self):
        result = TTM(self.cores, n_batches=self._n_batches)
        result._bonds = self._bonds
        return result


class QTRM(_QuanticsMatrix, TRM):
    """Cyclic operator with separate input/output Quantics layouts."""

    def as_trm(self):
        result = TRM(self.cores, n_batches=self._n_batches)
        result._bonds = self._bonds
        return result

    def to_ttm(self):
        base = self.as_trm().to_ttm()
        return QTTM(base.cores, self.in_layout, self.out_layout,
                                         self.in_coordinate_map, self.out_coordinate_map,
                                         self.in_domain, self.out_domain, n_batches=self._n_batches,
                                         computational_grid=self.computational_grid,
                                         out_of_domain=self.out_of_domain)

    def rotate(self, first=0):
        base = self.as_trm().rotate(first)
        def rotated_layout(layout):
            schedule = layout.sites()
            return QuantizedLayout(layout.n_variables, layout.base, layout.level,
                                   ordering='custom', digit_order=layout.digit_order,
                                   permutation=(*schedule[first:], *schedule[:first]))
        result = QTRM(base.cores, rotated_layout(self.in_layout),
                                          rotated_layout(self.out_layout),
                                          self.in_coordinate_map, self.out_coordinate_map,
                                          self.in_domain, self.out_domain,
                                          n_batches=self._n_batches,
                                          computational_grid=self.computational_grid,
                                          out_of_domain=self.out_of_domain)
        result._bonds = base.bonds
        return result
