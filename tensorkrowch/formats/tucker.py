"""Two-level Quantics Tucker representations without algorithm provenance."""

from typing import ClassVar, Dict, List, Optional, Tuple, Type
import torch
from .base import TensorFormat
from ._chain import TensorFormat1D
from .tt import TT
from .tr import TR
from .quantization import CoordinateMap, QuantizedLayout, _unit_to_indices
from .quantics import _map_structure


class _QuantizedTuckerFormat(TensorFormat):
    """Quantized local factors connected to a small upper tensor network."""

    _upper_type: ClassVar[Type[TensorFormat1D]]
    _family = 'quantized_tucker'

    def __init__(self, upper, factors, layout, coordinate_map=None, domain=None, *,
                 variable_positions=None, computational_grid='endpoints', out_of_domain='error'):
        if not isinstance(upper, self._upper_type):
            raise TypeError(f'`upper` should be {self._upper_type.__name__} type')
        if not isinstance(layout, QuantizedLayout):
            raise TypeError('`layout` should be QuantizedLayout type')
        if coordinate_map is not None and not isinstance(coordinate_map, CoordinateMap):
            raise TypeError('`coordinate_map` should implement CoordinateMap')
        if computational_grid not in ('endpoints', 'cell_centers') or out_of_domain not in ('error', 'clip'):
            raise ValueError('Invalid computational grid or coordinate policy')
        self.upper = upper
        self.factors = tuple(factors)
        self.layout = layout
        self.coordinate_map = coordinate_map
        self.domain = domain
        if len(self.factors) != layout.n_variables or not all(isinstance(factor, TT) for factor in self.factors):
            raise ValueError('There should be one TT factor per variable')
        if variable_positions is None:
            if upper.n_sites != layout.n_variables:
                raise ValueError('variable_positions is required when upper output sites are present')
            variable_positions = range(layout.n_variables)
        self.variable_positions = tuple(variable_positions)
        positions = self.variable_positions
        if len(positions) != layout.n_variables or any(isinstance(site, bool) or not isinstance(site, int) or not 0 <= site < upper.n_sites for site in positions):
            raise ValueError('Variable positions should select valid upper sites')
        if any(a >= b for a, b in zip(positions, positions[1:])):
            raise ValueError('Variable positions should be strictly increasing')
        self.computational_grid = computational_grid
        self.out_of_domain = out_of_domain
        self.validate()

    @property
    def cores(self):
        """Upper cores are owned by upper only; no duplicate core container."""
        return self.upper.cores

    @cores.setter
    def cores(self, values):
        self.upper.cores = values

    @property
    def device(self):
        return self.upper.device

    @property
    def dtype(self):
        return self.upper.dtype

    @property
    def n_sites(self):
        return self.upper.n_sites

    @property
    def n_batches(self):
        return 0

    @property
    def batch_shape(self):
        return ()

    @property
    def rank(self):
        return self.upper.rank

    @property
    def topology(self):
        return self._topology

    @property
    def in_dim(self):
        return self._flattened_in_dim()

    @property
    def out_dim(self):
        return None

    def validate(self):
        self.upper.validate()
        for factor in self.factors:
            factor.validate()
        self._validate_cores()
        return self

    def _map_tensors(self, function):
        return type(self)(self.upper._map_tensors(function),
                          [factor._map_tensors(function) for factor in self.factors],
                          self.layout, _map_structure(self.coordinate_map, function),
                          _map_structure(self.domain, function),
                          variable_positions=self.variable_positions,
                          computational_grid=self.computational_grid,
                          out_of_domain=self.out_of_domain)

    def to(self, device=None, dtype=None, copy=False):
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('dtype should be torch.dtype type')
        if not isinstance(copy, bool):
            raise TypeError('copy should be bool type')
        from .quantics import _same_references
        upper = self.upper.to(device=device, dtype=dtype, copy=copy)
        factors = [factor.to(device=device, dtype=dtype, copy=copy) for factor in self.factors]
        function = lambda tensor: tensor.to(device=device, dtype=dtype, copy=copy)
        coordinate_map = _map_structure(self.coordinate_map, function)
        domain = _map_structure(self.domain, function)
        if not copy and upper is self.upper and all(
                new is old for new, old in zip(factors, self.factors)) and \
                _same_references(self.coordinate_map, coordinate_map) and _same_references(self.domain, domain):
            return self
        return type(self)(upper, factors, self.layout, coordinate_map, domain,
                          variable_positions=self.variable_positions,
                          computational_grid=self.computational_grid, out_of_domain=self.out_of_domain)

    def copy(self):
        return self._map_tensors(lambda tensor: tensor.clone())

    def detach(self):
        return self._map_tensors(lambda tensor: tensor.detach())

    def detach_(self):
        detached = self.detach()
        self.__dict__.update(detached.__dict__)
        return self

    @property
    def out_shape(self) -> Tuple[int, ...]:
        """Tensor-output dimensions retained as open upper-network sites."""
        variable_positions = set(self.variable_positions)
        return tuple(
            dimension
            for site, dimension in enumerate(self.upper.in_dim)
            if site not in variable_positions)


    @property
    def factor_rank(self) -> Tuple[Tuple[int, ...], ...]:
        """TT ranks internal to every local quantized factor."""
        return tuple(tuple(factor.rank) for factor in self.factors)


    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        upper = self.upper
        if upper.n_batches:
            raise ValueError(
                'Quantized Tucker upper decompositions cannot be batched')
        if any(factor.n_batches for factor in self.factors):
            raise ValueError('Quantized Tucker factors cannot be batched')
        for variable, (factor, position) in enumerate(zip(
                self.factors, self.variable_positions)):
            expected = (
                (self.layout.base[variable],) * self.layout.level[variable])
            if factor.in_dim[:-1] != expected:
                raise ValueError(
                    'Factor digit dimensions should match the quantized layout')
            if factor.in_dim[-1] != upper.in_dim[position]:
                raise ValueError(
                    'Factor connector dimension should match its upper site')
            if factor.device != upper.device or factor.dtype != upper.dtype:
                raise ValueError(
                    'Upper cores and factors should share device and dtype')
        return upper.rank, (), upper.in_dim, None


    def _flattened_in_dim(self) -> Tuple[int, ...]:
        """Expands each upper connector into its factor digit dimensions."""
        variable_by_position = {
            position: variable
            for variable, position in enumerate(self.variable_positions)}
        dimensions = []
        for site, dimension in enumerate(self.upper.in_dim):
            variable = variable_by_position.get(site)
            if variable is None:
                dimensions.append(dimension)
            else:
                dimensions.extend(self.factors[variable].in_dim[:-1])
        return tuple(dimensions)


    def _standard_cores(self) -> List[torch.Tensor]:
        return self.flatten()._standard_cores()


    def _physical_to_indices(self, points: torch.Tensor) -> torch.Tensor:
        """Maps physical points to grid indices without retaining the source."""
        if self.coordinate_map is None:
            raise ValueError("Physical evaluation requires a coordinate map")
        if not isinstance(points, torch.Tensor):
            raise TypeError('`points` should be torch.Tensor type')
        if points.ndim < 1 or points.shape[-1] != self.layout.n_variables:
            raise ValueError(
                'The last `points` dimension should match layout variables')
        points = points.to(device=self.device)
        direct = getattr(self.coordinate_map, 'to_indices', None)
        if callable(direct):
            return direct(
                points,
                self.layout.grid_size,
                self.domain,
                out_of_domain=self.out_of_domain)
        inverse = getattr(self.coordinate_map, 'inverse', None)
        if not callable(inverse):
            raise NotImplementedError(
                'Physical evaluation requires a coordinate-map inverse')
        unit = inverse(
            points,
            self.domain,
            out_of_domain=self.out_of_domain)
        return _unit_to_indices(
            unit,
            self.layout.grid_size,
            self.computational_grid,
            self.out_of_domain)


    def _factor_vectors(self, digits: torch.Tensor) -> List[torch.Tensor]:
        """Contracts every local factor while leaving gamma open."""
        schedule = self.layout.sites()
        vectors = []
        for variable, factor in enumerate(self.factors):
            columns = [
                column
                for column, site in enumerate(schedule)
                if site[0] == variable]
            variable_digits = digits.index_select(
                -1,
                torch.tensor(columns, device=digits.device))
            state = None
            factor_cores = factor._standard_cores()
            for site, core in enumerate(factor_cores[:-1]):
                local = core[:, variable_digits[..., site], :].movedim(0, -2)
                state = local if state is None else state @ local
            connector = factor_cores[-1].squeeze(-1)
            vectors.append((state @ connector).squeeze(-2))
        return vectors


    def evaluate_digits(self, digits: torch.Tensor) -> torch.Tensor:
        """Evaluates scheduled digit configurations through both levels."""
        digits = self.layout._integer_tensor(digits, 'digits').to(self.device)
        if digits.ndim != 2 or digits.shape[-1] != self.layout.n_sites:
            raise ValueError(
                '`digits` should have shape (batch_size, layout.n_sites)')
        self.layout.decode_digits(digits)
        factor_vectors = self._factor_vectors(digits)
        vectors_by_position = {
            position: factor_vectors[variable]
            for variable, position in enumerate(self.variable_positions)}
        return self._contract_upper(vectors_by_position, digits.shape[0])


    def evaluate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Evaluates one integer grid index per original variable."""
        indices = self.layout._integer_tensor(indices, 'indices')
        if indices.ndim != 2 or indices.shape[-1] != self.layout.n_variables:
            raise ValueError(
                '`indices` should have shape (batch_size, n_variables)')
        return self.evaluate_digits(self.layout.encode_indices(indices))


    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
        """Quantizes physical points and contracts factors with the upper TN."""
        return self.evaluate_indices(self._physical_to_indices(points))


    def _contract_upper(self,
                        vectors: Dict[int, torch.Tensor],
                        batch_size: int) -> torch.Tensor:
        raise NotImplementedError


    @staticmethod
    def _carry_upper_rank(core: torch.Tensor,
                          upper_rank: int) -> torch.Tensor:
        """Carries an upper rank unchanged through one factor digit core."""
        identity = torch.eye(
            upper_rank, device=core.device, dtype=core.dtype)
        combined = torch.einsum('ab,lpr->alpbr', identity, core)
        return combined.reshape(
            upper_rank * core.shape[0],
            core.shape[1],
            upper_rank * core.shape[2])


    def _flat_standard_cores(self) -> List[torch.Tensor]:
        """Substitutes every gamma site by its local factor TT block."""
        variable_by_position = {
            position: variable
            for variable, position in enumerate(self.variable_positions)}
        flat = []
        for site, upper_core in enumerate(self.upper._standard_cores()):
            variable = variable_by_position.get(site)
            if variable is None:
                flat.append(upper_core)
                continue

            factor_cores = self.factors[variable]._standard_cores()
            digit_cores = factor_cores[:-1]
            connector = factor_cores[-1].squeeze(-1)
            upper_left = upper_core.shape[0]
            for digit_core in digit_cores[:-1]:
                flat.append(self._carry_upper_rank(
                    digit_core, upper_left))
            last = torch.einsum(
                'lpr,rg,agb->alpb',
                digit_cores[-1],
                connector,
                upper_core)
            flat.append(last.reshape(
                upper_left * digit_cores[-1].shape[0],
                digit_cores[-1].shape[1],
                upper_core.shape[-1]))
        return flat


    def contract_dense(self) -> torch.Tensor:
        """Contracts the explicit flattened network for small-grid oracles."""
        return self.flatten().contract_dense()


    def norm(self) -> torch.Tensor:
        return self.flatten().norm()


    def normalized_overlap(
            self, other: TensorFormat1D) -> torch.Tensor:
        if not isinstance(other, _QuantizedTuckerFormat):
            raise TypeError(
                '`other` should be a quantized Tucker decomposition')
        return self.flatten().normalized_overlap(other.flatten())


    def fidelity(self, other: TensorFormat1D) -> torch.Tensor:
        return self.normalized_overlap(other).abs().square()


    def flatten(self):
        """Returns a Quantics TT/TR with grouped factor blocks and open outputs."""
        from .operations import _build_network
        from .quantics import QTT, QTR
        standard = self._flat_standard_cores()
        dimensions = self._flattened_in_dim()
        cyclic = self._upper_type is TR
        base = _build_network(standard, dimensions, None, 0, cyclic)
        schedule = []
        positions = []
        column = 0
        by_position = {site: variable for variable, site in enumerate(self.variable_positions)}
        for site in range(self.upper.n_sites):
            if site not in by_position:
                column += 1
                continue
            variable = by_position[site]
            for digit_site in self.layout.sites():
                if digit_site[0] != variable:
                    continue
                schedule.append(digit_site)
                positions.append(column)
                column += 1
        layout = QuantizedLayout(self.layout.n_variables, self.layout.base, self.layout.level,
                                 ordering='custom', permutation=schedule)
        cls = QTR if cyclic else QTT
        return cls(base.cores, layout, self.coordinate_map, self.domain,
                   digit_positions=positions, computational_grid=self.computational_grid,
                   out_of_domain=self.out_of_domain)

    def evaluate_points(self, points):
        return self.evaluate(points)


class QTTTucker(_QuantizedTuckerFormat):
    """Quantics factors connected to a TT upper network."""

    _upper_type = TT
    _topology = 'qtt_tucker'

    def _contract_upper(self,
                        vectors: Dict[int, torch.Tensor],
                        batch_size: int) -> torch.Tensor:
        state = self.cores[0].new_ones(batch_size, 1)
        for site, core in enumerate(self.upper._standard_cores()):
            if site in vectors:
                local = torch.einsum(
                    'bp,lpr->blr', vectors[site], core)
                state = torch.einsum('b...l,blr->b...r', state, local)
            else:
                state = torch.einsum(
                    'b...l,lpr->b...pr', state, core)
        return state.squeeze(-1)



class QTRTucker(_QuantizedTuckerFormat):
    """Quantics factors connected to a TR upper network."""

    _upper_type = TR
    _topology = 'qtr_tucker'

    def _contract_upper(self,
                        vectors: Dict[int, torch.Tensor],
                        batch_size: int) -> torch.Tensor:
        cyclic_rank = self.upper.cores[0].shape[0]
        state = torch.eye(
            cyclic_rank,
            device=self.device,
            dtype=self.dtype).expand(batch_size, -1, -1)
        for site, core in enumerate(self.upper._standard_cores()):
            if site in vectors:
                local = torch.einsum(
                    'bp,lpr->blr', vectors[site], core)
                state = torch.einsum(
                    'ba...l,blr->ba...r', state, local)
            else:
                state = torch.einsum(
                    'ba...l,lpr->ba...pr', state, core)
        return state.diagonal(dim1=1, dim2=-1).sum(-1)
