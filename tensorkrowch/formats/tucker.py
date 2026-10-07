"""
This script contains:

    Internal classes:
        * _QuantizedTuckerFormat

    Public classes:
        * QTTTucker
        * QTRTucker
"""

from typing import (Callable, ClassVar, Dict, List, Optional, Sequence, Tuple,
                    Type, Union)

import torch

from tensorkrowch.formats.base import TensorFormat
from tensorkrowch.formats.formats1d import (TensorFormat1D, TT, TR,
                                         _restore_cores)
from tensorkrowch.formats.quantics import (QTT, QTR, _map_structure,
                                         _coordinates_to_indices, _same_references)
from tensorkrowch.formats.quantization import (AffineCoordinateMap,
                                             CoordinateMap, QuantizedLayout,
                                             Domain, _grid_offset,
                                             _validate_explicit_grid)


class _QuantizedTuckerFormat(TensorFormat):
    """Local digit factors connected to a small upper TT/TR."""

    _upper_type: ClassVar[Type[TensorFormat1D]]
    _family = 'quantized_tucker'

    def __init__(self,
                 upper: Union[TT, TR],
                 factors: Sequence[TT],
                 layout: QuantizedLayout,
                 coordinate_map: Optional[CoordinateMap] = None,
                 domain: Domain = None,
                 *,
                 coordinate_positions: Optional[Sequence[int]] = None,
                 computational_grid: Union[str, float] = 'endpoints',
                 out_of_domain: str = 'error') -> None:
        if not isinstance(upper, self._upper_type):
            raise TypeError(f'`upper` should be {self._upper_type.__name__} type')
        if not isinstance(layout, QuantizedLayout):
            raise TypeError('`layout` should be QuantizedLayout type')
        if coordinate_map is not None and not isinstance(coordinate_map, CoordinateMap):
            raise TypeError('`coordinate_map` should implement CoordinateMap')
        _validate_explicit_grid(layout, coordinate_map)
        _grid_offset(computational_grid)
        if out_of_domain not in ('error', 'clip'):
            raise ValueError('Invalid `out_of_domain`')

        self.upper = upper
        self.factors = tuple(factors)
        self.layout = layout
        self.coordinate_map = coordinate_map
        self.domain = domain

        if len(self.factors) != layout.n_coordinates or not all(
            isinstance(factor, TT) for factor in self.factors):
            raise ValueError('`factors` should contain one TT per coordinate')
        if coordinate_positions is None:
            if upper.n_sites != layout.n_coordinates:
                raise ValueError(
                    '`coordinate_positions` is required when `upper` has output '
                    'sites')
            coordinate_positions = range(layout.n_coordinates)
        self.coordinate_positions = tuple(coordinate_positions)
        positions = self.coordinate_positions
        if len(positions) != layout.n_coordinates or any(
                isinstance(site, bool) or not isinstance(site, int) or
                not 0 <= site < upper.n_sites for site in positions):
            raise ValueError(
                '`coordinate_positions` should select valid `upper` sites')
        if any(left >= right for left, right in zip(positions, positions[1:])):
            raise ValueError(
                '`coordinate_positions` should be strictly increasing')

        self.computational_grid = computational_grid
        self.out_of_domain = out_of_domain
        self.validate()

    @property
    def cores(self) -> List[torch.Tensor]:
        """
        Mutable cores of the upper TT/TR, shared with :attr:`upper`.

        Local digit cores remain in :attr:`factors`. Use :meth:`flatten` to
        obtain a chain whose cores represent the digit sites directly.
        """
        return self.upper.cores

    @cores.setter
    def cores(self, values: Sequence[torch.Tensor]) -> None:
        """Replaces upper cores through the upper format's validated setter."""
        self.upper.cores = values

    @property
    def device(self) -> torch.device:
        """Device shared by the stored structural tensors."""
        return self.upper.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype shared by the stored structural tensors."""
        return self.upper.dtype

    @property
    def n_sites(self) -> int:
        """Number of sites in the upper format."""
        return self.upper.n_sites

    @property
    def n_batches(self) -> int:
        """Number of structural batch axes; always zero for Tucker formats."""
        return 0

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Structural batch shape; empty for hierarchical Tucker formats."""
        return ()

    @property
    def rank(self) -> List[int]:
        """Defensive list of upper-format virtual ranks."""
        return self.upper.rank

    @property
    def factor_rank(self) -> Tuple[Tuple[int, ...], ...]:
        """TT ranks internal to every local quantized factor."""
        return tuple(tuple(factor.rank) for factor in self.factors)

    @property
    def topology(self) -> str:
        """Topology identifier of the concrete format."""
        return self._topology

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimensions of the flattened digit and output sites."""
        return self._flattened_in_dim()

    @property
    def out_dim(self) -> Optional[Tuple[int, ...]]:
        """Dedicated matrix output dimensions; ``None`` for vector Tucker formats."""
        return None

    @property
    def out_shape(self) -> Tuple[int, ...]:
        """Output dimensions of the upper sites that have no factor attached."""
        coordinate_positions = set(self.coordinate_positions)
        return tuple(
            dimension
            for site, dimension in enumerate(self.upper.in_dim)
            if site not in coordinate_positions)

    def validate(self) -> '_QuantizedTuckerFormat':
        """
        Validates upper cores, local factors and connector dimensions.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTTTucker` or :class:`~tensorkrowch.formats.QTRTucker`
            The current format. Factors and upper cores should share device and
            dtype, and structural batches are unsupported.
        """
        self.upper.validate()
        for factor in self.factors:
            factor.validate()
        self._validate_cores()
        return self

    def _validate_cores(self) -> None:
        """Checks dimensions and runtime shared by the upper format and factors."""
        upper = self.upper
        if upper.n_batches:
            raise ValueError(
                '`upper` cannot have structural batches')
        if any(factor.n_batches for factor in self.factors):
            raise ValueError('`factors` cannot have structural batches')
        for coordinate, (factor, position) in enumerate(zip(
                self.factors, self.coordinate_positions)):
            expected = (
                (self.layout.base[coordinate],) * self.layout.level[coordinate])
            if factor.in_dim[:-1] != expected:
                raise ValueError(
                    'Factor digit dimensions should match `layout`')
            if factor.in_dim[-1] != upper.in_dim[position]:
                raise ValueError(
                    'Factor connector dimension should match its site in '
                    '`upper`')
            if factor.device != upper.device or factor.dtype != upper.dtype:
                raise ValueError(
                    '`upper` and `factors` should share device and dtype')

    def _map_tensors(
            self, function: Callable[[torch.Tensor], torch.Tensor]
        ) -> '_QuantizedTuckerFormat':
        """Maps stored tensors while preserving concrete container semantics."""
        return type(self)(self.upper._map_tensors(function),
                          [factor._map_tensors(function) for factor in self.factors],
                          self.layout, _map_structure(self.coordinate_map, function),
                          _map_structure(self.domain, function),
                          coordinate_positions=self.coordinate_positions,
                          computational_grid=self.computational_grid,
                          out_of_domain=self.out_of_domain)

    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           copy: bool = False) -> '_QuantizedTuckerFormat':
        """
        Returns a device/dtype conversion, preserving the concrete format.

        PyTorch device errors propagate without a CPU fallback. Autograd is
        retained.

        Parameters
        ----------
        device : str or torch.device, optional
            Target device. ``None`` preserves the current device.
        dtype : torch.dtype, optional
            Target dtype. ``None`` preserves the current dtype. Coordinate
            grids and Schmidt spectra remain real when cores are complex.
        copy : bool
            If ``True``, copies tensors even when device and dtype are
            unchanged. If ``False``, an unchanged conversion may return
            ``self``.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTTTucker` or :class:`~tensorkrowch.formats.QTRTucker`
            Converted format; ``self`` when no conversion is needed and
            ``copy`` is ``False``.

        Examples
        --------
        >>> layout = tk.formats.QuantizedLayout(1, base=2, level=1)
        >>> upper = tk.formats.TT([torch.ones(2)])
        >>> factor = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> format = tk.formats.QTTTucker(upper, [factor], layout)
        >>> format.to() is format
        True
        >>> format.to(dtype=torch.float64).dtype == torch.float64
        True
        """
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type')
        if not isinstance(copy, bool):
            raise TypeError('`copy` should be bool type')
        upper = self.upper.to(device=device, dtype=dtype, copy=copy)
        factors = [factor.to(device=device, dtype=dtype, copy=copy)
                   for factor in self.factors]

        def convert_tensor(tensor: torch.Tensor) -> torch.Tensor:
            """Converts one stored tensor to the requested device and dtype."""
            return tensor.to(device=device, dtype=dtype, copy=copy)

        coordinate_map = _map_structure(self.coordinate_map, convert_tensor)
        domain = _map_structure(self.domain, convert_tensor)
        if not copy and upper is self.upper and all(
                new is old for new, old in zip(factors, self.factors)) and \
                _same_references(self.coordinate_map, coordinate_map) and \
                _same_references(self.domain, domain):
            return self
        return type(self)(upper, factors, self.layout, coordinate_map, domain,
                          coordinate_positions=self.coordinate_positions,
                          computational_grid=self.computational_grid,
                          out_of_domain=self.out_of_domain)

    def clone(self) -> '_QuantizedTuckerFormat':
        """
        Clones the structural tensors, preserving autograd.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTTTucker` or :class:`~tensorkrowch.formats.QTRTucker`
            Independent tensor storage with the same represented tensor.
        """
        return self._map_tensors(lambda tensor: tensor.clone())

    def detach(self) -> '_QuantizedTuckerFormat':
        """
        Returns a detached format sharing tensor storage.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTTTucker` or :class:`~tensorkrowch.formats.QTRTucker`
            Separate containers with detached tensor references. Value edits to
            shared storage affect both formats.
        """
        return self._map_tensors(lambda tensor: tensor.detach())

    def detach_(self) -> '_QuantizedTuckerFormat':
        """
        Detaches structural tensors in-place by replacing references.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTTTucker` or :class:`~tensorkrowch.formats.QTRTucker`
            The current format. Tensor shapes and canonical metadata are
            preserved.
        """
        detached = self.detach()
        self.__dict__.update(detached.__dict__)
        return self

    def _flattened_in_dim(self) -> Tuple[int, ...]:
        """Expands each upper connector into its factor digit dimensions."""
        coordinate_by_position = {
            position: coordinate
            for coordinate, position in enumerate(self.coordinate_positions)}
        dimensions = []
        for site, dimension in enumerate(self.upper.in_dim):
            coordinate = coordinate_by_position.get(site)
            if coordinate is None:
                dimensions.append(dimension)
            else:
                dimensions.extend(self.factors[coordinate].in_dim[:-1])
        return tuple(dimensions)

    def _effective_cores(self) -> List[torch.Tensor]:
        """Replaces upper input sites by their factors with bonds absorbed."""
        coordinate_by_position = {
            position: coordinate
            for coordinate, position in enumerate(self.coordinate_positions)}
        flat = []
        for site, upper_core in enumerate(self.upper._effective_cores()):
            coordinate = coordinate_by_position.get(site)
            if coordinate is None:
                flat.append(upper_core)
                continue

            factor_cores = self.factors[coordinate]._effective_cores()
            digit_cores = factor_cores[:-1]
            connector = factor_cores[-1].squeeze(-1)
            upper_left = upper_core.shape[0]
            if len(digit_cores) > 1:
                identity = torch.eye(upper_left, device=self.device, dtype=self.dtype)
                for digit_core in digit_cores[:-1]:
                    combined = torch.einsum('ab,lpr->alpbr', identity, digit_core)
                    flat.append(combined.reshape(
                        upper_left * digit_core.shape[0], digit_core.shape[1],
                        upper_left * digit_core.shape[2]))
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

    def flatten(self) -> Union[QTT, QTR]:
        """
        Substitutes each upper connector with its local Quantics factor.

        Only scalar-output formats can be flattened to QTT/QTR. Tensor-valued
        outputs raise ``ValueError`` because QTT/QTR require a digit at every
        site.

        Returns
        -------
        :class:`~tensorkrowch.formats.QTT` or :class:`~tensorkrowch.formats.QTR`
            Exact flat format for scalar outputs, preserving coordinate maps.
            Digit sites become contiguous factor blocks with a
            corresponding custom layout. Upper ranks are carried through
            identity factors.

        Examples
        --------
        >>> upper = tk.formats.TT([torch.ones(2)])
        >>> factor = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> layout = tk.formats.QuantizedLayout(1, 2, 1)
        >>> format = tk.formats.QTTTucker(upper, [factor], layout)
        >>> flat = format.flatten()
        >>> indices = torch.tensor([[0], [1]])
        >>> torch.allclose(flat.evaluate_indices(indices),
        ...                format.evaluate_indices(indices))
        True
        """
        if self.out_shape:
            raise ValueError(
                'Cannot flatten tensor-valued Tucker formats to QTT/QTR; '
                'every QTT/QTR site should represent a digit')

        effective_cores = self._effective_cores()
        dimensions = self._flattened_in_dim()
        cyclic = self._upper_type is TR
        cores = _restore_cores(effective_cores, dimensions, None, 0, cyclic)

        # Keep the coordinate schedule aligned with the flattened factor blocks.
        schedule = [site for coordinate in range(self.layout.n_coordinates)
                    for site in self.layout.sites() if site[0] == coordinate]
        layout = QuantizedLayout(
            self.layout.n_coordinates, self.layout.base, self.layout.level,
            ordering='custom', digit_order=self.layout.digit_order,
            permutation=schedule)
        coordinate_map = self.coordinate_map
        if coordinate_map is None:
            domain = self.domain if self.domain is not None else torch.tensor(
                [0., 1.], device=self.device, dtype=self.upper.cores[0].real.dtype)
            coordinate_map = AffineCoordinateMap(
                domain, layout.grid_size, grid_offset=self.computational_grid,
                out_of_domain=self.out_of_domain)
        cls = QTR if cyclic else QTT
        return cls(cores, layout.n_coordinates, layout=layout,
                   coordinate_map=coordinate_map)

    def _factor_vectors(self, digits: torch.Tensor) -> List[torch.Tensor]:
        """Returns each factor's connector vector for the scheduled digits."""
        schedule = self.layout.sites()
        vectors = []
        for coordinate, factor in enumerate(self.factors):
            columns = [
                column
                for column, site in enumerate(schedule)
                if site[0] == coordinate]
            coordinate_digits = digits.index_select(
                -1,
                torch.tensor(columns, device=digits.device))
            state = None
            factor_cores = factor._effective_cores()
            for site, core in enumerate(factor_cores[:-1]):
                local = core[:, coordinate_digits[..., site], :].movedim(0, -2)
                state = local if state is None else state @ local
            connector = factor_cores[-1].squeeze(-1)
            vectors.append((state @ connector).squeeze(-2))
        return vectors

    def _contract_upper(self,
                        vectors: Dict[int, torch.Tensor],
                        batch_size: int) -> torch.Tensor:
        """Contracts upper cores with local factor vectors, retaining output sites."""
        cores = self.upper._effective_cores()
        closing = cores[0].shape[-3]
        # Open chains use the same contraction with a unit closing bond.
        state = torch.eye(closing, device=self.device, dtype=self.dtype)
        state = state.expand(batch_size, -1, -1)
        for site, core in enumerate(cores):
            if site in vectors:
                local = torch.einsum('bp,lpr->blr', vectors[site], core)
                state = torch.einsum('ba...l,blr->ba...r', state, local)
            else:
                state = torch.einsum('ba...l,lpr->ba...pr', state, core)
        return state.diagonal(dim1=1, dim2=-1).sum(-1)

    def evaluate_digits(self, digits: torch.Tensor) -> torch.Tensor:
        """
        Evaluates digit configurations through the factors and upper TT/TR.

        Parameters
        ----------
        digits : torch.Tensor
            Integer digit configurations in layout schedule order, with shape
            ``(batch_size, layout.n_sites)``. Every digit should lie within its
            site base.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch_size, *out_shape)``. Upper sites outside
            ``coordinate_positions`` remain open.
        """
        digits = self.layout._integer_tensor(digits, 'digits').to(self.device)
        if digits.ndim != 2 or digits.shape[-1] != self.layout.n_sites:
            raise ValueError(
                '`digits` should have shape (batch_size, layout.n_sites)')
        self.layout.decode_digits(digits)

        # Factor connectors provide the inputs to their upper sites.
        factor_vectors = self._factor_vectors(digits)
        vectors_by_position = {
            position: factor_vectors[coordinate]
            for coordinate, position in enumerate(self.coordinate_positions)}
        return self._contract_upper(vectors_by_position, digits.shape[0])

    def evaluate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Evaluates integer grid indices in the original coordinate order.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape ``(batch_size, n_coordinates)``; each
            value lies in ``[0, grid_size[coordinate] - 1]``.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch_size, *out_shape)``. Upper sites outside
            ``coordinate_positions`` remain open.
        """
        indices = self.layout._integer_tensor(indices, 'indices')
        if indices.ndim != 2 or indices.shape[-1] != self.layout.n_coordinates:
            raise ValueError(
                '`indices` should have shape (batch_size, n_coordinates)')
        return self.evaluate_digits(self.layout.encode_indices(indices))

    def evaluate(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Evaluates coordinates in the domain using the digit grid.

        Coordinates in the domain require a coordinate map and are quantized
        before contraction.

        Parameters
        ----------
        coordinates : torch.Tensor
            Finite coordinates in the domain with shape
            ``(batch_size, n_coordinates)``. A coordinate map is required.
            Coordinates are quantized to the computational grid; no
            interpolation of the represented function is performed.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch_size, *out_shape)``. Upper sites outside
            ``coordinate_positions`` remain open.

        Examples
        --------
        >>> upper = tk.formats.TT([torch.tensor([2., 3.])])
        >>> factor = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> layout = tk.formats.QuantizedLayout(1, base=2, level=1)
        >>> format = tk.formats.QTTTucker(upper, [factor], layout,
        ...     tk.formats.UniformCoordinateMap(), domain=torch.tensor([0., 1.]))
        >>> format.evaluate(torch.tensor([[0.], [1.]])).tolist()
        [2.0, 3.0]
        """
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.to(device=self.device)
        indices = _coordinates_to_indices(
            coordinates, self.layout, self.coordinate_map, self.domain,
            self.computational_grid, self.out_of_domain)
        return self.evaluate_indices(indices)

    def evaluate_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Evaluates coordinates in the domain, as in :meth:`evaluate`.

        Coordinates in the domain require a coordinate map and are quantized
        before contraction.

        Parameters
        ----------
        coordinates : torch.Tensor
            Finite coordinates in the domain with shape
            ``(batch_size, n_coordinates)``. A coordinate map is required.
            Coordinates are quantized to the computational grid; no
            interpolation of the represented function is performed.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch_size, *out_shape)``. Upper sites outside
            ``coordinate_positions`` remain open.
        """
        return self.evaluate(coordinates)

    def contract_dense(self) -> torch.Tensor:
        """
        Returns the full tensor with axes in flattened digit order.

        This allocates the entire tensor, including open output sites.

        Returns
        -------
        torch.Tensor
            Dense tensor in flattened digit-site order, including open output
            sites.
        """
        return self.flatten().contract_dense()

    def norm(self) -> torch.Tensor:
        """
        Returns the norm of the flattened represented tensor.

        Returns
        -------
        torch.Tensor
            Scalar Frobenius norm.
        """
        return self.flatten().norm()

    def normalized_overlap(
            self, other: '_QuantizedTuckerFormat') -> torch.Tensor:
        """
        Returns the phase-preserving normalized overlap.

        Parameters
        ----------
        other : :class:`~tensorkrowch.formats.QTTTucker` or :class:`~tensorkrowch.formats.QTRTucker`
            Other hierarchical format whose flattened local dimensions and
            device match this format.

        Returns
        -------
        torch.Tensor
            Scalar normalized overlap, retaining its complex phase. Zero-norm
            operands raise ``ValueError``.
        """
        if not isinstance(other, _QuantizedTuckerFormat):
            raise TypeError(
                '`other` should be a quantized Tucker format')
        return self.flatten().normalized_overlap(other.flatten())

    def fidelity(self, other: '_QuantizedTuckerFormat') -> torch.Tensor:
        """
        Returns the squared magnitude of :meth:`normalized_overlap`.

        Parameters
        ----------
        other : :class:`~tensorkrowch.formats.QTTTucker` or :class:`~tensorkrowch.formats.QTRTucker`
            Other hierarchical format whose flattened local dimensions and
            device match this format.

        Returns
        -------
        torch.Tensor
            Real scalar fidelity. Zero-norm operands raise ``ValueError``.
        """
        return self.normalized_overlap(other).abs().square()


class QTTTucker(_QuantizedTuckerFormat):
    """
    Quantics factors connected to an upper tensor train.

    Parameters
    ----------
    upper : :class:`~tensorkrowch.formats.TT`
        Unbatched upper tensor train. Its coordinate sites represent connector
        indices.
    factors : sequence of TT
        One unbatched factor per original coordinate. Each contains its digit
        sites followed by a connector site matching the corresponding
        upper dimension.
    layout : :class:`~tensorkrowch.formats.QuantizedLayout`
        Original-coordinate bases, levels and evaluation digit schedule.
    coordinate_map : :class:`~tensorkrowch.formats.CoordinateMap`, optional
        Map used to quantize coordinates in the domain. Integer and digit
        evaluation do not require one.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Domain intervals as ``(2,)`` for a shared interval or
        ``(n_coordinates, 2)`` for separate intervals. Interval-based maps
        require a domain; maps with their own grid or domain geometry can use
        ``None``.
    coordinate_positions : sequence of int, optional
        Strictly increasing upper sites receiving factor connectors. Other
        upper sites remain output axes. Required when upper contains output
        sites.
    computational_grid : {"endpoints", "left", "centers", "right"} or float
        Uniform grid convention or within-cell offset in ``[0, 1]``, as in
        :class:`~tensorkrowch.formats.UniformCoordinateMap`. Used when
        the coordinate map provides no direct index lookup.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ``ValueError`` or are
        clipped to the domain boundary.
    """

    _upper_type = TT
    _topology = 'qtt_tucker'


class QTRTucker(_QuantizedTuckerFormat):
    """
    Quantics factors connected to an upper tensor ring.

    Parameters
    ----------
    upper : :class:`~tensorkrowch.formats.TR`
        Unbatched upper tensor ring. Its coordinate sites represent connector
        indices.
    factors : sequence of :class:`~tensorkrowch.formats.TT`
        One unbatched factor per original coordinate. Each contains its digit
        sites followed by a connector site matching the corresponding
        upper dimension.
    layout : :class:`~tensorkrowch.formats.QuantizedLayout`
        Original-coordinate bases, levels and evaluation digit schedule.
    coordinate_map : :class:`~tensorkrowch.formats.CoordinateMap`, optional
        Map used to quantize coordinates in the domain. Integer and digit
        evaluation do not require one.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Domain intervals as ``(2,)`` for a shared interval or
        ``(n_coordinates, 2)`` for separate intervals. Interval-based maps
        require a domain; maps with their own grid or domain geometry can use
        ``None``.
    coordinate_positions : sequence of int, optional
        Strictly increasing upper sites receiving factor connectors. Other
        upper sites remain output axes. Required when upper contains output
        sites.
    computational_grid : {"endpoints", "left", "centers", "right"} or float
        Uniform grid convention or within-cell offset in ``[0, 1]``, as in
        :class:`~tensorkrowch.formats.UniformCoordinateMap`. Used when
        the coordinate map provides no direct index lookup.
    out_of_domain : {"error", "clip"}
        Whether coordinates outside the domain raise ``ValueError`` or are
        clipped to the domain boundary.
    """

    _upper_type = TR
    _topology = 'qtr_tucker'
