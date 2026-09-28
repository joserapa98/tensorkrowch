"""Shared raw-tensor chain operations, independent of decomposition engines."""

from abc import abstractmethod
from copy import copy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.formats.base import TensorFormat, SampleError
from tensorkrowch.formats.bonds import VidalGauge


EvaluationData = Union[torch.Tensor, Sequence[torch.Tensor]]

_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64
)


class _CoreList(list):
    """Fixed-length core container with invalidation on replacement."""

    def __init__(self, cores: Sequence[torch.Tensor], owner) -> None:
        super().__init__(cores)
        self._owner = owner

    def __setitem__(self, key, value):
        values = list(value) if isinstance(key, slice) else [value]

        if not all(isinstance(core, torch.Tensor) for core in values):
            raise TypeError('`cores` should contain torch.Tensor objects')
        if isinstance(key, slice) and (len(values) != len(self[key])):
            raise ValueError('Core slice replacement should preserve length')

        super().__setitem__(key, values if isinstance(key, slice) else value)
        self._owner._dirty = True
        self._owner._orth_center = None
        if isinstance(self._owner._bonds, VidalGauge):
            self._owner._bonds._valid = False

    def _structural_error(self, *args, **kwargs):
        raise TypeError('Use the full cores setter to change the network structure')

    append = extend = insert = pop = remove = clear = _structural_error
    reverse = sort = __delitem__ = __iadd__ = __imul__ = _structural_error


class TensorFormat1D(TensorFormat):
    """Compact chain of raw cores, with cached dimensions and bond ranks.

    The constructor copies the container and shares tensor storage. Element
    and same-length slice replacement invalidate structural metadata; the next
    public access validates the complete network. Tensor value updates preserve
    dimensions. Shape changes through ``tensor.resize_`` are outside this contract.
    """

    _family = 'tensor'
    _topology = 'tensor'
    _cyclic = False

    def __init__(self, cores: Sequence[torch.Tensor], n_batches: int = 0) -> None:
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        self._n_batches = n_batches
        self._dirty = True
        self._orth_center = None
        self._bonds = None
        self.cores = cores

    @property
    def cores(self):
        """Mutable, fixed-length core sequence."""
        return self._cores

    @cores.setter
    def cores(self, cores: Sequence[torch.Tensor]):
        if isinstance(cores, torch.Tensor):
            raise TypeError('`cores` should be a sequence of torch.Tensor objects')

        cores = list(cores)
        previous = self.__dict__.copy()
        self._cores = _CoreList(cores, self)
        self._dirty = True

        try:
            self.validate()
        except (TypeError, ValueError):
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise

        self._orth_center = None

    @property
    def n_batches(self) -> int:
        """Number of leading structural batch axes."""
        return self._n_batches

    def validate(self):
        """Validates the complete network and refreshes cached metadata."""
        if not self._cores:
            raise ValueError('`cores` should contain at least one tensor')
        if not all(isinstance(core, torch.Tensor) for core in self._cores):
            raise TypeError('`cores` should contain torch.Tensor objects')

        device, dtype = self._cores[0].device, self._cores[0].dtype
        for core in self._cores:
            if any(dim < 1 for dim in core.shape):
                raise ValueError('Core dimensions should be positive')
            if core.device != device:
                raise ValueError('All cores should be on the same device')
            if core.dtype != dtype:
                raise ValueError('All cores should have the same dtype')

        rank, batch_shape, in_dim, out_dim = self._validate_cores()

        self._rank = tuple(rank)
        self._batch_shape = batch_shape
        self._dirty = False

        self._in_dim = in_dim
        self._out_dim = out_dim
        self._same_in_dim = all(dim == in_dim[0] for dim in in_dim)
        self._same_out_dim = out_dim is None or all(
            dim == out_dim[0] for dim in out_dim)

        if self._bonds is not None:
            self._bonds.validate(self._raw_standard_cores(),
                                 self._cyclic)

        return self

    def _ensure_valid(self):
        if self._dirty:
            self.validate()

    def _map_tensors(self, function):
        self._ensure_valid()
        result = copy(self)
        result._cores = _CoreList([function(core) for core in self._cores], result)
        if self._bonds is not None:
            result._bonds = self._bonds._map_tensors(function)
        return result

    def _same_aux_tensors(self, other):
        return True

    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           copy: bool = False):
        """Returns a conversion preserving the concrete class and autograd.

        With copy=False, a no-op conversion returns self. Unsupported device
        operations propagate PyTorch errors, without a CPU fallback.
        """
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type')
        if not isinstance(copy, bool):
            raise TypeError('`copy` should be bool type')

        result = self._map_tensors(lambda tensor: tensor.to(
            device=device, dtype=dtype, copy=copy))

        same_cores = all(
            new is old for new, old in zip(result._cores, self._cores))
        same_bonds = self._bonds is None or all(
            new is old for new, old in zip(result._bonds.values, self._bonds.values))

        if not copy and same_cores and same_bonds and self._same_aux_tensors(result):
            return self

        result.validate()
        return result

    def clone(self):
        """Returns a network with independent tensor storage, preserving autograd."""
        return self._map_tensors(lambda tensor: tensor.clone())

    def detach(self):
        """Returns a detached container sharing tensor storage."""
        return self._map_tensors(lambda tensor: tensor.detach())

    def detach_(self):
        """Detaches every core by replacing references, including tensor views."""
        self._cores = _CoreList([core.detach() for core in self._cores], self)
        if self._bonds is not None:
            self._bonds = self._bonds._map_tensors(lambda tensor: tensor.detach())
        return self

    def _standard_cores(self):
        self._ensure_valid()
        cores = self._raw_standard_cores()
        if self._bonds is not None:
            self._bonds.validate(cores, self._cyclic)
            cores = list(cores)
            for site, value in enumerate(self._bonds.values):
                if value is not None:
                    cores[site] = cores[site] * value[..., None, None, :]
        return cores

    @property
    def bonds(self):
        """Optional diagonal factors, with one entry per network bond."""
        return self._bonds

    @bonds.setter
    def bonds(self, value):
        from tensorkrowch.formats.bonds import BondFactors

        self._ensure_valid()
        if value is not None:
            if not isinstance(value, BondFactors):
                raise TypeError('`bonds` should be BondFactors type or None')
            value.validate(self._raw_standard_cores(), self._cyclic)
        self._bonds = value
        self._orth_center = None

    def materialize_bonds(self, oc: Optional[int] = None):
        """Absorbs factors towards the selected orthogonality center in-place."""
        from tensorkrowch.formats.canonical import materialize_bonds

        return materialize_bonds(self, oc)

    def _set_standard_cores(self, cores: Sequence[torch.Tensor], bonds=None):
        from tensorkrowch.formats.operations import _build_network

        result = _build_network(cores, self._in_dim, self._out_dim,
                                self._n_batches, self._cyclic)
        self._cores = _CoreList(result._cores, self)
        self._bonds = bonds
        self._dirty = True
        self.validate()

    def canonicalize(self, oc: Optional[int] = None, renormalize: bool = False):
        """QR/RQ sweeps in-place, preserving the tensor and its global scale.

        oc defaults to the last site. On cyclic networks this is a local gauge
        relative to the stored cut, without a global Schmidt interpretation.
        """
        from tensorkrowch.formats.canonical import canonicalize

        return canonicalize(self, oc, renormalize)

    def canonicalize_vidal(self,
                           mode: str = 'implicit',
                           inverse_positions: Optional[Sequence[int]] = None,
                           remaining_mode: str = 'implicit',
                           inverse_cutoff: float = 0.0):
        """Builds or redistributes an open-chain Vidal gauge in-place.

        Inverse bonds require every retained Schmidt value to be strictly above
        inverse_cutoff. No truncation is performed to manufacture an inverse.
        Cyclic networks do not admit this global open-chain Schmidt gauge.
        """
        from tensorkrowch.formats.canonical import canonicalize_vidal

        return canonicalize_vidal(self, mode, inverse_positions,
                                  remaining_mode, inverse_cutoff)

    def rounding(self,
                 rank: Optional[int] = None,
                 cutoff: Optional[float] = None,
                 atol: Optional[float] = None,
                 rtol: Optional[float] = None,
                 cum_percentage: Optional[float] = None,
                 renormalize: bool = False,
                 *,
                 rel_error: Optional[float] = None,
                 return_info: bool = False):
        """Compresses ranks in-place with one QR/SVD execution.

        Open chains use left QR followed by sitewise right SVD. Cyclic chains
        implement Algorithm 4 of Mickelin and Karaman,
        https://arxiv.org/pdf/1807.02513, including the closure reduction.
        TR ranks need not become minimal, especially after block-diagonal sums
        or products. rel_error specifies a global norm-error budget; rtol keeps
        the squared-tail-energy meaning of utils.truncated_svd.
        """
        from tensorkrowch.formats.rounding import rounding

        return rounding(self, rank, cutoff, atol, rtol, cum_percentage,
                        renormalize, rel_error, return_info)

    def canonicalize_minimal(self, max_iter: int = 200, lr: float = 0.05,
                             tol: float = 1e-8, *, return_info: bool = False):
        """Uses implicit Vidal for trains or experimental gauge balancing for rings.

        Ring optimization preserves the best finite iterate and may stop before
        convergence. It optimizes temporary gauges without accumulating input
        core gradients; batched rings share one gauge per bond.
        return_info returns (self, MinimalCanonicalInfo), reporting convergence
        and the final Gram imbalance without creating a diagnostic history.
        """
        from tensorkrowch.formats.orbits import canonicalize_minimal

        return canonicalize_minimal(self, max_iter, lr, tol, return_info)

    def block(self, groups: Sequence[int], return_info: bool = False):
        """Returns a network of contiguous blocks with recoverable dimensions."""
        from tensorkrowch.formats.blocking import block

        return block(self, groups, return_info)

    def unblock(self, info=None, **kwargs):
        """Returns the original site layout, optionally truncating local splits."""
        from tensorkrowch.formats.blocking import unblock

        return unblock(self, info, **kwargs)

    def contract_block(self, first, last):
        """Returns a local tensor with both external ranks left open."""
        from tensorkrowch.formats.blocking import contract_block

        return contract_block(self, first, last)

    def split_block(self, block: torch.Tensor, first, last, **kwargs):
        """Splits a local tensor into standard fused cores and internal factors."""
        from tensorkrowch.formats.blocking import split_block

        self._ensure_valid()
        if any(isinstance(site, bool) or not isinstance(site, int)
               for site in (first, last)):
            raise TypeError('Block endpoints should be integers')
        if not 0 <= first <= last < self.n_sites:
            raise ValueError('Block endpoints should select an ordered region')
        outputs = None if self._out_dim is None else self._out_dim[first:last + 1]
        return split_block(block, self._in_dim[first:last + 1], outputs,
                           self._n_batches, **kwargs)

    def replace_block(self, first, last, replacement):
        """Replaces a region atomically, preserving external interfaces."""
        from tensorkrowch.formats.blocking import replace_block

        return replace_block(self, first, last, replacement)

    def absorb_bond(self, bond, side: str = 'left'):
        """Moves one bond's diagonal weights into the selected neighbour."""
        from tensorkrowch.formats.blocking import absorb_bond

        return absorb_bond(self, bond, side)

    def redistribute_bond(self, bond: int, mode: str = 'implicit',
                          inverse_cutoff: float = 0.0):
        """Changes one stored Vidal bond distribution without recomputing SVDs.

        Modes are explicit, implicit, inverse, left and right. A locally split
        block can use this operation even when its spectra are not globally
        certified Schmidt values. The other bonds keep their distribution.
        """
        from tensorkrowch.formats.blocking import redistribute_bond

        return redistribute_bond(self, bond, mode, inverse_cutoff)

    def add(self, other, method='stacked'):
        """Returns an exact sum; cyclic sums default to stacked endpoints."""
        from tensorkrowch.formats.operations import add

        return add(self, other, method=method)

    def sub(self, other, method='stacked'):
        """Returns an exact difference with the chosen cyclic sum construction."""
        from tensorkrowch.formats.operations import add

        return add(self, other, method=method, coefficient=-1)

    def hadamard(self, other):
        """Returns the element-wise product with another compatible format."""
        from tensorkrowch.formats.operations import hadamard

        return hadamard(self, other)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.sub(other)

    def __neg__(self):
        return self * -1

    def __mul__(self, other):
        from tensorkrowch.formats.operations import scale

        return self.hadamard(other) if isinstance(
            other, TensorFormat1D) else scale(self, other)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        from tensorkrowch.formats.operations import apply

        return apply(self, other)

    def apply(self, other):
        """Returns self @ other for vector-matrix application."""
        return self @ other

    def conj(self):
        """Returns the conjugate format, preserving bond-factor conjugation."""
        return self._map_tensors(lambda tensor: tensor.conj())

    def contract_dense(self) -> torch.Tensor:
        """Contracts a small dense tensor; matrix axes remain interleaved."""
        self._ensure_valid()
        cores = self._standard_cores()
        closing = cores[0].shape[-3]
        result = cores[0]
        dimensions = [result.shape[-2]]
        for core in cores[1:]:
            result = result.reshape(*self._batch_shape, closing, -1, result.shape[-1])
            result = torch.einsum('...apr,...rqb->...apqb', result, core)
            dimensions.append(core.shape[-2])
        result = result.reshape(*self._batch_shape, closing, *
                                dimensions, cores[-1].shape[-1])
        result = result.diagonal(dim1=self._n_batches, dim2=-1).sum(-1)
        if self._out_dim is not None:
            dimensions = [dim for pair in zip(
                self._in_dim, self._out_dim) for dim in pair]
            result = result.reshape(*self._batch_shape, *dimensions)
        return result

    def inner(self, other) -> torch.Tensor:
        """Contracts the conjugate of self with other using scaled environments."""
        phase, log_magnitude = self._log_overlap(other)
        return phase * log_magnitude.exp()

    @property
    def device(self) -> torch.device:
        """Device shared by all cores."""
        self._ensure_valid()
        return self.cores[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Data type shared by all cores."""
        self._ensure_valid()
        return self.cores[0].dtype

    @property
    def rank(self) -> List[int]:
        """Bond ranks inferred from the cores."""
        self._ensure_valid()
        return list(self._rank)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Batch dimensions shared by the cores."""
        self._ensure_valid()
        return self._batch_shape

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension associated with every site."""
        self._ensure_valid()
        return self._in_dim

    @property
    def n_sites(self) -> int:
        """Number of sites in the stored core network.

        Each stored core represents one site. A hierarchical quantized result
        counts its upper-network sites; :meth:`flatten` constructs a separate
        result whose sites include the digit factors.
        """
        self._ensure_valid()
        return len(self.cores)

    @property
    def out_dim(self) -> Optional[Tuple[int, ...]]:
        """Output dimension per site, when the decomposition has one."""
        self._ensure_valid()
        return self._out_dim

    @property
    def topology(self) -> str:
        """Topology identifier used in serialized result information."""
        self._ensure_valid()
        return self._topology

    @abstractmethod
    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates cores and returns rank, batch, input and output dims."""

    @abstractmethod
    def _raw_standard_cores(self) -> List[torch.Tensor]:
        """Returns (*batch, left, physical, right) cores without bond factors."""

    def _normalize_data(
            self,
            data: EvaluationData,
            dimensions: Sequence[int],
            same_dim: bool,
            n_batches: int
    ) -> Tuple[List[torch.Tensor], bool, Tuple[int, ...]]:
        """Normalizes discrete indices or embedded vectors by site."""
        self._ensure_valid()
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        n_sites = len(dimensions)
        if isinstance(data, torch.Tensor):
            if data.ndim == (n_batches + 1):
                if data.shape[-1] != n_sites:
                    raise ValueError(
                        'The last dimension of discrete inputs should equal '
                        'the number of sites')
                if data.dtype not in _INTEGER_DTYPES:
                    raise TypeError(
                        'Discrete inputs should have an integer dtype')
                site_data = list(data.to(device=self.device).unbind(-1))
                discrete = True
            elif data.ndim == (n_batches + 2):
                if not same_dim:
                    raise ValueError(
                        'Embedded inputs should be provided as a sequence when '
                        'site dimensions differ')
                if data.shape[-2:] != (n_sites, dimensions[0]):
                    raise ValueError(
                        'Embedded inputs should end in `(n_sites, site_dim)`')
                site_data = list(
                    data.to(device=self.device, dtype=self.dtype).unbind(-2))
                discrete = False
            else:
                raise ValueError(
                    '`data` has an incompatible number of batch dimensions')
        else:
            try:
                site_data = list(data)
            except TypeError as exc:
                raise TypeError(
                    '`data` should be a tensor or a sequence of tensors') \
                    from exc
            if len(site_data) != n_sites:
                raise ValueError(
                    '`data` should contain one tensor per site')
            if not all(isinstance(item, torch.Tensor)
                       for item in site_data):
                raise TypeError('Every site input should be a torch.Tensor')

            if all(item.ndim == n_batches for item in site_data):
                if not all(item.dtype in _INTEGER_DTYPES
                           for item in site_data):
                    raise TypeError(
                        'Discrete inputs should have an integer dtype')
                discrete = True
            elif all(item.ndim == (n_batches + 1)
                     for item in site_data):
                for item, site_dim in zip(site_data, dimensions):
                    if item.shape[-1] != site_dim:
                        raise ValueError(
                            'The last dimension of each embedded input should '
                            'match its site dimension')
                discrete = False
            else:
                raise ValueError(
                    'Site inputs should all be discrete indices or embedded '
                    'vectors')

            target_dtype = None if discrete else self.dtype
            site_data = [
                item.to(device=self.device, dtype=target_dtype)
                for item in site_data
            ]

        batch_shape = tuple(site_data[0].shape[:n_batches])
        if any(tuple(item.shape[:n_batches]) != batch_shape
               for item in site_data[1:]):
            raise ValueError('All site inputs should have the same batch shape')

        if discrete:
            for indices, site_dim in zip(site_data, dimensions):
                if torch.any(indices < 0) or torch.any(indices >= site_dim):
                    raise ValueError(
                        'Discrete indices should lie in each site dimension')

        return site_data, discrete, batch_shape

    @staticmethod
    def _contract_open_chain(
            matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts an open chain of matrices along adjacent ranks."""
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result

    def _check_overlap_compatibility(
            self, other: 'TensorFormat1D') -> None:
        """Validates topology, shapes and runtime for an overlap."""
        self._ensure_valid()
        if not isinstance(other, TensorFormat1D):
            raise TypeError('`other` should be TensorFormat1D type')
        other._ensure_valid()
        if self._family != other._family:
            raise ValueError('The decomposition families are incompatible')
        if len(self.cores) != len(other.cores):
            raise ValueError('Decompositions should have the same number of sites')
        if self._in_dim != other._in_dim:
            raise ValueError('Decompositions should have matching input dimensions')
        if self._out_dim != other._out_dim:
            raise ValueError('Decompositions should have matching output dimensions')
        if self._batch_shape != other._batch_shape:
            raise ValueError('Decompositions should have matching batch shapes')
        if self.device != other.device:
            raise ValueError('Decompositions should be on the same device')

    def _log_overlap(
            self, other: 'TensorFormat1D') -> Tuple[torch.Tensor,
                                                    torch.Tensor]:
        """Returns overlap phase and log-magnitude using scaled environments."""
        self._ensure_valid()
        self._check_overlap_compatibility(other)
        dtype = torch.promote_types(self.dtype, other.dtype)
        self_cores = self._standard_cores()
        other_cores = self_cores if other is self else other._standard_cores()

        self_eye = torch.eye(
            self_cores[0].shape[-3], device=self.device, dtype=dtype)
        other_eye = torch.eye(
            other_cores[0].shape[-3], device=self.device, dtype=dtype)
        environment = torch.einsum('ai,bj->abij', self_eye, other_eye)
        if self.n_batches:
            environment = environment.reshape(
                *((1,) * self.n_batches), *environment.shape)
            environment = environment.expand(
                *self._batch_shape, *environment.shape[self.n_batches:])

        real_dtype = torch.empty((), dtype=dtype).real.dtype
        log_scale = torch.zeros(
            self._batch_shape, device=self.device, dtype=real_dtype)

        for self_core, other_core in zip(self_cores, other_cores):
            self_core = self_core.to(dtype=dtype)
            self_scale = self_core.abs().amax(dim=(-3, -2, -1))
            self_scale = torch.where(self_scale > 0, self_scale,
                                     torch.ones_like(self_scale))
            self_core = self_core / self_scale[..., None, None, None]
            if other is self:
                other_scale, other_core = self_scale, self_core
            else:
                other_core = other_core.to(dtype=dtype)
                other_scale = other_core.abs().amax(dim=(-3, -2, -1))
                other_scale = torch.where(
                    other_scale > 0, other_scale, torch.ones_like(other_scale))
                other_core = other_core / other_scale[..., None, None, None]
            log_scale = log_scale + self_scale.log() + other_scale.log()
            environment = torch.einsum(
                '...xyab,...apr->...xybpr',
                environment,
                self_core.conj())
            environment = torch.einsum(
                '...xybpr,...bps->...xyrs',
                environment,
                other_core)

            scale = torch.linalg.vector_norm(
                environment, dim=(-4, -3, -2, -1))
            nonzero = scale > 0
            safe_scale = torch.where(nonzero, scale, torch.ones_like(scale))
            environment = environment / safe_scale[..., None, None, None, None]
            log_scale = log_scale + torch.where(
                nonzero, safe_scale.log(), torch.zeros_like(safe_scale))

        overlap = torch.einsum('...ijij->...', environment)
        magnitude = overlap.abs()
        nonzero = magnitude > 0
        safe_magnitude = torch.where(
            nonzero, magnitude, torch.ones_like(magnitude))
        phase = torch.where(nonzero,
                            overlap / safe_magnitude,
                            torch.zeros_like(overlap))
        log_magnitude = torch.where(
            nonzero,
            safe_magnitude.log() + log_scale,
            torch.full_like(log_scale, -torch.inf))
        return phase, log_magnitude

    def norm(self) -> torch.Tensor:
        """Returns the norm obtained by a scaled double-layer contraction."""
        self._ensure_valid()
        _, log_squared_norm = self._log_overlap(self)
        return torch.exp(log_squared_norm / 2)

    def normalized_overlap(
            self, other: 'TensorFormat1D') -> torch.Tensor:
        """Returns ``<self, other> / (||self|| ||other||)`` with its phase."""
        self._ensure_valid()
        phase, log_overlap = self._log_overlap(other)
        _, log_self = self._log_overlap(self)
        _, log_other = other._log_overlap(other)

        if torch.any(torch.isneginf(log_self)) or \
                torch.any(torch.isneginf(log_other)):
            raise ValueError(
                'Normalized overlap is undefined for a zero-norm decomposition')

        log_denominator = (log_self + log_other) / 2
        return phase * torch.exp(log_overlap - log_denominator)

    def fidelity(self, other: 'TensorFormat1D') -> torch.Tensor:
        """Returns ``abs(normalized_overlap(other)) ** 2``."""
        self._ensure_valid()
        return self.normalized_overlap(other).abs().square()


class _VectorFormat1D(TensorFormat1D):
    """Shared raw-tensor vector operations."""

    _family = 'state'

    def __call__(self,
                 data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        """Calls :meth:`evaluate`."""
        return self.evaluate(data, n_batches=n_batches)

    def _local_matrices(
            self,
            site_data: Sequence[torch.Tensor],
            discrete: bool,
            data_batch_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Contracts TT/TR cores with one input at each site."""
        core_batch_size = int(torch.Size(self._batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        matrices = []

        for core, site_value in zip(self._standard_cores(), site_data):
            left_rank, site_in_dim, right_rank = core.shape[-3:]
            core = core.reshape(
                core_batch_size, left_rank, site_in_dim, right_rank)

            if discrete:
                indices = site_value.reshape(data_batch_size).to(torch.long)
                matrix = core[:, :, indices, :].permute(0, 2, 1, 3)
            else:
                vectors = site_value.reshape(data_batch_size, site_in_dim)
                matrix = torch.einsum('dp,clpr->cdlr', vectors, core)

            matrices.append(matrix.reshape(
                *self._batch_shape,
                *data_batch_shape,
                left_rank,
                right_rank))

        return matrices

    def evaluate(self,
                 data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        """Evaluates the decomposition on indices or embedded input vectors.

        A tensor of discrete indices has shape ``(*batch, n_sites)``. A tensor
        of embedded inputs requires uniform input dimensions and has shape
        ``(*batch, n_sites, in_dim)``. A sequence may instead contain one
        index tensor of shape ``(*batch,)`` or one embedded tensor of shape
        ``(*batch, in_dim[site])`` per site.

        ``n_batches`` describes the leading batch dimensions of ``data``;
        :attr:`n_batches` describes independent batch dimensions stored in the
        cores. Both groups are preserved in the returned tensor, with core
        batches followed by input batches.
        """
        self._ensure_valid()
        site_data, discrete, data_batch_shape = self._normalize_data(
            data, self._in_dim, self._same_in_dim, n_batches)
        matrices = self._local_matrices(
            site_data, discrete, data_batch_shape)
        return self._contract_local_matrices(matrices)

    def error(self,
              function: Callable[..., torch.Tensor],
              samples: torch.Tensor,
              data: Optional[EvaluationData] = None,
              n_batches: int = 1,
              **kwargs: Any) -> SampleError:
        """Measures errors on samples, optionally using embedded inputs."""
        self._ensure_valid()
        if not callable(function):
            raise TypeError('`function` should be callable')
        if not isinstance(samples, torch.Tensor):
            raise TypeError('`samples` should be torch.Tensor type')
        if samples.ndim != (n_batches + 1):
            raise ValueError(
                '`samples` has an incompatible number of batch dimensions')

        samples = samples.to(device=self.device)
        approximation = self.evaluate(
            samples if data is None else data,
            n_batches=n_batches)
        target = function(samples, **kwargs)
        if not isinstance(target, torch.Tensor):
            raise TypeError('`function` should return a torch.Tensor')
        target = target.to(device=approximation.device,
                           dtype=approximation.dtype)
        target = target.reshape(samples.shape[:n_batches])
        if self.n_batches:
            target = target.reshape(
                *((1,) * self.n_batches), *target.shape)
            target = target.expand(*self._batch_shape, *target.shape[self.n_batches:])

        absolute = torch.linalg.vector_norm(approximation - target)
        denominator = torch.linalg.vector_norm(target)
        if denominator > 0:
            relative = absolute / denominator
        elif absolute == 0:
            relative = torch.zeros_like(absolute)
        else:
            relative = torch.full_like(absolute, torch.inf)

        size = int(torch.Size(samples.shape[:n_batches]).numel())
        return SampleError(
            kind='samples',
            absolute=absolute,
            relative=relative,
            size=size,
            denominator=denominator)

    @abstractmethod
    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts input-selected local matrices along the ranks."""


class _MatrixFormat1D(TensorFormat1D):
    """Shared raw-tensor matrix operations."""

    _family = 'matrix'

    def _operator_cores(self):
        self._ensure_valid()
        return [core.reshape(*self._batch_shape, core.shape[-3],
                             self._in_dim[site], self._out_dim[site],
                             core.shape[-1]).transpose(-1, -2)
                for site, core in enumerate(self._standard_cores())]

    def transpose(self):
        """Swaps local input/output axes, preserving site and bond order."""
        from tensorkrowch.formats.operations import _build_network

        self._ensure_valid()
        standard = []
        for site, core in enumerate(self._raw_standard_cores()):
            core = core.reshape(*self._batch_shape, core.shape[-3],
                                self._in_dim[site], self._out_dim[site], core.shape[-1])
            core = core.transpose(-3, -2)
            standard.append(core.reshape(*self._batch_shape,
                            core.shape[-4], -1, core.shape[-1]))
        result = _build_network(standard, self._out_dim, self._in_dim,
                                self._n_batches, self._cyclic)
        result._bonds = self._bonds
        return result

    def adjoint(self):
        """Returns the conjugate transpose."""
        return self.transpose().conj()

    @property
    def T(self):
        """Matrix transpose, without reversing the chain."""
        return self.transpose()

    @property
    def H(self):
        """Matrix adjoint, including conjugation of bond factors."""
        return self.adjoint()

    def trace(self) -> torch.Tensor:
        """Returns the trace; each local input/output dimension must match."""
        self._ensure_valid()
        if self._in_dim != self._out_dim:
            raise ValueError('Trace requires matching local input/output dimensions')
        matrices = [core.diagonal(dim1=-3, dim2=-1).sum(-1)
                    for core in self._operator_cores()]
        return self._contract_open_chain(matrices).diagonal(dim1=-2, dim2=-1).sum(-1)

    def __call__(
            self,
            in_data: EvaluationData,
            out_data: Optional[EvaluationData] = None,
            n_batches: int = 1
    ) -> Union[torch.Tensor, TensorFormat1D]:
        """Applies the matrix or evaluates it when outputs are provided."""
        if out_data is None:
            return self.apply(in_data, n_batches=n_batches)
        return self.evaluate(in_data, out_data, n_batches=n_batches)

    @abstractmethod
    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts entry-selected matrices with the topology closure."""

    @abstractmethod
    def _build_applied_decomposition(
            self,
            cores: List[torch.Tensor],
            n_batches: int) -> TensorFormat1D:
        """Builds the vector-like result produced by :meth:`apply`."""

    def _entry_matrices(
            self,
            in_data_by_site: Sequence[torch.Tensor],
            out_data_by_site: Sequence[torch.Tensor],
            in_discrete: bool,
            out_discrete: bool,
            data_batch_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Builds local matrices for paired matrix-entry evaluation."""
        core_batch_size = int(torch.Size(self._batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        matrices = []

        for core, in_site_data, out_site_data in zip(
                self._operator_cores(), in_data_by_site, out_data_by_site):
            left_rank, in_dim, right_rank, out_dim = core.shape[-4:]
            core = core.reshape(
                core_batch_size, left_rank, in_dim, right_rank, out_dim)

            if in_discrete and out_discrete:
                in_indices = in_site_data.reshape(data_batch_size).to(torch.long)
                out_indices = out_site_data.reshape(data_batch_size).to(torch.long)
                fused = in_indices * out_dim + out_indices
                matrix = core.permute(0, 1, 3, 2, 4).reshape(
                    core_batch_size, left_rank, right_rank, in_dim * out_dim)
                matrix = matrix[..., fused].permute(0, 3, 1, 2)
            elif in_discrete:
                indices = in_site_data.reshape(data_batch_size).to(torch.long)
                vectors = out_site_data.reshape(data_batch_size, out_dim)
                selected = core.index_select(2, indices)
                matrix = torch.einsum('cldro,do->cdlr', selected, vectors)
            elif out_discrete:
                indices = out_site_data.reshape(data_batch_size).to(torch.long)
                vectors = in_site_data.reshape(data_batch_size, in_dim)
                selected = core.index_select(4, indices)
                matrix = torch.einsum('clird,di->cdlr', selected, vectors)
            else:
                in_vectors = in_site_data.reshape(data_batch_size, in_dim)
                out_vectors = out_site_data.reshape(data_batch_size, out_dim)
                matrix = torch.einsum(
                    'di,do,cliro->cdlr', in_vectors, out_vectors, core)

            matrices.append(matrix.reshape(
                *self._batch_shape,
                *data_batch_shape,
                left_rank,
                right_rank))

        return matrices

    def evaluate(self,
                 in_data: EvaluationData,
                 out_data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        """Evaluates entries at paired input and output configurations.

        Both data groups follow the discrete/embedded conventions of
        :meth:`TT.evaluate` and must share the same batch shape.
        This is equivalent to evaluating fused matrix cores as a tensor-vector
        decomposition on local tensor products of input and output vectors,
        without materializing those products.
        """
        self._ensure_valid()
        in_data_by_site, in_discrete, data_batch_shape = self._normalize_data(
            in_data, self._in_dim, self._same_in_dim, n_batches)
        out_data_by_site, out_discrete, out_batch_shape = self._normalize_data(
            out_data, self._out_dim, self._same_out_dim, n_batches)
        if out_batch_shape != data_batch_shape:
            raise ValueError(
                'Input and output data should have the same batch shape')

        matrices = self._entry_matrices(
            in_data_by_site,
            out_data_by_site,
            in_discrete,
            out_discrete,
            data_batch_shape)
        return self._contract_local_matrices(matrices)

    def apply(self,
              data: EvaluationData,
              n_batches: int = 1) -> TensorFormat1D:
        """Applies the matrix to product inputs and returns a 1D result."""
        self._ensure_valid()
        if isinstance(data, TensorFormat1D):
            from tensorkrowch.formats.operations import apply

            return apply(self, data)
        site_data, discrete, data_batch_shape = self._normalize_data(
            data, self._in_dim, self._same_in_dim, n_batches)
        core_batch_size = int(torch.Size(self._batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        output_cores = []

        for core, site_value in zip(self._operator_cores(), site_data):
            left_rank, in_dim, right_rank, out_dim = core.shape[-4:]
            core = core.reshape(
                core_batch_size, left_rank, in_dim, right_rank, out_dim)
            if discrete:
                indices = site_value.reshape(data_batch_size).to(torch.long)
                output_core = core.index_select(2, indices)
                output_core = output_core.permute(0, 2, 1, 4, 3)
            else:
                vectors = site_value.reshape(data_batch_size, in_dim)
                output_core = torch.einsum(
                    'di,cliro->cdlor', vectors, core)
            output_cores.append(output_core.reshape(
                *self._batch_shape,
                *data_batch_shape,
                left_rank,
                out_dim,
                right_rank))

        # NOTE: A future `apply_tt` could apply the matrix to a general TT
        return self._build_applied_decomposition(
            output_cores, self.n_batches + n_batches)
