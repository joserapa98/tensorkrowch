"""
This script contains:

    Class for tensor decomposition results:
        * TensorDecomposition:
            + TensorDecomposition1D:
                - _VectorDecomposition1D:
                    · TTDecomposition
                    · TRDecomposition
                - _MatrixDecomposition1D:
                    · TTMDecomposition
                    · TRMDecomposition
                - _QuantizedTuckerDecomposition:
                    · QTTTuckerDecomposition
                    · QTRTuckerDecomposition
            + TensorDecomposition2D:
                - PEPSDecomposition
                - PEPODecomposition
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Sequence,
                    Tuple, Type, Union)

import torch
import torch.nn.functional as nnf

from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord)


StateInput = Union[torch.Tensor, Sequence[torch.Tensor]]


class TensorDecomposition(ABC):
    """Topology-neutral interface for lightweight decomposition results."""

    metrics: DecompositionMetrics
    metadata: Dict[str, Any]

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device used by the decomposition tensors."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Data type used by the decomposition tensors."""

    @abstractmethod
    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           copy: bool = False) -> 'TensorDecomposition':
        """Returns the decomposition on another device or dtype."""

    @abstractmethod
    def cpu(self) -> 'TensorDecomposition':
        """Returns the decomposition on CPU."""

    @abstractmethod
    def as_info(self) -> Dict[str, Any]:
        """Returns structured decomposition information."""


@dataclass
class TensorDecomposition1D(TensorDecomposition):
    """Base class for lightweight tensor decomposition results.

    The object stores raw cores, derived rank, structured metrics and small
    metadata only. It does not build a TensorKrowch graph or retain the source
    function used during fitting.
    """

    cores: Sequence[torch.Tensor]  # Raw tensors forming the decomposition
    # Structured diagnostics collected during the fit
    metrics: DecompositionMetrics = field(default_factory=DecompositionMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Algorithm metadata
    n_batches: int = 0  # Number of leading batch dimensions in every core
    rank: List[int] = field(init=False)  # Rank inferred from adjacent cores
    _batch_shape: Tuple[int, ...] = field(init=False, repr=False)  # Shared batch shape
    _in_dim: Tuple[int, ...] = field(init=False, repr=False)  # Input dimensions
    _same_in_dim: bool = field(init=False, repr=False)  # Uniform input dims
    # Optional output dimensions represented by dedicated sites
    _out_dim: Optional[Tuple[int, ...]] = field(init=False, repr=False)

    _family: ClassVar[str] = 'tensor'
    _topology: ClassVar[str] = 'tensor'
    _integer_dtypes: ClassVar[Tuple[torch.dtype, ...]] = (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    )

    def __post_init__(self) -> None:
        if isinstance(self.n_batches, bool) or \
                not isinstance(self.n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if self.n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')
        if not isinstance(self.metrics, DecompositionMetrics):
            raise TypeError('`metrics` should be DecompositionMetrics type')
        if not isinstance(self.metadata, dict):
            raise TypeError('`metadata` should be dict type')

        if isinstance(self.cores, torch.Tensor):
            raise TypeError('`cores` should be a sequence of torch.Tensor objects')
        try:
            self.cores = list(self.cores)
        except TypeError as exc:
            raise TypeError(
                '`cores` should be a sequence of torch.Tensor objects') from exc
        self.metadata = dict(self.metadata)
        if not self.cores:
            raise ValueError('`cores` should contain at least one tensor')
        if not all(isinstance(core, torch.Tensor) for core in self.cores):
            raise TypeError('`cores` should contain torch.Tensor objects')

        device = self.cores[0].device
        dtype = self.cores[0].dtype
        for core in self.cores:
            if any(dim < 1 for dim in core.shape):
                raise ValueError('Core dimensions should be positive')
            if core.device != device:
                raise ValueError('All cores should be on the same device')
            if core.dtype != dtype:
                raise ValueError('All cores should have the same dtype')

        rank, batch_shape, in_dim, out_dim = self._validate_cores()
        self.rank = rank
        self._batch_shape = batch_shape
        self._in_dim = in_dim
        self._same_in_dim = all(dim == in_dim[0] for dim in in_dim[1:])
        self._out_dim = out_dim

    @property
    def device(self) -> torch.device:
        """Device shared by all cores."""
        return self.cores[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Data type shared by all cores."""
        return self.cores[0].dtype

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Batch dimensions shared by the cores."""
        return self._batch_shape

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension associated with every site."""
        return self._in_dim

    @property
    def out_dim(self) -> Optional[Tuple[int, ...]]:
        """Output dimension per site, when the decomposition has one."""
        return self._out_dim

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Temporary internal compatibility alias for :attr:`in_dim`."""
        return self.in_dim

    @property
    def output_dim(self) -> Optional[Tuple[int, ...]]:
        """Temporary internal compatibility alias for :attr:`out_dim`."""
        return self.out_dim

    @property
    def topology(self) -> str:
        """Topology identifier used in serialized result information."""
        return self._topology

    @abstractmethod
    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates cores and returns rank, batch, input and output dims."""

    @abstractmethod
    def _standard_cores(self) -> List[torch.Tensor]:
        """Returns cores with shape ``(*batch, left, input, right)``."""

    @abstractmethod
    def contract_dense(self) -> torch.Tensor:
        """Contracts all cores into a dense tensor."""

    def _normalize_inputs(
            self,
            inputs: StateInput,
            dimensions: Sequence[int],
            same_dim: bool,
            n_batches: int
            ) -> Tuple[List[torch.Tensor], bool, Tuple[int, ...]]:
        """Normalizes discrete indices or embedded vectors by site."""
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        n_sites = len(dimensions)
        if isinstance(inputs, torch.Tensor):
            if inputs.ndim == (n_batches + 1):
                if inputs.shape[-1] != n_sites:
                    raise ValueError(
                        'The last dimension of discrete inputs should equal '
                        'the number of sites')
                if inputs.dtype not in self._integer_dtypes:
                    raise TypeError(
                        'Discrete inputs should have an integer dtype')
                site_inputs = list(inputs.to(device=self.device).unbind(-1))
                discrete = True
            elif inputs.ndim == (n_batches + 2):
                if not same_dim:
                    raise ValueError(
                        'Embedded inputs should be provided as a sequence when '
                        'site dimensions differ')
                if inputs.shape[-2:] != (n_sites, dimensions[0]):
                    raise ValueError(
                        'Embedded inputs should end in `(n_sites, site_dim)`')
                site_inputs = list(
                    inputs.to(device=self.device, dtype=self.dtype).unbind(-2))
                discrete = False
            else:
                raise ValueError(
                    '`inputs` has an incompatible number of batch dimensions')
        else:
            try:
                site_inputs = list(inputs)
            except TypeError as exc:
                raise TypeError(
                    '`inputs` should be a tensor or a sequence of tensors') \
                    from exc
            if len(site_inputs) != n_sites:
                raise ValueError(
                    '`inputs` should contain one tensor per site')
            if not all(isinstance(item, torch.Tensor)
                       for item in site_inputs):
                raise TypeError('Every site input should be a torch.Tensor')

            if all(item.ndim == n_batches for item in site_inputs):
                if not all(item.dtype in self._integer_dtypes
                           for item in site_inputs):
                    raise TypeError(
                        'Discrete inputs should have an integer dtype')
                discrete = True
            elif all(item.ndim == (n_batches + 1)
                     for item in site_inputs):
                for item, site_dim in zip(site_inputs, dimensions):
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
            site_inputs = [
                item.to(device=self.device, dtype=target_dtype)
                for item in site_inputs
            ]

        batch_shape = tuple(site_inputs[0].shape[:n_batches])
        if any(tuple(item.shape[:n_batches]) != batch_shape
               for item in site_inputs[1:]):
            raise ValueError('All site inputs should have the same batch shape')

        if discrete:
            for indices, site_dim in zip(site_inputs, dimensions):
                if torch.any(indices < 0) or torch.any(indices >= site_dim):
                    raise ValueError(
                        'Discrete indices should lie in each site dimension')

        return site_inputs, discrete, batch_shape

    @staticmethod
    def _contract_open_chain(
            matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts an open chain of matrices along adjacent ranks."""
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result

    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           copy: bool = False) -> 'TensorDecomposition1D':
        """Returns this result on another device or dtype.

        ``copy`` has the same meaning as in :meth:`torch.Tensor.to`. If no core
        needs conversion and ``copy=False``, this result is returned unchanged.
        """
        if (dtype is not None) and (not isinstance(dtype, torch.dtype)):
            raise TypeError('`dtype` should be torch.dtype type')
        if not isinstance(copy, bool):
            raise TypeError('`copy` should be bool type')

        target_device = None if device is None else torch.device(device)
        cores = [
            core.to(device=target_device, dtype=dtype, copy=copy)
            for core in self.cores
        ]
        if not copy and all(new is old
                            for new, old in zip(cores, self.cores)):
            return self
        return replace(self, cores=cores)

    def cpu(self) -> 'TensorDecomposition1D':
        """Returns this result with all cores stored on CPU."""
        return self.to(device='cpu')

    def _check_overlap_compatibility(
            self, other: 'TensorDecomposition1D') -> None:
        """Validates topology, shapes and runtime for an overlap."""
        if not isinstance(other, TensorDecomposition1D):
            raise TypeError('`other` should be TensorDecomposition1D type')
        if self._family != other._family:
            raise ValueError('The decomposition families are incompatible')
        if len(self.cores) != len(other.cores):
            raise ValueError('Decompositions should have the same number of sites')
        if self.in_dim != other.in_dim:
            raise ValueError('Decompositions should have matching input dimensions')
        if self.out_dim != other.out_dim:
            raise ValueError('Decompositions should have matching output dimensions')
        if self.batch_shape != other.batch_shape:
            raise ValueError('Decompositions should have matching batch shapes')
        if self.device != other.device:
            raise ValueError('Decompositions should be on the same device')

    def _log_overlap(
            self, other: 'TensorDecomposition1D') -> Tuple[torch.Tensor,
                                                           torch.Tensor]:
        """Returns overlap phase and log-magnitude using scaled environments."""
        self._check_overlap_compatibility(other)
        dtype = torch.promote_types(self.dtype, other.dtype)
        self_cores = self._standard_cores()
        other_cores = other._standard_cores()

        self_eye = torch.eye(
            self_cores[0].shape[-3], device=self.device, dtype=dtype)
        other_eye = torch.eye(
            other_cores[0].shape[-3], device=self.device, dtype=dtype)
        environment = torch.einsum('ai,bj->abij', self_eye, other_eye)
        if self.n_batches:
            environment = environment.reshape(
                *((1,) * self.n_batches), *environment.shape)
            environment = environment.expand(
                *self.batch_shape, *environment.shape[self.n_batches:])

        real_dtype = torch.empty((), dtype=dtype).real.dtype
        log_scale = torch.zeros(
            self.batch_shape, device=self.device, dtype=real_dtype)

        for self_core, other_core in zip(self_cores, other_cores):
            self_core = self_core.to(dtype=dtype)
            other_core = other_core.to(dtype=dtype)
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
        _, log_squared_norm = self._log_overlap(self)
        return torch.exp(log_squared_norm / 2)

    def normalized_overlap(
            self, other: 'TensorDecomposition1D') -> torch.Tensor:
        """Returns ``<self, other> / (||self|| ||other||)`` with its phase."""
        phase, log_overlap = self._log_overlap(other)
        _, log_self = self._log_overlap(self)
        _, log_other = other._log_overlap(other)

        if torch.any(torch.isneginf(log_self)) or \
                torch.any(torch.isneginf(log_other)):
            raise ValueError(
                'Normalized overlap is undefined for a zero-norm decomposition')

        log_denominator = (log_self + log_other) / 2
        return phase * torch.exp(log_overlap - log_denominator)

    def fidelity(self, other: 'TensorDecomposition1D') -> torch.Tensor:
        """Returns ``abs(normalized_overlap(other)) ** 2``."""
        return self.normalized_overlap(other).abs().square()

    def as_info(self) -> Dict[str, Any]:
        """Returns rank, dimensions, metrics and metadata for functional APIs."""
        return {
            'topology': self.topology,
            'rank': list(self.rank),
            'in_dim': list(self.in_dim),
            'out_dim': (None if self.out_dim is None else list(self.out_dim)),
            # Remove these compatibility keys after later decomposition phases
            # have migrated to the canonical names.
            'input_dim': list(self.in_dim),
            'output_dim': (None if self.out_dim is None
                           else list(self.out_dim)),
            'n_batches': self.n_batches,
            'metrics': self.metrics.as_info(),
            'metadata': dict(self.metadata),
        }


@dataclass
class _VectorDecomposition1D(TensorDecomposition1D):
    """Common input evaluation for 1D tensor-vector results."""

    _family: ClassVar[str] = 'state'

    def __call__(self,
                 inputs: StateInput,
                 n_batches: int = 1) -> torch.Tensor:
        """Calls :meth:`evaluate`."""
        return self.evaluate(inputs, n_batches=n_batches)

    def evaluate(self,
                 inputs: StateInput,
                 n_batches: int = 1) -> torch.Tensor:
        """Evaluates the decomposition on indices or embedded input vectors.

        A tensor of discrete indices has shape ``(*batch, n_sites)``. A tensor
        of embedded inputs requires uniform input dimensions and has shape
        ``(*batch, n_sites, in_dim)``. A sequence may instead contain one
        index tensor of shape ``(*batch,)`` or one embedded tensor of shape
        ``(*batch, in_dim[site])`` per site.

        ``n_batches`` describes the leading batch dimensions of ``inputs``;
        :attr:`n_batches` describes independent batch dimensions stored in the
        cores. Both groups are preserved in the returned tensor, with core
        batches followed by input batches.
        """
        site_inputs, discrete, data_batch_shape = self._normalize_inputs(
            inputs, self.in_dim, self._same_in_dim, n_batches)
        matrices = self._local_matrices(
            site_inputs, discrete, data_batch_shape)
        return self._contract_local_matrices(matrices)

    def _local_matrices(
            self,
            site_inputs: Sequence[torch.Tensor],
            discrete: bool,
            data_batch_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Contracts TT/TR cores with one input at each site."""
        core_batch_size = int(torch.Size(self.batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        matrices = []

        for core, site_input in zip(self._standard_cores(), site_inputs):
            left_rank, site_in_dim, right_rank = core.shape[-3:]
            core = core.reshape(
                core_batch_size, left_rank, site_in_dim, right_rank)

            if discrete:
                indices = site_input.reshape(data_batch_size).to(torch.long)
                matrix = core[:, :, indices, :].permute(0, 2, 1, 3)
            else:
                vectors = site_input.reshape(data_batch_size, site_in_dim)
                matrix = torch.einsum('dp,clpr->cdlr', vectors, core)

            matrices.append(matrix.reshape(
                *self.batch_shape,
                *data_batch_shape,
                left_rank,
                right_rank))

        return matrices

    def error(self,
              function: Callable[..., torch.Tensor],
              samples: torch.Tensor,
              inputs: Optional[StateInput] = None,
              n_batches: int = 1,
              **kwargs: Any) -> ErrorRecord:
        """Measures errors on samples, optionally using embedded inputs."""
        if not callable(function):
            raise TypeError('`function` should be callable')
        if not isinstance(samples, torch.Tensor):
            raise TypeError('`samples` should be torch.Tensor type')
        if samples.ndim != (n_batches + 1):
            raise ValueError(
                '`samples` has an incompatible number of batch dimensions')

        samples = samples.to(device=self.device)
        approximation = self.evaluate(
            samples if inputs is None else inputs,
            n_batches=n_batches)
        target = function(samples, **kwargs)
        if not isinstance(target, torch.Tensor):
            raise TypeError('`function` should return a torch.Tensor')
        target = target.to(device=approximation.device,
                           dtype=approximation.dtype)
        target = target.reshape(*samples.shape[:n_batches])
        if self.n_batches:
            target = target.reshape(
                *((1,) * self.n_batches), *target.shape)
            target = target.expand(*self.batch_shape, *target.shape[self.n_batches:])

        absolute = torch.linalg.vector_norm(approximation - target)
        denominator = torch.linalg.vector_norm(target)
        if denominator > 0:
            relative = absolute / denominator
        elif absolute == 0:
            relative = torch.zeros_like(absolute)
        else:
            relative = torch.full_like(absolute, torch.inf)

        size = int(torch.Size(samples.shape[:n_batches]).numel())
        return ErrorRecord(
            kind='samples',
            absolute=absolute,
            relative=relative,
            size=size,
            denominator=denominator)

    @abstractmethod
    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts input-selected local matrices along the ranks."""


@dataclass
class TTDecomposition(_VectorDecomposition1D):
    """Lightweight tensor train decomposition with open boundaries.

    This result stores the TT cores, ranks, metrics and metadata without
    constructing a TensorKrowch graph. Methods such as :meth:`contract_dense`
    and :meth:`evaluate` operate directly on the core tensors with PyTorch.
    Consequently, creating or inspecting a decomposition has less overhead
    than creating an MPS model.

    For graph contractions, training or the rest of the model API, an
    :class:`~tensorkrowch.models.MPS` can be initialized directly from the
    cores:

    >>> tensor = torch.randn(2, 3, 4)
    >>> result = tk.decompositions.TTSVD(tensor).fit(rank=2)
    >>> mps = tk.models.MPS(tensors=result.cores)
    >>> mps.boundary
    'obc'

    The model infers open boundaries and dimensions from the core shapes.
    Metrics and metadata remain attached to ``result`` and are not transferred
    to the model. Pass ``parameterized=False`` to :class:`~tensorkrowch.models.MPS`
    when trainable parameter nodes are not required. Clone the cores before
    construction if independent tensor storage is required.

    If :attr:`n_batches` is positive, the batch dimensions belong to the cores
    themselves. Such a result should instead initialize an
    :class:`~tensorkrowch.models.MPSData`:

    >>> batched = tk.decompositions.TTSVD(
    ...     torch.randn(8, 2, 3, 4), n_batches=1).fit(rank=2)
    >>> mps_data = tk.models.MPSData(tensors=batched.cores,
    ...                              n_batches=batched.n_batches)

    The latter form applies only to a result that was created with batch
    dimensions; ordinary non-batched decompositions use MPS as above.
    """

    _topology: ClassVar[str] = 'tt'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        n_sites = len(self.cores)
        batch_shape = tuple(self.cores[0].shape[:self.n_batches])
        in_dim = []

        if n_sites == 1:
            if self.cores[0].ndim != (self.n_batches + 1):
                raise ValueError(
                    'A one-site TT core should have one input dimension')
            in_dim.append(self.cores[0].shape[-1])
            return [], batch_shape, tuple(in_dim), None

        rank = []
        for site, core in enumerate(self.cores):
            if tuple(core.shape[:self.n_batches]) != batch_shape:
                raise ValueError('All TT cores should have the same batch shape')

            if site == 0:
                if core.ndim != (self.n_batches + 2):
                    raise ValueError(
                        'The first TT core should have input and right rank '
                        'dimensions')
                in_dim.append(core.shape[-2])
                rank.append(core.shape[-1])
            elif site == (n_sites - 1):
                if core.ndim != (self.n_batches + 2):
                    raise ValueError(
                        'The last TT core should have left rank and input '
                        'dimensions')
                if core.shape[-2] != rank[-1]:
                    raise ValueError('Adjacent TT ranks should match')
                in_dim.append(core.shape[-1])
            else:
                if core.ndim != (self.n_batches + 3):
                    raise ValueError(
                        'Interior TT cores should have left, input and '
                        'right dimensions')
                if core.shape[-3] != rank[-1]:
                    raise ValueError('Adjacent TT ranks should match')
                in_dim.append(core.shape[-2])
                rank.append(core.shape[-1])

        return rank, batch_shape, tuple(in_dim), None

    def _standard_cores(self) -> List[torch.Tensor]:
        if len(self.cores) == 1:
            return [self.cores[0].unsqueeze(self.n_batches).unsqueeze(-1)]

        cores = [self.cores[0].unsqueeze(self.n_batches)]
        cores.extend(self.cores[1:-1])
        cores.append(self.cores[-1].unsqueeze(-1))
        return cores

    def contract_dense(self) -> torch.Tensor:
        """Contracts the TT into a dense tensor without TensorKrowch models."""
        if len(self.cores) == 1:
            return self.cores[0]

        result = self.cores[0]
        for site, core in enumerate(self.cores[1:], 1):
            previous_in_dim = result.shape[self.n_batches:-1]
            previous_rank = result.shape[-1]
            result = result.reshape(*self.batch_shape, -1, previous_rank)

            if site < (len(self.cores) - 1):
                site_in_dim = core.shape[-2]
                rank = core.shape[-1]
                core = core.reshape(*self.batch_shape, previous_rank, -1)
                result = (result @ core).reshape(
                    *self.batch_shape,
                    *previous_in_dim,
                    site_in_dim,
                    rank)
            else:
                site_in_dim = core.shape[-1]
                core = core.reshape(*self.batch_shape, previous_rank, -1)
                result = (result @ core).reshape(
                    *self.batch_shape,
                    *previous_in_dim,
                    site_in_dim)
        return result

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        result = self._contract_open_chain(matrices)
        return result.squeeze(-1).squeeze(-1)


@dataclass
class TRDecomposition(_VectorDecomposition1D):
    """Lightweight tensor ring decomposition with cyclic boundaries.

    This result stores raw TR cores, ranks, metrics and metadata, but it is not
    itself a TensorKrowch graph. :meth:`contract_dense` and :meth:`evaluate`
    close the cyclic trace directly with PyTorch operations.

    TensorKrowch represents a TR as an :class:`~tensorkrowch.models.MPS` with
    periodic boundaries. The model can be initialized directly from the cores;
    their shapes identify the cyclic topology:

    >>> cores = [torch.randn(2, 3, 4), torch.randn(4, 5, 2)]
    >>> result = tk.decompositions.TRDecomposition(cores)
    >>> mps = tk.models.MPS(tensors=result.cores)
    >>> mps.boundary
    'pbc'

    Metrics and metadata remain attached to ``result`` and are not transferred
    to the model. Pass ``parameterized=False`` when trainable parameter nodes
    are not required. Clone the cores before construction if independent
    tensor storage is required.

    Batched TR cores should initialize
    :class:`~tensorkrowch.models.MPSData` instead:

    >>> batched_cores = [torch.randn(8, 2, 3, 4),
    ...                  torch.randn(8, 4, 5, 2)]
    >>> batched = tk.decompositions.TRDecomposition(
    ...     batched_cores, n_batches=1)
    >>> mps_data = tk.models.MPSData(tensors=batched.cores,
    ...                              n_batches=batched.n_batches)

    The MPSData form applies only when :attr:`n_batches` is positive.
    """

    _topology: ClassVar[str] = 'tr'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        batch_shape = tuple(self.cores[0].shape[:self.n_batches])
        rank = []
        in_dim = []

        for site, core in enumerate(self.cores):
            if core.ndim != (self.n_batches + 3):
                raise ValueError(
                    'TR cores should have left rank, input and right rank '
                    'dimensions')
            if tuple(core.shape[:self.n_batches]) != batch_shape:
                raise ValueError('All TR cores should have the same batch shape')
            if site and (core.shape[-3] != rank[-1]):
                raise ValueError('Adjacent TR ranks should match')
            in_dim.append(core.shape[-2])
            rank.append(core.shape[-1])

        if self.cores[-1].shape[-1] != self.cores[0].shape[-3]:
            raise ValueError('The last and first cyclic TR ranks should match')
        return rank, batch_shape, tuple(in_dim), None

    def _standard_cores(self) -> List[torch.Tensor]:
        return list(self.cores)

    def contract_dense(self) -> torch.Tensor:
        """Contracts the TR into a dense tensor and closes the cyclic trace."""
        result = self.cores[0]
        in_dim = [self.cores[0].shape[-2]]
        for core in self.cores[1:]:
            initial_rank = result.shape[self.n_batches]
            previous_rank = result.shape[-1]
            result = result.reshape(
                *self.batch_shape, initial_rank, -1, previous_rank)
            result = torch.einsum(
                '...apr,...rqb->...apqb', result, core)
            in_dim.append(core.shape[-2])
            result = result.reshape(
                *self.batch_shape,
                initial_rank,
                *in_dim,
                core.shape[-1])

        return result.diagonal(dim1=self.n_batches, dim2=-1).sum(-1)

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        result = self._contract_open_chain(matrices)
        return result.diagonal(dim1=-2, dim2=-1).sum(-1)


@dataclass
class _MatrixDecomposition1D(TensorDecomposition1D):
    """Common evaluation and product-state application for 1D matrices."""

    _family: ClassVar[str] = 'matrix'
    _same_out_dim: bool = field(init=False, repr=False)  # Uniform output dims

    def __post_init__(self) -> None:
        super().__post_init__()
        self._same_out_dim = all(
            dim == self.out_dim[0] for dim in self.out_dim[1:])

    def __call__(
            self,
            inputs: StateInput,
            out_samples: Optional[StateInput] = None,
            n_batches: int = 1
            ) -> Union[torch.Tensor, TensorDecomposition1D]:
        """Applies the matrix or evaluates it when outputs are provided."""
        if out_samples is None:
            return self.apply(inputs, n_batches=n_batches)
        return self.evaluate(inputs, out_samples, n_batches=n_batches)

    def evaluate(self,
                 in_samples: StateInput,
                 out_samples: StateInput,
                 n_batches: int = 1) -> torch.Tensor:
        """Evaluates entries at paired input and output configurations.

        Both sample groups follow the discrete/embedded conventions of
        :meth:`TTDecomposition.evaluate` and must share the same batch shape.
        This is equivalent to evaluating fused matrix cores as a tensor-vector
        decomposition on local tensor products of input and output vectors,
        without materializing those products.
        """
        in_inputs, in_discrete, data_batch_shape = self._normalize_inputs(
            in_samples, self.in_dim, self._same_in_dim, n_batches)
        out_inputs, out_discrete, out_batch_shape = self._normalize_inputs(
            out_samples, self.out_dim, self._same_out_dim, n_batches)
        if out_batch_shape != data_batch_shape:
            raise ValueError(
                'Input and output samples should have the same batch shape')

        matrices = self._entry_matrices(
            in_inputs,
            out_inputs,
            in_discrete,
            out_discrete,
            data_batch_shape)
        return self._contract_local_matrices(matrices)

    def _entry_matrices(
            self,
            in_inputs: Sequence[torch.Tensor],
            out_inputs: Sequence[torch.Tensor],
            in_discrete: bool,
            out_discrete: bool,
            data_batch_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Builds local matrices for paired matrix-entry evaluation."""
        core_batch_size = int(torch.Size(self.batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        matrices = []

        for core, in_input, out_input in zip(
                self._operator_cores(), in_inputs, out_inputs):
            left_rank, in_dim, right_rank, out_dim = core.shape[-4:]
            core = core.reshape(
                core_batch_size, left_rank, in_dim, right_rank, out_dim)

            if in_discrete and out_discrete:
                in_indices = in_input.reshape(data_batch_size).to(torch.long)
                out_indices = out_input.reshape(data_batch_size).to(torch.long)
                fused = in_indices * out_dim + out_indices
                matrix = core.permute(0, 1, 3, 2, 4).reshape(
                    core_batch_size, left_rank, right_rank, in_dim * out_dim)
                matrix = matrix[..., fused].permute(0, 3, 1, 2)
            elif in_discrete:
                indices = in_input.reshape(data_batch_size).to(torch.long)
                vectors = out_input.reshape(data_batch_size, out_dim)
                selected = core.index_select(2, indices)
                matrix = torch.einsum('cldro,do->cdlr', selected, vectors)
            elif out_discrete:
                indices = out_input.reshape(data_batch_size).to(torch.long)
                vectors = in_input.reshape(data_batch_size, in_dim)
                selected = core.index_select(4, indices)
                matrix = torch.einsum('clird,di->cdlr', selected, vectors)
            else:
                in_vectors = in_input.reshape(data_batch_size, in_dim)
                out_vectors = out_input.reshape(data_batch_size, out_dim)
                matrix = torch.einsum(
                    'di,do,cliro->cdlr', in_vectors, out_vectors, core)

            matrices.append(matrix.reshape(
                *self.batch_shape,
                *data_batch_shape,
                left_rank,
                right_rank))

        return matrices

    def apply(self,
              inputs: StateInput,
              n_batches: int = 1) -> TensorDecomposition1D:
        """Applies the matrix to product inputs and returns a 1D result."""
        site_inputs, discrete, data_batch_shape = self._normalize_inputs(
            inputs, self.in_dim, self._same_in_dim, n_batches)
        core_batch_size = int(torch.Size(self.batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        output_cores = []

        for core, site_input in zip(self._operator_cores(), site_inputs):
            left_rank, in_dim, right_rank, out_dim = core.shape[-4:]
            core = core.reshape(
                core_batch_size, left_rank, in_dim, right_rank, out_dim)
            if discrete:
                indices = site_input.reshape(data_batch_size).to(torch.long)
                output_core = core.index_select(2, indices)
                output_core = output_core.permute(0, 2, 1, 4, 3)
            else:
                vectors = site_input.reshape(data_batch_size, in_dim)
                output_core = torch.einsum(
                    'di,cliro->cdlor', vectors, core)
            output_cores.append(output_core.reshape(
                *self.batch_shape,
                *data_batch_shape,
                left_rank,
                out_dim,
                right_rank))

        # NOTE: A future `apply_tt` could apply the matrix to a general TT
        return self._build_applied_decomposition(
            output_cores, self.n_batches + n_batches)

    @abstractmethod
    def _operator_cores(self) -> List[torch.Tensor]:
        """Returns cores with left, input, right and output axes."""

    @abstractmethod
    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts entry-selected matrices with the topology closure."""

    @abstractmethod
    def _build_applied_decomposition(
            self,
            cores: List[torch.Tensor],
            n_batches: int) -> TensorDecomposition1D:
        """Builds the vector-like result produced by :meth:`apply`."""


@dataclass
class TTMDecomposition(_MatrixDecomposition1D):
    """Lightweight tensor train matrix decomposition with open boundaries.

    This result stores TTM cores, ranks, input/output dimensions, metrics and
    metadata without constructing a TensorKrowch graph. Dense contraction and
    application to product inputs operate directly on its PyTorch tensors.

    TensorKrowch models call this structure an
    :class:`~tensorkrowch.models.MPO`. A model can be initialized directly from
    the TTM cores when graph contractions or training are required:

    >>> tensor = torch.randn(2, 3, 4, 5)
    >>> result = tk.decompositions.TTMSVD(tensor).fit(rank=2)
    >>> mpo = tk.models.MPO(tensors=result.cores)
    >>> mpo.boundary
    'obc'

    The model infers dimensions and open boundaries from the core shapes.
    Metrics and metadata remain attached to ``result`` and are not transferred
    to the model. Pass ``parameterized=False`` when trainable parameter nodes
    are not required, and clone the cores first if independent tensor storage
    is required. TTM decomposition batches are currently unsupported.
    """

    _topology: ClassVar[str] = 'ttm'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        if self.n_batches:
            raise ValueError('TTM decomposition batches are not supported')

        n_sites = len(self.cores)
        in_dim = []
        out_dim = []
        if n_sites == 1:
            if self.cores[0].ndim != 2:
                raise ValueError(
                    'A one-site TTM core should have input and output dimensions')
            in_dim.append(self.cores[0].shape[0])
            out_dim.append(self.cores[0].shape[1])
            return [], (), tuple(in_dim), tuple(out_dim)

        rank = []
        for site, core in enumerate(self.cores):
            if site == 0:
                if core.ndim != 3:
                    raise ValueError(
                        'The first TTM core should have input, right rank and '
                        'output dimensions')
                in_dim.append(core.shape[0])
                out_dim.append(core.shape[2])
                rank.append(core.shape[1])
            elif site == (n_sites - 1):
                if core.ndim != 3:
                    raise ValueError(
                        'The last TTM core should have left rank, input and '
                        'output dimensions')
                if core.shape[0] != rank[-1]:
                    raise ValueError('Adjacent TTM ranks should match')
                in_dim.append(core.shape[1])
                out_dim.append(core.shape[2])
            else:
                if core.ndim != 4:
                    raise ValueError(
                        'Interior TTM cores should have left, input, right and '
                        'output dimensions')
                if core.shape[0] != rank[-1]:
                    raise ValueError('Adjacent TTM ranks should match')
                in_dim.append(core.shape[1])
                out_dim.append(core.shape[3])
                rank.append(core.shape[2])

        return rank, (), tuple(in_dim), tuple(out_dim)

    def _standard_cores(self) -> List[torch.Tensor]:
        if len(self.cores) == 1:
            core = self.cores[0]
            return [core.reshape(1, core.numel(), 1)]

        cores = []
        first = self.cores[0].permute(0, 2, 1)
        cores.append(first.reshape(1, first.shape[0] * first.shape[1],
                                   first.shape[2]))
        for core in self.cores[1:-1]:
            core = core.permute(0, 1, 3, 2)
            cores.append(core.reshape(core.shape[0],
                                      core.shape[1] * core.shape[2],
                                      core.shape[3]))
        last = self.cores[-1]
        cores.append(last.reshape(last.shape[0], -1, 1))
        return cores

    def contract_dense(self) -> torch.Tensor:
        """Contracts the TTM into interleaved input/output dimensions."""
        if len(self.cores) == 1:
            return self.cores[0]

        result = self.cores[0].permute(0, 2, 1)
        for core in self.cores[1:-1]:
            core = core.permute(0, 1, 3, 2)
            result = torch.tensordot(result, core, dims=([-1], [0]))
        return torch.tensordot(result, self.cores[-1], dims=([-1], [0]))

    def _operator_cores(self) -> List[torch.Tensor]:
        """Returns cores with separate left, input, right and output axes."""
        if len(self.cores) == 1:
            return [self.cores[0].unsqueeze(0).unsqueeze(2)]

        cores = [self.cores[0].unsqueeze(0)]
        cores.extend(self.cores[1:-1])
        cores.append(self.cores[-1].unsqueeze(2))
        return cores

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        result = self._contract_open_chain(matrices)
        return result.squeeze(-1).squeeze(-1)

    def _build_applied_decomposition(
            self,
            cores: List[torch.Tensor],
            n_batches: int) -> TTDecomposition:
        batch_shape = cores[0].shape[:n_batches]
        if len(cores) == 1:
            cores[0] = cores[0].squeeze(-1).squeeze(-2)
        else:
            cores[0] = cores[0].squeeze(len(batch_shape))
            cores[-1] = cores[-1].squeeze(-1)

        return TTDecomposition(
            cores=cores,
            n_batches=n_batches,
            metadata={'operation': 'ttm_apply'})

@dataclass
class TRMDecomposition(_MatrixDecomposition1D):
    """Placeholder for tensor ring matrix decomposition results."""

    _topology: ClassVar[str] = 'trm'


class _QuantizedTuckerDecomposition(TensorDecomposition1D):
    """Common two-level contraction for quantized Tucker results."""

    _upper_type: ClassVar[Type[TensorDecomposition1D]]
    _family: ClassVar[str] = 'quantized_tucker'

    def __init__(
            self,
            upper: TensorDecomposition1D,
            factors: Sequence[TTDecomposition],
            layout,
            coordinate_map,
            domain=None,
            *,
            variable_positions: Optional[Sequence[int]] = None,
            computational_grid: str = 'endpoints',
            out_of_domain: str = 'error',
            metrics: Optional[DecompositionMetrics] = None,
            metadata: Optional[Dict[str, Any]] = None) -> None:
        from tensorkrowch.decompositions.sketching.quantization import (
            CoordinateMap,
            QuantizedLayout,
        )

        if not isinstance(upper, self._upper_type):
            raise TypeError(
                f'`upper` should be {self._upper_type.__name__} type')
        if not isinstance(layout, QuantizedLayout):
            raise TypeError('`layout` should be QuantizedLayout type')
        if not isinstance(coordinate_map, CoordinateMap):
            raise TypeError('`coordinate_map` should implement CoordinateMap')
        if computational_grid not in ('endpoints', 'cell_centers'):
            raise ValueError(
                "`computational_grid` should be 'endpoints' or "
                "'cell_centers'")
        if out_of_domain not in ('error', 'clip'):
            raise ValueError("`out_of_domain` should be 'error' or 'clip'")
        if isinstance(factors, torch.Tensor):
            raise TypeError(
                '`factors` should be a sequence of TTDecomposition objects')
        try:
            factors = tuple(factors)
        except TypeError as exc:
            raise TypeError(
                '`factors` should be a sequence of TTDecomposition objects') \
                from exc
        if len(factors) != layout.n_variables or not all(
                isinstance(factor, TTDecomposition) for factor in factors):
            raise ValueError(
                '`factors` should contain one TTDecomposition per variable')

        if variable_positions is None:
            if len(upper.cores) != layout.n_variables:
                raise ValueError(
                    '`variable_positions` is required when upper output sites '
                    'are present')
            variable_positions = tuple(range(layout.n_variables))
        else:
            try:
                variable_positions = tuple(variable_positions)
            except TypeError as exc:
                raise TypeError(
                    '`variable_positions` should be a sequence of integers') \
                    from exc
        if len(variable_positions) != layout.n_variables or any(
                isinstance(position, bool) or not isinstance(position, int)
                for position in variable_positions):
            raise ValueError(
                '`variable_positions` should contain one integer per variable')
        if any(position < 0 or position >= len(upper.cores)
               for position in variable_positions):
            raise ValueError('`variable_positions` contains an invalid site')
        if any(left >= right for left, right in zip(
                variable_positions, variable_positions[1:])):
            raise ValueError(
                '`variable_positions` should be strictly increasing')

        self.factors = factors
        self.layout = layout
        self.coordinate_map = coordinate_map
        self.domain = domain
        self.variable_positions = variable_positions
        self.computational_grid = computational_grid
        self.out_of_domain = out_of_domain
        self._source_upper_metadata = dict(upper.metadata)
        super().__init__(
            cores=upper.cores,
            metrics=upper.metrics if metrics is None else metrics,
            metadata={} if metadata is None else metadata)
        self.upper = self._upper_type(
            self.cores,
            metrics=self.metrics,
            metadata=self._source_upper_metadata)
        self._hierarchical_input_dim = self._flattened_input_dim()

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Input dimensions of the explicitly flattened digit-site network."""
        return self._hierarchical_input_dim

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Tensor-output dimensions retained as open upper-network sites."""
        variable_positions = set(self.variable_positions)
        return tuple(
            dimension
            for site, dimension in enumerate(self.upper.input_dim)
            if site not in variable_positions)

    @property
    def factor_rank(self) -> Tuple[Tuple[int, ...], ...]:
        """TT ranks internal to every local quantized factor."""
        return tuple(tuple(factor.rank) for factor in self.factors)

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        upper = self._upper_type(self.cores)
        if upper.n_batches:
            raise ValueError(
                'Quantized Tucker upper decompositions cannot be batched')
        if any(factor.n_batches for factor in self.factors):
            raise ValueError('Quantized Tucker factors cannot be batched')
        for variable, (factor, position) in enumerate(zip(
                self.factors, self.variable_positions)):
            expected = (
                (self.layout.base[variable],) * self.layout.level[variable])
            if factor.input_dim[:-1] != expected:
                raise ValueError(
                    'Factor digit dimensions should match the quantized layout')
            if factor.input_dim[-1] != upper.input_dim[position]:
                raise ValueError(
                    'Factor connector dimension should match its upper site')
            if factor.device != upper.device or factor.dtype != upper.dtype:
                raise ValueError(
                    'Upper cores and factors should share device and dtype')
        return upper.rank, (), upper.input_dim, None

    def _flattened_input_dim(self) -> Tuple[int, ...]:
        """Expands each upper connector into its factor digit dimensions."""
        variable_by_position = {
            position: variable
            for variable, position in enumerate(self.variable_positions)}
        dimensions = []
        for site, dimension in enumerate(self.upper.input_dim):
            variable = variable_by_position.get(site)
            if variable is None:
                dimensions.append(dimension)
            else:
                dimensions.extend(self.factors[variable].input_dim[:-1])
        return tuple(dimensions)

    def _standard_cores(self) -> List[torch.Tensor]:
        return self.flatten()._standard_cores()

    def _physical_to_indices(self, points: torch.Tensor) -> torch.Tensor:
        """Maps physical points to grid indices without retaining the source."""
        from tensorkrowch.decompositions.sketching.quantization import (
            _unit_to_indices,
        )

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
            for site, core in enumerate(factor._standard_cores()[:-1]):
                values = nnf.one_hot(
                    variable_digits[..., site],
                    num_classes=core.shape[-2]).to(factor.dtype)
                local = torch.einsum('...p,lpr->...lr', values, core)
                state = local if state is None else state @ local
            connector = factor._standard_cores()[-1].squeeze(-1)
            vectors.append((state @ connector).squeeze(-2))
        return vectors

    def evaluate_digits(self, digits: torch.Tensor) -> torch.Tensor:
        """Evaluates scheduled digit configurations through both levels."""
        digits = self.layout._integer_tensor(digits, 'digits').to(self.device)
        if digits.ndim != 2 or digits.shape[-1] != self.layout.n_sites:
            raise ValueError(
                '`digits` should have shape (batch_size, layout.n_sites)')
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

    def flatten(self) -> TensorDecomposition1D:
        """Returns an explicit flat TT/TR over grouped local digit blocks."""
        standard = self._flat_standard_cores()
        if self._upper_type is TTDecomposition:
            if len(standard) == 1:
                cores = [standard[0].squeeze(0).squeeze(-1)]
            else:
                cores = [standard[0].squeeze(0),
                         *standard[1:-1],
                         standard[-1].squeeze(-1)]
            result_type = TTDecomposition
        else:
            cores = standard
            result_type = TRDecomposition
        metadata = dict(self.metadata)
        metadata.update({
            'algorithm': f'{self.topology}_flatten',
            'hierarchical_topology': self.topology,
            'variable_positions': tuple(self.variable_positions),
        })
        return result_type(cores, metrics=self.metrics, metadata=metadata)

    def contract_dense(self) -> torch.Tensor:
        """Contracts the explicit flattened network for small-grid oracles."""
        return self.flatten().contract_dense()

    def norm(self) -> torch.Tensor:
        return self.flatten().norm()

    def normalized_overlap(
            self, other: TensorDecomposition1D) -> torch.Tensor:
        if not isinstance(other, _QuantizedTuckerDecomposition):
            raise TypeError(
                '`other` should be a quantized Tucker decomposition')
        return self.flatten().normalized_overlap(other.flatten())

    def fidelity(self, other: TensorDecomposition1D) -> torch.Tensor:
        return self.normalized_overlap(other).abs().square()

    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           copy: bool = False) -> '_QuantizedTuckerDecomposition':
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type')
        if not isinstance(copy, bool):
            raise TypeError('`copy` should be bool type')
        upper = self.upper.to(device=device, dtype=dtype, copy=copy)
        factors = tuple(
            factor.to(device=device, dtype=dtype, copy=copy)
            for factor in self.factors)
        if not copy and upper is self.upper and all(
                new is old for new, old in zip(factors, self.factors)):
            return self
        return type(self)(
            upper,
            factors,
            self.layout,
            self.coordinate_map,
            self.domain,
            variable_positions=self.variable_positions,
            computational_grid=self.computational_grid,
            out_of_domain=self.out_of_domain,
            metrics=self.metrics,
            metadata=self.metadata)

    def cpu(self) -> '_QuantizedTuckerDecomposition':
        return self.to(device='cpu')

    def as_info(self) -> Dict[str, Any]:
        info = super().as_info()
        info.update({
            'upper_rank': list(self.upper.rank),
            'factor_rank': [list(rank) for rank in self.factor_rank],
            'grid_size': list(self.layout.grid_size),
            'variable_positions': list(self.variable_positions),
            'output_shape': list(self.output_shape),
        })
        return info


class QTTTuckerDecomposition(_QuantizedTuckerDecomposition):
    """Experimental QTT factors connected to an open upper TT.

    ``cores`` and :attr:`upper` refer only to the small upper TT over the
    connector indices. Each entry of :attr:`factors` is a local TT whose last
    input axis is the matching connector ``gamma``. Use :meth:`evaluate` to
    contract both levels at physical points, or :meth:`flatten` to construct
    an explicit ordinary :class:`TTDecomposition` over grouped digit blocks.
    """

    _upper_type = TTDecomposition
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


class QTRTuckerDecomposition(_QuantizedTuckerDecomposition):
    """Experimental QTT factors connected to a cyclic upper TR."""

    _upper_type = TRDecomposition
    _topology = 'qtr_tucker'

    def _contract_upper(self,
                        vectors: Dict[int, torch.Tensor],
                        batch_size: int) -> torch.Tensor:
        cyclic_rank = self.upper.cores[0].shape[0]
        state = torch.eye(
            cyclic_rank,
            device=self.device,
            dtype=self.dtype).expand(batch_size, -1, -1)
        for site, core in enumerate(self.upper.cores):
            if site in vectors:
                local = torch.einsum(
                    'bp,lpr->blr', vectors[site], core)
                state = torch.einsum(
                    'ba...l,blr->ba...r', state, local)
            else:
                state = torch.einsum(
                    'ba...l,lpr->ba...pr', state, core)
        return state.diagonal(dim1=1, dim2=-1).sum(-1)


class TensorDecomposition2D(TensorDecomposition, ABC):
    """Abstract base reserved for two-dimensional decomposition results."""


class PEPSDecomposition(TensorDecomposition2D):
    """Placeholder for projected entangled-pair state results."""


class PEPODecomposition(TensorDecomposition2D):
    """Placeholder for projected entangled-pair operator results."""


__all__ = [
    'TensorDecomposition',
    'TensorDecomposition1D',
    'TTDecomposition',
    'TRDecomposition',
    'TTMDecomposition',
    'TRMDecomposition',
    'QTTTuckerDecomposition',
    'QTRTuckerDecomposition',
    'TensorDecomposition2D',
    'PEPSDecomposition',
    'PEPODecomposition',
]
