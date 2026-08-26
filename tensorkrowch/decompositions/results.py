"""Lightweight result objects for tensor decompositions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Sequence,
                    Tuple, Union)

import torch
import torch.nn.functional as nnf

from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord)


Embedding = Optional[Union[Callable[[torch.Tensor], torch.Tensor],
                           Sequence[Callable[[torch.Tensor], torch.Tensor]]]]


def _site_vectors(samples: torch.Tensor,
                  embedding: Embedding,
                  input_dim: Sequence[int],
                  device: torch.device,
                  dtype: torch.dtype) -> List[torch.Tensor]:
    """Builds one input vector per site from samples or embeddings."""
    if not isinstance(samples, torch.Tensor):
        raise TypeError('`samples` should be torch.Tensor type')
    if (samples.ndim < 1) or (samples.shape[-1] != len(input_dim)):
        raise ValueError(
            'The last dimension of `samples` should equal the number of sites')
    samples = samples.to(device=device)

    if embedding is None:
        integer_dtypes = (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )
        if samples.dtype not in integer_dtypes:
            raise TypeError(
                '`samples` should have an integer dtype when `embedding` is '
                'not provided')

        vectors = []
        for site, site_input_dim in enumerate(input_dim):
            indices = samples[..., site]
            if torch.any(indices < 0) or \
                    torch.any(indices >= site_input_dim):
                raise ValueError('Sample indices should lie in the input '
                                 'dimension of each site')
            vector = nnf.one_hot(indices.to(torch.long), site_input_dim)
            vectors.append(vector.to(dtype=dtype))
        return vectors

    if isinstance(embedding, (list, tuple)):
        if len(embedding) != len(input_dim):
            raise ValueError(
                '`embedding` should contain one callable per site')
        embeddings = list(embedding)
    elif callable(embedding):
        embeddings = [embedding] * len(input_dim)
    else:
        raise TypeError('`embedding` should be callable or a sequence of '
                        'callables')

    vectors = []
    for site, (site_embedding, site_input_dim) in enumerate(
            zip(embeddings, input_dim)):
        if not callable(site_embedding):
            raise TypeError('Each element of `embedding` should be callable')
        vector = site_embedding(samples[..., site])
        if not isinstance(vector, torch.Tensor):
            raise TypeError('`embedding` should return torch.Tensor objects')
        if vector.shape != (*samples.shape[:-1], site_input_dim):
            raise ValueError('The last dimension returned by `embedding` '
                             'should match the input dimension')
        vectors.append(vector.to(device=device, dtype=dtype))
    return vectors


@dataclass
class TensorDecomposition(ABC):
    """Base class for lightweight tensor decomposition results.

    The object stores raw cores, derived rank, structured metrics and small
    metadata only. It does not build a TensorKrowch graph or retain the source
    function used during fitting.
    """

    cores: Sequence[torch.Tensor]
    metrics: DecompositionMetrics = field(default_factory=DecompositionMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)
    n_batches: int = 0
    rank: List[int] = field(init=False)
    _batch_shape: Tuple[int, ...] = field(init=False, repr=False)
    _input_dim: Tuple[int, ...] = field(init=False, repr=False)
    _output_dim: Optional[Tuple[int, ...]] = field(init=False, repr=False)

    _family: ClassVar[str] = 'tensor'
    _topology: ClassVar[str] = 'tensor'

    def __post_init__(self) -> None:
        if not isinstance(self.n_batches, int):
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

        rank, batch_shape, input_dim, output_dim = self._validate_cores()
        self.rank = rank
        self._batch_shape = batch_shape
        self._input_dim = input_dim
        self._output_dim = output_dim

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
    def input_dim(self) -> Tuple[int, ...]:
        """Input dimension associated with every site."""
        return self._input_dim

    @property
    def output_dim(self) -> Optional[Tuple[int, ...]]:
        """Output dimension per site, when the decomposition has one."""
        return self._output_dim

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

    @abstractmethod
    def _evaluate_samples(self,
                          samples: torch.Tensor,
                          embedding: Embedding = None) -> torch.Tensor:
        """Evaluates the represented tensor on user-provided samples."""

    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           copy: bool = False) -> 'TensorDecomposition':
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

    def cpu(self) -> 'TensorDecomposition':
        """Returns this result with all cores stored on CPU."""
        return self.to(device='cpu')

    def _check_overlap_compatibility(
            self, other: 'TensorDecomposition') -> None:
        """Validates topology, shapes and runtime for an overlap."""
        if not isinstance(other, TensorDecomposition):
            raise TypeError('`other` should be TensorDecomposition type')
        if self._family != other._family:
            raise ValueError('The decomposition families are incompatible')
        if len(self.cores) != len(other.cores):
            raise ValueError('Decompositions should have the same number of sites')
        if self.input_dim != other.input_dim:
            raise ValueError('Decompositions should have matching input dimensions')
        if self.output_dim != other.output_dim:
            raise ValueError('Decompositions should have matching output dimensions')
        if self.batch_shape != other.batch_shape:
            raise ValueError('Decompositions should have matching batch shapes')
        if self.device != other.device:
            raise ValueError('Decompositions should be on the same device')

    def _log_overlap(
            self, other: 'TensorDecomposition') -> Tuple[torch.Tensor,
                                                         torch.Tensor]:
        """Returns overlap phase and log-magnitude using scaled transfers."""
        self._check_overlap_compatibility(other)
        dtype = torch.promote_types(self.dtype, other.dtype)
        self_cores = self._standard_cores()
        other_cores = other._standard_cores()

        environment = None
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        log_scale = torch.zeros(
            self.batch_shape, device=self.device, dtype=real_dtype)

        for self_core, other_core in zip(self_cores, other_cores):
            self_core = self_core.to(dtype=dtype)
            other_core = other_core.to(dtype=dtype)
            transfer = torch.einsum(
                '...apr,...bps->...abrs',
                self_core.conj(),
                other_core)
            transfer = transfer.flatten(-4, -3).flatten(-2, -1)

            if environment is None:
                environment = transfer
            else:
                environment = environment @ transfer

            scale = torch.linalg.vector_norm(environment, dim=(-2, -1))
            nonzero = scale > 0
            safe_scale = torch.where(nonzero, scale, torch.ones_like(scale))
            environment = environment / safe_scale[..., None, None]
            log_scale = log_scale + torch.where(
                nonzero, safe_scale.log(), torch.zeros_like(safe_scale))

        overlap = environment.diagonal(dim1=-2, dim2=-1).sum(-1)
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
            self, other: 'TensorDecomposition') -> torch.Tensor:
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

    def fidelity(self, other: 'TensorDecomposition') -> torch.Tensor:
        """Returns ``abs(normalized_overlap(other)) ** 2``."""
        return self.normalized_overlap(other).abs().square()

    def error(self,
              function: Callable[..., torch.Tensor],
              samples: torch.Tensor,
              embedding: Embedding = None,
              **kwargs: Any) -> ErrorRecord:
        """Measures absolute and relative errors on user-provided samples."""
        if not callable(function):
            raise TypeError('`function` should be callable')

        if not isinstance(samples, torch.Tensor):
            raise TypeError('`samples` should be torch.Tensor type')
        samples = samples.to(device=self.device)
        approximation = self._evaluate_samples(samples, embedding=embedding)
        target = function(samples, **kwargs)
        if not isinstance(target, torch.Tensor):
            raise TypeError('`function` should return a torch.Tensor')
        target = target.to(device=approximation.device,
                           dtype=approximation.dtype)
        if target.shape != approximation.shape:
            if target.numel() != approximation.numel():
                raise ValueError(
                    'Function values should match the decomposition output shape')
            target = target.reshape(approximation.shape)

        absolute = torch.linalg.vector_norm(approximation - target)
        denominator = torch.linalg.vector_norm(target)
        if denominator > 0:
            relative = absolute / denominator
        elif absolute == 0:
            relative = torch.zeros_like(absolute)
        else:
            relative = torch.full_like(absolute, torch.inf)

        sample_shape = samples.shape[:-1]
        size = int(torch.Size(sample_shape).numel()) if sample_shape else 1
        return ErrorRecord(
            kind='samples',
            absolute=absolute,
            relative=relative,
            size=size,
            denominator=denominator)

    def as_info(self) -> Dict[str, Any]:
        """Returns rank, dimensions, metrics and metadata for functional APIs."""
        return {
            'topology': self.topology,
            'rank': list(self.rank),
            'input_dim': list(self.input_dim),
            'output_dim': (None if self.output_dim is None
                           else list(self.output_dim)),
            'n_batches': self.n_batches,
            'metrics': self.metrics.as_info(),
            'metadata': dict(self.metadata),
        }


@dataclass
class TTDecomposition(TensorDecomposition):
    """Lightweight tensor-train decomposition with open boundaries.

    This result stores the TT cores, ranks, metrics and metadata without
    constructing a TensorKrowch graph. Methods such as :meth:`contract_dense`
    and :meth:`evaluate` operate directly on the core tensors with PyTorch.
    Consequently, creating or inspecting a decomposition has less overhead
    than creating a :class:`~tensorkrowch.models.MPS` model.

    For graph contractions, training or the rest of the model API, an MPS can
    be initialized directly from the cores:

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
    dimensions; ordinary non-batched decompositions use ``MPS`` as above.
    """

    _family: ClassVar[str] = 'state'
    _topology: ClassVar[str] = 'tt'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        n_sites = len(self.cores)
        batch_shape = tuple(self.cores[0].shape[:self.n_batches])
        input_dim = []

        if n_sites == 1:
            if self.cores[0].ndim != (self.n_batches + 1):
                raise ValueError(
                    'A one-site TT core should have one input dimension')
            input_dim.append(self.cores[0].shape[-1])
            return [], batch_shape, tuple(input_dim), None

        rank = []
        for site, core in enumerate(self.cores):
            if tuple(core.shape[:self.n_batches]) != batch_shape:
                raise ValueError('All TT cores should have the same batch shape')

            if site == 0:
                if core.ndim != (self.n_batches + 2):
                    raise ValueError(
                        'The first TT core should have input and right rank '
                        'dimensions')
                input_dim.append(core.shape[-2])
                rank.append(core.shape[-1])
            elif site == (n_sites - 1):
                if core.ndim != (self.n_batches + 2):
                    raise ValueError(
                        'The last TT core should have left rank and input '
                        'dimensions')
                if core.shape[-2] != rank[-1]:
                    raise ValueError('Adjacent TT ranks should match')
                input_dim.append(core.shape[-1])
            else:
                if core.ndim != (self.n_batches + 3):
                    raise ValueError(
                        'Interior TT cores should have left, input and '
                        'right dimensions')
                if core.shape[-3] != rank[-1]:
                    raise ValueError('Adjacent TT ranks should match')
                input_dim.append(core.shape[-2])
                rank.append(core.shape[-1])

        return rank, batch_shape, tuple(input_dim), None

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
            previous_input_dim = result.shape[self.n_batches:-1]
            previous_rank = result.shape[-1]
            result = result.reshape(*self.batch_shape, -1, previous_rank)

            if site < (len(self.cores) - 1):
                site_input_dim = core.shape[(self.n_batches + 1):-1]
                rank = core.shape[-1]
                core = core.reshape(*self.batch_shape, previous_rank, -1)
                result = (result @ core).reshape(
                    *self.batch_shape,
                    *previous_input_dim,
                    *site_input_dim,
                    rank)
            else:
                site_input_dim = core.shape[(self.n_batches + 1):]
                core = core.reshape(*self.batch_shape, previous_rank, -1)
                result = (result @ core).reshape(
                    *self.batch_shape,
                    *previous_input_dim,
                    *site_input_dim)
        return result

    def evaluate(self,
                 samples: torch.Tensor,
                 embedding: Embedding = None) -> torch.Tensor:
        """Evaluates the TT at discrete or embedded sample coordinates."""
        return self._evaluate_samples(samples, embedding=embedding)

    def _evaluate_samples(self,
                          samples: torch.Tensor,
                          embedding: Embedding = None) -> torch.Tensor:
        if self.n_batches:
            raise ValueError(
                '`evaluate` is not defined for decomposition batch dimensions')
        vectors = _site_vectors(samples, embedding, self.input_dim,
                                self.device, self.dtype)

        matrices = [
            torch.einsum('...p,lpr->...lr', vector, core)
            for vector, core in zip(vectors, self._standard_cores())
        ]
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result.squeeze(-1).squeeze(-1)


@dataclass
class TRDecomposition(TensorDecomposition):
    """Lightweight tensor-ring decomposition with cyclic boundaries.

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
    to the model. Pass ``parameterized=False`` to :class:`~tensorkrowch.models.MPS`
    when trainable parameter nodes are not required. Clone the cores before
    construction if independent tensor storage is required.

    Batched TR cores should initialize
    :class:`~tensorkrowch.models.MPSData` instead:

    >>> batched_cores = [torch.randn(8, 2, 3, 4),
    ...                  torch.randn(8, 4, 5, 2)]
    >>> batched = tk.decompositions.TRDecomposition(
    ...     batched_cores, n_batches=1)
    >>> mps_data = tk.models.MPSData(tensors=batched.cores,
    ...                              n_batches=batched.n_batches)

    The ``MPSData`` form applies only when :attr:`n_batches` is positive.
    """

    _family: ClassVar[str] = 'state'
    _topology: ClassVar[str] = 'tr'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        batch_shape = tuple(self.cores[0].shape[:self.n_batches])
        rank = []
        input_dim = []

        for site, core in enumerate(self.cores):
            if core.ndim != (self.n_batches + 3):
                raise ValueError(
                    'TR cores should have left rank, input and right rank '
                    'dimensions')
            if tuple(core.shape[:self.n_batches]) != batch_shape:
                raise ValueError('All TR cores should have the same batch shape')
            if site and (core.shape[-3] != rank[-1]):
                raise ValueError('Adjacent TR ranks should match')
            input_dim.append(core.shape[-2])
            rank.append(core.shape[-1])

        if self.cores[-1].shape[-1] != self.cores[0].shape[-3]:
            raise ValueError('The last and first cyclic TR ranks should match')
        return rank, batch_shape, tuple(input_dim), None

    def _standard_cores(self) -> List[torch.Tensor]:
        return list(self.cores)

    def contract_dense(self) -> torch.Tensor:
        """Contracts the TR into a dense tensor and closes the cyclic trace."""
        result = self.cores[0]
        input_dim = [self.cores[0].shape[-2]]
        for core in self.cores[1:]:
            initial_rank = result.shape[self.n_batches]
            previous_rank = result.shape[-1]
            result = result.reshape(
                *self.batch_shape, initial_rank, -1, previous_rank)
            result = torch.einsum(
                '...apr,...rqb->...apqb', result, core)
            input_dim.append(core.shape[-2])
            result = result.reshape(
                *self.batch_shape,
                initial_rank,
                *input_dim,
                core.shape[-1])

        return result.diagonal(dim1=self.n_batches, dim2=-1).sum(-1)

    def evaluate(self,
                 samples: torch.Tensor,
                 embedding: Embedding = None) -> torch.Tensor:
        """Evaluates the TR at discrete or embedded sample coordinates."""
        return self._evaluate_samples(samples, embedding=embedding)

    def _evaluate_samples(self,
                          samples: torch.Tensor,
                          embedding: Embedding = None) -> torch.Tensor:
        if self.n_batches:
            raise ValueError(
                '`evaluate` is not defined for decomposition batch dimensions')
        vectors = _site_vectors(samples, embedding, self.input_dim,
                                self.device, self.dtype)
        matrices = [
            torch.einsum('...p,lpr->...lr', vector, core)
            for vector, core in zip(vectors, self.cores)
        ]
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result.diagonal(dim1=-2, dim2=-1).sum(-1)


@dataclass
class TTMDecomposition(TensorDecomposition):
    """Lightweight tensor-train matrix decomposition with open boundaries."""

    _family: ClassVar[str] = 'ttm'
    _topology: ClassVar[str] = 'ttm'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        if self.n_batches:
            raise ValueError('TTM decomposition batches are not supported')

        n_sites = len(self.cores)
        input_dim = []
        output_dim = []
        if n_sites == 1:
            if self.cores[0].ndim != 2:
                raise ValueError(
                    'A one-site TTM core should have input and output dimensions')
            input_dim.append(self.cores[0].shape[0])
            output_dim.append(self.cores[0].shape[1])
            return [], (), tuple(input_dim), tuple(output_dim)

        rank = []
        for site, core in enumerate(self.cores):
            if site == 0:
                if core.ndim != 3:
                    raise ValueError(
                        'The first TTM core should have input, right rank and '
                        'output dimensions')
                input_dim.append(core.shape[0])
                output_dim.append(core.shape[2])
                rank.append(core.shape[1])
            elif site == (n_sites - 1):
                if core.ndim != 3:
                    raise ValueError(
                        'The last TTM core should have left rank, input and '
                        'output dimensions')
                if core.shape[0] != rank[-1]:
                    raise ValueError('Adjacent TTM ranks should match')
                input_dim.append(core.shape[1])
                output_dim.append(core.shape[2])
            else:
                if core.ndim != 4:
                    raise ValueError(
                        'Interior TTM cores should have left, input, right and '
                        'output dimensions')
                if core.shape[0] != rank[-1]:
                    raise ValueError('Adjacent TTM ranks should match')
                input_dim.append(core.shape[1])
                output_dim.append(core.shape[3])
                rank.append(core.shape[2])

        return rank, (), tuple(input_dim), tuple(output_dim)

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

    def apply(self,
              inputs: torch.Tensor,
              embedding: Embedding = None) -> torch.Tensor:
        """Applies the TTM to discrete or embedded product inputs."""
        vectors = _site_vectors(inputs, embedding, self.input_dim,
                                self.device, self.dtype)

        if len(self.cores) == 1:
            core = self.cores[0].unsqueeze(0).unsqueeze(2)
        else:
            cores = [self.cores[0].unsqueeze(0)]
            cores.extend(self.cores[1:-1])
            cores.append(self.cores[-1].unsqueeze(2))

        if len(self.cores) == 1:
            cores = [core]

        local_tensors = [
            torch.einsum('...i,liro->...lor', vector, core)
            for vector, core in zip(vectors, cores)
        ]
        result = local_tensors[0].squeeze(-3)
        output_dim = [self.output_dim[0]]
        for local, site_output_dim in zip(local_tensors[1:],
                                          self.output_dim[1:]):
            previous_rank = result.shape[-1]
            result = result.reshape(*inputs.shape[:-1], -1, previous_rank)
            result = torch.einsum('...ar,...rob->...aob', result, local)
            output_dim.append(site_output_dim)
            result = result.reshape(*inputs.shape[:-1], *output_dim,
                                    local.shape[-1])
        return result.squeeze(-1)

    def _evaluate_samples(self,
                          samples: torch.Tensor,
                          embedding: Embedding = None) -> torch.Tensor:
        return self.apply(samples, embedding=embedding)


__all__ = [
    'TensorDecomposition',
    'TTDecomposition',
    'TRDecomposition',
    'TTMDecomposition',
]
