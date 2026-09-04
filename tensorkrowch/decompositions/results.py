"""Lightweight result objects for tensor decompositions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Sequence,
                    Tuple, Type, Union)

import torch
import torch.nn.functional as nnf

from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord)


Embedding = Optional[Union[Callable[[torch.Tensor], torch.Tensor],
                           Sequence[Callable[[torch.Tensor], torch.Tensor]]]]


def _site_vectors(samples: torch.Tensor,
                  embedding: Embedding,
                  in_dim: Sequence[int],
                  device: torch.device,
                  dtype: torch.dtype) -> List[torch.Tensor]:
    """Builds one input vector per site from samples or embeddings."""
    if not isinstance(samples, torch.Tensor):
        raise TypeError('`samples` should be torch.Tensor type')
    if (samples.ndim < 1) or (samples.shape[-1] != len(in_dim)):
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
        for site, site_in_dim in enumerate(in_dim):
            indices = samples[..., site]
            if torch.any(indices < 0) or \
                    torch.any(indices >= site_in_dim):
                raise ValueError('Sample indices should lie in the input '
                                 'dimension of each site')
            vector = nnf.one_hot(indices.to(torch.long), site_in_dim)
            vectors.append(vector.to(dtype=dtype))
        return vectors

    if isinstance(embedding, (list, tuple)):
        if len(embedding) != len(in_dim):
            raise ValueError(
                '`embedding` should contain one callable per site')
        embeddings = list(embedding)
    elif callable(embedding):
        embeddings = [embedding] * len(in_dim)
    else:
        raise TypeError('`embedding` should be callable or a sequence of '
                        'callables')

    vectors = []
    for site, (site_embedding, site_in_dim) in enumerate(
            zip(embeddings, in_dim)):
        if not callable(site_embedding):
            raise TypeError('Each element of `embedding` should be callable')
        vector = site_embedding(samples[..., site])
        if not isinstance(vector, torch.Tensor):
            raise TypeError('`embedding` should return torch.Tensor objects')
        if vector.shape != (*samples.shape[:-1], site_in_dim):
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
    _in_dim: Tuple[int, ...] = field(init=False, repr=False)
    _out_dim: Optional[Tuple[int, ...]] = field(init=False, repr=False)

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

        rank, batch_shape, in_dim, out_dim = self._validate_cores()
        self.rank = rank
        self._batch_shape = batch_shape
        self._in_dim = in_dim
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
        if self.in_dim != other.in_dim:
            raise ValueError('Decompositions should have matching input dimensions')
        if self.out_dim != other.out_dim:
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
class TTDecomposition(TensorDecomposition):
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

    _family: ClassVar[str] = 'state'
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
                site_in_dim = core.shape[(self.n_batches + 1):-1]
                rank = core.shape[-1]
                core = core.reshape(*self.batch_shape, previous_rank, -1)
                result = (result @ core).reshape(
                    *self.batch_shape,
                    *previous_in_dim,
                    *site_in_dim,
                    rank)
            else:
                site_in_dim = core.shape[(self.n_batches + 1):]
                core = core.reshape(*self.batch_shape, previous_rank, -1)
                result = (result @ core).reshape(
                    *self.batch_shape,
                    *previous_in_dim,
                    *site_in_dim)
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
        vectors = _site_vectors(samples, embedding, self.in_dim,
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

    _family: ClassVar[str] = 'state'
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
        vectors = _site_vectors(samples, embedding, self.in_dim,
                                self.device, self.dtype)
        matrices = [
            torch.einsum('...p,lpr->...lr', vector, core)
            for vector, core in zip(vectors, self.cores)
        ]
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result.diagonal(dim1=-2, dim2=-1).sum(-1)


class _QuantizedTuckerDecomposition(TensorDecomposition):
    """Common two-level contraction for quantized Tucker results."""

    _upper_type: ClassVar[Type[TensorDecomposition]]
    _family: ClassVar[str] = 'quantized_tucker'

    def __init__(
            self,
            upper: TensorDecomposition,
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

    def _evaluate_samples(self,
                          samples: torch.Tensor,
                          embedding: Embedding = None) -> torch.Tensor:
        if embedding is not None:
            raise ValueError(
                '`embedding` is fixed by the quantized Tucker decomposition')
        return self.evaluate(samples)

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

    def flatten(self) -> TensorDecomposition:
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
            self, other: TensorDecomposition) -> torch.Tensor:
        if not isinstance(other, _QuantizedTuckerDecomposition):
            raise TypeError(
                '`other` should be a quantized Tucker decomposition')
        return self.flatten().normalized_overlap(other.flatten())

    def fidelity(self, other: TensorDecomposition) -> torch.Tensor:
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


@dataclass
class TTMDecomposition(TensorDecomposition):
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

    _family: ClassVar[str] = 'ttm'
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

    def apply(self,
              inputs: torch.Tensor,
              embedding: Embedding = None) -> torch.Tensor:
        """Applies the TTM to discrete or embedded product inputs."""
        vectors = _site_vectors(inputs, embedding, self.in_dim,
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
        out_dim = [self.out_dim[0]]
        for local, site_out_dim in zip(local_tensors[1:],
                                       self.out_dim[1:]):
            previous_rank = result.shape[-1]
            result = result.reshape(*inputs.shape[:-1], -1, previous_rank)
            result = torch.einsum('...ar,...rob->...aob', result, local)
            out_dim.append(site_out_dim)
            result = result.reshape(*inputs.shape[:-1], *out_dim,
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
    'QTTTuckerDecomposition',
    'QTRTuckerDecomposition',
]
