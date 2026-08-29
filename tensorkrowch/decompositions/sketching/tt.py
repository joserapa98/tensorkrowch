"""Tensor-train decompositions based on recursive sketching from samples."""

from math import prod
from time import perf_counter
from typing import (Any, Callable, Dict, List, Optional, Sequence, Tuple, Union)

import torch

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.metrics import (ErrorRecord,
                                                 TimingRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   DecompositionObserver,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.sketching.base import (
    RecursiveSketching,
    _SketchingFitContext,
)
from tensorkrowch.decompositions.sketching.evaluations import (
    _EvaluationPlanBuilder,
    _EvaluationSession,
)
from tensorkrowch.decompositions.sketching.fitting import InputFitter
from tensorkrowch.decompositions.sketching.phi import (
    PhiOperator,
    _MaterializedPhi,
)
from tensorkrowch.decompositions.sketching.projections import (
    ProjectedRange,
    RangeProjector,
)
from tensorkrowch.decompositions.sketching.quantization import (
    CoordinateMap,
    QuantizedLayout,
    QuantizedSourceAdapter,
)
from tensorkrowch.decompositions.sketching.regions import (
    SiteRegion,
    _SamplePool,
)
from tensorkrowch.decompositions.sketching.sketches import (
    MarginalSketch,
    SketchOperator,
)
from tensorkrowch.decompositions.sketching.sources import (
    SupportTensorSource,
    _resolve_rs_source,
)
from tensorkrowch.decompositions.sketching.specs import (
    _DomainSpec,
    _EmbeddingSpec,
    _OutputSpec,
    _split_samples,
)
from tensorkrowch.decompositions.sketching.transforms import (
    GlobalValueTransform,
    LocalValueTransform,
    _apply_local_transform,
    _collect_local_queries,
    _prepare_global_transform,
)
from tensorkrowch.decompositions.sources import (
    CallableTensorSource,
    ConfigurationBatch,
    TensorSource,
    as_tensor_source,
)
from tensorkrowch.utils import random_unitary


Domain = Optional[Union[torch.Tensor, Sequence[torch.Tensor]]]
Embedding = Union[
    torch.Tensor,
    Callable[[torch.Tensor], torch.Tensor],
    Sequence[Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]],
]
Samples = Union[
    torch.Tensor,
    Sequence[torch.Tensor],
    ConfigurationBatch,
]
Device = Optional[Union[str, torch.device]]
_Rank = Union[int, Sequence[int]]


class TTRSS(RecursiveSketching):
    r"""Reusable Tensor Train Recursive Sketching from Samples problem.

    The fixed object stores the function (or :class:`TensorSource`), embedding,
    domain and output layout. Each :meth:`fit` receives a sample set and its
    truncation options, and returns a lightweight
    :class:`~tensorkrowch.decompositions.TTDecomposition`. The result cores can
    initialize an :class:`~tensorkrowch.models.MPS` directly with
    ``tk.models.MPS(tensors=result.cores)``.

    A shared ``embedding`` and ``domain`` are broadcast to every input site;
    sequences select site-dependent values. Tensor-valued outputs are split
    into one basis-embedded site per output axis, placed as evenly as possible
    by default. Metadata that depends on ``sketch_samples`` is normalized
    independently in every fit, so repeating a fit does not retain mutable
    numerical state.

    Parameters
    ----------
    function : callable or TensorSource, optional
        Scalar- or vector-valued object to approximate. A callable receives a
        tensor with shape ``(batch_size, n_features)`` or
        ``(batch_size, n_features, in_dim)`` and returns shape
        ``(batch_size, output_dim)``. A scalar callable uses
        ``output_dim = 1``.
    embedding : callable, torch.Tensor or sequence
        Shared input embedding or one entry per input site. A site callable
        maps shape ``(batch_size, *coordinate_shape)`` to
        ``(batch_size, input_dim)``; a tensor stores its finite-domain matrix.
    input_dim : int or sequence of int, optional
        Expected embedding dimension. One integer is broadcast to every input
        site. If omitted, dimensions are inferred from the embeddings.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Finite values used to fit the embedding. One tensor is broadcast to
        every input site; a sequence supplies one domain per site. If omitted,
        each domain is inferred from the corresponding sketch samples.
    domain_multiplier : int, optional
        Maximum inferred-domain size in multiples of ``input_dim``.
    out_position : int or sequence of int, optional
        Position of each tensor-output axis. By default the output sites split
        the input chain into groups as evenly as possible.
    source : TensorSource, optional
        Explicit source alternative to ``function``. Exactly one of
        ``function`` and ``source`` should be supplied.
    device : str or torch.device, optional
        Device used for source evaluations and numerical decomposition.
    dtype : torch.dtype, optional
        Dtype used for callable values, fitted tensors and resulting cores.
    output_device : str, torch.device or None, optional
        Device receiving completed cores. The default is CPU; ``None`` keeps
        them on the compute device.

    Examples
    --------
    >>> def function(data):
    ...     return data.prod(dim=1, keepdim=True)
    >>> def embedding(data):
    ...     return torch.stack([data, 1 - data], dim=-1)
    >>> samples = torch.rand(32, 3)
    >>> decomposer = tk.decompositions.TTRSS(function, embedding)
    >>> result = decomposer.fit(samples, rank=2)
    >>> len(result.cores)
    3
    >>> model = tk.models.MPS(tensors=result.cores, parameterized=False)
    >>> model.boundary
    'obc'
    """

    def __init__(
            self,
            function=None,
            embedding: Optional[Embedding] = None,
            input_dim: Optional[Union[int, Sequence[int]]] = None,
            domain: Domain = None,
            domain_multiplier: int = 1,
            out_position: Optional[Union[int, Sequence[int]]] = None,
            *,
            source: Optional[TensorSource] = None,
            device: Device = None,
            dtype: Optional[torch.dtype] = None,
            output_device: Device = 'cpu',
            input_fitters: Optional[Sequence[InputFitter]] = None,
            range_projector: Optional[RangeProjector] = None,
            global_transform: Optional[GlobalValueTransform] = None,
            local_transform: Optional[LocalValueTransform] = None,
            local_solver: Optional[LeastSquaresSolver] = None,
            synchronize_timers: bool = True) -> None:
        if (function is None) == (source is None):
            raise ValueError(
                'Exactly one of `function` and `source` should be provided')
        source_like = function if source is None else source
        if source is None and not callable(source_like):
            raise TypeError('`function` should be callable')
        if source is not None and not isinstance(source, TensorSource):
            raise TypeError('`source` should implement TensorSource')
        if embedding is None:
            raise TypeError('`embedding` should be provided')
        if not (callable(embedding) or isinstance(embedding, torch.Tensor)):
            if isinstance(embedding, (str, bytes)):
                raise TypeError(
                    '`embedding` should be callable, a tensor or a sequence')
            try:
                embedding = tuple(embedding)
            except TypeError as exc:
                raise TypeError(
                    '`embedding` should be callable, a tensor or a sequence') \
                    from exc
            if not embedding or not all(
                    callable(entry) or isinstance(entry, torch.Tensor)
                    for entry in embedding):
                raise TypeError(
                    'Every `embedding` entry should be callable or a tensor')
        if input_dim is not None:
            if isinstance(input_dim, bool):
                raise TypeError(
                    '`input_dim` should be int, a sequence of ints or None')
            if isinstance(input_dim, int):
                if input_dim < 1:
                    raise ValueError('`input_dim` should be positive')
            else:
                if isinstance(input_dim, (str, bytes)):
                    raise TypeError(
                        '`input_dim` should be int, a sequence of ints or None')
                try:
                    input_dim = tuple(input_dim)
                except TypeError as exc:
                    raise TypeError(
                        '`input_dim` should be int, a sequence of ints or None') \
                        from exc
                if not input_dim or any(
                        isinstance(dim, bool) or not isinstance(dim, int)
                        or dim < 1 for dim in input_dim):
                    raise ValueError(
                        '`input_dim` should contain positive integers')
        if domain is not None and not isinstance(domain, torch.Tensor):
            if isinstance(domain, (str, bytes)):
                raise TypeError(
                    '`domain` should be a tensor or a sequence of tensors')
            try:
                domain = tuple(domain)
            except TypeError as exc:
                raise TypeError(
                    '`domain` should be a tensor or a sequence of tensors') \
                    from exc
            if not all(isinstance(value, torch.Tensor) for value in domain):
                raise TypeError(
                    '`domain` should be a tensor or a sequence of tensors')
        if isinstance(domain_multiplier, bool) or \
                not isinstance(domain_multiplier, int):
            raise TypeError('`domain_multiplier` should be int type')
        if domain_multiplier < 1:
            raise ValueError('`domain_multiplier` should be positive')
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type or None')
        if local_solver is not None and \
                not isinstance(local_solver, LeastSquaresSolver):
            raise TypeError(
                '`local_solver` should be LeastSquaresSolver type or None')
        if not isinstance(synchronize_timers, bool):
            raise TypeError('`synchronize_timers` should be bool type')

        self._source_like = source_like
        self._embedding = embedding
        self._input_dim_option = input_dim
        self._domain = domain
        self._domain_multiplier = domain_multiplier
        self._out_position = out_position
        self._device = None if device is None else torch.device(device)
        self._dtype = dtype
        self._output_device = output_device
        self._input_fitters_option = input_fitters
        self._range_projector_option = range_projector
        self._global_transform_option = global_transform
        self._local_transform_option = local_transform
        self._local_solver = LeastSquaresSolver() \
            if local_solver is None else local_solver
        self._synchronize_timers = synchronize_timers
        self._input_kind = 'coordinates'

    @classmethod
    def quantized(
            cls,
            function=None,
            *,
            source: Optional[TensorSource] = None,
            layout: Optional[QuantizedLayout] = None,
            n_variables: Optional[int] = None,
            base: Union[int, Sequence[int]] = 2,
            level: Union[int, Sequence[int]] = 1,
            ordering: str = 'grouped',
            digit_order: str = 'coarse_to_fine',
            permutation=None,
            coordinate_map: Optional[
                Union[CoordinateMap, Sequence[CoordinateMap]]] = None,
            domain: Domain = None,
            source_space: str = 'physical',
            source_layout: Optional[QuantizedLayout] = None,
            sample_space: str = 'physical',
            computational_grid: str = 'endpoints',
            out_of_domain: str = 'error',
            out_position: Optional[Union[int, Sequence[int]]] = None,
            device: Device = None,
            dtype: Optional[torch.dtype] = None,
            output_device: Device = 'cpu',
            input_fitters: Optional[Sequence[InputFitter]] = None,
            range_projector: Optional[RangeProjector] = None,
            global_transform: Optional[GlobalValueTransform] = None,
            local_transform: Optional[LocalValueTransform] = None,
            local_solver: Optional[LeastSquaresSolver] = None,
            synchronize_timers: bool = True) -> 'TTRSS':
        """Creates a QTT-RSS problem with basis-embedded digit sites.

        ``domain`` describes one physical interval per original variable; it
        is not the finite domain passed to ordinary RSS sites. The returned
        object accepts physical sketch samples by default and quantizes them
        before every fit. ``sample_space="digits"`` is available when a map
        has no inverse or samples are already encoded.
        """
        adapter, layout = _quantized_source(
            function=function,
            source=source,
            layout=layout,
            n_variables=n_variables,
            base=base,
            level=level,
            ordering=ordering,
            digit_order=digit_order,
            permutation=permutation,
            coordinate_map=coordinate_map,
            domain=domain,
            source_space=source_space,
            source_layout=source_layout,
            computational_grid=computational_grid,
            out_of_domain=out_of_domain,
            device=device,
            dtype=dtype)
        embeddings = tuple(
            torch.eye(dimension) for dimension in layout.input_dim)
        digit_domains = tuple(
            torch.arange(dimension) for dimension in layout.input_dim)
        return _QuantizedTTRSS(
            source=adapter,
            embedding=embeddings,
            input_dim=layout.input_dim,
            domain=digit_domains,
            out_position=out_position,
            device=device,
            dtype=dtype,
            output_device=output_device,
            input_fitters=input_fitters,
            range_projector=range_projector,
            global_transform=global_transform,
            local_transform=local_transform,
            local_solver=local_solver,
            synchronize_timers=synchronize_timers,
            quantized_layout=layout,
            quantized_adapter=adapter,
            sample_space=sample_space)

    def _normalize_samples(self, sketch_samples: Samples) -> ConfigurationBatch:
        """Normalizes packed or heterogeneous samples for the fixed source."""
        kind = 'coordinates'
        if isinstance(self._source_like, TensorSource) and \
                not isinstance(self._source_like, CallableTensorSource):
            kind = 'indices'
        if isinstance(sketch_samples, ConfigurationBatch):
            samples = sketch_samples
            if samples.kind != kind:
                raise ValueError(
                    f'`sketch_samples.kind` should be {kind!r}')
        else:
            samples = ConfigurationBatch(sketch_samples, kind=kind)
        if samples.batch_size < 1:
            raise ValueError('`sketch_samples` should contain samples')
        if isinstance(self._source_like, TensorSource) and \
                samples.n_sites != len(self._source_like.input_dim):
            raise ValueError(
                '`source` and `sketch_samples` should share input sites')
        return samples

    def _domain_on_device(self, device: torch.device) -> Domain:
        """Moves the fixed domain to the active fit device."""
        if self._domain is None:
            return None
        if isinstance(self._domain, torch.Tensor):
            return self._domain.to(device)
        return tuple(value.to(device) for value in self._domain)

    def _embeddings_on_device(
            self,
            n_input_sites: int,
            device: torch.device,
            dtype: torch.dtype) -> Tuple[Any, ...]:
        """Broadcasts embeddings and makes their result placement explicit."""
        entries = (self._embedding,) * n_input_sites \
            if callable(self._embedding) or \
            isinstance(self._embedding, torch.Tensor) \
            else tuple(self._embedding)
        if len(entries) != n_input_sites:
            raise ValueError(
                '`embedding` should contain one entry per input site')
        normalized = []
        for entry in entries:
            if isinstance(entry, torch.Tensor):
                normalized.append(entry.to(device=device, dtype=dtype))
                continue

            def site_embedding(values, function=entry):
                result = function(values.to(device))
                if not isinstance(result, torch.Tensor):
                    raise TypeError('`embedding` should return a torch.Tensor')
                return result.to(device=device, dtype=dtype)

            normalized.append(site_embedding)
        return tuple(normalized)

    def _validate_input_dim(self, input_dim: Sequence[int]) -> None:
        """Checks an optional public input-dimension declaration."""
        expected = self._input_dim_option
        if expected is None:
            return
        expected = (expected,) * len(input_dim) \
            if isinstance(expected, int) else tuple(expected)
        if len(expected) != len(input_dim):
            raise ValueError(
                '`input_dim` should contain one value per input site')
        if tuple(input_dim) != expected:
            raise ValueError(
                '`input_dim` should match the dimensions returned by '
                '`embedding`')

    def _source_probe(
            self,
            samples: ConfigurationBatch,
            n_input_sites: int) -> Tuple[torch.Tensor, Optional[TensorSource]]:
        """Evaluates one row and returns any already-constructed source."""
        if isinstance(self._source_like, TensorSource):
            source = self._source_like
            if len(source.input_dim) != n_input_sites:
                raise ValueError(
                    '`source` and `sketch_samples` should share input sites')
            if self._device is not None and source.device != self._device:
                raise ValueError('`source` and `device` should match')
            self._input_kind = 'coordinates' \
                if isinstance(source, CallableTensorSource) else 'indices'
            ids = torch.zeros(1, dtype=torch.long, device=samples.device)
            configurations = samples.index_select(ids).to(source.device)
            probe = source.evaluate(configurations)
            if self._dtype is not None and probe.dtype != self._dtype:
                raise ValueError('`source` and `dtype` should match')
            return probe, source

        device = torch.device('cpu') if self._device is None else self._device
        ids = torch.zeros(1, dtype=torch.long, device=samples.device)
        configurations = samples.index_select(ids).to(device)
        try:
            probe = self._source_like(configurations.values)
        except Exception as exc:
            raise ValueError(
                '`function` failed on a sketch-sample batch') from exc
        if not isinstance(probe, torch.Tensor):
            raise TypeError('`function` should return a torch.Tensor')
        if probe.device != device:
            raise ValueError(
                '`function` should return values on the compute device')
        return probe, None

    def _initialize_fit(
            self,
            sketch_samples: Samples,
            generator: Optional[torch.Generator]) -> ConfigurationBatch:
        """Normalizes all sample-dependent fixed objects for one fit."""
        sample_batch = self._normalize_samples(sketch_samples)
        n_input_sites = sample_batch.n_sites
        probe, source = self._source_probe(sample_batch, n_input_sites)
        if probe.ndim < 1 or probe.shape[0] != 1:
            raise ValueError(
                '`function` should preserve the leading batch dimension')
        if not (probe.is_floating_point() or probe.is_complex()):
            raise TypeError('`function` output should be floating or complex')
        if not torch.isfinite(probe).all():
            raise ValueError('`function` output should be finite')

        device = probe.device
        dtype = probe.dtype if self._dtype is None else self._dtype
        probe = probe.to(dtype=dtype)
        samples = sample_batch.to(device)
        domains = _DomainSpec.normalize(
            self._domain_on_device(device), n_input_sites, samples=samples)
        site_embeddings = self._embeddings_on_device(
            n_input_sites, device, dtype)
        embeddings = _EmbeddingSpec.normalize(site_embeddings, domains)
        self._validate_input_dim(embeddings.input_dim)

        if domains.inferred:
            inferred_values = []
            for values, input_dim in zip(
                    domains.values, embeddings.input_dim):
                maximum = self._domain_multiplier * input_dim
                if values.shape[0] >= maximum:
                    random_device = torch.device('cpu') \
                        if generator is None else generator.device
                    ids = torch.randperm(
                        values.shape[0],
                        generator=generator,
                        device=random_device)[:maximum]
                    values = values.index_select(0, ids.to(values.device))
                inferred_values.append(values)
            domains = _DomainSpec(inferred_values, inferred=True)
            embeddings = _EmbeddingSpec.normalize(site_embeddings, domains)

        out_position = self._out_position
        outputs = _OutputSpec.normalize(
            probe, n_input_sites, out_position=out_position)

        if source is None:
            function = self._source_like

            def typed_function(values):
                result = function(values)
                if not isinstance(result, torch.Tensor):
                    raise TypeError(
                        '`function` should return a torch.Tensor')
                return result.to(dtype=dtype)

            source = as_tensor_source(
                typed_function,
                input_dim=domains.n_values,
                output_shape=tuple(probe.shape[1:]),
                dtype=dtype,
                device=device)

        RecursiveSketching.__init__(
            self,
            source=source,
            domains=domains,
            embeddings=embeddings,
            outputs=outputs,
            input_fitters=self._input_fitters_option,
            range_projector=self._range_projector_option,
            global_transform=self._global_transform_option,
            local_transform=self._local_transform_option,
            output_device=self._output_device,
            synchronize_timers=self._synchronize_timers)
        return samples

    def _configuration_batch(self, samples: Samples) -> ConfigurationBatch:
        """Builds the source configurations for original input samples."""
        if isinstance(samples, ConfigurationBatch):
            return samples
        return ConfigurationBatch(samples, kind=self._input_kind)

    def _evaluate_samples(
            self,
            samples: Samples,
            batch_size: int) -> torch.Tensor:
        """Evaluates original samples in deterministic contiguous batches."""
        configurations = self._configuration_batch(samples)
        chunks = []
        for start in range(0, configurations.batch_size, batch_size):
            stop = min(start + batch_size, configurations.batch_size)
            ids = torch.arange(start, stop, device=configurations.device)
            chunks.append(self.source.evaluate(
                configurations.index_select(ids)))
        return torch.cat(chunks, dim=0)

    def _prepare_source(self, context: _SketchingFitContext) -> None:
        """Inserts sampled output indices and retains validation targets."""
        samples = context.state['input_samples']
        labels = context.state['labels']
        need_values = (
            (not self.outputs.scalar and labels is None) or
            context.collect_metrics)
        values = self._evaluate_samples(
            samples, context.spec.batch_size) if need_values else None

        if self.outputs.scalar:
            if labels is not None:
                raise ValueError(
                    '`labels` should be None for a scalar function')
            output_indices = torch.empty(
                samples.batch_size, 0,
                device=samples.device, dtype=torch.long)
            selected = None if values is None \
                else self.outputs.validate_values(values)
            extended_samples = _split_samples(
                samples, self.outputs.n_input_sites)
        elif labels is None:
            _, output_indices, selected = self.outputs.resolve_labels(
                values, generator=context.generator)
            extended_samples = self.outputs.insert_indices(
                samples, output_indices)
        else:
            flat_labels = self.outputs._validate_flat_labels(labels).to(
                samples.device)
            if flat_labels.shape[0] != samples.batch_size:
                raise ValueError(
                    '`labels` and `sketch_samples` should share batch size')
            output_indices = self.outputs.unflatten_labels(flat_labels)
            extended_samples = self.outputs.insert_indices(
                samples, output_indices)
            selected = None
            if values is not None:
                values = self.outputs.validate_values(values)
                selected = values.reshape(values.shape[0], -1).gather(
                    1, flat_labels.unsqueeze(1)).squeeze(1)

        context.state['extended_samples'] = extended_samples
        context.state['selected_values'] = selected
        context.state['output_indices'] = output_indices

    def fit(
            self,
            sketch_samples: Samples,
            labels: Optional[torch.Tensor] = None,
            rank: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: int = 64,
            generator: Optional[torch.Generator] = None,
            random_projection: Optional[bool] = None,
            projection_dim: Optional[int] = None,
            projection_oversampling: int = 0,
            n_power_iter: int = 0,
            legacy_projection: bool = True,
            warm_start: Optional[TTDecomposition] = None,
            verbose: Union[bool, int] = 0,
            collect_metrics: bool = False,
            observer: Optional[DecompositionObserver] = None
            ) -> TTDecomposition:
        r"""Decomposes the fixed function using correlated sketch samples.

        ``sketch_samples`` contains one sample coordinate per original input
        site. It may be a packed tensor or one tensor per site when coordinate
        shapes differ. Every axis after the callable's batch dimension becomes
        a separate basis-embedded output site; those sites are absent from the
        supplied samples. Every non-final cut fits its sampled input axis,
        optionally projects its range, and calls :func:`truncated_svd` with all
        active truncation conditions combined.

        Parameters
        ----------
        sketch_samples : torch.Tensor, sequence of torch.Tensor or ConfigurationBatch
            Packed coordinates with leading shape
            ``(batch_size, n_features)``, or one tensor with leading batch
            dimension per input site.
        labels : torch.Tensor, optional
            Flattened output labels with shape ``(batch_size,)`` for a vector
            function. If omitted, labels are sampled proportionally to the
            squared magnitude of the function output.
        rank : int, optional
            Number of singular values to keep.
        cutoff : float, optional
            Minimum singular value to keep. It must be non-negative. Singular
            values ``<= cutoff`` are removed.
        atol : float, optional
            Absolute tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded while
            the accumulated sum of squares is ``<= atol``. It must be non-negative.
        rtol : float, optional
            Relative tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded while
            the tail sum of squares divided by the total sum of squares is
            ``<= rtol``. It must be in ``[0, 1]``.
        cum_percentage : float, optional
            Minimum fraction of squared singular-value mass to keep. Equivalent to
            setting ``rtol = 1 - cum_percentage``. It must be in ``[0, 1]``.

            .. math::

                \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
                cum\_percentage

        batch_size : int, optional
            Maximum number of unique configurations evaluated together.
        generator : torch.Generator, optional
            Generator for domain subsampling, labels and random rotations.
        random_projection : bool, optional
            Selects the shared range-projection strategy explicitly. ``True``
            uses a randomized range finder and ``False`` keeps the fitted range
            unchanged. If omitted, ``legacy_projection`` selects the
            compatibility behavior.
        projection_dim : int, optional
            Dimension of the randomized projection output. It defaults to
            ``rank``; if both are omitted, the projection is square.
        projection_oversampling : int, optional
            Extra randomized range dimensions retained before truncation.
        n_power_iter : int, optional
            Power iterations used by the randomized range finder.
        legacy_projection : bool, optional
            Whether to preserve the legacy square Haar rotation before every
            non-final SVD. An explicit ``random_projection`` takes precedence.
        warm_start : TTDecomposition, optional
            Reserved for a future mathematically defined TT-RSS update. Passing
            a result currently raises ``NotImplementedError`` rather than
            retaining fit state accidentally.
        verbose : bool or int, optional
            Verbosity level from 0 to 3. Level 1 prints phases and site titles,
            level 2 adds timings and structured details, and level 3 also
            prints every final core.
        collect_metrics : bool, optional
            Collects timings, truncation, fitting, solve and sample-error
            records. The fast path avoids those diagnostics when ``False``.
        observer : DecompositionObserver, optional
            Receives structured decomposition events.

        Returns
        -------
        TTDecomposition
            Lightweight OBC TT result. Completed cores are moved to
            ``output_device`` (CPU by default).

        Examples
        --------
        >>> def function(data):
        ...     return data.sum(dim=1, keepdim=True)
        >>> def embedding(data):
        ...     return torch.stack([torch.ones_like(data), data], dim=-1)
        >>> samples = torch.rand(24, 4)
        >>> result = tk.decompositions.TTRSS(function, embedding).fit(
        ...     samples, rank=3)
        >>> result.input_dim
        (2, 2, 2, 2)
        """
        sample_batch = self._normalize_samples(sketch_samples)
        if rank is None and cum_percentage is None:
            raise ValueError(
                'At least one of `rank` and `cum_percentage` should be given')
        if generator is not None and \
                not isinstance(generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator type or None')
        if not isinstance(legacy_projection, bool):
            raise TypeError('`legacy_projection` should be bool type')
        if random_projection is not None and not isinstance(
                random_projection, bool):
            raise TypeError('`random_projection` should be bool type or None')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        if warm_start is not None:
            if not isinstance(warm_start, TTDecomposition):
                raise TypeError(
                    '`warm_start` should be TTDecomposition type or None')
            raise NotImplementedError(
                'TT-RSS does not yet define a warm-start update; pass None')
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                raise TypeError('`labels` should be torch.Tensor type')
            if labels.shape != (sample_batch.batch_size,):
                raise ValueError(
                    '`labels` should have shape (batch_size,)')

        samples = self._initialize_fit(sample_batch, generator)
        use_legacy_projection = legacy_projection \
            if random_projection is None else False
        use_random_projection = not use_legacy_projection \
            if random_projection is None else random_projection
        context = self._new_context(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            random_projection=use_random_projection,
            projection_dim=projection_dim,
            projection_oversampling=projection_oversampling,
            n_power_iter=n_power_iter,
            batch_size=batch_size,
            generator=generator,
            collect_metrics=collect_metrics,
            verbose=verbose,
            observer=observer)
        context.state.update({
            'input_samples': samples,
            'labels': labels,
            'legacy_projection': use_legacy_projection,
        })
        return self._execute(context)

    def _build_regions(
            self,
            context: _SketchingFitContext) -> Dict[str, object]:
        """Builds every contiguous prefix and suffix sketch once."""
        pool = _SamplePool(
            context.state['extended_samples'],
            sites=range(self.outputs.n_sites))
        prefixes = tuple(
            pool.restrict(SiteRegion(range(site)))
            for site in range(self.outputs.n_sites + 1))
        suffixes = tuple(
            pool.restrict(SiteRegion(range(site, self.outputs.n_sites)))
            for site in range(self.outputs.n_sites + 1))
        return {
            'pool': pool,
            'prefixes': prefixes,
            'suffixes': suffixes,
        }

    def _build_phi(
            self,
            site: int,
            regions: Dict[str, object],
            context: _SketchingFitContext) -> PhiOperator:
        """Builds the prefix/current/suffix Phi operator for one TT site."""
        components = []
        prefix = regions['prefixes'][site]
        if len(prefix.region):
            components.append(prefix)

        kind, axis = self.outputs.layout[site]
        values = self.domains.for_site(axis) if kind == 'input' \
            else torch.arange(
                self.outputs.output_shape[axis], device=self.source.device)
        components.append((site, values))

        suffix = regions['suffixes'][site + 1]
        if len(suffix.region):
            components.append(suffix)
        return PhiOperator(
            self.source,
            components,
            self.outputs,
            input_sites=self.outputs.input_positions,
            output_sites=self.outputs.positions,
            input_kind=self._input_kind)

    @staticmethod
    def _current_axis(phi: PhiOperator, site: int) -> int:
        """Returns the explicit current-site axis in one Phi layout."""
        return phi.layout.index(site)

    def _legacy_project(
            self,
            site: int,
            tensor: torch.Tensor,
            context: _SketchingFitContext) -> ProjectedRange:
        """Applies the former square right rotation without reducing rank."""
        with context.phase('range.project', site=site):
            random_device = torch.device('cpu') if context.generator is None \
                else context.generator.device
            rotation = random_unitary(
                n=tensor.shape[-1],
                device=random_device,
                dtype=tensor.dtype,
                generator=context.generator).to(tensor.device)
            rotated = tensor @ rotation
            matrix = rotated.reshape(-1, rotated.shape[-1])
        return ProjectedRange(
            small_matrix=matrix,
            basis=None,
            original_shape=tuple(rotated.shape),
            axis=rotated.ndim - 1)

    def _decompose(self, context: _SketchingFitContext) -> TTDecomposition:
        """Executes the serial five-stage TT recursive-sketching workflow."""
        n_sites = self.outputs.n_sites

        # Plan every Phi before one globally deduplicated source evaluation.
        phis = [self._plan_phi(site, context) for site in range(n_sites)]
        builder = _EvaluationPlanBuilder(self.source)
        main_handles = [phi.collect(builder) for phi in phis]
        local_handles = [
            _collect_local_queries(
                builder, phi, context.local_transform, context)
            for phi in phis
        ]
        input_handles = []
        for site, phi in enumerate(phis):
            axis = self._current_axis(phi, site)
            queries = self._required_input_queries(
                site, phi, axis, context)
            input_handles.append(tuple(
                phi.collect(builder, query) for query in queries))
        _prepare_global_transform(
            builder, context.global_transform, context)
        session = _EvaluationSession(
            builder.freeze(),
            global_transform=context.global_transform,
            context=context)
        with context.phase('source.evaluate'):
            session.evaluate_source(batch_size=context.spec.batch_size)
        with context.phase('values.global_transform'):
            results = session.evaluate()
        if context.collect_metrics:
            context.metrics.evaluations.append(session.stats)
        evaluation_view = session.view()
        context.state['input_query_results'] = {
            site: tuple(results[handle] for handle in handles)
            for site, handles in enumerate(input_handles)
        }

        # Fit every sampled current axis before topology-dependent recursion.
        fitted_tensors = {}
        for site, phi in enumerate(phis):
            materialized = _MaterializedPhi(
                results[main_handles[site]], phi.layout)
            with context.phase('values.local_transform', site=site):
                local_view = _apply_local_transform(
                    context.local_transform,
                    materialized,
                    data=context,
                    evaluation=evaluation_view,
                    query_results=tuple(
                        results[handle] for handle in local_handles[site]))
            fitter = context.input_fitters[site]
            if getattr(fitter, 'requires_functional_phi', False):
                if not context.global_transform.is_identity or \
                        not context.local_transform.is_identity:
                    raise ValueError(
                        'A functional input fitter currently requires '
                        'identity value transforms')
                local_view = phi
            fitted = self._fit_input_axis(
                site,
                local_view,
                self._current_axis(phi, site),
                context)
            fitted_tensors[site] = fitted.tensor

        # Trim the left ranges in site order so theoretical rank caps include
        # the rank actually selected at the preceding cut.
        site_dim = self.outputs.site_dim(self.embeddings)
        b_tensors = {}
        previous_rank = 1
        for site in range(n_sites - 1):
            tensor = fitted_tensors[site]
            if context.state['legacy_projection']:
                projected = self._legacy_project(site, tensor, context)
            else:
                projected = self._project_range(
                    site, tensor, tensor.ndim - 1, context)
            right_capacity = prod(site_dim[site + 1:])
            effective_rank = min(
                previous_rank * site_dim[site], right_capacity)
            if context.spec.rank is not None:
                effective_rank = min(effective_rank, context.spec.rank)
            u, _, _, _ = self._trim(
                site, projected, context, rank=effective_rank)
            b_tensors[site] = u
            previous_rank = u.shape[-1]
        b_tensors[n_sites - 1] = fitted_tensors[n_sites - 1]

        # Apply each contained-prefix recursion independently to construct A_k.
        a_tensors = {}
        prefixes = context.regions['prefixes']
        for site in range(n_sites - 1):
            with context.phase('recursion.apply', site=site):
                recursion = prefixes[site].recursive_projector(
                    prefixes[site + 1])
                b_tensor = b_tensors[site].reshape(
                    recursion.child_size, site_dim[site], -1)
                gathered = b_tensor.index_select(
                    0, recursion.gather.to(b_tensor.device))
                embedded = self.outputs.embed_site(
                    site,
                    recursion.new_values[0],
                    self.embeddings,
                    dtype=b_tensor.dtype).to(b_tensor.device)
                a_tensors[site] = torch.einsum(
                    'bpr,bp->br', gathered, embedded)

        # Solve the local changes of sketch basis after all B_k and A_k exist.
        cores = [b_tensors[0]]
        for site in range(1, n_sites):
            target = b_tensors[site]
            target_shape = target.shape[1:]
            target_matrix = target.reshape(target.shape[0], -1)
            with context.phase('core.solve', site=site):
                solution, record = self._solve_local(
                    a_tensors[site - 1], target_matrix, site, context)
            if context.collect_metrics and record is not None:
                context.metrics.local_solves.append(record)
            core = solution.reshape(
                solution.shape[0], *target_shape)
            cores.append(core)

        context.cores.extend(cores)
        return self._assemble_result(context)

    def _solve_local(
            self,
            environment: torch.Tensor,
            target: torch.Tensor,
            site: int,
            context: _SketchingFitContext):
        """Solves one TT sketch-basis equation with the shared ALS solver."""
        return self._local_solver.solve(
            environment,
            target,
            site=site,
            return_record=context.need_diagnostics)

    def _evaluate_extended_cores(
            self,
            cores: Sequence[torch.Tensor],
            samples: Sequence[torch.Tensor]) -> torch.Tensor:
        """Evaluates TT cores on already output-extended sample rows."""
        vectors = [
            self.outputs.embed_site(
                site,
                values,
                self.embeddings,
                dtype=cores[0].dtype).to(cores[0].device)
            for site, values in enumerate(samples)
        ]
        if len(cores) == 1:
            return torch.einsum('bp,p->b', vectors[0], cores[0])

        state = torch.einsum('bp,pr->br', vectors[0], cores[0])
        for vector, core in zip(vectors[1:-1], cores[1:-1]):
            state = torch.einsum('bl,lpr,bp->br', state, core, vector)
        return torch.einsum(
            'bl,lp,bp->b', state, cores[-1], vectors[-1])

    def _assemble_result(
            self,
            context: _SketchingFitContext) -> TTDecomposition:
        """Adds optional sample error and moves completed cores for storage."""
        selected = context.state['selected_values']
        metadata = {
            'algorithm': 'tt_rss',
            'output_shape': tuple(self.outputs.output_shape),
            'out_position': None if self.outputs.scalar else (
                self.outputs.positions[0]
                if self.outputs.n_output_sites == 1
                else tuple(self.outputs.positions)),
            'legacy_projection': context.state['legacy_projection'],
            'core_shapes': [tuple(core.shape) for core in context.cores],
        }
        if context.collect_metrics and selected is not None:
            approximation = self._evaluate_extended_cores(
                context.cores, context.state['extended_samples'])
            selected = selected.to(
                device=approximation.device, dtype=approximation.dtype)
            absolute = torch.linalg.vector_norm(approximation - selected)
            denominator = torch.linalg.vector_norm(selected)
            if denominator > 0:
                relative = absolute / denominator
            elif absolute == 0:
                relative = torch.zeros_like(absolute)
            else:
                relative = torch.full_like(absolute, torch.inf)
            context.metrics.errors.append(ErrorRecord(
                kind='sketch_samples',
                absolute=absolute,
                relative=relative,
                size=selected.shape[0],
                denominator=denominator))
            metadata['sample_error'] = float(relative.detach().cpu())

        if context.collect_metrics and context.metrics.truncations:
            local_squared = sum(
                record.discarded_squared_norm
                for record in context.metrics.truncations)
            local_aggregate = local_squared ** 0.5
            context.metrics.errors.append(ErrorRecord(
                kind='sketch_svd_local_aggregate',
                absolute=local_aggregate,
                size=len(context.metrics.truncations)))
            metadata['sketch_svd_local_aggregate'] = local_aggregate

        for site, core in enumerate(context.cores):
            context.emit('site_complete', site=site, values={
                'total_sites': len(context.cores),
                'shape': tuple(core.shape),
            })
            context.emit(
                'core', site=site, level=3, values={'tensor': core})
        cores = [context.runtime.finalize(core) for core in context.cores]
        return TTDecomposition(
            cores=cores,
            metrics=context.metrics,
            metadata=metadata)

    def _validate_result(
            self,
            result: TTDecomposition,
            context: _SketchingFitContext) -> None:
        """Checks the TT topology and final site dimensions."""
        if not isinstance(result, TTDecomposition):
            raise TypeError('`result` should be TTDecomposition type')
        if len(result.cores) != self.outputs.n_sites:
            raise ValueError('The result should contain one core per TT site')
        if result.input_dim != self.outputs.site_dim(self.embeddings):
            raise ValueError('The result input dimensions are inconsistent')


def _quantized_source(
        *,
        function,
        source,
        layout,
        n_variables,
        base,
        level,
        ordering,
        digit_order,
        permutation,
        coordinate_map,
        domain,
        source_space,
        source_layout,
        computational_grid,
        out_of_domain,
        device,
        dtype):
    """Builds the common digit-source adapter used by TT and TR QTT-RSS."""
    if (function is None) == (source is None):
        raise ValueError(
            'Exactly one of `function` and `source` should be provided')
    source_like = function if source is None else source
    if layout is None:
        if isinstance(n_variables, bool) or not isinstance(n_variables, int):
            raise TypeError(
                '`n_variables` should be int type when `layout` is omitted')
        layout = QuantizedLayout(
            n_variables=n_variables,
            base=base,
            level=level,
            ordering=ordering,
            digit_order=digit_order,
            permutation=permutation)
    elif not isinstance(layout, QuantizedLayout):
        raise TypeError('`layout` should be QuantizedLayout type or None')
    elif n_variables is not None and n_variables != layout.n_variables:
        raise ValueError('`n_variables` should match `layout`')

    if isinstance(source_like, QuantizedSourceAdapter):
        if source_like.layout != layout:
            raise ValueError('Quantized source and requested layout should match')
        return source_like, layout
    adapter = QuantizedSourceAdapter(
        source_like,
        layout,
        coordinate_map,
        domain,
        source_space=source_space,
        source_layout=source_layout,
        output_shape=None,
        dtype=dtype,
        device=device,
        computational_grid=computational_grid,
        out_of_domain=out_of_domain)
    return adapter, layout


class _QuantizedRSSMixin:
    """Converts physical RSS samples to digits before the ordinary workflow."""

    def __init__(self,
                 *args,
                 quantized_layout: QuantizedLayout,
                 quantized_adapter: QuantizedSourceAdapter,
                 sample_space: str = 'physical',
                 **kwargs) -> None:
        if not isinstance(quantized_layout, QuantizedLayout):
            raise TypeError('`quantized_layout` should be QuantizedLayout type')
        if not isinstance(quantized_adapter, QuantizedSourceAdapter):
            raise TypeError(
                '`quantized_adapter` should be QuantizedSourceAdapter type')
        if sample_space not in ('physical', 'digits'):
            raise ValueError(
                "`sample_space` should be 'physical' or 'digits'")
        self.quantized_layout = quantized_layout
        self.quantized_adapter = quantized_adapter
        self.sample_space = sample_space
        self._fit_sample_space = sample_space
        super().__init__(*args, **kwargs)

    def _normalize_samples(self, sketch_samples: Samples) -> ConfigurationBatch:
        """Normalizes already encoded digits or quantizes physical samples."""
        if self._fit_sample_space == 'digits':
            if isinstance(sketch_samples, ConfigurationBatch) and \
                    sketch_samples.kind != 'indices':
                raise ValueError(
                    'Digit sketch samples should use `kind="indices"`')
            return super()._normalize_samples(sketch_samples)

        if isinstance(sketch_samples, ConfigurationBatch):
            # TTRSS normalizes once before `_initialize_fit`, which validates
            # the resulting batch a second time. An index batch here is that
            # already-quantized internal representation.
            if sketch_samples.kind == 'indices':
                return super()._normalize_samples(sketch_samples)
            if sketch_samples.kind != 'coordinates' or \
                    not sketch_samples.packed:
                raise ValueError(
                    'Physical sketch samples should be packed coordinates')
            physical = sketch_samples.values
        else:
            physical = sketch_samples
        if not isinstance(physical, torch.Tensor) or physical.ndim != 2 or \
                physical.shape[1] != self.quantized_layout.n_variables:
            raise ValueError(
                'Physical sketch samples should have shape '
                '(samples, n_variables)')
        digits = self.quantized_adapter.physical_to_digits(
            physical.to(self.quantized_adapter.device))
        return super()._normalize_samples(ConfigurationBatch(
            digits, kind='indices'))

    def fit(self,
            sketch_samples: Samples,
            *args,
            sample_space: Optional[str] = None,
            **kwargs):
        """Fits QTT/QTR cores from physical coordinates or encoded digits."""
        active_space = self.sample_space if sample_space is None else sample_space
        if active_space not in ('physical', 'digits'):
            raise ValueError(
                "`sample_space` should be 'physical' or 'digits'")
        self._fit_sample_space = active_space
        try:
            result = super().fit(sketch_samples, *args, **kwargs)
        finally:
            self._fit_sample_space = self.sample_space
        physical_domain = self.quantized_adapter.domain
        if isinstance(physical_domain, torch.Tensor):
            physical_domain = physical_domain.detach().cpu()
        elif physical_domain is not None:
            physical_domain = tuple(
                value.detach().cpu() if isinstance(value, torch.Tensor)
                else value
                for value in physical_domain)
        result.metadata['quantization'] = {
            'n_variables': self.quantized_layout.n_variables,
            'base': self.quantized_layout.base,
            'level': self.quantized_layout.level,
            'ordering': self.quantized_layout.ordering,
            'digit_order': self.quantized_layout.digit_order,
            'sites': self.quantized_layout.sites(),
            'grid_size': self.quantized_layout.grid_size,
            'coordinate_map': type(
                self.quantized_adapter.coordinate_map).__name__,
            'domain': physical_domain,
            'computational_grid': (
                self.quantized_adapter.computational_grid),
            'out_of_domain': self.quantized_adapter.out_of_domain,
            'sample_space': active_space,
        }
        result.metadata['algorithm'] = self._quantized_algorithm
        return result


class _QuantizedTTRSS(_QuantizedRSSMixin, TTRSS):
    """Internal TTRSS specialization that normalizes physical QTT samples."""

    _quantized_algorithm = 'qtt_rss'


class TTRS:
    r"""Reusable Tensor Train Recursive Sketching problem.

    Unlike :class:`TTRSS`, this class projects the complete discrete source and
    therefore does not receive ``sketch_samples`` in :meth:`fit`. The source
    may be supplied directly, or built from a dataset as a normalized
    :class:`~tensorkrowch.decompositions.EmpiricalDistribution`. Sparse and
    empirical sources are contracted only on their declared non-zero support.

    The implementation follows the core-determining equations of
    `Generative modeling via tensor train sketching
    <https://arxiv.org/abs/2202.11788>`_ by Hur, Hoskins, Lindsey, Stoudenmire
    and Khoo (2022). ``sketch_operator`` controls how the recursive left and
    right sketches are constructed; the default is
    :meth:`MarginalSketch.markov`.

    The returned :class:`~tensorkrowch.decompositions.TTDecomposition` is a
    lightweight result. Its cores can initialize an
    :class:`~tensorkrowch.models.MPS` with
    ``tk.models.MPS(tensors=result.cores)``.

    Parameters
    ----------
    source : TensorSource, TTDecomposition, torch.Tensor or callable, optional
        Complete discrete scalar source. Exactly one of ``source`` and
        ``dataset`` is required. Sources without explicit support use a finite
        grid fallback until their structured backend is selected.
    dataset : torch.Tensor, optional
        Integer observations with shape ``(samples, sites)``. Duplicates are
        coalesced into an empirical distribution.
    input_dim : sequence of int, optional
        Complete dimensions. Required for callable sources and optional for a
        dataset, where it otherwise follows the largest observed indices.
    weights : torch.Tensor, optional
        Non-negative empirical mass per dataset row.
    sketch_operator : SketchOperator, optional
        Operator that provides a compatible :class:`SketchSystemBuilder`.
    dtype : torch.dtype, optional
        Empirical/callable value dtype.
    device : str or torch.device, optional
        Device for a dataset or callable source.
    output_device : str, torch.device or None, optional
        Device receiving completed cores. The default is CPU; ``None`` keeps
        the source device.

    Examples
    --------
    >>> dataset = torch.tensor([[0, 0], [0, 0], [1, 1]])
    >>> decomposer = TTRS(dataset=dataset, input_dim=(2, 2))
    >>> result = decomposer.fit(rank=2)
    >>> result.input_dim
    (2, 2)
    >>> len(result.cores)
    2
    """

    def __init__(
            self,
            source=None,
            *,
            dataset: Optional[torch.Tensor] = None,
            input_dim: Optional[Sequence[int]] = None,
            weights: Optional[torch.Tensor] = None,
            sketch_operator: Optional[SketchOperator] = None,
            dtype: Optional[torch.dtype] = None,
            device: Device = None,
            output_device: Device = 'cpu',
            synchronize_timers: bool = True) -> None:
        self._source = _resolve_rs_source(
            source=source,
            dataset=dataset,
            input_dim=input_dim,
            weights=weights,
            dtype=dtype,
            device=device)
        if sketch_operator is None:
            sketch_operator = MarginalSketch.markov()
        elif not isinstance(sketch_operator, SketchOperator):
            raise TypeError('`sketch_operator` should implement SketchOperator')
        self.sketch_operator = sketch_operator
        self.output_device = None if output_device is None \
            else torch.device(output_device)
        if not isinstance(synchronize_timers, bool):
            raise TypeError('`synchronize_timers` should be bool type')
        self.synchronize_timers = synchronize_timers

    @property
    def source(self) -> TensorSource:
        """Discrete source fixed for repeated independent fits."""
        return self._source

    @torch.no_grad()
    def fit(
            self,
            rank: _Rank = 1,
            *,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: Optional[int] = None,
            generator: Optional[torch.Generator] = None,
            strict_system: bool = False,
            warm_start: Optional[TTDecomposition] = None,
            verbose: Union[bool, int] = 0,
            collect_metrics: bool = False,
            observer: Optional[DecompositionObserver] = None
            ) -> TTDecomposition:
        r"""Projects the fixed source and solves its TT core equations.

        ``rank`` is one upper bound shared by all open links or one value per
        link. Each local Phi is trimmed with the same truncation policy before
        forming the next coefficient matrix. ``strict_system=True`` rejects a
        coefficient matrix that cannot identify all retained core columns.

        ``verbose`` ranges from 0 (silent), through 1 (sites and summary) and
        2 (system/timing details), to 3 (complete final cores). Metrics and
        synchronized timers are skipped unless ``collect_metrics=True``, an
        observer is supplied, or console output is requested.

        Parameters
        ----------
        rank : int or sequence of int, optional
            Shared maximum rank or one maximum per open TT link.
        cutoff : float, optional
            Minimum singular value to keep. It must be non-negative. Singular
            values ``<= cutoff`` are removed.
        atol : float, optional
            Absolute tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded while
            the accumulated sum of squares is ``<= atol``. It must be non-negative.
        rtol : float, optional
            Relative tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded while
            the tail sum of squares divided by the total sum of squares is
            ``<= rtol``. It must be in ``[0, 1]``.
        cum_percentage : float, optional
            Minimum fraction of squared singular-value mass to keep. Equivalent to
            setting ``rtol = 1 - cum_percentage``. It must be in ``[0, 1]``.

            .. math::

                \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
                cum\_percentage

        batch_size : int, optional
            Support or finite-grid evaluation batch size.
        generator : torch.Generator, optional
            Generator owned by this fit for randomized sketches.
        strict_system : bool, optional
            Rejects numerically rank-deficient coefficient systems.
        warm_start : TTDecomposition, optional
            Reserved for a future defined update. Non-``None`` values are
            rejected rather than reused implicitly.
        verbose : bool or int, optional
            Structured console verbosity from 0 to 3.
        collect_metrics : bool, optional
            Collects timing, truncation, local solves, source statistics and
            error over a declared sparse support.
        observer : DecompositionObserver, optional
            Additional structured-event consumer.

        Returns
        -------
        TTDecomposition
            Lightweight TT result stored on ``output_device``.

        Examples
        --------
        >>> indices = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        >>> values = torch.tensor([1., 2., 2., 4.])
        >>> source = tk.decompositions.SparseTensorSource(
        ...     indices, values, input_dim=(2, 2))
        >>> result = TTRS(
        ...     source,
        ...     sketch_operator=tk.decompositions.SampledSketch()).fit(rank=1)
        >>> result.rank
        [1]
        """
        if warm_start is not None:
            if not isinstance(warm_start, TTDecomposition):
                raise TypeError(
                    '`warm_start` should be TTDecomposition type or None')
            raise NotImplementedError(
                'TT-RS does not yet define a warm-start update; pass None')
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator type or None')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        verbosity = _normalize_verbosity(verbose)
        need_diagnostics = collect_metrics or bool(verbosity) or \
            observer is not None
        fit_observer = _resolve_observer(verbosity, observer) \
            if bool(verbosity) or observer is not None else None
        before_stats = getattr(self.source, 'evaluation_stats', None)

        if fit_observer is not None:
            fit_observer.emit(DecompositionEvent(
                name='start',
                phase='TT-RS',
                values={
                    'sites': len(self.source.input_dim),
                    'input_dim': self.source.input_dim,
                    'operator': type(self.sketch_operator).__name__,
                }))
        timer = None
        start = perf_counter() if need_diagnostics else None
        system = self.sketch_operator.builder(self.source).build(
            batch_size=batch_size,
            generator=generator)
        result = system.solve(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            strict_system=strict_system,
            collect_metrics=need_diagnostics)
        if start is not None:
            timer = perf_counter() - start
            result.metrics.timings.append(TimingRecord(
                name='fit', elapsed=timer))
        after_stats = getattr(self.source, 'evaluation_stats', None)
        if collect_metrics and before_stats is not None and after_stats is not None:
            result.metrics.evaluations.append(after_stats.delta(before_stats))

        metadata = dict(result.metadata)
        metadata.update({
            'algorithm': 'tt_rs',
            'input_dim': tuple(self.source.input_dim),
            'source_type': type(self.source).__name__,
        })
        active = TTDecomposition(
            result.cores,
            metrics=result.metrics,
            metadata=metadata)
        if collect_metrics and isinstance(self.source, SupportTensorSource):
            samples = self.source.support.as_tensor()
            approximation = active.evaluate(samples)
            target = self.source.support_values.to(
                device=approximation.device, dtype=approximation.dtype)
            absolute = torch.linalg.vector_norm(approximation - target)
            denominator = torch.linalg.vector_norm(target)
            relative = absolute / denominator if denominator > 0 else absolute
            active.metrics.errors.append(ErrorRecord(
                kind='source_support',
                absolute=absolute,
                relative=relative,
                denominator=denominator,
                size=target.shape[0]))

        if fit_observer is not None:
            for site, core in enumerate(active.cores):
                fit_observer.emit(DecompositionEvent(
                    name='site_complete',
                    phase='TT-RS',
                    site=site,
                    values={
                        'total_sites': len(active.cores),
                        'shape': tuple(core.shape),
                    }))
            fit_observer.emit(DecompositionEvent(
                name='summary',
                phase='TT-RS',
                values={
                    'rank': active.rank,
                    'operator': type(self.sketch_operator).__name__,
                    'elapsed': None if timer is None else f'{timer:.6f} s',
                }))
            for site, core in enumerate(active.cores):
                fit_observer.emit(DecompositionEvent(
                    name='core',
                    phase='TT-RS',
                    level=3,
                    site=site,
                    values={'shape': tuple(core.shape), 'tensor': core}))
            fit_observer.close(active.metrics)

        runtime = _RuntimePolicy(
            device=active.device,
            output_device=self.output_device,
            dtype=active.dtype,
            synchronize_timers=self.synchronize_timers)
        return TTDecomposition(
            [runtime.finalize(core) for core in active.cores],
            metrics=active.metrics,
            metadata=active.metadata)


@torch.no_grad()
def tt_rs(
        source=None,
        *,
        dataset: Optional[torch.Tensor] = None,
        input_dim: Optional[Sequence[int]] = None,
        weights: Optional[torch.Tensor] = None,
        sketch_operator: Optional[SketchOperator] = None,
        rank: _Rank = 1,
        cutoff: Optional[float] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        cum_percentage: Optional[float] = None,
        batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Device = None,
        generator: Optional[torch.Generator] = None,
        strict_system: bool = False,
        output_device: Device = 'cpu',
        verbose: Union[bool, int] = 0,
        return_info: bool = False):
    """Projects a complete discrete source into TT cores with TT-RS.

    This simple interface constructs :class:`TTRS`, calls :meth:`TTRS.fit`
    and returns only the cores unless ``return_info=True``. Exactly one of
    ``source`` and ``dataset`` is required; a dataset is normalized into an
    empirical distribution and is not interpreted as RSS sketch samples.

    Examples
    --------
    >>> dataset = torch.tensor([[0, 0], [0, 0], [1, 1]])
    >>> cores = tt_rs(dataset=dataset, input_dim=(2, 2), rank=2)
    >>> len(cores)
    2
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    result = TTRS(
        source=source,
        dataset=dataset,
        input_dim=input_dim,
        weights=weights,
        sketch_operator=sketch_operator,
        dtype=dtype,
        device=device,
        output_device=output_device).fit(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            batch_size=batch_size,
            generator=generator,
            strict_system=strict_system,
            verbose=verbose,
            collect_metrics=return_info)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


@torch.no_grad()
def qtt_rss(
        function=None,
        sketch_samples: Samples = None,
        *,
        source: Optional[TensorSource] = None,
        layout: Optional[QuantizedLayout] = None,
        n_variables: Optional[int] = None,
        base: Union[int, Sequence[int]] = 2,
        level: Union[int, Sequence[int]] = 1,
        ordering: str = 'grouped',
        digit_order: str = 'coarse_to_fine',
        permutation=None,
        coordinate_map: Optional[
            Union[CoordinateMap, Sequence[CoordinateMap]]] = None,
        domain: Domain = None,
        source_space: str = 'physical',
        source_layout: Optional[QuantizedLayout] = None,
        sample_space: str = 'physical',
        computational_grid: str = 'endpoints',
        out_of_domain: str = 'error',
        labels: Optional[torch.Tensor] = None,
        out_position: Optional[Union[int, Sequence[int]]] = None,
        rank: Optional[int] = None,
        cutoff: Optional[float] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        cum_percentage: Optional[float] = None,
        batch_size: int = 64,
        device: Device = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        random_projection: Optional[bool] = None,
        projection_dim: Optional[int] = None,
        projection_oversampling: int = 0,
        n_power_iter: int = 0,
        legacy_projection: bool = True,
        output_device: Device = 'cpu',
        verbose: Union[bool, int] = 0,
        return_info: bool = False):
    """Decomposes a multivariable physical function into QTT cores.

    Digit sites always use the corresponding basis embedding, so this API has
    no ``embedding`` argument. Physical samples with shape
    ``(samples, n_variables)`` are quantized by default; advanced callers can
    pass already encoded rows with ``sample_space="digits"``. Grouped and
    interleaved layouts each fit the physical function directly and are not
    interpreted as permutations of existing cores.

    Examples
    --------
    >>> samples = torch.tensor([[0.], [1 / 3], [2 / 3], [1.]])
    >>> cores = qtt_rss(
    ...     lambda x: 1 + x[:, 0],
    ...     samples,
    ...     n_variables=1,
    ...     base=2,
    ...     level=2,
    ...     domain=torch.tensor([0., 1.]),
    ...     rank=2)
    >>> len(cores)
    2
    """
    if sketch_samples is None:
        raise TypeError('`sketch_samples` should be provided')
    if layout is None and n_variables is None:
        if sample_space != 'physical':
            raise ValueError(
                '`n_variables` is required for digit-space samples')
        values = sketch_samples.values \
            if isinstance(sketch_samples, ConfigurationBatch) \
            else sketch_samples
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise ValueError(
                '`n_variables` could not be inferred from sketch samples')
        n_variables = values.shape[1]
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    decomposer = TTRSS.quantized(
        function=function,
        source=source,
        layout=layout,
        n_variables=n_variables,
        base=base,
        level=level,
        ordering=ordering,
        digit_order=digit_order,
        permutation=permutation,
        coordinate_map=coordinate_map,
        domain=domain,
        source_space=source_space,
        source_layout=source_layout,
        sample_space=sample_space,
        computational_grid=computational_grid,
        out_of_domain=out_of_domain,
        out_position=out_position,
        device=device,
        dtype=dtype,
        output_device=output_device)
    result = decomposer.fit(
        sketch_samples,
        labels=labels,
        rank=rank,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
        cum_percentage=cum_percentage,
        batch_size=batch_size,
        generator=generator,
        random_projection=random_projection,
        projection_dim=projection_dim,
        projection_oversampling=projection_oversampling,
        n_power_iter=n_power_iter,
        legacy_projection=legacy_projection,
        sample_space=sample_space,
        verbose=verbose,
        collect_metrics=return_info)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


@torch.no_grad()
def tt_rss(
        function: Callable,
        embedding: Embedding,
        sketch_samples: Samples,
        labels: Optional[torch.Tensor] = None,
        input_dim: Optional[Union[int, Sequence[int]]] = None,
        domain: Domain = None,
        domain_multiplier: int = 1,
        out_position: Optional[Union[int, Sequence[int]]] = None,
        rank: Optional[int] = None,
        cutoff: Optional[float] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        cum_percentage: Optional[float] = None,
        batch_size: int = 64,
        device: Device = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        random_projection: Optional[bool] = None,
        projection_dim: Optional[int] = None,
        projection_oversampling: int = 0,
        n_power_iter: int = 0,
        legacy_projection: bool = True,
        output_device: Device = 'cpu',
        verbose: Union[bool, int] = 1,
        return_info: bool = False
        ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], dict]]:
    r"""Decomposes a sampled scalar or tensor-valued function into a TT.

    The callable receives packed samples or a tuple with one coordinate tensor
    per input site. It may return a scalar batch with shape ``(batch_size,)``
    (the legacy ``(batch_size, 1)`` form is also accepted), or a tensor batch
    with shape ``(batch_size, *output_shape)``. Every tensor-output axis becomes
    a basis-embedded TT site. The returned OBC cores can be passed directly to
    :class:`~tensorkrowch.models.MPS`; callers may use the recorded output
    positions to interpret output sites.

    This compatibility function constructs :class:`TTRSS`, calls
    :meth:`TTRSS.fit`, and returns only its core list by default. Use the class
    directly to repeat fits while retaining the fixed problem definition.

    Parameters
    ----------
    function : callable
        Scalar- or vector-valued function to approximate.
    embedding : callable, torch.Tensor or sequence
        Shared site embedding, shared finite-domain table, or one entry per
        input site.
    sketch_samples : torch.Tensor, sequence of torch.Tensor or ConfigurationBatch
        Correlated sketch rows in packed or heterogeneous site form.
    labels : torch.Tensor, optional
        Flattened tensor-output labels with shape ``(batch_size,)``. If absent,
        labels are sampled proportionally to ``abs(function(samples)) ** 2``.
    input_dim : int or sequence of int, optional
        Expected embedding dimension, shared or specified per input site.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Shared domain or one finite domain per input site.
    domain_multiplier : int, optional
        Maximum inferred-domain size in multiples of ``input_dim``.
    out_position : int or sequence of int, optional
        Positions of the output sites. Defaults to an evenly spaced layout.
    rank : int, optional
        Number of singular values to keep.
    cutoff : float, optional
        Minimum singular value to keep. It must be non-negative. Singular
        values ``<= cutoff`` are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the accumulated sum of squares is ``<= atol``. It must be non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the tail sum of squares divided by the total sum of squares is
        ``<= rtol``. It must be in ``[0, 1]``.
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent to
        setting ``rtol = 1 - cum_percentage``. It must be in ``[0, 1]``.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
            cum\_percentage

    batch_size : int, optional
        Maximum source-evaluation batch size.
    device : str or torch.device, optional
        Device used to evaluate and decompose the function.
    dtype : torch.dtype, optional
        Dtype of source values and resulting cores.
    generator : torch.Generator, optional
        Controls output labels, inferred domains and random rotations.
    random_projection : bool, optional
        Explicitly enables randomized range projection or disables projection.
    projection_dim : int, optional
        Randomized output dimension; defaults to ``rank``.
    projection_oversampling : int, optional
        Extra dimensions used by the randomized range finder.
    n_power_iter : int, optional
        Power iterations used by the randomized range finder.
    legacy_projection : bool, optional
        Preserves the former square Haar rotation unless
        ``random_projection`` is explicitly supplied.
    output_device : str, torch.device or None, optional
        Device receiving final cores. The default is CPU; ``None`` keeps the
        compute device.
    verbose : bool or int, optional
        Verbosity from 0 (silent) to 3 (including final core tensors).
    return_info : bool, optional
        Also returns legacy ``total_time`` and ``val_eps`` keys together with
        structured result information.

    Returns
    -------
    list[torch.Tensor]
        TT cores stored on ``output_device``.
    tuple[list[torch.Tensor], dict]
        Cores and diagnostic information when ``return_info=True``.

    Examples
    --------
    >>> def function(data):
    ...     return data.prod(dim=1, keepdim=True)
    >>> def embedding(data):
    ...     return torch.stack([data, 1 - data], dim=-1)
    >>> sketch_samples = torch.rand(32, 3)
    >>> tensors = tk.decompositions.tt_rss(
    ...     function=function,
    ...     embedding=embedding,
    ...     sketch_samples=sketch_samples,
    ...     rank=2,
    ...     rtol=1e-2,
    ...     verbose=False)
    >>> len(tensors)
    3
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    start = perf_counter() if return_info else None
    decomposer = TTRSS(
        function=function,
        embedding=embedding,
        input_dim=input_dim,
        domain=domain,
        domain_multiplier=domain_multiplier,
        out_position=out_position,
        device=device,
        dtype=dtype,
        output_device=output_device)
    result = decomposer.fit(
        sketch_samples=sketch_samples,
        labels=labels,
        rank=rank,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
        cum_percentage=cum_percentage,
        batch_size=batch_size,
        generator=generator,
        random_projection=random_projection,
        projection_dim=projection_dim,
        projection_oversampling=projection_oversampling,
        n_power_iter=n_power_iter,
        legacy_projection=legacy_projection,
        verbose=verbose,
        collect_metrics=return_info)
    if not return_info:
        return result.cores

    info = result.as_info()
    info['total_time'] = perf_counter() - start
    info['val_eps'] = result.metadata.get('sample_error')
    return result.cores, info


__all__ = ['TTRSS', 'tt_rss', 'TTRS', 'tt_rs', 'qtt_rss']
