"""Tensor-train decompositions based on recursive sketching from samples."""

from math import prod
from time import perf_counter
from typing import (Callable, Dict, List, Optional, Sequence, Tuple, Union)
import warnings

import torch

from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.metrics import ErrorRecord
from tensorkrowch.decompositions.observers import DecompositionObserver
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
from tensorkrowch.decompositions.sketching.regions import (
    SiteRegion,
    _SamplePool,
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
Device = Optional[Union[str, torch.device]]


class TTRSS(RecursiveSketching):
    r"""Reusable Tensor Train Recursive Sketching from Samples problem.

    The fixed object stores the function (or :class:`TensorSource`), embedding,
    domain and output layout. Each :meth:`fit` receives a sample set and its
    truncation options, and returns a lightweight
    :class:`~tensorkrowch.decompositions.TTDecomposition`. The result cores can
    initialize an :class:`~tensorkrowch.models.MPS` directly with
    ``tk.models.MPS(tensors=result.cores)``.

    This first refactored implementation preserves the scalar/vector contract
    of the legacy routine. Site-dependent embeddings, tensor-valued outputs
    and the non-legacy projection options are activated in the next phase.
    Metadata that depends on ``sketch_samples`` is normalized independently in
    every fit, so repeating a fit does not retain mutable numerical state.

    Parameters
    ----------
    function : callable or TensorSource, optional
        Scalar- or vector-valued object to approximate. A callable receives a
        tensor with shape ``(batch_size, n_features)`` or
        ``(batch_size, n_features, in_dim)`` and returns shape
        ``(batch_size, output_dim)``. A scalar callable uses
        ``output_dim = 1``.
    embedding : callable
        Maps the input tensor to shape
        ``(batch_size, n_features, input_dim)``. Its last dimension becomes
        the input dimension of each TT core.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Finite values used to fit the embedding. One tensor is broadcast to
        every input site; a sequence supplies one domain per site. If omitted,
        each domain is inferred from the corresponding sketch samples.
    domain_multiplier : int, optional
        Maximum inferred-domain size in multiples of ``input_dim``.
    out_position : int, optional
        Position of the output core for a vector-valued function. The default
        splits the input sites into two groups as evenly as possible.
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
            embedding: Optional[Callable] = None,
            domain: Domain = None,
            domain_multiplier: int = 1,
            out_position: Optional[int] = None,
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
        if not callable(embedding):
            raise TypeError('`embedding` should be callable')
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

    @staticmethod
    def _validate_samples(sketch_samples: torch.Tensor) -> int:
        """Validates legacy packed samples and returns their site count."""
        if not isinstance(sketch_samples, torch.Tensor):
            raise TypeError('`sketch_samples` should be torch.Tensor type')
        if sketch_samples.ndim not in (2, 3):
            raise ValueError(
                '`sketch_samples` should have shape (batch_size, n_features) '
                'or (batch_size, n_features, in_dim)')
        if sketch_samples.shape[0] < 1:
            raise ValueError('`sketch_samples` should contain samples')
        if sketch_samples.shape[1] < 1:
            raise ValueError('`sketch_samples` should contain input sites')
        return sketch_samples.shape[1]

    def _domain_on_device(self, device: torch.device) -> Domain:
        """Moves the fixed domain to the active fit device."""
        if self._domain is None:
            return None
        if isinstance(self._domain, torch.Tensor):
            return self._domain.to(device)
        return tuple(value.to(device) for value in self._domain)

    def _legacy_embedding_for_site(
            self,
            device: torch.device,
            dtype: torch.dtype) -> Callable[[torch.Tensor], torch.Tensor]:
        """Adapts the legacy all-sites embedding to one input site."""
        def site_embedding(values: torch.Tensor) -> torch.Tensor:
            result = self._embedding(values.to(device).unsqueeze(1))
            if not isinstance(result, torch.Tensor):
                raise TypeError('`embedding` should return a torch.Tensor')
            if result.ndim != 3 or result.shape[0] != values.shape[0] or \
                    result.shape[1] != 1 or result.shape[2] < 1:
                raise ValueError(
                    '`embedding` should return shape '
                    '(batch_size, n_features, input_dim)')
            return result.squeeze(1).to(device=device, dtype=dtype)

        return site_embedding

    def _source_probe(
            self,
            samples: torch.Tensor,
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
            configurations = ConfigurationBatch(
                samples[:1].to(source.device), kind=self._input_kind)
            probe = source.evaluate(configurations)
            if self._dtype is not None and probe.dtype != self._dtype:
                raise ValueError('`source` and `dtype` should match')
            return probe, source

        device = torch.device('cpu') if self._device is None else self._device
        try:
            probe = self._source_like(samples[:1].to(device))
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
            sketch_samples: torch.Tensor,
            generator: Optional[torch.Generator]) -> torch.Tensor:
        """Normalizes all sample-dependent fixed objects for one fit."""
        n_input_sites = self._validate_samples(sketch_samples)
        probe, source = self._source_probe(sketch_samples, n_input_sites)
        if probe.ndim != 2 or probe.shape[0] != 1 or probe.shape[1] < 1:
            raise ValueError(
                '`function` should return shape (batch_size, output_dim)')
        if not (probe.is_floating_point() or probe.is_complex()):
            raise TypeError('`function` output should be floating or complex')
        if not torch.isfinite(probe).all():
            raise ValueError('`function` output should be finite')

        device = probe.device
        dtype = probe.dtype if self._dtype is None else self._dtype
        probe = probe.to(dtype=dtype)
        samples = sketch_samples.to(device)
        domains = _DomainSpec.normalize(
            self._domain_on_device(device), n_input_sites, samples=samples)
        site_embedding = self._legacy_embedding_for_site(device, dtype)
        embeddings = _EmbeddingSpec.normalize(site_embedding, domains)

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
            embeddings = _EmbeddingSpec.normalize(site_embedding, domains)

        out_position = self._out_position
        if probe.shape[1] == 1 and out_position is not None:
            warnings.warn(
                '`out_position` is ignored for a scalar function',
                stacklevel=3)
            out_position = None
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

    def _configuration_batch(self, samples: torch.Tensor) -> ConfigurationBatch:
        """Builds the source configurations for original input samples."""
        return ConfigurationBatch(samples, kind=self._input_kind)

    def _evaluate_samples(
            self,
            samples: torch.Tensor,
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
                samples.shape[0], 0, device=samples.device, dtype=torch.long)
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
            if flat_labels.shape[0] != samples.shape[0]:
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
            sketch_samples: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            rank: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: int = 64,
            generator: Optional[torch.Generator] = None,
            legacy_projection: bool = True,
            verbose: bool = False,
            collect_metrics: bool = False,
            observer: Optional[DecompositionObserver] = None
            ) -> TTDecomposition:
        r"""Decomposes the fixed function using correlated sketch samples.

        ``sketch_samples`` contains one sample coordinate per original input
        site. A vector output adds one basis-embedded output site to the final
        TT, but that site is not present in the supplied sample tensor. Every
        non-final cut fits its sampled input axis, optionally applies the
        compatibility random rotation, and calls :func:`truncated_svd` with
        all active truncation conditions combined.

        Parameters
        ----------
        sketch_samples : torch.Tensor
            Tensor with shape ``(batch_size, n_features)`` or
            ``(batch_size, n_features, in_dim)``.
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
        legacy_projection : bool, optional
            Whether to preserve the legacy square Haar rotation before every
            non-final SVD. ``False`` selects the configured range projector.
        verbose : bool, optional
            Emits the current high-level phase when ``True``.
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
        self._validate_samples(sketch_samples)
        if rank is None and cum_percentage is None:
            raise ValueError(
                'At least one of `rank` and `cum_percentage` should be given')
        if generator is not None and \
                not isinstance(generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator type or None')
        if not isinstance(legacy_projection, bool):
            raise TypeError('`legacy_projection` should be bool type')
        if not isinstance(verbose, bool):
            raise TypeError('`verbose` should be bool type')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                raise TypeError('`labels` should be torch.Tensor type')
            if labels.shape != sketch_samples.shape[:1]:
                raise ValueError(
                    '`labels` should have shape (batch_size,)')

        samples = self._initialize_fit(sketch_samples, generator)
        context = self._new_context(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            random_projection=not legacy_projection,
            batch_size=batch_size,
            generator=generator,
            collect_metrics=collect_metrics,
            verbose=verbose,
            observer=observer)
        context.state.update({
            'input_samples': samples,
            'labels': labels,
            'legacy_projection': legacy_projection,
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
            'out_position': (
                None if self.outputs.scalar else self.outputs.positions[0]),
            'legacy_projection': context.state['legacy_projection'],
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


@torch.no_grad()
def tt_rss(
        function: Callable,
        embedding: Callable,
        sketch_samples: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        domain: Domain = None,
        domain_multiplier: int = 1,
        out_position: Optional[int] = None,
        rank: Optional[int] = None,
        cutoff: Optional[float] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        cum_percentage: Optional[float] = None,
        batch_size: int = 64,
        device: Device = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        legacy_projection: bool = True,
        verbose: bool = True,
        return_info: bool = False
        ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], dict]]:
    r"""Decomposes a sampled scalar or vector function into a Tensor Train.

    The callable receives samples with shape ``(batch_size, n_features)`` or
    ``(batch_size, n_features, in_dim)``. Its embedding returns the same leading
    dimensions followed by ``input_dim``. Scalar functions return shape
    ``(batch_size, 1)``; vector functions add one basis-embedded output core at
    ``out_position``. The returned OBC cores can be passed directly to
    :class:`~tensorkrowch.models.MPS` (or
    :class:`~tensorkrowch.models.MPSLayer` for a vector output).

    This compatibility function constructs :class:`TTRSS`, calls
    :meth:`TTRSS.fit`, and returns only its core list by default. Use the class
    directly to repeat fits while retaining the fixed problem definition.

    Parameters
    ----------
    function : callable
        Scalar- or vector-valued function to approximate.
    embedding : callable
        Maps inputs to shape
        ``(batch_size, n_features, input_dim)``.
    sketch_samples : torch.Tensor
        Correlated sketch rows with shape ``(batch_size, n_features)`` or
        ``(batch_size, n_features, in_dim)``.
    labels : torch.Tensor, optional
        Flattened vector-output labels with shape ``(batch_size,)``. If absent,
        labels are sampled proportionally to ``abs(function(samples)) ** 2``.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Shared domain or one finite domain per input site.
    domain_multiplier : int, optional
        Maximum inferred-domain size in multiples of ``input_dim``.
    out_position : int, optional
        Position of a vector-output core. The default is centered.
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
    legacy_projection : bool, optional
        Preserves the former square Haar rotation when ``True``.
    verbose : bool, optional
        Prints high-level decomposition phases when ``True``.
    return_info : bool, optional
        Also returns legacy ``total_time`` and ``val_eps`` keys together with
        structured result information.

    Returns
    -------
    list[torch.Tensor]
        TT cores stored on CPU.
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
        domain=domain,
        domain_multiplier=domain_multiplier,
        out_position=out_position,
        device=device,
        dtype=dtype,
        output_device='cpu')
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
        legacy_projection=legacy_projection,
        verbose=verbose,
        collect_metrics=return_info)
    if not return_info:
        return result.cores

    info = result.as_info()
    info['total_time'] = perf_counter() - start
    info['val_eps'] = result.metadata.get('sample_error')
    return result.cores, info


__all__ = ['TTRSS', 'tt_rss']
