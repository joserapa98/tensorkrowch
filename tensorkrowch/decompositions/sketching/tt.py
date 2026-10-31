"""
This script contains:

    Classes:
        * TTRSS
        * TTRS

    Functions:
        * tt_rs
        * qtt_rss
        * tt_rss
"""

from dataclasses import replace
from math import prod
from typing import (Any,
                    Callable,
                    Dict,
                    Optional,
                    Sequence,
                    TYPE_CHECKING,
                    Tuple,
                    Union)

import torch

from tensorkrowch.utils import safe_divide

from tensorkrowch.decompositions._utils import (_normalize_dims,
                                                _normalize_positive_int,
                                                _normalize_rank)
from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord)
from tensorkrowch.decompositions.observers import (_normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import (TensorDecomposition,
                                                 TTDecomposition,
                                                 _result_from_source)
from tensorkrowch.decompositions.sketching.base import (RecursiveSketching,
                                                        _SketchingFitContext)
from tensorkrowch.decompositions.sketching.evaluations import (
    ConfigurationMap, _EvaluationPlanBuilder)
from tensorkrowch.decompositions.sketching.fitting import InputFitter
from tensorkrowch.decompositions.sketching.phi import PhiOperator
from tensorkrowch.decompositions.sketching.projections import RangeProjector
from tensorkrowch.decompositions.sketching.regions import (SiteRegion,
                                                           _SamplePool)
from tensorkrowch.decompositions.sketching.sketches import (MarginalSketch,
                                                            SketchOperator,
                                                            _prepare_phi)
from tensorkrowch.decompositions.sketching.sources import SupportTensorSource
from tensorkrowch.decompositions.sketching.specs import (_DomainSpec,
                                                         _EmbeddingSpec,
                                                         _OutputSpec,
                                                         _SketchingFitSpec,
                                                         _split_samples)
from tensorkrowch.decompositions.sources import (CallableSource,
                                                 ConfigurationBatch,
                                                 BaseSource, as_tensor_source)
from tensorkrowch.decompositions.sources.quantics import CallableQuanticsSource
from tensorkrowch.decompositions.sources.base import _tensor_view


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


if TYPE_CHECKING:
    from tensorkrowch.decompositions.sources.factory import SourceLike
    from tensorkrowch.formats.quantization import Domain


class TTRSS(RecursiveSketching):  # MARK: TTRSS
    """
    Reusable Tensor Train Recursive Sketching from Samples problem.

    The fixed object stores the function (or :class:`BaseSource`), embedding,
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

    Directional SVDs select the retained subspaces and absorb their singular
    values into the local tensors. Only the center keeps the isometric
    lifting maps needed to start both sketch recursions.

    ``source`` defines how inputs are evaluated. Accepted finite sources are
    :class:`~tensorkrowch.decompositions.DenseTensorSource`,
    :class:`~tensorkrowch.decompositions.SparseTensorSource`,
    :class:`~tensorkrowch.decompositions.FormatSource`,
    :class:`~tensorkrowch.decompositions.CallableTensorSource` and
    :class:`~tensorkrowch.decompositions.CallableQuanticsSource`; tensors
    are wrapped automatically.
    Wrap a non-Quantics format or :class:`~tensorkrowch.models.MPS` in
    ``FormatSource``. Continuous functions use
    :class:`~tensorkrowch.decompositions.CallableSource`
    or a raw callable and require an ``embedding``. Finite sources use basis
    embeddings by default. The dense, sparse and callable source classes
    provide ``.matrix(...)`` constructors for paired input/output dimensions;
    ``FormatSource`` obtains these from a matrix format. Structural batches
    are not supported.

    Parameters
    ----------
    source : BaseSource, torch.Tensor or callable
        Input values and their evaluation rules; continuous inputs also
        require an embedding.
    embedding : callable, torch.Tensor or sequence, optional
        Shared embedding or one per input site. Required for continuous
        sources. An embedding maps site values to vectors of size ``in_dim``.
    in_dim : int or sequence[int], optional
        Expected embedding dimensions; otherwise inferred from embeddings.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Domain values used to fit the embeddings, inferred from samples when
        omitted. Finite sources supply their index domains by default.
    domain_multiplier : int
        Maximum inferred-domain size in multiples of the embedding dimension.
    out_position : int or sequence[int], optional
        Positions of response sites in a vector chain. Matrix sources place
        each output site next to its input before combining their cores.
    device : str or torch.device, optional
        Evaluation device. Existing sources retain their device.
    dtype : torch.dtype, optional
        Declared function output dtype. Existing sources retain their dtype.
    out_device : str or torch.device, optional
        Device receiving completed cores; ``None`` retains the source device.
    input_fitter : InputFitter or sequence of InputFitter, optional
        Shared fitter or one per input/output axis in the expanded chain;
        inferred when omitted. Matrix pairs are combined after fitting.
    range_projector : RangeProjector, optional
        Complete range-projection strategy. ``None`` uses direct SVD.
    transform : callable, optional
        Called once with a :class:`ConfigurationMap`. Returns transformed
        values with the same shape as its initial ``values`` table. Extra
        raw source evaluations are available through its ``evaluate`` method.
    local_solver : LeastSquaresSolver, optional
        Solver for local core-determining equations.

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

    def __init__(self,
                 source: 'SourceLike',
                 embedding: Optional[Embedding] = None,
                 in_dim: Optional[Union[int, Sequence[int]]] = None,
                 domain: Domain = None,
                 domain_multiplier: int = 1,
                 out_position: Optional[Union[int, Sequence[int]]] = None,
                 *,
                 device: Device = None,
                 dtype: Optional[torch.dtype] = None,
                 out_device: Device = 'cpu',
                 input_fitter: Optional[Union[
                     InputFitter, Sequence[InputFitter]]] = None,
                 range_projector: Optional[RangeProjector] = None,
                 transform: Optional[Callable] = None,
                 local_solver: Optional[LeastSquaresSolver] = None) -> None:
        source_like = (as_tensor_source(source)
                       if isinstance(source, torch.Tensor) else source)
        if not isinstance(source_like, BaseSource):
            if not callable(source_like):
                raise TypeError(
                    '`source` should be a tensor source or callable')
        source = source_like
        if getattr(source, 'n_batches', 0):
            raise ValueError('Batched sources are not supported by RSS')
        if isinstance(source, BaseSource) and source.is_discrete_tensor and \
                embedding is None:
            in_dim = source._in_sites_dims
            embedding = tuple(
                torch.eye(dim, device=source.device) for dim in in_dim)
            domain = tuple(
                torch.arange(dim, device=source.device) for dim in in_dim)
        if embedding is None:
            raise TypeError('`embedding` should be provided')
        if not (callable(embedding) or isinstance(embedding, torch.Tensor)):
            if isinstance(embedding, (str, bytes)) or \
                    not isinstance(embedding, Sequence):
                raise TypeError(
                    '`embedding` should be callable, a tensor or a sequence')
            embedding = tuple(embedding)
            if not embedding or not all(
                    callable(entry) or isinstance(entry, torch.Tensor)
                    for entry in embedding):
                raise TypeError(
                    'Every `embedding` entry should be callable or a tensor')
        if in_dim is not None:
            in_dim = (_normalize_positive_int(in_dim, 'in_dim')
                      if isinstance(in_dim, int) else
                      _normalize_dims(in_dim, 'in_dim'))
        if (domain is not None) and not isinstance(domain, torch.Tensor):
            if isinstance(domain, (str, bytes)) or \
                    not isinstance(domain, Sequence):
                raise TypeError(
                    '`domain` should be a tensor or a sequence of tensors')
            domain = tuple(domain)
            if not all(isinstance(value, torch.Tensor) for value in domain):
                raise TypeError(
                    '`domain` should be a tensor or a sequence of tensors')
        if isinstance(domain_multiplier, bool) or \
                not isinstance(domain_multiplier, int):
            raise TypeError('`domain_multiplier` should be int type')
        if domain_multiplier < 1:
            raise ValueError('`domain_multiplier` should be positive')
        if (dtype is not None) and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type or None')
        if (local_solver is not None) and \
                not isinstance(local_solver, LeastSquaresSolver):
            raise TypeError(
                '`local_solver` should be LeastSquaresSolver type or None')

        self._source_like = source_like
        self._source = source_like
        self._embedding = embedding
        self._in_dim_option = in_dim
        self._domain = domain
        self._domain_multiplier = domain_multiplier
        self._out_position = (
            tuple(2 * site + 1 for site in range(source.n_sites))
            if getattr(source, 'is_matrix', False) else out_position)
        self._device = None if device is None else \
            torch.empty(0, device=device).device
        self._dtype = dtype
        self._out_device = out_device
        self._input_fitter_option = input_fitter
        self._range_projector_option = range_projector
        self._transform_option = transform
        self._local_solver = LeastSquaresSolver() \
            if local_solver is None else local_solver
        self._input_kind = 'features'

    def _normalize_samples(self,
                           sketch_samples: Samples) -> ConfigurationBatch:
        """Normalizes packed or heterogeneous samples for the fixed source."""
        source = self._source_like
        kind = getattr(source, 'input_kind', 'features')
        if isinstance(source, CallableQuanticsSource) and not (
                isinstance(sketch_samples, ConfigurationBatch) and
                (sketch_samples.kind == 'indices')):
            coordinates = sketch_samples.values if isinstance(
                sketch_samples, ConfigurationBatch) else sketch_samples
            indices = source.coordinate_map.to_indices(coordinates)
            sketch_samples = ConfigurationBatch(
                source.layout.encode_indices(indices))
        if isinstance(sketch_samples, ConfigurationBatch):
            samples = sketch_samples
            if samples.kind != kind:
                raise ValueError(
                    f'`sketch_samples.kind` should be {kind!r}')
        else:
            samples = ConfigurationBatch(sketch_samples, kind=kind)
        if samples.batch_size < 1:
            raise ValueError('`sketch_samples` should contain samples')
        if isinstance(self._source_like, BaseSource) and \
                (samples.n_inputs != len(self._source_like.in_sites)):
            raise ValueError(
                '`source` and `sketch_samples` should share input sites')
        return samples

    def _domain_on_device(self, device: torch.device) -> Domain:
        """Moves the fixed domain to the active fit ``device``."""
        if self._domain is None:
            return None
        if isinstance(self._domain, torch.Tensor):
            return self._domain.to(device)
        return tuple(value.to(device) for value in self._domain)

    def _embeddings_on_device(self,
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

            def site_embedding(values: torch.Tensor,
                               function: Callable = entry) -> torch.Tensor:
                result = function(values.to(device))
                if not isinstance(result, torch.Tensor):
                    raise TypeError('`embedding` should return a torch.Tensor')
                return result.to(device=device, dtype=dtype)

            normalized.append(site_embedding)
        return tuple(normalized)

    def _validate_in_dim(self, in_dim: Sequence[int]) -> None:
        """Checks an optional public input-dimension declaration."""
        expected = self._in_dim_option
        if expected is None:
            return
        expected = (expected,) * len(in_dim) \
            if isinstance(expected, int) else tuple(expected)
        if len(expected) != len(in_dim):
            raise ValueError(
                '`in_dim` should contain one value per input site')
        if tuple(in_dim) != expected:
            raise ValueError(
                '`in_dim` should match the dimensions returned by '
                '`embedding`')

    def _source_probe(self,
                      samples: ConfigurationBatch,
                      n_input_sites: int) -> Tuple[torch.Tensor, BaseSource]:
        """Evaluates one row using the fixed source's input convention."""
        source = self._source_like
        ids = torch.zeros(1, dtype=torch.long, device=samples.device)
        probe_batch = samples.index_select(ids)
        probe = None
        if not isinstance(source, BaseSource):
            probe_batch = probe_batch.to(self._device or 'cpu')
            probe = source(probe_batch.values)
            if not isinstance(probe, torch.Tensor):
                raise TypeError('`function` should return a torch.Tensor')
            if (probe.ndim < 1) or (probe.shape[0] != 1):
                raise ValueError(
                    '`function` should preserve the leading batch dimension')
            source = CallableSource(
                source, n_input_sites + probe.ndim - 1,
                in_sites=tuple(range(n_input_sites)),
                out_shape=probe.shape[1:],
                dtype=self._dtype, device=self._device or 'cpu')
            probe = source._validate_result(probe, 1)
            source._record_evaluation(points=1)
            self._source_like = source
        if len(source.in_sites) != n_input_sites:
            raise ValueError(
                '`source` and `sketch_samples` should share input sites')
        if (self._device is not None) and (source.device != self._device):
            raise ValueError('`source` and `device` should match')
        self._input_kind = getattr(source, 'input_kind', 'indices')
        if probe is None:
            probe = source.evaluate(probe_batch.to(source.device))
        if (self._dtype is not None) and (probe.dtype != self._dtype):
            raise ValueError('`source` and `dtype` should match')
        return probe, source

    def _initialize_fit(self,
                        sketch_samples: Samples,
                        generator: Optional[torch.Generator]
                        ) -> ConfigurationBatch:
        """Normalizes all sample-dependent fixed objects for one fit."""
        if (generator is not None) and not isinstance(generator,
                                                      torch.Generator):
            raise TypeError('`generator` should be torch.Generator or None')
        sample_batch = self._normalize_samples(sketch_samples)
        n_input_sites = sample_batch.n_inputs
        probe, source = self._source_probe(sample_batch, n_input_sites)
        if (probe.ndim < 1) or (probe.shape[0] != 1):
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
        self._validate_in_dim(embeddings.in_dim)

        if domains.inferred:
            inferred_values = []
            for values, in_dim in zip(
                    domains.values, embeddings.in_dim):
                maximum = self._domain_multiplier * in_dim
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

        RecursiveSketching.__init__(
            self,
            source=source,
            domains=domains,
            embeddings=embeddings,
            outputs=outputs,
            input_fitter=self._input_fitter_option,
            range_projector=self._range_projector_option,
            transform=self._transform_option,
            out_device=self._out_device)
        return samples

    def _configuration_batch(self, samples: Samples) -> ConfigurationBatch:
        """Builds the source configurations for original input ``samples``."""
        if isinstance(samples, ConfigurationBatch):
            return samples
        return ConfigurationBatch(samples, kind=self._input_kind)

    def _evaluate_samples(self,
                          samples: Samples,
                          batch_size: int) -> torch.Tensor:
        """
        Evaluates original ``samples`` in deterministic contiguous batches.
        """
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
            (not self.outputs.scalar and (labels is None)) or
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

    def fit(self,
            sketch_samples: Samples,
            labels: Optional[torch.Tensor] = None,
            rank: Optional[_Rank] = None,
            *,
            center: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: int = 64,
            generator: Optional[torch.Generator] = None,
            verbose: Union[bool, int] = 0,
            collect_metrics: bool = False) -> TTDecomposition:
        """
        Fits a TT from correlated samples, starting at ``center``.

        All local queries are deduplicated and evaluated together. Each Phi
        then fits its physical axis and selects an outward singular subspace:
        both directions at the center, left on its left, right on its right.
        Spectra stay outside these bases. The cores are connected by local
        least-squares equations without a final rounding operation.

        Parameters
        ----------
        sketch_samples : torch.Tensor, sequence or ConfigurationBatch
            Packed inputs with shape ``(batch_size, n_inputs, ...)``, or one
            tensor with leading batch dimension per source input.
        labels : torch.Tensor, optional
            Flattened output-component indices with shape ``(batch_size,)``.
            When omitted, tensor outputs select indices with probabilities
            proportional to squared magnitudes. Scalar outputs need no labels.
        rank : int or sequence[int], optional
            Shared maximum rank or one per open TT bond. ``None`` discovers
            numerical ranks under the truncation criteria. Matrix sources
            specify one rank between consecutive input/output pairs; each
            internal input/output split retains its full local capacity.
        center : int, optional
            Initial site in the final chain, including output sites. Defaults
            to the middle. An endpoint gives a single outward branch. Matrix
            inputs and outputs occupy separate sites before pairing cores.
        cutoff : float, optional
            Minimum singular value to keep. Values ``<= cutoff`` are removed.
        atol : float, optional
            Maximum discarded sum of squared singular values.
        rtol : float, optional
            Maximum discarded sum of squared singular values divided by their
            total sum of squares. Must lie in ``[0, 1]``.
        cum_percentage : float, optional
            Minimum retained squared singular-value fraction. Equivalent to
            ``rtol=1-cum_percentage``.
        batch_size : int
            Maximum number of unique configurations evaluated together.
        generator : torch.Generator, optional
            Generator for domain subsampling, labels and randomized projectors.
        verbose : bool or int
            Level from 0 to 3: silent, progress, local details, core tensors.
        collect_metrics : bool
            Retains evaluation, fitting, truncation, solve and timing records.

        Returns
        -------
        TTDecomposition
            TT result, or the corresponding matrix/Quantics result when the
            source declares that structure. Cores are stored on ``out_device``.

        Examples
        --------
        >>> tensor = torch.arange(16.).reshape(2, 2, 2, 2)
        >>> samples = torch.cartesian_prod(*[torch.arange(2)] * 4)
        >>> result = tk.decompositions.TTRSS(tensor).fit(
        ...     samples, rank=2, center=2)
        >>> result.in_dim
        (2, 2, 2, 2)
        """
        samples = self._initialize_fit(sketch_samples, generator)
        count = self.outputs.n_sites
        center = self._normalize_center(center, count)
        ranks = self._chain_rank(rank)
        self._validate_labels(labels, samples.batch_size)
        context = self._new_context(
            cutoff=cutoff, atol=atol, rtol=rtol,
            cum_percentage=cum_percentage, batch_size=batch_size,
            generator=generator, collect_metrics=collect_metrics,
            verbose=verbose)
        context.state.update(input_samples=samples, labels=labels,
                             rank=ranks, center=center)
        return self._execute(context)

    def _chain_rank(self, rank: Optional[_Rank], *, cyclic: bool = False
                    ) -> Optional[Tuple[int, ...]]:
        """Expands matrix bond caps to the interleaved input/output chain."""
        if rank is None:
            return None
        if not getattr(self._source_like, 'is_matrix', False):
            count = self.outputs.n_sites
            return _normalize_rank(rank, count if cyclic else count - 1)

        count = self.outputs.n_input_sites
        ranks = _normalize_rank(rank, count if cyclic else count - 1)
        left = (ranks[-1], *ranks[:-1]) if cyclic else (1, *ranks)
        right = ranks if cyclic else (*ranks, 1)
        expanded = []
        for site, (in_dim, out_dim) in enumerate(zip(
                self.embeddings.in_dim, self.outputs.out_shape)):
            expanded.append(min(left[site] * in_dim, right[site] * out_dim))
            if cyclic or (site < count - 1):
                expanded.append(right[site])
        return tuple(expanded)

    @staticmethod
    def _normalize_center(center: Optional[int], n_sites: int) -> int:
        """Selects the middle or validates an explicitly selected site."""
        if center is None:
            return n_sites // 2
        if isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int or None')
        if (center < 0) or (center >= n_sites):
            raise ValueError('`center` should identify a final site')
        return center

    @staticmethod
    def _validate_labels(labels: Optional[torch.Tensor], size: int) -> None:
        """Checks output-component indices before planning source queries."""
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                raise TypeError('`labels` should be torch.Tensor or None')
            if labels.shape != (size,):
                raise ValueError('`labels` should have shape (batch_size,)')

    def _build_regions(self,
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

    def _build_phi(self,
                   site: int,
                   regions: Dict[str, object],
                   context: _SketchingFitContext) -> PhiOperator:
        """
        Builds the prefix/current/suffix Phi operator for one TT ``site``.
        """
        components = []
        prefix = regions['prefixes'][site]
        if len(prefix.region):
            components.append(prefix)

        kind, axis = self.outputs.layout[site]
        values = self.domains.for_site(axis) if kind == 'input' \
            else torch.arange(
                self.outputs.out_shape[axis], device=self.source.device)
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

    def _prepare_phis(self, context: _SketchingFitContext) -> list:
        """Evaluates once, then fits and trims one Phi at a time."""
        count, center = self.outputs.n_sites, context.state['center']
        phis = [self._build_phi(site, context.regions, context)
                for site in range(count)]
        builder = _EvaluationPlanBuilder(self._source)
        handles = [phi.collect(builder) for phi in phis]
        configurations = ConfigurationMap(
            builder.freeze(), batch_size=context.spec.batch_size,
            transform=self.transform)
        with context.phase('source.evaluate'):
            configurations.evaluate_source()
        with context.phase('values.transform'):
            configurations.prepare_values()
        if context.collect_metrics:
            context.metrics.evaluations.append(configurations.stats)
            if self.transform is not None:
                context.state['configuration_map'] = configurations

        dimensions = self.outputs.site_dim(self.embeddings)
        ranks = context.state['rank']
        options = context.spec.truncation.as_kwargs()
        options.pop('rank')
        if (ranks is None) and all(value is None
                                   for value in options.values()):
            scale = max(max(phi.shape) for phi in phis)
            options['rtol'] = (scale * torch.finfo(
                self._source.dtype).eps) ** 2
        prepared = []
        for site, (phi, handle) in enumerate(zip(phis, handles)):
            phi._bind(configurations, handle)
            axis = phi.layout.index(site)
            tensor = self._fit_input_axis(site, phi, axis, context)
            # Empty prefix/suffix axes are implicit in the query layout.
            if site == 0:
                tensor = tensor.unsqueeze(0)
            if site == count - 1:
                tensor = tensor.unsqueeze(-1)
            phi.tensor = None
            direction = ('center' if site == center else
                         'left' if site < center else 'right')
            if ((site == 0) or (site == count - 1)) and (site != center):
                direction = None
            left_cap = min(prod(dimensions[:site]),
                           prod(dimensions[site:]))
            right_cap = min(prod(dimensions[:site + 1]),
                            prod(dimensions[site + 1:]))
            if ranks is not None:
                if site:
                    left_cap = min(left_cap, ranks[site - 1])
                if site < count - 1:
                    right_cap = min(right_cap, ranks[site])
            # Matrix input/output splits have capacity caps but no prescribed
            # public rank. Discover their support without retaining null axes.
            matrix = getattr(self._source_like, 'is_matrix', False)
            numerical = (matrix and (site % 2 == 1),
                         matrix and (site % 2 == 0))
            with context.phase('svd.trim', site=site):
                item = _prepare_phi(
                    tensor, direction, rank_left=left_cap,
                    rank_right=right_cap, **options,
                    numerical=numerical,
                    projector=self._range_projector,
                    generator=context.generator, site=site,
                    collect_metrics=context.need_diagnostics)
            if context.collect_metrics:
                context.metrics.truncations.extend(
                    record for record in item.truncations
                    if record.site < count - 1)
                context.metrics.range_projections.extend(item.projections)
            prepared.append(item)

        # Custom fitters can return different coefficient dtypes. Include the
        # embeddings' source dtype before any recursive contractions.
        dtype = self._source.dtype
        for item in prepared:
            dtype = torch.promote_types(dtype, item.tensor.dtype)
        prepared = [replace(
            item, tensor=item.tensor.to(dtype),
            left=None if item.left is None else item.left.to(dtype),
            right=None if item.right is None else item.right.to(dtype))
            for item in prepared]

        # No branch reconstructs a factor outside the first or last site.
        initial = prepared[center]
        if center == 0:
            initial = replace(initial, tensor=torch.einsum(
                'ab,bpc->apc', initial.left, initial.tensor),
                left=torch.ones_like(initial.left))
        if center == count - 1:
            initial = replace(initial, tensor=torch.einsum(
                'apb,bc->apc', initial.tensor, initial.right),
                right=torch.ones_like(initial.right))
        prepared[center] = initial
        context.state['fitted_site_dim'] = tuple(
            item.tensor.shape[1] for item in prepared)
        return prepared

    def _recursion_matrix(self, site: int, tensor: torch.Tensor,
                          direction: str,
                          context: _SketchingFitContext) -> torch.Tensor:
        """Evaluates a retained tensor on the next correlated sketch rows."""
        if direction == 'right':
            recursion = context.regions['prefixes'][site].recursive_projector(
                context.regions['prefixes'][site + 1])
            selected = tensor.index_select(0, recursion.gather.to(
                tensor.device))
            vectors = self.outputs.embed_site(
                site, recursion.new_values[0], self.embeddings,
                dtype=tensor.dtype).to(device=tensor.device,
                                       dtype=tensor.dtype)
            return torch.einsum('mpr,mp->mr', selected, vectors)
        recursion = context.regions['suffixes'][site + 1].recursive_projector(
            context.regions['suffixes'][site])
        selected = tensor.index_select(2, recursion.gather.to(tensor.device))
        vectors = self.outputs.embed_site(
            site, recursion.new_values[0], self.embeddings,
            dtype=tensor.dtype).to(device=tensor.device, dtype=tensor.dtype)
        return torch.einsum('lpm,mp->lm', selected, vectors)

    def _solve_tt(self, prepared: list, context: _SketchingFitContext,
                  *, report: bool = True) -> list:
        """Connects outward tensors, preserving the center's lifting maps."""
        count, center = len(prepared), context.state['center']
        cores = [None] * count
        initial = prepared[center]
        cores[center] = initial.tensor
        if report:
            context.complete(center, cores[center], count)
        for direction, sites in (
                ('right', range(center + 1, count)),
                ('left', range(center - 1, -1, -1))):
            previous = (torch.einsum('ab,bpc->apc', initial.left,
                                     initial.tensor)
                        if direction == 'right' else torch.einsum(
                            'apb,bc->apc', initial.tensor, initial.right))
            for site in sites:
                prior_site = site - 1 if direction == 'right' else site + 1
                environment = self._recursion_matrix(
                    prior_site, previous, direction, context)
                target = prepared[site].tensor
                with context.phase('core.solve', site=site):
                    if direction == 'right':
                        matrix = target.reshape(target.shape[0], -1)
                        solution, record = self._local_solver.solve(
                            environment, matrix, site=site,
                            return_record=context.need_diagnostics)
                        core = solution.reshape(
                            environment.shape[1], *target.shape[1:])
                    else:
                        matrix = target.reshape(-1, target.shape[-1])
                        solution, record = self._local_solver.solve(
                            environment.T, matrix.T, site=site,
                            return_record=context.need_diagnostics)
                        core = solution.T.reshape(
                            *target.shape[:-1], environment.shape[0])
                if context.collect_metrics and (record is not None):
                    context.metrics.local_solves.append(record)
                cores[site], previous = core, target
                if report:
                    context.complete(site, core, count)
        # Formats omit the unit dimensions at open boundaries.
        if count == 1:
            return [cores[0].reshape(-1)]
        return [cores[0].squeeze(0), *cores[1:-1], cores[-1].squeeze(-1)]

    def _decompose(self, context: _SketchingFitContext) -> TTDecomposition:
        """Runs common preparation followed by centered TT alignment solves."""
        prepared = self._prepare_phis(context)
        context.cores.extend(self._solve_tt(prepared, context))
        return self._assemble_result(context)

    def _error_target(self, context: _SketchingFitContext
                      ) -> Tuple[Sequence[torch.Tensor], torch.Tensor, str]:
        """Returns diagnostic points and values in the fitted target metric."""
        if self.transform is None:
            return (context.state['extended_samples'],
                    context.state['selected_values'], 'sketch_samples')

        # A generic table transform need not define values at arbitrary new
        # points. Validate on its initial table, including every output entry.
        configurations = context.state['configuration_map']
        inputs = configurations._plan.configurations
        values = self.outputs.validate_values(configurations.prepare_values())
        if self.outputs.scalar:
            return (_split_samples(inputs, self.outputs.n_input_sites),
                    values, 'configuration_map')
        n_outputs = prod(self.outputs.out_shape)
        ids = torch.arange(inputs.batch_size, device=inputs.device)
        inputs = inputs.index_select(ids.repeat_interleave(n_outputs))
        labels = torch.arange(n_outputs, device=inputs.device).repeat(
            values.shape[0])
        points = self.outputs.insert_indices(
            inputs, self.outputs.unflatten_labels(labels))
        return points, values.reshape(-1), 'configuration_map'

    def _evaluate_extended_cores(self,
                                 cores: Sequence[torch.Tensor],
                                 samples: Sequence[torch.Tensor],
                                 context: _SketchingFitContext
                                 ) -> torch.Tensor:
        """Evaluates TT ``cores`` on already output-extended sample rows."""
        vectors = [
            self.outputs.embed_site(
                site, values,
                self.embeddings, dtype=cores[0].dtype).to(
                    device=cores[0].device, dtype=cores[0].dtype)
            for site, values in enumerate(samples)
        ]
        if len(cores) == 1:
            return torch.einsum('bp,p->b', vectors[0], cores[0])

        state = torch.einsum('bp,pr->br', vectors[0], cores[0])
        for vector, core in zip(vectors[1:-1], cores[1:-1]):
            state = torch.einsum('bl,lpr,bp->br', state, core, vector)
        return torch.einsum(
            'bl,lp,bp->b', state, cores[-1], vectors[-1])

    def _assemble_result(self,
                         context: _SketchingFitContext) -> TTDecomposition:
        """Adds optional sample error and moves completed cores for storage."""
        metadata = {
            'algorithm': 'tt_rss',
            'out_shape': tuple(self.outputs.out_shape),
            'out_position': None if self.outputs.scalar else (
                self.outputs.positions[0]
                if self.outputs.n_output_sites == 1
                else tuple(self.outputs.positions)),
            'center': context.state['center'],
            'core_shapes': [tuple(core.shape) for core in context.cores],
        }
        if context.collect_metrics:
            points, selected, kind = self._error_target(context)
            approximation = self._evaluate_extended_cores(
                context.cores, points, context)
            selected = selected.to(device=approximation.device)
            absolute = torch.linalg.vector_norm(approximation - selected)
            denominator = torch.linalg.vector_norm(selected)
            relative = safe_divide(
                absolute, denominator,
                zero_value=torch.where(absolute == 0, 0, torch.inf))
            context.metrics.errors.append(ErrorRecord(
                kind=kind,
                absolute=absolute,
                relative=relative,
                size=selected.shape[0],
                denominator=denominator))
            metadata['sample_error'] = float(relative.detach().cpu())
            metadata['error_target'] = kind

        if context.collect_metrics and context.metrics.truncations:
            local_squared = sum(
                record.discarded_sq_norm
                for record in context.metrics.truncations)
            local_aggregate = local_squared ** 0.5
            context.metrics.errors.append(ErrorRecord(
                kind='sketch_svd_local_aggregate',
                absolute=local_aggregate,
                size=len(context.metrics.truncations)))
            metadata['sketch_svd_local_aggregate'] = local_aggregate

        context.metrics.local_solves.sort(key=lambda item: item.site)
        context.metrics.truncations.sort(key=lambda item: item.site)
        cores = [context.runtime.finalize(core) for core in context.cores]
        result = TTDecomposition(
            cores=cores,
            in_sites=self.outputs.input_positions,
            metrics=context.metrics,
            metadata=metadata)
        return _result_from_source(result, self._source_like) if \
            getattr(self._source_like, 'is_matrix', False) or isinstance(
                self._source_like, CallableQuanticsSource) else result

    def _validate_result(self,
                         result: TensorDecomposition,
                         context: _SketchingFitContext) -> None:
        """Checks the TT topology and final site dimensions."""
        if not isinstance(result, TensorDecomposition) or result._cyclic:
            raise TypeError(
                '`result` should be a decomposition with TT format')
        if result.out_dim is not None:
            dimensions = tuple(
                dim for pair in zip(result.in_dim, result.out_dim)
                for dim in pair)
            if dimensions != self.outputs.site_dim(self.embeddings):
                raise ValueError(
                    'The result matrix dimensions are inconsistent')
            return
        if len(result.cores) != self.outputs.n_sites:
            raise ValueError('The result should contain one core per TT site')
        expected = context.state.get(
            'fitted_site_dim', self.outputs.site_dim(self.embeddings))
        if result.in_dim != expected:
            raise ValueError('The result input dimensions are inconsistent')


class TTRS:  # MARK: TTRS
    """
    Projects a finite source and solves centered TT core equations.

    Unlike :class:`TTRSS`, TT-RS contracts a discrete source with recursive
    sketch operators, rather than taking pointwise sketch samples in ``fit``.
    Sparse sources are contracted on their declared nonzero support;
    :class:`~tensorkrowch.decompositions.FormatSource` uses structured
    contractions when available. Matrix and Quantics sources retain their
    result family. Structural batches are not supported.

    The core-determining equations follow `Generative modeling via tensor
    train sketching <https://arxiv.org/abs/2202.11788>`_. Starting from an
    interior center and transforming individual Phi tensors are extensions;
    a generic transform need not preserve a common underlying tensor.

    Parameters
    ----------
    source : BaseSource, TensorFormat1D, MPS or torch.Tensor
        Finite tensor input, with dimensions and evaluation defined by the
        source. Continuous callables require discretization first.
    sketch_operator : SketchOperator, optional
        Recursive projections. Defaults to ``MarginalSketch()``.
    transform : callable, optional
        Applied independently to each three-axis Phi before truncation. Must
        return a finite floating-point or complex tensor with the same shape
        and device; dtype promotion is allowed. Mathematical compatibility
        between transformed Phi tensors is the user's choice.
    range_projector : RangeProjector, optional
        Complete range-projection strategy. ``None`` uses direct SVD.
    out_device : str or torch.device, optional
        Device receiving completed cores; ``None`` retains the source device.

    Examples
    --------
    >>> indices = torch.tensor([[0, 0, 0], [1, 1, 1]])
    >>> source = tk.decompositions.SparseTensorSource.from_data(
    ...     indices, in_dim=(2, 2, 2))
    >>> result = tk.decompositions.TTRS(
    ...     source, sketch_operator=tk.decompositions.SampledSketch()).fit(2)
    >>> result.in_dim
    (2, 2, 2)
    """

    def __init__(self, source: 'SourceLike', *,
                 sketch_operator: Optional[SketchOperator] = None,
                 transform: Optional[Callable] = None,
                 range_projector: Optional[RangeProjector] = None,
                 out_device: Device = 'cpu') -> None:
        self._source = as_tensor_source(source)
        self._tensor_source = _tensor_view(self._source)
        if self._source.n_batches:
            raise ValueError('Batched sources are not supported by RS')
        if sketch_operator is None:
            sketch_operator = MarginalSketch()
        if not isinstance(sketch_operator, SketchOperator):
            raise TypeError(
                '`sketch_operator` should implement SketchOperator')
        if (transform is not None) and not callable(transform):
            raise TypeError('`transform` should be callable or None')
        if (range_projector is not None) and not isinstance(
                range_projector, RangeProjector):
            raise TypeError(
                '`range_projector` should implement RangeProjector')
        self.sketch_operator = sketch_operator
        self.transform = transform
        self.range_projector = range_projector
        self._runtime = _RuntimePolicy(
            device=self._source.device, out_device=out_device,
            dtype=self._source.dtype)

    @property
    def source(self) -> BaseSource:
        """Discrete source fixed for repeated independent fits."""
        return self._source

    @torch.no_grad()
    def fit(self, rank: Optional[_Rank] = None, *,
            center: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: Optional[int] = None,
            generator: Optional[torch.Generator] = None,
            strict_system: bool = False,
            verbose: Union[bool, int] = 0,
            collect_metrics: bool = False) -> TTDecomposition:
        """
        Builds local Phi tensors and solves the two outward branches.

        Parameters
        ----------
        rank : int or sequence[int], optional
            Shared maximum rank or one per open TT bond. ``None`` discovers
            numerical ranks under the truncation criteria.
        center : int, optional
            Initial site. Defaults to the middle. An endpoint gives a single
            outward branch.
        cutoff : float, optional
            Minimum singular value to keep. Values ``<= cutoff`` are removed.
        atol : float, optional
            Maximum discarded sum of squared singular values.
        rtol : float, optional
            Maximum discarded sum of squared singular values divided by their
            total sum of squares. Must lie in ``[0, 1]``.
        cum_percentage : float, optional
            Minimum retained squared singular-value fraction. Equivalent to
            ``rtol=1-cum_percentage``.
        batch_size : int, optional
            Maximum support or finite-grid evaluation batch size.
        generator : torch.Generator, optional
            Generator for randomized sketches and projectors.
        strict_system : bool
            Rejects coefficient matrices without the rank needed to identify
            all retained core coefficients.
        verbose : bool or int
            Level from 0 to 3: silent, progress, local details, core tensors.
        collect_metrics : bool
            Retains evaluation, truncation, solve and timing records.

        Returns
        -------
        TTDecomposition
            TT result, or its matrix/Quantics counterpart, on ``out_device``.

        Examples
        --------
        >>> tensor = torch.ones(2, 3, 2, 3)
        >>> result = tk.decompositions.TTRS(tensor).fit(1, center=2)
        >>> result.rank
        [1, 1, 1]
        """
        if (generator is not None) and not isinstance(generator,
                                                      torch.Generator):
            raise TypeError('`generator` should be torch.Generator or None')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        verbosity = _normalize_verbosity(verbose)
        count = self._tensor_source.n_sites
        center = TTRSS._normalize_center(center, count)
        spec = _SketchingFitSpec(
            cutoff=cutoff, atol=atol, rtol=rtol,
            cum_percentage=cum_percentage, verbose=verbose,
            collect_metrics=collect_metrics)
        observer = _resolve_observer(verbosity, None) if verbosity else None
        context = _SketchingFitContext(spec, self._runtime,
                                       generator=generator, observer=observer)
        before = self._source.evaluation_stats
        context.emit('start', values={
            'n_sites': count, 'operator': type(self.sketch_operator).__name__})
        with context.phase('fit'):
            system = self.sketch_operator.builder(self._tensor_source).build(
                batch_size=batch_size, generator=generator)
            result = system.solve(
                rank=rank, center=center, cutoff=cutoff, atol=atol, rtol=rtol,
                cum_percentage=cum_percentage, strict_system=strict_system,
                transform=self.transform, range_projector=self.range_projector,
                generator=generator, collect_metrics=context.need_diagnostics,
                _on_core=lambda site, core: context.complete(
                    site, core, count))
        # Console-only diagnostics are not retained by the fast result path.
        if collect_metrics:
            result.metrics.timings.extend(context.metrics.timings)
            result.metrics.evaluations.append(
                self._source.evaluation_stats.delta(before))
        if collect_metrics and isinstance(self._source, SupportTensorSource):
            samples = self._source.support.as_tensor()
            if self._source.is_matrix:
                samples = samples[:, ::2] * samples.new_tensor(
                    self._source.out_dim) + samples[:, 1::2]
            approximation = result.evaluate(samples)
            target = self._source.support_values.to(
                device=approximation.device)
            absolute = torch.linalg.vector_norm(approximation - target)
            denominator = torch.linalg.vector_norm(target)
            result.metrics.errors.append(ErrorRecord(
                kind='source_support', absolute=absolute,
                relative=safe_divide(
                    absolute, denominator, zero_value=absolute),
                denominator=denominator, size=target.shape[0]))
        metadata = {**result.metadata, 'algorithm': 'tt_rs',
                    'source_type': type(self._source).__name__,
                    'center': center}
        active = TTDecomposition(
            [self._runtime.finalize(core) for core in result.cores],
            metrics=(result.metrics if collect_metrics else
                     DecompositionMetrics()), metadata=metadata)
        context.emit('summary', values={'rank': active.rank})
        if observer is not None:
            observer.close(active.metrics)
        return _result_from_source(active, self._source)


@torch.no_grad()
def tt_rs(source: 'SourceLike', rank: Optional[_Rank] = None, *,
          center: Optional[int] = None,
          sketch_operator: Optional[SketchOperator] = None,
          transform: Optional[Callable] = None,
          cutoff: Optional[float] = None,
          atol: Optional[float] = None,
          rtol: Optional[float] = None,
          cum_percentage: Optional[float] = None,
          batch_size: Optional[int] = None,
          generator: Optional[torch.Generator] = None,
          out_device: Device = 'cpu', verbose: Union[bool, int] = 0,
          collect_metrics: bool = False) -> TTDecomposition:
    """
    Projects a finite source with TT-RS and returns a decomposition result.

    Parameters follow :class:`TTRS` and :meth:`TTRS.fit`. ``rank`` is a shared
    maximum or one value per TT bond; ``None`` requests discovery. ``center``
    selects the initial site. ``transform`` acts on each three-axis Phi.
    Use the class for custom range projectors or strict-system checks.

    Examples
    --------
    >>> result = tk.decompositions.tt_rs(torch.ones(2, 3, 2), rank=1)
    >>> result.rank
    [1, 1]
    """
    return TTRS(source, sketch_operator=sketch_operator,
                transform=transform, out_device=out_device).fit(
                    rank, center=center, cutoff=cutoff, atol=atol, rtol=rtol,
                    cum_percentage=cum_percentage, batch_size=batch_size,
                    generator=generator, verbose=verbose,
                    collect_metrics=collect_metrics)


@torch.no_grad()
def tt_rss(source: 'SourceLike', sketch_samples: Samples,
           embedding: Optional[Embedding] = None,
           rank: Optional[_Rank] = None, *,
           labels: Optional[torch.Tensor] = None,
           in_dim: Optional[Union[int, Sequence[int]]] = None,
           domain: Domain = None, domain_multiplier: int = 1,
           out_position: Optional[Union[int, Sequence[int]]] = None,
           center: Optional[int] = None,
           transform: Optional[Callable] = None,
           cutoff: Optional[float] = None,
           atol: Optional[float] = None,
           rtol: Optional[float] = None,
           cum_percentage: Optional[float] = None,
           batch_size: int = 64, device: Device = None,
           dtype: Optional[torch.dtype] = None,
           generator: Optional[torch.Generator] = None,
           out_device: Device = 'cpu', verbose: Union[bool, int] = 0,
           collect_metrics: bool = False) -> TTDecomposition:
    """
    Fits a sampled source with TT-RSS and returns a decomposition result.

    Parameters follow :class:`TTRSS` and :meth:`TTRSS.fit`. ``sketch_samples``
    contains the source inputs; finite sources supply basis embeddings by
    default, while continuous sources require ``embedding``. ``rank`` bounds
    TT bonds, or bonds between matrix pairs, and ``center`` selects the initial
    site in the input/output-expanded chain.
    ``transform(configuration_map)`` acts once on the shared value table.
    Use the class for custom fitters, range projectors or local solvers.

    Examples
    --------
    >>> samples = torch.cartesian_prod(*[torch.arange(2)] * 3)
    >>> result = tk.decompositions.tt_rss(torch.ones(2, 2, 2), samples, rank=1)
    >>> result.rank
    [1, 1]
    """
    return TTRSS(
        source, embedding, in_dim, domain, domain_multiplier, out_position,
        device=device, dtype=dtype, out_device=out_device,
        transform=transform).fit(
            sketch_samples, labels, rank, center=center, cutoff=cutoff,
            atol=atol, rtol=rtol, cum_percentage=cum_percentage,
            batch_size=batch_size, generator=generator, verbose=verbose,
            collect_metrics=collect_metrics)


@torch.no_grad()
def qtt_rss(source: CallableQuanticsSource, sketch_samples: Samples, *,
            sample_space: str = 'coordinates',
            rank: Optional[_Rank] = None, center: Optional[int] = None,
            transform: Optional[Callable] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: int = 64,
            generator: Optional[torch.Generator] = None,
            out_device: Device = 'cpu', verbose: Union[bool, int] = 0,
            collect_metrics: bool = False) -> TensorDecomposition:
    """
    Fits a Quantics source, using coordinate or encoded digit sketch samples.

    ``source`` owns the layout and coordinate map. ``sample_space`` is
    ``"coordinates"`` (quantized to digit rows) or ``"digits"`` (already
    encoded rows). Remaining arguments follow :meth:`TTRSS.fit`;
    ``transform`` follows :class:`TTRSS`. Returns a Quantics decomposition,
    preserving matrix structure when supplied by the source.

    Examples
    --------
    >>> source = tk.decompositions.CallableQuanticsSource(
    ...     lambda x: 1 + x[:, 0], 1, base=2, level=2,
    ...     domain=torch.tensor([0., 1.]))
    >>> samples = torch.arange(4).float().reshape(-1, 1) / 4
    >>> result = tk.decompositions.qtt_rss(source, samples, rank=2)
    >>> len(result.cores)
    2
    """
    if not isinstance(source, CallableQuanticsSource):
        raise TypeError('`source` should be CallableQuanticsSource type')
    if sample_space not in ('coordinates', 'digits'):
        raise ValueError("`sample_space` should be 'coordinates' or 'digits'")
    if sample_space == 'digits':
        sketch_samples = ConfigurationBatch(sketch_samples)
    return TTRSS(source, transform=transform, out_device=out_device).fit(
        sketch_samples, rank=rank, center=center, cutoff=cutoff, atol=atol,
        rtol=rtol, cum_percentage=cum_percentage, batch_size=batch_size,
        generator=generator, verbose=verbose, collect_metrics=collect_metrics)


__all__ = ['TTRSS', 'tt_rss', 'TTRS', 'tt_rs', 'qtt_rss']
