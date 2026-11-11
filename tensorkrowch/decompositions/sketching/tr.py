"""
This script contains:

    Classes:
        * _SketchCoreProvider
        * TRRSS

    Functions:
        * tr_rss
        * qtr_rss
"""

from typing import Any, Callable, Mapping, Optional, Sequence, Union

import torch

from tensorkrowch.formats import TT
from tensorkrowch.utils import safe_divide

from tensorkrowch.decompositions.metrics import ErrorRecord
from tensorkrowch.decompositions.results import (TensorDecomposition,
                                                 TRDecomposition,
                                                 _result_from_source)
from tensorkrowch.decompositions.ring.driver import BidirectionalRingDriver
from tensorkrowch.decompositions.ring.gauges import GaugeMap
from tensorkrowch.decompositions.ring.opening import (LoopOpener,
                                                      LoopOpening,
                                                      resolve_loop_opener)
from tensorkrowch.decompositions.ring.tt2tr import _TTCoreProvider
from tensorkrowch.decompositions.sketching.base import _SketchingFitContext
from tensorkrowch.decompositions.sketching.tt import (Device,
                                                      Domain,
                                                      Embedding,
                                                      Samples,
                                                      TTRSS)
from tensorkrowch.decompositions.sources import ConfigurationBatch
from tensorkrowch.decompositions.sources.quantics import CallableQuanticsSource


_Rank = Union[int, Sequence[int]]


class _SketchCoreProvider(_TTCoreProvider):  # MARK: _SketchCoreProvider
    """Uses TT blocks and full inherited sketch coordinates in one driver."""

    def __init__(self, tt: TT, prepared: list, engine: TTRSS,
                 context: _SketchingFitContext) -> None:
        super().__init__(tt)
        self.prepared = prepared
        self.engine = engine
        self.fit_context = context
        self.central_sites = ()
        self.boundary_sites = ()
        self.reported = set()

    def local_target(self, sites: Sequence[int],
                     context: Mapping[str, Any]) -> torch.Tensor:
        """Contracts existing TT cores for blocks; reuses single-site Phi."""
        sites = tuple(sites)
        if not self.central_sites:
            return super().local_target(sites, context)
        if any(site in self.boundary_sites for site in sites):
            target = super().local_target(sites, context)
            if (len(sites) > 1) and (context.get('rank_caps') is not None):
                # Match the incoming sketch gauge without projecting away its
                # sample-space constraints at a prescribed boundary opening.
                if sites[0] == 0:
                    lift = self._sample_environment(sites[-1] + 1, 'right')
                    return torch.einsum('...b,sb->...s', target, lift)
                lift = self._sample_environment(sites[0], 'left')
                return torch.einsum('sa,a...->s...', lift, target)
            return target
        if sites == self.central_sites:
            return super().local_target(sites, context)
        return self.prepared[sites[0]].coefficients

    def _sample_environment(self, edge: int, direction: str) -> torch.Tensor:
        """Evaluates the TT exterior on the requested unique sketch rows."""
        name = 'prefixes' if direction == 'left' else 'suffixes'
        region = self.fit_context.regions[name][edge]
        dtype, device = self.cores[0].dtype, self.cores[0].device
        state = torch.ones(region.n_unique, 1, dtype=dtype, device=device)
        pairs = tuple(zip(region.region.sites, region.values))
        if direction == 'right':
            pairs = tuple(reversed(pairs))
        for site, values in pairs:
            vectors = self.engine.outputs.embed_site(
                site, values, self.engine.embeddings, dtype=dtype).to(
                    device=device, dtype=dtype)
            if direction == 'left':
                state = torch.einsum('ml,lpr,mp->mr', state,
                                     self.cores[site], vectors)
            else:
                state = torch.einsum('lpr,mr,mp->ml', self.cores[site],
                                     state, vectors)
        return state

    def _report(self, opening: LoopOpening, sites: Sequence[int]) -> None:
        """Prints each completed local core once, in solve order."""
        for site, core in zip(sites, opening.cores):
            if site not in self.reported:
                self.fit_context.complete(site, core, len(self.cores))
                self.reported.add(site)

    def finalize_center(self, opening: LoopOpening,
                        sites: Sequence[int]) -> LoopOpening:
        """Lifts central TT coordinates back to the two sample spaces."""
        opening = super().finalize_center(opening, sites)
        opening.left_map = self._sample_environment(sites[0], 'left')
        opening.right_map = self._sample_environment(sites[-1] + 1, 'right')
        self._report(opening, sites)
        return opening

    def advance_right(self, opening: LoopOpening, local_target: Any,
                      recursion_context: Mapping[str, Any]
                      ) -> GaugeMap:
        """Extends the incoming gauge through TR cores on prefix samples."""
        sites = recursion_context['from_sites']
        self._report(opening, sites)
        gauge, _ = opening.transform_gauges()
        prefixes = self.fit_context.regions['prefixes']
        for site, core in zip(sites, opening.cores):
            recursion = prefixes[site].recursive_projector(prefixes[site + 1])
            selected = gauge.index_select(1, recursion.gather.to(gauge.device))
            vectors = self.engine.outputs.embed_site(
                site, recursion.new_values[0], self.engine.embeddings,
                dtype=core.dtype).to(device=core.device, dtype=core.dtype)
            gauge = torch.einsum('gml,lpr,mp->gmr', selected, core, vectors)
        return GaugeMap(gauge, 'left', metadata={
            'algorithm': 'sketch_recursion', 'direction': 'right'})

    def advance_left(self, opening: LoopOpening, local_target: Any,
                     recursion_context: Mapping[str, Any]
                     ) -> GaugeMap:
        """Extends the incoming gauge through TR cores on suffix samples."""
        sites = recursion_context['from_sites']
        self._report(opening, sites)
        _, gauge = opening.transform_gauges()
        suffixes = self.fit_context.regions['suffixes']
        for site, core in reversed(tuple(zip(sites, opening.cores))):
            recursion = suffixes[site + 1].recursive_projector(suffixes[site])
            selected = gauge.index_select(1, recursion.gather.to(gauge.device))
            vectors = self.engine.outputs.embed_site(
                site, recursion.new_values[0], self.engine.embeddings,
                dtype=core.dtype).to(device=core.device, dtype=core.dtype)
            gauge = torch.einsum('lpr,mp,rmg->lmg', core, vectors, selected)
        return GaugeMap(gauge, 'right', metadata={
            'algorithm': 'sketch_recursion', 'direction': 'left'})

    def prepare_boundary(self, opening: LoopOpening, local_target: Any,
                         recursion_context: Mapping[str, Any], direction: str
                         ) -> GaugeMap:
        """Keeps the actual outgoing gauge for TT end-block absorption."""
        self._report(opening, recursion_context['from_sites'])
        return super().prepare_boundary(
            opening, local_target, recursion_context, direction)

    def close_boundary(self, sites: Sequence[int], direction: str,
                       opening: LoopOpening, context: Mapping[str, Any]
                       ) -> Any:
        """Absorbs the outgoing gauge and splits the existing TT end block."""
        cores = super().close_boundary(sites, direction, opening, context)
        for site, core in zip(sites, cores):
            self.fit_context.complete(site, core, len(self.cores))
        return cores


class TRRSS(TTRSS):  # MARK: TRRSS
    """
    Fits a tensor ring from correlated source samples.

    Construction arguments are those of :class:`TTRSS`. The two algorithms
    share evaluation, global transformation, physical fitting and directional
    SVDs. TR-RSS then aligns an intermediate TT, contracts its central/end
    blocks and opens the central loop. Singular spectra remain absorbed in
    the local tensors. A :class:`~tensorkrowch.decompositions.QRLoopOpener`
    can balance these targets and refine after restoring its factors.
    Subsequent sites retain the full
    inherited sketch space, so sketch recursion can preserve constraints
    outside the rank selected by an outgoing SVD.

    No new source-query Phi is constructed for a block. The intermediate TT
    alignment can introduce approximation error; it is measured separately
    from local loop-opening residuals when metrics are requested. The
    ``tt_alignment`` error is measured on the supplied sketch samples, or on
    the shared configuration table when ``transform`` changes target values.
    It is a residual of the intermediate TT, rather than an error bound for
    the final ring.

    Examples
    --------
    >>> samples = torch.cartesian_prod(*[torch.arange(2)] * 5)
    >>> result = tk.decompositions.TRRSS(torch.ones(2, 2, 2, 2, 2)).fit(
    ...     samples, rank=1)
    >>> result.rank
    [1, 1, 1, 1, 1]
    """

    def fit(self, sketch_samples: Samples,
            labels: Optional[torch.Tensor] = None,
            rank: Optional[_Rank] = None, *,
            center: Optional[int] = None,
            loop_opener: Union[str, LoopOpener] = 'als',
            block_boundaries: bool = True,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: int = 64,
            generator: Optional[torch.Generator] = None,
            verbose: Union[bool, int] = 0,
            collect_metrics: bool = False) -> TRDecomposition:
        """
        Opens an automatic central block and recurses toward both ends.

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
            Shared maximum rank or one right-bond rank per final TR site; the
            last entry is cyclic. Phi TT-cut caps are cyclic rank times each
            internal rank. ``None`` estimates a complete TR profile from three
            disjoint TT blocks before using the same physical-core ALS solves.
        center : int, optional
            Seed site of the initial block. Defaults to the middle; blocks
            grow to provide sufficient physical capacity. Discovery requires
            an interior center and rejects overlap with the capacity end
            blocks. Prescribed-rank blocks may merge at the ends.
        loop_opener : str or LoopOpener
            Local cyclic factorization, default ``"als"``. Configure ALS
            solvers and convergence through an ``ALSLoopOpener`` object.
            Use ``QRLoopOpener`` for local balancing and refinement.
        block_boundaries : bool
            Group boundary sites into capacity blocks. Discovery absorbs
            gauges and splits these blocks by SVD; prescribed ranks use ALS
            with separate physical cores. ``False`` keeps single-site ends.
            Discovery requires three disjoint capacity blocks in either case.
        cutoff : float, optional
            Minimum singular value to keep. Values ``<= cutoff`` are removed.
        atol : float, optional
            Maximum discarded sum of squared singular values.
        rtol : float, optional
            Maximum discarded sum of squared singular values divided by their
            total sum of squares. Must lie in ``[0, 1]``. Also controls SVD
            splitting within discovery end blocks. Prescribed ALS ranks stay
            fixed even when the Phi SVDs select smaller TT dimensions.
            Local ALS convergence is configured independently through the
            loop opener's convergence policy.
        cum_percentage : float, optional
            Minimum retained squared singular-value fraction. Equivalent to
            ``rtol=1-cum_percentage``.
        batch_size : int
            Maximum number of unique configurations evaluated together.
        generator : torch.Generator, optional
            Generator for domain subsampling, labels and local initializations.
        verbose : bool or int
            Level from 0 to 3: silent, progress, local details, core tensors.
        collect_metrics : bool
            Retains evaluation, fitting, truncation, solve and timing records.

        Returns
        -------
        TRDecomposition
            Cyclic result, or its matrix/Quantics counterpart, on
            ``out_device``.

        Examples
        --------
        >>> tensor = torch.ones(2, 3, 2, 3)
        >>> samples = torch.cartesian_prod(
        ...     *[torch.arange(d) for d in tensor.shape])
        >>> result = tk.decompositions.TRRSS(tensor).fit(samples, rank=1)
        >>> result.in_dim
        (2, 3, 2, 3)
        """
        if not isinstance(block_boundaries, bool):
            raise TypeError('`block_boundaries` should be bool type')
        opener = resolve_loop_opener(loop_opener)
        samples = self._initialize_fit(sketch_samples, generator)
        count = self.outputs.n_sites
        center = self._normalize_center(center, count)
        if (rank is None) and (center in (0, count - 1)):
            raise ValueError(
                'Cannot discover ranks from an endpoint `center`; '
                'choose an interior site or provide `rank`')
        self._validate_labels(labels, samples.batch_size)
        ranks = self._chain_rank(rank, cyclic=True)
        tt_ranks = (None if ranks is None else
                    tuple(ranks[-1] * value for value in ranks[:-1]))
        context = self._new_context(
            cutoff=cutoff, atol=atol, rtol=rtol,
            cum_percentage=cum_percentage, batch_size=batch_size,
            generator=generator, collect_metrics=collect_metrics,
            verbose=verbose)
        context.state.update(
            input_samples=samples, labels=labels, rank=tt_ranks,
            tr_rank=ranks,
            requested_rank=(ranks[1::2] if (ranks is not None) and
                            getattr(self._source_like, 'is_matrix', False)
                            else ranks),
            center=center, loop_opener=opener,
            block_boundaries=block_boundaries)
        return self._execute(context)

    def _decompose(self, context: _SketchingFitContext) -> TRDecomposition:
        """Reuses prepared Phi tensors and a small TT alignment scaffold."""
        prepared = self._prepare_phis(context)
        tt_cores = self._solve_tt(prepared, context, report=False)
        tt = TT(tt_cores)
        if context.collect_metrics:
            samples, target, _ = self._error_target(context)
            approximation = TTRSS._evaluate_extended_cores(
                self, tt_cores, samples, context)
            target = target.to(device=approximation.device)
            absolute = torch.linalg.vector_norm(approximation - target)
            denominator = torch.linalg.vector_norm(target)
            relative = safe_divide(absolute, denominator,
                                   zero_value=torch.where(
                                       absolute == 0, 0, torch.inf))
            context.metrics.errors.append(ErrorRecord(
                kind='tt_alignment', absolute=absolute, relative=relative,
                denominator=denominator, size=target.shape[0]))
        provider = _SketchCoreProvider(tt, prepared, self, context)
        center = context.state['center']
        prescribed = context.state['tr_rank']
        driver = BidirectionalRingDriver()
        result = driver.fit(
            provider, prescribed, context.state['loop_opener'], provider,
            center=center, block_boundaries=context.state['block_boundaries'],
            rtol=context.spec.rtol,
            context={'generator': context.generator,
                     'collect_metrics': context.need_diagnostics})
        sites = driver.central_block
        left, right = driver.boundary_blocks
        discovery = driver.rank_discovery
        if context.collect_metrics:
            context.metrics.local_solves.extend(driver.metrics.local_solves)
            context.metrics.truncations.extend(driver.metrics.truncations)
            context.metrics.errors.extend(driver.metrics.errors)
        context.cores.extend(result.cores)
        context.state.update(center_block=sites, boundary_blocks=(left, right),
                             rank_discovery=discovery, order=driver.order)
        return self._assemble_result(context)

    def _evaluate_extended_cores(self, cores: Sequence[torch.Tensor],
                                 samples: Sequence[torch.Tensor],
                                 context: _SketchingFitContext
                                 ) -> torch.Tensor:
        """Evaluates the trace of the sitewise embedded TR matrices."""
        state = None
        for site, (core, values) in enumerate(zip(cores, samples)):
            vectors = self.outputs.embed_site(
                site, values, self.embeddings, dtype=core.dtype).to(
                    device=core.device, dtype=core.dtype)
            matrix = torch.einsum('lpr,bp->blr', core, vectors)
            state = matrix if state is None else torch.bmm(state, matrix)
        return state.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    def _assemble_result(self, context: _SketchingFitContext
                         ) -> TRDecomposition:
        """Stores ranks, construction blocks and optional evaluation error."""
        metadata = {name: context.state[name] for name in (
            'center', 'center_block', 'boundary_blocks', 'rank_discovery',
            'order')}
        metadata.update(algorithm='tr_rss',
                        requested_rank=context.state['requested_rank'],
                        tt_rank_caps=context.state['rank'])
        if context.collect_metrics:
            samples, target, kind = self._error_target(context)
            approximation = self._evaluate_extended_cores(
                context.cores, samples, context)
            target = target.to(device=approximation.device)
            absolute = torch.linalg.vector_norm(approximation - target)
            denominator = torch.linalg.vector_norm(target)
            relative = safe_divide(absolute, denominator,
                                   zero_value=torch.where(
                                       absolute == 0, 0, torch.inf))
            context.metrics.errors.append(ErrorRecord(
                kind=kind, absolute=absolute, relative=relative,
                denominator=denominator, size=target.shape[0]))
            metadata['sample_error'] = float(relative.detach().cpu())
        context.metrics.local_solves.sort(key=lambda item: item.site)
        result = TRDecomposition(
            [context.runtime.finalize(core) for core in context.cores],
            in_sites=self.outputs.input_positions,
            metrics=context.metrics, metadata=metadata)
        structured = getattr(self._source_like, 'is_matrix', False) or \
            isinstance(self._source_like, CallableQuanticsSource)
        return (_result_from_source(result, self._source_like)
                if structured else result)

    def _validate_result(self, result: TensorDecomposition,
                         context: _SketchingFitContext) -> None:
        """Checks cyclic topology and the fitted physical dimensions."""
        if not isinstance(result, TensorDecomposition) or not result._cyclic:
            raise TypeError('`result` should have cyclic format')
        expected = context.state['fitted_site_dim']
        dimensions = (result.in_dim if result.out_dim is None else
                      tuple(dim for pair in zip(result.in_dim, result.out_dim)
                            for dim in pair))
        if dimensions != expected:
            raise ValueError('The result input dimensions are inconsistent')


@torch.no_grad()
def tr_rss(source: Any, sketch_samples: Samples,
           embedding: Optional[Embedding] = None,
           rank: Optional[_Rank] = None, *,
           labels: Optional[torch.Tensor] = None,
           in_dim: Optional[Union[int, Sequence[int]]] = None,
           domain: Domain = None, domain_multiplier: int = 1,
           out_position: Optional[Union[int, Sequence[int]]] = None,
           center: Optional[int] = None,
           loop_opener: Union[str, LoopOpener] = 'als',
           block_boundaries: bool = True,
           transform: Optional[Callable] = None,
           cutoff: Optional[float] = None,
           atol: Optional[float] = None,
           rtol: Optional[float] = None,
           cum_percentage: Optional[float] = None,
           batch_size: int = 64, device: Device = None,
           dtype: Optional[torch.dtype] = None,
           generator: Optional[torch.Generator] = None,
           out_device: Device = 'cpu', verbose: Union[bool, int] = 0,
           collect_metrics: bool = False) -> TRDecomposition:
    """
    Fits a sampled source with TR-RSS and returns a decomposition result.

    Parameters follow :class:`TTRSS` and :meth:`TRRSS.fit`. ``rank`` is a
    shared maximum or a right-bond profile whose last entry is cyclic.
    ``None`` requests discovery. ``loop_opener`` owns local ALS configuration.
    ``transform(configuration_map)`` acts once on the shared value table.

    Examples
    --------
    >>> samples = torch.cartesian_prod(*[torch.arange(2)] * 4)
    >>> result = tk.decompositions.tr_rss(torch.ones(2, 2, 2, 2), samples,
    ...                                     rank=1)
    >>> result.rank
    [1, 1, 1, 1]
    """
    return TRRSS(
        source, embedding, in_dim, domain, domain_multiplier, out_position,
        device=device, dtype=dtype, out_device=out_device,
        transform=transform).fit(
            sketch_samples, labels, rank, center=center,
            loop_opener=loop_opener, block_boundaries=block_boundaries,
            cutoff=cutoff, atol=atol, rtol=rtol, cum_percentage=cum_percentage,
            batch_size=batch_size, generator=generator, verbose=verbose,
            collect_metrics=collect_metrics)


@torch.no_grad()
def qtr_rss(source: CallableQuanticsSource, sketch_samples: Samples, *,
            sample_space: str = 'coordinates',
            rank: Optional[_Rank] = None, center: Optional[int] = None,
            loop_opener: Union[str, LoopOpener] = 'als',
            block_boundaries: bool = True,
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
    Fits a Quantics ring from coordinate or encoded digit sketch samples.

    ``sample_space`` is ``"coordinates"`` (quantized to digit rows) or
    ``"digits"`` (already encoded rows). The source owns its layout and
    coordinate map. Remaining arguments follow :meth:`TRRSS.fit`;
    ``transform`` follows :class:`TTRSS`. Returns the Quantics decomposition,
    preserving matrix structure when supplied by the source.

    Parameters
    ----------
    source : CallableQuanticsSource
        Source owning the digit layout and coordinate map.
    sketch_samples : torch.Tensor, sequence or ConfigurationBatch
        Coordinate rows or encoded digit rows, selected by ``sample_space``.
    sample_space : {"coordinates", "digits"}
        Interpretation of ``sketch_samples``. Default is ``"coordinates"``.
    rank : int or sequence[int], optional
        Shared maximum rank or one right-bond rank per final TR site; the
        last entry is cyclic. ``None`` requests rank discovery.
    center : int, optional
        Initial digit site, defaulting to the middle site.
    loop_opener : str or LoopOpener
        Local factorization, configured as in :meth:`TRRSS.fit`.
    block_boundaries : bool
        Whether to group boundary sites into automatic blocks.
    transform : callable, optional
        Global callback as in :class:`TTRSS`.
    cutoff, atol, rtol, cum_percentage : float, optional
        SVD truncation criteria, as in :meth:`TRRSS.fit`.
    batch_size : int
        Maximum number of unique configurations evaluated together.
    generator : torch.Generator, optional
        Generator for domain subsampling, labels and local initializations.
    out_device : str or torch.device, optional
        Device receiving completed cores; ``None`` retains the source device.
    verbose : bool or int
        Level from 0 to 3: silent, progress, local details, core tensors.
    collect_metrics : bool
        Retains evaluation, fitting, truncation, solve and timing records.

    Returns
    -------
    TensorDecomposition
        Quantics ring result with the source layout and coordinate map.
    """
    if not isinstance(source, CallableQuanticsSource):
        raise TypeError('`source` should be CallableQuanticsSource type')
    if sample_space not in ('coordinates', 'digits'):
        raise ValueError("`sample_space` should be 'coordinates' or 'digits'")
    if sample_space == 'digits':
        values = (sketch_samples.values if isinstance(
            sketch_samples, ConfigurationBatch) else sketch_samples)
        sketch_samples = ConfigurationBatch(values, kind='indices')
    return TRRSS(source, transform=transform, out_device=out_device).fit(
        sketch_samples, rank=rank, center=center, loop_opener=loop_opener,
        block_boundaries=block_boundaries, cutoff=cutoff, atol=atol, rtol=rtol,
        cum_percentage=cum_percentage, batch_size=batch_size,
        generator=generator, verbose=verbose, collect_metrics=collect_metrics)


__all__ = ['TRRSS', 'tr_rss', 'qtr_rss']
