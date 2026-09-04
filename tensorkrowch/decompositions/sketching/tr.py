"""Tensor ring decompositions based on recursive sketching."""

import warnings
from dataclasses import dataclass
from math import ceil, prod
from typing import (Any, Dict, Mapping, Optional, Sequence, Tuple, Union)

import torch

from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord,
                                                 TruncationRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   DecompositionObserver,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import (QTRTuckerDecomposition,
                                                 TRDecomposition)
from tensorkrowch.decompositions.ring.blocks import (BlockSelection,
                                                     CentralBlockSelector,
                                                     PrescribedCentralBlockSelector,
                                                     RingRankEstimator,
                                                     _normalize_rank_spec)
from tensorkrowch.decompositions.ring.driver import (BidirectionalRingDriver,
                                                     BoundaryClosure)
from tensorkrowch.decompositions.ring.gauges import (ExperimentalWarning,
                                                     GaugeRecursionStep)
from tensorkrowch.decompositions.ring.opening import (LoopOpener,
                                                      LoopOpening,
                                                      FixedGaugeCoreOpener,
                                                      resolve_loop_opener)
from tensorkrowch.decompositions.ring.tt2tr import TT2TR
from tensorkrowch.decompositions.ring.schedules import AlternatingRingDriver
from tensorkrowch.decompositions.sketching.base import _SketchingFitContext
from tensorkrowch.decompositions.sketching.evaluations import (
    _EvaluationPlanBuilder,
    _EvaluationSession,
)
from tensorkrowch.decompositions.sketching.phi import (
    PhiOperator,
    _MaterializedPhi,
)
from tensorkrowch.decompositions.sketching.fitting import (BasisFitter,
                                                           InputFitter,
                                                           QTTInputFitter)
from tensorkrowch.decompositions.sketching.projections import RangeProjector
from tensorkrowch.decompositions.sketching.quantization import (
    CoordinateMap,
    QuantizedLayout,
)
from tensorkrowch.decompositions.sketching.transforms import (
    GlobalValueTransform,
    LocalValueTransform,
    _apply_local_transform,
    _collect_local_queries,
    _prepare_global_transform,
)
from tensorkrowch.decompositions.sketching.specs import _OutputSpec
from tensorkrowch.decompositions.sketching.sketches import SketchOperator
from tensorkrowch.decompositions.sketching.sources import SupportTensorSource
from tensorkrowch.decompositions.sketching.tt import (Device, QTTTuckerRSS,
                                                      Samples, TTRS, TTRSS,
                                                      _QTTTuckerFitMixin,
                                                      _quantized_source,
                                                      _QuantizedRSSMixin)
from tensorkrowch.decompositions.sources import ConfigurationBatch
from tensorkrowch.utils import truncated_svd


_Rank = Union[int, Sequence[int]]


class _SelectedBlockSelector(CentralBlockSelector):
    """Returns one already-selected center block to the shared ring driver."""

    def __init__(self, selection: BlockSelection) -> None:
        if not isinstance(selection, BlockSelection):
            raise TypeError('`selection` should be BlockSelection type')
        self.selection = selection

    def select(self, provider, rank, center=None, *, bounds=None):
        return self.selection


@dataclass(frozen=True)
class _SketchLocalTarget:
    """Carries one fitted Phi and the trimming policy for its sketch axes."""

    tensor: torch.Tensor
    sites: Tuple[int, ...]
    truncation: Mapping[str, Any]
    collect_metrics: bool


class _SketchLoopOpener:
    """Trims free sketch axes around one delegated loop-opening strategy."""

    def __init__(self,
                 opener: LoopOpener,
                 records: list) -> None:
        if not isinstance(opener, LoopOpener):
            raise TypeError('`opener` should implement LoopOpener')
        self.opener = opener
        self.records = records

    @property
    def capabilities(self):
        return self.opener.capabilities

    def _trim(self,
              matrix: torch.Tensor,
              rank: int,
              site: int,
              target: _SketchLocalTarget):
        options = dict(target.truncation)
        options['rank'] = min(rank, min(matrix.shape))
        if target.collect_metrics:
            u, s, vh, info = truncated_svd(
                matrix, return_info=True, **options)
            self.records.append(TruncationRecord.from_svd_info(
                info, site=site))
        else:
            u, s, vh = truncated_svd(matrix, **options)
        return u, s.unsqueeze(1) * vh

    def open(self,
             target,
             rank,
             *,
             fixed_left=None,
             fixed_right=None,
             orientation='right',
             context=None) -> LoopOpening:
        if not isinstance(target, _SketchLocalTarget):
            raise TypeError('Sketch loop opening requires a fitted Phi target')
        ranks = list(rank)
        tensor = target.tensor
        left_lift = None
        right_lift = None

        if fixed_left is not None:
            ranks[0] = fixed_left.shape[-1]
            ranks[-1] = fixed_left.shape[0]
        if fixed_right is not None:
            ranks[-2] = fixed_right.shape[0]
            ranks[-1] = fixed_right.shape[-1]

        if fixed_right is None:
            matrix = tensor.reshape(-1, tensor.shape[-1])
            u, right_lift = self._trim(
                matrix,
                ranks[-2] * ranks[-1],
                target.sites[-1],
                target)
            tensor = u.reshape(*tensor.shape[:-1], u.shape[-1])

        if fixed_left is None:
            matrix = tensor.reshape(tensor.shape[0], -1)
            left_lift, remainder = self._trim(
                matrix,
                ranks[-1] * ranks[0],
                target.sites[0],
                target)
            tensor = remainder.reshape(
                remainder.shape[0], *tensor.shape[1:])

        local_context = {} if context is None else dict(context)
        local_context.update({
            'input_dim': tuple(tensor.shape),
            'dtype': tensor.dtype,
            'device': tensor.device,
        })
        opening = self.opener.open(
            tensor,
            tuple(ranks),
            fixed_left=fixed_left,
            fixed_right=fixed_right,
            orientation=orientation,
            context=local_context)

        left_gauge = opening.left_gauge
        if left_lift is not None:
            left_gauge = torch.einsum(
                'sa,gar->gsr',
                left_lift.to(left_gauge.dtype),
                left_gauge)
        right_gauge = opening.right_gauge
        if right_lift is not None:
            right_gauge = torch.einsum(
                'rbg,bt->rtg',
                right_gauge,
                right_lift.to(right_gauge.dtype))
        all_cores = (left_gauge, *opening.cores, right_gauge)
        return LoopOpening(
            left_gauge=left_gauge,
            cores=opening.cores,
            right_gauge=right_gauge,
            rank=tuple(core.shape[-1] for core in all_cores),
            orientation=opening.orientation,
            local_records=opening.local_records,
            diagnostics={
                **opening.diagnostics,
                'sketch_trimmed_left': left_lift is not None,
                'sketch_trimmed_right': right_lift is not None,
            })


class SketchGaugeRecursion:
    """Extends RSS gauges through opened cores and sketch recursions."""

    @staticmethod
    def _provider(recursion_context: Mapping[str, Any]):
        provider = recursion_context.get('provider')
        if not isinstance(provider, _SketchTargetProvider):
            raise TypeError(
                'Sketch gauge recursion requires a sketch target provider')
        return provider

    def advance_right(self,
                      opening: LoopOpening,
                      local_target: Any,
                      recursion_context: Mapping[str, Any]
                      ) -> GaugeRecursionStep:
        provider = self._provider(recursion_context)
        gauge = provider.extend_right(
            opening, recursion_context['from_sites'])
        return GaugeRecursionStep(
            gauge=gauge,
            diagnostics={
                'algorithm': 'sketch_recursion',
                'direction': 'right',
                'shape': tuple(gauge.shape),
            })

    def advance_left(self,
                     opening: LoopOpening,
                     local_target: Any,
                     recursion_context: Mapping[str, Any]
                     ) -> GaugeRecursionStep:
        provider = self._provider(recursion_context)
        gauge = provider.extend_left(
            opening, recursion_context['from_sites'])
        return GaugeRecursionStep(
            gauge=gauge,
            diagnostics={
                'algorithm': 'sketch_recursion',
                'direction': 'left',
                'shape': tuple(gauge.shape),
            })

    def prepare_boundary(self,
                         opening: LoopOpening,
                         local_target: Any,
                         recursion_context: Mapping[str, Any],
                         direction: str) -> GaugeRecursionStep:
        """Extends the final environment to one open sketch boundary."""
        if direction == 'right':
            return self.advance_right(
                opening, local_target, recursion_context)
        if direction == 'left':
            return self.advance_left(
                opening, local_target, recursion_context)
        raise ValueError("`direction` should be 'left' or 'right'")


@dataclass
class _SketchTargetProvider:
    """Exposes fitted Phi targets and recursive sketch bases to the driver."""

    targets: Mapping[Tuple[int, ...], _SketchLocalTarget]
    input_dim: Sequence[int]
    rank: Sequence[int]
    prefixes: Sequence[Any]
    suffixes: Sequence[Any]
    outputs: _OutputSpec
    embeddings: Any
    boundary_records: list
    embedding_function: Optional[Any] = None

    boundary_mode = 'open'

    def local_target(self, sites, context):
        sites = tuple(sites)
        try:
            return self.targets[sites]
        except KeyError as exc:
            raise ValueError(f'No fitted Phi target was planned for {sites}') \
                from exc

    def local_rank(self, sites, rank, context):
        sites = tuple(sites)
        return (
            self.rank[(sites[0] - 1) % len(self.rank)],
            *(self.rank[site] for site in sites),
            self.rank[-1],
        )

    def local_context(self, sites, context):
        target = self.local_target(sites, context)
        return {
            'input_dim': tuple(target.tensor.shape),
            'dtype': target.tensor.dtype,
            'device': target.tensor.device,
            'generator': context.get('generator'),
        }

    def _embed(self, site: int, values: torch.Tensor, dtype) -> torch.Tensor:
        if self.embedding_function is not None:
            return self.embedding_function(site, values, dtype)
        return self.outputs.embed_site(
            site, values, self.embeddings, dtype=dtype)

    def extend_right(self,
                     opening: LoopOpening,
                     sites: Sequence[int]) -> torch.Tensor:
        gauge = opening.left_gauge
        for site, core in zip(sites, opening.cores):
            recursion = self.prefixes[site].recursive_projector(
                self.prefixes[site + 1])
            selected = gauge.index_select(
                1, recursion.gather.to(gauge.device))
            vectors = self._embed(
                site, recursion.new_values[0], core.dtype).to(core.device)
            gauge = torch.einsum(
                'gmr,rpd,mp->gmd', selected, core, vectors)
        return gauge

    def extend_left(self,
                    opening: LoopOpening,
                    sites: Sequence[int]) -> torch.Tensor:
        gauge = opening.right_gauge
        for site, core in reversed(tuple(zip(sites, opening.cores))):
            recursion = self.suffixes[site + 1].recursive_projector(
                self.suffixes[site])
            selected = gauge.index_select(
                1, recursion.gather.to(gauge.device))
            vectors = self._embed(
                site, recursion.new_values[0], core.dtype).to(core.device)
            gauge = torch.einsum(
                'rpd,mp,dmg->rmg', core, vectors, selected)
        return gauge

    def close_boundary(self,
                       site: int,
                       direction: str,
                       opening: LoopOpening,
                       context: Mapping[str, Any]) -> BoundaryClosure:
        """Solves the first or last core against the propagated environment."""
        gauge = context.get('boundary_gauge')
        if not isinstance(gauge, torch.Tensor):
            raise ValueError('A propagated boundary gauge is required')
        target = self.targets[(site,)].tensor
        solver = LeastSquaresSolver()
        if direction == 'right':
            if site != len(self.input_dim) - 1:
                raise ValueError('The right boundary should be the last site')
            values = target.squeeze(-1)
            matrix = gauge.permute(1, 2, 0).reshape(
                values.shape[0], -1)
            solution, record = solver.solve(
                matrix,
                values.to(matrix.dtype),
                site=site,
                return_record=True)
            core = solution.reshape(
                gauge.shape[-1], gauge.shape[0], values.shape[-1]
            ).permute(0, 2, 1)
        elif direction == 'left':
            if site != 0:
                raise ValueError('The left boundary should be the first site')
            values = target.squeeze(0)
            matrix = gauge.permute(1, 2, 0).reshape(
                values.shape[-1], -1)
            solution, record = solver.solve(
                matrix,
                values.transpose(0, 1).to(matrix.dtype),
                site=site,
                return_record=True)
            core = solution.reshape(
                gauge.shape[-1], gauge.shape[0], values.shape[0])
            core = core.permute(0, 2, 1)
        else:
            raise ValueError("`direction` should be 'left' or 'right'")
        self.boundary_records.append(record)
        return BoundaryClosure(
            site=site,
            direction=direction,
            core=core,
            diagnostics={
                'algorithm': 'sketch_boundary_solve',
                'residual_relative': record.residual_relative,
            })


class TRRSS(TTRSS):
    r"""Reusable Tensor Ring Recursive Sketching from Samples problem.

    The constructor fixes the same source, site embeddings, domains and output
    layout as :class:`~tensorkrowch.decompositions.TTRSS`. Each :meth:`fit`
    opens fitted local Phi tensors from an internal center, recursively extends
    their sketch gauges in both directions and solves the two open boundaries.
    Scalar outputs add no site; every tensor-output axis becomes a separated
    basis-embedded site.

    The returned :class:`~tensorkrowch.decompositions.TRDecomposition` is a
    lightweight cyclic result. Its cores can initialize a periodic
    :class:`~tensorkrowch.models.MPS` with
    ``tk.models.MPS(tensors=result.cores)``.

    Examples
    --------
    >>> domain = torch.tensor([0., 1.])
    >>> samples = torch.cartesian_prod(domain, domain, domain, domain)
    >>> def function(data):
    ...     return (1 + data).prod(dim=1)
    >>> def embedding(values):
    ...     return torch.stack((1 - values, values), dim=-1)
    >>> decomposer = TRRSS(function, embedding, domain=domain)
    >>> result = decomposer.fit(samples, rank=1)
    >>> result.rank
    [1, 1, 1, 1]
    """

    @classmethod
    def quantized(
            cls,
            function=None,
            *,
            source=None,
            layout: Optional[QuantizedLayout] = None,
            n_variables: Optional[int] = None,
            base: Union[int, Sequence[int]] = 2,
            level: Union[int, Sequence[int]] = 1,
            ordering: str = 'grouped',
            digit_order: str = 'coarse_to_fine',
            permutation=None,
            coordinate_map: Optional[
                Union[CoordinateMap, Sequence[CoordinateMap]]] = None,
            domain=None,
            source_space: str = 'physical',
            source_layout: Optional[QuantizedLayout] = None,
            sample_space: str = 'physical',
            computational_grid: str = 'endpoints',
            out_of_domain: str = 'error',
            out_position=None,
            device: Device = None,
            dtype: Optional[torch.dtype] = None,
            output_device: Device = 'cpu',
            input_fitters: Optional[Sequence[InputFitter]] = None,
            range_projector: Optional[RangeProjector] = None,
            global_transform: Optional[GlobalValueTransform] = None,
            local_transform: Optional[LocalValueTransform] = None,
            local_solver: Optional[LeastSquaresSolver] = None,
            synchronize_timers: bool = True) -> 'TRRSS':
        """Creates a QTR-RSS problem with basis-embedded digit sites."""
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
        return _QuantizedTRRSS(
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

    def fit(
            self,
            sketch_samples: Samples,
            labels: Optional[torch.Tensor] = None,
            rank: _Rank = 1,
            center: Optional[int] = None,
            loop_opener: Union[str, LoopOpener] = 'als',
            schedule: str = 'center_out',
            schedule_block_size: int = 1,
            block_selector: Optional[CentralBlockSelector] = None,
            adaptive: bool = False,
            pad_to_rank: bool = False,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: int = 64,
            generator: Optional[torch.Generator] = None,
            warm_start: Optional[TRDecomposition] = None,
            verbose: Union[bool, int] = 0,
            collect_metrics: bool = False,
            observer: Optional[DecompositionObserver] = None
            ) -> TRDecomposition:
        r"""Decomposes the fixed function into a sampled Tensor Ring.

        Samples contain only the original input sites. Output indices are
        selected from flattened labels, or sampled proportionally to
        ``abs(function(samples)) ** 2``, and inserted internally. ``rank`` is
        one upper bound shared by all right links, or one bound per final site;
        its last entry is always the cyclic link.

        In prescribed mode the center is one site and the requested ranks are
        passed to the local loop opener. With ``adaptive=True``, a centered
        injective block is fitted and doubly trimmed, the cyclic and adjacent
        ranks are estimated, and internal/outward ranks are discovered under
        the same caps. Effective ranks are returned unless ``pad_to_rank=True``
        explicitly zero-pads them back to the caps.

        Parameters
        ----------
        sketch_samples : torch.Tensor, sequence of torch.Tensor or ConfigurationBatch
            Correlated samples in packed or heterogeneous per-site form.
        labels : torch.Tensor, optional
            Flattened tensor-output labels with shape ``(batch_size,)``.
        rank : int or sequence of int
            Shared rank cap or one right-link cap per final TR site. The last
            value is the cyclic rank cap.
        center : int, optional
            Internal seed site. It defaults to the middle site.
        loop_opener : {``"als"``, ``"blostr+als"``} or LoopOpener, optional
            Encapsulated local loop-opening strategy. Advanced ALS options are
            supplied through an
            :class:`~tensorkrowch.decompositions.ALSLoopOpener`, not expanded
            into this signature.
        schedule : {``"center_out"``, ``"alternating"``}, optional
            Serial ring schedule. The alternating path opens independent
            anchors first and then solves intervening sites with two fixed
            recursively propagated gauges. It is experimental and reports any
            explicit fallback to ``"center_out"`` in the result metadata.
        schedule_block_size : int, optional
            Alternating block size. The sampled open-boundary provider
            currently supports an exact checkerboard for size one.
        block_selector : CentralBlockSelector, optional
            Advanced central-block policy. Adaptive mode otherwise uses the
            balanced common selector restricted to internal sites.
        adaptive : bool, optional
            Whether ``rank`` entries are caps for rank discovery.
        pad_to_rank : bool, optional
            Whether adaptive effective ranks are explicitly zero-padded to the
            requested caps. It is disabled by default.
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
        generator : torch.Generator, optional
            Generator used by output sampling and local initializations.
        warm_start : TRDecomposition, optional
            Reserved for a future defined update; non-``None`` values are
            rejected rather than retaining hidden state.
        verbose : bool or int, optional
            Verbosity from 0 (silent) to 3 (including final core tensors).
        collect_metrics : bool, optional
            Collects timings, fitting/trimming records, local solve diagnostics,
            evaluation counts and absolute/relative sketch-sample error.
        observer : DecompositionObserver, optional
            Additional structured-event consumer.

        Returns
        -------
        TRDecomposition
            Lightweight cyclic decomposition stored on ``output_device``.

        Examples
        --------
        >>> domain = torch.tensor([0., 1.])
        >>> samples = torch.cartesian_prod(domain, domain, domain)
        >>> function = lambda data: (1 + data).prod(dim=1)
        >>> embedding = lambda values: torch.stack(
        ...     (1 - values, values), dim=-1)
        >>> result = TRRSS(
        ...     function, embedding, domain=domain).fit(samples, rank=1)
        >>> len(result.cores)
        3
        """
        samples = self._normalize_samples(sketch_samples)
        if labels is not None and (
                not isinstance(labels, torch.Tensor) or
                labels.shape != (samples.batch_size,)):
            raise ValueError(
                '`labels` should be a tensor with shape (batch_size,)')
        if warm_start is not None:
            if not isinstance(warm_start, TRDecomposition):
                raise TypeError(
                    '`warm_start` should be TRDecomposition type or None')
            raise NotImplementedError(
                'TR-RSS does not yet define a warm-start update; pass None')
        if not isinstance(adaptive, bool):
            raise TypeError('`adaptive` should be bool type')
        if schedule not in ('center_out', 'alternating'):
            raise ValueError(
                "`schedule` should be 'center_out' or 'alternating'")
        if isinstance(schedule_block_size, bool) or \
                not isinstance(schedule_block_size, int):
            raise TypeError('`schedule_block_size` should be int type')
        if schedule_block_size < 1:
            raise ValueError('`schedule_block_size` should be positive')
        if not isinstance(pad_to_rank, bool):
            raise TypeError('`pad_to_rank` should be bool type')
        if pad_to_rank and not adaptive:
            raise ValueError('`pad_to_rank` requires `adaptive=True`')

        samples = self._initialize_fit(samples, generator)
        if self.outputs.n_sites < 3:
            raise ValueError('TR-RSS requires at least three final sites')
        rank_spec = _normalize_rank_spec(rank, self.outputs.n_sites)
        if block_selector is None and not adaptive:
            selection = PrescribedCentralBlockSelector().select(
                self.outputs.site_dim(self.embeddings),
                rank_spec,
                center=center)
        else:
            if block_selector is None:
                block_selector = CentralBlockSelector()
            elif not isinstance(block_selector, CentralBlockSelector):
                raise TypeError(
                    '`block_selector` should be CentralBlockSelector type')
            selection = block_selector.select(
                self.outputs.site_dim(self.embeddings),
                rank_spec,
                center=center,
                bounds=(1, self.outputs.n_sites - 2))
        if len(selection.sites) == self.outputs.n_sites:
            raise ValueError(
                'The central block should leave at least one boundary site')

        context = self._new_context(
            rank=max(rank_spec),
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            random_projection=False,
            batch_size=batch_size,
            generator=generator,
            collect_metrics=collect_metrics,
            verbose=verbose,
            observer=observer)
        context.state.update({
            'input_samples': samples,
            'labels': labels,
            'rank_spec': rank_spec,
            'rank_caps': rank_spec,
            'selection': selection,
            'loop_opener': loop_opener,
            'schedule': schedule,
            'schedule_block_size': schedule_block_size,
            'adaptive': adaptive,
            'pad_to_rank': pad_to_rank,
        })
        return self._execute(context)

    @staticmethod
    def _discovery_svd(matrix: torch.Tensor,
                       rank: int,
                       truncation: Mapping[str, Any]):
        """Truncates for discovery, dropping numerical null directions."""
        options = dict(truncation)
        options['rank'] = min(rank, min(matrix.shape))
        u, s, vh = truncated_svd(matrix, **options)
        explicit_tolerance = any(
            options.get(name) is not None
            for name in ('cutoff', 'atol', 'rtol', 'cum_percentage'))
        if not explicit_tolerance:
            threshold = max(matrix.shape) * torch.finfo(s.dtype).eps * s[0]
            selected = max(1, int(
                (s > threshold).sum().detach().cpu().item()))
            u = u[..., :selected]
            s = s[..., :selected]
            vh = vh[..., :selected, :]
        return u, s, vh

    @classmethod
    def _selected_rank(cls,
                       matrix: torch.Tensor,
                       rank: int,
                       truncation: Mapping[str, Any]) -> int:
        """Returns a stable truncated rank under one explicit upper bound."""
        return cls._discovery_svd(
            matrix, rank, truncation)[1].shape[-1]

    def _adaptive_ranks(
            self,
            targets: Mapping[Tuple[int, ...], _SketchLocalTarget],
            selection: BlockSelection,
            rank_caps: Sequence[int],
            context: _SketchingFitContext
            ) -> Tuple[Tuple[int, ...], Mapping[str, Any]]:
        """Estimates effective right-link ranks from doubly-trimmed sketches."""
        left = selection.left
        right = selection.right
        tensor = targets[selection.sites].tensor
        truncation = context.spec.truncation.as_kwargs()
        cyclic_cap = rank_caps[-1]

        right_matrix = tensor.reshape(-1, tensor.shape[-1])
        right_u, _, _ = self._discovery_svd(
            right_matrix,
            cyclic_cap * rank_caps[right],
            truncation)
        right_dim = right_u.shape[-1]
        right_u = right_u.reshape(*tensor.shape[:-1], right_dim)

        left_matrix = right_u.reshape(right_u.shape[0], -1)
        _, left_s, left_vh = self._discovery_svd(
            left_matrix,
            cyclic_cap * rank_caps[(left - 1) % len(rank_caps)],
            truncation)
        left_dim = left_s.shape[-1]
        compact = (left_s.unsqueeze(1) * left_vh).reshape(
            left_dim, *tensor.shape[1:-1], right_dim)

        physical_axes = tuple(range(1, compact.ndim - 1))
        physical_virtual = compact.permute(
            *physical_axes, 0, compact.ndim - 1).reshape(
                prod(compact.shape[1:-1]), left_dim * right_dim)
        auxiliary_rank = self._selected_rank(
            physical_virtual,
            rank_caps[(left - 1) % len(rank_caps)] * rank_caps[right],
            truncation)
        estimate = RingRankEstimator().estimate(
            left_dim=left_dim,
            right_dim=right_dim,
            auxiliary_rank=auxiliary_rank,
            rank_caps=(
                rank_caps[(left - 1) % len(rank_caps)],
                rank_caps[right],
                cyclic_cap,
            ))
        if not estimate.feasible:
            raise ValueError(
                'Adaptive TR-RSS rank caps are infeasible: '
                f'{estimate.limitations}')

        ranks = list(rank_caps)
        ranks[-1] = estimate.cyclic_rank
        ranks[(left - 1) % len(ranks)] = estimate.left_rank
        ranks[right] = estimate.right_rank
        site_dim = self.outputs.site_dim(self.embeddings)

        for site in range(left, right):
            matrix = compact.reshape(
                left_dim * prod(site_dim[left:(site + 1)]),
                prod(site_dim[(site + 1):(right + 1)]) * right_dim)
            ranks[site] = self._selected_rank(
                matrix, rank_caps[site], truncation)

        cyclic_estimate = estimate.cyclic_rank_estimate
        for site in range(right + 1, len(ranks) - 1):
            local = targets[(site,)].tensor
            selected = self._selected_rank(
                local.reshape(-1, local.shape[-1]),
                estimate.cyclic_rank * rank_caps[site],
                truncation)
            ranks[site] = min(
                rank_caps[site],
                max(1, int(ceil(selected / cyclic_estimate))))
        for site in range(left - 1, 0, -1):
            local = targets[(site,)].tensor
            selected = self._selected_rank(
                local.reshape(local.shape[0], -1),
                estimate.cyclic_rank * rank_caps[site - 1],
                truncation)
            ranks[site - 1] = min(
                rank_caps[site - 1],
                max(1, int(ceil(selected / cyclic_estimate))))

        return tuple(ranks), {
            'left_trim_rank': left_dim,
            'right_trim_rank': right_dim,
            'auxiliary_rank': auxiliary_rank,
            'cyclic_rank_estimate': estimate.cyclic_rank_estimate,
            'limitations': estimate.limitations,
        }

    @staticmethod
    def _pad_cores(cores: Sequence[torch.Tensor],
                   rank: Sequence[int]) -> Tuple[torch.Tensor, ...]:
        """Pads effective TR ranks to explicit right-link caps with zeros."""
        padded = []
        for site, core in enumerate(cores):
            left_rank = rank[(site - 1) % len(rank)]
            right_rank = rank[site]
            if core.shape[0] > left_rank or core.shape[-1] > right_rank:
                raise ValueError('Effective ranks exceed requested rank caps')
            result = core.new_zeros(left_rank, core.shape[1], right_rank)
            result[:core.shape[0], :, :core.shape[-1]] = core
            padded.append(result)
        return tuple(padded)

    def _build_block_phi(self,
                         sites: Tuple[int, ...],
                         regions: Dict[str, object]) -> PhiOperator:
        components = []
        prefix = regions['prefixes'][sites[0]]
        if len(prefix.region):
            components.append(prefix)
        for site in sites:
            kind, axis = self.outputs.layout[site]
            values = self.domains.for_site(axis) if kind == 'input' \
                else torch.arange(
                    self.outputs.output_shape[axis],
                    device=self.source.device)
            components.append((site, values))
        suffix = regions['suffixes'][sites[-1] + 1]
        if len(suffix.region):
            components.append(suffix)
        return PhiOperator(
            self.source,
            components,
            self.outputs,
            input_sites=self.outputs.input_positions,
            output_sites=self.outputs.positions,
            input_kind=self._input_kind)

    def _decompose(self, context: _SketchingFitContext) -> TRDecomposition:
        selection = context.state['selection']
        target_sites = [selection.sites]
        if context.state['schedule'] == 'alternating':
            target_sites.extend(
                (site,) for site in range(self.outputs.n_sites))
        target_sites.extend(
            (site,) for site in range(self.outputs.n_sites)
            if site not in selection.sites)
        target_sites = list(dict.fromkeys(target_sites))
        phis = [
            self._build_block_phi(sites, context.regions)
            for sites in target_sites]

        builder = _EvaluationPlanBuilder(self.source)
        handles = [phi.collect(builder) for phi in phis]
        local_handles = [
            _collect_local_queries(
                builder, phi, context.local_transform, context)
            for phi in phis]
        input_handles = []
        for sites, phi in zip(target_sites, phis):
            site_handles = {}
            for site in sites:
                axis = phi.layout.index(site)
                queries = self._required_input_queries(
                    site, phi, axis, context)
                site_handles[site] = tuple(
                    phi.collect(builder, query) for query in queries)
            input_handles.append(site_handles)
        _prepare_global_transform(
            builder, context.global_transform, context)
        session = _EvaluationSession(
            builder.freeze(),
            global_transform=context.global_transform,
            context=context)
        with context.phase('source.evaluate'):
            session.evaluate_source(batch_size=context.spec.batch_size)
        with context.phase('values.global_transform'):
            session.prepare_values()
        if context.collect_metrics:
            context.metrics.evaluations.append(session.stats)
        evaluation_view = session.view()
        context.state['input_query_results'] = {}

        targets = {}
        for sites, phi, handle, query_handles, site_handles in zip(
                target_sites, phis, handles, local_handles, input_handles):
            view = _MaterializedPhi(session.result(handle), phi.layout)
            with context.phase('values.local_transform', site=sites[0]):
                view = _apply_local_transform(
                    context.local_transform,
                    view,
                    data=context,
                    evaluation=evaluation_view,
                    query_results=tuple(
                        session.result(item) for item in query_handles))
            tensor = view.materialize()
            for site in sites:
                key = (sites, site)
                context.state['input_query_results'][key] = tuple(
                    session.result(item) for item in site_handles[site])
                axis = phi.layout.index(site)
                fitter = context.input_fitters[site]
                fit_view = _MaterializedPhi(tensor, phi.layout)
                if getattr(fitter, 'requires_functional_phi', False):
                    if len(sites) != 1:
                        raise ValueError(
                            'Functional input fitting requires one-site TR '
                            'targets')
                    if not context.global_transform.is_identity or \
                            not context.local_transform.is_identity:
                        raise ValueError(
                            'A functional input fitter currently requires '
                            'identity value transforms')
                    fit_view = phi
                fitted = self._fit_input_axis(
                    site, fit_view, axis, context)
                tensor = self._select_fitted_tensor(site, fitted)
                context.state['input_query_results'].pop(key)
                for item in site_handles[site]:
                    session.release(item)
            session.release(handle)
            for item in query_handles:
                session.release(item)
            if not len(context.regions['prefixes'][sites[0]].region):
                tensor = tensor.unsqueeze(0)
            if not len(context.regions['suffixes'][sites[-1] + 1].region):
                tensor = tensor.unsqueeze(-1)
            targets[sites] = _SketchLocalTarget(
                tensor=tensor,
                sites=sites,
                truncation=context.spec.truncation.as_kwargs(),
                collect_metrics=context.collect_metrics)

        site_dim = tuple(
            self._select_fitted_tensor(
                site, context.fitted_axes[site]
            ).shape[context.fitted_axes[site].axis]
            for site in range(self.outputs.n_sites))
        context.state['fitted_site_dim'] = site_dim

        rank_spec = context.state['rank_spec']
        adaptive_info = None
        if context.state['adaptive']:
            rank_spec, adaptive_info = self._adaptive_ranks(
                targets,
                selection,
                context.state['rank_caps'],
                context)
            context.state['rank_spec'] = rank_spec
        provider = _SketchTargetProvider(
            targets=targets,
            input_dim=site_dim,
            rank=rank_spec,
            prefixes=context.regions['prefixes'],
            suffixes=context.regions['suffixes'],
            outputs=self.outputs,
            embeddings=self.embeddings,
            boundary_records=[],
            embedding_function=lambda site, values, dtype:
                self._embed_fitted_site(
                    site, values, context, dtype))
        truncation_records = []
        opener = _SketchLoopOpener(
            resolve_loop_opener(context.state['loop_opener']),
            truncation_records)
        with context.phase('core.solve'):
            driver_options = {
                'provider': provider,
                'rank': rank_spec,
                'opener': opener,
                'recursion': SketchGaugeRecursion(),
                'block_selector': _SelectedBlockSelector(selection),
                'center': selection.sites[len(selection.sites) // 2],
                'context': {'generator': context.generator},
            }
            if context.state['schedule'] == 'alternating':
                driver_result = AlternatingRingDriver().fit(
                    **driver_options,
                    fixed_opener=_SketchLoopOpener(
                        FixedGaugeCoreOpener(), truncation_records),
                    block_size=context.state['schedule_block_size'])
            else:
                driver_result = BidirectionalRingDriver().fit(
                    **driver_options)

        context.metrics.local_solves.extend(
            driver_result.metrics.local_solves)
        context.metrics.local_solves.extend(provider.boundary_records)
        context.metrics.gauges.extend(driver_result.metrics.gauges)
        context.metrics.truncations.extend(truncation_records)
        effective_cores = tuple(driver_result.cores)
        context.state['effective_rank'] = tuple(
            core.shape[-1] for core in effective_cores)
        if context.state['pad_to_rank']:
            effective_cores = self._pad_cores(
                effective_cores, context.state['rank_caps'])
        context.cores.extend(effective_cores)
        context.state['driver_result'] = driver_result
        context.state['adaptive_info'] = adaptive_info
        return self._assemble_result(context)

    def _evaluate_extended_cores(self, cores, samples, context):
        vectors = [
            self._embed_fitted_site(
                site,
                values,
                context,
                dtype=cores[0].dtype).to(cores[0].device)
            for site, values in enumerate(samples)]
        matrices = [
            torch.einsum('lpr,bp->blr', core, vector)
            for core, vector in zip(cores, vectors)]
        state = matrices[0]
        for matrix in matrices[1:]:
            state = torch.bmm(state, matrix)
        return state.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    def _assemble_result(self,
                         context: _SketchingFitContext) -> TRDecomposition:
        metadata = {
            'algorithm': 'tr_rss',
            'output_shape': tuple(self.outputs.output_shape),
            'out_position': None if self.outputs.scalar else (
                self.outputs.positions[0]
                if self.outputs.n_output_sites == 1
                else tuple(self.outputs.positions)),
            'center_block': context.state['selection'].sites,
            'schedule': context.state['driver_result'].diagnostics.get(
                'schedule', 'center_out'),
            'requested_schedule': context.state['schedule'],
            'schedule_block_size': context.state['schedule_block_size'],
            'requested_rank': list(context.state['rank_caps']),
            'work_rank': list(context.state['rank_spec']),
            'rank_caps': list(context.state['rank_caps']),
            'effective_rank': list(context.state['effective_rank']),
            'adaptive': context.state['adaptive'],
            'pad_to_rank': context.state['pad_to_rank'],
        }
        if context.state['adaptive_info'] is not None:
            metadata['rank_estimate'] = dict(context.state['adaptive_info'])
        selected = context.state['selected_values']
        if context.collect_metrics and selected is not None:
            approximation = self._evaluate_extended_cores(
                context.cores, context.state['extended_samples'], context)
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
        for site, core in enumerate(context.cores):
            context.emit('site_complete', site=site, values={
                'total_sites': len(context.cores),
                'shape': tuple(core.shape),
            })
            context.emit('core', site=site, level=3, values={'tensor': core})
        return TRDecomposition(
            cores=[context.runtime.finalize(core) for core in context.cores],
            metrics=context.metrics,
            metadata=metadata)

    def _validate_result(self, result, context):
        if not isinstance(result, TRDecomposition):
            raise TypeError('`result` should be TRDecomposition type')
        expected = context.state.get(
            'fitted_site_dim', self.outputs.site_dim(self.embeddings))
        if result.input_dim != expected:
            raise ValueError('The result input dimensions are inconsistent')


class _QuantizedTRRSS(_QuantizedRSSMixin, TRRSS):
    """Internal TRRSS specialization that normalizes physical QTR samples."""

    _quantized_algorithm = 'qtr_rss'


class _QTTTuckerTRRSS(_QTTTuckerFitMixin, TRRSS):
    """Internal TR-RSS driver operating directly on fitted gamma axes."""

    def _assemble_result(
            self, context: _SketchingFitContext) -> TRDecomposition:
        result = super()._assemble_result(context)
        result.metadata['algorithm'] = 'qtr_tucker_rss_upper'
        return result


class QTRTuckerRSS(QTTTuckerRSS):
    r"""Experimental QTT factors connected through an upper Tensor Ring.

    Local factors use the same functional-Phi split as
    :class:`~tensorkrowch.decompositions.QTTTuckerRSS`; only the upper
    connector network is cyclic. The ring construction and the RSS extension
    are experimental, and no open-TT recovery guarantee is implied.

    The local hierarchy follows the QTT-Tucker format of Dolgov and
    Khoromskij (2013),
    `paper <https://doi.org/10.1137/120882597>`_, while respecting the rounding
    caveat of Etter, Dolgov and Khoromskij (2016),
    `paper <https://doi.org/10.1137/15M104089X>`_. The cyclic upper network is
    a TensorKrowch extension not covered by either paper.
    """

    @torch.no_grad()
    def fit(
            self,
            sketch_samples: Samples,
            labels: Optional[torch.Tensor] = None,
            rank: _Rank = 1,
            connector_rank: Optional[int] = None,
            factor_rank: Optional[int] = None,
            center: Optional[int] = None,
            loop_opener: Union[str, LoopOpener] = 'als',
            schedule: str = 'center_out',
            schedule_block_size: int = 1,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: int = 64,
            generator: Optional[torch.Generator] = None,
            sample_space: str = 'physical',
            verbose: Union[bool, int] = 0,
            collect_metrics: bool = False,
            observer: Optional[DecompositionObserver] = None
            ) -> QTRTuckerDecomposition:
        """Fits local QTT factors and their cyclic upper TR connectors.

        ``rank`` follows the TR right-link convention, including the final
        cyclic link. ``connector_rank`` is a shared upper bound for every
        Tucker index and ``factor_rank`` bounds the internal ranks of each
        local QTT factor. The current functional route deliberately uses
        one-site center targets and therefore does not expose adaptive
        multi-site rank discovery.
        """
        indices = self._sample_indices(sketch_samples, sample_space)
        probe = self._evaluate_indices(indices[:1])
        outputs = _OutputSpec.normalize(
            probe, self.layout.n_variables, self.out_position)
        if outputs.n_sites < 3:
            raise ValueError(
                'QTR-Tucker RSS requires at least three upper sites')
        maximum_rank = rank if isinstance(rank, int) else max(tuple(rank))
        if factor_rank is None:
            factor_rank = maximum_rank
        if connector_rank is None:
            connector_rank = maximum_rank
        seed = 0 if generator is None else generator.initial_seed()
        fitters = []
        for kind, axis in outputs.layout:
            if kind == 'output':
                fitters.append(BasisFitter(outputs.output_shape[axis]))
                continue
            size = self.layout.grid_size[axis]
            interval = torch.tensor(
                [0., float(size - 1)],
                device=self.adapter.device,
                dtype=probe.real.dtype)
            fitters.append(QTTInputFitter(
                base=self.layout.base[axis],
                level=self.layout.level[axis],
                digit_order=self.layout.digit_order,
                domain=interval,
                rank=factor_rank,
                connector_rank=connector_rank,
                cutoff=cutoff,
                atol=atol,
                rtol=rtol,
                cum_percentage=cum_percentage,
                batch_size=batch_size,
                seed=int((seed + axis) % (2 ** 63 - 1)),
                materialize_tensor=False))

        index_domains = tuple(
            torch.arange(
                size,
                device=self.adapter.device,
                dtype=probe.real.dtype)
            for size in self.layout.grid_size)
        embeddings = tuple(
            torch.eye(size, device=self.adapter.device, dtype=probe.dtype)
            for size in self.layout.grid_size)

        def indexed_function(values):
            return self._evaluate_indices(values.to(torch.long))

        decomposer = _QTTTuckerTRRSS(
            indexed_function,
            embedding=embeddings,
            input_dim=self.layout.grid_size,
            domain=index_domains,
            out_position=self.out_position,
            device=self.adapter.device,
            dtype=probe.dtype,
            output_device=None,
            input_fitters=fitters,
            synchronize_timers=self.synchronize_timers)
        upper = decomposer.fit(
            indices.to(dtype=probe.real.dtype),
            labels=labels,
            rank=rank,
            center=center,
            loop_opener=loop_opener,
            schedule=schedule,
            schedule_block_size=schedule_block_size,
            adaptive=False,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            batch_size=batch_size,
            generator=generator,
            verbose=verbose,
            collect_metrics=collect_metrics,
            observer=observer)
        if collect_metrics:
            self._merge_factor_metrics(upper, decomposer.factors)
        metadata = dict(upper.metadata)
        metadata.update({
            'algorithm': 'qtr_tucker_rss',
            'experimental': True,
            'rss_recovery_guarantee': False,
            'variable_positions': tuple(decomposer.variable_positions),
            'connector_rank': [
                factor.input_dim[-1] for factor in decomposer.factors],
            'quantization': {
                'base': self.layout.base,
                'level': self.layout.level,
                'ordering': self.layout.ordering,
                'digit_order': self.layout.digit_order,
                'grid_size': self.layout.grid_size,
                'computational_grid': self.adapter.computational_grid,
                'out_of_domain': self.adapter.out_of_domain,
            },
        })
        result = QTRTuckerDecomposition(
            upper,
            decomposer.factors,
            self.layout,
            self.adapter.coordinate_map,
            self.adapter.domain,
            variable_positions=decomposer.variable_positions,
            computational_grid=self.adapter.computational_grid,
            out_of_domain=self.adapter.out_of_domain,
            metrics=upper.metrics,
            metadata=metadata)
        return result if self.output_device is None else result.to(
            device=self.output_device)


def _merge_rs_metrics(tt_metrics: DecompositionMetrics,
                      tr_metrics: DecompositionMetrics
                      ) -> DecompositionMetrics:
    """Combines TT-RS construction and TT-to-TR opening diagnostics."""
    return DecompositionMetrics(
        errors=[
            record for record in tt_metrics.errors
            if record.kind != 'source_support'
        ] + tr_metrics.errors,
        truncations=tt_metrics.truncations + tr_metrics.truncations,
        timings=tt_metrics.timings + tr_metrics.timings,
        evaluations=tt_metrics.evaluations + tr_metrics.evaluations,
        fidelities=tt_metrics.fidelities + tr_metrics.fidelities,
        warnings=tt_metrics.warnings + tr_metrics.warnings,
        local_solves=tt_metrics.local_solves + tr_metrics.local_solves,
        input_fits=tt_metrics.input_fits + tr_metrics.input_fits,
        range_projections=(tt_metrics.range_projections +
                           tr_metrics.range_projections),
        gauges=tt_metrics.gauges + tr_metrics.gauges,
        sweeps=tt_metrics.sweeps + tr_metrics.sweeps)


class TRRS(TTRS):
    r"""Experimental Tensor Ring Recursive Sketching problem.

    The fixed source and recursive sketch operator have the same semantics as
    in :class:`~tensorkrowch.decompositions.TTRS`. Each :meth:`fit` first
    solves the open TT core-determining equations and then opens their loop
    with the common TT-to-TR ring driver. This separates source sketching from
    cyclic gauge construction and lets the latter reuse the same
    ``LoopOpener`` and schedule contracts as TT-to-TR and TR-RSS.

    The open TT equations follow `Generative modeling via tensor train
    sketching <https://arxiv.org/abs/2202.11788>`_ by Hur, Hoskins, Lindsey,
    Stoudenmire and Khoo (2022). Their approximation guarantees are for an
    open TT and do not automatically transfer to this cyclic extension;
    :class:`TRRS` therefore emits
    :class:`~tensorkrowch.decompositions.ExperimentalWarning`.

    The returned :class:`~tensorkrowch.decompositions.TRDecomposition` is a
    lightweight result. Its cores can initialize a periodic
    :class:`~tensorkrowch.models.MPS` with
    ``tk.models.MPS(tensors=result.cores)``.

    Parameters are the same as :class:`TTRS`; exactly one of ``source`` and
    ``dataset`` is required.

    Examples
    --------
    >>> dataset = torch.tensor([[0, 0, 0], [1, 1, 1]])
    >>> decomposer = TRRS(dataset=dataset, input_dim=(2, 2, 2))
    >>> result = decomposer.fit(rank=1)
    >>> result.rank
    [1, 1, 1]
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
        super().__init__(
            source=source,
            dataset=dataset,
            input_dim=input_dim,
            weights=weights,
            sketch_operator=sketch_operator,
            dtype=dtype,
            device=device,
            output_device=None,
            synchronize_timers=synchronize_timers)
        self._tr_output_device = output_device

    @torch.no_grad()
    def fit(
            self,
            rank: int = 1,
            *,
            center: Optional[int] = None,
            loop_opener: Union[str, LoopOpener] = 'als',
            schedule: str = 'center_out',
            schedule_block_size: int = 1,
            gauge_recursion: str = 'pseudoinverse',
            allow_projective_gauges: bool = False,
            gauge_tolerance: float = 1e-8,
            inverse_policy: str = 'pinv',
            rank_rtol: Optional[float] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: Optional[int] = None,
            generator: Optional[torch.Generator] = None,
            strict_system: bool = False,
            warm_start: Optional[TRDecomposition] = None,
            verbose: Union[bool, int] = 0,
            collect_metrics: bool = False,
            observer: Optional[DecompositionObserver] = None
            ) -> TRDecomposition:
        r"""Projects the fixed source and opens the result into a TR.

        ``rank`` is the common maximum rank for the TT solve and every TR
        link, including the cyclic link. Advanced local-loop options remain
        encapsulated by ``loop_opener``. Fidelity between the intermediate TT
        and final TR is always recorded because it validates the loop opening.

        ``verbose`` ranges from 0 (silent), through 1 (sites and summary) and
        2 (system/timing details), to 3 (complete final cores).

        Parameters
        ----------
        rank : int, optional
            Positive maximum rank shared by all links.
        center : int, optional
            Internal site at which cyclic loop opening begins.
        loop_opener : {``"als"``, ``"blostr+als"``} or LoopOpener, optional
            Encapsulated local loop-opening strategy.
        schedule : {``"center_out"``, ``"alternating"``}, optional
            Serial cyclic construction schedule.
        schedule_block_size : int, optional
            Consecutive sites in each alternating block.
        gauge_recursion : {``"pseudoinverse"``, ``"tt_core"``}, optional
            Strategy used to propagate cyclic virtual bases.
        allow_projective_gauges : bool, optional
            Allows rank-deficient directional gauges to propagate projectors.
        gauge_tolerance : float, optional
            Maximum relative gauge-cancellation error.
        inverse_policy : {``"auto"``, ``"solve"``, ``"inverse"``, ``"pinv"``}
            Linear algebra used for directional gauge duals.
        rank_rtol : float, optional
            Relative threshold for pseudoinverses and numerical gauge ranks.
        cutoff, atol, rtol, cum_percentage : float, optional
            Singular-value truncation controls used by the TT-RS systems.
        batch_size : int, optional
            Support or finite-grid evaluation batch size.
        generator : torch.Generator, optional
            Generator owned by this fit for randomized sketches and openings.
        strict_system : bool, optional
            Rejects rank-deficient TT core-determining systems.
        warm_start : TRDecomposition, optional
            Reserved for a future defined update; non-``None`` is rejected.
        verbose : bool or int, optional
            Structured console verbosity from 0 to 3.
        collect_metrics : bool, optional
            Collects TT-RS diagnostics and final sparse-support error.
        observer : DecompositionObserver, optional
            Additional structured-event consumer.

        Returns
        -------
        TRDecomposition
            Lightweight cyclic result stored on ``output_device``.
        """
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError('`rank` should be int type')
        if rank < 1:
            raise ValueError('`rank` should be positive')
        if len(self.source.input_dim) < 3:
            raise ValueError('TR-RS requires at least three sites')
        if warm_start is not None:
            if not isinstance(warm_start, TRDecomposition):
                raise TypeError(
                    '`warm_start` should be TRDecomposition type or None')
            raise NotImplementedError(
                'TR-RS does not yet define a warm-start update; pass None')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')

        warnings.warn(
            'TR-RS is an experimental cyclic extension of open TT-RS; '
            'the TT-RS paper guarantees do not transfer automatically',
            ExperimentalWarning,
            stacklevel=2)
        verbosity = _normalize_verbosity(verbose)
        fit_observer = _resolve_observer(verbosity, observer) \
            if verbosity or observer is not None else None
        if fit_observer is not None:
            fit_observer.emit(DecompositionEvent(
                name='start',
                phase='TR-RS',
                values={
                    'sites': len(self.source.input_dim),
                    'input_dim': self.source.input_dim,
                    'rank': rank,
                    'operator': type(self.sketch_operator).__name__,
                }))

        tt_result = super().fit(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            batch_size=batch_size,
            generator=generator,
            strict_system=strict_system,
            verbose=0,
            collect_metrics=collect_metrics)
        tr_result = TT2TR(
            tt_result,
            output_device=self._tr_output_device).fit(
                rank=rank,
                tr_rank=rank,
                center=center,
                loop_opener=loop_opener,
                schedule=schedule,
                schedule_block_size=schedule_block_size,
                gauge_recursion=gauge_recursion,
                allow_projective_gauges=allow_projective_gauges,
                gauge_tolerance=gauge_tolerance,
                inverse_policy=inverse_policy,
                rank_rtol=rank_rtol,
                verbose=0)
        metrics = _merge_rs_metrics(tt_result.metrics, tr_result.metrics)
        metadata = dict(tr_result.metadata)
        metadata.update({
            'algorithm': 'tr_rs',
            'source_type': type(self.source).__name__,
            'sketch_operator': type(self.sketch_operator).__name__,
            'tt_rank': list(tt_result.rank),
            'experimental': True,
        })
        result = TRDecomposition(
            tr_result.cores,
            metrics=metrics,
            metadata=metadata)

        if collect_metrics and isinstance(self.source, SupportTensorSource):
            samples = self.source.support.as_tensor()
            approximation = result.evaluate(samples)
            target = self.source.support_values.to(
                device=approximation.device, dtype=approximation.dtype)
            absolute = torch.linalg.vector_norm(approximation - target)
            denominator = torch.linalg.vector_norm(target)
            relative = absolute / denominator if denominator > 0 else absolute
            result.metrics.errors.append(ErrorRecord(
                kind='source_support',
                absolute=absolute,
                relative=relative,
                denominator=denominator,
                size=target.shape[0]))

        if fit_observer is not None:
            for site, core in enumerate(result.cores):
                fit_observer.emit(DecompositionEvent(
                    name='site_complete',
                    phase='TR-RS',
                    site=site,
                    values={
                        'total_sites': len(result.cores),
                        'shape': tuple(core.shape),
                    }))
            fit_observer.emit(DecompositionEvent(
                name='summary',
                phase='TR-RS',
                values={
                    'rank': result.rank,
                    'operator': type(self.sketch_operator).__name__,
                    'fidelity': result.metrics.fidelities[-1].fidelity,
                }))
            for site, core in enumerate(result.cores):
                fit_observer.emit(DecompositionEvent(
                    name='core',
                    phase='TR-RS',
                    level=3,
                    site=site,
                    values={'shape': tuple(core.shape), 'tensor': core}))
            fit_observer.close(result.metrics)
        return result


@torch.no_grad()
def tr_rs(
        source=None,
        *,
        dataset: Optional[torch.Tensor] = None,
        input_dim: Optional[Sequence[int]] = None,
        weights: Optional[torch.Tensor] = None,
        sketch_operator: Optional[SketchOperator] = None,
        rank: int = 1,
        center: Optional[int] = None,
        loop_opener: Union[str, LoopOpener] = 'als',
        schedule: str = 'center_out',
        schedule_block_size: int = 1,
        gauge_recursion: str = 'pseudoinverse',
        allow_projective_gauges: bool = False,
        gauge_tolerance: float = 1e-8,
        inverse_policy: str = 'pinv',
        rank_rtol: Optional[float] = None,
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
    """Projects a complete discrete source into TR cores with TR-RS.

    This simple interface constructs :class:`TRRS`, calls :meth:`TRRS.fit`
    and returns only the cores unless ``return_info=True``. Exactly one of
    ``source`` and ``dataset`` is required. The method is experimental and
    warns that open TT-RS guarantees do not transfer automatically.

    Examples
    --------
    >>> dataset = torch.tensor([[0, 0, 0], [1, 1, 1]])
    >>> cores = tr_rs(dataset=dataset, input_dim=(2, 2, 2), rank=1)
    >>> len(cores)
    3
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    result = TRRS(
        source=source,
        dataset=dataset,
        input_dim=input_dim,
        weights=weights,
        sketch_operator=sketch_operator,
        dtype=dtype,
        device=device,
        output_device=output_device).fit(
            rank=rank,
            center=center,
            loop_opener=loop_opener,
            schedule=schedule,
            schedule_block_size=schedule_block_size,
            gauge_recursion=gauge_recursion,
            allow_projective_gauges=allow_projective_gauges,
            gauge_tolerance=gauge_tolerance,
            inverse_policy=inverse_policy,
            rank_rtol=rank_rtol,
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
def qtr_tucker_rss(
        function=None,
        sketch_samples: Samples = None,
        *,
        source=None,
        layout: Optional[QuantizedLayout] = None,
        n_variables: Optional[int] = None,
        base: Union[int, Sequence[int]] = 2,
        level: Union[int, Sequence[int]] = 1,
        ordering: str = 'grouped',
        digit_order: str = 'coarse_to_fine',
        permutation=None,
        coordinate_map: Optional[
            Union[CoordinateMap, Sequence[CoordinateMap]]] = None,
        domain=None,
        source_space: str = 'physical',
        source_layout: Optional[QuantizedLayout] = None,
        sample_space: str = 'physical',
        computational_grid: str = 'endpoints',
        out_of_domain: str = 'error',
        labels: Optional[torch.Tensor] = None,
        out_position=None,
        rank: _Rank = 1,
        connector_rank: Optional[int] = None,
        factor_rank: Optional[int] = None,
        center: Optional[int] = None,
        loop_opener: Union[str, LoopOpener] = 'als',
        schedule: str = 'center_out',
        schedule_block_size: int = 1,
        cutoff: Optional[float] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        cum_percentage: Optional[float] = None,
        batch_size: int = 64,
        device: Device = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        output_device: Device = 'cpu',
        verbose: Union[bool, int] = 0,
        return_info: bool = False):
    """Builds local QTT factors connected through an upper Tensor Ring.

    This is the experimental cyclic counterpart of :func:`qtt_tucker_rss`.
    It returns the complete
    :class:`~tensorkrowch.decompositions.QTRTuckerDecomposition`, not only the
    upper cores, and does not transfer TT/QTT-Tucker recovery guarantees to
    the ring topology.
    """
    if sketch_samples is None:
        raise TypeError('`sketch_samples` should be provided')
    if layout is None and n_variables is None:
        if sample_space != 'physical':
            raise ValueError(
                '`n_variables` is required for non-physical samples')
        values = sketch_samples.values \
            if isinstance(sketch_samples, ConfigurationBatch) \
            else sketch_samples
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise ValueError(
                '`n_variables` could not be inferred from sketch samples')
        n_variables = values.shape[1]
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    result = QTRTuckerRSS(
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
        out_position=out_position,
        device=device,
        dtype=dtype,
        output_device=output_device).fit(
            sketch_samples,
            labels=labels,
            rank=rank,
            connector_rank=connector_rank,
            factor_rank=factor_rank,
            center=center,
            loop_opener=loop_opener,
            schedule=schedule,
            schedule_block_size=schedule_block_size,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            batch_size=batch_size,
            generator=generator,
            sample_space=sample_space,
            verbose=verbose,
            collect_metrics=return_info)
    if return_info:
        return result, result.as_info()
    return result


@torch.no_grad()
def qtr_rss(
        function=None,
        sketch_samples: Samples = None,
        *,
        source=None,
        layout: Optional[QuantizedLayout] = None,
        n_variables: Optional[int] = None,
        base: Union[int, Sequence[int]] = 2,
        level: Union[int, Sequence[int]] = 1,
        ordering: str = 'grouped',
        digit_order: str = 'coarse_to_fine',
        permutation=None,
        coordinate_map: Optional[
            Union[CoordinateMap, Sequence[CoordinateMap]]] = None,
        domain=None,
        source_space: str = 'physical',
        source_layout: Optional[QuantizedLayout] = None,
        sample_space: str = 'physical',
        computational_grid: str = 'endpoints',
        out_of_domain: str = 'error',
        labels: Optional[torch.Tensor] = None,
        out_position=None,
        rank: _Rank = 1,
        center: Optional[int] = None,
        loop_opener: Union[str, LoopOpener] = 'als',
        schedule: str = 'center_out',
        schedule_block_size: int = 1,
        adaptive: bool = False,
        pad_to_rank: bool = False,
        cutoff: Optional[float] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
        cum_percentage: Optional[float] = None,
        batch_size: int = 64,
        device: Device = None,
        dtype: Optional[torch.dtype] = None,
        generator: Optional[torch.Generator] = None,
        output_device: Device = 'cpu',
        verbose: Union[bool, int] = 0,
        return_info: bool = False):
    """Decomposes a multivariable physical function into QTR cores.

    This is the cyclic counterpart of :func:`qtt_rss`; it uses basis digit
    sites and the ordinary TR-RSS ring driver after quantizing physical sketch
    samples. At least three final digit/output sites are required.
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
    decomposer = TRRSS.quantized(
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
        center=center,
        loop_opener=loop_opener,
        schedule=schedule,
        schedule_block_size=schedule_block_size,
        adaptive=adaptive,
        pad_to_rank=pad_to_rank,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
        cum_percentage=cum_percentage,
        batch_size=batch_size,
        generator=generator,
        sample_space=sample_space,
        verbose=verbose,
        collect_metrics=return_info)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


@torch.no_grad()
def tr_rss(function,
           embedding,
           sketch_samples: Samples,
           labels: Optional[torch.Tensor] = None,
           input_dim=None,
           domain=None,
           domain_multiplier: int = 1,
           out_position=None,
           rank: _Rank = 1,
           center: Optional[int] = None,
           loop_opener: Union[str, LoopOpener] = 'als',
           schedule: str = 'center_out',
           schedule_block_size: int = 1,
           adaptive: bool = False,
           pad_to_rank: bool = False,
           cutoff: Optional[float] = None,
           atol: Optional[float] = None,
           rtol: Optional[float] = None,
           cum_percentage: Optional[float] = None,
           batch_size: int = 64,
           device: Device = None,
           dtype: Optional[torch.dtype] = None,
           generator: Optional[torch.Generator] = None,
           output_device: Device = 'cpu',
           verbose: Union[bool, int] = 0,
           return_info: bool = False):
    r"""Decomposes a sampled function into Tensor Ring cores.

    This compatibility function constructs :class:`TRRSS`, calls
    :meth:`TRRSS.fit` and returns its core list. The callable may be scalar or
    tensor-valued; every output axis is represented by a basis site. Input
    samples, embeddings and domains follow :func:`tt_rss`.

    Parameters
    ----------
    function : callable
        Scalar- or tensor-valued function to approximate.
    embedding : callable, torch.Tensor or sequence
        Shared input embedding or one entry per input site.
    sketch_samples : torch.Tensor, sequence of torch.Tensor or ConfigurationBatch
        Correlated sketch samples in packed or per-site form.
    labels : torch.Tensor, optional
        Flattened output labels with shape ``(batch_size,)``.
    input_dim : int or sequence of int, optional
        Expected shared or per-site embedding dimensions.
    domain : torch.Tensor or sequence of torch.Tensor, optional
        Shared finite domain or one domain per input site.
    domain_multiplier : int, optional
        Maximum inferred-domain size in multiples of ``input_dim``.
    out_position : int or sequence of int, optional
        Explicit positions of output sites; defaults to an even distribution.
    rank : int or sequence of int
        Shared rank cap or one right-link cap per final site. The last element
        is the cyclic rank.
    center : int, optional
        Internal center site used to start both recursions.
    loop_opener : {``"als"``, ``"blostr+als"``} or LoopOpener, optional
        Local loop-opening strategy with advanced options encapsulated in it.
    schedule : {``"center_out"``, ``"alternating"``}, optional
        Serial ring-construction schedule. The alternating path is
        experimental and records any center-out fallback.
    schedule_block_size : int, optional
        Number of consecutive sites per alternating block.
    adaptive : bool, optional
        Enables central-block rank discovery under ``rank`` caps.
    pad_to_rank : bool, optional
        Explicitly pads adaptive effective ranks back to their caps.
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
        Compute device.
    dtype : torch.dtype, optional
        Dtype of source values and numerical cores.
    generator : torch.Generator, optional
        Generator used for randomized choices.
    output_device : str, torch.device or None, optional
        Device receiving final cores; CPU by default.
    verbose : bool or int, optional
        Verbosity from 0 to 3.
    return_info : bool, optional
        Whether to return ``(cores, info)`` with structured diagnostics.

    Returns
    -------
    list[torch.Tensor]
        Tensor Ring cores.
    tuple[list[torch.Tensor], dict]
        Cores and structured information when ``return_info=True``.

    Examples
    --------
    >>> domain = torch.tensor([0., 1.])
    >>> samples = torch.cartesian_prod(domain, domain, domain)
    >>> function = lambda data: (1 + data).prod(dim=1)
    >>> embedding = lambda values: torch.stack(
    ...     (1 - values, values), dim=-1)
    >>> cores = tr_rss(
    ...     function, embedding, samples, domain=domain, rank=1)
    >>> [tuple(core.shape) for core in cores]
    [(1, 2, 1), (1, 2, 1), (1, 2, 1)]
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    result = TRRSS(
        function=function,
        embedding=embedding,
        input_dim=input_dim,
        domain=domain,
        domain_multiplier=domain_multiplier,
        out_position=out_position,
        device=device,
        dtype=dtype,
        output_device=output_device).fit(
            sketch_samples=sketch_samples,
            labels=labels,
            rank=rank,
        center=center,
        loop_opener=loop_opener,
        schedule=schedule,
        schedule_block_size=schedule_block_size,
            adaptive=adaptive,
            pad_to_rank=pad_to_rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            batch_size=batch_size,
            generator=generator,
            verbose=verbose,
            collect_metrics=return_info)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


__all__ = [
    'SketchGaugeRecursion',
    'TRRSS',
    'tr_rss',
    'TRRS',
    'tr_rs',
    'qtr_rss',
    'QTRTuckerRSS',
    'qtr_tucker_rss',
]
