"""Conversion of open-boundary tensor trains into tensor rings."""

from dataclasses import dataclass
from typing import (Any, Mapping, Optional, Sequence, Tuple, Union)

import torch

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions.als.convergence import ConvergencePolicy
from tensorkrowch.decompositions.metrics import (ErrorRecord, FidelityRecord,
                                                 TimingRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   DecompositionObserver,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import (TTDecomposition,
                                                 TRDecomposition)
from tensorkrowch.decompositions.ring.blostr import BLOSTRLoopOpener
from tensorkrowch.decompositions.ring.blocks import (BlockSelection,
                                                     CentralBlockSelector)
from tensorkrowch.decompositions.ring.driver import (BoundaryClosure,
                                                     BidirectionalRingDriver)
from tensorkrowch.decompositions.ring.gauges import (
    GaugeRecursion,
    PseudoinverseGaugeRecursion,
    TTCoreGaugeRecursion,
)
from tensorkrowch.decompositions.ring.opening import (ALSLoopOpener,
                                                      CallableLoopOpener,
                                                      CompositeLoopOpener,
                                                      LoopOpener,
                                                      LoopOpenerCapabilities)
from tensorkrowch.decompositions.sources.tt import TTTensorSource


_Device = Optional[Union[str, torch.device]]


def _as_tt_decomposition(tt) -> TTDecomposition:
    """Normalizes TT results, raw cores and open-boundary MPS adapters."""
    if isinstance(tt, TTDecomposition):
        if tt.n_batches:
            raise ValueError('TT-to-TR does not support decomposition batches')
        result = tt
    else:
        source = TTTensorSource(tt)
        compact_cores = list(source.cores)
        if len(compact_cores) == 1:
            compact_cores[0] = compact_cores[0].reshape(-1)
        else:
            compact_cores[0] = compact_cores[0].squeeze(0)
            compact_cores[-1] = compact_cores[-1].squeeze(-1)
        result = TTDecomposition(compact_cores)
    if len(result.cores) < 3:
        raise ValueError('TT-to-TR requires at least three TT cores')
    return result


def _normalize_rank(rank: int,
                    tr_rank: Optional[int],
                    n_sites: int) -> Tuple[int, ...]:
    """Builds one prescribed right-link rank per final TR core."""
    for name, value in (('rank', rank), ('tr_rank', tr_rank)):
        if value is None and name == 'tr_rank':
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f'`{name}` should be int type')
        if value < 1:
            raise ValueError(f'`{name}` should be positive')
    cyclic_rank = rank if tr_rank is None else tr_rank
    return (rank,) * (n_sites - 1) + (cyclic_rank,)


class _PrescribedCentralBlockSelector(CentralBlockSelector):
    """Selects one fixed-rank center without adaptive injectivity growth."""

    def select(self,
               provider: Any,
               rank,
               center: Optional[int] = None,
               *,
               bounds: Optional[Tuple[int, int]] = None) -> BlockSelection:
        input_dim = tuple(provider.input_dim)
        if center is None:
            center = len(input_dim) // 2
        if isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int type or None')
        if center <= 0 or center >= len(input_dim) - 1:
            raise ValueError('`center` should be an internal TT site')
        rank = tuple(rank)
        return BlockSelection(
            sites=(center,),
            input_dim=(input_dim[center],),
            left_rank_cap=rank[center - 1],
            right_rank_cap=rank[center],
            input_capacity=input_dim[center],
            required_input_capacity=1,
            feasible=True,
            reason='prescribed_fixed_rank_center',
            boundary=None,
            growth=((center, center),))


@dataclass
class _TTCoreProvider:
    """Exposes TT supercores and open-edge absorptions to the ring driver."""

    tt: TTDecomposition

    boundary_mode = 'open'

    def __post_init__(self) -> None:
        source = TTTensorSource(self.tt)
        self.cores = tuple(source.cores)
        self.input_dim = source.input_dim

    def local_target(self,
                     sites: Sequence[int],
                     context: Mapping[str, Any]) -> torch.Tensor:
        sites = tuple(sites)
        if not sites or sites != tuple(range(sites[0], sites[-1] + 1)):
            raise ValueError('TT local sites should form a contiguous interval')
        tensor = self.cores[sites[0]]
        for site in sites[1:]:
            tensor = torch.tensordot(
                tensor, self.cores[site], dims=([-1], [0]))
        return tensor

    def local_rank(self,
                   sites: Sequence[int],
                   rank,
                   context: Mapping[str, Any]) -> Tuple[int, ...]:
        sites = tuple(sites)
        rank = tuple(rank)
        return (
            rank[(sites[0] - 1) % len(rank)],
            *(rank[site] for site in sites),
            rank[-1])

    def local_context(self,
                      sites: Sequence[int],
                      context: Mapping[str, Any]) -> Mapping[str, Any]:
        target = self.local_target(sites, context)
        return {
            'input_dim': tuple(target.shape),
            'dtype': target.dtype,
            'device': target.device,
        }

    def close_boundary(self,
                       site: int,
                       direction: str,
                       opening,
        context: Mapping[str, Any]) -> BoundaryClosure:
        """Absorbs the propagated gauge into the matching unit TT edge."""
        if direction == 'left':
            gauge = context.get('boundary_gauge', opening.left_gauge)
            if site != 0 or gauge is None:
                raise ValueError('Invalid left TT boundary closure')
            edge = self.cores[site].squeeze(0)
            core = torch.einsum(
                'dm,cmr->cdr', edge, gauge)
        elif direction == 'right':
            gauge = context.get('boundary_gauge', opening.right_gauge)
            if site != len(self.cores) - 1 or gauge is None:
                raise ValueError('Invalid right TT boundary closure')
            edge = self.cores[site].squeeze(-1)
            core = torch.einsum(
                'rmc,md->rdc', gauge, edge)
        else:
            raise ValueError("`direction` should be 'left' or 'right'")
        return BoundaryClosure(
            site=site,
            direction=direction,
            core=core,
            diagnostics={'algorithm': 'tt_boundary_absorption'})


def _resolve_loop_opener(loop_opener) -> LoopOpener:
    """Normalizes the simple ALS preset or one advanced opening strategy."""
    def als_opener() -> ALSLoopOpener:
        return ALSLoopOpener({
            'gauge': 'none',
            'normalize': False,
            'convergence': ConvergencePolicy(
                max_sweeps=100,
                error_rtol=1e-10,
                keep_best=True),
        })
    if isinstance(loop_opener, str):
        if loop_opener == 'als':
            return als_opener()
        if loop_opener == 'blostr+als':
            return CompositeLoopOpener(
                BLOSTRLoopOpener(),
                als_opener(),
                fallback_on_error=True)
        raise ValueError(
            "`loop_opener` should be 'als', 'blostr+als' or an advanced "
            'opening strategy')
    if isinstance(loop_opener, LoopOpener):
        return loop_opener
    if callable(loop_opener):
        return CallableLoopOpener(
            loop_opener,
            LoopOpenerCapabilities(
                supports_fixed_left=True,
                supports_fixed_right=True,
                supports_two_fixed_gauges=True,
                supports_blocks=True))
    raise TypeError(
        '`loop_opener` should be "als", a LoopOpener or a callable')


def _resolve_gauge_recursion(
        gauge_recursion,
        *,
        inverse_policy: str,
        allow_projective: bool,
        tolerance: float,
        rank_rtol: Optional[float]) -> GaugeRecursion:
    """Normalizes stable and experimental gauge-recursion strategies."""
    options = {
        'inverse_policy': inverse_policy,
        'allow_projective': allow_projective,
        'tolerance': tolerance,
        'rank_rtol': rank_rtol,
    }
    if isinstance(gauge_recursion, str):
        if gauge_recursion == 'pseudoinverse':
            return PseudoinverseGaugeRecursion(**options)
        if gauge_recursion == 'tt_core':
            return TTCoreGaugeRecursion(**options)
        raise ValueError(
            "`gauge_recursion` should be 'pseudoinverse', 'tt_core' or a "
            'GaugeRecursion object')
    if isinstance(gauge_recursion, GaugeRecursion):
        return gauge_recursion
    raise TypeError(
        '`gauge_recursion` should be str type or implement GaugeRecursion')


def _fidelity_error(tt: TTDecomposition,
                    tr: TRDecomposition) -> FidelityRecord:
    """Computes phase-aware fidelity and relative L2 error without densifying."""
    normalized_overlap = tt.normalized_overlap(tr)
    target_norm = tt.norm()
    approximation_norm = tr.norm()
    if target_norm <= 0:
        raise ValueError('TT-to-TR fidelity is undefined for a zero TT')
    norm_ratio = approximation_norm / target_norm
    # This equivalent form avoids subtracting ``2 * norm_ratio`` from two
    # order-one squared norms when their magnitudes are slightly different.
    # The overlap term still has the unavoidable square-root precision floor
    # of contraction-based residuals for nearly identical networks.
    relative_squared = (
        (1 - norm_ratio).square() +
        2 * norm_ratio * (1 - normalized_overlap.real)).clamp_min(0)
    relative = relative_squared.sqrt()
    absolute = relative * target_norm
    error = ErrorRecord(
        kind='reconstruction',
        absolute=absolute,
        relative=relative,
        denominator=target_norm)
    return FidelityRecord(normalized_overlap, error=error)


class TT2TR:
    """Converts one fixed TT into a tensor ring by local loop openings.

    The TT is fixed on construction, while :meth:`fit` can compare prescribed
    TR ranks, cyclic ranks, centers and opening strategies. Inputs may be a
    :class:`~tensorkrowch.decompositions.TTDecomposition`, raw open-boundary TT
    cores or an open-boundary :class:`~tensorkrowch.models.MPS` adapter.

    The returned :class:`~tensorkrowch.decompositions.TRDecomposition` is a
    lightweight result. Its cores can initialize a periodic
    :class:`~tensorkrowch.models.MPS` through
    ``tk.models.MPS(tensors=result.cores)``.

    Parameters
    ----------
    tt : TTDecomposition, sequence of torch.Tensor or MPS
        Open-boundary TT to convert. At least three sites are required.
    output_device : str or torch.device, optional
        Device where finalized TR cores are stored. The default is ``"cpu"``;
        ``None`` keeps them on the input device.
    """

    def __init__(self,
                 tt,
                 *,
                 output_device: _Device = 'cpu') -> None:
        self._tt = _as_tt_decomposition(tt)
        self._runtime = _RuntimePolicy.from_tensor(
            self._tt.cores[0], output_device=output_device)

    @property
    def tt(self) -> TTDecomposition:
        """Open-boundary TT fixed for repeated conversions."""
        return self._tt

    def fit(self,
            rank: int,
            tr_rank: Optional[int] = None,
            center: Optional[int] = None,
            loop_opener: Union[str, LoopOpener] = 'als',
            gauge_recursion: Union[str, GaugeRecursion] = 'pseudoinverse',
            allow_projective_gauges: bool = False,
            gauge_tolerance: float = 1e-8,
            inverse_policy: str = 'pinv',
            rank_rtol: Optional[float] = None,
            verbose: Union[bool, int] = 0,
            observer: Optional[
                DecompositionObserver] = None) -> TRDecomposition:
        """Converts the TT with prescribed non-cyclic and cyclic TR ranks.

        ``rank`` is used for every non-cyclic right link. ``tr_rank`` sets the
        final cyclic link and defaults to ``rank``. These are prescribed fixed
        ranks: they are never silently reduced. A future adaptive mode will be
        explicit and separate.

        The center is opened without fixed gauges. Two independent sweeps then
        propagate directional pseudoinverses toward the TT boundaries, whose
        unit ranks are absorbed into the first and last TR cores. Fidelity,
        normalized overlap and absolute/relative reconstruction error are
        always measured by scaled TT/TR contractions without densifying.

        .. warning::
           ``gauge_recursion="tt_core"`` is experimental. It emits
           :class:`~tensorkrowch.decompositions.ExperimentalWarning` and may
           accept projected local transports only when
           ``allow_projective_gauges=True``.

        Parameters
        ----------
        rank : int
            Positive rank prescribed on every non-cyclic TR link.
        tr_rank : int, optional
            Positive cyclic rank. Defaults to ``rank``.
        center : int, optional
            Internal TT site opened first. Defaults to the middle site.
        loop_opener : {``"als"``, ``"blostr+als"``}, LoopOpener or callable
            Local loop-opening strategy. The simple preset uses exact TR-ALS;
            advanced ALS options should be encapsulated in an
            :class:`~tensorkrowch.decompositions.ALSLoopOpener`.
            ``"blostr+als"`` tries an experimental spectral initialization
            and falls back cleanly to the same ALS path if BLOSTR assumptions
            are not satisfied.
        gauge_recursion : {``"pseudoinverse"``, ``"tt_core"``} or GaugeRecursion
            Strategy used to propagate virtual bases. ``"pseudoinverse"`` is
            the stable characterized default. ``"tt_core"`` uses the
            original TT cores as recursive projectors and is experimental.
        allow_projective_gauges : bool
            Whether rank-deficient directional pseudoinverses may propagate a
            projector instead of cancelling exactly.
        gauge_tolerance : float
            Maximum relative error accepted for gauge cancellation.
        inverse_policy : {``"auto"``, ``"solve"``, ``"inverse"``, ``"pinv"``}
            Linear algebra used to construct directional gauge duals.
        rank_rtol : float, optional
            Relative singular-value threshold for pseudoinverses and numerical
            gauge ranks.
        verbose : bool or int
            Console verbosity from 0 (silent) to 3 (final cores included).
        observer : DecompositionObserver, optional
            Additional consumer of structured conversion events.

        Returns
        -------
        TRDecomposition
            Lightweight cyclic decomposition with structured local, gauge,
            fidelity, error and timing metrics.

        Examples
        --------
        Convert a rank-one TT while preserving its represented tensor:

        >>> tt = [torch.tensor([[1.], [2.]]),
        ...       torch.tensor([[[1.], [3.]]]),
        ...       torch.tensor([[2., 1.]])]
        >>> result = TT2TR(tt).fit(rank=1)
        >>> result.rank
        [1, 1, 1]
        >>> result.metrics.fidelities[0].fidelity > 0.999
        True
        """
        if not isinstance(allow_projective_gauges, bool):
            raise TypeError('`allow_projective_gauges` should be bool type')
        rank_spec = _normalize_rank(rank, tr_rank, len(self.tt.cores))
        if center is None:
            center = len(self.tt.cores) // 2
        verbosity = _normalize_verbosity(verbose)
        fit_observer = _resolve_observer(verbosity, observer) \
            if verbosity or observer is not None else None
        opener = _resolve_loop_opener(loop_opener)
        recursion = _resolve_gauge_recursion(
            gauge_recursion,
            inverse_policy=inverse_policy,
            allow_projective=allow_projective_gauges,
            tolerance=gauge_tolerance,
            rank_rtol=rank_rtol)
        provider = _TTCoreProvider(self.tt)

        if fit_observer is not None:
            fit_observer.emit(DecompositionEvent(
                name='start',
                phase='TT to TR',
                values={
                    'sites': len(self.tt.cores),
                    'input_dim': self.tt.input_dim,
                    'rank': rank_spec,
                    'center': center,
                }))

        with self._runtime.timer() as timer:
            driver_result = BidirectionalRingDriver().fit(
                provider=provider,
                rank=rank_spec,
                opener=opener,
                recursion=recursion,
                block_selector=_PrescribedCentralBlockSelector(),
                center=center)
            active_result = driver_result.as_decomposition()
            fidelity = _fidelity_error(self.tt, active_result)
            active_result.metrics.fidelities.append(fidelity)
            active_result.metrics.errors.append(fidelity.error)

        active_result.metrics.timings.append(TimingRecord(
            name='fit', elapsed=timer.elapsed))
        if list(active_result.rank) != list(rank_spec):
            raise RuntimeError(
                'TT-to-TR changed prescribed ranks: '
                f'{active_result.rank} != {list(rank_spec)}')
        metadata = dict(active_result.metadata)
        metadata.update({
            'algorithm': 'tt2tr',
            'center': center,
            'requested_rank': list(rank_spec),
            'adaptive': False,
            'inverse_policy': inverse_policy,
            'gauge_recursion': (
                gauge_recursion if isinstance(gauge_recursion, str)
                else type(gauge_recursion).__name__),
            'allow_projective_gauges': allow_projective_gauges,
        })
        result = TRDecomposition(
            cores=[self._runtime.finalize(core)
                   for core in active_result.cores],
            metrics=active_result.metrics,
            metadata=metadata)

        if fit_observer is not None:
            for step, (sites, direction) in enumerate(zip(
                    driver_result.order, driver_result.directions)):
                fit_observer.emit(DecompositionEvent(
                    name='site_complete',
                    phase='TT to TR',
                    site=sites[0],
                    values={
                        'step': step + 1,
                        'total_sites': len(driver_result.order),
                        'sites': sites,
                        'direction': direction,
                    }))
            for record in result.metrics.gauges:
                fit_observer.emit(DecompositionEvent(
                    name='gauge',
                    phase='TT to TR',
                    level=2,
                    site=record.site,
                    values={
                        'orientation': record.orientation,
                        'shape': record.shape,
                        'rank': (
                            f'{record.numerical_rank}/'
                            f'{record.cancellable_rank}'),
                        'condition_number': record.condition_number,
                        'cancellation_error': record.cancellation_error,
                        'projective': record.projective,
                    }))
            fit_observer.emit(DecompositionEvent(
                name='summary',
                phase='TT to TR',
                values={
                    'rank': result.rank,
                    'fidelity': fidelity.fidelity,
                    'absolute_error': fidelity.error.absolute,
                    'relative_error': fidelity.error.relative,
                    'elapsed': f'{timer.elapsed:.6f} s',
                }))
            for site, core in enumerate(result.cores):
                fit_observer.emit(DecompositionEvent(
                    name='core',
                    phase='TT to TR',
                    level=3,
                    site=site,
                    values={'shape': tuple(core.shape), 'tensor': core}))
            fit_observer.close(result.metrics)
        return result


def tt2tr(tt,
          rank: int,
          tr_rank: Optional[int] = None,
          center: Optional[int] = None,
          loop_opener: Union[str, LoopOpener] = 'als',
          gauge_recursion: Union[str, GaugeRecursion] = 'pseudoinverse',
          allow_projective_gauges: bool = False,
          gauge_tolerance: float = 1e-8,
          inverse_policy: str = 'pinv',
          rank_rtol: Optional[float] = None,
          output_device: _Device = 'cpu',
          verbose: Union[bool, int] = 0,
          return_info: bool = False):
    """Converts open-boundary TT cores into prescribed tensor-ring cores.

    This is the simple functional interface. Use :class:`TT2TR` for repeated
    conversions of the same TT or direct access to the lightweight result.
    ``rank`` controls all non-cyclic links and ``tr_rank`` controls the cyclic
    closing link. The local ALS configuration remains encapsulated by
    ``loop_opener`` rather than expanding this function's signature.

    Parameters are equivalent to :meth:`TT2TR.fit`, with ``output_device``
    selecting final core storage and ``return_info=True`` returning
    ``(cores, info)``.

    Examples
    --------
    >>> tt = [torch.tensor([[1.], [2.]]),
    ...       torch.tensor([[[1.], [3.]]]),
    ...       torch.tensor([[2., 1.]])]
    >>> cores = tt2tr(tt, rank=1)
    >>> [tuple(core.shape) for core in cores]
    [(1, 2, 1), (1, 2, 1), (1, 2, 1)]
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    result = TT2TR(tt, output_device=output_device).fit(
        rank=rank,
        tr_rank=tr_rank,
        center=center,
        loop_opener=loop_opener,
        gauge_recursion=gauge_recursion,
        allow_projective_gauges=allow_projective_gauges,
        gauge_tolerance=gauge_tolerance,
        inverse_policy=inverse_policy,
        rank_rtol=rank_rtol,
        verbose=verbose)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


__all__ = ['TT2TR', 'tt2tr']
