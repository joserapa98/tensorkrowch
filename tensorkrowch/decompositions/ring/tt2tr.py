"""
This script contains:

    Internal classes:
        * _TTCoreProvider
        * _GramTTCoreProvider
        * _CanonicalTTCoreProvider
        * _VidalCoreProvider

    Public classes:
        * TT2TR

    Internal functions:
        * _as_tt
        * _normalize_rank
        * _resolve_recursion
        * _fidelity_error

    Public functions:
        * tt2tr
"""

from dataclasses import dataclass
from typing import (Any,
                    Mapping,
                    Optional,
                    Sequence,
                    TYPE_CHECKING,
                    Tuple,
                    Union)

import torch

from tensorkrowch.formats import TT
from tensorkrowch.formats.bonds import VidalGauge
from tensorkrowch.formats.formats1d import _restore_cores
from tensorkrowch.models.mps import MPS

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions._utils import _normalize_positive_int
from tensorkrowch.decompositions.metrics import (ErrorRecord,
                                                 FidelityRecord,
                                                 TimingRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import (TRDecomposition,
                                                 TTDecomposition)
from tensorkrowch.decompositions.ring.blocks import (CentralBlockSelector,
                                                    PrescribedCentralBlockSelector)
from tensorkrowch.decompositions.ring.driver import (BidirectionalRingDriver,
                                                     BoundaryClosure)
from tensorkrowch.decompositions.ring.gauges import (AdjointGaugeRecursion,
                                                     GaugeRecursion,
                                                     GaugeRecursionStep,
                                                     GramGaugeRecursion,
                                                     OrthGaugeRecursion,
                                                     PinvGaugeRecursion,
                                                     _ExpandedGaugeRecursion,
                                                     _schmidt_power)
from tensorkrowch.decompositions.ring.opening import (FixedGaugeCoreOpener,
                                                      LoopOpener,
                                                      resolve_loop_opener)
from tensorkrowch.decompositions.ring.schedules import AlternatingRingDriver
from tensorkrowch.decompositions.sources.format import FormatSource


_Device = Optional[Union[str, torch.device]]


if TYPE_CHECKING:
    from tensorkrowch.decompositions.results import _DecompositionOutput


def _as_tt(
    tt: Union['TT', Sequence[torch.Tensor], 'MPS']
) -> TT:
    """Normalizes TT results, raw cores and open-boundary MPS adapters."""
    if isinstance(tt, MPS):
        if tt.boundary != 'obc':
            raise ValueError('`tt` should have open boundaries')
        tt = tt.to_tt()
    if isinstance(tt, TT):
        if tt.n_batches:
            raise ValueError('TT-to-TR does not support structural batches')
        result = tt
    else:
        if isinstance(tt, (str, bytes)) or not isinstance(tt, Sequence):
            raise TypeError('`tt` should be a TT, open MPS or core sequence')
        source = FormatSource(tt)
        compact_cores = source.format._effective_cores()
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
    """Builds one prescribed right-link ``rank`` per final TR core."""
    rank = _normalize_positive_int(rank, 'rank')
    cyclic_rank = rank if tr_rank is None else \
        _normalize_positive_int(tr_rank, 'tr_rank')
    return (rank,) * (n_sites - 1) + (cyclic_rank,)


@dataclass
class _TTCoreProvider:  # MARK: _TTCoreProvider
    """Exposes TT supercores and open-edge absorptions to the ring driver."""

    tt: TT  # Original open-boundary format supplying local targets

    boundary_mode = 'open'

    def __post_init__(self) -> None:
        source = FormatSource(self.tt.to_tt() if self.tt._quantized else self.tt)
        self.cores = tuple(source.format._effective_cores())
        self.in_dim = tuple(core.shape[1] for core in self.cores)

    def local_target(self,
                     sites: Sequence[int],
                     context: Mapping[str, Any]) -> torch.Tensor:
        """
        Returns the TT core or contracted supercore for the requested
        ``sites``.
        """
        sites = tuple(sites)
        if not sites or sites != tuple(range(sites[0], sites[-1] + 1)):
            raise ValueError('TT local sites should form a contiguous interval')
        tensor = self.cores[sites[0]]
        for site in sites[1:]:
            tensor = torch.tensordot(
                tensor, self.cores[site], dims=([-1], [0]))
        return tensor

    def expand_coordinates(self,
                           site: int,
                           count: int,
                           direction: str) -> None:
        """Adds zero reference rows and unit-metric auxiliary coordinates."""
        if not count:
            return
        self.cores = list(self.cores)
        core = self.cores[site]
        axis = 0 if direction == 'right' else 2
        shape = list(core.shape)
        shape[axis] = count
        self.cores[site] = torch.cat((core, core.new_zeros(shape)), dim=axis)
        name = 'left_factors' if direction == 'right' else 'right_factors'
        factors = getattr(self, name, None)
        if factors is not None:
            factors = list(factors)
            index = site if direction == 'right' else site + 1
            unit = torch.eye(count, dtype=core.dtype, device=core.device)
            factors[index] = torch.block_diag(factors[index], unit)
            setattr(self, name, factors)

    def local_rank(self,
                   sites: Sequence[int],
                   rank: Union[int, Sequence[int]],
                   context: Mapping[str, Any]) -> Tuple[int, ...]:
        """Maps global prescribed ranks to the local cyclic factorization."""
        sites = tuple(sites)
        rank = tuple(rank)
        return (
            rank[(sites[0] - 1) % len(rank)],
            *(rank[site] for site in sites),
            rank[-1])

    def local_context(self,
                      sites: Sequence[int],
                      context: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Adds target dimensions and runtime to the local execution
        ``context``.
        """
        target = self.local_target(sites, context)
        return {
            **context,
            'in_dim': tuple(target.shape),
            'dtype': target.dtype,
            'device': target.device,
        }

    def close_boundary(self,
                       site: int,
                       direction: str,
                       opening: Any,
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

    def prepare_boundary(self,
                         opening: Any,
                         local_target: Any,
                         recursion_context: Mapping[str, Any],
                         direction: str) -> GaugeRecursionStep:
        """Keeps the learned outgoing gauge for direct TT-edge absorption."""
        if direction == 'right':
            gauge = opening.right_gauge
        elif direction == 'left':
            gauge = opening.left_gauge
        else:
            raise ValueError("`direction` should be 'left' or 'right'")
        return GaugeRecursionStep(
            gauge=gauge,
            diagnostics={'boundary': True,
                         'algorithm': 'absorb_learned_gauge'})


class _CanonicalTTCoreProvider(_TTCoreProvider):  # MARK: _CanonicalTTCoreProvider
    """Moves independent canonical centers without changing solved targets."""

    def __init__(self, tt: TT, center: int) -> None:
        self.center = center
        super().__init__(tt)

    def __post_init__(self) -> None:
        super().__post_init__()
        canonical = TT(_restore_cores(
            self.cores, self.in_dim, None, 0, False)).canonicalize(
                orth_center=self.center)
        self.cores = [core.clone() for core in canonical._effective_cores()]

    def shift_center(self,
                     site: int,
                     factor: torch.Tensor,
                     direction: str) -> None:
        """Absorbs a zero-padded QR factor into the next unsolved core."""
        if direction == 'right':
            self.cores[site] = torch.einsum(
                'xa,apb->xpb', factor, self.cores[site])
        else:
            self.cores[site] = torch.einsum(
                'apb,xb->apx', self.cores[site], factor)


class _GramTTCoreProvider(_TTCoreProvider):  # MARK: _GramTTCoreProvider
    """Caches QR square roots of all prefix and transposed-suffix Grams."""

    def __post_init__(self) -> None:
        super().__post_init__()
        left = [self.cores[0].new_ones(1, 1)]
        for core in self.cores:
            extended = torch.einsum('xa,apb->xpb', left[-1], core)
            left.append(torch.linalg.qr(
                extended.reshape(-1, core.shape[-1]), mode='r').R)
        right = [self.cores[0].new_ones(1, 1)]
        for core in reversed(self.cores):
            extended = torch.einsum('xb,apb->pxa', right[-1], core)
            right.append(torch.linalg.qr(
                extended.reshape(-1, core.shape[0]), mode='r').R)
        self.left_factors = tuple(left)
        self.right_factors = tuple(reversed(right))


class _VidalCoreProvider(_TTCoreProvider):  # MARK: _VidalCoreProvider
    """Supplies canonical Vidal cores and their remaining bond factors."""

    def __post_init__(self) -> None:
        tt = self.tt.to_tt() if self.tt._quantized else self.tt
        bonds = tt.bonds
        if not isinstance(bonds, VidalGauge) or not bonds._valid:
            raise ValueError('Adjoint Vidal transport requires valid Vidal data')
        self.in_dim = tuple(tt.in_dim)
        self.cores = list(tt._standard_cores())
        self.spectra = tuple(bonds.spectra)
        self.powers = tuple(bonds.powers)
        self.bond_factors = tuple(_schmidt_power(s, 1 - left - right)
                                  for s, (left, right) in
                                  zip(self.spectra, self.powers))
        # Auxiliary expansion coordinates carry unit metric, not Schmidt data.
        unit = self.cores[0].new_ones(1, 1)
        self.left_factors = (unit, *(torch.diag(_schmidt_power(s, 1 - right)).to(
            self.cores[0].dtype) for s, (_, right) in
            zip(self.spectra, self.powers)), unit)
        self.right_factors = (unit, *(torch.diag(_schmidt_power(s, 1 - left)).to(
            self.cores[0].dtype) for s, (left, _) in
            zip(self.spectra, self.powers)), unit)

    def local_target(self,
                     sites: Sequence[int],
                     context: Mapping[str, Any]) -> torch.Tensor:
        """Contracts Vidal cores with each remaining internal factor once."""
        sites = tuple(sites)
        if not sites or sites != tuple(range(sites[0], sites[-1] + 1)):
            raise ValueError('TT local sites should form a contiguous interval')
        tensor = self.cores[sites[0]]
        for site in sites[1:]:
            tensor = tensor * self.bond_factors[site - 1]
            tensor = torch.tensordot(
                tensor, self.cores[site], dims=([-1], [0]))
        return tensor

    def prepare_boundary(self,
                         opening: Any,
                         local_target: Any,
                         recursion_context: Mapping[str, Any],
                         direction: str) -> GaugeRecursionStep:
        """Includes the one remaining Vidal bond factor in the learned gauge."""
        step = super().prepare_boundary(
            opening, local_target, recursion_context, direction)
        sites = recursion_context['from_sites']
        bond = sites[-1] if direction == 'right' else sites[0] - 1
        gauge = step.gauge * self.bond_factors[bond][None, :, None]
        return GaugeRecursionStep(gauge=gauge, diagnostics=step.diagnostics)


def _resolve_recursion(recursion: Any,
                       *,
                       inverse_policy: str,
                       allow_projective: bool,
                       tolerance: float,
                       rank_rtol: Optional[float]) -> GaugeRecursion:
    """Resolves one of four transport families or a custom strategy."""
    if isinstance(recursion, str):
        strategies = {'adjoint': AdjointGaugeRecursion,
                      'pinv': PinvGaugeRecursion,
                      'gram': GramGaugeRecursion,
                      'orth': OrthGaugeRecursion}
        if recursion not in strategies:
            raise ValueError(
                "`recursion` should be 'adjoint', 'pinv', 'gram' or 'orth'")
        return strategies[recursion](
            inverse_policy=inverse_policy, allow_projective=allow_projective,
            tolerance=tolerance, rank_rtol=rank_rtol)
    if isinstance(recursion, GaugeRecursion):
        return recursion
    raise TypeError('`recursion` should be str type or implement GaugeRecursion')


def _fidelity_error(tt: TT,
                    tr: TRDecomposition) -> FidelityRecord:
    """
    Computes phase-aware fidelity and relative L2 error without densifying.
    """
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


class TT2TR:  # MARK: TT2TR
    """
    Converts one fixed TT into a tensor ring by local loop openings.

    The TT is fixed on construction, while :meth:`fit` can compare prescribed
    TR ranks, cyclic ranks, centers and opening strategies. Inputs may be a
    :class:`~tensorkrowch.decompositions.TTDecomposition`, raw open-boundary TT
    cores or an open-boundary :class:`~tensorkrowch.models.MPS` adapter.

    The returned :class:`~tensorkrowch.decompositions.TRDecomposition` is a
    lightweight result. Its cores can initialize a periodic
    :class:`~tensorkrowch.models.MPS` through
    ``tk.models.MPS(tensors=result.cores)``.

    The two TT edge cores absorb their adjacent learned outgoing gauges
    directly. Vidal representations include the remaining explicit or inverse
    bond factor once. No recursive-coordinate pseudoinverse is used to close
    an edge.

    Parameters
    ----------
    tt : TT, sequence of torch.Tensor or MPS
        Open-boundary TT to convert. At least three sites are required.
        Valid Vidal metadata are retained for adjoint transport.
    out_device : str or torch.device, optional
        Device where finalized TR cores are stored. The default is ``"cpu"``;
        ``None`` keeps them on the input device.
    """

    def __init__(self,
                 tt: Union['TT', Sequence[torch.Tensor], 'MPS'],
                 *,
                 out_device: _Device = 'cpu') -> None:
        self._tt = _as_tt(tt)
        self._runtime = _RuntimePolicy.from_tensor(
            self._tt.cores[0], out_device=out_device)

    @property
    def tt(self) -> TT:
        """Open-boundary TT fixed for repeated conversions."""
        return self._tt

    def fit(self,
            rank: int,
            tr_rank: Optional[int] = None,
            center: Optional[int] = None,
            loop_opener: Union[str, LoopOpener] = 'als',
            schedule: str = 'center_out',
            schedule_block_size: int = 1,
            recursion: Union[str, GaugeRecursion] = 'adjoint',
            allow_projective_gauges: bool = True,
            gauge_tolerance: float = 1e-08,
            inverse_policy: str = 'pinv',
            rank_rtol: Optional[float] = None,
            verbose: Union[bool, int] = 0,
            *,
            collect_metrics: bool = False,
            expand_directions: bool = True,
            max_extra_rank: Optional[int] = None,
            residual_rtol: float = 1e-12,
            block_selector: Optional[CentralBlockSelector] = None
        ) -> TRDecomposition:
        """
        Converts the TT with prescribed non-cyclic and cyclic TR ranks.

        ``rank`` is used for every non-cyclic right link. ``tr_rank`` sets the
        final cyclic link and defaults to ``rank``. These are prescribed fixed
        ranks: they are never silently reduced. A future adaptive mode will be
        explicit and separate.

        The ``center`` is opened without fixed gauges. Two independent sweeps
        then propagate the selected virtual-basis maps toward the TT boundaries,
        whose unit ranks are absorbed into the first and last TR cores. Fidelity,
        normalized overlap and absolute/relative reconstruction error are
        always measured by scaled TT/TR contractions without densifying.

        Adjoint transport and orthogonal expansion are enabled by default.
        These strategies remain experimental; a small local residual does not
        certify a small final TR reconstruction error.

        Parameters
        ----------
        rank : int
            Positive ``rank`` prescribed on every non-cyclic TR link.
        tr_rank : int, optional
            Positive cyclic ``rank``. Defaults to ``rank``.
        center : int, optional
            Internal TT site opened first. Defaults to the middle site.
        block_selector : CentralBlockSelector, optional
            Initial internal block policy. By default opens one site. Passing
            :class:`CentralBlockSelector` grows an initial block under the
            prescribed ranks; subsequent recursive openings remain one-site.
            Internal Vidal factors appear exactly once in a block target.
        loop_opener : {``"als"``, ``"blostr+als"``}, LoopOpener or callable
            Local loop-opening strategy. The simple preset uses exact TR-ALS;
            advanced ALS options should be encapsulated in an
            :class:`~tensorkrowch.decompositions.ALSLoopOpener`.
            :class:`~tensorkrowch.decompositions.SVDLoopOpener` optionally balances each free external
            axis and refines after restoring its singular values.
            ``"blostr+als"`` tries an experimental spectral initialization and
            falls back cleanly to the same ALS path if BLOSTR assumptions are
            not satisfied.
        schedule : {``"center_out"``, ``"alternating"``}
            Serial ring-construction ``schedule``. ``"alternating"`` is an
            experimental anchor/fixed-block ``schedule`` and falls back explicitly
            to ``"center_out"`` when the site layout or propagated gauges are
            incompatible.
        schedule_block_size : int
            Consecutive sites per alternating block. Open-boundary TT targets
            currently support the exact checkerboard with size one; other
            layouts use the documented fallback.
        recursion : str or GaugeRecursion
            Strategy for propagating virtual coordinates. ``"adjoint"`` uses
            QR and moves the center for ordinary TT inputs; valid Vidal inputs
            use their stored Schmidt powers and directional adjoints directly.
            ``"pinv"`` pseudoinverts the current TT core or contracted initial
            block. ``"gram"`` includes cached complete-subchain Gram metrics.
            ``"orth"`` uses the Moore-Penrose dual of the learned outgoing
            gauge. Strategy objects implement :class:`GaugeRecursion`.
            Adjoint and Gram require ``"center_out"``. Local ALS metrics are
            unchanged by Gram transport; Vidal targets retain their input gauge.
        allow_projective_gauges : bool
            Accept projected transport when a basis cannot be cancelled
            exactly. Default True supports approximate fixed-rank inputs.
            False enforces ``gauge_tolerance`` for inverse-based transport.
        gauge_tolerance : float
            Maximum relative error accepted for gauge cancellation.
        inverse_policy : {``"auto"``, ``"solve"``, ``"inverse"``, ``"pinv"``}
            Linear algebra used to construct directional gauge duals.
        rank_rtol : float, optional
            Relative singular-value threshold for pseudoinverses and numerical
            gauge ranks. Expansion uses the separate ``residual_rtol``.
        verbose : bool or int
            Console verbosity from 0 (silent) to 3 (final cores included).
        expand_directions : bool
            Append orthogonal opened-factor residual coordinates and zero
            target rows to the next local problem. Default True retains these
            compatibility constraints for all four built-in recursions.
            Requires ``"center_out"``; prescribed TR ranks and the input
            tensor are preserved. False disables expansion.
        max_extra_rank : int, optional
            Maximum auxiliary coordinates added per transition. None retains
            all numerically significant directions; zero is a matched control.
        residual_rtol : float
            Auxiliary residual singular-value cutoff relative to the weighted
            opened-factor Frobenius norm. Default 1e-12. Reference support
            uses only a machine-precision cutoff, not this truncation.
        collect_metrics : bool
            Whether to retain local ALS solve diagnostics. Fidelity, final
            reconstruction error and gauge-cancellation checks remain enabled.
            Verbosity also enables local metrics.

        Returns
        -------
        TRDecomposition
            Lightweight cyclic decomposition with structured local, gauge,
            fidelity, error and timing metrics.

        Examples
        --------
        Convert a ``rank``-one TT while preserving its represented tensor:

        >>> tt = [torch.tensor([[1.], [2.]]),
        ...       torch.tensor([[[1.], [3.]]]),
        ...       torch.tensor([[2., 1.]])]
        >>> result = tk.decompositions.TT2TR(tt).fit(rank=1)
        >>> result.rank
        [1, 1, 1]
        >>> bool(result.metrics.fidelities[0].fidelity > 0.999)
        True
        """
        if not isinstance(expand_directions, bool):
            raise TypeError('`expand_directions` should be bool type')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        if not isinstance(allow_projective_gauges, bool):
            raise TypeError('`allow_projective_gauges` should be bool type')
        if schedule not in ('center_out', 'alternating'):
            raise ValueError(
                "`schedule` should be 'center_out' or 'alternating'")
        if isinstance(schedule_block_size, bool) or \
                not isinstance(schedule_block_size, int):
            raise TypeError('`schedule_block_size` should be int type')
        if schedule_block_size < 1:
            raise ValueError('`schedule_block_size` should be positive')
        if block_selector is not None and not isinstance(
                block_selector, CentralBlockSelector):
            raise TypeError('`block_selector` should be CentralBlockSelector type')
        if block_selector is not None and schedule != 'center_out':
            raise ValueError(
                '`block_selector` requires the `center_out` schedule')
        rank_spec = _normalize_rank(rank, tr_rank, len(self._tt.cores))
        if center is None:
            center = len(self._tt.cores) // 2
        verbosity = _normalize_verbosity(verbose)
        fit_observer = _resolve_observer(verbosity, None) \
            if verbosity else None
        opener = resolve_loop_opener(loop_opener)
        requested_recursion = recursion
        recursion = _resolve_recursion(
            recursion, inverse_policy=inverse_policy,
            allow_projective=allow_projective_gauges,
            tolerance=gauge_tolerance, rank_rtol=rank_rtol)
        if isinstance(recursion, (GramGaugeRecursion, AdjointGaugeRecursion)):
            if schedule != 'center_out' or schedule_block_size != 1:
                raise ValueError(
                    'Adjoint and Gram recursion require `center_out` '
                    'with `schedule_block_size=1`')
        if isinstance(recursion, AdjointGaugeRecursion):
            bonds = self._tt.bonds
            if isinstance(bonds, VidalGauge) and bonds._valid:
                provider = _VidalCoreProvider(self._tt)
            else:
                provider = _CanonicalTTCoreProvider(self._tt, center)
        elif isinstance(recursion, GramGaugeRecursion):
            provider = _GramTTCoreProvider(self._tt)
        else:
            provider = _TTCoreProvider(self._tt)

        if expand_directions:
            if schedule != 'center_out' or schedule_block_size != 1:
                raise ValueError('`expand_directions` requires `center_out`')
            if not isinstance(recursion, (OrthGaugeRecursion,
                                          PinvGaugeRecursion)):
                raise TypeError('Expansion requires a built-in recursion')
            recursion = _ExpandedGaugeRecursion(
                recursion, max_extra_rank, residual_rtol)

        if fit_observer is not None:
            fit_observer.emit(DecompositionEvent(
                name='start',
                phase='TT to TR',
                values={
                    'sites': len(self._tt.cores),
                    'in_dim': self._tt.in_dim,
                    'rank': rank_spec,
                    'center': center,
                }))

        with self._runtime.timer() as timer:
            driver_options = {
                'provider': provider,
                'rank': rank_spec,
                'opener': opener,
                'recursion': recursion,
                'block_selector': (PrescribedCentralBlockSelector()
                                   if block_selector is None else block_selector),
                'center': center,
                'context': {
                    '_observer': fit_observer,
                    'collect_metrics': collect_metrics or bool(verbosity),
                    'block_bounds': (1, len(provider.in_dim) - 2),
                },
            }
            if schedule == 'alternating':
                driver_result = AlternatingRingDriver().fit(
                    **driver_options,
                    fixed_opener=FixedGaugeCoreOpener(),
                    block_size=schedule_block_size)
            else:
                driver_result = BidirectionalRingDriver().fit(
                    **driver_options)
            active_result = driver_result.as_decomposition()
            fidelity = _fidelity_error(self._tt, active_result)
            active_result.metrics.fidelities.append(fidelity)
            active_result.metrics.errors.insert(0, fidelity.error)

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
            'center_block': driver_result.central_block.sites,
            'schedule': driver_result.diagnostics.get(
                'schedule', 'center_out'),
            'requested_schedule': schedule,
            'schedule_block_size': schedule_block_size,
            'requested_rank': list(rank_spec),
            'adaptive': False,
            'inverse_policy': inverse_policy,
            'recursion': (
                requested_recursion if isinstance(requested_recursion, str)
                else type(requested_recursion).__name__),
            'allow_projective_gauges': allow_projective_gauges,
            'boundary_closure': 'learned_gauge_absorption',
            'expand_directions': expand_directions,
            **({'vidal_powers': provider.powers}
               if isinstance(provider, _VidalCoreProvider) else {}),
            **({'max_extra_rank': recursion.max_extra_rank,
                'residual_rtol': recursion.residual_rtol,
                'enrichment': driver_result.diagnostics.get('recursions', [])}
               if isinstance(recursion, _ExpandedGaugeRecursion) else {}),
        })
        result = TRDecomposition(
            cores=[self._runtime.finalize(core)
                   for core in active_result.cores],
            metrics=active_result.metrics,
            metadata=metadata)

        if fit_observer is not None:
            fit_observer.emit(DecompositionEvent(
                name='summary',
                phase='TT to TR',
                values={
                    'rank': result.rank,
                    'fidelity': fidelity.fidelity,
                    'absolute_error': fidelity.error.absolute,
                    'relative_error': fidelity.error.relative,
                    'elapsed': timer.elapsed,
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


def tt2tr(tt: Union['TT', Sequence[torch.Tensor], 'MPS'],
          rank: int,
          tr_rank: Optional[int] = None,
          center: Optional[int] = None,
          loop_opener: Union[str, LoopOpener] = 'als',
          schedule: str = 'center_out',
          schedule_block_size: int = 1,
          recursion: Union[str, GaugeRecursion] = 'adjoint',
          allow_projective_gauges: bool = True,
          gauge_tolerance: float = 1e-08,
          inverse_policy: str = 'pinv',
          rank_rtol: Optional[float] = None,
          out_device: _Device = 'cpu',
          verbose: Union[bool, int] = 0,
          return_info: bool = False,
          *,
          block_selector: Optional[CentralBlockSelector] = None,
          expand_directions: bool = True,
          max_extra_rank: Optional[int] = None,
          residual_rtol: float = 1e-12) -> '_DecompositionOutput':
    """
    Converts open-boundary TT cores into prescribed tensor ring cores.

    This is the simple functional interface. Use :class:`TT2TR` for repeated
    conversions of the same TT or direct access to the lightweight result.
    ``rank`` controls all non-cyclic links and ``tr_rank`` controls the cyclic
    closing link. The local ALS configuration remains encapsulated by
    ``loop_opener`` rather than expanding this function's signature.

    Parameters
    ----------
    tt : TT, sequence of torch.Tensor or MPS
        Open-boundary TT to convert. At least three sites are required.
        Valid Vidal metadata are retained for adjoint transport.
    rank : int
        Positive ``rank`` prescribed on every non-cyclic TR link.
    tr_rank : int, optional
        Positive cyclic ``rank``. Defaults to ``rank``.
    center : int, optional
        Internal TT site opened first. Defaults to the middle site.
    block_selector : CentralBlockSelector, optional
        Initial block policy for ``"center_out"``. Defaults to one site;
        :class:`CentralBlockSelector` grows a block under the prescribed ranks.
        Subsequent recursive openings remain one-site.
    loop_opener : {``"als"``, ``"blostr+als"``}, LoopOpener or callable
        Local loop-opening strategy. The simple preset uses exact TR-ALS;
        advanced ALS options should be encapsulated in an
        :class:`~tensorkrowch.decompositions.ALSLoopOpener`.
        :class:`~tensorkrowch.decompositions.SVDLoopOpener` adds local SVD balancing
        and optional refinement.
        ``"blostr+als"``
        tries an experimental spectral initialization and falls back cleanly to
        the same ALS path if BLOSTR assumptions are not satisfied.
    schedule : {``"center_out"``, ``"alternating"``}
        Serial ring-construction ``schedule``. ``"alternating"`` is an experimental
        anchor/fixed-block ``schedule`` and falls back explicitly to
        ``"center_out"`` when the site layout or propagated gauges are
        incompatible.
    schedule_block_size : int
        Consecutive sites per alternating block. Open-boundary TT targets
        currently support the exact checkerboard with size one; other layouts
        use the documented fallback.
    recursion : str or GaugeRecursion
        Strategy for propagating virtual coordinates. ``"adjoint"`` uses
        QR and moves the center for ordinary TT inputs; valid Vidal inputs
        use their stored Schmidt powers and directional adjoints directly.
        ``"pinv"`` pseudoinverts the current TT core or contracted initial
        block. ``"gram"`` includes cached complete-subchain Gram metrics.
        ``"orth"`` uses the Moore-Penrose dual of the learned outgoing
        gauge. Strategy objects implement :class:`GaugeRecursion`.
        Adjoint and Gram require ``"center_out"``. Local ALS metrics are
        unchanged by Gram transport; Vidal targets retain their input gauge.
    allow_projective_gauges : bool
        Accept projected transport when a basis cannot be cancelled
        exactly. Default True supports approximate fixed-rank inputs.
        False enforces ``gauge_tolerance`` for inverse-based transport.
    gauge_tolerance : float
        Maximum relative error accepted for gauge cancellation.
    inverse_policy : {``"auto"``, ``"solve"``, ``"inverse"``, ``"pinv"``}
        Linear algebra used to construct directional gauge duals.
    rank_rtol : float, optional
        Relative singular-value threshold for pseudoinverses and numerical
        gauge ranks.
    out_device : str or torch.device, optional
        Device where finalized TR cores are stored. The default is ``"cpu"``;
        ``None`` keeps them on the input device.
    verbose : bool or int
        Console verbosity from 0 (silent) to 3 (final cores included).
    expand_directions : bool
        Append orthogonal opened-factor residual coordinates and zero
        target rows to the next local problem. Default True retains these
        compatibility constraints for all four built-in recursions.
        Requires ``"center_out"``; prescribed TR ranks and the input
        tensor are preserved. False disables expansion.
    max_extra_rank : int, optional
        Maximum auxiliary coordinates added per transition. None retains
        all numerically significant directions; zero is a matched control.
    residual_rtol : float
        Auxiliary residual singular-value cutoff relative to the weighted
        opened-factor Frobenius norm. Default 1e-12. Reference support
        uses only a machine-precision cutoff, not this truncation.
    return_info : bool
        If ``True``, also returns metadata and structured ALS metrics.

    Returns
    -------
    list[torch.Tensor] or tuple
        Cores in original site order, or ``(cores, info)`` with
        ``return_info=True``.

    Examples
    --------
    >>> tt = [torch.tensor([[1.], [2.]]),
    ...       torch.tensor([[[1.], [3.]]]),
    ...       torch.tensor([[2., 1.]])]
    >>> cores = tk.decompositions.tt2tr(tt, rank=1)
    >>> [tuple(core.shape) for core in cores]
    [(1, 2, 1), (1, 2, 1), (1, 2, 1)]
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    result = TT2TR(tt, out_device=out_device).fit(
        rank=rank,
        tr_rank=tr_rank,
        center=center,
        loop_opener=loop_opener,
        schedule=schedule,
        schedule_block_size=schedule_block_size,
        recursion=recursion,
        allow_projective_gauges=allow_projective_gauges,
        gauge_tolerance=gauge_tolerance,
        inverse_policy=inverse_policy,
        rank_rtol=rank_rtol,
        verbose=verbose,
        block_selector=block_selector,
        collect_metrics=return_info,
        expand_directions=expand_directions,
        max_extra_rank=max_extra_rank,
        residual_rtol=residual_rtol)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


__all__ = [
    'TT2TR',
    'tt2tr',
]
