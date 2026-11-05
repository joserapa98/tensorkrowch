"""
This script contains:

    Classes:
        * _TTCoreProvider
        * _CanonicalTTCoreProvider
        * _VidalCoreProvider
        * TT2TR

    Functions:
        * _as_tt
        * _schmidt_power
        * _weight_rows
        * _resolve_recursion
        * _fidelity_error
        * tt2tr
"""

from dataclasses import dataclass, replace
from math import sqrt
from typing import (Any,
                    Mapping,
                    Optional,
                    Sequence,
                    Tuple,
                    Union)
from warnings import warn

import torch

from tensorkrowch.formats import TT
from tensorkrowch.formats.bonds import VidalGauge
from tensorkrowch.formats.formats1d import _restore_cores
from tensorkrowch.models.mps import MPS

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions._utils import _normalize_rank
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord,
                                                 FidelityRecord,
                                                 TimingRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import (TRDecomposition,
                                                 TTDecomposition)
from tensorkrowch.decompositions.ring.blocks import (_ring_partition,
                                                     _spectral_block,
                                                     _split_block)
from tensorkrowch.decompositions.ring.driver import (BidirectionalRingDriver,
                                                     )
from tensorkrowch.decompositions.ring.gauges import (AdjointGaugeRecursion,
                                                     GaugeRecursion,
                                                     GaugeMap,
                                                     OrthGaugeRecursion,
                                                     PinvGaugeRecursion,
                                                     _ExpandedGaugeRecursion)
from tensorkrowch.decompositions.ring.opening import (BLOSTRLoopOpener,
                                                      LoopOpener,
                                                      resolve_loop_opener)
from tensorkrowch.decompositions.sources.format import FormatSource


_Device = Optional[Union[str, torch.device]]


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
    return result


@dataclass
class _TTCoreProvider:  # MARK: _TTCoreProvider
    """Exposes TT supercores and open-edge absorptions to the ring driver."""

    tt: TT  # Original open-boundary format supplying local targets

    boundary_mode = 'open'

    def __post_init__(self) -> None:
        source = FormatSource(
            self.tt.to_tt() if self.tt._quantized else self.tt)
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
        if not sites or (sites != tuple(range(sites[0], sites[-1] + 1))):
            raise ValueError(
                '`sites` should form a contiguous TT interval')
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

    def configure_partition(self, sites: Sequence[int],
                            left: Sequence[int], right: Sequence[int]) -> None:
        """Retains the common construction intervals for local targets."""
        self.central_sites = tuple(sites)
        self.boundary_sites = (*left, *right)

    def partition(self, rank: Any, center: int, block_boundaries: bool,
                  rtol: Optional[float]) -> Tuple[Any, ...]:
        """Measures the shared TT2TR/TR-RSS rank and block partition."""
        return _ring_partition(self, rank, center, block_boundaries, rtol)

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
                       sites: Sequence[int],
                       direction: str,
                       opening: Any,
                       context: Mapping[str, Any]) -> Tuple[torch.Tensor, ...]:
        """Absorbs a learned gauge and splits a boundary block sitewise."""
        target = self.local_target(sites, context)
        if direction == 'left':
            gauge = context.get('boundary_gauge', opening.left_gauge)
            block = torch.tensordot(target.squeeze(0), gauge, dims=([-1], [1]))
            block = block.movedim(-2, 0)
        elif direction == 'right':
            gauge = context.get('boundary_gauge', opening.right_gauge)
            block = torch.tensordot(gauge, target.squeeze(-1), dims=([1], [0]))
            block = block.movedim(1, -1)
        else:
            raise ValueError("`direction` should be 'left' or 'right'")
        dimensions = tuple(self.in_dim[site] for site in sites)

        # Apply bounds only to cuts inside the absorbed boundary block.
        profile = context.get('rank_caps')
        rank = (None if profile is None else
                tuple(profile[site] for site in sites[:-1]))
        metrics = DecompositionMetrics() if context.get(
            'collect_metrics', False) else None
        cores = _split_block(
            block, dimensions, rank=rank, rtol=context.get('rtol'),
            metrics=metrics)

        # Local SVD indices must be translated to positions in the whole chain.
        records = (() if metrics is None else tuple(
            replace(record, site=sites[0] + record.site)
            for record in metrics.truncations))
        if metrics is not None:
            context['_metrics'].truncations.extend(records)
        return cores

    def finalize_center(self, opening: Any, sites: Sequence[int]) -> Any:
        """Absorbs dummy external gauges when discovery blocks overlap."""
        # After absorption, identity gauges prevent transport from applying the
        # same boundary map a second time.
        cores = list(opening.cores)
        left, right = opening.left_gauge, opening.right_gauge
        if sites[0] == 0:
            cores[0] = torch.einsum(
                'ga,aib->gib', opening.left_gauge[:, 0, :], cores[0])
            left = torch.eye(left.shape[0], dtype=left.dtype,
                             device=left.device).unsqueeze(1)
        if sites[-1] == len(self.in_dim) - 1:
            cores[-1] = torch.einsum(
                'aib,bg->aig', cores[-1], opening.right_gauge[:, 0, :])
            right = torch.eye(right.shape[0], dtype=right.dtype,
                              device=right.device).unsqueeze(1)
        return replace(opening, cores=cores, left_gauge=left,
                       right_gauge=right)

    def prepare_boundary(self,
                         opening: Any,
                         local_target: Any,
                         recursion_context: Mapping[str, Any],
                         direction: str) -> GaugeMap:
        """Keeps the learned outgoing gauge for direct TT-edge absorption."""
        if direction == 'right':
            gauge = opening.right_gauge
        elif direction == 'left':
            gauge = opening.left_gauge
        else:
            raise ValueError("`direction` should be 'left' or 'right'")
        return GaugeMap(
            core=gauge, orientation=direction,
            metadata={'boundary': True,
                         'algorithm': 'absorb_learned_gauge'})


class _CanonicalTTCoreProvider(_TTCoreProvider):  # MARK: CanonicalProvider
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


def _schmidt_power(spectrum: torch.Tensor, power: float) -> torch.Tensor:
    """Applies a spectral power on positive support, zeroing inverse zeros."""
    if power == 0:
        return torch.ones_like(spectrum)
    positive = spectrum > 0
    safe = torch.where(positive, spectrum, torch.ones_like(spectrum))
    result = torch.where(positive, safe.pow(power), torch.zeros_like(spectrum))
    if not torch.isfinite(result).all():
        raise ValueError('Schmidt powers overflow on the retained support')
    return result


def _weight_rows(matrix: torch.Tensor,
                 basis: torch.Tensor,
                 factor: torch.Tensor,
                 direction: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weights the incoming virtual rows by a Schmidt factor."""
    physical_dim = matrix.shape[0] // factor.shape[1]
    if direction == 'right':
        matrix = torch.einsum('xa,apb->xpb', factor, matrix.reshape(
            factor.shape[1], physical_dim, -1))
        basis = torch.einsum('xa,apy->xpy', factor, basis.reshape(
            factor.shape[1], physical_dim, -1))
    else:
        matrix = torch.einsum('xb,pba->pxa', factor, matrix.reshape(
            physical_dim, factor.shape[1], -1))
        basis = torch.einsum('xb,pby->pxy', factor, basis.reshape(
            physical_dim, factor.shape[1], -1))
    return (matrix.reshape(-1, matrix.shape[-1]),
            basis.reshape(-1, basis.shape[-1]))


class _VidalCoreProvider(_TTCoreProvider):  # MARK: _VidalCoreProvider
    """Supplies canonical Vidal cores and their remaining bond factors."""

    def __post_init__(self) -> None:
        tt = self.tt.to_tt() if self.tt._quantized else self.tt
        bonds = tt.bonds
        if not isinstance(bonds, VidalGauge) or not bonds._valid:
            raise ValueError(
                'Adjoint transport requires valid Vidal data')
        self.in_dim = tuple(tt.in_dim)
        self.cores = list(tt._standard_cores())
        self.spectra = tuple(bonds.spectra)
        self.powers = tuple(bonds.powers)
        self.bond_factors = tuple(_schmidt_power(s, 1 - left - right)
                                  for s, (left, right) in
                                  zip(self.spectra, self.powers))
        # Auxiliary expansion coordinates carry unit metric, not Schmidt data.
        unit = self.cores[0].new_ones(1, 1)
        self.left_factors = (unit, *(torch.diag(
            _schmidt_power(s, 1 - right)).to(
            self.cores[0].dtype) for s, (_, right) in
            zip(self.spectra, self.powers)), unit)
        self.right_factors = (unit, *(torch.diag(
            _schmidt_power(s, 1 - left)).to(
            self.cores[0].dtype) for s, (left, _) in
            zip(self.spectra, self.powers)), unit)

    def expand_coordinates(self, site: int, count: int,
                           direction: str) -> None:
        """Adds zero target rows with unit metric in auxiliary coordinates."""
        super().expand_coordinates(site, count, direction)
        if not count:
            return
        name = 'left_factors' if direction == 'right' else 'right_factors'
        factors = list(getattr(self, name))
        index = site if direction == 'right' else site + 1
        core = self.cores[site]
        unit = torch.eye(count, dtype=core.dtype, device=core.device)
        factors[index] = torch.block_diag(factors[index], unit)
        setattr(self, name, factors)

    def _transport_matrices(self, matrix: torch.Tensor,
                            basis: torch.Tensor, sites: Sequence[int],
                            direction: str
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Weights incoming rows and removes outgoing absorption powers."""
        first, last = sites[0], sites[-1]
        if direction == 'right':
            factor = self.left_factors[first]
            spectrum = self.spectra[last]
            power = -self.powers[last][0]
        else:
            factor = self.right_factors[last + 1]
            spectrum = self.spectra[first - 1]
            power = -self.powers[first - 1][1]
        matrix, basis = _weight_rows(matrix, basis, factor, direction)
        return matrix * _schmidt_power(spectrum, power).unsqueeze(0), basis

    def _transport_coordinates(self, coordinates: torch.Tensor,
                               sites: Sequence[int], direction: str
                               ) -> torch.Tensor:
        """Converts isometric coordinates to the next target's powers."""
        bond = sites[-1] if direction == 'right' else sites[0] - 1
        power = self.powers[bond][1 if direction == 'right' else 0] - 1
        return coordinates * _schmidt_power(
            self.spectra[bond], power).unsqueeze(1)

    def local_target(self,
                     sites: Sequence[int],
                     context: Mapping[str, Any]) -> torch.Tensor:
        """Contracts Vidal cores with each remaining internal factor once."""
        sites = tuple(sites)
        if not sites or (sites != tuple(range(sites[0], sites[-1] + 1))):
            raise ValueError(
                '`sites` should form a contiguous TT interval')
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
                         direction: str) -> GaugeMap:
        """Includes the remaining Vidal factor in the outgoing gauge."""
        step = super().prepare_boundary(
            opening, local_target, recursion_context, direction)
        sites = recursion_context['from_sites']
        bond = sites[-1] if direction == 'right' else sites[0] - 1
        gauge = step.core * self.bond_factors[bond][None, :, None]
        return GaugeMap(core=gauge, orientation=direction,
                        metadata=step.metadata)


def _resolve_recursion(recursion: Any) -> GaugeRecursion:
    """Resolves a built-in transport or a configured strategy object."""
    if isinstance(recursion, str):
        strategies = {'adjoint': AdjointGaugeRecursion,
                      'pinv': PinvGaugeRecursion,
                      'orth': OrthGaugeRecursion}
        if recursion not in strategies:
            raise ValueError(
                "`recursion` should be 'adjoint', 'pinv' or 'orth'")
        return strategies[recursion]()
    if isinstance(recursion, GaugeRecursion):
        return recursion
    raise TypeError('`recursion` should implement GaugeRecursion')


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
    Converts an open tensor train into a tensor ring.

    Optimization openers use center-out gauge transport with optional residual
    expansion. BLOSTR openers instead recover an exact central opening and
    absorb its gauges outward, splitting with SVDs. Input cores remain
    unchanged. Spectral factors and the working TT cores use a common complex
    dtype, so BLOSTR conversion returns complex cores even for real inputs.

    Parameters
    ----------
    tt : TT, MPS or sequence of torch.Tensor
        Open tensor train, open-boundary MPS or cores in standard TT order.
        Structural batches are not supported.
    out_device : str or torch.device, optional
        Device for the returned cores. Default is ``"cpu"``; ``None`` keeps
        the input device.
    """

    def __init__(self, tt: Any, out_device: _Device = 'cpu') -> None:
        self._tt = _as_tt(tt)
        self._runtime = _RuntimePolicy.from_tensor(
            self._tt.cores[0], out_device=out_device)

    @property
    def tt(self) -> TT:
        """Input tensor train used for repeated conversions."""
        return self._tt

    def fit(self,
            rank: Optional[Union[int, Sequence[int]]] = None,
            center: Optional[int] = None,
            loop_opener: Union[str, LoopOpener] = 'als',
            recursion: Union[str, GaugeRecursion] = 'adjoint',
            *,
            block_boundaries: bool = True,
            enrich: bool = True,
            enrich_rank: Optional[int] = None,
            enrich_rtol: float = 1e-12,
            rtol: Optional[float] = None,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0) -> TRDecomposition:
        """
        Fits right-link ranks and optionally truncates SVD splits adaptively.

        A prescribed ``rank`` starts at ``center`` and grows an initial block
        when one site has insufficient physical capacity, then proceeds
        sitewise. By default, boundary sites are grouped into blocks, whose
        gauges are absorbed before SVD splitting. Overlapping blocks merge;
        the whole chain may be opened at once. With known ranks, boundary
        blocking can be disabled independently of the initial block.
        Discovery first fits one physical supercore for the central block
        and splits it with SVDs to determine its internal ranks. Prescribed
        blocks instead optimize their physical cores individually. Discovery
        is a numerical rank heuristic, not unique rank recovery.

        BLOSTR instead selects the shortest interior block containing
        ``center`` with equal external TT bond dimensions. It uses no
        physical-capacity criterion; absent a matching pair, it opens the
        complete chain. The selected opener determines spectral feasibility.
        Full BLOSTR still requires uniform ranks and injective physical
        unfoldings; any physical blocking is left to the caller. A multi-site
        spectral block with slice projections warns to use Gaussian ones.

        Parameters
        ----------
        rank : int or sequence of int, optional
            Local ALS rank, or one right-link rank per core including the
            cyclic last entry. At SVD splits these are upper bounds: ``rtol``
            may select smaller ranks, which are returned without zero padding.
            ``None`` discovers ranks using SVDs or spectral multiplicities.
        center : int, optional
            Initial site, or the center of the discovery block. Defaults to
            the middle site.
        loop_opener : str or LoopOpener, optional
            ``"als"`` optimizes local openings. ``"blostr"`` uses complete
            spectral recovery; ``"blostr_svd"`` recovers the first factor
            spectrally and splits the remainder with SVDs. Configured opener
            objects expose solver, convergence and projection options.
        recursion : str or GaugeRecursion, optional
            ``"adjoint"`` moves QR centers or uses stored Vidal isometries;
            ``"pinv"`` uses TT-core pseudoinverses; ``"orth"`` dualizes the
            learned gauge. Configured objects accept PyTorch pseudoinverse
            keyword options. Exact BLOSTR conversion does not use transport.
        block_boundaries : bool
            Group boundary sites into automatic blocks before gauge absorption
            and SVD splitting. Default is True. False keeps one-site boundaries
            and requires supplied ``rank`` for optimization openers. The
            initial block still grows automatically. Exact spectral conversion
            does not use this option.
        enrich : bool
            Append orthogonal residual constraints without changing TR ranks.
            Default is True. Exact BLOSTR conversion does not use expansion.
        enrich_rank : int, optional
            Maximum orthogonal coordinates added per transition. ``None``
            retains all significant directions; zero adds none.
        enrich_rtol : float
            Residual singular-value cutoff relative to the opened factor's
            Frobenius norm. Default is 1e-12.
        rtol : float, optional
            Relative discarded squared singular-value mass for rank discovery
            and block SVD splits. Default ``None`` truncates only to ``rank``
            when ranks are supplied. A tolerance can reduce SVD ranks below
            these bounds. With no ranks, omitted ``rtol`` uses 1e-12 for
            discovery, bounded below by machine precision. Spectral conversion
            also uses the tolerance for
            reconstruction checks. It does not define ALS convergence.
        collect_metrics : bool
            Retain local solver and decomposition diagnostics. Final fidelity
            and reconstruction error are always measured.
        verbose : bool or int
            Console verbosity from 0 (silent) to 3 (final cores included).

        Returns
        -------
        TRDecomposition
            Cores in site order, metrics and construction metadata.

        Warns
        -----
        UserWarning
            Multi-site spectral blocks with slices recommend Gaussian
            projection. Full BLOSTR also warns before rejecting unequal
            local TR ranks or insufficient physical dimensions, recommending
            BLOSTR-SVD or blocking the input TT, respectively.

        Examples
        --------
        >>> tt = tk.formats.TT([torch.ones(2, 1), torch.ones(1, 2)])
        >>> result = tk.decompositions.TT2TR(tt).fit(rank=1)
        >>> result.rank
        [1, 1]
        """
        # Validate controls before preparing reference coordinates or local
        # fits.
        for name, value in (('block_boundaries', block_boundaries),
                            ('enrich', enrich),
                            ('collect_metrics', collect_metrics)):
            if not isinstance(value, bool):
                raise TypeError(f'`{name}` should be bool type')
        count = len(self._tt.cores)
        if center is None:
            center = count // 2
        if isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int type')
        if not 0 <= center < count:
            raise ValueError('`center` should identify an input site')
        if rtol is not None:
            if isinstance(rtol, bool) or not isinstance(rtol, (int, float)):
                raise TypeError('`rtol` should be a number in [0, 1] or None')
            if not 0 <= rtol <= 1:
                raise ValueError('`rtol` should lie in [0, 1]')
        if self._tt.norm() <= 0:
            raise ValueError('`tt` should represent a nonzero tensor')
        verbosity = _normalize_verbosity(verbose)
        observer = _resolve_observer(verbosity, None) if verbosity else None
        opener = resolve_loop_opener(loop_opener)
        if (not block_boundaries) and (rank is None) and not isinstance(
                opener, BLOSTRLoopOpener):
            raise ValueError(
                '`block_boundaries=False` requires supplied `rank` '
                'for optimization openers')

        # Choose the reference gauge required by the selected transport.
        strategy = _resolve_recursion(recursion)
        if isinstance(opener, BLOSTRLoopOpener):
            provider = _TTCoreProvider(self._tt)
        elif isinstance(strategy, AdjointGaugeRecursion):
            bonds = self._tt.bonds
            provider = (_VidalCoreProvider(self._tt)
                        if isinstance(bonds, VidalGauge) and bonds._valid
                        else _CanonicalTTCoreProvider(self._tt, center))
        else:
            provider = _TTCoreProvider(self._tt)

        ranks = None if rank is None else _normalize_rank(rank, count)
        if (rank is None) and isinstance(opener, BLOSTRLoopOpener):
            precision = torch.finfo(provider.cores[0].real.dtype).eps
            size = max(core.numel() for core in provider.cores)
            rtol = max(1e-12 if rtol is None else rtol,
                       (size * precision) ** 2)
        if observer is not None:
            observer.emit(DecompositionEvent(
                name='start', phase='TT to TR',
                values={'sites': count, 'in_dim': provider.in_dim,
                        'rank': None if ranks is None else list(ranks)}))
        context = {'collect_metrics': collect_metrics or bool(verbosity),
                   '_observer': observer, 'rtol': rtol,
                   'rank_caps': None if rank is None else ranks}

        # Construct local factors first, then check the complete
        # reconstruction.
        with self._runtime.timer() as timer:
            if isinstance(opener, BLOSTRLoopOpener):
                prescribed = ranks if rank is not None else None
                result = self._exact(
                    provider, opener, prescribed, center, rtol, context)
            else:
                if enrich:
                    strategy = _ExpandedGaugeRecursion(
                        strategy, enrich_rank, enrich_rtol)
                result = BidirectionalRingDriver().fit(
                    provider, ranks, opener, strategy, center=center,
                    block_boundaries=block_boundaries, rtol=rtol,
                    context=context)
            fidelity = _fidelity_error(self._tt, result)
            if isinstance(opener, BLOSTRLoopOpener):
                # Contraction-based errors have a square-root precision floor.
                tolerance = max(sqrt(rtol or 0.0), 10 * sqrt(torch.finfo(
                    result.cores[0].real.dtype).eps))
                if fidelity.error.relative > tolerance:
                    raise RuntimeError(
                        'Exact BLOSTR conversion failed its reconstruction '
                        'check; use a Gaussian projection or an ALS opener')
            if (rank is not None) and any(
                    actual > cap for actual, cap in zip(result.rank, ranks)):
                raise RuntimeError('TT2TR exceeded the requested `rank`')
            result.metrics.fidelities.append(fidelity)
            result.metrics.errors.insert(0, fidelity.error)
        result.metrics.timings.append(TimingRecord('fit', timer.elapsed))

        # Finalize storage separately from the device used for local solves.
        metadata = {**result.metadata, 'algorithm': 'tt2tr', 'center': center,
                    'requested_rank': None if rank is None else list(ranks),
                    'recursion': (recursion if isinstance(recursion, str)
                                  else type(recursion).__name__),
                    'enrich': enrich, 'block_boundaries': block_boundaries}
        if isinstance(provider, _VidalCoreProvider):
            metadata['vidal_powers'] = provider.powers
        result = TRDecomposition(
            [self._runtime.finalize(core) for core in result.cores],
            metrics=result.metrics, metadata=metadata)
        if observer is not None:
            observer.emit(DecompositionEvent(
                name='summary', phase='TT to TR', values={
                    'rank': result.rank, 'fidelity': fidelity.fidelity,
                    'absolute_error': fidelity.error.absolute,
                    'relative_error': fidelity.error.relative,
                    'elapsed': timer.elapsed}))
            for site, core in enumerate(result.cores):
                observer.emit(DecompositionEvent(
                    name='core', phase='TT to TR', level=3, site=site,
                    values={'shape': tuple(core.shape), 'tensor': core}))
            observer.close(result.metrics)
        return result

    def _exact(self, provider: _TTCoreProvider, opener: BLOSTRLoopOpener,
               rank: Optional[Sequence[int]], center: int,
               rtol: Optional[float],
               context: Mapping[str, Any]) -> TRDecomposition:
        """Opens the center spectrally and splits the two outward branches."""
        count = len(provider.cores)
        sites = _spectral_block(
            tuple(core.shape[-1] for core in provider.cores[:-1]), center)
        if (len(sites) > 1) and (
                opener.fit_options.get('projection', 'slices') == 'slices'):
            warn("A multi-site spectral block uses projection='slices'; "
                 "use projection='gaussian' to avoid rank-deficient "
                 "coordinate projections.", UserWarning, stacklevel=2)
        left, right = sites[0], sites[-1]
        target = provider.local_target(sites, context)
        if (left == 0) and (right == count - 1):
            dense = target.squeeze(0).squeeze(-1)
            result = opener.decompose(dense, rank, context=context)
            return TRDecomposition(
                result.cores, metrics=result.metrics, metadata={
                    **result.metadata, 'exact': True, 'center_block': sites})
        local_rank = None if rank is None else (
            rank[left - 1], *(rank[site] for site in sites), rank[-1])
        opening = opener.open(target, local_rank, context={
            **context, 'sites': sites, 'in_dim': target.shape})
        for gauge in (opening.left_gauge, opening.right_gauge):
            matrix = gauge.movedim(1, 0).reshape(gauge.shape[1], -1)
            if (matrix.shape[0] != matrix.shape[1]) or (
                    torch.linalg.matrix_rank(matrix) < matrix.shape[0]):
                raise RuntimeError(
                    'Exact BLOSTR conversion requires square, full-rank '
                    'central gauges; use a larger block or a Gaussian '
                    'projection for nonuniform interior ranks')
        error = (opening.contract_dense() - target).norm() / target.norm()
        tolerance = max(sqrt(rtol or 0.0), 10 * sqrt(
            torch.finfo(target.real.dtype).eps))
        if error > tolerance:
            raise RuntimeError(
                'BLOSTR did not recover the central block to the requested '
                'accuracy; try Gaussian projection or an ALS opener')

        # Exact propagation absorbs learned gauges directly, without gauge
        # duals. Spectral factors can be complex even for a real input TT.
        dtype = torch.promote_types(target.dtype, opening.left_gauge.dtype)
        tt_cores = [core.to(dtype=dtype) for core in provider.cores]
        cores = [None] * count
        cores[left:right + 1] = [
            core.to(dtype=dtype) for core in opening.cores]
        gauge = opening.right_gauge.to(dtype=dtype)

        # Expose one rightward physical core and retain the remaining gauge.
        for site in range(right + 1, count - 1):
            block = torch.tensordot(
                gauge, tt_cores[site], dims=([1], [0]))
            block = block.permute(0, 2, 3, 1)
            cap = None if rank is None else (rank[site],)
            core, gauge = _split_block(
                block, (provider.in_dim[site], block.shape[2]), rank=cap,
                rtol=rtol)
            cores[site] = core
        cores[-1] = torch.einsum(
            'abg,bi->aig', gauge, tt_cores[-1].squeeze(-1))
        gauge = opening.left_gauge.to(dtype=dtype)

        # Mirror the split for the left branch while preserving site order.
        for site in range(left - 1, 0, -1):
            block = torch.tensordot(
                tt_cores[site], gauge, dims=([-1], [1]))
            block = block.permute(2, 0, 1, 3)
            cap = None if rank is None else (rank[site - 1],)
            gauge, core = _split_block(
                block, (block.shape[1], provider.in_dim[site]), rank=cap,
                rtol=rtol)
            cores[site] = core
        cores[0] = torch.einsum(
            'ia,gab->gib', tt_cores[0].squeeze(0), gauge)
        return TRDecomposition(cores, metadata={
            'center_block': sites, 'exact': True,
            'local_opening': dict(opening.diagnostics)})


def tt2tr(tt: Any, rank: Optional[Union[int, Sequence[int]]] = None,
          center: Optional[int] = None,
          loop_opener: Union[str, LoopOpener] = 'als',
          recursion: Union[str, GaugeRecursion] = 'adjoint',
          *, block_boundaries: bool = True, enrich: bool = True,
          enrich_rank: Optional[int] = None,
          enrich_rtol: float = 1e-12, rtol: Optional[float] = None,
          out_device: _Device = 'cpu', collect_metrics: bool = False,
          verbose: Union[bool, int] = 0) -> TRDecomposition:
    """
    Converts a tensor train to a tensor ring.

    Use :class:`TT2TR` for repeated fits of one input tensor train. All
    arguments have the same meaning as in :meth:`TT2TR.fit`.

    Parameters
    ----------
    tt : TT, MPS or sequence of torch.Tensor
        Open tensor train, open-boundary MPS or cores in standard TT order.
    rank : int or sequence of int, optional
        Local ALS rank or one right-link rank per core, including the closing
        link. At SVD splits these are upper bounds; ``rtol`` may reduce them.
        ``None`` discovers the rank profile. No zero padding is applied.
    center : int, optional
        Initial site or discovery-block center. Defaults to the middle site.
    loop_opener : str or LoopOpener, optional
        Local strategy: ``"als"``, ``"blostr"``, ``"blostr_svd"`` or a
        configured opener object.
    recursion : str or GaugeRecursion, optional
        ``"adjoint"``, ``"pinv"``, ``"orth"`` or a configured strategy.
        Exact spectral conversion does not use gauge recursion.
    block_boundaries : bool
        Group boundary sites into automatic blocks before gauge absorption
        and SVD splitting. Default is True. False keeps one-site boundaries
        and requires supplied ``rank`` for optimization openers. The
        initial block still grows automatically. Exact spectral conversion
        does not use this option.
    enrich : bool
        Whether to append orthogonal residual constraints. Default is True.
    enrich_rank : int, optional
        Maximum auxiliary coordinates per transition. ``None`` retains all
        significant directions.
    enrich_rtol : float
        Relative residual singular-value threshold. Default is 1e-12.
    rtol : float, optional
        Relative SVD tail-energy tolerance for discovery and block splits.
        Default ``None`` truncates only to ``rank`` when ranks are supplied;
        discovery without ranks uses 1e-12.
    out_device : str or torch.device, optional
        Final core device. Default is ``"cpu"``; ``None`` keeps the input
        device.
    collect_metrics : bool
        Whether to retain local diagnostics. Final error is always measured.
    verbose : bool or int
        Console verbosity from 0 (silent) to 3 (final cores included).

    Returns
    -------
    TRDecomposition
        Tensor ring cores, reconstruction metrics and construction metadata.
    """
    return TT2TR(tt, out_device=out_device).fit(
        rank, center, loop_opener, recursion,
        block_boundaries=block_boundaries, enrich=enrich,
        enrich_rank=enrich_rank, enrich_rtol=enrich_rtol, rtol=rtol,
        collect_metrics=collect_metrics, verbose=verbose)
