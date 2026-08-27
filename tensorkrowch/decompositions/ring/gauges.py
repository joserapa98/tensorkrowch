"""Oriented virtual-basis maps used by tensor-ring recursions."""

import warnings
from dataclasses import dataclass, field
from math import isfinite
from typing import (Any, Mapping, Optional, Protocol, Sequence, Tuple,
                    runtime_checkable)

import torch

from tensorkrowch.decompositions.metrics import GaugeRecord
from tensorkrowch.decompositions.observers import DecompositionEvent
from tensorkrowch.decompositions.ring.opening import LoopOpening


class ExperimentalWarning(UserWarning):
    """Warns that an opt-in decomposition strategy is still experimental."""


def _validate_non_negative_float(value: Optional[float],
                                 name: str) -> Optional[float]:
    """Validates an optional finite non-negative scalar."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'`{name}` should be a real scalar or None')
    value = float(value)
    if value < 0 or not isfinite(value):
        raise ValueError(f'`{name}` should be finite and non-negative')
    return value


def _matrix_from_core(core: torch.Tensor,
                      orientation: str) -> torch.Tensor:
    """Places the transported dimension in rows and both TR ranks in columns."""
    if orientation == 'left':
        return core.permute(1, 0, 2).reshape(core.shape[1], -1)
    return core.permute(1, 2, 0).reshape(core.shape[1], -1)


def _core_from_matrix(matrix: torch.Tensor,
                      orientation: str,
                      cyclic_rank: int,
                      local_rank: int) -> torch.Tensor:
    """Restores the standard oriented gauge-core shape from its matrix."""
    external_dim = matrix.shape[0]
    tensor = matrix.reshape(external_dim, cyclic_rank, local_rank)
    if orientation == 'left':
        return tensor.permute(1, 0, 2)
    return tensor.permute(2, 0, 1)


@dataclass(frozen=True)
class GaugeMap:
    """Represents an oriented map between external and cyclic virtual bases.

    A left gauge has shape ``(cyclic_rank, external_dim, local_rank)`` and a
    right gauge has shape ``(local_rank, external_dim, cyclic_rank)``. Both
    orientations are matricized as
    ``external_dim x (cyclic_rank * local_rank)``. Mirroring a gauge therefore
    changes its core orientation while preserving the represented matrix.

    Calling :meth:`inverse_or_pinv` returns the directional dual ``F`` that
    aims to satisfy ``G.T @ F = I``. Tensor-network links use a bilinear index
    contraction, so this directional transpose deliberately does not conjugate
    complex gauges.
    """

    core: torch.Tensor
    orientation: str
    site: Optional[int] = None
    inverse_method: Optional[str] = None
    _reference_matrix: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False)
    _rank_rtol: Optional[float] = field(
        default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.core, torch.Tensor):
            raise TypeError('`core` should be torch.Tensor type')
        if self.core.ndim != 3:
            raise ValueError('`core` should be a three-dimensional gauge')
        if not self.core.is_floating_point() and not self.core.is_complex():
            raise TypeError('Gauge cores should have a floating or complex dtype')
        if self.orientation not in ('left', 'right'):
            raise ValueError("`orientation` should be 'left' or 'right'")
        if self.site is not None:
            if isinstance(self.site, bool) or not isinstance(self.site, int):
                raise TypeError('`site` should be int type or None')
            if self.site < 0:
                raise ValueError('`site` should be non-negative')
        if self.inverse_method is not None and \
                self.inverse_method not in ('solve', 'inverse', 'pinv'):
            raise ValueError(
                "`inverse_method` should be 'solve', 'inverse', 'pinv' or None")
        rank_rtol = _validate_non_negative_float(
            self._rank_rtol, 'rank_rtol')
        object.__setattr__(self, '_rank_rtol', rank_rtol)
        if self._reference_matrix is not None:
            reference = self._reference_matrix
            if not isinstance(reference, torch.Tensor) or reference.ndim != 2:
                raise ValueError('`_reference_matrix` should be a matrix')
            if reference.shape != self.matrix.shape:
                raise ValueError(
                    'The reference and dual gauge matrices should match')
            if reference.device != self.core.device or \
                    reference.dtype != self.core.dtype:
                raise ValueError(
                    'The reference and dual gauge should share runtime')

    @property
    def matrix(self) -> torch.Tensor:
        """Returns shape ``external_dim x (cyclic_rank * local_rank)``."""
        return _matrix_from_core(self.core, self.orientation)

    @property
    def cyclic_rank(self) -> int:
        """Cyclic rank encoded by this orientation."""
        return self.core.shape[0] if self.orientation == 'left' \
            else self.core.shape[-1]

    @property
    def local_rank(self) -> int:
        """Non-cyclic local rank encoded by this orientation."""
        return self.core.shape[-1] if self.orientation == 'left' \
            else self.core.shape[0]

    @property
    def is_dual(self) -> bool:
        """Whether this map was created as the dual of another gauge."""
        return self._reference_matrix is not None

    def mirror(self) -> 'GaugeMap':
        """Returns the equivalent gauge in the opposite orientation."""
        orientation = 'right' if self.orientation == 'left' else 'left'
        return GaugeMap(
            core=self.core.permute(2, 1, 0),
            orientation=orientation,
            site=self.site,
            inverse_method=self.inverse_method,
            _reference_matrix=self._reference_matrix,
            _rank_rtol=self._rank_rtol)

    def inverse_or_pinv(self,
                        policy: str = 'auto',
                        *,
                        rank_rtol: Optional[float] = None) -> 'GaugeMap':
        """Builds a dual with solve, inverse or Moore--Penrose pseudoinverse.

        ``policy='auto'`` uses a linear solve for square gauges and falls back
        to a pseudoinverse if the solve fails. Rectangular gauges always use a
        pseudoinverse. ``rank_rtol`` controls both the pseudoinverse cutoff and
        the later numerical-rank diagnostic.
        """
        if not isinstance(policy, str):
            raise TypeError('`policy` should be str type')
        if policy not in ('auto', 'solve', 'inverse', 'pinv'):
            raise ValueError(
                "`policy` should be 'auto', 'solve', 'inverse' or 'pinv'")
        rank_rtol = _validate_non_negative_float(rank_rtol, 'rank_rtol')
        matrix = self.matrix
        square = matrix.shape[0] == matrix.shape[1]

        method = policy
        if policy == 'auto':
            method = 'solve' if square else 'pinv'
        if method in ('solve', 'inverse') and not square:
            raise ValueError(
                f"Gauge policy '{method}' requires a square matrix")

        if method == 'solve':
            identity = torch.eye(
                matrix.shape[1], dtype=matrix.dtype, device=matrix.device)
            try:
                dual_matrix = torch.linalg.solve(matrix.T, identity)
            except RuntimeError:
                if policy != 'auto':
                    raise
                method = 'pinv'
        if method == 'inverse':
            dual_matrix = torch.linalg.inv(matrix).T
        elif method == 'pinv':
            if rank_rtol is None:
                dual_matrix = torch.linalg.pinv(matrix).T
            else:
                dual_matrix = torch.linalg.pinv(
                    matrix, rcond=rank_rtol).T

        dual_core = _core_from_matrix(
            dual_matrix,
            self.orientation,
            self.cyclic_rank,
            self.local_rank)
        return GaugeMap(
            core=dual_core,
            orientation=self.orientation,
            site=self.site,
            inverse_method=method,
            _reference_matrix=matrix,
            _rank_rtol=rank_rtol)

    def diagnostics(self,
                    tolerance: float = 1e-8,
                    *,
                    rank_rtol: Optional[float] = None) -> GaugeRecord:
        """Measures numerical rank, condition and dual cancellation error."""
        tolerance = _validate_non_negative_float(tolerance, 'tolerance')
        rank_rtol = _validate_non_negative_float(rank_rtol, 'rank_rtol')
        if not self.is_dual:
            dual = self.inverse_or_pinv('pinv', rank_rtol=rank_rtol)
            return dual.diagnostics(
                tolerance=tolerance, rank_rtol=rank_rtol)

        reference = self._reference_matrix
        singular_values = torch.linalg.svdvals(reference)
        if rank_rtol is None:
            rank_rtol = self._rank_rtol
        if rank_rtol is None:
            rank_rtol = (
                torch.finfo(singular_values.dtype).eps * max(reference.shape))
        rank_tolerance = singular_values[0] * rank_rtol
        numerical_rank = int(
            (singular_values > rank_tolerance).sum().detach().cpu().item())
        if numerical_rank < min(reference.shape):
            condition_number = float('inf')
        else:
            condition_number = float(
                (singular_values[0] / singular_values[-1]).detach().cpu().item())

        product = reference.T @ self.matrix
        identity = torch.eye(
            product.shape[0], dtype=product.dtype, device=product.device)
        cancellation_error = (
            torch.linalg.vector_norm(product - identity) /
            torch.linalg.vector_norm(identity))
        cancellation_error = float(
            cancellation_error.detach().cpu().item())

        return GaugeRecord(
            orientation=self.orientation,
            shape=tuple(reference.shape),
            numerical_rank=numerical_rank,
            cancellable_rank=reference.shape[1],
            condition_number=condition_number,
            cancellation_error=cancellation_error,
            projective=numerical_rank < reference.shape[1],
            inverse_method=self.inverse_method,
            tolerance=tolerance,
            rank_tolerance=float(rank_tolerance.detach().cpu().item()),
            site=self.site)

    def require_cancellable(
            self,
            tolerance: float = 1e-8,
            allow_projective: bool = False,
            *,
            rank_rtol: Optional[float] = None) -> GaugeRecord:
        """Returns diagnostics or rejects a non-cancellable propagated gauge."""
        if not isinstance(allow_projective, bool):
            raise TypeError('`allow_projective` should be bool type')
        record = self.diagnostics(
            tolerance=tolerance, rank_rtol=rank_rtol)
        if not allow_projective and not record.cancellable:
            location = '' if record.site is None else f' at site {record.site}'
            raise ValueError(
                f'Cannot cancel {record.orientation} propagated gauge'
                f'{location}: shape={record.shape}, '
                f'rank={record.numerical_rank}/{record.cancellable_rank}, '
                f'cancellation_error={record.cancellation_error:.2e}. '
                'Increase the relevant source rank or pass '
                'allow_projective=True to accept the projected map.')
        return record

    def as_event(self,
                 phase: str,
                 tolerance: float = 1e-8,
                 *,
                 rank_rtol: Optional[float] = None,
                 level: int = 2) -> DecompositionEvent:
        """Returns a structured observer event with gauge diagnostics."""
        if not isinstance(phase, str):
            raise TypeError('`phase` should be str type')
        record = self.diagnostics(
            tolerance=tolerance, rank_rtol=rank_rtol)
        return DecompositionEvent(
            name='gauge',
            phase=phase,
            level=level,
            site=record.site,
            values={
                'orientation': record.orientation,
                'shape': record.shape,
                'rank': (
                    f'{record.numerical_rank}/{record.cancellable_rank}'),
                'condition_number': record.condition_number,
                'cancellation_error': record.cancellation_error,
                'projective': record.projective,
                'inverse_method': record.inverse_method,
            })


@dataclass(frozen=True)
class GaugeRecursionStep:
    """Stores the fixed gauge and diagnostics produced by one recursion."""

    gauge: torch.Tensor
    records: Sequence[GaugeRecord] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.gauge, torch.Tensor):
            raise TypeError('`gauge` should be torch.Tensor type')
        if self.gauge.ndim != 3:
            raise ValueError('`gauge` should be a three-dimensional core')
        records = tuple(self.records)
        if not all(isinstance(record, GaugeRecord) for record in records):
            raise TypeError('`records` should contain GaugeRecord objects')
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError('`diagnostics` should be a mapping')
        object.__setattr__(self, 'records', records)
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))


@runtime_checkable
class GaugeRecursion(Protocol):
    """Protocol for advancing an opened gauge to either neighboring site."""

    def advance_left(
            self,
            opening: LoopOpening,
            local_target: Any,
            recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Builds the fixed right gauge for the next site to the left."""

    def advance_right(
            self,
            opening: LoopOpening,
            local_target: Any,
            recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Builds the fixed left gauge for the next site to the right."""


class PseudoinverseGaugeRecursion:
    """Advances gauges by directional inverse or pseudoinverse cancellation.

    This is the direct recursion used by the characterized TT-to-TR method.
    An outgoing right gauge is dualized and mirrored into the fixed left gauge
    of the next site; the leftward operation is its exact mirror.
    """

    def __init__(self,
                 inverse_policy: str = 'pinv',
                 allow_projective: bool = False,
                 tolerance: float = 1e-8,
                 rank_rtol: Optional[float] = None) -> None:
        if inverse_policy not in ('auto', 'solve', 'inverse', 'pinv'):
            raise ValueError(
                "`inverse_policy` should be 'auto', 'solve', 'inverse' or "
                "'pinv'")
        if not isinstance(allow_projective, bool):
            raise TypeError('`allow_projective` should be bool type')
        self.inverse_policy = inverse_policy
        self.allow_projective = allow_projective
        self.tolerance = _validate_non_negative_float(
            tolerance, 'tolerance')
        self.rank_rtol = _validate_non_negative_float(
            rank_rtol, 'rank_rtol')

    def _advance(self,
                 opening: LoopOpening,
                 recursion_context: Mapping[str, Any],
                 direction: str) -> GaugeRecursionStep:
        """Dualizes and mirrors the outgoing gauge in one direction."""
        if not isinstance(opening, LoopOpening):
            raise TypeError('`opening` should be LoopOpening type')
        if not isinstance(recursion_context, Mapping):
            raise TypeError('`recursion_context` should be a mapping')
        to_sites = recursion_context.get('to_sites')
        if not isinstance(to_sites, tuple) or len(to_sites) != 1:
            raise ValueError(
                '`recursion_context` should identify one destination site')
        if direction == 'right':
            outgoing = opening.right_gauge
            orientation = 'right'
        else:
            outgoing = opening.left_gauge
            orientation = 'left'
        if outgoing is None:
            raise ValueError(
                f'The opening has no outgoing {orientation} gauge')

        fixed_map = GaugeMap(
            outgoing,
            orientation=orientation,
            site=to_sites[0]).inverse_or_pinv(
                self.inverse_policy,
                rank_rtol=self.rank_rtol).mirror()
        record = fixed_map.require_cancellable(
            tolerance=self.tolerance,
            allow_projective=self.allow_projective,
            rank_rtol=self.rank_rtol)
        return GaugeRecursionStep(
            gauge=fixed_map.core,
            records=(record,),
            diagnostics={
                'inverse_method': record.inverse_method,
                'projective': record.projective,
                'cancellation_error': record.cancellation_error,
            })

    def advance_left(
            self,
            opening: LoopOpening,
            local_target: Any,
            recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Builds the fixed right gauge for the next site to the left."""
        return self._advance(opening, recursion_context, 'left')

    def advance_right(
            self,
            opening: LoopOpening,
            local_target: Any,
            recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Builds the fixed left gauge for the next site to the right."""
        return self._advance(opening, recursion_context, 'right')


class TTCoreGaugeRecursion:
    """Transports opened TR bases through the original TT cores.

    The incoming gauge and retained TR core define a prefix or suffix basis.
    This strategy expresses that basis in the next TT virtual basis by solving
    a local coordinate problem against the original TT core. Thus the
    recursion mirrors the environment extension used by recursive sketching,
    with a TT core acting as the recursive projector.

    This strategy is experimental. It assumes that every recursion source is
    one TT site and that the provider exposes standardized TT cores.
    """

    def __init__(self,
                 inverse_policy: str = 'auto',
                 allow_projective: bool = False,
                 tolerance: float = 1e-8,
                 rank_rtol: Optional[float] = None) -> None:
        if inverse_policy not in ('auto', 'solve', 'inverse', 'pinv'):
            raise ValueError(
                "`inverse_policy` should be 'auto', 'solve', 'inverse' or "
                "'pinv'")
        if not isinstance(allow_projective, bool):
            raise TypeError('`allow_projective` should be bool type')
        self.inverse_policy = inverse_policy
        self.allow_projective = allow_projective
        self.tolerance = _validate_non_negative_float(
            tolerance, 'tolerance')
        self.rank_rtol = _validate_non_negative_float(
            rank_rtol, 'rank_rtol')
        warnings.warn(
            'TTCoreGaugeRecursion is experimental and its numerical '
            'behavior may change.',
            ExperimentalWarning,
            stacklevel=2)

    def _solve_coordinates(
            self,
            matrix: torch.Tensor,
            basis: torch.Tensor,
            orientation: str,
            site: int) -> Tuple[torch.Tensor, GaugeRecord]:
        """Expresses ``basis`` in the columns of one TT-core unfolding."""
        square = matrix.shape[0] == matrix.shape[1]
        method = self.inverse_policy
        if method == 'auto':
            method = 'solve' if square else 'pinv'
        if method in ('solve', 'inverse') and not square:
            raise ValueError(
                f"TT-core policy '{method}' requires a square unfolding")

        if method == 'solve':
            try:
                coordinates = torch.linalg.solve(matrix, basis)
            except RuntimeError:
                if self.inverse_policy != 'auto':
                    raise
                method = 'pinv'
        if method == 'inverse':
            coordinates = torch.linalg.inv(matrix) @ basis
        elif method == 'pinv':
            if self.rank_rtol is None:
                coordinates = torch.linalg.pinv(matrix) @ basis
            else:
                coordinates = torch.linalg.pinv(
                    matrix, rcond=self.rank_rtol) @ basis

        singular_values = torch.linalg.svdvals(matrix)
        if self.rank_rtol is None:
            rank_rtol = (
                torch.finfo(singular_values.dtype).eps * max(matrix.shape))
        else:
            rank_rtol = self.rank_rtol
        rank_tolerance = singular_values[0] * rank_rtol
        numerical_rank = int(
            (singular_values > rank_tolerance).sum().detach().cpu().item())
        if numerical_rank < min(matrix.shape):
            condition_number = float('inf')
        else:
            condition_number = float(
                (singular_values[0] / singular_values[-1])
                .detach().cpu().item())

        residual = matrix @ coordinates - basis
        residual_norm = torch.linalg.vector_norm(residual)
        denominator = torch.linalg.vector_norm(basis)
        if denominator > 0:
            projection_error = residual_norm / denominator
        else:
            projection_error = residual_norm
        projection_error = float(
            projection_error.detach().cpu().item())
        projective = (
            numerical_rank < matrix.shape[1] or
            projection_error > self.tolerance)
        record = GaugeRecord(
            orientation=orientation,
            shape=tuple(matrix.shape),
            numerical_rank=numerical_rank,
            cancellable_rank=matrix.shape[1],
            condition_number=condition_number,
            cancellation_error=projection_error,
            projective=projective,
            inverse_method=method,
            tolerance=self.tolerance,
            rank_tolerance=float(rank_tolerance.detach().cpu().item()),
            site=site)
        if not self.allow_projective and (
                record.projective or not record.cancellable):
            raise ValueError(
                f'Cannot transport the opened {orientation} basis through TT '
                f'core at site {site}: shape={record.shape}, '
                f'rank={record.numerical_rank}/{record.cancellable_rank}, '
                f'projection_error={record.cancellation_error:.2e}. Pass '
                'allow_projective=True to accept the projected recursion.')
        return coordinates, record

    @staticmethod
    def _source_core(
            recursion_context: Mapping[str, Any]) -> Tuple[torch.Tensor, int]:
        """Gets the standardized TT core associated with the source opening."""
        if not isinstance(recursion_context, Mapping):
            raise TypeError('`recursion_context` should be a mapping')
        from_sites = recursion_context.get('from_sites')
        if not isinstance(from_sites, tuple) or len(from_sites) != 1:
            raise ValueError(
                'TT-core recursion requires one source TT site')
        provider = recursion_context.get('provider')
        cores = getattr(provider, 'cores', None)
        if cores is None:
            raise TypeError(
                'TT-core recursion requires a provider with standardized cores')
        site = from_sites[0]
        core = cores[site]
        if not isinstance(core, torch.Tensor) or core.ndim != 3:
            raise ValueError('The source TT core should be three-dimensional')
        return core, site

    def _advance(self,
                 opening: LoopOpening,
                 recursion_context: Mapping[str, Any],
                 direction: str) -> GaugeRecursionStep:
        """Extends a retained TR prefix or suffix through one TT core."""
        if not isinstance(opening, LoopOpening):
            raise TypeError('`opening` should be LoopOpening type')
        if len(opening.cores) != 1:
            raise ValueError(
                'TT-core recursion currently requires one opened TT site')
        tt_core, site = self._source_core(recursion_context)
        tr_core = opening.cores[0]
        if tt_core.device != tr_core.device or tt_core.dtype != tr_core.dtype:
            raise ValueError(
                'TT and opened TR cores should share dtype and device')

        if direction == 'right':
            if opening.left_gauge is None:
                raise ValueError('The opening has no incoming left gauge')
            basis = torch.einsum(
                'gma,apb->mpgb', opening.left_gauge, tr_core)
            matrix = tt_core.reshape(-1, tt_core.shape[-1])
            coordinates, record = self._solve_coordinates(
                matrix,
                basis.reshape(matrix.shape[0], -1),
                orientation='right',
                site=site)
            gauge = coordinates.reshape(
                tt_core.shape[-1],
                opening.left_gauge.shape[0],
                tr_core.shape[-1]).permute(1, 0, 2)
        else:
            if opening.right_gauge is None:
                raise ValueError('The opening has no incoming right gauge')
            basis = torch.einsum(
                'apb,bng->pnag', tr_core, opening.right_gauge)
            matrix = tt_core.permute(1, 2, 0).reshape(
                -1, tt_core.shape[0])
            coordinates, record = self._solve_coordinates(
                matrix,
                basis.reshape(matrix.shape[0], -1),
                orientation='left',
                site=site)
            gauge = coordinates.reshape(
                tt_core.shape[0],
                tr_core.shape[0],
                opening.right_gauge.shape[-1]).permute(1, 0, 2)

        return GaugeRecursionStep(
            gauge=gauge,
            records=(record,),
            diagnostics={
                'inverse_method': record.inverse_method,
                'projective': record.projective,
                'projection_error': record.cancellation_error,
                'source_site': site,
            })

    def advance_left(
            self,
            opening: LoopOpening,
            local_target: Any,
            recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Builds the fixed right environment for the next site to the left."""
        return self._advance(opening, recursion_context, 'left')

    def advance_right(
            self,
            opening: LoopOpening,
            local_target: Any,
            recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Builds the fixed left environment for the next site to the right."""
        return self._advance(opening, recursion_context, 'right')

    def prepare_boundary(
            self,
            opening: LoopOpening,
            local_target: Any,
            recursion_context: Mapping[str, Any],
            direction: str) -> GaugeRecursionStep:
        """Returns an outgoing-oriented environment for TT-edge absorption."""
        if direction not in ('left', 'right'):
            raise ValueError("`direction` should be 'left' or 'right'")
        step = self._advance(opening, recursion_context, direction)
        incoming_orientation = 'right' if direction == 'left' else 'left'
        to_sites = recursion_context.get('to_sites')
        boundary_site = None if not to_sites else to_sites[0]
        outgoing_map = GaugeMap(
            step.gauge,
            orientation=incoming_orientation,
            site=boundary_site).inverse_or_pinv(
                self.inverse_policy,
                rank_rtol=self.rank_rtol).mirror()
        boundary_record = outgoing_map.require_cancellable(
            tolerance=self.tolerance,
            allow_projective=self.allow_projective,
            rank_rtol=self.rank_rtol)
        return GaugeRecursionStep(
            gauge=outgoing_map.core,
            records=(*step.records, boundary_record),
            diagnostics={
                **step.diagnostics,
                'boundary': True,
                'boundary_inverse_method': boundary_record.inverse_method,
                'boundary_cancellation_error': (
                    boundary_record.cancellation_error),
            })


__all__ = [
    'GaugeMap',
    'GaugeRecursionStep',
    'GaugeRecursion',
    'PseudoinverseGaugeRecursion',
    'TTCoreGaugeRecursion',
    'ExperimentalWarning',
]
