"""Oriented virtual-basis maps used by tensor-ring recursions."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Optional

import torch

from tensorkrowch.decompositions.metrics import GaugeRecord
from tensorkrowch.decompositions.observers import DecompositionEvent


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
    aims to satisfy ``G.mH @ F = I``. The conjugate transpose is essential for
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
                dual_matrix = torch.linalg.solve(matrix.mH, identity)
            except RuntimeError:
                if policy != 'auto':
                    raise
                method = 'pinv'
        if method == 'inverse':
            dual_matrix = torch.linalg.inv(matrix).mH
        elif method == 'pinv':
            if rank_rtol is None:
                dual_matrix = torch.linalg.pinv(matrix).mH
            else:
                dual_matrix = torch.linalg.pinv(
                    matrix, rcond=rank_rtol).mH

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

        product = reference.mH @ self.matrix
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


__all__ = ['GaugeMap']
