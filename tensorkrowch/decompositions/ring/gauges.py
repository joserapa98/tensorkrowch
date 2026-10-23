"""
This script contains:

    Public classes:
        * ExperimentalWarning
        * GaugeMap
        * GaugeRecursionStep
        * GaugeRecursion
        * OrthGaugeRecursion
        * PinvGaugeRecursion
        * GramGaugeRecursion
        * AdjointGaugeRecursion

    Internal functions:
        * _validate_non_negative_float
        * _matrix_from_core
        * _core_from_matrix
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import (Any,
                    Mapping,
                    Optional,
                    Protocol,
                    Sequence,
                    Tuple,
                    runtime_checkable)
import warnings

import torch

from tensorkrowch.utils import safe_divide

from tensorkrowch.decompositions.metrics import GaugeRecord
from tensorkrowch.decompositions.observers import DecompositionEvent
from tensorkrowch.decompositions.ring.opening import LoopOpening


class ExperimentalWarning(UserWarning):  # MARK: ExperimentalWarning
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


def _matrix_from_core(core: torch.Tensor, orientation: str) -> torch.Tensor:
    """
    Places the transported dimension in rows and both TR ranks in columns.
    """
    if orientation == 'left':
        return core.permute(1, 0, 2).reshape(core.shape[1], -1)
    return core.permute(1, 2, 0).reshape(core.shape[1], -1)


def _core_from_matrix(matrix: torch.Tensor,
                      orientation: str,
                      cyclic_rank: int,
                      local_rank: int) -> torch.Tensor:
    """Restores the standard oriented gauge-core shape from its ``matrix``."""
    external_dim = matrix.shape[0]
    tensor = matrix.reshape(external_dim, cyclic_rank, local_rank)
    if orientation == 'left':
        return tensor.permute(1, 0, 2)
    return tensor.permute(2, 0, 1)


def _weight_rows(matrix: torch.Tensor,
                 basis: torch.Tensor,
                 factor: torch.Tensor,
                 direction: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies a Gram square root to the incoming virtual row index."""
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


@dataclass(frozen=True)
class GaugeMap:  # MARK: GaugeMap
    """
    Represents an oriented map between external and cyclic virtual bases.

    A left gauge has shape ``(cyclic_rank, external_dim, local_rank)`` and a
    right gauge has shape ``(local_rank, external_dim, cyclic_rank)``. Both
    orientations are matricized as ``external_dim x (cyclic_rank *
    local_rank)``. Mirroring a gauge therefore changes its ``core`` ``orientation``
    while preserving the represented matrix.

    Calling :meth:`inverse_or_pinv` returns the directional dual ``F`` that
    aims to satisfy ``G.T @ F = I``. Tensor network links use a bilinear index
    contraction, so this directional transpose deliberately does not conjugate
    complex gauges.
    """

    core: torch.Tensor  # Raw local tensor with standard rank axes
    orientation: str  # Left or right interpretation of the gauge axes
    site: Optional[int] = None  # Optional zero-based active site
    # Effective method used to construct the directional dual
    inverse_method: Optional[str] = None
    # Original gauge matrix whose dual is represented
    _reference_matrix: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False)
    # Relative singular-value cutoff used for the dual
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
        """
        Builds a dual with solve, inverse or Moore--Penrose pseudoinverse.

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
                    tolerance: float = 1e-08,
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
            condition_number = singular_values.new_tensor(torch.inf)
        else:
            condition_number = singular_values[0] / singular_values[-1]

        product = reference.T @ self.matrix
        identity = torch.eye(
            product.shape[0], dtype=product.dtype, device=product.device)
        cancellation_error = (
            torch.linalg.vector_norm(product - identity) /
            torch.linalg.vector_norm(identity))

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
            rank_tolerance=rank_tolerance,
            site=self.site)

    def require_cancellable(self,
                            tolerance: float = 1e-08,
                            allow_projective: bool = False,
                            *,
                            rank_rtol: Optional[float] = None) -> GaugeRecord:
        """
        Returns diagnostics or rejects a non-cancellable propagated
        gauge.
        """
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
                 tolerance: float = 1e-08,
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
class GaugeRecursionStep:  # MARK: GaugeRecursionStep
    """
    Stores the fixed ``gauge`` and ``diagnostics`` produced by one recursion.
    """

    gauge: torch.Tensor  # Gauge passed to the next local problem
    # Gauge measurements produced by this transition
    records: Sequence[GaugeRecord] = ()
    # Local strategy diagnostics and construction state
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
class GaugeRecursion(Protocol):  # MARK: GaugeRecursion
    """Protocol for advancing an opened gauge to either neighboring site."""

    def advance_left(self,
                     opening: LoopOpening,
                     local_target: Any,
                     recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Builds the fixed right gauge for the next site to the left."""

    def advance_right(self,
                      opening: LoopOpening,
                      local_target: Any,
                      recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Builds the fixed left gauge for the next site to the right."""


class OrthGaugeRecursion:  # MARK: OrthGaugeRecursion
    """
    Transports using the directional dual of the learned outgoing gauge.

    The Moore-Penrose dual constructs an orthogonal support projector. An
    outgoing right gauge is dualized and mirrored into the fixed left gauge of
    the next site; the leftward operation is its exact mirror.

    Parameters
    ----------
    inverse_policy : {"auto", "solve", "inverse", "pinv"}
        Method used for the directional map. Default is ``"pinv"``. ``"auto"``
        uses a solve for square systems and a pseudoinverse otherwise.
    allow_projective : bool
        Whether a rank-deficient map or a projection residual above tolerance
        is allowed. Default is False; incompatible cancellation then raises an
        error.
    tolerance : float
        Maximum accepted relative cancellation/projection error. Default is
        1e-8.
    rank_rtol : float, optional
        Relative singular-value cutoff for pseudoinverses and numerical ranks.
        If omitted, the dtype and matrix dimensions determine the cutoff.
    """

    def __init__(self,
                 inverse_policy: str = 'pinv',
                 allow_projective: bool = False,
                 tolerance: float = 1e-08,
                 rank_rtol: Optional[float] = None) -> None:
        if not isinstance(inverse_policy, str):
            raise TypeError('`inverse_policy` should be str type')
        if inverse_policy not in ('auto', 'solve', 'inverse', 'pinv'):
            raise ValueError(
                "`inverse_policy` should be 'auto', 'solve', 'inverse' or "
                "'pinv'")
        if not isinstance(allow_projective, bool):
            raise TypeError('`allow_projective` should be bool type')
        self.inverse_policy = inverse_policy
        self.allow_projective = allow_projective
        self.tolerance = _validate_non_negative_float(tolerance, 'tolerance')
        if self.tolerance is None:
            raise TypeError('`tolerance` should be a non-negative float')
        self.rank_rtol = _validate_non_negative_float(
            rank_rtol, 'rank_rtol')

    def _advance(self,
                 opening: LoopOpening,
                 recursion_context: Mapping[str, Any],
                 direction: str) -> GaugeRecursionStep:
        """Dualizes and mirrors the outgoing gauge in one ``direction``."""
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

    def advance_left(self,
                     opening: LoopOpening,
                     local_target: Any,
                     recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Builds the fixed right gauge for the next site to the left."""
        return self._advance(opening, recursion_context, 'left')

    def advance_right(self,
                      opening: LoopOpening,
                      local_target: Any,
                      recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Builds the fixed left gauge for the next site to the right."""
        return self._advance(opening, recursion_context, 'right')


class PinvGaugeRecursion(OrthGaugeRecursion):  # MARK: PinvGaugeRecursion
    """
    Transports opened TR bases through local TT-core pseudoinverses.

    The incoming gauge and retained TR core define a prefix or suffix basis.
    This strategy expresses that basis in the next TT virtual basis by solving
    a local coordinate problem against the original TT core. Thus the recursion
    mirrors the environment extension used by recursive sketching, with a TT
    unfolding acting as the recursive coordinate map. An initial
    multi-site block is contracted and pseudoinverted as a whole; this is
    not a product of its individual core pseudoinverses.

    This strategy is experimental. The provider exposes standardized TT cores
    or initial contiguous blocks. TT2TR boundary providers absorb the outgoing
    learned gauges directly, without constructing a second directional dual.

    Parameters
    ----------
    inverse_policy : {"auto", "solve", "inverse", "pinv"}
        Method used for the directional map. Default is ``"auto"``. ``"auto"``
        uses a solve for square systems and a pseudoinverse otherwise.
    allow_projective : bool
        Whether a rank-deficient map or a projection residual above tolerance
        is allowed. Default is False; incompatible cancellation then raises an
        error.
    tolerance : float
        Maximum accepted relative cancellation/projection error. Default is
        1e-8.
    rank_rtol : float, optional
        Relative singular-value cutoff for pseudoinverses and numerical ranks.
        If omitted, the dtype and matrix dimensions determine the cutoff.
    """

    def __init__(self,
                 inverse_policy: str = 'auto',
                 allow_projective: bool = False,
                 tolerance: float = 1e-08,
                 rank_rtol: Optional[float] = None) -> None:
        super().__init__(inverse_policy, allow_projective, tolerance, rank_rtol)
        warnings.warn(
            f'{type(self).__name__} is experimental and its numerical '
            'behavior may change.',
            ExperimentalWarning,
            stacklevel=2)

    def _solve_coordinates(self,
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

        return coordinates, self._coordinate_record(
            matrix, basis, coordinates, method, orientation, site)

    def _coordinate_record(self,
                           matrix: torch.Tensor,
                           basis: torch.Tensor,
                           coordinates: torch.Tensor,
                           method: str,
                           orientation: str,
                           site: int) -> GaugeRecord:
        """Measures the residual in the coordinates used by the transport."""
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
            condition_number = singular_values.new_tensor(torch.inf)
        else:
            condition_number = singular_values[0] / singular_values[-1]

        residual = matrix @ coordinates - basis
        residual_norm = torch.linalg.vector_norm(residual)
        denominator = torch.linalg.vector_norm(basis)
        projection_error = safe_divide(
            residual_norm, denominator, zero_value=residual_norm)
        projective = (
            numerical_rank < matrix.shape[1] or
            bool(projection_error > self.tolerance))
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
            rank_tolerance=rank_tolerance,
            site=site)
        if not self.allow_projective and (
                record.projective or not record.cancellable):
            raise ValueError(
                f'Cannot transport the opened {orientation} basis through TT '
                f'core at site {site}: shape={record.shape}, '
                f'rank={record.numerical_rank}/{record.cancellable_rank}, '
                f'projection_error={record.cancellation_error:.2e}. Pass '
                'allow_projective=True to accept the projected recursion.')
        return record

    def _transport_coordinates(self,
                               matrix: torch.Tensor,
                               basis: torch.Tensor,
                               direction: str,
                               site: int,
                               context: Mapping[str, Any]
                               ) -> Tuple[torch.Tensor, GaugeRecord]:
        """Solves the Euclidean coordinate problem for one core unfolding."""
        return self._solve_coordinates(matrix, basis, direction, site)

    @staticmethod
    def _source_core(
        recursion_context: Mapping[str, Any]
    ) -> Tuple[torch.Tensor, int]:
        """Gets the contracted TT target associated with the source opening."""
        if not isinstance(recursion_context, Mapping):
            raise TypeError('`recursion_context` should be a mapping')
        from_sites = recursion_context.get('from_sites')
        if not isinstance(from_sites, tuple) or not from_sites:
            raise ValueError(
                'TT-core recursion requires source TT sites')
        provider = recursion_context.get('provider')
        cores = getattr(provider, 'cores', None)
        if cores is None:
            raise TypeError(
                'TT-core recursion requires a provider with standardized cores')
        site = from_sites[0]
        core = (cores[site] if len(from_sites) == 1 else
                provider.local_target(from_sites, recursion_context))
        if isinstance(core, torch.Tensor) and core.ndim > 3:
            core = core.reshape(core.shape[0], -1, core.shape[-1])
        if not isinstance(core, torch.Tensor) or core.ndim != 3:
            raise ValueError('The source TT core should be three-dimensional')
        return core, site

    @staticmethod
    def _local_matrices(opening: LoopOpening,
                        recursion_context: Mapping[str, Any],
                        direction: str
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forms the source unfolding and learned prefix/suffix factor once."""
        if not isinstance(opening, LoopOpening):
            raise TypeError('`opening` should be LoopOpening type')
        if len(opening.cores) != len(recursion_context['from_sites']):
            raise ValueError(
                'Opened TR cores should match the source TT sites')
        tt_core, _ = PinvGaugeRecursion._source_core(recursion_context)
        tr_core = opening.cores[0]
        for core in opening.cores[1:]:
            tr_core = torch.tensordot(tr_core, core, dims=([-1], [0]))
        tr_core = tr_core.reshape(tr_core.shape[0], -1, tr_core.shape[-1])
        if tt_core.device != tr_core.device or tt_core.dtype != tr_core.dtype:
            raise ValueError(
                'TT and opened TR cores should share dtype and device')

        if direction == 'right':
            if opening.left_gauge is None:
                raise ValueError('The opening has no incoming left gauge')
            basis = torch.einsum(
                'gma,apb->mpgb', opening.left_gauge, tr_core)
            matrix = tt_core.reshape(-1, tt_core.shape[-1])
        else:
            if opening.right_gauge is None:
                raise ValueError('The opening has no incoming right gauge')
            basis = torch.einsum(
                'apb,bng->pnag', tr_core, opening.right_gauge)
            matrix = tt_core.permute(1, 2, 0).reshape(-1, tt_core.shape[0])
        return matrix, basis.reshape(matrix.shape[0], -1), tr_core

    def _advance(self,
                 opening: LoopOpening,
                 recursion_context: Mapping[str, Any],
                 direction: str) -> GaugeRecursionStep:
        """Extends a retained TR prefix or suffix through its source block."""
        matrix, basis, tr_core = self._local_matrices(
            opening, recursion_context, direction)
        sites = recursion_context['from_sites']
        site = sites[-1] if direction == 'right' else sites[0]
        coordinates, record = self._transport_coordinates(
            matrix, basis, direction, site, recursion_context)
        cyclic_rank = (opening.left_gauge.shape[0] if direction == 'right'
                       else tr_core.shape[0])
        local_rank = (tr_core.shape[-1] if direction == 'right'
                      else opening.right_gauge.shape[-1])
        gauge = coordinates.reshape(
            matrix.shape[1], cyclic_rank, local_rank).permute(1, 0, 2)

        return GaugeRecursionStep(
            gauge=gauge,
            records=(record,),
            diagnostics={
                'inverse_method': record.inverse_method,
                'projective': record.projective,
                'projection_error': record.cancellation_error,
                'source_site': site,
            })

class GramGaugeRecursion(PinvGaugeRecursion):  # MARK: GramGaugeRecursion
    r"""
    Transports gauges using the pseudoinverse of each complete TT subchain.

    For a left prefix :math:`L_k=(L_{k-1}\otimes I)B_k`, cache QR factors
    :math:`F_k` with :math:`H_k=L_k^\dagger L_k=F_k^\dagger F_k`. The
    rightward coordinate map is

    .. math::

        D_k=[(F_{k-1}\otimes I)B_k]^+(F_{k-1}\otimes I)
           =H_k^+B_k^\dagger(H_{k-1}\otimes I).

    Suffixes use the mirrored construction, including complex conjugation
    through the matrix adjoint. QR factors avoid squaring the condition
    number in the numerical solve; ``rank_rtol`` applies to subchain singular
    values, not their squares. The provider caches both families once per fit.

    Local ALS targets keep their original coordinates and Euclidean norm.
    This strategy changes transport, not the ALS objective. TT2TR edge closure
    directly absorbs the learned outgoing gauge. This strategy does not
    optimize the global TR error.
    Initial multi-site blocks are supported with the ``center_out`` schedule.

    Parameters are those of :class:`PinvGaugeRecursion`. With
    ``inverse_policy="pinv"`` the coordinate map is Moore-Penrose on its
    retained support. ``"solve"`` and ``"inverse"`` require square systems.
    """

    def _transport_coordinates(self,
                               matrix: torch.Tensor,
                               basis: torch.Tensor,
                               direction: str,
                               site: int,
                               context: Mapping[str, Any]
                               ) -> Tuple[torch.Tensor, GaugeRecord]:
        """Weights local rows by a cached square root of the subchain Gram."""
        provider = context['provider']
        factor = (provider.left_factors[context['from_sites'][0]]
                  if direction == 'right' else
                  provider.right_factors[context['from_sites'][-1] + 1])
        matrix, basis = _weight_rows(matrix, basis, factor, direction)
        return self._solve_coordinates(matrix, basis, direction, site)


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


class AdjointGaugeRecursion(PinvGaugeRecursion):  # MARK: AdjointGaugeRecursion
    r"""
    Transports opened factors through directional TT isometries.

    For an ordinary TT, :class:`~tensorkrowch.decompositions.TT2TR` first
    prepares a mixed canonical form. Each transition factorizes the source
    core or contracted block as :math:`B=QF`, transports with :math:`Q^H`,
    and absorbs :math:`F` into the next reference target. This preserves the
    input tensor while moving its orthogonality center independently along
    the two outward branches.

    A valid :class:`~tensorkrowch.formats.VidalGauge` instead supplies the
    isometries directly. Stored Schmidt powers are removed or absorbed on
    both sides before applying the adjoint; powers may differ at each bond.
    The input's local targets and their ALS metric are retained. Explicit,
    implicit and inverse Vidal therefore share the same transport strategy.

    Parameters are those of :class:`PinvGaugeRecursion`; the adjoint replaces
    its coordinate solve. Optional TT2TR expansion adds orthogonal residual
    coordinates and zero reference rows without increasing the TR ranks.
    The schedule must be ``"center_out"``. This strategy is experimental.
    """

    def _advance_qr(self,
                    opening: LoopOpening,
                    recursion_context: Mapping[str, Any],
                    direction: str,
                    max_extra_rank: Optional[int] = 0,
                    residual_rtol: float = 1e-12) -> GaugeRecursionStep:
        """Moves the canonical center while retaining residual coordinates."""
        provider = recursion_context['provider']
        shift_center = getattr(provider, 'shift_center', None)
        if not callable(shift_center):
            raise TypeError(
                'Adjoint QR transport requires a canonical TT target provider')
        matrix, basis, block = self._local_matrices(
            opening, recursion_context, direction)
        q, factor = torch.linalg.qr(matrix, mode='reduced')
        residual = basis - q @ (q.mH @ basis)
        residual = residual - q @ (q.mH @ residual)
        u, singular, _ = torch.linalg.svd(residual, full_matrices=False)
        capacity = max(0, matrix.shape[0] - q.shape[1])
        if max_extra_rank is not None:
            capacity = min(capacity, max_extra_rank)
        selected = min(capacity, int((
            singular > basis.norm() * residual_rtol).sum()))
        old_dimension = q.shape[1]
        if selected:
            extra = u[:, :selected] - q @ (q.mH @ u[:, :selected])
            extra = torch.linalg.qr(extra, mode='reduced').Q
            q = torch.cat((q, extra), dim=1)
            factor = torch.cat((
                factor, factor.new_zeros(selected, factor.shape[1])), dim=0)
        coordinates = q.mH @ basis
        shift_center(recursion_context['to_sites'][0], factor, direction)
        if direction == 'right':
            gauge = coordinates.reshape(
                q.shape[1], opening.left_gauge.shape[0],
                block.shape[-1]).permute(1, 0, 2)
        else:
            gauge = coordinates.reshape(
                q.shape[1], block.shape[0],
                opening.right_gauge.shape[-1]).permute(1, 0, 2)
        identity = torch.eye(q.shape[1], dtype=q.dtype, device=q.device)
        return GaugeRecursionStep(gauge=gauge, diagnostics={
            'algorithm': 'orthogonal_residual_enrichment',
            'old_dimension': old_dimension, 'new_dimension': q.shape[1],
            'added': selected,
            'residual_before': safe_divide(residual.norm(), basis.norm()),
            'residual_after': safe_divide(
                (basis - q @ coordinates).norm(), basis.norm()),
            'target_preservation_error': safe_divide(
                (q @ factor - matrix).norm(), matrix.norm()),
            'orthogonality_error': (q.mH @ q - identity).norm(),
        })

    def _advance(self,
                 opening: LoopOpening,
                 recursion_context: Mapping[str, Any],
                 direction: str) -> GaugeRecursionStep:
        """Uses QR for ordinary targets and Schmidt adjoints for Vidal."""
        if callable(getattr(recursion_context['provider'], 'shift_center', None)):
            return self._advance_qr(opening, recursion_context, direction)
        return super()._advance(opening, recursion_context, direction)

    def _isometric_coordinates(self,
                               matrix: torch.Tensor,
                               basis: torch.Tensor,
                               direction: str,
                               context: Mapping[str, Any]
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Weights incoming rows and removes outgoing absorbed powers."""
        provider = context['provider']
        first, last = context['from_sites'][0], context['from_sites'][-1]
        if direction == 'right':
            factor = provider.left_factors[first]
            spectrum = provider.spectra[last]
            power = -provider.powers[last][0]
        else:
            factor = provider.right_factors[last + 1]
            spectrum = provider.spectra[first - 1]
            power = -provider.powers[first - 1][1]
        matrix, basis = _weight_rows(matrix, basis, factor, direction)
        return (matrix * _schmidt_power(spectrum, power).unsqueeze(0), basis)

    def _transport_coordinates(self,
                               matrix: torch.Tensor,
                               basis: torch.Tensor,
                               direction: str,
                               site: int,
                               context: Mapping[str, Any]
                               ) -> Tuple[torch.Tensor, GaugeRecord]:
        """Projects isometrically, then converts to the next target's powers."""
        isometry, weighted = self._isometric_coordinates(
            matrix, basis, direction, context)
        center_coordinates = isometry.mH @ weighted
        record = self._coordinate_record(
            isometry, weighted, center_coordinates, 'adjoint', direction, site)
        bond = site if direction == 'right' else site - 1
        provider = context['provider']
        power = provider.powers[bond][1 if direction == 'right' else 0] - 1
        coordinates = center_coordinates * _schmidt_power(
            context['provider'].spectra[bond], power).unsqueeze(1)
        return coordinates, record


class _ExpandedGaugeRecursion:
    """
    Appends orthogonal zero-target constraints, retaining the base transport.

    For weighted source columns ``M`` and retained factor ``X``, the new
    rows are ``U.H @ X`` with ``U`` spanning ``(I - M @ pinv(M)) @ X``.
    The original coordinates ``E`` are kept. The identity
    ``X = M @ E + U @ U.H @ X`` requires ``E = pinv(M) @ X`` and complete
    residual support; it does not hold for every learned-gauge transport.
    """

    def __init__(self,
                 recursion: GaugeRecursion,
                 max_extra_rank: Optional[int],
                 residual_rtol: float) -> None:
        if max_extra_rank is not None:
            if isinstance(max_extra_rank, bool) or not isinstance(
                    max_extra_rank, int):
                raise TypeError('`max_extra_rank` should be int type or None')
            if max_extra_rank < 0:
                raise ValueError('`max_extra_rank` should be non-negative')
        self.recursion = recursion
        self.max_extra_rank = max_extra_rank
        self.residual_rtol = _validate_non_negative_float(
            residual_rtol, 'residual_rtol')
        if self.residual_rtol is None:
            raise TypeError('`residual_rtol` should be a non-negative float')

    def _advance(self,
                 opening: LoopOpening,
                 local_target: Any,
                 context: Mapping[str, Any],
                 direction: str) -> GaugeRecursionStep:
        """Preserves base coordinates and appends the orthogonal residual."""
        if isinstance(self.recursion, AdjointGaugeRecursion) and callable(
                getattr(context['provider'], 'shift_center', None)):
            return self.recursion._advance_qr(
                opening, context, direction, self.max_extra_rank,
                self.residual_rtol)
        advance = (self.recursion.advance_right if direction == 'right'
                   else self.recursion.advance_left)
        step = advance(opening, local_target, context)
        matrix, basis, block = PinvGaugeRecursion._local_matrices(
            opening, context, direction)
        if isinstance(self.recursion, AdjointGaugeRecursion):
            matrix, basis = self.recursion._isometric_coordinates(
                matrix, basis, direction, context)
        elif isinstance(self.recursion, GramGaugeRecursion):
            provider = context['provider']
            first, last = context['from_sites'][0], context['from_sites'][-1]
            factor = (provider.left_factors[first] if direction == 'right'
                      else provider.right_factors[last + 1])
            matrix, basis = _weight_rows(matrix, basis, factor, direction)
        # An SVD identifies support without assigning arbitrary bases to null
        # reference columns. Its cutoff is numerical, not a TT truncation.
        q, values, _ = torch.linalg.svd(matrix, full_matrices=False)
        cutoff = torch.finfo(values.dtype).eps * max(matrix.shape) * values[0]
        q = q[:, values > cutoff]
        residual = basis - q @ (q.mH @ basis)
        residual = residual - q @ (q.mH @ residual)
        u, values, _ = torch.linalg.svd(residual, full_matrices=False)
        capacity = matrix.shape[0] - q.shape[1]
        if self.max_extra_rank is not None:
            capacity = min(capacity, self.max_extra_rank)
        selected = min(capacity, int((
            values > basis.norm() * self.residual_rtol).sum()))
        extra = u[:, :selected]
        if selected:
            extra = extra - q @ (q.mH @ extra)
            extra = torch.linalg.qr(extra, mode='reduced').Q
        coordinates = extra.mH @ basis
        if direction == 'right':
            added_gauge = coordinates.reshape(
                selected, opening.left_gauge.shape[0], block.shape[-1])
        else:
            added_gauge = coordinates.reshape(
                selected, block.shape[0], opening.right_gauge.shape[-1])
        old_dimension = step.gauge.shape[1]
        gauge = torch.cat((step.gauge, added_gauge.permute(1, 0, 2)), dim=1)
        provider = context['provider']
        destination = context['to_sites'][0]
        before = provider.cores[destination]
        provider.expand_coordinates(destination, selected, direction)
        after = provider.cores[destination]
        axis = 0 if direction == 'right' else 2
        original = after.narrow(axis, 0, before.shape[axis])
        zeros = after.narrow(axis, before.shape[axis], selected)
        preservation_error = safe_divide(
            ((original - before).norm().square() + zeros.norm().square()).sqrt(),
            before.norm())
        identity = torch.eye(selected, dtype=extra.dtype, device=extra.device)
        diagnostics = {**step.diagnostics,
            'algorithm': 'orthogonal_residual_enrichment',
            'base_recursion': type(self.recursion).__name__,
            'old_dimension': old_dimension,
            'new_dimension': gauge.shape[1], 'added': selected,
            'residual_before': safe_divide(residual.norm(), basis.norm()),
            'residual_after': safe_divide(
                (residual - extra @ coordinates).norm(), basis.norm()),
            'target_preservation_error': preservation_error,
            'orthogonality_error': (extra.mH @ extra - identity).norm(),
            'cross_orthogonality_error': (q.mH @ extra).norm(),
        }
        return GaugeRecursionStep(gauge, step.records, diagnostics)

    def advance_right(self,
                      opening: LoopOpening,
                      local_target: Any,
                      recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Expands the next left gauge in rightward transport."""
        return self._advance(opening, local_target, recursion_context, 'right')

    def advance_left(self,
                     opening: LoopOpening,
                     local_target: Any,
                     recursion_context: Mapping[str, Any]) -> GaugeRecursionStep:
        """Expands the next right gauge in leftward transport."""
        return self._advance(opening, local_target, recursion_context, 'left')


__all__ = [
    'GaugeMap',
    'GaugeRecursionStep',
    'GaugeRecursion',
    'OrthGaugeRecursion',
    'PinvGaugeRecursion',
    'GramGaugeRecursion',
    'AdjointGaugeRecursion',
    'ExperimentalWarning',
]
