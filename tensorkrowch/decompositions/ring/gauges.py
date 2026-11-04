"""
This script contains:

    Classes:
        * ExperimentalWarning
        * GaugeMap
        * GaugeRecursion
        * OrthGaugeRecursion
        * PinvGaugeRecursion
        * AdjointGaugeRecursion
        * _ExpandedGaugeRecursion

    Functions:
        * _validate_non_negative_float
        * _matrix_from_core
        * _core_from_matrix
        * _pinv_options
        * _rank_statistics
        * _residual_directions
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
    if (value < 0) or (not isfinite(value)):
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


def _pinv_options(options: Mapping[str, Any]) -> dict:
    """Validates scalar pseudoinverse options for an unbatched gauge."""
    unknown = set(options).difference(('atol', 'rtol', 'hermitian'))
    if unknown:
        raise TypeError(
            f'Unsupported pseudoinverse options: {sorted(unknown)}')
    result = dict(options)
    for name in ('atol', 'rtol'):
        value = result.get(name)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if (value.ndim != 0) or value.is_complex():
                raise ValueError(f'`{name}` should be a real scalar tensor')
            valid = torch.isfinite(value) & (value >= 0)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f'`{name}` should be a non-negative scalar')
            valid = isfinite(value) and (value >= 0)
        if not bool(valid):
            raise ValueError(f'`{name}` should be finite and non-negative')
    if ('hermitian' in result) and not isinstance(result['hermitian'], bool):
        raise TypeError('`hermitian` should be bool type')
    return result


def _rank_statistics(matrix: torch.Tensor,
                     options: Mapping[str, Any]
                     ) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """Measures rank using the same cutoff as the scalar pseudoinverse."""
    if options.get('hermitian', False):
        values = torch.linalg.eigvalsh(matrix).abs()
        values = values.sort(descending=True).values
    else:
        values = torch.linalg.svdvals(matrix)
    atol = options.get('atol')
    atol = values.new_zeros(()) if atol is None else torch.as_tensor(
        atol, dtype=values.dtype, device=values.device)
    rtol = options.get('rtol')
    if rtol is None:
        rtol = 0.0 if bool(atol > 0) else (
            max(matrix.shape) * torch.finfo(values.dtype).eps)
    rtol = torch.as_tensor(rtol, dtype=values.dtype, device=values.device)
    cutoff = torch.maximum(atol, rtol * values[0])
    rank = int((values > cutoff).sum().detach().cpu().item())
    condition = (values.new_tensor(torch.inf) if rank < min(matrix.shape)
                 else values[0] / values[-1])
    return rank, cutoff, condition


@dataclass(frozen=True)
class GaugeMap:  # MARK: GaugeMap
    r"""
    Represents a map between a TT bond and two TR bonds.

    A left gauge has shape ``(cyclic_rank, external_dim, local_rank)`` and a
    right gauge has shape ``(local_rank, external_dim, cyclic_rank)``. Both
    orientations expose the same matrix layout through :attr:`matrix`.
    :meth:`pinv` returns its directional dual. The transpose in that dual
    is not conjugated because tensor network bonds contract bilinearly.

    Parameters
    ----------
    core : torch.Tensor
        Three-dimensional floating or complex gauge tensor.
    orientation : {"left", "right"}
        Which external gauge is represented by ``core``.
    site : int, optional
        Destination site used in diagnostics.
    records : sequence of GaugeRecord
        Optional measurements from the transport that produced this map.
    metadata : mapping
        Optional transport diagnostics, copied when the map is constructed.

    Examples
    --------
    >>> gauge = tk.decompositions.GaugeMap(
    ...     torch.eye(2).reshape(1, 2, 2), 'left')
    >>> gauge.pinv().matrix.shape
    torch.Size([2, 2])
    """

    core: torch.Tensor  # Raw tensor with standard TR axes
    orientation: str  # Left or right interpretation of the TR axes
    site: Optional[int] = None  # Destination site used in diagnostics
    # Original map retained for dual-cancellation diagnostics
    _reference_matrix: Optional[torch.Tensor] = field(
        default=None, repr=False, compare=False)
    # Validated options for the directional pseudoinverse
    _pinv_kwargs: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False)

    # Measurements from the transport that produced this map
    records: Sequence[GaugeRecord] = ()
    # Transport diagnostics independent of the gauge tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.core, torch.Tensor):
            raise TypeError('`core` should be torch.Tensor type')
        if self.core.ndim != 3:
            raise ValueError('`core` should be a three-dimensional gauge')
        if not self.core.is_floating_point() and not self.core.is_complex():
            raise TypeError('`core` should have a floating or complex dtype')
        if any(dim < 1 for dim in self.core.shape):
            raise ValueError('`core` axes should have positive dimensions')
        if self.orientation not in ('left', 'right'):
            raise ValueError("`orientation` should be 'left' or 'right'")
        if self.site is not None:
            if isinstance(self.site, bool) or not isinstance(self.site, int):
                raise TypeError('`site` should be int type or None')
            if self.site < 0:
                raise ValueError('`site` should be non-negative')
        object.__setattr__(
            self, '_pinv_kwargs', _pinv_options(self._pinv_kwargs))
        if self._reference_matrix is not None:
            reference = self._reference_matrix
            if not isinstance(reference, torch.Tensor):
                raise TypeError('`_reference_matrix` should be torch.Tensor')
            if reference.shape != self.matrix.shape:
                raise ValueError('Reference and dual matrices should match')
            if (reference.device != self.core.device) or \
                    (reference.dtype != self.core.dtype):
                raise ValueError('Reference and dual should share runtime')

        if not all(isinstance(record, GaugeRecord) for record in self.records):
            raise TypeError('`records` should contain GaugeRecord objects')
        if not isinstance(self.metadata, Mapping):
            raise TypeError('`metadata` should be a mapping')
        object.__setattr__(self, 'records', tuple(self.records))
        object.__setattr__(self, 'metadata', dict(self.metadata))

    @property
    def matrix(self) -> torch.Tensor:
        """Returns shape ``(external_dim, cyclic_rank * local_rank)``."""
        return _matrix_from_core(self.core, self.orientation)

    @property
    def cyclic_rank(self) -> int:
        """Cyclic rank encoded by this orientation."""
        return self.core.shape[0] if self.orientation == 'left' \
            else self.core.shape[-1]

    @property
    def local_rank(self) -> int:
        """Adjacent TR rank encoded by this orientation."""
        return self.core.shape[-1] if self.orientation == 'left' \
            else self.core.shape[0]

    @property
    def is_dual(self) -> bool:
        """Whether this map was constructed as a directional dual."""
        return self._reference_matrix is not None

    def mirror(self) -> 'GaugeMap':
        """Swaps the two TR axes while preserving the matrix representation."""
        return GaugeMap(
            self.core.permute(2, 1, 0),
            'right' if self.orientation == 'left' else 'left',
            self.site, self._reference_matrix, self._pinv_kwargs)

    def pinv(self, **pinv_kwargs: Any) -> 'GaugeMap':
        """
        Constructs the directional dual with a Moore-Penrose pseudoinverse.

        Parameters
        ----------
        **pinv_kwargs
            ``atol``, ``rtol`` and ``hermitian`` forwarded to
            :func:`torch.linalg.pinv`. Tolerances must be numbers or real
            scalar tensors; batched gauge maps are not supported.

        Returns
        -------
        GaugeMap
            Dual with the same external and TR dimensions. Rank-deficient
            maps retain their support projector; no cancellation is enforced.
        """
        options = _pinv_options(pinv_kwargs)
        matrix = self.matrix
        dual = torch.linalg.pinv(matrix, **options).T
        return GaugeMap(
            _core_from_matrix(dual, self.orientation,
                              self.cyclic_rank, self.local_rank),
            self.orientation, self.site, matrix, options)

    def diagnostics(self) -> GaugeRecord:
        """Measures rank, condition and identity-cancellation error."""
        if not self.is_dual:
            return self.pinv().diagnostics()
        matrix = self._reference_matrix
        rank, cutoff, condition = _rank_statistics(matrix, self._pinv_kwargs)
        product = matrix.T @ self.matrix
        identity = torch.eye(
            product.shape[0], dtype=product.dtype, device=product.device)
        error = (product - identity).norm() / identity.norm()
        tolerance = torch.finfo(matrix.real.dtype).eps * max(matrix.shape)
        return GaugeRecord(
            orientation=self.orientation, shape=tuple(matrix.shape),
            numerical_rank=rank, cancellable_rank=matrix.shape[1],
            condition_number=condition, cancellation_error=error,
            projective=rank < matrix.shape[1], inverse_method='pinv',
            tolerance=tolerance, rank_tolerance=cutoff, site=self.site)

    def as_event(self, phase: str, level: int = 2) -> DecompositionEvent:
        """
        Creates an observer event from the measured gauge diagnostics.

        Parameters
        ----------
        phase : str
            Algorithm phase reported by the observer.
        level : int
            Verbosity level associated with the event.
        """
        if not isinstance(phase, str):
            raise TypeError('`phase` should be str type')
        record = self.diagnostics()
        return DecompositionEvent(
            name='gauge', phase=phase, level=level, site=record.site,
            values={
                'orientation': record.orientation, 'shape': record.shape,
                'rank': (record.numerical_rank, record.cancellable_rank),
                'condition_number': record.condition_number,
                'cancellation_error': record.cancellation_error,
                'projective': record.projective, 'inverse_method': 'pinv',
            })


@runtime_checkable
class GaugeRecursion(Protocol):  # MARK: GaugeRecursion
    """
    Defines directional gauge transport between neighboring local problems.

    Each method receives a completed ``opening``, the destination
    ``local_target`` and a ``recursion_context`` with the provider and source
    and destination sites. It returns the gauge fixed in the next opening.
    """

    def advance_left(self,
                     opening: LoopOpening,
                     local_target: Any,
                     recursion_context: Mapping[str, Any]
                     ) -> GaugeMap:
        """
        Builds the fixed right gauge for the next site to the left.

        Parameters
        ----------
        opening : LoopOpening
            Completed source opening with its learned cores and gauges.
        local_target : object
            Destination objective supplied by the target provider.
        recursion_context : mapping
            Provider and source ``from_sites`` and destination ``to_sites``.

        Returns
        -------
        GaugeMap
            Gauge fixed on the right of the destination and its diagnostics.
        """

    def advance_right(self,
                      opening: LoopOpening,
                      local_target: Any,
                      recursion_context: Mapping[str, Any]
                      ) -> GaugeMap:
        """
        Builds the fixed left gauge for the next site to the right.

        Parameters
        ----------
        opening : LoopOpening
            Completed source opening with its learned cores and gauges.
        local_target : object
            Destination objective supplied by the target provider.
        recursion_context : mapping
            Provider and source ``from_sites`` and destination ``to_sites``.

        Returns
        -------
        GaugeMap
            Gauge fixed on the left of the destination and its diagnostics.
        """


class OrthGaugeRecursion:  # MARK: OrthGaugeRecursion
    r"""
    Transports using the pseudoinverse of the learned outgoing gauge.

    The outgoing right gauge is dualized and mirrored into the next fixed
    left gauge. The leftward operation is its mirror. The induced support
    projector is orthogonal; deficient rank is allowed.

    Parameters
    ----------
    **pinv_kwargs
        ``atol``, ``rtol`` and ``hermitian`` forwarded to
        :func:`torch.linalg.pinv`. Tolerances must be numbers or real scalar
        tensors; batched gauge maps are not supported.

    Examples
    --------
    >>> recursion = tk.decompositions.OrthGaugeRecursion(rtol=1e-12)
    >>> isinstance(recursion, tk.decompositions.GaugeRecursion)
    True
    """

    def __init__(self, **pinv_kwargs: Any) -> None:
        self._pinv_kwargs = _pinv_options(pinv_kwargs)

    def _advance(self,
                 opening: LoopOpening,
                 recursion_context: Mapping[str, Any],
                 direction: str) -> GaugeMap:
        """Dualizes and mirrors the outgoing gauge in one ``direction``."""
        if not isinstance(opening, LoopOpening):
            raise TypeError('`opening` should be LoopOpening type')
        if not isinstance(recursion_context, Mapping):
            raise TypeError('`recursion_context` should be a mapping')
        to_sites = recursion_context.get('to_sites')
        if not isinstance(to_sites, tuple) or (len(to_sites) != 1):
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

        # This transport dualizes the learned gauge, not a TT-core unfolding.
        fixed_map = GaugeMap(
            outgoing, orientation, to_sites[0]).pinv(
                **self._pinv_kwargs).mirror()
        record = fixed_map.diagnostics()
        return GaugeMap(
            orientation='left' if direction == 'right' else 'right',
            core=fixed_map.core,
            records=(record,),
            metadata={
                'inverse_method': record.inverse_method,
                'projective': record.projective,
                'cancellation_error': record.cancellation_error,
            })

    def advance_left(self,
                     opening: LoopOpening,
                     local_target: Any,
                     recursion_context: Mapping[str, Any]
                     ) -> GaugeMap:
        """Builds the fixed right gauge for the next site to the left."""
        return self._advance(opening, recursion_context, 'left')

    def advance_right(self,
                      opening: LoopOpening,
                      local_target: Any,
                      recursion_context: Mapping[str, Any]
                      ) -> GaugeMap:
        """Builds the fixed left gauge for the next site to the right."""
        return self._advance(opening, recursion_context, 'right')


class PinvGaugeRecursion(OrthGaugeRecursion):  # MARK: PinvGaugeRecursion
    r"""
    Transports learned TR factors through the local TT-core pseudoinverse.

    For the directional TT unfolding :math:`M` and the factor :math:`X`
    formed by the incoming gauge and recovered TR core, the next gauge is
    :math:`E=M^+X`. An initial multi-site block is contracted and inverted
    as a whole. Components outside the column space of :math:`M` are allowed.

    Parameters
    ----------
    **pinv_kwargs
        ``atol``, ``rtol`` and ``hermitian`` forwarded to
        :func:`torch.linalg.pinv`. Tolerances must be numbers or real scalar
        tensors; batched gauge maps are not supported.

    Examples
    --------
    >>> recursion = tk.decompositions.PinvGaugeRecursion(rtol=1e-12)
    """

    def _solve_coordinates(self,
                           matrix: torch.Tensor,
                           basis: torch.Tensor,
                           orientation: str,
                           site: int) -> Tuple[torch.Tensor, GaugeRecord]:
        """Expresses ``basis`` in the columns of one TT-core unfolding."""
        coordinates = torch.linalg.pinv(matrix, **self._pinv_kwargs) @ basis
        return coordinates, self._coordinate_record(
            matrix, basis, coordinates, 'pinv', orientation, site)

    def _coordinate_record(self,
                           matrix: torch.Tensor,
                           basis: torch.Tensor,
                           coordinates: torch.Tensor,
                           method: str,
                           orientation: str,
                           site: int) -> GaugeRecord:
        """Measures projection error without rejecting rank deficiency."""
        options = self._pinv_kwargs if method == 'pinv' else {}
        rank, cutoff, condition = _rank_statistics(matrix, options)
        residual = (matrix @ coordinates - basis).norm()
        error = safe_divide(residual, basis.norm(), zero_value=residual)
        tolerance = torch.finfo(matrix.real.dtype).eps * max(matrix.shape)
        return GaugeRecord(
            orientation=orientation, shape=tuple(matrix.shape),
            numerical_rank=rank, cancellable_rank=matrix.shape[1],
            condition_number=condition, cancellation_error=error,
            projective=(rank < matrix.shape[1]) or bool(error > tolerance),
            inverse_method=method, tolerance=tolerance,
            rank_tolerance=cutoff, site=site)

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
                'TT-core recursion requires standardized provider cores')
        site = from_sites[0]
        core = (cores[site] if len(from_sites) == 1 else
                provider.local_target(from_sites, recursion_context))
        if isinstance(core, torch.Tensor) and (core.ndim > 3):
            core = core.reshape(core.shape[0], -1, core.shape[-1])
        if not isinstance(core, torch.Tensor) or (core.ndim != 3):
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
        if (tt_core.device != tr_core.device) or \
                (tt_core.dtype != tr_core.dtype):
            raise ValueError(
                'TT and opened TR cores should share dtype and device')

        # Use identical row coordinates for the TT reference and learned
        # factor.
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
                 direction: str) -> GaugeMap:
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

        return GaugeMap(
            orientation='left' if direction == 'right' else 'right',
            core=gauge,
            records=(record,),
            metadata={
                'inverse_method': record.inverse_method,
                'projective': record.projective,
                'projection_error': record.cancellation_error,
                'source_site': site,
            })


def _residual_directions(q: torch.Tensor, basis: torch.Tensor,
                         rank: Optional[int], rtol: float
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Finds significant learned directions outside an orthonormal range."""
    residual = basis - q @ (q.mH @ basis)
    residual = residual - q @ (q.mH @ residual)
    u, values, _ = torch.linalg.svd(residual, full_matrices=False)
    capacity = max(0, basis.shape[0] - q.shape[1])
    if rank is not None:
        capacity = min(capacity, rank)
    selected = min(capacity, int((values > basis.norm() * rtol).sum()))
    extra = u[:, :selected]
    if selected:
        extra = extra - q @ (q.mH @ extra)
        extra = torch.linalg.qr(extra, mode='reduced').Q
    return extra, residual


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

    Parameters
    ----------
    **pinv_kwargs
        Accepted for the common recursion interface. The directional adjoint
        and its numerical-rank diagnostics do not use these options. Optional
        TT2TR expansion adds orthogonal residual coordinates and zero reference
        rows without changing TR ranks.

    Examples
    --------
    >>> recursion = tk.decompositions.AdjointGaugeRecursion()
    """

    def _advance_qr(self,
                    opening: LoopOpening,
                    recursion_context: Mapping[str, Any],
                    direction: str,
                    enrich_rank: Optional[int] = 0,
                    enrich_rtol: float = 1e-12) -> GaugeMap:
        """Moves the canonical center while retaining residual coordinates."""
        provider = recursion_context['provider']
        shift_center = getattr(provider, 'shift_center', None)
        if not callable(shift_center):
            raise TypeError(
                'Adjoint QR transport requires a canonical TT target provider')
        matrix, basis, block = self._local_matrices(
            opening, recursion_context, direction)

        # Move the triangular factor into the unsolved reference core; the
        # isometry supplies the adjoint transport.
        q, factor = torch.linalg.qr(matrix, mode='reduced')

        extra, residual = _residual_directions(
            q, basis, enrich_rank, enrich_rtol)
        selected = extra.shape[1]
        old_dimension = q.shape[1]
        if selected:
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
        return GaugeMap(core=gauge,
                        orientation=('left' if direction == 'right'
                                     else 'right'),
                        metadata={
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
                 direction: str) -> GaugeMap:
        """Uses QR for ordinary targets and Schmidt adjoints for Vidal."""
        provider = recursion_context['provider']
        if callable(getattr(provider, 'shift_center', None)):
            return self._advance_qr(opening, recursion_context, direction)
        return super()._advance(opening, recursion_context, direction)

    def _transport_coordinates(self,
                               matrix: torch.Tensor,
                               basis: torch.Tensor,
                               direction: str,
                               site: int,
                               context: Mapping[str, Any]
                               ) -> Tuple[torch.Tensor, GaugeRecord]:
        """Projects and converts to the next target's Schmidt powers."""
        provider = context['provider']
        sites = context['from_sites']
        isometry, weighted = provider._transport_matrices(
            matrix, basis, sites, direction)
        center_coordinates = isometry.mH @ weighted
        record = self._coordinate_record(
            isometry, weighted, center_coordinates, 'adjoint', direction, site)
        coordinates = provider._transport_coordinates(
            center_coordinates, sites, direction)
        return coordinates, record


class _ExpandedGaugeRecursion:  # MARK: _ExpandedGaugeRecursion
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
                 enrich_rank: Optional[int],
                 enrich_rtol: float) -> None:
        if enrich_rank is not None:
            if isinstance(enrich_rank, bool) or not isinstance(
                    enrich_rank, int):
                raise TypeError('`enrich_rank` should be int type or None')
            if enrich_rank < 0:
                raise ValueError('`enrich_rank` should be non-negative')
        self.recursion = recursion
        self.enrich_rank = enrich_rank
        self.enrich_rtol = _validate_non_negative_float(
            enrich_rtol, 'enrich_rtol')
        if self.enrich_rtol is None:
            raise TypeError('`enrich_rtol` should be a non-negative float')

    def _advance(self,
                 opening: LoopOpening,
                 local_target: Any,
                 context: Mapping[str, Any],
                 direction: str) -> GaugeMap:
        """Preserves base coordinates and appends the orthogonal residual."""
        if isinstance(self.recursion, AdjointGaugeRecursion) and callable(
                getattr(context['provider'], 'shift_center', None)):
            return self.recursion._advance_qr(
                opening, context, direction, self.enrich_rank,
                self.enrich_rtol)
        advance = (self.recursion.advance_right if direction == 'right'
                   else self.recursion.advance_left)
        step = advance(opening, local_target, context)
        matrix, basis, block = PinvGaugeRecursion._local_matrices(
            opening, context, direction)
        if isinstance(self.recursion, AdjointGaugeRecursion):
            matrix, basis = context['provider']._transport_matrices(
                matrix, basis, context['from_sites'], direction)
        # An SVD identifies support without assigning arbitrary bases to null
        # reference columns. Its cutoff is numerical, not a TT truncation.
        q, values, _ = torch.linalg.svd(matrix, full_matrices=False)
        cutoff = torch.finfo(values.dtype).eps * max(matrix.shape) * values[0]
        q = q[:, values > cutoff]
        extra, residual = _residual_directions(
            q, basis, self.enrich_rank, self.enrich_rtol)
        selected = extra.shape[1]
        coordinates = extra.mH @ basis
        if direction == 'right':
            added_gauge = coordinates.reshape(
                selected, opening.left_gauge.shape[0], block.shape[-1])
        else:
            added_gauge = coordinates.reshape(
                selected, block.shape[0], opening.right_gauge.shape[-1])
        old_dimension = step.core.shape[1]
        gauge = torch.cat((step.core, added_gauge.permute(1, 0, 2)), dim=1)
        provider = context['provider']
        destination = context['to_sites'][0]
        before = provider.cores[destination]

        # Added equations have zero target rows, preserving the reference
        # tensor.
        provider.expand_coordinates(destination, selected, direction)
        after = provider.cores[destination]
        axis = 0 if direction == 'right' else 2
        original = after.narrow(axis, 0, before.shape[axis])
        zeros = after.narrow(axis, before.shape[axis], selected)
        preservation_error = safe_divide(
            torch.hypot((original - before).norm(), zeros.norm()),
            before.norm())
        identity = torch.eye(selected, dtype=extra.dtype, device=extra.device)
        diagnostics = {
            **step.metadata,
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
        return GaugeMap(gauge,
                        'left' if direction == 'right' else 'right',
                        records=step.records, metadata=diagnostics)

    def advance_right(self,
                      opening: LoopOpening,
                      local_target: Any,
                      recursion_context: Mapping[str, Any]
                      ) -> GaugeMap:
        """Expands the next left gauge in rightward transport."""
        return self._advance(opening, local_target, recursion_context, 'right')

    def advance_left(self,
                     opening: LoopOpening,
                     local_target: Any,
                     recursion_context: Mapping[str, Any]
                     ) -> GaugeMap:
        """Expands the next right gauge in leftward transport."""
        return self._advance(opening, local_target, recursion_context, 'left')


__all__ = [
    'GaugeMap',
    'GaugeRecursion',
    'OrthGaugeRecursion',
    'PinvGaugeRecursion',
    'AdjointGaugeRecursion',
    'ExperimentalWarning',
]
