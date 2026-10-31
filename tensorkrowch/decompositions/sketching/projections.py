"""
This script contains:

    Classes:
        * ProjectedRange
        * RangeProjector
        * IdentityRangeProjector
        * RandomizedRangeProjector

    Functions:
        * _normalize_matrix_axis
        * _projection_error
        * _validate_optional_rank
"""

from contextlib import nullcontext
from dataclasses import dataclass
from math import prod
from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch

from tensorkrowch.utils import safe_divide, stable_norm

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions._utils import _normalize_dims
from tensorkrowch.decompositions.metrics import RangeProjectionRecord


def _normalize_matrix_axis(matrix: torch.Tensor,
                           axis: int) -> Tuple[torch.Tensor, int]:
    """Flattens every non-projected ``axis`` into ``matrix`` rows."""
    if not isinstance(matrix, torch.Tensor):
        raise TypeError('`matrix` should be torch.Tensor type')
    if matrix.ndim < 2:
        raise ValueError('`matrix` should have at least two dimensions')
    if not (matrix.is_floating_point() or matrix.is_complex()):
        raise TypeError('`matrix` should be floating or complex')
    if not torch.isfinite(matrix).all():
        raise ValueError('`matrix` should contain finite values')
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError('`axis` should be int type')
    if axis < 0:
        axis += matrix.ndim
    if (axis < 0) or (axis >= matrix.ndim):
        raise ValueError('`axis` is out of bounds for `matrix`')
    if (matrix.ndim == 2) and (axis == 1):
        return matrix, axis
    moved = matrix.movedim(axis, -1)
    return moved.reshape(-1, matrix.shape[axis]), axis


def _projection_error(matrix: torch.Tensor,
                      basis: Optional[torch.Tensor],
                      small_matrix: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes absolute and zero-safe relative range-projection errors."""
    approximation = small_matrix if basis is None \
        else basis @ small_matrix
    absolute = stable_norm(matrix - approximation)
    denominator = stable_norm(matrix)
    relative = safe_divide(
        absolute, denominator,
        zero_value=torch.where(absolute == 0, 0, torch.inf))
    return absolute, relative


def _validate_optional_rank(rank: Optional[int]) -> None:
    """Validates an optional ``rank`` upper bound."""
    if rank is not None:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError('`rank` should be int type or None')
        if rank < 1:
            raise ValueError('`rank` should be positive')


@dataclass(frozen=True)
class ProjectedRange:  # MARK: ProjectedRange
    """
    Projected matrix, its lifting map and optional approximation diagnostics.

    ``small_matrix`` represents the original tensor with ``axis`` moved to
    the last position and all remaining axes flattened into rows. ``basis``
    maps projected rows to those original rows; ``None`` denotes the identity.
    """

    small_matrix: torch.Tensor
    basis: Optional[torch.Tensor]
    original_shape: Sequence[int]
    axis: int
    record: Optional[RangeProjectionRecord] = None

    def __post_init__(self) -> None:
        if not isinstance(self.small_matrix, torch.Tensor) or \
                (self.small_matrix.ndim != 2):
            raise TypeError('`small_matrix` should be a matrix')
        original_shape = _normalize_dims(self.original_shape, 'original_shape')
        if len(original_shape) < 2:
            raise ValueError(
                '`original_shape` should contain at least two positive sizes')
        axis = self.axis
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError('`axis` should be int type')
        if axis < 0:
            axis += len(original_shape)
        if (axis < 0) or (axis >= len(original_shape)):
            raise ValueError('`axis` is out of bounds for `original_shape`')
        n_rows = prod(original_shape[:axis] + original_shape[(axis + 1):])
        n_columns = original_shape[axis]
        if self.small_matrix.shape[1] != n_columns:
            raise ValueError(
                '`small_matrix` columns should match the projected axis')
        if self.basis is None:
            if self.small_matrix.shape[0] != n_rows:
                raise ValueError(
                    'Identity `small_matrix` rows should match the input')
        else:
            if not isinstance(self.basis, torch.Tensor) or \
                    (self.basis.ndim != 2):
                raise TypeError('`basis` should be a matrix or None')
            if self.basis.shape != (n_rows, self.small_matrix.shape[0]):
                raise ValueError(
                    '`basis` should map small rows to flattened input rows')
            if (self.basis.device != self.small_matrix.device) or \
                    (self.basis.dtype != self.small_matrix.dtype):
                raise ValueError(
                    '`basis` and `small_matrix` should share device and dtype')
        if (self.record is not None) and \
                not isinstance(self.record, RangeProjectionRecord):
            raise TypeError(
                '`record` should be RangeProjectionRecord type or None')
        object.__setattr__(self, 'original_shape', original_shape)
        object.__setattr__(self, 'axis', axis)

    @property
    def range_dim(self) -> int:
        """Dimension of the represented left range."""
        return self.small_matrix.shape[0]

    def lift_left(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Lifts left singular vectors to the original flattened row space.

        Parameters
        ----------
        vectors : torch.Tensor
            Matrix with shape ``(range_dim, rank)``.

        Returns
        -------
        torch.Tensor
            Matrix with shape ``(original_rows, rank)``.
        """
        if not isinstance(vectors, torch.Tensor) or (vectors.ndim != 2):
            raise TypeError('`vectors` should be a matrix')
        if vectors.shape[0] != self.range_dim:
            raise ValueError('`vectors` rows should match `range_dim`')
        if (vectors.device != self.small_matrix.device) or \
                (vectors.dtype != self.small_matrix.dtype):
            raise ValueError(
                '`vectors` and `small_matrix` should share device and dtype')
        return vectors if self.basis is None else self.basis @ vectors

    def restore_left(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Lifts vectors and restores the input layout with a new axis size.

        Parameters
        ----------
        vectors : torch.Tensor
            Matrix with shape ``(range_dim, rank)``.

        Returns
        -------
        torch.Tensor
            Tensor with ``original_shape``, replacing the size of ``axis``
            by ``rank``.
        """
        lifted = self.lift_left(vectors)
        other_shape = self.original_shape[:self.axis] + \
            self.original_shape[(self.axis + 1):]
        restored = lifted.reshape(*other_shape, vectors.shape[1])
        return restored.movedim(-1, self.axis)

    def reconstruct(self) -> torch.Tensor:
        """Reconstructs the projected approximation in the original layout."""
        if (self.basis is None) and (len(self.original_shape) == 2) and \
                (self.axis == 1):
            return self.small_matrix
        matrix = self.small_matrix if self.basis is None \
            else self.basis @ self.small_matrix
        other_shape = self.original_shape[:self.axis] + \
            self.original_shape[(self.axis + 1):]
        tensor = matrix.reshape(*other_shape, self.original_shape[self.axis])
        return tensor.movedim(-1, self.axis)


@runtime_checkable
class RangeProjector(Protocol):  # MARK: RangeProjector
    """Strategy producing a small matrix and a left-vector lifting map."""

    def project(self,
                matrix: torch.Tensor,
                rank: Optional[int] = None,
                generator: Optional[torch.Generator] = None,
                axis: int = -1,
                return_info: bool = False) -> ProjectedRange:
        """
        Projects a tensor range before an SVD.

        Parameters
        ----------
        matrix : torch.Tensor
            Floating-point or complex tensor with at least two axes.
        rank : int, optional
            Rank cap used as the default projection dimension. This method
            does not truncate singular values.
        generator : torch.Generator, optional
            Random generator for the Gaussian projection.
        axis : int
            Axis kept as matrix columns; the remaining axes become rows.
            Default is ``-1``.
        return_info : bool
            Whether to measure approximation errors and elapsed time.
            Default is ``False``.

        Returns
        -------
        ProjectedRange
            Matrix for the SVD, lifting map and optional diagnostics.
        """


class IdentityRangeProjector:  # MARK: IdentityRangeProjector
    """
    Uses the exact matrix without a random projection.

    This strategy gives the same numerical input to the SVD as
    ``range_projector=None``. It is useful when supplying an explicit
    :class:`RangeProjector` object.
    """

    def project(self,
                matrix: torch.Tensor,
                rank: Optional[int] = None,
                generator: Optional[torch.Generator] = None,
                axis: int = -1,
                return_info: bool = False) -> ProjectedRange:
        """
        Returns the exact matrix without a random projection.

        Parameters
        ----------
        matrix : torch.Tensor
            Floating-point or complex tensor with at least two axes.
        rank : int, optional
            Rank cap used as the default projection dimension. This method
            does not truncate singular values.
        generator : torch.Generator, optional
            Random generator for the Gaussian projection.
        axis : int
            Axis kept as matrix columns; the remaining axes become rows.
            Default is ``-1``.
        return_info : bool
            Whether to measure approximation errors and elapsed time.
            Default is ``False``.

        Returns
        -------
        ProjectedRange
            Matrix for the SVD, lifting map and optional diagnostics.
        """
        _validate_optional_rank(rank)
        if (generator is not None) and not isinstance(
                generator, torch.Generator):
            raise TypeError(
                '`generator` should be torch.Generator type or None')
        if not isinstance(return_info, bool):
            raise TypeError('`return_info` should be bool type')
        matrix_view, normalized_axis = _normalize_matrix_axis(matrix, axis)
        timer_context = _RuntimePolicy.from_tensor(
            matrix_view, out_device=None).timer() \
            if return_info else nullcontext()
        with timer_context as timer:
            small_matrix = matrix_view
        record = None
        if return_info:
            absolute, relative = _projection_error(
                matrix_view, None, matrix_view)
            record = RangeProjectionRecord(
                method='identity',
                input_shape=tuple(matrix_view.shape),
                axis=normalized_axis,
                requested_dim=rank,
                projection_dim=matrix_view.shape[1],
                range_dim=matrix_view.shape[0],
                error_absolute=absolute,
                error_relative=relative,
                elapsed=timer.elapsed)
        return ProjectedRange(
            small_matrix=small_matrix,
            basis=None,
            original_shape=tuple(matrix.shape),
            axis=normalized_axis,
            record=record)


class RandomizedRangeProjector:  # MARK: RandomizedRangeProjector
    """
    Approximates a matrix range using Gaussian projection and reduced QR.

    For a flattened matrix :math:`M`, the method forms an orthonormal basis
    :math:`Q` from a Gaussian projection and returns :math:`Q^H M`. Singular
    vectors of the smaller matrix are lifted by multiplication with
    :math:`Q`. Increasing the projection dimension or applying power
    iterations can improve the approximation at additional cost.

    Parameters
    ----------
    projection_dim : int, optional
        Number of Gaussian vectors before oversampling. Defaults to the rank
        cap supplied to :meth:`project`, or all columns if no cap is supplied.
    projection_oversampling : int
        Extra Gaussian vectors beyond ``projection_dim``. Default is ``0``.
        The total is capped at the number of matrix columns.
    n_power_iter : int
        Stabilized power iterations, each alternating multiplication by the
        matrix adjoint and the matrix with reduced QR. Default is ``0``.

    Examples
    --------
    >>> projector = tk.decompositions.RandomizedRangeProjector(
    ...     projection_dim=2, projection_oversampling=1)
    >>> projection = projector.project(torch.eye(3))
    >>> torch.allclose(projection.reconstruct(), torch.eye(3), atol=1e-5)
    True
    """

    def __init__(self,
                 projection_dim: Optional[int] = None,
                 projection_oversampling: int = 0,
                 n_power_iter: int = 0) -> None:
        if projection_dim is not None:
            if isinstance(projection_dim, bool) or \
                    not isinstance(projection_dim, int):
                raise TypeError(
                    '`projection_dim` should be int type or None')
            if projection_dim < 1:
                raise ValueError('`projection_dim` should be positive')
        for name, value in (
                ('projection_oversampling', projection_oversampling),
                ('n_power_iter', n_power_iter)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            if value < 0:
                raise ValueError(f'`{name}` should be non-negative')
        self.projection_dim = projection_dim
        self.projection_oversampling = projection_oversampling
        self.n_power_iter = n_power_iter

    def _dimensions(self,
                    n_columns: int,
                    rank: Optional[int]) -> Tuple[Optional[int], int]:
        """Resolves requested and oversampled Gaussian dimensions."""
        requested = self.projection_dim \
            if self.projection_dim is not None else rank
        base_dim = n_columns if requested is None \
            else min(requested, n_columns)
        projection_dim = min(
            n_columns, base_dim + self.projection_oversampling)
        return requested, projection_dim

    @staticmethod
    def _omega(matrix: torch.Tensor,
               projection_dim: int,
               generator: Optional[torch.Generator]) -> torch.Tensor:
        """Draws a real/complex Gaussian map on the ``generator``'s device."""
        random_device = matrix.device
        if generator is not None:
            random_device = torch.device(
                getattr(generator, 'device', torch.device('cpu')))
        omega = torch.randn(
            matrix.shape[1],
            projection_dim,
            device=random_device,
            dtype=matrix.dtype,
            generator=generator)
        return omega.to(matrix.device)

    def _range_find(self,
                    matrix: torch.Tensor,
                    projection_dim: int,
                    generator: Optional[torch.Generator]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs stabilized subspace iteration and forms ``Q^H A``."""
        omega = self._omega(matrix, projection_dim, generator)
        basis = torch.linalg.qr(matrix @ omega, mode='reduced').Q
        for _ in range(self.n_power_iter):
            co_basis = torch.linalg.qr(
                matrix.mH @ basis, mode='reduced').Q
            basis = torch.linalg.qr(
                matrix @ co_basis, mode='reduced').Q
        small_matrix = basis.mH @ matrix
        return basis, small_matrix

    def project(self,
                matrix: torch.Tensor,
                rank: Optional[int] = None,
                generator: Optional[torch.Generator] = None,
                axis: int = -1,
                return_info: bool = False) -> ProjectedRange:
        """
        Builds a Gaussian range basis and the projected matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            Floating-point or complex tensor with at least two axes.
        rank : int, optional
            Rank cap used as the default projection dimension. This method
            does not truncate singular values.
        generator : torch.Generator, optional
            Random generator for the Gaussian projection.
        axis : int
            Axis kept as matrix columns; the remaining axes become rows.
            Default is ``-1``.
        return_info : bool
            Whether to measure approximation errors and elapsed time.
            Default is ``False``.

        Returns
        -------
        ProjectedRange
            Matrix for the SVD, lifting map and optional diagnostics.
        """
        _validate_optional_rank(rank)
        if (generator is not None) and not isinstance(
                generator, torch.Generator):
            raise TypeError(
                '`generator` should be torch.Generator type or None')
        if not isinstance(return_info, bool):
            raise TypeError('`return_info` should be bool type')
        matrix_view, normalized_axis = _normalize_matrix_axis(matrix, axis)
        requested_dim, projection_dim = self._dimensions(
            matrix_view.shape[1], rank)
        timer_context = _RuntimePolicy.from_tensor(
            matrix_view, out_device=None).timer() \
            if return_info else nullcontext()
        with timer_context as timer:
            basis, small_matrix = self._range_find(
                matrix_view, projection_dim, generator)
        record = None
        if return_info:
            absolute, relative = _projection_error(
                matrix_view, basis, small_matrix)
            record = RangeProjectionRecord(
                method='randomized',
                input_shape=tuple(matrix_view.shape),
                axis=normalized_axis,
                requested_dim=requested_dim,
                projection_dim=projection_dim,
                range_dim=small_matrix.shape[0],
                oversampling=self.projection_oversampling,
                n_power_iter=self.n_power_iter,
                error_absolute=absolute,
                error_relative=relative,
                elapsed=timer.elapsed)
        return ProjectedRange(
            small_matrix=small_matrix,
            basis=basis,
            original_shape=tuple(matrix.shape),
            axis=normalized_axis,
            record=record)


__all__ = [
    'RangeProjector',
    'IdentityRangeProjector',
    'RandomizedRangeProjector',
]
