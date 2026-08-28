"""Input-axis fitting strategies for recursive-sketching decompositions."""

from dataclasses import dataclass
from math import prod
from typing import (Any, Callable, Optional, Protocol, Sequence, Tuple, Union,
                    runtime_checkable)

import torch

from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.metrics import InputFitRecord
from tensorkrowch.decompositions.sketching.phi import PhiView


Embedding = Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]


def _phi_shape(phi_view: PhiView) -> Tuple[int, ...]:
    """Returns and validates the shape required for input-axis fitting."""
    shape = getattr(phi_view, 'shape', None)
    if shape is None:
        raise TypeError('`phi_view` should expose its complete `shape`')
    try:
        shape = tuple(shape)
    except TypeError as exc:
        raise TypeError('`phi_view.shape` should be a sequence of integers') \
            from exc
    if not shape or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
            for dim in shape):
        raise ValueError(
            '`phi_view.shape` should contain positive integers')
    return shape


def _normalize_axis(axis: int, shape: Sequence[int]) -> int:
    """Normalizes one possibly-negative Phi axis."""
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError('`axis` should be int type')
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError('`axis` is out of bounds for Phi')
    return axis


def _phi_device(phi_view: PhiView, domain: torch.Tensor) -> torch.device:
    """Infers a device for fixed-index batches without evaluating Phi."""
    source = getattr(phi_view, 'source', None)
    if source is not None and hasattr(source, 'device'):
        return source.device
    tensor = getattr(phi_view, 'tensor', None)
    if isinstance(tensor, torch.Tensor):
        return tensor.device
    return domain.device


def _unravel_fixed_ids(ids: torch.Tensor,
                       fixed_shape: Sequence[int]) -> torch.Tensor:
    """Converts row-major fiber ids to fixed-axis index tuples."""
    columns = []
    remainder = ids
    for dim in reversed(tuple(fixed_shape)):
        columns.append(remainder.remainder(dim))
        remainder = torch.div(remainder, dim, rounding_mode='floor')
    return torch.stack(tuple(reversed(columns)), dim=1)


def _collect_phi_target(
        phi_view: PhiView,
        axis: int,
        domain: torch.Tensor,
        fiber_batch_size: Optional[int]) -> Tuple[
            torch.Tensor, Tuple[int, ...], bool]:
    """Returns Phi as ``domain_size x n_fibers`` using one selected path."""
    if not isinstance(phi_view, PhiView):
        raise TypeError('`phi_view` should implement PhiView')
    if not isinstance(domain, torch.Tensor):
        raise TypeError('`domain` should be torch.Tensor type')
    if domain.ndim < 1 or domain.shape[0] < 1:
        raise ValueError('`domain` should have a non-empty values dimension')
    shape = _phi_shape(phi_view)
    axis = _normalize_axis(axis, shape)
    if shape[axis] != domain.shape[0]:
        raise ValueError(
            '`domain` should contain one value per selected Phi-axis entry')
    if fiber_batch_size is not None and (
            isinstance(fiber_batch_size, bool) or
            not isinstance(fiber_batch_size, int) or
            fiber_batch_size < 1):
        raise ValueError(
            '`fiber_batch_size` should be a positive integer or None')

    if fiber_batch_size is None:
        tensor = phi_view.materialize()
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(
                '`phi_view.materialize()` should preserve the declared shape')
        target = tensor.movedim(axis, 0).reshape(shape[axis], -1)
        return target, shape, False

    fixed_shape = shape[:axis] + shape[(axis + 1):]
    n_fibers = prod(fixed_shape) if fixed_shape else 1
    target = None
    device = _phi_device(phi_view, domain)
    for start in range(0, n_fibers, fiber_batch_size):
        stop = min(start + fiber_batch_size, n_fibers)
        if fixed_shape:
            ids = torch.arange(start, stop, device=device)
            fixed_indices = _unravel_fixed_ids(ids, fixed_shape)
        else:
            fixed_indices = None
        fibers = phi_view.fiber(axis, fixed_indices)
        if not isinstance(fibers, torch.Tensor):
            raise TypeError('`phi_view.fiber()` should return a torch.Tensor')
        expected = (stop - start, shape[axis]) if fixed_shape \
            else (shape[axis],)
        if tuple(fibers.shape) != expected:
            raise ValueError(
                '`phi_view.fiber()` returned an incompatible shape')
        fibers = fibers.reshape(stop - start, shape[axis]).transpose(0, 1)
        if target is None:
            target = fibers.new_empty((shape[axis], n_fibers))
        elif fibers.device != target.device or fibers.dtype != target.dtype:
            raise ValueError('All Phi fibers should share a device and dtype')
        target[:, start:stop] = fibers
    return target, shape, True


def _restore_fitted_axis(solution: torch.Tensor,
                         shape: Sequence[int],
                         axis: int) -> torch.Tensor:
    """Restores a solved ``input_dim x n_fibers`` table to Phi layout."""
    other_shape = tuple(shape[:axis]) + tuple(shape[(axis + 1):])
    return solution.reshape(solution.shape[0], *other_shape).movedim(0, axis)


@dataclass(frozen=True)
class FittedInputAxis:
    """Tensor obtained after replacing one sampled Phi axis by an input axis."""

    tensor: torch.Tensor
    axis: int
    domain_size: int
    input_dim: int
    record: Optional[InputFitRecord] = None

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if self.tensor.ndim < 1:
            raise ValueError('`tensor` should have at least one axis')
        if isinstance(self.axis, bool) or not isinstance(self.axis, int):
            raise TypeError('`axis` should be int type')
        if self.axis < 0 or self.axis >= self.tensor.ndim:
            raise ValueError('`axis` is out of bounds for `tensor`')
        for name in ('domain_size', 'input_dim'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            if value < 1:
                raise ValueError(f'`{name}` should be positive')
        if self.tensor.shape[self.axis] != self.input_dim:
            raise ValueError(
                '`tensor` size at `axis` should equal `input_dim`')
        if self.record is not None and \
                not isinstance(self.record, InputFitRecord):
            raise TypeError('`record` should be InputFitRecord type or None')


@runtime_checkable
class InputFitter(Protocol):
    """Strategy that replaces one sampled Phi axis by an input basis."""

    def required_queries(
            self,
            phi_view: PhiView,
            axis: int,
            domain: torch.Tensor,
            context: Any = None) -> Sequence[torch.Tensor]:
        """Declares additional Phi selections before evaluation-plan freeze."""

    def fit(
            self,
            phi_view: PhiView,
            axis: int,
            domain: torch.Tensor,
            context: Any = None,
            return_info: bool = False) -> FittedInputAxis:
        """Fits one Phi axis and optionally records fitting diagnostics."""


class FixedEmbeddingFitter:
    """Fits Phi values in a fixed finite-domain embedding by least squares."""

    def __init__(
            self,
            embedding: Embedding,
            solver: Optional[LeastSquaresSolver] = None,
            fiber_batch_size: Optional[int] = None) -> None:
        if not (isinstance(embedding, torch.Tensor) or callable(embedding)):
            raise TypeError('`embedding` should be a tensor or callable')
        if solver is not None and not isinstance(solver, LeastSquaresSolver):
            raise TypeError('`solver` should be LeastSquaresSolver type or None')
        if fiber_batch_size is not None and (
                isinstance(fiber_batch_size, bool) or
                not isinstance(fiber_batch_size, int) or
                fiber_batch_size < 1):
            raise ValueError(
                '`fiber_batch_size` should be a positive integer or None')
        self.embedding = embedding
        self.solver = LeastSquaresSolver() if solver is None else solver
        self.fiber_batch_size = fiber_batch_size

    def _embedding_matrix(self, domain: torch.Tensor) -> torch.Tensor:
        """Evaluates and validates the fixed embedding on one domain."""
        try:
            matrix = self.embedding(domain) if callable(self.embedding) \
                else self.embedding
        except Exception as exc:
            raise ValueError('`embedding` failed on `domain`') from exc
        if not isinstance(matrix, torch.Tensor):
            raise TypeError('`embedding` should produce a torch.Tensor')
        if matrix.ndim != 2 or matrix.shape[0] != domain.shape[0] or \
                matrix.shape[1] < 1:
            raise ValueError(
                '`embedding` should produce shape (domain_size, input_dim)')
        if not (matrix.is_floating_point() or matrix.is_complex()):
            raise TypeError('`embedding` should be floating or complex')
        if not torch.isfinite(matrix).all():
            raise ValueError('`embedding` should contain finite values')
        return matrix

    def required_queries(
            self,
            phi_view: PhiView,
            axis: int,
            domain: torch.Tensor,
            context: Any = None) -> Sequence[torch.Tensor]:
        """Fixed finite-domain fitting requires no additional Phi queries."""
        shape = _phi_shape(phi_view)
        axis = _normalize_axis(axis, shape)
        if not isinstance(domain, torch.Tensor):
            raise TypeError('`domain` should be torch.Tensor type')
        if domain.ndim < 1 or domain.shape[0] != shape[axis]:
            raise ValueError(
                '`domain` should match the selected Phi-axis size')
        return ()

    def fit(
            self,
            phi_view: PhiView,
            axis: int,
            domain: torch.Tensor,
            context: Any = None,
            return_info: bool = False) -> FittedInputAxis:
        """Solves all fibers together against the fixed embedding matrix."""
        if not isinstance(return_info, bool):
            raise TypeError('`return_info` should be bool type')
        target, shape, used_fibers = _collect_phi_target(
            phi_view, axis, domain, self.fiber_batch_size)
        axis = _normalize_axis(axis, shape)
        matrix = self._embedding_matrix(domain)
        dtype = torch.promote_types(matrix.dtype, target.dtype)
        matrix = matrix.to(device=target.device, dtype=dtype)
        target = target.to(dtype=dtype)
        solution, local_record = self.solver.solve(
            matrix,
            target,
            site=axis,
            return_record=return_info)
        record = None
        if return_info:
            singular_values = torch.linalg.svdvals(matrix)
            smallest = singular_values[-1]
            condition = torch.where(
                smallest > 0,
                singular_values[0] / smallest,
                torch.full_like(smallest, torch.inf))
            record = InputFitRecord(
                method='fixed_embedding',
                axis=axis,
                domain_size=matrix.shape[0],
                input_dim=matrix.shape[1],
                residual_absolute=local_record.residual_absolute,
                residual_relative=local_record.residual_relative,
                condition_number=condition,
                used_fibers=used_fibers,
                local_solve=local_record)
        tensor = _restore_fitted_axis(solution, shape, axis)
        return FittedInputAxis(
            tensor=tensor,
            axis=axis,
            domain_size=matrix.shape[0],
            input_dim=matrix.shape[1],
            record=record)


class BasisFitter:
    """Fits an integer-labelled Phi axis in the corresponding basis exactly."""

    def __init__(self,
                 input_dim: Optional[int] = None,
                 fiber_batch_size: Optional[int] = None) -> None:
        if input_dim is not None and (
                isinstance(input_dim, bool) or not isinstance(input_dim, int)
                or input_dim < 1):
            raise ValueError('`input_dim` should be a positive integer or None')
        if fiber_batch_size is not None and (
                isinstance(fiber_batch_size, bool) or
                not isinstance(fiber_batch_size, int) or
                fiber_batch_size < 1):
            raise ValueError(
                '`fiber_batch_size` should be a positive integer or None')
        self.input_dim = input_dim
        self.fiber_batch_size = fiber_batch_size

    @staticmethod
    def _labels(domain: torch.Tensor,
                input_dim: Optional[int]) -> Tuple[torch.Tensor, int]:
        """Validates basis labels and resolves the complete input dimension."""
        if not isinstance(domain, torch.Tensor):
            raise TypeError('`domain` should be torch.Tensor type')
        integer_dtypes = (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if domain.ndim != 1 or domain.shape[0] < 1 or \
                domain.dtype not in integer_dtypes:
            raise ValueError(
                '`domain` should be a non-empty integer vector of basis labels')
        labels = domain.to(torch.long)
        if torch.unique(labels).shape[0] != labels.shape[0]:
            raise ValueError('Basis labels in `domain` should be unique')
        resolved_dim = input_dim
        if resolved_dim is None:
            resolved_dim = int(labels.max().detach().cpu()) + 1
        if torch.any(labels < 0) or torch.any(labels >= resolved_dim):
            raise ValueError(
                'Basis labels in `domain` should lie inside `input_dim`')
        return labels, resolved_dim

    def required_queries(
            self,
            phi_view: PhiView,
            axis: int,
            domain: torch.Tensor,
            context: Any = None) -> Sequence[torch.Tensor]:
        """Basis selection requires no additional Phi queries."""
        shape = _phi_shape(phi_view)
        axis = _normalize_axis(axis, shape)
        labels, _ = self._labels(domain, self.input_dim)
        if labels.shape[0] != shape[axis]:
            raise ValueError(
                '`domain` should match the selected Phi-axis size')
        return ()

    def fit(
            self,
            phi_view: PhiView,
            axis: int,
            domain: torch.Tensor,
            context: Any = None,
            return_info: bool = False) -> FittedInputAxis:
        """Places sampled values at their basis labels without solving."""
        if not isinstance(return_info, bool):
            raise TypeError('`return_info` should be bool type')
        labels, input_dim = self._labels(domain, self.input_dim)
        target, shape, used_fibers = _collect_phi_target(
            phi_view, axis, domain, self.fiber_batch_size)
        axis = _normalize_axis(axis, shape)
        solution = target.new_zeros((input_dim, target.shape[1]))
        solution.index_copy_(0, labels.to(solution.device), target)
        record = InputFitRecord(
            method='basis',
            axis=axis,
            domain_size=domain.shape[0],
            input_dim=input_dim,
            residual_absolute=0.,
            residual_relative=0.,
            condition_number=1.,
            used_fibers=used_fibers) if return_info else None
        tensor = _restore_fitted_axis(solution, shape, axis)
        return FittedInputAxis(
            tensor=tensor,
            axis=axis,
            domain_size=domain.shape[0],
            input_dim=input_dim,
            record=record)


__all__ = [
    'InputFitter',
    'FittedInputAxis',
    'FixedEmbeddingFitter',
    'BasisFitter',
]
