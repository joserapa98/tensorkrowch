"""Input-axis fitting strategies for recursive-sketching decompositions."""

from dataclasses import dataclass, field, replace
from math import prod
from typing import (Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple,
                    Union, runtime_checkable)

import torch

from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.metrics import (InputFitRecord,
                                                 TruncationRecord)
from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.sketching.phi import PhiView
from tensorkrowch.decompositions.sketching.quantization import (
    CoordinateMap,
    QuantizedLayout,
    QuantizedSourceAdapter,
    UniformCoordinateMap,
)
from tensorkrowch.utils import truncated_svd


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
    model: Optional[torch.nn.Module] = None
    model_state: Optional[Mapping[str, torch.Tensor]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    factor: Optional[TTDecomposition] = None
    reduced_tensor: Optional[torch.Tensor] = None
    truncation: Optional[TruncationRecord] = None

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
        if self.model is not None and not isinstance(
                self.model, torch.nn.Module):
            raise TypeError('`model` should be torch.nn.Module type or None')
        if self.model_state is not None:
            if not isinstance(self.model_state, Mapping) or not all(
                    isinstance(name, str) and isinstance(value, torch.Tensor)
                    for name, value in self.model_state.items()):
                raise TypeError(
                    '`model_state` should map strings to tensors or be None')
            object.__setattr__(self, 'model_state', dict(self.model_state))
        if not isinstance(self.metadata, Mapping):
            raise TypeError('`metadata` should be a mapping')
        object.__setattr__(self, 'metadata', dict(self.metadata))
        if self.factor is not None and not isinstance(
                self.factor, TTDecomposition):
            raise TypeError('`factor` should be TTDecomposition type or None')
        if self.reduced_tensor is not None:
            if not isinstance(self.reduced_tensor, torch.Tensor):
                raise TypeError(
                    '`reduced_tensor` should be torch.Tensor type or None')
            if self.reduced_tensor.ndim != self.tensor.ndim:
                raise ValueError(
                    '`reduced_tensor` should preserve the fitted tensor order')
            if any(
                    reduced != original
                    for item, (reduced, original) in enumerate(zip(
                        self.reduced_tensor.shape, self.tensor.shape))
                    if item != self.axis):
                raise ValueError(
                    '`reduced_tensor` should only replace the fitted axis')
        if self.truncation is not None and not isinstance(
                self.truncation, TruncationRecord):
            raise TypeError(
                '`truncation` should be TruncationRecord type or None')


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


class TrainableEmbeddingFitter:
    """Fits a Phi axis with a locally trained embedding model.

    ``model(domain)`` must return a matrix with shape
    ``(domain_size, input_dim)``. Training jointly optimizes the model and one
    temporary coefficient table against functional Phi fibers. A final shared
    least-squares solve removes optimizer error from the returned coefficients.
    Gradients and random state are confined to this fitter; the surrounding
    decomposition remains an ordinary numerical routine.

    The trained model and a detached CPU state dictionary are attached to
    :class:`FittedInputAxis` only when ``return_info=True``. Otherwise the
    result contains only the fitted tensor and the model remains accessible as
    ``fitter.model``.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            input_dim: Optional[int] = None,
            optimizer_factory: Optional[Callable] = None,
            optimizer_kwargs: Optional[Mapping[str, Any]] = None,
            solver: Optional[LeastSquaresSolver] = None,
            max_steps: int = 500,
            tolerance: float = 1e-6,
            patience: int = 50,
            fiber_batch_size: Optional[int] = 64,
            seed: int = 0) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError('`model` should be torch.nn.Module type')
        if input_dim is not None and (
                isinstance(input_dim, bool) or not isinstance(input_dim, int)):
            raise TypeError('`input_dim` should be int type or None')
        if input_dim is not None and input_dim < 1:
            raise ValueError('`input_dim` should be positive')
        if optimizer_factory is not None and not callable(optimizer_factory):
            raise TypeError('`optimizer_factory` should be callable or None')
        if optimizer_kwargs is not None and not isinstance(
                optimizer_kwargs, Mapping):
            raise TypeError('`optimizer_kwargs` should be a mapping or None')
        if solver is not None and not isinstance(solver, LeastSquaresSolver):
            raise TypeError('`solver` should be LeastSquaresSolver type or None')
        for name, value, minimum in (
                ('max_steps', max_steps, 1),
                ('patience', patience, 1),
                ('seed', seed, 0)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            if value < minimum:
                qualifier = 'positive' if minimum else 'non-negative'
                raise ValueError(f'`{name}` should be {qualifier}')
        if isinstance(tolerance, bool) or not isinstance(
                tolerance, (int, float)):
            raise TypeError('`tolerance` should be a real scalar')
        if tolerance < 0:
            raise ValueError('`tolerance` should be non-negative')
        if fiber_batch_size is not None and (
                isinstance(fiber_batch_size, bool) or
                not isinstance(fiber_batch_size, int)):
            raise TypeError(
                '`fiber_batch_size` should be int type or None')
        if fiber_batch_size is not None and fiber_batch_size < 1:
            raise ValueError('`fiber_batch_size` should be positive')

        self.model = model
        self.input_dim = input_dim
        self.optimizer_factory = torch.optim.Adam \
            if optimizer_factory is None else optimizer_factory
        self.optimizer_kwargs = {'lr': 1e-2} \
            if optimizer_kwargs is None else dict(optimizer_kwargs)
        self.solver = LeastSquaresSolver() if solver is None else solver
        self.max_steps = max_steps
        self.tolerance = float(tolerance)
        self.patience = patience
        self.fiber_batch_size = fiber_batch_size
        self.seed = seed

    def required_queries(
            self,
            phi_view: PhiView,
            axis: int,
            domain: torch.Tensor,
            context: Any = None) -> Sequence[torch.Tensor]:
        """Declares no points beyond the complete selected training domain."""
        shape = _phi_shape(phi_view)
        axis = _normalize_axis(axis, shape)
        if not isinstance(domain, torch.Tensor):
            raise TypeError('`domain` should be torch.Tensor type')
        if domain.ndim < 1 or domain.shape[0] != shape[axis]:
            raise ValueError(
                '`domain` should match the selected Phi-axis size')
        return ()

    def _embedding_matrix(self,
                          domain: torch.Tensor,
                          device: torch.device) -> torch.Tensor:
        """Evaluates and validates the current trainable embedding."""
        matrix = self.model(domain.to(device=device))
        if not isinstance(matrix, torch.Tensor):
            raise TypeError('`model` should return a torch.Tensor')
        if matrix.ndim != 2 or matrix.shape[0] != domain.shape[0] or \
                matrix.shape[1] < 1:
            raise ValueError(
                '`model` should return shape (domain_size, input_dim)')
        if self.input_dim is not None and matrix.shape[1] != self.input_dim:
            raise ValueError('`model` output does not match `input_dim`')
        if not (matrix.is_floating_point() or matrix.is_complex()):
            raise TypeError('`model` output should be floating or complex')
        if not torch.isfinite(matrix).all():
            raise ValueError('`model` output should contain finite values')
        return matrix

    def fit(
            self,
            phi_view: PhiView,
            axis: int,
            domain: torch.Tensor,
            context: Any = None,
            return_info: bool = False) -> FittedInputAxis:
        """Trains the embedding on batched Phi fibers and returns coefficients."""
        if not isinstance(return_info, bool):
            raise TypeError('`return_info` should be bool type')
        target, shape, used_fibers = _collect_phi_target(
            phi_view, axis, domain, self.fiber_batch_size)
        axis = _normalize_axis(axis, shape)
        parameters = [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError(
                '`model` should expose at least one trainable parameter')
        model_device = parameters[0].device
        if any(parameter.device != model_device for parameter in parameters):
            raise ValueError('All model parameters should share a device')
        if model_device != target.device:
            self.model.to(device=target.device)
            model_device = target.device

        cuda_devices = [model_device] if model_device.type == 'cuda' else []
        best_loss = float('inf')
        stale_steps = 0
        converged = False
        step = 0
        with torch.random.fork_rng(devices=cuda_devices), torch.enable_grad():
            torch.manual_seed(self.seed)
            matrix = self._embedding_matrix(domain, model_device)
            input_dim = matrix.shape[1]
            dtype = torch.promote_types(matrix.dtype, target.dtype)
            coefficient = torch.nn.Parameter(torch.randn(
                input_dim,
                target.shape[1],
                device=model_device,
                dtype=dtype) / max(input_dim, 1) ** 0.5)
            optimizer = self.optimizer_factory(
                [*parameters, coefficient], **self.optimizer_kwargs)
            denominator = torch.linalg.vector_norm(
                target.to(device=model_device, dtype=dtype)).clamp_min(
                    torch.finfo(target.real.dtype).tiny)
            for step in range(1, self.max_steps + 1):
                optimizer.zero_grad(set_to_none=True)
                matrix = self._embedding_matrix(domain, model_device).to(
                    dtype=dtype)
                promoted_target = target.to(device=model_device, dtype=dtype)
                residual = matrix @ coefficient - promoted_target
                loss = residual.abs().square().mean()
                loss.backward()
                optimizer.step()

                relative = torch.linalg.vector_norm(residual) / denominator
                loss_value = float(loss.detach().cpu())
                relative_value = float(relative.detach().cpu())
                if relative_value <= self.tolerance:
                    converged = True
                    break
                improvement = best_loss - loss_value
                threshold = max(abs(best_loss), 1.0) * 1e-12
                if best_loss == float('inf') or improvement > threshold:
                    best_loss = loss_value
                    stale_steps = 0
                else:
                    stale_steps += 1
                    if stale_steps >= self.patience:
                        break

        with torch.no_grad():
            matrix = self._embedding_matrix(domain, model_device)
            dtype = torch.promote_types(matrix.dtype, target.dtype)
            matrix = matrix.to(dtype=dtype)
            target = target.to(device=model_device, dtype=dtype)
            solution, local_record = self.solver.solve(
                matrix,
                target,
                site=axis,
                return_record=return_info)
            prediction = matrix @ solution
            absolute = torch.linalg.vector_norm(prediction - target)
            denominator = torch.linalg.vector_norm(target)
            if denominator > 0:
                relative = absolute / denominator
            elif absolute == 0:
                relative = torch.zeros_like(absolute)
            else:
                relative = torch.full_like(absolute, torch.inf)

            record = None
            model_state = None
            model = None
            if return_info:
                singular_values = torch.linalg.svdvals(matrix)
                smallest = singular_values[-1]
                condition = torch.where(
                    smallest > 0,
                    singular_values[0] / smallest,
                    torch.full_like(smallest, torch.inf))
                record = InputFitRecord(
                    method='trainable_embedding',
                    axis=axis,
                    domain_size=matrix.shape[0],
                    input_dim=matrix.shape[1],
                    residual_absolute=absolute,
                    residual_relative=relative,
                    condition_number=condition,
                    used_fibers=used_fibers,
                    local_solve=local_record)
                model = self.model
                model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model.state_dict().items()
                }
        tensor = _restore_fitted_axis(solution, shape, axis)
        return FittedInputAxis(
            tensor=tensor,
            axis=axis,
            domain_size=domain.shape[0],
            input_dim=matrix.shape[1],
            record=record,
            model=model,
            model_state=model_state,
            metadata={
                'steps': step,
                'converged': converged,
                'final_relative_residual': float(relative.detach().cpu()),
            })


class QTTInputFitter:
    """Represents one functional Phi input axis by a local QTT factor.

    The fitter creates an independent one-variable QTT-RSS problem. Every
    remaining Phi axis is treated as a tensor-output axis and placed
    consecutively after all digits. The returned :class:`FittedInputAxis`
    contains both a dense-grid axis, which preserves the ordinary
    ``InputFitter`` contract, and ``factor``, the lightweight local TT used by
    native QTT-Tucker assembly. With ``materialize_tensor=False``, the dense
    axis is not reconstructed: ``tensor`` and ``reduced_tensor`` contain only
    the small connector axis required by the upper decomposition.

    The digit block is orthogonalized with a QR sweep before an SVD separates
    it from the remaining Phi axes. Consequently, the native hierarchical
    path applies the SVD only to the reduced environment matrix and never
    densifies the complete local Phi unless metrics are requested.

    The local problem owns its generator, observer-free execution and source
    sessions. It never extends a frozen outer evaluation plan. One fitter can
    be configured per original variable to use different base, level, map and
    domain.
    """

    requires_functional_phi = True

    def __init__(
            self,
            base: int = 2,
            level: int = 4,
            *,
            digit_order: str = 'coarse_to_fine',
            coordinate_map: Optional[CoordinateMap] = None,
            domain=None,
            rank: Optional[int] = None,
            connector_rank: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            batch_size: int = 64,
            seed: int = 0,
            materialize_tensor: bool = True) -> None:
        self.layout = QuantizedLayout(
            n_variables=1,
            base=base,
            level=level,
            digit_order=digit_order)
        if coordinate_map is None:
            coordinate_map = UniformCoordinateMap()
        if not isinstance(coordinate_map, CoordinateMap):
            raise TypeError('`coordinate_map` should implement CoordinateMap')
        if rank is not None and (
                isinstance(rank, bool) or not isinstance(rank, int)):
            raise TypeError('`rank` should be int type or None')
        if rank is not None and rank < 1:
            raise ValueError('`rank` should be positive')
        if connector_rank is not None and (
                isinstance(connector_rank, bool) or
                not isinstance(connector_rank, int)):
            raise TypeError('`connector_rank` should be int type or None')
        if connector_rank is not None and connector_rank < 1:
            raise ValueError('`connector_rank` should be positive')
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError('`batch_size` should be int type')
        if batch_size < 1:
            raise ValueError('`batch_size` should be positive')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError('`seed` should be int type')
        if seed < 0:
            raise ValueError('`seed` should be non-negative')
        if not isinstance(materialize_tensor, bool):
            raise TypeError('`materialize_tensor` should be bool type')
        self.coordinate_map = coordinate_map
        self.domain = domain
        self.rank = rank
        self.connector_rank = connector_rank
        self.cutoff = cutoff
        self.atol = atol
        self.rtol = rtol
        self.cum_percentage = cum_percentage
        self.batch_size = batch_size
        self.seed = seed
        self.materialize_tensor = materialize_tensor

    def required_queries(
            self,
            phi_view: PhiView,
            axis: int,
            domain: torch.Tensor,
            context: Any = None) -> Sequence[torch.Tensor]:
        """Declares an independent local session instead of outer-plan rows."""
        shape = _phi_shape(phi_view)
        _normalize_axis(axis, shape)
        if not callable(getattr(phi_view, 'with_axis_values', None)):
            raise TypeError(
                'QTTInputFitter requires a functional PhiOperator input axis')
        return ()

    @staticmethod
    def _contract_factor_digits(
            factor: TTDecomposition,
            digits: torch.Tensor) -> torch.Tensor:
        """Contracts digit sites while leaving the final gamma axis open."""
        state = None
        for site, core in enumerate(factor._standard_cores()[:-1]):
            vector = torch.nn.functional.one_hot(
                digits[:, site], num_classes=core.shape[-2]).to(factor.dtype)
            local = torch.einsum('bp,lpr->blr', vector, core)
            state = local if state is None else state @ local
        connector = factor._standard_cores()[-1].squeeze(-1)
        return (state @ connector).squeeze(-2)

    def _split_local_factor(
            self,
            full_factor: TTDecomposition,
            fixed_shape: Tuple[int, ...],
            axis: int,
            return_info: bool
            ) -> Tuple[TTDecomposition, torch.Tensor,
                       Optional[TruncationRecord]]:
        """Separates digit and environment blocks through a small interface."""
        standard = full_factor._standard_cores()
        digit_cores = []
        carry = None
        for site in range(self.layout.n_sites):
            core = standard[site]
            if carry is not None:
                core = torch.einsum('ab,bpr->apr', carry, core)
            matrix = core.reshape(-1, core.shape[-1])
            q, carry = torch.linalg.qr(matrix, mode='reduced')
            digit_cores.append(q.reshape(
                core.shape[0], core.shape[1], q.shape[-1]))

        if fixed_shape:
            environment = torch.einsum(
                'ab,bpr->apr', carry, standard[self.layout.n_sites])
            for core in standard[self.layout.n_sites + 1:]:
                environment = torch.tensordot(
                    environment, core, dims=([-1], [0]))
            environment = environment.squeeze(-1)
        else:
            environment = carry.squeeze(-1)
        matrix = environment.reshape(environment.shape[0], -1)
        options = {
            'rank': self.connector_rank,
            'cutoff': self.cutoff,
            'atol': self.atol,
            'rtol': self.rtol,
            'cum_percentage': self.cum_percentage,
        }
        if return_info:
            u, s, vh, split_info = truncated_svd(
                matrix, return_info=True, **options)
            truncation = replace(
                TruncationRecord.from_svd_info(split_info, site=axis),
                phase='qtt_connector')
        else:
            u, s, vh = truncated_svd(matrix, **options)
            truncation = None
        digit_cores[-1] = torch.einsum(
            'lpr,rg->lpg', digit_cores[-1], u)
        gamma = u.shape[-1]
        connector = torch.eye(
            gamma, device=u.device, dtype=u.dtype).unsqueeze(-1)
        standard_factor = [*digit_cores, connector]
        factor_cores = [
            standard_factor[0].squeeze(0),
            *standard_factor[1:-1],
            standard_factor[-1].squeeze(-1),
        ]
        factor = TTDecomposition(
            factor_cores,
            metrics=full_factor.metrics,
            metadata={
                'algorithm': 'qtt_input_factor',
                'connector_rank': gamma,
            })
        reduced = (s.unsqueeze(1) * vh).reshape(gamma, *fixed_shape)
        return factor, reduced, truncation

    def fit(
            self,
            phi_view: PhiView,
            axis: int,
            domain: torch.Tensor,
            context: Any = None,
            return_info: bool = False) -> FittedInputAxis:
        """Runs local QTT-RSS and restores the quantized axis position."""
        if not isinstance(return_info, bool):
            raise TypeError('`return_info` should be bool type')
        shape = _phi_shape(phi_view)
        axis = _normalize_axis(axis, shape)
        if not callable(getattr(phi_view, 'with_axis_values', None)):
            raise TypeError(
                'QTTInputFitter requires a functional PhiOperator input axis')

        adapter = QuantizedSourceAdapter(
            lambda values: values[:, 0],
            self.layout,
            self.coordinate_map,
            self.domain,
            dtype=torch.get_default_dtype(),
            device=_phi_device(phi_view, domain))
        indices = torch.arange(
            self.layout.grid_size[0], device=adapter.device).reshape(-1, 1)
        digits = self.layout.encode_indices(indices)
        fixed_shape = shape[:axis] + shape[axis + 1:]

        def local_function(values):
            local_values = values[:, 0]
            functional = phi_view.with_axis_values(axis, local_values)
            tensor = functional.materialize(batch_size=self.batch_size)
            return tensor.movedim(axis, 0)

        n_outputs = prod(fixed_shape) if fixed_shape else 1
        if fixed_shape:
            sketch_digits = digits.repeat_interleave(n_outputs, dim=0)
            labels = torch.arange(
                n_outputs, device=digits.device).repeat(digits.shape[0])
            out_position = tuple(range(
                self.layout.n_sites,
                self.layout.n_sites + len(fixed_shape)))
        else:
            sketch_digits = digits
            labels = None
            out_position = None

        from tensorkrowch.decompositions.sketching.tt import TTRSS

        full_factor = TTRSS.quantized(
            local_function,
            layout=self.layout,
            coordinate_map=self.coordinate_map,
            domain=self.domain,
            sample_space='digits',
            out_position=out_position,
            output_device=None).fit(
                sketch_digits,
                labels=labels,
                rank=self.rank,
                cutoff=self.cutoff,
                atol=self.atol,
                rtol=self.rtol,
                cum_percentage=self.cum_percentage,
                batch_size=self.batch_size,
                generator=torch.Generator().manual_seed(self.seed),
                legacy_projection=False,
                verbose=0,
                collect_metrics=return_info)
        factor, reduced, truncation = self._split_local_factor(
            full_factor, fixed_shape, axis, return_info)
        target = None
        if return_info:
            physical = adapter.indices_to_physical(indices)
            target = local_function(physical).to(
                device=factor.device, dtype=factor.dtype)
        reconstructed = None
        if self.materialize_tensor or return_info:
            factor_matrix = self._contract_factor_digits(factor, digits)
            reconstructed = (factor_matrix @ reduced.reshape(
                reduced.shape[0], -1)).reshape(
                    self.layout.grid_size[0], *fixed_shape)
        reduced_tensor = reduced.movedim(0, axis)
        tensor = reduced_tensor if not self.materialize_tensor \
            else reconstructed.movedim(0, axis)
        record = None
        relative = None
        if return_info:
            absolute = torch.linalg.vector_norm(reconstructed - target)
            denominator = torch.linalg.vector_norm(target)
            if denominator > 0:
                relative = absolute / denominator
            elif absolute == 0:
                relative = torch.zeros_like(absolute)
            else:
                relative = torch.full_like(absolute, torch.inf)
            record = InputFitRecord(
                method='qtt',
                axis=axis,
                domain_size=self.layout.grid_size[0],
                input_dim=self.layout.grid_size[0],
                residual_absolute=absolute,
                residual_relative=relative,
                condition_number=1.,
                used_fibers=True)
        metadata = {
            'algorithm': 'qtt_input_fit',
            'layout': self.layout,
            'output_shape': fixed_shape,
            'out_position': out_position,
            'connector_rank': reduced.shape[0],
            'materialized_tensor': self.materialize_tensor,
        }
        if return_info:
            metadata.update({
                'split_relative_residual': (
                    truncation.local_relative_error),
                'final_relative_residual': float(relative.detach().cpu()),
            })
        return FittedInputAxis(
            tensor=tensor,
            axis=axis,
            domain_size=self.layout.grid_size[0],
            input_dim=tensor.shape[axis],
            record=record,
            factor=factor,
            reduced_tensor=reduced_tensor,
            truncation=truncation,
            metadata=metadata)

    def factor_values(
            self,
            fitted: FittedInputAxis,
            values: torch.Tensor,
            dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Evaluates the fitted QTT factor while leaving gamma uncontracted."""
        if not isinstance(fitted, FittedInputAxis) or fitted.factor is None:
            raise TypeError('`fitted` should contain a QTT factor')
        if not isinstance(values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        factor = fitted.factor
        real_dtype = factor.cores[0].real.dtype
        flat = values.to(device=factor.device, dtype=real_dtype).reshape(-1, 1)
        adapter = QuantizedSourceAdapter(
            lambda data: data[:, 0],
            self.layout,
            self.coordinate_map,
            self.domain,
            dtype=real_dtype,
            device=factor.device)
        digits = adapter.physical_to_digits(flat)
        state = None
        for site, core in enumerate(factor._standard_cores()[:-1]):
            vector = torch.nn.functional.one_hot(
                digits[:, site], num_classes=core.shape[-2]).to(factor.dtype)
            local = torch.einsum('bp,lpr->blr', vector, core)
            state = local if state is None else state @ local
        connector = factor._standard_cores()[-1].squeeze(-1)
        result = (state @ connector).squeeze(-2)
        result = result.reshape(*values.shape, result.shape[-1])
        return result if dtype is None else result.to(dtype=dtype)


__all__ = [
    'InputFitter',
    'FixedEmbeddingFitter',
    'BasisFitter',
    'TrainableEmbeddingFitter',
    'QTTInputFitter',
]
