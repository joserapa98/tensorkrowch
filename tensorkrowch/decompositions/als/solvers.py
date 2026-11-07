"""
This script contains:

    Classes:
        * NonFiniteLocalSystemError
        * NonFiniteSolutionError
        * LeastSquaresSolver

    Functions:
        * _column_norms
        * _column_scales
        * _solve_local_proposal
"""

from dataclasses import replace
from math import isfinite, sqrt
from typing import Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.utils import safe_divide, scale_tensor, stable_norm

from tensorkrowch.decompositions.als.problem import ALSProblem
from tensorkrowch.decompositions.als.sampling import SampleBatch
from tensorkrowch.decompositions.metrics import LocalSolveRecord


ColumnScaling = Union[bool, str]


class NonFiniteLocalSystemError(ValueError):  # MARK: NonFiniteLocalSystemError
    """Raised when a local environment or target contains NaN/Inf."""


class NonFiniteSolutionError(RuntimeError):  # MARK: NonFiniteSolutionError
    """Raised when a local least-squares solution contains NaN/Inf."""


def _column_norms(environment: torch.Tensor) -> torch.Tensor:
    """Computes L2 column norms with a finite maximum-magnitude fallback."""
    scaled, maxima = scale_tensor(environment, dim=0)
    normalized = torch.linalg.vector_norm(scaled, dim=0)
    column_norms = maxima * normalized
    return torch.where(torch.isfinite(column_norms), column_norms, maxima)


def _column_scales(environment: torch.Tensor, n_copies: int = 1
                   ) -> torch.Tensor:
    """Returns safe L2 column scales for an augmented design matrix."""
    column_norms = _column_norms(environment)
    if not column_norms.numel():
        return column_norms
    maximum = column_norms.max()
    threshold = n_copies * max(environment.shape) * \
        torch.finfo(column_norms.dtype).eps * maximum
    return torch.where(
        column_norms > threshold,
        column_norms,
        torch.ones_like(column_norms))


class LeastSquaresSolver:  # MARK: LeastSquaresSolver
    r"""
    Solves local least-squares systems with optional Tikhonov regularization.

    The solver minimizes :math:`\|A x - b\|_2^2 + \lambda \|x\|_2^2`. Tikhonov
    regularization is represented by augmented rows, never by normal equations.
    Column scaling is then applied as a change of variables to the complete
    augmented matrix, followed by one global scaling of both matrix and
    right-hand side.

    Parameters
    ----------
    l2_reg : float, optional
        Non-negative regularization coefficient. In ``"absolute"`` mode this is
        ``lambda``. In ``"relative"`` mode it is multiplied by the square of
        the RMS column norm of the original environment.
    l2_reg_mode : {``"absolute"``, ``"relative"``}, optional
        Interpretation of ``l2_reg`` before any numerical scaling.
    rcond : float or None, optional
        Cutoff forwarded to :func:`torch.linalg.lstsq`.
    column_scaling : bool or ``"auto"``, optional
        Whether to balance columns. ``"auto"`` activates when the ratio of
        non-zero column norms exceeds ``1 / sqrt(machine epsilon)``.
    system_scaling : bool, optional
        Whether to divide the complete transformed system by its largest
        magnitude before solving.
    driver : str or None, optional
        Preferred :func:`torch.linalg.lstsq` driver. CPU failures are retried
        with the remaining supported drivers and the effective driver is stored
        in :class:`LocalSolveRecord`.

    Examples
    --------
    >>> matrix = torch.eye(2)
    >>> target = torch.tensor([1., 2.])
    >>> solver = tk.decompositions.LeastSquaresSolver()
    >>> solution, record = solver.solve(matrix, target)
    >>> torch.allclose(solution, target)
    True
    """

    _DRIVERS = ('gelsy', 'gelsd', 'gelss', 'gels')

    def __init__(self,
                 l2_reg: float = 0.0,
                 l2_reg_mode: str = 'absolute',
                 rcond: Optional[float] = None,
                 column_scaling: ColumnScaling = 'auto',
                 system_scaling: bool = True,
                 driver: Optional[str] = None) -> None:
        if isinstance(l2_reg, bool) or not isinstance(l2_reg, (int, float)):
            raise TypeError('`l2_reg` should be a non-negative number')
        if (l2_reg < 0) or (not isfinite(l2_reg)):
            raise ValueError('`l2_reg` should be a non-negative number')
        if l2_reg_mode not in ('absolute', 'relative'):
            raise ValueError(
                "`l2_reg_mode` should be 'absolute' or 'relative'")
        if rcond is not None:
            if isinstance(rcond, bool) or not isinstance(rcond, (int, float)):
                raise TypeError(
                    '`rcond` should be a non-negative number or None')
            if (rcond < 0) or (not isfinite(rcond)):
                raise ValueError('`rcond` should be a non-negative number')
        if not (isinstance(column_scaling, bool) or
                (column_scaling == 'auto')):
            raise ValueError(
                '`column_scaling` should be bool or "auto"')
        if not isinstance(system_scaling, bool):
            raise TypeError('`system_scaling` should be bool type')
        if (driver is not None) and (driver not in self._DRIVERS):
            raise ValueError(
                '`driver` should be a torch.linalg.lstsq driver or None')

        self.l2_reg = float(l2_reg)
        self.l2_reg_mode = l2_reg_mode
        self.rcond = None if rcond is None else float(rcond)
        self.column_scaling = column_scaling
        self.system_scaling = system_scaling
        self.driver = driver

    def _effective_regularization(self,
                                  environment: torch.Tensor,
                                  regularization_scale: Optional[
                                      torch.Tensor] = None,
                                  n_parameters: Optional[int] = None
                                  ) -> torch.Tensor:
        """Determines lambda in the unscaled problem."""
        value = environment.real.new_tensor(self.l2_reg)
        if regularization_scale is not None:
            if not isinstance(regularization_scale, torch.Tensor):
                raise TypeError(
                    '`regularization_scale` should be a scalar tensor or None')
            if regularization_scale.numel() != 1:
                raise ValueError('`regularization_scale` should be a scalar')
            if regularization_scale.is_complex():
                raise TypeError('`regularization_scale` should be real')
            if regularization_scale.device != environment.device:
                raise ValueError(
                    '`regularization_scale` and environment should share a '
                    'device')
            regularization_scale = regularization_scale.to(
                dtype=environment.real.dtype)
            if (not torch.isfinite(regularization_scale)) or \
                    (regularization_scale < 0):
                raise ValueError(
                    '`regularization_scale` should be finite and non-negative')
            value = value * regularization_scale
        if not torch.isfinite(value):
            raise ValueError('Effective `l2_reg` should be finite')
        if (self.l2_reg_mode == 'absolute') or (self.l2_reg == 0):
            return value
        if n_parameters is None:
            n_parameters = environment.shape[1]
        rms_column_norm = stable_norm(environment) / sqrt(n_parameters)
        value = value * rms_column_norm.square()
        if not torch.isfinite(value):
            raise ValueError(
                'Relative `l2_reg` is not finite at the environment scale')
        return value

    def _augment(self,
                 environment: torch.Tensor,
                 target: torch.Tensor,
                 effective_l2_reg: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """Appends Tikhonov rows to the original system."""
        if effective_l2_reg == 0:
            return environment, target
        n_parameters = environment.shape[1]
        identity = torch.eye(
            n_parameters,
            device=environment.device,
            dtype=environment.dtype)
        augmented_environment = torch.cat((
            environment,
            effective_l2_reg.sqrt().to(environment.dtype) * identity,
        ), dim=0)
        zeros = target.new_zeros((n_parameters, target.shape[1]))
        augmented_target = torch.cat((target, zeros), dim=0)
        return augmented_environment, augmented_target

    def _should_scale_columns(self, environment: torch.Tensor) -> bool:
        """Resolves the explicit or automatic column-scaling policy."""
        if isinstance(self.column_scaling, bool):
            return self.column_scaling
        norms = _column_norms(environment)
        positive = norms[norms > 0]
        if positive.numel() < 2:
            return False
        imbalance = positive.max() / positive.min()
        threshold = 1 / sqrt(torch.finfo(norms.dtype).eps)
        return bool(imbalance > threshold)

    def _lstsq(self,
               environment: torch.Tensor,
               target: torch.Tensor,
               rcond: Optional[float]) -> Tuple[torch.Tensor, str]:
        """Solves with a visible, controlled CPU fallback sequence."""
        attempted = 'default' if self.driver is None else self.driver
        try:
            result = torch.linalg.lstsq(
                environment,
                target,
                rcond=rcond,
                driver=self.driver)
            return result.solution, attempted
        except RuntimeError as initial_error:
            if environment.device.type != 'cpu':
                raise RuntimeError(
                    f'Least-squares driver {attempted!r} failed on '
                    f'{environment.device.type}') from initial_error

            last_error = initial_error
            for driver in self._DRIVERS:
                if driver == self.driver:
                    continue
                try:
                    result = torch.linalg.lstsq(
                        environment,
                        target,
                        rcond=rcond,
                        driver=driver)
                    return result.solution, driver
                except RuntimeError as fallback_error:
                    last_error = fallback_error
            raise RuntimeError(
                'All CPU least-squares drivers failed') from last_error

    def solve(self,
              environment: torch.Tensor,
              target: torch.Tensor,
              site: Optional[Union[int, Tuple[int, ...]]] = None,
              sweep: Optional[int] = None,
              return_record: bool = True,
              regularization_scale: Optional[
                  torch.Tensor] = None) -> Tuple[
            torch.Tensor,
            Optional['LocalSolveRecord']]:
        """
        Solves one local system and optionally records its diagnostics.

        A vector ``target`` produces a vector solution. Multiple right-hand
        sides
        share the same factorization. Diagnostics describe the original local
        system before column and system scaling; records are detached, while
        the solution retains its PyTorch autograd graph. No device fallback is
        performed.

        Parameters
        ----------
        environment : torch.Tensor
            Real or complex design matrix with shape ``(n_rows,
            n_parameters)``.
        target : torch.Tensor
            Right-hand side with shape ``(n_rows,)`` or ``(n_rows,
            n_targets)``, sharing the matrix device and dtype.
        site : int or tuple[int], optional
            Site or block identifier attached to the local record.
        sweep : int, optional
            Directional sweep identifier attached to the local record.
        return_record : bool
            Whether to calculate residuals and create a ``LocalSolveRecord``.
            Defaults to ``True``. ``False`` skips diagnostic calculations.
        regularization_scale : torch.Tensor, optional
            Non-negative scalar factor for the regularization coefficient when
            a caller has normalized both the design and ``target``. Environment
            caches use it to preserve absolute Tikhonov regularization under
            that normalization.

        Returns
        -------
        tuple[torch.Tensor, LocalSolveRecord or None]
            Solution and optional diagnostics, including the effective
            least-squares driver. CPU driver retries remain on CPU; accelerator
            errors propagate.
        """
        return self._solve(environment, target, site, sweep, return_record,
                           regularization_scale)

    def _solve(self, environment: torch.Tensor, target: torch.Tensor,
               site: Optional[Union[int, Tuple[int, ...]]] = None,
               sweep: Optional[int] = None, return_record: bool = True,
               regularization_scale: Optional[torch.Tensor] = None,
               n_copies: int = 1) -> Tuple[
            torch.Tensor, Optional[LocalSolveRecord]]:
        """Solves shared physical slices with the expanded rank thresholds."""
        if not isinstance(environment, torch.Tensor):
            raise TypeError('`environment` should be torch.Tensor type')
        if not isinstance(target, torch.Tensor):
            raise TypeError('`target` should be torch.Tensor type')
        if environment.ndim != 2:
            raise ValueError('`environment` should be a matrix')
        if target.ndim not in (1, 2):
            raise ValueError('`target` should be a vector or matrix')
        if (environment.shape[0] < 1) or (environment.shape[1] < 1):
            raise ValueError('`environment` dimensions should be positive')
        if target.shape[0] != environment.shape[0]:
            raise ValueError(
                '`target` and `environment` should have matching rows')
        if (target.ndim == 2) and (target.shape[1] < 1):
            raise ValueError(
                '`target` should contain at least one right-hand side')
        if target.device != environment.device:
            raise ValueError(
                '`target` and `environment` should share a device')
        if target.dtype != environment.dtype:
            raise ValueError('`target` and `environment` should share a dtype')
        if not (environment.is_floating_point() or environment.is_complex()):
            raise TypeError(
                '`environment` should have a floating or complex dtype')
        if not torch.isfinite(environment).all():
            raise NonFiniteLocalSystemError(
                '`environment` should contain only finite values')
        if not torch.isfinite(target).all():
            raise NonFiniteLocalSystemError(
                '`target` should contain only finite values')
        if not isinstance(return_record, bool):
            raise TypeError('`return_record` should be bool type')

        vector_target = target.ndim == 1
        target_matrix = target.unsqueeze(1) if vector_target else target
        effective_l2_reg = self._effective_regularization(
            environment, regularization_scale=regularization_scale)
        transformed_environment, transformed_target = self._augment(
            environment, target_matrix, effective_l2_reg)

        apply_column_scaling = self._should_scale_columns(
            transformed_environment)
        if apply_column_scaling:
            column_scales = _column_scales(transformed_environment, n_copies)
            transformed_environment = transformed_environment / \
                column_scales.to(transformed_environment.dtype).unsqueeze(0)
        else:
            column_scales = transformed_environment.real.new_ones(
                transformed_environment.shape[1])

        system_scale = transformed_environment.real.new_tensor(1.)
        if self.system_scaling:
            system_scale = torch.maximum(
                transformed_environment.abs().amax(),
                transformed_target.abs().amax())
            if system_scale > 0:
                transformed_environment = (
                    transformed_environment / system_scale)
                transformed_target = transformed_target / system_scale
            else:
                system_scale = system_scale.new_tensor(1.)

        # Repeated physical blocks have the same singular values. Preserve
        # the expanded system's automatic cutoff without constructing it.
        rcond = self.rcond
        if (rcond is None) and (n_copies != 1):
            rcond = (torch.finfo(environment.real.dtype).eps * n_copies *
                     max(transformed_environment.shape))
        scaled_solution, effective_driver = self._lstsq(
            transformed_environment, transformed_target, rcond)
        solution = scaled_solution / \
            column_scales.to(scaled_solution.dtype).unsqueeze(1)
        if not torch.isfinite(solution).all():
            raise NonFiniteSolutionError(
                'Least-squares solution contains non-finite values')

        returned_solution = solution.squeeze(1) if vector_target else solution
        if not return_record:
            return returned_solution, None

        residual = environment @ solution - target_matrix
        abs_residual = torch.linalg.vector_norm(residual)
        target_norm = torch.linalg.vector_norm(target_matrix)
        rel_residual = safe_divide(
            abs_residual, target_norm,
            zero_value=torch.where(abs_residual == 0, 0, torch.inf))
        record = LocalSolveRecord(
            environment_shape=tuple(environment.shape),
            target_shape=tuple(target.shape),
            driver=effective_driver,
            abs_residual=abs_residual,
            rel_residual=rel_residual,
            effective_l2_reg=effective_l2_reg,
            column_scaling=apply_column_scaling,
            site=site,
            sweep=sweep)
        return returned_solution, record

    def _solve_blocks(self, environments: Sequence[torch.Tensor],
                      targets: Sequence[torch.Tensor], n_rows: int,
                      site: int, sweep: int, return_record: bool,
                      regularization_scale: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, Optional[LocalSolveRecord]]:
        """Solves physical blocks with shared numerical policies."""
        n_inputs = len(environments)
        n_parameters = environments[0].shape[1]
        reference = environments[0]
        for matrix, target in zip(environments, targets):
            if not torch.isfinite(matrix).all() or \
                    not torch.isfinite(target).all():
                raise NonFiniteLocalSystemError(
                    'Local blocks should contain only finite values')
        # Relative regularization belongs to the complete core, including
        # slices with no observations, rather than to each slice separately.
        norms = torch.stack([stable_norm(matrix) for matrix in environments])
        penalty = self._effective_regularization(
            norms, regularization_scale, n_inputs * n_parameters)
        augmented = [self._augment(matrix, target.unsqueeze(1), penalty)
                     for matrix, target in zip(environments, targets)]
        column_norms = torch.stack([
            _column_norms(matrix) if matrix.shape[0] else
            reference.real.new_zeros(n_parameters)
            for matrix, _ in augmented])
        positive = column_norms[column_norms > 0]
        apply_scaling = self.column_scaling is True
        if (self.column_scaling == 'auto') and (positive.numel() > 1):
            threshold = 1 / sqrt(torch.finfo(positive.dtype).eps)
            apply_scaling = bool(positive.max() / positive.min() > threshold)
        logical_rows = n_rows + (n_inputs * n_parameters if penalty > 0 else 0)
        logical_size = max(logical_rows, n_inputs * n_parameters)
        if apply_scaling:
            threshold = (logical_size * torch.finfo(column_norms.dtype).eps *
                         column_norms.max())
            scales = torch.where(column_norms > threshold, column_norms,
                                 torch.ones_like(column_norms))
        else:
            scales = torch.ones_like(column_norms)
        matrices = [matrix / scale.to(matrix.dtype).unsqueeze(0)
                    for (matrix, _), scale in zip(augmented, scales)]
        rhs = [target for _, target in augmented]
        if self.system_scaling:
            magnitudes = [value.abs().amax() for value in [*matrices, *rhs]
                          if value.numel()]
            scale = torch.stack(magnitudes).max()
            if scale > 0:
                matrices = [matrix / scale for matrix in matrices]
                rhs = [target / scale for target in rhs]

        # The expanded matrix is block diagonal up to permutations. A single
        # absolute singular-value cutoff must therefore apply to all blocks.
        rcond = self.rcond
        if rcond is None:
            rcond = torch.finfo(reference.real.dtype).eps * logical_size
        maxima = [torch.linalg.svdvals(matrix)[0] if matrix.numel() else
                  reference.real.new_zeros(()) for matrix in matrices]
        cutoff = rcond * torch.stack(maxima).max()
        solutions, drivers = [], []
        for matrix, target, maximum, scale in zip(
                matrices, rhs, maxima, scales):
            if (maximum == 0) or (
                    (self.driver != 'gels') and (maximum <= cutoff)):
                solution = reference.new_zeros((n_parameters, 1))
            else:
                solution, driver = self._lstsq(
                    matrix, target, float((cutoff / maximum).detach()))
                drivers.append(driver)
            solutions.append(solution[:, 0] / scale.to(solution.dtype))
        solution = torch.stack(solutions, dim=1)
        if not torch.isfinite(solution).all():
            raise NonFiniteSolutionError(
                'Least-squares solution contains non-finite values')
        if not return_record:
            return solution, None
        residuals = [stable_norm(matrix @ solution[:, j] - target)
                     for j, (matrix, target) in enumerate(
                         zip(environments, targets))]
        absolute = stable_norm(torch.stack(residuals))
        target_norm = stable_norm(torch.stack([
            stable_norm(target) for target in targets]))
        relative = safe_divide(
            absolute, target_norm,
            zero_value=torch.where(absolute == 0, 0, torch.inf))
        record = LocalSolveRecord(
            environment_shape=(n_rows, n_parameters),
            target_shape=(n_rows,),
            driver=','.join(dict.fromkeys(drivers)) if drivers else 'zero',
            abs_residual=absolute, rel_residual=relative,
            effective_l2_reg=penalty, column_scaling=apply_scaling,
            site=site, sweep=sweep)
        return solution, record


def _solve_local_proposal(solver: LeastSquaresSolver,
                          environment: torch.Tensor,
                          target: torch.Tensor,
                          current: torch.Tensor,
                          site: int,
                          sweep: int,
                          return_record: bool,
                          *,
                          problem: ALSProblem,
                          log_scale: torch.Tensor,
                          indices: Optional[torch.Tensor] = None,
                          samples: Optional[SampleBatch] = None,
                          core_versions: Sequence[int] = (),
                          compact: bool = False) -> Tuple[
        torch.Tensor, Optional[LocalSolveRecord]]:
    """Solves scalar rows or shared physical fibers and restores the core."""
    regularization_scale = None
    if (solver.l2_reg_mode == 'absolute') and (solver.l2_reg > 0):
        regularization_scale = (-2 * log_scale).exp()
    target = target * (-log_scale).exp().to(target.dtype)
    weights = problem._row_weights(indices)
    if weights is not None:
        weights = weights.to(environment.real.dtype)
    if samples is not None:
        sample_weights = samples.weights.to(environment.real.dtype)
        weights = (sample_weights if weights is None else
                   weights * sample_weights)

    if compact:
        dimension = current.shape[1]
        if problem.observations is None:
            if samples is None:
                target = target.reshape(problem.in_dim).movedim(site, -1)
                if weights is not None:
                    weights = weights.reshape(problem.in_dim).movedim(site, -1)
            target = target.reshape(-1, dimension)
            if weights is not None:
                weights = weights.reshape(-1, dimension)
            if (weights is None) or torch.equal(
                    weights, weights[:, :1].expand_as(weights)):
                if weights is not None:
                    environment = environment * weights[:, :1]
                    target = target * weights
                solution, record = solver._solve(
                    environment, target, site, sweep, return_record,
                    regularization_scale, n_copies=dimension)
            else:
                matrices = [environment * weights[:, j:j + 1]
                            for j in range(dimension)]
                targets = [target[:, j] * weights[:, j]
                           for j in range(dimension)]
                solution, record = solver._solve_blocks(
                    matrices, targets, target.numel(), site, sweep,
                    return_record, regularization_scale)
        else:
            if weights is not None:
                environment = environment * weights.unsqueeze(1)
                target = target * weights
            labels = problem.observations.support.values[:, site]
            selections = [labels == j for j in range(dimension)]
            matrices = [environment[selected] for selected in selections]
            targets = [target[selected] for selected in selections]
            solution, record = solver._solve_blocks(
                matrices, targets, target.numel(), site, sweep,
                return_record, regularization_scale)
    else:
        if weights is not None:
            environment = environment * weights.unsqueeze(1)
            target = target * weights
        solution, record = solver.solve(
            environment, target, site, sweep, return_record,
            regularization_scale)
    if (record is not None) and (samples is not None):
        record = replace(record,
                         sampling_exact=samples.is_exact_for(core_versions),
                         sample_generation=samples.generation)
    if compact:
        solution = solution.reshape(
            current.shape[0], current.shape[2], current.shape[1]
        ).transpose(1, 2)
    return solution.reshape(current.shape), record


__all__ = [
    'LeastSquaresSolver',
    'NonFiniteLocalSystemError',
    'NonFiniteSolutionError',
]
