"""Stable local least-squares solvers used by ALS drivers."""

from math import isfinite, sqrt
from typing import Optional, Tuple, Union

import torch

from tensorkrowch.decompositions.metrics import LocalSolveRecord


ColumnScaling = Union[bool, str]


def _stable_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Computes a Frobenius norm after removing the largest magnitude."""
    scale = tensor.abs().amax()
    if scale == 0:
        return scale
    return scale * torch.linalg.vector_norm(tensor / scale)


def _column_norms(environment: torch.Tensor) -> torch.Tensor:
    """Computes L2 column norms with a finite maximum-magnitude fallback."""
    maxima = environment.abs().amax(dim=0)
    safe_maxima = torch.where(
        maxima > 0, maxima, torch.ones_like(maxima))
    normalized = torch.linalg.vector_norm(
        environment / safe_maxima.to(environment.dtype).unsqueeze(0), dim=0)
    column_norms = maxima * normalized
    return torch.where(torch.isfinite(column_norms), column_norms, maxima)


def _column_scales(environment: torch.Tensor) -> torch.Tensor:
    """Returns safe L2 column scales for an augmented design matrix."""
    column_norms = _column_norms(environment)
    if not column_norms.numel():
        return column_norms
    maximum = column_norms.max()
    threshold = max(environment.shape) * \
        torch.finfo(column_norms.dtype).eps * maximum
    return torch.where(
        column_norms > threshold,
        column_norms,
        torch.ones_like(column_norms))


class LeastSquaresSolver:
    """Solve stable local least-squares systems with optional Tikhonov terms.

    The solver minimizes ``||A x - b||^2 + lambda ||x||^2``. Tikhonov
    regularization is represented by augmented rows, never by normal
    equations. Column scaling is then applied as a change of variables to the
    complete augmented matrix, followed by one global scaling of both matrix
    and right-hand side.

    Parameters
    ----------
    l2_reg : float, optional
        Non-negative regularization coefficient. In ``"absolute"`` mode this
        is ``lambda``. In ``"relative"`` mode it is multiplied by the square
        of the RMS column norm of the original environment.
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
        with the remaining supported drivers and the effective driver is
        stored in :class:`LocalSolveRecord`.
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
                raise TypeError('`rcond` should be a non-negative number or None')
            if (rcond < 0) or (not isfinite(rcond)):
                raise ValueError('`rcond` should be a non-negative number')
        if not (isinstance(column_scaling, bool) or
                column_scaling == 'auto'):
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

    def _effective_regularization(
            self, environment: torch.Tensor) -> torch.Tensor:
        """Determines lambda in the unscaled problem."""
        value = environment.real.new_tensor(self.l2_reg)
        if (self.l2_reg_mode == 'absolute') or (self.l2_reg == 0):
            return value
        rms_column_norm = _stable_norm(environment) / sqrt(
            environment.shape[1])
        value = value * rms_column_norm.square()
        if not torch.isfinite(value):
            raise ValueError(
                'Relative `l2_reg` is not finite at the environment scale')
        return value

    def _augment(self,
                 environment: torch.Tensor,
                 target: torch.Tensor,
                 effective_l2_reg: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _should_scale_columns(self,
                              environment: torch.Tensor) -> bool:
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
               target: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """Solves with a visible, controlled CPU fallback sequence."""
        attempted = 'default' if self.driver is None else self.driver
        try:
            result = torch.linalg.lstsq(
                environment,
                target,
                rcond=self.rcond,
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
                        rcond=self.rcond,
                        driver=driver)
                    return result.solution, driver
                except RuntimeError as fallback_error:
                    last_error = fallback_error
            raise RuntimeError(
                'All CPU least-squares drivers failed') from last_error

    def solve(self,
              environment: torch.Tensor,
              target: torch.Tensor,
              site=None,
              sweep: Optional[int] = None,
              return_record: bool = True):
        """Solves one local system and optionally records its diagnostics.

        ``target`` may be one- or two-dimensional. A one-dimensional target
        produces a one-dimensional solution; multiple right-hand sides are
        solved together. With ``return_record=False`` the second tuple element
        is ``None`` and no residual or Python scalar diagnostics are computed.
        """
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
            raise ValueError('`target` should contain at least one right-hand side')
        if target.device != environment.device:
            raise ValueError('`target` and `environment` should share a device')
        if target.dtype != environment.dtype:
            raise ValueError('`target` and `environment` should share a dtype')
        if not (environment.is_floating_point() or environment.is_complex()):
            raise TypeError(
                '`environment` should have a floating or complex dtype')
        if not torch.isfinite(environment).all():
            raise ValueError('`environment` should contain only finite values')
        if not torch.isfinite(target).all():
            raise ValueError('`target` should contain only finite values')
        if not isinstance(return_record, bool):
            raise TypeError('`return_record` should be bool type')

        vector_target = target.ndim == 1
        target_matrix = target.unsqueeze(1) if vector_target else target
        effective_l2_reg = self._effective_regularization(environment)
        transformed_environment, transformed_target = self._augment(
            environment, target_matrix, effective_l2_reg)

        apply_column_scaling = self._should_scale_columns(
            transformed_environment)
        if apply_column_scaling:
            column_scales = _column_scales(transformed_environment)
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
                transformed_environment = transformed_environment / system_scale
                transformed_target = transformed_target / system_scale
            else:
                system_scale = system_scale.new_tensor(1.)

        scaled_solution, effective_driver = self._lstsq(
            transformed_environment, transformed_target)
        solution = scaled_solution / \
            column_scales.to(scaled_solution.dtype).unsqueeze(1)
        if not torch.isfinite(solution).all():
            raise RuntimeError('Least-squares solution contains non-finite values')

        returned_solution = solution.squeeze(1) if vector_target else solution
        if not return_record:
            return returned_solution, None

        residual = environment @ solution - target_matrix
        residual_absolute = torch.linalg.vector_norm(residual)
        target_norm = torch.linalg.vector_norm(target_matrix)
        safe_target_norm = torch.where(
            target_norm > 0, target_norm, torch.ones_like(target_norm))
        residual_relative = residual_absolute / safe_target_norm
        residual_relative = torch.where(
            target_norm > 0,
            residual_relative,
            torch.where(
                residual_absolute == 0,
                torch.zeros_like(residual_absolute),
                torch.full_like(residual_absolute, torch.inf)))
        record = LocalSolveRecord(
            environment_shape=tuple(environment.shape),
            target_shape=tuple(target.shape),
            driver=effective_driver,
            residual_absolute=residual_absolute,
            residual_relative=residual_relative,
            target_norm=target_norm,
            l2_reg=self.l2_reg,
            effective_l2_reg=effective_l2_reg,
            l2_reg_mode=self.l2_reg_mode,
            column_scaling=apply_column_scaling,
            system_scaling=self.system_scaling,
            system_scale=system_scale,
            site=site,
            sweep=sweep)
        return returned_solution, record


__all__ = ['LeastSquaresSolver']
