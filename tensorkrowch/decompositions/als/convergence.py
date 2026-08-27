"""Convergence and update policies for ALS sweeps."""

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Optional

import torch

from tensorkrowch.decompositions.metrics import SweepRecord


@dataclass(frozen=True)
class UpdatePolicy:
    """Applies damping and optional local non-increasing acceptance."""

    damping: float = 1.0
    acceptance: str = 'always'

    def __post_init__(self) -> None:
        if isinstance(self.damping, bool) or \
                (not isinstance(self.damping, (int, float))):
            raise TypeError('`damping` should be a number between 0 and 1')
        if (self.damping <= 0) or (self.damping > 1) or \
                (not isfinite(self.damping)):
            raise ValueError('`damping` should be a number between 0 and 1')
        object.__setattr__(self, 'damping', float(self.damping))
        if self.acceptance not in ('always', 'non_increasing'):
            raise ValueError(
                "`acceptance` should be 'always' or 'non_increasing'")

    def apply(self,
              current: torch.Tensor,
              proposal: torch.Tensor) -> torch.Tensor:
        """Returns the damped proposal without mutating either input."""
        if not isinstance(current, torch.Tensor) or \
                not isinstance(proposal, torch.Tensor):
            raise TypeError('Updates should be torch.Tensor objects')
        if current.shape != proposal.shape:
            raise ValueError('Current and proposed cores should have equal shapes')
        if (current.device != proposal.device) or \
                (current.dtype != proposal.dtype):
            raise ValueError('Current and proposed cores should share runtime')
        if self.damping == 1:
            return proposal
        return current + self.damping * (proposal - current)

    def accepts(self,
                previous_local_error: Optional[float],
                proposed_local_error: Optional[float]) -> bool:
        """Whether a local proposal satisfies the acceptance strategy."""
        if self.acceptance == 'always':
            return True
        if (previous_local_error is None) or (proposed_local_error is None):
            raise ValueError(
                'Non-increasing acceptance requires both local errors')
        return proposed_local_error <= previous_local_error


@dataclass(frozen=True)
class ConvergencePolicy:
    """Primary ALS stopping criteria evaluated at complete sweep boundaries."""

    max_sweeps: int = 10
    error_atol: Optional[float] = None
    error_rtol: Optional[float] = None
    change_rtol: Optional[float] = None
    patience: Optional[int] = None
    keep_best: bool = False
    callback: Optional[Callable] = None

    def __post_init__(self) -> None:
        if isinstance(self.max_sweeps, bool) or \
                (not isinstance(self.max_sweeps, int)) or \
                (self.max_sweeps < 1):
            raise ValueError('`max_sweeps` should be a positive integer')
        for name in ('error_atol', 'error_rtol', 'change_rtol'):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f'`{name}` should be a non-negative number')
            if (value < 0) or (not isfinite(value)):
                raise ValueError(f'`{name}` should be a non-negative number')
            object.__setattr__(self, name, float(value))
        if self.patience is not None:
            if isinstance(self.patience, bool) or \
                    (not isinstance(self.patience, int)) or (self.patience < 1):
                raise ValueError('`patience` should be a positive integer')
            if self.change_rtol is None:
                raise ValueError('`patience` requires `change_rtol`')
        if not isinstance(self.keep_best, bool):
            raise TypeError('`keep_best` should be bool type')
        if (self.callback is not None) and (not callable(self.callback)):
            raise TypeError('`callback` should be callable or None')

    @property
    def requires_fixed_objective(self) -> bool:
        """Whether the policy compares objective values across sweeps."""
        return any(value is not None for value in (
            self.error_atol,
            self.error_rtol,
            self.change_rtol,
        )) or self.keep_best

    def validate_objective(self, has_fixed_objective: bool) -> None:
        """Rejects convergence on renewable, incomparable sample batches."""
        if self.requires_fixed_objective and not has_fixed_objective:
            raise ValueError(
                'Error tolerances, stability and best state require a fixed '
                'global objective')

    def stopping_reason(self,
                        record: SweepRecord,
                        stable_sweeps: int) -> tuple:
        """Returns a normalized reason and updated stability count."""
        if (self.error_atol is not None) and \
                (record.absolute_error is not None) and \
                (record.absolute_error <= self.error_atol):
            return 'error_atol', stable_sweeps
        if (self.error_rtol is not None) and \
                (record.relative_error is not None) and \
                (record.relative_error <= self.error_rtol):
            return 'error_rtol', stable_sweeps

        if self.change_rtol is not None:
            if (record.relative_change is not None) and \
                    (record.relative_change <= self.change_rtol):
                stable_sweeps += 1
            else:
                stable_sweeps = 0
            required = 1 if self.patience is None else self.patience
            if stable_sweeps >= required:
                return 'relative_stability', stable_sweeps
        if (record.sweep + 1) >= self.max_sweeps:
            return 'max_sweeps', stable_sweeps
        return None, stable_sweeps


__all__ = ['UpdatePolicy', 'ConvergencePolicy']
