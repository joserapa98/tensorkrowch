"""
This script contains:

    Public classes:
        * ConvergencePolicy
"""

from dataclasses import dataclass
from math import isfinite
from typing import Callable, Optional, Tuple

from tensorkrowch.decompositions.metrics import SweepRecord


@dataclass(frozen=True)
class ConvergencePolicy:  # MARK: ConvergencePolicy
    """
    Defines ALS stopping rules at complete sweep boundaries.

    Error, stability and best-state policies require a fixed comparable
    objective: the full tensor for exact ALS or the permanent observed set for
    completion. Renewable uniform/leverage batches support a sweep budget and
    an optional ``callback``, because their local residuals are not global
    validation errors.

    Parameters
    ----------
    max_sweeps : int
        Positive maximum number of directional sweeps. Default is ``10``.
    atol : float, optional
        Stop when the absolute objective error is at most this value.
    rtol : float, optional
        Stop when the relative objective error is at most this value.
    stagnation_rtol : float, optional
        Maximum relative change between consecutive complete-sweep errors for a
        sweep to count as stable. These tolerances are objective tolerances,
        not singular-value truncation criteria.
    patience : int, optional
        Consecutive stable sweeps required. Defaults to one when
        ``stagnation_rtol`` is set; otherwise it must be omitted.
    keep_best : bool
        Restore the best complete-sweep state before returning. Default is
        ``False``.
    callback : callable, optional
        Called as ``callback(sweep_record, cores)`` after a complete sweep.
        Return ``True`` to stop. Cores are the current state; the callback
        should not mutate them or the backend caches.

    Examples
    --------
    >>> policy = tk.decompositions.ConvergencePolicy(max_sweeps=20, rtol=1e-8)
    >>> policy.max_sweeps, policy.requires_fixed_objective
    (20, True)
    """

    max_sweeps: int = 10  # Maximum number of directional sweeps
    atol: Optional[float] = None  # Absolute objective stopping tolerance
    rtol: Optional[float] = None  # Relative objective stopping tolerance
    stagnation_rtol: Optional[float] = None  # Relative stagnation tolerance
    patience: Optional[int] = None  # Required consecutive stable sweeps
    keep_best: bool = False  # Whether to restore the best complete-sweep state
    callback: Optional[Callable] = None  # Complete-sweep stopping callback

    def __post_init__(self) -> None:
        if isinstance(self.max_sweeps, bool) or \
                (not isinstance(self.max_sweeps, int)) or \
                (self.max_sweeps < 1):
            raise ValueError('`max_sweeps` should be a positive integer')
        for name in ('atol', 'rtol', 'stagnation_rtol'):
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
                    (not isinstance(self.patience, int)) or \
                    (self.patience < 1):
                raise ValueError('`patience` should be a positive integer')
            if self.stagnation_rtol is None:
                raise ValueError('`patience` requires `stagnation_rtol`')
        if not isinstance(self.keep_best, bool):
            raise TypeError('`keep_best` should be bool type')
        if (self.callback is not None) and (not callable(self.callback)):
            raise TypeError('`callback` should be callable or None')

    @property
    def requires_fixed_objective(self) -> bool:
        """Whether the policy compares objective values across sweeps."""
        return any(value is not None for value in (
            self.atol,
            self.rtol,
            self.stagnation_rtol,
        )) or self.keep_best

    def validate_objective(self, has_fixed_objective: bool) -> None:
        """
        Validates that the objective supports the requested stopping rules.

        Parameters
        ----------
        has_fixed_objective : bool
            Whether complete sweeps measure the same target entries.
        """
        if not isinstance(has_fixed_objective, bool):
            raise TypeError('`has_fixed_objective` should be bool type')
        if self.requires_fixed_objective and not has_fixed_objective:
            raise ValueError(
                'Error tolerances, stability and best state require a fixed '
                'global objective')

    def stopping_reason(self,
                        record: SweepRecord,
                        stable_sweeps: int) -> Tuple[Optional[str], int]:
        """
        Evaluates objective tolerances, stability and the sweep budget.

        Parameters
        ----------
        record : SweepRecord
            Measurements at the end of the current directional sweep.
        stable_sweeps : int
            Number of preceding consecutive sweeps satisfying
            ``stagnation_rtol``.

        Returns
        -------
        tuple[str or None, int]
            Stopping reason, if any, and updated consecutive stability count.
        """
        if (self.atol is not None) and \
                (record.abs_error is not None) and \
                (record.abs_error <= self.atol):
            return 'atol', stable_sweeps
        if (self.rtol is not None) and \
                (record.rel_error is not None) and \
                (record.rel_error <= self.rtol):
            return 'rtol', stable_sweeps

        if self.stagnation_rtol is not None:
            if (record.rel_change is not None) and \
                    (record.rel_change <= self.stagnation_rtol):
                stable_sweeps += 1
            else:
                stable_sweeps = 0
            required = 1 if self.patience is None else self.patience
            if stable_sweeps >= required:
                return 'relative_stability', stable_sweeps
        if (record.sweep + 1) >= self.max_sweeps:
            return 'max_sweeps', stable_sweeps
        return None, stable_sweeps


__all__ = [
    'ConvergencePolicy',
]
