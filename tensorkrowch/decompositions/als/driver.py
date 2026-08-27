"""Topology-independent ALS sweep orchestration."""

from dataclasses import dataclass, replace
from math import isfinite
from time import perf_counter
from typing import Optional, Protocol, Sequence, Tuple

import torch

from tensorkrowch.decompositions.als.convergence import (ConvergencePolicy,
                                                         UpdatePolicy)
from tensorkrowch.decompositions.als.environments import CoreUpdateSet
from tensorkrowch.decompositions.als.problem import ALSProblem
from tensorkrowch.decompositions.als.solvers import (
    NonFiniteLocalSystemError,
    NonFiniteSolutionError,
)
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 LocalSolveRecord,
                                                 SweepRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   DecompositionObserver)


class ALSBackend(Protocol):
    """Topology-specific operations consumed by ``ALSSweepDriver``."""

    @property
    def n_sites(self) -> int:
        """Number of sites in the decomposition."""

    @property
    def trainable_sites(self) -> Sequence[int]:
        """Sites that may be changed by local solves."""

    @property
    def cores(self) -> Sequence[torch.Tensor]:
        """Current cores."""

    def prepare_sweep(self,
                      order: Sequence[int],
                      sweep: int) -> Tuple[Optional[int], bool]:
        """Prepares caches/samples and reports generation plus refresh."""

    def solve_site(self,
                   site: int,
                   sweep: int,
                   update_policy: UpdatePolicy,
                   return_record: bool
                   ) -> Tuple[CoreUpdateSet, Optional[LocalSolveRecord]]:
        """Builds one atomic local update without committing it."""

    def skip_site(self, site: int) -> None:
        """Advances caches through a fixed site."""

    def commit(self, update_set: CoreUpdateSet) -> None:
        """Commits a topology-specific update atomically."""

    def measure_objective(
            self, problem: ALSProblem) -> Tuple[torch.Tensor, torch.Tensor]:
        """Measures one fixed/global objective after a complete sweep."""

    def snapshot(self) -> Sequence[torch.Tensor]:
        """Returns the current state for optional best-state retention."""

    def restore(self, cores: Sequence[torch.Tensor]) -> None:
        """Restores a previously captured state."""


@dataclass(frozen=True)
class _ALSDriverResult:
    """Internal result returned from the generic ALS sweep driver."""

    cores: Tuple[torch.Tensor, ...]
    metrics: DecompositionMetrics
    converged: bool
    stop_reason: str
    n_sweeps: int


def _relative_objective_change(previous: SweepRecord,
                               current_absolute: float,
                               current_relative: Optional[float]
                               ) -> Optional[float]:
    """Computes relative change between complete comparable objectives."""
    if (previous.relative_error is not None) and \
            (current_relative is not None):
        old = previous.relative_error
        new = current_relative
    elif previous.absolute_error is not None:
        old = previous.absolute_error
        new = current_absolute
    else:
        return None
    scale = max(abs(old), abs(new), 1e-16)
    return abs(old - new) / scale


class ALSSweepDriver:
    """Run alternating sweeps without topology-specific contractions.

    The backend owns environment construction and local solves. This driver
    chooses sweep directions, commits atomic updates, measures only complete
    sweep objectives, applies convergence and emits structured events.
    """

    def fit(self,
            problem: ALSProblem,
            backend: ALSBackend,
            convergence: Optional[ConvergencePolicy] = None,
            update_policy: Optional[UpdatePolicy] = None,
            observer: Optional[DecompositionObserver] = None,
            collect_metrics: bool = False) -> _ALSDriverResult:
        """Runs ALS until a normalized stopping reason is reached."""
        if not isinstance(problem, ALSProblem):
            raise TypeError('`problem` should be ALSProblem type')
        if convergence is None:
            convergence = ConvergencePolicy()
        elif not isinstance(convergence, ConvergencePolicy):
            raise TypeError('`convergence` should be ConvergencePolicy type')
        if update_policy is None:
            update_policy = UpdatePolicy()
        elif not isinstance(update_policy, UpdatePolicy):
            raise TypeError('`update_policy` should be UpdatePolicy type')
        if observer is not None:
            if not callable(getattr(observer, 'emit', None)) or \
                    not callable(getattr(observer, 'close', None)):
                raise TypeError('`observer` should implement `emit` and `close`')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        if isinstance(backend.n_sites, bool) or \
                (not isinstance(backend.n_sites, int)) or \
                (backend.n_sites < 1):
            raise ValueError('Backend `n_sites` should be a positive integer')

        convergence.validate_objective(problem.has_fixed_objective)
        trainable_sites = tuple(backend.trainable_sites)
        if any(isinstance(site, bool) or
               (not isinstance(site, int)) or
               (site < 0) or (site >= backend.n_sites)
               for site in trainable_sites):
            raise ValueError('Backend trainable sites are invalid')
        if len(set(trainable_sites)) != len(trainable_sites):
            raise ValueError('Backend trainable sites should be unique')

        metrics = DecompositionMetrics()
        if not trainable_sites:
            result = _ALSDriverResult(
                cores=tuple(backend.cores),
                metrics=metrics,
                converged=True,
                stop_reason='all_cores_fixed',
                n_sweeps=0)
            if observer is not None:
                observer.emit(DecompositionEvent(
                    name='summary',
                    phase='ALS',
                    values={
                        'n_sweeps': 0,
                        'stop_reason': 'all_cores_fixed',
                    }))
                observer.close(metrics)
            return result

        if observer is not None:
            observer.emit(DecompositionEvent(
                name='start',
                phase='ALS',
                values={
                    'n_sites': backend.n_sites,
                    'max_sweeps': convergence.max_sweeps,
                }))

        fixed_objective = problem.has_fixed_objective
        need_objective = fixed_objective and (
            collect_metrics or
            convergence.requires_fixed_objective or
            (convergence.callback is not None) or
            (observer is not None))
        need_sweep_record = collect_metrics or need_objective or \
            (convergence.callback is not None) or (observer is not None)
        need_local_records = collect_metrics or (observer is not None)

        previous_record = None
        stable_sweeps = 0
        best_score = None
        best_cores = None
        stop_reason = None
        completed_sweeps = 0

        for sweep in range(convergence.max_sweeps):
            order = tuple(range(backend.n_sites))
            if sweep % 2:
                order = tuple(reversed(order))
            start = perf_counter() if need_sweep_record else None
            generation, refreshed = backend.prepare_sweep(order, sweep)
            if observer is not None:
                observer.emit(DecompositionEvent(
                    name='sweep_start',
                    phase='ALS',
                    sweep=sweep,
                    values={
                        'direction': 'forward' if not sweep % 2 else 'reverse',
                        'sample_generation': generation,
                    }))
                if refreshed:
                    observer.emit(DecompositionEvent(
                        name='sample_refresh',
                        phase='ALS sampling',
                        sweep=sweep,
                        values={'sample_generation': generation}))

            for site in order:
                if site not in trainable_sites:
                    backend.skip_site(site)
                    continue
                try:
                    update_set, local_record = backend.solve_site(
                        site=site,
                        sweep=sweep,
                        update_policy=update_policy,
                        return_record=need_local_records)
                    backend.commit(update_set)
                except NonFiniteLocalSystemError:
                    stop_reason = 'nonfinite_local_system'
                    break
                except NonFiniteSolutionError:
                    stop_reason = 'nonfinite_solution'
                    break
                if collect_metrics and (local_record is not None):
                    metrics.local_solves.append(local_record)
                if observer is not None:
                    values = {'total_sites': backend.n_sites}
                    if local_record is not None:
                        values.update({
                            'residual_absolute':
                                local_record.residual_absolute,
                            'residual_relative':
                                local_record.residual_relative,
                        })
                    observer.emit(DecompositionEvent(
                        name='site_complete',
                        phase='ALS local solve',
                        level=2,
                        site=site,
                        sweep=sweep,
                        values=values))

            if stop_reason is not None:
                elapsed = None if start is None else perf_counter() - start
                record = SweepRecord(
                    sweep=sweep,
                    elapsed=elapsed,
                    sample_generation=generation,
                    stop_reason=stop_reason)
                if collect_metrics:
                    metrics.sweeps.append(record)
                break

            completed_sweeps = sweep + 1
            absolute_error = None
            relative_error = None
            if need_objective:
                absolute_tensor, relative_tensor = backend.measure_objective(
                    problem)
                if not isinstance(absolute_tensor, torch.Tensor) or \
                        absolute_tensor.numel() != 1:
                    raise TypeError(
                        'Backend objective absolute error should be a scalar '
                        'tensor')
                if not isinstance(relative_tensor, torch.Tensor) or \
                        relative_tensor.numel() != 1:
                    raise TypeError(
                        'Backend objective relative error should be a scalar '
                        'tensor')
                absolute_error = float(absolute_tensor.detach().cpu().item())
                relative_error = float(relative_tensor.detach().cpu().item())
                if (not isfinite(absolute_error)) or \
                        (not isfinite(relative_error)):
                    stop_reason = 'nonfinite_solution'

            relative_change = None
            if (previous_record is not None) and \
                    (absolute_error is not None):
                relative_change = _relative_objective_change(
                    previous_record, absolute_error, relative_error)
            elapsed = None if start is None else perf_counter() - start
            record = SweepRecord(
                sweep=sweep,
                absolute_error=absolute_error,
                relative_error=relative_error,
                relative_change=relative_change,
                elapsed=elapsed,
                sample_generation=generation)

            if (stop_reason is None) and (convergence.callback is not None):
                if convergence.callback(record, tuple(backend.cores)):
                    stop_reason = 'callback'
            if stop_reason is None:
                stop_reason, stable_sweeps = convergence.stopping_reason(
                    record, stable_sweeps)
            if stop_reason is not None:
                record = replace(record, stop_reason=stop_reason)

            if convergence.keep_best and (absolute_error is not None):
                score = relative_error if relative_error is not None \
                    else absolute_error
                if (best_score is None) or (score < best_score):
                    best_score = score
                    best_cores = tuple(core.clone() for core in backend.snapshot())

            if collect_metrics:
                metrics.sweeps.append(record)
            if observer is not None:
                observer.emit(DecompositionEvent(
                    name='sweep_complete',
                    phase='ALS',
                    sweep=sweep,
                    elapsed=elapsed,
                    values={
                        'absolute_error': absolute_error,
                        'relative_error': relative_error,
                        'relative_change': relative_change,
                        'sample_generation': generation,
                        'stop_reason': stop_reason,
                    }))
            previous_record = record
            if stop_reason is not None:
                break

        if stop_reason is None:
            stop_reason = 'max_sweeps'
        if convergence.keep_best and (best_cores is not None):
            backend.restore(best_cores)

        converged = stop_reason not in (
            'max_sweeps',
            'nonfinite_local_system',
            'nonfinite_solution',
        )
        result = _ALSDriverResult(
            cores=tuple(backend.cores),
            metrics=metrics,
            converged=converged,
            stop_reason=stop_reason,
            n_sweeps=completed_sweeps)
        if observer is not None:
            observer.emit(DecompositionEvent(
                name='summary',
                phase='ALS',
                values={
                    'n_sweeps': completed_sweeps,
                    'converged': converged,
                    'stop_reason': stop_reason,
                }))
            observer.close(metrics)
        return result


__all__ = ['ALSBackend', 'ALSSweepDriver']
