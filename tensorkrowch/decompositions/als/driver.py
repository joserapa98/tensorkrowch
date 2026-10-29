"""
This script contains:

    Internal classes:
        * _ALSDriverResult

    Public classes:
        * ALSBackend
        * ALSSweepDriver

    Internal functions:
        * _relative_objective_change
        * _report_als_result
"""

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Optional, Protocol, Sequence, Tuple, TYPE_CHECKING

import torch

from tensorkrowch.decompositions.als.convergence import ConvergencePolicy
from tensorkrowch.decompositions.als.environments import CoreUpdateSet
from tensorkrowch.decompositions.als.problem import ALSProblem
from tensorkrowch.decompositions.als.solvers import (NonFiniteLocalSystemError,
                                                     NonFiniteSolutionError)
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 LocalSolveRecord,
                                                 SweepRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   DecompositionObserver)


if TYPE_CHECKING:
    from tensorkrowch.decompositions.results import TensorDecomposition


class ALSBackend(Protocol):  # MARK: ALSBackend
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
                   return_record: bool) -> Tuple[
            CoreUpdateSet, Optional[LocalSolveRecord]]:
        """Builds one atomic local update without committing it."""

    def skip_site(self, site: int) -> None:
        """Advances caches through a fixed ``site``."""

    def commit(self, update_set: CoreUpdateSet) -> None:
        """Commits a topology-specific update atomically."""

    def measure_objective(self,
                          problem: ALSProblem) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """Measures one fixed/global objective after a complete sweep."""

    def snapshot(self) -> Sequence[torch.Tensor]:
        """Returns the current state for optional best-state retention."""

    def restore(self, cores: Sequence[torch.Tensor]) -> None:
        """Restores a previously captured state."""


@dataclass(frozen=True)
class _ALSDriverResult:  # MARK: _ALSDriverResult
    """Internal result returned from the generic ALS sweep driver."""

    cores: Tuple[torch.Tensor, ...]  # Raw cores in site order
    metrics: DecompositionMetrics  # Collected solve and sweep measurements
    converged: bool  # Whether the normalized stop reason denotes convergence
    stop_reason: str  # Normalized reason for stopping
    n_sweeps: int  # Number of completed directional sweeps


def _relative_objective_change(previous: SweepRecord,
                               current_absolute: torch.Tensor,
                               current_relative: Optional[
                                   torch.Tensor]) -> Optional[torch.Tensor]:
    """Computes relative change between complete comparable objectives."""
    if previous.abs_error is not None:
        old = previous.abs_error
        new = current_absolute
    elif (previous.rel_error is not None) and \
            (current_relative is not None) and \
            torch.isfinite(previous.rel_error) and \
            torch.isfinite(current_relative):
        old = previous.rel_error
        new = current_relative
    else:
        return None
    scale = torch.maximum(old.abs(), new.abs())
    return abs(old - new) / torch.where(scale > 0, scale, 1)


class ALSSweepDriver:  # MARK: ALSSweepDriver
    """
    Runs directional sweeps without topology-specific contractions.

    The backend owns environment construction and local solves. This driver
    chooses sweep directions, commits atomic updates, measures only complete
    sweep objectives, applies convergence and emits structured events.
    """

    def fit(self,
            problem: ALSProblem,
            backend: ALSBackend,
            convergence: Optional[ConvergencePolicy] = None,
            observer: Optional[DecompositionObserver] = None,
            collect_metrics: bool = False,
            phase: str = 'ALS',
            *,
            sweep_direction: str = 'forward') -> _ALSDriverResult:
        """
        Runs ALS until a normalized stopping reason is reached.

        ``sweep_direction="forward"`` visits sites in increasing order on
        every sweep. ``"alternating"`` reverses that order on odd sweeps.
        Each directional pass counts once toward ``max_sweeps``.

        Parameters
        ----------
        problem : ALSProblem
            Target values, residual weights and fixed-objective semantics.
        backend : ALSBackend
            Owns the cores, local solves and environment updates. Each local
            solution replaces the active core before moving its gauge.
        convergence : ConvergencePolicy, optional
            Complete-sweep stopping criteria. Defaults to ten sweeps.
        observer : DecompositionObserver, optional
            Receives structured progress events. Its lifecycle belongs to the
            calling decomposition, which closes it after finalizing results.
        collect_metrics : bool
            Whether to retain local solve and sweep records. Defaults to False.
        phase : str
            Label attached to progress events. Defaults to ``"ALS"``.
        sweep_direction : {``"forward"``, ``"alternating"``}
            Internal traversal selected by the decomposition: alternating
            directions for TT and forward passes for TR.

        Returns
        -------
        _ALSDriverResult
            Final cores, optional metrics, completed sweep count and stopping
            reason. Stagnation alone does not certify a small residual.
        """
        if not isinstance(sweep_direction, str):
            raise TypeError('`sweep_direction` should be str type')
        if sweep_direction not in ('forward', 'alternating'):
            raise ValueError(
                "`sweep_direction` should be 'forward' or 'alternating'")
        if not isinstance(problem, ALSProblem):
            raise TypeError('`problem` should be ALSProblem type')
        if convergence is None:
            convergence = ConvergencePolicy()
        elif not isinstance(convergence, ConvergencePolicy):
            raise TypeError('`convergence` should be ConvergencePolicy type')
        if observer is not None:
            if not callable(getattr(observer, 'emit', None)) or \
                    not callable(getattr(observer, 'close', None)):
                raise TypeError(
                    '`observer` should implement `emit` and `close`')
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
            return result

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
            reverse = (sweep_direction == 'alternating') and (sweep % 2 == 1)
            if reverse:
                order = tuple(reversed(order))
            start = perf_counter() if need_sweep_record else None
            generation, refreshed = backend.prepare_sweep(order, sweep)
            if observer is not None:
                observer.emit(DecompositionEvent(
                    name='sweep_start',
                    phase=phase,
                    sweep=sweep,
                    values={
                        'direction': 'reverse' if reverse else 'forward',
                        'sample_generation': generation,
                    }))
                if refreshed:
                    observer.emit(DecompositionEvent(
                        name='sample_refresh',
                        phase=f'{phase} sampling',
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
                            'abs_residual':
                                local_record.abs_residual,
                            'rel_residual':
                                local_record.rel_residual,
                        })
                    observer.emit(DecompositionEvent(
                        name='site_complete',
                        phase=f'{phase} local solve',
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
            if not need_sweep_record:
                if completed_sweeps == convergence.max_sweeps:
                    stop_reason = 'max_sweeps'
                    break
                continue

            abs_error = None
            rel_error = None
            if need_objective:
                absolute_tensor, relative_tensor = backend.measure_objective(
                    problem)
                if not isinstance(absolute_tensor, torch.Tensor) or \
                        (absolute_tensor.numel() != 1):
                    raise TypeError(
                        'Backend objective absolute error should be a scalar '
                        'tensor')
                if not isinstance(relative_tensor, torch.Tensor) or \
                        (relative_tensor.numel() != 1):
                    raise TypeError(
                        'Backend objective relative error should be a scalar '
                        'tensor')
                abs_error = absolute_tensor.detach().cpu()
                rel_error = relative_tensor.detach().cpu()
                if (not torch.isfinite(abs_error)) or torch.isnan(rel_error):
                    stop_reason = 'nonfinite_solution'

            rel_change = None
            if (previous_record is not None) and \
                    (abs_error is not None):
                rel_change = _relative_objective_change(
                    previous_record, abs_error, rel_error)
            elapsed = None if start is None else perf_counter() - start
            record = SweepRecord(
                sweep=sweep,
                abs_error=abs_error,
                rel_error=rel_error,
                rel_change=rel_change,
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

            if convergence.keep_best and (abs_error is not None):
                score = abs_error
                if (best_score is None) or (score < best_score):
                    best_score = score
                    best_cores = tuple(
                        core.clone() for core in backend.snapshot())

            if collect_metrics:
                metrics.sweeps.append(record)
            if observer is not None:
                observer.emit(DecompositionEvent(
                    name='sweep_complete',
                    phase=phase,
                    sweep=sweep,
                    elapsed=elapsed,
                    values={
                        'absolute_error': abs_error,
                        'relative_error': rel_error,
                        'relative_change': rel_change,
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
        return result


def _report_als_result(result: 'TensorDecomposition',
                       observer: DecompositionObserver,
                       phase: str) -> None:
    """Reports the final approximation after its cores have been finalized."""
    values = {
        'rank': result.rank,
        'n_sweeps': result.metadata['n_sweeps'],
        'converged': result.metadata['converged'],
        'stop_reason': result.metadata['stop_reason'],
    }
    if result.metrics.sweeps:
        # A restored best state need not be the last recorded sweep
        records = [record for record in result.metrics.sweeps
                   if record.abs_error is not None]
        if records:
            record = (min(records, key=lambda item: item.abs_error)
                      if result.metadata.get('keep_best') else records[-1])
            values.update(absolute_error=record.abs_error,
                          relative_error=record.rel_error)
        values['elapsed'] = sum(record.elapsed or 0
                                for record in result.metrics.sweeps)
    observer.emit(DecompositionEvent(
        name='summary', phase=phase, values=values))
    for site, core in enumerate(result.cores):
        observer.emit(DecompositionEvent(
            name='core', phase=phase, level=3, site=site,
            values={'shape': tuple(core.shape), 'tensor': core}))
    observer.close(result.metrics)


__all__ = [
    'ALSBackend',
    'ALSSweepDriver',
]
