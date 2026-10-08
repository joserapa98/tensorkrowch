"""Tests retained for redistribution."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.als.solvers import (NonFiniteLocalSystemError,
                                                     NonFiniteSolutionError)
from tensorkrowch.decompositions.observers import HistoryObserver


class _FakeALSBackend:
    """Small deterministic backend used to isolate driver semantics."""

    def __init__(self,
                 objective,
                 n_sites=3,
                 trainable_sites=None,
                 fail=None):
        self._n_sites = n_sites
        self._trainable_sites = tuple(range(n_sites)) \
            if trainable_sites is None else tuple(trainable_sites)
        self._cores = [torch.tensor([float(site)])
                       for site in range(n_sites)]
        self.versions = [0] * n_sites
        self.objective = list(objective)
        self.fail = fail
        self.current_sweep = None
        self.measure_count = 0
        self.skipped = []
        self.restored = None
        self.previous_generation = None

    @property
    def n_sites(self):
        return self._n_sites

    @property
    def trainable_sites(self):
        return self._trainable_sites

    @property
    def cores(self):
        return self._cores

    def prepare_sweep(self, order, sweep):
        self.current_sweep = sweep
        generation = sweep // 2
        refreshed = generation != self.previous_generation
        self.previous_generation = generation
        return generation, refreshed

    def solve_site(self, site, sweep, update_policy, return_record):
        if self.fail == ('system', sweep, site):
            raise NonFiniteLocalSystemError('forced system failure')
        if self.fail == ('solution', sweep, site):
            raise NonFiniteSolutionError('forced solution failure')
        proposal = self._cores[site] + 1
        proposal = update_policy.apply(self._cores[site], proposal)
        version = self.versions[site] + 1
        update = tk.decompositions.CoreUpdateSet(
            sites=(site,), cores=(proposal,), versions=(version,))
        if return_record:
            record = tk.decompositions.LocalSolveRecord(
                environment_shape=(4, 1),
                target_shape=(4,),
                driver='fake',
                abs_residual=100,
                rel_residual=100,
                target_norm=1,
                site=site,
                sweep=sweep)
        else:
            record = None
        return update, record

    def skip_site(self, site):
        self.skipped.append((self.current_sweep, site))

    def commit(self, update_set):
        for site, (core, version) in update_set.updates.items():
            self._cores[site] = core
            self.versions[site] = version

    def measure_objective(self, problem):
        self.measure_count += 1
        value = self.objective[min(self.current_sweep,
                                   len(self.objective) - 1)]
        return torch.tensor(float(value)), torch.tensor(float(value))

    def snapshot(self):
        return tuple(core.clone() for core in self._cores)

    def restore(self, cores):
        self._cores = [core.clone() for core in cores]
        self.restored = tuple(core.clone() for core in cores)


def _fixed_problem():
    """Returns a tiny problem with a comparable dense objective."""
    return tk.decompositions.ALSProblem(
        source=tk.decompositions.DenseTensorSource(torch.ones(2, 2)))


class TestALSSweepDriver:  # MARK: TestALSSweepDriver

    def test_fast_path_skips_objective_records_and_observers(self):
        backend = _FakeALSBackend([1., 0.5])

        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(),
            backend,
            tk.decompositions.ConvergencePolicy(max_sweeps=2),
            collect_metrics=False)

        assert result.stop_reason == 'max_sweeps'
        assert backend.measure_count == 0
        assert result.metrics.as_info() == {
            'errors': [],
            'truncations': [],
            'timings': [],
            'fidelities': [],
            'warnings': [],
        }

    def test_all_fixed_cores_stop_without_a_sweep(self):
        backend = _FakeALSBackend([1.], trainable_sites=())

        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(), backend)

        assert result.converged
        assert result.stop_reason == 'all_cores_fixed'
        assert result.n_sweeps == 0

    def test_fixed_sites_are_skipped_in_both_directions(self):
        backend = _FakeALSBackend([1., 0.5], trainable_sites=(0, 2))

        tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(),
            backend,
            tk.decompositions.ConvergencePolicy(max_sweeps=2))

        assert backend.skipped == [(0, 1), (1, 1)]

    def test_refresh_events_do_not_reset_fixed_objective_history(self):
        backend = _FakeALSBackend([1., 0.8, 0.7])
        history = HistoryObserver()

        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(),
            backend,
            tk.decompositions.ConvergencePolicy(max_sweeps=3),
            observer=history,
            collect_metrics=True)

        refresh_sweeps = [event.sweep for event in history.events
                          if event.name == 'sample_refresh']
        assert refresh_sweeps == [0, 2]
        assert result.metrics.sweeps[2].rel_change == pytest.approx(0.125)
        assert history.metrics is None  # The public engine closes its observer

    @pytest.mark.parametrize(
        'failure, reason',
        [(('system', 0, 1), 'nonfinite_local_system'),
         (('solution', 0, 1), 'nonfinite_solution')])
    def test_nonfinite_failures_have_normalized_reasons(self, failure, reason):
        backend = _FakeALSBackend([1.], fail=failure)

        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(), backend, collect_metrics=True)

        assert not result.converged
        assert result.stop_reason == reason
        assert result.n_sweeps == 0
        assert result.metrics.sweeps[-1].stop_reason == reason

    def test_callback_can_stop_without_a_fixed_objective(self):
        problem = tk.decompositions.ALSProblem(
            source=tk.decompositions.DenseTensorSource(torch.ones(2, 2)),
            selector=object())
        backend = _FakeALSBackend([1., 0.5])
        policy = tk.decompositions.ConvergencePolicy(
            max_sweeps=5,
            callback=lambda record, cores: record.sweep == 1)

        result = tk.decompositions.ALSSweepDriver().fit(
            problem, backend, policy)

        assert result.stop_reason == 'callback'
        assert result.n_sweeps == 2
