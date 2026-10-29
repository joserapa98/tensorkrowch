"""Tests for als/convergence."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.als.solvers import (NonFiniteLocalSystemError,
                                                     NonFiniteSolutionError)


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

    def solve_site(self, site, sweep, return_record):
        if self.fail == ('system', sweep, site):
            raise NonFiniteLocalSystemError('forced system failure')
        if self.fail == ('solution', sweep, site):
            raise NonFiniteSolutionError('forced solution failure')
        proposal = self._cores[site] + 1
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


class TestConvergencePolicy:  # MARK: TestConvergencePolicy

    def test_error_criteria_use_end_of_sweep_objective(self):
        backend = _FakeALSBackend([1., 0.2, 0.01])
        driver = tk.decompositions.ALSSweepDriver()
        policy = tk.decompositions.ConvergencePolicy(
            max_sweeps=10, rtol=0.05)

        result = driver.fit(
            _fixed_problem(), backend, policy, collect_metrics=True)

        assert result.stop_reason == 'rtol'
        assert result.converged
        assert result.n_sweeps == 3
        assert [record.rel_error for record in result.metrics.sweeps] == [
            1., pytest.approx(0.2), pytest.approx(0.01)]
        assert all(record.rel_residual == 100
                   for record in result.metrics.local_solves)

    def test_relative_stability_counts_complete_sweeps(self):
        backend = _FakeALSBackend([1., 0.8, 0.8001, 0.8002])
        policy = tk.decompositions.ConvergencePolicy(
            max_sweeps=10,
            change_rtol=1e-3,
            patience=2)

        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(), backend, policy, collect_metrics=True)

        assert result.stop_reason == 'relative_stability'
        assert result.n_sweeps == 4
        assert result.metrics.sweeps[0].rel_change is None
        assert result.metrics.sweeps[-1].rel_change < 1e-3

    def test_renewable_sampling_rejects_incomparable_error_criteria(self):
        problem = tk.decompositions.ALSProblem(
            source=tk.decompositions.DenseTensorSource(torch.ones(2, 2)),
            sampled=True)
        backend = _FakeALSBackend([1., 0.5])

        with pytest.raises(ValueError, match='fixed global objective'):
            tk.decompositions.ALSSweepDriver().fit(
                problem,
                backend,
                tk.decompositions.ConvergencePolicy(
                    max_sweeps=2, rtol=0.1))

        result = tk.decompositions.ALSSweepDriver().fit(
            problem,
            backend,
            tk.decompositions.ConvergencePolicy(max_sweeps=2))
        assert result.stop_reason == 'max_sweeps'
        assert backend.measure_count == 0

    def test_completion_is_a_fixed_objective(self):
        observations = tk.decompositions.SparseTensorSource(
            indices=torch.tensor([[0, 0], [1, 1]]),
            values=torch.tensor([1., 2.]),
            in_dim=(2, 2))
        problem = tk.decompositions.ALSProblem(observations=observations)
        backend = _FakeALSBackend([0.5, 0.01])

        result = tk.decompositions.ALSSweepDriver().fit(
            problem,
            backend,
            tk.decompositions.ConvergencePolicy(
                max_sweeps=5, atol=0.05),
            collect_metrics=True)

        assert result.stop_reason == 'atol'
        assert result.n_sweeps == 2

    def test_best_state_is_restored_only_for_fixed_objectives(self):
        backend = _FakeALSBackend([1., 0.2, 0.5])
        policy = tk.decompositions.ConvergencePolicy(
            max_sweeps=3, keep_best=True)

        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(), backend, policy, collect_metrics=True)

        assert result.stop_reason == 'max_sweeps'
        assert backend.restored is not None
        assert all(torch.equal(core, torch.tensor([float(site + 2)]))
                   for site, core in enumerate(result.cores))


@pytest.mark.parametrize('kwargs, error', [
    ({'max_sweeps': 0}, ValueError), ({'max_sweeps': True}, ValueError),
    ({'max_sweeps': 2.5}, ValueError),
    ({'atol': -1.}, ValueError), ({'atol': float('inf')}, ValueError),
    ({'atol': True}, TypeError), ({'rtol': 'small'}, TypeError),
    ({'rtol': float('nan')}, ValueError),
    ({'change_rtol': -0.1}, ValueError),
    ({'patience': 0}, ValueError), ({'patience': 2}, ValueError),
    ({'patience': True, 'change_rtol': 0.1}, ValueError),
    ({'keep_best': 1}, TypeError), ({'callback': 1}, TypeError),
])
def test_policy_rejects_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        tk.decompositions.ConvergencePolicy(**kwargs)


@pytest.mark.parametrize('kwargs', [
    {'atol': 0.}, {'rtol': 0.}, {'change_rtol': 0.}, {'keep_best': True},
])
def test_fixed_objective_requirement(kwargs):
    policy = tk.decompositions.ConvergencePolicy(**kwargs)
    assert policy.requires_fixed_objective
    policy.validate_objective(True)
    with pytest.raises(ValueError, match='fixed'):
        policy.validate_objective(False)
    with pytest.raises(TypeError, match='has_fixed_objective'):
        policy.validate_objective(1)


def test_default_policy_and_explicit_budget():
    policy = tk.decompositions.ConvergencePolicy()
    assert policy.max_sweeps == 10
    assert not policy.requires_fixed_objective
    policy.validate_objective(False)
    record = tk.decompositions.SweepRecord(sweep=9)
    assert policy.stopping_reason(record, 0) == ('max_sweeps', 0)


def test_stability_restarts_after_large_change():
    policy = tk.decompositions.ConvergencePolicy(
        max_sweeps=10, change_rtol=0.01, patience=2)
    records = [tk.decompositions.SweepRecord(sweep=site, rel_change=value)
               for site, value in enumerate([None, 0.001, 0.2, 0.001, 0.001])]
    stable = 0
    observed = []
    for record in records:
        reason, stable = policy.stopping_reason(record, stable)
        observed.append((reason, stable))
    assert observed == [(None, 0), (None, 1), (None, 0), (None, 1),
                        ('relative_stability', 2)]


def test_stability_defaults_to_one_sweep():
    policy = tk.decompositions.ConvergencePolicy(change_rtol=0.)
    record = tk.decompositions.SweepRecord(sweep=1, rel_change=0.)
    assert policy.stopping_reason(record, 0) == ('relative_stability', 1)


def test_callback_stops_sampled_objective_without_measuring_global_error():
    seen = []

    def callback(record, cores):
        seen.append((record.sweep, tuple(core.clone() for core in cores)))
        return record.sweep == 1

    problem = tk.decompositions.ALSProblem(
        tk.decompositions.DenseTensorSource(torch.ones(2)), sampled=True)
    backend = _FakeALSBackend([1.])
    result = tk.decompositions.ALSSweepDriver().fit(
        problem, backend,
        tk.decompositions.ConvergencePolicy(max_sweeps=8, callback=callback),
        collect_metrics=True)
    assert result.stop_reason == 'callback'
    assert result.n_sweeps == 2
    assert backend.measure_count == 0
    assert [sweep for sweep, _ in seen] == [0, 1]
    assert all(record.abs_error is None for record in result.metrics.sweeps)
