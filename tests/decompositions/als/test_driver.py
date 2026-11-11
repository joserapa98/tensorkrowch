"""Tests for generic ALS sweep coordination."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.als.solvers import (NonFiniteLocalSystemError,
                                                     NonFiniteSolutionError)
from tensorkrowch.decompositions.als.driver import _relative_objective_change
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
        self.initial_objective = 2.

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

    def update_site(self, site, sweep, return_record):
        if self.fail == ('system', sweep, site):
            raise NonFiniteLocalSystemError('forced system failure')
        if self.fail == ('solution', sweep, site):
            raise NonFiniteSolutionError('forced solution failure')
        proposal = self._cores[site] + 1
        version = self.versions[site] + 1
        self._cores[site] = proposal
        self.versions[site] = version
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
        return record

    def skip_site(self, site):
        self.skipped.append((self.current_sweep, site))

    def measure_objective(self, problem):
        self.measure_count += 1
        value = (self.initial_objective if self.current_sweep is None else
                 self.objective[min(self.current_sweep,
                                    len(self.objective) - 1)])
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

    @pytest.mark.parametrize('sampled', [False, True])
    @pytest.mark.parametrize('errors', [[.3, .4], [.2, .2]])
    def test_keep_best_includes_initialization_and_ties(self, sampled, errors):
        backend = _FakeALSBackend(errors)
        backend.initial_objective = .2
        initial = backend.snapshot()
        problem = _fixed_problem()
        problem.sampled = sampled
        result = tk.decompositions.ALSSweepDriver().fit(
            problem, backend,
            tk.decompositions.ConvergencePolicy(max_sweeps=2, keep_best=True))

        assert result.n_sweeps == 2
        assert backend.measure_count == 3
        for actual, expected in zip(result.cores, initial):
            torch.testing.assert_close(actual, expected)
        final, = result.metrics.errors
        assert final.kind == 'final'
        assert final.absolute == pytest.approx(.2)
        assert not result.metrics.sweeps

    @pytest.mark.parametrize('all_fixed', [False, True])
    def test_initial_state_survives_failure_or_no_updates(self, all_fixed):
        backend = _FakeALSBackend(
            [1.], trainable_sites=() if all_fixed else None,
            fail=None if all_fixed else ('system', 0, 1))
        initial = backend.snapshot()
        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(), backend,
            tk.decompositions.ConvergencePolicy(keep_best=True))
        assert result.n_sweeps == 0
        assert len(result.metrics.errors) == 1
        assert result.metrics.errors[0].kind == 'final'
        for actual, expected in zip(result.cores, initial):
            torch.testing.assert_close(actual, expected)

    def test_nonfinite_objective_cannot_replace_finite_initialization(self):
        backend = _FakeALSBackend([.1])
        initial = backend.snapshot()

        def measure(problem):
            if backend.current_sweep is None:
                return torch.tensor(.2), torch.tensor(.2)
            return torch.tensor(torch.inf), torch.tensor(torch.inf)

        backend.measure_objective = measure
        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(), backend,
            tk.decompositions.ConvergencePolicy(keep_best=True))
        assert result.stop_reason == 'nonfinite_solution'
        assert result.metrics.errors[-1].relative == pytest.approx(.2)
        for actual, expected in zip(result.cores, initial):
            torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize('engine', [tk.decompositions.TTALS,
                                        tk.decompositions.TRALS])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('objective', ['full', 'weighted', 'sampled',
                                           'observed'])
    def test_public_als_retains_exact_initialization_when_ridge_worsens_fit(
            self, engine, dtype, objective):
        target = torch.ones(3, 4, 2, dtype=dtype)
        cores = [torch.ones(1, dim, 1, dtype=dtype) for dim in target.shape]
        options = {}
        sampler = None
        if objective == 'weighted':
            options['weights'] = torch.linspace(.1, 2., 24).reshape(
                target.shape)
            options['mask'] = target.real > 0
        elif objective == 'observed':
            indices = torch.cartesian_prod(torch.arange(3), torch.arange(4),
                                           torch.arange(2))[::2]
            options['mask'] = tk.decompositions.SparseTensorSource(
                indices, torch.ones(len(indices), dtype=torch.bool),
                target.shape)
        elif objective == 'sampled':
            sampler = tk.decompositions.SamplingPolicy(8)
        result = engine(target, **options).fit(
            initial_cores=cores, orthogonalize=False, sampler=sampler,
            solver=tk.decompositions.LeastSquaresSolver(l2_reg=10.),
            convergence=tk.decompositions.ConvergencePolicy(
                max_sweeps=2, keep_best=True, error_samples=5),
            generator=torch.Generator().manual_seed(42), collect_metrics=True)

        torch.testing.assert_close(result.contract_dense(), target)
        final, = result.metrics.errors
        assert final.kind == 'final'
        assert final.absolute == 0
        assert result.metrics.sweeps[-1].abs_error > 0

    @pytest.mark.parametrize(
        'sweep_direction', [None, 'forward', 'alternating']
    )
    def test_sweep_order_and_observer_direction(self, sweep_direction):
        backend = _FakeALSBackend([1., 0.5, 0.25])
        history = HistoryObserver()
        options = {} if sweep_direction is None else {
            'sweep_direction': sweep_direction}

        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(), backend,
            tk.decompositions.ConvergencePolicy(max_sweeps=3),
            observer=history, collect_metrics=True, **options)

        forward = [0, 1, 2]
        middle = [2, 1, 0] if sweep_direction == 'alternating' else forward
        assert [event.site for event in history.events
                if event.name == 'site_complete'] == forward + middle + forward
        assert [
            event.values['direction']
            for event in history.events
            if event.name == 'sweep_start'
        ] == [
            'forward',
            'reverse' if sweep_direction == 'alternating' else 'forward',
            'forward',
        ]
        assert result.n_sweeps == 3

    @pytest.mark.parametrize('value, error', [
        (None, TypeError), (True, TypeError), ('reverse', ValueError)])
    def test_invalid_sweep_direction(self, value, error):
        backend = _FakeALSBackend([1.])

        with pytest.raises(error, match='sweep_direction'):
            tk.decompositions.ALSSweepDriver().fit(
                _fixed_problem(), backend, sweep_direction=value)

        assert backend.current_sweep is None

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

    @pytest.mark.parametrize('sweep_direction', ['forward', 'alternating'])
    def test_fixed_sites_are_skipped_in_both_directions(self, sweep_direction):
        backend = _FakeALSBackend([1., 0.5], trainable_sites=(0, 2))

        tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(),
            backend,
            tk.decompositions.ConvergencePolicy(max_sweeps=2),
            sweep_direction=sweep_direction)

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
            sampled=True)
        backend = _FakeALSBackend([1., 0.5])
        policy = tk.decompositions.ConvergencePolicy(
            max_sweeps=5,
            callback=lambda record, cores: record.sweep == 1)

        result = tk.decompositions.ALSSweepDriver().fit(
            problem, backend, policy)

        assert result.stop_reason == 'callback'
        assert result.n_sweeps == 2

    def test_keep_best_restores_complete_sweep_snapshot(self):
        backend = _FakeALSBackend([0.5, 0.1, 0.3])

        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(), backend,
            tk.decompositions.ConvergencePolicy(max_sweeps=3, keep_best=True),
            collect_metrics=True)

        assert result.n_sweeps == 3
        assert backend.restored is not None
        for site, core in enumerate(result.cores):
            torch.testing.assert_close(core, torch.tensor([site + 2.]))
        assert result.metrics.sweeps[-1].abs_error == pytest.approx(0.3)
        assert result.metrics.errors[-1].absolute == pytest.approx(.1)

    def test_infinite_relative_error_for_zero_target_is_not_solver_failure(
        self,
    ):
        backend = _FakeALSBackend([2., 1.])

        def measure_objective(problem):
            value = (backend.initial_objective if backend.current_sweep is None
                     else backend.objective[backend.current_sweep])
            return torch.tensor(value), torch.tensor(torch.inf)

        backend.measure_objective = measure_objective
        result = tk.decompositions.ALSSweepDriver().fit(
            _fixed_problem(), backend,
            tk.decompositions.ConvergencePolicy(max_sweeps=2, keep_best=True),
            collect_metrics=True)

        assert result.stop_reason == 'max_sweeps'
        assert result.n_sweeps == 2
        assert result.metrics.sweeps[-1].rel_change == 0.5
        assert torch.isinf(result.metrics.sweeps[-1].rel_error)
        for site, core in enumerate(result.cores):
            torch.testing.assert_close(core, torch.tensor([site + 2.]))

    @pytest.mark.parametrize('option, value, error', [
        ('problem', None, TypeError),
        ('convergence', object(), TypeError),
        ('observer', object(), TypeError),
        ('collect_metrics', 1, TypeError),
    ])
    def test_invalid_driver_options_fail_before_a_sweep(
        self, option, value, error
    ):
        backend = _FakeALSBackend([1.])
        options = {'problem': _fixed_problem(), 'backend': backend}
        options[option] = value

        with pytest.raises(error):
            tk.decompositions.ALSSweepDriver().fit(**options)

        assert backend.current_sweep is None

    @pytest.mark.parametrize('sites', [(0, 0), (-1,), (3,), (True,)])
    def test_invalid_backend_trainable_sites(self, sites):
        backend = _FakeALSBackend([1.], trainable_sites=sites)

        with pytest.raises(ValueError, match='trainable sites'):
            tk.decompositions.ALSSweepDriver().fit(_fixed_problem(), backend)

    def test_callback_exception_is_not_swallowed(self):
        backend = _FakeALSBackend([1.])

        def callback(record, cores):
            raise RuntimeError('callback failed')

        with pytest.raises(RuntimeError, match='callback failed'):
            tk.decompositions.ALSSweepDriver().fit(
                _fixed_problem(), backend,
                tk.decompositions.ConvergencePolicy(callback=callback))


@pytest.mark.parametrize('previous,current,expected', [
    (0., 0., 0.),
    (1e-30, 5e-31, 0.5),
    (1e30, 5e29, 0.5),
])
def test_relative_change_does_not_depend_on_absolute_scale(
        previous, current, expected):
    record = tk.decompositions.SweepRecord(
        sweep=0, abs_error=torch.tensor(previous, dtype=torch.float64))

    change = _relative_objective_change(
        record, torch.tensor(current, dtype=torch.float64), None)

    assert change == pytest.approx(expected)
