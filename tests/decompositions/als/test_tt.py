"""Tests for exact tensor-train alternating least squares."""

import pytest

import torch
import tensorkrowch as tk


def _exact_tt(dtype=torch.float64):
    """Returns a small heterogeneous TT and its dense contraction."""
    generator = torch.Generator().manual_seed(7)
    cores = [
        torch.randn(2, 2, dtype=dtype, generator=generator),
        torch.randn(2, 3, 2, dtype=dtype, generator=generator),
        torch.randn(2, 2, dtype=dtype, generator=generator),
    ]
    result = tk.decompositions.TTDecomposition(cores)
    return cores, result.contract_dense()


class TestTTALSExact:  # MARK: TestTTALSExact

    @pytest.mark.parametrize('gauge', ['none', 'qr', 'svd'])
    def test_svd_initialization_recovers_dense_tensor(self, gauge):
        _, tensor = _exact_tt()
        result = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                rank=2,
                init='svd',
                gauge=gauge,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=2),
                collect_metrics=True)

        assert isinstance(result, tk.decompositions.TTDecomposition)
        assert result.input_dim == tensor.shape
        assert result.rank == [2, 2]
        assert torch.allclose(
            result.contract_dense(), tensor, atol=1e-10, rtol=1e-10)
        assert len(result.metrics.sweeps) == 2
        assert result.metrics.sweeps[-1].absolute_error < 1e-10

    def test_exact_objective_is_monotone_up_to_roundoff(self):
        generator = torch.Generator().manual_seed(11)
        tensor = torch.randn(2, 3, 2, dtype=torch.float64,
                             generator=generator)
        result = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                rank=2,
                generator=torch.Generator().manual_seed(12),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=4),
                collect_metrics=True)

        errors = [record.absolute_error for record in result.metrics.sweeps]
        for previous, current in zip(errors, errors[1:]):
            tolerance = 1e-12 * max(1., previous)
            assert current <= previous + tolerance
        assert errors[-1] < 1e-10

    def test_complex_callable_uses_common_source_contract(self):
        dense = torch.tensor(
            [[[1 + 2j, 2 - 1j], [3j, -1 + 0.5j]],
             [[2 + 0j, 1j], [-2j, 4 - 1j]]],
            dtype=torch.complex128)

        def function(indices):
            return dense[indices[:, 0], indices[:, 1], indices[:, 2]]

        result = tk.decompositions.TTALS(
            function,
            input_dim=(2, 2, 2),
            dtype=torch.complex128,
            output_device=None).fit(
                rank=2,
                init='svd',
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1))

        assert result.dtype == torch.complex128
        assert torch.allclose(
            result.contract_dense(), dense, atol=1e-11, rtol=1e-11)

    def test_callable_target_is_cached_across_repeated_fits(self):
        dense = torch.arange(8., dtype=torch.float64).reshape(2, 2, 2)
        evaluations = []

        def function(indices):
            evaluations.append(indices.clone())
            return dense[indices[:, 0], indices[:, 1], indices[:, 2]]

        decomposition = tk.decompositions.TTALS(
            function,
            input_dim=(2, 2, 2),
            dtype=torch.float64,
            output_device=None)
        convergence = tk.decompositions.ConvergencePolicy(max_sweeps=1)

        first = decomposition.fit(
            rank=2, init='svd', convergence=convergence)
        second = decomposition.fit(
            rank=1, init='svd', convergence=convergence)

        assert len(evaluations) == 1
        assert first.rank == [2, 2]
        assert second.rank == [1, 1]

    def test_feasible_ranks_are_clipped_per_cut(self):
        tensor = torch.randn(2, 3, 4, dtype=torch.float64)
        result = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                rank=100,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1))

        assert result.rank == [2, 4]

    def test_one_site_tensor_uses_the_same_driver(self):
        tensor = torch.tensor([1., -2., 3.], dtype=torch.float64)
        result = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                rank=1,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
                collect_metrics=True)

        assert result.rank == []
        assert torch.allclose(result.cores[0], tensor)
        assert result.metrics.sweeps[0].absolute_error == pytest.approx(0.)

    def test_fixed_core_remains_bitwise_equal_and_blocks_absorption(self):
        initial, tensor = _exact_tt()
        fixed = initial[1].clone()
        fixed_cores = [None, fixed, None]
        result = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                initial_cores=initial,
                fixed_cores=fixed_cores,
                gauge='qr',
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=2))

        assert torch.equal(result.cores[1], fixed)
        assert result.metadata['fixed_sites'] == [1]
        assert torch.allclose(
            result.contract_dense(), tensor, atol=1e-10, rtol=1e-10)

    def test_all_fixed_cores_stop_without_updates(self):
        initial, tensor = _exact_tt()
        result = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                initial_cores=initial,
                fixed_cores=initial)

        assert result.metadata['converged']
        assert result.metadata['stop_reason'] == 'all_cores_fixed'
        assert result.metadata['n_sweeps'] == 0
        assert all(torch.equal(actual, expected)
                   for actual, expected in zip(result.cores, initial))

    def test_fast_path_does_not_collect_local_or_sweep_records(self):
        _, tensor = _exact_tt()
        result = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                rank=2,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
                collect_metrics=False)

        assert result.metrics.local_solves == []
        assert result.metrics.sweeps == []

    def test_relative_error_convergence_uses_exact_dense_objective(self):
        _, tensor = _exact_tt()
        result = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                rank=2,
                init='svd',
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=5, error_rtol=1e-10),
                collect_metrics=True)

        assert result.metadata['converged']
        assert result.metadata['stop_reason'] == 'error_rtol'
        assert result.metadata['n_sweeps'] == 1


class TestTTALSValidationAndWrapper:  # MARK: TestTTALSValidationAndWrapper

    def test_rank_is_required_without_initial_cores(self):
        decomposition = tk.decompositions.TTALS(torch.ones(2, 2))

        with pytest.raises(ValueError, match='rank.*required'):
            decomposition.fit()

    def test_initial_rank_above_cap_is_rejected(self):
        tensor = torch.randn(2, 2, dtype=torch.float64)
        initial = [
            torch.randn(2, 2, dtype=torch.float64),
            torch.randn(2, 2, dtype=torch.float64),
        ]

        with pytest.raises(ValueError, match='rank.*cap'):
            tk.decompositions.TTALS(tensor).fit(
                rank=1, initial_cores=initial)

    def test_vector_output_is_rejected_explicitly(self):
        tensor = torch.randn(2, 2, 3)
        source = tk.decompositions.DenseTensorSource(
            tensor, input_dim=(2, 2))

        with pytest.raises(ValueError, match='scalar tensor source'):
            tk.decompositions.TTALS(source).fit(rank=2)

    def test_functional_wrapper_returns_cores_and_optional_info(self):
        _, tensor = _exact_tt()
        cores = tk.decompositions.tt_als(
            tensor,
            rank=2,
            init='svd',
            max_sweeps=1,
            output_device=None)
        cores_info, info = tk.decompositions.tt_als(
            tensor,
            rank=2,
            init='svd',
            max_sweeps=1,
            output_device=None,
            return_info=True)

        assert len(cores) == len(cores_info) == tensor.ndim
        assert info['metadata']['algorithm'] == 'tt_als'
        assert len(info['metrics']['sweeps']) == 1
        assert torch.allclose(
            tk.decompositions.TTDecomposition(cores).contract_dense(),
            tensor,
            atol=1e-10,
            rtol=1e-10)

    def test_absolute_regularization_respects_environment_normalization(self):
        _, tensor = _exact_tt()
        initial = tk.decompositions.TTSVD(
            tensor, output_device=None).fit(rank=2)
        solver = tk.decompositions.LeastSquaresSolver(
            l2_reg=1e-4,
            l2_reg_mode='absolute')

        normalized = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                initial_cores=initial,
                solver=solver,
                gauge='none',
                renormalize=True,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1))
        direct = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                initial_cores=initial,
                solver=solver,
                gauge='none',
                renormalize=False,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1))

        assert torch.allclose(
            normalized.contract_dense(),
            direct.contract_dense(),
            atol=1e-9,
            rtol=1e-9)


class TestTTALSCompletion:  # MARK: TestTTALSCompletion

    def test_matrix_completion_minimizes_permanent_observations(self):
        left = torch.tensor([1., 2., -1.], dtype=torch.float64)
        right = torch.tensor([2., -1., 3., 4.], dtype=torch.float64)
        tensor = left[:, None] * right
        indices = torch.cartesian_prod(torch.arange(3), torch.arange(4))[:-1]
        values = tensor[indices[:, 0], indices[:, 1]]
        decomposition = tk.decompositions.TTALS.completion(
            indices,
            values,
            input_dim=tensor.shape,
            output_device=None)

        result = decomposition.fit(
            rank=1,
            generator=torch.Generator().manual_seed(50),
            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=20),
            collect_metrics=True)

        errors = [record.relative_error for record in result.metrics.sweeps]
        assert all(current <= previous + 1e-12
                   for previous, current in zip(errors, errors[1:]))
        assert errors[-1] < 1e-4
        assert result.metadata['sampling'] == 'observed'

    def test_unsorted_weighted_tensor_observations_define_objective(self):
        tensor = torch.arange(8., dtype=torch.float64).reshape(2, 2, 2)
        indices = torch.tensor([
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
        values = tensor[indices[:, 0], indices[:, 1], indices[:, 2]]
        weights = torch.tensor([2., 1., 0.5, 3.], dtype=torch.float64)
        observations = tk.decompositions.ObservedEntries(
            indices=indices,
            values=values,
            input_dim=tensor.shape,
            weights=weights)
        history = tk.decompositions.HistoryObserver()

        result = tk.decompositions.TTALS.completion(
            observations, output_device=None).fit(
                rank=2,
                generator=torch.Generator().manual_seed(51),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=3),
                collect_metrics=True,
                observer=history)

        approximation = result.evaluate(observations.indices)
        absolute, relative = observations.error(approximation)
        assert result.metrics.sweeps[-1].absolute_error == pytest.approx(
            absolute.item())
        assert result.metrics.sweeps[-1].relative_error == pytest.approx(
            relative.item())
        assert [event.sweep for event in history.events
                if event.name == 'sample_refresh'] == [0]

    def test_completion_rejects_exact_or_svd_initialization(self):
        decomposition = tk.decompositions.TTALS.completion(
            torch.tensor([[0, 0], [1, 1]]),
            torch.tensor([1., 2.]),
            input_dim=(2, 2))

        with pytest.raises(ValueError, match='permanently observed'):
            decomposition.fit(rank=2, sampling='exact')
        with pytest.raises(ValueError, match='requires an exact'):
            decomposition.fit(rank=2, init='svd')


class TestTTALSSampling:  # MARK: TestTTALSSampling

    def test_uniform_samples_and_values_are_reused_by_generation(self):
        tensor = torch.arange(16., dtype=torch.float64).reshape(2, 2, 2, 2)
        evaluations = []

        def function(indices):
            evaluations.append(indices.clone())
            return tensor[tuple(indices[:, site]
                                for site in range(indices.shape[1]))]

        history = tk.decompositions.HistoryObserver()
        result = tk.decompositions.TTALS(
            function,
            input_dim=tensor.shape,
            dtype=torch.float64,
            output_device=None).fit(
                rank=2,
                sampling='uniform',
                n_samples=12,
                sample_reuse_sweeps=2,
                generator=torch.Generator().manual_seed(52),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=3),
                collect_metrics=True,
                observer=history)

        assert len(evaluations) == 2
        assert [event.sweep for event in history.events
                if event.name == 'sample_refresh'] == [0, 2]
        assert [record.sample_generation
                for record in result.metrics.sweeps] == [0, 0, 1]
        assert all(record.absolute_error is None
                   for record in result.metrics.sweeps)
        assert result.metadata['exact_configurations'] is None

    def test_uniform_sampling_is_deterministic_with_generator(self):
        tensor = torch.randn(2, 3, 2, dtype=torch.float64)

        def fit(seed):
            return tk.decompositions.TTALS(
                tensor, output_device=None).fit(
                    rank=2,
                    sampling='uniform',
                    n_samples=9,
                    sample_reuse_sweeps=2,
                    generator=torch.Generator().manual_seed(seed),
                    convergence=tk.decompositions.ConvergencePolicy(
                        max_sweeps=3))

        first = fit(53)
        second = fit(53)

        assert all(torch.equal(first_core, second_core)
                   for first_core, second_core in zip(
                       first.cores, second.cores))

    def test_uniform_sampling_rejects_global_error_convergence(self):
        decomposition = tk.decompositions.TTALS(torch.ones(2, 2))

        with pytest.raises(ValueError, match='fixed global objective'):
            decomposition.fit(
                rank=2,
                sampling='uniform',
                n_samples=3,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=2, error_rtol=0.1))

    def test_functional_wrapper_supports_uniform_sampling(self):
        tensor = torch.randn(2, 2, 2, dtype=torch.float64)

        cores, info = tk.decompositions.tt_als(
            tensor,
            rank=2,
            sampling='uniform',
            n_samples=6,
            sample_reuse_sweeps=2,
            max_sweeps=2,
            generator=torch.Generator().manual_seed(54),
            output_device=None,
            return_info=True)

        assert len(cores) == tensor.ndim
        assert info['metadata']['sampling'] == 'uniform'
        assert [record['sample_generation']
                for record in info['metrics']['sweeps']] == [0, 0]


class TestTTALSLeverageSampling:  # MARK: TestTTALSLeverageSampling

    @pytest.mark.parametrize('gauge', ['qr', 'svd'])
    def test_exact_mode_redraws_current_design_at_every_site(self, gauge):
        tensor = torch.randn(2, 3, 2, dtype=torch.float64)
        result = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                rank=2,
                sampling='leverage',
                n_samples=12,
                gauge=gauge,
                leverage_uniform_mix=0.1,
                generator=torch.Generator().manual_seed(55),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=2),
                collect_metrics=True)

        assert result.metadata['sampling_exact']
        assert all(record.sampling_exact
                   for record in result.metrics.local_solves)
        assert [record.sample_generation
                for record in result.metrics.sweeps] == [0, 1]
        assert all(record.absolute_error is None
                   for record in result.metrics.sweeps)

    def test_frozen_mode_preserves_original_probabilities_and_marks_staleness(
            self):
        tensor = torch.randn(2, 3, 2, dtype=torch.float64)
        result = tk.decompositions.TTALS(
            tensor, output_device=None).fit(
                rank=2,
                sampling='leverage',
                n_samples=10,
                leverage_mode='frozen',
                sample_reuse_sweeps=2,
                generator=torch.Generator().manual_seed(56),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=3),
                collect_metrics=True)

        exactness = [record.sampling_exact
                     for record in result.metrics.local_solves]
        generations = [record.sample_generation
                       for record in result.metrics.local_solves]
        assert exactness == [True, True, True, False, False, False,
                             True, True, True]
        assert generations == [0, 0, 0, 0, 0, 0, 1, 1, 1]
        assert not result.metadata['sampling_exact']

    def test_leverage_evaluates_only_site_batches(self):
        tensor = torch.randn(2, 2, 2, dtype=torch.float64)
        evaluations = []

        def function(indices):
            evaluations.append(indices.clone())
            return tensor[tuple(indices[:, site]
                                for site in range(indices.shape[1]))]

        tk.decompositions.TTALS(
            function,
            input_dim=tensor.shape,
            dtype=torch.float64).fit(
                rank=2,
                sampling='leverage',
                n_samples=7,
                leverage_mode='frozen',
                sample_reuse_sweeps=2,
                generator=torch.Generator().manual_seed(57),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=3))

        assert len(evaluations) == 2 * tensor.ndim
        assert all(values.shape == (7, tensor.ndim)
                   for values in evaluations)

    def test_leverage_requires_canonical_gauges_and_consistent_reuse(self):
        decomposition = tk.decompositions.TTALS(torch.ones(2, 2, 2))

        with pytest.raises(ValueError, match='QR or SVD'):
            decomposition.fit(
                rank=2, sampling='leverage', n_samples=5, gauge='none')
        with pytest.raises(ValueError, match='reuse_sweeps=1'):
            decomposition.fit(
                rank=2,
                sampling='leverage',
                n_samples=5,
                sample_reuse_sweeps=2)

    def test_leverage_rejects_fixed_cores(self):
        initial, tensor = _exact_tt()

        with pytest.raises(ValueError, match='does not support fixed cores'):
            tk.decompositions.TTALS(
                tensor, output_device=None).fit(
                    initial_cores=initial,
                    fixed_cores=[None, initial[1], None],
                    sampling='leverage',
                    n_samples=8,
                    convergence=tk.decompositions.ConvergencePolicy(
                        max_sweeps=1))
