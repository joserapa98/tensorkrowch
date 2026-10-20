"""Tests for exact tensor train alternating least squares."""

import pytest
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions.als.problem import _ObservedEntries

from tensorkrowch.decompositions.observers import HistoryObserver


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
            tensor, out_device=None).fit(
                rank=2,
                init='svd',
                gauge=gauge,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=2),
                collect_metrics=True)

        assert isinstance(result, tk.decompositions.TTDecomposition)
        assert result.in_dim == tensor.shape
        assert result.rank == [2, 2]
        assert torch.allclose(
            result.contract_dense(), tensor, atol=1e-10, rtol=1e-10)
        assert len(result.metrics.sweeps) == 2
        assert result.metrics.sweeps[-1].abs_error < 1e-10

    def test_exact_objective_is_monotone_up_to_roundoff(self):
        generator = torch.Generator().manual_seed(11)
        tensor = torch.randn(2, 3, 2, dtype=torch.float64,
                             generator=generator)
        result = tk.decompositions.TTALS(
            tensor, out_device=None).fit(
                rank=2,
                generator=torch.Generator().manual_seed(12),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=4),
                collect_metrics=True)

        errors = [record.abs_error for record in result.metrics.sweeps]
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

        result = tk.decompositions.TTALS(tk.decompositions.CallableTensorSource(function, in_dim=(2, 2, 2), dtype=torch.complex128, out_shape=()), out_device=None).fit(
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

        decomposition = tk.decompositions.TTALS(tk.decompositions.CallableTensorSource(function, in_dim=(2, 2, 2), dtype=torch.float64, out_shape=()), out_device=None)
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
            tensor, out_device=None).fit(
                rank=100,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1))

        assert result.rank == [2, 4]

    def test_one_site_tensor_uses_the_same_driver(self):
        tensor = torch.tensor([1., -2., 3.], dtype=torch.float64)
        result = tk.decompositions.TTALS(
            tensor, out_device=None).fit(
                rank=1,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
                collect_metrics=True)

        assert result.rank == []
        assert torch.allclose(result.cores[0], tensor)
        assert result.metrics.sweeps[0].abs_error == pytest.approx(0.)

    def test_fixed_core_remains_bitwise_equal_and_blocks_absorption(self):
        initial, tensor = _exact_tt()
        fixed = initial[1].clone()
        fixed_cores = [None, fixed, None]
        result = tk.decompositions.TTALS(
            tensor, out_device=None).fit(
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
            tensor, out_device=None).fit(
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
            tensor, out_device=None).fit(
                rank=2,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
                collect_metrics=False)

        assert result.metrics.local_solves == []
        assert result.metrics.sweeps == []

    def test_relative_error_convergence_uses_exact_dense_objective(self):
        _, tensor = _exact_tt()
        result = tk.decompositions.TTALS(
            tensor, out_device=None).fit(
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

    def test_vector_output_uses_response_sites(self):
        tensor = torch.arange(12., dtype=torch.float64).reshape(2, 2, 3)
        source = tk.decompositions.DenseTensorSource(tensor, in_sites=(0, 1))
        result = tk.decompositions.TTALS(source).fit(rank=3, init='svd')
        assert result.in_sites == (0, 1) and result.out_shape == (3,)
        torch.testing.assert_close(result.contract_dense(), tensor)

    def test_functional_wrapper_returns_cores_and_optional_info(self):
        _, tensor = _exact_tt()
        cores = tk.decompositions.tt_als(
            tensor,
            rank=2,
            init='svd',
            max_sweeps=1,
            out_device=None)
        cores_info, info = tk.decompositions.tt_als(
            tensor,
            rank=2,
            init='svd',
            max_sweeps=1,
            out_device=None,
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
            tensor, out_device=None).fit(rank=2)
        solver = tk.decompositions.LeastSquaresSolver(
            l2_reg=1e-4,
            l2_reg_mode='absolute')

        normalized = tk.decompositions.TTALS(
            tensor, out_device=None).fit(
                initial_cores=initial,
                solver=solver,
                gauge='none',
                renormalize=True,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1))
        direct = tk.decompositions.TTALS(
            tensor, out_device=None).fit(
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
        decomposition = tk.decompositions.TTALS(tk.decompositions.SparseTensorSource(indices, values, tensor.shape), mask=tk.decompositions.SparseTensorSource(indices, torch.ones(indices.shape[0], dtype=torch.bool, device=indices.device), tensor.shape), out_device=None)

        result = decomposition.fit(
            rank=1,
            generator=torch.Generator().manual_seed(50),
            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=20),
            collect_metrics=True)

        errors = [record.rel_error for record in result.metrics.sweeps]
        assert all(current <= previous + 1e-12
                   for previous, current in zip(errors, errors[1:]))
        assert errors[-1] < 1e-4
        assert result.metadata['sampling'] == 'observed'

    def test_unsorted_weighted_tensor_observations_define_objective(self, monkeypatch):
        tensor = torch.arange(8., dtype=torch.float64).reshape(2, 2, 2)
        indices = torch.tensor([
            [1, 1, 1],
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
        values = tensor[indices[:, 0], indices[:, 1], indices[:, 2]]
        weights = torch.tensor([2., 1., 0.5, 3.], dtype=torch.float64)
        observations = _ObservedEntries(
            indices=indices,
            values=values,
            in_dim=tensor.shape,
            weights=weights)
        history = HistoryObserver()
        from importlib import import_module
        monkeypatch.setattr(
            import_module('tensorkrowch.decompositions.als.tt'),
            '_resolve_observer', lambda *args: history)

        result = tk.decompositions.TTALS(tk.decompositions.SparseTensorSource(observations.indices, observations.values, observations.in_dim), mask=tk.decompositions.SparseTensorSource(observations.indices, torch.ones(observations.indices.shape[0], dtype=torch.bool, device=observations.indices.device), observations.in_dim), weights=None if observations.weights is None else tk.decompositions.SparseTensorSource(observations.indices, observations.weights, observations.in_dim).contract_dense(), out_device=None).fit(
                rank=2,
                generator=torch.Generator().manual_seed(51),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=3),
                collect_metrics=True,
                verbose=1)

        approximation = result.evaluate(observations.indices)
        absolute, relative = observations.error(approximation)
        assert result.metrics.sweeps[-1].abs_error == pytest.approx(
            absolute.item())
        assert result.metrics.sweeps[-1].rel_error == pytest.approx(
            relative.item())
        assert [event.sweep for event in history.events
                if event.name == 'sample_refresh'] == [0]

    def test_completion_rejects_exact_or_svd_initialization(self):
        decomposition = tk.decompositions.TTALS(tk.decompositions.SparseTensorSource(torch.tensor([[0, 0], [1, 1]]), torch.tensor([1., 2.]), (2, 2)), mask=tk.decompositions.SparseTensorSource(torch.tensor([[0, 0], [1, 1]]), torch.ones(torch.tensor([[0, 0], [1, 1]]).shape[0], dtype=torch.bool, device=torch.tensor([[0, 0], [1, 1]]).device), (2, 2)), out_device='cpu')

        with pytest.raises(ValueError, match='permanently observed'):
            decomposition.fit(rank=2, sampling='exact')
        with pytest.raises(ValueError, match='requires an exact'):
            decomposition.fit(rank=2, init='svd')


class TestTTALSSampling:  # MARK: TestTTALSSampling

    def test_uniform_samples_and_values_are_reused_by_generation(self, monkeypatch):
        tensor = torch.arange(16., dtype=torch.float64).reshape(2, 2, 2, 2)
        evaluations = []

        def function(indices):
            evaluations.append(indices.clone())
            return tensor[tuple(indices[:, site]
                                for site in range(indices.shape[1]))]

        history = HistoryObserver()
        from importlib import import_module
        monkeypatch.setattr(
            import_module('tensorkrowch.decompositions.als.tt'),
            '_resolve_observer', lambda *args: history)
        result = tk.decompositions.TTALS(tk.decompositions.CallableTensorSource(function, in_dim=tensor.shape, dtype=torch.float64, out_shape=()), out_device=None).fit(
                rank=2,
                sampling='uniform',
                n_samples=12,
                sample_reuse_sweeps=2,
                generator=torch.Generator().manual_seed(52),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=3),
                collect_metrics=True,
                verbose=1)

        assert len(evaluations) == 2
        assert [event.sweep for event in history.events
                if event.name == 'sample_refresh'] == [0, 2]
        assert [record.sample_generation
                for record in result.metrics.sweeps] == [0, 0, 1]
        assert all(record.abs_error is None
                   for record in result.metrics.sweeps)
        assert result.metadata['exact_configurations'] is None

    def test_uniform_sampling_is_deterministic_with_generator(self):
        tensor = torch.randn(2, 3, 2, dtype=torch.float64)

        def fit(seed):
            return tk.decompositions.TTALS(
                tensor, out_device=None).fit(
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
            out_device=None,
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
            tensor, out_device=None).fit(
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
        assert all(record.abs_error is None
                   for record in result.metrics.sweeps)

    def test_frozen_mode_preserves_original_probabilities_and_marks_staleness(
            self):
        tensor = torch.randn(2, 3, 2, dtype=torch.float64)
        result = tk.decompositions.TTALS(
            tensor, out_device=None).fit(
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

        tk.decompositions.TTALS(tk.decompositions.CallableTensorSource(function, in_dim=tensor.shape, dtype=torch.float64, out_shape=())).fit(
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
                tensor, out_device=None).fit(
                    initial_cores=initial,
                    fixed_cores=[None, initial[1], None],
                    sampling='leverage',
                    n_samples=8,
                    convergence=tk.decompositions.ConvergencePolicy(
                        max_sweeps=1))








@pytest.mark.parametrize('gauge', ['none', 'qr', 'svd'])
@pytest.mark.parametrize('quantized', [False, True])
def test_als_formats_devices_and_quantization(gauge,
                                              quantized,
                                              device_dtype,
                                              assert_close):
    engine = tk.decompositions.TTALS
    device, dtype = device_dtype
    data = torch.tensor([[[1., 2.], [2., 4.]], [[3., 6.], [6., 12.]]],
                        dtype=dtype, device=device)
    if dtype.is_complex:
        data = data * (1 + 1j)
    if quantized:
        real_dtype = data.real.dtype
        source = tk.decompositions.CallableQuanticsSource(
            lambda coordinates: torch.exp(coordinates[:, 0]).to(dtype) *
            ((1 + 1j) if dtype.is_complex else 1),
            1, base=2, level=3,
            domain=torch.tensor([0., 1.], dtype=real_dtype, device=device),
            dtype=dtype, device=device)
        problem = engine(source, out_device=None)
        data = source.contract_dense_grid_axes()
    else:
        problem = engine(data, out_device=None)
    options = dict(rank=1, init='svd', gauge=gauge,
                   convergence=tk.decompositions.ConvergencePolicy(max_sweeps=2),
                   collect_metrics=True)
    if device == 'mps' and dtype.is_complex and gauge == 'qr':
        with pytest.raises(RuntimeError, match='geqrf.*float32'):
            problem.fit(**options)
        return
    result = problem.fit(**options)
    dense = result.contract_dense_grid_axes() if quantized else result.contract_dense()
    assert_close(dense, data.reshape(8) if quantized else data)
    assert result.dtype == dtype and result.device.type == device
    assert result.metrics.sweeps
    assert result.metrics.sweeps[-1].abs_error.device.type == 'cpu'
    assert result.metrics.sweeps[-1].abs_error < 5e-5 * data.norm()


@pytest.mark.parametrize('ordering', ['interleaved', 'grouped'])
def test_quantized_als_callable_and_repeated_fits(ordering):
    layout = tk.formats.QuantizedLayout(2, 2, 2, ordering=ordering)
    coordinate_map = tk.formats.AffineCoordinateMap(
        torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64), layout.grid_size)
    source = tk.decompositions.CallableQuanticsSource(
        lambda coordinates: torch.exp(coordinates[:, 0] + 2 * coordinates[:, 1]),
        2, layout=layout, coordinate_map=coordinate_map)
    engine = tk.decompositions.TTALS(source, out_device=None)
    for _ in range(2):
        result = engine.fit(rank=2, init='svd',
                            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
        torch.testing.assert_close(result.contract_dense_grid_axes(), source.contract_dense_grid_axes())
        coordinates = torch.tensor([[0., 0.], [0.75, 0.75]], dtype=torch.float64)
        torch.testing.assert_close(result.evaluate_coordinates(coordinates),
                                   source.function(coordinates))
        assert not result.metrics.sweeps
    other = tk.decompositions.CallableQuanticsSource(
        source.function, 2, layout=tk.formats.QuantizedLayout(
            2, 2, 2, ordering='grouped' if ordering == 'interleaved' else 'interleaved'),
        coordinate_map=coordinate_map)
    with pytest.raises(ValueError, match='fixed digit layout'):
        tk.decompositions.TTALS(other).fit(initial_cores=result)
    assert tk.decompositions.TTALS(torch.ones(4, 4)).source.is_discrete_tensor


def test_tensor_and_completion_axes_are_not_quantized():
    data = torch.arange(16., dtype=torch.float64).reshape(4, 4)
    engine = tk.decompositions.TTALS(data)
    assert engine.in_dim == (4, 4)
    indices = torch.tensor([[0, 0], [1, 3]])
    values = torch.tensor([1., 2.])
    weights = torch.tensor([2., 3.])
    completion = tk.decompositions.TTALS(tk.decompositions.SparseTensorSource(indices, values, (4, 4)), mask=tk.decompositions.SparseTensorSource(indices, torch.ones(indices.shape[0], dtype=torch.bool, device=indices.device), (4, 4)), weights=tk.decompositions.SparseTensorSource(indices, weights, (4, 4)).contract_dense(), out_device='cpu')
    observed = completion.problem.observations
    assert completion.in_dim == (4, 4)
    torch.testing.assert_close(observed.indices, indices)
    torch.testing.assert_close(observed.values, values)
    torch.testing.assert_close(observed.weights, weights)


@pytest.mark.parametrize('mask_kind', ['dense', 'sparse', 'callable'])
@pytest.mark.parametrize('sampling', ['exact', 'uniform', 'leverage'])
def test_boolean_masks_include_observed_zeros(mask_kind, sampling, device_dtype, assert_close):
    device, dtype = device_dtype
    tensor = torch.tensor([[0., 2.], [3., 9.]], dtype=dtype, device=device)
    selected = torch.tensor([[True, True], [True, False]], device=device)
    if mask_kind == 'dense':
        mask = tk.decompositions.DenseTensorSource(selected)
    elif mask_kind == 'sparse':
        indices = torch.tensor([[0, 0], [0, 1], [1, 0], [0, 0]], device=device)
        mask = tk.decompositions.SparseTensorSource(
            indices, torch.ones(4, device=device, dtype=torch.bool), (2, 2))
    else:
        mask = tk.decompositions.CallableTensorSource(
            lambda indices: selected[tuple(indices.T)], (2, 2),
            dtype=torch.bool, device=device, out_shape=())
    engine = tk.decompositions.TTALS(tensor, mask=mask, out_device=None)
    options = {'rank': 2, 'generator': torch.Generator(device=device).manual_seed(120),
               'collect_metrics': True,
               'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=2)}
    if mask_kind == 'sparse':
        if sampling != 'exact':
            with pytest.raises(ValueError, match='permanently observed'):
                engine.fit(sampling=sampling, n_samples=8, **options)
            return
    elif sampling != 'exact':
        options.update(sampling=sampling, n_samples=128)
    result = engine.fit(**options)
    approximation = result.contract_dense()
    assert_close(approximation[selected], tensor[selected])
    error = torch.linalg.vector_norm((approximation - tensor)[selected])
    if sampling == 'exact' or mask_kind == 'sparse':
        assert result.metrics.sweeps[-1].abs_error == pytest.approx(error.item(), abs=5e-5)
    assert engine.source.tensor is tensor


@pytest.mark.parametrize('source_kind', ['dense', 'callable', 'format'])
def test_matrix_sources_use_shared_als(source_kind, device_dtype, assert_close):
    device, dtype = device_dtype
    tensor = torch.arange(16., device=device).to(dtype).reshape(2, 2, 2, 2)
    if dtype.is_complex:
        tensor = tensor * (1 + 1j)
    if source_kind == 'dense':
        source = tk.decompositions.DenseTensorSource.matrix(tensor, ordering='interleaved')
    elif source_kind == 'callable':
        source = tk.decompositions.CallableTensorSource.matrix(
            lambda indices: tensor.permute(0, 2, 1, 3)[tuple(indices.T)],
            (2, 2), (2, 2), dtype=dtype, device=device)
    else:
        source = tk.decompositions.FormatSource(
            tk.decompositions.TTMSVD(
                tk.decompositions.DenseTensorSource.matrix(tensor, ordering='interleaved'),
                out_device=None).fit().to_format())
    result = tk.decompositions.TTALS(source, out_device=None).fit(
        rank=4, init='svd',
        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
    assert isinstance(result, tk.decompositions.TTMDecomposition)
    assert_close(result.contract_dense(), tensor)
    assert result.to_format().out_dim == (2, 2)


def test_masks_reject_wrong_dimensions_and_numeric_values():
    target = torch.ones(2, 2)
    for mask in (torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 2)):
        with pytest.raises((TypeError, ValueError)):
            tk.decompositions.TTALS(target, mask=mask)
    source = tk.decompositions.CallableSource(lambda x: x.sum(-1), 2)
    with pytest.raises(ValueError, match='discrete tensor'):
        tk.decompositions.TTALS(source)


def test_quantics_matrix_source_retains_both_layouts():
    source = tk.decompositions.CallableQuanticsSource.matrix(
        lambda inputs, outputs: torch.exp(inputs.sum(-1) + outputs.sum(-1)),
        1, 1, in_base=2, out_base=2, in_level=2, out_level=2,
        in_domain=(0., 1.), out_domain=(0., 1.))
    engine = tk.decompositions.TTALS(source)
    result = engine.fit(rank=1, init='svd',
                        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
    assert isinstance(result, tk.decompositions.QTTMDecomposition)
    assert result.in_layout is source.in_layout and result.out_layout is source.out_layout
    torch.testing.assert_close(result.contract_dense_grid_axes(), source.contract_dense_grid_axes())


def test_matrix_mask_and_initial_cores_share_source_grouping():
    tensor = torch.arange(16., dtype=torch.float64).reshape(2, 2, 2, 2)
    source = tk.decompositions.DenseTensorSource.matrix(tensor, ordering='interleaved')
    initial = tk.decompositions.TTSVD(source).fit()
    engine = tk.decompositions.TTALS(source, mask=torch.ones_like(tensor, dtype=torch.bool))
    result = engine.fit(initial_cores=initial,
                        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
    torch.testing.assert_close(result.contract_dense(), tensor)
