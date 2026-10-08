"""Tests for tensor ring alternating least squares."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.observers import HistoryObserver

from tests.decompositions.als._oracles import (contract_tr_dense,
                                               make_tr_cores,
                                               reference_tr_sweep)


def _exact_tr(dtype=torch.float64):
    """Returns a heterogeneous TR and its dense contraction."""
    cores = make_tr_cores(
        in_dim=(2, 3, 2),
        rank=(2, 3, 2),
        dtype=dtype,
        generator=torch.Generator().manual_seed(100))
    return cores, contract_tr_dense(cores)


class TestTRALSExact:  # MARK: TestTRALSExact

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_one_sweep_matches_direct_dense_oracle(self, dtype):
        target_cores, tensor = _exact_tr(dtype)
        initial = make_tr_cores(
            in_dim=tensor.shape,
            rank=(2, 3, 2),
            dtype=dtype,
            generator=torch.Generator().manual_seed(101))
        expected = reference_tr_sweep(initial, tensor, qr=False)
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                initial_cores=initial,
                gauge='none',
                solver=tk.decompositions.LeastSquaresSolver(
                    column_scaling=False,
                    system_scaling=False),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
                normalize=False,
                renormalize=False)

        assert isinstance(result, tk.decompositions.TRDecomposition)
        for actual, direct in zip(result.cores, expected):
            assert torch.allclose(
                actual, direct, rtol=2e-10, atol=2e-10)
        assert result.rank == [2, 3, 2]
        assert target_cores[0].dtype == result.dtype

    @pytest.mark.parametrize('gauge', ['none', 'qr', 'svd'])
    def test_exact_objective_is_monotone(self, gauge):
        _, tensor = _exact_tr()
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                rank=(2, 2, 2),
                gauge=gauge,
                generator=torch.Generator().manual_seed(102),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=4),
                collect_metrics=True)

        errors = [record.abs_error for record in result.metrics.sweeps]
        for previous, current in zip(errors, errors[1:]):
            assert current <= previous + 2e-10 * max(1., previous)
        assert errors[-1] <= errors[0] + 2e-10

    def test_svd_initialization_uses_tr_svd_and_shared_rank(self):
        _, tensor = _exact_tr()
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                rank=3,
                init='svd',
                gauge='none',
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1))

        assert len(result.rank) == tensor.ndim
        assert all(value <= 3 for value in result.rank)
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)

    def test_fixed_core_is_unchanged_and_blocks_cyclic_absorption(self):
        initial, tensor = _exact_tr()
        fixed = initial[0].clone()
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                initial_cores=initial,
                fixed_cores=(fixed, None, None),
                gauge='qr',
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=2))

        assert torch.equal(result.cores[0], fixed)
        assert result.metadata['fixed_sites'] == [0]
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)

    def test_all_fixed_cores_stop_without_updates(self):
        initial, tensor = _exact_tr()
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                initial_cores=initial,
                fixed_cores=initial)

        assert result.metadata['converged']
        assert result.metadata['stop_reason'] == 'all_cores_fixed'
        assert result.metadata['n_sweeps'] == 0
        assert all(torch.equal(actual, expected)
                   for actual, expected in zip(result.cores, initial))

    def test_fast_path_omits_records_and_objective_reductions(self):
        _, tensor = _exact_tr()
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                rank=2,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
                collect_metrics=False)

        assert result.metrics.local_solves == []
        assert result.metrics.sweeps == []


class TestTRALSSamplingAndCompletion:  # MARK: TestTRALSSamplingAndCompletion

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
            import_module('tensorkrowch.decompositions.als.tr'),
            '_resolve_observer', lambda *args: history)
        result = tk.decompositions.TRALS(
            function,
            in_dim=tensor.shape,
            dtype=torch.float64,
            out_device=None).fit(
                rank=2,
                sampling='uniform',
                n_samples=12,
                sample_reuse_sweeps=2,
                generator=torch.Generator().manual_seed(103),
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

    def test_completion_measures_only_permanent_observations(self):
        tensor = torch.arange(8., dtype=torch.float64).reshape(2, 2, 2)
        indices = torch.tensor([
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
        values = tensor[tuple(indices[:, site]
                              for site in range(indices.shape[1]))]
        decomposition = tk.decompositions.TRALS.completion(
            indices,
            values,
            in_dim=tensor.shape,
            out_device=None)
        result = decomposition.fit(
            rank=2,
            generator=torch.Generator().manual_seed(104),
            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=3),
            collect_metrics=True)

        approximation = result.evaluate(indices)
        absolute = torch.linalg.vector_norm(approximation - values)
        relative = absolute / torch.linalg.vector_norm(values)
        assert result.metrics.sweeps[-1].abs_error == pytest.approx(
            absolute.item())
        assert result.metrics.sweeps[-1].rel_error == pytest.approx(
            relative.item())
        assert result.metadata['sampling'] == 'observed'

    def test_product_leverage_redraws_current_approximation_per_site(self):
        tensor = torch.arange(8., dtype=torch.float64).reshape(2, 2, 2)
        evaluations = []

        def function(indices):
            evaluations.append(indices.clone())
            return tensor[tuple(indices[:, site]
                                for site in range(indices.shape[1]))]

        result = tk.decompositions.TRALS(
            function,
            in_dim=tensor.shape,
            dtype=torch.float64,
            out_device=None).fit(
                rank=2,
                gauge='none',
                sampling='leverage',
                n_samples=10,
                leverage_uniform_mix=0.2,
                generator=torch.Generator().manual_seed(106),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=2),
                collect_metrics=True)

        assert len(evaluations) == 2 * tensor.ndim
        for site, indices in enumerate(evaluations[:tensor.ndim]):
            fibers = indices.reshape(10, tensor.shape[site], tensor.ndim)
            assert torch.equal(
                fibers[:, :, site],
                torch.arange(tensor.shape[site]).expand(10, -1))
        assert len(result.metrics.local_solves) == 2 * tensor.ndim
        assert [record.sample_generation
                for record in result.metrics.local_solves] == [0] * 3 + [1] * 3
        assert all(record.sampling_exact is False
                   for record in result.metrics.local_solves)
        assert result.metadata['sampling_exact'] is False
        assert result.metadata['leverage_uniform_mix'] == pytest.approx(0.2)
        assert all(record.abs_error is None
                   for record in result.metrics.sweeps)

    def test_product_leverage_requires_sitewise_refresh(self):
        decomposition = tk.decompositions.TRALS(torch.ones(2, 2, 2))

        with pytest.raises(ValueError, match='redraws per site'):
            decomposition.fit(
                rank=2,
                sampling='leverage',
                n_samples=5,
                sample_reuse_sweeps=2)

    def test_exact_leverage_uses_fibers_and_current_conditional_design(self):
        tensor = torch.arange(16., dtype=torch.float64).reshape(2, 2, 2, 2)
        evaluations = []

        def function(indices):
            evaluations.append(indices.clone())
            return tensor[tuple(indices[:, site]
                                for site in range(indices.shape[1]))]

        result = tk.decompositions.TRALS(
            function,
            in_dim=tensor.shape,
            dtype=torch.float64,
            out_device=None).fit(
                rank=2,
                gauge='none',
                sampling='leverage',
                leverage_method='exact',
                n_samples=6,
                generator=torch.Generator().manual_seed(108),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
                collect_metrics=True)

        assert len(evaluations) == tensor.ndim
        for site, indices in enumerate(evaluations):
            fibers = indices.reshape(6, tensor.shape[site], tensor.ndim)
            assert torch.equal(
                fibers[:, :, site],
                torch.arange(tensor.shape[site]).expand(6, -1))
        assert all(record.sampling_exact is True
                   for record in result.metrics.local_solves)
        assert result.metadata['sampling_exact'] is True
        assert result.metadata['leverage_method'] == 'exact'
        assert result.metadata['n_samples'] == 6


class TestTRALSValidationAndWrapper:  # MARK: TestTRALSValidationAndWrapper

    def test_rank_sequence_uses_right_link_semantics(self):
        tensor = torch.randn(2, 3, 4, dtype=torch.float64)
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                rank=(2, 3, 4),
                gauge='none',
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1))

        assert result.rank == [2, 3, 4]
        assert result.cores[0].shape == (4, 2, 2)
        assert result.cores[1].shape == (2, 3, 3)
        assert result.cores[2].shape == (3, 4, 4)

    def test_rank_length_and_cap_are_validated(self):
        tensor = torch.randn(2, 2, 2, dtype=torch.float64)
        initial = make_tr_cores(
            in_dim=tensor.shape,
            rank=(2, 2, 2),
            generator=torch.Generator().manual_seed(105))

        with pytest.raises(ValueError, match='one right-link rank'):
            tk.decompositions.TRALS(tensor).fit(rank=(2, 2))
        with pytest.raises(ValueError, match='rank.*caps'):
            tk.decompositions.TRALS(tensor).fit(
                rank=1, initial_cores=initial)

        with pytest.raises(ValueError, match='leverage_method'):
            tk.decompositions.TRALS(tensor).fit(
                rank=2, leverage_method='diagonal')

    def test_wrapper_returns_cores_and_optional_info(self):
        _, tensor = _exact_tr()
        cores = tk.decompositions.tr_als(
            tensor,
            rank=2,
            max_sweeps=1,
            out_device=None)
        cores_info, info = tk.decompositions.tr_als(
            tensor,
            rank=2,
            max_sweeps=1,
            out_device=None,
            return_info=True)

        assert len(cores) == len(cores_info) == tensor.ndim
        assert info['metadata']['algorithm'] == 'tr_als'
        assert len(info['metrics']['sweeps']) == 1
        assert tk.decompositions.TRDecomposition(cores).topology == 'tr'

    def test_wrapper_supports_approximate_leverage_sampling(self):
        tensor = torch.randn(2, 2, 2, dtype=torch.float64)
        cores, info = tk.decompositions.tr_als(
            tensor,
            rank=2,
            sampling='leverage',
            n_samples=8,
            leverage_uniform_mix=0.1,
            max_sweeps=1,
            out_device=None,
            generator=torch.Generator().manual_seed(107),
            return_info=True)

        assert len(cores) == tensor.ndim
        assert info['metadata']['sampling'] == 'leverage'
        assert info['metadata']['sampling_exact'] is False

    def test_wrapper_supports_exact_leverage_sampling(self):
        tensor = torch.randn(2, 2, 2, dtype=torch.float64)
        cores, info = tk.decompositions.tr_als(
            tensor,
            rank=2,
            sampling='leverage',
            leverage_method='exact',
            n_samples=5,
            max_sweeps=1,
            out_device=None,
            generator=torch.Generator().manual_seed(109),
            return_info=True)

        assert len(cores) == tensor.ndim
        assert info['metadata']['leverage_method'] == 'exact'
        assert info['metadata']['sampling_exact'] is True






@pytest.mark.parametrize('gauge', ['none', 'qr', 'svd'])
@pytest.mark.parametrize('quantized', [False, True])
def test_als_formats_devices_and_quantization(gauge,
                                              quantized,
                                              device_dtype,
                                              assert_close):
    engine = tk.decompositions.TRALS
    device, dtype = device_dtype
    data = torch.tensor([[[1., 2.], [2., 4.]], [[3., 6.], [6., 12.]]],
                        dtype=dtype, device=device)
    if dtype.is_complex:
        data = data * (1 + 1j)
    if quantized:
        real_dtype = data.real.dtype
        source = tk.decompositions.QuanticsVectorSource(
            lambda coordinates: torch.exp(coordinates[:, 0]).to(dtype) *
            ((1 + 1j) if dtype.is_complex else 1),
            1, base=2, level=3,
            domain=torch.tensor([0., 1.], dtype=real_dtype, device=device),
            dtype=dtype, device=device)
        problem = engine.quantized(source, out_device=None)
        data = source.to_dense_grid()
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
    dense = result.to_dense_grid() if quantized else result.contract_dense()
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
    source = tk.decompositions.QuanticsVectorSource(
        lambda coordinates: torch.exp(coordinates[:, 0] + 2 * coordinates[:, 1]),
        2, layout=layout, coordinate_map=coordinate_map)
    engine = tk.decompositions.TRALS.quantized(source, out_device=None)
    for _ in range(2):
        result = engine.fit(rank=2, init='svd',
                            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
        torch.testing.assert_close(result.to_dense_grid(), source.to_dense_grid())
        coordinates = torch.tensor([[0., 0.], [0.75, 0.75]], dtype=torch.float64)
        torch.testing.assert_close(result.evaluate_coordinates(coordinates),
                                   source.function(coordinates))
        assert not result.metrics.sweeps
    other = tk.decompositions.QuanticsVectorSource(
        source.function, 2, layout=tk.formats.QuantizedLayout(
            2, 2, 2, ordering='grouped' if ordering == 'interleaved' else 'interleaved'),
        coordinate_map=coordinate_map)
    with pytest.raises(ValueError, match='fixed digit layout'):
        tk.decompositions.TRALS.quantized(other).fit(initial_cores=result)
    with pytest.raises(TypeError, match='QuanticsVectorSource'):
        tk.decompositions.TRALS.quantized(torch.ones(4, 4))


def test_tensor_and_completion_axes_are_not_quantized():
    data = torch.arange(16., dtype=torch.float64).reshape(4, 4)
    engine = tk.decompositions.TRALS(data)
    assert engine.in_dim == (4, 4)
    indices = torch.tensor([[0, 0], [1, 3]])
    values = torch.tensor([1., 2.])
    weights = torch.tensor([2., 3.])
    completion = tk.decompositions.TRALS.completion(
        indices, values, in_dim=(4, 4), weights=weights)
    observed = completion.problem.observations
    assert completion.in_dim == (4, 4)
    torch.testing.assert_close(observed.indices, indices)
    torch.testing.assert_close(observed.values, values)
    torch.testing.assert_close(observed.weights, weights)
