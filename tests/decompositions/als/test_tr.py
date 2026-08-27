"""Tests for tensor-ring alternating least squares."""

import pytest

import torch
import tensorkrowch as tk

from tests.decompositions.als._oracles import (contract_tr_dense,
                                               make_tr_cores,
                                               reference_tr_sweep)


def _exact_tr(dtype=torch.float64):
    """Returns a heterogeneous TR and its dense contraction."""
    cores = make_tr_cores(
        input_dim=(2, 3, 2),
        rank=(2, 3, 2),
        dtype=dtype,
        generator=torch.Generator().manual_seed(100))
    return cores, contract_tr_dense(cores)


class TestTRALSExact:  # MARK: TestTRALSExact

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_one_sweep_matches_direct_dense_oracle(self, dtype):
        target_cores, tensor = _exact_tr(dtype)
        initial = make_tr_cores(
            input_dim=tensor.shape,
            rank=(2, 3, 2),
            dtype=dtype,
            generator=torch.Generator().manual_seed(101))
        expected = reference_tr_sweep(initial, tensor, qr=False)
        result = tk.decompositions.TRALS(
            tensor, output_device=None).fit(
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
            tensor, output_device=None).fit(
                rank=(2, 2, 2),
                gauge=gauge,
                generator=torch.Generator().manual_seed(102),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=4),
                collect_metrics=True)

        errors = [record.absolute_error for record in result.metrics.sweeps]
        for previous, current in zip(errors, errors[1:]):
            assert current <= previous + 2e-10 * max(1., previous)
        assert errors[-1] <= errors[0] + 2e-10

    def test_svd_initialization_uses_tr_svd_and_shared_rank(self):
        _, tensor = _exact_tr()
        result = tk.decompositions.TRALS(
            tensor, output_device=None).fit(
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
            tensor, output_device=None).fit(
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
            tensor, output_device=None).fit(
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
            tensor, output_device=None).fit(
                rank=2,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
                collect_metrics=False)

        assert result.metrics.local_solves == []
        assert result.metrics.sweeps == []


class TestTRALSSamplingAndCompletion:  # MARK: TestTRALSSamplingAndCompletion

    def test_uniform_samples_and_values_are_reused_by_generation(self):
        tensor = torch.arange(16., dtype=torch.float64).reshape(2, 2, 2, 2)
        evaluations = []

        def function(indices):
            evaluations.append(indices.clone())
            return tensor[tuple(indices[:, site]
                                for site in range(indices.shape[1]))]

        history = tk.decompositions.HistoryObserver()
        result = tk.decompositions.TRALS(
            function,
            input_dim=tensor.shape,
            dtype=torch.float64,
            output_device=None).fit(
                rank=2,
                sampling='uniform',
                n_samples=12,
                sample_reuse_sweeps=2,
                generator=torch.Generator().manual_seed(103),
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
            input_dim=tensor.shape,
            output_device=None)
        result = decomposition.fit(
            rank=2,
            generator=torch.Generator().manual_seed(104),
            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=3),
            collect_metrics=True)

        approximation = result.evaluate(indices)
        absolute = torch.linalg.vector_norm(approximation - values)
        relative = absolute / torch.linalg.vector_norm(values)
        assert result.metrics.sweeps[-1].absolute_error == pytest.approx(
            absolute.item())
        assert result.metrics.sweeps[-1].relative_error == pytest.approx(
            relative.item())
        assert result.metadata['sampling'] == 'observed'


class TestTRALSValidationAndWrapper:  # MARK: TestTRALSValidationAndWrapper

    def test_rank_sequence_uses_right_link_semantics(self):
        tensor = torch.randn(2, 3, 4, dtype=torch.float64)
        result = tk.decompositions.TRALS(
            tensor, output_device=None).fit(
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
            input_dim=tensor.shape,
            rank=(2, 2, 2),
            generator=torch.Generator().manual_seed(105))

        with pytest.raises(ValueError, match='one right-link rank'):
            tk.decompositions.TRALS(tensor).fit(rank=(2, 2))
        with pytest.raises(ValueError, match='rank.*caps'):
            tk.decompositions.TRALS(tensor).fit(
                rank=1, initial_cores=initial)

    def test_wrapper_returns_cores_and_optional_info(self):
        _, tensor = _exact_tr()
        cores = tk.decompositions.tr_als(
            tensor,
            rank=2,
            max_sweeps=1,
            output_device=None)
        cores_info, info = tk.decompositions.tr_als(
            tensor,
            rank=2,
            max_sweeps=1,
            output_device=None,
            return_info=True)

        assert len(cores) == len(cores_info) == tensor.ndim
        assert info['metadata']['algorithm'] == 'tr_als'
        assert len(info['metrics']['sweeps']) == 1
        assert tk.decompositions.TRDecomposition(cores).topology == 'tr'
