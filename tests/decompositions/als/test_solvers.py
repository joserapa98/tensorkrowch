"""Tests for stable local least-squares solving."""

import pytest

import torch
import tensorkrowch as tk

import tensorkrowch.decompositions.als.solvers as solver_module


class TestLeastSquaresSolver:  # MARK: TestLeastSquaresSolver

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('n_targets', [1, 3])
    def test_matches_dense_full_rank_solution(self, dtype, n_targets):
        generator = torch.Generator().manual_seed(20)
        environment = torch.randn(
            12, 4, dtype=dtype, generator=generator)
        target = torch.randn(
            12, n_targets, dtype=dtype, generator=generator)
        if n_targets == 1:
            target = target.squeeze(1)
        solver = tk.decompositions.LeastSquaresSolver(
            column_scaling=False,
            system_scaling=False)

        solution, record = solver.solve(environment, target, site=2, sweep=3)
        expected = torch.linalg.lstsq(environment, target).solution

        assert torch.allclose(solution, expected, rtol=1e-11, atol=1e-11)
        assert record.environment_shape == (12, 4)
        assert record.target_shape == tuple(target.shape)
        assert record.site == 2
        assert record.sweep == 3
        assert record.driver == 'default'

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_absolute_tikhonov_matches_closed_form(self, dtype):
        generator = torch.Generator().manual_seed(21)
        environment = torch.randn(
            9, 4, dtype=dtype, generator=generator)
        target = torch.randn(9, 2, dtype=dtype, generator=generator)
        regularization = 0.3
        solver = tk.decompositions.LeastSquaresSolver(
            l2_reg=regularization,
            column_scaling=True,
            system_scaling=True)

        solution, record = solver.solve(environment, target)
        gram = environment.mH @ environment
        expected = torch.linalg.solve(
            gram + regularization * torch.eye(
                gram.shape[0], dtype=dtype),
            environment.mH @ target)

        assert torch.allclose(solution, expected, rtol=1e-10, atol=1e-10)
        assert record.effective_l2_reg == pytest.approx(regularization)
        assert record.column_scaling
        assert record.system_scaling

    def test_relative_regularization_uses_original_rms_column_scale(self):
        environment = torch.tensor([
            [3., 0.],
            [0., 4.],
            [0., 0.],
        ], dtype=torch.float64)
        target = torch.tensor([1., 2., 3.], dtype=torch.float64)
        solver = tk.decompositions.LeastSquaresSolver(
            l2_reg=0.2,
            l2_reg_mode='relative',
            column_scaling=True,
            system_scaling=True)

        solution, record = solver.solve(environment, target)
        effective = 0.2 * environment.square().sum().item() / 2
        expected = torch.linalg.solve(
            environment.mT @ environment + effective * torch.eye(2),
            environment.mT @ target)

        assert record.effective_l2_reg == pytest.approx(effective)
        assert torch.allclose(solution, expected, rtol=1e-12, atol=1e-12)

    def test_scalings_preserve_an_imbalanced_regularized_solution(self):
        generator = torch.Generator().manual_seed(22)
        q, _ = torch.linalg.qr(torch.randn(
            10, 3, dtype=torch.float64, generator=generator))
        environment = q * torch.tensor([1e-6, 1., 1e6])
        target = torch.randn(
            10, 2, dtype=torch.float64, generator=generator)
        baseline = tk.decompositions.LeastSquaresSolver(
            l2_reg=0.5,
            column_scaling=False,
            system_scaling=False)
        stabilized = tk.decompositions.LeastSquaresSolver(
            l2_reg=0.5,
            column_scaling=True,
            system_scaling=True)

        baseline_solution, _ = baseline.solve(environment, target)
        stabilized_solution, _ = stabilized.solve(environment, target)

        assert torch.allclose(
            stabilized_solution,
            baseline_solution,
            rtol=1e-9,
            atol=1e-10)

    def test_auto_column_scaling_records_effective_choice(self):
        environment = torch.tensor([
            [1e-12, 1.],
            [2e-12, -1.],
            [3e-12, 2.],
        ], dtype=torch.float64)
        target = torch.tensor([1., 2., 3.], dtype=torch.float64)

        _, imbalanced = tk.decompositions.LeastSquaresSolver(
            column_scaling='auto').solve(environment, target)
        _, balanced = tk.decompositions.LeastSquaresSolver(
            column_scaling='auto').solve(
                torch.tensor(
                    [[1., 0.], [0., 1.], [1., 1.]],
                    dtype=torch.float64),
                target)

        assert imbalanced.column_scaling
        assert not balanced.column_scaling

    def test_global_system_scaling_handles_large_common_magnitude(self):
        environment = 1e150 * torch.tensor([
            [1., 0.],
            [0., 2.],
            [1., 1.],
        ], dtype=torch.float64)
        target = environment @ torch.tensor([2., -3.], dtype=torch.float64)
        solver = tk.decompositions.LeastSquaresSolver(
            column_scaling=False,
            system_scaling=True)

        solution, record = solver.solve(environment, target)

        assert torch.allclose(
            solution, torch.tensor([2., -3.], dtype=torch.float64),
            rtol=1e-12, atol=1e-12)
        assert record.system_scale == pytest.approx(6e150)

    def test_rank_deficient_regularized_system_is_finite(self):
        environment = torch.tensor([
            [1., 1., 0.],
            [2., 2., 0.],
            [3., 3., 0.],
        ], dtype=torch.float64)
        target = torch.tensor([1., 2., 4.], dtype=torch.float64)
        solver = tk.decompositions.LeastSquaresSolver(
            l2_reg=1e-4,
            column_scaling=True)

        solution, record = solver.solve(environment, target)

        assert torch.isfinite(solution).all()
        assert record.residual_absolute >= 0
        assert record.residual_relative >= 0

    def test_cpu_fallback_driver_is_reported(self, monkeypatch):
        environment = torch.randn(8, 3, dtype=torch.float64)
        target = torch.randn(8, dtype=torch.float64)
        original_lstsq = torch.linalg.lstsq
        calls = []

        def fail_default(*args, **kwargs):
            calls.append(kwargs.get('driver'))
            if kwargs.get('driver') is None:
                raise RuntimeError('forced default failure')
            return original_lstsq(*args, **kwargs)

        monkeypatch.setattr(solver_module.torch.linalg, 'lstsq', fail_default)
        solver = tk.decompositions.LeastSquaresSolver(
            column_scaling=False,
            system_scaling=False)

        _, record = solver.solve(environment, target)

        assert calls[:2] == [None, 'gelsy']
        assert record.driver == 'gelsy'

    def test_fast_path_skips_residual_diagnostics(self, monkeypatch):
        environment = torch.randn(8, 3, dtype=torch.float64)
        target = torch.randn(8, dtype=torch.float64)
        solver = tk.decompositions.LeastSquaresSolver(
            column_scaling=False,
            system_scaling=False)

        def unexpected_norm(*args, **kwargs):
            pytest.fail('Fast local solves should not compute residual norms')

        monkeypatch.setattr(
            solver_module.torch.linalg, 'vector_norm', unexpected_norm)
        solution, record = solver.solve(
            environment, target, return_record=False)

        assert torch.isfinite(solution).all()
        assert record is None

    @pytest.mark.parametrize('name', ['environment', 'target'])
    @pytest.mark.parametrize('value', [torch.nan, torch.inf])
    def test_nonfinite_inputs_fail_before_solve(self, name, value):
        environment = torch.eye(3)
        target = torch.ones(3)
        if name == 'environment':
            environment[0, 0] = value
        else:
            target[0] = value
        solver = tk.decompositions.LeastSquaresSolver()

        with pytest.raises(ValueError, match='finite'):
            solver.solve(environment, target)

    def test_invalid_configuration(self):
        with pytest.raises(ValueError, match='l2_reg'):
            tk.decompositions.LeastSquaresSolver(l2_reg=-1)
        with pytest.raises(ValueError, match='l2_reg'):
            tk.decompositions.LeastSquaresSolver(l2_reg=torch.inf)
        with pytest.raises(ValueError, match='l2_reg_mode'):
            tk.decompositions.LeastSquaresSolver(l2_reg_mode='scaled')
        with pytest.raises(ValueError, match='column_scaling'):
            tk.decompositions.LeastSquaresSolver(column_scaling='always')
        with pytest.raises(ValueError, match='driver'):
            tk.decompositions.LeastSquaresSolver(driver='unknown')
