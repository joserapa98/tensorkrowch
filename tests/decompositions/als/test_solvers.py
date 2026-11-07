"""Tests for stable local least-squares solving."""

import pytest
import torch

import tensorkrowch as tk

import tensorkrowch.decompositions.als.solvers as solver_module


class TestLeastSquaresSolver:  # MARK: TestLeastSquaresSolver

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_records_detach_without_detaching_solution(self, dtype):
        generator = torch.Generator().manual_seed(23)
        environment = torch.randn(
            8, 3, dtype=dtype, generator=generator, requires_grad=True)
        target = torch.randn(8, dtype=dtype, generator=generator)
        solution, record = tk.decompositions.LeastSquaresSolver(
            l2_reg=0.1).solve(environment, target)

        for value in (record.abs_residual, record.rel_residual,
                      record.effective_l2_reg):
            assert value.shape == ()
            assert value.dtype == torch.float64
            assert value.device.type == 'cpu'
            assert not value.requires_grad
        solution.abs().square().sum().backward()
        assert torch.isfinite(environment.grad).all()

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

    def test_explicit_regularization_scale_preserves_normalized_system(self):
        generator = torch.Generator().manual_seed(25)
        environment = torch.randn(
            9, 4, dtype=torch.float64, generator=generator)
        target = torch.randn(9, dtype=torch.float64, generator=generator)
        scale = environment.new_tensor(7.)
        solver = tk.decompositions.LeastSquaresSolver(
            l2_reg=0.3,
            column_scaling=True,
            system_scaling=True)

        direct, _ = solver.solve(environment, target)
        normalized, record = solver.solve(
            environment / scale,
            target / scale,
            regularization_scale=scale.square().reciprocal())

        assert record.effective_l2_reg == pytest.approx(0.3 / 49)
        assert torch.allclose(normalized, direct, rtol=1e-11, atol=1e-11)

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
        assert torch.isfinite(record.abs_residual)

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
        assert record.abs_residual >= 0
        assert record.rel_residual >= 0

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


@pytest.mark.parametrize('column_scaling', [False, True, 'auto'])
def test_solver_devices_complex_values_and_residuals(
    column_scaling, device_dtype, assert_close
):
    device, dtype = device_dtype
    matrix = torch.tensor([[1., 0.], [0., 2.], [1., 1.]],
                          dtype=dtype, device=device)
    solution = torch.tensor([2., -1.], dtype=dtype, device=device)
    if dtype.is_complex:
        solution = solution + 1j * solution.flip(0)
    target = matrix @ solution
    solver = tk.decompositions.LeastSquaresSolver(
        column_scaling=column_scaling)
    actual, info = solver.solve(matrix, target, return_record=True)
    assert_close(actual, solution)
    assert info.abs_residual.device.type == 'cpu'
    assert info.abs_residual < 5e-5 * target.norm()
    assert torch.isfinite(info.abs_residual)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('shape', [(1, 1), (1, 3), (6, 1), (3, 5)])
def test_minimal_and_underdetermined_systems(dtype, shape):
    generator = torch.Generator().manual_seed(71)
    matrix = torch.randn(*shape, dtype=dtype, generator=generator)
    target = torch.randn(shape[0], dtype=dtype, generator=generator)
    solver = tk.decompositions.LeastSquaresSolver(column_scaling=False)
    actual, record = solver.solve(matrix, target)
    expected = torch.linalg.lstsq(matrix, target, driver='gelsd').solution
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(
        record.abs_residual, (matrix @ actual - target).norm())


@pytest.mark.parametrize('kwargs, error', [
    ({'l2_reg': True}, TypeError), ({'l2_reg': '0'}, TypeError),
    ({'l2_reg': float('nan')}, ValueError),
    ({'rcond': True}, TypeError), ({'rcond': -1.}, ValueError),
    ({'rcond': float('inf')}, ValueError),
    ({'system_scaling': 1}, TypeError),
    ({'column_scaling': 1}, ValueError),
])
def test_solver_constructor_rejects_invalid_types_and_values(kwargs, error):
    with pytest.raises(error):
        tk.decompositions.LeastSquaresSolver(**kwargs)


@pytest.mark.parametrize(
    'matrix, target, error',
    [
        ([1.0], torch.ones(1), TypeError),
        (torch.ones(1, 1), [1.0], TypeError),
        (torch.ones(2), torch.ones(2), ValueError),
        (torch.ones(1, 1), torch.tensor(1.0), ValueError),
        (torch.ones(0, 2), torch.ones(0), ValueError),
        (torch.ones(2, 0), torch.ones(2), ValueError),
        (torch.ones(2, 2), torch.ones(3), ValueError),
        (torch.ones(2, 2), torch.ones(2, 0), ValueError),
        (torch.ones(2, 2), torch.ones(2, dtype=torch.float64), ValueError),
        (torch.ones(2, 2), torch.ones(2, device='meta'), ValueError),
        (
            torch.ones(2, 2, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
            TypeError,
        ),
    ],
)
def test_solver_rejects_invalid_local_systems(matrix, target, error):
    with pytest.raises(error):
        tk.decompositions.LeastSquaresSolver().solve(matrix, target)


@pytest.mark.parametrize('scale, error', [
    (1., TypeError), (torch.ones(2), ValueError),
    (torch.tensor(-1.), ValueError), (torch.tensor(float('inf')), ValueError),
    (torch.ones((), device='meta'), ValueError),
    (torch.tensor(1 + 2j), TypeError),
])
def test_regularization_scale_validation(scale, error):
    with pytest.raises(error):
        tk.decompositions.LeastSquaresSolver(l2_reg=0.1).solve(
            torch.eye(2), torch.ones(2), regularization_scale=scale)


def test_regularization_coefficient_overflow_is_rejected():
    with pytest.raises(ValueError, match='l2_reg'):
        tk.decompositions.LeastSquaresSolver(l2_reg=1e100).solve(
            torch.eye(2), torch.ones(2))


def test_return_record_requires_boolean():
    with pytest.raises(TypeError, match='return_record'):
        tk.decompositions.LeastSquaresSolver().solve(
            torch.eye(2), torch.ones(2), return_record=1)


def test_all_cpu_drivers_failing_propagates(monkeypatch):
    attempted = []

    def fail(*args, **kwargs):
        attempted.append(kwargs['driver'])
        raise RuntimeError('forced failure')

    monkeypatch.setattr(torch.linalg, 'lstsq', fail)
    with pytest.raises(RuntimeError, match='All CPU'):
        tk.decompositions.LeastSquaresSolver(driver='gelsd').solve(
            torch.eye(2), torch.ones(2))
    assert attempted == ['gelsd', 'gelsy', 'gelss', 'gels']


def test_nonfinite_solution_has_specific_exception(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        torch.linalg,
        'lstsq',
        lambda *args, **kwargs: SimpleNamespace(
            solution=torch.full((2, 1), torch.inf)
        ),
    )
    with pytest.raises(solver_module.NonFiniteSolutionError):
        tk.decompositions.LeastSquaresSolver().solve(
            torch.eye(2), torch.ones(2)
        )


@pytest.mark.parametrize('zero_matrix', [False, True])
def test_zero_target_and_zero_system(zero_matrix):
    matrix = torch.zeros(3, 2) if zero_matrix else torch.tensor(
        [[1., 0.], [0., 1.], [1., 1.]])
    solution, record = tk.decompositions.LeastSquaresSolver().solve(
        matrix, torch.zeros(3))
    torch.testing.assert_close(solution, torch.zeros(2))
    assert record.abs_residual == 0
    assert record.rel_residual == 0


def test_local_proposal_restores_core_shape_and_sampling_metadata():
    core = torch.zeros(1, 2, 1, dtype=torch.complex128)
    target = torch.tensor([1 + 2j, 3 - 1j], dtype=core.dtype)
    problem = tk.decompositions.ALSProblem(
        tk.decompositions.DenseTensorSource(target))
    samples = tk.decompositions.SampleBatch(
        ids=torch.arange(2), probabilities=torch.full((2,), .5),
        weights=torch.ones(2), generation=3, proposal_exact=False)
    actual, record = solver_module._solve_local_proposal(
        tk.decompositions.LeastSquaresSolver(),
        torch.eye(2, dtype=core.dtype), target, core,
        site=0, sweep=2, return_record=True,
        problem=problem, log_scale=torch.tensor(0.), samples=samples)
    torch.testing.assert_close(actual, target.reshape(core.shape))
    torch.testing.assert_close(core, torch.zeros_like(core))
    assert record.sampling_exact is False
    assert record.sample_generation == 3


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('route', ['full', 'sampled', 'observed'])
@pytest.mark.parametrize('log_scale', [0., 5.])
@pytest.mark.parametrize('mode', ['absolute', 'relative'])
def test_local_system_matches_original_weighted_regularized_loss(
        dtype, route, log_scale, mode):
    generator = torch.Generator().manual_seed(319)
    matrix = torch.randn(6, 2, dtype=dtype, generator=generator)
    target = torch.randn(2, 3, dtype=dtype, generator=generator)
    weights = torch.linspace(.5, 2., 6, dtype=torch.float64)
    source = tk.decompositions.DenseTensorSource(target)
    samples = None
    indices = None
    if route == 'full':
        mask = torch.tensor([[False, True, True], [True, True, True]])
        problem = tk.decompositions.ALSProblem(
            source, weights=weights.reshape(2, 3),
            mask=tk.decompositions.DenseTensorSource(mask))
        target = target.reshape(-1)
        weights = weights * mask.reshape(-1)
    else:
        ids = torch.tensor([0, 4, 4, 2] if route == 'sampled' else
                           [0, 2, 4, 5])
        indices = torch.stack((ids // 3, ids % 3), dim=1)
        matrix = matrix[ids]
        target = target.reshape(-1)[ids]
        if route == 'observed':
            observations = tk.decompositions.SparseTensorSource(
                indices, target, (2, 3))
            problem = tk.decompositions.ALSProblem(
                observations=observations, weights=weights[ids])
            indices = None
        else:
            problem = tk.decompositions.ALSProblem(
                source, weights=weights.reshape(2, 3), sampled=True)
            probabilities = torch.tensor([.1, .2, .2, .3])
            samples = tk.decompositions.SampleBatch(
                ids=ids, probabilities=probabilities,
                weights=(4 * probabilities).rsqrt(), generation=2)
        weights = weights[ids]
    if samples is not None:
        weights = weights * samples.weights
    solver = tk.decompositions.LeastSquaresSolver(
        l2_reg=.03, l2_reg_mode=mode, column_scaling=False,
        system_scaling=False)
    expected, _ = solver.solve(matrix * weights[:, None], target * weights)
    scale = matrix.real.new_tensor(log_scale)
    actual, _ = solver_module._solve_local_proposal(
        solver, matrix * (-scale).exp(), target,
        torch.zeros(1, 2, 1, dtype=dtype), site=0, sweep=0,
        return_record=False, problem=problem, log_scale=scale,
        indices=indices, samples=samples)
    torch.testing.assert_close(actual.reshape(-1), expected)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('sampled', [False, True])
@pytest.mark.parametrize('mode, penalty', [('absolute', 0.), ('absolute', .04),
                                         ('relative', .04)])
@pytest.mark.parametrize('log_scale', [-6., 0., 6.])
@pytest.mark.parametrize('column_scaling', [False, True, 'auto'])
def test_shared_physical_system_preserves_the_regularized_objective(
        dtype, sampled, mode, penalty, log_scale, column_scaling):
    generator = torch.Generator().manual_seed(322)
    core = torch.zeros(2, 3, 2, dtype=dtype)
    dimensions = (4, 3, 5)
    target = torch.randn(dimensions, dtype=dtype, generator=generator)
    environment = torch.randn(4, 5, 2, 2, dtype=dtype, generator=generator)
    identity = torch.eye(3, dtype=dtype)
    full = torch.einsum(
        'lrab,ij->lirajb', environment, identity).reshape(60, 12)
    indices, batch = None, None
    problem = tk.decompositions.ALSProblem(
        tk.decompositions.DenseTensorSource(target), sampled=sampled)
    if sampled:
        external = torch.tensor([[0, 0], [3, 2], [0, 0], [2, 4], [1, 1]])
        indices = torch.zeros(5, 3, 3, dtype=torch.long)
        indices[:, :, 0] = external[:, :1]
        indices[:, :, 1] = torch.arange(3)
        indices[:, :, 2] = external[:, 1:]
        indices = indices.reshape(-1, 3)
        ids = indices[:, 0] * 15 + indices[:, 1] * 5 + indices[:, 2]
        probabilities = torch.tensor([.1, .15, .1, .05, .2],
                                     dtype=torch.float64).repeat_interleave(3)
        batch = tk.decompositions.SampleBatch(
            ids, probabilities, (15 * probabilities).rsqrt(), generation=2)
        full = full[ids]
        environment = environment[external[:, 0], external[:, 1]]
        target = target[tuple(indices.T)]
    else:
        target = target.flatten()
    matrix = environment.reshape(-1, 4)
    solver = tk.decompositions.LeastSquaresSolver(
        l2_reg=penalty, l2_reg_mode=mode, column_scaling=column_scaling,
        rcond=1e-12, driver='gelsd')
    common = dict(site=1, sweep=0, return_record=True, problem=problem,
                  indices=indices, samples=batch,
                  log_scale=torch.tensor(log_scale, dtype=torch.float64))
    expected, old_record = solver_module._solve_local_proposal(
        solver, full * torch.exp(torch.tensor(-log_scale)), target, core,
        **common)
    actual, record = solver_module._solve_local_proposal(
        solver, matrix * torch.exp(torch.tensor(-log_scale)), target, core,
        compact=True, **common)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(record.abs_residual, old_record.abs_residual)
    torch.testing.assert_close(record.rel_residual, old_record.rel_residual)
    assert record.environment_shape == (matrix.shape[0], 4)
    assert record.target_shape == (matrix.shape[0], 3)
    if batch is not None:
        assert record.sample_generation == 2


@pytest.mark.parametrize('rcond', [None, 1e-18, 1e-12])
@pytest.mark.parametrize('regularized', [False, True])
def test_shared_solver_retains_expanded_default_rank_cutoff(
        rcond, regularized):
    environment = torch.diag(torch.tensor([1., 2e-15], dtype=torch.float64))
    target = torch.ones(2, 8, dtype=torch.float64)
    solver = tk.decompositions.LeastSquaresSolver(
        rcond=rcond, column_scaling=False, system_scaling=False,
        driver='gelsd', l2_reg=1e-32 if regularized else 0.)
    expanded = torch.kron(torch.eye(8, dtype=torch.float64), environment)
    expected, _ = solver.solve(expanded, target.T.reshape(-1))
    actual, record = solver._solve(environment, target, n_copies=8)
    torch.testing.assert_close(actual.T.reshape(-1), expected)
    assert record.environment_shape == (2, 2)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_shared_solver_zero_and_rank_deficient_systems(dtype):
    solver = tk.decompositions.LeastSquaresSolver(driver='gelsd', rcond=1e-12)
    matrices = [torch.zeros(4, 3, dtype=dtype), torch.ones(2, 3, dtype=dtype)]
    for matrix in matrices:
        target = torch.arange(matrix.shape[0] * 5).reshape(-1, 5).to(dtype)
        actual, _ = solver._solve(matrix, target, n_copies=5)
        expected = torch.linalg.pinv(matrix, rtol=1e-12) @ target
        torch.testing.assert_close(actual, expected)



def test_compact_proposal_supports_physical_dependent_sampling_weights():
    target = torch.ones(2, dtype=torch.float64)
    problem = tk.decompositions.ALSProblem(
        tk.decompositions.DenseTensorSource(target), sampled=True)
    probabilities = torch.tensor([1 / 3, 2 / 3], dtype=torch.float64)
    batch = tk.decompositions.SampleBatch(
        torch.arange(2), probabilities, (2 * probabilities).rsqrt(), 0)
    core, _ = solver_module._solve_local_proposal(
        tk.decompositions.LeastSquaresSolver(), torch.ones(1, 1).double(),
        target, torch.zeros(1, 2, 1).double(), site=0, sweep=0,
        return_record=False, problem=problem, samples=batch,
        log_scale=torch.tensor(0.), compact=True)
    torch.testing.assert_close(core.flatten(), target)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_shared_solver_preserves_gradients(dtype):
    generator = torch.Generator().manual_seed(328)
    matrix = torch.randn(5, 3, dtype=dtype, generator=generator,
                         requires_grad=True)
    target = torch.randn(5, 4, dtype=dtype, generator=generator,
                         requires_grad=True)
    solver = tk.decompositions.LeastSquaresSolver(l2_reg=.1, driver='gelsd')
    shared, _ = solver._solve(matrix, target, n_copies=4)
    expanded, _ = solver.solve(torch.kron(torch.eye(4, dtype=dtype), matrix),
                                target.T.reshape(-1))
    expected = torch.autograd.grad(expanded.abs().square().sum(),
                                   (matrix, target), retain_graph=True)
    actual = torch.autograd.grad(shared.abs().square().sum(), (matrix, target))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('mode, penalty', [('absolute', 0.), ('absolute', .03),
                                         ('relative', .03)])
@pytest.mark.parametrize('scaling', [False, True, 'auto'])
@pytest.mark.parametrize('missing', [False, True])
def test_physical_blocks_preserve_the_complete_solver_objective(
        dtype, mode, penalty, scaling, missing):
    generator = torch.Generator().manual_seed(412)
    lengths = [7, 4, 0 if missing else 5, 8]
    matrices = [torch.randn(m, 3, dtype=dtype, generator=generator)
                for m in lengths]
    targets = [torch.randn(m, dtype=dtype, generator=generator)
               for m in lengths]
    matrices[0] *= 1e3
    matrices[1] *= 1e-3
    solver = tk.decompositions.LeastSquaresSolver(
        l2_reg=penalty, l2_reg_mode=mode, column_scaling=scaling,
        driver='gelsd', rcond=1e-12)
    full = torch.block_diag(*matrices)
    expected, previous = solver.solve(full, torch.cat(targets))
    actual, record = solver._solve_blocks(
        matrices, targets, sum(lengths), 1, 2, True)
    torch.testing.assert_close(actual.T.flatten(), expected,
                               atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(record.effective_l2_reg,
                               previous.effective_l2_reg)
    torch.testing.assert_close(record.abs_residual, previous.abs_residual)
    torch.testing.assert_close(record.rel_residual, previous.rel_residual)
    assert record.column_scaling == previous.column_scaling
    assert record.environment_shape == (sum(lengths), 3)
    assert record.site == 1 and record.sweep == 2


@pytest.mark.parametrize('driver', ['gelsy', 'gelsd', 'gelss', 'gels'])
@pytest.mark.parametrize('scaling', [False, True, 'auto'])
def test_block_rank_decisions_use_the_global_singular_value_scale(
        driver, scaling):
    matrices = [torch.diag(torch.tensor([1., .5], dtype=torch.float64)),
                torch.diag(torch.tensor([1e-10, 1e-16], dtype=torch.float64))]
    targets = [torch.ones(2, dtype=torch.float64)] * 2
    solver = tk.decompositions.LeastSquaresSolver(
        driver=driver, rcond=1e-8, column_scaling=scaling)
    expected, _ = solver.solve(torch.block_diag(*matrices), torch.cat(targets))
    actual, _ = solver._solve_blocks(matrices, targets, 4, 0, 0, True)
    torch.testing.assert_close(actual.T.flatten(), expected)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_block_solver_preserves_gradients(dtype):
    generator = torch.Generator().manual_seed(413)
    matrices = [torch.randn(6, 3, dtype=dtype, generator=generator,
                            requires_grad=True) for _ in range(3)]
    targets = [torch.randn(6, dtype=dtype, generator=generator,
                           requires_grad=True) for _ in matrices]
    solver = tk.decompositions.LeastSquaresSolver(l2_reg=.05, driver='gelsd')
    expected, _ = solver.solve(torch.block_diag(*matrices), torch.cat(targets))
    actual, _ = solver._solve_blocks(matrices, targets, 18, 0, 0, False)
    reference = torch.autograd.grad(expected.abs().square().sum(),
                                    [*matrices, *targets], retain_graph=True)
    gradients = torch.autograd.grad(actual.abs().square().sum(),
                                    [*matrices, *targets])
    for a, b in zip(gradients, reference):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
