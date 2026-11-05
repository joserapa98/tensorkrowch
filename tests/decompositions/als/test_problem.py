"""Tests for scalar source and stored-support ALS objectives."""

import math

import pytest
import torch

import tensorkrowch as tk


def _observations(dtype=torch.float64, n_sites=2):
    """Returns ordered stored entries, including an explicit zero target."""
    dimensions = (3,) if n_sites == 1 else (2, 3, 2, 3, 2)[:n_sites]
    indices = torch.tensor([
        [0] * n_sites,
        [dim - 1 for dim in dimensions],
    ])
    values = torch.tensor([0., 3.], dtype=dtype)
    if dtype.is_complex:
        values = values + 1j * values
    return tk.decompositions.SparseTensorSource(indices, values, dimensions)


class TestALSProblem:  # MARK: TestALSProblem

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('n_sites', [1, 2, 5])
    def test_source_evaluation_mask_and_weighted_objective(
        self, dtype, n_sites
    ):
        observations = _observations(dtype, n_sites)
        tensor = torch.arange(math.prod(observations.in_dim),
                              dtype=torch.float64).reshape(
                                  observations.in_dim).to(dtype)
        if dtype.is_complex:
            tensor = tensor * (1 + 2j)
        source = tk.decompositions.DenseTensorSource(tensor)
        mask_tensor = torch.ones_like(tensor, dtype=torch.bool)
        mask_tensor.reshape(-1)[0] = False
        weights = torch.linspace(0.5, 2., tensor.numel()).reshape(tensor.shape)
        problem = tk.decompositions.ALSProblem(
            source, weights=weights,
            mask=tk.decompositions.DenseTensorSource(mask_tensor))
        configurations = observations.support
        indices = tuple(configurations.values.unbind(1))
        target = tensor[indices]
        approximation = target + torch.tensor([2., -1.], dtype=dtype)
        expected_weights = weights[indices] * mask_tensor[indices]

        assert problem.in_dim == source.in_dim
        assert problem.out_shape == ()
        assert problem.has_fixed_objective
        torch.testing.assert_close(problem.evaluate(configurations), target)
        torch.testing.assert_close(problem.entry_weights(
            configurations.values), expected_weights)
        absolute, relative = problem.objective_error(
            approximation, configurations)
        expected = ((approximation - target) * expected_weights).norm()
        torch.testing.assert_close(absolute, expected)
        torch.testing.assert_close(
            relative, expected / (target * expected_weights).norm()
        )

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('n_sites', [1, 2, 5])
    def test_observed_objective_uses_stored_support_only(self, dtype, n_sites):
        observations = _observations(dtype, n_sites)
        weights = torch.tensor([2., 0.5], dtype=torch.float64)
        problem = tk.decompositions.ALSProblem(
            observations=observations, weights=weights)
        target = observations.support_values
        approximation = target + target.new_tensor([1., -2.])

        assert problem.in_dim == observations.in_dim
        assert problem.out_shape == ()
        assert problem.has_fixed_objective
        absolute, relative = problem.objective_error(approximation)
        torch.testing.assert_close(
            absolute, ((approximation - target) * weights).norm())
        torch.testing.assert_close(
            relative, absolute / (target * weights).norm())
        reordered = observations.support.values.flip(0)
        torch.testing.assert_close(problem.entry_weights(reordered),
                                   weights.flip(0))
        with pytest.raises(ValueError, match='support'):
            problem.evaluate(observations.support)
        with pytest.raises(ValueError, match='fixed support'):
            problem.objective_error(approximation, observations.support)

    @pytest.mark.parametrize('observed', [False, True])
    @pytest.mark.parametrize('approximation, expected', [([0., 0.], 0.),
                                                       ([1., 0.], math.inf)])
    def test_zero_target_relative_error(
        self, observed, approximation, expected
    ):
        indices = torch.tensor([[0], [1]])
        configurations = tk.decompositions.ConfigurationBatch(indices)
        if observed:
            problem = tk.decompositions.ALSProblem(observations=
                tk.decompositions.SparseTensorSource(
                    indices, torch.zeros(2), (2,)))
        else:
            problem = tk.decompositions.ALSProblem(
                tk.decompositions.DenseTensorSource(torch.zeros(2)))
        absolute, relative = problem.objective_error(
            torch.tensor(approximation), None if observed else configurations)
        assert relative.item() == expected
        assert absolute.item() == torch.tensor(approximation).norm().item()

    def test_sparse_source_and_observation_semantics_are_distinct(self):
        sparse = tk.decompositions.SparseTensorSource(
            torch.tensor([[0, 1]]), torch.tensor([2.]), (2, 2))
        missing = tk.decompositions.ConfigurationBatch(torch.tensor([[1, 1]]))
        known = tk.decompositions.ALSProblem(source=sparse)
        observed = tk.decompositions.ALSProblem(observations=sparse)
        assert known.evaluate(missing).item() == 0
        with pytest.raises(ValueError, match='support'):
            observed.evaluate(missing)
        torch.testing.assert_close(
            observed.objective_error(torch.tensor([2.0]))[0], torch.tensor(0.0)
        )

    def test_sampled_flag_and_unweighted_rows(self):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 3))
        problem = tk.decompositions.ALSProblem(source=source, sampled=True)
        assert not problem.has_fixed_objective
        assert problem.entry_weights(torch.tensor([[0, 1]])) is None

    def test_observed_weights_reject_unknown_queries(self):
        observations = _observations()
        problem = tk.decompositions.ALSProblem(
            observations=observations, weights=torch.ones(2))
        with pytest.raises(ValueError, match='support'):
            problem.entry_weights(torch.tensor([[0, 1]]))

    def test_source_and_observations_can_share_shape(self):
        observations = _observations()
        source = tk.decompositions.DenseTensorSource(
            torch.ones(observations.in_dim))
        problem = tk.decompositions.ALSProblem(source, observations)
        torch.testing.assert_close(problem.evaluate(observations.support),
                                   torch.ones(2))
        assert problem.objective_error(observations.support_values)[0] == 0

    @pytest.mark.parametrize('kwargs, error', [
        ({}, ValueError), ({'source': object()}, TypeError),
        ({'observations': object()}, TypeError),
        ({'sampled': 1}, TypeError), ({'mask': object()}, TypeError),
        ({'weights': [1., 2.]}, TypeError),
        ({'weights': torch.ones(2)}, ValueError),
        ({'weights': torch.ones(2, 3, dtype=torch.long)}, TypeError),
        ({'weights': torch.ones(2, 3, dtype=torch.complex64)}, TypeError),
        ({'weights': torch.full((2, 3), -1.)}, ValueError),
        ({'weights': torch.full((2, 3), float('nan'))}, ValueError),
        ({'weights': torch.ones(2, 3, device='meta')}, ValueError),
        ({'mask': tk.decompositions.DenseTensorSource(torch.ones(2, 3))},
         TypeError),
        ({'mask': tk.decompositions.DenseTensorSource(
            torch.ones(2, 2, dtype=torch.bool))}, ValueError),
    ])
    def test_invalid_constructor_arguments(self, kwargs, error):
        if kwargs and ('source' not in kwargs) and \
                ('observations' not in kwargs):
            kwargs = dict(source=tk.decompositions.DenseTensorSource(
                torch.ones(2, 3)), **kwargs)
        with pytest.raises(error):
            tk.decompositions.ALSProblem(**kwargs)

    @pytest.mark.parametrize('kind', ['empty', 'boolean', 'nonfinite',
                                      'matrix', 'open_output', 'reordered'])
    def test_invalid_observations(self, kind):
        if kind == 'empty':
            observations = tk.decompositions.SparseTensorSource(
                torch.empty(0, 1, dtype=torch.long), torch.empty(0), (2,))
        elif kind == 'matrix':
            observations = tk.decompositions.SparseTensorSource.matrix(
                torch.tensor([[0, 0]]), torch.ones(1), (2,), (2,))
        elif kind == 'open_output':
            observations = tk.decompositions.SparseTensorSource(
                torch.tensor([[0]]), torch.ones(1, 2), (2, 2))
        else:
            indices = torch.tensor([[0, 1]])
            values = (
                torch.ones(1, dtype=torch.bool)
                if kind == 'boolean'
                else torch.tensor(
                    [float('inf') if kind == 'nonfinite' else 1.0]
                )
            )
            observations = tk.decompositions.SparseTensorSource(
                indices, values, (2, 3),
                in_sites=(1, 0) if kind == 'reordered' else None)
        with pytest.raises((TypeError, ValueError)):
            tk.decompositions.ALSProblem(observations=observations)

    @pytest.mark.parametrize('kwargs', [
        {'sampled': True},
        {'mask': tk.decompositions.DenseTensorSource(
            torch.ones(2, 3, dtype=torch.bool))},
        {'weights': torch.ones(2, 3)},
        {'source': tk.decompositions.DenseTensorSource(torch.ones(3, 2))},
        {'source': tk.decompositions.DenseTensorSource(
            torch.ones(2, 3, device='meta'))},
    ])
    def test_invalid_observation_combinations(self, kwargs):
        with pytest.raises(ValueError):
            tk.decompositions.ALSProblem(
                observations=_observations(), **kwargs)

    @pytest.mark.parametrize(
        'kind', ['continuous', 'matrix', 'output', 'batch']
    )
    def test_problem_rejects_non_scalar_or_continuous_sources(self, kind):
        if kind == 'continuous':
            source = tk.decompositions.CallableSource(lambda x: x.sum(-1), 2)
        elif kind == 'matrix':
            source = tk.decompositions.DenseTensorSource.matrix(
                torch.ones(2, 3))
        elif kind == 'output':
            source = tk.decompositions.DenseTensorSource(
                torch.ones(2, 3), in_sites=(0,))
        else:
            source = tk.decompositions.DenseTensorSource(
                torch.ones(2, 3), n_batches=1)
        with pytest.raises(ValueError):
            tk.decompositions.ALSProblem(source)

    def test_objective_error_rejects_invalid_queries_and_values(self):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 3))
        problem = tk.decompositions.ALSProblem(source)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [1, 2]]))
        with pytest.raises(TypeError, match='approximation'):
            problem.objective_error([1., 1.], configurations)
        with pytest.raises(ValueError, match='configurations'):
            problem.objective_error(torch.ones(2))
        with pytest.raises(ValueError, match='target values'):
            problem.objective_error(torch.ones(3), configurations)
        with pytest.raises(ValueError, match='target device'):
            problem.objective_error(torch.ones(
                2, device='meta'), configurations)


class TestFixedEvaluation:  # MARK: TestFixedEvaluation

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_uniform_error_uses_cached_weighted_targets(self, dtype):
        calls = []
        tensor = torch.arange(1., 25., dtype=torch.float64).reshape(2, 3, 4)
        tensor = tensor.to(dtype)
        if dtype.is_complex:
            tensor = tensor * (1 + 2j)

        def function(indices):
            calls.append(indices.clone())
            return tensor[tuple(indices.unbind(1))]

        source = tk.decompositions.CallableTensorSource(
            function, tensor.shape, out_shape=(), dtype=dtype)
        weights = torch.linspace(.1, 2., 24).reshape(tensor.shape).double()
        mask = tk.decompositions.DenseTensorSource(weights > .5)
        problem = tk.decompositions.ALSProblem(
            source, weights=weights, mask=mask, sampled=True)
        generator = torch.Generator().manual_seed(234)
        state = generator.get_state().clone()
        global_state = torch.random.get_rng_state().clone()
        problem.prepare_evaluation(5, generator)
        indices = problem._evaluation_indices.clone()
        target = tensor[tuple(indices.unbind(1))]
        selected = weights[tuple(indices.unbind(1))]
        selected = selected * (selected > .5)
        cached_calls = len(calls)
        for scale in (0., .5, 2.):
            approximation = scale * target
            absolute, relative = problem.evaluation_error(approximation)
            expected = ((approximation - target) * selected).norm()
            torch.testing.assert_close(absolute, expected * (24 / 5) ** .5)
            torch.testing.assert_close(relative, expected /
                                       (target * selected).norm())
        assert len(calls) == cached_calls == 1
        assert torch.equal(indices, problem._evaluation_indices)
        assert torch.equal(generator.get_state(), state)
        assert torch.equal(torch.random.get_rng_state(), global_state)
        assert problem.has_fixed_objective
        assert problem.evaluation_info == {
            'estimated': True, 'n_samples': 5, 'population_size': 24,
            'seed': 235}
        problem.prepare_evaluation(5, generator)
        assert torch.equal(indices, problem._evaluation_indices)

    @pytest.mark.parametrize('shape', [(5,), (2, 3, 2, 2)])
    @pytest.mark.parametrize('budget', [None, 24, 100])
    def test_complete_evaluation_is_the_exact_objective(self, shape, budget):
        tensor = torch.arange(math.prod(shape), dtype=torch.float64)
        tensor = tensor.reshape(shape)
        weights = torch.linspace(.5, 1.5, tensor.numel()).reshape(shape)
        problem = tk.decompositions.ALSProblem(
            tk.decompositions.DenseTensorSource(tensor),
            weights=weights, sampled=True)
        problem.prepare_evaluation(budget)
        absolute, relative = problem.evaluation_error(tensor.flatten() + 1)
        torch.testing.assert_close(absolute, weights.double().norm())
        torch.testing.assert_close(relative, weights.norm() /
                                   (tensor * weights).norm())
        assert not problem.evaluation_info['estimated']
        assert problem.evaluation_info['n_samples'] == tensor.numel()

    def test_observations_evaluate_only_the_complete_known_support(self):
        observed = _observations()
        weights = torch.tensor([2., 3.], dtype=torch.float64)
        problem = tk.decompositions.ALSProblem(
            observations=observed, weights=weights)
        problem.prepare_evaluation(1)
        approximation = observed.support_values + 1
        measured = problem.evaluation_error(approximation)
        exact = problem.objective_error(approximation)
        for a, b in zip(measured, exact):
            torch.testing.assert_close(a, b)
        assert problem.evaluation_info['n_samples'] == 2
        assert not problem.evaluation_info['estimated']

    def test_sampling_does_not_enumerate_large_tensor_domains(self):
        calls = []

        def function(indices):
            calls.append(indices.shape)
            return torch.ones(indices.shape[0], dtype=torch.float64)

        problem = tk.decompositions.ALSProblem(
            tk.decompositions.CallableTensorSource(
                function, (2,) * 70, out_shape=(), dtype=torch.float64),
            sampled=True)
        problem.prepare_evaluation(7)
        absolute, relative = problem.evaluation_error(torch.zeros(7))
        assert calls == [torch.Size((7, 70))]
        assert absolute.item() == pytest.approx((2 ** 70) ** .5)
        assert relative.item() == pytest.approx(1.)

    def test_zero_targets_keep_the_existing_relative_error_convention(self):
        problem = tk.decompositions.ALSProblem(
            tk.decompositions.DenseTensorSource(torch.zeros(10)), sampled=True)
        problem.prepare_evaluation(3)
        assert problem.evaluation_error(torch.zeros(3))[1] == 0
        assert torch.isinf(problem.evaluation_error(torch.ones(3))[1])

    @pytest.mark.parametrize('budget, error', [
        (True, TypeError), (1.5, TypeError), ('4', TypeError),
        (0, ValueError), (-1, ValueError),
    ])
    def test_invalid_evaluation_budgets(self, budget, error):
        problem = tk.decompositions.ALSProblem(
            tk.decompositions.DenseTensorSource(torch.ones(2)))
        with pytest.raises(error):
            problem.prepare_evaluation(budget)
        with pytest.raises(error):
            tk.decompositions.ConvergencePolicy(error_samples=budget)

    def test_preparation_and_prediction_validation(self):
        problem = tk.decompositions.ALSProblem(
            tk.decompositions.DenseTensorSource(torch.ones(2)))
        with pytest.raises(RuntimeError, match='prepare_evaluation'):
            problem.evaluation_error(torch.ones(2))
        with pytest.raises(TypeError, match='generator'):
            problem.prepare_evaluation(generator=1)
        problem.prepare_evaluation(None)
        with pytest.raises(TypeError, match='approximation'):
            problem.evaluation_error([1., 1.])
        with pytest.raises(ValueError, match='match target'):
            problem.evaluation_error(torch.ones(3))

    def test_source_buffer_reuse_preserves_cached_values(self):
        buffer = torch.empty(4, dtype=torch.float64)

        def function(indices):
            buffer.copy_(1 + indices[:, 0])
            return buffer

        source = tk.decompositions.CallableTensorSource(
            function, (20,), out_shape=(), dtype=torch.float64)
        problem = tk.decompositions.ALSProblem(source, sampled=True)
        problem.prepare_evaluation(4, torch.Generator().manual_seed(235))
        reference = 1 + problem._evaluation_indices[:, 0].double()
        source.evaluate(tk.decompositions.ConfigurationBatch(
            torch.zeros(4, 1, dtype=torch.long)))
        absolute, relative = problem.evaluation_error(reference)
        assert absolute == 0
        assert relative == 0

    def test_callable_may_use_autograd_to_compute_its_values(self):
        def function(indices):
            coordinates = indices.double().requires_grad_()
            values = coordinates.square().sum()
            return torch.autograd.grad(values, coordinates)[0].sum(-1)

        problem = tk.decompositions.ALSProblem(
            tk.decompositions.CallableTensorSource(
                function, (10, 10), out_shape=(), dtype=torch.float64),
            sampled=True)
        problem.prepare_evaluation(4)
        reference = 2 * problem._evaluation_indices.sum(1).double()
        assert not problem._evaluation_values.requires_grad
        assert problem.evaluation_error(reference)[0] == 0
