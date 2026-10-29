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
