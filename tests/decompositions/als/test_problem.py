"""Tests for fixed observations and ALS problem semantics."""

import math

import pytest

import torch
import tensorkrowch as tk


class TestObservedEntries:  # MARK: TestObservedEntries

    def test_repeated_equal_observations_are_deduplicated(self):
        observations = tk.decompositions.ObservedEntries(
            indices=torch.tensor([[1, 2], [0, 1], [1, 2]]),
            values=torch.tensor([4., 3., 4.]),
            input_dim=(2, 3),
            weights=torch.tensor([2., 1., 2.]))

        assert torch.equal(observations.indices,
                           torch.tensor([[0, 1], [1, 2]]))
        assert torch.equal(observations.flat_ids, torch.tensor([1, 5]))
        assert torch.equal(observations.values, torch.tensor([3., 4.]))
        assert torch.equal(observations.weights, torch.tensor([1., 2.]))

    @pytest.mark.parametrize(
        'values, weights, match',
        [
            (torch.tensor([1., 2.]), None, 'identical values'),
            (torch.tensor([1., 1.]), torch.tensor([1., 2.]),
             'identical weights'),
        ])
    def test_conflicting_repeated_observations_are_rejected(
            self, values, weights, match):
        with pytest.raises(ValueError, match=match):
            tk.decompositions.ObservedEntries(
                indices=torch.tensor([[0, 1], [0, 1]]),
                values=values,
                input_dim=(2, 2),
                weights=weights)

    def test_weighted_absolute_and_relative_errors(self):
        observations = tk.decompositions.ObservedEntries(
            indices=torch.tensor([[0, 0], [1, 1]]),
            values=torch.tensor([3., 4.]),
            input_dim=(2, 2),
            weights=torch.tensor([2., 0.5]))

        absolute, relative = observations.error(torch.tensor([2., 6.]))

        expected_absolute = torch.tensor([2., 1.]).norm()
        expected_denominator = torch.tensor([6., 2.]).norm()
        assert absolute == expected_absolute
        assert relative == expected_absolute / expected_denominator

    @pytest.mark.parametrize(
        'approximation, expected',
        [(torch.tensor([0., 0.]), 0.), (torch.tensor([1., 0.]), math.inf)])
    def test_zero_target_relative_error_policy(self,
                                               approximation,
                                               expected):
        observations = tk.decompositions.ObservedEntries(
            indices=torch.tensor([[0, 0], [1, 1]]),
            values=torch.tensor([0., 0.]),
            input_dim=(2, 2))

        _, relative = observations.error(approximation)

        assert relative.item() == expected


class TestALSProblem:  # MARK: TestALSProblem

    def test_source_objective_uses_requested_configurations(self):
        tensor = torch.tensor([[1., 2.], [3., 4.]])
        source = tk.decompositions.DenseTensorSource(tensor)
        problem = tk.decompositions.ALSProblem(source=source)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [1, 1]]))

        absolute, relative = problem.objective_error(
            torch.tensor([2., 2.]), configurations)

        assert absolute == torch.tensor([1., -2.]).norm()
        assert relative == absolute / torch.tensor([1., 4.]).norm()
        assert problem.has_fixed_objective

    def test_completion_objective_never_queries_unknown_entries(self):
        observations = tk.decompositions.ObservedEntries(
            indices=torch.tensor([[0, 1], [1, 0]]),
            values=torch.tensor([2., 3.]),
            input_dim=(2, 2))
        problem = tk.decompositions.ALSProblem(observations=observations)

        absolute, relative = problem.objective_error(torch.tensor([1., 5.]))

        assert absolute == torch.tensor([-1., 2.]).norm()
        assert relative == absolute / observations.values.norm()
        with pytest.raises(ValueError, match='known only'):
            problem.evaluate(tk.decompositions.ConfigurationBatch(
                torch.tensor([[0, 0]])))

    def test_sparse_zero_and_unobserved_value_have_distinct_semantics(self):
        sparse = tk.decompositions.SparseTensorSource(
            indices=torch.tensor([[0, 1]]),
            values=torch.tensor([2.]),
            input_dim=(2, 2))
        missing = tk.decompositions.ConfigurationBatch(
            torch.tensor([[1, 1]]))
        observations = tk.decompositions.ObservedEntries(
            indices=torch.tensor([[0, 1]]),
            values=torch.tensor([2.]),
            input_dim=(2, 2))
        completion = tk.decompositions.ALSProblem(
            observations=observations)

        assert sparse.evaluate(missing).item() == 0
        with pytest.raises(ValueError, match='known only'):
            completion.evaluate(missing)

    def test_source_and_observations_require_matching_shapes(self):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 3))
        observations = tk.decompositions.ObservedEntries(
            indices=torch.tensor([[0, 0]]),
            values=torch.tensor([1.]),
            input_dim=(2, 2))

        with pytest.raises(ValueError, match='input dimensions'):
            tk.decompositions.ALSProblem(
                source=source, observations=observations)
