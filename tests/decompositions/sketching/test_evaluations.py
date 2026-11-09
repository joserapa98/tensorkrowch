"""Tests for shared RSS evaluation plans and configuration callbacks."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.sketching.evaluations import (
    ConfigurationMap)
from tensorkrowch.decompositions.sketching.phi import PhiOperator
from tensorkrowch.decompositions.sketching.specs import _OutputSpec


def _discrete_phi(source, reverse=False):
    """Creates a full two-site Phi, optionally reversing its axes."""
    output_spec = _OutputSpec.normalize(torch.ones(1), n_input_sites=2)
    components = ((0, torch.arange(2)), (1, torch.arange(2)))
    if reverse:
        components = tuple(reversed(components))
    return PhiOperator(source, components, output_spec)


def _configuration_map(source, *, batch_size=64, transform=None,
                       selection=None):
    configurations = ConfigurationMap(source, batch_size=batch_size,
                                      transform=transform)
    handle = configurations.register(_discrete_phi(source), selection)
    return configurations, handle


class TestEvaluationPlan:  # MARK: TestEvaluationPlan

    def test_shared_plan_deduplicates_across_phi_operators(self):
        calls = []

        def function(indices):
            calls.append(indices.clone())
            return (indices[:, 0] + 10 * indices[:, 1]).to(torch.float64)

        source = tk.decompositions.CallableTensorSource(
            function, in_dim=(2, 2), dtype=torch.float64)
        builder = ConfigurationMap(source)
        first = _discrete_phi(source).collect(builder)
        second = _discrete_phi(source, reverse=True).collect(builder)
        builder.freeze()
        configurations = builder
        expected = torch.tensor([[0., 10.], [1., 11.]])

        assert configurations.stats == tk.decompositions.EvaluationStats(
            requested_points=8, unique_points=4, cache_hits=4)
        assert torch.equal(configurations._result(first), expected)
        assert torch.equal(configurations._result(second), expected.T)
        configurations._result(first)
        assert len(calls) == 1
        assert calls[0].shape == (4, 2)
        assert configurations.stats == tk.decompositions.EvaluationStats(
            requested_points=8, unique_points=4, batches=1,
            cache_hits=4, source_calls=1)
        assert not hasattr(configurations, '_results')

    def test_batching_is_deterministic_and_counted(self):
        calls = []

        def function(indices):
            calls.append(indices.shape[0])
            return indices.sum(dim=1).to(torch.float64)

        source = tk.decompositions.CallableTensorSource(
            function, in_dim=(2, 2), dtype=torch.float64)
        configurations, handle = _configuration_map(source, batch_size=2)

        assert configurations._result(handle).shape == (2, 2)
        assert calls == [2, 2]
        assert configurations.stats.batches == 2
        assert configurations.stats.source_calls == 2

    def test_empty_frozen_map_and_wrong_source_are_rejected(self):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        configurations = ConfigurationMap(source)
        with pytest.raises(ValueError, match='registered Phi'):
            configurations.freeze()
        configurations.register(_discrete_phi(source))
        configurations.freeze()
        configurations.freeze()
        with pytest.raises(RuntimeError, match='already frozen'):
            configurations.register(_discrete_phi(source))
        with pytest.raises(TypeError, match='source'):
            ConfigurationMap(torch.ones(2))
        other = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        with pytest.raises(ValueError, match='share a source'):
            ConfigurationMap(other).register(_discrete_phi(source))
        assert configurations._result(0).shape == (2, 2)

    @pytest.mark.parametrize('selection', [
        torch.zeros(1, 3, dtype=torch.long), torch.tensor([[2, 0]]),
        torch.zeros(1, 2), torch.tensor([[-1, 0]])])
    def test_registered_selection_is_validated(self, selection):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        with pytest.raises((TypeError, ValueError)):
            ConfigurationMap(source).register(_discrete_phi(source), selection)

    @pytest.mark.parametrize('handle', [True, '0', -1, 3])
    def test_invalid_handles_are_rejected(self, handle):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        configurations, _ = _configuration_map(source)
        with pytest.raises((TypeError, ValueError), match='handle'):
            configurations._result(handle)


class TestConfigurationMap:  # MARK: TestConfigurationMap

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('singleton', [False, True])
    @pytest.mark.parametrize('partial', [False, True])
    def test_phi_indices_preserve_shapes_order_and_gradients(
            self, dtype, singleton, partial):
        def function(indices):
            values = (indices[:, 0] + 10 * indices[:, 1]).to(dtype)
            if dtype.is_complex:
                values = values * (1 + 2j)
            return values[:, None] if singleton else values

        source = tk.decompositions.CallableTensorSource(
            function, in_dim=(2, 2, 1) if singleton else (2, 2),
            in_sites=(0, 1), out_shape=(1,) if singleton else (), dtype=dtype)
        builder = ConfigurationMap(source)
        selection = torch.tensor([[[1, 0], [0, 1], [1, 0]]]) \
            if partial else None
        phis = [_discrete_phi(source, reverse=reverse)
                for reverse in (False, True)]
        handles = [builder.register(phi, selection) for phi in phis]
        indices = builder.phi_indices
        assert builder.phi_indices is indices
        assert len(indices) == len(phis)

        parameter = builder.values.detach().clone().requires_grad_()
        actual = parameter.real.sum() * 0
        counts = torch.zeros_like(parameter.real)
        for handle, phi in zip(handles, phis):
            tensor = parameter[indices[handle]]
            shape = selection.shape[:-1] if partial else phi.shape
            assert tensor.shape == shape
            assert all(index.shape == shape for index in indices[handle])
            assert len(indices[handle]) == (2 if singleton else 1)
            torch.testing.assert_close(tensor, builder._result(handle))
            expected_values = function(phi.configuration_batch(
                selection).values).reshape(shape)
            torch.testing.assert_close(tensor, expected_values)
            actual = actual + tensor.abs().square().sum()
            counts.index_put_(indices[handle], torch.ones_like(tensor.real),
                              accumulate=True)

        actual.backward()
        expected_gradient = 2 * parameter.detach() * counts
        torch.testing.assert_close(parameter.grad, expected_gradient)

    @pytest.mark.parametrize('samples, present', [
        ([[1, 1], [3, 3], [0, 1], [1, 1]], [0, 2, 3]),
        ([[3, 3], [2, 2]], []),
        ([], []),
    ])
    def test_sample_indices_only_match_initial_queries(self, samples, present):
        calls = []

        def function(indices):
            calls.append(indices.clone())
            return indices.sum(-1).to(torch.float64)

        source = tk.decompositions.CallableTensorSource(
            function, in_dim=(4, 4), dtype=torch.float64)
        samples = torch.tensor(samples, dtype=torch.long).reshape(-1, 2)
        builder = ConfigurationMap(source, sketch_samples=samples)
        builder.register(_discrete_phi(source))
        rows, sample_rows = builder.sample_indices()

        assert not calls
        assert torch.equal(sample_rows,
                           torch.tensor(present, dtype=torch.long))
        assert rows.dtype == sample_rows.dtype == torch.long
        assert torch.equal(builder.configurations[rows], samples[sample_rows])
        assert builder.sample_indices()[0] is rows
        before = builder.stats
        builder.values
        assert before.unique_points == builder.stats.unique_points == 4

        # Auxiliary queries never enter the initial indexed value table.
        builder.evaluate(torch.tensor([[3, 3], [2, 2]]))
        assert builder.sample_indices()[0] is rows
        assert torch.equal(builder.sample_indices()[1], sample_rows)
        assert builder.values.shape == (4,)
        if len(present) == 3:
            assert rows[0] == rows[2]

    def test_sample_indices_require_original_samples(self):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        builder, _ = _configuration_map(source)
        with pytest.raises(RuntimeError, match='sketch_samples'):
            builder.sample_indices()

    @pytest.mark.parametrize('samples', [1, True, 'samples', torch.zeros(2)])
    def test_invalid_sketch_samples_are_rejected(self, samples):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        with pytest.raises((TypeError, ValueError)):
            ConfigurationMap(source, sketch_samples=samples)

    @pytest.mark.parametrize('samples', [
        torch.zeros(1, 1, dtype=torch.long),
        tk.decompositions.ConfigurationBatch(torch.zeros(1, 2),
                                              kind='features'),
    ])
    def test_incompatible_sketch_samples_are_rejected(self, samples):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        builder = ConfigurationMap(source, sketch_samples=samples)
        builder.register(_discrete_phi(source))
        with pytest.raises(ValueError, match='input count|value kind'):
            builder.sample_indices()

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_transform_auxiliary_queries_preserve_initial_tables(self, dtype):
        calls = []
        transforms = []

        def function(indices):
            calls.append(indices.clone())
            values = (indices[:, 0] + 10 * indices[:, 1]).to(dtype)
            return values * (1 + 1j) if dtype.is_complex else values

        def transform(configurations):
            transforms.append(configurations)
            initial_inputs = configurations.configurations
            initial_values = configurations.values
            extra = torch.tensor([[0, 0], [1, 1], [1, 1], [0, 1]])
            first = configurations.evaluate(extra)
            second = configurations.evaluate(extra.flip(0))
            assert torch.equal(second, first.flip(0))
            assert configurations.configurations is initial_inputs
            assert configurations.values is initial_values
            assert initial_inputs.shape == (1, 2)
            return initial_values + configurations.evaluate(
                torch.tensor([[1, 1]]))

        source = tk.decompositions.CallableTensorSource(
            function, in_dim=(2, 2), dtype=dtype)
        configurations, handle = _configuration_map(
            source, batch_size=1, transform=transform,
            selection=torch.tensor([[0, 0]]))
        result = configurations._result(handle)

        factor = 1 + 1j if dtype.is_complex else 1
        assert torch.equal(result, torch.tensor([11 * factor], dtype=dtype))
        assert len(transforms) == 1
        assert len(calls) == 3
        assert configurations.stats.unique_points == 3
        assert configurations.stats.requested_points == 10
        assert configurations.stats.cache_hits == 7
        assert configurations._result(handle).data_ptr() != result.data_ptr()
        assert len(transforms) == 1

    def test_tensor_output_is_cached_before_component_selection(self):
        def function(indices):
            base = (indices[:, 0] + 10 * indices[:, 1]).to(torch.float32)
            return base[:, None, None] + torch.arange(6).reshape(1, 2, 3)

        source = tk.decompositions.CallableTensorSource(
            function, in_dim=(2, 2, 2, 3), in_sites=(0, 1),
            out_shape=(2, 3), dtype=torch.float32)
        output_spec = _OutputSpec.normalize(
            torch.ones(1, 2, 3), n_input_sites=2, out_position=(2, 3))
        phi = PhiOperator(source, tuple((site, torch.arange(dim))
                                        for site, dim in enumerate(
                                            (2, 2, 2, 3))), output_spec)
        builder = ConfigurationMap(source)
        handle = phi.collect(builder)
        configurations = builder
        configurations._transform = lambda table: table.values + 100

        assert configurations.values.shape == (4, 2, 3)
        indices = configurations.phi_indices[handle]
        assert len(indices) == 3
        assert all(index.shape == phi.shape for index in indices)
        expected = function(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
        torch.testing.assert_close(configurations.values[indices],
                                   expected.reshape(phi.shape))
        result = configurations._result(handle)
        assert result.shape == phi.shape
        assert torch.equal(result[1, 1], function(
            torch.tensor([[1, 1]]))[0] + 100)
        extra = configurations.evaluate(torch.tensor([[1, 1], [0, 0]]))
        assert extra.shape == (2, 2, 3)
        assert torch.equal(extra, function(torch.tensor([[1, 1], [0, 0]])))

    def test_heterogeneous_feature_inputs_and_gradients(self):
        scale = torch.tensor(2., dtype=torch.float64, requires_grad=True)

        def function(values):
            scalar, vector = values
            return scale * (scalar + vector.sum(-1))

        source = tk.decompositions.CallableSource(
            function, n_sites=2, dtype=torch.float64)
        phi = PhiOperator(
            source, (
                (0, torch.tensor([1., 2.], dtype=torch.float64)),
                (1, torch.tensor([[3., 4.]], dtype=torch.float64))),
            _OutputSpec.normalize(torch.ones(1), n_input_sites=2),
            input_kind='features')
        builder = ConfigurationMap(source)
        handle = phi.collect(builder)
        configurations = builder
        extra = (torch.tensor([2., 3., 3.], dtype=torch.float64),
                 torch.tensor([[3., 4.], [4., 5.], [4., 5.]],
                              dtype=torch.float64))

        assert isinstance(configurations.configurations, tuple)
        result = configurations.evaluate(extra)
        assert torch.equal(result, torch.tensor([18., 24., 24.],
                                                dtype=torch.float64))
        (result.sum() + configurations._result(handle).sum()).backward()
        assert scale.grad.item() == 50
        assert configurations.stats.unique_points == 3

    def test_heterogeneous_samples_match_without_evaluation(self):
        calls = []

        def function(values):
            calls.append(values)
            return values[0] + values[1].sum(-1)

        source = tk.decompositions.CallableSource(
            function, n_sites=2, dtype=torch.float64)
        samples = (
            torch.tensor([2., 3., 1., 2.], dtype=torch.float64),
            torch.tensor([[3., 4.], [3., 4.], [3., 4.], [3., 4.]],
                         dtype=torch.float64))
        builder = ConfigurationMap(source, sketch_samples=samples)
        phi = PhiOperator(source, (
            (0, torch.tensor([1., 2.], dtype=torch.float64)),
            (1, torch.tensor([[3., 4.]], dtype=torch.float64))),
            _OutputSpec.normalize(torch.ones(1), n_input_sites=2))
        builder.register(phi)
        rows, sample_rows = builder.sample_indices()

        assert not calls
        assert torch.equal(sample_rows, torch.tensor([0, 2, 3]))
        assert torch.equal(rows, torch.tensor([1, 0, 1]))

    @pytest.mark.parametrize('batch_size', [True, 0, -1, 1.5])
    def test_batch_size_is_validated(self, batch_size):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        with pytest.raises((TypeError, ValueError), match='batch_size'):
            _configuration_map(source, batch_size=batch_size)

    @pytest.mark.parametrize('transform', [lambda table: 1,
                                           lambda table: table.values[:1]])
    def test_transform_must_preserve_tensor_shape(self, transform):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        configurations, handle = _configuration_map(
            source, transform=transform)
        with pytest.raises((TypeError, ValueError), match='transform'):
            configurations._result(handle)

    def test_invalid_auxiliary_feature_shapes_and_kind_are_rejected(self):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        configurations, _ = _configuration_map(source)
        with pytest.raises(ValueError, match='input count'):
            configurations.evaluate(torch.zeros(2, 1, dtype=torch.long))
        with pytest.raises(TypeError, match='integers'):
            configurations.evaluate(torch.zeros(2, 2))
