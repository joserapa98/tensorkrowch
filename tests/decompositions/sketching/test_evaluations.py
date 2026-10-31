"""Tests for shared RSS evaluation plans and configuration callbacks."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.sketching.evaluations import (
    ConfigurationMap, _EvaluationPlanBuilder, _EvaluationRequest)
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
    """Plans one two-site Phi and returns its map and handle."""
    builder = _EvaluationPlanBuilder(source)
    handle = _discrete_phi(source).collect(builder, selection)
    return ConfigurationMap(builder.freeze(), batch_size=batch_size,
                            transform=transform), handle


class TestEvaluationPlan:  # MARK: TestEvaluationPlan

    def test_shared_plan_deduplicates_across_phi_operators(self):
        calls = []

        def function(indices):
            calls.append(indices.clone())
            return (indices[:, 0] + 10 * indices[:, 1]).to(torch.float64)

        source = tk.decompositions.CallableTensorSource(
            function, in_dim=(2, 2), dtype=torch.float64)
        builder = _EvaluationPlanBuilder(source)
        first = _discrete_phi(source).collect(builder)
        second = _discrete_phi(source, reverse=True).collect(builder)
        plan = builder.freeze()
        configurations = ConfigurationMap(plan)
        expected = torch.tensor([[0., 10.], [1., 11.]])

        assert plan.stats == tk.decompositions.EvaluationStats(
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

    def test_empty_frozen_builder_and_wrong_source_are_rejected(self):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        builder = _EvaluationPlanBuilder(source)
        with pytest.raises(ValueError, match='contain a request'):
            builder.freeze()
        _discrete_phi(source).collect(builder)
        plan = builder.freeze()
        with pytest.raises(RuntimeError, match='already frozen'):
            builder.freeze()
        with pytest.raises(RuntimeError, match='already frozen'):
            _discrete_phi(source).collect(builder)
        with pytest.raises(TypeError, match='source'):
            _EvaluationPlanBuilder(torch.ones(2))
        with pytest.raises(ValueError, match='share a source'):
            _discrete_phi(source).collect(_EvaluationPlanBuilder(
                tk.decompositions.DenseTensorSource(torch.ones(2, 2))))
        with pytest.raises(TypeError, match='plan'):
            ConfigurationMap(source)
        assert ConfigurationMap(plan)._result(0).shape == (2, 2)

    @pytest.mark.parametrize('shape', [(3,), (-1,), (True,), ('2',)])
    def test_request_shape_is_validated(self, shape):
        batch = tk.decompositions.ConfigurationBatch(torch.tensor([[0], [1]]))
        with pytest.raises(ValueError, match='result_shape'):
            _EvaluationRequest(batch, shape)

    @pytest.mark.parametrize('handle', [True, '0', -1, 3])
    def test_invalid_handles_are_rejected(self, handle):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        configurations, _ = _configuration_map(source)
        with pytest.raises((TypeError, ValueError), match='handle'):
            configurations._result(handle)


class TestConfigurationMap:  # MARK: TestConfigurationMap

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
        builder = _EvaluationPlanBuilder(source)
        handle = phi.collect(builder)
        configurations = ConfigurationMap(builder.freeze(),
                                          transform=lambda table:
                                          table.values + 100)

        assert configurations.values.shape == (4, 2, 3)
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
        phi = PhiOperator(source, (
            (0, torch.tensor([1., 2.], dtype=torch.float64)),
            (1, torch.tensor([[3., 4.]], dtype=torch.float64))),
            _OutputSpec.normalize(torch.ones(1), n_input_sites=2),
            input_kind='features')
        builder = _EvaluationPlanBuilder(source)
        handle = phi.collect(builder)
        configurations = ConfigurationMap(builder.freeze())
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
