"""Tests for global and local value-transform pipelines."""

from dataclasses import FrozenInstanceError

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.decompositions.sketching.evaluations import (
    _EvaluationPlanBuilder,
    _EvaluationSession,
)
from tensorkrowch.decompositions.sketching.phi import (
    PhiOperator,
    _MaterializedPhi,
)
from tensorkrowch.decompositions.sketching.specs import _OutputSpec
from tensorkrowch.decompositions.sketching.transforms import (
    CallableGlobalValueTransform,
    CallableLocalValueTransform,
    CompositeGlobalValueTransform,
    CompositeLocalValueTransform,
    IdentityGlobalValueTransform,
    IdentityLocalValueTransform,
    LocalTransformContext,
    _apply_local_transform,
    _collect_local_queries,
    _prepare_global_transform,
    _structured_path_allowed,
)


def _one_site_phi(source, values=None):
    """Creates a scalar one-site Phi over explicit discrete values."""
    if values is None:
        values = torch.arange(source.input_dim[0])
    output_spec = _OutputSpec.normalize(
        torch.ones(1), n_input_sites=1)
    return PhiOperator(source, ((0, values),), output_spec)


def _global_session(phi, transform, *, context=None, selection=None):
    """Builds the complete prepared global-transform lifecycle for a Phi."""
    builder = _EvaluationPlanBuilder(phi.source)
    handle = phi.collect(builder, selection)
    closure_handle = _prepare_global_transform(builder, transform, context)
    session = _EvaluationSession(
        builder.freeze(), global_transform=transform, context=context)
    return session, handle, closure_handle


class TestGlobalValueTransform:  # MARK: TestGlobalValueTransform

    def test_advanced_transform_types_are_exported(self):
        assert tk.decompositions.EvaluationView is not None
        assert tk.decompositions.GlobalValueTransform is not None
        assert tk.decompositions.LocalValueTransform is not None

    def test_identity_returns_the_exact_value_table_and_allows_structured_path(
            self):
        source = tk.decompositions.DenseTensorSource(torch.tensor([1., 2.]))
        phi = _one_site_phi(source)
        transform = IdentityGlobalValueTransform()
        session, handle, _ = _global_session(phi, transform)

        evaluation = session.view()

        assert transform.apply(evaluation) is evaluation.values
        assert torch.equal(session.result(handle), torch.tensor([1., 2.]))
        assert _structured_path_allowed(transform)

    def test_global_normalization_is_applied_once_before_all_scatter_maps(self):
        source = tk.decompositions.DenseTensorSource(torch.tensor([3., 4.]))
        phi = _one_site_phi(source)
        calls = []

        def normalize(view, context):
            calls.append((view.values.data_ptr(), len(view.incidences)))
            return view.values / torch.linalg.vector_norm(view.values)

        transform = CallableGlobalValueTransform(normalize)
        builder = _EvaluationPlanBuilder(source)
        first = phi.collect(builder)
        second = phi.collect(builder, torch.tensor([[1], [0], [1]]))
        _prepare_global_transform(builder, transform)
        session = _EvaluationSession(
            builder.freeze(), global_transform=transform)

        assert torch.allclose(
            session.result(first), torch.tensor([0.6, 0.8]))
        assert torch.allclose(
            session.result(second), torch.tensor([0.8, 0.6, 0.8]))
        assert len(calls) == 1
        assert calls[0][1] == 2
        assert session.stats.requested_points == 5
        assert session.stats.unique_points == 2
        assert not _structured_path_allowed(transform)

    def test_closure_points_are_declared_before_freeze_and_transformed_together(
            self):
        calls = []

        def function(indices):
            calls.append(indices.shape[0])
            return indices[:, 0].to(torch.float64)

        source = tk.decompositions.CallableTensorSource(
            function,
            input_dim=(3,),
            dtype=torch.float64)
        phi = _one_site_phi(source, torch.tensor([0, 1]))

        def required_points(view, context):
            assert view.phase == 'collect'
            return tk.decompositions.ConfigurationBatch(torch.tensor([[2]]))

        transform = CallableGlobalValueTransform(
            lambda view, context: view.values - view.values.mean(),
            required_points_function=required_points)
        session, handle, closure_handle = _global_session(phi, transform)

        assert torch.equal(session.result(handle), torch.tensor([-1., 0.]))
        assert torch.equal(
            session.result(closure_handle), torch.tensor([1.]))
        assert calls == [3]
        assert session.stats.requested_points == 3
        assert session.stats.unique_points == 3

    def test_non_identity_transform_requires_explicit_preparation(self):
        source = tk.decompositions.DenseTensorSource(torch.tensor([1., 2.]))
        phi = _one_site_phi(source)
        transform = CallableGlobalValueTransform(
            lambda view, context: view.values)
        builder = _EvaluationPlanBuilder(source)
        phi.collect(builder)
        plan = builder.freeze()

        with pytest.raises(RuntimeError, match='prepared before freeze'):
            _EvaluationSession(plan, global_transform=transform)
        with pytest.raises(RuntimeError, match='after freeze'):
            _prepare_global_transform(builder, transform)

    def test_composite_global_transform_preserves_declared_order(self):
        source = tk.decompositions.DenseTensorSource(torch.tensor([1., 2.]))
        phi = _one_site_phi(source)
        add = CallableGlobalValueTransform(
            lambda view, context: view.values + context['offset'])
        scale = CallableGlobalValueTransform(
            lambda view, context: view.values * context['scale'])
        transform = CompositeGlobalValueTransform((add, scale))
        session, handle, _ = _global_session(
            phi, transform, context={'offset': 1, 'scale': 3})

        assert torch.equal(session.result(handle), torch.tensor([6., 9.]))

    def test_request_order_does_not_change_transformed_phi_values(self):
        source = tk.decompositions.DenseTensorSource(torch.tensor([3., 4.]))
        phi = _one_site_phi(source)
        transform = CallableGlobalValueTransform(
            lambda view, context: view.values /
            torch.linalg.vector_norm(view.values))
        selections = (
            torch.tensor([[0], [1]]),
            torch.tensor([[1], [0], [1]]),
        )

        results = []
        for order in ((0, 1), (1, 0)):
            builder = _EvaluationPlanBuilder(source)
            handles = {
                key: phi.collect(builder, selections[key]) for key in order}
            _prepare_global_transform(builder, transform)
            session = _EvaluationSession(
                builder.freeze(), global_transform=transform)
            results.append(tuple(session.result(handles[key])
                                 for key in range(2)))

        assert all(torch.equal(left, right)
                   for left, right in zip(*results))

    @pytest.mark.parametrize(
        'function, error, match',
        [
            (lambda view, context: [1, 2], TypeError, 'torch.Tensor'),
            (lambda view, context: view.values[:-1], ValueError, 'shape'),
        ])
    def test_global_transform_validates_return_contract(
            self, function, error, match):
        source = tk.decompositions.DenseTensorSource(torch.tensor([1., 2.]))
        phi = _one_site_phi(source)
        transform = CallableGlobalValueTransform(function)
        session, handle, _ = _global_session(phi, transform)

        with pytest.raises(error, match=match):
            session.result(handle)


class TestLocalValueTransform:  # MARK: TestLocalValueTransform

    def test_identity_returns_the_exact_phi_view_without_materializing(self):
        source = tk.decompositions.DenseTensorSource(torch.tensor([1., 2.]))
        phi = _one_site_phi(source)
        transform = IdentityLocalValueTransform()

        result = _apply_local_transform(transform, phi)

        assert result is phi
        assert source.evaluation_stats.source_calls == 0

    def test_local_queries_are_collected_before_global_preparation(self):
        source = tk.decompositions.DenseTensorSource(
            torch.arange(4.).reshape(2, 2))
        output_spec = _OutputSpec.normalize(torch.ones(1), n_input_sites=2)
        phi = PhiOperator(
            source,
            ((0, torch.arange(2)), (1, torch.arange(2))),
            output_spec)
        query = torch.tensor([[0, 0], [1, 1]])
        transform = CallableLocalValueTransform(
            lambda view, context: _MaterializedPhi(
                view.materialize() + context.query_results[0].mean(),
                view.layout),
            required_queries_function=lambda view, context: (query,))
        builder = _EvaluationPlanBuilder(source)
        main_handle = phi.collect(builder)
        query_handles = _collect_local_queries(builder, phi, transform)
        _prepare_global_transform(builder, IdentityGlobalValueTransform())
        with pytest.raises(RuntimeError, match='global preparation'):
            _collect_local_queries(builder, phi, transform)
        session = _EvaluationSession(
            builder.freeze(),
            global_transform=IdentityGlobalValueTransform())
        materialized = _MaterializedPhi(
            session.result(main_handle), phi.layout)
        query_results = tuple(session.result(handle)
                              for handle in query_handles)

        result = _apply_local_transform(
            transform,
            materialized,
            evaluation=session.view(),
            query_results=query_results)

        assert len(query_results) == 1
        assert torch.equal(query_results[0], torch.tensor([0., 3.]))
        assert torch.equal(
            result.materialize(),
            torch.arange(4.).reshape(2, 2) + 1.5)

    def test_composite_local_transform_preserves_order_and_context(self):
        base = _MaterializedPhi(torch.tensor([1., 2.]), layout=(0,))

        def add(view, context):
            return _MaterializedPhi(
                view.materialize() + context.data['offset'], view.layout)

        def scale(view, context):
            return _MaterializedPhi(
                view.materialize() * context.data['scale'], view.layout)

        transform = CompositeLocalValueTransform((
            CallableLocalValueTransform(add),
            CallableLocalValueTransform(scale),
        ))

        result = _apply_local_transform(
            transform,
            base,
            data={'offset': 1, 'scale': 3})

        assert torch.equal(result.materialize(), torch.tensor([6., 9.]))

    def test_local_callable_should_return_a_phi_view(self):
        base = _MaterializedPhi(torch.tensor([1., 2.]), layout=(0,))
        transform = CallableLocalValueTransform(
            lambda view, context: view.materialize())

        with pytest.raises(TypeError, match='return a PhiView'):
            _apply_local_transform(transform, base)

    def test_empirical_values_admit_a_generic_local_kernel_transform(self):
        source = tk.decompositions.EmpiricalDistribution(
            dataset=torch.tensor([[0], [0], [2], [2]]),
            input_dim=(3,))
        phi = _one_site_phi(source)
        kernel = torch.tensor([
            [0.75, 0.25, 0.00],
            [0.25, 0.50, 0.25],
            [0.00, 0.25, 0.75],
        ])
        transform = CallableLocalValueTransform(
            lambda view, context: _MaterializedPhi(
                context.data @ view.materialize(), view.layout))
        materialized = _MaterializedPhi(phi.materialize(), phi.layout)

        smoothed = _apply_local_transform(
            transform, materialized, data=kernel)

        assert torch.allclose(
            smoothed.materialize(), torch.tensor([0.375, 0.25, 0.375]))

    def test_local_context_is_frozen_and_validates_query_results(self):
        context = LocalTransformContext(
            data={'name': 'test'},
            query_results=(torch.ones(2),))

        assert len(context.query_results) == 1
        assert torch.equal(context.query_results[0], torch.ones(2))
        with pytest.raises(FrozenInstanceError):
            context.query_results = ()
