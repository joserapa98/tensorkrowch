"""Tests for sketching/evaluations."""


import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.sketching.evaluations import (EvaluationView,
                                                               _EvaluationPlanBuilder,
                                                               _EvaluationSession)
from tensorkrowch.decompositions.sketching.phi import PhiOperator
from tensorkrowch.decompositions.sketching.specs import _OutputSpec


def _scalar_output_spec(n_inputs):
    """Creates the scalar output layout used by most Phi tests."""
    return _OutputSpec.normalize(torch.ones(1), n_input_sites=n_inputs)


def _discrete_phi(source, reverse=False):
    """Creates a full two-site discrete Phi, optionally reversing its axes."""
    output_spec = _scalar_output_spec(2)
    components = (
        (0, torch.arange(2)),
        (1, torch.arange(2)),
    )
    if reverse:
        components = tuple(reversed(components))
    return PhiOperator(source, components, output_spec)


class TestEvaluationPlan:  # MARK: TestEvaluationPlan

    def test_shared_plan_deduplicates_points_across_phi_operators(self):
        calls = []

        def function(indices):
            calls.append(indices.clone())
            return (indices[:, 0] + 10 * indices[:, 1]).to(torch.float64)

        source = tk.decompositions.CallableTensorSource(
            function,
            in_dim=(2, 2),
            dtype=torch.float64)
        first = _discrete_phi(source)
        second = _discrete_phi(source, reverse=True)
        builder = _EvaluationPlanBuilder(source)

        first_handle = first.collect(builder)
        second_handle = second.collect(builder)
        snapshot = builder.snapshot()
        plan = builder.freeze()
        session = _EvaluationSession(plan)

        assert isinstance(snapshot, EvaluationView)
        assert snapshot.phase == 'collect'
        assert snapshot.configurations.batch_size == 8
        assert plan.configurations.batch_size == 4
        assert plan.stats == tk.decompositions.EvaluationStats(
            requested_points=8,
            unique_points=4,
            cache_hits=4)
        expected = torch.tensor([[0., 10.], [1., 11.]])
        assert torch.equal(session.result(first_handle), expected)
        assert torch.equal(session.result(second_handle), expected.T)
        assert len(calls) == 1
        assert calls[0].shape == (4, 2)
        session.evaluate()
        assert len(calls) == 1
        assert session.stats == tk.decompositions.EvaluationStats(
            requested_points=8,
            unique_points=4,
            batches=1,
            cache_hits=4,
            source_calls=1)
        assert session.view().phase == 'evaluated'

    def test_session_can_release_scattered_phi_but_keep_unique_values(self):
        calls = []

        def function(indices):
            calls.append(indices.clone())
            return (indices[:, 0] + 10 * indices[:, 1]).to(torch.float64)

        source = tk.decompositions.CallableTensorSource(
            function,
            in_dim=(2, 2),
            dtype=torch.float64)
        builder = _EvaluationPlanBuilder(source)
        first = _discrete_phi(source).collect(builder)
        second = _discrete_phi(source, reverse=True).collect(builder)
        session = _EvaluationSession(builder.freeze())

        session.prepare_values()

        assert session._results == [None, None]
        expected = torch.tensor([[0., 10.], [1., 11.]])
        assert torch.equal(session.result(first), expected)
        assert session._results[first] is not None
        assert session._results[second] is None
        session.release(first)
        assert session._results[first] is None
        assert torch.equal(session.result(first), expected)
        assert len(calls) == 1

    def test_session_batching_is_deterministic_and_counted_at_source_level(self):
        calls = []

        def function(indices):
            calls.append(indices.shape[0])
            return indices.sum(dim=1).to(torch.float64)

        source = tk.decompositions.CallableTensorSource(
            function,
            in_dim=(2, 2),
            dtype=torch.float64)
        phi = _discrete_phi(source)
        builder = _EvaluationPlanBuilder(source)
        handle = phi.collect(builder)
        session = _EvaluationSession(builder.freeze())

        result = session.result(handle, batch_size=2)

        assert result.shape == (2, 2)
        assert calls == [2, 2]
        assert session.stats.batches == 2
        assert session.stats.source_calls == 2

    def test_expand_adds_closure_points_and_rejects_late_phi_requests(self):
        source = tk.decompositions.CallableTensorSource(
            lambda indices: indices.sum(dim=1).to(torch.float64),
            in_dim=(2, 2),
            dtype=torch.float64)
        phi = _discrete_phi(source)
        builder = _EvaluationPlanBuilder(source)
        selection = torch.tensor([[0, 0]])
        phi.collect(builder, selection)
        closure = tk.decompositions.ConfigurationBatch(
            torch.tensor([[1, 1], [0, 1]]))

        closure_handle = builder.expand(closure)

        assert builder.phase == 'expand'
        assert builder.snapshot().phase == 'expand'
        with pytest.raises(RuntimeError, match='before expand'):
            phi.collect(builder)
        plan = builder.freeze()
        session = _EvaluationSession(plan)
        assert torch.equal(
            session.result(closure_handle), torch.tensor([2., 1.]))
        with pytest.raises(RuntimeError, match='after freeze'):
            builder.expand(closure)
        with pytest.raises(RuntimeError, match='already frozen'):
            builder.freeze()

    def test_empty_builder_and_invalid_handles_are_rejected(self):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        builder = _EvaluationPlanBuilder(source)

        with pytest.raises(ValueError, match='contain a request'):
            builder.freeze()
        phi = _discrete_phi(source)
        phi.collect(builder)
        session = _EvaluationSession(builder.freeze())
        with pytest.raises(ValueError, match='outside'):
            session.result(2)
