"""Tests retained for redistribution."""

from itertools import product

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.sketching.evaluations import (_EvaluationPlanBuilder)
from tensorkrowch.decompositions.sketching.phi import (PhiOperator,
                                                       PhiView,
                                                       _MaterializedPhi)
from tensorkrowch.decompositions.sketching.regions import (SiteRegion,
                                                           _SamplePool)
from tensorkrowch.decompositions.sketching.specs import _OutputSpec

from tests.decompositions.als._oracles import contract_tt_dense, make_tt_cores


def _scalar_output_spec(n_inputs):
    """Creates the scalar output layout used by most Phi tests."""
    return _OutputSpec.normalize(torch.ones(1), n_input_sites=n_inputs)


class TestPhiOperator:  # MARK: TestPhiOperator

    def test_coordinate_phi_matches_explicit_cartesian_evaluation(self):
        source = tk.decompositions.CallableTensorSource(
            lambda values: (
                values[:, 0] + 10 * values[:, 1] + 100 * values[:, 2]
            ).to(torch.float64),
            in_dim=(2, 2, 2),
            dtype=torch.float64)
        pool = _SamplePool(torch.tensor([
            [0., 0., 3.],
            [1., 0., 4.],
            [0., 1., 4.],
        ]))
        left = pool.restrict(SiteRegion((0,)))
        right = pool.restrict(SiteRegion((2,)))
        middle_values = torch.tensor([-1., 2.])
        phi = PhiOperator(
            source,
            (left, (1, middle_values), right),
            _scalar_output_spec(3),
            input_kind='features')

        result = phi.materialize(batch_size=3)

        expected = torch.empty(2, 2, 2, dtype=torch.float64)
        for left_id, middle_id, right_id in product(range(2), repeat=3):
            expected[left_id, middle_id, right_id] = (
                left.values[0][left_id] +
                10 * middle_values[middle_id] +
                100 * right.values[0][right_id])
        assert torch.equal(result, expected)
        assert phi.shape == (2, 2, 2)
        assert phi.layout == (left.region, 1, right.region)
        assert phi.configuration_batch().batch_size == 8

    def test_functional_fiber_accepts_new_axis_values(self):
        source = tk.decompositions.CallableTensorSource(
            lambda values: (
                values[:, 0] + 10 * values[:, 1] + 100 * values[:, 2]
            ).to(torch.float64),
            in_dim=(2, 2, 2),
            dtype=torch.float64)
        phi = PhiOperator(
            source,
            tuple((site, torch.tensor([0., 1.])) for site in range(3)),
            _scalar_output_spec(3),
            input_kind='features')
        values = torch.tensor([-1., 0.5, 2.])

        fiber = phi.fiber_at(
            axis=1,
            values=values,
            fixed_indices=torch.tensor([1, 0]))
        replaced = phi.with_axis_values(1, values)

        assert replaced.shape == (2, 3, 2)
        assert torch.equal(fiber, 1 + 10 * values)
        assert torch.equal(
            replaced.fiber(1, torch.tensor([1, 0])), fiber)

    def test_functional_fiber_after_freeze_uses_an_independent_session(self):
        source = tk.decompositions.CallableTensorSource(
            lambda values: values.sum(dim=1).to(torch.float64),
            in_dim=(2, 2),
            dtype=torch.float64)
        phi = PhiOperator(
            source,
            ((0, torch.tensor([0., 1.])),
             (1, torch.tensor([0., 1.]))),
            _scalar_output_spec(2),
            input_kind='features')
        builder = _EvaluationPlanBuilder(source)
        phi.collect(builder)
        builder.freeze()

        with pytest.raises(RuntimeError, match='before expand'):
            phi.collect(builder)
        fiber = phi.fiber_at(
            0, torch.tensor([-1., 2.]), fixed_indices=torch.tensor([1]))

        assert torch.equal(fiber, torch.tensor([0., 3.]))
        assert source.evaluation_stats.source_calls == 1

    def test_phi_fit_delegates_without_materializing_operator(self):
        source = tk.decompositions.DenseTensorSource(
            torch.tensor([[1., 2.], [3., 4.], [4., 6.]]))
        phi = PhiOperator(
            source,
            ((0, torch.arange(3)), (1, torch.arange(2))),
            _scalar_output_spec(2))
        embedding = torch.tensor([
            [1., 0.], [0., 1.], [1., 1.]])

        fitted = phi.fit(
            axis=0,
            fitter=tk.decompositions.FixedEmbeddingFitter(
                embedding, fiber_batch_size=2),
            domain=torch.arange(3))

        assert torch.allclose(
            fitted.tensor, torch.tensor([[1., 2.], [3., 4.]]))

    def test_evaluate_select_and_materialized_views_are_consistent(self):
        source = tk.decompositions.DenseTensorSource(
            torch.arange(8.).reshape(2, 2, 2))
        phi = PhiOperator(
            source,
            tuple((site, torch.arange(2)) for site in range(3)),
            _scalar_output_spec(3))
        selection = torch.tensor([[0, 1, 1], [1, 0, 0]])

        selected = phi.evaluate(selection)
        view = phi.select(selection)
        materialized = _MaterializedPhi(phi.materialize(), phi.layout)

        assert isinstance(view, PhiView)
        assert torch.equal(selected, torch.tensor([3., 4.]))
        assert torch.equal(view.materialize(), selected)
        assert torch.equal(materialized.evaluate(selection), selected)
        assert torch.equal(
            materialized.fiber(1, torch.tensor([1, 0])),
            torch.tensor([4., 6.]))

    def test_input_fiber_uses_source_partial_contraction(self):
        cores = make_tt_cores(
            generator=torch.Generator().manual_seed(41))
        source = tk.decompositions.TTTensorSource(cores)
        phi = PhiOperator(
            source,
            tuple((site, torch.arange(dim))
                  for site, dim in enumerate(source.in_dim)),
            _scalar_output_spec(3))
        dense = contract_tt_dense(cores)

        fiber = phi.fiber(1, torch.tensor([0, 1]))

        assert torch.allclose(fiber, dense[0, :, 1])
        assert source.evaluation_stats == tk.decompositions.EvaluationStats(
            requested_points=3,
            unique_points=3,
            batches=1,
            source_calls=1)
        assert phi.last_stats == source.evaluation_stats

    def test_heterogeneous_input_coordinates_are_passed_as_a_tuple(self):
        def function(values):
            scalar, vector = values
            return scalar + vector[:, 0] - vector[:, 1]

        source = tk.decompositions.CallableTensorSource(
            function,
            in_dim=(2, 2),
            dtype=torch.float64)
        phi = PhiOperator(
            source,
            (
                (0, torch.tensor([1., 2.], dtype=torch.float64)),
                (1, torch.tensor(
                    [[3., 1.], [4., -1.]], dtype=torch.float64)),
            ),
            _scalar_output_spec(2),
            input_kind='features')

        result = phi.materialize()

        assert torch.equal(
            result, torch.tensor([[3., 6.], [4., 7.]]))
        assert not phi.configuration_batch().packed

    def test_legacy_singleton_scalar_output_needs_no_output_site(self):
        source = tk.decompositions.CallableTensorSource(
            lambda indices: indices.sum(dim=1, keepdim=True).to(torch.float64),
            in_dim=(2, 2),
            out_shape=(1,),
            dtype=torch.float64)
        output_spec = _OutputSpec.normalize(
            torch.ones(1, 1), n_input_sites=2)
        phi = PhiOperator(
            source,
            ((0, torch.arange(2)), (1, torch.arange(2))),
            output_spec)

        assert torch.equal(
            phi.materialize(), torch.tensor([[0., 1.], [1., 2.]]))

    def test_multiple_output_sites_are_removed_and_gathered_row_major(self):
        def function(indices):
            base = (indices[:, 0] + 10 * indices[:, 1]).to(torch.float64)
            return base[:, None, None] + torch.arange(6).reshape(1, 2, 3)

        source = tk.decompositions.CallableTensorSource(
            function,
            in_dim=(2, 2),
            out_shape=(2, 3),
            dtype=torch.float64)
        output_spec = _OutputSpec.normalize(
            torch.ones(1, 2, 3),
            n_input_sites=2,
            out_position=(1, 3))
        phi = PhiOperator(
            source,
            (
                (0, torch.arange(2)),
                (1, torch.arange(2)),
                (2, torch.arange(2)),
                (3, torch.arange(3)),
            ),
            output_spec)

        result = phi.materialize()

        expected = torch.empty(2, 2, 2, 3, dtype=torch.float64)
        for x, out_0, y, out_1 in product(
                range(2), range(2), range(2), range(3)):
            expected[x, out_0, y, out_1] = \
                x + 10 * y + 3 * out_0 + out_1
        assert torch.equal(result, expected)
        configurations = phi.configuration_batch()
        assert configurations.n_sites == 2
        assert configurations.batch_size == 24
        assert phi.last_stats == tk.decompositions.EvaluationStats(
            requested_points=24,
            unique_points=4,
            batches=1,
            cache_hits=20,
            source_calls=1)
        assert torch.equal(
            phi.fiber(0, torch.tensor([1, 1, 2])),
            torch.tensor([15., 16.], dtype=torch.float64))

    @pytest.mark.parametrize('source_kind', ['dense', 'sparse', 'tt'])
    def test_dense_sparse_and_tt_sources_share_phi_semantics(self, source_kind):
        cores = make_tt_cores(
            generator=torch.Generator().manual_seed(42))
        dense = contract_tt_dense(cores)
        if source_kind == 'dense':
            source = tk.decompositions.DenseTensorSource(dense)
        elif source_kind == 'sparse':
            indices = torch.tensor(list(product(
                *(range(dim) for dim in dense.shape))))
            source = tk.decompositions.SparseTensorSource(
                indices, dense.reshape(-1), in_dim=dense.shape)
        else:
            source = tk.decompositions.TTTensorSource(cores)
        phi = PhiOperator(
            source,
            tuple((site, torch.arange(dim))
                  for site, dim in enumerate(dense.shape)),
            _scalar_output_spec(3))

        assert torch.allclose(phi.materialize(), dense)

    def test_grid_coordinates_require_no_peps_specific_branch(self):
        sites = ((0, 0), (0, 1), (1, 0), (1, 1))
        source = tk.decompositions.CallableTensorSource(
            lambda values: values.sum(dim=1),
            in_dim=(2, 2, 2, 2),
            dtype=torch.float32)
        samples = torch.tensor([
            [0., 1., 0., 0.],
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
        ])
        pool = _SamplePool(samples, sites=sites)
        top = pool.restrict(SiteRegion(((0, 0), (0, 1))))
        phi = PhiOperator(
            source,
            (
                top,
                ((1, 0), torch.tensor([0., 1.])),
                ((1, 1), torch.tensor([0., 1.])),
            ),
            _scalar_output_spec(4),
            input_sites=sites,
            input_kind='features')

        result = phi.materialize()

        expected = torch.empty(top.n_unique, 2, 2)
        for top_id, lower_left, lower_right in product(
                range(top.n_unique), range(2), range(2)):
            expected[top_id, lower_left, lower_right] = (
                top.values[0][top_id] + top.values[1][top_id] +
                lower_left + lower_right)
        assert torch.equal(result, expected)

    def test_component_coverage_and_collisions_are_validated(self):
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 2))
        output_spec = _scalar_output_spec(2)

        with pytest.raises(ValueError, match='cover every input'):
            PhiOperator(source, ((0, torch.arange(2)),), output_spec)
        with pytest.raises(ValueError, match='only once'):
            PhiOperator(
                source,
                ((0, torch.arange(2)), (0, torch.arange(2))),
                output_spec)
