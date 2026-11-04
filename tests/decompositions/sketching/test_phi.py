"""Tests for lazy, cached and shared local Phi operators."""

from itertools import product

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.sketching.evaluations import (
    ConfigurationMap)
from tensorkrowch.decompositions.sketching.phi import PhiOperator, PhiView
from tensorkrowch.decompositions.sketching.regions import (SiteRegion,
                                                           _SamplePool)
from tensorkrowch.decompositions.sketching.specs import _OutputSpec

from tests.decompositions.als._oracles import contract_tt_dense, make_tt_cores


def _scalar_output_spec(n_inputs):
    """Creates the scalar output layout used by most Phi tests."""
    return _OutputSpec.normalize(torch.ones(1), n_input_sites=n_inputs)


class TestPhiOperator:  # MARK: TestPhiOperator

    def test_five_site_shared_and_cached_queries_on_available_devices(
            self, device_dtype, assert_close):
        device, dtype = device_dtype
        tensor = torch.randn(2, 3, 2, 2, 3, device=device, dtype=dtype)
        source = tk.decompositions.DenseTensorSource(tensor)
        phi = PhiOperator(
            source, tuple((site, torch.arange(dim, device=device))
                          for site, dim in enumerate(tensor.shape)),
            _scalar_output_spec(5))
        builder = ConfigurationMap(source)
        handle = phi.collect(builder)
        configurations = builder
        configurations._batch_size = 7
        configurations._transform = lambda table: table.values * 2
        phi._bind(configurations, handle)
        selection = torch.tensor([[0, 2, 1, 0, 2], [1, 0, 0, 1, 1]],
                                 device=device)

        expected = 2 * tensor[tuple(selection.unbind(-1))]
        assert_close(phi.evaluate(selection), expected)
        assert phi.tensor is None
        assert_close(phi.materialize(), 2 * tensor)
        assert_close(phi.evaluate(selection), expected)
        assert phi.tensor.device == tensor.device
        assert phi.tensor.dtype == tensor.dtype
        assert configurations.stats.unique_points == tensor.numel()
        assert source.evaluation_stats.source_calls == 11

    @pytest.mark.parametrize('selection', [
        torch.tensor([[0, 2]]), torch.tensor([[-1]]),
        torch.tensor([[2]]), torch.tensor([[0.]]), torch.tensor(0)])
    def test_invalid_selected_indices_are_rejected(self, selection):
        source = tk.decompositions.DenseTensorSource(torch.ones(2))
        phi = PhiOperator(source, ((0, torch.arange(2)),),
                          _scalar_output_spec(1))
        with pytest.raises((TypeError, ValueError), match='index_selection'):
            phi.evaluate(selection)

    @pytest.mark.parametrize('axis', [True, '0', 1, -2])
    def test_invalid_fiber_axes_are_rejected(self, axis):
        source = tk.decompositions.DenseTensorSource(torch.ones(2))
        phi = PhiOperator(source, ((0, torch.arange(2)),),
                          _scalar_output_spec(1))
        with pytest.raises((TypeError, ValueError), match='axis'):
            phi.fiber(axis)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_cached_queries_release_and_rematerialize_preserve_gradients(
            self, dtype):
        tensor = torch.arange(12., dtype=torch.float64).reshape(2, 3, 2)
        tensor = tensor.to(dtype).requires_grad_()
        source = tk.decompositions.DenseTensorSource(tensor)
        phi = PhiOperator(
            source, tuple((site, torch.arange(dim))
                          for site, dim in enumerate(tensor.shape)),
            _scalar_output_spec(3))
        selection = torch.tensor([[0, 1, 1], [1, 2, 0]])

        assert phi.tensor is None
        expected = phi.evaluate(selection)
        cached = phi.materialize(batch_size=4)
        before = source.evaluation_stats
        assert cached is phi.tensor
        assert phi.materialize() is cached
        assert torch.equal(phi.evaluate(selection), expected)
        assert torch.equal(phi.fiber(1, torch.tensor([1, 0])), tensor[1, :, 0])
        assert source.evaluation_stats == before
        phi.evaluate(selection).real.sum().backward()
        assert tensor.grad[0, 1, 1] == 1
        assert tensor.grad[1, 2, 0] == 1
        assert tensor.grad.abs().sum() == 2

        phi.tensor = None
        assert torch.equal(phi.materialize(), cached)
        assert source.evaluation_stats.source_calls == before.source_calls + 1

    def test_bound_map_preserves_transformed_selected_queries(self):
        source = tk.decompositions.DenseTensorSource(
            torch.arange(12.).reshape(2, 3, 2))
        phi = PhiOperator(
            source, tuple((site, torch.arange(dim))
                          for site, dim in enumerate(source.in_dim)),
            _scalar_output_spec(3))
        builder = ConfigurationMap(source)
        handle = phi.collect(builder)
        configurations = builder
        configurations._transform = lambda table: 2 * table.values + 1
        phi._bind(configurations, handle)
        selection = torch.tensor([[0, 1, 1], [1, 2, 0]])

        assert torch.equal(phi.evaluate(selection), torch.tensor([7., 21.]))
        assert phi.tensor is None
        assert torch.equal(phi.fiber(1, torch.tensor([1, 0])),
                           torch.tensor([13., 17., 21.]))
        cached = phi.materialize()
        phi.tensor = None
        assert torch.equal(phi.materialize(), cached)
        assert source.evaluation_stats.source_calls == 1

    def test_selected_views_map_queries_without_materializing_the_parent(self):
        source = tk.decompositions.DenseTensorSource(
            torch.arange(12.).reshape(2, 3, 2))
        phi = PhiOperator(
            source, tuple((site, torch.arange(dim))
                          for site, dim in enumerate(source.in_dim)),
            _scalar_output_spec(3))
        view = phi.select(torch.tensor([
            [[0, 0, 0], [0, 1, 1]], [[1, 0, 0], [1, 2, 1]]]))
        assert torch.equal(view.evaluate(torch.tensor([[0, 1], [1, 1]])),
                           torch.tensor([3., 11.]))
        assert torch.equal(view.fiber(1, torch.tensor([1])),
                           torch.tensor([6., 11.]))
        assert phi.tensor is None
        assert source.evaluation_stats.requested_points == 4

    def test_one_axis_and_scalar_selected_views(self):
        source = tk.decompositions.DenseTensorSource(torch.tensor([2., 3.]))
        phi = PhiOperator(source, ((0, torch.arange(2)),),
                          _scalar_output_spec(1))
        assert torch.equal(phi.fiber(0), torch.tensor([2., 3.]))
        assert torch.equal(phi.fiber(0, torch.empty(2, 0, dtype=torch.long)),
                           torch.tensor([[2., 3.], [2., 3.]]))
        view = phi.select(torch.tensor([1]))
        assert view.materialize().shape == ()
        assert torch.equal(view.evaluate(torch.empty(2, 0, dtype=torch.long)),
                           torch.tensor([3., 3.]))

    @pytest.mark.parametrize('batch_size', [True, 0, -1, 1.5])
    def test_materialize_validates_batch_size_even_with_cached_data(
            self, batch_size):
        source = tk.decompositions.DenseTensorSource(torch.ones(2))
        phi = PhiOperator(source, ((0, torch.arange(2)),),
                          _scalar_output_spec(1))
        phi.materialize()
        with pytest.raises((TypeError, ValueError), match='batch_size'):
            phi.materialize(batch_size=batch_size)

    def test_coordinate_phi_matches_explicit_cartesian_evaluation(self):
        source = tk.decompositions.CallableSource(lambda values: (
                values[:, 0] + 10 * values[:, 1] + 100 * values[:, 2]
            ).to(torch.float64), n_sites=3, dtype=torch.float64)
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
        source = tk.decompositions.CallableSource(lambda values: (
                values[:, 0] + 10 * values[:, 1] + 100 * values[:, 2]
            ).to(torch.float64), n_sites=3, dtype=torch.float64)
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
        source = tk.decompositions.CallableSource(
            lambda values: values.sum(dim=1).to(torch.float64),
            n_sites=2, dtype=torch.float64)
        phi = PhiOperator(
            source,
            ((0, torch.tensor([0., 1.])),
             (1, torch.tensor([0., 1.]))),
            _scalar_output_spec(2),
            input_kind='features')
        builder = ConfigurationMap(source)
        phi.collect(builder)
        builder.freeze()

        with pytest.raises(RuntimeError, match='already frozen'):
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
            fitted.coefficients, torch.tensor([[1., 2.], [3., 4.]]))

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
        materialized = phi.materialize()

        assert isinstance(view, PhiView)
        assert torch.equal(selected, torch.tensor([3., 4.]))
        assert torch.equal(view.materialize(), selected)
        assert torch.equal(phi.evaluate(selection), selected)
        assert phi.materialize() is materialized
        assert torch.equal(
            phi.fiber(1, torch.tensor([1, 0])),
            torch.tensor([4., 6.]))

    def test_input_fiber_uses_source_partial_contraction(self):
        cores = make_tt_cores(
            generator=torch.Generator().manual_seed(41))
        source = tk.decompositions.FormatSource(cores)
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

        source = tk.decompositions.CallableSource(
            function, n_sites=2, dtype=torch.float64)
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
        source = tk.decompositions.CallableSource(
            lambda indices: indices.sum(dim=1, keepdim=True).to(torch.float64),
            n_sites=3, in_sites=(0, 1), out_shape=(1,), dtype=torch.float64)
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

        source = tk.decompositions.CallableSource(
            function, n_sites=4, in_sites=(0, 2), out_shape=(2, 3),
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
        assert configurations.n_inputs == 2
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
    def test_dense_sparse_and_tt_sources_share_phi_semantics(
            self, source_kind):
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
            source = tk.decompositions.FormatSource(cores)
        phi = PhiOperator(
            source,
            tuple((site, torch.arange(dim))
                  for site, dim in enumerate(dense.shape)),
            _scalar_output_spec(3))

        assert torch.allclose(phi.materialize(), dense)

    def test_custom_site_order_preserves_correlated_samples(self):
        sites = (3, 1, 5, 7)
        source = tk.decompositions.CallableSource(
            lambda values: values.sum(dim=1), n_sites=4, dtype=torch.float32)
        samples = torch.tensor([
            [0., 1., 0., 0.],
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
        ])
        pool = _SamplePool(samples, sites=sites)
        region = pool.restrict(SiteRegion((3, 1)))
        phi = PhiOperator(
            source,
            (
                region,
                (5, torch.tensor([0., 1.])),
                (7, torch.tensor([0., 1.])),
            ),
            _scalar_output_spec(4),
            input_sites=sites,
            input_kind='features')

        result = phi.materialize()

        expected = torch.empty(region.n_unique, 2, 2)
        for region_id, first_value, second_value in product(
                range(region.n_unique), range(2), range(2)):
            expected[region_id, first_value, second_value] = (
                region.values[0][region_id] + region.values[1][region_id] +
                first_value + second_value)
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


def _prepare_phi(tensor, direction, **options):
    return PhiOperator.from_tensor(tensor).truncate(direction, **options)


class TestPhiPreparation:

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('direction', ['left', 'right'])
    def test_auxiliary_rank_discovery_respects_caps_and_selected_axis(
            self, dtype, direction):
        tensor = torch.ones(4, 2, 4, dtype=dtype)
        left = direction == 'left'
        prepared = _prepare_phi(
            tensor, direction, rank_left=3, rank_right=3,
            numerical=(left, not left))
        prescribed = _prepare_phi(
            tensor, direction, rank_left=3, rank_right=3,
            numerical=(not left, left))
        axis = 0 if left else -1

        assert prepared.coefficients.shape[axis] == 1
        assert prescribed.coefficients.shape[axis] == 3

        tensor = torch.randn(4, 2, 4, dtype=dtype)
        capped = _prepare_phi(
            tensor, direction, rank_left=2, rank_right=2,
            numerical=(True, True))
        assert capped.coefficients.shape[axis] == 2

    @pytest.mark.parametrize('numerical, shape', [
        ((True, False), (1, 2, 3)),
        ((False, True), (2, 2, 1)),
        ((True, True), (1, 2, 1))])
    def test_central_auxiliary_rank_discovery_is_axis_specific(
            self, numerical, shape):
        tensor = torch.ones(4, 2, 4, dtype=torch.float64)
        prepared = _prepare_phi(
            tensor, 'center', rank_left=2, rank_right=3,
            numerical=numerical)

        assert prepared.coefficients.shape == shape
        approximation = torch.einsum(
            'ab,bpc,cd->apd', prepared.left_lift, prepared.coefficients,
            prepared.right_lift)
        torch.testing.assert_close(approximation, tensor)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('rtol', [None, 1e-22])
    def test_central_support_selection_preserves_small_singular_values(
            self, dtype, rtol):
        factors = [torch.randn(size, 2, dtype=dtype) for size in (6, 2, 6)]
        tensor = torch.einsum('a,p,b->apb', *(x[:, 0] for x in factors))
        tensor = tensor + 1e-8 * torch.einsum(
            'a,p,b->apb', *(x[:, 1] for x in factors))
        prepared = _prepare_phi(tensor, 'center', rtol=rtol, site=2)

        assert prepared.coefficients.shape == (2, 2, 2)
        approximation = torch.einsum(
            'ab,bpc,cd->apd', prepared.left_lift, prepared.coefficients,
            prepared.right_lift)
        torch.testing.assert_close(approximation, tensor,
                                   rtol=1e-12, atol=1e-13)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_central_factors_restore_full_target(self, dtype):
        tensor = torch.randn(3, 4, 5, dtype=dtype)
        prepared = _prepare_phi(tensor, 'center', site=2,
                                collect_metrics=True)
        reconstructed = torch.einsum(
            'ab,bpc,cd->apd', prepared.left_lift, prepared.coefficients,
            prepared.right_lift)

        torch.testing.assert_close(reconstructed, tensor)
        torch.testing.assert_close(prepared.coefficients.norm(), tensor.norm())
        torch.testing.assert_close(
            prepared.left_lift.mH @ prepared.left_lift,
            torch.eye(prepared.left_lift.shape[-1], dtype=dtype))
        torch.testing.assert_close(
            prepared.right_lift @ prepared.right_lift.mH,
            torch.eye(prepared.right_lift.shape[0], dtype=dtype))
        assert len(prepared.truncations) == 2
        assert [record.site for record in prepared.truncations] == [2, 1]

    @pytest.mark.parametrize('direction', ['left', 'right'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_outward_tensors_retain_singular_values(self, direction, dtype):
        tensor = torch.randn(3, 4, 5, dtype=dtype)
        prepared = _prepare_phi(tensor, direction, rank_left=2, rank_right=2)
        basis = prepared.coefficients
        if direction == 'left':
            basis = basis.reshape(basis.shape[0], -1).mH
            matrix = tensor.reshape(tensor.shape[0], -1).mH
        else:
            basis = basis.reshape(-1, basis.shape[-1])
            matrix = tensor.reshape(-1, tensor.shape[-1])

        spectra = torch.linalg.svdvals(matrix)[:2]
        torch.testing.assert_close(
            basis.mH @ basis, torch.diag(spectra.square()).to(dtype))
        assert prepared.left_lift is None
        assert prepared.right_lift is None

    @pytest.mark.parametrize('direction', ['left', 'right', 'center'])
    def test_truncated_tensors_preserve_retained_scales(self, direction):
        tensor = torch.diag(torch.tensor([4., 2., 0.1])).unsqueeze(1)
        prepared = _prepare_phi(
            tensor, direction, rank_left=2, rank_right=2)
        assert prepared.coefficients.norm().item() == pytest.approx(20 ** 0.5)
        if direction == 'center':
            approximation = torch.einsum(
                'ab,bpc,cd->apd', prepared.left_lift, prepared.coefficients,
                prepared.right_lift)
            assert (tensor - approximation).norm().item() == \
                pytest.approx(0.1)

    def test_randomized_projection_is_optional_and_owns_its_parameters(self):
        tensor = torch.randn(3, 4, 5, dtype=torch.float64)
        projector = tk.decompositions.RandomizedRangeProjector(
            projection_dim=5, projection_oversampling=1, n_power_iter=1)
        prepared = _prepare_phi(
            tensor, 'center', projector=projector,
            generator=torch.Generator().manual_seed(101),
            collect_metrics=True, site=2)
        reconstructed = torch.einsum(
            'ab,bpc,cd->apd', prepared.left_lift, prepared.coefficients,
            prepared.right_lift)

        torch.testing.assert_close(reconstructed, tensor)
        assert len(prepared.projections) == 2

    def test_raw_boundary_keeps_original_tensor(self):
        tensor = torch.randn(1, 3, 2)
        prepared = _prepare_phi(tensor, None)
        assert prepared.coefficients is tensor
        assert prepared.truncations == ()


class TestPreparedPhiOwnership:

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_raw_queries_survive_fitting_and_truncation(self, dtype):
        generator = torch.Generator().manual_seed(708)
        tensor = torch.randn(3, 4, 5, dtype=dtype, generator=generator)
        phi = PhiOperator.from_tensor(tensor)
        fitted = phi.fit(1, tk.decompositions.BasisFitter(), torch.arange(4))
        assert fitted is phi
        assert phi.tensor is tensor
        assert phi.coefficients is not None
        prepared = phi.truncate('center', rank_left=2, rank_right=2,
                                collect_metrics=True, site=2)
        assert prepared is phi
        assert phi.coefficients.shape == (2, 4, 2)
        assert phi.shape == tuple(tensor.shape)
        torch.testing.assert_close(phi.materialize(), tensor)
        selected = torch.tensor([[0, 1, 2], [2, 3, 4]])
        torch.testing.assert_close(phi.evaluate(selected), tensor[
            tuple(selected.unbind(1))])
        assert len(phi.truncations) == 2
        assert phi.left_lift.shape == (3, 2)
        assert phi.right_lift.shape == (2, 5)
