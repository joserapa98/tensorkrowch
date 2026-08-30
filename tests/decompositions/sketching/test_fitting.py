"""Tests for sampled-input fitting strategies."""

import math

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.decompositions.sketching.phi import (PhiOperator,
                                                       _MaterializedPhi)
from tensorkrowch.decompositions.sketching.fitting import FittedInputAxis
from tensorkrowch.decompositions.sketching.specs import (
    _DomainSpec,
    _EmbeddingSpec,
    _OutputSpec,
)


class _FiberOnlyPhi:
    """Phi test double that forbids complete materialization."""

    def __init__(self, tensor):
        self.base = _MaterializedPhi(tensor, range(tensor.ndim))
        self.shape = tuple(tensor.shape)
        self.fiber_calls = 0

    def evaluate(self, index_selection):
        return self.base.evaluate(index_selection)

    def fiber(self, axis, fixed_indices=None):
        self.fiber_calls += 1
        return self.base.fiber(axis, fixed_indices)

    def materialize(self, batch_size=None):
        raise AssertionError('The fitter should consume fibers')


def _materialized(tensor):
    return _MaterializedPhi(tensor, range(tensor.ndim))


class _AffineEmbedding(torch.nn.Module):
    """Small trainable map whose columns span affine scalar functions."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([
            [1.0, 0.1], [0.1, 1.0]], dtype=torch.float64))

    def forward(self, values):
        basis = torch.stack((torch.ones_like(values), values), dim=1)
        return basis @ self.weight


class TestFixedEmbeddingFitter:  # MARK: TestFixedEmbeddingFitter

    def test_exact_deembedding_preserves_phi_axis_order(self):
        embedding = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
        ], dtype=torch.float64)
        coefficients = torch.arange(
            12, dtype=torch.float64).reshape(2, 2, 3)
        phi = torch.einsum('xi,aib->axb', embedding, coefficients)
        fitter = tk.decompositions.FixedEmbeddingFitter(embedding)

        fitted = fitter.fit(
            _materialized(phi), axis=1, domain=torch.arange(4.))

        assert isinstance(fitter, tk.decompositions.InputFitter)
        assert isinstance(fitted, FittedInputAxis)
        assert fitted.axis == 1
        assert fitted.domain_size == 4
        assert fitted.input_dim == 2
        assert fitted.record is None
        assert torch.allclose(fitted.tensor, coefficients)

    def test_overdetermined_fit_records_residual_and_condition(self):
        embedding = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, -1.0],
            [-1.0, 2.0],
        ], dtype=torch.float64)
        coefficients = torch.tensor([
            [1.0, -2.0, 0.5],
            [3.0, 4.0, -1.0],
        ], dtype=torch.float64)
        phi = embedding @ coefficients

        fitted = tk.decompositions.FixedEmbeddingFitter(embedding).fit(
            _materialized(phi),
            axis=0,
            domain=torch.linspace(-1, 1, 5),
            return_info=True)

        assert torch.allclose(fitted.tensor, coefficients)
        assert fitted.record.method == 'fixed_embedding'
        assert fitted.record.residual_absolute < 1e-12
        assert fitted.record.residual_relative < 1e-12
        assert math.isfinite(fitted.record.condition_number)
        assert fitted.record.local_solve is not None
        info = tk.decompositions.DecompositionMetrics(
            input_fits=[fitted.record]).as_info()
        assert info['input_fits'][0]['method'] == 'fixed_embedding'

    def test_complex_embedding_and_vector_coordinate_domain(self):
        domain = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=torch.float64)

        def embedding(values):
            return torch.stack((
                torch.ones(values.shape[0], dtype=torch.complex128),
                values[:, 0] + 1j * values[:, 1],
            ), dim=1)

        coefficients = torch.tensor([
            [1.0 + 2.0j, -1.0j],
            [3.0 - 1.0j, 2.0 + 0.5j],
        ], dtype=torch.complex128)
        phi = embedding(domain) @ coefficients

        fitted = tk.decompositions.FixedEmbeddingFitter(embedding).fit(
            _materialized(phi), axis=0, domain=domain)

        assert fitted.tensor.dtype == torch.complex128
        assert torch.allclose(fitted.tensor, coefficients)

    def test_fiber_path_matches_materialization_without_calling_it(self):
        embedding = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        coefficients = torch.randn(2, 2, 2)
        phi = torch.einsum('xi,aib->axb', embedding, coefficients)
        fiber_phi = _FiberOnlyPhi(phi)
        fitter = tk.decompositions.FixedEmbeddingFitter(
            embedding, fiber_batch_size=3)

        fitted = fitter.fit(
            fiber_phi,
            axis=1,
            domain=torch.arange(3),
            return_info=True)

        assert torch.allclose(fitted.tensor, coefficients, atol=1e-6)
        assert fitted.record.used_fibers
        assert fiber_phi.fiber_calls == math.ceil((2 * 2) / 3)

    def test_regularization_is_delegated_to_shared_solver(self):
        embedding = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ], dtype=torch.float64)
        target = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
        l2_reg = 0.2
        solver = tk.decompositions.LeastSquaresSolver(
            l2_reg=l2_reg,
            column_scaling=False,
            system_scaling=False)
        fitter = tk.decompositions.FixedEmbeddingFitter(
            embedding, solver=solver)
        augmented = torch.cat((
            embedding,
            math.sqrt(l2_reg) * torch.eye(2, dtype=torch.float64),
        ), dim=0)
        augmented_target = torch.cat((target, torch.zeros(
            2, dtype=torch.float64)))
        expected = torch.linalg.lstsq(
            augmented, augmented_target).solution

        fitted = fitter.fit(
            _materialized(target), axis=0, domain=torch.arange(3))

        assert torch.allclose(fitted.tensor, expected)

    def test_no_diagnostics_skips_condition_number_svd(self, monkeypatch):
        embedding = torch.eye(2)
        fitter = tk.decompositions.FixedEmbeddingFitter(embedding)

        def fail_condition(*args, **kwargs):
            raise AssertionError('Condition number should not be computed')

        monkeypatch.setattr(torch.linalg, 'svdvals', fail_condition)

        fitted = fitter.fit(
            _materialized(torch.tensor([1.0, 2.0])),
            axis=0,
            domain=torch.arange(2),
            return_info=False)

        assert torch.equal(fitted.tensor, torch.tensor([1.0, 2.0]))
        assert fitted.record is None

    def test_fixed_fitter_declares_no_additional_queries(self):
        phi = _materialized(torch.ones(3, 2))
        fitter = tk.decompositions.FixedEmbeddingFitter(torch.randn(3, 2))

        assert fitter.required_queries(
            phi, axis=0, domain=torch.arange(3)) == ()

    def test_cached_heterogeneous_site_embeddings_feed_independent_fitters(
            self):
        domains = _DomainSpec((
            torch.tensor([-1., 0., 1.]),
            torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        ))
        embeddings = _EmbeddingSpec.normalize((
            lambda x: torch.stack((torch.ones_like(x), x), dim=1),
            lambda x: torch.stack((
                torch.ones(x.shape[0]), x[:, 0], x[:, 1]), dim=1),
        ), domains)
        coefficients = (torch.randn(2, 2), torch.randn(3, 2))

        for site in range(2):
            phi = embeddings.matrix(site) @ coefficients[site]
            fitted = tk.decompositions.FixedEmbeddingFitter(
                embeddings.matrix(site)).fit(
                    _materialized(phi),
                    axis=0,
                    domain=domains.for_site(site))

            assert fitted.input_dim == coefficients[site].shape[0]
            assert torch.allclose(fitted.tensor, coefficients[site], atol=1e-6)


class TestBasisFitter:  # MARK: TestBasisFitter

    def test_multiple_basis_axes_can_be_fitted_independently(self):
        tensor = torch.arange(12.).reshape(3, 2, 2)
        first_labels = torch.tensor([2, 0, 1])
        second_labels = torch.tensor([1, 0])
        first = tk.decompositions.BasisFitter(input_dim=3).fit(
            _materialized(tensor),
            axis=0,
            domain=first_labels,
            return_info=True)
        second = tk.decompositions.BasisFitter(input_dim=2).fit(
            _materialized(first.tensor),
            axis=2,
            domain=second_labels,
            return_info=True)
        expected_first = torch.zeros_like(tensor)
        expected_first.index_copy_(0, first_labels, tensor)
        expected = torch.zeros_like(expected_first)
        expected.index_copy_(2, second_labels, expected_first)

        assert torch.equal(second.tensor, expected)
        assert first.record.method == 'basis'
        assert second.record.condition_number == 1.

    def test_missing_basis_labels_are_filled_with_zero(self):
        values = torch.tensor([3.0, 5.0])

        fitted = tk.decompositions.BasisFitter(input_dim=4).fit(
            _materialized(values),
            axis=0,
            domain=torch.tensor([3, 1]))

        assert torch.equal(fitted.tensor, torch.tensor([0., 5., 0., 3.]))

    def test_basis_fitter_supports_fibers(self):
        tensor = torch.arange(12.).reshape(2, 3, 2)
        phi = _FiberOnlyPhi(tensor)

        fitted = tk.decompositions.BasisFitter(
            input_dim=3, fiber_batch_size=2).fit(
                phi,
                axis=1,
                domain=torch.tensor([1, 2, 0]),
                return_info=True)
        expected = torch.zeros_like(tensor)
        expected.index_copy_(1, torch.tensor([1, 2, 0]), tensor)

        assert torch.equal(fitted.tensor, expected)
        assert fitted.record.used_fibers

    @pytest.mark.parametrize(
        'domain, match',
        [
            (torch.tensor([0., 1.]), 'integer vector'),
            (torch.tensor([0, 0]), 'unique'),
            (torch.tensor([-1, 0]), 'inside'),
        ])
    def test_invalid_basis_domains_are_rejected(self, domain, match):
        fitter = tk.decompositions.BasisFitter(input_dim=2)

        with pytest.raises(ValueError, match=match):
            fitter.fit(_materialized(torch.ones(2)), 0, domain)


class TestTrainableEmbeddingFitter:  # MARK: TestTrainableEmbeddingFitter

    def test_trains_from_functional_fibers_and_returns_optional_state(self):
        domain = torch.linspace(-1, 1, 7, dtype=torch.float64)
        coefficients = torch.tensor([
            [1., -2., 0.5], [3., 4., -1.]], dtype=torch.float64)
        target = torch.stack((torch.ones_like(domain), domain), dim=1) @ \
            coefficients
        phi = _FiberOnlyPhi(target)
        fitter = tk.decompositions.TrainableEmbeddingFitter(
            _AffineEmbedding(),
            input_dim=2,
            max_steps=20,
            tolerance=1e-10,
            patience=10,
            fiber_batch_size=2,
            seed=17)
        random_state = torch.random.get_rng_state().clone()

        fitted = fitter.fit(
            phi,
            axis=0,
            domain=domain,
            return_info=True)

        matrix = fitter.model(domain)
        assert torch.allclose(
            matrix @ fitted.tensor, target, rtol=1e-9, atol=1e-10)
        assert phi.fiber_calls == 2
        assert fitted.record.method == 'trainable_embedding'
        assert fitted.record.used_fibers
        assert fitted.model is fitter.model
        assert fitted.model_state is not None
        assert fitted.metadata['steps'] <= 20
        assert torch.equal(torch.random.get_rng_state(), random_state)

    def test_fast_result_does_not_attach_model_or_state(self):
        domain = torch.linspace(-1, 1, 5, dtype=torch.float64)
        target = torch.stack((torch.ones_like(domain), domain), dim=1)
        fitter = tk.decompositions.TrainableEmbeddingFitter(
            _AffineEmbedding(), max_steps=1, fiber_batch_size=2)

        fitted = fitter.fit(
            _FiberOnlyPhi(target),
            axis=0,
            domain=domain,
            return_info=False)

        assert fitted.record is None
        assert fitted.model is None
        assert fitted.model_state is None
        assert torch.allclose(fitter.model(domain) @ fitted.tensor, target)

    def test_query_declaration_validates_domain_before_freeze(self):
        phi = _materialized(torch.ones(3, 2))
        fitter = tk.decompositions.TrainableEmbeddingFitter(
            _AffineEmbedding(), max_steps=1)

        assert fitter.required_queries(
            phi, axis=0, domain=torch.arange(3.)) == ()
        with pytest.raises(ValueError, match='match'):
            fitter.required_queries(
                phi, axis=0, domain=torch.arange(2.))


class TestQTTInputFitter:  # MARK: TestQTTInputFitter

    def test_tt_rss_driver_keeps_functional_phi_for_qtt_fitting(self):
        domain = torch.linspace(0, 1, 4, dtype=torch.float64)
        samples = torch.cartesian_prod(domain, domain)

        def function(values):
            x, y = values.unbind(dim=1)
            return (1 + x + 2 * y + x * y).unsqueeze(1)

        fitters = tuple(
            tk.decompositions.QTTInputFitter(
                base=2,
                level=2,
                domain=torch.tensor([0., 1.], dtype=torch.float64),
                rank=4,
                batch_size=16,
                seed=31 + site)
            for site in range(2))
        result = tk.decompositions.TTRSS(
            function,
            embedding=torch.eye(4, dtype=torch.float64),
            domain=domain,
            input_fitters=fitters,
            output_device=None).fit(
                samples,
                rank=4,
                legacy_projection=False)
        expected = function(samples).reshape(4, 4)

        assert torch.allclose(result.contract_dense(), expected)
        assert result.input_dim == (4, 4)

    def test_tensor_environment_is_kept_at_factor_endpoint(self):
        physical_domain = torch.tensor([0., 1.], dtype=torch.float64)

        def function(values):
            base = 1 + values[:, 0]
            return base[:, None, None] + torch.arange(
                6, dtype=values.dtype).reshape(1, 2, 3)

        source = tk.decompositions.CallableTensorSource(
            function,
            input_dim=(4,),
            output_shape=(2, 3),
            dtype=torch.float64)
        output_spec = _OutputSpec.normalize(
            torch.ones(1, 2, 3),
            n_input_sites=1,
            out_position=(1, 2))
        initial_domain = torch.linspace(0, 1, 4, dtype=torch.float64)
        phi = PhiOperator(
            source,
            ((0, initial_domain),
             (1, torch.arange(2)),
             (2, torch.arange(3))),
            output_spec,
            input_kind='coordinates')
        fitter = tk.decompositions.QTTInputFitter(
            base=2,
            level=2,
            digit_order='fine_to_coarse',
            domain=physical_domain,
            rank=4,
            connector_rank=2,
            batch_size=8,
            seed=23)

        fitted = fitter.fit(
            phi,
            axis=0,
            domain=initial_domain,
            return_info=True)
        expected = function(initial_domain.reshape(-1, 1))
        oracle = tk.decompositions.TTSVD(
            expected, output_device=None).fit(rank=4)

        assert fitted.tensor.shape == (4, 2, 3)
        assert torch.allclose(fitted.tensor, expected)
        assert torch.allclose(fitted.tensor, oracle.contract_dense())
        assert fitted.factor is not None
        assert fitted.factor.input_dim == (2, 2, 2)
        assert fitted.reduced_tensor.shape == (2, 2, 3)
        factor_values = fitter.factor_values(fitted, initial_domain)
        assert torch.allclose(
            torch.einsum(
                'ig,gab->iab', factor_values, fitted.reduced_tensor),
            fitted.tensor)
        assert fitted.truncation.selected_rank == 2
        assert fitted.metadata['out_position'] == (2, 3)
        assert fitted.metadata['connector_rank'] == 2
        assert fitted.record.method == 'qtt'
        assert fitted.record.residual_relative < 1e-10

    def test_reduced_mode_avoids_materializing_the_grid_axis(self):
        physical_domain = torch.tensor([0., 1.], dtype=torch.float64)

        def function(values):
            base = 1 + values[:, 0]
            return base[:, None, None] + torch.arange(
                6, dtype=values.dtype).reshape(1, 2, 3)

        source = tk.decompositions.CallableTensorSource(
            function,
            input_dim=(4,),
            output_shape=(2, 3),
            dtype=torch.float64)
        output_spec = _OutputSpec.normalize(
            torch.ones(1, 2, 3),
            n_input_sites=1,
            out_position=(1, 2))
        initial_domain = torch.linspace(0, 1, 4, dtype=torch.float64)
        phi = PhiOperator(
            source,
            ((0, initial_domain),
             (1, torch.arange(2)),
             (2, torch.arange(3))),
            output_spec,
            input_kind='coordinates')
        expected = function(initial_domain.reshape(-1, 1))
        fitter = tk.decompositions.QTTInputFitter(
            base=2,
            level=2,
            domain=physical_domain,
            rank=4,
            connector_rank=2,
            batch_size=8,
            seed=24,
            materialize_tensor=False)

        fitted = fitter.fit(phi, axis=0, domain=initial_domain)

        assert fitted.tensor is fitted.reduced_tensor
        assert fitted.tensor.shape == (2, 2, 3)
        assert fitted.metadata['materialized_tensor'] is False
        factor_values = fitter.factor_values(fitted, initial_domain)
        reconstructed = torch.einsum(
            'ig,gab->iab', factor_values, fitted.reduced_tensor)
        assert torch.allclose(reconstructed, expected)

    def test_requires_functional_phi_and_declares_independent_session(self):
        fitter = tk.decompositions.QTTInputFitter(
            base=2, level=2, domain=torch.tensor([0., 1.]))
        materialized = _materialized(torch.ones(4, 2))

        with pytest.raises(TypeError, match='functional'):
            fitter.required_queries(
                materialized, axis=0, domain=torch.arange(4.))
