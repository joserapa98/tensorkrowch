"""Tests for the internal specifications shared by sketching methods."""

from itertools import product

import pytest

import torch

from tensorkrowch.decompositions.sketching.specs import (
    _DomainSpec,
    _EmbeddingSpec,
    _OutputSpec,
    _SketchingFitSpec,
)
from tensorkrowch.decompositions.sources import ConfigurationBatch


class TestDomainSpec:  # MARK: TestDomainSpec

    def test_shared_domain_is_broadcast_to_every_input_site(self):
        domain = torch.tensor([0., 0.5, 1.])

        spec = _DomainSpec.normalize(domain, n_sites=3)

        assert spec.n_sites == 3
        assert spec.n_values == (3, 3, 3)
        assert spec.coordinate_shape == ((), (), ())
        assert all(value is domain for value in spec.values)
        assert not spec.inferred

    def test_heterogeneous_domains_preserve_size_and_coordinate_shape(self):
        domains = (
            torch.tensor([-1., 0., 1.]),
            torch.tensor([[0., 1.], [2., 3.]]),
        )

        spec = _DomainSpec.normalize(domains, n_sites=2)

        assert spec.n_values == (3, 2)
        assert spec.coordinate_shape == ((), (2,))
        assert torch.equal(spec.for_site(1), domains[1])

    def test_domain_is_inferred_independently_from_heterogeneous_samples(self):
        samples = ConfigurationBatch(
            (
                torch.tensor([1., 0., 1., 2.]),
                torch.tensor([[1., 2.], [1., 2.], [3., 4.], [1., 2.]]),
            ),
            kind='coordinates')

        spec = _DomainSpec.normalize(None, n_sites=2, samples=samples)

        assert spec.inferred
        assert torch.equal(spec.values[0], torch.tensor([0., 1., 2.]))
        assert torch.equal(
            spec.values[1], torch.tensor([[1., 2.], [3., 4.]]))

    @pytest.mark.parametrize(
        'domain, match',
        [
            ([torch.ones(2)], 'one tensor per input site'),
            ([torch.ones(2), torch.tensor([0., float('nan')])], 'site 1'),
            ([torch.ones(2), torch.empty(0)], 'site 1'),
        ])
    def test_invalid_domain_reports_the_affected_site(self, domain, match):
        with pytest.raises(ValueError, match=match):
            _DomainSpec.normalize(domain, n_sites=2)

    def test_inference_requires_samples_with_matching_sites(self):
        with pytest.raises(ValueError, match='one value per input site'):
            _DomainSpec.normalize(
                None, n_sites=2, samples=torch.ones(3, 1))


class TestEmbeddingSpec:  # MARK: TestEmbeddingSpec

    def test_site_embeddings_are_cached_once_with_heterogeneous_dimensions(
            self):
        calls = [0, 0]

        def scalar_embedding(values):
            calls[0] += 1
            return torch.stack((values, 1 - values), dim=1)

        def vector_embedding(values):
            calls[1] += 1
            return torch.cat((values, values.sum(dim=1, keepdim=True)), dim=1)

        domains = _DomainSpec.normalize(
            (
                torch.tensor([0., 0.5, 1.]),
                torch.tensor([[1., 2.], [3., 4.]]),
            ),
            n_sites=2)

        spec = _EmbeddingSpec.normalize(
            (scalar_embedding, vector_embedding), domains)

        assert calls == [1, 1]
        assert spec.input_dim == (2, 3)
        assert spec.matrix(0).shape == (3, 2)
        assert spec.matrix(1).shape == (2, 3)
        assert calls == [1, 1]
        assert torch.equal(
            spec.evaluate(0, torch.tensor([1., 0.])),
            torch.tensor([[1., 0.], [0., 1.]]))
        assert calls == [2, 1]

    def test_shared_embedding_is_evaluated_once_per_site_domain(self):
        calls = []

        def embedding(values):
            calls.append(values.shape[0])
            return torch.stack((values, values.square()), dim=1)

        domains = _DomainSpec.normalize(
            (torch.tensor([0., 1.]), torch.tensor([0., 1., 2.])),
            n_sites=2)

        spec = _EmbeddingSpec.normalize(embedding, domains)

        assert calls == [2, 3]
        assert spec.input_dim == (2, 2)

    def test_tensor_embedding_looks_up_values_in_its_site_domain(self):
        domains = _DomainSpec.normalize(
            torch.tensor([10, 20, 30]), n_sites=1)
        table = torch.tensor([
            [1. + 0.j, 0. + 1.j],
            [2. + 0.j, 0. + 2.j],
            [3. + 0.j, 0. + 3.j],
        ], dtype=torch.complex128)
        spec = _EmbeddingSpec.normalize(table, domains)

        result = spec.evaluate(0, torch.tensor([30, 10]))

        assert spec.input_dim == (2,)
        assert result.dtype == torch.complex128
        assert torch.equal(result, table[[2, 0]])
        with pytest.raises(ValueError, match='site 0.*outside'):
            spec.evaluate(0, torch.tensor([20, 40]))

    @pytest.mark.parametrize(
        'embedding, match',
        [
            (lambda values: values, 'site 1.*shape'),
            (lambda values: torch.ones(values.shape[0], 0),
             'site 1.*positive input_dim'),
            (lambda values: torch.full(
                (values.shape[0], 2), float('inf')), 'site 1.*finite'),
        ])
    def test_invalid_embedding_reports_the_affected_site(
            self, embedding, match):
        domains = _DomainSpec.normalize(
            (torch.arange(2.), torch.arange(3.)), n_sites=2)

        with pytest.raises(ValueError, match=match):
            _EmbeddingSpec.normalize(
                (lambda values: torch.stack((values, values), dim=1),
                 embedding),
                domains)

    def test_embedding_list_should_match_number_of_input_sites(self):
        domains = _DomainSpec.normalize(torch.arange(2.), n_sites=2)

        with pytest.raises(ValueError, match='one entry per input site'):
            _EmbeddingSpec.normalize((lambda x: x[:, None],), domains)


class TestOutputSpec:  # MARK: TestOutputSpec

    @pytest.mark.parametrize('shape', [(4,), (4, 1)])
    def test_scalar_outputs_have_no_output_site(self, shape):
        values = torch.ones(shape)

        spec = _OutputSpec.normalize(values, n_input_sites=3)

        assert spec.scalar
        assert spec.output_shape == ()
        assert spec.positions == ()
        assert spec.n_sites == 3
        assert spec.layout == (
            ('input', 0), ('input', 1), ('input', 2))

    def test_scalar_output_rejects_an_output_position(self):
        with pytest.raises(ValueError, match='scalar'):
            _OutputSpec.normalize(
                torch.ones(4), n_input_sites=3, out_position=1)

    @pytest.mark.parametrize(
        'output_shape, n_inputs, positions, expected_layout',
        [
            (
                (3,), 4, (2,),
                (
                    ('input', 0), ('input', 1), ('output', 0),
                    ('input', 2), ('input', 3),
                ),
            ),
            (
                (2, 3), 5, (2, 4),
                (
                    ('input', 0), ('input', 1), ('output', 0),
                    ('input', 2), ('output', 1), ('input', 3),
                    ('input', 4),
                ),
            ),
            (
                (2, 3), 1, (0, 2),
                (('output', 0), ('input', 0), ('output', 1)),
            ),
        ])
    def test_default_positions_split_inputs_into_balanced_groups(
            self, output_shape, n_inputs, positions, expected_layout):
        values = torch.ones(4, *output_shape)

        spec = _OutputSpec.normalize(values, n_input_sites=n_inputs)

        assert spec.positions == positions
        assert spec.layout == expected_layout

    def test_explicit_separated_outputs_keep_axis_and_input_order(self):
        spec = _OutputSpec.normalize(
            torch.ones(4, 2, 3),
            n_input_sites=5,
            out_position=(0, 6))

        assert spec.layout == (
            ('output', 0),
            ('input', 0),
            ('input', 1),
            ('input', 2),
            ('input', 3),
            ('input', 4),
            ('output', 1),
        )
        assert spec.input_positions == (1, 2, 3, 4, 5)

    @pytest.mark.parametrize(
        'out_position, error, match',
        [
            (1, ValueError, 'requires one output axis'),
            ((1,), ValueError, 'one site per output axis'),
            ((2, 2), ValueError, 'strictly increasing'),
            ((3, 1), ValueError, 'strictly increasing'),
            ((0, 7), ValueError, 'between 0 and 6'),
            ((0, True), TypeError, 'integers'),
        ])
    def test_invalid_multiple_output_positions(
            self, out_position, error, match):
        with pytest.raises(error, match=match):
            _OutputSpec.normalize(
                torch.ones(4, 2, 3),
                n_input_sites=5,
                out_position=out_position)

    def test_flatten_and_unflatten_labels_are_row_major_and_reversible(self):
        spec = _OutputSpec((2, 3, 4), n_input_sites=2, positions=(0, 2, 4))
        indices = torch.tensor(list(product(range(2), range(3), range(4))))

        labels = spec.flatten_labels(indices)

        assert torch.equal(labels, torch.arange(24))
        assert torch.equal(spec.unflatten_labels(labels), indices)

    def test_complex_labels_use_squared_magnitude_and_select_values(self):
        values = torch.zeros(2, 2, 3, dtype=torch.complex128)
        values[0, 1, 2] = 3j
        values[1, 0, 1] = 2 + 2j
        spec = _OutputSpec.normalize(values, n_input_sites=3)

        labels, indices, selected = spec.resolve_labels(
            values, generator=torch.Generator().manual_seed(3))

        assert torch.equal(labels, torch.tensor([5, 1]))
        assert torch.equal(indices, torch.tensor([[1, 2], [0, 1]]))
        assert torch.equal(selected, torch.tensor(
            [3j, 2 + 2j], dtype=torch.complex128))

    def test_explicit_flat_labels_select_row_major_tensor_entries(self):
        values = torch.arange(24, dtype=torch.float64).reshape(4, 2, 3)
        spec = _OutputSpec.normalize(values, n_input_sites=2)

        labels, indices, selected = spec.resolve_labels(
            values, labels=torch.tensor([0, 5, 3, 1]))

        assert torch.equal(labels, torch.tensor([0, 5, 3, 1]))
        assert torch.equal(
            indices, torch.tensor([[0, 0], [1, 2], [1, 0], [0, 1]]))
        assert torch.equal(selected, torch.tensor([0., 11., 15., 19.]))

    def test_zero_norm_rows_have_an_explicit_sampling_policy(self):
        values = torch.zeros(2, 2, 3)
        spec = _OutputSpec.normalize(values, n_input_sites=2)

        with pytest.raises(ValueError, match='zero-norm rows'):
            spec.sample_labels(values)
        labels = spec.sample_labels(
            values,
            generator=torch.Generator().manual_seed(0),
            zero_policy='uniform')
        assert labels.shape == (2,)
        assert torch.all((labels >= 0) & (labels < 6))

    def test_output_indices_are_inserted_without_packing_heterogeneous_inputs(
            self):
        samples = (
            torch.tensor([0.1, 0.2]),
            torch.tensor([[1., 2.], [3., 4.]]),
        )
        indices = torch.tensor([[0, 2], [1, 0]])
        spec = _OutputSpec((2, 3), n_input_sites=2, positions=(1, 3))

        result = spec.insert_indices(samples, indices)

        assert len(result) == 4
        assert torch.equal(result[0], samples[0])
        assert torch.equal(result[1], indices[:, 0])
        assert torch.equal(result[2], samples[1])
        assert torch.equal(result[3], indices[:, 1])

    def test_all_output_sites_use_basis_and_input_sites_use_their_embedding(
            self):
        domains = _DomainSpec.normalize(
            (torch.tensor([0., 1.]), torch.tensor([0., 1., 2.])),
            n_sites=2)
        embeddings = _EmbeddingSpec.normalize(
            (
                lambda x: torch.stack((1 - x, x), dim=1),
                lambda x: torch.stack((x, x.square(), 1 + x), dim=1),
            ),
            domains)
        spec = _OutputSpec((2, 4), n_input_sites=2, positions=(1, 3))

        assert spec.site_dim(embeddings) == (2, 2, 3, 4)
        assert torch.equal(
            spec.embed_site(
                1, torch.tensor([1, 0]), embeddings, dtype=torch.complex64),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64))
        assert torch.equal(
            spec.embed_site(3, torch.tensor([3, 1]), embeddings),
            torch.tensor([[0, 0, 0, 1], [0, 1, 0, 0]]))
        assert torch.equal(
            spec.embed_site(0, torch.tensor([1., 0.]), embeddings),
            torch.tensor([[0., 1.], [1., 0.]]))


class TestSketchingFitSpec:  # MARK: TestSketchingFitSpec

    def test_fit_options_are_normalized_once(self):
        spec = _SketchingFitSpec(
            rank=5,
            cutoff=1e-8,
            atol=1e-7,
            rtol=1e-6,
            cum_percentage=0.99,
            batch_size=32,
            verbose=2,
            collect_metrics=True)

        assert spec.truncation.as_kwargs() == {
            'rank': 5,
            'cutoff': 1e-8,
            'atol': 1e-7,
            'rtol': 1e-6,
            'cum_percentage': 0.99,
        }
        assert spec.effective_projection_dim == 5
        assert spec.verbosity == 2
        assert spec.diagnostics_enabled

    def test_projection_dimension_defaults_to_rank_or_square_projection(self):
        assert _SketchingFitSpec(
            rank=None).effective_projection_dim is None
        assert _SketchingFitSpec(
            rank=None, projection_dim=7).effective_projection_dim == 7
        assert _SketchingFitSpec(
            rank=4, random_projection=False).effective_projection_dim is None

    @pytest.mark.parametrize(
        'kwargs, error, match',
        [
            ({'rank': True}, TypeError, 'rank'),
            ({'projection_dim': 0}, ValueError, 'positive'),
            ({'random_projection': False, 'projection_dim': 3},
             ValueError, 'requires'),
            ({'batch_size': 0}, ValueError, 'positive'),
            ({'verbose': 4}, ValueError, 'between 0 and 3'),
            ({'collect_metrics': 1}, TypeError, 'bool'),
        ])
    def test_invalid_fit_options(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            _SketchingFitSpec(**kwargs)

    def test_diagnostics_can_be_disabled_completely(self):
        spec = _SketchingFitSpec(verbose=0, collect_metrics=False)

        assert spec.verbosity == 0
        assert not spec.diagnostics_enabled
