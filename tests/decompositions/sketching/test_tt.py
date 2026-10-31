"""Tests for centered TT-RSS, TT-RS and their public wrappers."""

from dataclasses import replace
import gc
import math
import weakref

import pytest
import torch

import tensorkrowch as tk


def _grid(dimensions, device='cpu'):
    """Returns every indexed configuration, including the one-input case."""
    return torch.cartesian_prod(*[
        torch.arange(dim, device=device) for dim in dimensions
    ]).reshape(-1, len(dimensions))


def _rank_two_tensor(dimensions, dtype=torch.float64, device='cpu'):
    """Builds a nonconstant rank-two tensor with different local dimensions."""
    first = torch.ones(dimensions, dtype=dtype, device=device)
    second = torch.ones_like(first)
    for site, dimension in enumerate(dimensions):
        values = torch.arange(dimension, device=device).to(dtype)
        shape = [1] * len(dimensions)
        shape[site] = dimension
        first = first * (1 + values).reshape(shape)
        factor = .6 - .4 * values
        if dtype.is_complex:
            factor = factor + .2j * values
        second = second * factor.reshape(shape)
    return first + (.3 + .2j if dtype.is_complex else .3) * second


def _continuous_problem(n_sites=3, dtype=torch.float64):
    """Returns a polynomial represented by two basis features per variable."""
    domain = torch.tensor([-1., 0., 1.], dtype=torch.float64)
    samples = torch.cartesian_prod(*[domain] * n_sites).reshape(-1, n_sites)

    def function(values):
        first = (1 + .2 * values).prod(-1)
        second = (1 - .5 * values).prod(-1)
        coefficient = .3 + .2j if dtype.is_complex else .3
        return (first + coefficient * second).to(dtype)

    def embedding(values):
        return torch.stack((torch.ones_like(values), values), dim=-1)

    return function, embedding, samples, domain


class TestTTRSS:  # MARK: TestTTRSS

    def test_error_metric_preserves_complex_reference_for_real_fit(self):
        class RealFitter:

            def fit(self, phi_view, axis, domain, return_info=False):
                fitted = tk.decompositions.BasisFitter().fit(
                    phi_view, axis, domain, return_info=return_info)
                return replace(fitted, tensor=fitted.tensor.real)

        target = torch.full((2, 2), 1 + 1j, dtype=torch.complex128)
        result = tk.decompositions.TTRSS(
            target, input_fitter=RealFitter()).fit(
                _grid(target.shape), rank=1, collect_metrics=True)

        expected = (result.contract_dense() - target).norm() / target.norm()
        torch.testing.assert_close(result.metrics.errors[0].relative, expected)
        assert expected > 0.5

    @pytest.mark.parametrize('rank', [2, (2, 3)])
    @pytest.mark.parametrize('source_kind', ['dense', 'format'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_matrix_rank_caps_apply_between_pairs(self, rank, source_kind,
                                                  dtype):
        generator = torch.Generator().manual_seed(96)
        ranks = (2, 2) if isinstance(rank, int) else rank
        cores = [
            torch.randn(2, ranks[0], 3, dtype=dtype, generator=generator),
            torch.randn(ranks[0], 3, ranks[1], 2, dtype=dtype,
                        generator=generator),
            torch.randn(ranks[1], 2, 3, dtype=dtype, generator=generator)]
        format = tk.formats.TTM(cores)
        source = tk.decompositions.FormatSource(format) \
            if source_kind == 'format' else \
            tk.decompositions.DenseTensorSource.matrix(
                format.contract_dense(), ordering='interleaved')
        inputs = _grid(format.in_dim)
        n_outputs = math.prod(format.out_dim)
        result = tk.decompositions.tt_rss(
            source, inputs.repeat_interleave(n_outputs, 0), rank=rank,
            labels=torch.arange(n_outputs).repeat(inputs.shape[0]),
            collect_metrics=True)

        assert isinstance(result, tk.decompositions.TTMDecomposition)
        assert result.rank == list(ranks)
        assert result.in_dim == format.in_dim
        assert result.out_dim == format.out_dim
        torch.testing.assert_close(result.contract_dense(),
                                   format.contract_dense(),
                                   rtol=1e-9, atol=1e-10)
        assert result.metrics.errors[0].relative < 1e-9

    def test_one_matrix_site_does_not_truncate_its_local_matrix(self):
        tensor = torch.tensor([[1., 2., 3.], [4., 8., 6.]],
                              dtype=torch.float64)
        source = tk.decompositions.DenseTensorSource.matrix(tensor)
        samples = torch.arange(2).repeat_interleave(3).reshape(-1, 1)
        labels = torch.arange(3).repeat(2)
        result = tk.decompositions.tt_rss(source, samples, rank=1,
                                         labels=labels)
        assert result.rank == []
        assert result.n_sites == 1
        torch.testing.assert_close(result.contract_dense(), tensor)

    def test_noisy_matrix_keeps_auxiliary_svd_capacity_bounds(self):
        generator = torch.Generator().manual_seed(97)
        cores = [torch.randn(2, 2, 2, dtype=torch.float64,
                              generator=generator)]
        cores.extend(torch.randn(2, 2, 2, 2, dtype=torch.float64,
                                  generator=generator) for _ in range(2))
        cores.append(torch.randn(2, 2, 2, dtype=torch.float64,
                                  generator=generator))
        tensor = tk.formats.TTM(cores).contract_dense()
        noise = torch.randn(tensor.shape, dtype=tensor.dtype,
                             generator=generator)
        noise = noise * (1e-7 * tensor.norm() / noise.norm())
        target = tensor + noise
        source = tk.decompositions.DenseTensorSource.matrix(
            target, ordering='interleaved')
        inputs = _grid((2, 2, 2, 2))
        result = tk.decompositions.tt_rss(
            source, inputs.repeat_interleave(16, 0), rank=2,
            labels=torch.arange(16).repeat(inputs.shape[0]),
            collect_metrics=True)

        assert result.rank == [2, 2, 2]
        selected = [record.selected_rank
                    for record in result.metrics.truncations]
        assert selected == [2, 2, 4, 2, 4, 2, 2]
        error = (result.contract_dense() - target).norm() / target.norm()
        assert 0 < error < 1e-4

    @pytest.mark.parametrize('center', [0, 2, 4, None])
    @pytest.mark.parametrize('rank', [2, (2, 2, 2, 2), None])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_centered_five_site_finite_tensor_recovery(self, center, rank,
                                                     dtype):
        tensor = _rank_two_tensor((2, 3, 2, 3, 2), dtype)
        result = tk.decompositions.tt_rss(
            tensor, _grid(tensor.shape), rank=rank, center=center,
            collect_metrics=True)

        assert isinstance(result, tk.decompositions.TTDecomposition)
        assert result.in_dim == tensor.shape
        assert result.rank == [2, 2, 2, 2]
        torch.testing.assert_close(result.contract_dense(), tensor,
                                   rtol=1e-9, atol=1e-10)
        assert result.metrics.errors[0].relative < 1e-9
        assert len(result.metrics.truncations) == 4
        assert len(result.metrics.local_solves) == 4
        assert len(result.metrics.evaluations) == 1
        stats = result.metrics.evaluations[0]
        assert stats.requested_points > stats.unique_points
        assert stats.unique_points == tensor.numel()
        assert stats.cache_hits == stats.requested_points - stats.unique_points

    @pytest.mark.parametrize('center', [0, 2, 4])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_centered_continuous_embedding_recovery(self, center, dtype):
        function, embedding, samples, domain = _continuous_problem(5, dtype)
        result = tk.decompositions.TTRSS(
            function, embedding, domain=domain).fit(
                samples, rank=(2, 2, 2, 2), center=center,
                collect_metrics=True)

        torch.testing.assert_close(result.evaluate(embedding(samples)),
                                   function(samples), rtol=1e-9, atol=1e-10)
        assert result.dtype == dtype
        assert result.in_dim == (2,) * 5
        assert len(result.metrics.input_fits) == 5
        assert result.metrics.errors[0].relative < 1e-9

    @pytest.mark.parametrize('dimensions', [(1,), (3,), (2, 3)])
    @pytest.mark.parametrize('center', [0, None])
    def test_minimal_finite_chains(self, dimensions, center):
        tensor = torch.arange(math.prod(dimensions), dtype=torch.float64)
        tensor = (tensor + 1).reshape(dimensions)
        result = tk.decompositions.tt_rss(
            tensor, _grid(dimensions), center=center)
        assert len(result.cores) == len(dimensions)
        assert result.in_dim == dimensions
        torch.testing.assert_close(result.contract_dense(), tensor)

    def test_repeated_fits_reuse_problem_without_reusing_results(self):
        function, embedding, samples, domain = _continuous_problem()
        engine = tk.decompositions.TTRSS(function, embedding, domain=domain)
        results = [engine.fit(samples, rank=2,
                              generator=torch.Generator().manual_seed(91))
                   for _ in range(2)]
        assert results[0] is not results[1]
        for first, second in zip(results[0].cores, results[1].cores):
            assert first is not second
            torch.testing.assert_close(first, second)

    def test_explicit_source_and_custom_range_projector(self):
        function, embedding, samples, domain = _continuous_problem()
        source = tk.decompositions.CallableSource(
            function, n_sites=3, out_shape=(), dtype=torch.float64)
        projector = tk.decompositions.RandomizedRangeProjector(
            projection_dim=3, projection_oversampling=1, n_power_iter=1)
        result = tk.decompositions.TTRSS(
            source, embedding, domain=domain, range_projector=projector,
            out_device=None).fit(
                samples, rank=2, collect_metrics=True,
                generator=torch.Generator().manual_seed(93))

        torch.testing.assert_close(result.evaluate(embedding(samples)),
                                   function(samples))
        assert len(result.metrics.range_projections) == 2
        assert all(record.method == 'randomized'
                   for record in result.metrics.range_projections)
        assert result.metrics.timings[-1].name == 'total'
        assert result.device == samples.device

    @pytest.mark.parametrize('individual', [False, True])
    def test_public_input_fitter_and_solver_objects(self, individual):
        fitter = tk.decompositions.BasisFitter(in_dim=2)
        fitters = (fitter,) * 3 if individual else fitter
        tensor = _rank_two_tensor((2, 2, 2))
        result = tk.decompositions.TTRSS(
            tensor, input_fitter=fitters,
            local_solver=tk.decompositions.LeastSquaresSolver()).fit(
                _grid(tensor.shape), rank=2)
        torch.testing.assert_close(result.contract_dense(), tensor)

    @pytest.mark.parametrize('domain_kind', ['shared', 'per_site', 'inferred'])
    def test_domain_and_embedding_shorthands(self, domain_kind):
        function, embedding, samples, shared = _continuous_problem()
        domain = shared if domain_kind == 'shared' else (
            (shared,) * 3 if domain_kind == 'per_site' else None)
        embeddings = embedding if domain_kind == 'shared' else (embedding,) * 3
        result = tk.decompositions.tt_rss(
            function, samples, embeddings, rank=2, domain=domain,
            generator=torch.Generator().manual_seed(94))
        torch.testing.assert_close(result.evaluate(embedding(samples)),
                                   function(samples))

    @pytest.mark.parametrize('center', [0, 1])
    def test_heterogeneous_coordinates_embeddings_and_dimensions(self, center):
        scalar_domain = torch.tensor([0., 1.], dtype=torch.float64)
        vector_domain = torch.tensor([[0., 0.], [1., 0.], [0., 1.]],
                                     dtype=torch.float64)
        samples = (scalar_domain.repeat_interleave(3),
                   vector_domain.repeat(2, 1))

        def function(values):
            scalar, vector = values
            return 1 + scalar + vector[:, 0] + 2 * vector[:, 1]

        embeddings = (
            lambda x: torch.stack((torch.ones_like(x), x), -1),
            lambda x: torch.cat((torch.ones_like(x[:, :1]), x), -1))
        result = tk.decompositions.TTRSS(
            function, embeddings, in_dim=(2, 3),
            domain=(scalar_domain, vector_domain)).fit(
                samples, rank=2, center=center, collect_metrics=True)
        embedded = tuple(embed(values) for embed, values in zip(
            embeddings, samples))
        assert result.in_dim == (2, 3)
        torch.testing.assert_close(result.evaluate(embedded),
                                   function(samples))
        assert result.metrics.errors[0].relative < 1e-10

    def test_vector_valued_coordinate_axes(self):
        domain = torch.tensor([[0., 0.], [1., 0.], [0., 1.]],
                              dtype=torch.float64)
        indices = _grid((3, 3, 3))
        samples = torch.stack([domain[indices[:, site]]
                               for site in range(3)], 1)

        def embedding(values):
            return torch.cat((torch.ones_like(values[..., :1]), values), -1)

        def function(values):
            return 1 + values.sum((1, 2))

        result = tk.decompositions.tt_rss(
            function, samples, embedding, domain=domain, rank=2)
        torch.testing.assert_close(result.evaluate(embedding(samples)),
                                   function(samples))

    @pytest.mark.parametrize('out_position', [(0, 4), (1, 3)])
    def test_tensor_output_axis_placement_and_flattened_labels(self,
                                                            out_position):
        domain = torch.tensor([0., 1.], dtype=torch.float64)
        inputs = torch.cartesian_prod(domain, domain, domain)
        samples = inputs.repeat_interleave(6, 0)
        labels = torch.arange(6).repeat(inputs.shape[0])

        def function(values):
            base = 1 + values.sum(-1)
            factors = torch.arange(6, dtype=values.dtype).reshape(1, 2, 3)
            return base[:, None, None] + factors

        def embedding(values):
            return torch.stack((1 - values, values), -1)

        result = tk.decompositions.tt_rss(
            function, samples, embedding, labels=labels, domain=domain,
            out_position=out_position, rank=4, collect_metrics=True)
        expected_shape = [2] * 5
        expected_shape[out_position[1]] = 3
        assert result.in_dim == tuple(expected_shape)
        assert result.metadata['out_shape'] == (2, 3)
        assert result.metadata['out_position'] == out_position
        assert result.metrics.errors[0].relative < 1e-9
        torch.testing.assert_close(result.evaluate(
            embedding(inputs), in_sites=result.in_sites), function(inputs))

    @pytest.mark.parametrize('out_position', [0, 2, 3])
    def test_result_cores_construct_an_mps_layer(self, out_position):
        domain = torch.tensor([0., 1.], dtype=torch.float64)
        inputs = torch.cartesian_prod(domain, domain, domain)
        samples = inputs.repeat_interleave(2, 0)

        def function(values):
            base = 1 + values.sum(-1)
            return torch.stack((base, 2 * base), -1)

        def embedding(x):
            return torch.stack((1 - x, x), -1)
        result = tk.decompositions.tt_rss(
            function, samples, embedding, domain=domain, rank=2,
            labels=torch.arange(2).repeat(inputs.shape[0]),
            out_position=out_position)
        model = tk.models.MPSLayer(tensors=result.cores,
                                   out_position=out_position,
                                   parameterized=False)
        assert model.out_position == out_position
        torch.testing.assert_close(model(embedding(inputs)), function(inputs))

    @pytest.mark.parametrize('truncation', [
        {'cutoff': 0.}, {'atol': 0.}, {'rtol': 0.}, {'cum_percentage': 1.}])
    @pytest.mark.parametrize('method', ['svd', 'qr_svd'])
    def test_svd_truncation_controls(self, truncation, method):
        function, embedding, samples, domain = _continuous_problem()
        with tk.svd_method(method):
            result = tk.decompositions.tt_rss(
                function, samples, embedding, domain=domain, rank=2,
                **truncation)
        assert result.rank == [2, 2]
        torch.testing.assert_close(result.evaluate(embedding(samples)),
                                   function(samples))

    def test_generator_preserves_global_random_state(self):
        domain = torch.tensor([0., 1.], dtype=torch.float64)
        samples = torch.cartesian_prod(domain, domain, domain).repeat(4, 1)
        def embedding(x):
            return torch.stack((1 - x, x), -1)

        def function(values):
            base = 1 + values.sum(-1)
            return torch.stack((base, base + 1), -1)

        state = torch.random.get_rng_state().clone()
        results = [tk.decompositions.tt_rss(
            function, samples, embedding, domain=domain, rank=3,
            generator=torch.Generator().manual_seed(95)) for _ in range(2)]
        assert torch.equal(torch.random.get_rng_state(), state)
        for first, second in zip(results[0].cores, results[1].cores):
            assert torch.equal(first, second)

    @pytest.mark.parametrize('verbose', [0, 1, 2, 3])
    def test_verbosity_reports_each_resolved_core_once(self, verbose, capsys):
        result = tk.decompositions.tt_rss(
            torch.ones(2, 2, 2), _grid((2, 2, 2)), rank=1, verbose=verbose)
        output = capsys.readouterr().out
        assert result.n_sites == 3
        if verbose == 0:
            assert output == ''
        elif verbose < 3:
            assert output and 'tensor(' not in output
        else:
            assert output.count('tensor(') == 3

    @pytest.mark.parametrize('rank', [True, 0, -1, (2,), (2, 0)])
    def test_invalid_rank_is_rejected(self, rank):
        with pytest.raises((TypeError, ValueError), match='rank'):
            tk.decompositions.tt_rss(torch.ones(2, 2, 2),
                                     _grid((2, 2, 2)), rank=rank)

    @pytest.mark.parametrize('center', [True, -1, 3, 'middle'])
    def test_invalid_center_is_rejected(self, center):
        with pytest.raises((TypeError, ValueError), match='center'):
            tk.decompositions.tt_rss(torch.ones(2, 2, 2),
                                     _grid((2, 2, 2)), center=center)

    def test_embedding_dimension_mismatch_is_rejected(self):
        function, embedding, samples, domain = _continuous_problem()
        with pytest.raises(ValueError, match='in_dim'):
            tk.decompositions.TTRSS(
                function, embedding, in_dim=3, domain=domain).fit(samples)
        with pytest.raises(TypeError, match='embedding'):
            tk.decompositions.TTRSS(function)

    @pytest.mark.parametrize('generator', [1, True, object()])
    def test_invalid_generator_is_rejected_before_evaluation(self, generator):
        calls = []

        def function(values):
            calls.append(values)
            return values.sum(-1)

        engine = tk.decompositions.TTRSS(
            function, torch.eye(2), domain=torch.arange(2))
        with pytest.raises(TypeError, match='generator'):
            engine.fit(torch.zeros(2, 2, dtype=torch.long),
                       generator=generator)
        assert calls == []

    @pytest.mark.parametrize('labels', [torch.ones(4, 1, dtype=torch.long),
                                      torch.full((4,), 2),
                                      torch.zeros(4)])
    def test_invalid_output_labels_are_rejected(self, labels):
        source = tk.decompositions.DenseTensorSource(
            torch.ones(2, 2, 2), in_sites=(0, 1))
        with pytest.raises((TypeError, ValueError), match='labels'):
            tk.decompositions.tt_rss(source, _grid((2, 2)), labels=labels)

    @pytest.mark.parametrize('options', [
        {'input_fitter': object()}, {'range_projector': object()},
        {'transform': 1}, {'domain_multiplier': True},
        {'domain_multiplier': 0}, {'local_solver': object()}])
    def test_invalid_strategy_options_are_rejected(self, options):
        with pytest.raises((TypeError, ValueError)):
            tk.decompositions.TTRSS(torch.ones(2, 2), **options).fit(
                _grid((2, 2)))

    @pytest.mark.parametrize('error', [AssertionError, IndexError])
    def test_callback_errors_propagate(self, error):
        def function(values):
            raise error('Callback failure')
        engine = tk.decompositions.TTRSS(
            function, torch.eye(2), domain=torch.arange(2))
        with pytest.raises(error, match='Callback failure'):
            engine.fit(torch.zeros(2, 2, dtype=torch.long), rank=1)


class TestRSSTransform:  # MARK: TestRSSTransform

    @pytest.mark.parametrize('tensor_output', [False, True])
    def test_metrics_use_transformed_initial_table(self, tensor_output):
        tensor = torch.arange(8., dtype=torch.float64).reshape(2, 2, 2) + 1
        source = tk.decompositions.DenseTensorSource(
            tensor, in_sites=(0, 1)) if tensor_output else tensor
        result = tk.decompositions.tt_rss(
            source, _grid((2, 2) if tensor_output else tensor.shape),
            rank=2, transform=lambda table: 2 * table.values,
            out_position=2 if tensor_output else None,
            collect_metrics=True)
        torch.testing.assert_close(result.contract_dense(), 2 * tensor)
        error = result.metrics.errors[0]
        assert error.kind == 'configuration_map'
        assert error.relative < 1e-10

    def test_global_callback_can_evaluate_auxiliary_raw_points(self):
        domain = torch.tensor([0., 1.], dtype=torch.float64)
        samples = torch.cartesian_prod(domain, domain, domain)
        extra = torch.full((1, 3), .25, dtype=torch.float64)
        callbacks = []
        seen_maps = []

        def function(values):
            return (1 + values).prod(-1)

        def transform(configuration_map):
            callbacks.append(configuration_map.configurations.shape)
            seen_maps.append(configuration_map)
            extra_values = configuration_map.evaluate(extra)
            assert torch.equal(extra_values, function(extra))
            assert torch.equal(configuration_map.evaluate(extra), extra_values)
            return configuration_map.values + extra_values

        def embedding(x):
            return torch.stack((1 - x, x), -1)
        result = tk.decompositions.tt_rss(
            function, samples, embedding, domain=domain,
            transform=transform, rank=2, collect_metrics=True)
        torch.testing.assert_close(result.evaluate(embedding(samples)),
                                   function(samples) + function(extra))
        assert callbacks == [torch.Size((8, 3))]
        assert seen_maps[0].configurations.shape == (8, 3)
        assert seen_maps[0].values.shape == (8,)
        assert seen_maps[0].stats.unique_points == 9

    @pytest.mark.parametrize('center', [0, 1])
    def test_global_transform_can_promote_real_values_to_complex(self, center):
        tensor = torch.tensor([[1., 3.], [2., 6.]], dtype=torch.float64)

        def transform(configuration_map):
            inputs = configuration_map.configurations
            return .5j * (configuration_map.values +
                          configuration_map.evaluate(inputs.flip(1)))

        result = tk.decompositions.tt_rss(
            tensor, _grid(tensor.shape), rank=2, center=center,
            transform=transform)
        assert result.dtype == torch.complex128
        torch.testing.assert_close(result.contract_dense(),
                                   .5j * (tensor + tensor.T))


class TestTTRS:  # MARK: TestTTRS

    @pytest.mark.parametrize('operator', [
        tk.decompositions.SampledSketch(), tk.decompositions.MarginalSketch(),
        tk.decompositions.TTStackSketch(tt_rank=2, stack_size=2)])
    @pytest.mark.parametrize('center', [0, 2, 4])
    def test_centered_nontrivial_source_recovery(self, operator, center):
        tensor = _rank_two_tensor((2, 3, 2, 3, 2))
        source = tk.decompositions.SparseTensorSource(
            _grid(tensor.shape), tensor.reshape(-1), tensor.shape)
        result = tk.decompositions.TTRS(
            source, sketch_operator=operator, out_device=None).fit(
                rank=(2,) * 4, center=center,
                generator=torch.Generator().manual_seed(301),
                collect_metrics=True)
        torch.testing.assert_close(result.contract_dense(), tensor,
                                   rtol=1e-9, atol=1e-10)
        assert result.metrics.errors[0].kind == 'source_support'
        assert result.metrics.errors[0].relative < 1e-9

    def test_dataset_functional_wrapper_returns_result(self):
        dataset = torch.tensor([[0, 0], [0, 0], [0, 1], [1, 0],
                                [1, 1], [1, 1]])
        source = tk.decompositions.SparseTensorSource.from_data(
            dataset, in_dim=(2, 2))
        result = tk.decompositions.tt_rs(source, rank=2, collect_metrics=True)
        expected = torch.tensor([[2., 1.], [1., 2.]]) / 6
        assert isinstance(result, tk.decompositions.TTDecomposition)
        torch.testing.assert_close(result.contract_dense(), expected)
        assert result.metadata['source_type'] == 'SparseTensorSource'
        assert result.metrics.errors[0].kind == 'source_support'

    @pytest.mark.parametrize('center', [0, 2])
    def test_marginal_kernel_recovers_smoothed_empirical_density(self, center):
        indices = _grid((3,) * 4)
        ascending = torch.arange(1, 4)
        descending = ascending.flip(0)
        counts = (ascending[indices].prod(1) +
                  descending[indices].prod(1))
        data = indices.repeat_interleave(counts, dim=0)
        source = tk.decompositions.SparseTensorSource.from_data(
            data, in_dim=(3,) * 4, dtype=torch.float64)
        coordinates = torch.linspace(-1, 1, 3, dtype=torch.float64)[:, None]
        kernel = tk.decompositions.gaussian_kernel(
            coordinates, centers=coordinates, bandwidth=0.45)
        # A finite-grid kernel preserves probability mass when columns sum
        # to one. Its product action then commutes with marginalization.
        kernel = kernel / kernel.sum(0, keepdim=True)
        shapes = []

        def smooth(phi):
            shapes.append(tuple(phi.shape))
            for axis, dimension in enumerate(phi.shape):
                if dimension == 3:
                    phi = torch.tensordot(
                        kernel, phi, dims=([1], [axis])).movedim(0, axis)
            return phi

        reference = source.contract_dense()
        for axis in range(reference.ndim):
            reference = torch.tensordot(
                kernel, reference, dims=([1], [axis])).movedim(0, axis)
        result = tk.decompositions.TTRS(
            source, sketch_operator=tk.decompositions.MarginalSketch(),
            transform=smooth).fit(rank=2, center=center)
        density = result.contract_dense()

        torch.testing.assert_close(density, reference, rtol=1e-9, atol=1e-12)
        assert density.min() >= -1e-12
        assert density.sum().item() == pytest.approx(1., abs=1e-10)
        assert shapes == [(1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 1)]

    @pytest.mark.parametrize('operator', [
        tk.decompositions.SampledSketch(), tk.decompositions.MarginalSketch(),
        tk.decompositions.TTStackSketch(tt_rank=2, stack_size=2)])
    def test_probability_tensor_recovery_with_each_sketch(self, operator):
        density = _rank_two_tensor((3,) * 5)
        density = density / density.sum()
        result = tk.decompositions.TTRS(
            density, sketch_operator=operator).fit(
                rank=2, center=2,
                generator=torch.Generator().manual_seed(391))
        approximation = result.contract_dense()

        torch.testing.assert_close(approximation, density,
                                   rtol=1e-9, atol=1e-12)
        assert approximation.min() >= -1e-12
        assert approximation.sum().item() == pytest.approx(1., abs=1e-10)

    @pytest.mark.parametrize('center', [0, 2, 4])
    def test_format_source_uses_structured_contractions(self, center):
        tensor = _rank_two_tensor((2, 3, 2, 3, 2))
        train = tk.decompositions.tt_svd(tensor, rank=2)
        source = tk.decompositions.FormatSource(train)
        result = tk.decompositions.TTRS(source).fit(rank=2, center=center,
                                                  batch_size=3)
        torch.testing.assert_close(result.contract_dense(), tensor,
                                   rtol=1e-9, atol=1e-10)
        assert result.metadata['system']['source_path'] == 'structured_tt'
        assert source.evaluation_stats.requested_points == 0

    def test_repeated_random_fits_are_independent_and_reproducible(self):
        tensor = _rank_two_tensor((2, 3, 2, 3, 2))
        engine = tk.decompositions.TTRS(
            tensor, sketch_operator=tk.decompositions.TTStackSketch(
                tt_rank=2, stack_size=2))
        first = engine.fit(rank=2,
                           generator=torch.Generator().manual_seed(302))
        second = engine.fit(rank=2,
                            generator=torch.Generator().manual_seed(302))
        assert first is not second
        for left, right in zip(first.cores, second.cores):
            assert left is not right
            torch.testing.assert_close(left, right)

    @pytest.mark.parametrize('dimensions', [(1,), (3,), (2, 3)])
    def test_minimal_sources(self, dimensions):
        tensor = torch.arange(math.prod(dimensions), dtype=torch.float64)
        tensor = (tensor + 1).reshape(dimensions)
        result = tk.decompositions.tt_rs(tensor)
        torch.testing.assert_close(result.contract_dense(), tensor)

    def test_strict_system_on_a_nondegenerate_centered_problem(self):
        tensor = _rank_two_tensor((2, 3, 2, 3, 2))
        result = tk.decompositions.TTRS(tensor).fit(
            rank=2, center=2, strict_system=True)
        torch.testing.assert_close(result.contract_dense(), tensor,
                                   rtol=1e-9, atol=1e-10)

    @pytest.mark.parametrize('options', [
        {'center': -1}, {'rank': (2,)}, {'generator': 1},
        {'collect_metrics': 1}, {'strict_system': 1}])
    def test_invalid_fit_controls_are_rejected(self, options):
        with pytest.raises((TypeError, ValueError)):
            tk.decompositions.TTRS(torch.ones(2, 2, 2)).fit(**options)

    def test_local_transform_receives_and_returns_three_axis_phis(self):
        seen = []

        def transform(phi):
            seen.append(tuple(phi.shape))
            return 2 * phi

        tensor = _rank_two_tensor((2, 3, 2))
        result = tk.decompositions.tt_rs(tensor, rank=2, transform=transform)
        torch.testing.assert_close(result.contract_dense(), 2 * tensor)
        assert len(seen) == 3
        assert all(len(shape) == 3 for shape in seen)

    @pytest.mark.parametrize('transform', [lambda phi: 1,
                                          lambda phi: phi[..., :0]])
    def test_invalid_local_transform_is_rejected(self, transform):
        with pytest.raises((TypeError, ValueError), match='transform'):
            tk.decompositions.tt_rs(torch.ones(2, 2, 2), transform=transform)


class TestQTTRSS:  # MARK: TestQTTRSS

    @pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
    @pytest.mark.parametrize('center', [0, 2, 3])
    def test_layouts_and_centers_fit_same_physical_function(self, ordering,
                                                          center):
        layout = tk.formats.QuantizedLayout(2, base=2, level=2,
                                            ordering=ordering)
        domain = torch.tensor([[0., 1.], [-1., 1.]], dtype=torch.float64)
        coordinate_map = tk.formats.AffineCoordinateMap(
            domain, layout.grid_size, grid_offset='endpoints')
        indices = _grid(layout.grid_size)
        physical = coordinate_map.from_indices(indices)
        source = tk.decompositions.CallableQuanticsSource(
            lambda x: 1 + x[:, 0] + 2 * x[:, 1], 2, layout=layout,
            coordinate_map=coordinate_map)
        result = tk.decompositions.qtt_rss(source, physical, rank=4,
                                          center=center)
        assert isinstance(result, tk.decompositions.QTTDecomposition)
        torch.testing.assert_close(result.evaluate_coordinates(physical),
                                   1 + physical[:, 0] + 2 * physical[:, 1])
        assert result.layout == layout

    def test_class_reuses_coordinate_and_digit_samples(self):
        layout = tk.formats.QuantizedLayout(1, base=2, level=3)
        source = tk.decompositions.CallableQuanticsSource(
            lambda x: 1 + x[:, 0], 1, layout=layout,
            coordinate_map=tk.formats.AffineCoordinateMap(
                torch.tensor([0., 1.]), layout.grid_size,
                grid_offset='endpoints'))
        indices = torch.arange(8).reshape(-1, 1)
        coordinates = source.coordinate_map.from_indices(indices)
        digits = source.layout.encode_indices(indices)
        engine = tk.decompositions.TTRSS(source)
        first = engine.fit(coordinates, rank=2)
        second = engine.fit(tk.decompositions.ConfigurationBatch(digits),
                            rank=2)
        torch.testing.assert_close(first.contract_dense(),
                                   second.contract_dense())
        assert first.layout == second.layout

    def test_forward_only_map_accepts_digits_and_rejects_coordinates(self):
        layout = tk.formats.QuantizedLayout(1, base=2, level=2)
        coordinate_map = tk.formats.FunctionalCoordinateMap(
            None, layout.grid_size, grid_offset='endpoints',
            forward_function=lambda unit, domain: unit.square())
        source = tk.decompositions.CallableQuanticsSource(
            lambda x: 1 + x[:, 0], 1, layout=layout,
            coordinate_map=coordinate_map)
        indices = torch.arange(4).reshape(-1, 1)
        digits = layout.encode_indices(indices)
        result = tk.decompositions.qtt_rss(
            source, digits, sample_space='digits', rank=2)
        expected = 1 + coordinate_map.from_indices(indices)[:, 0]
        torch.testing.assert_close(result.evaluate(digits), expected)
        with pytest.raises(NotImplementedError, match='inverse'):
            tk.decompositions.qtt_rss(source, torch.tensor([[0.], [1.]]))

    @pytest.mark.parametrize('out_shape', [(), (1,), (2,), (2, 2)])
    def test_scalar_and_tensor_outputs(self, out_shape):
        layout = tk.formats.QuantizedLayout(1, base=2, level=3)
        source = tk.decompositions.CallableQuanticsSource(
            lambda x: torch.ones(x.shape[0], *out_shape, dtype=x.dtype),
            1, layout=layout,
            coordinate_map=tk.formats.AffineCoordinateMap(
                torch.tensor([0., 1.], dtype=torch.float64), layout.grid_size,
                grid_offset='endpoints'))
        coordinates = torch.arange(8, dtype=torch.float64).reshape(-1, 1) / 7
        labels_count = math.prod(out_shape) if out_shape else 1
        samples = coordinates.repeat_interleave(labels_count, 0)
        labels = torch.arange(labels_count).repeat(8) if labels_count > 1 \
            else None
        result = tk.decompositions.qtt_rss(
            source, samples, rank=1) if labels is None else \
            tk.decompositions.TTRSS(source).fit(samples, labels, rank=1)
        expected_shape = () if out_shape == (1,) else out_shape
        assert result.out_shape == expected_shape
        expected = torch.ones(8, *expected_shape, dtype=torch.float64)
        torch.testing.assert_close(result.evaluate_coordinates(coordinates),
                                   expected)

    @pytest.mark.parametrize('digit_order', [
        'coarse_to_fine', ('coarse_to_fine', 'fine_to_coarse')])
    def test_result_retains_coordinates_without_retaining_source(self,
                                                               digit_order):
        layout = tk.formats.QuantizedLayout(
            2, 2, 2, ordering='interleaved', digit_order=digit_order)
        domain = torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64)
        indices = _grid((4, 4))
        coordinates = indices.to(torch.float64) / 3
        def function(x):
            return 1 + x[:, 0] + 2 * x[:, 1]
        reference = weakref.ref(function)
        source = tk.decompositions.CallableQuanticsSource(
            function, 2, layout=layout,
            coordinate_map=tk.formats.AffineCoordinateMap(
                domain, layout.grid_size, grid_offset='endpoints'))
        result = tk.decompositions.qtt_rss(source, coordinates, rank=4)
        del function, source
        gc.collect()
        assert reference() is None
        expected = 1 + coordinates[:, 0] + 2 * coordinates[:, 1]
        torch.testing.assert_close(result.evaluate_coordinates(coordinates),
                                   expected)
        restored = tk.formats.QTT.from_mps(
            result.to_mps(), n_coordinates=layout.n_coordinates, layout=layout,
            coordinate_map=result.coordinate_map)
        torch.testing.assert_close(restored.evaluate_coordinates(coordinates),
                                   expected)

    def test_invalid_wrapper_arguments(self):
        with pytest.raises(TypeError, match='source'):
            tk.decompositions.qtt_rss(torch.ones(2), torch.zeros(1, 1))
        source = tk.decompositions.CallableQuanticsSource(
            lambda x: x[:, 0], 1, base=2, level=2, domain=(0., 1.))
        with pytest.raises(ValueError, match='sample_space'):
            tk.decompositions.qtt_rss(source, torch.zeros(1, 1),
                                      sample_space='unknown')


@pytest.mark.parametrize('quantized', [False, True])
def test_rss_function_formats_devices_and_sample_error(quantized, device_dtype,
                                                       assert_close):
    device, dtype = device_dtype
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    domain = torch.tensor([0., 1.], dtype=real_dtype, device=device)
    coordinates = torch.cartesian_prod(domain, domain, domain)
    if quantized:
        coordinates = torch.arange(
            8, dtype=real_dtype, device=device)[:, None] / 8

    def function(values):
        result = torch.exp(values.sum(-1)).to(dtype)
        return result * (1 + 1j) if dtype.is_complex else result

    def embedding(values):
        return tk.embeddings.basis(values.to(torch.long), dim=2).to(dtype)

    if quantized:
        layout = tk.formats.QuantizedLayout(1, 2, 3)
        coordinate_map = tk.formats.AffineCoordinateMap(
            domain, layout.grid_size)
        source = tk.decompositions.CallableQuanticsSource(
            function, 1, layout=layout, coordinate_map=coordinate_map,
            device=device, dtype=dtype)
        engine = tk.decompositions.TTRSS(source, out_device=None)
    else:
        engine = tk.decompositions.TTRSS(
            function, embedding, domain=domain, device=device,
            dtype=dtype, out_device=None)
    result = engine.fit(coordinates, rank=1, collect_metrics=True)
    if quantized:
        actual = result.evaluate_coordinates(coordinates)
        data = layout.encode_indices(coordinate_map.to_indices(coordinates))
    else:
        data = embedding(coordinates)
        actual = result.evaluate(data)
    error = result.error(function, coordinates, data=data)
    assert_close(actual, function(coordinates))
    assert isinstance(error, tk.decompositions.ErrorRecord)
    assert error.relative < 5e-5
    assert result.dtype == dtype
    assert result.device.type == device


@pytest.mark.parametrize('quantics', [False, True])
def test_matrix_sources_are_paired_after_rss(quantics, device_dtype,
                                             assert_close):
    device, dtype = device_dtype
    if quantics:
        real_dtype = torch.empty((), dtype=dtype).real.dtype

        def function(inputs, outputs):
            return torch.exp(inputs.sum(-1) + outputs.sum(-1)).to(dtype)

        source = tk.decompositions.CallableQuanticsSource.matrix(
            function, 1, 1, in_base=2, out_base=2, in_level=2, out_level=2,
            in_domain=torch.tensor([0., 1.], dtype=real_dtype, device=device),
            out_domain=torch.tensor([0., 1.], dtype=real_dtype, device=device),
            device=device, dtype=dtype)
        samples = torch.arange(4, device=device, dtype=real_dtype)[:, None] / 4
    else:
        tensor = torch.ones(2, 2, 2, 2, dtype=dtype, device=device)
        source = tk.decompositions.DenseTensorSource.matrix(tensor)
        samples = _grid((2, 2), device)
    samples = samples.repeat_interleave(4, 0)
    labels = torch.arange(4, device=device).repeat(4)
    result = tk.decompositions.TTRSS(source, out_device=None).fit(
        samples, labels, rank=1)
    assert result.out_dim == (2, 2)
    assert_close(result.contract_dense(), source.contract_dense())
    expected_type = tk.decompositions.QTTMDecomposition if quantics \
        else tk.decompositions.TTMDecomposition
    assert isinstance(result, expected_type)


def test_finite_quantics_source_is_inferred_by_rs():
    source = tk.decompositions.CallableQuanticsSource(
        lambda x: torch.exp(x[:, 0]), 1, base=2, level=3, domain=(0., 1.))
    engine = tk.decompositions.TTRS(
        source, sketch_operator=tk.decompositions.SampledSketch())
    result = engine.fit(rank=1)
    assert engine.source is source
    assert isinstance(result, tk.decompositions.QTTDecomposition)
    torch.testing.assert_close(result.contract_dense_grid_axes(),
                               source.contract_dense_grid_axes())


@pytest.mark.parametrize('out_shape', [None, (2, 2)])
def test_continuous_matrix_rss_uses_output_basis_sites(out_shape):
    def function(coordinates):
        return torch.einsum('b,b,i,j->bij', 1 + coordinates[:, 0],
                            1 + coordinates[:, 1],
                            torch.tensor([1., 2.]), torch.tensor([3., 4.]))

    source = tk.decompositions.CallableSource.matrix(
        function, n_sites=2, out_shape=out_shape)
    coordinates = torch.cartesian_prod(torch.tensor([0., 1.]),
                                        torch.tensor([0., 1.]))
    samples = coordinates.repeat_interleave(4, 0)
    def embedding(x):
        return torch.stack((torch.ones_like(x), x), -1)
    result = tk.decompositions.TTRSS(
        source, embedding, domain=(torch.tensor([0., 1.]),) * 2).fit(
            samples, labels=torch.arange(4).repeat(4), rank=1)
    assert isinstance(result, tk.decompositions.TTMDecomposition)
    torch.testing.assert_close(result.evaluate(
        embedding(coordinates), in_sites=(0, 1), out_sites=()),
        function(coordinates))
