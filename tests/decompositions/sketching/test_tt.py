"""Tests for the refactored TT recursive-sketching decomposition."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.observers import HistoryObserver
from tensorkrowch.decompositions.results import TTDecomposition


def _problem():
    """Returns a small exactly representable scalar function."""
    domain = torch.tensor([0., 1.], dtype=torch.float64)
    samples = torch.cartesian_prod(domain, domain, domain)

    def function(data):
        return 1 + data.prod(dim=1, keepdim=True)

    def embedding(data):
        return torch.stack((1 - data, data), dim=-1)

    return function, embedding, samples, domain


class TestTTRSS:

    def test_class_returns_lightweight_result_and_reuses_fixed_problem(self):
        function, embedding, samples, domain = _problem()
        decomposer = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain)

        first = decomposer.fit(
            samples,
            rank=2,
            generator=torch.Generator().manual_seed(91))
        second = decomposer.fit(
            samples,
            rank=2,
            generator=torch.Generator().manual_seed(91))

        assert isinstance(first, TTDecomposition)
        assert first.rank == [2, 2]
        assert first.in_dim == (2, 2, 2)
        assert all(core.device.type == 'cpu' for core in first.cores)
        assert all(torch.equal(left, right)
                   for left, right in zip(first.cores, second.cores))

    def test_refactored_math_matches_dense_tensor(self):
        function, embedding, samples, domain = _problem()
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=2,
                generator=torch.Generator().manual_seed(92))

        assert torch.allclose(
            result.contract_dense(),
            function(samples).reshape(2, 2, 2),
            rtol=1e-10,
            atol=1e-12)

    def test_one_global_evaluation_plan_and_structured_records(self):
        function, embedding, samples, domain = _problem()
        observer = HistoryObserver()
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=2,
                generator=torch.Generator().manual_seed(93),
                collect_metrics=True,
                observer=observer)

        assert len(result.metrics.evaluations) == 1
        stats = result.metrics.evaluations[0]
        assert stats.requested_points > stats.unique_points
        assert stats.cache_hits == stats.requested_points - stats.unique_points
        assert len(result.metrics.input_fits) == 3
        assert len(result.metrics.truncations) == 2
        assert len(result.metrics.local_solves) == 2
        assert result.metrics.errors[0].kind == 'sketch_samples'

        names = [event.name for event in observer.events]
        assert names[0] == 'start'
        assert names[-1] == 'summary'
        assert names.count('phi.plan') == 3
        assert names.count('values.global_transform') == 1
        assert names.count('recursion.apply') == 2
        assert names.count('core.solve') == 2

    def test_nonlegacy_mode_uses_shared_range_projector(self):
        function, embedding, samples, domain = _problem()
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=2,
                generator=torch.Generator().manual_seed(95),
                legacy_projection=False,
                collect_metrics=True)

        approximation = result.evaluate(embedding(samples))
        assert torch.allclose(
            approximation,
            function(samples).squeeze(1),
            rtol=1e-10,
            atol=1e-12)
        assert len(result.metrics.range_projections) == 2
        assert all(record.method == 'randomized'
                   for record in result.metrics.range_projections)

    def test_explicit_callable_source_uses_the_same_fit_contract(self):
        function, embedding, samples, domain = _problem()
        source = tk.decompositions.CallableTensorSource(
            function,
            in_dim=(2, 2, 2),
            out_shape=(1,),
            dtype=torch.float64)
        result = tk.decompositions.TTRSS(
            source=source,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=2,
                generator=torch.Generator().manual_seed(96))

        assert torch.allclose(
            result.evaluate(embedding(samples)),
            function(samples).squeeze(1),
            rtol=1e-10,
            atol=1e-12)

    def test_functional_wrapper_keeps_legacy_and_structured_info(self):
        function, embedding, samples, domain = _problem()
        cores, info = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            generator=torch.Generator().manual_seed(94),
            verbose=False,
            return_info=True)

        assert len(cores) == 3
        assert info['total_time'] >= 0
        assert info['val_eps'] < 1e-10
        assert info['topology'] == 'tt'
        assert info['rank'] == [2, 2]
        assert info['metrics']['errors'][0]['kind'] == 'sketch_samples'

    def test_heterogeneous_coordinates_embeddings_and_in_dim(self):
        scalar_domain = torch.tensor([0., 1.], dtype=torch.float64)
        vector_domain = torch.tensor(
            [[0., 0.], [1., 1.]], dtype=torch.float64)
        first = scalar_domain.repeat_interleave(2)
        second = vector_domain.repeat((2, 1))

        def function(values):
            x, vector = values
            return 1 + x + vector.sum(dim=1)

        embeddings = (
            lambda values: torch.stack((1 - values, values), dim=1),
            lambda values: torch.cat(
                (torch.ones(values.shape[0], 1, dtype=values.dtype), values),
                dim=1),
        )
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embeddings,
            in_dim=(2, 3),
            domain=(scalar_domain, vector_domain)).fit(
                (first, second),
                rank=2,
                collect_metrics=True)

        assert result.in_dim == (2, 3)
        assert result.metrics.errors[0].kind == 'sketch_samples'
        assert result.metrics.errors[0].relative < 1e-10

    def test_tensor_output_uses_separated_sites_and_flat_labels(self):
        function, embedding, samples, domain = _problem()

        def tensor_function(data):
            value = function(data).squeeze(1)
            return torch.stack(
                (value, value + 1, 2 * value, 2 * value + 1), dim=1
            ).reshape(-1, 2, 2)

        labels = torch.arange(samples.shape[0]).remainder(4)
        result = tk.decompositions.TTRSS(
            function=tensor_function,
            embedding=embedding,
            domain=domain,
            out_position=(0, 4)).fit(
                samples,
                labels=labels,
                rank=4,
                collect_metrics=True)

        assert result.in_dim == (2, 2, 2, 2, 2)
        assert result.metadata['out_shape'] == (2, 2)
        assert result.metadata['out_position'] == (0, 4)
        assert result.metrics.errors[0].relative < 1e-10

    def test_projection_controls_total_metrics_and_out_device(self):
        function, embedding, samples, domain = _problem()
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain,
            out_device=None).fit(
                samples,
                rank=2,
                random_projection=True,
                projection_dim=2,
                collect_metrics=True)

        assert result.device == samples.device
        assert result.metrics.timings[-1].name == 'total'
        assert result.metadata['core_shapes'] == [
            tuple(core.shape) for core in result.cores]
        assert all(record.requested_dim == 2
                   for record in result.metrics.range_projections)

    def test_in_dim_and_warm_start_are_explicit(self):
        function, embedding, samples, domain = _problem()
        with pytest.raises(ValueError, match='in_dim'):
            tk.decompositions.TTRSS(
                function=function,
                embedding=embedding,
                in_dim=3,
                domain=domain).fit(samples, rank=2)

        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(samples, rank=2)
        with pytest.raises(NotImplementedError, match='warm-start'):
            tk.decompositions.TTRSS(
                function=function,
                embedding=embedding,
                domain=domain).fit(
                    samples, rank=2, warm_start=result)


__all__ = []


def _physical_grid(layout, coordinate_map, domain):
    variable_indices = torch.cartesian_prod(*(
        torch.arange(size) for size in layout.grid_size))
    if layout.n_coordinates == 1:
        variable_indices = variable_indices.reshape(-1, 1)
    physical = coordinate_map.from_indices(variable_indices)
    return variable_indices, physical


class TestQTTRSS:  # MARK: TestQTTRSS

    @pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
    def test_standard_layouts_fit_the_same_physical_function(self, ordering):
        layout = tk.decompositions.QuantizedLayout(
            2, base=2, level=2, ordering=ordering)
        domain = torch.tensor([[0., 1.], [-1., 1.]], dtype=torch.float64)
        coordinate_map = tk.formats.AffineCoordinateMap(
            domain, layout.grid_size, grid_offset="endpoints")
        variable_indices, physical = _physical_grid(
            layout, coordinate_map, domain)

        def function(values):
            return 1 + values[:, 0] + 2 * values[:, 1]

        cores, info = tk.decompositions.qtt_rss(
            function,
            physical,
            layout=layout,
            coordinate_map=coordinate_map,
            rank=4,
            legacy_projection=False,
            return_info=True)
        result = tk.decompositions.TTDecomposition(cores)
        digits = layout.encode_indices(variable_indices)

        assert torch.allclose(
            result.evaluate(digits), function(physical),
            rtol=1e-9, atol=1e-11)
        assert info['metadata']['algorithm'] == 'qtt_rss'
        assert info['metadata']['quantization']['ordering'] == ordering
        assert info['metadata']['quantization']['sample_space'] == 'physical'

    def test_class_reuses_problem_with_physical_or_digit_samples(self):
        layout = tk.decompositions.QuantizedLayout(1, base=2, level=3)
        coordinate_map = tk.formats.AffineCoordinateMap(
            torch.tensor([0., 1.]), layout.grid_size, grid_offset="endpoints")
        indices, physical = _physical_grid(
            layout, coordinate_map, torch.tensor([0., 1.]))
        digits = layout.encode_indices(indices)
        decomposer = tk.decompositions.TTRSS.quantized(
            lambda values: 1 + values[:, 0],
            layout=layout,
            coordinate_map=coordinate_map,
            )

        physical_result = decomposer.fit(
            physical, rank=2, legacy_projection=False)
        digit_result = decomposer.fit(
            digits,
            rank=2,
            legacy_projection=False,
            sample_space='digits')

        assert torch.allclose(
            physical_result.contract_dense(), digit_result.contract_dense())
        assert physical_result.metadata['quantization']['sample_space'] == \
            'physical'
        assert digit_result.metadata['quantization']['sample_space'] == \
            'digits'

    def test_digit_samples_allow_forward_only_warp(self):
        layout = tk.decompositions.QuantizedLayout(1, base=2, level=2)
        coordinate_map = tk.formats.FunctionalCoordinateMap(
            None, layout.grid_size, grid_offset="endpoints",
            forward_function=lambda unit, domain: unit.square())
        indices = torch.arange(4).reshape(-1, 1)
        digits = layout.encode_indices(indices)

        cores = tk.decompositions.qtt_rss(
            lambda values: 1 + values[:, 0],
            digits,
            layout=layout,
            coordinate_map=coordinate_map,
            sample_space='digits',
            rank=2,
            legacy_projection=False)
        result = tk.decompositions.TTDecomposition(cores)
        physical = coordinate_map.forward(
            indices.to(torch.float32) / 3)

        assert torch.allclose(result.evaluate(digits), 1 + physical[:, 0])

    @pytest.mark.parametrize('out_shape', [(2,), (2, 2)])
    def test_quantized_rss_rejects_tensor_outputs(self, out_shape):
        layout = tk.decompositions.QuantizedLayout(1, base=2, level=3)
        physical = torch.arange(8, dtype=torch.float64).reshape(-1, 1) / 7

        def function(values):
            return torch.ones(values.shape[0], *out_shape, dtype=values.dtype,
                              device=values.device)

        with pytest.raises(ValueError, match='scalar function outputs'):
            tk.decompositions.qtt_rss(
                function, physical, layout=layout,
                domain=torch.tensor([0., 1.], dtype=torch.float64), rank=4,
                return_result=True)

    @pytest.mark.parametrize('singleton_axis', [False, True])
    def test_quantized_rss_accepts_scalar_outputs(self, singleton_axis):
        layout = tk.decompositions.QuantizedLayout(1, base=2, level=3)
        physical = torch.arange(8, dtype=torch.float64).reshape(-1, 1) / 7

        def function(values):
            result = torch.ones_like(values[:, 0])
            return result.unsqueeze(-1) if singleton_axis else result

        result = tk.decompositions.qtt_rss(
            function, physical, layout=layout,
            domain=torch.tensor([0., 1.], dtype=torch.float64), rank=1,
            return_result=True)
        assert result.n_sites == layout.n_sites
        assert torch.allclose(result.evaluate_coordinates(physical), torch.ones(8, dtype=torch.float64))

    def test_physical_samples_require_an_inverse_for_custom_maps(self):
        layout = tk.decompositions.QuantizedLayout(1, base=2, level=2)
        coordinate_map = tk.formats.FunctionalCoordinateMap(
            None, layout.grid_size, grid_offset="endpoints",
            forward_function=lambda unit, domain: unit.square())
        decomposer = tk.decompositions.TTRSS.quantized(
            lambda values: values[:, 0],
            layout=layout,
            coordinate_map=coordinate_map)

        with pytest.raises(NotImplementedError, match='inverse'):
            decomposer.fit(torch.tensor([[0.], [1.]]), rank=2)


def _rank_one_sparse(dtype=torch.float64):
    left = torch.tensor([1., 2.], dtype=dtype)
    middle = torch.tensor([1., 3., 2.], dtype=dtype)
    right = torch.tensor([2., 1.], dtype=dtype)
    dense = torch.einsum('i,j,k->ijk', left, middle, right)
    indices = torch.cartesian_prod(
        torch.arange(2), torch.arange(3), torch.arange(2))
    source = tk.decompositions.SparseTensorSource(
        indices, dense.reshape(-1), dense.shape)
    return source, dense


class TestTTRS:  # MARK: TestTTRS

    @pytest.mark.parametrize('operator', [
        tk.decompositions.SampledSketch(),
        tk.decompositions.MarginalSketch.markov(),
        tk.decompositions.TTStackSketch(tt_rank=2, n_stacks=2),
    ])
    def test_sparse_rank_one_source_is_recovered(self, operator):
        source, dense = _rank_one_sparse()
        result = tk.decompositions.TTRS(
            source,
            sketch_operator=operator,
            out_device=None).fit(
                rank=1,
                generator=torch.Generator().manual_seed(301),
                strict_system=True,
                collect_metrics=True)

        assert result.rank == [1, 1]
        assert torch.allclose(
            result.contract_dense(), dense, rtol=1e-10, atol=1e-12)
        assert result.metrics.errors[0].kind == 'source_support'
        assert result.metrics.errors[0].relative < 1e-10
        assert result.metadata['source_type'] == 'SparseTensorSource'
        assert result.metadata['system']['source_path'] == 'support'

    def test_dataset_and_functional_api_are_distinct_from_rss_samples(self):
        dataset = torch.tensor([
            [0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
        cores, info = tk.decompositions.tt_rs(
            dataset=dataset,
            in_dim=(2, 2),
            rank=2,
            return_info=True)
        expected = torch.tensor(
            [[2., 1.], [1., 2.]], dtype=torch.get_default_dtype()) / 6

        result = tk.decompositions.TTDecomposition(cores)
        assert torch.allclose(result.contract_dense(), expected)
        assert info['metadata']['source_type'] == 'EmpiricalDistribution'
        assert info['metrics']['errors'][0]['kind'] == 'source_support'

    def test_tt_source_uses_structured_contractions(self):
        source, dense = _rank_one_sparse()
        tt = tk.decompositions.TTSVD(
            dense, out_device=None).fit(rank=1)
        tt_source = tk.decompositions.TTTensorSource(tt)
        result = tk.decompositions.TTRS(
            tt_source,
            sketch_operator=tk.decompositions.MarginalSketch.markov(),
            out_device=None).fit(rank=1, batch_size=3)

        assert torch.allclose(result.contract_dense(), dense)
        assert result.metadata['system']['source_path'] == 'structured_tt'
        assert result.metadata['system']['structured_kernel'] == \
            'markov_marginal'
        assert tt_source.evaluation_stats.requested_points == 0

    def test_repeated_random_fits_are_independent_and_reproducible(self):
        source, _ = _rank_one_sparse()
        decomposer = tk.decompositions.TTRS(
            source,
            sketch_operator=tk.decompositions.TTStackSketch(
                tt_rank=2, n_stacks=2),
            out_device=None)
        first = decomposer.fit(
            rank=1, generator=torch.Generator().manual_seed(302))
        second = decomposer.fit(
            rank=1, generator=torch.Generator().manual_seed(302))

        assert first is not second
        assert all(left is not right
                   for left, right in zip(first.cores, second.cores))
        assert all(torch.allclose(left, right)
                   for left, right in zip(first.cores, second.cores))

    def test_warm_start_is_explicitly_rejected(self):
        source, _ = _rank_one_sparse()
        initial = tk.decompositions.TTRS(source).fit(rank=1)
        with pytest.raises(NotImplementedError, match='warm-start'):
            tk.decompositions.TTRS(source).fit(
                rank=1, warm_start=initial)

    def test_history_observer_receives_clean_hierarchy(self):
        source, _ = _rank_one_sparse()
        observer = HistoryObserver()
        result = tk.decompositions.TTRS(source).fit(
            rank=1, observer=observer)

        assert observer.metrics is result.metrics
        assert observer.events[0].name == 'start'
        assert sum(event.name == 'site_complete'
                   for event in observer.events) == 3
        assert next(event for event in observer.events
                    if event.name == 'summary').values['rank'] == [1, 1]


def _binary_problem(dtype=torch.float64):
    """Returns a small scalar problem exactly represented at rank two."""
    domain = torch.tensor([0., 1.], dtype=torch.float64)
    samples = torch.cartesian_prod(domain, domain, domain)

    def function(data):
        values = 1 + data.prod(dim=1, keepdim=True)
        if dtype.is_complex:
            values = values + 1j * data.sum(dim=1, keepdim=True)
        return values.to(dtype)

    def embedding(data):
        return torch.stack((1 - data, data), dim=-1)

    return function, embedding, samples, domain


def _vector_problem():
    """Returns a vector function whose input/output dimensions differ."""
    domain = torch.tensor([-1., 0., 1.], dtype=torch.float64)
    samples = torch.cartesian_prod(domain, domain, domain)

    def function(data):
        return torch.stack(
            (1 + data.sum(dim=1), 2 + data.prod(dim=1)), dim=1)

    def embedding(data):
        return torch.stack((torch.ones_like(data), data, data.square()), dim=-1)

    labels = (samples[:, 0] > 0).long()
    return function, embedding, samples, domain, labels


def _in_dim(cores):
    """Extracts standard TT input dimensions from raw open-boundary cores."""
    if len(cores) == 1:
        return [cores[0].shape[0]]
    return [cores[0].shape[0], *(
        core.shape[1] for core in cores[1:])]

_DEVICE_CASES = [
    torch.device('cpu'),
    pytest.param(
        torch.device('cuda'),
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason='CUDA is unavailable')),
    pytest.param(
        torch.device('mps'),
        marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason='MPS is unavailable')),
]


class TestTTRSSPublicWorkflow:  # MARK: TestTTRSSPublicWorkflow

    def test_scalar_function_returns_mps_compatible_cores_and_info(self):
        function, embedding, samples, domain = _binary_problem()
        cores, info = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            generator=torch.Generator().manual_seed(700),
            verbose=False,
            return_info=True)
        model = tk.models.MPS(tensors=cores, parameterized=False)

        assert [tuple(core.shape) for core in cores] == [
            (2, 2), (2, 2, 2), (2, 2)]
        assert model.phys_dim == [2, 2, 2]
        assert info['total_time'] >= 0
        assert info['val_eps'] < 1e-10

    @pytest.mark.parametrize('out_position', [0, 2, 3])
    def test_vector_output_first_middle_last_with_explicit_labels(
            self, out_position):
        function, embedding, samples, domain, labels = _vector_problem()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            labels=labels,
            domain=domain,
            out_position=out_position,
            rank=4,
            generator=torch.Generator().manual_seed(701),
            verbose=False)
        model = tk.models.MPSLayer(
            tensors=cores,
            out_position=out_position,
            parameterized=False)

        expected = [3, 3, 3]
        expected.insert(out_position, 2)
        assert _in_dim(cores) == expected
        assert model.phys_dim == expected
        assert model.out_position == out_position

    def test_default_vector_output_is_equally_centered_for_one_axis(self):
        function, embedding, samples, domain, labels = _vector_problem()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            labels=labels,
            domain=domain,
            rank=4,
            generator=torch.Generator().manual_seed(702),
            verbose=False)

        assert _in_dim(cores) == [3, 3, 2, 3]

    @pytest.mark.parametrize('domain_kind', ['shared', 'per_site', 'inferred'])
    def test_shared_per_site_and_inferred_domains(self, domain_kind):
        function, embedding, samples, shared = _binary_problem()
        if domain_kind == 'shared':
            domain = shared
        elif domain_kind == 'per_site':
            domain = [shared.clone() for _ in range(samples.shape[1])]
        else:
            domain = None
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            generator=torch.Generator().manual_seed(703),
            verbose=False)

        assert [tuple(core.shape) for core in cores] == [
            (2, 2), (2, 2, 2), (2, 2)]
        assert all(torch.isfinite(core).all() for core in cores)

    def test_vector_coordinate_samples_and_domains(self):
        domain = torch.tensor(
            [[0., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)
        ids = torch.cartesian_prod(*(torch.arange(3) for _ in range(3)))
        samples = torch.stack(
            [domain.index_select(0, ids[:, site]) for site in range(3)],
            dim=1)

        def function(data):
            return (1 + data.sum(dim=(1, 2))).unsqueeze(1)

        def embedding(data):
            return torch.cat((torch.ones_like(data[..., :1]), data), dim=-1)

        cores, info = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=3,
            generator=torch.Generator().manual_seed(704),
            verbose=False,
            return_info=True)

        assert [tuple(core.shape) for core in cores] == [
            (3, 3), (3, 3, 3), (3, 3)]
        assert info['val_eps'] < 1e-10

    @pytest.mark.parametrize(
        'truncation',
        [
            {'cutoff': 0.0},
            {'atol': 0.0},
            {'rtol': 0.0},
            {'cum_percentage': 1.0},
        ])
    @pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
    def test_modern_truncation_options_reach_truncated_svd(
            self, truncation, svd_method):
        function, embedding, samples, domain = _binary_problem()
        with tk.svd_method(svd_method):
            cores = tk.decompositions.tt_rss(
                function=function,
                embedding=embedding,
                sketch_samples=samples,
                domain=domain,
                rank=2,
                generator=torch.Generator().manual_seed(705),
                verbose=False,
                **truncation)

        assert all(torch.isfinite(core).all() for core in cores)
        assert all(max(core.shape) <= 2 for core in cores)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_real_and_complex_dtype(self, dtype):
        function, embedding, samples, domain = _binary_problem(dtype)
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            device=torch.device('cpu'),
            dtype=dtype,
            generator=torch.Generator().manual_seed(706),
            verbose=False)

        assert all(core.dtype == dtype for core in cores)
        assert all(core.device.type == 'cpu' for core in cores)
        assert all(torch.isfinite(core).all() for core in cores)

    @pytest.mark.parametrize('device', _DEVICE_CASES)
    def test_explicit_compute_device_returns_final_cpu_cores(self, device):
        function, embedding, samples, domain = _binary_problem(torch.float32)
        samples = samples.float()
        domain = domain.float()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            device=device,
            dtype=torch.float32,
            generator=torch.Generator().manual_seed(706),
            verbose=False)

        assert all(core.dtype == torch.float32 for core in cores)
        assert all(core.device.type == 'cpu' for core in cores)
        assert all(torch.isfinite(core).all() for core in cores)

    def test_generator_is_reproducible_and_does_not_touch_global_rng(self):
        function, embedding, samples, _ = _vector_problem()[:4]
        torch.manual_seed(707)
        initial_state = torch.random.get_rng_state().clone()
        results = []
        for _ in range(2):
            results.append(tk.decompositions.tt_rss(
                function=function,
                embedding=embedding,
                sketch_samples=samples,
                domain=None,
                rank=3,
                generator=torch.Generator().manual_seed(708),
                verbose=False))

        assert torch.equal(torch.random.get_rng_state(), initial_state)
        assert all(torch.equal(first, second)
                   for first, second in zip(*results))


class TestGeneralizedTTRSS:  # MARK: TestGeneralizedTTRSS

    def test_scalar_output_without_artificial_axis(self):
        function, embedding, samples, domain = _binary_problem()
        tk.decompositions.tt_rss(
            function=lambda data: function(data).squeeze(1),
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            verbose=False)

    def test_heterogeneous_embeddings(self):
        function, embedding, samples, domain = _binary_problem()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=[embedding, embedding, embedding],
            sketch_samples=samples,
            domain=domain,
            rank=2,
            verbose=False)
        assert len(cores) == 3

    def test_multiple_output_axes(self):
        _, embedding, samples, domain = _binary_problem()

        def function(data):
            scalar = 1 + data.sum(dim=1)
            return torch.stack((scalar, scalar + 1, scalar + 2, scalar + 3),
                               dim=1).reshape(-1, 2, 2)

        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            verbose=False)
        assert len(cores) == 5

    def test_random_projection_can_be_disabled(self):
        function, embedding, samples, domain = _binary_problem()
        cores = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            random_projection=False,
            verbose=False)
        assert len(cores) == 3

    def test_verbosity_level_two_does_not_print_full_cores(self, capsys):
        function, embedding, samples, domain = _binary_problem()
        tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=1,
            verbose=2)
        assert 'tensor(' not in capsys.readouterr().out


def test_quantized_rss_result_retains_coordinates_without_source():
    import gc
    import weakref
    layout = tk.formats.QuantizedLayout(2, 2, 2, ordering='interleaved')
    domain = torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64)
    indices = torch.cartesian_prod(torch.arange(4), torch.arange(4))
    coordinates = indices.to(torch.float64) / 3

    def function(values):
        return 1 + values[:, 0] + 2 * values[:, 1]

    reference = weakref.ref(function)
    result = tk.decompositions.qtt_rss(
        function, coordinates, layout=layout, domain=domain, rank=4,
        legacy_projection=False, return_result=True)
    assert isinstance(result, tk.decompositions.QTTDecomposition)
    del function
    gc.collect()
    assert reference() is None
    expected = 1 + coordinates[:, 0] + 2 * coordinates[:, 1]
    assert torch.allclose(result.evaluate_coordinates(coordinates), expected, atol=1e-9)
    restored = tk.formats.QTT.from_mps(
        result.to_mps(), n_coordinates=layout.n_coordinates,
        layout=layout, coordinate_map=result.coordinate_map)
    assert torch.allclose(restored.evaluate_coordinates(coordinates), expected, atol=1e-9)


@pytest.mark.parametrize('quantized', [False, True])
def test_rss_function_formats_devices_and_sample_error(quantized,
                                                       device_dtype,
                                                       assert_close):
    engine = tk.decompositions.TTRSS
    device, dtype = device_dtype
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    domain = torch.tensor([0., 1.], dtype=real_dtype, device=device)
    coordinates = torch.cartesian_prod(domain, domain, domain)
    if quantized:
        coordinates = torch.arange(8, dtype=real_dtype, device=device).reshape(-1, 1) / 8
    def function(values):
        result = torch.exp(values.sum(-1)).to(dtype)
        return result * (1 + 1j) if dtype.is_complex else result
    def embedding(values):
        return tk.embeddings.basis(values.to(torch.long), dim=2).to(dtype)
    if quantized:
        layout = tk.formats.QuantizedLayout(1, 2, 3)
        coordinate_map = tk.formats.AffineCoordinateMap(
            torch.tensor([0., 1.], dtype=real_dtype, device=device), layout.grid_size)
        problem = engine.quantized(function, layout=layout,
                                   coordinate_map=coordinate_map,
                                   device=device, dtype=dtype, out_device=None)
    else:
        problem = engine(function, embedding, domain=domain, device=device,
                         dtype=dtype, out_device=None)
    result = problem.fit(coordinates, rank=1, collect_metrics=True)
    if quantized:
        actual = result.evaluate_coordinates(coordinates)
        error = result.error(
            function, coordinates, data=layout.encode_indices(
                coordinate_map.to_indices(coordinates)))
    else:
        inputs = embedding(coordinates)
        actual = result.evaluate(inputs)
        error = result.error(function, coordinates, data=inputs)
    assert_close(actual, function(coordinates))
    assert isinstance(error, tk.decompositions.ErrorRecord)
    assert error.relative < 5e-5
    assert result.dtype == dtype and result.device.type == device


@pytest.mark.parametrize('error', [AssertionError, IndexError])
def test_rss_keeps_unexpected_function_errors(error):
    def function(values):
        raise error('Callback failure')

    engine = tk.decompositions.TTRSS(
        function, embedding=torch.eye(2), domain=torch.arange(2))
    with pytest.raises(error, match='Callback failure'):
        engine.fit(sketch_samples=torch.zeros(2, 2, dtype=torch.long), rank=1)
