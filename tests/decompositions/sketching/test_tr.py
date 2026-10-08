"""Tests for Tensor Ring recursive sketching from samples."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.observers import HistoryObserver


def _rank_one_problem(n_sites=4):
    domain = torch.tensor([0., 1.], dtype=torch.float64)
    samples = torch.cartesian_prod(*(domain for _ in range(n_sites)))

    def function(data):
        return (1 + data).prod(dim=1)

    def embedding(values):
        return torch.stack((1 - values, values), dim=-1)

    return function, embedding, samples, domain


class TestTRRSS:

    def test_rank_one_function_matches_dense_tensor(self):
        function, embedding, samples, domain = _rank_one_problem()
        result = tk.decompositions.TRRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=1,
                collect_metrics=True)

        expected = function(samples).reshape(2, 2, 2, 2)
        assert isinstance(result, tk.decompositions.TRDecomposition)
        assert result.rank == [1, 1, 1, 1]
        assert torch.allclose(
            result.contract_dense(), expected, rtol=1e-8, atol=1e-10)
        assert result.metrics.errors[0].kind == 'sketch_samples'
        assert result.metrics.errors[0].relative < 1e-8
        assert len(result.metrics.truncations) == 3

    def test_repeated_fits_are_independent_and_functional_api_is_compatible(
            self):
        function, embedding, samples, domain = _rank_one_problem()
        decomposer = tk.decompositions.TRRSS(
            function=function,
            embedding=embedding,
            domain=domain)
        first = decomposer.fit(samples, rank=1)
        second = decomposer.fit(samples, rank=1)
        cores, info = tk.decompositions.tr_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=1,
            return_info=True)

        assert first is not second
        assert all(left is not right for left, right in zip(
            first.cores, second.cores))
        assert len(cores) == 4
        assert info['topology'] == 'tr'
        assert info['metadata']['requested_rank'] == [1, 1, 1, 1]

    def test_tensor_outputs_use_basis_sites_and_flat_labels(self):
        function, embedding, samples, domain = _rank_one_problem()

        def tensor_function(data):
            value = function(data)
            left = torch.tensor([1., 2.], dtype=data.dtype)
            right = torch.tensor([1., 3.], dtype=data.dtype)
            return value[:, None, None] * left[None, :, None] * \
                right[None, None, :]

        labels = torch.arange(samples.shape[0]).remainder(4)
        result = tk.decompositions.TRRSS(
            function=tensor_function,
            embedding=embedding,
            domain=domain,
            out_position=(1, 4)).fit(
                samples,
                labels=labels,
                rank=1,
                collect_metrics=True)

        assert result.in_dim == (2, 2, 2, 2, 2, 2)
        assert result.metadata['out_position'] == (1, 4)
        assert result.metrics.errors[0].relative < 1e-8

    def test_adaptive_block_discovers_ranks_and_padding_is_opt_in(self):
        function, embedding, samples, domain = _rank_one_problem(n_sites=5)
        decomposer = tk.decompositions.TRRSS(
            function=function,
            embedding=embedding,
            domain=domain)
        adaptive = decomposer.fit(
            samples,
            rank=(2, 2, 2, 2, 2),
            adaptive=True,
            collect_metrics=True)
        padded = decomposer.fit(
            samples,
            rank=2,
            adaptive=True,
            pad_to_rank=True)

        assert adaptive.rank == [1, 1, 1, 1, 1]
        assert adaptive.metadata['center_block'] == (1, 2, 3)
        assert adaptive.metadata['rank_estimate']['limitations'] == ()
        assert adaptive.metrics.errors[0].relative < 1e-8
        assert padded.rank == [2, 2, 2, 2, 2]
        assert torch.allclose(
            adaptive.contract_dense(), padded.contract_dense(),
            rtol=1e-10, atol=1e-12)

    def test_alternating_schedule_solves_odd_ring_in_two_stages(self):
        function, embedding, samples, domain = _rank_one_problem(n_sites=5)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRSS(
                function=function,
                embedding=embedding,
                domain=domain).fit(
                    samples,
                    rank=1,
                    schedule='alternating',
                    collect_metrics=True)

        assert result.metadata['schedule'] == 'alternating'
        assert result.metadata['requested_schedule'] == 'alternating'
        assert result.metadata['center_block'] == (2,)
        assert result.metrics.errors[0].relative < 1e-8

    def test_alternating_schedule_records_even_ring_fallback(self):
        function, embedding, samples, domain = _rank_one_problem(n_sites=4)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRSS(
                function=function,
                embedding=embedding,
                domain=domain).fit(
                    samples,
                    rank=1,
                    schedule='alternating')

        assert result.metadata['schedule'] == 'center_out'
        assert result.metadata['requested_schedule'] == 'alternating'
        assert torch.allclose(
            result.contract_dense(), function(samples).reshape(2, 2, 2, 2),
            rtol=1e-8, atol=1e-10)


__all__ = []


def _physical_grid(layout, coordinate_map, domain):
    variable_indices = torch.cartesian_prod(*(
        torch.arange(size) for size in layout.grid_size))
    if layout.n_coordinates == 1:
        variable_indices = variable_indices.reshape(-1, 1)
    physical = coordinate_map.from_indices(variable_indices)
    return variable_indices, physical


class TestQTRRSS:  # MARK: TestQTRRSS

    def test_qtr_functional_and_class_apis_use_ring_driver(self):
        layout = tk.decompositions.QuantizedLayout(1, base=2, level=3)
        coordinate_map = tk.formats.AffineCoordinateMap(
            torch.tensor([0., 1.]), layout.grid_size, grid_offset="endpoints")
        indices, physical = _physical_grid(
            layout, coordinate_map, torch.tensor([0., 1.]))

        cores, info = tk.decompositions.qtr_rss(
            lambda values: torch.ones_like(values[:, 0]),
            physical,
            layout=layout,
            coordinate_map=coordinate_map,
            rank=1,
            return_info=True)
        result = tk.decompositions.TRDecomposition(cores)

        assert result.rank == [1, 1, 1]
        assert torch.allclose(
            result.evaluate(layout.encode_indices(indices)),
            torch.ones(indices.shape[0]))
        assert info['metadata']['algorithm'] == 'qtr_rss'
        assert info['metadata']['quantization']['n_coordinates'] == 1


def _rank_one_sparse(dtype=torch.float64):
    factors = [
        torch.tensor([1., 2.], dtype=dtype),
        torch.tensor([1., 3.], dtype=dtype),
        torch.tensor([2., 1.], dtype=dtype),
    ]
    dense = torch.einsum('i,j,k->ijk', *factors)
    indices = torch.cartesian_prod(*(
        torch.arange(dim) for dim in dense.shape))
    source = tk.decompositions.SparseTensorSource(
        indices, dense.reshape(-1), dense.shape)
    return source, dense


class TestTRRS:  # MARK: TestTRRS

    @pytest.mark.parametrize('operator', [
        tk.decompositions.SampledSketch(),
        tk.decompositions.MarginalSketch.markov(),
        tk.decompositions.TTStackSketch(tt_rank=2, n_stacks=2),
    ])
    def test_sparse_rank_one_source_is_recovered(self, operator):
        source, dense = _rank_one_sparse()
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRS(
                source,
                sketch_operator=operator,
                out_device=None).fit(
                    rank=1,
                    generator=torch.Generator().manual_seed(311),
                    strict_system=True,
                    collect_metrics=True)

        assert result.rank == [1, 1, 1]
        assert torch.allclose(
            result.contract_dense(), dense, rtol=1e-9, atol=1e-11)
        assert result.metrics.fidelities[0].fidelity > 1 - 1e-10
        assert result.metrics.errors[-1].kind == 'source_support'
        assert result.metrics.errors[-1].relative < 1e-9
        assert result.metadata['algorithm'] == 'tr_rs'
        assert result.metadata['experimental'] is True

    def test_dataset_functional_api_returns_information(self):
        dataset = torch.tensor([
            [0, 0, 0], [0, 0, 0], [1, 1, 1]])
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            cores, info = tk.decompositions.tr_rs(
                dataset=dataset,
                in_dim=(2, 2, 2),
                rank=1,
                return_info=True)

        result = tk.decompositions.TRDecomposition(cores)
        assert result.rank == [1, 1, 1]
        assert info['metadata']['source_type'] == 'EmpiricalDistribution'
        assert info['metrics']['fidelities'][0]['fidelity'] > 1 - 1e-5

    def test_tt_source_uses_structured_contractions(self):
        _, dense = _rank_one_sparse()
        tt = tk.decompositions.TTSVD(
            dense, out_device=None).fit(rank=1)
        source = tk.decompositions.TTTensorSource(tt)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRS(
                source, out_device=None).fit(rank=1)

        assert torch.allclose(result.contract_dense(), dense)
        assert result.metrics.fidelities[0].fidelity > 1 - 1e-10
        assert source.evaluation_stats.requested_points == 0

    def test_history_observer_receives_one_tr_rs_hierarchy(self):
        source, _ = _rank_one_sparse()
        observer = HistoryObserver()
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRS(source).fit(
                rank=1, observer=observer)

        assert observer.metrics is result.metrics
        assert observer.events[0].name == 'start'
        assert all(event.phase == 'TR-RS' for event in observer.events)
        assert sum(event.name == 'site_complete'
                   for event in observer.events) == 3

    def test_rank_and_warm_start_validation(self):
        source, _ = _rank_one_sparse()
        with pytest.raises(TypeError, match='rank'):
            tk.decompositions.TRRS(source).fit(rank=[1, 1, 1])
        initial = tk.decompositions.TRDecomposition([
            torch.ones(1, 2, 1) for _ in range(3)])
        with pytest.raises(NotImplementedError, match='warm-start'):
            tk.decompositions.TRRS(source).fit(
                rank=1, warm_start=initial)


@pytest.mark.parametrize('quantized', [False, True])
def test_rss_function_formats_devices_and_sample_error(quantized,
                                                       device_dtype,
                                                       assert_close):
    engine = tk.decompositions.TRRSS
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


@pytest.mark.parametrize('out_shape', [(2,), (2, 2)])
def test_quantized_rss_rejects_tensor_outputs(out_shape):
    layout = tk.decompositions.QuantizedLayout(1, base=2, level=3)
    physical = torch.arange(8, dtype=torch.float64).reshape(-1, 1) / 7

    def function(values):
        return torch.ones(values.shape[0], *out_shape, dtype=values.dtype,
                          device=values.device)

    with pytest.raises(ValueError, match='scalar function outputs'):
        tk.decompositions.qtr_rss(
            function, physical, layout=layout,
            domain=torch.tensor([0., 1.], dtype=torch.float64), rank=4,
            return_result=True)


@pytest.mark.parametrize('singleton_axis', [False, True])
def test_quantized_rss_accepts_scalar_outputs(singleton_axis):
    layout = tk.decompositions.QuantizedLayout(1, base=2, level=3)
    physical = torch.arange(8, dtype=torch.float64).reshape(-1, 1) / 7

    def function(values):
        result = torch.ones_like(values[:, 0])
        return result.unsqueeze(-1) if singleton_axis else result

    result = tk.decompositions.qtr_rss(
        function, physical, layout=layout,
        domain=torch.tensor([0., 1.], dtype=torch.float64), rank=1,
        return_result=True)
    assert result.n_sites == layout.n_sites
    assert torch.allclose(result.evaluate_coordinates(physical), torch.ones(8, dtype=torch.float64))
