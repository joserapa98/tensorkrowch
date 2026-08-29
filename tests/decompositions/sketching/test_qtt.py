"""Tests for QTT/QTR recursive sketching adapters and public APIs."""

import pytest

import torch
import tensorkrowch as tk


def _physical_grid(layout, coordinate_map, domain):
    variable_indices = torch.cartesian_prod(*(
        torch.arange(size) for size in layout.grid_size))
    if layout.n_variables == 1:
        variable_indices = variable_indices.reshape(-1, 1)
    physical = coordinate_map.from_indices(
        variable_indices, layout.grid_size, domain)
    return variable_indices, physical


class TestQTTRSS:  # MARK: TestQTTRSS

    @pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
    def test_standard_layouts_fit_the_same_physical_function(self, ordering):
        layout = tk.decompositions.QuantizedLayout(
            2, base=2, level=2, ordering=ordering)
        coordinate_map = tk.decompositions.UniformCoordinateMap()
        domain = torch.tensor([[0., 1.], [-1., 1.]], dtype=torch.float64)
        variable_indices, physical = _physical_grid(
            layout, coordinate_map, domain)

        def function(values):
            return 1 + values[:, 0] + 2 * values[:, 1]

        cores, info = tk.decompositions.qtt_rss(
            function,
            physical,
            layout=layout,
            coordinate_map=coordinate_map,
            domain=domain,
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
        coordinate_map = tk.decompositions.UniformCoordinateMap()
        indices, physical = _physical_grid(
            layout, coordinate_map, torch.tensor([0., 1.]))
        digits = layout.encode_indices(indices)
        decomposer = tk.decompositions.TTRSS.quantized(
            lambda values: 1 + values[:, 0],
            layout=layout,
            coordinate_map=coordinate_map,
            domain=torch.tensor([0., 1.]))

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
        coordinate_map = tk.decompositions.WarpedCoordinateMap(
            lambda unit, domain: unit.square())
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

    def test_tensor_outputs_become_basis_sites(self):
        layout = tk.decompositions.QuantizedLayout(1, base=2, level=2)
        indices = torch.arange(4).reshape(-1, 1)
        physical = indices.to(torch.float64) / 3
        repeated = physical.repeat_interleave(4, dim=0)
        labels = torch.arange(4).repeat(4)

        def function(values):
            base = 1 + values[:, 0]
            return torch.stack((
                base, base + 1, 2 * base, 2 * base + 1), dim=1
            ).reshape(-1, 2, 2)

        cores, info = tk.decompositions.qtt_rss(
            function,
            repeated,
            layout=layout,
            domain=torch.tensor([0., 1.], dtype=torch.float64),
            labels=labels,
            rank=4,
            legacy_projection=False,
            return_info=True)

        assert len(cores) == 4
        assert info['metadata']['output_shape'] == (2, 2)
        assert info['metadata']['quantization']['domain'].dtype == \
            torch.float64
        assert info['metrics']['errors'][0]['relative'] < 1e-9

    def test_physical_samples_require_an_inverse_for_custom_maps(self):
        layout = tk.decompositions.QuantizedLayout(1, base=2, level=2)
        coordinate_map = tk.decompositions.WarpedCoordinateMap(
            lambda unit, domain: unit.square())
        decomposer = tk.decompositions.TTRSS.quantized(
            lambda values: values[:, 0],
            layout=layout,
            coordinate_map=coordinate_map)

        with pytest.raises(NotImplementedError, match='inverse'):
            decomposer.fit(torch.tensor([[0.], [1.]]), rank=2)


class TestQTRRSS:  # MARK: TestQTRRSS

    def test_qtr_functional_and_class_apis_use_ring_driver(self):
        layout = tk.decompositions.QuantizedLayout(1, base=2, level=3)
        coordinate_map = tk.decompositions.UniformCoordinateMap()
        indices, physical = _physical_grid(
            layout, coordinate_map, torch.tensor([0., 1.]))

        cores, info = tk.decompositions.qtr_rss(
            lambda values: torch.ones_like(values[:, 0]),
            physical,
            layout=layout,
            coordinate_map=coordinate_map,
            domain=torch.tensor([0., 1.]),
            rank=1,
            return_info=True)
        result = tk.decompositions.TRDecomposition(cores)

        assert result.rank == [1, 1, 1]
        assert torch.allclose(
            result.evaluate(layout.encode_indices(indices)),
            torch.ones(indices.shape[0]))
        assert info['metadata']['algorithm'] == 'qtr_rss'
        assert info['metadata']['quantization']['n_variables'] == 1


__all__ = []
