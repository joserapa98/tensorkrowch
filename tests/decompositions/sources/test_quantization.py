"""Tests for sources/quantization."""

import pytest
import torch

import tensorkrowch as tk


class TestQuantizedSourceAdapter:  # MARK: TestQuantizedSourceAdapter

    @pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
    def test_physical_callable_decodes_each_layout(self, ordering):
        layout = tk.formats.QuantizedLayout(
            3, base=2, level=2, ordering=ordering)
        domain = torch.tensor([[0., 1.], [-1., 1.], [2., 4.]])
        coordinate_map = tk.formats.AffineCoordinateMap(
            domain, layout.grid_size, grid_offset="endpoints")

        def function(physical):
            value = physical[:, 0] + \
                2 * physical[:, 1] + 3 * physical[:, 2]
            return torch.stack((value, value.square()), dim=1)

        adapter = tk.decompositions.QuantizedSourceAdapter(
            function,
            layout,
            coordinate_map,
            out_shape=(2,),
            dtype=torch.float32)
        indices = torch.tensor([[0, 0, 0], [1, 2, 3], [3, 1, 2]])
        digits = layout.encode_indices(indices)

        result = adapter.evaluate(
            tk.decompositions.ConfigurationBatch(digits))
        physical = coordinate_map.from_indices(indices)

        assert torch.equal(result, function(physical))
        assert adapter.out_shape == (2,)

    def test_indexed_tensor_source_uses_decoded_variable_indices(self):
        layout = tk.formats.QuantizedLayout(
            2, base=2, level=(2, 1), ordering='interleaved')
        dense = torch.arange(8., dtype=torch.float64).reshape(4, 2)
        adapter = tk.decompositions.QuantizedSourceAdapter(
            tk.decompositions.DenseTensorSource(dense),
            layout,
            source_space='indices')
        indices = torch.tensor([[0, 0], [3, 1], [2, 0]])

        values = adapter.evaluate(tk.decompositions.ConfigurationBatch(
            layout.encode_indices(indices)))

        assert torch.equal(values, dense[indices[:, 0], indices[:, 1]])

    def test_per_variable_coordinate_maps_compose_without_driver_branches(self):
        layout = tk.formats.QuantizedLayout(2, base=2, level=2)
        maps = (
            tk.formats.AffineCoordinateMap(torch.tensor([-1., 1.]), (4,), grid_offset="endpoints"),
            tk.formats.FunctionalCoordinateMap(
                None, (4,), grid_offset="endpoints",
                forward_function=lambda unit, domain: unit.square(),
                inverse_function=lambda coordinates, domain: coordinates.sqrt()),
        )
        adapter = tk.decompositions.QuantizedSourceAdapter(
            lambda physical: physical.sum(dim=1),
            layout,
            coordinate_map=maps,
            dtype=torch.float32)
        physical = torch.tensor([[-1., 0.], [1., 1.]])

        digits = adapter.physical_to_digits(physical)

        assert torch.allclose(adapter.digits_to_physical(digits), physical)

    def test_digit_tt_bypass_requires_and_respects_layout_metadata(self):
        grouped = tk.formats.QuantizedLayout(
            2, base=2, level=2, ordering='grouped')
        interleaved = tk.formats.QuantizedLayout(
            2, base=2, level=2, ordering='interleaved')
        variable_indices = torch.cartesian_prod(
            torch.arange(4), torch.arange(4))
        grouped_digits = grouped.encode_indices(variable_indices)
        values = (variable_indices[:, 0] +
                  10 * variable_indices[:, 1]).to(torch.float64)
        dense = torch.empty(grouped.in_dim, dtype=torch.float64)
        dense[tuple(grouped_digits.T)] = values
        tt = tk.decompositions.TTSVD(
            dense, out_device=None).fit(rank=4)
        source = tk.decompositions.TTTensorSource(tt)

        with pytest.raises(ValueError, match='source_layout'):
            tk.decompositions.QuantizedSourceAdapter(
                source, interleaved, source_space='digits')
        adapter = tk.decompositions.QuantizedSourceAdapter(
            source,
            interleaved,
            source_space='digits',
            source_layout=grouped)
        test_indices = torch.tensor([[0, 0], [2, 3], [3, 1]])
        result = adapter.evaluate(tk.decompositions.ConfigurationBatch(
            interleaved.encode_indices(test_indices)))

        assert torch.allclose(
            result,
            (test_indices[:, 0] + 10 * test_indices[:, 1]).to(torch.float64))

    def test_physical_sparse_collisions_are_coalesced(self):
        layout = tk.formats.QuantizedLayout(1, base=3, level=1)
        coordinates = torch.tensor([[0.1], [0.2], [0.9]])
        values = torch.tensor([1., 2., 4.])
        adapter = tk.decompositions.QuantizedSourceAdapter.from_physical_support(
            coordinates,
            values,
            layout,
            domain=torch.tensor([0., 1.]))
        digits = torch.tensor([[0], [1], [2]])

        result = adapter.evaluate(
            tk.decompositions.ConfigurationBatch(digits))

        assert torch.equal(result, torch.tensor([3., 0., 4.]))

    def test_physical_dataset_collisions_form_empirical_distribution(self):
        layout = tk.formats.QuantizedLayout(1, base=3, level=1)
        dataset = torch.tensor([[0.1], [0.2], [0.9], [0.9]])
        adapter = tk.decompositions.QuantizedSourceAdapter.from_physical_dataset(
            dataset,
            layout,
            domain=torch.tensor([0., 1.]))

        result = adapter.evaluate(tk.decompositions.ConfigurationBatch(
            torch.tensor([[0], [1], [2]])))

        assert torch.equal(result, torch.tensor([0.5, 0., 0.5]))


@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
@pytest.mark.parametrize('sample_space', ['indices', 'digits', 'physical'])
def test_quantized_adapter_function_evaluation_on_devices(ordering, sample_space,
                                                         device_dtype, assert_close):
    device, dtype = device_dtype
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    layout = tk.formats.QuantizedLayout(2, 2, 2, ordering=ordering)
    coordinate_map = tk.formats.AffineCoordinateMap(
        torch.tensor([[-1., 1.], [0., 2.]], dtype=real_dtype, device=device),
        layout.grid_size)
    def function(coordinates):
        values = torch.exp(coordinates[:, 0] + 2 * coordinates[:, 1]).to(dtype)
        return values * (1 + 1j) if dtype.is_complex else values
    adapter = tk.decompositions.QuantizedSourceAdapter(
        function, layout, coordinate_map=coordinate_map, dtype=dtype, device=device)
    indices = torch.tensor([[0, 0], [1, 2], [3, 1]], device=device)
    digits = layout.encode_indices(indices)
    coordinates = coordinate_map.from_indices(indices)
    inputs = {'indices': indices, 'digits': digits, 'physical': coordinates}[sample_space]
    if sample_space == 'indices':
        encoded = layout.encode_indices(inputs)
    elif sample_space == 'physical':
        encoded = adapter.physical_to_digits(inputs)
    else:
        encoded = inputs
    assert_close(adapter.evaluate(tk.decompositions.ConfigurationBatch(encoded)), function(coordinates))
    assert_close(adapter.digits_to_physical(encoded), coordinates)
    with pytest.raises(ValueError, match='domain'):
        tk.decompositions.QuantizedSourceAdapter(
            function, layout, coordinate_map=coordinate_map, domain=torch.tensor([0., 1.]))
