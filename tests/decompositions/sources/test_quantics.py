"""Tests for callable Quantics sources and their finite tensor views."""

import pytest
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions.sources.base import _tensor_view


class TestCallableQuanticsSource:  # MARK: TestCallableQuanticsSource

    @pytest.mark.parametrize('ordering', ['interleaved', 'grouped', 'custom'])
    @pytest.mark.parametrize('digit_order', [
        'coarse_to_fine', 'fine_to_coarse',
        ('coarse_to_fine', 'fine_to_coarse')])
    def test_evaluation_and_materialization(self, ordering, digit_order,
                                           device_dtype, assert_close):
        device, dtype = device_dtype
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        layout = tk.formats.QuantizedLayout(
            2, base=(2, 3), level=(2, 1), ordering=ordering,
            digit_order=digit_order,
            permutation=[(1, 0), (0, 1), (0, 0)] if ordering == 'custom' else None)
        coordinate_map = tk.formats.AffineCoordinateMap(
            torch.tensor([[-1., 1.], [2., 4.]], dtype=real_dtype, device=device),
            layout.grid_size)

        def function(coordinates):
            result = torch.exp(coordinates.sum(-1)).to(dtype)
            return result * (1 + 1j) if dtype.is_complex else result

        source = tk.decompositions.CallableQuanticsSource(
            function, 2, layout=layout, coordinate_map=coordinate_map,
            dtype=dtype, device=device, batch_size=5)
        indices = torch.cartesian_prod(torch.arange(4), torch.arange(3)).to(device)
        digits = layout.encode_indices(indices)
        expected = function(coordinate_map.from_indices(indices))

        assert source.is_discrete_tensor and not source.is_matrix
        assert not callable(source)
        assert source.out_shape is None
        assert_close(source.evaluate(tk.decompositions.ConfigurationBatch(digits)),
                     expected)
        assert source.out_shape == ()
        assert_close(source.contract_dense_grid_axes(), expected.reshape(4, 3))
        assert_close(source.contract_dense()[tuple(digits.T)], expected)
        assert source.dtype == dtype and source.device.type == device
        assert isinstance(source, tk.decompositions.BaseSource)

    @pytest.mark.parametrize('path', ['uniform', 'explicit', 'objects'])
    def test_constructor_paths_match_formats(self, path):
        if path == 'uniform':
            kwargs = dict(base=2, level=2, domain=(0., 1.))
        elif path == 'explicit':
            kwargs = dict(base=2, level=2,
                          grid_coordinates=(torch.tensor([-2., 0., 1., 3.]),))
        else:
            layout = tk.formats.QuantizedLayout(1, 2, 2)
            kwargs = dict(layout=layout, coordinate_map=tk.formats.AffineCoordinateMap(
                (0., 1.), layout.grid_size))
        source = tk.decompositions.CallableQuanticsSource(
            lambda coordinates: coordinates.sum(-1), 1, **kwargs)
        expected = source.coordinate_map.from_indices(torch.arange(4)[:, None]).sum(-1)
        torch.testing.assert_close(source.contract_dense_grid_axes(), expected)
        assert source.layout.ordering == 'interleaved'
        assert source.layout.digit_order == ('coarse_to_fine',)

    def test_fibers_chunking_empty_batches_and_gradients(self):
        calls = []
        parameter = torch.tensor(2., requires_grad=True)

        def function(coordinates):
            calls.append(coordinates.shape[0])
            return parameter * coordinates[:, 0]

        source = tk.decompositions.CallableQuanticsSource(
            function, 1, base=2, level=2, domain=(0., 1.),
            out_shape=(), dtype=torch.float32, batch_size=3)
        assert source.evaluate(tk.decompositions.ConfigurationBatch(
            torch.empty(0, 2, dtype=torch.long))).shape == (0,)
        assert not calls
        source.contract_dense().sum().backward()
        assert calls == [3, 1]
        torch.testing.assert_close(parameter.grad, torch.tensor(1.5))
        batch = tk.decompositions.ConfigurationBatch(torch.tensor([[0, 0], [1, 0]]))
        torch.testing.assert_close(source.fiber(batch, 1),
                                   torch.tensor([[0., 0.5], [1., 1.5]]))

    @pytest.mark.parametrize('kwargs, error', [
        ({}, ValueError),
        ({'base': 2, 'domain': (0., 1.)}, ValueError),
        ({'base': 2, 'level': 2}, ValueError),
        ({'base': 2, 'level': 2, 'domain': (0., 1.),
          'grid_coordinates': (torch.arange(4.),)}, ValueError),
        ({'base': 2, 'level': 2,
          'grid_coordinates': (torch.arange(3.),)}, ValueError),
        ({'layout': tk.formats.QuantizedLayout(1, 2, 2)}, TypeError),
        ({'layout': tk.formats.QuantizedLayout(1, 2, 2),
          'coordinate_map': tk.formats.AffineCoordinateMap((0., 1.), (3,))}, ValueError),
    ])
    def test_invalid_construction(self, kwargs, error):
        with pytest.raises(error):
            tk.decompositions.CallableQuanticsSource(lambda x: x[:, 0], 1, **kwargs)

    @pytest.mark.parametrize('function', [torch.ones(4),
        tk.decompositions.DenseTensorSource(torch.ones(4)),
        tk.formats.TT([torch.ones(4)])])
    def test_tensor_sources_are_not_quantized(self, function):
        with pytest.raises(TypeError, match='coordinate callable'):
            tk.decompositions.CallableQuanticsSource(
                function, 1, base=2, level=2, domain=(0., 1.))

    @pytest.mark.parametrize('digits, error', [
        (torch.tensor([[0, 2]]), ValueError),
        (torch.tensor([[-1, 0]]), ValueError),
        (torch.tensor([[0, 0, 0]]), ValueError),
        (torch.tensor([[0., 1.]]), TypeError),
    ])
    def test_invalid_digit_configurations(self, digits, error):
        source = tk.decompositions.CallableQuanticsSource(
            lambda x: x[:, 0], 1, base=2, level=2, domain=(0., 1.))
        with pytest.raises(error):
            source.evaluate(tk.decompositions.ConfigurationBatch(digits))

    def test_response_axes_are_selected_by_the_numerical_view(self):
        source = tk.decompositions.CallableQuanticsSource(
            lambda x: torch.stack((x[:, 0], x[:, 0] + 1), -1),
            1, base=2, level=2, domain=(0., 1.))
        view = _tensor_view(source)
        assert view.in_dim == (2, 2, 2)
        result = tk.decompositions.TTSVD(source).fit()
        assert result.in_sites == (0, 1)
        assert result.out_shape == (2,)
        torch.testing.assert_close(result.contract_dense_grid_axes(),
                                   source.contract_dense_grid_axes())
        coordinates = torch.tensor([[0.], [0.5]])
        torch.testing.assert_close(result.evaluate_coordinates(coordinates),
                                   source.function(coordinates))

    def test_empirical_data_is_quantized_before_coalescing(self):
        source = tk.decompositions.CallableQuanticsSource.from_data(
            torch.tensor([[0.1], [0.2], [0.8]]), 1,
            weights=torch.tensor([1., 2., 1.]),
            base=2, level=1, domain=(0., 1.))
        torch.testing.assert_close(source.contract_dense_grid_axes(),
                                   torch.tensor([0.75, 0.25]))
        assert source.is_discrete_tensor and not source.is_matrix


class TestCallableQuanticsMatrixSource:  # MARK: TestCallableQuanticsMatrixSource

    @pytest.mark.parametrize('in_path', ['uniform', 'explicit', 'objects'])
    @pytest.mark.parametrize('out_path', ['uniform', 'explicit', 'objects'])
    def test_independent_paths(self, in_path, out_path, device_dtype, assert_close):
        device, dtype = device_dtype
        real_dtype = torch.empty((), dtype=dtype).real.dtype

        def options(side, path):
            if path == 'uniform':
                return {side + '_base': 2, side + '_level': 2,
                        side + '_domain': torch.tensor([0., 1.], device=device,
                                                       dtype=real_dtype)}
            if path == 'explicit':
                return {side + '_base': 2, side + '_level': 2,
                        side + '_grid_coordinates': (torch.tensor(
                            [-2., 0., 1., 3.], device=device, dtype=real_dtype),)}
            layout = tk.formats.QuantizedLayout(1, 2, 2, digit_order='fine_to_coarse')
            return {side + '_layout': layout,
                    side + '_coordinate_map': tk.formats.AffineCoordinateMap(
                        torch.tensor([1., 2.], device=device, dtype=real_dtype),
                        layout.grid_size)}

        def function(inputs, outputs):
            values = (inputs[:, 0] + 2 * outputs[:, 0]).to(dtype)
            return values * (1 + 1j) if dtype.is_complex else values

        source = tk.decompositions.CallableQuanticsSource.matrix(
            function, 1, 1, **options('in', in_path), **options('out', out_path),
            dtype=dtype, device=device, batch_size=3)
        inputs, outputs = torch.cartesian_prod(
            torch.arange(4, device=device), torch.arange(4, device=device)).split(1, -1)
        in_digits = source.in_layout.encode_indices(inputs)
        out_digits = source.out_layout.encode_indices(outputs)
        expected = function(source.in_coordinate_map.from_indices(inputs),
                            source.out_coordinate_map.from_indices(outputs))
        fused = in_digits * in_digits.new_tensor(source.out_dim) + out_digits
        assert_close(_tensor_view(source).evaluate(
            tk.decompositions.ConfigurationBatch(fused)), expected)
        response = source.evaluate(tk.decompositions.ConfigurationBatch(in_digits))
        assert_close(response.reshape(16, -1).gather(
            1, (out_digits[:, 0] * 2 + out_digits[:, 1])[:, None]).squeeze(1), expected)
        assert_close(source.contract_dense_grid_axes(), expected.reshape(4, 4))
        paired = torch.stack((in_digits, out_digits), -1).flatten(-2)
        assert_close(source.contract_dense()[tuple(paired.T)], expected)

    def test_mismatched_site_counts(self):
        with pytest.raises(ValueError, match='sites'):
            tk.decompositions.CallableQuanticsSource.matrix(
                lambda x, y: x[:, 0], 1, 1, in_base=2, out_base=2,
                in_level=2, out_level=3, in_domain=(0., 1.), out_domain=(0., 1.))


def test_data_kernel_uses_quantized_dataset_positions():
    dataset = torch.tensor([[0.1], [0.2], [0.8]], dtype=torch.float64)
    source = tk.decompositions.CallableQuanticsSource.from_data(
        dataset, 1, base=2, level=1, domain=(0., 1.),
        kernel=tk.decompositions.gaussian_kernel, bandwidth=0.2)
    quantized_data = torch.tensor([[0.], [0.], [0.5]], dtype=torch.float64)
    coordinates = torch.tensor([[0.], [0.5]], dtype=torch.float64)
    expected = tk.decompositions.gaussian_kernel(
        coordinates, quantized_data, bandwidth=0.2).mean(-1)
    torch.testing.assert_close(source.contract_dense_grid_axes(), expected)
