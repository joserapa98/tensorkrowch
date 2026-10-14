"""Tests for sources/quantics."""

import pytest
import torch

import tensorkrowch as tk


class TestQuanticsVectorSource:  # MARK: TestQuanticsVectorSource

    @pytest.mark.parametrize('ordering', ['interleaved', 'grouped', 'custom'])
    @pytest.mark.parametrize('digit_order', ['coarse_to_fine', 'fine_to_coarse'])
    def test_evaluation_and_materialization(self, ordering, digit_order,
                                           device_dtype, assert_close):
        device, dtype = device_dtype
        real_dtype = torch.empty((), dtype=dtype).real.dtype
        permutation = [(1, 0), (0, 1), (0, 0)] if ordering == 'custom' else None
        layout = tk.formats.QuantizedLayout(
            2, base=(2, 3), level=(2, 1), ordering=ordering,
            digit_order=digit_order, permutation=permutation)
        coordinate_map = tk.formats.AffineCoordinateMap(
            torch.tensor([[-1., 1.], [2., 4.]], dtype=real_dtype, device=device),
            layout.grid_size)

        def function(coordinates):
            result = torch.exp(coordinates[:, 0] + coordinates[:, 1]).to(dtype)
            return result * (1 + 1j) if dtype.is_complex else result

        source = tk.decompositions.QuanticsVectorSource(
            function, 2, layout=layout, coordinate_map=coordinate_map,
            dtype=dtype, device=device, batch_size=5)
        indices = torch.cartesian_prod(torch.arange(4), torch.arange(3)).to(device)
        coordinates = coordinate_map.from_indices(indices)
        digits = layout.encode_indices(indices)
        expected = function(coordinates)

        assert isinstance(source, tk.decompositions.TensorSource)
        assert isinstance(source, tk.decompositions.FiberTensorSource)
        assert not callable(source)
        assert source.in_dim == layout.in_dim
        assert source.out_shape == ()
        assert_close(source.evaluate_digits(digits), expected)
        assert_close(source.evaluate_indices(indices), expected)
        assert_close(source.evaluate_coordinates(coordinates), expected)
        assert_close(source.evaluate(tk.decompositions.ConfigurationBatch(digits)),
                     expected)
        assert_close(source.coordinates_to_digits(coordinates), digits)
        assert_close(source.digits_to_coordinates(digits), coordinates)
        assert_close(source.to_dense_grid(), expected.reshape(4, 3))
        dense = source.to_dense_digits()
        assert dense.shape == source.in_dim
        assert_close(dense[tuple(digits.T)], expected)
        assert source.dtype == dtype and source.device.type == device

    @pytest.mark.parametrize('path', ['uniform', 'explicit', 'objects'])
    def test_constructor_paths_match_formats(self, path):
        def function(coordinates):
            return coordinates.sum(-1)
        if path == 'uniform':
            kwargs = dict(base=2, level=2, domain=torch.tensor([0., 1.]))
        elif path == 'explicit':
            kwargs = dict(base=2, level=2,
                          grid_coordinates=(torch.tensor([-2., 0., 1., 3.]),))
        else:
            layout = tk.formats.QuantizedLayout(1, 2, 2)
            coordinate_map = tk.formats.FunctionalCoordinateMap(
                None, layout.grid_size,
                forward_function=lambda unit, domain: unit.square(),
                inverse_function=lambda coordinates, domain: coordinates.sqrt())
            kwargs = dict(layout=layout, coordinate_map=coordinate_map)
        source = tk.decompositions.QuanticsVectorSource(function, 1, **kwargs)
        indices = torch.arange(4).reshape(-1, 1)
        expected = source.coordinate_map.from_indices(indices).sum(-1)
        torch.testing.assert_close(source.to_dense_grid(), expected)
        assert source.layout.ordering == 'interleaved'
        assert source.layout.digit_order == 'coarse_to_fine'
        assert source.dtype == expected.dtype

    def test_grid_evaluation_does_not_replace_continuous_function(self):
        source = tk.decompositions.QuanticsVectorSource(
            lambda coordinates: coordinates[:, 0].square(), 1,
            base=2, level=2, domain=torch.tensor([0., 1.]))
        coordinates = torch.tensor([[0.3]])
        torch.testing.assert_close(source.function(coordinates), torch.tensor([0.09]))
        torch.testing.assert_close(source.evaluate_coordinates(coordinates),
                                   torch.tensor([0.0625]))

    def test_fibers_and_batch_shapes(self):
        source = tk.decompositions.QuanticsVectorSource(
            lambda coordinates: coordinates[:, 0], 1,
            base=2, level=2, domain=torch.tensor([0., 1.]))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [1, 0]]))
        torch.testing.assert_close(source.fiber(configurations, 1),
                                   torch.tensor([[0., 0.25], [0.5, 0.75]]))
        torch.testing.assert_close(
            source.fiber(configurations, 1, torch.tensor([1])),
            torch.tensor([[0.25], [0.75]]))
        digits = torch.tensor([[[0, 1]], [[1, 1]]])
        torch.testing.assert_close(source.evaluate_digits(digits),
                                   torch.tensor([[0.25], [0.75]]))
        assert source.evaluate_digits(torch.tensor([1, 1])).shape == ()
        for site in (-1, 2):
            with pytest.raises(ValueError, match='site'):
                source.fiber(configurations, site)
        with pytest.raises(TypeError, match='site'):
            source.fiber(configurations, True)

    def test_chunking_empty_batches_and_autograd(self):
        calls = []
        parameter = torch.tensor(2., requires_grad=True)

        def function(coordinates):
            calls.append(coordinates.shape[0])
            return parameter * coordinates[:, 0]

        source = tk.decompositions.QuanticsVectorSource(
            function, 1, base=2, level=2, domain=torch.tensor([0., 1.]),
            dtype=torch.float32, batch_size=3)
        assert source.evaluate_digits(torch.empty(0, 2, dtype=torch.long)).shape == (0,)
        assert calls == []
        source.to_dense_digits().sum().backward()
        assert calls == [3, 1]
        torch.testing.assert_close(parameter.grad, torch.tensor(1.5))
        assert source.evaluation_stats.requested_points == 4
        assert source.evaluation_stats.batches == 2
        source.reset_evaluation_stats()
        assert source.evaluation_stats.requested_points == 0

    @pytest.mark.parametrize('kwargs, error', [
        ({}, ValueError),
        ({'base': 2, 'domain': torch.tensor([0., 1.])}, ValueError),
        ({'base': 2, 'level': 2}, ValueError),
        ({'base': 2, 'level': 2, 'domain': torch.tensor([0., 1.]),
          'grid_coordinates': (torch.arange(4.),)}, ValueError),
        ({'base': 2, 'level': 2,
          'grid_coordinates': (torch.arange(3.),)}, ValueError),
        ({'layout': tk.formats.QuantizedLayout(1, 2, 2)}, TypeError),
        ({'layout': tk.formats.QuantizedLayout(1, 2, 2),
          'coordinate_map': tk.formats.AffineCoordinateMap([0., 1.], (3,))},
         ValueError),
        ({'base': 2, 'level': 2,
          'layout': tk.formats.QuantizedLayout(1, 2, 2),
          'coordinate_map': tk.formats.AffineCoordinateMap([0., 1.], (4,))},
         ValueError),
    ])
    def test_invalid_construction(self, kwargs, error):
        with pytest.raises(error):
            tk.decompositions.QuanticsVectorSource(lambda x: x[:, 0], 1, **kwargs)

    def test_tensor_sources_are_not_quantized(self):
        tensor_source = tk.decompositions.DenseTensorSource(torch.ones(4))
        for function in (torch.ones(4), tensor_source,
                         tk.formats.TT([torch.ones(4)])):
            with pytest.raises(TypeError, match='coordinate callable'):
                tk.decompositions.QuanticsVectorSource(
                    function, 1, base=2, level=2, domain=torch.tensor([0., 1.]))

    @pytest.mark.parametrize('digits, error', [
        (torch.tensor([[0, 2]]), ValueError),
        (torch.tensor([[-1, 0]]), ValueError),
        (torch.tensor([[0, 0, 0]]), ValueError),
        (torch.tensor([[0., 1.]]), TypeError),
    ])
    def test_invalid_digit_configurations(self, digits, error):
        source = tk.decompositions.QuanticsVectorSource(
            lambda x: x[:, 0], 1, base=2, level=2, domain=torch.tensor([0., 1.]))
        with pytest.raises(error):
            source.evaluate_digits(digits)

    def test_invalid_evaluation_contracts(self):
        source = tk.decompositions.QuanticsVectorSource(
            lambda x: x[:, 0], 1, base=2, level=2, domain=torch.tensor([0., 1.]))
        with pytest.raises(TypeError, match='ConfigurationBatch'):
            source.evaluate(torch.zeros(1, 2, dtype=torch.long))
        with pytest.raises(ValueError, match='digit index'):
            source.evaluate(tk.decompositions.ConfigurationBatch(
                torch.zeros(1, 2), kind='features'))
        with pytest.raises(ValueError, match='unit_coordinates'):
            source.evaluate_coordinates(torch.tensor([[2.]]))
        for function, error in (
                (lambda x: torch.ones(x.shape[0], 2), ValueError),
                (lambda x: torch.ones(x.shape[0], dtype=torch.long), TypeError),
                (lambda x: [1.], TypeError)):
            other = tk.decompositions.QuanticsVectorSource(
                function, 1, base=2, level=2, domain=torch.tensor([0., 1.]))
            with pytest.raises(error):
                other.to_dense_digits()
        with pytest.raises(ValueError, match='dtype'):
            tk.decompositions.QuanticsVectorSource(
                lambda x: x[:, 0], 1, base=2, level=2,
                domain=torch.tensor([0., 1.]), dtype=torch.float64).to_dense_digits()


class TestQuanticsMatrixSource:  # MARK: TestQuanticsMatrixSource

    @pytest.mark.parametrize('in_path', ['uniform', 'explicit', 'objects'])
    @pytest.mark.parametrize('out_path', ['uniform', 'explicit', 'objects'])
    def test_independent_construction_paths(self, in_path, out_path,
                                            device_dtype, assert_close):
        device, dtype = device_dtype
        real_dtype = torch.empty((), dtype=dtype).real.dtype

        def parameters(side, path):
            if path == 'uniform':
                return {side + '_base': 2, side + '_level': 2,
                        side + '_domain': torch.tensor([0., 1.], dtype=real_dtype,
                                                       device=device)}
            if path == 'explicit':
                return {side + '_base': 2, side + '_level': 2,
                        side + '_grid_coordinates': (torch.tensor(
                            [-2., 0., 1., 3.], dtype=real_dtype, device=device),)}
            layout = tk.formats.QuantizedLayout(1, 2, 2, digit_order='fine_to_coarse')
            return {side + '_layout': layout,
                    side + '_coordinate_map': tk.formats.AffineCoordinateMap(
                        torch.tensor([1., 2.], dtype=real_dtype, device=device),
                        layout.grid_size)}

        def function(inputs, outputs):
            values = (inputs[:, 0] + 2 * outputs[:, 0]).to(dtype)
            return values * (1 + 1j) if dtype.is_complex else values

        source = tk.decompositions.QuanticsMatrixSource(
            function, 1, 1, **parameters('in', in_path),
            **parameters('out', out_path), dtype=dtype, device=device, batch_size=3)
        in_indices, out_indices = torch.cartesian_prod(
            torch.arange(4, device=device), torch.arange(4, device=device)).split(1, -1)
        inputs = source.in_coordinate_map.from_indices(in_indices)
        outputs = source.out_coordinate_map.from_indices(out_indices)
        expected = function(inputs, outputs)
        in_digits = source.in_layout.encode_indices(in_indices)
        out_digits = source.out_layout.encode_indices(out_indices)
        assert_close(source.evaluate_digits(in_digits, out_digits), expected)
        assert_close(source.evaluate_indices(in_indices, out_indices), expected)
        assert_close(source.evaluate_coordinates(inputs, outputs), expected)
        actual_inputs, actual_outputs = source.digits_to_coordinates(
            in_digits, out_digits)
        assert_close(actual_inputs, inputs)
        assert_close(actual_outputs, outputs)
        actual_in_digits, actual_out_digits = source.coordinates_to_digits(
            inputs, outputs)
        assert_close(actual_in_digits, in_digits)
        assert_close(actual_out_digits, out_digits)
        assert_close(source.to_dense_grid(), expected.reshape(4, 4))
        dense = source.to_dense_digits()
        paired = torch.stack((in_digits, out_digits), dim=-1).flatten(-2)
        assert_close(dense[tuple(paired.T)], expected)
        assert source.dtype == dtype and source.device.type == device

    def test_mismatched_site_counts_and_batches(self):
        with pytest.raises(ValueError, match='n_sites'):
            tk.decompositions.QuanticsMatrixSource(
                lambda x, y: x[:, 0], 1, 1, in_base=2, out_base=2,
                in_level=2, out_level=3, in_domain=[0., 1.], out_domain=[0., 1.])
        source = tk.decompositions.QuanticsMatrixSource(
            lambda x, y: x[:, 0] + y[:, 0], 1, 1, in_base=2, out_base=2,
            in_level=2, out_level=2, in_domain=[0., 1.], out_domain=[0., 1.])
        with pytest.raises(ValueError, match='matching batches'):
            source.evaluate_digits(torch.zeros(2, 2, dtype=torch.long),
                                   torch.zeros(1, 2, dtype=torch.long))
