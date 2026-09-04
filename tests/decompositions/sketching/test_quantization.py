"""Tests for multivariable quantized layouts and coordinate maps."""

import pytest

import torch
import tensorkrowch as tk


class TestQuantizedLayout:  # MARK: TestQuantizedLayout

    @pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
    @pytest.mark.parametrize(
        'digit_order', ['coarse_to_fine', 'fine_to_coarse'])
    def test_multivariable_roundtrip(self, ordering, digit_order):
        layout = tk.decompositions.QuantizedLayout(
            n_variables=3,
            base=(2, 3, 2),
            level=(3, 2, 1),
            ordering=ordering,
            digit_order=digit_order)
        indices = torch.tensor([
            [0, 0, 0], [7, 8, 1], [3, 5, 0], [6, 2, 1]])

        digits = layout.encode_indices(indices)

        assert digits.shape == (4, 6)
        assert torch.equal(layout.decode_digits(digits), indices)
        assert layout.grid_size == (8, 9, 2)
        assert layout.input_dim == tuple(
            layout.base[variable] for variable, _ in layout.sites())

    def test_standard_xyz_schedules_with_unequal_levels(self):
        grouped = tk.decompositions.QuantizedLayout(
            3, level=(3, 2, 1), ordering='grouped')
        interleaved = tk.decompositions.QuantizedLayout(
            3, level=(3, 2, 1), ordering='interleaved')

        assert grouped.sites() == (
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1),
            (2, 0))
        assert interleaved.sites() == (
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1),
            (0, 2))

    def test_fine_to_coarse_reverses_each_variable_schedule(self):
        grouped = tk.decompositions.QuantizedLayout(
            2, level=(3, 2), digit_order='fine_to_coarse')
        interleaved = tk.decompositions.QuantizedLayout(
            2,
            level=(3, 2),
            ordering='interleaved',
            digit_order='fine_to_coarse')

        assert grouped.sites() == (
            (0, 2), (0, 1), (0, 0), (1, 1), (1, 0))
        assert interleaved.sites() == (
            (0, 2), (1, 1), (0, 1), (1, 0), (0, 0))

    def test_custom_permutation_and_reordering_are_reversible(self):
        grouped = tk.decompositions.QuantizedLayout(
            2, base=2, level=(2, 3))
        custom = tk.decompositions.QuantizedLayout(
            2,
            base=2,
            level=(2, 3),
            ordering='custom',
            permutation=((1, 2), (0, 0), (1, 0), (0, 1), (1, 1)))
        indices = torch.tensor([[0, 0], [3, 7], [2, 5]])
        grouped_digits = grouped.encode_indices(indices)

        custom_digits = grouped.reorder_configurations(
            grouped_digits, custom)
        restored = custom.reorder_configurations(
            custom_digits, grouped)

        assert torch.equal(custom.decode_digits(custom_digits), indices)
        assert torch.equal(restored, grouped_digits)

    @pytest.mark.parametrize(
        'kwargs, error, match',
        [
            ({'n_variables': 0}, ValueError, 'positive'),
            ({'n_variables': 2, 'base': (2,)}, ValueError, 'one value'),
            ({'n_variables': 1, 'base': 1}, ValueError, 'at least two'),
            ({'n_variables': 1, 'level': 0}, ValueError, 'positive'),
            ({'n_variables': 1, 'ordering': 'custom'}, ValueError,
             'permutation'),
            ({'n_variables': 1, 'level': 2, 'ordering': 'custom',
              'permutation': ((0, 0), (0, 0))}, ValueError, 'every'),
            ({'n_variables': 1, 'base': 2, 'level': 63}, OverflowError,
             'int64'),
        ])
    def test_invalid_layouts_are_rejected(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            tk.decompositions.QuantizedLayout(**kwargs)

    def test_invalid_shapes_and_bounds_are_rejected(self):
        layout = tk.decompositions.QuantizedLayout(2, level=2)

        with pytest.raises(ValueError, match='n_variables'):
            layout.encode_indices(torch.tensor([[0, 1, 2]]))
        with pytest.raises(ValueError, match='out of bounds'):
            layout.encode_indices(torch.tensor([[4, 0]]))
        with pytest.raises(ValueError, match='layout sites'):
            layout.decode_digits(torch.tensor([[0, 1]]))
        with pytest.raises(ValueError, match='out of bounds'):
            layout.decode_digits(torch.tensor([[0, 1, 2, 0]]))


class TestUniformCoordinateMap:  # MARK: TestUniformCoordinateMap

    @pytest.mark.parametrize('grid', ['endpoints', 'cell_centers'])
    def test_index_physical_roundtrip_with_per_variable_domains(self, grid):
        coordinate_map = tk.decompositions.UniformCoordinateMap(grid=grid)
        grid_size = (5, 4)
        domain = torch.tensor(
            [[-2., 2.], [10., 16.]], dtype=torch.float64)
        indices = torch.tensor([[0, 0], [2, 1], [4, 3]])

        physical = coordinate_map.from_indices(indices, grid_size, domain)
        restored = coordinate_map.to_indices(
            physical, grid_size, domain)

        assert torch.equal(restored, indices)
        if grid == 'endpoints':
            assert torch.allclose(
                physical[[0, -1]], domain[:, [0, 1]].T)
        else:
            assert torch.all(physical[0] > domain[:, 0])
            assert torch.all(physical[-1] < domain[:, 1])

    def test_nearest_ties_choose_lower_index(self):
        coordinate_map = tk.decompositions.UniformCoordinateMap()
        physical = torch.tensor([[0.125], [0.375], [0.625], [0.875]])

        indices = coordinate_map.to_indices(
            physical, grid_size=(5,), domain=torch.tensor([0., 1.]))

        assert torch.equal(indices.squeeze(1), torch.tensor([0, 1, 2, 3]))

    def test_out_of_domain_requires_explicit_clip(self):
        coordinate_map = tk.decompositions.UniformCoordinateMap()
        physical = torch.tensor([[-0.1], [1.2]])

        with pytest.raises(ValueError, match='outside'):
            coordinate_map.to_indices(
                physical, (4,), domain=torch.tensor([0., 1.]))
        clipped = coordinate_map.to_indices(
            physical,
            (4,),
            domain=torch.tensor([0., 1.]),
            out_of_domain='clip')

        assert torch.equal(clipped.squeeze(1), torch.tensor([0, 3]))

    def test_shared_domain_broadcasts_to_all_variables(self):
        coordinate_map = tk.decompositions.UniformCoordinateMap()
        unit = torch.tensor([[0., 0.5, 1.]])

        physical = coordinate_map.forward(
            unit, domain=torch.tensor([-2., 2.]))

        assert torch.equal(physical, torch.tensor([[-2., 0., 2.]]))
        assert isinstance(coordinate_map, tk.decompositions.CoordinateMap)


class TestWarpedCoordinateMap:  # MARK: TestWarpedCoordinateMap

    def test_coupled_forward_and_inverse_need_no_domain(self):
        def forward(unit, domain):
            assert domain is None
            return torch.stack((
                unit[..., 0],
                unit[..., 1] * (1 + unit[..., 0])), dim=-1)

        def inverse(physical, domain):
            assert domain is None
            return torch.stack((
                physical[..., 0],
                physical[..., 1] / (1 + physical[..., 0])), dim=-1)

        coordinate_map = tk.decompositions.WarpedCoordinateMap(
            forward, inverse)
        unit = torch.tensor([[0., 0.2], [0.5, 0.8], [1., 0.4]])

        physical = coordinate_map.forward(unit)

        assert torch.allclose(coordinate_map.inverse(physical), unit)

    def test_missing_inverse_is_explicit(self):
        coordinate_map = tk.decompositions.WarpedCoordinateMap(
            lambda unit, domain: unit.square())

        with pytest.raises(NotImplementedError, match='inverse'):
            coordinate_map.inverse(torch.tensor([[0.5]]))


class TestExplicitGridMap:  # MARK: TestExplicitGridMap

    def test_arbitrary_grids_map_exact_indices_and_nearest_points(self):
        coordinate_map = tk.decompositions.ExplicitGridMap((
            torch.tensor([0., 1., 4., 10.]),
            torch.tensor([3., 1., -2.]),
        ))
        indices = torch.tensor([[0, 0], [2, 1], [3, 2]])

        physical = coordinate_map.from_indices(indices)
        restored = coordinate_map.to_indices(physical)
        tie = coordinate_map.to_indices(torch.tensor([[2.5, -0.5]]))

        assert torch.equal(restored, indices)
        assert torch.equal(
            physical,
            torch.tensor([[0., 3.], [4., 1.], [10., -2.]]))
        assert torch.equal(tie, torch.tensor([[1, 1]]))

    def test_shared_grid_broadcast_and_clip_are_explicit(self):
        coordinate_map = tk.decompositions.ExplicitGridMap(
            torch.tensor([0., 2., 5.]))
        indices = torch.tensor([[0, 2], [1, 1]])

        assert torch.equal(
            coordinate_map.from_indices(indices),
            torch.tensor([[0., 5.], [2., 2.]]))
        with pytest.raises(ValueError, match='outside'):
            coordinate_map.to_indices(torch.tensor([[-1., 6.]]))
        assert torch.equal(
            coordinate_map.to_indices(
                torch.tensor([[-1., 6.]]), out_of_domain='clip'),
            torch.tensor([[0, 2]]))

    def test_nonmonotonic_grid_is_rejected(self):
        with pytest.raises(ValueError, match='monotonic'):
            tk.decompositions.ExplicitGridMap(
                torch.tensor([0., 2., 1.]))


class TestQuantizedSourceAdapter:  # MARK: TestQuantizedSourceAdapter

    @pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
    def test_physical_callable_decodes_each_layout(self, ordering):
        layout = tk.decompositions.QuantizedLayout(
            3, base=2, level=2, ordering=ordering)
        coordinate_map = tk.decompositions.UniformCoordinateMap()
        domain = torch.tensor([[0., 1.], [-1., 1.], [2., 4.]])

        def function(physical):
            value = physical[:, 0] + \
                2 * physical[:, 1] + 3 * physical[:, 2]
            return torch.stack((value, value.square()), dim=1)

        adapter = tk.decompositions.QuantizedSourceAdapter(
            function,
            layout,
            coordinate_map,
            domain,
            output_shape=(2,),
            dtype=torch.float32)
        indices = torch.tensor([[0, 0, 0], [1, 2, 3], [3, 1, 2]])
        digits = layout.encode_indices(indices)

        result = adapter.evaluate(
            tk.decompositions.ConfigurationBatch(digits))
        physical = coordinate_map.from_indices(
            indices, layout.grid_size, domain)

        assert torch.equal(result, function(physical))
        assert adapter.output_shape == (2,)

    def test_indexed_tensor_source_uses_decoded_variable_indices(self):
        layout = tk.decompositions.QuantizedLayout(
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
        layout = tk.decompositions.QuantizedLayout(2, base=2, level=2)
        maps = (
            tk.decompositions.UniformCoordinateMap(),
            tk.decompositions.WarpedCoordinateMap(
                lambda unit, domain: unit.square(),
                lambda physical, domain: physical.sqrt()),
        )
        domain = (torch.tensor([-1., 1.]), None)
        adapter = tk.decompositions.QuantizedSourceAdapter(
            lambda physical: physical.sum(dim=1),
            layout,
            coordinate_map=maps,
            domain=domain,
            dtype=torch.float32)
        physical = torch.tensor([[-1., 0.], [1., 1.]])

        digits = adapter.physical_to_digits(physical)

        assert torch.allclose(adapter.digits_to_physical(digits), physical)

    def test_digit_tt_bypass_requires_and_respects_layout_metadata(self):
        grouped = tk.decompositions.QuantizedLayout(
            2, base=2, level=2, ordering='grouped')
        interleaved = tk.decompositions.QuantizedLayout(
            2, base=2, level=2, ordering='interleaved')
        variable_indices = torch.cartesian_prod(
            torch.arange(4), torch.arange(4))
        grouped_digits = grouped.encode_indices(variable_indices)
        values = (variable_indices[:, 0] +
                  10 * variable_indices[:, 1]).to(torch.float64)
        dense = torch.empty(grouped.input_dim, dtype=torch.float64)
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
        layout = tk.decompositions.QuantizedLayout(1, base=3, level=1)
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
        layout = tk.decompositions.QuantizedLayout(1, base=3, level=1)
        dataset = torch.tensor([[0.1], [0.2], [0.9], [0.9]])
        adapter = tk.decompositions.QuantizedSourceAdapter.from_physical_dataset(
            dataset,
            layout,
            domain=torch.tensor([0., 1.]))

        result = adapter.evaluate(tk.decompositions.ConfigurationBatch(
            torch.tensor([[0], [1], [2]])))

        assert torch.equal(result, torch.tensor([0.5, 0., 0.5]))


__all__ = []
