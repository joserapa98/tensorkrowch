"""Tests for multivariable quantized layouts and coordinate maps."""

import pytest

import torch
import tensorkrowch as tk


class TestQuantizedLayout:  # MARK: TestQuantizedLayout

    def test_grid_inference_and_uniform_expansion(self):
        with pytest.warns(UserWarning, match=r'\(3,\) to \(4,\)'):
            by_base = tk.formats.QuantizedLayout.from_grid(1, 3, base=2)
        with pytest.warns(UserWarning, match=r'\(5,\) to \(9,\)'):
            by_level = tk.formats.QuantizedLayout.from_grid(1, 5, level=2)

        assert by_base.base == (2,)
        assert by_base.level == (2,)
        assert by_base.grid_size == (4,)
        assert by_level.base == (3,)
        assert by_level.level == (2,)
        assert by_level.grid_size == (9,)

    def test_grid_requires_resolution_and_exact_explicit_size(self):
        grid = tk.formats.ExplicitGridMap(torch.tensor([0., 1., 4., 10.]))
        with pytest.raises(ValueError, match='base.*level'):
            tk.formats.QuantizedLayout.from_grid(1, 4)
        with pytest.raises(ValueError, match='match'):
            tk.formats.QuantizedLayout.from_grid(1, 4, base=2, level=3)
        with pytest.raises(ValueError, match='explicit grid'):
            tk.formats.QuantizedLayout.from_grid(1, grid, base=2, level=3)
        with pytest.raises(ValueError, match='explicit grid'):
            tk.formats.QuantizedLayout.from_grid(
                1, tk.formats.ExplicitGridMap(torch.tensor([0., 1., 4.])),
                base=2)

        layout = tk.formats.QuantizedLayout.from_grid(1, grid, base=2)
        assert layout.level == (2,)
        assert layout.grid_size == (4,)


    @pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
    @pytest.mark.parametrize(
        'digit_order', ['coarse_to_fine', 'fine_to_coarse'])
    def test_multivariable_roundtrip(self, ordering, digit_order):
        layout = tk.formats.QuantizedLayout(
            n_coordinates=3,
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
        assert layout.in_dim == tuple(
            layout.base[variable] for variable, _ in layout.sites())

    def test_standard_xyz_schedules_with_unequal_levels(self):
        grouped = tk.formats.QuantizedLayout(
            3, level=(3, 2, 1), ordering='grouped')
        interleaved = tk.formats.QuantizedLayout(
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
        grouped = tk.formats.QuantizedLayout(
            2, level=(3, 2), digit_order='fine_to_coarse')
        interleaved = tk.formats.QuantizedLayout(
            2,
            level=(3, 2),
            ordering='interleaved',
            digit_order='fine_to_coarse')

        assert grouped.sites() == (
            (0, 2), (0, 1), (0, 0), (1, 1), (1, 0))
        assert interleaved.sites() == (
            (0, 2), (1, 1), (0, 1), (1, 0), (0, 0))

    def test_custom_permutation_and_reordering_are_reversible(self):
        grouped = tk.formats.QuantizedLayout(
            2, base=2, level=(2, 3))
        custom = tk.formats.QuantizedLayout(
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
            ({'n_coordinates': 0}, ValueError, 'positive'),
            ({'n_coordinates': 2, 'base': (2,)}, ValueError, 'one value'),
            ({'n_coordinates': 1, 'base': 1}, ValueError, 'at least two'),
            ({'n_coordinates': 1, 'level': 0}, ValueError, 'positive'),
            ({'n_coordinates': 1, 'ordering': 'custom'}, ValueError,
             'permutation'),
            ({'n_coordinates': 1, 'level': 2, 'ordering': 'custom',
              'permutation': ((0, 0), (0, 0))}, ValueError, 'every'),
            ({'n_coordinates': 1, 'base': 2, 'level': 63}, OverflowError,
             'int64'),
        ])
    def test_invalid_layouts_are_rejected(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            tk.formats.QuantizedLayout(**kwargs)

    def test_invalid_shapes_and_bounds_are_rejected(self):
        layout = tk.formats.QuantizedLayout(2, level=2)

        with pytest.raises(ValueError, match='n_coordinates'):
            layout.encode_indices(torch.tensor([[0, 1, 2]]))
        with pytest.raises(ValueError, match='out of bounds'):
            layout.encode_indices(torch.tensor([[4, 0]]))
        with pytest.raises(ValueError, match='layout sites'):
            layout.decode_digits(torch.tensor([[0, 1]]))
        with pytest.raises(ValueError, match='out of bounds'):
            layout.decode_digits(torch.tensor([[0, 1, 2, 0]]))


class TestUniformCoordinateMap:  # MARK: TestUniformCoordinateMap

    @pytest.mark.parametrize('grid, offset', [
        ('left', 0.), ('centers', 0.5), ('right', 1.),
        (0., 0.), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1., 1.),
    ])
    def test_cell_positions_and_roundtrip(self, grid, offset):
        coordinate_map = tk.formats.UniformCoordinateMap(grid=grid)
        indices = torch.arange(4).unsqueeze(-1)
        domain = torch.tensor([-2., 2.], dtype=torch.float64)

        coordinates = coordinate_map.from_indices(indices, (4,), domain)

        assert torch.equal(coordinates, indices.to(torch.float64) + offset - 2)
        assert torch.equal(
            coordinate_map.to_indices(coordinates, (4,), domain), indices)

    @pytest.mark.parametrize('grid, expected', [
        ('left', [0, 1, 1, 3, 3]), (0., [0, 1, 1, 3, 3]),
        ('right', [0, 0, 1, 2, 3]), (1., [0, 0, 1, 2, 3]),
        ('centers', [0, 0, 1, 2, 3]), (0.25, [0, 1, 1, 3, 3]),
    ])
    def test_quantization_at_boundaries_and_inside_cells(self, grid, expected):
        coordinate_map = tk.formats.UniformCoordinateMap(grid=grid)
        coordinates = torch.tensor([[0.], [0.25], [0.375], [0.75], [1.]])

        indices = coordinate_map.to_indices(
            coordinates, (4,), domain=torch.tensor([0., 1.]))

        assert indices.squeeze(-1).tolist() == expected

    def test_left_grid_matches_discretize(self):
        layout = tk.formats.QuantizedLayout(1, base=2, level=3)
        coordinate_map = tk.formats.UniformCoordinateMap(grid='left')
        coordinates = torch.tensor([[0.], [0.1], [0.125], [0.9], [1.]])
        indices = coordinate_map.to_indices(
            coordinates, layout.grid_size, torch.tensor([0., 1.]))

        assert torch.equal(
            layout.encode_indices(indices),
            tk.embeddings.discretize(coordinates, level=3).squeeze(-2).long())

    @pytest.mark.parametrize('grid, error', [
        ('cell_centers', ValueError), ('unknown', ValueError),
        (-0.1, ValueError), (1.1, ValueError),
        (float('nan'), ValueError), (float('inf'), ValueError),
        (True, TypeError), (None, TypeError), ([0.5], TypeError),
    ])
    def test_invalid_grid_conventions_are_rejected(self, grid, error):
        with pytest.raises(error, match='grid'):
            tk.formats.UniformCoordinateMap(grid=grid)

    @pytest.mark.parametrize('grid', ['endpoints', 'centers'])
    def test_index_physical_roundtrip_with_per_variable_domains(self, grid):
        coordinate_map = tk.formats.UniformCoordinateMap(grid=grid)
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
        coordinate_map = tk.formats.UniformCoordinateMap()
        physical = torch.tensor([[0.125], [0.375], [0.625], [0.875]])

        indices = coordinate_map.to_indices(
            physical, grid_size=(5,), domain=torch.tensor([0., 1.]))

        assert torch.equal(indices.squeeze(1), torch.tensor([0, 1, 2, 3]))

    def test_out_of_domain_requires_explicit_clip(self):
        coordinate_map = tk.formats.UniformCoordinateMap()
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
        coordinate_map = tk.formats.UniformCoordinateMap()
        unit = torch.tensor([[0., 0.5, 1.]])

        physical = coordinate_map.forward(
            unit, domain=torch.tensor([-2., 2.]))

        assert torch.equal(physical, torch.tensor([[-2., 0., 2.]]))
        assert isinstance(coordinate_map, tk.formats.CoordinateMap)


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

        coordinate_map = tk.formats.WarpedCoordinateMap(
            forward, inverse)
        unit = torch.tensor([[0., 0.2], [0.5, 0.8], [1., 0.4]])

        physical = coordinate_map.forward(unit)

        assert torch.allclose(coordinate_map.inverse(physical), unit)

    def test_missing_inverse_is_explicit(self):
        coordinate_map = tk.formats.WarpedCoordinateMap(
            lambda unit, domain: unit.square())

        with pytest.raises(NotImplementedError, match='inverse'):
            coordinate_map.inverse(torch.tensor([[0.5]]))


class TestExplicitGridMap:  # MARK: TestExplicitGridMap

    def test_arbitrary_grids_map_exact_indices_and_nearest_points(self):
        coordinate_map = tk.formats.ExplicitGridMap((
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
        coordinate_map = tk.formats.ExplicitGridMap(
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
            tk.formats.ExplicitGridMap(
                torch.tensor([0., 2., 1.]))
