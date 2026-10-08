"""Tests for multicoordinate quantized layouts and coordinate maps."""

import pytest

import torch
import tensorkrowch as tk


class TestQuantizedLayout:  # MARK: TestQuantizedLayout

    def test_defaults_and_no_grid_inference(self):
        layout = tk.formats.QuantizedLayout(2, base=(2, 3), level=(2, 1))
        assert layout.ordering == 'interleaved'
        assert layout.digit_order == 'coarse_to_fine'
        assert layout.grid_size == (4, 3)
        assert not hasattr(type(layout), 'from_grid')

    @pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
    @pytest.mark.parametrize(
        'digit_order', ['coarse_to_fine', 'fine_to_coarse'])
    def test_multicoordinate_roundtrip(self, ordering, digit_order):
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
            layout.base[coordinate] for coordinate, _ in layout.sites())

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

    def test_fine_to_coarse_reverses_each_coordinate_schedule(self):
        grouped = tk.formats.QuantizedLayout(
            2, level=(3, 2), ordering='grouped', digit_order='fine_to_coarse')
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


class TestAffineCoordinateMap:  # MARK: TestAffineCoordinateMap

    @pytest.mark.parametrize('grid, offset', [
        ('left', 0.), ('centers', 0.5), ('right', 1.),
        (0., 0.), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1., 1.),
    ])
    def test_cell_positions_and_roundtrip(self, grid, offset):
        coordinate_map = tk.formats.AffineCoordinateMap([-2., 2.], (4,), grid_offset=grid)
        indices = torch.arange(4).unsqueeze(-1)

        coordinates = coordinate_map.from_indices(indices)

        assert torch.equal(coordinates, indices.to(torch.float64) + offset - 2)
        assert torch.equal(
            coordinate_map.to_indices(coordinates), indices)

    @pytest.mark.parametrize('grid, expected', [
        ('left', [0, 1, 1, 3, 3]), (0., [0, 1, 1, 3, 3]),
        ('right', [0, 0, 1, 2, 3]), (1., [0, 0, 1, 2, 3]),
        ('centers', [0, 0, 1, 2, 3]), (0.25, [0, 1, 1, 3, 3]),
    ])
    def test_quantization_at_boundaries_and_inside_cells(self, grid, expected):
        coordinate_map = tk.formats.AffineCoordinateMap([0., 1.], (4,), grid_offset=grid)
        coordinates = torch.tensor([[0.], [0.25], [0.375], [0.75], [1.]])

        indices = coordinate_map.to_indices(coordinates)

        assert indices.squeeze(-1).tolist() == expected

    def test_left_grid_matches_discretize(self):
        layout = tk.formats.QuantizedLayout(1, base=2, level=3)
        coordinate_map = tk.formats.AffineCoordinateMap([0., 1.], layout.grid_size)
        coordinates = torch.tensor([[0.], [0.1], [0.125], [0.9], [1.]])
        indices = coordinate_map.to_indices(coordinates)

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
            tk.formats.AffineCoordinateMap([0., 1.], (4,), grid_offset=grid)

    @pytest.mark.parametrize('grid', ['endpoints', 'centers'])
    def test_index_domain_coordinates_roundtrip_with_per_coordinate_domains(self, grid):
        grid_size = (5, 4)
        domain = torch.tensor(
            [[-2., 2.], [10., 16.]], dtype=torch.float64)
        indices = torch.tensor([[0, 0], [2, 1], [4, 3]])

        coordinate_map = tk.formats.AffineCoordinateMap(
            domain, grid_size, grid_offset=grid)
        domain_coordinates = coordinate_map.from_indices(indices)
        restored = coordinate_map.to_indices(domain_coordinates)

        assert torch.equal(restored, indices)
        if grid == 'endpoints':
            assert torch.allclose(
                domain_coordinates[[0, -1]], domain[:, [0, 1]].T)
        else:
            assert torch.all(domain_coordinates[0] > domain[:, 0])
            assert torch.all(domain_coordinates[-1] < domain[:, 1])

    def test_nearest_ties_choose_lower_index(self):
        coordinate_map = tk.formats.AffineCoordinateMap([0., 1.], (5,), grid_offset='endpoints')
        domain_coordinates = torch.tensor([[0.125], [0.375], [0.625], [0.875]])

        indices = coordinate_map.to_indices(domain_coordinates)

        assert torch.equal(indices.squeeze(1), torch.tensor([0, 1, 2, 3]))

    def test_out_of_domain_requires_explicit_clip(self):
        coordinate_map = tk.formats.AffineCoordinateMap([0., 1.], (4,))
        domain_coordinates = torch.tensor([[-0.1], [1.2]])

        with pytest.raises(ValueError, match='outside|lie in'):
            coordinate_map.to_indices(domain_coordinates)
        clipped = tk.formats.AffineCoordinateMap(
            [0., 1.], (4,), out_of_domain='clip').to_indices(domain_coordinates)

        assert torch.equal(clipped.squeeze(1), torch.tensor([0, 3]))

    def test_shared_domain_broadcasts_to_all_coordinates(self):
        coordinate_map = tk.formats.AffineCoordinateMap([-2., 2.], (4, 4, 4))
        unit = torch.tensor([[0., 0.5, 1.]])

        domain_coordinates = coordinate_map.forward(unit)

        assert torch.equal(domain_coordinates, torch.tensor([[-2., 0., 2.]]))
        assert isinstance(coordinate_map, tk.formats.CoordinateMap)


class TestFunctionalCoordinateMap:  # MARK: TestFunctionalCoordinateMap

    def test_coupled_forward_and_inverse_need_no_domain(self):
        def forward(unit, domain):
            assert domain is None
            return torch.stack((
                unit[..., 0],
                unit[..., 1] * (1 + unit[..., 0])), dim=-1)

        def inverse(domain_coordinates, domain):
            assert domain is None
            return torch.stack((
                domain_coordinates[..., 0],
                domain_coordinates[..., 1] / (1 + domain_coordinates[..., 0])), dim=-1)

        coordinate_map = tk.formats.FunctionalCoordinateMap(
            grid_size=(4, 4), forward_function=forward, inverse_function=inverse)
        unit = torch.tensor([[0., 0.2], [0.5, 0.8], [1., 0.4]])

        domain_coordinates = coordinate_map.forward(unit)

        assert torch.allclose(coordinate_map.inverse(domain_coordinates), unit)

    def test_missing_inverse_is_explicit(self):
        coordinate_map = tk.formats.FunctionalCoordinateMap(
            grid_size=(4,), forward_function=lambda unit, domain: unit.square())

        with pytest.raises(NotImplementedError, match='inverse'):
            coordinate_map.inverse(torch.tensor([[0.5]]))


class TestExplicitGridMap:  # MARK: TestExplicitGridMap

    def test_arbitrary_grids_map_exact_indices_and_nearest_points(self):
        coordinate_map = tk.formats.ExplicitGridMap((
            torch.tensor([0., 1., 4., 10.]),
            torch.tensor([3., 1., -2.]),
        ))
        indices = torch.tensor([[0, 0], [2, 1], [3, 2]])

        domain_coordinates = coordinate_map.from_indices(indices)
        restored = coordinate_map.to_indices(domain_coordinates)
        tie = coordinate_map.to_indices(torch.tensor([[2.5, -0.5]]))

        assert torch.equal(restored, indices)
        assert torch.equal(
            domain_coordinates,
            torch.tensor([[0., 3.], [4., 1.], [10., -2.]]))
        assert torch.equal(tie, torch.tensor([[1, 1]]))

    def test_shared_grid_broadcast_and_clip_are_explicit(self):
        coordinate_map = tk.formats.ExplicitGridMap(
            torch.tensor([[0., 2., 5.], [0., 2., 5.]]))
        indices = torch.tensor([[0, 2], [1, 1]])

        assert torch.equal(
            coordinate_map.from_indices(indices),
            torch.tensor([[0., 5.], [2., 2.]]))
        with pytest.raises(ValueError, match='outside'):
            coordinate_map.to_indices(torch.tensor([[-1., 6.]]))
        assert torch.equal(
            tk.formats.ExplicitGridMap(
                coordinate_map.grid_coordinates, out_of_domain='clip').to_indices(
                    torch.tensor([[-1., 6.]])),
            torch.tensor([[0, 2]]))

    def test_nonmonotonic_grid_is_rejected(self):
        with pytest.raises(ValueError, match='monotonic'):
            tk.formats.ExplicitGridMap(
                torch.tensor([0., 2., 1.]))


# Complete integer layouts and coordinate transformations


@pytest.mark.parametrize('ordering', ['grouped', 'interleaved', 'custom'])
@pytest.mark.parametrize('digit_order', ['coarse_to_fine', 'fine_to_coarse'])
@pytest.mark.parametrize('dtype', [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_all_grid_indices_and_site_order(ordering, digit_order, dtype):
    permutation = ((1, 1), (0, 0), (1, 0), (0, 1)) if ordering == 'custom' else None
    layout = tk.formats.QuantizedLayout(2, base=(2, 3), level=2,
                                       ordering=ordering, digit_order=digit_order,
                                       permutation=permutation)
    indices = torch.cartesian_prod(torch.arange(4), torch.arange(9)).to(dtype)
    digits = layout.encode_indices(indices.reshape(2, 3, 6, 2))
    assert digits.dtype == torch.long and digits.shape == (2, 3, 6, 4)
    assert torch.equal(layout.decode_digits(digits).reshape(-1, 2), indices.long())
    for site, (coordinate, digit) in enumerate(layout.sites()):
        stride = layout.base[coordinate] ** (layout.level[coordinate] - digit - 1)
        expected = indices[:, coordinate].long() // stride % layout.base[coordinate]
        assert torch.equal(digits.reshape(-1, 4)[:, site], expected)
    for target in ('grouped', 'interleaved'):
        target_layout = tk.formats.QuantizedLayout(2, base=(2, 3), level=2,
                                                  ordering=target,
                                                  digit_order=digit_order)
        reordered = layout.reorder_configurations(digits, target)
        assert torch.equal(reordered, target_layout.encode_indices(indices.reshape(2, 3, 6, 2)))


@pytest.mark.parametrize('options,error', [
    ({'n_coordinates': True}, TypeError), ({'n_coordinates': 1.5}, TypeError),
    ({'base': True}, TypeError), ({'base': 2.5}, TypeError),
    ({'base': '2'}, TypeError), ({'level': [1, 1]}, ValueError),
    ({'level': False}, TypeError), ({'ordering': 'diagonal'}, ValueError),
    ({'digit_order': 'coarse_to_grain'}, ValueError),
    ({'permutation': ((0, 0),)}, ValueError),
    ({'ordering': 'custom', 'permutation': 1}, TypeError),
    ({'ordering': 'custom', 'permutation': ((False, 0),)}, TypeError),
    ({'ordering': 'custom', 'permutation': ((1, 0),)}, ValueError),
])
def test_layout_constructor_type_and_order_errors(options, error):
    arguments = {'n_coordinates': 1}
    arguments.update(options)
    with pytest.raises(error):
        tk.formats.QuantizedLayout(**arguments)


@pytest.mark.parametrize('method', ['encode_indices', 'decode_digits'])
@pytest.mark.parametrize('value,error', [
    ([0], TypeError), (torch.tensor([0.]), TypeError),
    (torch.tensor([False]), TypeError), (torch.tensor([0j]), TypeError),
    (torch.tensor(0), TypeError), (torch.tensor([-1]), ValueError),
    (torch.tensor([2]), ValueError),
])
def test_layout_tensor_errors(method, value, error):
    with pytest.raises(error):
        getattr(tk.formats.QuantizedLayout(1), method)(value)


@pytest.mark.parametrize('target,error', [
    ('custom', TypeError), ('unknown', TypeError), (1, TypeError),
    (tk.formats.QuantizedLayout(1, base=3), ValueError),
])
def test_reordering_rejects_unknown_or_incompatible_target(target, error):
    layout = tk.formats.QuantizedLayout(1)
    with pytest.raises(error):
        layout.reorder_configurations(torch.tensor([[0], [1]]), target)


@pytest.mark.parametrize('offset', ['endpoints', 'left', 'centers', 'right', 0.25, 0.75])
@pytest.mark.parametrize('reverse', [False, True])
def test_affine_transformations_and_grid_formula(offset, reverse):
    domain = torch.tensor([[-2., 3.], [5., 11.]], dtype=torch.float64)
    if reverse:
        with pytest.raises(ValueError, match='strictly increasing'):
            tk.formats.AffineCoordinateMap(domain.flip(-1), (4, 5), grid_offset=offset)
        return
    coordinate_map = tk.formats.AffineCoordinateMap(domain, (4, 5), grid_offset=offset)
    generator = torch.Generator().manual_seed(47)
    unit = torch.rand(2, 3, 2, dtype=torch.float64, generator=generator)
    expected = domain[:, 0] + unit * (domain[:, 1] - domain[:, 0])
    assert torch.allclose(coordinate_map.forward(unit), expected)
    assert torch.allclose(coordinate_map.inverse(expected), unit)
    indices = torch.cartesian_prod(torch.arange(4), torch.arange(5))
    sizes = torch.tensor([4, 5])
    numeric = {'left': 0., 'centers': 0.5, 'right': 1.}.get(offset, offset)
    grid_unit = (indices / (sizes - 1) if offset == 'endpoints' else
                 (indices + numeric) / sizes)
    expected_grid = domain[:, 0] + grid_unit * (domain[:, 1] - domain[:, 0])
    actual = coordinate_map.from_indices(indices)
    assert torch.allclose(actual, expected_grid)
    assert torch.equal(coordinate_map.to_indices(actual), indices)


@pytest.mark.parametrize('options,error', [
    ({'domain': None}, ValueError), ({'domain': [0., 0.]}, ValueError),
    ({'domain': [0., float('inf')]}, ValueError),
    ({'domain': [[0., 1.], [0., 2.]]}, ValueError),
    ({'grid_size': ()}, ValueError), ({'grid_size': (1,)}, ValueError),
    ({'grid_size': (True,)}, TypeError), ({'grid_size': '4'}, TypeError),
    ({'out_of_domain': 'ignore'}, ValueError),
])
def test_affine_constructor_errors(options, error):
    arguments = {'domain': [0., 1.], 'grid_size': (4,)}
    arguments.update(options)
    with pytest.raises(error):
        tk.formats.AffineCoordinateMap(**arguments)


@pytest.mark.parametrize('method', ['forward', 'inverse', 'to_indices'])
@pytest.mark.parametrize('value,error', [
    ([0.5], TypeError), (torch.tensor([0]), TypeError),
    (torch.tensor([0.5j]), TypeError), (torch.tensor([float('nan')]), ValueError),
    (torch.tensor([float('inf')]), ValueError), (torch.tensor(0.5), ValueError),
    (torch.zeros(2, 2), ValueError), (torch.tensor([-0.5]), ValueError),
    (torch.tensor([1.5]), ValueError),
])
def test_coordinate_input_errors(method, value, error):
    coordinate_map = tk.formats.AffineCoordinateMap([0., 1.], (4,))
    with pytest.raises(error):
        getattr(coordinate_map, method)(value)


@pytest.mark.parametrize('value,error', [
    ([0], TypeError), (torch.tensor([0.]), TypeError),
    (torch.tensor([True]), TypeError), (torch.tensor([-1]), ValueError),
    (torch.tensor([4]), ValueError), (torch.tensor([0, 1]), ValueError),
])
def test_map_grid_index_errors(value, error):
    with pytest.raises(error):
        tk.formats.AffineCoordinateMap([0., 1.], (4,)).from_indices(value)


@pytest.mark.parametrize('offset', ['left', 'centers', 'right', 'endpoints'])
def test_functional_uniform_grid_and_inverse(offset):
    domain = torch.tensor([2., 5.], dtype=torch.float64)
    coordinate_map = tk.formats.FunctionalCoordinateMap(
        domain, (4,), grid_offset=offset,
        forward_function=lambda unit, domain: domain[0] + (domain[1] - domain[0]) * unit.square(),
        inverse_function=lambda value, domain: ((value - domain[0]) / (domain[1] - domain[0])).sqrt())
    affine = tk.formats.AffineCoordinateMap([0., 1.], (4,), grid_offset=offset)
    indices = torch.arange(4).reshape(-1, 1)
    unit = affine.from_indices(indices).double()
    expected = 2 + 3 * unit.square()
    actual = coordinate_map.from_indices(indices)
    assert torch.allclose(actual, expected)
    assert torch.equal(coordinate_map.to_indices(actual), indices)
    assert torch.allclose(coordinate_map.inverse(expected), unit)


@pytest.mark.parametrize('result,error', [
    ([0.], TypeError), (torch.ones(1, 1, 1), ValueError),
    (torch.tensor([[float('nan')]]), ValueError),
    (torch.tensor([[1j]]), TypeError),
])
def test_functional_result_errors(result, error):
    coordinate_map = tk.formats.FunctionalCoordinateMap(
        grid_size=(4,), forward_function=lambda unit, domain: result,
        inverse_function=lambda value, domain: result)
    with pytest.raises(error):
        coordinate_map.forward(torch.tensor([[0.5]]))
    with pytest.raises(error):
        coordinate_map.inverse(torch.tensor([[0.5]]))


@pytest.mark.parametrize('function', ['forward_function', 'inverse_function'])
def test_functional_requires_callable_functions(function):
    arguments = {'grid_size': (4,), 'forward_function': lambda unit, domain: unit}
    arguments[function] = 1
    with pytest.raises(TypeError):
        tk.formats.FunctionalCoordinateMap(**arguments)


@pytest.mark.parametrize('reverse', [False, True])
def test_explicit_piecewise_interpolation_and_inverse(reverse):
    grid = torch.tensor([0., 1., 4., 10.], dtype=torch.float64)
    if reverse:
        grid = grid.flip(0)
    coordinate_map = tk.formats.ExplicitGridMap(grid)
    unit = torch.tensor([[0.], [1 / 6], [1 / 3], [0.5], [2 / 3], [5 / 6], [1.]],
                        dtype=torch.float64)
    expected = torch.stack((grid[0], (grid[0] + grid[1]) / 2, grid[1],
                            (grid[1] + grid[2]) / 2, grid[2],
                            (grid[2] + grid[3]) / 2, grid[3])).reshape(-1, 1)
    assert torch.allclose(coordinate_map.forward(unit), expected)
    assert torch.allclose(coordinate_map.inverse(expected), unit)
    indices = torch.arange(4).reshape(-1, 1)
    assert torch.equal(coordinate_map.from_indices(indices).flatten(), grid)
    assert torch.equal(coordinate_map.to_indices(grid[:, None]), indices)


@pytest.mark.parametrize('grid,error', [
    (None, TypeError), ('grid', TypeError), ((), ValueError),
    (torch.ones(2, 2, 2), ValueError), (torch.tensor([1.]), ValueError),
    (torch.tensor([0, 1]), ValueError), (torch.tensor([0j, 1j]), ValueError),
    (torch.tensor([0., float('inf')]), ValueError),
    (torch.tensor([0., 1., 1.]), ValueError),
    (torch.tensor([0., 2., 1.]), ValueError),
])
def test_explicit_grid_constructor_errors(grid, error):
    with pytest.raises(error):
        tk.formats.ExplicitGridMap(grid)


def test_composite_coordinate_map_independent_transformations():
    from tensorkrowch.formats.quantization import _CompositeCoordinateMap

    affine = tk.formats.AffineCoordinateMap([-2., 2.], (4,), grid_offset='endpoints')
    explicit = tk.formats.ExplicitGridMap(torch.tensor([0., 1., 4.]))
    coordinate_map = _CompositeCoordinateMap([affine, explicit])
    unit = torch.tensor([[0., 0.], [0.5, 0.75], [1., 1.]])
    expected = torch.tensor([[-2., 0.], [0., 2.5], [2., 4.]])
    assert coordinate_map.grid_size == (4, 3)
    assert torch.allclose(coordinate_map.forward(unit), expected)
    assert torch.allclose(coordinate_map.inverse(expected), unit)
    indices = torch.tensor([[0, 0], [1, 1], [3, 2]])
    assert torch.equal(coordinate_map.to_indices(coordinate_map.from_indices(indices)), indices)
    for invalid in (None, torch.ones(3), torch.ones(3, 1)):
        with pytest.raises(ValueError):
            coordinate_map.forward(invalid)
    for maps in ([], [1], [tk.formats.AffineCoordinateMap([0., 1.], (2, 3))]):
        with pytest.raises(ValueError):
            _CompositeCoordinateMap(maps)


@pytest.mark.parametrize('map_kind', ['affine', 'functional', 'explicit'])
def test_continuous_coordinate_transform_gradients(map_kind):
    if map_kind == 'affine':
        coordinate_map = tk.formats.AffineCoordinateMap([0., 3.], (4,))
        derivative = torch.tensor([[3.], [3.]], dtype=torch.float64)
    elif map_kind == 'functional':
        coordinate_map = tk.formats.FunctionalCoordinateMap(
            grid_size=(4,), forward_function=lambda unit, domain: unit.square(),
            inverse_function=lambda value, domain: value.sqrt())
        derivative = torch.tensor([[0.4], [1.6]], dtype=torch.float64)
    else:
        coordinate_map = tk.formats.ExplicitGridMap(torch.tensor([0., 1., 4.], dtype=torch.float64))
        derivative = torch.tensor([[2.], [6.]], dtype=torch.float64)
    unit = torch.tensor([[0.2], [0.8]], dtype=torch.float64, requires_grad=True)
    coordinates = coordinate_map.forward(unit)
    actual, = torch.autograd.grad(coordinates.sum(), unit, retain_graph=True)
    assert torch.allclose(actual, derivative)
    restored = coordinate_map.inverse(coordinates)
    inverse_grad, = torch.autograd.grad(restored.sum(), unit)
    assert torch.allclose(restored, unit)
    assert torch.allclose(inverse_grad, torch.ones_like(unit))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('offset', ['left', 'right', 'centers', 'endpoints'])
@pytest.mark.parametrize('size', [3, 5, 7])
@pytest.mark.parametrize('domain', [[-1., 1.], [5., 11.], [1., 10.]])
def test_uniform_grid_points_recover_their_own_indices(dtype, offset, size, domain):
    coordinate_map = tk.formats.AffineCoordinateMap(
        torch.tensor(domain, dtype=dtype), (size,), grid_offset=offset)
    indices = torch.arange(size).reshape(-1, 1)
    assert torch.equal(coordinate_map.to_indices(coordinate_map.from_indices(indices)), indices)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('offset', ['left', 'right'])
def test_affine_indices_snap_rounding_at_grid_point(dtype, offset):
    coordinate_map = tk.formats.AffineCoordinateMap(
        torch.tensor([5., 11.], dtype=dtype), (5,), grid_offset=offset)
    point = coordinate_map.from_indices(torch.tensor([[2]]))
    below = torch.nextafter(point, torch.full_like(point, -torch.inf))
    above = torch.nextafter(point, torch.full_like(point, torch.inf))
    coordinates = torch.cat([below, point, above])
    assert coordinate_map.to_indices(coordinates).flatten().tolist() == [2, 2, 2]


@pytest.mark.parametrize('offset', ['centers', 'endpoints', 0.25, 0.75])
def test_affine_nearest_indices_match_unit_grid(offset):
    coordinate_map = tk.formats.AffineCoordinateMap(
        torch.tensor([[-2., 3.], [5., 11.]], dtype=torch.float64),
        (5, 7), grid_offset=offset)
    generator = torch.Generator().manual_seed(47)
    unit = torch.rand(101, 2, dtype=torch.float64, generator=generator)
    coordinates = coordinate_map.forward(unit)
    expected = []
    for coordinate, size in enumerate(coordinate_map.grid_size):
        indices = torch.zeros(size, 2, dtype=torch.long)
        indices[:, coordinate] = torch.arange(size)
        grid = tk.formats.quantization._indices_to_unit(
            indices, coordinate_map.grid_size, offset,
            dtype=unit.dtype)[:, coordinate]
        expected.append((unit[:, coordinate, None] - grid).abs().argmin(-1))
    assert torch.equal(coordinate_map.to_indices(coordinates), torch.stack(expected, -1))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('offset', ['left', 'right', 'centers', 'endpoints', 0.25, 0.75])
@pytest.mark.parametrize('functional', [False, True])
def test_unit_indices_snap_only_within_boundary_tolerance(dtype, offset, functional):
    size = 9 if offset == 'endpoints' else 8
    if offset in ('left', 'right'):
        boundary = 2 / size
        expected = [1, 2, 2, 2, 2] if offset == 'left' else [1, 1, 1, 1, 2]
    else:
        numeric = {'centers': 0.5, 'endpoints': 0.}.get(offset, offset)
        boundary = (2.5 + numeric) / (size - 1 if offset == 'endpoints' else size)
        expected = [2, 2, 2, 2, 3]
    eps = torch.finfo(dtype).eps
    unit = torch.tensor([
        boundary - 8 * eps, boundary - eps / 2, boundary,
        boundary + eps / 2, boundary + 8 * eps], dtype=dtype).unsqueeze(-1)
    if functional:
        coordinate_map = tk.formats.FunctionalCoordinateMap(
            forward_function=lambda values, domain: values,
            inverse_function=lambda values, domain: values,
            domain=None, grid_size=(size,), grid_offset=offset)
    else:
        coordinate_map = tk.formats.AffineCoordinateMap(
            torch.tensor([0., 1.], dtype=dtype), (size,), grid_offset=offset)
    assert coordinate_map.to_indices(unit).flatten().tolist() == expected


@pytest.mark.parametrize('offset', ['left', 'right', 'centers', 'endpoints', 0.25])
def test_layout_and_coordinate_maps_on_device(device_dtype, assert_close, offset):
    device, dtype = device_dtype
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    layout = tk.formats.QuantizedLayout(2, base=(2, 3), level=(2, 2))
    indices = torch.cartesian_prod(torch.arange(4), torch.arange(9)).to(device)
    digits = layout.encode_indices(indices)
    assert digits.device.type == device
    assert torch.equal(layout.decode_digits(digits), indices)
    target = tk.formats.QuantizedLayout(2, base=(2, 3), level=(2, 2), ordering='grouped')
    reordered = layout.reorder_configurations(digits, target)
    assert torch.equal(target.decode_digits(reordered), indices)
    domain = torch.tensor([[-1., 1.], [5., 11.]], dtype=real_dtype, device=device)
    affine = tk.formats.AffineCoordinateMap(domain, layout.grid_size, grid_offset=offset)
    functional = tk.formats.FunctionalCoordinateMap(
        domain=domain, grid_size=layout.grid_size, grid_offset=offset,
        forward_function=lambda unit, intervals: intervals[:, 0] +
        unit * (intervals[:, 1] - intervals[:, 0]),
        inverse_function=lambda coordinates, intervals: (coordinates - intervals[:, 0]) /
        (intervals[:, 1] - intervals[:, 0]))
    for coordinate_map in (affine, functional):
        coordinates = coordinate_map.from_indices(indices)
        assert coordinates.device.type == device and not coordinates.is_complex()
        assert torch.equal(coordinate_map.to_indices(coordinates), indices)
        unit = torch.tensor([[0.25, 0.75]], dtype=real_dtype, device=device,
                             requires_grad=True)
        restored = coordinate_map.inverse(coordinate_map.forward(unit))
        assert_close(restored, unit)
        gradient, = torch.autograd.grad(restored.sum(), unit)
        assert_close(gradient, torch.ones_like(unit))
        if dtype.is_complex:
            with pytest.raises(TypeError, match='floating'):
                coordinate_map.to_indices(coordinates.to(dtype))

    explicit = tk.formats.ExplicitGridMap([
        torch.tensor([-1., 0., 1., 3.], dtype=real_dtype, device=device),
        torch.linspace(5., 11., 9, dtype=real_dtype, device=device)])
    coordinates = explicit.from_indices(indices)
    assert torch.equal(explicit.to_indices(coordinates), indices)
