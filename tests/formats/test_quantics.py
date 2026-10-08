"""Quantics construction, evaluation, algebra and coordinate metadata."""

import pytest
import torch

import tensorkrowch as tk


# Mutations


def test_quantics_replacement_restores_layout_and_metadata():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    format = tk.formats.QTT(
        [torch.ones(2, 3), torch.ones(3, 2)], layout.n_coordinates,
        layout=layout, coordinate_map=_coordinate_map(layout))
    cores = format.cores
    with pytest.raises(ValueError, match='[Dd]igit core dimensions'):
        cores[0] = torch.ones(4, 3)
    with pytest.raises(ValueError, match='[Dd]igit core dimensions'):
        format.cores = [torch.ones(2)]
    assert format.cores is cores
    assert format.in_dim == layout.in_dim and format.rank == [3]

    matrix = tk.formats.QTTM([torch.ones(2, 3, 2), torch.ones(3, 2, 2)],
                             layout.n_coordinates, layout.n_coordinates,
                             in_layout=layout, out_layout=layout,
                             in_coordinate_map=_coordinate_map(layout),
                             out_coordinate_map=_coordinate_map(layout))
    previous = matrix.cores[0]
    with pytest.raises(ValueError, match='paired digit layouts'):
        matrix.cores[0] = torch.ones(4, 3, 2)
    assert matrix.cores[0] is previous
    assert matrix.in_dim == matrix.out_dim == layout.in_dim


# Quantics


def _coordinate_map(layout, domain=None, grid_offset='endpoints'):
    return tk.formats.AffineCoordinateMap(
        torch.tensor([0., 1.]) if domain is None else domain,
        layout.grid_size, grid_offset=grid_offset)


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('matrix', [False, True])
@pytest.mark.parametrize('grid, index', [
    ('left', 1), ('centers', 0), ('right', 0), (0.25, 1),
])
def test_coordinate_map_uses_the_selected_grid(cyclic, matrix, grid, index):
    domain = torch.tensor([0., 1.])
    coordinates = torch.tensor([[0.25]])
    layout = tk.formats.QuantizedLayout(1, 4, 1)
    coordinate_map = _coordinate_map(layout, domain, grid_offset=grid)
    if matrix:
        core = torch.arange(16.).reshape(4, 4)
        cls = tk.formats.QTRM if cyclic else tk.formats.QTTM
        format = cls(
            [core.reshape(1, 4, 1, 4) if cyclic else core], 1, 1,
            in_layout=layout, out_layout=layout,
            in_coordinate_map=coordinate_map, out_coordinate_map=coordinate_map)
        actual = format.evaluate_coordinates(coordinates, coordinates)
        expected = core[index, index]
    else:
        core = torch.arange(4.)
        cls = tk.formats.QTR if cyclic else tk.formats.QTT
        format = cls(
            [core.reshape(1, 4, 1) if cyclic else core], 1,
            layout=layout, coordinate_map=coordinate_map)
        actual = format.evaluate_coordinates(coordinates)
        expected = core[index]

    assert actual.item() == expected.item()


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('explicit_data', [False, True])
def test_error_requires_scalar_quantics_samples(cyclic, explicit_data):
    layout = tk.formats.QuantizedLayout(1, 2, 1)
    cls = tk.formats.QTR if cyclic else tk.formats.QTT
    core = torch.ones(1, 2, 1) if cyclic else torch.ones(2)
    format = cls([core], 1, layout=layout,
                 coordinate_map=_coordinate_map(layout))
    indices = torch.tensor([[0], [1]])

    def function(samples):
        return torch.ones(samples.shape[0])

    record = format.error(function, indices,
                          data=indices if explicit_data else None)
    assert record.absolute == 0
    features = torch.ones(2, 1, 2)
    with pytest.raises(ValueError, match='batch dimensions'):
        format.error(function, features,
                     data=indices if explicit_data else None)


def test_explicit_grid_size_is_checked_when_building_quantics():
    layout = tk.formats.QuantizedLayout(1, base=2, level=2)
    coordinate_map = tk.formats.ExplicitGridMap(torch.tensor([0., 1., 4.]))
    cores = [torch.ones(2, 1), torch.ones(1, 2)]

    with pytest.raises(ValueError, match='coordinate_map.grid_size'):
        tk.formats.QTT(cores, 1, layout=layout,
                       coordinate_map=coordinate_map)


def test_vector_shorthand_builds_uniform_and_explicit_coordinates():
    cores = [torch.ones(2, 1), torch.ones(1, 2)]
    uniform = tk.formats.QTT(
        cores, 1, base=2, level=2, domain=torch.tensor([0., 3.]))
    explicit = tk.formats.QTT(
        cores, 1, grid_coordinates=torch.tensor([0., 1., 4., 10.]),
        base=2, level=2)

    assert uniform.layout.grid_size == explicit.layout.grid_size == (4,)
    assert isinstance(uniform.coordinate_map, tk.formats.AffineCoordinateMap)
    assert isinstance(explicit.coordinate_map, tk.formats.ExplicitGridMap)
    assert torch.equal(uniform.evaluate_coordinates(torch.tensor([[3.]])),
                       explicit.evaluate_coordinates(torch.tensor([[10.]])))

    with pytest.raises(ValueError, match='Do not combine'):
        tk.formats.QTT(cores, 1, layout=uniform.layout, base=2)
    with pytest.raises(ValueError, match='domain'):
        tk.formats.QTT(cores, 1, grid_coordinates=torch.tensor([0., 1., 2., 3.]),
                       base=2, level=2, domain=torch.tensor([0., 3.]))


def test_matrix_shorthand_resolves_input_and_output_separately():
    matrix = tk.formats.QTTM(
        [torch.ones(2, 3)], 1, 1,
        in_base=2, in_level=1, in_domain=torch.tensor([0., 1.]),
        out_grid_coordinates=torch.tensor([0., 2., 5.]), out_base=3, out_level=1)

    assert isinstance(matrix.in_coordinate_map, tk.formats.AffineCoordinateMap)
    assert isinstance(matrix.out_coordinate_map, tk.formats.ExplicitGridMap)
    assert matrix.evaluate_coordinates(
        torch.tensor([[1.]]), torch.tensor([[5.]])).item() == 1

    with pytest.raises(ValueError, match='equal `n_sites`'):
        tk.formats.QTTM([torch.ones(2, 3)], 1, 1,
                        in_base=2, in_level=2, in_domain=torch.tensor([0., 1.]),
                        out_base=3, out_level=1, out_domain=torch.tensor([0., 1.]))


def test_ring_shorthand_uses_the_same_coordinate_resolution():
    vector = tk.formats.QTR(
        [torch.ones(1, 2, 1)], 1,
        base=2, level=1, domain=torch.tensor([0., 1.]))
    matrix = tk.formats.QTRM(
        [torch.ones(1, 2, 1, 3)], 1, 1,
        in_base=2, in_level=1, in_domain=torch.tensor([0., 1.]),
        out_grid_coordinates=torch.tensor([0., 2., 5.]), out_base=3, out_level=1)

    assert vector.evaluate_coordinates(torch.tensor([[1.]])).item() == 1
    assert matrix.evaluate_indices(torch.tensor([[1]]),
                                   torch.tensor([[2]])).item() == 1


def test_matrix_coordinate_counts_can_differ_with_equal_site_counts():
    matrix = tk.formats.QTTM(
        [torch.ones(2, 1, 2), torch.ones(1, 2, 3)], 1, 2,
        in_base=2, in_level=2, in_domain=torch.tensor([0., 1.]),
        out_base=(2, 3), out_level=1, out_domain=torch.tensor([0., 1.]))

    assert matrix.in_layout.n_sites == matrix.out_layout.n_sites == 2
    assert matrix.to_dense_grid().shape == (4, 2, 3)


@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_quantized_grid_and_arithmetic(ordering, dense_cores):
    layout = tk.formats.QuantizedLayout(2, 2, (3, 4), ordering=ordering)
    indices = torch.cartesian_prod(torch.arange(8), torch.arange(16))
    digits = layout.encode_indices(indices)
    values = (indices[:, 0] + 2 * indices[:, 1]).to(torch.float64)
    tensor = torch.empty(layout.in_dim, dtype=torch.float64)
    tensor[tuple(digits.T)] = values
    cores = dense_cores(tensor, layout.in_dim)
    format = tk.formats.QTT(
        cores, 2, layout=layout,
        coordinate_map=_coordinate_map(
            layout, torch.tensor([[0., 7.], [0., 15.]])))
    assert torch.allclose(format.evaluate_indices(indices), values)
    assert torch.allclose(format.evaluate_coordinates(indices.to(torch.float64)), values)
    assert torch.allclose(format.to_dense_grid(), values.reshape(8, 16))
    assert isinstance(format + format, tk.formats.QTT)
    assert torch.allclose((format * format).evaluate_indices(indices), values.square())
    assert torch.allclose((2 * format).to_dense_grid(), 2 * values.reshape(8, 16))
    assert torch.allclose(format.to_tt().contract_dense(), tensor)
    with pytest.raises(ValueError, match='semantics'):
        format + format.to_tt()
    detached = format.clone().detach()
    assert detached.coordinate_map.domain.data_ptr() != \
        format.coordinate_map.domain.data_ptr()
    complex_format = format.to(dtype=torch.complex128)
    assert not complex_format.coordinate_map.domain.is_complex()
    row = format.H
    assert isinstance(row, tk.formats.QTT)
    assert row.layout is layout
    assert torch.allclose(row.to_tt() @ format.to_tt(), tensor.square().sum())


@pytest.mark.parametrize('n_coordinates, base, level', [
    (1, 2, 3), (2, (2, 3), (2, 1)),
])
def test_ring_coordinate_rotation_and_conversion(n_coordinates, base, level):
    layout = tk.formats.QuantizedLayout(n_coordinates, base, level)
    generator = torch.Generator().manual_seed(4)
    cores = [torch.randn(2, dim, 2, dtype=torch.float64, generator=generator)
             for dim in layout.in_dim]
    format = tk.formats.QTR(cores, layout.n_coordinates, layout=layout,
                            coordinate_map=_coordinate_map(layout))
    indices = torch.cartesian_prod(
        *[torch.arange(size) for size in layout.grid_size]).reshape(-1, n_coordinates)
    values = format.evaluate_indices(indices)
    assert format.rotate(0).layout is layout
    with pytest.raises(NotImplementedError, match='to_qtt'):
        format.to_tt()
    train = format.to_qtt()
    assert type(train) is tk.formats.QTT
    assert train.layout is layout
    assert train.coordinate_map is format.coordinate_map
    assert torch.allclose(train.to_tt().contract_dense(), format.contract_dense())
    assert torch.allclose(format.to_tr().to_tt().contract_dense(), format.contract_dense())
    assert torch.allclose(format.to_qtt().evaluate_indices(indices), values)
    assert torch.allclose(format.H.to_qtt() @ format.to_qtt(), values.square().sum())
    for first in range(3):
        assert torch.allclose(format.rotate(first).evaluate_indices(indices), values)
        assert torch.allclose(format.H.rotate(first) @ format.rotate(first), values.square().sum())


@pytest.mark.parametrize('cyclic', [False, True])
def test_quantized_vectors_require_one_digit_per_site(cyclic):
    layout = tk.formats.QuantizedLayout(1, 2, 1)
    cls = tk.formats.QTR if cyclic else tk.formats.QTT
    cores = ([torch.ones(1, 2, 1), torch.ones(1, 3, 1)] if cyclic
             else [torch.ones(2, 1), torch.ones(1, 3)])
    with pytest.raises(ValueError, match='layout.in_dim'):
        cls(cores, 1, layout=layout,
            coordinate_map=_coordinate_map(layout))

    wrong_core = torch.ones(1, 3, 1) if cyclic else torch.ones(3)
    with pytest.raises(ValueError, match='layout.in_dim'):
        cls([wrong_core], 1, layout=layout,
            coordinate_map=_coordinate_map(layout))


@pytest.mark.parametrize('cyclic', [False, True])
def test_quantized_core_replacement_preserves_digit_dimensions(cyclic):
    layout = tk.formats.QuantizedLayout(1, 2, 1)
    cls = tk.formats.QTR if cyclic else tk.formats.QTT
    core = torch.ones(1, 2, 1) if cyclic else torch.ones(2)
    format = cls([core], 1, layout=layout,
                 coordinate_map=_coordinate_map(layout))
    wrong_core = torch.ones(1, 3, 1) if cyclic else torch.ones(3)
    with pytest.raises(ValueError, match='layout.in_dim'):
        format.cores[0] = wrong_core
    assert format.cores[0] is core
    assert format.in_dim == (2,)


@pytest.mark.parametrize('cyclic', [False, True])
def test_quantized_evaluation_keeps_structural_and_data_batches(cyclic):
    layout = tk.formats.QuantizedLayout(1, 2, 1)
    values = torch.tensor([[1., 2.], [3., 4.]])
    cls = tk.formats.QTR if cyclic else tk.formats.QTT
    core = values.reshape(2, 1, 2, 1) if cyclic else values
    format = cls([core], 1, layout=layout, coordinate_map=_coordinate_map(layout),
                 n_batches=1)
    indices = torch.tensor([[[0], [1]], [[1], [0]]])
    expected = values[:, indices[..., 0]]
    assert torch.equal(format.evaluate_indices(indices), expected)
    assert torch.equal(format.evaluate_digits(layout.encode_indices(indices)), expected)
    assert torch.equal(format.to_dense_grid(), values)


def test_quantized_matrix_transpose_apply(dense_cores):
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    generator = torch.Generator().manual_seed(6)
    dense = torch.randn(2, 2, 2, 2, dtype=torch.complex128, generator=generator)
    matrix = tk.formats.QTTM(
        dense_cores(dense, layout.in_dim, layout.in_dim), 1, 1,
        in_layout=layout, out_layout=layout,
        in_coordinate_map=_coordinate_map(layout),
        out_coordinate_map=_coordinate_map(layout))
    vector = tk.formats.QTT(
        [torch.ones(2, 1, dtype=dense.dtype),
         torch.ones(1, 2, dtype=dense.dtype)], 1, layout=layout,
                            coordinate_map=_coordinate_map(layout))
    assert isinstance(matrix @ vector, tk.formats.QTT)
    assert isinstance(matrix @ matrix.H, tk.formats.QTTM)
    assert matrix.T.in_layout == matrix.out_layout
    assert torch.allclose(matrix.H.H.contract_dense(), dense)
    assert isinstance(matrix.apply(torch.tensor([[0, 0]])), tk.formats.QTT)


def test_quantized_outer_product_preserves_coordinate_spaces():
    output_layout = tk.formats.QuantizedLayout(1, 2, 1)
    input_layout = tk.formats.QuantizedLayout(1, 3, 1)
    x = tk.formats.QTT(
        [torch.tensor([1., 2.])], 1, layout=output_layout,
        coordinate_map=_coordinate_map(output_layout))
    y = tk.formats.QTT(
        [torch.tensor([2., 3., 4.])], 1, layout=input_layout,
        coordinate_map=_coordinate_map(input_layout, torch.tensor([-1., 1.])))
    outer = x @ y.H
    assert isinstance(outer, tk.formats.QTTM)
    assert outer.in_layout is input_layout
    assert outer.out_layout is output_layout
    in_indices = torch.arange(3).unsqueeze(-1)
    out_indices = torch.arange(2).unsqueeze(-1)
    assert torch.equal(outer.in_coordinate_map.from_indices(in_indices),
                       y.coordinate_map.from_indices(in_indices))
    assert torch.equal(outer.out_coordinate_map.from_indices(out_indices),
                       x.coordinate_map.from_indices(out_indices))
    assert torch.equal(outer.contract_dense(), torch.outer(
        y.contract_dense(), x.contract_dense()))
    assert torch.allclose((x.H @ outer).T.contract_dense(),
                          (x.H @ x) * y.contract_dense())
    with pytest.raises(ValueError, match='layouts'):
        x.H @ tk.formats.QTT(
            x.cores, 1, layout=output_layout,
            coordinate_map=_coordinate_map(output_layout, torch.tensor([0., 2.])))
    with pytest.raises(ValueError, match='semantics'):
        x @ y.to_tt().H


@pytest.mark.parametrize('operation', ['sum', 'scale', 'hadamard', 'apply', 'transpose'])
def test_quantics_constructs_results_directly(operation, monkeypatch):
    from tensorkrowch.formats.formats1d import TensorFormat1D

    layout = tk.formats.QuantizedLayout(1, 2, 2)
    matrix = tk.formats.QTTM(
        [torch.eye(2).unsqueeze(1), torch.eye(2).unsqueeze(0)], 1, 1,
        in_layout=layout, out_layout=layout,
        in_coordinate_map=_coordinate_map(layout),
        out_coordinate_map=_coordinate_map(layout))
    vector = tk.formats.QTT([torch.ones(2, 1), torch.ones(1, 2)], 1,
                            layout=layout,
                            coordinate_map=_coordinate_map(layout))
    calls = []
    validate = TensorFormat1D.validate

    def counted_validate(self):
        calls.append(type(self))
        return validate(self)

    monkeypatch.setattr(TensorFormat1D, 'validate', counted_validate)
    if operation == 'sum':
        result = vector + vector
    elif operation == 'scale':
        result = vector * 2
    elif operation == 'hadamard':
        result = vector * vector
    elif operation == 'apply':
        result = matrix @ vector
    else:
        result = matrix.T
    assert len(calls) == (0 if operation == 'transpose' else 1)
    assert isinstance(result, tk.formats.QTTM if operation == 'transpose' else tk.formats.QTT)


def test_plain_operands_reject_quantics_in_both_orders():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    quantics = tk.formats.QTT([torch.ones(2, 1), torch.ones(1, 2)], 1,
                              layout=layout,
                            coordinate_map=_coordinate_map(layout))
    plain = quantics.to_tt()
    for first, second in [(plain, quantics), (quantics, plain)]:
        with pytest.raises(ValueError, match='semantics'):
            first + second
        with pytest.raises(ValueError, match='semantics'):
            first * second


def test_quantics_blocking_preserves_coordinate_contract():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    format = tk.formats.QTT([torch.ones(2, 1), torch.ones(1, 2)], 1,
                            layout=layout,
                            coordinate_map=_coordinate_map(layout))
    cores = format.cores
    with pytest.raises(ValueError, match='layout'):
        format.block([2])
    assert format.cores is cores
    assert format.layout is layout
    assert format.in_dim == (2, 2)
    plain = format.to_tt()
    grouped = plain.block([2])
    plain.unblock(grouped)
    assert torch.allclose(plain.contract_dense(), format.contract_dense())


@pytest.mark.parametrize('cyclic', [False, True])
def test_plain_conversion_owns_its_bonds(cyclic):
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    if cyclic:
        format = tk.formats.QTR([torch.ones(2, 2, 2)] * 2, 1,
                                layout=layout,
                            coordinate_map=_coordinate_map(layout))
        plain = format.to_tr
    else:
        format = tk.formats.QTT([torch.eye(2), torch.eye(2)], 1,
                                layout=layout,
                            coordinate_map=_coordinate_map(layout))
        plain = format.to_tt
    format.bonds = [torch.ones(2)] * (2 if cyclic else 1)
    converted = plain()
    assert converted.bonds is not format.bonds
    converted.bonds.factors[0] = torch.full((2,), 2.)
    assert torch.equal(format.bonds.factors[0], torch.ones(2))
    with pytest.raises(ValueError, match='factor dimensions'):
        converted.bonds.factors[0] = torch.ones(3)
    assert torch.equal(converted.bonds.factors[0], torch.full((2,), 2.))


@pytest.mark.parametrize('n_batches', [0, 1])
def test_quantized_ring_matrix_conversion_requires_explicit_target(n_batches):
    layout = tk.formats.QuantizedLayout(1, 2, 1)
    coordinate_map = _coordinate_map(layout)
    core = torch.arange(16., dtype=torch.float64).reshape(2, 2, 2, 2)
    if n_batches:
        core = torch.stack([core, 2 * core])
    format = tk.formats.QTRM(
        [core], 1, 1, in_layout=layout, out_layout=layout,
        in_coordinate_map=coordinate_map, out_coordinate_map=coordinate_map,
        n_batches=n_batches)

    with pytest.raises(NotImplementedError, match='to_qttm'):
        format.to_ttm()
    matrix = format.to_qttm()
    assert type(matrix) is tk.formats.QTTM
    assert matrix.in_layout is format.in_layout
    assert matrix.out_layout is format.out_layout
    assert matrix.in_coordinate_map is coordinate_map
    assert matrix.out_coordinate_map is coordinate_map
    assert matrix.batch_shape == format.batch_shape
    assert torch.allclose(matrix.contract_dense(), format.contract_dense())
    assert torch.allclose(matrix.to_ttm().contract_dense(), format.contract_dense())
    assert torch.allclose(format.to_trm().to_ttm().contract_dense(), format.contract_dense())


# Structural batches


def test_batched_qttm_coordinates_and_plain_conversion():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    cores = [torch.arange(8., dtype=torch.float64).reshape(2, 2, 1, 2),
             torch.arange(8., dtype=torch.float64).reshape(2, 1, 2, 2)]
    format = tk.formats.QTTM(
        cores, layout.n_coordinates, layout.n_coordinates,
        in_layout=layout, out_layout=layout,
        in_coordinate_map=_coordinate_map(layout),
        out_coordinate_map=_coordinate_map(layout), n_batches=1)
    indices = torch.tensor([[0], [1], [3]])
    actual = format.evaluate_indices(indices, indices)
    for batch in range(2):
        member = tk.formats.QTTM(
            [core[batch] for core in cores],
            layout.n_coordinates, layout.n_coordinates,
            in_layout=layout, out_layout=layout,
            in_coordinate_map=_coordinate_map(layout),
            out_coordinate_map=_coordinate_map(layout))
        assert torch.allclose(actual[batch],
                              member.evaluate_indices(indices, indices))
    plain = format.to_ttm()
    assert plain.batch_shape == (2,)
    assert torch.equal(plain.contract_dense(), format.contract_dense())


# Strict construction paths and their numerical equivalence


def _quantics_arguments(path, n_coordinates=2, level=2, *, dtype=torch.float64,
                        device='cpu'):
    layout = tk.formats.QuantizedLayout(n_coordinates, base=2, level=level)
    domain = torch.tensor([-1., 1.], dtype=dtype, device=device)
    if path == 'uniform':
        return {'base': 2, 'level': level, 'domain': domain}
    if path == 'explicit':
        grids = [torch.linspace(-1., 1., size, dtype=dtype, device=device).pow(3)
                 for size in layout.grid_size]
        return {'base': 2, 'level': level, 'grid_coordinates': grids}
    coordinate_map = tk.formats.FunctionalCoordinateMap(
        domain, layout.grid_size,
        forward_function=lambda unit, domain: 2 * unit.square() - 1,
        inverse_function=lambda coordinates, domain: ((coordinates + 1) / 2).sqrt())
    return {'layout': layout, 'coordinate_map': coordinate_map}


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('path', ['uniform', 'explicit', 'prebuilt'])
@pytest.mark.parametrize('n_batches', [0, 1])
def test_all_quantics_construction_paths(make_format, topology, path, device_dtype,
                                         assert_close, n_batches):
    device, dtype = device_dtype
    plain = make_format(topology, 4, n_batches=n_batches, dtype=dtype,
                        in_dim=(2,) * 4, out_dim=(2,) * 4, device=device)
    cls = getattr(tk.formats, 'Q' + topology.upper())
    arguments = _quantics_arguments(path, dtype=plain.cores[0].real.dtype,
                                     device=device)
    if plain.out_dim is None:
        format = cls(plain.cores, 2, n_batches=n_batches, **arguments)
        layouts = (format.layout,)
    else:
        paired = {prefix + name: value for prefix in ('in_', 'out_')
                  for name, value in arguments.items()}
        format = cls(plain.cores, 2, 2, n_batches=n_batches, **paired)
        layouts = (format.in_layout, format.out_layout)
    assert_close(format.contract_dense(), plain.contract_dense())
    assert format.dtype == dtype and format.n_batches == n_batches
    assert format.device.type == device
    for layout in layouts:
        assert layout.ordering == 'interleaved'
        assert layout.digit_order == 'coarse_to_fine'
        assert layout.grid_size == (4, 4)


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('in_path', ['uniform', 'explicit', 'prebuilt'])
@pytest.mark.parametrize('out_path', ['uniform', 'explicit', 'prebuilt'])
def test_matrix_independent_quantization_paths(make_format, cyclic, in_path, out_path):
    plain = make_format('trm' if cyclic else 'ttm', 4,
                        in_dim=(2,) * 4, out_dim=(2,) * 4)
    arguments = {'in_' + name: value
                 for name, value in _quantics_arguments(in_path).items()}
    arguments.update({'out_' + name: value
                      for name, value in _quantics_arguments(out_path, 1, 4).items()})
    cls = tk.formats.QTRM if cyclic else tk.formats.QTTM
    format = cls(plain.cores, 2, 1, **arguments)
    assert format.to_dense_grid().shape == (4, 4, 16)
    assert torch.allclose(format.contract_dense(), plain.contract_dense())
    transposed = format.T
    assert transposed.in_layout == format.out_layout
    assert transposed.out_layout == format.in_layout
    assert transposed.in_coordinate_map is format.out_coordinate_map
    assert transposed.out_coordinate_map is format.in_coordinate_map
    inputs = torch.tensor([[0, 1], [2, 3]])
    outputs = torch.tensor([[2], [15]])
    assert torch.allclose(format.evaluate_indices(inputs, outputs),
                          transposed.evaluate_indices(outputs, inputs))


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('options,error', [
    ({'n_coordinates': 0}, ValueError), ({'n_coordinates': True}, TypeError),
    ({'n_coordinates': 1.5}, TypeError), ({'base': None}, ValueError),
    ({'level': None}, ValueError), ({'domain': None}, ValueError),
    ({'layout': tk.formats.QuantizedLayout(1)}, ValueError),
    ({'grid_coordinates': torch.arange(3.).double(), 'domain': None}, ValueError),
])
def test_quantics_shorthand_errors(cyclic, options, error):
    arguments = {'n_coordinates': 1, 'base': 2, 'level': 2, 'domain': [0., 1.]}
    arguments.update(options)
    cls = tk.formats.QTR if cyclic else tk.formats.QTT
    cores = [torch.ones(1, 2, 1)] * 2 if cyclic else [torch.ones(2, 1), torch.ones(1, 2)]
    with pytest.raises(error):
        cls(cores, **arguments)


@pytest.mark.parametrize('options,error', [
    ({'layout': None}, TypeError), ({'coordinate_map': None}, TypeError),
    ({'layout': 'layout'}, TypeError), ({'coordinate_map': 'map'}, TypeError),
    ({'n_coordinates': 2}, ValueError),
])
def test_quantics_prebuilt_errors(options, error):
    layout = tk.formats.QuantizedLayout(1, level=2)
    arguments = {'n_coordinates': 1, 'layout': layout,
                 'coordinate_map': _coordinate_map(layout)}
    arguments.update(options)
    with pytest.raises(error):
        tk.formats.QTT([torch.ones(2, 1), torch.ones(1, 2)], **arguments)


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_exponential_evaluation_on_and_between_grid_points(exponential_format,
                                                          cyclic, device_dtype,
                                                          assert_close, ordering):
    device, dtype = device_dtype
    format, function = exponential_format(cyclic, dtype, ordering, device=device)
    indices = torch.cartesian_prod(torch.arange(4), torch.arange(9)).to(device)
    coordinates = format.coordinate_map.from_indices(indices)
    digits = format.layout.encode_indices(indices)
    expected = function(coordinates)
    assert_close(format.evaluate_digits(digits), expected)
    assert_close(format.evaluate_indices(indices), expected)
    assert_close(format.evaluate_coordinates(coordinates), expected)
    assert_close(format.to_dense_grid(), expected.reshape(4, 9))
    assert format.contract_dense().shape == format.layout.in_dim

    # Arbitrary domain coordinates evaluate the selected grid point's function.
    generator = torch.Generator().manual_seed(47)
    unit = torch.rand(2, 5, 2, generator=generator,
                      dtype=coordinates.dtype).to(device)
    samples = format.coordinate_map.forward(unit)
    selected = format.coordinate_map.to_indices(samples)
    selected_coordinates = format.coordinate_map.from_indices(selected)
    assert_close(format.evaluate_coordinates(samples), function(selected_coordinates))
    record = format.error(function, samples, data=format.layout.encode_indices(selected), n_batches=2)
    reference = (function(selected_coordinates) - function(samples)).norm()
    assert_close(record.absolute, reference)
    assert_close(record.relative, reference / function(samples).norm())
    assert record.size == 10


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_matrix_dense_grid_and_evaluations(make_format, cyclic, dtype):
    plain = make_format('trm' if cyclic else 'ttm', 3, n_batches=1, dtype=dtype,
                        in_dim=(2, 2, 2), out_dim=(3, 3, 3))
    inputs = tk.formats.QuantizedLayout(1, 2, 3, digit_order='fine_to_coarse')
    outputs = tk.formats.QuantizedLayout(1, 3, 3)
    in_map = tk.formats.AffineCoordinateMap([-1., 1.], inputs.grid_size)
    out_map = tk.formats.ExplicitGridMap(torch.linspace(0., 2., 27, dtype=torch.float64).square())
    cls = tk.formats.QTRM if cyclic else tk.formats.QTTM
    format = cls(plain.cores, 1, 1, in_layout=inputs, out_layout=outputs,
                 in_coordinate_map=in_map, out_coordinate_map=out_map, n_batches=1)
    pairs = torch.cartesian_prod(torch.arange(8), torch.arange(27))
    in_indices, out_indices = pairs[:, :1], pairs[:, 1:]
    in_digits, out_digits = inputs.encode_indices(in_indices), outputs.encode_indices(out_indices)
    dense = plain.contract_dense()
    axes = tuple(axis for pair in zip(in_digits.T, out_digits.T) for axis in pair)
    expected = dense[(slice(None), *axes)]
    assert torch.allclose(format.evaluate_digits(in_digits, out_digits), expected)
    assert torch.allclose(format.evaluate_indices(in_indices, out_indices), expected)
    assert torch.allclose(format.evaluate_coordinates(in_map.from_indices(in_indices),
                                                      out_map.from_indices(out_indices)), expected)
    assert torch.allclose(format.to_dense_grid(), expected.reshape(2, 8, 27))


@pytest.mark.parametrize('method', ['evaluate_digits', 'evaluate_indices', 'evaluate_coordinates'])
@pytest.mark.parametrize('invalid', ['type', 'shape', 'dtype', 'bounds'])
def test_quantics_evaluation_errors(make_format, method, invalid):
    format = make_format('tt', 2, quantized=True)
    values = torch.zeros(3, 2, dtype=torch.float64 if method == 'evaluate_coordinates' else torch.long)
    if invalid == 'type':
        values = values.tolist()
    elif invalid == 'shape':
        values = values[:, :1]
    elif invalid == 'dtype':
        values = values.long() if method == 'evaluate_coordinates' else values.double()
    else:
        values[0, 0] = 10
    with pytest.raises(TypeError if invalid in ('type', 'dtype') else ValueError):
        getattr(format, method)(values)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_quantics_copy_keeps_real_coordinate_tensors(make_format, topology):
    format = make_format(topology, quantized=True)
    converted = format.to(dtype=torch.complex128, copy=True)
    assert converted.dtype == torch.complex128
    maps = ([converted.coordinate_map] if converted.out_dim is None else
            [converted.in_coordinate_map, converted.out_coordinate_map])
    originals = ([format.coordinate_map] if format.out_dim is None else
                 [format.in_coordinate_map, format.out_coordinate_map])
    for coordinate_map, original in zip(maps, originals):
        assert not coordinate_map.domain.is_complex()
        assert coordinate_map.domain.dtype == torch.float64
        assert coordinate_map.domain.data_ptr() != original.domain.data_ptr()
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(format.cores, converted.cores))
    assert torch.allclose(converted.contract_dense(), format.contract_dense().to(converted.dtype))


@pytest.mark.parametrize('topology', ['tr', 'trm'])
@pytest.mark.parametrize('first', [0, 1, 2, 3])
def test_quantics_ring_rotation_with_factors(make_format, topology, first,
                                             device_dtype, assert_close):
    device, dtype = device_dtype
    format = make_format(topology, 4, n_batches=1, dtype=dtype,
                         quantized=True, device=device)
    format.bonds = [torch.ones(rank, dtype=format.cores[0].real.dtype,
                               device=device) * (site + 1)
                    for site, rank in enumerate(format.rank)]
    rotated = format.rotate(first)
    assert rotated is not format and type(rotated) is type(format)
    expected_factors = format.bonds.factors[first:] + format.bonds.factors[:first]
    assert all(a is b for a, b in zip(rotated.bonds.factors, expected_factors))
    indices = torch.tensor([[0, 0, 0, 0], [1, 2, 1, 2]], device=device)
    if format.out_dim is None:
        assert_close(rotated.evaluate_indices(indices), format.evaluate_indices(indices))
        assert format.H.rotate(first).is_row
    else:
        outputs = torch.tensor([[0, 0, 0, 0], [2, 1, 2, 1]], device=device)
        assert_close(rotated.evaluate_indices(indices, outputs),
                      format.evaluate_indices(indices, outputs))
    dense = format.to_dense_grid()
    relative_error = (rotated.to_dense_grid() - dense).norm() / dense.norm()
    assert relative_error <= 32 * torch.finfo(dense.real.dtype).eps


@pytest.mark.parametrize('topology', ['tr', 'trm'])
@pytest.mark.parametrize('first,error', [(True, TypeError), (1.5, TypeError),
                                        (-1, ValueError), (3, ValueError)])
def test_quantics_rotation_errors(make_format, topology, first, error):
    with pytest.raises(error):
        make_format(topology, quantized=True).rotate(first)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('map_kind', ['explicit', 'functional'])
def test_quantics_coordinate_map_storage_and_semantics(make_format, topology, map_kind):
    plain = make_format(topology, in_dim=(2, 2, 2), out_dim=(2, 2, 2))
    layout = tk.formats.QuantizedLayout(3)
    if map_kind == 'explicit':
        coordinate_map = tk.formats.ExplicitGridMap(
            tuple(torch.tensor([0., 1.]) for _ in range(3)))
    else:
        coordinate_map = tk.formats.FunctionalCoordinateMap(
            domain=[torch.tensor([0., 1.]) for _ in range(3)], grid_size=(2, 2, 2),
            forward_function=lambda unit, domain: unit.square(),
            inverse_function=lambda value, domain: value.sqrt())
    cls = getattr(tk.formats, 'Q' + topology.upper())
    factors = [torch.ones(rank, dtype=plain.dtype) for rank in plain.rank]
    if plain.out_dim is None:
        format = cls(plain.cores, 3, layout=layout, coordinate_map=coordinate_map,
                     bonds=factors)
    else:
        format = cls(plain.cores, 3, 3, in_layout=layout, out_layout=layout,
                     in_coordinate_map=coordinate_map, out_coordinate_map=coordinate_map,
                     bonds=factors)
    assert format.to() is format
    copied = format.clone()
    detached = format.detach()
    assert copied is not format and detached is not format
    assert copied.bonds is not format.bonds
    assert torch.allclose((format + copied).contract_dense(), 2 * plain.contract_dense())
    assert torch.allclose((format + detached).contract_dense(), 2 * plain.contract_dense())
    converted = format.to(dtype=torch.complex128)
    copied_map = copied.coordinate_map if plain.out_dim is None else copied.in_coordinate_map
    converted_map = converted.coordinate_map if plain.out_dim is None else converted.in_coordinate_map
    tensors = ('grid_coordinates' if map_kind == 'explicit' else 'domain')
    for original, clone, complex_view in zip(getattr(coordinate_map, tensors),
                                             getattr(copied_map, tensors),
                                             getattr(converted_map, tensors)):
        assert original.data_ptr() != clone.data_ptr()
        assert not complex_view.is_complex() and complex_view.dtype == torch.float64
    wrong_map = tk.formats.ExplicitGridMap(
        tuple(torch.tensor([0., 2.]) for _ in range(3)))
    if plain.out_dim is None:
        wrong = cls(plain.cores, 3, layout=layout, coordinate_map=wrong_map)
    else:
        wrong = cls(plain.cores, 3, 3, in_layout=layout, out_layout=layout,
                    in_coordinate_map=wrong_map, out_coordinate_map=wrong_map)
    with pytest.raises(ValueError, match='coordinate maps'):
        format + wrong
    if plain.out_dim is not None:
        with pytest.raises(ValueError, match='coordinate spaces'):
            format @ wrong


@pytest.mark.parametrize('cyclic', [False, True])
def test_exponential_at_float32_domain_grid_points(cyclic):
    coordinate_map = tk.formats.AffineCoordinateMap(torch.tensor([-1., 1.]), (5,))
    layout = tk.formats.QuantizedLayout(1, base=5)
    indices = torch.arange(5).reshape(-1, 1)
    coordinates = coordinate_map.from_indices(indices)
    values = coordinates[:, 0].double().exp()
    cls = tk.formats.QTR if cyclic else tk.formats.QTT
    format = cls([values.reshape(1, 5, 1) if cyclic else values], 1,
                 layout=layout, coordinate_map=coordinate_map)
    assert torch.allclose(format.evaluate_indices(indices), values)
    assert torch.allclose(format.evaluate_coordinates(coordinates), values)
