"""Physical-coordinate meaning and inherited numerical Quantics operations."""

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('matrix', [False, True])
@pytest.mark.parametrize('grid, index', [
    ('left', 1), ('centers', 0), ('right', 0), (0.25, 1),
])
def test_shorthand_uses_the_selected_grid(cyclic, matrix, grid, index):
    domain = torch.tensor([0., 1.])
    coordinates = torch.tensor([[0.25]])
    if matrix:
        core = torch.arange(16.).reshape(4, 4)
        cls = tk.formats.QTRM if cyclic else tk.formats.QTTM
        format = cls(
            [core.reshape(1, 4, 1, 4) if cyclic else core], 1, 1,
            in_base=4, in_level=1, in_domain=domain,
            out_base=4, out_level=1, out_domain=domain,
            computational_grid=grid)
        actual = format.evaluate_coordinates(coordinates, coordinates)
        expected = core[index, index]
    else:
        core = torch.arange(4.)
        cls = tk.formats.QTR if cyclic else tk.formats.QTT
        format = cls(
            [core.reshape(1, 4, 1) if cyclic else core], 1,
            base=4, level=1, domain=domain, computational_grid=grid)
        actual = format.evaluate_coordinates(coordinates)
        expected = core[index]

    assert actual.item() == expected.item()


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('explicit_data', [False, True])
def test_error_requires_scalar_quantics_samples(cyclic, explicit_data):
    layout = tk.formats.QuantizedLayout(1, 2, 1)
    format = (tk.formats.QTR([torch.ones(1, 2, 1)], 1, layout=layout)
              if cyclic else tk.formats.QTT([torch.ones(2)], 1,
                                            layout=layout))
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

    with pytest.raises(ValueError, match='Explicit grid size'):
        tk.formats.QTT(cores, 1, layout=layout,
                       coordinate_map=coordinate_map)


def test_vector_shorthand_builds_uniform_and_explicit_coordinates():
    cores = [torch.ones(2, 1), torch.ones(1, 2)]
    uniform = tk.formats.QTT(
        cores, 1, base=2, level=2, domain=torch.tensor([0., 3.]))
    explicit = tk.formats.QTT(
        cores, 1, grid_coordinates=torch.tensor([0., 1., 4., 10.]),
        base=2)

    assert uniform.layout.grid_size == explicit.layout.grid_size == (4,)
    assert isinstance(uniform.coordinate_map, tk.formats.UniformCoordinateMap)
    assert isinstance(explicit.coordinate_map, tk.formats.ExplicitGridMap)
    assert torch.equal(uniform.evaluate_coordinates(torch.tensor([[3.]])),
                       explicit.evaluate_coordinates(torch.tensor([[10.]])))

    with pytest.raises(ValueError, match='Do not combine'):
        tk.formats.QTT(cores, 1, layout=uniform.layout, base=2)
    with pytest.raises(ValueError, match='domain'):
        tk.formats.QTT(cores, 1, grid_coordinates=torch.tensor([0., 1., 2., 3.]),
                       base=2, domain=torch.tensor([0., 3.]))


def test_matrix_shorthand_resolves_input_and_output_separately():
    matrix = tk.formats.QTTM(
        [torch.ones(2, 3)], 1, 1,
        in_base=2, in_level=1, in_domain=torch.tensor([0., 1.]),
        out_grid_coordinates=torch.tensor([0., 2., 5.]), out_base=3)

    assert isinstance(matrix.in_coordinate_map, tk.formats.UniformCoordinateMap)
    assert isinstance(matrix.out_coordinate_map, tk.formats.ExplicitGridMap)
    assert matrix.evaluate_coordinates(
        torch.tensor([[1.]]), torch.tensor([[5.]])).item() == 1

    with pytest.raises(ValueError, match='equal `n_sites`'):
        tk.formats.QTTM([torch.ones(2, 3)], 1, 1,
                        in_base=2, in_level=2, out_base=3, out_level=1)


def test_ring_shorthand_uses_the_same_coordinate_resolution():
    vector = tk.formats.QTR(
        [torch.ones(1, 2, 1)], 1,
        base=2, level=1, domain=torch.tensor([0., 1.]))
    matrix = tk.formats.QTRM(
        [torch.ones(1, 2, 1, 3)], 1, 1,
        in_base=2, in_level=1,
        out_grid_coordinates=torch.tensor([0., 2., 5.]), out_base=3)

    assert vector.evaluate_coordinates(torch.tensor([[1.]])).item() == 1
    assert matrix.evaluate_indices(torch.tensor([[1]]),
                                   torch.tensor([[2]])).item() == 1


def test_matrix_coordinate_counts_can_differ_with_equal_site_counts():
    matrix = tk.formats.QTTM(
        [torch.ones(2, 1, 2), torch.ones(1, 2, 3)], 1, 2,
        in_base=2, in_level=2,
        out_base=(2, 3), out_level=1)

    assert matrix.in_layout.n_sites == matrix.out_layout.n_sites == 2
    assert matrix.to_dense_grid().shape == (4, 2, 3)


@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_quantized_grid_and_arithmetic(ordering):
    layout = tk.formats.QuantizedLayout(2, 2, (3, 4), ordering=ordering)
    indices = torch.cartesian_prod(torch.arange(8), torch.arange(16))
    digits = layout.encode_indices(indices)
    values = (indices[:, 0] + 2 * indices[:, 1]).to(torch.float64)
    tensor = torch.empty(layout.in_dim, dtype=torch.float64)
    tensor[tuple(digits.T)] = values
    cores = tk.decompositions.tt_svd(tensor, out_device=None)
    format = tk.formats.QTT(
        cores, 2, layout=layout,
        coordinate_map=tk.formats.UniformCoordinateMap(),
        domain=torch.tensor([[0., 7.], [0., 15.]]))
    assert torch.allclose(format.evaluate_indices(indices), values)
    assert torch.allclose(format.evaluate_coordinates(indices.to(torch.float64)), values)
    assert torch.allclose(format.to_dense_grid(), values.reshape(8, 16))
    assert isinstance(format + format, tk.formats.QTT)
    assert torch.allclose((format * format).evaluate_indices(indices), values.square())
    assert torch.allclose((2 * format).to_dense_grid(), 2 * values.reshape(8, 16))
    assert torch.allclose(format.as_tt().contract_dense(), tensor)
    with pytest.raises(ValueError, match='semantics'):
        format + format.as_tt()
    detached = format.clone().detach()
    assert detached.domain.data_ptr() != format.domain.data_ptr()
    complex_format = format.to(dtype=torch.complex128)
    assert not complex_format.domain.is_complex()
    row = format.H
    assert isinstance(row, tk.formats.QTT)
    assert row.layout is layout
    assert torch.allclose(row.as_tt() @ format.as_tt(), tensor.square().sum())


def test_ring_coordinate_rotation_and_conversion():
    layout = tk.formats.QuantizedLayout(1, 2, 3)
    generator = torch.Generator().manual_seed(4)
    cores = [torch.randn(2, 2, 2, dtype=torch.float64, generator=generator) for _ in range(3)]
    format = tk.formats.QTR(cores, layout.n_coordinates, layout=layout)
    indices = torch.arange(8).reshape(-1, 1)
    values = format.evaluate_indices(indices)
    assert torch.allclose(format.to_tt().evaluate_indices(indices), values)
    assert torch.allclose(format.H.to_tt() @ format.to_tt(), values.square().sum())
    for first in range(3):
        assert torch.allclose(format.rotate(first).evaluate_indices(indices), values)
        assert torch.allclose(format.H.rotate(first) @ format.rotate(first), values.square().sum())


def test_quantized_output_sites():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    generator = torch.Generator().manual_seed(5)
    dense = torch.randn(2, 3, 2, dtype=torch.float64, generator=generator)
    cores = tk.decompositions.tt_svd(dense, out_device=None)
    format = tk.formats.QTT(cores, 1, layout=layout,
                            digit_positions=(0, 2))
    indices = torch.arange(4).reshape(-1, 1)
    digits = layout.encode_indices(indices)
    assert torch.allclose(format.evaluate_indices(indices), dense[digits[:, 0], :, digits[:, 1]])
    assert format.to_dense_grid().shape == (4, 3)


def test_quantized_matrix_transpose_apply():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    generator = torch.Generator().manual_seed(6)
    dense = torch.randn(2, 2, 2, 2, dtype=torch.complex128, generator=generator)
    matrix = tk.formats.QTTM(
        tk.decompositions.ttm_svd(dense, out_device=None), 1, 1,
        in_layout=layout, out_layout=layout)
    vector = tk.formats.QTT(
        [torch.ones(2, 1, dtype=dense.dtype),
         torch.ones(1, 2, dtype=dense.dtype)], 1, layout=layout)
    assert isinstance(matrix @ vector, tk.formats.QTT)
    assert isinstance(matrix @ matrix.H, tk.formats.QTTM)
    assert matrix.T.in_layout == matrix.out_layout
    assert torch.allclose(matrix.H.H.contract_dense(), dense)
    assert isinstance(matrix.apply(torch.tensor([[0, 0]])), tk.formats.QTT)


def test_quantized_outer_product_preserves_coordinate_spaces():
    output_layout = tk.formats.QuantizedLayout(1, 2, 1)
    input_layout = tk.formats.QuantizedLayout(1, 3, 1)
    coordinate_map = tk.formats.UniformCoordinateMap()
    x = tk.formats.QTT(
        [torch.tensor([1., 2.])], 1, layout=output_layout,
        coordinate_map=coordinate_map, domain=torch.tensor([[0., 1.]]))
    y = tk.formats.QTT(
        [torch.tensor([2., 3., 4.])], 1, layout=input_layout,
        coordinate_map=coordinate_map, domain=torch.tensor([[-1., 1.]]))
    outer = x @ y.H
    assert isinstance(outer, tk.formats.QTTM)
    assert outer.in_layout is input_layout
    assert outer.out_layout is output_layout
    assert outer.in_domain is y.domain
    assert outer.out_domain is x.domain
    assert torch.equal(outer.contract_dense(), torch.outer(
        y.contract_dense(), x.contract_dense()))
    assert torch.allclose((x.H @ outer).T.contract_dense(),
                          (x.H @ x) * y.contract_dense())
    with pytest.raises(ValueError, match='layouts'):
        x.H @ tk.formats.QTT(x.cores, 1, layout=output_layout)
    with pytest.raises(ValueError, match='semantics'):
        x @ y.as_tt().H


def test_quantized_outer_product_rejects_extra_output_sites():
    layout = tk.formats.QuantizedLayout(1, 2, 1)
    x = tk.formats.QTT(
        [torch.ones(2, 1), torch.ones(1, 3)], 1, layout=layout,
        digit_positions=(0,))
    with pytest.raises(ValueError, match='only digit sites'):
        x @ x.T


@pytest.mark.parametrize('operation', ['sum', 'scale', 'hadamard', 'apply', 'transpose'])
def test_quantics_constructs_results_directly(operation, monkeypatch):
    from tensorkrowch.formats.formats1d import TensorFormat1D

    layout = tk.formats.QuantizedLayout(1, 2, 2)
    matrix = tk.formats.QTTM(
        [torch.eye(2).unsqueeze(1), torch.eye(2).unsqueeze(0)], 1, 1,
        in_layout=layout, out_layout=layout)
    vector = tk.formats.QTT([torch.ones(2, 1), torch.ones(1, 2)], 1,
                            layout=layout)
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
                              layout=layout)
    plain = quantics.as_tt()
    for first, second in [(plain, quantics), (quantics, plain)]:
        with pytest.raises(ValueError, match='semantics'):
            first + second
        with pytest.raises(ValueError, match='semantics'):
            first * second


def test_quantics_blocking_preserves_coordinate_contract():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    format = tk.formats.QTT([torch.ones(2, 1), torch.ones(1, 2)], 1,
                            layout=layout)
    cores = format.cores
    with pytest.raises(ValueError, match='layout'):
        format.block([2])
    assert format.cores is cores
    assert format.layout is layout
    assert format.in_dim == (2, 2)
    plain = format.as_tt()
    grouped = plain.block([2])
    plain.unblock(grouped)
    assert torch.allclose(plain.contract_dense(), format.contract_dense())


@pytest.mark.parametrize('cyclic', [False, True])
def test_plain_conversion_owns_its_bonds(cyclic):
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    if cyclic:
        format = tk.formats.QTR([torch.ones(2, 2, 2)] * 2, 1,
                                layout=layout)
        plain = format.as_tr
    else:
        format = tk.formats.QTT([torch.eye(2), torch.eye(2)], 1,
                                layout=layout)
        plain = format.as_tt
    format.bonds = [torch.ones(2)] * (2 if cyclic else 1)
    converted = plain()
    assert converted.bonds is not format.bonds
    converted.bonds.factors[0] = torch.full((2,), 2.)
    assert torch.equal(format.bonds.factors[0], torch.ones(2))
    with pytest.raises(ValueError, match='factor dimensions'):
        converted.bonds.factors[0] = torch.ones(3)
    assert torch.equal(converted.bonds.factors[0], torch.full((2,), 2.))
