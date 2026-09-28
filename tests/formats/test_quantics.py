"""Physical-coordinate meaning and inherited numerical Quantics operations."""

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_quantized_grid_and_arithmetic(ordering):
    layout = tk.formats.QuantizedLayout(2, 2, (3, 4), ordering=ordering)
    indices = torch.cartesian_prod(torch.arange(8), torch.arange(16))
    digits = layout.encode_indices(indices)
    values = (indices[:, 0] + 2 * indices[:, 1]).to(torch.float64)
    tensor = torch.empty(layout.in_dim, dtype=torch.float64)
    tensor[tuple(digits.T)] = values
    cores = tk.decompositions.tt_svd(tensor, out_device=None)
    format = tk.formats.QTT(cores, layout,
                                            tk.formats.UniformCoordinateMap(),
                                            domain=torch.tensor([[0., 7.], [0., 15.]]))
    assert torch.allclose(format.evaluate_indices(indices), values)
    assert torch.allclose(format.evaluate_points(indices.to(torch.float64)), values)
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


def test_ring_coordinate_rotation_and_conversion():
    layout = tk.formats.QuantizedLayout(1, 2, 3)
    generator = torch.Generator().manual_seed(4)
    cores = [torch.randn(2, 2, 2, dtype=torch.float64, generator=generator) for _ in range(3)]
    format = tk.formats.QTR(cores, layout)
    indices = torch.arange(8).reshape(-1, 1)
    values = format.evaluate_indices(indices)
    assert torch.allclose(format.to_tt().evaluate_indices(indices), values)
    for first in range(3):
        assert torch.allclose(format.rotate(first).evaluate_indices(indices), values)


def test_quantized_output_sites():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    generator = torch.Generator().manual_seed(5)
    dense = torch.randn(2, 3, 2, dtype=torch.float64, generator=generator)
    cores = tk.decompositions.tt_svd(dense, out_device=None)
    format = tk.formats.QTT(cores, layout, digit_positions=(0, 2))
    indices = torch.arange(4).reshape(-1, 1)
    digits = layout.encode_indices(indices)
    assert torch.allclose(format.evaluate_indices(indices), dense[digits[:, 0], :, digits[:, 1]])
    assert format.to_dense_grid().shape == (4, 3)


def test_quantized_matrix_transpose_apply():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    generator = torch.Generator().manual_seed(6)
    dense = torch.randn(2, 2, 2, 2, dtype=torch.complex128, generator=generator)
    matrix = tk.formats.QTTM(
        tk.decompositions.ttm_svd(dense, out_device=None), layout, layout)
    vector = tk.formats.QTT(
        [torch.ones(2, 1, dtype=dense.dtype), torch.ones(1, 2, dtype=dense.dtype)], layout)
    assert isinstance(matrix @ vector, tk.formats.QTT)
    assert isinstance(matrix @ matrix.H, tk.formats.QTTM)
    assert matrix.T.in_layout == matrix.out_layout
    assert torch.allclose(matrix.H.H.contract_dense(), dense)
    assert isinstance(matrix.apply(torch.tensor([[0, 0]])), tk.formats.QTT)


@pytest.mark.parametrize('operation', ['sum', 'scale', 'hadamard', 'apply', 'transpose'])
def test_quantics_constructs_results_directly(operation, monkeypatch):
    from tensorkrowch.formats.formats1d import TensorFormat1D

    layout = tk.formats.QuantizedLayout(1, 2, 2)
    matrix = tk.formats.QTTM([torch.eye(2).unsqueeze(1), torch.eye(2).unsqueeze(0)],
                            layout, layout)
    vector = tk.formats.QTT([torch.ones(2, 1), torch.ones(1, 2)], layout)
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
    assert len(calls) == 1
    assert isinstance(result, tk.formats.QTTM if operation == 'transpose' else tk.formats.QTT)


def test_plain_operands_reject_quantics_in_both_orders():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    quantics = tk.formats.QTT([torch.ones(2, 1), torch.ones(1, 2)], layout)
    plain = quantics.as_tt()
    for first, second in [(plain, quantics), (quantics, plain)]:
        with pytest.raises(ValueError, match='semantics'):
            first + second
        with pytest.raises(ValueError, match='semantics'):
            first * second


def test_quantics_blocking_preserves_coordinate_contract():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    format = tk.formats.QTT([torch.ones(2, 1), torch.ones(1, 2)], layout)
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
        format = tk.formats.QTR([torch.ones(2, 2, 2)] * 2, layout)
        plain = format.as_tr
    else:
        format = tk.formats.QTT([torch.eye(2), torch.eye(2)], layout)
        plain = format.as_tt
    format.bonds = tk.formats.BondFactors1D([torch.ones(2)] * (2 if cyclic else 1))
    converted = plain()
    assert converted.bonds is not format.bonds
    converted.bonds.values[0] = torch.full((2,), 2.)
    assert torch.equal(format.bonds.values[0], torch.ones(2))
    with pytest.raises(ValueError, match='factor dimensions'):
        converted.bonds.values[0] = torch.ones(3)
    assert torch.equal(converted.bonds.values[0], torch.full((2,), 2.))
