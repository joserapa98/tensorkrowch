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
    network = tk.formats.QTT(cores, layout,
                                            tk.formats.UniformCoordinateMap(),
                                            domain=torch.tensor([[0., 7.], [0., 15.]]))
    assert torch.allclose(network.evaluate_indices(indices), values)
    assert torch.allclose(network.evaluate_points(indices.to(torch.float64)), values)
    assert torch.allclose(network.to_dense_grid(), values.reshape(8, 16))
    assert isinstance(network + network, tk.formats.QTT)
    assert torch.allclose((network * network).evaluate_indices(indices), values.square())
    assert torch.allclose((2 * network).to_dense_grid(), 2 * values.reshape(8, 16))
    assert torch.allclose(network.as_tt().contract_dense(), tensor)
    with pytest.raises(ValueError, match='semantics'):
        network + network.as_tt()
    detached = network.clone().detach()
    assert detached.domain.data_ptr() != network.domain.data_ptr()
    complex_network = network.to(dtype=torch.complex128)
    assert not complex_network.domain.is_complex()


def test_ring_coordinate_rotation_and_conversion():
    layout = tk.formats.QuantizedLayout(1, 2, 3)
    generator = torch.Generator().manual_seed(4)
    cores = [torch.randn(2, 2, 2, dtype=torch.float64, generator=generator) for _ in range(3)]
    network = tk.formats.QTR(cores, layout)
    indices = torch.arange(8).reshape(-1, 1)
    values = network.evaluate_indices(indices)
    assert torch.allclose(network.to_tt().evaluate_indices(indices), values)
    for first in range(3):
        assert torch.allclose(network.rotate(first).evaluate_indices(indices), values)


def test_quantized_output_sites():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    generator = torch.Generator().manual_seed(5)
    dense = torch.randn(2, 3, 2, dtype=torch.float64, generator=generator)
    cores = tk.decompositions.tt_svd(dense, out_device=None)
    network = tk.formats.QTT(cores, layout, digit_positions=(0, 2))
    indices = torch.arange(4).reshape(-1, 1)
    digits = layout.encode_indices(indices)
    assert torch.allclose(network.evaluate_indices(indices), dense[digits[:, 0], :, digits[:, 1]])
    assert network.to_dense_grid().shape == (4, 3)


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
