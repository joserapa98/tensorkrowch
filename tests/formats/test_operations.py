"""Raw-core algebra compared with independent dense vector/matrix products."""

import pytest
import torch
import tensorkrowch as tk


def _matrix(network):
    dense = network.contract_dense()
    n, b = network.n_sites, network.n_batches
    order = [*range(b), *range(b + 1, b + 2 * n, 2),
             *range(b, b + 2 * n, 2)]
    return dense.permute(order).reshape(*network.batch_shape,
                                       int(torch.tensor(network.out_dim).prod()),
                                       int(torch.tensor(network.in_dim).prod()))


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('method', ['stacked', 'block_diagonal'])
def test_exact_algebra(make_format, topology, n_sites, method):
    a = make_format(topology, n_sites, dtype=torch.complex128)
    b = a * (0.2 + 0.3j)
    dense_a, dense_b = a.contract_dense(), b.contract_dense()
    assert torch.allclose(a.add(b, method).contract_dense(), dense_a + dense_b)
    assert torch.allclose(a.sub(b, method).contract_dense(), dense_a - dense_b)
    assert torch.allclose((a * b).contract_dense(), dense_a * dense_b)
    assert torch.allclose((-a).contract_dense(), -dense_a)
    assert torch.equal(a.contract_dense(), dense_a)


@pytest.mark.parametrize('matrix_topology', ['ttm', 'trm'])
@pytest.mark.parametrize('vector_topology', ['tt', 'tr'])
@pytest.mark.parametrize('n_sites', [1, 2, 3])
def test_apply_and_matrix_products(make_format, matrix_topology, vector_topology, n_sites):
    a = make_format(matrix_topology, n_sites, dtype=torch.complex128)
    x = make_format(vector_topology, n_sites, dtype=torch.complex128)
    matrix, vector = _matrix(a), x.contract_dense().flatten()
    y = a @ x
    assert torch.allclose(y.contract_dense().flatten(), matrix @ vector)
    assert torch.allclose(a.apply(x).contract_dense(), y.contract_dense())
    assert y.topology == ('tr' if 'tr' in (matrix_topology, vector_topology) or
                         matrix_topology == 'trm' else 'tt')
    right = y @ a
    assert torch.allclose(right.contract_dense().flatten(), (matrix @ vector) @ matrix)
    assert torch.allclose(_matrix(a @ a.H), matrix @ matrix.adjoint())
    assert torch.allclose(_matrix(a.T), matrix.T)
    assert torch.allclose(_matrix(a.H), matrix.adjoint())
    assert torch.allclose((a @ a.H).trace(), torch.trace(matrix @ matrix.adjoint()))
    assert torch.allclose(((a @ a.H) @ a).contract_dense(),
                          (a @ (a.H @ a)).contract_dense())


def test_mixed_sum_batches_and_errors(make_format):
    a, b = make_format('tt'), make_format('tr', n_batches=1)
    assert torch.allclose((a + b).contract_dense(), a.contract_dense() + b.contract_dense())
    with pytest.raises(ValueError):
        a.add(b, method='unknown')
    with pytest.raises(ValueError):
        a + make_format('ttm')
    with pytest.raises(TypeError):
        a @ a
    with pytest.raises(ValueError):
        a * torch.ones(2)
    with pytest.raises(ValueError):
        make_format('ttm').trace()


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_factored_operations(make_format, topology):
    a = make_format(topology, dtype=torch.complex128)
    count = a.n_sites if topology.startswith('tr') else a.n_sites - 1
    a.bonds = tk.formats.BondFactors([
        torch.linspace(1, 2, a.rank[site], dtype=torch.float64) * (1 + 0.1j)
        for site in range(count)])
    dense = a.contract_dense()
    assert torch.allclose(a.norm(), dense.norm())
    assert torch.allclose((a * a).contract_dense(), dense.square())
    assert torch.allclose(a.conj().contract_dense(), dense.conj())
    assert torch.allclose(a.clone().materialize_bonds(oc=1).contract_dense(), dense)
    copied = a.clone()
    assert copied.bonds.values[0].data_ptr() != a.bonds.values[0].data_ptr()
    a.bonds.values[0] = torch.ones(1)
    with pytest.raises(ValueError, match='factor dimensions'):
        a.norm()
