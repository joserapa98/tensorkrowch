"""Raw-core algebra compared with independent dense vector/matrix products."""

import pytest
import torch
import tensorkrowch as tk


def _matrix(format):
    dense = format.contract_dense()
    n, b = format.n_sites, format.n_batches
    order = [*range(b), *range(b + 1, b + 2 * n, 2),
             *range(b, b + 2 * n, 2)]
    return dense.permute(order).reshape(*format.batch_shape,
                                       int(torch.tensor(format.out_dim).prod()),
                                       int(torch.tensor(format.in_dim).prod()))


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
    right = y.T @ a
    assert torch.allclose(right.contract_dense().flatten(), (matrix @ vector) @ matrix)
    assert torch.allclose(y.apply(a).contract_dense(), right.T.contract_dense())
    assert torch.allclose((y.H @ a).contract_dense().flatten(),
                          (matrix @ vector).conj() @ matrix)
    assert torch.allclose(_matrix(a @ a.H), matrix @ matrix.adjoint())
    assert torch.allclose(_matrix(a.T), matrix.T)
    assert torch.allclose(_matrix(a.H), matrix.adjoint())
    assert torch.allclose((a @ a.H).trace(), torch.trace(matrix @ matrix.adjoint()))
    assert torch.allclose(((a @ a.H) @ a).contract_dense(),
                          (a @ (a.H @ a)).contract_dense())


@pytest.mark.parametrize('left_topology', ['tt', 'tr'])
@pytest.mark.parametrize('right_topology', ['tt', 'tr'])
@pytest.mark.parametrize('n_sites', [1, 3])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_kets_rows_and_outer_products(make_format, left_topology,
                                     right_topology, n_sites, dtype):
    x = make_format(left_topology, n_sites, dtype=dtype)
    y = make_format(right_topology, n_sites, dtype=dtype) * (1 + 2j)
    x_dense = x.contract_dense().flatten()
    y_dense = y.contract_dense().flatten()
    x_dense = x_dense.to(y_dense.dtype)

    assert torch.allclose(x.T @ y, torch.dot(x_dense, y_dense))
    assert torch.allclose(x.H @ y, torch.vdot(x_dense, y_dense))
    assert type(x.T) is type(x)
    assert type(x.H) is type(x)
    assert torch.equal(x.T.T.contract_dense(), x.contract_dense())
    assert torch.equal(x.H.H.contract_dense(), x.contract_dense())
    assert torch.equal(x.T.contract_dense(), x.contract_dense())
    assert torch.equal(x.H.contract_dense(), x.contract_dense().conj())

    outer = x @ y.H
    assert torch.allclose(_matrix(outer), torch.outer(x_dense, y_dense.conj()))
    assert torch.allclose(_matrix(x @ y.T), torch.outer(x_dense, y_dense))
    cyclic = left_topology == 'tr' or right_topology == 'tr'
    assert isinstance(outer, tk.formats.TRM if cyclic else tk.formats.TTM)


def test_rectangular_outer_product_and_energy():
    x = tk.formats.TT([torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex128)])
    y = tk.formats.TT([torch.tensor([2 - 1j, 4j, 1], dtype=torch.complex128)])
    outer = x @ y.H
    assert outer.in_dim == (3,)
    assert outer.out_dim == (2,)
    assert torch.allclose(_matrix(outer),
                          torch.outer(x.contract_dense(), y.contract_dense().conj()))
    a = tk.formats.TTM([torch.diag(torch.tensor([2., 5.], dtype=torch.complex128))])
    dense = x.contract_dense()
    energy = (x.H @ a @ x) / (x.H @ x)
    expected = torch.vdot(dense, _matrix(a) @ dense) / torch.vdot(dense, dense)
    assert torch.allclose(energy, expected)

    for left, right in [(x, x), (x, a), (x.T, x.H), (a, x.H)]:
        with pytest.raises(TypeError):
            left @ right
    assert (x.H @ (x * 0)).item() == 0


@pytest.mark.parametrize('topology', ['tt', 'tr'])
def test_vector_rows_keep_format_operations_and_own_containers(make_format, topology):
    x = make_format(topology, dtype=torch.complex128)
    count = x.n_sites if x._cyclic else x.n_sites - 1
    x.bonds = [torch.ones(rank, dtype=torch.float64) for rank in x.rank[:count]]
    dense = x.contract_dense()
    row = x.H
    assert row.cores is not x.cores
    assert row.bonds is not x.bonds
    assert row.cores[0].data_ptr() == x.cores[0].data_ptr()
    assert torch.allclose(row.norm(), dense.norm())

    rows = [row.clone(), row.detach(), row.to(copy=True), row.conj(),
            row * 2, row + row, row * row,
            row.clone().canonicalize(orth_center=1), row.clone().rounding()]
    rows.append(row.to_tt() if x._cyclic else row.clone().canonicalize_minimal())
    for transformed in rows:
        assert torch.allclose(transformed @ x,
                              torch.dot(transformed.contract_dense().flatten(), dense.flatten()))
    with pytest.raises(ValueError, match='orientation'):
        x + row
    with pytest.raises(ValueError, match='orientation'):
        x * row

    with pytest.raises(ValueError):
        row.cores[0] = torch.ones(1, dtype=x.dtype)
    with pytest.raises(ValueError, match='factor dimensions'):
        row.bonds.factors[0] = torch.ones(1)
    assert torch.allclose(row.contract_dense(), dense.conj())

    row.cores[0] = row.cores[0] * 2
    row.bonds.factors[0] = row.bonds.factors[0] * 3
    assert torch.allclose(row.contract_dense(), 6 * dense.conj())
    assert torch.equal(x.contract_dense(), dense)


def test_outer_product_structural_batches(make_format):
    x = make_format('tr', n_batches=1)
    y = make_format('tt')
    outer = x @ y.H
    dense_x, dense_y = x.contract_dense().reshape(2, -1), y.contract_dense().flatten()
    assert outer.n_batches == 1
    assert torch.allclose(_matrix(outer), dense_x.unsqueeze(-1) * dense_y.unsqueeze(0))
    x = make_format('tt', n_batches=1)
    outer = x @ y.T
    dense_x = x.contract_dense().reshape(2, -1)
    assert outer.topology == 'ttm'
    assert outer.n_batches == 1
    assert torch.allclose(_matrix(outer), dense_x.unsqueeze(-1) * dense_y.unsqueeze(0))


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 3])
def test_core_views_and_factored_products(make_format, topology, n_sites):
    format = make_format(topology, n_sites, dtype=torch.complex128)
    count = n_sites if format._cyclic else n_sites - 1
    format.bonds = [torch.full((format.rank[site],), 2 + 1j) for site in range(count)]
    standard, effective, operator = (
        format._standard_cores(), format._effective_cores(), format._operator_cores())
    for site in range(n_sites):
        expected = standard[site] * (2 + 1j) if site < count else standard[site]
        assert torch.equal(effective[site], expected)
        assert operator[site].shape[-4] == standard[site].shape[-3]
        assert operator[site].shape[-2] == standard[site].shape[-1]
    if topology.endswith('m'):
        assert torch.allclose(_matrix(format @ format.H),
                              _matrix(format) @ _matrix(format).adjoint())
    else:
        dense = format.contract_dense().flatten()
        assert torch.allclose(_matrix(format @ format.H), torch.outer(dense, dense.conj()))


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
    a.bonds = [
        torch.linspace(1, 2, a.rank[site], dtype=torch.float64) * (1 + 0.1j)
        for site in range(count)]
    dense = a.contract_dense()
    assert torch.allclose(a.norm(), dense.norm())
    assert torch.allclose((a * a).contract_dense(), dense.square())
    assert torch.allclose(a.conj().contract_dense(), dense.conj())
    assert torch.allclose(a.clone().materialize_bonds(orth_center=1).contract_dense(), dense)
    copied = a.clone()
    assert copied.bonds.factors[0].data_ptr() != a.bonds.factors[0].data_ptr()
    with pytest.raises(ValueError, match='factor dimensions'):
        a.bonds.factors[0] = torch.ones(1)
    assert torch.allclose(a.contract_dense(), dense)
