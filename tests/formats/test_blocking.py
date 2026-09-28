"""Block round trips and local solver-style updates retaining interfaces."""

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('groups', [(1, 1, 1, 1), (2, 2), (1, 3), (4,)])
def test_block_roundtrip(make_format, topology, groups):
    format = make_format(topology, 4, dtype=torch.complex128)
    count = 4 if topology.startswith('tr') else 3
    format.bonds = tk.formats.BondFactors1D([
        torch.arange(1, rank + 1, dtype=torch.float64) for rank in format.rank[:count]])
    dense = format.contract_dense()
    in_dim, out_dim = format.in_dim, format.out_dim
    layout = format.block(groups)
    assert format.n_sites == len(groups)
    assert layout.in_dim == in_dim and layout.out_dim == out_dim
    assert format.unblock(layout) is format
    assert torch.allclose(format.contract_dense(), dense)
    assert format.in_dim == in_dim and format.out_dim == out_dim


@pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse', 'left', 'right'])
def test_local_update(make_format, mode):
    format = make_format('tr', 4)
    block = format.contract_block(1, 2)
    replacement = tk.formats.split_block(
        block * 2, format.in_dim[1:3], mode=mode)
    dense = format.contract_dense()
    format.replace_cores(1, replacement.cores, bonds=replacement.bonds)
    assert torch.allclose(format.contract_dense(), 2 * dense)
    format.absorb_bond(1, 'right')
    assert torch.allclose(format.contract_dense(), 2 * dense)


def test_invalid_blocks(make_format):
    format = make_format()
    for groups in [[], [0, 3], [True, 2], [2]]:
        with pytest.raises(ValueError):
            format.block(groups)
    with pytest.raises(ValueError):
        format.unblock(tk.formats.BlockLayout((2,), (2, 3)))
    with pytest.raises(ValueError):
        format.contract_block(2, 0)
    before = format.contract_dense()
    with pytest.raises(ValueError):
        format.replace_cores(1, [torch.ones(7, 3, 9)])
    assert torch.equal(format.contract_dense(), before)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
def test_layout_applies_to_new_solution(make_format, topology, n_sites):
    format = make_format(topology, n_sites, dtype=torch.complex128)
    dense = format.contract_dense()
    layout = format.block([n_sites])
    solution = type(format)([2 * core for core in format.cores])
    assert solution.unblock(layout) is solution
    assert torch.allclose(solution.contract_dense(), 2 * dense)
    assert format.n_sites == 1
    assert format.bonds is None


@pytest.mark.parametrize('topology', ['tt', 'tr', 'trm'])
@pytest.mark.parametrize('n_batches', [1, 2])
def test_batched_blocking(make_format, topology, n_batches):
    format = make_format(topology, 4, n_batches=n_batches)
    dense = format.contract_dense()
    layout = format.block([2, 2])
    assert format.batch_shape == (2,) * n_batches
    format.unblock(layout)
    assert torch.allclose(format.contract_dense(), dense)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_unblocking_truncates_only_internal_bonds(make_format, topology):
    format = make_format(topology, 4)
    layout = format.block([2, 2])
    boundary_rank = format.rank
    format.unblock(layout, rank=1)
    assert format.rank[0] == format.rank[2] == 1
    assert format.rank[1] == boundary_rank[0]
    if topology.startswith('tr'):
        assert format.rank[-1] == boundary_rank[-1]


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('failure', ['dimensions', 'family', 'svd'])
def test_unblocking_failure_preserves_state(make_format, topology, failure):
    format = make_format(topology, 4)
    layout = format.block([2, 2])
    if failure == 'dimensions':
        layout = tk.formats.BlockLayout(layout.groups, (3, 3, 2, 3), layout.out_dim)
    elif failure == 'family':
        outputs = layout.in_dim if format.out_dim is None else None
        layout = tk.formats.BlockLayout(layout.groups, layout.in_dim, outputs)
    cores = format.cores
    before = format.contract_dense()
    dimensions, rank = format.in_dim, format.rank
    with pytest.raises(ValueError):
        format.unblock(layout, mode='invalid' if failure == 'svd' else 'right')
    assert format.cores is cores
    assert format.in_dim == dimensions and format.rank == rank
    assert torch.equal(format.contract_dense(), before)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse', 'left', 'right'])
def test_coupled_core_and_bond_update(make_format, topology, mode, monkeypatch):
    format = make_format(topology, 4, dtype=torch.complex128)
    count = format.n_sites if topology.startswith('tr') else format.n_sites - 1
    format.bonds = tk.formats.BondFactors1D([
        torch.arange(1, rank + 1, dtype=torch.float64)
        for rank in format.rank[:count]])
    dense = format.contract_dense()
    factors = format.bonds.values
    block = format.contract_block(1, 2)
    outputs = None if format.out_dim is None else format.out_dim[1:3]
    replacement = tk.formats.split_block(
        block, format.in_dim[1:3], outputs, rank=1, mode=mode)
    calls = []
    validate = format.validate

    def counted_validate():
        calls.append(True)
        return validate()

    monkeypatch.setattr(format, 'validate', counted_validate)
    assert format.replace_cores(1, replacement.cores, bonds=replacement.bonds) is format
    assert len(calls) == 1
    assert format.rank[1] == 1
    assert format.bonds.values[0] is factors[0]
    assert format.bonds.values[2] is factors[2]
    assert format.contract_dense().shape == dense.shape
    with pytest.raises(ValueError, match='factor dimensions'):
        format.bonds.values[0] = torch.ones(100, dtype=format.dtype)
    assert format.bonds.values[0] is factors[0]
    left, right = replacement.cores
    factor = replacement.bonds.values[0]
    if factor is not None:
        left = left * factor[None, None, :]
    local = torch.einsum('aib,bjc->aijc', left, right).reshape(block.shape)
    assert torch.allclose(format.contract_block(1, 2), local)


@pytest.mark.parametrize('failure', ['rank', 'dtype', 'factor', 'batch'])
def test_replacement_failure_preserves_cores_and_bonds(make_format, failure):
    format = make_format('tr', 4)
    format.bonds = tk.formats.BondFactors1D([
        torch.ones(rank, dtype=format.dtype) for rank in format.rank])
    block = format.contract_block(1, 2)
    replacement = tk.formats.split_block(block, format.in_dim[1:3], mode='explicit')
    values = list(replacement.cores)
    bonds = replacement.bonds
    if failure == 'rank':
        values[0] = values[0][..., :-1]
    elif failure == 'dtype':
        values[0] = values[0].to(torch.float32)
    elif failure == 'factor':
        bonds = tk.formats.BondFactors1D([torch.ones(100)])
    else:
        values[0] = values[0].unsqueeze(0)
    cores, factors = format.cores, format.bonds
    dense = format.contract_dense()
    with pytest.raises(ValueError):
        format.replace_cores(1, values, bonds=bonds)
    assert format.cores is cores and format.bonds is factors
    with pytest.raises(ValueError, match='factor dimensions'):
        factors.values[0] = torch.ones(100, dtype=format.dtype)
    assert torch.equal(format.contract_dense(), dense)
