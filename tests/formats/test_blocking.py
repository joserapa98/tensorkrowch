"""Block round trips and local solver-style updates retaining interfaces."""

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('groups', [(1, 1, 1, 1), (2, 2), (1, 3), (4,)])
def test_block_roundtrip(make_format, topology, groups):
    network = make_format(topology, 4, dtype=torch.complex128)
    count = 4 if topology.startswith('tr') else 3
    network.bonds = tk.formats.BondFactors([
        torch.arange(1, rank + 1, dtype=torch.float64) for rank in network.rank[:count]])
    blocked, info = network.block(groups, return_info=True)
    restored = blocked.unblock(info)
    assert torch.allclose(restored.contract_dense(), network.contract_dense())
    assert restored.in_dim == network.in_dim and restored.out_dim == network.out_dim


@pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse', 'left', 'right'])
def test_local_update(make_format, mode):
    network = make_format('tr', 4)
    block = network.contract_block(1, 2)
    replacement = network.split_block(block * 2, 1, 2, mode=mode)
    dense = network.contract_dense()
    network.replace_block(1, 2, replacement)
    assert torch.allclose(network.contract_dense(), 2 * dense)
    network.absorb_bond(1, 'right')
    assert torch.allclose(network.contract_dense(), 2 * dense)


def test_invalid_blocks(make_format):
    network = make_format()
    for groups in [[], [0, 3], [True, 2], [2]]:
        with pytest.raises(ValueError):
            network.block(groups)
    with pytest.raises(ValueError):
        network.unblock()
    with pytest.raises(ValueError):
        network.contract_block(2, 0)
    before = network.contract_dense()
    with pytest.raises(ValueError):
        network.replace_block(1, 1, [torch.ones(7, 3, 9)])
    assert torch.equal(network.contract_dense(), before)
