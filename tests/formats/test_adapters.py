"""Exact ring rotations and identity-carried conversions."""

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('topology', ['tr', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('n_batches', [0, 1])
def test_rotation_and_train_conversion(make_format, topology, n_sites, n_batches):
    network = make_format(topology, n_sites, n_batches, torch.complex128)
    network.bonds = tk.formats.BondFactors([
        torch.arange(1, rank + 1, dtype=torch.float64) for rank in network.rank])
    dense = network.contract_dense()
    b, width = n_batches, 2 if topology == 'trm' else 1
    for first in range(n_sites):
        rotated = network.rotate(first)
        axes = [*range(b), *range(b + first * width, b + n_sites * width),
                *range(b, b + first * width)]
        assert torch.allclose(rotated.contract_dense(), dense.permute(axes))
        if topology == 'trm' and n_batches:
            with pytest.raises(ValueError, match='batches'):
                rotated.to_ttm()
            continue
        train = rotated.to_tt() if topology == 'tr' else rotated.to_ttm()
        # The initial TTM contract rejects structural batches explicitly.
        assert torch.allclose(train.contract_dense(), rotated.contract_dense())
        if n_sites > 1:
            closing = rotated.rank[-1]
            assert train.rank == [closing * rank for rank in rotated.rank[:-1]]


def test_invalid_rotation(make_format):
    for first in [-1, 3]:
        with pytest.raises(ValueError):
            make_format('tr').rotate(first)
    with pytest.raises(TypeError):
        make_format('tr').rotate(True)
