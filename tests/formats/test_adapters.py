"""Exact ring rotations and identity-carried conversions."""

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('topology', ['tr', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('n_batches', [0, 1])
def test_rotation_and_train_conversion(make_format, topology, n_sites, n_batches):
    format = make_format(topology, n_sites, n_batches, torch.complex128)
    format.bonds = [
        torch.arange(1, rank + 1, dtype=torch.float64) for rank in format.rank]
    dense = format.contract_dense()
    b, width = n_batches, 2 if topology == 'trm' else 1
    for first in range(n_sites):
        rotated = format.rotate(first)
        axes = [*range(b), *range(b + first * width, b + n_sites * width),
                *range(b, b + first * width)]
        assert torch.allclose(rotated.contract_dense(), dense.permute(axes))
        train = rotated.to_tt() if topology == 'tr' else rotated.to_ttm()
        assert train.batch_shape == rotated.batch_shape
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


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
def test_model_roundtrip(make_format, topology, n_sites):
    format = make_format(topology, n_sites, dtype=torch.complex128)
    dense = format.contract_dense()
    if topology.endswith('m'):
        model = format.to_mpo()
        restored = model.to_trm() if topology == 'trm' else model.to_ttm()
    else:
        model = format.to_mps()
        restored = model.to_tr() if topology == 'tr' else model.to_tt()
    assert torch.allclose(restored.contract_dense(), dense)


@pytest.mark.parametrize('topology', ['tt', 'tr'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
def test_mps_data_roundtrip(make_format, topology, n_sites):
    format = make_format(topology, n_sites, n_batches=2)
    model = format.to_mps()
    assert isinstance(model, tk.models.MPSData)
    restored = model.to_tt() if topology == 'tt' else model.to_tr()
    assert restored.batch_shape == (2, 2)
    assert torch.allclose(restored.contract_dense(), format.contract_dense())
