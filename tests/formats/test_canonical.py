"""QR gauges preserve tensors and produce the expected open-chain isometries."""

import pytest
import torch


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('renormalize', [False, True])
def test_qr_gauges(make_format, topology, n_sites, renormalize, monkeypatch):
    network = make_format(topology, n_sites, dtype=torch.complex128)
    dense = network.contract_dense()

    def unexpected_svd(*args, **kwargs):
        raise AssertionError('QR canonicalization should not use SVD')

    monkeypatch.setattr(torch.linalg, 'svd', unexpected_svd)
    for oc in range(n_sites):
        result = network.copy().canonicalize(oc, renormalize)
        assert torch.allclose(result.contract_dense(), dense, rtol=1e-10, atol=1e-12)
        for site, core in enumerate(result._standard_cores()):
            if site < oc:
                matrix = core.reshape(-1, core.shape[-1])
                assert torch.allclose(matrix.adjoint() @ matrix,
                                      torch.eye(matrix.shape[-1], dtype=matrix.dtype))
            elif site > oc:
                matrix = core.reshape(core.shape[0], -1)
                assert torch.allclose(matrix @ matrix.adjoint(),
                                      torch.eye(matrix.shape[0], dtype=matrix.dtype))


def test_invalid_qr_and_zero(make_format):
    network = make_format()
    for oc in [-1, 3]:
        with pytest.raises(ValueError):
            network.canonicalize(oc)
    with pytest.raises(TypeError):
        network.canonicalize(True)
    with pytest.raises(TypeError):
        network.canonicalize(renormalize=1)
    network.cores[0] = network.cores[0] * 0
    network.canonicalize(renormalize=True)
    assert network.norm() == 0
