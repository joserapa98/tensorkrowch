"""Public format exports and decomposition-result parity."""

import pytest
import torch

import tensorkrowch as tk


# Public API parity


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_result_parity(make_format, topology, n_sites, dtype):
    format = make_format(topology, n_sites, dtype=dtype)
    result_class = {'tt': tk.decompositions.TTDecomposition,
                    'tr': tk.decompositions.TRDecomposition,
                    'ttm': tk.decompositions.TTMDecomposition,
                    'trm': tk.decompositions.TRMDecomposition}[topology]
    result = result_class(format.cores)
    dense = result.contract_dense()
    assert torch.allclose(format.contract_dense(), dense)
    assert torch.allclose(format.norm(), dense.norm())
    assert torch.allclose(format.inner(format),
                          dense.abs().square().sum().to(dtype))
    assert torch.allclose(format.fidelity(format), dense.real.new_ones(()))
    inputs = torch.zeros(5, n_sites, dtype=torch.long)
    if format.out_dim is None:
        assert torch.allclose(format.evaluate(inputs), result.evaluate(inputs))
    else:
        outputs = torch.ones_like(inputs)
        assert torch.allclose(format.evaluate(inputs, outputs),
                              result.evaluate(inputs, outputs))
        assert torch.allclose(format.apply(inputs).contract_dense(),
                              result.apply(inputs).contract_dense())
