"""Two-level Quantics contraction and ownership compared with dense oracles."""

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('cyclic', [False, True])
def test_tucker_factor_contraction_and_clone(cyclic):
    generator = torch.Generator().manual_seed(20)
    dtype = torch.float64
    factor_values = [torch.randn(2, 2, 2, dtype=dtype, generator=generator) for _ in range(2)]
    factors = [tk.formats.TT(tk.decompositions.tt_svd(value, out_device=None))
               for value in factor_values]
    upper_dense = torch.randn(2, 3, 2, dtype=dtype, generator=generator)
    engine = tk.decompositions.tr_svd if cyclic else tk.decompositions.tt_svd
    upper_cls = tk.formats.TR if cyclic else tk.formats.TT
    upper = upper_cls(engine(upper_dense, out_device=None))
    cls = tk.formats.QTRTucker if cyclic else tk.formats.QTTTucker
    network = cls(upper, factors, tk.formats.QuantizedLayout(2, 2, 2),
                  variable_positions=(0, 2))
    indices = torch.cartesian_prod(torch.arange(4), torch.arange(4))
    expected = torch.einsum('ag,goh,bh->abo', factor_values[0].reshape(4, 2),
                             upper_dense, factor_values[1].reshape(4, 2))
    assert network.cores is network.upper.cores
    assert torch.allclose(network.evaluate_indices(indices), expected.reshape(16, 3))
    assert torch.allclose(network.flatten().evaluate_indices(indices), expected.reshape(16, 3))
    copied = network.clone()
    assert copied.upper.cores[0].data_ptr() != upper.cores[0].data_ptr()
    assert copied.factors[0].cores[0].data_ptr() != factors[0].cores[0].data_ptr()
    assert torch.allclose(network.norm(), expected.norm())
