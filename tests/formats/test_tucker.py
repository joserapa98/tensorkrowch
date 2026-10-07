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
    format = cls(upper, factors, tk.formats.QuantizedLayout(2, 2, 2),
                  coordinate_positions=(0, 2))
    indices = torch.cartesian_prod(torch.arange(4), torch.arange(4))
    expected = torch.einsum('ag,goh,bh->abo', factor_values[0].reshape(4, 2),
                             upper_dense, factor_values[1].reshape(4, 2))
    assert format.cores is format.upper.cores
    assert torch.allclose(format.evaluate_indices(indices), expected.reshape(16, 3))
    with pytest.raises(ValueError, match='tensor-valued Tucker'):
        format.flatten()
    copied = format.clone()
    assert copied.upper.cores[0].data_ptr() != upper.cores[0].data_ptr()
    assert copied.factors[0].cores[0].data_ptr() != factors[0].cores[0].data_ptr()
    assert torch.allclose(format.norm(), expected.norm())


@pytest.mark.parametrize('cyclic', [False, True])
def test_scalar_tucker_flattens_to_digit_only_quantics(cyclic):
    upper_cls = tk.formats.TR if cyclic else tk.formats.TT
    upper_core = torch.tensor([1., 2.])
    if cyclic:
        upper_core = upper_core.reshape(1, 2, 1)
    upper = upper_cls([upper_core])
    factor = tk.formats.TT([torch.eye(2), torch.eye(2)])
    cls = tk.formats.QTRTucker if cyclic else tk.formats.QTTTucker
    format = cls(upper, [factor], tk.formats.QuantizedLayout(1, 2, 1))
    flat = format.flatten()
    indices = torch.tensor([[0], [1]])
    assert flat.n_sites == flat.layout.n_sites
    assert torch.allclose(flat.evaluate_indices(indices), format.evaluate_indices(indices))
