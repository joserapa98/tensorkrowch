"""Rounding reconstruction, known spectra and cyclic compression limits."""

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('method', ['svd', 'qr_svd'])
def test_exact_rounding(make_format, topology, n_sites, method):
    format = make_format(topology, n_sites, dtype=torch.complex128)
    dense = format.contract_dense()
    with tk.svd_method(method, refine=True):
        result, info = format.rounding(return_info=True)
    assert result is format
    assert torch.allclose(format.contract_dense(), dense, rtol=1e-10, atol=1e-12)
    assert info.rank == tuple(format.rank)
    assert torch.all(info.error_bound == 0)


def test_known_spectrum_and_budget():
    diagonal = torch.diag(torch.tensor([4., 2., 1., 0.1], dtype=torch.float64))
    format = tk.formats.TT([diagonal, torch.eye(4, dtype=diagonal.dtype)])
    expected = diagonal.clone()
    expected[2:, 2:] = 0
    format.rounding(rank=2)
    assert torch.allclose(format.contract_dense(), expected)
    full = tk.formats.TT([diagonal, torch.eye(4, dtype=diagonal.dtype)])
    _, info = full.rounding(rel_error=0.03, return_info=True)
    assert info.bound_satisfied
    assert (full.contract_dense() - diagonal).norm() <= 0.03 * diagonal.norm()
    full = tk.formats.TT([diagonal, torch.eye(4, dtype=diagonal.dtype)])
    with pytest.warns(UserWarning, match='budget'):
        _, info = full.rounding(rank=1, rel_error=1e-5, return_info=True)
    assert not info.bound_satisfied


def test_stacked_cyclic_sum_compresses(make_format):
    format = make_format('tr', 4)
    summed = format + format
    _, info = summed.rounding(rel_error=1e-12, return_info=True)
    assert info.bound_satisfied
    assert torch.allclose(summed.contract_dense(), 2 * format.contract_dense())
    assert all(actual <= original for actual, original in zip(summed.rank, format.rank))
    usual = format.add(format, method='block_diagonal')
    usual.rounding(rel_error=1e-12)
    assert torch.allclose(usual.contract_dense(), 2 * format.contract_dense())
    assert usual.rank[-1] > summed.rank[-1]


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_rounding_extreme_scales(scale):
    diagonal = torch.diag(torch.tensor([4., 2., 1., 0.1], dtype=torch.float64)) * scale
    format = tk.formats.TT([diagonal, torch.eye(4, dtype=diagonal.dtype)])
    _, info = format.rounding(rtol=0.001, rel_error=0.03, return_info=True)
    assert format.rank == [3]
    assert info.bound_satisfied
    assert torch.isfinite(info.error_bound)
    assert torch.allclose(format.contract_dense() / scale,
                          torch.diag(torch.tensor([4., 2., 1., 0.], dtype=diagonal.dtype)))


@pytest.mark.parametrize('kwargs', [{'rank': 0}, {'rtol': -1}, {'rel_error': -1},
                                   {'cutoff': float('nan')}, {'return_info': 1}])
def test_invalid_even_one_site(kwargs):
    with pytest.raises((TypeError, ValueError)):
        tk.formats.TT([torch.ones(2)]).rounding(**kwargs)
