"""Batched open operators agree with independently evaluated batch members."""

from itertools import product

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('n_batches', [1, 2])
def test_batched_ttm_operations(make_format, n_sites, n_batches):
    format = make_format('ttm', n_sites, n_batches, torch.complex128)
    dense = format.contract_dense()
    inputs = torch.zeros(3, n_sites, dtype=torch.long)
    outputs = torch.ones_like(inputs)
    values = format.evaluate(inputs, outputs)
    applied = format.apply(inputs).contract_dense()
    norm = format.norm()
    adjoint = format.H.contract_dense()
    doubled = format.add(format).contract_dense()
    squared = format.hadamard(format).contract_dense()
    gram = (format.H @ format).contract_dense()

    for index in product(*[range(size) for size in format.batch_shape]):
        member = tk.formats.TTM([core[index] for core in format.cores])
        assert torch.allclose(dense[index], member.contract_dense())
        assert torch.allclose(values[index], member.evaluate(inputs, outputs))
        assert torch.allclose(applied[index], member.apply(inputs).contract_dense())
        assert torch.allclose(norm[index], member.norm())
        assert torch.allclose(adjoint[index], member.H.contract_dense())
        assert torch.allclose(doubled[index], 2 * member.contract_dense())
        assert torch.allclose(squared[index], member.contract_dense().square())
        assert torch.allclose(gram[index], (member.H @ member).contract_dense())

    canonical = format.clone().canonicalize()
    minimal = format.clone().canonicalize_minimal(method='gradient')
    rounded = format.clone().rounding()
    for result in (canonical, minimal, rounded):
        assert result.batch_shape == format.batch_shape
        assert torch.allclose(result.contract_dense(), dense, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('site', [0, 1, 2])
def test_batched_ttm_rejects_malformed_cores(make_format, site):
    cores = list(make_format('ttm', 3, n_batches=1).cores)
    cores[site] = cores[site].unsqueeze(-1)
    with pytest.raises(ValueError, match='dimensions'):
        tk.formats.TTM(cores, n_batches=1)


def test_batched_ttm_retains_autograd(make_format):
    format = make_format('ttm', 3, n_batches=1)
    original = list(format.cores)
    for core in original:
        core.requires_grad_()
    format.clone().canonicalize().contract_dense().square().sum().backward()
    assert all(core.grad is not None and torch.isfinite(core.grad).all()
               for core in original)


def test_batched_qttm_coordinates_and_plain_conversion():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    cores = [torch.arange(8., dtype=torch.float64).reshape(2, 2, 1, 2),
             torch.arange(8., dtype=torch.float64).reshape(2, 1, 2, 2)]
    format = tk.formats.QTTM(cores, layout, layout, n_batches=1)
    indices = torch.tensor([[0], [1], [3]])
    actual = format.evaluate_indices(indices, indices)
    for batch in range(2):
        member = tk.formats.QTTM([core[batch] for core in cores], layout, layout)
        assert torch.allclose(actual[batch],
                              member.evaluate_indices(indices, indices))
    plain = format.as_ttm()
    assert plain.batch_shape == (2,)
    assert torch.equal(plain.contract_dense(), format.contract_dense())
