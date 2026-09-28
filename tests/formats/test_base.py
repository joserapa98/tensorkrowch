"""Construction, replacement, storage and parity of compact formats."""

import pytest
import torch
import tensorkrowch as tk


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


@pytest.mark.parametrize('topology', ['tt', 'tr', 'trm'])
def test_independent_batches(make_format, topology):
    format = make_format(topology, n_batches=2)
    inputs = torch.zeros(2, format.n_sites, dtype=torch.long)
    if format.out_dim is None:
        values = format.evaluate(inputs)
    else:
        values = format.evaluate(inputs, inputs)
    assert values.shape == (2, 2, 2)
    expected = format.contract_dense()[(slice(None), slice(None)) +
                                       (0,) * (format.n_sites *
                                               (2 if format.out_dim else 1))]
    assert torch.allclose(values, expected.unsqueeze(-1).expand_as(values))


def test_core_replacement_and_cache():
    format = tk.formats.TT([torch.ones(2, 3), torch.ones(3, 4)])
    format.cores[:] = [torch.ones(2, 5), torch.ones(5, 4)]
    assert format.rank == [5]
    rank = format.rank
    rank[0] = 1
    assert format.rank == [5]
    assert torch.equal(format.contract_dense(), torch.full((2, 4), 5.))
    format.cores[0] = torch.ones(6, 5)
    assert format.in_dim == (6, 4)
    with pytest.raises(ValueError):
        format.cores = [torch.ones(6, 2), torch.ones(3, 4)]
    assert format.rank == [5]
    with pytest.raises(ValueError, match='Adjacent'):
        format.cores[0] = torch.ones(6, 2)
    assert format.rank == [5]
    assert format.in_dim == (6, 4)
    assert torch.equal(format.contract_dense(), torch.full((6, 4), 5.))
    format.cores[:] = [torch.ones(6, 2), torch.ones(2, 4)]
    assert format.rank == [2]


@pytest.mark.parametrize('operation', [lambda c: c.append(torch.ones(1)),
                                      lambda c: c.reverse(),
                                      lambda c: c.pop(),
                                      lambda c: c.clear()])
def test_structural_mutations_rejected(make_format, operation):
    format = make_format()
    with pytest.raises(TypeError):
        operation(format.cores)
    with pytest.raises(ValueError):
        format.cores[:] = [torch.ones(2)]
    with pytest.raises(TypeError):
        format.cores[0] = 1


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_clone_detach_and_conversion(make_format, topology):
    format = make_format(topology)
    format.cores[0].requires_grad_()
    assert format.to() is format and format.cpu() is format
    copied, detached = format.clone(), format.detach()
    assert type(copied) is type(format)
    assert copied.cores[0].data_ptr() != format.cores[0].data_ptr()
    assert copied.cores[0].requires_grad
    assert detached.cores[0].data_ptr() == format.cores[0].data_ptr()
    assert not detached.cores[0].requires_grad
    assert format.to(dtype=torch.complex128).dtype == torch.complex128
    assert format.detach_() is format
    assert not format.cores[0].requires_grad


def test_invalid_construction_and_data(make_format):
    for cores in [[], [torch.empty(0)], [torch.ones(2), torch.ones(2)],
                  [torch.ones(2, 1), torch.ones(1, 3, dtype=torch.float64)]]:
        with pytest.raises(ValueError):
            tk.formats.TT(cores)
    with pytest.raises(TypeError):
        tk.formats.TT([1])
    with pytest.raises(TypeError):
        tk.formats.TT([torch.ones(2)], n_batches=True)
    with pytest.raises(ValueError):
        make_format('ttm', n_batches=1)
    format = make_format()
    with pytest.raises(ValueError):
        format.evaluate(torch.full((2, 3), -1, dtype=torch.long))
    with pytest.raises(ValueError):
        format.evaluate(torch.ones(2, 3, 2))
    zero = tk.formats.TT([torch.zeros(2)])
    assert zero.norm() == 0
    with pytest.raises(ValueError, match='zero-norm'):
        zero.normalized_overlap(zero)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_scaled_norm_preserves_core_gradients(make_format, dtype):
    format = make_format('tt', 2, dtype=dtype)
    for core in format.cores:
        core.requires_grad_()
    expected = format.contract_dense().norm()
    actual = format.norm()
    expected_grad = torch.autograd.grad(expected, format.cores, retain_graph=True)
    actual_grad = torch.autograd.grad(actual, format.cores)
    assert torch.allclose(actual, expected)
    assert all(torch.allclose(a, b, rtol=1e-9, atol=1e-10)
               for a, b in zip(actual_grad, expected_grad))
