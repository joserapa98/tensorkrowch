"""Construction, replacement, storage and parity of compact formats."""

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_result_parity(make_format, topology, n_sites, dtype):
    network = make_format(topology, n_sites, dtype=dtype)
    result_class = {'tt': tk.decompositions.TTDecomposition,
                    'tr': tk.decompositions.TRDecomposition,
                    'ttm': tk.decompositions.TTMDecomposition,
                    'trm': tk.decompositions.TRMDecomposition}[topology]
    result = result_class(network.cores)
    dense = result.contract_dense()
    assert torch.allclose(network.contract_dense(), dense)
    assert torch.allclose(network.norm(), dense.norm())
    assert torch.allclose(network.inner(network),
                          dense.abs().square().sum().to(dtype))
    assert torch.allclose(network.fidelity(network), dense.real.new_ones(()))
    inputs = torch.zeros(5, n_sites, dtype=torch.long)
    if network.out_dim is None:
        assert torch.allclose(network.evaluate(inputs), result.evaluate(inputs))
    else:
        outputs = torch.ones_like(inputs)
        assert torch.allclose(network.evaluate(inputs, outputs),
                              result.evaluate(inputs, outputs))
        assert torch.allclose(network.apply(inputs).contract_dense(),
                              result.apply(inputs).contract_dense())


@pytest.mark.parametrize('topology', ['tt', 'tr', 'trm'])
def test_independent_batches(make_format, topology):
    network = make_format(topology, n_batches=2)
    inputs = torch.zeros(2, network.n_sites, dtype=torch.long)
    if network.out_dim is None:
        values = network.evaluate(inputs)
    else:
        values = network.evaluate(inputs, inputs)
    assert values.shape == (2, 2, 2)
    expected = network.contract_dense()[(slice(None), slice(None)) +
                                       (0,) * (network.n_sites *
                                               (2 if network.out_dim else 1))]
    assert torch.allclose(values, expected.unsqueeze(-1).expand_as(values))


def test_core_replacement_and_cache():
    network = tk.formats.TT([torch.ones(2, 3), torch.ones(3, 4)])
    network.cores[:] = [torch.ones(2, 5), torch.ones(5, 4)]
    assert network.rank == [5]
    rank = network.rank
    rank[0] = 1
    assert network.rank == [5]
    assert torch.equal(network.contract_dense(), torch.full((2, 4), 5.))
    network.cores[0] = torch.ones(6, 5)
    assert network.in_dim == (6, 4)
    with pytest.raises(ValueError):
        network.cores = [torch.ones(6, 2), torch.ones(3, 4)]
    assert network.rank == [5]
    network.cores[0] = torch.ones(6, 2)
    with pytest.raises(ValueError, match='Adjacent'):
        network.norm()
    network.cores[1] = torch.ones(2, 4)
    assert network.rank == [2]


@pytest.mark.parametrize('operation', [lambda c: c.append(torch.ones(1)),
                                      lambda c: c.reverse(),
                                      lambda c: c.pop(),
                                      lambda c: c.clear()])
def test_structural_mutations_rejected(make_format, operation):
    network = make_format()
    with pytest.raises(TypeError):
        operation(network.cores)
    with pytest.raises(ValueError):
        network.cores[:] = [torch.ones(2)]
    with pytest.raises(TypeError):
        network.cores[0] = 1


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_copy_detach_and_conversion(make_format, topology):
    network = make_format(topology)
    network.cores[0].requires_grad_()
    assert network.to() is network and network.cpu() is network
    copied, detached = network.copy(), network.detach()
    assert type(copied) is type(network)
    assert copied.cores[0].data_ptr() != network.cores[0].data_ptr()
    assert copied.cores[0].requires_grad
    assert detached.cores[0].data_ptr() == network.cores[0].data_ptr()
    assert not detached.cores[0].requires_grad
    assert network.to(dtype=torch.complex128).dtype == torch.complex128
    assert network.detach_() is network
    assert not network.cores[0].requires_grad


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
    network = make_format()
    with pytest.raises(ValueError):
        network.evaluate(torch.full((2, 3), -1, dtype=torch.long))
    with pytest.raises(ValueError):
        network.evaluate(torch.ones(2, 3, 2))
    zero = tk.formats.TT([torch.zeros(2)])
    assert zero.norm() == 0
    with pytest.raises(ValueError, match='zero-norm'):
        zero.normalized_overlap(zero)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_scaled_norm_preserves_core_gradients(make_format, dtype):
    network = make_format('tt', 2, dtype=dtype)
    for core in network.cores:
        core.requires_grad_()
    expected = network.contract_dense().norm()
    actual = network.norm()
    expected_grad = torch.autograd.grad(expected, network.cores, retain_graph=True)
    actual_grad = torch.autograd.grad(actual, network.cores)
    assert torch.allclose(actual, expected)
    assert all(torch.allclose(a, b, rtol=1e-9, atol=1e-10)
               for a, b in zip(actual_grad, expected_grad))
