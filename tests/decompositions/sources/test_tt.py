"""Tests for sources/tt."""


from itertools import product

import pytest
import torch

import tensorkrowch as tk

from tests.decompositions.als._oracles import contract_tt_dense, make_tt_cores


def test_base_tt_source_absorbs_bonds_without_mutating_format():
    result = tk.formats.TT([torch.ones(2, 2), torch.ones(2, 3)])
    result.bonds = [torch.tensor([2., 3.])]
    source = tk.decompositions.as_tensor_source(result)
    indices = torch.tensor([[0, 0], [1, 2]])
    actual = source.evaluate(tk.decompositions.ConfigurationBatch(indices, kind='indices'))
    assert torch.equal(actual, torch.full((2,), 5.))
    assert result.bonds is not None


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_tt_source_evaluation_and_fibers_match_dense(dtype):
    cores = make_tt_cores(
        dtype=dtype, generator=torch.Generator().manual_seed(11))
    source = tk.decompositions.TTTensorSource(cores)
    dense = contract_tt_dense(cores)
    configurations = tk.decompositions.ConfigurationBatch(torch.tensor(
        list(product(*(range(dim) for dim in dense.shape)))))

    assert torch.allclose(
        source.evaluate(configurations), dense.reshape(-1))

    base = tk.decompositions.ConfigurationBatch(
        torch.tensor([[0, 0, 0], [1, 2, 1]]))
    expected = torch.stack((dense[0, :, 0], dense[1, :, 1]))
    assert torch.allclose(source.fiber(base, site=1), expected)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_tt_structured_prefix_suffix_and_phi_match_dense(dtype):
    cores = make_tt_cores(
        dtype=dtype, generator=torch.Generator().manual_seed(13))
    source = tk.decompositions.TTTensorSource(cores)
    dense = contract_tt_dense(cores)
    prefixes = torch.arange(dense.shape[0]).reshape(-1, 1)
    suffixes = torch.arange(dense.shape[2]).reshape(-1, 1)

    left = source.left_environments(prefixes)
    right = source.right_environments(suffixes)
    phi = source.local_phi(1, left, right)

    assert torch.allclose(phi, dense)
    assert source.evaluation_stats.requested_points == 0


@pytest.mark.parametrize('order', [1, 2])
def test_tt_structured_marginal_phi_matches_dense(order):
    cores = make_tt_cores(generator=torch.Generator().manual_seed(14))
    source = tk.decompositions.TTTensorSource(cores)
    dense = contract_tt_dense(cores)
    factor = (
        torch.tensor([1., 2.]),
        torch.tensor([1., 3., 2.]),
        torch.tensor([2., 1.]),
    )
    weighted = torch.einsum('ijk,i,j,k->ijk', dense, *factor)

    for site in range(3):
        phi = source.marginal_phi(site, order=order, factor=factor)
        if order == 1:
            expected = (
                weighted.sum(dim=2).unsqueeze(0) if site == 0 else
                weighted if site == 1 else
                weighted.sum(dim=0).unsqueeze(-1))
        else:
            left_size = 1
            for dimension in dense.shape[:site]:
                left_size *= dimension
            right_size = 1
            for dimension in dense.shape[site + 1:]:
                right_size *= dimension
            expected = weighted.reshape(
                left_size, dense.shape[site], right_size)
        assert torch.allclose(phi, expected)


def test_tt_source_bonds_device_and_complex_features(device_dtype, assert_close):
    device, dtype = device_dtype
    first = torch.tensor([[1., 2.], [3., 4.]], dtype=dtype, device=device)
    second = torch.tensor([[1., 2., 3.], [-1., 0., 2.]], dtype=dtype, device=device)
    if dtype.is_complex:
        first = first * (1 + 1j)
    format = tk.formats.TT([first, second])
    format.bonds = [torch.tensor([2., 3.], device=device)]
    source = tk.decompositions.as_tensor_source(format)
    dense = (first * format.bonds.factors[0]) @ second
    indices = torch.tensor([[0, 2], [1, 0]], device=device)
    values = source.evaluate(tk.decompositions.ConfigurationBatch(indices))
    assert_close(values, dense[indices[:, 0], indices[:, 1]])
    all_indices = torch.cartesian_prod(
        torch.arange(2, device=device), torch.arange(3, device=device))
    assert_close(source.evaluate(tk.decompositions.ConfigurationBatch(
        all_indices)).reshape(2, 3), dense)
    assert format.bonds is not None
