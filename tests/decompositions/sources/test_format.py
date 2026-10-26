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
    source = tk.decompositions.FormatSource(cores)
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
    source = tk.decompositions.FormatSource(cores)
    dense = contract_tt_dense(cores)
    prefixes = torch.arange(dense.shape[0]).reshape(-1, 1)
    suffixes = torch.arange(dense.shape[2]).reshape(-1, 1)

    left = source.left_envs(prefixes)
    right = source.right_envs(suffixes)
    phi = source.local_phi(1, left, right)

    assert torch.allclose(phi, dense)
    assert source.evaluation_stats.requested_points == 0


@pytest.mark.parametrize('order', [1, 2])
def test_tt_structured_marginal_phi_matches_dense(order):
    cores = make_tt_cores(generator=torch.Generator().manual_seed(14))
    source = tk.decompositions.FormatSource(cores)
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


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('matrix', [False, True])
def test_all_plain_format_families(cyclic, matrix, device_dtype, assert_close):
    device, dtype = device_dtype
    cores = [torch.ones(1, 2, 1, dtype=dtype, device=device) for _ in range(3)]
    if matrix:
        cores = [core.unsqueeze(-1).expand(1, 2, 1, 3).clone() for core in cores]
    if not cyclic:
        cores = [cores[0].squeeze(0), *cores[1:-1], cores[-1].squeeze(-2 if matrix else -1)]
    cls = tk.formats.TRM if cyclic and matrix else tk.formats.TTM if matrix else tk.formats.TR if cyclic else tk.formats.TT
    format = cls(cores)
    source = tk.decompositions.FormatSource(format)
    indices = torch.tensor([[0, 0, 0], [1, 1, 1]], device=device)
    values = source.evaluate(tk.decompositions.ConfigurationBatch(indices))
    expected = torch.ones((2, 3, 3, 3) if matrix else (2,), dtype=dtype, device=device)
    assert_close(values, expected)
    assert_close(source.contract_dense(), format.contract_dense())


def test_batched_partial_format_source_orders_configuration_axis_first():
    tensor = torch.arange(24., dtype=torch.float64).reshape(2, 2, 2, 3)
    format = tk.decompositions.TTSVD(
        tk.decompositions.DenseTensorSource(tensor, n_batches=1)).fit().to_format()
    format.in_sites = (0, 1)
    source = tk.decompositions.FormatSource(format)
    indices = torch.tensor([[0, 1], [1, 0], [0, 0]])
    expected = torch.stack([tensor[:, i, j] for i, j in indices])
    torch.testing.assert_close(source.evaluate(tk.decompositions.ConfigurationBatch(indices)),
                               expected)


@pytest.mark.parametrize('batched', [False, True])
def test_partial_format_retains_total_count_and_source_fibers(batched):
    from tensorkrowch.decompositions.sources.base import _tensor_view

    shape = (2, 2, 3, 4) if batched else (2, 3, 4)
    tensor = torch.arange(torch.tensor(shape).prod(), dtype=torch.float64).reshape(shape)
    format = tk.decompositions.TTSVD(tk.decompositions.DenseTensorSource(
        tensor, n_batches=int(batched))).fit().to_format()
    format.in_sites = (2, 0)
    source = tk.decompositions.FormatSource(format)
    assert source.n_sites == 3
    assert source.in_sites == (2, 0)
    assert source.out_shape == (3,)
    dense = tk.decompositions.DenseTensorSource(
        tensor, in_sites=(2, 0), n_batches=int(batched))
    batch = tk.decompositions.ConfigurationBatch(torch.tensor([[1, 0], [3, 1]]))
    torch.testing.assert_close(source.evaluate(batch), dense.evaluate(batch))
    torch.testing.assert_close(source.fiber(batch, 2), dense.fiber(batch, 2))
    torch.testing.assert_close(_tensor_view(source).contract_dense(), tensor)


def test_matrix_format_source_evaluates_every_pair():
    tensor = torch.arange(36., dtype=torch.float64).reshape(2, 3, 2, 3)
    format = tk.decompositions.TTMSVD(tk.decompositions.DenseTensorSource.matrix(
        tensor, ordering='interleaved')).fit().to_format()
    format.in_sites = (1,)
    source = tk.decompositions.FormatSource(format)
    assert source.n_sites == len(source.in_sites) == 2
    assert source.in_sites == (0, 1)
    assert source.out_dim == source.out_shape == (3, 3)
    result = source.evaluate(tk.decompositions.ConfigurationBatch(
        torch.tensor([[1, 0]])))
    torch.testing.assert_close(result, tensor[1, :, 0, :].unsqueeze(0))
