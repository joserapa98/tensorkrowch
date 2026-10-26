"""Tests for sources/sparse."""


import pytest
import torch

import tensorkrowch as tk


class TestSparseAndTTSources:  # MARK: TestSparseAndTTSources

    def test_sparse_source_coalesces_and_declares_missing_zeros(self):
        source = tk.decompositions.SparseTensorSource(
            indices=torch.tensor([[0, 0], [0, 0], [1, 2]]),
            values=torch.tensor([1., 2., 4.]),
            in_dim=(2, 3))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [0, 2], [1, 2]]))

        assert torch.equal(source.support.as_tensor(),
                           torch.tensor([[0, 0], [1, 2]]))
        assert torch.equal(source.support_values, torch.tensor([3., 4.]))
        assert torch.equal(source.evaluate(configurations),
                           torch.tensor([3., 0., 4.]))
        base = tk.decompositions.ConfigurationBatch(torch.tensor([[0, 0]]))
        assert torch.equal(source.fiber(base, site=1),
                           torch.tensor([[3., 0., 0.]]))

    def test_empirical_distribution_accumulates_normalized_mass(self):
        distribution = tk.decompositions.SparseTensorSource.from_data(
            torch.tensor([[0, 1], [0, 1], [1, 0], [1, 1]]),
            in_dim=(2, 2))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))

        assert torch.equal(
            distribution.evaluate(configurations),
            torch.tensor([0., 0.5, 0.25, 0.25]))
        assert distribution.support_values.sum() == 1


def test_boolean_sparse_repetitions_use_logical_or():
    source = tk.decompositions.SparseTensorSource(
        torch.tensor([[0, 1], [0, 1], [1, 0]]),
        torch.tensor([False, True, False]), (2, 2))
    assert source.contract_dense().tolist() == [[False, True], [False, False]]
    assert source.support.as_tensor().shape == (2, 2)


def test_sparse_infers_output_dimensions_from_values():
    source = tk.decompositions.SparseTensorSource(
        torch.tensor([[0], [2]]), torch.arange(12.).reshape(2, 2, 3),
        in_dim=(4,))
    assert source.in_dim == (4, 2, 3)
    assert source.n_sites == 3
    assert source.in_sites == (0,)
    assert source.out_shape == (2, 3)
    torch.testing.assert_close(source.evaluate(
        tk.decompositions.ConfigurationBatch(torch.tensor([[2], [1]]))),
        torch.stack((torch.arange(6., 12.).reshape(2, 3), torch.zeros(2, 3))))


@pytest.mark.parametrize('in_dim', [(2, 3, 4), (2, 3, 4, 6),
                                   (2, 3, 4, 5, 6), (2, 3, 0, 5)])
def test_sparse_rejects_inconsistent_output_dimensions(in_dim):
    with pytest.raises(ValueError):
        tk.decompositions.SparseTensorSource(
            torch.tensor([[0, 0]]), torch.ones(1, 4, 5), in_dim)


def test_sparse_rejects_indices_outside_selected_site_dimensions():
    with pytest.raises(ValueError, match='out of bounds'):
        tk.decompositions.SparseTensorSource(
            torch.tensor([[1, 2]]), torch.ones(1), (2, 3), in_sites=(1, 0))


def test_sparse_rejects_separate_storage_and_evaluation_sites():
    with pytest.raises(ValueError, match='one column per input site'):
        tk.decompositions.SparseTensorSource(
            torch.tensor([[0, 1]]), torch.ones(1), (2, 3), in_sites=(0,))


def test_sparse_matrix_rejects_tensor_values():
    with pytest.raises(ValueError, match='one scalar per matrix entry'):
        tk.decompositions.SparseTensorSource.matrix(
            torch.tensor([[0, 1]]), torch.ones(1, 2), (2,), (3,))


def test_sparse_matrix_uses_input_output_contract():
    source = tk.decompositions.SparseTensorSource.matrix(
        torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]]),
        torch.tensor([2., 3.]), (2, 2), (2, 2), ordering='interleaved')
    values = source.evaluate(tk.decompositions.ConfigurationBatch(torch.tensor([[0, 1]])))
    assert values.shape == (1, 2, 2)
    assert values[0, 1, 0] == 5
    result = tk.decompositions.TTSVD(source).fit()
    torch.testing.assert_close(result.contract_dense(), source.contract_dense())


@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_sparse_matrix_column_ordering(ordering, device_dtype, assert_close):
    device, dtype = device_dtype
    indices = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4], [0, 1, 2, 1]],
                           device=device)
    values = torch.tensor([2., 3., 7.], device=device).to(dtype)
    if dtype.is_complex:
        values = values * (1 + 0.5j)
    if ordering == 'interleaved':
        indices = indices[:, (0, 2, 1, 3)]
    source = tk.decompositions.SparseTensorSource.matrix(
        indices, values, (2, 3), (4, 5), ordering=ordering)
    expected = values.new_zeros(2, 4, 3, 5)
    expected[1, 3, 2, 4] = values[0] + values[1]
    expected[0, 2, 1, 1] = values[2]
    assert_close(source.contract_dense(), expected)
    configurations = tk.decompositions.ConfigurationBatch(
        torch.tensor([[1, 2], [0, 1]], device=device))
    assert_close(source.evaluate(configurations),
                 torch.stack((expected[1, :, 2, :], expected[0, :, 1, :])))
    assert source._ordering == ordering


def test_sparse_matrix_defaults_to_grouped_columns():
    source = tk.decompositions.SparseTensorSource.matrix(
        torch.tensor([[1, 2, 3, 4]]), torch.tensor([5.]), (2, 3), (4, 5))
    values = source.evaluate(tk.decompositions.ConfigurationBatch(
        torch.tensor([[1, 2]])))
    assert values.shape == (1, 4, 5)
    assert values[0, 3, 4] == 5
    assert torch.count_nonzero(values) == 1


@pytest.mark.parametrize('ordering, error', [(None, TypeError), (1, TypeError),
                                           ('other', ValueError)])
def test_sparse_matrix_rejects_invalid_ordering(ordering, error):
    with pytest.raises(error, match='`ordering` should be'):
        tk.decompositions.SparseTensorSource.matrix(
            torch.tensor([[0, 0]]), torch.tensor([1.]), (2,), (3,),
            ordering=ordering)


@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_sparse_matrix_rejects_incomplete_pairs(ordering):
    with pytest.raises(ValueError, match='two columns per site'):
        tk.decompositions.SparseTensorSource.matrix(
            torch.tensor([[0, 0, 0]]), torch.tensor([1.]), (2, 3), (4, 5),
            ordering=ordering)


@pytest.mark.parametrize('sites', [(2, 0), (3, 1), (1,), (3, 2, 1, 0)])
@pytest.mark.parametrize('infer_outputs', [False, True])
def test_sparse_selected_variables_match_dense(sites, infer_outputs):
    from tensorkrowch.decompositions.sources.base import _tensor_view

    generator = torch.Generator().manual_seed(42)
    shape = (2, 3, 4, 5)
    out_sites = tuple(site for site in range(4) if site not in sites)
    indices = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3], [1, 2, 2, 3]])[:, sites]
    values = torch.randn((3, *(shape[site] for site in out_sites)),
                         dtype=torch.complex128, generator=generator,
                         requires_grad=True)
    in_dim = (tuple(shape[site] for site in sites)
              if infer_outputs and out_sites else shape)
    source = tk.decompositions.SparseTensorSource(
        indices, values, in_dim, in_sites=sites)
    tensor = values.new_zeros(shape)
    for row, value in zip(indices, values):
        selection = [slice(None)] * 4
        for site, index in zip(sites, row):
            selection[site] = index
        tensor[tuple(selection)] += value
    dense = tk.decompositions.DenseTensorSource(tensor, in_sites=sites)
    assert source.n_sites == dense.n_sites == 4
    assert source.in_dim == dense.in_dim
    assert source.out_shape == dense.out_shape
    torch.testing.assert_close(source.contract_dense(), tensor)
    queries = torch.cartesian_prod(
        *(torch.arange(source.in_dim[site]) for site in sites))
    batch = tk.decompositions.ConfigurationBatch(queries.reshape(-1, len(sites)))
    torch.testing.assert_close(source.evaluate(batch), dense.evaluate(batch))
    torch.testing.assert_close(source.fiber(batch, sites[0]),
                               dense.fiber(batch, sites[0]))
    torch.testing.assert_close(_tensor_view(source).contract_dense(), tensor)
    source.evaluate(batch).real.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))


@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_sparse_matrix_counts_pairs_and_evaluates_all_inputs(ordering):
    indices = torch.tensor([[1, 2, 3, 4]])
    if ordering == 'interleaved':
        indices = indices[:, [0, 2, 1, 3]]
    source = tk.decompositions.SparseTensorSource.matrix(
        indices, torch.tensor([7.]), (2, 3), (4, 5), ordering=ordering)
    assert source.n_sites == len(source.in_sites) == 2
    assert source.in_sites == (0, 1)
    assert source.out_dim == source.out_shape == (4, 5)
    values = source.evaluate(tk.decompositions.ConfigurationBatch(
        torch.tensor([[1, 2]])))
    assert values[0, 3, 4] == 7
    assert source.contract_dense().shape == (2, 4, 3, 5)
