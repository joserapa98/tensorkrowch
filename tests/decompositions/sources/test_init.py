"""Tests for sources/init."""

import pytest
import torch

import tensorkrowch as tk


def test_source_factory_uses_base_source_and_preserves_runtime():
    tensor = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
    source = tk.decompositions.as_tensor_source(tensor)
    assert isinstance(source, tk.decompositions.BaseSource)
    assert source.dtype == tensor.dtype and source.device == tensor.device


def test_vector_sources_evaluation_fibers_and_dense_order(vector_source_case):
    source, dense = vector_source_case
    assert source.n_sites == dense.ndim
    assert source.in_dim == dense.shape
    assert source.in_sites == tuple(range(dense.ndim))
    assert source.out_shape == ()
    assert source.dtype is None or source.dtype == dense.dtype
    indices = torch.stack((torch.zeros(dense.ndim, dtype=torch.long),
                            torch.tensor(dense.shape) - 1))
    configurations = tk.decompositions.ConfigurationBatch(
        tuple(indices.unbind(1)))
    torch.testing.assert_close(source.evaluate(configurations),
                               dense[tuple(indices.T)])
    site = dense.ndim // 2
    expected = torch.stack([
        dense[tuple(slice(None) if k == site else row[k]
                    for k in range(dense.ndim))] for row in indices])
    torch.testing.assert_close(source.fiber(configurations, site), expected)
    torch.testing.assert_close(source.fiber(
        configurations, site, values=torch.tensor([dense.shape[site] - 1, 0])),
        expected[:, [dense.shape[site] - 1, 0]])
    torch.testing.assert_close(source.contract_dense(), dense)
    assert source.evaluate(tk.decompositions.ConfigurationBatch(
        torch.empty(0, dense.ndim, dtype=torch.long))).shape == (0,)
    _check_invalid_queries(source)


def test_matrix_sources_evaluation_fibers_and_dense_order(matrix_source_case):
    source, dense = matrix_source_case
    n_sites = dense.ndim // 2
    assert source.n_sites == n_sites
    assert source.in_dim == dense.shape[::2]
    assert source.out_dim == source.out_shape == dense.shape[1::2]
    indices = torch.stack((torch.zeros(n_sites, dtype=torch.long),
                            torch.tensor(source.in_dim) - 1))
    batch = tk.decompositions.ConfigurationBatch(tuple(indices.unbind(1)))
    grouped = dense.permute(*range(0, dense.ndim, 2),
                             *range(1, dense.ndim, 2))
    torch.testing.assert_close(source.evaluate(batch), grouped[tuple(indices.T)])
    site = n_sites // 2
    expected = []
    for row in indices:
        entries = []
        for value in range(source.in_dim[site]):
            query = row.clone()
            query[site] = value
            entries.append(grouped[tuple(query)])
        expected.append(torch.stack(entries))
    torch.testing.assert_close(source.fiber(batch, site), torch.stack(expected))
    torch.testing.assert_close(source.contract_dense(), dense)
    assert source.evaluate(tk.decompositions.ConfigurationBatch(
        torch.empty(0, n_sites, dtype=torch.long))).shape == (0, *source.out_dim)
    _check_invalid_queries(source)


def _check_invalid_queries(source):
    indices = torch.zeros(1, source.n_sites, dtype=torch.long)
    batch = tk.decompositions.ConfigurationBatch(indices)
    stats = source.evaluation_stats
    with pytest.raises(TypeError):
        source.evaluate(indices)
    with pytest.raises(ValueError):
        source.evaluate(tk.decompositions.ConfigurationBatch(
            indices, kind='features'))
    with pytest.raises(ValueError):
        source.fiber(batch, source.n_sites)
    with pytest.raises(ValueError):
        source.fiber(batch, 0, values=torch.tensor([source.in_dim[0]]))
    indices[0, -1] = source.in_dim[-1]
    with pytest.raises(ValueError):
        source.evaluate(tk.decompositions.ConfigurationBatch(indices))
    assert source.evaluation_stats == stats
