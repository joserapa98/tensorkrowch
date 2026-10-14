"""Tests for sources/sparse."""


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


def test_sparse_matrix_uses_input_response_contract():
    source = tk.decompositions.SparseTensorSource.matrix(
        torch.tensor([[0, 1, 1, 0], [0, 1, 1, 0]]),
        torch.tensor([2., 3.]), (2, 2), (2, 2))
    values = source.evaluate(tk.decompositions.ConfigurationBatch(torch.tensor([[0, 1]])))
    assert values.shape == (1, 2, 2)
    assert values[0, 1, 0] == 5
    result = tk.decompositions.TTSVD(source).fit()
    torch.testing.assert_close(result.contract_dense(), source.contract_dense())
