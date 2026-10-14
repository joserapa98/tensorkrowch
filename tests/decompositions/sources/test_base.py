"""Tests for sources/base."""


import pytest
import torch

import tensorkrowch as tk


class TestConfigurationBatch:  # MARK: TestConfigurationBatch

    def test_packed_discrete_metadata_and_selection(self):
        values = torch.tensor([[0, 1, 2], [1, 0, 3]])
        configurations = tk.decompositions.ConfigurationBatch(values)

        assert configurations.packed
        assert configurations.batch_size == 2
        assert configurations.n_sites == 3
        assert configurations.feature_shape is None
        assert torch.equal(configurations.as_tensor(), values)
        assert torch.equal(
            configurations.index_select(torch.tensor([1])).as_tensor(),
            values[1:])

    def test_heterogeneous_features_preserve_feature_shapes(self):
        configurations = tk.decompositions.ConfigurationBatch(
            (
                torch.tensor([0.1, 0.2]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ),
            kind='features')

        assert not configurations.packed
        assert configurations.feature_shape == ((), (2,))
        with pytest.raises(ValueError, match='cannot be packed'):
            configurations.as_tensor()

    def test_packed_feature_shapes(self):
        scalar = tk.decompositions.ConfigurationBatch(
            torch.randn(2, 3), kind='features')
        vector = tk.decompositions.ConfigurationBatch(
            torch.randn(2, 3, 4), kind='features')

        assert scalar.feature_shape == ((), (), ())
        assert vector.feature_shape == ((4,), (4,), (4,))

    @pytest.mark.parametrize(
        'values, kind, error',
        [
            (torch.tensor([0, 1]), 'indices', ValueError),
            (torch.randn(2, 3), 'indices', TypeError),
            (torch.ones(2, 3, 2, dtype=torch.long), 'indices', TypeError),
            (torch.ones(2, 3), 'mixed', ValueError),
        ])
    def test_invalid_batches(self, values, kind, error):
        with pytest.raises(error):
            tk.decompositions.ConfigurationBatch(values, kind=kind)
