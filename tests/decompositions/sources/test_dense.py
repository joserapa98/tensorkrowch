"""Tests for sources/dense."""


import pytest
import torch

import tensorkrowch as tk


class TestDenseAndCallableSources:  # MARK: TestDenseAndCallableSources

    def test_dense_scalar_evaluation_and_fiber(self):
        tensor = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        source = tk.decompositions.DenseTensorSource(tensor)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 1, 2], [1, 2, 3]]))

        assert torch.equal(source.evaluate(configurations),
                           torch.tensor([6., 23.], dtype=tensor.dtype))
        fiber = source.fiber(configurations, site=1)
        expected = torch.stack((tensor[0, :, 2], tensor[1, :, 3]))
        assert torch.equal(fiber, expected)

    def test_dense_tensor_output(self):
        tensor = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        source = tk.decompositions.DenseTensorSource(
            tensor, in_features=(0, 1))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 1], [1, 2]]))

        assert source.out_shape == (4,)
        assert torch.equal(source.evaluate(configurations),
                           torch.stack((tensor[0, 1], tensor[1, 2])))
        assert source.fiber(configurations, site=0).shape == (2, 2, 4)

    def test_dense_nonleading_input_axes(self):
        tensor = torch.arange(120).reshape(2, 3, 4, 5)
        source = tk.decompositions.DenseTensorSource(
            tensor, in_features=(2, 0))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[1, 0], [3, 1]]))

        assert source.in_features == (2, 0)
        assert source.out_features == (1, 3)
        assert source.in_dim == (4, 2)
        assert source.out_shape == (3, 5)
        assert torch.equal(source.evaluate(configurations),
                           torch.stack((tensor[0, :, 1, :],
                                        tensor[1, :, 3, :])))
        assert source.fiber(configurations, site=0).shape == (2, 4, 3, 5)

    @pytest.mark.parametrize('features', [(), (0, 0), (2,), (-1,), (True,)])
    def test_dense_rejects_invalid_input_axes(self, features):
        with pytest.raises(ValueError):
            tk.decompositions.DenseTensorSource(
                torch.ones(2, 3), in_features=features)
