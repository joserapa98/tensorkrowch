"""Tests for kernels used by dataset-backed callable sources."""

import math

import pytest
import torch

import tensorkrowch as tk


def test_gaussian_normalization_bandwidth_and_gradients(device_dtype, assert_close):
    device, dtype = device_dtype
    dtype = torch.empty((), dtype=dtype).real.dtype
    data = torch.tensor([[0., 0.], [1., 2.]], device=device, dtype=dtype)
    coordinates = data.clone().requires_grad_()
    widths = torch.tensor([0.5, 2.], device=device, dtype=dtype)
    expected = torch.exp(-0.5 * ((coordinates[:, None] - data) / widths).square().sum(-1))
    expected = expected / (2 * math.pi * widths.prod())
    values = tk.decompositions.gaussian_kernel(coordinates, data, bandwidth=widths)
    assert_close(values, expected)
    values.sum().backward()
    assert torch.isfinite(coordinates.grad).all()


@pytest.mark.parametrize('bandwidth', [0., -1., float('inf'), [1., 0.], [1., 2., 3.]])
def test_gaussian_rejects_invalid_bandwidth(bandwidth):
    with pytest.raises(ValueError):
        tk.decompositions.gaussian_kernel(torch.zeros(2, 2), torch.zeros(3, 2), bandwidth)


def test_gaussian_rejects_incompatible_coordinates():
    with pytest.raises(ValueError):
        tk.decompositions.gaussian_kernel(torch.zeros(2, 2), torch.zeros(3, 3))
    with pytest.raises(TypeError):
        tk.decompositions.gaussian_kernel(torch.zeros(2, 2, dtype=torch.complex64),
                                         torch.zeros(3, 2))
