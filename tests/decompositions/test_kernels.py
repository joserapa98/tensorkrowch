"""Tests for pairwise kernels shared by sources and sketching."""

import math

import pytest
import torch

import tensorkrowch as tk


def test_gaussian_normalization_bandwidth_and_gradients(
        device_dtype, assert_close):
    device, dtype = device_dtype
    dtype = torch.empty((), dtype=dtype).real.dtype
    centers = torch.tensor([[0., 0.], [1., 2.]], device=device, dtype=dtype)
    centers.requires_grad_()
    coordinates = centers.detach().clone().requires_grad_()
    widths = torch.tensor([0.5, 2.], device=device, dtype=dtype,
                          requires_grad=True)
    differences = (coordinates[:, None] - centers) / widths
    expected = torch.exp(-0.5 * differences.square().sum(-1))
    expected = expected / (2 * math.pi * widths.prod())

    values = tk.decompositions.gaussian_kernel(
        coordinates, centers=centers, bandwidth=widths)

    assert_close(values, expected)
    values.sum().backward()
    for tensor in (coordinates, centers, widths):
        assert torch.isfinite(tensor.grad).all()


@pytest.mark.parametrize('bandwidth', [0., -1., float('inf'),
                                     [1., 0.], [1., 2., 3.]])
def test_gaussian_rejects_invalid_bandwidth(bandwidth):
    with pytest.raises(ValueError):
        tk.decompositions.gaussian_kernel(
            torch.zeros(2, 2), torch.zeros(3, 2), bandwidth)


def test_gaussian_rejects_incompatible_coordinates():
    with pytest.raises(ValueError):
        tk.decompositions.gaussian_kernel(
            torch.zeros(2, 2), torch.zeros(3, 3))
    with pytest.raises(TypeError):
        tk.decompositions.gaussian_kernel(
            torch.zeros(2, 2, dtype=torch.complex64), torch.zeros(3, 2))
    with pytest.raises(TypeError):
        tk.decompositions.gaussian_kernel(torch.zeros(2, 2), [[0., 0.]])


def test_gaussian_scalar_width_matches_per_coordinate_width():
    coordinates = torch.tensor([[0., 1., 2.], [3., 4., 5.]])
    centers = torch.tensor([[1., 2., 3.]])
    scalar = tk.decompositions.gaussian_kernel(coordinates, centers, 0.7)
    vector = tk.decompositions.gaussian_kernel(
        coordinates, centers, [0.7, 0.7, 0.7])

    torch.testing.assert_close(scalar, vector)
    assert scalar.shape == (2, 1)


def test_gaussian_integer_coordinates_promote_and_empty_queries_are_valid():
    centers = torch.tensor([[0], [1]])
    values = tk.decompositions.gaussian_kernel(torch.tensor([[0]]), centers)
    assert values.is_floating_point()
    expected = torch.tensor([[1., math.exp(-0.5)]]) / math.sqrt(2 * math.pi)
    torch.testing.assert_close(values, expected)
    assert tk.decompositions.gaussian_kernel(
        torch.empty(0, 1), centers).shape == (0, 2)
