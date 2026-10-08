"""Tests for sources/test_callable.py."""


import torch

import tensorkrowch as tk


def test_callable_resolves_effective_device():
    configurations = tk.decompositions.ConfigurationBatch(
        torch.tensor([[0, 1], [1, 2]]))
    source = tk.decompositions.CallableTensorSource(
        lambda indices: indices.sum(dim=1).to(torch.float64),
        in_dim=(2, 3),
        dtype=torch.float64,
        device='cpu:0')

    assert source.device == configurations.device
    assert torch.equal(
        source.evaluate(configurations),
        torch.tensor([1., 3.], dtype=torch.float64))


def test_callable_batches_are_contiguous_and_deterministic():
    batch_sizes = []

    def function(configurations):
        batch_sizes.append(configurations.shape[0])
        return configurations.sum(dim=1).to(torch.float64)

    source = tk.decompositions.CallableTensorSource(
        function,
        in_dim=(2, 3, 4),
        dtype=torch.float64,
        batch_size=2)
    configurations = tk.decompositions.ConfigurationBatch(
        torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [1, 2, 0],
            [1, 2, 3],
            [0, 1, 2],
        ]))

    values = source.evaluate(configurations)

    assert batch_sizes == [2, 2, 1]
    assert torch.equal(values, torch.tensor(
        [0., 1., 3., 6., 3.], dtype=torch.float64))


def test_callable_infers_tensor_output_and_dtype():
    source = tk.decompositions.CallableTensorSource(
        lambda x: torch.stack((x[:, 0], x[:, 1]), dim=1).to(
            torch.complex128),
        in_dim=(2, 3),
        out_shape=None,
        dtype=None)
    configurations = tk.decompositions.ConfigurationBatch(
        torch.tensor([[0, 1], [1, 2]]))

    result = source.evaluate(configurations)

    assert result.shape == (2, 2)
    assert source.out_shape == (2,)
    assert source.dtype == torch.complex128


def test_callable_receives_heterogeneous_coordinates():
    def function(values):
        x, vector = values
        return x + vector.sum(dim=1)

    source = tk.decompositions.CallableTensorSource(
        function,
        in_dim=(1, 1),
        dtype=torch.float64)
    configurations = tk.decompositions.ConfigurationBatch(
        (
            torch.tensor([1., 2.], dtype=torch.float64),
            torch.tensor([[3., 4.], [5., 6.]], dtype=torch.float64),
        ),
        kind='features')

    assert torch.equal(
        source.evaluate(configurations),
        torch.tensor([8., 13.], dtype=torch.float64))


def test_callable_discrete_fiber():
    source = tk.decompositions.CallableTensorSource(
        lambda x: (x[:, 0] + 2 * x[:, 1]).to(torch.float64),
        in_dim=(2, 3),
        dtype=torch.float64)
    configurations = tk.decompositions.ConfigurationBatch(
        torch.tensor([[0, 1], [1, 2]]))

    assert torch.equal(
        source.fiber(configurations, site=1),
        torch.tensor([[0., 2., 4.], [1., 3., 5.]],
                     dtype=torch.float64))
