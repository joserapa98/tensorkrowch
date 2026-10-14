"""Tests for sources/test_callable.py."""


import pytest
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

    source = tk.decompositions.CallableSource(
        function,
        n_sites=2,
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


def test_callable_source_inference_chunking_and_changes(device_dtype, assert_close):
    device, dtype = device_dtype
    calls = []
    def function(indices):
        calls.append(indices.shape[0])
        values = indices.to(dtype).sum(-1)
        if dtype.is_complex:
            values = values * (1 + 2j)
        return values.unsqueeze(-1)
    source = tk.decompositions.CallableTensorSource(
        function, (2, 3), out_shape=None, device=device, batch_size=2)
    indices = torch.tensor([[0, 0], [0, 2], [1, 1]], device=device)
    values = source.evaluate(tk.decompositions.ConfigurationBatch(indices))
    assert calls == [2, 1]
    expected = indices.to(dtype).sum(-1, keepdim=True)
    if dtype.is_complex:
        expected = expected * (1 + 2j)
    assert_close(values, expected)
    assert source.dtype == dtype and source.out_shape == (1,)
    source.function = lambda batch: batch.to(dtype).sum(-1)
    with pytest.raises(ValueError, match='shape'):
        source.evaluate(tk.decompositions.ConfigurationBatch(indices))


@pytest.mark.parametrize('kind', ['continuous', 'discrete'])
@pytest.mark.parametrize('kernel', [None, tk.decompositions.gaussian_kernel])
def test_dataset_sources_reuse_evaluation_and_kernels(kind, kernel, device_dtype, assert_close):
    device, dtype = device_dtype
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    dataset = torch.tensor([[0, 0], [0, 0], [1, 1]], device=device)
    if kind == 'continuous':
        dataset = dataset.to(real_dtype)
        cls = tk.decompositions.CallableSource
        options = {}
        query_kind = 'features'
    else:
        cls = tk.decompositions.CallableTensorSource
        options = {'in_dim': (2, 2)}
        query_kind = 'indices'
    weights = torch.tensor([1., 2., 1.], dtype=real_dtype, device=device)
    source = cls.from_data(dataset, weights=weights, kernel=kernel, **options,
                           **({'bandwidth': 0.5} if kernel is not None else {}))
    query = torch.tensor([[0, 0], [1, 1], [0, 1]], device=device)
    if kind == 'continuous':
        query = query.to(real_dtype)
    expected = torch.tensor([0.75, 0.25, 0.], device=device, dtype=real_dtype)
    if kernel is not None:
        expected = kernel(query, dataset, bandwidth=0.5).to(real_dtype) @ (weights / weights.sum())
    assert_close(source.evaluate(tk.decompositions.ConfigurationBatch(query, kind=query_kind)),
                 expected)
    assert source.is_discrete_tensor == (kind == 'discrete')
    if kind == 'continuous':
        with pytest.raises(ValueError, match='discrete tensor'):
            source.contract_dense()
    else:
        assert_close(source.contract_dense()[tuple(query.T)], expected)


@pytest.mark.parametrize('kwargs', [{'weights': torch.tensor([-1., 1.])},
                                    {'weights': torch.tensor([0., 0.])},
                                    {'weights': torch.ones(3)}])
def test_data_source_rejects_invalid_masses(kwargs):
    with pytest.raises(ValueError):
        tk.decompositions.CallableSource.from_data(torch.zeros(2, 1), **kwargs)


def test_custom_kernel_preserves_complex_values_and_receives_dataset():
    data = torch.tensor([[0.], [1.]])
    seen = []

    def kernel(query, dataset, scale):
        seen.append(dataset)
        return (query[:, None, 0] + dataset[None, :, 0]) * scale

    source = tk.decompositions.CallableSource.from_data(
        data, kernel=kernel, scale=1 + 2j)
    values = source.evaluate(tk.decompositions.ConfigurationBatch(
        torch.tensor([[0.5]]), kind='features'))
    torch.testing.assert_close(values, torch.tensor([1 + 2j]))
    assert seen[0] is data
