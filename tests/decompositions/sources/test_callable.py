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


def test_callable_declares_response_axes_and_infers_dtype():
    source = tk.decompositions.CallableTensorSource(
        lambda x: torch.stack((x[:, 0], x[:, 1]), dim=1).to(
            torch.complex128),
        in_dim=(2, 3, 2), in_sites=(0, 1),
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
        function, (2, 3, 1), in_sites=(0, 1), device=device, batch_size=2)
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
    assert isinstance(source, cls)
    assert source.n_sites == 2
    assert source.in_dim == (None if kind == 'continuous' else (2, 2))
    assert source.out_shape == ()
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


def test_discrete_data_infers_dimensions_only_without_kernel():
    dataset = torch.tensor([[0, 1], [2, 0], [0, 1]])
    source = tk.decompositions.CallableTensorSource.from_data(dataset)
    assert isinstance(source, tk.decompositions.CallableTensorSource)
    assert source.in_dim == (3, 2)
    torch.testing.assert_close(source.contract_dense(), torch.tensor(
        [[0., 2 / 3], [0., 0.], [1 / 3, 0.]]))
    with pytest.raises(ValueError, match='require `in_dim`'):
        tk.decompositions.CallableTensorSource.from_data(
            dataset, kernel=tk.decompositions.gaussian_kernel, bandwidth=1.)


def test_discrete_empirical_masses_preserve_weight_gradients():
    weights = torch.tensor([1., 2., 1.], dtype=torch.float64,
                           requires_grad=True)
    source = tk.decompositions.CallableTensorSource.from_data(
        torch.tensor([[0], [0], [2]]), weights=weights)
    values = source.evaluate(tk.decompositions.ConfigurationBatch(
        torch.tensor([[0], [1], [2], [0]])))
    expected = (2 * (weights[0] + weights[1]) + weights[2]) / weights.sum()
    torch.testing.assert_close(values.sum(), expected)
    actual_gradient, = torch.autograd.grad(values.sum(), weights,
                                          retain_graph=True)
    expected_gradient, = torch.autograd.grad(expected, weights)
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_discrete_empirical_lookup_handles_large_flat_indices(dtype):
    dataset = torch.tensor([[0, 0], [50000, 50000], [0, 0]], dtype=dtype)
    source = tk.decompositions.CallableTensorSource.from_data(dataset)
    values = source.evaluate(tk.decompositions.ConfigurationBatch(torch.tensor(
        [[50000, 50000], [0, 50000], [0, 0]], dtype=dtype)))
    torch.testing.assert_close(values, torch.tensor([1 / 3, 0., 2 / 3]))


@pytest.mark.parametrize('kernel', [None, tk.decompositions.gaussian_kernel])
@pytest.mark.parametrize('dataset, error', [
    (torch.tensor([[0.5]]), TypeError),
    (torch.tensor([[-1]]), ValueError),
    (torch.tensor([[2]]), ValueError),
])
def test_discrete_data_validates_sample_indices(kernel, dataset, error):
    with pytest.raises(error):
        tk.decompositions.CallableTensorSource.from_data(
            dataset, in_dim=(2,), kernel=kernel)


def test_callable_total_count_and_nonleading_inputs():
    def function(inputs):
        return (1000 * inputs[:, 1, None, None] +
                100 * torch.arange(3)[None, :, None] +
                10 * inputs[:, 0, None, None] + torch.arange(5)[None, None, :])

    source = tk.decompositions.CallableTensorSource(
        function, (2, 3, 4, 5), in_sites=(2, 0))
    continuous = tk.decompositions.CallableSource(
        function, 4, in_sites=(2, 0), out_shape=(3, 5))
    assert source.n_sites == continuous.n_sites == 4
    assert source.in_sites == continuous.in_sites == (2, 0)
    i, j, k, last = torch.meshgrid(
        *(torch.arange(n) for n in (2, 3, 4, 5)), indexing='ij')
    torch.testing.assert_close(
        source.contract_dense(), 1000 * i + 100 * j + 10 * k + last)
    indices = torch.tensor([[1, 0], [3, 1]])
    torch.testing.assert_close(
        continuous.evaluate(tk.decompositions.ConfigurationBatch(
            indices.float(), kind='features')),
        function(indices.float()))
    batch = tk.decompositions.ConfigurationBatch(indices)
    assert source.fiber(batch, 2).shape == (2, 4, 3, 5)
    with pytest.raises(ValueError, match='per output variable'):
        tk.decompositions.CallableSource(function, 4, in_sites=(2, 0), out_shape=())


def test_declared_output_axes_are_counted_before_evaluation():
    source = tk.decompositions.CallableTensorSource(
        lambda x: x[:, :1].float().expand(-1, 3), (2, 3), in_sites=(0,))
    source.evaluate(tk.decompositions.ConfigurationBatch(torch.tensor([[0]])))
    assert source.n_sites == 2
    assert source.in_sites == (0,)
    assert source.out_shape == (3,)
    assert source.contract_dense().shape == (2, 3)


def test_matrix_callable_has_no_separate_output_sites():
    source = tk.decompositions.CallableTensorSource.matrix(
        lambda x: (x[:, :1] + torch.arange(3)).float(), (2,), (3,))
    assert source.n_sites == len(source.in_sites) == 1
    assert source.out_dim == source.out_shape == (3,)


@pytest.mark.parametrize('out_shape', [None, (2, 3)])
def test_continuous_matrix_declares_or_infers_output_axes(out_shape):
    def function(x):
        return (x[:, 0, None, None] * torch.arange(2)[None, :, None] +
                x[:, 1, None, None] * torch.arange(3)[None, None, :])

    source = tk.decompositions.CallableSource.matrix(
        function, n_sites=2, out_shape=out_shape, batch_size=1)
    assert source.is_matrix
    assert source.n_sites == 2
    assert source.in_sites == (0, 1)
    assert source.in_dim is None
    assert source.out_dim == source.out_shape == out_shape
    coordinates = torch.tensor([[0.25, 0.5], [0.75, 1.]],
                               dtype=torch.float64, requires_grad=True)
    values = source.evaluate(tk.decompositions.ConfigurationBatch(
        coordinates, kind='features'))
    torch.testing.assert_close(values, function(coordinates))
    assert source.out_dim == source.out_shape == (2, 3)
    assert source.dtype == torch.float64
    gradient, = torch.autograd.grad(values.sum(), coordinates)
    torch.testing.assert_close(gradient, torch.tensor(
        [[3., 6.], [3., 6.]], dtype=torch.float64))
    with pytest.raises(ValueError, match='discrete tensor'):
        source.contract_dense()


@pytest.mark.parametrize('out_shape', [(), (2,), (2, 3, 4)])
def test_continuous_matrix_requires_one_output_axis_per_input(out_shape):
    def function(x):
        return x.new_zeros((x.shape[0], *out_shape))

    with pytest.raises(ValueError):
        tk.decompositions.CallableSource.matrix(
            function, n_sites=2, out_shape=out_shape)
    source = tk.decompositions.CallableSource.matrix(function, n_sites=2)
    with pytest.raises(ValueError, match='one axis per output variable'):
        source.evaluate(tk.decompositions.ConfigurationBatch(
            torch.ones(1, 2), kind='features'))
