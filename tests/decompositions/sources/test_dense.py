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
            tensor, in_sites=(0, 1))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 1], [1, 2]]))

        assert source.out_shape == (4,)
        assert torch.equal(source.evaluate(configurations),
                           torch.stack((tensor[0, 1], tensor[1, 2])))
        assert source.fiber(configurations, site=0).shape == (2, 2, 4)

    def test_dense_nonleading_input_axes(self):
        tensor = torch.arange(120).reshape(2, 3, 4, 5)
        source = tk.decompositions.DenseTensorSource(
            tensor, in_sites=(2, 0))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[1, 0], [3, 1]]))

        assert source.in_sites == (2, 0)
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
                torch.ones(2, 3), in_sites=features)


def test_dense_source_device_fibers_and_values(device_dtype, assert_close):
    device, dtype = device_dtype
    tensor = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2).to(dtype).to(device)
    if dtype.is_complex:
        tensor = tensor + 1j * tensor.flip(1)
    source = tk.decompositions.DenseTensorSource(tensor, in_sites=(0, 1))
    indices = torch.tensor([[0, 0], [1, 2]], device=device)
    batch = tk.decompositions.ConfigurationBatch(indices)
    assert_close(source.evaluate(batch), tensor[indices[:, 0], indices[:, 1]])
    assert source.out_shape == (2,)
    assert_close(source.fiber(batch, site=1), tensor[indices[:, 0]])
    assert_close(source.tensor, tensor)
    with pytest.raises(ValueError):
        source.evaluate(tk.decompositions.ConfigurationBatch(indices + 3))


@pytest.mark.parametrize('ordering', ['interleaved', 'grouped'])
def test_matrix_tensor_view_and_source_ownership(ordering, device_dtype, assert_close):
    from tensorkrowch.decompositions.sources.base import _tensor_view

    device, dtype = device_dtype
    tensor = torch.arange(24., device=device).to(dtype).reshape(2, 3, 2, 2)
    if dtype.is_complex:
        tensor = tensor * (1 + 2j)
    raw = tensor if ordering == 'interleaved' else tensor.permute(0, 2, 1, 3)
    source = tk.decompositions.DenseTensorSource.matrix(raw, ordering=ordering)
    view = _tensor_view(source)
    indices = torch.cartesian_prod(torch.arange(6), torch.arange(4)).to(device)
    assert_close(view.evaluate(tk.decompositions.ConfigurationBatch(indices)),
                 tensor.reshape(-1))
    result = tk.decompositions.TTSVD(source, out_device=None).fit()
    assert isinstance(result, tk.decompositions.TTMDecomposition)
    assert_close(result.contract_dense(), tensor)
    assert source.tensor is raw


@pytest.mark.parametrize('axis_layout', ['grouped', 'interleaved'])
def test_matrix_source_preserves_pairing_and_gradients(axis_layout):
    matrix = torch.arange(24, dtype=torch.float64).reshape(4, 6).requires_grad_()
    grouped = matrix.reshape(2, 2, 2, 3)
    tensor = grouped if axis_layout == 'grouped' else grouped.permute(0, 2, 1, 3)
    source = tk.decompositions.DenseTensorSource.matrix(tensor, ordering=axis_layout)
    assert source.in_dim == (2, 2)
    assert source.out_dim == (2, 3)
    torch.testing.assert_close(source.contract_dense(), grouped.permute(0, 2, 1, 3))
    source.contract_dense().sum().backward()
    torch.testing.assert_close(matrix.grad, torch.ones_like(matrix))


@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_two_axis_matrix_source_has_one_site(ordering):
    tensor = torch.arange(24.).reshape(4, 6)
    source = tk.decompositions.DenseTensorSource.matrix(tensor, ordering=ordering)
    assert source.n_sites == 1
    assert source.in_dim == (4,)
    assert source.out_dim == source.out_shape == (6,)
    torch.testing.assert_close(source.contract_dense(), tensor)
    configurations = tk.decompositions.ConfigurationBatch(torch.tensor([[2]]))
    torch.testing.assert_close(source.evaluate(configurations), tensor[2:3])


@pytest.mark.parametrize('name', ['in_dim', 'out_dim'])
def test_matrix_source_rejects_explicit_dimensions(name):
    with pytest.raises(TypeError, match='unexpected keyword argument'):
        tk.decompositions.DenseTensorSource.matrix(torch.ones(4, 6), **{name: (2, 2)})


def test_matrix_source_defaults_to_grouped_axes():
    tensor = torch.arange(120.).reshape(2, 3, 4, 5)
    source = tk.decompositions.DenseTensorSource.matrix(tensor)
    assert source.in_dim == (2, 3)
    assert source.out_dim == (4, 5)
    torch.testing.assert_close(source.contract_dense(), tensor.permute(0, 2, 1, 3))


def test_dense_input_sites_accept_range():
    source = tk.decompositions.DenseTensorSource(torch.ones(2, 3), in_sites=range(2))
    assert source.in_sites == (0, 1)
    assert source.out_shape == ()


@pytest.mark.parametrize('sites', [{0, 1}, iter((0, 1)), '01', b'01'])
def test_dense_input_sites_require_an_ordered_sequence(sites):
    with pytest.raises(TypeError, match='sequence of integers'):
        tk.decompositions.DenseTensorSource(torch.ones(2, 3), in_sites=sites)
