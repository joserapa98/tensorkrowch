"""Tests for svd/utils."""

import pytest
import torch

from tensorkrowch.decompositions.svd.utils import (_log_tensor_norm,
                                                   _normalize_tensor)


@pytest.mark.parametrize('n_axes', [1, 2, 3])
@pytest.mark.parametrize('keepdim', [False, True])
def test_stable_norm_extreme_batches(device_dtype, assert_close, n_axes, keepdim):
    device, dtype = device_dtype
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    large = 1e20 if real_dtype == torch.float32 else 1e200
    reference = torch.tensor([1., 2., -1., 0.], dtype=dtype, device=device)
    if dtype.is_complex:
        reference = reference * (1 + 1j)
    shape = (4,) if n_axes == 1 else (2, 2) if n_axes == 2 else (1, 2, 2)
    batch = torch.stack((reference * large, reference / large, reference * 0)).reshape(3, *shape)
    dim = tuple(range(-n_axes, 0))
    normalized, log_norm = _normalize_tensor(batch, dim)
    expected_log = torch.tensor([1., -1.], dtype=real_dtype, device=device) * torch.log(torch.tensor(large, dtype=real_dtype, device=device)) + reference.norm().log()
    assert_close(log_norm[:2], expected_log)
    assert torch.isneginf(log_norm[-1])
    assert_close(normalized[:2], (reference / reference.norm()).reshape(1, *shape).expand(2, *shape))
    assert_close(normalized[-1], torch.zeros(shape, dtype=dtype, device=device))
    expected = log_norm.reshape(3, *((1,) * n_axes)) if keepdim else log_norm
    assert_close(_log_tensor_norm(batch, dim, keepdim), expected)
