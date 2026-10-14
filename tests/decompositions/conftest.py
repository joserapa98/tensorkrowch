"""Shared numerical configurations for decomposition tests."""

import pytest
import torch

import tensorkrowch as tk


@pytest.fixture(autouse=True)
def isolated_random_state():
    with torch.random.fork_rng(), tk.svd_method('svd', refine=False):
        torch.manual_seed(0)
        yield


@pytest.fixture(params=[
    pytest.param((device, dtype), id=device + '-' + str(dtype).split('.')[-1])
    for device, dtypes in (
        ('cpu', (torch.float32, torch.float64, torch.complex64, torch.complex128)),
        ('cuda', (torch.float32, torch.float64, torch.complex64, torch.complex128)),
        ('mps', (torch.float32, torch.complex64)),
    )
    for dtype in dtypes
])
def device_dtype(request):
    device, dtype = request.param
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is unavailable')
    return device, dtype


@pytest.fixture
def assert_close(device_dtype):
    _, dtype = device_dtype
    low_precision = dtype in (torch.float32, torch.complex64)

    def check(actual, expected):
        torch.testing.assert_close(
            actual, expected, rtol=5e-5 if low_precision else 1e-9,
            atol=5e-6 if low_precision else 1e-10)

    return check
