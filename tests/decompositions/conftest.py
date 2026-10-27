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


def _source_case(kind, n_sites, dtype, matrix):
    """Builds equivalent source families from a small rank-two tensor train."""
    inputs = (3,) if n_sites == 1 else (2, 3, 2, 3, 2)
    outputs = ((2,) if n_sites == 1 else (3, 2, 2, 2, 2)) if matrix else None
    dimensions = tuple(i * o for i, o in zip(inputs, outputs)) \
        if matrix else inputs
    ranks = (1, *((2,) * (n_sites - 1)), 1)
    cores = [torch.randn(ranks[k], dim, ranks[k + 1], dtype=dtype)
             for k, dim in enumerate(dimensions)]
    cores[0][:, 0] = 0
    dense = cores[0]
    for core in cores[1:]:
        dense = torch.tensordot(dense, core, dims=([-1], [0]))
    dense = dense.squeeze(0).squeeze(-1)
    if matrix:
        dense = dense.reshape(*(dim for pair in zip(inputs, outputs)
                                 for dim in pair))

    if kind == 'dense':
        source = tk.decompositions.DenseTensorSource.matrix(
            dense, ordering='interleaved') if matrix else \
            tk.decompositions.DenseTensorSource(dense)
    elif kind == 'sparse':
        indices = dense.nonzero()
        values = dense[tuple(indices.T)]
        indices = torch.cat((indices, indices))
        values = torch.cat((values, values)) / 2
        source = tk.decompositions.SparseTensorSource.matrix(
            indices, values, inputs, outputs, ordering='interleaved') \
            if matrix else tk.decompositions.SparseTensorSource(
                indices, values, inputs)
    elif kind == 'callable':
        if matrix:
            def function(indices):
                grouped = dense.permute(*range(0, dense.ndim, 2),
                                         *range(1, dense.ndim, 2))
                return grouped[tuple(indices.T)]
            source = tk.decompositions.CallableTensorSource.matrix(
                function, inputs, outputs)
        else:
            source = tk.decompositions.CallableTensorSource(
                lambda indices: dense[tuple(indices.T)], inputs)
    else:
        if matrix:
            cores = [core.reshape(core.shape[0], i, o, core.shape[-1])
                     .permute(0, 1, 3, 2)
                     for core, i, o in zip(cores, inputs, outputs)]
        if kind == 'ring':
            format = tk.formats.TRM(cores) if matrix else tk.formats.TR(cores)
        else:
            compact = list(cores)
            compact[0] = compact[0].squeeze(0)
            compact[-1] = compact[-1].squeeze(-2 if matrix else -1)
            format = tk.formats.TTM(compact) if matrix else tk.formats.TT(compact)
        source = tk.decompositions.FormatSource(
            cores if kind == 'cores' else format)
    return source, dense


@pytest.fixture(params=[
    pytest.param((kind, n_sites, dtype),
                 id=f'{kind}-{n_sites}-{str(dtype).split(".")[-1]}')
    for kind in ('dense', 'sparse', 'callable', 'format', 'ring', 'cores')
    for n_sites in (1, 5)
    for dtype in (torch.float64, torch.complex128)
])
def vector_source_case(request):
    kind, n_sites, dtype = request.param
    return _source_case(kind, n_sites, dtype, matrix=False)


@pytest.fixture(params=[
    pytest.param((kind, n_sites, dtype),
                 id=f'{kind}-{n_sites}-{str(dtype).split(".")[-1]}')
    for kind in ('dense', 'sparse', 'callable', 'format', 'ring')
    for n_sites in (1, 5)
    for dtype in (torch.float64, torch.complex128)
])
def matrix_source_case(request):
    kind, n_sites, dtype = request.param
    return _source_case(kind, n_sites, dtype, matrix=True)
