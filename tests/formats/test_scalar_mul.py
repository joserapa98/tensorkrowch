"""Uniform scalar magnitudes, phase placement and multiplication gradients."""

import pytest
import torch


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 5])
@pytest.mark.parametrize('n_batches', [0, 1])
@pytest.mark.parametrize('quantized', [False, True])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('scalar', [9., -9., 3 + 4j])
def test_scalar_magnitude_is_distributed_to_every_core(
        make_format, topology, n_sites, n_batches, quantized, dtype, scalar):
    format = make_format(topology, n_sites, n_batches=n_batches,
                         quantized=quantized, dtype=dtype)
    before = [core.clone() for core in format.cores]
    expected = format.contract_dense() * scalar
    magnitude = abs(scalar)**(1 / n_sites)
    phase = scalar / abs(scalar)
    for result in (format.mul(scalar), format * scalar, scalar * format):
        assert result.dtype == expected.dtype
        assert result.rank == format.rank
        assert result.batch_shape == format.batch_shape
        assert result.in_sites == format.in_sites
        if format.out_dim is not None:
            assert result.out_sites == format.out_sites
        for site, (core, original) in enumerate(zip(result.cores, before)):
            factor = magnitude * phase if site == 0 else magnitude
            torch.testing.assert_close(core,
                                       original.to(result.dtype) * factor)
        torch.testing.assert_close(result.contract_dense(), expected,
                                   rtol=1e-10, atol=1e-10)
        if quantized:
            if format.out_dim is None:
                assert result.layout == format.layout
            else:
                assert result.in_layout == format.in_layout
                assert result.out_layout == format.out_layout
    for core, original in zip(format.cores, before):
        torch.testing.assert_close(core, original, rtol=0, atol=0)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64,
                                 torch.complex64, torch.complex128])
@pytest.mark.parametrize('scalar', [
    torch.tensor(2, dtype=torch.int64),
    torch.tensor(0.25, dtype=torch.float64),
    torch.tensor(3 + 4j, dtype=torch.complex128),
    torch.tensor(True),
])
def test_scalar_tensor_dtype_promotion(make_format, topology, dtype, scalar):
    format = make_format(topology, dtype=dtype)
    expected = format.contract_dense() * scalar
    result = format.mul(scalar)
    assert result.dtype == expected.dtype
    torch.testing.assert_close(result.contract_dense(), expected,
                               rtol=5e-5, atol=5e-6)


@pytest.mark.parametrize('scalar', [0., 2., -3., 0j, 3 + 4j])
@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_scalar_gradients_include_zero(make_format, scalar, topology):
    format = make_format(topology, dtype=torch.complex128)
    dtype = torch.complex128 if isinstance(scalar, complex) else torch.float64
    value = torch.tensor(scalar, dtype=dtype, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda multiplier: format.mul(multiplier).contract_dense(),
        (value,), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('scalar', [0., -2., 3 + 4j])
@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_core_gradients_match_dense_scaling(make_format, scalar, topology):
    format = make_format(topology, dtype=torch.complex128)
    for core in format.cores:
        core.requires_grad_()
    expected = format.contract_dense() * scalar
    actual = format.mul(scalar).contract_dense()
    actual_grad = torch.autograd.grad(actual.real.sum(), tuple(format.cores))
    expected_grad = torch.autograd.grad(expected.real.sum(),
                                       tuple(format.cores))
    for actual_core, expected_core in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual_core, expected_core,
                                   atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_scalar_scaling_absorbs_bond_factors(make_format, topology):
    format = make_format(topology, dtype=torch.complex128)
    format.bonds = [torch.linspace(0.5, 1.5, rank, dtype=torch.float64)
                    for rank in format.rank]
    before = [core.clone() for core in format._effective_cores()]
    expected = format.contract_dense() * (3 + 4j)
    result = format.mul(3 + 4j)
    assert result.bonds is None
    assert format.bonds is not None
    magnitude = 5**(1 / format.n_sites)
    for site, (core, original) in enumerate(zip(result._effective_cores(),
                                               before)):
        factor = magnitude * (3 + 4j) / 5 if site == 0 else magnitude
        torch.testing.assert_close(core, original * factor)
    torch.testing.assert_close(result.contract_dense(), expected)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_named_mul_retains_hadamard_behavior(make_format, topology):
    left = make_format(topology)
    right = make_format(topology, seed=73)
    torch.testing.assert_close(left.mul(right).contract_dense(),
                               left.contract_dense() * right.contract_dense())


@pytest.mark.parametrize('scalar', [0., float('inf'), float('nan')])
def test_zero_and_nonfinite_scalars_retain_direct_scaling(make_format, scalar):
    format = make_format('tt')
    result = format.mul(scalar)
    torch.testing.assert_close(result.cores[0], format.cores[0] * scalar,
                               equal_nan=True)
    for core, original in zip(result.cores[1:], format.cores[1:]):
        torch.testing.assert_close(core, original)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_scalar_scaling_on_accelerators(make_format, topology, device_dtype):
    device, dtype = device_dtype
    format = make_format(topology, 3, dtype=dtype, device=device)
    scalar = torch.tensor(-8., dtype=format.cores[0].real.dtype,
                          device=format.device, requires_grad=True)
    result = format.mul(scalar)
    assert result.device == format.device
    for site, (core, original) in enumerate(zip(result.cores, format.cores)):
        torch.testing.assert_close(core, original * (-2 if site == 0 else 2))
    tolerance = 5e-5 if dtype in (torch.float32, torch.complex64) else 1e-9
    torch.testing.assert_close(result.contract_dense(),
                               format.contract_dense() * scalar,
                               atol=tolerance, rtol=tolerance)
    gradient, = torch.autograd.grad(result.contract_dense().real.sum(), scalar)
    expected = format.contract_dense().real.sum()
    torch.testing.assert_close(gradient, expected.to(gradient.dtype),
                               atol=tolerance, rtol=tolerance)
