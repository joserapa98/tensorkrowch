"""Tests for ALS gauge factorization and absorption policies."""

import pytest

import torch
import tensorkrowch as tk


def _pair_contraction(left, right):
    """Contracts neighboring standard cores over their shared rank."""
    return torch.einsum('apb,bqc->apqc', left, right)


@pytest.mark.parametrize('gauge_name', ['qr', 'svd'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_forward_gauge_preserves_neighboring_tensor(gauge_name, dtype):
    generator = torch.Generator().manual_seed(1)
    left = torch.randn(2, 3, 4, dtype=dtype, generator=generator)
    right = torch.randn(4, 5, 2, dtype=dtype, generator=generator)
    gauge = tk.decompositions.QRGauge() if gauge_name == 'qr' \
        else tk.decompositions.SVDGauge()

    gauged, factor = gauge.factor(left, 'forward')
    absorbed = gauge.absorb(factor, right, 'forward')

    assert gauged.shape == left.shape
    assert absorbed.shape == right.shape
    assert torch.allclose(
        _pair_contraction(gauged, absorbed),
        _pair_contraction(left, right),
        atol=1e-11,
        rtol=1e-11)


@pytest.mark.parametrize('gauge_name', ['qr', 'svd'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_reverse_gauge_preserves_neighboring_tensor(gauge_name, dtype):
    generator = torch.Generator().manual_seed(2)
    left = torch.randn(2, 3, 4, dtype=dtype, generator=generator)
    right = torch.randn(4, 5, 2, dtype=dtype, generator=generator)
    gauge = tk.decompositions.QRGauge() if gauge_name == 'qr' \
        else tk.decompositions.SVDGauge()

    gauged, factor = gauge.factor(right, 'reverse')
    absorbed = gauge.absorb(factor, left, 'reverse')

    assert gauged.shape == right.shape
    assert absorbed.shape == left.shape
    assert torch.allclose(
        _pair_contraction(absorbed, gauged),
        _pair_contraction(left, right),
        atol=1e-11,
        rtol=1e-11)


def test_no_gauge_never_produces_a_factor():
    core = torch.randn(1, 3, 2)
    gauge = tk.decompositions.NoGauge()

    unchanged, factor = gauge.factor(core, 'forward')

    assert unchanged is core
    assert factor is None
    with pytest.raises(RuntimeError, match='does not produce'):
        gauge.absorb(torch.eye(2), torch.randn(2, 4, 1), 'forward')


def test_infeasible_factorization_fails_before_changing_rank():
    core = torch.randn(1, 2, 3)

    with pytest.raises(ValueError, match='not algebraically feasible'):
        tk.decompositions.QRGauge().factor(core, 'forward')
    with pytest.raises(ValueError, match='not algebraically feasible'):
        tk.decompositions.SVDGauge().factor(core, 'forward')


def test_gauge_invalidated_regions_include_receiver_only_when_present():
    gauge = tk.decompositions.QRGauge()

    assert gauge.invalidated_regions(1, 'forward', 4) == (1, 2)
    assert gauge.invalidated_regions(3, 'forward', 4) == (3,)
    assert gauge.invalidated_regions(2, 'reverse', 4) == (2, 1)
    assert tk.decompositions.NoGauge().invalidated_regions(
        2, 'reverse', 4) == (2,)
