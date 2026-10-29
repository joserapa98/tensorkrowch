"""Tests for private ALS QR factorization and absorption operations."""

import pytest
import torch

from tensorkrowch.decompositions.als.gauges import (_absorb_factor,
                                                    _factor_core)


def _pair_contraction(left, right):
    """Contracts neighboring standard cores over their shared rank."""
    return torch.einsum('apb,bqc->apqc', left, right)


@pytest.mark.parametrize('direction', ['forward', 'reverse'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('shapes', [
    ((1, 1, 1), (1, 1, 1)),
    ((1, 3, 2), (2, 4, 1)),
    ((2, 3, 4), (4, 5, 2)),
])
def test_qr_preserves_pair_and_produces_isometry(direction, dtype, shapes):
    generator = torch.Generator().manual_seed(1)
    left = torch.randn(*shapes[0], dtype=dtype, generator=generator)
    right = torch.randn(*shapes[1], dtype=dtype, generator=generator)
    core, neighbor = (left, right) if direction == 'forward' else (right, left)

    gauged, factor = _factor_core(core, direction)
    absorbed = _absorb_factor(factor, neighbor, direction)
    pair = (gauged, absorbed) if direction == 'forward' else (absorbed, gauged)

    assert gauged.shape == core.shape
    assert absorbed.shape == neighbor.shape
    torch.testing.assert_close(
        _pair_contraction(*pair), _pair_contraction(left, right))
    if direction == 'forward':
        matrix = gauged.reshape(-1, core.shape[-1])
        gram = matrix.mH @ matrix
    else:
        matrix = gauged.reshape(core.shape[0], -1)
        gram = matrix @ matrix.mH
    torch.testing.assert_close(gram, torch.eye(gram.shape[0], dtype=dtype))


@pytest.mark.parametrize('direction, shape', [
    ('forward', (1, 2, 3)), ('reverse', (3, 2, 1))])
def test_infeasible_qr_fails_without_mutation(direction, shape):
    core = torch.randn(shape)
    original = core.clone()

    with pytest.raises(ValueError, match='feasible'):
        _factor_core(core, direction)

    assert torch.equal(core, original)


@pytest.mark.parametrize('core, direction, error', [
    (None, 'forward', TypeError),
    (torch.ones(2, 2), 'forward', ValueError),
    (torch.ones(1, 2, 1), 'sideways', ValueError),
])
def test_invalid_factorization_arguments(core, direction, error):
    with pytest.raises(error):
        _factor_core(core, direction)


@pytest.mark.parametrize('factor, neighbor, direction, error', [
    (None, torch.ones(1, 2, 1), 'forward', TypeError),
    (torch.ones(1, 1), None, 'forward', TypeError),
    (torch.ones(1), torch.ones(1, 2, 1), 'forward', ValueError),
    (torch.ones(1, 1), torch.ones(2, 1), 'forward', ValueError),
    (torch.ones(1, 1), torch.ones(1, 2, 1), 'sideways', ValueError),
    (torch.ones(1, 1), torch.ones(1, 2, 1).double(), 'forward', ValueError),
    (torch.ones(2, 2), torch.ones(3, 2, 1), 'forward', ValueError),
    (torch.ones(2, 2), torch.ones(1, 2, 3), 'reverse', ValueError),
])
def test_invalid_absorption_arguments(factor, neighbor, direction, error):
    with pytest.raises(error):
        _absorb_factor(factor, neighbor, direction)


@pytest.mark.parametrize('direction', ['forward', 'reverse'])
def test_qr_pair_retains_gradients(direction):
    left = torch.randn(2, 3, 4, dtype=torch.complex128, requires_grad=True)
    right = torch.randn(4, 5, 2, dtype=torch.complex128, requires_grad=True)
    active, neighbor = (
        left, right) if direction == 'forward' else (right, left)
    gauged, factor = _factor_core(active, direction)
    absorbed = _absorb_factor(factor, neighbor, direction)
    pair = (gauged, absorbed) if direction == 'forward' else (absorbed, gauged)

    _pair_contraction(*pair).abs().square().sum().backward()

    assert (left.grad is not None) and torch.isfinite(left.grad).all()
    assert (right.grad is not None) and torch.isfinite(right.grad).all()
