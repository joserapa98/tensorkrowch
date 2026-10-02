"""Gauge cancellation and numerical finite-ring balancing."""

import pytest
import torch
import tensorkrowch as tk
from tensorkrowch.formats.orbits import GaugeOrbit, TensorRingOrbit


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_ring_gauge_cancellation_and_balancing(make_format, dtype):
    format = make_format('tr', 3, dtype=dtype)
    dense = format.contract_dense()
    orbit = TensorRingOrbit(format)
    gauges = [torch.diag(torch.linspace(1, 2, rank, dtype=torch.float64)).to(dtype)
              for rank in format.rank]
    transformed = tk.formats.TR(orbit.apply(gauges))
    assert torch.allclose(transformed.contract_dense(), dense)
    transformed.cores[0].requires_grad_()
    before = sum(core.abs().square().sum() for core in transformed.cores)
    transformed.canonicalize_minimal(max_iter=80)
    after = sum(core.abs().square().sum() for core in transformed.cores)
    assert after < before
    assert torch.allclose(transformed.contract_dense(), dense, rtol=1e-9, atol=1e-10)
    assert all(core.grad is None for core in format.cores)
    assert transformed.dtype == dtype


def test_orbit_invalid_gauge():
    orbit = GaugeOrbit([torch.ones(2, 3), torch.ones(3, 2)], [(0, 1, 1, 0)])
    with pytest.raises(ValueError):
        orbit.apply([torch.eye(2)])
    with pytest.raises(RuntimeError):
        orbit.apply([torch.zeros(3, 3)])
    with pytest.raises(ValueError):
        GaugeOrbit(orbit.cores, [(0, 0, 1, 0)])


def test_minimal_train_route(make_format):
    format = make_format()
    dense = format.contract_dense()
    format.canonicalize_minimal()
    assert format.bonds.powers == [(0.5, 0.5)] * (format.n_sites - 1)
    assert torch.allclose(format.contract_dense(), dense)
    with pytest.raises(ValueError):
        format.canonicalize_minimal(max_iter=0)


def test_minimal_optional_convergence_information(make_format):
    format = make_format('tr', 3)
    _, info = format.canonicalize_minimal(max_iter=1, return_info=True)
    assert info.iterations == 1
    assert isinstance(info.converged, bool)
    assert torch.isfinite(info.gram_imbalance)
    zero = tk.formats.TR([torch.zeros(1, 2, 1)])
    _, info = zero.canonicalize_minimal(return_info=True)
    assert info.converged and info.iterations == 0
    assert info.gram_imbalance == 0
    assert info.stop_reason == 'tolerance'
    with pytest.raises(TypeError):
        zero.canonicalize_minimal(return_info=1)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_joint_gram_imbalance_and_relative_scaling(dtype):
    cores = [torch.ones(1, 1, 1, dtype=dtype),
             2 * torch.ones(1, 1, 1, dtype=dtype)]
    orbit = TensorRingOrbit(tk.formats.TR(cores))
    expected = torch.tensor(18., dtype=torch.float64).sqrt()
    assert torch.allclose(orbit.gram_imbalance(), expected)
    assert torch.allclose(orbit.gram_imbalance(relative=True), expected / 5)
    scaled = TensorRingOrbit(tk.formats.TR([7 * core for core in cores]))
    assert torch.allclose(scaled.gram_imbalance(), 49 * expected)
    assert torch.allclose(scaled.gram_imbalance(relative=True), expected / 5)
    zero = TensorRingOrbit(tk.formats.TR([torch.zeros_like(core)
                                        for core in cores]))
    assert zero.gram_imbalance(relative=True) == 0
    with pytest.raises(TypeError, match='`relative`'):
        orbit.gram_imbalance(relative=1)


@pytest.mark.parametrize('kind', ['tr', 'trm'])
@pytest.mark.parametrize('reason', ['tolerance', 'stagnation', 'max_iter'])
def test_minimal_stopping_criteria(make_format, kind, reason):
    format = make_format(kind, 3, dtype=torch.float64)
    dense = format.contract_dense()
    options = dict(rtol=1e-15, stagnation_rtol=1e-15, max_iter=1)
    if reason == 'tolerance':
        options['rtol'] = 10.
    elif reason == 'stagnation':
        options.update(max_iter=8, stagnation_rtol=1., patience=2)
    _, info = format.canonicalize_minimal(return_info=True, **options)
    assert info.stop_reason == reason
    assert info.converged == (reason != 'max_iter')
    assert info.iterations == (3 if reason == 'stagnation' else 1)
    assert torch.allclose(info.gram_imbalance,
                          TensorRingOrbit(format).gram_imbalance(relative=True))
    assert torch.allclose(format.contract_dense(), dense, rtol=1e-9, atol=1e-10)


def test_minimal_relative_stopping_is_scale_invariant(make_format):
    format = make_format('tr', 3, dtype=torch.float64)
    scaled = tk.formats.TR([100 * core for core in format.cores])
    options = dict(max_iter=12, rtol=1e-15, stagnation_rtol=1., patience=3,
                   return_info=True)
    _, info = format.canonicalize_minimal(**options)
    _, scaled_info = scaled.canonicalize_minimal(**options)
    assert (info.iterations, info.stop_reason) == (
        scaled_info.iterations, scaled_info.stop_reason)
    assert torch.allclose(info.gram_imbalance, scaled_info.gram_imbalance)


def test_minimal_stagnation_counter_resets(make_format, monkeypatch):
    measure = GaugeOrbit.gram_imbalance
    errors = iter([0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9])

    def controlled_imbalance(orbit, relative=False):
        value = next(errors, None)
        if value is None:
            return measure(orbit, relative=relative)
        return orbit.cores[0].real.new_tensor(value)

    monkeypatch.setattr(GaugeOrbit, 'gram_imbalance', controlled_imbalance)
    _, info = make_format('tr', 3).canonicalize_minimal(
        max_iter=10, rtol=1e-8, stagnation_rtol=1e-6, patience=3,
        return_info=True)
    assert info.stop_reason == 'stagnation'
    assert info.iterations == 7


def test_minimal_internal_gradients_are_isolated(make_format):
    format = make_format('tr', 3, dtype=torch.float64)
    original = list(format.cores)
    for core in original:
        core.requires_grad_()
    format.canonicalize_minimal(max_iter=3, rtol=1e-15)
    assert all(core.grad is None for core in original)
    format.contract_dense().square().sum().backward()
    assert all(core.grad is not None and torch.isfinite(core.grad).all()
               for core in original)

    with torch.no_grad():
        _, info = make_format('tr', 3).canonicalize_minimal(
            max_iter=3, return_info=True)
    assert torch.isfinite(info.gram_imbalance)


def test_minimal_numerical_failure_returns_finite_fallback(make_format,
                                                        monkeypatch):
    def nonfinite_step(optimizer):
        with torch.no_grad():
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    parameter.fill_(float('nan'))

    monkeypatch.setattr(torch.optim.Adam, 'step', nonfinite_step)
    format = make_format('tr', 3, dtype=torch.float64)
    dense = format.contract_dense()
    _, info = format.canonicalize_minimal(rtol=1e-15, return_info=True)
    assert not info.converged
    assert info.stop_reason == 'numerical_failure'
    assert torch.isfinite(info.gram_imbalance)
    assert torch.allclose(format.contract_dense(), dense)


@pytest.mark.parametrize('kind', ['tt', 'tr'])
@pytest.mark.parametrize('options, error', [
    ({'rtol': 0}, ValueError),
    ({'rtol': float('nan')}, ValueError),
    ({'stagnation_rtol': float('inf')}, ValueError),
    ({'stagnation_rtol': True}, TypeError),
    ({'patience': 0}, ValueError),
    ({'patience': 1.5}, TypeError),
])
def test_minimal_stopping_options(make_format, kind, options, error):
    with pytest.raises(error):
        make_format(kind, 3).canonicalize_minimal(**options)
