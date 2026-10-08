"""Gauge cancellation and numerical finite-ring balancing."""

import pytest
import torch
import tensorkrowch as tk
from tensorkrowch.formats.orbits import GaugeOrbit, TensorRingOrbit


@pytest.mark.parametrize('topology', ['tr', 'trm'])
@pytest.mark.parametrize('method', ['adam', 'gradient'])
def test_ring_gauge_cancellation_and_balancing(make_format, device_dtype,
                                               assert_close, topology, method):
    device, dtype = device_dtype
    format = make_format(topology, 3, dtype=dtype, device=device)
    dense = format.contract_dense()
    orbit = TensorRingOrbit(format)
    gauges = [torch.diag(torch.linspace(1, 2, rank, device=device,
                                        dtype=format.cores[0].real.dtype)).to(dtype)
              for rank in format.rank]
    if device == 'mps' and dtype.is_complex:
        with pytest.raises(RuntimeError, match="doesn't support complex"):
            format.canonicalize_minimal(method=method)
        assert_close(format.contract_dense(), dense)
        return
    transformed = format.clone()
    transformed._set_standard_cores(orbit.apply(gauges))
    assert_close(transformed.contract_dense(), dense)
    transformed.cores[0].requires_grad_()
    before = sum(core.abs().square().sum() for core in transformed.cores)
    transformed.canonicalize_minimal(max_iter=80, method=method)
    after = sum(core.abs().square().sum() for core in transformed.cores)
    assert after < before
    assert_close(transformed.contract_dense(), dense)
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
@pytest.mark.parametrize('method', ['adam', 'gradient'])
def test_minimal_stopping_criteria(make_format, kind, reason, method):
    format = make_format(kind, 3, dtype=torch.float64)
    dense = format.contract_dense()
    options = dict(rtol=1e-15, stagnation_rtol=1e-15, max_iter=1)
    if reason == 'tolerance':
        options['rtol'] = 10.
    elif reason == 'stagnation':
        options.update(max_iter=8, stagnation_rtol=1., patience=2)
    _, info = format.canonicalize_minimal(method=method, return_info=True,
                                         **options)
    assert info.stop_reason == reason
    assert info.converged == (reason != 'max_iter')
    assert info.iterations == (3 if reason == 'stagnation' else 1)
    assert torch.allclose(info.gram_imbalance,
                          TensorRingOrbit(format).gram_imbalance(relative=True))
    assert torch.allclose(format.contract_dense(), dense, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('method', ['adam', 'gradient'])
def test_minimal_relative_stopping_is_scale_invariant(make_format, method):
    format = make_format('tr', 3, dtype=torch.float64)
    scaled = tk.formats.TR([100 * core for core in format.cores])
    options = dict(max_iter=12, rtol=1e-15, stagnation_rtol=1., patience=3,
                   method=method, return_info=True)
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


@pytest.mark.parametrize('method', ['adam', 'gradient'])
def test_minimal_internal_gradients_are_isolated(make_format, method):
    format = make_format('tr', 3, dtype=torch.float64)
    original = list(format.cores)
    for core in original:
        core.requires_grad_()
    format.canonicalize_minimal(max_iter=3, rtol=1e-15, method=method)
    assert all(core.grad is None for core in original)
    format.contract_dense().square().sum().backward()
    assert all(core.grad is not None and torch.isfinite(core.grad).all()
               for core in original)

    with torch.no_grad():
        _, info = make_format('tr', 3).canonicalize_minimal(
            max_iter=3, method=method, return_info=True)
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
    ({'method': 'newton'}, ValueError),
    ({'method': 1}, TypeError),
])
def test_minimal_stopping_options(make_format, kind, options, error):
    with pytest.raises(error):
        make_format(kind, 3).canonicalize_minimal(**options)


def test_gram_matrices_contract_batches_and_preserve_gradients(device_dtype, assert_close):
    device, dtype = device_dtype
    left = torch.arange(12).reshape(2, 2, 3).to(device=device, dtype=dtype)
    right = torch.arange(24).reshape(2, 3, 4).to(device=device, dtype=dtype)
    if dtype.is_complex:
        left = left + 1j * left.flip(-1)
        right = right + 1j * right.flip(-1)
    left.requires_grad_()
    right.requires_grad_()
    orbit = GaugeOrbit([left, right], [(0, -1, 1, 1)])
    left_gram, right_gram = orbit.gram_matrices()[0]
    expected_left = sum(core.conj().T @ core for core in left)
    expected_right = sum(core @ core.conj().T for core in right)
    assert_close(left_gram, expected_left)
    assert_close(right_gram, expected_right)
    assert_close(orbit.gram_imbalance(), (expected_left - expected_right).norm())
    (left_gram.real.sum() + right_gram.real.sum()).backward()
    assert left.grad is not None and right.grad is not None
    assert GaugeOrbit([left], []).gram_matrices() == []


def test_gradient_update_matches_simultaneous_increment(make_format):
    format = make_format('tr', 3, dtype=torch.complex128)
    orbit = TensorRingOrbit(format)
    norm_squared = sum(core.abs().square().sum() for core in orbit.cores)
    increments = [torch.matrix_exp(-0.05 * (left - right) / norm_squared)
                  for left, right in orbit.gram_matrices()]
    expected = orbit.apply(increments)
    before = sum(core.abs().square().sum() for core in orbit.cores)
    assert sum(core.abs().square().sum() for core in expected) < before
    format.canonicalize_minimal(method='gradient', max_iter=2, rtol=1e-15)
    assert all(torch.allclose(actual, target)
               for actual, target in zip(format.cores, expected))


@pytest.mark.parametrize('kind', ['tr', 'trm'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_gradient_search_needs_no_backward(make_format, monkeypatch, kind,
                                          dtype):
    def reject_backward(*args, **kwargs):
        raise AssertionError('Gradient gauge updates should not call backward')

    monkeypatch.setattr(torch.Tensor, 'backward', reject_backward)
    format = make_format(kind, 3, n_batches=1, dtype=dtype)
    dense = format.contract_dense()
    before = TensorRingOrbit(format).gram_imbalance(relative=True)
    _, info = format.canonicalize_minimal(method='gradient', max_iter=40,
                                         return_info=True)
    assert info.gram_imbalance < before
    assert torch.allclose(format.contract_dense(), dense, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('self_bond', [False, True])
def test_general_orbit_nondiagonal_exponential_gauges(device_dtype, assert_close,
                                                     self_bond):
    device, dtype = device_dtype
    generator = torch.Generator().manual_seed(47)
    gauge_parameter = (0.1 * torch.randn(3, 3, dtype=dtype,
                                        generator=generator)).to(device).requires_grad_()
    gauge = torch.matrix_exp(gauge_parameter)
    if self_bond:
        cores = [torch.randn(3, 2, 3, dtype=dtype, generator=generator).to(device)]
        orbit = GaugeOrbit(cores, [(0, 0, 0, -1)])
        original = cores[0].diagonal(dim1=0, dim2=-1).sum(-1)
        if device == 'mps' and dtype.is_complex:
            with pytest.raises(RuntimeError, match="doesn't support complex"):
                orbit.apply([gauge])
            return
        gauged = orbit.apply([gauge])
        actual = gauged[0].diagonal(dim1=0, dim2=-1).sum(-1)
    else:
        cores = [torch.randn(2, 3, 4, dtype=dtype, generator=generator).to(device),
                 torch.randn(5, 6, 3, dtype=dtype, generator=generator).to(device)]
        orbit = GaugeOrbit(cores, [(0, 1, 1, -1)])
        original = torch.tensordot(cores[0], cores[1], dims=([1], [2]))
        if device == 'mps' and dtype.is_complex:
            with pytest.raises(RuntimeError, match="doesn't support complex"):
                orbit.apply([gauge])
            return
        gauged = orbit.apply([gauge])
        actual = torch.tensordot(gauged[0], gauged[1], dims=([1], [2]))
    assert_close(actual, original)
    objective = orbit.objective([gauge])
    expected = sum(core.abs().square().sum() for core in gauged) / 2
    assert_close(objective, expected)
    objective.backward()
    assert gauge_parameter.grad is not None
    assert torch.isfinite(gauge_parameter.grad).all()


@pytest.mark.parametrize('bond', [
    (0, 1, 0, 1), (0, -1, 0, 1), (True, 0, 0, 1),
    (2, 0, 0, 1), (0, 2, 0, 1), (0, 0, 0, False),
])
def test_orbit_rejects_invalid_site_and_axis(bond):
    with pytest.raises(ValueError):
        GaugeOrbit([torch.eye(2)], [bond])


@pytest.mark.parametrize('gauges,error', [
    ([], ValueError), ([torch.eye(2), torch.eye(2)], ValueError),
    ([1], TypeError), ([torch.eye(3)], ValueError),
    ([torch.eye(2, dtype=torch.complex128)], ValueError),
    ([torch.eye(2, device='meta')], ValueError),
])
def test_orbit_gauge_type_shape_and_device_errors(gauges, error):
    orbit = GaugeOrbit([torch.eye(2), torch.eye(2)], [(0, 1, 1, 0)])
    with pytest.raises(error):
        orbit.apply(gauges)


@pytest.mark.parametrize('topology', ['tt', 'ttm'])
@pytest.mark.parametrize('quantized', [False, True])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_minimal_open_format_information(make_format, topology, quantized, dtype):
    format = make_format(topology, quantized=quantized, dtype=dtype)
    before = format.contract_dense()
    result, info = format.canonicalize_minimal(return_info=True)
    assert result is format
    assert info.iterations == 0 and info.converged
    assert info.gram_imbalance is None and info.stop_reason == 'vidal'
    assert format.bonds.powers == [(0.5, 0.5)] * (format.n_sites - 1)
    assert torch.allclose(format.contract_dense(), before, rtol=1e-10, atol=1e-11)
