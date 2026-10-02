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
    with pytest.raises(TypeError):
        zero.canonicalize_minimal(return_info=1)
