"""Gauge cancellation and numerical finite-ring balancing."""

import pytest
import torch
import tensorkrowch as tk
from tensorkrowch.formats.orbits import GaugeOrbit, TensorRingOrbit


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_ring_gauge_cancellation_and_balancing(make_format, dtype):
    network = make_format('tr', 3, dtype=dtype)
    dense = network.contract_dense()
    orbit = TensorRingOrbit(network)
    gauges = [torch.diag(torch.linspace(1, 2, rank, dtype=torch.float64)).to(dtype)
              for rank in network.rank]
    transformed = tk.formats.TensorRing(orbit.apply(gauges))
    assert torch.allclose(transformed.contract_dense(), dense)
    transformed.cores[0].requires_grad_()
    before = sum(core.abs().square().sum() for core in transformed.cores)
    transformed.canonicalize_minimal(max_iter=80)
    after = sum(core.abs().square().sum() for core in transformed.cores)
    assert after < before
    assert torch.allclose(transformed.contract_dense(), dense, rtol=1e-9, atol=1e-10)
    assert all(core.grad is None for core in network.cores)
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
    network = make_format()
    dense = network.contract_dense()
    network.canonicalize_minimal()
    assert network.bonds.powers == [(0.5, 0.5)] * (network.n_sites - 1)
    assert torch.allclose(network.contract_dense(), dense)
    with pytest.raises(ValueError):
        network.canonicalize_minimal(max_iter=0)
