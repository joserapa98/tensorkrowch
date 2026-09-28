"""Gauge actions by tensor axes and experimental finite-ring norm balancing."""

from math import isfinite
from numbers import Real

import torch


class GaugeOrbit:
    """Invertible gauge action on arbitrary pairs of virtual tensor axes.

    bonds contains ``(left_site, left_axis, right_site, right_axis)`` entries.
    A gauge multiplies the left endpoint; its inverse acts on the right
    endpoint via solve. This axis-based action does not assume a 1D geometry.
    """

    def __init__(self, cores, bonds):
        self.cores = tuple(cores)
        self.bonds = tuple(bonds)
        if not self.cores or not all(isinstance(core, torch.Tensor) for core in self.cores):
            raise TypeError('`cores` should be a nonempty tensor sequence')
        for left, left_axis, right, right_axis in self.bonds:
            for site, axis in [(left, left_axis), (right, right_axis)]:
                if isinstance(site, bool) or not isinstance(site, int) or not 0 <= site < len(self.cores):
                    raise ValueError('Gauge endpoints should select valid tensor sites')
                if isinstance(axis, bool) or not isinstance(axis, int) or not -self.cores[site].ndim <= axis < self.cores[site].ndim:
                    raise ValueError('Gauge endpoints should select valid tensor axes')
            if self.cores[left].shape[left_axis] != self.cores[right].shape[right_axis]:
                raise ValueError('Gauge endpoint dimensions should match')

    def apply(self, gauges):
        """Applies matching square gauges; singular solves propagate an error."""
        gauges = list(gauges)
        if len(gauges) != len(self.bonds):
            raise ValueError('There should be one gauge per virtual bond')
        cores = list(self.cores)
        for gauge, (left, left_axis, right, right_axis) in zip(gauges, self.bonds):
            rank = cores[left].shape[left_axis]
            if not isinstance(gauge, torch.Tensor):
                raise TypeError('Gauges should be tensors')
            if gauge.shape != (rank, rank):
                raise ValueError('Gauge dimensions should match the virtual bond')
            if gauge.device != cores[left].device or gauge.dtype != cores[left].dtype:
                raise ValueError('Gauges and cores should share device and dtype')
            value = cores[left].movedim(left_axis, -1)
            cores[left] = (value @ gauge).movedim(-1, left_axis)
            value = cores[right].movedim(right_axis, 0)
            transformed = torch.linalg.solve(gauge, value.reshape(rank, -1))
            cores[right] = transformed.reshape(value.shape).movedim(0, right_axis)
        return cores

    def objective(self, gauges):
        """Returns half the sum of squared core Frobenius norms."""
        return sum(core.abs().square().sum() / 2 for core in self.apply(gauges))


class TensorRingOrbit(GaugeOrbit):
    """Finite-ring gauge orbit, with physical axes fused within each core."""

    def __init__(self, network):
        network._ensure_valid()
        if not network._topology.startswith('tr'):
            raise ValueError('TensorRingOrbit requires a cyclic format')
        cores = network._standard_cores()
        super().__init__(cores, [(site, -1, (site + 1) % len(cores), -3)
                                for site in range(len(cores))])

    def balance_residual(self):
        """Returns the maximum virtual-bond Gram imbalance."""
        residuals = []
        for left, _, right, _ in self.bonds:
            a = self.cores[left].reshape(-1, self.cores[left].shape[-1])
            b = self.cores[right].movedim(-3, 0).reshape(self.cores[right].shape[-3], -1)
            residuals.append((a.transpose(-2, -1).conj() @ a -
                              b @ b.transpose(-2, -1).conj()).norm())
        return torch.stack(residuals).amax()


def canonicalize_minimal(network, max_iter=200, lr=0.05, tol=1e-8):
    """Balances a finite ring through Hermitian exponential gauges.

    This is an experimental finite, nonuniform-ring adaptation of the gauge
    norm objective, inspired by Acuaviva et al., https://arxiv.org/pdf/2209.14358.
    It does not assert the uniform-network theorems or uniqueness of a minimum.
    Batched cores use a common gauge minimizing their summed objective.
    """
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise TypeError('`max_iter` should be int type')
    if max_iter < 1:
        raise ValueError('`max_iter` should be positive')
    for name, value in [('lr', lr), ('tol', tol)]:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'`{name}` should be a real number')
        if not isfinite(value) or value <= 0:
            raise ValueError(f'`{name}` should be finite and positive')
    network._ensure_valid()
    if not network._topology.startswith('tr'):
        return network.canonicalize_vidal('implicit')
    orbit = TensorRingOrbit(network)
    if not all(torch.isfinite(core).all() for core in orbit.cores):
        raise ValueError('Minimal canonicalization requires finite cores')
    detached = GaugeOrbit([core.detach() for core in orbit.cores], orbit.bonds)
    best = [torch.eye(rank, dtype=network.dtype, device=network.device) for rank in network._rank]
    scale = max(core.abs().amax().item() for core in detached.cores)
    if scale == 0:
        return network
    detached.cores = tuple(core / scale for core in detached.cores)
    best_loss = detached.objective(best).item()
    with torch.enable_grad():
        parameters = [torch.zeros_like(gauge, requires_grad=True) for gauge in best]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        for _ in range(max_iter):
            optimizer.zero_grad()
            gauges = [torch.matrix_exp((parameter + parameter.transpose(-2, -1).conj()) / 2)
                      for parameter in parameters]
            if not all(torch.isfinite(gauge).all() for gauge in gauges):
                break
            try:
                loss = detached.objective(gauges)
            except torch.linalg.LinAlgError:
                break
            if not torch.isfinite(loss):
                break
            value = loss.item()
            if value < best_loss:
                best_loss = value
                best = [gauge.detach() for gauge in gauges]
            loss.backward()
            if not all(torch.isfinite(parameter.grad).all() for parameter in parameters):
                break
            if max(parameter.grad.abs().amax().item() for parameter in parameters) <= tol:
                break
            optimizer.step()
    network._set_standard_cores(orbit.apply(best))
    return network
