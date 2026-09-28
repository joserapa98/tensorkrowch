"""Gauge actions by tensor axes and experimental finite-ring norm balancing."""

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class MinimalCanonicalInfo:
    r"""Optional convergence information for finite-ring gauge optimization.

    Frozen record: field references cannot be reassigned. Tensor contents and
    autograd are preserved without copying or detaching.

    Parameters
    ----------
    iterations : int
        Number of optimization iterations executed; zero for the direct
        train path.
    converged : bool
        Whether the ring optimizer met its gradient tolerance; True for the
        direct train path.
    balance_residual : torch.Tensor or None
        Largest final ring Gram imbalance; None for the direct train path.
    """

    iterations: int  # Number of optimization iterations executed
    converged: bool  # Whether the gauge gradient met the stopping tolerance
    balance_residual: Optional[torch.Tensor]  # Final Gram imbalance for rings


class GaugeOrbit:
    r"""Invertible gauge action on arbitrary pairs of virtual tensor axes.

    bonds contains ``(left_site, left_axis, right_site, right_axis)`` entries.
    A gauge multiplies the left endpoint; its inverse acts on the right
    endpoint via solve. This axis-based action does not assume a 1D geometry.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Nonempty collection of tensors. Containers are copied and tensor
        references retained.
    bonds : sequence of tuple[int, int, int, int]
        Virtual interfaces (left_site, left_axis, right_site, right_axis).
        Endpoint axis dimensions should match. Negative axes are accepted.
    """

    def __init__(self, cores: Sequence[torch.Tensor], bonds) -> None:
        """Stores tensor references and validates gauge interfaces."""
        self.cores = tuple(cores)
        self.bonds = tuple(bonds)
        if not self.cores or not all(isinstance(core, torch.Tensor)
                                     for core in self.cores):
            raise TypeError('`cores` should be a nonempty tensor sequence')
        for left, left_axis, right, right_axis in self.bonds:
            for site, axis in [(left, left_axis), (right, right_axis)]:
                if isinstance(site, bool) or not isinstance(
                    site, int) or not 0 <= site < len(self.cores):
                    raise ValueError('Gauge endpoints should select valid tensor sites')
                if isinstance(axis, bool) or not isinstance(axis, int) or not - \
                              self.cores[site].ndim <= axis < self.cores[site].ndim:
                    raise ValueError('Gauge endpoints should select valid tensor axes')
            if self.cores[left].shape[left_axis] != self.cores[right].shape[right_axis]:
                raise ValueError('Gauge endpoint dimensions should match')

    def apply(self, gauges):
        r"""Applies invertible gauges at the configured virtual interfaces.

        Parameters
        ----------
        gauges : sequence of torch.Tensor
            One square invertible gauge per configured bond, matching its rank
            and the endpoint dtype/device. The left endpoint is multiplied by
            the gauge; the inverse acts on the right endpoint via solve.

        Returns
        -------
        list[torch.Tensor]
            Transformed cores. Multiplication and inverse solve preserve the
            contracted tensor. Singular gauges propagate a linear algebra error.

        Examples
        --------
        >>> orbit = tk.formats.GaugeOrbit([torch.eye(2), torch.eye(2)],
        ...     [(0, 1, 1, 0)])
        >>> cores = orbit.apply([2 * torch.eye(2)])
        >>> torch.allclose(cores[0] @ cores[1], torch.eye(2))
        True
        """
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
        r"""Returns half the sum of squared gauged core Frobenius norms.

        Parameters
        ----------
        gauges : sequence of torch.Tensor
            One square invertible gauge per configured bond, matching its rank
            and the endpoint dtype/device. The left endpoint is multiplied by
            the gauge; the inverse acts on the right endpoint via solve.

        Returns
        -------
        torch.Tensor
            Real scalar objective retaining gradients through gauges.
        """
        return sum(core.abs().square().sum() / 2 for core in self.apply(gauges))


class TensorRingOrbit(GaugeOrbit):
    r"""Finite-ring gauge orbit, with physical axes fused within each core.

    Parameters
    ----------
    format : TR or TRM
        Cyclic format supplying effective standard cores, including diagonal
        factors. Matrix physical axes are fused for the gauge action.
    """

    def __init__(self, format) -> None:
        """Builds cyclic gauge interfaces from effective format cores."""
        if not format._cyclic:
            raise ValueError('TensorRingOrbit requires a cyclic format')
        cores = format._standard_cores()
        super().__init__(cores, [(site, -1, (site + 1) % len(cores), -3)
                                 for site in range(len(cores))])

    def balance_residual(self):
        r"""Returns the largest virtual-bond Gram imbalance.

        Returns
        -------
        torch.Tensor
            Maximum Frobenius norm of left Gram minus right Gram across
            configured bonds. Structural batches are included in the Gram
            contractions; this is not a relative convergence tolerance.
        """
        residuals = []
        for left, _, right, _ in self.bonds:
            a = self.cores[left].reshape(-1, self.cores[left].shape[-1])
            b = self.cores[right].movedim(-3,
                                          0).reshape(self.cores[right].shape[-3], -1)
            residuals.append((a.transpose(-2, -1).conj() @ a -
                              b @ b.transpose(-2, -1).conj()).norm())
        return torch.stack(residuals).amax()
