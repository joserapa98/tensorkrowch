"""
This script contains:

    Public classes:
        * MinimalCanonicalInfo
        * GaugeOrbit
        * TensorRingOrbit
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import torch


if TYPE_CHECKING:
    from tensorkrowch.formats.formats1d import TR, TRM


@dataclass(frozen=True)
class MinimalCanonicalInfo:
    """Convergence information returned by minimal canonicalization.

    Parameters
    ----------
    iterations : int
        Number of gauge optimization iterations; zero for TT/TTM.
    converged : bool
        Whether the ring optimizer met its gradient tolerance; True for TT/TTM.
    balance_residual : torch.Tensor or None
        Largest final ring Gram imbalance; None for TT/TTM.
    """

    iterations: int  # Number of optimization iterations executed
    converged: bool  # Whether the gauge gradient met the stopping tolerance
    balance_residual: Optional[torch.Tensor]  # Final Gram imbalance for rings


class GaugeOrbit:
    """Tensor representations related by invertible virtual gauge transforms.

    Each bond identifies two tensor axes joined in the represented contraction.
    A gauge multiplies the first core and its inverse acts on the second, so
    the contracted tensor stays unchanged. Cores and gauges may be real or
    complex. The action accepts arbitrary tensor axes, allowing geometries
    beyond chains.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Nonempty collection of tensors. Containers are copied and tensor
        references retained.
    bonds : sequence of tuple[int, int, int, int]
        Virtual interfaces (left_site, left_axis, right_site, right_axis).
        Joined axis dimensions should match. Negative axes are accepted.
    """

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 bonds: Sequence[Tuple[int, int, int, int]]) -> None:
        self.cores = tuple(cores)
        self.bonds = tuple(bonds)
        if not self.cores or not all(isinstance(core, torch.Tensor)
                                     for core in self.cores):
            raise TypeError('`cores` should be a nonempty tensor sequence')
        for left_site, left_axis, right_site, right_axis in self.bonds:
            for site, axis in [(left_site, left_axis), (right_site, right_axis)]:
                if isinstance(site, bool) or not isinstance(
                    site, int) or not 0 <= site < len(self.cores):
                    raise ValueError('Gauge bonds should select valid tensor sites')
                if isinstance(axis, bool) or not isinstance(axis, int) or not (
                        -self.cores[site].ndim <= axis < self.cores[site].ndim):
                    raise ValueError('Gauge bonds should select valid tensor axes')
            if self.cores[left_site].shape[left_axis] != self.cores[right_site].shape[right_axis]:
                raise ValueError('Gauge bond dimensions should match')


    def apply(self, gauges: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Applies invertible gauges at the configured virtual interfaces.

        Parameters
        ----------
        gauges : sequence of torch.Tensor
            One square invertible gauge per configured bond, matching its rank
            and core dtype/device. The first core is multiplied by the gauge;
            its inverse acts on the second core through a linear solve.

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
        >>> identity = torch.eye(2, dtype=torch.complex128)
        >>> orbit = tk.formats.GaugeOrbit([identity, identity], [(0, 1, 1, 0)])
        >>> gauge = torch.matrix_exp(1j * identity)
        >>> cores = orbit.apply([gauge])
        >>> torch.allclose(cores[0] @ cores[1], identity)
        True
        """
        gauges = list(gauges)
        if len(gauges) != len(self.bonds):
            raise ValueError('There should be one gauge per virtual bond')

        cores = list(self.cores)
        for gauge, (left_site, left_axis, right_site, right_axis) in zip(gauges, self.bonds):
            rank = cores[left_site].shape[left_axis]
            if not isinstance(gauge, torch.Tensor):
                raise TypeError('Gauges should be tensors')
            if gauge.shape != (rank, rank):
                raise ValueError('Gauge dimensions should match the virtual bond')
            if gauge.device != cores[left_site].device or gauge.dtype != cores[left_site].dtype:
                raise ValueError('Gauges and cores should share device and dtype')
            # Apply the gauge and its inverse at the joined virtual axes.
            left_core = cores[left_site].movedim(left_axis, -1)
            cores[left_site] = (left_core @ gauge).movedim(-1, left_axis)

            right_core = cores[right_site].movedim(right_axis, 0)
            transformed = torch.linalg.solve(gauge, right_core.reshape(rank, -1))
            cores[right_site] = transformed.reshape(right_core.shape).movedim(0, right_axis)
        return cores


    def objective(self, gauges: Sequence[torch.Tensor]) -> torch.Tensor:
        """Returns half the sum of squared gauged core Frobenius norms.

        Parameters
        ----------
        gauges : sequence of torch.Tensor
            One square invertible gauge per configured bond, matching its rank
            and core dtype/device, as in :meth:`apply`.

        Returns
        -------
        torch.Tensor
            Real scalar objective retaining gradients through gauges.
        """
        return sum(core.abs().square().sum() / 2 for core in self.apply(gauges))


class TensorRingOrbit(GaugeOrbit):
    """Gauge-related representations of a finite tensor ring.

    Gauges act on adjacent virtual axes, including the closing bond. The
    objective and balance residual support both real and complex cores.

    Parameters
    ----------
    format : TR or TRM
        Cyclic format supplying cores with explicit bond factors absorbed.
        Matrix input/output axes are combined for the gauge action.
    """

    def __init__(self, format: Union['TR', 'TRM']) -> None:
        if not format._cyclic:
            raise ValueError('TensorRingOrbit requires a cyclic format')
        cores = format._effective_cores()
        super().__init__(cores, [(site, -1, (site + 1) % len(cores), -3)
                                 for site in range(len(cores))])


    def balance_residual(self) -> torch.Tensor:
        """Returns the largest virtual-bond Gram imbalance.

        Returns
        -------
        torch.Tensor
            Maximum Frobenius norm of left Gram minus right Gram across
            configured bonds. Structural batches are included in the Gram
            contractions; this is not a relative convergence tolerance.
        """
        residuals = []
        for left_site, _, right_site, _ in self.bonds:
            left_core, right_core = self.cores[left_site], self.cores[right_site]
            left_matrix = left_core.reshape(-1, left_core.shape[-1])
            right_matrix = right_core.movedim(-3, 0).reshape(right_core.shape[-3], -1)
            left_gram = left_matrix.transpose(-2, -1).conj() @ left_matrix
            right_gram = right_matrix @ right_matrix.transpose(-2, -1).conj()
            residuals.append((left_gram - right_gram).norm())
        return torch.stack(residuals).amax()
