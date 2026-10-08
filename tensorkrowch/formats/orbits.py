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
    """
    Convergence information returned by minimal canonicalization.

    Parameters
    ----------
    iterations : int
        Number of gauge optimization iterations; zero for TT/TTM.
    converged : bool
        Whether a relative imbalance or stagnation criterion was met;
        ``True`` for TT/TTM. Stagnation need not imply a small imbalance.
    gram_imbalance : torch.Tensor or None
        Final relative ring Gram imbalance, as returned by
        :meth:`GaugeOrbit.gram_imbalance` with ``relative=True``;
        ``None`` for TT/TTM.
    stop_reason : str or None
        ``'tolerance'``, ``'stagnation'``, ``'max_iter'`` or
        ``'numerical_failure'`` for rings; ``'vidal'`` for TT/TTM.
        ``None`` when no reason is supplied.
    """

    iterations: int  # Number of optimization iterations executed
    converged: bool  # Whether a stopping criterion was met
    gram_imbalance: Optional[torch.Tensor]  # Final relative imbalance for rings
    stop_reason: Optional[str] = None


class GaugeOrbit:
    """
    Tensor representations related by invertible virtual gauge transforms.

    Each bond identifies two tensor axes joined in the represented contraction.
    A gauge multiplies the first core and its inverse acts on the second, so
    the contracted tensor stays unchanged. Cores and gauges may be real or
    complex. The action accepts arbitrary tensor axes, allowing geometries
    beyond chains.

    Each bond is described by ``(left_site, left_axis, right_site, right_axis)``.
    ``left_site`` and ``right_site`` index the two cores, while ``left_axis``
    and ``right_axis`` select their connected axes. For cores shaped
    ``(left_rank, in_dim, right_rank)``, the bond ``(0, -1, 1, -3)`` connects
    the right rank of core 0 to the left rank of core 1.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Nonempty collection of tensors. Containers are copied and tensor
        references retained.
    bonds : sequence of tuple[int, int, int, int]
        Connections ``(left_site, left_axis, right_site, right_axis)``.
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
                if isinstance(site, bool) or not isinstance(site, int) or not (
                    0 <= site < len(self.cores)):
                    raise ValueError(
                        'Gauge bonds should select valid tensor sites')
                if isinstance(axis, bool) or not isinstance(axis, int) or not (
                        -self.cores[site].ndim <= axis < self.cores[site].ndim):
                    raise ValueError(
                        'Gauge bonds should select valid tensor axes')
            if left_site == right_site:
                ndim = self.cores[left_site].ndim
                if (left_axis % ndim) == (right_axis % ndim):
                    raise ValueError(
                        'A gauge bond should connect two distinct tensor axes')
            if self.cores[left_site].shape[left_axis] != \
                self.cores[right_site].shape[right_axis]:
                raise ValueError('Gauge bond dimensions should match')

    def apply(self, gauges: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """
        Applies invertible gauges at the configured bonds.

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
            contracted tensor. Gauges must be invertible. Singular gauges may
            cause ``torch.linalg.solve`` to raise ``torch.linalg.LinAlgError``.

        Examples
        --------
        >>> orbit = tk.formats.GaugeOrbit([torch.eye(2), torch.eye(2)],
        ...                               [(0, 1, 1, 0)])
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
        for gauge, (left_site, left_axis,
                    right_site, right_axis) in zip(gauges, self.bonds):
            rank = cores[left_site].shape[left_axis]

            if not isinstance(gauge, torch.Tensor):
                raise TypeError('Gauges should be tensors')
            if gauge.shape != (rank, rank):
                raise ValueError(
                    'Gauge dimensions should match the virtual bond')
            if (gauge.device != cores[left_site].device) or (
                gauge.dtype != cores[left_site].dtype):
                raise ValueError(
                    'Gauges and cores should share device and dtype')

            # Apply the gauge and its inverse at the joined virtual axes.
            left_core = cores[left_site].movedim(left_axis, -1)
            cores[left_site] = (left_core @ gauge).movedim(-1, left_axis)

            right_core = cores[right_site].movedim(right_axis, 0)
            transformed = torch.linalg.solve(gauge, right_core.reshape(rank, -1))
            cores[right_site] = transformed.reshape(right_core.shape).\
                movedim(0, right_axis)

        return cores

    def objective(self, gauges: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Computes the objective used for minimal canonicalization: half the sum
        of the squared Frobenius norms of the cores after applying the gauges.

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

    def gram_matrices(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Constructs the left and right Gram matrices at each configured bond.

        For each bond, the left core is reshaped into :math:`L_b` with the
        bond axis in its columns; the right core becomes :math:`R_b` with
        that axis in its rows. All other axes, including structural batches,
        are contracted to form :math:`L_b^\dagger L_b` and
        :math:`R_b R_b^\dagger`. These are individual-core contractions,
        not contracted subchains.

        Returns
        -------
        list[tuple[torch.Tensor, torch.Tensor]]
            One ``(left_gram, right_gram)`` pair per bond, in the order of
            ``bonds``. Both matrices have shape ``(rank, rank)`` and retain
            gradients through the cores. Returns an empty list without bonds.

        Examples
        --------
        >>> orbit = tk.formats.GaugeOrbit([2 * torch.eye(2), torch.eye(2)],
        ...                               [(0, 1, 1, 0)])
        >>> left, right = orbit.gram_matrices()[0]
        >>> left, right
        (tensor([[4., 0.],
                 [0., 4.]]),
         tensor([[1., 0.],
                 [0., 1.]]))
        """
        grams = []
        for left_site, left_axis, right_site, right_axis in self.bonds:
            left_core, right_core = self.cores[left_site], self.cores[right_site]

            left_matrix = left_core.movedim(left_axis, -1).reshape(
                -1, left_core.shape[left_axis])
            right_matrix = right_core.movedim(right_axis, 0).reshape(
                right_core.shape[right_axis], -1)

            left_gram = left_matrix.transpose(-2, -1).conj() @ left_matrix
            right_gram = right_matrix @ right_matrix.transpose(-2, -1).conj()
            grams.append((left_gram, right_gram))

        return grams

    def gram_imbalance(self, relative: bool = False) -> torch.Tensor:
        r"""
        Measures the joint Gram-matrix imbalance across the bonds.

        Uses the left and right Gram matrices from :meth:`gram_matrices`
        to compute

        .. math::

            D = \sqrt{\sum_b
                \left\|L_b^\dagger L_b - R_b R_b^\dagger\right\|_F^2}.

        With ``relative=True``, returns :math:`D / \sum_i\|A_i\|_F^2`,
        where :math:`A_i` are the current cores. This is the norm of the
        gradient of their joint log-norm along Hermitian gauge directions,
        following Definition 5.9 of `The minimal canonical form of a tensor
        network <https://arxiv.org/pdf/2209.14358>`_. The denominator is twice
        :meth:`objective` at identity gauges, not the squared norm of the
        contracted tensor. The relative measure is unchanged by a common
        rescaling of all cores; all-zero cores have zero imbalance.

        Equality of these matrices at every bond is necessary and sufficient
        for minimizing :meth:`objective` over the gauge orbit, that is, for
        being in minimal canonical form. Structural batch axes are summed
        in these contractions, so the condition concerns the total objective.

        This method compares individual cores. The open-chain minimal
        canonical condition instead compares contracted left/right subchains;
        its implicit Vidal representation need not have zero local imbalance.

        Parameters
        ----------
        relative : bool
            Whether to divide by the sum of squared core norms.

        Returns
        -------
        torch.Tensor
            Joint absolute or relative Gram imbalance. This measures
            stationarity of the core-norm objective, not reconstruction error
            or distance to a minimal canonical form.
        """
        if not isinstance(relative, bool):
            raise TypeError('`relative` should be bool type')
        if not self.bonds:
            return self.cores[0].real.new_zeros(())

        imbalances = [(left - right).norm()
                      for left, right in self.gram_matrices()]

        imbalance = torch.stack(imbalances).norm()
        if relative:
            norm_squared = sum(core.abs().square().sum() for core in self.cores)
            if norm_squared == 0:
                return imbalance
            imbalance = imbalance / norm_squared
        return imbalance


class TensorRingOrbit(GaugeOrbit):
    """
    Gauge-related representations of a finite tensor ring.

    Gauges act on adjacent virtual axes, including the closing bond. The
    objective and Gram imbalance support both real and complex cores.

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
