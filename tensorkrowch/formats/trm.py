"""Raw-tensor TensorRingMatrix format."""

from typing import ClassVar, List, Optional, Sequence, Tuple
import torch
from ._chain import _MatrixFormat1D, TensorFormat1D
from .tt import TensorTrain
from .tr import TensorRing


class TensorRingMatrix(_MatrixFormat1D):
    """Lightweight cyclic raw-tensor network.

    Cores use the reviewed decomposition layouts and retain tensor storage and
    autograd. Numerical methods operate without TensorKrowch nodes or edges.
    """

    _topology = 'trm'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        batch_shape = tuple(self.cores[0].shape[:self.n_batches])
        rank = []
        in_dim = []
        out_dim = []

        for site, core in enumerate(self.cores):
            if core.ndim != (self.n_batches + 4):
                raise ValueError(
                    'TRM cores should have left rank, input, right rank and '
                    'output dimensions')
            if tuple(core.shape[:self.n_batches]) != batch_shape:
                raise ValueError(
                    'All TRM cores should have the same batch shape')
            if site and (core.shape[-4] != rank[-1]):
                raise ValueError('Adjacent TRM ranks should match')
            in_dim.append(core.shape[-3])
            rank.append(core.shape[-2])
            out_dim.append(core.shape[-1])

        if self.cores[-1].shape[-2] != self.cores[0].shape[-4]:
            raise ValueError(
                'The last and first cyclic TRM ranks should match')
        return rank, batch_shape, tuple(in_dim), tuple(out_dim)


    def _raw_standard_cores(self) -> List[torch.Tensor]:
        cores = []
        for core in self.cores:
            core = core.movedim(-1, -2)
            cores.append(core.reshape(
                *self._batch_shape,
                core.shape[-4],
                core.shape[-3] * core.shape[-2],
                core.shape[-1]))
        return cores


    def _operator_cores(self) -> List[torch.Tensor]:
        """Returns cores with separate left, input, right and output axes."""
        return list(self.cores)


    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        result = self._contract_open_chain(matrices)
        return result.diagonal(dim1=-2, dim2=-1).sum(-1)


    def _build_applied_decomposition(
            self,
            cores: List[torch.Tensor],
            n_batches: int) -> TensorRing:
        return TensorRing(
            cores=cores,
            n_batches=n_batches)


