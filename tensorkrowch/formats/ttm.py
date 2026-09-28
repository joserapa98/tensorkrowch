"""Raw-tensor TensorTrainMatrix format."""

from typing import ClassVar, List, Optional, Sequence, Tuple
import torch
from ._chain import _MatrixFormat1D, TensorFormat1D
from .tt import TensorTrain
from .tr import TensorRing


class TensorTrainMatrix(_MatrixFormat1D):
    """Lightweight open raw-tensor network.

    Cores use the reviewed decomposition layouts and retain tensor storage and
    autograd. Numerical methods operate without TensorKrowch nodes or edges.
    """

    _topology = 'ttm'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        if self.n_batches:
            raise ValueError('TTM decomposition batches are not supported')

        n_sites = len(self.cores)
        in_dim = []
        out_dim = []
        if n_sites == 1:
            if self.cores[0].ndim != 2:
                raise ValueError(
                    'A one-site TTM core should have input and output dimensions')
            in_dim.append(self.cores[0].shape[0])
            out_dim.append(self.cores[0].shape[1])
            return [], (), tuple(in_dim), tuple(out_dim)

        rank = []
        for site, core in enumerate(self.cores):
            if site == 0:
                if core.ndim != 3:
                    raise ValueError(
                        'The first TTM core should have input, right rank and '
                        'output dimensions')
                in_dim.append(core.shape[0])
                out_dim.append(core.shape[2])
                rank.append(core.shape[1])
            elif site == (n_sites - 1):
                if core.ndim != 3:
                    raise ValueError(
                        'The last TTM core should have left rank, input and '
                        'output dimensions')
                if core.shape[0] != rank[-1]:
                    raise ValueError('Adjacent TTM ranks should match')
                in_dim.append(core.shape[1])
                out_dim.append(core.shape[2])
            else:
                if core.ndim != 4:
                    raise ValueError(
                        'Interior TTM cores should have left, input, right and '
                        'output dimensions')
                if core.shape[0] != rank[-1]:
                    raise ValueError('Adjacent TTM ranks should match')
                in_dim.append(core.shape[1])
                out_dim.append(core.shape[3])
                rank.append(core.shape[2])

        return rank, (), tuple(in_dim), tuple(out_dim)


    def _raw_standard_cores(self) -> List[torch.Tensor]:
        if len(self.cores) == 1:
            core = self.cores[0]
            return [core.reshape(1, core.numel(), 1)]

        cores = []
        first = self.cores[0].permute(0, 2, 1)
        cores.append(first.reshape(1, first.shape[0] * first.shape[1],
                                   first.shape[2]))
        for core in self.cores[1:-1]:
            core = core.permute(0, 1, 3, 2)
            cores.append(core.reshape(core.shape[0],
                                      core.shape[1] * core.shape[2],
                                      core.shape[3]))
        last = self.cores[-1]
        cores.append(last.reshape(last.shape[0], -1, 1))
        return cores


    def _raw_operator_cores(self) -> List[torch.Tensor]:
        """Returns cores with separate left, input, right and output axes."""
        if len(self.cores) == 1:
            return [self.cores[0].unsqueeze(0).unsqueeze(2)]

        cores = [self.cores[0].unsqueeze(0)]
        cores.extend(self.cores[1:-1])
        cores.append(self.cores[-1].unsqueeze(2))
        return cores


    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        result = self._contract_open_chain(matrices)
        return result.squeeze(-1).squeeze(-1)


    def _build_applied_decomposition(
            self,
            cores: List[torch.Tensor],
            n_batches: int) -> TensorTrain:
        batch_shape = cores[0].shape[:n_batches]
        if len(cores) == 1:
            cores[0] = cores[0].squeeze(-1).squeeze(-2)
        else:
            cores[0] = cores[0].squeeze(len(batch_shape))
            cores[-1] = cores[-1].squeeze(-1)

        return TensorTrain(
            cores=cores,
            n_batches=n_batches)


