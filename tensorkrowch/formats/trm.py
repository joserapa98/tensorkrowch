"""Raw-tensor TRM format."""

from typing import List, Optional, Sequence, Tuple

import torch

from tensorkrowch.formats._chain import _MatrixFormat1D
from tensorkrowch.formats.tr import TR


class TRM(_MatrixFormat1D):
    """Lightweight cyclic raw-tensor network.

    Every core has shape ``(*batch, left, input, right, output)`` and
    adjacent ranks match through the cyclic closure. Structural batches are
    independent of evaluation-data batches. Tensor storage and autograd are
    retained without constructing TensorKrowch nodes or edges.
    """

    _topology = 'trm'
    _cyclic = True

    def to_mpo(self, parameterized: bool = False, **kwargs):
        """Builds a periodic MPO; batched MPO cores are explicitly unsupported."""
        from tensorkrowch.formats.adapters import to_mpo

        return to_mpo(self, parameterized, **kwargs)

    @classmethod
    def from_mpo(cls, model, **kwargs):
        """Collects effective periodic MPO tensors."""
        from tensorkrowch.formats.adapters import from_mpo

        return cls(from_mpo(model, cyclic=True).cores, **kwargs)

    def rotate(self, first=0):
        """Returns a cyclic rotation with input/output pairs moving together."""
        from tensorkrowch.formats.adapters import rotate

        return rotate(self, first)

    def to_ttm(self):
        """Returns the exact open-chain matrix carrying the closing index."""
        from tensorkrowch.formats.adapters import ring_to_train

        return ring_to_train(self)

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

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        result = self._contract_open_chain(matrices)
        return result.diagonal(dim1=-2, dim2=-1).sum(-1)

    def _build_applied_decomposition(
            self,
            cores: List[torch.Tensor],
            n_batches: int) -> TR:
        return TR(
            cores=cores,
            n_batches=n_batches)
