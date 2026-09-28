"""Raw-tensor TensorRing format."""

from typing import ClassVar, List, Optional, Sequence, Tuple
import torch
from ._chain import _VectorFormat1D, TensorFormat1D


class TensorRing(_VectorFormat1D):
    """Lightweight cyclic raw-tensor network.

    Cores use the reviewed decomposition layouts and retain tensor storage and
    autograd. Numerical methods operate without TensorKrowch nodes or edges.
    """

    _topology = 'tr'

    def to_mps(self, parameterized=False, **kwargs):
        """Builds periodic MPS or MPSData from effective cores."""
        from .adapters import to_mps
        return to_mps(self, parameterized, **kwargs)

    @classmethod
    def from_mps(cls, model):
        """Collects effective periodic MPS/MPSData tensors."""
        from .adapters import from_mps
        result = from_mps(model, cyclic=True)
        return cls(result.cores, n_batches=result.n_batches)

    def rotate(self, first=0):
        """Returns a cyclic rotation with the selected site first."""
        from .adapters import rotate
        return rotate(self, first)

    def to_tt(self):
        """Returns an exact TT with the closing bond carried through identities."""
        from .adapters import ring_to_train
        return ring_to_train(self)

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        batch_shape = tuple(self.cores[0].shape[:self.n_batches])
        rank = []
        in_dim = []

        for site, core in enumerate(self.cores):
            if core.ndim != (self.n_batches + 3):
                raise ValueError(
                    'TR cores should have left rank, input and right rank '
                    'dimensions')
            if tuple(core.shape[:self.n_batches]) != batch_shape:
                raise ValueError('All TR cores should have the same batch shape')
            if site and (core.shape[-3] != rank[-1]):
                raise ValueError('Adjacent TR ranks should match')
            in_dim.append(core.shape[-2])
            rank.append(core.shape[-1])

        if self.cores[-1].shape[-1] != self.cores[0].shape[-3]:
            raise ValueError('The last and first cyclic TR ranks should match')
        return rank, batch_shape, tuple(in_dim), None


    def _raw_standard_cores(self) -> List[torch.Tensor]:
        return list(self.cores)


    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        result = self._contract_open_chain(matrices)
        return result.diagonal(dim1=-2, dim2=-1).sum(-1)
