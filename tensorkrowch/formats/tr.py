"""Raw-tensor TR format."""

from typing import List, Optional, Sequence, Tuple

import torch

from tensorkrowch.formats._chain import _VectorFormat1D, TensorFormat1D


class TR(_VectorFormat1D):
    """Lightweight cyclic raw-tensor network.

    Every core has shape ``(*batch, left, input, right)``. Adjacent ranks
    match, including the last-to-first closure. A one-site ring is a trace
    over its two virtual axes. Tensors retain storage and autograd without
    constructing TensorKrowch nodes or edges.
    """

    _topology = 'tr'
    _cyclic = True

    def to_mps(self, parameterized: bool = False, **kwargs):
        """Builds periodic MPS or MPSData from effective cores."""
        from tensorkrowch.formats.adapters import to_mps

        return to_mps(self, parameterized, **kwargs)

    @classmethod
    def from_mps(cls, model, **kwargs):
        """Collects effective periodic MPS/MPSData tensors."""
        from tensorkrowch.formats.adapters import from_mps

        result = from_mps(model, cyclic=True)
        return cls(result.cores, n_batches=result.n_batches, **kwargs)

    def rotate(self, first=0):
        """Returns a cyclic rotation with the selected site first."""
        from tensorkrowch.formats.adapters import rotate

        return rotate(self, first)

    def to_tt(self):
        """Returns an exact TT with the closing bond carried through identities."""
        from tensorkrowch.formats.adapters import ring_to_train

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
