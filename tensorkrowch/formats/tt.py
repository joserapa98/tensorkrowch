"""Raw-tensor TT format."""

from typing import ClassVar, List, Optional, Sequence, Tuple

import torch

from tensorkrowch.formats._chain import _VectorFormat1D, TensorFormat1D


class TT(_VectorFormat1D):
    """Lightweight open raw-tensor network.

    With leading structural batch axes B, endpoint cores have shapes
    ``(*B, input, right)`` and ``(*B, left, input)``; interiors use
    ``(*B, left, input, right)``. A single core is ``(*B, input)``.
    The constructor shares tensors and copies their container. No nodes or
    edges are constructed, and input tensors retain autograd.
    """

    _topology = 'tt'

    def to_mps(self, parameterized=False, **kwargs):
        """Builds MPS or MPSData from these cores without detaching."""
        from tensorkrowch.formats.adapters import to_mps

        return to_mps(self, parameterized, **kwargs)

    @classmethod
    def from_mps(cls, model):
        """Collects effective open-boundary MPS/MPSData tensors."""
        from tensorkrowch.formats.adapters import from_mps

        result = from_mps(model, cyclic=False)
        return cls(result.cores, n_batches=result.n_batches)

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        n_sites = len(self.cores)
        batch_shape = tuple(self.cores[0].shape[:self.n_batches])
        in_dim = []

        if n_sites == 1:
            if self.cores[0].ndim != (self.n_batches + 1):
                raise ValueError(
                    'A one-site TT core should have one input dimension')
            in_dim.append(self.cores[0].shape[-1])
            return [], batch_shape, tuple(in_dim), None

        rank = []
        for site, core in enumerate(self.cores):
            if tuple(core.shape[:self.n_batches]) != batch_shape:
                raise ValueError('All TT cores should have the same batch shape')

            if site == 0:
                if core.ndim != (self.n_batches + 2):
                    raise ValueError(
                        'The first TT core should have input and right rank '
                        'dimensions')
                in_dim.append(core.shape[-2])
                rank.append(core.shape[-1])
            elif site == (n_sites - 1):
                if core.ndim != (self.n_batches + 2):
                    raise ValueError(
                        'The last TT core should have left rank and input '
                        'dimensions')
                if core.shape[-2] != rank[-1]:
                    raise ValueError('Adjacent TT ranks should match')
                in_dim.append(core.shape[-1])
            else:
                if core.ndim != (self.n_batches + 3):
                    raise ValueError(
                        'Interior TT cores should have left, input and '
                        'right dimensions')
                if core.shape[-3] != rank[-1]:
                    raise ValueError('Adjacent TT ranks should match')
                in_dim.append(core.shape[-2])
                rank.append(core.shape[-1])

        return rank, batch_shape, tuple(in_dim), None


    def _raw_standard_cores(self) -> List[torch.Tensor]:
        if len(self.cores) == 1:
            return [self.cores[0].unsqueeze(self.n_batches).unsqueeze(-1)]

        cores = [self.cores[0].unsqueeze(self.n_batches)]
        cores.extend(self.cores[1:-1])
        cores.append(self.cores[-1].unsqueeze(-1))
        return cores


    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        result = self._contract_open_chain(matrices)
        return result.squeeze(-1).squeeze(-1)
