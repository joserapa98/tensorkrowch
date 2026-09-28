"""Raw-tensor TTM format."""

from typing import List, Optional, Sequence, Tuple

import torch

from tensorkrowch.formats._chain import _MatrixFormat1D
from tensorkrowch.formats.tt import TT


class TTM(_MatrixFormat1D):
    r"""Lightweight open raw-tensor network.

    Endpoint shapes are ``(input, right, output)`` and
    ``(left, input, output)``; interiors are
    ``(left, input, right, output)``. A single core is ``(input, output)``.
    Structural batches are currently unsupported. Tensor storage and autograd
    are retained without constructing TensorKrowch nodes or edges.

    TTM currently requires n_batches=0; batched operator formats use TRM.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores in the endpoint layout of the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    """

    _topology = 'ttm'

    def to_mpo(self, parameterized: bool = False, **kwargs):
        r"""Builds a open-boundary MPO from effective cores.

        MPO conversion requires unbatched cores. The model may share effective
        tensor storage with the format.

        Parameters
        ----------
        parameterized : bool
            Whether the constructed model uses trainable parameter nodes. Inputs
            are not detached implicitly.
        **kwargs : keyword arguments
            Additional model constructor options. Tensor cores and boundary are
            supplied by the adapter.

        Returns
        -------
        MPO
            New graph model. Stored factors are materialized in temporary
            tensors; the source format is unchanged.
        """
        from tensorkrowch.formats.adapters import to_mpo

        return to_mpo(self, parameterized, **kwargs)

    @classmethod
    def from_mpo(cls, model, **kwargs):
        r"""Collects effective open-boundary MPO tensors.

        Parameters
        ----------
        model : MPO
            Source model with open boundaries. Public tensors include boundary
            contractions where applicable.
        **kwargs : keyword arguments
            Additional options for the concrete format constructor, such as
            Quantics metadata on a subclass.

        Returns
        -------
        TTM
            Format sharing the effective tensor storage. Graph nodes and fit
            metrics are not retained.
        """
        from tensorkrowch.formats.adapters import from_mpo

        return cls(from_mpo(model, cyclic=False).cores, **kwargs)

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates core layouts and returns structural dimensions and ranks."""
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
        """Returns standard fused cores without explicit bond factors."""
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

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts selected local matrices across the stored virtual ranks."""
        result = self._contract_open_chain(matrices)
        return result.squeeze(-1).squeeze(-1)

    def _build_applied_decomposition(
            self,
            cores: List[torch.Tensor],
            n_batches: int) -> TT:
        """Builds a vector format from applied matrix cores and their batches."""
        batch_shape = cores[0].shape[:n_batches]
        if len(cores) == 1:
            cores[0] = cores[0].squeeze(-1).squeeze(-2)
        else:
            cores[0] = cores[0].squeeze(len(batch_shape))
            cores[-1] = cores[-1].squeeze(-1)

        return TT(
            cores=cores,
            n_batches=n_batches)
