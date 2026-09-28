"""Raw-tensor TRM format."""

from typing import List, Optional, Sequence, Tuple

import torch

from tensorkrowch.formats._chain import _MatrixFormat1D
from tensorkrowch.formats.tr import TR


class TRM(_MatrixFormat1D):
    r"""Lightweight cyclic raw-tensor network.

    Every core has shape ``(*batch, left, input, right, output)`` and
    adjacent ranks match through the cyclic closure. Structural batches are
    independent of evaluation-data batches. Tensor storage and autograd are
    retained without constructing TensorKrowch nodes or edges.

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

    _topology = 'trm'
    _cyclic = True

    def to_mpo(self, parameterized: bool = False, **kwargs):
        r"""Builds a periodic-boundary MPO from effective cores.

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
        r"""Collects effective periodic-boundary MPO tensors.

        Parameters
        ----------
        model : MPO
            Source model with periodic boundaries. Public tensors include
            boundary contractions where applicable.
        **kwargs : keyword arguments
            Additional options for the concrete format constructor, such as
            Quantics metadata on a subclass.

        Returns
        -------
        TRM
            Format sharing the effective tensor storage. Graph nodes and fit
            metrics are not retained.
        """
        from tensorkrowch.formats.adapters import from_mpo

        return cls(from_mpo(model, cyclic=True).cores, **kwargs)

    def rotate(self, first=0):
        r"""Rotates the stored ring cut to start at a selected site.

        Parameters
        ----------
        first : int
            Site that becomes index zero, in [0, n_sites - 1]. No arbitrary site
            permutation is performed.

        Returns
        -------
        TRM
            Separate format with rotated cores, physical dimensions and factors.
            Dense physical axes undergo the same cyclic rotation.
        """
        from tensorkrowch.formats.adapters import rotate

        return rotate(self, first)

    def to_ttm(self):
        r"""Opens the ring exactly by carrying the closure index through all sites.

        A batched ring matrix cannot be converted because TTM does not support
        structural batches.

        Returns
        -------
        TTM
            Open format with the same dense tensor. Endpoint ranks incorporate
            the closure rank; intermediate cores carry an identity on that
            index. No truncation or densification is performed.
        """
        from tensorkrowch.formats.adapters import ring_to_train

        return ring_to_train(self)

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates core layouts and returns structural dimensions and ranks."""
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
        """Returns standard fused cores without explicit bond factors."""
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
        """Contracts selected local matrices across the stored virtual ranks."""
        result = self._contract_open_chain(matrices)
        return result.diagonal(dim1=-2, dim2=-1).sum(-1)

    def _build_applied_decomposition(
            self,
            cores: List[torch.Tensor],
            n_batches: int) -> TR:
        """Builds a vector format from applied matrix cores and their batches."""
        return TR(
            cores=cores,
            n_batches=n_batches)
