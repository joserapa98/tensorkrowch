"""Raw-tensor TR format."""

from typing import List, Optional, Sequence, Tuple

import torch

from tensorkrowch.formats._chain import _VectorFormat1D, TensorFormat1D


class TR(_VectorFormat1D):
    r"""Lightweight cyclic raw-tensor network.

    Every core has shape ``(*batch, left, input, right)``. Adjacent ranks
    match, including the last-to-first closure. A one-site ring is a trace
    over its two virtual axes. Tensors retain storage and autograd without
    constructing TensorKrowch nodes or edges.

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

    _topology = 'tr'
    _cyclic = True

    def to_mps(self, parameterized: bool = False, **kwargs):
        r"""Builds a periodic-boundary MPS or MPSData from effective cores.

        Batched vectors produce MPSData and reject parameterized=True. Batched
        matrices cannot be converted to MPO.

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
        MPS or MPSData
            New graph model. Stored factors are materialized in temporary
            tensors; the source format is unchanged.
        """
        from tensorkrowch.formats.adapters import to_mps

        return to_mps(self, parameterized, **kwargs)

    @classmethod
    def from_mps(cls, model, **kwargs):
        r"""Collects effective periodic-boundary MPS or MPSData tensors.

        Parameters
        ----------
        model : MPS or MPSData
            Source model with periodic boundaries. Public tensors include
            boundary contractions where applicable.
        **kwargs : keyword arguments
            Additional options for the concrete format constructor, such as
            Quantics metadata on a subclass.

        Returns
        -------
        TR
            Format sharing the effective tensor storage. Graph nodes and fit
            metrics are not retained.
        """
        from tensorkrowch.formats.adapters import from_mps

        result = from_mps(model, cyclic=True)
        return cls(result.cores, n_batches=result.n_batches, **kwargs)

    def rotate(self, first=0):
        r"""Rotates the stored ring cut to start at a selected site.

        Parameters
        ----------
        first : int
            Site that becomes index zero, in [0, n_sites - 1]. No arbitrary site
            permutation is performed.

        Returns
        -------
        TR
            Separate format with rotated cores, physical dimensions and factors.
            Dense physical axes undergo the same cyclic rotation.

        Examples
        --------
        >>> format = tk.formats.TR([torch.ones(1, 2, 2), torch.ones(2, 3, 1)])
        >>> rotated = format.rotate(first=1)
        >>> rotated.in_dim
        (3, 2)
        >>> torch.equal(rotated.contract_dense(), format.contract_dense().T)
        True
        """
        from tensorkrowch.formats.adapters import rotate

        return rotate(self, first)

    def to_tt(self):
        r"""Opens the ring exactly by carrying the closure index through all sites.

        Returns
        -------
        TT
            Open format with the same dense tensor. Endpoint ranks incorporate
            the closure rank; intermediate cores carry an identity on that
            index. No truncation or densification is performed.

        Examples
        --------
        >>> ring = tk.formats.TR([torch.eye(2).reshape(2, 1, 2)] * 2)
        >>> train = ring.to_tt()
        >>> torch.allclose(train.contract_dense(), ring.contract_dense())
        True
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
        """Returns standard fused cores without explicit bond factors."""
        return list(self.cores)

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts selected local matrices across the stored virtual ranks."""
        result = self._contract_open_chain(matrices)
        return result.diagonal(dim1=-2, dim2=-1).sum(-1)
