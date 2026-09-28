"""Raw-tensor TT format."""

from typing import List, Optional, Sequence, Tuple

import torch

from tensorkrowch.formats._chain import _VectorFormat1D, TensorFormat1D


class TT(_VectorFormat1D):
    r"""Lightweight open raw-tensor network.

    With leading structural batch axes B, endpoint cores have shapes
    ``(*B, input, right)`` and ``(*B, left, input)``; interiors use
    ``(*B, left, input, right)``. A single core is ``(*B, input)``.
    The constructor shares tensors and copies their container. No nodes or
    edges are constructed, and input tensors retain autograd.

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

    _topology = 'tt'

    def to_mps(self, parameterized: bool = False, **kwargs):
        r"""Builds a open-boundary MPS or MPSData from effective cores.

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

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> model = format.to_mps()
        >>> restored = tk.formats.TT.from_mps(model)
        >>> torch.allclose(restored.contract_dense(), format.contract_dense())
        True
        """
        from tensorkrowch.formats.adapters import to_mps

        return to_mps(self, parameterized, **kwargs)

    @classmethod
    def from_mps(cls, model, **kwargs):
        r"""Collects effective open-boundary MPS or MPSData tensors.

        Parameters
        ----------
        model : MPS or MPSData
            Source model with open boundaries. Public tensors include boundary
            contractions where applicable.
        **kwargs : keyword arguments
            Additional options for the concrete format constructor, such as
            Quantics metadata on a subclass.

        Returns
        -------
        TT
            Format sharing the effective tensor storage. Graph nodes and fit
            metrics are not retained.
        """
        from tensorkrowch.formats.adapters import from_mps

        result = from_mps(model, cyclic=False)
        return cls(result.cores, n_batches=result.n_batches, **kwargs)

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates core layouts and returns structural dimensions and ranks."""
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
        """Returns standard fused cores without explicit bond factors."""
        if len(self.cores) == 1:
            return [self.cores[0].unsqueeze(self.n_batches).unsqueeze(-1)]

        cores = [self.cores[0].unsqueeze(self.n_batches)]
        cores.extend(self.cores[1:-1])
        cores.append(self.cores[-1].unsqueeze(-1))
        return cores

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts selected local matrices across the stored virtual ranks."""
        result = self._contract_open_chain(matrices)
        return result.squeeze(-1).squeeze(-1)
