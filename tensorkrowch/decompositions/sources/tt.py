"""Tensor-train backed tensor sources."""

from typing import Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.sources.base import (ConfigurationBatch,
                                                      _discrete_indices)


class TTTensorSource:
    """Scalar tensor source represented directly by TT cores.

    The source contracts raw PyTorch cores without constructing a
    TensorKrowch graph. It accepts a
    :class:`~tensorkrowch.decompositions.TTDecomposition` or a core sequence
    with the same open-boundary conventions. Keeping the specialized
    contractions here avoids routing repeated ALS/sketching evaluations
    through :class:`~tensorkrowch.models.MPS`.

    Parameters
    ----------
    tensor : TTDecomposition or sequence of torch.Tensor
        Lightweight TT result or raw open-boundary TT cores.
    """

    def __init__(
            self,
            tensor: Union[TTDecomposition, Sequence[torch.Tensor]]) -> None:
        if isinstance(tensor, TTDecomposition):
            if tensor.n_batches:
                raise ValueError('Batched TT sources are not supported')
            cores = list(tensor.cores)
        else:
            if isinstance(tensor, torch.Tensor):
                raise TypeError(
                    '`tensor` should be TTDecomposition or a core sequence')
            try:
                cores = list(tensor)
            except TypeError as exc:
                raise TypeError(
                    '`tensor` should be TTDecomposition or a core sequence') \
                    from exc
        if not cores:
            raise ValueError('`tensor` should contain at least one core')
        if not all(isinstance(core, torch.Tensor) for core in cores):
            raise TypeError('TT cores should be torch.Tensor objects')

        if len(cores) == 1:
            if cores[0].ndim == 1:
                standard_cores = [cores[0].reshape(1, -1, 1)]
            elif (cores[0].ndim == 3) and \
                    (cores[0].shape[0] == cores[0].shape[-1] == 1):
                standard_cores = [cores[0]]
            else:
                raise ValueError(
                    'A one-site TT core should contain only its input dimension')
        else:
            standard_cores = []
            first = cores[0]
            if first.ndim == 2:
                first = first.unsqueeze(0)
            if (first.ndim != 3) or (first.shape[0] != 1):
                raise ValueError(
                    'The first TT core should have a unit left boundary rank')
            standard_cores.append(first)

            for core in cores[1:-1]:
                if core.ndim != 3:
                    raise ValueError(
                        'Interior TT cores should have left, input and right '
                        'rank dimensions')
                standard_cores.append(core)

            last = cores[-1]
            if last.ndim == 2:
                last = last.unsqueeze(-1)
            if (last.ndim != 3) or (last.shape[-1] != 1):
                raise ValueError(
                    'The last TT core should have a unit right boundary rank')
            standard_cores.append(last)

        device = standard_cores[0].device
        dtype = standard_cores[0].dtype
        for site, core in enumerate(standard_cores):
            if core.device != device:
                raise ValueError('All TT cores should be on the same device')
            if core.dtype != dtype:
                raise ValueError('All TT cores should have the same dtype')
            if site and (core.shape[0] != standard_cores[site - 1].shape[-1]):
                raise ValueError('Adjacent TT ranks should match')

        self.cores = standard_cores
        self._input_dim = tuple(core.shape[1] for core in standard_cores)

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Input dimension at every TT site."""
        return self._input_dim

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Empty shape because this source is scalar."""
        return ()

    @property
    def dtype(self) -> torch.dtype:
        """Dtype shared by the TT cores."""
        return self.cores[0].dtype

    @property
    def device(self) -> torch.device:
        """Device shared by the TT cores."""
        return self.cores[0].device

    def _selected_matrices(self, indices: torch.Tensor):
        """Selects one TT matrix per configuration and site."""
        return [
            core[:, indices[:, site], :].permute(1, 0, 2)
            for site, core in enumerate(self.cores)
        ]

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Evaluates discrete configurations by batched TT contraction."""
        indices = _discrete_indices(
            configurations, self.input_dim, self.device)
        matrices = self._selected_matrices(indices)
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result[:, 0, 0]

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Contracts both TT environments and leaves one input site open."""
        if not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= len(self.input_dim)):
            raise ValueError('`site` should identify an input site')
        indices = _discrete_indices(
            configurations, self.input_dim, self.device)
        matrices = self._selected_matrices(indices)

        left = self.cores[0].new_ones((indices.shape[0], 1))
        for matrix in matrices[:site]:
            left = torch.einsum('ba,bar->br', left, matrix)
        right = self.cores[0].new_ones((indices.shape[0], 1))
        for matrix in reversed(matrices[site + 1:]):
            right = torch.einsum('bar,br->ba', matrix, right)
        result = torch.einsum(
            'ba,apr,br->bp', left, self.cores[site], right)

        if values is None:
            return result
        if not isinstance(values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        if (values.ndim != 1) or (values.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64)):
            raise TypeError('`values` should be a one-dimensional integer tensor')
        values = values.to(device=self.device, dtype=torch.long)
        if torch.any(values < 0) or torch.any(values >= self.input_dim[site]):
            raise ValueError('`values` are out of bounds for the selected site')
        return result.index_select(1, values)


__all__ = ['TTTensorSource']
