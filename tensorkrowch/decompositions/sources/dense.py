"""Dense tensor sources."""

from typing import Optional, Sequence, Tuple

import torch

from tensorkrowch.decompositions.sources.base import (
    ConfigurationBatch,
    _discrete_indices,
    _fiber_configurations,
    _normalize_input_dim,
)


class DenseTensorSource:
    """Tensor source backed by an explicitly stored dense tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor whose leading dimensions are input dimensions.
    input_dim : sequence of int, optional
        Input dimension at every site. If omitted, every tensor dimension is
        interpreted as an input site and the source is scalar. Supplying a
        prefix leaves the remaining tensor dimensions as output dimensions.
    """

    def __init__(self,
                 tensor: torch.Tensor,
                 input_dim: Optional[Sequence[int]] = None) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if tensor.ndim < 1:
            raise ValueError('`tensor` should contain at least one input site')
        if input_dim is None:
            normalized_input_dim = tuple(tensor.shape)
        else:
            normalized_input_dim = _normalize_input_dim(input_dim)
            if tuple(tensor.shape[:len(normalized_input_dim)]) != \
                    normalized_input_dim:
                raise ValueError(
                    '`input_dim` should match the leading tensor dimensions')
        self.tensor = tensor
        self._input_dim = normalized_input_dim
        self._output_shape = tuple(tensor.shape[len(normalized_input_dim):])

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Discrete input dimension at every site."""
        return self._input_dim

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Shape returned after the configuration batch."""
        return self._output_shape

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the stored tensor."""
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        """Device of the stored tensor."""
        return self.tensor.device

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Gathers dense values at discrete global configurations."""
        indices = _discrete_indices(
            configurations, self.input_dim, self.device)
        return self.tensor[tuple(indices[:, site]
                                 for site in range(indices.shape[1]))]

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates a discrete site fiber for every base configuration."""
        if not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= len(self.input_dim)):
            raise ValueError('`site` should identify an input site')
        if values is None:
            values = torch.arange(
                self.input_dim[site], device=configurations.device)
        expanded, n_values = _fiber_configurations(
            configurations, site, values)
        result = self.evaluate(expanded)
        return result.reshape(
            configurations.batch_size, n_values, *self.output_shape)


__all__ = ['DenseTensorSource']
