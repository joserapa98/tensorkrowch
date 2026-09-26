"""
This script contains:

    Public classes:
        * DenseTensorSource
"""

from typing import Optional, Sequence, Tuple

import torch

from tensorkrowch.decompositions.sources.base import (ConfigurationBatch,
                                                      _SourceEvaluationTracker,
                                                      _discrete_indices,
                                                      _fiber_configurations)


class DenseTensorSource(_SourceEvaluationTracker):
    """Tensor source backed by an explicitly stored dense tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor containing input and optional output axes.
    in_features : sequence of int, optional
        Tensor axes used as input sites, in configuration order. If omitted,
        every axis is an input site and the source is scalar. Remaining axes
        form the output, in their original order.
    """

    def __init__(self,
                 tensor: torch.Tensor,
                 in_features: Optional[Sequence[int]] = None) -> None:
        self._initialize_evaluation_stats()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if tensor.ndim < 1:
            raise ValueError('`tensor` should contain at least one input site')
        if in_features is None:
            in_features = tuple(range(tensor.ndim))
        elif not isinstance(in_features, (list, tuple)):
            raise TypeError('`in_features` should be a list or tuple of ints')
        else:
            in_features = tuple(in_features)
        if not in_features:
            raise ValueError('`in_features` should contain at least one axis')
        if any(isinstance(axis, bool) or not isinstance(axis, int) or
               axis < 0 or axis >= tensor.ndim for axis in in_features):
            raise ValueError('`in_features` should contain valid tensor axes')
        if len(set(in_features)) != len(in_features):
            raise ValueError('`in_features` should not contain duplicate axes')
        if any(tensor.shape[axis] < 1 for axis in in_features):
            raise ValueError('Input dimensions should be positive')
        out_features = tuple(axis for axis in range(tensor.ndim)
                             if axis not in in_features)
        self.tensor = tensor
        self._in_features = in_features
        self._out_features = out_features
        self._ordered_tensor = tensor.permute(*in_features, *out_features)
        self._in_dim = tuple(tensor.shape[axis] for axis in in_features)
        self._out_shape = tuple(tensor.shape[axis] for axis in out_features)

    @property
    def in_features(self) -> Tuple[int, ...]:
        """Tensor axes used as input sites, in configuration order."""
        return self._in_features

    @property
    def out_features(self) -> Tuple[int, ...]:
        """Remaining tensor axes, in their original order."""
        return self._out_features

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Discrete input dimension at every site."""
        return self._in_dim

    @property
    def out_shape(self) -> Tuple[int, ...]:
        """Shape returned after the configuration batch."""
        return self._out_shape

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
            configurations, self._in_dim, self.device)
        result = self._ordered_tensor[tuple(
            indices[:, site] for site in range(indices.shape[1]))]
        self._record_evaluation(points=indices.shape[0])
        return result

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates a discrete site fiber for every base configuration."""
        if not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= len(self._in_dim)):
            raise ValueError('`site` should identify an input site')
        if values is None:
            values = torch.arange(
                self._in_dim[site], device=configurations.device)
        expanded, n_values = _fiber_configurations(
            configurations, site, values)
        result = self.evaluate(expanded)
        return result.reshape(
            configurations.batch_size, n_values, *self._out_shape)


__all__ = ['DenseTensorSource']
