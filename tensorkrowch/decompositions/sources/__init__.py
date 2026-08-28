"""Tensor sources shared by ALS and sketching decompositions."""

import builtins
from typing import Callable, Optional, Sequence, Union

import torch

from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.sources.base import (ConfigurationBatch,
                                                      FiberTensorSource,
                                                      TensorSource)
from tensorkrowch.decompositions.sources.callable import CallableTensorSource
from tensorkrowch.decompositions.sources.dense import DenseTensorSource
from tensorkrowch.decompositions.sources.sparse import (EmpiricalDistribution,
                                                        SparseTensorSource)
from tensorkrowch.decompositions.sources.tt import TTTensorSource


SourceLike = Union[TensorSource, TTDecomposition, torch.Tensor, Callable]


def as_tensor_source(
        source: SourceLike,
        input_dim: Optional[Sequence[int]] = None,
        output_shape: Optional[Sequence[int]] = (),
        dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device] = 'cpu',
        batch_size: Optional[int] = None) -> TensorSource:
    """Normalizes tensors, functions and existing sources consistently.

    This adapter is shared by ALS and sketching so the same callable contract
    and metadata rules apply to both families. Existing ``TensorSource``
    instances are returned unchanged.
    """
    if isinstance(source, TTDecomposition):
        return TTTensorSource(source)
    if isinstance(source, (
            CallableTensorSource,
            DenseTensorSource,
            SparseTensorSource,
            TTTensorSource)):
        return source
    if hasattr(source, 'boundary') and hasattr(type(source), 'tensors'):
        return TTTensorSource(source)
    if isinstance(source, torch.Tensor):
        return DenseTensorSource(source, input_dim=input_dim)
    if builtins.callable(source):
        if input_dim is None:
            raise ValueError(
                '`input_dim` is required for a callable tensor source')
        return CallableTensorSource(
            source,
            input_dim=input_dim,
            output_shape=output_shape,
            dtype=dtype,
            device=device,
            batch_size=batch_size)
    if isinstance(source, TensorSource):
        return source
    raise TypeError(
        '`source` should be a TensorSource, TTDecomposition, tensor or callable')


__all__ = [
    'ConfigurationBatch',
    'TensorSource',
    'FiberTensorSource',
    'CallableTensorSource',
    'DenseTensorSource',
    'SparseTensorSource',
    'EmpiricalDistribution',
    'TTTensorSource',
    'as_tensor_source',
]
