"""
This script contains:

    Public functions:
        * as_tensor_source
"""

import builtins
from typing import Callable, Optional, Sequence, Union

import torch

from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.sources.base import (TensorSource,
                                                      _normalize_in_dim)
from tensorkrowch.decompositions.sources.callable import CallableTensorSource
from tensorkrowch.decompositions.sources.dense import DenseTensorSource
from tensorkrowch.decompositions.sources.sparse import SparseTensorSource
from tensorkrowch.decompositions.sources.tt import TTTensorSource


SourceLike = Union[TensorSource, TTDecomposition, torch.Tensor, Callable]


def as_tensor_source(
        source: SourceLike,
        in_dim: Optional[Sequence[int]] = None,
        out_shape: Optional[Sequence[int]] = (),
        dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device] = 'cpu',
        batch_size: Optional[int] = None,
        *,
        in_features: Optional[Sequence[int]] = None) -> TensorSource:
    """Normalizes a tensor, callable or TT into the shared source interface.

    Existing sources are returned unchanged. A dense tensor is wrapped without
    copying it, and a TT result or open-boundary MPS adapter is evaluated
    directly from its cores. Runtime overrides describe callables; existing
    sources retain their own device and dtype.

    Parameters
    ----------
    source : TensorSource, TTDecomposition, torch.Tensor, MPS or callable
        Value provider to normalize. A callable receives packed configurations
        or a tuple of site tensors and must preserve their leading batch
        dimension.
    in_dim : sequence[int], optional
        Discrete input dimensions. Required for a callable. For a dense tensor,
        they may validate the dimensions selected by ``in_features``. Without
        ``in_features``, they identify the leading input axes for compatibility.
    in_features : sequence[int], optional
        Axes used as input sites when ``source`` is a dense tensor. If omitted
        with no ``in_dim``, every tensor axis is an input site.
    out_shape : sequence[int], optional
        Callable output shape after the batch axis. The default ``()`` denotes
        a scalar; ``None`` infers the shape on first evaluation.
    dtype : torch.dtype, optional
        Callable output dtype, inferred on first evaluation when omitted.
    device : str or torch.device
        Callable evaluation device. The default is ``"cpu"``.
    batch_size : int, optional
        Maximum configurations per callable invocation.

    Returns
    -------
    TensorSource
        Source with consistent value, dimension and runtime contracts.
    """
    if in_features is not None and not isinstance(source, torch.Tensor):
        raise TypeError('`in_features` is only used for dense tensor sources')
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
        dimensions = None if in_dim is None else _normalize_in_dim(in_dim)
        if in_features is None and dimensions is not None:
            in_features = tuple(range(len(dimensions)))
        dense_source = DenseTensorSource(source, in_features=in_features)
        if dimensions is not None and dense_source.in_dim != dimensions:
            raise ValueError('`in_dim` should match the selected tensor axes')
        return dense_source
    if builtins.callable(source):
        if in_dim is None:
            raise ValueError(
                '`in_dim` is required for a callable tensor source')
        return CallableTensorSource(
            source,
            in_dim=in_dim,
            out_shape=out_shape,
            dtype=dtype,
            device=device,
            batch_size=batch_size)
    if isinstance(source, TensorSource):
        return source
    raise TypeError(
        '`source` should be a TensorSource, TTDecomposition, tensor or callable')


__all__ = ['as_tensor_source']
