"""
This script contains:

    Public functions:
        * as_tensor_source
"""

import builtins
from typing import Callable, Optional, Sequence, Union

import torch

from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.sources.base import TensorSource
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
        batch_size: Optional[int] = None) -> TensorSource:
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
        they identify the leading input axes; remaining axes form its output.
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
        if in_dim is None:
            return DenseTensorSource(source)
        if tuple(source.shape[:len(in_dim)]) != tuple(in_dim):
            raise ValueError(
                '`in_dim` should match the leading tensor dimensions')
        return DenseTensorSource(source, in_features=tuple(range(len(in_dim))))
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
