"""
Tensor sources shared by ALS and recursive sketching.

        as_tensor_source(tensor / callable / TT / TensorSource)
            ├─ DenseTensorSource
            ├─ CallableTensorSource
            ├─ SparseTensorSource / EmpiricalDistribution
            └─ TTTensorSource

        ConfigurationBatch
            └─ packed indices or heterogeneous coordinates ─> source.evaluate()

        FiberTensorSource
            └─ optional source.fiber() over one varying input site

        TensorSource
            ├─ ALSProblem ─> TTALS / TRALS
            └─ sketching Phi operators and evaluation sessions

    Sources supply values and runtime metadata. ALSProblem separately defines the
    observed or sampled objective. Sparse sources are zero outside their support;
    unobserved entries in completion remain unknown. All implementations operate
    on raw PyTorch tensors without constructing a TensorKrowch graph.

    This script contains:

        Source normalization function:
            * as_tensor_source
"""

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
        in_dim: Optional[Sequence[int]] = None,
        output_shape: Optional[Sequence[int]] = (),
        dtype: Optional[torch.dtype] = None,
        device: Union[str, torch.device] = 'cpu',
        batch_size: Optional[int] = None) -> TensorSource:
    """Normalizes a tensor, callable or TT into the shared source interface.

    Existing sources are returned unchanged. A dense tensor is wrapped without
    copying it, and a TT result or open-boundary MPS adapter is evaluated
    directly
    from its cores. Runtime overrides describe callables; existing sources
    retain
    their own device and dtype.

    Parameters
    ----------
    source : TensorSource, TTDecomposition, torch.Tensor, MPS or callable
        Value provider to normalize. A callable receives packed configurations
        or
        a tuple of site tensors and must preserve their leading batch
        dimension.
    in_dim : sequence[int], optional
        Discrete input dimensions. Required for a callable. For a dense tensor,
        they identify the leading input axes; remaining axes form its output.
    output_shape : sequence[int], optional
        Callable output shape after the batch axis. The default ``()`` denotes
        a
        scalar; ``None`` infers the shape on first evaluation.
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
        return DenseTensorSource(source, in_dim=in_dim)
    if builtins.callable(source):
        if in_dim is None:
            raise ValueError(
                '`in_dim` is required for a callable tensor source')
        return CallableTensorSource(
            source,
            in_dim=in_dim,
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
