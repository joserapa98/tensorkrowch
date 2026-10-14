"""
This script contains:

    Public functions:
        * as_tensor_source
"""

from typing import Callable, Union

import torch

from tensorkrowch.formats.formats1d import TensorFormat1D
from tensorkrowch.models.mps import MPS

from tensorkrowch.decompositions.sources.base import TensorSource
from tensorkrowch.decompositions.sources.dense import DenseTensorSource
from tensorkrowch.decompositions.sources.format import FormatSource


SourceLike = Union[TensorSource, TensorFormat1D, MPS, torch.Tensor, Callable]


def as_tensor_source(source: SourceLike) -> TensorSource:
    """
    Wraps a tensor or format, retaining an existing source unchanged.

    Parameters
    ----------
    source : TensorSource, TensorFormat1D, MPS or torch.Tensor
        Fixed values and their dimensions. Construct callable sources explicitly
        to specify discrete dimensions or continuous evaluation semantics.

    Returns
    -------
    TensorSource
        Source carrying its own shape, dtype, device and evaluation rules.

    Examples
    --------
    >>> source = tk.decompositions.as_tensor_source(torch.arange(4.).reshape(2, 2))
    >>> source.contract_dense()
    tensor([[0., 1.],
            [2., 3.]])
    """
    if isinstance(source, torch.Tensor):
        return DenseTensorSource(source)
    if isinstance(source, (TensorFormat1D, MPS)):
        return FormatSource(source)
    if isinstance(source, TensorSource):
        return source
    raise TypeError('`source` should be a tensor, format, MPS or TensorSource')


__all__ = [
    'as_tensor_source',
]
