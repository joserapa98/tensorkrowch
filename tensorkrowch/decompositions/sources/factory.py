"""
This script contains:

    Public functions:
        * as_tensor_source
"""

from typing import Callable, Union

import torch

from tensorkrowch.formats.formats1d import TensorFormat1D
from tensorkrowch.models.mps import MPS

from tensorkrowch.decompositions.sources.base import BaseSource
from tensorkrowch.decompositions.sources.dense import DenseTensorSource
from tensorkrowch.decompositions.sources.format import FormatSource


SourceLike = Union[BaseSource, TensorFormat1D, MPS, torch.Tensor, Callable]


def as_tensor_source(source: SourceLike) -> BaseSource:
    """
    Wraps a tensor or format, retaining an existing source unchanged.

    A ``torch.Tensor`` is wrapped in :class:`DenseTensorSource`, and a
    ``TensorFormat1D`` or ``MPS`` in :class:`FormatSource`; these sources obtain
    dimensions and evaluation rules from the supplied object. An existing
    :class:`BaseSource` is returned unchanged. Raw callables are not accepted:
    construct a callable source explicitly to define its sites and whether it
    takes discrete indices or continuous coordinates.

    Parameters
    ----------
    source : BaseSource, TensorFormat1D, MPS or torch.Tensor
        Object providing the values to decompose.

    Returns
    -------
    BaseSource
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
    if isinstance(source, BaseSource):
        return source
    raise TypeError('`source` should be a tensor, format, MPS or BaseSource')


__all__ = [
    'as_tensor_source',
]
