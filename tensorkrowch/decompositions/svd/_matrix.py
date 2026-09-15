"""
This script contains:

    Internal matrix tensorization:
        * _MatrixTensorization
"""

from dataclasses import dataclass
from math import prod
from typing import Optional, Sequence, Tuple, Union

import torch


_Dimension = Optional[Union[int, Sequence[int]]]


def _normalize_dim(dim: _Dimension, name: str) -> Tuple[int, ...]:
    """Normalizes one or several positive site dimensions."""
    if isinstance(dim, bool):
        raise TypeError(f'`{name}` should be int or a sequence of ints')
    if isinstance(dim, int):
        values = (dim,)
    elif isinstance(dim, Sequence) and not isinstance(dim, (str, bytes)):
        values = tuple(dim)
    else:
        raise TypeError(f'`{name}` should be int or a sequence of ints')

    if not values:
        raise ValueError(f'`{name}` should contain at least one dimension')
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in values):
        raise TypeError(f'`{name}` should contain only ints')
    if any(value < 1 for value in values):
        raise ValueError(f'`{name}` should contain only positive dimensions')
    return values


@dataclass(frozen=True)
class _MatrixTensorization:
    """Normalizes dense matrix layouts into fused site dimensions."""

    in_dim: Tuple[int, ...]  # Input dimension of every matrix site
    out_dim: Tuple[int, ...]  # Output dimension of every matrix site
    interleaved: torch.Tensor  # Tensor ordered as in_1, out_1, ..., in_n, out_n
    fused: torch.Tensor  # Tensor with each local input/output pair fused
    matrix_input: bool  # Whether the original tensor had two matrix axes

    @classmethod
    def from_tensor(cls,
                    tensor: torch.Tensor,
                    in_dim: _Dimension,
                    out_dim: _Dimension,
                    layout: str,
                    family: str) -> '_MatrixTensorization':
        """Validates and tensorizes a dense matrix or matrix-like tensor."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if not isinstance(layout, str):
            raise TypeError('`layout` should be str type')
        if layout not in ('interleaved', 'grouped'):
            raise ValueError(
                '`layout` should be either "interleaved" or "grouped"')
        if (in_dim is None) != (out_dim is None):
            raise ValueError(
                '`in_dim` and `out_dim` should be provided together')

        matrix_input = False
        if in_dim is None:
            if (tensor.ndim < 2) or (tensor.ndim % 2):
                raise ValueError(
                    f'A tensorized {family} input should have a positive even '
                    'number of dimensions')
            n_sites = tensor.ndim // 2
            if layout == 'interleaved':
                normalized_in_dim = tuple(tensor.shape[::2])
                normalized_out_dim = tuple(tensor.shape[1::2])
            else:
                normalized_in_dim = tuple(tensor.shape[:n_sites])
                normalized_out_dim = tuple(tensor.shape[n_sites:])
            if any(value < 1
                   for value in normalized_in_dim + normalized_out_dim):
                raise ValueError(
                    f'{family} input and output dimensions should be positive')
            tensorized = tensor
            tensorized_layout = layout
        else:
            normalized_in_dim = _normalize_dim(in_dim, 'in_dim')
            normalized_out_dim = _normalize_dim(out_dim, 'out_dim')
            if len(normalized_in_dim) != len(normalized_out_dim):
                raise ValueError(
                    '`in_dim` and `out_dim` should have the same length')
            n_sites = len(normalized_in_dim)

            if tensor.ndim == 2:
                expected_shape = (
                    prod(normalized_in_dim),
                    prod(normalized_out_dim),
                )
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        'The matrix shape should equal '
                        '(prod(in_dim), prod(out_dim))')
                tensorized = tensor.reshape(
                    *normalized_in_dim, *normalized_out_dim)
                tensorized_layout = 'grouped'
                matrix_input = True
            else:
                if tensor.ndim != (2 * n_sites):
                    raise ValueError(
                        f'A tensorized {family} input should have two '
                        'dimensions per site')
                expected_shape = (
                    tuple(value
                          for pair in zip(normalized_in_dim,
                                          normalized_out_dim)
                          for value in pair)
                    if layout == 'interleaved'
                    else normalized_in_dim + normalized_out_dim
                )
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        'The tensor shape is incompatible with `in_dim`, '
                        '`out_dim` and `layout`')
                tensorized = tensor
                tensorized_layout = layout

        if (tensorized_layout == 'interleaved') or (n_sites == 1):
            interleaved = tensorized
        else:
            axes = tuple(
                axis
                for site in range(n_sites)
                for axis in (site, n_sites + site))
            interleaved = tensorized.permute(axes)

        fused_dim = tuple(
            in_value * out_value
            for in_value, out_value
            in zip(normalized_in_dim, normalized_out_dim))
        return cls(
            in_dim=normalized_in_dim,
            out_dim=normalized_out_dim,
            interleaved=interleaved,
            fused=interleaved.reshape(*fused_dim),
            matrix_input=matrix_input)


__all__ = ['_Dimension', '_MatrixTensorization']
