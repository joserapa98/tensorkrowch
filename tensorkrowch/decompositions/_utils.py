"""
This script contains:

    Internal validation functions:
        * _normalize_positive_int
        * _normalize_dims
        * _normalize_rank

    Internal index conversions:
        * _ravel_indices
        * _unravel_indices
"""

from math import prod
from typing import Sequence, Tuple, Union

import torch


def _normalize_positive_int(value: int, name: str) -> int:
    """Validates one positive integer without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'`{name}` should be int type')
    if value < 1:
        raise ValueError(f'`{name}` should be positive')
    return value


def _normalize_dims(dimensions: Sequence[int],
                    name: str,
                    *,
                    allow_empty: bool = False) -> Tuple[int, ...]:
    """Normalizes positive dimensions, optionally allowing a scalar shape."""
    if isinstance(dimensions, (str, bytes)) or \
            not isinstance(dimensions, Sequence):
        raise TypeError(f'`{name}` should be a sequence of integers')
    dimensions = tuple(dimensions)
    if not dimensions and not allow_empty:
        raise ValueError(f'`{name}` should contain at least one dimension')
    for axis, dimension in enumerate(dimensions):
        _normalize_positive_int(dimension, f'{name}[{axis}]')
    return dimensions


def _normalize_rank(rank: Union[int, Sequence[int]],
                    n_links: int) -> Tuple[int, ...]:
    """Normalizes one positive rank per link, including an empty TT link set."""
    if isinstance(rank, int):
        return (_normalize_positive_int(rank, 'rank'),) * n_links
    ranks = _normalize_dims(rank, 'rank', allow_empty=n_links == 0)
    if len(ranks) != n_links:
        raise ValueError(
            f'`rank` should contain {n_links} values, one per link')
    return ranks


def _ravel_indices(indices: torch.Tensor,
                   dimensions: Sequence[int]) -> torch.Tensor:
    """Flattens valid row-major multi-indices, including scalar shapes."""
    strides = indices.new_tensor([
        prod(dimensions[site + 1:]) for site in range(len(dimensions))])
    return (indices * strides).sum(dim=1)


def _unravel_indices(flat_ids: torch.Tensor,
                     dimensions: Sequence[int]) -> torch.Tensor:
    """Unflattens valid row-major ids, including scalar shapes and empty batches."""
    if not dimensions:
        return flat_ids.new_empty((flat_ids.shape[0], 0))
    remainder = flat_ids
    indices = []
    for dimension in reversed(dimensions):
        indices.append(torch.remainder(remainder, dimension))
        remainder = torch.div(remainder, dimension, rounding_mode='floor')
    return torch.stack(tuple(reversed(indices)), dim=1)
