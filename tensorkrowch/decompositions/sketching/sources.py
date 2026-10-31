"""
This script contains:

    Classes:
        * SupportTensorSource

    Functions:
        * _iter_support
"""

from typing import (Iterator,
                    Optional,
                    Protocol,
                    Tuple,
                    runtime_checkable)

import torch

from tensorkrowch.decompositions.sources import ConfigurationBatch


@runtime_checkable
class SupportTensorSource(Protocol):  # MARK: SupportTensorSource
    """
    Source exposing its complete finite nonzero support.

    ``support`` contains the unique configurations with nonzero values;
    ``support_values`` contains their aligned values. Entries outside
    this support are zero, rather than unobserved. Sketch builders use these
    tables to contract sparse tensors without enumerating a Cartesian grid.
    """

    @property
    def support(self) -> ConfigurationBatch:
        """Returns unique configurations with declared non-zero values."""

    @property
    def support_values(self) -> torch.Tensor:
        """Returns values aligned with :attr:`support`."""


def _iter_support(source: SupportTensorSource,
                  batch_size: Optional[
                      int] = None) -> Iterator[
        Tuple[ConfigurationBatch, torch.Tensor]]:
    """Yields aligned non-zero support batches without a Cartesian grid."""
    if not isinstance(source, SupportTensorSource):
        raise TypeError('`source` should expose finite support')
    support = source.support
    values = source.support_values
    if support.batch_size != values.shape[0]:
        raise ValueError(
            '`support` and `support_values` should have matching batch sizes')
    if batch_size is None:
        if not support.batch_size:
            return
        batch_size = support.batch_size
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError('`batch_size` should be int type or None')
    if batch_size < 1:
        raise ValueError('`batch_size` should be positive')
    for start in range(0, support.batch_size, batch_size):
        indices = torch.arange(
            start,
            min(start + batch_size, support.batch_size),
            device=support.device)
        yield support.index_select(indices), values.index_select(0, indices)


__all__ = [
    'SupportTensorSource',
]
