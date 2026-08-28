"""Optional source capabilities and input adapters for sketching methods."""

from typing import (Iterator, Optional, Protocol, Sequence, Tuple, Union,
                    runtime_checkable)

import torch

from tensorkrowch.decompositions.results import (TTDecomposition,
                                                 TTMDecomposition)
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 EmpiricalDistribution,
                                                 TensorSource,
                                                 as_tensor_source)


TTStructuredSketch = Union[
    TTDecomposition,
    TTMDecomposition,
    Sequence[torch.Tensor],
]


@runtime_checkable
class SketchContractableSource(TensorSource, Protocol):
    """Optional source capable of contracting a TT-structured sketch."""

    def contract_sketch(
            self,
            sketch: TTStructuredSketch,
            conjugate_sketch: bool = True
            ) -> Union[torch.Tensor, TTDecomposition]:
        """Contracts all source input axes with ``sketch`` without a graph."""


@runtime_checkable
class SupportTensorSource(TensorSource, Protocol):
    """Optional source exposing its complete finite non-zero support."""

    @property
    def support(self) -> ConfigurationBatch:
        """Returns unique configurations with declared non-zero values."""

    @property
    def support_values(self) -> torch.Tensor:
        """Returns values aligned with :attr:`support`."""


def _resolve_rs_source(
        *,
        source=None,
        dataset: Optional[torch.Tensor] = None,
        input_dim: Optional[Sequence[int]] = None,
        weights: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = None) -> TensorSource:
    """Normalizes the mutually exclusive source/dataset RS input contract."""
    if (source is None) == (dataset is None):
        raise ValueError('Exactly one of `source` and `dataset` is required')
    if dataset is not None:
        return EmpiricalDistribution(
            dataset=dataset,
            input_dim=input_dim,
            weights=weights,
            dtype=dtype)
    if weights is not None:
        raise ValueError('`weights` can only be passed together with `dataset`')
    return as_tensor_source(
        source,
        input_dim=input_dim,
        output_shape=(),
        dtype=dtype)


def _iter_support(
        source: SupportTensorSource,
        batch_size: Optional[int] = None
        ) -> Iterator[Tuple[ConfigurationBatch, torch.Tensor]]:
    """Yields aligned non-zero support batches without a Cartesian grid."""
    if not isinstance(source, SupportTensorSource):
        raise TypeError('`source` should expose finite support')
    support = source.support
    values = source.support_values
    if support.batch_size != values.shape[0]:
        raise ValueError('Source support and values should have matching sizes')
    if batch_size is None:
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
    'SketchContractableSource',
    'SupportTensorSource',
    'TTStructuredSketch',
]
