"""
This script contains:

    Internal SVD progress class:
        * _SVDProgress

    Internal SVD source preparation:
        * _prepare_svd_source
"""

from dataclasses import dataclass
from typing import Optional

import torch

from tensorkrowch.utils import safe_divide

from tensorkrowch.decompositions.metrics import TruncationRecord
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   DecompositionObserver)
from tensorkrowch.decompositions.sources.dense import DenseTensorSource
from tensorkrowch.decompositions.sources.factory import SourceLike


@dataclass(frozen=True)
class _SVDProgress:  # MARK: _SVDProgress
    """Emits one live event for each completed SVD cut."""

    observer: DecompositionObserver  # Consumer of live SVD events
    phase: str  # Public algorithm name shown by the observer
    site_offset: int = 0  # Offset mapping local cuts to global sites
    subphase: Optional[str] = None  # Optional nested algorithmic phase

    def cut_complete(self,
                     site: int,
                     record: TruncationRecord,
                     elapsed: float) -> None:
        """Emits a completed cut using its global ``site`` position."""
        abs_error = record.local_abs_error
        rel_error = record.local_rel_error
        if abs_error.ndim:
            abs_error = torch.linalg.vector_norm(abs_error)
            local_norm = torch.linalg.vector_norm(record.local_norm)
            rel_error = safe_divide(abs_error, local_norm)
        values = {
            'full_rank': record.full_rank,
            'selected_rank': record.selected_rank,
            'absolute_error': abs_error,
            'relative_error': rel_error,
        }
        if self.subphase is not None:
            values['subphase'] = self.subphase

        self.observer.emit(DecompositionEvent(
            name='cut_complete',
            phase=self.phase,
            site=self.site_offset + site,
            elapsed=elapsed,
            values=values))


def _prepare_svd_source(source: SourceLike):
    """Obtains canonical dense site axes from a source's materialization."""
    if isinstance(source, torch.Tensor):
        source = DenseTensorSource(source)
    if not callable(getattr(source, 'contract_dense', None)):
        raise TypeError('`source` should provide `contract_dense`')
    if not getattr(source, 'is_discrete_tensor', True):
        raise ValueError('`source` should represent a discrete tensor')
    tensor = source.contract_dense()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('`contract_dense` should return a torch.Tensor')
    batches = getattr(source, 'n_batches', 0)
    if isinstance(batches, bool) or not isinstance(batches, int):
        raise TypeError('`n_batches` should be int type')
    if batches < 0 or batches >= tensor.ndim or any(size < 1 for size in tensor.shape):
        raise ValueError('The source should have positive finite input dimensions')
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise TypeError('Decomposition target values should be floating or complex')
    if getattr(source, 'is_matrix', False):
        dims = tuple(a * b for a, b in zip(source.in_dim, source.out_dim))
        tensor = tensor.reshape(*tensor.shape[:batches], *dims)
    return source, tensor, batches


__all__ = [
    '_SVDProgress',
    '_prepare_svd_source',
]
