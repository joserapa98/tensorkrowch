"""Common specifications used by SVD decomposition drivers."""

from dataclasses import dataclass
from math import exp, log
from typing import Dict, Optional, Union

import torch


def _log_vector_norm(tensor: torch.Tensor,
                     dim: Optional[int] = None,
                     keepdim: bool = False) -> torch.Tensor:
    """Computes a vector log-norm without squaring the original scale."""
    absolute = tensor.abs()
    if dim is None:
        if not tensor.numel():
            return absolute.new_tensor(-torch.inf)
        scale = absolute.amax()
        safe_scale = torch.where(
            scale > 0, scale, torch.ones_like(scale))
        normalized_norm = (absolute / safe_scale).square().sum().sqrt()
        log_norm = safe_scale.log() + normalized_norm.log()
        return torch.where(
            scale > 0,
            log_norm,
            torch.full_like(log_norm, -torch.inf))

    if not tensor.shape[dim]:
        shape = list(tensor.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim % tensor.ndim)
        return absolute.new_full(shape, -torch.inf)

    scale = absolute.amax(dim=dim, keepdim=True)
    safe_scale = torch.where(
        scale > 0, scale, torch.ones_like(scale))
    normalized_norm = (absolute / safe_scale).square().sum(
        dim=dim, keepdim=True).sqrt()
    log_norm = safe_scale.log() + normalized_norm.log()
    log_norm = torch.where(
        scale > 0,
        log_norm,
        torch.full_like(log_norm, -torch.inf))
    return log_norm if keepdim else log_norm.squeeze(dim)


def _normalize_vector(tensor: torch.Tensor,
                      dim: int = -1):
    """Normalizes vectors and returns their log-norms without overflow."""
    absolute = tensor.abs()
    scale = absolute.amax(dim=dim, keepdim=True)
    positive = scale > 0
    safe_scale = torch.where(positive, scale, torch.ones_like(scale))
    scaled = tensor / safe_scale
    scaled_norm = scaled.abs().square().sum(dim=dim, keepdim=True).sqrt()
    safe_scaled_norm = torch.where(
        positive, scaled_norm, torch.ones_like(scaled_norm))
    normalized = scaled / safe_scaled_norm
    log_norm = torch.where(
        positive,
        safe_scale.log() + safe_scaled_norm.log(),
        torch.full_like(safe_scale, -torch.inf))
    return normalized, log_norm.squeeze(dim)


def _rescale_absolute_tolerance(value: Optional[float],
                                log_scale: float,
                                power: int) -> Optional[float]:
    """Expresses an absolute tolerance in a normalized local scale."""
    if (value is None) or (value == 0):
        return value
    try:
        return exp(log(value) - power * log_scale)
    except OverflowError:
        return float('inf')


@dataclass(frozen=True)
class _TruncationSpec:
    """Groups the truncation criteria shared by consecutive SVD cuts."""

    rank: Optional[int] = None
    cutoff: Optional[float] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    cum_percentage: Optional[float] = None

    def __post_init__(self) -> None:
        if self.rank is not None:
            if (not isinstance(self.rank, int)) or (self.rank < 1):
                raise ValueError('`rank` should be a positive integer')
        for name in ('cutoff', 'atol'):
            value = getattr(self, name)
            if (value is not None) and \
                    ((not isinstance(value, (int, float))) or (value < 0)):
                raise ValueError(f'`{name}` should be a non-negative number')
        for name in ('rtol', 'cum_percentage'):
            value = getattr(self, name)
            if (value is not None) and \
                    ((not isinstance(value, (int, float))) or
                     (value < 0) or (value > 1)):
                raise ValueError(
                    f'`{name}` should be a number between 0 and 1')

    def as_kwargs(self) -> Dict[str, Union[int, float, None]]:
        """Returns keyword arguments accepted by ``truncated_svd``."""
        return {
            'rank': self.rank,
            'cutoff': self.cutoff,
            'atol': self.atol,
            'rtol': self.rtol,
            'cum_percentage': self.cum_percentage,
        }

    @property
    def requires_absolute_rescaling(self) -> bool:
        """Whether normalization changes an active absolute criterion."""
        return (self.cutoff not in (None, 0)) or (self.atol not in (None, 0))

    def as_normalized_kwargs(
            self,
            log_scale: float) -> Dict[str, Union[int, float, None]]:
        """Returns criteria expressed in a matrix's normalized scale."""
        kwargs = self.as_kwargs()
        kwargs['cutoff'] = _rescale_absolute_tolerance(
            self.cutoff, log_scale, power=1)
        kwargs['atol'] = _rescale_absolute_tolerance(
            self.atol, log_scale, power=2)
        return kwargs


__all__ = ['_TruncationSpec']
