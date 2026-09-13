"""
This script contains:

    Internal SVD numerical helpers:
        * _vector_norm_components
        * _log_vector_norm
        * _normalize_vector
"""

from typing import Optional, Tuple

import torch


def _vector_norm_components(tensor: torch.Tensor,
                            dim: Optional[int] = None) -> Tuple[
                                torch.Tensor,
                                torch.Tensor,
                                torch.Tensor]:
    """Returns stable scale factors and the log-norm of vectors."""
    absolute = tensor.abs()
    if dim is None:
        scale = absolute.amax()
    else:
        scale = absolute.amax(dim=dim, keepdim=True)

    positive = scale > 0
    safe_scale = torch.where(positive, scale, torch.ones_like(scale))
    if dim is None:
        scaled_norm = (absolute / safe_scale).square().sum().sqrt()
    else:
        scaled_norm = (absolute / safe_scale).square().sum(
            dim=dim, keepdim=True).sqrt()

    safe_scaled_norm = torch.where(
        positive, scaled_norm, torch.ones_like(scaled_norm))
    log_norm = torch.where(
        positive,
        safe_scale.log() + safe_scaled_norm.log(),
        torch.full_like(safe_scale, -torch.inf))
    return safe_scale, safe_scaled_norm, log_norm


def _log_vector_norm(tensor: torch.Tensor,
                     dim: Optional[int] = None,
                     keepdim: bool = False) -> torch.Tensor:
    """Computes a vector log-norm without squaring the original scale."""
    if dim is None:
        if not tensor.numel():
            return tensor.real.new_tensor(-torch.inf)
        return _vector_norm_components(tensor)[-1]

    if not tensor.shape[dim]:
        shape = list(tensor.shape)
        if keepdim:
            shape[dim] = 1
        else:
            shape.pop(dim % tensor.ndim)
        return tensor.real.new_full(shape, -torch.inf)

    log_norm = _vector_norm_components(tensor, dim)[-1]
    return log_norm if keepdim else log_norm.squeeze(dim)


def _normalize_vector(tensor: torch.Tensor,
                      dim: int = -1):
    """Normalizes vectors and returns their log-norms without overflow."""
    safe_scale, safe_scaled_norm, log_norm = _vector_norm_components(
        tensor, dim)
    normalized = tensor / safe_scale / safe_scaled_norm
    return normalized, log_norm.squeeze(dim)


__all__ = ['_log_vector_norm', '_normalize_vector']
