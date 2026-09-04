"""
This script contains:

    Internal SVD numerical helpers:
        * _log_vector_norm
        * _normalize_vector
"""

from typing import Optional

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


__all__ = ['_log_vector_norm', '_normalize_vector']
