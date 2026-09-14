"""
This script contains:

    Internal SVD numerical helpers:
        * _tensor_norm_components
        * _log_tensor_norm
        * _normalize_tensor
"""

from typing import Optional, Sequence, Tuple, Union

import torch


_Dimension = Optional[Union[int, Sequence[int]]]


def _tensor_norm_components(tensor: torch.Tensor,
                            dim: _Dimension = None) -> Tuple[
                                torch.Tensor,
                                torch.Tensor,
                                torch.Tensor]:
    """Returns stable scale factors and the log-norm of tensors."""
    absolute = tensor.abs()
    if dim is None:
        scale = absolute.amax()
    else:
        if not isinstance(dim, int):
            dim = tuple(dim)
        scale = absolute.amax(dim=dim, keepdim=True)

    positive = scale > 0
    safe_scale = torch.where(positive, scale, torch.ones_like(scale))
    if dim is None:
        scaled_norm = torch.linalg.vector_norm(absolute / safe_scale)
    else:
        scaled_norm = torch.linalg.vector_norm(
            absolute / safe_scale, dim=dim, keepdim=True)

    safe_scaled_norm = torch.where(
        positive, scaled_norm, torch.ones_like(scaled_norm))
    log_norm = torch.where(
        positive,
        safe_scale.log() + safe_scaled_norm.log(),
        torch.full_like(safe_scale, -torch.inf))
    return safe_scale, safe_scaled_norm, log_norm


def _log_tensor_norm(tensor: torch.Tensor,
                     dim: _Dimension = None,
                     keepdim: bool = False) -> torch.Tensor:
    """Computes a tensor log-norm without squaring the original scale."""
    if dim is None:
        if not tensor.numel():
            return tensor.real.new_tensor(-torch.inf)
        return _tensor_norm_components(tensor)[-1]

    dims = (dim,) if isinstance(dim, int) else dim
    dims = tuple(axis % tensor.ndim for axis in dims)
    if any(not tensor.shape[axis] for axis in dims):
        shape = list(tensor.shape)
        if keepdim:
            for axis in dims:
                shape[axis] = 1
        else:
            shape = [size for axis, size in enumerate(shape)
                     if axis not in dims]
        return tensor.real.new_full(shape, -torch.inf)

    log_norm = _tensor_norm_components(tensor, dim)[-1]
    if keepdim:
        return log_norm
    for axis in sorted(dims, reverse=True):
        log_norm = log_norm.squeeze(axis)
    return log_norm


def _normalize_tensor(tensor: torch.Tensor,
                      dim: _Dimension = None):
    """Normalizes tensors and returns their log-norms without overflow."""
    safe_scale, safe_scaled_norm, log_norm = _tensor_norm_components(
        tensor, dim)
    scaled = tensor / safe_scale
    normalized = scaled / safe_scaled_norm
    if dim is None:
        return normalized, log_norm

    dims = (dim,) if isinstance(dim, int) else dim
    dims = tuple(axis % tensor.ndim for axis in dims)
    for axis in sorted(dims, reverse=True):
        log_norm = log_norm.squeeze(axis)
    return normalized, log_norm


__all__ = ['_log_tensor_norm', '_normalize_tensor']
