"""
Shared QR gauge operations for one-site ALS.

    Internal functions:
        * _factor_core
        * _absorb_factor

The decomposition chooses whether to move a gauge and identifies its receiving
core. These helpers factor and absorb it without changing the contracted pair
or truncating its prescribed bond dimension.
"""

from typing import Tuple

import torch


def _factor_core(core: torch.Tensor,
                 direction: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factors a standard core into a directional isometry and a square gauge.
    """
    if not isinstance(core, torch.Tensor):
        raise TypeError('`core` should be torch.Tensor type')
    if core.ndim != 3:
        raise ValueError(
            '`core` should have left rank, input and right rank dimensions')
    if direction not in ('forward', 'reverse'):
        raise ValueError("`direction` should be 'forward' or 'reverse'")

    left_rank, in_dim, right_rank = core.shape
    if direction == 'forward':
        matrix = core.reshape(left_rank * in_dim, right_rank)
        isometry, factor = torch.linalg.qr(matrix, mode='reduced')
        if isometry.shape[1] != right_rank:
            raise ValueError('`core` right rank should be feasible for QR')
        return isometry.reshape(core.shape), factor

    matrix = core.reshape(left_rank, in_dim * right_rank)
    isometry, factor = torch.linalg.qr(matrix.mH, mode='reduced')
    if isometry.shape[1] != left_rank:
        raise ValueError('`core` left rank should be feasible for QR')
    return isometry.mH.reshape(core.shape), factor.mH


def _absorb_factor(factor: torch.Tensor,
                   neighbor: torch.Tensor,
                   direction: str) -> torch.Tensor:
    """
    Absorbs a gauge into the immediate receiver without changing the pair.
    """
    if not isinstance(factor, torch.Tensor):
        raise TypeError('`factor` should be torch.Tensor type')
    if not isinstance(neighbor, torch.Tensor):
        raise TypeError('`neighbor` should be torch.Tensor type')
    if (factor.ndim != 2) or (neighbor.ndim != 3):
        raise ValueError('`factor` should be a matrix and `neighbor` a core')
    if direction not in ('forward', 'reverse'):
        raise ValueError("`direction` should be 'forward' or 'reverse'")
    if (factor.device != neighbor.device) or (factor.dtype != neighbor.dtype):
        raise ValueError(
            '`factor` and `neighbor` should share dtype and device')

    if direction == 'forward':
        if factor.shape[1] != neighbor.shape[0]:
            raise ValueError(
                '`factor` should match the left rank of `neighbor`')
        return torch.einsum('ab,bpc->apc', factor, neighbor)
    if factor.shape[0] != neighbor.shape[-1]:
        raise ValueError('`factor` should match the right rank of `neighbor`')
    return torch.einsum('apb,bc->apc', neighbor, factor)
