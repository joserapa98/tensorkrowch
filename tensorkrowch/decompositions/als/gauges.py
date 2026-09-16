"""
This script contains:

    Public classes:
        * GaugePolicy
        * NoGauge
        * QRGauge
        * SVDGauge

    Internal functions:
        * _validate_factor_input
        * _absorb_factor
        * _validate_region_arguments
        * _neighbor_regions

    Public functions:
        * resolve_gauge_policy
"""

from typing import (Optional,
                    Protocol,
                    Sequence,
                    Tuple,
                    Union,
                    runtime_checkable)

import torch


Direction = str


def _validate_factor_input(core: torch.Tensor, direction: Direction) -> None:
    """Validates one standard three-dimensional TT/TR core."""
    if not isinstance(core, torch.Tensor):
        raise TypeError('`core` should be torch.Tensor type')
    if core.ndim != 3:
        raise ValueError(
            '`core` should have left rank, input and right rank dimensions')
    if direction not in ('forward', 'reverse'):
        raise ValueError("`direction` should be 'forward' or 'reverse'")


@runtime_checkable
class GaugePolicy(Protocol):
    """Factors a solved core and absorbs its remaining gauge into a neighbor.

    ``factor(core, direction)`` returns a replacement core and an optional
    matrix. ``absorb(matrix, neighbor, direction)`` returns the receiving
    neighbor. Their contraction preserves the original pair, including complex
    values. Drivers choose NoGauge before factorization when no legal trainable
    receiver exists.

    ``direction`` is ``"forward"`` or ``"reverse"``. Standard core axes are
    ``(left rank, input, right rank)``; factorization never interprets leading
    axes as data batches. ``invalidated_regions`` identifies the affected
    positions.
    """

    def factor(
            self,
            core: torch.Tensor,
            direction: Direction
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Returns a gauged core and the factor to absorb into its neighbor."""

    def absorb(self,
               factor: torch.Tensor,
               neighbor: torch.Tensor,
               direction: Direction) -> torch.Tensor:
        """Absorbs a factor into the legal receiver of the sweep."""

    def invalidated_regions(self,
                            site: int,
                            direction: Direction,
                            n_sites: int) -> Sequence[int]:
        """Returns sites whose cached dependencies change after absorption."""


class NoGauge:
    """Leaves a solved core unchanged and produces no neighboring update.

    Implements :class:`GaugePolicy` in both sweep directions. This policy does
    not choose ranks or apply the singular-value truncation criteria of TT-SVD.
    """

    name = 'none'

    def factor(
            self,
            core: torch.Tensor,
            direction: Direction
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Factors the core along the outgoing rank for the requested
        direction."""
        _validate_factor_input(core, direction)
        return core, None

    def absorb(self,
               factor: torch.Tensor,
               neighbor: torch.Tensor,
               direction: Direction) -> torch.Tensor:
        """Absorbs the factor into the receiving core without changing the
        pair."""
        raise RuntimeError('NoGauge does not produce an absorbable factor')

    def invalidated_regions(self,
                            site: int,
                            direction: Direction,
                            n_sites: int) -> Sequence[int]:
        """Returns positions whose cached contractions depend on this gauge
        move."""
        _validate_region_arguments(site, direction, n_sites)
        return (site,)


class QRGauge:
    """Moves the orthogonality center using a non-truncating QR factorization.

    Implements :class:`GaugePolicy` in both sweep directions. This policy does
    not choose ranks or apply the singular-value truncation criteria of TT-SVD.
    """

    name = 'qr'

    def factor(
            self,
            core: torch.Tensor,
            direction: Direction
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Factors the core along the outgoing rank for the requested
        direction."""
        _validate_factor_input(core, direction)
        left_rank, in_dim, right_rank = core.shape
        if direction == 'forward':
            matrix = core.reshape(left_rank * in_dim, right_rank)
            isometry, factor = torch.linalg.qr(matrix, mode='reduced')
            if isometry.shape[1] != right_rank:
                raise ValueError(
                    'The right rank is not algebraically feasible for QR')
            return isometry.reshape(core.shape), factor

        matrix = core.reshape(left_rank, in_dim * right_rank)
        isometry, factor = torch.linalg.qr(
            matrix.mT.conj(), mode='reduced')
        if isometry.shape[1] != left_rank:
            raise ValueError(
                'The left rank is not algebraically feasible for QR')
        return isometry.mT.conj().reshape(core.shape), factor.mT.conj()

    def absorb(self,
               factor: torch.Tensor,
               neighbor: torch.Tensor,
               direction: Direction) -> torch.Tensor:
        """Absorbs the factor into the receiving core without changing the
        pair."""
        return _absorb_factor(factor, neighbor, direction)

    def invalidated_regions(self,
                            site: int,
                            direction: Direction,
                            n_sites: int) -> Sequence[int]:
        """Returns positions whose cached contractions depend on this gauge
        move."""
        return _neighbor_regions(site, direction, n_sites)


class SVDGauge:
    """Moves the orthogonality center using a non-truncating SVD factorization.

    Implements :class:`GaugePolicy` in both sweep directions. This policy does
    not choose ranks or apply the singular-value truncation criteria of TT-SVD.
    """

    name = 'svd'

    def factor(
            self,
            core: torch.Tensor,
            direction: Direction
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Factors the core along the outgoing rank for the requested
        direction."""
        _validate_factor_input(core, direction)
        left_rank, in_dim, right_rank = core.shape
        if direction == 'forward':
            matrix = core.reshape(left_rank * in_dim, right_rank)
            u, singular_values, vh = torch.linalg.svd(
                matrix, full_matrices=False)
            if u.shape[1] != right_rank:
                raise ValueError(
                    'The right rank is not algebraically feasible for SVD')
            factor = singular_values.to(vh.dtype).unsqueeze(1) * vh
            return u.reshape(core.shape), factor

        matrix = core.reshape(left_rank, in_dim * right_rank)
        u, singular_values, vh = torch.linalg.svd(
            matrix, full_matrices=False)
        if vh.shape[0] != left_rank:
            raise ValueError(
                'The left rank is not algebraically feasible for SVD')
        factor = u * singular_values.to(u.dtype).unsqueeze(0)
        return vh.reshape(core.shape), factor

    def absorb(self,
               factor: torch.Tensor,
               neighbor: torch.Tensor,
               direction: Direction) -> torch.Tensor:
        """Absorbs the factor into the receiving core without changing the
        pair."""
        return _absorb_factor(factor, neighbor, direction)

    def invalidated_regions(self,
                            site: int,
                            direction: Direction,
                            n_sites: int) -> Sequence[int]:
        """Returns positions whose cached contractions depend on this gauge
        move."""
        return _neighbor_regions(site, direction, n_sites)


def _absorb_factor(factor: torch.Tensor,
                   neighbor: torch.Tensor,
                   direction: Direction) -> torch.Tensor:
    """Absorbs a square factor into a standard neighboring core."""
    if not isinstance(factor, torch.Tensor):
        raise TypeError('`factor` should be torch.Tensor type')
    _validate_factor_input(neighbor, direction)
    if factor.ndim != 2:
        raise ValueError('`factor` should be a matrix')
    if (factor.device != neighbor.device) or \
            (factor.dtype != neighbor.dtype):
        raise ValueError('`factor` and `neighbor` should share runtime')
    if direction == 'forward':
        if factor.shape[1] != neighbor.shape[0]:
            raise ValueError('The factor does not match the neighbor left rank')
        return torch.einsum('ab,bpc->apc', factor, neighbor)
    if factor.shape[0] != neighbor.shape[-1]:
        raise ValueError('The factor does not match the neighbor right rank')
    return torch.einsum('apb,bc->apc', neighbor, factor)


def _validate_region_arguments(site: int,
                               direction: Direction,
                               n_sites: int) -> None:
    """Validates invalidation-region arguments."""
    if isinstance(site, bool) or not isinstance(site, int):
        raise TypeError('`site` should be int type')
    if isinstance(n_sites, bool) or not isinstance(n_sites, int):
        raise TypeError('`n_sites` should be int type')
    if (n_sites < 1) or (site < 0) or (site >= n_sites):
        raise ValueError('`site` should identify one of `n_sites` sites')
    if direction not in ('forward', 'reverse'):
        raise ValueError("`direction` should be 'forward' or 'reverse'")


def _neighbor_regions(site: int,
                      direction: Direction,
                      n_sites: int) -> Sequence[int]:
    """Returns the solved site and its receiver when one exists."""
    _validate_region_arguments(site, direction, n_sites)
    neighbor = site + 1 if direction == 'forward' else site - 1
    if (neighbor < 0) or (neighbor >= n_sites):
        return (site,)
    return (site, neighbor)


def resolve_gauge_policy(
        gauge: Union[str, GaugePolicy]) -> GaugePolicy:
    """Normalizes public gauge names or compatible advanced policies."""
    if gauge == 'none':
        return NoGauge()
    if gauge == 'qr':
        return QRGauge()
    if gauge == 'svd':
        return SVDGauge()
    if isinstance(gauge, str):
        raise ValueError("`gauge` should be 'none', 'qr' or 'svd'")
    if not isinstance(gauge, GaugePolicy):
        raise TypeError('`gauge` should be a gauge name or GaugePolicy')
    return gauge


__all__ = [
    'GaugePolicy',
    'NoGauge',
    'QRGauge',
    'SVDGauge',
    'resolve_gauge_policy',
]
