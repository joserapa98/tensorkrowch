"""
This script contains:

    Internal truncation tools:
        * _rescale_absolute_tolerance
        * _TruncationSpec
"""

from dataclasses import dataclass
from math import exp, log
from typing import Dict, Optional, Union

from tensorkrowch.utils import _validate_truncation


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
    """Groups truncation criteria shared by consecutive SVD cuts."""

    rank: Optional[int] = None
    cutoff: Optional[float] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    cum_percentage: Optional[float] = None

    def __post_init__(self) -> None:
        _validate_truncation(
            rank=self.rank,
            cutoff=self.cutoff,
            atol=self.atol,
            rtol=self.rtol,
            cum_percentage=self.cum_percentage)

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
