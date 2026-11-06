"""
This script contains:

    Internal truncation tools:
        * _rescale_absolute_tolerance
        * _TruncationSpec
"""

from dataclasses import dataclass, replace
from math import exp, log
from typing import Dict, Optional, Sequence, Union

from tensorkrowch.utils import _validate_truncation

from tensorkrowch.decompositions._utils import (_normalize_dims,
                                               _normalize_positive_int)


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
class _TruncationSpec:  # MARK: _TruncationSpec
    """Owns shared or per-link rank caps and consecutive SVD tolerances."""

    # Shared maximum rank or one maximum per link
    rank: Optional[Union[int, Sequence[int]]] = None
    cutoff: Optional[float] = None  # Minimum singular value retained
    atol: Optional[float] = None  # Absolute discarded-energy tolerance
    rtol: Optional[float] = None  # Relative discarded-energy tolerance
    cum_percentage: Optional[float] = None  # Retained spectral-energy fraction

    def __post_init__(self) -> None:
        if self.rank is not None:
            if isinstance(self.rank, int):
                _normalize_positive_int(self.rank, 'rank')
            else:
                object.__setattr__(self, 'rank', _normalize_dims(
                    self.rank, 'rank', allow_empty=True))
        _validate_truncation(cutoff=self.cutoff,
                             atol=self.atol,
                             rtol=self.rtol,
                             cum_percentage=self.cum_percentage)

    def validate_rank(self, n_links: int) -> None:
        """Checks a rank profile against the number of represented links."""
        if isinstance(self.rank, tuple) and (len(self.rank) != n_links):
            raise ValueError(
                f'`rank` should contain {n_links} values, one per link')

    def rank_at(self, site: Optional[int] = None) -> Optional[int]:
        """Returns one scalar cap, requiring a site for per-link profiles."""
        if (self.rank is None) or isinstance(self.rank, int):
            return self.rank
        if site is None:
            raise ValueError('A per-link `rank` requires `site`')
        return self.rank[site]

    def for_links(self, start: int, stop: int) -> '_TruncationSpec':
        """Restricts per-link caps to a subchain, retaining all tolerances."""
        if isinstance(self.rank, tuple):
            return replace(self, rank=self.rank[start:stop])
        return self

    def as_kwargs(self, site: Optional[int] = None
                  ) -> Dict[str, Union[int, float, None]]:
        """Returns scalar ``truncated_svd`` controls for one cut."""
        return {
            'rank': self.rank_at(site),
            'cutoff': self.cutoff,
            'atol': self.atol,
            'rtol': self.rtol,
            'cum_percentage': self.cum_percentage,
        }

    @property
    def requires_absolute_rescaling(self) -> bool:
        """Whether normalization changes an active absolute criterion."""
        return (self.cutoff not in (None, 0)) or (self.atol not in (None, 0))

    def as_normalized_kwargs(self,
                             log_scale: float,
                             site: Optional[int] = None
                             ) -> Dict[str, Union[int, float, None]]:
        """Returns criteria expressed in a matrix's normalized scale."""
        kwargs = self.as_kwargs(site)
        kwargs['cutoff'] = _rescale_absolute_tolerance(
            self.cutoff, log_scale, power=1)
        kwargs['atol'] = _rescale_absolute_tolerance(
            self.atol, log_scale, power=2)
        return kwargs


__all__ = [
    '_TruncationSpec',
]
