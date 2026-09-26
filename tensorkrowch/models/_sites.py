"""Shared compatibility handling for MPS site counts."""

import warnings
from typing import Optional


def _resolve_n_sites(n_sites: Optional[int],
                     n_features: Optional[int]) -> Optional[int]:
    """Resolves the deprecated MPS constructor name before value validation."""
    if n_features is None:
        return n_sites
    if n_sites is not None:
        raise TypeError('`n_sites` and `n_features` cannot both be provided')
    warnings.warn(
        '`n_features` is deprecated and will be removed; use `n_sites`.',
        DeprecationWarning,
        stacklevel=3)
    return n_features
