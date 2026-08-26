"""Singular-value tensor decompositions."""

from tensorkrowch.decompositions.svd.tt import TTSVD, tt_svd
from tensorkrowch.decompositions.svd.ttm import TTMSVD, ttm_svd


__all__ = ['TTSVD', 'TTMSVD', 'tt_svd', 'ttm_svd']
