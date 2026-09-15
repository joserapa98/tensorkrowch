"""Singular-value tensor decompositions."""

from tensorkrowch.decompositions.svd.tt import TTSVD, tt_svd
from tensorkrowch.decompositions.svd.ttm import TTMSVD, ttm_svd
from tensorkrowch.decompositions.svd.tr import TRSVD, tr_svd
from tensorkrowch.decompositions.svd.trm import TRMSVD, trm_svd


__all__ = [
    'TTSVD',
    'TTMSVD',
    'TRSVD',
    'TRMSVD',
    'tt_svd',
    'ttm_svd',
    'tr_svd',
    'trm_svd',
]
