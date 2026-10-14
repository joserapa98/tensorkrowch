"""
Singular-value decomposition algorithms for tensor trains and tensor rings.

    Public engines:
        * TTSVD, TRSVD
        * TTMSVD, TRMSVD

    Public functions:
        * tt_svd, tr_svd, ttm_svd, trm_svd
        * qtt_svd, qtr_svd, qttm_svd, qtrm_svd

    Workflow:
        source.contract_dense() -> dense site axes
        Matrix source -> fuse input/output pairs
        TTSVD / TRSVD -> numerical TT / TR cores
        Source metadata -> vector/matrix/Quantics decomposition

The source owns dimensions, ordering and evaluation runtime. Engines fix it
at construction; ``fit`` controls truncation and numerical normalization.
``TTMSVD`` and ``TRMSVD`` restrict the same shared algorithms to matrix sources.
The functional interfaces return decomposition objects directly.
"""

from tensorkrowch.decompositions.svd.tr import TRSVD, qtr_svd, tr_svd
from tensorkrowch.decompositions.svd.trm import TRMSVD, qtrm_svd, trm_svd
from tensorkrowch.decompositions.svd.tt import TTSVD, qtt_svd, tt_svd
from tensorkrowch.decompositions.svd.ttm import TTMSVD, qttm_svd, ttm_svd


__all__ = [
    'TTSVD',
    'TRSVD',
    'TTMSVD',
    'TRMSVD',
    'tt_svd',
    'qtt_svd',
    'tr_svd',
    'qtr_svd',
    'ttm_svd',
    'qttm_svd',
    'trm_svd',
    'qtrm_svd',
]
