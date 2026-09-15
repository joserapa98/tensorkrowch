"""
Singular-value decomposition algorithms for tensor networks.

Public vector decomposition engines:

    TTSVD.fit()
        └─ successive open-boundary SVD cuts ──────> TTDecomposition

    TRSVD.fit()
        ├─ initial interior bipartition
        ├─ split the central rank into cyclic ranks
        └─ TTSVD on the left and right subchains ─> TRDecomposition

Public matrix decomposition engines:

    TTMSVD.fit()
        ├─ _prepare_matrix_input() ────────────────> _MatrixInput
        ├─ fuse each local (in, out) pair
        ├─ TTSVD.fit()
        └─ reopen local pairs ────────────────────> TTMDecomposition

    TRMSVD.fit()
        ├─ _prepare_matrix_input() ────────────────> _MatrixInput
        ├─ fuse each local (in, out) pair
        ├─ TRSVD.fit()
        └─ reopen local pairs ────────────────────> TRMDecomposition

Simple functional interfaces:

    tt_svd(...)   = TTSVD(...).fit(...)
    tr_svd(...)   = TRSVD(...).fit(...)
    ttm_svd(...)  = TTMSVD(...).fit(...)
    trm_svd(...)  = TRMSVD(...).fit(...)

Shared internal infrastructure:

    svd.utils
        Stable tensor normalization and logarithmic norm calculations.

    _TruncationSpec
        Shared rank and singular-value truncation criteria.

    _SVDProgress
        Shared live reporting of completed SVD cuts.

    _MatrixInput
        Normalized matrix dimensions and interleaved/fused dense tensors.

    _prepare_matrix_input(...)
        Validates and converts dense matrix inputs into _MatrixInput.

The engine classes fix the tensor and structural arguments at construction and
allow repeated calls to fit() with different truncation options. The functional
interfaces instantiate the corresponding engine, run one fit and return its
cores, optionally together with decomposition information.
"""

from tensorkrowch.decompositions.svd.tt import TTSVD, tt_svd
from tensorkrowch.decompositions.svd.tr import TRSVD, tr_svd
from tensorkrowch.decompositions.svd.ttm import TTMSVD, ttm_svd
from tensorkrowch.decompositions.svd.trm import TRMSVD, trm_svd


__all__ = [
    'TTSVD',
    'TRSVD',
    'TTMSVD',
    'TRMSVD',
    'tt_svd',
    'tr_svd',
    'ttm_svd',
    'trm_svd',
]
