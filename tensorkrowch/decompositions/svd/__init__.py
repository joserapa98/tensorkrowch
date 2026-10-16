"""
Singular-value decomposition algorithms for tensor trains and tensor rings.

Public engine hierarchy:

    TTSVD
        └─ TTMSVD

    TRSVD
        └─ TRMSVD

:class:`TTSVD` performs successive open-chain SVD cuts. :class:`TRSVD` first
splits an interior bipartition into two blocks and then uses the TT-SVD
numerical routines to decompose both subchains with their boundary ranks
preserved. These are independent engine classes; TR-SVD reuses TT-SVD
through composition, not inheritance.

:class:`TTMSVD` and :class:`TRMSVD` inherit the corresponding algorithms and
restrict their inputs to matrix sources. Matrix dimensions and ordering
belong to the source; each local input/output pair is fused for the numerical
algorithm and restored in the resulting matrix cores. The general engines
also accept matrix sources and infer the corresponding result type.

Input preparation at engine construction:

    torch.Tensor
        └─ DenseTensorSource
            Matrix interfaces use DenseTensorSource.matrix(...).

    Finite source
        └─ _prepare_svd_source(source)
            ├─ source.contract_dense() ───────────> dense site axes
            ├─ matrix source: fuse local input/output pairs
            └─ retain source and prepared tensor in the engine

The source owns dimensions, structural batches, matrix ordering and evaluation
runtime. A :class:`~tensorkrowch.decompositions.CallableQuanticsSource` also
owns its layouts and coordinate maps, and evaluates its full digit grid for
SVD. Ordinary tensor sources retain their fixed axes without quantization.
Preparation materializes the source once; the public ``source`` property
retains the original source, while repeated calls to ``fit`` use the prepared
dense tensor.

Numerical workflows:

    TTSVD(source).fit(...)
        ├─ successive open-chain SVD cuts
        ├─ truncate singular values at each cut
        └─ _result_from_source(...) ─────────────> TT / TTM decomposition

    TRSVD(source, center=...).fit(...)
        ├─ SVD across the initial interior bipartition
        ├─ factor its retained rank into two cyclic ranks
        ├─ shared TT-SVD routines on the left and right subchains
        └─ _result_from_source(...) ─────────────> TR / TRM decomposition

    TTMSVD(matrix_source).fit(...)
        └─ inherited TTSVD.fit(...) ─────────────> TTMDecomposition

    TRMSVD(matrix_source, center=...).fit(...)
        └─ inherited TRSVD.fit(...) ─────────────> TRMDecomposition

``fit`` controls truncation, numerical normalization and optional diagnostics.
TT-SVD supports leading structural batches; TR-SVD does not. The source
selects the result family: vector or matrix, and ordinary or Quantics.
``_result_from_source`` restores paired matrix axes and attaches the source's
evaluation information, layouts and coordinate maps. Quantics inputs produce
``QTTDecomposition``, ``QTRDecomposition``, ``QTTMDecomposition`` or
``QTRMDecomposition`` through the same numerical algorithms.

Functional interfaces for a single fit:

    tt_svd(source, ...)   = TTSVD(source, ...).fit(...)
    tr_svd(source, ...)   = TRSVD(source, ...).fit(...)
    ttm_svd(source, ...)  = TTMSVD(source, ...).fit(...)
    trm_svd(source, ...)  = TRMSVD(source, ...).fit(...)

    qtt_svd(source, ...)  -> tt_svd(...)
    qtr_svd(source, ...)  -> tr_svd(...)
    qttm_svd(source, ...) -> ttm_svd(...)
    qtrm_svd(source, ...) -> trm_svd(...)

The Quantics functions require a ``CallableQuanticsSource`` of the appropriate
vector or matrix kind. There are no separate Quantics engines or numerical
algorithms. All functional interfaces return decomposition objects containing
cores, metadata and optional metrics. Use ``result.to_format()`` to continue
with the corresponding format without decomposition metadata.

Shared internal infrastructure:

    svd.utils
        Source preparation and _SVDProgress events.

    _TruncationSpec
        Shared rank selection and singular-value truncation criteria.

    _RuntimePolicy
        Numerical runtime and device placement of finalized cores.

    results._result_from_source
        Constructs the vector/matrix/Quantics result with source information.
"""

from tensorkrowch.decompositions.svd.tt import TTSVD, tt_svd, qtt_svd
from tensorkrowch.decompositions.svd.tr import TRSVD, tr_svd, qtr_svd
from tensorkrowch.decompositions.svd.ttm import TTMSVD, ttm_svd, qttm_svd
from tensorkrowch.decompositions.svd.trm import TRMSVD, trm_svd, qtrm_svd


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
