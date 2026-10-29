"""
Alternating least-squares algorithms for tensor trains and tensor rings.

Sources define the tensor values; ALS optimizes one core at a time while the
others remain fixed. Both engines and functions return decomposition objects
with cores, source metadata and optional diagnostics.

Public engine hierarchy and basic interfaces:

    TTALS                                      tt.py
        └─ TRALS                               tr.py

    tt_als(source, rank=..., ...) ─> TTALS(source).fit(...)
    tr_als(source, rank=..., ...) ─> TRALS(source).fit(...)

:class:`TTALS` fits open chains with alternating forward and reverse sweeps.
:class:`TRALS` reuses its source preparation and fits cyclic chains with
forward sweeps. Each directional pass counts as one sweep. Matrix and Quantics
sources select their corresponding result family through the same algorithms;
structural batches are not supported.

Construction and initialization:

    source + optional mask and residual weights
        └─ ALSProblem
            └─ TTALS / TRALS
                └─ fit(rank=... OR initial_cores=...)
                    ├─ random initialization at prescribed rank caps
                    └─ supplied cores copied before optimization

``rank`` is a shared bound or one bound per link: TT has ``n_sites - 1`` links,
TR has ``n_sites``, including the closing link. Supplied ranks must respect the
bounds when both arguments are provided. ``fixed_cores`` freezes selected
sites. Use a separate SVD decomposition as ``initial_cores`` when desired.

:class:`ALSProblem` owns the objective. A boolean mask selects participating
entries; a dense mask still queries the complete target, while a sparse mask
queries only its stored true entries. Explicit observations use a
:class:`~tensorkrowch.decompositions.SparseTensorSource` as storage, with only
its support participating. Without such a mask or observation objective,
unstored sparse source entries remain known zeros.

Advanced fitting controls:

    TTALS(source).fit(...)
        ├─ gauge="qr" / "none"     QR movement or direct core replacement
        ├─ renormalize             Scale environments without changing the loss
        ├─ sampler                 SamplingPolicy, or full objective by default
        ├─ solver                  LeastSquaresSolver
        └─ convergence             ConvergencePolicy

:class:`SamplingPolicy` groups sample count, method, reuse and optional uniform
mixing. TT leverage sampling uses mixed-canonical cores; TR offers product or
exact cyclic leverage sampling and retains complete active input fibers.
:class:`LeastSquaresSolver` wraps PyTorch least squares with scaling and
optional Tikhonov regularization. :class:`ConvergencePolicy` controls the sweep
budget, ``atol``, ``rtol``, stability and optional best-state retention.
Error-based stopping requires a fixed comparable objective; renewable sampled
residuals are local measurements. The functional interfaces expose basic exact
fits; use the engines for sampler, solver and convergence objects.

Shared execution and topology-specific contractions:

    ALSSweepDriver
        ├─ backend prepares rows and environments
        ├─ LeastSquaresSolver ─> solved core ─> optional QR absorption
        ├─ CoreUpdateSet ─> atomic core and cache update
        └─ complete-sweep objective ─> ConvergencePolicy

    TT backends
        ├─ TTEnvironmentCache ─> TTLocalEnvironment
        └─ mixed-canonical site-dependent leverage environments

    TR backends
        ├─ TRSegmentEnvironmentCache ─> TRLocalEnvironment
        └─ direct sampled environments for cyclic leverage

``EnvironmentCache`` and ``RowSampler`` describe low-level contracts.
``SampleBatch`` retains selected rows and their draw probabilities;
``SampleRefreshPolicy`` controls reusable generations. ``DirectTREnvironment``
provides a simple reference contraction. QR factorization is shared through
private helpers; it introduces no rank truncation.

``collect_metrics=True`` records local solves and sweep measurements in the
result's metrics. ``verbose`` controls console events. Use
``result.to_format()`` to retain the approximation without decomposition
diagnostics.
"""

from tensorkrowch.decompositions.als.convergence import ConvergencePolicy
from tensorkrowch.decompositions.als.driver import ALSSweepDriver
from tensorkrowch.decompositions.als.environments import (
    CoreUpdateSet,
    DirectTREnvironment,
    EnvironmentCache,
    TRLocalEnvironment,
    TRSegmentEnvironmentCache,
    TTEnvironmentCache,
    TTLocalEnvironment,
)
from tensorkrowch.decompositions.als.problem import ALSProblem
from tensorkrowch.decompositions.als.sampling import (ExactRows,
                                                      ObservedRows,
                                                      RowSampler,
                                                      SampleBatch,
                                                      SampleRefreshPolicy,
                                                      SamplingPolicy,
                                                      TRExactLeverageRows,
                                                      TRProductLeverageRows,
                                                      TTLeverageRows,
                                                      UniformRows)
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.als.tr import TRALS, tr_als
from tensorkrowch.decompositions.als.tt import TTALS, tt_als


__all__ = [
    'ALSProblem',
    'LeastSquaresSolver',
    'SampleBatch',
    'RowSampler',
    'ExactRows',
    'ObservedRows',
    'UniformRows',
    'TTLeverageRows',
    'TRProductLeverageRows',
    'TRExactLeverageRows',
    'SampleRefreshPolicy',
    'SamplingPolicy',
    'EnvironmentCache',
    'CoreUpdateSet',
    'TTLocalEnvironment',
    'TTEnvironmentCache',
    'TRLocalEnvironment',
    'DirectTREnvironment',
    'TRSegmentEnvironmentCache',
    'ConvergencePolicy',
    'ALSSweepDriver',
    'TTALS',
    'tt_als',
    'TRALS',
    'tr_als',
]
