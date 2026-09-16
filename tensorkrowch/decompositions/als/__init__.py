"""
Alternating least-squares algorithms for tensor trains and tensor rings.

Public engines and functional interfaces:

    tt_als(...) = TTALS(source).fit(...) ─> TTDecomposition
    tr_als(...) = TRALS(source).fit(...) ─> TRDecomposition

    TTALS.completion(...) / TRALS.completion(...)
        └─ ObservedEntries ─> fixed weighted objective on observed entries

Shared execution:

    ALSProblem + initialized cores + optional strategies
        └─ ALSSweepDriver
            ├─ exact / observed / uniform / leverage rows
            ├─ topology-specific local environment and target
            ├─ LeastSquaresSolver ─> UpdatePolicy ─> GaugePolicy
            ├─ CoreUpdateSet ─> atomic core and cache update
            └─ ConvergencePolicy at complete sweep boundaries

Topology-specific backends:

    TTALS
        ├─ TTEnvironmentCache for exact, uniform and observed rows
        └─ mixed-canonical per-site environments for leverage sampling

    TRALS
        ├─ TRSegmentEnvironmentCache for exact, uniform and observed rows
        └─ product or experimental exact leverage with complete input fibers

TRALS reuses TTALS source preparation and completion construction, while its
backends own cyclic ranks, contractions and gauge receivers. The common driver
contains no TT/TR contraction logic. Renewable sample residuals are local
measurements and cannot replace a fixed objective for convergence. Public
engines own console lifecycle; their internal drivers emit live updates.
"""

from tensorkrowch.decompositions.als.convergence import (ConvergencePolicy,
                                                         UpdatePolicy)
from tensorkrowch.decompositions.als.driver import ALSSweepDriver
from tensorkrowch.decompositions.als.environments import (CoreUpdateSet,
                                                          DirectTREnvironment,
                                                          EnvironmentCache,
                                                          TRLocalEnvironment,
                                                          TRSegmentEnvironmentCache,
                                                          TTEnvironmentCache,
                                                          TTLocalEnvironment)
from tensorkrowch.decompositions.als.gauges import (GaugePolicy,
                                                    NoGauge,
                                                    QRGauge,
                                                    SVDGauge)
from tensorkrowch.decompositions.als.problem import ALSProblem, ObservedEntries
from tensorkrowch.decompositions.als.sampling import (ExactRows,
                                                      ObservedRows,
                                                      RowSampler,
                                                      SampleBatch,
                                                      SampleRefreshPolicy,
                                                      TRExactLeverageRows,
                                                      TRProductLeverageRows,
                                                      TTLeverageRows,
                                                      UniformRows)
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.als.tr import TRALS, tr_als
from tensorkrowch.decompositions.als.tt import TTALS, tt_als


__all__ = [
    'ALSProblem',
    'ObservedEntries',
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
    'EnvironmentCache',
    'CoreUpdateSet',
    'TTLocalEnvironment',
    'TTEnvironmentCache',
    'TRLocalEnvironment',
    'DirectTREnvironment',
    'TRSegmentEnvironmentCache',
    'ConvergencePolicy',
    'UpdatePolicy',
    'ALSSweepDriver',
    'GaugePolicy',
    'NoGauge',
    'QRGauge',
    'SVDGauge',
    'TTALS',
    'tt_als',
    'TRALS',
    'tr_als',
]
