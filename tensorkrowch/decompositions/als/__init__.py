"""Alternating least-squares decomposition infrastructure."""

from tensorkrowch.decompositions.als.environments import (
    CoreUpdateSet,
    DirectTREnvironment,
    EnvironmentCache,
    TRLocalEnvironment,
    TRSegmentEnvironmentCache,
    TTEnvironmentCache,
    TTLocalEnvironment,
)
from tensorkrowch.decompositions.als.gauges import (GaugePolicy, NoGauge,
                                                    QRGauge, SVDGauge)
from tensorkrowch.decompositions.als.convergence import (ConvergencePolicy,
                                                         UpdatePolicy)
from tensorkrowch.decompositions.als.driver import ALSSweepDriver
from tensorkrowch.decompositions.als.problem import (ALSProblem,
                                                     ObservedEntries)
from tensorkrowch.decompositions.als.sampling import (ExactRows, ObservedRows,
                                                      RowSampler, SampleBatch,
                                                      SampleRefreshPolicy,
                                                      TTLeverageRows,
                                                      TRExactLeverageRows,
                                                      TRProductLeverageRows,
                                                      UniformRows)
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.als.tt import TTALS, tt_als
from tensorkrowch.decompositions.als.tr import TRALS, tr_als


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
