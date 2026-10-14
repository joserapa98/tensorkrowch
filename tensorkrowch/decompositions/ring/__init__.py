"""
Local loop opening, gauge recursion and tensor train to ring conversion.

Public conversion:

    tt2tr(...) = TT2TR(tt).fit(...) ─> TRDecomposition
        ├─ TT core provider ─> local core or supercore targets
        ├─ BidirectionalRingDriver ─> center, outward sweeps and boundaries
        ├─ LoopOpener ─> left gauge, input cores and right gauge
        └─ GaugeRecursion ─> the next local problem

Interchangeable loop openers:

    ALSLoopOpener ─> TRALS with zero or one fixed gauge
    FixedGaugeCoreOpener ─> LeastSquaresSolver with both gauges fixed
    BLOSTRLoopOpener ─> experimental unrestricted spectral initialization
    CallableLoopOpener / CompositeLoopOpener ─> custom or composed strategies

Shared ring infrastructure:

    CentralBlockSelector / RingRankEstimator ─> block capacity and rank choices
    split_block_ttsvd ─> TTSVD with retained external rank axes
    GaugeMap ─> directional duals and cancellation diagnostics
    PseudoinverseGaugeRecursion ─> characterized TT-to-TR transport
    TTCoreGaugeRecursion ─> experimental transport through original TT cores

The ring driver is also reused by TR recursive sketching with its own target
provider and gauge recursion. AlternatingRingDriver is the experimental serial
schedule added in phase 3. The standalone tr_blostr function uses the same
spectral engine as BLOSTRLoopOpener. Conversion computes final fidelity and
reconstruction error without densifying the TT/TR networks.
"""

from tensorkrowch.decompositions.ring.blostr import BLOSTRLoopOpener, tr_blostr
from tensorkrowch.decompositions.ring.gauges import (ExperimentalWarning,
                                                     GaugeMap,
                                                     GaugeRecursion,
                                                     GaugeRecursionStep,
                                                     PseudoinverseGaugeRecursion,
                                                     TTCoreGaugeRecursion)
from tensorkrowch.decompositions.ring.opening import (ALSLoopOpener,
                                                      CallableLoopOpener,
                                                      CompositeLoopOpener,
                                                      FixedGaugeCoreOpener,
                                                      LoopOpener,
                                                      LoopOpenerCapabilities,
                                                      LoopOpening)
from tensorkrowch.decompositions.ring.schedules import AlternatingRingDriver
from tensorkrowch.decompositions.ring.tt2tr import TT2TR, tt2tr


__all__ = [
    'GaugeMap',
    'ExperimentalWarning',
    'BLOSTRLoopOpener',
    'tr_blostr',
    'GaugeRecursion',
    'GaugeRecursionStep',
    'PseudoinverseGaugeRecursion',
    'TTCoreGaugeRecursion',
    'LoopOpening',
    'LoopOpenerCapabilities',
    'LoopOpener',
    'ALSLoopOpener',
    'FixedGaugeCoreOpener',
    'CallableLoopOpener',
    'CompositeLoopOpener',
    'AlternatingRingDriver',
    'TT2TR',
    'tt2tr',
]
