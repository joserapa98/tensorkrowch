"""
Tensor train to tensor ring conversion and local loop-opening algorithms.

A tensor ring connects the first and last core through a cyclic bond. This
module converts an open-boundary TT to that representation and provides the
local factorizations used by recursive ring sketching. Algorithms return
:class:`~tensorkrowch.decompositions.TRDecomposition` objects containing cores,
errors and optional local diagnostics.

The main conversion interface is:

    tt2tr(tt, ...) = TT2TR(tt).fit(...)
        |
        +-- prescribed rank --> automatic block and outward construction
        +-- omitted rank ----> central and boundary blocks, rank discovery
        |
        +-- optimization opener --> gauge recursion and residual expansion
        +-- spectral opener ----> exact gauge absorption and SVD propagation

Local loop opening separates the target into a left gauge, physical TR cores
and a right gauge. Its public strategies are:

    LoopOpener                         shared local factorization protocol
        +-- ALSLoopOpener              iterative TR least squares
        +-- QRLoopOpener               QR balancing plus local refinement
        +-- BLOSTRLoopOpener           complete spectral recovery
        +-- BLOSTRSVDLoopOpener        first spectral core plus SVD tail
        +-- CallableLoopOpener         custom local factorization

    LoopOpening                        gauges, physical cores and diagnostics
    LoopOpenerCapabilities             supported fixed gauges and block sizes

The optimization route transports the learned factors using:

    GaugeRecursion
        +-- AdjointGaugeRecursion      moving QR centers or Vidal isometries
        +-- PinvGaugeRecursion         local TT-core pseudoinverses
        +-- OrthGaugeRecursion         learned-gauge pseudoinverses

Orthogonal residual expansion supplements these transports with constraints
that would otherwise be omitted by projection. Auxiliary coordinates may
increase, while prescribed TR ranks and the input tensor remain unchanged.
:class:`GaugeMap` exposes directional gauge duals and their numerical checks.

Standalone :func:`tr_blostr` and :func:`tr_blostr_svd` decompose dense tensors
using the same two spectral algorithms as their corresponding openers.
BLOSTR requires exact spectral structure; a small local optimization residual
alone does not certify a small global TR error. TT conversion measures final
fidelity and reconstruction error without materializing the full tensor.
"""

from tensorkrowch.decompositions.ring.blostr import tr_blostr, tr_blostr_svd
from tensorkrowch.decompositions.ring.gauges import (AdjointGaugeRecursion,
                                                     ExperimentalWarning,
                                                     GaugeMap,
                                                     GaugeRecursion,
                                                     GaugeRecursionStep,
                                                     OrthGaugeRecursion,
                                                     PinvGaugeRecursion)
from tensorkrowch.decompositions.ring.opening import (ALSLoopOpener,
                                                      BLOSTRLoopOpener,
                                                      BLOSTRSVDLoopOpener,
                                                      CallableLoopOpener,
                                                      LoopOpener,
                                                      LoopOpenerCapabilities,
                                                      LoopOpening,
                                                      QRLoopOpener)
from tensorkrowch.decompositions.ring.tt2tr import TT2TR, tt2tr


__all__ = [
    'TT2TR',
    'tt2tr',
    'tr_blostr',
    'tr_blostr_svd',
    'LoopOpener',
    'LoopOpenerCapabilities',
    'LoopOpening',
    'ALSLoopOpener',
    'QRLoopOpener',
    'BLOSTRLoopOpener',
    'BLOSTRSVDLoopOpener',
    'CallableLoopOpener',
    'GaugeRecursion',
    'GaugeRecursionStep',
    'AdjointGaugeRecursion',
    'PinvGaugeRecursion',
    'OrthGaugeRecursion',
    'GaugeMap',
    'ExperimentalWarning',
]
