"""
This module contains:

    Format interfaces:
        * TensorFormat, TensorFormat1D

    Formats:
        * TT, TR, TTM, TRM
        * QTT, QTR, QTTM, QTRM
        * QTTTucker, QTRTucker

    Bonds and gauges:
        * BondFactors1D, VidalGauge
        * GaugeOrbit, TensorRingOrbit

    Layouts and coordinate maps:
        * QuantizedLayout, CoordinateMap
        * UniformCoordinateMap, WarpedCoordinateMap, ExplicitGridMap

    Blocks:
        * BlockLayout, SplitBlock
        * split_block

    Diagnostics:
        * SampleError, RoundingInfo, MinimalCanonicalInfo

Module flow:

    cores ─> TT / TR / TTM / TRM
    cores + QuantizedLayout + CoordinateMap ─> QTT / QTR / QTTM / QTRM
    TT / TR + Quantics factors ─> QTTTucker / QTRTucker
    formats <─> models
    decompositions ─> formats + diagnostics
"""

from tensorkrowch.formats.base import (RoundingInfo, SampleError, BlockLayout,
                                       SplitBlock, TensorFormat)
from tensorkrowch.formats.bonds import BondFactors1D, VidalGauge
from tensorkrowch.formats.formats1d import (TensorFormat1D, TT, TR, TTM, TRM,
                                         RoundingInfo, BlockLayout,
                                         SplitBlock, split_block)
from tensorkrowch.formats.orbits import (GaugeOrbit, TensorRingOrbit,
                                       MinimalCanonicalInfo)
from tensorkrowch.formats.quantics import QTT, QTR, QTTM, QTRM
from tensorkrowch.formats.quantization import (QuantizedLayout, CoordinateMap,
                                             UniformCoordinateMap,
                                             WarpedCoordinateMap,
                                             ExplicitGridMap)
from tensorkrowch.formats.tucker import QTTTucker, QTRTucker


__all__ = [
    'TensorFormat',
    'TensorFormat1D',
    'SampleError',

    'TT',
    'TR',
    'TTM',
    'TRM',
    'QTT',
    'QTR',
    'QTTM',
    'QTRM',
    'QTTTucker',
    'QTRTucker',

    'BondFactors1D',
    'VidalGauge',
    'RoundingInfo',

    'BlockLayout',
    'SplitBlock',
    'split_block',

    'GaugeOrbit',
    'TensorRingOrbit',
    'MinimalCanonicalInfo',

    'QuantizedLayout',
    'CoordinateMap',
    'UniformCoordinateMap',
    'WarpedCoordinateMap',
    'ExplicitGridMap',
]
