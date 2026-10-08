"""
This module contains:

    Format interfaces:
        * TensorFormat, TensorFormat1D

    Formats:
        * TT, TR, TTM, TRM
        * QTT, QTR, QTTM, QTRM

    Bonds and gauges:
        * BondFactors1D, VidalGauge
        * GaugeOrbit, TensorRingOrbit

    Layouts and coordinate maps:
        * QuantizedLayout, CoordinateMap
        * AffineCoordinateMap, FunctionalCoordinateMap, ExplicitGridMap

    Blocks:
        * BlockLayout, SplitBlock
        * split_block

    Diagnostics:
        * SampleError, RoundingInfo, MinimalCanonicalInfo

Module flow:

    cores ─> TT / TR / TTM / TRM
    cores + QuantizedLayout + CoordinateMap ─> QTT / QTR / QTTM / QTRM
    formats <─> models
    decompositions ─> formats + diagnostics
"""

from tensorkrowch.formats.base import (RoundingInfo, SampleError, BlockLayout,
                                     SplitBlock, TensorFormat)
from tensorkrowch.formats.bonds import BondFactors1D, VidalGauge
from tensorkrowch.formats.formats1d import (TensorFormat1D, TT, TR, TTM, TRM,
                                         split_block)
from tensorkrowch.formats.orbits import (GaugeOrbit, TensorRingOrbit,
                                       MinimalCanonicalInfo)
from tensorkrowch.formats.quantics import QTT, QTR, QTTM, QTRM
from tensorkrowch.formats.quantization import (QuantizedLayout, CoordinateMap,
                                             AffineCoordinateMap,
                                             FunctionalCoordinateMap,
                                             ExplicitGridMap)


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
    'AffineCoordinateMap',
    'FunctionalCoordinateMap',
    'ExplicitGridMap',
]
