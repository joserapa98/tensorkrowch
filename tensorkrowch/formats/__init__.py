"""
Compact tensor formats implemented directly with PyTorch.

These lightweight classes represent fixed data or functions and support
numerical decomposition and solver algorithms. They do not construct the
nodes, edges or operation graphs of TensorKrowch. Models remain useful for
parameterized functions learned from data, with optimized repeated operation
flows across gradient descent steps. Formats preserve autograd where their
PyTorch operations support it; they do not detach inputs implicitly.

This script contains:
    * TensorFormat, TensorFormat1D, TensorFormat2D
    * TT, TR
    * TTM, TRM
    * SampleError
    * BondFactors, VidalGauge, RoundingInfo
    * GaugeOrbit, TensorRingOrbit
    * BlockLayout, UnblockInfo, SplitBlock, split_block
    * QuantizedLayout, CoordinateMap and coordinate maps
    * QTT, QTR
    * QTTM, QTRM
    * QTTTucker, QTRTucker
"""

from tensorkrowch.formats.base import TensorFormat, TensorFormat2D, SampleError
from tensorkrowch.formats._chain import TensorFormat1D
from tensorkrowch.formats.blocking import (BlockLayout, UnblockInfo,
                                           SplitBlock, split_block)
from tensorkrowch.formats.bonds import BondFactors, VidalGauge
from tensorkrowch.formats.orbits import (GaugeOrbit, TensorRingOrbit,
                                         MinimalCanonicalInfo)
from tensorkrowch.formats.quantics import QTT, QTR, QTTM, QTRM
from tensorkrowch.formats.quantization import (QuantizedLayout, CoordinateMap,
                                               UniformCoordinateMap,
                                               WarpedCoordinateMap,
                                               ExplicitGridMap)
from tensorkrowch.formats.rounding import RoundingInfo
from tensorkrowch.formats.tr import TR
from tensorkrowch.formats.trm import TRM
from tensorkrowch.formats.tt import TT
from tensorkrowch.formats.ttm import TTM
from tensorkrowch.formats.tucker import QTTTucker, QTRTucker


__all__ = [
    'TensorFormat',
    'TensorFormat1D',
    'TensorFormat2D',
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

    'BondFactors',
    'VidalGauge',
    'RoundingInfo',

    'BlockLayout',
    'UnblockInfo',
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
