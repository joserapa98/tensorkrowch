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

from .base import TensorFormat, TensorFormat2D, SampleError
from ._chain import TensorFormat1D
from .tt import TT
from .tr import TR
from .ttm import TTM
from .trm import TRM
from .bonds import BondFactors, VidalGauge
from .rounding import RoundingInfo
from .orbits import GaugeOrbit, TensorRingOrbit, MinimalCanonicalInfo
from .blocking import BlockLayout, UnblockInfo, SplitBlock, split_block
from .quantization import (QuantizedLayout, CoordinateMap, UniformCoordinateMap,
                           WarpedCoordinateMap, ExplicitGridMap)
from .quantics import QTT, QTR, QTTM, QTRM
from .tucker import QTTTucker, QTRTucker

__all__ = ['TensorFormat', 'TensorFormat1D', 'TensorFormat2D', 'TT',
           'TR', 'TTM', 'TRM', 'SampleError',
           'BondFactors', 'VidalGauge', 'RoundingInfo', 'GaugeOrbit',
           'TensorRingOrbit', 'MinimalCanonicalInfo', 'QuantizedLayout', 'CoordinateMap',
           'UniformCoordinateMap', 'WarpedCoordinateMap', 'ExplicitGridMap',
           'BlockLayout', 'UnblockInfo', 'SplitBlock', 'split_block',
           'QTT', 'QTR', 'QTTM', 'QTRM', 'QTTTucker', 'QTRTucker']
