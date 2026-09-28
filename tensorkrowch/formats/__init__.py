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
    * TensorTrain, TensorRing
    * TensorTrainMatrix, TensorRingMatrix
    * SampleError
    * BondFactors, VidalGauge, RoundingInfo
    * GaugeOrbit, TensorRingOrbit
    * BlockLayout, UnblockInfo, SplitBlock, split_block
    * QuantizedLayout, CoordinateMap and coordinate maps
    * QuanticsTensorTrain, QuanticsTensorRing
    * QuanticsTensorTrainMatrix, QuanticsTensorRingMatrix
    * QTTTucker, QTRTucker
"""

from .base import TensorFormat, TensorFormat2D, SampleError
from ._chain import TensorFormat1D
from .tt import TensorTrain
from .tr import TensorRing
from .ttm import TensorTrainMatrix
from .trm import TensorRingMatrix
from .bonds import BondFactors, VidalGauge
from .rounding import RoundingInfo
from .orbits import GaugeOrbit, TensorRingOrbit
from .blocking import BlockLayout, UnblockInfo, SplitBlock, split_block
from .quantization import (QuantizedLayout, CoordinateMap, UniformCoordinateMap,
                           WarpedCoordinateMap, ExplicitGridMap)
from .quantics import (QuanticsTensorTrain, QuanticsTensorRing,
                       QuanticsTensorTrainMatrix, QuanticsTensorRingMatrix)
from .tucker import QTTTucker, QTRTucker

__all__ = ['TensorFormat', 'TensorFormat1D', 'TensorFormat2D', 'TensorTrain',
           'TensorRing', 'TensorTrainMatrix', 'TensorRingMatrix', 'SampleError',
           'BondFactors', 'VidalGauge', 'RoundingInfo', 'GaugeOrbit',
           'TensorRingOrbit', 'QuantizedLayout', 'CoordinateMap',
           'UniformCoordinateMap', 'WarpedCoordinateMap', 'ExplicitGridMap',
           'BlockLayout', 'UnblockInfo', 'SplitBlock', 'split_block',
           'QuanticsTensorTrain', 'QuanticsTensorRing',
           'QuanticsTensorTrainMatrix', 'QuanticsTensorRingMatrix',
           'QTTTucker', 'QTRTucker']
