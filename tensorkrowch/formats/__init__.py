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
"""

from .base import TensorFormat, TensorFormat2D, SampleError
from ._chain import TensorFormat1D
from .tt import TensorTrain
from .tr import TensorRing
from .ttm import TensorTrainMatrix
from .trm import TensorRingMatrix

__all__ = ['TensorFormat', 'TensorFormat1D', 'TensorFormat2D', 'TensorTrain',
           'TensorRing', 'TensorTrainMatrix', 'TensorRingMatrix', 'SampleError']
