"""
Sources shared by tensor decomposition algorithms.

    Input objects:
        * DenseTensorSource, SparseTensorSource, FormatSource
        * CallableSource, CallableTensorSource, CallableQuanticsSource
        * BaseSource, ConfigurationBatch
        * gaussian_kernel, as_tensor_source

    Evaluation:
        ConfigurationBatch -> source.evaluate() -> values
        Finite source -> source.contract_dense() -> tensor
        CallableQuanticsSource:
            digit indices -> QuantizedLayout -> grid indices
            grid indices -> CoordinateMap -> domain coordinates -> callable

    Algorithms:
        Finite sources -> SVD / ALS / RS
        Continuous or finite sources + samples -> RSS
        Source matrix/Quantics metadata -> corresponding result format

Evaluation configurations contain ``len(source.in_sites)`` inputs in that
order; ``source.n_sites`` counts all represented variables. Matrix sources
evaluate one input per paired site and return every paired output axis.

Dataset constructors reuse the same evaluation methods. Sparse values outside
stored indices are zero. In ALS a separate boolean mask determines participating
entries; ``True`` participates, including entries whose target value is zero.
"""

from tensorkrowch.decompositions.sources.base import (BaseSource,
                                                      ConfigurationBatch)
from tensorkrowch.decompositions.sources.callable import (CallableSource,
                                                          CallableTensorSource)
from tensorkrowch.decompositions.sources.dense import DenseTensorSource
from tensorkrowch.decompositions.sources.factory import as_tensor_source
from tensorkrowch.decompositions.sources.format import FormatSource
from tensorkrowch.decompositions.sources.kernels import gaussian_kernel
from tensorkrowch.decompositions.sources.quantics import CallableQuanticsSource
from tensorkrowch.decompositions.sources.sparse import SparseTensorSource

__all__ = [
    'ConfigurationBatch',
    'BaseSource',
    'CallableSource',
    'CallableTensorSource',
    'CallableQuanticsSource',
    'DenseTensorSource',
    'SparseTensorSource',
    'FormatSource',
    'as_tensor_source',
    'gaussian_kernel',
]
