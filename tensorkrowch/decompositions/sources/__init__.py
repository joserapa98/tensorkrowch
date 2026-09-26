"""
Tensor sources shared by ALS and recursive sketching.

        as_tensor_source(tensor / callable / TT / TensorSource)
            ├─ DenseTensorSource
            ├─ CallableTensorSource
            ├─ SparseTensorSource / EmpiricalDistribution
            └─ TTTensorSource

        ConfigurationBatch
            └─ packed indices or heterogeneous features ─> source.evaluate()

        FiberTensorSource
            └─ optional source.fiber() over one varying input site

        TensorSource
            ├─ ALSProblem ─> TTALS / TRALS
            └─ sketching Phi operators and evaluation sessions

    Sources supply values and runtime metadata. ALSProblem separately defines the
    observed or sampled objective. Sparse sources are zero outside their support;
    unobserved entries in completion remain unknown. All implementations operate
    on raw PyTorch tensors without constructing a TensorKrowch graph.

"""

from tensorkrowch.decompositions.sources.base import (ConfigurationBatch,
                                                      FiberTensorSource,
                                                      TensorSource)
from tensorkrowch.decompositions.sources.callable import CallableTensorSource
from tensorkrowch.decompositions.sources.dense import DenseTensorSource
from tensorkrowch.decompositions.sources.factory import as_tensor_source
from tensorkrowch.decompositions.sources.sparse import (EmpiricalDistribution,
                                                        SparseTensorSource)
from tensorkrowch.decompositions.sources.tt import TTTensorSource


__all__ = [
    'ConfigurationBatch',
    'TensorSource',
    'FiberTensorSource',
    'CallableTensorSource',
    'DenseTensorSource',
    'SparseTensorSource',
    'EmpiricalDistribution',
    'TTTensorSource',
    'as_tensor_source',
]
