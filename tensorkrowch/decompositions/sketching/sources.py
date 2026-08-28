"""Optional tensor-source capabilities used by sketching methods."""

from typing import (Protocol, Sequence, Union, runtime_checkable)

import torch

from tensorkrowch.decompositions.results import (TTDecomposition,
                                                 TTMDecomposition)
from tensorkrowch.decompositions.sources import TensorSource


TTStructuredSketch = Union[
    TTDecomposition,
    TTMDecomposition,
    Sequence[torch.Tensor],
]


@runtime_checkable
class SketchContractableSource(TensorSource, Protocol):
    """Optional source capable of contracting a TT-structured sketch."""

    def contract_sketch(
            self,
            sketch: TTStructuredSketch,
            conjugate_sketch: bool = True
            ) -> Union[torch.Tensor, TTDecomposition]:
        """Contracts all source input axes with ``sketch`` without a graph."""


__all__ = ['SketchContractableSource', 'TTStructuredSketch']
