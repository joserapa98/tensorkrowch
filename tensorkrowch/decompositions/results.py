"""
This script contains:

    Provenance interfaces:
        * TensorDecomposition, TensorDecomposition1D, TensorDecomposition2D
    Format subclasses with decomposition diagnostics:
        * TTDecomposition, TRDecomposition, TTMDecomposition, TRMDecomposition
        * QTTDecomposition, QTRDecomposition, QTTMDecomposition, QTRMDecomposition
        * QTTTuckerDecomposition, QTRTuckerDecomposition
    Deferred 2D results:
        * PEPSDecomposition, PEPODecomposition
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from tensorkrowch.formats import (
    TensorFormat, TensorFormat1D, TT, TR, TTM, TRM,
    QTT, QTR, QTTM, QTRM, QTTTucker, QTRTucker, QuantizedLayout)
from tensorkrowch.formats.formats1d import _restore_cores
from tensorkrowch.decompositions.metrics import DecompositionMetrics, ErrorRecord


class TensorDecomposition(TensorFormat, ABC):
    """Numerical format together with historical fit diagnostics.

    Structural operations describe the current tensors. Metrics and metadata
    describe how the result was obtained; editing cores does not rewrite them.
    """

    @abstractmethod
    def as_info(self) -> Dict[str, Any]:
        """Returns decomposition provenance and current structural dimensions."""


class TensorDecomposition1D(TensorFormat1D, TensorDecomposition, ABC):
    """Compatibility interface for raw-core 1D decomposition results."""


class TensorDecomposition2D(TensorDecomposition, ABC):
    """Reserved provenance interface for future 2D results."""


@dataclass(init=False)
class _ResultState:
    """Owns provenance; numerical representation is inherited from formats."""

    cores: Sequence[torch.Tensor] = field()  # Numerical format cores
    metrics: DecompositionMetrics = field(default_factory=DecompositionMetrics)  # Historical fit records
    metadata: Dict[str, Any] = field(default_factory=dict)  # Small algorithm configuration
    n_batches: int = field()  # Leading structural batch axes

    def __init__(self, cores: Sequence[torch.Tensor],
                 metrics: Optional[DecompositionMetrics] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 n_batches: int = 0, **kwargs: Any) -> None:
        self.metrics = DecompositionMetrics() if metrics is None else metrics
        self.metadata = {} if metadata is None else metadata
        if not isinstance(self.metrics, DecompositionMetrics):
            raise TypeError('`metrics` should be DecompositionMetrics type')
        if not isinstance(self.metadata, dict):
            raise TypeError('`metadata` should be dict type')
        super().__init__(cores, n_batches=n_batches, **kwargs)

    @property
    def input_dim(self):
        """Compatibility alias for in_dim."""
        return self.in_dim

    @property
    def output_dim(self):
        """Compatibility alias for out_dim."""
        return self.out_dim

    def as_info(self) -> Dict[str, Any]:
        """Returns small structural metadata and detached fit diagnostics."""
        return {
            'topology': self.topology,
            'rank': self.rank,
            'in_dim': list(self.in_dim),
            'out_dim': None if self.out_dim is None else list(self.out_dim),
            'input_dim': list(self.in_dim),
            'output_dim': None if self.out_dim is None else list(self.out_dim),
            'n_batches': self.n_batches,
            'metrics': self.metrics.as_info(),
            'metadata': dict(self.metadata),
        }

    def error(self, *args: Any, **kwargs: Any) -> ErrorRecord:
        """Collects a sample error using the shared numerical format kernel."""
        record = super().error(*args, **kwargs)
        return ErrorRecord(kind=record.kind, absolute=record.absolute,
                           relative=record.relative, size=record.size,
                           denominator=record.denominator)

    def _new_from_standard_cores(self, cores, in_dim, out_dim, n_batches,
                                 cyclic, other=None, product=False):
        """Preserves the result contract when applying an operator to data."""
        if self._family == 'matrix' and product and other is None:
            cls = TRDecomposition if cyclic else TTDecomposition
            cores = _restore_cores(cores, in_dim, out_dim, n_batches, cyclic)
            return cls(cores, n_batches=n_batches, metadata={
                'operation': 'trm_apply' if cyclic else 'ttm_apply'})
        return super()._new_from_standard_cores(
            cores, in_dim, out_dim, n_batches, cyclic, other=other,
            product=product)


class TTDecomposition(_ResultState, TT, TensorDecomposition1D):
    """TT format with decomposition metrics and algorithm metadata."""


class TRDecomposition(_ResultState, TR, TensorDecomposition1D):
    """TR format with decomposition metrics and algorithm metadata."""


class TTMDecomposition(_ResultState, TTM, TensorDecomposition1D):
    """TTM format with decomposition metrics and algorithm metadata."""


class TRMDecomposition(_ResultState, TRM, TensorDecomposition1D):
    """TRM format with decomposition metrics and algorithm metadata."""


class _QuanticsResultState:
    """Adds provenance after the coordinate-aware constructor initializes cores."""

    def __init__(self, cores, metrics=None, metadata=None, n_batches=0, **kwargs):
        super().__init__(cores, n_batches=n_batches, **kwargs)
        if metrics is not None:
            if not isinstance(metrics, DecompositionMetrics):
                raise TypeError('`metrics` should be DecompositionMetrics type')
            self.metrics = metrics
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError('`metadata` should be dict type')
            self.metadata = metadata


@dataclass(init=False)
class QTTDecomposition(_QuanticsResultState, QTT, TTDecomposition):
    """Quantics TT retaining its actual digit layout and coordinate map."""

    layout: QuantizedLayout = field()  # Logical variable-to-digit schedule
    coordinate_map: Any = None  # Actual physical-coordinate map
    domain: Any = None  # Physical variable domains
    digit_positions: Sequence[int] = field()  # Network sites carrying digits
    computational_grid: str = 'endpoints'  # Grid-node convention
    out_of_domain: str = 'error'  # Explicit coordinate boundary policy


@dataclass(init=False)
class QTRDecomposition(_QuanticsResultState, QTR, TRDecomposition):
    """Quantics TR retaining its actual digit layout and coordinate map."""

    layout: QuantizedLayout = field()  # Logical variable-to-digit schedule
    coordinate_map: Any = None  # Actual physical-coordinate map
    domain: Any = None  # Physical variable domains
    digit_positions: Sequence[int] = field()  # Network sites carrying digits
    computational_grid: str = 'endpoints'  # Grid-node convention
    out_of_domain: str = 'error'  # Explicit coordinate boundary policy


@dataclass(init=False)
class QTTMDecomposition(_QuanticsResultState, QTTM,
                       TTMDecomposition):
    """Quantics TTM with separate input and output coordinate spaces."""

    in_layout: QuantizedLayout = field()  # Input digit schedule
    out_layout: QuantizedLayout = field()  # Output digit schedule
    in_coordinate_map: Any = None  # Actual input-coordinate map
    out_coordinate_map: Any = None  # Actual output-coordinate map
    in_domain: Any = None  # Physical input domains
    out_domain: Any = None  # Physical output domains
    computational_grid: str = 'endpoints'  # Grid-node convention
    out_of_domain: str = 'error'  # Explicit coordinate boundary policy


@dataclass(init=False)
class QTRMDecomposition(_QuanticsResultState, QTRM,
                       TRMDecomposition):
    """Quantics TRM with separate input and output coordinate spaces."""

    in_layout: QuantizedLayout = field()  # Input digit schedule
    out_layout: QuantizedLayout = field()  # Output digit schedule
    in_coordinate_map: Any = None  # Actual input-coordinate map
    out_coordinate_map: Any = None  # Actual output-coordinate map
    in_domain: Any = None  # Physical input domains
    out_domain: Any = None  # Physical output domains
    computational_grid: str = 'endpoints'  # Grid-node convention
    out_of_domain: str = 'error'  # Explicit coordinate boundary policy


class _TuckerResultState(TensorDecomposition):
    """Provenance for a hierarchical format with a single upper owner."""

    def __init__(self, upper, factors, layout, coordinate_map, domain=None, *,
                 variable_positions=None, computational_grid='endpoints',
                 out_of_domain='error', metrics=None, metadata=None):
        self.metrics = upper.metrics if metrics is None and isinstance(
            upper, TensorDecomposition) else (
                DecompositionMetrics() if metrics is None else metrics)
        self.metadata = {} if metadata is None else metadata
        if not isinstance(self.metrics, DecompositionMetrics):
            raise TypeError('`metrics` should be DecompositionMetrics type')
        if not isinstance(self.metadata, dict):
            raise TypeError('`metadata` should be dict type')
        super().__init__(upper, factors, layout, coordinate_map, domain,
                         variable_positions=variable_positions,
                         computational_grid=computational_grid,
                         out_of_domain=out_of_domain)

    input_dim = _ResultState.input_dim
    output_dim = _ResultState.output_dim

    def as_info(self) -> Dict[str, Any]:
        info = _ResultState.as_info(self)
        info.update(upper_rank=self.upper.rank,
                    factor_rank=[list(rank) for rank in self.factor_rank],
                    grid_size=list(self.layout.grid_size),
                    variable_positions=list(self.variable_positions),
                    out_shape=list(self.out_shape))
        return info

    def _map_tensors(self, function):
        result = super()._map_tensors(function)
        result.metrics = self.metrics
        result.metadata = self.metadata
        return result

    def to(self, device=None, dtype=None, copy=False):
        result = super().to(device=device, dtype=dtype, copy=copy)
        result.metrics = self.metrics
        result.metadata = self.metadata
        return result


class QTTTuckerDecomposition(_TuckerResultState, QTTTucker):
    """QTT-Tucker format together with fit diagnostics."""


class QTRTuckerDecomposition(_TuckerResultState, QTRTucker):
    """QTR-Tucker format together with fit diagnostics."""


# Hierarchical formats satisfy the public 1D provenance interface without
# inheriting the flat core-container implementation a second time.
TensorDecomposition1D.register(QTTTuckerDecomposition)
TensorDecomposition1D.register(QTRTuckerDecomposition)


class PEPSDecomposition(TensorDecomposition2D):
    """Reserved PEPS result interface."""


class PEPODecomposition(TensorDecomposition2D):
    """Reserved PEPO result interface."""


def _quantics_result(result, quantization=None, *, adapter=None,
                     digit_positions=None):
    """Attaches coordinate meaning to fitted cores without numerical refitting."""
    if quantization is None:
        return result
    kwargs = dict(metrics=result.metrics, metadata=result.metadata,
                  n_batches=result.n_batches)
    if isinstance(quantization, tuple):
        cls = QTRMDecomposition if result.topology == 'trm' else QTTMDecomposition
        kwargs.update(in_layout=quantization[0], out_layout=quantization[1])
    else:
        cls = QTRDecomposition if result.topology == 'tr' else QTTDecomposition
        kwargs.update(layout=quantization, digit_positions=digit_positions)
        if adapter is not None:
            kwargs.update(coordinate_map=adapter.coordinate_map,
                          domain=adapter.domain,
                          computational_grid=adapter.computational_grid,
                          out_of_domain=adapter.out_of_domain)
    wrapped = cls(result.cores, **kwargs)
    wrapped.metadata = dict(result.metadata)
    if 'quantization' not in wrapped.metadata:
        layouts = quantization if isinstance(quantization, tuple) else (quantization,)
        wrapped.metadata['quantization'] = [
            {'base': layout.base, 'level': layout.level,
             'sites': layout.sites(), 'grid_size': layout.grid_size}
            for layout in layouts]
    return wrapped.to(device=result.device)
