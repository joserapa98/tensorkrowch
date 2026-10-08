"""
This script contains:

    Metadata and diagnostics interfaces:
        * TensorDecomposition, TensorDecomposition1D, TensorDecomposition2D
    Format subclasses with decomposition diagnostics:
        * TTDecomposition, TRDecomposition, TTMDecomposition, TRMDecomposition
        * QTTDecomposition, QTRDecomposition, QTTMDecomposition, QTRMDecomposition
    Deferred 2D results:
        * PEPSDecomposition, PEPODecomposition
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (Any, Dict, List, Optional, Sequence, Tuple, Union,
                    TYPE_CHECKING)

import torch

from tensorkrowch.formats import (TensorFormat, TensorFormat1D,
                                  TT, TR, TTM, TRM,
                                  QuantizedLayout, CoordinateMap,
                                  AffineCoordinateMap,
                                  QTT, QTR, QTTM, QTRM)
from tensorkrowch.formats.formats1d import _restore_cores

from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord)


if TYPE_CHECKING:
    from tensorkrowch.decompositions.sources.quantization import QuantizedSourceAdapter


class TensorDecomposition(TensorFormat, ABC):  # MARK: TensorDecomposition
    """
    Numerical format together with historical fit diagnostics.

    Structural operations describe the current tensors. Metrics and metadata
    describe how the result was obtained; editing cores does not rewrite them.
    """

    @abstractmethod
    def as_info(self) -> Dict[str, Any]:
        """
        Returns decomposition metadata, diagnostics and current dimensions.
        """


class TensorDecomposition1D(TensorFormat1D, TensorDecomposition, ABC):  # MARK: TensorDecomposition1D
    """One-dimensional formats carrying decomposition metadata and diagnostics."""


class TensorDecomposition2D(TensorDecomposition, ABC):  # MARK: TensorDecomposition2D
    """Reserved metadata and diagnostics interface for future 2D results."""


@dataclass(init=False)
class _ResultState:  # MARK: _ResultState
    """Stores decomposition metadata and diagnostics; formats hold the tensors."""

    cores: Sequence[torch.Tensor] = field()  # Numerical format cores
    # Historical fit records
    metrics: DecompositionMetrics = field(default_factory=DecompositionMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Small algorithm configuration
    n_batches: int = field()  # Leading structural batch axes

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 metrics: Optional[DecompositionMetrics] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 n_batches: int = 0,
                 **kwargs) -> None:
        self.metrics = DecompositionMetrics() if metrics is None else metrics
        self.metadata = {} if metadata is None else metadata
        if not isinstance(self.metrics, DecompositionMetrics):
            raise TypeError('`metrics` should be DecompositionMetrics type')
        if not isinstance(self.metadata, dict):
            raise TypeError('`metadata` should be dict type')
        super().__init__(cores, n_batches=n_batches, **kwargs)

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Compatibility alias for ``in_dim``."""
        return self.in_dim

    @property
    def output_dim(self) -> Optional[Tuple[int, ...]]:
        """Compatibility alias for ``out_dim``."""
        return self.out_dim

    def as_info(self) -> Dict[str, Any]:
        """
        Returns current dimensions and historical decomposition diagnostics.

        ``rank``, ``in_dim`` and ``out_dim`` describe the current cores. Metrics
        describe the original fit, even after editing or rounding the result.
        Tensor-valued diagnostics are detached and moved to CPU by
        :meth:`~tensorkrowch.decompositions.DecompositionMetrics.as_info`.

        Returns
        -------
        dict
            Structure, a shallow copy of ``metadata`` and fit diagnostics.
        """
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

    def error(self, *args, **kwargs) -> ErrorRecord:
        """
        Measures sample error with the inherited format's evaluation rules.

        Arguments follow :meth:`~tensorkrowch.formats.TensorFormat1D.error`.
        This measures the current cores and returns a detached CPU
        :class:`~tensorkrowch.decompositions.ErrorRecord`; it does not change
        the historical records in ``metrics``.
        """
        record = super().error(*args, **kwargs)
        return ErrorRecord(kind=record.kind, absolute=record.absolute,
                           relative=record.relative, size=record.size,
                           denominator=record.denominator)

    def _new_from_standard_cores(self,
                                 cores: Sequence[torch.Tensor],
                                 in_dim: Optional[Sequence[int]],
                                 out_dim: Optional[Sequence[int]],
                                 n_batches: int,
                                 cyclic: bool,
                                 other: Optional['TensorFormat1D'] = None,
                                 product: bool = False) -> 'TensorFormat1D':
        """Preserves the result contract when applying an operator to data."""
        if self._family == 'matrix' and product and other is None:
            cls = TRDecomposition if cyclic else TTDecomposition
            cores = _restore_cores(cores, in_dim, out_dim, n_batches, cyclic)
            return cls(cores, n_batches=n_batches, metadata={
                'operation': 'trm_apply' if cyclic else 'ttm_apply'})
        return super()._new_from_standard_cores(
            cores, in_dim, out_dim, n_batches, cyclic, other=other,
            product=product)


class TTDecomposition(_ResultState, TT, TensorDecomposition1D):  # MARK: TTDecomposition
    """
    :class:`~tensorkrowch.formats.TT` with decomposition diagnostics.

    The numerical API is inherited from ``TT``. ``metrics`` and ``metadata``
    describe the fit that produced the cores and remain historical when the
    result is modified. Arithmetic produces ordinary formats.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        TT cores following the public ``TT`` layout.
    metrics : DecompositionMetrics, optional
        Historical fit records; a new empty collection is created by default.
    metadata : dict, optional
        Algorithm configuration; defaults to a new empty dictionary.
    n_batches : int, optional
        Number of leading batch axes in each core. Defaults to ``0``.
    **kwargs
        Additional arguments of the underlying format, such as ``bonds``.

    Examples
    --------
    >>> result = tk.decompositions.tt_svd(
    ...     torch.eye(2), rank=2, return_result=True)
    >>> result.contract_dense()
    tensor([[1., 0.],
            [0., 1.]])
    >>> result.as_info()['rank']
    [2]
    """


class TRDecomposition(_ResultState, TR, TensorDecomposition1D):  # MARK: TRDecomposition
    """
    :class:`~tensorkrowch.formats.TR` with decomposition diagnostics.

    Construction adds ``metrics`` and ``metadata`` as in
    :class:`TTDecomposition`; cores and numerical operations follow ``TR``.
    """


class TTMDecomposition(_ResultState, TTM, TensorDecomposition1D):  # MARK: TTMDecomposition
    """
    :class:`~tensorkrowch.formats.TTM` with decomposition diagnostics.

    Construction adds ``metrics`` and ``metadata`` as in
    :class:`TTDecomposition`; cores and numerical operations follow ``TTM``.
    """


class TRMDecomposition(_ResultState, TRM, TensorDecomposition1D):  # MARK: TRMDecomposition
    """
    :class:`~tensorkrowch.formats.TRM` with decomposition diagnostics.

    Construction adds ``metrics`` and ``metadata`` as in
    :class:`TTDecomposition`; cores and numerical operations follow ``TRM``.
    """


class _QuanticsResultState:  # MARK: _QuanticsResultState
    """
    Adds decomposition metadata and diagnostics after initializing the cores.
    """

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 metrics: Optional['DecompositionMetrics'] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 n_batches: int = 0,
                 **kwargs) -> None:
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
class QTTDecomposition(_QuanticsResultState, QTT, TTDecomposition):  # MARK: QTTDecomposition
    """:class:`~tensorkrowch.formats.QTT` retaining digit and coordinate metadata."""

    n_coordinates: int = field()  # Number of original input coordinates
    layout: QuantizedLayout = field()  # Input coordinate-to-digit schedule
    coordinate_map: CoordinateMap = field()  # Domain and grid conversions


@dataclass(init=False)
class QTRDecomposition(_QuanticsResultState, QTR, TRDecomposition):  # MARK: QTRDecomposition
    """:class:`~tensorkrowch.formats.QTR` retaining digit and coordinate metadata."""

    n_coordinates: int = field()  # Number of original input coordinates
    layout: QuantizedLayout = field()  # Input coordinate-to-digit schedule
    coordinate_map: CoordinateMap = field()  # Domain and grid conversions


@dataclass(init=False)
class QTTMDecomposition(_QuanticsResultState, QTTM,
                       TTMDecomposition):
    """Quantics TTM with separate input and output coordinate spaces."""

    in_n_coordinates: int = field()  # Number of original input coordinates
    out_n_coordinates: int = field()  # Number of original output coordinates
    in_layout: QuantizedLayout = field()  # Input digit schedule
    out_layout: QuantizedLayout = field()  # Output digit schedule
    in_coordinate_map: CoordinateMap = field()  # Actual input-coordinate map
    out_coordinate_map: CoordinateMap = field()  # Actual output-coordinate map


@dataclass(init=False)
class QTRMDecomposition(_QuanticsResultState, QTRM,
                       TRMDecomposition):
    """Quantics TRM with separate input and output coordinate spaces."""

    in_n_coordinates: int = field()  # Number of original input coordinates
    out_n_coordinates: int = field()  # Number of original output coordinates
    in_layout: QuantizedLayout = field()  # Input digit schedule
    out_layout: QuantizedLayout = field()  # Output digit schedule
    in_coordinate_map: CoordinateMap = field()  # Actual input-coordinate map
    out_coordinate_map: CoordinateMap = field()  # Actual output-coordinate map


class PEPSDecomposition(TensorDecomposition2D):  # MARK: PEPSDecomposition
    """Reserved PEPS result interface."""


class PEPODecomposition(TensorDecomposition2D):  # MARK: PEPODecomposition
    """Reserved PEPO result interface."""


def _quantics_result(result: 'TensorDecomposition1D',
                     quantization: Optional[
                         Union[QuantizedLayout, Tuple[QuantizedLayout, QuantizedLayout]]] = None,
                     *,
                     adapter: Optional['QuantizedSourceAdapter'] = None) -> 'TensorDecomposition1D':
    """
    Attaches coordinate meaning to fitted cores without numerical refitting.
    """
    if quantization is None:
        return result
    kwargs = dict(metrics=result.metrics, metadata=result.metadata,
                  n_batches=result.n_batches)
    unit_domain = torch.tensor([0., 1.], device=result.device,
                               dtype=result.cores[0].real.dtype)
    if isinstance(quantization, tuple):
        cls = QTRMDecomposition if result.topology == 'trm' else QTTMDecomposition
        kwargs.update(in_n_coordinates=quantization[0].n_coordinates,
                      out_n_coordinates=quantization[1].n_coordinates,
                      in_layout=quantization[0], out_layout=quantization[1],
                      in_coordinate_map=AffineCoordinateMap(
                          unit_domain, quantization[0].grid_size),
                      out_coordinate_map=AffineCoordinateMap(
                          unit_domain, quantization[1].grid_size))
    else:
        cls = QTRDecomposition if result.topology == 'tr' else QTTDecomposition
        kwargs.update(n_coordinates=quantization.n_coordinates,
                      layout=quantization)
        kwargs['coordinate_map'] = (adapter.coordinate_map
                                    if adapter is not None else
                                    AffineCoordinateMap(
                                        unit_domain, quantization.grid_size))
    wrapped = cls(result.cores, **kwargs)
    wrapped.metadata = dict(result.metadata)
    if 'quantization' not in wrapped.metadata:
        layouts = quantization if isinstance(quantization, tuple) else (quantization,)
        wrapped.metadata['quantization'] = [
            {'base': layout.base, 'level': layout.level,
             'sites': layout.sites(), 'grid_size': layout.grid_size}
            for layout in layouts]
    return wrapped.to(device=result.device)


_DecompositionOutput = Union[
    List[torch.Tensor],
    Tuple[List[torch.Tensor], Dict[str, Any]],
    TensorDecomposition1D,
]
