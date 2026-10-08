"""
This script contains:

    Shared decomposition diagnostics:
        * TensorDecomposition
    Format subclasses with decomposition diagnostics:
        * TTDecomposition, TRDecomposition, TTMDecomposition, TRMDecomposition
        * QTTDecomposition, QTRDecomposition, QTTMDecomposition, QTRMDecomposition
    Deferred 2D results:
        * PEPSDecomposition, PEPODecomposition
"""

from dataclasses import dataclass, field
from typing import (Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Type,
                    TYPE_CHECKING, Union)

import torch

from tensorkrowch.formats import (TensorFormat, TensorFormat1D,
                                  TT, TR, TTM, TRM,
                                  QuantizedLayout,
                                  AffineCoordinateMap,
                                  QTT, QTR, QTTM, QTRM)
from tensorkrowch.formats.formats1d import _restore_cores

from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord)


if TYPE_CHECKING:
    from tensorkrowch.decompositions.sources.quantization import QuantizedSourceAdapter


@dataclass(init=False, eq=False)
class TensorDecomposition:  # MARK: TensorDecomposition
    """
    Historical fit diagnostics shared by decomposition results.

    Combine this base with a numerical format, as in :class:`TTDecomposition`
    or :class:`QTTDecomposition`. The format owns the cores and numerical
    operations; ``metrics`` and ``metadata`` describe how the result was
    obtained. Editing cores does not rewrite these historical records.
    Use :meth:`to_format` to work with the numerical format alone.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Cores following the public layout of the concrete format.
    metrics : DecompositionMetrics, optional
        Historical fit records; defaults to a new empty collection.
    metadata : dict, optional
        Algorithm configuration; defaults to a new empty dictionary.
    n_batches : int, optional
        Number of leading structural batch axes. Defaults to ``0``.
    **kwargs
        Additional constructor arguments of the concrete format.
    """

    metrics: DecompositionMetrics = field(default_factory=DecompositionMetrics)
    metadata: Dict[str, Any] = field(default_factory=dict)

    _format_type: ClassVar[Type[TensorFormat]]
    _format_parameters: ClassVar[Tuple[str, ...]] = ('n_batches',)

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

    def to_format(self) -> TensorFormat:
        """
        Returns the numerical format without decomposition diagnostics.

        Cores, bond factors, Vidal information, vector orientation and the
        orthogonality center are preserved. Quantics results also retain their
        layouts and coordinate maps. ``metrics`` and ``metadata`` are omitted.

        Core and bond containers are independent, but tensors are reused
        without cloning or detaching them. Replacing cores or bonds affects
        only the new format; editing shared tensor values affects both objects.
        Use :meth:`~tensorkrowch.formats.TensorFormat1D.clone` on the returned
        format when independent tensor storage is needed.

        Returns
        -------
        TensorFormat1D
            Corresponding ``TT``, ``TR``, ``TTM``, ``TRM`` or Quantics format.

        Examples
        --------
        >>> result = tk.decompositions.tt_svd(
        ...     torch.eye(2), rank=2, return_result=True)
        >>> format = result.to_format()
        >>> type(format)
        <class 'tensorkrowch.formats.formats1d.TT'>
        >>> format.contract_dense()
        tensor([[1., 0.],
                [0., 1.]])
        """
        kwargs = {name: getattr(self, name) for name in self._format_parameters}
        format = self._format_type(self.cores, **kwargs)
        if self._bonds is not None:
            format._bonds = self._bonds._map_tensors(
                lambda tensor: tensor, format._on_bonds_changed)

        # Preserve structural state that is not a constructor argument.
        for name in ('_orth_center', '_is_row'):
            if name in self.__dict__:
                setattr(format, name, getattr(self, name))
        return format


class TTDecomposition(TensorDecomposition, TT):  # MARK: TTDecomposition
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

    _format_type = TT


class TRDecomposition(TensorDecomposition, TR):  # MARK: TRDecomposition
    """
    :class:`~tensorkrowch.formats.TR` with decomposition diagnostics.

    Construction adds ``metrics`` and ``metadata`` as in
    :class:`TTDecomposition`; cores and numerical operations follow ``TR``.
    """

    _format_type = TR


class TTMDecomposition(TensorDecomposition, TTM):  # MARK: TTMDecomposition
    """
    :class:`~tensorkrowch.formats.TTM` with decomposition diagnostics.

    Construction adds ``metrics`` and ``metadata`` as in
    :class:`TTDecomposition`; cores and numerical operations follow ``TTM``.
    """

    _format_type = TTM

    def _new_from_standard_cores(self,
                                 cores: Sequence[torch.Tensor],
                                 in_dim: Optional[Sequence[int]],
                                 out_dim: Optional[Sequence[int]],
                                 n_batches: int,
                                 cyclic: bool,
                                 other: Optional['TensorFormat1D'] = None,
                                 product: bool = False) -> 'TensorFormat1D':
        """Preserves the result contract when applying an operator to data."""
        if product and other is None:
            cores = _restore_cores(cores, in_dim, out_dim, n_batches, cyclic)
            return TTDecomposition(
                cores, n_batches=n_batches,
                metadata={'operation': 'ttm_apply'})
        return super()._new_from_standard_cores(
            cores, in_dim, out_dim, n_batches, cyclic,
            other=other, product=product)


class TRMDecomposition(TensorDecomposition, TRM):  # MARK: TRMDecomposition
    """
    :class:`~tensorkrowch.formats.TRM` with decomposition diagnostics.

    Construction adds ``metrics`` and ``metadata`` as in
    :class:`TTDecomposition`; cores and numerical operations follow ``TRM``.
    """

    _format_type = TRM

    def _new_from_standard_cores(self,
                                 cores: Sequence[torch.Tensor],
                                 in_dim: Optional[Sequence[int]],
                                 out_dim: Optional[Sequence[int]],
                                 n_batches: int,
                                 cyclic: bool,
                                 other: Optional['TensorFormat1D'] = None,
                                 product: bool = False) -> 'TensorFormat1D':
        """Preserves the result contract when applying an operator to data."""
        if product and other is None:
            cores = _restore_cores(cores, in_dim, out_dim, n_batches, cyclic)
            return TRDecomposition(
                cores, n_batches=n_batches,
                metadata={'operation': 'trm_apply'})
        return super()._new_from_standard_cores(
            cores, in_dim, out_dim, n_batches, cyclic,
            other=other, product=product)


class QTTDecomposition(TensorDecomposition, QTT):  # MARK: QTTDecomposition
    """
    :class:`~tensorkrowch.formats.QTT` with decomposition diagnostics.

    Construction adds ``metrics`` and ``metadata`` as in
    :class:`TensorDecomposition`; coordinate arguments follow ``QTT``.
    :meth:`to_format` returns a ``QTT`` retaining its layout and coordinate map.
    """

    _format_type = QTT
    _format_parameters = ('n_batches', 'n_coordinates',
                          'layout', 'coordinate_map')


class QTRDecomposition(TensorDecomposition, QTR):  # MARK: QTRDecomposition
    """
    :class:`~tensorkrowch.formats.QTR` with decomposition diagnostics.

    Construction adds ``metrics`` and ``metadata`` as in
    :class:`TensorDecomposition`; coordinate arguments follow ``QTR``.
    :meth:`to_format` returns a ``QTR`` retaining its layout and coordinate map.
    """

    _format_type = QTR
    _format_parameters = ('n_batches', 'n_coordinates',
                          'layout', 'coordinate_map')


class QTTMDecomposition(TensorDecomposition, QTTM):  # MARK: QTTMDecomposition
    """
    :class:`~tensorkrowch.formats.QTTM` with decomposition diagnostics.

    Construction adds ``metrics`` and ``metadata`` as in
    :class:`TensorDecomposition`; coordinate arguments follow ``QTTM``.
    :meth:`to_format` retains both input and output coordinate spaces.
    """

    _format_type = QTTM
    _format_parameters = ('n_batches', 'in_n_coordinates', 'out_n_coordinates',
                          'in_layout', 'out_layout',
                          'in_coordinate_map', 'out_coordinate_map')


class QTRMDecomposition(TensorDecomposition, QTRM):  # MARK: QTRMDecomposition
    """
    :class:`~tensorkrowch.formats.QTRM` with decomposition diagnostics.

    Construction adds ``metrics`` and ``metadata`` as in
    :class:`TensorDecomposition`; coordinate arguments follow ``QTRM``.
    :meth:`to_format` retains both input and output coordinate spaces.
    """

    _format_type = QTRM
    _format_parameters = ('n_batches', 'in_n_coordinates', 'out_n_coordinates',
                          'in_layout', 'out_layout',
                          'in_coordinate_map', 'out_coordinate_map')


class PEPSDecomposition(TensorDecomposition, TensorFormat):  # MARK: PEPSDecomposition
    """Reserved PEPS result interface until its numerical format is available."""


class PEPODecomposition(TensorDecomposition, TensorFormat):  # MARK: PEPODecomposition
    """Reserved PEPO result interface until its numerical format is available."""


def _quantics_result(result: 'TensorDecomposition',
                     quantization: Optional[
                         Union[QuantizedLayout, Tuple[QuantizedLayout, QuantizedLayout]]] = None,
                     *,
                     adapter: Optional['QuantizedSourceAdapter'] = None) -> 'TensorDecomposition':
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
    TensorDecomposition,
]
