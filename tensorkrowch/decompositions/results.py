"""
This script contains:

    Shared decomposition diagnostics:
        * TensorDecomposition

    Format subclasses with decomposition diagnostics:
        * TTDecomposition, TRDecomposition, TTMDecomposition, TRMDecomposition
        * QTTDecomposition, QTRDecomposition, QTTMDecomposition, QTRMDecomposition
"""

from dataclasses import dataclass, field
from typing import (Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Type,
                    TYPE_CHECKING, Union)

import torch

from tensorkrowch.formats import (TensorFormat, TensorFormat1D,
                                  TT, TR, TTM, TRM,
                                  QTT, QTR, QTTM, QTRM)
from tensorkrowch.formats.formats1d import _restore_cores

from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord)


if TYPE_CHECKING:
    from tensorkrowch.decompositions.sources.quantics import (QuanticsMatrixSource,
                                                              QuanticsVectorSource)


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

    Examples
    --------
    Construct a concrete result combining this base with its numerical format:

    >>> result = tk.decompositions.TTDecomposition(
    ...     [torch.tensor([1., 2.])], metadata={'algorithm': 'manual'})
    >>> result.contract_dense()
    tensor([1., 2.])
    >>> result.as_info()['metadata']
    {'algorithm': 'manual'}
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

        Examples
        --------
        >>> result = tk.decompositions.TTDecomposition(
        ...     [torch.ones(2, 1), torch.ones(1, 3)],
        ...     metadata={'algorithm': 'manual'})
        >>> info = result.as_info()
        >>> info['in_dim'], info['rank']
        ([2, 3], [1])
        >>> info['metadata']
        {'algorithm': 'manual'}
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
        Measures sample error for a vector decomposition result.

        Evaluation follows :meth:`~tensorkrowch.formats.TT.error`, using the
        current cores. The returned record is detached and moved to CPU;
        historical records in ``metrics`` are not modified.

        Parameters
        ----------
        function : callable
            Target function receiving ``samples`` and returning one value per
            data configuration, as a ``torch.Tensor``.
        samples : torch.Tensor or sequence of torch.Tensor
            Inputs to ``function``, with ``n_batches`` leading data batch axes.
            A tensor groups sites on the next axis; a sequence contains one
            tensor per site. Quantics results require scalar coordinate or
            index inputs, without feature axes.
        data : torch.Tensor or sequence of torch.Tensor, optional
            Inputs used to evaluate the format. Defaults to ``samples``.
            Supply embedded inputs for generic vectors, or encoded digits
            when Quantics target samples are domain coordinates or grid indices.
        n_batches : int, optional
            Number of leading data batch axes, independent of the structural
            batches stored in the cores. Defaults to ``1``.
        **kwargs
            Additional keyword arguments passed to ``function``.

        Returns
        -------
        ErrorRecord
            Absolute error, relative error, sample count and target norm,
            detached and stored on CPU. Errors include all samples and
            structural batches. The relative error divides the absolute error
            by the target norm. It is zero when both norms vanish and infinity
            when only the target norm vanishes.

        Examples
        --------
        >>> result = tk.decompositions.TTDecomposition(
        ...     [torch.tensor([1., 2.])])
        >>> samples = torch.tensor([[0], [1]])
        >>> def function(indices):
        ...     return indices[:, 0].float() + 1
        >>> record = result.error(function, samples)
        >>> record.absolute.item(), record.relative.item()
        (0.0, 0.0)
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
        ...     torch.eye(2), rank=2)
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
    result is modified. Use :meth:`~TensorDecomposition.to_format` to obtain
    the corresponding ``TT`` without decomposition diagnostics.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors following the public
        :class:`~tensorkrowch.formats.TT` layout, with open boundaries.
        The container is copied; tensors retain storage and autograd.
    metrics : DecompositionMetrics, optional
        Historical fit records; defaults to a new empty collection.
    metadata : dict, optional
        Algorithm configuration; defaults to a new empty dictionary.
    n_batches : int, optional
        Number of leading structural batch axes in each core, independent of
        data batches during evaluation. Defaults to ``0``.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores. Defaults to ``None``. Core and factor
        containers are copied; tensors are reused without cloning or detaching.

    Examples
    --------
    >>> cores = [torch.tensor([[1.], [2.]]), torch.tensor([[3., 4.]])]
    >>> result = tk.decompositions.TTDecomposition(
    ...     cores, metadata={'algorithm': 'manual'})
    >>> result.contract_dense()
    tensor([[3., 4.],
            [6., 8.]])
    >>> result.as_info()['rank']
    [1]
    """

    _format_type = TT


class TRDecomposition(TensorDecomposition, TR):  # MARK: TRDecomposition
    """
    :class:`~tensorkrowch.formats.TR` with decomposition diagnostics.

    The numerical API is inherited from ``TR``. ``metrics`` and ``metadata``
    describe the fit that produced the cores and remain historical when the
    result is modified. Use :meth:`~TensorDecomposition.to_format` to obtain
    the corresponding ``TR`` without decomposition diagnostics.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors following the public
        :class:`~tensorkrowch.formats.TR` layout, with a cyclic closure.
        The container is copied; tensors retain storage and autograd.
    metrics : DecompositionMetrics, optional
        Historical fit records; defaults to a new empty collection.
    metadata : dict, optional
        Algorithm configuration; defaults to a new empty dictionary.
    n_batches : int, optional
        Number of leading structural batch axes in each core, independent of
        data batches during evaluation. Defaults to ``0``.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores. Defaults to ``None``. Core and factor
        containers are copied; tensors are reused without cloning or detaching.

    Examples
    --------
    >>> cores = [torch.tensor([1., 2.]).reshape(1, 2, 1),
    ...          torch.tensor([3., 4.]).reshape(1, 2, 1)]
    >>> result = tk.decompositions.TRDecomposition(
    ...     cores, metadata={'algorithm': 'manual'})
    >>> result.contract_dense()
    tensor([[3., 4.],
            [6., 8.]])
    >>> result.as_info()['rank']
    [1, 1]
    """

    _format_type = TR


class TTMDecomposition(TensorDecomposition, TTM):  # MARK: TTMDecomposition
    """
    :class:`~tensorkrowch.formats.TTM` with decomposition diagnostics.

    The numerical API is inherited from ``TTM``. ``metrics`` and ``metadata``
    describe the fit that produced the cores and remain historical when the
    result is modified. Use :meth:`~TensorDecomposition.to_format` to obtain
    the corresponding ``TTM`` without decomposition diagnostics.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors following the public
        :class:`~tensorkrowch.formats.TTM` layout, with open boundaries.
        The container is copied; tensors retain storage and autograd.
    metrics : DecompositionMetrics, optional
        Historical fit records; defaults to a new empty collection.
    metadata : dict, optional
        Algorithm configuration; defaults to a new empty dictionary.
    n_batches : int, optional
        Number of leading structural batch axes in each core, independent of
        data batches during evaluation. Defaults to ``0``.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores. Defaults to ``None``. Core and factor
        containers are copied; tensors are reused without cloning or detaching.

    Examples
    --------
    >>> cores = [torch.diag(torch.tensor([2., 3.]))]
    >>> result = tk.decompositions.TTMDecomposition(
    ...     cores, metadata={'algorithm': 'manual'})
    >>> result.contract_dense()
    tensor([[2., 0.],
            [0., 3.]])
    >>> indices = torch.tensor([[0], [1]])
    >>> result.evaluate(indices, indices).tolist()
    [2.0, 3.0]
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

    The numerical API is inherited from ``TRM``. ``metrics`` and ``metadata``
    describe the fit that produced the cores and remain historical when the
    result is modified. Use :meth:`~TensorDecomposition.to_format` to obtain
    the corresponding ``TRM`` without decomposition diagnostics.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors following the public
        :class:`~tensorkrowch.formats.TRM` layout, with a cyclic closure.
        The container is copied; tensors retain storage and autograd.
    metrics : DecompositionMetrics, optional
        Historical fit records; defaults to a new empty collection.
    metadata : dict, optional
        Algorithm configuration; defaults to a new empty dictionary.
    n_batches : int, optional
        Number of leading structural batch axes in each core, independent of
        data batches during evaluation. Defaults to ``0``.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores. Defaults to ``None``. Core and factor
        containers are copied; tensors are reused without cloning or detaching.

    Examples
    --------
    >>> cores = [torch.diag(torch.tensor([2., 3.])).reshape(1, 2, 1, 2)]
    >>> result = tk.decompositions.TRMDecomposition(
    ...     cores, metadata={'algorithm': 'manual'})
    >>> result.contract_dense()
    tensor([[2., 0.],
            [0., 3.]])
    >>> indices = torch.tensor([[0], [1]])
    >>> result.evaluate(indices, indices).tolist()
    [2.0, 3.0]
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

    The numerical API is inherited from ``QTT``. ``metrics`` and ``metadata``
    describe the fit that produced the cores and remain historical when the
    result is modified. Use :meth:`~TensorDecomposition.to_format` to obtain
    the corresponding ``QTT`` without decomposition diagnostics.

    Coordinate evaluation follows the stored
    :class:`~tensorkrowch.formats.QuantizedLayout` and
    :class:`~tensorkrowch.formats.CoordinateMap`. Shorthand construction uses
    ``interleaved`` ordering, ``coarse_to_fine`` digits and an affine map with
    ``grid_offset="left"`` when ``domain`` is supplied.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors following the public
        :class:`~tensorkrowch.formats.TT` layout, with open boundaries.
        The container is copied; tensors retain storage and autograd.
    n_coordinates : int
        Number of original coordinates. Required as a keyword argument.
    base : int or sequence of int, optional
        Digit bases for shorthand construction. Required together with ``level``
        and either ``domain`` or ``grid_coordinates``.
    level : int or sequence of int, optional
        Number of digits per coordinate for shorthand construction.
    domain : torch.Tensor or sequence, optional
        Coordinate intervals used to construct an
        :class:`~tensorkrowch.formats.AffineCoordinateMap` with
        ``grid_offset="left"``. Cannot be combined with ``grid_coordinates``
        or prebuilt objects.
    grid_coordinates : torch.Tensor or sequence, optional
        Domain grid points used to construct an
        :class:`~tensorkrowch.formats.ExplicitGridMap`. A vector describes one
        coordinate; a matrix or sequence of vectors describes multiple
        coordinates. Each grid size must equal ``base ** level``.
    layout : QuantizedLayout, optional
        Prebuilt digit layout, supplied together with ``coordinate_map``.
        Cannot be combined with shorthand construction arguments.
    coordinate_map : CoordinateMap, optional
        Prebuilt map with grid sizes matching ``layout.grid_size``.
    metrics : DecompositionMetrics, optional
        Historical fit records; defaults to a new empty collection.
    metadata : dict, optional
        Algorithm configuration; defaults to a new empty dictionary.
    n_batches : int, optional
        Number of leading structural batch axes in each core, independent of
        data batches during evaluation. Defaults to ``0``.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores. Defaults to ``None``. Core and factor
        containers are copied; tensors are reused without cloning or detaching.

    Examples
    --------
    >>> cores = [torch.eye(2), torch.eye(2)]
    >>> result = tk.decompositions.QTTDecomposition(
    ...     cores, n_coordinates=1, base=2, level=2, domain=(0., 1.),
    ...     metadata={'algorithm': 'manual'})
    >>> coordinates = torch.tensor([[0.], [0.25], [0.75]])
    >>> result.evaluate_coordinates(coordinates).tolist()
    [1.0, 0.0, 1.0]
    >>> result.to_format().to_dense_grid().tolist()
    [1.0, 0.0, 0.0, 1.0]
    """

    _format_type = QTT
    _format_parameters = ('n_batches', 'n_coordinates',
                          'layout', 'coordinate_map')


class QTRDecomposition(TensorDecomposition, QTR):  # MARK: QTRDecomposition
    """
    :class:`~tensorkrowch.formats.QTR` with decomposition diagnostics.

    The numerical API is inherited from ``QTR``. ``metrics`` and ``metadata``
    describe the fit that produced the cores and remain historical when the
    result is modified. Use :meth:`~TensorDecomposition.to_format` to obtain
    the corresponding ``QTR`` without decomposition diagnostics.

    Coordinate evaluation follows the stored
    :class:`~tensorkrowch.formats.QuantizedLayout` and
    :class:`~tensorkrowch.formats.CoordinateMap`. Shorthand construction uses
    ``interleaved`` ordering, ``coarse_to_fine`` digits and an affine map with
    ``grid_offset="left"`` when ``domain`` is supplied.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors following the public
        :class:`~tensorkrowch.formats.TR` layout, with a cyclic closure.
        The container is copied; tensors retain storage and autograd.
    n_coordinates : int
        Number of original coordinates. Required as a keyword argument.
    base : int or sequence of int, optional
        Digit bases for shorthand construction. Required together with ``level``
        and either ``domain`` or ``grid_coordinates``.
    level : int or sequence of int, optional
        Number of digits per coordinate for shorthand construction.
    domain : torch.Tensor or sequence, optional
        Coordinate intervals used to construct an
        :class:`~tensorkrowch.formats.AffineCoordinateMap` with
        ``grid_offset="left"``. Cannot be combined with ``grid_coordinates``
        or prebuilt objects.
    grid_coordinates : torch.Tensor or sequence, optional
        Domain grid points used to construct an
        :class:`~tensorkrowch.formats.ExplicitGridMap`. A vector describes one
        coordinate; a matrix or sequence of vectors describes multiple
        coordinates. Each grid size must equal ``base ** level``.
    layout : QuantizedLayout, optional
        Prebuilt digit layout, supplied together with ``coordinate_map``.
        Cannot be combined with shorthand construction arguments.
    coordinate_map : CoordinateMap, optional
        Prebuilt map with grid sizes matching ``layout.grid_size``.
    metrics : DecompositionMetrics, optional
        Historical fit records; defaults to a new empty collection.
    metadata : dict, optional
        Algorithm configuration; defaults to a new empty dictionary.
    n_batches : int, optional
        Number of leading structural batch axes in each core, independent of
        data batches during evaluation. Defaults to ``0``.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores. Defaults to ``None``. Core and factor
        containers are copied; tensors are reused without cloning or detaching.

    Examples
    --------
    >>> cores = [torch.tensor([1., 2.]).reshape(1, 2, 1),
    ...          torch.tensor([3., 4.]).reshape(1, 2, 1)]
    >>> result = tk.decompositions.QTRDecomposition(
    ...     cores, n_coordinates=1, base=2, level=2, domain=(0., 1.),
    ...     metadata={'algorithm': 'manual'})
    >>> coordinates = torch.tensor([[0.], [0.25], [0.75]])
    >>> result.evaluate_coordinates(coordinates).tolist()
    [3.0, 4.0, 8.0]
    >>> result.to_format().to_dense_grid().tolist()
    [3.0, 4.0, 6.0, 8.0]
    """

    _format_type = QTR
    _format_parameters = ('n_batches', 'n_coordinates',
                          'layout', 'coordinate_map')


class QTTMDecomposition(TensorDecomposition, QTTM):  # MARK: QTTMDecomposition
    """
    :class:`~tensorkrowch.formats.QTTM` with decomposition diagnostics.

    The numerical API is inherited from ``QTTM``. ``metrics`` and ``metadata``
    describe the fit that produced the cores and remain historical when the
    result is modified. Use :meth:`~TensorDecomposition.to_format` to obtain
    the corresponding ``QTTM`` without decomposition diagnostics.

    Input and output coordinate spaces independently accept shorthand
    construction or a prebuilt :class:`~tensorkrowch.formats.QuantizedLayout`
    and :class:`~tensorkrowch.formats.CoordinateMap`. Their layouts must have
    equally many sites. Shorthand layouts use ``interleaved`` ordering and
    ``coarse_to_fine`` digits; affine maps use ``grid_offset="left"``.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors following the public
        :class:`~tensorkrowch.formats.TTM` layout, with open boundaries.
        The container is copied; tensors retain storage and autograd.
    in_n_coordinates, out_n_coordinates : int
        Number of original input and output coordinates. Both are required
        as keyword arguments.
    in_base, out_base : int or sequence of int, optional
        Digit bases for shorthand construction of each coordinate space.
        Each base requires its corresponding level and domain or grid points.
    in_level, out_level : int or sequence of int, optional
        Number of digits per coordinate in each space.
    in_domain, out_domain : torch.Tensor or sequence, optional
        Coordinate intervals used to construct affine maps with
        ``grid_offset="left"``. Cannot be combined with grid points or
        prebuilt objects for the same space.
    in_grid_coordinates, out_grid_coordinates : torch.Tensor or sequence, optional
        Domain grid points used to construct explicit maps. A vector describes
        one coordinate; a matrix or sequence of vectors describes multiple
        coordinates. Each grid size must equal the corresponding
        ``base ** level``.
    in_layout, out_layout : QuantizedLayout, optional
        Prebuilt digit layouts, each supplied with its corresponding map.
        Both layouts must have the same number of sites.
    in_coordinate_map, out_coordinate_map : CoordinateMap, optional
        Prebuilt maps with grid sizes matching their respective layouts.
        Cannot be combined with shorthand arguments for the same space.
    metrics : DecompositionMetrics, optional
        Historical fit records; defaults to a new empty collection.
    metadata : dict, optional
        Algorithm configuration; defaults to a new empty dictionary.
    n_batches : int, optional
        Number of leading structural batch axes in each core, independent of
        data batches during evaluation. Defaults to ``0``.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores. Defaults to ``None``. Core and factor
        containers are copied; tensors are reused without cloning or detaching.

    Examples
    --------
    >>> cores = [torch.eye(2).unsqueeze(1), torch.eye(2).unsqueeze(0)]
    >>> result = tk.decompositions.QTTMDecomposition(
    ...     cores, in_n_coordinates=1, out_n_coordinates=1,
    ...     in_base=2, in_level=2, in_domain=(0., 1.),
    ...     out_base=2, out_level=2, out_domain=(-1., 1.),
    ...     metadata={'algorithm': 'manual'})
    >>> in_coordinates = torch.tensor([[0.], [0.5]])
    >>> out_coordinates = torch.tensor([[-1.], [0.5]])
    >>> result.evaluate_coordinates(in_coordinates, out_coordinates).tolist()
    [1.0, 0.0]
    """

    _format_type = QTTM
    _format_parameters = ('n_batches',
                          'in_n_coordinates', 'out_n_coordinates',
                          'in_layout', 'out_layout',
                          'in_coordinate_map', 'out_coordinate_map')


class QTRMDecomposition(TensorDecomposition, QTRM):  # MARK: QTRMDecomposition
    """
    :class:`~tensorkrowch.formats.QTRM` with decomposition diagnostics.

    The numerical API is inherited from ``QTRM``. ``metrics`` and ``metadata``
    describe the fit that produced the cores and remain historical when the
    result is modified. Use :meth:`~TensorDecomposition.to_format` to obtain
    the corresponding ``QTRM`` without decomposition diagnostics.

    Input and output coordinate spaces independently accept shorthand
    construction or a prebuilt :class:`~tensorkrowch.formats.QuantizedLayout`
    and :class:`~tensorkrowch.formats.CoordinateMap`. Their layouts must have
    equally many sites. Shorthand layouts use ``interleaved`` ordering and
    ``coarse_to_fine`` digits; affine maps use ``grid_offset="left"``.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors following the public
        :class:`~tensorkrowch.formats.TRM` layout, with a cyclic closure.
        The container is copied; tensors retain storage and autograd.
    in_n_coordinates, out_n_coordinates : int
        Number of original input and output coordinates. Both are required
        as keyword arguments.
    in_base, out_base : int or sequence of int, optional
        Digit bases for shorthand construction of each coordinate space.
        Each base requires its corresponding level and domain or grid points.
    in_level, out_level : int or sequence of int, optional
        Number of digits per coordinate in each space.
    in_domain, out_domain : torch.Tensor or sequence, optional
        Coordinate intervals used to construct affine maps with
        ``grid_offset="left"``. Cannot be combined with grid points or
        prebuilt objects for the same space.
    in_grid_coordinates, out_grid_coordinates : torch.Tensor or sequence, optional
        Domain grid points used to construct explicit maps. A vector describes
        one coordinate; a matrix or sequence of vectors describes multiple
        coordinates. Each grid size must equal the corresponding
        ``base ** level``.
    in_layout, out_layout : QuantizedLayout, optional
        Prebuilt digit layouts, each supplied with its corresponding map.
        Both layouts must have the same number of sites.
    in_coordinate_map, out_coordinate_map : CoordinateMap, optional
        Prebuilt maps with grid sizes matching their respective layouts.
        Cannot be combined with shorthand arguments for the same space.
    metrics : DecompositionMetrics, optional
        Historical fit records; defaults to a new empty collection.
    metadata : dict, optional
        Algorithm configuration; defaults to a new empty dictionary.
    n_batches : int, optional
        Number of leading structural batch axes in each core, independent of
        data batches during evaluation. Defaults to ``0``.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores. Defaults to ``None``. Core and factor
        containers are copied; tensors are reused without cloning or detaching.

    Examples
    --------
    >>> cores = [torch.eye(2).reshape(1, 2, 1, 2) for _ in range(2)]
    >>> result = tk.decompositions.QTRMDecomposition(
    ...     cores, in_n_coordinates=1, out_n_coordinates=1,
    ...     in_base=2, in_level=2, in_domain=(0., 1.),
    ...     out_base=2, out_level=2, out_domain=(-1., 1.),
    ...     metadata={'algorithm': 'manual'})
    >>> in_coordinates = torch.tensor([[0.], [0.5]])
    >>> out_coordinates = torch.tensor([[-1.], [0.5]])
    >>> result.evaluate_coordinates(in_coordinates, out_coordinates).tolist()
    [1.0, 0.0]
    """

    _format_type = QTRM
    _format_parameters = ('n_batches',
                          'in_n_coordinates', 'out_n_coordinates',
                          'in_layout', 'out_layout',
                          'in_coordinate_map', 'out_coordinate_map')


def _quantics_result(
        result: TensorDecomposition,
        source: Optional[Union['QuanticsVectorSource',
                               'QuanticsMatrixSource']] = None
    ) -> TensorDecomposition:
    """Attaches the Quantics source's layout and maps to fitted cores."""
    if source is None:
        return result

    from tensorkrowch.decompositions.sources.quantics import QuanticsMatrixSource

    kwargs = dict(metrics=result.metrics,
                  metadata=dict(result.metadata),
                  n_batches=result.n_batches)
    if isinstance(source, QuanticsMatrixSource):
        cls = QTRMDecomposition if result.topology == 'trm' else QTTMDecomposition
        layouts = (source.in_layout, source.out_layout)
        kwargs.update(in_n_coordinates=source.in_n_coordinates,
                      out_n_coordinates=source.out_n_coordinates,
                      in_layout=source.in_layout,
                      out_layout=source.out_layout,
                      in_coordinate_map=source.in_coordinate_map,
                      out_coordinate_map=source.out_coordinate_map)
    else:
        cls = QTRDecomposition if result.topology == 'tr' else QTTDecomposition
        layouts = (source.layout,)
        kwargs.update(n_coordinates=source.n_coordinates,
                      layout=source.layout,
                      coordinate_map=source.coordinate_map)

    wrapped = cls(result.cores, **kwargs)
    if 'quantization' not in wrapped.metadata:
        wrapped.metadata['quantization'] = [
            {'base': layout.base,
             'level': layout.level,
             'sites': layout.sites(),
             'grid_size': layout.grid_size}
            for layout in layouts]
    return wrapped.to(device=result.device)


_DecompositionOutput = Union[
    List[torch.Tensor],
    Tuple[List[torch.Tensor], Dict[str, Any]],
    TensorDecomposition,
]
