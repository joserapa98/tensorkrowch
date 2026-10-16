"""
This script contains:

    Matrix SVD interface:
        * TRMSVD
        * trm_svd
        * qtrm_svd
"""

from typing import Optional, Union

import torch

from tensorkrowch.decompositions.results import (TRMDecomposition,
                                                 QTRMDecomposition)
from tensorkrowch.decompositions.sources.dense import DenseTensorSource
from tensorkrowch.decompositions.sources.factory import SourceLike
from tensorkrowch.decompositions.sources.quantics import CallableQuanticsSource
from tensorkrowch.decompositions.svd.tr import TRSVD


class TRMSVD(TRSVD):  # MARK: TRMSVD
    """
    Decomposes a matrix source using the shared TR-SVD algorithm.

    The source owns paired dimensions and axis ordering. Input/output pairs
    are fused for the numerical sweep and restored in the returned matrix
    decomposition. Quantics source metadata selects a Quantics result.

    ``source`` defines the input's dimensions, evaluation rules and matrix
    grouping. Finite sources include
    :class:`~tensorkrowch.decompositions.DenseTensorSource`,
    :class:`~tensorkrowch.decompositions.SparseTensorSource`,
    :class:`~tensorkrowch.decompositions.FormatSource`,
    :class:`~tensorkrowch.decompositions.CallableTensorSource` and
    :class:`~tensorkrowch.decompositions.CallableQuanticsSource`.
    SVD materializes their complete tensor once.
    A ``torch.Tensor`` is accepted directly, as is a non-Quantics TT/TR format
    or its matrix version. Wrap an :class:`~tensorkrowch.models.MPS`
    in ``FormatSource`` first.

    To declare matrix inputs, use
    :meth:`~tensorkrowch.decompositions.DenseTensorSource.matrix` or the
    corresponding ``.matrix(...)`` constructor of ``SparseTensorSource``,
    ``CallableTensorSource`` or ``CallableQuanticsSource``. ``FormatSource``
    instead infers matrix grouping from a matrix format. The source selects the
    result family and retains any Quantics layouts and coordinate maps.

    Parameters
    ----------
    source : TensorSource or torch.Tensor
        Finite matrix source declaring paired dimensions, or a tensor with
        interleaved input/output axes. Wrap matrix formats in ``FormatSource``.
    center : int, optional
        Preferred interior cut, passed to ``TRSVD``.
    out_device : str or torch.device, optional
        Device for finalized cores. ``None`` retains the evaluation device.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource.matrix(
    ...     torch.eye(4), in_dim=(2, 2), out_dim=(2, 2))
    >>> result = tk.decompositions.TRMSVD(source).fit(rank=2)
    >>> result.in_dim
    (2, 2)
    """

    def __init__(self,
                 source: 'SourceLike',
                 center: Optional[int] = None,
                 *,
                 out_device: Optional[Union[str, torch.device]] = 'cpu') -> None:
        if isinstance(source, torch.Tensor):
            source = DenseTensorSource.matrix(source)
        if not callable(getattr(source, 'contract_dense', None)):
            raise TypeError('`source` should provide `contract_dense`')
        if not getattr(source, 'is_matrix', False):
            raise ValueError('`source` should describe paired matrix axes')
        super().__init__(source, center=center, out_device=out_device)


def trm_svd(source: 'SourceLike',
            center: Optional[int] = None,
            *,
            rank: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            renormalize: bool = False,
            out_device: Optional[Union[str, torch.device]] = 'cpu',
            verbose: Union[bool, int] = 0,
            collect_metrics: bool = False) -> TRMDecomposition:
    r"""
    Decomposes a finite source using :class:`TRMSVD`.

    ``source`` defines the input's dimensions, evaluation rules and matrix
    grouping. Finite sources include
    :class:`~tensorkrowch.decompositions.DenseTensorSource`,
    :class:`~tensorkrowch.decompositions.SparseTensorSource`,
    :class:`~tensorkrowch.decompositions.FormatSource`,
    :class:`~tensorkrowch.decompositions.CallableTensorSource` and
    :class:`~tensorkrowch.decompositions.CallableQuanticsSource`.
    SVD materializes their complete tensor once.
    A ``torch.Tensor`` is accepted directly, as is a non-Quantics TT/TR format
    or its matrix version. Wrap an :class:`~tensorkrowch.models.MPS`
    in ``FormatSource`` first.

    To declare matrix inputs, use
    :meth:`~tensorkrowch.decompositions.DenseTensorSource.matrix` or the
    corresponding ``.matrix(...)`` constructor of ``SparseTensorSource``,
    ``CallableTensorSource`` or ``CallableQuanticsSource``. ``FormatSource``
    instead infers matrix grouping from a matrix format. The source selects the
    result family and retains any Quantics layouts and coordinate maps.

    Parameters
    ----------
    source : TensorSource or torch.Tensor
        Finite matrix source declaring paired dimensions, or a tensor with
        interleaved input/output axes. Wrap matrix formats in ``FormatSource``.
    center : int, optional
        Interior cut satisfying ``1 <= center < n_sites``. The default is the
        middle cut.
    rank : int, optional
        Maximum ``rank`` allowed at every link. At each subchain SVD cut, at
        most this many singular values are retained. The initial bipartition
        retains at most ``rank ** 2`` singular values before its selected
        ``rank`` is factorized into two TRM ranks.
    cutoff : float, optional
        Minimum singular value to keep. It must be finite and non-negative.
        Singular values ``<= cutoff`` are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the accumulated sum of squares is ``<= atol``. It must be finite and
        non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the tail sum of squares divided by the total sum of squares is
        ``<= rtol``. It must be finite and in ``[0, 1]``.
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent to
        setting ``rtol = 1 - cum_percentage``. It must be finite and in
        ``[0, 1]``.

        .. math::

        \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
            cum\_percentage

    renormalize : bool
        If ``True``, normalizes residuals before SVDs and accumulates their
        scales logarithmically before redistributing them over final cores.
    out_device : str or torch.device, optional
        Device where finalized cores are stored. If ``None``, they remain on
        the input device. The default is ``"cpu"``.
    verbose : bool or int
        Console verbosity level:

        - ``0`` or ``False``: no console output;
        - ``1`` or ``True``: phase title, input configuration, cut progress
          and final summary;
        - ``2``: detailed per-cut ``rank``, error and timing information;
        - ``3``: level 2 output followed by every final core.

    collect_metrics : bool
        Whether to collect structured error, truncation and timing metrics.
        With ``False`` and ``verbose=0``, diagnostic reductions and
        synchronized timings are skipped.

    Returns
    -------
    TRMDecomposition
        Decomposition with cores and metadata. Access ``cores`` for the
        tensors and :meth:`~tensorkrowch.decompositions.TensorDecomposition.as_info`
        for structured information.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource.matrix(
    ...     torch.arange(16.).reshape(2, 2, 2, 2))
    >>> result = tk.decompositions.trm_svd(source, rank=2)
    >>> result.in_dim
    (2, 2)
    """
    result = TRMSVD(
        source=source,
        center=center,
        out_device=out_device).fit(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            renormalize=renormalize,
            collect_metrics=collect_metrics,
            verbose=verbose)
    return result


def qtrm_svd(source: CallableQuanticsSource,
             center: Optional[int] = None,
             *,
             rank: Optional[int] = None,
             cutoff: Optional[float] = None,
             atol: Optional[float] = None,
             rtol: Optional[float] = None,
             cum_percentage: Optional[float] = None,
             renormalize: bool = False,
             out_device: Optional[Union[str, torch.device]] = 'cpu',
             verbose: Union[bool, int] = 0,
             collect_metrics: bool = False) -> QTRMDecomposition:
    r"""
    Decomposes a discretized coordinate callable into QTRM cores.

    Materializes the full digit tensor from ``source`` and runs
    :meth:`TRMSVD.fit` through :class:`TRMSVD`. The result retains
    the source's layouts and coordinate maps for coordinate evaluation.

    Construct the input with
    :meth:`~tensorkrowch.decompositions.CallableQuanticsSource.matrix`
    to define paired input/output coordinates, digit layouts and grids.
    This function controls the decomposition and retains that information.

    Parameters
    ----------
    source : CallableQuanticsSource
        Callable with its digit layout and coordinate maps.
    center : int, optional
        Interior cut from which the cyclic decomposition starts. If omitted,
        uses the middle of the digit chain.
    rank : int, optional
        Maximum ``rank`` allowed at every link. At each subchain SVD cut, at
        most this many singular values are retained. The initial bipartition
        retains at most ``rank ** 2`` singular values before its selected
        ``rank`` is factorized into two TRM ranks.
    cutoff : float, optional
        Minimum singular value to keep. It must be finite and non-negative.
        Singular values ``<= cutoff`` are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the accumulated sum of squares is ``<= atol``. It must be finite and
        non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the tail sum of squares divided by the total sum of squares is
        ``<= rtol``. It must be finite and in ``[0, 1]``.
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent to
        setting ``rtol = 1 - cum_percentage``. It must be finite and in
        ``[0, 1]``.

    renormalize : bool
        If ``True``, normalizes residuals before SVDs and accumulates their
        scales logarithmically before redistributing them over final cores.
    out_device : str or torch.device, optional
        Device where finalized cores are stored. If ``None``, they remain on
        the source device. The default is ``"cpu"``.
    verbose : bool or int
        Console verbosity level:

        - ``0`` or ``False``: no console output;
        - ``1`` or ``True``: phase title, input configuration, cut progress
          and final summary;
        - ``2``: detailed per-cut ``rank``, error and timing information;
        - ``3``: level 2 output followed by every final core.

    collect_metrics : bool
        Whether to collect structured error, truncation and timing metrics.
        With ``False`` and ``verbose=0``, diagnostic reductions and
        synchronized timings are skipped.

    Returns
    -------
    QTRMDecomposition
        Quantics decomposition with cores, coordinate information and
        optional metrics.

    Examples
    --------
    >>> source = tk.decompositions.CallableQuanticsSource.matrix(
    ...     lambda inputs, outputs: torch.exp(
    ...         inputs.sum(-1) + outputs.sum(-1)),
    ...     in_n_coordinates=1, out_n_coordinates=1,
    ...     in_base=2, out_base=2, in_level=3, out_level=3,
    ...     in_domain=(0., 1.), out_domain=(0., 1.))
    >>> result = tk.decompositions.qtrm_svd(source, rank=1)
    >>> result.evaluate_indices(
    ...     torch.tensor([[0], [4]]), torch.tensor([[0], [0]]))
    tensor([1.0000, 1.6487])
    """
    if not isinstance(source, CallableQuanticsSource):
        raise TypeError('`source` should be CallableQuanticsSource type')
    return TRMSVD(
        source=source,
        center=center,
        out_device=out_device).fit(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            renormalize=renormalize,
            collect_metrics=collect_metrics,
            verbose=verbose)


__all__ = [
    'TRMSVD',
    'trm_svd',
    'qtrm_svd',
]
