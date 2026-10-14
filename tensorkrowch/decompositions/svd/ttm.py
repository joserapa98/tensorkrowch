"""
This script contains:

    Matrix SVD interface:
        * TTMSVD
        * ttm_svd
        * qttm_svd
        * mat_to_mpo
"""

from typing import Optional, TYPE_CHECKING, Union
import warnings

import torch

from tensorkrowch.decompositions.results import (TTMDecomposition,
                                                 QTTMDecomposition)
from tensorkrowch.decompositions.sources.dense import DenseTensorSource
from tensorkrowch.decompositions.sources.factory import SourceLike
from tensorkrowch.decompositions.sources.quantics import CallableQuanticsSource
from tensorkrowch.decompositions.svd.tt import TTSVD

if TYPE_CHECKING:
    from tensorkrowch.decompositions.results import _DecompositionOutput


class TTMSVD(TTSVD):  # MARK: TTMSVD
    """
    Decomposes a matrix source using the shared TT-SVD algorithm.

    The source owns paired dimensions and axis ordering. Input/output pairs
    are fused for the numerical sweep and restored in the returned matrix
    decomposition. Quantics source metadata selects a Quantics result.

    Parameters
    ----------
    source : TensorSource or torch.Tensor
        Matrix source, or a tensor with interleaved input/output site axes.
        Use ``DenseTensorSource.matrix`` to tensorize an ordinary matrix or
        specify a different axis ordering.
    out_device : str or torch.device, optional
        Device for finalized cores. ``None`` retains the evaluation device.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource.matrix(
    ...     torch.eye(4), in_dim=(2, 2), out_dim=(2, 2))
    >>> result = tk.decompositions.TTMSVD(source).fit(rank=2)
    >>> result.in_dim
    (2, 2)
    """

    def __init__(self, source: 'SourceLike', *,
                 out_device: Optional[Union[str, torch.device]] = 'cpu') -> None:
        if isinstance(source, torch.Tensor):
            source = DenseTensorSource.matrix(source)
        if not callable(getattr(source, 'contract_dense', None)):
            raise TypeError('`source` should provide `contract_dense`')
        if not getattr(source, 'is_matrix', False):
            raise ValueError('`source` should describe paired matrix axes')
        super().__init__(source, out_device=out_device)


def ttm_svd(source: 'SourceLike',
            *,
            rank: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            renormalize: bool = False,
            out_device: Optional[Union[str, torch.device]] = 'cpu',
            verbose: Union[bool, int] = 0,
            collect_metrics: bool = False) -> TTMDecomposition:
    r"""
    Decomposes a finite source using :class:`TTMSVD`.

    The source owns its dimensions, batch axes and matrix ordering. A plain
    tensor is wrapped as a dense source. The result retains Quantics metadata
    when the source describes a coordinate discretization.

    Parameters
    ----------
    source : TensorSource or torch.Tensor
        Finite tensor values with source-owned dimensions and runtime. For
        matrices, use ``DenseTensorSource.matrix``; for coordinate functions,
        use ``CallableQuanticsSource``.
    rank : int, optional
        Maximum ``rank`` allowed at every link. At each SVD cut, at most this many
        singular values are retained.
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
        If ``True``, normalizes the residual before every SVD, accumulates its
        scale logarithmically and evenly redistributes the complete scale over
        the final cores. Absolute criteria and reported errors preserve the
        scale of the original ``tensor``.
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
    TTMDecomposition
        Decomposition with cores and metadata. Access ``cores`` for the
        tensors and :meth:`~tensorkrowch.decompositions.TensorDecomposition.as_info`
        for structured information.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource.matrix(
    ...     torch.arange(16.).reshape(2, 2, 2, 2))
    >>> result = tk.decompositions.ttm_svd(source, rank=2)
    >>> result.in_dim
    (2, 2)
    """
    result = TTMSVD(
        source=source,
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


def qttm_svd(source: CallableQuanticsSource,
             *,
             rank: Optional[int] = None,
             cutoff: Optional[float] = None,
             atol: Optional[float] = None,
             rtol: Optional[float] = None,
             cum_percentage: Optional[float] = None,
             renormalize: bool = False,
             out_device: Optional[Union[str, torch.device]] = 'cpu',
             verbose: Union[bool, int] = 0,
             collect_metrics: bool = False) -> QTTMDecomposition:
    r"""
    Decomposes a discretized coordinate callable into QTTM cores.

    Materializes the full digit tensor from ``source`` and runs
    :meth:`TTMSVD.fit` through :class:`TTMSVD`. The result retains
    the source's layouts and coordinate maps for coordinate evaluation.
    Construct ``source`` first to choose its discretization; this function
    only controls the decomposition.

    Parameters
    ----------
    source : CallableQuanticsSource
        Callable with its digit layout and coordinate maps.
    rank : int, optional
        Maximum ``rank`` allowed at every link. At each SVD cut, at most this many
        singular values are retained.
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
        If ``True``, normalizes the residual before every SVD, accumulates its
        scale logarithmically and evenly redistributes the complete scale over
        the final cores. Absolute criteria and reported errors preserve the
        scale of the original sampled tensor.
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
    QTTMDecomposition
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
    >>> result = tk.decompositions.qttm_svd(source, rank=1)
    >>> result.evaluate_indices(
    ...     torch.tensor([[0], [4]]), torch.tensor([[0], [0]]))
    tensor([1.0000, 1.6487])
    """
    if not isinstance(source, CallableQuanticsSource):
        raise TypeError('`source` should be CallableQuanticsSource type')
    return TTMSVD(
        source=source,
        out_device=out_device).fit(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            renormalize=renormalize,
            collect_metrics=collect_metrics,
            verbose=verbose)


def mat_to_mpo(mat: torch.Tensor,
               rank: Optional[int] = None,
               cutoff: Optional[float] = None,
               atol: Optional[float] = None,
               rtol: Optional[float] = None,
               cum_percentage: Optional[float] = None,
               renormalize: bool = False,
               verbose: Union[bool, int] = 0,
               return_info: bool = False) -> '_DecompositionOutput':
    r"""
    Compatibility wrapper for :func:`ttm_svd`.

    .. deprecated:: 1.2
        Use :func:`ttm_svd` for TTM terminology, grouped or matrix layouts,
        explicit output-device policy and repeated fits through
        :class:`TTMSVD`.

    The historical ``mat`` argument and interleaved ordering are preserved.
    Final cores remain on the input device, matching the previous behavior.
    ``mat`` should have shape
    ``(in_1, out_1, ..., in_n, out_n)``.

    Parameters
    ----------
    mat : torch.Tensor
        Dense tensor with interleaved input/output dimensions.
    rank : int, optional
        Maximum ``rank`` allowed at every link. At each SVD cut, at most this many
        singular values are retained.
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
        If ``True``, normalizes each residual and accumulates its scale
        logarithmically before redistributing it over the final cores.
    verbose : bool or int
        Console verbosity level forwarded to :func:`ttm_svd`.
    return_info : bool
        If ``True``, returns ``(cores, info)`` with ranks, dimensions,
        metadata and structured metrics.

    Returns
    -------
    list[torch.Tensor] or tuple
        TTM cores, optionally followed by their structured information.

    Examples
    --------
    The canonical replacement keeps the historical interleaved ordering:

    >>> tensor = torch.arange(16.).reshape(2, 2, 2, 2)
    >>> cores = tk.decompositions.ttm_svd(tensor, rank=2).cores
    >>> [tuple(core.shape) for core in cores]
    [(2, 2, 2), (2, 2, 2)]
    """
    warnings.warn(
        '`mat_to_mpo` is deprecated; use `ttm_svd` instead',
        FutureWarning,
        stacklevel=2)
    if not isinstance(mat, torch.Tensor):
        raise TypeError('`mat` should be torch.Tensor type')
    if mat.ndim < 2 or (mat.ndim % 2):
        raise ValueError('`mat` have an even number of dimensions')

    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')

    result = ttm_svd(
        source=DenseTensorSource.matrix(mat),
        rank=rank,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
        cum_percentage=cum_percentage,
        renormalize=renormalize,
        out_device=None,
        verbose=verbose,
        collect_metrics=return_info)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


__all__ = [
    'TTMSVD',
    'ttm_svd',
    'qttm_svd',
    'mat_to_mpo',
]
