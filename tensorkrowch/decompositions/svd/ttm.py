"""Tensor train matrix decomposition through the TT-SVD engine."""

from math import prod
from typing import List, Optional, Sequence, Tuple, Union
import warnings

import torch

from tensorkrowch.decompositions.observers import (
    DecompositionEvent,
    _normalize_verbosity,
    _resolve_observer,
)
from tensorkrowch.decompositions.results import TTMDecomposition
from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.decompositions.svd.tt import TTSVD


_Dimension = Optional[Union[int, Sequence[int]]]


def _normalize_dim(dim: _Dimension, name: str) -> Tuple[int, ...]:
    """Normalizes one or several positive site dimensions."""
    if isinstance(dim, bool):
        raise TypeError(f'`{name}` should be int or a sequence of ints')
    if isinstance(dim, int):
        values = (dim,)
    elif isinstance(dim, Sequence) and not isinstance(dim, (str, bytes)):
        values = tuple(dim)
    else:
        raise TypeError(f'`{name}` should be int or a sequence of ints')

    if not values:
        raise ValueError(f'`{name}` should contain at least one dimension')
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in values):
        raise TypeError(f'`{name}` should contain only ints')
    if any(value < 1 for value in values):
        raise ValueError(f'`{name}` should contain only positive dimensions')
    return values


class TTMSVD:
    """Decomposes a fixed dense tensor into a tensor train matrix.

    The tensor, its input/output dimensions and its axis layout are fixed when
    this object is created. :meth:`fit` can then be called repeatedly with
    different truncation criteria. The numerical sweep is performed by
    :class:`~tensorkrowch.decompositions.TTSVD` after fusing each local
    input/output pair, so both algorithms share the same truncation, error,
    normalization and SVD-backend semantics.

    A tensorized input can use either ``layout="interleaved"`` with shape
    ``(in_1, out_1, ..., in_n, out_n)`` or ``layout="grouped"`` with shape
    ``(in_1, ..., in_n, out_1, ..., out_n)``. When ``in_dim`` and
    ``out_dim`` are provided, a two-dimensional tensor is instead treated
    as a matrix with shape ``(prod(in_dim), prod(out_dim))`` and is
    tensorized internally. TTM batch dimensions are not supported.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor or matrix fixed for repeated fits.
    in_dim : int or sequence[int], optional
        Input dimension of each site. It is required together with
        ``out_dim`` for the matrix input route and otherwise can be omitted
        because dimensions are inferred from ``tensor``.
    out_dim : int or sequence[int], optional
        Output dimension of each site. It should contain the same number of
        sites as ``in_dim``.
    layout : {"interleaved", "grouped"}
        Axis layout of a tensorized input. The default is ``"interleaved"``.
        It does not alter the two matrix axes in the explicit-dimension route.
    out_device : str or torch.device, optional
        Device where finalized cores are stored. The default is ``"cpu"``.
        If ``None``, cores remain on the input device.
    """

    def __init__(self,
                 tensor: torch.Tensor,
                 in_dim: _Dimension = None,
                 out_dim: _Dimension = None,
                 *,
                 layout: str = 'interleaved',
                 out_device: Optional[
                     Union[str, torch.device]] = 'cpu') -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if not isinstance(layout, str):
            raise TypeError('`layout` should be str type')
        if layout not in ('interleaved', 'grouped'):
            raise ValueError(
                '`layout` should be either "interleaved" or "grouped"')
        if (in_dim is None) != (out_dim is None):
            raise ValueError(
                '`in_dim` and `out_dim` should be provided together')

        matrix_input = False
        if in_dim is None:
            if (tensor.ndim < 2) or (tensor.ndim % 2):
                raise ValueError(
                    'A tensorized TTM input should have a positive even '
                    'number of dimensions')
            n_sites = tensor.ndim // 2
            if layout == 'interleaved':
                normalized_in_dim = tuple(tensor.shape[::2])
                normalized_out_dim = tuple(tensor.shape[1::2])
            else:
                normalized_in_dim = tuple(tensor.shape[:n_sites])
                normalized_out_dim = tuple(tensor.shape[n_sites:])
            if any(value < 1
                   for value in normalized_in_dim + normalized_out_dim):
                raise ValueError(
                    'TTM input and output dimensions should be positive')
            tensorized = tensor
        else:
            normalized_in_dim = _normalize_dim(in_dim, 'in_dim')
            normalized_out_dim = _normalize_dim(out_dim, 'out_dim')
            if len(normalized_in_dim) != len(normalized_out_dim):
                raise ValueError(
                    '`in_dim` and `out_dim` should have the same length')
            n_sites = len(normalized_in_dim)

            if tensor.ndim == 2:
                expected_shape = (
                    prod(normalized_in_dim),
                    prod(normalized_out_dim),
                )
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        'The matrix shape should equal '
                        '(prod(in_dim), prod(out_dim))')
                tensorized = tensor.reshape(
                    *normalized_in_dim, *normalized_out_dim)
                matrix_input = True
            else:
                if tensor.ndim != (2 * n_sites):
                    raise ValueError(
                        'A tensorized TTM input should have two dimensions '
                        'per site')
                expected_shape = (
                    tuple(value
                          for pair in zip(normalized_in_dim,
                                          normalized_out_dim)
                          for value in pair)
                    if layout == 'interleaved'
                    else normalized_in_dim + normalized_out_dim
                )
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(
                        'The tensor shape is incompatible with `in_dim`, '
                        '`out_dim` and `layout`')
                tensorized = tensor

        interleaved = self._interleave_axes(
            tensorized,
            n_sites=n_sites,
            layout=('grouped' if matrix_input else layout))
        fused_dim = tuple(
            in_value * out_value
            for in_value, out_value
            in zip(normalized_in_dim, normalized_out_dim))
        fused_tensor = interleaved.reshape(*fused_dim)

        self._tensor = tensor
        self._in_dim = normalized_in_dim
        self._out_dim = normalized_out_dim
        self._layout = layout
        self._matrix_input = matrix_input
        self._engine = TTSVD(
            fused_tensor,
            out_device=out_device)

    @property
    def tensor(self) -> torch.Tensor:
        """Dense tensor or matrix fixed for repeated fits."""
        return self._tensor

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension associated with every TTM site."""
        return self._in_dim

    @property
    def out_dim(self) -> Tuple[int, ...]:
        """Output dimension associated with every TTM site."""
        return self._out_dim

    @property
    def layout(self) -> str:
        """Axis layout specified for the fixed input tensor."""
        return self._layout

    @staticmethod
    def _interleave_axes(tensor: torch.Tensor,
                         n_sites: int,
                         layout: str) -> torch.Tensor:
        """Moves a grouped tensor to interleaved input/output order."""
        if (layout == 'interleaved') or (n_sites == 1):
            return tensor
        axes = tuple(
            axis
            for site in range(n_sites)
            for axis in (site, n_sites + site))
        return tensor.permute(axes)

    def _unfuse_in_out_axes(
            self, cores: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Reopens fused TT input axes into TTM input/output axes."""
        if len(cores) == 1:
            return [cores[0].reshape(self.in_dim[0], self.out_dim[0])]

        ttm_cores = []
        first = cores[0].reshape(
            self.in_dim[0], self.out_dim[0], cores[0].shape[-1])
        ttm_cores.append(first.permute(0, 2, 1))

        for site, core in enumerate(cores[1:-1], 1):
            core = core.reshape(
                core.shape[0],
                self.in_dim[site],
                self.out_dim[site],
                core.shape[-1])
            ttm_cores.append(core.permute(0, 1, 3, 2))

        last = cores[-1].reshape(
            cores[-1].shape[0],
            self.in_dim[-1],
            self.out_dim[-1])
        ttm_cores.append(last)
        return ttm_cores

    def fit(self,
            rank: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            renormalize: bool = False,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0) -> TTMDecomposition:
        r"""Runs TTM-SVD with a shared truncation policy at every cut.

        The active exact SVD backend is selected through
        :func:`tensorkrowch.set_svd_method` or
        :func:`tensorkrowch.svd_method`. When metrics are collected, local
        discarded energies and their accumulated absolute and relative
        reconstruction errors are returned in ``result.metrics``.

        If several truncation criteria are specified, each one provides an
        upper bound for the selected rank and the most restrictive bound is
        used. At least one singular value is always retained. The same
        criteria are applied at every TTM cut.

        The fixed tensor is normalized to interleaved shape
        ``(in_1, out_1, ..., in_n, out_n)``. A grouped tensor has initial shape
        ``(in_1, ..., in_n, out_1, ..., out_n)`` and is interleaved internally.
        A matrix has shape ``(prod(in_dim), prod(out_dim))`` and requires
        explicit input/output dimensions. Each local pair is then fused and
        decomposed through TT-SVD without an additional factorization.

        For more than one site, the returned cores have shapes
        ``(in_1, rank_1, out_1)``,
        ``(rank_{k-1}, in_k, rank_k, out_k)`` at interior sites, and
        ``(rank_{n-1}, in_n, out_n)`` at the last site. A one-site tensor is
        returned as a single core with shape ``(in_1, out_1)``. Consequently,
        the result always represents an open-boundary TTM.

        Parameters
        ----------
        rank : int, optional
            Number of singular values to keep.
        cutoff : float, optional
            Minimum singular value to keep. It must be non-negative. Singular
            values ``<= cutoff`` are removed.
        atol : float, optional
            Absolute tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded while
            the accumulated sum of squares is ``<= atol``. It must be non-negative.
        rtol : float, optional
            Relative tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded while
            the tail sum of squares divided by the total sum of squares is
            ``<= rtol``. It must be in ``[0, 1]``.
        cum_percentage : float, optional
            Minimum fraction of squared singular-value mass to keep. Equivalent to
            setting ``rtol = 1 - cum_percentage``. It must be in ``[0, 1]``.

            .. math::

                \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
                cum\_percentage

        renormalize : bool
            If ``True``, normalizes the residual before every SVD and
            accumulates the extracted scale logarithmically. The complete
            scale is evenly redistributed over the final cores. Absolute
            truncation criteria and reported errors remain expressed in the
            scale of the original tensor.
        collect_metrics : bool
            If ``True``, collects local truncation records, accumulated errors
            and synchronized timings in ``result.metrics``. The default is
            ``False`` to avoid diagnostic norm reductions, records and device
            synchronizations. Metrics are always collected when console output
            is requested.
        verbose : bool or int
            Console verbosity level:

            - ``0`` or ``False``: no console output;
            - ``1`` or ``True``: phase title, site progress and final summary;
            - ``2``: input configuration and detailed per-site rank, error and
              timing information;
            - ``3``: level 2 output followed by every final core.

        Returns
        -------
        TTMDecomposition
            Lightweight result containing cores and ranks. Its structured
            metrics are empty unless ``collect_metrics=True`` or
            ``verbose>0``.

        Examples
        --------
        Fix a grouped tensor once and compare two maximum ranks:

        >>> tensor = torch.arange(36.).reshape(2, 2, 3, 3)
        >>> decomposer = TTMSVD(tensor, layout='grouped')
        >>> rank_one = decomposer.fit(rank=1)
        >>> rank_two = decomposer.fit(rank=2)
        >>> rank_one.rank
        [1]
        >>> rank_two.rank
        [2]
        """
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        _TruncationSpec(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage)
        verbosity = _normalize_verbosity(verbose)
        emit_events = bool(verbosity)
        fit_observer = (
            _resolve_observer(verbosity, None) if emit_events else None)
        collect_metrics = collect_metrics or emit_events

        if fit_observer is not None:
            fit_observer.emit(DecompositionEvent(
                name='start',
                phase='TTM-SVD',
                values={
                    'sites': len(self.in_dim),
                    'in_dim': self.in_dim,
                    'out_dim': self.out_dim,
                    'layout': self.layout,
                    'matrix_input': self._matrix_input,
                    'renormalize': renormalize,
                }))

        tt_result = self._engine.fit(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            renormalize=renormalize,
            collect_metrics=collect_metrics)
        cores = self._unfuse_in_out_axes(tt_result.cores)
        result = TTMDecomposition(
            cores=cores,
            metrics=tt_result.metrics,
            metadata={
                'algorithm': 'ttm_svd',
                'layout': self.layout,
                'matrix_input': self._matrix_input,
                'renormalize': renormalize,
            })

        if fit_observer is not None:
            timing = result.metrics.timings[0]
            for site, (record, cut_timing) in enumerate(zip(
                    result.metrics.truncations, timing.children)):
                fit_observer.emit(DecompositionEvent(
                    name='site_complete',
                    phase='TTM-SVD',
                    site=site,
                    elapsed=cut_timing.elapsed,
                    values={
                        'total_sites': len(self.in_dim) - 1,
                        'full_rank': record.full_rank,
                        'selected_rank': record.selected_rank,
                        'absolute_error': record.local_absolute_error,
                        'relative_error': record.local_relative_error,
                    }))
            error = result.metrics.errors[0]
            fit_observer.emit(DecompositionEvent(
                name='summary',
                phase='TTM-SVD',
                values={
                    'rank': result.rank,
                    'absolute_error': error.absolute,
                    'relative_error': error.relative,
                    'elapsed': f'{timing.elapsed:.6f} s',
                }))
            for site, core in enumerate(result.cores):
                fit_observer.emit(DecompositionEvent(
                    name='core',
                    phase='TTM-SVD',
                    level=3,
                    site=site,
                    values={'shape': tuple(core.shape), 'tensor': core}))
            fit_observer.close(result.metrics)
        return result


def ttm_svd(tensor: torch.Tensor,
            in_dim: _Dimension = None,
            out_dim: _Dimension = None,
            *,
            layout: str = 'interleaved',
            rank: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            renormalize: bool = False,
            out_device: Optional[Union[str, torch.device]] = 'cpu',
            verbose: Union[bool, int] = 0,
            return_info: bool = False):
    r"""Decomposes a dense tensor or matrix into TTM cores.

    This is the simple functional interface. Use :class:`TTMSVD` to repeat
    fits of the same tensor or matrix or to access the lightweight result
    object. If several truncation criteria are specified, their most
    restrictive rank is used at every cut and at least one singular value is
    retained.

    A tensorized input can have interleaved shape
    ``(in_1, out_1, ..., in_n, out_n)`` or grouped shape
    ``(in_1, ..., in_n, out_1, ..., out_n)``, selected through ``layout``. A
    two-dimensional matrix can be split into several sites by supplying
    ``in_dim`` and ``out_dim``; their products should match its two axes.
    All routes are normalized internally to the same interleaved order.

    For multiple sites, the first core has shape
    ``(in_1, rank_1, out_1)``, interior cores have shape
    ``(rank_{k-1}, in_k, rank_k, out_k)``, and the final core has shape
    ``(rank_{n-1}, in_n, out_n)``. For one site, the only core has shape
    ``(in_1, out_1)``. These shapes are compatible with
    :class:`~tensorkrowch.models.MPO`.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor or matrix to decompose.
    in_dim : int or sequence[int], optional
        Input dimension per site. Provide it together with ``out_dim`` to
        tensorize a matrix, or omit both arguments to infer dimensions.
    out_dim : int or sequence[int], optional
        Output dimension per site.
    layout : {"interleaved", "grouped"}
        Axis layout of a tensorized input. The default is ``"interleaved"``.
    rank : int, optional
        Number of singular values to keep.
    cutoff : float, optional
        Minimum singular value to keep. It must be non-negative. Singular
        values ``<= cutoff`` are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the accumulated sum of squares is ``<= atol``. It must be non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the tail sum of squares divided by the total sum of squares is
        ``<= rtol``. It must be in ``[0, 1]``.
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent to
        setting ``rtol = 1 - cum_percentage``. It must be in ``[0, 1]``.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
            cum\_percentage

    renormalize : bool
        If ``True``, normalizes the residual before every SVD, accumulates its
        scale logarithmically and evenly redistributes the complete scale over
        the final cores. Absolute criteria and reported errors preserve the
        scale of the original tensor.
    out_device : str or torch.device, optional
        Device where finalized cores are stored. If ``None``, they remain on
        the input device. The default is ``"cpu"``.
    verbose : bool or int
        Console verbosity level:

        - ``0`` or ``False``: no console output;
        - ``1`` or ``True``: phase title, site progress and final summary;
        - ``2``: input configuration and detailed per-site rank, error and
          timing information;
        - ``3``: level 2 output followed by every final core.

    return_info : bool
        If ``True``, also returns ranks, dimensions, metadata and structured
        error and timing metrics. With the default ``False`` and
        ``verbose=0``, diagnostic norm reductions, records and synchronized
        timings are skipped.

    Returns
    -------
    list[torch.Tensor] or tuple
        TTM cores by default. If ``return_info=True``, returns
        ``(cores, info)`` with ranks and structured metrics.

    Examples
    --------
    Decompose a grouped two-site tensor:

    >>> tensor = torch.arange(36.).reshape(2, 2, 3, 3)
    >>> cores = ttm_svd(tensor, layout='grouped', rank=2)
    >>> [tuple(core.shape) for core in cores]
    [(2, 2, 3), (2, 2, 3)]

    Tensorize an ordinary matrix with heterogeneous site dimensions:

    >>> matrix = torch.arange(144.).reshape(12, 12)
    >>> cores = ttm_svd(
    ...     matrix, in_dim=(3, 4), out_dim=(2, 6))
    >>> [tuple(core.shape) for core in cores]
    [(3, 6, 2), (6, 4, 6)]
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    result = TTMSVD(
        tensor=tensor,
        in_dim=in_dim,
        out_dim=out_dim,
        layout=layout,
        out_device=out_device).fit(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            renormalize=renormalize,
            collect_metrics=return_info,
            verbose=verbose)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


def mat_to_mpo(mat: torch.Tensor,
               rank: Optional[int] = None,
               cutoff: Optional[float] = None,
               atol: Optional[float] = None,
               rtol: Optional[float] = None,
               cum_percentage: Optional[float] = None,
               renormalize: bool = False,
               verbose: Union[bool, int] = 0,
               return_info: bool = False):
    r"""Compatibility wrapper for :func:`ttm_svd`.

    .. deprecated:: 1.2
        Use :func:`ttm_svd` for TTM terminology, grouped or matrix layouts,
        explicit output-device policy and repeated fits through
        :class:`TTMSVD`.

    The historical ``mat`` argument and interleaved layout are preserved.
    Final cores remain on the input device, matching the previous behavior.
    ``mat`` should have shape
    ``(in_1, out_1, ..., in_n, out_n)``.

    Parameters
    ----------
    mat : torch.Tensor
        Dense tensor with interleaved input/output dimensions.
    rank : int, optional
        Number of singular values to keep.
    cutoff : float, optional
        Minimum singular value to keep. It must be non-negative. Singular
        values ``<= cutoff`` are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the accumulated sum of squares is ``<= atol``. It must be non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the tail sum of squares divided by the total sum of squares is
        ``<= rtol``. It must be in ``[0, 1]``.
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent to
        setting ``rtol = 1 - cum_percentage``. It must be in ``[0, 1]``.

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
    The canonical replacement keeps the historical interleaved layout:

    >>> tensor = torch.arange(16.).reshape(2, 2, 2, 2)
    >>> cores = ttm_svd(tensor, rank=2)
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

    return ttm_svd(
        tensor=mat,
        layout='interleaved',
        rank=rank,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
        cum_percentage=cum_percentage,
        renormalize=renormalize,
        out_device=None,
        verbose=verbose,
        return_info=return_info)


__all__ = ['TTMSVD', 'ttm_svd', 'mat_to_mpo']
