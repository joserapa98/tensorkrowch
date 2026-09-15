"""
This script contains:

    Class for TRM-SVD decompositions:
        * TRMSVD

    TRM-SVD function:
        * trm_svd
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.decompositions.metrics import _ratio_from_log_norms
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import TRMDecomposition
from tensorkrowch.decompositions.svd._matrix import (_Dimension,
                                                    _prepare_matrix_input)
from tensorkrowch.decompositions.svd.tr import TRSVD
from tensorkrowch.decompositions.svd.utils import (_SVDProgress,
                                                   _log_tensor_norm)


class TRMSVD:
    """Decomposes a fixed dense tensor into a tensor ring matrix.

    The tensor, its input/output dimensions, axis layout and preferred
    interior cut are fixed when this object is created. :meth:`fit` can then
    be called repeatedly with different truncation criteria and centers. Each
    local input/output pair is fused and decomposed by
    :class:`~tensorkrowch.decompositions.TRSVD`, so both algorithms share rank
    discovery, truncation, normalization and SVD-backend semantics.

    A tensorized input can use either ``layout="interleaved"`` with shape
    ``(in_1, out_1, ..., in_n, out_n)`` or ``layout="grouped"`` with shape
    ``(in_1, ..., in_n, out_1, ..., out_n)``. When ``in_dim`` and
    ``out_dim`` are provided, a two-dimensional tensor is instead treated as
    a matrix with shape ``(prod(in_dim), prod(out_dim))`` and tensorized
    internally. At least two TRM sites are required and batch dimensions are
    not supported.

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
    center : int, optional
        Interior cut between sites ``center - 1`` and ``center``. It should
        satisfy ``1 <= center < n_sites``. The default is the middle cut.
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
                 center: Optional[int] = None,
                 *,
                 layout: str = 'interleaved',
                 out_device: Optional[
                     Union[str, torch.device]] = 'cpu') -> None:
        matrix_input = _prepare_matrix_input(
            tensor=tensor,
            in_dim=in_dim,
            out_dim=out_dim,
            layout=layout,
            family='TRM')
        if len(matrix_input.in_dim) < 2:
            raise ValueError('TRM-SVD requires at least two sites')

        self._tensor = tensor
        self._in_dim = matrix_input.in_dim
        self._out_dim = matrix_input.out_dim
        self._interleaved = matrix_input.interleaved
        self._layout = layout
        self._matrix_input = matrix_input.matrix_input
        self._engine = TRSVD(
            matrix_input.fused,
            center=center,
            out_device=out_device)

    @property
    def tensor(self) -> torch.Tensor:
        """Dense tensor or matrix fixed for repeated fits."""
        return self._tensor

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension associated with every TRM site."""
        return self._in_dim

    @property
    def out_dim(self) -> Tuple[int, ...]:
        """Output dimension associated with every TRM site."""
        return self._out_dim

    @property
    def center(self) -> int:
        """Preferred interior cut used when :meth:`fit` omits ``center``."""
        return self._engine.center

    @property
    def layout(self) -> str:
        """Axis layout specified for the fixed input tensor."""
        return self._layout

    def _unfuse_in_out_axes(
            self, cores: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Reopens fused TR input axes into TRM input/output axes."""
        trm_cores = []
        for site, core in enumerate(cores):
            core = core.reshape(
                core.shape[0],
                self.in_dim[site],
                self.out_dim[site],
                core.shape[-1])
            trm_cores.append(core.permute(0, 1, 3, 2))
        return trm_cores

    def fit(self,
            center: Optional[int] = None,
            rank: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            renormalize: bool = False,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0) -> TRMDecomposition:
        r"""Runs TRM-SVD from an interior bipartition of the fixed tensor.

        The active exact SVD backend is selected through
        :func:`tensorkrowch.set_svd_method` or
        :func:`tensorkrowch.svd_method`. When metrics are collected, discarded
        energies are returned as phase-labelled local diagnostics. They are
        not combined into a global reconstruction bound because the successive
        cyclic truncations are not all orthogonal in one common scale.

        If ``rank`` is an integer, it is a shared upper bound for every TRM
        rank and its square bounds the initial bipartition. Local truncation
        criteria may therefore select a different effective rank at each
        site. With ``None``, ranks are discovered from the selected SVD
        dimensions. The selected rank is split with the smallest admissible
        product and the most balanced pair breaks ties. Any extra capacity is
        padded only with structural zeros and recorded in ``result.metadata``.

        Here, ``initial_rank`` is the rank actually selected by the initial
        SVD, whereas ``initial_capacity`` is the product of the two TRM ranks
        used to represent it. Their difference is ``initial_padding``. For
        example, an ``initial_rank`` of 7 may use ranks 2 and 4, giving an
        ``initial_capacity`` of 8 and an ``initial_padding`` of 1.

        The fixed tensor is normalized to interleaved shape
        ``(in_1, out_1, ..., in_n, out_n)`` and each local pair is fused before
        applying the cyclic SVD. Every returned core has shape
        ``(rank_{k-1}, in_k, rank_k, out_k)``, where the right rank of the last
        core matches the left rank of the first core.

        Parameters
        ----------
        center : int, optional
            Interior cut used for this fit. If omitted, the center fixed at
            construction is used.
        rank : int, optional
            Maximum rank allowed at every link. At each subchain SVD cut, at
            most this many singular values are retained. The initial
            bipartition retains at most ``rank ** 2`` singular values before
            its selected rank is factorized into two TRM ranks.
        cutoff : float, optional
            Minimum singular value to keep. It must be finite and
            non-negative. Singular values ``<= cutoff`` are removed.
        atol : float, optional
            Absolute tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded while
            the accumulated sum of squares is ``<= atol``. It must be finite
            and non-negative.
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
            If ``True``, normalizes the residual before every SVD and
            accumulates extracted scales logarithmically. Complete scales are
            redistributed without changing the represented tensor.
        collect_metrics : bool
            If ``True``, collects phase-labelled local truncation records and
            synchronized timings in ``result.metrics``. The default is
            ``False`` to avoid diagnostic norm reductions, records and device
            synchronizations. Metrics are always collected when console output
            is requested.
        verbose : bool or int
            Console verbosity level:

            - ``0`` or ``False``: no console output;
            - ``1`` or ``True``: phase title, cut progress and final summary;
            - ``2``: input configuration and detailed per-cut rank, error and
              timing information;
            - ``3``: level 2 output followed by every final core.

        Returns
        -------
        TRMDecomposition
            Lightweight result containing cores and ranks. Its structured
            metrics are empty unless ``collect_metrics=True`` or
            ``verbose>0``.

        Examples
        --------
        Fix a tensor and compare decompositions with two shared rank caps:

        >>> tensor = torch.arange(36.).reshape(2, 3, 2, 3)
        >>> decomposer = TRMSVD(tensor)
        >>> rank_one = decomposer.fit(rank=1)
        >>> rank_two = decomposer.fit(rank=2)
        >>> rank_one.rank
        [1, 1]
        >>> max(rank_two.rank) <= 2
        True
        """
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        truncation = _TruncationSpec(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage)
        if center is None:
            center = self.center
        else:
            self._engine._validate_center(center, len(self.in_dim))
        verbosity = _normalize_verbosity(verbose)
        emit_events = bool(verbosity)
        collect_metrics = collect_metrics or emit_events
        fit_observer = (
            _resolve_observer(verbosity, None) if emit_events else None)

        if fit_observer is not None:
            fit_observer.emit(DecompositionEvent(
                name='start',
                phase='TRM-SVD',
                values={
                    'sites': len(self.in_dim),
                    'in_dim': self.in_dim,
                    'out_dim': self.out_dim,
                    'layout': self.layout,
                    'matrix_input': self._matrix_input,
                    'center': center,
                    'rank_mode': (
                        'discovery' if truncation.rank is None else 'shared'),
                    'renormalize': renormalize,
                }))
        progress = (
            None if fit_observer is None else _SVDProgress(
                observer=fit_observer,
                phase='TRM-SVD'))
        tr_result = self._engine._fit_validated(
            center=center,
            truncation=truncation,
            renormalize=renormalize,
            collect_metrics=collect_metrics,
            progress=progress)
        result = TRMDecomposition(
            cores=self._unfuse_in_out_axes(tr_result.cores),
            metrics=tr_result.metrics,
            metadata={
                **tr_result.metadata,
                'algorithm': 'trm_svd',
                'layout': self.layout,
                'matrix_input': self._matrix_input,
            })

        if fit_observer is not None:
            initial_padding = result.metadata['initial_padding']
            cycle_rank = result.rank[-1]
            center_rank = result.rank[center - 1]
            initial_capacity = cycle_rank * center_rank
            selected_rank = initial_capacity - initial_padding
            timing = result.metrics.timings[0]
            approximation = result.contract_dense()
            target = self._interleaved.to(
                device=approximation.device, dtype=approximation.dtype)
            abs_log_error = _log_tensor_norm(approximation - target)
            target_log_norm = _log_tensor_norm(target)
            abs_error = abs_log_error.exp()
            rel_error = _ratio_from_log_norms(
                abs_log_error, target_log_norm)
            fit_observer.emit(DecompositionEvent(
                name='summary',
                phase='TRM-SVD',
                values={
                    'rank': result.rank,
                    'center': center,
                    'initial_rank': selected_rank,
                    'initial_capacity': initial_capacity,
                    'initial_padding': initial_padding,
                    'absolute_error': abs_error,
                    'relative_error': rel_error,
                    'elapsed': timing.elapsed,
                }))
            for site, core in enumerate(result.cores):
                fit_observer.emit(DecompositionEvent(
                    name='core',
                    phase='TRM-SVD',
                    level=3,
                    site=site,
                    values={'shape': tuple(core.shape), 'tensor': core}))
            fit_observer.close(result.metrics)
        return result


def trm_svd(tensor: torch.Tensor,
            in_dim: _Dimension = None,
            out_dim: _Dimension = None,
            center: Optional[int] = None,
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
    r"""Decomposes a dense tensor or matrix into cyclic TRM cores.

    This is the simple functional interface. Use :class:`TRMSVD` to repeat
    fits of the same tensor or matrix or to access the lightweight result
    object. The input/output pair of every site is fused, decomposed through
    TR-SVD and reopened without another factorization.

    A tensorized input can have interleaved shape
    ``(in_1, out_1, ..., in_n, out_n)`` or grouped shape
    ``(in_1, ..., in_n, out_1, ..., out_n)``, selected through ``layout``. A
    two-dimensional matrix can be split into several sites by supplying
    ``in_dim`` and ``out_dim``; their products should match its two axes. At
    least two sites are required.

    Every returned core has shape
    ``(rank_{k-1}, in_k, rank_k, out_k)`` and the last right rank matches the
    first left rank. An integer ``rank`` is a shared upper bound, while local
    truncation may select different effective ranks across the ring.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor or matrix to decompose.
    in_dim : int or sequence[int], optional
        Input dimension per site. Provide it together with ``out_dim`` to
        tensorize a matrix, or omit both arguments to infer dimensions.
    out_dim : int or sequence[int], optional
        Output dimension per site.
    center : int, optional
        Interior cut satisfying ``1 <= center < n_sites``. The default is the
        middle cut.
    layout : {"interleaved", "grouped"}
        Axis layout of a tensorized input. The default is ``"interleaved"``.
    rank : int, optional
        Maximum rank allowed at every link. At each subchain SVD cut, at most
        this many singular values are retained. The initial bipartition
        retains at most ``rank ** 2`` singular values before its selected rank
        is factorized into two TRM ranks.
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
        - ``1`` or ``True``: phase title, cut progress and final summary;
        - ``2``: input configuration and detailed per-cut rank, error and
          timing information;
        - ``3``: level 2 output followed by every final core.

    return_info : bool
        If ``True``, also returns ranks, dimensions, rank factorization,
        metadata and structured local metrics. With the default ``False`` and
        ``verbose=0``, diagnostic norm reductions, records and synchronized
        timings are skipped.

    Returns
    -------
    list[torch.Tensor] or tuple
        TRM cores by default. If ``return_info=True``, returns ``(cores, info)``
        with ranks and structured metrics.

    Examples
    --------
    Decompose a grouped two-site tensor with a shared rank cap:

    >>> tensor = torch.arange(36.).reshape(2, 2, 3, 3)
    >>> cores = trm_svd(tensor, layout='grouped', rank=2)
    >>> [tuple(core.shape) for core in cores]
    [(2, 2, 2, 3), (2, 2, 2, 3)]

    Tensorize an ordinary matrix with heterogeneous site dimensions:

    >>> matrix = torch.arange(144.).reshape(12, 12)
    >>> cores = trm_svd(
    ...     matrix, in_dim=(3, 4), out_dim=(2, 6), rank=3)
    >>> len(cores)
    2
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    result = TRMSVD(
        tensor=tensor,
        in_dim=in_dim,
        out_dim=out_dim,
        center=center,
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


__all__ = ['TRMSVD', 'trm_svd']
