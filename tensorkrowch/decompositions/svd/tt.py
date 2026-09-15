"""
This script contains:

    Internal TT-SVD state classes:
        * _TTSVDErrorState
        * _TTSVDFitContext
        * _TTSVDSplit

    Class for TT-SVD decompositions:
        * TTSVD

    TT-SVD function:
        * tt_svd
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import warnings

import torch

from tensorkrowch.utils import truncated_svd

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 ErrorRecord,
                                                 TimingRecord,
                                                 TruncationRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.svd.utils import (_SVDProgress,
                                                   _log_tensor_norm,
                                                   _normalize_tensor)


@dataclass
class _TTSVDErrorState:
    """Holds error quantities that are only needed for diagnostics."""

    norm: torch.Tensor  # Input norm, optionally resolved by batch
    log_norm: torch.Tensor  # Stable logarithm of the input norm
    relative_sq_error: torch.Tensor  # Accumulated squared relative error


@dataclass
class _TTSVDFitContext:
    """Holds numerical state local to one TT-SVD fit."""

    batch_shape: Tuple[int, ...]  # Leading dimensions treated as batches
    in_dim: Tuple[int, ...]  # Input dimensions represented by TT sites
    truncation: _TruncationSpec  # Shared truncation criteria for every cut
    renormalize: bool  # Whether residual norms are extracted before SVDs
    log_scale: torch.Tensor  # Accumulated logarithmic residual scale
    error_state: Optional[_TTSVDErrorState]  # Optional metric state


@dataclass
class _TTSVDSplit:
    """Contains the outputs and diagnostics of one TT-SVD cut."""

    core: torch.Tensor  # Finalized TT core produced at this cut
    residual: torch.Tensor  # Tensor passed to the next cut
    selected_rank: int  # Rank retained at this cut
    record: Optional[TruncationRecord]  # Optional truncation diagnostics


class TTSVD:
    """Decomposes a fixed dense tensor into a tensor train.

    The tensor and its batch dimensions are fixed when this object is created.
    :meth:`fit` can then be called repeatedly with different truncation
    criteria. Each fit is independent and returns a lightweight
    :class:`~tensorkrowch.decompositions.TTDecomposition`.

    The numerical sweep follows the TT-SVD algorithm introduced by
    I. V. Oseledets in *Tensor-Train Decomposition* (2011), `paper
    <https://doi.org/10.1137/090752286>`_.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor with optional leading batch dimensions.
    n_batches : int
        Number of leading batch dimensions. At least one non-batch dimension
        must remain.
    out_device : str or torch.device, optional
        Device where finalized cores are stored. The default is ``"cpu"``.
        If ``None``, cores remain on the tensor's device.
    """

    def __init__(self,
                 tensor: torch.Tensor,
                 n_batches: int = 0,
                 *,
                 out_device: Optional[
                     Union[str, torch.device]] = 'cpu') -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if (n_batches < 0) or (n_batches >= tensor.ndim):
            raise ValueError(
                '`n_batches` should leave at least one tensor dimension')
        if any(dim < 1 for dim in tensor.shape):
            raise ValueError(
                'TT input and batch dimensions should be positive')

        self._tensor = tensor
        self._n_batches = n_batches
        self._runtime = _RuntimePolicy.from_tensor(
            tensor, out_device=out_device)

    @property
    def tensor(self) -> torch.Tensor:
        """Dense tensor fixed for repeated fits."""
        return self._tensor

    @property
    def n_batches(self) -> int:
        """Number of leading batch dimensions."""
        return self._n_batches

    def _split_site(self,
                    residual: torch.Tensor,
                    site: int,
                    previous_rank: int,
                    context: _TTSVDFitContext) -> _TTSVDSplit:
        """Splits one site and updates its error and normalization state."""
        residual = residual.reshape(
            *context.batch_shape,
            previous_rank * context.in_dim[site],
            -1)
        if context.renormalize:
            residual, log_norm = _normalize_tensor(residual)
            log_norm = torch.where(
                torch.isneginf(log_norm),
                torch.zeros_like(log_norm),
                log_norm)
            context.log_scale = context.log_scale + log_norm

        cut_log_scale = context.log_scale
        if context.renormalize and \
                context.truncation.requires_absolute_rescaling:
            truncation_kwargs = context.truncation.as_normalized_kwargs(
                cut_log_scale.detach().cpu().item())
        else:
            truncation_kwargs = context.truncation.as_kwargs()
        collect_metrics = context.error_state is not None
        svd_result = truncated_svd(
            tensor=residual,
            return_info=collect_metrics,
            **truncation_kwargs)
        if collect_metrics:
            u, s, vh, info = svd_result
        else:
            u, s, vh = svd_result
        selected_rank = s.shape[-1]
        if site:
            u = u.reshape(
                *context.batch_shape,
                previous_rank,
                context.in_dim[site],
                selected_rank)

        record = None
        if context.error_state is not None:
            error_state = context.error_state
            discarded_log_norm = \
                info.discarded_sq_norm.log() / 2
            if context.renormalize:
                discarded_log_norm = discarded_log_norm + cut_log_scale
            positive_input = error_state.norm > 0
            safe_log_norm = torch.where(
                positive_input,
                error_state.log_norm,
                torch.zeros_like(error_state.log_norm))
            rel_contribution = (
                discarded_log_norm - safe_log_norm).exp()
            zero_input_contribution = torch.where(
                torch.isneginf(discarded_log_norm),
                torch.zeros_like(discarded_log_norm),
                torch.full_like(discarded_log_norm, torch.inf))
            rel_contribution = torch.where(
                positive_input,
                rel_contribution,
                zero_input_contribution)
            if not torch.isfinite(rel_contribution).all():
                raise ValueError(
                    'The relative TT-SVD truncation error should be finite')
            error_state.relative_sq_error = (
                error_state.relative_sq_error + rel_contribution.square())

            record = TruncationRecord.from_svd_info(
                info,
                site=site,
                log_scale=(
                    cut_log_scale.expand(context.batch_shape)
                    if self._n_batches and context.renormalize
                    else cut_log_scale if context.renormalize else None),
                global_norm=error_state.norm)
        core = self._runtime.finalize(u)
        residual = s.to(vh.dtype).unsqueeze(-1) * vh
        return _TTSVDSplit(
            core=core,
            residual=residual,
            selected_rank=selected_rank,
            record=record)

    def _redistribute_scale(
            self,
            cores: List[torch.Tensor],
            context: _TTSVDFitContext) -> List[torch.Tensor]:
        """Redistributes the extracted norm and finalizes all pending cores."""
        if not context.renormalize:
            cores[-1] = self._runtime.finalize(cores[-1])
            return cores

        cores[-1], log_norm = _normalize_tensor(cores[-1])
        log_norm = torch.where(
            torch.isneginf(log_norm),
            torch.zeros_like(log_norm),
            log_norm)
        context.log_scale = context.log_scale + log_norm

        cores[-1] = self._runtime.finalize(cores[-1])
        core_scale = (context.log_scale / len(cores)).exp()
        if not torch.isfinite(core_scale).all():
            raise ValueError('The final TT-SVD scale should be finite')
        core_scale = self._runtime.finalize(core_scale)
        return [core * core_scale for core in cores]

    def _fit_validated(
            self,
            truncation: _TruncationSpec,
            renormalize: bool,
            collect_metrics: bool,
            progress: Optional[_SVDProgress] = None) -> TTDecomposition:
        """Runs TT-SVD from already validated fit options."""
        tensor = self._runtime.prepare(self._tensor)
        batch_shape = tuple(tensor.shape[:self._n_batches])
        in_dim = tuple(tensor.shape[self._n_batches:])
        n_sites = len(in_dim)

        error_state = None
        if collect_metrics:
            input_axes = tuple(range(self._n_batches, tensor.ndim))
            log_norm = _log_tensor_norm(tensor, dim=input_axes)
            norm = log_norm.exp()
            if not torch.isfinite(norm).all():
                raise ValueError('The input tensor norm should be finite')
            error_state = _TTSVDErrorState(
                norm=norm,
                log_norm=log_norm,
                relative_sq_error=torch.zeros_like(norm))
        context = _TTSVDFitContext(
            batch_shape=batch_shape,
            in_dim=in_dim,
            truncation=truncation,
            renormalize=renormalize,
            log_scale=tensor.real.new_zeros(()),
            error_state=error_state)

        metrics = DecompositionMetrics()
        cores = []
        cut_timings = []

        total_timer_context = (
            self._runtime.timer() if collect_metrics else nullcontext())
        with total_timer_context as total_timer:
            residual = tensor
            previous_rank = 1
            for site in range(n_sites - 1):
                cut_timer_context = (
                    self._runtime.timer() if collect_metrics else nullcontext())
                with cut_timer_context as cut_timer:
                    split = self._split_site(
                        residual=residual,
                        site=site,
                        previous_rank=previous_rank,
                        context=context)

                cores.append(split.core)
                residual = split.residual
                previous_rank = split.selected_rank
                if collect_metrics:
                    metrics.truncations.append(split.record)
                    cut_timing = TimingRecord(
                        name='svd_cut',
                        elapsed=cut_timer.elapsed,
                        site=site)
                    cut_timings.append(cut_timing)
                if progress is not None:
                    progress.cut_complete(
                        site=site,
                        record=split.record,
                        elapsed=cut_timer.elapsed)

            cores.append(residual)
            cores = self._redistribute_scale(cores, context)

        if error_state is not None:
            rel_error = error_state.relative_sq_error.sqrt()
            abs_error = rel_error * error_state.norm
            metrics.errors.append(ErrorRecord(
                kind='truncation',
                absolute=abs_error,
                relative=rel_error,
                size=n_sites - 1,
                denominator=error_state.norm))
            metrics.timings.append(TimingRecord(
                name='fit',
                elapsed=total_timer.elapsed,
                children=cut_timings))

        result = TTDecomposition(
            cores=cores,
            metrics=metrics,
            metadata={
                'algorithm': 'tt_svd',
                'renormalize': renormalize,
            },
            n_batches=self._n_batches)
        return result

    def fit(self,
            rank: Optional[int] = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            renormalize: bool = False,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0) -> TTDecomposition:
        r"""Runs TT-SVD with a shared truncation policy at every cut.

        The active exact SVD backend is selected through
        :func:`tensorkrowch.set_svd_method` or
        :func:`tensorkrowch.svd_method`. When metrics are collected, local
        discarded energies and their accumulated absolute and relative
        reconstruction errors are returned in ``result.metrics``.

        If several truncation criteria are specified, each one provides an
        upper bound for the selected rank and the most restrictive bound is
        used. At least one singular value is always retained. The same
        criteria are applied at every TT cut.

        The fixed tensor should have shape
        ``(*batch_shape, d_1, ..., d_n)``, where the first ``n_batches`` axes
        are optional batch dimensions and each remaining axis is the input
        dimension of one TT site. Thus, the desired number and input
        dimensions of the cores are specified directly by the non-batch shape
        of the tensor; reshape the tensor before constructing :class:`TTSVD`
        if a different site factorization is required.

        For more than one site, the returned cores have shapes
        ``(*batch_shape, d_1, rank_1)``,
        ``(*batch_shape, rank_{k-1}, d_k, rank_k)`` at interior sites, and
        ``(*batch_shape, rank_{n-1}, d_n)`` at the last site. A one-site tensor
        is returned as a single core with its original shape. Consequently,
        the result always represents an open-boundary TT.

        Parameters
        ----------
        rank : int, optional
            Maximum rank allowed at every link. At each SVD cut, at most this
            many singular values are retained.
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
            - ``1`` or ``True``: phase title, input configuration, cut
              progress and final summary;
            - ``2``: detailed per-cut rank, error and timing information;
            - ``3``: level 2 output followed by every final core.

        Returns
        -------
        TTDecomposition
            Lightweight result containing cores and ranks. Its structured
            metrics are empty unless ``collect_metrics=True`` or
            ``verbose>0``.

        Examples
        --------
        Fix a tensor once and compare decompositions with different maximum
        ranks:

        >>> tensor = torch.arange(24.).reshape(2, 3, 4)
        >>> decomposer = TTSVD(tensor)
        >>> rank_one = decomposer.fit(rank=1)
        >>> rank_two = decomposer.fit(rank=2, collect_metrics=True)
        >>> rank_one.rank
        [1, 1]
        >>> rank_two.rank
        [2, 2]
        >>> len(rank_two.metrics.truncations)
        2
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
        verbosity = _normalize_verbosity(verbose)
        emit_events = bool(verbosity)
        collect_metrics = collect_metrics or emit_events
        fit_observer = (
            _resolve_observer(verbosity, None) if emit_events else None)

        if fit_observer is not None:
            batch_shape = tuple(self.tensor.shape[:self.n_batches])
            in_dim = tuple(self.tensor.shape[self.n_batches:])
            fit_observer.emit(DecompositionEvent(
                name='start',
                phase='TT-SVD',
                values={
                    'sites': len(in_dim),
                    'batch_shape': batch_shape,
                    'in_dim': in_dim,
                    'renormalize': renormalize,
                }))
            progress = _SVDProgress(
                observer=fit_observer,
                phase='TT-SVD')
        else:
            progress = None
        result = self._fit_validated(
            truncation=truncation,
            renormalize=renormalize,
            collect_metrics=collect_metrics,
            progress=progress)

        if fit_observer is not None:
            error = result.metrics.errors[0]
            timing = result.metrics.timings[0]
            abs_error = error.absolute
            rel_error = error.relative
            if abs_error.ndim:
                abs_error = torch.linalg.vector_norm(abs_error)
                global_norm = torch.linalg.vector_norm(error.denominator)
                rel_error = torch.where(
                    global_norm > 0,
                    abs_error / global_norm,
                    torch.zeros_like(abs_error))
            fit_observer.emit(DecompositionEvent(
                name='summary',
                phase='TT-SVD',
                values={
                    'rank': result.rank,
                    'absolute_error': abs_error,
                    'relative_error': rel_error,
                    'elapsed': timing.elapsed,
                }))
            for site, core in enumerate(result.cores):
                fit_observer.emit(DecompositionEvent(
                    name='core',
                    phase='TT-SVD',
                    level=3,
                    site=site,
                    values={'shape': tuple(core.shape), 'tensor': core}))
            fit_observer.close(result.metrics)
        return result


def tt_svd(tensor: torch.Tensor,
           n_batches: int = 0,
           rank: Optional[int] = None,
           cutoff: Optional[float] = None,
           atol: Optional[float] = None,
           rtol: Optional[float] = None,
           cum_percentage: Optional[float] = None,
           renormalize: bool = False,
           out_device: Optional[Union[str, torch.device]] = 'cpu',
           verbose: Union[bool, int] = 0,
           return_info: bool = False):
    r"""Decomposes a dense tensor into TT cores by consecutive SVDs.

    This is the simple functional interface. Use :class:`TTSVD` to repeat
    fits of the same tensor or to access the lightweight result object. If
    several truncation criteria are specified, their most restrictive rank is
    used at every cut and at least one singular value is retained.

    The input should have shape ``(*batch_shape, d_1, ..., d_n)``. The first
    ``n_batches`` axes are interpreted as optional batch dimensions, while
    every remaining axis defines the input dimension of one TT site. The
    function therefore returns ``n`` open-boundary cores. To obtain a TT with
    a particular sequence of input dimensions, reshape ``tensor`` to those
    dimensions before calling this function.

    For multiple sites, the first core has shape
    ``(*batch_shape, d_1, rank_1)``, interior cores have shape
    ``(*batch_shape, rank_{k-1}, d_k, rank_k)``, and the final core has shape
    ``(*batch_shape, rank_{n-1}, d_n)``. For one site, the only core has the
    same shape as the input tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor whose optional leading batch axes are followed by one
        input dimension per TT site.
    n_batches : int
        Number of leading tensor axes interpreted as batch dimensions. At
        least one non-batch input dimension should remain.
    rank : int, optional
        Maximum rank allowed at every link. At each SVD cut, at most this many
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
        scale of the original tensor.
    out_device : str or torch.device, optional
        Device where finalized cores are stored. If ``None``, they remain on
        the input device. The default is ``"cpu"``.
    verbose : bool or int
        Console verbosity level:

        - ``0`` or ``False``: no console output;
        - ``1`` or ``True``: phase title, input configuration, cut progress
          and final summary;
        - ``2``: detailed per-cut rank, error and timing information;
        - ``3``: level 2 output followed by every final core.

    return_info : bool
        If ``True``, also returns ranks, dimensions, metadata and structured
        error and timing metrics. With the default ``False`` and
        ``verbose=0``, diagnostic norm reductions, records and synchronized
        timings are skipped.

    Returns
    -------
    list[torch.Tensor] or tuple
        TT cores by default. If ``return_info=True``, returns
        ``(cores, info)`` with ranks and structured metrics.

    Examples
    --------
    Decompose a four-site tensor and inspect the resulting core shapes:

    >>> tensor = torch.arange(16.).reshape(2, 2, 2, 2)
    >>> cores = tt_svd(tensor, rank=2)
    >>> [tuple(core.shape) for core in cores]
    [(2, 2), (2, 2, 2), (2, 2, 2), (2, 2)]

    Request structured ranks, errors and timings when they are needed:

    >>> cores, info = tt_svd(
    ...     tensor, rank=2, return_info=True)
    >>> info['rank']
    [2, 2, 2]
    >>> len(info['metrics']['truncations'])
    3
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    result = TTSVD(
        tensor=tensor,
        n_batches=n_batches,
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


def vec_to_mps(vec: torch.Tensor,
               n_batches: int = 0,
               rank: Optional[int] = None,
               cutoff: Optional[float] = None,
               atol: Optional[float] = None,
               rtol: Optional[float] = None,
               cum_percentage: Optional[float] = None,
               renormalize: bool = False,
               verbose: Union[bool, int] = 0,
               return_info: bool = False):
    r"""Compatibility wrapper for :func:`tt_svd`.

    .. deprecated:: 1.2
        Use :func:`tt_svd` for TT terminology, explicit output-device policy
        and repeated fits through :class:`TTSVD`.

    The historical ``vec`` and ``n_batches`` arguments are preserved. Final
    cores remain on the input device, matching the previous behavior. The
    tensor should have shape ``(*batch_shape, d_1, ..., d_n)``, where each
    non-batch axis is the input dimension of one TT site.

    Parameters
    ----------
    vec : torch.Tensor
        Dense tensor to decompose.
    n_batches : int
        Number of leading tensor axes interpreted as batch dimensions.
    rank : int, optional
        Maximum rank allowed at every link. At each SVD cut, at most this many
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
        Console verbosity level forwarded to :func:`tt_svd`.
    return_info : bool
        If ``True``, returns ``(cores, info)`` with ranks, dimensions,
        metadata and structured metrics.

    Returns
    -------
    list[torch.Tensor] or tuple
        TT cores, optionally followed by their structured information.

    Examples
    --------
    The canonical replacement only changes the function and tensor argument
    names:

    >>> tensor = torch.arange(16.).reshape(2, 2, 2, 2)
    >>> cores = tt_svd(tensor, rank=2)
    >>> [tuple(core.shape) for core in cores]
    [(2, 2), (2, 2, 2), (2, 2, 2), (2, 2)]
    """
    warnings.warn(
        '`vec_to_mps` is deprecated; use `tt_svd` instead',
        FutureWarning,
        stacklevel=2)
    if not isinstance(vec, torch.Tensor):
        raise TypeError('`vec` should be torch.Tensor type')
    if isinstance(n_batches, bool) or not isinstance(n_batches, int):
        raise TypeError('`n_batches` should be int type')
    if (n_batches < 0) or (n_batches >= vec.ndim):
        raise ValueError(
            '`n_batches` should be between 0 and the rank of `vec`')

    return tt_svd(
        tensor=vec,
        n_batches=n_batches,
        rank=rank,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
        cum_percentage=cum_percentage,
        renormalize=renormalize,
        out_device=None,
        verbose=verbose,
        return_info=return_info)


__all__ = ['TTSVD', 'tt_svd', 'vec_to_mps']
