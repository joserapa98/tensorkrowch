"""
This script contains:

    Internal TR-SVD rank policy:
        * _TRRankPolicy

    Class for TR-SVD decompositions:
        * TRSVD

    TR-SVD function:
        * tr_svd
        * qtr_svd
"""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from math import prod
from typing import List, Optional, Tuple, Union

import torch

from tensorkrowch.utils import log_tensor_norm, ratio_from_log

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 TimingRecord,
                                                 TruncationRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import (TensorDecomposition,
                                                 TRDecomposition,
                                                 QTRDecomposition,
                                                 _result_from_source)

from tensorkrowch.decompositions.sources.factory import SourceLike
from tensorkrowch.decompositions.sources.quantics import CallableQuanticsSource

from tensorkrowch.decompositions.svd.tt import TTSVD
from tensorkrowch.decompositions.svd._utils import (_SVDProgress,
                                                   _prepare_svd_source)


_Rank = Optional[int]


@dataclass(frozen=True)
class _TRRankPolicy:  # MARK: _TRRankPolicy
    """Stores the shared rank constraints for one TR-SVD fit."""

    mode: str  # Rank discovery or shared-cap policy
    initial_cap: Optional[int]  # Cap for the initial bipartition rank
    rank_cap: Optional[int]  # Cap for every final TR link


class TRSVD:  # MARK: TRSVD
    """
    Decomposes a fixed dense tensor into a tensor ring.

    The tensor and a preferred interior cut are fixed when this object is
    created. :meth:`fit` can then be called repeatedly with different
    truncation criteria, centers and rank policies. Each fit is independent
    and returns a lightweight
    :class:`~tensorkrowch.decompositions.TRDecomposition`.

    The algorithm first reshapes the tensor across an interior bipartition and
    computes one SVD. Its selected rank is represented as the product of the
    cyclic rank and the rank crossing the interior cut. The two resulting
    blocks are then decomposed by the shared TT-SVD engine while preserving
    those boundary ranks. No batch dimensions are supported.

    This construction is based on TR-SVD by Q. Zhao et al. in *Tensor Ring
    Decomposition* (2016), `paper <https://arxiv.org/abs/1606.05535>`_. The
    implementation uses an interior bipartition followed by two TT-SVD
    subchains, rather than the one-direction sequence in the original
    algorithm.

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
    source : TensorSource, TensorFormat1D or torch.Tensor
        Finite tensor input, with shape and evaluation defined by the source.
    center : int, optional
        Interior cut between sites ``center - 1`` and ``center``. It should
        satisfy ``1 <= center < tensor.ndim``. The default is the middle cut.
    out_device : str or torch.device, optional
        Device where finalized cores are stored. The default is ``"cpu"``.
        If ``None``, cores remain on the tensor's device.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(torch.ones(2, 2, 2))
    >>> result = tk.decompositions.TRSVD(source).fit(rank=1)
    >>> result.rank
    [1, 1, 1]
    """

    def __init__(self,
                 source: 'SourceLike',
                 center: Optional[int] = None,
                 *,
                 out_device: Optional[Union[str, torch.device]] = 'cpu') -> None:
        self._source, tensor, batches = _prepare_svd_source(source)
        if batches:
            raise ValueError('TR-SVD does not support `n_batches`')
        if tensor.ndim < 2:
            raise ValueError('The source should contain at least two TR sites')
        center = tensor.ndim // 2 if center is None else center
        self._validate_center(center, tensor.ndim)
        self._center = center
        self._tensor = tensor
        self._runtime = _RuntimePolicy.from_tensor(tensor,
                                                   out_device=out_device)
        self._phase = 'TRM-SVD' if getattr(
            self._source, 'is_matrix', False) else 'TR-SVD'

    @property
    def source(self) -> SourceLike:
        """Original source describing the fixed decomposition input."""
        return self._source

    @staticmethod
    def _validate_center(center: int, n_sites: int) -> None:
        """Validates that a ``center`` identifies an interior chain cut."""
        if isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int type')
        if (center < 1) or (center >= n_sites):
            raise ValueError(
                '`center` should satisfy 1 <= center < tensor.ndim')

    @property
    def center(self) -> int:
        """Preferred interior cut used when :meth:`fit` omits ``center``."""
        return self._center

    @staticmethod
    def _resolve_rank_policy(rank: _Rank) -> _TRRankPolicy:
        """Builds the TR ``rank`` policy from a validated shared ``rank``."""

        if rank is None:
            return _TRRankPolicy(
                mode='discovery',
                initial_cap=None,
                rank_cap=None)

        return _TRRankPolicy(
            mode='shared',
            initial_cap=rank * rank,
            rank_cap=rank)

    @staticmethod
    def _split_cycle_rank(selected_rank: int,
                          rank_cap: Optional[int]) -> Tuple[int, int]:
        """Finds the smallest admissible cyclic/interior rank capacity."""
        if rank_cap is None:
            rank_cap = selected_rank

        candidates = []
        for cycle_rank in range(1, rank_cap + 1):
            center_rank = (
                selected_rank + cycle_rank - 1) // cycle_rank
            if center_rank <= rank_cap:
                capacity = cycle_rank * center_rank
                candidates.append((
                    capacity,
                    abs(cycle_rank - center_rank),
                    cycle_rank,
                    center_rank))

        _, _, cycle_rank, center_rank = min(candidates)
        return cycle_rank, center_rank

    def _decompose_subchain(self,
                            subchain: torch.Tensor,
                            in_dim: Tuple[int, ...],
                            truncation: _TruncationSpec,
                            renormalize: bool,
                            collect_metrics: bool,
                            progress: Optional[_SVDProgress] = None
                            ) -> Tuple[List[torch.Tensor],
                                       Optional[DecompositionMetrics]]:
        """Applies TT-SVD while preserving a ``subchain``'s boundary ranks."""
        left_rank = subchain.shape[0]
        right_rank = subchain.shape[-1]
        if len(in_dim) == 1:
            core = subchain.reshape(left_rank, in_dim[0], right_rank)
            return [self._runtime.finalize(core)], None

        fused = subchain.reshape(
            left_rank * in_dim[0],
            *in_dim[1:-1],
            in_dim[-1] * right_rank)
        engine = TTSVD(
            fused,
            out_device=self._runtime.out_device)
        result = engine._fit_validated(
            truncation=truncation,
            renormalize=renormalize,
            collect_metrics=collect_metrics,
            progress=progress)

        cores = list(result.cores)
        metrics = result.metrics if collect_metrics else None
        cores[0] = cores[0].reshape(
            left_rank, in_dim[0], cores[0].shape[-1])
        cores[-1] = cores[-1].reshape(
            cores[-1].shape[0], in_dim[-1], right_rank)
        return cores, metrics

    @staticmethod
    def _phase_truncations(metrics: DecompositionMetrics,
                           phase: str,
                           site_offset: int) -> List[TruncationRecord]:
        """Labels TT-SVD records as local TR-SVD diagnostics."""
        return [
            replace(
                record,
                site=site_offset + record.site,
                phase=phase,
                global_rel_contribution=None)
            for record in metrics.truncations
        ]

    @staticmethod
    def _phase_timing(metrics: DecompositionMetrics,
                      name: str,
                      site_offset: int) -> TimingRecord:
        """Relabels one nested TT-SVD timing phase with global sites."""
        timing = metrics.timings[0]
        children = [
            replace(
                child,
                site=(None if child.site is None
                      else site_offset + child.site))
            for child in timing.children
        ]
        return replace(timing, name=name, children=children)

    def _fit_validated(self,
                       center: int,
                       truncation: _TruncationSpec,
                       renormalize: bool,
                       collect_metrics: bool,
                       progress: Optional[_SVDProgress] = None
                       ) -> TRDecomposition:
        """Runs the cyclic SVD kernel from already validated fit options."""
        rank_policy = self._resolve_rank_policy(rank=truncation.rank)
        initial_truncation = (
            truncation if rank_policy.initial_cap == truncation.rank else
            replace(truncation, rank=rank_policy.initial_cap))
        in_dim = tuple(self._tensor.shape)

        total_timer_context = (
            self._runtime.timer() if collect_metrics else nullcontext())
        with total_timer_context as total_timer:
            tensor = self._runtime.prepare(self._tensor)
            left_size = prod(in_dim[:center])
            right_size = prod(in_dim[center:])
            matrix = tensor.reshape(left_size, right_size)
            initial_result = TTSVD(
                matrix,
                out_device=None)._fit_validated(
                    truncation=initial_truncation,
                    renormalize=renormalize,
                    collect_metrics=collect_metrics)
            selected_rank = initial_result.rank[0]

            cycle_rank, center_rank = self._split_cycle_rank(
                selected_rank=selected_rank,
                rank_cap=rank_policy.rank_cap)
            initial_capacity = cycle_rank * center_rank
            initial_padding = initial_capacity - selected_rank

            left_factor, right_factor = initial_result.cores
            initial_metrics = (
                initial_result.metrics if collect_metrics else None)
            del initial_result

            if progress is not None:
                initial_record = initial_metrics.truncations[0]
                initial_timing = initial_metrics.timings[0]
                progress.observer.emit(DecompositionEvent(
                    name='bipartition_complete',
                    phase=progress.phase,
                    elapsed=initial_timing.elapsed,
                    values={
                        'blocks': (
                            tuple(range(1, center + 1)),
                            tuple(range(center + 1, len(in_dim) + 1))),
                        'full_rank': initial_record.full_rank,
                        'selected_rank': selected_rank,
                        'cycle_rank': cycle_rank,
                        'center_rank': center_rank,
                        'absolute_error': initial_record.local_abs_error,
                        'relative_error': initial_record.local_rel_error,
                    }))
            if initial_padding:
                left_factor = torch.cat([
                    left_factor,
                    left_factor.new_zeros(
                        left_factor.shape[0], initial_padding),
                ], dim=-1)
                right_factor = torch.cat([
                    right_factor,
                    right_factor.new_zeros(
                        initial_padding, right_factor.shape[1]),
                ], dim=0)

            left_subchain = left_factor.reshape(
                *in_dim[:center], cycle_rank, center_rank)
            left_subchain = left_subchain.movedim(-2, 0)
            right_subchain = right_factor.reshape(
                cycle_rank, center_rank, *in_dim[center:])
            right_subchain = right_subchain.movedim(0, -1)

            left_progress = (
                None if progress is None else replace(
                    progress,
                    subphase='left_subchain'))
            right_progress = (
                None if progress is None else replace(
                    progress,
                    site_offset=center,
                    subphase='right_subchain'))
            left_cores, left_metrics = self._decompose_subchain(
                subchain=left_subchain,
                in_dim=in_dim[:center],
                truncation=truncation,
                renormalize=renormalize,
                collect_metrics=collect_metrics,
                progress=left_progress)
            right_cores, right_metrics = self._decompose_subchain(
                subchain=right_subchain,
                in_dim=in_dim[center:],
                truncation=truncation,
                renormalize=renormalize,
                collect_metrics=collect_metrics,
                progress=right_progress)

        metrics = DecompositionMetrics()
        if collect_metrics:
            initial_records = self._phase_truncations(
                initial_metrics,
                phase='initial_bipartition',
                site_offset=center - 1)
            left_records = (
                [] if left_metrics is None else self._phase_truncations(
                    left_metrics,
                    phase='left_subchain',
                    site_offset=0))
            right_records = (
                [] if right_metrics is None else self._phase_truncations(
                    right_metrics,
                    phase='right_subchain',
                    site_offset=center))
            metrics.truncations.extend(
                left_records + initial_records + right_records)

            phase_timings = [
                self._phase_timing(
                    initial_metrics, 'initial_bipartition', center - 1),
            ]
            if left_metrics is not None:
                phase_timings.append(self._phase_timing(
                    left_metrics, 'left_subchain', 0))
            if right_metrics is not None:
                phase_timings.append(self._phase_timing(
                    right_metrics, 'right_subchain', center))
            metrics.timings.append(TimingRecord(
                name='fit',
                elapsed=total_timer.elapsed,
                children=phase_timings))
            if initial_padding:
                metrics.warnings.append(
                    f'Initial SVD rank {selected_rank} uses capacity '
                    f'{initial_capacity} with {initial_padding} structural '
                    'zero dimensions')
            metrics.warnings.append(
                'Cyclic SVD truncation records are local diagnostics and are '
                'not combined into a global reconstruction bound')

        return TRDecomposition(
            cores=left_cores + right_cores,
            metrics=metrics,
            metadata={
                'algorithm': 'tr_svd',
                'center': center,
                'rank_mode': rank_policy.mode,
                'renormalize': renormalize,
                'initial_padding': initial_padding,
            })

    def fit(self,
            center: Optional[int] = None,
            rank: _Rank = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            renormalize: bool = False,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0) -> 'TensorDecomposition':
        r"""
        Runs TR-SVD from an interior bipartition of the fixed tensor.

        The active exact SVD backend is selected through
        :func:`tensorkrowch.set_svd_method` or
        :func:`tensorkrowch.svd_method`. When metrics are collected, discarded
        energies are returned as phase-labelled local diagnostics. They are
        not combined into a global reconstruction bound because the successive
        TR-SVD truncations are not all orthogonal in one common scale.

        If ``rank`` is an integer, it is a shared upper bound for every TR
        ``rank`` and its square bounds the initial bipartition. Local truncation
        criteria may therefore select a different effective ``rank`` at each
        site. With ``None``, ranks are discovered from the selected SVD
        dimensions. The selected ``rank`` is split with the smallest admissible
        product and the most balanced pair breaks ties. Any extra capacity
        required by the cap is padded only with structural zeros and recorded
        in ``result.metadata``.

        Here, ``initial_rank`` is the ``rank`` actually selected by the initial
        SVD, whereas ``initial_capacity`` is the product of the two TR ranks
        used to represent it. Their difference is ``initial_padding``. For
        example, an ``initial_rank`` of 7 may use ranks 2 and 4, giving an
        ``initial_capacity`` of 8 and an ``initial_padding`` of 1.

        The fixed tensor should have shape ``(d_1, ..., d_n)``, with one input
        dimension per TR site. The returned cores all have shape
        ``(rank_{k-1}, d_k, rank_k)``, where the left ``rank`` of the first core
        matches the right ``rank`` of the last core. Consequently, the result
        always represents a cyclic TR.

        Parameters
        ----------
        center : int, optional
            Interior cut used for this fit. If omitted, the ``center`` fixed at
            construction is used.
        rank : int, optional
            Maximum ``rank`` allowed at every link. At each subchain SVD cut,
            at most this many singular values are retained. The initial
            bipartition retains at most ``rank ** 2`` singular values before
            its selected ``rank`` is factorized into two TR ranks.
        cutoff : float, optional
            Minimum singular value to keep. It must be finite and
            non-negative. Singular values ``<= cutoff`` are removed.
        atol : float, optional
            Absolute tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded
            while the accumulated sum of squares is ``<= atol``. It must be
            finite and non-negative.
        rtol : float, optional
            Relative tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded
            while the tail sum of squares divided by the total sum of squares
            is ``<= rtol``. It must be finite and in ``[0, 1]``.
        cum_percentage : float, optional
            Minimum fraction of squared singular-value mass to keep. Equivalent
            to setting ``rtol = 1 - cum_percentage``. It must be finite and in
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
            - ``1`` or ``True``: phase title, input configuration, cut
              progress and final summary;
            - ``2``: detailed per-cut ``rank``, error and timing information;
            - ``3``: level 2 output followed by every final core.

        Returns
        -------
        TRDecomposition
            Lightweight result containing cores and ranks. Its structured
            metrics are empty unless ``collect_metrics=True`` or
            ``verbose>0``.

        Examples
        --------
        Fix a tensor and compare shared ``rank`` caps at the middle cut:

        >>> tensor = torch.arange(16.).reshape(2, 2, 2, 2)
        >>> decomposer = tk.decompositions.TRSVD(tensor)
        >>> rank_one = decomposer.fit(rank=1)
        >>> rank_two = decomposer.fit(rank=2)
        >>> max(rank_one.rank)
        1
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
            center = self._center
        else:
            self._validate_center(center, self._tensor.ndim)
        verbosity = _normalize_verbosity(verbose)
        emit_events = bool(verbosity)
        collect_metrics = collect_metrics or emit_events
        fit_observer = (
            _resolve_observer(verbosity, None) if emit_events else None)

        in_dim = tuple(self._tensor.shape)
        if fit_observer is not None:
            fit_observer.emit(DecompositionEvent(
                name='start',
                phase=self._phase,
                values={
                    'sites': len(in_dim),
                    'in_dim': self._source.in_dim if getattr(
                        self._source, 'is_matrix', False) else in_dim,
                    'out_dim': getattr(self._source, 'out_dim', None),
                    'center': center,
                    'rank_mode': (
                        'discovery' if truncation.rank is None else 'shared'),
                    'renormalize': renormalize,
                }))
        progress = (
            None if fit_observer is None else _SVDProgress(
                observer=fit_observer,
                phase=self._phase))
        result = self._fit_validated(
            center=center,
            truncation=truncation,
            renormalize=renormalize,
            collect_metrics=collect_metrics,
            progress=progress)

        if fit_observer is not None:
            initial_padding = result.metadata['initial_padding']
            cycle_rank = result.rank[-1]
            center_rank = result.rank[center - 1]
            initial_capacity = cycle_rank * center_rank
            selected_rank = initial_capacity - initial_padding
            timing = result.metrics.timings[0]
            approximation = result.contract_dense()
            target = self._tensor.to(
                device=approximation.device, dtype=approximation.dtype)
            abs_log_error = log_tensor_norm(approximation - target)
            target_log_norm = log_tensor_norm(target)
            abs_error = abs_log_error.exp()
            rel_error = ratio_from_log(
                abs_log_error, target_log_norm)
            fit_observer.emit(DecompositionEvent(
                name='summary',
                phase=self._phase,
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
            displayed = _result_from_source(result, self._source)
            for site, core in enumerate(displayed.cores):
                fit_observer.emit(DecompositionEvent(
                    name='core',
                    phase=self._phase,
                    level=3,
                    site=site,
                    values={'shape': tuple(core.shape), 'tensor': core}))
            fit_observer.close(result.metrics)
        return _result_from_source(result, self._source)


def tr_svd(source: 'SourceLike',
           center: Optional[int] = None,
           rank: _Rank = None,
           cutoff: Optional[float] = None,
           atol: Optional[float] = None,
           rtol: Optional[float] = None,
           cum_percentage: Optional[float] = None,
           renormalize: bool = False,
           out_device: Optional[Union[str, torch.device]] = 'cpu',
           verbose: Union[bool, int] = 0,
           collect_metrics: bool = False) -> 'TensorDecomposition':
    r"""
    Decomposes a finite source using :class:`TRSVD`.

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
    source : TensorSource, TensorFormat1D or torch.Tensor
        Finite tensor input, with shape and evaluation defined by the source.
    center : int, optional
        Interior cut satisfying ``1 <= center < tensor.ndim``. The default is
        the middle cut.
    rank : int, optional
        Maximum ``rank`` allowed at every link. At each subchain SVD cut, at
        most this many singular values are retained. The initial bipartition
        retains at most ``rank ** 2`` singular values before its selected
        ``rank`` is factorized into two TR ranks.
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
    TRDecomposition
        Decomposition with cores and metadata. Access ``cores`` for the
        tensors and :meth:`~tensorkrowch.decompositions.TensorDecomposition.as_info`
        for structured information.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(
    ...     torch.arange(16.).reshape(2, 2, 2, 2))
    >>> result = tk.decompositions.tr_svd(source, rank=2)
    >>> result.in_dim
    (2, 2, 2, 2)
    """
    result = TRSVD(
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


def qtr_svd(source: CallableQuanticsSource,
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
            collect_metrics: bool = False) -> QTRDecomposition:
    r"""
    Decomposes a discretized coordinate callable into QTR cores.

    Materializes the full digit tensor from ``source`` and runs
    :meth:`TRSVD.fit` through :class:`TRSVD`. The result retains
    the source's layouts and coordinate maps for coordinate evaluation.

    Construct :class:`~tensorkrowch.decompositions.CallableQuanticsSource`
    first to define the callable, digit ordering and coordinate grid;
    use :meth:`~tensorkrowch.decompositions.CallableQuanticsSource.matrix`
    for paired input/output coordinates. The source selects the vector or
    matrix result; this function only controls its decomposition.

    Parameters
    ----------
    source : CallableQuanticsSource
        Callable with its digit layout and coordinate map.
    center : int, optional
        Interior cut from which the cyclic decomposition starts. If omitted,
        uses the middle of the digit chain.
    rank : int, optional
        Maximum ``rank`` allowed at every link. At each subchain SVD cut, at
        most this many singular values are retained. The initial bipartition
        retains at most ``rank ** 2`` singular values before its selected
        ``rank`` is factorized into two TR ranks.
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
    QTRDecomposition
        Quantics decomposition with cores, coordinate information and
        optional metrics.

    Examples
    --------
    >>> source = tk.decompositions.CallableQuanticsSource(
    ...     lambda coordinates: torch.exp(coordinates.sum(-1)),
    ...     n_coordinates=1, base=2, level=3, domain=(0., 1.))
    >>> result = tk.decompositions.qtr_svd(source, rank=1)
    >>> result.evaluate_indices(torch.tensor([[0], [4]]))
    tensor([1.0000, 1.6487])
    """
    if not isinstance(source, CallableQuanticsSource):
        raise TypeError('`source` should be CallableQuanticsSource type')
    return TRSVD(
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
    'TRSVD',
    'tr_svd',
    'qtr_svd',
]
