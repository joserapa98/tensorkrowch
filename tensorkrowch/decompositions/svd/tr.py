"""
This script contains:

    Internal TR-SVD rank policy:
        * _TRRankPolicy

    Class for TR-SVD decompositions:
        * TRSVD

    TR-SVD function:
        * tr_svd
"""

from contextlib import nullcontext
from dataclasses import dataclass, replace
from math import prod
from typing import List, Optional, Tuple, Union

import torch

from tensorkrowch.decompositions._runtime import _RuntimePolicy
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 TimingRecord,
                                                 TruncationRecord)
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import (TRDecomposition,
                                                 TTDecomposition)
from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.decompositions.svd.tt import TTSVD


_Rank = Optional[int]


@dataclass(frozen=True)
class _TRRankPolicy:
    """Stores the shared rank constraints for one TR-SVD fit."""

    mode: str
    requested: Optional[int]
    initial_cap: Optional[int]
    rank_cap: Optional[int]


class TRSVD:
    """Decomposes a fixed dense tensor into a tensor ring.

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

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor with one input dimension per TR site. At least two sites
        are required.
    center : int, optional
        Interior cut between sites ``center - 1`` and ``center``. It should
        satisfy ``1 <= center < tensor.ndim``. The default is the middle cut.
    out_device : str or torch.device, optional
        Device where finalized cores are stored. The default is ``"cpu"``.
        If ``None``, cores remain on the tensor's device.
    """

    def __init__(self,
                 tensor: torch.Tensor,
                 center: Optional[int] = None,
                 *,
                 out_device: Optional[
                     Union[str, torch.device]] = 'cpu') -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if tensor.ndim < 2:
            raise ValueError('`tensor` should contain at least two TR sites')
        if any(value < 1 for value in tensor.shape):
            raise ValueError('TR input dimensions should be positive')

        if center is None:
            center = tensor.ndim // 2
        else:
            self._validate_center(center, tensor.ndim)

        self._tensor = tensor
        self._center = center
        self._runtime = _RuntimePolicy.from_tensor(
            tensor,
            out_device=out_device)

    @staticmethod
    def _validate_center(center: int, n_sites: int) -> None:
        """Validates that a center identifies an interior chain cut."""
        if isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int type')
        if (center < 1) or (center >= n_sites):
            raise ValueError(
                '`center` should satisfy 1 <= center < tensor.ndim')

    @property
    def tensor(self) -> torch.Tensor:
        """Dense tensor fixed for repeated fits."""
        return self._tensor

    @property
    def center(self) -> int:
        """Preferred interior cut used when :meth:`fit` omits ``center``."""
        return self._center

    @staticmethod
    def _resolve_rank_policy(rank: _Rank) -> _TRRankPolicy:
        """Builds the TR rank policy from a validated shared rank."""

        if rank is None:
            return _TRRankPolicy(
                mode='discovery',
                requested=None,
                initial_cap=None,
                rank_cap=None)

        return _TRRankPolicy(
            mode='shared',
            requested=rank,
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

    def _decompose_subchain(
            self,
            subchain: torch.Tensor,
            in_dim: Tuple[int, ...],
            truncation: _TruncationSpec,
            renormalize: bool,
            collect_metrics: bool) -> Tuple[List[torch.Tensor],
                                             Optional[TTDecomposition]]:
        """Applies TT-SVD while preserving a subchain's boundary ranks."""
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
        result = engine.fit(
            **truncation.as_kwargs(),
            renormalize=renormalize,
            collect_metrics=collect_metrics)

        cores = list(result.cores)
        cores[0] = cores[0].reshape(
            left_rank, in_dim[0], cores[0].shape[-1])
        cores[-1] = cores[-1].reshape(
            cores[-1].shape[0], in_dim[-1], right_rank)
        return cores, result

    @staticmethod
    def _local_records(result: TTDecomposition,
                       phase: str,
                       site_offset: int) -> List[TruncationRecord]:
        """Labels TT-SVD records as local TR-SVD diagnostics."""
        return [
            replace(
                record,
                site=site_offset + record.site,
                phase=phase,
                global_relative_contribution=None,
                global_relative_contribution_per_batch=None)
            for record in result.metrics.truncations
        ]

    @staticmethod
    def _phase_timing(result: TTDecomposition,
                      name: str,
                      site_offset: int) -> TimingRecord:
        """Relabels one nested TT-SVD timing phase with global sites."""
        timing = result.metrics.timings[0]
        children = [
            replace(
                child,
                site=(None if child.site is None
                      else site_offset + child.site))
            for child in timing.children
        ]
        return replace(timing, name=name, children=children)

    def fit(self,
            center: Optional[int] = None,
            rank: _Rank = None,
            cutoff: Optional[float] = None,
            atol: Optional[float] = None,
            rtol: Optional[float] = None,
            cum_percentage: Optional[float] = None,
            renormalize: bool = False,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0) -> TRDecomposition:
        r"""Runs TR-SVD from an interior bipartition of the fixed tensor.

        The active exact SVD backend is selected through
        :func:`tensorkrowch.set_svd_method` or
        :func:`tensorkrowch.svd_method`. When metrics are collected, discarded
        energies are returned as phase-labelled local diagnostics. They are
        not combined into a global reconstruction bound because the successive
        TR-SVD truncations are not all orthogonal in one common scale.

        If ``rank`` is an integer, it is a shared upper bound for every TR
        rank and its square bounds the initial bipartition. Local truncation
        criteria may therefore select a different effective rank at each
        site. With ``None``, ranks are discovered from the selected SVD
        dimensions. The selected rank is split with the smallest admissible
        product and the most balanced pair breaks ties. Any extra capacity
        required by the cap is padded only with structural zeros and recorded
        in ``result.metadata``.

        The fixed tensor should have shape ``(d_1, ..., d_n)``, with one input
        dimension per TR site. The returned cores all have shape
        ``(rank_{k-1}, d_k, rank_k)``, where the left rank of the first core
        matches the right rank of the last core. Consequently, the result
        always represents a cyclic TR.

        Parameters
        ----------
        center : int, optional
            Interior cut used for this fit. If omitted, the center fixed at
            construction is used.
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
            - ``1`` or ``True``: phase title, site progress and final summary;
            - ``2``: input configuration and detailed per-site rank, error and
              timing information;
            - ``3``: level 2 output followed by every final core.

        Returns
        -------
        TRDecomposition
            Lightweight result containing cores and ranks. Its structured
            metrics are empty unless ``collect_metrics=True`` or
            ``verbose>0``.

        Examples
        --------
        Fix a tensor and compare shared rank caps at the middle cut:

        >>> tensor = torch.arange(16.).reshape(2, 2, 2, 2)
        >>> decomposer = TRSVD(tensor)
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
            center = self.center
        else:
            self._validate_center(center, self.tensor.ndim)
        rank_policy = self._resolve_rank_policy(rank=truncation.rank)
        verbosity = _normalize_verbosity(verbose)
        emit_events = bool(verbosity)
        collect_metrics = collect_metrics or emit_events
        fit_observer = (
            _resolve_observer(verbosity, None) if emit_events else None)

        in_dim = tuple(self.tensor.shape)
        if fit_observer is not None:
            fit_observer.emit(DecompositionEvent(
                name='start',
                phase='TR-SVD',
                values={
                    'sites': len(in_dim),
                    'in_dim': in_dim,
                    'center': center,
                    'rank_mode': rank_policy.mode,
                    'renormalize': renormalize,
                }))

        total_timer_context = (
            self._runtime.timer() if collect_metrics else nullcontext())
        with total_timer_context as total_timer:
            tensor = self._runtime.prepare(self.tensor)
            left_size = prod(in_dim[:center])
            right_size = prod(in_dim[center:])
            matrix = tensor.reshape(left_size, right_size)
            initial_result = TTSVD(
                matrix,
                out_device=None).fit(
                    rank=rank_policy.initial_cap,
                    cutoff=cutoff,
                    atol=atol,
                    rtol=rtol,
                    cum_percentage=cum_percentage,
                    renormalize=renormalize,
                    collect_metrics=collect_metrics)
            selected_rank = initial_result.rank[0]

            cycle_rank, center_rank = self._split_cycle_rank(
                selected_rank=selected_rank,
                rank_cap=rank_policy.rank_cap)
            initial_capacity = cycle_rank * center_rank
            padding = initial_capacity - selected_rank

            left_factor, right_factor = initial_result.cores
            if padding:
                left_factor = torch.cat([
                    left_factor,
                    left_factor.new_zeros(left_factor.shape[0], padding),
                ], dim=-1)
                right_factor = torch.cat([
                    right_factor,
                    right_factor.new_zeros(padding, right_factor.shape[1]),
                ], dim=0)

            left_subchain = left_factor.reshape(
                *in_dim[:center], cycle_rank, center_rank)
            left_subchain = left_subchain.movedim(-2, 0)
            right_subchain = right_factor.reshape(
                cycle_rank, center_rank, *in_dim[center:])
            right_subchain = right_subchain.movedim(0, -1)

            left_cores, left_result = self._decompose_subchain(
                subchain=left_subchain,
                in_dim=in_dim[:center],
                truncation=truncation,
                renormalize=renormalize,
                collect_metrics=collect_metrics)
            right_cores, right_result = self._decompose_subchain(
                subchain=right_subchain,
                in_dim=in_dim[center:],
                truncation=truncation,
                renormalize=renormalize,
                collect_metrics=collect_metrics)

        metrics = DecompositionMetrics()
        if collect_metrics:
            initial_records = self._local_records(
                initial_result,
                phase='initial_bipartition',
                site_offset=center - 1)
            left_records = (
                [] if left_result is None else self._local_records(
                    left_result,
                    phase='left_subchain',
                    site_offset=0))
            right_records = (
                [] if right_result is None else self._local_records(
                    right_result,
                    phase='right_subchain',
                    site_offset=center))
            metrics.truncations.extend(
                left_records + initial_records + right_records)

            phase_timings = [
                self._phase_timing(
                    initial_result, 'initial_bipartition', center - 1),
            ]
            if left_result is not None:
                phase_timings.append(self._phase_timing(
                    left_result, 'left_subchain', 0))
            if right_result is not None:
                phase_timings.append(self._phase_timing(
                    right_result, 'right_subchain', center))
            metrics.timings.append(TimingRecord(
                name='fit',
                elapsed=total_timer.elapsed,
                children=phase_timings))
            if padding:
                metrics.warnings.append(
                    f'Initial SVD rank {selected_rank} uses capacity '
                    f'{initial_capacity} with {padding} structural zero '
                    'dimensions')
            metrics.warnings.append(
                'TR-SVD truncation records are local diagnostics and are not '
                'combined into a global reconstruction bound')

        result = TRDecomposition(
            cores=left_cores + right_cores,
            metrics=metrics,
            metadata={
                'algorithm': 'tr_svd',
                'center': center,
                'rank_mode': rank_policy.mode,
                'requested_rank': rank_policy.requested,
                'renormalize': renormalize,
                'initial_selected_rank': selected_rank,
                'initial_capacity': initial_capacity,
                'cycle_rank': cycle_rank,
                'center_rank': center_rank,
                'structural_padding': padding,
                'truncation_errors': 'local_diagnostics',
            })

        if fit_observer is not None:
            for position, record in enumerate(result.metrics.truncations):
                fit_observer.emit(DecompositionEvent(
                    name='site_complete',
                    phase='TR-SVD',
                    site=record.site,
                    values={
                        'total_sites': len(result.metrics.truncations),
                        'step': position + 1,
                        'subphase': record.phase,
                        'full_rank': record.full_rank,
                        'selected_rank': record.selected_rank,
                        'absolute_error': record.local_absolute_error,
                        'relative_error': record.local_relative_error,
                    }))
            fit_observer.emit(DecompositionEvent(
                name='summary',
                phase='TR-SVD',
                values={
                    'rank': result.rank,
                    'center': center,
                    'initial_rank': selected_rank,
                    'initial_capacity': initial_capacity,
                    'structural_padding': padding,
                    'elapsed': f'{total_timer.elapsed:.6f} s',
                }))
            for site, core in enumerate(result.cores):
                fit_observer.emit(DecompositionEvent(
                    name='core',
                    phase='TR-SVD',
                    level=3,
                    site=site,
                    values={'shape': tuple(core.shape), 'tensor': core}))
            fit_observer.close(result.metrics)
        return result


def tr_svd(tensor: torch.Tensor,
           center: Optional[int] = None,
           rank: _Rank = None,
           cutoff: Optional[float] = None,
           atol: Optional[float] = None,
           rtol: Optional[float] = None,
           cum_percentage: Optional[float] = None,
           renormalize: bool = False,
           out_device: Optional[Union[str, torch.device]] = 'cpu',
           verbose: Union[bool, int] = 0,
           return_info: bool = False):
    r"""Decomposes a dense tensor into TR cores through an interior SVD.

    This is the simple functional interface. Use :class:`TRSVD` to repeat
    fits of the same tensor or to access the lightweight result object. The
    initial SVD rank is opened as cyclic and interior ranks, after which both
    tensor blocks are decomposed through TT-SVD.

    The input should have shape ``(d_1, ..., d_n)``, with one input dimension
    per site and at least two sites. Every returned core has shape
    ``(rank_{k-1}, d_k, rank_k)`` and the last right rank matches the first
    left rank, closing the TR.

    An integer ``rank`` is a shared upper bound, while local truncation may
    select different effective ranks across the chain. With ``None``, the
    initial SVD rank is split using the smallest admissible product and the
    most balanced pair breaks ties. Any extra capacity required by the cap
    contains structural zeros and is reported in ``info['metadata']``.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor with one input dimension per TR site.
    center : int, optional
        Interior cut satisfying ``1 <= center < tensor.ndim``. The default is
        the middle cut.
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
        If ``True``, normalizes residuals before SVDs and accumulates their
        scales logarithmically before redistributing them over final cores.
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
        If ``True``, also returns ranks, dimensions, rank factorization,
        metadata and structured local metrics. With the default ``False`` and
        ``verbose=0``, diagnostic norm reductions, records and synchronized
        timings are skipped.

    Returns
    -------
    list[torch.Tensor] or tuple
        TR cores by default. If ``return_info=True``, returns ``(cores, info)``
        with ranks and structured metrics.

    Examples
    --------
    Decompose a four-site tensor with a shared rank cap:

    >>> tensor = torch.arange(16.).reshape(2, 2, 2, 2)
    >>> cores = tr_svd(tensor, rank=2)
    >>> [tuple(core.shape) for core in cores]
    [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]

    Inspect the structural padding required by a shared rank cap:

    >>> singular_values = torch.tensor([5., 3., 1., 0.])
    >>> _, info = tr_svd(torch.diag(singular_values),
    ...                  rank=2, return_info=True)
    >>> info['rank']
    [2, 2]
    >>> info['metadata']['structural_padding']
    1
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    result = TRSVD(
        tensor=tensor,
        center=center,
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


__all__ = ['TRSVD', 'tr_svd']
