"""
This script contains:

    Public classes:
        * LoopOpenerCapabilities
        * LoopOpening
        * LoopOpener
        * ALSLoopOpener
        * SVDLoopOpener
        * FixedGaugeCoreOpener
        * CallableLoopOpener
        * CompositeLoopOpener

    Internal functions:
        * _normalize_context
        * _normalize_orientation
        * _mirror_rank
        * _mirror_cores
        * _source_from_context
        * _mirror_source
        * _cast_source
        * _opening_from_result

    Public functions:
        * resolve_loop_opener
"""

from dataclasses import dataclass, field
from typing import (Any,
                    Callable,
                    Dict,
                    Mapping,
                    Optional,
                    Protocol,
                    Sequence,
                    Tuple,
                    Union,
                    runtime_checkable)

import torch

from tensorkrowch.formats import TR
from tensorkrowch.utils import safe_divide, truncated_svd

from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.decompositions._utils import _normalize_rank
from tensorkrowch.decompositions.als.convergence import ConvergencePolicy
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.als.tr import TRALS
from tensorkrowch.decompositions.metrics import (LocalSolveRecord,
                                                 TruncationRecord)
from tensorkrowch.decompositions.results import (TRDecomposition,
                                                 TensorDecomposition)
from tensorkrowch.decompositions.sources import BaseSource, ConfigurationBatch
from tensorkrowch.decompositions.sources.callable import CallableTensorSource
from tensorkrowch.decompositions.sources.factory import as_tensor_source


_Rank = Union[int, Sequence[int]]
_Context = Optional[Mapping[str, Any]]


@dataclass(frozen=True)
class LoopOpenerCapabilities:  # MARK: LoopOpenerCapabilities
    """Declares the constraints accepted by a loop-opening strategy."""

    supports_fixed_left: bool = False  # Whether the opener can preserve a left gauge
    supports_fixed_right: bool = False  # Whether the opener can preserve a right gauge
    # Whether both gauges can be fixed simultaneously
    supports_two_fixed_gauges: bool = False
    supports_blocks: bool = False  # Whether more than one input core can be opened

    def __post_init__(self) -> None:
        for name in (
                'supports_fixed_left',
                'supports_fixed_right',
                'supports_two_fixed_gauges',
                'supports_blocks'):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'`{name}` should be bool type')

    def require(self,
                *,
                fixed_left: bool,
                fixed_right: bool,
                block_size: int) -> None:
        """
        Raises before execution when requested constraints are
        unsupported.
        """
        if fixed_left and not self.supports_fixed_left:
            raise ValueError('The loop opener does not support a fixed left gauge')
        if fixed_right and not self.supports_fixed_right:
            raise ValueError('The loop opener does not support a fixed right gauge')
        if fixed_left and fixed_right and \
                not self.supports_two_fixed_gauges:
            raise ValueError(
                'The loop opener does not support two fixed gauges')
        if block_size > 1 and not self.supports_blocks:
            raise ValueError('The loop opener does not support input blocks')


@dataclass(frozen=True)
class LoopOpening:  # MARK: LoopOpening
    """
    Stores gauges, local TR ``cores`` and ``diagnostics`` from one loop opening.

    ``left_gauge`` and ``right_gauge`` are the environment ``cores`` surrounding
    the local input ``cores``. When both are present, ``all_cores`` follows
    standard TR order and contracts to the local target. ``rank[k]`` is the
    right-link ``rank`` of ``all_cores[k]``.
    """

    left_gauge: Optional[torch.Tensor]  # Left external gauge core
    cores: Sequence[torch.Tensor]  # Raw cores in site order
    right_gauge: Optional[torch.Tensor]  # Right external gauge core
    rank: Sequence[int]  # Right-link ranks in core order
    orientation: str = 'right'  # Left or right interpretation of the gauge axes
    # Optional least-squares measurements for this opening
    local_records: Sequence[LocalSolveRecord] = ()
    # Local strategy diagnostics and construction state
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.orientation not in ('right', 'left'):
            raise ValueError("`orientation` should be 'right' or 'left'")
        cores = tuple(self.cores)
        if not cores:
            raise ValueError('`cores` should contain at least one input core')
        if not all(isinstance(core, torch.Tensor) and core.ndim == 3
                   for core in cores):
            raise ValueError(
                '`cores` should contain standard three-dimensional TR cores')
        for name in ('left_gauge', 'right_gauge'):
            gauge = getattr(self, name)
            if gauge is not None and \
                    (not isinstance(gauge, torch.Tensor) or gauge.ndim != 3):
                raise ValueError(f'`{name}` should be a three-dimensional core')
        all_cores = tuple(
            core for core in (self.left_gauge, *cores, self.right_gauge)
            if core is not None)
        TRDecomposition(all_cores)
        rank = tuple(self.rank)
        actual_rank = tuple(core.shape[-1] for core in all_cores)
        if rank != actual_rank:
            raise ValueError(
                '`rank` should contain the actual right-link rank of every core')
        records = tuple(self.local_records)
        if not all(isinstance(record, LocalSolveRecord) for record in records):
            raise TypeError(
                '`local_records` should contain LocalSolveRecord objects')
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError('`diagnostics` should be a mapping')
        object.__setattr__(self, 'cores', cores)
        object.__setattr__(self, 'rank', rank)
        object.__setattr__(self, 'local_records', records)
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))

    @property
    def all_cores(self) -> Tuple[torch.Tensor, ...]:
        """Returns gauges and input cores in standard cyclic order."""
        return tuple(
            core for core in (self.left_gauge, *self.cores, self.right_gauge)
            if core is not None)

    def contract_dense(self) -> torch.Tensor:
        """
        Contracts the complete local opening without constructing a
        model.
        """
        return TRDecomposition(self.all_cores).contract_dense()


@runtime_checkable
class LoopOpener(Protocol):  # MARK: LoopOpener
    """
    Solves one local cyclic factorization with explicit gauge constraints.

    An opening represents its target as left gauge, input core(s) and right
    gauge. Capabilities declare whether a strategy accepts either or both fixed
    gauges and whether it supports multiple input cores. Fixed gauges must be
    preserved exactly. The target and gauge recursion are independent of the
    opener.
    """

    @property
    def capabilities(self) -> LoopOpenerCapabilities:
        """Returns constraints supported by this strategy."""

    def open(self,
             target: Any,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """
        Opens a local tensor into gauges and input cores.

        Parameters
        ----------
        target : torch.Tensor or BaseSource
            Local scalar tensor with axes ``(left external, *in_dim, right
            external)``, or a source representing that tensor.
        rank : int or sequence[int]
            Prescribed right-link ranks of the complete local ring, including
            the two external gauge cores. The last link closes the ring.
        fixed_left, fixed_right : torch.Tensor, optional
            Gauges to retain exactly. The selected strategy must support the
            supplied constraints. Their shapes follow ordinary three-axis TR
            cores.
        orientation : {"right", "left"}
            Construction direction. Returned cores always follow the original
            ``target`` order, including when the algorithm uses a mirrored local
            problem.
        context : mapping, optional
            Provider metadata and internal execution options, such as
            initialization, generator and whether to collect local metrics.

        Returns
        -------
        LoopOpening
            Left gauge, input cores, right gauge, actual ranks and local
            diagnostics.
        """


def _normalize_context(context: _Context) -> Dict[str, Any]:
    """Copies an optional opening ``context`` after validating its type."""
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise TypeError('`context` should be a mapping or None')
    return dict(context)


def _normalize_orientation(orientation: str) -> str:
    """Validates the explicit opening ``orientation``."""
    if not isinstance(orientation, str):
        raise TypeError('`orientation` should be str type')
    if orientation not in ('right', 'left'):
        raise ValueError("`orientation` should be 'right' or 'left'")
    return orientation


def _mirror_rank(rank: Sequence[int]) -> Tuple[int, ...]:
    """Maps right-link ranks through reversal and core transposition."""
    return (*reversed(rank[:-1]), rank[-1])


def _mirror_cores(cores: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """
    Mirrors standard TR ``cores`` while preserving their represented tensor.
    """
    return tuple(core.permute(2, 1, 0) for core in reversed(cores))


def _source_from_context(target: Any,
                         context: Mapping[str, Any]) -> BaseSource:
    """
    Normalizes a local ``target`` through the shared BaseSource interface.
    """
    if isinstance(target, (torch.Tensor, BaseSource)):
        return as_tensor_source(target)
    return CallableTensorSource(
        target,
        in_dim=context.get('in_dim'),
        out_shape=(),
        dtype=context.get('dtype'),
        device=context.get('device', 'cpu'),
        batch_size=context.get('batch_size'))


def _mirror_source(source: BaseSource) -> BaseSource:
    """Creates a lazy ``source`` whose variables are in reverse site order."""
    def evaluate(indices: torch.Tensor) -> torch.Tensor:
        configurations = ConfigurationBatch(
            indices.flip(1), kind='indices')
        return source.evaluate(configurations)

    return CallableTensorSource(
        evaluate,
        in_dim=tuple(reversed(source.in_dim)),
        out_shape=(),
        dtype=source.dtype,
        device=source.device)


def _cast_source(source: BaseSource, dtype: torch.dtype) -> BaseSource:
    """Lazily casts target values to an initializer's numerical ``dtype``."""
    if source.dtype == dtype:
        return source

    def evaluate(indices: torch.Tensor) -> torch.Tensor:
        configurations = ConfigurationBatch(indices, kind='indices')
        return source.evaluate(configurations).to(dtype=dtype)

    return CallableTensorSource(
        evaluate,
        in_dim=source.in_dim,
        out_shape=(),
        dtype=dtype,
        device=source.device)


def _opening_from_result(result: TensorDecomposition,
                         orientation: str) -> LoopOpening:
    """Converts an ALS ``result`` back from the requested ``orientation``."""
    result_cores = tuple(result.cores)
    if orientation == 'left':
        result_cores = _mirror_cores(result_cores)
    return LoopOpening(
        left_gauge=result_cores[0],
        cores=result_cores[1:-1],
        right_gauge=result_cores[-1],
        rank=tuple(core.shape[-1] for core in result_cores),
        orientation=orientation,
        local_records=tuple(result.metrics.local_solves),
        diagnostics={
            'algorithm': 'als',
            'metadata': dict(result.metadata),
        })


class ALSLoopOpener:  # MARK: ALSLoopOpener
    """
    Opens local loops using a reusable, encapsulated TR-ALS policy.

    Parameters
    ----------
    fit_options : mapping, optional
        Options forwarded to :meth:`TRALS.fit`. Target, rank, fixed cores,
        output device and orientation remain controlled by :meth:`open`.
        ALS sweeps visit sites in increasing order on every pass.
    """

    _capabilities = LoopOpenerCapabilities(
        supports_fixed_left=True,
        supports_fixed_right=True,
        supports_two_fixed_gauges=False,
        supports_blocks=True)

    def __init__(self,
                 fit_options: Optional[Mapping[str, Any]] = None) -> None:
        if fit_options is None:
            fit_options = {}
        if not isinstance(fit_options, Mapping):
            raise TypeError('`fit_options` should be a mapping or None')
        reserved = {'rank', 'fixed_cores', 'initial_cores', 'collect_metrics'}
        overlap = reserved.intersection(fit_options)
        if overlap:
            raise ValueError(
                f'`fit_options` should not override {sorted(overlap)}')
        self.fit_options = dict(fit_options)

    @property
    def capabilities(self) -> LoopOpenerCapabilities:
        """Returns ALS constraint support."""
        return self._capabilities

    def open(self,
             target: Any,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """
        Fits a local TR with zero or one fixed environment gauge.

        Parameters
        ----------
        target : torch.Tensor or BaseSource
            Local scalar tensor with axes ``(left external, *in_dim, right
            external)``, or a source representing that tensor.
        rank : int or sequence[int]
            Prescribed right-link ranks of the complete local ring, including
            the two external gauge cores. The last link closes the ring.
        fixed_left : torch.Tensor, optional
            Gauges to retain exactly. The selected strategy must support the
            supplied constraints. Their shapes follow ordinary three-axis TR
            cores.
        fixed_right : torch.Tensor, optional
            Gauges to retain exactly. The selected strategy must support the
            supplied constraints. Their shapes follow ordinary three-axis TR
            cores.
        orientation : {"right", "left"}
            Construction direction. Returned cores always follow the original
            ``target`` order, including when the algorithm uses a mirrored local
            problem.
        context : mapping, optional
            Provider metadata and internal execution options, such as
            initialization, generator and whether to collect local metrics.

        Returns
        -------
        LoopOpening
            Gauges and cores in original ``target`` order with optional local
            metrics.
        """
        orientation = _normalize_orientation(orientation)
        context = _normalize_context(context)
        source = _source_from_context(target, context)
        initial_cores = context.get('initial_cores')
        runtime_dtype = source.dtype
        if initial_cores is not None:
            if isinstance(initial_cores, (str, bytes)) or \
                    not isinstance(initial_cores, Sequence):
                raise TypeError(
                    '`initial_cores` should contain torch.Tensor objects')
            initial_cores = tuple(initial_cores)
            if not initial_cores or not all(
                    isinstance(core, torch.Tensor) for core in initial_cores):
                raise TypeError(
                    '`initial_cores` should contain torch.Tensor objects')
            for core in initial_cores:
                runtime_dtype = torch.promote_types(
                    runtime_dtype, core.dtype)
        for gauge in (fixed_left, fixed_right):
            if gauge is not None:
                runtime_dtype = torch.promote_types(
                    runtime_dtype, gauge.dtype)
        source = _cast_source(source, runtime_dtype)
        if initial_cores is not None:
            initial_cores = tuple(
                core.to(dtype=runtime_dtype) for core in initial_cores)
            context['initial_cores'] = initial_cores
        if fixed_left is not None:
            fixed_left = fixed_left.to(dtype=runtime_dtype)
        if fixed_right is not None:
            fixed_right = fixed_right.to(dtype=runtime_dtype)
        if len(source.in_dim) < 3:
            raise ValueError(
                'A local loop opening should contain left, physical and right '
                'input dimensions')
        ranks = _normalize_rank(rank, len(source.in_dim))
        self._capabilities.require(
            fixed_left=fixed_left is not None,
            fixed_right=fixed_right is not None,
            block_size=len(source.in_dim) - 2)

        if orientation == 'left':
            source = _mirror_source(source)
            ranks = _mirror_rank(ranks)
            fixed_left, fixed_right = (
                None if fixed_right is None else fixed_right.permute(2, 1, 0),
                None if fixed_left is None else fixed_left.permute(2, 1, 0),
            )
            if initial_cores is not None:
                initial_cores = _mirror_cores(tuple(initial_cores))
        fixed_cores = [None] * len(source.in_dim)
        fixed_cores[0] = fixed_left
        fixed_cores[-1] = fixed_right

        fit_options = dict(self.fit_options)
        fit_options.setdefault(
            'convergence', ConvergencePolicy(max_sweeps=10))
        if context.get('generator') is not None:
            fit_options.setdefault('generator', context['generator'])
        result = TRALS(source, out_device=None).fit(
            rank=ranks,
            initial_cores=initial_cores,
            fixed_cores=fixed_cores,
            collect_metrics=context.get('collect_metrics', True),
            **fit_options)
        return _opening_from_result(result, orientation)


class SVDLoopOpener:  # MARK: SVDLoopOpener
    """
    Balances free external axes before opening a local tensor ring.

    SVDs run right then left, by default skipping inherited fixed gauges.
    With spectra outside, each SVD passes its unweighted factor to the next
    SVD and the wrapped solver. Optional exact ALS refinement restores the
    spectra in the retained objective and warm-start gauges before lifting
    the result.
    Refinement preserves fixed gauges and rejects a larger local residual.
    It does not guarantee a smaller global tensor ring error.
    With both gauges fixed, the wrapped solver runs without extra refinement.

    Parameters
    ----------
    opener : str or LoopOpener, optional
        Wrapped strategy, default ``"als"``. Numerical ALS fit options are
        reused when it is an :class:`ALSLoopOpener`; other strategies use
        default ALS options for refinement.
    absorb_singular_values : bool, optional
        Keep spectra in the initial objective if ``True``. Default ``False``
        moves spectra outside until lifting or refinement.
    refinement_sweeps : int, optional
        Additional exact ALS sweeps in the restored retained metric.
        Default ``0`` disables refinement. The best candidate is compared
        with the initializer, and inherited gauges stay fixed.
    project_fixed_gauges : bool, optional
        Also trim inherited axes and project their fixed gauges into the
        compact SVD bases. Default ``False`` trims only free axes. With
        ``True``, original fixed gauges are restored on return; refinement
        runs after lifting in the full original target, so it also sees
        components omitted by the compact fixed-gauge projection.
    cutoff, atol, rtol, cum_percentage : float, optional
        Criteria passed to :func:`~tensorkrowch.utils.truncated_svd`. Each
        free axis is also capped by its adjacent TR-rank product. These
        criteria change external sketch dimensions, never prescribed TR ranks.

    Notes
    -----
    ``open`` accepts a dense tensor with axes ``(left external, *physical,
    right external)``. It returns gauges in the original external coordinates.
    This works with arbitrary TT core gauges and initial multi-site blocks;
    no Vidal representation is required. Initial ALS residuals use the
    balanced metric; ``diagnostics['svd_refinement']`` records refinement
    errors in the restored refinement metric.
    """

    def __init__(self,
                 opener: Union[str, LoopOpener] = 'als',
                 *,
                 absorb_singular_values: bool = False,
                 refinement_sweeps: int = 0,
                 project_fixed_gauges: bool = False,
                 cutoff: Optional[float] = None,
                 atol: Optional[float] = None,
                 rtol: Optional[float] = None,
                 cum_percentage: Optional[float] = None) -> None:
        self.opener = resolve_loop_opener(opener)
        if not isinstance(absorb_singular_values, bool):
            raise TypeError('`absorb_singular_values` should be bool type')
        if isinstance(refinement_sweeps, bool) or \
                not isinstance(refinement_sweeps, int):
            raise TypeError('`refinement_sweeps` should be int type')
        if refinement_sweeps < 0:
            raise ValueError('`refinement_sweeps` should be non-negative')
        self.absorb_singular_values = absorb_singular_values
        if not isinstance(project_fixed_gauges, bool):
            raise TypeError('`project_fixed_gauges` should be bool type')
        self.project_fixed_gauges = project_fixed_gauges
        self.truncation = _TruncationSpec(
            cutoff=cutoff, atol=atol, rtol=rtol, cum_percentage=cum_percentage)
        self.refiner = None
        if refinement_sweeps:
            options = dict(self.opener.fit_options) \
                if isinstance(self.opener, ALSLoopOpener) else {}
            options.update({
                'sampler': None,
                'convergence': ConvergencePolicy(
                    max_sweeps=refinement_sweeps, keep_best=True),
            })
            self.refiner = ALSLoopOpener(options)

    @property
    def capabilities(self) -> 'LoopOpenerCapabilities':
        """Returns the wrapped solver's gauge and block constraints."""
        return self.opener.capabilities

    def _trim(self,
              matrix: torch.Tensor,
              rank: int,
              site: int,
              context: Mapping[str, Any],
              records: list
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        options = dict(context.get(
            '_svd_truncation', self.truncation.as_kwargs()))
        options['rank'] = min(rank, min(matrix.shape))
        collect_metrics = context.get(
            '_svd_collect_metrics', context.get('collect_metrics', True))
        if collect_metrics:
            u, s, vh, info = truncated_svd(
                matrix, return_info=True, **options)
            records.append(TruncationRecord.from_svd_info(
                info, site=site))
        else:
            u, s, vh = truncated_svd(matrix, **options)
        return u, s, vh

    def _refine(self,
                tensor: torch.Tensor,
                initial: LoopOpening,
                fixed_left: Optional[torch.Tensor],
                fixed_right: Optional[torch.Tensor],
                context: Dict[str, Any]) -> LoopOpening:
        """Refines the restored metric and rejects a larger residual."""
        before = torch.linalg.vector_norm(tensor - initial.contract_dense())
        context['initial_cores'] = initial.all_cores
        refined = self.refiner.open(
            tensor, initial.rank, fixed_left=fixed_left,
            fixed_right=fixed_right, orientation=initial.orientation,
            context=context)
        candidate = torch.linalg.vector_norm(tensor - refined.contract_dense())
        accepted = bool(torch.isfinite(candidate) & (candidate <= before))
        after = candidate if accepted else before
        denominator = torch.linalg.vector_norm(tensor)
        refinement = {
            'accepted': accepted,
            'relative_error_before': safe_divide(
                before, denominator,
                zero_value=torch.where(before == 0, 0, torch.inf)),
            'relative_error_after': safe_divide(
                after, denominator,
                zero_value=torch.where(after == 0, 0, torch.inf)),
            'metadata': refined.diagnostics['metadata'],
        }
        selected = refined if accepted else initial
        return LoopOpening(
            left_gauge=selected.left_gauge, cores=selected.cores,
            right_gauge=selected.right_gauge, rank=initial.rank,
            orientation=initial.orientation,
            local_records=initial.local_records + refined.local_records,
            diagnostics={**initial.diagnostics, 'svd_refinement': refinement})

    def open(self,
             target: Any,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """
        Opens a dense local target after balancing its external axes.

        Arguments follow :meth:`ALSLoopOpener.open`. ``target`` must be a
        tensor with at least one physical axis between its two external
        axes. Returned gauges use the original external coordinates. The
        target and fixed gauges are promoted to their common dtype before
        projecting inherited gauges.
        """
        if not isinstance(target, torch.Tensor):
            raise TypeError(
                '`target` should be a tensor with two external axes')
        if target.ndim < 3:
            raise ValueError(
                '`target` should contain two external axes and at least '
                'one physical axis')
        context = _normalize_context(context)
        orientation = _normalize_orientation(orientation)
        self.capabilities.require(
            fixed_left=fixed_left is not None,
            fixed_right=fixed_right is not None,
            block_size=target.ndim - 2)
        ranks = list(_normalize_rank(rank, target.ndim))
        runtime_dtype = target.dtype
        for name, gauge in (('fixed_left', fixed_left),
                            ('fixed_right', fixed_right)):
            if gauge is not None:
                if not isinstance(gauge, torch.Tensor):
                    raise TypeError(f'`{name}` should be torch.Tensor or None')
                external_axis = 0 if name == 'fixed_left' else -1
                if gauge.ndim != 3 or \
                        gauge.shape[1] != target.shape[external_axis]:
                    raise ValueError(
                        f'`{name}` should have three axes and match the '
                        'corresponding external target dimension')
                runtime_dtype = torch.promote_types(runtime_dtype, gauge.dtype)
        tensor = target.to(dtype=runtime_dtype)
        if fixed_left is not None:
            fixed_left = fixed_left.to(dtype=runtime_dtype)
        if fixed_right is not None:
            fixed_right = fixed_right.to(dtype=runtime_dtype)
        sites = context.get('sites', tuple(range(1, target.ndim - 1)))
        records = []
        left_lift = None
        right_lift = None
        left_spectrum = None
        right_spectrum = None
        local_fixed_left, local_fixed_right = fixed_left, fixed_right
        full_refinement = self.project_fixed_gauges and (
            fixed_left is not None or fixed_right is not None)

        if fixed_left is not None:
            ranks[0] = fixed_left.shape[-1]
            ranks[-1] = fixed_left.shape[0]
        if fixed_right is not None:
            ranks[-2] = fixed_right.shape[0]
            ranks[-1] = fixed_right.shape[-1]

        if fixed_right is None or self.project_fixed_gauges:
            matrix = tensor.reshape(-1, tensor.shape[-1])
            u, s, right_lift = self._trim(
                matrix,
                ranks[-2] * ranks[-1],
                sites[-1], context, records)
            if fixed_right is not None:
                local_fixed_right = torch.einsum(
                    'rmg,bm->rbg', fixed_right, right_lift.conj())
                if not self.absorb_singular_values:
                    local_fixed_right = local_fixed_right * safe_divide(
                        torch.ones_like(s), s)[None, :, None]
            if self.absorb_singular_values:
                u = u * s.unsqueeze(0)
            elif self.refiner is not None:
                right_spectrum = s
            else:
                right_lift = s.unsqueeze(1) * right_lift
            tensor = u.reshape(*tensor.shape[:-1], u.shape[-1])

        if fixed_left is None or self.project_fixed_gauges:
            matrix = tensor.reshape(tensor.shape[0], -1)
            left_lift, s, vh = self._trim(
                matrix,
                ranks[-1] * ranks[0],
                sites[0], context, records)
            if fixed_left is not None:
                local_fixed_left = torch.einsum(
                    'sa,gsr->gar', left_lift.conj(), fixed_left)
                if not self.absorb_singular_values:
                    local_fixed_left = local_fixed_left * safe_divide(
                        torch.ones_like(s), s)[None, :, None]
            if self.absorb_singular_values:
                vh = s.unsqueeze(1) * vh
            elif self.refiner is not None:
                left_spectrum = s
            else:
                left_lift = left_lift * s.unsqueeze(0)
            tensor = vh.reshape(s.shape[0], *tensor.shape[1:])

        local_context = dict(context)
        local_context.update({
            'in_dim': tuple(tensor.shape),
            'dtype': tensor.dtype,
            'device': tensor.device,
        })
        opening = self.opener.open(
            tensor,
            tuple(ranks),
            fixed_left=local_fixed_left,
            fixed_right=local_fixed_right,
            orientation=orientation,
            context=local_context)

        refine = self.refiner is not None and (
            fixed_left is None or fixed_right is None)
        if refine:
            left_gauge, right_gauge = opening.left_gauge, opening.right_gauge
            if left_spectrum is not None:
                tensor = tensor * left_spectrum.reshape(
                    -1, *((1,) * (tensor.ndim - 1)))
                left_gauge = left_gauge * left_spectrum[None, :, None]
            if right_spectrum is not None:
                tensor = tensor * right_spectrum
                right_gauge = right_gauge * right_spectrum[None, :, None]
            initial = LoopOpening(
                left_gauge=left_gauge, cores=opening.cores,
                right_gauge=right_gauge, rank=opening.rank,
                orientation=opening.orientation,
                local_records=opening.local_records,
                diagnostics=opening.diagnostics)
            opening = initial if full_refinement else self._refine(
                tensor, initial, fixed_left, fixed_right, local_context)

        left_gauge = opening.left_gauge
        if left_lift is not None:
            left_gauge = torch.einsum(
                'sa,gar->gsr',
                left_lift.to(left_gauge.dtype),
                left_gauge)
        right_gauge = opening.right_gauge
        if right_lift is not None:
            right_gauge = torch.einsum(
                'rbg,bt->rtg',
                right_gauge,
                right_lift.to(right_gauge.dtype))
        if full_refinement:
            if fixed_left is not None:
                left_gauge = fixed_left
            if fixed_right is not None:
                right_gauge = fixed_right
            opening = LoopOpening(
                left_gauge=left_gauge, cores=opening.cores,
                right_gauge=right_gauge, rank=opening.rank,
                orientation=opening.orientation,
                local_records=opening.local_records,
                diagnostics=opening.diagnostics)
            if refine:
                local_context['in_dim'] = tuple(target.shape)
                opening = self._refine(
                    target, opening, fixed_left, fixed_right, local_context)
            left_gauge, right_gauge = opening.left_gauge, opening.right_gauge
        all_cores = (left_gauge, *opening.cores, right_gauge)
        return LoopOpening(
            left_gauge=left_gauge,
            cores=opening.cores,
            right_gauge=right_gauge,
            rank=tuple(core.shape[-1] for core in all_cores),
            orientation=opening.orientation,
            local_records=opening.local_records,
            diagnostics={
                **opening.diagnostics,
                'svd_trimmed_left': left_lift is not None,
                'svd_trimmed_right': right_lift is not None,
                'absorb_singular_values': self.absorb_singular_values,
                'project_fixed_gauges': self.project_fixed_gauges,
                'refinement_space': 'full' if full_refinement else 'retained',
                'truncations': tuple(records),
            })


class FixedGaugeCoreOpener:  # MARK: FixedGaugeCoreOpener
    """
    Solves the single remaining input core with both gauges fixed.

    The full effective local design is solved once using LeastSquaresSolver.
    The external maps remain fixed; they are not pseudoinverted separately.
    This opener requires exactly one input core between the two gauges.

    Parameters
    ----------
    solver : LeastSquaresSolver, optional
        Reusable local solver. Defaults to the shared stable least-squares
        policy.
    """

    _capabilities = LoopOpenerCapabilities(
        supports_fixed_left=True,
        supports_fixed_right=True,
        supports_two_fixed_gauges=True,
        supports_blocks=False)

    def __init__(self, solver: Optional[LeastSquaresSolver] = None) -> None:
        if solver is None:
            solver = LeastSquaresSolver()
        elif not isinstance(solver, LeastSquaresSolver):
            raise TypeError('`solver` should be LeastSquaresSolver type')
        self.solver = solver

    @property
    def capabilities(self) -> LoopOpenerCapabilities:
        """Returns direct fixed-gauge constraint support."""
        return self._capabilities

    def open(self,
             target: Any,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """
        Solves the unique unknown core by one dense least-squares system.

        Parameters
        ----------
        target : torch.Tensor or BaseSource
            Local scalar tensor with axes ``(left external, *in_dim, right
            external)``, or a source representing that tensor.
        rank : int or sequence[int]
            Prescribed right-link ranks of the complete local ring, including
            the two external gauge cores. The last link closes the ring.
        fixed_left : torch.Tensor, optional
            Gauges to retain exactly. The selected strategy must support the
            supplied constraints. Their shapes follow ordinary three-axis TR
            cores.
        fixed_right : torch.Tensor, optional
            Gauges to retain exactly. The selected strategy must support the
            supplied constraints. Their shapes follow ordinary three-axis TR
            cores.
        orientation : {"right", "left"}
            Construction direction. Returned cores always follow the original
            ``target`` order, including when the algorithm uses a mirrored local
            problem.
        context : mapping, optional
            Provider metadata and internal execution options, such as
            initialization, generator and whether to collect local metrics.

        Returns
        -------
        LoopOpening
            Gauges and cores in original ``target`` order with optional local
            metrics.
        """
        orientation = _normalize_orientation(orientation)
        context = _normalize_context(context)
        source = _source_from_context(target, context)
        if len(source.in_dim) != 3:
            raise ValueError(
                'FixedGaugeCoreOpener requires exactly one physical site')
        self._capabilities.require(
            fixed_left=fixed_left is not None,
            fixed_right=fixed_right is not None,
            block_size=1)
        if fixed_left is None or fixed_right is None:
            raise ValueError('Both fixed gauges are required for a direct solve')
        ranks = _normalize_rank(rank, 3)

        if orientation == 'left':
            source = _mirror_source(source)
            ranks = _mirror_rank(ranks)
            fixed_left, fixed_right = (
                fixed_right.permute(2, 1, 0),
                fixed_left.permute(2, 1, 0),
            )
        if fixed_left.shape != (
                ranks[-1], source.in_dim[0], ranks[0]):
            raise ValueError('`fixed_left` shape should match target and ranks')
        if fixed_right.shape != (
                ranks[1], source.in_dim[-1], ranks[-1]):
            raise ValueError('`fixed_right` shape should match target and ranks')
        if fixed_left.device != fixed_right.device or \
                fixed_left.dtype != fixed_right.dtype:
            raise ValueError('Fixed gauges should share dtype and device')

        configurations = ConfigurationBatch(
            torch.cartesian_prod(*(
                torch.arange(dim, device=fixed_left.device)
                for dim in source.in_dim)),
            kind='indices')
        values = source.evaluate(configurations)
        if values.shape != (configurations.batch_size,):
            raise ValueError('The local target should be scalar-valued')
        if values.device != fixed_left.device or values.dtype != fixed_left.dtype:
            raise ValueError('Target values and fixed gauges should share runtime')

        left_environment = fixed_left.permute(1, 0, 2)
        right_environment = fixed_right.permute(1, 0, 2)
        environment = torch.einsum(
            'iab,jca->ijbc', left_environment, right_environment)
        identity = torch.eye(
            source.in_dim[1],
            device=values.device,
            dtype=values.dtype)
        design = torch.einsum(
            'ijbc,pq->ipjbqc', environment, identity).reshape(
                values.numel(), ranks[0] * source.in_dim[1] * ranks[1])
        solution, record = self.solver.solve(
            design,
            values.reshape(-1),
            site=1,
            sweep=0,
            return_record=context.get('collect_metrics', True))
        core = solution.reshape(
            ranks[0], source.in_dim[1], ranks[1])
        result_cores = (fixed_left, core, fixed_right)
        if orientation == 'left':
            result_cores = _mirror_cores(result_cores)
        return LoopOpening(
            left_gauge=result_cores[0],
            cores=(result_cores[1],),
            right_gauge=result_cores[2],
            rank=tuple(candidate.shape[-1] for candidate in result_cores),
            orientation=orientation,
            local_records=() if record is None else (record,),
            diagnostics={
                'algorithm': 'fixed_gauge_core_solve',
            })


class CallableLoopOpener:  # MARK: CallableLoopOpener
    """
    Adapts a callable to the loop-opening protocol.

    Parameters
    ----------
    opener : callable
        Receives the arguments of :meth:`LoopOpener.open` and returns a
        :class:`LoopOpening` or a :class:`~tensorkrowch.formats.TR` format
        carrying :class:`~tensorkrowch.decompositions.TensorDecomposition`
        diagnostics, including Quantics results.
    capabilities : LoopOpenerCapabilities, optional
        Constraints supported by the callable. Unsupported fixed gauges or
        block sizes are rejected before invoking it.
    """

    def __init__(self,
                 opener: Callable[..., Any],
                 capabilities: Optional[LoopOpenerCapabilities] = None) -> None:
        if not callable(opener):
            raise TypeError('`opener` should be callable')
        if capabilities is None:
            capabilities = LoopOpenerCapabilities()
        elif not isinstance(capabilities, LoopOpenerCapabilities):
            raise TypeError(
                '`capabilities` should be LoopOpenerCapabilities type')
        self.opener = opener
        self._capabilities = capabilities

    @property
    def capabilities(self) -> LoopOpenerCapabilities:
        """Returns declared callable capabilities."""
        return self._capabilities

    def open(self,
             target: Any,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """
        Validates constraints and normalizes the callable result.

        Parameters
        ----------
        target : torch.Tensor or BaseSource
            Local scalar tensor with axes ``(left external, *in_dim, right
            external)``, or a source representing that tensor.
        rank : int or sequence[int]
            Prescribed right-link ranks of the complete local ring, including
            the two external gauge cores. The last link closes the ring.
        fixed_left : torch.Tensor, optional
            Gauges to retain exactly. The selected strategy must support the
            supplied constraints. Their shapes follow ordinary three-axis TR
            cores.
        fixed_right : torch.Tensor, optional
            Gauges to retain exactly. The selected strategy must support the
            supplied constraints. Their shapes follow ordinary three-axis TR
            cores.
        orientation : {"right", "left"}
            Construction direction. Returned cores always follow the original
            ``target`` order, including when the algorithm uses a mirrored local
            problem.
        context : mapping, optional
            Provider metadata and internal execution options, such as
            initialization, generator and whether to collect local metrics.

        Returns
        -------
        LoopOpening
            Gauges and cores in original ``target`` order with optional local
            metrics.
        """
        orientation = _normalize_orientation(orientation)
        context = _normalize_context(context)
        in_dim = context.get('in_dim')
        block_size = 1 if in_dim is None else len(tuple(in_dim)) - 2
        self._capabilities.require(
            fixed_left=fixed_left is not None,
            fixed_right=fixed_right is not None,
            block_size=block_size)
        result = self.opener(
            target=target,
            rank=rank,
            fixed_left=fixed_left,
            fixed_right=fixed_right,
            orientation=orientation,
            context=context)
        if isinstance(result, LoopOpening):
            return result
        if isinstance(result, TensorDecomposition) and isinstance(result, TR):
            return _opening_from_result(result, orientation)
        raise TypeError('The adapted callable should return a LoopOpening')


class CompositeLoopOpener:  # MARK: CompositeLoopOpener
    """
    Initializes an opening and refines it under the requested gauge
    constraints.

    The initializer opens the target without fixed gauges. Its cores initialize
    the refiner, which imposes the final constraints. A fallback can preserve
    an initializer failure as a diagnostic and let the refiner choose its own
    init.

    Parameters
    ----------
    initializer : LoopOpener
        Unrestricted initializer, for example BLOSTRLoopOpener.
    refiner : LoopOpener
        Strategy supporting the final gauge constraints, normally
        ALSLoopOpener.
    fallback_on_error : bool
        Whether an initializer failure should allow refinement without its
        cores. The default is False. Refiner failures always propagate.
    """

    def __init__(self,
                 initializer: LoopOpener,
                 refiner: LoopOpener,
                 fallback_on_error: bool = False) -> None:
        if not isinstance(initializer, LoopOpener):
            raise TypeError('`initializer` should implement LoopOpener')
        if not isinstance(refiner, LoopOpener):
            raise TypeError('`refiner` should implement LoopOpener')
        if not isinstance(fallback_on_error, bool):
            raise TypeError('`fallback_on_error` should be bool type')
        self.initializer = initializer
        self.refiner = refiner
        self.fallback_on_error = fallback_on_error

    @property
    def capabilities(self) -> LoopOpenerCapabilities:
        """Returns constraints accepted by the refinement stage."""
        return self.refiner.capabilities

    def open(self,
             target: Any,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """
        Initializes without constraints and refines with requested gauges.

        Parameters
        ----------
        target : torch.Tensor or BaseSource
            Local scalar tensor with axes ``(left external, *in_dim, right
            external)``, or a source representing that tensor.
        rank : int or sequence[int]
            Prescribed right-link ranks of the complete local ring, including
            the two external gauge cores. The last link closes the ring.
        fixed_left : torch.Tensor, optional
            Gauges to retain exactly. The selected strategy must support the
            supplied constraints. Their shapes follow ordinary three-axis TR
            cores.
        fixed_right : torch.Tensor, optional
            Gauges to retain exactly. The selected strategy must support the
            supplied constraints. Their shapes follow ordinary three-axis TR
            cores.
        orientation : {"right", "left"}
            Construction direction. Returned cores always follow the original
            ``target`` order, including when the algorithm uses a mirrored local
            problem.
        context : mapping, optional
            Provider metadata and internal execution options, such as
            initialization, generator and whether to collect local metrics.

        Returns
        -------
        LoopOpening
            Gauges and cores in original ``target`` order with optional local
            metrics.
        """
        orientation = _normalize_orientation(orientation)
        context = _normalize_context(context)
        self.capabilities.require(
            fixed_left=fixed_left is not None,
            fixed_right=fixed_right is not None,
            block_size=(
                1 if context.get('in_dim') is None
                else len(tuple(context['in_dim'])) - 2))
        initialization_error = None
        try:
            initial = self.initializer.open(
                target,
                rank,
                orientation=orientation,
                context=context)
        except (RuntimeError, ValueError) as exc:
            if not self.fallback_on_error:
                raise
            initial = None
            initialization_error = str(exc)
        refine_context = dict(context)
        if initial is not None:
            refine_context['initial_cores'] = initial.all_cores
        result = self.refiner.open(
            target,
            rank,
            fixed_left=fixed_left,
            fixed_right=fixed_right,
            orientation=orientation,
            context=refine_context)
        diagnostics = dict(result.diagnostics)
        if initial is None:
            diagnostics['initializer'] = {
                'used': False,
                'error': initialization_error,
            }
        else:
            diagnostics['initializer'] = {
                'used': True,
                **dict(initial.diagnostics),
            }
        return LoopOpening(
            left_gauge=result.left_gauge,
            cores=result.cores,
            right_gauge=result.right_gauge,
            rank=result.rank,
            orientation=result.orientation,
            local_records=result.local_records,
            diagnostics=diagnostics)


def resolve_loop_opener(loop_opener: Any) -> LoopOpener:
    """Normalizes simple ALS presets, advanced strategies and callables."""
    def als_opener() -> ALSLoopOpener:
        return ALSLoopOpener({
            'gauge': 'none',
            'convergence': ConvergencePolicy(
                max_sweeps=100,
                rtol=1e-10,
                keep_best=True),
        })

    if isinstance(loop_opener, str):
        if loop_opener == 'als':
            return als_opener()
        if loop_opener == 'blostr+als':
            from tensorkrowch.decompositions.ring.blostr import (
                BLOSTRLoopOpener,
            )
            return CompositeLoopOpener(
                BLOSTRLoopOpener(),
                als_opener(),
                fallback_on_error=True)
        raise ValueError(
            "`loop_opener` should be 'als', 'blostr+als' or an advanced "
            'opening strategy')
    if isinstance(loop_opener, LoopOpener):
        return loop_opener
    if callable(loop_opener):
        return CallableLoopOpener(
            loop_opener,
            LoopOpenerCapabilities(
                supports_fixed_left=True,
                supports_fixed_right=True,
                supports_two_fixed_gauges=True,
                supports_blocks=True))
    raise TypeError(
        '`loop_opener` should be a supported literal, LoopOpener or callable')


__all__ = [
    'LoopOpening',
    'LoopOpenerCapabilities',
    'LoopOpener',
    'ALSLoopOpener',
    'SVDLoopOpener',
    'FixedGaugeCoreOpener',
    'CallableLoopOpener',
    'CompositeLoopOpener',
    'resolve_loop_opener',
]
