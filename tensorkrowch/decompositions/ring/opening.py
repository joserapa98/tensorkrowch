"""
This script contains:

    Classes:
        * LoopOpenerCapabilities
        * LoopOpening
        * LoopOpener
        * ALSLoopOpener
        * QRLoopOpener
        * BLOSTRLoopOpener
        * BLOSTRSVDLoopOpener
        * _FixedGaugeCoreOpener
        * CallableLoopOpener

    Functions:
        * _normalize_context
        * _normalize_orientation
        * _mirror_rank
        * _mirror_cores
        * _source_from_context
        * _mirror_source
        * _cast_source
        * _opening_from_result
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
from tensorkrowch.utils import safe_divide, scale_tensor

from tensorkrowch.decompositions._utils import _normalize_rank
from tensorkrowch.decompositions.als.convergence import ConvergencePolicy
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.als.tr import TRALS
from tensorkrowch.decompositions.metrics import LocalSolveRecord
from tensorkrowch.decompositions.results import (TRDecomposition,
                                                 TensorDecomposition)
from tensorkrowch.decompositions.sources import BaseSource, ConfigurationBatch
from tensorkrowch.decompositions.sources.callable import CallableTensorSource
from tensorkrowch.decompositions.sources.factory import as_tensor_source


_Rank = Union[int, Sequence[int]]
_Context = Optional[Mapping[str, Any]]


@dataclass(frozen=True)
class LoopOpenerCapabilities:  # MARK: LoopOpenerCapabilities
    """
    Declares the constraints accepted by a loop-opening strategy.

    Parameters
    ----------
    supports_fixed_left, supports_fixed_right : bool
        Whether the corresponding external gauge may be fixed.
    supports_two_fixed_gauges : bool
        Whether both external gauges may be fixed simultaneously.
    supports_blocks : bool
        Whether a target may contain more than one physical site.
    """

    # Whether the opener can preserve a left gauge
    supports_fixed_left: bool = False
    # Whether the opener can preserve a right gauge
    supports_fixed_right: bool = False
    # Whether both gauges can be fixed simultaneously
    supports_two_fixed_gauges: bool = False
    supports_blocks: bool = False  # Whether input blocks are supported

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
        Rejects unsupported constraints before executing a local solver.

        Parameters
        ----------
        fixed_left, fixed_right : bool
            Whether the corresponding external gauge is fixed.
        block_size : int
            Number of physical sites in the local target.
        """
        if fixed_left and not self.supports_fixed_left:
            raise ValueError(
                'The loop opener does not support a fixed left gauge')
        if fixed_right and not self.supports_fixed_right:
            raise ValueError(
                'The loop opener does not support a fixed right gauge')
        if fixed_left and fixed_right and \
                not self.supports_two_fixed_gauges:
            raise ValueError(
                'The loop opener does not support two fixed gauges')
        if (block_size > 1) and not self.supports_blocks:
            raise ValueError('The loop opener does not support input blocks')


@dataclass
class LoopOpening:  # MARK: LoopOpening
    """
    Stores gauges, local TR cores and diagnostics from one loop opening.

    ``left_gauge`` and ``right_gauge`` surround the local physical cores.
    When both are present, ``all_cores`` follows
    standard TR order and contracts to the local target. ``rank[k]`` is the
    right-link ``rank`` of ``all_cores[k]``.

    Parameters
    ----------
    left_gauge, right_gauge : torch.Tensor or None
        Three-axis external gauge cores. ``None`` omits that gauge.
    cores : sequence[torch.Tensor]
        Physical TR cores in target order, with shape
        ``(left rank, input dimension, right rank)``.
    rank : sequence[int]
        Actual right-link rank of every core in ``all_cores``.
    orientation : {"right", "left"}
        Direction used to solve the local target. Default is ``"right"``.
    local_records : sequence[LocalSolveRecord]
        Optional local least-squares measurements.
    diagnostics : mapping
        Optional algorithm metadata and preparation measurements.
    left_map, right_map : torch.Tensor, optional
        External coordinate maps with shape
        ``(new_external, local_external)``, used by :meth:`transform_gauges`.
        They leave the stored local gauges and physical cores unchanged.
    """

    left_gauge: Optional[torch.Tensor]  # Left external gauge core
    cores: Sequence[torch.Tensor]  # Raw cores in site order
    right_gauge: Optional[torch.Tensor]  # Right external gauge core
    rank: Sequence[int]  # Right-link ranks in core order
    orientation: str = 'right'  # Construction direction
    # Optional least-squares measurements for this opening
    local_records: Sequence[LocalSolveRecord] = ()
    # Local strategy diagnostics and construction state
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    # Optional left external coordinate map
    left_map: Optional[torch.Tensor] = None
    # Optional right external coordinate map
    right_map: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.orientation not in ('right', 'left'):
            raise ValueError("`orientation` should be 'right' or 'left'")
        cores = tuple(self.cores)
        if not cores:
            raise ValueError('`cores` should contain at least one input core')
        if not all(isinstance(core, torch.Tensor) and (core.ndim == 3)
                   for core in cores):
            raise ValueError(
                '`cores` should contain standard three-dimensional TR cores')
        for name in ('left_gauge', 'right_gauge'):
            gauge = getattr(self, name)
            if (gauge is not None) and \
                    (not isinstance(gauge, torch.Tensor) or (gauge.ndim != 3)):
                raise ValueError(
                    f'`{name}` should be a three-dimensional core')
        all_cores = tuple(
            core for core in (self.left_gauge, *cores, self.right_gauge)
            if core is not None)
        TRDecomposition(all_cores)
        rank = tuple(self.rank)
        actual_rank = tuple(core.shape[-1] for core in all_cores)
        if rank != actual_rank:
            raise ValueError(
                '`rank` should contain one actual right-link rank per core')
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

    def transform_gauges(self, left_map: Optional[torch.Tensor] = None,
                         right_map: Optional[torch.Tensor] = None
                         ) -> Tuple[Optional[torch.Tensor],
                                    Optional[torch.Tensor]]:
        """
        Applies external linear maps without changing the local gauges.

        Parameters
        ----------
        left_map, right_map : torch.Tensor, optional
            Matrices with shape ``(new_external, local_external)``. Omitted
            maps use those retained by this opening, or the identity.

        Returns
        -------
        tuple
            Gauges in the mapped external coordinates. No recursion, inverse
            or boundary contraction is performed.
        """
        left_map = self.left_map if left_map is None else left_map
        right_map = self.right_map if right_map is None else right_map
        left, right = self.left_gauge, self.right_gauge
        for matrix, gauge in ((left_map, left), (right_map, right)):
            if matrix is not None:
                if not isinstance(matrix, torch.Tensor) or (matrix.ndim != 2):
                    raise TypeError('Gauge coordinate maps should be matrices')
                if (gauge is None) or (matrix.shape[1] != gauge.shape[1]):
                    raise ValueError('Gauge coordinate maps should match '
                                     'their external axes')
                if matrix.device != gauge.device:
                    raise ValueError('Gauge coordinate maps and gauges '
                                     'should share a device')
        if left_map is not None:
            dtype = torch.promote_types(left_map.dtype, left.dtype)
            left = torch.einsum('ma,gar->gmr', left_map.to(dtype),
                                left.to(dtype))
        if right_map is not None:
            dtype = torch.promote_types(right_map.dtype, right.dtype)
            right = torch.einsum('rag,ma->rmg', right.to(dtype),
                                 right_map.to(dtype))
        return left, right

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
            target order, including when solving a mirrored local problem.
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
    """Restores a local factorization to the original target orientation."""
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
            'algorithm': result.metadata.get('algorithm', 'als'),
            'metadata': dict(result.metadata),
            'errors': tuple(result.metrics.errors),
            'truncations': tuple(result.metrics.truncations),
            'warnings': tuple(result.metrics.warnings),
        })


class ALSLoopOpener:  # MARK: ALSLoopOpener
    """
    Opens local loops using a reusable, encapsulated TR-ALS policy.

    Parameters
    ----------
    fit_options : mapping, optional
        Options forwarded to :meth:`TRALS.fit`. Target, rank, fixed cores,
        output device and orientation remain controlled by :meth:`open`.
        ALS sweeps visit sites in increasing order on every pass. Defaults
        use ``orthogonalize=False``, ``max_sweeps=100``, ``rtol=1e-10`` and
        ``keep_best=True``.

    Examples
    --------
    >>> opener = tk.decompositions.ALSLoopOpener({
    ...     'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=20),
    ... })
    >>> opener.capabilities.supports_blocks
    True
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
        self.fit_options.setdefault('orthogonalize', False)
        self.fit_options.setdefault(
            'convergence', ConvergencePolicy(
                max_sweeps=100, rtol=1e-10, keep_best=True))

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
            target order, including when solving a mirrored local problem.
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

        # Promote the target, initializer and fixed gauges to one common dtype.
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

        # Reflect leftward problems so forward ALS follows the outward
        # direction.
        if orientation == 'left':
            source = _mirror_source(source)
            ranks = _mirror_rank(ranks)
            fixed_left, fixed_right = (
                None if fixed_right is None else fixed_right.permute(2, 1, 0),
                None if fixed_left is None else fixed_left.permute(2, 1, 0),
            )
            if initial_cores is not None:
                initial_cores = _mirror_cores(tuple(initial_cores))

        # Environment gauges stay frozen while physical factors are optimized.
        fixed_cores = [None] * len(source.in_dim)
        fixed_cores[0] = fixed_left
        fixed_cores[-1] = fixed_right

        fit_options = dict(self.fit_options)
        if context.get('generator') is not None:
            fit_options.setdefault('generator', context['generator'])
        result = TRALS(source, out_device=None).fit(
            rank=ranks,
            initial_cores=initial_cores,
            fixed_cores=fixed_cores,
            collect_metrics=context.get('collect_metrics', True),
            **fit_options)
        return _opening_from_result(result, orientation)


class QRLoopOpener:  # MARK: QRLoopOpener
    """
    Balances external axes with QR before opening a local tensor ring.

    Free axes are factorized right first, then left using QR of its adjoint.
    Both factorizations retain their complete reduced bases: there is no
    singular-value threshold or TR-rank cap. Triangular factors are kept
    outside the initial objective and restored in the returned gauges.
    With one fixed gauge, QR factorizes its external-by-virtual matrix as
    ``L @ Q.mH``. The target is transformed by ``pinv(L)`` and
    ``Q.mH`` becomes the fixed gauge in the balanced problem. The original
    fixed gauge is restored exactly afterwards, without optimizing it.

    Fixed gauges are balanced before free axes. This changes the initial
    residual's metric and projects out target components outside the fixed
    gauge's external column space. Refinement uses the original target and
    fixed gauges, accepting only a smaller residual in those coordinates.
    With two fixed gauges, the wrapped solver is used directly, without
    balancing or ALS refinement.

    Parameters
    ----------
    opener : str or LoopOpener, optional
        Local strategy applied to the balanced target. Default is ``"als"``.
    refinement_sweeps : int, optional
        Additional ALS sweeps on the original target after restoring the
        factors. Default is one. Zero disables refinement. Only candidates
        reducing the local residual are accepted. This does not guarantee a
        smaller global reconstruction error.

    Examples
    --------
    >>> opener = tk.decompositions.QRLoopOpener(refinement_sweeps=1)
    >>> opener.capabilities.supports_blocks
    True
    """

    def __init__(self,
                 opener: Union[str, LoopOpener] = 'als',
                 refinement_sweeps: int = 1) -> None:
        self.opener = resolve_loop_opener(opener)
        if isinstance(refinement_sweeps, bool) or \
                not isinstance(refinement_sweeps, int):
            raise TypeError('`refinement_sweeps` should be int type')
        if refinement_sweeps < 0:
            raise ValueError('`refinement_sweeps` should be non-negative')
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
    def capabilities(self) -> LoopOpenerCapabilities:
        """Returns the wrapped solver's gauge and block constraints."""
        return self.opener.capabilities

    @staticmethod
    def _fixed_qr(gauge: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Separates a fixed gauge's external factor and its pseudoinverse."""
        left, external, right = gauge.shape
        matrix = gauge.permute(1, 0, 2).reshape(external, -1)
        matrix, scale = scale_tensor(matrix)
        q, factor = torch.linalg.qr(matrix.mH, mode='reduced')
        lift = factor.mH
        inverse = torch.linalg.pinv(lift) / scale
        orthogonal = q.mH.reshape(-1, left, right).permute(1, 0, 2)
        return orthogonal, lift * scale, inverse

    def open(self,
             target: torch.Tensor,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """
        Opens a dense target and restores its original external coordinates.

        Parameters
        ----------
        target : torch.Tensor
            Local tensor with left and right external axes surrounding its
            physical axes, with shape ``(left, *in_dim, right)``.
        rank : int or sequence[int]
            Prescribed right-link ranks of the complete local ring, including
            the two external gauge cores. The last link closes the ring.
        fixed_left, fixed_right : torch.Tensor, optional
            Gauges retained exactly in the returned opening. With one fixed
            gauge, its external factor is removed with a pseudoinverse and
            the orthonormal gauge stays fixed during the balanced solve.
        orientation : {"right", "left"}
            Construction direction. ALS mirrors leftward problems so its
            forward sweeps follow that direction. Returned cores always follow
            the original target order.
        context : mapping, optional
            Provider metadata, initialization, generator and metric controls.

        Returns
        -------
        LoopOpening
            Gauges and cores in the original external coordinates, with
            optional full-target refinement diagnostics.
        """
        if not isinstance(target, torch.Tensor):
            raise TypeError('`target` should be torch.Tensor type')
        if target.ndim < 3:
            raise ValueError('`target` should have at least three axes')
        context = _normalize_context(context)
        orientation = _normalize_orientation(orientation)
        self.capabilities.require(
            fixed_left=fixed_left is not None,
            fixed_right=fixed_right is not None,
            block_size=target.ndim - 2)
        runtime_dtype = target.dtype
        initial = context.get('initial_cores')
        if initial is not None:
            if isinstance(initial, (str, bytes)) or \
                    not isinstance(initial, Sequence):
                raise TypeError('`initial_cores` should be a tensor sequence')
            if len(initial) != target.ndim:
                raise ValueError(
                    '`initial_cores` should match the target axes')
            for core in initial:
                if not isinstance(core, torch.Tensor):
                    raise TypeError('`initial_cores` should contain tensors')
                runtime_dtype = torch.promote_types(runtime_dtype, core.dtype)
        for name, gauge in (('fixed_left', fixed_left),
                            ('fixed_right', fixed_right)):
            if gauge is not None:
                if not isinstance(gauge, torch.Tensor):
                    raise TypeError(f'`{name}` should be torch.Tensor type')
                axis = 0 if name == 'fixed_left' else -1
                if (gauge.ndim != 3) or \
                        (gauge.shape[1] != target.shape[axis]):
                    raise ValueError(
                        f'`{name}` should match its external target axis')
                if gauge.device != target.device:
                    raise ValueError(
                        f'`{name}` and `target` should share a device')
                runtime_dtype = torch.promote_types(runtime_dtype, gauge.dtype)
        target = target.to(dtype=runtime_dtype)
        fixed_left = None if fixed_left is None else fixed_left.to(
            dtype=runtime_dtype)
        fixed_right = None if fixed_right is None else fixed_right.to(
            dtype=runtime_dtype)
        tensor = target
        left_factor = right_factor = None
        balanced_left, balanced_right = fixed_left, fixed_right
        two_fixed = (fixed_left is not None) and (fixed_right is not None)

        # Whiten inherited coordinates before balancing the free target axes.
        # Two fixed gauges use the original two-sided least-squares metric.
        if (fixed_right is not None) and not two_fixed:
            balanced_right, lift, inverse = self._fixed_qr(fixed_right)
            tensor = torch.einsum('...t,at->...a', tensor, inverse)
            right_factor = lift.T
        if (fixed_left is not None) and not two_fixed:
            balanced_left, left_factor, inverse = self._fixed_qr(fixed_left)
            tensor = torch.einsum('as,s...->a...', inverse, tensor)

        if fixed_right is None:
            matrix = tensor.reshape(-1, tensor.shape[-1])
            q, right_factor = torch.linalg.qr(matrix, mode='reduced')
            tensor = q.reshape(*tensor.shape[:-1], q.shape[-1])
        if fixed_left is None:
            matrix = tensor.reshape(tensor.shape[0], -1)
            q, factor = torch.linalg.qr(matrix.mH, mode='reduced')
            left_factor = factor.mH
            tensor = q.mH.reshape(q.shape[-1], *tensor.shape[1:])

        local_context = {**context, 'in_dim': tuple(tensor.shape),
                         'dtype': tensor.dtype, 'device': tensor.device}
        if context.get('initial_cores') is not None:
            initial = [core.to(dtype=runtime_dtype) for core in initial]
            if fixed_left is not None:
                initial[0] = balanced_left
            elif left_factor is not None:
                initial[0] = torch.einsum(
                    'as,gsr->gar', torch.linalg.pinv(left_factor), initial[0])
            if fixed_right is not None:
                initial[-1] = balanced_right
            elif right_factor is not None:
                initial[-1] = torch.einsum(
                    'rtg,tb->rbg', initial[-1],
                    torch.linalg.pinv(right_factor))
            local_context['initial_cores'] = tuple(initial)
        opening = self.opener.open(
            tensor, rank, fixed_left=balanced_left, fixed_right=balanced_right,
            orientation=orientation, context=local_context)
        left_gauge, right_gauge = opening.left_gauge, opening.right_gauge
        if fixed_left is not None:
            left_gauge = fixed_left
        elif left_factor is not None:
            left_gauge = torch.einsum(
                'sa,gar->gsr', left_factor.to(left_gauge.dtype), left_gauge)
        if fixed_right is not None:
            right_gauge = fixed_right
        elif right_factor is not None:
            right_gauge = torch.einsum(
                'rbg,bt->rtg', right_gauge,
                right_factor.to(right_gauge.dtype))
        initial = LoopOpening(
            left_gauge=left_gauge, cores=opening.cores,
            right_gauge=right_gauge, rank=opening.rank,
            orientation=orientation, local_records=opening.local_records,
            diagnostics={**opening.diagnostics,
                         'qr_left': left_factor is not None,
                         'qr_right': right_factor is not None})
        if (self.refiner is None) or two_fixed:
            return initial

        before = torch.linalg.vector_norm(target - initial.contract_dense())
        refined = self.refiner.open(
            target, initial.rank, fixed_left=fixed_left,
            fixed_right=fixed_right, orientation=orientation,
            context={**context, 'in_dim': tuple(target.shape),
                     'initial_cores': initial.all_cores})
        candidate = torch.linalg.vector_norm(target - refined.contract_dense())
        accepted = bool(torch.isfinite(candidate) & (candidate < before))
        selected = refined if accepted else initial
        after = candidate if accepted else before
        denominator = torch.linalg.vector_norm(target)
        return LoopOpening(
            left_gauge=selected.left_gauge, cores=selected.cores,
            right_gauge=selected.right_gauge, rank=selected.rank,
            orientation=orientation,
            local_records=initial.local_records + refined.local_records,
            diagnostics={**initial.diagnostics, 'qr_refinement': {
                'accepted': accepted,
                'relative_error_before': safe_divide(before, denominator),
                'relative_error_after': safe_divide(after, denominator),
                'metadata': refined.diagnostics['metadata'],
            }})


class BLOSTRLoopOpener:  # MARK: BLOSTRLoopOpener
    """
    Opens exact local tensor rings with the complete BLOSTR algorithm.

    Parameters
    ----------
    fit_options : mapping, optional
        Spectral projection and clustering options forwarded to
        :meth:`BLOSTR.fit`. Target, rank, output device and metric collection
        are controlled by :meth:`open`. Fixed gauges are unsupported.

    Examples
    --------
    >>> opener = tk.decompositions.BLOSTRLoopOpener({'n_attempts': 4})
    >>> opener.capabilities.supports_fixed_left
    False
    """

    exact = True
    _capabilities = LoopOpenerCapabilities(supports_blocks=True)

    def __init__(self,
                 fit_options: Optional[Mapping[str, Any]] = None) -> None:
        if fit_options is None:
            fit_options = {}
        if not isinstance(fit_options, Mapping):
            raise TypeError('`fit_options` should be a mapping or None')
        reserved = {'source', 'rank', 'out_device',
                    'collect_metrics', 'verbose'}
        overlap = reserved.intersection(fit_options)
        if overlap:
            raise ValueError(
                f'`fit_options` should not override {sorted(overlap)}')
        self.fit_options = dict(fit_options)

    @property
    def capabilities(self) -> LoopOpenerCapabilities:
        """Returns unrestricted block support without fixed gauges."""
        return self._capabilities

    def _options(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        """Combines constructor options with the active local orientation."""
        options = dict(self.fit_options)
        if context.get('generator') is not None:
            options.setdefault('generator', context['generator'])
        if context.get('_spectral_orientation') == 'left':
            if options.get('slices') is not None:
                options['slices'] = tuple(
                    tuple(reversed(configuration))
                    for configuration in options['slices'])
        return options

    def _decompose(self,
                   source: BaseSource,
                   rank: Optional[_Rank],
                   context: Mapping[str, Any]) -> TensorDecomposition:
        """Runs the complete spectral factorization."""
        from tensorkrowch.decompositions.ring.blostr import BLOSTR
        return BLOSTR(source, out_device=None).fit(
            rank=rank,
            collect_metrics=context.get('collect_metrics', False),
            **self._options(context))

    def decompose(self,
                  target: Any,
                  rank: Optional[_Rank] = None,
                  *,
                  context: _Context = None) -> TensorDecomposition:
        """
        Decomposes a complete target without adding external gauge cores.

        Parameters
        ----------
        target : torch.Tensor or BaseSource
            Tensor with one physical axis per resulting TR core. Sources
            are materialized before applying the spectral construction.
        rank : int or sequence[int], optional
            Right-link TR ranks. The last entry is the cyclic link. ``None``
            discovers ranks from spectral multiplicities.
        context : mapping, optional
            Provider metadata, generator and metric controls.

        Returns
        -------
        TensorDecomposition
            Spectral factorization retaining the source's axis semantics.
        """
        context = _normalize_context(context)
        source = _source_from_context(target, context)
        return self._decompose(source, rank, context)

    def open(self,
             target: Any,
             rank: Optional[_Rank] = None,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """
        Opens an unrestricted exact local target, discovering omitted ranks.

        Parameters
        ----------
        target : torch.Tensor or BaseSource
            Local tensor with left and right external axes surrounding its
            physical axes, with shape ``(left, *in_dim, right)``.
            Sources are materialized before the spectral construction.
        rank : int or sequence[int], optional
            Right-link TR ranks, including both gauges. ``None`` discovers
            ranks from spectral multiplicities.
        fixed_left, fixed_right : torch.Tensor, optional
            Unsupported constraints. Passing either gauge raises an error.
        orientation : {"right", "left"}
            Construction direction. Leftward targets are reflected internally;
            returned cores follow the original target order.
        context : mapping, optional
            Provider metadata, generator and metric controls.

        Returns
        -------
        LoopOpening
            Spectrally recovered gauges and physical cores.
        """
        context = _normalize_context(context)
        orientation = _normalize_orientation(orientation)
        source = _source_from_context(target, context)
        self.capabilities.require(
            fixed_left=fixed_left is not None,
            fixed_right=fixed_right is not None,
            block_size=source.n_sites - 2)
        if orientation == 'left':
            source = _mirror_source(source)
            if rank is not None:
                rank = _mirror_rank(_normalize_rank(rank, source.n_sites))
        context['_spectral_orientation'] = orientation
        result = self.decompose(source, rank, context=context)
        return _opening_from_result(result, orientation)


class BLOSTRSVDLoopOpener(BLOSTRLoopOpener):  # MARK: BLOSTRSVDLoopOpener
    """
    Recovers the first local gauge with BLOSTR and splits the tail by SVD.

    Parameters
    ----------
    fit_options : mapping, optional
        Options forwarded to :meth:`BLOSTRSVD.fit`, including spectral
        projections, clustering and tail truncation. Target, rank, output
        device and metric collection are controlled by :meth:`open`.

    Examples
    --------
    >>> opener = tk.decompositions.BLOSTRSVDLoopOpener({
    ...     'projection': 'gaussian',
    ... })
    >>> opener.exact
    True
    """

    def _decompose(self,
                   source: BaseSource,
                   rank: Optional[_Rank],
                   context: Mapping[str, Any]) -> TensorDecomposition:
        """Runs first-core spectral recovery followed by TT-SVD."""
        from tensorkrowch.decompositions.ring.blostr import BLOSTRSVD
        return BLOSTRSVD(source, out_device=None).fit(
            rank=rank,
            collect_metrics=context.get('collect_metrics', False),
            **self._options(context))


class _FixedGaugeCoreOpener:  # MARK: _FixedGaugeCoreOpener
    """
    Solves the single remaining input core with both gauges fixed.

    The external design is factored once for all physical slices using
    LeastSquaresSolver, without expanding physical identity blocks.
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
            target order, including when solving a mirrored local problem.
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
        if (fixed_left is None) or (fixed_right is None):
            raise ValueError(
                'Both fixed gauges are required for a direct solve')
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
            raise ValueError(
                '`fixed_left` shape should match target and ranks')
        if fixed_right.shape != (
                ranks[1], source.in_dim[-1], ranks[-1]):
            raise ValueError(
                '`fixed_right` shape should match target and ranks')
        if (fixed_left.device != fixed_right.device) or \
                (fixed_left.dtype != fixed_right.dtype):
            raise ValueError('Fixed gauges should share dtype and device')

        configurations = ConfigurationBatch(
            torch.cartesian_prod(*(
                torch.arange(dim, device=fixed_left.device)
                for dim in source.in_dim)),
            kind='indices')
        values = source.evaluate(configurations)
        if values.shape != (configurations.batch_size,):
            raise ValueError('The local target should be scalar-valued')
        if (values.device != fixed_left.device) or \
                (values.dtype != fixed_left.dtype):
            raise ValueError(
                'Target values and fixed gauges should share dtype and device')

        # Solve the complete two-sided design rather than inverting gauges
        # separately; this preserves the actual least-squares objective.
        left_environment = fixed_left.permute(1, 0, 2)
        right_environment = fixed_right.permute(1, 0, 2)
        environment = torch.einsum(
            'iab,jca->ijbc', left_environment, right_environment)
        # Every physical slice shares this two-sided external design.
        design = environment.reshape(-1, ranks[0] * ranks[1])
        target_matrix = values.reshape(source.in_dim).movedim(1, -1).reshape(
            -1, source.in_dim[1])
        solution, record = self.solver._solve(
            design, target_matrix, site=1, sweep=0,
            return_record=context.get('collect_metrics', True),
            n_copies=source.in_dim[1])
        core = solution.reshape(
            ranks[0], ranks[1], source.in_dim[1]).transpose(1, 2)
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
                 capabilities: Optional[LoopOpenerCapabilities] = None
                 ) -> None:
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
            target order, including when solving a mirrored local problem.
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
        if (in_dim is None) and isinstance(target, torch.Tensor):
            in_dim = target.shape
        elif (in_dim is None) and isinstance(target, BaseSource):
            in_dim = target.in_dim
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


def resolve_loop_opener(loop_opener: Any) -> LoopOpener:
    """
    Resolves ALS, exact spectral strategies and custom local solvers.

    Parameters
    ----------
    loop_opener : {"als", "blostr", "blostr_svd"}, LoopOpener or callable
        Local strategy. Callables receive the arguments of
        :meth:`LoopOpener.open` and return a :class:`LoopOpening`.

    Returns
    -------
    LoopOpener
        Strategy implementing local loop opening.
    """
    if isinstance(loop_opener, str):
        if loop_opener == 'als':
            return ALSLoopOpener()
        if loop_opener == 'blostr':
            return BLOSTRLoopOpener()
        if loop_opener == 'blostr_svd':
            return BLOSTRSVDLoopOpener()
        raise ValueError(
            "`loop_opener` should be 'als', 'blostr', 'blostr_svd' or "
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
    'QRLoopOpener',
    'BLOSTRLoopOpener',
    'BLOSTRSVDLoopOpener',
    'CallableLoopOpener',
    'resolve_loop_opener',
]
