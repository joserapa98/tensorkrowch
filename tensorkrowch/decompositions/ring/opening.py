"""Strategies and contracts for opening local tensor-network loops."""

from dataclasses import dataclass, field
from typing import (Any, Callable, Mapping, Optional, Protocol, Sequence, Tuple,
                    Union, runtime_checkable)

import torch

from tensorkrowch.decompositions.als.convergence import ConvergencePolicy
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.als.tr import TRALS
from tensorkrowch.decompositions.metrics import LocalSolveRecord
from tensorkrowch.decompositions.results import TRDecomposition
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 TensorSource,
                                                 as_tensor_source)


_Rank = Union[int, Sequence[int]]
_Context = Optional[Mapping[str, Any]]


@dataclass(frozen=True)
class LoopOpenerCapabilities:
    """Declares the constraints accepted by a loop-opening strategy."""

    supports_fixed_left: bool = False
    supports_fixed_right: bool = False
    supports_two_fixed_gauges: bool = False
    supports_blocks: bool = False

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
        """Raises before execution when requested constraints are unsupported."""
        if fixed_left and not self.supports_fixed_left:
            raise ValueError('The loop opener does not support a fixed left gauge')
        if fixed_right and not self.supports_fixed_right:
            raise ValueError('The loop opener does not support a fixed right gauge')
        if fixed_left and fixed_right and \
                not self.supports_two_fixed_gauges:
            raise ValueError(
                'The loop opener does not support two fixed gauges')
        if block_size > 1 and not self.supports_blocks:
            raise ValueError('The loop opener does not support physical blocks')


@dataclass(frozen=True)
class LoopOpening:
    """Stores gauges, local TR cores and diagnostics from one loop opening.

    ``left_gauge`` and ``right_gauge`` are the environment cores surrounding
    the local physical ``cores``. When both are present, ``all_cores`` follows
    standard TR order and contracts to the local target. ``rank[k]`` is the
    right-link rank of ``all_cores[k]``.
    """

    left_gauge: Optional[torch.Tensor]
    cores: Sequence[torch.Tensor]
    right_gauge: Optional[torch.Tensor]
    rank: Sequence[int]
    orientation: str = 'right'
    local_records: Sequence[LocalSolveRecord] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.orientation not in ('right', 'left'):
            raise ValueError("`orientation` should be 'right' or 'left'")
        cores = tuple(self.cores)
        if not cores:
            raise ValueError('`cores` should contain at least one physical core')
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
        """Returns gauges and physical cores in standard cyclic order."""
        return tuple(
            core for core in (self.left_gauge, *self.cores, self.right_gauge)
            if core is not None)

    def contract_dense(self) -> torch.Tensor:
        """Contracts the complete local opening without constructing a model."""
        return TRDecomposition(self.all_cores).contract_dense()


@runtime_checkable
class LoopOpener(Protocol):
    """Protocol implemented by local loop-opening strategies."""

    @property
    def capabilities(self) -> LoopOpenerCapabilities:
        """Returns constraints supported by this strategy."""

    def open(self,
             target,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """Opens one local target with optional fixed incoming gauges."""


def _normalize_context(context: _Context) -> Mapping[str, Any]:
    """Copies an optional opening context after validating its type."""
    if context is None:
        return {}
    if not isinstance(context, Mapping):
        raise TypeError('`context` should be a mapping or None')
    return dict(context)


def _normalize_orientation(orientation: str) -> str:
    """Validates the explicit opening orientation."""
    if not isinstance(orientation, str):
        raise TypeError('`orientation` should be str type')
    if orientation not in ('right', 'left'):
        raise ValueError("`orientation` should be 'right' or 'left'")
    return orientation


def _normalize_rank(rank: _Rank, n_sites: int) -> Tuple[int, ...]:
    """Normalizes a shared rank or one right-link rank per local site."""
    if isinstance(rank, bool):
        raise TypeError('`rank` should be int or a sequence of ints')
    if isinstance(rank, int):
        ranks = (rank,) * n_sites
    else:
        if isinstance(rank, (str, bytes)):
            raise TypeError('`rank` should be int or a sequence of ints')
        try:
            ranks = tuple(rank)
        except TypeError as exc:
            raise TypeError(
                '`rank` should be int or a sequence of ints') from exc
        if len(ranks) != n_sites:
            raise ValueError(
                '`rank` should contain one right-link value per local site')
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in ranks):
        raise TypeError('Local ranks should be integers')
    if any(value < 1 for value in ranks):
        raise ValueError('Local ranks should be positive')
    return ranks


def _mirror_rank(rank: Sequence[int]) -> Tuple[int, ...]:
    """Maps right-link ranks through reversal and core transposition."""
    return (*reversed(rank[:-1]), rank[-1])


def _mirror_cores(cores: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """Mirrors standard TR cores while preserving their represented tensor."""
    return tuple(core.permute(2, 1, 0) for core in reversed(cores))


def _source_from_context(target, context: Mapping[str, Any]) -> TensorSource:
    """Normalizes a local target through the shared TensorSource contract."""
    return as_tensor_source(
        target,
        input_dim=context.get('input_dim'),
        output_shape=(),
        dtype=context.get('dtype'),
        device=context.get('device', 'cpu'),
        batch_size=context.get('batch_size'))


def _mirror_source(source: TensorSource) -> TensorSource:
    """Creates a lazy source whose variables are in reverse site order."""
    def evaluate(indices: torch.Tensor) -> torch.Tensor:
        configurations = ConfigurationBatch(
            indices.flip(1), kind='indices')
        return source.evaluate(configurations)

    return as_tensor_source(
        evaluate,
        input_dim=tuple(reversed(source.input_dim)),
        output_shape=(),
        dtype=source.dtype,
        device=source.device)


def _cast_source(source: TensorSource,
                 dtype: torch.dtype) -> TensorSource:
    """Lazily casts target values to an initializer's numerical dtype."""
    if source.dtype == dtype:
        return source

    def evaluate(indices: torch.Tensor) -> torch.Tensor:
        configurations = ConfigurationBatch(indices, kind='indices')
        return source.evaluate(configurations).to(dtype=dtype)

    return as_tensor_source(
        evaluate,
        input_dim=source.input_dim,
        output_shape=(),
        dtype=dtype,
        device=source.device)


def _opening_from_result(result: TRDecomposition,
                         orientation: str) -> LoopOpening:
    """Converts an ALS result back from the requested orientation."""
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


class ALSLoopOpener:
    """Opens local loops using a reusable, encapsulated TR-ALS policy.

    Parameters
    ----------
    fit_options : mapping, optional
        Options forwarded to :meth:`TRALS.fit`. Target, rank, fixed cores,
        output device and orientation remain controlled by :meth:`open`.
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
             target,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """Fits a local TR with zero or one fixed environment gauge."""
        orientation = _normalize_orientation(orientation)
        context = _normalize_context(context)
        source = _source_from_context(target, context)
        initial_cores = context.get('initial_cores')
        runtime_dtype = source.dtype
        if initial_cores is not None:
            if isinstance(initial_cores, torch.Tensor):
                raise TypeError(
                    '`initial_cores` should contain torch.Tensor objects')
            try:
                initial_cores = tuple(initial_cores)
            except TypeError as exc:
                raise TypeError(
                    '`initial_cores` should contain torch.Tensor objects') \
                    from exc
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
        if len(source.input_dim) < 3:
            raise ValueError(
                'A local loop opening should contain left, physical and right '
                'input dimensions')
        ranks = _normalize_rank(rank, len(source.input_dim))
        self.capabilities.require(
            fixed_left=fixed_left is not None,
            fixed_right=fixed_right is not None,
            block_size=len(source.input_dim) - 2)

        if orientation == 'left':
            source = _mirror_source(source)
            ranks = _mirror_rank(ranks)
            fixed_left, fixed_right = (
                None if fixed_right is None else fixed_right.permute(2, 1, 0),
                None if fixed_left is None else fixed_left.permute(2, 1, 0),
            )
            if initial_cores is not None:
                initial_cores = _mirror_cores(tuple(initial_cores))
        fixed_cores = [None] * len(source.input_dim)
        fixed_cores[0] = fixed_left
        fixed_cores[-1] = fixed_right

        fit_options = dict(self.fit_options)
        fit_options.setdefault(
            'convergence', ConvergencePolicy(max_sweeps=10))
        if context.get('generator') is not None:
            fit_options.setdefault('generator', context['generator'])
        result = TRALS(source, output_device=None).fit(
            rank=ranks,
            initial_cores=initial_cores,
            fixed_cores=fixed_cores,
            collect_metrics=True,
            **fit_options)
        return _opening_from_result(result, orientation)


class FixedGaugeCoreOpener:
    """Solves one physical core directly between two fixed gauges."""

    _capabilities = LoopOpenerCapabilities(
        supports_fixed_left=True,
        supports_fixed_right=True,
        supports_two_fixed_gauges=True,
        supports_blocks=False)

    def __init__(self,
                 solver: Optional[LeastSquaresSolver] = None) -> None:
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
             target,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """Solves the unique unknown core by one dense least-squares system."""
        orientation = _normalize_orientation(orientation)
        context = _normalize_context(context)
        source = _source_from_context(target, context)
        if source.input_dim.__len__() != 3:
            raise ValueError(
                'FixedGaugeCoreOpener requires exactly one physical site')
        self.capabilities.require(
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
                ranks[-1], source.input_dim[0], ranks[0]):
            raise ValueError('`fixed_left` shape should match target and ranks')
        if fixed_right.shape != (
                ranks[1], source.input_dim[-1], ranks[-1]):
            raise ValueError('`fixed_right` shape should match target and ranks')
        if fixed_left.device != fixed_right.device or \
                fixed_left.dtype != fixed_right.dtype:
            raise ValueError('Fixed gauges should share dtype and device')

        configurations = ConfigurationBatch(
            torch.cartesian_prod(*(
                torch.arange(dim, device=fixed_left.device)
                for dim in source.input_dim)),
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
            source.input_dim[1],
            device=values.device,
            dtype=values.dtype)
        design = torch.einsum(
            'ijbc,pq->ipjbqc', environment, identity).reshape(
                values.numel(), ranks[0] * source.input_dim[1] * ranks[1])
        solution, record = self.solver.solve(
            design,
            values.reshape(-1),
            site=1,
            sweep=0,
            return_record=True)
        core = solution.reshape(
            ranks[0], source.input_dim[1], ranks[1])
        result_cores = (fixed_left, core, fixed_right)
        if orientation == 'left':
            result_cores = _mirror_cores(result_cores)
        return LoopOpening(
            left_gauge=result_cores[0],
            cores=(result_cores[1],),
            right_gauge=result_cores[2],
            rank=tuple(candidate.shape[-1] for candidate in result_cores),
            orientation=orientation,
            local_records=(record,),
            diagnostics={
                'algorithm': 'fixed_gauge_core_solve',
                'residual_absolute': record.residual_absolute,
                'residual_relative': record.residual_relative,
            })


class CallableLoopOpener:
    """Adapts a compatible callable to the :class:`LoopOpener` protocol."""

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
             target,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """Validates constraints and normalizes the callable result."""
        orientation = _normalize_orientation(orientation)
        context = _normalize_context(context)
        input_dim = context.get('input_dim')
        block_size = 1 if input_dim is None else len(tuple(input_dim)) - 2
        self.capabilities.require(
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
        if isinstance(result, TRDecomposition):
            return _opening_from_result(result, orientation)
        raise TypeError('The adapted callable should return a LoopOpening')


class CompositeLoopOpener:
    """Uses one unrestricted opener as initialization for a refining opener."""

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
             target,
             rank: _Rank,
             *,
             fixed_left: Optional[torch.Tensor] = None,
             fixed_right: Optional[torch.Tensor] = None,
             orientation: str = 'right',
             context: _Context = None) -> LoopOpening:
        """Initializes without constraints and refines with requested gauges."""
        orientation = _normalize_orientation(orientation)
        context = _normalize_context(context)
        self.capabilities.require(
            fixed_left=fixed_left is not None,
            fixed_right=fixed_right is not None,
            block_size=(
                1 if context.get('input_dim') is None
                else len(tuple(context['input_dim'])) - 2))
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


__all__ = [
    'LoopOpening',
    'LoopOpenerCapabilities',
    'LoopOpener',
    'ALSLoopOpener',
    'FixedGaugeCoreOpener',
    'CallableLoopOpener',
    'CompositeLoopOpener',
]
