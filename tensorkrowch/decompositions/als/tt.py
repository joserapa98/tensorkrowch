"""Exact tensor-train alternating least-squares decompositions."""

from dataclasses import replace
from math import prod
from typing import (Optional, Sequence, Tuple, Union)

import torch

from tensorkrowch.decompositions.als.convergence import (ConvergencePolicy,
                                                         UpdatePolicy)
from tensorkrowch.decompositions.als.driver import ALSSweepDriver
from tensorkrowch.decompositions.als.environments import (CoreUpdateSet,
                                                          TTEnvironmentCache)
from tensorkrowch.decompositions.als.gauges import (GaugePolicy, NoGauge,
                                                    resolve_gauge_policy)
from tensorkrowch.decompositions.als.problem import ALSProblem
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.observers import (DecompositionObserver,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 as_tensor_source)
from tensorkrowch.decompositions.sources.base import _unravel_indices
from tensorkrowch.decompositions.svd.tt import TTSVD


def _standard_tt_cores(
        cores: Union[TTDecomposition, Sequence[torch.Tensor]],
        input_dim: Sequence[int]) -> Tuple[torch.Tensor, ...]:
    """Normalizes lightweight or standard OBC core shapes."""
    if isinstance(cores, TTDecomposition):
        if cores.n_batches:
            raise ValueError('Batched TT cores are not supported by TT-ALS')
        cores = cores.cores
    elif isinstance(cores, torch.Tensor):
        raise TypeError(
            '`initial_cores` should be a TTDecomposition or a core sequence')
    try:
        cores = tuple(cores)
    except TypeError as exc:
        raise TypeError(
            '`initial_cores` should be a TTDecomposition or a core sequence') \
            from exc
    if len(cores) != len(input_dim):
        raise ValueError('`initial_cores` should contain one core per site')
    if not all(isinstance(core, torch.Tensor) for core in cores):
        raise TypeError('`initial_cores` should contain torch.Tensor objects')

    standard = []
    n_sites = len(cores)
    for site, (core, site_input_dim) in enumerate(zip(cores, input_dim)):
        if n_sites == 1:
            if core.ndim == 1:
                core = core.reshape(1, core.shape[0], 1)
            elif (core.ndim != 3) or \
                    (core.shape[0] != 1) or (core.shape[-1] != 1):
                raise ValueError(
                    'A one-site core should have shape (input,) or '
                    '(1, input, 1)')
        elif site == 0:
            if core.ndim == 2:
                core = core.unsqueeze(0)
            elif (core.ndim != 3) or (core.shape[0] != 1):
                raise ValueError(
                    'The first core should have shape (input, right rank) or '
                    '(1, input, right rank)')
        elif site == (n_sites - 1):
            if core.ndim == 2:
                core = core.unsqueeze(-1)
            elif (core.ndim != 3) or (core.shape[-1] != 1):
                raise ValueError(
                    'The last core should have shape (left rank, input) or '
                    '(left rank, input, 1)')
        elif core.ndim != 3:
            raise ValueError(
                'Interior cores should have left rank, input and right rank '
                'dimensions')
        if core.shape[1] != site_input_dim:
            raise ValueError(
                '`initial_cores` input dimensions should match the source')
        standard.append(core)

    return TTEnvironmentCache._validate_cores(tuple(standard))


def _result_tt_cores(
        cores: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    """Removes explicit unit ranks from TT boundary cores."""
    cores = tuple(cores)
    if len(cores) == 1:
        return (cores[0].squeeze(0).squeeze(-1),)
    return (cores[0].squeeze(0), *cores[1:-1], cores[-1].squeeze(-1))


def _feasible_tt_rank(input_dim: Sequence[int], rank: int) -> Tuple[int, ...]:
    """Clips one shared rank cap to every algebraically feasible TT cut."""
    ranks = []
    for cut in range(1, len(input_dim)):
        ranks.append(min(
            rank,
            prod(input_dim[:cut]),
            prod(input_dim[cut:])))
    return tuple(ranks)


def _contract_standard_tt(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Contracts standard OBC cores without constructing a model graph."""
    result = cores[0].squeeze(0)
    for core in cores[1:]:
        result = torch.einsum('...a,apb->...pb', result, core)
    return result.squeeze(-1)


def _relative_error(absolute: torch.Tensor,
                    target_norm: torch.Tensor) -> torch.Tensor:
    """Applies the decomposition-wide zero-target relative-error policy."""
    if target_norm > 0:
        return absolute / target_norm
    if absolute == 0:
        return torch.zeros_like(absolute)
    return torch.full_like(absolute, torch.inf)


class _TTALSBackend:
    """Adapts exact TT contractions to the topology-independent driver."""

    def __init__(self,
                 problem: ALSProblem,
                 cores: Sequence[torch.Tensor],
                 target: torch.Tensor,
                 solver: LeastSquaresSolver,
                 gauge: GaugePolicy,
                 fixed_sites: Sequence[int],
                 renormalize: bool) -> None:
        self.problem = problem
        self.target = target.reshape(-1)
        self.solver = solver
        self.gauge = gauge
        self.fixed_sites = frozenset(fixed_sites)
        self.cache = TTEnvironmentCache(
            cores, renormalize=renormalize)
        self._direction = None

    @property
    def n_sites(self) -> int:
        return len(self.cache.cores)

    @property
    def trainable_sites(self) -> Sequence[int]:
        return tuple(site for site in range(self.n_sites)
                     if site not in self.fixed_sites)

    @property
    def cores(self) -> Sequence[torch.Tensor]:
        return self.cache.cores

    def prepare_sweep(self,
                      order: Sequence[int],
                      sweep: int) -> Tuple[Optional[int], bool]:
        self.cache.prepare_sweep(order)
        self._direction = 'forward' if order[0] == 0 else 'reverse'
        return None, False

    def _updated_record(self,
                        record,
                        environment: torch.Tensor,
                        target: torch.Tensor,
                        proposal: torch.Tensor):
        """Updates diagnostics when damping or acceptance changes a solve."""
        if record is None:
            return None
        residual = environment @ proposal.reshape(-1) - target
        residual_absolute = torch.linalg.vector_norm(residual)
        target_norm = torch.linalg.vector_norm(target)
        residual_relative = _relative_error(residual_absolute, target_norm)
        return replace(
            record,
            residual_absolute=residual_absolute,
            residual_relative=residual_relative,
            target_norm=target_norm)

    def solve_site(self,
                   site: int,
                   sweep: int,
                   update_policy: UpdatePolicy,
                   return_record: bool):
        local_environment = self.cache.local_environment(site)
        environment = local_environment.design()
        target = local_environment.scale_target(self.target)
        if self.problem.weights is not None:
            weights = self.problem.weights.reshape(-1).to(environment.dtype)
            environment = environment * weights.unsqueeze(1)
            target = target * weights

        regularization_scale = None
        if (self.solver.l2_reg_mode == 'absolute') and \
                (self.solver.l2_reg > 0):
            regularization_scale = (-2 * local_environment.log_scale).exp()
        solution, record = self.solver.solve(
            environment,
            target,
            site=site,
            sweep=sweep,
            return_record=return_record,
            regularization_scale=regularization_scale)

        current = self.cache.cores[site]
        proposal = solution.reshape(current.shape)
        proposal = update_policy.apply(current, proposal)
        record_needs_update = return_record and \
            (update_policy.damping != 1)
        if update_policy.acceptance == 'non_increasing':
            current_error = torch.linalg.vector_norm(
                environment @ current.reshape(-1) - target)
            proposal_error = torch.linalg.vector_norm(
                environment @ proposal.reshape(-1) - target)
            if not update_policy.accepts(
                    float(current_error.detach().cpu().item()),
                    float(proposal_error.detach().cpu().item())):
                proposal = current
            record_needs_update = return_record
        if record_needs_update:
            record = self._updated_record(
                record, environment, target, proposal)

        receiver = site + 1 if self._direction == 'forward' else site - 1
        legal_receiver = (0 <= receiver < self.n_sites) and \
            (receiver not in self.fixed_sites)
        gauge = self.gauge if legal_receiver else NoGauge()
        gauged_core, factor = gauge.factor(proposal, self._direction)

        sites = [site]
        cores = [gauged_core]
        versions = [self.cache.core_versions[site] + 1]
        if factor is not None:
            neighbor = gauge.absorb(
                factor, self.cache.cores[receiver], self._direction)
            sites.append(receiver)
            cores.append(neighbor)
            versions.append(self.cache.core_versions[receiver] + 1)
        return CoreUpdateSet(
            sites=sites,
            cores=cores,
            versions=versions,
            reason='local_solve'), record

    def skip_site(self, site: int) -> None:
        self.cache.local_environment(site)
        self.cache.commit(CoreUpdateSet(
            sites=(site,),
            cores=(self.cache.cores[site],),
            versions=(self.cache.core_versions[site] + 1,),
            reason='fixed_core'))

    def commit(self, update_set: CoreUpdateSet) -> None:
        self.cache.commit(update_set)

    def measure_objective(
            self, problem: ALSProblem) -> Tuple[torch.Tensor, torch.Tensor]:
        approximation = _contract_standard_tt(self.cache.cores).reshape(-1)
        residual = approximation - self.target
        target = self.target
        if problem.weights is not None:
            weights = problem.weights.reshape(-1).to(residual.dtype)
            residual = residual * weights
            target = target * weights
        absolute = torch.linalg.vector_norm(residual)
        target_norm = torch.linalg.vector_norm(target)
        return absolute, _relative_error(absolute, target_norm)

    def snapshot(self) -> Sequence[torch.Tensor]:
        return tuple(core.clone() for core in self.cache.cores)

    def restore(self, cores: Sequence[torch.Tensor]) -> None:
        self.cache = TTEnvironmentCache(
            cores, renormalize=self.cache.renormalize)


class TTALS:
    """Approximates one fixed scalar tensor source by exact TT-ALS.

    The source and its discrete input dimensions are fixed on construction.
    Repeated calls to :meth:`fit` may then compare initializations, ranks,
    gauges, solvers and convergence policies without rebuilding the source
    adapter. Exact ALS enumerates the complete discrete tensor once and caches
    those values for every local solve and global objective evaluation.

    Parameters
    ----------
    source : TensorSource, TTDecomposition, torch.Tensor or callable
        Scalar tensor or function to approximate. A callable receives batches
        of integer configurations with shape ``(batch, sites)``.
    input_dim : sequence of int, optional
        Input dimension at every site. It is required for callables and may be
        omitted when the source already declares it.
    dtype : torch.dtype, optional
        Callable output dtype. Existing sources and tensors retain their dtype.
    device : str or torch.device, optional
        Device used by a callable source. Existing sources retain their device.
    batch_size : int, optional
        Maximum callable evaluation batch used while enumerating the target.
    output_device : str or torch.device, optional
        Device where finalized cores are stored. The default is ``"cpu"``;
        ``None`` keeps them on the computation device.
    """

    def __init__(self,
                 source,
                 input_dim: Optional[Sequence[int]] = None,
                 *,
                 dtype: Optional[torch.dtype] = None,
                 device: Union[str, torch.device] = 'cpu',
                 batch_size: Optional[int] = None,
                 output_device: Optional[
                     Union[str, torch.device]] = 'cpu') -> None:
        self.source = as_tensor_source(
            source,
            input_dim=input_dim,
            output_shape=(),
            dtype=dtype,
            device=device,
            batch_size=batch_size)
        self.output_device = None if output_device is None \
            else torch.device(output_device)
        self._configurations = None
        self._target = None

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Input dimension fixed by the tensor source."""
        return self.source.input_dim

    def _exact_target(self) -> Tuple[ConfigurationBatch, torch.Tensor]:
        """Enumerates and caches the scalar source in global row order."""
        if self._target is None:
            flat_ids = torch.arange(
                prod(self.input_dim), device=self.source.device)
            indices = _unravel_indices(flat_ids, self.input_dim)
            configurations = ConfigurationBatch(indices, kind='indices')
            target = self.source.evaluate(configurations)
            if target.shape != (flat_ids.numel(),):
                raise ValueError(
                    'TT-ALS currently requires a scalar tensor source')
            if not (target.is_floating_point() or target.is_complex()):
                raise TypeError(
                    'The tensor source should return floating or complex values')
            if not torch.isfinite(target).all():
                raise ValueError(
                    'The tensor source should return only finite values')
            self._configurations = configurations
            self._target = target.reshape(self.input_dim)
        return self._configurations, self._target

    def _random_cores(self,
                      rank: int,
                      dtype: torch.dtype,
                      device: torch.device,
                      generator: Optional[torch.Generator]
                      ) -> Tuple[torch.Tensor, ...]:
        """Initializes random TT cores at feasible ranks."""
        ranks = _feasible_tt_rank(self.input_dim, rank)
        boundary_ranks = (1, *ranks, 1)
        cores = []
        for site, site_input_dim in enumerate(self.input_dim):
            core = torch.randn(
                boundary_ranks[site],
                site_input_dim,
                boundary_ranks[site + 1],
                device=device,
                dtype=dtype,
                generator=generator)
            core = core / max(1, boundary_ranks[site] * site_input_dim) ** 0.5
            cores.append(core)
        return tuple(cores)

    def _initial_cores(self,
                       target: torch.Tensor,
                       initial_cores,
                       rank: Optional[int],
                       init: str,
                       fixed_cores,
                       generator: Optional[torch.Generator]
                       ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[int, ...]]:
        """Builds and validates initialization plus fixed-site semantics."""
        if init not in ('random', 'svd'):
            raise ValueError("`init` should be 'random' or 'svd'")
        if rank is not None:
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TypeError('`rank` should be int type or None')
            if rank < 1:
                raise ValueError('`rank` should be positive')
        if initial_cores is None:
            if rank is None:
                raise ValueError(
                    '`rank` is required when `initial_cores` is not provided')
            if init == 'random':
                cores = self._random_cores(
                    rank, target.dtype, target.device, generator)
            else:
                decomposition = TTSVD(
                    target,
                    output_device=None).fit(
                        rank=rank,
                        collect_metrics=False)
                cores = _standard_tt_cores(
                    decomposition, self.input_dim)
        else:
            cores = _standard_tt_cores(initial_cores, self.input_dim)

        if any((core.device != target.device) or (core.dtype != target.dtype)
               for core in cores):
            raise ValueError(
                'Initial cores and source values should share dtype and device')
        feasible_ranks = _feasible_tt_rank(
            self.input_dim,
            rank if rank is not None else max(
                (core.shape[-1] for core in cores[:-1]), default=1))
        current_ranks = tuple(core.shape[-1] for core in cores[:-1])
        if rank is not None and any(current_rank > allowed
                                    for current_rank, allowed in zip(
                                        current_ranks, feasible_ranks)):
            raise ValueError(
                'Initial TT ranks should not exceed the requested feasible '
                '`rank` cap')
        algebraic_caps = _feasible_tt_rank(
            self.input_dim,
            max(current_ranks, default=1))
        if any(current_rank > allowed
               for current_rank, allowed in zip(current_ranks,
                                                 algebraic_caps)):
            raise ValueError('Initial TT ranks are not algebraically feasible')

        if fixed_cores is None:
            return tuple(core.clone() for core in cores), ()
        if isinstance(fixed_cores, torch.Tensor):
            raise TypeError(
                '`fixed_cores` should contain one tensor or None per site')
        try:
            fixed_cores = tuple(fixed_cores)
        except TypeError as exc:
            raise TypeError(
                '`fixed_cores` should contain one tensor or None per site') \
                from exc
        if len(fixed_cores) != len(self.input_dim):
            raise ValueError('`fixed_cores` should contain one entry per site')

        final_cores = list(cores)
        fixed_sites = []
        for site, fixed_core in enumerate(fixed_cores):
            if fixed_core is None:
                continue
            if not isinstance(fixed_core, torch.Tensor):
                raise TypeError(
                    '`fixed_cores` should contain torch.Tensor objects or None')
            if len(self.input_dim) == 1:
                normalized = _standard_tt_cores(
                    (fixed_core,), self.input_dim)[0]
            else:
                core = fixed_core
                if site == 0 and core.ndim == 2:
                    core = core.unsqueeze(0)
                elif site == (len(self.input_dim) - 1) and core.ndim == 2:
                    core = core.unsqueeze(-1)
                if core.ndim != 3:
                    raise ValueError(
                        'Every fixed core should use its standard or '
                        'lightweight TT shape')
                normalized = core
            if normalized.shape != cores[site].shape:
                raise ValueError(
                    'Every fixed core should match its initialized core shape')
            if (normalized.device != target.device) or \
                    (normalized.dtype != target.dtype):
                raise ValueError(
                    'Fixed cores and source values should share runtime')
            final_cores[site] = normalized
            fixed_sites.append(site)
        final_cores = TTEnvironmentCache._validate_cores(final_cores)
        return final_cores, tuple(fixed_sites)

    def fit(self,
            rank: Optional[int] = None,
            initial_cores=None,
            init: str = 'random',
            fixed_cores=None,
            gauge: Union[str, GaugePolicy] = 'qr',
            solver: Optional[LeastSquaresSolver] = None,
            convergence: Optional[ConvergencePolicy] = None,
            update_policy: Optional[UpdatePolicy] = None,
            renormalize: bool = True,
            generator: Optional[torch.Generator] = None,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0,
            observer: Optional[DecompositionObserver] = None
            ) -> TTDecomposition:
        """Fits a TT by alternating exact one-site least-squares solves.

        ``initial_cores`` may be a lightweight :class:`TTDecomposition` or a
        sequence using standard or boundary-squeezed TT shapes. Without it,
        ``rank`` is required and acts as one shared upper bound: every cut is
        initialized with
        ``min(rank, prod(input_dim[:k]), prod(input_dim[k:]))``. Existing cores
        that exceed this cap are rejected rather than silently truncated.

        Parameters
        ----------
        rank : int, optional
            Shared maximum TT rank. Required without ``initial_cores``.
        initial_cores : sequence of torch.Tensor or TTDecomposition, optional
            Initial TT approximation. Its dtype and device must match the
            source values.
        init : {``"random"``, ``"svd"``}
            Initialization used when cores are not supplied.
        fixed_cores : sequence of torch.Tensor or None, optional
            One entry per site. Tensor entries replace the corresponding
            initial core and remain bitwise unchanged throughout ALS.
        gauge : {``"none"``, ``"qr"``, ``"svd"``} or GaugePolicy
            Factorization moved to the next trainable site after a local
            solve. If the immediate receiver is fixed or absent, ``NoGauge``
            is used before factorization so no factor is ever discarded.
        solver : LeastSquaresSolver, optional
            Stable local solver. The default uses its standard scaling and no
            regularization.
        convergence : ConvergencePolicy, optional
            Complete-sweep stopping criteria. The default performs ten sweeps.
        update_policy : UpdatePolicy, optional
            Optional damping and local non-increasing acceptance.
        renormalize : bool
            Whether cached environments remove global norms and keep their
            scales logarithmically. This does not change the represented
            local least-squares problem.
        generator : torch.Generator, optional
            Generator used only by random initialization.
        collect_metrics : bool
            If ``True``, records local solves, exact errors and sweep timings.
            With ``False``, no observer and criteria independent of errors,
            the driver skips those reductions and synchronization points.
        verbose : bool or int
            Console verbosity from 0 (silent) to 3 (most detailed).
        observer : DecompositionObserver, optional
            Additional consumer of structured ALS events.

        Returns
        -------
        TTDecomposition
            Lightweight TT result. Its cores can initialize an
            :class:`~tensorkrowch.models.MPS` directly.

        Examples
        --------
        Fit a three-site tensor and create a TensorKrowch model:

        >>> tensor = torch.randn(2, 3, 2)
        >>> result = TTALS(tensor).fit(rank=2)
        >>> [tuple(core.shape) for core in result.cores]
        [(2, 2), (2, 3, 2), (2, 2)]
        >>> model = tk.models.MPS(tensors=result.cores)
        """
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        if (generator is not None) and \
                (not isinstance(generator, torch.Generator)):
            raise TypeError('`generator` should be torch.Generator or None')
        if solver is None:
            solver = LeastSquaresSolver()
        elif not isinstance(solver, LeastSquaresSolver):
            raise TypeError('`solver` should be LeastSquaresSolver type')
        if convergence is None:
            convergence = ConvergencePolicy()
        elif not isinstance(convergence, ConvergencePolicy):
            raise TypeError(
                '`convergence` should be ConvergencePolicy type')
        if update_policy is None:
            update_policy = UpdatePolicy()
        elif not isinstance(update_policy, UpdatePolicy):
            raise TypeError('`update_policy` should be UpdatePolicy type')
        gauge_policy = resolve_gauge_policy(gauge)
        verbosity = _normalize_verbosity(verbose)
        emit_events = bool(verbosity) or (observer is not None)
        collect_metrics = collect_metrics or emit_events
        fit_observer = _resolve_observer(verbosity, observer) \
            if emit_events else None

        configurations, target = self._exact_target()
        if generator is not None and generator.device != target.device:
            raise ValueError(
                '`generator` and source values should use the same device')
        cores, fixed_sites = self._initial_cores(
            target=target,
            initial_cores=initial_cores,
            rank=rank,
            init=init,
            fixed_cores=fixed_cores,
            generator=generator)
        problem = ALSProblem(source=self.source)
        backend = _TTALSBackend(
            problem=problem,
            cores=cores,
            target=target,
            solver=solver,
            gauge=gauge_policy,
            fixed_sites=fixed_sites,
            renormalize=renormalize)
        driver_result = ALSSweepDriver().fit(
            problem=problem,
            backend=backend,
            convergence=convergence,
            update_policy=update_policy,
            observer=fit_observer,
            collect_metrics=collect_metrics)

        result_cores = _result_tt_cores(driver_result.cores)
        if self.output_device is not None:
            result_cores = tuple(
                core.to(device=self.output_device) for core in result_cores)
        return TTDecomposition(
            cores=result_cores,
            metrics=driver_result.metrics,
            metadata={
                'algorithm': 'tt_als',
                'initialization': init if initial_cores is None else 'cores',
                'gauge': getattr(gauge_policy, 'name',
                                 type(gauge_policy).__name__),
                'converged': driver_result.converged,
                'stop_reason': driver_result.stop_reason,
                'n_sweeps': driver_result.n_sweeps,
                'fixed_sites': list(fixed_sites),
                'exact_configurations': configurations.batch_size,
            })


def tt_als(source,
           rank: Optional[int] = None,
           input_dim: Optional[Sequence[int]] = None,
           initial_cores=None,
           init: str = 'random',
           fixed_cores=None,
           gauge: Union[str, GaugePolicy] = 'qr',
           max_sweeps: int = 10,
           error_atol: Optional[float] = None,
           error_rtol: Optional[float] = None,
           change_rtol: Optional[float] = None,
           patience: Optional[int] = None,
           keep_best: bool = False,
           l2_reg: float = 0.0,
           l2_reg_mode: str = 'absolute',
           rcond: Optional[float] = None,
           column_scaling='auto',
           system_scaling: bool = True,
           damping: float = 1.0,
           acceptance: str = 'always',
           renormalize: bool = True,
           dtype: Optional[torch.dtype] = None,
           device: Union[str, torch.device] = 'cpu',
           batch_size: Optional[int] = None,
           output_device: Optional[Union[str, torch.device]] = 'cpu',
           generator: Optional[torch.Generator] = None,
           verbose: Union[bool, int] = 0,
           return_info: bool = False):
    """Approximates a scalar tensor source with exact TT-ALS.

    This is the simple functional interface. ``source`` may be a dense tensor,
    callable, :class:`TensorSource` or existing :class:`TTDecomposition`.
    Callables require ``input_dim`` and receive integer configurations with
    shape ``(batch, sites)``. Use :class:`TTALS` for repeated fits of the same
    source or to pass advanced policy objects directly.

    Parameters
    ----------
    source : TensorSource, TTDecomposition, torch.Tensor or callable
        Scalar discrete tensor or function to approximate.
    rank : int, optional
        Shared maximum TT rank. Required without ``initial_cores``.
    input_dim : sequence of int, optional
        Input dimension at every site; required for callables.
    initial_cores : sequence of torch.Tensor or TTDecomposition, optional
        Initial TT approximation.
    init : {``"random"``, ``"svd"``}
        Initialization used when cores are not supplied.
    fixed_cores : sequence of torch.Tensor or None, optional
        Tensor entries remain fixed throughout all sweeps.
    gauge : {``"none"``, ``"qr"``, ``"svd"``} or GaugePolicy
        Gauge applied after each local solve when its receiver is trainable.
    max_sweeps : int
        Maximum number of complete alternating sweeps.
    error_atol, error_rtol, change_rtol : float, optional
        Absolute error, relative error and complete-sweep stability criteria.
    patience : int, optional
        Consecutive stable sweeps required by ``change_rtol``.
    keep_best : bool
        Whether to restore the best complete-sweep state.
    l2_reg : float
        Local Tikhonov regularization coefficient.
    l2_reg_mode : {``"absolute"``, ``"relative"``}
        Interpretation of ``l2_reg`` before numerical scaling.
    rcond : float, optional
        Cutoff forwarded to :func:`torch.linalg.lstsq`.
    column_scaling : bool or ``"auto"``
        Local least-squares column balancing policy.
    system_scaling : bool
        Whether to scale each complete local system globally.
    damping : float
        Fraction of each local proposal committed, in ``(0, 1]``.
    acceptance : {``"always"``, ``"non_increasing"``}
        Optional local acceptance rule.
    renormalize : bool
        Whether environments accumulate removed norms logarithmically.
    dtype : torch.dtype, optional
        Declared callable output dtype.
    device : str or torch.device
        Callable evaluation device.
    batch_size : int, optional
        Callable evaluation batch size.
    output_device : str or torch.device, optional
        Device where finalized cores are stored. The default is ``"cpu"``.
    generator : torch.Generator, optional
        Random-initialization generator.
    verbose : bool or int
        Console verbosity level from 0 to 3.
    return_info : bool
        If ``True``, also returns metadata and structured ALS metrics.

    Returns
    -------
    list[torch.Tensor] or tuple
        TT cores by default. With ``return_info=True``, returns
        ``(cores, info)``.

    Examples
    --------
    >>> tensor = torch.randn(2, 2, 2)
    >>> cores = tt_als(tensor, rank=2, max_sweeps=2)
    >>> [tuple(core.shape) for core in cores]
    [(2, 2), (2, 2, 2), (2, 2)]
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    solver = LeastSquaresSolver(
        l2_reg=l2_reg,
        l2_reg_mode=l2_reg_mode,
        rcond=rcond,
        column_scaling=column_scaling,
        system_scaling=system_scaling)
    convergence = ConvergencePolicy(
        max_sweeps=max_sweeps,
        error_atol=error_atol,
        error_rtol=error_rtol,
        change_rtol=change_rtol,
        patience=patience,
        keep_best=keep_best)
    update_policy = UpdatePolicy(
        damping=damping,
        acceptance=acceptance)
    result = TTALS(
        source=source,
        input_dim=input_dim,
        dtype=dtype,
        device=device,
        batch_size=batch_size,
        output_device=output_device).fit(
            rank=rank,
            initial_cores=initial_cores,
            init=init,
            fixed_cores=fixed_cores,
            gauge=gauge,
            solver=solver,
            convergence=convergence,
            update_policy=update_policy,
            renormalize=renormalize,
            generator=generator,
            collect_metrics=return_info,
            verbose=verbose)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


__all__ = ['TTALS', 'tt_als']
