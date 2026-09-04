"""Exact, sampled and completion tensor ring ALS decompositions."""

from dataclasses import replace
from math import prod
from typing import Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.decompositions.als.convergence import (ConvergencePolicy,
                                                         UpdatePolicy)
from tensorkrowch.decompositions.als.driver import ALSSweepDriver
from tensorkrowch.decompositions.als.environments import (
    CoreUpdateSet,
    DirectTREnvironment,
    TRSegmentEnvironmentCache,
    _environment_norm,
)
from tensorkrowch.decompositions.als.gauges import (GaugePolicy, NoGauge,
                                                    resolve_gauge_policy)
from tensorkrowch.decompositions.als.problem import ALSProblem
from tensorkrowch.decompositions.als.sampling import (
    ObservedRows,
    RowSampler,
    SampleRefreshPolicy,
    TRExactLeverageRows,
    TRProductLeverageRows,
    UniformRows,
    _RowSamplingState,
)
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.als.tt import (TTALS, _relative_error,
                                                _solve_local_proposal)
from tensorkrowch.decompositions.observers import (DecompositionObserver,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import TRDecomposition
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                  FiberTensorSource)
from tensorkrowch.decompositions.sources.base import _unravel_indices
from tensorkrowch.decompositions.svd.tr import TRSVD


_Rank = Optional[Union[int, Sequence[int]]]


def _standard_tr_cores(
        cores: Union[TRDecomposition, Sequence[torch.Tensor]],
        input_dim: Sequence[int]) -> Tuple[torch.Tensor, ...]:
    """Normalizes lightweight TR cores to standard cyclic shapes."""
    if isinstance(cores, TRDecomposition):
        if cores.n_batches:
            raise ValueError('Batched TR cores are not supported by TR-ALS')
        cores = cores.cores
    elif isinstance(cores, torch.Tensor):
        raise TypeError(
            '`initial_cores` should be a TRDecomposition or a core sequence')
    try:
        cores = tuple(cores)
    except TypeError as exc:
        raise TypeError(
            '`initial_cores` should be a TRDecomposition or a core sequence') \
            from exc
    if len(cores) != len(input_dim):
        raise ValueError('`initial_cores` should contain one core per site')
    if not all(isinstance(core, torch.Tensor) for core in cores):
        raise TypeError('`initial_cores` should contain torch.Tensor objects')
    for core, site_input_dim in zip(cores, input_dim):
        if core.ndim != 3:
            raise ValueError(
                'TR cores should have left rank, input and right rank dimensions')
        if core.shape[1] != site_input_dim:
            raise ValueError(
                '`initial_cores` input dimensions should match the source')
    return TRSegmentEnvironmentCache._validate_cores(cores)


def _normalize_tr_rank(rank: _Rank,
                       n_sites: int) -> Optional[Tuple[int, ...]]:
    """Normalizes a shared TR rank or one right-link rank per site."""
    if rank is None:
        return None
    if isinstance(rank, bool):
        raise TypeError('`rank` should be int, a sequence of ints or None')
    if isinstance(rank, int):
        ranks = (rank,) * n_sites
    else:
        if isinstance(rank, (str, bytes)):
            raise TypeError('`rank` should be int, a sequence of ints or None')
        try:
            ranks = tuple(rank)
        except TypeError as exc:
            raise TypeError(
                '`rank` should be int, a sequence of ints or None') from exc
        if len(ranks) != n_sites:
            raise ValueError(
                'A TR `rank` sequence should contain one right-link rank per '
                'site')
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in ranks):
        raise TypeError('TR ranks should be integers')
    if any(value < 1 for value in ranks):
        raise ValueError('TR ranks should be positive')
    return ranks


def _contract_standard_tr(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Contracts standard cyclic cores without constructing a model graph."""
    result = cores[0]
    for core in cores[1:]:
        result = torch.einsum('a...b,bpc->a...pc', result, core)
    return result.diagonal(dim1=0, dim2=-1).sum(-1)


def _evaluate_standard_tr(cores: Sequence[torch.Tensor],
                          indices: torch.Tensor) -> torch.Tensor:
    """Evaluates standard cyclic cores at packed configurations."""
    selected = cores[0][:, indices[:, 0], :].permute(1, 0, 2)
    environment = selected
    for site, core in enumerate(cores[1:], 1):
        selected = core[:, indices[:, site], :].permute(1, 0, 2)
        environment = torch.bmm(environment, selected)
    return environment.diagonal(dim1=-2, dim2=-1).sum(-1)


def _tr_gauge_core_update(
        cores: Sequence[torch.Tensor],
        versions: Sequence[int],
        fixed_sites: Sequence[int],
        gauge: GaugePolicy,
        direction: str,
        site: int,
        proposal: torch.Tensor) -> CoreUpdateSet:
    """Builds an atomic cyclic current-plus-receiver gauge update."""
    n_sites = len(cores)
    receiver = (site + 1) % n_sites if direction == 'forward' else \
        (site - 1) % n_sites
    legal_receiver = (receiver != site) and (receiver not in fixed_sites)
    if direction == 'forward':
        feasible = proposal.shape[0] * proposal.shape[1] >= proposal.shape[2]
    else:
        feasible = proposal.shape[1] * proposal.shape[2] >= proposal.shape[0]
    effective_gauge = gauge if legal_receiver and feasible else NoGauge()
    gauged_core, factor = effective_gauge.factor(proposal, direction)

    sites = [site]
    updated_cores = [gauged_core]
    updated_versions = [versions[site] + 1]
    if factor is not None:
        neighbor = effective_gauge.absorb(
            factor, cores[receiver], direction)
        sites.append(receiver)
        updated_cores.append(neighbor)
        updated_versions.append(versions[receiver] + 1)
    return CoreUpdateSet(
        sites=sites,
        cores=updated_cores,
        versions=updated_versions,
        reason='local_solve')


def _normalize_tr_core_update(
        update_set: CoreUpdateSet,
        cores: Sequence[torch.Tensor],
        versions: Sequence[int],
        fixed_sites: Sequence[int],
        direction: str,
        site: int) -> CoreUpdateSet:
    """Moves a square-root core norm to the next trainable cyclic site."""
    updates = dict(update_set.updates)
    current = updates[site][0]
    norm = torch.linalg.vector_norm(current)
    if (not torch.isfinite(norm)) or (norm == 0):
        return update_set
    scale = norm.sqrt()

    receiver = None
    step = 1 if direction == 'forward' else -1
    for offset in range(1, len(cores)):
        candidate = (site + step * offset) % len(cores)
        if candidate not in fixed_sites:
            receiver = candidate
            break
    if receiver is None:
        return update_set

    normalized = current / scale.to(current.dtype)
    receiver_core, receiver_version = updates.get(
        receiver, (cores[receiver], versions[receiver] + 1))
    receiver_core = receiver_core * scale.to(receiver_core.dtype)
    updates[site] = (normalized, updates[site][1])
    updates[receiver] = (receiver_core, receiver_version)
    sites = tuple(updates)
    return CoreUpdateSet(
        sites=sites,
        cores=tuple(updates[updated_site][0] for updated_site in sites),
        versions=tuple(updates[updated_site][1] for updated_site in sites),
        reason='local_solve_normalization')


class _TRALSBackend:
    """Adapts exact, sampled and completion TR solves to the common driver."""

    def __init__(self,
                 problem: ALSProblem,
                 cores: Sequence[torch.Tensor],
                 target: Optional[torch.Tensor],
                 solver: LeastSquaresSolver,
                 gauge: GaugePolicy,
                 fixed_sites: Sequence[int],
                 normalize: bool,
                 renormalize: bool,
                 n_segments: Optional[int],
                 sampler: Optional[RowSampler] = None,
                 n_samples: Optional[int] = None,
                 refresh_policy: Optional[SampleRefreshPolicy] = None,
                 generator: Optional[torch.Generator] = None) -> None:
        self.problem = problem
        self.full_target = None if target is None else target.reshape(-1)
        self.solver = solver
        self.gauge = gauge
        self.fixed_sites = frozenset(fixed_sites)
        self.normalize = normalize
        self.cache = TRSegmentEnvironmentCache(
            cores, n_segments=n_segments, renormalize=renormalize)
        self.sampler = sampler
        self.n_samples = n_samples
        self.refresh_policy = refresh_policy
        self.generator = generator
        self.sample_batch = None
        self.current_target = self.full_target
        self._sampling_state = _RowSamplingState(
            n_rows=prod(self.cache.input_dim),
            core_versions=self.cache.core_versions,
            device=self.cache.cores[0].device)
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
        self._direction = 'forward' if order[0] == 0 else 'reverse'
        if self.sampler is None:
            self.cache.prepare_sweep(order)
            self.current_target = self.full_target
            return None, False

        self._sampling_state = replace(
            self._sampling_state,
            core_versions=self.cache.core_versions)
        batch, state, refreshed = self.refresh_policy.sample(
            sampler=self.sampler,
            state=self._sampling_state,
            site=order[0],
            n_samples=self.n_samples,
            generator=self.generator,
            sweep=sweep,
            current=self.sample_batch,
            invalidate=lambda old, new: self.cache.invalidate(
                'sample generation changed'))
        self._sampling_state = state
        if refreshed:
            if self.problem.observations is not None:
                observations = self.problem.observations
                positions = torch.searchsorted(
                    observations.flat_ids.to(batch.ids.device), batch.ids)
                if torch.any(positions >= observations.flat_ids.numel()) or \
                        not torch.equal(
                            observations.flat_ids.to(batch.ids.device)
                            .index_select(0, positions), batch.ids):
                    raise ValueError(
                        'Observed sample ids should match fixed observations')
                self.current_target = observations.values.index_select(
                    0, positions.to(observations.values.device))
            else:
                indices = _unravel_indices(batch.ids, self.cache.input_dim)
                evaluated = self.problem.evaluate(
                    ConfigurationBatch(indices, kind='indices'))
                if evaluated.shape != (batch.ids.numel(),):
                    raise ValueError(
                        'TR-ALS currently requires a scalar tensor source')
                if not torch.isfinite(evaluated).all():
                    raise ValueError(
                        'The tensor source should return only finite values')
                self.current_target = evaluated
        self.sample_batch = batch
        self.cache.prepare_sweep(order, samples=batch)
        return batch.generation, refreshed

    def solve_site(self,
                   site: int,
                   sweep: int,
                   update_policy: UpdatePolicy,
                   return_record: bool):
        local = self.cache.local_environment(site)
        environment = local.design()
        target = local.scale_target(self.current_target)
        if self.sample_batch is not None:
            sample_weights = self.sample_batch.weights.to(environment.dtype)
            environment = environment * sample_weights.unsqueeze(1)
            target = target * sample_weights
        if self.problem.observations is not None and \
                (self.problem.observations.weights is not None):
            observation_weights = self.problem.observations.weights
            positions = torch.searchsorted(
                self.problem.observations.flat_ids.to(
                    self.sample_batch.ids.device),
                self.sample_batch.ids)
            observation_weights = observation_weights.index_select(
                0, positions.to(observation_weights.device))
            observation_weights = observation_weights.to(environment.dtype)
            environment = environment * observation_weights.unsqueeze(1)
            target = target * observation_weights
        if self.problem.weights is not None:
            weights = self.problem.weights.reshape(-1).to(environment.dtype)
            environment = environment * weights.unsqueeze(1)
            target = target * weights

        regularization_scale = None
        if (self.solver.l2_reg_mode == 'absolute') and \
                (self.solver.l2_reg > 0):
            regularization_scale = (-2 * local.log_scale).exp()
        sampling_exact = None if self.sample_batch is None else \
            self.sample_batch.is_exact_for(self.cache.core_versions)
        proposal, record = _solve_local_proposal(
            solver=self.solver,
            environment=environment,
            target=target,
            current=self.cache.cores[site],
            site=site,
            sweep=sweep,
            update_policy=update_policy,
            return_record=return_record,
            regularization_scale=regularization_scale,
            sampling_exact=sampling_exact,
            sample_generation=(
                None if self.sample_batch is None
                else self.sample_batch.generation))
        update_set = _tr_gauge_core_update(
            cores=self.cache.cores,
            versions=self.cache.core_versions,
            fixed_sites=self.fixed_sites,
            gauge=self.gauge,
            direction=self._direction,
            site=site,
            proposal=proposal)
        if self.normalize:
            update_set = _normalize_tr_core_update(
                update_set=update_set,
                cores=self.cache.cores,
                versions=self.cache.core_versions,
                fixed_sites=self.fixed_sites,
                direction=self._direction,
                site=site)
        return update_set, record

    def skip_site(self, site: int) -> None:
        self.cache.local_environment(site)
        self.cache.commit(CoreUpdateSet((), (), (), reason='fixed_core'))

    def commit(self, update_set: CoreUpdateSet) -> None:
        self.cache.commit(update_set)
        if self.sampler is not None:
            for site in update_set.sites:
                self._sampling_state = self.sampler.update_after_core(
                    self._sampling_state, site)

    def measure_objective(
            self, problem: ALSProblem) -> Tuple[torch.Tensor, torch.Tensor]:
        if problem.observations is not None:
            indices = problem.observations.indices.to(
                self.cache.cores[0].device)
            approximation = _evaluate_standard_tr(self.cache.cores, indices)
            return problem.observations.error(approximation)

        approximation = _contract_standard_tr(self.cache.cores).reshape(-1)
        residual = approximation - self.full_target
        target = self.full_target
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
        self.cache = TRSegmentEnvironmentCache(
            cores,
            n_segments=len(self.cache.segments),
            renormalize=self.cache.renormalize)


class _TRLeverageALSBackend:
    """Runs product or exact TR leverage sampling by active fibers."""

    def __init__(self,
                 problem: ALSProblem,
                 cores: Sequence[torch.Tensor],
                 solver: LeastSquaresSolver,
                 gauge: GaugePolicy,
                 fixed_sites: Sequence[int],
                 n_samples: int,
                 leverage_method: str,
                 uniform_mix: float,
                 generator: Optional[torch.Generator],
                 normalize: bool,
                 renormalize: bool) -> None:
        self.problem = problem
        self._cores = TRSegmentEnvironmentCache._validate_cores(cores)
        self._versions = (0,) * len(self._cores)
        self.solver = solver
        self.gauge = gauge
        self.fixed_sites = frozenset(fixed_sites)
        self.n_samples = n_samples
        self.generator = generator
        self.normalize = normalize
        self.renormalize = renormalize
        if leverage_method == 'product':
            self.sampler = TRProductLeverageRows(
                lambda: self._cores, uniform_mix=uniform_mix)
        elif leverage_method == 'exact':
            self.sampler = TRExactLeverageRows(
                lambda: self._cores, uniform_mix=uniform_mix)
        else:
            raise ValueError(
                "`leverage_method` should be 'product' or 'exact'")
        self.leverage_method = leverage_method
        self._sampling_state = _RowSamplingState(
            n_rows=prod(core.shape[1] for core in self._cores),
            core_versions=self._versions,
            device=self._cores[0].device)
        self._direction = None
        self._generation = None

    @property
    def n_sites(self) -> int:
        return len(self._cores)

    @property
    def trainable_sites(self) -> Sequence[int]:
        return tuple(site for site in range(self.n_sites)
                     if site not in self.fixed_sites)

    @property
    def cores(self) -> Sequence[torch.Tensor]:
        return self._cores

    def prepare_sweep(self,
                      order: Sequence[int],
                      sweep: int) -> Tuple[Optional[int], bool]:
        self._direction = 'forward' if order[0] == 0 else 'reverse'
        self._generation = sweep
        return sweep, True

    def solve_site(self,
                   site: int,
                   sweep: int,
                   update_policy: UpdatePolicy,
                   return_record: bool):
        state = replace(
            self._sampling_state,
            core_versions=self._versions,
            generation=self._generation)
        batch = self.sampler.draw(
            state=state,
            site=site,
            n_samples=self.n_samples,
            generator=self.generator)
        indices = _unravel_indices(
            batch.ids, tuple(core.shape[1] for core in self._cores))
        input_dim = self._cores[site].shape[1]
        if isinstance(self.problem.source, FiberTensorSource):
            base_indices = indices[::input_dim].clone()
            target = self.problem.source.fiber(
                ConfigurationBatch(base_indices, kind='indices'), site)
            target = target.reshape(-1)
        else:
            target = self.problem.evaluate(
                ConfigurationBatch(indices, kind='indices'))
        if target.shape != (batch.ids.numel(),):
            raise ValueError(
                'TR-ALS currently requires a scalar tensor source')
        if not torch.isfinite(target).all():
            raise ValueError(
                'The tensor source should return only finite values')

        local = DirectTREnvironment(self._cores).local_environment(
            site, samples=batch)
        environment = local.design()
        log_scale = environment.real.new_zeros(())
        if self.renormalize:
            norm = _environment_norm(environment)
            if norm > 0:
                environment = environment / norm
                log_scale = norm.log()
                target = target / norm.to(target.dtype)
        sample_weights = batch.weights.to(environment.dtype)
        environment = environment * sample_weights.unsqueeze(1)
        target = target * sample_weights

        regularization_scale = None
        if (self.solver.l2_reg_mode == 'absolute') and \
                (self.solver.l2_reg > 0):
            regularization_scale = (-2 * log_scale).exp()
        sampling_exact = batch.is_exact_for(self._versions)
        proposal, record = _solve_local_proposal(
            solver=self.solver,
            environment=environment,
            target=target,
            current=self._cores[site],
            site=site,
            sweep=sweep,
            update_policy=update_policy,
            return_record=return_record,
            regularization_scale=regularization_scale,
            sampling_exact=sampling_exact,
            sample_generation=batch.generation)
        update_set = _tr_gauge_core_update(
            cores=self._cores,
            versions=self._versions,
            fixed_sites=self.fixed_sites,
            gauge=self.gauge,
            direction=self._direction,
            site=site,
            proposal=proposal)
        if self.normalize:
            update_set = _normalize_tr_core_update(
                update_set=update_set,
                cores=self._cores,
                versions=self._versions,
                fixed_sites=self.fixed_sites,
                direction=self._direction,
                site=site)
        return update_set, record

    def skip_site(self, site: int) -> None:
        return None

    def commit(self, update_set: CoreUpdateSet) -> None:
        cores = list(self._cores)
        versions = list(self._versions)
        for site, (core, version) in update_set.updates.items():
            if version <= versions[site]:
                raise ValueError('Every updated core version should increase')
            cores[site] = core
            versions[site] = version
        self._cores = TRSegmentEnvironmentCache._validate_cores(cores)
        self._versions = tuple(versions)
        self._sampling_state = replace(
            self._sampling_state, core_versions=self._versions)

    def measure_objective(
            self, problem: ALSProblem) -> Tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            'Renewable leverage batches do not define a global objective')

    def snapshot(self) -> Sequence[torch.Tensor]:
        return tuple(core.clone() for core in self._cores)

    def restore(self, cores: Sequence[torch.Tensor]) -> None:
        self._cores = TRSegmentEnvironmentCache._validate_cores(cores)


class TRALS(TTALS):
    """Approximates a fixed scalar tensor problem by cyclic TR-ALS.

    The source and input dimensions are fixed on construction. Repeated
    :meth:`fit` calls may compare cyclic ranks, initialization, gauges,
    segment partitions, sampling and convergence policies. Use
    :meth:`completion` to fix one permanent observed objective.

    Parameters are the same source/runtime parameters as :class:`TTALS`, but
    every result is a :class:`TRDecomposition` with cyclic cores of shape
    ``(left rank, input, right rank)``.

    ``sampling="leverage"`` with the product method implements Algorithm 2 of
    *A Sampling-Based Method for Tensor Ring Decomposition* (2021), available
    in this `paper <https://arxiv.org/abs/2010.08581>`_ by Osman Asif Malik and
    Stephen Becker. It uses their approximate product-leverage proposal and
    evaluates complete active input fibers. The optional uniform mixture and
    the common TensorKrowch solver, gauge and convergence policies are library
    extensions.

    With ``sampling="leverage", leverage_method="exact"``, sampling instead
    specializes Sections 4.1--4.2 and Appendix B.2 of *Sampling-Based
    Decomposition Algorithms for Arbitrary Tensor Networks* (2022), available
    in this `paper <https://arxiv.org/abs/2210.03828>`_ by Osman Asif Malik,
    Vivek Bharadwaj and Riley Murray. This contracts the cyclic double-layer
    Gram and samples exact conditional leverage probabilities without forming
    the exponentially tall design.
    """

    @classmethod
    def completion(cls,
                   observations,
                   values: Optional[torch.Tensor] = None,
                   input_dim: Optional[Sequence[int]] = None,
                   weights: Optional[torch.Tensor] = None,
                   *,
                   output_device: Optional[
                       Union[str, torch.device]] = 'cpu') -> 'TRALS':
        """Creates TR-ALS for a permanently observed completion objective."""
        return super().completion(
            observations=observations,
            values=values,
            input_dim=input_dim,
            weights=weights,
            output_device=output_device)

    def _random_cores(self,
                      ranks: Sequence[int],
                      dtype: torch.dtype,
                      device: torch.device,
                      generator: Optional[torch.Generator]
                      ) -> Tuple[torch.Tensor, ...]:
        """Initializes random cyclic cores with prescribed right-link ranks."""
        cores = []
        for site, site_input_dim in enumerate(self.input_dim):
            core = torch.randn(
                ranks[site - 1],
                site_input_dim,
                ranks[site],
                dtype=dtype,
                device=device,
                generator=generator)
            core = core / max(1, ranks[site - 1] * site_input_dim) ** 0.5
            cores.append(core)
        return tuple(cores)

    def _initial_tr_cores(self,
                          target: torch.Tensor,
                          initial_cores,
                          rank: _Rank,
                          init: str,
                          fixed_cores,
                          generator: Optional[torch.Generator]
                          ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[int, ...]]:
        """Builds and validates initialization plus fixed-site semantics."""
        if init not in ('random', 'svd'):
            raise ValueError("`init` should be 'random' or 'svd'")
        rank_caps = _normalize_tr_rank(rank, len(self.input_dim))
        if initial_cores is None:
            if rank_caps is None:
                raise ValueError(
                    '`rank` is required when `initial_cores` is not provided')
            if init == 'random':
                cores = self._random_cores(
                    rank_caps, target.dtype, target.device, generator)
            else:
                decomposition = TRSVD(
                    target, out_device=None).fit(
                        rank=max(rank_caps),
                        collect_metrics=False)
                cores = _standard_tr_cores(decomposition, self.input_dim)
        else:
            cores = _standard_tr_cores(initial_cores, self.input_dim)

        if any((core.device != target.device) or (core.dtype != target.dtype)
               for core in cores):
            raise ValueError(
                'Initial cores and source values should share dtype and device')
        current_ranks = tuple(core.shape[-1] for core in cores)
        if rank_caps is not None and any(
                current > cap
                for current, cap in zip(current_ranks, rank_caps)):
            raise ValueError(
                'Initial TR ranks should not exceed the requested `rank` caps')

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
            if fixed_core.shape != cores[site].shape:
                raise ValueError(
                    'Every fixed core should match its initialized core shape')
            if (fixed_core.device != target.device) or \
                    (fixed_core.dtype != target.dtype):
                raise ValueError(
                    'Fixed cores and source values should share runtime')
            final_cores[site] = fixed_core
            fixed_sites.append(site)
        final_cores = TRSegmentEnvironmentCache._validate_cores(final_cores)
        return final_cores, tuple(fixed_sites)

    def fit(self,
            rank: _Rank = None,
            initial_cores=None,
            init: str = 'random',
            fixed_cores=None,
            gauge: Union[str, GaugePolicy] = 'qr',
            sampling: Optional[str] = None,
            n_samples: Optional[int] = None,
            sample_reuse_sweeps: int = 1,
            leverage_method: str = 'product',
            leverage_uniform_mix: float = 0.0,
            n_segments: Optional[int] = None,
            solver: Optional[LeastSquaresSolver] = None,
            convergence: Optional[ConvergencePolicy] = None,
            update_policy: Optional[UpdatePolicy] = None,
            normalize: bool = True,
            renormalize: bool = True,
            generator: Optional[torch.Generator] = None,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0,
            observer: Optional[DecompositionObserver] = None
            ) -> TRDecomposition:
        """Fits a TR by alternating cyclic one-site least-squares solves.

        A scalar ``rank`` is used on every link. A sequence contains one right
        rank per core, so ``rank[-1]`` is the cyclic closing link. With supplied
        cores, these values are upper bounds and no core is silently truncated.

        ``sampling`` may be ``"exact"``, ``"uniform"``, ``"leverage"`` or
        ``"observed"``. For leverage, ``leverage_method="product"`` implements
        Algorithm 2 of *A Sampling-Based Method for Tensor Ring Decomposition*
        (2021), available in this `paper <https://arxiv.org/abs/2010.08581>`_
        by Osman Asif Malik and Stephen Becker. ``leverage_method="exact"``
        implements Sections 4.1--4.2 and Appendix B.2 of *Sampling-Based
        Decomposition Algorithms for Arbitrary Tensor Networks* (2022),
        available in this `paper <https://arxiv.org/abs/2210.03828>`_ by Osman
        Asif Malik, Vivek Bharadwaj and Riley Murray. Exact and observed
        objectives record comparable complete-sweep errors; renewable sampled
        batches deliberately do not.

        Parameters
        ----------
        rank : int or sequence of int, optional
            Shared rank or one upper bound for every right link.
        initial_cores : sequence of torch.Tensor or TRDecomposition, optional
            Initial cyclic approximation in standard core shapes.
        init : {``"random"``, ``"svd"``}
            Initialization used when cores are not supplied. SVD requires an
            exact known source.
        fixed_cores : sequence of torch.Tensor or None, optional
            Tensor entries remain bitwise unchanged throughout all sweeps.
        gauge : {``"none"``, ``"qr"``, ``"svd"``} or GaugePolicy
            Gauge moved to the immediate cyclic receiver only when that core
            is trainable and the prescribed ranks make the factorization legal.
        sampling : {``"exact"``, ``"uniform"``, ``"leverage"``,
            ``"observed"``}, optional
            Row strategy. Completion always uses its permanent observations.
        n_samples : int, optional
            Rows per uniform generation. With leverage, number of sampled
            environments; every active input fiber is retained.
        sample_reuse_sweeps : int
            Sweeps reusing sampled ids, probabilities and source values.
            TR leverage redraws after every design change and requires 1.
        leverage_method : {``"product"``, ``"exact"``}
            Product mode uses independent core-unfolding leverage bounds. Exact
            mode contracts the cyclic Gram and samples conditional leverage.
        leverage_uniform_mix : float
            Uniform component mixed with the selected leverage proposal, in
            ``[0, 1]``. Only zero is the pure distribution analyzed in the
            corresponding paper.
        n_segments : int, optional
            Number of balanced environment-cache segments. Defaults to at most
            three and already defines future worker partitions.
        solver : LeastSquaresSolver, optional
            Stable local least-squares solver.
        convergence : ConvergencePolicy, optional
            Complete-sweep stopping criteria.
        update_policy : UpdatePolicy, optional
            Optional damping and local non-increasing acceptance.
        normalize : bool
            Whether to extract the square root of each solved core norm and
            absorb it into the next trainable cyclic site. Fixed cores are
            skipped and never rescaled.
        renormalize : bool
            Whether environments remove global norms and retain log-scales.
        generator : torch.Generator, optional
            Generator used by initialization and randomized sampling.
        collect_metrics : bool
            Whether to retain local solves, sweep errors and timings.
        verbose : bool or int
            Console verbosity from 0 (silent) to 3 (most detailed).
        observer : DecompositionObserver, optional
            Additional consumer of structured ALS events.

        Returns
        -------
        TRDecomposition
            Lightweight cyclic result whose cores can initialize an
            :class:`~tensorkrowch.models.MPS` directly.

        Examples
        --------
        >>> tensor = torch.randn(2, 3, 2)
        >>> result = TRALS(tensor).fit(
        ...     rank=(2, 2, 2),
        ...     convergence=ConvergencePolicy(max_sweeps=2))
        >>> [tuple(core.shape) for core in result.cores]
        [(2, 2, 2), (2, 3, 2), (2, 2, 2)]
        >>> model = tk.models.MPS(tensors=result.cores)
        >>> model.boundary
        'pbc'
        """
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        if not isinstance(normalize, bool):
            raise TypeError('`normalize` should be bool type')
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
            raise TypeError('`convergence` should be ConvergencePolicy type')
        if update_policy is None:
            update_policy = UpdatePolicy()
        elif not isinstance(update_policy, UpdatePolicy):
            raise TypeError('`update_policy` should be UpdatePolicy type')
        gauge_policy = resolve_gauge_policy(gauge)
        if sampling is None:
            sampling = 'observed' if self.problem.observations is not None \
                else 'exact'
        if sampling not in ('exact', 'uniform', 'leverage', 'observed'):
            raise ValueError(
                "`sampling` should be 'exact', 'uniform', 'leverage' or "
                "'observed'")
        if self.problem.observations is not None:
            if sampling != 'observed':
                raise ValueError(
                    'Completion should use its permanently observed rows')
        elif sampling == 'observed':
            raise ValueError(
                '`sampling="observed"` requires TRALS.completion')
        if sampling in ('uniform', 'leverage'):
            if isinstance(n_samples, bool) or \
                    (not isinstance(n_samples, int)) or (n_samples < 1):
                raise ValueError(
                    '`n_samples` should be positive for sampled ALS')
        elif n_samples is not None:
            raise ValueError(
                '`n_samples` is only used with sampled ALS')
        refresh_policy = SampleRefreshPolicy(
            reuse_sweeps=sample_reuse_sweeps)
        if leverage_method not in ('product', 'exact'):
            raise ValueError(
                "`leverage_method` should be 'product' or 'exact'")
        if isinstance(leverage_uniform_mix, bool) or \
                (not isinstance(leverage_uniform_mix, (int, float))) or \
                (leverage_uniform_mix < 0) or (leverage_uniform_mix > 1):
            raise ValueError(
                '`leverage_uniform_mix` should be in [0, 1]')
        if sampling == 'leverage' and sample_reuse_sweeps != 1:
            raise ValueError(
                'TR leverage sampling redraws per site and requires '
                '`sample_reuse_sweeps=1`')
        verbosity = _normalize_verbosity(verbose)
        emit_events = bool(verbosity) or (observer is not None)
        collect_metrics = collect_metrics or emit_events
        fit_observer = _resolve_observer(verbosity, observer) \
            if emit_events else None

        configurations = None
        target = None
        sampler = None
        problem = self.problem
        if sampling == 'exact':
            configurations, target = self._exact_target()
            runtime_reference = target
        elif sampling in ('uniform', 'leverage'):
            sampler = UniformRows()
            problem = ALSProblem(source=self.source, selector=sampler)
            runtime_reference = self._runtime_reference()
        else:
            sampler = ObservedRows(self.problem.observations)
            runtime_reference = self.problem.observations.values
        if not (runtime_reference.is_floating_point() or
                runtime_reference.is_complex()):
            raise TypeError(
                'The tensor source should return floating or complex values')
        if (sampling != 'exact') and (init == 'svd') and \
                (initial_cores is None):
            raise ValueError(
                '`init="svd"` requires an exact known tensor source')
        if generator is not None and generator.device != \
                runtime_reference.device:
            raise ValueError(
                '`generator` and source values should use the same device')
        cores, fixed_sites = self._initial_tr_cores(
            target=runtime_reference,
            initial_cores=initial_cores,
            rank=rank,
            init=init,
            fixed_cores=fixed_cores,
            generator=generator)
        if sampling == 'leverage':
            backend = _TRLeverageALSBackend(
                problem=problem,
                cores=cores,
                solver=solver,
                gauge=gauge_policy,
                fixed_sites=fixed_sites,
                n_samples=n_samples,
                leverage_method=leverage_method,
                uniform_mix=leverage_uniform_mix,
                generator=generator,
                normalize=normalize,
                renormalize=renormalize)
        else:
            backend = _TRALSBackend(
                problem=problem,
                cores=cores,
                target=target,
                solver=solver,
                gauge=gauge_policy,
                fixed_sites=fixed_sites,
                normalize=normalize,
                renormalize=renormalize,
                n_segments=n_segments,
                sampler=sampler,
                n_samples=n_samples,
                refresh_policy=refresh_policy,
                generator=generator)
        driver_result = ALSSweepDriver().fit(
            problem=problem,
            backend=backend,
            convergence=convergence,
            update_policy=update_policy,
            observer=fit_observer,
            collect_metrics=collect_metrics)

        result_cores = tuple(driver_result.cores)
        if self.output_device is not None:
            result_cores = tuple(
                core.to(device=self.output_device) for core in result_cores)
        return TRDecomposition(
            cores=result_cores,
            metrics=driver_result.metrics,
            metadata={
                'algorithm': 'tr_als',
                'initialization': init if initial_cores is None else 'cores',
                'gauge': getattr(gauge_policy, 'name',
                                 type(gauge_policy).__name__),
                'converged': driver_result.converged,
                'stop_reason': driver_result.stop_reason,
                'n_sweeps': driver_result.n_sweeps,
                'fixed_sites': list(fixed_sites),
                'sampling': sampling,
                'n_samples': n_samples,
                'sample_reuse_sweeps': sample_reuse_sweeps,
                'n_segments': (
                    None if sampling == 'leverage'
                    else len(backend.cache.segments)),
                'normalize': normalize,
                'sampling_exact': (
                    backend.sampler.proposal_exact
                    if sampling == 'leverage' else
                    (None if backend.sample_batch is None else
                     backend.sample_batch.is_exact_for(
                         backend.cache.core_versions))),
                'leverage_method': (
                    leverage_method if sampling == 'leverage' else None),
                'leverage_uniform_mix': (
                    leverage_uniform_mix
                    if sampling == 'leverage' else None),
                'exact_configurations': (
                    None if configurations is None else
                    configurations.batch_size),
            })


def tr_als(source,
           rank: _Rank = None,
           input_dim: Optional[Sequence[int]] = None,
           initial_cores=None,
           init: str = 'random',
           fixed_cores=None,
           gauge: Union[str, GaugePolicy] = 'qr',
           sampling: str = 'exact',
           n_samples: Optional[int] = None,
           sample_reuse_sweeps: int = 1,
           leverage_method: str = 'product',
           leverage_uniform_mix: float = 0.0,
           n_segments: Optional[int] = None,
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
           normalize: bool = True,
           renormalize: bool = True,
           dtype: Optional[torch.dtype] = None,
           device: Union[str, torch.device] = 'cpu',
           batch_size: Optional[int] = None,
           output_device: Optional[Union[str, torch.device]] = 'cpu',
           generator: Optional[torch.Generator] = None,
           verbose: Union[bool, int] = 0,
           return_info: bool = False):
    """Approximates a scalar tensor source with cyclic TR-ALS.

    This functional interface returns a core list. Use :class:`TRALS` for
    repeated fits, completion or advanced policy objects. ``rank`` is either a
    shared value or one right-link value per site; the last value is the cyclic
    closing rank. With ``sampling="leverage"``, the product method implements
    Algorithm 2 of *A Sampling-Based Method for Tensor Ring Decomposition*
    (2021), available in this `paper <https://arxiv.org/abs/2010.08581>`_ by
    Osman Asif Malik and Stephen Becker. The exact method implements Sections
    4.1--4.2 and Appendix B.2 of *Sampling-Based Decomposition Algorithms for
    Arbitrary Tensor Networks* (2022), available in this
    `paper <https://arxiv.org/abs/2210.03828>`_ by Osman Asif Malik, Vivek
    Bharadwaj and Riley Murray.
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
    result = TRALS(
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
            sampling=sampling,
            n_samples=n_samples,
            sample_reuse_sweeps=sample_reuse_sweeps,
            leverage_method=leverage_method,
            leverage_uniform_mix=leverage_uniform_mix,
            n_segments=n_segments,
            solver=solver,
            convergence=convergence,
            update_policy=update_policy,
            normalize=normalize,
            renormalize=renormalize,
            generator=generator,
            collect_metrics=return_info,
            verbose=verbose)
    if return_info:
        return result.cores, result.as_info()
    return result.cores


__all__ = ['TRALS', 'tr_als']
