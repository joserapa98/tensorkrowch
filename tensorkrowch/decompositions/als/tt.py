"""Exact tensor-train alternating least-squares decompositions."""

from dataclasses import replace
from math import prod
from typing import (Optional, Sequence, Tuple, Union)

import torch

from tensorkrowch.decompositions.als.convergence import (ConvergencePolicy,
                                                         UpdatePolicy)
from tensorkrowch.decompositions.als.driver import ALSSweepDriver
from tensorkrowch.decompositions.als.environments import (CoreUpdateSet,
                                                          TTEnvironmentCache,
                                                          _environment_norm)
from tensorkrowch.decompositions.als.gauges import (GaugePolicy, NoGauge,
                                                    QRGauge, SVDGauge,
                                                    resolve_gauge_policy)
from tensorkrowch.decompositions.als.problem import (ALSProblem,
                                                     ObservedEntries)
from tensorkrowch.decompositions.als.sampling import (
    ObservedRows,
    RowSampler,
    SampleBatch,
    SampleRefreshPolicy,
    TTLeverageRows,
    UniformRows,
    _RowSamplingState,
)
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


def _evaluate_standard_tt(cores: Sequence[torch.Tensor],
                          indices: torch.Tensor) -> torch.Tensor:
    """Evaluates standard OBC cores at packed discrete configurations."""
    environment = cores[0][:, indices[:, 0], :].squeeze(0)
    for site, core in enumerate(cores[1:], 1):
        selected = core[:, indices[:, site], :].permute(1, 0, 2)
        environment = torch.einsum('ja,jab->jb', environment, selected)
    return environment.squeeze(-1)


def _relative_error(absolute: torch.Tensor,
                    target_norm: torch.Tensor) -> torch.Tensor:
    """Applies the decomposition-wide zero-target relative-error policy."""
    if target_norm > 0:
        return absolute / target_norm
    if absolute == 0:
        return torch.zeros_like(absolute)
    return torch.full_like(absolute, torch.inf)


def _solve_local_proposal(
        solver: LeastSquaresSolver,
        environment: torch.Tensor,
        target: torch.Tensor,
        current: torch.Tensor,
        site: int,
        sweep: int,
        update_policy: UpdatePolicy,
        return_record: bool,
        regularization_scale: Optional[torch.Tensor] = None,
        sampling_exact: Optional[bool] = None,
        sample_generation: Optional[int] = None
        ) -> Tuple[torch.Tensor, object]:
    """Solves, damps and optionally accepts one TT local proposal."""
    solution, record = solver.solve(
        environment,
        target,
        site=site,
        sweep=sweep,
        return_record=return_record,
        regularization_scale=regularization_scale)
    proposal = update_policy.apply(current, solution.reshape(current.shape))
    record_needs_update = return_record and (update_policy.damping != 1)
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
        residual = environment @ proposal.reshape(-1) - target
        residual_absolute = torch.linalg.vector_norm(residual)
        target_norm = torch.linalg.vector_norm(target)
        record = replace(
            record,
            residual_absolute=residual_absolute,
            residual_relative=_relative_error(
                residual_absolute, target_norm),
            target_norm=target_norm)
    if record is not None and sampling_exact is not None:
        record = replace(
            record,
            sampling_exact=sampling_exact,
            sample_generation=sample_generation)
    return proposal, record


def _gauge_core_update(
        cores: Sequence[torch.Tensor],
        versions: Sequence[int],
        fixed_sites: Sequence[int],
        gauge: GaugePolicy,
        direction: str,
        site: int,
        proposal: torch.Tensor) -> CoreUpdateSet:
    """Builds one atomic current-plus-receiver gauge update."""
    receiver = site + 1 if direction == 'forward' else site - 1
    legal_receiver = (0 <= receiver < len(cores)) and \
        (receiver not in fixed_sites)
    effective_gauge = gauge if legal_receiver else NoGauge()
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


class _TTALSBackend:
    """Adapts exact, sampled and completion TT solves to the common driver."""

    def __init__(self,
                 problem: ALSProblem,
                 cores: Sequence[torch.Tensor],
                 target: Optional[torch.Tensor],
                 solver: LeastSquaresSolver,
                 gauge: GaugePolicy,
                 fixed_sites: Sequence[int],
                 renormalize: bool,
                 sampler: Optional[RowSampler] = None,
                 n_samples: Optional[int] = None,
                 refresh_policy: Optional[SampleRefreshPolicy] = None,
                 generator: Optional[torch.Generator] = None) -> None:
        self.problem = problem
        self.full_target = None if target is None else target.reshape(-1)
        self.solver = solver
        self.gauge = gauge
        self.fixed_sites = frozenset(fixed_sites)
        self.cache = TTEnvironmentCache(
            cores, renormalize=renormalize)
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
                            .index_select(0, positions),
                            batch.ids):
                    raise ValueError(
                        'Observed sample ids should match fixed observations')
                self.current_target = observations.values.index_select(
                    0, positions.to(observations.values.device))
            else:
                indices = _unravel_indices(
                    batch.ids, self.cache.input_dim)
                configurations = ConfigurationBatch(
                    indices, kind='indices')
                evaluated = self.problem.evaluate(configurations)
                if evaluated.shape != (batch.ids.numel(),):
                    raise ValueError(
                        'TT-ALS currently requires a scalar tensor source')
                self.current_target = evaluated
        self.sample_batch = batch
        self.cache.prepare_sweep(order, samples=batch)
        return batch.generation, refreshed

    def solve_site(self,
                   site: int,
                   sweep: int,
                   update_policy: UpdatePolicy,
                   return_record: bool):
        local_environment = self.cache.local_environment(site)
        environment = local_environment.design()
        target = local_environment.scale_target(self.current_target)
        if self.sample_batch is not None:
            sample_weights = self.sample_batch.weights.to(environment.dtype)
            environment = environment * sample_weights.unsqueeze(1)
            target = target * sample_weights
        if self.problem.observations is not None and \
                (self.problem.observations.weights is not None):
            observation_weights = self.problem.observations.weights
            if self.sample_batch is not None:
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
            regularization_scale = (-2 * local_environment.log_scale).exp()
        current = self.cache.cores[site]
        sampling_exact = None if self.sample_batch is None else \
            self.sample_batch.is_exact_for(self.cache.core_versions)
        proposal, record = _solve_local_proposal(
            solver=self.solver,
            environment=environment,
            target=target,
            current=current,
            site=site,
            sweep=sweep,
            update_policy=update_policy,
            return_record=return_record,
            regularization_scale=regularization_scale,
            sampling_exact=sampling_exact,
            sample_generation=(
                None if self.sample_batch is None
                else self.sample_batch.generation))
        update_set = _gauge_core_update(
            cores=self.cache.cores,
            versions=self.cache.core_versions,
            fixed_sites=self.fixed_sites,
            gauge=self.gauge,
            direction=self._direction,
            site=site,
            proposal=proposal)
        return update_set, record

    def skip_site(self, site: int) -> None:
        self.cache.local_environment(site)
        self.commit(CoreUpdateSet(
            sites=(site,),
            cores=(self.cache.cores[site],),
            versions=(self.cache.core_versions[site] + 1,),
            reason='fixed_core'))

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
            approximation = _evaluate_standard_tt(
                self.cache.cores, indices)
            return problem.observations.error(approximation)

        approximation = _contract_standard_tt(self.cache.cores).reshape(-1)
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
        self.cache = TTEnvironmentCache(
            cores, renormalize=self.cache.renormalize)


def _direct_sampled_tt_design(
        cores: Sequence[torch.Tensor],
        site: int,
        indices: torch.Tensor,
        renormalize: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds correlated sampled TT rows for one independently drawn site."""
    n_samples = indices.shape[0]
    left = cores[0].new_ones((n_samples, 1))
    right = cores[0].new_ones((n_samples, 1))
    log_scale = cores[0].real.new_zeros(())

    for current in range(site):
        selected = cores[current][:, indices[:, current], :].permute(1, 0, 2)
        left = torch.einsum('ja,jab->jb', left, selected)
        if renormalize:
            norm = _environment_norm(left)
            if norm > 0:
                left = left / norm
                log_scale = log_scale + norm.log()
    for current in reversed(range(site + 1, len(cores))):
        selected = cores[current][:, indices[:, current], :].permute(1, 0, 2)
        right = torch.einsum('jab,jb->ja', selected, right)
        if renormalize:
            norm = _environment_norm(right)
            if norm > 0:
                right = right / norm
                log_scale = log_scale + norm.log()

    basis = torch.nn.functional.one_hot(
        indices[:, site], cores[site].shape[1]).to(
            device=cores[0].device, dtype=cores[0].dtype)
    design = torch.einsum(
        'ja,jp,jb->japb', left, basis, right)
    return design.reshape(n_samples, -1), log_scale


class _TTLeverageALSBackend:
    """Runs site-dependent recursive TT leverage sampling."""

    def __init__(self,
                 problem: ALSProblem,
                 cores: Sequence[torch.Tensor],
                 solver: LeastSquaresSolver,
                 gauge: GaugePolicy,
                 n_samples: int,
                 mode: str,
                 uniform_mix: float,
                 refresh_policy: SampleRefreshPolicy,
                 generator: Optional[torch.Generator],
                 renormalize: bool) -> None:
        self.problem = problem
        self._cores = TTEnvironmentCache._validate_cores(cores)
        self._versions = (0,) * len(self._cores)
        self.solver = solver
        self.gauge = gauge
        self.n_samples = n_samples
        self.mode = mode
        self.refresh_policy = refresh_policy
        self.generator = generator
        self.renormalize = renormalize
        self.sampler = TTLeverageRows(
            lambda: self._cores, uniform_mix=uniform_mix)
        self._sampling_state = _RowSamplingState(
            n_rows=prod(core.shape[1] for core in self._cores),
            core_versions=self._versions,
            device=self._cores[0].device)
        self._batches = {}
        self._targets = {}
        self._generation = None
        self._direction = None
        self._initialized = False
        self.sample_exact_flags = []

    @property
    def n_sites(self) -> int:
        return len(self._cores)

    @property
    def trainable_sites(self) -> Sequence[int]:
        return tuple(range(self.n_sites))

    @property
    def cores(self) -> Sequence[torch.Tensor]:
        return self._cores

    def _canonicalize_initial(self, direction: str) -> None:
        """Places the initial TT in mixed-canonical form at the first site."""
        cores = list(self._cores)
        versions = list(self._versions)
        gauge = self.gauge
        if direction == 'forward':
            sites = reversed(range(1, self.n_sites))
            factor_direction = 'reverse'
            receiver_offset = -1
        else:
            sites = range(self.n_sites - 1)
            factor_direction = 'forward'
            receiver_offset = 1
        for site in sites:
            cores[site], factor = gauge.factor(
                cores[site], factor_direction)
            receiver = site + receiver_offset
            cores[receiver] = gauge.absorb(
                factor, cores[receiver], factor_direction)
            versions[site] += 1
            versions[receiver] += 1
        self._cores = TTEnvironmentCache._validate_cores(cores)
        self._versions = tuple(versions)
        self._sampling_state = replace(
            self._sampling_state, core_versions=self._versions)

    def prepare_sweep(self,
                      order: Sequence[int],
                      sweep: int) -> Tuple[Optional[int], bool]:
        self._direction = 'forward' if order[0] == 0 else 'reverse'
        if not self._initialized:
            self._canonicalize_initial(self._direction)
            self._initialized = True

        generation = sweep if self.mode == 'exact' \
            else self.refresh_policy.generation(sweep)
        refreshed = generation != self._generation
        if refreshed:
            self._batches = {}
            self._targets = {}
            self._generation = generation
        return generation, refreshed

    def _site_batch(self, site: int) -> Tuple[SampleBatch, torch.Tensor, bool]:
        """Draws or reuses one site-dependent proposal and its target values."""
        if (self.mode == 'exact') or (site not in self._batches):
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
            target = self.problem.evaluate(
                ConfigurationBatch(indices, kind='indices'))
            if target.shape != (self.n_samples,):
                raise ValueError(
                    'TT-ALS currently requires a scalar tensor source')
            if not torch.isfinite(target).all():
                raise ValueError(
                    'The tensor source should return only finite values')
            self._batches[site] = batch
            self._targets[site] = target
        batch = self._batches[site]
        target = self._targets[site]
        exact = batch.is_exact_for(self._versions)
        self.sample_exact_flags.append(exact)
        return batch, target, exact

    def solve_site(self,
                   site: int,
                   sweep: int,
                   update_policy: UpdatePolicy,
                   return_record: bool):
        batch, target, sampling_exact = self._site_batch(site)
        indices = _unravel_indices(
            batch.ids, tuple(core.shape[1] for core in self._cores))
        environment, log_scale = _direct_sampled_tt_design(
            self._cores, site, indices, self.renormalize)
        target = target * (-log_scale).exp().to(target.dtype)
        sample_weights = batch.weights.to(environment.dtype)
        environment = environment * sample_weights.unsqueeze(1)
        target = target * sample_weights

        regularization_scale = None
        if (self.solver.l2_reg_mode == 'absolute') and \
                (self.solver.l2_reg > 0):
            regularization_scale = (-2 * log_scale).exp()
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
        update_set = _gauge_core_update(
            cores=self._cores,
            versions=self._versions,
            fixed_sites=(),
            gauge=self.gauge,
            direction=self._direction,
            site=site,
            proposal=proposal)
        return update_set, record

    def skip_site(self, site: int) -> None:
        raise RuntimeError('Leverage TT-ALS does not support fixed sites')

    def commit(self, update_set: CoreUpdateSet) -> None:
        cores = list(self._cores)
        versions = list(self._versions)
        for site, (core, version) in update_set.updates.items():
            if version <= versions[site]:
                raise ValueError('Every updated core version should increase')
            cores[site] = core
            versions[site] = version
        self._cores = TTEnvironmentCache._validate_cores(cores)
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
        self._cores = TTEnvironmentCache._validate_cores(cores)


class TTALS:
    """Approximates a fixed scalar tensor problem by TT-ALS.

    The source and its discrete input dimensions are fixed on construction;
    :meth:`completion` instead fixes a permanent set of observed entries.
    Repeated calls to :meth:`fit` may then compare initializations, ranks,
    gauges, solvers and convergence policies without rebuilding the source
    adapter. Exact ALS enumerates the complete discrete tensor once. Uniform
    ALS evaluates and caches only one sampled generation, while completion
    never queries entries outside its fixed observations.

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
        self.problem = ALSProblem(source=self.source)
        self.output_device = None if output_device is None \
            else torch.device(output_device)
        self._configurations = None
        self._target = None

    @classmethod
    def completion(cls,
                   observations,
                   values: Optional[torch.Tensor] = None,
                   input_dim: Optional[Sequence[int]] = None,
                   weights: Optional[torch.Tensor] = None,
                   *,
                   output_device: Optional[
                       Union[str, torch.device]] = 'cpu') -> 'TTALS':
        """Creates TT-ALS for a permanently observed completion objective.

        ``observations`` may already be :class:`ObservedEntries` or may be an
        integer tensor of global multi-indices. In the latter case, ``values``
        and the complete ``input_dim`` are required. Entries outside this
        fixed set remain unknown and are never interpreted as zeros.

        Parameters
        ----------
        observations : ObservedEntries or torch.Tensor
            Fixed observations or integer multi-indices with shape
            ``(observations, sites)``.
        values : torch.Tensor, optional
            Scalar value at every supplied index.
        input_dim : sequence of int, optional
            Complete input dimension, required with raw indices.
        weights : torch.Tensor, optional
            Non-negative multiplicative weight per observed value.
        output_device : str or torch.device, optional
            Device where finalized cores are stored. The default is ``"cpu"``.

        Returns
        -------
        TTALS
            Reusable completion decomposition whose :meth:`fit` defaults to
            fixed observed rows.

        Examples
        --------
        >>> indices = torch.tensor([[0, 0], [0, 1], [1, 1]])
        >>> values = torch.tensor([1., 2., 4.])
        >>> decomposition = TTALS.completion(
        ...     indices, values, input_dim=(2, 2))
        >>> result = decomposition.fit(rank=2)
        """
        if isinstance(observations, ObservedEntries):
            if (values is not None) or (input_dim is not None) or \
                    (weights is not None):
                raise ValueError(
                    '`values`, `input_dim` and `weights` belong inside an '
                    'existing ObservedEntries object')
            observed_entries = observations
        else:
            if not isinstance(observations, torch.Tensor):
                raise TypeError(
                    '`observations` should be ObservedEntries or torch.Tensor')
            if values is None:
                raise ValueError('`values` is required with observation indices')
            if input_dim is None:
                raise ValueError(
                    '`input_dim` is required with observation indices')
            observed_entries = ObservedEntries(
                indices=observations,
                values=values,
                input_dim=input_dim,
                weights=weights)
        if observed_entries.output_shape:
            raise ValueError(
                'TT-ALS completion currently requires scalar observations')

        instance = cls.__new__(cls)
        instance.source = None
        instance.problem = ALSProblem(observations=observed_entries)
        instance.output_device = None if output_device is None \
            else torch.device(output_device)
        instance._configurations = None
        instance._target = None
        return instance

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Input dimension fixed by the source or completion problem."""
        return self.problem.input_dim

    def _runtime_reference(self) -> torch.Tensor:
        """Returns one scalar carrying the source runtime without densifying."""
        if self.problem.observations is not None:
            return self.problem.observations.values
        if self.source.dtype is not None:
            return torch.empty(
                (), device=self.source.device, dtype=self.source.dtype)

        indices = torch.zeros(
            (1, len(self.input_dim)),
            device=self.source.device,
            dtype=torch.long)
        value = self.source.evaluate(
            ConfigurationBatch(indices, kind='indices'))
        if value.shape != (1,):
            raise ValueError(
                'TT-ALS currently requires a scalar tensor source')
        if not (value.is_floating_point() or value.is_complex()):
            raise TypeError(
                'The tensor source should return floating or complex values')
        if not torch.isfinite(value).all():
            raise ValueError(
                'The tensor source should return only finite values')
        return value

    def _exact_target(self) -> Tuple[ConfigurationBatch, torch.Tensor]:
        """Enumerates and caches the scalar source in global row order."""
        if self._target is None:
            if self.source is None:
                raise ValueError(
                    'Completion does not define unknown target entries')
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
            sampling: Optional[str] = None,
            n_samples: Optional[int] = None,
            sample_reuse_sweeps: int = 1,
            leverage_mode: str = 'exact',
            leverage_uniform_mix: float = 0.0,
            solver: Optional[LeastSquaresSolver] = None,
            convergence: Optional[ConvergencePolicy] = None,
            update_policy: Optional[UpdatePolicy] = None,
            renormalize: bool = True,
            generator: Optional[torch.Generator] = None,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0,
            observer: Optional[DecompositionObserver] = None
            ) -> TTDecomposition:
        """Fits a TT by alternating one-site least-squares solves.

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
        sampling : {``"exact"``, ``"uniform"``, ``"leverage"``,
            ``"observed"``}, optional
            Row strategy. The default is ``"exact"`` for a known source and
            ``"observed"`` for :meth:`completion`.
        n_samples : int, optional
            Number of uniformly sampled global configurations per generation.
            It is required only for ``sampling="uniform"``.
        sample_reuse_sweeps : int
            Complete sweeps that reuse exactly the same sampled ids,
            probabilities and cached source values before refreshing.
        leverage_mode : {``"exact"``, ``"frozen"``}
            Exact mode redraws from the current mixed-canonical design at every
            site. Frozen mode reuses each site's original ids and draw
            probabilities for the selected sample generation.
        leverage_uniform_mix : float
            Global uniform component mixed into leverage probabilities, in
            ``[0, 1]``. Positive values guarantee full row support.
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
                '`sampling="observed"` requires TTALS.completion')
        if sampling in ('uniform', 'leverage'):
            if isinstance(n_samples, bool) or \
                    (not isinstance(n_samples, int)) or (n_samples < 1):
                raise ValueError(
                    '`n_samples` should be a positive integer for sampled '
                    'sampling')
        elif n_samples is not None:
            raise ValueError(
                '`n_samples` is only used with uniform or leverage sampling')
        refresh_policy = SampleRefreshPolicy(
            reuse_sweeps=sample_reuse_sweeps)
        if leverage_mode not in ('exact', 'frozen'):
            raise ValueError(
                "`leverage_mode` should be 'exact' or 'frozen'")
        if sampling == 'leverage':
            if leverage_mode == 'exact' and sample_reuse_sweeps != 1:
                raise ValueError(
                    'Exact leverage sampling requires '
                    '`sample_reuse_sweeps=1`')
            if not isinstance(gauge_policy, (QRGauge, SVDGauge)):
                raise ValueError(
                    'Leverage sampling requires a QR or SVD gauge policy')
            if isinstance(leverage_uniform_mix, bool) or \
                    (not isinstance(leverage_uniform_mix, (int, float))) or \
                    (leverage_uniform_mix < 0) or \
                    (leverage_uniform_mix > 1):
                raise ValueError(
                    '`leverage_uniform_mix` should be in [0, 1]')
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
        cores, fixed_sites = self._initial_cores(
            target=runtime_reference,
            initial_cores=initial_cores,
            rank=rank,
            init=init,
            fixed_cores=fixed_cores,
            generator=generator)
        if sampling == 'leverage':
            if fixed_sites:
                raise ValueError(
                    'Leverage sampling does not support fixed cores because '
                    'they prevent mixed-canonical gauge preparation')
            backend = _TTLeverageALSBackend(
                problem=problem,
                cores=cores,
                solver=solver,
                gauge=gauge_policy,
                n_samples=n_samples,
                mode=leverage_mode,
                uniform_mix=leverage_uniform_mix,
                refresh_policy=refresh_policy,
                generator=generator,
                renormalize=renormalize)
        else:
            backend = _TTALSBackend(
                problem=problem,
                cores=cores,
                target=target,
                solver=solver,
                gauge=gauge_policy,
                fixed_sites=fixed_sites,
                renormalize=renormalize,
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
                'sampling': sampling,
                'n_samples': n_samples,
                'sample_reuse_sweeps': sample_reuse_sweeps,
                'leverage_mode': (
                    leverage_mode if sampling == 'leverage' else None),
                'leverage_uniform_mix': (
                    leverage_uniform_mix
                    if sampling == 'leverage' else None),
                'sampling_exact': (
                    None if sampling != 'leverage'
                    else all(backend.sample_exact_flags)),
                'exact_configurations': (
                    None if configurations is None
                    else configurations.batch_size),
            })


def tt_als(source,
           rank: Optional[int] = None,
           input_dim: Optional[Sequence[int]] = None,
           initial_cores=None,
           init: str = 'random',
           fixed_cores=None,
           gauge: Union[str, GaugePolicy] = 'qr',
           sampling: str = 'exact',
           n_samples: Optional[int] = None,
           sample_reuse_sweeps: int = 1,
           leverage_mode: str = 'exact',
           leverage_uniform_mix: float = 0.0,
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
    sampling : {``"exact"``, ``"uniform"``, ``"leverage"``}
        Whether local systems use all rows, uniform rows or recursive TT
        leverage rows.
    n_samples : int, optional
        Rows per uniform sample generation. Required for uniform sampling.
    sample_reuse_sweeps : int
        Sweeps that reuse sampled ids, probabilities and source values.
    leverage_mode : {``"exact"``, ``"frozen"``}
        Whether leverage probabilities are redrawn at every site or frozen by
        sample generation.
    leverage_uniform_mix : float
        Uniform mixture component added to leverage probabilities.
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
            sampling=sampling,
            n_samples=n_samples,
            sample_reuse_sweeps=sample_reuse_sweeps,
            leverage_mode=leverage_mode,
            leverage_uniform_mix=leverage_uniform_mix,
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
