"""
This script contains:

    Internal classes:
        * _TTALSBackend
        * _TTLeverageALSBackend

    Public classes:
        * TTALS

    Internal functions:
        * _standard_tt_cores
        * _result_tt_cores
        * _feasible_tt_rank
        * _contract_standard_tt
        * _evaluate_standard_tt
        * _prepare_tt_cores
        * _gauge_core_update
        * _direct_sampled_tt_design

    Public functions:
        * tt_als
"""

from dataclasses import asdict, replace
from math import prod
from typing import Optional, Sequence, TYPE_CHECKING, Tuple, Union

import torch

from tensorkrowch.formats.formats1d import TensorFormat1D, _MatrixFormat1D
from tensorkrowch.formats.quantics import _QuanticsVector, _QuanticsMatrix

from tensorkrowch.decompositions._utils import (_normalize_rank,
                                                _unravel_indices)
from tensorkrowch.decompositions.als.convergence import ConvergencePolicy
from tensorkrowch.decompositions.als.driver import (ALSSweepDriver,
                                                    _report_als_result)
from tensorkrowch.decompositions.als.environments import (CoreUpdateSet,
                                                          TTEnvironmentCache,
                                                          _environment_norm)
from tensorkrowch.decompositions.als.gauges import (_absorb_factor,
                                                    _factor_core)
from tensorkrowch.decompositions.als.problem import ALSProblem
from tensorkrowch.decompositions.als.sampling import (ObservedRows,
                                                      RowSampler,
                                                      SampleBatch,
                                                      SampleRefreshPolicy,
                                                      SamplingPolicy,
                                                      TTLeverageRows,
                                                      UniformRows,
                                                      _RowSamplingState)
from tensorkrowch.decompositions.als.solvers import (LeastSquaresSolver,
                                                     _relative_error,
                                                     _solve_local_proposal)
from tensorkrowch.decompositions.metrics import LocalSolveRecord
from tensorkrowch.decompositions.observers import (DecompositionEvent,
                                                   _normalize_verbosity,
                                                   _resolve_observer)
from tensorkrowch.decompositions.results import (TTDecomposition,
                                                 _result_from_source)
from tensorkrowch.decompositions.sources import (
    ConfigurationBatch,
    as_tensor_source,
    SparseTensorSource,
    BaseSource,
)
from tensorkrowch.decompositions.sources.base import _tensor_view
from tensorkrowch.decompositions.sources.dense import DenseTensorSource


if TYPE_CHECKING:
    from tensorkrowch.decompositions.sources.factory import SourceLike


def _standard_tt_cores(cores: Union[TTDecomposition, Sequence[torch.Tensor]],
                       in_dim: Sequence[int]) -> Tuple[torch.Tensor, ...]:
    """Normalizes lightweight or standard OBC core shapes."""
    if isinstance(cores, TensorFormat1D):
        if cores._cyclic:
            raise ValueError('Initial cores should have open topology')
        if cores.n_batches:
            raise ValueError('Batched TT cores are not supported by TT-ALS')
        cores = cores._effective_cores()
    if isinstance(cores, (str, bytes)) or \
            not isinstance(cores, Sequence):
        raise TypeError(
            '`initial_cores` should be a TTDecomposition or a core sequence')
    cores = tuple(cores)
    if len(cores) != len(in_dim):
        raise ValueError('`initial_cores` should contain one core per site')
    if not all(isinstance(core, torch.Tensor) for core in cores):
        raise TypeError('`initial_cores` should contain torch.Tensor objects')

    standard = []
    n_sites = len(cores)
    for site, (core, site_in_dim) in enumerate(zip(cores, in_dim)):
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
        if core.shape[1] != site_in_dim:
            raise ValueError(
                '`initial_cores` input dimensions should match the source')
        standard.append(core)

    return TTEnvironmentCache._validate_cores(tuple(standard))


def _result_tt_cores(
    cores: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    """Removes explicit unit ranks from TT boundary ``cores``."""
    cores = tuple(cores)
    if len(cores) == 1:
        return (cores[0].squeeze(0).squeeze(-1),)
    return (cores[0].squeeze(0), *cores[1:-1], cores[-1].squeeze(-1))


def _feasible_tt_rank(in_dim: Sequence[int],
                      rank: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    """Clips each requested TT rank to its algebraic maximum."""
    ranks = _normalize_rank(rank, len(in_dim) - 1)
    ranks = [min(cap, prod(in_dim[:cut]), prod(in_dim[cut:]))
             for cut, cap in enumerate(ranks, 1)]
    for site in range(1, len(ranks)):
        ranks[site] = min(ranks[site], ranks[site - 1] * in_dim[site])
    for site in reversed(range(len(ranks) - 1)):
        ranks[site] = min(ranks[site], ranks[site + 1] * in_dim[site + 1])
    return tuple(ranks)


def _contract_standard_tt(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Contracts standard OBC ``cores`` without constructing a model graph."""
    result = cores[0].squeeze(0)
    for core in cores[1:]:
        result = torch.einsum('...a,apb->...pb', result, core)
    return result.squeeze(-1)


def _evaluate_standard_tt(cores: Sequence[torch.Tensor],
                          indices: torch.Tensor) -> torch.Tensor:
    """Evaluates standard OBC ``cores`` at packed discrete configurations."""
    environment = cores[0][:, indices[:, 0], :].squeeze(0)
    for site, core in enumerate(cores[1:], 1):
        selected = core[:, indices[:, site], :].permute(1, 0, 2)
        environment = torch.einsum('ja,jab->jb', environment, selected)
    return environment.squeeze(-1)


def _prepare_tt_cores(cores: Sequence[torch.Tensor],
                      sites: Sequence[int],
                      direction: str) -> Tuple[torch.Tensor, ...]:
    """Moves QR factors through ``sites`` into their directional neighbors."""
    prepared = list(cores)
    offset = 1 if direction == 'forward' else -1
    for site in sites:
        prepared[site], factor = _factor_core(prepared[site], direction)
        receiver = site + offset
        prepared[receiver] = _absorb_factor(
            factor, prepared[receiver], direction)
    return tuple(prepared)


def _gauge_core_update(cores: Sequence[torch.Tensor],
                       versions: Sequence[int],
                       fixed_sites: Sequence[int],
                       gauge: str,
                       direction: str,
                       site: int,
                       proposal: torch.Tensor) -> CoreUpdateSet:
    """Builds one atomic current-plus-receiver ``gauge`` update."""
    receiver = site + 1 if direction == 'forward' else site - 1
    legal_receiver = (0 <= receiver < len(cores)) and \
        (receiver not in fixed_sites)
    if (gauge == 'qr') and legal_receiver:
        gauged_core, factor = _factor_core(proposal, direction)
    else:
        gauged_core, factor = proposal, None

    sites = [site]
    updated_cores = [gauged_core]
    updated_versions = [versions[site] + 1]
    if factor is not None:
        neighbor = _absorb_factor(
            factor, cores[receiver], direction)
        sites.append(receiver)
        updated_cores.append(neighbor)
        updated_versions.append(versions[receiver] + 1)
    return CoreUpdateSet(
        sites=sites,
        cores=updated_cores,
        versions=updated_versions,
        reason='local_solve')


class _TTALSBackend:  # MARK: _TTALSBackend
    """Adapts exact, sampled and completion TT solves to the common driver."""

    def __init__(self,
                 problem: ALSProblem,
                 cores: Sequence[torch.Tensor],
                 target: Optional[torch.Tensor],
                 solver: LeastSquaresSolver,
                 gauge: str,
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
            n_rows=prod(self.cache.in_dim),
            core_versions=self.cache.core_versions,
            device=self.cache.cores[0].device)
        self._direction = None

    @property
    def n_sites(self) -> int:
        """Number of cores in this backend."""
        return len(self.cache.cores)

    @property
    def trainable_sites(self) -> Sequence[int]:
        """Sites that may receive local updates."""
        return tuple(site for site in range(self.n_sites)
                     if site not in self.fixed_sites)

    @property
    def cores(self) -> Sequence[torch.Tensor]:
        """Current cores in original site order."""
        return self.cache.cores

    def prepare_sweep(self,
                      order: Sequence[int],
                      sweep: int) -> Tuple[Optional[int], bool]:
        """
        Prepares directional environments and the current sampling
        generation.
        """
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
                self.current_target = self.problem.observations.support_values
            else:
                indices = _unravel_indices(
                    batch.ids, self.cache.in_dim)
                configurations = ConfigurationBatch(
                    indices, kind='indices')
                evaluated = self.problem.evaluate(configurations)
                if evaluated.shape != (batch.ids.numel(),):
                    raise ValueError(
                        'TT-ALS currently requires a scalar tensor source')
                if not torch.isfinite(evaluated).all():
                    raise ValueError(
                        '`source` should return only finite values')
                self.current_target = evaluated
        self.sample_batch = batch
        self.cache.prepare_sweep(order, samples=batch)
        return batch.generation, refreshed

    def solve_site(self,
                   site: int,
                   sweep: int,
                   return_record: bool) -> Tuple[
            CoreUpdateSet, Optional[LocalSolveRecord]]:
        """Builds one local proposal and its atomic gauge update."""
        local_environment = self.cache.local_environment(site)
        environment = local_environment.design()
        target = local_environment.scale_target(self.current_target)
        if self.sample_batch is not None:
            sample_weights = self.sample_batch.weights.to(environment.dtype)
            environment = environment * sample_weights.unsqueeze(1)
            target = target * sample_weights
        if (self.problem.observations is not None) and \
                (self.problem.weights is not None):
            weights = self.problem.weights.to(environment.dtype)
            environment = environment * weights.unsqueeze(1)
            target = target * weights
        if self.problem.observations is None:
            ids = self.sample_batch.ids if self.sample_batch is not None else \
                torch.arange(prod(self.problem.in_dim), device=target.device)
            indices = _unravel_indices(ids, self.problem.in_dim)
            weights = self.problem.entry_weights(indices)
            if weights is not None:
                weights = weights.to(environment.dtype)
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
        """Advances the sweep through a fixed core."""
        self.cache.local_environment(site)
        self.commit(CoreUpdateSet(
            sites=(site,),
            cores=(self.cache.cores[site],),
            versions=(self.cache.core_versions[site] + 1,),
            reason='fixed_core'))

    def commit(self, update_set: CoreUpdateSet) -> None:
        """
        Commits all replaced cores and invalidates their cache
        dependencies.
        """
        self.cache.commit(update_set)
        if self.sampler is not None:
            for site in update_set.sites:
                self._sampling_state = self.sampler.update_after_core(
                    self._sampling_state, site)

    def measure_objective(self,
                          problem: ALSProblem) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """Measures the fixed complete-sweep objective when one exists."""
        if problem.observations is not None:
            indices = problem.observations.support.values.to(
                self.cache.cores[0].device)
            approximation = _evaluate_standard_tt(
                self.cache.cores, indices)
            return problem.objective_error(approximation)

        approximation = _contract_standard_tt(self.cache.cores).reshape(-1)
        residual = approximation - self.full_target
        target = self.full_target
        indices = _unravel_indices(torch.arange(target.numel(),
                                                device=target.device),
                                    problem.in_dim)
        weights = problem.entry_weights(indices)
        if weights is not None:
            residual = residual * weights
            target = target * weights
        absolute = torch.linalg.vector_norm(residual)
        target_norm = torch.linalg.vector_norm(target)
        return absolute, _relative_error(absolute, target_norm)

    def snapshot(self) -> Sequence[torch.Tensor]:
        """Returns the cores forming the current approximation."""
        return tuple(core.clone() for core in self.cache.cores)

    def restore(self, cores: Sequence[torch.Tensor]) -> None:
        """Restores a saved approximation and refreshes its dependent state."""
        self.cache = TTEnvironmentCache(
            cores, renormalize=self.cache.renormalize)


def _direct_sampled_tt_design(cores: Sequence[torch.Tensor],
                              site: int,
                              indices: torch.Tensor,
                              renormalize: bool) -> Tuple[
            torch.Tensor, torch.Tensor]:
    """
    Builds correlated sampled TT rows for one independently drawn ``site``.
    """
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


class _TTLeverageALSBackend:  # MARK: _TTLeverageALSBackend
    """Runs site-dependent recursive TT leverage sampling."""

    def __init__(self,
                 problem: ALSProblem,
                 cores: Sequence[torch.Tensor],
                 solver: LeastSquaresSolver,
                 gauge: str,
                 n_samples: int,
                 uniform_mix: float,
                 generator: Optional[torch.Generator],
                 renormalize: bool) -> None:
        self.problem = problem
        self._cores = TTEnvironmentCache._validate_cores(cores)
        self._versions = (0,) * len(self._cores)
        self.solver = solver
        self.gauge = gauge
        self.n_samples = n_samples
        self.generator = generator
        self.renormalize = renormalize
        self.sampler = TTLeverageRows(
            lambda: self._cores, uniform_mix=uniform_mix)
        self._sampling_state = _RowSamplingState(
            n_rows=prod(core.shape[1] for core in self._cores),
            core_versions=self._versions,
            device=self._cores[0].device)
        self._generation = None
        self._direction = None

    @property
    def n_sites(self) -> int:
        """Number of cores in this backend."""
        return len(self._cores)

    @property
    def trainable_sites(self) -> Sequence[int]:
        """Sites that may receive local updates."""
        return tuple(range(self.n_sites))

    @property
    def cores(self) -> Sequence[torch.Tensor]:
        """Current cores in original site order."""
        return self._cores

    def _canonicalize_start(self, direction: str) -> None:
        """Places the TT in mixed-canonical form at the first active site."""
        versions = list(self._versions)
        if direction == 'forward':
            sites = range(self.n_sites - 1, 0, -1)
            factor_direction = 'reverse'
            receiver_offset = -1
        else:
            sites = range(self.n_sites - 1)
            factor_direction = 'forward'
            receiver_offset = 1
        self._cores = _prepare_tt_cores(self._cores, sites, factor_direction)
        for site in sites:
            versions[site] += 1
            versions[site + receiver_offset] += 1
        self._versions = tuple(versions)
        self._sampling_state = replace(
            self._sampling_state, core_versions=self._versions)

    def _site_batch(self, site: int) -> Tuple[SampleBatch, torch.Tensor, bool]:
        """Draws leverage rows from the current mixed-canonical TT."""
        state = replace(self._sampling_state, core_versions=self._versions,
                        generation=self._generation)
        batch = self.sampler.draw(state, site, self.n_samples, self.generator)
        indices = _unravel_indices(
            batch.ids, tuple(core.shape[1] for core in self._cores))
        target = self.problem.evaluate(ConfigurationBatch(indices))
        if target.shape != (self.n_samples,):
            raise ValueError(
                '`source` should return one scalar per configuration')
        if not torch.isfinite(target).all():
            raise ValueError('`source` should return only finite values')
        return batch, target, batch.is_exact_for(self._versions)

    def prepare_sweep(self,
                      order: Sequence[int],
                      sweep: int) -> Tuple[Optional[int], bool]:
        """Places the orthogonality center at the sweep's first site."""
        direction = 'forward' if order[0] == 0 else 'reverse'
        if (self._direction is None) or (direction == self._direction):
            self._canonicalize_start(direction)
        self._direction = direction
        self._generation = sweep
        return sweep, True

    def solve_site(self,
                   site: int,
                   sweep: int,
                   return_record: bool) -> Tuple[
            CoreUpdateSet, Optional[LocalSolveRecord]]:
        """Builds one local proposal and its atomic gauge update."""
        batch, target, sampling_exact = self._site_batch(site)
        indices = _unravel_indices(
            batch.ids, tuple(core.shape[1] for core in self._cores))
        environment, log_scale = _direct_sampled_tt_design(
            self._cores, site, indices, self.renormalize)
        target = target * (-log_scale).exp().to(target.dtype)
        sample_weights = batch.weights.to(environment.dtype)
        environment = environment * sample_weights.unsqueeze(1)
        target = target * sample_weights
        weights = self.problem.entry_weights(indices)
        if weights is not None:
            weights = weights.to(environment.dtype)
            environment = environment * weights.unsqueeze(1)
            target = target * weights

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
        """Advances the sweep through a fixed core."""
        raise RuntimeError('Leverage TT-ALS does not support fixed sites')

    def commit(self, update_set: CoreUpdateSet) -> None:
        """
        Commits all replaced cores and invalidates their cache
        dependencies.
        """
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

    def measure_objective(self,
                          problem: ALSProblem) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """Measures the fixed complete-sweep objective when one exists."""
        raise RuntimeError(
            'Renewable leverage batches do not define a global objective')

    def snapshot(self) -> Sequence[torch.Tensor]:
        """Returns the cores forming the current approximation."""
        return tuple(core.clone() for core in self._cores)

    def restore(self, cores: Sequence[torch.Tensor]) -> None:
        """Restores a saved approximation and refreshes its dependent state."""
        self._cores = TTEnvironmentCache._validate_cores(cores)


class TTALS:  # MARK: TTALS
    """
    Approximates a fixed finite tensor by one-site TT-ALS.

    Construction fixes the source, its dimensions and the weighted objective.
    Repeated :meth:`fit` calls can use different ranks, initial cores, samplers
    and solvers. Exact fits cache all target values on first use; sampled fits
    query only their selected configurations. A sparse boolean mask defines
    permanent observations, including observed values that are zero.

    Vector, matrix and Quantics sources use the same scalar local solves;
    the result restores the source's paired dimensions and coordinate metadata.
    Continuous inputs need discretization before ALS. Structural batches are
    not supported. Sweeps alternate forward and reverse; each directional
    pass counts once toward ``max_sweeps``.

    Parameters
    ----------
    source : BaseSource, TensorFormat1D, MPS, MPO or torch.Tensor
        Finite tensor input, with shape and evaluation defined by the source.
    mask : BaseSource or torch.Tensor, optional
        Boolean tensor of participating entries. ``True`` participates;
        ``False`` is missing. A sparse mask fixes its stored true entries.
    weights : torch.Tensor, optional
        Non-negative multiplicative residual weights with shape ``in_dim``
        of the scalar problem; matrix input/output pairs are fused.
    out_device : str or torch.device, optional
        Device where finalized cores are stored. The default is ``"cpu"``;
        ``None`` keeps them on the computation device.

    Examples
    --------
    >>> engine = tk.decompositions.TTALS(torch.ones(2, 3))
    >>> result = engine.fit(rank=1)
    >>> tuple(result.contract_dense().shape)
    (2, 3)
    """

    def __init__(self,
                 source: 'SourceLike',
                 *,
                 mask: Optional[Union[BaseSource, torch.Tensor]] = None,
                 weights: Optional[torch.Tensor] = None,
                 out_device: Optional[
                     Union[str, torch.device]] = 'cpu') -> None:
        self._source = as_tensor_source(source)
        if not self._source.is_discrete_tensor:
            raise ValueError('`source` should represent a discrete tensor')
        if self._source.n_batches:
            raise ValueError('`source` should not have structural batch axes')
        tensor_source = _tensor_view(self._source)
        if isinstance(mask, torch.Tensor) and self._source.is_matrix and \
                (tuple(mask.shape) != tensor_source.in_dim):
            mask = DenseTensorSource.matrix(
                mask, ordering=getattr(self._source, '_ordering', 'grouped'))
        mask_source = None if mask is None else as_tensor_source(mask)
        mask_view = None if mask_source is None else _tensor_view(mask_source)
        problem = ALSProblem(source=tensor_source,
                             mask=mask_view, weights=weights)
        if isinstance(mask_source, SparseTensorSource):
            if mask_source.dtype != torch.bool:
                raise TypeError('`mask` should contain boolean values')
            if mask_source.out_shape and not mask_source.is_matrix:
                raise ValueError('Sparse `mask` should store scalar entries')
            indices = mask_source.support.values[mask_source.support_values]
            if mask_source.is_matrix:
                sizes = indices.new_tensor(mask_source.out_dim)
                indices = indices[:, ::2] * sizes + indices[:, 1::2]
            elif mask_source.in_sites != tuple(range(mask_source.n_sites)):
                indices = indices[:, tuple(mask_source.in_sites.index(site)
                                           for site in range(
                                               mask_source.n_sites))]
            values = tensor_source.evaluate(ConfigurationBatch(indices))
            observations = SparseTensorSource(
                indices, values, tensor_source.in_dim)
            row_weights = problem.entry_weights(observations.support.values)
            if row_weights is not None:
                row_weights = row_weights.to(values.real.dtype)
            problem = ALSProblem(source=tensor_source,
                                 observations=observations,
                                 weights=row_weights)
        self._problem = problem
        self._out_device = None if out_device is None else torch.device(
            out_device)
        self._configurations = None
        self._target = None

    @property
    def source(self) -> 'BaseSource':
        """Original source, including its tensor or coordinate metadata."""
        return self._source

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension fixed by the source or completion problem."""
        return self._problem.in_dim

    @property
    def problem(self) -> ALSProblem:
        """Fixed source, mask and residual weights used by this engine."""
        return self._problem

    @property
    def out_device(self) -> Optional[torch.device]:
        """Device where finalized cores are stored."""
        return self._out_device

    def _validate_initial_format(self, initial_cores: object) -> None:
        """
        Checks paired dimensions and Quantics layouts before scalarization.
        """
        if isinstance(initial_cores, _MatrixFormat1D):
            if not self._source.is_matrix or \
                    (initial_cores.in_dim != self._source.in_dim) or \
                    (initial_cores.out_dim != self._source.out_dim):
                raise ValueError(
                    '`initial_cores` should match source input/output '
                    'dimensions')
        if isinstance(initial_cores, _QuanticsVector):
            if initial_cores.layout != getattr(self._source, 'layout', None):
                raise ValueError(
                    '`initial_cores` should match the source digit layout')
        elif isinstance(initial_cores, _QuanticsMatrix):
            if (initial_cores.in_layout != getattr(
                    self._source, 'in_layout', None)) or \
                    (initial_cores.out_layout != getattr(
                        self._source, 'out_layout', None)):
                raise ValueError(
                    '`initial_cores` should match source matrix layouts')

    def _runtime_reference(self) -> torch.Tensor:
        """
        Returns one scalar carrying the source runtime without densifying.
        """
        if self._problem.observations is not None:
            return self._problem.observations.support_values
        if self._problem.source.dtype is not None:
            return torch.empty(
                (), device=self._problem.source.device,
                dtype=self._problem.source.dtype)

        indices = torch.zeros(
            (1, len(self.in_dim)),
            device=self._problem.source.device,
            dtype=torch.long)
        value = self._problem.source.evaluate(
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
            if self._problem.source is None:
                raise ValueError(
                    'Completion does not define unknown target entries')
            flat_ids = torch.arange(
                prod(self.in_dim), device=self._problem.source.device)
            indices = _unravel_indices(flat_ids, self.in_dim)
            configurations = ConfigurationBatch(indices, kind='indices')
            target = self._problem.source.evaluate(configurations)
            if target.shape != (flat_ids.numel(),):
                raise ValueError(
                    'TT-ALS currently requires a scalar tensor source')
            if not (target.is_floating_point() or target.is_complex()):
                raise TypeError(
                    'The tensor source should return floating or complex'
                    ' values')
            if not torch.isfinite(target).all():
                raise ValueError(
                    'The tensor source should return only finite values')
            self._configurations = configurations
            self._target = target.reshape(self.in_dim)
        return self._configurations, self._target

    def _random_cores(self,
                      ranks: Sequence[int],
                      dtype: torch.dtype,
                      device: torch.device,
                      generator: Optional[torch.Generator]) -> Tuple[
            torch.Tensor, ...]:
        """Initializes random TT cores at feasible ranks."""
        boundary_ranks = (1, *ranks, 1)
        cores = []
        for site, site_in_dim in enumerate(self.in_dim):
            core = torch.randn(
                boundary_ranks[site],
                site_in_dim,
                boundary_ranks[site + 1],
                device=device,
                dtype=dtype,
                generator=generator)
            core = core / max(1, boundary_ranks[site] * site_in_dim) ** 0.5
            cores.append(core)
        return tuple(cores)

    def _initial_cores(self,
                       target: torch.Tensor,
                       initial_cores: Optional[Union[
                           TensorFormat1D, Sequence[torch.Tensor]]],
                       rank: Optional[Union[int, Sequence[int]]],
                       fixed_cores: Optional[Sequence[Optional[torch.Tensor]]],
                       generator: Optional[
                           torch.Generator]) -> Tuple[
            Tuple[torch.Tensor, ...],
            Tuple[int, ...]]:
        """Builds and validates initialization plus fixed-site semantics."""
        rank_caps = None if rank is None else _feasible_tt_rank(
            self.in_dim, rank)
        if initial_cores is None:
            if rank_caps is None:
                raise ValueError('`rank` is required without `initial_cores`')
            cores = self._random_cores(
                rank_caps, target.dtype, target.device, generator)
        else:
            cores = _standard_tt_cores(initial_cores, self.in_dim)

        if any((core.device != target.device) or (core.dtype != target.dtype)
               for core in cores):
            raise ValueError(
                'Initial cores and source values should share dtype and '
                'device')
        current_ranks = tuple(core.shape[-1] for core in cores[:-1])
        if (rank_caps is not None) and any(
                current > cap
                for current, cap in zip(current_ranks, rank_caps)):
            raise ValueError(
                '`initial_cores` exceed the requested `rank` caps')
        algebraic_caps = _feasible_tt_rank(self.in_dim, current_ranks)
        if any(current_rank > allowed
               for current_rank, allowed in zip(current_ranks,
                                                 algebraic_caps)):
            raise ValueError('Initial TT ranks are not algebraically feasible')

        if any(not torch.isfinite(core).all() for core in cores):
            raise ValueError(
                '`initial_cores` should contain only finite values')
        if fixed_cores is None:
            return tuple(core.clone() for core in cores), ()
        if isinstance(fixed_cores, (str, bytes)) or \
                not isinstance(fixed_cores, Sequence):
            raise TypeError(
                '`fixed_cores` should contain one tensor or None per site')
        fixed_cores = tuple(fixed_cores)
        if len(fixed_cores) != len(self.in_dim):
            raise ValueError('`fixed_cores` should contain one entry per site')

        final_cores = list(cores)
        fixed_sites = []
        for site, fixed_core in enumerate(fixed_cores):
            if fixed_core is None:
                continue
            if not isinstance(fixed_core, torch.Tensor):
                raise TypeError(
                    '`fixed_cores` should contain torch.Tensor objects '
                    'or None')
            if len(self.in_dim) == 1:
                normalized = _standard_tt_cores(
                    (fixed_core,), self.in_dim)[0]
            else:
                core = fixed_core
                if (site == 0) and (core.ndim == 2):
                    core = core.unsqueeze(0)
                elif (site == (len(self.in_dim) - 1)) and (core.ndim == 2):
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
            if not torch.isfinite(normalized).all():
                raise ValueError(
                    '`fixed_cores` should contain only finite values')
            final_cores[site] = normalized
            fixed_sites.append(site)
        final_cores = TTEnvironmentCache._validate_cores(final_cores)
        return final_cores, tuple(fixed_sites)

    def fit(self,
            rank: Optional[Union[int, Sequence[int]]] = None,
            initial_cores: Optional[Union[TensorFormat1D,
                Sequence[torch.Tensor]]] = None,
            fixed_cores: Optional[Sequence[Optional[torch.Tensor]]] = None,
            gauge: str = 'qr',
            renormalize: bool = True,
            sampler: Optional[SamplingPolicy] = None,
            solver: Optional[LeastSquaresSolver] = None,
            convergence: Optional[ConvergencePolicy] = None,
            generator: Optional[torch.Generator] = None,
            collect_metrics: bool = False,
            verbose: Union[bool, int] = 0) -> 'TTDecomposition':
        """
        Fits a TT by alternating one-site least-squares solves.

        Sweeps alternate forward and reverse. Each directional pass counts
        once toward ``max_sweeps``.
        Without ``initial_cores``, initializes random cores at the requested
        ranks. To start from an SVD approximation, compute it separately and
        pass its result as ``initial_cores``. Supplied cores are copied.

        With ``gauge="qr"``, prepares right-isometric cores before the first
        sweep, starting at the right end of the initial trainable segment.
        This segment ends before the first fixed core, or at the chain's
        right boundary when no cores are fixed. The contraction is preserved;
        fixed cores and sites beyond that segment are unchanged.

        Parameters
        ----------
        rank : int or sequence of int, optional
            One upper bound per internal link, or a shared bound. Random ranks
            are clipped to feasible dimensions. Required without
            ``initial_cores``.
        initial_cores : TensorFormat1D or sequence of torch.Tensor, optional
            Initial approximation with matching topology, physical dimensions,
            dtype and device. Its ranks must respect ``rank`` when provided.
        fixed_cores : sequence of torch.Tensor or None, optional
            One entry per site. Tensor entries replace the corresponding
            initial core and remain bitwise unchanged throughout ALS.
        gauge : {``"none"``, ``"qr"``}
            QR moves a factor into the immediate trainable neighbor after each
            local solve. No factor is discarded if that receiver is
            unavailable.
        renormalize : bool
            Whether cached environments remove global norms and retain their
            logarithmic scales. The local least-squares objective is preserved.
        sampler : SamplingPolicy, optional
            Row-sampling configuration. ``None`` uses the full tensor or the
            permanent observations selected by a sparse boolean mask.
        solver : LeastSquaresSolver, optional
            Local solver and its scaling and regularization settings. Defaults
            to an unregularized solver with automatic column scaling.
        convergence : ConvergencePolicy, optional
            Complete-sweep stopping criteria. Defaults to ten sweeps.
        generator : torch.Generator, optional
            Generator used by random initialization and row sampling.
            Its device
            must match the source values.
        collect_metrics : bool
            Whether to record local solves, complete-sweep errors and timings.
        verbose : bool or int
            Console verbosity from 0 (silent) to 3 (most detailed).

        Returns
        -------
        TTDecomposition
            Approximation with cores, metadata and optional metrics. Matrix and
            Quantics sources select the corresponding result family.

        Examples
        --------
        >>> tensor = torch.ones(2, 3, 2, dtype=torch.float64)
        >>> result = tk.decompositions.TTALS(tensor).fit(rank=1)
        >>> torch.allclose(result.contract_dense(), tensor)
        True
        """
        self._validate_initial_format(initial_cores)
        if not isinstance(gauge, str):
            raise TypeError('`gauge` should be str type')
        if gauge not in ('none', 'qr'):
            raise ValueError("`gauge` should be 'none' or 'qr'")
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        if (generator is not None) and \
                not isinstance(generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator or None')
        if solver is None:
            solver = LeastSquaresSolver()
        elif not isinstance(solver, LeastSquaresSolver):
            raise TypeError('`solver` should be LeastSquaresSolver type')
        if convergence is None:
            convergence = ConvergencePolicy()
        elif not isinstance(convergence, ConvergencePolicy):
            raise TypeError('`convergence` should be ConvergencePolicy type')
        if sampler is not None:
            if not isinstance(sampler, SamplingPolicy):
                raise TypeError('`sampler` should be SamplingPolicy or None')
            sampler.validate('tt')
            if self._problem.observations is not None:
                raise ValueError(
                    '`sampler` cannot replace permanent observations')
        convergence.validate_objective(sampler is None)

        verbosity = _normalize_verbosity(verbose)
        collect_metrics = collect_metrics or bool(verbosity)
        fit_observer = _resolve_observer(
            verbosity, None) if verbosity else None
        configurations = None
        target = None
        problem = self._problem
        row_sampler = None
        n_samples = None if sampler is None else sampler.n_samples
        refresh_policy = SampleRefreshPolicy(
            reuse_sweeps=1 if sampler is None else sampler.reuse_sweeps)
        if problem.observations is not None:
            sampling = 'observed'
            row_sampler = ObservedRows(problem.observations)
            runtime_reference = problem.observations.support_values
        elif sampler is None:
            sampling = 'exact'
            configurations, target = self._exact_target()
            runtime_reference = target
        else:
            sampling = sampler.method
            problem = replace(problem, sampled=True)
            runtime_reference = self._runtime_reference()
            if sampling == 'uniform':
                row_sampler = UniformRows()
        if not (runtime_reference.is_floating_point() or
                runtime_reference.is_complex()):
            raise TypeError('`source` should have floating or complex values')
        if (generator is not None) and \
                (torch.device(generator.device) != runtime_reference.device):
            raise ValueError(
                '`generator` and source values should share a device')
        cores, fixed_sites = self._initial_cores(
            target=runtime_reference, initial_cores=initial_cores, rank=rank,
            fixed_cores=fixed_cores, generator=generator)
        if sampling == 'leverage':
            if (gauge != 'qr') or fixed_sites:
                raise ValueError(
                    'TT leverage requires `gauge="qr"` without fixed cores')
            backend = _TTLeverageALSBackend(
                problem, cores, solver, gauge, n_samples,
                sampler.uniform_mix, generator, renormalize)
        else:
            if gauge == 'qr':
                stop = min(fixed_sites, default=len(cores))
                cores = _prepare_tt_cores(
                    cores, range(stop - 1, 0, -1), 'reverse')
            backend = _TTALSBackend(
                problem, cores, target, solver, gauge, fixed_sites,
                renormalize, row_sampler, n_samples, refresh_policy, generator)
        if fit_observer is not None:
            fit_observer.emit(DecompositionEvent(
                name='start', phase='TT-ALS', values={
                    'sites': len(cores), 'in_dim': self.in_dim,
                    'sampling': sampling, 'gauge': gauge,
                    'max_sweeps': convergence.max_sweeps,
                    'sweep_direction': 'alternating',
                }))
        fitted = ALSSweepDriver().fit(
            problem, backend, convergence, observer=fit_observer,
            collect_metrics=collect_metrics, phase='TT-ALS',
            sweep_direction='alternating')
        result_cores = _result_tt_cores(fitted.cores)
        if self._out_device is not None:
            result_cores = tuple(core.to(self._out_device)
                                 for core in result_cores)
        result = TTDecomposition(
            cores=result_cores, metrics=fitted.metrics, metadata={
                'algorithm': 'tt_als',
                'initialization': (
                    'random' if initial_cores is None else 'cores'),
                'gauge': gauge, 'converged': fitted.converged,
                'keep_best': convergence.keep_best,
                'stop_reason': fitted.stop_reason, 'n_sweeps': fitted.n_sweeps,
                'sweep_direction': 'alternating',
                'fixed_sites': list(fixed_sites),
                'sampling': sampling, 'n_samples': n_samples,
                'sampler': None if sampler is None else asdict(sampler),
                'solver': {'l2_reg': solver.l2_reg,
                           'l2_reg_mode': solver.l2_reg_mode,
                           'rcond': solver.rcond,
                           'column_scaling': solver.column_scaling,
                           'system_scaling': solver.system_scaling},
                'exact_configurations': None if configurations is None
                else configurations.batch_size,
            })
        result = _result_from_source(result, self._source)
        if fit_observer is not None:
            _report_als_result(result, fit_observer, 'TT-ALS')
        return result


def tt_als(source: 'SourceLike',
           rank: Optional[Union[int, Sequence[int]]] = None,
           initial_cores: Optional[Union[TensorFormat1D,
               Sequence[torch.Tensor]]] = None,
           fixed_cores: Optional[Sequence[Optional[torch.Tensor]]] = None,
           gauge: str = 'qr',
           renormalize: bool = True,
           max_sweeps: int = 10,
           atol: Optional[float] = None,
           rtol: Optional[float] = None,
           out_device: Optional[Union[str, torch.device]] = 'cpu',
           generator: Optional[torch.Generator] = None,
           collect_metrics: bool = False,
           verbose: Union[bool, int] = 0,
           *,
           mask: Optional[Union[BaseSource, torch.Tensor]] = None,
           weights: Optional[torch.Tensor] = None) -> 'TTDecomposition':
    """
    Approximates a finite tensor with one-site TT-ALS.

    Returns a decomposition object. Use :class:`TTALS` and its :meth:`fit
    <TTALS.fit>` method for sampled fits, customized local solvers or advanced
    convergence criteria. Sweeps alternate forward and reverse; each
    directional pass counts once toward ``max_sweeps``.

    Parameters
    ----------
    source : BaseSource, TensorFormat1D, MPS, MPO or torch.Tensor
        Finite tensor input, with shape and evaluation defined by the source.
    rank : int or sequence of int, optional
        One upper bound per internal link, or a shared bound. Random ranks
        are clipped to feasible dimensions. Required without ``initial_cores``.
    initial_cores : TensorFormat1D or sequence of torch.Tensor, optional
        Initial approximation with matching topology, physical dimensions,
        dtype and device. Its ranks must respect ``rank`` when provided.
    fixed_cores : sequence of torch.Tensor or None, optional
        One entry per site. Tensor entries replace the corresponding
        initial core and remain bitwise unchanged throughout ALS.
    gauge : {``"none"``, ``"qr"``}
        QR moves a factor into the immediate trainable neighbor after each
        local solve. No factor is discarded if that receiver is unavailable.
    renormalize : bool
        Whether cached environments remove global norms and retain their
        logarithmic scales. The local least-squares objective is preserved.
    max_sweeps : int
        Positive maximum number of directional sweeps. Default is ``10``.
    atol : float, optional
        Stop when the absolute objective error is at most this value.
    rtol : float, optional
        Stop when the relative objective error is at most this value.
    out_device : str or torch.device, optional
        Device where finalized cores are stored. The default is ``"cpu"``;
        ``None`` keeps them on the computation device.
    generator : torch.Generator, optional
        Generator used by random initialization and row sampling. Its device
        must match the source values.
    collect_metrics : bool
        Whether to record local solves, complete-sweep errors and timings.
    verbose : bool or int
        Console verbosity from 0 (silent) to 3 (most detailed).
    mask : BaseSource or torch.Tensor, optional
        Boolean tensor of participating entries. ``True`` participates;
        ``False`` is missing. A sparse mask fixes its stored true entries.
    weights : torch.Tensor, optional
        Non-negative multiplicative residual weights with shape ``in_dim``
        of the scalar problem; matrix input/output pairs are fused.

    Returns
    -------
    TTDecomposition
        Approximation with cores, metadata and optional metrics. Matrix and
        Quantics sources select the corresponding result family.

    Examples
    --------
    >>> tensor = torch.ones(2, 3, 2, dtype=torch.float64)
    >>> result = tk.decompositions.tt_als(tensor, rank=1)
    >>> torch.allclose(result.contract_dense(), tensor)
    True
    """
    return TTALS(source, mask=mask, weights=weights, out_device=out_device
    ).fit(
        rank, initial_cores, fixed_cores, gauge, renormalize,
        convergence=ConvergencePolicy(
            max_sweeps=max_sweeps, atol=atol, rtol=rtol),
        generator=generator, collect_metrics=collect_metrics, verbose=verbose)


__all__ = [
    'TTALS',
    'tt_als',
]
