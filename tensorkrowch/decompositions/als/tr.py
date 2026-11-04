"""
This script contains:

    Classes:
        * _TRALSBackend
        * _TRLeverageALSBackend
        * TRALS

    Functions:
        * _standard_tr_cores
        * _contract_standard_tr
        * _evaluate_standard_tr
        * _tr_gauge_core_update
        * _normalize_tr_core_update
        * tr_als
"""

from typing import Mapping, Optional, Sequence, TYPE_CHECKING, Tuple, Union

import torch

from tensorkrowch.formats.formats1d import TensorFormat1D

from tensorkrowch.decompositions._utils import (_normalize_rank,
                                                _unravel_indices)
from tensorkrowch.decompositions.als.convergence import ConvergencePolicy
from tensorkrowch.decompositions.als.driver import ALSBackend
from tensorkrowch.decompositions.als.environments import (
    _updated_cores,
    DirectTREnvironment,
    TRSegmentEnvironmentCache,
    _environment_norm,
)
from tensorkrowch.decompositions.als.gauges import (_absorb_factor,
                                                    _factor_core)
from tensorkrowch.decompositions.als.problem import ALSProblem
from tensorkrowch.decompositions.als.sampling import (RowSampler,
                                                      SamplingPolicy,
                                                      TRExactLeverageRows,
                                                      TRProductLeverageRows,
                                                      )
from tensorkrowch.decompositions.als.solvers import (LeastSquaresSolver,
                                                     _solve_local_proposal)
from tensorkrowch.decompositions.als.tt import TTALS
from tensorkrowch.decompositions.metrics import LocalSolveRecord
from tensorkrowch.decompositions.results import TRDecomposition
from tensorkrowch.decompositions.sources import (BaseSource,
                                                 ConfigurationBatch)


_Rank = Optional[Union[int, Sequence[int]]]


if TYPE_CHECKING:
    from tensorkrowch.decompositions.sources.factory import SourceLike


def _standard_tr_cores(cores: Union[TRDecomposition, Sequence[torch.Tensor]],
                       in_dim: Sequence[int]) -> Tuple[torch.Tensor, ...]:
    """Normalizes lightweight TR ``cores`` to standard cyclic shapes."""
    if isinstance(cores, TensorFormat1D):
        if not cores._cyclic:
            raise ValueError('Initial cores should have cyclic topology')
        if cores.n_batches:
            raise ValueError('Batched TR cores are not supported by TR-ALS')
        cores = cores._effective_cores()
    if isinstance(cores, (str, bytes)) or \
            not isinstance(cores, Sequence):
        raise TypeError(
            '`initial_cores` should be a TRDecomposition or a core sequence')
    cores = tuple(cores)
    if len(cores) != len(in_dim):
        raise ValueError('`initial_cores` should contain one core per site')
    if not all(isinstance(core, torch.Tensor) for core in cores):
        raise TypeError('`initial_cores` should contain torch.Tensor objects')
    for core, site_in_dim in zip(cores, in_dim):
        if core.ndim != 3:
            raise ValueError(
                'TR cores should have left rank, input and right rank '
                'dimensions')
        if core.shape[1] != site_in_dim:
            raise ValueError(
                '`initial_cores` input dimensions should match the source')
    return TRSegmentEnvironmentCache._validate_cores(cores)


def _contract_standard_tr(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Contracts standard cyclic ``cores`` without constructing a model graph.
    """
    result = cores[0]
    for core in cores[1:]:
        result = torch.einsum('a...b,bpc->a...pc', result, core)
    return result.diagonal(dim1=0, dim2=-1).sum(-1)


def _evaluate_standard_tr(cores: Sequence[torch.Tensor],
                          indices: torch.Tensor) -> torch.Tensor:
    """Evaluates standard cyclic ``cores`` at packed configurations."""
    selected = cores[0][:, indices[:, 0], :].permute(1, 0, 2)
    environment = selected
    for site, core in enumerate(cores[1:], 1):
        selected = core[:, indices[:, site], :].permute(1, 0, 2)
        environment = torch.bmm(environment, selected)
    return environment.diagonal(dim1=-2, dim2=-1).sum(-1)


def _tr_gauge_core_update(cores: Sequence[torch.Tensor],
                          fixed_sites: Sequence[int],
                          gauge: str,
                          direction: str,
                          site: int,
                          proposal: torch.Tensor
                          ) -> Mapping[int, torch.Tensor]:
    """Builds an atomic cyclic current-plus-receiver ``gauge`` update."""
    n_sites = len(cores)
    receiver = (site + 1) % n_sites if direction == 'forward' else \
        (site - 1) % n_sites
    legal_receiver = (receiver != site) and (receiver not in fixed_sites)
    if direction == 'forward':
        feasible = proposal.shape[0] * proposal.shape[1] >= proposal.shape[2]
    else:
        feasible = proposal.shape[1] * proposal.shape[2] >= proposal.shape[0]
    if (gauge == 'qr') and legal_receiver and feasible:
        gauged_core, factor = _factor_core(proposal, direction)
    else:
        gauged_core, factor = proposal, None

    replacements = {site: gauged_core}
    if factor is not None:
        replacements[receiver] = _absorb_factor(
            factor, cores[receiver], direction)
    return replacements


def _normalize_tr_core_update(replacements: Mapping[int, torch.Tensor],
                              cores: Sequence[torch.Tensor],
                              fixed_sites: Sequence[int], direction: str,
                              site: int) -> Mapping[int, torch.Tensor]:
    """Moves a square-root norm to the next trainable cyclic core."""
    current = replacements[site]
    norm = torch.linalg.vector_norm(current)
    if (not torch.isfinite(norm)) or (norm == 0):
        return replacements
    scale = norm.sqrt()
    step = 1 if direction == 'forward' else -1
    receiver = next((candidate for offset in range(1, len(cores))
                     if (candidate := (site + step * offset) % len(cores))
                     not in fixed_sites), None)
    if receiver is None:
        return replacements
    updates = dict(replacements)
    updates[site] = current / scale.to(current.dtype)
    core = updates.get(receiver, cores[receiver])
    updates[receiver] = core * scale.to(core.dtype)
    return updates


class _TRALSBackend:  # MARK: _TRALSBackend
    """Adapts exact, sampled and completion TR solves to the common driver."""

    def __init__(self,
                 problem: ALSProblem,
                 cores: Sequence[torch.Tensor],
                 target: Optional[torch.Tensor],
                 solver: LeastSquaresSolver,
                 gauge: str,
                 fixed_sites: Sequence[int],
                 normalize: bool,
                 renormalize: bool,
                 n_segments: Optional[int],
                 sampler: Optional[RowSampler] = None,
                 n_samples: Optional[int] = None,
                 sampling_policy: Optional[SamplingPolicy] = None,
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
        self.sampling_policy = sampling_policy
        self.generator = generator
        self.sample_batch = None
        self.current_target = self.full_target
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

        batch, refreshed = self.sampling_policy.sample(
            sampler=self.sampler, owner=self.cache, site=order[0],
            n_samples=self.n_samples, generator=self.generator,
            sweep=sweep, current=self.sample_batch)
        if refreshed:
            self.cache.invalidate('sample generation changed')
        if refreshed:
            if self.problem.observations is not None:
                self.current_target = self.problem.observations.support_values
            else:
                indices = _unravel_indices(batch.ids, self.cache.in_dim)
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

    def update_site(self,
                    site: int,
                    sweep: int,
                    return_record: bool) -> Optional[LocalSolveRecord]:
        """Builds one local proposal and its atomic gauge update."""
        environment = self.cache.design(site)
        indices = (None if (self.sample_batch is None) or
                   (self.problem.observations is not None) else
                   _unravel_indices(self.sample_batch.ids,
                                    self.problem.in_dim))
        proposal, record = _solve_local_proposal(
            solver=self.solver, environment=environment,
            target=self.current_target, current=self.cache.cores[site],
            site=site, sweep=sweep, return_record=return_record,
            problem=self.problem, log_scale=self.cache.log_scale,
            indices=indices,
            samples=self.sample_batch, core_versions=self.cache.core_versions)
        replacements = _tr_gauge_core_update(
            cores=self.cache.cores,
            fixed_sites=self.fixed_sites,
            gauge=self.gauge,
            direction=self._direction,
            site=site,
            proposal=proposal)
        if self.normalize:
            replacements = _normalize_tr_core_update(
                replacements=replacements,
                cores=self.cache.cores,
                fixed_sites=self.fixed_sites,
                direction=self._direction,
                site=site)
        self._update(replacements)
        return record

    def skip_site(self, site: int) -> None:
        """Advances the cache through one frozen site."""
        self.cache.design(site)
        self.cache.update({})

    def _update(self, replacements: Mapping[int, torch.Tensor]) -> None:
        """Applies one complete local update to its environment owner."""
        self.cache.update(replacements)

    def measure_objective(self,
                          problem: ALSProblem) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """Measures the fixed complete-sweep objective when one exists."""
        if problem.observations is not None:
            indices = problem.observations.support.values.to(
                self.cache.cores[0].device)
            approximation = _evaluate_standard_tr(self.cache.cores, indices)
            return problem.objective_error(approximation)

        approximation = _contract_standard_tr(self.cache.cores).reshape(-1)
        return problem._objective_error(approximation, self.full_target)

    def snapshot(self) -> Sequence[torch.Tensor]:
        """Returns the cores forming the current approximation."""
        return tuple(core.clone() for core in self.cache.cores)

    def restore(self, cores: Sequence[torch.Tensor]) -> None:
        """Restores a saved approximation and refreshes its dependent state."""
        self.cache = TRSegmentEnvironmentCache(
            cores,
            n_segments=len(self.cache.segments),
            renormalize=self.cache.renormalize)


class _TRLeverageALSBackend:  # MARK: _TRLeverageALSBackend
    """Runs product or exact TR leverage sampling by active fibers."""

    def __init__(self,
                 problem: ALSProblem,
                 cores: Sequence[torch.Tensor],
                 solver: LeastSquaresSolver,
                 gauge: str,
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
        self._direction = None
        self._generation = None

    @property
    def core_versions(self) -> Tuple[int, ...]:
        """Versions read by each proposal at draw time."""
        return self._versions

    @property
    def n_sites(self) -> int:
        """Number of cores in this backend."""
        return len(self._cores)

    @property
    def trainable_sites(self) -> Sequence[int]:
        """Sites that may receive local updates."""
        return tuple(site for site in range(self.n_sites)
                     if site not in self.fixed_sites)

    @property
    def cores(self) -> Sequence[torch.Tensor]:
        """Current cores in original site order."""
        return self._cores

    def prepare_sweep(self,
                      order: Sequence[int],
                      sweep: int) -> Tuple[Optional[int], bool]:
        """
        Prepares directional environments and the current sampling
        generation.
        """
        self._direction = 'forward' if order[0] == 0 else 'reverse'
        self._generation = sweep
        return sweep, True

    def update_site(self,
                    site: int,
                    sweep: int,
                    return_record: bool) -> Optional[LocalSolveRecord]:
        """Builds one local proposal and its atomic gauge update."""
        self.sampler.generation = self._generation
        batch = self.sampler.draw(self, site, self.n_samples, self.generator)
        indices = _unravel_indices(
            batch.ids, tuple(core.shape[1] for core in self._cores))
        in_dim = self._cores[site].shape[1]
        base_indices = indices[::in_dim].clone()
        target = self.problem.source.fiber(
            ConfigurationBatch(base_indices, kind='indices'), site).reshape(-1)
        if target.shape != (batch.ids.numel(),):
            raise ValueError(
                'TR-ALS currently requires a scalar tensor source')
        if not torch.isfinite(target).all():
            raise ValueError(
                'The tensor source should return only finite values')

        environment = DirectTREnvironment(self._cores).design(
            site, samples=batch)
        log_scale = environment.real.new_zeros(())
        if self.renormalize:
            norm = _environment_norm(environment)
            if norm > 0:
                environment = environment / norm
                log_scale = norm.log()
        proposal, record = _solve_local_proposal(
            solver=self.solver, environment=environment, target=target,
            current=self._cores[site], site=site, sweep=sweep,
            return_record=return_record, problem=self.problem,
            log_scale=log_scale, indices=indices, samples=batch,
            core_versions=self._versions)
        replacements = _tr_gauge_core_update(
            cores=self._cores,
            fixed_sites=self.fixed_sites,
            gauge=self.gauge,
            direction=self._direction,
            site=site,
            proposal=proposal)
        if self.normalize:
            replacements = _normalize_tr_core_update(
                replacements=replacements,
                cores=self._cores,
                fixed_sites=self.fixed_sites,
                direction=self._direction,
                site=site)
        self._update(replacements)
        return record

    def skip_site(self, site: int) -> None:
        """Advances the sweep through a fixed core."""
        return None

    def _update(self, replacements: Mapping[int, torch.Tensor]) -> None:
        """Validates and applies the solved core and any gauge receiver."""
        self._cores, self._versions = _updated_cores(
            self._cores, self._versions, replacements,
            TRSegmentEnvironmentCache._validate_cores)

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
        self._cores = TRSegmentEnvironmentCache._validate_cores(cores)


class TRALS(TTALS):  # MARK: TRALS
    """
    Approximates a fixed finite tensor by one-site TR-ALS.

    Construction fixes the source, its dimensions and the weighted objective.
    Repeated :meth:`fit` calls can use different ranks, initial cores, samplers
    and solvers. Exact fits cache all target values on first use; sampled fits
    query only their selected configurations. A sparse boolean mask defines
    permanent observations, including observed values that are zero.

    Vector, matrix and Quantics sources use the same scalar local solves;
    the result restores the source's paired dimensions and coordinate metadata.
    Continuous inputs need discretization before ALS. Structural batches are
    not supported. Sweeps visit sites in increasing order around the ring;
    each complete pass counts once toward ``max_sweeps``.

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
    >>> engine = tk.decompositions.TRALS(torch.ones(2, 3))
    >>> result = engine.fit(rank=1)
    >>> tuple(result.contract_dense().shape)
    (2, 3)
    """

    _topology = 'tr'
    _result_type = TRDecomposition

    def _random_cores(self,
                      ranks: Sequence[int],
                      dtype: torch.dtype,
                      device: torch.device,
                      generator: Optional[torch.Generator]) -> Tuple[
            torch.Tensor, ...]:
        """
        Initializes random cyclic cores with prescribed right-link ``ranks``.
        """
        cores = []
        for site, site_in_dim in enumerate(self.in_dim):
            core = torch.randn(
                ranks[site - 1],
                site_in_dim,
                ranks[site],
                dtype=dtype,
                device=device,
                generator=generator)
            core = core / max(1, ranks[site - 1] * site_in_dim) ** 0.5
            cores.append(core)
        return tuple(cores)

    def _initial_cores(self,
                       target: torch.Tensor,
                       initial_cores: Optional[Union[
                           TensorFormat1D, Sequence[torch.Tensor]]],
                       rank: _Rank,
                       fixed_cores: Optional[
                           Sequence[Optional[torch.Tensor]]],
                       generator: Optional[
                           torch.Generator]) -> Tuple[
            Tuple[torch.Tensor, ...],
            Tuple[int, ...]]:
        """Builds and validates initialization plus fixed-site semantics."""
        rank_caps = None if rank is None else _normalize_rank(
            rank, len(self.in_dim))
        if initial_cores is None:
            if rank_caps is None:
                raise ValueError('`rank` is required without `initial_cores`')
            cores = self._random_cores(
                rank_caps, target.dtype, target.device, generator)
        else:
            cores = _standard_tr_cores(initial_cores, self.in_dim)

        if any((core.device != target.device) or (core.dtype != target.dtype)
               for core in cores):
            raise ValueError(
                'Initial cores and source values should share dtype and '
                'device')
        current_ranks = tuple(core.shape[-1] for core in cores)
        if (rank_caps is not None) and any(
                current > cap
                for current, cap in zip(current_ranks, rank_caps)):
            raise ValueError(
                'Initial TR ranks should not exceed the requested `rank` caps')

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
            if fixed_core.shape != cores[site].shape:
                raise ValueError(
                    'Every fixed core should match its initialized core shape')
            if (fixed_core.device != target.device) or \
                    (fixed_core.dtype != target.dtype):
                raise ValueError(
                    'Fixed cores and source values should share runtime')
            if not torch.isfinite(fixed_core).all():
                raise ValueError(
                    '`fixed_cores` should contain only finite values')
            final_cores[site] = fixed_core
            fixed_sites.append(site)
        final_cores = TRSegmentEnvironmentCache._validate_cores(final_cores)
        return final_cores, tuple(fixed_sites)

    def _make_backend(self, problem: ALSProblem,
                      cores: Sequence[torch.Tensor],
                      target: Optional[torch.Tensor],
                      solver: LeastSquaresSolver, gauge: str,
                      fixed_sites: Sequence[int], renormalize: bool,
                      row_sampler: Optional[RowSampler],
                      n_samples: Optional[int],
                      sampling_policy: SamplingPolicy,
                      generator: Optional[torch.Generator],
                      sampler: Optional[SamplingPolicy]) -> ALSBackend:
        """Builds the topology-specific full-row or leverage backend."""
        if (sampler is not None) and (sampler.method in (
                'leverage', 'exact_leverage')):
            return _TRLeverageALSBackend(
                problem, cores, solver, gauge, fixed_sites, n_samples,
                'product' if sampler.method == 'leverage' else 'exact',
                sampler.uniform_mix, generator, True, renormalize)
        return _TRALSBackend(
            problem, cores, target, solver, gauge, fixed_sites,
            True, renormalize, None, row_sampler, n_samples,
            sampling_policy, generator)

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
            verbose: Union[bool, int] = 0) -> 'TRDecomposition':
        """
        Fits a TR by alternating one-site least-squares solves.

        Sweeps visit sites in increasing order around the ring. Each complete
        pass counts once toward ``max_sweeps``.
        Without ``initial_cores``, initializes random cores at the requested
        ranks. To start from an SVD approximation, compute it separately and
        pass its result as ``initial_cores``. Supplied cores are copied.

        Parameters
        ----------
        rank : int or sequence of int, optional
            One upper bound per right link, or a shared bound. The last link
            closes the ring. Required without ``initial_cores``.
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
        TRDecomposition
            Approximation with cores, metadata and optional metrics. Matrix and
            Quantics sources select the corresponding result family.

        Examples
        --------
        >>> tensor = torch.ones(2, 3, 2, dtype=torch.float64)
        >>> result = tk.decompositions.TRALS(tensor).fit(rank=1)
        >>> torch.allclose(result.contract_dense(), tensor)
        True
        """
        return super().fit(
            rank=rank, initial_cores=initial_cores, fixed_cores=fixed_cores,
            gauge=gauge, renormalize=renormalize, sampler=sampler,
            solver=solver, convergence=convergence, generator=generator,
            collect_metrics=collect_metrics, verbose=verbose)

def tr_als(source: 'SourceLike',
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
           weights: Optional[torch.Tensor] = None) -> 'TRDecomposition':
    """
    Approximates a finite tensor with one-site TR-ALS.

    Returns a decomposition object. Use :class:`TRALS` and its :meth:`fit
    <TRALS.fit>` method for sampled fits, customized local solvers or advanced
    convergence criteria. Sweeps visit sites in increasing order around the
    ring; each complete pass counts once toward ``max_sweeps``.

    Parameters
    ----------
    source : BaseSource, TensorFormat1D, MPS, MPO or torch.Tensor
        Finite tensor input, with shape and evaluation defined by the source.
    rank : int or sequence of int, optional
        One upper bound per right link, or a shared bound. The last link
        closes the ring. Required without ``initial_cores``.
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
    TRDecomposition
        Approximation with cores, metadata and optional metrics. Matrix and
        Quantics sources select the corresponding result family.

    Examples
    --------
    >>> tensor = torch.ones(2, 3, 2, dtype=torch.float64)
    >>> result = tk.decompositions.tr_als(tensor, rank=1)
    >>> torch.allclose(result.contract_dense(), tensor)
    True
    """
    return TRALS(source, mask=mask, weights=weights, out_device=out_device
    ).fit(
        rank, initial_cores, fixed_cores, gauge, renormalize,
        convergence=ConvergencePolicy(
            max_sweeps=max_sweeps, atol=atol, rtol=rtol),
        generator=generator, collect_metrics=collect_metrics, verbose=verbose)


__all__ = [
    'TRALS',
    'tr_als',
]
