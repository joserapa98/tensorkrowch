"""
This script contains:

    Classes:
        * _TRALSBackend
        * _TRSampledALSBackend
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
from tensorkrowch.decompositions.als.sampling import (ObservedRows,
                                                      SamplingPolicy,
                                                      TRExactLeverageRows,
                                                      TRProductLeverageRows,
                                                      UniformRows,
                                                      _evaluate_fibers,
                                                      )
from tensorkrowch.decompositions.als.solvers import (LeastSquaresSolver,
                                                     _solve_local_proposal)
from tensorkrowch.decompositions.als.tt import TTALS
from tensorkrowch.decompositions.metrics import LocalSolveRecord
from tensorkrowch.decompositions.results import TRDecomposition
from tensorkrowch.decompositions.sources import BaseSource


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
                          orthogonalize: bool,
                          direction: str,
                          site: int,
                          proposal: torch.Tensor
                          ) -> Mapping[int, torch.Tensor]:
    """Builds an atomic cyclic current-plus-receiver gauge update."""
    n_sites = len(cores)
    receiver = (site + 1) % n_sites if direction == 'forward' else \
        (site - 1) % n_sites
    legal_receiver = (receiver != site) and (receiver not in fixed_sites)
    if direction == 'forward':
        feasible = proposal.shape[0] * proposal.shape[1] >= proposal.shape[2]
    else:
        feasible = proposal.shape[1] * proposal.shape[2] >= proposal.shape[0]
    if orthogonalize and legal_receiver and feasible:
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
                 orthogonalize: bool,
                 fixed_sites: Sequence[int],
                 normalize: bool,
                 renormalize: bool,
                 n_segments: Optional[int]) -> None:
        self.problem = problem
        self.full_target = None if target is None else target.reshape(-1)
        self.solver = solver
        self.orthogonalize = orthogonalize
        self.fixed_sites = frozenset(fixed_sites)
        self.normalize = normalize
        self.cache = TRSegmentEnvironmentCache(
            cores, n_segments=n_segments, renormalize=renormalize)
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
        if self.problem.observations is None:
            self.cache.prepare_sweep(order)
            self.current_target = self.full_target
            return None, False

        refreshed = self.sample_batch is None
        if refreshed:
            self.sample_batch = ObservedRows(self.problem.observations).draw(
                self.cache, order[0], None)
            self.current_target = self.problem.observations.support_values
        self.cache.prepare_sweep(order, samples=self.sample_batch)
        return self.sample_batch.generation, refreshed

    def update_site(self,
                    site: int,
                    sweep: int,
                    return_record: bool) -> Optional[LocalSolveRecord]:
        """Builds one local proposal and its atomic gauge update."""
        environment = self.cache.design(site)
        proposal, record = _solve_local_proposal(
            solver=self.solver, environment=environment,
            target=self.current_target, current=self.cache.cores[site],
            site=site, sweep=sweep, return_record=return_record,
            problem=self.problem, log_scale=self.cache.log_scale,
            indices=None,
            samples=self.sample_batch, core_versions=self.cache.core_versions)
        replacements = _tr_gauge_core_update(
            cores=self.cache.cores,
            fixed_sites=self.fixed_sites,
            orthogonalize=self.orthogonalize,
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
        """Measures the complete-sweep objective on its fixed entries."""
        if problem.sampled:
            approximation = _evaluate_standard_tr(
                self.cores, problem._evaluation_indices)
            return problem.evaluation_error(approximation)
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


class _TRSampledALSBackend:  # MARK: _TRSampledALSBackend
    """Runs uniform, product or exact TR sampling by active fibers."""

    def __init__(self,
                 problem: ALSProblem,
                 cores: Sequence[torch.Tensor],
                 solver: LeastSquaresSolver,
                 orthogonalize: bool,
                 fixed_sites: Sequence[int],
                 sampling_policy: SamplingPolicy,
                 generator: Optional[torch.Generator],
                 normalize: bool,
                 renormalize: bool) -> None:
        self.problem = problem
        self._cores = TRSegmentEnvironmentCache._validate_cores(cores)
        self._versions = (0,) * len(self._cores)
        self.solver = solver
        self.orthogonalize = orthogonalize
        self.fixed_sites = frozenset(fixed_sites)
        self.sampling_policy = sampling_policy
        self.n_samples = sampling_policy.n_samples
        self.generator = generator
        self.normalize = normalize
        self.renormalize = renormalize
        if sampling_policy.method == 'uniform':
            self.sampler = UniformRows()
        elif sampling_policy.method == 'product_leverage':
            self.sampler = TRProductLeverageRows(
                lambda: self._cores, uniform_mix=sampling_policy.uniform_mix)
        else:
            self.sampler = TRExactLeverageRows(
                lambda: self._cores, uniform_mix=sampling_policy.uniform_mix)
        self._direction = None
        self._generation = None
        self._sweep = None

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
        """Prepares the current uniform or leverage sample generation."""
        generation = self.sampling_policy.generation(sweep)
        refreshed = (self.sampling_policy.n_reuse_sweeps == 0) or (
            generation != self._generation)
        self._direction = 'forward' if order[0] == 0 else 'reverse'
        self._generation = generation
        self._sweep = sweep
        return generation, refreshed

    def update_site(self,
                    site: int,
                    sweep: int,
                    return_record: bool) -> Optional[LocalSolveRecord]:
        """Builds one local proposal and its atomic gauge update."""
        if isinstance(self.sampler, UniformRows):
            batch, _ = self.sampling_policy.sample(
                self.sampler, self, site, self.n_samples, self.generator,
                self._sweep)
            target = self.sampler.evaluate(self.problem.source, batch)
        else:
            self.sampler.generation = self._generation
            batch = self.sampler.draw(
                self, site, self.n_samples, self.generator)
            target = _evaluate_fibers(self.problem.source, batch)
        indices = _unravel_indices(
            batch.ids, tuple(core.shape[1] for core in self._cores))

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
            orthogonalize=self.orthogonalize,
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
        """Measures the complete-sweep objective on its fixed entries."""
        if problem.sampled:
            approximation = _evaluate_standard_tr(
                self.cores, problem._evaluation_indices)
            return problem.evaluation_error(approximation)
        raise RuntimeError('Sampled sweeps require prepared evaluation')

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
                      solver: LeastSquaresSolver, orthogonalize: bool,
                      fixed_sites: Sequence[int], renormalize: bool,
                      generator: Optional[torch.Generator],
                      sampler: Optional[SamplingPolicy]) -> ALSBackend:
        """Builds the topology-specific full-row or sampled backend."""
        if sampler is not None:
            return _TRSampledALSBackend(
                problem, cores, solver, orthogonalize, fixed_sites, sampler,
                generator, True, renormalize)
        return _TRALSBackend(
            problem, cores, target, solver, orthogonalize, fixed_sites,
            True, renormalize, None)

    def fit(self,
            rank: Optional[Union[int, Sequence[int]]] = None,
            initial_cores: Optional[Union[TensorFormat1D,
                Sequence[torch.Tensor]]] = None,
            fixed_cores: Optional[Sequence[Optional[torch.Tensor]]] = None,
            orthogonalize: bool = True,
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
        pass counts once toward ``max_sweeps``. QR orthogonalization is local;
        it does not establish a global canonical form for the ring.
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
        orthogonalize : bool
            Whether QR moves a factor into the immediate trainable neighbor
            after each local solve. Defaults to ``True``. ``False`` replaces
            cores directly. No factor is discarded if the receiver is
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
            Complete-sweep stopping criteria and independent error-evaluation
            budget. Defaults to ten sweeps. Sampled fits cache fixed uniform
            evaluation entries and measure the current approximation on them
            after every sweep.
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
            orthogonalize=orthogonalize, renormalize=renormalize,
            sampler=sampler,
            solver=solver, convergence=convergence, generator=generator,
            collect_metrics=collect_metrics, verbose=verbose)

def tr_als(source: 'SourceLike',
           rank: Optional[Union[int, Sequence[int]]] = None,
           initial_cores: Optional[Union[TensorFormat1D,
               Sequence[torch.Tensor]]] = None,
           fixed_cores: Optional[Sequence[Optional[torch.Tensor]]] = None,
           orthogonalize: bool = True,
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
    orthogonalize : bool
        Whether QR moves a factor into the immediate trainable neighbor
        after each local solve. Defaults to ``True``. ``False`` replaces
        cores directly. No factor is discarded if the receiver is
        unavailable.
    renormalize : bool
        Whether cached environments remove global norms and retain their
        logarithmic scales. The local least-squares objective is preserved.
    max_sweeps : int
        Positive maximum number of directional sweeps. Default is ``10``.
    atol : float, optional
        Stop when the objective residual norm is at most this value.
        This is a norm tolerance, not a squared-error tolerance.
    rtol : float, optional
        Stop when the objective residual norm divided by the target norm
        is at most this value. Both norms use the objective's weights.
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
        rank, initial_cores, fixed_cores, orthogonalize, renormalize,
        convergence=ConvergencePolicy(
            max_sweeps=max_sweeps, atol=atol, rtol=rtol),
        generator=generator, collect_metrics=collect_metrics, verbose=verbose)


__all__ = [
    'TRALS',
    'tr_als',
]
