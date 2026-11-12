"""
This script contains:

    Classes:
        * SamplingPolicy
        * SampleBatch
        * RowSampler
        * ExactRows
        * ObservedRows
        * UniformRows
        * TTLeverageRows
        * TRProductLeverageRows
        * TRExactLeverageRows

    Functions:
        * _sample_batch
        * _expand_fibers
        * _evaluate_fibers
        * _region_metrics
        * _sample_region
        * _region_row_probability
        * _sampling_cores
"""

from dataclasses import dataclass
from math import isfinite, prod
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from tensorkrowch.utils import _INTEGER_DTYPES

from tensorkrowch.decompositions._utils import _ravel_indices, _unravel_indices
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 SparseTensorSource)
from tensorkrowch.decompositions.sources.base import _discrete_indices


@dataclass(frozen=True)
class SamplingPolicy:  # MARK: SamplingPolicy
    """
    Configures row sampling for :class:`TTALS` and :class:`TRALS`.

    Omit the engine's ``sampler`` argument to fit the complete tensor or its
    fixed observations. Sampled fits select external configurations and retain
    the complete active input fiber. Leverage distributions are rebuilt at
    each active site.

    Parameters
    ----------
    n_samples : int
        Positive number of sampled external configurations. Each local system
        contains ``n_samples * in_dim[site]`` scalar rows.
    method : {``"uniform"``, ``"leverage"``, ``"product_leverage"``}, optional
        Uniform sampling is the default. ``"leverage"`` uses exact leverage
        in mixed-canonical TT form or the cyclic Gram construction for TR.
        ``"product_leverage"`` uses a product-leverage proposal for TR only.
    n_reuse_sweeps : int, optional
        Number of directional sweeps sharing a uniform sample generation.
        Defaults to ``0``, drawing new external configurations at every site.
        A positive value draws complete base configurations, opens the active
        fiber at each site and caches its target values for that generation.
        ``1`` shares the base configurations within one sweep; larger values
        also reuse targets across consecutive sweeps. Leverage requires ``0``.
    uniform_mix : float, optional
        Uniform component mixed into a leverage proposal, in ``[0, 1]``.
        Defaults to ``0``. Uniform sampling requires ``0``.
    zipup : bool, optional
        Whether to contract each core before selecting its sampled physical
        slices. Defaults to ``False``, selecting slices before contraction.
        This changes contraction order only, not samples or probabilities.

    Examples
    --------
    >>> sampler = tk.decompositions.SamplingPolicy(64, method='leverage')
    >>> sampler.n_samples, sampler.n_reuse_sweeps
    (64, 0)
    """

    n_samples: int  # Number of external configurations per proposal
    method: str = 'uniform'  # Uniform or topology-specific leverage proposal
    n_reuse_sweeps: int = 0  # Zero draws at each site; positive shares a base
    uniform_mix: float = 0.0  # Uniform fraction in a leverage proposal
    zipup: bool = False  # Contract before selecting sampled physical slices

    def __post_init__(self) -> None:
        if not isinstance(self.zipup, bool):
            raise TypeError('`zipup` should be bool type')
        for name in ('n_samples', 'n_reuse_sweeps'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            minimum = 1 if name == 'n_samples' else 0
            if value < minimum:
                raise ValueError(f'`{name}` should be at least {minimum}')
        if not isinstance(self.method, str):
            raise TypeError('`method` should be str type')
        if self.method not in ('uniform', 'leverage', 'product_leverage'):
            raise ValueError(
                "`method` should be 'uniform', 'leverage' or "
                "'product_leverage'")
        if isinstance(self.uniform_mix, bool) or \
                not isinstance(self.uniform_mix, (int, float)):
            raise TypeError('`uniform_mix` should be a number in [0, 1]')
        if not isfinite(self.uniform_mix) or not (0 <= self.uniform_mix <= 1):
            raise ValueError('`uniform_mix` should be finite and in [0, 1]')
        object.__setattr__(self, 'uniform_mix', float(self.uniform_mix))
        if (self.method == 'uniform') and (self.uniform_mix != 0):
            raise ValueError('Uniform sampling requires `uniform_mix=0`')
        if (self.method != 'uniform') and (self.n_reuse_sweeps != 0):
            raise ValueError('Leverage sampling requires `n_reuse_sweeps=0`')

    def validate(self, topology: str) -> None:
        """
        Checks that the selected method supports the engine topology.

        Parameters
        ----------
        topology : {``"tt"``, ``"tr"``}
            Tensor topology of the receiving ALS engine.
        """
        if topology not in ('tt', 'tr'):
            raise ValueError("`topology` should be 'tt' or 'tr'")
        if (topology == 'tt') and (self.method == 'product_leverage'):
            raise ValueError(
                '`method="product_leverage"` requires TR topology')

    def generation(self, sweep: int) -> int:
        """Returns the draw generation for a non-negative sweep index."""
        if isinstance(sweep, bool) or not isinstance(sweep, int) or \
                (sweep < 0):
            raise ValueError('`sweep` should be a non-negative integer')
        return sweep // max(self.n_reuse_sweeps, 1)

    def sample(self, sampler: 'RowSampler', owner: Any, site: int,
               n_samples: Optional[int], generator: Optional[torch.Generator],
               sweep: int, current: Optional['SampleBatch'] = None
               ) -> Tuple['SampleBatch', bool]:
        """Reuses a frozen draw or starts the next generation."""
        generation = self.generation(sweep)
        if isinstance(sampler, UniformRows):
            return sampler.sample(
                owner, site, n_samples, generator, generation,
                reuse=self.n_reuse_sweeps > 0)
        if (current is not None) and (
                (not sampler.refreshable) or (
                    (self.n_reuse_sweeps > 0) and
                    (current.site == site) and
                    (current.generation == generation))):
            return current, False
        sampler.generation = generation
        return sampler.draw(owner, site, n_samples, generator), True


@dataclass(frozen=True)
class SampleBatch:  # MARK: SampleBatch
    """
    Stores selected ALS rows and their original sampling probabilities.

    Probabilities and weights remain unchanged while a batch is reused.

    Parameters
    ----------
    ids : torch.Tensor
        Integer global row identifiers with shape ``(n_rows,)``.
    probabilities : torch.Tensor
        Positive draw probabilities with shape ``(n_rows,)``.
    weights : torch.Tensor
        Multipliers ``1 / sqrt(n_rows * probabilities)`` for the selected
        matrix and target rows. Shares shape and device with ``probabilities``.
    generation : int
        Non-negative identifier of the sample generation.
    proposal_core_versions : tuple of int, optional
        Core versions used for a design-dependent proposal. Empty by default.
    proposal_exact : bool, optional
        Whether the proposal is exact for its recorded design. Defaults to
        ``True``; product-leverage proposals use ``False``.
    site : int, optional
        Active site associated with the proposal.
    """

    ids: torch.Tensor  # Global row identifiers drawn by the sampler
    probabilities: torch.Tensor  # Original draw probabilities
    weights: torch.Tensor  # Multiplicative residual weights in row order
    generation: int  # Sampling generation number
    # Core versions used to build the sampling proposal
    proposal_core_versions: Tuple[int, ...] = ()
    proposal_exact: bool = True  # Exact proposal for the recorded design
    site: Optional[int] = None  # Optional zero-based active site

    def __post_init__(self) -> None:
        if not isinstance(self.ids, torch.Tensor):
            raise TypeError('`ids` should be torch.Tensor type')
        if (self.ids.ndim != 1) or (self.ids.dtype not in _INTEGER_DTYPES):
            raise TypeError('`ids` should be a one-dimensional integer tensor')
        if self.ids.numel() < 1:
            raise ValueError('`ids` should contain at least one sampled row')
        for name in ('probabilities', 'weights'):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f'`{name}` should be torch.Tensor type')
            if value.shape != self.ids.shape:
                raise ValueError(f'`{name}` should contain one value per id')
            if value.device != self.ids.device:
                raise ValueError(f'`{name}` and `ids` should share a device')
            if (not value.is_floating_point()) or value.is_complex():
                raise TypeError(f'`{name}` should have a real floating dtype')
            if not torch.isfinite(value).all():
                raise ValueError(f'`{name}` should contain finite values')
            if torch.any(value <= 0):
                raise ValueError(f'`{name}` should contain positive values')
        if torch.any(self.ids < 0):
            raise ValueError('`ids` should be non-negative')

        expected_weights = (
            self.ids.numel() * self.probabilities).rsqrt()
        if not torch.allclose(
                self.weights,
                expected_weights,
                rtol=8 * torch.finfo(self.weights.dtype).eps,
                atol=0):
            raise ValueError(
                '`weights` should equal 1 / sqrt(n_samples * probabilities)')
        if isinstance(self.generation, bool) or \
                (not isinstance(self.generation, int)) or \
                (self.generation < 0):
            raise ValueError('`generation` should be a non-negative integer')
        versions = tuple(self.proposal_core_versions)
        if any(isinstance(version, bool) or
               (not isinstance(version, int)) or (version < 0)
               for version in versions):
            raise ValueError(
                '`proposal_core_versions` should contain non-negative '
                'integers')
        object.__setattr__(self, 'proposal_core_versions', versions)
        if not isinstance(self.proposal_exact, bool):
            raise TypeError('`proposal_exact` should be bool type')
        if self.site is not None:
            if isinstance(self.site, bool) or \
                    (not isinstance(self.site, int)) or (self.site < 0):
                raise ValueError('`site` should be a non-negative integer')

    def is_exact_for(self, current_core_versions: Sequence[int]) -> bool:
        """
        Reports whether the exact proposal still matches the core versions.

        Parameters
        ----------
        current_core_versions : sequence of int
            Current version of every core participating in the proposal.
        """
        if not self.proposal_exact:
            return False
        if not self.proposal_core_versions:
            return True
        return self.proposal_core_versions == tuple(current_core_versions)

    def gather_and_weight(self,
                          environment: torch.Tensor,
                          target: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """
        Selects global rows and applies their sampling weights.

        Parameters
        ----------
        environment : torch.Tensor
            Complete design matrix with shape ``(n_rows, n_parameters)``.
        target : torch.Tensor
            Complete target with shape ``(n_rows,)`` or
            ``(n_rows, n_targets)``.

        Returns
        -------
        tuple of torch.Tensor
            Selected, weighted design and target rows, in ``ids`` order.
        """
        if not isinstance(environment, torch.Tensor):
            raise TypeError('`environment` should be torch.Tensor type')
        if not isinstance(target, torch.Tensor):
            raise TypeError('`target` should be torch.Tensor type')
        if environment.ndim != 2:
            raise ValueError('`environment` should be a matrix')
        if target.ndim not in (1, 2):
            raise ValueError('`target` should be a vector or matrix')
        if target.shape[0] != environment.shape[0]:
            raise ValueError(
                '`target` and `environment` should have matching rows')
        if (environment.device != self.ids.device) or \
                (target.device != self.ids.device):
            raise ValueError('Sampled tensors and ids should share a device')
        if self.ids.max() >= environment.shape[0]:
            raise ValueError('A sampled row id is out of bounds')

        sampled_environment = environment.index_select(0, self.ids)
        sampled_target = target.index_select(0, self.ids)
        environment_weights = self.weights.to(
            sampled_environment.dtype).unsqueeze(1)
        sampled_environment = sampled_environment * environment_weights
        if sampled_target.ndim == 1:
            sampled_target = sampled_target * self.weights.to(
                sampled_target.dtype)
        else:
            sampled_target = sampled_target * self.weights.to(
                sampled_target.dtype).unsqueeze(1)
        return sampled_environment, sampled_target


class RowSampler:  # MARK: RowSampler
    """
    Draws rows using current cores and versions owned by an ALS backend.

    ``draw(owner, site, n_samples, generator)`` reads ``owner.cores`` and
    ``owner.core_versions``. Draw probabilities and their version snapshot are
    retained independently in ``SampleBatch``.
    """

    generation = 0  # Identifier assigned to the next draw
    proposal_exact = True  # Exact proposal for the declared design
    refreshable = True  # Permanent observations disable resampling

    def _prepare_draw(self, owner: Any, site: int,
                      n_samples: Optional[int]) -> None:
        """Validates the draw against its current core owner once."""
        cores = owner.cores
        self.core_versions = tuple(owner.core_versions)
        self.n_rows = prod(core.shape[1] for core in cores)
        self.device = cores[0].device
        if len(self.core_versions) != len(cores):
            raise ValueError('Core versions should match the current cores')
        if isinstance(site, bool) or not isinstance(site, int) or \
                (site < 0) or (site >= len(cores)):
            raise ValueError('`site` should identify a core')
        if (n_samples is not None) and (
                isinstance(n_samples, bool) or not isinstance(n_samples, int)
                or (n_samples < 1)):
            raise ValueError('`n_samples` should be positive or None')

    def draw(self, owner: Any, site: int, n_samples: Optional[int],
             generator: Optional[torch.Generator] = None) -> 'SampleBatch':
        """Returns one independent draw from the configured proposal."""
        raise NotImplementedError


def _sample_batch(ids: torch.Tensor,
                  probabilities: torch.Tensor,
                  sampler: RowSampler,
                  site: int,
                  proposal_core_versions: Tuple[int, ...] = (),
                  proposal_exact: bool = True) -> SampleBatch:
    """Builds weights from immutable draw ``probabilities``."""
    weights = (ids.numel() * probabilities).rsqrt()
    return SampleBatch(
        ids=ids,
        probabilities=probabilities,
        weights=weights,
        generation=sampler.generation,
        proposal_core_versions=proposal_core_versions,
        proposal_exact=proposal_exact,
        site=site)


def _expand_fibers(indices: torch.Tensor, site: int,
                    in_dim: Sequence[int]) -> torch.Tensor:
    """Expands external configurations in environment-major physical order."""
    indices = indices.unsqueeze(1).expand(
        -1, in_dim[site], len(in_dim)).clone()
    indices[:, :, site] = torch.arange(
        in_dim[site], device=indices.device).unsqueeze(0)
    return indices.reshape(-1, len(in_dim))


def _evaluate_fibers(source: Any, batch: SampleBatch) -> torch.Tensor:
    """Evaluates complete fibers through the source's specialized access."""
    site = batch.site
    in_dim = source.in_dim
    indices = _unravel_indices(batch.ids, in_dim)[::in_dim[site]]
    target = source.fiber(ConfigurationBatch(indices, kind='indices'), site)
    if target.shape != (indices.shape[0], in_dim[site]):
        raise ValueError('`source` should return one scalar per configuration')
    if not torch.isfinite(target).all():
        raise ValueError('`source` should return only finite values')
    return target.reshape(-1)


class ExactRows(RowSampler):  # MARK: ExactRows
    """Deterministic sampler containing every row exactly once."""

    proposal_exact = True
    refreshable = False

    def draw(self,
             owner: Any,
             site: int,
             n_samples: Optional[int] = None,
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """
        Returns all rows with unit effective weights.

        Parameters
        ----------
        owner : object
            Current cores and their versions, supplied by the ALS backend.
        site : int
            Active site whose local system defines the proposal.
        n_samples : int, optional
            Number of external configurations before expanding active input
            fibers. Deterministic selectors enumerate
            their complete fixed rows and allow this argument to be omitted.
        generator : torch.Generator, optional
            Generator on the sampling device; unused for deterministic rows.

        Returns
        -------
        SampleBatch
            Selected rows with their original probabilities and weights.
        """
        self._prepare_draw(owner, site, n_samples)
        if (n_samples is not None) and (n_samples != self.n_rows):
            raise ValueError(
                '`n_samples` should equal all rows for ExactRows')
        ids = torch.arange(
            self.n_rows, device=self.device, dtype=torch.long)
        probabilities = torch.full(
            (self.n_rows,),
            1 / self.n_rows,
            device=self.device,
            dtype=torch.get_default_dtype())
        return _sample_batch(ids, probabilities, self, site)


class ObservedRows(RowSampler):  # MARK: ObservedRows
    """
    Deterministic non-refreshable rows from fixed observations.

    Parameters
    ----------
    observations : SparseTensorSource
        Scalar observed values with all sites selected as inputs. Only stored
        entries participate, including stored zeros. Their indices are reused
        unchanged; weights belonging to the objective are applied separately.
    """

    proposal_exact = True
    refreshable = False

    def __init__(self, observations: SparseTensorSource) -> None:
        if not isinstance(observations, SparseTensorSource):
            raise TypeError('`observations` should be SparseTensorSource type')
        if observations.is_matrix or observations.n_batches or \
                (observations.in_sites != tuple(range(observations.n_sites))):
            raise ValueError(
                '`observations` should select all vector sites as inputs')
        if observations.out_shape or (observations.support_values.ndim != 1):
            raise ValueError('`observations` should contain scalar values')
        if not observations.support_values.numel():
            raise ValueError(
                '`observations` should contain at least one entry')
        self.observations = observations

    def draw(self,
             owner: Any,
             site: int,
             n_samples: Optional[int] = None,
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """
        Returns every fixed observed id with unit effective weights.

        Parameters
        ----------
        owner : object
            Current cores and their versions, supplied by the ALS backend.
        site : int
            Active site whose local system defines the proposal.
        n_samples : int, optional
            Number of external configurations before expanding active input
            fibers. Deterministic selectors enumerate
            their complete fixed rows and allow this argument to be omitted.
        generator : torch.Generator, optional
            Generator on the sampling device; unused for deterministic rows.

        Returns
        -------
        SampleBatch
            Selected rows with their original probabilities and weights.
        """
        self._prepare_draw(owner, site, n_samples)
        indices = self.observations.support.values
        n_observations = indices.shape[0]
        if self.n_rows != prod(self.observations.in_dim):
            raise ValueError(
                'Sampling state rows should match the observation input shape')
        if (n_samples is not None) and (n_samples != n_observations):
            raise ValueError(
                '`n_samples` should equal all observations for ObservedRows')
        ids = _ravel_indices(
            indices, self.observations.in_dim).to(self.device)
        probabilities = torch.full(
            (n_observations,),
            1 / n_observations,
            device=self.device,
            dtype=torch.get_default_dtype())
        return _sample_batch(ids, probabilities, self, site)


class UniformRows(RowSampler):  # MARK: UniformRows
    """
    Samples uniform external configurations and opens the active input fiber.

    Independent draws select only the other input indices. Reusable draws
    share complete base configurations and cache each site's expanded batch
    and target values until the sample generation changes.
    """

    proposal_exact = True
    refreshable = True

    def __init__(self) -> None:
        self._base_indices = None
        self._batches = {}
        self._targets = {}

    def draw(self, owner: Any, site: int, n_samples: Optional[int],
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """
        Draws external indices and expands the complete active input fiber.

        Parameters
        ----------
        owner : object
            Current cores and their versions, supplied by the ALS backend.
        site : int
            Active site whose local system defines the proposal.
        n_samples : int
            Number of external configurations drawn with replacement.
        generator : torch.Generator, optional
            Generator on the sampling device.

        Returns
        -------
        SampleBatch
            Expanded rows with their probabilities and importance weights.
        """
        self._prepare_draw(owner, site, n_samples)
        if n_samples is None:
            raise ValueError('`n_samples` is required for UniformRows')
        if generator is not None:
            if not isinstance(generator, torch.Generator):
                raise TypeError(
                    '`generator` should be torch.Generator or None')
            if torch.device(generator.device).type != self.device.type:
                raise ValueError(
                    '`generator` device should match the sampling state '
                    'device')

        in_dim = tuple(core.shape[1] for core in owner.cores)
        if self._base_indices is None:
            indices = torch.zeros(n_samples, len(in_dim), dtype=torch.long,
                                  device=self.device)
            for current, dimension in enumerate(in_dim):
                if current != site:
                    indices[:, current] = torch.randint(
                        dimension, (n_samples,), device=self.device,
                        generator=generator)
        else:
            indices = self._base_indices
        indices = _expand_fibers(indices, site, in_dim)
        ids = _ravel_indices(indices, in_dim)
        probabilities = torch.full(
            (ids.numel(),), 1 / self.n_rows, device=self.device,
            dtype=torch.get_default_dtype())
        return _sample_batch(ids, probabilities, self, site)

    def sample(self, owner: Any, site: int, n_samples: int,
               generator: Optional[torch.Generator], generation: int,
               reuse: bool) -> Tuple[SampleBatch, bool]:
        """Reuses shared-base fibers or draws fresh environments."""
        if (self.generation != generation) or not reuse:
            self._base_indices = None
            self._batches.clear()
            self._targets.clear()
        self.generation = generation
        if reuse:
            if site in self._batches:
                return self._batches[site], False
            if self._base_indices is None:
                self._prepare_draw(owner, site, n_samples)
                ids = torch.randint(self.n_rows, (n_samples,),
                                    device=self.device, generator=generator)
                self._base_indices = _unravel_indices(
                    ids, tuple(core.shape[1] for core in owner.cores))
        batch = self.draw(owner, site, n_samples, generator)
        if reuse:
            self._batches[site] = batch
        return batch, True

    def evaluate(self, source: Any, batch: SampleBatch) -> torch.Tensor:
        """Caches target fibers only while a shared base remains active."""
        if batch.site not in self._targets:
            target = _evaluate_fibers(source, batch)
            if self._base_indices is None:
                return target
            self._targets[batch.site] = target
        return self._targets[batch.site]


def _region_metrics(cores: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    """Builds backward density metrics for row-norm sampling."""
    if not cores:
        return ()
    metric = torch.eye(
        cores[-1].shape[-1],
        device=cores[-1].device,
        dtype=cores[-1].dtype)
    metrics = [metric]
    for core in reversed(cores):
        paired = torch.einsum('bc,dpc->bdp', metric, core.conj())
        metric = torch.einsum('apb,bdp->ad', core, paired)
        metrics.append(metric)
    return tuple(reversed(metrics))


def _sample_region(cores: Sequence[torch.Tensor],
                   n_samples: int,
                   generator: Optional[torch.Generator]) -> Tuple[
            torch.Tensor, torch.Tensor]:
    """Samples a chain region from its normalized row-norm distribution."""
    if not cores:
        device = torch.device('cpu') if generator is None \
            else torch.device(generator.device)
        return (torch.empty((n_samples, 0), device=device, dtype=torch.long),
                torch.ones(n_samples, device=device))

    metrics = _region_metrics(cores)
    environment = cores[0].new_ones((n_samples, cores[0].shape[0]))
    joint_probability = environment.real.new_ones(n_samples)
    sampled_sites = []
    for site, core in enumerate(cores):
        candidates = torch.einsum('ja,apb->jpb', environment, core)
        paired = torch.einsum('jpb,bc->jpc', candidates, metrics[site + 1])
        scores = (paired * candidates.conj()).sum(-1).real.clamp_min(0)
        normalization = scores.sum(dim=1, keepdim=True)
        if torch.any(normalization <= 0):
            raise ValueError(
                'A leverage recursion reached a zero-probability prefix')
        conditional = scores / normalization
        selected = torch.multinomial(
            conditional, 1, replacement=True, generator=generator).squeeze(1)
        selected_probability = conditional.gather(
            1, selected.unsqueeze(1)).squeeze(1)
        joint_probability = joint_probability * selected_probability
        selected_slices = core[:, selected, :].permute(1, 0, 2)
        environment = torch.einsum(
            'ja,jab->jb', environment, selected_slices)
        sampled_sites.append(selected)
    return torch.stack(sampled_sites, dim=1), joint_probability


def _region_row_probability(cores: Sequence[torch.Tensor],
                            indices: torch.Tensor) -> torch.Tensor:
    """Evaluates normalized row-norm probabilities for selected ``indices``."""
    if not cores:
        return torch.ones(
            indices.shape[0], device=indices.device,
            dtype=torch.get_default_dtype())
    environment = cores[0].new_ones((indices.shape[0], cores[0].shape[0]))
    for site, core in enumerate(cores):
        selected = core[:, indices[:, site], :].permute(1, 0, 2)
        environment = torch.einsum('ja,jab->jb', environment, selected)
    row_norm = environment.abs().square().sum(dim=1)
    total = _region_metrics(cores)[0].trace().real
    if total <= 0:
        raise ValueError('A leverage region should have positive total norm')
    return row_norm / total


def _sampling_cores(cores: Sequence[torch.Tensor],
                    cyclic: bool) -> Tuple[torch.Tensor, ...]:
    """Validates the current standard cores used by a leverage proposal."""
    if isinstance(cores, (str, bytes)) or not isinstance(cores, Sequence):
        raise TypeError('`cores` should return a sequence of torch.Tensor')
    cores = tuple(cores)
    if not cores:
        raise ValueError('`cores` should contain at least one core')
    if not all(isinstance(core, torch.Tensor) for core in cores):
        raise TypeError('`cores` should contain torch.Tensor objects')
    if any((core.ndim != 3) or any(dim < 1 for dim in core.shape)
           for core in cores):
        raise ValueError('`cores` should have three positive dimensions')
    dtype, device = cores[0].dtype, cores[0].device
    if not (cores[0].is_floating_point() or cores[0].is_complex()):
        raise TypeError('`cores` should have floating or complex dtype')
    for site, core in enumerate(cores):
        if (core.dtype != dtype) or (core.device != device):
            raise ValueError('`cores` should share dtype and device')
        if (site or cyclic) and (core.shape[0] != cores[site - 1].shape[-1]):
            raise ValueError('Adjacent ranks in `cores` should match')
    if not cyclic and ((cores[0].shape[0] != 1) or (cores[-1].shape[-1] != 1)):
        raise ValueError('TT `cores` should have unit boundary ranks')
    return cores


class TTLeverageRows(RowSampler):  # MARK: TTLeverageRows
    """
    Samples TT local-design rows from mixed-canonical leverage scores.

    The exact mixed-canonical construction follows *Efficient Leverage Score
    Sampling for Tensor Train Decomposition* (2024), available in this `paper
    <https://arxiv.org/abs/2406.02749>`__ by Vivek Bharadwaj, Beheshteh T.
    Rakhshan, Osman Asif Malik and Guillaume Rabusseau.

    The current cores are obtained from ``cores`` at every draw. Cores to the
    left of the selected site must be left-isometric, and cores to its right
    right-isometric. Under this invariant, the leverage distribution factors
    into left and right row norms. External indices are sampled recursively
    without materializing the full local design; every selected environment
    retains the complete active input fiber.

    ``uniform_mix`` mixes the normalized leverage distribution with a global
    uniform distribution. A positive value gives every row non-zero support.
    Importance weighting makes the sampled Gram matrix and right-hand side
    unbiased when support is sufficient; it does not make the nonlinear
    least-squares solution itself an unbiased estimator.

    Parameters
    ----------
    cores : callable
        Returns the current standard cores whenever a distribution is built.
        Each core has shape ``(left rank, input, right rank)``. The callback
        exposes updates without copying or retaining an obsolete decomposition.
    uniform_mix : float
        Uniform proposal weight in ``[0, 1]``. Default is zero. A positive
        value gives every global row nonzero probability; draw probabilities
        always include the mixture and stay attached to the sampled rows.
    """

    proposal_exact = True
    refreshable = True

    def __init__(self,
                 cores: Callable[[], Sequence[torch.Tensor]],
                 uniform_mix: float = 0.0) -> None:
        if not callable(cores):
            raise TypeError('`cores` should be a callable returning TT cores')
        if isinstance(uniform_mix, bool) or \
                (not isinstance(uniform_mix, (int, float))):
            raise TypeError('`uniform_mix` should be a number in [0, 1]')
        if not isfinite(uniform_mix) or not (0 <= uniform_mix <= 1):
            raise ValueError('`uniform_mix` should be finite and in [0, 1]')
        self._cores = cores
        self.uniform_mix = float(uniform_mix)

    @staticmethod
    def _right_sampling_cores(
        cores: Sequence[torch.Tensor]
    ) -> Tuple[torch.Tensor, ...]:
        """Reverses a right-canonical region into left-sampling form."""
        return tuple(
            core.permute(2, 1, 0).conj() for core in reversed(cores))

    def _current_cores(self) -> Tuple[torch.Tensor, ...]:
        """Returns current validated standard cores."""
        return _sampling_cores(self._cores(), cyclic=False)

    @staticmethod
    def _validate_mixed_canonical(cores: Sequence[torch.Tensor],
                                  site: int) -> None:
        """Checks the isometries that turn row norms into leverage scores."""
        real_dtype = cores[0].real.dtype
        tolerance = 100 * torch.finfo(real_dtype).eps * max(
            max(core.shape) for core in cores)
        for core in cores[:site]:
            matrix = core.reshape(-1, core.shape[-1])
            identity = torch.eye(
                matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
            if not torch.allclose(
                    matrix.mH @ matrix, identity,
                    atol=tolerance, rtol=tolerance):
                raise ValueError(
                    'Cores left of the leverage site should be left-isometric')
        for core in cores[site + 1:]:
            matrix = core.reshape(core.shape[0], -1)
            identity = torch.eye(
                matrix.shape[0], device=matrix.device, dtype=matrix.dtype)
            if not torch.allclose(
                    matrix @ matrix.mH, identity,
                    atol=tolerance, rtol=tolerance):
                raise ValueError(
                    'Cores right of the leverage site should be '
                    'right-isometric')

    def _probabilities_from_indices(self,
                                    cores: Sequence[torch.Tensor],
                                    site: int,
                                    indices: torch.Tensor) -> torch.Tensor:
        """Evaluates mixed probabilities after canonical validation."""
        in_dim = tuple(core.shape[1] for core in cores)
        left_probability = _region_row_probability(
            cores[:site], indices[:, :site])
        right_cores = self._right_sampling_cores(cores[site + 1:])
        right_probability = _region_row_probability(
            right_cores,
            indices[:, site + 1:].flip(1))
        left_probability = left_probability.to(cores[0].real)
        right_probability = right_probability.to(cores[0].real)
        leverage = left_probability * right_probability / in_dim[site]
        uniform = leverage.new_full(
            leverage.shape, 1 / prod(in_dim))
        return (1 - self.uniform_mix) * leverage + \
            self.uniform_mix * uniform

    def probabilities(self,
                      site: int,
                      configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Returns proposal probabilities for selected global configurations.

        Parameters
        ----------
        site : int
            Active site whose local system defines the proposal.
        configurations : ConfigurationBatch
            Discrete configurations with one input per tensor site.

        Returns
        -------
        torch.Tensor
            Real probabilities with shape ``(batch,)``.
        """
        cores = self._current_cores()
        if isinstance(site, bool) or \
                (not isinstance(site, int)) or \
                (site < 0) or (site >= len(cores)):
            raise ValueError('`site` should identify a TT core')
        self._validate_mixed_canonical(cores, site)
        indices = _discrete_indices(
            configurations,
            tuple(core.shape[1] for core in cores),
            cores[0].device)
        return self._probabilities_from_indices(cores, site, indices)

    def draw(self,
             owner: Any,
             site: int,
             n_samples: Optional[int],
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """
        Draws recursive leverage rows and records proposal versions.

        Parameters
        ----------
        owner : object
            Current cores and their versions, supplied by the ALS backend.
        site : int
            Active site whose local system defines the proposal.
        n_samples : int, optional
            Number of external configurations before expanding active input
            fibers. Deterministic selectors enumerate
            their complete fixed rows and allow this argument to be omitted.
        generator : torch.Generator, optional
            Generator on the sampling device; unused for deterministic rows.

        Returns
        -------
        SampleBatch
            Selected rows with their original probabilities and weights.
        """
        self._prepare_draw(owner, site, n_samples)
        if n_samples is None:
            raise ValueError('`n_samples` is required for TTLeverageRows')
        cores = self._current_cores()
        if len(cores) != len(self.core_versions):
            raise ValueError('Sampling state should match the current TT')
        if self.n_rows != prod(core.shape[1] for core in cores):
            raise ValueError(
                '`self.n_rows` should match the tensor dimensions')
        if cores[0].device != self.device:
            raise ValueError(
                'Sampling state and TT cores should share a device')
        self._validate_mixed_canonical(cores, site)
        if (generator is not None) and \
                not isinstance(generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator or None')
        if (generator is not None) and \
                (torch.device(generator.device).type != self.device.type):
            raise ValueError(
                '`generator` device should match the sampling state device')

        left_ids, left_probability = _sample_region(
            cores[:site], n_samples, generator)
        current_ids = torch.zeros(
            (n_samples, 1), device=self.device, dtype=torch.long)
        right_sampling_cores = self._right_sampling_cores(
            cores[site + 1:])
        reversed_right_ids, right_probability = _sample_region(
            right_sampling_cores, n_samples, generator)
        right_ids = reversed_right_ids.flip(1)
        leverage_indices = torch.cat(
            (left_ids.to(self.device), current_ids,
             right_ids.to(self.device)), dim=1)

        in_dim = tuple(core.shape[1] for core in cores)
        if self.uniform_mix > 0:
            external_dim = in_dim[:site] + in_dim[site + 1:]
            uniform_ids = torch.randint(
                prod(external_dim), (n_samples,), device=self.device,
                generator=generator)
            external_indices = _unravel_indices(uniform_ids, external_dim)
            uniform_indices = leverage_indices.clone()
            uniform_indices[:, :site] = external_indices[:, :site]
            uniform_indices[:, site + 1:] = external_indices[:, site:]
            use_uniform = torch.rand(
                n_samples, device=self.device,
                generator=generator) < self.uniform_mix
            indices = torch.where(
                use_uniform.unsqueeze(1), uniform_indices, leverage_indices)
        else:
            indices = leverage_indices
        indices = _expand_fibers(indices, site, in_dim)
        ids = _ravel_indices(indices, in_dim)
        if self.uniform_mix == 0:
            environment_probability = (left_probability.to(cores[0].real) *
                                       right_probability.to(cores[0].real))
            probabilities = environment_probability.repeat_interleave(
                in_dim[site]) / in_dim[site]
        else:
            probabilities = self._probabilities_from_indices(
                cores, site, indices)
        if torch.any(probabilities <= 0):
            raise ValueError(
                'Drawn leverage rows should have positive support')
        return _sample_batch(
            ids=ids,
            probabilities=probabilities,
            sampler=self,
            site=site,
            proposal_core_versions=self.core_versions,
            proposal_exact=True)


class TRProductLeverageRows(RowSampler):  # MARK: TRProductLeverageRows
    """
    Samples an approximate product-leverage proposal for TR designs.

    This implements the product proposal from Algorithm 2 of *A Sampling-Based
    Method for Tensor Ring Decomposition* (2021), available in this `paper
    <https://arxiv.org/abs/2010.08581>`__ by Osman Asif Malik and Stephen
    Becker. For every core outside the active site, it computes row leverage
    scores of the mode-input unfolding with shape ``(input, left rank * right
    rank)``. Their product bounds the leverage distribution of the complete
    cyclic design, but is not that exact distribution, so batches are labelled
    ``proposal_exact=False``.

    The published algorithm samples environment configurations and retains the
    full active input fiber. Accordingly, ``n_samples`` counts environments;
    the returned batch contains ``n_samples * in_dim[site]`` scalar rows.
    The scalar row interface expands these fibers explicitly and preserves
    the paper's sampling weights.

    ``uniform_mix`` adds global uniform support. Importance weights always use
    the actual mixed draw probability, so sampled Gram matrices and right-hand
    sides target the full row objective even though the proposal is only an
    approximation to leverage scores.

    Parameters
    ----------
    cores : callable
        Returns the current standard cores whenever a distribution is built.
        Each core has shape ``(left rank, input, right rank)``. The callback
        exposes updates without copying or retaining an obsolete decomposition.
    uniform_mix : float
        Uniform proposal weight in ``[0, 1]``. Default is zero. A positive
        value gives every global row nonzero probability; draw probabilities
        always include the mixture and stay attached to the sampled rows.
    """

    proposal_exact = False
    refreshable = True

    def __init__(self,
                 cores: Callable[[], Sequence[torch.Tensor]],
                 uniform_mix: float = 0.0) -> None:
        if not callable(cores):
            raise TypeError('`cores` should be a callable returning TR cores')
        if isinstance(uniform_mix, bool) or \
                (not isinstance(uniform_mix, (int, float))):
            raise TypeError('`uniform_mix` should be a number in [0, 1]')
        if not isfinite(uniform_mix) or not (0 <= uniform_mix <= 1):
            raise ValueError('`uniform_mix` should be finite and in [0, 1]')
        self._cores = cores
        self.uniform_mix = float(uniform_mix)

    def _current_cores(self) -> Tuple[torch.Tensor, ...]:
        """Returns current validated standard cores."""
        return _sampling_cores(self._cores(), cyclic=True)

    @staticmethod
    def _row_leverage(matrix: torch.Tensor) -> torch.Tensor:
        """Computes normalized numerical row leverage of one ``matrix``."""
        u, singular_values, _ = torch.linalg.svd(
            matrix, full_matrices=False)
        if singular_values.numel() == 0:
            return matrix.real.new_full(
                (matrix.shape[0],), 1 / matrix.shape[0])
        tolerance = (
            max(matrix.shape) * torch.finfo(matrix.real.dtype).eps *
            singular_values.max())
        active = singular_values > tolerance
        scores = (u.abs().square() * active.unsqueeze(0)).sum(dim=1)
        total = scores.sum()
        if total > 0:
            return scores / total
        return scores.new_full((matrix.shape[0],), 1 / matrix.shape[0])

    @staticmethod
    def _input_leverage(core: torch.Tensor) -> torch.Tensor:
        """Returns row leverage of the ``core``'s mode-input unfolding."""
        left_rank, in_dim, right_rank = core.shape
        unfolding = core.permute(1, 0, 2).reshape(
            in_dim, left_rank * right_rank)
        return TRProductLeverageRows._row_leverage(unfolding)

    @staticmethod
    def _site_probabilities(cores: Sequence[torch.Tensor],
                            site: int) -> Tuple[torch.Tensor, ...]:
        """Returns independent unfolding-leverage marginals."""
        probabilities = []
        for current, core in enumerate(cores):
            in_dim = core.shape[1]
            if current == site:
                probability = core.real.new_full(
                    (in_dim,), 1 / in_dim)
            else:
                probability = TRProductLeverageRows._input_leverage(core)
            probabilities.append(probability)
        return tuple(probabilities)

    def _probabilities_from_indices(self,
                                    cores: Sequence[torch.Tensor],
                                    site: int,
                                    indices: torch.Tensor) -> torch.Tensor:
        """Evaluates the mixed product proposal at selected rows."""
        marginals = self._site_probabilities(cores, site)
        return self._mixed_probabilities(
            marginals, indices, prod(core.shape[1] for core in cores))

    def _mixed_probabilities(self,
                             marginals: Sequence[torch.Tensor],
                             indices: torch.Tensor,
                             n_rows: int) -> torch.Tensor:
        """
        Evaluates already-computed ``marginals`` and their uniform mixture.
        """
        product_probability = marginals[0].index_select(0, indices[:, 0])
        for current, marginal in enumerate(marginals[1:], 1):
            product_probability = product_probability * marginal.index_select(
                0, indices[:, current])
        uniform = product_probability.new_full(
            product_probability.shape, 1 / n_rows)
        return (1 - self.uniform_mix) * product_probability + \
            self.uniform_mix * uniform

    def probabilities(self,
                      site: int,
                      configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Returns proposal probabilities for selected global configurations.

        Parameters
        ----------
        site : int
            Active site whose local system defines the proposal.
        configurations : ConfigurationBatch
            Discrete configurations with one input per tensor site.

        Returns
        -------
        torch.Tensor
            Real probabilities with shape ``(batch,)``.
        """
        cores = self._current_cores()
        if isinstance(site, bool) or not isinstance(site, int) or \
                (site < 0) or (site >= len(cores)):
            raise ValueError('`site` should identify a TR core')
        indices = _discrete_indices(
            configurations,
            tuple(core.shape[1] for core in cores),
            cores[0].device)
        return self._probabilities_from_indices(cores, site, indices)

    def draw(self,
             owner: Any,
             site: int,
             n_samples: Optional[int],
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """
        Draws product environments and expands complete active fibers.

        Parameters
        ----------
        owner : object
            Current cores and their versions, supplied by the ALS backend.
        site : int
            Active site whose local system defines the proposal.
        n_samples : int, optional
            Number of external configurations before expanding active input
            fibers. Deterministic selectors enumerate
            their complete fixed rows and allow this argument to be omitted.
        generator : torch.Generator, optional
            Generator on the sampling device; unused for deterministic rows.

        Returns
        -------
        SampleBatch
            Selected rows with their original probabilities and weights.
        """
        self._prepare_draw(owner, site, n_samples)
        if n_samples is None:
            raise ValueError(
                '`n_samples` is required for TRProductLeverageRows')
        cores = self._current_cores()
        if len(cores) != len(self.core_versions):
            raise ValueError('Sampling state should match the current TR')
        if self.n_rows != prod(core.shape[1] for core in cores):
            raise ValueError(
                '`self.n_rows` should match the tensor dimensions')
        if cores[0].device != self.device:
            raise ValueError(
                'Sampling state and TR cores should share a device')
        if (generator is not None) and \
                not isinstance(generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator or None')
        if (generator is not None) and \
                (torch.device(generator.device).type != self.device.type):
            raise ValueError(
                '`generator` device should match the sampling state device')

        marginals = self._site_probabilities(cores, site)
        product_indices = torch.zeros(
            (n_samples, len(cores)),
            device=self.device,
            dtype=torch.long)
        for current, marginal in enumerate(marginals):
            if current != site:
                product_indices[:, current] = torch.multinomial(
                    marginal,
                    n_samples,
                    replacement=True,
                    generator=generator)
        if self.uniform_mix > 0:
            environment_rows = self.n_rows // cores[site].shape[1]
            uniform_ids = torch.randint(
                environment_rows,
                (n_samples,),
                device=self.device,
                generator=generator)
            environment_dim = tuple(
                core.shape[1] for current, core in enumerate(cores)
                if current != site)
            uniform_environment = _unravel_indices(
                uniform_ids, environment_dim)
            uniform_indices = product_indices.clone()
            uniform_indices[:, :site] = uniform_environment[:, :site]
            uniform_indices[:, site + 1:] = uniform_environment[:, site:]
            use_uniform = torch.rand(
                n_samples,
                device=self.device,
                generator=generator) < self.uniform_mix
            indices = torch.where(
                use_uniform.unsqueeze(1), uniform_indices, product_indices)
        else:
            indices = product_indices

        in_dim = tuple(core.shape[1] for core in cores)
        indices = _expand_fibers(indices, site, in_dim)
        ids = _ravel_indices(indices, in_dim)
        probabilities = self._mixed_probabilities(
            marginals, indices, self.n_rows)
        if torch.any(probabilities <= 0):
            raise ValueError(
                'Drawn product-leverage rows should have positive support')
        return _sample_batch(
            ids=ids,
            probabilities=probabilities,
            sampler=self,
            site=site,
            proposal_core_versions=self.core_versions,
            proposal_exact=False)


class TRExactLeverageRows(RowSampler):  # MARK: TRExactLeverageRows
    """
    Samples exact leverage rows of a cyclic TR local design.

    This specializes Sections 4.1--4.2 and Appendix B.2 of *Sampling-Based
    Decomposition Algorithms for Arbitrary Tensor Networks* (2022), available
    in this `paper <https://arxiv.org/abs/2210.03828>`__ by Osman Asif Malik,
    Vivek Bharadwaj and Riley Murray, to a TR one-site ALS environment. It
    contracts the double-layer Gram matrix, computes its small pseudoinverse
    and draws the joint input configuration sequentially from exact conditional
    probabilities. The exponentially tall design matrix and its complete
    leverage vector are never formed.

    As in the paper, ``n_samples`` counts environment configurations and every
    selected environment retains the complete active input fiber. TensorKrowch
    additionally permits ``uniform_mix`` for full-support robustness. A mixed
    proposal still stores and uses its exact draw probabilities, although only
    ``uniform_mix=0`` is the pure leverage distribution analyzed in the paper.

    Parameters
    ----------
    cores : callable
        Returns the current standard cores whenever a distribution is built.
        Each core has shape ``(left rank, input, right rank)``. The callback
        exposes updates without copying or retaining an obsolete decomposition.
    uniform_mix : float
        Uniform proposal weight in ``[0, 1]``. Default is zero. A positive
        value gives every global row nonzero probability; draw probabilities
        always include the mixture and stay attached to the sampled rows.
    """

    proposal_exact = True
    refreshable = True

    def __init__(self,
                 cores: Callable[[], Sequence[torch.Tensor]],
                 uniform_mix: float = 0.0) -> None:
        if not callable(cores):
            raise TypeError('`cores` should be a callable returning TR cores')
        if isinstance(uniform_mix, bool) or \
                (not isinstance(uniform_mix, (int, float))):
            raise TypeError('`uniform_mix` should be a number in [0, 1]')
        if not isfinite(uniform_mix) or not (0 <= uniform_mix <= 1):
            raise ValueError('`uniform_mix` should be finite and in [0, 1]')
        self._cores = cores
        self.uniform_mix = float(uniform_mix)

    def _current_cores(self) -> Tuple[torch.Tensor, ...]:
        """Returns current validated standard cores."""
        return _sampling_cores(self._cores(), cyclic=True)

    @staticmethod
    def _suffix_metrics(cores: Sequence[torch.Tensor],
                        order: Sequence[int]) -> Tuple[torch.Tensor, ...]:
        """Contracts suffix double layers after summing their input edges."""
        end_rank = cores[order[-1]].shape[-1] if order \
            else cores[0].shape[0]
        identity = torch.eye(
            end_rank, device=cores[0].device, dtype=cores[0].dtype)
        metric = torch.einsum('xa,yb->xayb', identity, identity.conj())
        metrics = [None] * (len(order) + 1)
        metrics[-1] = metric
        for position in reversed(range(len(order))):
            core = cores[order[position]]
            paired = torch.einsum('xiu,uavb->xiavb', core, metric)
            metric = torch.einsum('xiavb,yiv->xayb', paired, core.conj())
            metrics[position] = metric
        return tuple(metrics)

    @staticmethod
    def _gram_pseudoinverse(
        suffix_metric: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        """Builds the Hermitian Gram pseudoinverse and numerical rank."""
        gram = suffix_metric.conj().permute(1, 0, 3, 2)
        gram = gram.reshape(
            gram.shape[0] * gram.shape[1],
            gram.shape[2] * gram.shape[3])
        gram = (gram + gram.mH) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        scale = eigenvalues.abs().amax()
        tolerance = max(gram.shape) * torch.finfo(gram.real.dtype).eps * scale
        active = eigenvalues > tolerance
        numerical_rank = int(active.sum().item())
        if numerical_rank == 0:
            raise ValueError(
                'The cyclic local design should have positive numerical rank')
        active_vectors = eigenvectors[:, active]
        inverse = active_vectors / eigenvalues[active].unsqueeze(0)
        pseudoinverse = inverse @ active_vectors.mH
        return pseudoinverse, numerical_rank

    def _exact_state(self, cores: Sequence[torch.Tensor], site: int
                     ) -> 'TRExactLeverageRows':
        """Prepares the double-layer contractions for one active site."""
        self.order = (*range(site + 1, len(cores)), *range(site))
        self.suffix_metrics = self._suffix_metrics(cores, self.order)
        self.gram_pseudoinverse, self.numerical_rank = \
            self._gram_pseudoinverse(self.suffix_metrics[0])
        return self

    @staticmethod
    def _environment_features(cores: Sequence[torch.Tensor],
                              site: int,
                              indices: torch.Tensor) -> torch.Tensor:
        """Contracts selected cyclic environments in local column order."""
        order = (*range(site + 1, len(cores)), *range(site))
        if not order:
            rank = cores[site].shape[0]
            environment = torch.eye(
                rank, device=cores[0].device, dtype=cores[0].dtype)
            environment = environment.expand(indices.shape[0], -1, -1)
        else:
            environment = None
            for current in order:
                core = cores[current]
                selected = core[:, indices[:, current], :].permute(1, 0, 2)
                environment = selected if environment is None else \
                    torch.bmm(environment, selected)
        return environment.transpose(-2, -1).flatten(start_dim=1)

    @classmethod
    def _environment_probabilities(cls,
                                   cores: Sequence[torch.Tensor],
                                   site: int,
                                   indices: torch.Tensor,
                                   exact_state: 'TRExactLeverageRows'
                                   ) -> torch.Tensor:
        """Evaluates normalized exact environment leverage probabilities."""
        features = cls._environment_features(cores, site, indices)
        if not features.shape[0]:
            return features.real.new_empty(0)
        scores = torch.einsum(
            'ja,ab,jb->j',
            features,
            exact_state.gram_pseudoinverse,
            features.conj()).real
        tolerance = 100 * torch.finfo(scores.dtype).eps * scores.abs().amax()
        if torch.any(scores < -tolerance):
            raise ValueError('Exact leverage contraction became non-positive')
        return scores.clamp_min(0) / exact_state.numerical_rank

    @staticmethod
    def _conditional_scores(candidates: torch.Tensor,
                            suffix_metric: torch.Tensor,
                            pseudoinverse: torch.Tensor,
                            left_rank: int,
                            right_rank: int) -> torch.Tensor:
        """Contracts all suffixes for every candidate next input value."""
        phi = pseudoinverse.reshape(
            left_rank, right_rank, left_rank, right_rank)
        scores = torch.einsum(
            'jirc,arAR,caCA,jiRC->ji',
            candidates,
            phi,
            suffix_metric,
            candidates.conj()).real
        tolerance = 100 * torch.finfo(scores.dtype).eps * \
            scores.abs().amax(dim=1, keepdim=True)
        if torch.any(scores < -tolerance):
            raise ValueError(
                'A conditional leverage contraction became non-positive')
        return scores.clamp_min(0)

    def probabilities(self,
                      site: int,
                      configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Returns proposal probabilities for selected global configurations.

        Parameters
        ----------
        site : int
            Active site whose local system defines the proposal.
        configurations : ConfigurationBatch
            Discrete configurations with one input per tensor site.

        Returns
        -------
        torch.Tensor
            Real probabilities with shape ``(batch,)``.
        """
        cores = self._current_cores()
        if isinstance(site, bool) or not isinstance(site, int) or \
                (site < 0) or (site >= len(cores)):
            raise ValueError('`site` should identify a TR core')
        in_dim = tuple(core.shape[1] for core in cores)
        indices = _discrete_indices(
            configurations, in_dim, cores[0].device)
        exact_state = self._exact_state(cores, site)
        environment_probability = self._environment_probabilities(
            cores, site, indices, exact_state)
        environment_rows = prod(in_dim) // in_dim[site]
        mixed_environment = (
            (1 - self.uniform_mix) * environment_probability +
            self.uniform_mix / environment_rows)
        return mixed_environment / in_dim[site]

    def draw(self,
             owner: Any,
             site: int,
             n_samples: Optional[int],
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """
        Draws exact conditional environments and expands active fibers.

        Parameters
        ----------
        owner : object
            Current cores and their versions, supplied by the ALS backend.
        site : int
            Active site whose local system defines the proposal.
        n_samples : int, optional
            Number of external configurations before expanding active input
            fibers. Deterministic selectors enumerate
            their complete fixed rows and allow this argument to be omitted.
        generator : torch.Generator, optional
            Generator on the sampling device; unused for deterministic rows.

        Returns
        -------
        SampleBatch
            Selected rows with their original probabilities and weights.
        """
        self._prepare_draw(owner, site, n_samples)
        if n_samples is None:
            raise ValueError(
                '`n_samples` is required for TRExactLeverageRows')
        cores = self._current_cores()
        if len(cores) != len(self.core_versions):
            raise ValueError('Sampling state should match the current TR')
        if self.n_rows != prod(core.shape[1] for core in cores):
            raise ValueError(
                '`self.n_rows` should match the tensor dimensions')
        if cores[0].device != self.device:
            raise ValueError(
                'Sampling state and TR cores should share a device')
        if (generator is not None) and \
                not isinstance(generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator or None')
        if (generator is not None) and \
                (torch.device(generator.device).type != self.device.type):
            raise ValueError(
                '`generator` device should match the sampling state device')

        exact_state = self._exact_state(cores, site)
        right_rank = cores[site].shape[-1]
        left_rank = cores[site].shape[0]
        prefix = torch.eye(
            right_rank, device=self.device, dtype=cores[0].dtype)
        prefix = prefix.expand(n_samples, -1, -1)
        indices = torch.zeros(
            (n_samples, len(cores)), device=self.device, dtype=torch.long)
        for position, current in enumerate(exact_state.order):
            core = cores[current]
            candidates = torch.einsum('jrc,cid->jird', prefix, core)
            scores = self._conditional_scores(
                candidates=candidates,
                suffix_metric=exact_state.suffix_metrics[position + 1],
                pseudoinverse=exact_state.gram_pseudoinverse,
                left_rank=left_rank,
                right_rank=right_rank)
            normalization = scores.sum(dim=1, keepdim=True)
            if torch.any(normalization <= 0):
                raise ValueError(
                    'An exact leverage prefix has zero conditional mass')
            conditional = scores / normalization
            selected = torch.multinomial(
                conditional, 1, replacement=True,
                generator=generator).squeeze(1)
            indices[:, current] = selected
            prefix = candidates[
                torch.arange(n_samples, device=self.device), selected]

        if self.uniform_mix > 0:
            uniform_indices = indices.clone()
            for current, dim in enumerate(core.shape[1] for core in cores):
                if current != site:
                    uniform_indices[:, current] = torch.randint(
                        dim, (n_samples,), device=self.device,
                        generator=generator)
            use_uniform = torch.rand(
                n_samples, device=self.device,
                generator=generator) < self.uniform_mix
            indices = torch.where(
                use_uniform.unsqueeze(1), uniform_indices, indices)

        environment_probability = self._environment_probabilities(
            cores, site, indices, exact_state)
        in_dim = tuple(core.shape[1] for core in cores)
        environment_rows = self.n_rows // in_dim[site]
        mixed_environment = (
            (1 - self.uniform_mix) * environment_probability +
            self.uniform_mix / environment_rows)

        indices = _expand_fibers(indices, site, in_dim)
        ids = _ravel_indices(indices, in_dim)
        probabilities = mixed_environment.repeat_interleave(
            in_dim[site]) / in_dim[site]
        if torch.any(probabilities <= 0):
            raise ValueError(
                'Drawn exact-leverage rows should have positive support')
        return _sample_batch(
            ids=ids,
            probabilities=probabilities,
            sampler=self,
            site=site,
            proposal_core_versions=self.core_versions,
            proposal_exact=True)


__all__ = [
    'SamplingPolicy',
    'SampleBatch',
    'RowSampler',
    'ExactRows',
    'ObservedRows',
    'UniformRows',
    'TTLeverageRows',
    'TRProductLeverageRows',
    'TRExactLeverageRows',
]
