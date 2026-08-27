"""Row sampling and refresh policies for ALS local systems."""

from dataclasses import dataclass, replace
from math import prod
from typing import Callable, Optional, Protocol, Sequence, Tuple

import torch

from tensorkrowch.decompositions.als.problem import ObservedEntries
from tensorkrowch.decompositions.sources import ConfigurationBatch
from tensorkrowch.decompositions.sources.base import (_discrete_indices,
                                                      _ravel_indices,
                                                      _unravel_indices)


_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@dataclass(frozen=True)
class SampleBatch:
    """Rows drawn for one ALS design together with immutable probabilities.

    ``probabilities`` are the probabilities used when the ids were drawn.
    ``weights`` must equal ``1 / sqrt(n_samples * probabilities)`` and remain
    unchanged while the batch is reused.
    """

    ids: torch.Tensor
    probabilities: torch.Tensor
    weights: torch.Tensor
    generation: int
    proposal_core_versions: Tuple[int, ...] = ()
    proposal_exact: bool = True
    site: Optional[int] = None

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
                '`proposal_core_versions` should contain non-negative integers')
        object.__setattr__(self, 'proposal_core_versions', versions)
        if not isinstance(self.proposal_exact, bool):
            raise TypeError('`proposal_exact` should be bool type')
        if self.site is not None:
            if isinstance(self.site, bool) or \
                    (not isinstance(self.site, int)) or (self.site < 0):
                raise ValueError('`site` should be a non-negative integer')

    def is_exact_for(self, current_core_versions: Sequence[int]) -> bool:
        """Whether proposal probabilities match the current local design."""
        if not self.proposal_exact:
            return False
        if not self.proposal_core_versions:
            return True
        return self.proposal_core_versions == tuple(current_core_versions)

    def gather_and_weight(
            self,
            environment: torch.Tensor,
            target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Selects global rows and applies their sampling weights."""
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


@dataclass(frozen=True)
class _RowSamplingState:
    """Immutable state passed to row samplers by ALS drivers."""

    n_rows: int
    core_versions: Tuple[int, ...]
    generation: int = 0
    device: torch.device = torch.device('cpu')

    def __post_init__(self) -> None:
        if isinstance(self.n_rows, bool) or \
                (not isinstance(self.n_rows, int)) or (self.n_rows < 1):
            raise ValueError('`n_rows` should be a positive integer')
        versions = tuple(self.core_versions)
        if not versions:
            raise ValueError('`core_versions` should contain at least one site')
        if any(isinstance(version, bool) or
               (not isinstance(version, int)) or (version < 0)
               for version in versions):
            raise ValueError(
                '`core_versions` should contain non-negative integers')
        if isinstance(self.generation, bool) or \
                (not isinstance(self.generation, int)) or \
                (self.generation < 0):
            raise ValueError('`generation` should be a non-negative integer')
        object.__setattr__(self, 'core_versions', versions)
        object.__setattr__(self, 'device', torch.device(self.device))

    def update_core(self, site: int) -> '_RowSamplingState':
        """Returns state with one incremented core version."""
        if isinstance(site, bool) or \
                (not isinstance(site, int)) or \
                (site < 0) or (site >= len(self.core_versions)):
            raise ValueError('`site` should identify a core version')
        versions = list(self.core_versions)
        versions[site] += 1
        return replace(self, core_versions=tuple(versions))


class RowSampler(Protocol):
    """Protocol for selecting and weighting rows of a local ALS design."""

    @property
    def proposal_exact(self) -> bool:
        """Whether generated probabilities are exact for their proposal."""

    @property
    def refreshable(self) -> bool:
        """Whether a new generation should redraw ids."""

    def draw(self,
             state: _RowSamplingState,
             site: int,
             n_samples: Optional[int],
             generator: Optional[torch.Generator]) -> SampleBatch:
        """Draws one immutable sample batch."""

    def update_after_core(self,
                          state: _RowSamplingState,
                          site: int) -> _RowSamplingState:
        """Updates proposal state after a core commit."""


def _validate_draw(state: _RowSamplingState,
                   site: int,
                   n_samples: Optional[int]) -> None:
    """Validates arguments shared by stateless row samplers."""
    if not isinstance(state, _RowSamplingState):
        raise TypeError('`state` should be _RowSamplingState type')
    if isinstance(site, bool) or \
            (not isinstance(site, int)) or \
            (site < 0) or (site >= len(state.core_versions)):
        raise ValueError('`site` should identify a core version')
    if n_samples is not None:
        if isinstance(n_samples, bool) or \
                (not isinstance(n_samples, int)) or (n_samples < 1):
            raise ValueError('`n_samples` should be a positive integer or None')


def _sample_batch(ids: torch.Tensor,
                  probabilities: torch.Tensor,
                  state: _RowSamplingState,
                  site: int,
                  proposal_core_versions: Tuple[int, ...] = (),
                  proposal_exact: bool = True) -> SampleBatch:
    """Builds weights from immutable draw probabilities."""
    weights = (ids.numel() * probabilities).rsqrt()
    return SampleBatch(
        ids=ids,
        probabilities=probabilities,
        weights=weights,
        generation=state.generation,
        proposal_core_versions=proposal_core_versions,
        proposal_exact=proposal_exact,
        site=site)


class ExactRows:
    """Deterministic sampler containing every row exactly once."""

    proposal_exact = True
    refreshable = False

    def draw(self,
             state: _RowSamplingState,
             site: int,
             n_samples: Optional[int] = None,
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """Returns all rows with unit effective weights."""
        _validate_draw(state, site, n_samples)
        if (n_samples is not None) and (n_samples != state.n_rows):
            raise ValueError(
                '`n_samples` should equal all rows for ExactRows')
        ids = torch.arange(
            state.n_rows, device=state.device, dtype=torch.long)
        probabilities = torch.full(
            (state.n_rows,),
            1 / state.n_rows,
            device=state.device,
            dtype=torch.get_default_dtype())
        return _sample_batch(ids, probabilities, state, site)

    def update_after_core(self,
                          state: _RowSamplingState,
                          site: int) -> _RowSamplingState:
        """Increments the core version; exact rows remain design-independent."""
        return state.update_core(site)


class ObservedRows:
    """Deterministic non-refreshable rows from fixed observations."""

    proposal_exact = True
    refreshable = False

    def __init__(self, observations: ObservedEntries) -> None:
        if not isinstance(observations, ObservedEntries):
            raise TypeError('`observations` should be ObservedEntries type')
        self.observations = observations

    def draw(self,
             state: _RowSamplingState,
             site: int,
             n_samples: Optional[int] = None,
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """Returns every fixed observed id with unit effective weights."""
        _validate_draw(state, site, n_samples)
        n_observations = self.observations.flat_ids.numel()
        if state.n_rows != prod(self.observations.input_dim):
            raise ValueError(
                'Sampling state rows should match the observation input shape')
        if (n_samples is not None) and (n_samples != n_observations):
            raise ValueError(
                '`n_samples` should equal all observations for ObservedRows')
        ids = self.observations.flat_ids.to(device=state.device)
        probabilities = torch.full(
            (n_observations,),
            1 / n_observations,
            device=state.device,
            dtype=torch.get_default_dtype())
        return _sample_batch(ids, probabilities, state, site)

    def update_after_core(self,
                          state: _RowSamplingState,
                          site: int) -> _RowSamplingState:
        """Increments the core version without changing observed rows."""
        return state.update_core(site)


class UniformRows:
    """Uniform row sampling with replacement."""

    proposal_exact = True
    refreshable = True

    def draw(self,
             state: _RowSamplingState,
             site: int,
             n_samples: Optional[int],
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """Draws uniform ids and stores their original probabilities."""
        _validate_draw(state, site, n_samples)
        if n_samples is None:
            raise ValueError('`n_samples` is required for UniformRows')
        if generator is not None:
            generator_device = torch.device(generator.device)
            if generator_device.type != state.device.type:
                raise ValueError(
                    '`generator` device should match the sampling state device')
        ids = torch.randint(
            state.n_rows,
            (n_samples,),
            device=state.device,
            generator=generator)
        probabilities = torch.full(
            (n_samples,),
            1 / state.n_rows,
            device=state.device,
            dtype=torch.get_default_dtype())
        return _sample_batch(ids, probabilities, state, site)

    def update_after_core(self,
                          state: _RowSamplingState,
                          site: int) -> _RowSamplingState:
        """Increments the core version; uniform probabilities remain exact."""
        return state.update_core(site)


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
        metric = torch.einsum(
            'apb,bc,dpc->ad', core, metric, core.conj())
        metrics.append(metric)
    return tuple(reversed(metrics))


def _sample_region(cores: Sequence[torch.Tensor],
                   n_samples: int,
                   generator: Optional[torch.Generator]
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        scores = torch.einsum(
            'jpb,bc,jpc->jp',
            candidates,
            metrics[site + 1],
            candidates.conj()).real.clamp_min(0)
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
    """Evaluates normalized row-norm probabilities for selected indices."""
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


class TTLeverageRows:
    """Samples TT local-design rows from mixed-canonical leverage scores.

    The current cores are obtained from ``cores`` at every draw. Cores to the
    left of the selected site must be left-isometric, and cores to its right
    right-isometric. Under this invariant, the leverage distribution factors
    into left row norms, a uniform current input and right row norms, so rows
    are sampled recursively without materializing the full local design.

    ``uniform_mix`` mixes the normalized leverage distribution with a global
    uniform distribution. A positive value gives every row non-zero support.
    Importance weighting makes the sampled Gram matrix and right-hand side
    unbiased when support is sufficient; it does not make the nonlinear
    least-squares solution itself an unbiased estimator.
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
        if (uniform_mix < 0) or (uniform_mix > 1):
            raise ValueError('`uniform_mix` should be in [0, 1]')
        self._cores = cores
        self.uniform_mix = float(uniform_mix)

    @staticmethod
    def _right_sampling_cores(
            cores: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Reverses a right-canonical region into left-sampling form."""
        return tuple(
            core.permute(2, 1, 0).conj() for core in reversed(cores))

    def _current_cores(self) -> Tuple[torch.Tensor, ...]:
        """Validates current standard TT cores."""
        cores = tuple(self._cores())
        if not cores:
            raise ValueError('The leverage sampler requires TT cores')
        if any((not isinstance(core, torch.Tensor)) or (core.ndim != 3)
               for core in cores):
            raise ValueError('Leverage sampling requires standard TT cores')
        return cores

    @staticmethod
    def _validate_mixed_canonical(
            cores: Sequence[torch.Tensor], site: int) -> None:
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
                    'Cores right of the leverage site should be right-isometric')

    def _probabilities_from_indices(
            self,
            cores: Sequence[torch.Tensor],
            site: int,
            indices: torch.Tensor) -> torch.Tensor:
        """Evaluates mixed probabilities after canonical validation."""
        input_dim = tuple(core.shape[1] for core in cores)
        left_probability = _region_row_probability(
            cores[:site], indices[:, :site])
        right_cores = self._right_sampling_cores(cores[site + 1:])
        right_probability = _region_row_probability(
            right_cores,
            indices[:, site + 1:].flip(1))
        leverage = left_probability * right_probability / input_dim[site]
        uniform = leverage.new_full(
            leverage.shape, 1 / prod(input_dim))
        return (1 - self.uniform_mix) * leverage + \
            self.uniform_mix * uniform

    def probabilities(self,
                      site: int,
                      configurations: ConfigurationBatch) -> torch.Tensor:
        """Returns the mixed proposal probability of selected global rows."""
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
             state: _RowSamplingState,
             site: int,
             n_samples: Optional[int],
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """Draws recursive leverage rows and records proposal versions."""
        _validate_draw(state, site, n_samples)
        if n_samples is None:
            raise ValueError('`n_samples` is required for TTLeverageRows')
        cores = self._current_cores()
        if len(cores) != len(state.core_versions):
            raise ValueError('Sampling state should match the current TT')
        if cores[0].device != state.device:
            raise ValueError('Sampling state and TT cores should share a device')
        self._validate_mixed_canonical(cores, site)
        if generator is not None and \
                torch.device(generator.device).type != state.device.type:
            raise ValueError(
                '`generator` device should match the sampling state device')

        left_ids, left_probability = _sample_region(
            cores[:site], n_samples, generator)
        current_ids = torch.randint(
            cores[site].shape[1],
            (n_samples, 1),
            device=state.device,
            generator=generator)
        right_sampling_cores = self._right_sampling_cores(
            cores[site + 1:])
        reversed_right_ids, right_probability = _sample_region(
            right_sampling_cores, n_samples, generator)
        right_ids = reversed_right_ids.flip(1)
        leverage_indices = torch.cat(
            (left_ids.to(state.device), current_ids,
             right_ids.to(state.device)), dim=1)

        if self.uniform_mix > 0:
            uniform_ids = torch.randint(
                state.n_rows,
                (n_samples,),
                device=state.device,
                generator=generator)
            uniform_indices = _unravel_indices(
                uniform_ids, tuple(core.shape[1] for core in cores))
            use_uniform = torch.rand(
                n_samples,
                device=state.device,
                generator=generator) < self.uniform_mix
            indices = torch.where(
                use_uniform.unsqueeze(1), uniform_indices, leverage_indices)
        else:
            indices = leverage_indices

        input_dim = tuple(core.shape[1] for core in cores)
        ids = _ravel_indices(indices, input_dim)
        if self.uniform_mix == 0:
            probabilities = left_probability.to(state.device) * \
                right_probability.to(state.device) / input_dim[site]
        else:
            probabilities = self._probabilities_from_indices(
                cores, site, indices)
        if torch.any(probabilities <= 0):
            raise ValueError('Drawn leverage rows should have positive support')
        return _sample_batch(
            ids=ids,
            probabilities=probabilities,
            state=state,
            site=site,
            proposal_core_versions=state.core_versions,
            proposal_exact=True)

    def update_after_core(self,
                          state: _RowSamplingState,
                          site: int) -> _RowSamplingState:
        """Invalidates design-dependent probabilities after core changes."""
        return state.update_core(site)


class TRProductLeverageRows:
    """Samples an approximate product-leverage proposal for TR designs.

    For every core outside the active site, the proposal computes row leverage
    in both oriented core unfoldings and marginalizes their virtual row index
    onto the input index. The active input is uniform. Their product is an
    inexpensive surrogate for product-leverage bounds, but it ignores cyclic
    correlations and cancellations. It is therefore always labelled
    ``proposal_exact=False`` and does not inherit TT leverage guarantees.

    ``uniform_mix`` adds global uniform support. Importance weights always use
    the actual mixed draw probability, so sampled Gram matrices and right-hand
    sides target the full row objective even though the proposal is only an
    approximation to leverage scores.
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
        if (uniform_mix < 0) or (uniform_mix > 1):
            raise ValueError('`uniform_mix` should be in [0, 1]')
        self._cores = cores
        self.uniform_mix = float(uniform_mix)

    def _current_cores(self) -> Tuple[torch.Tensor, ...]:
        """Validates current standard TR cores and cyclic ranks."""
        cores = tuple(self._cores())
        if not cores:
            raise ValueError('The product-leverage sampler requires TR cores')
        if any((not isinstance(core, torch.Tensor)) or (core.ndim != 3)
               for core in cores):
            raise ValueError(
                'Product-leverage sampling requires standard TR cores')
        device = cores[0].device
        dtype = cores[0].dtype
        for site, core in enumerate(cores):
            if core.device != device or core.dtype != dtype:
                raise ValueError('All TR cores should share dtype and device')
            if core.shape[0] != cores[site - 1].shape[-1]:
                raise ValueError('Adjacent TR ranks should match cyclically')
        return cores

    @staticmethod
    def _row_leverage(matrix: torch.Tensor) -> torch.Tensor:
        """Computes normalized numerical row leverage of one matrix."""
        u, singular_values, _ = torch.linalg.svd(
            matrix, full_matrices=False)
        if singular_values.numel() == 0:
            return matrix.real.new_full((matrix.shape[0],), 1 / matrix.shape[0])
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
        """Marginalizes row leverage from both oriented core unfoldings."""
        left_rank, input_dim, right_rank = core.shape
        forward = core.reshape(left_rank * input_dim, right_rank)
        forward = TRProductLeverageRows._row_leverage(forward).reshape(
            left_rank, input_dim).sum(dim=0)
        reverse = core.permute(1, 2, 0).reshape(
            input_dim * right_rank, left_rank)
        reverse = TRProductLeverageRows._row_leverage(reverse).reshape(
            input_dim, right_rank).sum(dim=1)
        scores = forward + reverse
        return scores / scores.sum()

    @staticmethod
    def _site_probabilities(
            cores: Sequence[torch.Tensor], site: int) -> Tuple[torch.Tensor, ...]:
        """Returns independent unfolding-leverage marginals."""
        probabilities = []
        for current, core in enumerate(cores):
            input_dim = core.shape[1]
            if current == site:
                probability = core.real.new_full(
                    (input_dim,), 1 / input_dim)
            else:
                probability = TRProductLeverageRows._input_leverage(core)
            probabilities.append(probability)
        return tuple(probabilities)

    def _probabilities_from_indices(
            self,
            cores: Sequence[torch.Tensor],
            site: int,
            indices: torch.Tensor) -> torch.Tensor:
        """Evaluates the mixed product proposal at selected rows."""
        marginals = self._site_probabilities(cores, site)
        return self._mixed_probabilities(
            marginals, indices, prod(core.shape[1] for core in cores))

    def _mixed_probabilities(
            self,
            marginals: Sequence[torch.Tensor],
            indices: torch.Tensor,
            n_rows: int) -> torch.Tensor:
        """Evaluates already-computed marginals and their uniform mixture."""
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
        """Returns approximate mixed proposal probabilities for rows."""
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
             state: _RowSamplingState,
             site: int,
             n_samples: Optional[int],
             generator: Optional[torch.Generator] = None) -> SampleBatch:
        """Draws independent product rows with immutable mixed weights."""
        _validate_draw(state, site, n_samples)
        if n_samples is None:
            raise ValueError(
                '`n_samples` is required for TRProductLeverageRows')
        cores = self._current_cores()
        if len(cores) != len(state.core_versions):
            raise ValueError('Sampling state should match the current TR')
        if cores[0].device != state.device:
            raise ValueError('Sampling state and TR cores should share a device')
        if generator is not None and \
                torch.device(generator.device).type != state.device.type:
            raise ValueError(
                '`generator` device should match the sampling state device')

        marginals = self._site_probabilities(cores, site)
        product_indices = torch.stack([
            torch.multinomial(
                marginal,
                n_samples,
                replacement=True,
                generator=generator)
            for marginal in marginals
        ], dim=1)
        if self.uniform_mix > 0:
            uniform_ids = torch.randint(
                state.n_rows,
                (n_samples,),
                device=state.device,
                generator=generator)
            uniform_indices = _unravel_indices(
                uniform_ids, tuple(core.shape[1] for core in cores))
            use_uniform = torch.rand(
                n_samples,
                device=state.device,
                generator=generator) < self.uniform_mix
            indices = torch.where(
                use_uniform.unsqueeze(1), uniform_indices, product_indices)
        else:
            indices = product_indices

        input_dim = tuple(core.shape[1] for core in cores)
        ids = _ravel_indices(indices, input_dim)
        probabilities = self._mixed_probabilities(
            marginals, indices, state.n_rows)
        if torch.any(probabilities <= 0):
            raise ValueError(
                'Drawn product-leverage rows should have positive support')
        return _sample_batch(
            ids=ids,
            probabilities=probabilities,
            state=state,
            site=site,
            proposal_core_versions=state.core_versions,
            proposal_exact=False)

    def update_after_core(self,
                          state: _RowSamplingState,
                          site: int) -> _RowSamplingState:
        """Invalidates the design-dependent product after a core update."""
        return state.update_core(site)


@dataclass(frozen=True)
class SampleRefreshPolicy:
    """Deterministic generation policy for reusable sampled rows.

    A refresh returns a new immutable batch and invokes ``invalidate`` once.
    Non-refreshable samplers, especially ``ObservedRows``, keep the original
    ids and probabilities for the complete fit.
    """

    reuse_sweeps: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.reuse_sweeps, bool) or \
                (not isinstance(self.reuse_sweeps, int)) or \
                (self.reuse_sweeps < 1):
            raise ValueError('`reuse_sweeps` should be a positive integer')

    def generation(self, sweep: int) -> int:
        """Returns the deterministic sample generation for ``sweep``."""
        if isinstance(sweep, bool) or \
                (not isinstance(sweep, int)) or (sweep < 0):
            raise ValueError('`sweep` should be a non-negative integer')
        return sweep // self.reuse_sweeps

    def sample(self,
               sampler: RowSampler,
               state: _RowSamplingState,
               site: int,
               n_samples: Optional[int],
               generator: Optional[torch.Generator],
               sweep: int,
               current: Optional[SampleBatch] = None,
               invalidate: Optional[Callable[[Optional[SampleBatch],
                                              SampleBatch], None]] = None
               ) -> Tuple[SampleBatch, _RowSamplingState, bool]:
        """Reuses or redraws a batch and centralizes refresh invalidation."""
        target_generation = self.generation(sweep)
        if current is not None:
            if (not sampler.refreshable) or \
                    (current.generation == target_generation):
                return current, state, False

        draw_state = replace(state, generation=target_generation)
        batch = sampler.draw(
            draw_state,
            site=site,
            n_samples=n_samples,
            generator=generator)
        if invalidate is not None:
            invalidate(current, batch)
        return batch, draw_state, True


__all__ = [
    'SampleBatch',
    'RowSampler',
    'ExactRows',
    'ObservedRows',
    'UniformRows',
    'TTLeverageRows',
    'TRProductLeverageRows',
    'SampleRefreshPolicy',
]
