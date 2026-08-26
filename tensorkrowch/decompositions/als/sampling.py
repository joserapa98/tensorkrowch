"""Row sampling and refresh policies for ALS local systems."""

from dataclasses import dataclass, replace
from math import prod
from typing import Callable, Optional, Protocol, Sequence, Tuple

import torch

from tensorkrowch.decompositions.als.problem import ObservedEntries


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
    'SampleRefreshPolicy',
]
