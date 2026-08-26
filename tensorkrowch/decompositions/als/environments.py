"""Reusable environment caches for ALS sweeps."""

from dataclasses import dataclass
from math import isfinite, prod
from typing import (Mapping, Optional, Protocol, Sequence, Tuple, Union)

import torch
import torch.nn.functional as nnf

from tensorkrowch.decompositions.als.sampling import SampleBatch
from tensorkrowch.decompositions.sources import ConfigurationBatch
from tensorkrowch.decompositions.sources.base import (_discrete_indices,
                                                      _unravel_indices)


@dataclass(frozen=True)
class CoreUpdateSet:
    """Atomic collection of updated cores and their new versions."""

    sites: Sequence[int]
    cores: Sequence[torch.Tensor]
    versions: Sequence[int]
    reason: str = 'local_solve'

    def __post_init__(self) -> None:
        sites = tuple(self.sites)
        cores = tuple(self.cores)
        versions = tuple(self.versions)
        if not (len(sites) == len(cores) == len(versions)):
            raise ValueError(
                '`sites`, `cores` and `versions` should have matching lengths')
        if len(set(sites)) != len(sites):
            raise ValueError('`sites` should not contain duplicates')
        if any(isinstance(site, bool) or
               (not isinstance(site, int)) or (site < 0) for site in sites):
            raise ValueError('`sites` should contain non-negative integers')
        if not all(isinstance(core, torch.Tensor) for core in cores):
            raise TypeError('`cores` should contain torch.Tensor objects')
        if any(isinstance(version, bool) or
               (not isinstance(version, int)) or (version < 0)
               for version in versions):
            raise ValueError('`versions` should contain non-negative integers')
        if not isinstance(self.reason, str):
            raise TypeError('`reason` should be str type')
        object.__setattr__(self, 'sites', sites)
        object.__setattr__(self, 'cores', cores)
        object.__setattr__(self, 'versions', versions)

    @property
    def updates(self) -> Mapping[int, Tuple[torch.Tensor, int]]:
        """Maps every touched site to its new core and version."""
        return {
            site: (core, version)
            for site, core, version in zip(
                self.sites, self.cores, self.versions)
        }


class EnvironmentCache(Protocol):
    """Protocol shared by TT, TR and future PEPS environment caches."""

    def prepare_sweep(self, order, samples=None) -> None:
        """Prepares reusable environments for a sweep order."""

    def local_environment(self, site: int):
        """Returns the environment for the next local solve."""

    def commit(self, update_set: CoreUpdateSet) -> None:
        """Commits every tensor change atomically."""

    def invalidate(self, reason: str) -> None:
        """Invalidates all prepared environments."""


@dataclass(frozen=True)
class _EnvironmentKey:
    """Complete dependency key for one cached local environment."""

    site: int
    direction: str
    dependency_versions: Tuple[Tuple[int, int], ...]
    sample_generation: Optional[int]
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class TTLocalEnvironment:
    """Left/right TT contractions defining one local design matrix."""

    site: int
    input_dim: int
    left: torch.Tensor
    right: torch.Tensor
    site_ids: Optional[torch.Tensor]
    log_scale: torch.Tensor
    key: _EnvironmentKey

    @property
    def sampled(self) -> bool:
        """Whether left and right rows are correlated sampled rows."""
        return self.site_ids is not None

    def design(self) -> torch.Tensor:
        """Materializes the exact or sampled local TT design matrix."""
        if self.sampled:
            basis = nnf.one_hot(
                self.site_ids.to(torch.long), self.input_dim).to(
                    device=self.left.device, dtype=self.left.dtype)
            design = torch.einsum(
                'ja,jp,jb->japb', self.left, basis, self.right)
            return design.reshape(design.shape[0], -1)

        identity = torch.eye(
            self.input_dim,
            device=self.left.device,
            dtype=self.left.dtype)
        design = (
            self.left[:, None, None, :, None, None] *
            identity[None, :, None, None, :, None] *
            self.right[None, None, :, None, None, :]
        )
        return design.reshape(
            self.left.shape[0] * self.input_dim * self.right.shape[0], -1)

    def scale_target(self, target: torch.Tensor) -> torch.Tensor:
        """Applies the same global normalization used by the environment."""
        if not isinstance(target, torch.Tensor):
            raise TypeError('`target` should be torch.Tensor type')
        if self.sampled:
            n_rows = self.left.shape[0]
        else:
            n_rows = self.left.shape[0] * self.input_dim * self.right.shape[0]
        if target.shape[0] != n_rows:
            raise ValueError(
                '`target` rows should match the local environment design')
        factor = (-self.log_scale).exp().to(target.dtype)
        return target * factor

    def scale_l2_reg(self, l2_reg: float) -> float:
        """Rescales an absolute Tikhonov lambda with the environment."""
        if isinstance(l2_reg, bool) or not isinstance(l2_reg, (int, float)):
            raise TypeError('`l2_reg` should be a non-negative number')
        if (l2_reg < 0) or (not isfinite(l2_reg)):
            raise ValueError('`l2_reg` should be a non-negative number')
        value = self.log_scale.new_tensor(l2_reg)
        return float((value * (-2 * self.log_scale).exp()).item())


def _environment_norm(environment: torch.Tensor) -> torch.Tensor:
    """Computes a finite global norm after extracting the largest magnitude."""
    maximum = environment.abs().amax()
    if maximum == 0:
        return maximum
    normalized_norm = torch.linalg.vector_norm(environment / maximum)
    norm = maximum * normalized_norm
    return torch.where(torch.isfinite(norm), norm, maximum)


class TTEnvironmentCache:
    """Zip-up TT environment cache for exact and sampled ALS sweeps.

    A forward sweep precomputes old right suffixes and grows a prefix from
    committed cores. A reverse sweep mirrors the process. Cores touched by a
    local solve, gauge absorption or scalar normalization are committed in one
    :class:`CoreUpdateSet`; only cached contractions that depend on those cores
    are rebuilt.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Standard TT cores with shape ``(left rank, input, right rank)`` and
        unit boundary ranks.
    core_versions : sequence of int, optional
        Initial version of every core. Defaults to zero.
    renormalize : bool, optional
        Whether intermediate prefixes/suffixes are divided by one global norm.
        The accumulated scale is stored logarithmically in each local
        environment.
    """

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 core_versions: Optional[Sequence[int]] = None,
                 renormalize: bool = False) -> None:
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        self._cores = self._validate_cores(tuple(cores))
        if core_versions is None:
            versions = (0,) * len(self._cores)
        else:
            versions = tuple(core_versions)
            if len(versions) != len(self._cores):
                raise ValueError(
                    '`core_versions` should contain one value per core')
            if any(isinstance(version, bool) or
                   (not isinstance(version, int)) or (version < 0)
                   for version in versions):
                raise ValueError(
                    '`core_versions` should contain non-negative integers')
        self._versions = versions
        self.renormalize = renormalize
        self._prepared = False
        self._invalidation_reason = None

    @staticmethod
    def _validate_cores(
            cores: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Validates standard TT shapes without mutating cache state."""
        if not cores:
            raise ValueError('`cores` should contain at least one TT core')
        if not all(isinstance(core, torch.Tensor) for core in cores):
            raise TypeError('`cores` should contain torch.Tensor objects')
        if any(core.ndim != 3 for core in cores):
            raise ValueError(
                'TT cores should have left rank, input and right rank dimensions')
        if (cores[0].shape[0] != 1) or (cores[-1].shape[-1] != 1):
            raise ValueError('TT boundary ranks should be one')
        device = cores[0].device
        dtype = cores[0].dtype
        for site, core in enumerate(cores):
            if any(dim < 1 for dim in core.shape):
                raise ValueError('TT core dimensions should be positive')
            if core.device != device:
                raise ValueError('All TT cores should be on the same device')
            if core.dtype != dtype:
                raise ValueError('All TT cores should have the same dtype')
            if not (core.is_floating_point() or core.is_complex()):
                raise TypeError('TT cores should have floating or complex dtype')
            if site and (core.shape[0] != cores[site - 1].shape[-1]):
                raise ValueError('Adjacent TT ranks should match')
        return tuple(cores)

    @property
    def cores(self) -> Tuple[torch.Tensor, ...]:
        """Current TT cores in site order."""
        return self._cores

    @property
    def core_versions(self) -> Tuple[int, ...]:
        """Current version of every core."""
        return self._versions

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Input dimension at every TT site."""
        return tuple(core.shape[1] for core in self._cores)

    def _boundary_environment(
            self, n_rows: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates a unit boundary and zero log-scale."""
        environment = self._cores[0].new_ones((n_rows, 1))
        log_scale = environment.real.new_zeros(())
        return environment, log_scale

    def _normalize(self,
                   environment: torch.Tensor,
                   log_scale: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optionally removes one global norm and accumulates its logarithm."""
        if not self.renormalize:
            return environment, log_scale
        norm = _environment_norm(environment)
        if norm == 0:
            return environment, log_scale
        return environment / norm, log_scale + norm.log()

    def _extend_left(
            self,
            environment: torch.Tensor,
            log_scale: torch.Tensor,
            site: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends one core to a prefix."""
        core = self._cores[site]
        if self._sample_indices is None:
            environment = torch.einsum(
                'la,apb->lpb', environment, core).reshape(-1, core.shape[-1])
        else:
            ids = self._sample_indices[:, site]
            selected = core[:, ids, :].permute(1, 0, 2)
            environment = torch.einsum(
                'ja,jab->jb', environment, selected)
        return self._normalize(environment, log_scale)

    def _extend_right(
            self,
            environment: torch.Tensor,
            log_scale: torch.Tensor,
            site: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepends one core to a suffix."""
        core = self._cores[site]
        if self._sample_indices is None:
            environment = torch.einsum(
                'apb,qb->pqa', core, environment).reshape(-1, core.shape[0])
        else:
            ids = self._sample_indices[:, site]
            selected = core[:, ids, :].permute(1, 0, 2)
            environment = torch.einsum(
                'jab,jb->ja', selected, environment)
        return self._normalize(environment, log_scale)

    def _prepare_samples(
            self,
            samples: Optional[Union[ConfigurationBatch, SampleBatch]]) -> None:
        """Normalizes exact, configuration and flat-id sample specifications."""
        if samples is None:
            self._sample_indices = None
            self._sample_generation = None
            return
        if isinstance(samples, SampleBatch):
            if samples.ids.device != self._cores[0].device:
                raise ValueError('Samples and TT cores should share a device')
            if samples.ids.max() >= prod(self.input_dim):
                raise ValueError('A sampled row id is out of bounds')
            self._sample_indices = _unravel_indices(
                samples.ids.to(torch.long), self.input_dim)
            self._sample_generation = samples.generation
            return
        if not isinstance(samples, ConfigurationBatch):
            raise TypeError(
                '`samples` should be ConfigurationBatch, SampleBatch or None')
        self._sample_indices = _discrete_indices(
            samples, self.input_dim, self._cores[0].device)
        self._sample_generation = 0

    def prepare_sweep(
            self,
            order: Sequence[int],
            samples: Optional[Union[ConfigurationBatch, SampleBatch]] = None
            ) -> None:
        """Builds old suffixes or prefixes for one complete sweep."""
        if getattr(self, '_active_site', None) is not None:
            raise RuntimeError(
                'The active local environment should be committed first')
        order = tuple(order)
        forward = tuple(range(len(self._cores)))
        reverse = tuple(reversed(forward))
        if order == forward:
            self._direction = 'forward'
        elif order == reverse:
            self._direction = 'reverse'
        else:
            raise ValueError(
                '`order` should contain all TT sites forward or reverse')
        self._prepare_samples(samples)
        n_rows = 1 if self._sample_indices is None \
            else self._sample_indices.shape[0]
        self._order = order
        self._cursor = 0
        self._active_site = None

        if self._direction == 'forward':
            self._right = [None] * (len(self._cores) + 1)
            self._right_logs = [None] * (len(self._cores) + 1)
            self._right[-1], self._right_logs[-1] = \
                self._boundary_environment(n_rows)
            for site in reversed(range(len(self._cores))):
                self._right[site], self._right_logs[site] = \
                    self._extend_right(
                        self._right[site + 1],
                        self._right_logs[site + 1],
                        site)
            self._running, self._running_log = \
                self._boundary_environment(n_rows)
        else:
            self._left = [None] * (len(self._cores) + 1)
            self._left_logs = [None] * (len(self._cores) + 1)
            self._left[0], self._left_logs[0] = \
                self._boundary_environment(n_rows)
            for site in range(len(self._cores)):
                self._left[site + 1], self._left_logs[site + 1] = \
                    self._extend_left(
                        self._left[site], self._left_logs[site], site)
            self._running, self._running_log = \
                self._boundary_environment(n_rows)

        self._prepared = True
        self._invalidation_reason = None

    def local_environment(self, site: int) -> TTLocalEnvironment:
        """Returns the next local environment in the prepared sweep."""
        if not self._prepared:
            reason = '' if self._invalidation_reason is None \
                else f' after {self._invalidation_reason}'
            raise RuntimeError(f'Environment cache is not prepared{reason}')
        if self._cursor >= len(self._order):
            raise RuntimeError('The prepared sweep is already complete')
        if site != self._order[self._cursor]:
            raise ValueError('`site` should be the next site in the sweep order')
        if self._active_site is not None:
            raise RuntimeError(
                'The previous local environment should be committed first')

        if self._direction == 'forward':
            left = self._running
            left_log = self._running_log
            right = self._right[site + 1]
            right_log = self._right_logs[site + 1]
        else:
            left = self._left[site]
            left_log = self._left_logs[site]
            right = self._running
            right_log = self._running_log
        site_ids = None if self._sample_indices is None \
            else self._sample_indices[:, site]
        dependencies = tuple(
            (other_site, version)
            for other_site, version in enumerate(self._versions)
            if other_site != site)
        key = _EnvironmentKey(
            site=site,
            direction=self._direction,
            dependency_versions=dependencies,
            sample_generation=self._sample_generation,
            device=self._cores[0].device,
            dtype=self._cores[0].dtype)
        self._active_site = site
        return TTLocalEnvironment(
            site=site,
            input_dim=self.input_dim[site],
            left=left,
            right=right,
            site_ids=site_ids,
            log_scale=left_log + right_log,
            key=key)

    def _rebuild_forward_dependencies(self,
                                      site: int,
                                      updated_sites: Sequence[int]) -> None:
        """Repairs prefix/suffix dependencies after an atomic forward update."""
        if any(updated_site < site for updated_site in updated_sites):
            n_rows = 1 if self._sample_indices is None \
                else self._sample_indices.shape[0]
            self._running, self._running_log = \
                self._boundary_environment(n_rows)
            for previous_site in range(site):
                self._running, self._running_log = self._extend_left(
                    self._running, self._running_log, previous_site)

        future = [updated_site for updated_site in updated_sites
                  if updated_site >= (site + 2)]
        if future:
            for rebuild_site in range(max(future), site + 1, -1):
                self._right[rebuild_site], self._right_logs[rebuild_site] = \
                    self._extend_right(
                        self._right[rebuild_site + 1],
                        self._right_logs[rebuild_site + 1],
                        rebuild_site)

    def _rebuild_reverse_dependencies(self,
                                      site: int,
                                      updated_sites: Sequence[int]) -> None:
        """Repairs prefix/suffix dependencies after an atomic reverse update."""
        if any(updated_site > site for updated_site in updated_sites):
            n_rows = 1 if self._sample_indices is None \
                else self._sample_indices.shape[0]
            self._running, self._running_log = \
                self._boundary_environment(n_rows)
            for previous_site in reversed(range(site + 1, len(self._cores))):
                self._running, self._running_log = self._extend_right(
                    self._running, self._running_log, previous_site)

        past = [updated_site for updated_site in updated_sites
                if updated_site <= (site - 2)]
        if past:
            for rebuild_site in range(min(past), site - 1):
                self._left[rebuild_site + 1], \
                    self._left_logs[rebuild_site + 1] = self._extend_left(
                        self._left[rebuild_site],
                        self._left_logs[rebuild_site],
                        rebuild_site)

    def commit(self, update_set: CoreUpdateSet) -> None:
        """Validates and commits all touched cores before advancing the zip-up."""
        if not self._prepared or (self._active_site is None):
            raise RuntimeError(
                'A local environment should be active before `commit`')
        if not isinstance(update_set, CoreUpdateSet):
            raise TypeError('`update_set` should be CoreUpdateSet type')
        if any(site >= len(self._cores) for site in update_set.sites):
            raise ValueError('An updated site is out of bounds')

        candidate_cores = list(self._cores)
        candidate_versions = list(self._versions)
        for site, (core, version) in update_set.updates.items():
            if version <= candidate_versions[site]:
                raise ValueError(
                    'Every updated core version should increase')
            candidate_cores[site] = core
            candidate_versions[site] = version
        validated_cores = self._validate_cores(candidate_cores)

        site = self._active_site
        self._cores = validated_cores
        self._versions = tuple(candidate_versions)
        if self._direction == 'forward':
            self._rebuild_forward_dependencies(site, update_set.sites)
            self._running, self._running_log = self._extend_left(
                self._running, self._running_log, site)
        else:
            self._rebuild_reverse_dependencies(site, update_set.sites)
            self._running, self._running_log = self._extend_right(
                self._running, self._running_log, site)

        self._active_site = None
        self._cursor += 1

    def invalidate(self, reason: str) -> None:
        """Invalidates all prepared environments with an explicit reason."""
        if not isinstance(reason, str):
            raise TypeError('`reason` should be str type')
        self._prepared = False
        self._active_site = None
        self._invalidation_reason = reason
        self._sample_indices = None
        self._sample_generation = None
        for name in (
                '_left', '_left_logs', '_right', '_right_logs', '_running',
                '_running_log'):
            if hasattr(self, name):
                setattr(self, name, None)


__all__ = [
    'EnvironmentCache',
    'CoreUpdateSet',
    'TTLocalEnvironment',
    'TTEnvironmentCache',
]
