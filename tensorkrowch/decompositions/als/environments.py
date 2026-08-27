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


@dataclass(frozen=True)
class TRLocalEnvironment:
    """Cyclic TR contraction defining one local design matrix."""

    site: int
    input_dim: int
    environment: torch.Tensor
    site_ids: torch.Tensor
    log_scale: torch.Tensor
    key: _EnvironmentKey
    sampled: bool = False

    def design(self) -> torch.Tensor:
        """Materializes local rows in standard core vectorization order."""
        basis = nnf.one_hot(
            self.site_ids.to(torch.long), self.input_dim).to(
                device=self.environment.device,
                dtype=self.environment.dtype)
        design = torch.einsum(
            'jp,jba->japb', basis, self.environment)
        return design.reshape(design.shape[0], -1)

    def scale_target(self, target: torch.Tensor) -> torch.Tensor:
        """Applies the same global normalization used by the environment."""
        if not isinstance(target, torch.Tensor):
            raise TypeError('`target` should be torch.Tensor type')
        if target.shape[0] != self.environment.shape[0]:
            raise ValueError(
                '`target` rows should match the local environment design')
        return target * (-self.log_scale).exp().to(target.dtype)

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


class DirectTREnvironment:
    """Reference TR environments built by direct cyclic contraction.

    This deliberately simple implementation is intended as a correctness
    oracle for segmented caches. It contracts every core except the active one
    independently at every site and therefore should not be used as the main
    ALS sweep implementation.
    """

    def __init__(self, cores: Sequence[torch.Tensor]) -> None:
        self._cores = TRSegmentEnvironmentCache._validate_cores(tuple(cores))

    def local_environment(
            self,
            site: int,
            samples: Optional[Union[ConfigurationBatch, SampleBatch]] = None
            ) -> TRLocalEnvironment:
        """Returns one unnormalized direct environment."""
        n_sites = len(self._cores)
        if isinstance(site, bool) or not isinstance(site, int) or \
                (site < 0) or (site >= n_sites):
            raise ValueError('`site` should identify a TR core')
        indices, generation, sampled = _tr_sample_indices(
            samples=samples,
            input_dim=tuple(core.shape[1] for core in self._cores),
            device=self._cores[0].device)
        order = (*range(site + 1, n_sites), *range(site))
        environment = None
        for other_site in order:
            core = self._cores[other_site]
            selected = core[:, indices[:, other_site], :].permute(1, 0, 2)
            environment = selected if environment is None else \
                torch.bmm(environment, selected)
        if environment is None:
            rank = self._cores[site].shape[0]
            environment = torch.eye(
                rank,
                dtype=self._cores[0].dtype,
                device=self._cores[0].device).expand(indices.shape[0], -1, -1)
        dependencies = tuple(
            (other_site, 0) for other_site in range(n_sites)
            if other_site != site)
        key = _EnvironmentKey(
            site=site,
            direction='direct',
            dependency_versions=dependencies,
            sample_generation=generation,
            device=self._cores[0].device,
            dtype=self._cores[0].dtype)
        return TRLocalEnvironment(
            site=site,
            input_dim=self._cores[site].shape[1],
            environment=environment,
            site_ids=indices[:, site],
            log_scale=environment.real.new_zeros(()),
            key=key,
            sampled=sampled)


def _tr_sample_indices(
        samples: Optional[Union[ConfigurationBatch, SampleBatch]],
        input_dim: Sequence[int],
        device: torch.device
        ) -> Tuple[torch.Tensor, Optional[int], bool]:
    """Normalizes exact and sampled TR rows to packed discrete indices."""
    if samples is None:
        ids = torch.arange(prod(input_dim), device=device, dtype=torch.long)
        return _unravel_indices(ids, input_dim), None, False
    if isinstance(samples, SampleBatch):
        if samples.ids.device != device:
            raise ValueError('Samples and TR cores should share a device')
        if samples.ids.max() >= prod(input_dim):
            raise ValueError('A sampled row id is out of bounds')
        return (_unravel_indices(samples.ids.to(torch.long), input_dim),
                samples.generation,
                True)
    if not isinstance(samples, ConfigurationBatch):
        raise TypeError(
            '`samples` should be ConfigurationBatch, SampleBatch or None')
    return _discrete_indices(samples, input_dim, device), 0, True


class TRSegmentEnvironmentCache:
    """Segmented zip-up cache for exact and sampled TR ALS sweeps.

    Sites are partitioned into contiguous segments. Within the active segment,
    old suffixes are combined with a growing updated prefix (and conversely in
    reverse sweeps). Segment summaries provide the remaining cyclic path. All
    products follow the ring orientation and no inverse or pseudoinverse is
    used.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Standard TR cores with shape ``(left rank, input, right rank)``.
    n_segments : int, optional
        Number of balanced contiguous segments. Defaults to at most three.
    core_versions : sequence of int, optional
        Initial version of every core. Defaults to zero.
    renormalize : bool, optional
        Whether intermediate matrix products remove one global norm and retain
        the accumulated scale logarithmically.
    """

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 n_segments: Optional[int] = None,
                 core_versions: Optional[Sequence[int]] = None,
                 renormalize: bool = False) -> None:
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        self._cores = self._validate_cores(tuple(cores))
        n_sites = len(self._cores)
        if n_segments is None:
            n_segments = min(3, n_sites)
        if isinstance(n_segments, bool) or not isinstance(n_segments, int):
            raise TypeError('`n_segments` should be int type')
        if (n_segments < 1) or (n_segments > n_sites):
            raise ValueError(
                '`n_segments` should be between one and the number of sites')
        self._segments = self._balanced_segments(n_sites, n_segments)
        self._segment_of = tuple(
            segment for segment, (start, stop) in enumerate(self._segments)
            for _ in range(start, stop))
        if core_versions is None:
            versions = (0,) * n_sites
        else:
            versions = tuple(core_versions)
            if len(versions) != n_sites:
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
        self._active_site = None
        self._invalidation_reason = None

    @staticmethod
    def _validate_cores(
            cores: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Validates standard cyclic core shapes without changing state."""
        if not cores:
            raise ValueError('`cores` should contain at least one TR core')
        if not all(isinstance(core, torch.Tensor) for core in cores):
            raise TypeError('`cores` should contain torch.Tensor objects')
        if any(core.ndim != 3 for core in cores):
            raise ValueError(
                'TR cores should have left rank, input and right rank dimensions')
        device = cores[0].device
        dtype = cores[0].dtype
        for site, core in enumerate(cores):
            if any(dim < 1 for dim in core.shape):
                raise ValueError('TR core dimensions should be positive')
            if core.device != device:
                raise ValueError('All TR cores should be on the same device')
            if core.dtype != dtype:
                raise ValueError('All TR cores should have the same dtype')
            if not (core.is_floating_point() or core.is_complex()):
                raise TypeError('TR cores should have floating or complex dtype')
            if core.shape[0] != cores[site - 1].shape[-1]:
                raise ValueError('Adjacent TR ranks should match cyclically')
        return tuple(cores)

    @staticmethod
    def _balanced_segments(n_sites: int,
                           n_segments: int) -> Tuple[Tuple[int, int], ...]:
        """Returns balanced half-open site intervals."""
        width, remainder = divmod(n_sites, n_segments)
        segments = []
        start = 0
        for segment in range(n_segments):
            stop = start + width + (segment < remainder)
            segments.append((start, stop))
            start = stop
        return tuple(segments)

    @property
    def cores(self) -> Tuple[torch.Tensor, ...]:
        """Current TR cores in site order."""
        return self._cores

    @property
    def core_versions(self) -> Tuple[int, ...]:
        """Current version of every core."""
        return self._versions

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Input dimension at every TR site."""
        return tuple(core.shape[1] for core in self._cores)

    @property
    def segments(self) -> Tuple[Tuple[int, int], ...]:
        """Balanced half-open intervals usable as future worker partitions."""
        return self._segments

    def _identity(self, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates one batched identity and zero log-scale."""
        identity = torch.eye(
            rank, dtype=self._cores[0].dtype,
            device=self._cores[0].device).expand(self._indices.shape[0], -1, -1)
        return identity, identity.real.new_zeros(())

    def _normalize(self,
                   environment: torch.Tensor,
                   log_scale: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optionally extracts a global matrix-batch norm."""
        if not self.renormalize:
            return environment, log_scale
        norm = _environment_norm(environment)
        if norm == 0:
            return environment, log_scale
        return environment / norm, log_scale + norm.log()

    def _selected(self, site: int) -> torch.Tensor:
        """Returns all selected matrix slices for one core."""
        core = self._cores[site]
        return core[:, self._indices[:, site], :].permute(1, 0, 2)

    def _multiply(self,
                  left: torch.Tensor,
                  left_log: torch.Tensor,
                  right: torch.Tensor,
                  right_log: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multiplies compatible batches following the ring orientation."""
        product_ = torch.bmm(left, right)
        return self._normalize(product_, left_log + right_log)

    def _append_core(self,
                     product_: torch.Tensor,
                     log_scale: torch.Tensor,
                     site: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends a selected core to an oriented product."""
        zero = log_scale.new_zeros(())
        return self._multiply(
            product_, log_scale, self._selected(site), zero)

    def _prepend_core(self,
                      product_: torch.Tensor,
                      log_scale: torch.Tensor,
                      site: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepends a selected core to an oriented product."""
        zero = log_scale.new_zeros(())
        return self._multiply(
            self._selected(site), zero, product_, log_scale)

    def _segment_summary(
            self, segment: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Contracts one complete segment in forward ring order."""
        start, stop = self._segments[segment]
        product_, log_scale = self._identity(self._cores[start].shape[0])
        for site in range(start, stop):
            product_, log_scale = self._append_core(
                product_, log_scale, site)
        return product_, log_scale

    def _prepare_segment_products(self) -> None:
        """Builds segment summaries and old outer prefix/suffix products."""
        self._summaries = [
            self._segment_summary(segment)
            for segment in range(len(self._segments))]
        closure_rank = self._cores[0].shape[0]
        if self._direction == 'forward':
            n_segments = len(self._segments)
            self._segment_right = [None] * (n_segments + 1)
            self._segment_right[n_segments] = self._identity(closure_rank)
            for segment in reversed(range(n_segments)):
                summary, summary_log = self._summaries[segment]
                product_, product_log = self._segment_right[segment + 1]
                self._segment_right[segment] = self._multiply(
                    summary, summary_log, product_, product_log)
            self._segment_running = self._identity(closure_rank)
        else:
            n_segments = len(self._segments)
            self._segment_left = [None] * (n_segments + 1)
            self._segment_left[0] = self._identity(closure_rank)
            for segment in range(n_segments):
                product_, product_log = self._segment_left[segment]
                summary, summary_log = self._summaries[segment]
                self._segment_left[segment + 1] = self._multiply(
                    product_, product_log, summary, summary_log)
            self._segment_running = self._identity(closure_rank)
        self._active_segment = None

    def _start_segment(self, segment: int) -> None:
        """Prepares old intrasegment products for the active direction."""
        start, stop = self._segments[segment]
        if self._direction == 'forward':
            self._inside_right = [None] * (stop - start + 1)
            self._inside_right[-1] = self._identity(
                self._cores[stop - 1].shape[-1])
            for site in reversed(range(start, stop)):
                selected = self._selected(site)
                zero = selected.real.new_zeros(())
                tail, tail_log = self._inside_right[site - start + 1]
                self._inside_right[site - start] = self._multiply(
                    selected, zero, tail, tail_log)
            self._inside_running = self._identity(
                self._cores[start].shape[0])
        else:
            self._inside_left = [None] * (stop - start + 1)
            self._inside_left[0] = self._identity(
                self._cores[start].shape[0])
            for site in range(start, stop):
                head, head_log = self._inside_left[site - start]
                self._inside_left[site - start + 1] = self._append_core(
                    head, head_log, site)
            self._inside_running = self._identity(
                self._cores[stop - 1].shape[-1])
        self._active_segment = segment

    def prepare_sweep(
            self,
            order: Sequence[int],
            samples: Optional[Union[ConfigurationBatch, SampleBatch]] = None
            ) -> None:
        """Builds old segment products for a complete cyclic sweep."""
        if self._active_site is not None:
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
                '`order` should contain all TR sites forward or reverse')
        self._indices, self._sample_generation, self._sampled = \
            _tr_sample_indices(samples, self.input_dim, self._cores[0].device)
        self._order = order
        self._cursor = 0
        self._active_site = None
        self._prepare_segment_products()
        self._prepared = True
        self._invalidation_reason = None

    def _external_product(
            self, segment: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns every segment outside the active one in cyclic order."""
        running, running_log = self._segment_running
        if self._direction == 'forward':
            tail, tail_log = self._segment_right[segment + 1]
            return self._multiply(
                tail, tail_log, running, running_log)
        head, head_log = self._segment_left[segment]
        return self._multiply(
            running, running_log, head, head_log)

    def local_environment(self, site: int) -> TRLocalEnvironment:
        """Returns the next segmented cyclic environment in the sweep."""
        if not self._prepared:
            reason = '' if self._invalidation_reason is None else \
                f' after {self._invalidation_reason}'
            raise RuntimeError(f'Environment cache is not prepared{reason}')
        if self._cursor >= len(self._order):
            raise RuntimeError('The prepared sweep is already complete')
        if site != self._order[self._cursor]:
            raise ValueError('`site` should be the next site in the sweep order')
        if self._active_site is not None:
            raise RuntimeError(
                'The previous local environment should be committed first')

        segment = self._segment_of[site]
        if segment != self._active_segment:
            self._start_segment(segment)
        start, _ = self._segments[segment]
        external, external_log = self._external_product(segment)
        if self._direction == 'forward':
            suffix, suffix_log = self._inside_right[site - start + 1]
            prefix, prefix_log = self._inside_running
        else:
            suffix, suffix_log = self._inside_running
            prefix, prefix_log = self._inside_left[site - start]
        environment, log_scale = self._multiply(
            suffix, suffix_log, external, external_log)
        environment, log_scale = self._multiply(
            environment, log_scale, prefix, prefix_log)

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
        return TRLocalEnvironment(
            site=site,
            input_dim=self.input_dim[site],
            environment=environment,
            site_ids=self._indices[:, site],
            log_scale=log_scale,
            key=key,
            sampled=self._sampled)

    def commit(self, update_set: CoreUpdateSet) -> None:
        """Commits core changes atomically and advances segmented products."""
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
        segment = self._segment_of[site]
        start, stop = self._segments[segment]
        expected_receiver = (site + 1) % len(self._cores) \
            if self._direction == 'forward' else \
            (site - 1) % len(self._cores)
        unexpected = set(update_set.sites) - {site, expected_receiver}
        if unexpected:
            raise ValueError(
                'Segmented sweeps only accept current and gauge-receiver updates')
        self._cores = validated_cores
        self._versions = tuple(candidate_versions)
        if self._direction == 'forward':
            running, running_log = self._inside_running
            self._inside_running = self._append_core(
                running, running_log, site)
            if site == (stop - 1):
                self._summaries[segment] = self._inside_running
                outer, outer_log = self._segment_running
                summary, summary_log = self._summaries[segment]
                self._segment_running = self._multiply(
                    outer, outer_log, summary, summary_log)
        else:
            running, running_log = self._inside_running
            self._inside_running = self._prepend_core(
                running, running_log, site)
            if site == start:
                self._summaries[segment] = self._inside_running
                summary, summary_log = self._summaries[segment]
                outer, outer_log = self._segment_running
                self._segment_running = self._multiply(
                    summary, summary_log, outer, outer_log)

        self._active_site = None
        self._cursor += 1

    def invalidate(self, reason: str) -> None:
        """Invalidates prepared products with an explicit reason."""
        if not isinstance(reason, str):
            raise TypeError('`reason` should be str type')
        self._prepared = False
        self._active_site = None
        self._invalidation_reason = reason
        for name in (
                '_indices', '_summaries', '_segment_right', '_segment_left',
                '_segment_running', '_inside_right', '_inside_left',
                '_inside_running'):
            if hasattr(self, name):
                setattr(self, name, None)


__all__ = [
    'EnvironmentCache',
    'CoreUpdateSet',
    'TTLocalEnvironment',
    'TTEnvironmentCache',
    'TRLocalEnvironment',
    'DirectTREnvironment',
    'TRSegmentEnvironmentCache',
]
