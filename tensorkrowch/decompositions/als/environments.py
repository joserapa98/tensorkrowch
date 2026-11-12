"""
This script contains:

    Classes:
        * EnvironmentCache
        * TTEnvironmentCache
        * DirectTREnvironment
        * TRSegmentEnvironmentCache

    Functions:
        * _updated_cores
        * _tr_design
        * _fiber_indices
        * _environment_norm
        * _normalize_environment
        * _tr_sample_indices
        * _sampled_product
        * _evaluate_cores
        * _exact_design
        * _tr_exact_design
"""

from math import prod, sqrt
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, Union

import torch
import torch.nn.functional as nnf

from tensorkrowch.utils import stable_norm

from tensorkrowch.decompositions._utils import _unravel_indices
from tensorkrowch.decompositions.als.sampling import (SampleBatch,
                                                      _expand_fibers)
from tensorkrowch.decompositions.sources import ConfigurationBatch
from tensorkrowch.decompositions.sources.base import _discrete_indices


class EnvironmentCache(Protocol):  # MARK: EnvironmentCache
    """Owns cores, directional contractions and atomic core updates."""

    log_scale: torch.Tensor  # Logarithmic scale removed from the active design

    def prepare_sweep(self, order: Sequence[int],
                      samples: Optional[Union[ConfigurationBatch,
                                              SampleBatch]] = None) -> None:
        """Prepares reusable contractions for a sweep."""

    def design(self, site: int, *, compact: bool = False) -> torch.Tensor:
        """Builds the active matrix and retains its logarithmic scale."""

    def update(self, replacements: Mapping[int, torch.Tensor]) -> None:
        """Validates and applies all replacement cores together."""

    def invalidate(self, reason: str) -> None:
        """Discards prepared contractions."""


def _updated_cores(cores: Sequence[torch.Tensor], versions: Sequence[int],
                   replacements: Mapping[int, torch.Tensor], validate: Any
                   ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[int, ...]]:
    """Validates the complete candidate before replacing cores or versions."""
    if not isinstance(replacements, Mapping):
        raise TypeError('`replacements` should map sites to tensors')
    candidate, updated_versions = list(cores), list(versions)
    for site, core in replacements.items():
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('Replacement sites should be integers')
        if (site < 0) or (site >= len(cores)):
            raise ValueError('An updated site is out of bounds')
        if not isinstance(core, torch.Tensor):
            raise TypeError('Replacement cores should be tensors')
        if (core.ndim != 3) or (core.shape[1] != cores[site].shape[1]):
            raise ValueError('Updated cores should preserve input dimensions')
        if (core.dtype != cores[site].dtype) or \
                (core.device != cores[site].device):
            raise ValueError('Updated cores should preserve dtype and device')
        candidate[site] = core
        updated_versions[site] += 1
    return tuple(validate(candidate)), tuple(updated_versions)


def _tr_design(environment: torch.Tensor, in_dim: int,
               site_ids: torch.Tensor, compact: bool = False) -> torch.Tensor:
    """Forms cyclic rows, optionally sharing them across physical inputs."""
    if compact:
        return environment.transpose(1, 2).reshape(environment.shape[0], -1)
    basis = nnf.one_hot(site_ids.to(torch.long), in_dim).to(
        device=environment.device, dtype=environment.dtype)
    matrix = torch.einsum('jp,jba->japb', basis, environment)
    return matrix.reshape(matrix.shape[0], -1)


def _fiber_indices(indices: torch.Tensor, site: int,
                    in_dim: Sequence[int]) -> torch.Tensor:
    """Extracts environments from complete, environment-major input fibers."""
    dimension = in_dim[site]
    if indices.shape[0] % dimension:
        raise ValueError('Compact systems require complete input fibers')
    base = indices[::dimension]
    if not torch.equal(indices, _expand_fibers(base, site, in_dim)):
        raise ValueError(
            'Compact systems require ordered complete input fibers')
    return base


def _environment_norm(environment: torch.Tensor, n_copies: int = 1
                      ) -> torch.Tensor:
    """Computes a finite norm, including unmaterialized physical copies."""
    maximum = environment.abs().amax()
    norm = stable_norm(environment) * sqrt(n_copies)
    return torch.where(torch.isfinite(norm), norm, maximum)


def _normalize_environment(environment: torch.Tensor,
                           log_scale: torch.Tensor
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts one finite environment scale and accumulates its logarithm."""
    norm = _environment_norm(environment)
    safe_norm = torch.where(norm == 0, 1, norm)
    return environment / safe_norm, log_scale + safe_norm.log()


def _sampled_product(environment: torch.Tensor, core: torch.Tensor,
                      indices: torch.Tensor, *, prepend: bool = False,
                      zipup: bool = False) -> torch.Tensor:
    """Contracts matrices, selecting physical slices before or afterward."""
    if zipup:
        product_ = (torch.einsum('apb,jbc->japc', core, environment)
                    if prepend else
                    torch.einsum('jab,bpc->japc', environment, core))
        rows = torch.arange(len(indices), device=indices.device)
        return product_[rows, :, indices, :]
    selected = core[:, indices, :].permute(1, 0, 2)
    return (torch.bmm(selected, environment) if prepend else
            torch.bmm(environment, selected))


def _evaluate_cores(cores: Sequence[torch.Tensor],
                     indices: Optional[torch.Tensor] = None, *,
                     zipup: bool = False, batch_size: int = 1024
                     ) -> torch.Tensor:
    """Contracts tensors exactly or evaluates selected entries in batches."""
    if indices is None:
        if len(cores) == 1:
            return cores[0].diagonal(dim1=0, dim2=-1).sum(-1)
        # Closing one cyclic index at a time avoids an output-sized pair of
        # virtual axes. TT is the same contraction with one closure value.
        result = None
        for closure in range(cores[0].shape[0]):
            tensor = cores[0][closure]
            for core in cores[1:-1]:
                tensor = torch.tensordot(tensor, core, dims=([-1], [0]))
            tensor = torch.tensordot(
                tensor, cores[-1][:, :, closure], dims=([-1], [0]))
            result = tensor if result is None else result + tensor
        return result.reshape(-1)
    count = indices.shape[0]
    # Bound selected-core and zip-up temporaries as well as the sample count.
    closure = cores[0].shape[0]
    width = max(max(core.shape[0] * core.shape[-1],
                    closure * core.shape[-1] *
                    (core.shape[1] if zipup else 1)) for core in cores)
    batch_size = min(batch_size, max(1, 2 ** 20 // width))
    values = []
    for start in range(0, count, batch_size):
        batch = indices[start:start + batch_size]
        environment = cores[0][:, batch[:, 0], :].permute(1, 0, 2)
        for site, core in enumerate(cores[1:], 1):
            environment = _sampled_product(
                environment, core, batch[:, site], zipup=zipup)
        values.append(environment.diagonal(dim1=-2, dim2=-1).sum(-1))
    return torch.cat(values) if values else cores[0].new_empty(0)


def _exact_design(environment: torch.Tensor, dimension: int,
                   compact: bool) -> torch.Tensor:
    """Shares an external environment over all active physical values."""
    left_size, right_size, left_rank, right_rank = environment.shape
    if compact:
        return environment.reshape(left_size * right_size, -1)
    identity = torch.eye(dimension, dtype=environment.dtype,
                         device=environment.device)
    matrix = torch.einsum('lrab,pi->lpraib', environment, identity)
    return matrix.reshape(left_size * dimension * right_size,
                          left_rank * dimension * right_rank)


def _tr_exact_design(environment: torch.Tensor, site: int,
                      in_dim: Sequence[int], compact: bool) -> torch.Tensor:
    """Orders an exact cyclic environment by the original physical sites."""
    count = len(in_dim)
    left_rank, right_rank = environment.shape[-1], environment.shape[0]
    axes = (*range(count - site, count),
            *range(1, count - site), count, 0)
    environment = environment.permute(axes).reshape(
        prod(in_dim[:site]), prod(in_dim[site + 1:]), left_rank, right_rank)
    return _exact_design(environment, in_dim[site], compact)


class TTEnvironmentCache:  # MARK: TTEnvironmentCache
    """
    Directional TT environment cache for exact and sampled ALS sweeps.

    A forward sweep precomputes old right suffixes and grows a prefix from
    committed cores. A reverse sweep mirrors the process. Cores touched by a
    local solve, gauge absorption or scalar normalization are committed in one
    atomic update; only cached contractions that depend on those cores
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
        The active design's accumulated scale is retained as ``log_scale``.
    zipup : bool, optional
        Whether sampled contractions select physical slices after contracting
        the next core. Defaults to ``False``, selecting before contraction.
        Exact contractions keep all physical indices open in either case.

    Examples
    --------
    >>> cache = tk.decompositions.TTEnvironmentCache(
    ...     [torch.ones(1, 2, 1), torch.ones(1, 3, 1)])
    >>> cache.prepare_sweep((0, 1))
    >>> cache.design(0).shape
    torch.Size([6, 2])
    """

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 core_versions: Optional[Sequence[int]] = None,
                 renormalize: bool = False,
                 zipup: bool = False) -> None:
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        if not isinstance(zipup, bool):
            raise TypeError('`zipup` should be bool type')
        if isinstance(cores, (str, bytes)) or not isinstance(cores, Sequence):
            raise TypeError('`cores` should be a sequence of torch.Tensor')
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
        self.zipup = zipup
        self._prepared = False
        self._active_site = None
        self._invalidation_reason = None

    @property
    def cores(self) -> Tuple[torch.Tensor, ...]:
        """Current TT cores in site order."""
        return self._cores

    @property
    def core_versions(self) -> Tuple[int, ...]:
        """Current version of every core."""
        return self._versions

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension at every TT site."""
        return tuple(core.shape[1] for core in self._cores)

    @staticmethod
    def _validate_cores(cores: Sequence[torch.Tensor]) -> Tuple[
            torch.Tensor, ...]:
        """Validates standard TT shapes without mutating cache state."""
        if not cores:
            raise ValueError('`cores` should contain at least one TT core')
        if not all(isinstance(core, torch.Tensor) for core in cores):
            raise TypeError('`cores` should contain torch.Tensor objects')
        if any(core.ndim != 3 for core in cores):
            raise ValueError(
                'TT cores should have left rank, input and right rank '
                'dimensions')
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
                raise TypeError(
                    'TT cores should have floating or complex dtype')
            if site and (core.shape[0] != cores[site - 1].shape[-1]):
                raise ValueError('Adjacent TT ranks should match')
        return tuple(cores)

    def _boundary_environment(self,
                              n_rows: int) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """Creates a unit boundary and zero log-scale."""
        environment = self._cores[0].new_ones((n_rows, 1))
        log_scale = environment.real.new_zeros(())
        return environment, log_scale

    def _normalize(self,
                   environment: torch.Tensor,
                   log_scale: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """Optionally removes one global norm and accumulates its logarithm."""
        if not self.renormalize:
            return environment, log_scale
        return _normalize_environment(environment, log_scale)

    def _extend_left(self,
                     environment: torch.Tensor,
                     log_scale: torch.Tensor,
                     site: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends one core to a prefix."""
        core = self._cores[site]
        if self._sample_indices is None:
            environment = torch.einsum(
                'la,apb->lpb', environment, core).reshape(-1, core.shape[-1])
        else:
            environment = _sampled_product(
                environment.unsqueeze(1), core, self._sample_indices[:, site],
                zipup=self.zipup).squeeze(1)
        return self._normalize(environment, log_scale)

    def _extend_right(self,
                      environment: torch.Tensor,
                      log_scale: torch.Tensor,
                      site: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepends one core to a suffix."""
        core = self._cores[site]
        if self._sample_indices is None:
            environment = torch.einsum(
                'apb,qb->pqa', core, environment).reshape(-1, core.shape[0])
        else:
            environment = _sampled_product(
                environment.unsqueeze(-1), core,
                self._sample_indices[:, site], prepend=True,
                zipup=self.zipup).squeeze(-1)
        return self._normalize(environment, log_scale)

    def _prepare_samples(self,
                         samples: Optional[
                             Union[ConfigurationBatch, SampleBatch]]) -> None:
        """
        Normalizes exact, configuration and flat-id sample specifications.
        """
        if samples is None:
            self._sample_indices = None
            self._sample_generation = None
            return
        if isinstance(samples, SampleBatch):
            if samples.ids.device != self._cores[0].device:
                raise ValueError('Samples and TT cores should share a device')
            if samples.ids.max() >= prod(self.in_dim):
                raise ValueError('A sampled row id is out of bounds')
            self._sample_indices = _unravel_indices(
                samples.ids.to(torch.long), self.in_dim)
            self._sample_generation = samples.generation
            return
        if not isinstance(samples, ConfigurationBatch):
            raise TypeError(
                '`samples` should be ConfigurationBatch, SampleBatch or None')
        self._sample_indices = _discrete_indices(
            samples, self.in_dim, self._cores[0].device)
        self._sample_generation = 0

    def _rebuild_forward_dependencies(self,
                                      site: int,
                                      updated_sites: Sequence[int]) -> None:
        """
        Repairs prefix/suffix dependencies after an atomic forward update.
        """
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
        """
        Repairs prefix/suffix dependencies after an atomic reverse update.
        """
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

    def prepare_sweep(self,
                      order: Sequence[int],
                      samples: Optional[
                             Union[ConfigurationBatch, SampleBatch]] = None
                      ) -> None:
        """
        Builds old suffixes or prefixes for one complete sweep.

        Parameters
        ----------
        order : sequence of int
            Every site in increasing or decreasing order.
        samples : ConfigurationBatch or SampleBatch, optional
            Selected configurations. ``None`` uses the complete tensor.
        """
        if self._active_site is not None:
            raise RuntimeError(
                'The active local environment should be committed first')
        if isinstance(order, (str, bytes)) or not isinstance(order, Sequence):
            raise TypeError('`order` should be a sequence of integers')
        if any(isinstance(site, bool) or not isinstance(site, int)
               for site in order):
            raise TypeError('`order` should contain integers')
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
            for site in reversed(range(1, len(self._cores))):
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
            for site in range(len(self._cores) - 1):
                self._left[site + 1], self._left_logs[site + 1] = \
                    self._extend_left(
                        self._left[site], self._left_logs[site], site)
            self._running, self._running_log = \
                self._boundary_environment(n_rows)

        self._prepared = True
        self._invalidation_reason = None

    def design(self, site: int, *, compact: bool = False) -> torch.Tensor:
        """
        Builds the active TT design and stores its ``log_scale``.

        Parameters
        ----------
        site : int
            Next site in the prepared sweep order. Commit its update before
            requesting another environment.
        compact : bool
            Whether to return one row per external configuration and one
            column per bond pair, sharing the matrix across physical inputs.
            Defaults to ``False``. Selected entries retain one row each;
            complete sweeps retain one row per external configuration.
        """
        if not isinstance(compact, bool):
            raise TypeError('`compact` should be bool type')
        if not self._prepared:
            reason = '' if self._invalidation_reason is None \
                else f' after {self._invalidation_reason}'
            raise RuntimeError(f'Environment cache is not prepared{reason}')
        if self._cursor >= len(self._order):
            raise RuntimeError('The prepared sweep is already complete')
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if site != self._order[self._cursor]:
            raise ValueError(
                '`site` should be the next site in the sweep order')
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
        self._active_site = site
        self.log_scale = left_log + right_log
        self.sampled = site_ids is not None
        dimension = self.in_dim[site]
        if site_ids is not None:
            if compact:
                self.matrix = torch.einsum('ja,jb->jab', left, right).reshape(
                    left.shape[0], -1)
                return self.matrix
            basis = nnf.one_hot(site_ids.to(torch.long), dimension).to(
                device=left.device, dtype=left.dtype)
            matrix = torch.einsum('ja,jp,jb->japb', left, basis, right)
            self.matrix = matrix.reshape(matrix.shape[0], -1)
            return self.matrix
        environment = torch.einsum('la,rb->lrab', left, right)
        self.matrix = _exact_design(environment, dimension, compact)
        return self.matrix

    def update(self, replacements: Mapping[int, torch.Tensor]) -> None:
        """Validates replacements atomically, then advances the sweep."""
        if not self._prepared or (self._active_site is None):
            raise RuntimeError(
                'A local environment should be active before `update`')
        validated_cores, candidate_versions = _updated_cores(
            self._cores, self._versions, replacements, self._validate_cores)

        site = self._active_site
        self._cores = validated_cores
        self._versions = tuple(candidate_versions)
        if self._cursor == len(self._order) - 1:
            self._active_site = None
            self._cursor += 1
            return
        if self._direction == 'forward':
            self._rebuild_forward_dependencies(site, tuple(replacements))
            self._running, self._running_log = self._extend_left(
                self._running, self._running_log, site)
        else:
            self._rebuild_reverse_dependencies(site, tuple(replacements))
            self._running, self._running_log = self._extend_right(
                self._running, self._running_log, site)

        self._active_site = None
        self._cursor += 1

    def invalidate(self, reason: str) -> None:
        """
        Discards prepared contractions before a new sweep or sample batch.

        Parameters
        ----------
        reason : str
            Description attached to a subsequent unprepared-cache error.
        """
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


class DirectTREnvironment:  # MARK: DirectTREnvironment
    """
    Reference TR environments built by direct cyclic contraction.

    This deliberately simple implementation is intended as a correctness oracle
    for segmented caches. It contracts every core except the active one
    independently at every site and therefore should not be used as the main
    ALS sweep implementation.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Standard cyclic cores with matching neighboring ranks. This direct
        contraction is the reference for the segmented cache.
    zipup : bool, optional
        Whether sampled contractions select physical slices after contracting
        the next core. Defaults to ``False``, selecting before contraction.
        Exact contractions keep all physical indices open in either case.
    """

    def __init__(self, cores: Sequence[torch.Tensor],
                 zipup: bool = False) -> None:
        if isinstance(cores, (str, bytes)) or not isinstance(cores, Sequence):
            raise TypeError('`cores` should be a sequence of torch.Tensor')
        if not isinstance(zipup, bool):
            raise TypeError('`zipup` should be bool type')
        self._cores = TRSegmentEnvironmentCache._validate_cores(tuple(cores))
        self.zipup = zipup

    def design(self, site: int,
               samples: Optional[Union[ConfigurationBatch,
                                       SampleBatch]] = None,
               *, compact: bool = False) -> torch.Tensor:
        """
        Builds direct cyclic rows; ``log_scale`` is zero.

        Parameters
        ----------
        site : int
            Core excluded from the cyclic contraction.
        samples : ConfigurationBatch or SampleBatch, optional
            Selected configurations; omitted for the complete tensor.
        compact : bool
            Whether to share the design across physical inputs. Defaults to
            ``False``. Sampled rows must contain complete input fibers in
            environment-major order.
        """
        n_sites = len(self._cores)
        if isinstance(site, bool) or not isinstance(site, int) or \
                (site < 0) or (site >= n_sites):
            raise ValueError('`site` should identify a TR core')
        if not isinstance(compact, bool):
            raise TypeError('`compact` should be bool type')
        in_dim = tuple(core.shape[1] for core in self._cores)
        indices, _, sampled = _tr_sample_indices(
            samples=samples, in_dim=in_dim, device=self._cores[0].device)
        if compact and sampled:
            indices = _fiber_indices(indices, site, in_dim)
        order = (*range(site + 1, n_sites), *range(site))
        rank = self._cores[site].shape[-1]
        environment = torch.eye(rank, dtype=self._cores[0].dtype,
                                device=self._cores[0].device)
        if sampled:
            environment = environment.expand(len(indices), -1, -1)
        for other_site in order:
            core = self._cores[other_site]
            environment = (_sampled_product(
                environment, core, indices[:, other_site], zipup=self.zipup)
                           if sampled else torch.tensordot(
                               environment, core, dims=([-1], [0])))
        self.log_scale = environment.real.new_zeros(())
        self.sampled = sampled
        if sampled:
            self.matrix = _tr_design(environment, in_dim[site],
                                     indices[:, site], compact=compact)
        else:
            self.matrix = _tr_exact_design(environment, site, in_dim, compact)
        return self.matrix


def _tr_sample_indices(
        samples: Optional[Union[ConfigurationBatch, SampleBatch]],
        in_dim: Sequence[int],
        device: torch.device) -> Tuple[Optional[torch.Tensor], Optional[int],
                                      bool]:
    """Normalizes sampled TR rows; exact contractions need no indices."""
    if samples is None:
        return None, None, False
    if isinstance(samples, SampleBatch):
        if samples.ids.device != device:
            raise ValueError('Samples and TR cores should share a device')
        if samples.ids.max() >= prod(in_dim):
            raise ValueError('A sampled row id is out of bounds')
        return (_unravel_indices(samples.ids.to(torch.long), in_dim),
                samples.generation,
                True)
    if not isinstance(samples, ConfigurationBatch):
        raise TypeError(
            '`samples` should be ConfigurationBatch, SampleBatch or None')
    return _discrete_indices(samples, in_dim, device), 0, True


class TRSegmentEnvironmentCache:  # MARK: TRSegmentEnvironmentCache
    """
    Segmented cache for exact and sampled TR ALS sweeps.

    Sites are partitioned into contiguous segments. Within the active segment,
    old suffixes are combined with a growing updated prefix (and conversely in
    reverse sweeps). Segment summaries provide the remaining cyclic path. All
    products follow the ring orientation and no inverse or pseudoinverse is
    used. Exact products retain only the physical axes of their segment;
    sampled products retain one matrix per requested configuration.

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
    zipup : bool, optional
        Whether sampled contractions select physical slices after contracting
        the next core. Defaults to ``False``, selecting before contraction.
        Exact contractions keep all physical indices open in either case.

    Examples
    --------
    >>> cache = tk.decompositions.TRSegmentEnvironmentCache(
    ...     [torch.ones(2, 2, 2), torch.ones(2, 3, 2)])
    >>> cache.prepare_sweep((0, 1))
    >>> cache.design(0).shape
    torch.Size([6, 8])
    """

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 n_segments: Optional[int] = None,
                 core_versions: Optional[Sequence[int]] = None,
                 renormalize: bool = False,
                 zipup: bool = False) -> None:
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        if not isinstance(zipup, bool):
            raise TypeError('`zipup` should be bool type')
        if isinstance(cores, (str, bytes)) or not isinstance(cores, Sequence):
            raise TypeError('`cores` should be a sequence of torch.Tensor')
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
        self.zipup = zipup
        self._prepared = False
        self._active_site = None
        self._invalidation_reason = None

    @property
    def cores(self) -> Tuple[torch.Tensor, ...]:
        """Current TR cores in site order."""
        return self._cores

    @property
    def core_versions(self) -> Tuple[int, ...]:
        """Current version of every core."""
        return self._versions

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension at every TR site."""
        return tuple(core.shape[1] for core in self._cores)

    @property
    def segments(self) -> Tuple[Tuple[int, int], ...]:
        """Balanced half-open site intervals ``(start, stop)``."""
        return self._segments

    @staticmethod
    def _validate_cores(cores: Sequence[torch.Tensor]) -> Tuple[
            torch.Tensor, ...]:
        """Validates standard cyclic core shapes without changing state."""
        if not cores:
            raise ValueError('`cores` should contain at least one TR core')
        if not all(isinstance(core, torch.Tensor) for core in cores):
            raise TypeError('`cores` should contain torch.Tensor objects')
        if any(core.ndim != 3 for core in cores):
            raise ValueError(
                'TR cores should have left rank, input and right rank '
                'dimensions')
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
                raise TypeError(
                    'TR cores should have floating or complex dtype')
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

    def _identity(self, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Creates an exact or sampled identity and zero log-scale."""
        identity = torch.eye(
            rank, dtype=self._cores[0].dtype, device=self._cores[0].device)
        if self._sampled:
            identity = identity.expand(self._indices.shape[0], -1, -1)
        return identity, identity.real.new_zeros(())

    def _normalize(self,
                   environment: torch.Tensor,
                   log_scale: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """Optionally extracts a global matrix-batch norm."""
        if not self.renormalize:
            return environment, log_scale
        return _normalize_environment(environment, log_scale)

    def _multiply(self,
                  left: torch.Tensor,
                  left_log: torch.Tensor,
                  right: torch.Tensor,
                  right_log: torch.Tensor) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """Contracts neighboring virtual bonds in the ring orientation."""
        product_ = (torch.bmm(left, right) if self._sampled else
                    torch.tensordot(left, right, dims=([-1], [0])))
        return self._normalize(product_, left_log + right_log)

    def _append_core(self,
                     product_: torch.Tensor,
                     log_scale: torch.Tensor,
                     site: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends a selected core to an oriented product."""
        if self._sampled:
            product_ = _sampled_product(
                product_, self._cores[site], self._indices[:, site],
                zipup=self.zipup)
        else:
            product_ = torch.tensordot(
                product_, self._cores[site], dims=([-1], [0]))
        return self._normalize(product_, log_scale)

    def _prepend_core(self,
                      product_: torch.Tensor,
                      log_scale: torch.Tensor,
                      site: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepends a selected core to an oriented product."""
        if self._sampled:
            product_ = _sampled_product(
                product_, self._cores[site], self._indices[:, site],
                prepend=True, zipup=self.zipup)
        else:
            product_ = torch.tensordot(
                self._cores[site], product_, dims=([-1], [0]))
        return self._normalize(product_, log_scale)

    def _segment_summary(self,
                         segment: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Contracts one complete ``segment`` in forward ring order."""
        start, stop = self._segments[segment]
        product_, log_scale = self._identity(self._cores[start].shape[0])
        for site in range(start, stop):
            product_, log_scale = self._append_core(
                product_, log_scale, site)
        return product_, log_scale

    def _prepare_segment_products(self) -> None:
        """Builds segment summaries and old outer prefix/suffix products."""
        n_segments = len(self._segments)
        self._summaries = [self._segment_summary(segment)
                           if n_segments > 1 else None
                           for segment in range(n_segments)]
        closure_rank = self._cores[0].shape[0]
        if self._direction == 'forward':
            self._segment_right = [None] * (n_segments + 1)
            self._segment_right[n_segments] = self._identity(closure_rank)
            for segment in reversed(range(1, n_segments)):
                summary, summary_log = self._summaries[segment]
                product_, product_log = self._segment_right[segment + 1]
                self._segment_right[segment] = self._multiply(
                    summary, summary_log, product_, product_log)
            self._segment_running = self._identity(closure_rank)
        else:
            self._segment_left = [None] * (n_segments + 1)
            self._segment_left[0] = self._identity(closure_rank)
            for segment in range(n_segments - 1):
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
            for site in reversed(range(start + 1, stop)):
                tail, tail_log = self._inside_right[site - start + 1]
                self._inside_right[site - start] = self._prepend_core(
                    tail, tail_log, site)
            self._inside_running = self._identity(
                self._cores[start].shape[0])
        else:
            self._inside_left = [None] * (stop - start + 1)
            self._inside_left[0] = self._identity(
                self._cores[start].shape[0])
            for site in range(start, stop - 1):
                head, head_log = self._inside_left[site - start]
                self._inside_left[site - start + 1] = self._append_core(
                    head, head_log, site)
            self._inside_running = self._identity(
                self._cores[stop - 1].shape[-1])
        self._active_segment = segment

    def _external_product(self,
                          segment: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns every ``segment`` outside the active one in cyclic order."""
        running, running_log = self._segment_running
        if self._direction == 'forward':
            tail, tail_log = self._segment_right[segment + 1]
            return self._multiply(
                tail, tail_log, running, running_log)
        head, head_log = self._segment_left[segment]
        return self._multiply(
            running, running_log, head, head_log)

    def _rebuild_dependencies(self, next_site: int) -> None:
        """Rebuilds products after a change outside the adjacent core pair."""
        self._prepare_segment_products()
        segment = self._segment_of[next_site]
        previous = range(segment) if self._direction == 'forward' else \
            reversed(range(segment + 1, len(self._segments)))
        for other_segment in previous:
            summary, summary_log = self._summaries[other_segment]
            running, running_log = self._segment_running
            if self._direction == 'forward':
                self._segment_running = self._multiply(
                    running, running_log, summary, summary_log)
            else:
                self._segment_running = self._multiply(
                    summary, summary_log, running, running_log)

        self._start_segment(segment)
        start, stop = self._segments[segment]
        if self._direction == 'forward':
            for site in range(start, next_site):
                running, running_log = self._inside_running
                self._inside_running = self._append_core(
                    running, running_log, site)
        else:
            for site in reversed(range(next_site + 1, stop)):
                running, running_log = self._inside_running
                self._inside_running = self._prepend_core(
                    running, running_log, site)

    def prepare_sweep(self,
                      order: Sequence[int],
                      samples: Optional[
                             Union[ConfigurationBatch, SampleBatch]] = None
                      ) -> None:
        """
        Builds old segment products for a complete cyclic sweep.

        Parameters
        ----------
        order : sequence of int
            Every site in increasing or decreasing order.
        samples : ConfigurationBatch or SampleBatch, optional
            Selected configurations. ``None`` uses the complete tensor.
        """
        if self._active_site is not None:
            raise RuntimeError(
                'The active local environment should be committed first')
        if isinstance(order, (str, bytes)) or not isinstance(order, Sequence):
            raise TypeError('`order` should be a sequence of integers')
        if any(isinstance(site, bool) or not isinstance(site, int)
               for site in order):
            raise TypeError('`order` should contain integers')
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
            _tr_sample_indices(samples, self.in_dim, self._cores[0].device)
        self._order = order
        self._cursor = 0
        self._active_site = None
        self._prepare_segment_products()
        self._prepared = True
        self._invalidation_reason = None

    def design(self, site: int, *, compact: bool = False) -> torch.Tensor:
        """
        Builds the active TR design and stores its ``log_scale``.

        Parameters
        ----------
        site : int
            Next site in the prepared sweep order. Commit its update before
            requesting another environment.
        compact : bool
            Whether to return one row per external configuration and one
            column per bond pair, sharing the matrix across physical inputs.
            Defaults to ``False``. Selected entries retain one row each;
            complete sweeps retain one row per external configuration.
        """
        if not isinstance(compact, bool):
            raise TypeError('`compact` should be bool type')
        if not self._prepared:
            reason = '' if self._invalidation_reason is None else \
                f' after {self._invalidation_reason}'
            raise RuntimeError(f'Environment cache is not prepared{reason}')
        if self._cursor >= len(self._order):
            raise RuntimeError('The prepared sweep is already complete')
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if site != self._order[self._cursor]:
            raise ValueError(
                '`site` should be the next site in the sweep order')
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

        self._active_site = site
        self.log_scale = log_scale
        self.sampled = self._sampled
        if self._sampled:
            self.matrix = _tr_design(environment, self.in_dim[site],
                                     self._indices[:, site], compact=compact)
        else:
            self.matrix = _tr_exact_design(
                environment, site, self.in_dim, compact)
        return self.matrix

    def update(self, replacements: Mapping[int, torch.Tensor]) -> None:
        """Validates replacements atomically, then advances the sweep."""
        if not self._prepared or (self._active_site is None):
            raise RuntimeError(
                'A local environment should be active before `update`')
        validated_cores, candidate_versions = _updated_cores(
            self._cores, self._versions, replacements, self._validate_cores)

        site = self._active_site
        segment = self._segment_of[site]
        start, stop = self._segments[segment]
        self._cores = validated_cores
        self._versions = tuple(candidate_versions)
        if self._cursor == len(self._order) - 1:
            self._active_site = None
            self._cursor += 1
            return
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

        receiver = site + 1 if self._direction == 'forward' else site - 1
        if (self._cursor < len(self._order)) and any(
                other_site not in (site, receiver)
                for other_site in tuple(replacements)):
            self._rebuild_dependencies(self._order[self._cursor])

    def invalidate(self, reason: str) -> None:
        """
        Discards prepared contractions before a new sweep or sample batch.

        Parameters
        ----------
        reason : str
            Description attached to a subsequent unprepared-cache error.
        """
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
    'TTEnvironmentCache',
    'DirectTREnvironment',
    'TRSegmentEnvironmentCache',
]
