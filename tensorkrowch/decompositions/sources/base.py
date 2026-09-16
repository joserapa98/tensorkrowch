"""
This script contains:

    Internal classes:
        * _SourceEvaluationTracker

    Public classes:
        * ConfigurationBatch
        * TensorSource
        * FiberTensorSource

    Internal functions:
        * _normalize_in_dim
        * _discrete_indices
        * _ravel_indices
        * _unravel_indices
        * _fiber_configurations
"""

from dataclasses import dataclass
from math import prod
from typing import (Optional,
                    Protocol,
                    Sequence,
                    Tuple,
                    Union,
                    runtime_checkable)

import torch

from tensorkrowch.decompositions.metrics import EvaluationStats


ConfigurationValues = Union[torch.Tensor, Sequence[torch.Tensor]]


_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


@dataclass(frozen=True)
class ConfigurationBatch:
    """Batch of discrete indices or physical coordinates.

    ``values`` can be a packed tensor whose first two dimensions are ``(batch,
    sites)``, or a sequence containing one tensor per site. The sequence form
    permits different coordinate shapes at different sites. ``kind`` applies to
    the complete batch, so discrete indices and physical coordinates cannot be
    mixed implicitly.

    Parameters
    ----------
    values : torch.Tensor or sequence of torch.Tensor
        Packed configurations or one tensor per site. Every site tensor in a
        heterogeneous batch must share its leading batch dimension, device and
        dtype.
    kind : {``"indices"``, ``"coordinates"``}
        Semantic type of every value in the batch. Discrete indices must be
        scalar integers at every site.
    """

    values: ConfigurationValues  # Values in original configuration or observation order
    kind: str = 'indices'  # Whether configurations contain indices or coordinates

    def __post_init__(self) -> None:
        if self.kind not in ('indices', 'coordinates'):
            raise ValueError(
                "`kind` should be 'indices' or 'coordinates'")

        if isinstance(self.values, torch.Tensor):
            if self.values.ndim < 2:
                raise ValueError(
                    'Packed `values` should have batch and site dimensions')
            if self.values.shape[1] < 1:
                raise ValueError('`values` should contain at least one site')
            if (self.kind == 'indices') and (
                    (self.values.ndim != 2) or
                    (self.values.dtype not in _INTEGER_DTYPES)):
                raise TypeError(
                    'Discrete packed `values` should contain scalar integers')
            return

        if isinstance(self.values, (str, bytes)):
            raise TypeError(
                '`values` should be a tensor or a sequence of tensors')
        try:
            site_values = tuple(self.values)
        except TypeError as exc:
            raise TypeError(
                '`values` should be a tensor or a sequence of tensors') from exc
        if not site_values:
            raise ValueError('`values` should contain at least one site')
        if not all(isinstance(value, torch.Tensor) for value in site_values):
            raise TypeError('Every site value should be a torch.Tensor')

        first = site_values[0]
        if first.ndim < 1:
            raise ValueError('Every site value should have a batch dimension')
        for value in site_values:
            if value.ndim < 1:
                raise ValueError(
                    'Every site value should have a batch dimension')
            if value.shape[0] != first.shape[0]:
                raise ValueError(
                    'All site values should have the same batch size')
            if value.device != first.device:
                raise ValueError('All site values should be on the same device')
            if value.dtype != first.dtype:
                raise ValueError('All site values should have the same dtype')
            if (self.kind == 'indices') and (
                    (value.ndim != 1) or
                    (value.dtype not in _INTEGER_DTYPES)):
                raise TypeError(
                    'Discrete heterogeneous `values` should contain scalar '
                    'integers')
        object.__setattr__(self, 'values', site_values)

    @property
    def packed(self) -> bool:
        """Whether all sites are stored in one tensor."""
        return isinstance(self.values, torch.Tensor)

    @property
    def batch_size(self) -> int:
        """Number of configurations in the batch."""
        if self.packed:
            return self.values.shape[0]
        return self.values[0].shape[0]

    @property
    def n_sites(self) -> int:
        """Number of sites represented by each configuration."""
        if self.packed:
            return self.values.shape[1]
        return len(self.values)

    @property
    def device(self) -> torch.device:
        """Device shared by the configuration values."""
        if self.packed:
            return self.values.device
        return self.values[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Data type shared by the configuration values."""
        if self.packed:
            return self.values.dtype
        return self.values[0].dtype

    @property
    def site_shape(self) -> Tuple[Tuple[int, ...], ...]:
        """Coordinate shape stored at every site, excluding the batch."""
        if self.packed:
            shape = tuple(self.values.shape[2:])
            return (shape,) * self.n_sites
        return tuple(tuple(value.shape[1:]) for value in self.values)

    def as_tensor(self) -> torch.Tensor:
        """Returns a packed tensor when all site shapes are compatible."""
        if self.packed:
            return self.values
        if any(shape != self.site_shape[0] for shape in self.site_shape[1:]):
            raise ValueError(
                'Heterogeneous site shapes cannot be packed into one tensor')
        return torch.stack(self.values, dim=1)

    def index_select(self, indices: torch.Tensor) -> 'ConfigurationBatch':
        """Selects configurations along the batch dimension."""
        if not isinstance(indices, torch.Tensor):
            raise TypeError('`indices` should be torch.Tensor type')
        if (indices.ndim != 1) or (indices.dtype not in _INTEGER_DTYPES):
            raise TypeError('`indices` should be a one-dimensional integer tensor')
        indices = indices.to(device=self.device, dtype=torch.long)
        if self.packed:
            values = self.values.index_select(0, indices)
        else:
            values = tuple(value.index_select(0, indices)
                           for value in self.values)
        return ConfigurationBatch(values, kind=self.kind)

    def to(self, device: Union[str, torch.device]) -> 'ConfigurationBatch':
        """Returns the configurations on ``device`` without changing dtype."""
        device = torch.device(device)
        if self.packed:
            values = self.values.to(device=device)
        else:
            values = tuple(value.to(device=device) for value in self.values)
        return ConfigurationBatch(values, kind=self.kind)


@runtime_checkable
class TensorSource(Protocol):
    """Provides values for a fixed tensor or function without imposing an
    algorithm.

    ``in_dim`` declares the discrete input dimensions. ``output_shape``
    describes the tensor returned after each configuration; it is ``()`` for
    scalar sources and may be ``None`` until a callable is first evaluated.
    ``dtype`` can likewise be inferred on first evaluation. ``device`` is the
    effective evaluation device.

    ALS and sketching consume the same protocol. A source does not define a
    loss, choose sampled rows or interpret absent observations. Implementations
    must preserve the configuration order and return deterministic values for a
    fixed input and fixed source state.
    """

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Discrete input dimension at every site."""

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """Shape returned after the leading configuration batch."""

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Output dtype, or ``None`` until a callable source is evaluated."""

    @property
    def device(self) -> torch.device:
        """Device on which evaluations are performed."""

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Evaluates configurations in their original order.

        Parameters
        ----------
        configurations : ConfigurationBatch
            A batch containing one discrete index or coordinate per source
            site.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch, *output_shape)`` on the source device
            and with its declared or inferred dtype. Evaluation retains the
            source's autograd behavior; diagnostic counters do not change the
            returned values.
        """


@runtime_checkable
class FiberTensorSource(TensorSource, Protocol):
    """Optional source capability for evaluating one varying site."""

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates one varying input site for every base configuration.

        Parameters
        ----------
        configurations : ConfigurationBatch
            Base configurations defining all sites except the varying site.
        site : int
            Zero-based input site replaced by the fiber values.
        values : torch.Tensor, optional
            Candidate indices or coordinates, with the candidate axis first. If
            omitted, discrete sources enumerate ``range(in_dim[site])``.
            Coordinate fibers require explicit values.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_values, *output_shape)``. The base batch and
            candidate axis are independent and preserve their original orders.
        """


class _SourceEvaluationTracker:
    """Adds inexpensive cumulative evaluation counters to built-in sources."""

    @property
    def evaluation_stats(self) -> EvaluationStats:
        """Cumulative point-evaluation counters for this source."""
        return EvaluationStats(
            requested_points=self._requested_points,
            unique_points=self._unique_points,
            batches=self._evaluation_batches,
            cache_hits=self._cache_hits,
            source_calls=self._source_calls)

    def _initialize_evaluation_stats(self) -> None:
        self._requested_points = 0
        self._unique_points = 0
        self._evaluation_batches = 0
        self._cache_hits = 0
        self._source_calls = 0

    def _record_evaluation(self,
                           points: int,
                           batches: int = 1,
                           unique_points: Optional[int] = None,
                           cache_hits: int = 0) -> None:
        """Records one successful source query without tensor
        synchronization."""
        if unique_points is None:
            unique_points = points
        self._requested_points += points
        self._unique_points += unique_points
        self._evaluation_batches += batches
        self._cache_hits += cache_hits
        self._source_calls += 1

    def reset_evaluation_stats(self) -> None:
        """Resets cumulative point-evaluation counters to zero."""
        self._initialize_evaluation_stats()


def _normalize_in_dim(in_dim: Sequence[int]) -> Tuple[int, ...]:
    """Validates and normalizes a discrete input dimension."""
    if isinstance(in_dim, (str, bytes)):
        raise TypeError('`in_dim` should be a sequence of integers')
    try:
        normalized = tuple(in_dim)
    except TypeError as exc:
        raise TypeError(
            '`in_dim` should be a sequence of integers') from exc
    if not normalized:
        raise ValueError('`in_dim` should contain at least one site')
    if any((not isinstance(dim, int)) or (dim < 1) for dim in normalized):
        raise ValueError('`in_dim` should contain positive integers')
    return normalized


def _discrete_indices(configurations: ConfigurationBatch,
                      in_dim: Sequence[int],
                      device: torch.device) -> torch.Tensor:
    """Validates discrete configurations and returns packed long indices."""
    if not isinstance(configurations, ConfigurationBatch):
        raise TypeError(
            '`configurations` should be ConfigurationBatch type')
    if configurations.kind != 'indices':
        raise ValueError('This source requires discrete index configurations')
    if configurations.n_sites != len(in_dim):
        raise ValueError(
            'Configurations should contain one value per input site')
    indices = configurations.as_tensor().to(device=device, dtype=torch.long)
    for site, dim in enumerate(in_dim):
        if torch.any(indices[:, site] < 0) or \
                torch.any(indices[:, site] >= dim):
            raise ValueError(
                f'Configuration indices at site {site} are out of bounds')
    return indices


def _ravel_indices(indices: torch.Tensor,
                   in_dim: Sequence[int]) -> torch.Tensor:
    """Converts global multi-indices to flat row ids."""
    strides = []
    for site in range(len(in_dim)):
        strides.append(prod(in_dim[site + 1:]))
    stride_tensor = indices.new_tensor(strides)
    return (indices * stride_tensor).sum(dim=1)


def _unravel_indices(flat_ids: torch.Tensor,
                     in_dim: Sequence[int]) -> torch.Tensor:
    """Converts flat row ids to global multi-indices."""
    remainder = flat_ids
    sites = []
    for site, dim in enumerate(in_dim):
        stride = prod(in_dim[site + 1:])
        sites.append(torch.div(remainder, stride, rounding_mode='floor'))
        remainder = torch.remainder(remainder, stride) if stride > 1 \
            else torch.zeros_like(remainder)
        sites[-1] = torch.remainder(sites[-1], dim)
    return torch.stack(sites, dim=1)


def _fiber_configurations(
        configurations: ConfigurationBatch,
        site: int,
        values: torch.Tensor) -> Tuple[ConfigurationBatch, int]:
    """Expands packed base configurations over a one-site value grid."""
    if not isinstance(site, int):
        raise TypeError('`site` should be int type')
    if (site < 0) or (site >= configurations.n_sites):
        raise ValueError('`site` should identify an input site')
    if not configurations.packed:
        raise ValueError(
            'Generic fiber expansion requires packed configurations')
    if not isinstance(values, torch.Tensor):
        raise TypeError('`values` should be torch.Tensor type')
    if values.ndim < 1:
        raise ValueError('`values` should have a leading grid dimension')
    if (configurations.kind == 'indices') and (
            (values.ndim != 1) or (values.dtype not in _INTEGER_DTYPES)):
        raise TypeError(
            'Discrete fiber `values` should be a one-dimensional integer '
            'tensor')

    packed = configurations.as_tensor()
    if tuple(values.shape[1:]) != tuple(packed.shape[2:]):
        raise ValueError(
            '`values` coordinate shape should match the selected site')
    values = values.to(device=packed.device, dtype=packed.dtype)
    n_values = values.shape[0]
    expanded = packed.unsqueeze(1).expand(
        packed.shape[0], n_values, *packed.shape[1:]).clone()
    expanded[:, :, site] = values.unsqueeze(0)
    expanded = expanded.reshape(
        packed.shape[0] * n_values, *packed.shape[1:])
    return ConfigurationBatch(expanded, kind=configurations.kind), n_values


__all__ = [
    'ConfigurationBatch',
    'TensorSource',
    'FiberTensorSource',
]
