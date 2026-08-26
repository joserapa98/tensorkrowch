"""ALS target problems and fixed observed entries."""

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import torch

from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 TensorSource)
from tensorkrowch.decompositions.sources.base import (_discrete_indices,
                                                      _normalize_input_dim,
                                                      _ravel_indices)


def _zero_safe_relative_error(absolute: torch.Tensor,
                              denominator: torch.Tensor) -> torch.Tensor:
    """Returns a relative error with explicit zero-target semantics."""
    safe_denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator))
    ratio = absolute / safe_denominator
    zero_target = torch.where(
        absolute == 0,
        torch.zeros_like(absolute),
        torch.full_like(absolute, torch.inf))
    return torch.where(denominator > 0, ratio, zero_target)


@dataclass(frozen=True)
class ObservedEntries:
    """Fixed entries defining a matrix/tensor completion objective.

    Repeated indices are deduplicated only when their values and weights are
    identical. Values outside these global indices remain unknown; they are
    not interpreted as zeros.

    Parameters
    ----------
    indices : torch.Tensor
        Global integer multi-indices with shape ``(observations, sites)``.
    values : torch.Tensor
        Observed target values with shape
        ``(observations, *output_shape)``.
    input_dim : sequence of int
        Complete discrete input dimension.
    weights : torch.Tensor, optional
        Non-negative multiplicative weights ``W`` in the observed objective.
    """

    indices: torch.Tensor
    values: torch.Tensor
    input_dim: Sequence[int]
    weights: Optional[torch.Tensor] = None
    flat_ids: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.indices, torch.Tensor):
            raise TypeError('`indices` should be torch.Tensor type')
        if not isinstance(self.values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        if self.values.ndim < 1:
            raise ValueError('`values` should have an observation dimension')
        if self.values.shape[0] != self.indices.shape[0]:
            raise ValueError(
                '`indices` and `values` should contain the same observations')
        if self.values.device != self.indices.device:
            raise ValueError('`indices` and `values` should share a device')
        if not (self.values.is_floating_point() or self.values.is_complex()):
            raise TypeError('`values` should have a floating or complex dtype')
        if self.indices.shape[0] < 1:
            raise ValueError('At least one observed entry is required')

        input_dim = _normalize_input_dim(self.input_dim)
        indices = _discrete_indices(
            ConfigurationBatch(self.indices, kind='indices'),
            input_dim,
            self.values.device)

        weights = self.weights
        if weights is not None:
            if not isinstance(weights, torch.Tensor):
                raise TypeError('`weights` should be torch.Tensor type')
            if weights.shape != (indices.shape[0],):
                raise ValueError(
                    '`weights` should contain one value per observation')
            if weights.device != self.values.device:
                raise ValueError(
                    '`weights` and observed values should share a device')
            if (not weights.is_floating_point()) or weights.is_complex():
                raise TypeError('`weights` should have a real floating dtype')
            if (not torch.isfinite(weights).all()) or torch.any(weights < 0):
                raise ValueError('`weights` should be finite and non-negative')

        flat_ids = _ravel_indices(indices, input_dim)
        order = torch.argsort(flat_ids)
        flat_ids = flat_ids.index_select(0, order)
        indices = indices.index_select(0, order)
        values = self.values.index_select(0, order)
        if weights is not None:
            weights = weights.index_select(0, order)

        selected = []
        start = 0
        while start < flat_ids.numel():
            stop = start + 1
            while (stop < flat_ids.numel()) and \
                    (flat_ids[stop] == flat_ids[start]):
                stop += 1
            if stop > (start + 1):
                repeated_values = values[start:stop]
                if not torch.equal(
                        repeated_values,
                        repeated_values[0].expand_as(repeated_values)):
                    raise ValueError(
                        'Repeated observations should have identical values')
                if weights is not None:
                    repeated_weights = weights[start:stop]
                    if not torch.equal(
                            repeated_weights,
                            repeated_weights[0].expand_as(repeated_weights)):
                        raise ValueError(
                            'Repeated observations should have identical weights')
            selected.append(start)
            start = stop

        selected_tensor = torch.tensor(
            selected, device=indices.device, dtype=torch.long)
        object.__setattr__(self, 'indices', indices.index_select(
            0, selected_tensor))
        object.__setattr__(self, 'values', values.index_select(
            0, selected_tensor))
        object.__setattr__(self, 'input_dim', input_dim)
        object.__setattr__(self, 'flat_ids', flat_ids.index_select(
            0, selected_tensor))
        if weights is not None:
            object.__setattr__(self, 'weights', weights.index_select(
                0, selected_tensor))

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Shape stored at every observed index."""
        return tuple(self.values.shape[1:])

    @property
    def configurations(self) -> ConfigurationBatch:
        """Observed global indices as an immutable configuration batch."""
        return ConfigurationBatch(self.indices, kind='indices')

    def error(self,
              approximation: torch.Tensor) -> Tuple[torch.Tensor,
                                                     torch.Tensor]:
        """Returns absolute and relative weighted observed errors.

        The errors are

        ``absolute = ||W * P_omega(approximation - target)||`` and
        ``relative = absolute / ||W * P_omega(target)||``.
        """
        if not isinstance(approximation, torch.Tensor):
            raise TypeError('`approximation` should be torch.Tensor type')
        if approximation.shape != self.values.shape:
            raise ValueError(
                '`approximation` should match the observed value shape')
        if approximation.device != self.values.device:
            raise ValueError(
                '`approximation` and observed values should share a device')

        residual = approximation - self.values
        target = self.values
        if self.weights is not None:
            shape = (self.weights.shape[0],) + (1,) * len(self.output_shape)
            weights = self.weights.reshape(shape)
            residual = residual * weights
            target = target * weights
        absolute = torch.linalg.vector_norm(residual)
        denominator = torch.linalg.vector_norm(target)
        relative = _zero_safe_relative_error(absolute, denominator)
        return absolute, relative


@dataclass
class ALSProblem:
    """Composition of a tensor source and its ALS objective semantics.

    A source supplies known tensor/function values. ``ObservedEntries`` instead
    defines a completion objective and may be used without a source because
    values outside the observed set are unknown. Sampling strategies are added
    separately and do not change either meaning.
    """

    source: Optional[TensorSource] = None
    observations: Optional[ObservedEntries] = None
    selector: Optional[Any] = None
    weights: Optional[torch.Tensor] = None
    loss: str = 'l2'

    def __post_init__(self) -> None:
        if (self.source is None) and (self.observations is None):
            raise ValueError('`source` or `observations` should be provided')
        if (self.source is not None) and \
                (not isinstance(self.source, TensorSource)):
            raise TypeError('`source` should implement TensorSource')
        if (self.observations is not None) and \
                (not isinstance(self.observations, ObservedEntries)):
            raise TypeError(
                '`observations` should be ObservedEntries type')
        if self.loss != 'l2':
            raise ValueError("`loss` should currently be 'l2'")

        if (self.source is not None) and (self.observations is not None):
            if self.source.input_dim != self.observations.input_dim:
                raise ValueError(
                    'Source and observations should have matching input '
                    'dimensions')
            if (self.source.output_shape is not None) and \
                    (self.source.output_shape !=
                     self.observations.output_shape):
                raise ValueError(
                    'Source and observations should have matching output shapes')

        if self.weights is not None:
            if self.observations is not None:
                raise ValueError(
                    'Completion weights belong to `ObservedEntries`')
            if not isinstance(self.weights, torch.Tensor):
                raise TypeError('`weights` should be torch.Tensor type')
            output_shape = self.source.output_shape
            if output_shape is None:
                raise ValueError(
                    'Source output shape is required for global weights')
            expected_shape = (*self.source.input_dim, *output_shape)
            if self.weights.shape != expected_shape:
                raise ValueError(
                    '`weights` should match the complete source tensor shape')
            if self.weights.device != self.source.device:
                raise ValueError('`weights` should be on the source device')
            if (not self.weights.is_floating_point()) or \
                    self.weights.is_complex():
                raise TypeError('`weights` should have a real floating dtype')
            if (not torch.isfinite(self.weights).all()) or \
                    torch.any(self.weights < 0):
                raise ValueError('`weights` should be finite and non-negative')

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Input dimension defined by the source or observations."""
        if self.source is not None:
            return self.source.input_dim
        return self.observations.input_dim

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """Target output shape when already known."""
        if self.source is not None:
            return self.source.output_shape
        return self.observations.output_shape

    @property
    def has_fixed_objective(self) -> bool:
        """Whether errors remain comparable between complete sweeps."""
        return (self.observations is not None) or \
            (self.source is not None and self.selector is None)

    def evaluate(self,
                 configurations: ConfigurationBatch) -> torch.Tensor:
        """Evaluates a known source target on configurations."""
        if self.source is None:
            raise ValueError(
                'Completion targets are known only at observed entries')
        return self.source.evaluate(configurations)

    def objective_error(
            self,
            approximation: torch.Tensor,
            configurations: Optional[ConfigurationBatch] = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns weighted absolute and relative L2 objective errors."""
        if self.observations is not None:
            if configurations is not None:
                raise ValueError(
                    'Completion objective always uses its fixed observations')
            return self.observations.error(approximation)
        if configurations is None:
            raise ValueError(
                '`configurations` are required for a source objective')

        target = self.source.evaluate(configurations)
        if approximation.shape != target.shape:
            raise ValueError(
                '`approximation` should match evaluated source values')
        residual = approximation - target
        weighted_target = target
        if self.weights is not None:
            indices = _discrete_indices(
                configurations, self.input_dim, self.source.device)
            weights = self.weights[tuple(
                indices[:, site] for site in range(indices.shape[1]))]
            residual = residual * weights
            weighted_target = target * weights
        absolute = torch.linalg.vector_norm(residual)
        denominator = torch.linalg.vector_norm(weighted_target)
        relative = _zero_safe_relative_error(absolute, denominator)
        return absolute, relative


__all__ = ['ALSProblem', 'ObservedEntries']
