"""
This script contains:

    Public classes:
        * ALSProblem

    Internal functions:
        * _zero_safe_relative_error
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from tensorkrowch.utils import safe_divide

from tensorkrowch.decompositions._utils import _ravel_indices
from tensorkrowch.decompositions.sources import (BaseSource,
                                                 ConfigurationBatch,
                                                 SparseTensorSource)
from tensorkrowch.decompositions.sources.base import _discrete_indices


def _zero_safe_relative_error(absolute: torch.Tensor,
                              denominator: torch.Tensor) -> torch.Tensor:
    """Returns a relative error with explicit zero-target semantics."""
    return safe_divide(absolute, denominator,
                       zero_value=torch.where(absolute == 0, 0, torch.inf))


@dataclass
class ALSProblem:  # MARK: ALSProblem
    r"""
    Defines the values and weighted L2 objective of an ALS fit.

    ``source`` describes a known finite tensor. ``observations`` instead uses
    the stored support of a :class:`SparseTensorSource` as the permanent set
    of known entries. In that objective, entries outside the support are
    unknown and never queried. This interpretation belongs to the problem;
    an ordinary sparse source still represents zeros outside its support.

    Residual weights multiply the differences before their squared norm is
    taken: :math:`\|W(\widehat{T}-T)\|_2^2`. A mask gives zero weight to
    excluded entries. Sampling is performed by the backend; ``sampled`` only
    declares whether errors refer to a changing sampled objective.

    Parameters
    ----------
    source : BaseSource, optional
        Finite scalar tensor source. Required without ``observations``.
    observations : SparseTensorSource, optional
        Stored scalar observations. Their order is ``support`` order, including
        explicitly stored zero values. Sparse construction coalesces repeated
        indices by summing their values before this objective is formed.
    weights : torch.Tensor, optional
        Non-negative multiplicative residual weights. Shape is ``in_dim`` for
        a known source or ``(n_observations,)`` in stored support order for an
        observed objective.
    mask : BaseSource, optional
        Boolean scalar tensor source with the same dimensions and device as
        ``source``. ``True`` entries participate. Use either a mask or an
        explicit observation objective.
    sampled : bool
        Whether a backend selects renewable samples from a known source.
        Defaults to ``False``. Observations always define a fixed objective.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(torch.ones(2, 3))
    >>> problem = tk.decompositions.ALSProblem(source)
    >>> problem.in_dim, problem.has_fixed_objective
    ((2, 3), True)
    """

    source: Optional[BaseSource] = None
    observations: Optional[SparseTensorSource] = None
    weights: Optional[torch.Tensor] = None
    mask: Optional[BaseSource] = None
    sampled: bool = False

    def __post_init__(self) -> None:
        if (self.source is None) and (self.observations is None):
            raise ValueError('`source` or `observations` should be provided')
        if self.source is not None:
            if not isinstance(self.source, BaseSource):
                raise TypeError('`source` should inherit from BaseSource')
            if not self.source.is_discrete_tensor or \
                    (self.source.out_shape not in (None, ())):
                raise ValueError('`source` should represent a scalar tensor')
            if self.source.n_batches:
                raise ValueError('`source` should not have structural batches')
        if not isinstance(self.sampled, bool):
            raise TypeError('`sampled` should be bool type')

        if self.observations is not None:
            if not isinstance(self.observations, SparseTensorSource):
                raise TypeError('`observations` should be SparseTensorSource')
            if self.observations.is_matrix or self.observations.out_shape or \
                    (self.observations.in_sites != tuple(
                        range(self.observations.n_sites))):
                raise ValueError(
                    '`observations` should store scalar tensor entries in '
                    'site order')
            values = self.observations.support_values
            if not values.numel():
                raise ValueError(
                    '`observations` should contain stored entries')
            if not (values.is_floating_point() or values.is_complex()):
                raise TypeError(
                    '`observations` should have floating or complex values')
            if not torch.isfinite(values).all():
                raise ValueError('`observations` should contain finite values')
            if self.sampled or (self.mask is not None):
                raise ValueError(
                    '`observations` cannot be combined with `sampled` or'
                    ' `mask`')
            if (self.source is not None) and (
                    (self.source.in_dim != self.observations.in_dim) or
                    (self.source.device != self.observations.device)):
                raise ValueError(
                    '`source` and `observations` should share dimensions and '
                    'device')

        reference = self.observations if self.observations is not None \
            else self.source
        if self.mask is not None:
            if not isinstance(self.mask, BaseSource):
                raise TypeError('`mask` should inherit from BaseSource')
            if not self.mask.is_discrete_tensor or \
                    (self.mask.in_dim != self.in_dim) or \
                    self.mask.out_shape or \
                    self.mask.n_batches:
                raise ValueError(
                    '`mask` should have the scalar source tensor dimensions')
            if self.mask.device != reference.device:
                raise ValueError('`mask` and `source` should share a device')
            indices = torch.zeros((1, len(self.in_dim)), dtype=torch.long,
                                  device=reference.device)
            values = self.mask.evaluate(ConfigurationBatch(indices))
            if (values.dtype != torch.bool) or (values.shape != (1,)):
                raise TypeError('`mask` should return one boolean per entry')

        if self.weights is not None:
            if not isinstance(self.weights, torch.Tensor):
                raise TypeError('`weights` should be torch.Tensor type')
            expected_shape = self.in_dim if self.observations is None else \
                tuple(self.observations.support_values.shape)
            if self.weights.shape != expected_shape:
                raise ValueError(
                    '`weights` should match the source tensor shape or the '
                    'stored observation values')
            if self.weights.device != reference.device:
                raise ValueError('`weights` should be on the target device')
            if not self.weights.is_floating_point():
                raise TypeError('`weights` should have a real floating dtype')
            if not torch.isfinite(self.weights).all() or \
                    torch.any(self.weights < 0):
                raise ValueError('`weights` should be finite and non-negative')

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Dimensions of all represented tensor variables."""
        if self.source is not None:
            return self.source.in_dim
        return self.observations.in_dim

    @property
    def out_shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of each evaluated value, when already known."""
        if self.source is not None:
            return self.source.out_shape
        return self.observations.out_shape

    @property
    def has_fixed_objective(self) -> bool:
        """Whether complete-sweep errors refer to the same objective."""
        return not self.sampled

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Evaluates a known source at the supplied configurations.

        Parameters
        ----------
        configurations : ConfigurationBatch
            Discrete indices in the source's ``in_sites`` order.

        Returns
        -------
        torch.Tensor
            One target value per configuration. Observation-only problems
            expose their targets through ``observations.support_values``.
        """
        if self.source is None:
            raise ValueError(
                'Observation targets are available only on stored support')
        return self.source.evaluate(configurations)

    def entry_weights(self, indices: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Returns residual weights for selected scalar tensor entries.

        Parameters
        ----------
        indices : torch.Tensor
            Integer configurations with shape ``(n_entries, n_sites)``.
            Observed problems require indices on their stored support.

        Returns
        -------
        torch.Tensor or None
            One weight per entry, or ``None`` for an unweighted objective.
        """
        weights = None
        if self.weights is not None:
            if self.observations is None:
                weights = self.weights[tuple(indices.unbind(1))]
            else:
                stored = _ravel_indices(
                    self.observations.support.values, self.in_dim)
                queried = _ravel_indices(indices, self.in_dim)
                positions = torch.searchsorted(stored, queried)
                valid = positions < stored.numel()
                positions = positions.clamp_max(stored.numel() - 1)
                if not torch.all(valid & (stored[positions] == queried)):
                    raise ValueError(
                        '`indices` should belong to observed support')
                weights = self.weights[positions]
        if self.mask is not None:
            selected = self.mask.evaluate(ConfigurationBatch(indices))
            weights = selected if weights is None else weights * selected
        return weights

    def objective_error(self,
                        approximation: torch.Tensor,
                        configurations: Optional[
                            ConfigurationBatch] = None) -> Tuple[
            torch.Tensor, torch.Tensor]:
        """
        Computes absolute and relative weighted L2 residual norms.

        Parameters
        ----------
        approximation : torch.Tensor
            Approximate values in configuration order or stored observation
            order, with one scalar value per entry.
        configurations : ConfigurationBatch, optional
            Configurations for a known source objective. Omit for observations.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Absolute residual norm and its ratio to the weighted target norm.
            A zero target gives zero relative error for an exact fit and
            infinity otherwise.
        """
        if not isinstance(approximation, torch.Tensor):
            raise TypeError('`approximation` should be torch.Tensor type')
        if self.observations is not None:
            if configurations is not None:
                raise ValueError('Observed objectives use their fixed support')
            target = self.observations.support_values
            weights = self.weights
        else:
            if configurations is None:
                raise ValueError(
                    '`configurations` are required for a source objective')
            target = self.source.evaluate(configurations)
            indices = _discrete_indices(configurations, self.in_dim,
                                        self.source.device)
            weights = self.entry_weights(indices)
        if approximation.shape != target.shape:
            raise ValueError('`approximation` should match target values')
        if approximation.device != target.device:
            raise ValueError('`approximation` should be on the target device')
        residual = approximation - target
        if weights is not None:
            residual = residual * weights
            target = target * weights
        absolute = torch.linalg.vector_norm(residual)
        relative = _zero_safe_relative_error(
            absolute, torch.linalg.vector_norm(target))
        return absolute, relative


__all__ = ['ALSProblem']
