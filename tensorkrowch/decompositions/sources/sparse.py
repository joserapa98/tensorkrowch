"""Sparse tensor sources and empirical distributions."""

from typing import Optional, Sequence, Tuple

import torch

from tensorkrowch.decompositions.sources.base import (
    ConfigurationBatch,
    _discrete_indices,
    _fiber_configurations,
    _normalize_input_dim,
    _ravel_indices,
    _SourceEvaluationTracker,
    _unravel_indices,
)


class SparseTensorSource(_SourceEvaluationTracker):
    """Sparse tensor source with declared zeros outside its support.

    Repeated support indices are coalesced by summing their values, matching
    sparse COO semantics. In contrast with matrix/tensor completion,
    configurations absent from the support are known to have value zero.

    Parameters
    ----------
    indices : torch.Tensor
        Integer tensor of shape ``(nnz, sites)``.
    values : torch.Tensor
        Non-zero values with shape ``(nnz, *output_shape)``.
    input_dim : sequence of int
        Complete discrete input dimension.
    """

    def __init__(self,
                 indices: torch.Tensor,
                 values: torch.Tensor,
                 input_dim: Sequence[int]) -> None:
        self._initialize_evaluation_stats()
        if not isinstance(indices, torch.Tensor):
            raise TypeError('`indices` should be torch.Tensor type')
        if not isinstance(values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        if (indices.ndim != 2) or (indices.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64)):
            raise TypeError(
                '`indices` should be a two-dimensional integer tensor')
        if values.ndim < 1:
            raise ValueError('`values` should have a leading support dimension')
        if values.shape[0] != indices.shape[0]:
            raise ValueError(
                '`values` and `indices` should have the same leading size')
        if values.device != indices.device:
            raise ValueError('`values` and `indices` should share a device')
        if not (values.is_floating_point() or values.is_complex()):
            raise TypeError('`values` should have a floating or complex dtype')

        self._input_dim = _normalize_input_dim(input_dim)
        if indices.shape[1] != len(self.input_dim):
            raise ValueError('`indices` should contain one column per site')
        indices = _discrete_indices(
            ConfigurationBatch(indices, kind='indices'),
            self.input_dim,
            values.device)

        flat_ids = _ravel_indices(indices, self.input_dim)
        unique_ids, inverse = torch.unique(
            flat_ids, sorted=True, return_inverse=True)
        coalesced_values = values.new_zeros(
            (unique_ids.numel(), *values.shape[1:]))
        coalesced_values.index_add_(0, inverse, values)

        self._flat_ids = unique_ids
        self._indices = _unravel_indices(unique_ids, self.input_dim)
        self._values = coalesced_values
        self._output_shape = tuple(values.shape[1:])

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Discrete input dimension at every site."""
        return self._input_dim

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Shape stored at every sparse support entry."""
        return self._output_shape

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of sparse values."""
        return self._values.dtype

    @property
    def device(self) -> torch.device:
        """Device of sparse indices and values."""
        return self._values.device

    @property
    def support(self) -> ConfigurationBatch:
        """Unique coalesced support configurations."""
        return ConfigurationBatch(self._indices, kind='indices')

    @property
    def support_values(self) -> torch.Tensor:
        """Values associated with the unique support configurations."""
        return self._values

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Evaluates sparse values, returning zero outside the support."""
        indices = _discrete_indices(
            configurations, self.input_dim, self.device)
        flat_ids = _ravel_indices(indices, self.input_dim)
        positions = torch.searchsorted(self._flat_ids, flat_ids)
        safe_positions = positions.clamp(max=max(self._flat_ids.numel() - 1,
                                                 0))
        result = self._values.new_zeros(
            (flat_ids.numel(), *self.output_shape))
        if self._flat_ids.numel():
            matched = (positions < self._flat_ids.numel()) & \
                (self._flat_ids.index_select(0, safe_positions) == flat_ids)
            result[matched] = self._values.index_select(
                0, safe_positions[matched])
        self._record_evaluation(points=indices.shape[0])
        return result

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates a sparse discrete fiber without densifying the source."""
        if not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= len(self.input_dim)):
            raise ValueError('`site` should identify an input site')
        if values is None:
            values = torch.arange(
                self.input_dim[site], device=configurations.device)
        expanded, n_values = _fiber_configurations(
            configurations, site, values)
        result = self.evaluate(expanded)
        return result.reshape(
            configurations.batch_size, n_values, *self.output_shape)


class EmpiricalDistribution(SparseTensorSource):
    """Normalized sparse empirical distribution built from a dataset.

    Parameters
    ----------
    dataset : torch.Tensor
        Integer observations with shape ``(samples, sites)``.
    input_dim : sequence of int, optional
        Complete discrete input dimension. If omitted, each dimension is one
        plus the largest observed index.
    weights : torch.Tensor, optional
        Non-negative mass assigned to every observation before normalization.
        Repeated observations are accumulated.
    dtype : torch.dtype, optional
        Probability dtype used when ``weights`` is omitted or integral.
    """

    def __init__(self,
                 dataset: torch.Tensor,
                 input_dim: Optional[Sequence[int]] = None,
                 weights: Optional[torch.Tensor] = None,
                 dtype: Optional[torch.dtype] = None) -> None:
        if not isinstance(dataset, torch.Tensor):
            raise TypeError('`dataset` should be torch.Tensor type')
        if (dataset.ndim != 2) or (dataset.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64)):
            raise TypeError(
                '`dataset` should be a two-dimensional integer tensor')
        if dataset.shape[0] < 1:
            raise ValueError('`dataset` should contain at least one sample')
        if input_dim is None:
            if torch.any(dataset < 0):
                raise ValueError('`dataset` indices should be non-negative')
            input_dim = tuple(
                int(dataset[:, site].max().item()) + 1
                for site in range(dataset.shape[1]))
        normalized_input_dim = _normalize_input_dim(input_dim)

        if dtype is None:
            dtype = torch.get_default_dtype()
        if not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type')
        if weights is None:
            mass = torch.ones(
                dataset.shape[0], device=dataset.device, dtype=dtype)
        else:
            if not isinstance(weights, torch.Tensor):
                raise TypeError('`weights` should be torch.Tensor type')
            if weights.shape != (dataset.shape[0],):
                raise ValueError('`weights` should contain one value per sample')
            if weights.device != dataset.device:
                raise ValueError('`weights` and `dataset` should share a device')
            if not (weights.is_floating_point() or weights.is_complex()):
                mass = weights.to(dtype=dtype)
            else:
                mass = weights
            if mass.is_complex():
                raise TypeError('`weights` should have a real dtype')
            if (not torch.isfinite(mass).all()) or torch.any(mass < 0):
                raise ValueError('`weights` should be finite and non-negative')
        total = mass.sum()
        if (not torch.isfinite(total)) or (total <= 0):
            raise ValueError('The total empirical mass should be positive')

        super().__init__(
            indices=dataset,
            values=mass / total,
            input_dim=normalized_input_dim)


__all__ = ['SparseTensorSource', 'EmpiricalDistribution']
