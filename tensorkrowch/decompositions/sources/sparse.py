"""
This script contains:

    Public classes:
        * SparseTensorSource

    Internal classes:
        * _SparseLookup
"""

from dataclasses import dataclass, field
from math import prod
from typing import Optional, Sequence, Tuple

import torch

from tensorkrowch.utils import _INTEGER_DTYPES

from tensorkrowch.decompositions._utils import (_normalize_dims,
                                                _ravel_indices,
                                                _unravel_indices)
from tensorkrowch.decompositions.sources.base import (BaseSource,
                                                      ConfigurationBatch,
                                                      _discrete_indices,
                                                      _data_weights)


@dataclass
class _SparseLookup:  # MARK: _SparseLookup
    """Stores coalesced indexed values and performs ordered sparse lookups."""

    indices: torch.Tensor
    values: torch.Tensor
    in_dim: Tuple[int, ...]
    flat_ids: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        flat_ids = _ravel_indices(self.indices, self.in_dim)
        self.flat_ids, inverse = torch.unique(flat_ids, sorted=True,
                                              return_inverse=True)
        values = self.values
        if values.dtype == torch.bool:
            values = values.to(torch.long)
        combined = values.new_zeros((self.flat_ids.numel(), *values.shape[1:]))
        self.values = combined.index_add(0, inverse, values).to(self.values.dtype)
        self.indices = _unravel_indices(self.flat_ids, self.in_dim)

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        """Returns stored values or zero for absent configurations."""
        flat_ids = _ravel_indices(indices, self.in_dim)
        positions = torch.searchsorted(self.flat_ids, flat_ids)
        result = self.values.new_zeros((flat_ids.numel(), *self.values.shape[1:]))
        if self.flat_ids.numel():
            safe = positions.clamp(max=self.flat_ids.numel() - 1)
            matched = (positions < self.flat_ids.numel()) & (
                self.flat_ids.index_select(0, safe) == flat_ids)
            result[matched] = self.values.index_select(0, safe[matched])
        return result


class SparseTensorSource(BaseSource):  # MARK: SparseTensorSource
    """
    Sparse tensor source with declared zeros outside its support.

    Repeated support indices are coalesced by summing their values, matching
    sparse COO semantics. In contrast with matrix/tensor completion,
    configurations absent from the support are known to have value zero.

    Parameters
    ----------
    indices : torch.Tensor
        Integer tensor of shape ``(nnz, sites)``.
    values : torch.Tensor
        Non-zero values with shape ``(nnz, *out_shape)``.
    in_dim : sequence of int
        Complete discrete input dimension.
    """

    def __init__(self,
                 indices: torch.Tensor,
                 values: torch.Tensor,
                 in_dim: Sequence[int]) -> None:
        if not isinstance(indices, torch.Tensor):
            raise TypeError('`indices` should be torch.Tensor type')
        if not isinstance(values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        if (indices.ndim != 2) or (indices.dtype not in _INTEGER_DTYPES):
            raise TypeError(
                '`indices` should be a two-dimensional integer tensor')
        if values.ndim < 1:
            raise ValueError('`values` should have a leading support dimension')
        if values.shape[0] != indices.shape[0]:
            raise ValueError(
                '`values` and `indices` should have the same leading size')
        if values.device != indices.device:
            raise ValueError('`values` and `indices` should share a device')
        if not (values.is_floating_point() or values.is_complex() or values.dtype == torch.bool):
            raise TypeError('`values` should have a floating, complex or boolean dtype')

        self._in_dim = _normalize_dims(in_dim, 'in_dim')
        super().__init__(len(self._in_dim))
        if indices.shape[1] != len(self._in_dim):
            raise ValueError('`indices` should contain one column per site')
        indices = _discrete_indices(
            ConfigurationBatch(indices, kind='indices'),
            self._in_dim,
            values.device)

        self._lookup = _SparseLookup(indices, values, self._in_dim)
        self._indices, self._values = self._lookup.indices, self._lookup.values
        self._flat_ids = self._lookup.flat_ids
        self._out_shape = tuple(values.shape[1:])

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Discrete input dimension at every site."""
        return self._in_dim

    @property
    def out_shape(self) -> Tuple[int, ...]:
        """Shape stored at every sparse support entry."""
        return self._out_shape

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
            configurations, self._in_dim, self.device)
        if self.is_matrix:
            outputs = _unravel_indices(torch.arange(prod(self.out_dim),
                                                    device=self.device), self.out_dim)
            inputs = indices[:, None].expand(-1, outputs.shape[0], -1)
            outputs = outputs[None].expand(indices.shape[0], -1, -1)
            paired = torch.stack((inputs, outputs), -1).flatten(-2)
            result = self._lookup(paired.reshape(-1, paired.shape[-1]))
            result = result.reshape(indices.shape[0], *self.out_dim)
        else:
            result = self._lookup(indices)
        self._record_evaluation(points=indices.shape[0])
        return result

    @classmethod
    def from_data(cls,
                  dataset: torch.Tensor,
                  in_dim: Optional[Sequence[int]] = None,
                  weights: Optional[torch.Tensor] = None,
                  dtype: Optional[torch.dtype] = None) -> 'SparseTensorSource':
        """
        Constructs a discrete empirical distribution from sample indices.

        Parameters
        ----------
        dataset : torch.Tensor
            Integer samples with shape ``(samples, sites)``. Repeats accumulate.
        in_dim : sequence[int], optional
            Complete dimensions; inferred from observed maxima when omitted.
        weights : torch.Tensor, optional
            Non-negative sample weights, normalized to total mass one.
        dtype : torch.dtype, optional
            Dtype for uniform or integral weights.

        Returns
        -------
        SparseTensorSource
            Vector source with known zeros outside the observed support.

        Examples
        --------
        >>> source = tk.decompositions.SparseTensorSource.from_data(
        ...     torch.tensor([[0], [2], [0]]), in_dim=(3,))
        >>> source.contract_dense()
        tensor([0.6667, 0.0000, 0.3333])
        """
        weights = _data_weights(dataset, weights, dtype)
        if dataset.dtype not in _INTEGER_DTYPES:
            raise TypeError('`dataset` should contain integer indices')
        if torch.any(dataset < 0):
            raise ValueError('`dataset` indices should be non-negative')
        if in_dim is None:
            in_dim = tuple(int(value) + 1 for value in dataset.amax(0))
        return cls(dataset, weights, in_dim)

    def contract_dense(self) -> torch.Tensor:
        """Materializes the tensor by placing its stored entries into zeros."""
        shape = self._entry_dim if self.is_matrix else (*self._in_dim, *self.out_shape)
        tensor = self._values.new_zeros(shape)
        tensor[tuple(self._indices.T)] = self._values
        return tensor

    @classmethod
    def matrix(cls, indices: torch.Tensor, values: torch.Tensor,
               in_dim: Sequence[int], out_dim: Sequence[int]) -> 'SparseTensorSource':
        """
        Constructs a matrix source from interleaved entry indices.

        Parameters
        ----------
        indices : torch.Tensor
            Integer pairs with shape ``(entries, 2 * sites)``.
        values : torch.Tensor
            Scalar entry values.
        in_dim, out_dim : sequence[int]
            Dimensions of each paired matrix site.

        Returns
        -------
        SparseTensorSource
            Source retaining paired axes for matrix decomposition.
        """
        inputs = _normalize_dims(in_dim, 'in_dim')
        outputs = _normalize_dims(out_dim, 'out_dim')
        if len(inputs) != len(outputs):
            raise ValueError('`in_dim` and `out_dim` should have equal lengths')
        source = cls(indices, values, tuple(dim for pair in zip(inputs, outputs)
                                            for dim in pair))
        source._entry_dim = source._in_dim
        source._matrix_dims = inputs, outputs
        source._in_dim, source._out_shape = inputs, outputs
        source._n_sites = len(inputs)
        return source


__all__ = ['SparseTensorSource']
