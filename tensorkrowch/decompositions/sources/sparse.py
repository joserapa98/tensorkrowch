"""
This script contains:

    Internal classes:
        * _SparseLookup

    Public classes:
        * SparseTensorSource
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
        self.flat_ids, inverse = torch.unique(flat_ids,
                                              sorted=True,
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
        result = self.values.new_zeros((flat_ids.numel(),
                                        *self.values.shape[1:]))
        if self.flat_ids.numel():
            safe = positions.clamp(max=self.flat_ids.numel() - 1)
            matched = (positions < self.flat_ids.numel()) & (
                self.flat_ids.index_select(0, safe) == flat_ids)
            result[matched] = self.values.index_select(0, safe[matched])
        return result


class SparseTensorSource(BaseSource):  # MARK: SparseTensorSource
    """
    Sparse tensor source with declared zeros outside its support.

    The usual constructor represents a tensor with one axis per vector site.
    The source's ``in_dim`` contains every site's dimension. ``n_sites`` counts
    all sites, including those returned as outputs. ``in_sites`` selects the
    evaluated sites, in configuration order; the remaining axes form
    ``out_shape`` in their original order, as in :class:`DenseTensorSource`.

    Columns of ``indices`` and evaluation configurations follow ``in_sites``
    order. Trailing axes of ``values`` are output axes, in the original order
    of the remaining sites. By default, inputs occupy the first sites.

    Repeated indices are coalesced by summing their values, matching sparse
    COO semantics; boolean values are combined by logical OR. Absent entries
    are known zeros, rather than missing observations.

    Use :meth:`matrix` for scalar entries with paired input/output indices,
    or :meth:`from_data` to construct a vector source representing a normalized
    empirical distribution. The usual constructor and :meth:`matrix` preserve
    the supplied values without normalizing them.

    Parameters
    ----------
    indices : torch.Tensor
        Integer indices of shape ``(n_entries, len(in_sites))``, with columns
        ordered by ``in_sites``.
    values : torch.Tensor
        Stored values of shape ``(n_entries, *out_shape)``, with output axes
        in the original order of sites outside ``in_sites``.
    in_dim : sequence of int
        Dimensions of all sites in chain order, or only input dimensions in
        ``in_sites`` order when output sites are present. Omitted output
        dimensions are inferred from the trailing axes of ``values``.
    in_sites : sequence of int, optional
        Distinct input sites, in configuration order. Defaults to the first
        ``indices.shape[1]`` sites, in increasing order.

    Examples
    --------
    >>> source = tk.decompositions.SparseTensorSource(
    ...     torch.tensor([[0], [2]]), torch.tensor([2., 5.]), in_dim=(3,))
    >>> source.contract_dense()
    tensor([2., 0., 5.])

    Select only the first variable while retaining the full tensor dimensions:

    >>> source = tk.decompositions.SparseTensorSource(
    ...     torch.tensor([[0], [1]]),
    ...     torch.tensor([[0., 2., 0.], [0., 0., 5.]]),
    ...     in_dim=(2, 3), in_sites=(0,))
    >>> source.n_sites, source.in_dim, source.out_shape
    (2, (2, 3), (3,))
    >>> source.evaluate(tk.decompositions.ConfigurationBatch(torch.tensor([[1]])))
    tensor([[0., 0., 5.]])
    """

    def __init__(self,
                 indices: torch.Tensor,
                 values: torch.Tensor,
                 in_dim: Sequence[int],
                 *,
                 in_sites: Optional[Sequence[int]] = None) -> None:
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
        if not (values.is_floating_point() or values.is_complex() or \
                values.dtype == torch.bool):
            raise TypeError(
                '`values` should have a floating, complex or boolean dtype')

        n_inputs = indices.shape[1]
        super().__init__(n_inputs + values.ndim - 1, in_sites=(
            tuple(range(n_inputs)) if in_sites is None else in_sites))
        if len(self.in_sites) != n_inputs:
            raise ValueError('`indices` should contain one column per input site')

        self._in_dim = _normalize_dims(in_dim, 'in_dim')
        if len(self._in_dim) == n_inputs and values.ndim > 1:
            outputs = iter(values.shape[1:])
            self._in_dim = _normalize_dims(
                tuple(self._in_dim[self.in_sites.index(site)]
                      if site in self.in_sites else next(outputs)
                      for site in range(self.n_sites)), 'in_dim')
        if len(self._in_dim) != self.n_sites or \
                tuple(values.shape[1:]) != self.out_shape:
            raise ValueError(
                '`indices`, `values` and `in_dim` should describe all axes')
        indices = _discrete_indices(
            ConfigurationBatch(indices, kind='indices'),
            self._in_sites_dims, values.device)
        self._lookup = _SparseLookup(indices, values, self._in_sites_dims)
        self._indices, self._values = self._lookup.indices, self._lookup.values
        self._flat_ids = self._lookup.flat_ids

    @classmethod
    def matrix(cls,
               indices: torch.Tensor,
               values: torch.Tensor,
               in_dim: Sequence[int],
               out_dim: Sequence[int],
               *,
               ordering: str = 'grouped') -> 'SparseTensorSource':
        """
        Constructs a matrix source from paired input/output entry indices.

        Unlike the usual constructor, each row of ``indices`` identifies both
        sides of a scalar matrix entry. By default, columns are grouped as
        ``(in_0, in_1, ..., out_0, out_1, ...)``. With
        ``ordering="interleaved"``, columns alternate as
        ``(in_0, out_0, in_1, out_1, ...)``. For ``n_sites`` pairs,
        ``in_dim`` and ``out_dim`` must each contain ``n_sites`` dimensions;
        they cannot be inferred from the stored entries alone.

        :meth:`evaluate` receives only input indices, with shape
        ``(batch, n_sites)`` in site order, and returns all output axes with
        ``out_shape == out_dim``. Unstored entries remain zero.
        :meth:`contract_dense` returns interleaved input/output
        axes regardless of input ordering, as in :meth:`DenseTensorSource.matrix`.

        Parameters
        ----------
        indices : torch.Tensor
            Integer indices with shape ``(n_entries, 2 * n_sites)`` in
            ``ordering``.
        values : torch.Tensor
            Scalar entry values with shape ``(n_entries,)``.
        in_dim, out_dim : sequence of int
            Input and output dimensions, respectively, for each matrix site.
        ordering : {``"grouped"``, ``"interleaved"``}
            Order of index columns; defaults to ``"grouped"``.

        Returns
        -------
        SparseTensorSource
            Source retaining paired axes for matrix decomposition.

        Examples
        --------
        >>> source = tk.decompositions.SparseTensorSource.matrix(
        ...     torch.tensor([[0, 1], [1, 0]]), torch.tensor([2., 3.]),
        ...     in_dim=(2,), out_dim=(2,))
        >>> source.evaluate(tk.decompositions.ConfigurationBatch(
        ...     torch.tensor([[0]])))
        tensor([[0., 2.]])
        >>> source.contract_dense()
        tensor([[0., 2.],
                [3., 0.]])
        """
        inputs = _normalize_dims(in_dim, 'in_dim')
        outputs = _normalize_dims(out_dim, 'out_dim')
        if len(inputs) != len(outputs):
            raise ValueError(
                '`in_dim` and `out_dim` should have equal lengths')
        if not isinstance(ordering, str):
            raise TypeError('`ordering` should be str type')
        if ordering not in ('grouped', 'interleaved'):
            raise ValueError(
                '`ordering` should be either "grouped" or "interleaved"')
        if not isinstance(indices, torch.Tensor):
            raise TypeError('`indices` should be torch.Tensor type')
        if indices.ndim != 2:
            raise TypeError(
                '`indices` should be a two-dimensional integer tensor')
        if isinstance(values, torch.Tensor) and (values.ndim != 1):
            raise ValueError(
                '`values` should contain one scalar per matrix entry')
        n_sites = len(inputs)
        if indices.shape[1] != 2 * n_sites:
            raise ValueError('`indices` should contain two columns per site')

        if ordering == 'grouped':
            axes = tuple(axis for site in range(n_sites)
                         for axis in (site, n_sites + site))
            indices = indices[:, axes]

        source = cls(indices, values,
                     tuple(dim for pair in zip(inputs, outputs)
                           for dim in pair))
        source._matrix_dims = inputs, outputs
        source._in_dim = inputs
        source._set_sites(n_sites)
        source._ordering = ordering
        return source

    @classmethod
    def from_data(cls,
                  dataset: torch.Tensor,
                  in_dim: Optional[Sequence[int]] = None,
                  weights: Optional[torch.Tensor] = None,
                  dtype: Optional[torch.dtype] = None) -> 'SparseTensorSource':
        """
        Constructs a discrete empirical distribution from sample indices.

        Each row of ``dataset`` is one observation, rather than an entry with
        an explicitly supplied value as in the usual constructor. Samples
        receive equal weights by default; explicit ``weights`` are normalized
        to sum to one. Repeated configurations accumulate their weights, and
        unobserved configurations have zero mass. The result is a scalar
        vector source with ``out_shape=()``.

        Parameters
        ----------
        dataset : torch.Tensor
            Integer samples with shape ``(n_samples, len(in_sites))``. All
            columns are inputs in their given order, so this constructor has
            ``len(in_sites) == n_sites``. Repeats accumulate.
        in_dim : sequence of int, optional
            Dimensions of all vector sites; inferred from sample maxima when
            omitted.
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

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Dimensions of all vector variables or matrix input axes."""
        return self._in_dim

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
        """
        Coalesced stored indices in ``in_sites`` order for vectors.

        Matrices return interleaved input/output indices in site order.
        """
        return ConfigurationBatch(self._indices, kind='indices')

    @property
    def support_values(self) -> torch.Tensor:
        """Stored values corresponding to :attr:`support`."""
        return self._values

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Evaluates integer configurations of shape ``(batch, len(in_sites))``
        in ``in_sites`` order. Returns shape ``(batch, *out_shape)``, with
        zeros outside the support.
        """
        indices = _discrete_indices(
            configurations, self._in_sites_dims, self.device)
        if self.is_matrix:
            outputs = _unravel_indices(torch.arange(
                prod(self.out_dim), device=self.device), self.out_dim)
            full = indices.new_empty(indices.shape[0],
                                     outputs.shape[0],
                                     2 * self.n_sites)
            full[:, :, ::2] = indices[:, None, :]
            full[:, :, 1::2] = outputs[None, :, :]
            result = self._lookup(full.reshape(-1, 2 * self.n_sites)).reshape(
                indices.shape[0], *self.out_dim)
        else:
            result = self._lookup(indices)
        self._record_evaluation(points=indices.shape[0])
        return result

    def contract_dense(self) -> torch.Tensor:
        """
        Materializes all sites in their original order, with zeros outside
        the support. Matrix input/output axes are interleaved in site order.
        """
        shape = (tuple(dim for pair in zip(self.in_dim, self.out_dim)
                       for dim in pair) if self.is_matrix else self.in_dim)
        tensor = self._values.new_zeros(shape)
        ordered = tensor
        if not self.is_matrix:
            out_sites = tuple(site for site in range(self.n_sites)
                              if site not in self.in_sites)
            ordered = tensor.permute(*self.in_sites, *out_sites)
        ordered[tuple(self._indices.T)] = self._values
        return tensor


__all__ = ['SparseTensorSource']
