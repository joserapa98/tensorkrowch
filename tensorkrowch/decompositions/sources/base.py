"""
This script contains:

    Public classes:
        * ConfigurationBatch
        * BaseSource

    Internal classes:
        * _TensorView

    Internal functions:
        * _discrete_indices
        * _data_weights
        * _tensor_view
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.utils import _INTEGER_DTYPES

from tensorkrowch.decompositions._utils import (_normalize_positive_int,
                                                _ravel_indices,
                                                _unravel_indices)
from tensorkrowch.decompositions.metrics import EvaluationStats


ConfigurationValues = Union[torch.Tensor, Sequence[torch.Tensor]]


@dataclass(frozen=True)
class ConfigurationBatch:  # MARK: ConfigurationBatch
    """
    Batch of discrete indices or possibly continuous features.

    ``values`` can be a packed tensor with shape ``(batch, n_inputs)``, or a
    sequence of ``n_inputs`` tensors. For source evaluation,
    ``n_inputs = len(source.in_sites)`` and columns follow ``source.in_sites``.
    The batch's :attr:`n_inputs` returns this column count, which can be smaller
    than ``source.n_sites``.

    For ``kind="indices"``, each column holds one integer index. For
    ``kind="features"``, each column holds a scalar or tensor-valued feature
    passed to a callable. A packed feature batch has shape
    ``(batch, n_inputs, *feature_shape)``; the sequence form permits different
    feature shapes for each input. ``kind`` applies to the entire batch.

    Parameters
    ----------
    values : torch.Tensor or sequence of torch.Tensor
        Packed configurations or one tensor per evaluated input, in
        ``in_sites`` order. All tensors in a heterogeneous batch share their
        leading batch dimension, device and dtype.
    kind : {``"indices"``, ``"features"``}
        Meaning of the ``values`` at every input. ``"indices"`` is the default
        and requires scalar integers. ``"features"`` accepts scalar or
        tensor-valued inputs for callables.
    """

    values: ConfigurationValues  # Values in original configuration or observation order
    kind: str = 'indices'  # Whether configurations contain indices or features

    def __post_init__(self) -> None:
        if self.kind not in ('indices', 'features'):
            raise ValueError(
                "`kind` should be 'indices' or 'features'")

        if isinstance(self.values, torch.Tensor):
            if self.values.ndim < 2:
                raise ValueError(
                    'Packed `values` should have batch and input dimensions')
            if self.values.shape[1] < 1:
                raise ValueError('`values` should contain at least one input')
            if (self.kind == 'indices') and (
                    (self.values.ndim != 2) or
                    (self.values.dtype not in _INTEGER_DTYPES)):
                raise TypeError(
                    'Discrete packed `values` should contain scalar integers')
            return

        if isinstance(self.values, (str, bytes)) or \
                not isinstance(self.values, Sequence):
            raise TypeError(
                '`values` should be a tensor or a sequence of tensors')
        input_values = tuple(self.values)
        if not input_values:
            raise ValueError('`values` should contain at least one input')
        if not all(isinstance(value, torch.Tensor) for value in input_values):
            raise TypeError('Every input value should be a torch.Tensor')

        first = input_values[0]
        for value in input_values:
            if value.ndim < 1:
                raise ValueError(
                    'Every input value should have a batch dimension')
            if value.shape[0] != first.shape[0]:
                raise ValueError(
                    'All input values should have the same batch size')
            if value.device != first.device:
                raise ValueError('All input values should be on the same device')
            if value.dtype != first.dtype:
                raise ValueError('All input values should have the same dtype')
            if (self.kind == 'indices') and (
                    (value.ndim != 1) or
                    (value.dtype not in _INTEGER_DTYPES)):
                raise TypeError(
                    'Discrete heterogeneous `values` should contain scalar '
                    'integers')

        object.__setattr__(self, 'values', input_values)

    @property
    def packed(self) -> bool:
        """Whether all inputs are stored in one tensor."""
        return isinstance(self.values, torch.Tensor)

    @property
    def batch_size(self) -> int:
        """Number of configurations in the batch."""
        if self.packed:
            return self.values.shape[0]
        return self.values[0].shape[0]

    @property
    def n_inputs(self) -> int:
        """Input column count; equals ``len(source.in_sites)`` for source queries."""
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
    def feature_shape(self) -> Optional[Tuple[Tuple[int, ...], ...]]:
        """Feature shape per input column, or ``None`` for discrete indices."""
        if self.kind == 'indices':
            return None
        if self.packed:
            shape = tuple(self.values.shape[2:])
            return (shape,) * self.n_inputs
        return tuple(tuple(value.shape[1:]) for value in self.values)

    def as_tensor(self) -> torch.Tensor:
        """Returns a packed tensor when all input shapes are compatible."""
        if self.packed:
            return self.values
        shapes = tuple(tuple(value.shape[1:]) for value in self.values)
        if any(shape != shapes[0] for shape in shapes[1:]):
            raise ValueError(
                'Heterogeneous feature shapes cannot be packed into one tensor')
        return torch.stack(self.values, dim=1)

    def index_select(self, indices: torch.Tensor) -> 'ConfigurationBatch':
        """Selects configurations along the batch dimension."""
        if not isinstance(indices, torch.Tensor):
            raise TypeError('`indices` should be torch.Tensor type')
        if (indices.ndim != 1) or (indices.dtype not in _INTEGER_DTYPES):
            raise TypeError(
                '`indices` should be a one-dimensional integer tensor')
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


class BaseSource(ABC):  # MARK: BaseSource
    """
    Common interface and shared operations for decomposition input data.

    Sources let decomposition algorithms query tensors or callables through
    the same interface, independently of how their values are stored or
    computed. Concrete sources adapt dense tensors, sparse tensors, formats
    or functions; the algorithm defines how to approximate those values.

    The main operation is :meth:`evaluate`, which receives a
    :class:`ConfigurationBatch` with one column per entry of ``in_sites``
    containing discrete indices or continuous features, and returns values with
    shape ``(batch, *core_batch, *out_shape)``, preserving their order.
    ``core_batch`` denotes optional structural batch dimensions, independent
    of the queried configurations and represented variables.

    Sources record successful queries with ``_record_evaluation``;
    :attr:`evaluation_stats` exposes cumulative counters so algorithms can
    measure their evaluation cost. Recording does not cache values or
    deduplicate configurations automatically.

    Properties describe the evaluated data: ``in_dim`` bounds discrete
    indices, ``out_shape`` describes each returned value, and ``dtype`` and
    ``device`` specify its numerical representation. ``out_shape`` and
    ``dtype`` may be inferred on first evaluation. Continuous sources normally
    have ``in_dim=None``; matrix grouping alone does not discretize a callable.
    Vector ``in_dim`` contains every variable dimension, and ``in_sites``
    selects the evaluated subset. Matrix dimensions describe one input/output
    pair per variable; every pair receives an input. ``is_discrete_tensor``
    distinguishes finite indexed tensors.

    Two shared operations build on evaluation: :meth:`fiber` varies one input
    site while fixing the others, and :meth:`contract_dense` evaluates the
    complete tensor when ``is_discrete_tensor=True``. Subclasses may replace
    these generic implementations with specialized contractions.

    To implement a source, provide :meth:`evaluate` and the abstract data
    properties, and call ``super().__init__(n_sites, in_sites=...)`` to
    declare all represented variables and the ordered evaluated inputs.
    Sources supply values and evaluation information; they do not define a
    fitting loss or decide which entries are missing.

    Parameters
    ----------
    n_sites : int
        Total number of represented variables. A matrix variable has one
        input axis and one output axis.
    in_sites : sequence of int, optional
        Ordered variables received by evaluation configurations. The complement
        supplies the output axes. Defaults to all variables; matrices always
        evaluate every variable and return their paired output axes.
    n_batches : int, optional
        Number of structural batch axes, whose sizes form ``core_batch``.
        Defaults to ``0``.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(torch.ones(2, 3))
    >>> source.contract_dense()
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    """

    input_kind: str = 'indices'

    def __init__(self,
                 n_sites: int,
                 *,
                 in_sites: Optional[Sequence[int]] = None,
                 n_batches: int = 0) -> None:
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        self._set_sites(n_sites, in_sites)
        self.n_batches = n_batches
        self._matrix_dims = None

        self._requested_points = 0  # Total configurations requested
        self._unique_points = 0  # Sum of unique configurations per query
        self._evaluation_batches = 0  # Batches used to evaluate queries
        self._cache_hits = 0  # Requested values retrieved from a cache
        self._source_calls = 0  # Successful source queries recorded

    def _set_sites(self,
                   n_sites: int,
                   in_sites: Optional[Sequence[int]] = None) -> None:
        """Sets the total variable count and its ordered input selection."""
        n_sites = _normalize_positive_int(n_sites, 'n_sites')
        if in_sites is None:
            in_sites = tuple(range(n_sites))
        elif isinstance(in_sites, (str, bytes)) or not \
                isinstance(in_sites, Sequence):
            raise TypeError('`in_sites` should be a sequence of integers')
        else:
            in_sites = tuple(in_sites)
        if not in_sites:
            raise ValueError('`in_sites` should contain at least one input')
        if any(isinstance(site, bool) or not \
                isinstance(site, int) for site in in_sites):
            raise TypeError('`in_sites` should be a sequence of integers')
        if (len(set(in_sites)) != len(in_sites)) or any(
                (site < 0) or (site >= n_sites) for site in in_sites):
            raise ValueError(
                '`in_sites` should contain distinct valid variables')
        self._n_sites = n_sites
        self._in_sites = in_sites

    @property
    def n_sites(self) -> int:
        """Total number of represented variables, including output variables."""
        return self._n_sites

    @property
    def is_matrix(self) -> bool:
        """Whether local axes are paired as matrix inputs and outputs."""
        return self._matrix_dims is not None

    @property
    def is_discrete_tensor(self) -> bool:
        """Whether all configurations index a fixed finite tensor."""
        return self.input_kind == 'indices' and self.in_dim is not None

    @property
    @abstractmethod
    def in_dim(self) -> Optional[Tuple[int, ...]]:
        """
        Dimensions of all vector variables or each matrix pair's input axis.

        ``None`` when no finite index grid is defined.
        """

    @property
    def out_dim(self) -> Optional[Tuple[int, ...]]:
        """Output dimension of each matrix pair, or ``None`` for vectors."""
        return self._matrix_dims[1] if self.is_matrix else None

    @property
    def in_sites(self) -> Tuple[int, ...]:
        """Ordered variables evaluated by the configuration columns."""
        return self._in_sites

    @property
    def _in_sites_dims(self) -> Optional[Tuple[int, ...]]:
        """Dimensions in input order, or ``None`` without a finite index grid."""
        dimensions = self.in_dim
        if dimensions is None:
            return None
        return tuple(dimensions[site] for site in self.in_sites)

    @property
    def out_shape(self) -> Optional[Tuple[int, ...]]:
        """
        Response dimensions, excluding query and structural batch axes.

        Vectors retain the dimensions outside ``in_sites`` in their original
        variable order. Matrices return all ``out_dim`` axes. Callable sources
        may return ``None`` until the response shape is known.
        """
        if self.is_matrix:
            return self.out_dim
        if self.in_dim is None:
            return None
        return tuple(dim for site, dim in enumerate(self.in_dim)
                     if site not in self.in_sites)

    @property
    @abstractmethod
    def dtype(self) -> Optional[torch.dtype]:
        """Output dtype, or ``None`` until a callable source is evaluated."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device on which evaluations are performed."""

    @property
    def evaluation_stats(self) -> EvaluationStats:
        """Cumulative point-evaluation counters for this source."""
        return EvaluationStats(
            requested_points=self._requested_points,
            unique_points=self._unique_points,
            batches=self._evaluation_batches,
            cache_hits=self._cache_hits,
            source_calls=self._source_calls)

    def _record_evaluation(self,
                           points: int,
                           batches: int = 1,
                           unique_points: Optional[int] = None,
                           cache_hits: int = 0) -> None:
        """
        Records one successful source query without tensor synchronization.
        """
        if unique_points is None:
            unique_points = points
        self._requested_points += points
        self._unique_points += unique_points
        self._evaluation_batches += batches
        self._cache_hits += cache_hits
        self._source_calls += 1

    @abstractmethod
    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Evaluates ``configurations`` in their original order.

        Parameters
        ----------
        configurations : ConfigurationBatch
            Discrete indices of shape ``(batch, len(in_sites))``, or continuous
            features of shape ``(batch, len(in_sites), *feature_shape)``.
            A heterogeneous batch contains ``len(in_sites)`` tensors. Columns
            and sequence entries follow ``in_sites`` order.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch, *core_batch, *out_shape)`` on the
            source device and with its declared or inferred dtype.
            Evaluation retains the source's autograd behavior; diagnostic
            counters do not change the returned values.
        """

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluates a source while varying one ``site`` of each configuration.

        For example, if a scalar source represents ``f(i, j)`` and a base
        configuration is ``(i=1, j=0)``, setting ``site=1`` and
        ``values=[0, 1, 2]`` returns ``[f(1, 0), f(1, 1), f(1, 2)]`` for that
        configuration. Other ``configurations`` in the batch are expanded in
        the same way, without changing their order.

        Parameters
        ----------
        configurations : ConfigurationBatch
            Base configurations with ``len(in_sites)`` inputs per row, ordered
            by ``in_sites``. Packed indices have shape
            ``(batch, len(in_sites))``; features may add trailing feature axes
            or use a tuple with one tensor per input. Only the column
            ``in_sites.index(site)`` is varied; all other inputs remain fixed.
        site : int
            Represented variable whose value varies; it must belong to
            ``in_sites``.
        values : torch.Tensor, optional
            Candidate ``values`` for the selected ``site``. Discrete indices
            have shape ``(n_values,)``; features have shape
            ``(n_values, *feature_shape[in_sites.index(site)])``. If omitted
            for discrete indices, every index in ``range(in_dim[site])`` is
            evaluated. Feature fibers require explicit ``values``.

        Returns
        -------
        torch.Tensor
            Values with shape
            ``(batch, n_values, *core_batch, *out_shape)``. The first axis
            follows the input ``configurations``; the second follows
            ``values`` (or increasing discrete indices when omitted).

        Examples
        --------
        >>> source = tk.decompositions.CallableTensorSource(
        ...     lambda indices: indices.sum(-1).float(), in_dim=(2, 3))
        >>> configurations = tk.decompositions.ConfigurationBatch(
        ...     torch.tensor([[1, 0]]))
        >>> source.fiber(configurations, site=1)
        tensor([[1., 2., 3.]])
        """
        if not isinstance(configurations, ConfigurationBatch):
            raise TypeError(
                '`configurations` should be ConfigurationBatch type')
        if configurations.kind != self.input_kind:
            raise ValueError(
                f'Configurations should have kind `{self.input_kind}`')
        if configurations.n_inputs != len(self.in_sites):
            raise ValueError(
                'Configurations should contain one value per input site')
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if site not in self.in_sites:
            raise ValueError('`site` should identify an input site')
        column = self.in_sites.index(site)

        if values is None:
            if not self.is_discrete_tensor:
                raise ValueError('Continuous fibers require explicit `values`')
            values = torch.arange(self.in_dim[site],
                                  device=configurations.device)
        else:
            if not isinstance(values, torch.Tensor):
                raise TypeError('`values` should be torch.Tensor type')
            if configurations.kind == 'indices':
                if (values.ndim != 1) or (values.dtype not in _INTEGER_DTYPES):
                    raise TypeError(
                        'Discrete fiber `values` should be a one-dimensional '
                        'integer tensor')
            else:
                if values.ndim < 1:
                    raise ValueError(
                        '`values` should have a leading grid dimension')
                feature_shape = configurations.values.shape[2:] \
                    if configurations.packed else \
                    configurations.values[column].shape[1:]
                if values.shape[1:] != feature_shape:
                    raise ValueError(
                        '`values` feature shape should match the input site')
        dtype = torch.promote_types(configurations.dtype, values.dtype)
        values = values.to(device=configurations.device, dtype=dtype)

        # Expand each configuration over the candidate values at this site.
        n_values = values.shape[0]
        if configurations.packed:
            packed = configurations.values.to(dtype=dtype)
            expanded = packed.unsqueeze(1).expand(
                packed.shape[0], n_values, *packed.shape[1:]).clone()
            expanded[:, :, column] = values.unsqueeze(0)
            expanded = expanded.flatten(0, 1)
        else:
            expanded = []
            for current_site, site_values in enumerate(configurations.values):
                if current_site == column:
                    site_values = values.unsqueeze(0).expand(
                        configurations.batch_size, n_values, *values.shape[1:])
                    site_values = site_values.flatten(0, 1)
                else:
                    site_values = site_values.to(dtype=dtype)
                    site_values = site_values.repeat_interleave(n_values, dim=0)
                expanded.append(site_values)
            expanded = tuple(expanded)

        result = self.evaluate(
            ConfigurationBatch(expanded, kind=configurations.kind))
        return result.reshape(
            configurations.batch_size, n_values, *result.shape[1:])

    def contract_dense(self) -> torch.Tensor:
        """
        Evaluates the complete finite tensor in its represented axis order.

        Returns
        -------
        torch.Tensor
            Shape ``(*core_batch, *in_dim)`` for vectors, with variables in
            their original order regardless of ``in_sites``. Matrices return
            shape ``(*core_batch, in_0, out_0, in_1, out_1, ...)``, pairing
            input and output axes in site order. Continuous sources require
            an explicit finite discretization.

        Examples
        --------
        >>> source = tk.decompositions.CallableTensorSource(
        ...     lambda indices: indices.sum(-1).float(), in_dim=(2, 2))
        >>> source.contract_dense()
        tensor([[0., 1.],
                [1., 2.]])
        """
        if not self.is_discrete_tensor:
            raise ValueError('`source` should represent a discrete tensor')

        dimensions = self._in_sites_dims
        indices = _unravel_indices(
            torch.arange(prod(dimensions), device=self.device), dimensions)
        values = self.evaluate(ConfigurationBatch(indices))
        tensor = values.reshape(*dimensions, *values.shape[1:])
        n_inputs = len(self.in_sites)
        batches = tuple(range(n_inputs, n_inputs + self.n_batches))
        if self.is_matrix:
            axes = tuple(axis for site in range(n_inputs)
                         for axis in (site, n_inputs + self.n_batches + site))
        else:
            outputs = tuple(site for site in range(self.n_sites)
                            if site not in self.in_sites)
            order = (*self.in_sites, *outputs)
            positions = (*range(n_inputs),
                         *range(n_inputs + self.n_batches, tensor.ndim))
            axes = tuple(positions[order.index(site)]
                         for site in range(self.n_sites))
        return tensor.permute(*batches, *axes)


def _discrete_indices(configurations: ConfigurationBatch,
                      in_dim: Sequence[int],
                      device: torch.device) -> torch.Tensor:
    """
    Validates discrete ``configurations`` and returns packed long indices.
    """
    if not isinstance(configurations, ConfigurationBatch):
        raise TypeError(
            '`configurations` should be ConfigurationBatch type')
    if configurations.kind != 'indices':
        raise ValueError('This source requires discrete index configurations')
    if configurations.n_inputs != len(in_dim):
        raise ValueError(
            'Configurations should contain one value per input site')
    indices = configurations.as_tensor().to(device=device, dtype=torch.long)
    for site, dim in enumerate(in_dim):
        if torch.any(indices[:, site] < 0) or torch.any(indices[:, site] >= dim):
            raise ValueError(
                f'Configuration indices at site {site} are out of bounds')
    return indices


def _data_weights(dataset: torch.Tensor,
                  weights: Optional[torch.Tensor] = None,
                  dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Validates a finite dataset and normalizes non-negative sample weights."""
    if not isinstance(dataset, torch.Tensor):
        raise TypeError('`dataset` should be torch.Tensor type')
    if (dataset.ndim != 2) or not dataset.shape[0] or not dataset.shape[1]:
        raise ValueError('`dataset` should have non-empty sample and site axes')
    if dataset.is_complex() or not torch.isfinite(dataset).all():
        raise ValueError('`dataset` should contain finite real coordinates')
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if not isinstance(dtype, torch.dtype):
        raise TypeError('`dtype` should be torch.dtype type')
    if weights is None:
        weights = torch.ones(dataset.shape[0],
                             dtype=dtype, device=dataset.device)
    if not isinstance(weights, torch.Tensor):
        raise TypeError('`weights` should be torch.Tensor type')
    if weights.shape != (dataset.shape[0],) or weights.device != dataset.device:
        raise ValueError('`weights` should match dataset samples and device')
    if weights.is_complex() or not torch.isfinite(weights).all() or \
            torch.any(weights < 0):
        raise ValueError('`weights` should be finite, real and non-negative')
    if not weights.is_floating_point():
        weights = weights.to(dtype)
    total = weights.sum()
    if not torch.isfinite(total) or (total <= 0):
        raise ValueError('The total sample weight should be positive')
    return weights / total


class _TensorView(BaseSource):  # MARK: _TensorView
    """
    Adapts a finite source so ALS and RS can query individual scalar entries.

    For vector sources, all existing variables become inputs in their original
    order. Their dimensions and total count are preserved. For example, a source
    with ``in_dim=(2, 3, 4)`` and ``in_sites=(2, 0)`` receives ``(k, i)`` and
    returns ``T[i, :, k]``. The view receives ``(i, j, k)`` and selects
    ``T[i, j, k]`` from that response.

    For matrix sources, each input/output index pair is instead fused into
    one site. Evaluation decodes those pairs, queries the original source
    and selects the requested output entries. The fused index is
    ``p = i * out_dim[site] + j``, with dimension
    ``in_dim[site] * out_dim[site]``. The view represents a vector tensor,
    so ``is_matrix=False`` and ``out_dim=None``.

    In both cases, ``in_sites=tuple(range(n_sites))`` and ``out_shape=()``.
    Queries have shape ``(batch, n_sites)`` and return values with shape
    ``(batch, *core_batch)`` without materializing the complete tensor.
    Evaluation statistics are read directly from the original source.
    """

    def __init__(self, source: BaseSource) -> None:
        self.source = source

        if source.is_matrix:
            self._in_dim = tuple(
                a * b for a, b in zip(source.in_dim, source.out_dim))
        else:
            self._out_axes = tuple(site for site in range(source.n_sites)
                                   if site not in self.source.in_sites)
            self._in_dim = source.in_dim

        super().__init__(len(self._in_dim), n_batches=source.n_batches)

    @property
    def in_dim(self) -> Tuple[int, ...]:
        return self._in_dim

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self.source.dtype

    @property
    def device(self) -> torch.device:
        return self.source.device

    @property
    def evaluation_stats(self) -> EvaluationStats:
        """Cumulative evaluation counters of the original source."""
        return self.source.evaluation_stats

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Selects output indices or decodes fused input/output pairs."""
        indices = _discrete_indices(configurations, self._in_dim, self.device)
        if self.source.is_matrix:
            sizes = indices.new_tensor(self.source.out_dim)
            inputs, outputs = indices // sizes, indices % sizes
        else:
            inputs = indices[:, self.source.in_sites]
            outputs = indices[:, self._out_axes]
        values = self.source.evaluate(ConfigurationBatch(inputs))
        if not self.source.out_shape:
            return values
        labels = _ravel_indices(outputs, self.source.out_shape)
        values = values.flatten(start_dim=1 + self.n_batches)
        labels = labels.reshape(-1, *((1,) * (self.n_batches + 1)))
        labels = labels.expand(*values.shape[:-1], 1)
        return values.gather(-1, labels).squeeze(-1)


def _tensor_view(source: BaseSource) -> BaseSource:
    """Prepares a scalar chain view without materializing the source."""
    if not source.is_discrete_tensor:
        raise ValueError('`source` should represent a discrete tensor')
    if source.out_shape is None:
        source.evaluate(ConfigurationBatch(torch.zeros(
            1, len(source.in_sites), device=source.device, dtype=torch.long)))
    if source.is_matrix or (source.out_shape != ()) or \
            (source.in_sites != tuple(range(source.n_sites))):
        return _TensorView(source)
    return source


__all__ = [
    'ConfigurationBatch',
    'BaseSource',
]
