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

    ``values`` can be a packed tensor whose first two dimensions are ``(batch,
    n_sites)``, or a sequence containing one tensor per site. For discrete
    ``kind="indices"``, each site holds one integer that selects a slice of a
    tensor or an input to a discrete function. For ``kind="features"``, each
    site holds a scalar or vector-valued feature passed to a callable, such as
    a continuous input or a latent feature from another model layer. A packed
    batch then has shape ``(batch, n_sites, *feature_shape)``, while the
    sequence form permits different feature shapes at different sites.
    ``kind`` applies to the entire batch; it does not mix indices and features.

    Parameters
    ----------
    values : torch.Tensor or sequence of torch.Tensor
        Packed configurations or one tensor per site. Every site tensor in a
        heterogeneous batch must share its leading batch dimension, device and
        dtype.
    kind : {``"indices"``, ``"features"``}
        Meaning of the ``values`` at every site. ``"indices"`` is the default
        and requires scalar integers. ``"features"`` accepts scalar or
        vector-valued inputs for callables.
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
    def feature_shape(self) -> Optional[Tuple[Tuple[int, ...], ...]]:
        """Feature shape at each site, or ``None`` for discrete indices."""
        if self.kind == 'indices':
            return None
        if self.packed:
            shape = tuple(self.values.shape[2:])
            return (shape,) * self.n_sites
        return tuple(tuple(value.shape[1:]) for value in self.values)

    def as_tensor(self) -> torch.Tensor:
        """Returns a packed tensor when all site shapes are compatible."""
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
    :class:`ConfigurationBatch` of discrete indices or continuous features
    and returns values with shape ``(batch, *out_shape)``, preserving their
    order. Sources record successful queries with ``_record_evaluation``;
    :attr:`evaluation_stats` exposes cumulative counters so algorithms can
    measure their evaluation cost. Recording does not cache values or
    deduplicate configurations automatically.

    Properties describe the evaluated data: ``in_dim`` bounds discrete
    indices, ``out_shape`` describes each returned value, and ``dtype`` and
    ``device`` specify its numerical representation. ``out_shape`` and
    ``dtype`` may be inferred on first evaluation. Continuous sources normally
    have ``in_dim=None``; matrix grouping alone does not discretize a callable.
    ``is_discrete_tensor`` distinguishes finite indexed tensors.

    Two shared operations build on evaluation: :meth:`fiber` varies one input
    site while fixing the others, and :meth:`contract_dense` evaluates the
    complete tensor when ``is_discrete_tensor=True``. Subclasses may replace
    these generic implementations with specialized contractions.

    To implement a source, provide :meth:`evaluate` and the abstract data
    properties, and call ``super().__init__(n_sites)`` to fix its input-site
    count and initialize its counters.
    Sources supply values and evaluation information; they do not define a
    fitting loss or decide which entries are missing.

    Parameters
    ----------
    n_sites : int
        Number of input sites received by each evaluation configuration.
    n_batches : int, optional
        Number of structural batch axes stored in the source. Defaults to ``0``.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(torch.ones(2, 3))
    >>> source.contract_dense()
    tensor([[1., 1., 1.],
            [1., 1., 1.]])
    """

    input_kind: str = 'indices'

    def __init__(self, n_sites: int, *, n_batches: int = 0) -> None:
        n_sites = _normalize_positive_int(n_sites, 'n_sites')
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        self._n_sites = n_sites
        self.n_batches = n_batches
        self._matrix_dims = None

        self._requested_points = 0  # Total configurations requested
        self._unique_points = 0  # Sum of unique configurations per query
        self._evaluation_batches = 0  # Batches used to evaluate queries
        self._cache_hits = 0  # Requested values retrieved from a cache
        self._source_calls = 0  # Successful source queries recorded

    @property
    def n_sites(self) -> int:
        """Number of input sites fixed on construction."""
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
        """Discrete dimensions, or ``None`` for continuous inputs."""

    @property
    def out_dim(self) -> Optional[Tuple[int, ...]]:
        """Local output dimensions when the source has matrix grouping."""
        return self._matrix_dims[1] if self.is_matrix else None

    @property
    def in_sites(self) -> Tuple[int, ...]:
        """Positions of function inputs in the represented vector chain."""
        return tuple(range(self._n_sites))

    @property
    @abstractmethod
    def out_shape(self) -> Optional[Tuple[int, ...]]:
        """Shape returned after the leading configuration batch."""

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
            A batch containing one discrete index or feature per source site.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch, *out_shape)`` on the source device
            and with its declared or inferred dtype. Evaluation retains the
            source's autograd behavior; diagnostic counters do not change the
            returned values.
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
            Base ``configurations`` in batch order, stored as a packed tensor
            or a tuple of site tensors. Values at sites other than ``site``
            remain fixed; the value at ``site`` is replaced.
        site : int
            Zero-based input ``site`` whose value varies.
        values : torch.Tensor, optional
            Candidate ``values`` for the selected ``site``. Discrete indices
            have shape ``(n_values,)``; features have shape
            ``(n_values, *feature_shape[site])``. If omitted for discrete
            indices, the source evaluates every index in
            ``range(in_dim[site])``. Feature fibers require explicit ``values``
            because there is no finite default grid for a continuous domain.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch, n_values, *out_shape)``. The first
            axis follows the input ``configurations``; the second follows
            ``values`` (or increasing discrete indices when omitted). A scalar
            source has ``out_shape=()`` and returns ``(batch, n_values)``.

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
        if configurations.n_sites != self._n_sites:
            raise ValueError(
                'Configurations should contain one value per input site')
        if not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= self._n_sites):
            raise ValueError('`site` should identify an input site')

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
                    configurations.values[site].shape[1:]
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
            expanded[:, :, site] = values.unsqueeze(0)
            expanded = expanded.flatten(0, 1)
        else:
            expanded = []
            for current_site, site_values in enumerate(configurations.values):
                if current_site == site:
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
            Input-site axes followed by output axes for vectors, or paired
            input/output axes for matrices. Continuous sources need an explicit
            discretization before this operation is defined.

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

        indices = _unravel_indices(
            torch.arange(prod(self.in_dim), device=self.device), self.in_dim)
        values = self.evaluate(ConfigurationBatch(indices))
        tensor = values.reshape(*self.in_dim, *values.shape[1:])
        if self.is_matrix and self.out_shape:
            n = len(self.in_dim)
            tensor = tensor.permute(*(axis for site in range(n)
                                      for axis in (site, n + site)))
        return tensor


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
    if configurations.n_sites != len(in_dim):
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
        weights = torch.ones(dataset.shape[0], dtype=dtype, device=dataset.device)
    if not isinstance(weights, torch.Tensor):
        raise TypeError('`weights` should be torch.Tensor type')
    if weights.shape != (dataset.shape[0],) or weights.device != dataset.device:
        raise ValueError('`weights` should match dataset samples and device')
    if weights.is_complex() or not torch.isfinite(weights).all() or torch.any(weights < 0):
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

    For vector sources, output axes become additional indexed sites. A source
    with ``in_dim=(2,)`` and ``out_shape=(3,)`` returns a vector for each input
    index ``i``; this view has ``in_dim=(2, 3)`` and returns one scalar for
    each pair ``(i, j)``. Site positions follow the source's ``in_sites``.

    For matrix sources, each input/output index pair is instead fused into
    one site. Evaluation decodes those pairs, queries the original source
    and selects the requested output entries. Both adaptations expose
    ``out_shape=()`` without materializing the complete tensor. Evaluation
    statistics are read directly from the original source.
    """

    def __init__(self, source: BaseSource) -> None:
        self.source = source
        self._in_sites = source.in_sites

        if source.is_matrix:
            self._in_dim = tuple(a * b for a, b in zip(source.in_dim,
                                                       source.out_dim))
        else:
            n_sites = source.n_sites + len(source.out_shape)
            self._out_axes = tuple(site for site in range(n_sites)
                                   if site not in self._in_sites)
            dimensions = [None] * n_sites
            for site, dim in zip(self._in_sites, source.in_dim):
                dimensions[site] = dim
            for site, dim in zip(self._out_axes, source.out_shape):
                dimensions[site] = dim
            self._in_dim = tuple(dimensions)

        super().__init__(len(self._in_dim), n_batches=source.n_batches)

    @property
    def in_dim(self) -> Tuple[int, ...]:
        return self._in_dim

    @property
    def out_shape(self) -> Tuple[int, ...]:
        return ()

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
            inputs = indices[:, self._in_sites]
            outputs = indices[:, self._out_axes]
        values = self.source.evaluate(ConfigurationBatch(inputs))
        if not self.source.out_shape:
            return values
        labels = _ravel_indices(outputs, self.source.out_shape)
        return values.flatten(start_dim=1).gather(1, labels[:, None]).squeeze(1)


def _tensor_view(source: BaseSource) -> BaseSource:
    """Prepares a scalar chain view without materializing the source."""
    if not source.is_discrete_tensor:
        raise ValueError('`source` should represent a discrete tensor')
    if source.out_shape is None:
        source.evaluate(ConfigurationBatch(torch.zeros(
            1, source.n_sites, device=source.device, dtype=torch.long)))
    if source.is_matrix or (source.out_shape != ()) or \
            (source.in_sites != tuple(range(source.n_sites))):
        return _TensorView(source)
    return source


__all__ = [
    'ConfigurationBatch',
    'BaseSource',
]
