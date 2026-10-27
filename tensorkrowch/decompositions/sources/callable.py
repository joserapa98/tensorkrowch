"""
This script contains:

    Public classes:
        * CallableSource
        * CallableTensorSource

    Internal classes:
        * _DataEvaluator
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.decompositions._utils import _normalize_dims
from tensorkrowch.decompositions.sources.base import (BaseSource,
                                                      ConfigurationBatch,
                                                      _data_weights,
                                                      _discrete_indices)
from tensorkrowch.decompositions.sources.sparse import _SparseLookup


@dataclass
class _DataEvaluator:  # MARK: _DataEvaluator
    """Evaluates exact sample masses or a weighted user kernel."""

    dataset: torch.Tensor
    weights: torch.Tensor
    kernel: Optional[Callable]
    kwargs: dict

    def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Returns one mass or kernel value per queried configuration."""
        if self.kernel is None:
            dtype = torch.promote_types(coordinates.dtype, self.dataset.dtype)
            values = (coordinates.to(dtype)[:, None] ==
                      self.dataset.to(dtype)[None]).all(-1).to(self.weights.dtype)
        else:
            values = self.kernel(coordinates, self.dataset, **self.kwargs)
            if not isinstance(values, torch.Tensor) or values.shape != (
                    coordinates.shape[0], self.dataset.shape[0]):
                raise ValueError(
                    '`kernel` should return `(batch, samples)` values')
        dtype = torch.promote_types(values.dtype, self.weights.dtype)
        return values.to(dtype) @ self.weights.to(device=values.device,
                                                  dtype=dtype)


class CallableSource(BaseSource):  # MARK: CallableSource
    """
    Continuous function evaluated by a user callable.

    Packed configurations are passed to ``function`` with shape
    ``(batch, len(in_sites), *feature_shape)``; scalar inputs have no trailing
    feature axes. Heterogeneous inputs are passed as a tuple of
    ``len(in_sites)`` tensors. Both forms follow ``in_sites`` order. The
    callable preserves the leading batch and returns ``(batch, *out_shape)``.

    Construct the source with ``function`` and its total variable count.
    ``in_sites`` selects the inputs; all remaining variables are output axes.
    Output shape and dtype can be declared or discovered on first evaluation.
    This source accepts continuous features and has no finite grid to
    materialize. Use :class:`CallableTensorSource` for discrete indices, or
    :class:`CallableQuanticsSource` to discretize a coordinate function.

    The alternative constructor :meth:`matrix` pairs each continuous input
    with an output axis for a function returning all output entries per input.
    :meth:`from_data` instead builds a scalar vector source from observed
    coordinates, using exact matches or a user-supplied kernel.

    Parameters
    ----------
    function : callable
        Receives the inputs selected by ``in_sites``, in that order, and
        returns shape ``(batch, *out_shape)``.
    n_sites : int
        Total number of input and output variables. An input variable receives
        a scalar or embedded vector; each remaining variable is an output axis.
    out_shape : sequence of int or None, optional
        Declared response shape. The default ``None`` discovers it on the
        first non-empty evaluation; its rank must equal
        ``n_sites - len(in_sites)``. ``()`` requires scalar responses.
    dtype : torch.dtype or None, optional
        Declared output dtype. ``None`` infers it from the first evaluation.
    device : str or torch.device, optional
        Device to which configurations are moved before evaluation. Returned
        values must use the same device. Defaults to ``'cpu'``.
    batch_size : int or None, optional
        Maximum number of configurations passed to one callable invocation.
    in_sites : sequence of int, optional
        Ordered input variables. Defaults to all variables; output axes follow
        the complement in increasing variable order.

    Examples
    --------
    >>> source = tk.decompositions.CallableSource(
    ...     lambda coordinates: coordinates.sum(-1), n_sites=2)
    >>> source.evaluate(tk.decompositions.ConfigurationBatch(
    ...     torch.tensor([[0.25, 0.5], [1., 2.]]), kind='features'))
    tensor([0.7500, 3.0000])
    """

    input_kind: str = 'features'

    def __init__(self,
                 function: Callable,
                 n_sites: int,
                 out_shape: Optional[Sequence[int]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Union[str, torch.device] = 'cpu',
                 batch_size: Optional[int] = None,
                 *,
                 in_sites: Optional[Sequence[int]] = None) -> None:
        super().__init__(n_sites, in_sites=in_sites)
        if not callable(function):
            raise TypeError('`function` should be callable')
        normalized_out_shape = None if out_shape is None else \
            _normalize_dims(out_shape, 'out_shape', allow_empty=True)
        if normalized_out_shape is not None and len(normalized_out_shape) != (
                self.n_sites - len(self.in_sites)):
            raise ValueError(
                '`out_shape` should have one axis per output variable')
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type or None')
        if batch_size is not None:
            if not isinstance(batch_size, int) or (batch_size < 1):
                raise ValueError('`batch_size` should be a positive integer')

        self.function = function
        self._in_dim = None
        self._out_shape = normalized_out_shape
        self._dtype = dtype
        self._device = torch.empty(0, device=device).device
        self.batch_size = batch_size

    @classmethod
    def matrix(cls,
               function: Callable,
               n_sites: int,
               out_shape: Optional[Sequence[int]] = None,
               **kwargs) -> 'CallableSource':
        """
        Constructs a continuous matrix source with one output axis per input.

        :meth:`evaluate` receives inputs for every site in site order and
        returns all output axes, with ``out_shape == out_dim``. Each input is
        continuous, so ``in_dim`` remains ``None``. Output dimensions can be
        declared or inferred on first evaluation; their count must equal
        ``n_sites``. Output indices are not passed separately.

        Parameters
        ----------
        function : callable
            Receives shape ``(batch, n_sites, *feature_shape)`` or a tuple of
            ``n_sites`` tensors, and returns shape ``(batch, *out_shape)``.
        n_sites : int
            Number of paired continuous inputs and output axes.
        out_shape : sequence of int or None, optional
            Output dimension for each site. ``None`` infers the dimensions
            from the first non-empty evaluation.
        **kwargs
            ``dtype``, ``device`` and ``batch_size`` options of
            :class:`CallableSource`.

        Returns
        -------
        CallableSource
            Matrix source with ``out_shape == out_dim``.

        Examples
        --------
        >>> source = tk.decompositions.CallableSource.matrix(
        ...     lambda x: torch.stack((x[:, 0], 1 - x[:, 0]), dim=-1),
        ...     n_sites=1)
        >>> source.evaluate(tk.decompositions.ConfigurationBatch(
        ...     torch.tensor([[0.25], [0.75]]), kind='features'))
        tensor([[0.2500, 0.7500],
                [0.7500, 0.2500]])
        >>> source.out_dim
        (2,)
        """
        source = cls(function, n_sites, in_sites=None, **kwargs)
        source._matrix_dims = None, None
        if out_shape is not None:
            source._set_output_shape(_normalize_dims(out_shape, 'out_shape'))
        return source

    @classmethod
    def from_data(cls,
                  dataset: torch.Tensor,
                  weights: Optional[torch.Tensor] = None,
                  *,
                  kernel: Optional[Callable] = None,
                  **kernel_kwargs) -> 'CallableSource':
        """
        Constructs a scalar vector source from exact samples or a kernel.

        Unlike the usual constructor, this method builds the evaluation
        function from ``dataset``. Sample weights sum to one after
        normalization; omitted ``weights`` give equal mass to every sample.
        With ``kernel=None``, evaluation returns the total weight of samples
        exactly matching each query, and zero elsewhere. With a kernel, it
        returns the weighted sum of its query-to-sample values. For example,
        pass :func:`gaussian_kernel` with ``bandwidth`` to obtain a Gaussian
        density estimate; no kernel is selected automatically.

        All dataset columns are continuous inputs in their given order, with
        ``n_sites == dataset.shape[1]`` and ``in_dim=None``. The returned
        source has scalar responses, with ``out_shape=()``.

        Parameters
        ----------
        dataset : torch.Tensor
            Continuous coordinates with shape ``(n_samples, n_sites)``.
        weights : torch.Tensor, optional
            Non-negative weights normalized to total mass one.
        kernel : callable, optional
            Receives ``(coordinates, dataset, **kernel_kwargs)`` and returns
            ``(batch, n_samples)`` values. ``None`` uses exact coordinate matches.
        **kernel_kwargs
            Parameters owned by the selected kernel, such as ``bandwidth``.

        Returns
        -------
        CallableSource
            Scalar vector source evaluating exact masses or kernel estimates.

        Examples
        --------
        >>> source = tk.decompositions.CallableSource.from_data(
        ...     torch.tensor([[0.1], [0.4], [0.1]]))
        >>> source.evaluate(tk.decompositions.ConfigurationBatch(
        ...     torch.tensor([[0.1], [0.3]]), kind='features'))
        tensor([0.6667, 0.0000])

        A Gaussian kernel smooths the samples into a continuous density:

        >>> source = tk.decompositions.CallableSource.from_data(
        ...     torch.tensor([[0.], [0.]]),
        ...     kernel=tk.decompositions.gaussian_kernel, bandwidth=1.)
        >>> source.evaluate(tk.decompositions.ConfigurationBatch(
        ...     torch.tensor([[0.], [1.]]), kind='features'))
        tensor([0.3989, 0.2420])
        """
        weights = _data_weights(dataset, weights,
                                dataset.dtype if dataset.is_floating_point()
                                else None)

        if kernel is not None and not callable(kernel):
            raise TypeError('`kernel` should be callable')
        if kernel is None and kernel_kwargs:
            raise ValueError('Kernel parameters require a `kernel`')

        evaluator = _DataEvaluator(dataset, weights, kernel, kernel_kwargs)
        return cls(evaluator, dataset.shape[1], out_shape=(),
                   dtype=weights.dtype if kernel is None else None,
                   device=dataset.device)

    @property
    def in_dim(self) -> Optional[Tuple[int, ...]]:
        """Dimensions of all vector variables or paired matrix input axes."""
        return self._in_dim

    @property
    def out_shape(self) -> Optional[Tuple[int, ...]]:
        """Declared or inferred function output shape."""
        return super().out_shape if self.is_matrix or self.in_dim is not None \
            else self._out_shape

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Declared or inferred function output dtype."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device on which evaluations are performed."""
        return self._device

    def _set_output_shape(self, out_shape: Tuple[int, ...]) -> None:
        """Resolves response dimensions for the declared output variables."""
        n_outputs = self.n_sites if self.is_matrix else \
            self.n_sites - len(self.in_sites)
        if len(out_shape) != n_outputs:
            raise ValueError(
                '`function` should return one axis per output variable')
        self._out_shape = out_shape
        if self.is_matrix:
            self._matrix_dims = self.in_dim, out_shape

    def _validate_result(self, result: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Checks batch shape and stable result runtime metadata."""
        if not isinstance(result, torch.Tensor):
            raise TypeError('`function` should return a torch.Tensor')
        if result.device != self._device:
            raise ValueError(
                '`function` should return values on the source device')
        if (result.ndim < 1) or (result.shape[0] != batch_size):
            raise ValueError(
                '`function` should preserve the configuration batch dimension')

        out_shape = tuple(result.shape[1:])
        expected = self.out_shape
        if expected is None:
            self._set_output_shape(out_shape)
        elif out_shape != expected:
            raise ValueError(
                '`function` output shape does not match `out_shape`')
        if self._dtype is None:
            self._dtype = result.dtype
        elif result.dtype != self._dtype:
            raise ValueError('`function` output dtype changed between calls')
        return result

    def _evaluate_batch(self,
                        configurations: ConfigurationBatch) -> torch.Tensor:
        """Evaluates and validates one callable batch."""
        result = self.function(configurations.values)
        return self._validate_result(result, configurations.batch_size)

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Evaluates configurations using the source's input convention.

        Parameters
        ----------
        configurations : ConfigurationBatch
            Inputs in ``in_sites`` order: continuous features with shape
            ``(batch, len(in_sites), *feature_shape)`` for ``CallableSource``,
            or integer indices with shape ``(batch, len(in_sites))`` for
            ``CallableTensorSource`` and its Quantics subclass. Heterogeneous
            continuous inputs may instead use ``len(in_sites)`` tensors.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch, *out_shape)``, preserving input order
            and autograd. ``batch_size`` bounds each callable invocation.
        """
        if not isinstance(configurations, ConfigurationBatch):
            raise TypeError(
                '`configurations` should be ConfigurationBatch type')
        if configurations.n_inputs != len(self.in_sites):
            raise ValueError(
                'Configurations should contain one value per input site')
        configurations = configurations.to(self._device)
        if configurations.kind != self.input_kind:
            raise ValueError(
                f'Configurations should have kind `{self.input_kind}`')
        if self.input_kind == 'indices':
            _discrete_indices(configurations, self._in_sites_dims, self._device)

        if configurations.batch_size == 0:
            if self.out_shape is None or self._dtype is None:
                raise ValueError(
                    'An empty first evaluation requires output shape and dtype')
            result = torch.empty(
                (0, *self.out_shape),
                device=self._device,
                dtype=self._dtype)
            self._record_evaluation(points=0, batches=0)
            return result

        batch_size = self.batch_size or configurations.batch_size
        if batch_size >= configurations.batch_size:
            result = self._evaluate_batch(configurations)
            self._record_evaluation(points=configurations.batch_size)
            return result
        chunks = []
        for start in range(0, configurations.batch_size, batch_size):
            stop = min(start + batch_size, configurations.batch_size)
            ids = torch.arange(start, stop, device=configurations.device)
            chunks.append(
                self._evaluate_batch(configurations.index_select(ids)))
        result = torch.cat(chunks, dim=0)
        self._record_evaluation(
            points=configurations.batch_size, batches=len(chunks))
        return result


class CallableTensorSource(CallableSource):  # MARK: CallableTensorSource
    """
    Finite tensor whose values are evaluated by a callable.

    ``in_dim`` gives the dimension of every tensor axis, so
    ``n_sites == len(in_dim)``. By default, all axes are inputs and the
    callable returns one scalar per configuration.

    Use ``in_sites`` to select only some axes as inputs. The callable receives
    integer indices with shape ``(batch, len(in_sites))``, in that order,
    and returns the remaining axes in their original tensor order. Its output
    has shape ``(batch, *out_shape)``, where ``out_shape`` is determined by
    the dimensions of the unselected axes.

    Values are computed on demand. :meth:`contract_dense` evaluates all input
    configurations to construct the full tensor. Dtype, device and callable
    batching follow :class:`CallableSource`; dtype can be inferred on the
    first evaluation.

    Use :meth:`matrix` to pair each input axis with an output axis.
    :meth:`from_data` instead constructs the callable from discrete samples,
    using exact matches or a user-supplied kernel.

    Parameters
    ----------
    function : callable
        Receives integer indices with shape ``(batch, len(in_sites))``, in
        ``in_sites`` order, and returns shape ``(batch, *out_shape)``.
    in_dim : sequence of int
        Dimensions of all represented vector axes.
    in_sites : sequence of int, optional
        Ordered input variables. Defaults to every axis. The callable returns
        the complementary axes in their represented order.
    **kwargs
        Output shape, dtype, device and batching options of ``CallableSource``.

    Examples
    --------
    >>> source = tk.decompositions.CallableTensorSource(
    ...     lambda indices: indices.sum(-1).float(), in_dim=(2, 3))
    >>> source.contract_dense()
    tensor([[0., 1., 2.],
            [1., 2., 3.]])
    """

    input_kind: str = 'indices'

    def __init__(self,
                 function: Callable,
                 in_dim: Sequence[int],
                 **kwargs) -> None:
        dimensions = _normalize_dims(in_dim, 'in_dim')
        super().__init__(function, len(dimensions), **kwargs)
        self._in_dim = dimensions
        if self._out_shape is not None and (self._out_shape != self.out_shape):
            raise ValueError(
                '`out_shape` should match the unselected `in_dim` axes')

    @classmethod
    def matrix(cls,
               function: Callable,
               in_dim: Sequence[int],
               out_dim: Sequence[int],
               **kwargs) -> 'CallableTensorSource':
        """
        Constructs a matrix source from a callable on discrete input indices.

        ``in_dim`` and ``out_dim`` must have equal lengths, with one pair per
        site. :meth:`evaluate` receives indices with shape ``(batch, n_sites)``
        in site order and returns all output axes, with ``out_shape == out_dim``.
        :meth:`contract_dense` returns interleaved input/output axes.

        Parameters
        ----------
        function : callable
            Receives input indices and returns shape ``(batch, *out_dim)``.
        in_dim, out_dim : sequence of int
            Input and output dimensions, respectively, for each matrix site.
        **kwargs
            ``dtype``, ``device`` and ``batch_size`` options of
            :class:`CallableSource`.

        Returns
        -------
        CallableTensorSource
            Matrix source with ``out_shape == out_dim``.

        Examples
        --------
        >>> source = tk.decompositions.CallableTensorSource.matrix(
        ...     lambda indices: torch.nn.functional.one_hot(
        ...         indices[:, 0], num_classes=2).float(),
        ...     in_dim=(2,), out_dim=(2,))
        >>> source.contract_dense()
        tensor([[1., 0.],
                [0., 1.]])
        """
        inputs = _normalize_dims(in_dim, 'in_dim')
        outputs = _normalize_dims(out_dim, 'out_dim')
        if len(inputs) != len(outputs):
            raise ValueError(
                '`in_dim` and `out_dim` should have equal lengths')
        source = cls(function, (*inputs, *outputs),
                     in_sites=tuple(range(len(inputs))), out_shape=outputs,
                     **kwargs)
        source._matrix_dims = inputs, outputs
        source._in_dim = inputs
        source._set_sites(len(inputs))
        return source

    @classmethod
    def from_data(cls,
                  dataset: torch.Tensor,
                  in_dim: Optional[Sequence[int]] = None,
                  weights: Optional[torch.Tensor] = None,
                  *,
                  kernel: Optional[Callable] = None,
                  **kernel_kwargs) -> 'CallableTensorSource':
        """
        Constructs a scalar tensor source from discrete samples or a kernel.

        With ``kernel=None``, evaluation returns the total normalized weight
        of samples matching each query, and zero elsewhere.
        With a kernel, evaluation returns the weighted sum of its
        query-to-sample values. Both cases use all dataset columns as inputs
        in their given order and return scalar responses, with ``out_shape=()``.

        Parameters
        ----------
        dataset : torch.Tensor
            Integer indices with shape ``(n_samples, n_sites)``.
        in_dim : sequence of int, optional
            Dimension of each discrete input axis, in dataset column order.
            Inferred from observed indices when omitted without a kernel;
            required when a kernel is supplied.
        weights : torch.Tensor, optional
            Non-negative weights normalized to total mass one.
        kernel : callable, optional
            Receives ``(indices, dataset, **kernel_kwargs)`` and returns
            ``(batch, n_samples)`` values. ``None`` uses exact index matches.
        **kernel_kwargs
            Parameters owned by the selected kernel, such as ``bandwidth``.

        Returns
        -------
        CallableTensorSource
            Scalar tensor source evaluating kernel estimates or exact masses.

        Examples
        --------
        >>> source = tk.decompositions.CallableTensorSource.from_data(
        ...     torch.tensor([[0], [2], [0]]), in_dim=(3,))
        >>> source.contract_dense()
        tensor([0.6667, 0.0000, 0.3333])
        """
        weights = _data_weights(dataset, weights)
        configurations = ConfigurationBatch(dataset)

        if kernel is not None and not callable(kernel):
            raise TypeError('`kernel` should be callable')
        if kernel is None and kernel_kwargs:
            raise ValueError('Kernel parameters require a `kernel`')
        if in_dim is None:
            if kernel is not None:
                raise ValueError('Discrete data callables require `in_dim`')
            in_dim = tuple(int(value) + 1 for value in dataset.amax(0))

        in_dim = _normalize_dims(in_dim, 'in_dim')
        indices = _discrete_indices(configurations, in_dim, dataset.device)
        evaluator = _SparseLookup(indices, weights, in_dim) if kernel is None \
            else _DataEvaluator(dataset, weights, kernel, kernel_kwargs)
        return cls(evaluator, in_dim, out_shape=(),
                   dtype=weights.dtype if kernel is None else None,
                   device=dataset.device)


__all__ = [
    'CallableSource',
    'CallableTensorSource',
]
