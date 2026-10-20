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
                                                      _discrete_indices,
                                                      _data_weights)
from tensorkrowch.decompositions.sources.sparse import SparseTensorSource


class CallableSource(BaseSource):  # MARK: CallableSource
    """
    Continuous function evaluated by a user callable.

    Packed configurations are passed to ``function`` as a tensor. A
    heterogeneous configuration batch is passed as a tuple containing one
    tensor per site. The callable must preserve the leading configuration batch
    and return shape ``(batch, *out_shape)``.

    Parameters
    ----------
    function : callable
        Function evaluated on configuration batches.
    n_sites : int
        Number of continuous input sites. Each site can receive a scalar or
        an embedded vector.
    out_shape : sequence of int or None, optional
        Declared response shape. The default ``None`` discovers it on the
        first non-empty evaluation; ``()`` requires scalar responses.
    dtype : torch.dtype or None, optional
        Declared output dtype. ``None`` infers it from the first evaluation.
    device : str or torch.device, optional
        Device on which configurations and returned values must live.
    batch_size : int or None, optional
        Maximum number of configurations passed to one callable invocation.
    """

    input_kind: str = 'features'

    def __init__(self,
                 function: Callable,
                 n_sites: int,
                 out_shape: Optional[Sequence[int]] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Union[str, torch.device] = 'cpu',
                 batch_size: Optional[int] = None) -> None:
        super().__init__(n_sites)
        if not callable(function):
            raise TypeError('`function` should be callable')
        normalized_out_shape = None if out_shape is None else \
            _normalize_dims(out_shape, 'out_shape', allow_empty=True)
        if (dtype is not None) and (not isinstance(dtype, torch.dtype)):
            raise TypeError('`dtype` should be torch.dtype type or None')
        if batch_size is not None:
            if (not isinstance(batch_size, int)) or (batch_size < 1):
                raise ValueError('`batch_size` should be a positive integer')

        self.function = function
        self._in_dim = None
        self._out_shape = normalized_out_shape
        self._dtype = dtype
        self._device = torch.empty(0, device=device).device
        self.batch_size = batch_size

    @property
    def in_dim(self) -> Optional[Tuple[int, ...]]:
        """Discrete dimensions, or ``None`` for continuous features."""
        return self._in_dim

    @property
    def out_shape(self) -> Optional[Tuple[int, ...]]:
        """Declared or inferred function output shape."""
        return self._out_shape

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Declared or inferred function output dtype."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device on which evaluations are performed."""
        return self._device

    def _evaluate_batch(self,
                        configurations: ConfigurationBatch) -> torch.Tensor:
        """Evaluates and validates one callable batch."""
        argument = configurations.values
        result = self.function(argument)
        return self._validate_result(result, configurations.batch_size)

    def _validate_result(self, result: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Checks batch shape and stable result runtime metadata."""
        if not isinstance(result, torch.Tensor):
            raise TypeError('`function` should return a torch.Tensor')
        if result.device != self._device:
            raise ValueError(
                '`function` should return values on the source device')
        if (result.ndim < 1) or \
                (result.shape[0] != batch_size):
            raise ValueError(
                '`function` should preserve the configuration batch dimension')

        out_shape = tuple(result.shape[1:])
        if self._out_shape is None:
            self._out_shape = out_shape
        elif out_shape != self._out_shape:
            raise ValueError(
                '`function` output shape does not match `out_shape`')
        if self._dtype is None:
            self._dtype = result.dtype
        elif result.dtype != self._dtype:
            raise ValueError('`function` output dtype changed between calls')
        return result

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Evaluates configurations using the source's input convention.

        Parameters
        ----------
        configurations : ConfigurationBatch
            Continuous features for ``CallableSource``; discrete indices for
            ``CallableTensorSource`` and its Quantics subclass.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch, *out_shape)``, preserving input order
            and autograd. ``batch_size`` bounds each callable invocation.
        """
        if not isinstance(configurations, ConfigurationBatch):
            raise TypeError(
                '`configurations` should be ConfigurationBatch type')
        if configurations.n_sites != self.n_sites:
            raise ValueError(
                'Configurations should contain one value per input site')
        configurations = configurations.to(self._device)
        if configurations.kind != self.input_kind:
            raise ValueError(f'Configurations should have kind `{self.input_kind}`')
        if self.input_kind == 'indices':
            _discrete_indices(
                configurations, self._in_dim, self._device)

        if configurations.batch_size == 0:
            if (self._out_shape is None) or (self._dtype is None):
                raise ValueError(
                    'An empty first evaluation requires output shape and dtype')
            result = torch.empty(
                (0, *self._out_shape),
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
            chunks.append(self._evaluate_batch(
                configurations.index_select(ids)))
        result = torch.cat(chunks, dim=0)
        self._record_evaluation(
            points=configurations.batch_size, batches=len(chunks))
        return result

    @classmethod
    def from_data(cls, dataset: torch.Tensor,
                  weights: Optional[torch.Tensor] = None,
                  *, kernel: Optional[Callable] = None,
                  in_dim: Optional[Sequence[int]] = None,
                  **kernel_kwargs) -> 'CallableSource':
        """
        Constructs a scalar vector source from exact samples or a kernel.

        Parameters
        ----------
        dataset : torch.Tensor
            Real sample coordinates, or indices for a discrete callable.
        weights : torch.Tensor, optional
            Non-negative weights normalized to total mass one.
        kernel : callable, optional
            Receives ``(coordinates, dataset, **kernel_kwargs)`` and returns
            ``(batch, samples)`` values. ``None`` uses exact coordinate matches.
        in_dim : sequence[int], optional
            Required for ``CallableTensorSource``; absent for continuous sources.
        **kernel_kwargs
            Parameters owned by the selected kernel, such as ``bandwidth``.

        Returns
        -------
        CallableSource or CallableTensorSource
            Source reusing the class's validation, batching and statistics.

        Examples
        --------
        >>> source = tk.decompositions.CallableSource.from_data(
        ...     torch.tensor([[0.1], [0.4], [0.1]]))
        >>> source.evaluate(tk.decompositions.ConfigurationBatch(
        ...     torch.tensor([[0.1], [0.3]]), kind='features'))
        tensor([0.6667, 0.0000])
        """
        mass = _data_weights(dataset, weights,
                             dataset.dtype if dataset.is_floating_point() else None)
        if kernel is not None and not callable(kernel):
            raise TypeError('`kernel` should be callable')
        if kernel is None and kernel_kwargs:
            raise ValueError('Kernel parameters require a `kernel`')
        if cls.input_kind == 'indices' and kernel is None:
            return SparseTensorSource.from_data(dataset, in_dim=in_dim, weights=weights)
        evaluator = _DataEvaluator(dataset, mass, kernel, kernel_kwargs)
        if cls.input_kind == 'indices':
            if in_dim is None:
                raise ValueError('Discrete data callables require `in_dim`')
            return cls(evaluator, in_dim, out_shape=(), dtype=mass.dtype if kernel is None else None,
                       device=dataset.device)
        if in_dim is not None:
            raise ValueError('Continuous data callables do not define `in_dim`')
        return cls(evaluator, dataset.shape[1], out_shape=(), dtype=mass.dtype if kernel is None else None,
                   device=dataset.device)

    @classmethod
    def matrix(cls, function: Callable, in_dim: Sequence[int],
               out_dim: Sequence[int], **kwargs) -> 'CallableSource':
        """
        Constructs a callable with one paired output dimension per input.

        Parameters
        ----------
        function : callable
            Evaluates input configurations and returns all output axes.
        in_dim, out_dim : sequence[int]
            Paired local dimensions. Output shape must match ``out_dim``.
        **kwargs
            Callable runtime options.

        Returns
        -------
        CallableSource or CallableTensorSource
            Source with explicit local matrix grouping.
        """
        inputs = _normalize_dims(in_dim, 'in_dim')
        outputs = _normalize_dims(out_dim, 'out_dim')
        if len(inputs) != len(outputs):
            raise ValueError('Matrix dimensions should have equal lengths')
        source = cls(function, inputs if cls.input_kind == 'indices' else len(inputs),
                     out_shape=outputs, **kwargs)
        source._matrix_dims = inputs, outputs
        source._in_dim = inputs
        return source


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
                raise ValueError('`kernel` should return `(batch, samples)` values')
        dtype = torch.promote_types(values.dtype, self.weights.dtype)
        return values.to(dtype) @ self.weights.to(device=values.device, dtype=dtype)


class CallableTensorSource(CallableSource):  # MARK: CallableTensorSource
    """
    Callable-backed finite tensor evaluated on discrete indices.

    Parameters
    ----------
    function : callable
        Receives integer configurations and preserves their leading batch axis.
    in_dim : sequence[int]
        Positive finite dimensions of the indexed input sites.
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

    def __init__(self, function: Callable, in_dim: Sequence[int], **kwargs) -> None:
        dimensions = _normalize_dims(in_dim, 'in_dim')
        super().__init__(function, len(dimensions), **kwargs)
        self._in_dim = dimensions


__all__ = [
    'CallableSource',
    'CallableTensorSource',
]
