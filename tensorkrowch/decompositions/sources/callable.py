"""Callable tensor sources with deterministic evaluation batching."""

from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.decompositions.sources.base import (
    ConfigurationBatch,
    _discrete_indices,
    _fiber_configurations,
    _normalize_input_dim,
)


class CallableTensorSource:
    """Tensor source evaluated by a user callable.

    Packed configurations are passed to ``function`` as a tensor. A
    heterogeneous configuration batch is passed as a tuple containing one
    tensor per site. The callable must preserve the leading configuration
    batch and return shape ``(batch, *output_shape)``.

    Parameters
    ----------
    function : callable
        Function evaluated on configuration batches.
    input_dim : sequence of int
        Discrete input dimension at every site. These dimensions also provide
        the default grid for discrete fibers.
    output_shape : sequence of int or None, optional
        Declared function output shape. The default ``()`` denotes a scalar.
        ``None`` infers the shape from the first non-empty evaluation.
    dtype : torch.dtype or None, optional
        Declared output dtype. ``None`` infers it from the first evaluation.
    device : str or torch.device, optional
        Device on which configurations and returned values must live.
    batch_size : int or None, optional
        Maximum number of configurations passed to one callable invocation.
    """

    def __init__(self,
                 function: Callable,
                 input_dim: Sequence[int],
                 output_shape: Optional[Sequence[int]] = (),
                 dtype: Optional[torch.dtype] = None,
                 device: Union[str, torch.device] = 'cpu',
                 batch_size: Optional[int] = None) -> None:
        if not callable(function):
            raise TypeError('`function` should be callable')
        if output_shape is None:
            normalized_output_shape = None
        else:
            if isinstance(output_shape, (str, bytes)):
                raise TypeError(
                    '`output_shape` should be a sequence of integers or None')
            try:
                normalized_output_shape = tuple(output_shape)
            except TypeError as exc:
                raise TypeError(
                    '`output_shape` should be a sequence of integers or None') \
                    from exc
            if any((not isinstance(dim, int)) or (dim < 1)
                   for dim in normalized_output_shape):
                raise ValueError(
                    '`output_shape` should contain positive integers')
        if (dtype is not None) and (not isinstance(dtype, torch.dtype)):
            raise TypeError('`dtype` should be torch.dtype type or None')
        if batch_size is not None:
            if (not isinstance(batch_size, int)) or (batch_size < 1):
                raise ValueError('`batch_size` should be a positive integer')

        self.function = function
        self._input_dim = _normalize_input_dim(input_dim)
        self._output_shape = normalized_output_shape
        self._dtype = dtype
        self._device = torch.device(device)
        self.batch_size = batch_size

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Discrete input dimension at every site."""
        return self._input_dim

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """Declared or inferred function output shape."""
        return self._output_shape

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Declared or inferred function output dtype."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Device on which evaluations are performed."""
        return self._device

    def _evaluate_batch(
            self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Evaluates and validates one callable batch."""
        argument = configurations.values
        result = self.function(argument)
        if not isinstance(result, torch.Tensor):
            raise TypeError('`function` should return a torch.Tensor')
        if result.device != self.device:
            raise ValueError(
                '`function` should return values on the source device')
        if (result.ndim < 1) or \
                (result.shape[0] != configurations.batch_size):
            raise ValueError(
                '`function` should preserve the configuration batch dimension')

        output_shape = tuple(result.shape[1:])
        if self._output_shape is None:
            self._output_shape = output_shape
        elif output_shape != self._output_shape:
            raise ValueError(
                '`function` output shape does not match `output_shape`')
        if self._dtype is None:
            self._dtype = result.dtype
        elif result.dtype != self._dtype:
            raise ValueError('`function` output dtype changed between calls')
        return result

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Evaluates configurations in deterministic contiguous batches."""
        if not isinstance(configurations, ConfigurationBatch):
            raise TypeError(
                '`configurations` should be ConfigurationBatch type')
        if configurations.n_sites != len(self.input_dim):
            raise ValueError(
                'Configurations should contain one value per input site')
        configurations = configurations.to(self.device)
        if configurations.kind == 'indices':
            _discrete_indices(
                configurations, self.input_dim, self.device)

        if configurations.batch_size == 0:
            if (self.output_shape is None) or (self.dtype is None):
                raise ValueError(
                    'An empty first evaluation requires output shape and dtype')
            return torch.empty(
                (0, *self.output_shape),
                device=self.device,
                dtype=self.dtype)

        batch_size = self.batch_size or configurations.batch_size
        chunks = []
        for start in range(0, configurations.batch_size, batch_size):
            stop = min(start + batch_size, configurations.batch_size)
            ids = torch.arange(start, stop, device=configurations.device)
            chunks.append(self._evaluate_batch(
                configurations.index_select(ids)))
        return torch.cat(chunks, dim=0)

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates a callable fiber over explicit or discrete site values."""
        if not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= len(self.input_dim)):
            raise ValueError('`site` should identify an input site')
        if values is None:
            if configurations.kind != 'indices':
                raise ValueError(
                    'Coordinate fibers require explicit `values`')
            values = torch.arange(
                self.input_dim[site], device=configurations.device)
        expanded, n_values = _fiber_configurations(
            configurations, site, values)
        result = self.evaluate(expanded)
        return result.reshape(
            configurations.batch_size, n_values, *result.shape[1:])


__all__ = ['CallableTensorSource']
