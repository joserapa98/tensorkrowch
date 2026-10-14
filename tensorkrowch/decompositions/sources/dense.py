"""
This script contains:

    Public classes:
        * DenseTensorSource

    Internal classes/functions:
        * _MatrixInput
        * _normalize_dim
        * _prepare_matrix_input
"""

from dataclasses import dataclass
from math import prod
from typing import Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.decompositions.sources.base import (ConfigurationBatch,
                                                      _SourceEvaluationTracker,
                                                      _discrete_indices,
                                                      _fiber_configurations)


_Dimension = Optional[Union[int, Sequence[int]]]


def _normalize_dim(dim: _Dimension, name: str) -> Tuple[int, ...]:
    """Normalizes one or several positive site dimensions."""
    if isinstance(dim, bool):
        raise TypeError(f'`{name}` should be int or a sequence of ints')
    if isinstance(dim, int):
        values = (dim,)
    elif isinstance(dim, Sequence) and not isinstance(dim, (str, bytes)):
        values = tuple(dim)
    else:
        raise TypeError(f'`{name}` should be int or a sequence of ints')

    if not values:
        raise ValueError(f'`{name}` should contain at least one dimension')
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in values):
        raise TypeError(f'`{name}` should contain only ints')
    if any(value < 1 for value in values):
        raise ValueError(f'`{name}` should contain only positive dimensions')
    return values


@dataclass(frozen=True)
class _MatrixInput:  # MARK: _MatrixInput
    """Contains a normalized dense matrix input."""

    in_dim: Tuple[int, ...]  # Input dimension of every matrix site
    out_dim: Tuple[int, ...]  # Output dimension of every matrix site
    interleaved: torch.Tensor  # Tensor ordered as in_1, out_1, ..., in_n, out_n
    matrix_input: bool  # Whether the original tensor had two matrix axes


def _prepare_matrix_input(tensor: torch.Tensor,
                          in_dim: _Dimension,
                          out_dim: _Dimension,
                          ordering: str,
                          family: str) -> _MatrixInput:
    """Validates and normalizes a dense matrix or matrix-like ``tensor``."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('`tensor` should be torch.Tensor type')
    if not isinstance(ordering, str):
        raise TypeError('`ordering` should be str type')
    if ordering not in ('interleaved', 'grouped'):
        raise ValueError(
            '`ordering` should be either "interleaved" or "grouped"')
    if (in_dim is None) != (out_dim is None):
        raise ValueError(
            '`in_dim` and `out_dim` should be provided together')

    matrix_input = False
    if in_dim is None:
        if (tensor.ndim < 2) or (tensor.ndim % 2):
            raise ValueError(
                f'A tensorized {family} input should have a positive even '
                'number of dimensions')
        n_sites = tensor.ndim // 2
        if ordering == 'interleaved':
            normalized_in_dim = tuple(tensor.shape[::2])
            normalized_out_dim = tuple(tensor.shape[1::2])
        else:
            normalized_in_dim = tuple(tensor.shape[:n_sites])
            normalized_out_dim = tuple(tensor.shape[n_sites:])
        if any(value < 1
               for value in normalized_in_dim + normalized_out_dim):
            raise ValueError(
                f'{family} input and output dimensions should be positive')
        tensorized = tensor
        tensorized_ordering = ordering
    else:
        normalized_in_dim = _normalize_dim(in_dim, 'in_dim')
        normalized_out_dim = _normalize_dim(out_dim, 'out_dim')
        if len(normalized_in_dim) != len(normalized_out_dim):
            raise ValueError(
                '`in_dim` and `out_dim` should have the same length')
        n_sites = len(normalized_in_dim)

        if tensor.ndim == 2:
            expected_shape = (
                prod(normalized_in_dim),
                prod(normalized_out_dim),
            )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    'The matrix shape should equal '
                    '(prod(`in_dim`), prod(`out_dim`))')
            tensorized = tensor.reshape(
                *normalized_in_dim, *normalized_out_dim)
            tensorized_ordering = 'grouped'
            matrix_input = True
        else:
            if tensor.ndim != (2 * n_sites):
                raise ValueError(
                    f'A tensorized {family} input should have two '
                    'dimensions per site')
            expected_shape = (
                tuple(value
                      for pair in zip(normalized_in_dim,
                                      normalized_out_dim)
                      for value in pair)
                if ordering == 'interleaved'
                else normalized_in_dim + normalized_out_dim
            )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    'The tensor shape is incompatible with `in_dim`, '
                    '`out_dim` and `ordering`')
            tensorized = tensor
            tensorized_ordering = ordering

    if (tensorized_ordering == 'interleaved') or (n_sites == 1):
        interleaved = tensorized
    else:
        axes = tuple(
            axis
            for site in range(n_sites)
            for axis in (site, n_sites + site))
        interleaved = tensorized.permute(axes)

    return _MatrixInput(in_dim=normalized_in_dim,
                        out_dim=normalized_out_dim,
                        interleaved=interleaved,
                        matrix_input=matrix_input)


class DenseTensorSource(_SourceEvaluationTracker):  # MARK: DenseTensorSource
    """
    Tensor source backed by an explicitly stored dense tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor containing input and optional output axes.
    in_sites : sequence of int, optional
        Tensor axes used as input sites, in configuration order. If omitted,
        every axis is an input site and the source is scalar. Remaining axes
        form the output, in their original order.
    """

    def __init__(self,
                 tensor: torch.Tensor,
                 in_sites: Optional[Sequence[int]] = None,
                 *, n_batches: int = 0) -> None:
        self._initialize_evaluation_stats()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0 or n_batches >= tensor.ndim:
            raise ValueError('`n_batches` should leave input axes')
        self.n_batches = n_batches
        n_axes = tensor.ndim - n_batches
        if tensor.ndim < 1:
            raise ValueError('`tensor` should contain at least one input site')
        if in_sites is None:
            in_sites = tuple(range(n_axes))
        elif not isinstance(in_sites, (list, tuple)):
            raise TypeError('`in_sites` should be a list or tuple of ints')
        else:
            in_sites = tuple(in_sites)
        if not in_sites:
            raise ValueError('`in_sites` should contain at least one axis')
        if any(isinstance(axis, bool) or not isinstance(axis, int) or
               axis < 0 or axis >= n_axes for axis in in_sites):
            raise ValueError('`in_sites` should contain valid tensor axes')
        if len(set(in_sites)) != len(in_sites):
            raise ValueError('`in_sites` should not contain duplicate axes')
        if any(tensor.shape[n_batches + axis] < 1 for axis in in_sites):
            raise ValueError('Input dimensions should be positive')
        out_sites = tuple(axis for axis in range(n_axes)
                             if axis not in in_sites)
        self.tensor = tensor
        self._in_sites = in_sites
        self._out_sites = out_sites
        self._ordered_tensor = tensor.permute(
            *(n_batches + axis for axis in in_sites),
            *range(n_batches), *(n_batches + axis for axis in out_sites))
        self._in_dim = tuple(tensor.shape[n_batches + axis] for axis in in_sites)
        self._out_shape = tuple(tensor.shape[n_batches + axis] for axis in out_sites)

    @property
    def in_sites(self) -> Tuple[int, ...]:
        """Tensor axes used as input sites, in configuration order."""
        return self._in_sites

    @property
    def out_sites(self) -> Tuple[int, ...]:
        """Remaining tensor axes, in their original order."""
        return self._out_sites

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Discrete input dimension at every site."""
        return self._in_dim

    @property
    def out_shape(self) -> Tuple[int, ...]:
        """Shape returned after the configuration batch."""
        return self._out_shape

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the stored tensor."""
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        """Device of the stored tensor."""
        return self.tensor.device

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Gathers dense values at discrete global ``configurations``."""
        indices = _discrete_indices(
            configurations, self._in_dim, self.device)
        result = self._ordered_tensor[tuple(
            indices[:, site] for site in range(indices.shape[1]))]
        self._record_evaluation(points=indices.shape[0])
        return result

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluates a discrete ``site`` fiber for every base configuration."""
        if not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= len(self._in_dim)):
            raise ValueError('`site` should identify an input site')
        if values is None:
            values = torch.arange(
                self._in_dim[site], device=configurations.device)
        expanded, n_values = _fiber_configurations(
            configurations, site, values)
        result = self.evaluate(expanded)
        return result.reshape(
            configurations.batch_size, n_values, *self._out_shape)


    def contract_dense(self) -> torch.Tensor:
        """Returns the stored tensor without copying its tensor storage."""
        return self._matrix_tensor if self.is_matrix else self.tensor

    @classmethod
    def matrix(cls, tensor: torch.Tensor,
               in_dim: _Dimension = None, out_dim: _Dimension = None,
               *, ordering: str = 'interleaved') -> 'DenseTensorSource':
        """
        Constructs a paired matrix source from dense matrix or tensor axes.

        Parameters
        ----------
        tensor : torch.Tensor
            Ordinary matrix or tensor with paired input/output site axes.
        in_dim, out_dim : sequence[int], optional
            Local matrix dimensions; required for an ordinary two-axis matrix.
        ordering : {``"interleaved"``, ``"grouped"``}
            Ordering of tensorized axes; defaults to ``"interleaved"``.

        Returns
        -------
        DenseTensorSource
            Source materializing interleaved pairs for matrix decomposition.
        """
        prepared = _prepare_matrix_input(tensor, in_dim, out_dim, ordering, 'matrix')
        source = cls(prepared.interleaved,
                     tuple(range(0, prepared.interleaved.ndim, 2)))
        source._matrix_dims = prepared.in_dim, prepared.out_dim
        source._matrix_tensor = prepared.interleaved
        source._matrix_input = prepared.matrix_input
        source._ordering = ordering
        source.tensor = tensor
        return source


__all__ = [
    'DenseTensorSource',
]
