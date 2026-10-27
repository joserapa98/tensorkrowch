"""
This script contains:

    Public classes:
        * DenseTensorSource
"""

from typing import Optional, Sequence, Tuple

import torch

from tensorkrowch.decompositions.sources.base import (BaseSource,
                                                      ConfigurationBatch,
                                                      _discrete_indices)


class DenseTensorSource(BaseSource):  # MARK: DenseTensorSource
    """
    Tensor source backed by an explicitly stored dense tensor.

    The usual constructor represents a tensor with one axis per vector site.
    By default, :meth:`evaluate` receives one integer index per axis and returns
    a scalar for each configuration. Pass ``in_sites`` to evaluate only some
    axes: configurations have shape ``(batch, len(in_sites))`` in that
    selection's order, and the remaining axes form ``out_shape`` in their
    original order. Leading structural batch axes are excluded from the
    variable numbering. ``n_sites`` counts all non-batch axes, including those
    returned as outputs.

    Use :meth:`matrix` instead to represent a matrix with one input/output
    axis pair per site. The tensor must already have its input/output axes
    separated; a two-axis tensor represents a single site. This grouping sets
    ``is_matrix=True`` and provides ``in_dim`` and ``out_dim`` for matrix
    decompositions. A matrix source evaluates the input indices and returns
    all output axes, with ``out_shape == out_dim``.

    Parameters
    ----------
    tensor : torch.Tensor
        Dense tensor containing input and optional output axes.
    n_batches : int
        Number of leading structural batch axes. Defaults to ``0``.
    in_sites : sequence of int, optional
        Distinct non-batch axes, in configuration order. Defaults to all
        non-batch axes in increasing order.

    Examples
    --------
    >>> tensor = torch.arange(6.).reshape(2, 3)
    >>> source = tk.decompositions.DenseTensorSource(tensor)
    >>> source.evaluate(tk.decompositions.ConfigurationBatch(
    ...     torch.tensor([[1, 2]])))
    tensor([5.])

    Keeping the second axis as output returns a vector per configuration:

    >>> source = tk.decompositions.DenseTensorSource(tensor, in_sites=(0,))
    >>> source.out_shape
    (3,)
    >>> source.evaluate(tk.decompositions.ConfigurationBatch(
    ...     torch.tensor([[1]])))
    tensor([[3., 4., 5.]])
    """

    def __init__(self,
                 tensor: torch.Tensor,
                 n_batches: int = 0,
                 *,
                 in_sites: Optional[Sequence[int]] = None) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if (n_batches < 0) or (n_batches >= tensor.ndim):
            raise ValueError('`n_batches` should leave input axes')
        n_axes = tensor.ndim - n_batches
        if tensor.ndim < 1:
            raise ValueError('`tensor` should contain at least one input site')
        super().__init__(n_axes, in_sites=in_sites, n_batches=n_batches)
        in_sites = self.in_sites
        if any(tensor.shape[n_batches + axis] < 1 for axis in in_sites):
            raise ValueError('Input dimensions should be positive')

        self.tensor = tensor
        self._in_sites = in_sites
        out_axes = tuple(axis for axis in range(n_axes) if axis not in in_sites)
        self._ordered_tensor = tensor.permute(
            *(n_batches + axis for axis in in_sites),
            *range(n_batches), *(n_batches + axis for axis in out_axes))
        self._in_dim = tuple(tensor.shape[n_batches:])

    @classmethod
    def matrix(cls,
               tensor: torch.Tensor,
               *,
               ordering: str = 'grouped') -> 'DenseTensorSource':
        """
        Constructs a paired matrix source from dense matrix or tensor axes.

        Each site has an input dimension ``in_dim[i]`` and an output dimension
        ``out_dim[i]``. Input indices are evaluated together; the returned
        tensor contains every output axis in site order. :meth:`evaluate`
        receives shape ``(batch, n_sites)`` in site order and returns all
        output axes, with ``out_shape == out_dim``.

        The tensor must have an even number of axes, with one input/output
        pair per site. ``ordering="interleaved"`` means axes
        ``(in_0, out_0, in_1, out_1, ...)``; ``ordering="grouped"`` means axes
        ``(in_0, in_1, ..., out_0, out_1, ...)``. Local dimensions are inferred
        directly from these axes. A two-axis tensor represents one site;
        separate its axes explicitly with ``reshape`` to represent more sites.
        Structural batch axes are not supported by this constructor.

        The source exposes paired dimensions and :meth:`contract_dense`
        returns the tensor with interleaved axes, regardless of input ordering.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor with explicit input/output axes in the specified ``ordering``.
        ordering : {``"grouped"``, ``"interleaved"``}
            Ordering of tensorized axes; defaults to ``"grouped"``.

        Returns
        -------
        DenseTensorSource
            Matrix source with ``out_shape == out_dim``.

        Examples
        --------
        Explicitly separate a matrix into two sites before constructing the source:

        >>> source = tk.decompositions.DenseTensorSource.matrix(
        ...     torch.eye(4).reshape(2, 2, 2, 2))
        >>> source.evaluate(tk.decompositions.ConfigurationBatch(
        ...     torch.tensor([[0, 1]])))
        tensor([[[0., 1.],
                 [0., 0.]]])

        Infer dimensions from grouped axes, then retrieve interleaved axes:

        >>> source = tk.decompositions.DenseTensorSource.matrix(
        ...     torch.arange(24.).reshape(2, 3, 2, 2))
        >>> source.in_dim, source.out_dim
        ((2, 3), (2, 2))
        >>> source.contract_dense().shape
        torch.Size([2, 2, 3, 2])
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if not isinstance(ordering, str):
            raise TypeError('`ordering` should be str type')
        if ordering not in ('interleaved', 'grouped'):
            raise ValueError(
                '`ordering` should be either "interleaved" or "grouped"')
        if (tensor.ndim < 2) or (tensor.ndim % 2):
            raise ValueError(
                'A matrix source should have a positive even number of dimensions')
        if any(size < 1 for size in tensor.shape):
            raise ValueError('Input and output dimensions should be positive')

        n_sites = tensor.ndim // 2
        interleaved = tensor
        if ordering == 'grouped':
            axes = tuple(axis for site in range(n_sites)
                         for axis in (site, n_sites + site))
            interleaved = tensor.permute(axes)

        source = cls(interleaved, in_sites=tuple(range(0, tensor.ndim, 2)))
        source._matrix_dims = (tuple(interleaved.shape[::2]),
                               tuple(interleaved.shape[1::2]))
        source._in_dim = source._matrix_dims[0]
        source._set_sites(n_sites)
        source._matrix_tensor = interleaved
        source._ordering = ordering
        source.tensor = tensor
        return source

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Dimensions of all vector variables or matrix input axes."""
        return self._in_dim

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the stored tensor."""
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        """Device of the stored tensor."""
        return self.tensor.device

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Gathers values for integer configurations of shape
        ``(batch, len(in_sites))``, with columns ordered by ``in_sites``.
        """
        indices = _discrete_indices(
            configurations, self._in_sites_dims, self.device)
        result = self._ordered_tensor[tuple(
            indices[:, site] for site in range(indices.shape[1]))]
        self._record_evaluation(points=indices.shape[0])
        return result

    def contract_dense(self) -> torch.Tensor:
        """Returns the stored tensor without copying its tensor storage."""
        return self._matrix_tensor if self.is_matrix else self.tensor


__all__ = [
    'DenseTensorSource',
]
