"""
This script contains:

    Public classes:
        * CallableQuanticsSource
"""

from math import prod
from typing import Callable, Optional, Sequence, Union

import torch

from tensorkrowch.formats.quantics import _resolve_quantization
from tensorkrowch.formats.quantization import CoordinateMap, Domain, QuantizedLayout

from tensorkrowch.decompositions._utils import _unravel_indices
from tensorkrowch.decompositions.sources.base import (ConfigurationBatch,
                                                      _data_weights)
from tensorkrowch.decompositions.sources.callable import (CallableTensorSource,
                                                         _DataEvaluator)
from tensorkrowch.decompositions.sources.sparse import _SparseLookup


class CallableQuanticsSource(CallableTensorSource):  # MARK: CallableQuanticsSource
    """
    Coordinate callable evaluated through a finite Quantics discretization.

    :meth:`evaluate` receives digit configurations in :class:`QuantizedLayout`
    order. The layout decodes grid indices, and the :class:`CoordinateMap`
    converts them to domain coordinates for ``function``. The callable returns
    one scalar or fixed-shape tensor per configuration. Values are sampled
    without interpolation; validation and batching follow
    :class:`CallableTensorSource`.

    The usual constructor accepts three ways to define the discretization:
    pass ``base``, ``level`` and ``domain`` for a uniform grid with left grid
    points; replace ``domain`` with ``grid_coordinates`` for explicit positions;
    or pass an existing ``layout`` and ``coordinate_map`` together, without
    shorthand arguments. Shorthand layouts use ``ordering="interleaved"`` and
    ``digit_order="coarse_to_fine"``. Each coordinate has ``base ** level``
    grid points; explicit grid sizes must match exactly.

    Use :meth:`matrix` for a scalar function of input and output coordinates
    with independent discretizations on both sides. :meth:`from_data` instead
    quantizes a dataset and constructs its empirical vector distribution,
    optionally smoothed by a kernel. :meth:`contract_dense` retains digit axes;
    :meth:`contract_dense_grid_axes` combines them into one axis per coordinate.

    Parameters
    ----------
    function : callable
        Receives domain coordinates of shape ``(batch, n_coordinates)`` and
        returns one scalar or fixed-shape tensor per configuration.
    n_coordinates : int
        Number of domain coordinates.
    base, level : int or sequence[int], optional
        Digit bases and levels. Required with the domain/grid shorthand.
    domain : pair or sequence of pairs, optional
        Domain intervals for an affine map with left grid points.
    grid_coordinates : torch.Tensor or sequence of torch.Tensor, optional
        Explicit grid positions instead of ``domain``.
    layout : QuantizedLayout, optional
        Digit schedule, supplied together with ``coordinate_map``.
    coordinate_map : CoordinateMap, optional
        Grid mapping, supplied together with ``layout``.
    **kwargs
        Callable output shape, dtype, device and batching options.

    Examples
    --------
    >>> source = tk.decompositions.CallableQuanticsSource(
    ...     lambda coordinates: torch.exp(coordinates.sum(-1)),
    ...     1, base=2, level=2, domain=(0., 1.))
    >>> source.evaluate(tk.decompositions.ConfigurationBatch(
    ...     torch.tensor([[0, 0], [1, 0]])))
    tensor([1.0000, 1.6487])
    >>> source.contract_dense_grid_axes()
    tensor([1.0000, 1.2840, 1.6487, 2.1170])
    """

    def __init__(self,
                 function: Callable,
                 n_coordinates: int,
                 base: Optional[Union[int, Sequence[int]]] = None,
                 level: Optional[Union[int, Sequence[int]]] = None,
                 domain: Domain = None,
                 grid_coordinates: Optional[Union[
                     torch.Tensor, Sequence[torch.Tensor]]] = None,
                 *, layout: Optional[QuantizedLayout] = None,
                 coordinate_map: Optional[CoordinateMap] = None,
                 **kwargs) -> None:
        if not callable(function) or isinstance(function, torch.Tensor) or \
                hasattr(function, '_cores'):
            raise TypeError('`function` should be a coordinate callable')
        self.layout, self.coordinate_map = _resolve_quantization(
            n_coordinates, base=base, level=level, domain=domain,
            grid_coordinates=grid_coordinates, layout=layout,
            coordinate_map=coordinate_map)
        self.n_coordinates = n_coordinates
        kwargs.setdefault('out_shape', None)
        super().__init__(function, self.layout.in_dim, **kwargs)

    @classmethod
    def matrix(cls, function: Callable, in_n_coordinates: int,
               out_n_coordinates: int, **kwargs) -> 'CallableQuanticsSource':
        """
        Constructs a kernel source with independent input/output discretizations.

        Unlike the usual constructor, ``function`` receives two coordinate
        tensors, one for each side, and returns a scalar for each paired row.
        Each side uses the same construction options as the usual constructor,
        prefixed with ``in_`` or ``out_``. For example, ``in_base``, ``in_level``
        and ``in_domain`` define the input grid, while ``out_layout`` and
        ``out_coordinate_map`` can define the output grid explicitly.

        Both layouts must have the same number of digit sites, although their
        coordinate counts and local dimensions can differ. :meth:`evaluate`
        receives only input digits and evaluates the callable at every output
        grid configuration. It returns ``(batch, *out_dim)``, with one output
        digit axis per site. :meth:`contract_dense` interleaves input and output
        digit axes; :meth:`contract_dense_grid_axes` groups input coordinate
        axes before output coordinate axes.

        Parameters
        ----------
        function : callable
            Receives ``(in_coordinates, out_coordinates)`` and returns scalar
            entries with their shared leading batch shape.
        in_n_coordinates, out_n_coordinates : int
            Coordinate counts on each side.
        **kwargs
            ``in_``/``out_`` prefixed ``base``, ``level``, ``domain``,
            ``grid_coordinates``, ``layout`` and ``coordinate_map`` options,
            plus shared ``dtype``, ``device`` and ``batch_size`` options.

        Returns
        -------
        CallableQuanticsSource
            Matrix source with equal numbers of paired digit sites.

        Examples
        --------
        >>> source = tk.decompositions.CallableQuanticsSource.matrix(
        ...     lambda x, y: (x + y).sum(-1), 1, 1,
        ...     in_base=2, in_level=1, in_domain=(0., 1.),
        ...     out_base=2, out_level=1, out_domain=(0., 1.))
        >>> source.evaluate(tk.decompositions.ConfigurationBatch(
        ...     torch.tensor([[1]])))
        tensor([[0.5000, 1.0000]])
        >>> source.contract_dense_grid_axes()
        tensor([[0.0000, 0.5000],
                [0.5000, 1.0000]])
        """
        options = ('base', 'level', 'domain', 'grid_coordinates',
                   'layout', 'coordinate_map')
        inputs = {name: kwargs.pop('in_' + name) for name in options
                  if 'in_' + name in kwargs}
        outputs = {name: kwargs.pop('out_' + name, None) for name in options}
        output_layout, output_map = _resolve_quantization(out_n_coordinates, **outputs)
        source = cls(function, in_n_coordinates, **inputs, **kwargs)
        if source.layout.n_sites != output_layout.n_sites:
            raise ValueError('Matrix layouts should have equal numbers of sites')
        source.in_layout, source.in_coordinate_map = source.layout, source.coordinate_map
        source.out_layout, source.out_coordinate_map = output_layout, output_map
        source.in_n_coordinates = in_n_coordinates
        source.out_n_coordinates = out_n_coordinates
        source._matrix_dims = source.in_dim, output_layout.in_dim
        source._out_shape = output_layout.in_dim
        return source

    @classmethod
    def from_data(cls, dataset: torch.Tensor, n_coordinates: int,
                  weights: Optional[torch.Tensor] = None,
                  *, kernel: Optional[Callable] = None,
                  **kwargs) -> 'CallableQuanticsSource':
        """
        Discretizes samples before constructing their empirical vector tensor.

        Pass the same discretization options as in the usual constructor.
        The coordinate map assigns each dataset row to a grid index before
        accumulating its normalized weight. Samples mapped to the same grid
        point contribute to the same mass. Without a kernel, evaluation
        returns that mass or zero at unobserved grid points.

        With ``kernel``, the weighted kernel is centered at the occupied grid
        points, rather than the original sample coordinates. Its values are
        sampled on the grid without renormalizing their sum. Additional keyword
        arguments, such as ``bandwidth``, are passed to the kernel. Both cases
        construct vector sources with ``out_shape=()``; this constructor does
        not create matrix sources.

        Parameters
        ----------
        dataset : torch.Tensor
            Domain coordinates of shape ``(samples, n_coordinates)``.
        n_coordinates : int
            Coordinate count, consistent with the dataset.
        weights : torch.Tensor, optional
            Non-negative sample weights, normalized to total mass one.
        kernel : callable, optional
            Evaluates query coordinates against the dataset after discretizing
            its coordinates. ``None`` returns exact empirical masses.
        **kwargs
            Quantization arguments as in the constructor, and kernel parameters
            such as ``bandwidth`` for :func:`gaussian_kernel`.

        Returns
        -------
        CallableQuanticsSource
            Discretized empirical masses or kernel estimates.

        Examples
        --------
        >>> source = tk.decompositions.CallableQuanticsSource.from_data(
        ...     torch.tensor([[0.], [0.], [0.5]]), 1,
        ...     base=2, level=1, domain=(0., 1.))
        >>> source.contract_dense_grid_axes()
        tensor([0.6667, 0.3333])

        Kernel estimates use the discretized samples as centers:

        >>> source = tk.decompositions.CallableQuanticsSource.from_data(
        ...     torch.tensor([[0.1], [0.2]]), 1,
        ...     base=2, level=1, domain=(0., 1.),
        ...     kernel=tk.decompositions.gaussian_kernel, bandwidth=1.)
        >>> source.contract_dense_grid_axes()
        tensor([0.3989, 0.3521])
        """
        if kernel is not None and not callable(kernel):
            raise TypeError('`kernel` should be callable')
        mass = _data_weights(dataset, weights,
                             dataset.dtype if dataset.is_floating_point() else None)
        options = {name: kwargs.pop(name, None) for name in
                   ('base', 'level', 'domain', 'grid_coordinates',
                    'layout', 'coordinate_map')}
        if kernel is None and kwargs:
            raise TypeError('Kernel arguments require `kernel`')
        layout, coordinate_map = _resolve_quantization(n_coordinates, **options)
        indices = coordinate_map.to_indices(dataset)
        lookup = _SparseLookup(indices, mass, layout.grid_size)

        def empirical(coordinates: torch.Tensor) -> torch.Tensor:
            return lookup(coordinate_map.to_indices(coordinates))

        function = empirical if kernel is None else _DataEvaluator(
            coordinate_map.from_indices(lookup.indices), lookup.values,
            kernel, kwargs)
        return cls(function, n_coordinates, layout=layout,
                   coordinate_map=coordinate_map, out_shape=(),
                   dtype=mass.dtype if kernel is None else None, device=dataset.device)

    def _evaluate_batch(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """Decodes digit inputs before using the shared result validation."""
        indices = self.layout.decode_digits(configurations.as_tensor())
        coordinates = self.coordinate_map.from_indices(indices).to(self.device)
        if not self.is_matrix:
            result = self.function(coordinates)
        else:
            outputs = _unravel_indices(torch.arange(prod(self.out_shape),
                                                    device=self.device), self.out_shape)
            output_indices = self.out_layout.decode_digits(outputs)
            output_coordinates = self.out_coordinate_map.from_indices(
                output_indices).to(self.device)
            n_outputs = outputs.shape[0]
            result = self.function(coordinates.repeat_interleave(n_outputs, 0),
                                   output_coordinates.repeat(coordinates.shape[0], 1))
            if not isinstance(result, torch.Tensor):
                raise TypeError('`function` should return a torch.Tensor')
            result = result.reshape(coordinates.shape[0], *self.out_shape)
        return self._validate_result(result, configurations.batch_size)

    def contract_dense_grid_axes(self) -> torch.Tensor:
        """
        Materializes the same tensor with one axis per original grid coordinate.

        Returns
        -------
        torch.Tensor
            Coordinate grid axes followed by function response axes. For a
            matrix, all input coordinate axes precede output coordinate axes.
        """
        dense = self.contract_dense()
        if self.is_matrix:
            inputs = sorted(range(self.layout.n_sites),
                            key=self.layout.sites().__getitem__)
            outputs = sorted(range(self.out_layout.n_sites),
                             key=self.out_layout.sites().__getitem__)
            dense = dense.permute(*(2 * site for site in inputs),
                                  *(2 * site + 1 for site in outputs))
            return dense.reshape(*self.layout.grid_size, *self.out_layout.grid_size)
        order = sorted(range(self.layout.n_sites), key=self.layout.sites().__getitem__)
        dense = dense.permute(*order, *range(len(order), dense.ndim))
        return dense.reshape(*self.layout.grid_size, *self.out_shape)


__all__ = ['CallableQuanticsSource']
