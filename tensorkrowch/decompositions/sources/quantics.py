"""
This script contains:

    Classes:
        * _QuanticsSource
        * QuanticsVectorSource
        * QuanticsMatrixSource

    Callable evaluation:
        coordinates -> CoordinateMap -> indices -> QuantizedLayout -> digits
        digits -> QuantizedLayout -> indices -> CoordinateMap -> function
"""

from math import prod
from typing import Callable, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.formats import CoordinateMap, QuantizedLayout, TensorFormat
from tensorkrowch.formats.quantics import _resolve_quantization
from tensorkrowch.formats.quantization import Domain

from tensorkrowch.decompositions.metrics import EvaluationStats
from tensorkrowch.decompositions.sources.base import (ConfigurationBatch,
                                                      TensorSource,
                                                      _fiber_configurations,
                                                      _unravel_indices)
from tensorkrowch.decompositions.sources.callable import CallableTensorSource


class _QuanticsSource:  # MARK: _QuanticsSource
    """Shares scalar callable evaluation, batching and runtime information."""

    def __init__(self,
                 function: Callable,
                 in_dim: Sequence[int],
                 dtype: Optional[torch.dtype],
                 device: Union[str, torch.device],
                 batch_size: Optional[int]) -> None:
        if not callable(function) or isinstance(function, (TensorSource,
                                                         TensorFormat)):
            raise TypeError('`function` should be a coordinate callable')

        self.function = function
        self._callable = CallableTensorSource(
            self._evaluate_scalar, in_dim, out_shape=(), dtype=dtype,
            device=device, batch_size=batch_size)

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Callable output dtype, inferred on first evaluation if omitted."""
        return self._callable.dtype

    @property
    def device(self) -> torch.device:
        """Device on which the callable is evaluated."""
        return self._callable.device

    @property
    def evaluation_stats(self) -> EvaluationStats:
        """Cumulative callable evaluation counters."""
        return self._callable.evaluation_stats

    def reset_evaluation_stats(self) -> None:
        """Resets the callable's cumulative evaluation counters."""
        self._callable.reset_evaluation_stats()

    def _evaluate_function(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluates the original coordinate callable."""
        return self.function(coordinates)

    def _evaluate_scalar(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Accepts a scalar output with an optional singleton trailing axis."""
        result = self._evaluate_function(coordinates)
        if isinstance(result, torch.Tensor) and \
                result.shape == (coordinates.shape[0], 1):
            result = result.squeeze(-1)
        return result

    def _evaluate_coordinates(self,
                              coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluates scalar values while preserving the input batch shape."""
        batch_shape = coordinates.shape[:-1]
        configurations = ConfigurationBatch(
            coordinates.reshape(-1, coordinates.shape[-1]), kind='features')
        values = self._callable.evaluate(configurations)
        if not (values.is_floating_point() or values.is_complex()):
            raise TypeError('`function` output should be floating or complex')
        return values.reshape(batch_shape)


class QuanticsVectorSource(_QuanticsSource):  # MARK: QuanticsVectorSource
    """
    Scalar coordinate callable discretized by a Quantics layout and map.

    :class:`~tensorkrowch.formats.CoordinateMap` selects grid points in the
    domain; :class:`~tensorkrowch.formats.QuantizedLayout` represents their
    indices as digit sites. Algorithms query these digits through
    :meth:`evaluate`. The original continuous callable remains available as
    ``function``. This source contains no cores and does not quantize existing
    tensors or tensor sources.

    Parameters
    ----------
    function : callable
        Scalar function receiving domain coordinates with shape
        ``(batch, n_coordinates)`` and returning shape ``(batch,)`` or ``(batch, 1)``.
    n_coordinates : int
        Number of coordinates in the domain.
    base, level : int or sequence of int, optional
        Digit base and number of digits per coordinate. Both are required
        without an explicit ``layout`` and ``coordinate_map``.
    domain : sequence or torch.Tensor, optional
        Coordinate intervals for an affine map with ``grid_offset="left"``.
        Requires ``base`` and ``level``; cannot accompany ``grid_coordinates``.
    grid_coordinates : torch.Tensor or sequence of torch.Tensor, optional
        Explicit grid points in the domain, one sequence per coordinate.
        Their sizes must equal ``base ** level``.
    layout : QuantizedLayout, optional
        Explicit digit layout. Supply together with ``coordinate_map`` and
        without the shorthand arguments above.
    coordinate_map : CoordinateMap, optional
        Explicit map whose ``grid_size`` matches ``layout.grid_size``.
    dtype : torch.dtype, optional
        Expected callable output dtype, inferred when omitted.
    device : str or torch.device, optional
        Callable evaluation device. The default is ``"cpu"``.
    batch_size : int, optional
        Maximum number of configurations per callable invocation.

    Examples
    --------
    >>> source = tk.decompositions.QuanticsVectorSource(
    ...     lambda coordinates: coordinates[:, 0].square(), 1,
    ...     base=2, level=2, domain=torch.tensor([0., 1.]))
    >>> source.evaluate_indices(torch.tensor([[0], [2], [3]]))
    tensor([0.0000, 0.2500, 0.5625])
    >>> source.coordinates_to_digits(torch.tensor([[0.], [0.5]]))
    tensor([[0, 0],
            [1, 0]])
    >>> source.to_dense_digits()
    tensor([[0.0000, 0.0625],
            [0.2500, 0.5625]])
    """

    def __init__(self,
                 function: Callable,
                 n_coordinates: int,
                 base: Optional[Union[int, Sequence[int]]] = None,
                 level: Optional[Union[int, Sequence[int]]] = None,
                 domain: Domain = None,
                 grid_coordinates: Optional[Union[
                     torch.Tensor, Sequence[torch.Tensor]]] = None,
                 *,
                 layout: Optional[QuantizedLayout] = None,
                 coordinate_map: Optional[CoordinateMap] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Union[str, torch.device] = 'cpu',
                 batch_size: Optional[int] = None) -> None:
        self.layout, self.coordinate_map = _resolve_quantization(
            n_coordinates, base=base, level=level, domain=domain,
            grid_coordinates=grid_coordinates, layout=layout,
            coordinate_map=coordinate_map)
        self.n_coordinates = n_coordinates
        super().__init__(function, self.layout.grid_size, dtype, device,
                         batch_size)

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Dimension of each digit site in layout order."""
        return self.layout.in_dim

    @property
    def out_shape(self) -> Tuple[int, ...]:
        """Scalar output shape after the configuration batch."""
        return ()

    def coordinates_to_digits(self,
                              coordinates: torch.Tensor) -> torch.Tensor:
        """
        Encodes domain coordinates as digit configurations.

        Parameters
        ----------
        coordinates : torch.Tensor
            Domain coordinates with shape ``(*data_batch, n_coordinates)``.
            The map's grid and out-of-domain policy select the grid indices.

        Returns
        -------
        torch.Tensor
            Integer digits with shape ``(*data_batch, layout.n_sites)``.
        """
        indices = self.coordinate_map.to_indices(coordinates)
        return self.layout.encode_indices(indices).to(self.device)

    def digits_to_coordinates(self, digits: torch.Tensor) -> torch.Tensor:
        """
        Decodes digits into the corresponding grid points in the domain.

        Parameters
        ----------
        digits : torch.Tensor
            Integer digits with shape ``(*data_batch, layout.n_sites)``.

        Returns
        -------
        torch.Tensor
            Domain coordinates with shape ``(*data_batch, n_coordinates)``.
        """
        indices = self.layout.decode_digits(digits)
        return self.coordinate_map.from_indices(indices.to(self.device))

    def evaluate_digits(self, digits: torch.Tensor) -> torch.Tensor:
        """
        Evaluates digit configurations in the layout's site order.

        Parameters
        ----------
        digits : torch.Tensor
            Integer digits with shape ``(*data_batch, layout.n_sites)``.

        Returns
        -------
        torch.Tensor
            Scalar function values with shape ``(*data_batch,)``.
        """
        return self._evaluate_coordinates(self.digits_to_coordinates(digits))

    def evaluate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Evaluates grid indices, with one index per original coordinate.

        Parameters
        ----------
        indices : torch.Tensor
            Integer grid indices with shape ``(*data_batch, n_coordinates)``.

        Returns
        -------
        torch.Tensor
            Scalar function values with shape ``(*data_batch,)``.
        """
        return self.evaluate_digits(self.layout.encode_indices(indices))

    def evaluate_coordinates(self,
                             coordinates: torch.Tensor) -> torch.Tensor:
        """
        Evaluates domain coordinates after discretizing them onto the grid.

        Parameters
        ----------
        coordinates : torch.Tensor
            Domain coordinates with shape ``(*data_batch, n_coordinates)``.
            Evaluation uses the selected grid points without interpolating
            function values, as in the Quantics formats.

        Returns
        -------
        torch.Tensor
            Scalar values with shape ``(*data_batch,)``. Use
            ``function(coordinates)`` to evaluate the original continuous
            function directly.
        """
        return self.evaluate_digits(self.coordinates_to_digits(coordinates))

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Evaluates the discrete digit configurations requested by an algorithm.

        Parameters
        ----------
        configurations : ConfigurationBatch
            Packed integer digits with ``kind="indices"`` and one column per
            digit site. Coordinate indices belong in :meth:`evaluate_indices`.

        Returns
        -------
        torch.Tensor
            Scalar values with shape ``(batch,)`` in configuration order.
        """
        if not isinstance(configurations, ConfigurationBatch):
            raise TypeError('`configurations` should be ConfigurationBatch type')
        if configurations.kind != 'indices':
            raise ValueError('Quantics sources require digit index configurations')
        return self.evaluate_digits(configurations.as_tensor())

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluates configurations while varying one digit site.

        Parameters
        ----------
        configurations : ConfigurationBatch
            Base digit configurations with ``kind="indices"``.
        site : int
            Digit site to vary. All other sites retain their original values.
        values : torch.Tensor, optional
            One-dimensional candidate digits. Defaults to every index in
            ``range(in_dim[site])``.

        Returns
        -------
        torch.Tensor
            Function values with shape ``(batch, n_values)``.
        """
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if site < 0 or site >= len(self.in_dim):
            raise ValueError('`site` should identify a digit site')
        if values is None:
            values = torch.arange(self.in_dim[site], device=self.device)
        expanded, n_values = _fiber_configurations(configurations, site, values)
        return self.evaluate(expanded).reshape(configurations.batch_size,
                                               n_values)

    def to_dense_grid(self) -> torch.Tensor:
        """
        Evaluates the full grid with one axis per original coordinate.

        Returns
        -------
        torch.Tensor
            Dense values with shape ``(*layout.grid_size,)``. Unlike
            :meth:`to_dense_digits`, the axes follow coordinate order.
        """
        indices = _unravel_indices(
            torch.arange(prod(self.layout.grid_size), device=self.device),
            self.layout.grid_size)
        return self.evaluate_indices(indices).reshape(self.layout.grid_size)

    def to_dense_digits(self) -> torch.Tensor:
        """
        Evaluates the full tensor with one axis per digit site.

        Returns
        -------
        torch.Tensor
            Dense values with shape ``(*in_dim,)`` in layout order, ready for
            TT/TR-SVD. Both dense materializations allocate the complete grid;
            ``batch_size`` only bounds individual callable evaluations.
        """
        digits = _unravel_indices(
            torch.arange(prod(self.in_dim), device=self.device), self.in_dim)
        return self.evaluate_digits(digits).reshape(self.in_dim)


class QuanticsMatrixSource(_QuanticsSource):  # MARK: QuanticsMatrixSource
    """
    Scalar kernel discretized on independent input and output Quantics grids.

    Input and output :class:`~tensorkrowch.formats.QuantizedLayout` objects
    pair their digit sites to form matrix axes. Each side uses its own
    :class:`~tensorkrowch.formats.CoordinateMap`. The original kernel remains
    available as ``function(in_coordinates, out_coordinates)``.

    Parameters
    ----------
    function : callable
        Kernel receiving two domain-coordinate tensors with matching leading
        batches and returning one scalar per configuration.
    in_n_coordinates, out_n_coordinates : int
        Number of original coordinates on each side.
    in_base, out_base, in_level, out_level : int or sequence of int, optional
        Digit bases and levels. Each side follows the same construction rules
        as :class:`QuanticsVectorSource`.
    in_domain, out_domain : sequence or torch.Tensor, optional
        Intervals for the default affine maps with ``grid_offset="left"``.
    in_grid_coordinates, out_grid_coordinates : tensor or sequence, optional
        Explicit grid points in each original domain.
    in_layout, out_layout : QuantizedLayout, optional
        Explicit layouts. They must contain equal numbers of digit sites.
    in_coordinate_map, out_coordinate_map : CoordinateMap, optional
        Explicit maps matching the corresponding layouts. Supply each map
        together with its layout, without shorthand arguments for that side.
    dtype : torch.dtype, optional
        Expected scalar output dtype, inferred when omitted.
    device : str or torch.device, optional
        Callable evaluation device. The default is ``"cpu"``.
    batch_size : int, optional
        Maximum number of configurations per callable invocation.

    Examples
    --------
    >>> source = tk.decompositions.QuanticsMatrixSource(
    ...     lambda inputs, outputs: inputs[:, 0] + 2 * outputs[:, 0], 1, 1,
    ...     in_base=2, out_base=2, in_level=1, out_level=1,
    ...     in_domain=torch.tensor([0., 1.]),
    ...     out_domain=torch.tensor([0., 1.]))
    >>> source.to_dense_grid()
    tensor([[0.0000, 1.0000],
            [0.5000, 1.5000]])
    """

    def __init__(self,
                 function: Callable,
                 in_n_coordinates: int,
                 out_n_coordinates: int,
                 *,
                 in_base: Optional[Union[int, Sequence[int]]] = None,
                 out_base: Optional[Union[int, Sequence[int]]] = None,
                 in_level: Optional[Union[int, Sequence[int]]] = None,
                 out_level: Optional[Union[int, Sequence[int]]] = None,
                 in_domain: Domain = None,
                 out_domain: Domain = None,
                 in_grid_coordinates: Optional[Union[
                     torch.Tensor, Sequence[torch.Tensor]]] = None,
                 out_grid_coordinates: Optional[Union[
                     torch.Tensor, Sequence[torch.Tensor]]] = None,
                 in_layout: Optional[QuantizedLayout] = None,
                 out_layout: Optional[QuantizedLayout] = None,
                 in_coordinate_map: Optional[CoordinateMap] = None,
                 out_coordinate_map: Optional[CoordinateMap] = None,
                 dtype: Optional[torch.dtype] = None,
                 device: Union[str, torch.device] = 'cpu',
                 batch_size: Optional[int] = None) -> None:
        self.in_layout, self.in_coordinate_map = _resolve_quantization(
            in_n_coordinates, base=in_base, level=in_level, domain=in_domain,
            grid_coordinates=in_grid_coordinates, layout=in_layout,
            coordinate_map=in_coordinate_map)
        self.out_layout, self.out_coordinate_map = _resolve_quantization(
            out_n_coordinates, base=out_base, level=out_level, domain=out_domain,
            grid_coordinates=out_grid_coordinates, layout=out_layout,
            coordinate_map=out_coordinate_map)
        if self.in_layout.n_sites != self.out_layout.n_sites:
            raise ValueError(
                '`in_layout` and `out_layout` should have equal `n_sites`')

        self.in_n_coordinates = in_n_coordinates
        self.out_n_coordinates = out_n_coordinates
        super().__init__(
            function, self.in_layout.grid_size + self.out_layout.grid_size,
            dtype, device, batch_size)

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension of each paired digit site."""
        return self.in_layout.in_dim

    @property
    def out_dim(self) -> Tuple[int, ...]:
        """Output dimension of each paired digit site."""
        return self.out_layout.in_dim

    def _evaluate_function(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Evaluates the kernel on paired input and output coordinates."""
        return self.function(coordinates[:, :self.in_n_coordinates],
                             coordinates[:, self.in_n_coordinates:])

    def coordinates_to_digits(self,
                              in_coordinates: torch.Tensor,
                              out_coordinates: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes paired domain coordinates into input and output digits.

        Parameters
        ----------
        in_coordinates, out_coordinates : torch.Tensor
            Domain coordinates with matching leading batches and respectively
            ``in_n_coordinates`` and ``out_n_coordinates`` trailing entries.

        Returns
        -------
        tuple of torch.Tensor
            Input and output digits in the order of their respective layouts.
        """
        in_indices = self.in_coordinate_map.to_indices(in_coordinates)
        out_indices = self.out_coordinate_map.to_indices(out_coordinates)
        return (self.in_layout.encode_indices(in_indices).to(self.device),
                self.out_layout.encode_indices(out_indices).to(self.device))

    def digits_to_coordinates(self,
                              in_digits: torch.Tensor,
                              out_digits: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes paired digits into grid points in the two domains.

        Parameters
        ----------
        in_digits, out_digits : torch.Tensor
            Integer digits with matching leading batches and one trailing
            entry per input/output digit site.

        Returns
        -------
        tuple of torch.Tensor
            Input and output domain-coordinate configurations.
        """
        in_indices = self.in_layout.decode_digits(in_digits)
        out_indices = self.out_layout.decode_digits(out_digits)
        return (self.in_coordinate_map.from_indices(in_indices.to(self.device)),
                self.out_coordinate_map.from_indices(out_indices.to(self.device)))

    def evaluate_digits(self,
                        in_digits: torch.Tensor,
                        out_digits: torch.Tensor) -> torch.Tensor:
        """
        Evaluates matrix entries using paired digit configurations.

        Parameters
        ----------
        in_digits, out_digits : torch.Tensor
            Integer digit configurations with matching leading batches.

        Returns
        -------
        torch.Tensor
            Scalar kernel values with the shared leading batch shape.
        """
        inputs, outputs = self.digits_to_coordinates(in_digits, out_digits)
        if inputs.shape[:-1] != outputs.shape[:-1]:
            raise ValueError(
                '`in_digits` and `out_digits` should have matching batches')
        return self._evaluate_coordinates(torch.cat((inputs, outputs), dim=-1))

    def evaluate_indices(self,
                         in_indices: torch.Tensor,
                         out_indices: torch.Tensor) -> torch.Tensor:
        """
        Evaluates paired grid indices in original coordinate order.

        Parameters
        ----------
        in_indices, out_indices : torch.Tensor
            Integer indices with matching leading batches and one trailing
            entry per original input/output coordinate.

        Returns
        -------
        torch.Tensor
            Scalar kernel values with the shared leading batch shape.
        """
        return self.evaluate_digits(self.in_layout.encode_indices(in_indices),
                                    self.out_layout.encode_indices(out_indices))

    def evaluate_coordinates(self,
                             in_coordinates: torch.Tensor,
                             out_coordinates: torch.Tensor) -> torch.Tensor:
        """
        Evaluates domain coordinates after discretizing both sides.

        Parameters
        ----------
        in_coordinates, out_coordinates : torch.Tensor
            Domain coordinates with matching leading batches. The maps select
            grid points without interpolating kernel values.

        Returns
        -------
        torch.Tensor
            Scalar kernel values with the shared leading batch shape.
        """
        return self.evaluate_digits(*self.coordinates_to_digits(in_coordinates,
                                                               out_coordinates))

    def to_dense_grid(self) -> torch.Tensor:
        """
        Evaluates the full input/output grid in original coordinate order.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*in_layout.grid_size, *out_layout.grid_size)``.
            Unlike :meth:`to_dense_digits`, input axes precede output axes.
        """
        shape = self.in_layout.grid_size + self.out_layout.grid_size
        indices = _unravel_indices(
            torch.arange(prod(shape), device=self.device), shape)
        return self.evaluate_indices(
            indices[:, :self.in_n_coordinates],
            indices[:, self.in_n_coordinates:]).reshape(shape)

    def to_dense_digits(self) -> torch.Tensor:
        """
        Evaluates the full matrix tensor with paired input/output digit axes.

        Returns
        -------
        torch.Tensor
            Values with interleaved axes ``(in_1, out_1, ..., in_n, out_n)`` in
            each layout's site order, ready for TTM/TRM-SVD. Allocates the full
            grid; ``batch_size`` only bounds individual callable evaluations.
        """
        shape = tuple(dim for pair in zip(self.in_dim, self.out_dim)
                      for dim in pair)
        digits = _unravel_indices(
            torch.arange(prod(shape), device=self.device), shape)
        return self.evaluate_digits(digits[:, ::2], digits[:, 1::2]).reshape(shape)


__all__ = [
    'QuanticsVectorSource',
    'QuanticsMatrixSource',
]
