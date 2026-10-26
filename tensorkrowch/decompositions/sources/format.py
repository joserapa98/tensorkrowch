"""
This script contains:

    Public classes:
        * FormatSource
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.utils import _INTEGER_DTYPES

from tensorkrowch.models.mps import MPS
from tensorkrowch.formats import TT, TTM
from tensorkrowch.formats.formats1d import TensorFormat1D
from tensorkrowch.decompositions.results import TTDecomposition

from tensorkrowch.decompositions.sources.base import (BaseSource,
                                                      ConfigurationBatch,
                                                      _discrete_indices)


class FormatSource(BaseSource):  # MARK: FormatSource
    """
    Finite tensor source represented directly by a lightweight format.

    Construct the source from a :class:`~tensorkrowch.formats.TT`,
    :class:`~tensorkrowch.formats.TR` or their matrix versions,
    :class:`~tensorkrowch.formats.TTM` and :class:`~tensorkrowch.formats.TRM`.
    Evaluation and dense contraction use the backing format without first
    materializing its complete tensor. For vectors, configuration columns
    have shape ``(batch, len(in_sites))`` in the format's ``in_sites`` order;
    the remaining axes form ``out_shape``.

    Matrix grouping is inferred from the format, so no separate ``matrix``
    constructor is needed. Matrix evaluations receive indices for every input
    site and return all output axes, independently of the format's evaluation
    defaults, with ``out_shape == out_dim`` and ``len(in_sites) == n_sites``.

    An :class:`~tensorkrowch.models.MPS` is converted to TT or TR according to
    its boundaries. A sequence of cores instead constructs an open TT.
    Quantics formats require an explicit :class:`CallableQuanticsSource`.
    Specialized sketch kernels apply only to unbatched scalar open TT formats.

    Parameters
    ----------
    tensor : TensorFormat1D, MPS or sequence of torch.Tensor
        Lightweight vector/matrix format, MPS or raw TT cores. MPS boundaries
        determine TT versus TR.

    Examples
    --------
    >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
    >>> source = tk.decompositions.FormatSource(format)
    >>> source.evaluate(tk.decompositions.ConfigurationBatch(
    ...     torch.tensor([[0, 0], [0, 1]])))
    tensor([1., 0.])

    A matrix format already provides the input/output grouping:

    >>> format = tk.formats.TTM([torch.eye(2)])
    >>> source = tk.decompositions.FormatSource(format)
    >>> source.evaluate(tk.decompositions.ConfigurationBatch(
    ...     torch.tensor([[1]])))
    tensor([[0., 1.]])
    """

    def __init__(self,
                 tensor: Union[TensorFormat1D, Sequence[torch.Tensor], MPS]) -> None:
        if isinstance(tensor, MPS):
            tensor = tensor.to_tr() if tensor.boundary == 'pbc' else tensor.to_tt()
        if not isinstance(tensor, TensorFormat1D):
            if isinstance(tensor, (str, bytes)) or not isinstance(tensor, Sequence):
                raise TypeError('`tensor` should be a format, MPS or core sequence')
            cores = tuple(tensor)
            if not cores or not all(isinstance(core, torch.Tensor) for core in cores):
                raise TypeError('`tensor` should contain torch.Tensor cores')
            if cores and cores[0].ndim == 3 and cores[-1].ndim == 3:
                cores = (cores[0].squeeze(0), *cores[1:-1], cores[-1].squeeze(-1)) \
                    if len(cores) > 1 else (cores[0].squeeze(0).squeeze(-1),)
            tensor = TT(cores)
        if tensor._quantized:
            raise TypeError('Quantics formats require an explicit Quantics source')
        self.format = tensor
        self.cores = list(tensor._effective_cores())
        self._in_dim = tensor.in_dim
        super().__init__(tensor.n_sites, n_batches=tensor.n_batches,
                         in_sites=(None if tensor.out_dim is not None
                                   else tensor.in_sites))
        self._structured = isinstance(tensor, TT) and not tensor._cyclic and \
            not tensor.n_batches and tensor.in_sites == tuple(range(tensor.n_sites))
        if tensor.out_dim is not None:
            self._matrix_dims = tensor.in_dim, tensor.out_dim

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Dimensions of all vector variables or matrix input axes."""
        return self._in_dim

    @property
    def dtype(self) -> torch.dtype:
        """Dtype shared by the cores."""
        return self.cores[0].dtype

    @property
    def device(self) -> torch.device:
        """Device shared by the cores."""
        return self.cores[0].device

    @staticmethod
    def _standard_ttm_cores(sketch: TTM) -> Tuple[torch.Tensor, ...]:
        """Returns TTM cores with left, input, output and right axes."""
        if sketch.n_batches:
            raise ValueError('Batched TTM sketches are not supported')
        return tuple(core.permute(0, 1, 3, 2)
                     for core in sketch._operator_cores())

    def _selected_matrices(self, indices: torch.Tensor) -> List[torch.Tensor]:
        """Selects one TT matrix per configuration and site."""
        return [
            core[:, indices[:, site], :].permute(1, 0, 2)
            for site, core in enumerate(self.cores)
        ]

    def _partial_indices(self,
                         configurations: torch.Tensor,
                         *,
                         start: int,
                         name: str) -> torch.Tensor:
        """Validates a consecutive partial discrete configuration batch."""
        if not isinstance(configurations, torch.Tensor):
            raise TypeError(f'`{name}` should be torch.Tensor type')
        if configurations.ndim != 2 or configurations.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                f'`{name}` should be a two-dimensional integer tensor')
        stop = start + configurations.shape[1]
        if start < 0 or stop > len(self.cores):
            raise ValueError(f'`{name}` contains too many sites')
        indices = configurations.to(device=self.device, dtype=torch.long)
        for offset, dimension in enumerate(self._in_dim[start:stop]):
            values = indices[:, offset]
            if torch.any(values < 0) or torch.any(values >= dimension):
                raise ValueError(
                    f'`{name}` is out of bounds at partial site {offset}')
        return indices

    def left_environments(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Contracts exact prefixes into their outgoing TT environments.

        ``configurations`` has shape ``(rows, prefix_sites)``. An empty prefix
        returns one unit boundary row. This kernel records no source
        evaluations because it contracts the stored representation directly.
        """
        indices = self._partial_indices(
            configurations, start=0, name='configurations')
        environment = self.cores[0].new_ones((indices.shape[0], 1))
        for site in range(indices.shape[1]):
            matrices = self.cores[site][
                :, indices[:, site], :].permute(1, 0, 2)
            environment = torch.einsum(
                'ba,bar->br', environment, matrices)
        return environment

    def right_environments(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Contracts exact suffixes into their incoming TT environments.

        ``configurations`` has shape ``(rows, suffix_sites)`` and is aligned
        with the final source sites. An empty suffix returns the unit boundary.
        """
        start = len(self.cores) - configurations.shape[1] \
            if isinstance(configurations, torch.Tensor) and \
            configurations.ndim == 2 else 0
        indices = self._partial_indices(
            configurations, start=start, name='configurations')
        environment = self.cores[0].new_ones((indices.shape[0], 1))
        for offset in range(indices.shape[1] - 1, -1, -1):
            site = start + offset
            matrices = self.cores[site][
                :, indices[:, offset], :].permute(1, 0, 2)
            environment = torch.einsum(
                'bar,br->ba', matrices, environment)
        return environment

    def local_phi(self,
                  site: int,
                  left: torch.Tensor,
                  right: torch.Tensor,
                  factor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Contracts one source core between external sketch environments."""
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if site < 0 or site >= len(self.cores):
            raise ValueError('`site` should identify one source site')
        if not isinstance(left, torch.Tensor) or left.ndim != 2:
            raise TypeError('`left` should be a matrix')
        if not isinstance(right, torch.Tensor) or right.ndim != 2:
            raise TypeError('`right` should be a matrix')
        core = self.cores[site]
        if left.shape[1] != core.shape[0]:
            raise ValueError('`left` does not match the source left rank')
        if right.shape[1] != core.shape[-1]:
            raise ValueError('`right` does not match the source right rank')
        dtype = torch.promote_types(
            core.dtype, torch.promote_types(left.dtype, right.dtype))
        core = core.to(dtype=dtype)
        if factor is not None:
            if not isinstance(factor, torch.Tensor) or \
                    factor.shape != (core.shape[1],):
                raise ValueError('`factor` should match the site input dimension')
            core = core * factor.to(device=self.device, dtype=dtype)[None, :, None]
        return torch.einsum(
            'la,aib,rb->lir',
            left.to(device=self.device, dtype=dtype),
            core,
            right.to(device=self.device, dtype=dtype))

    def marginal_phi(self,
                     site: int,
                     order: int = 1,
                     factor: Optional[Sequence[torch.Tensor]] = None) -> torch.Tensor:
        """Contracts a local Markov marginal without densifying the source."""
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError('`order` should be int type')
        if order < 1:
            raise ValueError('`order` should be positive')
        if factor is None:
            factors = [core.new_ones(core.shape[1]) for core in self.cores]
        else:
            factors = list(factor)
            if len(factors) != len(self.cores):
                raise ValueError('`factor` should contain one vector per site')
            for position, (value, dimension) in enumerate(zip(
                    factors, self._in_dim)):
                if not isinstance(value, torch.Tensor) or \
                        value.shape != (dimension,):
                    raise ValueError(
                        f'`factor` at site {position} has an invalid shape')
                factors[position] = value.to(
                    device=self.device, dtype=self.dtype)

        left_start = max(0, site - order)
        left = self.cores[0].new_ones((1, 1))
        for position in range(left_start):
            matrix = torch.einsum(
                'aib,i->ab', self.cores[position], factors[position])
            left = left @ matrix
        for position in range(left_start, site):
            core = self.cores[position] * \
                factors[position][None, :, None]
            left = torch.einsum('la,aib->lib', left, core).reshape(
                -1, core.shape[-1])

        right_stop = min(len(self.cores), site + 1 + order)
        right = self.cores[0].new_ones((1, 1))
        for position in range(len(self.cores) - 1, right_stop - 1, -1):
            matrix = torch.einsum(
                'aib,i->ab', self.cores[position], factors[position])
            right = torch.einsum('ab,rb->ra', matrix, right)
        for position in range(right_stop - 1, site, -1):
            core = self.cores[position] * \
                factors[position][None, :, None]
            right = torch.einsum('aib,rb->ira', core, right).reshape(
                -1, core.shape[0])
        return self.local_phi(site, left, right, factor=factors[site])

    def tt_sketch_phis(self,
                       left_cores: Sequence[Sequence[torch.Tensor]],
                       right_cores: Sequence[Sequence[torch.Tensor]],
                       boundary_scale: float = 1.0) -> Tuple[torch.Tensor, ...]:
        """Contracts stacked TT sketch networks into every local Phi."""
        left_cores = tuple(tuple(stack) for stack in left_cores)
        right_cores = tuple(tuple(stack) for stack in right_cores)
        if not left_cores or len(left_cores) != len(right_cores):
            raise ValueError('Left and right sketches should have equal stacks')
        if any(len(stack) != len(self.cores)
               for stack in (*left_cores, *right_cores)):
            raise ValueError('Every sketch stack should contain one core per site')

        left = [self.cores[0].new_ones((1, 1))]
        stack_left = self.cores[0].new_ones(
            (len(left_cores), 1, 1))
        for site in range(len(self.cores) - 1):
            next_stack = []
            for stack, cores in enumerate(left_cores):
                environment = torch.einsum(
                    'ac,aib,cid->bd',
                    stack_left[stack],
                    self.cores[site],
                    cores[site].to(self.dtype))
                if site == 0:
                    environment = environment * boundary_scale
                next_stack.append(environment)
            stack_left = torch.stack(next_stack)
            left.append(stack_left.permute(0, 2, 1).reshape(
                -1, stack_left.shape[1]))

        right = [None] * len(self.cores)
        right[-1] = self.cores[0].new_ones((1, 1))
        stack_right = self.cores[0].new_ones(
            (len(right_cores), 1, 1))
        for site in range(len(self.cores) - 1, 0, -1):
            next_stack = []
            for stack, cores in enumerate(right_cores):
                environment = torch.einsum(
                    'aib,cid,bd->ac',
                    self.cores[site],
                    cores[site].to(self.dtype),
                    stack_right[stack])
                if site == len(self.cores) - 1:
                    environment = environment * boundary_scale
                next_stack.append(environment)
            stack_right = torch.stack(next_stack)
            right[site - 1] = stack_right.permute(0, 2, 1).reshape(
                -1, stack_right.shape[1])
        return tuple(self.local_phi(site, left[site], right[site])
                     for site in range(len(self.cores)))

    def evaluate(self, configurations: ConfigurationBatch) -> torch.Tensor:
        """
        Evaluates discrete configurations through the backing format.

        Parameters
        ----------
        configurations : ConfigurationBatch
            Integer indices with shape ``(batch, len(in_sites))`` and columns
            ordered by ``in_sites``. Matrix sources evaluate every paired site.

        Returns
        -------
        torch.Tensor
            Values with shape ``(batch, *core_batch, *out_shape)``. Matrix
            output axes remain open; vectors retain their selected inputs.
        """
        indices = _discrete_indices(
            configurations, self._in_sites_dims,
            self.device)
        if self.is_matrix:
            result = self.format.evaluate(indices, in_sites=range(self.format.n_sites),
                                          out_sites=())
        else:
            result = self.format.evaluate(indices, in_sites=self.in_sites)
        if self.n_batches:
            result = result.movedim(self.n_batches, 0)
        self._record_evaluation(points=indices.shape[0])
        return result

    def contract_dense(self) -> torch.Tensor:
        """Contracts the backing format without changing its stored cores."""
        return self.format.contract_dense()

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evaluates a fiber using TT environments when the format supports them.
        """
        if not self._structured:
            return super().fiber(configurations, site, values)
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= len(self._in_dim)):
            raise ValueError('`site` should identify an input site')
        indices = _discrete_indices(
            configurations, self._in_dim, self.device)
        matrices = self._selected_matrices(indices)

        left = self.cores[0].new_ones((indices.shape[0], 1))
        for matrix in matrices[:site]:
            left = torch.einsum('ba,bar->br', left, matrix)
        right = self.cores[0].new_ones((indices.shape[0], 1))
        for matrix in reversed(matrices[site + 1:]):
            right = torch.einsum('bar,br->ba', matrix, right)
        result = torch.einsum(
            'ba,apr,br->bp', left, self.cores[site], right)

        if values is None:
            self._record_evaluation(
                points=indices.shape[0] * self._in_dim[site])
            return result
        if not isinstance(values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        if (values.ndim != 1) or (values.dtype not in _INTEGER_DTYPES):
            raise TypeError('`values` should be a one-dimensional integer tensor')
        values = values.to(device=self.device, dtype=torch.long)
        if torch.any(values < 0) or torch.any(values >= self._in_dim[site]):
            raise ValueError('`values` are out of bounds for the selected site')
        result = result.index_select(1, values)
        self._record_evaluation(points=indices.shape[0] * values.shape[0])
        return result

    def contract_sketch(self,
                        sketch: Union['FormatSource', TT, TTM, Sequence[torch.Tensor]],
                        conjugate_sketch: bool = True) -> Union[torch.Tensor, TTDecomposition]:
        """
        Contracts the source input indices with a scalar TT or TTM ``sketch``.

        A scalar TT ``sketch`` returns its inner product with the source. A TTM
        ``sketch`` returns a lightweight :class:`TTDecomposition` over the TTM
        output indices, keeping the contraction structured. By default the
        ``sketch`` is conjugated, as in a complex linear range projection.
        """
        if not isinstance(conjugate_sketch, bool):
            raise TypeError('`conjugate_sketch` should be bool type')

        if isinstance(sketch, TTM):
            if sketch.in_dim != self._in_dim:
                raise ValueError(
                    'Source and sketch should have matching input dimensions')
            if sketch.device != self.device:
                raise ValueError('Source and sketch should share a device')
            sketch_cores = self._standard_ttm_cores(sketch)
            dtype = torch.promote_types(self.dtype, sketch.dtype)
            result_cores = []
            for source_core, sketch_core in zip(self.cores, sketch_cores):
                source_core = source_core.to(dtype=dtype)
                sketch_core = sketch_core.to(dtype=dtype)
                if conjugate_sketch:
                    sketch_core = sketch_core.conj()
                core = torch.einsum(
                    'aib,ciod->acobd', source_core, sketch_core)
                result_cores.append(core.reshape(
                    source_core.shape[0] * sketch_core.shape[0],
                    sketch_core.shape[2],
                    source_core.shape[2] * sketch_core.shape[3]))
            if len(result_cores) == 1:
                compact_cores = [result_cores[0].reshape(-1)]
            else:
                compact_cores = [result_cores[0].squeeze(0)]
                compact_cores.extend(result_cores[1:-1])
                compact_cores.append(result_cores[-1].squeeze(-1))
            self._record_evaluation(
                points=0, batches=0, unique_points=0)
            return TTDecomposition(compact_cores)

        sketch_source = sketch if isinstance(sketch, FormatSource) \
            else FormatSource(sketch)
        if sketch_source.in_dim != self._in_dim:
            raise ValueError(
                'Source and sketch should have matching input dimensions')
        if sketch_source.device != self.device:
            raise ValueError('Source and sketch should share a device')
        dtype = torch.promote_types(self.dtype, sketch_source.dtype)
        environment = torch.ones((1, 1), device=self.device, dtype=dtype)
        for source_core, sketch_core in zip(
                self.cores, sketch_source.cores):
            sketch_core = sketch_core.to(dtype=dtype)
            if conjugate_sketch:
                sketch_core = sketch_core.conj()
            environment = torch.einsum(
                'ac,aib,cid->bd',
                environment,
                source_core.to(dtype=dtype),
                sketch_core)
        self._record_evaluation(points=0, batches=0, unique_points=0)
        return environment.squeeze()


__all__ = [
    'FormatSource',
]
