"""Tensor train backed tensor sources."""

from typing import (Optional, Protocol, Sequence, Tuple, Union)

import torch

from tensorkrowch.decompositions.results import (TTDecomposition,
                                                 TTMDecomposition)
from tensorkrowch.decompositions.sources.base import (ConfigurationBatch,
                                                      _discrete_indices,
                                                      _SourceEvaluationTracker)


class _MPSAdapter(Protocol):
    """Minimal model surface required to extract open-boundary TT cores."""

    @property
    def boundary(self) -> str:
        """Boundary-condition identifier."""

    @property
    def tensors(self) -> Sequence[torch.Tensor]:
        """Raw compact MPS tensors."""


class TTTensorSource(_SourceEvaluationTracker):
    """Scalar tensor source represented directly by TT cores.

    The source contracts raw PyTorch cores without constructing a
    TensorKrowch graph. It accepts a
    :class:`~tensorkrowch.decompositions.TTDecomposition`, an open-boundary
    :class:`~tensorkrowch.models.MPS` adapter or a core sequence with the same
    conventions. Model adapters only extract their tensors. Keeping the
    specialized contractions here avoids routing repeated ALS/sketching
    evaluations through a TensorKrowch graph.

    Parameters
    ----------
    tensor : TTDecomposition, MPS or sequence of torch.Tensor
        Lightweight TT result, open-boundary model or raw TT cores.
    """

    def __init__(
            self,
            tensor: Union[TTDecomposition, Sequence[torch.Tensor],
                          _MPSAdapter]) -> None:
        self._initialize_evaluation_stats()
        if hasattr(tensor, 'boundary') and hasattr(type(tensor), 'tensors'):
            if tensor.boundary != 'obc':
                raise ValueError(
                    'Only open-boundary MPS models can define TT sources')
            tensor = tensor.tensors
        if isinstance(tensor, TTDecomposition):
            if tensor.n_batches:
                raise ValueError('Batched TT sources are not supported')
            cores = list(tensor.cores)
        else:
            if isinstance(tensor, torch.Tensor):
                raise TypeError(
                    '`tensor` should be TTDecomposition or a core sequence')
            try:
                cores = list(tensor)
            except TypeError as exc:
                raise TypeError(
                    '`tensor` should be TTDecomposition or a core sequence') \
                    from exc
        if not cores:
            raise ValueError('`tensor` should contain at least one core')
        if not all(isinstance(core, torch.Tensor) for core in cores):
            raise TypeError('TT cores should be torch.Tensor objects')

        if len(cores) == 1:
            if cores[0].ndim == 1:
                standard_cores = [cores[0].reshape(1, -1, 1)]
            elif (cores[0].ndim == 3) and \
                    (cores[0].shape[0] == cores[0].shape[-1] == 1):
                standard_cores = [cores[0]]
            else:
                raise ValueError(
                    'A one-site TT core should contain only its input dimension')
        else:
            standard_cores = []
            first = cores[0]
            if first.ndim == 2:
                first = first.unsqueeze(0)
            if (first.ndim != 3) or (first.shape[0] != 1):
                raise ValueError(
                    'The first TT core should have a unit left boundary rank')
            standard_cores.append(first)

            for core in cores[1:-1]:
                if core.ndim != 3:
                    raise ValueError(
                        'Interior TT cores should have left, input and right '
                        'rank dimensions')
                standard_cores.append(core)

            last = cores[-1]
            if last.ndim == 2:
                last = last.unsqueeze(-1)
            if (last.ndim != 3) or (last.shape[-1] != 1):
                raise ValueError(
                    'The last TT core should have a unit right boundary rank')
            standard_cores.append(last)

        device = standard_cores[0].device
        dtype = standard_cores[0].dtype
        for site, core in enumerate(standard_cores):
            if core.device != device:
                raise ValueError('All TT cores should be on the same device')
            if core.dtype != dtype:
                raise ValueError('All TT cores should have the same dtype')
            if site and (core.shape[0] != standard_cores[site - 1].shape[-1]):
                raise ValueError('Adjacent TT ranks should match')

        self.cores = standard_cores
        self._input_dim = tuple(core.shape[1] for core in standard_cores)

    @staticmethod
    def _standard_ttm_cores(
            sketch: TTMDecomposition) -> Tuple[torch.Tensor, ...]:
        """Returns TTM cores with left, input, output and right axes."""
        if sketch.n_batches:
            raise ValueError('Batched TTM sketches are not supported')
        if len(sketch.cores) == 1:
            return (sketch.cores[0].unsqueeze(0).unsqueeze(-1),)
        cores = [sketch.cores[0].permute(0, 2, 1).unsqueeze(0)]
        cores.extend(core.permute(0, 1, 3, 2)
                     for core in sketch.cores[1:-1])
        cores.append(sketch.cores[-1].unsqueeze(-1))
        return tuple(cores)

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Input dimension at every TT site."""
        return self._input_dim

    @property
    def output_shape(self) -> Tuple[int, ...]:
        """Empty shape because this source is scalar."""
        return ()

    @property
    def dtype(self) -> torch.dtype:
        """Dtype shared by the TT cores."""
        return self.cores[0].dtype

    @property
    def device(self) -> torch.device:
        """Device shared by the TT cores."""
        return self.cores[0].device

    def _selected_matrices(self, indices: torch.Tensor):
        """Selects one TT matrix per configuration and site."""
        return [
            core[:, indices[:, site], :].permute(1, 0, 2)
            for site, core in enumerate(self.cores)
        ]

    def left_environments(self,
                          configurations: torch.Tensor) -> torch.Tensor:
        """Contracts exact prefixes into their outgoing TT environments.

        ``configurations`` has shape ``(rows, prefix_sites)``. An empty
        prefix returns one unit boundary row. This kernel records no source
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

    def right_environments(self,
                           configurations: torch.Tensor) -> torch.Tensor:
        """Contracts exact suffixes into their incoming TT environments.

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

    def _partial_indices(self,
                         configurations: torch.Tensor,
                         *,
                         start: int,
                         name: str) -> torch.Tensor:
        """Validates a consecutive partial discrete configuration batch."""
        if not isinstance(configurations, torch.Tensor):
            raise TypeError(f'`{name}` should be torch.Tensor type')
        if configurations.ndim != 2 or configurations.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64):
            raise TypeError(
                f'`{name}` should be a two-dimensional integer tensor')
        stop = start + configurations.shape[1]
        if start < 0 or stop > len(self.cores):
            raise ValueError(f'`{name}` contains too many sites')
        indices = configurations.to(device=self.device, dtype=torch.long)
        for offset, dimension in enumerate(self.input_dim[start:stop]):
            values = indices[:, offset]
            if torch.any(values < 0) or torch.any(values >= dimension):
                raise ValueError(
                    f'`{name}` is out of bounds at partial site {offset}')
        return indices

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
                     factor: Optional[Sequence[torch.Tensor]] = None
                     ) -> torch.Tensor:
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
                    factors, self.input_dim)):
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

    def tt_sketch_phis(
            self,
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
        """Evaluates discrete configurations by batched TT contraction."""
        indices = _discrete_indices(
            configurations, self.input_dim, self.device)
        matrices = self._selected_matrices(indices)
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        result = result[:, 0, 0]
        self._record_evaluation(points=indices.shape[0])
        return result

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Contracts both TT environments and leaves one input site open."""
        if not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= len(self.input_dim)):
            raise ValueError('`site` should identify an input site')
        indices = _discrete_indices(
            configurations, self.input_dim, self.device)
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
                points=indices.shape[0] * self.input_dim[site])
            return result
        if not isinstance(values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        if (values.ndim != 1) or (values.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64)):
            raise TypeError('`values` should be a one-dimensional integer tensor')
        values = values.to(device=self.device, dtype=torch.long)
        if torch.any(values < 0) or torch.any(values >= self.input_dim[site]):
            raise ValueError('`values` are out of bounds for the selected site')
        result = result.index_select(1, values)
        self._record_evaluation(points=indices.shape[0] * values.shape[0])
        return result

    def contract_sketch(
            self,
            sketch: Union['TTTensorSource', TTDecomposition, TTMDecomposition,
                          Sequence[torch.Tensor]],
            conjugate_sketch: bool = True
            ) -> Union[torch.Tensor, TTDecomposition]:
        """Contracts the source input indices with a scalar TT or TTM sketch.

        A scalar TT sketch returns its inner product with the source. A TTM
        sketch returns a lightweight :class:`TTDecomposition` over the TTM
        output indices, keeping the contraction structured. By default the
        sketch is conjugated, as in a complex linear range projection.
        """
        if not isinstance(conjugate_sketch, bool):
            raise TypeError('`conjugate_sketch` should be bool type')

        if isinstance(sketch, TTMDecomposition):
            if sketch.input_dim != self.input_dim:
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

        sketch_source = sketch if isinstance(sketch, TTTensorSource) \
            else TTTensorSource(sketch)
        if sketch_source.input_dim != self.input_dim:
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


__all__ = ['TTTensorSource']
