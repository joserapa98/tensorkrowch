"""
This script contains:

    Public classes:
        * FormatSource
"""

from typing import Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.utils import _INTEGER_DTYPES

from tensorkrowch.models.mps import MPS
from tensorkrowch.models.mpo import MPO
from tensorkrowch.formats import TT
from tensorkrowch.formats.formats1d import TensorFormat1D

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

    An :class:`~tensorkrowch.models.MPS` is converted to TT or TR, and an
    :class:`~tensorkrowch.models.MPO` to TTM or TRM, according to their
    boundaries. A sequence of cores instead constructs an open TT.
    Quantics formats require an explicit :class:`CallableQuanticsSource`.
    Environment and sketch contractions require an unbatched open TT with
    every site selected as an input. Effective cores are prepared for each
    contraction, with explicit bond factors absorbed.

    Parameters
    ----------
    format : TensorFormat1D, MPS, MPO or sequence of torch.Tensor
        Lightweight vector/matrix format, MPS, MPO or raw TT cores. Model
        boundaries determine the open or cyclic representation.

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
                 format: Union[TensorFormat1D, MPS, MPO,
                               Sequence[torch.Tensor]]) -> None:
        if isinstance(format, MPS):
            format = format.to_tr() if format.boundary == 'pbc' \
                else format.to_tt()
        elif isinstance(format, MPO):
            format = format.to_trm() if format.boundary == 'pbc' \
                else format.to_ttm()
        if not isinstance(format, TensorFormat1D):
            if isinstance(format, (str, bytes)) or \
                    not isinstance(format, Sequence):
                raise TypeError(
                    '`format` should be a format, MPS, MPO or core sequence')
            cores = tuple(format)
            if not cores or not all(isinstance(core, torch.Tensor)
                                    for core in cores):
                raise TypeError('`format` should contain torch.Tensor cores')
            if cores and (cores[0].ndim == 3) and (cores[-1].ndim == 3):
                cores = (cores[0].squeeze(0), *cores[1:-1], cores[-1].squeeze(-1)) \
                    if len(cores) > 1 else (cores[0].squeeze(0).squeeze(-1),)
            format = TT(cores)
        if format._quantized:
            raise TypeError(
                'Quantics formats require an explicit Quantics source')

        self.format = format
        super().__init__(format.n_sites,
                         n_batches=format.n_batches,
                         in_sites=(None if format.out_dim is not None
                                   else format.in_sites))
        if format.out_dim is not None:
            self._matrix_dims = format.in_dim, format.out_dim

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Dimensions of all vector variables or matrix input axes."""
        return self.format.in_dim

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the backing format."""
        return self.format.dtype

    @property
    def device(self) -> torch.device:
        """Device of the backing format."""
        return self.format.device

    @property
    def _supports_tt_contractions(self) -> bool:
        """Whether direct TT environment and sketch contractions are supported."""
        format = self.format
        return (isinstance(format, TT) and not format._cyclic and
                not format.n_batches and
                self.in_sites == tuple(range(format.n_sites)))

    def _partial_indices(self,
                         configurations: torch.Tensor,
                         *,
                         start: int,
                         name: str) -> torch.Tensor:
        """Validates index columns for consecutive sites beginning at ``start``."""
        if not isinstance(configurations, torch.Tensor):
            raise TypeError(f'`{name}` should be torch.Tensor type')
        if (configurations.ndim != 2) or \
                configurations.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                f'`{name}` should be a two-dimensional integer tensor')
        stop = start + configurations.shape[1]
        if (start < 0) or (stop > self.format.n_sites):
            raise ValueError(f'`{name}` contains too many sites')
        indices = configurations.to(device=self.device, dtype=torch.long)
        for offset, dimension in enumerate(self.in_dim[start:stop]):
            values = indices[:, offset]
            if torch.any(values < 0) or torch.any(values >= dimension):
                raise ValueError(
                    f'`{name}` is out of bounds at partial site {offset}')
        return indices

    def left_envs(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Contracts the first TT sites at fixed indices, leaving their right
        bond open.

        Parameters
        ----------
        configurations : torch.Tensor
            Integer indices with shape ``(batch, prefix_sites)`` for the first
            ``prefix_sites`` sites, in chain order. Zero columns select the
            empty prefix.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, right_rank)``, where ``right_rank`` is the bond
            dimension after the prefix. An empty prefix returns ones with
            shape ``(batch, 1)``. Direct contractions do not record point
            evaluations.
        """
        indices = self._partial_indices(
            configurations, start=0, name='configurations')
        cores = self.format._effective_cores()
        environment = cores[0].new_ones((indices.shape[0], 1))
        for site in range(indices.shape[1]):
            matrices = cores[site][:, indices[:, site], :].permute(1, 0, 2)
            environment = torch.einsum('ba,bar->br', environment, matrices)
        return environment

    def right_envs(self, configurations: torch.Tensor) -> torch.Tensor:
        """
        Contracts the final TT sites at fixed indices, leaving their left
        bond open.

        Parameters
        ----------
        configurations : torch.Tensor
            Integer indices with shape ``(batch, suffix_sites)`` for the final
            ``suffix_sites`` sites, in chain order. Zero columns select the
            empty suffix.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, left_rank)``, where ``left_rank`` is the bond
            dimension before the suffix. An empty suffix returns ones with
            shape ``(batch, 1)``. Direct contractions do not record point
            evaluations.
        """
        start = self.format.n_sites - configurations.shape[1] \
            if isinstance(configurations, torch.Tensor) and \
            (configurations.ndim == 2) else 0
        indices = self._partial_indices(
            configurations, start=start, name='configurations')
        cores = self.format._effective_cores()
        environment = cores[0].new_ones((indices.shape[0], 1))
        for offset in range(indices.shape[1] - 1, -1, -1):
            site = start + offset
            matrices = cores[site][:, indices[:, offset], :].permute(1, 0, 2)
            environment = torch.einsum('bar,br->ba', matrices, environment)
        return environment

    @staticmethod
    def _local_phi(core: torch.Tensor,
                   left: torch.Tensor,
                   right: torch.Tensor,
                   factor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Contracts a ``(left, input, right)`` core with sketch environments."""
        if not isinstance(left, torch.Tensor) or (left.ndim != 2):
            raise TypeError('`left` should be a matrix')
        if not isinstance(right, torch.Tensor) or (right.ndim != 2):
            raise TypeError('`right` should be a matrix')
        if left.shape[1] != core.shape[0]:
            raise ValueError('`left` does not match the source left rank')
        if right.shape[1] != core.shape[-1]:
            raise ValueError('`right` does not match the source right rank')
        dtype = torch.promote_types(
            core.dtype, torch.promote_types(left.dtype, right.dtype))
        if factor is not None:
            if not isinstance(factor, torch.Tensor) or (
                    factor.shape != (core.shape[1],)):
                raise ValueError(
                    '`factor` should match the site input dimension')
            dtype = torch.promote_types(dtype, factor.dtype)
        core = core.to(dtype=dtype)
        if factor is not None:
            core = core * factor.to(device=core.device,
                                    dtype=dtype)[None, :, None]
        return torch.einsum('la,aib,rb->lir',
                            left.to(device=core.device, dtype=dtype),
                            core,
                            right.to(device=core.device, dtype=dtype))

    def local_phi(self,
                  site: int,
                  left: torch.Tensor,
                  right: torch.Tensor,
                  factor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Contracts a TT core with left and right sketch environments, leaving
        its input axis open. Environments are used without conjugation.

        Parameters
        ----------
        site : int
            Site of the source core.
        left : torch.Tensor
            Shape ``(left_size, left_rank)``. Columns match the core's left
            bond; rows identify left sketch coordinates.
        right : torch.Tensor
            Shape ``(right_size, right_rank)``. Columns match the core's right
            bond; rows identify right sketch coordinates.
        factor : torch.Tensor, optional
            Multiplicative input weights of shape ``(in_dim[site],)``.
            ``None`` uses unit weights.

        Returns
        -------
        torch.Tensor
            Local Phi with shape ``(left_size, in_dim[site], right_size)``.
        """
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= self.format.n_sites):
            raise ValueError('`site` should identify one source site')
        cores = self.format._effective_cores()
        return self._local_phi(cores[site], left, right, factor)

    def marginal_phi(self,
                     site: int,
                     order: int = 1,
                     factor: Optional[Sequence[torch.Tensor]] = None
                     ) -> torch.Tensor:
        """
        Sums distant TT sites while retaining the central site and its local
        neighbors for a Markov marginal sketch.

        Parameters
        ----------
        site : int
            Central site.
        order : int
            Maximum number of neighboring sites retained on each side,
            limited by the chain boundaries. Defaults to ``1``.
        factor : sequence of torch.Tensor, optional
            One multiplicative weight vector of shape ``(in_dim[k],)`` per
            site ``k``, applied before retaining or summing its input axis.
            ``None`` uses unit weights.

        Returns
        -------
        torch.Tensor
            Shape ``(left_size, in_dim[site], right_size)``. Retained neighbors
            on each side are flattened in chain order into the first and last
            axes. A side without neighbors has size ``1``.
        """
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= self.format.n_sites):
            raise ValueError('`site` should identify one source site')
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError('`order` should be int type')
        if order < 1:
            raise ValueError('`order` should be positive')
        cores = self.format._effective_cores()
        if factor is None:
            factors = [core.new_ones(core.shape[1]) for core in cores]
        else:
            factors = list(factor)
            if len(factors) != self.format.n_sites:
                raise ValueError('`factor` should contain one vector per site')
            dtype = self.dtype
            for position, (value, dimension) in enumerate(zip(factors,
                                                              self.in_dim)):
                if not isinstance(value, torch.Tensor) or (
                        value.shape != (dimension,)):
                    raise ValueError(
                        f'`factor` at site {position} has an invalid shape')
                dtype = torch.promote_types(dtype, value.dtype)
            cores = [core.to(dtype=dtype) for core in cores]
            factors = [value.to(device=self.device, dtype=dtype)
                       for value in factors]

        left_start = max(0, site - order)
        left = cores[0].new_ones((1, 1))
        for position in range(left_start):
            matrix = torch.einsum(
                'aib,i->ab', cores[position], factors[position])
            left = left @ matrix
        for position in range(left_start, site):
            core = cores[position] * factors[position][None, :, None]
            left = torch.einsum('la,aib->lib', left, core).reshape(
                -1, core.shape[-1])

        right_stop = min(len(cores), site + 1 + order)
        right = cores[0].new_ones((1, 1))
        for position in range(len(cores) - 1, right_stop - 1, -1):
            matrix = torch.einsum(
                'aib,i->ab', cores[position], factors[position])
            right = torch.einsum('ab,rb->ra', matrix, right)
        for position in range(right_stop - 1, site, -1):
            core = cores[position] * factors[position][None, :, None]
            right = torch.einsum('aib,rb->ira', core, right).reshape(
                -1, core.shape[0])
        return self._local_phi(cores[site], left, right, factor=factors[site])

    def tt_sketch_phis(self,
                       left_cores: Sequence[Sequence[torch.Tensor]],
                       right_cores: Sequence[Sequence[torch.Tensor]],
                       boundary_scale: float = 1.0) -> Tuple[torch.Tensor, ...]:
        """
        Contracts TT sketch stacks with the source to build every local Phi.
        Sketches are contracted in a batch, reusing prefix and suffix
        environments across sites. Sketch cores are used without conjugation.

        Parameters
        ----------
        left_cores : sequence of sequences of torch.Tensor
            TT sketches in the left stack, each with one core per source site.
            Core ``k`` has shape ``(left_rank, in_dim[k], right_rank)``, with
            unit boundary ranks. Sketches share bond dimensions at each cut.
        right_cores : sequence of sequences of torch.Tensor
            TT sketches in the right stack, with the same layout and number
            of sketches as ``left_cores``.
        boundary_scale : float
            Multiplier applied to the first left and last right sketch cores
            in their environment contractions. Defaults to ``1.0``.

        Returns
        -------
        tuple of torch.Tensor
            One Phi per site ``k``, with shape
            ``(left_size, in_dim[k], right_size)``. Each outer axis combines
            sketch and bond indices; chain boundary axes have size ``1``.
        """
        left_cores = tuple(tuple(sketch_cores) for sketch_cores in left_cores)
        right_cores = tuple(tuple(sketch_cores) for sketch_cores in right_cores)
        if not left_cores or (len(left_cores) != len(right_cores)):
            raise ValueError(
                'Left and right stacks should contain the same number '
                'of sketches')
        if any(len(sketch_cores) != self.format.n_sites
               for sketch_cores in (*left_cores, *right_cores)):
            raise ValueError(
                'Every TT sketch should contain one core per site')

        source_cores = self.format._effective_cores()
        dtype = source_cores[0].dtype
        for sketch_cores in (*left_cores, *right_cores):
            for core in sketch_cores:
                dtype = torch.promote_types(dtype, core.dtype)

        stack_size = len(left_cores)
        left_cores = tuple(torch.stack(site_cores).to(device=self.device,
                                                      dtype=dtype)
                           for site_cores in zip(*left_cores))
        right_cores = tuple(torch.stack(site_cores).to(device=self.device,
                                                       dtype=dtype)
                            for site_cores in zip(*right_cores))
        source_cores = tuple(core.to(dtype=dtype) for core in source_cores)

        left = [source_cores[0].new_ones((1, 1))]
        left_envs = source_cores[0].new_ones((stack_size, 1, 1))
        for site in range(len(source_cores) - 1):
            left_envs = torch.einsum(
                'sac,aib,scid->sbd',
                left_envs, source_cores[site], left_cores[site])
            if site == 0:
                left_envs = left_envs * boundary_scale
            left.append(left_envs.permute(0, 2, 1).reshape(
                -1, left_envs.shape[1]))

        right = [None] * len(source_cores)
        right[-1] = source_cores[0].new_ones((1, 1))
        right_envs = source_cores[0].new_ones((stack_size, 1, 1))
        for site in range(len(source_cores) - 1, 0, -1):
            right_envs = torch.einsum(
                'aib,scid,sbd->sac',
                source_cores[site], right_cores[site], right_envs)
            if site == len(source_cores) - 1:
                right_envs = right_envs * boundary_scale
            right[site - 1] = right_envs.permute(0, 2, 1).reshape(
                -1, right_envs.shape[1])

        return tuple(self._local_phi(core, left[site], right[site])
                     for site, core in enumerate(source_cores))

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
            configurations, self._in_sites_dims, self.device)
        if self.is_matrix:
            result = self.format.evaluate(indices,
                                          in_sites=range(self.format.n_sites),
                                          out_sites=())
        else:
            result = self.format.evaluate(indices, in_sites=self.in_sites)
        if self.n_batches:
            result = result.movedim(self.n_batches, 0)
        self._record_evaluation(points=indices.shape[0])
        return result

    def fiber(self,
              configurations: ConfigurationBatch,
              site: int,
              values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Varies one input site while fixing the other inputs. Uses direct TT
        environments when available, otherwise :meth:`BaseSource.fiber`.

        Parameters
        ----------
        configurations : ConfigurationBatch
            Integer indices with shape ``(batch, len(in_sites))``, with columns
            ordered by ``in_sites``.
        site : int
            Represented variable to vary; it must belong to ``in_sites``.
        values : torch.Tensor, optional
            Candidate integer indices with shape ``(n_values,)``. ``None``
            evaluates every index in ``range(in_dim[site])``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_values, *core_batch, *out_shape)``. The first
            axis follows configurations; the second follows candidate values.
        """
        if not self._supports_tt_contractions:
            return super().fiber(configurations, site, values)

        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('`site` should be int type')
        if (site < 0) or (site >= self.format.n_sites):
            raise ValueError('`site` should identify an input site')

        indices = _discrete_indices(configurations, self.in_dim, self.device)
        cores = self.format._effective_cores()
        matrices = [core[:, indices[:, position], :].permute(1, 0, 2)
                    for position, core in enumerate(cores)]

        left = cores[0].new_ones((indices.shape[0], 1))
        for matrix in matrices[:site]:
            left = torch.einsum('ba,bar->br', left, matrix)
        right = cores[0].new_ones((indices.shape[0], 1))
        for matrix in reversed(matrices[site + 1:]):
            right = torch.einsum('bar,br->ba', matrix, right)
        result = torch.einsum('ba,apr,br->bp', left, cores[site], right)

        if values is None:
            self._record_evaluation(points=indices.shape[0] * self.in_dim[site])
            return result

        if not isinstance(values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        if (values.ndim != 1) or (values.dtype not in _INTEGER_DTYPES):
            raise TypeError(
                '`values` should be a one-dimensional integer tensor')
        values = values.to(device=self.device, dtype=torch.long)
        if torch.any(values < 0) or torch.any(values >= self.in_dim[site]):
            raise ValueError(
                '`values` are out of bounds for the selected site')

        result = result.index_select(1, values)
        self._record_evaluation(points=indices.shape[0] * values.shape[0])
        return result

    def contract_dense(self) -> torch.Tensor:
        """
        Contracts the complete backing format in original site order,
        independently of ``in_sites``.

        Returns
        -------
        torch.Tensor
            Shape ``(*core_batch, *in_dim)`` for vectors. Matrices interleave
            input/output axes as ``(*core_batch, in_0, out_0, in_1, out_1, ...)``.
        """
        return self.format.contract_dense()


__all__ = [
    'FormatSource',
]
