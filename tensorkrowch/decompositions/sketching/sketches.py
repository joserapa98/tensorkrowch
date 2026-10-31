"""
This script contains:

    Classes:
        * _PreparedPhi
        * _SketchWeights
        * SketchOperator
        * SketchSystemBuilder
        * CoreDeterminingSystem
        * _SupportSketchSystemBuilder
        * _SampledStructuredSystemBuilder
        * _MarginalStructuredSystemBuilder
        * _TTStackStructuredSystemBuilder
        * SampledSketch
        * MarginalSketch
        * TTStackSketch

    Functions:
        * _prepare_phi
        * _sampled_blocks
        * _one_hot_ids
        * _selected_ids
        * _sample_states
        * _structured_capability
        * _validate_structured_batch_size
        * _gaussian
        * _left_orthogonal
        * _right_orthogonal
"""

from dataclasses import dataclass, field
from math import prod, sqrt
from typing import (Any,
                    Callable,
                    Mapping,
                    Optional,
                    Protocol,
                    Sequence,
                    Tuple,
                    Union,
                    runtime_checkable)

import torch

from tensorkrowch.utils import _INTEGER_DTYPES, truncated_svd

from tensorkrowch.decompositions._utils import (_normalize_rank,
                                                _normalize_dims,
                                                _ravel_indices,
                                                _unravel_indices)
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver
from tensorkrowch.decompositions.metrics import (DecompositionMetrics,
                                                 RangeProjectionRecord,
                                                 TruncationRecord)
from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.sketching.projections import RangeProjector
from tensorkrowch.decompositions.sketching.sources import (SupportTensorSource,
                                                           _iter_support)
from tensorkrowch.decompositions.sources import (ConfigurationBatch,
                                                 BaseSource)


_Rank = Optional[Union[int, Sequence[int]]]


@dataclass(frozen=True)
class _PreparedPhi:
    """Truncated Phi with absorbed spectra and central isometric lifts."""

    tensor: torch.Tensor
    left: Optional[torch.Tensor] = None
    right: Optional[torch.Tensor] = None
    truncations: Sequence[TruncationRecord] = ()
    projections: Sequence[RangeProjectionRecord] = ()


def _prepare_phi(tensor: torch.Tensor,
                 direction: Optional[str],
                 *,
                 rank_left: Optional[int] = None,
                 rank_right: Optional[int] = None,
                 numerical: Tuple[bool, bool] = (False, False),
                 cutoff: Optional[float] = None,
                 atol: Optional[float] = None,
                 rtol: Optional[float] = None,
                 cum_percentage: Optional[float] = None,
                 projector: Optional[RangeProjector] = None,
                 generator: Optional[torch.Generator] = None,
                 site: Optional[int] = None,
                 collect_metrics: bool = False) -> _PreparedPhi:
    """Truncates outward axes, absorbing spectra into the retained tensor."""
    if direction is None:
        return _PreparedPhi(tensor)
    if direction not in ('left', 'right', 'center'):
        raise ValueError("`direction` should be 'left', 'right' or 'center'")
    truncations = []
    projections = []

    def split(matrix: torch.Tensor,
              rank: Optional[int],
              cut: Optional[int],
              discover: bool = False) -> Tuple[torch.Tensor, ...]:
        options = {'rank': rank, 'cutoff': cutoff, 'atol': atol,
                   'rtol': rtol, 'cum_percentage': cum_percentage}
        # Auxiliary matrix cuts discover numerical rank under their physical
        # cap even when the user prescribes the paired inter-site bonds.
        if ((rank is None) or discover) and all(value is None for value in (
                cutoff, atol, rtol, cum_percentage)):
            options['rtol'] = (max(matrix.shape) *
                               torch.finfo(matrix.real.dtype).eps) ** 2
        projection = None
        if projector is not None:
            projection = projector.project(
                matrix, rank=rank, generator=generator,
                return_info=collect_metrics)
            matrix = projection.small_matrix
            if projection.record is not None:
                projections.append(projection.record)
        if collect_metrics:
            u, s, vh, info = truncated_svd(
                matrix, return_info=True, **options)
            if (cut is not None) and (cut >= 0):
                truncations.append(TruncationRecord.from_svd_info(
                    info, site=cut))
        else:
            u, s, vh = truncated_svd(matrix, **options)
        if projection is not None:
            u = projection.lift_left(u)
        return u, s, vh

    left_size, dimension, right_size = tensor.shape
    if direction == 'left':
        u, s, _ = split(
            tensor.reshape(left_size, -1).mH, rank_left,
            None if site is None else site - 1, numerical[0])
        basis = (s.to(u.dtype).unsqueeze(-1) * u.mH).reshape(
            u.shape[-1], dimension, right_size)
        return _PreparedPhi(basis, truncations=tuple(truncations),
                            projections=tuple(projections))

    u, s, vh = split(
        tensor.reshape(-1, right_size), rank_right, site, numerical[1])
    basis = (u * s.to(u.dtype)).reshape(
        left_size, dimension, u.shape[-1])
    if direction == 'right':
        return _PreparedPhi(basis, truncations=tuple(truncations),
                            projections=tuple(projections))

    right = vh
    # Select both supports in the original Phi metric. Their isometric
    # lifts leave all retained spectral scales inside the central tensor.
    _, _, vh = split(
        tensor.reshape(left_size, -1).mH, rank_left,
        None if site is None else site - 1, numerical[0])
    left = vh.mH
    compact = left.mH @ basis.reshape(left_size, -1)
    basis = compact.reshape(left.shape[-1], dimension, right.shape[0])
    return _PreparedPhi(basis, left, right, tuple(truncations),
                        tuple(projections))


def _one_hot_ids(ids: torch.Tensor,
                 size: int,
                 dtype: torch.dtype) -> torch.Tensor:
    """Creates feature rows while mapping negative ``ids`` to zero."""
    result = torch.zeros(
        ids.shape[0], size, dtype=dtype, device=ids.device)
    valid = ids >= 0
    if torch.any(valid):
        result[valid, ids[valid]] = 1
    return result


def _selected_ids(values: torch.Tensor,
                  reference: torch.Tensor) -> torch.Tensor:
    """
    Maps sorted flat ``values`` to sorted ``reference`` positions or ``-1``.
    """
    positions = torch.searchsorted(reference, values)
    safe = positions.clamp(max=max(reference.numel() - 1, 0))
    result = torch.full_like(positions, -1)
    if reference.numel():
        matched = (positions < reference.numel()) & \
            (reference.index_select(0, safe) == values)
        result[matched] = positions[matched]
    return result


def _sample_states(values: torch.Tensor,
                   maximum: Optional[int],
                   generator: Optional[torch.Generator]) -> torch.Tensor:
    """Selects a deterministic sorted state set under an optional cap."""
    states = torch.unique(values, sorted=True)
    if (maximum is None) or (states.numel() <= maximum):
        return states
    permutation = torch.randperm(
        states.numel(), generator=generator, device=states.device)[:maximum]
    return states.index_select(0, permutation).sort().values


def _sampled_blocks(states: Sequence[torch.Tensor],
                    in_dim: Sequence[int],
                    dtype: torch.dtype,
                    device: torch.device,
                    *,
                    right: bool = False) -> Tuple[torch.Tensor, ...]:
    """Builds recursive one-hot blocks in prefix or suffix index order."""
    blocks = []
    for cut, current_states in enumerate(states):
        if right:
            site = cut + 1
            old_states = current_states.new_tensor([0]) \
                if site == len(in_dim) - 1 else states[cut + 1]
            stride = prod(in_dim[site + 1:])
            current = torch.div(
                current_states, stride, rounding_mode='floor')
            previous = current_states.remainder(stride)
        else:
            site = cut
            old_states = current_states.new_tensor([0]) \
                if site == 0 else states[cut - 1]
            current = current_states.remainder(in_dim[site])
            previous = torch.div(
                current_states, in_dim[site], rounding_mode='floor')
        positions = _selected_ids(previous, old_states)
        valid = positions >= 0
        block = torch.zeros(
            current_states.numel(), in_dim[site], old_states.numel(),
            dtype=dtype, device=device)
        rows = torch.arange(current_states.numel(), device=device)[valid]
        block[rows, current[valid], positions[valid]] = 1
        blocks.append(block)
    return tuple(blocks)


def _structured_capability(source: BaseSource,
                           name: str) -> Optional[Callable[..., Any]]:
    """Returns one optional structured ``source`` kernel by capability."""
    if not getattr(source, '_supports_tt_contractions', True):
        return None
    capability = getattr(source, name, None)
    return capability if callable(capability) else None


def _validate_structured_batch_size(batch_size: Optional[int]) -> None:
    """Validates a generic batch option unused by direct contractions."""
    if (batch_size is not None) and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int)):
        raise TypeError('`batch_size` should be int type or None')
    if (batch_size is not None) and (batch_size < 1):
        raise ValueError('`batch_size` should be positive')


@dataclass(frozen=True)
class _SketchWeights:  # MARK: _SketchWeights
    """
    Evaluated left/right sketches and their corresponding recursive blocks.
    """

    left: Sequence[torch.Tensor]
    right: Sequence[torch.Tensor]
    left_blocks: Sequence[torch.Tensor]
    right_blocks: Sequence[torch.Tensor]
    value_scale: torch.Tensor
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        left = tuple(self.left)
        right = tuple(self.right)
        blocks = tuple(self.left_blocks)
        right_blocks = tuple(self.right_blocks)
        if any(len(items) != len(left)
               for items in (right, blocks, right_blocks)):
            raise ValueError('Sketch weights should align on every TT cut')
        if not isinstance(self.value_scale, torch.Tensor) or \
                (self.value_scale.ndim != 1):
            raise ValueError('`value_scale` should be a vector')
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)
        object.__setattr__(self, 'left_blocks', blocks)
        object.__setattr__(self, 'right_blocks', right_blocks)
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))


@runtime_checkable
class SketchOperator(Protocol):  # MARK: SketchOperator
    """Protocol for operators that construct a validated TT-RS system."""

    def builder(self, source: BaseSource) -> 'SketchSystemBuilder':
        """Returns the ``source``-bound core-determining-system builder."""


@runtime_checkable
class SketchSystemBuilder(Protocol):  # MARK: SketchSystemBuilder
    """Protocol for constructing local TT-RS equations from one source."""

    def build(self,
              *,
              batch_size: Optional[int] = None,
              generator: Optional[torch.Generator] = None
              ) -> 'CoreDeterminingSystem':
        """Builds sketched local tensors and recursive left blocks."""


@dataclass(frozen=True)
class CoreDeterminingSystem:  # MARK: CoreDeterminingSystem
    r"""
    Stores local TT-RS tensors and recursive sketches in both directions.

    ``phi[k]`` has shape ``(left_size[k], in_dim[k], right_size[k])``.
    ``left_blocks[k]`` maps the left sketch through site ``k`` to the next
    left sketch. ``right_blocks[k]`` maps the right sketch through site
    ``k + 1`` to the previous right sketch. The outer sketch sizes are one.
    These recursive maps let :meth:`solve` start at any ``center``.

    The local equations follow
    `Generative modeling via tensor train sketching
    <https://arxiv.org/abs/2202.11788>`_ (Hur, Hoskins, Lindsey,
    Stoudenmire and Khoo, 2022), Algorithms 1--4. Centered solves and explicit
    sparse support
    are TensorKrowch extensions of that construction.

    Parameters
    ----------
    phi : sequence of torch.Tensor
        One three-axis local tensor per site.
    left_blocks : sequence of torch.Tensor
        One tensor per cut, with shape
        ``(left_size[k + 1], in_dim[k], left_size[k])``.
    right_blocks : sequence of torch.Tensor
        One tensor per cut, with shape
        ``(right_size[k], in_dim[k + 1], right_size[k + 1])``.
    in_dim : sequence of int
        Input dimension for each tensor axis.
    operator : str
        Name of the sketch operator used to build the system.
    diagnostics : mapping, optional
        Additional construction information, copied into the result metadata.
    """

    phi: Sequence[torch.Tensor]
    left_blocks: Sequence[torch.Tensor]
    right_blocks: Sequence[torch.Tensor]
    in_dim: Sequence[int]
    operator: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phi = tuple(self.phi)
        left_blocks = tuple(self.left_blocks)
        right_blocks = tuple(self.right_blocks)
        in_dim = _normalize_dims(self.in_dim, 'in_dim')
        if len(phi) != len(in_dim):
            raise ValueError('`phi` should contain one tensor per site')
        for name, blocks in (('left_blocks', left_blocks),
                             ('right_blocks', right_blocks)):
            if len(blocks) != len(in_dim) - 1:
                raise ValueError(f'`{name}` should contain one block per cut')
        for site, tensor in enumerate(phi):
            if not isinstance(tensor, torch.Tensor) or (tensor.ndim != 3):
                raise TypeError('`phi` should contain three-axis tensors')
            if (min(tensor.shape) < 1) or (tensor.shape[1] != in_dim[site]):
                raise ValueError('`phi` dimensions should match `in_dim`')
            if not (tensor.is_floating_point() or tensor.is_complex()):
                raise TypeError(
                    '`phi` should contain floating or complex data')
            if not torch.isfinite(tensor).all():
                raise ValueError('`phi` should contain finite values')
            if (tensor.dtype != phi[0].dtype) or \
                    (tensor.device != phi[0].device):
                raise ValueError('`phi` should share a device and dtype')
        if (phi[0].shape[0] != 1) or (phi[-1].shape[-1] != 1):
            raise ValueError('The outer sketch dimensions should be one')
        for cut, (left, right) in enumerate(zip(left_blocks, right_blocks)):
            expected = (
                (left, (phi[cut + 1].shape[0], in_dim[cut],
                        phi[cut].shape[0])),
                (right, (phi[cut].shape[-1], in_dim[cut + 1],
                         phi[cut + 1].shape[-1])))
            for block, shape in expected:
                if not isinstance(block, torch.Tensor):
                    raise TypeError('Recursive blocks should be tensors')
                if block.shape != shape:
                    raise ValueError('Recursive block dimensions do not match')
                if (block.dtype != phi[0].dtype) or \
                        (block.device != phi[0].device):
                    raise ValueError(
                        'Recursive blocks and `phi` should share device/dtype')
        object.__setattr__(self, 'phi', phi)
        object.__setattr__(self, 'left_blocks', left_blocks)
        object.__setattr__(self, 'right_blocks', right_blocks)
        object.__setattr__(self, 'in_dim', in_dim)
        object.__setattr__(self, 'diagnostics', dict(self.diagnostics))

    def solve(self,
              rank: _Rank = None,
              *,
              center: Optional[int] = None,
              cutoff: Optional[float] = None,
              atol: Optional[float] = None,
              rtol: Optional[float] = None,
              cum_percentage: Optional[float] = None,
              transform: Optional[Callable] = None,
              range_projector: Optional[RangeProjector] = None,
              generator: Optional[torch.Generator] = None,
              strict_system: bool = False,
              collect_metrics: bool = False,
              _on_core: Optional[Callable] = None) -> TTDecomposition:
        """
        Truncates local ranges and solves outward from ``center``.

        Each cut is selected by one local SVD, with both directions treated
        at the center. Singular values stay outside the retained bases.
        Least-squares alignment with the recursive sketches then produces
        the TT cores. Ranks are neither padded nor rounded afterward.

        Parameters
        ----------
        rank : int, sequence of int or None, optional
            Maximum rank at each TT cut. A sequence contains ``n_sites - 1``
            values. ``None`` discovers numerical ranks from the local SVDs.
        center : int or None, optional
            Initial site. ``None`` selects the middle site.
        cutoff, atol, rtol, cum_percentage : float or None, optional
            Truncation criteria passed to
            :func:`tensorkrowch.utils.truncated_svd`.
            With no rank or criterion, a numerical tolerance is used.
        transform : callable or None, optional
            Applied to each three-axis Phi before its SVDs. Returns a tensor
            with the same shape; consistency between sites is the user's
            responsibility.
        range_projector : RangeProjector or None, optional
            Complete range-projection strategy. ``None`` applies SVD directly.
        generator : torch.Generator or None, optional
            Generator used by a randomized range projector.
        strict_system : bool, optional
            Raises if a local alignment matrix lacks full column rank.
        collect_metrics : bool, optional
            Whether to collect truncation and local least-squares diagnostics.

        Returns
        -------
        TTDecomposition
            Tensor train containing the aligned cores and diagnostics.
        """
        n_sites = len(self.in_dim)
        if center is None:
            center = n_sites // 2
        if isinstance(center, bool) or not isinstance(center, int):
            raise TypeError('`center` should be int type or None')
        if (center < 0) or (center >= n_sites):
            raise ValueError('`center` should select a tensor site')
        if not isinstance(strict_system, bool):
            raise TypeError('`strict_system` should be bool type')
        if not isinstance(collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        if (transform is not None) and not callable(transform):
            raise TypeError('`transform` should be callable or None')
        if (range_projector is not None) and not isinstance(
                range_projector, RangeProjector):
            raise TypeError(
                '`range_projector` should implement RangeProjector')
        if (generator is not None) and not isinstance(generator,
                                                      torch.Generator):
            raise TypeError('`generator` should be torch.Generator or None')
        ranks = (None,) * (n_sites - 1) if rank is None else \
            _normalize_rank(rank, n_sites - 1)
        ranks = tuple(
            min(prod(self.in_dim[:cut + 1]), prod(self.in_dim[cut + 1:]),
                ranks[cut] if ranks[cut] is not None else
                prod(self.in_dim))
            for cut in range(n_sites - 1))
        if (rank is None) and all(value is None for value in (
                cutoff, atol, rtol, cum_percentage)):
            scale = max(max(tensor.shape) for tensor in self.phi)
            rtol = (scale * torch.finfo(self.phi[0].real.dtype).eps) ** 2
        options = {'cutoff': cutoff, 'atol': atol, 'rtol': rtol,
                   'cum_percentage': cum_percentage,
                   'projector': range_projector, 'generator': generator,
                   'collect_metrics': collect_metrics}
        metrics = DecompositionMetrics()
        bases = []
        central = None

        phis = []
        dtype = self.phi[0].dtype
        for tensor in self.phi:
            if transform is not None:
                transformed = transform(tensor)
                if not isinstance(transformed, torch.Tensor):
                    raise TypeError('`transform` should return a tensor')
                if transformed.shape != tensor.shape:
                    raise ValueError('`transform` should preserve Phi shape')
                if transformed.device != tensor.device:
                    raise ValueError('`transform` should preserve Phi device')
                if not (transformed.is_floating_point() or
                        transformed.is_complex()):
                    raise TypeError(
                        '`transform` should return floating or complex data')
                if not torch.isfinite(transformed).all():
                    raise ValueError('`transform` should return finite values')
                tensor = transformed
            dtype = torch.promote_types(dtype, tensor.dtype)
            phis.append(tensor)

        # Every non-boundary cut is determined by one outward-facing Phi.
        for site, tensor in enumerate(phis):
            tensor = tensor.to(dtype=dtype)
            direction = 'center' if site == center else \
                ('left' if site < center else 'right')
            if (site != center) and (site in (0, n_sites - 1)):
                prepared = _PreparedPhi(tensor)
            else:
                prepared = _prepare_phi(
                    tensor, direction,
                    rank_left=1 if site == 0 else ranks[site - 1],
                    rank_right=1 if site == n_sites - 1 else ranks[site],
                    site=site, **options)
            bases.append(prepared.tensor)
            metrics.truncations.extend(
                record for record in prepared.truncations
                if record.site < n_sites - 1)
            metrics.range_projections.extend(prepared.projections)
            if site == center:
                central = prepared

        # Unused outer factors belong in the center, since no branch restores
        # them when the selected center is an endpoint (including one site).
        central_core = central.tensor
        left_lift, right_lift = central.left, central.right
        if center == 0:
            central_core = torch.einsum('ab,bpc->apc', left_lift,
                                        central_core)
            left_lift = torch.ones_like(left_lift)
        if center == n_sites - 1:
            central_core = torch.einsum('apb,bc->apc', central_core,
                                        right_lift)
            right_lift = torch.ones_like(right_lift)
        bases[center] = central_core
        cores = [None] * n_sites
        cores[center] = central_core
        if _on_core is not None:
            _on_core(center, central_core)
        coefficient_ranks = [None] * n_sites
        solver = LeastSquaresSolver()

        def solve_local(matrix: torch.Tensor,
                        target: torch.Tensor,
                        site: int) -> torch.Tensor:
            if strict_system or collect_metrics:
                numerical_rank = int(torch.linalg.matrix_rank(
                    matrix).detach().cpu().item())
                coefficient_ranks[site] = numerical_rank
                if strict_system and (numerical_rank < matrix.shape[1]):
                    raise ValueError(
                        'Core-determining matrix is rank deficient '
                        f'at site {site}: rank={numerical_rank}/'
                        f'{matrix.shape[1]}')
            solution, record = solver.solve(matrix, target, site=site,
                                            return_record=collect_metrics)
            if collect_metrics:
                metrics.local_solves.append(record)
            return solution

        # A center lifting factor maps the neighboring TT bond back to sketch
        # coordinates. After one step, the next outward basis replaces it.
        previous = torch.einsum('ab,bpc->apc', left_lift, central_core)
        for site in range(center + 1, n_sites):
            coefficient = torch.einsum(
                'apb,bpc->ac', self.left_blocks[site - 1].to(dtype), previous)
            target = bases[site]
            solution = solve_local(
                coefficient, target.reshape(target.shape[0], -1), site)
            cores[site] = solution.reshape(
                solution.shape[0], self.in_dim[site], target.shape[-1])
            if _on_core is not None:
                _on_core(site, cores[site])
            previous = target

        previous = torch.einsum('apb,bc->apc', central_core, right_lift)
        for site in range(center - 1, -1, -1):
            coefficient = torch.einsum(
                'apb,mpb->am', previous, self.right_blocks[site].to(dtype))
            target = bases[site]
            solution = solve_local(
                coefficient.mT,
                target.reshape(-1, target.shape[-1]).mT, site).mT
            cores[site] = solution.reshape(
                target.shape[0], self.in_dim[site], solution.shape[-1])
            if _on_core is not None:
                _on_core(site, cores[site])
            previous = target

        metrics.truncations.sort(key=lambda record: record.site)
        metrics.local_solves.sort(key=lambda record: record.site)
        if n_sites == 1:
            compact = [cores[0].reshape(self.in_dim[0])]
        else:
            compact = [cores[0].squeeze(0), *cores[1:-1],
                       cores[-1].squeeze(-1)]
        return TTDecomposition(
            compact, metrics=metrics,
            metadata={'algorithm': 'tt_rs', 'center': center,
                      'sketch_operator': self.operator,
                      'coefficient_ranks': coefficient_ranks
                      if strict_system or collect_metrics else None,
                      'system': dict(self.diagnostics)})


class _SupportSketchSystemBuilder:  # MARK: _SupportSketchSystemBuilder
    """Contracts finite support, enumerating only non-support sources."""

    def __init__(self, source: BaseSource, operator: SketchOperator) -> None:
        if not isinstance(source, BaseSource):
            raise TypeError('`source` should inherit from BaseSource')
        if tuple(source.out_shape) != ():
            raise ValueError('TT-RS currently requires a scalar source')
        self.source = source
        self.operator = operator

    def build(self,
              *,
              batch_size: Optional[int] = None,
              generator: Optional[torch.Generator] = None
              ) -> CoreDeterminingSystem:
        """Builds all Phi tensors in support-linear memory when possible."""
        if isinstance(self.source, SupportTensorSource):
            batches = list(_iter_support(self.source, batch_size=batch_size))
            if not batches:
                raise ValueError(
                    'TT-RS cannot decompose an empty sparse support')
            indices = torch.cat(
                [batch.as_tensor() for batch, _ in batches], dim=0)
            values = torch.cat([value for _, value in batches], dim=0)
            source_path = 'support'
        else:
            axes = [
                torch.arange(dimension, device=self.source.device)
                for dimension in self.source.in_dim
            ]
            indices = torch.cartesian_prod(*axes).reshape(
                -1, len(self.source.in_dim))
            if batch_size is None:
                batch_size = indices.shape[0]
            if isinstance(batch_size, bool) or not isinstance(batch_size, int):
                raise TypeError('`batch_size` should be int type or None')
            if batch_size < 1:
                raise ValueError('`batch_size` should be positive')
            values = []
            for start in range(0, indices.shape[0], batch_size):
                values.append(self.source.evaluate(ConfigurationBatch(
                    indices[start:start + batch_size], kind='indices')))
            values = torch.cat(values, dim=0)
            source_path = 'enumerated_grid'
        nonzero = values != 0
        indices = indices[nonzero]
        values = values[nonzero]
        if not values.numel():
            raise ValueError('TT-RS cannot decompose an empty sparse support')
        weights = self.operator._weights(
            indices,
            self.source.in_dim,
            values.dtype,
            generator)
        values = values * weights.value_scale.to(values.dtype)
        phis = []
        n_sites = len(self.source.in_dim)
        for site, dimension in enumerate(self.source.in_dim):
            left = values.new_ones(values.shape[0], 1) if site == 0 \
                else weights.left[site - 1].to(values.dtype)
            right = values.new_ones(values.shape[0], 1) \
                if site == n_sites - 1 \
                else weights.right[site].to(values.dtype)
            contributions = values[:, None, None] * \
                left[:, :, None] * right[:, None, :]
            phi = values.new_zeros(
                dimension, left.shape[1], right.shape[1])
            phi.index_add_(0, indices[:, site], contributions)
            phis.append(phi.permute(1, 0, 2))
        return CoreDeterminingSystem(
            phi=phis,
            left_blocks=weights.left_blocks,
            right_blocks=weights.right_blocks,
            in_dim=self.source.in_dim,
            operator=type(self.operator).__name__,
            diagnostics={
                'support_size': values.shape[0],
                'source_path': source_path,
                'left_dimensions': tuple(
                    item.shape[1] for item in weights.left),
                'right_dimensions': tuple(item.shape[1]
                                          for item in weights.right),
                **weights.diagnostics,
            })


class _SampledStructuredSystemBuilder:  # MARK: _SampledStructuredSystemBuilder
    """Builds sampled equations from exact TT prefix/suffix kernels."""

    def __init__(self,
                 source: BaseSource,
                 operator: 'SampledSketch') -> None:
        self.source = source
        self.operator = operator

    def build(self,
              *,
              batch_size: Optional[int] = None,
              generator: Optional[torch.Generator] = None
              ) -> CoreDeterminingSystem:
        _validate_structured_batch_size(batch_size)
        dimensions = self.source.in_dim
        device = self.source.device
        if self.operator.samples is None:
            reference = None
        else:
            reference = self.operator.samples.to(
                device=device, dtype=torch.long)
            if reference.shape[1] != len(dimensions):
                raise ValueError('`samples` should contain one index per site')
            for site, dimension in enumerate(dimensions):
                if torch.any(reference[:, site] < 0) or \
                        torch.any(reference[:, site] >= dimension):
                    raise ValueError(
                        f'`samples` are out of bounds at site {site}')

        prefix_states = []
        suffix_states = []
        for site in range(len(dimensions) - 1):
            if reference is None:
                prefix_ids = torch.arange(
                    prod(dimensions[:site + 1]), device=device)
                suffix_ids = torch.arange(
                    prod(dimensions[site + 1:]), device=device)
            else:
                prefix_ids = _ravel_indices(
                    reference[:, :site + 1], dimensions[:site + 1])
                suffix_ids = _ravel_indices(
                    reference[:, site + 1:], dimensions[site + 1:])
            states = _sample_states(
                prefix_ids, self.operator.sketch_size, generator)
            suffix = _sample_states(
                suffix_ids, self.operator.sketch_size, generator)
            prefix_states.append(states)
            suffix_states.append(suffix)

        blocks = _sampled_blocks(
            prefix_states, dimensions, self.source.dtype, device)
        right_blocks = _sampled_blocks(
            suffix_states, dimensions, self.source.dtype, device, right=True)

        left_kernel = _structured_capability(
            self.source, 'left_envs')
        right_kernel = _structured_capability(
            self.source, 'right_envs')
        phi_kernel = _structured_capability(self.source, 'local_phi')
        phis = []
        for site in range(len(dimensions)):
            left = torch.ones(
                1, 1, dtype=self.source.dtype, device=device) if site == 0 \
                else left_kernel(_unravel_indices(
                    prefix_states[site - 1], dimensions[:site]))
            right = torch.ones(
                1, 1, dtype=self.source.dtype, device=device) \
                if site == len(dimensions) - 1 else right_kernel(
                    _unravel_indices(
                        suffix_states[site], dimensions[site + 1:]))
            phis.append(phi_kernel(site, left, right))
        return CoreDeterminingSystem(
            phi=phis,
            left_blocks=blocks,
            right_blocks=right_blocks,
            in_dim=dimensions,
            operator=type(self.operator).__name__,
            diagnostics={
                'source_path': 'structured_tt',
                'structured_kernel': 'sampled_prefix_suffix',
                'support_size': None,
                'sampled_states': tuple(
                    states.numel() for states in prefix_states),
                'left_dimensions': tuple(
                    states.numel() for states in prefix_states),
                'right_dimensions': tuple(
                    states.numel() for states in suffix_states),
            })


class _MarginalStructuredSystemBuilder:  # MARK: marginal structured builder
    """Builds Markov marginal equations from direct TT contractions."""

    def __init__(self,
                 source: BaseSource,
                 operator: 'MarginalSketch') -> None:
        self.source = source
        self.operator = operator

    def build(self,
              *,
              batch_size: Optional[int] = None,
              generator: Optional[torch.Generator] = None
              ) -> CoreDeterminingSystem:
        _validate_structured_batch_size(batch_size)
        del generator
        dimensions = self.source.in_dim
        blocks = self.operator._left_blocks(
            dimensions, self.source.dtype, self.source.device)
        phi_kernel = _structured_capability(self.source, 'marginal_phi')
        phis = [
            phi_kernel(site, self.operator.order, self.operator.factor)
            for site in range(len(dimensions))
        ]
        return CoreDeterminingSystem(
            phi=phis,
            left_blocks=blocks,
            right_blocks=self.operator._right_blocks(
                dimensions, self.source.dtype, self.source.device),
            in_dim=dimensions,
            operator=type(self.operator).__name__,
            diagnostics={
                'source_path': 'structured_tt',
                'structured_kernel': 'markov_marginal',
                'support_size': None,
                'marginal_order': self.operator.order,
                'weighted_marginal': self.operator.factor is not None,
                'left_dimensions': tuple(
                    block.shape[0] for block in blocks),
                'right_dimensions': tuple(
                    prod(dimensions[site + 1:min(
                        len(dimensions), site + 1 + self.operator.order)])
                    for site in range(len(dimensions) - 1)),
            })


class _TTStackStructuredSystemBuilder:  # MARK: _TTStackStructuredSystemBuilder
    """Builds Gaussian TT-stack equations by double-layer contraction."""

    def __init__(self,
                 source: BaseSource,
                 operator: 'TTStackSketch') -> None:
        self.source = source
        self.operator = operator

    def build(self,
              *,
              batch_size: Optional[int] = None,
              generator: Optional[torch.Generator] = None
              ) -> CoreDeterminingSystem:
        _validate_structured_batch_size(batch_size)
        components = self.operator._components(
            self.source.in_dim,
            self.source.dtype,
            self.source.device,
            generator)
        left_cores, right_cores, blocks, right_blocks = components
        phi_kernel = _structured_capability(self.source, 'tt_sketch_phis')
        phis = phi_kernel(
            left_cores, right_cores,
            boundary_scale=1 / sqrt(self.operator.stack_size))
        feature_dim = self.operator.tt_rank * self.operator.stack_size
        return CoreDeterminingSystem(
            phi=phis,
            left_blocks=blocks,
            right_blocks=right_blocks,
            in_dim=self.source.in_dim,
            operator=type(self.operator).__name__,
            diagnostics={
                'source_path': 'structured_tt',
                'structured_kernel': 'tt_stack_double_layer',
                'support_size': None,
                'tt_rank': self.operator.tt_rank,
                'stack_size': self.operator.stack_size,
                'sketch_dimension': feature_dim,
                'orthogonal': self.operator.orthogonal,
                'distribution': (
                    'orthogonal_variant' if self.operator.orthogonal
                    else 'gaussian_tt_stack'),
                'core_determining_gate': (
                    'recursive_left_and_suffix_right'),
                'left_dimensions': (feature_dim,) * (
                    len(self.source.in_dim) - 1),
                'right_dimensions': (feature_dim,) * (
                    len(self.source.in_dim) - 1),
            })


class SampledSketch:  # MARK: SampledSketch
    """
    Recursive one-hot sketches induced by correlated configurations.

    For a sparse source the default configurations are its non-zero support.
    Prefix and suffix indicators select states from the same reference data.
    Missing sparse entries contribute their known zero without evaluation.

    Parameters
    ----------
    samples : torch.Tensor or None, optional
        Reference index configurations with shape ``(batch, n_sites)``.
        ``None`` uses the source support or all states for a format source.
    sketch_size : int or None, optional
        Maximum number of distinct prefix and suffix states at each cut.
        ``None`` retains every reference state.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(torch.ones(2, 3, 2))
    >>> system = tk.decompositions.SampledSketch().builder(source).build()
    >>> result = system.solve(rank=1)
    >>> torch.allclose(result.contract_dense(), torch.ones(2, 3, 2))
    True
    """

    def __init__(self,
                 samples: Optional[torch.Tensor] = None,
                 sketch_size: Optional[int] = None) -> None:
        if (samples is not None) and (
                not isinstance(samples, torch.Tensor) or (samples.ndim != 2) or
                (samples.dtype not in _INTEGER_DTYPES)):
            raise TypeError(
                '`samples` should be a two-dimensional integer tensor')
        if (samples is not None) and (min(samples.shape) < 1):
            raise ValueError('`samples` should contain inputs and rows')
        if (sketch_size is not None) and (
                isinstance(sketch_size, bool) or
                not isinstance(sketch_size, int)):
            raise TypeError('`sketch_size` should be int type or None')
        if (sketch_size is not None) and (sketch_size < 1):
            raise ValueError('`sketch_size` should be positive')
        self.samples = samples
        self.sketch_size = sketch_size

    def builder(self, source: BaseSource) -> SketchSystemBuilder:
        """
        Binds the sketch to ``source`` and returns its system builder.

        Parameters
        ----------
        source : BaseSource
            Finite scalar source. Format sources use direct contractions;
            other sources use sparse support or finite-grid evaluation.

        Returns
        -------
        SketchSystemBuilder
            Builder whose ``build`` method constructs local Phi tensors.
        """
        if all(_structured_capability(source, name) is not None for name in (
                'left_envs', 'right_envs', 'local_phi')):
            return _SampledStructuredSystemBuilder(source, self)
        return _SupportSketchSystemBuilder(source, self)

    def _weights(self,
                 indices: torch.Tensor,
                 in_dim: Sequence[int],
                 dtype: torch.dtype,
                 generator: Optional[torch.Generator]) -> _SketchWeights:
        """Evaluates prefix/suffix indicators and their recursive maps."""
        reference = indices if self.samples is None else self.samples.to(
            device=indices.device, dtype=torch.long)
        if reference.shape[1] != len(in_dim):
            raise ValueError('`samples` should contain one index per site')
        for site, dimension in enumerate(in_dim):
            if torch.any(reference[:, site] < 0) or \
                    torch.any(reference[:, site] >= dimension):
                raise ValueError(f'`samples` are out of bounds at site {site}')

        left = []
        right = []
        prefix_states = []
        suffix_states = []
        for cut in range(len(in_dim) - 1):
            prefix_dim = in_dim[:cut + 1]
            states = _sample_states(
                _ravel_indices(reference[:, :cut + 1], prefix_dim),
                self.sketch_size, generator)
            prefix_states.append(states)
            ids = _ravel_indices(indices[:, :cut + 1], prefix_dim)
            left.append(_one_hot_ids(
                _selected_ids(ids, states), states.numel(), dtype))

            suffix_dim = in_dim[cut + 1:]
            states = _sample_states(
                _ravel_indices(reference[:, cut + 1:], suffix_dim),
                self.sketch_size, generator)
            suffix_states.append(states)
            ids = _ravel_indices(indices[:, cut + 1:], suffix_dim)
            right.append(_one_hot_ids(
                _selected_ids(ids, states), states.numel(), dtype))

        blocks = _sampled_blocks(
            prefix_states, in_dim, dtype, indices.device)
        right_blocks = _sampled_blocks(
            suffix_states, in_dim, dtype, indices.device, right=True)
        return _SketchWeights(
            left, right, blocks, right_blocks,
            torch.ones(indices.shape[0], device=indices.device, dtype=dtype),
            diagnostics={'sampled_states': tuple(
                states.numel() for states in prefix_states)})


class MarginalSketch:  # MARK: MarginalSketch
    """
    Marginal sketches that retain a local neighborhood at every cut.

    With ``order=1``, this implements the neighbor-preserving sketches of
    Section 5.1 of `Generative modeling via tensor train sketching
    <https://arxiv.org/abs/2202.11788>`_ (Hur, Hoskins, Lindsey,
    Stoudenmire and Khoo, 2022). Higher ``order`` values and an explicit
    contraction ``factor``
    are TensorKrowch extensions of the same recursive marginal structure.

    Parameters
    ----------
    order : int, optional
        Number of neighboring variables retained on each side of a cut.
    factor : sequence of torch.Tensor or None, optional
        One vector of shape ``(in_dim[k],)`` per site. Their product weights
        source values before marginalization. ``None`` uses unweighted sums.

    Examples
    --------
    >>> source = tk.decompositions.DenseTensorSource(torch.ones(2, 3, 2))
    >>> system = tk.decompositions.MarginalSketch().builder(source).build()
    >>> result = system.solve(rank=1, center=1)
    >>> torch.allclose(result.contract_dense(), torch.ones(2, 3, 2))
    True
    """

    def __init__(self,
                 order: int = 1,
                 factor: Optional[Sequence[torch.Tensor]] = None) -> None:
        if isinstance(order, bool) or not isinstance(order, int):
            raise TypeError('`order` should be int type')
        if order < 1:
            raise ValueError('`order` should be positive')
        if factor is not None:
            factor = tuple(factor)
            if not all(isinstance(value, torch.Tensor) and (value.ndim == 1)
                       for value in factor):
                raise TypeError('`factor` should contain vectors')
        self.order = order
        self.factor = factor

    def builder(self, source: BaseSource) -> SketchSystemBuilder:
        """
        Binds the sketch to ``source`` and returns its system builder.

        Parameters
        ----------
        source : BaseSource
            Finite scalar source. Format sources use direct contractions;
            other sources use sparse support or finite-grid evaluation.

        Returns
        -------
        SketchSystemBuilder
            Builder whose ``build`` method constructs local Phi tensors.
        """
        if _structured_capability(source, 'marginal_phi') is not None:
            return _MarginalStructuredSystemBuilder(source, self)
        return _SupportSketchSystemBuilder(source, self)

    def _left_blocks(self,
                     in_dim: Optional[Sequence[int]],
                     dtype: Optional[torch.dtype],
                     device: Optional[Union[str, torch.device]]
                     ) -> Tuple[torch.Tensor, ...]:
        """Constructs recursive Markov blocks independently of source rows."""
        blocks = []
        for site in range(len(in_dim) - 1):
            left_start = max(0, site + 1 - self.order)
            left_dim = in_dim[left_start:site + 1]
            old_start = max(0, site - self.order)
            old_dim = in_dim[old_start:site]
            old_size = prod(old_dim) if old_dim else 1
            new_size = prod(left_dim)
            block = torch.zeros(
                new_size, in_dim[site], old_size,
                dtype=dtype, device=device)
            old_ids = torch.arange(old_size, device=device)
            for value in range(in_dim[site]):
                if self.order == 1:
                    new_ids = torch.full_like(old_ids, value)
                else:
                    tail_modulus = prod(old_dim[-(self.order - 1):]) \
                        if old_dim else 1
                    tail = torch.remainder(old_ids, tail_modulus)
                    new_ids = tail * in_dim[site] + value
                block[new_ids, value, old_ids] = 1
            blocks.append(block)
        return blocks

    def _right_blocks(self,
                      in_dim: Sequence[int],
                      dtype: torch.dtype,
                      device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Constructs suffix blocks in the original physical-axis order."""
        blocks = []
        for site in range(1, len(in_dim)):
            new_dim = in_dim[site:min(len(in_dim), site + self.order)]
            old_dim = in_dim[site + 1:min(
                len(in_dim), site + 1 + self.order)]
            old_size = prod(old_dim)
            block = torch.zeros(
                prod(new_dim), in_dim[site], old_size,
                dtype=dtype, device=device)
            old_ids = torch.arange(old_size, device=device)
            retained_size = prod(new_dim[1:])
            dropped_size = prod(old_dim[len(new_dim) - 1:])
            head = torch.div(old_ids, dropped_size, rounding_mode='floor')
            for value in range(in_dim[site]):
                new_ids = value * retained_size + head
                block[new_ids, value, old_ids] = 1
            blocks.append(block)
        return tuple(blocks)

    def _weights(self,
                 indices: torch.Tensor,
                 in_dim: Optional[Sequence[int]],
                 dtype: Optional[torch.dtype],
                 generator: Optional[torch.Generator]) -> _SketchWeights:
        del generator
        if (self.factor is not None) and (len(self.factor) != len(in_dim)):
            raise ValueError('`factor` should contain one vector per site')
        scale = torch.ones(
            indices.shape[0], dtype=dtype, device=indices.device)
        if self.factor is not None:
            for site, (factor, dimension) in enumerate(zip(
                    self.factor, in_dim)):
                if factor.shape != (dimension,):
                    raise ValueError(
                        f'`factor` at site {site} has an invalid shape')
                scale = scale * factor.to(
                    device=indices.device, dtype=dtype).index_select(
                        0, indices[:, site])

        left = []
        right = []
        blocks = self._left_blocks(in_dim, dtype, indices.device)
        for site in range(len(in_dim) - 1):
            left_start = max(0, site + 1 - self.order)
            left_dim = in_dim[left_start:site + 1]
            left_ids = _ravel_indices(
                indices[:, left_start:site + 1], left_dim)
            left.append(_one_hot_ids(left_ids, prod(left_dim), dtype))

            right_stop = min(len(in_dim), site + 1 + self.order)
            right_dim = in_dim[site + 1:right_stop]
            right_ids = _ravel_indices(
                indices[:, site + 1:right_stop], right_dim)
            right.append(_one_hot_ids(right_ids, prod(right_dim), dtype))

        return _SketchWeights(
            left, right, blocks,
            self._right_blocks(in_dim, dtype, indices.device), scale,
            diagnostics={
                'marginal_order': self.order,
                'weighted_marginal': self.factor is not None,
            })


def _gaussian(shape: Sequence[int],
              dtype: Optional[torch.dtype],
              device: Optional[Union[str, torch.device]],
              generator: Optional[torch.Generator],
              scale: float) -> torch.Tensor:
    """Draws real or complex Gaussian entries with explicit ``scale``."""
    return torch.randn(
        *shape, dtype=dtype, device=device, generator=generator) * scale


def _left_orthogonal(core: torch.Tensor) -> torch.Tensor:
    """Orthogonalizes the outgoing columns of one TT ``core``."""
    matrix = core.reshape(-1, core.shape[-1])
    if matrix.shape[0] >= matrix.shape[1]:
        matrix = torch.linalg.qr(matrix, mode='reduced').Q
    else:
        matrix = torch.linalg.qr(matrix.T, mode='reduced').Q.T
    return matrix.reshape(core.shape)


def _right_orthogonal(core: torch.Tensor) -> torch.Tensor:
    """Orthogonalizes the incoming rows of one TT ``core``."""
    matrix = core.reshape(core.shape[0], -1)
    if matrix.shape[1] >= matrix.shape[0]:
        matrix = torch.linalg.qr(matrix.T, mode='reduced').Q.T
    else:
        matrix = torch.linalg.qr(matrix, mode='reduced').Q
    return matrix.reshape(core.shape)


class TTStackSketch:  # MARK: TTStackSketch
    """
    Stack of Gaussian TT sketches with configurable size and internal rank.

    ``stack_size`` is the number of TT sketches in each left or right stack.

    ``tt_rank=1`` is the separable Khatri-Rao limit. ``stack_size=1`` is one
    Gaussian TT projection with ``tt_rank`` output rows at every partial cut.
    The full partial-sketch dimension is ``stack_size * tt_rank``. Setting
    ``orthogonal=True`` selects a separately labelled orthogonalized variant;
    it does not claim the iid-Gaussian guarantees of the default construction.

    The random cores follow Definition 3.1 of `Linear-scaling Tensor Train
    Sketching <https://arxiv.org/abs/2603.11009>`_ by Paul Cazeaux, Mi-Song
    Dupuy and Rodrigo Figueroa Justiniano (2026). That paper establishes
    embedding guarantees for the global TTStack map. Using its partial suffix
    contractions inside TT-RS is an additional construction here, so the
    builder separately verifies the recursive left blocks and right-sketch
    dimensions required by :class:`CoreDeterminingSystem`.

    Parameters
    ----------
    tt_rank : int, optional
        Internal bond dimension of each random TT sketch.
    stack_size : int, optional
        Number of independent TT sketches in each left or right stack.
    orthogonal : bool, optional
        Whether to orthogonalize random cores toward their outgoing bond.

    Examples
    --------
    >>> operator = tk.decompositions.TTStackSketch(tt_rank=2, stack_size=3)
    >>> operator.tt_rank * operator.stack_size
    6
    """

    def __init__(self,
                 tt_rank: int = 1,
                 stack_size: int = 1,
                 orthogonal: bool = False) -> None:
        for name, value in (('tt_rank', tt_rank), ('stack_size', stack_size)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            if value < 1:
                raise ValueError(f'`{name}` should be positive')
        if not isinstance(orthogonal, bool):
            raise TypeError('`orthogonal` should be bool type')
        self.tt_rank = tt_rank
        self.stack_size = stack_size
        self.orthogonal = orthogonal

    def builder(self, source: BaseSource) -> SketchSystemBuilder:
        """
        Binds the sketch to ``source`` and returns its system builder.

        Parameters
        ----------
        source : BaseSource
            Finite scalar source. Format sources use direct contractions;
            other sources use sparse support or finite-grid evaluation.

        Returns
        -------
        SketchSystemBuilder
            Builder whose ``build`` method constructs local Phi tensors.
        """
        if _structured_capability(source, 'tt_sketch_phis') is not None:
            return _TTStackStructuredSystemBuilder(source, self)
        return _SupportSketchSystemBuilder(source, self)

    def _components(self,
                    in_dim: Optional[Sequence[int]],
                    dtype: Optional[torch.dtype],
                    device: Optional[Union[str, torch.device]],
                    generator: Optional[torch.Generator]) -> Any:
        """Draws left/right TT stacks and both recursive block sequences."""
        feature_dim = self.tt_rank * self.stack_size
        scale = 1 / sqrt(self.tt_rank)
        stack_scale = 1 / sqrt(self.stack_size)

        left_cores = []
        right_cores = []
        for _ in range(self.stack_size):
            left_sketch_cores = []
            right_sketch_cores = []
            for site, dimension in enumerate(in_dim):
                left_rank = 1 if site == 0 else self.tt_rank
                right_rank = 1 if site == len(in_dim) - 1 \
                    else self.tt_rank
                left_core = _gaussian(
                    (left_rank, dimension, right_rank),
                    dtype, device, generator, scale)
                right_core = _gaussian(
                    (left_rank, dimension, right_rank),
                    dtype, device, generator, scale)
                if self.orthogonal:
                    left_core = _left_orthogonal(left_core)
                    right_core = _right_orthogonal(right_core)
                left_sketch_cores.append(left_core)
                right_sketch_cores.append(right_core)
            left_cores.append(tuple(left_sketch_cores))
            right_cores.append(tuple(right_sketch_cores))

        blocks = []
        for site in range(len(in_dim) - 1):
            block = torch.zeros(
                feature_dim, in_dim[site],
                1 if site == 0 else feature_dim,
                dtype=dtype, device=device)
            for stack_idx in range(self.stack_size):
                core = left_cores[stack_idx][site]
                start = stack_idx * self.tt_rank
                if site == 0:
                    block[start:start + self.tt_rank, :, 0] = \
                        core[0].T * stack_scale
                else:
                    block[start:start + self.tt_rank, :,
                          start:start + self.tt_rank] = core.permute(2, 1, 0)
            blocks.append(block)
        right_blocks = []
        for site in range(1, len(in_dim)):
            block = torch.zeros(
                feature_dim, in_dim[site],
                1 if site == len(in_dim) - 1 else feature_dim,
                dtype=dtype, device=device)
            for stack_idx in range(self.stack_size):
                core = right_cores[stack_idx][site]
                start = stack_idx * self.tt_rank
                if site == len(in_dim) - 1:
                    block[start:start + self.tt_rank, :, 0] = \
                        core[:, :, 0] * stack_scale
                else:
                    block[start:start + self.tt_rank, :,
                          start:start + self.tt_rank] = core
            right_blocks.append(block)
        return (tuple(left_cores), tuple(right_cores), tuple(blocks),
                tuple(right_blocks))

    def _weights(self,
                 indices: torch.Tensor,
                 in_dim: Optional[Sequence[int]],
                 dtype: Optional[torch.dtype],
                 generator: Optional[torch.Generator]) -> _SketchWeights:
        feature_dim = self.tt_rank * self.stack_size
        device = indices.device
        stack_scale = 1 / sqrt(self.stack_size)
        left_cores, right_cores, blocks, right_blocks = self._components(
            in_dim, dtype, device, generator)

        left = []
        state = torch.ones(
            indices.shape[0], self.stack_size, 1,
            dtype=dtype, device=device)
        for site in range(len(in_dim) - 1):
            next_state = []
            for stack_idx in range(self.stack_size):
                core = left_cores[stack_idx][site]
                selected = core[:, indices[:, site], :].permute(1, 0, 2)
                value = torch.einsum(
                    'nba,nb->na', selected, state[:, stack_idx])
                if site == 0:
                    value = value * stack_scale
                next_state.append(value)
            state = torch.stack(next_state, dim=1)
            left.append(state.reshape(indices.shape[0], feature_dim))

        right = [None] * (len(in_dim) - 1)
        state = torch.ones(
            indices.shape[0], self.stack_size, 1,
            dtype=dtype, device=device)
        for site in range(len(in_dim) - 1, 0, -1):
            next_state = []
            for stack_idx in range(self.stack_size):
                core = right_cores[stack_idx][site]
                selected = core[:, indices[:, site], :].permute(1, 0, 2)
                value = torch.einsum(
                    'nba,na->nb', selected, state[:, stack_idx])
                if site == len(in_dim) - 1:
                    value = value * stack_scale
                next_state.append(value)
            state = torch.stack(next_state, dim=1)
            right[site - 1] = state.reshape(indices.shape[0], feature_dim)

        return _SketchWeights(
            left, right, blocks, right_blocks,
            torch.ones(indices.shape[0], dtype=dtype, device=device),
            diagnostics={
                'tt_rank': self.tt_rank,
                'stack_size': self.stack_size,
                'sketch_dimension': feature_dim,
                'orthogonal': self.orthogonal,
                'distribution': (
                    'orthogonal_variant' if self.orthogonal
                    else 'gaussian_tt_stack'),
                'core_determining_gate': 'recursive_left_and_suffix_right',
            })


__all__ = [
    'SketchOperator',
    'SketchSystemBuilder',
    'CoreDeterminingSystem',
    'SampledSketch',
    'MarginalSketch',
    'TTStackSketch',
]
