"""
This script contains:

    Internal functions:
        * _restore_cores
        * _from_standard_cores
        * _canonicalize_cores
        * _redistribute
        * _validate_minimal_options

    Public functions:
        * split_block

    Internal classes:
        * _OpenFormat1D
        * _CyclicFormat1D
        * _VectorFormat1D
        * _MatrixFormat1D

    Public classes:
        * TensorFormat1D
        * TT, TR, TTM, TRM

    Aliases:
        * EvaluationData

Core names used in this module:

    * ``cores`` / ``_cores``: stored cores in the shapes of the concrete format;
      open chains omit unit boundary axes and matrices keep input/output axes.
    * ``standard_cores``: cores with explicit boundary axes and a single input
      axis; matrix input/output dimensions are combined. Bond factors remain
      separate.
    * ``effective_cores``: standard cores with bond factors absorbed.
    * ``operator_cores``: effective cores with separate input/output axes;
      vectors use a unit axis according to their row or column orientation.
"""

import warnings
from abc import abstractmethod
from copy import copy
from math import isfinite, prod, sqrt
from numbers import Number, Real
from typing import (TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple,
                    Union)

import torch

from tensorkrowch.utils import (_INTEGER_DTYPES, _validate_truncation,
                                truncated_svd)

from tensorkrowch.formats.base import (_SafeList, RoundingInfo, SplitBlock,
                                       BlockLayout, SampleError, TensorFormat)
from tensorkrowch.formats.bonds import BondFactors1D, VidalGauge
from tensorkrowch.formats.orbits import (GaugeOrbit, TensorRingOrbit,
                                         MinimalCanonicalInfo)


if TYPE_CHECKING:
    from tensorkrowch.models import MPS, MPSData, MPO
    from tensorkrowch.utils import _TruncatedSVDInfo


EvaluationData = Union[torch.Tensor, Sequence[torch.Tensor]]


###############################################################################
#                              INTERNAL FUNCTIONS                             #
###############################################################################
def _restore_cores(cores: Sequence[torch.Tensor],
                   in_dim: Sequence[int],
                   out_dim: Optional[Sequence[int]],
                   n_batches: int,
                   cyclic: bool) -> List[torch.Tensor]:
    """Converts standard cores to the shapes stored by the vector or matrix format."""
    stored = []
    for site, core in enumerate(cores):
        if out_dim is not None:
            core = core.reshape(*core.shape[:-3], core.shape[-3],
                                in_dim[site], out_dim[site], core.shape[-1])
            core = core.transpose(-1, -2)
        if not cyclic:
            if site == 0:
                core = core.squeeze(n_batches)
            if site == len(cores) - 1:
                core = core.squeeze(-2 if out_dim is not None else -1)
        stored.append(core)
    return stored


def _from_standard_cores(cores: Sequence[torch.Tensor],
                         in_dim: Sequence[int],
                         out_dim: Optional[Sequence[int]],
                         n_batches: int,
                         cyclic: bool) -> 'TensorFormat1D':
    """Constructs a plain 1D format from standard cores."""
    cores = _restore_cores(cores, in_dim, out_dim, n_batches, cyclic)
    if out_dim is None:
        cls = TR if cyclic else TT
    else:
        cls = TRM if cyclic else TTM
    return cls(cores, n_batches=n_batches)


def _canonicalize_cores(cores: Sequence[torch.Tensor],
                        orth_center: int,
                        renormalize: bool) -> List[torch.Tensor]:
    """Returns QR/RQ-gauged cores without modifying a format."""
    cores = list(cores)
    if not all(torch.isfinite(core).all() for core in cores):
        raise ValueError('Canonicalization requires finite cores')

    batch_shape = cores[0].shape[:-3]
    log_scale = cores[0].real.new_zeros(batch_shape)

    # Move the orthogonality center from both sides.
    for site in range(orth_center):
        core = cores[site]
        matrix = core.reshape(*batch_shape, -1, core.shape[-1])
        q, r = torch.linalg.qr(matrix, mode='reduced')
        if renormalize:
            scale = torch.linalg.vector_norm(r, dim=(-2, -1))
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            r = r / scale[..., None, None]
            log_scale = log_scale + scale.log()
        cores[site] = q.reshape(*batch_shape,
                                core.shape[-3], core.shape[-2], q.shape[-1])
        cores[site + 1] = torch.einsum('...ab,...bpr->...apr',
                                       r, cores[site + 1])

    for site in range(len(cores) - 1, orth_center, -1):
        core = cores[site]
        matrix = core.reshape(*batch_shape, core.shape[-3], -1)
        q, r = torch.linalg.qr(matrix.transpose(-2, -1), mode='reduced')
        r, q = r.transpose(-2, -1), q.transpose(-2, -1)
        if renormalize:
            scale = torch.linalg.vector_norm(r, dim=(-2, -1))
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            r = r / scale[..., None, None]
            log_scale = log_scale + scale.log()
        cores[site] = q.reshape(*batch_shape,
                                q.shape[-2], core.shape[-2], core.shape[-1])
        cores[site - 1] = torch.einsum('...apb,...bc->...apc',
                                       cores[site - 1], r)

    if renormalize:
        rescale = (log_scale / len(cores)).exp()[..., None, None, None]
        cores = [core * rescale for core in cores]

    return cores


def _redistribute(cores: List[torch.Tensor],
                  spectra: Sequence[torch.Tensor],
                  old_powers: Sequence[Tuple[float, float]],
                  powers: Sequence[Tuple[float, float]]
                  ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
    """Moves stored Schmidt powers between neighbours without another SVD."""
    for bond, (spectrum, old, new) in enumerate(zip(spectra, old_powers, powers)):
        negative_power = min(new[0] - old[0], new[1] - old[1],
                             1 - new[0] - new[1])
        if negative_power < 0:
            positive = spectrum > 0
            cutoff = torch.finfo(spectrum.dtype).eps * spectrum.amax(dim=-1,
                                                                     keepdim=True)
            safe = torch.where(positive, spectrum, torch.ones_like(spectrum))
            if torch.any(positive & (spectrum <= cutoff)) or not torch.all(
                    torch.isfinite(safe.pow(negative_power))):
                raise ValueError(
                    f'Bond {bond} has singular values too small for stable '
                    'inverse powers; use rounding with a cutoff first')

        for neighbour, difference, left_core in ((bond, new[0] - old[0], True),
                                                 (bond + 1, new[1] - old[1], False)):
            if difference == 0:
                continue
            if difference < 0:
                safe = torch.where(spectrum > 0,
                                   spectrum,
                                   torch.ones_like(spectrum))
                factor = torch.where(spectrum > 0,
                                     safe.pow(difference),
                                     torch.zeros_like(spectrum))
            else:
                factor = spectrum.pow(difference)

            factor = factor[..., None, None, :] if left_core \
                else factor[..., :, None, None]
            cores[neighbour] = cores[neighbour] * factor

    bond_factors = []
    for spectrum, (left, right) in zip(spectra, powers):
        exponent = 1 - left - right
        if exponent < 0:
            safe = torch.where(spectrum > 0,
                               spectrum,
                               torch.ones_like(spectrum))
            factor = torch.where(spectrum > 0,
                                 safe.pow(exponent),
                                 torch.zeros_like(spectrum))
        else:
            factor = spectrum.pow(exponent)
        bond_factors.append(None if exponent == 0 else factor)

    return cores, bond_factors


def _validate_minimal_options(max_iter: int,
                              lr: float,
                              tol: float,
                              return_info: bool) -> None:
    """Checks the options shared by minimal canonicalization methods."""
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    if isinstance(max_iter, bool) or not isinstance(max_iter, int):
        raise TypeError('`max_iter` should be int type')
    if max_iter < 1:
        raise ValueError('`max_iter` should be positive')
    for name, value in [('lr', lr), ('tol', tol)]:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f'`{name}` should be a real number')
        if not isfinite(value) or value <= 0:
            raise ValueError(f'`{name}` should be finite and positive')


###############################################################################
#                               PUBLIC FUNCTIONS                              #
###############################################################################
def split_block(block: torch.Tensor,
                in_dim: Sequence[int],
                out_dim: Optional[Sequence[int]] = None,
                n_batches: int = 0,
                rank: Optional[int] = None,
                cutoff: Optional[float] = None,
                atol: Optional[float] = None,
                rtol: Optional[float] = None,
                cum_percentage: Optional[float] = None,
                mode: str = 'right',
                renormalize: bool = False,
                _svd_callback: Optional[Callable[
                    [int, '_TruncatedSVDInfo',
                     torch.Tensor, torch.Tensor], None]] = None
                ) -> SplitBlock:
    """
    Splits a local tensor sitewise with both external ranks preserved.

    Only internal bonds are truncated. External ranks remain unchanged, and
    structural batches share retained ranks. In inverse mode, each singular
    value is absorbed in both neighbouring cores and its pseudoinverse remains
    on the bond: zeros stay zero, while positive values too small to invert
    stably raise an error. Multiple truncation criteria select the most
    restrictive retained rank.

    Parameters
    ----------
    block : torch.Tensor
        Local tensor shaped ``(*core_batch, left, *physical, right)``. For
        vectors, ``physical`` contains one ``in_dim`` axis per site. For
        matrices, it contains interleaved ``in_dim`` and ``out_dim`` axes;
        each pair is combined internally into a dimension ``in_dim * out_dim``.
    in_dim : sequence of int
        Input dimension for each local site.
    out_dim : sequence of int, optional
        Matrix output dimensions paired with ``in_dim``. ``None`` treats the
        block as a vector format.
    n_batches : int
        Number of leading structural batch axes in block.
    rank : int, optional
        Maximum number of singular values to keep.
    cutoff : float, optional
        Minimum singular value to keep. It must be finite and non-negative.
        Singular values ``<= cutoff`` are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded
        while the accumulated sum of squares is ``<= atol``. It must be finite
        and non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded
        while the tail sum of squares divided by the total sum of squares is
        ``<= rtol``. It must be finite and in [0, 1].
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent
        to setting ``rtol = 1 - cum_percentage``. It must be finite and in [0,
        1].
    mode : {"explicit", "implicit", "inverse", "left", "right"}
        Distribution of each local spectrum between its neighboring cores.
        These select ``powers`` ``(0, 0)``, ``(0.5, 0.5)``, ``(1, 1)``,
        ``(1, 0)`` and ``(0, 1)``, respectively; see :class:`VidalGauge` for
        how these powers are applied. ``"left"`` and ``"right"`` name the core
        absorbing the spectrum, not the canonical direction: absorbing it on
        the left leaves the right core right-isometric, and vice versa.
    renormalize : bool
        Temporarily divides each local matrix by its largest absolute entry
        before SVD, then restores that scale to its singular values. This
        stabilizes each SVD without redistributing the block's overall scale
        among the returned cores.

    Returns
    -------
    SplitBlock
        Standard cores with both virtual axes and one physical axis per
        site, separate diagonal factors, and local singular values. Local
        spectra are not certified global Schmidt values.

    Examples
    --------
    >>> block = torch.eye(2).reshape(1, 2, 2, 1)
    >>> local = tk.formats.split_block(block, in_dim=(2, 2))
    >>> format = tk.formats.TT([torch.zeros(2, 1), torch.zeros(1, 2)])
    >>> _ = format.replace_cores(0, local.cores, bonds=local.bonds)
    >>> torch.allclose(format.contract_dense(), torch.eye(2))
    True
    """
    if not isinstance(block, torch.Tensor):
        raise TypeError('`block` should be torch.Tensor type')
    if isinstance(n_batches, bool) or not isinstance(n_batches, int):
        raise TypeError('`n_batches` should be int type')
    if n_batches < 0:
        raise ValueError('`n_batches` should be non-negative')

    in_dim = tuple(in_dim)
    if not in_dim or any(isinstance(dim, bool) or \
        not isinstance(dim, int) or (dim < 1) for dim in in_dim):
        raise ValueError('Input dimensions should be positive integers')

    if out_dim is not None:
        out_dim = tuple(out_dim)
        if (len(out_dim) != len(in_dim)) or any(isinstance(dim, bool) or \
            not isinstance(dim, int) or (dim < 1) for dim in out_dim):
            raise ValueError(
                'Output dimensions should match the positive site dimensions')

    dimensions = in_dim if out_dim is None else tuple(
        dim for pair in zip(in_dim, out_dim) for dim in pair)

    if (block.ndim != n_batches + len(dimensions) + 2) or (
        tuple(block.shape[n_batches + 1:-1]) != dimensions):
        raise ValueError(
            'Block physical axes should match the requested dimensions')
    if (block.shape[n_batches] < 1) or (block.shape[-1] < 1):
        raise ValueError('External ranks should be positive')

    _validate_truncation(rank, cutoff, atol, rtol, cum_percentage)

    if not isinstance(renormalize, bool):
        raise TypeError('`renormalize` should be bool type')

    powers = {'explicit': (0, 0),
              'implicit': (0.5, 0.5),
              'inverse': (1, 1),
              'left': (1, 0),
              'right': (0, 1)}
    if mode not in powers:
        raise ValueError('Invalid local bond distribution mode')

    # Reshape block
    batch_shape = block.shape[:n_batches]
    left = block.shape[n_batches]
    physical = in_dim if out_dim is None else tuple(
            a * b for a, b in zip(in_dim, out_dim))
    right = block.shape[-1]

    state = block.reshape(*batch_shape, left, *physical, right)

    # SVD sweep
    cores, spectra = [], []
    for site, site_dim in enumerate(physical[:-1]):
        left = state.shape[n_batches]
        matrix = state.reshape(*batch_shape, left * site_dim, -1)

        scale = matrix.real.new_ones(batch_shape)
        scaled_cutoff, scaled_atol = cutoff, atol
        if renormalize:
            scale = matrix.abs().amax(dim=(-2, -1))
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            matrix = matrix / scale[..., None, None]

            # A common retained rank is selected across structural batches.
            if cutoff is not None:
                scaled_cutoff = cutoff / scale.max().item()
            if atol is not None:
                scaled_atol = atol / scale.max().item() ** 2

        decomposition = truncated_svd(tensor=matrix,
                                      rank=rank,
                                      cutoff=scaled_cutoff,
                                      atol=scaled_atol,
                                      rtol=rtol,
                                      cum_percentage=cum_percentage,
                                      return_info=_svd_callback is not None)
        u, s, vh = decomposition[:3]

        if _svd_callback is not None:
            _svd_callback(site, decomposition[3], s, scale.log())

        s = s * scale.unsqueeze(-1)
        core = u.reshape(*batch_shape, left, site_dim, s.shape[-1])

        cores.append(core)
        spectra.append(s)
        state = s.unsqueeze(-1) * vh

    left = state.shape[n_batches]
    core = state.reshape(*batch_shape, left, physical[-1], right)

    cores.append(core)
    cores, factors = _redistribute(cores=cores,
                                   spectra=spectra,
                                   old_powers=[(0, 1)] * len(spectra),
                                   powers=[powers[mode]] * len(spectra))

    return SplitBlock(tuple(cores), tuple(factors), tuple(spectra))


class TensorFormat1D(TensorFormat):
    """
    Compact format for tensors with a 1D chain layout, formed by a sequence of
    cores and, possibly, explicit bond factors. Vector formats have one local
    input dimension per core; matrix formats have local input and output
    dimensions. These local dimensions are called physical dimensions. For a
    matrix core, the input and output dimensions may be kept as separate axes
    or fused into one axis of size ``in_dim * out_dim``.

    Serves as a base class for all 1D chain formats, such as :class:`TT`,
    :class:`TR`, :class:`TTM`, :class:`TRM`, and their quantized versions.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes required by the concrete format. The core
        list is copied, but its tensors are reused without cloning or detaching
        them, preserving autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores. The factor list is copied, but its
        tensors are reused without cloning or detaching them. Factors are
        validated together with the cores.
    """

    _family = 'tensor'
    _topology = 'tensor'
    _cyclic = False
    _quantized = False

    def __init__(self,
                 cores: Sequence[torch.Tensor],
                 n_batches: int = 0,
                 *,
                 bonds: Optional[Sequence[Optional[torch.Tensor]]] = None
                 ) -> None:
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        self._n_batches = n_batches
        self._orth_center = None
        self._bonds = None
        self._set_cores(cores, bonds)

    @property
    def n_sites(self) -> int:
        """Number of sites represented by the stored cores."""
        return len(self._cores)

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension associated with every site."""
        return self._in_dim

    @property
    def out_dim(self) -> Optional[Tuple[int, ...]]:
        """Output dimension per site for a matrix format."""
        return self._out_dim

    @property
    def rank(self) -> List[int]:
        """Bond ranks inferred from the cores."""
        return list(self._rank)

    @property
    def n_batches(self) -> int:
        """Number of leading structural batch axes."""
        return self._n_batches

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Batch dimensions shared by the cores."""
        return self._batch_shape

    @property
    def device(self) -> torch.device:
        """Device shared by all cores."""
        return self._cores[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Data type shared by all cores."""
        return self._cores[0].dtype

    @property
    def topology(self) -> str:
        """Topology identifier used in serialized result information."""
        return self._topology

    @property
    def orth_center(self) -> Optional[int]:
        """
        Recorded orthogonality center, or ``None`` when unavailable.

        For rings this is only a local gauge. With ``renormalize=True``,
        canonicalization distributes scale across the cores, so the center
        core's norm need not equal the represented tensor's norm.
        """
        return self._orth_center

    @property
    def cores(self) -> List[torch.Tensor]:
        """
        Mutable core sequence with the shapes described by the concrete class.

        Open chains omit the unit virtual axes at the first and last sites.
        Matrix cores keep their input and output axes separate. Replacing an
        entry or a same-length slice validates the new cores immediately.
        Explicit bond factors are stored separately in :attr:`bonds`.
        """
        return self._cores

    @cores.setter
    def cores(self, cores: Sequence[torch.Tensor]) -> None:
        if isinstance(cores, torch.Tensor):
            raise TypeError(
                '`cores` should be a sequence of torch.Tensor objects')
        previous = self._cores
        self._cores = _SafeList(cores, self._on_cores_changed)
        try:
            self._on_cores_changed()
        except Exception:
            self._cores = previous
            raise

    @abstractmethod
    def _validate_cores(self) -> Tuple[List[int], Tuple[int, ...],
                                       Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Validates cores and returns rank, batch, input and output dims."""

    def validate(self) -> 'TensorFormat1D':
        """
        Validates cores and bonds and refreshes structural metadata.

        Called when constructing or replacing cores. Ordinary queries and
        contractions do not call it. Invalid structure raises ``TypeError`` or
        ``ValueError``. Direct tensor shape changes remain outside the mutation
        contract.

        Returns
        -------
        TensorFormat1D
            The current format.
        """
        if not self._cores:
            raise ValueError('`cores` should contain at least one tensor')
        if not all(isinstance(core, torch.Tensor) for core in self._cores):
            raise TypeError('`cores` should contain torch.Tensor objects')

        device, dtype = self._cores[0].device, self._cores[0].dtype
        for core in self._cores:
            if any(dim < 1 for dim in core.shape):
                raise ValueError('Core dimensions should be positive')
            if core.device != device:
                raise ValueError('All cores should be on the same device')
            if core.dtype != dtype:
                raise ValueError('All cores should have the same dtype')

        rank, batch_shape, in_dim, out_dim = self._validate_cores()

        self._rank = tuple(rank)
        self._batch_shape = batch_shape

        self._in_dim = in_dim
        self._out_dim = out_dim
        self._same_in_dim = all(dim == in_dim[0] for dim in in_dim)
        self._same_out_dim = out_dim is None or all(
            dim == out_dim[0] for dim in out_dim)

        self.validate_bonds()
        return self

    def _on_cores_changed(self) -> None:
        """Validates manual core edits and restores metadata on failure."""
        previous = self.__dict__.copy()
        try:
            self.validate()
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise
        self._orth_center = None
        if isinstance(self._bonds, VidalGauge):
            self._bonds._valid = False

    def _set_cores(
            self,
            cores: Sequence[torch.Tensor],
            bonds: Optional[Sequence[Optional[torch.Tensor]]],
            *,
            spectra: Optional[Sequence[torch.Tensor]] = None,
            powers: Optional[Sequence[Tuple[float, float]]] = None
        ) -> None:
        """Replaces cores and bonds together, restoring state on invalid input."""
        if isinstance(cores, torch.Tensor):
            raise TypeError('`cores` should be a sequence of torch.Tensor objects')

        cores = list(cores)
        previous = self.__dict__.copy()
        self._cores = _SafeList(cores, self._on_cores_changed)
        try:
            if spectra is not None:
                self._bonds = VidalGauge(bonds, spectra, powers,
                                         self._on_bonds_changed)
            else:
                self._bonds = None if bonds is None else BondFactors1D(
                    bonds, self._on_bonds_changed)
            self.validate()
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise

        self._orth_center = None

    @abstractmethod
    def _standard_cores(self) -> List[torch.Tensor]:
        """Returns ``(*batch, left, physical, right)`` cores without bond factors."""

    def _effective_cores(self) -> List[torch.Tensor]:
        """Returns standard cores with explicit bond factors absorbed."""
        cores = self._standard_cores()
        if self._bonds is not None:
            for site, factor in enumerate(self._bonds.factors):
                if factor is not None:
                    cores[site] = cores[site] * factor[..., None, None, :]
        return cores

    @abstractmethod
    def _operator_cores(self) -> List[torch.Tensor]:
        """Returns effective ``(*batch, left, input, right, output)`` cores."""

    def _set_standard_cores(
            self,
            cores: Sequence[torch.Tensor],
            bonds: Optional[Sequence[Optional[torch.Tensor]]] = None,
            in_dim: Optional[Sequence[int]] = None,
            out_dim: Optional[Sequence[int]] = None,
            *,
            spectra: Optional[Sequence[torch.Tensor]] = None,
            powers: Optional[Sequence[Tuple[float, float]]] = None
        ) -> None:
        """Restores core layouts and publishes cores and factors together."""
        in_dim = self._in_dim if in_dim is None else in_dim
        out_dim = self._out_dim if out_dim is None else out_dim
        cores = _restore_cores(cores, in_dim, out_dim,
                               self._n_batches, self._cyclic)
        self._set_cores(cores, bonds, spectra=spectra, powers=powers)

    def replace_cores(self,
                      first: int,
                      cores: Sequence[torch.Tensor],
                      bonds: Optional[Sequence[Optional[torch.Tensor]]] = None
                      ) -> 'TensorFormat1D':
        """
        Replaces a consecutive group of cores and its internal bond factors.

        The number of sites and the ranks at both ends of the group stay the
        same. Ranks between the replacement cores may change. Unlike public
        :attr:`cores`, the supplied cores include both virtual axes even when
        the group touches an end of an open chain.

        Parameters
        ----------
        first : int
            Index of the first site to replace. The length of ``cores``
            determines the last site.
        cores : sequence of torch.Tensor
            Standard cores shaped ``(*batch, left, physical, right)``. For
            vectors, ``physical`` is ``in_dim``; for matrices, it is
            ``in_dim * out_dim`` at each site. The batch shape, site dimensions
            and external ranks must match the selected group.
        bonds : sequence of torch.Tensor or None, optional
            One factor per bond between replacement cores. ``None`` uses
            identity factors for those bonds. Factors outside the group remain
            unchanged. Cores and factors are validated together.

        Returns
        -------
        TensorFormat1D
            The current format. Invalid replacement leaves it unchanged.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> block = format.contract_block(0, 1)
        >>> local = tk.formats.split_block(block * 2, in_dim=format.in_dim)
        >>> _ = format.replace_cores(0, local.cores, bonds=local.bonds)
        >>> torch.allclose(format.contract_dense(), 2 * torch.eye(2))
        True
        """
        if isinstance(first, bool) or not isinstance(first, int):
            raise TypeError('The first replacement site should be an integer')
        if isinstance(cores, torch.Tensor):
            raise TypeError('Replacement cores should be a sequence')
        if isinstance(bonds, torch.Tensor):
            raise TypeError('Replacement bonds should be a sequence or None')

        cores = tuple(cores)
        if not cores:
            raise ValueError('Replacement cores should not be empty')

        last = first + len(cores) - 1
        if not 0 <= first <= last < self.n_sites:
            raise ValueError('Replacement sites should lie inside the format')

        for offset, core in enumerate(cores):
            if not isinstance(core, torch.Tensor):
                raise TypeError('Replacement cores should be tensors')
            site = first + offset
            dimension = self._in_dim[site] * \
                (self._out_dim[site] if self._out_dim else 1)
            if (core.ndim != self._n_batches + 3) or (core.shape[-2] != dimension):
                raise ValueError(
                    'Replacement physical dimensions should match the selected sites')

        stored = self._standard_cores()
        if (cores[0].shape[-3] != stored[first].shape[-3]) or \
                (cores[-1].shape[-1] != stored[last].shape[-1]):
            raise ValueError('Replacement should preserve external ranks')
        stored[first:(last + 1)] = cores

        count = self.n_sites if self._cyclic else (self.n_sites - 1)
        factors = list(self._bonds.factors) if self._bonds is not None \
            else [None] * count
        replacement_factors = [None] * (last - first) if bonds is None else list(bonds)
        if len(replacement_factors) != (last - first):
            raise ValueError(
                'Replacement factors should match its internal bonds')
        factors[first:last] = replacement_factors
        factors = factors if any(factor is not None for factor in factors) else None

        self._set_standard_cores(stored, factors)
        return self

    def _new_from_standard_cores(self,
                                 cores: Sequence[torch.Tensor],
                                 in_dim: Sequence[int],
                                 out_dim: Optional[Sequence[int]],
                                 n_batches: int,
                                 cyclic: bool,
                                 other: Optional['TensorFormat1D'] = None,
                                 product: bool = False) -> 'TensorFormat1D':
        """Builds an algebra result with the operand coordinate semantics."""
        return _from_standard_cores(cores, in_dim, out_dim, n_batches, cyclic)

    @property
    def bonds(self) -> Optional[BondFactors1D]:
        """Optional diagonal factors, with one entry per virtual bond."""
        return self._bonds

    @bonds.setter
    def bonds(self, factors: Optional[Sequence[Optional[torch.Tensor]]]) -> None:
        bonds = None if factors is None else BondFactors1D(
            factors, self._on_bonds_changed)
        if bonds is not None:
            bonds.validate(self._standard_cores(), self._cyclic)
        self._bonds = bonds
        self._orth_center = None

    def validate_bonds(self) -> 'TensorFormat1D':
        """
        Checks stored diagonal factors against current core dimensions.

        Checks factor count, ranks, batch shapes, device and compatible dtype.
        Called on controlled bond replacement; it does not certify Vidal
        spectra.

        Returns
        -------
        TensorFormat1D
            The current format.
        """
        if self._bonds is not None:
            self._bonds.validate(self._standard_cores(), self._cyclic)
        return self

    def _on_bonds_changed(self) -> None:
        """Validates manual bond edits and clears the orthogonality center."""
        self.validate_bonds()
        self._orth_center = None

    def materialize_bonds(self,
                          orth_center: Optional[int] = None) -> 'TensorFormat1D':
        """
        Absorbs stored factors into the cores in-place.

        A valid Vidal gauge redistributes its spectra towards the selected
        center, accounting for previously absorbed powers, including inverse
        Vidal. Generic or invalidated factors are absorbed once into the
        adjacent core. A mixed canonical form is obtained only when the initial
        Vidal gauge is valid.

        Parameters
        ----------
        orth_center : int, optional
            Orthogonality center in ``[0, n_sites - 1]``. ``None`` selects the
            last site.

        Returns
        -------
        TensorFormat1D
            The current format, with bonds set to ``None``.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> _ = format.canonicalize_vidal(mode='explicit')
        >>> _ = format.materialize_bonds(orth_center=0)
        >>> format.bonds is None
        True
        >>> torch.allclose(format.contract_dense(), torch.eye(2))
        True
        """
        orth_center = (self.n_sites - 1) if orth_center is None else orth_center

        if isinstance(orth_center, bool) or not isinstance(orth_center, int):
            raise TypeError('`orth_center` should be int type or None')
        if not 0 <= orth_center < self.n_sites:
            raise ValueError('`orth_center` should select a valid site')

        if self._bonds is None:
            return self

        cores = list(self._standard_cores())
        if isinstance(self._bonds, VidalGauge) and self._bonds._valid:
            powers = [(0, 1) if site < orth_center else (1, 0)
                      for site in range(len(self._bonds.spectra))]
            cores, _ = _redistribute(cores=cores,
                                     spectra=self._bonds.spectra,
                                     old_powers=self._bonds.powers,
                                     powers=powers)
            self._set_standard_cores(cores)
            return self

        for site, factor in enumerate(self._bonds.factors):
            if factor is None:
                continue
            if site < orth_center:
                cores[site + 1] = factor[..., :, None, None] * cores[site + 1]
            else:
                cores[site] = cores[site] * factor[..., None, None, :]
        self._set_standard_cores(cores)
        return self

    def absorb_bond(self, bond: int, side: str = 'left') -> 'TensorFormat1D':
        """
        Absorbs one diagonal factor into a neighboring core in-place.

        Parameters
        ----------
        bond : int
            Index of the right virtual bond of a core. The last bond closes a
            cyclic format.
        side : {"left", "right"}
            Neighbor receiving the factor. A valid Vidal gauge redistributes
            existing absorption powers; an ordinary or invalidated gauge absorbs
            its stored diagonal once.

        Returns
        -------
        TensorFormat1D
            The current format, with the selected explicit factor removed.
        """
        count = self.n_sites if self._cyclic else (self.n_sites - 1)
        if isinstance(bond, bool) or not isinstance(bond, int):
            raise TypeError('`bond` should be int type')
        if not 0 <= bond < count:
            raise ValueError('`bond` should select a valid virtual bond')
        if side not in ('left', 'right'):
            raise ValueError('`side` should be "left" or "right"')

        if self._bonds is None:
            return self

        cores = list(self._standard_cores())
        spectra = powers = None
        if isinstance(self._bonds, VidalGauge) and self._bonds._valid:
            spectra = self._bonds.spectra
            powers = list(self._bonds.powers)
            powers[bond] = (1, 0) if (side == 'left') else (0, 1)
            cores, factors = _redistribute(cores=cores,
                                           spectra=self._bonds.spectra,
                                           old_powers=self._bonds.powers,
                                           powers=powers)
        else:
            factors = list(self._bonds.factors)
            factor = factors[bond]
            if factor is not None:
                site = bond if (side == 'left') else (bond + 1) % self.n_sites
                cores[site] = cores[site] * (
                    factor[..., None, None, :] if (side == 'left')
                    else factor[..., :, None, None])
                factors[bond] = None
        self._set_standard_cores(cores, factors, spectra=spectra, powers=powers)
        return self

    def _map_tensors(self,
                     function: Callable[[torch.Tensor],torch.Tensor]
                     ) -> 'TensorFormat1D':
        """Maps stored tensors while preserving concrete container semantics."""
        result = copy(self)
        result._cores = _SafeList([function(core) for core in self._cores],
                                  result._on_cores_changed)
        if self._bonds is not None:
            result._bonds = self._bonds._map_tensors(
                function, result._on_bonds_changed)
        return result

    def _same_aux_tensors(self, other: 'TensorFormat1D') -> bool:
        """Checks whether auxiliary tensor references are unchanged."""
        return True

    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           copy: bool = False) -> 'TensorFormat1D':
        """
        Returns a device/dtype conversion, preserving the concrete format.

        PyTorch device errors propagate without a CPU fallback. Autograd is
        retained.

        Parameters
        ----------
        device : str or torch.device, optional
            Target device. ``None`` preserves the current device.
        dtype : torch.dtype, optional
            Target dtype. ``None`` preserves the current dtype. Coordinate
            grids and Schmidt spectra remain real when cores are complex.
        copy : bool
            If ``True``, copies tensors even when device and dtype are
            unchanged. If ``False``, an unchanged conversion may return
            ``self``.

        Returns
        -------
        TensorFormat
            Converted format; ``self`` when no conversion is needed and ``copy``
            is ``False``.

        Examples
        --------
        >>> format = tk.formats.TT([torch.ones(2)])
        >>> format.to() is format
        True
        >>> format.to(dtype=torch.float64).dtype
        torch.float64
        """
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type')
        if not isinstance(copy, bool):
            raise TypeError('`copy` should be bool type')

        result = self._map_tensors(lambda tensor: tensor.to(device=device,
                                                            dtype=dtype,
                                                            copy=copy))

        same_cores = all(
            new is old for new, old in zip(result._cores, self._cores))
        same_bonds = self._bonds is None or all(
            new is old for new, old in zip(result._bonds.factors,
                                           self._bonds.factors))

        if not copy and same_cores and same_bonds and self._same_aux_tensors(result):
            return self

        result.validate()
        return result

    def clone(self) -> 'TensorFormat1D':
        """
        Clones the structural tensors, preserving autograd.

        Returns
        -------
        TensorFormat
            Independent tensor storage with the same represented tensor.
        """
        return self._map_tensors(lambda tensor: tensor.clone())

    def detach(self) -> 'TensorFormat1D':
        """
        Returns a detached format sharing tensor storage.

        Returns
        -------
        TensorFormat
            Separate containers with detached tensor references. Value edits to
            shared storage affect both formats.
        """
        return self._map_tensors(lambda tensor: tensor.detach())

    def detach_(self) -> 'TensorFormat1D':
        """
        Detaches structural tensors in-place by replacing references.

        Returns
        -------
        TensorFormat
            The current format. Tensor shapes and canonical metadata are
            preserved.
        """
        self._cores = _SafeList([core.detach() for core in self._cores],
                                self._on_cores_changed)
        if self._bonds is not None:
            self._bonds = self._bonds._map_tensors(
                lambda tensor: tensor.detach(), self._on_bonds_changed)
        return self

    def conj(self) -> 'TensorFormat1D':
        """
        Returns the conjugate cores and bond factors.

        Returns
        -------
        TensorFormat1D
            Separate format with conjugated tensor references; storage may be
            shared.
        """
        return self._map_tensors(lambda tensor: tensor.conj())

    def canonicalize(self,
                     orth_center: Optional[int] = None,
                     renormalize: bool = False) -> 'TensorFormat1D':
        """
        Performs QR/RQ sweeps in-place without truncation.

        Open chains become left-isometric before the center and right-isometric
        after it. For rings, this is a local gauge relative to the stored cut,
        without a global Schmidt interpretation. Stored factors are materialized
        before sweeping.

        Parameters
        ----------
        orth_center : int, optional
            Orthogonality center in ``[0, n_sites - 1]``. ``None`` selects the
            last site.
        renormalize : bool
            Rescales intermediate factors to reduce numerical overflow or
            underflow. Their accumulated scale is divided equally among all
            cores, preserving the represented tensor's global scale.

        Returns
        -------
        TensorFormat1D
            The current format, with its selected orthogonality center recorded.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> _ = format.canonicalize(orth_center=0, renormalize=True)
        >>> torch.allclose(format.contract_dense(), torch.eye(2))
        True
        """
        orth_center = (self.n_sites - 1) if orth_center is None else orth_center

        if isinstance(orth_center, bool) or not isinstance(orth_center, int):
            raise TypeError('`orth_center` should be int type or None')
        if not 0 <= orth_center < self.n_sites:
            raise ValueError('`orth_center` should select a valid site')
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')

        cores = _canonicalize_cores(
            self._effective_cores(), orth_center, renormalize)
        self._set_standard_cores(cores)
        self._orth_center = orth_center
        return self

    def normalize(self) -> 'TensorFormat1D':
        """
        Scales the represented tensor or matrix to unit Frobenius norm in-place.

        Structural batches are normalized independently. Zero-norm batches
        raise ``ValueError``. A valid Vidal gauge is preserved when its
        rescaled tensors remain finite. Otherwise, a recorded center receives
        the scale when possible; without one, the scale is distributed among
        the cores. Autograd is preserved.

        Returns
        -------
        TensorFormat1D
            The current format with unit norm in every structural batch.
        """
        _, log_squared_norm = self._log_overlap(self)

        if torch.any(torch.isneginf(log_squared_norm)):
            raise ValueError('Cannot normalize a zero-norm format')
        if not torch.all(torch.isfinite(log_squared_norm)):
            raise ValueError('Cannot normalize a format with a non-finite norm')

        log_norm = log_squared_norm / 2

        def rescale(tensor: torch.Tensor,
                    exponent: float,
                    local_axes: int) -> torch.Tensor:
            if exponent == 0:
                return tensor
            factor = (-exponent * log_norm).exp().reshape(
                *self._batch_shape, *((1,) * local_axes))
            return tensor * factor

        # In a valid Vidal gauge, divide each spectrum by the norm and adjust
        # the cores and factors to preserve the gauge
        bonds = self._bonds
        if isinstance(bonds, VidalGauge) and bonds._valid:
            powers = list(bonds.powers)
            cores = []
            for site, core in enumerate(self._cores):
                if self.n_sites == 1:
                    exponent = 1
                else:
                    left = powers[site - 1][1] if site else 0
                    right = powers[site][0] if site < (self.n_sites - 1) else 0
                    exponent = left + right - int(0 < site < (self.n_sites - 1))
                cores.append(rescale(core, exponent,
                                     core.ndim - self._n_batches))

            factors = [None if factor is None else rescale(
                factor, 1 - left - right, 1)
                for factor, (left, right) in zip(bonds.factors, powers)]
            spectra = [rescale(spectrum, 1, 1)
                       for spectrum in bonds.spectra]
            all_tensors = (cores +
                           [factor for factor in factors if factor is not None] +
                           spectra)
            if all(torch.all(torch.isfinite(tensor)) for tensor in all_tensors):
                self._set_cores(cores, factors, spectra=spectra, powers=powers)
                return self

        # If a center is recorded, scale its core unless that overflows or
        # underflows
        center = self._orth_center
        if center is not None:
            cores = list(self._cores)
            cores[center] = rescale(cores[center], 1,
                                    cores[center].ndim - self._n_batches)
            if not torch.all(torch.isfinite(cores[center])) or \
                    torch.any((-log_norm).exp() == 0):
                center = None

        # Otherwise, spread the normalization factor across all cores
        if center is None:
            if torch.any((-log_norm / self.n_sites).exp() == 0):
                raise ValueError('Normalization scale is not representable')
            cores = [rescale(core, 1 / self.n_sites,
                             core.ndim - self._n_batches)
                     for core in self._cores]

        if not all(torch.all(torch.isfinite(core)) for core in cores):
            raise ValueError('Normalization would produce non-finite cores')

        if bonds is None:
            self._set_cores(cores, None)
        elif isinstance(bonds, VidalGauge):
            self._set_cores(cores,
                            list(bonds.factors),
                            spectra=list(bonds.spectra),
                            powers=list(bonds.powers))
            self._bonds._valid = False
        else:
            self._set_cores(cores, list(bonds.factors))
        self._orth_center = center

        return self

    def rounding(self,
                 rank: Optional[int] = None,
                 cutoff: Optional[float] = None,
                 atol: Optional[float] = None,
                 rtol: Optional[float] = None,
                 cum_percentage: Optional[float] = None,
                 renormalize: bool = False,
                 *,
                 rel_error: Optional[float] = None,
                 return_info: bool = False
                 ) -> Union['TensorFormat1D',
                            Tuple['TensorFormat1D', RoundingInfo]]:
        r"""
        Compresses bond ranks in-place with two sweeps along the cores.

        For open chains, a left-to-right QR sweep prepares the cores, followed
        by a right-to-left SVD sweep that truncates each internal bond. Rings use
        Algorithm 4 of Mickelin and Karaman, `On Algorithms for and Computing
        with the Tensor Ring Decomposition <https://arxiv.org/pdf/1807.02513>`_,
        including closure reduction. Ring ranks need not become minimal,
        especially after block-diagonal sums or products. Multiple criteria
        use the most restrictive retained rank.

        With ``rel_error`` set to :math:`\varepsilon`, rounding requests
        :math:`\lVert X-\widetilde X\rVert_F \leq \varepsilon\lVert X\rVert_F`
        for the full tensor or matrix :math:`X`, separately for each structural
        batch. The budget is divided among local cuts. If :math:`e_i` is the
        discarded Frobenius norm at cut :math:`i`, ``error_bound`` is
        :math:`B = \sqrt{c\sum_i e_i^2}`, where :math:`c` is the original
        closing-bond rank for a ring and 1 for an open chain. The factor
        :math:`\sqrt{c}` follows from
        :math:`|\operatorname{tr}(E)| \leq \sqrt{c}\lVert E\rVert_F` when
        closing the auxiliary open chain; see the proof of Algorithm 4 in
        `Mickelin and Karaman <https://arxiv.org/pdf/1807.02513>`_. This is a
        bound, not a measured reconstruction error. More restrictive
        truncation criteria can make the bound exceed the requested budget.

        Parameters
        ----------
        rank : int, optional
            Maximum number of singular values to keep.
        cutoff : float, optional
            Minimum singular value to keep. It must be finite and non-negative.
            Singular values ``<= cutoff`` are removed.
        atol : float, optional
            Absolute tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded
            while the accumulated sum of squares is ``<= atol``. It must be finite
            and non-negative.
        rtol : float, optional
            Relative tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded
            while the tail sum of squares divided by the total sum of squares is
            ``<= rtol``. It must be finite and in [0, 1].
        cum_percentage : float, optional
            Minimum fraction of squared singular-value mass to keep. Equivalent
            to setting ``rtol = 1 - cum_percentage``. It must be finite and in [0,
            1].
        renormalize : bool
            Rescales factors during the initial canonicalization to reduce
            numerical overflow or underflow, then divides their accumulated
            scale equally among all cores before truncation.
        rel_error : float, optional
            Finite non-negative target for the global relative Frobenius error.
        return_info : bool
            If ``True``, returns the format together with the operation-specific
            information record.

        Returns
        -------
        TensorFormat1D or tuple[TensorFormat1D, RoundingInfo]
            The current format, optionally with its truncation information.

        Examples
        --------
        >>> format = tk.formats.TT([
        ...     torch.diag(torch.tensor([4., 1.])), torch.eye(2)])
        >>> _, info = format.rounding(rank=1, return_info=True)
        >>> info.rank
        (1,)
        >>> format.contract_dense()
        torch.diag(torch.tensor([4., 0.]))

        Clone the original to measure the achieved global relative error
        without constructing the dense tensors:

        >>> original = tk.formats.TT([
        ...     torch.diag(torch.tensor([4., 1.])), torch.eye(2)])
        >>> rounded = original.clone().rounding(rank=1)
        >>> relative_error = original.distance(rounded) / original.norm()
        >>> round(relative_error.item(), 4)
        0.2425
        """
        _validate_truncation(rank, cutoff, atol, rtol, cum_percentage)
        for name, value in [('renormalize', renormalize),
                            ('return_info', return_info)]:
            if not isinstance(value, bool):
                raise TypeError(f'`{name}` should be bool type')
        if rel_error is not None:
            if isinstance(rel_error, bool) or not isinstance(rel_error, Real):
                raise TypeError('`rel_error` should be a real number')
            if not isfinite(rel_error) or rel_error < 0:
                raise ValueError('`rel_error` should be finite and non-negative')

        # Prepare a left-canonical chain and the per-cut error budget.
        batch_shape = self._batch_shape
        cyclic = self._cyclic
        closing = self._rank[-1] if cyclic else 1
        norm = self.norm() if rel_error is not None else None

        cores = _canonicalize_cores(
            self._effective_cores(), self.n_sites - 1, renormalize)

        cuts = len(cores) if cyclic else max(1, len(cores) - 1)
        delta = rel_error * norm / sqrt(cuts * closing) if norm is not None \
            else None # delta is the per-cut budget for the local absolute error
        records = []
        discarded_norms = []
        collect = return_info or rel_error is not None

        def split(matrix: torch.Tensor) -> Tuple[torch.Tensor,
                                                 torch.Tensor,
                                                 torch.Tensor]:
            """Truncates a scaled matrix and collects its discarded mass."""
            scale = matrix.abs().amax()
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            limit = torch.finfo(matrix.real.dtype).max

            scaled_cutoff = None if cutoff is None else min(
                cutoff / scale.item(), limit)
            scaled_atol = None if atol is None else min(
                atol / scale.item() / scale.item(), limit)
            if delta is not None:
                scaled_delta = (delta / scale).square().min().item()
                scaled_atol = min(scaled_delta, limit) if scaled_atol is None else min(
                    scaled_atol, scaled_delta)

            decomposition = truncated_svd(tensor=matrix / scale,
                                          rank=rank,
                                          cutoff=scaled_cutoff,
                                          atol=scaled_atol,
                                          rtol=rtol,
                                          cum_percentage=cum_percentage,
                                          return_info=collect)
            u, s, vh = decomposition[:3]

            if collect:
                discarded = decomposition[3].discarded_sq_norm.sqrt() * scale
                discarded_norms.append(discarded)
                records.append(discarded.square())

            return u, s * scale, vh

        # Rings also require reducing their closing bond.
        if cyclic and (len(cores) == 1):
            traced_cyclic = cores[0].diagonal(dim1=-3, dim2=-1).sum(-1)
            cores = [traced_cyclic.unsqueeze(-2).unsqueeze(-1)]
        elif cyclic:
            core = cores[-1]
            q, r = torch.linalg.qr(core.reshape(*batch_shape, -1, core.shape[-1]),
                                   mode='reduced')
            u, s, vh = split(r)
            if s.shape[-1] < core.shape[-1]:
                cores[-1] = ((q @ u) * s.unsqueeze(-2)).reshape(
                    *batch_shape, core.shape[-3], core.shape[-2], s.shape[-1])
                cores[0] = torch.einsum('...ab,...bpr->...apr', vh, cores[0])

        # Sweep back, truncating one bond at a time.
        for site in range(len(cores) - 1, 0, -1):
            core = cores[site]
            u, s, vh = split(core.reshape(*batch_shape, core.shape[-3], -1))
            cores[site] = vh.reshape(*batch_shape, s.shape[-1],
                                     core.shape[-2], core.shape[-1])
            cores[site - 1] = torch.einsum('...apb,...bc->...apc',
                                           cores[site - 1],
                                           u * s.unsqueeze(-2))

        self._set_standard_cores(cores)
        self._orth_center = None if cyclic else 0
        satisfied = None

        if collect:
            if discarded_norms:
                errors = torch.stack(discarded_norms)
                scale = errors.amax(dim=0)
                safe = torch.where(scale > 0, scale, torch.ones_like(scale))
                bound = torch.linalg.vector_norm(
                    errors / safe, dim=0) * scale * sqrt(closing)
            else:
                bound = self._cores[0].real.new_zeros(batch_shape)

            if rel_error is not None:
                satisfied = bool(torch.all(bound <= rel_error * norm +
                                           10 * torch.finfo(norm.dtype).eps * norm))
                if not satisfied:
                    warnings.warn('Truncation constraints exceed the requested global error budget',
                                  UserWarning, stacklevel=2)
            if return_info:
                return self, RoundingInfo(
                    self._rank, tuple(records), bound, satisfied)

        return self

    def contract_block(self, first: int, last: int) -> torch.Tensor:
        """
        Contracts a contiguous region with both external ranks left open.

        Parameters
        ----------
        first : int
            First site of the region, included.
        last : int
            Last site of the region, included. Should be at least ``first``.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``(*core_batch, left, *physical, right)``. Matrix
            physical axes are interleaved. Internal factors are included and
            external factors excluded.

        Examples
        --------
        >>> matrix = tk.formats.TTM([
        ...     torch.ones(2, 3, 4), torch.ones(3, 5, 6)])
        >>> matrix.contract_block(0, 1).shape
        torch.Size([1, 2, 4, 5, 6, 1])
        """
        for site in (first, last):
            if isinstance(site, bool) or not isinstance(site, int):
                raise TypeError('Block limits should be integers')
        if not 0 <= first <= last < self.n_sites:
            raise ValueError(
                'Block limits should select an ordered contiguous region')

        cores = self._standard_cores()
        result = cores[first]

        batch_shape = self._batch_shape
        left = result.shape[-3]
        physical = [result.shape[-2]]

        for site in range(first, last):
            if self._bonds is not None and self._bonds.factors[site] is not None:
                result = result * self._bonds.factors[site][..., None, None, :]

            result = result.reshape(*batch_shape, left, -1, result.shape[-1])
            result = torch.einsum('...apr,...rqb->...apqb',
                                  result, cores[site + 1])
            physical.append(cores[site + 1].shape[-2])

        if self._out_dim is not None:
            physical = [dim for pair in zip(
                self._in_dim[first:last + 1],
                self._out_dim[first:last + 1]) for dim in pair]

        return result.reshape(*batch_shape, left, *physical, cores[last].shape[-1])

    def block(self, groups: Sequence[int]) -> BlockLayout:
        """
        Contracts consecutive groups into effective sites in-place.

        Parameters
        ----------
        groups : sequence of int
            Positive group sizes whose sum is ``n_sites``. Physical dimensions
            are multiplied within each group; matrix inputs and outputs stay
            separate. Internal factors are absorbed and factors between groups
            are retained.

        Returns
        -------
        BlockLayout
            Original dimensions and group sizes. Pass it to :meth:`unblock` on
            this format or a solver result with the same effective dimensions.
            Clone the format first to retain its original structure. Quantics
            layouts must remain compatible; use
            :meth:`~tensorkrowch.formats.QTT.as_tt` /
            :meth:`~tensorkrowch.formats.QTR.as_tr` or
            :meth:`~tensorkrowch.formats.QTTM.as_ttm` /
            :meth:`~tensorkrowch.formats.QTRM.as_trm` to group arbitrary
            digits in a plain format.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> layout = format.block([2])
        >>> format.in_dim
        (4,)
        >>> _ = format.unblock(layout)
        >>> torch.allclose(format.contract_dense(), torch.eye(2))
        True
        """
        groups = tuple(groups)
        if not groups or any(isinstance(size, bool) or not isinstance(
            size, int) or (size < 1) for size in groups):
            raise ValueError('Block sizes should be positive integers')
        if sum(groups) != self.n_sites:
            raise ValueError('Block sizes should sum to the number of sites')

        layout = BlockLayout(groups, self._in_dim, self._out_dim)
        cores, in_dim, out_dim, factors = [], [], [], []
        first = 0
        for size in groups:
            last = first + (size - 1)
            block = self.contract_block(first, last)

            in_dim.append(prod(self._in_dim[first:(last + 1)]))
            if self._out_dim is not None:
                out_dim.append(prod(self._out_dim[first:(last + 1)]))
                b = self._n_batches
                order = [*range(b + 1), *range(b + 1, b + 1 + 2 * size, 2),
                         *range(b + 2, b + 1 + 2 * size, 2), block.ndim - 1]
                block = block.permute(order)

            cores.append(block.reshape(*self._batch_shape,
                                       block.shape[self._n_batches],
                                       -1,
                                       block.shape[-1]))

            if self._bonds is not None and (last < len(self._bonds.factors)):
                factors.append(self._bonds.factors[last])

            first = last + 1

        bonds = factors if factors else None
        self._set_standard_cores(cores, bonds,
                                 in_dim=tuple(in_dim),
                                 out_dim=tuple(out_dim) if out_dim else None)
        return layout

    def unblock(self, layout: BlockLayout, **kwargs) -> 'TensorFormat1D':
        """
        Restores the sites described by a blocking layout in-place.

        Parameters
        ----------
        layout : BlockLayout
            Original dimensions and group sizes returned by :meth:`block`. Its
            grouped dimensions must match the current format. It can describe
            another object, such as an initial guess used to produce a solver
            result.
        **kwargs : keyword arguments
            Options passed to :func:`split_block`: ``rank``, ``cutoff``,
            ``atol``, ``rtol``, ``cum_percentage``, ``mode`` and
            ``renormalize``. Only bonds inside groups are truncated; ranks
            between groups remain unchanged.

        Returns
        -------
        TensorFormat1D
            The current format. Without truncation the represented tensor is
            recovered up to numerical precision; original core gauges may
            differ. Invalid input leaves the current format unchanged.
        """
        if not isinstance(layout, BlockLayout) or (
            len(layout.groups) != self.n_sites):
            raise ValueError('Unblocking requires a matching BlockLayout')
        if (self._out_dim is None) != (layout.out_dim is None):
            raise ValueError(
                'Blocked vector/matrix family should match the layout')

        cores, first = [], 0
        effective_cores = self._effective_cores()
        for site, size in enumerate(layout.groups):
            block = effective_cores[site]
            inputs = layout.in_dim[first:(first + size)]
            outputs = None if layout.out_dim is None else \
                layout.out_dim[first:(first + size)]

            if (self._in_dim[site] != prod(inputs)) or (
                    outputs is not None and (
                        self._out_dim[site] != prod(outputs))):
                raise ValueError('Blocked dimensions should match the layout')

            dimensions = inputs
            if outputs is not None:
                block = block.reshape(
                    *self._batch_shape, block.shape[-3],
                    *inputs, *outputs, block.shape[-1])
                b = self._n_batches
                order = [*range(b + 1)]
                for index in range(size):
                    order.extend([b + 1 + index, b + 1 + size + index])
                order.append(block.ndim - 1)
                block = block.permute(order)
                dimensions = tuple(dim for pair in zip(inputs, outputs)
                                   for dim in pair)

            block = block.reshape(*self._batch_shape,
                                  block.shape[self._n_batches],
                                  *dimensions,
                                  block.shape[-1])
            local = split_block(block, inputs, outputs,
                                self._n_batches, **kwargs)

            # Restore the local representation with each factor absorbed once.
            for index, core in enumerate(local.cores):
                factor = local.bonds[index] if (
                    index < len(local.bonds)) else None
                cores.append(core if factor is None else \
                    core * factor[..., None, None, :])

            first += size

        self._set_standard_cores(cores,
                                 in_dim=layout.in_dim,
                                 out_dim=layout.out_dim)
        return self

    def _check_semantics(self,
                         other: 'TensorFormat1D',
                         product: bool = False) -> None:
        """Checks whether operands share their coordinate interpretation."""
        if self._quantized != other._quantized:
            raise ValueError(
                'Quantics algebra requires compatible coordinate semantics; '
                'use as_tt/as_tr/as_ttm/as_trm explicitly')

    def _prepare_binary_operands(
            self,
            other: 'TensorFormat1D'
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
                   Tuple[int, ...], bool]:
        """Prepares compatible effective cores and structural batches."""
        if not isinstance(other, TensorFormat1D):
            raise TypeError('`other` should be TensorFormat1D type')
        self._check_semantics(other)
        if self.n_sites != other.n_sites:
            raise ValueError('Formats should have the same number of sites')
        if self.device != other.device:
            raise ValueError('Formats should share device')
        if self._batch_shape and other._batch_shape and (
            self._batch_shape != other._batch_shape):
            raise ValueError(
                'Structural batches should match or one operand should '
                'be unbatched')
        if (self._in_dim != other._in_dim) or (self._out_dim != other._out_dim):
            raise ValueError(
                'Formats should have matching input and output dimensions')
        if self._out_dim is None and (self._is_row != other._is_row):
            raise ValueError(
                'Vectors should have the same row or column orientation')

        # Promote tensors and align structural batches before the local algebra.
        dtype = torch.promote_types(self.dtype, other.dtype)
        batch_shape = self._batch_shape or other._batch_shape
        cyclic = self._cyclic or other._cyclic
        
        left_cores = [core.to(dtype=dtype) for core in self._effective_cores()]
        right_cores = [core.to(dtype=dtype) for core in other._effective_cores()]

        left_cores = [core.expand(*batch_shape, *core.shape[-3:])
                      for core in left_cores]
        right_cores = [core.expand(*batch_shape, *core.shape[-3:])
                       for core in right_cores]

        return left_cores, right_cores, batch_shape, cyclic

    def _sum(self,
             other: 'TensorFormat1D',
             method: str,
             coefficient: Number) -> 'TensorFormat1D':
        """Builds an exact stacked or block-diagonal sum or difference."""
        if method not in ('stacked', 'block_diagonal'):
            raise ValueError('`method` should be "stacked" or "block_diagonal"')

        left_cores, right_cores, batch_shape, cyclic = \
            self._prepare_binary_operands(other)
        right_cores[0] = right_cores[0] * coefficient

        if len(left_cores) == 1:
            sum_core = left_cores[0].diagonal(dim1=-3, dim2=-1).sum(-1) + \
                right_cores[0].diagonal(dim1=-3, dim2=-1).sum(-1)
            cores = [sum_core.unsqueeze(-2).unsqueeze(-1)]
        else:
            # Stacked sums share a closing space at the first and last sites.
            if (method == 'stacked') or not cyclic:
                closing = max(left_cores[0].shape[-3], right_cores[0].shape[-3])
                for group in (left_cores, right_cores):
                    first, last = group[0], group[-1]

                    padded = first.new_zeros(*batch_shape,
                                             closing, *first.shape[-2:])
                    padded[..., :first.shape[-3], :, :] = first
                    group[0] = padded

                    padded = last.new_zeros(*batch_shape,
                                            *last.shape[-3:-1], closing)
                    padded[..., :last.shape[-1]] = last
                    group[-1] = padded

            cores = []
            for site, (left, right) in enumerate(zip(left_cores, right_cores)):
                if ((method == 'stacked') or not cyclic) and (site == 0):
                    core = torch.cat((left, right), dim=-1)
                elif ((method == 'stacked') or not cyclic) and (
                    site == len(left_cores) - 1):
                    core = torch.cat((left, right), dim=-3)
                else:
                    core = left.new_zeros(
                        *batch_shape, left.shape[-3] + right.shape[-3],
                        left.shape[-2], left.shape[-1] + right.shape[-1])
                    core[..., :left.shape[-3], :, :left.shape[-1]] = left
                    core[..., left.shape[-3]:, :, left.shape[-1]:] = right
                cores.append(core)

        return self._new_from_standard_cores(
            cores, self._in_dim, self._out_dim,
            len(batch_shape), cyclic, other=other)

    def add(self,
            other: 'TensorFormat1D',
            method: str = 'stacked') -> 'TensorFormat1D':
        """
        Returns the exact sum of compatible formats.

        For two or more sites, the construction adds the operands' internal
        bond ranks. For cyclic results, ``method='block_diagonal'`` also adds
        the closing ranks, while ``method='stacked'`` uses their maximum.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with compatible local input/output dimensions,
            structural batches and device.
        method : {"stacked", "block_diagonal"}
            Cyclic sum construction. Stacked first and last cores favor
            subsequent compression; ``"block_diagonal"`` uses the usual
            construction at every site. For open chains both choices use the
            ordinary first/last-core construction.

        Returns
        -------
        TensorFormat1D
            New format without implicit truncation. Either cyclic operand
            produces a cyclic result. Inputs are unchanged.

        Examples
        --------
        >>> left = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> right = tk.formats.TT([2 * torch.eye(2), torch.eye(2)])
        >>> result = left.add(right)
        >>> result.rank
        [4]
        >>> torch.allclose(result.contract_dense(), 3 * torch.eye(2))
        True
        """
        return self._sum(other, method, 1)

    def sub(self,
            other: 'TensorFormat1D',
            method: str = 'stacked') -> 'TensorFormat1D':
        """
        Returns the exact difference of compatible formats.

        For two or more sites, the construction adds the operands' internal
        bond ranks. For cyclic results, ``method='block_diagonal'`` also adds
        the closing ranks, while ``method='stacked'`` uses their maximum.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with compatible local input/output dimensions,
            structural batches and device.
        method : {"stacked", "block_diagonal"}
            Cyclic sum construction. Stacked first and last cores favor
            subsequent compression; ``"block_diagonal"`` uses the usual
            construction at every site. For open chains both choices use the
            ordinary first/last-core construction.

        Returns
        -------
        TensorFormat1D
            New format without implicit truncation. Either cyclic operand
            produces a cyclic result. Inputs are unchanged.

        Examples
        --------
        >>> left = tk.formats.TT([2 * torch.eye(2), torch.eye(2)])
        >>> right = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> result = left.sub(right)
        >>> result.rank
        [4]
        >>> torch.allclose(result.contract_dense(), torch.eye(2))
        True
        """
        return self._sum(other, method, -1)

    def hadamard(self, other: 'TensorFormat1D') -> 'TensorFormat1D':
        """
        Returns an exact element-wise product.

        The construction multiplies the operands' bond ranks at each bond,
        including the closing bond for cyclic results.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with compatible local input/output dimensions,
            structural batches and device.

        Returns
        -------
        TensorFormat1D
            New format with product bond ranks, before any explicit rounding.
            A cyclic operand produces a cyclic result.

        Examples
        --------
        >>> left = tk.formats.TT([
        ...     torch.tensor([[1., 2.], [3., 4.]]), torch.eye(2)])
        >>> right = tk.formats.TT([
        ...     torch.tensor([[2., 3.], [4., 5.]]), torch.eye(2)])
        >>> product = left.hadamard(right)
        >>> product.rank
        [4]
        >>> product.contract_dense()
        torch.tensor([[2., 6.], [12., 20.]])
        """
        left_cores, right_cores, batch_shape, cyclic = \
            self._prepare_binary_operands(other)

        cores = []
        for left, right in zip(left_cores, right_cores):
            core = torch.einsum('...lpr,...aps->...laprs', left, right)
            cores.append(core.reshape(*batch_shape,
                                      left.shape[-3] * right.shape[-3],
                                      left.shape[-2],
                                      left.shape[-1] * right.shape[-1]))

        return self._new_from_standard_cores(
            cores, self._in_dim, self._out_dim,
            len(batch_shape), cyclic, other=other)

    def __add__(self, other: 'TensorFormat1D') -> 'TensorFormat1D':
        """Returns the exact sum with another format."""
        return self.add(other)

    def __sub__(self, other: 'TensorFormat1D') -> 'TensorFormat1D':
        """Returns the exact difference with another format."""
        return self.sub(other)

    def __mul__(self,
                other: Union['TensorFormat1D', torch.Tensor, Number]
                ) -> 'TensorFormat1D':
        """Returns a Hadamard product or scalar-scaled format."""
        if isinstance(other, TensorFormat1D):
            return self.hadamard(other)

        if isinstance(other, torch.Tensor):
            if other.ndim != 0:
                raise ValueError('The scaling tensor should be scalar')
            if other.device != self.device:
                raise ValueError(
                    'The scaling tensor should share the format device')
        elif isinstance(other, bool) or not isinstance(other, Number):
            raise TypeError(
                'The scaling `other` should be a number or scalar tensor')

        cores = self._effective_cores()
        cores[0] = cores[0] * other
        dtype = cores[0].dtype
        cores = [core.to(dtype=dtype) for core in cores]

        return self._new_from_standard_cores(
            cores, self._in_dim, self._out_dim, self._n_batches, self._cyclic)

    def __rmul__(self,
                 other: Union['TensorFormat1D', torch.Tensor, Number]
                 ) -> 'TensorFormat1D':
        """Returns a scalar-scaled format or Hadamard product."""
        return self * other

    def __neg__(self) -> 'TensorFormat1D':
        """Returns the format with its represented tensor negated."""
        return self * -1

    def _product_cores(self,
                       other: 'TensorFormat1D'
                       ) -> Tuple[List[torch.Tensor], int, bool]:
        """Contracts operator cores with matching input and output spaces."""
        if self.n_sites != other.n_sites:
            raise ValueError('Formats should have the same number of sites')
        if self.device != other.device:
            raise ValueError('Formats should share device')
        if self._batch_shape and other._batch_shape and (
            self._batch_shape != other._batch_shape):
            raise ValueError(
                'Structural batches should match or one operand should '
                'be unbatched')

        dtype = torch.promote_types(self.dtype, other.dtype)
        batch_shape = self._batch_shape or other._batch_shape
        left_cores = self._operator_cores()
        right_cores = other._operator_cores()
        if any(left.shape[-3] != right.shape[-1]
               for left, right in zip(left_cores, right_cores)):
            raise ValueError('Contracted local dimensions should match')

        # The same contraction covers matrix products and vector outer products.
        cores = []
        for left, right in zip(left_cores, right_cores):
            left = left.to(dtype=dtype).expand(*batch_shape, *left.shape[-4:])
            right = right.to(dtype=dtype).expand(*batch_shape, *right.shape[-4:])
            core = torch.einsum('...liro,...ajbi->...lajorb', left, right)
            cores.append(core.reshape(
                *batch_shape, left.shape[-4] * right.shape[-4],
                right.shape[-3] * left.shape[-1],
                left.shape[-2] * right.shape[-2]))

        return cores, len(batch_shape), self._cyclic or other._cyclic

    def contract_dense(self) -> torch.Tensor:
        """
        Contracts the full represented tensor explicitly.

        Returns
        -------
        torch.Tensor
            Dense tensor with shape ``(*core_batch, *in_dim)`` for vectors and
            interleaved ``in_dim`` and ``out_dim`` axes for matrices. Intended
            for small tensors, not large-grid evaluation.
        """
        cores = self._effective_cores()
        closing = cores[0].shape[-3]
        result = cores[0]
        dimensions = [result.shape[-2]]
        for core in cores[1:]:
            result = result.reshape(*self._batch_shape,
                                    closing, -1, result.shape[-1])
            result = torch.einsum('...apr,...rqb->...apqb', result, core)
            dimensions.append(core.shape[-2])

        result = result.reshape(*self._batch_shape, closing,
                                *dimensions, cores[-1].shape[-1])
        result = result.diagonal(dim1=self._n_batches, dim2=-1).sum(-1)
        if self._out_dim is not None:
            dimensions = [dim for pair in zip(
                self._in_dim, self._out_dim) for dim in pair]
            result = result.reshape(*self._batch_shape, *dimensions)

        return result

    @staticmethod
    def _contract_open_chain(
            matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts an open chain of matrices along adjacent ranks."""
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result

    @abstractmethod
    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts local matrices according to the open or cyclic topology."""

    def _check_overlap_compatibility(
            self, other: 'TensorFormat1D') -> None:
        """Validates topology, shapes and runtime for an overlap."""
        if not isinstance(other, TensorFormat1D):
            raise TypeError('`other` should be TensorFormat1D type')
        if self._family != other._family:
            raise ValueError('The format families are incompatible')
        if len(self._cores) != len(other._cores):
            raise ValueError('Formats should have the same number of sites')
        if self._in_dim != other._in_dim:
            raise ValueError('Formats should have matching input dimensions')
        if self._out_dim != other._out_dim:
            raise ValueError('Formats should have matching output dimensions')
        if self._batch_shape != other._batch_shape:
            raise ValueError('Formats should have matching batch shapes')
        if self.device != other.device:
            raise ValueError('Formats should be on the same device')

    def _log_overlap(
            self, other: 'TensorFormat1D') -> Tuple[torch.Tensor,
                                                    torch.Tensor]:
        """Returns overlap phase and log-magnitude using scaled environments."""
        self._check_overlap_compatibility(other)
        dtype = torch.promote_types(self.dtype, other.dtype)
        self_cores = self._effective_cores()
        other_cores = self_cores if other is self else other._effective_cores()

        self_eye = torch.eye(
            self_cores[0].shape[-3], device=self.device, dtype=dtype)
        other_eye = torch.eye(
            other_cores[0].shape[-3], device=self.device, dtype=dtype)
        environment = torch.einsum('ai,bj->abij', self_eye, other_eye)
        if self._n_batches:
            environment = environment.reshape(
                *((1,) * self._n_batches), *environment.shape)
            environment = environment.expand(
                *self._batch_shape, *environment.shape[self._n_batches:])

        real_dtype = torch.empty((), dtype=dtype).real.dtype
        log_scale = torch.zeros(
            self._batch_shape, device=self.device, dtype=real_dtype)

        for self_core, other_core in zip(self_cores, other_cores):
            self_core = self_core.to(dtype=dtype)
            self_scale = self_core.abs().amax(dim=(-3, -2, -1))
            self_scale = torch.where(self_scale > 0,
                                     self_scale,
                                     torch.ones_like(self_scale))
            self_core = self_core / self_scale[..., None, None, None]
            if other is self:
                other_scale, other_core = self_scale, self_core
            else:
                other_core = other_core.to(dtype=dtype)
                other_scale = other_core.abs().amax(dim=(-3, -2, -1))
                other_scale = torch.where(other_scale > 0,
                                          other_scale,
                                          torch.ones_like(other_scale))
                other_core = other_core / other_scale[..., None, None, None]
            log_scale = log_scale + self_scale.log() + other_scale.log()

            environment = torch.einsum('...xyab,...apr->...xybpr',
                                       environment,
                                       self_core.conj())
            environment = torch.einsum('...xybpr,...bps->...xyrs',
                                       environment,
                                       other_core)

            scale = torch.linalg.vector_norm(
                environment, dim=(-4, -3, -2, -1))
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            environment = environment / safe_scale[..., None, None, None, None]
            log_scale = log_scale + safe_scale.log()

        overlap = torch.einsum('...ijij->...', environment)
        magnitude = overlap.abs()
        nonzero = magnitude > 0
        safe_magnitude = torch.where(nonzero,
                                     magnitude,
                                     torch.ones_like(magnitude))
        phase = torch.where(nonzero,
                            overlap / safe_magnitude,
                            torch.zeros_like(overlap))
        log_magnitude = torch.where(nonzero,
                                    safe_magnitude.log() + log_scale,
                                    torch.full_like(log_scale, -torch.inf))
        return phase, log_magnitude

    def inner(self, other: 'TensorFormat1D') -> torch.Tensor:
        """
        Contracts the conjugate of ``self`` with another format.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with matching input/output dimensions and structural
            batch shape, on the same device. Dtypes may be promoted; boundary
            topologies may differ.

        Returns
        -------
        torch.Tensor
            Overlap tensor with shape ``core_batch``, preserving complex phase.
        """
        phase, log_magnitude = self._log_overlap(other)
        return phase * log_magnitude.exp()

    def norm(self) -> torch.Tensor:
        """
        Returns the Frobenius norm using scaled double-layer contractions.

        Returns
        -------
        torch.Tensor
            Real tensor with shape ``core_batch``, or a scalar for an unbatched
            format.
        """
        _, log_squared_norm = self._log_overlap(self)
        return torch.exp(log_squared_norm / 2)

    def distance(self, other: 'TensorFormat1D') -> torch.Tensor:
        """
        Returns the absolute Frobenius distance to another format.

        The exact format difference is QR-canonicalized before taking its norm
        to reduce cancellation when the operands are close. The inputs are
        unchanged; the temporary difference may have larger bond ranks.
        Structural batches are resolved independently. Divide by a reference
        format's :meth:`norm` for a relative error.

        Parameters
        ----------
        other : TensorFormat1D
            Compatible format to compare with this one.

        Returns
        -------
        torch.Tensor
            Absolute Frobenius distance, with shape ``core_batch``.
        """
        if other is self:
            return self._cores[0].real.new_zeros(self._batch_shape)
        difference = self - other
        difference.canonicalize(orth_center=difference.n_sites - 1,
                                renormalize=True)
        return difference.norm()

    def normalized_overlap(
            self, other: 'TensorFormat1D') -> torch.Tensor:
        """
        Returns ``<self, other> / (||self|| ||other||)``, preserving phase.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with matching input/output dimensions and structural
            batch shape, on the same device. Dtypes may be promoted; boundary
            topologies may differ.

        Returns
        -------
        torch.Tensor
            Normalized overlap with shape ``core_batch``. Zero-norm operands raise
            ``ValueError``.
        """
        phase, log_overlap = self._log_overlap(other)
        _, log_self = self._log_overlap(self)
        _, log_other = other._log_overlap(other)

        if torch.any(torch.isneginf(log_self)) or \
                torch.any(torch.isneginf(log_other)):
            raise ValueError(
                'Normalized overlap is undefined for a zero-norm format')

        log_denominator = (log_self + log_other) / 2
        return phase * torch.exp(log_overlap - log_denominator)

    def fidelity(self, other: 'TensorFormat1D') -> torch.Tensor:
        """
        Returns the squared magnitude of :meth:`normalized_overlap`.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with matching input/output dimensions and structural
            batch shape, on the same device. Dtypes may be promoted; boundary
            topologies may differ.

        Returns
        -------
        torch.Tensor
            Real tensor with shape ``core_batch``. Zero-norm operands raise
            ``ValueError``.
        """
        return self.normalized_overlap(other).abs().square()

    def _prepare_data(
            self,
            data: EvaluationData,
            dimensions: Sequence[int],
            same_dim: bool,
            n_batches: int
        ) -> Tuple[List[torch.Tensor], bool, Tuple[int, ...]]:
        """Validates and prepares discrete indices or embedded vectors by site."""
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        n_sites = len(dimensions)
        if isinstance(data, torch.Tensor):
            if data.ndim == (n_batches + 1):
                if data.shape[-1] != n_sites:
                    raise ValueError(
                        'The last dimension of discrete inputs should equal '
                        'the number of sites')
                if data.dtype not in _INTEGER_DTYPES:
                    raise TypeError(
                        'Discrete inputs should have an integer dtype')
                site_data = list(data.to(device=self.device).unbind(-1))
                discrete = True
            elif data.ndim == (n_batches + 2):
                if not same_dim:
                    raise ValueError(
                        'Embedded inputs should be provided as a sequence when '
                        'site dimensions differ')
                if data.shape[-2:] != (n_sites, dimensions[0]):
                    raise ValueError(
                        'Embedded inputs should end in `(n_sites, site_dim)`')
                target_dtype = torch.promote_types(self.dtype, data.dtype)
                site_data = list(
                    data.to(device=self.device, dtype=target_dtype).unbind(-2))
                discrete = False
            else:
                raise ValueError(
                    '`data` has an incompatible number of batch dimensions')
        else:
            try:
                site_data = list(data)
            except TypeError as exc:
                raise TypeError(
                    '`data` should be a tensor or a sequence of tensors') \
                    from exc
            if len(site_data) != n_sites:
                raise ValueError(
                    '`data` should contain one tensor per site')
            if not all(isinstance(item, torch.Tensor)
                       for item in site_data):
                raise TypeError('Every site input should be a torch.Tensor')

            if all(item.ndim == n_batches for item in site_data):
                if not all(item.dtype in _INTEGER_DTYPES
                           for item in site_data):
                    raise TypeError(
                        'Discrete inputs should have an integer dtype')
                discrete = True
            elif all(item.ndim == (n_batches + 1)
                     for item in site_data):
                for item, site_dim in zip(site_data, dimensions):
                    if item.shape[-1] != site_dim:
                        raise ValueError(
                            'The last dimension of each embedded input should '
                            'match its site dimension')
                discrete = False
            else:
                raise ValueError(
                    'Site inputs should all be discrete indices or embedded '
                    'vectors')

            target_dtype = None
            if not discrete:
                target_dtype = self.dtype
                for item in site_data:
                    target_dtype = torch.promote_types(target_dtype, item.dtype)
            site_data = [item.to(device=self.device, dtype=target_dtype)
                         for item in site_data]

        batch_shape = tuple(site_data[0].shape[:n_batches])
        if any(tuple(item.shape[:n_batches]) != batch_shape
               for item in site_data[1:]):
            raise ValueError('All site inputs should have the same batch shape')

        if discrete:
            for indices, site_dim in zip(site_data, dimensions):
                if torch.any(indices < 0) or torch.any(indices >= site_dim):
                    raise ValueError(
                        'Discrete indices should lie in each site dimension')

        return site_data, discrete, batch_shape


class _VectorFormat1D(TensorFormat1D):
    """Shared raw-tensor vector operations."""

    _family = 'vector'
    _is_row = False

    @property
    def is_row(self) -> bool:
        """Whether this vector has row orientation; otherwise it is a column."""
        return self._is_row

    def _standard_cores(self) -> List[torch.Tensor]:
        """Adds unit boundary axes to open cores without absorbing bond factors."""
        cores = list(self._cores)
        if not self._cyclic:
            cores[0] = cores[0].unsqueeze(self._n_batches)
            cores[-1] = cores[-1].unsqueeze(-1)
        return cores

    def _operator_cores(self) -> List[torch.Tensor]:
        """Returns effective operator cores according to the vector orientation."""
        cores = self._effective_cores()
        if self._is_row:
            return [core.unsqueeze(-1) for core in cores]
        return [core.transpose(-2, -1).unsqueeze(-3) for core in cores]

    def _new_from_standard_cores(self,
                                 cores: Sequence[torch.Tensor],
                                 in_dim: Sequence[int],
                                 out_dim: Optional[Sequence[int]],
                                 n_batches: int,
                                 cyclic: bool,
                                 other: Optional['TensorFormat1D'] = None,
                                 product: bool = False) -> 'TensorFormat1D':
        """Preserves vector orientation when constructing an algebra result."""
        result = super()._new_from_standard_cores(
            cores, in_dim, out_dim, n_batches, cyclic,
            other=other, product=product)
        if not product:
            result._is_row = self._is_row
        return result

    def _new_outer_product(self,
                           cores: Sequence[torch.Tensor],
                           other: '_VectorFormat1D',
                           n_batches: int,
                           cyclic: bool) -> '_MatrixFormat1D':
        """Builds the operator represented by a column-row outer product."""
        return _from_standard_cores(
            cores, other._in_dim, self._in_dim, n_batches, cyclic)

    def transpose(self) -> '_VectorFormat1D':
        """
        Changes row/column orientation without conjugating coefficients.

        The result keeps the concrete format class and its methods. Its core
        and bond containers are independent, sharing tensor storage. Core
        shapes and coefficient evaluation are unchanged. Operator core views
        and products respect the selected orientation.

        Returns
        -------
        TT or TR
            Row for a column input, or column for a row input, including the
            corresponding Quantics class when applicable.

        Examples
        --------
        >>> x = tk.formats.TT([torch.tensor([1 + 2j, 3 - 4j])])
        >>> x.is_row
        False
        >>> row = x.transpose()
        >>> row.is_row
        True
        >>> row.contract_dense()
        tensor([1.+2.j, 3.-4.j])
        """
        result = self._map_tensors(lambda tensor: tensor)
        result._is_row = not self._is_row
        return result

    def adjoint(self) -> '_VectorFormat1D':
        """
        Changes row/column orientation and conjugates cores and bond factors.

        Returns
        -------
        TT or TR
            Conjugate row for a column input, or conjugate column for a
            row input. Containers are independent and tensors may share storage.
            The concrete format class and its methods are preserved.

        Examples
        --------
        >>> x = tk.formats.TT([torch.tensor([1 + 2j, 3 - 4j])])
        >>> x.is_row
        False
        >>> row = x.adjoint()
        >>> row.is_row
        True
        >>> row.contract_dense()
        tensor([1.-2.j, 3.+4.j])
        """
        result = self.conj()
        result._is_row = not self._is_row
        return result

    @property
    def T(self) -> '_VectorFormat1D':
        """Vector transpose, changing orientation without conjugation."""
        return self.transpose()

    @property
    def H(self) -> '_VectorFormat1D':
        """Vector adjoint, changing orientation and conjugating coefficients."""
        return self.adjoint()

    def matmul(self,
               other: Union['_VectorFormat1D', '_MatrixFormat1D']
               ) -> Union[torch.Tensor, 'TensorFormat1D']:
        """
        Contracts vectors and operators according to row/column orientation.

        Parameters
        ----------
        other : TT, TR, TTM or TRM
            A row can multiply a column or an operator; a column can multiply
            a row. Use ``T`` or ``H`` to change a vector's orientation. Two
            vectors should have the same number of sites; outer products allow
            different local dimensions.

        Returns
        -------
        torch.Tensor, TT, TR, TTM or TRM
            Row-column products return scalar overlaps, resolved by structural
            batch. Column-row products return operators with the row's inputs
            and the column's outputs. Row-operator products return rows. A
            cyclic operand gives a cyclic format. Two columns or two rows
            cannot multiply directly.

        Examples
        --------
        >>> x = tk.formats.TT([torch.tensor([1., 2.])])
        >>> inner = x.H.matmul(x)
        >>> inner
        tensor(5.)
        >>> torch.equal(x.H.matmul(x), x.H @ x)
        True
        
        >>> outer = x.matmul(x.H)
        >>> outer.contract_dense()
        tensor([[1., 2.],
                [2., 4.]])
        >>> torch.equal(outer.contract_dense(), (x @ x.H).contract_dense())
        True
        """
        if isinstance(other, _VectorFormat1D):
            if self._is_row == other._is_row:
                raise TypeError(
                    'Vector products require a row and a column; use T or H')
            if self._is_row:
                self._check_semantics(other)
                return self.conj().inner(other)
            TensorFormat1D._check_semantics(self, other)
            cores, n_batches, cyclic = self._product_cores(other)
            return self._new_outer_product(cores, other, n_batches, cyclic)

        if isinstance(other, _MatrixFormat1D) and self._is_row:
            return (other.T @ self.T).T

        raise TypeError(
            'A column can multiply a row; a row can multiply a '
            'column or operator')

    def __matmul__(self,
                   other: Union['_VectorFormat1D', '_MatrixFormat1D']
                   ) -> Union[torch.Tensor, 'TensorFormat1D']:
        """Calls :meth:`matmul` with the same operand."""
        return self.matmul(other)

    def apply(self, other: '_MatrixFormat1D') -> '_VectorFormat1D':
        """
        Applies an operator from the right without conjugating this vector.

        Parameters
        ----------
        other : TTM or TRM
            Operator whose output dimensions match this vector's dimensions.

        Returns
        -------
        TT or TR
            Vector with the original orientation. Columns contain the
            coefficients of ``(self.T @ other).T``; rows return ``self @ other``.
        """
        if not isinstance(other, _MatrixFormat1D):
            raise TypeError('`other` should be a matrix format')
        return self @ other if self._is_row else (self.T @ other).T

    def _local_matrices(
            self,
            site_data: Sequence[torch.Tensor],
            discrete: bool,
            data_batch_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Contracts TT/TR cores with one input at each site."""
        matrices = []
        for core, site_value in zip(self._effective_cores(), site_data):
            left_rank, site_in_dim, right_rank = core.shape[-3:]
            core = core.reshape(-1, left_rank, site_in_dim, right_rank)

            if discrete:
                indices = site_value.reshape(-1).to(torch.long)
                matrix = core[:, :, indices, :].permute(0, 2, 1, 3)
            else:
                vectors = site_value.reshape(-1, site_in_dim)
                matrix = torch.einsum('dp,clpr->cdlr',
                                      vectors, core.to(dtype=vectors.dtype))

            matrices.append(matrix.reshape(*self._batch_shape,
                                           *data_batch_shape,
                                           left_rank,
                                           right_rank))
        return matrices

    def evaluate(self,
                 data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        """
        Evaluates integer configurations or local embedded feature vectors.

        Parameters
        ----------
        data : torch.Tensor or sequence of torch.Tensor
            Integer configurations of shape ``(*data_batch, n_sites)``, or
            embedded feature vectors of shape ``(*data_batch, n_sites, in_dim)``
            for uniform dimensions. For heterogeneous dimensions, pass one
            tensor per site. Each site tensor has shape ``(*data_batch,)`` for
            indices or ``(*data_batch, in_dim[site])`` for embeddings.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *data_batch)``. Structural
            batches and data batches remain independent.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> format.evaluate(torch.tensor([[0, 0], [0, 1]])).tolist()
        [1.0, 0.0]
        >>> embeddings = torch.ones(1, 2, 2)
        >>> format.evaluate(embeddings).tolist()
        [2.0]
        """
        site_data, discrete, data_batch_shape = self._prepare_data(
            data, self._in_dim, self._same_in_dim, n_batches)
        matrices = self._local_matrices(
            site_data, discrete, data_batch_shape)
        return self._contract_local_matrices(matrices)

    def __call__(self,
                 data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        """Calls :meth:`evaluate` with the same input conventions."""
        return self.evaluate(data, n_batches=n_batches)

    def error(self,
              function: Callable[..., torch.Tensor],
              samples: EvaluationData,
              data: Optional[EvaluationData] = None,
              n_batches: int = 1,
              **kwargs) -> SampleError:
        """
        Measures absolute and relative errors on a sample set.

        Parameters
        ----------
        function : callable
            Target callable invoked as ``function(samples, **kwargs)``,
            receiving a tensor for packed samples or a tuple of site tensors
            for sequence samples. The callable returns one scalar per sample,
            with shape ``data_batch``. When the cores store a structural batch
            of formats, each format is compared with the same target values.
            For example, four formats evaluated on ten samples produce
            approximations of shape ``(4, 10)``, compared with target values
            of shape ``(10,)``.
        samples : torch.Tensor or sequence of torch.Tensor
            Inputs to ``function``. A single tensor has shape
            ``(*data_batch, n_sites, *feature_shape)``; scalar inputs have no
            feature axes. Alternatively, pass one tensor per site, each with
            shape ``(*data_batch, *feature_shape)``. All tensors share the same
            leading ``n_batches`` axes, but site shapes may differ.
            :class:`~tensorkrowch.formats.QTT` and
            :class:`~tensorkrowch.formats.QTR` require scalar inputs at each
            site, without feature axes.
        data : torch.Tensor or sequence of torch.Tensor, optional
            Inputs used to evaluate the format. ``None`` evaluates ``samples``
            directly; supply embedded inputs when target coordinates differ
            from format configurations or require another feature embedding.
            For generic formats with explicit ``data``, trailing sample axes
            are interpreted only by ``function``.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.
        **kwargs : keyword arguments
            Additional arguments passed only to the target callable.

        Returns
        -------
        SampleError
            Scalar error record retaining autograd. Returns one global error
            over all samples and all formats in the structural batch. The
            absolute error is the norm of all their differences from the target;
            the target norm includes one copy of the target values per format.
            Relative error is zero when both norms vanish and infinity when
            only the target norm vanishes.
        """
        if not callable(function):
            raise TypeError('`function` should be callable')
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        if isinstance(samples, torch.Tensor):
            if samples.ndim < (n_batches + 1) or (
                    self._quantized and (samples.ndim != (n_batches + 1))):
                raise ValueError(
                    '`samples` has an incompatible number of batch dimensions')
            samples = samples.to(device=self.device)
            data_batch_shape = tuple(samples.shape[:n_batches])
        else:
            if isinstance(samples, (str, bytes)):
                raise TypeError(
                    '`samples` should be a tensor or a sequence of tensors')
            try:
                samples = tuple(samples)
            except TypeError as exc:
                raise TypeError(
                    '`samples` should be a tensor or a sequence of tensors') \
                        from exc
            if not samples:
                raise ValueError('`samples` should contain at least one site')
            if not all(isinstance(item, torch.Tensor) for item in samples):
                raise TypeError('Every site sample should be a torch.Tensor')
            if any(item.ndim < n_batches or (
                    self._quantized and (item.ndim != n_batches))
                   for item in samples):
                raise ValueError(
                    '`samples` has an incompatible number of batch dimensions')
            data_batch_shape = tuple(samples[0].shape[:n_batches])
            if any(tuple(item.shape[:n_batches]) != data_batch_shape
                   for item in samples[1:]):
                raise ValueError(
                    'All site samples should have the same batch shape')
            samples = tuple(item.to(device=self.device) for item in samples)

        approximation = self.evaluate(samples if data is None else data,
                                      n_batches=n_batches)
        if approximation.shape != (*self._batch_shape, *data_batch_shape):
            raise ValueError(
                '`data` and `samples` should have matching batch shapes')

        target = function(samples, **kwargs)
        if not isinstance(target, torch.Tensor):
            raise TypeError('`function` should return a torch.Tensor')

        dtype = torch.promote_types(approximation.dtype, target.dtype)
        approximation = approximation.to(dtype=dtype)
        target = target.to(device=approximation.device, dtype=dtype)

        target = target.reshape(data_batch_shape)
        if self._n_batches:
            target = target.reshape(
                *((1,) * self._n_batches), *target.shape)
            target = target.expand(*self._batch_shape,
                                   *target.shape[self._n_batches:])

        absolute = torch.linalg.vector_norm(approximation - target)
        denominator = torch.linalg.vector_norm(target)
        if denominator > 0:
            relative = absolute / denominator
        elif absolute == 0:
            relative = torch.zeros_like(absolute)
        else:
            relative = torch.full_like(absolute, torch.inf)

        size = int(torch.Size(data_batch_shape).numel())
        return SampleError(kind='samples',
                           absolute=absolute,
                           relative=relative,
                           size=size,
                           denominator=denominator)

    def to_mps(self,
               parameterized: bool = False,
               **kwargs) -> Union['MPS', 'MPSData']:
        """
        Builds an open or periodic :class:`~tensorkrowch.models.MPS` or
        :class:`~tensorkrowch.models.MPSData` from effective cores.

        :class:`~tensorkrowch.formats.TT` produces open boundaries (``'obc'``);
        :class:`~tensorkrowch.formats.TR` produces periodic boundaries
        (``'pbc'``). Batched vectors produce :class:`~tensorkrowch.models.MPSData`
        and reject ``parameterized=True``.

        Parameters
        ----------
        parameterized : bool
            Whether the constructed model uses trainable parameter nodes.
            Inputs are not detached implicitly.
        **kwargs : keyword arguments
            Additional model constructor options. Tensor cores and boundary are
            supplied by the adapter.

        Returns
        -------
        :class:`~tensorkrowch.models.MPS` or :class:`~tensorkrowch.models.MPSData`
            New graph model. Stored factors are materialized in temporary
            tensors; the source format is unchanged.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> model = format.to_mps()
        >>> restored = tk.formats.TT.from_mps(model)
        >>> torch.allclose(restored.contract_dense(), format.contract_dense())
        True
        """
        from tensorkrowch.models import MPS, MPSData

        if not isinstance(parameterized, bool):
            raise TypeError('`parameterized` should be bool type')

        cores = _restore_cores(self._effective_cores(), self._in_dim,
                               self._out_dim, self._n_batches, self._cyclic)
        if self._n_batches:
            if parameterized:
                raise ValueError(
                    'MPSData does not expose parameterized model cores')
            return MPSData(tensors=cores, n_batches=self._n_batches, **kwargs)
        return MPS(tensors=cores, parameterized=parameterized, **kwargs)

    @classmethod
    def from_mps(cls, model: Union['MPS', 'MPSData'], **kwargs) -> '_VectorFormat1D':
        """
        Collects effective open or periodic :class:`~tensorkrowch.models.MPS`
        or :class:`~tensorkrowch.models.MPSData` tensors.

        Parameters
        ----------
        model : MPS or MPSData
            Source model: :class:`~tensorkrowch.formats.TT` requires open
            boundaries (``'obc'``), and :class:`~tensorkrowch.formats.TR`
            requires periodic boundaries (``'pbc'``). For open models,
            ``model.tensors`` already includes contractions with the end nodes.
        **kwargs : keyword arguments
            Additional options for the concrete format constructor, such as
            Quantics metadata on a subclass.

        Returns
        -------
        :class:`~tensorkrowch.formats.TT` or :class:`~tensorkrowch.formats.TR`
            Format sharing the effective tensor storage. Graph nodes and fit
            metrics are not retained.
        """
        from tensorkrowch.models import MPS, MPSData

        if not isinstance(model, (MPS, MPSData)):
            raise TypeError('`model` should be MPS or MPSData type')
        boundary = 'pbc' if cls._cyclic else 'obc'
        if model.boundary != boundary:
            raise ValueError(f'This adapter requires {boundary} boundaries')

        n_batches = model.n_batches if isinstance(model, MPSData) else 0
        return cls(model.tensors, n_batches=n_batches, **kwargs)


class _MatrixFormat1D(TensorFormat1D):
    """Shared raw-tensor matrix operations."""

    _family = 'matrix'

    def _standard_cores(self) -> List[torch.Tensor]:
        """Adds unit boundary axes and combines matrix input/output axes."""
        cores = list(self._cores)
        if not self._cyclic:
            cores[0] = cores[0].unsqueeze(self._n_batches)
            cores[-1] = cores[-1].unsqueeze(-2)
        return [core.movedim(-1, -2).flatten(-3, -2) for core in cores]

    def _operator_cores(self) -> List[torch.Tensor]:
        """Returns effective cores with separate input and output axes."""
        cores = list(self._cores)
        if not self._cyclic:
            cores[0] = cores[0].unsqueeze(self._n_batches)
            cores[-1] = cores[-1].unsqueeze(-2)

        if self._bonds is not None:
            for site, factor in enumerate(self._bonds.factors):
                if factor is not None:
                    cores[site] = cores[site] * factor[..., None, :, None]
        return cores

    def transpose(self) -> '_MatrixFormat1D':
        """
        Swaps local matrix input/output axes without reversing sites.

        The result keeps the concrete format class and its methods. Its core
        and bond containers are independent, sharing tensor storage.

        Returns
        -------
        TTM or TRM
            Transposed matrix, including the corresponding
            :class:`~tensorkrowch.formats.QTTM` or
            :class:`~tensorkrowch.formats.QTRM` class when applicable.
            Input/output dimensions and coordinate metadata are exchanged.

        Examples
        --------
        >>> matrix = tk.formats.TTM([torch.tensor([[1 + 2j, 3 - 4j],
        ...                                        [5 + 6j, 7 - 8j]])])
        >>> matrix.transpose().contract_dense()
        tensor([[1.+2.j, 5.+6.j],
                [3.-4.j, 7.-8.j]])
        """
        result = copy(self)

        # The final open core omits its right bond axis.
        last_in_axis = -3 if self._cyclic else -2
        cores = [core.transpose(
                    last_in_axis if site == (self.n_sites - 1) else -3, -1)
                 for site, core in enumerate(self._cores)]
        result._cores = _SafeList(cores, result._on_cores_changed)

        if self._bonds is not None:
            result._bonds = self._bonds._map_tensors(
                lambda tensor: tensor, result._on_bonds_changed)

        result._in_dim, result._out_dim = self._out_dim, self._in_dim
        result._same_in_dim, result._same_out_dim = (
            self._same_out_dim, self._same_in_dim)

        return result

    def adjoint(self) -> '_MatrixFormat1D':
        """
        Returns the conjugate transpose, including explicit factors.

        Returns
        -------
        TTM or TRM
            Conjugate transpose. Containers are independent and tensors may
            share storage. The concrete format class and its methods are
            preserved. Input/output coordinate metadata are exchanged for
            :class:`~tensorkrowch.formats.QTTM` and
            :class:`~tensorkrowch.formats.QTRM`.

        Examples
        --------
        >>> matrix = tk.formats.TTM([torch.tensor([[1 + 2j, 3 - 4j],
        ...                                        [5 + 6j, 7 - 8j]])])
        >>> matrix.adjoint().contract_dense()
        tensor([[1.-2.j, 5.-6.j],
                [3.+4.j, 7.+8.j]])
        """
        return self.transpose().conj()

    @property
    def T(self) -> '_MatrixFormat1D':
        """Matrix transpose, without reversing the chain."""
        return self.transpose()

    @property
    def H(self) -> '_MatrixFormat1D':
        """Matrix adjoint, including conjugation of bond factors."""
        return self.adjoint()

    def matmul(self,
               other: Union['_VectorFormat1D', '_MatrixFormat1D']
               ) -> 'TensorFormat1D':
        """
        Applies this operator to a column or composes it with an operator.

        Parameters
        ----------
        other : TT, TR, TTM or TRM
            Column or operator whose output space matches this operator's input
            space at every site. For vectors this is their physical space.

        Returns
        -------
        TT, TR, TTM or TRM
            Exact product with the same kind of object as ``other``. A cyclic
            operand gives a cyclic result. No rounding is performed.

        Examples
        --------
        >>> matrix = tk.formats.TTM([torch.diag(torch.tensor([2., 3.]))])
        >>> x = tk.formats.TT([torch.tensor([1., 2.])])
        >>> product = matrix.matmul(x)
        >>> product.contract_dense()
        tensor([2., 6.])
        >>> torch.equal(product.contract_dense(),
        ...             (matrix @ x).contract_dense())
        True

        >>> composition = matrix.matmul(matrix)
        >>> composition.contract_dense()
        tensor([[4., 0.],
                [0., 9.]])
        >>> torch.equal(composition.contract_dense(),
        ...             (matrix @ matrix).contract_dense())
        True
        """
        if not isinstance(other, (_VectorFormat1D, _MatrixFormat1D)):
            raise TypeError(
                'An operator can multiply a column or another operator')
        if isinstance(other, _VectorFormat1D) and other._is_row:
            raise TypeError('An operator can multiply a column, not a row')

        self._check_semantics(other, product=True)
        cores, n_batches, cyclic = self._product_cores(other)
        if isinstance(other, _VectorFormat1D):
            in_dim, out_dim = self._out_dim, None
        else:
            in_dim, out_dim = other._in_dim, self._out_dim
        return self._new_from_standard_cores(
            cores, in_dim, out_dim, n_batches, cyclic,
            other=other, product=True)

    def __matmul__(self,
                   other: Union['_VectorFormat1D', '_MatrixFormat1D']
                   ) -> 'TensorFormat1D':
        """Calls :meth:`matmul` with the same operand."""
        return self.matmul(other)

    def trace(self) -> torch.Tensor:
        """
        Contracts the operator trace without forming the dense matrix.

        Returns
        -------
        torch.Tensor
            Trace resolved by structural batch. Every site should have matching
            input/output dimensions; global squareness alone is insufficient.

        Examples
        --------
        >>> matrix = tk.formats.TTM([torch.diag(torch.tensor([2., 3.]))])
        >>> matrix.trace()
        tensor(5.)
        """
        if self._in_dim != self._out_dim:
            raise ValueError(
                'Trace requires matching local input/output dimensions')

        matrices = [core.diagonal(dim1=-3, dim2=-1).sum(-1)
                    for core in self._operator_cores()]
        return self._contract_open_chain(matrices).diagonal(
            dim1=-2, dim2=-1).sum(-1)

    def apply(self,
              data: Union[EvaluationData, TensorFormat1D],
              n_batches: int = 1) -> TensorFormat1D:
        """
        Applies the operator to product data or another format.

        Parameters
        ----------
        data : torch.Tensor, sequence of torch.Tensor or TensorFormat1D
            Integer configurations or embedded feature vectors in the same
            layouts as ``in_data`` in :meth:`evaluate`, or a
            :class:`~tensorkrowch.formats.TT` / :class:`~tensorkrowch.formats.TR`
            column or :class:`~tensorkrowch.formats.TTM` /
            :class:`~tensorkrowch.formats.TRM` matrix.
            With a vector, contracts ``in_dim``; with a matrix,
            contracts ``self.in_dim`` with ``data.out_dim``.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        :class:`~tensorkrowch.formats.TensorFormat1D`
            Vector or matrix format, according to the operand. Cyclic if either
            operand is cyclic. Product data become structural batches in the
            returned vector; applying a global dense vector does not
            automatically factor it into TT cores.

        Examples
        --------
        >>> operator = tk.formats.TTM([torch.eye(2)])
        >>> vector = tk.formats.TT([torch.tensor([2., 3.])])
        >>> operator.apply(vector).contract_dense()
        tensor([2., 3.])

        >>> operator.apply(torch.tensor([[0], [1]])).contract_dense().tolist()
        [[1.0, 0.0], [0.0, 1.0]]

        >>> embeddings = torch.tensor([[[2., 3.]]])
        >>> operator.apply(embeddings).contract_dense()
        tensor([[2., 3.]])
        """
        if isinstance(data, TensorFormat1D):
            return self @ data

        site_data, discrete, data_batch_shape = self._prepare_data(
            data, self._in_dim, self._same_in_dim, n_batches)

        output_cores = []
        for core, site_value in zip(self._operator_cores(), site_data):
            left_rank, in_dim, right_rank, out_dim = core.shape[-4:]
            core = core.reshape(-1, left_rank, in_dim, right_rank, out_dim)

            if discrete:
                indices = site_value.reshape(-1).to(torch.long)
                output_core = core.index_select(2, indices)
                output_core = output_core.permute(0, 2, 1, 4, 3)
            else:
                vectors = site_value.reshape(-1, in_dim)
                output_core = torch.einsum(
                    'di,cliro->cdlor', vectors, core.to(dtype=vectors.dtype))

            output_cores.append(output_core.reshape(
                *self._batch_shape,
                *data_batch_shape,
                left_rank,
                out_dim,
                right_rank))

        return self._new_from_standard_cores(
            output_cores, self._out_dim, None,
            self._n_batches + n_batches, self._cyclic, product=True)

    def _local_matrices(
            self,
            in_data_by_site: Sequence[torch.Tensor],
            out_data_by_site: Sequence[torch.Tensor],
            in_discrete: bool,
            out_discrete: bool,
            data_batch_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Builds local matrices for paired matrix-entry evaluation."""
        matrices = []
        for core, in_site_data, out_site_data in zip(
                self._operator_cores(), in_data_by_site, out_data_by_site):
            left_rank, in_dim, right_rank, out_dim = core.shape[-4:]
            core = core.reshape(-1, left_rank, in_dim, right_rank, out_dim)

            if in_discrete and out_discrete:
                in_indices = in_site_data.reshape(-1).to(torch.long)
                out_indices = out_site_data.reshape(-1).to(torch.long)
                fused = in_indices * out_dim + out_indices
                matrix = core.permute(0, 1, 3, 2, 4).reshape(
                    -1, left_rank, right_rank, in_dim * out_dim)
                matrix = matrix[..., fused].permute(0, 3, 1, 2)
            elif in_discrete:
                indices = in_site_data.reshape(-1).to(torch.long)
                vectors = out_site_data.reshape(-1, out_dim)
                selected = core.index_select(2, indices).to(dtype=vectors.dtype)
                matrix = torch.einsum('cldro,do->cdlr', selected, vectors)
            elif out_discrete:
                indices = out_site_data.reshape(-1).to(torch.long)
                vectors = in_site_data.reshape(-1, in_dim)
                selected = core.index_select(4, indices).to(dtype=vectors.dtype)
                matrix = torch.einsum('clird,di->cdlr', selected, vectors)
            else:
                in_vectors = in_site_data.reshape(-1, in_dim)
                out_vectors = out_site_data.reshape(-1, out_dim)
                dtype = torch.promote_types(in_vectors.dtype, out_vectors.dtype)
                matrix = torch.einsum('di,do,cliro->cdlr',
                                      in_vectors.to(dtype=dtype),
                                      out_vectors.to(dtype=dtype),
                                      core.to(dtype=dtype))

            matrices.append(matrix.reshape(*self._batch_shape,
                                           *data_batch_shape,
                                           left_rank,
                                           right_rank))
        return matrices

    def evaluate(self,
                 in_data: EvaluationData,
                 out_data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        """
        Evaluates paired integer configurations or local embedded feature
        vectors.

        Parameters
        ----------
        in_data : torch.Tensor or sequence of torch.Tensor
            Integer configurations of shape ``(*data_batch, n_sites)``, or
            embedded feature vectors of shape ``(*data_batch, n_sites, in_dim)``
            for uniform dimensions. For heterogeneous dimensions, pass one
            tensor per site. Each site tensor has shape ``(*data_batch,)`` for
            indices or ``(*data_batch, in_dim[site])`` for embeddings.
        out_data : torch.Tensor or sequence of torch.Tensor
            Output configurations or embeddings in the same layouts as
            ``in_data``, using ``out_dim`` instead of ``in_dim``. Both data
            batch shapes should match.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        torch.Tensor
            Values with shape ``(*core_batch, *data_batch)``. Structural
            batches and data batches remain independent. Embedded contractions
            use supplied vectors directly, without implicit conjugation.

        Examples
        --------
        >>> matrix = tk.formats.TTM([torch.diag(torch.tensor([2., 3.]))])
        >>> indices = torch.tensor([[0], [1]])
        >>> matrix.evaluate(indices, indices).tolist()
        [2.0, 3.0]
        >>> embeddings = torch.ones(1, 1, 2)
        >>> matrix.evaluate(embeddings, embeddings).tolist()
        [5.0]
        """
        in_data_by_site, in_discrete, data_batch_shape = self._prepare_data(
            in_data, self._in_dim, self._same_in_dim, n_batches)
        out_data_by_site, out_discrete, out_batch_shape = self._prepare_data(
            out_data, self._out_dim, self._same_out_dim, n_batches)
        if out_batch_shape != data_batch_shape:
            raise ValueError(
                'Input and output data should have the same batch shape')

        matrices = self._local_matrices(
            in_data_by_site, out_data_by_site,
            in_discrete, out_discrete, data_batch_shape)
        return self._contract_local_matrices(matrices)

    def __call__(self,
                 in_data: Union[EvaluationData, TensorFormat1D],
                 out_data: Optional[EvaluationData] = None,
                 n_batches: int = 1
                 ) -> Union[torch.Tensor, TensorFormat1D]:
        """
        Calls :meth:`evaluate` with paired inputs, or :meth:`apply` when
        ``out_data`` is omitted, using the same input conventions.
        """
        if out_data is None:
            return self.apply(in_data, n_batches=n_batches)
        return self.evaluate(in_data, out_data, n_batches=n_batches)

    def to_mpo(self,
               parameterized: bool = False,
               **kwargs) -> 'MPO':
        """
        Builds an open or periodic :class:`~tensorkrowch.models.MPO`
        from effective cores.

        :class:`~tensorkrowch.formats.TTM` produces open boundaries (``'obc'``);
        :class:`~tensorkrowch.formats.TRM` produces periodic boundaries
        (``'pbc'``). Batched matrices are not supported by
        :class:`~tensorkrowch.models.MPO`.

        Parameters
        ----------
        parameterized : bool
            Whether the constructed model uses trainable parameter nodes.
            Inputs are not detached implicitly.
        **kwargs : keyword arguments
            Additional model constructor options. Tensor cores and boundary are
            supplied by the adapter.

        Returns
        -------
        :class:`~tensorkrowch.models.MPO`
            New graph model. Stored factors are materialized in temporary
            tensors; the source format is unchanged.

        Examples
        --------
        >>> matrix = tk.formats.TTM([torch.eye(2)])
        >>> model = matrix.to_mpo()
        >>> restored = tk.formats.TTM.from_mpo(model)
        >>> torch.allclose(restored.contract_dense(), matrix.contract_dense())
        True
        """
        from tensorkrowch.models import MPO

        if self._n_batches:
            raise ValueError('Batched MPO model cores are not supported')
        if not isinstance(parameterized, bool):
            raise TypeError('`parameterized` should be bool type')

        cores = _restore_cores(self._effective_cores(), self._in_dim,
                               self._out_dim, self._n_batches, self._cyclic)
        return MPO(tensors=cores, parameterized=parameterized, **kwargs)

    @classmethod
    def from_mpo(cls, model: 'MPO', **kwargs) -> '_MatrixFormat1D':
        """
        Collects effective open or periodic :class:`~tensorkrowch.models.MPO`
        tensors.

        Parameters
        ----------
        model : MPO
            Source model: :class:`~tensorkrowch.formats.TTM` requires open
            boundaries (``'obc'``), and :class:`~tensorkrowch.formats.TRM`
            requires periodic boundaries (``'pbc'``). For open models,
            ``model.tensors`` already includes contractions with the end nodes.
        **kwargs : keyword arguments
            Additional options for the concrete format constructor, such as
            :class:`~tensorkrowch.formats.QTTM` or
            :class:`~tensorkrowch.formats.QTRM` metadata on a subclass.

        Returns
        -------
        :class:`~tensorkrowch.formats.TTM` or :class:`~tensorkrowch.formats.TRM`
            Format sharing the effective tensor storage. Graph nodes and fit
            metrics are not retained.
        """
        from tensorkrowch.models import MPO

        if not isinstance(model, MPO):
            raise TypeError('`model` should be MPO type')
        boundary = 'pbc' if cls._cyclic else 'obc'
        if model.boundary != boundary:
            raise ValueError(f'This adapter requires {boundary} boundaries')

        return cls(model.tensors, **kwargs)


class _OpenFormat1D(TensorFormat1D):
    """Canonical forms and contractions shared by open chains."""

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts selected matrices along an open chain."""
        return self._contract_open_chain(matrices).squeeze(-1).squeeze(-1)

    def canonicalize_vidal(self,
                           mode: str = 'implicit',
                           inverse_positions: Optional[Sequence[int]] = None,
                           remaining_mode: str = 'implicit',
                           inverse_cutoff: float = 0.0) -> 'TensorFormat1D':
        """
        Builds or redistributes an open-chain Vidal gauge in-place.

        Only :class:`TT` and :class:`TTM` admit this global Schmidt
        representation. Invalidated gauges are recomputed; valid gauges can
        be redistributed without another SVD. No truncation is performed to
        manufacture an inverse.

        Parameters
        ----------
        mode : {"implicit", "explicit", "inverse", "left", "right"}
            Absorbs spectrum powers ``(0.5, 0.5)``, ``(0, 0)``, ``(1, 1)``,
            ``(1, 0)`` or ``(0, 1)`` into the left/right neighboring cores,
            respectively. The remaining diagonal factor has power
            ``1 - left_power - right_power``.
        inverse_positions : sequence of int, optional
            Distinct virtual bonds to use in inverse form. Cannot be combined
            with ``mode="inverse"``. When supplied, ``remaining_mode`` controls
            every other bond.
        remaining_mode : {"implicit", "explicit", "left", "right"}
            Distribution for bonds not selected by ``inverse_positions``.
        inverse_cutoff : float
            Finite non-negative threshold for spectra used in inverse form.
            Positive values at or below this threshold raise ``ValueError``;
            call :meth:`rounding` first to truncate small values, then retry.
            Exact zeros use the pseudoinverse and remain zero. This method
            does not truncate values to create an inverse.

        Returns
        -------
        TensorFormat1D
            The current format with a valid
            :class:`~tensorkrowch.formats.VidalGauge`.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> _ = format.canonicalize_vidal(inverse_positions=[0])
        >>> format.bonds.powers
        [(1, 1)]
        >>> _ = format.canonicalize_vidal(mode='implicit')
        >>> format.bonds.powers
        [(0.5, 0.5)]
        >>> torch.allclose(format.contract_dense(), torch.eye(2))
        True
        """
        modes = {'explicit': (0, 0),
                 'implicit': (0.5, 0.5),
                 'inverse': (1, 1),
                 'left': (1, 0),
                 'right': (0, 1)}
        if mode not in modes or remaining_mode == 'inverse':
            raise ValueError('Invalid Vidal mode or `remaining_mode`')
        if isinstance(inverse_cutoff, bool) or \
                not isinstance(inverse_cutoff, Real):
            raise TypeError('`inverse_cutoff` should be a real number')
        if not isfinite(inverse_cutoff) or (inverse_cutoff < 0):
            raise ValueError(
                '`inverse_cutoff` should be finite and non-negative')

        n_bonds = self.n_sites - 1
        if inverse_positions is None:
            positions = range(n_bonds) if mode == 'inverse' else ()
            powers = [modes[mode]] * n_bonds
        else:
            positions = list(inverse_positions)
            if mode == 'inverse':
                raise ValueError(
                    'Select either mode="inverse" or `inverse_positions`')
            if any(isinstance(bond, bool) or not isinstance(bond, int)
                   for bond in positions):
                raise TypeError('Inverse bond positions should be integers')
            if (len(set(positions)) != len(positions)) or any(
                    (bond < 0) or (bond >= n_bonds) for bond in positions):
                raise ValueError(
                    'Inverse bond positions should be distinct valid bonds')
            powers = [modes['inverse' if bond in positions else remaining_mode]
                      for bond in range(n_bonds)]

        # Reuse stored spectra when the Vidal representation is still valid.
        if isinstance(self._bonds, VidalGauge) and self._bonds._valid:
            cores = list(self._standard_cores())
            spectra = self._bonds.spectra
            old_powers = self._bonds.powers
        else:
            cores = _canonicalize_cores(self._effective_cores(), 0, False)
            spectra = []
            batch_shape = self._batch_shape

            # Recover the spectra and cores of the explicit representation.
            for bond in range(n_bonds):
                core = cores[bond]
                u, s, vh = truncated_svd(
                    core.reshape(*batch_shape, -1, core.shape[-1]))
                cores[bond] = u.reshape(
                    *batch_shape, core.shape[-3], core.shape[-2], s.shape[-1])

                if bond:
                    previous = spectra[-1]
                    safe = torch.where(previous > 0, previous,
                                       torch.ones_like(previous))
                    inverse = torch.where(previous > 0, safe.reciprocal(),
                                          torch.zeros_like(previous))
                    cores[bond] = inverse[..., :, None, None] * cores[bond]

                spectra.append(s)
                cores[bond + 1] = torch.einsum('...ab,...bpr->...apr',
                                               s.unsqueeze(-1) * vh,
                                               cores[bond + 1])
            if n_bonds:
                last = spectra[-1]
                safe = torch.where(last > 0, last, torch.ones_like(last))
                inverse = torch.where(last > 0, safe.reciprocal(),
                                      torch.zeros_like(last))
                cores[-1] = inverse[..., :, None, None] * cores[-1]
            old_powers = [(0, 0)] * n_bonds

        for bond in positions:
            spectrum = spectra[bond]
            if torch.any((spectrum > 0) & (spectrum <= inverse_cutoff)):
                raise ValueError(
                    f'Inverse Vidal bond {bond} has values at or below '
                    '`inverse_cutoff`')

        cores, factors = _redistribute(cores, spectra, old_powers, powers)
        self._set_standard_cores(
            cores, factors, spectra=spectra, powers=powers)
        return self

    def redistribute_vidal(self,
                           bond: int,
                           mode: str = 'implicit',
                           inverse_cutoff: float = 0.0) -> 'TensorFormat1D':
        """
        Redistributes one valid Vidal spectrum without another SVD.

        Requires a valid stored :class:`~tensorkrowch.formats.VidalGauge`.
        Manual edits invalidate it;
        recompute :meth:`canonicalize_vidal` before redistributing spectra.

        Parameters
        ----------
        bond : int
            Index of an internal virtual bond, joining sites ``bond`` and
            ``bond + 1``.
        mode : {"implicit", "explicit", "inverse", "left", "right"}
            New powers absorbed into the left/right neighbors: ``(0.5, 0.5)``,
            ``(0, 0)``, ``(1, 1)``, ``(1, 0)`` or ``(0, 1)``, respectively.
        inverse_cutoff : float
            Finite non-negative threshold. Positive spectrum values used in an
            inverse should exceed it; exact zeros use the pseudoinverse and
            remain zero. Values are not truncated to create an inverse.

        Returns
        -------
        TensorFormat1D
            The current format. Other bonds retain their distribution.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> _ = format.canonicalize_vidal()
        >>> _ = format.redistribute_vidal(0, mode='left')
        >>> format.bonds.powers
        [(1, 0)]
        >>> format.contract_dense()
        tensor([[1., 0.],
                [0., 1.]])
        """
        if isinstance(bond, bool) or not isinstance(bond, int):
            raise TypeError('`bond` should be int type')
        if not isinstance(self._bonds, VidalGauge) or not self._bonds._valid:
            raise ValueError(
                'Bond redistribution requires valid stored Vidal spectra')
        if not 0 <= bond < len(self._bonds.spectra):
            raise ValueError('`bond` should select a valid virtual bond')

        modes = {'explicit': (0, 0),
                 'implicit': (0.5, 0.5),
                 'inverse': (1, 1),
                 'left': (1, 0),
                 'right': (0, 1)}
        if mode not in modes:
            raise ValueError('Invalid Vidal bond distribution mode')
        if isinstance(inverse_cutoff, bool) or \
                not isinstance(inverse_cutoff, Real):
            raise TypeError('`inverse_cutoff` should be a real number')
        if not isfinite(inverse_cutoff) or (inverse_cutoff < 0):
            raise ValueError(
                '`inverse_cutoff` should be finite and non-negative')
        if mode == 'inverse':
            spectrum = self._bonds.spectra[bond]
            if torch.any((spectrum > 0) & (spectrum <= inverse_cutoff)):
                raise ValueError(
                    f'Bond {bond} has values at or below `inverse_cutoff`')

        powers = list(self._bonds.powers)
        powers[bond] = modes[mode]
        cores, factors = _redistribute(self._standard_cores(),
                                       self._bonds.spectra,
                                       self._bonds.powers,
                                       powers)
        self._set_standard_cores(
            cores, factors, spectra=self._bonds.spectra, powers=powers)
        return self

    def canonicalize_minimal(self,
                             max_iter: int = 200,
                             lr: float = 0.05,
                             tol: float = 1e-8,
                             *,
                             return_info: bool = False) -> Union[
            'TensorFormat1D',
            Tuple['TensorFormat1D', MinimalCanonicalInfo]]:
        """
        Computes the minimal canonical form of an open chain in-place.

        Uses implicit Vidal form through :meth:`canonicalize_vidal`, without
        iterative optimization. At each bond, the Gram matrices of the
        contracted left and right subchains agree. This condition concerns
        whole subchains, not just neighboring cores, as described in Section
        3.3 of `The minimal canonical form of a tensor network
        <https://arxiv.org/pdf/2209.14358>`_. The represented tensor is
        preserved.

        Parameters
        ----------
        max_iter : int
            Positive iteration limit retained for a common interface with rings.
            Open chains do not perform iterative optimization.
        lr : float
            Positive learning rate, used only by cyclic formats.
        tol : float
            Positive convergence tolerance, used only by cyclic formats.
        return_info : bool
            Whether to return the canonicalization information with the format.

        Returns
        -------
        TensorFormat1D or tuple[TensorFormat1D, MinimalCanonicalInfo]
            The current format, optionally reporting zero iterations and
            convergence of the direct Vidal construction. ``gram_imbalance``
            is ``None`` because that diagnostic compares neighboring cores,
            rather than the subchains used by the open-chain condition.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> _, info = format.canonicalize_minimal(return_info=True)
        >>> info.iterations, info.converged
        (0, True)
        """
        _validate_minimal_options(max_iter, lr, tol, return_info)
        self.canonicalize_vidal('implicit')
        if return_info:
            return self, MinimalCanonicalInfo(0, True, None)
        return self


class _CyclicFormat1D(TensorFormat1D):
    """Contractions, gauges and transformations shared by rings."""

    _cyclic = True

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts selected matrices and closes the ring."""
        return self._contract_open_chain(matrices).diagonal(
            dim1=-2, dim2=-1).sum(-1)

    def rotate(self, first: int = 0) -> 'TensorFormat1D':
        """
        Returns a ring whose first core is the selected site.

        Parameters
        ----------
        first : int
            Site that becomes position zero. Physical axes and bond factors
            follow the same cyclic rotation.

        Returns
        -------
        TR or TRM
            Rotated format sharing the original core tensors.

        Examples
        --------
        >>> ring = tk.formats.TR([torch.ones(1, 2, 1), torch.ones(1, 3, 1)])
        >>> rotated = ring.rotate(1)
        >>> rotated.in_dim
        (3, 2)
        """
        if isinstance(first, bool) or not isinstance(first, int):
            raise TypeError('`first` should be int type')
        if first < 0 or first >= self.n_sites:
            raise ValueError('`first` should select a valid site')

        order = [*range(first, self.n_sites), *range(first)]
        cores = self._standard_cores()
        cores = [cores[site] for site in order]
        in_dim = tuple(self._in_dim[site] for site in order)
        out_dim = None if self._out_dim is None else tuple(
            self._out_dim[site] for site in order)

        result = self._new_from_standard_cores(
            cores, in_dim, out_dim, self._n_batches, True)
        if self._bonds is not None:
            result.bonds = [self._bonds.factors[site] for site in order]
        return result

    def _to_open(self) -> 'TensorFormat1D':
        """Carries the closing rank through identity factors to an open chain."""
        cores = self._effective_cores()
        batch_shape = self._batch_shape
        closing = cores[0].shape[-3]

        if len(cores) == 1:
            result = [cores[0].diagonal(
                dim1=-3, dim2=-1).sum(-1).unsqueeze(-2).unsqueeze(-1)]
        else:
            first = cores[0].transpose(-3, -2).reshape(
                *batch_shape, 1, cores[0].shape[-2], -1)
            result = [first]
            identity = torch.eye(closing, dtype=self.dtype, device=self.device)
            for core in cores[1:-1]:
                combined = torch.einsum('st,...aib->...saitb', identity, core)
                result.append(combined.reshape(
                    *batch_shape, closing * core.shape[-3],
                    core.shape[-2], closing * core.shape[-1]))
            last = cores[-1].movedim(-1, -3)
            result.append(last.reshape(*batch_shape,
                                       closing * cores[-1].shape[-3],
                                       cores[-1].shape[-2], 1))
        return self._new_from_standard_cores(
            result, self._in_dim, self._out_dim, self._n_batches, False)

    def canonicalize_minimal(
            self,
            max_iter: int = 200,
            lr: float = 0.05,
            tol: float = 1e-8,
            *,
            return_info: bool = False
        ) -> Union['TensorFormat1D',
                   Tuple['TensorFormat1D', MinimalCanonicalInfo]]:
        """
        Approximates the minimal canonical form of a ring in-place.

        Uses Adam to minimize half the sum of squared core norms over virtual
        gauges, parametrized as exponentials of Hermitian matrices. Each gauge
        and its inverse act on neighboring cores, preserving the represented
        tensor. At a minimum, neighboring bond Gram matrices are equal.

        Returns the best finite iterate, even if optimization stops before
        convergence. This implements the minimization objective discussed in
        `The minimal canonical form of a tensor network
        <https://arxiv.org/pdf/2209.14358>`_, but not its first- or second-order
        algorithms; their convergence guarantees do not apply here.
        Structural batches share one gauge per bond. Gauge optimization does
        not accumulate gradients on the input cores.

        Parameters
        ----------
        max_iter : int
            Positive maximum number of gauge optimization iterations for a ring.
        lr : float
            Finite positive learning rate of the Adam gauge optimizer.
        tol : float
            Finite positive stopping tolerance for the largest absolute
            gradient entry of the Hermitian gauge parameters. This does not
            bound :meth:`~tensorkrowch.formats.GaugeOrbit.gram_imbalance`.
        return_info : bool
            If ``True``, returns the format together with the operation-specific
            information record.

        Returns
        -------
        TensorFormat1D or tuple[TensorFormat1D, MinimalCanonicalInfo]
            The current format, optionally with iteration count, convergence
            status and final ``gram_imbalance``. Convergence refers to the
            parameter-gradient tolerance, not a certified global minimum.

        Examples
        --------
        >>> ring = tk.formats.TR([torch.ones(1, 2, 1)] * 2)
        >>> _, info = ring.canonicalize_minimal(return_info=True)
        >>> info.converged
        True
        """
        _validate_minimal_options(max_iter, lr, tol, return_info)

        # Optimize gauges on detached cores to avoid accumulating core gradients.
        orbit = TensorRingOrbit(self)
        if not all(torch.isfinite(core).all() for core in orbit.cores):
            raise ValueError('Minimal canonicalization requires finite cores')

        detached = GaugeOrbit(
            [core.detach() for core in orbit.cores], orbit.bonds)
        scale = max(core.abs().amax().item() for core in detached.cores)
        if scale == 0:
            info = MinimalCanonicalInfo(
                0, True, self._cores[0].real.new_zeros(()))
            return (self, info) if return_info else self
        detached.cores = tuple(core / scale for core in detached.cores)

        best_gauge = [torch.eye(rank, dtype=self.dtype, device=self.device)
                      for rank in self._rank]
        best_loss = detached.objective(best_gauge).item()
        converged, iterations = False, 0
        with torch.enable_grad():
            parameters = [torch.zeros_like(gauge, requires_grad=True)
                          for gauge in best_gauge]
            optimizer = torch.optim.Adam(parameters, lr=lr)

            for _ in range(max_iter):
                iterations += 1
                optimizer.zero_grad()

                gauges = [torch.matrix_exp(
                    (parameter + parameter.transpose(-2, -1).conj()) / 2)
                    for parameter in parameters]
                if not all(torch.isfinite(gauge).all() for gauge in gauges):
                    break

                try:
                    loss = detached.objective(gauges)
                except torch.linalg.LinAlgError:
                    break
                if not torch.isfinite(loss):
                    break

                loss_value = loss.item()
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_gauge = [gauge.detach() for gauge in gauges]
                loss.backward()

                if not all(torch.isfinite(parameter.grad).all()
                           for parameter in parameters):
                    break

                if max(parameter.grad.abs().amax().item()
                       for parameter in parameters) <= tol:
                    converged = True
                    break

                optimizer.step()

        # Apply the best finite iterate to the original cores.
        self._set_standard_cores(orbit.apply(best_gauge))
        if return_info:
            info = MinimalCanonicalInfo(iterations, converged,
                                        TensorRingOrbit(self).gram_imbalance())
            return self, info
        return self


class TT(_OpenFormat1D, _VectorFormat1D):
    """
    Open tensor train represented by a sequence of cores.

    With leading structural batch axes B, the first and last cores have shapes
    ``(*B, input, right)`` and ``(*B, left, input)``; interiors use
    ``(*B, left, input, right)``. A single core is ``(*B, input)``.
    The constructor shares tensors and copies their container. No nodes or
    edges are constructed, and input tensors retain autograd.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described above. The container is copied
        and tensor storage is shared; inputs retain autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    bonds : sequence[torch.Tensor or None], optional
        Explicit diagonal factors between cores. ``None`` uses no explicit
        factors.
    """

    _topology = 'tt'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates core layouts and returns structural dimensions and ranks."""
        n_sites = len(self._cores)
        batch_shape = tuple(self._cores[0].shape[:self._n_batches])
        in_dim = []

        if n_sites == 1:
            if self._cores[0].ndim != (self._n_batches + 1):
                raise ValueError(
                    'A one-site TT core should have one input dimension')
            in_dim.append(self._cores[0].shape[-1])
            return [], batch_shape, tuple(in_dim), None

        rank = []
        for site, core in enumerate(self._cores):
            if tuple(core.shape[:self._n_batches]) != batch_shape:
                raise ValueError(
                    'All TT cores should have the same batch shape')

            if site == 0:
                if core.ndim != (self._n_batches + 2):
                    raise ValueError(
                        'The first TT core should have input and right rank '
                        'dimensions')
                in_dim.append(core.shape[-2])
                rank.append(core.shape[-1])
            elif site == (n_sites - 1):
                if core.ndim != (self._n_batches + 2):
                    raise ValueError(
                        'The last TT core should have left rank and input '
                        'dimensions')
                if core.shape[-2] != rank[-1]:
                    raise ValueError('Adjacent TT ranks should match')
                in_dim.append(core.shape[-1])
            else:
                if core.ndim != (self._n_batches + 3):
                    raise ValueError(
                        'Interior TT cores should have left, input and '
                        'right dimensions')
                if core.shape[-3] != rank[-1]:
                    raise ValueError('Adjacent TT ranks should match')
                in_dim.append(core.shape[-2])
                rank.append(core.shape[-1])

        return rank, batch_shape, tuple(in_dim), None


class TR(_CyclicFormat1D, _VectorFormat1D):
    """
    Tensor ring represented by a sequence of cores.

    Every core has shape ``(*batch, left, input, right)``. Adjacent ranks
    match, including the last-to-first closure. A one-site ring is a trace
    over its two virtual axes. Tensors retain storage and autograd without
    constructing TensorKrowch nodes or edges.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described above. The container is copied
        and tensor storage is shared; inputs retain autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    bonds : sequence[torch.Tensor or None], optional
        Explicit diagonal factors between cores. ``None`` uses no explicit
        factors.
    """

    _topology = 'tr'

    def to_tt(self) -> 'TT':
        """
        Opens the ring exactly by carrying the closure index through all sites.

        Returns
        -------
        :class:`TT`
            Open format with the same dense tensor. The first and last ranks
            incorporate the closure rank; intermediate cores carry an identity
            on that index. No truncation or densification is performed.

        Examples
        --------
        >>> ring = tk.formats.TR([torch.eye(2).reshape(2, 1, 2)] * 2)
        >>> train = ring.to_tt()
        >>> torch.allclose(train.contract_dense(), ring.contract_dense())
        True
        """
        return self._to_open()

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates core layouts and returns structural dimensions and ranks."""
        batch_shape = tuple(self._cores[0].shape[:self._n_batches])
        rank = []
        in_dim = []

        for site, core in enumerate(self._cores):
            if core.ndim != (self._n_batches + 3):
                raise ValueError(
                    'TR cores should have left rank, input and right rank '
                    'dimensions')
            if tuple(core.shape[:self._n_batches]) != batch_shape:
                raise ValueError(
                    'All TR cores should have the same batch shape')
            if site and (core.shape[-3] != rank[-1]):
                raise ValueError('Adjacent TR ranks should match')
            in_dim.append(core.shape[-2])
            rank.append(core.shape[-1])

        if self._cores[-1].shape[-1] != self._cores[0].shape[-3]:
            raise ValueError('The last and first cyclic TR ranks should match')
        return rank, batch_shape, tuple(in_dim), None


class TTM(_OpenFormat1D, _MatrixFormat1D):
    """
    Tensor train operator with local input and output dimensions.

    The first and last core shapes are ``(input, right, output)`` and
    ``(left, input, output)``; interiors are
    ``(left, input, right, output)``. A single core is ``(input, output)``.
    Structural batches are currently unsupported. Tensor storage and autograd
    are retained without constructing TensorKrowch nodes or edges.

    TTM currently requires ``n_batches=0``; batched operator formats use
    :class:`TRM`.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described above. The container is copied
        and tensor storage is shared; inputs retain autograd.
    n_batches : int
        Number of leading structural batch axes. Only zero is supported.
    bonds : sequence[torch.Tensor or None], optional
        Explicit diagonal factors between cores. ``None`` uses no explicit
        factors.
    """

    _topology = 'ttm'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates core layouts and returns structural dimensions and ranks."""
        if self._n_batches:
            raise ValueError('TTM structural batches are not supported')

        n_sites = len(self._cores)
        in_dim = []
        out_dim = []
        if n_sites == 1:
            if self._cores[0].ndim != 2:
                raise ValueError(
                    'A one-site TTM core should have input and output dimensions')
            in_dim.append(self._cores[0].shape[0])
            out_dim.append(self._cores[0].shape[1])
            return [], (), tuple(in_dim), tuple(out_dim)

        rank = []
        for site, core in enumerate(self._cores):
            if site == 0:
                if core.ndim != 3:
                    raise ValueError(
                        'The first TTM core should have input, right rank and '
                        'output dimensions')
                in_dim.append(core.shape[0])
                out_dim.append(core.shape[2])
                rank.append(core.shape[1])
            elif site == (n_sites - 1):
                if core.ndim != 3:
                    raise ValueError(
                        'The last TTM core should have left rank, input and '
                        'output dimensions')
                if core.shape[0] != rank[-1]:
                    raise ValueError('Adjacent TTM ranks should match')
                in_dim.append(core.shape[1])
                out_dim.append(core.shape[2])
            else:
                if core.ndim != 4:
                    raise ValueError(
                        'Interior TTM cores should have left, input, right and '
                        'output dimensions')
                if core.shape[0] != rank[-1]:
                    raise ValueError('Adjacent TTM ranks should match')
                in_dim.append(core.shape[1])
                out_dim.append(core.shape[3])
                rank.append(core.shape[2])

        return rank, (), tuple(in_dim), tuple(out_dim)


class TRM(_CyclicFormat1D, _MatrixFormat1D):
    """
    Tensor ring operator with local input and output dimensions.

    Every core has shape ``(*batch, left, input, right, output)`` and
    adjacent ranks match through the cyclic closure. Structural batches are
    independent of evaluation-data batches. Tensor storage and autograd are
    retained without constructing TensorKrowch nodes or edges.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Core tensors with the shapes described above. The container is copied
        and tensor storage is shared; inputs retain autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    bonds : sequence[torch.Tensor or None], optional
        Explicit diagonal factors between cores. ``None`` uses no explicit
        factors.
    """

    _topology = 'trm'

    def to_ttm(self) -> 'TTM':
        """
        Opens the ring exactly by carrying the closure index through all sites.

        A batched ring matrix cannot be converted because :class:`TTM` does
        not support structural batches.

        Returns
        -------
        :class:`TTM`
            Open format with the same dense tensor. The first and last ranks
            incorporate the closure rank; intermediate cores carry an identity
            on that index. No truncation or densification is performed.

        Examples
        --------
        >>> ring = tk.formats.TRM([torch.eye(2).reshape(1, 2, 1, 2)])
        >>> matrix = ring.to_ttm()
        >>> matrix.contract_dense()
        tensor([[1., 0.],
                [0., 1.]])
        """
        return self._to_open()

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates core layouts and returns structural dimensions and ranks."""
        batch_shape = tuple(self._cores[0].shape[:self._n_batches])
        rank = []
        in_dim = []
        out_dim = []

        for site, core in enumerate(self._cores):
            if core.ndim != (self._n_batches + 4):
                raise ValueError(
                    'TRM cores should have left rank, input, right rank and '
                    'output dimensions')
            if tuple(core.shape[:self._n_batches]) != batch_shape:
                raise ValueError(
                    'All TRM cores should have the same batch shape')
            if site and (core.shape[-4] != rank[-1]):
                raise ValueError('Adjacent TRM ranks should match')
            in_dim.append(core.shape[-3])
            rank.append(core.shape[-2])
            out_dim.append(core.shape[-1])

        if self._cores[-1].shape[-2] != self._cores[0].shape[-4]:
            raise ValueError(
                'The last and first cyclic TRM ranks should match')
        return rank, batch_shape, tuple(in_dim), tuple(out_dim)
