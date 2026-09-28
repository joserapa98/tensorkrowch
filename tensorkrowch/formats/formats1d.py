"""
This script contains:

    Internal classes:
        * _VectorFormat1D
        * _MatrixFormat1D

    Public classes:
        * TensorFormat1D
        * TT, TR, TTM, TRM

    Internal functions:
        * _restore_cores
        * _from_standard_cores
        * _canonicalize_cores
        * _redistribute

    Public functions:
        * split_block

    Aliases:
        * EvaluationData
"""

import warnings
from abc import abstractmethod
from copy import copy
from math import isfinite, prod, sqrt
from numbers import Number, Real
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch

from tensorkrowch.utils import (_INTEGER_DTYPES, _validate_truncation,
                               truncated_svd)

from tensorkrowch.formats.base import (_SafeList, RoundingInfo, SampleError,
                                     BlockLayout, SplitBlock, TensorFormat)
from tensorkrowch.formats.bonds import BondFactors1D, VidalGauge
from tensorkrowch.formats.orbits import (GaugeOrbit, TensorRingOrbit,
                                       MinimalCanonicalInfo)


EvaluationData = Union[torch.Tensor, Sequence[torch.Tensor]]


def _restore_cores(cores, in_dim, out_dim, n_batches, cyclic):
    """Restores public vector/matrix endpoints from standard fused cores."""
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


def _from_standard_cores(cores, in_dim, out_dim, n_batches, cyclic):
    """Constructs a plain 1D format from standard fused cores."""
    cores = _restore_cores(cores, in_dim, out_dim, n_batches, cyclic)
    cls = (TRM if cyclic else TTM) if out_dim is not None else (
        TR if cyclic else TT)
    return cls(cores, n_batches=n_batches)


def _canonicalize_cores(cores, orth_center, renormalize):
    """Returns QR/RQ-gauged cores without modifying a format."""
    cores = list(cores)
    if not all(torch.isfinite(core).all() for core in cores):
        raise ValueError('Canonicalization requires finite cores')
    batch = cores[0].shape[:-3]
    log_scale = cores[0].real.new_zeros(batch)
    for site in range(orth_center):
        core = cores[site]
        matrix = core.reshape(*batch, -1, core.shape[-1])
        q, r = torch.linalg.qr(matrix, mode='reduced')
        if renormalize:
            scale = torch.linalg.vector_norm(r, dim=(-2, -1))
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            r = r / scale[..., None, None]
            log_scale = log_scale + scale.log()
        cores[site] = q.reshape(*batch, core.shape[-3], core.shape[-2], q.shape[-1])
        cores[site + 1] = torch.einsum('...ab,...bpr->...apr', r, cores[site + 1])
    for site in range(len(cores) - 1, orth_center, -1):
        core = cores[site]
        matrix = core.reshape(*batch, core.shape[-3], -1)
        q, r = torch.linalg.qr(matrix.transpose(-2, -1).conj(), mode='reduced')
        r = r.transpose(-2, -1).conj()
        if renormalize:
            scale = torch.linalg.vector_norm(r, dim=(-2, -1))
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            r = r / scale[..., None, None]
            log_scale = log_scale + scale.log()
        cores[site] = q.transpose(-2, -1).conj().reshape(
            *batch, q.shape[-1], core.shape[-2], core.shape[-1])
        cores[site - 1] = cores[site - 1] @ r
    if renormalize:
        cores[orth_center] = cores[orth_center] * log_scale.exp()[..., None, None, None]
    return cores


def _redistribute(cores, spectra, old_powers, powers):
    """Moves stored Schmidt powers between neighbours without another SVD."""
    for site, (spectrum, old, new) in enumerate(
        zip(spectra, old_powers, powers)):
        for neighbour, difference, left_axis in (
                (site, new[0] - old[0], False),
                (site + 1, new[1] - old[1], True)):
            if difference == 0:
                continue
            if difference < 0:
                safe = torch.where(spectrum > 0, spectrum, torch.ones_like(spectrum))
                factor = torch.where(spectrum > 0, safe.pow(difference),
                                     torch.zeros_like(spectrum))
            else:
                factor = spectrum.pow(difference)
            factor = factor[..., :, None,
                            None] if left_axis else factor[..., None, None, :]
            cores[neighbour] = cores[neighbour] * factor
    values = []
    for spectrum, (left, right) in zip(spectra, powers):
        exponent = 1 - left - right
        if exponent < 0 and torch.any(spectrum == 0):
            raise ValueError('Inverse Vidal requires nonzero Schmidt spectra')
        values.append(None if exponent == 0 else spectrum.pow(exponent))
    return cores, values


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
                _svd_callback=None):
    r"""Splits a local tensor sitewise with both external ranks preserved.

    Only internal bonds are truncated. External ranks remain unchanged, and
    structural batches share retained ranks. Inverse mode rejects retained zero
    singular values. Multiple truncation criteria select the most restrictive
    retained rank.

    Parameters
    ----------
    block : torch.Tensor
        Local tensor shaped (*core_batch, left, *physical, right). Matrix
        physical axes are interleaved by site.
    in_dim : sequence of int
        Positive physical input dimension for each local site.
    out_dim : sequence of int, optional
        Matrix output dimensions paired with in_dim. None treats the block
        as a vector format.
    n_batches : int
        Number of leading structural batch axes in block.
    rank : int, optional
        Maximum number of singular values to keep.
    cutoff : float, optional
        Minimum singular value to keep. It must be finite and non-negative.
        Singular values <= cutoff are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded
        while the accumulated sum of squares is <= atol. It must be finite
        and non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded
        while the tail sum of squares divided by the total sum of squares is
        <= rtol. It must be finite and in [0, 1].
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent
        to setting rtol = 1 - cum_percentage. It must be finite and in [0,
        1].
    mode : {"explicit", "implicit", "inverse", "left", "right"}
        Distribution of each local spectrum between its neighboring cores.
        These select powers (0, 0), (0.5, 0.5), (1, 1), (1, 0) and (0, 1),
        respectively.
    renormalize : bool
        Rescales intermediate factors to reduce numerical overflow or
        underflow and restores the accumulated scale in the final cores. The
        represented tensor retains its global scale.

    Returns
    -------
    SplitBlock
        Local standard fused cores, diagonal factors and local singular
        values. No graph or decomposition engine is constructed. Local
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
    if not in_dim or any(isinstance(dim, bool) or not isinstance(
        dim, int) or dim < 1 for dim in in_dim):
        raise ValueError('Input dimensions should be positive integers')
    if out_dim is not None:
        out_dim = tuple(out_dim)
        if len(out_dim) != len(in_dim) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in out_dim):
            raise ValueError(
                'Output dimensions should match the positive site dimensions')
    dimensions = in_dim if out_dim is None else tuple(
        dim for pair in zip(in_dim, out_dim) for dim in pair)
    if block.ndim != n_batches + \
        len(dimensions) + 2 or tuple(block.shape[n_batches + 1:-1]) != dimensions:
        raise ValueError('Block physical axes should match the requested dimensions')
    if block.shape[n_batches] < 1 or block.shape[-1] < 1:
        raise ValueError('External ranks should be positive')
    _validate_truncation(rank, cutoff, atol, rtol, cum_percentage)
    if not isinstance(renormalize, bool):
        raise TypeError('`renormalize` should be bool type')
    powers = {'explicit': (0, 0), 'implicit': (0.5, 0.5),
              'inverse': (1, 1), 'left': (1, 0), 'right': (0, 1)}
    if mode not in powers:
        raise ValueError('Invalid local bond distribution mode')
    physical = in_dim if out_dim is None else tuple(
        a * b for a, b in zip(in_dim, out_dim))
    batch = block.shape[:n_batches]
    right = block.shape[-1]
    state = block.reshape(*batch, block.shape[n_batches], *physical, right)
    cores, spectra = [], []
    for site, dimension in enumerate(physical[:-1]):
        left = state.shape[n_batches]
        matrix = state.reshape(*batch, left * dimension, -1)
        scale = matrix.real.new_ones(batch)
        local_cutoff, local_atol = cutoff, atol
        if renormalize:
            scale = matrix.abs().amax(dim=(-2, -1))
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            matrix = matrix / scale[..., None, None]
            # A common retained rank is selected across structural batches.
            if cutoff is not None:
                local_cutoff = cutoff / scale.max().item()
            if atol is not None:
                local_atol = atol / scale.max().item() ** 2
        decomposition = truncated_svd(
            matrix, rank, local_cutoff, local_atol, rtol, cum_percentage,
            return_info=_svd_callback is not None)
        u, s, vh = decomposition[:3]
        if _svd_callback is not None:
            _svd_callback(site, decomposition[3], s, scale.log())
        s = s * scale.unsqueeze(-1)
        core = u.reshape(*batch, left, dimension, s.shape[-1])
        if spectra:
            previous = spectra[-1]
            safe = torch.where(previous > 0, previous, torch.ones_like(previous))
            core = core * torch.where(previous > 0, safe.reciprocal(),
                                      torch.zeros_like(previous))[..., :, None, None]
        cores.append(core)
        spectra.append(s)
        state = s.unsqueeze(-1) * vh
    left = state.shape[-2] if spectra else block.shape[n_batches]
    core = state.reshape(*batch, left, physical[-1], right)
    if spectra:
        previous = spectra[-1]
        safe = torch.where(previous > 0, previous, torch.ones_like(previous))
        core = core * torch.where(previous > 0, safe.reciprocal(),
                                  torch.zeros_like(previous))[..., :, None, None]
    cores.append(core)
    if mode == 'inverse' and any(torch.any(spectrum == 0) for spectrum in spectra):
        raise ValueError('Inverse local bonds require nonzero retained spectra')
    cores, factors = _redistribute(cores, spectra, [(0, 0)] * len(spectra),
                                   [powers[mode]] * len(spectra))
    return SplitBlock(tuple(cores), tuple(factors), tuple(spectra))


class TensorFormat1D(TensorFormat):
    """
    Compact format for tensors with a 1D layout, formed by a sequence of cores
    and, possibly, explicit bond factors.

    Serves as a base class for all 1D chain formats, such as :class:`TT`,
    :class:`TR`, :class:`TTM`, :class:`TRM`, and their quantized versions.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores with the shapes required by the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    bonds : sequence of torch.Tensor or None, optional
        Diagonal factors between cores. The format constructs its own container
        and validates factors together with the cores; tensors are shared.
    """

    _family = 'tensor'
    _topology = 'tensor'
    _cyclic = False
    _quantized = False

    def __init__(self, cores: Sequence[torch.Tensor], n_batches: int = 0,
                 *, bonds=None) -> None:
        if isinstance(n_batches, bool) or not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        if n_batches < 0:
            raise ValueError('`n_batches` should be non-negative')

        self._n_batches = n_batches
        self._orth_center = None
        self._bonds = None
        self._replace_cores(cores, bonds)

    @property
    def cores(self):
        """Mutable, fixed-length core sequence."""
        return self._cores

    @cores.setter
    def cores(self, cores: Sequence[torch.Tensor]):
        if isinstance(cores, torch.Tensor):
            raise TypeError('`cores` should be a sequence of torch.Tensor objects')
        previous = self._cores
        self._cores = _SafeList(cores, self._on_cores_changed)
        try:
            self._on_cores_changed()
        except Exception:
            self._cores = previous
            raise

    def _replace_cores(self, cores: Sequence[torch.Tensor], bonds,
                       *, spectra=None, powers=None):
        """Replaces cores and bonds together, restoring state on invalid input."""
        if isinstance(cores, torch.Tensor):
            raise TypeError('`cores` should be a sequence of torch.Tensor objects')

        cores = list(cores)
        previous = self.__dict__.copy()
        self._cores = _SafeList(cores, self._on_cores_changed)
        try:
            if spectra is not None:
                self._bonds = VidalGauge(bonds, spectra, powers, self._on_bonds_changed)
            else:
                self._bonds = None if bonds is None else BondFactors1D(
                    bonds, self._on_bonds_changed)
            self.validate()
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(previous)
            raise

        self._orth_center = None

    def _on_cores_changed(self):
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

    def _on_bonds_changed(self):
        """Validates manual bond edits and clears the orthogonality center."""
        self.validate_bonds()
        self._orth_center = None

    @property
    def n_batches(self) -> int:
        """Number of leading structural batch axes."""
        return self._n_batches

    @abstractmethod
    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates cores and returns rank, batch, input and output dims."""

    def validate(self):
        r"""Validates cores and bonds and refreshes structural metadata.

        Called when constructing or replacing cores. Ordinary queries and
        contractions do not call it. Invalid structure raises TypeError or
        ValueError. Direct tensor shape changes remain outside the mutation
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

    def validate_bonds(self):
        r"""Checks stored diagonal factors against current core dimensions.

        Checks factor count, ranks, batch shapes, device and compatible dtype.
        Called on controlled bond replacement; it does not certify Vidal
        spectra.

        Returns
        -------
        TensorFormat1D
            The current format.
        """
        if self._bonds is not None:
            self._bonds.validate(self._raw_standard_cores(), self._cyclic)
        return self

    def _map_tensors(self, function):
        """Maps stored tensors while preserving concrete container semantics."""
        result = copy(self)
        result._cores = _SafeList([function(core) for core in self._cores],
                                  result._on_cores_changed)
        if self._bonds is not None:
            result._bonds = self._bonds._map_tensors(function, result._on_bonds_changed)
        return result

    def _same_aux_tensors(self, other):
        """Checks whether auxiliary tensor references are unchanged."""
        return True

    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           copy: bool = False):
        r"""Returns a device/dtype conversion, preserving the concrete format.

        PyTorch device errors propagate without a CPU fallback. Autograd is
        retained.

        Parameters
        ----------
        device : str or torch.device, optional
            Target device. None preserves the current device.
        dtype : torch.dtype, optional
            Target dtype. None preserves the current dtype. Coordinate grids and
            Schmidt spectra remain real when cores are complex.
        copy : bool
            If True, copies tensors even when device and dtype are unchanged. If
            False, an unchanged conversion may return self.

        Returns
        -------
        TensorFormat
            Converted format; self when no conversion is needed and copy is
            False.

        Examples
        --------
        >>> format = tk.formats.TT([torch.ones(2)])
        >>> format.to() is format
        True
        >>> format.to(dtype=torch.float64).dtype == torch.float64
        True
        """
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise TypeError('`dtype` should be torch.dtype type')
        if not isinstance(copy, bool):
            raise TypeError('`copy` should be bool type')

        result = self._map_tensors(lambda tensor: tensor.to(
            device=device, dtype=dtype, copy=copy))

        same_cores = all(
            new is old for new, old in zip(result._cores, self._cores))
        same_bonds = self._bonds is None or all(
            new is old for new, old in zip(result._bonds.values, self._bonds.values))

        if not copy and same_cores and same_bonds and self._same_aux_tensors(result):
            return self

        result.validate()
        return result

    def clone(self):
        r"""Clones the structural tensors, preserving autograd.

        Returns
        -------
        TensorFormat
            Independent tensor storage with the same represented tensor.
        """
        return self._map_tensors(lambda tensor: tensor.clone())

    def detach(self):
        r"""Returns a detached format sharing tensor storage.

        Returns
        -------
        TensorFormat
            Separate containers with detached tensor references. Value edits to
            shared storage affect both formats.
        """
        return self._map_tensors(lambda tensor: tensor.detach())

    def detach_(self):
        r"""Detaches structural tensors in-place by replacing references.

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

    @abstractmethod
    def _raw_standard_cores(self) -> List[torch.Tensor]:
        """Returns (*batch, left, physical, right) cores without bond factors."""

    def _standard_cores(self):
        """Returns standard fused cores including stored diagonal factors."""
        cores = self._raw_standard_cores()
        if self._bonds is not None:
            cores = list(cores)
            for site, value in enumerate(self._bonds.values):
                if value is not None:
                    cores[site] = cores[site] * value[..., None, None, :]
        return cores

    @property
    def bonds(self):
        """Optional diagonal factors, with one entry per virtual bond."""
        return self._bonds

    @bonds.setter
    def bonds(self, value):
        r"""Assigns compatible diagonal factors to the format.

        Factor count, ranks, batches and runtime are checked before acceptance.

        Parameters
        ----------
        value : sequence of torch.Tensor or None
            Diagonal factors to store, or None to remove them. A separate container is
            owned by this format while tensor references are shared.
        """
        bonds = None if value is None else BondFactors1D(value, self._on_bonds_changed)
        if bonds is not None:
            bonds.validate(self._raw_standard_cores(), self._cyclic)
        self._bonds = bonds
        self._orth_center = None

    def materialize_bonds(self, orth_center: Optional[int] = None):
        r"""Absorbs stored factors into the cores in-place.

        A valid Vidal gauge redistributes its spectra towards the selected
        center, accounting for previously absorbed powers, including inverse
        Vidal. Generic or invalidated factors are absorbed once into the
        adjacent core. A mixed canonical form is obtained only when the initial
        Vidal gauge is valid.

        Parameters
        ----------
        orth_center : int, optional
            Orthogonality center in [0, n_sites - 1]. None selects the last
            site.

        Returns
        -------
        TensorFormat1D
            The current format, with bonds set to None.

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
        orth_center = self.n_sites - 1 if orth_center is None else orth_center
        if isinstance(orth_center, bool) or not isinstance(orth_center, int):
            raise TypeError('`orth_center` should be int type or None')
        if not 0 <= orth_center < self.n_sites:
            raise ValueError('`orth_center` should select a valid site')
        if self._bonds is None:
            return self
        cores = list(self._raw_standard_cores())
        if isinstance(self._bonds, VidalGauge) and self._bonds._valid:
            powers = [(0, 1) if site < orth_center else (1, 0)
                      for site in range(len(self._bonds.spectra))]
            cores, _ = _redistribute(cores, self._bonds.spectra, self._bonds.powers, powers)
            self._set_standard_cores(cores)
            return self
        for site, value in enumerate(self._bonds.values):
            if value is None:
                continue
            if site >= orth_center:
                cores[site] = cores[site] * value[..., None, None, :]
            else:
                neighbour = (site + 1) % len(cores)
                cores[neighbour] = cores[neighbour] * value[..., :, None, None]
        self._set_standard_cores(cores)
        return self

    def _set_standard_cores(self, cores: Sequence[torch.Tensor], bonds=None,
                            in_dim=None, out_dim=None, *, spectra=None, powers=None):
        """Restores core layouts and publishes cores and factors together."""
        in_dim = self._in_dim if in_dim is None else in_dim
        out_dim = self._out_dim if out_dim is None else out_dim
        cores = _restore_cores(cores, in_dim, out_dim, self._n_batches, self._cyclic)
        self._replace_cores(cores, bonds, spectra=spectra, powers=powers)

    def canonicalize(self,
                     orth_center: Optional[int] = None,
                     renormalize: bool = False):
        r"""Performs QR/RQ sweeps in-place without truncation.

        Open chains become left-isometric before the center and right-isometric
        after it. For rings, this is a local gauge relative to the stored cut,
        without a global Schmidt interpretation. Stored factors are materialized
        before sweeping.

        Parameters
        ----------
        orth_center : int, optional
            Orthogonality center in [0, n_sites - 1]. None selects the last
            site.
        renormalize : bool
            Rescales intermediate factors to reduce numerical overflow or
            underflow and restores the accumulated scale in the final cores. The
            represented tensor retains its global scale.

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
        orth_center = self.n_sites - 1 if orth_center is None else orth_center
        if isinstance(orth_center, bool) or not isinstance(orth_center, int):
            raise TypeError('`orth_center` should be int type or None')
        if not 0 <= orth_center < self.n_sites:
            raise ValueError('`orth_center` should select a valid site')
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        cores = _canonicalize_cores(
            self._standard_cores(), orth_center, renormalize)
        self._set_standard_cores(cores)
        self._orth_center = orth_center
        return self

    def canonicalize_vidal(self,
                           mode: str = 'implicit',
                           inverse_positions: Optional[Sequence[int]] = None,
                           remaining_mode: str = 'implicit',
                           inverse_cutoff: float = 0.0):
        r"""Builds or redistributes an open-chain Vidal gauge in-place.

        Only TT/TTM admit this global Schmidt representation. Invalidated gauges
        are recomputed; valid gauges can be redistributed without another SVD.
        No truncation is performed to manufacture an inverse.

        Parameters
        ----------
        mode : {"implicit", "explicit", "inverse"}
            Absorbs spectrum powers (0.5, 0.5), (0, 0), or (1, 1) into
            neighboring cores, respectively. The remaining diagonal factor has
            power 1 minus their sum.
        inverse_positions : sequence of int, optional
            Distinct virtual bonds to use in inverse form. Cannot be combined
            with mode="inverse". When supplied, remaining_mode controls every
            other bond.
        remaining_mode : {"implicit", "explicit"}
            Distribution for bonds not selected by inverse_positions.
        inverse_cutoff : float
            Finite non-negative threshold. Every spectrum value used in an
            inverse should be strictly greater than this value; values are not
            truncated to create an inverse.

        Returns
        -------
        TensorFormat1D
            The current format with a valid VidalGauge.

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
        if self._cyclic:
            raise ValueError('Global Vidal canonicalization requires an open chain')
        modes = {'explicit': (0, 0), 'implicit': (0.5, 0.5), 'inverse': (1, 1)}
        if mode not in modes or remaining_mode not in ('explicit', 'implicit'):
            raise ValueError('Invalid Vidal mode or remaining_mode')
        if isinstance(inverse_cutoff, bool) or not isinstance(inverse_cutoff, Real):
            raise TypeError('`inverse_cutoff` should be a real number')
        if not isfinite(inverse_cutoff) or inverse_cutoff < 0:
            raise ValueError('`inverse_cutoff` should be finite and non-negative')
        count = self.n_sites - 1
        if inverse_positions is None:
            positions = set(range(count)) if mode == 'inverse' else set()
            powers = [modes[mode]] * count
        else:
            positions_list = list(inverse_positions)
            if mode == 'inverse':
                raise ValueError('Select either mode="inverse" or inverse_positions')
            if any(isinstance(site, bool) or not isinstance(site, int)
                   for site in positions_list):
                raise TypeError('Inverse bond positions should be integers')
            if len(set(positions_list)) != len(positions_list) or any(
                    site < 0 or site >= count for site in positions_list):
                raise ValueError('Inverse bond positions should be distinct valid bonds')
            positions = set(positions_list)
            powers = [modes['inverse' if site in positions else remaining_mode]
                      for site in range(count)]
        if isinstance(self._bonds, VidalGauge) and self._bonds._valid:
            cores = list(self._raw_standard_cores())
            spectra = self._bonds.spectra
            old_powers = self._bonds.powers
        else:
            cores = _canonicalize_cores(self._standard_cores(), 0, False)
            spectra = []
            batch = self._batch_shape
            for site in range(count):
                core = cores[site]
                u, s, vh = truncated_svd(core.reshape(*batch, -1, core.shape[-1]))
                cores[site] = u.reshape(*batch, core.shape[-3], core.shape[-2], s.shape[-1])
                if site:
                    previous = spectra[-1]
                    safe = torch.where(previous > 0, previous, torch.ones_like(previous))
                    inverse = torch.where(previous > 0, safe.reciprocal(),
                                          torch.zeros_like(previous))
                    cores[site] = cores[site] * inverse[..., :, None, None]
                spectra.append(s)
                cores[site + 1] = torch.einsum('...ab,...bpr->...apr',
                                               s.unsqueeze(-1) * vh, cores[site + 1])
            if count:
                last = spectra[-1]
                safe = torch.where(last > 0, last, torch.ones_like(last))
                inverse = torch.where(last > 0, safe.reciprocal(), torch.zeros_like(last))
                cores[-1] = cores[-1] * inverse[..., :, None, None]
            old_powers = [(0, 0)] * count
        for site in positions:
            if torch.any(spectra[site] <= inverse_cutoff):
                raise ValueError(
                    f'Inverse Vidal bond {site} has values at or below inverse_cutoff')
        cores, factors = _redistribute(cores, spectra, old_powers, powers)
        self._set_standard_cores(cores, factors, spectra=spectra, powers=powers)
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
                 return_info: bool = False):
        r"""Compresses bond ranks in-place with one QR/SVD execution.

        Open chains use left QR followed by sitewise right SVD. Rings use
        Algorithm 4 of Mickelin and Karaman, `On Algorithms for and Computing
        with the Tensor Ring Decomposition <https://arxiv.org/pdf/1807.02513>`_,
        including closure reduction. Ring ranks need not become minimal, especially after
        block-diagonal sums or products. Multiple criteria use the most
        restrictive retained rank.

        Parameters
        ----------
        rank : int, optional
            Maximum number of singular values to keep.
        cutoff : float, optional
            Minimum singular value to keep. It must be finite and non-negative.
            Singular values <= cutoff are removed.
        atol : float, optional
            Absolute tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded
            while the accumulated sum of squares is <= atol. It must be finite
            and non-negative.
        rtol : float, optional
            Relative tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded
            while the tail sum of squares divided by the total sum of squares is
            <= rtol. It must be finite and in [0, 1].
        cum_percentage : float, optional
            Minimum fraction of squared singular-value mass to keep. Equivalent
            to setting rtol = 1 - cum_percentage. It must be finite and in [0,
            1].
        renormalize : bool
            Rescales intermediate factors to reduce numerical overflow or
            underflow and restores the accumulated scale in the final cores. The
            represented tensor retains its global scale.
        rel_error : float, optional
            Finite non-negative relative Frobenius error budget. It is
            distributed over local cuts. Other truncation constraints can exceed
            it and then emit a warning.
        return_info : bool
            If True, returns the format together with the operation-specific
            information record.

        Returns
        -------
        TensorFormat1D or tuple[TensorFormat1D, RoundingInfo]
            The current format, optionally with discarded energies and an
            absolute error bound. The bound is not a measured reconstruction
            error.

        Examples
        --------
        >>> format = tk.formats.TT([
        ...     torch.diag(torch.tensor([4., 1.])), torch.eye(2)])
        >>> _, info = format.rounding(rank=1, return_info=True)
        >>> info.rank
        (1,)
        >>> torch.allclose(format.contract_dense(), torch.diag(torch.tensor([4., 0.])))
        True
        """
        _validate_truncation(rank, cutoff, atol, rtol, cum_percentage)
        for name, value in [('renormalize', renormalize), ('return_info', return_info)]:
            if not isinstance(value, bool):
                raise TypeError(f'`{name}` should be bool type')
        if rel_error is not None:
            if isinstance(rel_error, bool) or not isinstance(rel_error, Real):
                raise TypeError('`rel_error` should be a real number')
            if not isfinite(rel_error) or rel_error < 0:
                raise ValueError('`rel_error` should be finite and non-negative')
        cyclic = self._cyclic
        closing = self._raw_standard_cores()[0].shape[-3] if cyclic else 1
        norm = self.norm() if rel_error is not None else None
        cores = _canonicalize_cores(
            self._standard_cores(), self.n_sites - 1, renormalize)
        batch = self._batch_shape
        cuts = len(cores) if cyclic else max(1, len(cores) - 1)
        delta = rel_error * norm / sqrt(cuts * closing) if norm is not None else None
        records = []
        discarded_norms = []
        collect = return_info or rel_error is not None

        def split(matrix):
            """Truncates a scaled matrix and collects its discarded mass."""
            scale = matrix.abs().amax()
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            limit = torch.finfo(matrix.real.dtype).max
            local_cutoff = None if cutoff is None else min(cutoff / scale.item(), limit)
            local_atol = None if atol is None else min(
                atol / scale.item() / scale.item(), limit)
            if delta is not None:
                value = (delta / scale).square().min().item()
                local_atol = min(value, limit) if local_atol is None else min(
                    local_atol, value)
            result = truncated_svd(matrix / scale, rank=rank, cutoff=local_cutoff, atol=local_atol,
                                   rtol=rtol, cum_percentage=cum_percentage,
                                   return_info=collect)
            if collect:
                discarded = result[3].discarded_sq_norm.sqrt() * scale
                discarded_norms.append(discarded)
                records.append(discarded.square())
            return result[0], result[1] * scale, result[2]

        if cyclic and len(cores) == 1:
            value = cores[0].diagonal(dim1=-3, dim2=-1).sum(-1)
            cores = [value.unsqueeze(-2).unsqueeze(-1)]
        elif cyclic:
            core = cores[-1]
            q, r = torch.linalg.qr(core.reshape(*batch, -1, core.shape[-1]), mode='reduced')
            u, s, vh = split(r)
            if s.shape[-1] < core.shape[-1]:
                cores[-1] = ((q @ u) * s.unsqueeze(-2)).reshape(
                    *batch, core.shape[-3], core.shape[-2], s.shape[-1])
                cores[0] = torch.einsum('...ab,...bpr->...apr', vh, cores[0])
        for site in range(len(cores) - 1, 0, -1):
            core = cores[site]
            u, s, vh = split(core.reshape(*batch, core.shape[-3], -1))
            cores[site] = vh.reshape(*batch, s.shape[-1], core.shape[-2], core.shape[-1])
            cores[site - 1] = cores[site - 1] @ (u * s.unsqueeze(-2))
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
                bound = self.cores[0].real.new_zeros(batch)
            if rel_error is not None:
                satisfied = bool(torch.all(bound <= rel_error * norm +
                                           10 * torch.finfo(norm.dtype).eps * norm))
                if not satisfied:
                    warnings.warn('Truncation constraints exceed the requested global error budget',
                                  UserWarning, stacklevel=2)
            if return_info:
                return self, RoundingInfo(
                    tuple(self.rank), tuple(records), bound, satisfied)
        return self

    def canonicalize_minimal(self, max_iter: int = 200, lr: float = 0.05,
                             tol: float = 1e-8, *, return_info: bool = False):
        r"""Uses implicit Vidal for trains or experimental gauge balancing for
        rings.

        Ring optimization minimizes half the sum of squared core norms using
        Hermitian exponential gauges and retains the best finite iterate. It may
        stop without convergence and does not certify a global minimum. Batched
        rings share one gauge per bond. Input core gradients are not accumulated
        by gauge optimization. The objective is inspired by `The minimal
        canonical form of a tensor network <https://arxiv.org/pdf/2209.14358>`_.

        Parameters
        ----------
        max_iter : int
            Positive maximum number of gauge optimization iterations for a ring.
        lr : float
            Finite positive learning rate of the Adam gauge optimizer.
        tol : float
            Finite positive stopping tolerance for the maximum absolute gauge-
            parameter gradient.
        return_info : bool
            If True, returns the format together with the operation-specific
            information record.

        Returns
        -------
        TensorFormat1D or tuple[TensorFormat1D, MinimalCanonicalInfo]
            The current format, optionally with iteration count, convergence
            status and final ring Gram imbalance. Trains report zero iterations
            and no imbalance.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> _, info = format.canonicalize_minimal(return_info=True)
        >>> (info.iterations, info.converged)
        (0, True)
        >>> torch.allclose(format.contract_dense(), torch.eye(2))
        True
        """
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
        if not self._cyclic:
            self.canonicalize_vidal('implicit')
            return (self, MinimalCanonicalInfo(
                0, True, None)) if return_info else self
        orbit = TensorRingOrbit(self)
        if not all(torch.isfinite(core).all() for core in orbit.cores):
            raise ValueError('Minimal canonicalization requires finite cores')
        detached = GaugeOrbit([core.detach() for core in orbit.cores], orbit.bonds)
        best = [torch.eye(rank, dtype=self.dtype, device=self.device)
                for rank in self._rank]
        scale = max(core.abs().amax().item() for core in detached.cores)
        if scale == 0:
            info = MinimalCanonicalInfo(0, True, self.cores[0].real.new_zeros(()))
            return (self, info) if return_info else self
        detached.cores = tuple(core / scale for core in detached.cores)
        best_loss = detached.objective(best).item()
        converged, iterations = False, 0
        with torch.enable_grad():
            parameters = [torch.zeros_like(gauge, requires_grad=True) for gauge in best]
            optimizer = torch.optim.Adam(parameters, lr=lr)
            for _ in range(max_iter):
                iterations += 1
                optimizer.zero_grad()
                gauges = [torch.matrix_exp((parameter + parameter.transpose(-2, -1).conj()) / 2)
                          for parameter in parameters]
                if not all(torch.isfinite(gauge).all() for gauge in gauges):
                    break
                try:
                    loss = detached.objective(gauges)
                except torch.linalg.LinAlgError:
                    break
                if not torch.isfinite(loss):
                    break
                value = loss.item()
                if value < best_loss:
                    best_loss = value
                    best = [gauge.detach() for gauge in gauges]
                loss.backward()
                if not all(torch.isfinite(parameter.grad).all()
                           for parameter in parameters):
                    break
                if max(parameter.grad.abs().amax().item()
                       for parameter in parameters) <= tol:
                    converged = True
                    break
                optimizer.step()
        self._set_standard_cores(orbit.apply(best))
        if return_info:
            info = MinimalCanonicalInfo(iterations, converged,
                                        TensorRingOrbit(self).balance_residual())
            return self, info
        return self

    def block(self, groups: Sequence[int]) -> BlockLayout:
        r"""Contracts consecutive groups into effective sites in-place.

        Parameters
        ----------
        groups : sequence of int
            Positive group sizes whose sum is n_sites. Physical dimensions are
            multiplied within each group; matrix inputs and outputs stay separate.
            Internal factors are absorbed and factors between groups are retained.

        Returns
        -------
        BlockLayout
            Original dimensions and group sizes. Pass it to unblock() on this
            format or a solver result with the same effective dimensions.
            Clone the format first to retain its original structure. Quantics
            layouts must remain compatible; use as_tt()/as_tr() or as_ttm()/as_trm()
            to group arbitrary digits in a plain format.

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
            size, int) or size < 1 for size in groups):
            raise ValueError('Block sizes should be positive integers')
        if sum(groups) != self.n_sites:
            raise ValueError('Block sizes should sum to the number of sites')
        layout = BlockLayout(groups, self._in_dim, self._out_dim)
        cores, in_dim, out_dim, factors = [], [], [], []
        first = 0
        for size in groups:
            last = first + size - 1
            value = self.contract_block(first, last)
            in_dim.append(prod(self._in_dim[first:last + 1]))
            if self._out_dim is not None:
                out_dim.append(prod(self._out_dim[first:last + 1]))
                b = self._n_batches
                order = [*range(b + 1), *range(b + 1, b + 1 + 2 * size, 2),
                         *range(b + 2, b + 1 + 2 * size, 2), value.ndim - 1]
                value = value.permute(order)
            cores.append(value.reshape(*self._batch_shape, value.shape[self._n_batches],
                                       -1, value.shape[-1]))
            if self._bonds is not None and last < len(self._bonds.values):
                factors.append(self._bonds.values[last])
            first = last + 1
        bonds = factors if factors else None
        self._set_standard_cores(
            cores, bonds, in_dim=tuple(in_dim),
            out_dim=tuple(out_dim) if out_dim else None)
        return layout

    def unblock(self, layout: BlockLayout, **kwargs):
        r"""Restores the sites described by a blocking layout in-place.

        Parameters
        ----------
        layout : BlockLayout
            Original dimensions and group sizes returned by block(). Its grouped
            dimensions must match the current format. It can describe another
            object, such as an initial guess used to produce a solver result.
        **kwargs : keyword arguments
            Options passed to split_block(): rank, cutoff, atol, rtol,
            cum_percentage, mode and renormalize. Only bonds inside groups are
            truncated; ranks between groups remain unchanged.

        Returns
        -------
        TensorFormat1D
            The current format. Without truncation the represented tensor is
            recovered up to numerical precision; original core gauges may differ.
            Invalid input leaves the current format unchanged.
        """
        if not isinstance(layout, BlockLayout) or len(layout.groups) != self.n_sites:
            raise ValueError('Unblocking requires a matching BlockLayout')
        if (self._out_dim is None) != (layout.out_dim is None):
            raise ValueError('Blocked vector/matrix family should match the layout')
        cores, first = [], 0
        standard = self._standard_cores()
        for site, size in enumerate(layout.groups):
            value = standard[site]
            inputs = layout.in_dim[first:first + size]
            outputs = None if layout.out_dim is None else layout.out_dim[first:first + size]
            if self._in_dim[site] != prod(inputs) or (
                    outputs is not None and self._out_dim[site] != prod(outputs)):
                raise ValueError('Blocked dimensions should match the layout')
            dimensions = inputs
            if outputs is not None:
                value = value.reshape(*self._batch_shape,
                                      value.shape[-3], *inputs, *outputs, value.shape[-1])
                b = self._n_batches
                order = [*range(b + 1)]
                for index in range(size):
                    order.extend([b + 1 + index, b + 1 + size + index])
                order.append(value.ndim - 1)
                value = value.permute(order)
                dimensions = tuple(dim for pair in zip(inputs, outputs) for dim in pair)
            value = value.reshape(*self._batch_shape, value.shape[self._n_batches],
                                  *dimensions, value.shape[-1])
            local = split_block(value, inputs, outputs, self._n_batches, **kwargs)
            # Restore the local representation with each factor absorbed once.
            for index, core in enumerate(local.cores):
                factor = local.bonds[index] if index < len(
                    local.bonds) else None
                cores.append(core if factor is None else core * factor[..., None, None, :])
            first += size
        self._set_standard_cores(cores, in_dim=layout.in_dim, out_dim=layout.out_dim)
        return self

    def contract_block(self, first, last):
        r"""Contracts a contiguous region with both external ranks left open.

        Parameters
        ----------
        first : int
            First site of the region, included.
        last : int
            Last site of the region, included. Should be at least first.

        Returns
        -------
        torch.Tensor
            Tensor with shape (*core_batch, left, *physical, right). Matrix
            physical axes are interleaved. Internal factors are included and
            external factors excluded.
        """
        for site in (first, last):
            if isinstance(site, bool) or not isinstance(site, int):
                raise TypeError('Block endpoints should be integers')
        if not 0 <= first <= last < self.n_sites:
            raise ValueError('Block endpoints should select an ordered contiguous region')
        cores = self._raw_standard_cores()
        result = cores[first]
        dimensions = [cores[first].shape[-2]]
        batch = self._batch_shape
        left = result.shape[-3]
        for site in range(first, last):
            if self._bonds is not None and self._bonds.values[site] is not None:
                result = result * self._bonds.values[site][..., None, None, :]
            result = result.reshape(*batch, left, -1, result.shape[-1])
            result = torch.einsum('...apr,...rqb->...apqb', result, cores[site + 1])
            dimensions.append(cores[site + 1].shape[-2])
        if self._out_dim is not None:
            dimensions = [dim for pair in zip(self._in_dim[first:last + 1],
                                              self._out_dim[first:last + 1]) for dim in pair]
        return result.reshape(*batch, left, *dimensions, cores[last].shape[-1])

    def replace_cores(self, first: int, cores: Sequence[torch.Tensor],
                      bonds: Optional[Sequence[Optional[torch.Tensor]]] = None):
        r"""Installs consecutive standard cores and their internal factors in-place.

        Parameters
        ----------
        first : int
            First site to replace. The supplied core count determines the final
            site; the total number of sites remains unchanged.
        cores : sequence of torch.Tensor
            Standard cores shaped (*batch, left, physical, right), with matrix
            physical axes fused. Physical dimensions and external ranks must
            match the selected region; its internal ranks may change.
        bonds : sequence of torch.Tensor or None, optional
            New factors between replacement cores. None uses identity factors.
            Factors outside the region are retained. Cores and factors are
            installed together, validating the complete candidate once.

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
        cores = tuple(cores)
        if not cores:
            raise ValueError('Replacement cores should not be empty')
        last = first + len(cores) - 1
        if not 0 <= first <= last < self.n_sites:
            raise ValueError('Replacement sites should lie inside the format')
        if isinstance(bonds, torch.Tensor):
            raise TypeError('Replacement bonds should be a sequence or None')
        for offset, core in enumerate(cores):
            if not isinstance(core, torch.Tensor):
                raise TypeError('Replacement cores should be tensors')
            site = first + offset
            dimension = self._in_dim[site] * \
                (self._out_dim[site] if self._out_dim else 1)
            if core.ndim != self._n_batches + 3 or core.shape[-2] != dimension:
                raise ValueError(
                    'Replacement physical dimensions should match the selected sites')
        stored = list(self._raw_standard_cores())
        if cores[0].shape[-3] != stored[first].shape[-3] or \
                cores[-1].shape[-1] != stored[last].shape[-1]:
            raise ValueError('Replacement should preserve external ranks')
        stored[first:last + 1] = cores
        count = self.n_sites if self._cyclic else self.n_sites - 1
        factors = list(self._bonds.values) if self._bonds is not None else [
            None] * count
        values = [None] * (last - first) if bonds is None else list(bonds)
        if len(values) != last - first:
            raise ValueError('Replacement factors should match its internal bonds')
        factors[first:last] = values
        factors = factors if any(
            value is not None for value in factors) else None
        self._set_standard_cores(stored, factors)
        return self

    def absorb_bond(self, bond, side: str = 'left'):
        r"""Absorbs one diagonal factor into a neighboring core in-place.

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
        count = self.n_sites if self._cyclic else self.n_sites - 1
        if isinstance(bond, bool) or not isinstance(bond, int):
            raise TypeError('`bond` should be int type')
        if not 0 <= bond < count:
            raise ValueError('`bond` should select a valid virtual bond')
        if side not in ('left', 'right'):
            raise ValueError('`side` should be "left" or "right"')
        if self._bonds is None:
            return self
        cores = list(self._raw_standard_cores())
        spectra = powers = None
        if isinstance(self._bonds, VidalGauge) and self._bonds._valid:
            spectra = self._bonds.spectra
            powers = list(self._bonds.powers)
            powers[bond] = (1, 0) if side == 'left' else (0, 1)
            cores, factors = _redistribute(cores, self._bonds.spectra, self._bonds.powers, powers)
        else:
            factors = list(self._bonds.values)
            value = factors[bond]
            if value is not None:
                site = bond if side == 'left' else (bond + 1) % self.n_sites
                cores[site] = cores[site] * (value[..., None, None, :] if side == 'left'
                                             else value[..., :, None, None])
                factors[bond] = None
        self._set_standard_cores(cores, factors, spectra=spectra, powers=powers)
        return self

    def redistribute_bond(self, bond: int, mode: str = 'implicit',
                          inverse_cutoff: float = 0.0):
        r"""Redistributes one valid Vidal spectrum without another SVD.

        Requires a valid stored Vidal gauge. Manual edits invalidate it;
        recompute canonicalize_vidal() before redistributing spectra.

        Parameters
        ----------
        bond : int
            Index of the right virtual bond of a core. The last bond closes a
            cyclic format.
        mode : {"implicit", "explicit", "inverse", "left", "right"}
            New powers absorbed into the left/right neighbors: (0.5, 0.5), (0,
            0), (1, 1), (1, 0) or (0, 1), respectively.
        inverse_cutoff : float
            Finite non-negative threshold. Every spectrum value used in an
            inverse should be strictly greater than this value; values are not
            truncated to create an inverse.

        Returns
        -------
        TensorFormat1D
            The current format. Other bonds retain their distribution.
        """
        if isinstance(bond, bool) or not isinstance(bond, int):
            raise TypeError('`bond` should be int type')
        if not isinstance(self._bonds, VidalGauge) or not self._bonds._valid:
            raise ValueError('Bond redistribution requires valid stored Vidal spectra')
        if not 0 <= bond < len(self._bonds.spectra):
            raise ValueError('`bond` should select a valid virtual bond')
        modes = {'explicit': (0, 0), 'implicit': (0.5, 0.5),
                 'inverse': (1, 1), 'left': (1, 0), 'right': (0, 1)}
        if mode not in modes:
            raise ValueError('Invalid Vidal bond distribution mode')
        if isinstance(inverse_cutoff, bool) or not isinstance(inverse_cutoff, Real):
            raise TypeError('`inverse_cutoff` should be a real number')
        if not isfinite(inverse_cutoff) or inverse_cutoff < 0:
            raise ValueError('`inverse_cutoff` should be finite and non-negative')
        if mode == 'inverse' and torch.any(self._bonds.spectra[bond] <= inverse_cutoff):
            raise ValueError(f'Bond {bond} has non-invertible retained spectrum')
        powers = list(self._bonds.powers)
        powers[bond] = modes[mode]
        cores, factors = _redistribute(
            self._raw_standard_cores(), self._bonds.spectra, self._bonds.powers, powers)
        self._set_standard_cores(cores, factors, spectra=self._bonds.spectra, powers=powers)
        return self

    def _rotate(self, first):
        """Rotates cyclic cores, dimensions and factors to the requested cut."""
        if not self._cyclic:
            raise ValueError('Rotation requires a cyclic format')
        if isinstance(first, bool) or not isinstance(first, int):
            raise TypeError('`first` should be int type')
        if first < 0 or first >= self.n_sites:
            raise ValueError('`first` should select a valid site')
        order = [*range(first, self.n_sites), *range(first)]
        cores = self._raw_standard_cores()
        in_dim = tuple(self._in_dim[site] for site in order)
        out_dim = None if self._out_dim is None else tuple(
            self._out_dim[site] for site in order)
        result = self._new_from_standard_cores(
            [cores[site] for site in order], in_dim, out_dim,
            self._n_batches, True)
        if self._bonds is not None:
            result.bonds = [self._bonds.values[site] for site in order]
        return result

    def _to_open(self):
        """Carries the closing rank through identity factors to an open chain."""
        if not self._cyclic:
            raise ValueError('Conversion requires a cyclic format')
        cores = self._standard_cores()
        batch = self._batch_shape
        closing = cores[0].shape[-3]
        if len(cores) == 1:
            result = [cores[0].diagonal(
                dim1=-3, dim2=-1).sum(-1).unsqueeze(-2).unsqueeze(-1)]
        else:
            first = cores[0].transpose(-3, -2).reshape(*batch, 1, cores[0].shape[-2], -1)
            result = [first]
            identity = torch.eye(closing, dtype=self.dtype, device=self.device)
            for core in cores[1:-1]:
                combined = torch.einsum('st,...aib->...saitb', identity, core)
                result.append(combined.reshape(*batch, closing * core.shape[-3],
                                               core.shape[-2], closing * core.shape[-1]))
            last = cores[-1].movedim(-1, -3)
            result.append(last.reshape(*batch, closing * cores[-1].shape[-3],
                                       cores[-1].shape[-2], 1))
        return self._new_from_standard_cores(result, self._in_dim, self._out_dim,
                              self._n_batches, False)

    def _check_semantics(self, other, product=False):
        """Checks whether operands share their coordinate interpretation."""
        if self._quantized != other._quantized:
            raise ValueError(
                'Quantics algebra requires compatible coordinate semantics; '
                'use as_tt/as_tr/as_ttm/as_trm explicitly')

    def _new_from_standard_cores(self, cores, in_dim, out_dim, n_batches,
                                 cyclic, other=None, product=False,
                                 transpose=False):
        """Builds an algebra result with the operand coordinate semantics."""
        return _from_standard_cores(cores, in_dim, out_dim, n_batches, cyclic)

    def _binary_inputs(self, other, same_family=True):
        """Prepares compatible standard cores and structural batches."""
        if not isinstance(other, TensorFormat1D):
            raise TypeError('`other` should be TensorFormat1D type')
        self._check_semantics(other, product=not same_family)
        if self.n_sites != other.n_sites:
            raise ValueError('Formats should have the same number of sites')
        if self.device != other.device:
            raise ValueError('Formats should share device')
        if self._batch_shape and other._batch_shape and self._batch_shape != other._batch_shape:
            raise ValueError(
                'Structural batches should match or one operand should be unbatched')
        if same_family and (
            self._in_dim != other._in_dim or self._out_dim != other._out_dim):
            raise ValueError('Formats should have matching input and output dimensions')
        dtype = torch.promote_types(self.dtype, other.dtype)
        a = [core.to(dtype=dtype) for core in self._standard_cores()]
        b = [core.to(dtype=dtype) for core in other._standard_cores()]
        batch = self._batch_shape or other._batch_shape
        a = [core.expand(*batch, *core.shape[-3:]) for core in a]
        b = [core.expand(*batch, *core.shape[-3:]) for core in b]
        cyclic = self._cyclic or other._cyclic
        return a, b, batch, cyclic

    def _sum(self, other, method, coefficient):
        """Builds an exact stacked or block-diagonal sum or difference."""
        if method not in ('stacked', 'block_diagonal'):
            raise ValueError('`method` should be "stacked" or "block_diagonal"')
        a, b, batch, cyclic = self._binary_inputs(other)
        b[0] = b[0] * coefficient
        if len(a) == 1:
            value = a[0].diagonal(dim1=-3, dim2=-1).sum(-1) + \
                b[0].diagonal(dim1=-3, dim2=-1).sum(-1)
            cores = [value.unsqueeze(-2).unsqueeze(-1)]
        else:
            if method == 'stacked' or not cyclic:
                closing = max(a[0].shape[-3], b[0].shape[-3])
                for group in (a, b):
                    start, end = group[0], group[-1]
                    padded = start.new_zeros(*batch, closing, *start.shape[-2:])
                    padded[..., :start.shape[-3], :, :] = start
                    group[0] = padded
                    padded = end.new_zeros(*batch, *end.shape[-3:-1], closing)
                    padded[..., :end.shape[-1]] = end
                    group[-1] = padded
            cores = []
            for site, (x, y) in enumerate(zip(a, b)):
                if (method == 'stacked' or not cyclic) and site == 0:
                    core = torch.cat((x, y), dim=-1)
                elif (method == 'stacked' or not cyclic) and site == len(a) - 1:
                    core = torch.cat((x, y), dim=-3)
                else:
                    core = x.new_zeros(*batch, x.shape[-3] + y.shape[-3],
                                       x.shape[-2], x.shape[-1] + y.shape[-1])
                    core[..., :x.shape[-3], :, :x.shape[-1]] = x
                    core[..., x.shape[-3]:, :, x.shape[-1]:] = y
                cores.append(core)

        return self._new_from_standard_cores(
            cores, self._in_dim, self._out_dim, len(batch), cyclic, other=other)

    def add(self, other, method='stacked'):
        r"""Returns the exact sum of compatible formats.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with compatible local input/output dimensions,
            structural batches and device.
        method : {"stacked", "block_diagonal"}
            Cyclic sum construction. Stacked endpoints favor subsequent
            compression; block_diagonal uses the usual construction at every
            site. For open chains both choices use the ordinary endpoint
            construction.

        Returns
        -------
        TensorFormat1D
            New format without implicit truncation. Either cyclic operand
            produces a cyclic result. Inputs are unchanged.
        """
        return self._sum(other, method, 1)

    def sub(self, other, method='stacked'):
        r"""Returns the exact difference of compatible formats.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with compatible local input/output dimensions,
            structural batches and device.
        method : {"stacked", "block_diagonal"}
            Cyclic sum construction. Stacked endpoints favor subsequent
            compression; block_diagonal uses the usual construction at every
            site. For open chains both choices use the ordinary endpoint
            construction.

        Returns
        -------
        TensorFormat1D
            New format without implicit truncation. Either cyclic operand
            produces a cyclic result. Inputs are unchanged.
        """
        return self._sum(other, method, -1)

    def hadamard(self, other):
        r"""Returns an exact element-wise product.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with compatible local input/output dimensions,
            structural batches and device.

        Returns
        -------
        TensorFormat1D
            New format with product bond ranks, before any explicit rounding. A
            cyclic operand produces a cyclic result.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> torch.equal((format * format).contract_dense(), torch.eye(2))
        True
        """
        a, b, batch, cyclic = self._binary_inputs(other)
        cores = []
        for x, y in zip(a, b):
            core = torch.einsum('...lpr,...aps->...laprs', x, y)
            cores.append(core.reshape(*batch, x.shape[-3] * y.shape[-3],
                                      x.shape[-2], x.shape[-1] * y.shape[-1]))

        return self._new_from_standard_cores(
            cores, self._in_dim, self._out_dim, len(batch), cyclic, other=other)

    def __add__(self, other):
        """Returns the exact sum with another format."""
        return self.add(other)

    def __sub__(self, other):
        """Returns the exact difference with another format."""
        return self.sub(other)

    def __neg__(self):
        """Returns the format with its represented tensor negated."""
        return self * -1

    def __mul__(self, other):
        """Returns a Hadamard product or scalar-scaled format."""
        if isinstance(other, TensorFormat1D):
            return self.hadamard(other)
        if isinstance(other, torch.Tensor):
            if other.ndim != 0:
                raise ValueError('The scaling tensor should be scalar')
            if other.device != self.device:
                raise ValueError('The scaling tensor should share the format device')
        elif isinstance(other, bool) or not isinstance(other, Number):
            raise TypeError('The scaling other should be a number or scalar tensor')
        cores = list(self._standard_cores())
        cores[0] = cores[0] * other
        dtype = cores[0].dtype
        cores = [core.to(dtype=dtype) for core in cores]

        return self._new_from_standard_cores(
            cores, self._in_dim, self._out_dim, self._n_batches, self._cyclic)

    def __rmul__(self, other):
        """Returns a scalar-scaled format or Hadamard product."""
        return self * other

    def __matmul__(self, other):
        """Returns the exact matrix application or matrix product."""
        a, b, batch, cyclic = self._binary_inputs(other, same_family=False)
        left_matrix, right_matrix = self._out_dim is not None, other._out_dim is not None
        if not (left_matrix or right_matrix):
            raise TypeError('At least one operand of @ should be a matrix format')
        if left_matrix:
            contracted = other._out_dim if right_matrix else other._in_dim
            if self._in_dim != contracted:
                raise ValueError('Contracted local dimensions should match')
        elif self._in_dim != other._out_dim:
            raise ValueError('Contracted local dimensions should match')
        cores = []
        for site, (x, y) in enumerate(zip(a, b)):
            if left_matrix:
                x = x.reshape(*batch, x.shape[-3], self._in_dim[site],
                              self._out_dim[site], x.shape[-1]).transpose(-1, -2)
            if right_matrix:
                y = y.reshape(*batch, y.shape[-3], other._in_dim[site],
                              other._out_dim[site], y.shape[-1]).transpose(-1, -2)
            if left_matrix and right_matrix:
                core = torch.einsum('...liro,...ajbi->...lajorb', x, y)
                in_dim, out_dim = other._in_dim, self._out_dim
                physical = in_dim[site] * out_dim[site]
            elif left_matrix:
                core = torch.einsum('...liro,...aib->...laorb', x, y)
                in_dim, out_dim = self._out_dim, None
                physical = in_dim[site]
            else:
                core = torch.einsum('...lob,...airo->...laibr', x, y)
                in_dim, out_dim = other._in_dim, None
                physical = in_dim[site]
            cores.append(core.reshape(*batch, x.shape[-4 if left_matrix else -3] *
                                      y.shape[-4 if right_matrix else -3], physical,
                                      x.shape[-2 if left_matrix else -1] *
                                      y.shape[-2 if right_matrix else -1]))

        return self._new_from_standard_cores(
            cores, in_dim, out_dim, len(batch), cyclic, other=other, product=True)

    def apply(self, other):
        r"""Applies a matrix format from the right, equivalent to self @ other.

        Parameters
        ----------
        other : TTM or TRM
            Matrix whose output dimensions match the vector input dimensions.
            This contracts the matrix output axes, unlike matrix @ vector, which
            contracts matrix input axes.

        Returns
        -------
        TT or TR
            Exact vector result; cyclic when either operand is cyclic. Quantics
            subclasses retain compatible coordinate semantics.
        """
        return self @ other

    def conj(self):
        r"""Returns the conjugate cores and bond factors.

        Returns
        -------
        TensorFormat1D
            Separate format with conjugated tensor references; storage may be
            shared.
        """
        return self._map_tensors(lambda tensor: tensor.conj())

    def contract_dense(self) -> torch.Tensor:
        r"""Contracts the full represented tensor explicitly.

        Returns
        -------
        torch.Tensor
            Dense tensor with shape (*core_batch, *in_dim) for vectors and
            interleaved (in_0, out_0, ...) axes for matrices. Intended for small
            tensors, not large-grid evaluation.
        """
        cores = self._standard_cores()
        closing = cores[0].shape[-3]
        result = cores[0]
        dimensions = [result.shape[-2]]
        for core in cores[1:]:
            result = result.reshape(*self._batch_shape, closing, -1, result.shape[-1])
            result = torch.einsum('...apr,...rqb->...apqb', result, core)
            dimensions.append(core.shape[-2])
        result = result.reshape(*self._batch_shape, closing, *
                                dimensions, cores[-1].shape[-1])
        result = result.diagonal(dim1=self._n_batches, dim2=-1).sum(-1)
        if self._out_dim is not None:
            dimensions = [dim for pair in zip(
                self._in_dim, self._out_dim) for dim in pair]
            result = result.reshape(*self._batch_shape, *dimensions)
        return result

    def inner(self, other) -> torch.Tensor:
        r"""Contracts the conjugate of self with another format.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with matching input/output dimensions and structural
            batch shape, on the same device. Dtypes may be promoted; boundary
            topologies may differ.

        Returns
        -------
        torch.Tensor
            Overlap tensor with shape core_batch, preserving complex phase.
        """
        phase, log_magnitude = self._log_overlap(other)
        return phase * log_magnitude.exp()

    @property
    def device(self) -> torch.device:
        """Device shared by all cores."""
        return self.cores[0].device

    @property
    def dtype(self) -> torch.dtype:
        """Data type shared by all cores."""
        return self.cores[0].dtype

    @property
    def rank(self) -> List[int]:
        """Bond ranks inferred from the cores."""
        return list(self._rank)

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        """Batch dimensions shared by the cores."""
        return self._batch_shape

    @property
    def in_dim(self) -> Tuple[int, ...]:
        """Input dimension associated with every site."""
        return self._in_dim

    @property
    def n_sites(self) -> int:
        """Number of sites in the stored core network.

        Each stored core represents one site. A hierarchical quantized result
        counts its upper-network sites; :meth:`flatten` constructs a separate
        result whose sites include the digit factors.
        """
        return len(self.cores)

    @property
    def out_dim(self) -> Optional[Tuple[int, ...]]:
        """Output dimension per site, when the decomposition has one."""
        return self._out_dim

    @property
    def topology(self) -> str:
        """Topology identifier used in serialized result information."""
        return self._topology

    def _normalize_data(
            self,
            data: EvaluationData,
            dimensions: Sequence[int],
            same_dim: bool,
            n_batches: int
    ) -> Tuple[List[torch.Tensor], bool, Tuple[int, ...]]:
        """Normalizes discrete indices or embedded vectors by site."""
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
                site_data = list(
                    data.to(device=self.device, dtype=self.dtype).unbind(-2))
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

            target_dtype = None if discrete else self.dtype
            site_data = [
                item.to(device=self.device, dtype=target_dtype)
                for item in site_data
            ]

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

    @staticmethod
    def _contract_open_chain(
            matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts an open chain of matrices along adjacent ranks."""
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        return result

    def _check_overlap_compatibility(
            self, other: 'TensorFormat1D') -> None:
        """Validates topology, shapes and runtime for an overlap."""
        if not isinstance(other, TensorFormat1D):
            raise TypeError('`other` should be TensorFormat1D type')
        if self._family != other._family:
            raise ValueError('The decomposition families are incompatible')
        if len(self.cores) != len(other.cores):
            raise ValueError('Decompositions should have the same number of sites')
        if self._in_dim != other._in_dim:
            raise ValueError('Decompositions should have matching input dimensions')
        if self._out_dim != other._out_dim:
            raise ValueError('Decompositions should have matching output dimensions')
        if self._batch_shape != other._batch_shape:
            raise ValueError('Decompositions should have matching batch shapes')
        if self.device != other.device:
            raise ValueError('Decompositions should be on the same device')

    def _log_overlap(
            self, other: 'TensorFormat1D') -> Tuple[torch.Tensor,
                                                    torch.Tensor]:
        """Returns overlap phase and log-magnitude using scaled environments."""
        self._check_overlap_compatibility(other)
        dtype = torch.promote_types(self.dtype, other.dtype)
        self_cores = self._standard_cores()
        other_cores = self_cores if other is self else other._standard_cores()

        self_eye = torch.eye(
            self_cores[0].shape[-3], device=self.device, dtype=dtype)
        other_eye = torch.eye(
            other_cores[0].shape[-3], device=self.device, dtype=dtype)
        environment = torch.einsum('ai,bj->abij', self_eye, other_eye)
        if self.n_batches:
            environment = environment.reshape(
                *((1,) * self.n_batches), *environment.shape)
            environment = environment.expand(
                *self._batch_shape, *environment.shape[self.n_batches:])

        real_dtype = torch.empty((), dtype=dtype).real.dtype
        log_scale = torch.zeros(
            self._batch_shape, device=self.device, dtype=real_dtype)

        for self_core, other_core in zip(self_cores, other_cores):
            self_core = self_core.to(dtype=dtype)
            self_scale = self_core.abs().amax(dim=(-3, -2, -1))
            self_scale = torch.where(self_scale > 0, self_scale,
                                     torch.ones_like(self_scale))
            self_core = self_core / self_scale[..., None, None, None]
            if other is self:
                other_scale, other_core = self_scale, self_core
            else:
                other_core = other_core.to(dtype=dtype)
                other_scale = other_core.abs().amax(dim=(-3, -2, -1))
                other_scale = torch.where(
                    other_scale > 0, other_scale, torch.ones_like(other_scale))
                other_core = other_core / other_scale[..., None, None, None]
            log_scale = log_scale + self_scale.log() + other_scale.log()
            environment = torch.einsum(
                '...xyab,...apr->...xybpr',
                environment,
                self_core.conj())
            environment = torch.einsum(
                '...xybpr,...bps->...xyrs',
                environment,
                other_core)

            scale = torch.linalg.vector_norm(
                environment, dim=(-4, -3, -2, -1))
            nonzero = scale > 0
            safe_scale = torch.where(nonzero, scale, torch.ones_like(scale))
            environment = environment / safe_scale[..., None, None, None, None]
            log_scale = log_scale + torch.where(
                nonzero, safe_scale.log(), torch.zeros_like(safe_scale))

        overlap = torch.einsum('...ijij->...', environment)
        magnitude = overlap.abs()
        nonzero = magnitude > 0
        safe_magnitude = torch.where(
            nonzero, magnitude, torch.ones_like(magnitude))
        phase = torch.where(nonzero,
                            overlap / safe_magnitude,
                            torch.zeros_like(overlap))
        log_magnitude = torch.where(
            nonzero,
            safe_magnitude.log() + log_scale,
            torch.full_like(log_scale, -torch.inf))
        return phase, log_magnitude

    def norm(self) -> torch.Tensor:
        r"""Returns the Frobenius norm using scaled double-layer contractions.

        Returns
        -------
        torch.Tensor
            Real tensor with shape core_batch, or a scalar for an unbatched
            format.
        """
        _, log_squared_norm = self._log_overlap(self)
        return torch.exp(log_squared_norm / 2)

    def normalized_overlap(
            self, other: 'TensorFormat1D') -> torch.Tensor:
        r"""Returns <self, other> / (||self|| ||other||), preserving phase.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with matching input/output dimensions and structural
            batch shape, on the same device. Dtypes may be promoted; boundary
            topologies may differ.

        Returns
        -------
        torch.Tensor
            Normalized overlap with shape core_batch. Zero-norm operands raise
            ValueError.
        """
        phase, log_overlap = self._log_overlap(other)
        _, log_self = self._log_overlap(self)
        _, log_other = other._log_overlap(other)

        if torch.any(torch.isneginf(log_self)) or \
                torch.any(torch.isneginf(log_other)):
            raise ValueError(
                'Normalized overlap is undefined for a zero-norm decomposition')

        log_denominator = (log_self + log_other) / 2
        return phase * torch.exp(log_overlap - log_denominator)

    def fidelity(self, other: 'TensorFormat1D') -> torch.Tensor:
        r"""Returns the squared magnitude of the normalized overlap.

        Parameters
        ----------
        other : TensorFormat1D
            Other format with matching input/output dimensions and structural
            batch shape, on the same device. Dtypes may be promoted; boundary
            topologies may differ.

        Returns
        -------
        torch.Tensor
            Real tensor with shape core_batch. Zero-norm operands raise
            ValueError.
        """
        return self.normalized_overlap(other).abs().square()


class _VectorFormat1D(TensorFormat1D):
    """Shared raw-tensor vector operations."""

    _family = 'state'

    def __call__(self,
                 data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        r"""Calls :meth:`evaluate` with the same input conventions.

        Parameters
        ----------
        data : torch.Tensor or sequence of torch.Tensor
            Integer configurations of shape (*data_batch, n_sites), or embedded
            vectors of shape (*data_batch, n_sites, in_dim) for uniform
            dimensions. For heterogeneous dimensions, pass one tensor per site.
            Each site tensor has shape (*data_batch,) for indices or
            (*data_batch, in_dim[site]) for embeddings.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch). Structural batches and
            data batches remain independent.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> format.evaluate(torch.tensor([[0, 0], [0, 1]])).tolist()
        [1.0, 0.0]
        >>> embeddings = torch.ones(1, 2, 2)
        >>> format.evaluate(embeddings).tolist()
        [2.0]
        """
        return self.evaluate(data, n_batches=n_batches)

    def _local_matrices(
            self,
            site_data: Sequence[torch.Tensor],
            discrete: bool,
            data_batch_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Contracts TT/TR cores with one input at each site."""
        core_batch_size = int(torch.Size(self._batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        matrices = []

        for core, site_value in zip(self._standard_cores(), site_data):
            left_rank, site_in_dim, right_rank = core.shape[-3:]
            core = core.reshape(
                core_batch_size, left_rank, site_in_dim, right_rank)

            if discrete:
                indices = site_value.reshape(data_batch_size).to(torch.long)
                matrix = core[:, :, indices, :].permute(0, 2, 1, 3)
            else:
                vectors = site_value.reshape(data_batch_size, site_in_dim)
                matrix = torch.einsum('dp,clpr->cdlr', vectors, core)

            matrices.append(matrix.reshape(
                *self._batch_shape,
                *data_batch_shape,
                left_rank,
                right_rank))

        return matrices

    def evaluate(self,
                 data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        r"""Evaluates integer configurations or local embedded vectors.

        Parameters
        ----------
        data : torch.Tensor or sequence of torch.Tensor
            Integer configurations of shape (*data_batch, n_sites), or embedded
            vectors of shape (*data_batch, n_sites, in_dim) for uniform
            dimensions. For heterogeneous dimensions, pass one tensor per site.
            Each site tensor has shape (*data_batch,) for indices or
            (*data_batch, in_dim[site]) for embeddings.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch). Structural batches and
            data batches remain independent.

        Examples
        --------
        >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
        >>> format.evaluate(torch.tensor([[0, 0], [0, 1]])).tolist()
        [1.0, 0.0]
        >>> embeddings = torch.ones(1, 2, 2)
        >>> format.evaluate(embeddings).tolist()
        [2.0]
        """
        site_data, discrete, data_batch_shape = self._normalize_data(
            data, self._in_dim, self._same_in_dim, n_batches)
        matrices = self._local_matrices(
            site_data, discrete, data_batch_shape)
        return self._contract_local_matrices(matrices)

    def error(self,
              function: Callable[..., torch.Tensor],
              samples: torch.Tensor,
              data: Optional[EvaluationData] = None,
              n_batches: int = 1,
              **kwargs: Any) -> SampleError:
        r"""Measures absolute and relative errors on a sample set.

        Parameters
        ----------
        function : callable
            Target callable invoked as function(samples, **kwargs), returning
            one value per data configuration. Its output is broadcast over
            structural batches stored in the format.
        samples : torch.Tensor
            Sample coordinates or indices passed to the target callable, with
            n_batches leading batch axes.
        data : torch.Tensor or sequence of torch.Tensor, optional
            Inputs used to evaluate the format. None evaluates samples directly;
            supply embedded inputs when target coordinates differ from format
            configurations.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.
        **kwargs : keyword arguments
            Additional arguments passed only to the target callable.

        Returns
        -------
        SampleError
            Scalar error record retaining autograd. Absolute and target norms
            aggregate all data and structural batches. Relative error is zero
            when both norms vanish and infinity when only the target norm
            vanishes.
        """
        if not callable(function):
            raise TypeError('`function` should be callable')
        if not isinstance(samples, torch.Tensor):
            raise TypeError('`samples` should be torch.Tensor type')
        if samples.ndim != (n_batches + 1):
            raise ValueError(
                '`samples` has an incompatible number of batch dimensions')

        samples = samples.to(device=self.device)
        approximation = self.evaluate(
            samples if data is None else data,
            n_batches=n_batches)
        target = function(samples, **kwargs)
        if not isinstance(target, torch.Tensor):
            raise TypeError('`function` should return a torch.Tensor')
        target = target.to(device=approximation.device,
                           dtype=approximation.dtype)
        target = target.reshape(samples.shape[:n_batches])
        if self.n_batches:
            target = target.reshape(
                *((1,) * self.n_batches), *target.shape)
            target = target.expand(*self._batch_shape, *target.shape[self.n_batches:])

        absolute = torch.linalg.vector_norm(approximation - target)
        denominator = torch.linalg.vector_norm(target)
        if denominator > 0:
            relative = absolute / denominator
        elif absolute == 0:
            relative = torch.zeros_like(absolute)
        else:
            relative = torch.full_like(absolute, torch.inf)

        size = int(torch.Size(samples.shape[:n_batches]).numel())
        return SampleError(
            kind='samples',
            absolute=absolute,
            relative=relative,
            size=size,
            denominator=denominator)

    @abstractmethod
    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts input-selected local matrices along the ranks."""

    def to_mps(self, parameterized: bool = False, **kwargs):
        r"""Builds an open or periodic MPS or MPSData from effective cores.

        Batched vectors produce MPSData and reject parameterized=True.

        Parameters
        ----------
        parameterized : bool
            Whether the constructed model uses trainable parameter nodes. Inputs
            are not detached implicitly.
        **kwargs : keyword arguments
            Additional model constructor options. Tensor cores and boundary are
            supplied by the adapter.

        Returns
        -------
        MPS or MPSData
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

        if self._out_dim is not None:
            raise TypeError('MPS adapters require a vector format')
        if not isinstance(parameterized, bool):
            raise TypeError('`parameterized` should be bool type')
        cores = _restore_cores(self._standard_cores(), self._in_dim,
                               self._out_dim, self._n_batches, self._cyclic)
        if self._n_batches:
            if parameterized:
                raise ValueError('MPSData does not expose parameterized model cores')
            return MPSData(tensors=cores,
                           n_batches=self._n_batches, **kwargs)
        return MPS(tensors=cores, parameterized=parameterized, **kwargs)

    @classmethod
    def from_mps(cls, model, **kwargs):
        r"""Collects effective open or periodic MPS or MPSData tensors.

        Parameters
        ----------
        model : MPS or MPSData
            Source model with matching boundaries. Public tensors include boundary
            contractions where applicable.
        **kwargs : keyword arguments
            Additional options for the concrete format constructor, such as
            Quantics metadata on a subclass.

        Returns
        -------
        TT or TR
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

    def _operator_cores(self):
        """Returns matrix cores with separate input and output axes."""
        return [core.reshape(*self._batch_shape, core.shape[-3],
                             self._in_dim[site], self._out_dim[site],
                             core.shape[-1]).transpose(-1, -2)
                for site, core in enumerate(self._standard_cores())]

    def transpose(self):
        r"""Swaps local matrix input/output axes without reversing sites.

        Returns
        -------
        TTM or TRM
            Separate matrix format; tensor views may share storage. Quantics
            subclasses exchange input/output coordinate semantics.
        """
        standard = []
        for site, core in enumerate(self._raw_standard_cores()):
            core = core.reshape(*self._batch_shape, core.shape[-3],
                                self._in_dim[site], self._out_dim[site], core.shape[-1])
            core = core.transpose(-3, -2)
            standard.append(core.reshape(*self._batch_shape,
                            core.shape[-4], -1, core.shape[-1]))
        result = self._new_from_standard_cores(
            standard, self._out_dim, self._in_dim,
            self._n_batches, self._cyclic, transpose=True)
        if self._bonds is not None:
            result._bonds = self._bonds._map_tensors(
                lambda tensor: tensor, result._on_bonds_changed)
        return result

    def adjoint(self):
        r"""Returns the conjugate transpose, including explicit factors.

        Returns
        -------
        TTM or TRM
            Separate matrix format; tensor views may share storage. Quantics
            subclasses exchange input/output coordinate semantics.
        """
        return self.transpose().conj()

    @property
    def T(self):
        """Matrix transpose, without reversing the chain."""
        return self.transpose()

    @property
    def H(self):
        """Matrix adjoint, including conjugation of bond factors."""
        return self.adjoint()

    def trace(self) -> torch.Tensor:
        r"""Contracts the operator trace without forming the dense matrix.

        Returns
        -------
        torch.Tensor
            Trace resolved by structural batch. Every site should have matching
            input/output dimensions; global squareness alone is insufficient.
        """
        if self._in_dim != self._out_dim:
            raise ValueError('Trace requires matching local input/output dimensions')
        matrices = [core.diagonal(dim1=-3, dim2=-1).sum(-1)
                    for core in self._operator_cores()]
        return self._contract_open_chain(matrices).diagonal(dim1=-2, dim2=-1).sum(-1)

    def __call__(
            self,
            in_data: EvaluationData,
            out_data: Optional[EvaluationData] = None,
            n_batches: int = 1
    ) -> Union[torch.Tensor, TensorFormat1D]:
        r"""Calls :meth:`evaluate` with the same input conventions.

        Parameters
        ----------
        in_data : torch.Tensor or sequence of torch.Tensor
            Integer configurations of shape (*data_batch, n_sites), or embedded
            vectors of shape (*data_batch, n_sites, in_dim) for uniform
            dimensions. For heterogeneous dimensions, pass one tensor per site.
            Each site tensor has shape (*data_batch,) for indices or
            (*data_batch, in_dim[site]) for embeddings.
        out_data : torch.Tensor or sequence of torch.Tensor
            Output configurations or embeddings in the same layouts as in_data,
            using out_dim instead of in_dim. Both data batch shapes should
            match.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch). Embedded contractions
            use supplied vectors directly, without implicit conjugation.
        """
        if out_data is None:
            return self.apply(in_data, n_batches=n_batches)
        return self.evaluate(in_data, out_data, n_batches=n_batches)

    @abstractmethod
    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts entry-selected matrices with the topology closure."""

    def _entry_matrices(
            self,
            in_data_by_site: Sequence[torch.Tensor],
            out_data_by_site: Sequence[torch.Tensor],
            in_discrete: bool,
            out_discrete: bool,
            data_batch_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Builds local matrices for paired matrix-entry evaluation."""
        core_batch_size = int(torch.Size(self._batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        matrices = []

        for core, in_site_data, out_site_data in zip(
                self._operator_cores(), in_data_by_site, out_data_by_site):
            left_rank, in_dim, right_rank, out_dim = core.shape[-4:]
            core = core.reshape(
                core_batch_size, left_rank, in_dim, right_rank, out_dim)

            if in_discrete and out_discrete:
                in_indices = in_site_data.reshape(data_batch_size).to(torch.long)
                out_indices = out_site_data.reshape(data_batch_size).to(torch.long)
                fused = in_indices * out_dim + out_indices
                matrix = core.permute(0, 1, 3, 2, 4).reshape(
                    core_batch_size, left_rank, right_rank, in_dim * out_dim)
                matrix = matrix[..., fused].permute(0, 3, 1, 2)
            elif in_discrete:
                indices = in_site_data.reshape(data_batch_size).to(torch.long)
                vectors = out_site_data.reshape(data_batch_size, out_dim)
                selected = core.index_select(2, indices)
                matrix = torch.einsum('cldro,do->cdlr', selected, vectors)
            elif out_discrete:
                indices = out_site_data.reshape(data_batch_size).to(torch.long)
                vectors = in_site_data.reshape(data_batch_size, in_dim)
                selected = core.index_select(4, indices)
                matrix = torch.einsum('clird,di->cdlr', selected, vectors)
            else:
                in_vectors = in_site_data.reshape(data_batch_size, in_dim)
                out_vectors = out_site_data.reshape(data_batch_size, out_dim)
                matrix = torch.einsum(
                    'di,do,cliro->cdlr', in_vectors, out_vectors, core)

            matrices.append(matrix.reshape(
                *self._batch_shape,
                *data_batch_shape,
                left_rank,
                right_rank))

        return matrices

    def evaluate(self,
                 in_data: EvaluationData,
                 out_data: EvaluationData,
                 n_batches: int = 1) -> torch.Tensor:
        r"""Evaluates paired matrix configurations or embedded vectors.

        Parameters
        ----------
        in_data : torch.Tensor or sequence of torch.Tensor
            Integer configurations of shape (*data_batch, n_sites), or embedded
            vectors of shape (*data_batch, n_sites, in_dim) for uniform
            dimensions. For heterogeneous dimensions, pass one tensor per site.
            Each site tensor has shape (*data_batch,) for indices or
            (*data_batch, in_dim[site]) for embeddings.
        out_data : torch.Tensor or sequence of torch.Tensor
            Output configurations or embeddings in the same layouts as in_data,
            using out_dim instead of in_dim. Both data batch shapes should
            match.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        torch.Tensor
            Values with shape (*core_batch, *data_batch). Embedded contractions
            use supplied vectors directly, without implicit conjugation.
        """
        in_data_by_site, in_discrete, data_batch_shape = self._normalize_data(
            in_data, self._in_dim, self._same_in_dim, n_batches)
        out_data_by_site, out_discrete, out_batch_shape = self._normalize_data(
            out_data, self._out_dim, self._same_out_dim, n_batches)
        if out_batch_shape != data_batch_shape:
            raise ValueError(
                'Input and output data should have the same batch shape')

        matrices = self._entry_matrices(
            in_data_by_site,
            out_data_by_site,
            in_discrete,
            out_discrete,
            data_batch_shape)
        return self._contract_local_matrices(matrices)

    def apply(self,
              data: EvaluationData,
              n_batches: int = 1) -> TensorFormat1D:
        r"""Applies the operator to product data or another format.

        Parameters
        ----------
        data : torch.Tensor, sequence of torch.Tensor or TensorFormat1D
            Product input data accepted by vector evaluate(), or a TT/TR vector
            or TTM/TRM matrix. With a vector, contracts in_dim; with a matrix,
            contracts self.in_dim with data.out_dim.
        n_batches : int
            Number of leading data batch axes. These are independent of
            structural batch axes stored in the cores.

        Returns
        -------
        TensorFormat1D
            Vector or matrix format, according to the operand. Cyclic if either
            operand is cyclic. Product data become structural batches in the
            returned vector; applying a global dense vector does not
            automatically factor it into TT cores.

        Examples
        --------
        >>> operator = tk.formats.TTM([torch.eye(2)])
        >>> vector = tk.formats.TT([torch.tensor([2., 3.])])
        >>> torch.equal((operator @ vector).contract_dense(), vector.contract_dense())
        True
        >>> operator.apply(torch.tensor([[0], [1]])).contract_dense().tolist()
        [[1.0, 0.0], [0.0, 1.0]]
        >>> torch.equal((operator @ operator).contract_dense(), torch.eye(2))
        True
        """
        if isinstance(data, TensorFormat1D):
            return self @ data
        site_data, discrete, data_batch_shape = self._normalize_data(
            data, self._in_dim, self._same_in_dim, n_batches)
        core_batch_size = int(torch.Size(self._batch_shape).numel())
        data_batch_size = int(torch.Size(data_batch_shape).numel())
        output_cores = []

        for core, site_value in zip(self._operator_cores(), site_data):
            left_rank, in_dim, right_rank, out_dim = core.shape[-4:]
            core = core.reshape(
                core_batch_size, left_rank, in_dim, right_rank, out_dim)
            if discrete:
                indices = site_value.reshape(data_batch_size).to(torch.long)
                output_core = core.index_select(2, indices)
                output_core = output_core.permute(0, 2, 1, 4, 3)
            else:
                vectors = site_value.reshape(data_batch_size, in_dim)
                output_core = torch.einsum(
                    'di,cliro->cdlor', vectors, core)
            output_cores.append(output_core.reshape(
                *self._batch_shape,
                *data_batch_shape,
                left_rank,
                out_dim,
                right_rank))

        return self._new_from_standard_cores(
            output_cores, self._out_dim, None,
            self._n_batches + n_batches, self._cyclic, product=True)

    def to_mpo(self, parameterized: bool = False, **kwargs):
        r"""Builds an open or periodic MPO from effective cores.

        MPO conversion requires unbatched cores. The model may share effective
        tensor storage with the format.

        Parameters
        ----------
        parameterized : bool
            Whether the constructed model uses trainable parameter nodes. Inputs
            are not detached implicitly.
        **kwargs : keyword arguments
            Additional model constructor options. Tensor cores and boundary are
            supplied by the adapter.

        Returns
        -------
        MPO
            New graph model. Stored factors are materialized in temporary
            tensors; the source format is unchanged.
        """
        from tensorkrowch.models import MPO

        if self._out_dim is None:
            raise TypeError('MPO adapters require a matrix format')
        if self._n_batches:
            raise ValueError('Batched MPO model cores are not supported')
        if not isinstance(parameterized, bool):
            raise TypeError('`parameterized` should be bool type')
        cores = _restore_cores(self._standard_cores(), self._in_dim,
                               self._out_dim, self._n_batches, self._cyclic)
        return MPO(tensors=cores, parameterized=parameterized, **kwargs)

    @classmethod
    def from_mpo(cls, model, **kwargs):
        r"""Collects effective open or periodic MPO tensors.

        Parameters
        ----------
        model : MPO
            Source model with matching boundaries. Public tensors include boundary
            contractions where applicable.
        **kwargs : keyword arguments
            Additional options for the concrete format constructor, such as
            Quantics metadata on a subclass.

        Returns
        -------
        TTM or TRM
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


class TT(_VectorFormat1D):
    r"""Lightweight open raw-tensor format.

    With leading structural batch axes B, endpoint cores have shapes
    ``(*B, input, right)`` and ``(*B, left, input)``; interiors use
    ``(*B, left, input, right)``. A single core is ``(*B, input)``.
    The constructor shares tensors and copies their container. No nodes or
    edges are constructed, and input tensors retain autograd.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores in the endpoint layout of the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    """

    _topology = 'tt'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates core layouts and returns structural dimensions and ranks."""
        n_sites = len(self.cores)
        batch_shape = tuple(self.cores[0].shape[:self.n_batches])
        in_dim = []

        if n_sites == 1:
            if self.cores[0].ndim != (self.n_batches + 1):
                raise ValueError(
                    'A one-site TT core should have one input dimension')
            in_dim.append(self.cores[0].shape[-1])
            return [], batch_shape, tuple(in_dim), None

        rank = []
        for site, core in enumerate(self.cores):
            if tuple(core.shape[:self.n_batches]) != batch_shape:
                raise ValueError('All TT cores should have the same batch shape')

            if site == 0:
                if core.ndim != (self.n_batches + 2):
                    raise ValueError(
                        'The first TT core should have input and right rank '
                        'dimensions')
                in_dim.append(core.shape[-2])
                rank.append(core.shape[-1])
            elif site == (n_sites - 1):
                if core.ndim != (self.n_batches + 2):
                    raise ValueError(
                        'The last TT core should have left rank and input '
                        'dimensions')
                if core.shape[-2] != rank[-1]:
                    raise ValueError('Adjacent TT ranks should match')
                in_dim.append(core.shape[-1])
            else:
                if core.ndim != (self.n_batches + 3):
                    raise ValueError(
                        'Interior TT cores should have left, input and '
                        'right dimensions')
                if core.shape[-3] != rank[-1]:
                    raise ValueError('Adjacent TT ranks should match')
                in_dim.append(core.shape[-2])
                rank.append(core.shape[-1])

        return rank, batch_shape, tuple(in_dim), None

    def _raw_standard_cores(self) -> List[torch.Tensor]:
        """Returns standard fused cores without explicit bond factors."""
        if len(self.cores) == 1:
            return [self.cores[0].unsqueeze(self.n_batches).unsqueeze(-1)]

        cores = [self.cores[0].unsqueeze(self.n_batches)]
        cores.extend(self.cores[1:-1])
        cores.append(self.cores[-1].unsqueeze(-1))
        return cores

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts selected local matrices across the stored virtual ranks."""
        result = self._contract_open_chain(matrices)
        return result.squeeze(-1).squeeze(-1)


class TR(_VectorFormat1D):
    r"""Lightweight cyclic raw-tensor format.

    Every core has shape ``(*batch, left, input, right)``. Adjacent ranks
    match, including the last-to-first closure. A one-site ring is a trace
    over its two virtual axes. Tensors retain storage and autograd without
    constructing TensorKrowch nodes or edges.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores in the endpoint layout of the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    """

    _topology = 'tr'
    _cyclic = True

    def rotate(self, first=0):
        r"""Rotates the stored ring cut to start at a selected site.

        Parameters
        ----------
        first : int
            Site that becomes index zero, in [0, n_sites - 1]. No arbitrary site
            permutation is performed.

        Returns
        -------
        TR
            Separate format with rotated cores, physical dimensions and factors.
            Dense physical axes undergo the same cyclic rotation.

        Examples
        --------
        >>> format = tk.formats.TR([torch.ones(1, 2, 2), torch.ones(2, 3, 1)])
        >>> rotated = format.rotate(first=1)
        >>> rotated.in_dim
        (3, 2)
        >>> torch.equal(rotated.contract_dense(), format.contract_dense().T)
        True
        """
        return self._rotate(first)

    def to_tt(self):
        r"""Opens the ring exactly by carrying the closure index through all sites.

        Returns
        -------
        TT
            Open format with the same dense tensor. Endpoint ranks incorporate
            the closure rank; intermediate cores carry an identity on that
            index. No truncation or densification is performed.

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
        batch_shape = tuple(self.cores[0].shape[:self.n_batches])
        rank = []
        in_dim = []

        for site, core in enumerate(self.cores):
            if core.ndim != (self.n_batches + 3):
                raise ValueError(
                    'TR cores should have left rank, input and right rank '
                    'dimensions')
            if tuple(core.shape[:self.n_batches]) != batch_shape:
                raise ValueError('All TR cores should have the same batch shape')
            if site and (core.shape[-3] != rank[-1]):
                raise ValueError('Adjacent TR ranks should match')
            in_dim.append(core.shape[-2])
            rank.append(core.shape[-1])

        if self.cores[-1].shape[-1] != self.cores[0].shape[-3]:
            raise ValueError('The last and first cyclic TR ranks should match')
        return rank, batch_shape, tuple(in_dim), None

    def _raw_standard_cores(self) -> List[torch.Tensor]:
        """Returns standard fused cores without explicit bond factors."""
        return list(self.cores)

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts selected local matrices across the stored virtual ranks."""
        result = self._contract_open_chain(matrices)
        return result.diagonal(dim1=-2, dim2=-1).sum(-1)


class TTM(_MatrixFormat1D):
    r"""Lightweight open raw-tensor format.

    Endpoint shapes are ``(input, right, output)`` and
    ``(left, input, output)``; interiors are
    ``(left, input, right, output)``. A single core is ``(input, output)``.
    Structural batches are currently unsupported. Tensor storage and autograd
    are retained without constructing TensorKrowch nodes or edges.

    TTM currently requires n_batches=0; batched operator formats use TRM.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores in the endpoint layout of the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    """

    _topology = 'ttm'

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates core layouts and returns structural dimensions and ranks."""
        if self.n_batches:
            raise ValueError('TTM decomposition batches are not supported')

        n_sites = len(self.cores)
        in_dim = []
        out_dim = []
        if n_sites == 1:
            if self.cores[0].ndim != 2:
                raise ValueError(
                    'A one-site TTM core should have input and output dimensions')
            in_dim.append(self.cores[0].shape[0])
            out_dim.append(self.cores[0].shape[1])
            return [], (), tuple(in_dim), tuple(out_dim)

        rank = []
        for site, core in enumerate(self.cores):
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

    def _raw_standard_cores(self) -> List[torch.Tensor]:
        """Returns standard fused cores without explicit bond factors."""
        if len(self.cores) == 1:
            core = self.cores[0]
            return [core.reshape(1, core.numel(), 1)]

        cores = []
        first = self.cores[0].permute(0, 2, 1)
        cores.append(first.reshape(1, first.shape[0] * first.shape[1],
                                   first.shape[2]))
        for core in self.cores[1:-1]:
            core = core.permute(0, 1, 3, 2)
            cores.append(core.reshape(core.shape[0],
                                      core.shape[1] * core.shape[2],
                                      core.shape[3]))
        last = self.cores[-1]
        cores.append(last.reshape(last.shape[0], -1, 1))
        return cores

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts selected local matrices across the stored virtual ranks."""
        result = self._contract_open_chain(matrices)
        return result.squeeze(-1).squeeze(-1)


class TRM(_MatrixFormat1D):
    r"""Lightweight cyclic raw-tensor format.

    Every core has shape ``(*batch, left, input, right, output)`` and
    adjacent ranks match through the cyclic closure. Structural batches are
    independent of evaluation-data batches. Tensor storage and autograd are
    retained without constructing TensorKrowch nodes or edges.

    Parameters
    ----------
    cores : sequence of torch.Tensor
        Raw cores in the endpoint layout of the concrete format. The
        container is copied and tensor storage is shared; inputs retain
        autograd.
    n_batches : int
        Number of leading structural batch axes shared by all cores.
        Independent of data batches during evaluation.
    """

    _topology = 'trm'
    _cyclic = True

    def rotate(self, first=0):
        r"""Rotates the stored ring cut to start at a selected site.

        Parameters
        ----------
        first : int
            Site that becomes index zero, in [0, n_sites - 1]. No arbitrary site
            permutation is performed.

        Returns
        -------
        TRM
            Separate format with rotated cores, physical dimensions and factors.
            Dense physical axes undergo the same cyclic rotation.
        """
        return self._rotate(first)

    def to_ttm(self):
        r"""Opens the ring exactly by carrying the closure index through all sites.

        A batched ring matrix cannot be converted because TTM does not support
        structural batches.

        Returns
        -------
        TTM
            Open format with the same dense tensor. Endpoint ranks incorporate
            the closure rank; intermediate cores carry an identity on that
            index. No truncation or densification is performed.
        """
        return self._to_open()

    def _validate_cores(
            self) -> Tuple[List[int], Tuple[int, ...], Tuple[int, ...],
                           Optional[Tuple[int, ...]]]:
        """Validates core layouts and returns structural dimensions and ranks."""
        batch_shape = tuple(self.cores[0].shape[:self.n_batches])
        rank = []
        in_dim = []
        out_dim = []

        for site, core in enumerate(self.cores):
            if core.ndim != (self.n_batches + 4):
                raise ValueError(
                    'TRM cores should have left rank, input, right rank and '
                    'output dimensions')
            if tuple(core.shape[:self.n_batches]) != batch_shape:
                raise ValueError(
                    'All TRM cores should have the same batch shape')
            if site and (core.shape[-4] != rank[-1]):
                raise ValueError('Adjacent TRM ranks should match')
            in_dim.append(core.shape[-3])
            rank.append(core.shape[-2])
            out_dim.append(core.shape[-1])

        if self.cores[-1].shape[-2] != self.cores[0].shape[-4]:
            raise ValueError(
                'The last and first cyclic TRM ranks should match')
        return rank, batch_shape, tuple(in_dim), tuple(out_dim)

    def _raw_standard_cores(self) -> List[torch.Tensor]:
        """Returns standard fused cores without explicit bond factors."""
        cores = []
        for core in self.cores:
            core = core.movedim(-1, -2)
            cores.append(core.reshape(
                *self._batch_shape,
                core.shape[-4],
                core.shape[-3] * core.shape[-2],
                core.shape[-1]))
        return cores

    def _contract_local_matrices(
            self, matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        """Contracts selected local matrices across the stored virtual ranks."""
        result = self._contract_open_chain(matrices)
        return result.diagonal(dim1=-2, dim2=-1).sum(-1)
