"""Blocking and local SVD splits retaining external virtual interfaces."""

from dataclasses import dataclass
from math import prod
from math import isfinite
from numbers import Real
from typing import Optional, Sequence, Tuple

import torch

from tensorkrowch.utils import _validate_truncation, truncated_svd

from tensorkrowch.formats.bonds import BondFactors, VidalGauge
from tensorkrowch.formats.canonical import _redistribute
from tensorkrowch.formats.operations import _build_network


@dataclass(frozen=True)
class BlockLayout:
    r"""Original site dimensions and contiguous group sizes.

    Frozen record: field references cannot be reassigned. Tensor contents and
    autograd are preserved without copying or detaching.

    Parameters
    ----------
    groups : tuple[int, ...]
        Number of original consecutive sites in each block.
    in_dim : tuple[int, ...]
        Original input dimension of every site.
    out_dim : tuple[int, ...], optional
        Original matrix output dimensions; None for vectors.
    """

    groups: Tuple[int, ...]  # Number of sites per block
    in_dim: Tuple[int, ...]  # Original input dimensions
    out_dim: Optional[Tuple[int, ...]] = None  # Original matrix output dimensions

    def __post_init__(self):
        """Validates block sizes and original site dimensions."""
        if not self.groups or any(isinstance(size, bool) or not isinstance(size, int)
                                  or size < 1 for size in self.groups):
            raise ValueError('Block sizes should be positive integers')
        if sum(self.groups) != len(self.in_dim):
            raise ValueError('Block sizes should cover the original input dimensions')
        if self.out_dim is not None and len(self.out_dim) != len(self.in_dim):
            raise ValueError('Original matrix input/output sites should match')


UnblockInfo = BlockLayout


@dataclass(frozen=True)
class SplitBlock:
    r"""Local raw cores with open external ranks and optional internal factors.

    Frozen record: field references cannot be reassigned. Tensor contents and
    autograd are preserved without copying or detaching.

    Parameters
    ----------
    cores : tuple[torch.Tensor, ...]
        Local cores in standard (*batch, left, physical, right) layout with
        physical axes fused for matrices.
    bonds : BondFactors or None
        Factors internal to the local block; external interface factors are
        excluded.
    spectra : tuple[torch.Tensor, ...]
        Singular values at the local SVD cuts. They are not certified global
        Schmidt spectra.
    """

    cores: Tuple[torch.Tensor, ...]  # Standard fused core layout
    bonds: Optional[BondFactors]  # Internal bond factors only
    spectra: Tuple[torch.Tensor, ...]  # Singular values of the local cuts


def contract_block(format, first, last):
    """Contracts a contiguous region, including internal but not external factors."""
    for site in (first, last):
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('Block endpoints should be integers')
    if not 0 <= first <= last < format.n_sites:
        raise ValueError('Block endpoints should select an ordered contiguous region')
    cores = format._raw_standard_cores()
    result = cores[first]
    dimensions = [cores[first].shape[-2]]
    batch = format._batch_shape
    left = result.shape[-3]
    for site in range(first, last):
        if format._bonds is not None and format._bonds.values[site] is not None:
            result = result * format._bonds.values[site][..., None, None, :]
        result = result.reshape(*batch, left, -1, result.shape[-1])
        result = torch.einsum('...apr,...rqb->...apqb', result, cores[site + 1])
        dimensions.append(cores[site + 1].shape[-2])
    if format._out_dim is not None:
        dimensions = [dim for pair in zip(format._in_dim[first:last + 1],
                                          format._out_dim[first:last + 1]) for dim in pair]
    return result.reshape(*batch, left, *dimensions, cores[last].shape[-1])


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
    >>> _ = format.replace_block(0, 1, local)
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
    gauge = VidalGauge(spectra, [(0, 0)] * len(spectra))
    cores, gauge = _redistribute(cores, gauge, [powers[mode]] * len(spectra))
    gauge._valid = False
    return SplitBlock(tuple(cores), gauge, tuple(spectra))


def block(format, groups: Sequence[int], return_info: bool = False):
    """Blocks consecutive sites, preserving matrix input/output axis groups."""
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    groups = tuple(groups)
    if not groups or any(isinstance(size, bool) or not isinstance(
        size, int) or size < 1 for size in groups):
        raise ValueError('Block sizes should be positive integers')
    if sum(groups) != format.n_sites:
        raise ValueError('Block sizes should sum to the number of sites')
    info = BlockLayout(groups, format._in_dim, format._out_dim)
    cores, in_dim, out_dim, factors = [], [], [], []
    first = 0
    for size in groups:
        last = first + size - 1
        value = contract_block(format, first, last)
        in_dim.append(prod(format._in_dim[first:last + 1]))
        if format._out_dim is not None:
            out_dim.append(prod(format._out_dim[first:last + 1]))
            b = format._n_batches
            order = [*range(b + 1), *range(b + 1, b + 1 + 2 * size, 2),
                     *range(b + 2, b + 1 + 2 * size, 2), value.ndim - 1]
            value = value.permute(order)
        cores.append(value.reshape(*format._batch_shape, value.shape[format._n_batches],
                                   -1, value.shape[-1]))
        if format._bonds is not None and last < len(format._bonds.values):
            factors.append(format._bonds.values[last])
        first = last + 1
    result = _build_network(cores, tuple(in_dim), tuple(out_dim) if out_dim else None,
                            format._n_batches, format._cyclic)
    if factors:
        result.bonds = BondFactors(factors)
    result._block_layout = info
    return (result, info) if return_info else result


def unblock(format, info=None, **kwargs):
    """Restores a blocked network using stored or explicitly supplied layout."""
    info = getattr(format, '_block_layout', None) if info is None else info
    if not isinstance(info, BlockLayout) or len(info.groups) != format.n_sites:
        raise ValueError('Unblocking requires a matching BlockLayout')
    cores, first = [], 0
    standard = format._standard_cores()
    for site, size in enumerate(info.groups):
        value = standard[site]
        inputs = info.in_dim[first:first + size]
        outputs = None if info.out_dim is None else info.out_dim[first:first + size]
        dimensions = inputs
        if outputs is not None:
            value = value.reshape(*format._batch_shape,
                                  value.shape[-3], *inputs, *outputs, value.shape[-1])
            b = format._n_batches
            order = [*range(b + 1)]
            for index in range(size):
                order.extend([b + 1 + index, b + 1 + size + index])
            order.append(value.ndim - 1)
            value = value.permute(order)
            dimensions = tuple(dim for pair in zip(inputs, outputs) for dim in pair)
        value = value.reshape(*format._batch_shape, value.shape[format._n_batches],
                              *dimensions, value.shape[-1])
        local = split_block(value, inputs, outputs, format._n_batches, **kwargs)
        # Restore the local representation with each factor absorbed once.
        for index, core in enumerate(local.cores):
            factor = local.bonds.values[index] if index < len(
                local.bonds.values) else None
            cores.append(core if factor is None else core * factor[..., None, None, :])
        first += size
    return _build_network(cores, info.in_dim, info.out_dim, format._n_batches,
                          format._cyclic)


def replace_block(format, first, last, replacement):
    """Installs local cores and internal factors atomically, preserving interfaces."""
    if any(isinstance(site, bool) or not isinstance(site, int)
           for site in (first, last)):
        raise TypeError('Block endpoints should be integers')
    if not 0 <= first <= last < format.n_sites:
        raise ValueError('Block endpoints should select an ordered region')
    if not isinstance(replacement, SplitBlock):
        replacement = SplitBlock(tuple(replacement), None, ())
    if len(replacement.cores) != last - first + 1:
        raise ValueError('Replacement should preserve the number of block sites')
    for offset, core in enumerate(replacement.cores):
        if not isinstance(core, torch.Tensor):
            raise TypeError('Replacement cores should be tensors')
        site = first + offset
        dimension = format._in_dim[site] * \
            (format._out_dim[site] if format._out_dim else 1)
        if core.ndim != format._n_batches + 3 or core.shape[-2] != dimension:
            raise ValueError(
                'Replacement physical dimensions should match the selected sites')
    cores = list(format._raw_standard_cores())
    if replacement.cores[0].shape[-3] != cores[first].shape[-3] or \
            replacement.cores[-1].shape[-1] != cores[last].shape[-1]:
        raise ValueError('Replacement should preserve external block ranks')
    cores[first:last + 1] = replacement.cores
    count = format.n_sites if format._topology.startswith(
        'tr') else format.n_sites - 1
    factors = list(format._bonds.values) if format._bonds is not None else [
        None] * count
    values = [None] * \
        (last - first) if replacement.bonds is None else replacement.bonds.values
    if len(values) != last - first:
        raise ValueError('Replacement factors should match its internal bonds')
    factors[first:last] = values
    result = _build_network(cores, format._in_dim, format._out_dim,
                            format._n_batches, format._cyclic)
    result.bonds = BondFactors(factors)
    format._set_standard_cores(cores, result.bonds)
    format._orth_center = None
    return format


def absorb_bond(format, bond, side: str = 'left'):
    """Moves the selected diagonal/Schmidt weights to one neighbour in-place."""
    count = format.n_sites if format._topology.startswith(
        'tr') else format.n_sites - 1
    if isinstance(bond, bool) or not isinstance(bond, int):
        raise TypeError('`bond` should be int type')
    if not 0 <= bond < count:
        raise ValueError('`bond` should select a valid virtual bond')
    if side not in ('left', 'right'):
        raise ValueError('`side` should be "left" or "right"')
    if format._bonds is None:
        return format
    cores = list(format._raw_standard_cores())
    if isinstance(format._bonds, VidalGauge) and format._bonds._valid:
        powers = list(format._bonds.powers)
        powers[bond] = (1, 0) if side == 'left' else (0, 1)
        cores, factors = _redistribute(cores, format._bonds, powers)
    else:
        factors = BondFactors(format._bonds.values)
        value = factors.values[bond]
        if value is not None:
            site = bond if side == 'left' else (bond + 1) % format.n_sites
            cores[site] = cores[site] * (value[..., None, None, :] if side == 'left'
                                         else value[..., :, None, None])
            factors.values[bond] = None
    format._set_standard_cores(cores, factors)
    return format


def redistribute_bond(format, bond: int, mode: str = 'implicit',
                      inverse_cutoff: float = 0.0):
    """Redistributes one stored spectrum using its current absorption powers."""
    if isinstance(bond, bool) or not isinstance(bond, int):
        raise TypeError('`bond` should be int type')
    if not isinstance(format._bonds, VidalGauge) or not format._bonds._valid:
        raise ValueError('Bond redistribution requires valid stored Vidal spectra')
    if not 0 <= bond < len(format._bonds.spectra):
        raise ValueError('`bond` should select a valid virtual bond')
    modes = {'explicit': (0, 0), 'implicit': (0.5, 0.5),
             'inverse': (1, 1), 'left': (1, 0), 'right': (0, 1)}
    if mode not in modes:
        raise ValueError('Invalid Vidal bond distribution mode')
    if isinstance(inverse_cutoff, bool) or not isinstance(inverse_cutoff, Real):
        raise TypeError('`inverse_cutoff` should be a real number')
    if not isfinite(inverse_cutoff) or inverse_cutoff < 0:
        raise ValueError('`inverse_cutoff` should be finite and non-negative')
    if mode == 'inverse' and torch.any(format._bonds.spectra[bond] <= inverse_cutoff):
        raise ValueError(f'Bond {bond} has non-invertible retained spectrum')
    powers = list(format._bonds.powers)
    powers[bond] = modes[mode]
    cores, factors = _redistribute(
    format._raw_standard_cores(), format._bonds, powers)
    format._set_standard_cores(cores, factors)
    return format
