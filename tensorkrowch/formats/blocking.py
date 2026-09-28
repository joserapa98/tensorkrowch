"""Blocking and local SVD splits retaining external virtual interfaces."""

from dataclasses import dataclass
from math import prod
from typing import NamedTuple, Optional, Tuple

import torch

from tensorkrowch.utils import _validate_truncation, truncated_svd

from tensorkrowch.formats.bonds import BondFactors, VidalGauge
from tensorkrowch.formats.canonical import _redistribute
from tensorkrowch.formats.operations import _build_network


@dataclass(frozen=True)
class BlockLayout:
    """Original site dimensions and contiguous group sizes."""

    groups: Tuple[int, ...]  # Number of sites per block
    in_dim: Tuple[int, ...]  # Original input dimensions
    out_dim: Optional[Tuple[int, ...]] = None  # Original matrix output dimensions

    def __post_init__(self):
        if not self.groups or any(isinstance(size, bool) or not isinstance(size, int)
                                  or size < 1 for size in self.groups):
            raise ValueError('Block sizes should be positive integers')
        if sum(self.groups) != len(self.in_dim):
            raise ValueError('Block sizes should cover the original input dimensions')
        if self.out_dim is not None and len(self.out_dim) != len(self.in_dim):
            raise ValueError('Original matrix input/output sites should match')


UnblockInfo = BlockLayout


class SplitBlock(NamedTuple):
    """Local raw cores with open external ranks and optional internal factors."""

    cores: Tuple[torch.Tensor, ...]  # Standard fused core layout
    bonds: Optional[BondFactors]  # Internal bond factors only
    spectra: Tuple[torch.Tensor, ...]  # Singular values of the local cuts


def contract_block(network, first, last):
    """Contracts a contiguous region, including internal but not external factors."""
    network._ensure_valid()
    for site in (first, last):
        if isinstance(site, bool) or not isinstance(site, int):
            raise TypeError('Block endpoints should be integers')
    if not 0 <= first <= last < network.n_sites:
        raise ValueError('Block endpoints should select an ordered contiguous region')
    cores = network._raw_standard_cores()
    result = cores[first]
    dimensions = [cores[first].shape[-2]]
    batch = network._batch_shape
    left = result.shape[-3]
    for site in range(first, last):
        if network._bonds is not None and network._bonds.values[site] is not None:
            result = result * network._bonds.values[site][..., None, None, :]
        result = result.reshape(*batch, left, -1, result.shape[-1])
        result = torch.einsum('...apr,...rqb->...apqb', result, cores[site + 1])
        dimensions.append(cores[site + 1].shape[-2])
    if network._out_dim is not None:
        dimensions = [dim for pair in zip(network._in_dim[first:last + 1],
                                           network._out_dim[first:last + 1]) for dim in pair]
    return result.reshape(*batch, left, *dimensions, cores[last].shape[-1])


def split_block(block, in_dim, out_dim=None, n_batches=0, rank=None,
                cutoff=None, atol=None, rtol=None, cum_percentage=None,
                mode='right'):
    """Splits a block sitewise; external ranks remain open and unchanged.

    Physical axes are interleaved for matrix blocks. Spectra refer to this local
    factorization; they are not certified global Schmidt spectra. No source,
    metric collection, graph construction or decomposition engine is involved.
    """
    if not isinstance(block, torch.Tensor):
        raise TypeError('`block` should be torch.Tensor type')
    if isinstance(n_batches, bool) or not isinstance(n_batches, int):
        raise TypeError('`n_batches` should be int type')
    if n_batches < 0:
        raise ValueError('`n_batches` should be non-negative')
    in_dim = tuple(in_dim)
    if not in_dim or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in in_dim):
        raise ValueError('Input dimensions should be positive integers')
    if out_dim is not None:
        out_dim = tuple(out_dim)
        if len(out_dim) != len(in_dim) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 1 for dim in out_dim):
            raise ValueError('Output dimensions should match the positive site dimensions')
    dimensions = in_dim if out_dim is None else tuple(
        dim for pair in zip(in_dim, out_dim) for dim in pair)
    if block.ndim != n_batches + len(dimensions) + 2 or tuple(block.shape[n_batches + 1:-1]) != dimensions:
        raise ValueError('Block physical axes should match the requested dimensions')
    if block.shape[n_batches] < 1 or block.shape[-1] < 1:
        raise ValueError('External ranks should be positive')
    _validate_truncation(rank, cutoff, atol, rtol, cum_percentage)
    powers = {'explicit': (0, 0), 'implicit': (0.5, 0.5),
              'inverse': (1, 1), 'left': (1, 0), 'right': (0, 1)}
    if mode not in powers:
        raise ValueError('Invalid local bond distribution mode')
    physical = in_dim if out_dim is None else tuple(a * b for a, b in zip(in_dim, out_dim))
    batch = block.shape[:n_batches]
    right = block.shape[-1]
    state = block.reshape(*batch, block.shape[n_batches], *physical, right)
    cores, spectra = [], []
    for dimension in physical[:-1]:
        left = state.shape[n_batches]
        u, s, vh = truncated_svd(state.reshape(*batch, left * dimension, -1),
                                 rank, cutoff, atol, rtol, cum_percentage)
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


def block(network, groups, return_info=False):
    """Blocks consecutive sites, preserving matrix input/output axis groups."""
    network._ensure_valid()
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    groups = tuple(groups)
    if not groups or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in groups):
        raise ValueError('Block sizes should be positive integers')
    if sum(groups) != network.n_sites:
        raise ValueError('Block sizes should sum to the number of sites')
    info = BlockLayout(groups, network._in_dim, network._out_dim)
    cores, in_dim, out_dim, factors = [], [], [], []
    first = 0
    for size in groups:
        last = first + size - 1
        value = contract_block(network, first, last)
        in_dim.append(prod(network._in_dim[first:last + 1]))
        if network._out_dim is not None:
            out_dim.append(prod(network._out_dim[first:last + 1]))
            b = network._n_batches
            order = [*range(b + 1), *range(b + 1, b + 1 + 2 * size, 2),
                     *range(b + 2, b + 1 + 2 * size, 2), value.ndim - 1]
            value = value.permute(order)
        cores.append(value.reshape(*network._batch_shape, value.shape[network._n_batches],
                                   -1, value.shape[-1]))
        if network._bonds is not None and last < len(network._bonds.values):
            factors.append(network._bonds.values[last])
        first = last + 1
    result = _build_network(cores, tuple(in_dim), tuple(out_dim) if out_dim else None,
                            network._n_batches, network._topology.startswith('tr'))
    if factors:
        result.bonds = BondFactors(factors)
    result._block_layout = info
    return (result, info) if return_info else result


def unblock(network, info=None, **kwargs):
    """Restores a blocked network using stored or explicitly supplied layout."""
    network._ensure_valid()
    info = getattr(network, '_block_layout', None) if info is None else info
    if not isinstance(info, BlockLayout) or len(info.groups) != network.n_sites:
        raise ValueError('Unblocking requires a matching BlockLayout')
    cores, first = [], 0
    standard = network._standard_cores()
    for site, size in enumerate(info.groups):
        value = standard[site]
        inputs = info.in_dim[first:first + size]
        outputs = None if info.out_dim is None else info.out_dim[first:first + size]
        dimensions = inputs
        if outputs is not None:
            value = value.reshape(*network._batch_shape, value.shape[-3], *inputs, *outputs, value.shape[-1])
            b = network._n_batches
            order = [*range(b + 1)]
            for index in range(size):
                order.extend([b + 1 + index, b + 1 + size + index])
            order.append(value.ndim - 1)
            value = value.permute(order)
            dimensions = tuple(dim for pair in zip(inputs, outputs) for dim in pair)
        value = value.reshape(*network._batch_shape, value.shape[network._n_batches],
                              *dimensions, value.shape[-1])
        local = split_block(value, inputs, outputs, network._n_batches, **kwargs)
        # Restore the local representation with each factor absorbed once.
        for index, core in enumerate(local.cores):
            factor = local.bonds.values[index] if index < len(local.bonds.values) else None
            cores.append(core if factor is None else core * factor[..., None, None, :])
        first += size
    return _build_network(cores, info.in_dim, info.out_dim, network._n_batches,
                          network._topology.startswith('tr'))


def replace_block(network, first, last, replacement):
    """Installs local cores and internal factors atomically, preserving interfaces."""
    network._ensure_valid()
    if any(isinstance(site, bool) or not isinstance(site, int) for site in (first, last)):
        raise TypeError('Block endpoints should be integers')
    if not 0 <= first <= last < network.n_sites:
        raise ValueError('Block endpoints should select an ordered region')
    if not isinstance(replacement, SplitBlock):
        replacement = SplitBlock(tuple(replacement), None, ())
    if len(replacement.cores) != last - first + 1:
        raise ValueError('Replacement should preserve the number of block sites')
    for offset, core in enumerate(replacement.cores):
        if not isinstance(core, torch.Tensor):
            raise TypeError('Replacement cores should be tensors')
        site = first + offset
        dimension = network._in_dim[site] * (network._out_dim[site] if network._out_dim else 1)
        if core.ndim != network._n_batches + 3 or core.shape[-2] != dimension:
            raise ValueError('Replacement physical dimensions should match the selected sites')
    cores = list(network._raw_standard_cores())
    if replacement.cores[0].shape[-3] != cores[first].shape[-3] or \
            replacement.cores[-1].shape[-1] != cores[last].shape[-1]:
        raise ValueError('Replacement should preserve external block ranks')
    cores[first:last + 1] = replacement.cores
    count = network.n_sites if network._topology.startswith('tr') else network.n_sites - 1
    factors = list(network._bonds.values) if network._bonds is not None else [None] * count
    values = [None] * (last - first) if replacement.bonds is None else replacement.bonds.values
    if len(values) != last - first:
        raise ValueError('Replacement factors should match its internal bonds')
    factors[first:last] = values
    result = _build_network(cores, network._in_dim, network._out_dim,
                            network._n_batches, network._topology.startswith('tr'))
    result.bonds = BondFactors(factors)
    network._set_standard_cores(cores, result.bonds)
    network._orth_center = None
    return network


def absorb_bond(network, bond, side='left'):
    """Moves the selected diagonal/Schmidt weights to one neighbour in-place."""
    network._ensure_valid()
    count = network.n_sites if network._topology.startswith('tr') else network.n_sites - 1
    if isinstance(bond, bool) or not isinstance(bond, int):
        raise TypeError('`bond` should be int type')
    if not 0 <= bond < count:
        raise ValueError('`bond` should select a valid virtual bond')
    if side not in ('left', 'right'):
        raise ValueError('`side` should be "left" or "right"')
    if network._bonds is None:
        return network
    cores = list(network._raw_standard_cores())
    if isinstance(network._bonds, VidalGauge):
        powers = list(network._bonds.powers)
        powers[bond] = (1, 0) if side == 'left' else (0, 1)
        cores, factors = _redistribute(cores, network._bonds, powers)
    else:
        factors = BondFactors(network._bonds.values)
        value = factors.values[bond]
        if value is not None:
            site = bond if side == 'left' else (bond + 1) % network.n_sites
            cores[site] = cores[site] * (value[..., None, None, :] if side == 'left'
                                         else value[..., :, None, None])
            factors.values[bond] = None
    network._set_standard_cores(cores, factors)
    return network
