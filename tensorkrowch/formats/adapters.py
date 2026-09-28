"""Exact topology conversions and adapters to TensorKrowch models."""

import torch

from tensorkrowch.formats.bonds import BondFactors
from tensorkrowch.formats.operations import _build_network


def rotate(network, first=0):
    """Rotates a ring so the selected site is first; dense axes rotate equally."""
    network._ensure_valid()
    if not network._cyclic:
        raise ValueError('Rotation requires a cyclic network')
    if isinstance(first, bool) or not isinstance(first, int):
        raise TypeError('`first` should be int type')
    if first < 0 or first >= network.n_sites:
        raise ValueError('`first` should select a valid site')
    order = [*range(first, network.n_sites), *range(first)]
    cores = network._raw_standard_cores()
    in_dim = tuple(network._in_dim[site] for site in order)
    out_dim = None if network._out_dim is None else tuple(
        network._out_dim[site] for site in order)
    result = _build_network([cores[site] for site in order], in_dim, out_dim,
                            network._n_batches, True)
    if network._bonds is not None:
        result.bonds = BondFactors([network._bonds.values[site] for site in order])
    return result


def ring_to_train(network):
    """Carries the closing index through identity factors without approximation.

    Internal train ranks are the original rank times the stored closing rank.
    One-site rings reduce exactly to their trace. No dense tensor or SVD is
    constructed; matrix physical dimensions are fused only inside this kernel.
    """
    network._ensure_valid()
    if not network._cyclic:
        raise ValueError('Conversion requires a cyclic network')
    cores = network._standard_cores()
    batch = network._batch_shape
    closing = cores[0].shape[-3]
    if len(cores) == 1:
        result = [cores[0].diagonal(
            dim1=-3, dim2=-1).sum(-1).unsqueeze(-2).unsqueeze(-1)]
    else:
        first = cores[0].transpose(-3, -2).reshape(*batch, 1, cores[0].shape[-2], -1)
        result = [first]
        identity = torch.eye(closing, dtype=network.dtype, device=network.device)
        for core in cores[1:-1]:
            combined = torch.einsum('st,...aib->...saitb', identity, core)
            result.append(combined.reshape(*batch, closing * core.shape[-3],
                                           core.shape[-2], closing * core.shape[-1]))
        last = cores[-1].movedim(-1, -3)
        result.append(last.reshape(*batch, closing * cores[-1].shape[-3],
                                   cores[-1].shape[-2], 1))
    return _build_network(result, network._in_dim, network._out_dim,
                          network._n_batches, False)


def to_mps(network, parameterized: bool = False, **kwargs):
    """Builds MPS or MPSData from effective vector cores, without detaching."""
    from tensorkrowch.models import MPS, MPSData

    network._ensure_valid()
    if network._out_dim is not None:
        raise TypeError('MPS adapters require a vector format')
    if not isinstance(parameterized, bool):
        raise TypeError('`parameterized` should be bool type')
    effective = _build_network(network._standard_cores(), network._in_dim, None,
                               network._n_batches, network._cyclic)
    if network._n_batches:
        if parameterized:
            raise ValueError('MPSData does not expose parameterized model cores')
        return MPSData(tensors=list(effective.cores),
                       n_batches=network._n_batches, **kwargs)
    return MPS(tensors=list(effective.cores), parameterized=parameterized, **kwargs)


def from_mps(model, cyclic: bool):
    """Collects effective model tensors, including open boundary contractions."""
    from tensorkrowch.models import MPS, MPSData

    from tensorkrowch.formats.tr import TR
    from tensorkrowch.formats.tt import TT

    if not isinstance(model, (MPS, MPSData)):
        raise TypeError('`model` should be MPS or MPSData type')
    boundary = 'pbc' if cyclic else 'obc'
    if model.boundary != boundary:
        raise ValueError(f'This adapter requires {boundary} boundaries')
    n_batches = model.n_batches if isinstance(model, MPSData) else 0
    return (TR if cyclic else TT)(model.tensors, n_batches=n_batches)


def to_mpo(network, parameterized: bool = False, **kwargs):
    """Builds an MPO from effective matrix cores without modifying the format."""
    from tensorkrowch.models import MPO

    network._ensure_valid()
    if network._out_dim is None:
        raise TypeError('MPO adapters require a matrix format')
    if network._n_batches:
        raise ValueError('Batched MPO model cores are not supported')
    if not isinstance(parameterized, bool):
        raise TypeError('`parameterized` should be bool type')
    effective = _build_network(network._standard_cores(), network._in_dim,
                               network._out_dim, 0, network._cyclic)
    return MPO(tensors=list(effective.cores), parameterized=parameterized, **kwargs)


def from_mpo(model, cyclic: bool):
    """Collects effective MPO tensors, including open boundary contractions."""
    from tensorkrowch.models import MPO

    from tensorkrowch.formats.trm import TRM
    from tensorkrowch.formats.ttm import TTM

    if not isinstance(model, MPO):
        raise TypeError('`model` should be MPO type')
    boundary = 'pbc' if cyclic else 'obc'
    if model.boundary != boundary:
        raise ValueError(f'This adapter requires {boundary} boundaries')
    return (TRM if cyclic else TTM)(model.tensors)
