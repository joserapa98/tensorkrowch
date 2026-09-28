"""Sitewise open-chain rounding and the published cyclic rounding algorithm."""

import warnings
from math import isfinite, sqrt
from numbers import Real
from typing import NamedTuple, Optional, Tuple

import torch

from tensorkrowch.utils import _validate_truncation, truncated_svd

from tensorkrowch.formats.operations import _build_network


class RoundingInfo(NamedTuple):
    """Truncation bound rather than a measured global approximation error."""

    rank: Tuple[int, ...]
    discarded_sq_norm: Tuple[torch.Tensor, ...]
    error_bound: torch.Tensor
    bound_satisfied: Optional[bool]


def rounding(network, rank, cutoff, atol, rtol, cum_percentage,
             renormalize, rel_error, return_info):
    _validate_truncation(rank, cutoff, atol, rtol, cum_percentage)
    for name, value in [('renormalize', renormalize), ('return_info', return_info)]:
        if not isinstance(value, bool):
            raise TypeError(f'`{name}` should be bool type')
    if rel_error is not None:
        if isinstance(rel_error, bool) or not isinstance(rel_error, Real):
            raise TypeError('`rel_error` should be a real number')
        if not isfinite(rel_error) or rel_error < 0:
            raise ValueError('`rel_error` should be finite and non-negative')
    network._ensure_valid()
    cyclic = network._topology.startswith('tr')
    closing = network._raw_standard_cores()[0].shape[-3] if cyclic else 1
    norm = network.norm() if rel_error is not None else None
    work = _build_network(network._standard_cores(), network._in_dim,
                          network._out_dim, network._n_batches, cyclic)
    work.canonicalize(renormalize=renormalize)
    cores = list(work._standard_cores())
    batch = network._batch_shape
    cuts = len(cores) if cyclic else max(1, len(cores) - 1)
    delta = rel_error * norm / sqrt(cuts * closing) if norm is not None else None
    records = []
    discarded_norms = []
    collect = return_info or rel_error is not None

    def split(matrix):
        scale = matrix.abs().amax()
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        limit = torch.finfo(matrix.real.dtype).max
        local_cutoff = None if cutoff is None else min(cutoff / scale.item(), limit)
        local_atol = None if atol is None else min(atol / scale.item() / scale.item(), limit)
        if delta is not None:
            value = (delta / scale).square().min().item()
            local_atol = min(value, limit) if local_atol is None else min(local_atol, value)
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
    network._set_standard_cores(cores)
    network._orth_center = None if cyclic else 0
    satisfied = None
    if collect:
        if discarded_norms:
            errors = torch.stack(discarded_norms)
            scale = errors.amax(dim=0)
            safe = torch.where(scale > 0, scale, torch.ones_like(scale))
            bound = torch.linalg.vector_norm(errors / safe, dim=0) * scale * sqrt(closing)
        else:
            bound = network.cores[0].real.new_zeros(batch)
        if rel_error is not None:
            satisfied = bool(torch.all(bound <= rel_error * norm +
                                       10 * torch.finfo(norm.dtype).eps * norm))
            if not satisfied:
                warnings.warn('Truncation constraints exceed the requested global error budget',
                              UserWarning, stacklevel=2)
        if return_info:
            return network, RoundingInfo(tuple(network.rank), tuple(records), bound, satisfied)
    return network
