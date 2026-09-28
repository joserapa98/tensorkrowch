"""Exact raw-core algebra and construction of compact endpoint layouts."""

from numbers import Number

import torch


def _build_network(cores, in_dim, out_dim, n_batches, cyclic):
    """Restores public vector/matrix endpoints from standard fused cores."""
    from tensorkrowch.formats.tr import TR
    from tensorkrowch.formats.trm import TRM
    from tensorkrowch.formats.tt import TT
    from tensorkrowch.formats.ttm import TTM

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
    cls = (TRM if cyclic else TTM) if out_dim is not None else (
        TR if cyclic else TT)
    return cls(stored, n_batches=n_batches)


def _binary_inputs(first, second, same_family=True):
    """Checks local dimensions and prepares compatible structural batches."""
    from tensorkrowch.formats._chain import TensorFormat1D
    from tensorkrowch.formats.quantics import _check_semantics

    if not isinstance(second, TensorFormat1D):
        raise TypeError('`other` should be TensorFormat1D type')
    first._ensure_valid()
    second._ensure_valid()
    _check_semantics(first, second, product=not same_family)
    if first.n_sites != second.n_sites:
        raise ValueError('Formats should have the same number of sites')
    if first.device != second.device:
        raise ValueError('Formats should share device')
    if first._batch_shape and second._batch_shape and first._batch_shape != second._batch_shape:
        raise ValueError(
            'Structural batches should match or one operand should be unbatched')
    if same_family and (
        first._in_dim != second._in_dim or first._out_dim != second._out_dim):
        raise ValueError('Formats should have matching input and output dimensions')
    dtype = torch.promote_types(first.dtype, second.dtype)
    a = [core.to(dtype=dtype) for core in first._standard_cores()]
    b = [core.to(dtype=dtype) for core in second._standard_cores()]
    batch = first._batch_shape or second._batch_shape
    a = [core.expand(*batch, *core.shape[-3:]) for core in a]
    b = [core.expand(*batch, *core.shape[-3:]) for core in b]
    cyclic = first._topology.startswith('tr') or second._topology.startswith('tr')
    return a, b, batch, cyclic


def add(first, second, method='stacked', coefficient=1):
    """Exact sum using stacked endpoints or fully block-diagonal cyclic cores.

    Stacked endpoints follow Mickelin and Karaman, Section 3.2 of
    https://arxiv.org/pdf/1807.02513. Closing ranks are padded to their maximum;
    internal ranks are summed. No compression or rounding is performed.
    """
    if method not in ('stacked', 'block_diagonal'):
        raise ValueError('`method` should be "stacked" or "block_diagonal"')
    a, b, batch, cyclic = _binary_inputs(first, second)
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
    from tensorkrowch.formats.quantics import _inherit_semantics

    return _inherit_semantics(_build_network(
        cores, first._in_dim, first._out_dim, len(batch), cyclic), first)


def hadamard(first, second):
    """Contracts matching physical indices and forms products of bond ranks."""
    a, b, batch, cyclic = _binary_inputs(first, second)
    cores = []
    for x, y in zip(a, b):
        core = torch.einsum('...lpr,...aps->...laprs', x, y)
        cores.append(core.reshape(*batch, x.shape[-3] * y.shape[-3],
                                  x.shape[-2], x.shape[-1] * y.shape[-1]))
    from tensorkrowch.formats.quantics import _inherit_semantics

    return _inherit_semantics(_build_network(
        cores, first._in_dim, first._out_dim, len(batch), cyclic), first)


def scale(network, coefficient):
    """Scales one core while preserving the represented network topology."""
    if isinstance(coefficient, torch.Tensor):
        if coefficient.ndim != 0:
            raise ValueError('The scaling tensor should be scalar')
        if coefficient.device != network.device:
            raise ValueError('The scaling tensor should share the network device')
    elif isinstance(coefficient, bool) or not isinstance(coefficient, Number):
        raise TypeError('The scaling coefficient should be a number or scalar tensor')
    network._ensure_valid()
    cores = list(network._standard_cores())
    cores[0] = cores[0] * coefficient
    dtype = cores[0].dtype
    cores = [core.to(dtype=dtype) for core in cores]
    from tensorkrowch.formats.quantics import _inherit_semantics

    return _inherit_semantics(_build_network(cores, network._in_dim, network._out_dim,
                                             network._n_batches, network._topology.startswith('tr')), network)


def apply(first, second):
    """Matrix-vector, vector-matrix and matrix-matrix products without densification."""
    a, b, batch, cyclic = _binary_inputs(first, second, same_family=False)
    left_matrix, right_matrix = first._out_dim is not None, second._out_dim is not None
    if not (left_matrix or right_matrix):
        raise TypeError('At least one operand of @ should be a matrix format')
    if left_matrix:
        contracted = second._out_dim if right_matrix else second._in_dim
        if first._in_dim != contracted:
            raise ValueError('Contracted local dimensions should match')
    elif first._in_dim != second._out_dim:
        raise ValueError('Contracted local dimensions should match')
    cores = []
    for site, (x, y) in enumerate(zip(a, b)):
        if left_matrix:
            x = x.reshape(*batch, x.shape[-3], first._in_dim[site],
                          first._out_dim[site], x.shape[-1]).transpose(-1, -2)
        if right_matrix:
            y = y.reshape(*batch, y.shape[-3], second._in_dim[site],
                          second._out_dim[site], y.shape[-1]).transpose(-1, -2)
        if left_matrix and right_matrix:
            core = torch.einsum('...liro,...ajbi->...lajorb', x, y)
            in_dim, out_dim = second._in_dim, first._out_dim
            physical = in_dim[site] * out_dim[site]
        elif left_matrix:
            core = torch.einsum('...liro,...aib->...laorb', x, y)
            in_dim, out_dim = first._out_dim, None
            physical = in_dim[site]
        else:
            core = torch.einsum('...lob,...airo->...laibr', x, y)
            in_dim, out_dim = second._in_dim, None
            physical = in_dim[site]
        cores.append(core.reshape(*batch, x.shape[-4 if left_matrix else -3] *
                                  y.shape[-4 if right_matrix else -3], physical,
                                  x.shape[-2 if left_matrix else -1] *
                                  y.shape[-2 if right_matrix else -1]))
    from tensorkrowch.formats.quantics import _inherit_semantics

    return _inherit_semantics(_build_network(cores, in_dim, out_dim, len(batch), cyclic),
                              first, second, product=True)
