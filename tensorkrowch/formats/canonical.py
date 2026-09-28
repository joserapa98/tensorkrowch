"""QR/RQ gauge sweeps and materialization of explicit bond factors."""

import torch

from tensorkrowch.utils import truncated_svd
from .bonds import VidalGauge


def _redistribute(cores, gauge, powers):
    """Moves stored Schmidt powers between neighbours without another SVD."""
    for site, (spectrum, old, new) in enumerate(zip(gauge.spectra, gauge.powers, powers)):
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
            factor = factor[..., :, None, None] if left_axis else factor[..., None, None, :]
            cores[neighbour] = cores[neighbour] * factor
    return cores, VidalGauge(gauge.spectra, powers)


def materialize_bonds(network, oc=None):
    network._ensure_valid()
    oc = network.n_sites - 1 if oc is None else oc
    if isinstance(oc, bool) or not isinstance(oc, int):
        raise TypeError('`oc` should be int type or None')
    if not 0 <= oc < network.n_sites:
        raise ValueError('`oc` should select a valid site')
    if network._bonds is None:
        return network
    cores = list(network._raw_standard_cores())
    network._bonds.validate(cores, network._topology.startswith('tr'))
    if isinstance(network._bonds, VidalGauge):
        powers = [(0, 1) if site < oc else (1, 0)
                  for site in range(len(network._bonds.spectra))]
        cores, _ = _redistribute(cores, network._bonds, powers)
        network._set_standard_cores(cores)
        return network
    for site, value in enumerate(network._bonds.values):
        if value is None:
            continue
        if site >= oc:
            cores[site] = cores[site] * value[..., None, None, :]
        else:
            neighbour = (site + 1) % len(cores)
            cores[neighbour] = cores[neighbour] * value[..., :, None, None]
    network._set_standard_cores(cores)
    return network


def canonicalize_vidal(network, mode, inverse_positions, remaining_mode, inverse_cutoff):
    from math import isfinite
    from numbers import Real
    from .operations import _build_network
    network._ensure_valid()
    if network._topology.startswith('tr'):
        raise ValueError('Global Vidal canonicalization requires an open chain')
    modes = {'explicit': (0, 0), 'implicit': (0.5, 0.5), 'inverse': (1, 1)}
    if mode not in modes or remaining_mode not in ('explicit', 'implicit'):
        raise ValueError('Invalid Vidal mode or remaining_mode')
    if isinstance(inverse_cutoff, bool) or not isinstance(inverse_cutoff, Real):
        raise TypeError('`inverse_cutoff` should be a real number')
    if not isfinite(inverse_cutoff) or inverse_cutoff < 0:
        raise ValueError('`inverse_cutoff` should be finite and non-negative')
    count = network.n_sites - 1
    if inverse_positions is None:
        positions = set(range(count)) if mode == 'inverse' else set()
        powers = [modes[mode]] * count
    else:
        positions_list = list(inverse_positions)
        if mode == 'inverse':
            raise ValueError('Select either mode="inverse" or inverse_positions')
        if any(isinstance(site, bool) or not isinstance(site, int) for site in positions_list):
            raise TypeError('Inverse bond positions should be integers')
        if len(set(positions_list)) != len(positions_list) or any(
                site < 0 or site >= count for site in positions_list):
            raise ValueError('Inverse bond positions should be distinct valid bonds')
        positions = set(positions_list)
        powers = [modes['inverse' if site in positions else remaining_mode]
                  for site in range(count)]
    if isinstance(network._bonds, VidalGauge) and network._bonds._valid:
        cores = list(network._raw_standard_cores())
        gauge = network._bonds
    else:
        work = _build_network(network._standard_cores(), network._in_dim,
                              network._out_dim, network._n_batches, False)
        work.canonicalize(oc=0)
        cores = list(work._standard_cores())
        spectra = []
        batch = network._batch_shape
        for site in range(count):
            core = cores[site]
            u, s, vh = truncated_svd(core.reshape(*batch, -1, core.shape[-1]))
            cores[site] = u.reshape(*batch, core.shape[-3], core.shape[-2], s.shape[-1])
            if site:
                previous = spectra[-1]
                safe = torch.where(previous > 0, previous, torch.ones_like(previous))
                inverse = torch.where(previous > 0, safe.reciprocal(), torch.zeros_like(previous))
                cores[site] = cores[site] * inverse[..., :, None, None]
            spectra.append(s)
            cores[site + 1] = torch.einsum('...ab,...bpr->...apr',
                                         s.unsqueeze(-1) * vh, cores[site + 1])
        if count:
            last = spectra[-1]
            safe = torch.where(last > 0, last, torch.ones_like(last))
            inverse = torch.where(last > 0, safe.reciprocal(), torch.zeros_like(last))
            cores[-1] = cores[-1] * inverse[..., :, None, None]
        gauge = VidalGauge(spectra, [(0, 0)] * count)
    for site in positions:
        if torch.any(gauge.spectra[site] <= inverse_cutoff):
            raise ValueError(f'Inverse Vidal bond {site} has values at or below inverse_cutoff')
    cores, gauge = _redistribute(cores, gauge, powers)
    network._set_standard_cores(cores, gauge)
    return network


def canonicalize(network, oc=None, renormalize=False):
    network._ensure_valid()
    oc = network.n_sites - 1 if oc is None else oc
    if isinstance(oc, bool) or not isinstance(oc, int):
        raise TypeError('`oc` should be int type or None')
    if not 0 <= oc < network.n_sites:
        raise ValueError('`oc` should select a valid site')
    if not isinstance(renormalize, bool):
        raise TypeError('`renormalize` should be bool type')
    cores = list(network._standard_cores())
    if not all(torch.isfinite(core).all() for core in cores):
        raise ValueError('Canonicalization requires finite cores')
    batch = network._batch_shape
    log_scale = cores[0].real.new_zeros(batch)
    for site in range(oc):
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
    for site in range(len(cores) - 1, oc, -1):
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
        cores[oc] = cores[oc] * log_scale.exp()[..., None, None, None]
    network._set_standard_cores(cores)
    network._orth_center = oc
    return network
