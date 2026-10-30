"""Tests for fixed-rank and discovered TT-to-TR conversion."""

import pytest
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions.ring.blocks import _discovery_partition
from tensorkrowch.decompositions.ring.tt2tr import _TTCoreProvider


def _ring(length=7, ranks=2, dimension=4, dtype=torch.complex128):
    generator = torch.Generator().manual_seed(96)
    if isinstance(ranks, int):
        ranks = [ranks] * length
    return tk.formats.TR([
        torch.randn(ranks[k - 1], dimension, ranks[k], dtype=dtype,
                    generator=generator) for k in range(length)])


def _opener():
    return tk.decompositions.QRLoopOpener(
        tk.decompositions.ALSLoopOpener({
            'generator': torch.Generator().manual_seed(101),
            'convergence': tk.decompositions.ConvergencePolicy(
                max_sweeps=200, rtol=1e-10, keep_best=True),
        }))


def _error(tt, result):
    return (tt.contract_dense() - result.contract_dense()).norm() / tt.norm()


def _vidal(tt, mode='inverse', powers=None):
    """Prepares an input Vidal format, including nonuniform bond powers."""
    from tensorkrowch.formats.formats1d import _redistribute

    tt = tt.clone()
    tt.canonicalize_vidal(mode)
    if powers is not None:
        cores, factors = _redistribute(
            [core.clone() for core in tt._standard_cores()],
            tt.bonds.spectra, tt.bonds.powers, powers)
        tt._set_standard_cores(
            cores, factors, spectra=tt.bonds.spectra, powers=powers)
    return tt


def _prepare(tt, mode):
    if mode is None:
        return tt
    if mode == 'mixed':
        return _vidal(tt, 'explicit',
                      [(0., 1.), (.7, -.2), (1., 1.), (.25, .6),
                       (1., 0.), (.5, .5)][:tt.n_sites - 1])
    return _vidal(tt, mode)


class TestTT2TR:

    @pytest.mark.parametrize('length', [1, 2, 3, 7])
    @pytest.mark.parametrize('rank', [1, None])
    def test_minimal_and_general_rank_one(self, length, rank):
        tt = _ring(length, ranks=1, dtype=torch.float64).to_tt()
        result = tk.decompositions.tt2tr(tt, rank=rank, loop_opener=_opener())
        assert isinstance(result, tk.decompositions.TRDecomposition)
        assert result.rank == [1] * length
        assert _error(tt, result) < 1e-9

    @pytest.mark.parametrize('recursion', ['adjoint', 'pinv', 'orth'])
    @pytest.mark.parametrize('expand', [False, True])
    def test_exact_prescribed_rank_one(self, recursion, expand):
        tt = _ring(7, ranks=1).to_tt()
        result = tk.decompositions.TT2TR(tt).fit(
            [1] * 7, recursion=recursion, enrich=expand,
            loop_opener=_opener(), collect_metrics=True)
        assert _error(tt, result) < 1e-9
        assert result.metrics.fidelities
        assert result.metrics.local_solves

    @pytest.mark.parametrize(
        'gauge', ['explicit', 'implicit', 'inverse', None])
    def test_rank_two_and_vidal(self, gauge):
        tt = _ring(5).to_tt()
        if gauge is not None:
            tt = tt.canonicalize_vidal(gauge)
        before = tt.contract_dense().clone()
        result = tk.decompositions.tt2tr(tt, rank=2, loop_opener=_opener())
        assert result.rank == [2] * 5
        assert _error(tt, result) < 1e-6
        torch.testing.assert_close(tt.contract_dense(), before)
        if gauge is not None:
            assert 'vidal_powers' in result.metadata

    def test_nonuniform_prescribed_ranks_and_strategy(self):
        ranks = [1, 2, 1, 2, 1]
        tt = _ring(5, ranks=ranks).to_tt()
        result = tk.decompositions.tt2tr(
            tt, rank=ranks, loop_opener=_opener(),
            recursion=tk.decompositions.PinvGaugeRecursion(rtol=1e-12))
        assert result.rank == ranks
        assert _error(tt, result) < 1e-6

    def test_three_block_discovery(self):
        tt = _ring().to_tt()
        result = tk.decompositions.tt2tr(tt, loop_opener=_opener())
        assert result.metadata['requested_rank'] is None
        assert result.metadata['boundary_blocks'] == ((0, 1), (5, 6))
        assert result.metadata['center_block'] == (2, 3, 4)
        assert result.rank == [2] * 7
        assert _error(tt, result) < 1e-6

    @pytest.mark.parametrize('length', [3, 4, 5])
    def test_overlapping_discovery_blocks(self, length):
        tt = _ring(length).to_tt()
        result = tk.decompositions.tt2tr(tt, loop_opener=_opener())
        assert result.metadata['center_block'] == tuple(range(length))
        assert result.metadata['boundary_blocks'] == ((), ())
        assert _error(tt, result) < 1e-6

    @pytest.mark.parametrize('opener', ['blostr', 'blostr_svd'])
    @pytest.mark.parametrize('rank', [2, None])
    @pytest.mark.parametrize('rtol', [0., 1e-12])
    def test_exact_spectral_route(self, opener, rank, rtol):
        tt = _ring(5).to_tt()
        result = tk.decompositions.tt2tr(
            tt, rank=rank, loop_opener=opener, rtol=rtol)
        assert result.rank == [2] * 5
        assert result.metadata['exact']
        assert _error(tt, result) < 1e-8
        assert not result.metrics.local_solves

    def test_spectral_gaussian_nonuniform_block(self):
        ranks = [2, 1, 2, 1, 2, 1, 2, 2]
        tt = _ring(8, ranks=ranks).to_tt()
        opener = tk.decompositions.BLOSTRSVDLoopOpener({
            'projection': 'gaussian',
            'generator': torch.Generator().manual_seed(71),
        })
        result = tk.decompositions.tt2tr(
            tt, rank=ranks, center=3, loop_opener=opener)
        assert len(result.metadata['center_block']) >= 2
        assert result.rank == ranks
        assert _error(tt, result) < 1e-8

    @pytest.mark.parametrize('center', [0, 4])
    def test_prescribed_boundary_center(self, center):
        tt = _ring(5, ranks=1).to_tt()
        result = tk.decompositions.tt2tr(tt, rank=1, center=center)
        assert _error(tt, result) < 1e-9

    @pytest.mark.parametrize('kwargs,error', [
        ({'rank': True}, TypeError),
        ({'rank': 0}, ValueError),
        ({'rank': [1, 1]}, ValueError),
        ({'center': True}, TypeError),
        ({'center': -1}, ValueError),
        ({'center': 5}, ValueError),
        ({'block_boundaries': 1}, TypeError),
        ({'block_boundaries': False}, ValueError),
        ({'enrich': 1}, TypeError),
        ({'enrich_rank': -1}, ValueError),
        ({'enrich_rtol': -1}, ValueError),
        ({'rtol': -1}, ValueError),
        ({'recursion': 'gram'}, ValueError),
        ({'loop_opener': 'blostr+als'}, ValueError),
    ])
    def test_invalid_options(self, kwargs, error):
        tt = _ring(5, ranks=1).to_tt()
        with pytest.raises(error):
            tk.decompositions.tt2tr(tt, **kwargs)

    @pytest.mark.parametrize('value', [None, 'tt', 2, []])
    def test_invalid_input(self, value):
        with pytest.raises((TypeError, ValueError)):
            tk.decompositions.TT2TR(value)

    def test_raw_cores_and_open_mps(self):
        tt = _ring(3, ranks=1).to_tt()
        for value in (tt.cores, tk.models.MPS(
                n_features=3, phys_dim=4, bond_dim=1, boundary='obc',
                tensors=tt.cores)):
            result = tk.decompositions.tt2tr(value, rank=1)
            assert _error(tt, result) < 1e-9

    def test_prescribed_rank_grows_noninjective_start(self):
        tt = _ring(8, dimension=2).to_tt()
        result = tk.decompositions.tt2tr(tt, rank=2, loop_opener=_opener())
        assert len(result.metadata['center_block']) > 1
        assert result.rank == [2] * 8
        assert _error(tt, result) < 1e-6

    @pytest.mark.parametrize('rtol', [None, 0., .95])
    @pytest.mark.parametrize('ranks', [2, [1, 2, 2, 1, 2, 2, 1, 2, 2, 2]])
    def test_prescribed_boundary_blocks_and_svd_tolerance(self, rtol, ranks):
        tt = _ring(10, ranks=ranks, dimension=2).to_tt()
        result = tk.decompositions.tt2tr(
            tt, rank=ranks, loop_opener=_opener(), rtol=rtol,
            collect_metrics=True)
        assert result.metadata['boundary_blocks'] == ((0, 1, 2), (7, 8, 9))
        caps = [2] * 10 if isinstance(ranks, int) else ranks
        assert all(actual <= cap for actual, cap in zip(result.rank, caps))
        records = result.metrics.truncations
        assert sorted(record.site for record in records) == [0, 1, 7, 8]
        if (rtol is None) or (rtol == 0):
            assert result.rank == caps
            assert _error(tt, result) < 1e-6
        else:
            assert any(actual < cap for actual, cap
                       in zip(result.rank, caps))
            assert any(record.local_abs_error > 0 for record in records)
            assert _error(tt, result) > .01

    def test_prescribed_overlapping_blocks_open_whole_chain(self):
        tt = _ring(6, dimension=2).to_tt()
        result = tk.decompositions.tt2tr(tt, rank=2, loop_opener=_opener())
        assert result.metadata['center_block'] == tuple(range(6))
        assert result.metadata['boundary_blocks'] == ((), ())
        assert result.rank == [2] * 6
        assert _error(tt, result) < 1e-6

    @pytest.mark.parametrize('block_boundaries', [False, True])
    def test_boundary_block_option_preserves_automatic_initial_block(
            self, block_boundaries):
        tt = _ring(10, dimension=2).to_tt()
        result = tk.decompositions.tt2tr(
            tt, rank=2, loop_opener=_opener(),
            block_boundaries=block_boundaries)
        assert result.metadata['block_boundaries'] is block_boundaries
        assert result.metadata['center_block'] == (4, 5, 6)
        expected = ((0, 1, 2), (7, 8, 9)) if block_boundaries else ((0,), (9,))
        assert result.metadata['boundary_blocks'] == expected
        assert _error(tt, result) < 1e-6

    def test_disabling_boundary_blocks_avoids_initial_block_merging(self):
        tt = _ring(6, dimension=2).to_tt()
        result = tk.decompositions.tt2tr(
            tt, rank=2, loop_opener=_opener(), block_boundaries=False)
        assert result.metadata['center_block'] == (2, 3, 4)
        assert result.metadata['boundary_blocks'] == ((0,), (5,))
        assert _error(tt, result) < 1e-6

    def test_inexact_spectral_discovery_is_rejected(self):
        tt = _ring(8, ranks=[2, 1, 2, 1, 2, 1, 2, 2]).to_tt()
        opener = tk.decompositions.BLOSTRSVDLoopOpener({
            'generator': torch.Generator().manual_seed(71)})
        with pytest.raises(RuntimeError, match='BLOSTR'):
            tk.decompositions.tt2tr(tt, center=3, loop_opener=opener)

    def test_partition_capacity(self):
        provider = _TTCoreProvider(_ring().to_tt())
        central, left, right, ranks, info = _discovery_partition(
            provider, 3, 1e-12)
        assert (central, left, right) == ((2, 3, 4), (0, 1), (5, 6))
        assert ranks == (2,) * 7
        assert info['cyclic_rank'] == 2


class TestExpansionInvariants:

    @pytest.mark.parametrize(
        'mode', ['explicit', 'implicit', 'inverse', 'mixed'])
    @pytest.mark.parametrize('direction', ['left', 'right'])
    def test_vidal_expansion_preserves_orthogonal_error_identity(
            self, mode, direction):
        from tensorkrowch.decompositions.ring.tt2tr import _VidalCoreProvider
        from tensorkrowch.decompositions.ring.gauges import (
            _ExpandedGaugeRecursion)

        generator = torch.Generator().manual_seed(211)
        tt = _prepare(tk.decompositions.TTSVD(torch.randn(
            [3] * 5, dtype=torch.complex128, generator=generator)).fit(
                rank=2), mode)
        provider = _VidalCoreProvider(tt)
        target = provider.local_target((2,), {})
        left = torch.randn(2, target.shape[0], 2, dtype=target.dtype,
                           generator=generator)
        core = torch.randn(2, 3, 2, dtype=target.dtype, generator=generator)
        right = torch.randn(2, target.shape[-1], 2, dtype=target.dtype,
                            generator=generator)
        opening = tk.decompositions.LoopOpening(left, [core], right, (2,) * 3)
        strategy = tk.decompositions.AdjointGaugeRecursion()
        expanded = _ExpandedGaugeRecursion(strategy, None, 1e-12)
        site = 3 if direction == 'right' else 1
        context = {'provider': provider, 'from_sites': (2,),
                   'to_sites': (site,)}
        if direction == 'right':
            matrix = target.reshape(-1, target.shape[-1])
            basis = torch.einsum('gma,apb->mpgb', left, core).reshape(
                matrix.shape[0], -1)
        else:
            matrix = target.permute(1, 2, 0).reshape(-1, target.shape[0])
            basis = torch.einsum('apb,bng->pnag', core, right).reshape(
                matrix.shape[0], -1)
        q, x = strategy._isometric_coordinates(
            matrix, basis, direction, context)
        old_target = provider.cores[site].clone()
        step = expanded._advance(opening, old_target, context, direction)
        old_dim = step.diagnostics['old_dimension']
        coords = step.gauge.permute(1, 0, 2).reshape(step.gauge.shape[1], -1)
        bond = 2 if direction == 'right' else 1
        power = 1 - provider.powers[bond][1 if direction == 'right' else 0]
        scale = provider.spectra[bond].pow(power)
        residual = x - q @ (scale[:, None] * coords[:old_dim])
        w = torch.randn(x.shape[1], 4, dtype=x.dtype, generator=generator)
        torch.testing.assert_close((residual @ w).norm().square(),
                                   (coords[old_dim:] @ w).norm().square())
        padded = provider.cores[site]
        if direction == 'right':
            torch.testing.assert_close(padded[:old_dim], old_target)
            assert torch.count_nonzero(padded[old_dim:]) == 0
        else:
            torch.testing.assert_close(padded[:, :, :old_dim], old_target)
            assert torch.count_nonzero(padded[:, :, old_dim:]) == 0

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_expansion_exposes_zero_constraint(self, dtype):
        from tensorkrowch.decompositions.ring.tt2tr import (
            _CanonicalTTCoreProvider)
        from tensorkrowch.decompositions.ring.gauges import (
            _ExpandedGaugeRecursion)

        tt = tk.decompositions.TTDecomposition([
            torch.ones(1, 1, dtype=dtype),
            torch.tensor([1., 0.], dtype=dtype).reshape(1, 2, 1),
            torch.tensor([.6, .8], dtype=dtype).reshape(1, 2, 1),
            torch.ones(1, 1, dtype=dtype)])
        for extra in (0, None):
            provider = _CanonicalTTCoreProvider(tt, center=1)
            source = provider.local_target((1,), {})
            next_core = provider.local_target((2,), {}).clone()
            target_pair = source.reshape(2, 1) @ next_core.reshape(1, 2)
            left = torch.tensor([1., 0.], dtype=dtype).reshape(1, 1, 2)
            core = torch.zeros(2, 2, 2, dtype=dtype)
            core[0] = torch.eye(2, dtype=dtype)
            right = source.reshape(2, 1, 1)
            opening = tk.decompositions.LoopOpening(
                left, [core], right, (2, 2, 1))
            torch.testing.assert_close(opening.contract_dense(), source)
            strategy = tk.decompositions.AdjointGaugeRecursion()
            expanded = _ExpandedGaugeRecursion(strategy, extra, 1e-12)
            step = expanded.advance_right(
                opening, next_core, {'provider': provider,
                                    'from_sites': (1,), 'to_sites': (2,)})
            fixed = step.gauge[0]
            target = provider.local_target((2,), {}).reshape(
                fixed.shape[0], -1)
            factor = torch.linalg.pinv(fixed) @ target
            if extra == 0:
                null = (torch.eye(2, dtype=dtype) -
                        torch.linalg.pinv(fixed) @ fixed)
                factor = factor + null @ torch.ones(2, 2, dtype=dtype)
                assert (factor - target_pair).norm() > .5
            else:
                assert step.diagnostics['added'] == 1
                torch.testing.assert_close(factor, target_pair)
            torch.testing.assert_close(fixed @ factor, target)



class TestTT2TRIntegration:

    def test_general_vidal_powers_are_preserved(self):
        powers = [(0., 1.), (.7, -.2), (1., 1.), (.25, .6)]
        tt = _vidal(_ring(5, ranks=1).to_tt(), 'explicit', powers)
        before = tt.contract_dense().clone()
        result = tk.decompositions.tt2tr(tt, rank=1)
        torch.testing.assert_close(result.contract_dense(), before)
        assert tuple(result.metadata['vidal_powers']) == tuple(powers)
        assert list(tt.bonds.powers) == powers

    def test_observer_records_all_sites_and_closes_metrics(self, monkeypatch):
        from importlib import import_module
        from tensorkrowch.decompositions.observers import HistoryObserver

        observer = HistoryObserver()
        module = import_module('tensorkrowch.decompositions.ring.tt2tr')
        monkeypatch.setattr(
            module, '_resolve_observer', lambda *args: observer)
        result = tk.decompositions.tt2tr(
            _ring(5, ranks=1).to_tt(), rank=1, verbose=1)
        assert observer.events[0].name == 'start'
        assert sum(event.name == 'site_complete'
                   for event in observer.events) == 5
        assert observer.metrics is result.metrics
        assert any(event.name == 'summary' for event in observer.events)

    @pytest.mark.parametrize('explicit_bonds', [False, True])
    def test_quantics_result_keeps_input_diagnostics(self, explicit_bonds):
        metrics = tk.decompositions.DecompositionMetrics()
        tt = tk.decompositions.QTTDecomposition(
            [torch.ones(2, 1), torch.ones(1, 2, 1), torch.ones(1, 2)],
            n_coordinates=1, base=2, level=3, domain=(0., 1.),
            metrics=metrics, metadata={'fit': 'quantics'})
        if explicit_bonds:
            tt.bonds = [torch.tensor([2.]), torch.tensor([3.])]
        algorithm = tk.decompositions.TT2TR(tt)
        assert algorithm.tt is tt
        result = algorithm.fit(rank=1)
        torch.testing.assert_close(
            result.contract_dense(), tt.contract_dense())
        assert algorithm.tt.metrics is metrics

    def test_zero_target_and_closed_inputs_are_rejected(self):
        with pytest.raises(ValueError, match='nonzero'):
            tk.decompositions.tt2tr(tk.formats.TT([torch.zeros(2)]), rank=1)
        with pytest.raises(TypeError):
            tk.decompositions.TT2TR(_ring(3))
        model = tk.models.MPS(
            tensors=[torch.ones(1, 2, 1) for _ in range(3)],
            boundary='pbc', parameterized=False)
        with pytest.raises(ValueError, match='open boundaries'):
            tk.decompositions.TT2TR(model)
