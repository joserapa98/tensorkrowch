"""Tests for structured conversion from TT to TR."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.observers import HistoryObserver
from tensorkrowch.decompositions.ring.blocks import CentralBlockSelector


def _rank_one_tt(dtype=torch.float64, phase=1):
    generator = torch.Generator().manual_seed(150)
    cores = [
        torch.randn(2, 1, dtype=dtype, generator=generator),
        torch.randn(1, 3, 1, dtype=dtype, generator=generator),
        torch.randn(1, 2, 1, dtype=dtype, generator=generator),
        torch.randn(1, 4, 1, dtype=dtype, generator=generator),
        torch.randn(1, 2, dtype=dtype, generator=generator),
    ]
    cores[2] = cores[2] * phase
    return tk.decompositions.TTDecomposition(cores)


def _short_als():
    return tk.decompositions.ALSLoopOpener({
        'gauge': 'none',
        'normalize': False,
        'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=3),
    })


def _dense_overlap(first, second):
    first = first.reshape(-1)
    second = second.reshape(-1)
    return torch.vdot(first, second) / (first.norm() * second.norm())


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


class TestTT2TR:  # MARK: TestTT2TR

    def test_functional_api_preserves_general_vidal_powers(self, monkeypatch):
        powers = [(0., 1.), (.25, .75), (1., 1.), (-.2, .6)]
        tt = _vidal(_rank_one_tt(dtype=torch.complex128), 'explicit', powers)
        before = tt.contract_dense().clone()

        def no_recanonicalization(*args, **kwargs):
            raise AssertionError('A valid input Vidal gauge must be preserved')

        monkeypatch.setattr(tk.formats.TT, 'canonicalize_vidal',
                            no_recanonicalization)
        cores, info = tk.decompositions.tt2tr(tt, rank=1, return_info=True)
        torch.testing.assert_close(tk.formats.TR(cores).contract_dense(), before)
        assert info['metadata']['recursion'] == 'adjoint'
        assert info['metadata']['expand_directions']
        assert tuple(info['metadata']['vidal_powers']) == tuple(powers)
        assert list(tt.bonds.powers) == powers

    @pytest.mark.parametrize('kwargs,error', [
        ({'expand_directions': 1}, TypeError),
        ({'max_extra_rank': True}, TypeError),
        ({'max_extra_rank': -1}, ValueError),
        ({'residual_rtol': None}, TypeError),
        ({'residual_rtol': -1.}, ValueError),
        ({'schedule': 'alternating'}, ValueError),
        ({'recursion': 'unknown'}, ValueError),
    ])
    def test_recursion_and_expansion_validation(self, kwargs, error):
        with pytest.raises(error):
            tk.decompositions.TT2TR(_rank_one_tt()).fit(rank=1, **kwargs)

    @pytest.mark.parametrize('strategy,mode', [
        ('orth', None), ('pinv', None), ('gram', None), ('adjoint', None),
        ('adjoint', 'explicit'), ('adjoint', 'implicit'),
        ('adjoint', 'inverse'), ('adjoint', 'mixed')])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('block', [False, True])
    def test_expansion_preserves_targets_and_fixed_tr_ranks(
            self, strategy, mode, dtype, block):
        generator = torch.Generator().manual_seed(207)
        tt = _prepare(tk.decompositions.TTSVD(torch.randn(
            [3] * 7, dtype=dtype, generator=generator)).fit(rank=2), mode)
        before = [core.clone() for core in tt.cores]
        with torch.random.fork_rng():
            torch.manual_seed(208)
            result = tk.decompositions.TT2TR(tt).fit(
                rank=2, center=3, loop_opener=_short_als(), recursion=strategy,
                block_selector=CentralBlockSelector() if block else None)
        records = [r for r in result.metadata['enrichment'] if 'added' in r]
        assert records and any(r['added'] > 0 for r in records)
        assert all(float(r['residual_after']) < 1e-10 for r in records)
        assert all(float(r['orthogonality_error']) < 1e-11 for r in records)
        assert all(float(r.get('cross_orthogonality_error', 0)) < 1e-11
                   for r in records)
        assert all(float(r['target_preservation_error']) < 1e-12
                   for r in records)
        assert result.rank == [2] * 7
        assert all(torch.equal(a, b) for a, b in zip(before, tt.cores))

    @pytest.mark.parametrize('strategy,mode', [
        ('orth', None), ('pinv', None), ('gram', None), ('adjoint', None),
        ('adjoint', 'explicit'), ('adjoint', 'implicit'), ('adjoint', 'inverse')])
    def test_zero_expansion_preserves_original_fit(self, strategy, mode):
        tt = _prepare(tk.decompositions.TTSVD(torch.randn(
            [3] * 5, dtype=torch.complex128,
            generator=torch.Generator().manual_seed(209))).fit(rank=2), mode)
        results = []
        for expand in (False, True):
            with torch.random.fork_rng():
                torch.manual_seed(210)
                results.append(tk.decompositions.TT2TR(tt).fit(
                    rank=2, recursion=strategy, loop_opener=_short_als(),
                    expand_directions=expand, max_extra_rank=0))
        for first, second in zip(results[0].cores, results[1].cores):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    @pytest.mark.parametrize('mode', [None, 'explicit', 'inverse', 'mixed'])
    def test_expanded_subchain_metric_matches_dense_embedding(self, mode):
        from tensorkrowch.decompositions.ring.tt2tr import (
            _GramTTCoreProvider, _VidalCoreProvider)
        from tensorkrowch.decompositions.ring.gauges import _ExpandedGaugeRecursion

        generator = torch.Generator().manual_seed(212)
        tt = _prepare(tk.decompositions.TTSVD(torch.randn(
            [3] * 7, dtype=torch.complex128, generator=generator)).fit(rank=2), mode)
        provider = (_GramTTCoreProvider(tt) if mode is None else
                    _VidalCoreProvider(tt))
        strategy = (tk.decompositions.GramGaugeRecursion(allow_projective=True)
                    if mode is None else
                    tk.decompositions.AdjointGaugeRecursion(allow_projective=True))
        expanded = _ExpandedGaugeRecursion(strategy, None, 1e-12)
        prefix = provider.cores[0].squeeze(0)
        if mode is not None:
            prefix = prefix * provider.bond_factors[0]
        prefix = torch.einsum('sa,aib->sib', prefix, provider.cores[1])
        prefix = prefix.reshape(-1, prefix.shape[-1])
        if mode is not None:
            prefix = prefix * provider.bond_factors[1]
        for site in (2, 3, 4):
            target = provider.cores[site]
            left = torch.randn(2, target.shape[0], 2, dtype=target.dtype,
                               generator=generator)
            core = torch.randn(2, 3, 2, dtype=target.dtype, generator=generator)
            right = torch.randn(2, target.shape[-1], 2, dtype=target.dtype,
                                generator=generator)
            opening = tk.decompositions.LoopOpening(left, [core], right, (2,) * 3)
            factor = provider.left_factors[site]
            physical_frame = prefix @ torch.linalg.pinv(factor)
            torch.testing.assert_close(
                physical_frame.mH @ physical_frame,
                torch.eye(factor.shape[0], dtype=target.dtype))
            weighted_z = torch.einsum('xa,gab,bic->xigc', factor, left, core)
            weighted_z = weighted_z.reshape(factor.shape[0] * 3, -1)
            reference = torch.einsum('xa,aib->xib', factor, target).reshape(
                factor.shape[0] * 3, -1)
            q = torch.linalg.svd(reference, full_matrices=False).U
            residual = weighted_z - q @ (q.mH @ weighted_z)
            old_dim = target.shape[-1]
            step = expanded.advance_right(opening, provider.cores[site + 1], {
                'provider': provider, 'from_sites': (site,),
                'to_sites': (site + 1,)})
            h = step.gauge[:, old_dim:, :].permute(1, 0, 2).reshape(
                step.diagnostics['added'], -1)
            extra = residual @ torch.linalg.pinv(h)
            physical_extra = torch.einsum(
                'sx,xir->sir', physical_frame, extra.reshape(factor.shape[0], 3, -1))
            physical_extra = physical_extra.reshape(-1, h.shape[0])
            prefix = torch.einsum('sa,aib->sib', prefix, target).reshape(-1, old_dim)
            if mode is not None:
                prefix = prefix * provider.bond_factors[site]
            prefix = torch.cat((prefix, physical_extra), dim=1)
            next_factor = provider.left_factors[site + 1]
            torch.testing.assert_close(prefix.mH @ prefix,
                                       next_factor.mH @ next_factor)

    @pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse', 'mixed'])
    @pytest.mark.parametrize('direction', ['left', 'right'])
    def test_vidal_expansion_preserves_orthogonal_error_identity(
            self, mode, direction):
        from tensorkrowch.decompositions.ring.tt2tr import _VidalCoreProvider
        from tensorkrowch.decompositions.ring.gauges import _ExpandedGaugeRecursion

        generator = torch.Generator().manual_seed(211)
        tt = _prepare(tk.decompositions.TTSVD(torch.randn(
            [3] * 5, dtype=torch.complex128, generator=generator)).fit(rank=2), mode)
        provider = _VidalCoreProvider(tt)
        target = provider.local_target((2,), {})
        left = torch.randn(2, target.shape[0], 2, dtype=target.dtype,
                           generator=generator)
        core = torch.randn(2, 3, 2, dtype=target.dtype, generator=generator)
        right = torch.randn(2, target.shape[-1], 2, dtype=target.dtype,
                            generator=generator)
        opening = tk.decompositions.LoopOpening(left, [core], right, (2,) * 3)
        strategy = tk.decompositions.AdjointGaugeRecursion(allow_projective=True)
        expanded = _ExpandedGaugeRecursion(strategy, None, 1e-12)
        site = 3 if direction == 'right' else 1
        context = {'provider': provider, 'from_sites': (2,), 'to_sites': (site,)}
        if direction == 'right':
            matrix = target.reshape(-1, target.shape[-1])
            basis = torch.einsum('gma,apb->mpgb', left, core).reshape(
                matrix.shape[0], -1)
        else:
            matrix = target.permute(1, 2, 0).reshape(-1, target.shape[0])
            basis = torch.einsum('apb,bng->pnag', core, right).reshape(
                matrix.shape[0], -1)
        q, x = strategy._isometric_coordinates(matrix, basis, direction, context)
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
    def test_expansion_exposes_previously_invisible_zero_constraint(self, dtype):
        from tensorkrowch.decompositions.ring.tt2tr import _CanonicalTTCoreProvider
        from tensorkrowch.decompositions.ring.gauges import _ExpandedGaugeRecursion

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
            opening = tk.decompositions.LoopOpening(left, [core], right, (2, 2, 1))
            torch.testing.assert_close(opening.contract_dense(), source)
            strategy = tk.decompositions.AdjointGaugeRecursion()
            step = _ExpandedGaugeRecursion(strategy, extra, 1e-12).advance_right(
                opening, next_core, {'provider': provider,
                                    'from_sites': (1,), 'to_sites': (2,)})
            fixed = step.gauge[0]
            target = provider.local_target((2,), {}).reshape(fixed.shape[0], -1)
            factor = torch.linalg.pinv(fixed) @ target
            if extra == 0:
                null = torch.eye(2, dtype=dtype) - torch.linalg.pinv(fixed) @ fixed
                factor = factor + null @ torch.ones(2, 2, dtype=dtype)
                assert (factor - target_pair).norm() > .5
            else:
                assert step.diagnostics['added'] == 1
                torch.testing.assert_close(factor, target_pair)
            torch.testing.assert_close(fixed @ factor, target)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('strategy,mode', [
        ('orth', None), ('pinv', None), ('gram', None), ('adjoint', None),
        ('adjoint', 'explicit'), ('adjoint', 'inverse')])
    def test_boundaries_absorb_learned_gauges_without_recursion_duals(
            self, dtype, strategy, mode, monkeypatch):
        from tensorkrowch.decompositions.ring.tt2tr import (
            _CanonicalTTCoreProvider, _TTCoreProvider, _VidalCoreProvider,
            _resolve_recursion)

        tt = _prepare(tk.decompositions.TTSVD(torch.randn(
            3, 3, 3, dtype=dtype,
            generator=torch.Generator().manual_seed(198))).fit(), mode)
        recursion = _resolve_recursion(
            strategy, inverse_policy='pinv', allow_projective=True,
            tolerance=1e-8, rank_rtol=1e-10)

        def forbidden(*args, **kwargs):
            raise AssertionError('Boundary absorption must not dualize a gauge')

        monkeypatch.setattr(recursion, 'prepare_boundary', forbidden, raising=False)

        class RecordingALS(tk.decompositions.ALSLoopOpener):
            def open(self, *args, **kwargs):
                self.opening = super().open(*args, **kwargs)
                return self.opening

        opener = RecordingALS({'gauge': 'none', 'normalize': False,
                               'convergence': tk.decompositions.ConvergencePolicy(
                                   max_sweeps=2)})
        result = tk.decompositions.TT2TR(tt).fit(
            rank=2, loop_opener=opener, recursion=recursion)
        if mode is not None:
            provider = _VidalCoreProvider(tt)
        elif strategy == 'adjoint':
            provider = _CanonicalTTCoreProvider(tt, center=1)
        else:
            provider = _TTCoreProvider(tt)
        left, right = opener.opening.left_gauge, opener.opening.right_gauge
        if mode is not None:
            left = left * provider.bond_factors[0][None, :, None]
            right = right * provider.bond_factors[1][None, :, None]
        expected_left = torch.einsum(
            'dm,gma->gda', provider.cores[0].squeeze(0), left)
        expected_right = torch.einsum(
            'bmg,md->bdg', right, provider.cores[-1].squeeze(-1))
        torch.testing.assert_close(result.cores[0], expected_left)
        torch.testing.assert_close(result.cores[-1], expected_right)
        assert torch.equal(result.cores[1], opener.opening.cores[0])
        assert not result.metrics.gauges

    @pytest.mark.parametrize('strategy', ['adjoint', 'pinv', 'gram', 'orth'])
    @pytest.mark.parametrize('center', [1, 2, 3])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_rank_one_strategies_preserve_tensor_and_input(
            self, strategy, center, dtype):
        tt = _rank_one_tt(dtype=dtype)
        before = [core.clone() for core in tt.cores]
        result = tk.decompositions.TT2TR(tt).fit(
            rank=1, center=center, recursion=strategy)
        assert result.rank == [1] * 5
        torch.testing.assert_close(result.contract_dense(), tt.contract_dense())
        assert all(torch.equal(a, b) for a, b in zip(before, tt.cores))
        assert result.metadata['recursion'] == strategy

    @pytest.mark.parametrize('strategy,mode', [
        ('adjoint', None), ('pinv', None), ('gram', None), ('orth', None),
        ('adjoint', 'explicit'), ('adjoint', 'implicit'), ('adjoint', 'inverse')])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('block', [False, True])
    def test_exact_rank_two_tr_recovery(self, strategy, mode, dtype, block):
        generator = torch.Generator().manual_seed(218)
        tr = tk.formats.TR([
            torch.randn(2, 4, 2, dtype=dtype, generator=generator)
            for _ in range(5)])
        dense = tr.contract_dense()
        tt = tk.decompositions.TTSVD(dense).fit(rtol=1e-13)
        tt.canonicalize(orth_center=2)
        tt = _prepare(tt, mode)
        opener = tk.decompositions.SVDLoopOpener(
            tk.decompositions.ALSLoopOpener({
                'gauge': 'none',
                'normalize': False,
                'convergence': tk.decompositions.ConvergencePolicy(
                    max_sweeps=1000, error_rtol=1e-11),
            }), rtol=1e-13, refinement_sweeps=1)
        with torch.random.fork_rng():
            torch.manual_seed(221)
            result = tk.decompositions.TT2TR(tt).fit(
                rank=2, center=2, recursion=strategy, loop_opener=opener,
                block_selector=CentralBlockSelector() if block else None)
        relative = (result.contract_dense() - dense).norm() / dense.norm()
        assert relative < 2e-9
        assert result.rank == [2] * 5

    @pytest.mark.parametrize('strategy_type', [
        tk.decompositions.AdjointGaugeRecursion,
        tk.decompositions.PinvGaugeRecursion,
        tk.decompositions.GramGaugeRecursion,
        tk.decompositions.OrthGaugeRecursion])
    def test_strategy_objects_work_in_functional_api(self, strategy_type):
        tt = _rank_one_tt(torch.complex128)
        strategy = strategy_type()
        cores, info = tk.decompositions.tt2tr(
            tt, rank=1, recursion=strategy, return_info=True)
        torch.testing.assert_close(tk.formats.TR(cores).contract_dense(),
                                   tt.contract_dense())
        assert info['metadata']['recursion'] == strategy_type.__name__

    @pytest.mark.parametrize('strategy', ['adjoint', 'pinv', 'gram', 'orth'])
    def test_strategies_handle_exact_zero_reference_directions(self, strategy):
        edge = torch.tensor([[1., 0.], [2., 0.]], dtype=torch.complex128)
        core = torch.zeros(2, 2, 2, dtype=torch.complex128)
        core[0, :, 0] = torch.tensor([1., 3.])
        tt = tk.formats.TT([edge, core, core.clone(), edge.T])
        result = tk.decompositions.TT2TR(tt).fit(rank=1, recursion=strategy)
        assert torch.isfinite(result.contract_dense()).all()
        torch.testing.assert_close(result.contract_dense(), tt.contract_dense())

    def test_optional_local_metrics_preserve_conversion(self):
        results = []
        for collect_metrics in (False, True):
            with torch.random.fork_rng():
                torch.manual_seed(155)
                results.append(tk.decompositions.TT2TR(_rank_one_tt()).fit(
                    rank=1, collect_metrics=collect_metrics))

        fast, measured = results
        assert not fast.metrics.local_solves
        assert measured.metrics.local_solves
        assert len(fast.metrics.fidelities) == 1
        assert torch.allclose(fast.contract_dense(), measured.contract_dense())

    @pytest.mark.parametrize('center', [1, 2, 3])
    @pytest.mark.parametrize(
        ('dtype', 'phase'),
        [(torch.float64, 1.0), (torch.complex128, 1.0 + 0.5j)])
    def test_rank_one_conversion_is_exact_for_all_internal_centers(
            self, center, dtype, phase):
        tt = _rank_one_tt(dtype=dtype, phase=phase)
        result = tk.decompositions.TT2TR(
            tt, out_device=None).fit(rank=1, center=center)

        assert result.rank == [1, 1, 1, 1, 1]
        assert torch.allclose(
            result.contract_dense(),
            tt.contract_dense(),
            rtol=2e-9,
            atol=2e-9)
        assert result.metrics.fidelities[0].fidelity == pytest.approx(
            1.0, rel=2e-10, abs=2e-10)
        # The contraction-based squared residual has an O(sqrt(eps)) floor
        # when two independently gauged networks are effectively identical.
        assert result.metrics.errors[0].absolute < 1e-7
        assert result.metadata['center'] == center
        assert result.metadata['adaptive'] is False
        assert set(result.metadata['boundaries']) == {0, 4}

    def test_requested_nonuniform_cyclic_rank_is_preserved(self):
        dense = torch.randn(
            2, 2, 2, 2, 2,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(151))
        tt = tk.decompositions.TTSVD(
            dense, out_device=None).fit()
        result = tk.decompositions.TT2TR(
            tt, out_device=None).fit(
                rank=2,
                tr_rank=1,
                center=2,
                loop_opener=_short_als(),
                allow_projective_gauges=True)

        assert result.rank == [2, 2, 2, 2, 1]
        assert result.metadata['requested_rank'] == [2, 2, 2, 2, 1]

    def test_metrics_match_dense_oracles_for_nontrivial_ranks(self):
        dense = torch.randn(
            2, 2, 2, 2, 2,
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(152))
        tt = tk.decompositions.TTSVD(
            dense, out_device=None).fit()
        result = tk.decompositions.TT2TR(
            tt, out_device=None).fit(
                rank=2,
                tr_rank=2,
                center=2,
                loop_opener=_short_als(),
                allow_projective_gauges=True)
        approximation = result.contract_dense()
        overlap = _dense_overlap(dense, approximation)
        relative = (dense - approximation).norm() / dense.norm()
        fidelity = result.metrics.fidelities[0]

        assert result.rank == [2, 2, 2, 2, 2]
        assert fidelity.normalized_overlap == pytest.approx(
            overlap.item(), rel=2e-10, abs=2e-10)
        assert fidelity.fidelity == pytest.approx(
            overlap.abs().square().item(), rel=2e-10, abs=2e-10)
        assert result.metrics.errors[0].relative == pytest.approx(
            relative.item(), rel=2e-10, abs=2e-10)
        assert result.metrics.timings[0].name == 'fit'

    def test_projective_gauges_are_rejected_unless_explicitly_allowed(self):
        tt = _rank_one_tt()
        with pytest.raises(ValueError, match='Cannot cancel'):
            tk.decompositions.TT2TR(
                tt, out_device=None).fit(
                    rank=2,
                    tr_rank=2,
                    loop_opener=_short_als(),
                    recursion='orth', allow_projective_gauges=False)

        result = tk.decompositions.TT2TR(
            tt, out_device=None).fit(
                rank=2,
                tr_rank=2,
                loop_opener=_short_als(),
                recursion='orth')
        assert result.rank == [2, 2, 2, 2, 2]
        assert any(record.projective for record in result.metrics.gauges)

    def test_functional_api_and_mps_adapter(self):
        tt = _rank_one_tt()
        model = tk.models.MPS(tensors=tt.cores, parameterized=False)
        cores, info = tk.decompositions.tt2tr(
            model,
            rank=1,
            out_device=None,
            return_info=True)

        assert [tuple(core.shape) for core in cores] == [
            (1, 2, 1),
            (1, 3, 1),
            (1, 2, 1),
            (1, 4, 1),
            (1, 2, 1),
        ]
        assert info['topology'] == 'tr'
        assert info['rank'] == [1, 1, 1, 1, 1]
        assert info['metrics']['fidelities'][0]['fidelity'] == pytest.approx(1)

    def test_blostr_als_preset_falls_back_to_exact_als(self):
        tt = _rank_one_tt()
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TT2TR(
                tt, out_device=None).fit(
                    rank=1,
                    loop_opener='blostr+als')

        assert result.rank == [1, 1, 1, 1, 1]
        assert torch.allclose(
            result.contract_dense(),
            tt.contract_dense().to(result.cores[0].dtype),
            rtol=2e-9,
            atol=2e-9)

    def test_alternating_schedule_is_exact_for_odd_rank_one_tt(self):
        tt = _rank_one_tt()
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TT2TR(
                tt, out_device=None).fit(
                    rank=1,
                    schedule='alternating', recursion='orth',
                    expand_directions=False)

        assert result.metadata['schedule'] == 'alternating'
        assert result.metadata['requested_schedule'] == 'alternating'
        assert torch.allclose(
            result.contract_dense(), tt.contract_dense(),
            rtol=2e-9, atol=2e-9)

    def test_history_observer_receives_structured_steps_and_summary(self, monkeypatch):
        observer = HistoryObserver()
        from importlib import import_module
        monkeypatch.setattr(
            import_module('tensorkrowch.decompositions.ring.tt2tr'),
            '_resolve_observer', lambda *args: observer)
        completed_before_open = []
        original_open = tk.decompositions.ALSLoopOpener.open

        def open_loop(opener, *args, **kwargs):
            completed_before_open.append(sum(
                event.name == 'site_complete' for event in observer.events))
            return original_open(opener, *args, **kwargs)

        monkeypatch.setattr(tk.decompositions.ALSLoopOpener, 'open', open_loop)
        result = tk.decompositions.TT2TR(
            _rank_one_tt(), out_device=None).fit(
                rank=1, verbose=1)

        assert observer.metrics is result.metrics
        assert completed_before_open[0] == 0
        assert all(count > 0 for count in completed_before_open[1:])
        assert observer.events[0].name == 'start'
        assert sum(event.name == 'site_complete'
                   for event in observer.events) == 5
        summary = next(i for i, event in enumerate(observer.events)
                       if event.name == 'summary')
        assert all(event.name == 'core'
                   for event in observer.events[summary + 1:])
        assert len(observer.events[summary + 1:]) == 5

    def test_input_and_rank_validation(self):
        with pytest.raises(ValueError, match='at least three'):
            tk.decompositions.TT2TR([
                torch.randn(2, 2), torch.randn(2, 2)])
        with pytest.raises(ValueError, match='internal TT site'):
            tk.decompositions.TT2TR(
                _rank_one_tt(), out_device=None).fit(rank=1, center=0)
        with pytest.raises(ValueError, match='positive'):
            tk.decompositions.TT2TR(
                _rank_one_tt(), out_device=None).fit(rank=0)

    def test_input_requires_an_open_tensor_train(self):
        tr = tk.formats.TR([torch.ones(1, 2, 1) for _ in range(3)])
        with pytest.raises(TypeError, match='TT'):
            tk.decompositions.TT2TR(tr)
        matrix = tk.formats.TTM([
            torch.ones(2, 1, 2), torch.ones(1, 2, 1, 2), torch.ones(1, 2, 2)])
        with pytest.raises(TypeError, match='TT'):
            tk.decompositions.TT2TR(matrix)
        model = tk.models.MPS(
            tensors=[torch.ones(1, 2, 1) for _ in range(3)],
            boundary='pbc', parameterized=False)
        with pytest.raises(ValueError, match='open boundaries'):
            tk.decompositions.TT2TR(model)


@pytest.mark.parametrize('explicit_bonds', [False, True])
def test_tt2tr_accepts_quantics_result_with_historical_diagnostics(explicit_bonds):
    metrics = tk.decompositions.DecompositionMetrics()
    result = tk.decompositions.QTTDecomposition(
        [torch.ones(2, 1), torch.ones(1, 2, 1), torch.ones(1, 2)],
        n_coordinates=1, base=2, level=3, domain=(0., 1.),
        metrics=metrics, metadata={'fit': 'quantics'})
    if explicit_bonds:
        result.bonds = [torch.tensor([2.]), torch.tensor([3.])]

    algorithm = tk.decompositions.TT2TR(result)
    assert algorithm.tt.metrics is metrics
    assert algorithm.tt.metadata == result.metadata
    assert algorithm.tt is result
    torch.testing.assert_close(algorithm.tt.contract_dense(), result.contract_dense())
    converted = algorithm.fit(rank=1)
    torch.testing.assert_close(converted.contract_dense(), result.contract_dense())
