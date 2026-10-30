"""Tests for oriented gauge maps and cancellation diagnostics."""

import math
from types import SimpleNamespace

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.observers import DecompositionEvent


def _full_rank_core(orientation, dtype=torch.float64):
    matrix = torch.randn(
        8, 4, dtype=dtype, generator=torch.Generator().manual_seed(140))
    if orientation == 'left':
        return matrix.reshape(8, 2, 2).permute(1, 0, 2)
    return matrix.reshape(8, 2, 2).permute(2, 0, 1)


class TestGaugeMapOrientation:  # MARK: TestGaugeMapOrientation

    @pytest.mark.parametrize('orientation', ['left', 'right'])
    def test_matrixization_and_mirror_preserve_the_map(self, orientation):
        core = _full_rank_core(orientation)
        gauge = tk.decompositions.GaugeMap(core, orientation, site=2)
        mirrored = gauge.mirror()

        assert gauge.matrix.shape == (8, 4)
        assert mirrored.orientation != orientation
        assert torch.equal(mirrored.core, core.permute(2, 1, 0))
        assert torch.equal(mirrored.matrix, gauge.matrix)
        assert mirrored.site == 2

    @pytest.mark.parametrize('orientation', ['left', 'right'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_pseudoinverse_dual_cancels_real_and_complex_gauges(
            self, orientation, dtype):
        gauge = tk.decompositions.GaugeMap(
            _full_rank_core(orientation, dtype), orientation)
        dual = gauge.pinv()
        product = gauge.matrix.T @ dual.matrix

        assert dual.orientation == orientation
        assert dual.is_dual
        assert torch.allclose(
            product,
            torch.eye(4, dtype=dtype),
            rtol=2e-10,
            atol=2e-10)
        if dtype == torch.complex128:
            assert not torch.allclose(
                gauge.matrix.mH @ dual.matrix,
                torch.eye(4, dtype=dtype),
                rtol=2e-10,
                atol=2e-10)
        record = dual.diagnostics()
        assert isinstance(record, tk.decompositions.GaugeRecord)
        assert record.numerical_rank == 4
        assert record.cancellable
        assert not record.projective


class TestGaugeMapPseudoinverse:

    @pytest.mark.parametrize('options, expected_rank', [
        ({}, 3),
        ({'rtol': 1e-3}, 1),
        ({'atol': 1e-7}, 2),
        ({'atol': torch.tensor(1e-7), 'rtol': torch.tensor(0.)}, 2),
        ({'hermitian': True, 'rtol': 1e-3}, 1),
    ])
    def test_cutoffs_match_pytorch_and_diagnostics(
            self, options, expected_rank):
        matrix = torch.diag(torch.tensor(
            [1., 1e-4, 1e-9, 0.], dtype=torch.float64))
        gauge = tk.decompositions.GaugeMap(
            matrix.reshape(4, 2, 2).permute(1, 0, 2), 'left')
        dual = gauge.pinv(**options)
        torch.testing.assert_close(
            dual.matrix, torch.linalg.pinv(matrix, **options).T)
        assert dual.diagnostics().numerical_rank == expected_rank
        assert dual.diagnostics().projective

    def test_zero_gauge_has_finite_projection_error(self):
        gauge = tk.decompositions.GaugeMap(torch.zeros(1, 2, 2), 'left')
        dual = gauge.pinv()
        assert torch.equal(dual.core, torch.zeros_like(dual.core))
        assert dual.diagnostics().numerical_rank == 0
        assert dual.diagnostics().cancellation_error == 1
        assert math.isinf(dual.diagnostics().condition_number)

    @pytest.mark.parametrize('options, error', [
        ({'rtol': -1}, ValueError),
        ({'atol': float('nan')}, ValueError),
        ({'rtol': True}, TypeError),
        ({'rtol': torch.ones(2)}, ValueError),
        ({'atol': torch.tensor(1j)}, ValueError),
        ({'hermitian': 1}, TypeError),
        ({'inverse_policy': 'solve'}, TypeError),
    ])
    def test_invalid_options_fail_at_construction(self, options, error):
        with pytest.raises(error):
            tk.decompositions.PinvGaugeRecursion(**options)
        with pytest.raises(error):
            tk.decompositions.GaugeMap(
                torch.ones(1, 1, 1), 'left').pinv(**options)

    @pytest.mark.parametrize('core, orientation, error', [
        (None, 'left', TypeError),
        (torch.ones(2, 2), 'left', ValueError),
        (torch.ones(1, 0, 1), 'left', ValueError),
        (torch.ones(1, 2, 1, dtype=torch.long), 'left', TypeError),
        (torch.ones(1, 2, 1), 'invalid', ValueError),
    ])
    def test_invalid_gauges(self, core, orientation, error):
        with pytest.raises(error):
            tk.decompositions.GaugeMap(core, orientation)


class TestGaugeMapDiagnostics:  # MARK: TestGaugeMapDiagnostics

    @pytest.mark.parametrize('orientation', ['left', 'right'])
    def test_wide_gauge_is_explicitly_projective(self, orientation):
        matrix = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0]])
        if orientation == 'left':
            core = matrix.reshape(2, 2, 2).permute(1, 0, 2)
        else:
            core = matrix.reshape(2, 2, 2).permute(2, 0, 1)
        gauge = tk.decompositions.GaugeMap(
            core, orientation, site=3).pinv()
        record = gauge.diagnostics()

        assert record.numerical_rank == 2
        assert record.cancellable_rank == 4
        assert record.projective
        assert record.cancellation_error == pytest.approx(math.sqrt(0.5))
        assert not record.cancellable
        assert torch.isfinite(gauge.core).all()

    def test_rank_tolerance_controls_numerical_rank_and_pseudoinverse(self):
        matrix = torch.diag(torch.tensor([1.0, 1e-6, 1e-10, 0.0]))
        core = matrix.reshape(4, 2, 2).permute(1, 0, 2)
        dual = tk.decompositions.GaugeMap(
            core, 'left').pinv(rtol=1e-8)
        record = dual.diagnostics()

        assert record.numerical_rank == 2
        assert record.rank_tolerance == pytest.approx(1e-8)
        assert math.isinf(record.condition_number)

    def test_structured_event_contains_the_record_diagnostics(self):
        dual = tk.decompositions.GaugeMap(
            _full_rank_core('right'),
            'right',
            site=1).pinv()
        event = dual.as_event('TT to TR')

        assert isinstance(event, DecompositionEvent)
        assert event.name == 'gauge'
        assert event.phase == 'TT to TR'
        assert event.site == 1
        assert event.values['rank'] == (4, 4)
        assert not event.values['projective']


class TestOrthGaugeRecursion:  # MARK: TestOrthGaugeRecursion

    @pytest.mark.parametrize(
        ('direction', 'outgoing_orientation', 'fixed_orientation'),
        [('left', 'left', 'right'), ('right', 'right', 'left')])
    def test_directional_recursion_dualizes_and_mirrors_outgoing_gauge(
            self, direction, outgoing_orientation, fixed_orientation):
        opening = tk.decompositions.LoopOpening(
            left_gauge=_full_rank_core('left'),
            cores=(torch.randn(2, 3, 2, dtype=torch.float64),),
            right_gauge=_full_rank_core('right'),
            rank=(2, 2, 2))
        recursion = tk.decompositions.OrthGaugeRecursion()
        context = {'to_sites': (4,)}

        step = getattr(recursion, f'advance_{direction}')(
            opening, None, context)
        outgoing = getattr(opening, f'{direction}_gauge')
        outgoing_map = tk.decompositions.GaugeMap(
            outgoing, outgoing_orientation)
        fixed_map = tk.decompositions.GaugeMap(
            step.gauge, fixed_orientation)

        assert torch.allclose(
            outgoing_map.matrix.T @ fixed_map.matrix,
            torch.eye(4, dtype=torch.float64),
            rtol=2e-10,
            atol=2e-10)
        assert step.records[0].site == 4
        assert not step.records[0].projective


def _opening(tt_core, dtype=torch.float64, cyclic_rank=1):
    """Builds a compatible one-site opening around ``tt_core``."""
    generator = torch.Generator().manual_seed(160)
    left = torch.randn(
        cyclic_rank, tt_core.shape[0], 2, dtype=dtype, generator=generator)
    core = torch.randn(2, tt_core.shape[1], 2,
                       dtype=dtype, generator=generator)
    right = torch.randn(
        2, tt_core.shape[2], cyclic_rank, dtype=dtype, generator=generator)
    return tk.decompositions.LoopOpening(
        left_gauge=left,
        cores=(core,),
        right_gauge=right,
        rank=(2, 2, cyclic_rank))


def _recursion(**kwargs):
    return tk.decompositions.PinvGaugeRecursion(**kwargs)


def _context(tt_core):
    return {
        'from_sites': (0,),
        'to_sites': (1,),
        'provider': SimpleNamespace(cores=(tt_core,)),
    }


def _vidal(tt, mode='inverse', powers=None):
    """Prepares native Vidal metadata without changing the source tensor."""
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


class TestPinvGaugeRecursionLocal:  # MARK: TestPinvGaugeRecursionLocal

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_right_square_transport_matches_direct_coordinate_solve(
            self, dtype):
        generator = torch.Generator().manual_seed(161)
        matrix = torch.randn(4, 4, dtype=dtype, generator=generator)
        matrix = matrix + 4 * torch.eye(4, dtype=dtype)
        tt_core = matrix.reshape(2, 2, 4)
        opening = _opening(tt_core, dtype=dtype)
        basis = torch.einsum(
            'gma,apb->mpgb', opening.left_gauge, opening.cores[0])
        expected = torch.linalg.solve(matrix, basis.reshape(4, -1))

        step = _recursion().advance_right(
            opening, None, _context(tt_core))
        actual = step.gauge.permute(1, 0, 2).reshape(4, -1)

        assert torch.allclose(actual, expected, rtol=2e-10, atol=2e-10)
        assert torch.allclose(
            matrix @ actual, basis.reshape(4, -1),
            rtol=2e-10, atol=2e-10)
        assert step.records[0].inverse_method == 'pinv'
        assert not step.records[0].projective

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_left_square_transport_is_the_exact_mirror(
            self, dtype):
        generator = torch.Generator().manual_seed(162)
        matrix = torch.randn(4, 4, dtype=dtype, generator=generator)
        matrix = matrix + 4 * torch.eye(4, dtype=dtype)
        tt_core = matrix.reshape(2, 2, 4).permute(2, 1, 0)
        unfolding = tt_core.permute(1, 2, 0).reshape(4, 4)
        opening = _opening(tt_core, dtype=dtype)
        basis = torch.einsum(
            'apb,bng->pnag', opening.cores[0], opening.right_gauge)
        expected = torch.linalg.solve(unfolding, basis.reshape(4, -1))

        step = _recursion().advance_left(
            opening, None, _context(tt_core))
        actual = step.gauge.permute(1, 0, 2).reshape(4, -1)

        assert torch.allclose(actual, expected, rtol=2e-10, atol=2e-10)
        assert torch.allclose(
            unfolding @ actual, basis.reshape(4, -1),
            rtol=2e-10, atol=2e-10)
        assert not step.records[0].projective

    @pytest.mark.parametrize('shape', [(2, 2, 3), (1, 2, 3)])
    def test_rectangular_transport_matches_pseudoinverse_projection(
            self, shape):
        tt_core = torch.randn(
            *shape,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(163))
        opening = _opening(tt_core)
        matrix = tt_core.reshape(-1, tt_core.shape[-1])
        basis = torch.einsum(
            'gma,apb->mpgb', opening.left_gauge, opening.cores[0])
        basis = basis.reshape(matrix.shape[0], -1)

        step = _recursion().advance_right(
            opening, None, _context(tt_core))
        actual = step.gauge.permute(1, 0, 2).reshape(
            matrix.shape[1], -1)

        assert torch.allclose(
            actual, torch.linalg.pinv(matrix) @ basis,
            rtol=2e-10, atol=2e-10)
        assert step.records[0].projective
        assert step.diagnostics['projection_error'] == pytest.approx(
            (matrix @ actual - basis).norm().div(basis.norm()).item())

    def test_rank_tolerance_reports_ill_conditioned_core(self):
        matrix = torch.diag(torch.tensor([1.0, 1e-4, 1e-9, 1e-12]))
        tt_core = matrix.reshape(2, 2, 4)
        opening = _opening(tt_core, dtype=matrix.dtype)
        step = _recursion(rtol=1e-8).advance_right(
                opening, None, _context(tt_core))

        assert step.records[0].numerical_rank == 2
        assert step.records[0].projective
        assert step.records[0].condition_number == float('inf')



class TestPinvGaugeRecursionIntegration:

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('direction', ['left', 'right'])
    def test_block_transport_pseudoinverts_the_contracted_dense_block(
            self, dtype, direction):
        from tensorkrowch.decompositions.ring.tt2tr import _TTCoreProvider

        generator = torch.Generator().manual_seed(217)

        def random(*shape):
            return torch.randn(*shape, dtype=dtype, generator=generator)

        tt = tk.decompositions.TTDecomposition([
            random(3, 2), *(random(2, 3, 2) for _ in range(4)), random(2, 3)])
        provider = _TTCoreProvider(tt)
        source = provider.local_target((2, 3), {}).reshape(2, 9, 2)
        left, right = random(2, 2, 2), random(2, 2, 2)
        first, second = random(2, 3, 2), random(2, 3, 2)
        opening = tk.decompositions.LoopOpening(
            left, (first, second), right, (2,) * 4)
        block = torch.tensordot(first, second, dims=([-1], [0]))
        block = block.reshape(2, 9, 2)
        if direction == 'right':
            matrix = source.reshape(-1, 2)
            basis = torch.einsum('gma,apb->mpgb', left, block)
        else:
            matrix = source.permute(1, 2, 0).reshape(-1, 2)
            basis = torch.einsum('apb,bng->pnag', block, right)
        expected = torch.linalg.pinv(matrix) @ basis.reshape(
            matrix.shape[0], -1)
        step = getattr(_recursion(), f'advance_{direction}')(
            opening, None, {'provider': provider, 'from_sites': (2, 3),
                            'to_sites': (4 if direction == 'right' else 1,)})
        actual = step.gauge.permute(1, 0, 2).reshape(expected.shape)
        torch.testing.assert_close(actual, expected)

    def test_complete_rank_one_chain_preserves_tensor_and_fidelity(self):
        generator = torch.Generator().manual_seed(165)
        tt = tk.decompositions.TTDecomposition([
            torch.randn(2, 1, dtype=torch.float64, generator=generator),
            torch.randn(1, 3, 1, dtype=torch.float64, generator=generator),
            torch.randn(1, 2, 1, dtype=torch.float64, generator=generator),
            torch.randn(1, 4, 1, dtype=torch.float64, generator=generator),
            torch.randn(1, 2, dtype=torch.float64, generator=generator),
        ])

        result = tk.decompositions.TT2TR(tt, out_device=None).fit(
            rank=1, recursion='pinv')

        assert result.metadata['recursion'] == 'pinv'
        assert torch.allclose(
            result.contract_dense(), tt.contract_dense(),
            rtol=2e-9, atol=2e-9)
        assert result.metrics.fidelities[0].fidelity == pytest.approx(
            1, rel=2e-10, abs=2e-10)
        assert len(result.metrics.gauges) == 2

    def test_nontrivial_chain_reports_fidelity_against_dense_oracle(self):
        dense = torch.randn(
            2, 2, 2, 2, 2,
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(166))
        tt = tk.decompositions.TTSVD(
            dense, out_device=None).fit()
        opener = tk.decompositions.ALSLoopOpener({
            'gauge': 'none',
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=3),
        })

        result = tk.decompositions.TT2TR(tt, out_device=None).fit(
            rank=[2, 2, 2, 2, 1], loop_opener=opener, recursion='pinv')
        approximation = result.contract_dense()
        overlap = torch.vdot(
            dense.reshape(-1), approximation.reshape(-1))
        overlap = overlap / (dense.norm() * approximation.norm())
        relative = (dense - approximation).norm() / dense.norm()

        assert result.rank == [2, 2, 2, 2, 1]
        assert result.metrics.fidelities[0].normalized_overlap == \
            pytest.approx(overlap.item(), rel=2e-10, abs=2e-10)
        assert result.metrics.errors[0].relative == pytest.approx(
            relative.item(), rel=2e-10, abs=2e-10)


class TestAdjointGaugeRecursion:  # MARK: TestAdjointGaugeRecursion

    @pytest.mark.parametrize('direction', ['left', 'right'])
    def test_pseudoinverse_options_do_not_change_adjoint_transport(
            self, direction):
        from tensorkrowch.decompositions.ring.tt2tr import _VidalCoreProvider

        tt = _vidal(tk.decompositions.TTSVD(torch.randn(
            [3] * 5, dtype=torch.complex128,
            generator=torch.Generator().manual_seed(707))).fit())
        provider = _VidalCoreProvider(tt)
        opening = _opening(provider.cores[2], torch.complex128)
        context = {'provider': provider, 'from_sites': (2,),
                   'to_sites': (3 if direction == 'right' else 1,)}
        reference = tk.decompositions.AdjointGaugeRecursion()
        configured = tk.decompositions.AdjointGaugeRecursion(
            hermitian=True, atol=1., rtol=1.)
        expected = getattr(reference, f'advance_{direction}')(
            opening, None, context)
        actual = getattr(configured, f'advance_{direction}')(
            opening, None, context)
        assert actual.records[0].shape[0] != actual.records[0].shape[1]
        assert actual.records[0].inverse_method == 'adjoint'
        torch.testing.assert_close(actual.gauge, expected.gauge)
        assert (actual.records[0].numerical_rank ==
                expected.records[0].numerical_rank)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_inverse_vidal_centers_have_orthonormal_external_bases(
            self, dtype):
        from tensorkrowch.decompositions.ring.tt2tr import _VidalCoreProvider

        generator = torch.Generator().manual_seed(178)
        tt = _vidal(tk.decompositions.TTSVD(torch.randn(
            2, 3, 2, 3, 2, dtype=dtype, generator=generator)).fit())
        provider = _VidalCoreProvider(tt)
        dense = tt.contract_dense()
        reconstructed = provider.local_target(range(5), {})
        reconstructed = reconstructed.squeeze(0).squeeze(-1)
        torch.testing.assert_close(reconstructed, dense)
        for site in range(5):
            left = dense.new_ones(1, 1)
            for index in range(site):
                core = provider.cores[index] / provider.spectra[index]
                left = torch.einsum('xa,apb->xpb', left, core).reshape(
                    -1, core.shape[-1])
            right = dense.new_ones(1, 1)
            for index in reversed(range(site + 1, 5)):
                core = (provider.cores[index]
                        / provider.spectra[index - 1][:, None, None])
                right = torch.einsum('apb,bx->apx', core, right).reshape(
                    core.shape[0], -1)
            torch.testing.assert_close(
                left.mH @ left, torch.eye(left.shape[-1], dtype=dtype))
            torch.testing.assert_close(
                right @ right.mH, torch.eye(right.shape[0], dtype=dtype))
            expanded = torch.einsum(
                'xa,apb,by->xpy', left, provider.cores[site], right)
            torch.testing.assert_close(expanded.reshape_as(dense), dense)

    @pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse'])
    @pytest.mark.parametrize('direction', ['left', 'right'])
    def test_native_vidal_transport_uses_adjoint_without_pseudoinverse(
            self, mode, direction, monkeypatch):
        from tensorkrowch.decompositions.ring.tt2tr import _VidalCoreProvider

        tt = _vidal(tk.decompositions.TTSVD(torch.randn(
            [3] * 5, dtype=torch.complex128,
            generator=torch.Generator().manual_seed(179))).fit(), mode)
        provider = _VidalCoreProvider(tt)
        opening = _opening(provider.cores[2], torch.complex128)
        recursion = tk.decompositions.AdjointGaugeRecursion()

        def no_pinv(*args, **kwargs):
            raise AssertionError('Native Vidal transport should use adjoints')

        monkeypatch.setattr(torch.linalg, 'pinv', no_pinv)
        step = getattr(recursion, f'advance_{direction}')(
            opening, None, {'provider': provider, 'from_sites': (2,),
                            'to_sites': (3 if direction == 'right' else 1,)})
        assert step.records[0].inverse_method == 'adjoint'
        assert torch.isfinite(step.gauge).all()

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('direction', ['left', 'right'])
    @pytest.mark.parametrize('mode', [
        'explicit', 'implicit', 'inverse', 'mixed'])
    @pytest.mark.parametrize('sites', [(2,), (2, 3)])
    def test_vidal_transport_matches_full_subchain_pseudoinverse(
            self, dtype, direction, mode, sites):
        from tensorkrowch.decompositions.ring.tt2tr import _VidalCoreProvider

        generator = torch.Generator().manual_seed(981)

        def random(*shape):
            return torch.randn(*shape, dtype=dtype, generator=generator)

        tt = tk.formats.TT([
            random(4, 3), *(random(3, 4, 3) for _ in range(4)), random(3, 4)])
        powers = [(0., 1.), (.7, -.2), (1., 1.), (.25, .6), (1., 0.)]
        tt = _vidal(tt, mode if mode != 'mixed' else 'explicit',
                    powers if mode == 'mixed' else None)
        provider = _VidalCoreProvider(tt)
        target = provider.local_target(sites, {})
        left = random(2, target.shape[0], 2)
        right = random(2, target.shape[-1], 2)
        cores = [random(2, 4, 2) for _ in sites]
        opening = tk.decompositions.LoopOpening(
            left, cores, right, (2,) * (len(sites) + 2))
        block = cores[0]
        for core in cores[1:]:
            block = torch.tensordot(block, core, dims=([-1], [0]))
        block = block.reshape(2, -1, 2)
        source = target.reshape(target.shape[0], -1, target.shape[-1])
        if direction == 'right':
            physical = target.new_ones(1, 1)
            for site in range(sites[0]):
                physical = torch.einsum(
                    'xa,apb->xpb', physical, provider.cores[site]).reshape(
                        -1, provider.cores[site].shape[-1])
                physical = physical * provider.bond_factors[site]
            full = torch.einsum('xa,apb->xpb', physical, source)
            full = full.reshape(-1, source.shape[-1])
            full = full * provider.bond_factors[sites[-1]]
            basis = torch.einsum('xa,gab,bic->xigc', physical, left, block)
        else:
            physical = target.new_ones(1, 1)
            for site in reversed(range(sites[-1] + 1, len(provider.cores))):
                physical = torch.einsum(
                    'apb,bx->apx', provider.cores[site], physical).reshape(
                        provider.cores[site].shape[0], -1)
                physical = physical * provider.bond_factors[site - 1][:, None]
            full = torch.einsum('apb,bx->apx', source, physical)
            full = full.reshape(source.shape[0], -1).T
            full = full * provider.bond_factors[sites[0] - 1]
            basis = torch.einsum('apb,bng,nx->pxga', block, right, physical)
        expected = torch.linalg.pinv(full) @ basis.reshape(full.shape[0], -1)
        recursion = tk.decompositions.AdjointGaugeRecursion()
        step = getattr(recursion, f'advance_{direction}')(
            opening, None, {'provider': provider, 'from_sites': sites,
                            'to_sites': (sites[-1] + 1 if direction == 'right'
                                         else sites[0] - 1,)})
        actual = step.gauge.permute(
            (1, 0, 2) if direction == 'right' else (1, 2, 0))
        actual = actual.reshape(expected.shape)
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)


class TestExpandedGaugeRecursion:

    @pytest.mark.parametrize('recursion_type', [
        tk.decompositions.OrthGaugeRecursion,
        tk.decompositions.PinvGaugeRecursion,
        tk.decompositions.AdjointGaugeRecursion,
    ])
    @pytest.mark.parametrize('direction', ['left', 'right'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_enrichment_preserves_reference_and_resolves_residual_support(
            self, recursion_type, direction, dtype):
        from tensorkrowch.decompositions.ring.gauges import (
            _ExpandedGaugeRecursion,
        )
        from tensorkrowch.decompositions.ring.tt2tr import (
            _CanonicalTTCoreProvider, _TTCoreProvider,
        )

        generator = torch.Generator().manual_seed(904)
        tt = tk.formats.TT([
            torch.randn(3, 2, dtype=dtype, generator=generator),
            *(torch.randn(2, 3, 2, dtype=dtype, generator=generator)
              for _ in range(3)),
            torch.randn(2, 3, dtype=dtype, generator=generator),
        ])
        if recursion_type is tk.decompositions.AdjointGaugeRecursion:
            provider = _CanonicalTTCoreProvider(tt, 2)
        else:
            provider = _TTCoreProvider(tt)
        source = provider.cores[2]
        opening = _opening(source, dtype, cyclic_rank=2)
        destination = 3 if direction == 'right' else 1
        context = {'provider': provider, 'from_sites': (2,),
                   'to_sites': (destination,)}
        before = provider.cores[destination].clone()
        recursion = _ExpandedGaugeRecursion(recursion_type(), None, 1e-12)
        step = getattr(recursion, f'advance_{direction}')(
            opening, None, context)
        added = step.diagnostics['added']
        after = provider.cores[destination]
        axis = 0 if direction == 'right' else 2

        assert added > 0
        assert after.shape[axis] == before.shape[axis] + added
        assert torch.count_nonzero(after.narrow(
            axis, before.shape[axis], added)) == 0
        if recursion_type is not tk.decompositions.AdjointGaugeRecursion:
            torch.testing.assert_close(
                after.narrow(axis, 0, before.shape[axis]), before)
        assert step.diagnostics['target_preservation_error'] < 1e-12
        assert step.diagnostics['residual_after'] < 1e-12
        assert step.diagnostics['orthogonality_error'] < 1e-12
        assert step.gauge.shape[1] == after.shape[axis]
        assert torch.isfinite(step.gauge).all()

    @pytest.mark.parametrize('limit', [0, 1])
    @pytest.mark.parametrize('direction', ['left', 'right'])
    def test_expansion_limit_and_zero_control(self, limit, direction):
        from tensorkrowch.decompositions.ring.gauges import (
            _ExpandedGaugeRecursion,
        )
        from tensorkrowch.decompositions.ring.tt2tr import _TTCoreProvider

        generator = torch.Generator().manual_seed(905)
        tt = tk.formats.TT([
            torch.randn(3, 2, generator=generator, dtype=torch.float64),
            *(torch.randn(2, 3, 2, generator=generator, dtype=torch.float64)
              for _ in range(3)),
            torch.randn(2, 3, generator=generator, dtype=torch.float64),
        ])
        provider = _TTCoreProvider(tt)
        opening = _opening(provider.cores[2], cyclic_rank=2)
        destination = 3 if direction == 'right' else 1
        context = {'provider': provider, 'from_sites': (2,),
                   'to_sites': (destination,)}
        base = tk.decompositions.PinvGaugeRecursion()
        expected = getattr(base, f'advance_{direction}')(
            opening, None, context)
        step = getattr(_ExpandedGaugeRecursion(base, limit, 1e-12),
                       f'advance_{direction}')(opening, None, context)
        assert step.diagnostics['added'] == limit
        torch.testing.assert_close(
            step.gauge[:, :expected.gauge.shape[1]], expected.gauge)
        if limit == 0:
            torch.testing.assert_close(step.gauge, expected.gauge)

    @pytest.mark.parametrize('options, error', [
        ((-1, 1e-12), ValueError),
        ((True, 1e-12), TypeError),
        ((None, -1), ValueError),
        ((None, float('inf')), ValueError),
        ((None, None), TypeError),
    ])
    def test_invalid_expansion_options(self, options, error):
        from tensorkrowch.decompositions.ring.gauges import (
            _ExpandedGaugeRecursion,
        )
        with pytest.raises(error):
            _ExpandedGaugeRecursion(
                tk.decompositions.PinvGaugeRecursion(), *options)


class TestGaugeRecursionStep:

    def test_minimal_result_copies_containers_and_retains_gauge(self):
        core = torch.ones(1, 1, 1)
        diagnostics = {'source': 2}
        step = tk.decompositions.GaugeRecursionStep(
            core, records=[], diagnostics=diagnostics)
        assert step.gauge is core
        assert step.records == ()
        diagnostics['source'] = 3
        assert step.diagnostics['source'] == 2

    @pytest.mark.parametrize('gauge, kwargs, error', [
        (None, {}, TypeError),
        (torch.ones(1, 1), {}, ValueError),
        (torch.ones(1, 1, 1), {'records': [None]}, TypeError),
        (torch.ones(1, 1, 1), {'diagnostics': []}, TypeError),
    ])
    def test_invalid_result(self, gauge, kwargs, error):
        with pytest.raises(error):
            tk.decompositions.GaugeRecursionStep(gauge, **kwargs)
