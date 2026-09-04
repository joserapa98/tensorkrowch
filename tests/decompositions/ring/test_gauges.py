"""Tests for oriented gauge maps and cancellation diagnostics."""

import math

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
        dual = gauge.inverse_or_pinv('pinv')
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
        record = dual.require_cancellable()
        assert isinstance(record, tk.decompositions.GaugeRecord)
        assert record.numerical_rank == 4
        assert record.cancellable
        assert not record.projective


class TestGaugeMapInversePolicies:  # MARK: TestGaugeMapInversePolicies

    @pytest.mark.parametrize('policy', ['solve', 'inverse', 'pinv', 'auto'])
    def test_square_policies_produce_the_same_directional_dual(self, policy):
        matrix = torch.randn(
            4, 4,
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(141))
        core = matrix.reshape(4, 2, 2).permute(1, 0, 2)
        gauge = tk.decompositions.GaugeMap(core, 'left')
        dual = gauge.inverse_or_pinv(policy)

        assert torch.allclose(
            gauge.matrix.T @ dual.matrix,
            torch.eye(4, dtype=matrix.dtype),
            rtol=2e-10,
            atol=2e-10)
        expected_method = 'solve' if policy == 'auto' else policy
        assert dual.inverse_method == expected_method

    def test_rectangular_solve_and_inverse_are_rejected(self):
        gauge = tk.decompositions.GaugeMap(
            _full_rank_core('left'), 'left')

        with pytest.raises(ValueError, match='square'):
            gauge.inverse_or_pinv('solve')
        with pytest.raises(ValueError, match='square'):
            gauge.inverse_or_pinv('inverse')
        assert gauge.inverse_or_pinv('auto').inverse_method == 'pinv'

    def test_auto_falls_back_to_pinv_for_a_singular_square_gauge(self):
        matrix = torch.diag(torch.tensor([1.0, 1.0, 0.0, 0.0]))
        core = matrix.reshape(4, 2, 2).permute(1, 0, 2)
        dual = tk.decompositions.GaugeMap(
            core, 'left').inverse_or_pinv('auto')

        assert dual.inverse_method == 'pinv'
        assert dual.diagnostics().projective


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
            core, orientation, site=3).inverse_or_pinv('pinv')
        record = gauge.diagnostics()

        assert record.numerical_rank == 2
        assert record.cancellable_rank == 4
        assert record.projective
        assert record.cancellation_error == pytest.approx(math.sqrt(0.5))
        assert not record.cancellable
        with pytest.raises(ValueError, match='at site 3'):
            gauge.require_cancellable()
        assert gauge.require_cancellable(
            allow_projective=True) == record

    def test_rank_tolerance_controls_numerical_rank_and_pseudoinverse(self):
        matrix = torch.diag(torch.tensor([1.0, 1e-6, 1e-10, 0.0]))
        core = matrix.reshape(4, 2, 2).permute(1, 0, 2)
        dual = tk.decompositions.GaugeMap(
            core, 'left').inverse_or_pinv('pinv', rank_rtol=1e-8)
        record = dual.diagnostics()

        assert record.numerical_rank == 2
        assert record.rank_tolerance == pytest.approx(1e-8)
        assert math.isinf(record.condition_number)

    def test_structured_event_contains_the_record_diagnostics(self):
        dual = tk.decompositions.GaugeMap(
            _full_rank_core('right'),
            'right',
            site=1).inverse_or_pinv('pinv')
        event = dual.as_event('TT to TR')

        assert isinstance(event, DecompositionEvent)
        assert event.name == 'gauge'
        assert event.phase == 'TT to TR'
        assert event.site == 1
        assert event.values['rank'] == '4/4'
        assert not event.values['projective']


class TestPseudoinverseGaugeRecursion:  # MARK: TestPseudoinverseGaugeRecursion

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
        recursion = tk.decompositions.PseudoinverseGaugeRecursion()
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
