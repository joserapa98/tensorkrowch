"""Tests for TT-core recursive virtual-basis transport."""

from types import SimpleNamespace

import pytest

import torch
import tensorkrowch as tk


def _opening(tt_core, dtype=torch.float64):
    """Builds a compatible one-site opening around ``tt_core``."""
    generator = torch.Generator().manual_seed(160)
    left = torch.randn(
        1, tt_core.shape[0], 2, dtype=dtype, generator=generator)
    core = torch.randn(2, tt_core.shape[1], 2,
                       dtype=dtype, generator=generator)
    right = torch.randn(
        2, tt_core.shape[2], 1, dtype=dtype, generator=generator)
    return tk.decompositions.LoopOpening(
        left_gauge=left,
        cores=(core,),
        right_gauge=right,
        rank=(2, 2, 1))


def _recursion(policy='auto', allow_projective=False, rank_rtol=None):
    with pytest.warns(tk.decompositions.ExperimentalWarning):
        return tk.decompositions.TTCoreGaugeRecursion(
            inverse_policy=policy,
            allow_projective=allow_projective,
            rank_rtol=rank_rtol)


def _context(tt_core):
    return {
        'from_sites': (0,),
        'to_sites': (1,),
        'provider': SimpleNamespace(cores=(tt_core,)),
    }


class TestTTCoreGaugeRecursionLocal:  # MARK: TestTTCoreGaugeRecursionLocal

    @pytest.mark.parametrize('policy', ['solve', 'inverse', 'auto'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_right_square_transport_matches_direct_coordinate_solve(
            self, policy, dtype):
        generator = torch.Generator().manual_seed(161)
        matrix = torch.randn(4, 4, dtype=dtype, generator=generator)
        matrix = matrix + 4 * torch.eye(4, dtype=dtype)
        tt_core = matrix.reshape(2, 2, 4)
        opening = _opening(tt_core, dtype=dtype)
        basis = torch.einsum(
            'gma,apb->mpgb', opening.left_gauge, opening.cores[0])
        expected = torch.linalg.solve(matrix, basis.reshape(4, -1))

        step = _recursion(policy).advance_right(
            opening, None, _context(tt_core))
        actual = step.gauge.permute(1, 0, 2).reshape(4, -1)

        assert torch.allclose(actual, expected, rtol=2e-10, atol=2e-10)
        assert torch.allclose(
            matrix @ actual, basis.reshape(4, -1),
            rtol=2e-10, atol=2e-10)
        assert step.records[0].inverse_method == (
            'solve' if policy == 'auto' else policy)
        assert not step.records[0].projective

    @pytest.mark.parametrize('policy', ['solve', 'inverse', 'auto'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_left_square_transport_is_the_exact_mirror(
            self, policy, dtype):
        generator = torch.Generator().manual_seed(162)
        matrix = torch.randn(4, 4, dtype=dtype, generator=generator)
        matrix = matrix + 4 * torch.eye(4, dtype=dtype)
        tt_core = matrix.reshape(2, 2, 4).permute(2, 1, 0)
        unfolding = tt_core.permute(1, 2, 0).reshape(4, 4)
        opening = _opening(tt_core, dtype=dtype)
        basis = torch.einsum(
            'apb,bng->pnag', opening.cores[0], opening.right_gauge)
        expected = torch.linalg.solve(unfolding, basis.reshape(4, -1))

        step = _recursion(policy).advance_left(
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

        step = _recursion('pinv', allow_projective=True).advance_right(
            opening, None, _context(tt_core))
        actual = step.gauge.permute(1, 0, 2).reshape(
            matrix.shape[1], -1)

        assert torch.allclose(
            actual, torch.linalg.pinv(matrix) @ basis,
            rtol=2e-10, atol=2e-10)
        assert step.records[0].projective
        assert step.diagnostics['projection_error'] == pytest.approx(
            (matrix @ actual - basis).norm().div(basis.norm()).item())

    def test_projective_transport_requires_explicit_opt_in(self):
        tt_core = torch.randn(
            1, 2, 3,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(164))
        opening = _opening(tt_core)

        with pytest.raises(ValueError, match='allow_projective=True'):
            _recursion('pinv').advance_right(
                opening, None, _context(tt_core))

    def test_rank_tolerance_reports_ill_conditioned_core(self):
        matrix = torch.diag(torch.tensor([1.0, 1e-4, 1e-9, 1e-12]))
        tt_core = matrix.reshape(2, 2, 4)
        opening = _opening(tt_core, dtype=matrix.dtype)
        step = _recursion(
            'pinv', allow_projective=True, rank_rtol=1e-8).advance_right(
                opening, None, _context(tt_core))

        assert step.records[0].numerical_rank == 2
        assert step.records[0].projective
        assert step.records[0].condition_number == float('inf')

    def test_boundary_preparation_returns_outgoing_orientation(self):
        matrix = torch.eye(4, dtype=torch.float64)
        tt_core = matrix.reshape(2, 2, 4)
        opening = _opening(tt_core)
        recursion = _recursion('pinv')
        context = _context(tt_core)
        step = recursion.advance_right(opening, None, context)
        boundary = recursion.prepare_boundary(
            opening, None, context, 'right')

        incoming = tk.decompositions.GaugeMap(step.gauge, 'left')
        outgoing = tk.decompositions.GaugeMap(boundary.gauge, 'right')
        assert torch.allclose(
            incoming.matrix.T @ outgoing.matrix,
            torch.eye(2, dtype=torch.float64),
            rtol=2e-10,
            atol=2e-10)
        assert boundary.diagnostics['boundary']


class TestTTCoreGaugeRecursionIntegration:  # MARK: Integration

    def test_complete_rank_one_chain_preserves_tensor_and_fidelity(self):
        generator = torch.Generator().manual_seed(165)
        tt = tk.decompositions.TTDecomposition([
            torch.randn(2, 1, dtype=torch.float64, generator=generator),
            torch.randn(1, 3, 1, dtype=torch.float64, generator=generator),
            torch.randn(1, 2, 1, dtype=torch.float64, generator=generator),
            torch.randn(1, 4, 1, dtype=torch.float64, generator=generator),
            torch.randn(1, 2, dtype=torch.float64, generator=generator),
        ])

        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TT2TR(
                tt, output_device=None).fit(
                    rank=1, gauge_recursion='tt_core')

        assert result.metadata['gauge_recursion'] == 'tt_core'
        assert torch.allclose(
            result.contract_dense(), tt.contract_dense(),
            rtol=2e-9, atol=2e-9)
        assert result.metrics.fidelities[0].fidelity == pytest.approx(
            1, rel=2e-10, abs=2e-10)
        assert len(result.metrics.gauges) == 6

    def test_nontrivial_chain_reports_fidelity_against_dense_oracle(self):
        dense = torch.randn(
            2, 2, 2, 2, 2,
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(166))
        tt = tk.decompositions.TTSVD(
            dense, output_device=None).fit()
        opener = tk.decompositions.ALSLoopOpener({
            'gauge': 'none',
            'normalize': False,
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=3),
        })

        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TT2TR(
                tt, output_device=None).fit(
                    rank=2,
                    tr_rank=1,
                    loop_opener=opener,
                    gauge_recursion='tt_core',
                    allow_projective_gauges=True)
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
        assert any(record.projective for record in result.metrics.gauges)
