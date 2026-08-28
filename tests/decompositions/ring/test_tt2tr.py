"""Tests for structured conversion from TT to TR."""

import pytest

import torch
import tensorkrowch as tk


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


class TestTT2TR:  # MARK: TestTT2TR

    @pytest.mark.parametrize('center', [1, 2, 3])
    @pytest.mark.parametrize(
        ('dtype', 'phase'),
        [(torch.float64, 1.0), (torch.complex128, 1.0 + 0.5j)])
    def test_rank_one_conversion_is_exact_for_all_internal_centers(
            self, center, dtype, phase):
        tt = _rank_one_tt(dtype=dtype, phase=phase)
        result = tk.decompositions.TT2TR(
            tt, output_device=None).fit(rank=1, center=center)

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
            dense, output_device=None).fit()
        result = tk.decompositions.TT2TR(
            tt, output_device=None).fit(
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
            dense, output_device=None).fit()
        result = tk.decompositions.TT2TR(
            tt, output_device=None).fit(
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
                tt, output_device=None).fit(
                    rank=2,
                    tr_rank=2,
                    loop_opener=_short_als())

        result = tk.decompositions.TT2TR(
            tt, output_device=None).fit(
                rank=2,
                tr_rank=2,
                loop_opener=_short_als(),
                allow_projective_gauges=True)
        assert result.rank == [2, 2, 2, 2, 2]
        assert any(record.projective for record in result.metrics.gauges)

    def test_functional_api_and_mps_adapter(self):
        tt = _rank_one_tt()
        model = tk.models.MPS(tensors=tt.cores, parameterized=False)
        cores, info = tk.decompositions.tt2tr(
            model,
            rank=1,
            output_device=None,
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
                tt, output_device=None).fit(
                    rank=1,
                    loop_opener='blostr+als')

        assert result.rank == [1, 1, 1, 1, 1]
        assert torch.allclose(
            result.contract_dense(),
            tt.contract_dense().to(result.cores[0].dtype),
            rtol=2e-9,
            atol=2e-9)

    def test_history_observer_receives_structured_steps_and_summary(self):
        observer = tk.decompositions.HistoryObserver()
        result = tk.decompositions.TT2TR(
            _rank_one_tt(), output_device=None).fit(
                rank=1, verbose=0, observer=observer)

        assert observer.metrics is result.metrics
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
                _rank_one_tt(), output_device=None).fit(rank=1, center=0)
        with pytest.raises(ValueError, match='positive'):
            tk.decompositions.TT2TR(
                _rank_one_tt(), output_device=None).fit(rank=0)
