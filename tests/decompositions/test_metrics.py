"""Tests for structured decomposition metrics."""

import math
from dataclasses import replace

import pytest
import torch

import tensorkrowch as tk


class TestDecompositionMetrics:  # MARK: TestDecompositionMetrics

    def test_error_record_stores_batch_values_on_cpu(self):
        absolute = torch.tensor([1.0, 2.0], requires_grad=True)
        record = tk.decompositions.ErrorRecord(
            kind='reconstruction',
            absolute=absolute,
            relative=torch.tensor([0.5, 0.25]),
            size=2,
            denominator=torch.tensor([2.0, 8.0]))

        assert record.absolute.device.type == 'cpu'
        assert not record.absolute.requires_grad
        assert torch.allclose(record.absolute, torch.tensor([1.0, 2.0]))
        assert torch.allclose(record.relative, torch.tensor([0.5, 0.25]))

    def test_truncation_record(self):
        record = tk.decompositions.TruncationRecord(
            site=1,
            full_rank=5,
            selected_rank=3,
            local_abs_error=torch.tensor([1.0, 2.0], requires_grad=True),
            local_rel_error=torch.tensor([0.2, 0.4]),
            local_norm=torch.tensor([5.0, 5.0]),
            discarded_sq_norm=torch.tensor([1.0, 4.0]),
            global_rel_contribution=torch.tensor([0.1, 0.2]),
            svd_method='qr_svd')

        assert record.selected_rank == 3
        assert record.local_abs_error.device.type == 'cpu'
        assert not record.local_abs_error.requires_grad

    def test_truncation_record_optional_phase(self):
        record = tk.decompositions.TruncationRecord(
            site=0,
            full_rank=2,
            selected_rank=1,
            local_abs_error=torch.tensor(1.0),
            phase='initial_bipartition')

        assert record.phase == 'initial_bipartition'
        with pytest.raises(TypeError, match='`phase` should be str type'):
            tk.decompositions.TruncationRecord(
                site=0,
                full_rank=2,
                selected_rank=1,
                local_abs_error=torch.tensor(1.0),
                phase=1)

    def test_truncation_record_from_svd_info(self):
        tensor = torch.stack([
            torch.diag(torch.tensor([4.0, 3.0])),
            torch.diag(torch.tensor([2.0, 1.0])),
        ])
        *_, info = tk.utils.truncated_svd(
            tensor, rank=1, return_info=True)

        record = tk.decompositions.TruncationRecord.from_svd_info(
            info,
            site=2,
            log_scale=torch.tensor([0.0, math.log(2.0)]),
            global_norm=torch.tensor([5.0, 4.0]))

        # Discarded energies are 3**2 and (2 * 1)**2 after rescaling.
        assert torch.allclose(record.discarded_sq_norm,
                              torch.tensor([9.0, 4.0]))
        assert torch.allclose(record.local_abs_error,
                              torch.tensor([3.0, 2.0]))
        assert torch.allclose(record.local_norm,
                              torch.tensor([5.0, math.sqrt(20.0)]))
        assert torch.allclose(record.local_rel_error,
                              torch.tensor([3.0 / 5.0,
                                            1.0 / math.sqrt(5.0)]))
        assert torch.allclose(record.global_rel_contribution,
                              torch.tensor([0.6, 0.5]))

    def test_truncation_record_zero_norm_policy(self):
        *_, info = tk.utils.truncated_svd(
            torch.zeros(3, 3), rank=1, return_info=True)

        record = tk.decompositions.TruncationRecord.from_svd_info(
            info, site=0, global_norm=torch.tensor(0.0))

        assert record.local_abs_error == 0.0
        assert record.local_rel_error == 0.0
        assert record.global_rel_contribution == 0.0

    def test_timing_record_children(self):
        child = tk.decompositions.TimingRecord(name='svd', elapsed=0.1)
        parent = tk.decompositions.TimingRecord(
            name='fit', elapsed=0.2, children=[child])

        assert parent.children == (child,)

    def test_evaluation_stats_delta_and_validation(self):
        previous = tk.decompositions.EvaluationStats(
            requested_points=5,
            unique_points=4,
            batches=2,
            cache_hits=1,
            source_calls=1)
        current = tk.decompositions.EvaluationStats(
            requested_points=12,
            unique_points=9,
            batches=5,
            cache_hits=3,
            source_calls=3)

        assert current.delta(previous) == tk.decompositions.EvaluationStats(
            requested_points=7,
            unique_points=5,
            batches=3,
            cache_hits=2,
            source_calls=2)
        with pytest.raises(ValueError, match='non-negative'):
            tk.decompositions.EvaluationStats(requested_points=-1)
        with pytest.raises(TypeError, match='int type'):
            tk.decompositions.EvaluationStats(source_calls=True)
        with pytest.raises(ValueError, match='should not exceed'):
            previous.delta(current)

    def test_fidelity_record_preserves_phase(self):
        overlap = torch.tensor(
            0.0 + 0.5j, dtype=torch.complex128, requires_grad=True)
        record = tk.decompositions.FidelityRecord(overlap)

        assert record.normalized_overlap.dtype == torch.complex128
        assert record.fidelity.dtype == torch.float64
        assert not record.normalized_overlap.requires_grad
        assert not record.fidelity.requires_grad
        assert record.normalized_overlap == 0.5j
        assert record.fidelity == 0.25

    def test_gauge_record_reports_cancellability(self):
        record = tk.decompositions.GaugeRecord(
            orientation='left',
            shape=(8, 4),
            numerical_rank=4,
            cancellable_rank=4,
            condition_number=2.0,
            cancellation_error=1e-12,
            projective=False,
            inverse_method='solve',
            tolerance=1e-8,
            rank_tolerance=1e-14,
            site=2)

        assert record.cancellable

    def test_metrics_as_info(self):
        error = tk.decompositions.ErrorRecord(
            kind='samples', absolute=1.0, relative=0.5)
        fidelity = tk.decompositions.FidelityRecord(1j)
        metrics = tk.decompositions.DecompositionMetrics(
            errors=[error],
            fidelities=[fidelity],
            evaluations=[tk.decompositions.EvaluationStats(
                requested_points=8,
                unique_points=6,
                batches=2,
                cache_hits=2,
                source_calls=1)],
            gauges=[tk.decompositions.GaugeRecord(
                orientation='right',
                shape=(4, 2),
                numerical_rank=2,
                cancellable_rank=2,
                condition_number=1.0,
                cancellation_error=0.0,
                projective=False,
                inverse_method='pinv',
                tolerance=1e-8,
                rank_tolerance=1e-14)],
            warnings=['diagnostic'])

        info = metrics.as_info()

        assert info['errors'][0]['kind'] == 'samples'
        assert info['fidelities'][0]['fidelity'] == 1.0
        assert info['gauges'][0]['orientation'] == 'right'
        assert info['evaluations'][0]['unique_points'] == 6
        assert info['warnings'] == ['diagnostic']

    @pytest.mark.parametrize(
        'record, error_type',
        [
            ({'kind': 'error', 'absolute': -1.0}, ValueError),
            ({'kind': 'error', 'absolute': [1.0]}, TypeError),
        ],
    )
    def test_error_record_validation(self, record, error_type):
        with pytest.raises(error_type):
            tk.decompositions.ErrorRecord(**record)

    def test_metrics_validation(self):
        with pytest.raises(TypeError):
            tk.decompositions.DecompositionMetrics(errors=['invalid'])


@pytest.fixture
def local_solve_record():
    return tk.decompositions.LocalSolveRecord(
        environment_shape=(8, 3), target_shape=(8, 2), driver='gelsd',
        abs_residual=torch.tensor(0.1, requires_grad=True),
        rel_residual=0.05, effective_l2_reg=0.02, column_scaling=True,
        site=(1, 2), sweep=3,
        sampling_exact=False, sample_generation=2)


def test_nested_records_keep_diagnostics_and_serialize(local_solve_record):
    input_fit = tk.decompositions.InputFitRecord(
        method='least_squares', axis=1, domain_size=8, in_dim=3,
        abs_residual=0.1, rel_residual=0.05, condition_number=2.,
        used_fibers=True, local_solve=local_solve_record)
    projection = tk.decompositions.RangeProjectionRecord(
        method='randomized', input_shape=(8, 6), axis=1, requested_dim=3,
        projection_dim=5, range_dim=3, oversampling=2, n_power_iter=1,
        error_absolute=0.1, error_relative=0.05, elapsed=torch.tensor(0.01))
    sweep = tk.decompositions.SweepRecord(
        sweep=3, abs_error=0.2, rel_error=0.1, rel_change=0.01,
        elapsed=0.5, sample_generation=2, stop_reason='tolerance')
    timing = tk.decompositions.TimingRecord(
        'fit', 0.5, children=[tk.decompositions.TimingRecord('svd', 0.1, site=2)])
    metrics = tk.decompositions.DecompositionMetrics(
        local_solves=(local_solve_record,), input_fits=(input_fit,),
        range_projections=(projection,), sweeps=(sweep,), timings=(timing,))
    info = metrics.as_info()
    assert not local_solve_record.abs_residual.requires_grad
    assert local_solve_record.abs_residual.device.type == 'cpu'
    assert info['input_fits'][0]['local_solve']['target_shape'] == [8, 2]
    assert info['range_projections'][0]['n_power_iter'] == 1
    assert info['sweeps'][0]['stop_reason'] == 'tolerance'
    assert info['timings'][0]['children'][0]['site'] == 2
    torch.testing.assert_close(info['local_solves'][0]['abs_residual'],
                               torch.tensor(0.1))
    info['warnings'].append('outside')
    assert metrics.warnings == []


@pytest.mark.parametrize('kwargs, error', [
    ({'environment_shape': (2,)}, ValueError),
    ({'target_shape': ()}, ValueError), ({'driver': 1}, TypeError),
    ({'abs_residual': torch.ones(2)}, ValueError),
    ({'rel_residual': float('nan')}, ValueError),
    ({'effective_l2_reg': float('inf')}, ValueError),
    ({'effective_l2_reg': -1.}, ValueError),
    ({'environment_shape': (True, 2)}, ValueError),
    ({'target_shape': (7, 2)}, ValueError),
    ({'target_shape': (8, 2, 3)}, ValueError),
    ({'column_scaling': 1}, TypeError),
    ({'site': (True,)}, ValueError),
    ({'sweep': -1}, ValueError),
    ({'sample_generation': -1}, ValueError),
    ({'site': -1}, ValueError), ({'sampling_exact': 1}, TypeError),
])
def test_invalid_local_solve_records(local_solve_record, kwargs, error):
    with pytest.raises(error):
        replace(local_solve_record, **kwargs)


@pytest.mark.parametrize('kwargs, error', [
    ({'method': ''}, TypeError), ({'axis': True}, TypeError),
    ({'in_dim': 0}, ValueError), ({'abs_residual': float('inf')}, ValueError),
    ({'used_fibers': 1}, TypeError), ({'local_solve': {}}, TypeError),
])
def test_invalid_input_fit_records(kwargs, error):
    record = tk.decompositions.InputFitRecord(
        method='least_squares', axis=0, domain_size=8, in_dim=3,
        abs_residual=0.1, rel_residual=0.05, condition_number=2.)
    with pytest.raises(error):
        replace(record, **kwargs)


@pytest.mark.parametrize('kwargs, error', [
    ({'method': 'unknown'}, ValueError), ({'input_shape': (0, 2)}, ValueError),
    ({'projection_dim': 0}, ValueError), ({'requested_dim': True}, TypeError),
    ({'elapsed': float('nan')}, ValueError),
])
def test_invalid_range_projection_records(kwargs, error):
    record = tk.decompositions.RangeProjectionRecord(
        method='identity', input_shape=(8, 3), axis=0, requested_dim=None,
        projection_dim=3, range_dim=3)
    with pytest.raises(error):
        replace(record, **kwargs)


@pytest.mark.parametrize('kwargs, error', [
    ({'sweep': True}, ValueError), ({'abs_error': torch.ones(2)}, ValueError),
    ({'rel_change': float('inf')}, ValueError), ({'elapsed': -1.}, ValueError),
    ({'stop_reason': 1}, TypeError),
])
def test_invalid_sweep_records(kwargs, error):
    with pytest.raises(error):
        replace(tk.decompositions.SweepRecord(0), **kwargs)
