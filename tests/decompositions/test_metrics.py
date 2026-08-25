"""Tests for structured decomposition metrics."""

import math

import pytest

import torch
import tensorkrowch as tk


class TestDecompositionMetrics:  # MARK: TestDecompositionMetrics

    def test_error_record_stores_batch_values_on_cpu(self):
        per_batch = torch.tensor([1.0, 2.0], requires_grad=True)
        record = tk.decompositions.ErrorRecord(
            kind='reconstruction',
            absolute=torch.tensor(3.0),
            relative=0.5,
            size=2,
            denominator=6.0,
            absolute_per_batch=per_batch)

        assert record.absolute == 3.0
        assert record.relative == 0.5
        assert record.absolute_per_batch.device.type == 'cpu'
        assert not record.absolute_per_batch.requires_grad

    def test_truncation_record(self):
        record = tk.decompositions.TruncationRecord(
            site=1,
            full_rank=5,
            selected_rank=3,
            discarded_squared_norm=4.0,
            local_absolute_error=2.0,
            input_norm=5.0,
            local_relative_error=0.4,
            global_relative_contribution=0.2,
            svd_method='qr_svd',
            discarded_squared_norm_per_batch=torch.tensor([1.0, 3.0]))

        assert record.selected_rank == 3
        assert record.discarded_squared_norm_per_batch.device.type == 'cpu'

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
            log_scale_per_batch=torch.tensor([0.0, math.log(2.0)]),
            global_input_norm=10.0,
            global_input_norm_per_batch=torch.tensor([5.0, 4.0]))

        # Discarded energies are 3**2 and (2 * 1)**2 after rescaling.
        assert record.discarded_squared_norm == pytest.approx(13.0)
        assert record.local_absolute_error == pytest.approx(math.sqrt(13.0))
        assert record.input_norm == pytest.approx(math.sqrt(45.0))
        assert record.local_relative_error == pytest.approx(
            math.sqrt(13.0 / 45.0))
        assert record.global_relative_contribution == pytest.approx(
            math.sqrt(13.0) / 10.0)
        assert torch.allclose(record.local_absolute_error_per_batch,
                              torch.tensor([3.0, 2.0]))
        assert torch.allclose(
            record.global_relative_contribution_per_batch,
            torch.tensor([0.6, 0.5]))

    def test_truncation_record_zero_norm_policy(self):
        *_, info = tk.utils.truncated_svd(
            torch.zeros(3, 3), rank=1, return_info=True)

        record = tk.decompositions.TruncationRecord.from_svd_info(
            info, site=0, global_input_norm=0.0)

        assert record.local_absolute_error == 0.0
        assert record.local_relative_error == 0.0
        assert record.global_relative_contribution == 0.0

    def test_timing_record_children(self):
        child = tk.decompositions.TimingRecord(name='svd', elapsed=0.1)
        parent = tk.decompositions.TimingRecord(
            name='fit', elapsed=0.2, children=[child])

        assert parent.children == (child,)

    def test_fidelity_record_preserves_phase(self):
        overlap = torch.tensor(0.0 + 0.5j)
        record = tk.decompositions.FidelityRecord(overlap)

        assert record.normalized_overlap == 0.5j
        assert record.fidelity == 0.25

    def test_metrics_as_info(self):
        error = tk.decompositions.ErrorRecord(
            kind='samples', absolute=1.0, relative=0.5)
        fidelity = tk.decompositions.FidelityRecord(1j)
        metrics = tk.decompositions.DecompositionMetrics(
            errors=[error],
            fidelities=[fidelity],
            warnings=['diagnostic'])

        info = metrics.as_info()

        assert info['errors'][0]['kind'] == 'samples'
        assert info['fidelities'][0]['fidelity'] == 1.0
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
