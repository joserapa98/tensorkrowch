"""Tests for shared SVD source preparation and progress reporting."""

import torch

import tensorkrowch as tk
from tensorkrowch.decompositions.svd.utils import (_prepare_svd_source,
                                                   _SVDProgress)


def test_prepare_matrix_source():
    tensor = torch.arange(16., dtype=torch.float64).reshape(2, 2, 2, 2)
    source = tk.decompositions.DenseTensorSource.matrix(tensor)
    prepared_source, prepared, n_batches = _prepare_svd_source(source)
    assert prepared_source is source
    assert n_batches == 0
    torch.testing.assert_close(prepared, tensor.reshape(4, 4))


def test_prepare_batched_source():
    tensor = torch.arange(32., dtype=torch.float64).reshape(2, 2, 2, 2, 2)
    source = tk.decompositions.DenseTensorSource(tensor, n_batches=1)
    prepared_source, prepared, n_batches = _prepare_svd_source(source)
    assert prepared_source is source
    assert n_batches == 1
    torch.testing.assert_close(prepared, tensor)


def test_progress_aggregates_batched_errors():
    class Observer:
        def __init__(self):
            self.events = []

        def emit(self, event):
            self.events.append(event)

    observer = Observer()
    record = tk.decompositions.TruncationRecord(
        site=0, full_rank=2, selected_rank=1,
        local_abs_error=torch.tensor([3., 4.], dtype=torch.float64),
        local_rel_error=torch.tensor([0.5, 0.5], dtype=torch.float64),
        local_norm=torch.tensor([6., 8.], dtype=torch.float64))
    _SVDProgress(observer, 'TT-SVD').cut_complete(0, record, 0.1)
    values = observer.events[0].values
    torch.testing.assert_close(values['absolute_error'],
                               torch.tensor(5., dtype=torch.float64))
    torch.testing.assert_close(values['relative_error'],
                               torch.tensor(0.5, dtype=torch.float64))
