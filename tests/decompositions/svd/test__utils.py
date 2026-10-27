"""Tests for shared SVD source preparation and progress reporting."""

import pytest
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions.svd._utils import (_prepare_svd_source,
                                                   _SVDProgress)


def test_prepare_matrix_source():
    tensor = torch.arange(16., dtype=torch.float64).reshape(2, 2, 2, 2)
    source = tk.decompositions.DenseTensorSource.matrix(tensor, ordering='interleaved')
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


@pytest.mark.parametrize('matrix', [False, True])
@pytest.mark.parametrize('cyclic', [False, True])
def test_svd_rejects_continuous_and_integral_sources(matrix, cyclic):
    engine = (tk.decompositions.TRMSVD if cyclic else tk.decompositions.TTMSVD) \
        if matrix else (tk.decompositions.TRSVD if cyclic else tk.decompositions.TTSVD)
    continuous = tk.decompositions.CallableSource.matrix(
        lambda x: x, 2, out_shape=(2, 2)) if matrix else \
        tk.decompositions.CallableSource(lambda x: x.sum(-1), 2)
    with pytest.raises(ValueError, match='discrete tensor'):
        engine(continuous)
    assert continuous.evaluation_stats.source_calls == 0
    source = tk.decompositions.DenseTensorSource.matrix(
        torch.ones(2, 2, 2, 2, dtype=torch.long)) if matrix else \
        tk.decompositions.DenseTensorSource(torch.ones(2, 3, dtype=torch.long))
    with pytest.raises(TypeError, match='floating or complex'):
        engine(source)


@pytest.mark.parametrize('returned, batches, error', [
    ([1., 2.], 0, TypeError),
    (torch.ones(2, 3), True, TypeError),
    (torch.ones(2, 3), -1, ValueError),
    (torch.ones(2, 3), 2, ValueError),
    (torch.empty(2, 0), 0, ValueError),
])
def test_invalid_dense_contraction_capabilities(returned, batches, error):
    class Source:
        n_batches = batches

        def contract_dense(self):
            return returned

    with pytest.raises(error):
        _prepare_svd_source(Source())


@pytest.mark.parametrize('matrix', [False, True])
def test_ring_svd_rejects_batched_sources(matrix):
    if matrix:
        format = tk.formats.TTM([torch.randn(2, 2, 1, 3),
                                 torch.randn(2, 1, 2, 3)], n_batches=1)
        source = tk.decompositions.FormatSource(format)
        engine = tk.decompositions.TRMSVD
    else:
        source = tk.decompositions.DenseTensorSource(torch.ones(2, 3, 4), 1)
        engine = tk.decompositions.TRSVD
    with pytest.raises(ValueError, match='n_batches'):
        engine(source)
