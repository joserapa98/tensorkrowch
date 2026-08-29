"""Tests for experimental Tensor Ring Recursive Sketching."""

import pytest

import torch
import tensorkrowch as tk


def _rank_one_sparse(dtype=torch.float64):
    factors = [
        torch.tensor([1., 2.], dtype=dtype),
        torch.tensor([1., 3.], dtype=dtype),
        torch.tensor([2., 1.], dtype=dtype),
    ]
    dense = torch.einsum('i,j,k->ijk', *factors)
    indices = torch.cartesian_prod(*(
        torch.arange(dim) for dim in dense.shape))
    source = tk.decompositions.SparseTensorSource(
        indices, dense.reshape(-1), dense.shape)
    return source, dense


class TestTRRS:  # MARK: TestTRRS

    @pytest.mark.parametrize('operator', [
        tk.decompositions.SampledSketch(),
        tk.decompositions.MarginalSketch.markov(),
        tk.decompositions.TTStackSketch(tt_rank=2, n_stacks=2),
    ])
    def test_sparse_rank_one_source_is_recovered(self, operator):
        source, dense = _rank_one_sparse()
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRS(
                source,
                sketch_operator=operator,
                output_device=None).fit(
                    rank=1,
                    generator=torch.Generator().manual_seed(311),
                    strict_system=True,
                    collect_metrics=True)

        assert result.rank == [1, 1, 1]
        assert torch.allclose(
            result.contract_dense(), dense, rtol=1e-9, atol=1e-11)
        assert result.metrics.fidelities[0].fidelity > 1 - 1e-10
        assert result.metrics.errors[-1].kind == 'source_support'
        assert result.metrics.errors[-1].relative < 1e-9
        assert result.metadata['algorithm'] == 'tr_rs'
        assert result.metadata['experimental'] is True

    def test_dataset_functional_api_returns_information(self):
        dataset = torch.tensor([
            [0, 0, 0], [0, 0, 0], [1, 1, 1]])
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            cores, info = tk.decompositions.tr_rs(
                dataset=dataset,
                input_dim=(2, 2, 2),
                rank=1,
                return_info=True)

        result = tk.decompositions.TRDecomposition(cores)
        assert result.rank == [1, 1, 1]
        assert info['metadata']['source_type'] == 'EmpiricalDistribution'
        assert info['metrics']['fidelities'][0]['fidelity'] > 1 - 1e-5

    def test_tt_source_uses_current_generic_oracle(self):
        _, dense = _rank_one_sparse()
        tt = tk.decompositions.TTSVD(
            dense, output_device=None).fit(rank=1)
        source = tk.decompositions.TTTensorSource(tt)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRS(
                source, output_device=None).fit(rank=1)

        assert torch.allclose(result.contract_dense(), dense)
        assert result.metrics.fidelities[0].fidelity > 1 - 1e-10
        assert source.evaluation_stats.requested_points == dense.numel()

    def test_history_observer_receives_one_tr_rs_hierarchy(self):
        source, _ = _rank_one_sparse()
        observer = tk.decompositions.HistoryObserver()
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRS(source).fit(
                rank=1, observer=observer)

        assert observer.metrics is result.metrics
        assert observer.events[0].name == 'start'
        assert all(event.phase == 'TR-RS' for event in observer.events)
        assert sum(event.name == 'site_complete'
                   for event in observer.events) == 3

    def test_rank_and_warm_start_validation(self):
        source, _ = _rank_one_sparse()
        with pytest.raises(TypeError, match='rank'):
            tk.decompositions.TRRS(source).fit(rank=[1, 1, 1])
        initial = tk.decompositions.TRDecomposition([
            torch.ones(1, 2, 1) for _ in range(3)])
        with pytest.raises(NotImplementedError, match='warm-start'):
            tk.decompositions.TRRS(source).fit(
                rank=1, warm_start=initial)


__all__ = []
