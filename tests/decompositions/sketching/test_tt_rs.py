"""Tests for Tensor Train Recursive Sketching over complete sources."""

import pytest

import torch
import tensorkrowch as tk


def _rank_one_sparse(dtype=torch.float64):
    left = torch.tensor([1., 2.], dtype=dtype)
    middle = torch.tensor([1., 3., 2.], dtype=dtype)
    right = torch.tensor([2., 1.], dtype=dtype)
    dense = torch.einsum('i,j,k->ijk', left, middle, right)
    indices = torch.cartesian_prod(
        torch.arange(2), torch.arange(3), torch.arange(2))
    source = tk.decompositions.SparseTensorSource(
        indices, dense.reshape(-1), dense.shape)
    return source, dense


class TestTTRS:  # MARK: TestTTRS

    @pytest.mark.parametrize('operator', [
        tk.decompositions.SampledSketch(),
        tk.decompositions.MarginalSketch.markov(),
        tk.decompositions.TTStackSketch(tt_rank=2, n_stacks=2),
    ])
    def test_sparse_rank_one_source_is_recovered(self, operator):
        source, dense = _rank_one_sparse()
        result = tk.decompositions.TTRS(
            source,
            sketch_operator=operator,
            output_device=None).fit(
                rank=1,
                generator=torch.Generator().manual_seed(301),
                strict_system=True,
                collect_metrics=True)

        assert result.rank == [1, 1]
        assert torch.allclose(
            result.contract_dense(), dense, rtol=1e-10, atol=1e-12)
        assert result.metrics.errors[0].kind == 'source_support'
        assert result.metrics.errors[0].relative < 1e-10
        assert result.metadata['source_type'] == 'SparseTensorSource'
        assert result.metadata['system']['source_path'] == 'support'

    def test_dataset_and_functional_api_are_distinct_from_rss_samples(self):
        dataset = torch.tensor([
            [0, 0], [0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
        cores, info = tk.decompositions.tt_rs(
            dataset=dataset,
            input_dim=(2, 2),
            rank=2,
            return_info=True)
        expected = torch.tensor(
            [[2., 1.], [1., 2.]], dtype=torch.get_default_dtype()) / 6

        result = tk.decompositions.TTDecomposition(cores)
        assert torch.allclose(result.contract_dense(), expected)
        assert info['metadata']['source_type'] == 'EmpiricalDistribution'
        assert info['metrics']['errors'][0]['kind'] == 'source_support'

    def test_tt_source_uses_structured_contractions(self):
        source, dense = _rank_one_sparse()
        tt = tk.decompositions.TTSVD(
            dense, output_device=None).fit(rank=1)
        tt_source = tk.decompositions.TTTensorSource(tt)
        result = tk.decompositions.TTRS(
            tt_source,
            sketch_operator=tk.decompositions.MarginalSketch.markov(),
            output_device=None).fit(rank=1, batch_size=3)

        assert torch.allclose(result.contract_dense(), dense)
        assert result.metadata['system']['source_path'] == 'structured_tt'
        assert result.metadata['system']['structured_kernel'] == \
            'markov_marginal'
        assert tt_source.evaluation_stats.requested_points == 0

    def test_repeated_random_fits_are_independent_and_reproducible(self):
        source, _ = _rank_one_sparse()
        decomposer = tk.decompositions.TTRS(
            source,
            sketch_operator=tk.decompositions.TTStackSketch(
                tt_rank=2, n_stacks=2),
            output_device=None)
        first = decomposer.fit(
            rank=1, generator=torch.Generator().manual_seed(302))
        second = decomposer.fit(
            rank=1, generator=torch.Generator().manual_seed(302))

        assert first is not second
        assert all(left is not right
                   for left, right in zip(first.cores, second.cores))
        assert all(torch.allclose(left, right)
                   for left, right in zip(first.cores, second.cores))

    def test_warm_start_is_explicitly_rejected(self):
        source, _ = _rank_one_sparse()
        initial = tk.decompositions.TTRS(source).fit(rank=1)
        with pytest.raises(NotImplementedError, match='warm-start'):
            tk.decompositions.TTRS(source).fit(
                rank=1, warm_start=initial)

    def test_history_observer_receives_clean_hierarchy(self):
        source, _ = _rank_one_sparse()
        observer = tk.decompositions.HistoryObserver()
        result = tk.decompositions.TTRS(source).fit(
            rank=1, observer=observer)

        assert observer.metrics is result.metrics
        assert observer.events[0].name == 'start'
        assert sum(event.name == 'site_complete'
                   for event in observer.events) == 3
        assert next(event for event in observer.events
                    if event.name == 'summary').values['rank'] == [1, 1]


__all__ = []
