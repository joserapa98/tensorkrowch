"""Tests for tensor-source capabilities consumed by sketching methods."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.sketching.sources import (SupportTensorSource,
                                                           _iter_support)

from tests.decompositions.als._oracles import contract_tt_dense, make_tt_cores


def _compact_tt(cores):
    """Converts standard open-boundary cores to result/model convention."""
    if len(cores) == 1:
        return [cores[0].reshape(-1)]
    return [cores[0].squeeze(0), *cores[1:-1], cores[-1].squeeze(-1)]


class TestSourceEvaluationStats:  # MARK: TestSourceEvaluationStats

    def test_dense_evaluation_and_fiber_count_points_without_sync_timers(self):
        source = tk.decompositions.DenseTensorSource(
            torch.arange(24.).reshape(2, 3, 4))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 1, 2], [1, 2, 3]]))

        source.evaluate(configurations)
        source.fiber(configurations, site=1)

        assert source.evaluation_stats == tk.decompositions.EvaluationStats(
            requested_points=8,
            unique_points=8,
            batches=2,
            cache_hits=0,
            source_calls=2)

    def test_callable_stats_distinguish_source_calls_and_runtime_batches(self):
        source = tk.decompositions.CallableTensorSource(
            lambda indices: indices.sum(dim=1).to(torch.float64),
            in_dim=(2, 3),
            dtype=torch.float64,
            batch_size=2)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]))

        source.evaluate(configurations)

        assert source.evaluation_stats == tk.decompositions.EvaluationStats(
            requested_points=5,
            unique_points=5,
            batches=3,
            cache_hits=0,
            source_calls=1)

    def test_sparse_and_empirical_sources_keep_sparse_lookup_semantics(self):
        distribution = tk.decompositions.SparseTensorSource.from_data(
            dataset=torch.tensor([[0, 1], [0, 1], [1, 0]]),
            in_dim=(2, 2),
            weights=torch.tensor([1., 2., 3.], dtype=torch.float64),
            dtype=torch.float64)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))

        values = distribution.evaluate(configurations)

        assert torch.allclose(
            values, torch.tensor([0., 0.5, 0.5, 0.], dtype=torch.float64))
        assert distribution.support.n_inputs == 2
        assert distribution.evaluation_stats.requested_points == 4


class TestMPSAdapter:  # MARK: TestMPSAdapter

    def test_open_boundary_model_is_reduced_to_raw_tt_cores(self):
        standard = make_tt_cores(
            generator=torch.Generator().manual_seed(37))
        model = tk.models.MPS(
            tensors=_compact_tt(standard), parameterized=False)

        source = tk.decompositions.as_tensor_source(model)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0, 0], [1, 2, 1]]))

        assert isinstance(source, tk.decompositions.FormatSource)
        assert torch.allclose(
            source.evaluate(configurations),
            torch.stack((
                contract_tt_dense(standard)[0, 0, 0],
                contract_tt_dense(standard)[1, 2, 1],
            )))

    def test_periodic_model_is_not_silently_treated_as_a_tt(self):
        model = tk.models.MPS(
            n_features=3,
            phys_dim=2,
            bond_dim=2,
            boundary='pbc')

        source = tk.decompositions.as_tensor_source(model)
        assert source.format._cyclic
        indices = torch.tensor([[0, 0, 0], [1, 1, 1]])
        torch.testing.assert_close(source.evaluate(tk.decompositions.ConfigurationBatch(indices)),
                                   source.format.evaluate(indices))

    @pytest.mark.parametrize('boundary', ['obc', 'pbc'])
    def test_mpo_is_adapted_as_a_matrix(self, boundary):
        model = tk.models.MPO(
            n_features=2, in_dim=2, out_dim=2, bond_dim=2, boundary=boundary)
        source = tk.decompositions.FormatSource(model)
        expected_type = tk.formats.TTM if boundary == 'obc' else tk.formats.TRM
        assert isinstance(source.format, expected_type)
        assert source.is_matrix
        indices = torch.tensor([[0, 1], [1, 0]])
        dense = source.contract_dense()
        torch.testing.assert_close(source.evaluate(
            tk.decompositions.ConfigurationBatch(indices)),
            torch.stack((dense[0, :, 1, :], dense[1, :, 0, :])))


class TestRSInputSources:  # MARK: TestRSInputSources

    def test_dataset_is_normalized_and_duplicate_weights_are_coalesced(self):
        dataset = torch.tensor([[0, 1], [0, 1], [1, 0]])
        source = tk.decompositions.SparseTensorSource.from_data(
            dataset=dataset,
            in_dim=(2, 2),
            weights=torch.tensor([1., 2., 3.], dtype=torch.float64))

        assert isinstance(source, tk.decompositions.SparseTensorSource)
        assert isinstance(source, SupportTensorSource)
        assert torch.equal(
            source.support.as_tensor(), torch.tensor([[0, 1], [1, 0]]))
        assert torch.allclose(
            source.support_values,
            torch.tensor([0.5, 0.5], dtype=torch.float64))

    def test_rs_rejects_non_discrete_sources(self):
        source = tk.decompositions.CallableSource(lambda x: x.sum(-1), 2)
        with pytest.raises(ValueError, match='discrete tensor'):
            tk.decompositions.TTRS(source)

    def test_support_iterator_batches_only_declared_nonzero_entries(self):
        source = tk.decompositions.SparseTensorSource(
            indices=torch.tensor([[1, 1], [0, 1], [1, 1], [0, 0]]),
            values=torch.tensor([1., 2., 3., 4.]),
            in_dim=(3, 3))

        batches = list(_iter_support(source, batch_size=2))
        indices = torch.cat([batch.as_tensor() for batch, _ in batches])
        values = torch.cat([value for _, value in batches])

        assert len(batches) == 2
        assert torch.equal(indices, source.support.as_tensor())
        assert torch.equal(values, source.support_values)
        assert indices.shape[0] == 3
