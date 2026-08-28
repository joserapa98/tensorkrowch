"""Tests for tensor-source capabilities consumed by sketching methods."""

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.decompositions.sketching.sources import (
    SketchContractableSource,
)
from tests.decompositions.als._oracles import (contract_tt_dense,
                                               make_tt_cores)


def _compact_tt(cores):
    """Converts standard open-boundary cores to result/model convention."""
    if len(cores) == 1:
        return [cores[0].reshape(-1)]
    return [cores[0].squeeze(0), *cores[1:-1], cores[-1].squeeze(-1)]


def _random_ttm(input_dim, output_dim, rank, dtype, generator):
    """Creates a deterministic TTM decomposition for contraction tests."""
    ranks = (1, *rank, 1)
    standard = [
        torch.randn(
            ranks[site], input_value, output_value, ranks[site + 1],
            dtype=dtype,
            generator=generator)
        for site, (input_value, output_value) in enumerate(
            zip(input_dim, output_dim))
    ]
    if len(standard) == 1:
        compact = [standard[0].reshape(input_dim[0], output_dim[0])]
    else:
        compact = [standard[0].squeeze(0).permute(0, 2, 1)]
        compact.extend(core.permute(0, 1, 3, 2)
                       for core in standard[1:-1])
        compact.append(standard[-1].squeeze(-1))
    return tk.decompositions.TTMDecomposition(compact)


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
            input_dim=(2, 3),
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
        source.reset_evaluation_stats()
        assert source.evaluation_stats == tk.decompositions.EvaluationStats()

    def test_sparse_and_empirical_sources_keep_sparse_lookup_semantics(self):
        distribution = tk.decompositions.EmpiricalDistribution(
            dataset=torch.tensor([[0, 1], [0, 1], [1, 0]]),
            input_dim=(2, 2),
            weights=torch.tensor([1., 2., 3.], dtype=torch.float64),
            dtype=torch.float64)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))

        values = distribution.evaluate(configurations)

        assert torch.allclose(
            values, torch.tensor([0., 0.5, 0.5, 0.], dtype=torch.float64))
        assert distribution.support.n_sites == 2
        assert distribution.evaluation_stats.requested_points == 4


class TestStructuredSketchContraction:  # MARK: TestStructuredSketchContraction

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_scalar_tt_sketch_matches_dense_inner_product(self, dtype):
        source_cores = make_tt_cores(
            dtype=dtype,
            generator=torch.Generator().manual_seed(31))
        sketch_cores = make_tt_cores(
            dtype=dtype,
            generator=torch.Generator().manual_seed(32))
        source = tk.decompositions.TTTensorSource(source_cores)
        sketch = tk.decompositions.TTDecomposition(_compact_tt(sketch_cores))

        result = source.contract_sketch(sketch)
        expected = (
            contract_tt_dense(source_cores) *
            contract_tt_dense(sketch_cores).conj()).sum()

        assert isinstance(source, SketchContractableSource)
        assert torch.allclose(result, expected)
        assert source.evaluation_stats == tk.decompositions.EvaluationStats(
            source_calls=1)

    def test_conjugation_can_be_disabled_explicitly(self):
        source_cores = make_tt_cores(
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(33))
        sketch_cores = make_tt_cores(
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(34))
        source = tk.decompositions.TTTensorSource(source_cores)

        result = source.contract_sketch(
            _compact_tt(sketch_cores), conjugate_sketch=False)

        expected = (
            contract_tt_dense(source_cores) *
            contract_tt_dense(sketch_cores)).sum()
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_ttm_sketch_returns_output_tt_without_dense_source(self, dtype):
        input_dim = (2, 3, 2)
        output_dim = (2, 2, 3)
        source_cores = make_tt_cores(
            input_dim=input_dim,
            dtype=dtype,
            generator=torch.Generator().manual_seed(35))
        source = tk.decompositions.TTTensorSource(source_cores)
        sketch = _random_ttm(
            input_dim,
            output_dim,
            rank=(2, 2),
            dtype=dtype,
            generator=torch.Generator().manual_seed(36))

        result = source.contract_sketch(sketch)

        dense_source = contract_tt_dense(source_cores)
        dense_sketch = sketch.contract_dense()
        expected = torch.einsum(
            'abc,apbqcr->pqr', dense_source, dense_sketch.conj())
        assert isinstance(result, tk.decompositions.TTDecomposition)
        assert result.input_dim == output_dim
        assert torch.allclose(result.contract_dense(), expected)

    def test_structured_sketch_validates_input_dimensions(self):
        source = tk.decompositions.TTTensorSource(
            make_tt_cores(input_dim=(2, 3), rank=(2,)))
        sketch = tk.decompositions.TTDecomposition(
            _compact_tt(make_tt_cores(input_dim=(2, 4), rank=(2,))))

        with pytest.raises(ValueError, match='input dimensions'):
            source.contract_sketch(sketch)


class TestMPSAdapter:  # MARK: TestMPSAdapter

    def test_open_boundary_model_is_reduced_to_raw_tt_cores(self):
        standard = make_tt_cores(
            generator=torch.Generator().manual_seed(37))
        model = tk.models.MPS(
            tensors=_compact_tt(standard), parameterized=False)

        source = tk.decompositions.as_tensor_source(model)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0, 0], [1, 2, 1]]))

        assert isinstance(source, tk.decompositions.TTTensorSource)
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

        with pytest.raises(ValueError, match='open-boundary'):
            tk.decompositions.as_tensor_source(model)
