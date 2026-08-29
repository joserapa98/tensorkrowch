"""Tests for TT-RS sketch operators and core-determining systems."""

import pytest

import torch
import tensorkrowch as tk


def _rank_one_sparse(input_dim=(2, 3, 2), dtype=torch.float64):
    factors = [
        torch.arange(1, dimension + 1, dtype=dtype)
        for dimension in input_dim
    ]
    dense = factors[0]
    for factor in factors[1:]:
        dense = dense.unsqueeze(-1) * factor
    indices = torch.cartesian_prod(*(
        torch.arange(dimension) for dimension in input_dim))
    return tk.decompositions.SparseTensorSource(
        indices, dense.reshape(-1), input_dim), dense


class TestMarginalSketch:  # MARK: TestMarginalSketch

    def test_markov_order_one_builds_neighbor_marginals(self):
        source, dense = _rank_one_sparse()
        system = tk.decompositions.MarginalSketch.markov().builder(
            source).build()

        assert system.phi[0].shape == (1, 2, 3)
        assert system.phi[1].shape == (2, 3, 2)
        assert system.phi[2].shape == (3, 2, 1)
        assert torch.allclose(system.phi[0].squeeze(0), dense.sum(dim=2))
        assert torch.allclose(system.phi[1], dense)
        assert torch.allclose(system.phi[2].squeeze(-1), dense.sum(dim=0))
        assert system.diagnostics['marginal_order'] == 1

    def test_factor_changes_the_contracted_measure_explicitly(self):
        source, dense = _rank_one_sparse(input_dim=(2, 2, 2))
        factor = (
            torch.tensor([1., 2.]),
            torch.tensor([1., 3.]),
            torch.tensor([2., 1.]),
        )
        system = tk.decompositions.MarginalSketch(
            order=1, factor=factor).builder(source).build()
        weighted = torch.einsum(
            'ijk,i,j,k->ijk', dense, *factor)

        assert torch.allclose(system.phi[0].squeeze(0), weighted.sum(dim=2))
        assert system.diagnostics['weighted_marginal']


class TestSampledSketch:  # MARK: TestSampledSketch

    def test_sparse_route_uses_support_without_source_evaluations(self):
        source, dense = _rank_one_sparse()
        system = tk.decompositions.SampledSketch().builder(source).build(
            batch_size=5)
        result = system.solve(rank=1, strict_system=True)

        assert source.evaluation_stats.source_calls == 0
        assert system.diagnostics['support_size'] == dense.numel()
        assert torch.allclose(result.contract_dense(), dense)

    def test_sample_cap_preserves_recursive_prefix_dimensions(self):
        source, _ = _rank_one_sparse(input_dim=(2, 2, 2, 2))
        system = tk.decompositions.SampledSketch(
            sketch_size=2).builder(source).build(
                generator=torch.Generator().manual_seed(201))

        assert all(dimension <= 2
                   for dimension in system.diagnostics['left_dimensions'])
        assert all(block.shape[0] <= 2 for block in system.left_blocks)


class TestTTStackSketch:  # MARK: TestTTStackSketch

    @pytest.mark.parametrize(
        ('tt_rank', 'n_stacks'), [(1, 3), (3, 1), (2, 2)])
    def test_partial_sketch_dimension_is_stack_times_tt_rank(
            self, tt_rank, n_stacks):
        source, _ = _rank_one_sparse()
        system = tk.decompositions.TTStackSketch(
            tt_rank=tt_rank,
            n_stacks=n_stacks).builder(source).build(
                generator=torch.Generator().manual_seed(202))
        expected = tt_rank * n_stacks

        assert system.diagnostics['sketch_dimension'] == expected
        assert system.diagnostics['left_dimensions'] == (expected, expected)
        assert system.diagnostics['right_dimensions'] == (expected, expected)
        assert system.diagnostics['core_determining_gate'] == \
            'recursive_left_and_suffix_right'

    def test_gaussian_tt_stack_recovers_rank_one_sparse_tensor(self):
        source, dense = _rank_one_sparse()
        system = tk.decompositions.TTStackSketch(
            tt_rank=2,
            n_stacks=2).builder(source).build(
                generator=torch.Generator().manual_seed(203))
        result = system.solve(rank=1, strict_system=True)

        assert torch.allclose(
            result.contract_dense(), dense, rtol=1e-10, atol=1e-12)
        assert result.metadata['coefficient_ranks'] == [1, 1]

    def test_orthogonal_variant_is_labelled_separately(self):
        source, _ = _rank_one_sparse(input_dim=(3, 3, 3))
        system = tk.decompositions.TTStackSketch(
            tt_rank=2,
            n_stacks=1,
            orthogonal=True).builder(source).build(
                generator=torch.Generator().manual_seed(204))

        assert system.diagnostics['orthogonal']
        assert system.diagnostics['distribution'] == 'orthogonal_variant'


class TestCoreDeterminingSystem:  # MARK: TestCoreDeterminingSystem

    def test_strict_system_rejects_unidentifiable_rank(self):
        system = tk.decompositions.CoreDeterminingSystem(
            phi=(
                torch.ones(1, 2, 1),
                torch.ones(1, 2, 1),
            ),
            left_blocks=(torch.zeros(1, 2, 1),),
            input_dim=(2, 2),
            operator='degenerate_test')

        with pytest.raises(ValueError, match='rank deficient'):
            system.solve(rank=1, strict_system=True)

    def test_metrics_are_only_collected_when_requested(self):
        source, _ = _rank_one_sparse()
        system = tk.decompositions.MarginalSketch.markov().builder(
            source).build()
        fast = system.solve(rank=1)
        measured = system.solve(rank=1, collect_metrics=True)

        assert not fast.metrics.truncations
        assert not fast.metrics.local_solves
        assert fast.metadata['coefficient_ranks'] is None
        assert len(measured.metrics.truncations) == 2
        assert len(measured.metrics.local_solves) == 2


class TestStructuredTTBackend:  # MARK: TestStructuredTTBackend

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('operator', [
        tk.decompositions.SampledSketch(
            samples=torch.tensor([[0, 0, 0], [1, 1, 1]])),
        tk.decompositions.MarginalSketch.markov(),
        tk.decompositions.TTStackSketch(tt_rank=2, n_stacks=2),
    ])
    def test_structured_system_matches_sparse_dense_oracle(
            self, dtype, operator):
        generator = torch.Generator().manual_seed(205)
        cores = [
            torch.randn(1, 2, 2, dtype=dtype, generator=generator),
            torch.randn(2, 2, 2, dtype=dtype, generator=generator),
            torch.randn(2, 2, 1, dtype=dtype, generator=generator),
        ]
        tt_source = tk.decompositions.TTTensorSource(cores)
        dense = tk.decompositions.TTDecomposition([
            cores[0].squeeze(0), cores[1], cores[2].squeeze(-1)
        ]).contract_dense()
        indices = torch.cartesian_prod(*(
            torch.arange(dimension) for dimension in dense.shape))
        oracle_source = tk.decompositions.SparseTensorSource(
            indices, dense.reshape(-1), dense.shape)

        structured = operator.builder(tt_source).build(
            generator=torch.Generator().manual_seed(206))
        oracle = operator.builder(oracle_source).build(
            generator=torch.Generator().manual_seed(206))

        assert structured.diagnostics['source_path'] == 'structured_tt'
        assert tt_source.evaluation_stats.requested_points == 0
        assert all(torch.allclose(left, right)
                   for left, right in zip(structured.phi, oracle.phi))
        assert all(torch.allclose(left, right)
                   for left, right in zip(
                       structured.left_blocks, oracle.left_blocks))


__all__ = []
