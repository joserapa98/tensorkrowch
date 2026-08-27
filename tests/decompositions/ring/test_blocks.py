"""Tests for tensor-ring block selection, rank estimates and splitting."""

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.decompositions.ring.blocks import (
    BlockSelection,
    CentralBlockSelector,
    RingRankEstimator,
    split_block_ttsvd,
)


class TestCentralBlockSelector:  # MARK: TestCentralBlockSelector

    def test_grows_bilaterally_from_center(self):
        selection = CentralBlockSelector().select(
            input_dim := (2, 2, 2, 2, 2),
            rank=2,
            center=2)

        assert isinstance(selection, BlockSelection)
        assert selection.sites == (1, 2, 3)
        assert selection.input_dim == input_dim[1:4]
        assert selection.growth == ((2, 2), (2, 3), (1, 3))
        assert selection.input_capacity == 8
        assert selection.required_input_capacity == 5
        assert selection.feasible
        assert selection.boundary is None

    @pytest.mark.parametrize(
        ('center', 'sites', 'boundary'),
        [(0, (0, 1, 2), 'left'), (4, (2, 3, 4), 'right')])
    def test_selects_blocks_at_both_boundaries(self, center, sites, boundary):
        selection = CentralBlockSelector().select(
            (2, 2, 2, 2, 2), rank=2, center=center)

        assert selection.sites == sites
        assert selection.feasible
        assert selection.boundary == boundary

    def test_provider_object_and_heterogeneous_right_link_caps(self):
        class Provider:
            input_dim = (2, 3, 2, 4)

        selection = CentralBlockSelector().select(
            Provider(), rank=(2, 3, 4, 5), center=1)

        assert selection.sites == (1, 2, 3)
        assert selection.left_rank_cap == 2
        assert selection.right_rank_cap == 5
        assert selection.boundary == 'right'

    def test_insufficient_input_capacity_is_a_result_not_an_exception(self):
        selection = CentralBlockSelector().select(
            (2, 2, 2), rank=10, center=1)

        assert selection.sites == (0, 1, 2)
        assert not selection.feasible
        assert selection.reason == 'available_sites_exhausted'
        assert selection.input_capacity == 8
        assert selection.required_input_capacity == 101
        assert selection.boundary == 'both'

    def test_bounds_limit_growth_and_preserve_global_boundary_semantics(self):
        selection = CentralBlockSelector().select(
            (2, 2, 2, 2, 2), rank=3, center=1, bounds=(1, 3))

        assert selection.sites == (1, 2, 3)
        assert not selection.feasible
        assert selection.boundary is None


class TestRingRankEstimator:  # MARK: TestRingRankEstimator

    def test_reproduces_balanced_uncapped_estimate(self):
        estimate = RingRankEstimator().estimate(
            left_dim=12,
            right_dim=18,
            auxiliary_rank=6,
            rank_caps=(10, 10, 10))

        assert estimate.rank == (2, 3, 6)
        assert estimate.cyclic_rank_estimate == pytest.approx(6.0)
        assert estimate.feasible
        assert estimate.limitations == ()

    def test_adapts_adjacent_ranks_to_a_low_cyclic_cap(self):
        estimate = RingRankEstimator().estimate(
            left_dim=12,
            right_dim=18,
            auxiliary_rank=6,
            rank_caps=(10, 10, 3))

        assert estimate.rank == (4, 6, 3)
        assert estimate.feasible

    def test_reports_each_capacity_blocked_by_caps(self):
        estimate = RingRankEstimator().estimate(
            left_dim=20,
            right_dim=24,
            auxiliary_rank=20,
            rank_caps=(2, 3, 2))

        assert estimate.rank == (2, 3, 2)
        assert not estimate.feasible
        assert estimate.limitations == (
            'left_dim_exceeds_rank_capacity',
            'right_dim_exceeds_rank_capacity',
            'auxiliary_rank_exceeds_adjacent_rank_capacity',
        )


class TestBlockTTSVD:  # MARK: TestBlockTTSVD

    @pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_exact_split_reconstructs_supercore(self, svd_method, dtype):
        block = torch.randn(
            2, 3, 4, 5, 2,
            dtype=dtype,
            generator=torch.Generator().manual_seed(130))

        with tk.svd_method(svd_method):
            split = split_block_ttsvd(block, input_dim=(3, 4, 5))

        assert split.effective_rank == (6, 10)
        assert split.rank == (6, 10)
        assert not split.padded
        assert torch.allclose(
            split.contract_dense(), block, rtol=2e-10, atol=2e-10)

    def test_shared_rank_truncates_all_internal_cuts(self):
        block = torch.randn(
            2, 3, 4, 5, 2,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(131))
        split = split_block_ttsvd(block, input_dim=(3, 4, 5), rank=2)
        fused = block.reshape(6, 4, 10)
        reference = tk.decompositions.TTSVD(
            fused, output_device=None).fit(rank=2).contract_dense()

        assert split.rank == (2, 2)
        assert torch.allclose(
            split.contract_dense().reshape_as(reference),
            reference,
            rtol=1e-12,
            atol=1e-12)

    def test_padding_is_opt_in_and_preserves_the_split(self):
        block = torch.randn(
            1, 2, 2, 1,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(132))
        compact = split_block_ttsvd(block, input_dim=(2, 2), rank=5)
        padded = split_block_ttsvd(
            block, input_dim=(2, 2), rank=5, pad_rank=True)

        assert compact.effective_rank == (2,)
        assert compact.rank == (2,)
        assert compact.padding == (0,)
        assert padded.effective_rank == (2,)
        assert padded.rank == (5,)
        assert padded.padding == (3,)
        assert padded.padded
        assert padded.metadata['pad_rank'] is True
        assert torch.allclose(padded.contract_dense(), compact.contract_dense())

    def test_one_site_preserves_both_external_ranks(self):
        block = torch.randn(3, 4, 2)
        split = split_block_ttsvd(block, input_dim=(4,), rank=5)

        assert split.cores[0].shape == (3, 4, 2)
        assert split.rank == ()
        assert split.padding == ()
        assert torch.equal(split.contract_dense(), block)

    def test_metrics_are_collected_only_when_requested(self):
        block = torch.randn(2, 3, 4, 2)
        fast = split_block_ttsvd(block, input_dim=(3, 4), rank=2)
        measured = split_block_ttsvd(
            block,
            input_dim=(3, 4),
            rank=2,
            collect_metrics=True)

        assert fast.metrics.truncations == []
        assert len(measured.metrics.truncations) == 1
        assert measured.metrics.errors[0].kind == 'truncation'

    def test_padding_requires_an_explicit_rank(self):
        with pytest.raises(ValueError, match='required'):
            split_block_ttsvd(
                torch.randn(1, 2, 2, 1),
                input_dim=(2, 2),
                pad_rank=True)
