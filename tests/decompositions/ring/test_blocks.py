"""Tests for tensor ring block selection, rank estimates and splitting."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.ring.blocks import (
    _BlockSelection, _CentralBlockSelector, _PrescribedCentralBlockSelector,
    _RingRankEstimator,
)


class TestCentralBlockSelector:  # MARK: TestCentralBlockSelector

    def test_grows_bilaterally_from_center(self):
        selection = _CentralBlockSelector().select(
            in_dim := (2, 2, 2, 2, 2),
            rank=2,
            center=2)

        assert isinstance(selection, _BlockSelection)
        assert selection.sites == (1, 2, 3)
        assert selection.in_dim == in_dim[1:4]
        assert selection.growth == ((2, 2), (2, 3), (1, 3))
        assert selection.input_capacity == 8
        assert selection.required_input_capacity == 5
        assert selection.feasible
        assert selection.boundary is None

    @pytest.mark.parametrize(
        ('center', 'sites', 'boundary'),
        [(0, (0, 1, 2), 'left'), (4, (2, 3, 4), 'right')])
    def test_selects_blocks_at_both_boundaries(self, center, sites, boundary):
        selection = _CentralBlockSelector().select(
            (2, 2, 2, 2, 2), rank=2, center=center)

        assert selection.sites == sites
        assert selection.feasible
        assert selection.boundary == boundary

    def test_provider_object_and_heterogeneous_right_link_caps(self):
        class Provider:
            in_dim = (2, 3, 2, 4)

        selection = _CentralBlockSelector().select(
            Provider(), rank=(2, 3, 4, 5), center=1)

        assert selection.sites == (1, 2, 3)
        assert selection.left_rank_cap == 2
        assert selection.right_rank_cap == 5
        assert selection.boundary == 'right'

    def test_insufficient_input_capacity_is_a_result_not_an_exception(self):
        selection = _CentralBlockSelector().select(
            (2, 2, 2), rank=10, center=1)

        assert selection.sites == (0, 1, 2)
        assert not selection.feasible
        assert selection.reason == 'available_sites_exhausted'
        assert selection.input_capacity == 8
        assert selection.required_input_capacity == 101
        assert selection.boundary == 'both'

    def test_bounds_limit_growth_and_preserve_global_boundary_semantics(self):
        selection = _CentralBlockSelector().select(
            (2, 2, 2, 2, 2), rank=3, center=1, bounds=(1, 3))

        assert selection.sites == (1, 2, 3)
        assert not selection.feasible
        assert selection.boundary is None

    def test_prescribed_selector_is_shared_by_fixed_ring_workflows(self):
        selection = _PrescribedCentralBlockSelector().select(
            (2, 3, 4, 5), rank=(2, 3, 4, 5), center=2)

        assert selection.sites == (2,)
        assert selection.left_rank_cap == 3
        assert selection.right_rank_cap == 4
        assert selection.reason == 'prescribed_fixed_rank_center'


class TestRingRankEstimator:  # MARK: TestRingRankEstimator

    def test_reproduces_balanced_uncapped_estimate(self):
        estimate = _RingRankEstimator().estimate(
            left_dim=12,
            right_dim=18,
            auxiliary_rank=6,
            rank_caps=(10, 10, 10))

        assert estimate.rank == (2, 3, 6)
        assert estimate.cyclic_rank_estimate == pytest.approx(6.0)
        assert estimate.feasible
        assert estimate.limitations == ()

    def test_adapts_adjacent_ranks_to_a_low_cyclic_cap(self):
        estimate = _RingRankEstimator().estimate(
            left_dim=12,
            right_dim=18,
            auxiliary_rank=6,
            rank_caps=(10, 10, 3))

        assert estimate.rank == (4, 6, 3)
        assert estimate.feasible

    def test_reports_each_capacity_blocked_by_caps(self):
        estimate = _RingRankEstimator().estimate(
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


def _contract_cores(cores):
    tensor = cores[0]
    for core in cores[1:]:
        tensor = torch.tensordot(tensor, core, dims=([-1], [0]))
    return tensor


class TestSplitBlock:

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('in_dim', [(1,), (2,), (2, 3), (2, 3, 4, 2)])
    def test_unbounded_split_preserves_dense_block_and_external_dimensions(
            self, dtype, in_dim):
        from tensorkrowch.decompositions.ring.blocks import _split_block

        block = torch.randn(
            2, *in_dim, 3, dtype=dtype,
            generator=torch.Generator().manual_seed(906))
        cores = _split_block(block, in_dim)
        assert len(cores) == len(in_dim)
        assert cores[0].shape[0] == 2
        assert cores[-1].shape[-1] == 3
        assert all(core.ndim == 3 for core in cores)
        torch.testing.assert_close(_contract_cores(cores), block)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_per_cut_rank_profile_recovers_nonuniform_ring_block(self, dtype):
        from tensorkrowch.decompositions.ring.blocks import _split_block

        generator = torch.Generator().manual_seed(907)
        reference = [torch.randn(*shape, dtype=dtype, generator=generator)
                     for shape in ((2, 3, 2), (2, 4, 1),
                                   (1, 3, 3), (3, 4, 2))]
        block = _contract_cores(reference)
        metrics = tk.decompositions.DecompositionMetrics()
        cores = _split_block(
            block, (3, 4, 3, 4), rank=[2, 1, 3], metrics=metrics)
        assert [core.shape[-1] for core in cores[:-1]] == [2, 1, 3]
        assert [r.site for r in metrics.truncations] == [0, 1, 2]
        torch.testing.assert_close(_contract_cores(cores), block)

    @pytest.mark.parametrize('rank', [5, [5, 6]])
    def test_large_rank_caps_do_not_pad_unavailable_directions(self, rank):
        from tensorkrowch.decompositions.ring.blocks import _split_block

        block = torch.randn(
            1, 2, 2, 2, 1, dtype=torch.complex128,
            generator=torch.Generator().manual_seed(908))
        cores = _split_block(block, (2, 2, 2), rank=rank)
        assert [core.shape[-1] for core in cores[:-1]] == [2, 2]
        torch.testing.assert_close(_contract_cores(cores), block)

    def test_rank_one_profile_introduces_measurable_truncation(self):
        from tensorkrowch.decompositions.ring.blocks import _split_block

        block = torch.eye(4, dtype=torch.float64).reshape(1, 2, 2, 2, 2, 1)
        metrics = tk.decompositions.DecompositionMetrics()
        cores = _split_block(
            block, (2, 2, 2, 2), rank=[1, 1, 1], metrics=metrics)
        assert all(core.shape[-1] == 1 for core in cores[:-1])
        assert (_contract_cores(cores) - block).norm() > 0.1
        assert len(metrics.truncations) == 3
        assert any(record.local_abs_error > 0
                   for record in metrics.truncations)

    def test_empty_profile_is_valid_for_one_site(self):
        from tensorkrowch.decompositions.ring.blocks import _split_block

        block = torch.ones(2, 1, 3)
        cores = _split_block(block, (1,), rank=[])
        assert len(cores) == 1
        torch.testing.assert_close(cores[0], block)

    @pytest.mark.parametrize('options, error', [
        ({'rank': [2]}, ValueError),
        ({'rank': [2, 0]}, ValueError),
        ({'rank': [2, True]}, TypeError),
        ({'rank': '2'}, TypeError),
        ({'rank': True}, TypeError),
        ({'rtol': -1}, ValueError),
        ({'metrics': {}}, TypeError),
        ({'renormalize': 1}, TypeError),
    ])
    def test_invalid_profile_and_options(self, options, error):
        from tensorkrowch.decompositions.ring.blocks import _split_block

        with pytest.raises(error):
            _split_block(torch.ones(1, 2, 2, 2, 1), (2, 2, 2), **options)

    @pytest.mark.parametrize('block, dimensions, error', [
        (None, (2,), TypeError),
        (torch.ones(2, 2), (2,), ValueError),
        (torch.ones(1, 2, 2, 1), (2, 3), ValueError),
        (torch.ones(0, 2, 1), (2,), ValueError),
        (torch.ones(1, 2, 1), (), ValueError),
    ])
    def test_invalid_block_layout(self, block, dimensions, error):
        from tensorkrowch.decompositions.ring.blocks import _split_block

        with pytest.raises(error):
            _split_block(block, dimensions)
