"""Tests for tensor ring block selection, shared partitions and splitting."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.ring.blocks import (
    _discovery_partition, _select_block, _ring_partition, _spectral_block,
)


class TestSpectralBlockSelector:  # MARK: TestSpectralBlockSelector

    @pytest.mark.parametrize('ranks,center,sites', [
        ((), 0, (0,)),
        ((3,), 0, (0, 1)),
        ((9, 9, 9, 9), 2, (2,)),
        ((9, 6, 9), 1, (1, 2)),
        ((9, 6, 9), 2, (1, 2)),
        ((9, 6, 4, 6), 2, (2, 3)),
        ((4, 4, 6, 4, 8), 3, (2, 3)),
        ((9, 6, 4, 9, 6), 3, (2, 3, 4)),
        ((9, 6, 9, 6), 2, (1, 2)),
        ((2, 3, 4, 5), 2, (0, 1, 2, 3, 4)),
        ((9, 9, 9), 0, (0, 1, 2, 3)),
        ((9, 9, 9), 3, (0, 1, 2, 3)),
    ])
    def test_shortest_matching_tt_bonds(self, ranks, center, sites):
        assert _spectral_block(ranks, center) == sites


class TestCentralBlockSelector:  # MARK: TestCentralBlockSelector

    def test_grows_bilaterally_from_center(self):
        selection = _select_block(
            in_dim := (2, 2, 2, 2, 2),
            rank=2,
            center=2)

        assert selection['sites'] == (1, 2, 3)
        assert selection['in_dim'] == in_dim[1:4]
        assert selection['growth'] == ((2, 2), (2, 3), (1, 3))
        assert selection['input_capacity'] == 8
        assert selection['required_input_capacity'] == 5
        assert selection['feasible']
        assert selection['boundary'] is None

    @pytest.mark.parametrize(
        ('center', 'sites', 'boundary'),
        [(0, (0, 1, 2), 'left'), (4, (2, 3, 4), 'right')])
    def test_selects_blocks_at_both_boundaries(self, center, sites, boundary):
        selection = _select_block(
            (2, 2, 2, 2, 2), rank=2, center=center)

        assert selection['sites'] == sites
        assert selection['feasible']
        assert selection['boundary'] == boundary

    def test_provider_object_and_heterogeneous_right_link_caps(self):
        class Provider:
            in_dim = (2, 3, 2, 4)

        selection = _select_block(
            Provider(), rank=(2, 3, 4, 5), center=1)

        assert selection['sites'] == (1, 2, 3)
        assert selection['left_rank_cap'] == 2
        assert selection['right_rank_cap'] == 5
        assert selection['boundary'] == 'right'

    def test_insufficient_input_capacity_is_a_result_not_an_exception(self):
        selection = _select_block(
            (2, 2, 2), rank=10, center=1)

        assert selection['sites'] == (0, 1, 2)
        assert not selection['feasible']
        assert selection['reason'] == 'available_sites_exhausted'
        assert selection['input_capacity'] == 8
        assert selection['required_input_capacity'] == 101
        assert selection['boundary'] == 'both'

    def test_bounds_limit_growth_and_preserve_global_boundary_semantics(self):
        selection = _select_block(
            (2, 2, 2, 2, 2), rank=3, center=1, bounds=(1, 3))

        assert selection['sites'] == (1, 2, 3)
        assert not selection['feasible']
        assert selection['boundary'] is None


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


class TestRingPartition:

    @pytest.mark.parametrize('count', [1, 2, 3, 7])
    @pytest.mark.parametrize('boundaries', [False, True])
    def test_prescribed_caps_and_merged_boundaries(self, count, boundaries):
        from tensorkrowch.decompositions.ring.tt2tr import _TTCoreProvider

        generator = torch.Generator().manual_seed(96)
        ring = tk.formats.TR([
            torch.randn(2, 4, 2, dtype=torch.complex128,
                        generator=generator) for _ in range(count)])
        provider = _TTCoreProvider(ring.to_tt())
        center = count // 2
        sites, left, right, ranks, info, rtol = _ring_partition(
            provider, 2, center, boundaries, None)
        assert center in sites
        assert sites == tuple(range(sites[0], sites[-1] + 1))
        assert ranks == (2,) * count
        assert info is None
        assert rtol is None
        assert not set(sites).intersection(left + right)
        assert not set(left).intersection(right)
        assert (sites[0] == 0) or (left[0] == 0)
        assert (sites[-1] == count - 1) or (right[-1] == count - 1)
        if not boundaries:
            assert len(left) <= 1
            assert len(right) <= 1

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_discovers_rank_two_ring_and_numerical_floor(self, dtype):
        from tensorkrowch.decompositions.ring.tt2tr import _TTCoreProvider

        generator = torch.Generator().manual_seed(96)
        ring = tk.formats.TR([
            torch.randn(2, 4, 2, dtype=dtype, generator=generator)
            for _ in range(7)])
        sites, left, right, ranks, info, rtol = _ring_partition(
            _TTCoreProvider(ring.to_tt()), None, 3, True, 1e-12)
        assert (sites, left, right) == ((3, 4), (0, 1), (5, 6))
        assert ranks == (2,) * 7
        assert info['cyclic_rank'] == 2
        assert rtol >= 1e-12

    @pytest.mark.parametrize('count', [1, 2, 3, 5, 7])
    @pytest.mark.parametrize('dimension', [2, 4, 5])
    @pytest.mark.parametrize('position', ['left', 'middle', 'right'])
    def test_discovery_and_prescribed_ranks_share_geometric_partition(
            self, count, dimension, position):
        from tensorkrowch.decompositions.ring.tt2tr import _TTCoreProvider

        generator = torch.Generator().manual_seed(318)
        ring = tk.formats.TR([
            torch.randn(2, dimension, 2, dtype=torch.complex128,
                        generator=generator) for _ in range(count)])
        provider = _TTCoreProvider(ring.to_tt())
        centers = {'left': 0, 'middle': count // 2, 'right': count - 1}
        center = centers[position]
        known = _ring_partition(provider, 2, center, True, 1e-12)
        discovered = _ring_partition(provider, None, center, True, 1e-12)

        assert discovered[:3] == known[:3]
        sites, left, right = discovered[:3]
        assert not set(sites).intersection(left + right)
        assert not set(left).intersection(right)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('dimension', [3, 5])
    def test_full_physical_rank_does_not_expand_selected_block(
            self, dtype, dimension, monkeypatch):
        from tensorkrowch.decompositions.ring.tt2tr import _TTCoreProvider

        generator = torch.Generator().manual_seed(317)
        ranks = (3,) * 6 + (1,)
        ring = tk.formats.TR([
            torch.randn(ranks[site - 1], dimension, ranks[site], dtype=dtype,
                        generator=generator) for site in range(7)])
        provider = _TTCoreProvider(ring.to_tt())
        requested = []
        local_target = provider.local_target

        def record_target(sites, context):
            requested.append(sites)
            return local_target(sites, context)

        monkeypatch.setattr(provider, 'local_target', record_target)
        sites, left, right, ranks, info = _discovery_partition(
            provider, 3, 1e-12)

        # Provisional cyclic rank 2 gives TR caps 2. The first capacity > 4
        # has full physical rank, and SVD then finds cyclic rank 1.
        expected = ((3, 4), (0, 1), (5, 6)) if dimension == 3 else \
            ((3,), (0,), (6,))
        assert requested == [expected[0]]
        assert (sites, left, right) == expected
        assert info == {'left_dim': 3, 'right_dim': 3,
                        'physical_rank': dimension ** len(sites),
                        'cyclic_rank': 1}
        assert ranks == (3,) * 6 + (1,)

    def test_low_numerical_rank_does_not_replace_geometric_capacity(self):
        from tensorkrowch.decompositions.ring.tt2tr import _TTCoreProvider

        # Redundant TT bonds retain dimension 4, although every support is 1.
        tt = tk.formats.TT([
            torch.ones(4, 4, dtype=torch.float64),
            *[torch.ones(4, 4, 4, dtype=torch.float64) for _ in range(5)],
            torch.ones(4, 4, dtype=torch.float64)])
        sites, left, right, _, info = _discovery_partition(
            _TTCoreProvider(tt), 3, 1e-12)

        # One site's capacity equals 2*2, so geometry still requires growth.
        assert (sites, left, right) == ((3, 4), (0, 1), (5, 6))
        assert info['physical_rank'] == 1
