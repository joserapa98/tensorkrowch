"""Tests for segmented tensor-ring ALS environment caches."""

import pytest

import torch
import tensorkrowch as tk

from tests.decompositions.als._oracles import (absorb_right_qr,
                                               dense_local_design,
                                               make_tr_cores)


def _empty_update():
    """Advances a fixed site without changing a core."""
    return tk.decompositions.CoreUpdateSet((), (), (), reason='fixed')


class TestDirectTREnvironment:  # MARK: TestDirectTREnvironment

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_matches_dense_oracle(self, dtype):
        cores = make_tr_cores(
            input_dim=(2, 3, 2),
            rank=(2, 3, 2),
            dtype=dtype,
            generator=torch.Generator().manual_seed(90))
        direct = tk.decompositions.DirectTREnvironment(cores)

        for site in range(len(cores)):
            local = direct.local_environment(site)
            expected = dense_local_design(cores, site, topology='tr')
            assert torch.allclose(
                local.design(), expected, rtol=1e-12, atol=1e-12)
            assert not local.sampled
            assert local.key.direction == 'direct'

    def test_sampled_rows_preserve_order_and_duplicates(self):
        cores = make_tr_cores(generator=torch.Generator().manual_seed(91))
        ids = torch.tensor([7, 0, 7, 19])
        probabilities = torch.full((4,), 1 / 24)
        samples = tk.decompositions.SampleBatch(
            ids=ids,
            probabilities=probabilities,
            weights=(4 * probabilities).rsqrt(),
            generation=3)
        local = tk.decompositions.DirectTREnvironment(
            cores).local_environment(1, samples=samples)
        expected = dense_local_design(
            cores, 1, topology='tr').index_select(0, ids)

        assert torch.allclose(local.design(), expected)
        assert local.sampled
        assert local.key.sample_generation == 3


class TestTRSegmentEnvironmentCache:  # MARK: TestTRSegmentEnvironmentCache

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    @pytest.mark.parametrize('n_segments', [1, 2, 3, 5])
    def test_exact_design_matches_direct_oracle(
            self, dtype, direction, n_segments):
        cores = make_tr_cores(
            input_dim=(2, 2, 3, 2, 2),
            rank=(2, 3, 2, 4, 2),
            dtype=dtype,
            generator=torch.Generator().manual_seed(92))
        order = tuple(range(len(cores)))
        if direction == 'reverse':
            order = tuple(reversed(order))
        cache = tk.decompositions.TRSegmentEnvironmentCache(
            cores, n_segments=n_segments)
        cache.prepare_sweep(order)

        for site in order:
            local = cache.local_environment(site)
            expected = dense_local_design(cores, site, topology='tr')
            assert torch.allclose(
                local.design(), expected, rtol=2e-12, atol=2e-12)
            assert local.key.direction == direction
            cache.commit(_empty_update())

    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    def test_sampled_slices_match_direct_oracle(self, direction):
        cores = make_tr_cores(
            generator=torch.Generator().manual_seed(93))
        ids = torch.tensor([0, 4, 11, 11, 22])
        probabilities = torch.full((5,), 1 / 24)
        samples = tk.decompositions.SampleBatch(
            ids=ids,
            probabilities=probabilities,
            weights=(5 * probabilities).rsqrt(),
            generation=6)
        order = tuple(range(len(cores)))
        if direction == 'reverse':
            order = tuple(reversed(order))
        cache = tk.decompositions.TRSegmentEnvironmentCache(
            cores, n_segments=3)
        cache.prepare_sweep(order, samples=samples)

        for site in order:
            local = cache.local_environment(site)
            expected = dense_local_design(
                cores, site, topology='tr').index_select(0, ids)
            assert torch.allclose(local.design(), expected)
            assert local.sampled
            assert local.key.sample_generation == 6
            cache.commit(_empty_update())

    def test_cross_segment_gauge_update_is_atomic(self):
        cores = make_tr_cores(
            input_dim=(2, 2, 2, 2),
            rank=(2, 3, 3, 2),
            generator=torch.Generator().manual_seed(94))
        gauged, applied = absorb_right_qr(cores, site=1)
        assert applied
        cache = tk.decompositions.TRSegmentEnvironmentCache(
            cores, n_segments=2)
        cache.prepare_sweep(range(len(cores)))
        cache.local_environment(0)
        cache.commit(_empty_update())
        cache.local_environment(1)
        cache.commit(tk.decompositions.CoreUpdateSet(
            sites=(1, 2),
            cores=(gauged[1], gauged[2]),
            versions=(1, 1),
            reason='qr'))

        local = cache.local_environment(2)
        expected = tk.decompositions.DirectTREnvironment(
            gauged).local_environment(2)
        assert torch.allclose(local.design(), expected.design())
        assert cache.core_versions == (0, 1, 1, 0)

    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    def test_renormalized_design_and_target_preserve_local_system(
            self, direction):
        cores = make_tr_cores(
            generator=torch.Generator().manual_seed(95))
        order = tuple(range(len(cores)))
        if direction == 'reverse':
            order = tuple(reversed(order))
        cache = tk.decompositions.TRSegmentEnvironmentCache(
            cores, n_segments=3, renormalize=True)
        cache.prepare_sweep(order)
        target = torch.randn(24, dtype=cores[0].dtype)

        for site in order:
            local = cache.local_environment(site)
            direct = tk.decompositions.DirectTREnvironment(
                cache.cores).local_environment(site)
            scale = local.log_scale.exp().to(local.design().dtype)
            assert torch.allclose(
                local.design() * scale,
                direct.design(),
                rtol=2e-12,
                atol=2e-12)
            assert torch.allclose(
                local.scale_target(target) * scale,
                target,
                rtol=2e-12,
                atol=2e-12)
            cache.commit(_empty_update())

    def test_invalid_cross_segment_update_does_not_mutate_cache(self):
        cores = make_tr_cores(
            generator=torch.Generator().manual_seed(96))
        cache = tk.decompositions.TRSegmentEnvironmentCache(
            cores, n_segments=3)
        cache.prepare_sweep(range(len(cores)))
        cache.local_environment(0)

        invalid = torch.randn(
            cores[2].shape[0] + 1,
            cores[2].shape[1],
            cores[2].shape[2],
            dtype=cores[2].dtype)
        with pytest.raises(ValueError, match='Adjacent TR ranks'):
            cache.commit(tk.decompositions.CoreUpdateSet(
                sites=(2,),
                cores=(invalid,),
                versions=(1,),
                reason='invalid_far_update'))

        assert cache.core_versions == (0, 0, 0, 0)
        assert all(before is after
                   for before, after in zip(cores, cache.cores))
        cache.commit(_empty_update())

    def test_segment_partition_is_balanced_and_complete(self):
        cores = make_tr_cores(
            input_dim=(2,) * 7,
            rank=(2,) * 7,
            generator=torch.Generator().manual_seed(97))
        cache = tk.decompositions.TRSegmentEnvironmentCache(
            cores, n_segments=3)

        assert cache.segments == ((0, 3), (3, 5), (5, 7))
