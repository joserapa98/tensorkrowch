"""Tests for reusable TT ALS environment caches."""

import pytest

import torch
import tensorkrowch as tk

from tests.decompositions.als._oracles import (absorb_right_qr,
                                               contract_tt_dense,
                                               dense_local_design,
                                               make_tt_cores)


def _empty_update():
    """Returns an atomic commit that only advances a fixed site."""
    return tk.decompositions.CoreUpdateSet((), (), (), reason='fixed')


def _prepared_local(cores, versions, order, site, samples=None,
                    renormalize=False):
    """Builds one fresh local environment for comparison."""
    cache = tk.decompositions.TTEnvironmentCache(
        cores, core_versions=versions, renormalize=renormalize)
    cache.prepare_sweep(order, samples=samples)
    for previous_site in order:
        environment = cache.local_environment(previous_site)
        if previous_site == site:
            return environment
        cache.commit(_empty_update())
    raise RuntimeError('Requested site is not in the sweep')


class TestTTEnvironmentCache:  # MARK: TestTTEnvironmentCache

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    def test_exact_design_matches_dense_oracle(self, dtype, direction):
        cores = make_tt_cores(
            input_dim=(2, 3, 2, 2),
            rank=(2, 3, 2),
            dtype=dtype,
            generator=torch.Generator().manual_seed(40))
        order = tuple(range(len(cores)))
        if direction == 'reverse':
            order = tuple(reversed(order))
        cache = tk.decompositions.TTEnvironmentCache(cores)
        cache.prepare_sweep(order)

        for site in order:
            local = cache.local_environment(site)
            expected = dense_local_design(cores, site, topology='tt')
            assert torch.allclose(
                local.design(), expected, rtol=1e-12, atol=1e-12)
            assert local.key.direction == direction
            assert local.key.device == cores[0].device
            assert local.key.dtype == dtype
            cache.commit(_empty_update())

    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    def test_sampled_slices_match_selected_dense_rows(self, direction):
        cores = make_tt_cores(
            input_dim=(2, 3, 2),
            rank=(2, 3),
            generator=torch.Generator().manual_seed(41))
        flat_ids = torch.tensor([0, 3, 7, 7, 11])
        probabilities = torch.full((5,), 1 / 12)
        samples = tk.decompositions.SampleBatch(
            ids=flat_ids,
            probabilities=probabilities,
            weights=(5 * probabilities).rsqrt(),
            generation=4)
        order = tuple(range(len(cores)))
        if direction == 'reverse':
            order = tuple(reversed(order))
        cache = tk.decompositions.TTEnvironmentCache(cores)
        cache.prepare_sweep(order, samples=samples)

        for site in order:
            local = cache.local_environment(site)
            expected = dense_local_design(
                cores, site, topology='tt').index_select(0, flat_ids)
            assert torch.allclose(local.design(), expected)
            assert local.key.sample_generation == 4
            cache.commit(_empty_update())

    def test_local_commit_grows_prefix_from_updated_core(self):
        cores = make_tt_cores(generator=torch.Generator().manual_seed(42))
        cache = tk.decompositions.TTEnvironmentCache(cores)
        order = tuple(range(len(cores)))
        cache.prepare_sweep(order)
        cache.local_environment(0)
        new_core = cores[0] + 0.1

        cache.commit(tk.decompositions.CoreUpdateSet(
            sites=(0,), cores=(new_core,), versions=(1,)))
        local = cache.local_environment(1)
        expected_cores = [new_core, *cores[1:]]
        expected = _prepared_local(
            expected_cores, (1, 0, 0), order, site=1)

        assert torch.allclose(local.design(), expected.design())
        assert local.key.dependency_versions[0] == (0, 1)

    def test_qr_update_of_current_and_neighbour_is_atomic(self):
        cores = make_tt_cores(
            input_dim=(2, 3, 2),
            rank=(4, 3),
            generator=torch.Generator().manual_seed(43))
        gauged, applied = absorb_right_qr(cores, site=0)
        assert applied
        cache = tk.decompositions.TTEnvironmentCache(cores)
        order = tuple(range(len(cores)))
        cache.prepare_sweep(order)
        cache.local_environment(0)

        cache.commit(tk.decompositions.CoreUpdateSet(
            sites=(0, 1),
            cores=(gauged[0], gauged[1]),
            versions=(1, 1),
            reason='qr'))
        local = cache.local_environment(1)
        expected = _prepared_local(
            gauged, (1, 1, 0), order, site=1)

        assert torch.allclose(contract_tt_dense(cores),
                              contract_tt_dense(gauged))
        assert torch.allclose(local.design(), expected.design())

    def test_forward_far_normalization_rebuilds_affected_suffixes(self):
        cores = make_tt_cores(
            input_dim=(2, 2, 2, 2),
            rank=(2, 2, 2),
            generator=torch.Generator().manual_seed(44))
        updated = list(cores)
        updated[0] = updated[0] / 2
        updated[2] = updated[2] * 2
        cache = tk.decompositions.TTEnvironmentCache(cores)
        order = tuple(range(len(cores)))
        cache.prepare_sweep(order)
        cache.local_environment(0)

        cache.commit(tk.decompositions.CoreUpdateSet(
            sites=(0, 2),
            cores=(updated[0], updated[2]),
            versions=(1, 1),
            reason='normalization'))
        local = cache.local_environment(1)
        expected = _prepared_local(
            updated, (1, 0, 1, 0), order, site=1)

        assert torch.allclose(contract_tt_dense(cores),
                              contract_tt_dense(updated))
        assert torch.allclose(local.design(), expected.design())

    def test_reverse_far_normalization_rebuilds_affected_prefixes(self):
        cores = make_tt_cores(
            input_dim=(2, 2, 2, 2),
            rank=(2, 2, 2),
            generator=torch.Generator().manual_seed(45))
        updated = list(cores)
        updated[3] = updated[3] / 3
        updated[1] = updated[1] * 3
        cache = tk.decompositions.TTEnvironmentCache(cores)
        order = tuple(reversed(range(len(cores))))
        cache.prepare_sweep(order)
        cache.local_environment(3)

        cache.commit(tk.decompositions.CoreUpdateSet(
            sites=(3, 1),
            cores=(updated[3], updated[1]),
            versions=(1, 1),
            reason='normalization'))
        local = cache.local_environment(2)
        expected = _prepared_local(
            updated, (0, 1, 0, 1), order, site=2)

        assert torch.allclose(contract_tt_dense(cores),
                              contract_tt_dense(updated))
        assert torch.allclose(local.design(), expected.design())

    def test_incompatible_update_fails_before_mutating_cache(self):
        cores = make_tt_cores(generator=torch.Generator().manual_seed(46))
        cache = tk.decompositions.TTEnvironmentCache(cores)
        cache.prepare_sweep(range(len(cores)))
        cache.local_environment(0)
        invalid = torch.randn(
            1,
            cores[0].shape[1],
            cores[0].shape[-1] + 1,
            dtype=cores[0].dtype)

        with pytest.raises(ValueError, match='Adjacent TT ranks'):
            cache.commit(tk.decompositions.CoreUpdateSet(
                sites=(0,), cores=(invalid,), versions=(1,)))

        assert cache.core_versions == (0, 0, 0)
        assert all(before is after
                   for before, after in zip(cores, cache.cores))
        cache.commit(_empty_update())

    def test_changing_direction_after_a_sweep_rebuilds_the_mirror_cache(self):
        cores = make_tt_cores(generator=torch.Generator().manual_seed(47))
        cache = tk.decompositions.TTEnvironmentCache(cores)
        forward = tuple(range(len(cores)))
        reverse = tuple(reversed(forward))
        cache.prepare_sweep(forward)
        for site in forward:
            cache.local_environment(site)
            cache.commit(_empty_update())

        cache.prepare_sweep(reverse)
        for site in reverse:
            local = cache.local_environment(site)
            assert torch.allclose(
                local.design(), dense_local_design(cores, site, 'tt'))
            cache.commit(_empty_update())

    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    def test_standard_zip_up_contracts_each_core_once_per_side(
            self, direction):
        cores = make_tt_cores(
            input_dim=(2, 2, 2, 2),
            rank=(2, 2, 2),
            generator=torch.Generator().manual_seed(50))
        cache = tk.decompositions.TTEnvironmentCache(cores)
        calls = {'left': 0, 'right': 0}
        original_left = cache._extend_left
        original_right = cache._extend_right

        def count_left(*args, **kwargs):
            calls['left'] += 1
            return original_left(*args, **kwargs)

        def count_right(*args, **kwargs):
            calls['right'] += 1
            return original_right(*args, **kwargs)

        cache._extend_left = count_left
        cache._extend_right = count_right
        order = tuple(range(len(cores)))
        if direction == 'reverse':
            order = tuple(reversed(order))
        cache.prepare_sweep(order)
        for site in order:
            cache.local_environment(site)
            cache.commit(_empty_update())

        assert calls == {'left': len(cores), 'right': len(cores)}

    def test_sample_generation_and_explicit_invalidation_change_keys(self):
        cores = make_tt_cores(generator=torch.Generator().manual_seed(48))
        probabilities = torch.full((3,), 1 / 12)
        first_samples = tk.decompositions.SampleBatch(
            ids=torch.tensor([0, 4, 9]),
            probabilities=probabilities,
            weights=(3 * probabilities).rsqrt(),
            generation=0)
        second_samples = tk.decompositions.SampleBatch(
            ids=torch.tensor([1, 5, 10]),
            probabilities=probabilities,
            weights=(3 * probabilities).rsqrt(),
            generation=1)
        cache = tk.decompositions.TTEnvironmentCache(cores)
        order = tuple(range(len(cores)))
        cache.prepare_sweep(order, first_samples)
        first_key = cache.local_environment(0).key
        cache.invalidate('sample_refresh')

        with pytest.raises(RuntimeError, match='sample_refresh'):
            cache.local_environment(0)
        cache.prepare_sweep(order, second_samples)
        second_key = cache.local_environment(0).key

        assert first_key.sample_generation == 0
        assert second_key.sample_generation == 1

    def test_global_log_normalization_preserves_regularized_solve(self):
        cores = [
            20 * core
            for core in make_tt_cores(
                generator=torch.Generator().manual_seed(49))
        ]
        site = 1
        order = tuple(range(len(cores)))
        direct = _prepared_local(
            cores, (0, 0, 0), order, site, renormalize=False)
        normalized = _prepared_local(
            cores, (0, 0, 0), order, site, renormalize=True)
        target = contract_tt_dense(cores).reshape(-1)
        regularization = 0.2

        direct_solution, _ = tk.decompositions.LeastSquaresSolver(
            l2_reg=regularization,
            column_scaling=False,
            system_scaling=True).solve(direct.design(), target)
        normalized_solution, _ = tk.decompositions.LeastSquaresSolver(
            l2_reg=normalized.scale_l2_reg(regularization),
            column_scaling=False,
            system_scaling=True).solve(
                normalized.design(), normalized.scale_target(target))

        assert normalized.log_scale > 0
        assert torch.allclose(
            normalized_solution, direct_solution, rtol=1e-10, atol=1e-10)
