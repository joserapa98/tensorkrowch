"""Tests for reusable TT ALS environment caches."""

from math import prod

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.results import (TRDecomposition,
                                                 TTDecomposition)

from tests.decompositions.als._oracles import (absorb_right_qr,
                                               build_tr_environment,
                                               contract_tr_dense,
                                               contract_tt_dense,
                                               dense_local_design,
                                               direct_environment_slices,
                                               make_tr_cores,
                                               make_tt_cores,
                                               observed_error,
                                               reference_tr_sweep,
                                               sampled_rows,
                                               solve_local_core)


def _empty_update():
    """Returns an atomic commit that only advances a fixed site."""
    return dict(zip((), ()))


def _prepared_local(cores, versions, order, site, samples=None,
                    renormalize=False):
    """Builds one fresh local environment for comparison."""
    cache = tk.decompositions.TTEnvironmentCache(
        cores, core_versions=versions, renormalize=renormalize)
    cache.prepare_sweep(order, samples=samples)
    for previous_site in order:
        cache.design(previous_site)
        environment = cache
        if previous_site == site:
            return environment
        cache.update(_empty_update())
    raise RuntimeError('Requested site is not in the sweep')


class TestTTEnvironmentCache:  # MARK: TestTTEnvironmentCache

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    def test_exact_design_matches_dense_oracle(self, dtype, direction):
        cores = make_tt_cores(
            in_dim=(2, 3, 2, 2),
            rank=(2, 3, 2),
            dtype=dtype,
            generator=torch.Generator().manual_seed(40))
        order = tuple(range(len(cores)))
        if direction == 'reverse':
            order = tuple(reversed(order))
        cache = tk.decompositions.TTEnvironmentCache(cores)
        cache.prepare_sweep(order)

        for site in order:
            cache.design(site)
            local = cache
            expected = dense_local_design(cores, site, topology='tt')
            assert torch.allclose(
                local.matrix, expected, rtol=1e-12, atol=1e-12)
            cache.update(_empty_update())

    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    def test_sampled_slices_match_selected_dense_rows(self, direction):
        cores = make_tt_cores(
            in_dim=(2, 3, 2),
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
            cache.design(site)
            local = cache
            expected = dense_local_design(
                cores, site, topology='tt').index_select(0, flat_ids)
            assert torch.allclose(local.matrix, expected)
            cache.update(_empty_update())

    def test_local_commit_grows_prefix_from_updated_core(self):
        cores = make_tt_cores(generator=torch.Generator().manual_seed(42))
        cache = tk.decompositions.TTEnvironmentCache(cores)
        order = tuple(range(len(cores)))
        cache.prepare_sweep(order)
        cache.design(0)
        new_core = cores[0] + 0.1

        cache.update(dict(zip((0,), (new_core,))))
        cache.design(1)
        local = cache
        expected_cores = [new_core, *cores[1:]]
        expected = _prepared_local(
            expected_cores, (1, 0, 0), order, site=1)

        assert torch.allclose(local.matrix, expected.matrix)

    def test_qr_update_of_current_and_neighbour_is_atomic(self):
        cores = make_tt_cores(
            in_dim=(2, 3, 2),
            rank=(4, 3),
            generator=torch.Generator().manual_seed(43))
        gauged, applied = absorb_right_qr(cores, site=0)
        assert applied
        cache = tk.decompositions.TTEnvironmentCache(cores)
        order = tuple(range(len(cores)))
        cache.prepare_sweep(order)
        cache.design(0)

        cache.update(dict(zip((0, 1), (gauged[0], gauged[1]))))
        cache.design(1)
        local = cache
        expected = _prepared_local(
            gauged, (1, 1, 0), order, site=1)

        assert torch.allclose(contract_tt_dense(cores),
                              contract_tt_dense(gauged))
        assert torch.allclose(local.matrix, expected.matrix)

    def test_forward_far_normalization_rebuilds_affected_suffixes(self):
        cores = make_tt_cores(
            in_dim=(2, 2, 2, 2),
            rank=(2, 2, 2),
            generator=torch.Generator().manual_seed(44))
        updated = list(cores)
        updated[0] = updated[0] / 2
        updated[2] = updated[2] * 2
        cache = tk.decompositions.TTEnvironmentCache(cores)
        order = tuple(range(len(cores)))
        cache.prepare_sweep(order)
        cache.design(0)

        cache.update(dict(zip((0, 2), (updated[0], updated[2]))))
        cache.design(1)
        local = cache
        expected = _prepared_local(
            updated, (1, 0, 1, 0), order, site=1)

        assert torch.allclose(contract_tt_dense(cores),
                              contract_tt_dense(updated))
        assert torch.allclose(local.matrix, expected.matrix)

    def test_reverse_far_normalization_rebuilds_affected_prefixes(self):
        cores = make_tt_cores(
            in_dim=(2, 2, 2, 2),
            rank=(2, 2, 2),
            generator=torch.Generator().manual_seed(45))
        updated = list(cores)
        updated[3] = updated[3] / 3
        updated[1] = updated[1] * 3
        cache = tk.decompositions.TTEnvironmentCache(cores)
        order = tuple(reversed(range(len(cores))))
        cache.prepare_sweep(order)
        cache.design(3)

        cache.update(dict(zip((3, 1), (updated[3], updated[1]))))
        cache.design(2)
        local = cache
        expected = _prepared_local(
            updated, (0, 1, 0, 1), order, site=2)

        assert torch.allclose(contract_tt_dense(cores),
                              contract_tt_dense(updated))
        assert torch.allclose(local.matrix, expected.matrix)

    def test_incompatible_update_fails_before_mutating_cache(self):
        cores = make_tt_cores(generator=torch.Generator().manual_seed(46))
        cache = tk.decompositions.TTEnvironmentCache(cores)
        cache.prepare_sweep(range(len(cores)))
        cache.design(0)
        invalid = torch.randn(
            1,
            cores[0].shape[1],
            cores[0].shape[-1] + 1,
            dtype=cores[0].dtype)

        with pytest.raises(ValueError, match='Adjacent TT ranks'):
            cache.update(dict(zip((0,), (invalid,))))

        assert cache.core_versions == (0, 0, 0)
        assert all(before is after
                   for before, after in zip(cores, cache.cores))
        cache.update(_empty_update())

    def test_changing_direction_after_a_sweep_rebuilds_the_mirror_cache(self):
        cores = make_tt_cores(generator=torch.Generator().manual_seed(47))
        cache = tk.decompositions.TTEnvironmentCache(cores)
        forward = tuple(range(len(cores)))
        reverse = tuple(reversed(forward))
        cache.prepare_sweep(forward)
        for site in forward:
            cache.design(site)
            cache.update(_empty_update())

        cache.prepare_sweep(reverse)
        for site in reverse:
            cache.design(site)
            local = cache
            assert torch.allclose(
                local.matrix, dense_local_design(cores, site, 'tt'))
            cache.update(_empty_update())

    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    def test_standard_zip_up_contracts_each_core_once_per_side(
            self, direction):
        cores = make_tt_cores(
            in_dim=(2, 2, 2, 2),
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
            cache.design(site)
            cache.update(_empty_update())

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
        cache.design(0)
        first_generation = cache._sample_generation
        cache.invalidate('sample_refresh')

        with pytest.raises(RuntimeError, match='sample_refresh'):
            cache.design(0)
        cache.prepare_sweep(order, second_samples)
        cache.design(0)
        second_generation = cache._sample_generation

        assert first_generation == 0
        assert second_generation == 1

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
            system_scaling=True).solve(direct.matrix, target)
        normalized_solution, _ = tk.decompositions.LeastSquaresSolver(
            l2_reg=(regularization * (-2 * normalized.log_scale).exp()).item(),
            column_scaling=False,
            system_scaling=True).solve(
                normalized.matrix, (target * (-normalized.log_scale).exp()))

        assert normalized.log_scale > 0
        assert torch.allclose(
            normalized_solution, direct_solution, rtol=1e-10, atol=1e-10)


class TestDenseALSOracles:  # MARK: TestDenseALSOracles

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_tt_and_tr_contractions_match_results(self, dtype):
        generator = torch.Generator().manual_seed(0)
        tt_cores = make_tt_cores(dtype=dtype, generator=generator)
        tr_cores = make_tr_cores(dtype=dtype, generator=generator)

        tt_result_cores = [tt_cores[0].squeeze(0)]
        tt_result_cores.extend(tt_cores[1:-1])
        tt_result_cores.append(tt_cores[-1].squeeze(-1))

        assert torch.allclose(
            contract_tt_dense(tt_cores),
            TTDecomposition(tt_result_cores).contract_dense())
        assert torch.allclose(
            contract_tr_dense(tr_cores),
            TRDecomposition(tr_cores).contract_dense())

    def test_heterogeneous_tr_environment_matches_direct_products(self):
        cores = make_tr_cores(
            in_dim=(2, 3, 2, 2),
            rank=(2, 3, 2, 4),
            generator=torch.Generator().manual_seed(1))

        for site in range(len(cores)):
            environment = build_tr_environment(cores, site)
            assert environment.shape[0] == cores[site].shape[-1]
            assert environment.shape[-1] == cores[site].shape[0]
            for configuration, expected in direct_environment_slices(
                    cores, site):
                index = (slice(None), *configuration, slice(None))
                assert torch.allclose(environment[index], expected)

    @pytest.mark.parametrize('topology', ['tt', 'tr'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_local_design_reconstructs_original_tensor(self,
                                                       topology,
                                                       dtype):
        generator = torch.Generator().manual_seed(2)
        if topology == 'tt':
            cores = make_tt_cores(dtype=dtype, generator=generator)
            contract = contract_tt_dense
        else:
            cores = make_tr_cores(dtype=dtype, generator=generator)
            contract = contract_tr_dense

        for site, core in enumerate(cores):
            design = dense_local_design(cores, site, topology)
            actual = design @ core.reshape(-1)
            assert torch.allclose(
                actual.reshape_as(contract(cores)),
                contract(cores),
                rtol=1e-12,
                atol=1e-12)


class TestLegacyTRALSCharacterization:  # MARK: TestLegacyTRALSCharacterization

    def test_fixed_cores_are_preserved_bit_for_bit(self):
        generator = torch.Generator().manual_seed(3)
        target_cores = make_tr_cores(generator=generator)
        target = contract_tr_dense(target_cores)
        initial = [core.clone() for core in target_cores]
        initial[1] = torch.randn(
            initial[1].shape, dtype=initial[1].dtype, generator=generator)
        initial[3] = torch.randn(
            initial[3].shape, dtype=initial[3].dtype, generator=generator)
        fixed_sites = (0, 2)

        updated = reference_tr_sweep(
            initial, target, fixed_sites=fixed_sites, qr=True)

        for site in fixed_sites:
            assert torch.equal(updated[site], initial[site])
        assert torch.linalg.vector_norm(
            contract_tr_dense(updated) - target) <= torch.linalg.vector_norm(
                contract_tr_dense(initial) - target)

    def test_qr_is_absorbed_only_into_a_trainable_neighbour(self):
        cores = make_tr_cores(generator=torch.Generator().manual_seed(4))
        target = contract_tr_dense(cores)

        gauged, applied = absorb_right_qr(cores, site=0)
        q_matrix = gauged[0].reshape(-1, gauged[0].shape[-1])

        assert applied
        assert torch.allclose(
            q_matrix.mH @ q_matrix,
            torch.eye(q_matrix.shape[1], dtype=q_matrix.dtype),
            rtol=1e-12,
            atol=1e-12)
        assert torch.allclose(contract_tr_dense(gauged), target)

        skipped, applied = absorb_right_qr(
            cores, site=0, fixed_sites=(1,))
        assert not applied
        assert all(torch.equal(before, after)
                   for before, after in zip(cores, skipped))

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_zero_target_local_solve_is_finite(self, dtype):
        cores = make_tr_cores(
            dtype=dtype, generator=torch.Generator().manual_seed(5))
        target = torch.zeros_like(contract_tr_dense(cores))

        core = solve_local_core(cores, target, site=1, topology='tr')

        assert torch.isfinite(core).all()
        updated = list(cores)
        updated[1] = core
        assert torch.allclose(contract_tr_dense(updated), target)

    def test_rank_deficient_local_system_has_a_finite_solution(self):
        cores = make_tr_cores(generator=torch.Generator().manual_seed(6))
        cores[2] = torch.zeros_like(cores[2])
        target = torch.randn(
            contract_tr_dense(cores).shape,
            dtype=cores[0].dtype,
            generator=torch.Generator().manual_seed(7))
        design = dense_local_design(cores, site=0, topology='tr')

        core = solve_local_core(cores, target, site=0, topology='tr')

        assert torch.linalg.matrix_rank(design) < design.shape[1]
        assert torch.isfinite(core).all()

    def test_sampled_rows_are_deterministic_with_generator(self):
        first = sampled_rows(
            48, 20, generator=torch.Generator().manual_seed(8))
        second = sampled_rows(
            48, 20, generator=torch.Generator().manual_seed(8))
        different = sampled_rows(
            48, 20, generator=torch.Generator().manual_seed(9))

        assert torch.equal(first, second)
        assert not torch.equal(first, different)

    def test_completion_rows_remain_fixed_across_sweeps(self):
        generator = torch.Generator().manual_seed(10)
        target_cores = make_tr_cores(generator=generator)
        target = contract_tr_dense(target_cores)
        initial = make_tr_cores(generator=generator)
        rows = torch.tensor([0, 3, 7, 12, 18, 23])
        original_rows = rows.clone()
        errors = [observed_error(initial, target, rows)]

        updated = initial
        for _ in range(2):
            updated = reference_tr_sweep(updated, target, rows=rows)
            errors.append(observed_error(updated, target, rows))

        assert torch.equal(rows, original_rows)
        assert errors[1] <= errors[0]
        assert errors[2] <= errors[1] + 1e-12


def _tr_environments_empty_update():
    """Advances a fixed site without changing a core."""
    return dict(zip((), ()))


class TestDirectTREnvironment:  # MARK: TestDirectTREnvironment

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_matches_dense_oracle(self, dtype):
        cores = make_tr_cores(
            in_dim=(2, 3, 2),
            rank=(2, 3, 2),
            dtype=dtype,
            generator=torch.Generator().manual_seed(90))
        direct = tk.decompositions.DirectTREnvironment(cores)

        for site in range(len(cores)):
            direct.design(site)
            local = direct
            expected = dense_local_design(cores, site, topology='tr')
            assert torch.allclose(
                local.matrix, expected, rtol=1e-12, atol=1e-12)
            assert not local.sampled

    def test_sampled_rows_preserve_order_and_duplicates(self):
        cores = make_tr_cores(generator=torch.Generator().manual_seed(91))
        ids = torch.tensor([7, 0, 7, 19])
        probabilities = torch.full((4,), 1 / 24)
        samples = tk.decompositions.SampleBatch(
            ids=ids,
            probabilities=probabilities,
            weights=(4 * probabilities).rsqrt(),
            generation=3)
        local = tk.decompositions.DirectTREnvironment(cores)
        local.design(1, samples=samples)
        expected = dense_local_design(
            cores, 1, topology='tr').index_select(0, ids)

        assert torch.allclose(local.matrix, expected)
        assert local.sampled


class TestTRSegmentEnvironmentCache:  # MARK: TestTRSegmentEnvironmentCache

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    @pytest.mark.parametrize('n_segments', [1, 2, 3, 5])
    def test_exact_design_matches_direct_oracle(
            self, dtype, direction, n_segments):
        cores = make_tr_cores(
            in_dim=(2, 2, 3, 2, 2),
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
            cache.design(site)
            local = cache
            expected = dense_local_design(cores, site, topology='tr')
            assert torch.allclose(
                local.matrix, expected, rtol=2e-12, atol=2e-12)
            cache.update(_tr_environments_empty_update())

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
            cache.design(site)
            local = cache
            expected = dense_local_design(
                cores, site, topology='tr').index_select(0, ids)
            assert torch.allclose(local.matrix, expected)
            assert local.sampled
            cache.update(_tr_environments_empty_update())

    def test_cross_segment_gauge_update_is_atomic(self):
        cores = make_tr_cores(
            in_dim=(2, 2, 2, 2),
            rank=(2, 3, 3, 2),
            generator=torch.Generator().manual_seed(94))
        gauged, applied = absorb_right_qr(cores, site=1)
        assert applied
        cache = tk.decompositions.TRSegmentEnvironmentCache(
            cores, n_segments=2)
        cache.prepare_sweep(range(len(cores)))
        cache.design(0)
        cache.update(_tr_environments_empty_update())
        cache.design(1)
        cache.update(dict(zip((1, 2), (gauged[1], gauged[2]))))

        cache.design(2)
        local = cache
        expected = tk.decompositions.DirectTREnvironment(gauged)
        expected.design(2)
        assert torch.allclose(local.matrix, expected.matrix)
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
            cache.design(site)
            local = cache
            direct = tk.decompositions.DirectTREnvironment(cache.cores)
            direct.design(site)
            scale = local.log_scale.exp().to(local.matrix.dtype)
            assert torch.allclose(
                local.matrix * scale,
                direct.matrix,
                rtol=2e-12,
                atol=2e-12)
            assert torch.allclose(
                (target * (-local.log_scale).exp()) * scale,
                target,
                rtol=2e-12,
                atol=2e-12)
            cache.update(_tr_environments_empty_update())

    def test_invalid_cross_segment_update_does_not_mutate_cache(self):
        cores = make_tr_cores(
            generator=torch.Generator().manual_seed(96))
        cache = tk.decompositions.TRSegmentEnvironmentCache(
            cores, n_segments=3)
        cache.prepare_sweep(range(len(cores)))
        cache.design(0)

        invalid = torch.randn(
            cores[2].shape[0] + 1,
            cores[2].shape[1],
            cores[2].shape[2],
            dtype=cores[2].dtype)
        with pytest.raises(ValueError, match='Adjacent TR ranks'):
            cache.update(dict(zip((2,), (invalid,))))

        assert cache.core_versions == (0, 0, 0, 0)
        assert all(before is after
                   for before, after in zip(cores, cache.cores))
        cache.update(_tr_environments_empty_update())

    def test_segment_partition_is_balanced_and_complete(self):
        cores = make_tr_cores(
            in_dim=(2,) * 7,
            rank=(2,) * 7,
            generator=torch.Generator().manual_seed(97))
        cache = tk.decompositions.TRSegmentEnvironmentCache(
            cores, n_segments=3)

        assert cache.segments == ((0, 3), (3, 5), (5, 7))

    @pytest.mark.parametrize('direction', ['forward', 'reverse'])
    @pytest.mark.parametrize('n_segments', [1, 2, 3, 6])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_nonadjacent_normalization_rebuilds_remaining_products(
            self, direction, n_segments, dtype):
        cores = make_tr_cores(
            in_dim=(2,) * 6, rank=(2, 3, 2, 3, 2, 2), dtype=dtype,
            generator=torch.Generator().manual_seed(130))
        order = tuple(range(6)) if direction == 'forward' else tuple(
            reversed(range(6)))
        active = 1 if direction == 'forward' else 4
        receiver = 4 if direction == 'forward' else 1
        cache = tk.decompositions.TRSegmentEnvironmentCache(
            cores, n_segments=n_segments, renormalize=True)
        cache.prepare_sweep(order)
        updated = list(cores)
        for site in order:
            cache.design(site)
            local = cache
            expected = tk.decompositions.DirectTREnvironment(updated)
            expected.design(site)
            assert torch.allclose(
                local.matrix * local.log_scale.exp(), expected.matrix,
                rtol=1e-11, atol=1e-11)
            if site == active:
                updated[active] = updated[active] / 3
                updated[receiver] = updated[receiver] * 3
                cache.update({active: updated[active],
                              receiver: updated[receiver]})
            else:
                cache.update(_empty_update())
        assert torch.allclose(contract_tr_dense(
            updated), contract_tr_dense(cores))


class TestEnvironmentContracts:  # MARK: TestEnvironmentContracts

    @pytest.mark.parametrize('topology', ['tt', 'tr'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_one_site_design_and_scale_methods(self, topology, dtype):
        rank = 1 if topology == 'tt' else 2
        core = torch.randn(rank, 3, rank, dtype=dtype)
        cache_type = tk.decompositions.TTEnvironmentCache if topology == 'tt' \
            else tk.decompositions.TRSegmentEnvironmentCache
        cache = cache_type([core], renormalize=True)
        assert cache.in_dim == (3,)
        assert cache.core_versions == (0,)
        cache.prepare_sweep((0,))
        cache.design(0)
        local = cache
        expected = dense_local_design([core], 0, topology)
        assert torch.allclose(local.matrix * local.log_scale.exp(), expected)
        target = expected @ core.reshape(-1)
        assert torch.allclose(
            (target * (-local.log_scale).exp()) * local.log_scale.exp(),
            target)
        assert (0.5 * (-2 * local.log_scale).exp()).item() == pytest.approx(
            0.5 * torch.exp(-2 * local.log_scale).item())
        cache.update(_empty_update())
        with pytest.raises(RuntimeError, match='complete'):
            cache.design(0)

    @pytest.mark.parametrize('cache_type', [
        tk.decompositions.TTEnvironmentCache,
        tk.decompositions.TRSegmentEnvironmentCache,
    ])
    def test_cache_state_and_invalid_arguments(self, cache_type):
        cores = [torch.ones(1, 2, 1), torch.ones(1, 3, 1)]
        with pytest.raises(TypeError, match='sequence'):
            cache_type(iter(cores))
        with pytest.raises(ValueError, match='at least one'):
            cache_type([])
        with pytest.raises(ValueError, match='version'):
            cache_type(cores, core_versions=(0,))
        with pytest.raises(TypeError, match='renormalize'):
            cache_type(cores, renormalize=1)
        cache = cache_type(cores)
        with pytest.raises(RuntimeError, match='not prepared'):
            cache.design(0)
        with pytest.raises(ValueError, match='order'):
            cache.prepare_sweep((0,))
        with pytest.raises(TypeError, match='integers'):
            cache.prepare_sweep((False, True))
        cache.prepare_sweep((0, 1))
        with pytest.raises(TypeError, match='int type'):
            cache.design(False)
        with pytest.raises(ValueError, match='next site'):
            cache.design(1)
        cache.design(0)
        with pytest.raises(RuntimeError, match='committed first'):
            cache.prepare_sweep((0, 1))
        cache.invalidate('new samples')
        with pytest.raises(RuntimeError, match='new samples'):
            cache.design(0)

    @pytest.mark.parametrize('cache_type', [
        tk.decompositions.TTEnvironmentCache,
        tk.decompositions.TRSegmentEnvironmentCache,
    ])
    @pytest.mark.parametrize('change', ['dimension', 'dtype', 'site'])
    def test_invalid_commits_preserve_state(self, cache_type, change):
        cores = [torch.ones(1, 2, 1), torch.ones(1, 3, 1)]
        cache = cache_type(cores)
        cache.prepare_sweep((0, 1))
        cache.design(0)
        core = torch.ones(1, 3, 1) if change == 'dimension' else cores[0]
        if change == 'dtype':
            core = core.double()
        site = 3 if change == 'site' else 0
        with pytest.raises(ValueError):
            cache.update({site: core})
        assert cache.core_versions == (0, 0)
        assert all(original is current for original,
                   current in zip(cores, cache.cores))
        cache.update(_empty_update())


@pytest.mark.parametrize('topology', ['tt', 'tr', 'direct_tr'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('dimensions', [(3,), (2, 3, 1, 4), (2, 3, 2, 2, 3)])
@pytest.mark.parametrize('reverse', [False, True])
def test_compact_environments_match_expanded_physical_blocks(
        topology, dtype, dimensions, reverse):
    generator = torch.Generator().manual_seed(320)
    n_sites = len(dimensions)
    if topology == 'tt':
        cores = make_tt_cores(dimensions, (2,) * (n_sites - 1), dtype,
                              generator)
        cls = tk.decompositions.TTEnvironmentCache
    else:
        cores = make_tr_cores(dimensions, (2,) * n_sites, dtype, generator)
        cls = (tk.decompositions.DirectTREnvironment if topology == 'direct_tr'
               else tk.decompositions.TRSegmentEnvironmentCache)
    order = tuple(range(n_sites))
    if reverse:
        order = tuple(reversed(order))
    compact, expanded = cls(cores), cls(cores)
    if topology != 'direct_tr':
        compact.prepare_sweep(order)
        expanded.prepare_sweep(order)
    for site in order:
        matrix = compact.design(site, compact=True)
        full = expanded.design(site)
        left = prod(dimensions[:site])
        right = prod(dimensions[site + 1:])
        expected = full.reshape(
            left, dimensions[site], right,
            cores[site].shape[0], dimensions[site], cores[site].shape[2]
        )[:, 0, :, :, 0, :].reshape(left * right, -1)
        torch.testing.assert_close(matrix, expected, atol=1e-11, rtol=1e-11)
        assert full.numel() == dimensions[site] ** 2 * matrix.numel()
        if topology != 'direct_tr':
            compact.update({})
            expanded.update({})


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('site', [0, 1, 3])
def test_compact_sampled_designs_share_complete_fibers(dtype, site):
    from types import SimpleNamespace
    from tensorkrowch.decompositions.als.tt import _direct_sampled_tt_design
    dimensions = (2, 3, 2, 4)
    generator = torch.Generator().manual_seed(321)
    for topology in ['tt', 'tr']:
        cores = (make_tt_cores(dimensions, (2,) * 3, dtype, generator)
                 if topology == 'tt' else make_tr_cores(
                     dimensions, (2,) * 4, dtype, generator))
        owner = SimpleNamespace(cores=cores, core_versions=(0,) * 4)
        batch = tk.decompositions.UniformRows().draw(
            owner, site, 12, generator)
        indices = torch.stack(torch.unravel_index(batch.ids, dimensions), 1)
        if topology == 'tt':
            matrix, scale = _direct_sampled_tt_design(
                cores, site, indices, True, compact=True)
            full, full_scale = _direct_sampled_tt_design(
                cores, site, indices, True)
        else:
            compact = tk.decompositions.DirectTREnvironment(cores)
            expanded = tk.decompositions.DirectTREnvironment(cores)
            matrix = compact.design(site, batch, compact=True)
            full = expanded.design(site, batch)
            scale, full_scale = compact.log_scale, expanded.log_scale
        dimension = dimensions[site]
        expected = full.reshape(12, dimension, cores[site].shape[0],
                                dimension, cores[site].shape[2])[
                                    :, 0, :, 0, :].reshape(12, -1)
        torch.testing.assert_close(matrix * scale.exp(),
                                   expected * full_scale.exp())


@pytest.mark.parametrize('cls', [tk.decompositions.TTEnvironmentCache,
                                tk.decompositions.TRSegmentEnvironmentCache])
def test_compact_cache_supports_observed_rows_and_boolean_control(cls):
    cores = [torch.ones(1, 2, 1), torch.ones(1, 3, 1)]
    cache = cls(cores)
    samples = tk.decompositions.ConfigurationBatch(torch.tensor([[0, 1]]))
    cache.prepare_sweep((0, 1), samples)
    with pytest.raises(TypeError, match='compact'):
        cache.design(0, compact=1)
    torch.testing.assert_close(cache.design(0, compact=True), torch.ones(1, 1))


def test_direct_compact_design_rejects_incomplete_or_misaligned_fibers():
    cores = [torch.ones(1, 2, 1), torch.ones(1, 3, 1)]
    for values in [torch.tensor([[0, 0]]), torch.tensor([[1, 0], [0, 0]]),
                   torch.tensor([[0, 0], [1, 1]])]:
        with pytest.raises(ValueError, match='complete input fibers'):
            tk.decompositions.DirectTREnvironment(cores).design(
                0, tk.decompositions.ConfigurationBatch(values), compact=True)
