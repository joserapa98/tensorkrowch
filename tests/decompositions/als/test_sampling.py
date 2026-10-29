"""Tests for ALS row sampling and refresh semantics."""

from math import prod

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.als.sampling import _RowSamplingState

from tests.decompositions.als._oracles import dense_local_design, make_tr_cores


def _mixed_canonical_cores(tensor, site, rank=2):
    """Builds standard TT cores mixed-canonical around ``site``."""
    result = tk.decompositions.TTSVD(
        tensor, out_device=None).fit(rank=rank)
    cores = [result.cores[0].unsqueeze(0),
             *result.cores[1:-1],
             result.cores[-1].unsqueeze(-1)]
    for current in reversed(range(site + 1, len(cores))):
        core = cores[current]
        q, r = torch.linalg.qr(core.reshape(core.shape[0], -1).mH,
                               mode='reduced')
        cores[current] = q.mH.reshape_as(core)
        cores[current - 1] = torch.einsum(
            'apb,bc->apc', cores[current - 1], r.mH)
    return cores


class TestSamplingPolicy:  # MARK: TestSamplingPolicy
    @pytest.mark.parametrize(
        'method', ['uniform', 'leverage', 'exact_leverage']
    )
    def test_policy_selects_supported_topologies(self, method):
        policy = tk.decompositions.SamplingPolicy(12, method=method)
        policy.validate('tr')
        if method == 'exact_leverage':
            with pytest.raises(ValueError, match='TR topology'):
                policy.validate('tt')
        else:
            policy.validate('tt')
        assert policy.n_samples == 12
        assert policy.reuse_sweeps == 1
        assert policy.uniform_mix == 0

    @pytest.mark.parametrize('kwargs,error', [
        ({'n_samples': True}, TypeError),
        ({'n_samples': 2.5}, TypeError),
        ({'n_samples': 0}, ValueError),
        ({'reuse_sweeps': False}, TypeError),
        ({'reuse_sweeps': 0}, ValueError),
        ({'method': None}, TypeError),
        ({'method': 'frozen'}, ValueError),
        ({'uniform_mix': True}, TypeError),
        ({'uniform_mix': float('nan')}, ValueError),
        ({'uniform_mix': float('inf')}, ValueError),
        ({'uniform_mix': -0.1}, ValueError),
        ({'uniform_mix': 1.1}, ValueError),
        ({'uniform_mix': 0.1}, ValueError),
        ({'method': 'leverage', 'reuse_sweeps': 2}, ValueError),
        ({'method': 'exact_leverage', 'reuse_sweeps': 2}, ValueError),
    ])
    def test_invalid_policy_arguments(self, kwargs, error):
        with pytest.raises(error):
            tk.decompositions.SamplingPolicy(**{'n_samples': 4, **kwargs})

    def test_reuse_and_mixture_belong_to_the_appropriate_methods(self):
        uniform = tk.decompositions.SamplingPolicy(20, reuse_sweeps=3)
        leverage = tk.decompositions.SamplingPolicy(
            20, method='leverage', uniform_mix=1)
        assert uniform.reuse_sweeps == 3
        assert leverage.uniform_mix == 1
        with pytest.raises(ValueError, match='topology'):
            uniform.validate('peps')


class TestSampleBatch:  # MARK: TestSampleBatch

    def test_exact_rows_reproduce_the_complete_system(self):
        state = _RowSamplingState(n_rows=6, core_versions=(0, 0, 0))
        batch = tk.decompositions.ExactRows().draw(
            state, site=1, n_samples=None, generator=None)
        environment = torch.randn(6, 3)
        target = torch.randn(6, 2)

        sampled_environment, sampled_target = batch.gather_and_weight(
            environment, target)

        assert torch.equal(batch.ids, torch.arange(6))
        assert torch.equal(batch.weights, torch.ones(6))
        assert torch.equal(sampled_environment, environment)
        assert torch.equal(sampled_target, target)
        assert batch.is_exact_for((4, 2, 7))

    def test_weights_must_match_draw_probabilities(self):
        with pytest.raises(ValueError, match='weights'):
            tk.decompositions.SampleBatch(
                ids=torch.tensor([0, 1]),
                probabilities=torch.tensor([0.5, 0.5]),
                weights=torch.tensor([2., 2.]),
                generation=0)

    def test_design_dependent_exactness_uses_original_versions(self):
        probabilities = torch.tensor([0.25, 0.75])
        batch = tk.decompositions.SampleBatch(
            ids=torch.tensor([0, 1]),
            probabilities=probabilities,
            weights=(2 * probabilities).rsqrt(),
            generation=3,
            proposal_core_versions=(1, 4, 2),
            proposal_exact=True)
        approximate = tk.decompositions.SampleBatch(
            ids=torch.tensor([0, 1]),
            probabilities=probabilities,
            weights=(2 * probabilities).rsqrt(),
            generation=3,
            proposal_core_versions=(1, 4, 2),
            proposal_exact=False)

        assert batch.is_exact_for((1, 4, 2))
        assert not batch.is_exact_for((1, 5, 2))
        assert not approximate.is_exact_for((1, 4, 2))

    @pytest.mark.parametrize('field,value,error', [
        ('ids', [0, 1], TypeError),
        ('ids', torch.tensor([0., 1.]), TypeError),
        ('ids', torch.tensor([-1, 1]), ValueError),
        ('probabilities', torch.tensor([0., 1.]), ValueError),
        ('probabilities', torch.tensor([float('nan'), 0.5]), ValueError),
        ('weights', torch.ones(3), ValueError),
        ('generation', True, ValueError),
        ('proposal_core_versions', (-1,), ValueError),
        ('proposal_exact', 1, TypeError),
        ('site', True, ValueError),
    ])
    def test_invalid_batch_metadata(self, field, value, error):
        kwargs = dict(ids=torch.tensor([0, 1]),
                      probabilities=torch.full((2,), 0.5),
                      weights=torch.ones(2), generation=0)
        kwargs[field] = value
        with pytest.raises(error):
            tk.decompositions.SampleBatch(**kwargs)

    def test_duplicate_rows_and_vector_targets_keep_the_original_order(self):
        batch = tk.decompositions.SampleBatch(
            ids=torch.tensor([2, 0, 2]),
            probabilities=torch.full((3,), 1 / 3),
            weights=torch.ones(3), generation=0)
        matrix = torch.arange(6.).reshape(3, 2)
        target = torch.tensor([4., 5., 6.])
        actual_matrix, actual_target = batch.gather_and_weight(matrix, target)
        assert torch.equal(actual_matrix, matrix[[2, 0, 2]])
        assert torch.equal(actual_target, target[[2, 0, 2]])
        with pytest.raises(ValueError, match='matching rows'):
            batch.gather_and_weight(matrix, target[:2])


class TestRowSamplers:  # MARK: TestRowSamplers

    def test_uniform_sampling_is_deterministic(self):
        state = _RowSamplingState(n_rows=11, core_versions=(0, 0))
        sampler = tk.decompositions.UniformRows()

        first = sampler.draw(
            state,
            site=0,
            n_samples=20,
            generator=torch.Generator().manual_seed(30))
        second = sampler.draw(
            state,
            site=0,
            n_samples=20,
            generator=torch.Generator().manual_seed(30))

        assert torch.equal(first.ids, second.ids)
        assert torch.equal(first.probabilities, second.probabilities)
        assert torch.equal(first.weights, second.weights)
        assert torch.equal(
            first.probabilities,
            torch.full((20,), 1 / 11))

    def test_uniform_reweighted_gram_and_rhs_are_unbiased(self):
        generator = torch.Generator().manual_seed(31)
        environment = torch.randn(7, 3, generator=generator)
        target = torch.randn(7, 2, generator=generator)
        state = _RowSamplingState(n_rows=7, core_versions=(0, 0, 0))
        batch = tk.decompositions.UniformRows().draw(
            state,
            site=1,
            n_samples=100_000,
            generator=generator)

        sampled_environment, sampled_target = batch.gather_and_weight(
            environment, target)

        assert torch.allclose(
            sampled_environment.mT @ sampled_environment,
            environment.mT @ environment,
            rtol=2e-2,
            atol=2e-2)
        assert torch.allclose(
            sampled_environment.mT @ sampled_target,
            environment.mT @ target,
            rtol=2e-2,
            atol=2e-2)

    def test_observed_rows_are_fixed_and_unit_weighted(self):
        observations = tk.decompositions.SparseTensorSource(
            indices=torch.tensor([[0, 1], [1, 0], [1, 2]]),
            values=torch.tensor([2., 3., 4.]),
            in_dim=(2, 3))
        sampler = tk.decompositions.ObservedRows(observations)
        state = _RowSamplingState(n_rows=6, core_versions=(0, 0))

        batch = sampler.draw(
            state, site=0, n_samples=None, generator=None)

        assert torch.equal(batch.ids, torch.tensor([1, 3, 5]))
        assert torch.equal(batch.weights, torch.ones(3))
        assert not sampler.refreshable

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason='CUDA is not available')
    def test_cuda_sampling_uses_a_matching_generator(self):
        state = _RowSamplingState(
            n_rows=5,
            core_versions=(0, 0),
            device=torch.device('cuda'))
        generator = torch.Generator(device='cuda').manual_seed(32)

        batch = tk.decompositions.UniformRows().draw(
            state, site=0, n_samples=8, generator=generator)

        assert batch.ids.device.type == 'cuda'
        assert batch.probabilities.device.type == 'cuda'
        with pytest.raises(ValueError, match='generator'):
            tk.decompositions.UniformRows().draw(
                state,
                site=0,
                n_samples=8,
                generator=torch.Generator().manual_seed(32))

    def test_sampler_updates_only_the_committed_core_version(self):
        state = _RowSamplingState(n_rows=8, core_versions=(2, 3, 4))

        updated = tk.decompositions.UniformRows().update_after_core(
            state, site=1)

        assert state.core_versions == (2, 3, 4)
        assert updated.core_versions == (2, 4, 4)


class TestSampleRefreshPolicy:  # MARK: TestSampleRefreshPolicy

    def test_uniform_samples_and_probabilities_are_reused_by_generation(self):
        sampler = tk.decompositions.UniformRows()
        policy = tk.decompositions.SampleRefreshPolicy(reuse_sweeps=2)
        state = _RowSamplingState(n_rows=13, core_versions=(0, 0))
        generator = torch.Generator().manual_seed(33)
        invalidations = []

        first, state, refreshed = policy.sample(
            sampler,
            state,
            site=0,
            n_samples=10,
            generator=generator,
            sweep=0,
            invalidate=lambda old, new: invalidations.append(
                (old, new.generation)))
        reused, state, refreshed_reused = policy.sample(
            sampler,
            state,
            site=0,
            n_samples=10,
            generator=generator,
            sweep=1,
            current=first,
            invalidate=lambda old, new: invalidations.append(
                (old, new.generation)))
        second, state, refreshed_second = policy.sample(
            sampler,
            state,
            site=0,
            n_samples=10,
            generator=generator,
            sweep=2,
            current=reused,
            invalidate=lambda old, new: invalidations.append(
                (old.generation, new.generation)))

        assert refreshed
        assert not refreshed_reused
        assert refreshed_second
        assert reused is first
        assert torch.equal(reused.probabilities, first.probabilities)
        assert second.generation == 1
        assert invalidations == [(None, 0), (0, 1)]

    def test_completion_never_refreshes_observed_rows(self):
        observations = tk.decompositions.SparseTensorSource(
            indices=torch.tensor([[0, 0], [1, 1]]),
            values=torch.tensor([1., 2.]),
            in_dim=(2, 2))
        sampler = tk.decompositions.ObservedRows(observations)
        policy = tk.decompositions.SampleRefreshPolicy(reuse_sweeps=1)
        state = _RowSamplingState(n_rows=4, core_versions=(0, 0))

        first, state, refreshed = policy.sample(
            sampler, state, 0, None, None, sweep=0)
        later, state, refreshed_later = policy.sample(
            sampler, state, 0, None, None, sweep=50, current=first)

        assert refreshed
        assert not refreshed_later
        assert later is first
        assert later.generation == 0

    def test_generation_is_incremental_and_deterministic(self):
        policy = tk.decompositions.SampleRefreshPolicy(reuse_sweeps=3)

        assert [policy.generation(sweep) for sweep in range(8)] == [
            0, 0, 0, 1, 1, 1, 2, 2]


class TestTTLeverageRows:  # MARK: TestTTLeverageRows

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_probabilities_match_dense_leverage_scores(self, dtype):
        generator = torch.Generator().manual_seed(35)
        tensor = torch.randn(
            2, 3, 2, dtype=dtype, generator=generator)
        site = 1
        cores = _mixed_canonical_cores(tensor, site)
        sampler = tk.decompositions.TTLeverageRows(lambda: cores)
        flat_ids = torch.arange(prod(tensor.shape))
        indices = torch.stack(torch.unravel_index(
            flat_ids, tensor.shape), dim=1)

        probabilities = sampler.probabilities(
            site,
            tk.decompositions.ConfigurationBatch(indices))
        design = dense_local_design(cores, site, topology='tt')
        q, _ = torch.linalg.qr(design, mode='reduced')
        expected = q.abs().square().sum(dim=1) / q.shape[1]

        assert torch.allclose(
            probabilities, expected, atol=1e-11, rtol=1e-11)
        assert probabilities.sum() == pytest.approx(1.)

    def test_uniform_mixture_adds_full_support(self):
        cores = [
            torch.tensor([[[1.], [0.]]]),
            torch.tensor([[[1.], [0.]]]),
        ]
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))

        pure = tk.decompositions.TTLeverageRows(
            lambda: cores).probabilities(0, configurations)
        mixed = tk.decompositions.TTLeverageRows(
            lambda: cores, uniform_mix=0.2).probabilities(
                0, configurations)

        assert torch.any(pure < 1e-20)
        assert torch.all(mixed > 0)
        assert mixed.sum() == pytest.approx(1.)

    def test_recursive_draw_records_design_versions_and_probabilities(self):
        tensor = torch.randn(2, 3, 2, dtype=torch.float64)
        site = 1
        cores = _mixed_canonical_cores(tensor, site)
        sampler = tk.decompositions.TTLeverageRows(
            lambda: cores, uniform_mix=0.1)
        state = _RowSamplingState(
            n_rows=tensor.numel(),
            core_versions=(2, 4, 3))

        batch = sampler.draw(
            state,
            site=site,
            n_samples=100,
            generator=torch.Generator().manual_seed(36))

        indices = torch.stack(torch.unravel_index(
            batch.ids, tensor.shape), dim=1)
        expected = sampler.probabilities(
            site, tk.decompositions.ConfigurationBatch(indices))
        assert torch.allclose(batch.probabilities, expected)
        assert batch.is_exact_for((2, 4, 3))
        assert not batch.is_exact_for((2, 5, 3))
        assert batch.site == site

    def test_reweighted_gram_and_rhs_are_unbiased_in_expectation(self):
        generator = torch.Generator().manual_seed(38)
        tensor = torch.randn(2, 3, 2, dtype=torch.float64,
                             generator=generator)
        site = 1
        cores = _mixed_canonical_cores(tensor, site)
        design = dense_local_design(cores, site, topology='tt')
        target = torch.randn(design.shape[0], dtype=torch.float64,
                             generator=generator)
        sampler = tk.decompositions.TTLeverageRows(
            lambda: cores, uniform_mix=0.1)
        state = _RowSamplingState(
            n_rows=design.shape[0], core_versions=(0, 0, 0))

        batch = sampler.draw(
            state,
            site=site,
            n_samples=100_000,
            generator=generator)
        sampled_design, sampled_target = batch.gather_and_weight(
            design, target)

        assert torch.allclose(
            sampled_design.mH @ sampled_design,
            design.mH @ design,
            rtol=2e-2,
            atol=2e-2)
        assert torch.allclose(
            sampled_design.mH @ sampled_target,
            design.mH @ target,
            rtol=2e-2,
            atol=2e-2)

    def test_noncanonical_regions_are_rejected(self):
        cores = [
            torch.randn(1, 2, 2),
            torch.randn(2, 2, 2),
            torch.randn(2, 2, 1),
        ]
        sampler = tk.decompositions.TTLeverageRows(lambda: cores)
        state = _RowSamplingState(n_rows=8, core_versions=(0, 0, 0))

        with pytest.raises(ValueError, match='isometric'):
            sampler.draw(
                state,
                site=1,
                n_samples=4,
                generator=torch.Generator().manual_seed(37))


class TestTRProductLeverageRows:  # MARK: TestTRProductLeverageRows

    def test_product_probabilities_match_mode_input_leverage_formula(self):
        cores = make_tr_cores(
            in_dim=(2, 3, 2),
            rank=(2, 3, 2),
            generator=torch.Generator().manual_seed(70))
        sampler = tk.decompositions.TRProductLeverageRows(lambda: cores)
        indices = torch.cartesian_prod(
            torch.arange(2), torch.arange(3), torch.arange(2))
        probabilities = sampler.probabilities(
            1, tk.decompositions.ConfigurationBatch(indices))

        left_scores = sampler._input_leverage(cores[0])
        right_scores = sampler._input_leverage(cores[2])
        expected = (
            left_scores[indices[:, 0]] / 3 *
            right_scores[indices[:, 2]])
        assert torch.allclose(probabilities, expected)
        assert probabilities.sum() == pytest.approx(1.)
        assert not sampler.proposal_exact

        unfolding = cores[0].permute(1, 0, 2).reshape(2, -1)
        u, singular_values, _ = torch.linalg.svd(
            unfolding, full_matrices=False)
        tolerance = max(unfolding.shape) * torch.finfo(torch.float64).eps * \
            singular_values.max()
        expected_left = u[:, singular_values > tolerance].abs().square().sum(1)
        expected_left = expected_left / expected_left.sum()
        assert torch.allclose(left_scores, expected_left)

    def test_uniform_mix_adds_support_to_zero_slice_bound(self):
        cores = make_tr_cores(
            in_dim=(2, 2, 2),
            rank=(2, 2, 2),
            generator=torch.Generator().manual_seed(71))
        cores[0][:, 0, :] = 0
        indices = torch.cartesian_prod(
            torch.arange(2), torch.arange(2), torch.arange(2))
        configurations = tk.decompositions.ConfigurationBatch(indices)
        pure = tk.decompositions.TRProductLeverageRows(
            lambda: cores).probabilities(1, configurations)
        mixed = tk.decompositions.TRProductLeverageRows(
            lambda: cores, uniform_mix=0.2).probabilities(
                1, configurations)

        assert torch.any(pure < 1e-20)
        assert torch.all(mixed > 0)
        assert mixed.sum() == pytest.approx(1.)

    def test_draw_records_approximation_versions_and_exact_weights(self):
        cores = make_tr_cores(
            generator=torch.Generator().manual_seed(72))
        sampler = tk.decompositions.TRProductLeverageRows(
            lambda: cores, uniform_mix=0.1)
        state = _RowSamplingState(
            n_rows=prod(core.shape[1] for core in cores),
            core_versions=(2, 4, 3, 1))
        batch = sampler.draw(
            state,
            site=2,
            n_samples=100,
            generator=torch.Generator().manual_seed(73))
        indices = torch.stack(torch.unravel_index(
            batch.ids, tuple(core.shape[1] for core in cores)), dim=1)
        expected = sampler.probabilities(
            2, tk.decompositions.ConfigurationBatch(indices))

        assert torch.allclose(batch.probabilities, expected)
        assert torch.allclose(
            batch.weights, (batch.ids.numel() * expected).rsqrt())
        assert batch.proposal_core_versions == (2, 4, 3, 1)
        assert not batch.is_exact_for((2, 4, 3, 1))
        assert batch.site == 2
        assert batch.ids.numel() == 100 * cores[2].shape[1]
        fibers = indices.reshape(100, cores[2].shape[1], len(cores))
        assert torch.equal(
            fibers[:, :, 2],
            torch.arange(cores[2].shape[1]).expand(100, -1))
        assert torch.equal(
            fibers[:, :, :2], fibers[:, :1, :2].expand(-1, 2, -1))
        assert torch.equal(
            fibers[:, :, 3:], fibers[:, :1, 3:].expand(-1, 2, -1))
        assert sampler.update_after_core(
            state, 1).core_versions == (2, 5, 3, 1)

    def test_reweighted_full_objective_is_unbiased_in_expectation(self):
        generator = torch.Generator().manual_seed(74)
        cores = make_tr_cores(
            in_dim=(2, 2, 2),
            rank=(2, 2, 2),
            generator=generator)
        site = 1
        design = dense_local_design(cores, site, topology='tr')
        target = torch.randn(
            design.shape[0], dtype=design.dtype, generator=generator)
        sampler = tk.decompositions.TRProductLeverageRows(
            lambda: cores, uniform_mix=0.2)
        state = _RowSamplingState(
            n_rows=design.shape[0], core_versions=(0, 0, 0))
        batch = sampler.draw(
            state,
            site=site,
            n_samples=50_000,
            generator=generator)
        sampled_design, sampled_target = batch.gather_and_weight(
            design, target)

        sampled_gram = sampled_design.mH @ sampled_design
        exact_gram = design.mH @ design
        sampled_rhs = sampled_design.mH @ sampled_target
        exact_rhs = design.mH @ target
        assert torch.linalg.vector_norm(sampled_gram - exact_gram) / \
            torch.linalg.vector_norm(exact_gram) < 2e-2
        assert torch.linalg.vector_norm(sampled_rhs - exact_rhs) / \
            torch.linalg.vector_norm(exact_rhs) < 2e-2


class TestTRExactLeverageRows:  # MARK: TestTRExactLeverageRows

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_probabilities_match_dense_design_leverage(self, dtype):
        in_dim = (3, 2, 3, 2)
        cores = make_tr_cores(
            in_dim=in_dim,
            rank=(2, 3, 2, 2),
            dtype=dtype,
            generator=torch.Generator().manual_seed(75))
        site = 1
        sampler = tk.decompositions.TRExactLeverageRows(lambda: cores)
        ids = torch.arange(prod(in_dim))
        indices = torch.stack(torch.unravel_index(ids, in_dim), dim=1)
        probabilities = sampler.probabilities(
            site, tk.decompositions.ConfigurationBatch(indices))

        design = dense_local_design(cores, site, topology='tr')
        u, singular_values, _ = torch.linalg.svd(
            design, full_matrices=False)
        tolerance = max(design.shape) * torch.finfo(design.real.dtype).eps * \
            singular_values.max()
        expected = u[:, singular_values > tolerance].abs().square().sum(1)
        expected = expected / expected.sum()

        assert torch.allclose(probabilities, expected)
        assert probabilities.sum() == pytest.approx(1.)
        assert sampler.proposal_exact

    def test_draw_expands_fibers_and_records_exact_probabilities(self):
        in_dim = (3, 2, 3, 2)
        cores = make_tr_cores(
            in_dim=in_dim,
            rank=(2, 2, 2, 2),
            generator=torch.Generator().manual_seed(76))
        site = 2
        sampler = tk.decompositions.TRExactLeverageRows(
            lambda: cores, uniform_mix=0.1)
        state = _RowSamplingState(
            n_rows=prod(in_dim), core_versions=(1, 4, 2, 3))
        batch = sampler.draw(
            state,
            site=site,
            n_samples=40,
            generator=torch.Generator().manual_seed(77))
        indices = torch.stack(torch.unravel_index(
            batch.ids, in_dim), dim=1)
        expected = sampler.probabilities(
            site, tk.decompositions.ConfigurationBatch(indices))
        fibers = indices.reshape(40, in_dim[site], len(in_dim))

        assert batch.ids.numel() == 40 * in_dim[site]
        assert torch.equal(
            fibers[:, :, site],
            torch.arange(in_dim[site]).expand(40, -1))
        assert torch.equal(
            fibers[:, :, :site],
            fibers[:, :1, :site].expand(-1, in_dim[site], -1))
        assert torch.equal(
            fibers[:, :, site + 1:],
            fibers[:, :1, site + 1:].expand(-1, in_dim[site], -1))
        assert torch.allclose(batch.probabilities, expected)
        assert torch.allclose(
            batch.weights, (batch.ids.numel() * expected).rsqrt())
        assert batch.is_exact_for((1, 4, 2, 3))
        assert not batch.is_exact_for((1, 5, 2, 3))

    def test_conditional_draw_frequencies_match_exact_probabilities(self):
        in_dim = (3, 2, 3, 2)
        cores = make_tr_cores(
            in_dim=in_dim,
            rank=(2, 2, 2, 2),
            generator=torch.Generator().manual_seed(80))
        site = 1
        n_samples = 30_000
        sampler = tk.decompositions.TRExactLeverageRows(lambda: cores)
        state = _RowSamplingState(
            n_rows=prod(in_dim), core_versions=(0, 0, 0, 0))
        batch = sampler.draw(
            state,
            site=site,
            n_samples=n_samples,
            generator=torch.Generator().manual_seed(81))

        base_ids = batch.ids.reshape(n_samples, in_dim[site])[:, 0]
        frequencies = torch.bincount(
            base_ids, minlength=prod(in_dim)).to(torch.float64) / n_samples
        all_ids = torch.arange(prod(in_dim))
        all_indices = torch.stack(
            torch.unravel_index(all_ids, in_dim), dim=1)
        row_probabilities = sampler.probabilities(
            site, tk.decompositions.ConfigurationBatch(all_indices))
        expected = torch.where(
            all_indices[:, site] == 0,
            row_probabilities * in_dim[site],
            torch.zeros_like(row_probabilities))

        assert torch.max(torch.abs(frequencies - expected)) < 6e-3

    def test_zero_environment_design_is_rejected(self):
        cores = make_tr_cores(
            in_dim=(2, 2, 2),
            rank=(2, 2, 2),
            generator=torch.Generator().manual_seed(78))
        cores[0].zero_()
        sampler = tk.decompositions.TRExactLeverageRows(lambda: cores)
        state = _RowSamplingState(n_rows=8, core_versions=(0, 0, 0))

        with pytest.raises(ValueError, match='positive numerical rank'):
            sampler.draw(
                state,
                site=1,
                n_samples=4,
                generator=torch.Generator().manual_seed(79))


class TestLeverageContracts:  # MARK: TestLeverageContracts

    @pytest.mark.parametrize('sampler_type', [
        tk.decompositions.TTLeverageRows,
        tk.decompositions.TRProductLeverageRows,
        tk.decompositions.TRExactLeverageRows,
    ])
    @pytest.mark.parametrize(
        'mixture', [True, float('nan'), float('inf'), -0.1]
    )
    def test_invalid_mixtures(self, sampler_type, mixture):
        with pytest.raises((TypeError, ValueError), match='uniform_mix'):
            sampler_type(lambda: (), uniform_mix=mixture)

    @pytest.mark.parametrize('sampler_type', [
        tk.decompositions.TTLeverageRows,
        tk.decompositions.TRProductLeverageRows,
        tk.decompositions.TRExactLeverageRows,
    ])
    def test_constructor_and_core_validation(self, sampler_type):
        with pytest.raises(TypeError, match='callable'):
            sampler_type([])
        with pytest.raises(ValueError, match='at least one'):
            sampler_type(lambda: ()).probabilities(
                0,
                tk.decompositions.ConfigurationBatch(torch.zeros(1, 1).long()),
            )
        with pytest.raises(TypeError, match='Tensor'):
            sampler_type(lambda: [None]).probabilities(
                0,
                tk.decompositions.ConfigurationBatch(torch.zeros(1, 1).long()),
            )

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('method', ['tt', 'product', 'exact'])
    @pytest.mark.parametrize('n_sites', [1, 5])
    def test_minimal_and_longer_chains_have_consistent_probabilities(
            self, dtype, method, n_sites):
        dimensions = (3,) if n_sites == 1 else (2, 3, 2, 2, 2)
        site = n_sites // 2
        generator = torch.Generator().manual_seed(120)
        if method == 'tt':
            if n_sites == 1:
                cores = [torch.randn(1, 3, 1, dtype=dtype,
                                     generator=generator)]
            else:
                tensor = torch.randn(
                    dimensions, dtype=dtype, generator=generator)
                cores = _mixed_canonical_cores(tensor, site)
            sampler = tk.decompositions.TTLeverageRows(lambda: cores)
        else:
            cores = make_tr_cores(dimensions, (2,) * n_sites,
                                  dtype=dtype, generator=generator)
            sampler_type = (
                tk.decompositions.TRProductLeverageRows
                if method == 'product'
                else tk.decompositions.TRExactLeverageRows
            )
            sampler = sampler_type(lambda: cores)
        ids = torch.arange(prod(dimensions))
        indices = torch.stack(torch.unravel_index(ids, dimensions), dim=1)
        probabilities = sampler.probabilities(
            site, tk.decompositions.ConfigurationBatch(indices))
        assert torch.isfinite(probabilities).all()
        assert probabilities.sum().item() == pytest.approx(1., abs=1e-10)
        if method != 'product':
            design = dense_local_design(
                cores, site, 'tt' if method == 'tt' else 'tr')
            u, singular_values, _ = torch.linalg.svd(
                design, full_matrices=False)
            active = singular_values > max(design.shape) * \
                torch.finfo(design.real.dtype).eps * singular_values.max()
            expected = u[:, active].abs().square().sum(1) / active.sum()
            assert torch.allclose(probabilities, expected,
                                  atol=1e-10, rtol=1e-10)

        empty = tk.decompositions.ConfigurationBatch(indices[:0])
        assert sampler.probabilities(site, empty).shape == (0,)
        state = _RowSamplingState(prod(dimensions), (0,) * n_sites)
        batch = sampler.draw(state, site, 8, generator=generator)
        drawn = torch.stack(torch.unravel_index(batch.ids, dimensions), dim=1)
        expected = sampler.probabilities(
            site, tk.decompositions.ConfigurationBatch(drawn))
        assert torch.allclose(batch.probabilities, expected)
        assert batch.ids.numel() == (
            8 if method == 'tt' else 8 * dimensions[site])
        with pytest.raises(ValueError, match='n_rows'):
            sampler.draw(_RowSamplingState(1, (0,) * n_sites), site, 8)

    @pytest.mark.parametrize('n_sites', [1, 5])
    def test_observations_include_stored_zeros(self, n_sites):
        indices = torch.tensor([[0] * n_sites, [1] * n_sites])
        observations = tk.decompositions.SparseTensorSource(
            indices, torch.tensor([0., 2.]), (2,) * n_sites)
        sampler = tk.decompositions.ObservedRows(observations)
        state = _RowSamplingState(2 ** n_sites, (0,) * n_sites)
        batch = sampler.draw(state, 0)
        assert torch.equal(batch.ids, torch.tensor([0, 2 ** n_sites - 1]))
        assert torch.equal(batch.weights, torch.ones(2))
        with pytest.raises(ValueError, match='all observations'):
            sampler.draw(state, 0, n_samples=1)

    def test_observation_constructor_rejects_incompatible_sources(self):
        with pytest.raises(TypeError, match='SparseTensorSource'):
            tk.decompositions.ObservedRows(torch.ones(2))
        empty = tk.decompositions.SparseTensorSource(
            torch.empty(0, 1, dtype=torch.long), torch.empty(0), (2,))
        with pytest.raises(ValueError, match='at least one'):
            tk.decompositions.ObservedRows(empty)
        block = tk.decompositions.SparseTensorSource(
            torch.tensor([[0]]), torch.ones(1, 2), (2, 2))
        with pytest.raises(ValueError, match='all vector sites'):
            tk.decompositions.ObservedRows(block)
