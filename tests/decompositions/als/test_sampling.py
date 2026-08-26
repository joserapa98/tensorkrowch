"""Tests for ALS row sampling and refresh semantics."""

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.decompositions.als.sampling import _RowSamplingState


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
        observations = tk.decompositions.ObservedEntries(
            indices=torch.tensor([[0, 1], [1, 0], [1, 2]]),
            values=torch.tensor([2., 3., 4.]),
            input_dim=(2, 3))
        sampler = tk.decompositions.ObservedRows(observations)
        state = _RowSamplingState(n_rows=6, core_versions=(0, 0))

        batch = sampler.draw(
            state, site=0, n_samples=None, generator=None)

        assert torch.equal(batch.ids, observations.flat_ids)
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
        observations = tk.decompositions.ObservedEntries(
            indices=torch.tensor([[0, 0], [1, 1]]),
            values=torch.tensor([1., 2.]),
            input_dim=(2, 2))
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
