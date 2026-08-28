"""Tests for the refactored TT recursive-sketching decomposition."""

import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.results import TTDecomposition
from tensorkrowch.decompositions.tt_decompositions import tt_rss as legacy_tt_rss


def _problem():
    """Returns a small exactly representable scalar function."""
    domain = torch.tensor([0., 1.], dtype=torch.float64)
    samples = torch.cartesian_prod(domain, domain, domain)

    def function(data):
        return 1 + data.prod(dim=1, keepdim=True)

    def embedding(data):
        return torch.stack((1 - data, data), dim=-1)

    return function, embedding, samples, domain


class TestTTRSS:

    def test_class_returns_lightweight_result_and_reuses_fixed_problem(self):
        function, embedding, samples, domain = _problem()
        decomposer = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain)

        first = decomposer.fit(
            samples,
            rank=2,
            generator=torch.Generator().manual_seed(91))
        second = decomposer.fit(
            samples,
            rank=2,
            generator=torch.Generator().manual_seed(91))

        assert isinstance(first, TTDecomposition)
        assert first.rank == [2, 2]
        assert first.input_dim == (2, 2, 2)
        assert all(core.device.type == 'cpu' for core in first.cores)
        assert all(torch.equal(left, right)
                   for left, right in zip(first.cores, second.cores))

    def test_refactored_math_matches_legacy_dense_tensor(self):
        function, embedding, samples, domain = _problem()
        legacy_cores = legacy_tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            generator=torch.Generator().manual_seed(92),
            verbose=False)
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=2,
                generator=torch.Generator().manual_seed(92))

        legacy = TTDecomposition(legacy_cores)
        assert torch.allclose(
            result.contract_dense(),
            legacy.contract_dense(),
            rtol=1e-10,
            atol=1e-12)

    def test_one_global_evaluation_plan_and_structured_records(self):
        function, embedding, samples, domain = _problem()
        observer = tk.decompositions.HistoryObserver()
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=2,
                generator=torch.Generator().manual_seed(93),
                collect_metrics=True,
                observer=observer)

        assert len(result.metrics.evaluations) == 1
        stats = result.metrics.evaluations[0]
        assert stats.requested_points > stats.unique_points
        assert stats.cache_hits == stats.requested_points - stats.unique_points
        assert len(result.metrics.input_fits) == 3
        assert len(result.metrics.truncations) == 2
        assert len(result.metrics.local_solves) == 2
        assert result.metrics.errors[0].kind == 'sketch_samples'

        names = [event.name for event in observer.events]
        assert names[0] == 'start'
        assert names[-1] == 'summary'
        assert names.count('phi.plan') == 3
        assert names.count('values.global_transform') == 1
        assert names.count('recursion.apply') == 2
        assert names.count('core.solve') == 2

    def test_nonlegacy_mode_uses_shared_range_projector(self):
        function, embedding, samples, domain = _problem()
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=2,
                generator=torch.Generator().manual_seed(95),
                legacy_projection=False,
                collect_metrics=True)

        approximation = result.evaluate(samples, embedding=embedding)
        assert torch.allclose(
            approximation,
            function(samples).squeeze(1),
            rtol=1e-10,
            atol=1e-12)
        assert len(result.metrics.range_projections) == 2
        assert all(record.method == 'randomized'
                   for record in result.metrics.range_projections)

    def test_explicit_callable_source_uses_the_same_fit_contract(self):
        function, embedding, samples, domain = _problem()
        source = tk.decompositions.CallableTensorSource(
            function,
            input_dim=(2, 2, 2),
            output_shape=(1,),
            dtype=torch.float64)
        result = tk.decompositions.TTRSS(
            source=source,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=2,
                generator=torch.Generator().manual_seed(96))

        assert torch.allclose(
            result.evaluate(samples, embedding=embedding),
            function(samples).squeeze(1),
            rtol=1e-10,
            atol=1e-12)

    def test_functional_wrapper_keeps_legacy_and_structured_info(self):
        function, embedding, samples, domain = _problem()
        cores, info = tk.decompositions.tt_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=2,
            generator=torch.Generator().manual_seed(94),
            verbose=False,
            return_info=True)

        assert len(cores) == 3
        assert info['total_time'] >= 0
        assert info['val_eps'] < 1e-10
        assert info['topology'] == 'tt'
        assert info['rank'] == [2, 2]
        assert info['metrics']['errors'][0]['kind'] == 'sketch_samples'


__all__ = []
