"""Tests for the refactored TT recursive-sketching decomposition."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.observers import HistoryObserver
from tensorkrowch.decompositions.results import TTDecomposition


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

    def test_refactored_math_matches_dense_tensor(self):
        function, embedding, samples, domain = _problem()
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=2,
                generator=torch.Generator().manual_seed(92))

        assert torch.allclose(
            result.contract_dense(),
            function(samples).reshape(2, 2, 2),
            rtol=1e-10,
            atol=1e-12)

    def test_one_global_evaluation_plan_and_structured_records(self):
        function, embedding, samples, domain = _problem()
        observer = HistoryObserver()
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

    def test_heterogeneous_coordinates_embeddings_and_input_dim(self):
        scalar_domain = torch.tensor([0., 1.], dtype=torch.float64)
        vector_domain = torch.tensor(
            [[0., 0.], [1., 1.]], dtype=torch.float64)
        first = scalar_domain.repeat_interleave(2)
        second = vector_domain.repeat((2, 1))

        def function(values):
            x, vector = values
            return 1 + x + vector.sum(dim=1)

        embeddings = (
            lambda values: torch.stack((1 - values, values), dim=1),
            lambda values: torch.cat(
                (torch.ones(values.shape[0], 1, dtype=values.dtype), values),
                dim=1),
        )
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embeddings,
            input_dim=(2, 3),
            domain=(scalar_domain, vector_domain)).fit(
                (first, second),
                rank=2,
                collect_metrics=True)

        assert result.input_dim == (2, 3)
        assert result.metrics.errors[0].kind == 'sketch_samples'
        assert result.metrics.errors[0].relative < 1e-10

    def test_tensor_output_uses_separated_sites_and_flat_labels(self):
        function, embedding, samples, domain = _problem()

        def tensor_function(data):
            value = function(data).squeeze(1)
            return torch.stack(
                (value, value + 1, 2 * value, 2 * value + 1), dim=1
            ).reshape(-1, 2, 2)

        labels = torch.arange(samples.shape[0]).remainder(4)
        result = tk.decompositions.TTRSS(
            function=tensor_function,
            embedding=embedding,
            domain=domain,
            out_position=(0, 4)).fit(
                samples,
                labels=labels,
                rank=4,
                collect_metrics=True)

        assert result.input_dim == (2, 2, 2, 2, 2)
        assert result.metadata['output_shape'] == (2, 2)
        assert result.metadata['out_position'] == (0, 4)
        assert result.metrics.errors[0].relative < 1e-10

    def test_projection_controls_total_metrics_and_output_device(self):
        function, embedding, samples, domain = _problem()
        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain,
            output_device=None).fit(
                samples,
                rank=2,
                random_projection=True,
                projection_dim=2,
                collect_metrics=True)

        assert result.device == samples.device
        assert result.metrics.timings[-1].name == 'total'
        assert result.metadata['core_shapes'] == [
            tuple(core.shape) for core in result.cores]
        assert all(record.requested_dim == 2
                   for record in result.metrics.range_projections)

    def test_input_dim_and_warm_start_are_explicit(self):
        function, embedding, samples, domain = _problem()
        with pytest.raises(ValueError, match='input_dim'):
            tk.decompositions.TTRSS(
                function=function,
                embedding=embedding,
                input_dim=3,
                domain=domain).fit(samples, rank=2)

        result = tk.decompositions.TTRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(samples, rank=2)
        with pytest.raises(NotImplementedError, match='warm-start'):
            tk.decompositions.TTRSS(
                function=function,
                embedding=embedding,
                domain=domain).fit(
                    samples, rank=2, warm_start=result)


__all__ = []
