"""Tests for shared recursive-sketching preparation and diagnostics."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.observers import HistoryObserver
from tensorkrowch.decompositions.sketching import base as base_module
from tensorkrowch.decompositions.sketching.base import _SketchingFitContext
from tensorkrowch.decompositions.sketching.specs import (_DomainSpec,
                                                       _EmbeddingSpec,
                                                       _OutputSpec)


class _ToyRecursiveSketching(tk.decompositions.RecursiveSketching):
    """Minimal concrete driver exercising source, fitting and diagnostics."""

    def fit(self, **kwargs):
        return self._execute(self._new_context(**kwargs))

    def _prepare_source(self, context):
        context.state['prepared'] = True

    def _build_regions(self, context):
        assert context.state['prepared']
        return {'built': True}

    def _decompose(self, context):
        assert context.regions['built']
        tensor = self._fit_input_axis(
            0, self.source.contract_dense(), 0, context)
        with context.phase('core.solve', site=0):
            u, s, vh = torch.linalg.svd(tensor, full_matrices=False)
            rank = min(context.spec.rank or len(s), len(s))
            cores = [u[:, :rank], s[:rank, None] * vh[:rank]]
        for site, core in enumerate(cores):
            context.cores.append(core)
            context.complete(site, core, len(cores))
        return tk.decompositions.TTDecomposition(
            cores, metrics=context.metrics)

    def _validate_result(self, result, context):
        assert result.in_dim == self.embeddings.in_dim
        assert context.state['prepared']


def _toy(**kwargs):
    tensor = torch.arange(6., dtype=torch.float64).reshape(3, 2)
    source = tk.decompositions.DenseTensorSource(tensor)
    domains = _DomainSpec((torch.arange(3), torch.arange(2)))
    embeddings = _EmbeddingSpec.normalize(
        (torch.eye(3, dtype=torch.float64),
         torch.eye(2, dtype=torch.float64)), domains)
    outputs = _OutputSpec.normalize(torch.ones(1), n_input_sites=2)
    return _ToyRecursiveSketching(
        source, domains, embeddings, outputs, **kwargs)


class TestRecursiveSketchingBase:  # MARK: TestRecursiveSketchingBase

    def test_shared_lifecycle_records_timings_and_input_fit(self):
        decomposition = _toy()
        context = decomposition._new_context(rank=2, collect_metrics=True)
        history = HistoryObserver()
        context.observer = history

        result = decomposition._execute(context)

        torch.testing.assert_close(
            result.contract_dense(), decomposition.source.contract_dense())
        assert isinstance(context, _SketchingFitContext)
        assert history.metrics is result.metrics
        assert [record.name for record in result.metrics.timings] == [
            'source.prepare', 'regions.build', 'input.fit', 'core.solve',
            'total']
        assert all(record.elapsed >= 0 for record in result.metrics.timings)
        assert len(result.metrics.input_fits) == 1
        assert result.metrics.input_fits[0].rel_residual < 1e-12
        completed = [event.site for event in history.events
                     if event.name == 'site_complete']
        assert completed == [0, 1]

    def test_fast_path_creates_no_events_records_or_timers(self, monkeypatch):
        def forbidden(*args, **kwargs):
            raise AssertionError('Disabled diagnostics should not run')

        monkeypatch.setattr(base_module, 'DecompositionEvent', forbidden)
        monkeypatch.setattr(base_module._RuntimePolicy, 'timer', forbidden)
        monkeypatch.setattr(torch.linalg, 'svdvals', forbidden)

        result = _toy().fit(rank=2)

        assert result.metrics.timings == []
        assert result.metrics.input_fits == []
        assert result.metrics.range_projections == []
        assert result.metrics.truncations == []

    def test_repeated_fits_own_independent_mutable_contexts(self):
        decomposition = _toy()
        first_context = decomposition._new_context(rank=1)
        second_context = decomposition._new_context(rank=2)

        first = decomposition._execute(first_context)
        second = decomposition._execute(second_context)

        assert first_context is not second_context
        assert first_context.cores is not second_context.cores
        assert first_context.metrics is not second_context.metrics
        assert first.rank == [1]
        assert second.rank == [2]
        assert second_context.state == {'prepared': True}

    def test_single_fitter_can_be_shared_or_selected_per_site(self):
        shared = tk.decompositions.BasisFitter()
        decomposition = _toy(input_fitter=shared)
        assert decomposition.input_fitter == (shared, shared)
        result = decomposition.fit(rank=2)
        torch.testing.assert_close(
            result.contract_dense(), decomposition.source.contract_dense())

        individual = (tk.decompositions.BasisFitter(in_dim=3),
                      tk.decompositions.BasisFitter(in_dim=2))
        assert _toy(input_fitter=individual).input_fitter == individual

    @pytest.mark.parametrize('kwargs, error', [
        ({'input_fitter': []}, ValueError),
        ({'input_fitter': object()}, TypeError),
        ({'input_fitter': [object(), object()]}, ValueError),
        ({'range_projector': object()}, TypeError),
        ({'transform': 1}, TypeError)])
    def test_invalid_strategies_fail_at_construction(self, kwargs, error):
        with pytest.raises(error):
            _toy(**kwargs)

    def test_projector_object_is_preserved_without_fit_knobs(self):
        projector = tk.decompositions.RandomizedRangeProjector(
            projection_dim=1, n_power_iter=1)
        decomposition = _toy(range_projector=projector)
        assert decomposition._range_projector is projector
        assert _toy()._range_projector is None

    def test_context_validates_generator_and_diagnostic_options(self):
        with pytest.raises(TypeError, match='generator'):
            _toy()._new_context(generator=1)
        with pytest.raises(ValueError, match='between 0 and 3'):
            _toy()._new_context(verbose=4)
        with pytest.raises(TypeError, match='collect_metrics'):
            _toy()._new_context(collect_metrics=1)


class TestSketchingVerbosity:  # MARK: TestSketchingVerbosity

    def test_level_zero_is_silent(self, capsys):
        _toy().fit(rank=1, verbose=0)
        assert capsys.readouterr().out == ''

    def test_level_one_reports_progress_without_timings(self, capsys):
        result = _toy().fit(rank=1, verbose=1)
        output = capsys.readouterr().out
        assert 'Recursive sketching' in output
        assert 'elapsed:' not in output
        assert 'tensor(' not in output
        assert result.metrics.timings == []

    def test_level_two_adds_timings_without_storing_records(self, capsys):
        result = _toy().fit(rank=1, verbose=2)
        output = capsys.readouterr().out
        assert 'elapsed:' in output
        assert 'tensor(' not in output
        assert result.metrics.timings == []
        assert result.metrics.input_fits == []

    def test_level_three_prints_each_completed_core_once(self, capsys):
        _toy().fit(rank=1, verbose=3)
        output = capsys.readouterr().out
        assert output.count('tensor(') == 2
