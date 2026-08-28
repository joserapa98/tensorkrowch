"""Tests for topology-neutral recursive-sketching orchestration."""

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.decompositions.sketching import base as base_module
from tensorkrowch.decompositions.sketching.base import _SketchingFitContext
from tensorkrowch.decompositions.sketching.phi import (
    PhiOperator,
    _MaterializedPhi,
)
from tensorkrowch.decompositions.sketching.specs import (
    _DomainSpec,
    _EmbeddingSpec,
    _OutputSpec,
)


class _ToyRecursiveSketching(tk.decompositions.RecursiveSketching):
    """Minimal concrete driver exercising every shared phase."""

    def fit(self, **kwargs):
        context = self._new_context(**kwargs)
        result = self._execute(context)
        result['context'] = context
        return result

    def _build_regions(self, context):
        return {'built': True}

    def _build_phi(self, site, regions, context):
        assert regions['built']
        return PhiOperator(
            self.source,
            ((0, self.domains.for_site(0)),
             (1, self.domains.for_site(1))),
            self.outputs)

    def _decompose(self, context):
        phi = self._plan_phi(0, context)
        with context.phase('source.evaluate', site=0):
            values = phi.materialize()
        with context.phase('values.global_transform', site=0):
            materialized = _MaterializedPhi(values, phi.layout)
        local = self._transform_local(0, materialized, context)
        fitted = self._fit_input_axis(0, local, 0, context)
        projected = self._project_range(
            0, fitted.tensor, -1, context)
        u, _, _, _ = self._trim(0, projected, context)
        with context.phase('recursion.apply', site=0):
            context.state['recursed'] = True
        with context.phase('core.solve', site=0):
            context.cores.append(u)
        context.emit('core', site=0, level=3, values={'tensor': u})
        return self._assemble_result(context)

    def _solve_local(self, *args, **kwargs):
        return None

    def _assemble_result(self, context):
        return {'cores': tuple(context.cores), 'metrics': context.metrics}

    def _validate_result(self, result, context):
        assert len(result['cores']) == 1
        assert context.state['recursed']


def _toy():
    tensor = torch.arange(6., dtype=torch.float64).reshape(3, 2)
    source = tk.decompositions.DenseTensorSource(tensor)
    domains = _DomainSpec((torch.arange(3), torch.arange(2)))
    embeddings = _EmbeddingSpec.normalize(
        (torch.eye(3, dtype=torch.float64),
         torch.eye(2, dtype=torch.float64)),
        domains)
    outputs = _OutputSpec.normalize(torch.ones(1), n_input_sites=2)
    return _ToyRecursiveSketching(source, domains, embeddings, outputs)


class TestRecursiveSketchingBase:  # MARK: TestRecursiveSketchingBase

    def test_complete_shared_phase_order_and_metrics(self):
        history = tk.decompositions.HistoryObserver()

        result = _toy().fit(
            rank=2,
            random_projection=False,
            collect_metrics=True,
            observer=history)
        phase_names = [
            event.name for event in history.events
            if event.name not in ('start', 'summary', 'core')]

        assert phase_names == [
            'source.prepare',
            'regions.build',
            'phi.plan',
            'source.evaluate',
            'values.global_transform',
            'values.local_transform',
            'input.fit',
            'range.project',
            'svd.trim',
            'recursion.apply',
            'core.solve',
            'result.validate',
        ]
        assert history.metrics is result['metrics']
        assert len(result['metrics'].timings) == 13
        assert result['metrics'].timings[-1].name == 'total'
        assert len(result['metrics'].input_fits) == 1
        assert len(result['metrics'].range_projections) == 1
        assert len(result['metrics'].truncations) == 1

    def test_fast_path_creates_no_events_records_or_diagnostic_svds(
            self, monkeypatch):
        original_svdvals = torch.linalg.svdvals

        def fail_event(*args, **kwargs):
            raise AssertionError('No event should be constructed')

        def guarded_svdvals(tensor, *args, **kwargs):
            if tensor.shape == (3, 3):
                raise AssertionError('Fitting condition should not be computed')
            return original_svdvals(tensor, *args, **kwargs)

        monkeypatch.setattr(base_module, 'DecompositionEvent', fail_event)
        monkeypatch.setattr(torch.linalg, 'svdvals', guarded_svdvals)

        result = _toy().fit(
            rank=2,
            random_projection=False,
            verbose=0,
            collect_metrics=False)

        assert result['metrics'].timings == []
        assert result['metrics'].input_fits == []
        assert result['metrics'].range_projections == []
        assert result['metrics'].truncations == []
        assert result['context'].fitted_axes[0].record is None
        assert result['context'].projected_ranges[0].record is None

    def test_repeated_fits_own_independent_mutable_contexts(self):
        decomposition = _toy()

        first = decomposition.fit(rank=1, random_projection=False)
        second = decomposition.fit(rank=2, random_projection=False)

        assert first['context'] is not second['context']
        assert first['context'].cores is not second['context'].cores
        assert first['cores'][0].shape[-1] == 1
        assert second['cores'][0].shape[-1] == 2

    def test_projection_strategy_is_resolved_per_fit(self):
        decomposition = _toy()

        exact = decomposition.fit(rank=1, random_projection=False)
        random = decomposition.fit(
            rank=1,
            projection_dim=1,
            generator=torch.Generator().manual_seed(3))

        assert isinstance(
            exact['context'].projector,
            tk.decompositions.IdentityRangeProjector)
        assert isinstance(
            random['context'].projector,
            tk.decompositions.RandomizedRangeProjector)

    def test_context_rejects_unknown_phases(self):
        context = _toy()._new_context(random_projection=False)

        with pytest.raises(ValueError, match='sketching phase'):
            with context.phase('unknown'):
                pass


class TestSketchingVerbosity:  # MARK: TestSketchingVerbosity

    def test_level_one_prints_spaced_titles_without_details(self, capsys):
        _toy().fit(rank=1, random_projection=False, verbose=1)
        output = capsys.readouterr().out

        assert 'Recursive sketching' in output
        assert 'Source · Prepare' in output
        assert 'Site 1 — Phi · Plan' in output
        assert 'elapsed' not in output
        assert 'tensor(' not in output

    def test_level_two_adds_timings_but_not_cores(self, capsys):
        _toy().fit(rank=1, random_projection=False, verbose=2)
        output = capsys.readouterr().out

        assert 'elapsed:' in output
        assert '\nCore 1' not in output
        assert 'tensor(' not in output

    def test_level_three_prints_final_cores(self, capsys):
        _toy().fit(rank=1, random_projection=False, verbose=3)
        output = capsys.readouterr().out

        assert '\nCore 1' in output
        assert 'tensor:' in output
        assert 'tensor(' in output


def test_context_type_is_internal_but_base_is_advanced_public():
    context = _toy()._new_context(random_projection=False)

    assert isinstance(context, _SketchingFitContext)
    assert tk.decompositions.RecursiveSketching is base_module.RecursiveSketching
