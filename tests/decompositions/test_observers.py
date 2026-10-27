"""Tests for observers."""

from io import StringIO

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.observers import (ConsoleObserver,
                                                   DecompositionEvent,
                                                   HistoryObserver,
                                                   NullObserver,
                                                   _normalize_verbosity,
                                                   _resolve_observer)


@pytest.mark.parametrize('level', [1, 2, 3])
def test_observers_preserve_events_and_filter_console(level):
    history = HistoryObserver()
    output = StringIO()
    console = ConsoleObserver(verbose=level, stream=output)
    event = DecompositionEvent('site_complete', 'tt_als', level=2,
                               site=1, elapsed=0.01,
                               values={'error': torch.tensor(0.125)})
    for observer in (history, console, NullObserver()):
        observer.emit(event)
        observer.close(tk.decompositions.DecompositionMetrics())
    assert history.events == [event]
    assert bool(output.getvalue()) == (level >= 2)


@pytest.mark.parametrize('invalid', [-1, 4, 1.5, 'quiet'])
def test_invalid_verbosity_is_rejected(invalid):
    with pytest.raises((TypeError, ValueError)):
        _normalize_verbosity(invalid)


def test_observer_resolution_and_empty_history():
    history = HistoryObserver()
    observer = _resolve_observer(0, history)
    observer.emit(DecompositionEvent('start', 'tt_svd'))
    assert len(history.events) == 1


@pytest.mark.parametrize('kwargs, error', [
    ({'name': 1}, TypeError), ({'phase': None}, TypeError),
    ({'level': 0}, ValueError), ({'site': -1}, ValueError),
    ({'elapsed': -1.}, ValueError), ({'values': []}, TypeError),
])
def test_invalid_event_metadata(kwargs, error):
    options = dict(name='cut_complete', phase='TT-SVD')
    options.update(kwargs)
    with pytest.raises(error):
        DecompositionEvent(**options)


def test_composite_observer_preserves_history_and_metrics(capsys):
    history = HistoryObserver()
    observer = _resolve_observer(2, history)
    event = DecompositionEvent('cut_complete', 'TT-SVD', site=3,
                               elapsed=0.01, values={'selected_rank': 2})
    metrics = tk.decompositions.DecompositionMetrics()
    observer.emit(event)
    observer.close(metrics)
    assert history.events == [event]
    assert history.metrics is metrics
    assert 'Cut 4-5' in capsys.readouterr().out
