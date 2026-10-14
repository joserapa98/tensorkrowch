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
