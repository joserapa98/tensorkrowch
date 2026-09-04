"""Structured events and observers for tensor decompositions."""

from dataclasses import dataclass, field
import sys
from typing import (Any, Dict, List, Optional, Protocol, Sequence, TextIO,
                    Union)

from tensorkrowch.decompositions.metrics import DecompositionMetrics


@dataclass(frozen=True)
class DecompositionEvent:
    """Describes one structured event emitted by a decomposition driver."""

    name: str
    phase: str
    level: int = 1
    site: Optional[int] = None
    elapsed: Optional[float] = None
    values: Dict[str, Any] = field(default_factory=dict)
    worker: Optional[int] = None
    sweep: Optional[int] = None

    def __post_init__(self) -> None:
        for name in ('name', 'phase'):
            if not isinstance(getattr(self, name), str):
                raise TypeError(f'`{name}` should be str type')
        if not isinstance(self.level, int):
            raise TypeError('`level` should be int type')
        if self.level < 1:
            raise ValueError('`level` should be positive')
        for name in ('site', 'worker'):
            value = getattr(self, name)
            if (value is not None) and (not isinstance(value, int)):
                raise TypeError(f'`{name}` should be int type')
            if (value is not None) and (value < 0):
                raise ValueError(f'`{name}` should be non-negative')
        if self.sweep is not None:
            if isinstance(self.sweep, bool) or \
                    (not isinstance(self.sweep, int)) or (self.sweep < 0):
                raise ValueError('`sweep` should be a non-negative integer')
        if self.elapsed is not None:
            if not isinstance(self.elapsed, (int, float)):
                raise TypeError('`elapsed` should be a real scalar')
            if self.elapsed < 0:
                raise ValueError('`elapsed` should be non-negative')
            object.__setattr__(self, 'elapsed', float(self.elapsed))
        if not isinstance(self.values, dict):
            raise TypeError('`values` should be dict type')
        object.__setattr__(self, 'values', dict(self.values))


class DecompositionObserver(Protocol):
    """Protocol implemented by decomposition event consumers."""

    def emit(self, event: DecompositionEvent) -> None:
        """Consumes one decomposition event."""

    def close(self, metrics: DecompositionMetrics) -> None:
        """Consumes the metrics produced by a completed fit."""


class NullObserver:
    """Discards decomposition events."""

    def emit(self, event: DecompositionEvent) -> None:
        pass

    def close(self, metrics: DecompositionMetrics) -> None:
        pass


class HistoryObserver:
    """Stores structured events and the metrics from the latest fit."""

    def __init__(self) -> None:
        self.events: List[DecompositionEvent] = []
        self.metrics: Optional[DecompositionMetrics] = None

    def emit(self, event: DecompositionEvent) -> None:
        if not isinstance(event, DecompositionEvent):
            raise TypeError('`event` should be DecompositionEvent type')
        self.events.append(event)

    def close(self, metrics: DecompositionMetrics) -> None:
        if not isinstance(metrics, DecompositionMetrics):
            raise TypeError('`metrics` should be DecompositionMetrics type')
        self.metrics = metrics


class ConsoleObserver:
    """Prints hierarchical decomposition progress for a verbosity level."""

    def __init__(self,
                 verbose: Union[bool, int] = 1,
                 stream: Optional[TextIO] = None) -> None:
        self.verbose = _normalize_verbosity(verbose)
        self.stream = sys.stdout if stream is None else stream
        if not hasattr(self.stream, 'write'):
            raise TypeError('`stream` should be a text stream')

    def _print_values(self, values: Dict[str, Any]) -> None:
        labels = {
            'in_dim': 'input dim',
            'out_dim': 'output dim',
            'out_device': 'output device',
        }
        for name, value in values.items():
            label = labels.get(name, name.replace('_', ' '))
            print(f'  {label}: {value}', file=self.stream)

    def emit(self, event: DecompositionEvent) -> None:
        if not isinstance(event, DecompositionEvent):
            raise TypeError('`event` should be DecompositionEvent type')
        if event.level > self.verbose:
            return

        if event.name == 'start':
            print(f'\n{event.phase}', file=self.stream)
            print('=' * len(event.phase), file=self.stream)
            if self.verbose >= 2:
                self._print_values(event.values)
        elif event.name == 'site_complete':
            total = event.values.get('total_sites')
            position = event.site + 1 if event.site is not None else '?'
            suffix = f' / {total}' if total is not None else ''
            print(f'\nSite {position}{suffix}', file=self.stream)
            if self.verbose >= 2:
                details = {
                    name: value
                    for name, value in event.values.items()
                    if name != 'total_sites'
                }
                if event.elapsed is not None:
                    details['elapsed'] = f'{event.elapsed:.6f} s'
                self._print_values(details)
        elif event.name == 'summary':
            print('\nSummary', file=self.stream)
            print('-------', file=self.stream)
            self._print_values(event.values)
        elif event.name in ('sweep_start', 'sweep_complete'):
            position = '?' if event.sweep is None else event.sweep + 1
            label = 'Sweep' if event.name == 'sweep_start' else 'Sweep complete'
            print(f'\n{label} {position}', file=self.stream)
            if event.elapsed is not None:
                values = dict(event.values)
                values['elapsed'] = f'{event.elapsed:.6f} s'
            else:
                values = event.values
            self._print_values(values)
        elif event.name == 'core':
            position = event.site + 1 if event.site is not None else '?'
            print(f'\nCore {position}', file=self.stream)
            self._print_values(event.values)
        else:
            label = event.name.replace('.', ' · ').replace('_', ' ').title()
            if event.site is not None:
                label = f'Site {event.site + 1} — {label}'
            print(f'\n{label}', file=self.stream)
            print('-' * len(label), file=self.stream)
            if self.verbose >= 2:
                values = dict(event.values)
                if event.elapsed is not None:
                    values['elapsed'] = f'{event.elapsed:.6f} s'
                self._print_values(values)

    def close(self, metrics: DecompositionMetrics) -> None:
        if not isinstance(metrics, DecompositionMetrics):
            raise TypeError('`metrics` should be DecompositionMetrics type')


class _CompositeObserver:
    """Forwards decomposition events to several observers."""

    def __init__(self, observers: Sequence[DecompositionObserver]) -> None:
        self.observers = tuple(observers)

    def emit(self, event: DecompositionEvent) -> None:
        for observer in self.observers:
            observer.emit(event)

    def close(self, metrics: DecompositionMetrics) -> None:
        for observer in self.observers:
            observer.close(metrics)


def _normalize_verbosity(verbose: Union[bool, int]) -> int:
    """Normalizes boolean and integer verbosity levels."""
    if isinstance(verbose, bool):
        return int(verbose)
    if not isinstance(verbose, int):
        raise TypeError('`verbose` should be bool or int type')
    if (verbose < 0) or (verbose > 3):
        raise ValueError('`verbose` should be between 0 and 3')
    return verbose


def _resolve_observer(
        verbose: Union[bool, int],
        observer: Optional[DecompositionObserver]) -> DecompositionObserver:
    """Combines an optional observer with the selected console verbosity."""
    verbosity = _normalize_verbosity(verbose)
    if observer is not None:
        if not callable(getattr(observer, 'emit', None)) or \
                not callable(getattr(observer, 'close', None)):
            raise TypeError('`observer` should implement `emit` and `close`')
    if not verbosity:
        return NullObserver() if observer is None else observer

    console = ConsoleObserver(verbosity)
    if observer is None:
        return console
    return _CompositeObserver((console, observer))


__all__ = [
    'DecompositionEvent',
    'DecompositionObserver',
    'ConsoleObserver',
    'HistoryObserver',
]
