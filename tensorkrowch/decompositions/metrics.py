"""Structured metrics returned by tensor decomposition algorithms."""

from dataclasses import dataclass, field, fields, is_dataclass
from math import isfinite
from typing import Any, Dict, List, Optional, Sequence

import torch


def _scalar_float(value: Any, name: str) -> float:
    """Converts a real scalar to a Python float."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f'`{name}` should be a scalar')
        value = value.detach().cpu().item()
    if not isinstance(value, (int, float)):
        raise TypeError(f'`{name}` should be a real scalar')
    return float(value)


def _optional_cpu_tensor(value: Optional[torch.Tensor],
                         name: str) -> Optional[torch.Tensor]:
    """Detaches an optional diagnostic tensor and stores it on CPU."""
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'`{name}` should be torch.Tensor type')
    return value.detach().cpu()


def _record_as_dict(record: Any) -> Dict[str, Any]:
    """Returns dataclass fields without deep-copying diagnostic tensors."""
    result = {}
    for item in fields(record):
        value = getattr(record, item.name)
        if is_dataclass(value):
            value = _record_as_dict(value)
        elif isinstance(value, (list, tuple)):
            value = [
                _record_as_dict(element) if is_dataclass(element) else element
                for element in value
            ]
        result[item.name] = value
    return result


@dataclass(frozen=True)
class ErrorRecord:
    """Stores an absolute/relative error measured on a specified target."""

    kind: str
    absolute: float
    relative: Optional[float] = None
    size: Optional[int] = None
    denominator: Optional[float] = None
    absolute_per_batch: Optional[torch.Tensor] = None
    relative_per_batch: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise TypeError('`kind` should be str type')

        absolute = _scalar_float(self.absolute, 'absolute')
        if absolute < 0:
            raise ValueError('`absolute` should be non-negative')
        object.__setattr__(self, 'absolute', absolute)

        if self.relative is not None:
            relative = _scalar_float(self.relative, 'relative')
            if relative < 0:
                raise ValueError('`relative` should be non-negative')
            object.__setattr__(self, 'relative', relative)

        if self.size is not None:
            if not isinstance(self.size, int):
                raise TypeError('`size` should be int type')
            if self.size < 0:
                raise ValueError('`size` should be non-negative')

        if self.denominator is not None:
            denominator = _scalar_float(self.denominator, 'denominator')
            if denominator < 0:
                raise ValueError('`denominator` should be non-negative')
            object.__setattr__(self, 'denominator', denominator)

        object.__setattr__(
            self,
            'absolute_per_batch',
            _optional_cpu_tensor(self.absolute_per_batch,
                                 'absolute_per_batch'))
        object.__setattr__(
            self,
            'relative_per_batch',
            _optional_cpu_tensor(self.relative_per_batch,
                                 'relative_per_batch'))


@dataclass(frozen=True)
class TruncationRecord:
    """Stores ranks and discarded energy for one truncation cut."""

    site: int
    full_rank: int
    selected_rank: int
    discarded_squared_norm: float
    local_absolute_error: float
    input_norm: Optional[float] = None
    local_relative_error: Optional[float] = None
    global_relative_contribution: Optional[float] = None
    log_scale: Optional[float] = None
    svd_method: Optional[str] = None
    discarded_squared_norm_per_batch: Optional[torch.Tensor] = None
    local_absolute_error_per_batch: Optional[torch.Tensor] = None
    input_norm_per_batch: Optional[torch.Tensor] = None
    local_relative_error_per_batch: Optional[torch.Tensor] = None
    global_relative_contribution_per_batch: Optional[torch.Tensor] = None
    log_scale_per_batch: Optional[torch.Tensor] = None
    singular_values: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        for name in ('site', 'full_rank', 'selected_rank'):
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')

        if self.site < 0:
            raise ValueError('`site` should be non-negative')
        if self.full_rank < 1:
            raise ValueError('`full_rank` should be positive')
        if (self.selected_rank < 1) or \
                (self.selected_rank > self.full_rank):
            raise ValueError(
                '`selected_rank` should be between 1 and `full_rank`')

        non_negative_fields = (
            'discarded_squared_norm',
            'local_absolute_error',
            'input_norm',
            'local_relative_error',
            'global_relative_contribution',
        )
        for name in non_negative_fields:
            value = getattr(self, name)
            if value is None:
                continue
            value = _scalar_float(value, name)
            if value < 0:
                raise ValueError(f'`{name}` should be non-negative')
            object.__setattr__(self, name, value)

        if self.log_scale is not None:
            log_scale = _scalar_float(self.log_scale, 'log_scale')
            if not isfinite(log_scale):
                raise ValueError('`log_scale` should be finite')
            object.__setattr__(self, 'log_scale', log_scale)

        if (self.svd_method is not None) and \
                (self.svd_method not in ('svd', 'qr_svd')):
            raise ValueError('`svd_method` should be "svd" or "qr_svd"')

        tensor_fields = (
            'discarded_squared_norm_per_batch',
            'local_absolute_error_per_batch',
            'input_norm_per_batch',
            'local_relative_error_per_batch',
            'global_relative_contribution_per_batch',
            'log_scale_per_batch',
            'singular_values',
        )
        for name in tensor_fields:
            object.__setattr__(
                self,
                name,
                _optional_cpu_tensor(getattr(self, name), name))


@dataclass(frozen=True)
class TimingRecord:
    """Stores elapsed time for a decomposition phase or site."""

    name: str
    elapsed: float
    site: Optional[int] = None
    worker: Optional[int] = None
    children: Sequence['TimingRecord'] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError('`name` should be str type')
        elapsed = _scalar_float(self.elapsed, 'elapsed')
        if elapsed < 0:
            raise ValueError('`elapsed` should be non-negative')
        object.__setattr__(self, 'elapsed', elapsed)

        for name in ('site', 'worker'):
            value = getattr(self, name)
            if (value is not None) and (not isinstance(value, int)):
                raise TypeError(f'`{name}` should be int type')
            if (value is not None) and (value < 0):
                raise ValueError(f'`{name}` should be non-negative')

        children = tuple(self.children)
        if not all(isinstance(child, TimingRecord) for child in children):
            raise TypeError('`children` should contain TimingRecord objects')
        object.__setattr__(self, 'children', children)


@dataclass(frozen=True)
class FidelityRecord:
    """Stores a phase-aware normalized overlap and its fidelity."""

    normalized_overlap: complex
    error: Optional[ErrorRecord] = None
    fidelity: float = field(init=False)

    def __post_init__(self) -> None:
        overlap = self.normalized_overlap
        if isinstance(overlap, torch.Tensor):
            if overlap.numel() != 1:
                raise ValueError('`normalized_overlap` should be a scalar')
            overlap = overlap.detach().cpu().item()
        if not isinstance(overlap, (int, float, complex)):
            raise TypeError(
                '`normalized_overlap` should be a numeric scalar')
        overlap = complex(overlap)
        object.__setattr__(self, 'normalized_overlap', overlap)
        object.__setattr__(self, 'fidelity', float(abs(overlap) ** 2))

        if (self.error is not None) and \
                (not isinstance(self.error, ErrorRecord)):
            raise TypeError('`error` should be ErrorRecord type')


@dataclass
class DecompositionMetrics:
    """Collects structured records produced during a decomposition fit."""

    errors: List[ErrorRecord] = field(default_factory=list)
    truncations: List[TruncationRecord] = field(default_factory=list)
    timings: List[TimingRecord] = field(default_factory=list)
    fidelities: List[FidelityRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.errors = list(self.errors)
        self.truncations = list(self.truncations)
        self.timings = list(self.timings)
        self.fidelities = list(self.fidelities)
        self.warnings = list(self.warnings)

        collections = (
            ('errors', self.errors, ErrorRecord),
            ('truncations', self.truncations, TruncationRecord),
            ('timings', self.timings, TimingRecord),
            ('fidelities', self.fidelities, FidelityRecord),
        )
        for name, records, record_type in collections:
            if not all(isinstance(record, record_type) for record in records):
                raise TypeError(
                    f'`{name}` should contain {record_type.__name__} objects')
        if not all(isinstance(message, str) for message in self.warnings):
            raise TypeError('`warnings` should contain strings')

    def as_info(self) -> Dict[str, Any]:
        """Returns a dictionary suitable for functional ``return_info`` APIs."""
        return {
            'errors': [_record_as_dict(record) for record in self.errors],
            'truncations': [
                _record_as_dict(record) for record in self.truncations
            ],
            'timings': [_record_as_dict(record) for record in self.timings],
            'fidelities': [
                _record_as_dict(record) for record in self.fidelities
            ],
            'warnings': list(self.warnings),
        }


__all__ = [
    'ErrorRecord',
    'TruncationRecord',
    'TimingRecord',
    'FidelityRecord',
    'DecompositionMetrics',
]
