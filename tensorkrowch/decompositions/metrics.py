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


def _zero_safe_ratio(numerator: torch.Tensor,
                     denominator: torch.Tensor) -> torch.Tensor:
    """Divides non-negative errors with an explicit zero-denominator policy."""
    positive = denominator > 0
    safe_denominator = torch.where(
        positive, denominator, torch.ones_like(denominator))
    ratio = numerator / safe_denominator
    zero_ratio = torch.where(
        numerator == 0,
        torch.zeros_like(numerator),
        torch.full_like(numerator, torch.inf))
    return torch.where(positive, ratio, zero_ratio)


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

    @classmethod
    def from_svd_info(cls,
                      info: Any,
                      site: int,
                      log_scale: Optional[float] = None,
                      log_scale_per_batch: Optional[torch.Tensor] = None,
                      global_input_norm: Optional[float] = None,
                      global_input_norm_per_batch: Optional[
                          torch.Tensor] = None,
                      singular_values: Optional[torch.Tensor] = None
                      ) -> 'TruncationRecord':
        """Builds a high-level record from ``_TruncatedSVDInfo``."""
        if (log_scale is not None) and (log_scale_per_batch is not None):
            raise ValueError(
                'Only one of `log_scale` and `log_scale_per_batch` may be set')

        total_per_batch = info.total_squared_norm_per_batch
        discarded_per_batch = info.discarded_squared_norm_per_batch
        if log_scale_per_batch is not None:
            if not isinstance(log_scale_per_batch, torch.Tensor):
                raise TypeError(
                    '`log_scale_per_batch` should be torch.Tensor type')
            log_scale_per_batch = log_scale_per_batch.to(
                device=total_per_batch.device,
                dtype=total_per_batch.dtype)
            if log_scale_per_batch.shape != total_per_batch.shape:
                raise ValueError(
                    '`log_scale_per_batch` should match the SVD batch shape')
            if not torch.isfinite(log_scale_per_batch).all():
                raise ValueError('`log_scale_per_batch` should be finite')
            scale = log_scale_per_batch.exp()
        elif log_scale is not None:
            log_scale = _scalar_float(log_scale, 'log_scale')
            if not isfinite(log_scale):
                raise ValueError('`log_scale` should be finite')
            scale = total_per_batch.new_tensor(log_scale).exp()
        else:
            scale = torch.ones_like(total_per_batch)
        if not torch.isfinite(scale).all():
            raise ValueError('The truncation scale should be finite')

        total_per_batch = total_per_batch * scale.square()
        discarded_per_batch = discarded_per_batch * scale.square()
        if singular_values is not None:
            if not isinstance(singular_values, torch.Tensor):
                raise TypeError('`singular_values` should be torch.Tensor type')
            if singular_values.shape[:-1] != total_per_batch.shape:
                raise ValueError(
                    '`singular_values` should match the SVD batch shape')
            singular_values = singular_values.to(
                device=total_per_batch.device,
                dtype=total_per_batch.dtype)
            singular_values = singular_values * scale.unsqueeze(-1)
        input_norm_per_batch = total_per_batch.sqrt()
        absolute_per_batch = discarded_per_batch.sqrt()
        relative_per_batch = _zero_safe_ratio(
            absolute_per_batch, input_norm_per_batch)

        input_norm = input_norm_per_batch.square().sum().sqrt()
        absolute = absolute_per_batch.square().sum().sqrt()
        relative = _zero_safe_ratio(absolute, input_norm)

        global_contribution = None
        global_contribution_per_batch = None
        if global_input_norm is not None:
            global_input_norm = _scalar_float(
                global_input_norm, 'global_input_norm')
            if global_input_norm < 0:
                raise ValueError('`global_input_norm` should be non-negative')
            global_norm = total_per_batch.new_tensor(global_input_norm)
            global_contribution = _zero_safe_ratio(absolute, global_norm)
        if global_input_norm_per_batch is not None:
            if not isinstance(global_input_norm_per_batch, torch.Tensor):
                raise TypeError(
                    '`global_input_norm_per_batch` should be torch.Tensor type')
            global_norms = global_input_norm_per_batch.to(
                device=total_per_batch.device,
                dtype=total_per_batch.dtype)
            if global_norms.shape != total_per_batch.shape:
                raise ValueError(
                    '`global_input_norm_per_batch` should match the SVD batch '
                    'shape')
            if torch.any(global_norms < 0):
                raise ValueError(
                    '`global_input_norm_per_batch` should be non-negative')
            global_contribution_per_batch = _zero_safe_ratio(
                absolute_per_batch, global_norms)

        has_batches = total_per_batch.ndim > 0
        return cls(
            site=site,
            full_rank=info.full_rank,
            selected_rank=info.selected_rank,
            discarded_squared_norm=discarded_per_batch.sum(),
            local_absolute_error=absolute,
            input_norm=input_norm,
            local_relative_error=relative,
            global_relative_contribution=global_contribution,
            log_scale=log_scale,
            svd_method=info.svd_method,
            discarded_squared_norm_per_batch=(
                discarded_per_batch if has_batches else None),
            local_absolute_error_per_batch=(
                absolute_per_batch if has_batches else None),
            input_norm_per_batch=(
                input_norm_per_batch if has_batches else None),
            local_relative_error_per_batch=(
                relative_per_batch if has_batches else None),
            global_relative_contribution_per_batch=(
                global_contribution_per_batch if has_batches else None),
            log_scale_per_batch=(
                log_scale_per_batch if has_batches else None),
            singular_values=singular_values)

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
