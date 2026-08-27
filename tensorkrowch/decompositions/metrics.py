"""Structured metrics returned by tensor decomposition algorithms."""

from dataclasses import dataclass, field, fields, is_dataclass
from math import isfinite
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _ratio_from_log_norms(log_numerator: torch.Tensor,
                          log_denominator: torch.Tensor) -> torch.Tensor:
    """Computes a norm ratio before exponentiating its log difference."""
    positive = ~torch.isneginf(log_denominator)
    safe_log_denominator = torch.where(
        positive, log_denominator, torch.zeros_like(log_denominator))
    ratio = (log_numerator - safe_log_denominator).exp()
    zero_ratio = torch.where(
        torch.isneginf(log_numerator),
        torch.zeros_like(log_numerator),
        torch.full_like(log_numerator, torch.inf))
    return torch.where(positive, ratio, zero_ratio)


def _norm_from_log(log_norm: torch.Tensor,
                   reference_norm: Optional[
                       torch.Tensor] = None) -> torch.Tensor:
    """Materializes a log-norm, relative to a finite reference if possible."""
    if reference_norm is None:
        return log_norm.exp()

    positive = reference_norm > 0
    safe_reference = torch.where(
        positive, reference_norm, torch.ones_like(reference_norm))
    value = (log_norm - safe_reference.log()).exp() * safe_reference
    zero_reference_value = torch.where(
        torch.isneginf(log_norm),
        torch.zeros_like(log_norm),
        torch.full_like(log_norm, torch.inf))
    return torch.where(positive, value, zero_reference_value)


def _aggregate_log_norm(log_norms: torch.Tensor) -> torch.Tensor:
    """Aggregates independent log-norms without materializing their squares."""
    if not log_norms.ndim:
        return log_norms
    return torch.logsumexp(2 * log_norms.flatten(), dim=0) / 2


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
    phase: Optional[str] = None

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

        total_log_norm_per_batch = \
            info.total_squared_norm_per_batch.log() / 2
        discarded_log_norm_per_batch = \
            info.discarded_squared_norm_per_batch.log() / 2
        if log_scale_per_batch is not None:
            if not isinstance(log_scale_per_batch, torch.Tensor):
                raise TypeError(
                    '`log_scale_per_batch` should be torch.Tensor type')
            log_scale_per_batch = log_scale_per_batch.to(
                device=total_log_norm_per_batch.device,
                dtype=total_log_norm_per_batch.dtype)
            if log_scale_per_batch.shape != total_log_norm_per_batch.shape:
                raise ValueError(
                    '`log_scale_per_batch` should match the SVD batch shape')
            if not torch.isfinite(log_scale_per_batch).all():
                raise ValueError('`log_scale_per_batch` should be finite')
            scale_log = log_scale_per_batch
        elif log_scale is not None:
            log_scale = _scalar_float(log_scale, 'log_scale')
            if not isfinite(log_scale):
                raise ValueError('`log_scale` should be finite')
            scale_log = total_log_norm_per_batch.new_tensor(log_scale)
        else:
            scale_log = torch.zeros_like(total_log_norm_per_batch)

        total_log_norm_per_batch = total_log_norm_per_batch + scale_log
        discarded_log_norm_per_batch = (
            discarded_log_norm_per_batch + scale_log)

        global_norm = None
        if global_input_norm is not None:
            global_input_norm = _scalar_float(
                global_input_norm, 'global_input_norm')
            if (global_input_norm < 0) or \
                    (not isfinite(global_input_norm)):
                raise ValueError(
                    '`global_input_norm` should be finite and non-negative')
            global_norm = total_log_norm_per_batch.new_tensor(
                global_input_norm)

        global_norms = None
        if global_input_norm_per_batch is not None:
            if not isinstance(global_input_norm_per_batch, torch.Tensor):
                raise TypeError(
                    '`global_input_norm_per_batch` should be torch.Tensor type')
            global_norms = global_input_norm_per_batch.to(
                device=total_log_norm_per_batch.device,
                dtype=total_log_norm_per_batch.dtype)
            if global_norms.shape != total_log_norm_per_batch.shape:
                raise ValueError(
                    '`global_input_norm_per_batch` should match the SVD batch '
                    'shape')
            if torch.any(global_norms < 0) or \
                    (not torch.isfinite(global_norms).all()):
                raise ValueError(
                    '`global_input_norm_per_batch` should be finite and '
                    'non-negative')

        reference_norm = global_norms
        if (reference_norm is None) and (global_norm is not None):
            reference_norm = global_norm.expand(
                total_log_norm_per_batch.shape)
        input_norm_per_batch = _norm_from_log(
            total_log_norm_per_batch, reference_norm)
        absolute_per_batch = _norm_from_log(
            discarded_log_norm_per_batch, reference_norm)
        relative_per_batch = _ratio_from_log_norms(
            discarded_log_norm_per_batch, total_log_norm_per_batch)

        if singular_values is not None:
            if not isinstance(singular_values, torch.Tensor):
                raise TypeError('`singular_values` should be torch.Tensor type')
            if singular_values.shape[:-1] != total_log_norm_per_batch.shape:
                raise ValueError(
                    '`singular_values` should match the SVD batch shape')
            singular_values = singular_values.to(
                device=total_log_norm_per_batch.device,
                dtype=total_log_norm_per_batch.dtype)
            singular_log = singular_values.log() + scale_log.unsqueeze(-1)
            singular_reference = None if reference_norm is None \
                else reference_norm.unsqueeze(-1)
            singular_values = _norm_from_log(
                singular_log, singular_reference)

        input_log_norm = _aggregate_log_norm(total_log_norm_per_batch)
        absolute_log_norm = _aggregate_log_norm(
            discarded_log_norm_per_batch)
        aggregate_reference = global_norm
        if (aggregate_reference is None) and (global_norms is not None):
            global_log_norm = _aggregate_log_norm(global_norms.log())
            aggregate_reference = global_log_norm.exp()
        input_norm = _norm_from_log(input_log_norm, aggregate_reference)
        absolute = _norm_from_log(
            absolute_log_norm, aggregate_reference)
        relative = _ratio_from_log_norms(
            absolute_log_norm, input_log_norm)

        global_contribution = None
        global_contribution_per_batch = None
        if global_norm is not None:
            global_contribution = _ratio_from_log_norms(
                absolute_log_norm, global_norm.log())
        if global_norms is not None:
            global_contribution_per_batch = _ratio_from_log_norms(
                discarded_log_norm_per_batch, global_norms.log())

        discarded_per_batch = absolute_per_batch.square()
        has_batches = total_log_norm_per_batch.ndim > 0
        return cls(
            site=site,
            full_rank=info.full_rank,
            selected_rank=info.selected_rank,
            discarded_squared_norm=absolute.square(),
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
        if (self.phase is not None) and (not isinstance(self.phase, str)):
            raise TypeError('`phase` should be str type')

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
class LocalSolveRecord:
    """Stores diagnostics for one local least-squares solve."""

    environment_shape: Tuple[int, int]
    target_shape: Tuple[int, ...]
    driver: str
    residual_absolute: float
    residual_relative: float
    target_norm: float
    l2_reg: float = 0.0
    effective_l2_reg: float = 0.0
    l2_reg_mode: str = 'absolute'
    column_scaling: bool = False
    system_scaling: bool = False
    system_scale: float = 1.0
    site: Optional[Any] = None
    sweep: Optional[int] = None

    def __post_init__(self) -> None:
        environment_shape = tuple(self.environment_shape)
        target_shape = tuple(self.target_shape)
        if (len(environment_shape) != 2) or \
                any((not isinstance(dim, int)) or (dim < 1)
                    for dim in environment_shape):
            raise ValueError(
                '`environment_shape` should contain two positive integers')
        if (not target_shape) or \
                any((not isinstance(dim, int)) or (dim < 1)
                    for dim in target_shape):
            raise ValueError(
                '`target_shape` should contain positive integers')
        object.__setattr__(self, 'environment_shape', environment_shape)
        object.__setattr__(self, 'target_shape', target_shape)

        if not isinstance(self.driver, str):
            raise TypeError('`driver` should be str type')
        for name in (
                'residual_absolute',
                'residual_relative',
                'target_norm',
                'l2_reg',
                'effective_l2_reg',
                'system_scale'):
            value = _scalar_float(getattr(self, name), name)
            if value < 0:
                raise ValueError(f'`{name}` should be non-negative')
            if (name == 'residual_relative') and (value != value):
                raise ValueError('`residual_relative` should not be NaN')
            if (name != 'residual_relative') and (not isfinite(value)):
                raise ValueError(f'`{name}` should be finite')
            object.__setattr__(self, name, value)
        if self.l2_reg_mode not in ('absolute', 'relative'):
            raise ValueError(
                "`l2_reg_mode` should be 'absolute' or 'relative'")
        if not isinstance(self.column_scaling, bool):
            raise TypeError('`column_scaling` should be bool type')
        if not isinstance(self.system_scaling, bool):
            raise TypeError('`system_scaling` should be bool type')
        if self.site is not None:
            if isinstance(self.site, int) and not isinstance(self.site, bool):
                valid_site = self.site >= 0
            elif isinstance(self.site, tuple):
                valid_site = bool(self.site) and all(
                    isinstance(item, int) and item >= 0 for item in self.site)
            else:
                valid_site = False
            if not valid_site:
                raise ValueError(
                    '`site` should be a non-negative int or tuple of ints')
        if self.sweep is not None:
            if isinstance(self.sweep, bool) or \
                    (not isinstance(self.sweep, int)) or (self.sweep < 0):
                raise ValueError('`sweep` should be a non-negative integer')


@dataclass(frozen=True)
class SweepRecord:
    """Stores objective metrics measured once at the end of an ALS sweep."""

    sweep: int
    absolute_error: Optional[float] = None
    relative_error: Optional[float] = None
    relative_change: Optional[float] = None
    elapsed: Optional[float] = None
    sample_generation: Optional[int] = None
    stop_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.sweep, bool) or \
                (not isinstance(self.sweep, int)) or (self.sweep < 0):
            raise ValueError('`sweep` should be a non-negative integer')
        for name in (
                'absolute_error', 'relative_error', 'relative_change',
                'elapsed'):
            value = getattr(self, name)
            if value is None:
                continue
            value = _scalar_float(value, name)
            if (value < 0) or (value != value):
                raise ValueError(f'`{name}` should be non-negative and not NaN')
            if (name in ('relative_change', 'elapsed')) and \
                    (not isfinite(value)):
                raise ValueError(f'`{name}` should be finite')
            object.__setattr__(self, name, value)
        if self.sample_generation is not None:
            if isinstance(self.sample_generation, bool) or \
                    (not isinstance(self.sample_generation, int)) or \
                    (self.sample_generation < 0):
                raise ValueError(
                    '`sample_generation` should be a non-negative integer')
        if (self.stop_reason is not None) and \
                (not isinstance(self.stop_reason, str)):
            raise TypeError('`stop_reason` should be str type')


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
    local_solves: List[LocalSolveRecord] = field(default_factory=list)
    sweeps: List[SweepRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.errors = list(self.errors)
        self.truncations = list(self.truncations)
        self.timings = list(self.timings)
        self.fidelities = list(self.fidelities)
        self.warnings = list(self.warnings)
        self.local_solves = list(self.local_solves)
        self.sweeps = list(self.sweeps)

        collections = (
            ('errors', self.errors, ErrorRecord),
            ('truncations', self.truncations, TruncationRecord),
            ('timings', self.timings, TimingRecord),
            ('fidelities', self.fidelities, FidelityRecord),
            ('local_solves', self.local_solves, LocalSolveRecord),
            ('sweeps', self.sweeps, SweepRecord),
        )
        for name, records, record_type in collections:
            if not all(isinstance(record, record_type) for record in records):
                raise TypeError(
                    f'`{name}` should contain {record_type.__name__} objects')
        if not all(isinstance(message, str) for message in self.warnings):
            raise TypeError('`warnings` should contain strings')

    def as_info(self) -> Dict[str, Any]:
        """Returns a dictionary suitable for functional ``return_info`` APIs."""
        info = {
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
        if self.local_solves:
            info['local_solves'] = [
                _record_as_dict(record) for record in self.local_solves
            ]
        if self.sweeps:
            info['sweeps'] = [
                _record_as_dict(record) for record in self.sweeps
            ]
        return info


__all__ = [
    'ErrorRecord',
    'TruncationRecord',
    'LocalSolveRecord',
    'SweepRecord',
    'TimingRecord',
    'FidelityRecord',
    'DecompositionMetrics',
]
