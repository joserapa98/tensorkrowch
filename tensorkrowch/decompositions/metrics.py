"""
This script contains:

    Metric records:
        * ErrorRecord
        * TruncationRecord
        * TimingRecord
        * FidelityRecord
        * LocalSolveRecord
        * GaugeRecord
        * SweepRecord
        * EvaluationStats
        * InputFitRecord
        * RangeProjectionRecord

    Class for decomposition metrics:
        * DecompositionMetrics
"""

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


def _cpu_tensor(value: Any, name: str) -> torch.Tensor:
    """Converts a numerical metric to a detached CPU tensor."""
    if not isinstance(value, torch.Tensor):
        if not isinstance(value, (int, float)):
            raise TypeError(f'`{name}` should be torch.Tensor type')
        value = torch.as_tensor(value)
    return value.detach().cpu()


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
                   reference_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
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
class ErrorRecord:  # MARK: ErrorRecord
    """
    Stores an ``absolute``/``relative`` error measured on a specified target.
    """

    kind: str  # Target or mechanism on which the error was measured
    absolute: torch.Tensor  # Absolute error, optionally resolved by batch
    relative: Optional[torch.Tensor] = None  # Relative error with same shape
    size: Optional[int] = None  # Number of contributions represented
    denominator: Optional[torch.Tensor] = None  # Norm used for relative error

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise TypeError('`kind` should be str type')

        absolute = _cpu_tensor(self.absolute, 'absolute')
        if torch.any(absolute < 0):
            raise ValueError('`absolute` should be non-negative')
        object.__setattr__(self, 'absolute', absolute)

        if self.relative is not None:
            relative = _cpu_tensor(self.relative, 'relative')
            if relative.shape != absolute.shape:
                raise ValueError(
                    '`relative` should have the same shape as `absolute`')
            if torch.any(relative < 0):
                raise ValueError('`relative` should be non-negative')
            object.__setattr__(self, 'relative', relative)

        if self.size is not None:
            if not isinstance(self.size, int):
                raise TypeError('`size` should be int type')
            if self.size < 0:
                raise ValueError('`size` should be non-negative')

        if self.denominator is not None:
            denominator = _cpu_tensor(self.denominator, 'denominator')
            if denominator.shape != absolute.shape:
                raise ValueError(
                    '`denominator` should have the same shape as `absolute`')
            if torch.any(denominator < 0):
                raise ValueError('`denominator` should be non-negative')
            object.__setattr__(self, 'denominator', denominator)


@dataclass(frozen=True)
class TruncationRecord:  # MARK: TruncationRecord
    """Stores ranks and discarded energy for one truncation cut."""

    site: int  # Site immediately to the left of the truncation cut
    full_rank: int  # Rank available before truncation
    selected_rank: int  # Rank retained after truncation
    local_abs_error: torch.Tensor  # Discarded norm at the local cut
    local_rel_error: Optional[torch.Tensor] = None  # Error over local norm
    singular_values: Optional[torch.Tensor] = None  # Optional retained spectrum
    local_norm: Optional[torch.Tensor] = None  # Norm entering the local SVD
    discarded_sq_norm: Optional[torch.Tensor] = None  # Discarded SVD energy
    log_scale: Optional[torch.Tensor] = None  # Restored logarithmic scale
    global_rel_contribution: Optional[torch.Tensor] = None  # Global error term
    svd_method: Optional[str] = None  # Compact SVD implementation used
    phase: Optional[str] = None  # Algorithmic phase containing this cut

    @classmethod
    def from_svd_info(cls,
                      info: Any,
                      site: int,
                      log_scale: Optional[torch.Tensor] = None,
                      global_norm: Optional[torch.Tensor] = None,
                      singular_values: Optional[torch.Tensor] = None) -> 'TruncationRecord':
        """Builds a high-level record from ``_TruncatedSVDInfo``."""
        local_log_norm = info.total_sq_norm.log() / 2
        discarded_log_norm = info.discarded_sq_norm.log() / 2
        if log_scale is None:
            log_scale_tensor = torch.zeros_like(local_log_norm)
        else:
            if not isinstance(log_scale, torch.Tensor):
                raise TypeError('`log_scale` should be torch.Tensor type')
            log_scale_tensor = log_scale.to(
                device=local_log_norm.device, dtype=local_log_norm.dtype)
            if log_scale_tensor.shape != local_log_norm.shape:
                raise ValueError('`log_scale` should match the SVD batch shape')
            if not torch.isfinite(log_scale_tensor).all():
                raise ValueError('`log_scale` should be finite')

        local_log_norm = local_log_norm + log_scale_tensor
        discarded_log_norm = discarded_log_norm + log_scale_tensor

        if global_norm is not None:
            if not isinstance(global_norm, torch.Tensor):
                raise TypeError('`global_norm` should be torch.Tensor type')
            global_norm = global_norm.to(
                device=local_log_norm.device, dtype=local_log_norm.dtype)
            if global_norm.shape != local_log_norm.shape:
                raise ValueError('`global_norm` should match the SVD batch shape')
            if torch.any(global_norm < 0) or \
                    (not torch.isfinite(global_norm).all()):
                raise ValueError(
                    '`global_norm` should be finite and non-negative')

        local_norm = _norm_from_log(local_log_norm, global_norm)
        local_abs_error = _norm_from_log(discarded_log_norm, global_norm)
        local_rel_error = _ratio_from_log_norms(
            discarded_log_norm, local_log_norm)

        if singular_values is not None:
            if not isinstance(singular_values, torch.Tensor):
                raise TypeError('`singular_values` should be torch.Tensor type')
            if singular_values.shape[:-1] != local_log_norm.shape:
                raise ValueError(
                    '`singular_values` should match the SVD batch shape')
            singular_values = singular_values.to(
                device=local_log_norm.device, dtype=local_log_norm.dtype)
            singular_log = (
                singular_values.log() + log_scale_tensor.unsqueeze(-1))
            singular_reference = None if global_norm is None \
                else global_norm.unsqueeze(-1)
            singular_values = _norm_from_log(
                singular_log, singular_reference)

        global_rel_contribution = None
        if global_norm is not None:
            global_rel_contribution = _ratio_from_log_norms(
                discarded_log_norm, global_norm.log())

        return cls(
            site=site,
            full_rank=info.full_rank,
            selected_rank=info.selected_rank,
            local_abs_error=local_abs_error,
            local_rel_error=local_rel_error,
            singular_values=singular_values,
            local_norm=local_norm,
            log_scale=(log_scale_tensor if log_scale is not None else None),
            global_rel_contribution=global_rel_contribution,
            svd_method=info.svd_method,
        )

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
            'local_abs_error',
            'local_rel_error',
            'singular_values',
            'local_norm',
            'global_rel_contribution',
        )
        for name in non_negative_fields:
            value = getattr(self, name)
            if value is None:
                continue
            value = _cpu_tensor(value, name)
            if torch.any(value < 0):
                raise ValueError(f'`{name}` should be non-negative')
            object.__setattr__(self, name, value)

        discarded_sq_norm = self.discarded_sq_norm
        if discarded_sq_norm is None:
            discarded_sq_norm = self.local_abs_error.square()
        else:
            discarded_sq_norm = _cpu_tensor(
                discarded_sq_norm, 'discarded_sq_norm')
            if torch.any(discarded_sq_norm < 0):
                raise ValueError('`discarded_sq_norm` should be non-negative')
        object.__setattr__(self, 'discarded_sq_norm', discarded_sq_norm)

        if self.log_scale is not None:
            log_scale = _cpu_tensor(self.log_scale, 'log_scale')
            if not torch.isfinite(log_scale).all():
                raise ValueError('`log_scale` should be finite')
            object.__setattr__(self, 'log_scale', log_scale)

        if (self.svd_method is not None) and \
                (self.svd_method not in ('svd', 'qr_svd')):
            raise ValueError('`svd_method` should be "svd" or "qr_svd"')
        if (self.phase is not None) and (not isinstance(self.phase, str)):
            raise TypeError('`phase` should be str type')


@dataclass(frozen=True)
class LocalSolveRecord:  # MARK: LocalSolveRecord
    """Stores diagnostics for one local least-squares solve."""

    environment_shape: Tuple[int, int]  # Shape of the original local design matrix
    target_shape: Tuple[int, ...]  # Shape of the original right-hand side
    driver: str  # Effective least-squares driver
    abs_residual: torch.Tensor  # Norm of the local residual
    rel_residual: torch.Tensor  # Residual norm divided by the target norm
    target_norm: torch.Tensor  # Norm used to normalize the local residual
    l2_reg: float = 0.0  # Configured regularization coefficient
    # Regularization after relative or environment rescaling
    effective_l2_reg: torch.Tensor = 0.0
    # Absolute or relative interpretation of regularization
    l2_reg_mode: str = 'absolute'
    column_scaling: bool = False  # Whether columns were balanced
    system_scaling: bool = False  # Whether the augmented system was scaled
    system_scale: torch.Tensor = 1.0  # Global divisor applied to the augmented system
    site: Optional[Any] = None  # Optional zero-based active site
    sweep: Optional[int] = None  # Zero-based directional sweep index
    # Whether the recorded proposal matches the current design
    sampling_exact: Optional[bool] = None
    sample_generation: Optional[int] = None  # Generation of the sampled rows

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
                'abs_residual',
                'rel_residual',
                'target_norm',
                'effective_l2_reg',
                'system_scale'):
            value = _cpu_tensor(getattr(self, name), name)
            if value.ndim or value.is_complex():
                raise ValueError(f'`{name}` should be a real scalar tensor')
            if value < 0:
                raise ValueError(f'`{name}` should be non-negative')
            if (name == 'rel_residual') and (value != value):
                raise ValueError('`rel_residual` should not be NaN')
            if (name != 'rel_residual') and (not torch.isfinite(value)):
                raise ValueError(f'`{name}` should be finite')
            object.__setattr__(self, name, value)
        l2_reg = _scalar_float(self.l2_reg, 'l2_reg')
        if l2_reg < 0 or not isfinite(l2_reg):
            raise ValueError('`l2_reg` should be finite and non-negative')
        object.__setattr__(self, 'l2_reg', l2_reg)
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
        if (self.sampling_exact is not None) and \
                (not isinstance(self.sampling_exact, bool)):
            raise TypeError('`sampling_exact` should be bool type or None')
        if self.sample_generation is not None:
            if isinstance(self.sample_generation, bool) or \
                    (not isinstance(self.sample_generation, int)) or \
                    (self.sample_generation < 0):
                raise ValueError(
                    '`sample_generation` should be a non-negative integer')


@dataclass(frozen=True)
class InputFitRecord:  # MARK: InputFitRecord
    """Stores diagnostics for fitting one sampled Phi input ``axis``."""

    method: str
    axis: int
    domain_size: int
    in_dim: int
    abs_residual: float
    rel_residual: float
    condition_number: float
    used_fibers: bool = False
    local_solve: Optional[LocalSolveRecord] = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise TypeError('`method` should be a non-empty string')
        for name in ('axis', 'domain_size', 'in_dim'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            if value < (0 if name == 'axis' else 1):
                qualifier = 'non-negative' if name == 'axis' else 'positive'
                raise ValueError(f'`{name}` should be {qualifier}')
        for name in (
                'abs_residual',
                'rel_residual',
                'condition_number'):
            value = _scalar_float(getattr(self, name), name)
            if (value < 0) or (value != value):
                raise ValueError(f'`{name}` should be non-negative and not NaN')
            if (name == 'abs_residual') and (not isfinite(value)):
                raise ValueError('`abs_residual` should be finite')
            object.__setattr__(self, name, value)
        if not isinstance(self.used_fibers, bool):
            raise TypeError('`used_fibers` should be bool type')
        if self.local_solve is not None and \
                not isinstance(self.local_solve, LocalSolveRecord):
            raise TypeError(
                '`local_solve` should be LocalSolveRecord type or None')


@dataclass(frozen=True)
class RangeProjectionRecord:  # MARK: RangeProjectionRecord
    """
    Stores dimensions, approximation error and cost of a range projection.
    """

    method: str
    input_shape: Tuple[int, int]
    axis: int
    requested_dim: Optional[int]
    projection_dim: int
    range_dim: int
    oversampling: int = 0
    n_power_iter: int = 0
    error_absolute: Optional[float] = None
    error_relative: Optional[float] = None
    elapsed: Optional[float] = None

    def __post_init__(self) -> None:
        if self.method not in ('identity', 'randomized'):
            raise ValueError(
                "`method` should be 'identity' or 'randomized'")
        input_shape = tuple(self.input_shape)
        if len(input_shape) != 2 or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
                for dim in input_shape):
            raise ValueError(
                '`input_shape` should contain two positive integers')
        object.__setattr__(self, 'input_shape', input_shape)
        for name in ('axis', 'projection_dim', 'range_dim', 'oversampling',
                     'n_power_iter'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            minimum = 1 if name in ('projection_dim', 'range_dim') else 0
            if value < minimum:
                qualifier = 'positive' if minimum else 'non-negative'
                raise ValueError(f'`{name}` should be {qualifier}')
        if self.requested_dim is not None:
            if isinstance(self.requested_dim, bool) or \
                    not isinstance(self.requested_dim, int):
                raise TypeError('`requested_dim` should be int type or None')
            if self.requested_dim < 1:
                raise ValueError('`requested_dim` should be positive')
        for name in ('error_absolute', 'error_relative', 'elapsed'):
            value = getattr(self, name)
            if value is None:
                continue
            value = _scalar_float(value, name)
            if (value < 0) or (value != value) or (not isfinite(value)):
                raise ValueError(f'`{name}` should be finite and non-negative')
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class GaugeRecord:  # MARK: GaugeRecord
    """Stores rank, conditioning and cancellation diagnostics for one gauge."""

    orientation: str  # Left or right interpretation of the gauge axes
    shape: Tuple[int, int]  # Shape of the original matricized gauge
    numerical_rank: int  # Numerical rank at the specified tolerance
    cancellable_rank: int  # Number of columns required for exact cancellation
    condition_number: torch.Tensor  # Estimated spectral condition number
    cancellation_error: torch.Tensor  # Relative error of the directional cancellation
    projective: bool  # Whether the map cancels only a projected subspace
    inverse_method: str  # Effective method used to construct the directional dual
    tolerance: float  # Configured cancellation-error tolerance
    # Absolute singular-value threshold used for numerical rank
    rank_tolerance: torch.Tensor
    site: Optional[int] = None  # Optional zero-based active site

    def __post_init__(self) -> None:
        if self.orientation not in ('left', 'right'):
            raise ValueError("`orientation` should be 'left' or 'right'")
        shape = tuple(self.shape)
        if len(shape) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) or
                value < 1 for value in shape):
            raise ValueError('`shape` should contain two positive integers')
        object.__setattr__(self, 'shape', shape)
        for name in ('numerical_rank', 'cancellable_rank'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            if value < 0:
                raise ValueError(f'`{name}` should be non-negative')
        if self.numerical_rank > min(shape):
            raise ValueError('`numerical_rank` exceeds the matrix dimensions')
        if self.cancellable_rank != shape[1]:
            raise ValueError(
                '`cancellable_rank` should equal the number of columns')
        for name in (
                'condition_number',
                'cancellation_error',
                'rank_tolerance'):
            value = _cpu_tensor(getattr(self, name), name)
            if value.ndim or value.is_complex():
                raise ValueError(f'`{name}` should be a real scalar tensor')
            if (value < 0) or (value != value):
                raise ValueError(f'`{name}` should be non-negative and not NaN')
            if name != 'condition_number' and not torch.isfinite(value):
                raise ValueError(f'`{name}` should be finite')
            object.__setattr__(self, name, value)
        tolerance = _scalar_float(self.tolerance, 'tolerance')
        if tolerance < 0 or not isfinite(tolerance):
            raise ValueError('`tolerance` should be finite and non-negative')
        object.__setattr__(self, 'tolerance', tolerance)
        if not isinstance(self.projective, bool):
            raise TypeError('`projective` should be bool type')
        if self.inverse_method not in ('solve', 'inverse', 'pinv'):
            raise ValueError(
                "`inverse_method` should be 'solve', 'inverse' or 'pinv'")
        if self.site is not None:
            if isinstance(self.site, bool) or not isinstance(self.site, int):
                raise TypeError('`site` should be int type or None')
            if self.site < 0:
                raise ValueError('`site` should be non-negative')

    @property
    def cancellable(self) -> bool:
        """Whether the measured cancellation lies within its tolerance."""
        return bool(self.cancellation_error <= self.tolerance)


@dataclass(frozen=True)
class SweepRecord:  # MARK: SweepRecord
    """
    Stores objective metrics measured once at the end of an ALS ``sweep``.
    """

    sweep: int  # Zero-based directional sweep index
    # Absolute fixed-objective error at the end of the sweep
    abs_error: Optional[torch.Tensor] = None
    # Relative fixed-objective error at the end of the sweep
    rel_error: Optional[torch.Tensor] = None
    # Relative change since the preceding complete sweep
    rel_change: Optional[torch.Tensor] = None
    elapsed: Optional[float] = None  # Elapsed wall-clock time in seconds
    sample_generation: Optional[int] = None  # Generation of the sampled rows
    stop_reason: Optional[str] = None  # Normalized reason for stopping

    def __post_init__(self) -> None:
        if isinstance(self.sweep, bool) or \
                (not isinstance(self.sweep, int)) or (self.sweep < 0):
            raise ValueError('`sweep` should be a non-negative integer')
        for name in (
                'abs_error', 'rel_error', 'rel_change'):
            value = getattr(self, name)
            if value is None:
                continue
            value = _cpu_tensor(value, name)
            if value.ndim or value.is_complex():
                raise ValueError(f'`{name}` should be a real scalar tensor')
            if (value < 0) or (value != value):
                raise ValueError(f'`{name}` should be non-negative and not NaN')
            if (name == 'rel_change') and \
                    (not torch.isfinite(value)):
                raise ValueError(f'`{name}` should be finite')
            object.__setattr__(self, name, value)
        if self.elapsed is not None:
            elapsed = _scalar_float(self.elapsed, 'elapsed')
            if elapsed < 0 or not isfinite(elapsed):
                raise ValueError('`elapsed` should be finite and non-negative')
            object.__setattr__(self, 'elapsed', elapsed)
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
class TimingRecord:  # MARK: TimingRecord
    """Stores ``elapsed`` time for a decomposition phase or ``site``."""

    name: str  # Timed phase or operation
    elapsed: float  # Elapsed wall-clock time in seconds
    site: Optional[int] = None  # Optional zero-based site or cut position
    worker: Optional[int] = None  # Optional distributed worker index
    children: Sequence['TimingRecord'] = field(default_factory=tuple)  # Nested timings

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
class EvaluationStats:  # MARK: EvaluationStats
    """Counts point evaluations performed by a tensor source or session."""

    requested_points: int = 0
    unique_points: int = 0
    batches: int = 0
    cache_hits: int = 0
    source_calls: int = 0

    def __post_init__(self) -> None:
        for name in (
                'requested_points', 'unique_points', 'batches', 'cache_hits',
                'source_calls'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            if value < 0:
                raise ValueError(f'`{name}` should be non-negative')

    def delta(self, previous: 'EvaluationStats') -> 'EvaluationStats':
        """Returns the non-negative counter increment from ``previous``."""
        if not isinstance(previous, EvaluationStats):
            raise TypeError('`previous` should be EvaluationStats type')
        values = {}
        for name in (
                'requested_points', 'unique_points', 'batches', 'cache_hits',
                'source_calls'):
            value = getattr(self, name) - getattr(previous, name)
            if value < 0:
                raise ValueError(
                    '`previous` counters should not exceed current counters')
            values[name] = value
        return EvaluationStats(**values)


@dataclass(frozen=True)
class FidelityRecord:  # MARK: FidelityRecord
    """Stores a phase-aware normalized overlap and its ``fidelity``."""

    # Complex normalized overlap with its original phase
    normalized_overlap: torch.Tensor
    error: Optional[ErrorRecord] = None  # Optional final reconstruction error
    # Squared magnitude of the normalized overlap
    fidelity: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        overlap = self.normalized_overlap
        if not isinstance(overlap, torch.Tensor):
            if not isinstance(overlap, (int, float, complex)):
                raise TypeError('`normalized_overlap` should be a numeric tensor')
            overlap = torch.tensor(overlap, dtype=torch.complex128)
        overlap = overlap.detach().cpu()
        if overlap.ndim:
            raise ValueError('`normalized_overlap` should be a scalar tensor')
        if not torch.isfinite(overlap):
            raise ValueError('`normalized_overlap` should be finite')
        object.__setattr__(self, 'normalized_overlap', overlap)
        object.__setattr__(self, 'fidelity', overlap.abs().square())

        if (self.error is not None) and \
                (not isinstance(self.error, ErrorRecord)):
            raise TypeError('`error` should be ErrorRecord type')


@dataclass
class DecompositionMetrics:  # MARK: DecompositionMetrics
    """Collects structured records produced during a decomposition fit."""

    errors: List[ErrorRecord] = field(default_factory=list)  # Global errors
    # Local SVD truncations
    truncations: List[TruncationRecord] = field(default_factory=list)
    timings: List[TimingRecord] = field(default_factory=list)  # Runtime data
    # Tensor-source evaluations
    evaluations: List[EvaluationStats] = field(default_factory=list)
    # Normalized overlaps and fidelities
    fidelities: List[FidelityRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)  # Diagnostic warnings
    # Local ALS solve diagnostics
    local_solves: List[LocalSolveRecord] = field(default_factory=list)
    # Input-axis fitting diagnostics
    input_fits: List[InputFitRecord] = field(default_factory=list)
    # Recursive range-projection diagnostics
    range_projections: List[RangeProjectionRecord] = field(
        default_factory=list)
    gauges: List[GaugeRecord] = field(default_factory=list)  # Gauge data
    sweeps: List[SweepRecord] = field(default_factory=list)  # ALS sweeps

    def __post_init__(self) -> None:
        self.errors = list(self.errors)
        self.truncations = list(self.truncations)
        self.timings = list(self.timings)
        self.evaluations = list(self.evaluations)
        self.fidelities = list(self.fidelities)
        self.warnings = list(self.warnings)
        self.local_solves = list(self.local_solves)
        self.input_fits = list(self.input_fits)
        self.range_projections = list(self.range_projections)
        self.gauges = list(self.gauges)
        self.sweeps = list(self.sweeps)

        collections = (
            ('errors', self.errors, ErrorRecord),
            ('truncations', self.truncations, TruncationRecord),
            ('timings', self.timings, TimingRecord),
            ('evaluations', self.evaluations, EvaluationStats),
            ('fidelities', self.fidelities, FidelityRecord),
            ('local_solves', self.local_solves, LocalSolveRecord),
            ('input_fits', self.input_fits, InputFitRecord),
            ('range_projections', self.range_projections,
             RangeProjectionRecord),
            ('gauges', self.gauges, GaugeRecord),
            ('sweeps', self.sweeps, SweepRecord),
        )
        for name, records, record_type in collections:
            if not all(isinstance(record, record_type) for record in records):
                raise TypeError(
                    f'`{name}` should contain {record_type.__name__} objects')
        if not all(isinstance(message, str) for message in self.warnings):
            raise TypeError('`warnings` should contain strings')

    def as_info(self) -> Dict[str, Any]:
        """
        Returns a dictionary suitable for functional ``return_info`` APIs.
        """
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
        if self.input_fits:
            info['input_fits'] = [
                _record_as_dict(record) for record in self.input_fits
            ]
        if self.range_projections:
            info['range_projections'] = [
                _record_as_dict(record) for record in self.range_projections
            ]
        if self.evaluations:
            info['evaluations'] = [
                _record_as_dict(record) for record in self.evaluations
            ]
        if self.gauges:
            info['gauges'] = [
                _record_as_dict(record) for record in self.gauges
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
    'InputFitRecord',
    'RangeProjectionRecord',
    'GaugeRecord',
    'SweepRecord',
    'TimingRecord',
    'EvaluationStats',
    'FidelityRecord',
    'DecompositionMetrics',
]
