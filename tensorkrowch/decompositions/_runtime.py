"""
This script contains:

    Internal runtime classes:
        * _RuntimeTimer
        * _RuntimePolicy
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Union

import torch


Device = Optional[Union[str, torch.device]]


class _RuntimeTimer:
    """Context manager that measures a runtime policy phase."""

    def __init__(self,
                 device: Optional[torch.device],
                 synchronize: bool) -> None:
        self._device = device
        self._synchronize = synchronize
        self.elapsed: Optional[float] = None
        self._start: Optional[float] = None

    def _sync(self) -> None:
        if not self._synchronize or (self._device is None):
            return
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)
        elif self._device.type == 'mps':
            torch.mps.synchronize()

    def __enter__(self) -> '_RuntimeTimer':
        self._sync()
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._start is None:
            raise RuntimeError('Runtime timer was not started')
        self._sync()
        self.elapsed = perf_counter() - self._start


@dataclass(frozen=True)
class _RuntimePolicy:
    """Normalizes active and final devices, dtype and synchronized timers."""

    device: Device = None
    out_device: Device = 'cpu'
    dtype: Optional[torch.dtype] = None
    synchronize_timers: bool = True

    def __post_init__(self) -> None:
        if self.device is not None:
            object.__setattr__(self, 'device', torch.device(self.device))
        if self.out_device is not None:
            object.__setattr__(
                self, 'out_device', torch.device(self.out_device))
        if (self.dtype is not None) and \
                (not isinstance(self.dtype, torch.dtype)):
            raise TypeError('`dtype` should be torch.dtype type')
        if not isinstance(self.synchronize_timers, bool):
            raise TypeError('`synchronize_timers` should be bool type')

    @classmethod
    def from_tensor(cls,
                    tensor: torch.Tensor,
                    device: Device = None,
                    out_device: Device = 'cpu',
                    dtype: Optional[torch.dtype] = None,
                    synchronize_timers: bool = True) -> '_RuntimePolicy':
        """Infers unspecified active runtime properties from a tensor."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        return cls(
            device=tensor.device if device is None else device,
            out_device=out_device,
            dtype=tensor.dtype if dtype is None else dtype,
            synchronize_timers=synchronize_timers)

    def prepare(self, tensor: torch.Tensor) -> torch.Tensor:
        """Moves an input tensor to the active device and dtype."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        device = tensor.device if self.device is None else self.device
        dtype = tensor.dtype if self.dtype is None else self.dtype
        return tensor.to(device=device, dtype=dtype)

    def finalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Moves a finalized tensor to the configured output device."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if self.out_device is None:
            return tensor
        return tensor.to(device=self.out_device)

    def timer(self, device: Device = None) -> _RuntimeTimer:
        """Returns a synchronized wall-clock timer for the active device."""
        timer_device = self.device if device is None else torch.device(device)
        return _RuntimeTimer(timer_device, self.synchronize_timers)


__all__ = ['_RuntimePolicy']
