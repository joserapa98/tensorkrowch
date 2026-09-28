"""Topology-neutral interfaces for compact raw-tensor representations."""

from typing import NamedTuple, Optional, Union
from abc import ABC, abstractmethod

import torch


class SampleError(NamedTuple):
    """Numerical sample error, without decomposition provenance or CPU copies."""

    kind: str
    absolute: torch.Tensor
    relative: Optional[torch.Tensor] = None
    size: Optional[int] = None
    denominator: Optional[torch.Tensor] = None


class TensorFormat(ABC):
    """Geometry-neutral raw-tensor representation, without graph machinery."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device of the represented tensors."""

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Dtype of the represented tensors."""

    @abstractmethod
    def to(self, device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None, copy: bool = False):
        """Returns a device/dtype conversion."""

    def cpu(self):
        """Returns the format on CPU."""
        return self.to(device='cpu')

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None):
        """Returns the format on the selected CUDA device."""
        target = 'cuda' if device is None else (
            torch.device('cuda', device) if isinstance(device, int) else device)
        target = torch.device(target)
        if target.type != 'cuda':
            raise ValueError('`device` should select a CUDA device')
        return self.to(device=target)

    def mps(self):
        """Returns the format on MPS, without a CPU fallback."""
        return self.to(device='mps')

    @abstractmethod
    def copy(self):
        """Returns an independent copy of all structural tensors."""

    @abstractmethod
    def detach(self):
        """Returns a detached format sharing tensor storage."""

    @abstractmethod
    def detach_(self):
        """Detaches every structural tensor in this container."""


class TensorFormat2D(TensorFormat, ABC):
    """Reserved 2D interface; PEPS/PEPO implementations are deferred."""
