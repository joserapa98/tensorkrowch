"""Topology-neutral interfaces for compact raw-tensor representations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class SampleError:
    r"""Stores sample errors while preserving tensor storage and autograd.

    Frozen record: field references cannot be reassigned. Tensor contents and
    autograd are preserved without copying or detaching.

    Parameters
    ----------
    kind : str
        Target on which error was measured.
    absolute : torch.Tensor
        Absolute norm error, retaining tensor storage and autograd.
    relative : torch.Tensor, optional
        Relative norm error, when provided.
    size : int, optional
        Number of represented sample contributions.
    denominator : torch.Tensor, optional
        Norm used to normalize relative error.
    """

    kind: str  # Target on which the error was measured
    absolute: torch.Tensor  # Absolute error, optionally resolved by batch
    relative: Optional[torch.Tensor] = None  # Relative error with same shape
    size: Optional[int] = None  # Number of contributions represented
    denominator: Optional[torch.Tensor] = None  # Norm used for relative error


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
        r"""Returns a device/dtype conversion, preserving the concrete format.

        PyTorch device errors propagate without a CPU fallback. Autograd is
        retained.

        Parameters
        ----------
        device : str or torch.device, optional
            Target device. None preserves the current device.
        dtype : torch.dtype, optional
            Target dtype. None preserves the current dtype. Coordinate grids and
            Schmidt spectra remain real when cores are complex.
        copy : bool
            If True, copies tensors even when device and dtype are unchanged. If
            False, an unchanged conversion may return self.

        Returns
        -------
        TensorFormat
            Converted format; self when no conversion is needed and copy is
            False.

        Examples
        --------
        >>> format = tk.formats.TT([torch.ones(2)])
        >>> format.to() is format
        True
        >>> format.to(dtype=torch.float64).dtype == torch.float64
        True
        """

    def cpu(self):
        r"""Returns the format on CPU.

        Returns
        -------
        TensorFormat
            Converted format, or self when already on the target device. No
            fallback device is selected.
        """
        return self.to(device='cpu')

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None):
        r"""Returns the format on a CUDA device.

        Parameters
        ----------
        device : int, str or torch.device, optional
            CUDA device index or device specification. None selects the default
            CUDA device.

        Returns
        -------
        TensorFormat
            Converted format, preserving autograd. Unsupported device operations
            propagate PyTorch errors.
        """
        target = 'cuda' if device is None else (
            torch.device('cuda', device) if isinstance(device, int) else device)
        target = torch.device(target)
        if target.type != 'cuda':
            raise ValueError('`device` should select a CUDA device')
        return self.to(device=target)

    def mps(self):
        r"""Returns the format on MPS.

        Returns
        -------
        TensorFormat
            Converted format, or self when already on the target device. No
            fallback device is selected.
        """
        return self.to(device='mps')

    @abstractmethod
    def clone(self):
        r"""Clones the structural tensors, preserving autograd.

        Returns
        -------
        TensorFormat
            Independent tensor storage with the same represented tensor.
        """

    @abstractmethod
    def detach(self):
        r"""Returns a detached format sharing tensor storage.

        Returns
        -------
        TensorFormat
            Separate containers with detached tensor references. Value edits to
            shared storage affect both formats.
        """

    @abstractmethod
    def detach_(self):
        r"""Detaches structural tensors in-place by replacing references.

        Returns
        -------
        TensorFormat
            The current format. Tensor shapes and canonical metadata are
            preserved.
        """


class TensorFormat2D(TensorFormat, ABC):
    """Reserved 2D interface; PEPS/PEPO implementations are deferred."""
