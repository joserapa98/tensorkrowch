"""
This script contains:

    Internal classes:
        * _SafeList

    Public classes:
        * RoundingInfo
        * SampleError
        * BlockLayout
        * SplitBlock
        * TensorFormat
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Union, Tuple, Sequence, Any

import torch


class _SafeList(list):
    """Fixed-length list that validates replacements through a callback."""

    def __init__(self, values: Any, on_change: Callable[[], None]) -> None:
        super().__init__(values)
        self._on_change = on_change

    def __setitem__(self, key: Any, value: Any) -> None:
        """Applies a replacement and restores the entries if validation fails."""
        previous = self[key]
        if isinstance(key, slice):
            value = list(value)
            if len(value) != len(previous):
                raise ValueError('Slice replacement should preserve length')

        super().__setitem__(key, value)
        try:
            self._on_change()
        except Exception:
            super().__setitem__(key, previous)
            raise

    def _structural_error(self, *args, **kwargs) -> None:
        """Rejects changes that bypass controlled structural replacement."""
        raise TypeError('Replace the complete container to change its structure')

    append = extend = insert = pop = remove = clear = _structural_error
    reverse = sort = __delitem__ = __iadd__ = __imul__ = _structural_error


@dataclass(frozen=True)
class RoundingInfo:
    """
    Information returned by ``TensorFormat1D.rounding(return_info=True)``.

    ``rank`` gives the resulting representation size, while
    ``discarded_sq_norm`` records the squared Frobenius error at each local
    truncation. ``error_bound`` combines these local errors into an absolute
    bound for the full tensor or matrix (including the closing-rank factor for
    rings). It is not a measured reconstruction error. ``bound_satisfied``
    compares that bound with the requested relative budget; ``False`` does not
    imply that the actual error exceeds the budget.

    For more detail on the error bound, see
    :meth:`~tensorkrowch.formats.formats1d.TensorFormat1D.rounding`.

    Parameters
    ----------
    rank : tuple[int, ...]
        Final bond ranks, including the closing bond for rings.
    discarded_sq_norm : tuple[torch.Tensor, ...]
        Discarded squared singular-value mass at each processed cut, resolved
        over structural batches.
    error_bound : torch.Tensor
        Absolute global Frobenius error bound, resolved over structural batches.
    bound_satisfied : bool or None
        Whether the bound meets ``rel_error`` for all structural batches;
        ``None`` when ``rel_error`` was not supplied.
    """

    rank: Tuple[int, ...]  # Final right-bond ranks
    discarded_sq_norm: Tuple[torch.Tensor, ...]  # Discarded energy at each cut
    error_bound: torch.Tensor  # Absolute Frobenius error bound, resolved by batch
    bound_satisfied: Optional[bool]  # Whether the requested relative budget was met


@dataclass(frozen=True)
class SampleError:
    """
    Stores sample errors while preserving tensor storage and autograd.

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


@dataclass(frozen=True)
class BlockLayout:
    """
    Original site dimensions and contiguous group sizes.

    Parameters
    ----------
    groups : tuple[int, ...]
        Number of original consecutive sites in each block.
    in_dim : tuple[int, ...]
        Original input dimension of every site.
    out_dim : tuple[int, ...], optional
        Original matrix output dimensions; ``None`` for vectors.
    """

    groups: Tuple[int, ...]  # Number of sites per block
    in_dim: Tuple[int, ...]  # Original input dimensions
    out_dim: Optional[Tuple[int, ...]] = None  # Original matrix output dimensions

    def __post_init__(self) -> None:
        """Validates block sizes and original site dimensions."""
        if not self.groups or any(isinstance(size, bool) or
                                  not isinstance(size, int) or
                                  size < 1 for size in self.groups):
            raise ValueError('Block sizes should be positive integers')
        if sum(self.groups) != len(self.in_dim):
            raise ValueError(
                'Block sizes should cover the original input dimensions')
        if self.out_dim is not None and (len(self.out_dim) != len(self.in_dim)):
            raise ValueError('Original matrix input/output sites should match')


@dataclass(frozen=True)
class SplitBlock:
    """
    Local raw cores with open external ranks and optional internal factors.

    Parameters
    ----------
    cores : tuple[torch.Tensor, ...]
        Local cores in standard ``(*batch, left, physical, right)`` layout with
        physical axes fused for matrices.
    bonds : sequence[torch.Tensor or None] or None
        Factors internal to the local block; external interface factors are
        excluded. ``None`` denotes an identity factor or, for the complete
        sequence, the absence of explicit factors.
    spectra : tuple[torch.Tensor, ...]
        Singular values at the local SVD cuts. They are not certified global
        Schmidt spectra.
    """

    cores: Tuple[torch.Tensor, ...]  # Standard fused core layout
    bonds: Optional[Sequence[Optional[torch.Tensor]]]  # Internal bond factors only
    spectra: Tuple[torch.Tensor, ...]  # Singular values of the local cuts


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
    def to(self,
           device: Optional[Union[str, torch.device]] = None,
           dtype: Optional[torch.dtype] = None,
           copy: bool = False) -> 'TensorFormat':
        """
        Returns a device/dtype conversion, preserving the concrete format.

        PyTorch device errors propagate without a CPU fallback. Autograd is
        retained.

        Parameters
        ----------
        device : str or torch.device, optional
            Target device. ``None`` preserves the current device.
        dtype : torch.dtype, optional
            Target dtype. ``None`` preserves the current dtype. Coordinate
            grids and Schmidt spectra remain real when cores are complex.
        copy : bool
            If ``True``, copies tensors even when device and dtype are unchanged.
            If ``False``, an unchanged conversion may return ``self``.

        Returns
        -------
        TensorFormat
            Converted format; ``self`` when no conversion is needed and ``copy``
            is ``False``.

        Examples
        --------
        >>> format = tk.formats.TT([torch.ones(2)])
        >>> format.to() is format
        True
        >>> format.to(dtype=torch.float64).dtype
        torch.float64
        """

    def cpu(self) -> 'TensorFormat':
        """
        Returns the format on CPU.

        Returns
        -------
        TensorFormat
            Converted format, or ``self`` when already on the target device. No
            fallback device is selected.
        """
        return self.to(device='cpu')

    def cuda(self,
             device: Optional[Union[int, str, torch.device]] = None
             ) -> 'TensorFormat':
        """
        Returns the format on a CUDA device.

        Parameters
        ----------
        device : int, str or torch.device, optional
            CUDA device index or device specification. ``None`` selects the
            default CUDA device.

        Returns
        -------
        TensorFormat
            Converted format, preserving autograd. Unsupported device
            operations propagate PyTorch errors.
        """
        target = 'cuda' if device is None else (
            torch.device('cuda', device) if isinstance(device, int) else device)
        target = torch.device(target)
        if target.type != 'cuda':
            raise ValueError('`device` should select a CUDA device')
        return self.to(device=target)

    def mps(self) -> 'TensorFormat':
        """
        Returns the format on MPS.

        Returns
        -------
        TensorFormat
            Converted format, or ``self`` when already on the target device. No
            fallback device is selected.
        """
        return self.to(device='mps')

    @abstractmethod
    def clone(self) -> 'TensorFormat':
        """
        Clones the structural tensors, preserving autograd.

        Returns
        -------
        TensorFormat
            Independent tensor storage with the same represented tensor.
        """

    @abstractmethod
    def detach(self) -> 'TensorFormat':
        """
        Returns a detached format sharing tensor storage.

        Returns
        -------
        TensorFormat
            Separate containers with detached tensor references. Value edits to
            shared storage affect both formats.
        """

    @abstractmethod
    def detach_(self) -> 'TensorFormat':
        """
        Detaches structural tensors in-place by replacing references.

        Returns
        -------
        TensorFormat
            The current format. Tensor shapes and canonical metadata are
            preserved.
        """
