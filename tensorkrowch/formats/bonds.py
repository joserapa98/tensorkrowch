"""
This script contains:

    Public classes:
        * BondFactors1D
        * VidalGauge
"""

from typing import Callable, Sequence, Tuple

import torch

from tensorkrowch.formats.base import _SafeList


class BondFactors1D:
    """
    Optional diagonals between adjacent cores, possibly including a cyclic closure.

    Created by the owning format from a sequence of factors, sharing tensors.
    Supply factors through ``format.bonds`` or the format constructor.
    Element and same-length slice replacements validate against its cores
    immediately. Invalid replacements leave the container unchanged.

    Parameters
    ----------
    values : sequence[torch.Tensor or None]
        One diagonal per stored bond. A diagonal has shape ``(rank,)`` or
        ``(*core_batch, rank)``. ``None`` denotes an identity without allocation.
    on_change : callable
        Owning format callback, called after manual replacement.
    """

    def __init__(self,
                 values: Sequence[torch.Tensor],
                 on_change: Callable[[], None]) -> None:
        if isinstance(values, torch.Tensor):
            raise TypeError('`values` should be a sequence of diagonals')
        self._on_change = on_change
        self._values = _SafeList(values, self._on_sequence_changed)

    @property
    def values(self):
        """Diagonal factors at the bonds of a :class:`1D format <TensorFormat1D>`."""
        return self._values

    @values.setter
    def values(self, values: Sequence[torch.Tensor]) -> None:
        if isinstance(values, torch.Tensor):
            raise TypeError('`values` should be a sequence of diagonals')
        previous = getattr(self, '_values', None)
        self._values = _SafeList(values, self._on_sequence_changed)
        try:
            self._on_sequence_changed()
        except Exception:
            self._values = previous
            raise

    def _on_sequence_changed(self):
        """Validates factors and notifies the format."""
        if not all(value is None or isinstance(value, torch.Tensor)
                   for value in self._values):
            raise TypeError('Bond factors should be tensors or None')
        self._on_change()

    def validate(self, cores: Sequence[torch.Tensor], cyclic: bool) -> None:
        """
        Checks compatibility of diagonal factors with standard fused cores.

        Raises TypeError or ValueError for incompatible factor count, shapes,
        runtime or dtype. This validates stored factors, not the canonical
        interpretation of Vidal spectra.

        Parameters
        ----------
        cores : sequence of torch.Tensor
            Standard cores with shape (*core_batch, left, physical, right).
            Matrix physical dimensions should already be fused.
        cyclic : bool
            Whether the last core closes onto the first. Cyclic formats require
            one factor per core; open formats require one fewer.
        """
        count = len(cores) if cyclic else len(cores) - 1
        if len(self._values) != count:
            raise ValueError('There should be one factor per bond')

        for core, value in zip(cores, self._values):
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError('Bond factors should be tensors or None')

            batch = core.shape[:-3]
            if value.shape not in (torch.Size([core.shape[-1]]),
                                   torch.Size((*batch, core.shape[-1]))):
                raise ValueError('Bond factor dimensions should match the cores')
            if value.device != core.device:
                raise ValueError('Bond factors and cores should share device')
            if torch.promote_types(value.dtype, core.dtype) != core.dtype:
                raise ValueError('Bond factor dtype should be compatible with cores')

    def _map_tensors(self,
                     function: Callable[[torch.Tensor], torch.Tensor],
                     on_change: Callable[[], None]) -> 'BondFactors1D':
        """Maps factors into a container bound to the destination format."""
        values = [None if value is None else function(value) for value in self._values]
        return BondFactors1D(values, on_change)


class VidalGauge(BondFactors1D):
    r"""
    Schmidt spectra and their current absorption powers into neighbouring cores.
    That is, ``powers`` represents how :math:`\Lambda` factors are split into
    a product of powers of it that are then absorbed in their neighbours.

    Powers (0, 0), (0.5, 0.5), and (1, 1) represent explicit, implicit and
    inverse Vidal forms, respectively. The remaining bond factor has power
    ``1 - left_power - right_power``. Spectra are real and non-negative.

    Manual element or slice replacement of factors, spectra or powers
    invalidates the Vidal flag. Such records remain usable as stored diagonal
    factors; no automatic consistency repair is attempted.

    Parameters
    ----------
    values : sequence of torch.Tensor or None
        Stored diagonal factors, preserved independently of the Vidal metadata.
    spectra : sequence of torch.Tensor
        Real, finite, non-negative singular-value vectors, optionally
        carrying structural batches.
    powers : sequence of tuple[float, float]
        Absorption powers at each bond: (0, 0), (0.5, 0.5), (1, 1), (1, 0)
        or (0, 1). Inverse powers require strictly nonzero spectra.
    on_change : callable
        Owning format callback, called after manual replacement.
    valid : bool
        Whether spectra and powers still describe the stored cores and factors.
    """

    def __init__(self,
                 values: Sequence[torch.Tensor],
                 spectra: Sequence[torch.Tensor],
                 powers: Sequence[Tuple[float, float]],
                 on_change: Callable[[], None],
                 valid: bool = True) -> None:
        super().__init__(values, on_change)
        self._spectra = _SafeList(spectra, self._on_sequence_changed)
        self._powers = _SafeList(powers, self._on_sequence_changed)
        self._valid = valid

    @property
    def spectra(self):
        """Stored Schmidt spectra; manual replacement invalidates Vidal state."""
        return self._spectra

    @property
    def powers(self):
        """Absorption powers; manual replacement invalidates Vidal state."""
        return self._powers

    def _on_sequence_changed(self):
        """Validates factors, notifies the format and invalidates Vidal state."""
        super()._on_sequence_changed()
        self._valid = False

    def _map_tensors(self,
                     function: Callable[[torch.Tensor], torch.Tensor],
                     on_change: Callable[[], None]) -> 'VidalGauge':
        """Maps factors and spectra, preserving their stored interpretation."""
        values = [None if value is None else function(value) for value in self._values]
        spectra = []
        for spectrum in self._spectra:
            mapped = function(spectrum)
            spectra.append(mapped.real if mapped.is_complex() else mapped)
        return VidalGauge(values, spectra, self._powers, on_change, valid=self._valid)
