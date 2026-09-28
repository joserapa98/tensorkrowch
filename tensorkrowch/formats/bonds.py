"""
This script contains:

    Public classes:
        * BondFactors1D
        * VidalGauge
"""

from copy import copy
from typing import Sequence

import torch

from tensorkrowch.formats.base import _SafeList


class BondFactors1D:
    """
    Optional diagonals between adjacent cores, including a cyclic closure.

    Each format owns a separate bond container, sharing the supplied tensors.
    Element and same-length slice replacements validate against its cores
    immediately. Invalid replacements leave the container unchanged.

    Parameters
    ----------
    values : sequence[torch.Tensor or None]
        One diagonal per stored bond. A diagonal has shape ``(rank,)`` or
        ``(*core_batch, rank)``. ``None`` denotes an identity without allocation.
    """

    def __init__(self, values) -> None:
        self._on_change = None
        self.values = values

    @property
    def values(self):
        """Diagonal factors at the bonds of a :class:`1D format <TensorFormat1D>`."""
        return self._values

    @values.setter
    def values(self, values):
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
        if self._on_change is not None:
            self._on_change()

    def _with_callback(self, on_change):
        """Copies bond lists and binds their callback, preserving tensors."""
        result = copy(self)
        result._on_change = on_change
        result._values = _SafeList(self._values, result._on_sequence_changed)
        return result

    def validate(self, cores: Sequence[torch.Tensor], cyclic: bool):
        r"""Checks compatibility of diagonal factors with standard fused cores.

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
        if len(self.values) != count:
            raise ValueError('There should be one factor per bond')
        for core, value in zip(cores, self.values):
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

    def _map_tensors(self, function):
        """Maps stored tensors while preserving concrete container semantics."""
        result = copy(self)
        result._on_change = None
        values = []
        for value in self.values:
            if value is None:
                values.append(None)
                continue
            mapped = function(value)
            if not value.is_complex() and mapped.is_complex():
                mapped = mapped.real
            values.append(mapped)
        result._values = _SafeList(values, result._on_sequence_changed)
        return result


class VidalGauge(BondFactors1D):
    r"""Schmidt spectra and their current absorption into neighbouring cores.

    Powers (0, 0), (0.5, 0.5), and (1, 1) represent explicit, implicit and
    inverse Vidal respectively. The remaining bond factor has power
    ``1 - left_power - right_power``. Spectra are real and non-negative.

    Manual element or slice replacement of factors, spectra or powers
    invalidates the Vidal flag. Such records remain usable as stored diagonal
    factors; no automatic consistency repair is attempted.

    Parameters
    ----------
    spectra : sequence of torch.Tensor
        Real, finite, non-negative singular-value vectors, optionally
        carrying structural batches.
    powers : sequence of tuple[float, float]
        Absorption powers at each bond: (0, 0), (0.5, 0.5), (1, 1), (1, 0)
        or (0, 1). Inverse powers require strictly nonzero spectra.
    """

    def __init__(self, spectra, powers) -> None:
        """Initializes the stored tensor references and validates construction."""
        self._spectra = _SafeList(spectra, self._on_sequence_changed)
        self._powers = _SafeList(powers, self._on_sequence_changed)
        if len(self.spectra) != len(self.powers):
            raise ValueError('Spectra and powers should have the same length')
        values = []
        for spectrum, (left, right) in zip(self.spectra, self.powers):
            if not isinstance(spectrum, torch.Tensor) or spectrum.is_complex():
                raise TypeError('Schmidt spectra should be real tensors')
            if not torch.isfinite(spectrum).all() or torch.any(spectrum < 0):
                raise ValueError('Schmidt spectra should be finite and non-negative')
            if (left, right) not in ((0, 0), (0.5, 0.5), (1, 1), (1, 0), (0, 1)):
                raise ValueError('Invalid Vidal absorption powers')
            exponent = 1 - left - right
            if exponent < 0 and torch.any(spectrum == 0):
                raise ValueError('Inverse Vidal requires nonzero Schmidt spectra')
            values.append(None if exponent == 0 else spectrum.pow(exponent))
        super().__init__(values)
        self._valid = True

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

    def _with_callback(self, on_change):
        """Copies bond lists and binds their callback, preserving tensors."""
        result = super()._with_callback(on_change)
        result._spectra = _SafeList(self._spectra, result._on_sequence_changed)
        result._powers = _SafeList(self._powers, result._on_sequence_changed)
        return result

    def _map_tensors(self, function):
        """Maps stored tensors while preserving concrete container semantics."""
        result = super()._map_tensors(function)
        spectra = []
        for spectrum in self.spectra:
            mapped = function(spectrum)
            spectra.append(mapped.real if mapped.is_complex() else mapped)
        result._spectra = _SafeList(spectra, result._on_sequence_changed)
        result._powers = _SafeList(self._powers, result._on_sequence_changed)
        return result
