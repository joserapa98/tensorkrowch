"""Diagonal bond factors and explicit records of Vidal absorption."""

from copy import copy

import torch


class BondFactors:
    """Optional diagonals between adjacent cores, including a cyclic closure.

    Parameters
    ----------
    values : sequence[torch.Tensor or None]
        One diagonal per stored bond. A diagonal has shape ``(rank,)`` or
        ``(*core_batch, rank)``. None denotes an identity without allocation.
    """

    def __init__(self, values):
        if isinstance(values, torch.Tensor):
            raise TypeError('`values` should be a sequence of diagonals')
        self.values = list(values)

    def validate(self, cores, cyclic):
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
        result = copy(self)
        result.values = []
        for value in self.values:
            if value is None:
                result.values.append(None)
                continue
            mapped = function(value)
            if not value.is_complex() and mapped.is_complex():
                mapped = mapped.real
            result.values.append(mapped)
        return result


class VidalGauge(BondFactors):
    """Schmidt spectra and their current absorption into neighbouring cores.

    Powers (0, 0), (0.5, 0.5), and (1, 1) represent explicit, implicit and
    inverse Vidal respectively. The remaining bond factor has power
    ``1 - left_power - right_power``. Spectra are real and non-negative.
    """

    def __init__(self, spectra, powers):
        self.spectra = list(spectra)
        self.powers = list(powers)
        self._valid = True
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

    def _map_tensors(self, function):
        result = super()._map_tensors(function)
        result.spectra = []
        for spectrum in self.spectra:
            mapped = function(spectrum)
            result.spectra.append(mapped.real if mapped.is_complex() else mapped)
        result.powers = list(self.powers)
        return result
