"""Diagonal bond factors and explicit records of Vidal absorption."""

from copy import copy
from typing import Sequence

import torch


class _BondList(list):
    """Fixed-length bond sequence with validation on element replacement."""

    def __init__(self, values, owner, name: str) -> None:
        super().__init__(values)
        self._owner = owner
        self._name = name

    def __setitem__(self, key, value):
        values = list(value) if isinstance(key, slice) else [value]
        if isinstance(key, slice) and len(values) != len(self[key]):
            raise ValueError('Bond slice replacement should preserve length')
        replacement = list(self)
        replacement[key] = values if isinstance(key, slice) else value
        self._owner._replace_sequence(self._name, replacement)
        super().__setitem__(key, values if isinstance(key, slice) else value)
        setattr(self._owner, '_' + self._name, self)

    def _structural_error(self, *args, **kwargs):
        raise TypeError('Replace the bond object to change its structure')

    append = extend = insert = pop = remove = clear = _structural_error
    reverse = sort = __delitem__ = __iadd__ = __imul__ = _structural_error


class BondFactors:
    """Optional diagonals between adjacent cores, including a cyclic closure.

    Each format owns a separate bond container, sharing the supplied tensors.
    Element and same-length slice replacements validate against its cores
    immediately. Invalid replacements leave the container unchanged.

    Parameters
    ----------
    values : sequence[torch.Tensor or None]
        One diagonal per stored bond. A diagonal has shape ``(rank,)`` or
        ``(*core_batch, rank)``. None denotes an identity without allocation.

    """

    def __init__(self, values) -> None:
        self._owner = None
        self.values = values

    def _replace_sequence(self, name, values):
        """Validates a replacement and invalidates manually edited Vidal state."""
        if isinstance(values, torch.Tensor):
            raise TypeError('`values` should be a sequence of diagonals')
        values = list(values)
        if name == 'values' and not all(
                value is None or isinstance(value, torch.Tensor)
                for value in values):
            raise TypeError('Bond factors should be tensors or None')
        attribute = '_' + name
        previous = getattr(self, attribute, None)
        setattr(self, attribute, _BondList(values, self, name))
        try:
            if self._owner is not None:
                self._owner.validate_bonds()
        except (TypeError, ValueError):
            setattr(self, attribute, previous)
            raise

        if hasattr(self, '_valid'):
            self._valid = False
        if self._owner is not None:
            self._owner._orth_center = None

    @property
    def values(self):
        """Mutable diagonal factors, validated on replacement when attached."""
        return self._values

    @values.setter
    def values(self, values):
        self._replace_sequence('values', values)

    def _with_owner(self, owner):
        """Copies bond containers for a format, preserving tensor references."""
        result = copy(self)
        result._owner = owner
        result._values = _BondList(self._values, result, 'values')
        if isinstance(self, VidalGauge):
            result._spectra = _BondList(self._spectra, result, 'spectra')
            result._powers = _BondList(self._powers, result, 'powers')
        return result

    def validate(self, cores: Sequence[torch.Tensor], cyclic: bool):
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
        result._owner = None
        values = []
        for value in self.values:
            if value is None:
                values.append(None)
                continue
            mapped = function(value)
            if not value.is_complex() and mapped.is_complex():
                mapped = mapped.real
            values.append(mapped)
        result._values = _BondList(values, result, 'values')
        return result


class VidalGauge(BondFactors):
    """Schmidt spectra and their current absorption into neighbouring cores.

    Powers (0, 0), (0.5, 0.5), and (1, 1) represent explicit, implicit and
    inverse Vidal respectively. The remaining bond factor has power
    ``1 - left_power - right_power``. Spectra are real and non-negative.
    """

    def __init__(self, spectra, powers) -> None:
        self._spectra = _BondList(spectra, self, 'spectra')
        self._powers = _BondList(powers, self, 'powers')
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

    def _map_tensors(self, function):
        result = super()._map_tensors(function)
        spectra = []
        for spectrum in self.spectra:
            mapped = function(spectrum)
            spectra.append(mapped.real if mapped.is_complex() else mapped)
        result._spectra = _BondList(spectra, result, 'spectra')
        result._powers = _BondList(self._powers, result, 'powers')
        return result
