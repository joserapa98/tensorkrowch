"""Quantized layouts and coordinate maps for recursive sketching."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch


IntegerSpec = Union[int, Sequence[int]]
DigitSite = Tuple[int, int]


def _integer_spec(value: IntegerSpec,
                  n_variables: int,
                  name: str,
                  minimum: int) -> Tuple[int, ...]:
    """Broadcasts one integer or validates one value per variable."""
    if isinstance(value, bool):
        raise TypeError(f'`{name}` should be int or a sequence of ints')
    if isinstance(value, int):
        values = (value,) * n_variables
    else:
        if isinstance(value, (str, bytes)):
            raise TypeError(f'`{name}` should be int or a sequence of ints')
        try:
            values = tuple(value)
        except TypeError as exc:
            raise TypeError(
                f'`{name}` should be int or a sequence of ints') from exc
        if len(values) != n_variables:
            raise ValueError(
                f'`{name}` should contain one value per variable')
    if any(isinstance(item, bool) or not isinstance(item, int)
           for item in values):
        raise TypeError(f'`{name}` values should be integers')
    if any(item < minimum for item in values):
        qualifier = 'at least two' if minimum == 2 else 'positive'
        raise ValueError(f'`{name}` values should be {qualifier}')
    return values


@dataclass(frozen=True)
class QuantizedLayout:
    """Defines how multivariable integer indices are expanded into digits.

    A digit site is identified by ``(variable, digit)``. The digit coordinate
    is canonical and always runs from zero at the most significant (coarse)
    digit to ``level[variable] - 1`` at the least significant (fine) digit.
    ``digit_order`` only changes the TT site schedule.

    ``ordering="grouped"`` places all digits of each variable together.
    ``ordering="interleaved"`` cycles over variables at every available digit
    depth and omits variables whose levels are exhausted. A custom layout uses
    ``ordering="custom"`` and supplies every canonical digit site exactly once
    through ``permutation``.

    Changing the layout reorders configurations, not fitted TT cores. Moving
    non-neighboring cores would require explicit tensor swaps and possible
    rank truncation.

    Parameters
    ----------
    n_variables : int
        Number of independently quantized variables.
    base : int or sequence of int
        Digit base shared by all variables or specified per variable.
    level : int or sequence of int
        Number of digits shared by all variables or specified per variable.
    ordering : {``"grouped"``, ``"interleaved"``, ``"custom"``}
        Final digit-site schedule.
    digit_order : {``"coarse_to_fine"``, ``"fine_to_coarse"``}
        Direction in which each variable contributes its canonical digits.
    permutation : sequence of tuple[int, int], optional
        Complete custom schedule of ``(variable, canonical_digit)`` pairs.

    Examples
    --------
    >>> layout = QuantizedLayout(
    ...     n_variables=2, base=2, level=3, ordering='interleaved')
    >>> digits = layout.encode_indices(torch.tensor([[3, 5]]))
    >>> torch.equal(layout.decode_digits(digits), torch.tensor([[3, 5]]))
    True
    """

    n_variables: int
    base: IntegerSpec = 2
    level: IntegerSpec = 1
    ordering: str = 'grouped'
    digit_order: str = 'coarse_to_fine'
    permutation: Optional[Sequence[DigitSite]] = None

    def __post_init__(self) -> None:
        if isinstance(self.n_variables, bool) or not isinstance(
                self.n_variables, int):
            raise TypeError('`n_variables` should be int type')
        if self.n_variables < 1:
            raise ValueError('`n_variables` should be positive')
        base = _integer_spec(
            self.base, self.n_variables, 'base', minimum=2)
        level = _integer_spec(
            self.level, self.n_variables, 'level', minimum=1)
        if self.ordering not in ('grouped', 'interleaved', 'custom'):
            raise ValueError(
                "`ordering` should be 'grouped', 'interleaved' or 'custom'")
        if self.digit_order not in ('coarse_to_fine', 'fine_to_coarse'):
            raise ValueError(
                "`digit_order` should be 'coarse_to_fine' or "
                "'fine_to_coarse'")

        maximum = torch.iinfo(torch.long).max
        for variable, (site_base, site_level) in enumerate(zip(base, level)):
            if site_base ** site_level > maximum:
                raise OverflowError(
                    f'Quantized variable {variable} exceeds int64 indices')

        canonical = tuple(
            (variable, digit)
            for variable, variable_level in enumerate(level)
            for digit in range(variable_level))
        if self.ordering == 'custom':
            if self.permutation is None:
                raise ValueError(
                    '`permutation` is required for custom ordering')
            try:
                permutation = tuple(tuple(site) for site in self.permutation)
            except TypeError as exc:
                raise TypeError(
                    '`permutation` should contain digit-site pairs') from exc
            if any(len(site) != 2 or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in site) for site in permutation):
                raise TypeError(
                    '`permutation` should contain integer digit-site pairs')
            if len(permutation) != len(canonical) or \
                    set(permutation) != set(canonical):
                raise ValueError(
                    '`permutation` should contain every digit site once')
        elif self.permutation is not None:
            raise ValueError(
                '`permutation` is only valid with custom ordering')
        else:
            permutation = None

        object.__setattr__(self, 'base', base)
        object.__setattr__(self, 'level', level)
        object.__setattr__(self, 'permutation', permutation)

    @property
    def n_sites(self) -> int:
        """Total number of digit sites."""
        return sum(self.level)

    @property
    def grid_size(self) -> Tuple[int, ...]:
        """Number of representable integer indices per variable."""
        return tuple(site_base ** site_level for site_base, site_level in zip(
            self.base, self.level))

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Basis input dimension of every digit site in schedule order."""
        return tuple(self.base[variable] for variable, _ in self.sites())

    def _variable_digits(self, variable: int) -> Tuple[int, ...]:
        """Canonical digit ids in the requested within-variable direction."""
        digits = tuple(range(self.level[variable]))
        return digits if self.digit_order == 'coarse_to_fine' \
            else tuple(reversed(digits))

    def sites(self) -> Tuple[DigitSite, ...]:
        """Returns the ordered ``(variable, canonical_digit)`` schedule."""
        if self.ordering == 'custom':
            return tuple(self.permutation)
        variable_digits = tuple(
            self._variable_digits(variable)
            for variable in range(self.n_variables))
        if self.ordering == 'grouped':
            return tuple(
                (variable, digit)
                for variable, digits in enumerate(variable_digits)
                for digit in digits)
        return tuple(
            (variable, digits[depth])
            for depth in range(max(self.level))
            for variable, digits in enumerate(variable_digits)
            if depth < len(digits))

    @staticmethod
    def _integer_tensor(values: torch.Tensor, name: str) -> torch.Tensor:
        """Validates one integer tensor without changing its device."""
        if not isinstance(values, torch.Tensor):
            raise TypeError(f'`{name}` should be torch.Tensor type')
        if values.ndim < 1 or values.dtype not in (
                torch.uint8, torch.int8, torch.int16, torch.int32,
                torch.int64):
            raise TypeError(f'`{name}` should be an integer tensor')
        return values.to(dtype=torch.long)

    def encode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Encodes integer variable indices into scheduled digit columns."""
        indices = self._integer_tensor(indices, 'indices')
        if indices.shape[-1] != self.n_variables:
            raise ValueError(
                'The last `indices` dimension should match `n_variables`')
        canonical = {}
        for variable, (site_base, site_level, size) in enumerate(zip(
                self.base, self.level, self.grid_size)):
            values = indices[..., variable]
            if torch.any(values < 0) or torch.any(values >= size):
                raise ValueError(
                    f'`indices` is out of bounds for variable {variable}')
            for digit in range(site_level):
                stride = site_base ** (site_level - 1 - digit)
                canonical[(variable, digit)] = torch.remainder(
                    torch.div(values, stride, rounding_mode='floor'),
                    site_base)
        return torch.stack(
            [canonical[site] for site in self.sites()], dim=-1)

    def decode_digits(self, digits: torch.Tensor) -> torch.Tensor:
        """Decodes scheduled digits into one integer index per variable."""
        digits = self._integer_tensor(digits, 'digits')
        if digits.shape[-1] != self.n_sites:
            raise ValueError(
                'The last `digits` dimension should match the layout sites')
        canonical = {}
        for column, site in enumerate(self.sites()):
            variable, _ = site
            values = digits[..., column]
            if torch.any(values < 0) or torch.any(values >= self.base[variable]):
                raise ValueError(
                    f'`digits` is out of bounds at site {column}')
            canonical[site] = values
        indices = []
        for variable, (site_base, site_level) in enumerate(zip(
                self.base, self.level)):
            value = digits.new_zeros(digits.shape[:-1])
            for digit in range(site_level):
                stride = site_base ** (site_level - 1 - digit)
                value = value + canonical[(variable, digit)] * stride
            indices.append(value)
        return torch.stack(indices, dim=-1)

    def reorder_configurations(
            self,
            digits: torch.Tensor,
            target_ordering: Union[str, 'QuantizedLayout']
            ) -> torch.Tensor:
        """Reorders digit columns into another compatible layout."""
        if isinstance(target_ordering, str):
            target = QuantizedLayout(
                n_variables=self.n_variables,
                base=self.base,
                level=self.level,
                ordering=target_ordering,
                digit_order=self.digit_order)
        elif isinstance(target_ordering, QuantizedLayout):
            target = target_ordering
        else:
            raise TypeError(
                '`target_ordering` should be str or QuantizedLayout type')
        if target.n_variables != self.n_variables or \
                target.base != self.base or target.level != self.level:
            raise ValueError('Source and target layouts should be compatible')
        return target.encode_indices(self.decode_digits(digits))


__all__ = ['QuantizedLayout']
