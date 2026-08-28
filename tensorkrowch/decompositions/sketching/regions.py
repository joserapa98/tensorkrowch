"""Region geometry and gather-based sketch recursions."""

from dataclasses import dataclass, field
from typing import (Hashable, Iterator, Optional, Sequence, Tuple, Union)
from uuid import uuid4

import torch

from tensorkrowch.decompositions.sources import ConfigurationBatch


Site = Union[int, Tuple[Hashable, ...]]
SampleValues = Union[
    torch.Tensor,
    Sequence[torch.Tensor],
    ConfigurationBatch,
]


@dataclass(frozen=True)
class SiteRegion:
    """Ordered collection of distinct sites with set-like region operations."""

    sites: Sequence[Site] = ()
    _membership: frozenset = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.sites, (str, bytes)):
            raise TypeError('`sites` should be a sequence of site identifiers')
        try:
            sites = tuple(self.sites)
        except TypeError as exc:
            raise TypeError(
                '`sites` should be a sequence of site identifiers') from exc

        site_kind = None
        coordinate_size = None
        for site in sites:
            if isinstance(site, bool):
                raise TypeError('Site identifiers should be integers or tuples')
            if isinstance(site, int):
                current_kind = 'linear'
            elif isinstance(site, tuple) and site:
                current_kind = 'coordinate'
                if coordinate_size is None:
                    coordinate_size = len(site)
                elif len(site) != coordinate_size:
                    raise ValueError(
                        'Coordinate sites should have the same dimension')
                try:
                    hash(site)
                except TypeError as exc:
                    raise TypeError('Coordinate sites should be hashable') \
                        from exc
            else:
                raise TypeError('Site identifiers should be integers or tuples')
            if site_kind is None:
                site_kind = current_kind
            elif current_kind != site_kind:
                raise ValueError(
                    'A region should not mix linear and coordinate sites')

        membership = frozenset(sites)
        if len(membership) != len(sites):
            raise ValueError('`sites` should not contain duplicates')
        object.__setattr__(self, 'sites', sites)
        object.__setattr__(self, '_membership', membership)

    def __len__(self) -> int:
        return len(self.sites)

    def __iter__(self) -> Iterator[Site]:
        return iter(self.sites)

    def contains(self, other: Union[Site, 'SiteRegion']) -> bool:
        """Checks membership of one site or containment of another region."""
        if isinstance(other, SiteRegion):
            return other._membership.issubset(self._membership)
        try:
            return other in self._membership
        except TypeError:
            return False

    def union(self, *others: 'SiteRegion') -> 'SiteRegion':
        """Returns the ordered first-occurrence union of compatible regions."""
        sites = list(self.sites)
        membership = set(self._membership)
        for other in others:
            if not isinstance(other, SiteRegion):
                raise TypeError('`others` should contain SiteRegion objects')
            for site in other:
                if site not in membership:
                    sites.append(site)
                    membership.add(site)
        return SiteRegion(sites)

    def difference(self, other: 'SiteRegion') -> 'SiteRegion':
        """Returns sites absent from ``other`` while preserving local order."""
        if not isinstance(other, SiteRegion):
            raise TypeError('`other` should be SiteRegion type')
        return SiteRegion(
            site for site in self.sites if not other.contains(site))

    def intersection(self, other: 'SiteRegion') -> 'SiteRegion':
        """Returns common sites in this region's order."""
        if not isinstance(other, SiteRegion):
            raise TypeError('`other` should be SiteRegion type')
        return SiteRegion(site for site in self.sites if other.contains(site))


class _SamplePool:
    """Owns correlated sample rows and caches their regional restrictions."""

    def __init__(self,
                 samples: SampleValues,
                 sites: Optional[Sequence[Site]] = None,
                 pool_id: Optional[Hashable] = None) -> None:
        if isinstance(samples, ConfigurationBatch):
            if samples.packed:
                site_values = tuple(
                    samples.values[:, site]
                    for site in range(samples.n_sites))
            else:
                site_values = tuple(samples.values)
        elif isinstance(samples, torch.Tensor):
            if samples.ndim < 2:
                raise ValueError(
                    '`samples` should have batch and site dimensions')
            site_values = tuple(
                samples[:, site] for site in range(samples.shape[1]))
        else:
            if isinstance(samples, (str, bytes)):
                raise TypeError(
                    '`samples` should be a tensor or a sequence of tensors')
            try:
                site_values = tuple(samples)
            except TypeError as exc:
                raise TypeError(
                    '`samples` should be a tensor or a sequence of tensors') \
                    from exc
            if not site_values:
                raise ValueError('`samples` should contain at least one site')
            if not all(isinstance(value, torch.Tensor)
                       for value in site_values):
                raise TypeError(
                    'Every site in `samples` should be a torch.Tensor')

        if not site_values:
            raise ValueError('`samples` should contain at least one site')
        first = site_values[0]
        if first.ndim < 1 or first.shape[0] < 1:
            raise ValueError('Samples should contain a non-empty batch')
        for site, values in enumerate(site_values):
            if values.ndim < 1:
                raise ValueError(
                    f'Samples at site {site} should have a batch dimension')
            if values.shape[0] != first.shape[0]:
                raise ValueError(
                    f'Samples at site {site} should have batch size '
                    f'{first.shape[0]}')
            if values.device != first.device:
                raise ValueError('All sample sites should share a device')
            if (values.is_floating_point() or values.is_complex()) and \
                    not torch.isfinite(values).all():
                raise ValueError(f'Samples at site {site} should be finite')

        if sites is None:
            region = SiteRegion(range(len(site_values)))
        else:
            region = SiteRegion(sites)
            if len(region) != len(site_values):
                raise ValueError(
                    '`sites` should contain one identifier per sample site')
        if pool_id is None:
            pool_id = uuid4().hex
        try:
            hash(pool_id)
        except TypeError as exc:
            raise TypeError('`pool_id` should be hashable') from exc

        self._values = site_values
        self._site_to_axis = {
            site: axis for axis, site in enumerate(region)}
        self.region = region
        self.pool_id = pool_id
        self._cache = {}

    @property
    def n_rows(self) -> int:
        """Number of correlated sample rows."""
        return self._values[0].shape[0]

    @property
    def n_sites(self) -> int:
        """Number of sites represented by every row."""
        return len(self._values)

    @property
    def device(self) -> torch.device:
        """Device shared by all sample sites."""
        return self._values[0].device

    def values(self, site: Site) -> torch.Tensor:
        """Returns all original sample values at one site."""
        if not self.region.contains(site):
            raise ValueError('`site` should belong to the sample pool')
        return self._values[self._site_to_axis[site]]

    @staticmethod
    def _unique_inverse(values: torch.Tensor) -> torch.Tensor:
        """Returns row inverse ids, including complex coordinate tensors."""
        flat = values.reshape(values.shape[0], -1)
        if flat.is_complex():
            flat = torch.view_as_real(flat).reshape(flat.shape[0], -1)
        return torch.unique(
            flat, sorted=True, dim=0, return_inverse=True)[1]

    @staticmethod
    def _first_representatives(
            inverse_ids: torch.Tensor,
            n_unique: int) -> torch.Tensor:
        """Finds the first pool row associated with every unique row id."""
        order = torch.argsort(inverse_ids, stable=True)
        first = torch.ones_like(order, dtype=torch.bool)
        first[1:] = inverse_ids[order[1:]] != inverse_ids[order[:-1]]
        representatives = order[first]
        if representatives.shape[0] != n_unique:
            raise RuntimeError('Invalid regional inverse ids')
        return representatives

    def restrict(self, region: SiteRegion) -> 'RegionSketch':
        """Returns the unique restriction of every pool row to ``region``."""
        if not isinstance(region, SiteRegion):
            raise TypeError('`region` should be SiteRegion type')
        if not self.region.contains(region):
            raise ValueError('`region` should be contained in the sample pool')
        if region in self._cache:
            return self._cache[region]

        if len(region):
            site_inverse = [
                self._unique_inverse(self.values(site)) for site in region]
            signatures = torch.stack(site_inverse, dim=1)
            unique_signatures, inverse_ids = torch.unique(
                signatures, sorted=True, dim=0, return_inverse=True)
            n_unique = unique_signatures.shape[0]
            representative_row_ids = self._first_representatives(
                inverse_ids, n_unique)
            values = tuple(
                self.values(site).index_select(0, representative_row_ids)
                for site in region)
        else:
            inverse_ids = torch.zeros(
                self.n_rows, dtype=torch.long, device=self.device)
            representative_row_ids = torch.zeros(
                1, dtype=torch.long, device=self.device)
            values = ()

        sketch = RegionSketch(
            pool_id=self.pool_id,
            region=region,
            values=values,
            representative_row_ids=representative_row_ids,
            inverse_ids=inverse_ids,
            _pool=self)
        self._cache[region] = sketch
        return sketch


@dataclass(frozen=True)
class RegionSketch:
    """Unique correlated sample restrictions associated with one region."""

    pool_id: Hashable
    region: SiteRegion
    values: Sequence[torch.Tensor]
    representative_row_ids: torch.Tensor
    inverse_ids: torch.Tensor
    _pool: _SamplePool = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.region, SiteRegion):
            raise TypeError('`region` should be SiteRegion type')
        if not isinstance(self._pool, _SamplePool):
            raise TypeError('`_pool` should be _SamplePool type')
        if self.pool_id != self._pool.pool_id:
            raise ValueError('`pool_id` should match the owning sample pool')
        values = tuple(self.values)
        if len(values) != len(self.region):
            raise ValueError('`values` should contain one tensor per region site')
        if not isinstance(self.representative_row_ids, torch.Tensor) or \
                self.representative_row_ids.ndim != 1 or \
                self.representative_row_ids.dtype != torch.long:
            raise TypeError(
                '`representative_row_ids` should be a long vector')
        if not isinstance(self.inverse_ids, torch.Tensor) or \
                self.inverse_ids.ndim != 1 or \
                self.inverse_ids.dtype != torch.long:
            raise TypeError('`inverse_ids` should be a long vector')
        if self.inverse_ids.shape[0] != self._pool.n_rows:
            raise ValueError(
                '`inverse_ids` should contain one id per sample-pool row')
        if self.representative_row_ids.shape[0] < 1:
            raise ValueError('A region sketch should contain at least one row')
        if torch.any(self.representative_row_ids < 0) or \
                torch.any(self.representative_row_ids >= self._pool.n_rows):
            raise ValueError('`representative_row_ids` are out of bounds')
        if torch.any(self.inverse_ids < 0) or \
                torch.any(self.inverse_ids >= self.n_unique):
            raise ValueError('`inverse_ids` are out of bounds')
        for value in values:
            if not isinstance(value, torch.Tensor):
                raise TypeError('`values` should contain torch.Tensor objects')
            if value.shape[0] != self.n_unique:
                raise ValueError(
                    'Every tensor in `values` should contain one unique row')
        object.__setattr__(self, 'values', values)

    @property
    def n_unique(self) -> int:
        """Number of distinct regional sample rows."""
        return self.representative_row_ids.shape[0]

    @property
    def n_rows(self) -> int:
        """Number of original correlated rows represented by the sketch."""
        return self.inverse_ids.shape[0]

    def restrict(self, region: SiteRegion) -> 'RegionSketch':
        """Restricts this sketch to a contained region through its pool."""
        if not self.region.contains(region):
            raise ValueError('`region` should be contained in this sketch')
        return self._pool.restrict(region)

    def combine(self, *others: 'RegionSketch') -> 'RegionSketch':
        """Combines same-pool sketches by correlated rows, never Cartesian."""
        membership = set(self.region.sites)
        for other in others:
            if not isinstance(other, RegionSketch):
                raise TypeError('`others` should contain RegionSketch objects')
            if (other._pool is not self._pool) or \
                    (other.pool_id != self.pool_id) or \
                    (other.n_rows != self.n_rows):
                raise ValueError(
                    'Correlated sketches should belong to the same sample pool')
            membership.update(other.region.sites)
        region = SiteRegion(
            site for site in self._pool.region if site in membership)
        return self._pool.restrict(region)

    def compare(
            self,
            other: 'RegionSketch'
            ) -> Tuple[SiteRegion, SiteRegion, SiteRegion]:
        """Returns common, self-only and other-only ordered regions."""
        if not isinstance(other, RegionSketch):
            raise TypeError('`other` should be RegionSketch type')
        return (
            self.region.intersection(other.region),
            self.region.difference(other.region),
            other.region.difference(self.region),
        )

    def recursive_projector(self, target: 'RegionSketch') -> 'SketchRecursion':
        """Builds the gather and new values for a contained target sketch."""
        if not isinstance(target, RegionSketch):
            raise TypeError('`target` should be RegionSketch type')
        if (target._pool is not self._pool) or \
                (target.pool_id != self.pool_id) or \
                (target.n_rows != self.n_rows):
            raise ValueError(
                'Recursive sketches should belong to the same sample pool')
        if not target.region.contains(self.region):
            raise ValueError(
                'The target region should contain the source region')
        target_rows = target.representative_row_ids
        gather = self.inverse_ids.index_select(0, target_rows)
        new_region = target.region.difference(self.region)
        new_values = tuple(
            self._pool.values(site).index_select(0, target_rows)
            for site in new_region)
        return SketchRecursion(
            pool_id=self.pool_id,
            child_region=self.region,
            parent_region=target.region,
            child_size=self.n_unique,
            parent_size=target.n_unique,
            gather=gather,
            new_values=new_values)


@dataclass(frozen=True)
class SketchRecursion:
    """Gather map and new site values relating two regional sketch bases."""

    pool_id: Hashable
    child_region: SiteRegion
    parent_region: SiteRegion
    child_size: int
    parent_size: int
    gather: torch.Tensor
    new_values: Sequence[torch.Tensor] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.child_region, SiteRegion) or \
                not isinstance(self.parent_region, SiteRegion):
            raise TypeError(
                '`child_region` and `parent_region` should be SiteRegion')
        for name in ('child_size', 'parent_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f'`{name}` should be int type')
            if value < 1:
                raise ValueError(f'`{name}` should be positive')
        if not isinstance(self.gather, torch.Tensor) or \
                self.gather.ndim != 1 or self.gather.dtype != torch.long:
            raise TypeError('`gather` should be a one-dimensional long tensor')
        if self.gather.shape[0] != self.parent_size:
            raise ValueError('`gather` should contain one id per parent row')
        if torch.any(self.gather < 0) or \
                torch.any(self.gather >= self.child_size):
            raise ValueError('`gather` contains an out-of-range child id')

        new_values = tuple(self.new_values)
        if len(new_values) != len(self.new_region):
            raise ValueError(
                '`new_values` should contain one tensor per added site')
        for value in new_values:
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    '`new_values` should contain torch.Tensor objects')
            if value.shape[0] != self.parent_size:
                raise ValueError(
                    'Every tensor in `new_values` should match parent size')
            if value.device != self.gather.device:
                raise ValueError('Recursion tensors should share a device')
        object.__setattr__(self, 'new_values', new_values)

    @property
    def new_region(self) -> SiteRegion:
        """Sites introduced when moving from child to parent."""
        return self.parent_region.difference(self.child_region)

    @property
    def removed_region(self) -> SiteRegion:
        """Sites removed when this recursion is the inverse of an expansion."""
        return self.child_region.difference(self.parent_region)

    def value(self, site: Site) -> torch.Tensor:
        """Returns parent-aligned values for one newly introduced site."""
        if not self.new_region.contains(site):
            raise ValueError('`site` should belong to the new recursion region')
        return self.new_values[self.new_region.sites.index(site)]

    def apply(self, tensor: torch.Tensor, axis: int = 0) -> torch.Tensor:
        """Gathers a child sketch axis into parent-row order."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('`tensor` should be torch.Tensor type')
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError('`axis` should be int type')
        if tensor.ndim < 1:
            raise ValueError('`tensor` should have at least one dimension')
        if axis < 0:
            axis += tensor.ndim
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError('`axis` is out of bounds for `tensor`')
        if tensor.shape[axis] != self.child_size:
            raise ValueError(
                'The selected tensor axis should match the child sketch size')
        return tensor.index_select(axis, self.gather.to(tensor.device))

    def compose(self, next_recursion: 'SketchRecursion') -> 'SketchRecursion':
        """Composes consecutive monotone sketch expansions by gather."""
        if not isinstance(next_recursion, SketchRecursion):
            raise TypeError(
                '`next_recursion` should be SketchRecursion type')
        if self.pool_id != next_recursion.pool_id:
            raise ValueError('Recursions should belong to the same sample pool')
        if self.parent_region != next_recursion.child_region or \
                self.parent_size != next_recursion.child_size:
            raise ValueError('Recursions should have consecutive regions')
        if not self.parent_region.contains(self.child_region) or \
                not next_recursion.parent_region.contains(
                    next_recursion.child_region):
            raise ValueError(
                'Only monotone regional expansions can be composed')

        next_gather = next_recursion.gather.to(self.gather.device)
        gather = self.gather.index_select(0, next_gather)
        values_by_site = {
            site: value.index_select(0, next_gather.to(value.device))
            for site, value in zip(self.new_region, self.new_values)
        }
        values_by_site.update({
            site: value
            for site, value in zip(
                next_recursion.new_region, next_recursion.new_values)
        })
        new_region = next_recursion.parent_region.difference(
            self.child_region)
        return SketchRecursion(
            pool_id=self.pool_id,
            child_region=self.child_region,
            parent_region=next_recursion.parent_region,
            child_size=self.child_size,
            parent_size=next_recursion.parent_size,
            gather=gather,
            new_values=tuple(values_by_site[site] for site in new_region))

    def inverse(self) -> 'SketchRecursion':
        """Inverts a bijective expansion without building a dense matrix."""
        if not self.parent_region.contains(self.child_region):
            raise ValueError('Only a regional expansion can be inverted')
        if self.child_size != self.parent_size:
            raise ValueError('The recursion should be bijective to be inverted')
        expected = torch.arange(
            self.child_size, device=self.gather.device, dtype=torch.long)
        if not torch.equal(torch.sort(self.gather).values, expected):
            raise ValueError('The recursion should be bijective to be inverted')
        return SketchRecursion(
            pool_id=self.pool_id,
            child_region=self.parent_region,
            parent_region=self.child_region,
            child_size=self.parent_size,
            parent_size=self.child_size,
            gather=torch.argsort(self.gather),
            new_values=())


__all__ = [
    'SiteRegion',
    'RegionSketch',
    'SketchRecursion',
]
