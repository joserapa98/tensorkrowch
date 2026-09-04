"""Internal specifications shared by recursive-sketching decompositions."""

from dataclasses import dataclass, field
from math import prod
from typing import (Callable, Optional, Sequence, Tuple, Union)

import torch

from tensorkrowch.decompositions.observers import _normalize_verbosity
from tensorkrowch.decompositions.sources import ConfigurationBatch
from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.embeddings import basis


_Embedding = Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]
_Domain = Optional[Union[torch.Tensor, Sequence[torch.Tensor]]]
_Samples = Optional[Union[
    torch.Tensor,
    Sequence[torch.Tensor],
    ConfigurationBatch,
]]


def _normalize_n_sites(n_sites: int) -> int:
    """Validates a positive number of input sites."""
    if isinstance(n_sites, bool) or not isinstance(n_sites, int):
        raise TypeError('`n_sites` should be int type')
    if n_sites < 1:
        raise ValueError('`n_sites` should be positive')
    return n_sites


def _validate_site(site: int, n_sites: int) -> int:
    """Validates one site against a normalized site count."""
    if isinstance(site, bool) or not isinstance(site, int):
        raise TypeError('`site` should be int type')
    if (site < 0) or (site >= n_sites):
        raise ValueError(f'`site` should be between 0 and {n_sites - 1}')
    return site


def _split_samples(samples: _Samples,
                   n_sites: int) -> Tuple[torch.Tensor, ...]:
    """Normalizes packed or heterogeneous samples to one tensor per site."""
    if isinstance(samples, ConfigurationBatch):
        if samples.n_sites != n_sites:
            raise ValueError(
                '`samples` should contain one value per input site')
        if samples.packed:
            return tuple(samples.values[:, site] for site in range(n_sites))
        return tuple(samples.values)
    if isinstance(samples, torch.Tensor):
        if samples.ndim < 2:
            raise ValueError(
                '`samples` should have batch and site dimensions')
        if samples.shape[1] != n_sites:
            raise ValueError(
                '`samples` should contain one value per input site')
        return tuple(samples[:, site] for site in range(n_sites))
    if samples is None:
        raise ValueError('`samples` is required when `domain` is None')
    if isinstance(samples, (str, bytes)):
        raise TypeError(
            '`samples` should be a tensor or a sequence of tensors')
    try:
        site_samples = tuple(samples)
    except TypeError as exc:
        raise TypeError(
            '`samples` should be a tensor or a sequence of tensors') from exc
    if len(site_samples) != n_sites:
        raise ValueError(
            '`samples` should contain one tensor per input site')
    if not all(isinstance(value, torch.Tensor) for value in site_samples):
        raise TypeError('Every site in `samples` should be a torch.Tensor')
    batch_size = site_samples[0].shape[0] if site_samples[0].ndim else None
    if batch_size is None:
        raise ValueError('Samples at site 0 should have a batch dimension')
    for site, values in enumerate(site_samples):
        if values.ndim < 1:
            raise ValueError(
                f'Samples at site {site} should have a batch dimension')
        if values.shape[0] != batch_size:
            raise ValueError(
                f'Samples at site {site} should have batch size {batch_size}')
    return site_samples


@dataclass(frozen=True)
class _DomainSpec:
    """Stores one finite coordinate domain per input site."""

    values: Sequence[torch.Tensor]
    inferred: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.values, torch.Tensor) or \
                isinstance(self.values, (str, bytes)):
            raise TypeError('`values` should contain one domain per site')
        try:
            values = tuple(self.values)
        except TypeError as exc:
            raise TypeError(
                '`values` should contain one domain per site') from exc
        if not values:
            raise ValueError('`values` should contain at least one domain')
        for site, domain in enumerate(values):
            if not isinstance(domain, torch.Tensor):
                raise TypeError(
                    f'Domain at site {site} should be a torch.Tensor')
            if domain.ndim < 1:
                raise ValueError(
                    f'Domain at site {site} should have a values dimension')
            if domain.shape[0] < 1:
                raise ValueError(
                    f'Domain at site {site} should contain at least one value')
            if (domain.is_floating_point() or domain.is_complex()) and \
                    not torch.isfinite(domain).all():
                raise ValueError(f'Domain at site {site} should be finite')
        if not isinstance(self.inferred, bool):
            raise TypeError('`inferred` should be bool type')
        object.__setattr__(self, 'values', values)

    @classmethod
    def normalize(cls,
                  domain: _Domain,
                  n_sites: int,
                  samples: _Samples = None) -> '_DomainSpec':
        """Broadcasts explicit domains or infers them from input samples."""
        n_sites = _normalize_n_sites(n_sites)
        if domain is None:
            sample_values = _split_samples(samples, n_sites)
            values = tuple(
                torch.unique(site_values, dim=0)
                for site_values in sample_values)
            return cls(values, inferred=True)

        if isinstance(domain, torch.Tensor):
            return cls((domain,) * n_sites)
        if isinstance(domain, (str, bytes)):
            raise TypeError(
                '`domain` should be a tensor or a sequence of tensors')
        try:
            values = tuple(domain)
        except TypeError as exc:
            raise TypeError(
                '`domain` should be a tensor or a sequence of tensors') \
                from exc
        if len(values) != n_sites:
            raise ValueError(
                '`domain` should contain one tensor per input site')
        return cls(values)

    @property
    def n_sites(self) -> int:
        """Number of input sites."""
        return len(self.values)

    @property
    def n_values(self) -> Tuple[int, ...]:
        """Number of finite domain values at every site."""
        return tuple(value.shape[0] for value in self.values)

    @property
    def coordinate_shape(self) -> Tuple[Tuple[int, ...], ...]:
        """Shape of one coordinate at every site."""
        return tuple(tuple(value.shape[1:]) for value in self.values)

    def for_site(self, site: int) -> torch.Tensor:
        """Returns the domain associated with one input site."""
        return self.values[_validate_site(site, self.n_sites)]


@dataclass(frozen=True)
class _EmbeddingSpec:
    """Caches validated site embeddings and their finite-domain matrices."""

    embeddings: Sequence[_Embedding]
    domains: _DomainSpec
    matrices: Sequence[torch.Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.domains, _DomainSpec):
            raise TypeError('`domains` should be _DomainSpec type')
        embeddings = tuple(self.embeddings)
        matrices = tuple(self.matrices)
        if len(embeddings) != self.domains.n_sites:
            raise ValueError(
                '`embeddings` should contain one entry per input site')
        if len(matrices) != self.domains.n_sites:
            raise ValueError(
                '`matrices` should contain one entry per input site')
        for site, (embedding, matrix) in enumerate(zip(
                embeddings, matrices)):
            if not (callable(embedding) or isinstance(embedding, torch.Tensor)):
                raise TypeError(
                    f'Embedding at site {site} should be callable or a tensor')
            self._validate_matrix(site, matrix)
        object.__setattr__(self, 'embeddings', embeddings)
        object.__setattr__(self, 'matrices', matrices)

    @classmethod
    def normalize(cls,
                  embedding,
                  domains: _DomainSpec) -> '_EmbeddingSpec':
        """Broadcasts embeddings and evaluates every finite domain once."""
        if not isinstance(domains, _DomainSpec):
            raise TypeError('`domains` should be _DomainSpec type')
        if callable(embedding) or isinstance(embedding, torch.Tensor):
            embeddings = (embedding,) * domains.n_sites
        else:
            if isinstance(embedding, (str, bytes)):
                raise TypeError(
                    '`embedding` should be callable, a tensor or a sequence')
            try:
                embeddings = tuple(embedding)
            except TypeError as exc:
                raise TypeError(
                    '`embedding` should be callable, a tensor or a sequence') \
                    from exc
            if len(embeddings) != domains.n_sites:
                raise ValueError(
                    '`embedding` should contain one entry per input site')

        matrices = []
        for site, (entry, domain) in enumerate(zip(
                embeddings, domains.values)):
            if callable(entry):
                try:
                    matrix = entry(domain)
                except Exception as exc:
                    raise ValueError(
                        f'Embedding at site {site} failed on its domain') \
                        from exc
            elif isinstance(entry, torch.Tensor):
                matrix = entry
            else:
                raise TypeError(
                    f'Embedding at site {site} should be callable or a tensor')
            cls._validate_matrix_for_domain(site, matrix, domain)
            matrices.append(matrix)
        return cls(embeddings, domains, matrices)

    @staticmethod
    def _validate_matrix_for_domain(site: int,
                                    matrix: torch.Tensor,
                                    domain: torch.Tensor) -> None:
        """Validates one cached embedding matrix against its site domain."""
        if not isinstance(matrix, torch.Tensor):
            raise TypeError(
                f'Embedding at site {site} should return a torch.Tensor')
        if matrix.ndim != 2:
            raise ValueError(
                f'Embedding at site {site} should return shape '
                '(n_values, input_dim)')
        if matrix.shape[0] != domain.shape[0]:
            raise ValueError(
                f'Embedding at site {site} returned {matrix.shape[0]} rows '
                f'for a domain with {domain.shape[0]} values')
        if matrix.shape[1] < 1:
            raise ValueError(
                f'Embedding at site {site} should have positive input_dim')
        if not (matrix.is_floating_point() or matrix.is_complex()):
            raise TypeError(
                f'Embedding at site {site} should be floating or complex')
        if not torch.isfinite(matrix).all():
            raise ValueError(f'Embedding at site {site} should be finite')

    def _validate_matrix(self, site: int, matrix: torch.Tensor) -> None:
        """Validates one matrix against the already-normalized domain."""
        self._validate_matrix_for_domain(
            site, matrix, self.domains.for_site(site))

    @property
    def n_sites(self) -> int:
        """Number of input sites."""
        return self.domains.n_sites

    @property
    def input_dim(self) -> Tuple[int, ...]:
        """Embedding dimension associated with every input site."""
        return tuple(matrix.shape[1] for matrix in self.matrices)

    def matrix(self, site: int) -> torch.Tensor:
        """Returns the cached finite-domain embedding matrix for one site."""
        return self.matrices[_validate_site(site, self.n_sites)]

    def evaluate(self,
                 site: int,
                 values: torch.Tensor) -> torch.Tensor:
        """Embeds values or looks them up in a precomputed embedding table."""
        site = _validate_site(site, self.n_sites)
        if not isinstance(values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        entry = self.embeddings[site]
        if isinstance(entry, torch.Tensor):
            domain = self.domains.for_site(site)
            coordinate_shape = self.domains.coordinate_shape[site]
            if (values.ndim != (1 + len(coordinate_shape))) or \
                    (tuple(values.shape[1:]) != coordinate_shape):
                raise ValueError(
                    f'Values at site {site} should match coordinate shape '
                    f'{coordinate_shape}')
            flat_values = values.to(domain.device).reshape(values.shape[0], -1)
            flat_domain = domain.reshape(domain.shape[0], -1)
            matches = (flat_values[:, None] == flat_domain[None, :]).all(dim=2)
            found = matches.any(dim=1)
            if not torch.all(found):
                rows = torch.where(~found)[0].detach().cpu().tolist()
                raise ValueError(
                    f'Values at site {site} are outside its domain in rows '
                    f'{rows}')
            indices = matches.to(torch.int64).argmax(dim=1).to(entry.device)
            return entry.index_select(0, indices)

        coordinate_shape = self.domains.coordinate_shape[site]
        if (values.ndim != (1 + len(coordinate_shape))) or \
                (tuple(values.shape[1:]) != coordinate_shape):
            raise ValueError(
                f'Values at site {site} should have shape '
                f'(batch, {coordinate_shape})')
        try:
            result = entry(values)
        except Exception as exc:
            raise ValueError(f'Embedding at site {site} failed') from exc
        if not isinstance(result, torch.Tensor):
            raise TypeError(
                f'Embedding at site {site} should return a torch.Tensor')
        expected = (values.shape[0], self.input_dim[site])
        if result.shape != expected:
            raise ValueError(
                f'Embedding at site {site} should return shape {expected}')
        if not (result.is_floating_point() or result.is_complex()):
            raise TypeError(
                f'Embedding at site {site} should be floating or complex')
        if not torch.isfinite(result).all():
            raise ValueError(f'Embedding at site {site} should be finite')
        return result


def _default_output_positions(n_input_sites: int,
                              n_output_sites: int) -> Tuple[int, ...]:
    """Places output axes between maximally balanced input-site groups."""
    n_groups = n_output_sites + 1
    boundaries = (
        ((axis + 1) * n_input_sites + n_groups // 2) // n_groups
        for axis in range(n_output_sites))
    return tuple(boundary + axis
                 for axis, boundary in enumerate(boundaries))


@dataclass(frozen=True)
class _OutputSpec:
    """Maps tensor-output axes to ordered sites in the decomposed chain."""

    output_shape: Sequence[int]
    n_input_sites: int
    positions: Sequence[int] = ()

    def __post_init__(self) -> None:
        n_input_sites = _normalize_n_sites(self.n_input_sites)
        if isinstance(self.output_shape, (str, bytes)):
            raise TypeError('`output_shape` should be a sequence of integers')
        try:
            output_shape = tuple(self.output_shape)
        except TypeError as exc:
            raise TypeError(
                '`output_shape` should be a sequence of integers') from exc
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 1
               for dim in output_shape):
            raise ValueError(
                '`output_shape` should contain positive integers')
        try:
            positions = tuple(self.positions)
        except TypeError as exc:
            raise TypeError('`positions` should be a sequence of integers') \
                from exc
        if len(positions) != len(output_shape):
            raise ValueError(
                '`positions` should contain one site per output axis')
        if any(isinstance(position, bool) or not isinstance(position, int)
               for position in positions):
            raise TypeError('Output positions should be integers')
        n_sites = n_input_sites + len(output_shape)
        if any(position < 0 or position >= n_sites for position in positions):
            raise ValueError(
                f'Output positions should be between 0 and {n_sites - 1}')
        if any(left >= right for left, right in zip(
                positions, positions[1:])):
            raise ValueError(
                'Output positions should be distinct and strictly increasing')
        object.__setattr__(self, 'output_shape', output_shape)
        object.__setattr__(self, 'n_input_sites', n_input_sites)
        object.__setattr__(self, 'positions', positions)

    @classmethod
    def normalize(cls,
                  values: torch.Tensor,
                  n_input_sites: int,
                  out_position=None) -> '_OutputSpec':
        """Infers scalar/tensor output semantics from one evaluated batch."""
        if not isinstance(values, torch.Tensor):
            raise TypeError('Function output should be a torch.Tensor')
        if values.ndim < 1:
            raise ValueError('Function output should have a batch dimension')
        if values.shape[0] < 1:
            raise ValueError('Function output should contain at least one row')
        if not (values.is_floating_point() or values.is_complex()):
            raise TypeError('Function output should be floating or complex')
        if not torch.isfinite(values).all():
            raise ValueError('Function output should be finite')
        n_input_sites = _normalize_n_sites(n_input_sites)
        output_shape = tuple(values.shape[1:])
        if output_shape == (1,):
            output_shape = ()

        n_output_sites = len(output_shape)
        if not n_output_sites:
            if out_position not in (None, (), []):
                raise ValueError(
                    '`out_position` should be None for a scalar function')
            positions = ()
        elif out_position is None:
            positions = _default_output_positions(
                n_input_sites, n_output_sites)
        elif isinstance(out_position, bool):
            raise TypeError(
                '`out_position` should be int or a sequence of ints')
        elif isinstance(out_position, int):
            if n_output_sites != 1:
                raise ValueError(
                    'An integer `out_position` requires one output axis')
            positions = (out_position,)
        else:
            if isinstance(out_position, (str, bytes)):
                raise TypeError(
                    '`out_position` should be int or a sequence of ints')
            try:
                positions = tuple(out_position)
            except TypeError as exc:
                raise TypeError(
                    '`out_position` should be int or a sequence of ints') \
                    from exc
        return cls(output_shape, n_input_sites, positions)

    @property
    def n_output_sites(self) -> int:
        """Number of tensor-output axes represented as sites."""
        return len(self.output_shape)

    @property
    def n_sites(self) -> int:
        """Total sites after inserting every output axis."""
        return self.n_input_sites + self.n_output_sites

    @property
    def scalar(self) -> bool:
        """Whether the source has no explicit tensor-output axes."""
        return not self.output_shape

    @property
    def flat_dim(self) -> int:
        """Flattened row-major output dimension."""
        return prod(self.output_shape) if self.output_shape else 1

    @property
    def layout(self) -> Tuple[Tuple[str, int], ...]:
        """Returns ordered ``('input'|'output', axis)`` site descriptors."""
        output_by_site = {
            position: axis for axis, position in enumerate(self.positions)}
        input_axis = 0
        layout = []
        for site in range(self.n_sites):
            if site in output_by_site:
                layout.append(('output', output_by_site[site]))
            else:
                layout.append(('input', input_axis))
                input_axis += 1
        return tuple(layout)

    @property
    def input_positions(self) -> Tuple[int, ...]:
        """Final-chain positions occupied by original input variables."""
        return tuple(site for site, kind in enumerate(self.layout)
                     if kind[0] == 'input')

    def validate_values(self, values: torch.Tensor) -> torch.Tensor:
        """Validates a batch and removes the legacy scalar singleton axis."""
        if not isinstance(values, torch.Tensor):
            raise TypeError('Function output should be a torch.Tensor')
        if self.scalar:
            if values.ndim == 1:
                canonical = values
            elif values.ndim == 2 and values.shape[1] == 1:
                canonical = values.squeeze(1)
            else:
                raise ValueError(
                    'Scalar output should have shape (batch,) or (batch, 1)')
        else:
            expected_tail = self.output_shape
            if (values.ndim != (1 + len(expected_tail))) or \
                    (tuple(values.shape[1:]) != expected_tail):
                raise ValueError(
                    f'Tensor output should have shape (batch, {expected_tail})')
            canonical = values
        if not (canonical.is_floating_point() or canonical.is_complex()):
            raise TypeError('Function output should be floating or complex')
        if not torch.isfinite(canonical).all():
            raise ValueError('Function output should be finite')
        return canonical

    def flatten_labels(self, indices: torch.Tensor) -> torch.Tensor:
        """Flattens one index per output axis in row-major order."""
        if not isinstance(indices, torch.Tensor):
            raise TypeError('`indices` should be torch.Tensor type')
        if indices.ndim != 2 or indices.shape[1] != self.n_output_sites:
            raise ValueError(
                '`indices` should have shape (batch, n_output_sites)')
        if indices.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64,
                torch.uint8):
            raise TypeError('`indices` should contain integers')
        indices = indices.to(dtype=torch.long)
        for axis, dim in enumerate(self.output_shape):
            if torch.any(indices[:, axis] < 0) or \
                    torch.any(indices[:, axis] >= dim):
                raise ValueError(
                    f'Output indices at axis {axis} are out of bounds')
        strides = indices.new_tensor([
            prod(self.output_shape[axis + 1:])
            for axis in range(self.n_output_sites)])
        return (indices * strides).sum(dim=1)

    def unflatten_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Unflattens row-major labels to one index per output axis."""
        labels = self._validate_flat_labels(labels)
        if self.scalar:
            return labels.new_empty((labels.shape[0], 0))
        remainder = labels
        axes = [None] * self.n_output_sites
        for axis in range(self.n_output_sites - 1, -1, -1):
            dim = self.output_shape[axis]
            axes[axis] = torch.remainder(remainder, dim)
            remainder = torch.div(remainder, dim, rounding_mode='floor')
        return torch.stack(axes, dim=1)

    def _validate_flat_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Validates flattened row-major labels."""
        if not isinstance(labels, torch.Tensor):
            raise TypeError('`labels` should be torch.Tensor type')
        if labels.ndim != 1:
            raise ValueError('`labels` should have shape (batch,)')
        if labels.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64,
                torch.uint8):
            raise TypeError('`labels` should contain integers')
        labels = labels.to(dtype=torch.long)
        if torch.any(labels < 0) or torch.any(labels >= self.flat_dim):
            raise ValueError('`labels` contains an out-of-range output index')
        return labels

    def sample_labels(
            self,
            values: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            zero_policy: str = 'error') -> torch.Tensor:
        """Samples flattened labels proportionally to ``abs(values) ** 2``."""
        if self.scalar:
            raise ValueError('A scalar function has no output labels')
        if zero_policy not in ('error', 'uniform'):
            raise ValueError("`zero_policy` should be 'error' or 'uniform'")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator type or None')
        values = self.validate_values(values)
        weights = values.reshape(values.shape[0], -1).abs().square()
        norms = weights.sum(dim=1, keepdim=True)
        zero_rows = norms.squeeze(1) == 0
        if torch.any(zero_rows):
            if zero_policy == 'error':
                rows = torch.where(zero_rows)[0].detach().cpu().tolist()
                raise ValueError(
                    'Cannot sample output labels from zero-norm rows: '
                    f'{rows}')
            weights = weights.clone()
            weights[zero_rows] = 1
            norms = weights.sum(dim=1, keepdim=True)
        probabilities = weights / norms
        random_device = (
            probabilities.device if generator is None else generator.device)
        labels = torch.multinomial(
            probabilities.to(random_device),
            num_samples=1,
            replacement=True,
            generator=generator).squeeze(1)
        return labels.to(values.device)

    def resolve_labels(
            self,
            values: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            generator: Optional[torch.Generator] = None,
            zero_policy: str = 'error'
            ) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Returns flat labels, per-axis indices and selected output values."""
        values = self.validate_values(values)
        if self.scalar:
            if labels is not None:
                raise ValueError('`labels` should be None for a scalar function')
            indices = torch.empty(
                values.shape[0], 0, device=values.device, dtype=torch.long)
            return None, indices, values
        if labels is None:
            flat_labels = self.sample_labels(
                values, generator=generator, zero_policy=zero_policy)
        else:
            flat_labels = self._validate_flat_labels(labels).to(values.device)
            if flat_labels.shape[0] != values.shape[0]:
                raise ValueError(
                    '`labels` and function output should share batch size')
        output_indices = self.unflatten_labels(flat_labels)
        selected = values.reshape(values.shape[0], -1).gather(
            1, flat_labels.unsqueeze(1)).squeeze(1)
        return flat_labels, output_indices, selected

    def insert_indices(
            self,
            samples: Union[torch.Tensor, Sequence[torch.Tensor]],
            output_indices: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Inserts one discrete index tensor per output axis into samples."""
        input_values = _split_samples(samples, self.n_input_sites)
        if not isinstance(output_indices, torch.Tensor):
            raise TypeError('`output_indices` should be torch.Tensor type')
        if output_indices.ndim != 2 or \
                output_indices.shape[1] != self.n_output_sites:
            raise ValueError(
                '`output_indices` should have shape '
                '(batch, n_output_sites)')
        if output_indices.shape[0] != input_values[0].shape[0]:
            raise ValueError(
                '`output_indices` and samples should share batch size')
        self.flatten_labels(output_indices)
        result = []
        for kind, axis in self.layout:
            result.append(
                input_values[axis] if kind == 'input'
                else output_indices[:, axis])
        return tuple(result)

    def site_dim(self,
                 embeddings: _EmbeddingSpec) -> Tuple[int, ...]:
        """Returns input/output dimension at every final chain site."""
        if not isinstance(embeddings, _EmbeddingSpec):
            raise TypeError('`embeddings` should be _EmbeddingSpec type')
        if embeddings.n_sites != self.n_input_sites:
            raise ValueError(
                'Embeddings and output layout should have matching inputs')
        return tuple(
            embeddings.input_dim[axis] if kind == 'input'
            else self.output_shape[axis]
            for kind, axis in self.layout)

    def embed_site(self,
                   site: int,
                   values: torch.Tensor,
                   embeddings: _EmbeddingSpec,
                   dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Uses the input embedding or exact basis required at one site."""
        site = _validate_site(site, self.n_sites)
        kind, axis = self.layout[site]
        if kind == 'input':
            return embeddings.evaluate(axis, values)
        if not isinstance(values, torch.Tensor):
            raise TypeError('`values` should be torch.Tensor type')
        if values.ndim != 1 or values.dtype not in (
                torch.int8, torch.int16, torch.int32, torch.int64,
                torch.uint8):
            raise TypeError('Output-site values should be integer indices')
        if torch.any(values < 0) or \
                torch.any(values >= self.output_shape[axis]):
            raise ValueError(f'Output indices at axis {axis} are out of bounds')
        embedded = basis(values, dim=self.output_shape[axis])
        return embedded if dtype is None else embedded.to(dtype=dtype)


@dataclass(frozen=True)
class _SketchingFitSpec:
    """Groups truncation, range projection, batching and diagnostics."""

    rank: Optional[int] = None
    cutoff: Optional[float] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None
    cum_percentage: Optional[float] = None
    random_projection: bool = True
    projection_dim: Optional[int] = None
    batch_size: int = 64
    verbose: Union[bool, int] = 0
    collect_metrics: bool = False
    truncation: _TruncationSpec = field(init=False, repr=False)
    verbosity: int = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool):
            raise TypeError('`rank` should be int type or None')
        truncation = _TruncationSpec(
            rank=self.rank,
            cutoff=self.cutoff,
            atol=self.atol,
            rtol=self.rtol,
            cum_percentage=self.cum_percentage)
        if not isinstance(self.random_projection, bool):
            raise TypeError('`random_projection` should be bool type')
        if self.projection_dim is not None:
            if isinstance(self.projection_dim, bool) or \
                    not isinstance(self.projection_dim, int):
                raise TypeError('`projection_dim` should be int type or None')
            if self.projection_dim < 1:
                raise ValueError('`projection_dim` should be positive')
            if not self.random_projection:
                raise ValueError(
                    '`projection_dim` requires `random_projection=True`')
        if isinstance(self.batch_size, bool) or \
                not isinstance(self.batch_size, int):
            raise TypeError('`batch_size` should be int type')
        if self.batch_size < 1:
            raise ValueError('`batch_size` should be positive')
        if not isinstance(self.collect_metrics, bool):
            raise TypeError('`collect_metrics` should be bool type')
        object.__setattr__(self, 'truncation', truncation)
        object.__setattr__(
            self, 'verbosity', _normalize_verbosity(self.verbose))

    @property
    def effective_projection_dim(self) -> Optional[int]:
        """Projection dimension, with ``rank`` as the reducing default."""
        if not self.random_projection:
            return None
        return self.rank if self.projection_dim is None else self.projection_dim

    @property
    def diagnostics_enabled(self) -> bool:
        """Whether fitting should collect structured diagnostic records."""
        return self.collect_metrics or bool(self.verbosity)


__all__ = [
    '_EmbeddingSpec',
    '_DomainSpec',
    '_OutputSpec',
    '_SketchingFitSpec',
]
