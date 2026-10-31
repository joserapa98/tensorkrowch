"""
Sources shared by tensor decomposition algorithms.

Sources give decomposition algorithms a common interface for evaluating values,
querying dimensions and, for finite tensors, materializing all entries. They
describe the data independently of the algorithm that will approximate it.

Public source hierarchy:

    BaseSource
        ├─ DenseTensorSource       Values stored in a torch.Tensor.
        ├─ SparseTensorSource      Indexed entries; unstored entries are zero.
        ├─ FormatSource            Tensor formats, MPS/MPO models or TT cores.
        └─ CallableSource          Functions of continuous inputs.
            └─ CallableTensorSource
                Functions of finite discrete indices.
                └─ CallableQuanticsSource
                    Coordinate functions evaluated on a Quantics grid.

:class:`BaseSource` defines evaluation, dimensions, dtype, device and query
statistics. :class:`DenseTensorSource`, :class:`SparseTensorSource` and
:class:`FormatSource` obtain these from their stored tensor or format.
:class:`CallableSource` evaluates user functions; :class:`CallableTensorSource`
adds finite input dimensions, and :class:`CallableQuanticsSource` converts
digit indices into domain coordinates before evaluating the function.

Construction:

    torch.Tensor ───────────────────> DenseTensorSource
    indices + values + dimensions ──> SparseTensorSource
    format / MPS / MPO / TT cores ──> FormatSource
    function + variable count ──────> CallableSource
    function + input dimensions ────> CallableTensorSource
    function + layout and grid ────> CallableQuanticsSource

    as_tensor_source(tensor / format / model / source)
        └─ wraps the object or retains an existing source

:func:`as_tensor_source` handles tensors, formats and models; functions need an
explicit callable source to specify their inputs. Dense, sparse and callable
sources provide ``.matrix(...)`` constructors for paired input/output axes;
``FormatSource`` reads this structure from the format. Sparse and callable
``.from_data(...)`` constructors build sources from weighted observations.
Callable dataset constructors can also use a kernel such as
:func:`~tensorkrowch.decompositions.kernels.gaussian_kernel`.

Evaluation:

    tensor or per-input columns ─> ConfigurationBatch
        ├─ source.evaluate(configurations) ─> values
        └─ source.fiber(configurations, site, candidates)
            └─ values with one input varied over the candidates

    finite source ─> source.contract_dense() ─> complete tensor

    Quantics digit indices ─> QuantizedLayout ─> grid indices
        └─ CoordinateMap ─> domain coordinates ─> callable values

:class:`ConfigurationBatch` groups evaluation inputs; its ``n_inputs`` equals
``len(source.in_sites)``. Configuration columns follow the order of
``in_sites``, while ``n_sites`` counts all represented variables. For vector
sources, ``in_dim`` contains every variable dimension and ``out_shape`` contains
the dimensions left open by ``in_sites``, in their original order. Continuous
sources normally have ``in_dim=None``. Matrix sources evaluate one input per
paired site and return every paired output axis, so ``out_shape == out_dim``.

``contract_dense`` preserves the original variable order, with interleaved
input/output axes for matrices. Structural batch axes, when supported, precede
these axes. Evaluation instead places its configuration batch first, followed
by structural batch axes and ``out_shape``.

Use in decomposition algorithms:

    finite source ─> SVD materialization / indexed ALS or RS queries
    continuous or finite source + configurations ─> RSS queries
    source matrix/Quantics metadata ─> corresponding result format

Algorithms impose their own input restrictions. Sources supply values and
metadata; they do not define a fitting loss or decide which entries are
missing. In particular, unstored sparse entries are zero, not missing
observations. In ALS a separate boolean mask selects participating entries,
including entries whose target value is zero.
"""

from tensorkrowch.decompositions.sources.base import (ConfigurationBatch,
                                                      BaseSource)
from tensorkrowch.decompositions.sources.dense import DenseTensorSource
from tensorkrowch.decompositions.sources.sparse import SparseTensorSource
from tensorkrowch.decompositions.sources.format import FormatSource
from tensorkrowch.decompositions.sources.callable import (CallableSource,
                                                          CallableTensorSource)
from tensorkrowch.decompositions.sources.quantics import CallableQuanticsSource
from tensorkrowch.decompositions.sources.factory import as_tensor_source

__all__ = [
    'ConfigurationBatch',
    'BaseSource',
    'DenseTensorSource',
    'SparseTensorSource',
    'FormatSource',
    'CallableSource',
    'CallableTensorSource',
    'CallableQuanticsSource',
    'as_tensor_source',
]
