Decompositions
==============

.. currentmodule:: tensorkrowch.decompositions

Formats and fit metadata
--------------------------

Results inherit the numerical representations documented in :doc:`formats`.
``TTDecomposition`` is a ``TT``, and the TR/TTM/TRM families follow
the same relationship. Their additional ``metrics`` and ``metadata`` describe
the original fit. Copying, detaching or moving a result preserves this history;
editing or rounding its cores does not recompute historical errors. New exact
algebra returns numerical formats without attaching a fictitious fit history.
Use ``in_dim`` and ``out_dim`` for dimensions, and ``as_info`` to inspect
metadata and diagnostics.

Input sources
-------------

Sources inherit :class:`BaseSource`, the common base for finite tensors and
continuous functions. It provides evaluation statistics and generic
:meth:`BaseSource.fiber` and :meth:`BaseSource.contract_dense` methods;
concrete sources implement :meth:`BaseSource.evaluate` and may specialize
these operations. ``contract_dense`` requires a finite discrete tensor.

``source`` describes the input independently of the decomposition algorithm:
its dimensions, evaluation rules, matrix grouping and, when applicable,
Quantics discretization. Finite tensor sources are :class:`DenseTensorSource`,
:class:`SparseTensorSource`, :class:`FormatSource`,
:class:`CallableTensorSource` and :class:`CallableQuanticsSource`.
Continuous functions use :class:`CallableSource`; they require discretization
before SVD or ALS, whereas RSS can fit them with an embedding.

Matrix inputs use :meth:`DenseTensorSource.matrix`,
:meth:`SparseTensorSource.matrix`, :meth:`CallableTensorSource.matrix` or
:meth:`CallableQuanticsSource.matrix`. Continuous matrix functions can use
:meth:`CallableSource.matrix` with RSS. :class:`FormatSource` obtains matrix
grouping directly from a matrix format. These objects carry the input details;
the algorithm controls how to approximate their values.

``DenseTensorSource.matrix`` receives a tensor with explicit input/output axes:
``ordering="interleaved"`` uses ``(in_0, out_0, in_1, out_1, ...)``, whereas
the default ``ordering="grouped"`` uses
``(in_0, in_1, ..., out_0, out_1, ...)``.
Dimensions are inferred; a two-axis tensor represents one site. To represent
several sites from an ordinary matrix, reshape its axes before construction.
Raw tensors passed directly to ``TTMSVD`` or ``TRMSVD``, and to their function
interfaces, also use grouped ordering. Interleaved input tensors require an
explicit source with ``ordering="interleaved"``. The legacy
:func:`mat_to_mpo` retains interleaved inputs.

``SparseTensorSource.matrix`` uses the same ``ordering`` conventions for the
columns of its entry indices, defaulting to grouped columns. It also requires
``in_dim`` and ``out_dim``, since sparse entries do not determine full dimensions.

Quantics preparation
--------------------

Quantics discretization belongs to the callable source. Construct a
:class:`CallableQuanticsSource` with ``base``, ``level`` and ``domain`` (or
``grid_coordinates``), or supply a prebuilt
:class:`~tensorkrowch.formats.QuantizedLayout` and
:class:`~tensorkrowch.formats.CoordinateMap`. Dense tensors and tensor sources
retain their fixed axes and are never automatically split into digits.

``TTSVD(source)`` and ``TRSVD(source)`` materialize the full digit tensor.
``TTALS(source)`` and ``TRALS(source)`` query finite entries during fitting.
``TTRSS(source)`` and ``TRRSS(source)`` use basis embeddings for discrete
sources, including Quantics. Domain-coordinate samples are converted by the
source's map and layout; an index ``ConfigurationBatch`` supplies digits
explicitly. Continuous sources retain user-supplied embeddings.

.. code-block:: python

   import torch
   import tensorkrowch as tk

   source = tk.decompositions.CallableQuanticsSource(
       lambda coordinates: torch.exp(coordinates.sum(-1)),
       n_coordinates=2, base=2, level=(3, 4),
       domain=torch.tensor([[0., 1.], [-1., 1.]]))
   qtt = tk.decompositions.qtt_svd(source, rank=2)
   fitted = tk.decompositions.TTALS(source).fit(
       initial_cores=qtt,
       convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
   coordinates = torch.tensor([[0., -1.], [0.5, 0.]])
   values = fitted.evaluate_coordinates(coordinates)

For matrix sources, the shared algorithms fuse input/output pairs and return
the corresponding matrix format. Quantics results retain their maps and
layouts independently of the source.

Dataset sources use ``from_data``. Discrete empirical masses belong to
``SparseTensorSource``; continuous values belong to ``CallableSource``;
``CallableQuanticsSource`` discretizes the dataset first. Callable routes can
use :func:`gaussian_kernel` or a custom kernel with its own keyword arguments.

For ALS, ``mask`` is another boolean source: ``True`` participates and
``False`` is missing. Without a mask, sparse entries outside the stored
support are known zeros. A sparse boolean mask coalesces duplicate entries
with logical OR and can include observed numerical zeros.

SVD decompositions
------------------

SVD engines materialize a finite source once at construction and retain its
prepared tensor for independent calls to ``fit``. Sources own dimensions,
matrix ordering and evaluation runtime; ``fit`` controls truncation,
normalization and diagnostics.

SVD accepts the finite sources listed above and tensors directly. The general
:class:`TTSVD` and :class:`TRSVD` engines also accept non-Quantics TT/TR formats
and their matrix versions directly. Wrap an :class:`~tensorkrowch.models.MPS`
in :class:`FormatSource` first. Matrix engines require a source declaring
paired dimensions, or a tensor with grouped input/output axes; wrap matrix
formats in ``FormatSource``. All source representations are materialized for
the numerical SVD, including sparse and callable tensors.

Engine hierarchy and algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 18 18 44 20

   * - Engine
     - Inherits from
     - Numerical workflow
     - Ordinary result
   * - :class:`TTSVD`
     - Independent engine
     - Successive open-chain SVD cuts through :meth:`TTSVD.fit`.
     - :class:`TTDecomposition`
   * - :class:`TTMSVD`
     - :class:`TTSVD`
     - Inherited TT-SVD over fused input/output pairs.
     - :class:`TTMDecomposition`
   * - :class:`TRSVD`
     - Independent engine
     - Interior bipartition followed by two TT-SVD subchains through
       :meth:`TRSVD.fit`.
     - :class:`TRDecomposition`
   * - :class:`TRMSVD`
     - :class:`TRSVD`
     - Inherited TR-SVD over fused input/output pairs.
     - :class:`TRMDecomposition`

:class:`TRSVD` reuses the numerical routines of :class:`TTSVD` through
composition, rather than inheritance. Its initial SVD splits the tensor at
``center``; the retained rank is factored into the closing rank and the rank
crossing that cut. TT-SVD then decomposes each block with those boundary ranks
preserved. TT-SVD supports leading structural batches; TR-SVD does not.

The matrix engines restrict inputs to matrix sources. The general
:class:`TTSVD` and :class:`TRSVD` engines also accept those sources and infer
the matrix result type. Matrix dimensions and ordering belong to the source,
not to the numerical sweep.

Construction and fitting workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Describe the input.** Pass a tensor, format or finite source as described
   above. Plain tensors become :class:`DenseTensorSource` objects. Matrix
   sources declare paired dimensions through their ``.matrix(...)``
   constructors or a wrapped matrix format. Coordinate functions use
   :class:`CallableQuanticsSource`.
2. **Construct the engine.** ``TTSVD(source)`` or ``TRSVD(source)`` calls the
   source's ``contract_dense`` method and retains the dense tensor. Matrix
   input/output pairs are fused into one dimension per site. Quantics sources
   evaluate their full digit grid; ordinary tensor sources retain their fixed
   axes without quantization. ``engine.source`` retains the original source.
3. **Fit the prepared tensor.** :meth:`TTSVD.fit` or :meth:`TRSVD.fit` applies
   the requested ``rank``, ``cutoff``, ``atol``, ``rtol`` or ``cum_percentage``
   criteria through :func:`~tensorkrowch.utils.truncated_svd`.
   ``renormalize`` controls numerical scaling; ``collect_metrics=True``
   records errors and timings, and ``verbose`` controls progress reporting.
4. **Restore the representation.** The result reopens paired matrix axes and
   retains the source's evaluation sites. Quantics inputs also retain their
   layouts and coordinate maps, producing :class:`QTTDecomposition`,
   :class:`QTRDecomposition`, :class:`QTTMDecomposition` or
   :class:`QTRMDecomposition` through the same numerical algorithms.

Repeated fits reuse the prepared tensor with different truncation options:

.. code-block:: python

   source = tk.decompositions.DenseTensorSource(torch.eye(4))
   engine = tk.decompositions.TTSVD(source)
   rank_1 = engine.fit(rank=1)
   rank_2 = engine.fit(rank=2, collect_metrics=True)

Access ``result.cores`` for the tensors or
:meth:`TensorDecomposition.as_info` for structured information. Use
:meth:`TensorDecomposition.to_format` to continue with the numerical format
without fit metadata. The engines share source preparation, stable
normalization, logarithmic norm calculations, truncation and progress helpers;
these do not introduce additional public engine classes.

Functional interfaces
^^^^^^^^^^^^^^^^^^^^^

Each functional interface constructs its engine, performs one ``fit`` and
returns the decomposition object directly:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Function
     - Engine
     - Quantics function
   * - :func:`tt_svd`
     - :class:`TTSVD`
     - :func:`qtt_svd`
   * - :func:`tr_svd`
     - :class:`TRSVD`
     - :func:`qtr_svd`
   * - :func:`ttm_svd`
     - :class:`TTMSVD`
     - :func:`qttm_svd`
   * - :func:`trm_svd`
     - :class:`TRMSVD`
     - :func:`qtrm_svd`

The Quantics functions require a :class:`CallableQuanticsSource` of the
appropriate vector or matrix kind and delegate to the corresponding ordinary
function. There are no separate Quantics engines or SVD algorithms.

TT-SVD
^^^^^^

.. autoclass:: TTSVD
   :members: fit

.. autofunction:: tt_svd

.. autofunction:: qtt_svd

TTM-SVD
^^^^^^^

.. autoclass:: TTMSVD
   :members: fit

.. autofunction:: ttm_svd

.. autofunction:: qttm_svd

TR-SVD
^^^^^^

.. autoclass:: TRSVD
   :members: fit

.. autofunction:: tr_svd

.. autofunction:: qtr_svd

TRM-SVD
^^^^^^^

.. autoclass:: TRMSVD
   :members: fit

.. autofunction:: trm_svd

.. autofunction:: qtrm_svd

Lightweight results
-------------------

Results add historical diagnostics to their numerical format. Use
:meth:`TensorDecomposition.to_format` to obtain the corresponding format
without these diagnostics, retaining its cores, bonds and coordinate maps.

.. autoclass:: TensorDecomposition
    :members: as_info, error, to_format

.. autoclass:: QTTDecomposition

.. autoclass:: QTRDecomposition

.. autoclass:: QTTMDecomposition

.. autoclass:: QTRMDecomposition

.. autoclass:: TTDecomposition
   :members:

.. autoclass:: TTMDecomposition
   :members:

.. autoclass:: TRDecomposition
   :members:

.. autoclass:: TRMDecomposition
   :members:

Compatibility with official releases
------------------------------------

The historical names below remain available for compatibility and emit a
deprecation warning. New code should use :func:`tt_svd` and :func:`ttm_svd`.

.. autofunction:: vec_to_mps

.. autofunction:: mat_to_mpo

Sketching decompositions
------------------------

The class interfaces keep the source, embeddings, domains and output layout
fixed, and allow several independent ``fit`` calls with different ranks or
sketch samples. The functional interfaces construct the corresponding class
and return a list of cores for direct use in TensorKrowch models. Use the class
when structured metrics, metadata or repeated fits are needed.

:class:`TTRSS` and :class:`TRRSS` accept the finite sources listed above,
tensors, :class:`CallableSource` and raw callables. Wrap formats or MPS models
in :class:`FormatSource`. Continuous inputs require an embedding; finite
sources use basis embeddings by default. :class:`TTRS` and :class:`TRRS`
require finite tensor sources and also adapt tensors, non-Quantics formats
and MPS models directly. These sketching methods do not support structural
batches.

Maturity
^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 22 50

   * - Interface
     - Status
     - Scope
   * - :class:`TTRSS`, :func:`tt_rss`
     - Stable
     - Sampled recursive sketching for scalar and tensor-valued functions.
   * - :class:`TTRS`, :func:`tt_rs`
     - Experimental
     - Full projection of discrete, sparse, empirical or TT sources.
   * - :class:`TRRSS`, :func:`tr_rss`
     - Experimental
     - Cyclic RSS with configurable loop opening and gauge recursion.
   * - :class:`TRRS`, :func:`tr_rs`
     - Experimental
     - Cyclic extension of full-source recursive sketching.
   * - :func:`qtt_rss`, :func:`qtr_rss`
     - Experimental
     - Quantized coordinate adapters over TT-RSS and TR-RSS.

Sampled recursive sketching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TTRSS
   :members: fit

.. autofunction:: tt_rss

.. autoclass:: TRRSS
   :members: fit

.. autofunction:: tr_rss

Full-source recursive sketching
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: TTRS
   :members: fit

.. autofunction:: tt_rs

.. autoclass:: TRRS
   :members: fit

.. autofunction:: tr_rs

.. autoclass:: SampledSketch
   :members:

.. autoclass:: MarginalSketch
   :members:

.. autoclass:: TTStackSketch
   :members:

Quantized sketching
^^^^^^^^^^^^^^^^^^^^

``QuantizedLayout`` describes the digit sites independently of the physical
coordinate map. ``grouped`` orders all digits of each variable together;
``interleaved`` groups digits by level. Physical samples remain in the
original variable coordinates for both layouts.

.. autoclass:: QuantizedLayout
   :members:
   :noindex:

.. autoclass:: AffineCoordinateMap
   :members:
   :noindex:

.. autoclass:: FunctionalCoordinateMap
   :members:
   :noindex:

.. autoclass:: ExplicitGridMap
   :members:
   :noindex:

.. autofunction:: qtt_rss

.. autofunction:: qtr_rss

Sources and fitting strategies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The source abstraction is shared by SVD, ALS and sketching.
:class:`TTALS` and :class:`TRALS` accept finite sources, tensors,
non-Quantics formats and MPS models, adapting the latter inputs automatically.
They require ``is_discrete_tensor=True`` and do not support structural batches.
Specialized sparse and format sources evaluate their native representations
directly instead of building a TensorKrowch graph; ALS and sketching can query
these values without materializing the complete tensor.

.. autoclass:: BaseSource
   :members:

.. autoclass:: DenseTensorSource
   :members:

.. autoclass:: CallableSource
   :members:

.. autoclass:: CallableTensorSource
   :members:

.. autoclass:: CallableQuanticsSource
   :members: evaluate, fiber, matrix, from_data, contract_dense_grid_axes, contract_dense

.. autoclass:: SparseTensorSource
   :members:

.. autoclass:: FormatSource
   :members:

.. autoclass:: FixedEmbeddingFitter
   :members:

.. autoclass:: BasisFitter
   :members:

.. autoclass:: TrainableEmbeddingFitter
   :members:

.. autoclass:: QTTInputFitter
   :members:

Examples
^^^^^^^^

Scalar function and repeated TT-RSS fits:

.. code-block:: python

   domain = torch.linspace(0, 1, 8)
   samples = torch.rand(256, 4)
   embedding = lambda x: torch.stack((1 - x, x), dim=-1)
   function = lambda x: torch.exp(-x.square().sum(dim=1))

   decomposer = tk.decompositions.TTRSS(
       function, embedding=embedding, domain=domain)
   rank_4 = decomposer.fit(samples, rank=4)
   rank_8 = decomposer.fit(samples, rank=8)
   model = tk.models.MPS(tensors=rank_8.cores, parameterized=False)

Tensor-valued output with two output sites:

.. code-block:: python

   def tensor_function(x):
       value = x.sum(dim=1)
       return torch.stack(
           (value, value.square(), value.sin(), value.cos()), dim=1
       ).reshape(-1, 2, 2)

   labels = torch.arange(samples.shape[0]).remainder(4)
   result = tk.decompositions.TTRSS(
       tensor_function,
       embedding=embedding,
       domain=domain,
       out_position=(1, 4),
   ).fit(samples, labels=labels, rank=8)

Sparse and TT-backed sources:

.. code-block:: python

   indices = torch.tensor([[0, 0], [0, 1], [1, 1]])
   values = torch.tensor([1., 2., 3.])
   sparse = tk.decompositions.SparseTensorSource(
       indices, values, in_dim=(2, 2))
   sparse_result = tk.decompositions.TTRS(sparse).fit(rank=2)

   tt_source = tk.decompositions.FormatSource(sparse_result)
   projected_again = tk.decompositions.TTRS(tt_source).fit(rank=2)

Quantized RSS for a two-variable physical function:

.. code-block:: python

   grid = torch.linspace(0, 1, 8)
   samples = torch.cartesian_prod(grid, grid)
   function = lambda x: (1 + x).prod(dim=1)

   cores = tk.decompositions.qtt_rss(
       function,
       samples,
       n_coordinates=2,
       base=2,
       level=3,
       domain=torch.tensor([0., 1.]),
       rank=4,
   )
