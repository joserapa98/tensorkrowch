Decompositions
==============

.. currentmodule:: tensorkrowch.decompositions

Formats and fit provenance
--------------------------

Results inherit the numerical representations documented in :doc:`formats`.
``TTDecomposition`` is a ``TT``, and the TR/TTM/TRM families follow
the same relationship. Their additional ``metrics`` and ``metadata`` describe
the original fit. Copying, detaching or moving a result preserves this history;
editing or rounding its cores does not recompute historical errors. New exact
algebra returns numerical formats without attaching a fictitious fit history.
Use ``in_dim`` and ``out_dim`` for dimensions, and ``as_info`` to inspect
metadata and diagnostics.

Quantics preparation
--------------------

Quantics discretization belongs to the callable source. Construct a
:class:`QuanticsVectorSource` with ``base``, ``level`` and ``domain`` (or
``grid_coordinates``), or supply a prebuilt
:class:`~tensorkrowch.formats.QuantizedLayout` and
:class:`~tensorkrowch.formats.CoordinateMap`. Dense tensors and tensor sources
retain their fixed axes and are never automatically split into digits.

``TTSVD.quantized(source)`` and ``TRSVD.quantized(source)`` materialize the
full digit tensor before fitting. ``TTALS.quantized(source)`` and
``TRALS.quantized(source)`` query the source on digit configurations during
runtime. RSS ``quantized`` and ``qtt_rss``/``qtr_rss`` use basis embeddings
at every digit site; their samples are domain coordinates by default, or
already encoded digits with ``sample_space="digits"``. Ordinary RSS retains
continuous callable evaluation with a user-supplied embedding.

.. code-block:: python

   import torch
   import tensorkrowch as tk

   source = tk.decompositions.QuanticsVectorSource(
       lambda coordinates: torch.exp(coordinates.sum(-1)),
       n_coordinates=2, base=2, level=(3, 4),
       domain=torch.tensor([[0., 1.], [-1., 1.]]))
   qtt = tk.decompositions.TTSVD.quantized(source).fit(rank=2)
   fitted = tk.decompositions.TTALS.quantized(source).fit(
       initial_cores=qtt,
       convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
   coordinates = torch.tensor([[0., -1.], [0.5, 0.]])
   values = fitted.evaluate_coordinates(coordinates)

For matrix SVD, :class:`QuanticsMatrixSource` owns independent input/output
layouts and coordinate maps. ``TTMSVD.quantized(source)`` and
``TRMSVD.quantized(source)`` materialize paired digit axes and return
QTTM/QTRM results. Matrix ALS/RSS is not implemented.

Quantics results retain their layouts and maps independently of the source.
``to_dense_grid`` restores the original coordinate axes;
``to_dense_digits`` on the source returns the digit axes used by SVD.
Completion remains a problem on explicitly supplied discrete tensor indices.

SVD decompositions
------------------

TT-SVD
^^^^^^

.. autoclass:: TTSVD
   :members: fit, quantized

.. autofunction:: tt_svd

TTM-SVD
^^^^^^^

.. autoclass:: TTMSVD
   :members: fit, quantized

.. autofunction:: ttm_svd

TR-SVD
^^^^^^

.. autoclass:: TRSVD
   :members: fit, quantized

.. autofunction:: tr_svd

TRM-SVD
^^^^^^^

.. autoclass:: TRMSVD
   :members: fit, quantized

.. autofunction:: trm_svd

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
   :members: fit, quantized

.. autofunction:: tt_rss

.. autoclass:: TRRSS
   :members: fit, quantized

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

The source abstraction is shared by ALS and sketching. Specialized sparse and
TT sources evaluate their native representations directly instead of building
a TensorKrowch graph or densifying the complete tensor.

.. autoclass:: CallableTensorSource
   :members:

.. autoclass:: QuanticsVectorSource
   :members: evaluate, evaluate_digits, evaluate_indices, evaluate_coordinates,
             coordinates_to_digits, digits_to_coordinates, fiber,
             to_dense_grid, to_dense_digits

.. autoclass:: QuanticsMatrixSource
   :members: evaluate_digits, evaluate_indices, evaluate_coordinates,
             coordinates_to_digits, digits_to_coordinates,
             to_dense_grid, to_dense_digits

.. autoclass:: SparseTensorSource
   :members:

.. autoclass:: EmpiricalDistribution
   :members:

.. autoclass:: TTTensorSource
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

   tt_source = tk.decompositions.TTTensorSource(sparse_result)
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
