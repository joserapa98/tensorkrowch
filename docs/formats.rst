Formats
=======

.. currentmodule:: tensorkrowch.formats

Formats represent tensors and operators as chains of PyTorch cores. Construct
one from its cores, then evaluate entries, contract it, or use its numerical
operations. They do not create TensorKrowch graph nodes; graph-based models
remain available through :mod:`~tensorkrowch.models`.

.. code-block:: python

   import torch
   import tensorkrowch as tk

   f = tk.formats
   cores = [torch.randn(3, 2, dtype=torch.float64),
            torch.randn(2, 3, 2, dtype=torch.float64),
            torch.randn(2, 3, dtype=torch.float64)]
   tt = f.TT(cores)
   samples = torch.tensor([[0, 1, 2], [2, 0, 1]])
   values = tt.evaluate(samples)
   tensor = tt.contract_dense()

If the cores are not yet known, use a :doc:`decomposition <decompositions>`
to compute them from tensor values or a function. Decomposition results
inherit the corresponding numerical format.

Choosing a format
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Representation
     - Open chain
     - Cyclic chain
     - Physical dimensions
   * - Vector or tensor
     - :class:`TT`
     - :class:`TR`
     - One dimension per site.
   * - Matrix or operator
     - :class:`TTM`
     - :class:`TRM`
     - One input/output pair per site.
   * - Quantics vector
     - :class:`QTT`
     - :class:`QTR`
     - Coordinate indices split into digits.
   * - Quantics matrix
     - :class:`QTTM`
     - :class:`QTRM`
     - Separate input and output digit layouts.

:class:`TensorFormat` defines the common numerical interface.
:class:`TensorFormat1D` adds chain operations shared by these formats. The
:doc:`formats_api` contains all classes and their detailed method contracts.

Tensor trains and tensor rings
------------------------------

Tensor trains
^^^^^^^^^^^^^

A TT has open boundaries. An interior core has shape
``(left_rank, input_dimension, right_rank)``. The first core omits its unit
left bond, and the last omits its unit right bond. A one-site TT is a vector.
The neighboring bond dimensions must agree.

.. code-block:: python

   tt = f.TT([
       torch.randn(3, 2, dtype=torch.float64),
       torch.randn(2, 4, 2, dtype=torch.float64),
       torch.randn(2, 5, dtype=torch.float64),
   ])
   dimensions = tt.in_dim
   ranks = tt.rank
   tensor = tt.contract_dense()

This represents a tensor of shape ``(3, 4, 5)`` using internal ranks
``[2, 2]``. ``cores`` contains its stored tensors; ``in_dim`` describes the
physical dimensions and ``n_sites`` counts the sites.

Tensor rings
^^^^^^^^^^^^

A TR retains both virtual axes on every core, including the first and last.
The last core's right bond connects to the first core's left bond. Every
core therefore has shape ``(left_rank, input_dimension, right_rank)``.

.. code-block:: python

   tr = f.TR([
       torch.randn(2, 3, 2, dtype=torch.float64),
       torch.randn(2, 4, 2, dtype=torch.float64),
       torch.randn(2, 5, 2, dtype=torch.float64),
   ])
   tensor = tr.contract_dense()
   open_chain = tr.to_tt()

The dense tensor again has shape ``(3, 4, 5)``. TR ranks include the cyclic
bond. ``to_tt`` carries that closing index through the open chain, preserving
values but potentially increasing TT ranks. Conversely, ``TT.to_tr`` adds a
unit-rank closing bond; finding a compact nontrivial ring is the role of
:class:`~tensorkrowch.decompositions.TT2TR`.

Evaluation and numerical operations
-----------------------------------

Evaluating selected entries
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``evaluate`` accepts integer configurations or physical feature vectors.
Integer inputs select entries; feature vectors contract with each site's
physical axis. A tensor groups inputs on the site axis; a sequence can
provide differently sized feature vectors at different sites.

.. code-block:: python

   tt = f.TT([
       torch.randn(3, 2, dtype=torch.float64),
       torch.randn(2, 3, 2, dtype=torch.float64),
       torch.randn(2, 3, dtype=torch.float64),
   ])
   indices = torch.tensor([[0, 1, 2], [2, 0, 1]])
   features = torch.nn.functional.one_hot(indices, num_classes=3).to(tt.dtype)
   indexed_values = tt.evaluate(indices)
   embedded_values = tt.evaluate(features)
   torch.testing.assert_close(indexed_values, embedded_values)

``zipup=False`` evaluates each core's selected physical axes before contracting
its virtual bond with the accumulated environment. ``zipup=True`` reverses
that order. Both give the same values, up to floating-point precision, with
different computation and temporary-memory costs depending on dimensions.
The option is available for vectors, matrices and the Quantics evaluation
methods.

Use ``in_sites`` to evaluate only some variables or change their input order.
Configuration columns follow the order of that selection. Remaining axes
are returned in their original chain order, after the data and structural
batch axes:

.. code-block:: python

   tt = f.TT([
       torch.randn(3, 2, dtype=torch.float64),
       torch.randn(2, 4, 2, dtype=torch.float64),
       torch.randn(2, 5, dtype=torch.float64),
   ])
   # Columns select site 2 and then site 0; site 1 stays open.
   inputs = torch.tensor([[4, 0], [1, 2]])
   fibers = tt.evaluate(inputs, in_sites=(2, 0))

Here ``fibers`` has shape ``(2, 4)``. A constructor's ``in_sites`` sets the
default selection; an evaluation call can override it. TR uses the same input
conventions. ``contract_dense`` instead materializes every physical axis.

Formats can store leading structural batches through the constructor's
``n_batches`` argument. These are separate from the data batches supplied
when evaluating a format; see the API shapes for their ordering.

Arithmetic and errors
^^^^^^^^^^^^^^^^^^^^^

Arithmetic works directly on cores without constructing the full tensor.
For compatible formats, use ``+`` for addition, ``hadamard`` for entrywise
products, and ``norm`` for the Frobenius norm. ``@`` performs vector/operator
products according to row/column orientation. ``T`` transposes without
conjugation; ``H`` takes the adjoint.

For scalar multiplication, ``format.mul(s)`` and ``format * s`` distribute
:math:`|s|^{1/N}` across the :math:`N` cores. The sign or complex phase is
applied only to the first core, preserving real dtypes for real factors.
Zero is kept in the first core so scalar derivatives remain well-defined.
Multiplying two formats with ``mul`` or ``*`` instead gives their Hadamard
product.

.. code-block:: python

   tt = f.TT([torch.eye(3, dtype=torch.float64),
              torch.eye(3, dtype=torch.float64)])
   summed = tt + tt
   entrywise = tt.hadamard(tt)
   squared_norm = tt.H @ tt
   distance = tt.distance(summed)

These operations can increase ranks. Compression is an explicit subsequent
step. :meth:`TT.error` and its ring counterpart compare a format with a target
function on supplied configurations and return a :class:`SampleError`.
``clone``, ``detach`` and ``to`` provide copying, autograd and device/dtype
operations; their API documents when storage is shared.

Cores, bonds and gauges
-----------------------

Optional ``bonds`` hold diagonal factors between neighboring cores. Pass the
factor sequence to the constructor; the format creates its own
:class:`BondFactors1D` container. ``format.cores`` and ``format.bonds``
together represent the tensor, so stored factors must be included in
contractions.

.. code-block:: python

   cores = [torch.randn(3, 2, dtype=torch.float64),
            torch.randn(2, 4, 2, dtype=torch.float64),
            torch.randn(2, 5, dtype=torch.float64)]
   factors = [torch.tensor([1., 0.5], dtype=torch.float64),
              torch.tensor([1., 0.25], dtype=torch.float64)]
   tt = f.TT(cores, bonds=factors)
   absorbed = tt.clone().materialize_bonds(orth_center=1)
   torch.testing.assert_close(tt.contract_dense(), absorbed.contract_dense())

:meth:`TensorFormat1D.materialize_bonds` absorbs these factors into cores.
It acts in place; clone first to retain the original representation. TR and
matrix formats use the same factor container with their corresponding bond
layout.

Canonical forms
^^^^^^^^^^^^^^^

:meth:`TT.canonicalize <TensorFormat1D.canonicalize>` performs QR/RQ sweeps
without truncation and moves the orthogonality center to the requested site.
On an open chain, the cores to its left and right become isometric in the
corresponding directions.

.. code-block:: python

   tt = f.TT([
       torch.randn(3, 2, dtype=torch.float64),
       torch.randn(2, 4, 2, dtype=torch.float64),
       torch.randn(2, 5, dtype=torch.float64),
   ])
   canonical = tt.clone().canonicalize(orth_center=1)
   vidal = tt.clone().canonicalize_vidal(mode='explicit')
   spectra = vidal.bonds.spectra

For TT and TTM, :meth:`TT.canonicalize_vidal` stores a :class:`VidalGauge`.
Its ``spectra`` retain the Schmidt values and its ``powers`` describe how
those values are distributed between neighboring cores. Modes ``"explicit"``,
``"implicit"``, ``"inverse"``, ``"left"`` and ``"right"`` select common
distributions. The method preserves the tensor and does not truncate ranks.
The API explains ``inverse_positions`` and the cutoff for inverse factors.

For a ring, ``canonicalize`` sets a local gauge relative to the stored cut;
it is not a global Schmidt canonical form. To balance cyclic core gauges,
use :meth:`TR.canonicalize_minimal`:

.. code-block:: python

   tr = f.TR([
       torch.randn(2, 3, 2, dtype=torch.float64),
       torch.randn(2, 4, 2, dtype=torch.float64),
       torch.randn(2, 5, 2, dtype=torch.float64),
   ])
   balanced, info = tr.clone().canonicalize_minimal(
       max_iter=20, return_info=True)

This optimization preserves the tensor and returns optional
:class:`MinimalCanonicalInfo` diagnostics. It need not reach its balance
tolerance within a fixed iteration budget. Open-chain
:meth:`TT.canonicalize_minimal` uses implicit Vidal form without iterative
optimization.

For custom gauge analysis, :class:`TensorRingOrbit` exposes the ring's gauge
objective and bond Gram matrices; :class:`GaugeOrbit` provides the general
core-and-bond interface. Ordinary canonicalization does not require creating
these objects yourself.

Compression and blocking
------------------------

:meth:`TT.rounding <TensorFormat1D.rounding>` and
:meth:`TR.rounding <TensorFormat1D.rounding>` compress bond dimensions in
place. Use ``rank`` for a shared rank cap, or singular-value truncation
criteria such as ``rtol``. ``rtol`` limits relative discarded squared mass
locally; ``rel_error`` instead requests a global relative Frobenius error
budget.

.. code-block:: python

   tt = f.TT([
       torch.randn(3, 2, dtype=torch.float64),
       torch.randn(2, 4, 2, dtype=torch.float64),
       torch.randn(2, 5, dtype=torch.float64),
   ])
   rounded, info = tt.clone().rounding(rank=1, return_info=True)
   measured_error = tt.distance(rounded) / tt.norm()
   error_bound = info.error_bound

:class:`RoundingInfo` records the retained ranks and truncation information.
Its error bound is distinct from the measured reconstruction error. Stricter
rank caps or other simultaneous criteria may prevent the requested error
budget from being met. Ring rounding includes the cyclic bond but does not
guarantee minimal TR ranks.

To solve problems on larger local blocks, :meth:`TensorFormat1D.block`
contracts consecutive groups into effective sites. It returns a
:class:`BlockLayout` describing how to restore the original dimensions:

.. code-block:: python

   tt = f.TT([
       torch.randn(3, 2, dtype=torch.float64),
       torch.randn(2, 4, 2, dtype=torch.float64),
       torch.randn(2, 5, dtype=torch.float64),
   ])
   blocked = tt.clone()
   layout = blocked.block([2, 1])
   restored = blocked.unblock(layout)
   torch.testing.assert_close(restored.contract_dense(), tt.contract_dense())

Both operations modify the format. :func:`split_block` provides the local
splitting operation directly and returns a :class:`SplitBlock`. For Quantics,
convert to an ordinary format before grouping arbitrary digit sites, so the
coordinate layout is not implicitly changed.

Matrix formats
--------------

TTM and TRM attach an input and output physical axis to every site. Their
cores use the order ``(left_rank, input_dimension, right_rank,
output_dimension)``, omitting the first and last unit bonds for TTM. For
example, a product identity operator on dimensions ``(2, 3)`` is:

.. code-block:: python

   matrix = f.TTM([
       torch.eye(2, dtype=torch.float64).reshape(2, 1, 2),
       torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
   ])
   vector = f.TT([torch.randn(2, 2, dtype=torch.float64),
                  torch.randn(2, 3, dtype=torch.float64)])
   applied = matrix.apply(vector)
   product = matrix @ vector
   torch.testing.assert_close(
       applied.contract_dense(), vector.contract_dense())
   torch.testing.assert_close(
       product.contract_dense(), vector.contract_dense())

For a TRM, keep both bond axes on every core and match the closing bond.
``in_dim`` and ``out_dim`` describe the input/output dimensions. Matrix
``contract_dense`` returns interleaved physical axes
``(in_0, out_0, in_1, out_1, ...)``.

:meth:`TTM.evaluate` and :meth:`TRM.evaluate` accept separate ``in_data`` and
``out_data``. Their columns follow ``in_sites`` and ``out_sites``; unselected
axes are returned in chain order, with input-side axes before output-side
axes. Both selections default to all sites. ``apply`` contracts every input
physical dimension and returns an output format, rather than selected matrix
entries. Transpose and adjoint retain the concrete matrix format.

Quantics formats
----------------

Quantics splits each original coordinate index into several digit sites.
QTT and QTR use the TT/TR core layouts, with digit dimensions as their
physical dimensions. Construct them from cores and a grid specification:

.. code-block:: python

   cores = [torch.randn(2, 2, dtype=torch.float64),
            torch.randn(2, 2, 2, dtype=torch.float64),
            torch.randn(2, 2, dtype=torch.float64)]
   qtt = f.QTT(cores, n_coordinates=1, base=2, level=3,
               domain=torch.tensor([0., 1.], dtype=torch.float64))
   coordinates = torch.tensor([[0.], [0.25], [0.75]], dtype=torch.float64)
   values = qtt.evaluate_coordinates(coordinates)
   grid_tensor = qtt.contract_dense_grid_axes()

The three binary sites represent eight grid points for one coordinate.
``contract_dense`` retains one axis per digit site, while
``contract_dense_grid_axes`` combines them into one axis per coordinate.
``evaluate_coordinates`` quantizes using the coordinate map; it does not
interpolate between grid values.

A QTR uses three-axis cores including the cyclic bond. QTTM and QTRM use
matrix cores and separate input/output coordinate descriptions through their
``in_*`` and ``out_*`` arguments. The two sides can have different numbers
of coordinates and digit sites, selected by ``in_sites`` and ``out_sites``.

Layouts and coordinate maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The convenience arguments create these objects automatically:

* ``base``, ``level`` and ``domain`` create a :class:`QuantizedLayout` and
  an :class:`AffineCoordinateMap`. Defaults use interleaved digit sites,
  coarse-to-fine digit order and a left-offset grid.
* ``grid_coordinates`` replaces ``domain`` with an :class:`ExplicitGridMap`.
  Each coordinate grid must contain ``base ** level`` points.
* Explicit ``layout`` and ``coordinate_map`` objects allow other schedules
  and coordinate transformations.

The available coordinate objects are:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Object
     - Role
   * - :class:`QuantizedLayout`
     - Convert integer grid indices to digits and choose their site order.
   * - :class:`AffineCoordinateMap`
     - Map coordinates to an affine grid over a domain.
   * - :class:`ExplicitGridMap`
     - Map coordinates to explicitly supplied grid points.
   * - :class:`FunctionalCoordinateMap`
     - Apply user-defined forward and optional inverse coordinate functions.
   * - :class:`CoordinateMap`
     - Common interface for coordinate/grid-index conversion.

For example, two coordinates can use different digit counts:

.. code-block:: python

   layout = f.QuantizedLayout(
       n_coordinates=2, base=2, level=(2, 1), ordering='grouped')
   coordinate_map = f.AffineCoordinateMap(
       domain=torch.tensor([[0., 1.], [-1., 1.]], dtype=torch.float64),
       grid_size=layout.grid_size)
   cores = [torch.randn(2, 2, dtype=torch.float64),
            torch.randn(2, 2, 2, dtype=torch.float64),
            torch.randn(2, 2, dtype=torch.float64)]
   qtt = f.QTT(cores, n_coordinates=2, layout=layout,
               coordinate_map=coordinate_map)
   coordinates = torch.tensor([[0., -1.], [0.5, 0.]], dtype=torch.float64)
   indices = coordinate_map.to_indices(coordinates)
   digits = layout.encode_indices(indices)
   values = qtt.evaluate_coordinates(coordinates)
   torch.testing.assert_close(values, qtt.evaluate_indices(indices))
   torch.testing.assert_close(values, qtt.evaluate_digits(digits))

The evaluation path is coordinate map, integer grid indices, layout encoding,
then digit-core contraction. ``evaluate_indices`` starts at grid indices;
``evaluate_digits`` starts at digits. The map's ``grid_size`` must agree with
the layout. ``QuantizedLayout.decode_digits`` and
``CoordinateMap.from_indices`` provide the reverse conversions.

For matrix variants, pass ``in_layout``, ``out_layout``,
``in_coordinate_map`` and ``out_coordinate_map`` independently. Their API
explains the coordinate and digit shapes on each side.

Models and decomposition results
--------------------------------

Use :meth:`TT.to_mps` / :meth:`TR.to_mps` to create graph-based MPS models,
and :meth:`TTM.to_mpo` / :meth:`TRM.to_mpo` for MPO models. Open and cyclic
formats retain their corresponding boundary conditions. ``parameterized``
controls whether the new model uses trainable nodes.

:meth:`TT.from_mps` and :meth:`TTM.from_mpo`, and their cyclic counterparts,
convert compatible models back to formats. Conversions handle stored bond
factors; callers do not need to absorb them manually first.

:doc:`Decompositions <decompositions>` return numerical formats enriched with
fit metrics and metadata. Use ``result.to_format()`` to keep the numerical
representation alone. Source adapters provide the opposite direction:
:class:`~tensorkrowch.decompositions.FormatSource` lets algorithms evaluate an
existing format without turning it into a graph model.

API reference
-------------

.. toctree::
   :maxdepth: 2

   formats_api
