Formats
=======

.. currentmodule:: tensorkrowch.formats

Compact numerical representations
---------------------------------

Formats are lightweight representations of data, operators and functions,
constructed directly from PyTorch tensors. They do not use TensorKrowch Nodes,
Edges or the graph operation system. They support decomposition algorithms,
linear solvers and related numerical methods whose state is a fixed network.
Models provide parameterized networks and optimize repeated graph calculations
when learning functions from data. Explicit adapters connect both layers.
Formats preserve PyTorch autograd where their operations support it; they do
not detach inputs or manage training implicitly.

Implementation and review order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implementation follows a one-way dependency flow:

.. code-block:: text

   base.py
      |
      v
   bonds.py    orbits.py
         \      /
          v    v
           formats1d.py             quantization.py
                |                        |
                +----------+-------------+
                           v
                       quantics.py
                           |
                           v
                        tucker.py

``formats1d.py`` contains the shared container, vector/matrix and open/cyclic
bases, TT/TR/TTM/TRM, numerical operations and model adapters. Review it from
top to bottom: numerical helpers, core and bond management, conversions,
canonicalization, blocks and algebra. The vector and matrix bases then provide
their evaluation and adapters; the open and cyclic bases provide their gauges
and topology operations. Concrete classes define core layouts and validation.
The helpers restore tensor shapes or run reused numerical phases; they do not
import concrete format classes from other modules.

In-place algorithms compute local core/factor lists and install them together
through ``_set_standard_cores``, validating the final state once. Operations
that produce a separate format use ``_new_from_standard_cores``: ordinary
formats construct TT/TR/TTM/TRM directly, and Quantics overrides that step to
preserve coordinate metadata without an intermediate ordinary format.
Model imports occur only in the adapters. Decomposition results inherit the
formats and attach metrics and provenance; formats do not import decompositions.

Controlled mutations and tensor ownership
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The constructor copies the core container and shares tensor storage. Assigning
``format.cores[i]`` or a same-length slice immediately validates ranks,
dimensions and runtime and refreshes cached metadata. Invalid assignments
restore the previous state; queries and contractions do not revalidate it.
Changing the number or order of sites requires the full ``cores`` setter or
an explicit topology operation. ``rank`` returns a defensive list. Tensor ``resize_`` is
outside this contract; replace the tensor instead.

``cores``, bond factors, Vidal spectra and absorption powers share the private
``_SafeList`` implementation in ``base.py``. It preserves list identity on
element/slice replacement and restores entries when a validation callback
raises. The format callback restores core metadata on failure and clears
canonical state only after a successful edit.

``BondFactors1D`` stores factors for open or cyclic chains. Bond containers
are constructed by their owning format, already bound to its callback.
Pass a sequence of factors as ``TT(cores, bonds=factors)`` or assign
``format.bonds = factors``; the format copies the sequence and shares its
tensors. Element and same-length slice replacements in
``format.bonds.factors`` validate immediately against the current cores.
The container notifies its format through a callback; copies and conversions
construct their containers with the destination callback. There is no separate
attachment step. ``split_block`` returns raw factor tuples, which can be passed
to ``replace_cores`` together with the replacement cores.
Replace adjacent cores together when changing a shared rank. Algorithms work
with temporary lists and publish cores and bonds together at completion.
Manual replacement of cores, factors, Vidal spectra or absorption powers
invalidates canonical state. An invalidated Vidal gauge remains usable as
diagonal factors; materialization and absorption use its stored factors,
while spectrum redistribution requires a valid gauge. Recomputing
``canonicalize_vidal`` restores valid Schmidt spectra. Tensor value changes
are not intercepted; shape changes should use controlled replacement.

.. doctest::

   >>> format = tk.formats.TT([torch.ones(2, 2), torch.ones(2, 2)])
   >>> format.cores[:] = [torch.ones(2, 3), torch.ones(3, 2)]
   >>> format.rank
   [3]
   >>> format.bonds = [torch.ones(3)]
   >>> format.bonds.factors[0] = torch.full((3,), 2.)
   >>> torch.equal(format.contract_dense(), torch.full((2, 2), 6.))
   True
   >>> format.cores[0] = torch.ones(2, 4)
   Traceback (most recent call last):
       ...
   ValueError: Adjacent TT ranks should match
   >>> format.rank
   [3]

``clone()`` clones structural tensors while retaining autograd. ``detach()``
creates a separate container sharing storage, and ``detach_()`` replaces all
structural references in the current container. ``to`` preserves the concrete
class and returns self when no conversion is needed. CUDA/MPS operations use
PyTorch on the requested device and propagate unsupported-operation errors.
Coordinate grids and Schmidt spectra remain real when cores are complex.

Layouts and evaluation
----------------------

TT cores use ``(in, right)``, ``(left, in, right)``, ``(left, in)`` at the
first, interior and final sites. A one-site TT is ``(in,)``. TR cores all use
``(left, in, right)``, with the last right rank equal to the first left rank.
Leading structural batch axes precede these dimensions.

The public ``cores`` retain these shapes. Internally, ``_standard_cores()``
adds unit virtual axes to open chains and combines matrix input/output into a
single physical axis, leaving explicit bond factors separate.
``_effective_cores()`` has the same shapes with those factors absorbed.
``_operator_cores()`` also absorbs factors, but keeps input and output separate
as ``(*batch, left, input, right, output)``. A ket has input dimension one;
a vector row has output dimension one. These temporary views leave the stored
cores unchanged.

TTM cores use ``(in, right, out)``, ``(left, in, right, out)``, and
``(left, in, out)``; a one-site TTM is ``(in, out)``. TRM cores all use
``(left, in, right, out)``. The initial TTM contract rejects structural
batches; TRM supports them, but TensorKrowch has no batched MPO counterpart.
Consequently, converting a batched TRM to TTM/MPO is explicitly unsupported.

``contract_dense()`` is an explicit small-network oracle. Matrix axes remain
interleaved: ``(in_0, out_0, in_1, out_1, ...)``. Matrix input indices represent
columns and output indices represent rows of the conventional dense operator.

Vector ``evaluate`` accepts integer configurations ``(*data_batch, n_sites)``
or local embedded vectors ``(*data_batch, n_sites, in_dim)`` when dimensions
are uniform. A sequence supports heterogeneous local dimensions. Matrix
``evaluate(in_data, out_data)`` contracts paired configurations; their data
batch shapes must match. Matrix ``apply(data)`` contracts local product inputs
and returns a vector format. It does not implicitly factor a global dense
vector into a TT.

Core and data batches are independent, with output axes
``(*core_batch, *data_batch, ...)``, even if their sizes coincide. Binary
format operations require matching structural batches or one unbatched
operand. They do not silently form a Cartesian product of structural batches.
Norm and overlap use scaled contractions. ``inner`` conjugates its first
operand; normalized overlap retains its complex phase, and fidelity is its
squared magnitude. Normalized overlap with a zero-norm format raises an error.

Exact algebra
-------------

``+`` and ``-`` form exact sums/differences; ``*`` is Hadamard multiplication
between compatible formats, or scalar scaling. Vectors represent kets:
``A @ x`` applies an operator, ``x.T @ A`` applies a vector row from the left,
and ``A @ B`` composes operators. Local inputs of the left operand contract
with local outputs of the right operand. ``x.T`` and ``x.H`` retain the vector's
format class and full API, with independent core and bond containers sharing
tensor storage. ``T`` changes row/column orientation; ``H`` also conjugates the
coefficients. Stored core shapes and coefficient evaluation are unchanged.
Replacing a core or bond in the transposed format does not replace it in the
original. Thus ``x.T @ y`` is bilinear,
``x.H @ y`` is Hermitian, and ``x @ y.H`` forms an outer-product matrix.
``x @ y`` and ``x @ A`` are undefined for kets. The existing ``x.apply(A)``
returns the ket coefficients of ``(x.T @ A).T``.
A cyclic operand produces a cyclic result, including a closing rank of one.
Operations do not densify, mutate operands or perform hidden rounding.

.. doctest::

   >>> x = tk.formats.TT([torch.tensor([1., 2.])])
   >>> a = tk.formats.TTM([torch.diag(torch.tensor([1., 2.]))])
   >>> energy = (x.H @ a @ x) / (x.H @ x)
   >>> torch.allclose(energy, torch.tensor(1.8))
   True

TR/TRM sums default to stacked endpoints from Mickelin and Karaman,
`Section 3.2 <https://arxiv.org/pdf/1807.02513>`_. This construction facilitates
compression of sums. ``a.add(b, method='block_diagonal')`` and
``a.sub(b, method='block_diagonal')`` expose the usual fully block-diagonal
construction, whose redundancies can survive TR rounding.

``A.T``/``transpose()`` swap local input/output axes without reversing sites.
``A.H``/``adjoint()`` additionally conjugate cores and factors. ``conj`` also
works on vectors. ``trace`` requires equal input/output dimensions at every
site; a globally square but locally incompatible tensorization is rejected.
Views may share storage; use ``clone()`` for independent tensors.

Gauges and compression
----------------------

``canonicalize(orth_center=None, renormalize=False)`` performs only QR/RQ sweeps.
The default center is the last site. It preserves the represented tensor and
global scale. Open chains have left/right isometries around the center;
cyclic chains obtain a local gauge relative to the stored cut, with no claim
of a global Schmidt form. ``renormalize`` controls intermediate scales rather
than normalizing the state to unit norm.

``canonicalize_vidal`` operates on TT/TTM themselves, storing a ``VidalGauge``
alongside their cores. Explicit, implicit and inverse modes distribute
``Gamma -- Lambda -- Gamma``, square roots of Lambda on both neighbours, or
``Gamma Lambda -- Lambda^-1 -- Lambda Gamma``. ``inverse_positions`` selects
only block-interface inverses, with explicit or implicit bonds elsewhere.
Inverse mode rejects zero Schmidt values or values at/below
``inverse_cutoff``; it does not silently truncate support.

``materialize_bonds(orth_center=...)`` redistributes the current absorption, including
already absorbed roots and inverse factors. Explicit Lambda to the left of
the center goes to its right neighbour; to the right it goes to its left
neighbour. Inverse factors go to the opposite neighbour. A mixed-canonical
interpretation requires a still-valid Vidal gauge; algebraic reconstruction
is preserved independently. Replacing cores invalidates that certificate.
``absorb_bond`` permits a local redistribution without a new global sweep.

For improved small-value accuracy, independently of the base SVD backend:

.. code-block:: python

   import torch
   import tensorkrowch as tk
   tt = tk.formats.TT([
       torch.randn(2, 2, dtype=torch.float64),
       torch.randn(2, 2, 2, dtype=torch.float64),
       torch.randn(2, 2, 2, dtype=torch.float64),
       torch.randn(2, 2, dtype=torch.float64)])
   with tk.svd_method('qr_svd', refine=True):
       tt.canonicalize_vidal(inverse_positions=[2], remaining_mode='implicit')
   tt.materialize_bonds(orth_center=3)

The refinement follows the Appendix of
`Stoudenmire and White <https://arxiv.org/pdf/1301.3494>`_ and cannot recover
information already lost in the input precision.

``rounding`` performs a left QR sweep followed by right, sitewise truncated
SVDs on TT/TTM, ending at ``orth_center=0``. TR/TRM use one execution of
`Mickelin--Karaman Algorithm 4 <https://arxiv.org/pdf/1807.02513>`_, including
the cyclic closure reduction. The one-site ring case reduces its trace exactly.
TR rounding does not guarantee minimal ranks, especially after products or
fully block-diagonal sums. There is no ``n_sweeps`` parameter.

Local ``rank``, ``cutoff``, ``atol``, ``rtol`` and ``cum_percentage`` retain
``utils.truncated_svd`` semantics; ``atol`` and ``rtol`` concern squared tail
energy. The separate ``rel_error`` sets a budget for the global Frobenius
reconstruction error of the full tensor or matrix:
``||X - X_round||_F <= rel_error * ||X||_F``, where ``X`` is the original.
For example, ``rel_error=0.03`` requests an error of at most 3% of its norm,
separately for each structural batch. Rounding distributes this budget over
local cuts; ``rtol`` instead applies to the squared singular-value mass at each
local SVD. Hard rank caps or more restrictive criteria can exceed the global
budget. In that case, rounding warns and optional
``RoundingInfo.bound_satisfied`` is False.
``error_bound`` is a bound, not a measured dense error. No diagnostic history
is stored when it is not requested. Squared-energy records may underflow or
overflow at extreme scales even when the norm-error bound remains representable.

``canonicalize_minimal`` uses implicit Vidal on open chains. On finite rings
it experimentally balances half the sum of squared core norms using real or
complex Hermitian exponential gauges, preserving the best finite iterate.
This finite, nonuniform-ring objective is inspired by
`Acuaviva et al. <https://arxiv.org/pdf/2209.14358>`_; it does not claim the
uniform-network theorems or uniqueness, and finite iteration may not converge.
``return_info=True`` returns ``MinimalCanonicalInfo`` with the iteration count,
convergence flag and final virtual-bond Gram imbalance. The default stores no
diagnostic history.
Batched rings share gauges minimizing their aggregate objective. ``GaugeOrbit``
defines cancellation by virtual tensor axes without assuming a geometry;
no PEPS orbit or PEPS implementation is provided yet.

Blocks and topology conversions
-------------------------------

``block(groups)`` contracts contiguous sites in-place and returns a
``BlockLayout`` describing the original dimensions. ``unblock(layout, ...)``
restores those sites in-place, optionally truncating ranks inside each group.
The layout can also be applied to a solver result with matching blocked
dimensions; it need not belong to the same object. Matrices retain separate
grouped input/output dimensions. Clone a format before blocking to retain its
original structure. Quantics layouts must remain compatible with the cores;
use an explicit ``as_tt``/``as_tr``/``as_ttm``/``as_trm`` conversion to group
arbitrary digit sites without coordinate metadata.
``contract_block(first, last)`` leaves external ranks open and includes internal
factors only. ``split_block`` returns standard fused local cores, cut spectra
and internal factors; these spectra are local factorization values rather
than certified global Schmidt spectra. ``replace_cores(first, cores, bonds=...)``
installs consecutive standard cores and internal factors together, preserving
external ranks and factors. This also supports regional algorithms that update
cores without contracting their whole region. Solver caches and messages remain
the responsibility of the algorithm. Crossing a stored cyclic
cut is expressed explicitly by rotating the ring first.

.. doctest::

   >>> format = tk.formats.TT([torch.eye(2), torch.eye(2)])
   >>> layout = format.block([2])
   >>> solution = tk.formats.TT([2 * format.cores[0]])
   >>> _ = solution.unblock(layout)
   >>> torch.allclose(solution.contract_dense(), 2 * torch.eye(2))
   True

``tr.rotate(first=k)`` preserves circular order and rotates the dense axes.
``tr.to_tt()`` carries the closing index through identities in every interior
core. Internal ranks become ``closing_rank * original_rank``. It is exact,
with no dense reconstruction, truncation or SVD. Matrices use ``to_ttm``.

Quantics and Tucker
--------------------

Quantics subclasses add ``QuantizedLayout`` and coordinate maps to
the ordinary numerical formats. Layouts support grouped, interleaved and
custom schedules, heterogeneous bases/levels and both digit directions.
``QTT`` and ``QTR`` can construct these objects from ``n_coordinates``,
``base``, ``level`` and ``domain``. This constructs an interleaved,
coarse-to-fine layout and an ``AffineCoordinateMap`` with ``grid_offset='left'``.
Passing ``grid_coordinates`` instead of ``domain`` constructs an
``ExplicitGridMap``; both ``base`` and ``level`` are required, and
``base ** level`` must match the grid sizes exactly. The matrix formats accept
independent ``in_*`` and ``out_*`` arguments; all four formats also accept
prebuilt layouts and coordinate maps.
``evaluate_digits``, ``evaluate_indices`` and ``evaluate_coordinates`` distinguish
site digits, original grid indices and coordinates in the domain.
``digit_positions`` allows vector formats to retain tensor-valued output
sites. Maps store their domain, grid sizes and out-of-domain policy.
``AffineCoordinateMap`` and ``FunctionalCoordinateMap`` both use a uniform
grid in unit space; the latter applies supplied transformation functions.
``to_dense_grid`` is an explicit small-grid oracle.

Matrix Quantics has separate input/output layouts and maps with paired digit
schedules. ``T`` and ``H`` swap their meaning. Algebra preserves Quantics
classes when coordinate meanings match; incompatible layouts are rejected
even if core sizes coincide. ``as_tt``/``as_tr``/matrix equivalents deliberately
drop coordinate semantics. Reordering configurations does not reorder fitted
cores: grouped-to-interleaved conversion would require swaps and SVDs.

``QTTTucker``/``QTRTucker`` compose an upper TT/TR with one local TT per
variable, whose final site is the connector gamma. Upper cores have one
owner; ``cores`` delegates to ``upper.cores``. ``coordinate_positions`` locates
upper connectors among optional output sites. ``flatten`` returns a Quantics
network with the actual grouped factor schedule. Fitters and sketch recursion
remain in decompositions.

Models and numerical workflows
-------------------------------

``tt.to_mps()``/``tr.to_mps()`` build MPS, or MPSData for structural batches.
``ttm.to_mpo()``/``trm.to_mpo()`` build MPO. Default adapters use
``parameterized=False``; callers can request a parameterized model explicitly.
Adapters collect effective tensors and materialize factors temporarily, without
mutating or detaching the source format. Reverse ``from_mps``/``from_mpo``
adapters require the corresponding boundary type. Models expose matching
``to_tt``/``from_tt`` and cyclic/matrix facades. Models do not store Quantics
coordinate meaning; reverse adapters return the ordinary numerical formats.

.. code-block:: python

   import torch
   import tensorkrowch as tk

   a = tk.formats.TTM([
       torch.eye(2).reshape(2, 1, 2),
       torch.eye(2).reshape(1, 2, 2)])
   x = tk.formats.TT([torch.ones(2, 1), torch.ones(1, 2)])
   residual = a @ x - x
   assert residual.norm() == 0
   x.rounding(rank=1)
   mps = x.to_mps()
   restored = mps.to_tt()
   assert torch.allclose(restored.contract_dense(), x.contract_dense())

Public API
----------

.. automodule:: tensorkrowch.formats
   :members:
   :imported-members:
