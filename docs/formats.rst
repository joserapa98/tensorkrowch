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

The constructor copies the core container and shares tensor storage. Assigning
``network.cores[i]`` or a same-length slice invalidates structural metadata;
the next public access validates ranks, dimensions and runtime. Changing the
number or order of sites requires the full ``cores`` setter or an explicit
topology operation. ``rank`` returns a defensive list. Tensor ``resize_`` is
outside this contract; replace the tensor instead.

``copy()`` clones structural tensors while retaining autograd. ``detach()``
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
network operations require matching structural batches or one unbatched
operand. They do not silently form a Cartesian product of network batches.
Norm and overlap use scaled contractions. ``inner`` conjugates its first
operand; normalized overlap retains its complex phase, and fidelity is its
squared magnitude. Normalized overlap with a zero-norm network raises an error.

Exact algebra
-------------

``+`` and ``-`` form exact sums/differences; ``*`` is Hadamard multiplication
between compatible formats, or scalar scaling. ``@`` and ``apply`` implement
matrix-vector, vector-matrix and matrix-matrix products. ``A @ x`` contracts
``A.in_dim`` and ``x.in_dim``; ``x @ A`` contracts ``x.in_dim`` and
``A.out_dim``. ``A @ B`` contracts ``A.in_dim`` with ``B.out_dim``. A cyclic
operand produces a cyclic result, including a closing rank of one. Operations
do not densify, mutate operands or perform hidden rounding.

TR/TRM sums default to stacked endpoints from Mickelin and Karaman,
`Section 3.2 <https://arxiv.org/pdf/1807.02513>`_. This construction facilitates
compression of sums. ``a.add(b, method='block_diagonal')`` and
``a.sub(b, method='block_diagonal')`` expose the usual fully block-diagonal
construction, whose redundancies can survive TR rounding.

``A.T``/``transpose()`` swap local input/output axes without reversing sites.
``A.H``/``adjoint()`` additionally conjugate cores and factors. ``conj`` also
works on vectors. ``trace`` requires equal input/output dimensions at every
site; a globally square but locally incompatible tensorization is rejected.
Views may share storage; use ``copy()`` for independent tensors.

Gauges and compression
----------------------

``canonicalize(oc=None, renormalize=False)`` performs only QR/RQ sweeps.
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

``materialize_bonds(oc=...)`` redistributes the current absorption, including
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
   tt.materialize_bonds(oc=3)

The refinement follows the Appendix of
`Stoudenmire and White <https://arxiv.org/pdf/1301.3494>`_ and cannot recover
information already lost in the input precision.

``rounding`` performs a left QR sweep followed by right, sitewise truncated
SVDs on TT/TTM, ending at ``oc=0``. TR/TRM use one execution of
`Mickelin--Karaman Algorithm 4 <https://arxiv.org/pdf/1807.02513>`_, including
the cyclic closure reduction. The one-site ring case reduces its trace exactly.
TR rounding does not guarantee minimal ranks, especially after products or
fully block-diagonal sums. There is no ``n_sweeps`` parameter.

Local ``rank``, ``cutoff``, ``atol``, ``rtol`` and ``cum_percentage`` retain
``utils.truncated_svd`` semantics; ``atol`` and ``rtol`` concern squared tail
energy. The separate ``rel_error`` requests a global relative norm-error
budget. Hard rank caps or more restrictive criteria can exceed it, in which
case rounding warns and optional ``RoundingInfo.bound_satisfied`` is False.
``error_bound`` is a bound, not a measured dense error. No diagnostic history
is stored when it is not requested. Squared-energy records may underflow or
overflow at extreme scales even when the norm-error bound remains representable.

``canonicalize_minimal`` uses implicit Vidal on open chains. On finite rings
it experimentally balances half the sum of squared core norms using real or
complex Hermitian exponential gauges, preserving the best finite iterate.
This finite, nonuniform-ring objective is inspired by
`Acuaviva et al. <https://arxiv.org/pdf/2209.14358>`_; it does not claim the
uniform-network theorems or uniqueness, and finite iteration may not converge.
Batched rings share gauges minimizing their aggregate objective. ``GaugeOrbit``
defines cancellation by virtual tensor axes without assuming a geometry;
no PEPS orbit or PEPS implementation is provided yet.

Blocks and topology conversions
-------------------------------

``block(groups)`` contracts contiguous sites and stores ``BlockLayout`` for
``unblock``. Matrices retain separate grouped input/output dimensions.
``contract_block(first, last)`` leaves external ranks open and includes internal
factors only. ``split_block`` returns standard fused local cores, cut spectra
and internal factors; these spectra are local factorization values rather
than certified global Schmidt spectra. ``replace_block`` installs a compatible
replacement atomically, preserving external ranks. Crossing a stored cyclic
cut is expressed explicitly by rotating the ring first.

``tr.rotate(first=k)`` preserves circular order and rotates the dense axes.
``tr.to_tt()`` carries the closing index through identities in every interior
core. Internal ranks become ``closing_rank * original_rank``. It is exact,
with no dense reconstruction, truncation or SVD. Matrices use ``to_ttm``.

Quantics and Tucker
--------------------

Quantics subclasses add ``QuantizedLayout`` and optional coordinate maps to
the ordinary numerical formats. Layouts support grouped, interleaved and
custom schedules, heterogeneous bases/levels and both digit directions.
``evaluate_digits``, ``evaluate_indices`` and ``evaluate_points`` distinguish
network digits, original grid indices and physical coordinates.
``digit_positions`` allows vector formats to retain tensor-valued output
sites. Physical evaluation requires an actual map and its inverse; a stored
map name is insufficient. ``to_dense_grid`` is an explicit small-grid oracle.

Matrix Quantics has separate input/output layouts and maps with paired digit
schedules. ``T`` and ``H`` swap their meaning. Algebra preserves Quantics
classes when coordinate meanings match; incompatible layouts are rejected
even if core sizes coincide. ``as_tt``/``as_tr``/matrix equivalents deliberately
drop coordinate semantics. Reordering configurations does not reorder fitted
cores: grouped-to-interleaved conversion would require swaps and SVDs.

``QTTTucker``/``QTRTucker`` compose an upper TT/TR with one local TT per
variable, whose final site is the connector gamma. Upper cores have one
owner; ``cores`` delegates to ``upper.cores``. ``variable_positions`` locates
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
