Decompositions
==============

.. currentmodule:: tensorkrowch.decompositions

Decomposition algorithms approximate tensors and functions as tensor trains
(TT), tensor rings (TR), or their matrix and Quantics extensions. The usual
workflow is to describe the input with a **source**, construct a decomposition
**engine**, and call its **fit** method. Optional strategy objects configure
steps such as sampling, local optimization and convergence.

.. code-block:: python

   import torch
   import tensorkrowch as tk

   d = tk.decompositions
   source = d.DenseTensorSource(torch.randn(4, 5, 6, 4))
   engine = d.TTSVD(source)
   result = engine.fit(rank=4)
   approximation = result.contract_dense()

Results are numerical :doc:`formats` with additional fit diagnostics. A TT
has open boundaries; a TR connects the first and last core with a cyclic
bond. Both are supported throughout the guide.

Choosing a method
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 25 25

   * - Approach
     - Input and intended use
     - Tensor train
     - Tensor ring
   * - SVD
     - A finite tensor that fits in memory; successive matrix factorizations.
     - :class:`TTSVD`
     - :class:`TRSVD`
   * - Alternating least squares
     - Fit chosen ranks by optimizing cores, using all entries or sampled rows.
     - :class:`TTALS`
     - :class:`TRALS`
   * - Recursive sketching from samples
     - An evaluable source and configurations defining the region of interest.
     - :class:`TTRSS`
     - :class:`TRRSS`
   * - Recursive sketching with operators
     - A finite source and selected fibers, marginalizations or TT sketches.
     - :class:`TTRS`
     - Apply :class:`TT2TR` to the resulting TT if needed.
   * - Conversion and spectral recovery
     - Convert an existing TT, or recover an exact TR by spectral methods.
     - The TT is the input to conversion.
     - :class:`TT2TR`, :func:`tr_blostr`, :func:`tr_blostr_svd`

The examples use engine objects. Functions such as :func:`tt_svd`,
:func:`tr_als` and :func:`tt_rss` provide shorter interfaces for a single fit.
The :doc:`decompositions_api` lists their complete signatures and all
configurable components.

Describing the input
--------------------

A source separates how values are obtained from how they are decomposed.
Choose one of the following :class:`BaseSource` implementations:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Source
     - Use it for
   * - :class:`DenseTensorSource`
     - A PyTorch tensor with one axis per variable.
   * - :class:`SparseTensorSource`
     - Indexed entries of a finite tensor; unstored entries are zero.
   * - :class:`FormatSource`
     - An existing tensor format, TT cores, or an MPS/MPO model.
   * - :class:`CallableTensorSource`
     - A function of finite discrete indices, with explicit input dimensions.
   * - :class:`CallableSource`
     - A function of continuous inputs, for sampled sketching with embeddings.
   * - :class:`CallableQuanticsSource`
     - A coordinate function discretized into digit sites on a Quantics grid.

For example, a discrete callable evaluates a batch of index configurations:

.. code-block:: python

   def function(indices):
       return (indices.to(torch.float64) + 1).prod(dim=1)

   source = d.CallableTensorSource(function, in_dim=(4, 4, 4, 4))
   engine = d.TTSVD(source)
   result = engine.fit(rank=2)

Sources expose :meth:`BaseSource.evaluate` for configurations,
:meth:`BaseSource.fiber` to vary one input, and
:meth:`BaseSource.contract_dense` to materialize a finite tensor. Configuration
columns follow ``in_sites``; :class:`ConfigurationBatch` can explicitly group
index or feature inputs. Vector sources use ``in_sites`` to select evaluated
variables; unselected axes remain open in the output.

Dense tensors can also be passed directly to engines that accept them.
:func:`as_tensor_source` provides the corresponding adaptation explicitly.
SVD materializes even sparse or callable sources; ALS and sampled sketching
can query their entries without constructing the complete tensor.

Dataset constructors
^^^^^^^^^^^^^^^^^^^^

Use ``SparseTensorSource.from_data`` for discrete empirical masses.
``CallableSource.from_data`` describes continuous data, optionally smoothed
with a kernel. ``CallableTensorSource.from_data`` and
``CallableQuanticsSource.from_data`` provide the corresponding finite and
quantized constructions. For example:

.. code-block:: python

   dataset = torch.rand(64, 3)
   source = d.CallableSource.from_data(
       dataset, kernel=d.gaussian_kernel, bandwidth=0.2)

:func:`gaussian_kernel` also works independently on query coordinates and
kernel centers. See each ``from_data`` method for weights and domain options.

SVD decompositions
------------------

SVD methods factor a finite tensor into smaller cores. They provide direct
decompositions without iterative core optimization. The engine materializes
the source at construction; repeated fits reuse that prepared tensor.

TT-SVD
^^^^^^

:class:`TTSVD` applies successive truncated SVDs along an open chain, following
Oseledets' tensor train construction [ttsvd]_.

.. code-block:: python

   axis = torch.arange(1, 5, dtype=torch.float64)
   tensor = torch.einsum('i,j,k,l->ijkl', axis, axis, axis, axis) + 1
   source = d.DenseTensorSource(tensor)
   engine = d.TTSVD(source)
   result = engine.fit(rank=2, collect_metrics=True)
   approximation = result.contract_dense()

``rank`` is an upper bound at each cut. ``cutoff``, ``atol``, ``rtol`` and
``cum_percentage`` select singular values according to
:func:`~tensorkrowch.utils.truncated_svd`. In particular, SVD ``rtol`` limits
the discarded **squared** singular-value mass relative to the total squared
mass; it is not a relative residual stopping criterion. Multiple criteria
use the most restrictive retained rank.

TT-SVD accepts a shared integer rank cap. It supports leading structural
batches declared through ``DenseTensorSource(..., n_batches=...)``.

TR-SVD
^^^^^^

:class:`TRSVD` constructs a cyclic chain, based on the TR-SVD approach of
Zhao et al. [trdecomp]_. This implementation starts with an interior
bipartition and decomposes the two resulting subchains, rather than using
the original one-direction construction.

.. code-block:: python

   axis = torch.arange(1, 5, dtype=torch.float64)
   tensor = torch.einsum('i,j,k,l->ijkl', axis, axis, axis, axis) + 1
   source = d.DenseTensorSource(tensor)
   engine = d.TRSVD(source, center=2)
   result = engine.fit(rank=2, collect_metrics=True)
   approximation = result.contract_dense()

Here ``center`` specifies the cut between sites ``center - 1`` and ``center``.
An integer ``rank`` caps every TR bond, including the cyclic bond; its square
caps the initial matrix rank. The initial retained dimension is factored
into two TR dimensions. Any structural zero padding needed for that
factorization is recorded in ``metadata``.

The singular-value criteria are the same as in TT-SVD. TR-SVD does not support
structural batches, and its local truncation diagnostics are not a global
reconstruction bound.

Repeated fits and numerical options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both engines allow independent fits with different truncation controls:

.. code-block:: python

   source = d.DenseTensorSource(torch.eye(4, dtype=torch.float64))
   engine = d.TTSVD(source)
   rank_one = engine.fit(rank=1)
   rank_two = engine.fit(rank=2)

``renormalize`` controls numerical scaling, and ``out_device`` selects where
final cores are stored. The exact SVD backend is configured through
:func:`~tensorkrowch.set_svd_method` or
:func:`~tensorkrowch.svd_method`. These methods need no extra strategy object
for ordinary truncation. Matrix engines are :class:`TTMSVD` and
:class:`TRMSVD`; input preparation is described below.

Alternating least squares
-------------------------

ALS optimizes one core while holding the others fixed, then visits the next
site. Use it to fit prescribed ranks, refine an existing approximation, or
fit selected observations. Unlike a direct SVD, an ALS fit depends on its
initialization and stopping criteria.

TT-ALS
^^^^^^

:class:`TTALS` uses alternating forward and reverse passes over an open
chain. This is the one-core alternating linear scheme underlying TT
optimization [ttals]_; it does not implement two-core MALS rank adaptation.

.. code-block:: python

   axis = torch.arange(1, 5, dtype=torch.float64)
   tensor = torch.einsum('i,j,k,l->ijkl', axis, axis, axis, axis) + 1
   source = d.DenseTensorSource(tensor)
   engine = d.TTALS(source)
   convergence = d.ConvergencePolicy(max_sweeps=30, rtol=1e-8)
   result = engine.fit(rank=2, convergence=convergence)

Without ``initial_cores``, initialization is random. To refine another fit,
pass its result or cores as ``initial_cores``; an SVD initialization is an
explicit composition with ``TTSVD``. ``fixed_cores`` freezes selected cores.
The default ``gauge="qr"`` moves the orthogonality center between updates;
``gauge="none"`` replaces cores without that QR preparation.

A rank sequence has one entry per open-chain bond, so a four-site TT uses
three entries. Each directional pass counts as one sweep.

TR-ALS
^^^^^^

:class:`TRALS` optimizes a cyclic chain using forward passes. Its cyclic
least-squares formulation follows the tensor ring approach [trdecomp]_.
Initialization, conditioning and convergence can differ from TT-ALS because
every core participates in a closed loop.

.. code-block:: python

   cores = [torch.randn(2, 4, 2, dtype=torch.float64) for _ in range(5)]
   source = d.FormatSource(tk.formats.TR(cores))
   engine = d.TRALS(source)
   convergence = d.ConvergencePolicy(
       max_sweeps=30, rtol=1e-8, keep_best=True)
   result = engine.fit(rank=2, convergence=convergence)

A TR rank sequence has one right-link rank per core. Its last entry is the
cyclic rank, so a five-site TR uses five entries. QR is a local gauge transfer
in a ring; it does not make the whole cyclic environment orthonormal.

Configuring convergence and local solves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same objects configure both engines:

* :class:`ConvergencePolicy` sets the sweep budget, absolute and relative
  objective-error thresholds, stagnation criteria, and optional retention of
  the best complete-sweep approximation. A stagnation threshold indicates
  little change, not necessarily a small reconstruction error.
* :class:`LeastSquaresSolver` configures the PyTorch least-squares solve,
  conditioning controls and optional Tikhonov regularization.
* :class:`SamplingPolicy` replaces complete local systems with sampled rows.
  The next section lists its sampling choices.

For example, a regularized refinement can reuse a previous result:

.. code-block:: python

   source = d.DenseTensorSource(torch.ones(3, 4, 3, dtype=torch.float64))
   initial = d.TTSVD(source).fit(rank=2)
   engine = d.TTALS(source)
   solver = d.LeastSquaresSolver(l2_reg=1e-10)
   convergence = d.ConvergencePolicy(
       max_sweeps=20, stagnation_rtol=1e-8, patience=3, keep_best=True)
   result = engine.fit(
       initial_cores=initial, solver=solver, convergence=convergence)

Sampling and partial observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass a ``SamplingPolicy`` to ``fit`` when local systems should use a subset
of rows. The policy selects topology-specific samplers internally:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Policy method
     - TT
     - TR
   * - ``"uniform"``
     - :class:`UniformRows`
     - :class:`UniformRows`
   * - ``"leverage"``
     - :class:`TTLeverageRows`, using mixed-canonical cores [ttleverage]_.
     - :class:`TRProductLeverageRows`, using a product proposal [trproduct]_.
   * - ``"exact_leverage"``
     - Not available.
     - :class:`TRExactLeverageRows`, using cyclic Gram contractions
       [trexact]_.

.. code-block:: python

   source = d.DenseTensorSource(torch.ones(4, 4, 4, 4, dtype=torch.float64))
   engine = d.TTALS(source)
   sampler = d.SamplingPolicy(n_samples=64, method='leverage')
   convergence = d.ConvergencePolicy(
       max_sweeps=20, rtol=1e-8, error_samples=128)
   result = engine.fit(
       rank=2, sampler=sampler, convergence=convergence,
       collect_metrics=True)

Sampled least-squares rows are corrected using their proposal probabilities.
``reuse_sweeps`` controls reuse of uniform samples; leverage proposals depend
on the current site and cores. TR leverage methods retain complete active
physical fibers, so ``n_samples`` counts exterior configurations rather
than scalar rows.

Stopping errors are measured on a separate fixed uniform evaluation set,
not on each changing training sample. ``ConvergencePolicy.error_samples``
sets its budget; ``None`` requests exact evaluation. Estimated measurements
are identified in ``result.metadata['error_evaluation']``.

For missing data, pass a boolean ``mask`` to the engine constructor: true
entries participate, including observed zeros. ``weights`` separately
multiplies participating residuals before their squared norm is taken.
An ordinary sparse source still represents known zeros outside its support;
it is not automatically a completion dataset. :class:`ALSProblem` describes
these objective distinctions, and :class:`ObservedRows` exposes permanent
sparse observations for lower-level use. Neither is required for ordinary
engine fits.

TT to TR conversion and spectral recovery
-----------------------------------------

The ring module converts an existing TT to a cyclic representation. It also
provides local loop-opening strategies, reused by TR-RSS. The optimization
route combines local TR-ALS with recursive gauge transport; the spectral
route uses BLOSTR [blostr]_. The recursive enrichment and automatic block
handling are implementation extensions, not guarantees from the BLOSTR paper.

Converting an existing TT
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`TT2TR` takes a TT rather than a source. When starting from values,
first obtain that TT using a source and another decomposition:

.. code-block:: python

   axis = torch.arange(1, 5, dtype=torch.float64)
   tensor = torch.einsum('i,j,k,l->ijkl', axis, axis, axis, axis) + 1
   source = d.DenseTensorSource(tensor)
   initial_tt = d.TTSVD(source).fit(rank=4)
   converter = d.TT2TR(initial_tt)
   result = converter.fit(rank=2, collect_metrics=True)

An integer ``rank`` specifies the local TR ranks; a sequence gives one
right-link rank per core, with the cyclic rank last. Omitting it requests
rank discovery. With optimization openers, the initial site grows into a
block when necessary to expose sufficient physical dimensions.
``block_boundaries`` controls automatic end blocks; disabling it for
optimization requires prescribed ranks.

``center`` selects the initial site or the center of the discovery block.
At block SVD splits, ``rank`` acts as a cap and an optional ``rtol`` can
reduce the returned ranks. This tolerance controls discarded squared
singular-value mass, not ALS convergence.

Choosing a loop opener
^^^^^^^^^^^^^^^^^^^^^^

A loop opener factors a local target into TR cores and two external gauges.
Choose a strategy to control that factorization:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Object
     - Role
   * - :class:`ALSLoopOpener`
     - Optimize the local factors. Its ``fit_options`` accepts TR-ALS controls
       such as a solver and convergence policy.
   * - :class:`QRLoopOpener`
     - Balance an opening with QR, use another opener, restore the factors
       and optionally refine in the original target coordinates.
   * - :class:`BLOSTRLoopOpener`
     - Recover all local factors using the spectral BLOSTR algorithm.
   * - :class:`BLOSTRSVDLoopOpener`
     - Recover the first spectral factor, then split the remainder by SVD.
   * - :class:`CallableLoopOpener`
     - Supply a custom implementation of the :class:`LoopOpener` contract.

The strings ``"als"``, ``"blostr"`` and ``"blostr_svd"`` select the
corresponding built-in strategies with default settings. Use objects when
their detailed options are needed. :class:`LoopOpening` documents the
returned local factors; ordinary conversion code need not construct it.

For a conversion with customized local optimization, pass a configured
opener. This example also balances each target and refines its factors:

.. code-block:: python

   cores = [torch.randn(2, 4, 2, dtype=torch.float64) for _ in range(5)]
   source = d.FormatSource(tk.formats.TR(cores))
   initial_tt = d.TTSVD(source).fit(rank=4)
   convergence = d.ConvergencePolicy(
       max_sweeps=50, rtol=1e-10, keep_best=True)
   opener = d.QRLoopOpener(d.ALSLoopOpener({
       'convergence': convergence,
   }))
   converter = d.TT2TR(initial_tt)
   result = converter.fit(rank=2, loop_opener=opener, enrich=True)

Recursion and enrichment
^^^^^^^^^^^^^^^^^^^^^^^^

Optimization openings are connected using a gauge recursion. The supported
choices are:

* ``"adjoint"`` / :class:`AdjointGaugeRecursion`: move QR centers, or use
  the corresponding isometries of a Vidal representation. This is the default.
* ``"pinv"`` / :class:`PinvGaugeRecursion`: use pseudoinverses of local
  TT cores or contracted blocks.
* ``"orth"`` / :class:`OrthGaugeRecursion`: use pseudoinverses of the
  learned gauges.

A configured :class:`GaugeRecursion` strategy accepts PyTorch pseudoinverse
keyword arguments. ``enrich=True`` adds orthogonal residual constraints to
the transported sample space without increasing the requested TR ranks.
``enrich_rank`` and ``enrich_rtol`` limit those extra coordinates.

Exact spectral recovery
^^^^^^^^^^^^^^^^^^^^^^^

BLOSTR is intended for exact spectral structure under its dimensional and
genericity assumptions [blostr]_. Choosing a BLOSTR opener in ``TT2TR``
selects an exact propagation route: recover the central factors, absorb
square gauges into neighbors and continue with SVDs. It does not use the
optimization gauge recursions or enrichment.

The exact route selects the smallest interior block containing ``center``
whose two external TT bond dimensions agree. It searches both directions
without a physical-capacity requirement. Equal-size choices favor a balanced
block, then the leftmost one. If no interior block qualifies, the complete
chain is opened. This selection does not guarantee recovery by every
spectral opener: an internal rank bottleneck can require ``blostr_svd``
with a joint Gaussian projection. Full BLOSTR retains its uniform-rank and
injectivity requirements; physical blocking to meet them is left to the user.
With uniform TT bond dimensions the selected central block has one site.
For either opener, a multi-site spectral block with ``projection='slices'``
emits a warning recommending ``projection='gaussian'``.
Full BLOSTR also warns before rejecting unequal local TR ranks, recommending
BLOSTR-SVD instead. If a physical dimension is smaller than the product of
its TR ranks, it warns to block the input TT (or group axes of a dense input).
This dimension condition is necessary for injectivity, but meeting it does
not guarantee full-rank unfoldings or nonsingular slice environments.

The standalone functions :func:`tr_blostr` and :func:`tr_blostr_svd` instead
take a dense tensor and return a TR decomposition directly. Their API
documents rank discovery, slice selection and Gaussian projection options.
With ``projection='slices'``, full BLOSTR uses discrete configurations in both
its spectral and subsequent recovery steps. With ``projection='gaussian'``,
the first step uses dense projections over all interior physical axes. Each
subsequent step leaves two neighboring physical axes open and independently
contracts the known prefix and unknown suffix with dense Gaussian tensors.
Those projections are separable between the two groups, but dense within
each group. BLOSTR-SVD instead splits the remaining tensor with SVDs.
The Gaussian projection and first-factor/SVD variant are extensions of the
basic spectral construction. Arbitrary noisy inputs need not satisfy the
requirements for exact recovery.

For an exact generic TR input, the first-factor/SVD route can be selected
explicitly. BLOSTR uses complex spectral factors and automatically promotes
the working TT cores, so real inputs are accepted and the result is complex.
The input TT remains unchanged:

.. code-block:: python

   cores = [torch.randn(2, 4, 2, dtype=torch.float64) for _ in range(5)]
   source = d.FormatSource(tk.formats.TR(cores))
   initial_tt = d.TTSVD(source).fit(rank=4)
   opener = d.BLOSTRSVDLoopOpener({
       'projection': 'gaussian', 'n_attempts': 4,
   })
   converter = d.TT2TR(initial_tt)
   result = converter.fit(rank=2, loop_opener=opener, rtol=1e-8)

Recursive sketching
-------------------

Sketching builds smaller local tensors from evaluations or contractions,
then connects their factors. It avoids materializing the complete target.
TT-RSS and TR-RSS share sampled preparation; TT-RS uses explicitly selected
contraction operators. These methods do not support structural batches.

TT-RSS: a tensor train from evaluations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`TTRSS` implements Tensor Train Recursive Sketching from Samples
[ttrss]_. The source must be evaluable at new configurations: supplied
samples determine the sketches, rather than restricting queries to those
original points.

.. code-block:: python

   def function(coordinates):
       return (1 + coordinates).prod(dim=1)

   source = d.CallableSource(function, n_sites=4, dtype=torch.float64)
   domain = torch.linspace(0, 1, 8, dtype=torch.float64)
   samples = torch.rand(128, 4, dtype=torch.float64)

   def embedding(values):
       return torch.stack((torch.ones_like(values), values), dim=-1)

   engine = d.TTRSS(source, embedding=embedding, domain=domain)
   result = engine.fit(samples, rank=2, center=1, collect_metrics=True)
   embedded = embedding(samples)
   values = result.evaluate(embedded)

Continuous inputs need an embedding and a domain on which its coefficients
can be fitted. A shared embedding/domain applies to every input; sequences
allow different choices per input. Finite tensor sources use basis embeddings
by default, so they need neither argument.

``rank`` caps TT bonds; ``None`` discovers numerical ranks. ``center`` is a
site: local SVDs act on both sides there and on the outgoing side elsewhere.
Singular spectra remain absorbed in the prepared tensors. ``batch_size``
limits source-evaluation batches, not the number of sketch samples.

TR-RSS: a tensor ring from evaluations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`TRRSS` uses the same source, samples, embeddings and transformations,
but constructs a cyclic representation through local loop openings. It is
an experimental extension combining TT-RSS [ttrss]_ with the ring opening
machinery described above; it is not the TT algorithm with a final unit-rank
closing bond.

.. code-block:: python

   cores = [torch.randn(2, 4, 2, dtype=torch.float64) for _ in range(5)]
   source = d.FormatSource(tk.formats.TR(cores))
   samples = torch.randint(4, (128, 5))
   engine = d.TRRSS(source)
   result = engine.fit(samples, rank=2, collect_metrics=True)

The rank sequence has one right-link rank per final core, with the cyclic
rank last. Sampled TT-cut caps are formed from products of cyclic and
internal TR ranks. With no prescribed rank, local SVDs and blocks determine
the numerical dimensions used for discovery.

``center`` seeds an automatic initial block. End blocks absorb the incoming
gauges before being split; ``block_boundaries=False`` keeps single-site ends
when ranks are prescribed. The block and rank conventions follow ``TT2TR``.
Recursive steps transport gauges through sketch evaluations, retaining the
inherited sample coordinates. Local ALS stopping criteria belong to the
opener's convergence policy, while ``fit`` truncation criteria control SVDs.
Pass a configured ``ALSLoopOpener`` to customize those solves, or wrap it in
``QRLoopOpener`` for balancing and refinement as in the conversion example.

Choosing fitters and projectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both sampled engines accept strategy objects in their constructors:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Object
     - When to use it
   * - :class:`BasisFitter`
     - Represent a finite physical axis in its discrete basis.
   * - :class:`FixedEmbeddingFitter`
     - Fit coefficients in a supplied embedding, with optional solver settings.
   * - :class:`TrainableEmbeddingFitter`
     - Optimize a parameterized embedding while fitting physical fibers.
   * - :class:`IdentityRangeProjector`
     - Apply the local SVD directly; this is also the default when omitted.
   * - :class:`RandomizedRangeProjector`
     - Compress the local SVD input using a configured randomized projection.

Pass a fitter through ``input_fitter`` and a projector through
``range_projector``. :class:`InputFitter` and :class:`RangeProjector` document
the extension contracts. ``local_solver`` configures the linear solves used
to align the TT cores; it is separate from a TR loop opener's ALS solver.

.. code-block:: python

   source = d.DenseTensorSource(torch.ones(4, 4, 4, 4, dtype=torch.float64))
   samples = torch.randint(4, (64, 4))
   projector = d.RandomizedRangeProjector(
       projection_dim=4, projection_oversampling=2, n_power_iter=1)
   engine = d.TTRSS(source, range_projector=projector)
   result = engine.fit(samples, rank=2)

Transforming evaluations
^^^^^^^^^^^^^^^^^^^^^^^^

For TT-RSS and TR-RSS, ``transform`` is one callback on a shared
:class:`ConfigurationMap`. It receives query configurations and raw source
values, and returns transformed values of the same shape. Its ``evaluate``
method queries additional points through the cache without increasing the
sketch sample set. For example, a reflection average is:

.. code-block:: python

   def function(coordinates):
       return (1 + coordinates).prod(dim=1)

   def transform(configuration_map):
       reflected = 1 - configuration_map.configurations
       return (configuration_map.values +
               configuration_map.evaluate(reflected)) / 2

   source = d.CallableSource(function, n_sites=3, dtype=torch.float64)
   domain = torch.linspace(0, 1, 8, dtype=torch.float64)
   samples = torch.rand(64, 3, dtype=torch.float64)

   def embedding(values):
       return torch.stack((torch.ones_like(values), values), dim=-1)

   engine = d.TTRSS(
       source, embedding=embedding, domain=domain, transform=transform)
   result = engine.fit(samples, rank=2)

Tensor-valued functions
^^^^^^^^^^^^^^^^^^^^^^^

A callable can return output axes as well as its configuration batch. Declare
them through the source's ``out_shape`` and select the input variables with
``in_sites``; ``n_sites`` counts both input and output variables.
``out_position`` places the output sites in the fitted chain. ``labels``
optionally selects output components for building the sample sketches.

.. code-block:: python

   def function(coordinates):
       value = coordinates.sum(dim=1)
       return torch.stack((value, value.square()), dim=1)

   def embedding(values):
       return torch.stack((torch.ones_like(values), values,
                           values.square()), dim=-1)

   source = d.CallableSource(
       function, n_sites=4, out_shape=(2,), dtype=torch.float64,
       in_sites=(0, 1, 2))
   engine = d.TTRSS(
       source, embedding=embedding,
       domain=torch.linspace(0, 1, 8, dtype=torch.float64), out_position=1)
   samples = torch.rand(64, 3, dtype=torch.float64)
   labels = torch.arange(64).remainder(2)
   result = engine.fit(samples, labels=labels, rank=4)

TT-RS: explicit contraction sketches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`TTRS` builds local core-determining equations from a finite source
and a :class:`SketchOperator`, following tensor train sketching [ttrs]_.
Its finite-source interface also permits exact contractions rather than
only empirical estimates from observations. Centered construction and the
available operators extend that basic formulation.

.. code-block:: python

   dataset = torch.randint(4, (128, 4))
   source = d.SparseTensorSource.from_data(dataset, in_dim=(4, 4, 4, 4))
   sketch = d.MarginalSketch(order=1)
   engine = d.TTRS(source, sketch_operator=sketch)
   result = engine.fit(rank=2, collect_metrics=True)

Available contraction strategies are:

* :class:`SampledSketch`: select fibers using supplied or generated indices.
* :class:`MarginalSketch`: form local marginals, with configurable order and
  optional per-site factors. This is the default strategy.
* :class:`TTStackSketch`: contract a stack of TT sketches; its construction is
  related to randomized TT sketching [ttstack]_.

Here ``transform`` acts separately on each three-axis Phi tensor, returning
a tensor of the same shape. This allows local smoothing or other user-defined
operations on the projected tensors. It differs from the shared evaluation
callback used by RSS. The callback is responsible for preserving whatever
consistency its application requires.

``center``, SVD truncation and ``range_projector`` work as in the other
sketching engines. ``strict_system`` enables additional checks on the
core-determining equations. :class:`PhiOperator` and
:class:`CoreDeterminingSystem` document lower-level access for custom
sketches. There is no separate TR-RS wrapper: compose the resulting TT with
``TT2TR`` when needed.

Matrix and Quantics extensions
------------------------------

These extensions describe the input and result representation; the numerical
algorithms and their configurable strategies remain the same.

Matrices
^^^^^^^^

A matrix source has paired input/output dimensions at every site. Dense,
sparse and callable sources provide ``.matrix(...)`` constructors;
``FormatSource`` reads those dimensions from a matrix format.

.. code-block:: python

   # Two input axes followed by two output axes.
   tensor = torch.randn(3, 4, 3, 4, dtype=torch.float64)
   source = d.DenseTensorSource.matrix(tensor, ordering='grouped')
   tt_engine = d.TTMSVD(source)
   tr_engine = d.TRMSVD(source)
   ttm = tt_engine.fit(rank=4)
   trm = tr_engine.fit(rank=4)

``ordering="grouped"`` means ``(in_0, in_1, ..., out_0, out_1, ...)``;
``"interleaved"`` means ``(in_0, out_0, in_1, out_1, ...)``. An ordinary
two-axis matrix is one paired site; reshape it first to specify more sites.
Sparse matrix sources additionally need ``in_dim`` and ``out_dim``.

General engines such as ``TTALS``, ``TRALS``, ``TTRSS`` and ``TRRSS`` infer
matrix results from the source. They return :class:`TTMDecomposition` or
:class:`TRMDecomposition`, with each physical input/output pair restored.
``TTMSVD`` and ``TRMSVD`` explicitly require matrix inputs. Matrix ``apply``
and contraction behavior follow the corresponding formats.

Quantics
^^^^^^^^

Quantics sources discretize coordinate functions into digit sites. Construct
the source with ``base``, ``level`` and ``domain`` or explicit
``grid_coordinates``. A preconfigured
:class:`~tensorkrowch.formats.QuantizedLayout` and
:class:`~tensorkrowch.formats.CoordinateMap` can express more detailed layouts
and grids. These digit representations follow the Quantics construction
[quantics]_.

.. code-block:: python

   def function(coordinates):
       return torch.exp(coordinates.sum(dim=1))

   source = d.CallableQuanticsSource(
       function, n_coordinates=2, base=2, level=(3, 4),
       domain=torch.tensor([[0., 1.], [-1., 1.]], dtype=torch.float64),
       dtype=torch.float64)
   tt_engine = d.TTSVD(source)
   tr_engine = d.TRSVD(source)
   qtt = tt_engine.fit(rank=2)
   qtr = tr_engine.fit(rank=2)
   coordinates = torch.tensor([[0., -1.], [0.5, 0.]], dtype=torch.float64)
   values = qtt.evaluate_coordinates(coordinates)

The results are :class:`QTTDecomposition` or :class:`QTRDecomposition`;
matrix sources yield :class:`QTTMDecomposition` or
:class:`QTRMDecomposition`. Results retain their digit layouts and coordinate
maps. SVD materializes the full digit tensor, while ALS and RSS can query it.

Sampled sketching accepts domain-coordinate samples for Quantics sources.
An explicit ``ConfigurationBatch(..., kind="indices")`` supplies digit
indices instead. The layout can group digits by variable or interleave them
by level. Functions :func:`qtt_svd`, :func:`qtr_svd`, :func:`qttm_svd`,
:func:`qtrm_svd`, :func:`qtt_rss` and :func:`qtr_rss` are convenient adapters,
not separate numerical engines. Ordinary dense tensors are not automatically
quantized.

Results, progress and diagnostics
---------------------------------

Every decomposition returns a :class:`TensorDecomposition` together with the
appropriate numerical format interface. Use ``result.cores`` to access its
cores, ``evaluate`` for selected inputs, and ``contract_dense`` when full
materialization is affordable. Continuous RSS results are evaluated on
embedded inputs, as shown above; Quantics results also support coordinate
evaluation.

``verbose`` controls console progress. ``collect_metrics=True`` retains
algorithm-specific records in :class:`DecompositionMetrics`: truncations,
local solves, sweep errors, evaluations and timings. See each ``fit`` method
for its reporting levels. A small local opening or sampled solve residual
does not by itself certify a small global reconstruction error.

``result.as_info()`` combines representation information and diagnostics.
``result.metadata`` records the fit setup. These describe the original fit;
editing or rounding the result does not recompute historical errors.
``result.error(...)`` can measure the current approximation on supplied data.
``result.to_format()`` returns the numerical representation without fit
diagnostics. Copying, detaching and moving results preserve their history;
new algebraic operations return ordinary formats.

The legacy :func:`vec_to_mps` and :func:`mat_to_mpo` remain available for
compatibility and emit deprecation warnings. New code should use the engine
or functional decomposition interfaces and construct models explicitly.

API reference
-------------

.. toctree::
   :maxdepth: 2

   decompositions_api

References
----------

.. [ttsvd] I. V. Oseledets. `Tensor-Train Decomposition
   <https://doi.org/10.1137/090752286>`_ (2011).

.. [trdecomp] Q. Zhao et al. `Tensor Ring Decomposition
   <https://arxiv.org/abs/1606.05535>`_ (2016).

.. [ttals] S. Holtz, T. Rohwedder and R. Schneider. `The Alternating Linear
   Scheme for Tensor Optimization in the Tensor Train Format
   <https://doi.org/10.1137/100818893>`_ (2012).

.. [ttleverage] V. Bharadwaj et al. `Efficient Leverage Score Sampling for
   Tensor Train Decomposition <https://arxiv.org/abs/2406.02749>`_ (2024).

.. [trproduct] O. A. Malik and S. Becker. `A Sampling-Based Method for Tensor
   Ring Decomposition <https://arxiv.org/abs/2010.08581>`_ (2021).

.. [trexact] O. A. Malik, V. Bharadwaj and R. Murray. `Sampling-Based
   Decomposition Algorithms for Arbitrary Tensor Networks
   <https://arxiv.org/abs/2210.03828>`_ (2022).

.. [blostr] H. Chen, S. Chen and A. R. Zhang. `A Provably Efficient Method
   for Tensor Ring Decomposition and Its Applications
   <https://arxiv.org/abs/2512.01016>`_ (2025).

.. [ttrss] J. R. Pareja Monturiol, A. Pozas-Kerstjens and D. Perez-Garcia.
   `Tensorization of neural networks for improved privacy and interpretability
   <https://arxiv.org/abs/2501.06300>`_ (2025).

.. [ttrs] Y. Hur et al. `Generative modeling via tensor train sketching
   <https://arxiv.org/abs/2202.11788>`_ (2023).

.. [ttstack] P. Cazeaux, M.-S. Dupuy and R. Figueroa Justiniano.
   `Linear-Scaling Tensor Train Sketching <https://arxiv.org/abs/2603.11009>`_
   (2026).

.. [quantics] B. N. Khoromskij. `O(d log N)-Quantics Approximation of
   N-d Tensors in High-Dimensional Numerical Modeling
   <https://doi.org/10.1007/s00365-011-9131-1>`_ (2011).
