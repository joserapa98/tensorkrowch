Decomposition API reference
===========================

.. currentmodule:: tensorkrowch.decompositions

See :doc:`decompositions` for the source-to-engine workflow, examples and
help choosing an algorithm or strategy. This page lists the public objects
by responsibility. Matrix and Quantics representations are documented in
:doc:`formats`.

Input sources
-------------

.. autoclass:: BaseSource
   :members:

.. autoclass:: ConfigurationBatch
   :members:

.. autoclass:: DenseTensorSource
   :members:

.. autoclass:: SparseTensorSource
   :members:

.. autoclass:: FormatSource
   :members:

.. autoclass:: CallableSource
   :members:

.. autoclass:: CallableTensorSource
   :members:

.. autoclass:: CallableQuanticsSource
   :members:

.. autofunction:: as_tensor_source

.. autofunction:: gaussian_kernel

SVD engines and functions
-------------------------

.. autoclass:: TTSVD
   :members: fit

.. autoclass:: TRSVD
   :members: fit

.. autoclass:: TTMSVD
   :members: fit

.. autoclass:: TRMSVD
   :members: fit

.. autofunction:: tt_svd

.. autofunction:: tr_svd

.. autofunction:: ttm_svd

.. autofunction:: trm_svd

.. autofunction:: qtt_svd

.. autofunction:: qtr_svd

.. autofunction:: qttm_svd

.. autofunction:: qtrm_svd

.. autofunction:: tensorkrowch.set_svd_method

.. autofunction:: tensorkrowch.svd_method

ALS engines and policies
------------------------

.. autoclass:: TTALS
   :members: fit

.. autoclass:: TRALS
   :members: fit

.. autofunction:: tt_als

.. autofunction:: tr_als

.. autoclass:: ConvergencePolicy
   :members:

.. autoclass:: SamplingPolicy
   :members:

.. autoclass:: LeastSquaresSolver
   :members:

ALS objectives, samplers and environments
-----------------------------------------

These lower-level objects support custom objectives and execution. Ordinary
fits configure sampling through ``SamplingPolicy`` rather than constructing
environment caches or row samplers directly.

.. autoclass:: ALSProblem
   :members:

.. autoclass:: RowSampler
   :members:

.. autoclass:: ExactRows
   :members:

.. autoclass:: ObservedRows
   :members:

.. autoclass:: UniformRows
   :members:

.. autoclass:: TTLeverageRows
   :members:

.. autoclass:: TRProductLeverageRows
   :members:

.. autoclass:: TRExactLeverageRows
   :members:

.. autoclass:: SampleBatch
   :members:

.. autoclass:: EnvironmentCache
   :members:

.. autoclass:: TTEnvironmentCache
   :members:

.. autoclass:: DirectTREnvironment
   :members:

.. autoclass:: TRSegmentEnvironmentCache
   :members:

.. autoclass:: ALSSweepDriver
   :members:

Ring conversion and spectral functions
--------------------------------------

.. autoclass:: TT2TR
   :members: fit

.. autofunction:: tt2tr

.. autofunction:: tr_blostr

.. autofunction:: tr_blostr_svd

Loop-opening strategies and gauge recursions
--------------------------------------------

.. autoclass:: LoopOpener
   :members:

.. autoclass:: ALSLoopOpener
   :members:

.. autoclass:: QRLoopOpener
   :members:

.. autoclass:: BLOSTRLoopOpener
   :members:

.. autoclass:: BLOSTRSVDLoopOpener
   :members:

.. autoclass:: CallableLoopOpener
   :members:

.. autoclass:: LoopOpening
   :members:

.. autoclass:: LoopOpenerCapabilities
   :members:

.. autoclass:: GaugeRecursion
   :members:

.. autoclass:: AdjointGaugeRecursion
   :members:

.. autoclass:: PinvGaugeRecursion
   :members:

.. autoclass:: OrthGaugeRecursion
   :members:

.. autoclass:: GaugeMap
   :members:

.. autoexception:: ExperimentalWarning

Sketching engines and functions
-------------------------------

.. autoclass:: TTRSS
   :members: fit

.. autoclass:: TRRSS
   :members: fit

.. autoclass:: TTRS
   :members: fit

.. autofunction:: tt_rss

.. autofunction:: tr_rss

.. autofunction:: tt_rs

.. autofunction:: qtt_rss

.. autofunction:: qtr_rss

Sketching fitters, projectors and operators
-------------------------------------------

.. autoclass:: InputFitter
   :members:

.. autoclass:: BasisFitter
   :members:

.. autoclass:: FixedEmbeddingFitter
   :members:

.. autoclass:: TrainableEmbeddingFitter
   :members:

.. autoclass:: RangeProjector
   :members:

.. autoclass:: IdentityRangeProjector
   :members:

.. autoclass:: RandomizedRangeProjector
   :members:

.. autoclass:: SketchOperator
   :members:

.. autoclass:: SampledSketch
   :members:

.. autoclass:: MarginalSketch
   :members:

.. autoclass:: TTStackSketch
   :members:

Sketching evaluation and extension objects
------------------------------------------

These objects support transformations and custom sketches. Engines create
their evaluation maps and Phi operators automatically.

.. autoclass:: ConfigurationMap
   :members:

.. autoclass:: PhiOperator
   :members:

.. autoclass:: CoreDeterminingSystem
   :members:

.. autoclass:: RecursiveSketching
   :members:

Results and diagnostics
-----------------------

.. autoclass:: TensorDecomposition
   :members: as_info, error, to_format

.. autoclass:: TTDecomposition

.. autoclass:: TRDecomposition

.. autoclass:: TTMDecomposition

.. autoclass:: TRMDecomposition

.. autoclass:: QTTDecomposition

.. autoclass:: QTRDecomposition

.. autoclass:: QTTMDecomposition

.. autoclass:: QTRMDecomposition

.. autoclass:: DecompositionMetrics
   :members:

.. autoclass:: ErrorRecord

.. autoclass:: TruncationRecord

.. autoclass:: InputFitRecord

.. autoclass:: RangeProjectionRecord

.. autoclass:: LocalSolveRecord

.. autoclass:: GaugeRecord

.. autoclass:: SweepRecord

.. autoclass:: TimingRecord

.. autoclass:: EvaluationStats

.. autoclass:: FidelityRecord

Compatibility helpers
---------------------

These historical model-construction functions emit deprecation warnings.
Use the numerical decomposition interfaces for new code.

.. autofunction:: vec_to_mps

.. autofunction:: mat_to_mpo

