"""
Recursive sketching for tensor train and tensor ring decompositions.

Sketches retain information about a tensor through selected evaluations or
contractions, without forming the complete tensor. Local tensors, called Phi,
combine a physical axis with left and right sketch axes. Their singular
subspaces determine the retained bonds, and recursive solves connect the
resulting cores.

Public engine hierarchy and basic interfaces:

    RecursiveSketching                         base.py
        +-- TTRSS                              tt.py
            +-- TRRSS                          tr.py

    TTRS                                       tt.py

    tt_rss(source, sketch_samples, ...) -> TTRSS(source).fit(...)
    tr_rss(source, sketch_samples, ...) -> TRRSS(source).fit(...)
    tt_rs(source, ...)                  -> TTRS(source).fit(...)

:class:`TTRSS` constructs a TT from correlated sample configurations and
source evaluations. :class:`TRRSS` shares that preparation and constructs a
cyclic TR using local loop openings. :class:`TTRS` instead contracts a finite
source with supplied sketch operators. Engines store the source and strategy
choices for repeated fits; functions provide concise entry points. Both return
decomposition objects containing cores, metadata and optional diagnostics.

Sampled evaluation and common preparation:

    source + sketch_samples + optional embedding/domain
        +-- ConfigurationMap             Unique queries and cached values
        +-- PhiOperator                  Local sampled tensors
        +-- InputFitter                  Physical basis coefficients
        +-- optional RangeProjector      Smaller matrices for local SVDs
        +-- directional SVD              Truncated Phi tensors
            +-- TTRSS                    Linear solves joining TT cores
            +-- TRRSS                    Loop opening and sketch recursion

:class:`ConfigurationMap` combines shared queries before source evaluation.
An optional ``transform(configuration_map)`` acts once on that shared table;
its ``evaluate`` method permits cached auxiliary source queries without adding
sketch samples. :class:`PhiOperator` gathers the local tensors from these
values. Phi tensors are materialized for current SVD and ALS workflows; the
operator also supports lazy selected entries and fibers.

Physical fitting and range projection use separate strategy contracts:

    InputFitter                                fitting.py
        +-- BasisFitter                        Finite basis coefficients
        +-- FixedEmbeddingFitter               Supplied embedding
        +-- TrainableEmbeddingFitter           Learned embedding parameters

    RangeProjector                             projections.py
        +-- IdentityRangeProjector             Direct local SVD
        +-- RandomizedRangeProjector           Randomized range compression

The engine selects a suitable input fitter when none is supplied. A configured
fitter exposes its detailed options. Range projection is disabled by default;
pass a projector object to configure randomized compression.

Projected discrete construction:

    TTRS(source, sketch_operator=...).fit(...)
        +-- SketchOperator.builder(source)
            +-- SketchSystemBuilder.build(...)
                +-- CoreDeterminingSystem
                    +-- local transform, SVDs and linear core solves

    SketchOperator                             sketches.py
        +-- SampledSketch                      Selected source fibers
        +-- MarginalSketch                     Weighted local marginals
        +-- TTStackSketch                      Stacked random TT contractions

TT-RS uses sparse support or structured TT contractions when available. Its
optional ``transform(phi)`` receives each three-axis Phi and returns a tensor
of the same shape before truncation. Unlike the sampled global transformation,
this callback acts independently on each local tensor. For example, kernels
can smooth the physical variables retained by marginal sketches.

Centered reconstruction and ring opening:

    center Phi: SVD on both sides
        +-- left branch:  left-facing SVDs and recursive solves
        +-- right branch: right-facing SVDs and recursive solves

Every algorithm accepts ``center``; an endpoint gives a single branch. ``rank``
is a shared cap or one cap per bond, while ``None`` requests numerical rank
discovery. TR profiles include the cyclic bond last. Singular values remain
absorbed in the truncated tensors; only the center retains the isometric maps
needed to start both sketch recursions.

TR-RSS forms central and end blocks by contracting an intermediate TT. These
blocks introduce no new source queries. The central block is passed directly
to the configured ``loop_opener``, then learned gauges propagate through the
sketches. End blocks absorb their gauges and split by SVD. The default
``ALSLoopOpener`` performs local TR fitting; an optional ``QRLoopOpener``
balances the free external axes and refines after restoring its factors.
Both strategies belong to :mod:`~tensorkrowch.decompositions.ring`.

Results and diagnostics:

    TT-RSS / TT-RS -> TTDecomposition
    TR-RSS        -> TRDecomposition
        +-- matrix and Quantics result families when applicable

The adapters :func:`qtt_rss` and :func:`qtr_rss` use coordinate sources and
layouts from ``formats`` through the sampled algorithms. ``verbose`` controls
progress and ``collect_metrics`` retains evaluations, truncations, local solves
and timings. ``result.to_format()`` removes fit diagnostics while retaining the
numerical approximation. Small local residuals alone do not bound the global
reconstruction error.
"""

from tensorkrowch.decompositions.sketching.base import RecursiveSketching
from tensorkrowch.decompositions.sketching.evaluations import ConfigurationMap
from tensorkrowch.decompositions.sketching.fitting import (
    BasisFitter,
    FixedEmbeddingFitter,
    InputFitter,
    TrainableEmbeddingFitter)
from tensorkrowch.decompositions.sketching.phi import PhiOperator
from tensorkrowch.decompositions.sketching.projections import (
    IdentityRangeProjector,
    RandomizedRangeProjector,
    RangeProjector)
from tensorkrowch.decompositions.sketching.sketches import (
    CoreDeterminingSystem,
    MarginalSketch,
    SampledSketch,
    SketchOperator,
    SketchSystemBuilder,
    TTStackSketch)
from tensorkrowch.decompositions.sketching.tr import (TRRSS,
                                                      qtr_rss,
                                                      tr_rss)
from tensorkrowch.decompositions.sketching.tt import (TTRS,
                                                      TTRSS,
                                                      qtt_rss,
                                                      tt_rs,
                                                      tt_rss)


__all__ = [
    'RecursiveSketching',
    'ConfigurationMap',
    'PhiOperator',
    'InputFitter',
    'FixedEmbeddingFitter',
    'BasisFitter',
    'TrainableEmbeddingFitter',
    'RangeProjector',
    'IdentityRangeProjector',
    'RandomizedRangeProjector',
    'SketchOperator',
    'SketchSystemBuilder',
    'CoreDeterminingSystem',
    'SampledSketch',
    'MarginalSketch',
    'TTStackSketch',
    'TTRSS',
    'tt_rss',
    'TTRS',
    'tt_rs',
    'qtt_rss',
    'TRRSS',
    'tr_rss',
    'qtr_rss',
]
