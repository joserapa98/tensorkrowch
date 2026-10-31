"""
Recursive sketching for tensor train and tensor ring decompositions.

    Sources and sampled evaluation:
        BaseSource -> ConfigurationMap -> PhiOperator
    Physical fitting and range selection:
        InputFitter -> optional RangeProjector -> directional SVD
    Sampled algorithms:
        RecursiveSketching -> TTRSS -> TRRSS
    Projected discrete algorithm:
        SketchOperator -> SketchSystemBuilder -> CoreDeterminingSystem -> TTRS
    Results:
        TTDecomposition / TRDecomposition, with matrix and Quantics variants

``TTRSS`` and ``TRRSS`` evaluate correlated sketch configurations jointly,
apply an optional global ``transform(configuration_map)``, fit physical axes
and select singular subspaces. ``TTRS`` instead contracts a finite source with
sampled, marginal or TT-stack operators; its optional ``transform(phi)`` acts
on each local three-axis tensor. Every algorithm can start at an interior
``center`` and construct the two outward branches.

``rank`` supplies a shared bound or one bound per bond; ``None`` requests
numerical discovery. TR rank profiles include the cyclic bond last. Spectra
remain absorbed in the truncated local tensors. TR-RSS forms central and
end blocks from aligned
TT cores, opens the central loop and propagates gauges with recursive sketches.
No additional source queries are introduced by those blocks.
An optional ``QRLoopOpener`` balances local TR targets and refines after
restoring its factors.

Classes support repeated fits and advanced fitting, projection and local-solve
strategies. Functions provide concise entry points and also return numerical
result objects. ``verbose`` controls progress and ``collect_metrics`` retains
diagnostics. The full source tensor need not be formed; local Phi tensors are
materialized for SVDs and ALS. ``PhiOperator`` also supports lazy selected
entries and fibers independently of these algorithms.
"""

from tensorkrowch.decompositions.sketching.base import RecursiveSketching
from tensorkrowch.decompositions.sketching.evaluations import ConfigurationMap
from tensorkrowch.decompositions.sketching.fitting import (
    BasisFitter, FixedEmbeddingFitter, InputFitter, TrainableEmbeddingFitter)
from tensorkrowch.decompositions.sketching.phi import PhiOperator
from tensorkrowch.decompositions.sketching.projections import (
    IdentityRangeProjector, RandomizedRangeProjector, RangeProjector)
from tensorkrowch.decompositions.sketching.sketches import (
    CoreDeterminingSystem, MarginalSketch, SampledSketch, SketchOperator,
    SketchSystemBuilder, TTStackSketch)
from tensorkrowch.decompositions.sketching.tr import TRRSS, qtr_rss, tr_rss
from tensorkrowch.decompositions.sketching.tt import (TTRS,
                                                      TTRSS,
                                                      qtt_rss,
                                                      tt_rs,
                                                      tt_rss)


__all__ = [
    'RecursiveSketching', 'ConfigurationMap', 'PhiOperator',
    'InputFitter', 'FixedEmbeddingFitter', 'BasisFitter',
    'TrainableEmbeddingFitter', 'RangeProjector', 'IdentityRangeProjector',
    'RandomizedRangeProjector', 'SketchOperator', 'SketchSystemBuilder',
    'CoreDeterminingSystem', 'SampledSketch', 'MarginalSketch',
    'TTStackSketch', 'TTRSS', 'tt_rss', 'TTRS', 'tt_rs', 'qtt_rss',
    'TRRSS', 'tr_rss', 'qtr_rss',
]
