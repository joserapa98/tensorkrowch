"""
Recursive-sketching decompositions.

    Sources and evaluations:
        * TensorSource -> EvaluationView -> PhiOperator
    Local construction:
        * SketchOperator -> RangeProjector -> InputFitter
    Algorithms:
        * TTRSS, TRRSS: sampled function evaluations
        * TTRS, TRRS: supplied sketch systems
    Results:
        * TTDecomposition, TRDecomposition
        * QTTDecomposition, QTRDecomposition

Each algorithm prepares evaluations, constructs and trims local systems, fits
input axes and solves for the cores. Results inherit numerical operations from
``formats`` and retain the algorithm's metrics. Quantics variants use the same
flow through ``QuantizedSourceAdapter``; layouts and coordinate maps belong to
``formats``.
"""

from tensorkrowch.decompositions.sketching.base import RecursiveSketching
from tensorkrowch.decompositions.sketching.evaluations import EvaluationView
from tensorkrowch.decompositions.sketching.fitting import (BasisFitter,
                                                           FixedEmbeddingFitter,
                                                           InputFitter,
                                                           TrainableEmbeddingFitter)
from tensorkrowch.decompositions.sketching.projections import (IdentityRangeProjector,
                                                               RandomizedRangeProjector,
                                                               RangeProjector)
from tensorkrowch.decompositions.sketching.quantization import (AffineCoordinateMap,
                                                                CoordinateMap,
                                                                ExplicitGridMap,
                                                                FunctionalCoordinateMap,
                                                                QuantizedLayout,
                                                                QuantizedSourceAdapter)
from tensorkrowch.decompositions.sketching.sketches import (CoreDeterminingSystem,
                                                            MarginalSketch,
                                                            SampledSketch,
                                                            SketchOperator,
                                                            SketchSystemBuilder,
                                                            TTStackSketch)
from tensorkrowch.decompositions.sketching.tr import (SketchGaugeRecursion,
                                                      TRRS,
                                                      TRRSS,
                                                      qtr_rss,
                                                      tr_rs,
                                                      tr_rss)
from tensorkrowch.decompositions.sketching.transforms import (CallableGlobalValueTransform,
                                                              CallableLocalValueTransform,
                                                              CompositeGlobalValueTransform,
                                                              CompositeLocalValueTransform,
                                                              GlobalValueTransform,
                                                              IdentityGlobalValueTransform,
                                                              IdentityLocalValueTransform,
                                                              LocalTransformContext,
                                                              LocalValueTransform)
from tensorkrowch.decompositions.sketching.tt import (TTRS,
                                                      TTRSS,
                                                      qtt_rss,
                                                      tt_rs,
                                                      tt_rss)


__all__ = [
    'RecursiveSketching',
    'EvaluationView',
    'InputFitter',
    'FixedEmbeddingFitter',
    'BasisFitter',
    'TrainableEmbeddingFitter',
    'RangeProjector',
    'IdentityRangeProjector',
    'RandomizedRangeProjector',
    'QuantizedLayout',
    'CoordinateMap',
    'AffineCoordinateMap',
    'FunctionalCoordinateMap',
    'ExplicitGridMap',
    'QuantizedSourceAdapter',
    'SketchOperator',
    'SketchSystemBuilder',
    'CoreDeterminingSystem',
    'SampledSketch',
    'MarginalSketch',
    'TTStackSketch',
    'GlobalValueTransform',
    'IdentityGlobalValueTransform',
    'CallableGlobalValueTransform',
    'CompositeGlobalValueTransform',
    'LocalTransformContext',
    'LocalValueTransform',
    'IdentityLocalValueTransform',
    'CallableLocalValueTransform',
    'CompositeLocalValueTransform',
    'TTRSS',
    'tt_rss',
    'TTRS',
    'tt_rs',
    'qtt_rss',
    'SketchGaugeRecursion',
    'TRRS',
    'tr_rs',
    'qtr_rss',
    'TRRSS',
    'tr_rss',
]
