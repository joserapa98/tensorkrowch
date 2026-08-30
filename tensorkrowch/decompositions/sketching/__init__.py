"""Recursive-sketching decomposition infrastructure."""

from tensorkrowch.decompositions.sketching.base import RecursiveSketching
from tensorkrowch.decompositions.sketching.evaluations import EvaluationView
from tensorkrowch.decompositions.sketching.fitting import (
    BasisFitter,
    FixedEmbeddingFitter,
    InputFitter,
    QTTInputFitter,
    TrainableEmbeddingFitter,
)
from tensorkrowch.decompositions.sketching.projections import (
    IdentityRangeProjector,
    RandomizedRangeProjector,
    RangeProjector,
)
from tensorkrowch.decompositions.sketching.quantization import (
    CoordinateMap,
    ExplicitGridMap,
    QuantizedLayout,
    QuantizedSourceAdapter,
    UniformCoordinateMap,
    WarpedCoordinateMap,
)
from tensorkrowch.decompositions.sketching.sketches import (
    CoreDeterminingSystem,
    MarginalSketch,
    SampledSketch,
    SketchOperator,
    SketchSystemBuilder,
    TTStackSketch,
)
from tensorkrowch.decompositions.sketching.transforms import (
    CallableGlobalValueTransform,
    CallableLocalValueTransform,
    CompositeGlobalValueTransform,
    CompositeLocalValueTransform,
    GlobalValueTransform,
    IdentityGlobalValueTransform,
    IdentityLocalValueTransform,
    LocalTransformContext,
    LocalValueTransform,
)
from tensorkrowch.decompositions.sketching.tt import (QTTTuckerRSS, TTRS,
                                                      TTRSS, qtt_rss,
                                                      qtt_tucker_rss, tt_rs,
                                                      tt_rss)
from tensorkrowch.decompositions.sketching.tr import (QTRTuckerRSS,
                                                      SketchGaugeRecursion,
                                                      TRRS, TRRSS, qtr_rss,
                                                      qtr_tucker_rss, tr_rs,
                                                      tr_rss)


__all__ = [
    'RecursiveSketching',
    'EvaluationView',
    'InputFitter',
    'FixedEmbeddingFitter',
    'BasisFitter',
    'TrainableEmbeddingFitter',
    'QTTInputFitter',
    'RangeProjector',
    'IdentityRangeProjector',
    'RandomizedRangeProjector',
    'QuantizedLayout',
    'CoordinateMap',
    'UniformCoordinateMap',
    'WarpedCoordinateMap',
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
    'QTTTuckerRSS',
    'qtt_tucker_rss',
    'SketchGaugeRecursion',
    'TRRS',
    'tr_rs',
    'qtr_rss',
    'QTRTuckerRSS',
    'qtr_tucker_rss',
    'TRRSS',
    'tr_rss',
]
