"""Recursive-sketching decomposition infrastructure."""

from tensorkrowch.decompositions.sketching.base import RecursiveSketching
from tensorkrowch.decompositions.sketching.evaluations import EvaluationView
from tensorkrowch.decompositions.sketching.fitting import (
    BasisFitter,
    FixedEmbeddingFitter,
    InputFitter,
)
from tensorkrowch.decompositions.sketching.projections import (
    IdentityRangeProjector,
    RandomizedRangeProjector,
    RangeProjector,
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
from tensorkrowch.decompositions.sketching.tt import TTRSS, tt_rss
from tensorkrowch.decompositions.sketching.tr import (SketchGaugeRecursion,
                                                      TRRSS, tr_rss)


__all__ = [
    'RecursiveSketching',
    'EvaluationView',
    'InputFitter',
    'FixedEmbeddingFitter',
    'BasisFitter',
    'RangeProjector',
    'IdentityRangeProjector',
    'RandomizedRangeProjector',
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
    'SketchGaugeRecursion',
    'TRRSS',
    'tr_rss',
]
