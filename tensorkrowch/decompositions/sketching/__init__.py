"""Recursive-sketching decomposition infrastructure."""

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


__all__ = [
    'EvaluationView',
    'InputFitter',
    'FixedEmbeddingFitter',
    'BasisFitter',
    'RangeProjector',
    'IdentityRangeProjector',
    'RandomizedRangeProjector',
    'GlobalValueTransform',
    'IdentityGlobalValueTransform',
    'CallableGlobalValueTransform',
    'CompositeGlobalValueTransform',
    'LocalTransformContext',
    'LocalValueTransform',
    'IdentityLocalValueTransform',
    'CallableLocalValueTransform',
    'CompositeLocalValueTransform',
]
