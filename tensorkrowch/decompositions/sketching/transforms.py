"""Composable global and local value transforms for sketching pipelines."""

from dataclasses import dataclass
from typing import (Any, Callable, Optional, Protocol, Sequence, Tuple,
                    runtime_checkable)

import torch

from tensorkrowch.decompositions.sketching.evaluations import (
    EvaluationView,
    _EvaluationPlanBuilder,
    _concatenate_configurations,
)
from tensorkrowch.decompositions.sketching.phi import (PhiOperator, PhiView)
from tensorkrowch.decompositions.sources import ConfigurationBatch


@runtime_checkable
class GlobalValueTransform(Protocol):
    """Transform applied once to globally unique source evaluations."""

    @property
    def is_identity(self) -> bool:
        """Whether structured source contractions remain mathematically valid."""

    def required_points(
            self,
            view: EvaluationView,
            context: Any = None) -> Optional[ConfigurationBatch]:
        """Declares closure points required before freezing the plan."""

    def apply(
            self,
            view: EvaluationView,
            context: Any = None) -> torch.Tensor:
        """Transforms the unique value table without changing its shape."""


@dataclass(frozen=True)
class IdentityGlobalValueTransform:
    """No-op global transform that returns the original value tensor."""

    @property
    def is_identity(self) -> bool:
        return True

    def required_points(
            self,
            view: EvaluationView,
            context: Any = None) -> Optional[ConfigurationBatch]:
        return None

    def apply(
            self,
            view: EvaluationView,
            context: Any = None) -> torch.Tensor:
        if view.values is None:
            raise ValueError('The evaluation view should contain values')
        return view.values


@dataclass(frozen=True)
class CallableGlobalValueTransform:
    """Adapts callables to the global value-transform protocol."""

    function: Callable[[EvaluationView, Any], torch.Tensor]
    required_points_function: Optional[
        Callable[[EvaluationView, Any], Optional[ConfigurationBatch]]] = None

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError('`function` should be callable')
        if self.required_points_function is not None and \
                not callable(self.required_points_function):
            raise TypeError('`required_points_function` should be callable')

    @property
    def is_identity(self) -> bool:
        return False

    def required_points(
            self,
            view: EvaluationView,
            context: Any = None) -> Optional[ConfigurationBatch]:
        if self.required_points_function is None:
            return None
        result = self.required_points_function(view, context)
        if result is not None and not isinstance(result, ConfigurationBatch):
            raise TypeError(
                'A global transform should request ConfigurationBatch points')
        return result

    def apply(
            self,
            view: EvaluationView,
            context: Any = None) -> torch.Tensor:
        return self.function(view, context)


@dataclass(frozen=True)
class CompositeGlobalValueTransform:
    """Applies several global transforms in their declared order."""

    transforms: Sequence[GlobalValueTransform]

    def __post_init__(self) -> None:
        transforms = tuple(self.transforms)
        if not all(isinstance(transform, GlobalValueTransform)
                   for transform in transforms):
            raise TypeError(
                '`transforms` should contain GlobalValueTransform objects')
        object.__setattr__(self, 'transforms', transforms)

    @property
    def is_identity(self) -> bool:
        return all(transform.is_identity for transform in self.transforms)

    def required_points(
            self,
            view: EvaluationView,
            context: Any = None) -> Optional[ConfigurationBatch]:
        requests = [
            request
            for transform in self.transforms
            if (request := transform.required_points(view, context)) is not None
        ]
        return _concatenate_configurations(requests) if requests else None

    def apply(
            self,
            view: EvaluationView,
            context: Any = None) -> torch.Tensor:
        if view.values is None:
            raise ValueError('The evaluation view should contain values')
        values = view.values
        for transform in self.transforms:
            values = transform.apply(
                EvaluationView(
                    configurations=view.configurations,
                    values=values,
                    incidences=view.incidences,
                    phase=view.phase),
                context)
        return values


@dataclass(frozen=True)
class LocalTransformContext:
    """Immutable local callback context with predeclared query results."""

    data: Any = None
    evaluation: Optional[EvaluationView] = None
    query_results: Sequence[torch.Tensor] = ()

    def __post_init__(self) -> None:
        if self.evaluation is not None and \
                not isinstance(self.evaluation, EvaluationView):
            raise TypeError('`evaluation` should be EvaluationView type or None')
        query_results = tuple(self.query_results)
        if not all(isinstance(result, torch.Tensor)
                   for result in query_results):
            raise TypeError('`query_results` should contain torch.Tensor objects')
        object.__setattr__(self, 'query_results', query_results)


@runtime_checkable
class LocalValueTransform(Protocol):
    """Transform applied independently to one Phi view after global scatter."""

    @property
    def is_identity(self) -> bool:
        """Whether applying this transform returns the same Phi view."""

    def required_queries(
            self,
            phi_view: PhiView,
            context: LocalTransformContext) -> Sequence[torch.Tensor]:
        """Declares additional same-Phi selections before plan freeze."""

    def apply(
            self,
            phi_view: PhiView,
            context: LocalTransformContext) -> PhiView:
        """Returns a transformed lazy or materialized Phi view."""


@dataclass(frozen=True)
class IdentityLocalValueTransform:
    """No-op local transform preserving the exact Phi view object."""

    @property
    def is_identity(self) -> bool:
        return True

    def required_queries(
            self,
            phi_view: PhiView,
            context: LocalTransformContext) -> Sequence[torch.Tensor]:
        return ()

    def apply(
            self,
            phi_view: PhiView,
            context: LocalTransformContext) -> PhiView:
        return phi_view


@dataclass(frozen=True)
class CallableLocalValueTransform:
    """Adapts local Phi callbacks and optional query declarations."""

    function: Callable[[PhiView, LocalTransformContext], PhiView]
    required_queries_function: Optional[
        Callable[[PhiView, LocalTransformContext],
                 Sequence[torch.Tensor]]] = None

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError('`function` should be callable')
        if self.required_queries_function is not None and \
                not callable(self.required_queries_function):
            raise TypeError('`required_queries_function` should be callable')

    @property
    def is_identity(self) -> bool:
        return False

    def required_queries(
            self,
            phi_view: PhiView,
            context: LocalTransformContext) -> Sequence[torch.Tensor]:
        if self.required_queries_function is None:
            return ()
        queries = tuple(self.required_queries_function(phi_view, context))
        if not all(isinstance(query, torch.Tensor) for query in queries):
            raise TypeError(
                'A local transform should request index-selection tensors')
        return queries

    def apply(
            self,
            phi_view: PhiView,
            context: LocalTransformContext) -> PhiView:
        result = self.function(phi_view, context)
        if not isinstance(result, PhiView):
            raise TypeError('A local value transform should return a PhiView')
        return result


@dataclass(frozen=True)
class CompositeLocalValueTransform:
    """Applies several local transforms in their declared order."""

    transforms: Sequence[LocalValueTransform]

    def __post_init__(self) -> None:
        transforms = tuple(self.transforms)
        if not all(isinstance(transform, LocalValueTransform)
                   for transform in transforms):
            raise TypeError(
                '`transforms` should contain LocalValueTransform objects')
        object.__setattr__(self, 'transforms', transforms)

    @property
    def is_identity(self) -> bool:
        return all(transform.is_identity for transform in self.transforms)

    def required_queries(
            self,
            phi_view: PhiView,
            context: LocalTransformContext) -> Sequence[torch.Tensor]:
        return tuple(
            query
            for transform in self.transforms
            for query in transform.required_queries(phi_view, context))

    def apply(
            self,
            phi_view: PhiView,
            context: LocalTransformContext) -> PhiView:
        result = phi_view
        for transform in self.transforms:
            result = transform.apply(result, context)
        return result


def _prepare_global_transform(
        builder: _EvaluationPlanBuilder,
        transform: GlobalValueTransform,
        context: Any = None) -> Optional[int]:
    """Collects global closure points and marks the transform before freeze."""
    if not isinstance(builder, _EvaluationPlanBuilder):
        raise TypeError('`builder` should be _EvaluationPlanBuilder type')
    if not isinstance(transform, GlobalValueTransform):
        raise TypeError('`transform` should implement GlobalValueTransform')
    request = transform.required_points(builder.snapshot(), context)
    handle = None if request is None else builder.expand(request)
    builder._mark_global_transform_prepared()
    return handle


def _collect_local_queries(
        builder: _EvaluationPlanBuilder,
        phi: PhiOperator,
        transform: LocalValueTransform,
        context: Any = None) -> Tuple[int, ...]:
    """Collects every local query before global transform preparation/freeze."""
    if not isinstance(builder, _EvaluationPlanBuilder):
        raise TypeError('`builder` should be _EvaluationPlanBuilder type')
    if not isinstance(phi, PhiOperator):
        raise TypeError('`phi` should be PhiOperator type')
    if not isinstance(transform, LocalValueTransform):
        raise TypeError('`transform` should implement LocalValueTransform')
    local_context = LocalTransformContext(data=context)
    return tuple(
        phi.collect(builder, query)
        for query in transform.required_queries(phi, local_context))


def _apply_local_transform(
        transform: LocalValueTransform,
        phi_view: PhiView,
        *,
        data: Any = None,
        evaluation: Optional[EvaluationView] = None,
        query_results: Sequence[torch.Tensor] = ()) -> PhiView:
    """Applies one local transform with immutable pre-evaluated context."""
    if not isinstance(transform, LocalValueTransform):
        raise TypeError('`transform` should implement LocalValueTransform')
    if not isinstance(phi_view, PhiView):
        raise TypeError('`phi_view` should implement PhiView')
    context = LocalTransformContext(
        data=data,
        evaluation=evaluation,
        query_results=query_results)
    result = transform.apply(phi_view, context)
    if not isinstance(result, PhiView):
        raise TypeError('A local value transform should return a PhiView')
    return result


def _structured_path_allowed(transform: GlobalValueTransform) -> bool:
    """Returns whether a transform may bypass pointwise source evaluation."""
    if not isinstance(transform, GlobalValueTransform):
        raise TypeError('`transform` should implement GlobalValueTransform')
    return transform.is_identity


__all__ = [
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
