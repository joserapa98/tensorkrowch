"""Alternating least-squares decomposition infrastructure."""

from tensorkrowch.decompositions.als.problem import (ALSProblem,
                                                     ObservedEntries)
from tensorkrowch.decompositions.als.solvers import LeastSquaresSolver


__all__ = ['ALSProblem', 'ObservedEntries', 'LeastSquaresSolver']
