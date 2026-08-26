"""Deprecated import facade for the historical SVD decomposition names."""

from tensorkrowch.decompositions.svd.tt import vec_to_mps
from tensorkrowch.decompositions.svd.ttm import mat_to_mpo


__all__ = ['vec_to_mps', 'mat_to_mpo']
