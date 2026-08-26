"""Tests for shared decomposition docstring contracts."""

import inspect

import pytest

import tensorkrowch as tk

from tensorkrowch.utils import truncated_svd


def _truncation_parameters(obj, next_parameter):
    """Extracts the shared truncation-parameter section from a docstring."""
    docstring = inspect.getdoc(obj)
    start = docstring.index('rank : int, optional')
    end = docstring.index(f'\n{next_parameter} :', start)
    return docstring[start:end]


@pytest.mark.parametrize(
    'obj, next_parameter',
    [
        (tk.decompositions.TTSVD.fit, 'renormalize'),
        (tk.decompositions.tt_svd, 'renormalize'),
        (tk.decompositions.TTMSVD.fit, 'renormalize'),
        (tk.decompositions.ttm_svd, 'renormalize'),
        (tk.decompositions.vec_to_mps, 'renormalize'),
        (tk.decompositions.mat_to_mpo, 'renormalize'),
        (tk.decompositions.tt_rss, 'batch_size'),
    ],
)
def test_truncation_parameter_documentation_is_canonical(
        obj, next_parameter):
    reference = _truncation_parameters(truncated_svd, 'svd_method')
    assert _truncation_parameters(obj, next_parameter) == reference
