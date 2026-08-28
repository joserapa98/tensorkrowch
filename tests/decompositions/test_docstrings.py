"""Tests for shared decomposition docstring contracts."""

import inspect
import re

import pytest

import tensorkrowch as tk

from tensorkrowch.utils import truncated_svd


def _parameter_documentation(obj, parameter):
    """Extracts one complete parameter entry from a docstring."""
    docstring = inspect.getdoc(obj)
    match = re.search(
        rf'^{parameter} :[^\n]*\n.*?(?=^[A-Za-z_]\w* :|^Returns\n)',
        docstring,
        flags=re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f'Could not find `{parameter}` in the docstring')
    return match.group(0)


@pytest.mark.parametrize(
    'obj',
    [
        tk.decompositions.TTSVD.fit,
        tk.decompositions.tt_svd,
        tk.decompositions.TTMSVD.fit,
        tk.decompositions.ttm_svd,
        tk.decompositions.TRSVD.fit,
        tk.decompositions.tr_svd,
        tk.decompositions.vec_to_mps,
        tk.decompositions.mat_to_mpo,
        tk.decompositions.TTRSS.fit,
        tk.decompositions.tt_rss,
    ],
)
@pytest.mark.parametrize(
    'parameter',
    ['rank', 'cutoff', 'atol', 'rtol', 'cum_percentage'],
)
def test_truncation_parameter_documentation_is_canonical(obj, parameter):
    reference = _parameter_documentation(truncated_svd, parameter)
    assert _parameter_documentation(obj, parameter) == reference
