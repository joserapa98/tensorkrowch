"""Tests for _truncation."""

import math

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions._truncation import _TruncationSpec


@pytest.mark.parametrize('kwargs, error', [
    ({'rank': 0}, ValueError), ({'rank': True}, TypeError),
    ({'rank': [2, 0]}, ValueError), ({'rank': [2, True]}, TypeError),
    ({'rank': '2'}, TypeError), ({'rank': {1, 2}}, TypeError),
    ({'cutoff': -1}, ValueError), ({'atol': -1}, ValueError),
    ({'rtol': 2}, ValueError), ({'cum_percentage': 2}, ValueError),
])
def test_truncation_rejects_invalid_criteria(kwargs, error):
    with pytest.raises(error):
        _TruncationSpec(**kwargs)


def test_truncation_criteria_follow_shared_svd():
    matrix = torch.diag(torch.tensor([4., 2., 1., 0.1], dtype=torch.float64))
    spec = _TruncationSpec(rank=2, cutoff=0.05)
    u, s, vh = tk.utils.truncated_svd(matrix, **spec.as_kwargs())
    assert s.shape == (2,)
    torch.testing.assert_close(u @ torch.diag(s) @ vh, torch.diag(torch.tensor([4., 2., 0., 0.], dtype=matrix.dtype)))


@pytest.mark.parametrize('log_scale', [math.log(10), -1000., 1000.])
def test_normalized_absolute_tolerances_keep_relative_criteria(log_scale):
    spec = _TruncationSpec(rank=2, cutoff=1., atol=4., rtol=0.01)
    values = spec.as_normalized_kwargs(log_scale)
    assert values['rank'] == 2
    assert values['rtol'] == 0.01
    if log_scale == -1000:
        assert values['cutoff'] == values['atol'] == float('inf')
    elif log_scale == 1000:
        assert values['cutoff'] == values['atol'] == 0
    else:
        assert values['cutoff'] == pytest.approx(0.1)
        assert values['atol'] == pytest.approx(0.04)


def test_rank_profile_is_frozen_and_resolved_per_cut():
    rank = [2, 3, 1]
    spec = _TruncationSpec(rank=rank, cutoff=0.05, rtol=0.01)
    rank[0] = 9
    spec.validate_rank(3)
    assert spec.rank == (2, 3, 1)
    assert spec.rank_at(-1) == 1
    matrix = torch.diag(torch.tensor([4., 2., 1., 0.1], dtype=torch.float64))
    for site, cap in enumerate((2, 3, 1)):
        _, values, _ = tk.utils.truncated_svd(
            matrix, **spec.as_kwargs(site=site))
        assert values.numel() <= cap
    with pytest.raises(ValueError, match='contain 2 values'):
        spec.validate_rank(2)
    with pytest.raises(ValueError, match='requires `site`'):
        spec.as_kwargs()


@pytest.mark.parametrize('rank', [None, 3, [2, 3, 1, 4]])
def test_subchain_policy_preserves_tolerances(rank):
    spec = _TruncationSpec(rank=rank, cutoff=1., atol=4., rtol=0.01)
    subchain = spec.for_links(1, -1)
    assert subchain.rank == ((3, 1) if isinstance(rank, list) else rank)
    values = subchain.as_normalized_kwargs(math.log(10), site=0)
    assert values['rank'] == (3 if isinstance(rank, list) else rank)
    assert values['cutoff'] == pytest.approx(0.1)
    assert values['atol'] == pytest.approx(0.04)
    assert values['rtol'] == 0.01


def test_empty_rank_profile_represents_a_chain_without_links():
    spec = _TruncationSpec(rank=[])
    spec.validate_rank(0)
    assert spec.rank == ()
    with pytest.raises(ValueError, match='contain 1 values'):
        spec.validate_rank(1)
