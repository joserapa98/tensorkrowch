"""Tests for _truncation."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions._truncation import _TruncationSpec


@pytest.mark.parametrize('kwargs, error', [
    ({'rank': 0}, ValueError), ({'rank': True}, TypeError),
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
