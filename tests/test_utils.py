"""
This script contains tests for utils:

    * TestTruncatedSVD
"""

import pytest

import torch
import tensorkrowch as tk


class TestTruncatedSVD:  # MARK: TestTruncatedSVD

    @pytest.fixture
    def diag_tensor(self):
        # Singular values are exactly [5.0, 3.0, 1.0, 0.1]
        return torch.diag(torch.tensor([5.0, 3.0, 1.0, 0.1]))

    @pytest.mark.parametrize(
        'kwargs, expected_rank',
        [
            ({}, 4),
            ({'rank': 2}, 2),
            ({'cutoff': 1.0}, 2),
            ({'tol': 1.0}, 3),
            ({'rtol': 0.2}, 2),
            ({'cum_percentage': 0.8}, 2),
            ({'rank': 3, 'cutoff': 1.0, 'tol': 1.0}, 2),
            ({'rank': 4, 'rtol': 0.2, 'cum_percentage': 0.95}, 2),
        ],
    )
    def test_truncated_svd_rank_selection(self, diag_tensor, kwargs, expected_rank):
        u, s, vh = tk.utils.truncated_svd(diag_tensor, **kwargs)

        assert u.shape == (4, expected_rank)
        assert s.shape == (expected_rank,)
        assert vh.shape == (expected_rank, 4)
        assert torch.equal(s, torch.tensor([5.0, 3.0, 1.0, 0.1])[:expected_rank])

    def test_truncated_svd_batched_rank_selection(self):
        tensor = torch.stack([
            torch.diag(torch.tensor([5.0, 3.0, 1.0, 0.1])),
            torch.diag(torch.tensor([4.0, 2.0, 0.5, 0.05])),
        ])

        u, s, vh = tk.utils.truncated_svd(tensor, cutoff=0.5, tol=1.0)

        # cutoff gives rank 3 (because one batch has 1.0), tol gives rank 3 -> final rank 3
        assert u.shape == (2, 4, 3)
        assert s.shape == (2, 3)
        assert vh.shape == (2, 3, 4)

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'rank': 0},
            {'rank': 1.5},
            {'cutoff': -1.0},
            {'cutoff': '1'},
            {'tol': -0.1},
            {'rtol': -0.1},
            {'rtol': 1.1},
            {'cum_percentage': -0.1},
            {'cum_percentage': 1.1},
        ],
    )
    def test_truncated_svd_invalid_arguments(self, diag_tensor, kwargs):
        with pytest.raises(ValueError):
            tk.utils.truncated_svd(diag_tensor, **kwargs)
