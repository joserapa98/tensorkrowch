"""Tests for ring/init."""

import torch

import tensorkrowch as tk


def test_ring_public_conversion_preserves_format_values():
    format = tk.formats.TT([torch.ones(2, 1), torch.ones(1, 2, 1), torch.ones(1, 2)])
    result = tk.decompositions.TT2TR(format).fit(rank=1, tr_rank=1)
    assert isinstance(result, tk.formats.TR)
    torch.testing.assert_close(result.contract_dense(), format.contract_dense())
