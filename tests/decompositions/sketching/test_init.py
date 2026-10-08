"""Tests for sketching/init."""

import torch

import tensorkrowch as tk


def test_sketching_shared_public_infrastructure():
    assert tk.decompositions.TTRSS.quantized is not None
    assert tk.decompositions.TRRSS.quantized is not None
    layout = tk.formats.QuantizedLayout(1, 2, 3)
    source = tk.decompositions.QuantizedSourceAdapter(
        lambda coordinates: torch.ones_like(coordinates[:, 0]), layout)
    torch.testing.assert_close(source.evaluate(tk.decompositions.ConfigurationBatch(
        layout.encode_indices(torch.tensor([[0], [7]])))), torch.ones(2))
