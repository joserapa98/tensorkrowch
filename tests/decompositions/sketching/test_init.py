"""Tests for sketching/init."""

import torch

import tensorkrowch as tk


def test_sketching_shared_public_infrastructure():
    assert not hasattr(tk.decompositions.TTRSS, 'quantized')
    assert not hasattr(tk.decompositions.TRRSS, 'quantized')
    layout = tk.formats.QuantizedLayout(1, 2, 3)
    source = tk.decompositions.CallableQuanticsSource(
        lambda coordinates: torch.ones_like(coordinates[:, 0]), 1,
        layout=layout, coordinate_map=tk.formats.AffineCoordinateMap(
            [0., 1.], layout.grid_size))
    torch.testing.assert_close(source.evaluate(tk.decompositions.ConfigurationBatch(
        layout.encode_indices(torch.tensor([[0], [7]])))), torch.ones(2))
