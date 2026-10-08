"""Tests for sketching/quantization."""


import tensorkrowch as tk

from tensorkrowch.decompositions.sketching import quantization


def test_quantization_uses_shared_layouts_maps_and_source_adapter():
    for name in ('QuantizedLayout', 'CoordinateMap', 'AffineCoordinateMap',
                 'FunctionalCoordinateMap', 'ExplicitGridMap'):
        assert getattr(quantization, name) is getattr(tk.formats, name)
    assert quantization.QuantizedSourceAdapter is tk.decompositions.QuantizedSourceAdapter
