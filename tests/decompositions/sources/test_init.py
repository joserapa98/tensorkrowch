"""Tests for sources/init."""

import torch

import tensorkrowch as tk


def test_source_factory_uses_base_source_and_preserves_runtime():
    tensor = torch.arange(8, dtype=torch.float64).reshape(2, 2, 2)
    source = tk.decompositions.as_tensor_source(tensor)
    assert isinstance(source, tk.decompositions.BaseSource)
    assert source.dtype == tensor.dtype and source.device == tensor.device
