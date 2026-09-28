"""Shared format kernels and result provenance contracts."""

from dataclasses import replace
import torch
import tensorkrowch as tk
from tensorkrowch.decompositions.sketching.quantization import QuantizedLayout
from tensorkrowch.decompositions.sources.quantization import QuantizedSourceAdapter


def test_results_inherit_formats_and_keep_historical_metrics():
    result = tk.decompositions.TTDecomposition(
        [torch.ones(2, 1), torch.ones(1, 3)], metadata={'fit': 'original'})
    assert isinstance(result, tk.formats.TensorTrain)
    replaced = replace(result, cores=[torch.zeros(2, 1), torch.ones(1, 3)])
    assert replaced.norm() == 0
    for transformed in [result.copy(), result.detach(), result.to(dtype=torch.float64)]:
        assert type(transformed) is type(result)
        assert transformed.metrics is result.metrics
        assert transformed.metadata == result.metadata
    assert type(result + result) is tk.formats.TensorTrain
    assert QuantizedLayout is tk.formats.QuantizedLayout
    assert QuantizedSourceAdapter is tk.decompositions.QuantizedSourceAdapter


def test_base_tt_source_absorbs_bonds_without_mutating_format():
    result = tk.formats.TensorTrain([torch.ones(2, 2), torch.ones(2, 3)])
    result.bonds = tk.formats.BondFactors([torch.tensor([2., 3.])])
    source = tk.decompositions.as_tensor_source(result)
    indices = torch.tensor([[0, 0], [1, 2]])
    actual = source.evaluate(tk.decompositions.ConfigurationBatch(indices, kind='indices'))
    assert torch.equal(actual, torch.full((2,), 5.))
    assert result.bonds is not None

