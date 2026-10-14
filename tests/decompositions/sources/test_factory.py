"""Tests for sources/test_factory.py."""


import pytest
import torch

import tensorkrowch as tk

from tests.decompositions.als._oracles import make_tt_cores


def test_as_tensor_source_normalizes_supported_inputs():
    tensor = torch.randn(2, 3)
    dense = tk.decompositions.as_tensor_source(tensor)
    dense_with_ignored_dim = tk.decompositions.as_tensor_source(
        tensor, in_dim=(2,))
    assert dense_with_ignored_dim.in_features == (0, 1)
    assert dense_with_ignored_dim.in_dim == (2, 3)
    assert dense_with_ignored_dim.out_shape == ()
    dense_with_selected_axis = tk.decompositions.as_tensor_source(
        tensor, in_features=(1,), in_dim=(2,))
    assert dense_with_selected_axis.in_dim == (3,)
    assert dense_with_selected_axis.out_shape == (2,)
    callable_source = tk.decompositions.as_tensor_source(
        lambda x: x.sum(dim=1).float(), in_dim=(2, 3))

    assert isinstance(dense, tk.decompositions.DenseTensorSource)
    assert isinstance(callable_source,
                      tk.decompositions.CallableTensorSource)
    assert tk.decompositions.as_tensor_source(dense) is dense
    with pytest.raises(ValueError, match='in_dim'):
        tk.decompositions.as_tensor_source(lambda x: x)
    with pytest.raises(TypeError, match='in_features'):
        tk.decompositions.as_tensor_source(
            lambda x: x, in_dim=(2,), in_features=(0,))


def test_tt_decomposition_normalizes_to_tt_source():
    cores = make_tt_cores(generator=torch.Generator().manual_seed(12))
    result_cores = [cores[0].squeeze(0), cores[1], cores[2].squeeze(-1)]
    decomposition = tk.decompositions.TTDecomposition(result_cores)

    source = tk.decompositions.as_tensor_source(decomposition)

    assert isinstance(source, tk.decompositions.TTTensorSource)
    assert source.in_dim == decomposition.in_dim


@pytest.mark.parametrize('invalid', [None, 'tt', 1, object()])
def test_factory_rejects_invalid_sources(invalid):
    with pytest.raises(TypeError, match='source'):
        tk.decompositions.as_tensor_source(invalid)


def test_factory_accepts_formats_and_preserves_sources():
    format = tk.formats.TT([torch.tensor([1., 2.])])
    source = tk.decompositions.as_tensor_source(format)
    assert tk.decompositions.as_tensor_source(source) is source
    assert torch.equal(source.evaluate(tk.decompositions.ConfigurationBatch(torch.arange(2).reshape(-1, 1))), format.contract_dense())
    with pytest.raises(TypeError, match='in_features'):
        tk.decompositions.as_tensor_source(format, in_features=(0,))
