"""Tests for sources/test_factory.py."""


import pytest
import torch

import tensorkrowch as tk

from tests.decompositions.als._oracles import make_tt_cores


def test_as_tensor_source_normalizes_supported_inputs():
    tensor = torch.randn(2, 3)
    dense = tk.decompositions.as_tensor_source(tensor)
    dense_with_ignored_dim = tk.decompositions.as_tensor_source(tensor)
    assert dense_with_ignored_dim.in_sites == (0, 1)
    assert dense_with_ignored_dim.in_dim == (2, 3)
    assert dense_with_ignored_dim.out_shape == ()
    dense_with_selected_axis = tk.decompositions.DenseTensorSource(tensor, in_sites=(1,))
    assert dense_with_selected_axis.in_dim == (2, 3)
    assert dense_with_selected_axis.out_shape == (2,)
    callable_source = tk.decompositions.CallableTensorSource(lambda x: x.sum(dim=1).float(), in_dim=(2, 3))

    assert isinstance(dense, tk.decompositions.DenseTensorSource)
    assert isinstance(callable_source,
                      tk.decompositions.CallableTensorSource)
    assert tk.decompositions.as_tensor_source(dense) is dense
    with pytest.raises(TypeError, match='source'):
        tk.decompositions.as_tensor_source(lambda x: x)
    selected = tk.decompositions.CallableTensorSource(
        lambda x: x[:, 0].float(), in_dim=(2,), in_sites=(0,))
    assert selected.in_sites == (0,)
    assert selected.n_sites == 1


def test_tt_decomposition_normalizes_to_tt_source():
    cores = make_tt_cores(generator=torch.Generator().manual_seed(12))
    result_cores = [cores[0].squeeze(0), cores[1], cores[2].squeeze(-1)]
    decomposition = tk.decompositions.TTDecomposition(result_cores)

    source = tk.decompositions.as_tensor_source(decomposition)

    assert isinstance(source, tk.decompositions.FormatSource)
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
    with pytest.raises(TypeError, match='in_sites|torch.Tensor'):
        tk.decompositions.DenseTensorSource(format, in_sites=(0,))


@pytest.mark.parametrize('boundary', ['obc', 'pbc'])
def test_factory_accepts_mpo(boundary):
    mpo = tk.models.MPO(n_features=2, in_dim=2, out_dim=3,
                        bond_dim=2, boundary=boundary)
    source = tk.decompositions.as_tensor_source(mpo)
    expected = mpo.to_ttm() if boundary == 'obc' else mpo.to_trm()
    assert isinstance(source, tk.decompositions.FormatSource)
    assert source.is_matrix
    assert source.in_dim == (2, 2)
    assert source.out_dim == (3, 3)
    torch.testing.assert_close(source.contract_dense(), expected.contract_dense())


@pytest.mark.parametrize('kind', ['dense', 'sparse'])
def test_selected_inputs_keep_all_variables_in_svd(kind):
    tensor = torch.arange(120., dtype=torch.float64).reshape(2, 3, 4, 5)
    if kind == 'dense':
        source = tk.decompositions.DenseTensorSource(tensor, in_sites=(2, 0))
    else:
        indices = torch.cartesian_prod(torch.arange(4), torch.arange(2))
        source = tk.decompositions.SparseTensorSource(
            indices, tensor.permute(2, 0, 1, 3).reshape(8, 3, 5),
            tensor.shape, in_sites=(2, 0))
    assert source.n_sites == 4
    assert source.in_sites == (2, 0)
    result = tk.decompositions.TTSVD(source).fit()
    assert result.n_sites == 4
    assert result.in_dim == tensor.shape
    torch.testing.assert_close(result.contract_dense(), tensor)
