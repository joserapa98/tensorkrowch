"""Tests for sources/base."""


import pytest
import torch

import tensorkrowch as tk


class TestConfigurationBatch:  # MARK: TestConfigurationBatch

    def test_packed_discrete_metadata_and_selection(self):
        values = torch.tensor([[0, 1, 2], [1, 0, 3]])
        configurations = tk.decompositions.ConfigurationBatch(values)

        assert configurations.packed
        assert configurations.batch_size == 2
        assert configurations.n_inputs == 3
        assert not hasattr(configurations, 'n_sites')
        assert configurations.feature_shape is None
        assert torch.equal(configurations.as_tensor(), values)
        assert torch.equal(
            configurations.index_select(torch.tensor([1])).as_tensor(),
            values[1:])

    def test_heterogeneous_features_preserve_feature_shapes(self):
        configurations = tk.decompositions.ConfigurationBatch(
            (
                torch.tensor([0.1, 0.2]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ),
            kind='features')

        assert not configurations.packed
        assert configurations.n_inputs == 2
        assert configurations.index_select(torch.tensor([1])).n_inputs == 2
        assert configurations.feature_shape == ((), (2,))
        with pytest.raises(ValueError, match='cannot be packed'):
            configurations.as_tensor()

    def test_packed_feature_shapes(self):
        scalar = tk.decompositions.ConfigurationBatch(
            torch.randn(2, 3), kind='features')
        vector = tk.decompositions.ConfigurationBatch(
            torch.randn(2, 3, 4), kind='features')

        assert scalar.feature_shape == ((), (), ())
        assert vector.feature_shape == ((4,), (4,), (4,))
        assert scalar.n_inputs == vector.n_inputs == 3

    @pytest.mark.parametrize(
        'values, kind, error',
        [
            (torch.tensor([0, 1]), 'indices', ValueError),
            (torch.randn(2, 3), 'indices', TypeError),
            (torch.ones(2, 3, 2, dtype=torch.long), 'indices', TypeError),
            (torch.ones(2, 3), 'mixed', ValueError),
        ])
    def test_invalid_batches(self, values, kind, error):
        with pytest.raises(error):
            tk.decompositions.ConfigurationBatch(values, kind=kind)


@pytest.mark.parametrize('make_values', [lambda: iter((torch.ones(2),)),
                                        lambda: {0: torch.ones(2)}])
def test_configuration_batch_requires_a_tensor_or_sequence(make_values):
    with pytest.raises(TypeError, match='tensor or a sequence of tensors'):
        tk.decompositions.ConfigurationBatch(make_values(), kind='features')


def test_base_fiber_maps_global_variable_to_heterogeneous_input_column():
    def function(values):
        vector, scalar = values
        return vector.sum(-1, keepdim=True) + scalar[:, None] + torch.arange(3)

    source = tk.decompositions.CallableSource(
        function, n_sites=3, in_sites=(2, 0), out_shape=(3,))
    batch = tk.decompositions.ConfigurationBatch((
        torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([5., 6.])),
        kind='features')
    assert batch.n_inputs == len(source.in_sites) == 2
    assert source.n_sites == 3
    candidates = torch.tensor([[0., 1.], [2., 3.]])
    actual = source.fiber(batch, site=2, values=candidates)
    expected = torch.stack([function((candidate.expand(2, -1), batch.values[1]))
                            for candidate in candidates], dim=1)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match='input site'):
        source.fiber(batch, site=1, values=candidates)


class TestTensorView:  # MARK: TestTensorView

    @pytest.mark.parametrize('sites', [(2, 0), (2, 1, 0), (1,)])
    @pytest.mark.parametrize('n_batches', [0, 2])
    def test_vector_sites_evaluation_and_fibers(self, sites, n_batches):
        from tensorkrowch.decompositions.sources.base import _tensor_view

        shape = (2, 3, 2, 3, 4) if n_batches else (2, 3, 4)
        tensor = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
        source = tk.decompositions.DenseTensorSource(
            tensor, in_sites=sites, n_batches=n_batches)
        view = _tensor_view(source)

        assert view.n_sites == source.n_sites == 3
        assert view.in_sites == tuple(range(view.n_sites))
        assert view.in_dim == source.in_dim == (2, 3, 4)
        assert view.out_shape == ()
        assert view.out_dim is None
        assert not view.is_matrix
        assert view.n_batches == source.n_batches

        indices = torch.tensor([[1, 0, 3], [0, 2, 1]])
        batch = tk.decompositions.ConfigurationBatch(indices)
        assert batch.n_inputs == len(view.in_sites) == view.n_sites
        expected = tensor[..., indices[:, 0], indices[:, 1], indices[:, 2]]
        torch.testing.assert_close(
            view.evaluate(batch), expected.movedim(-1, 0))

        expected_fiber = torch.stack([
            tensor[..., i, :, k].movedim(-1, 0) for i, _, k in indices])
        torch.testing.assert_close(view.fiber(batch, site=1), expected_fiber)
        torch.testing.assert_close(view.contract_dense(), tensor)
        assert view.evaluation_stats == source.evaluation_stats

    def test_matrix_pairs_are_scalar_vector_sites(self):
        from tensorkrowch.decompositions.sources.base import _tensor_view

        tensor = torch.arange(120.).reshape(2, 3, 4, 5)
        source = tk.decompositions.DenseTensorSource.matrix(
            tensor, ordering='interleaved')
        view = _tensor_view(source)

        assert view.n_sites == source.n_sites == 2
        assert view.in_sites == (0, 1)
        assert view.in_dim == (6, 20)
        assert view.out_shape == ()
        assert view.out_dim is None
        assert not view.is_matrix

        indices = torch.tensor([[5, 19], [1, 7]])
        expected = tensor[indices[:, 0] // 3, indices[:, 0] % 3,
                          indices[:, 1] // 5, indices[:, 1] % 5]
        torch.testing.assert_close(view.evaluate(
            tk.decompositions.ConfigurationBatch(indices)), expected)
        torch.testing.assert_close(view.contract_dense(), tensor.reshape(6, 20))

    def test_scalar_source_needs_no_view(self):
        from tensorkrowch.decompositions.sources.base import _tensor_view

        source = tk.decompositions.DenseTensorSource(torch.ones(2, 3))
        assert _tensor_view(source) is source
