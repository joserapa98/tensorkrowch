"""Tests for tensor sources shared by ALS and sketching."""

from itertools import product

import pytest

import torch
import tensorkrowch as tk

from tests.decompositions.als._oracles import (contract_tt_dense,
                                               make_tt_cores)


class TestConfigurationBatch:  # MARK: TestConfigurationBatch

    def test_packed_discrete_metadata_and_selection(self):
        values = torch.tensor([[0, 1, 2], [1, 0, 3]])
        configurations = tk.decompositions.ConfigurationBatch(values)

        assert configurations.packed
        assert configurations.batch_size == 2
        assert configurations.n_sites == 3
        assert configurations.site_shape == ((), (), ())
        assert torch.equal(configurations.as_tensor(), values)
        assert torch.equal(
            configurations.index_select(torch.tensor([1])).as_tensor(),
            values[1:])

    def test_heterogeneous_coordinates_preserve_site_shapes(self):
        configurations = tk.decompositions.ConfigurationBatch(
            (
                torch.tensor([0.1, 0.2]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ),
            kind='coordinates')

        assert not configurations.packed
        assert configurations.site_shape == ((), (2,))
        with pytest.raises(ValueError, match='cannot be packed'):
            configurations.as_tensor()

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


class TestDenseAndCallableSources:  # MARK: TestDenseAndCallableSources

    def test_dense_scalar_evaluation_and_fiber(self):
        tensor = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        source = tk.decompositions.DenseTensorSource(tensor)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 1, 2], [1, 2, 3]]))

        assert torch.equal(source.evaluate(configurations),
                           torch.tensor([6., 23.], dtype=tensor.dtype))
        fiber = source.fiber(configurations, site=1)
        expected = torch.stack((tensor[0, :, 2], tensor[1, :, 3]))
        assert torch.equal(fiber, expected)

    def test_dense_tensor_output(self):
        tensor = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
        source = tk.decompositions.DenseTensorSource(
            tensor, input_dim=(2, 3))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 1], [1, 2]]))

        assert source.output_shape == (4,)
        assert torch.equal(source.evaluate(configurations),
                           torch.stack((tensor[0, 1], tensor[1, 2])))
        assert source.fiber(configurations, site=0).shape == (2, 2, 4)

    def test_callable_batches_are_contiguous_and_deterministic(self):
        batch_sizes = []

        def function(configurations):
            batch_sizes.append(configurations.shape[0])
            return configurations.sum(dim=1).to(torch.float64)

        source = tk.decompositions.CallableTensorSource(
            function,
            input_dim=(2, 3, 4),
            dtype=torch.float64,
            batch_size=2)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([
                [0, 0, 0],
                [1, 0, 0],
                [1, 2, 0],
                [1, 2, 3],
                [0, 1, 2],
            ]))

        values = source.evaluate(configurations)

        assert batch_sizes == [2, 2, 1]
        assert torch.equal(values, torch.tensor(
            [0., 1., 3., 6., 3.], dtype=torch.float64))

    def test_callable_infers_tensor_output_and_dtype(self):
        source = tk.decompositions.CallableTensorSource(
            lambda x: torch.stack((x[:, 0], x[:, 1]), dim=1).to(
                torch.complex128),
            input_dim=(2, 3),
            output_shape=None,
            dtype=None)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 1], [1, 2]]))

        result = source.evaluate(configurations)

        assert result.shape == (2, 2)
        assert source.output_shape == (2,)
        assert source.dtype == torch.complex128

    def test_callable_receives_heterogeneous_coordinates(self):
        def function(values):
            x, vector = values
            return x + vector.sum(dim=1)

        source = tk.decompositions.CallableTensorSource(
            function,
            input_dim=(1, 1),
            dtype=torch.float64)
        configurations = tk.decompositions.ConfigurationBatch(
            (
                torch.tensor([1., 2.], dtype=torch.float64),
                torch.tensor([[3., 4.], [5., 6.]], dtype=torch.float64),
            ),
            kind='coordinates')

        assert torch.equal(
            source.evaluate(configurations),
            torch.tensor([8., 13.], dtype=torch.float64))

    def test_callable_discrete_fiber(self):
        source = tk.decompositions.CallableTensorSource(
            lambda x: (x[:, 0] + 2 * x[:, 1]).to(torch.float64),
            input_dim=(2, 3),
            dtype=torch.float64)
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 1], [1, 2]]))

        assert torch.equal(
            source.fiber(configurations, site=1),
            torch.tensor([[0., 2., 4.], [1., 3., 5.]],
                         dtype=torch.float64))

    def test_as_tensor_source_normalizes_supported_inputs(self):
        tensor = torch.randn(2, 3)
        dense = tk.decompositions.as_tensor_source(tensor)
        callable_source = tk.decompositions.as_tensor_source(
            lambda x: x.sum(dim=1).float(), input_dim=(2, 3))

        assert isinstance(dense, tk.decompositions.DenseTensorSource)
        assert isinstance(callable_source,
                          tk.decompositions.CallableTensorSource)
        assert tk.decompositions.as_tensor_source(dense) is dense
        with pytest.raises(ValueError, match='input_dim'):
            tk.decompositions.as_tensor_source(lambda x: x)


class TestSparseAndTTSources:  # MARK: TestSparseAndTTSources

    def test_sparse_source_coalesces_and_declares_missing_zeros(self):
        source = tk.decompositions.SparseTensorSource(
            indices=torch.tensor([[0, 0], [0, 0], [1, 2]]),
            values=torch.tensor([1., 2., 4.]),
            input_dim=(2, 3))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [0, 2], [1, 2]]))

        assert torch.equal(source.support.as_tensor(),
                           torch.tensor([[0, 0], [1, 2]]))
        assert torch.equal(source.support_values, torch.tensor([3., 4.]))
        assert torch.equal(source.evaluate(configurations),
                           torch.tensor([3., 0., 4.]))
        base = tk.decompositions.ConfigurationBatch(torch.tensor([[0, 0]]))
        assert torch.equal(source.fiber(base, site=1),
                           torch.tensor([[3., 0., 0.]]))

    def test_empirical_distribution_accumulates_normalized_mass(self):
        distribution = tk.decompositions.EmpiricalDistribution(
            torch.tensor([[0, 1], [0, 1], [1, 0], [1, 1]]),
            input_dim=(2, 2))
        configurations = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))

        assert torch.equal(
            distribution.evaluate(configurations),
            torch.tensor([0., 0.5, 0.25, 0.25]))
        assert distribution.support_values.sum() == 1

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_tt_source_evaluation_and_fibers_match_dense(self, dtype):
        cores = make_tt_cores(
            dtype=dtype, generator=torch.Generator().manual_seed(11))
        source = tk.decompositions.TTTensorSource(cores)
        dense = contract_tt_dense(cores)
        configurations = tk.decompositions.ConfigurationBatch(torch.tensor(
            list(product(*(range(dim) for dim in dense.shape)))))

        assert torch.allclose(
            source.evaluate(configurations), dense.reshape(-1))

        base = tk.decompositions.ConfigurationBatch(
            torch.tensor([[0, 0, 0], [1, 2, 1]]))
        expected = torch.stack((dense[0, :, 0], dense[1, :, 1]))
        assert torch.allclose(source.fiber(base, site=1), expected)

    def test_tt_decomposition_normalizes_to_tt_source(self):
        cores = make_tt_cores(generator=torch.Generator().manual_seed(12))
        result_cores = [cores[0].squeeze(0), cores[1], cores[2].squeeze(-1)]
        decomposition = tk.decompositions.TTDecomposition(result_cores)

        source = tk.decompositions.as_tensor_source(decomposition)

        assert isinstance(source, tk.decompositions.TTTensorSource)
        assert source.input_dim == decomposition.input_dim
