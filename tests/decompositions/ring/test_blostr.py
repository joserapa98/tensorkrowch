"""Spectral TR recovery, rank discovery and joint-projection tests."""

import pytest
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions.ring.blostr import (tr_blostr,
                                                   tr_blostr_svd)

from tests.decompositions.als._oracles import contract_tr_dense


def _tensor(ranks, dimensions, dtype=torch.complex128, seed=303):
    """Builds a generic exact ring with independently drawn cores."""
    generator = torch.Generator().manual_seed(seed)
    cores = [torch.randn(ranks[site - 1], dimension, ranks[site],
                         dtype=dtype, generator=generator)
             for site, dimension in enumerate(dimensions)]
    return contract_tr_dense(cores)


def _error(result, tensor):
    """Measures the physical reconstruction, including complex gauges."""
    return (result.contract_dense() - tensor).norm() / tensor.norm()


class TestTRBLOSTR:  # MARK: TestTRBLOSTR

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('n_sites', [3, 5])
    def test_exact_recovery(self, function, dtype, n_sites):
        tensor = _tensor([2] * n_sites, [4] * n_sites, dtype)
        result = function(
            tensor, rank=[2] * n_sites, n_attempts=16,
            collect_metrics=True,
            generator=torch.Generator().manual_seed(81))
        assert isinstance(result, tk.decompositions.TRDecomposition)
        assert result.rank == [2] * n_sites
        assert result.dtype == torch.complex128
        assert _error(result, tensor) < 1e-9
        assert result.metrics.errors[0].relative < 1e-9
        assert result.metrics.timings[0].elapsed >= 0
        if function is tr_blostr_svd:
            assert len(result.metrics.truncations) == n_sites - 2

    def test_svd_tolerance_returns_smaller_ranks_without_padding(self):
        tensor = _tensor([2, 1, 2, 3], [6, 3, 3, 6])
        result = tr_blostr_svd(
            tensor, rank=[2, 2, 2, 3], rtol=1e-12,
            projection='gaussian',
            generator=torch.Generator().manual_seed(81))
        assert result.rank == [2, 1, 2, 3]
        assert _error(result, tensor) < 1e-9

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    def test_rank_one_minimal_physical_dimensions(self, function):
        tensor = _tensor([1] * 3, [1] * 3)
        result = function(
            tensor, rank=1, projection='gaussian',
            generator=torch.Generator().manual_seed(81))
        assert result.rank == [1] * 3
        assert _error(result, tensor) < 1e-12

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    @pytest.mark.parametrize('projection', ['slices', 'gaussian'])
    def test_balanced_kmeans_discovers_uniform_ranks(
            self, function, projection):
        tensor = _tensor([2] * 4, [4] * 4)
        result = function(
            tensor, projection=projection,
            generator=torch.Generator().manual_seed(81))
        assert result.rank == [2] * 4
        assert result.metadata['adaptive'] is True
        assert _error(result, tensor) < 1e-9

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    def test_injective_matrix_unit_gauges(self, function):
        generator = torch.Generator().manual_seed(12)
        tr = tk.formats.TR([
            torch.randn(2, 4, 2, dtype=torch.complex128, generator=generator)
            for _ in range(5)])
        tensor = tr.to_tt().cores[2]
        result = function(
            tensor, rank=2, generator=torch.Generator().manual_seed(81))
        assert _error(result, tensor) < 1e-10

    def test_explicit_slices(self):
        tensor = _tensor([2] * 3, [4] * 3)
        result = tr_blostr_svd(
            tensor, rank=2, slices=((0,), (1,), (2,), (3,)),
            generator=torch.Generator().manual_seed(81))
        assert result.metadata['attempts'] == 1
        assert result.metadata['slices'] == ((0,), (1,), (2,), (3,))
        assert _error(result, tensor) < 1e-9

    def test_generator_reproduces_spectral_search(self):
        tensor = _tensor([2] * 4, [4] * 4)
        results = [tr_blostr_svd(
            tensor, rank=2,
            generator=torch.Generator().manual_seed(81)) for _ in range(2)]
        assert results[0].metadata['slices'] == results[1].metadata['slices']
        for first, second in zip(results[0].cores, results[1].cores):
            torch.testing.assert_close(first, second)

    @pytest.mark.parametrize('discover', [False, True])
    def test_gaussian_projection_crosses_rank_one_bottleneck(self, discover):
        ranks = [2, 1, 2, 3]
        tensor = _tensor(ranks, [6, 3, 3, 6])
        result = tr_blostr_svd(
            tensor, rank=None if discover else ranks, projection='gaussian',
            generator=torch.Generator().manual_seed(81))
        assert result.rank == ranks
        assert _error(result, tensor) < 1e-9

    def test_coordinate_slices_cannot_cross_rank_one_bottleneck(self):
        tensor = _tensor([2, 1, 2, 3], [6, 3, 3, 6])
        with pytest.raises(RuntimeError, match='too few nonzero eigenvalues'):
            tr_blostr_svd(tensor, rank=[2, 1, 2, 3], n_attempts=2,
                          generator=torch.Generator().manual_seed(81))

    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
    def test_low_precision_preserves_dtype_family(self, dtype):
        tensor = _tensor([1] * 3, [3] * 3, dtype)
        result = tr_blostr_svd(
            tensor, rank=1,
            generator=torch.Generator().manual_seed(81), out_device=None)
        assert result.dtype == torch.complex64
        assert _error(result, tensor) < 2e-5

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    @pytest.mark.parametrize('verbose', [1, 2, 3])
    def test_verbosity(self, function, verbose, capsys):
        tensor = _tensor([1] * 3, [3] * 3)
        function(tensor, rank=1, n_attempts=1, verbose=verbose,
                 generator=torch.Generator().manual_seed(81))
        output = capsys.readouterr().out
        assert 'BLOSTR' in output
        assert 'Summary' in output
        assert ('tensor(' in output) == (verbose == 3)

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    @pytest.mark.parametrize('target,error', [
        ([1, 2], TypeError),
        (torch.ones(2, 2), ValueError),
        (torch.ones(3, 3, 3, dtype=torch.long), TypeError),
        (torch.full((3, 3, 3), float('nan')), ValueError),
    ])
    def test_invalid_target(self, function, target, error):
        with pytest.raises(error):
            function(target, rank=1)

    @pytest.mark.parametrize('options,error', [
        ({'rank': True}, TypeError),
        ({'rank': 0}, ValueError),
        ({'rank': [1, 1]}, ValueError),
        ({'projection': 'random'}, ValueError),
        ({'projection': 'gaussian', 'slices': ((0,),) * 4}, ValueError),
        ({'slices': 'bad'}, TypeError),
        ({'slices': ((0,), (1,))}, ValueError),
        ({'slices': ((0,), (1,), (0,), (1.5,))}, TypeError),
        ({'slices': ((0,), (1,), (0,), (9,))}, ValueError),
        ({'slices': ((0,), (0,), (1,), (2,))}, ValueError),
        ({'spectral_atol': -1}, ValueError),
        ({'spectral_atol': True}, TypeError),
        ({'n_attempts': 0}, ValueError),
        ({'n_iters': 0}, ValueError),
        ({'n_restarts': 0}, ValueError),
        ({'generator': 1}, TypeError),
        ({'collect_metrics': 1}, TypeError),
        ({'rtol': 2}, ValueError),
    ])
    def test_invalid_options(self, options, error):
        tensor = _tensor([1] * 3, [3] * 3)
        with pytest.raises(error):
            tr_blostr_svd(tensor, **{'rank': 1, **options})

    def test_full_rejects_nonuniform_profile(self):
        tensor = _tensor([2, 1, 2, 3], [6, 3, 3, 6])
        with pytest.raises(ValueError, match='uniform'):
            tr_blostr(tensor, rank=[2, 1, 2, 3])

    def test_hybrid_rejects_different_projected_endpoint_ranks(self):
        tensor = torch.randn(6, 3, 3, 6, dtype=torch.complex128)
        with pytest.raises(ValueError, match=r'rank\[0\]'):
            tr_blostr_svd(tensor, rank=[2, 1, 3, 2])

    def test_zero_tensor_has_no_spectral_support(self):
        with pytest.raises(RuntimeError, match='nonzero eigenvalues'):
            tr_blostr_svd(torch.zeros(4, 4, 4), rank=1, n_attempts=1)
