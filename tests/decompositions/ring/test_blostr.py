"""Spectral TR recovery, rank discovery and joint-projection tests."""

import pytest
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions.ring.blostr import (_recover_cores, tr_blostr,
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


class TestBLOSTR:  # MARK: TestBLOSTR

    @pytest.mark.parametrize('algorithm', [tk.decompositions.BLOSTR,
                                         tk.decompositions.BLOSTRSVD])
    @pytest.mark.parametrize('n_sites', [3, 5])
    @pytest.mark.parametrize('kind', ['tensor', 'dense', 'sparse', 'callable',
                                     'format', 'raw_format'])
    def test_finite_sources_and_function_equivalence(
            self, algorithm, n_sites, kind):
        tensor = _tensor([2] * n_sites, [4] * n_sites)
        if kind == 'tensor':
            source = tensor
        elif kind == 'dense':
            source = tk.decompositions.DenseTensorSource(tensor)
        elif kind == 'sparse':
            indices = torch.cartesian_prod(
                *(torch.arange(size) for size in tensor.shape))
            source = tk.decompositions.SparseTensorSource(
                indices, tensor.flatten(), tensor.shape)
        elif kind == 'callable':
            source = tk.decompositions.CallableTensorSource(
                lambda indices: tensor[tuple(indices.T)],
                in_dim=tensor.shape, out_shape=(), dtype=tensor.dtype,
                batch_size=37)
        else:
            format = tk.decompositions.TTSVD(tensor).fit().to_format()
            source = tk.decompositions.FormatSource(format) \
                if kind == 'format' else format
        engine = algorithm(source, out_device=None)
        result = engine.fit(
            rank=2, generator=torch.Generator().manual_seed(81),
            collect_metrics=True)
        function = tr_blostr if algorithm is tk.decompositions.BLOSTR \
            else tr_blostr_svd
        direct = function(source, rank=2, out_device=None,
                          generator=torch.Generator().manual_seed(81))

        assert isinstance(engine.source, tk.decompositions.BaseSource)
        if isinstance(source, tk.decompositions.BaseSource):
            assert engine.source is source
        assert result.device == tensor.device
        assert result.rank == [2] * n_sites
        assert _error(result, tensor) < 1e-9
        for core, expected in zip(result.cores, direct.cores):
            torch.testing.assert_close(core, expected)

    @pytest.mark.parametrize('algorithm', [tk.decompositions.BLOSTR,
                                         tk.decompositions.BLOSTRSVD])
    def test_materializes_once_and_repeats_fits(self, algorithm, monkeypatch):
        tensor = _tensor([2] * 3, [4] * 3)
        before = tensor.clone()
        source = tk.decompositions.DenseTensorSource(tensor)
        materializations = []
        contract_dense = source.contract_dense

        def materialize():
            materializations.append(True)
            return contract_dense()

        monkeypatch.setattr(source, 'contract_dense', materialize)
        engine = algorithm(source)
        results = [engine.fit(
            rank=rank, projection=projection, collect_metrics=True,
            generator=torch.Generator().manual_seed(81))
            for rank, projection in ((2, 'slices'), (None, 'gaussian'))]
        assert len(materializations) == 1
        assert results[0].metrics is not results[1].metrics
        for result in results:
            assert result.rank == [2] * 3
            assert _error(result, tensor) < 1e-9
            assert len(result.metrics.errors) == 1
        torch.testing.assert_close(tensor, before, atol=0, rtol=0)

    @pytest.mark.parametrize('algorithm', [tk.decompositions.BLOSTR,
                                         tk.decompositions.BLOSTRSVD])
    def test_minimal_rank_one(self, algorithm):
        result = algorithm(torch.ones(1, 1, 1)).fit(
            rank=1, projection='gaussian',
            generator=torch.Generator().manual_seed(81))
        assert result.rank == [1] * 3
        torch.testing.assert_close(result.contract_dense().real,
                                   torch.ones(1, 1, 1))

    @pytest.mark.parametrize('algorithm', [tk.decompositions.BLOSTR,
                                         tk.decompositions.BLOSTRSVD])
    def test_matrix_source_restores_paired_axes(self, algorithm):
        tensor = _tensor([2] * 3, [4] * 3).reshape((2, 2) * 3)
        source = tk.decompositions.DenseTensorSource.matrix(
            tensor, ordering='interleaved')
        result = algorithm(source).fit(
            rank=2, projection='gaussian',
            generator=torch.Generator().manual_seed(81))
        assert isinstance(result, tk.decompositions.TRMDecomposition)
        assert result.in_dim == (2,) * 3
        assert result.out_dim == (2,) * 3
        assert _error(result, tensor) < 1e-9

    @pytest.mark.parametrize('algorithm', [tk.decompositions.BLOSTR,
                                         tk.decompositions.BLOSTRSVD])
    def test_quantics_source_preserves_coordinates(self, algorithm):
        source = tk.decompositions.CallableQuanticsSource(
            lambda coordinates: torch.exp(
                coordinates.to(torch.float64).sum(-1)), 1,
            base=2, level=3, domain=(0., 1.), dtype=torch.float64)
        result = algorithm(source).fit(
            rank=1, projection='gaussian',
            generator=torch.Generator().manual_seed(81))
        assert isinstance(result, tk.decompositions.QTRDecomposition)
        assert result.layout == source.layout
        assert result.coordinate_map == source.coordinate_map
        assert _error(result, source.contract_dense()) < 1e-9

    @pytest.mark.parametrize('algorithm', [tk.decompositions.BLOSTR,
                                         tk.decompositions.BLOSTRSVD])
    def test_preserves_input_site_selection(self, algorithm):
        tensor = _tensor([2] * 3, [4] * 3)
        source = tk.decompositions.DenseTensorSource(tensor, in_sites=(2, 0))
        result = algorithm(source).fit(
            rank=2, generator=torch.Generator().manual_seed(81))
        assert result.in_sites == (2, 0)
        assert result.out_shape == (4,)
        assert _error(result, tensor) < 1e-9

    @pytest.mark.parametrize('algorithm', [tk.decompositions.BLOSTR,
                                         tk.decompositions.BLOSTRSVD])
    @pytest.mark.parametrize('kind', ['continuous', 'batched'])
    def test_unsupported_sources(self, algorithm, kind):
        if kind == 'continuous':
            source = tk.decompositions.CallableSource(
                lambda values: values.sum(-1), n_sites=3)
        else:
            source = tk.decompositions.DenseTensorSource(
                torch.ones(2, 3, 3, 3), n_batches=1)
        with pytest.raises(ValueError, match='discrete tensor|n_batches'):
            algorithm(source)

    @pytest.mark.parametrize('algorithm', [tk.decompositions.BLOSTR,
                                         tk.decompositions.BLOSTRSVD])
    @pytest.mark.parametrize('target,error', [
        ([1, 2], TypeError),
        (lambda values: values, TypeError),
        (torch.ones(3), ValueError),
        (torch.ones(2, 2), ValueError),
        (torch.ones(3, 3, 3, dtype=torch.long), TypeError),
        (torch.full((3, 3, 3), float('nan')), ValueError),
    ])
    def test_invalid_source(self, algorithm, target, error):
        with pytest.raises(error):
            algorithm(target)


class TestTRBLOSTR:  # MARK: TestTRBLOSTR

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('n_sites', [3, 5])
    @pytest.mark.parametrize('projection', ['slices', 'gaussian'])
    def test_exact_recovery(self, function, dtype, n_sites, projection):
        tensor = _tensor([2] * n_sites, [4] * n_sites, dtype)
        result = function(
            tensor, rank=[2] * n_sites, n_attempts=16, projection=projection,
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
    @pytest.mark.parametrize('projection', ['slices', 'gaussian'])
    def test_injective_matrix_unit_gauges(self, function, projection):
        generator = torch.Generator().manual_seed(12)
        tr = tk.formats.TR([
            torch.randn(2, 4, 2, dtype=torch.complex128, generator=generator)
            for _ in range(5)])
        tensor = tr.to_tt().cores[2]
        result = function(
            tensor, rank=2, projection=projection,
            generator=torch.Generator().manual_seed(81))
        # Matrix-unit coordinate slices are singular despite injectivity of
        # the complete unfolding. Only full slice recovery needs them.
        if (function is tr_blostr) and (projection == 'slices'):
            assert _error(result, tensor) > .1
        else:
            assert _error(result, tensor) < 1e-10

    @pytest.mark.parametrize('projection', ['slices', 'gaussian'])
    def test_recovery_projection_structure(self, projection, monkeypatch):
        generator = torch.Generator().manual_seed(303)
        cores = [torch.randn(2, 4, 2, dtype=torch.complex128,
                             generator=generator) for _ in range(6)]
        tensor = contract_tr_dense(cores)
        randn = torch.randn
        weights = []

        def draw(*args, **kwargs):
            weight = randn(*args, **kwargs)
            weights.append(weight)
            return weight

        monkeypatch.setattr(torch, 'randn', draw)
        recovered = _recover_cores(tensor, cores[0], projection, generator)
        torch.testing.assert_close(contract_tr_dense(recovered), tensor,
                                   rtol=1e-10, atol=1e-10)
        if projection == 'slices':
            assert not weights
        else:
            # At the middle step both groups have two physical axes. Dense
            # weights within either group cannot factor into local vectors.
            middle = [weight for weight in weights if weight.numel() == 16]
            assert len(middle) == 2
            assert all(torch.linalg.matrix_rank(weight.reshape(4, 4)) == 4
                       for weight in middle)

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
        with pytest.warns(UserWarning, match='use BLOSTR-SVD'):
            with pytest.raises(ValueError, match='uniform'):
                tr_blostr(tensor, rank=[2, 1, 2, 3])

    def test_full_warns_for_nonuniform_discovered_ranks(self):
        tensor = _tensor([2, 1, 2, 3], [6, 3, 3, 6])
        with pytest.warns(UserWarning, match='use BLOSTR-SVD'):
            with pytest.raises(RuntimeError, match='uniform discovered'):
                tr_blostr(
                    tensor, projection='gaussian', n_attempts=1,
                    generator=torch.Generator().manual_seed(81))

    @pytest.mark.parametrize('dimensions', [(3, 4, 4), (4, 3, 4), (4, 4, 3)])
    def test_full_warns_for_insufficient_physical_dimension(self, dimensions):
        tensor = _tensor([2] * 3, dimensions)
        with pytest.warns(UserWarning, match='Block the TT format'):
            with pytest.raises(RuntimeError, match=r'at least rank\*\*2'):
                tr_blostr(tensor, rank=2)

    def test_full_warns_for_discovered_injectivity_failure(self):
        tensor = _tensor([2] * 3, [4, 3, 4])
        with pytest.warns(UserWarning, match='Block the TT format'):
            with pytest.raises(RuntimeError, match=r'at least rank\*\*2'):
                tr_blostr(
                    tensor, projection='gaussian', n_attempts=1,
                    generator=torch.Generator().manual_seed(81))

    def test_hybrid_accepts_noninjective_interior_without_warning(
            self, recwarn):
        tensor = _tensor([2] * 3, [4, 3, 4])
        result = tr_blostr_svd(
            tensor, rank=2, projection='gaussian',
            generator=torch.Generator().manual_seed(81))
        assert _error(result, tensor) < 1e-9
        assert not recwarn

    def test_hybrid_rejects_different_projected_endpoint_ranks(self):
        tensor = torch.randn(6, 3, 3, 6, dtype=torch.complex128)
        with pytest.raises(ValueError, match=r'rank\[0\]'):
            tr_blostr_svd(tensor, rank=[2, 1, 3, 2])

    def test_zero_tensor_has_no_spectral_support(self):
        with pytest.raises(RuntimeError, match='nonzero eigenvalues'):
            tr_blostr_svd(torch.zeros(4, 4, 4), rank=1, n_attempts=1)
