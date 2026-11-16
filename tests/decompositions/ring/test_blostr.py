"""Spectral TR recovery, rank discovery and joint-projection tests."""

from itertools import permutations

import pytest
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions._truncation import _TruncationSpec
from tensorkrowch.decompositions.ring.blostr import (_balanced_order,
                                                   _first_core,
                                                   _linear_assignment,
                                                   _ratio_eigen,
                                                   _recover_cores,
                                                   _spectral_order,
                                                   tr_blostr, tr_blostr_svd)

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


class TestLinearAssignment:  # MARK: TestLinearAssignment

    @pytest.mark.parametrize('size', [0, 1, 2, 4, 6])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
    def test_matches_exhaustive_minimum(self, size, dtype):
        generator = torch.Generator().manual_seed(815)
        cost = torch.randn(size, size, dtype=dtype, generator=generator)
        assignment = _linear_assignment(cost)
        rows = torch.arange(size)
        expected = min(float(cost[rows, list(order)].sum())
                       for order in permutations(range(size)))
        assert assignment.dtype == torch.long
        assert assignment.device == cost.device
        assert sorted(assignment.tolist()) == list(range(size))
        torch.testing.assert_close(cost[rows, assignment].sum(),
                                   cost.new_tensor(expected))

    @pytest.mark.parametrize('cost', [
        torch.zeros(4, 4), torch.full((4, 4), -2.),
        torch.tensor([[4., 1., 3.], [2., 0., 5.], [3., 2., 2.]]),
    ])
    def test_ties_and_deterministic_assignment(self, cost):
        first = _linear_assignment(cost)
        second = _linear_assignment(cost)
        assert torch.equal(first, second)
        expected = min(float(cost[torch.arange(len(cost)), list(order)].sum())
                       for order in permutations(range(len(cost))))
        assert float(cost[torch.arange(len(cost)), first].sum()) == expected

    def test_avoids_greedy_capacity_failure(self):
        points = torch.tensor([-100., 1., 2., 11.], dtype=torch.float64)
        centers = torch.tensor([0., 10.], dtype=torch.float64)
        cost = (points[:, None] - centers).square().repeat_interleave(2, 1)
        labels = _linear_assignment(cost) // 2
        assert labels.tolist() == [0, 0, 1, 1]
        optimum = (points - centers[labels]).square().sum()
        greedy = (points - centers[torch.tensor([1, 0, 0, 1])]).square().sum()
        assert optimum < greedy

    @pytest.mark.parametrize('cost', [
        torch.ones(2, 3), torch.ones(3),
        torch.full((2, 2), float('inf')),
        torch.full((2, 2), float('nan')),
    ])
    def test_invalid_cost(self, cost):
        with pytest.raises(ValueError, match='cost'):
            _linear_assignment(cost)


class TestBalancedOrder:  # MARK: TestBalancedOrder

    def test_recovers_separated_complex_groups(self):
        values = torch.tensor([0., 10., 20., .01, 10.01, 20.01],
                               dtype=torch.complex128) * (1 + 1j)
        order, cost = _balanced_order(
            values, 2, 3, 50, 10, torch.Generator().manual_seed(816))
        groups = values[order].reshape(3, 2)
        assert torch.all((groups[:, 0] - groups[:, 1]).abs() < .02)
        expected = (groups - groups.mean(1, keepdim=True)).abs().square().sum()
        torch.testing.assert_close(cost, expected)

    def test_separates_close_groups_with_large_common_offset(self):
        values = torch.tensor([1e8, 1e8 + .1, 1e8 + .2],
                               dtype=torch.complex128).repeat_interleave(10)
        permutation = torch.randperm(
            len(values), generator=torch.Generator().manual_seed(820))
        values = values[permutation]
        order, cost = _balanced_order(
            values, 10, 3, 50, 10, torch.Generator().manual_seed(821))
        groups = values[order].reshape(3, 10)
        assert torch.all(groups == groups[:, :1])
        assert cost < 1e-10

    def test_more_iterations_do_not_increase_clustering_cost(self):
        values = torch.randn(12, dtype=torch.complex128,
                              generator=torch.Generator().manual_seed(817))
        costs = [_balanced_order(
            values, 3, 4, count, 3, torch.Generator().manual_seed(818))[1]
                 for count in (1, 10)]
        assert costs[1] <= costs[0] + 1e-12



class TestRatioEigen:  # MARK: TestRatioEigen

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_matches_full_physical_truncated_ratio(self, dtype):
        rng = torch.Generator().manual_seed(609)
        matrices = [torch.randn(9, 7, dtype=dtype, generator=rng)
                    for _ in range(2)]
        factors = [torch.linalg.svd(matrix, full_matrices=False)
                   for matrix in matrices]
        size = 4
        values, vectors = _ratio_eigen(*factors, size)
        approximations = [(left[:, :size] * singular[:size]) @ right[:size]
                          for left, singular, right in factors]
        ratio = approximations[0] @ torch.linalg.pinv(approximations[1])
        torch.testing.assert_close(
            ratio.to(vectors.dtype) @ vectors, vectors * values)
        assert vectors.shape == (9, size)
        # Truncating only the denominator is a different noisy operator.
        untruncated = matrices[0] @ torch.linalg.pinv(approximations[1])
        assert (untruncated.to(vectors.dtype) @ vectors -
                vectors * values).norm() > .01


class TestSpectralOrder:  # MARK: TestSpectralOrder

    @pytest.mark.parametrize('scale,adjacent', [(1., 2), (100., 4)])
    def test_cluster_acceptance_is_absolute(self, scale, adjacent):
        values = scale * torch.tensor([1., 1.005, 2., 2.005],
                                       dtype=torch.complex128)
        _, cyclic, found_adjacent, _ = _spectral_order(
            values, None, .01, 50, 10,
            torch.Generator().manual_seed(81))
        assert found_adjacent == adjacent
        assert cyclic * found_adjacent == 4



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

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
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
        ({'cluster_atol': -1}, ValueError),
        ({'cluster_atol': True}, TypeError),
        ({'n_attempts': 0}, ValueError),
        ({'n_iters': 0}, ValueError),
        ({'n_restarts': 0}, ValueError),
        ({'generator': 1}, TypeError),
        ({'collect_metrics': 1}, TypeError),
        ({'rtol': 2}, ValueError),
    ])
    def test_invalid_options(self, function, options, error):
        tensor = _tensor([1] * 3, [3] * 3)
        with pytest.raises(error):
            function(tensor, **{'rank': 1, **options})

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

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('projection', ['slices', 'gaussian'])
    def test_noisy_first_core_uses_rank_limited_ratios(
            self, function, dtype, projection, monkeypatch):
        clean = _tensor([2] * 4, [8] * 4, dtype, seed=304) * 10000
        target = clean + .1 * torch.randn(
            clean.shape, dtype=dtype,
            generator=torch.Generator().manual_seed(19))
        eig = torch.linalg.eig
        shapes = []

        def diagonalize(matrix):
            shapes.append(matrix.shape)
            return eig(matrix)

        monkeypatch.setattr(torch.linalg, 'eig', diagonalize)
        options = dict(slices=((0, 0), (1, 1), (2, 2), (3, 3))) \
            if projection == 'slices' else dict(projection='gaussian')
        result = function(
            target, rank=2, n_attempts=1, **options,
            generator=torch.Generator().manual_seed(81))
        assert shapes == [(4, 4), (4, 4)]
        assert result.rank == [2] * 4
        assert torch.isfinite(result.contract_dense()).all()
        assert _error(result, clean) < .05
        if function is tr_blostr_svd:
            assert _error(result, clean) < .001

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_noisy_discovery_estimates_slice_rank_before_ratios(
            self, function, dtype, monkeypatch):
        clean = _tensor([2] * 4, [8] * 4, dtype, seed=304)
        target = clean + 1e-5 * torch.randn(
            clean.shape, dtype=dtype,
            generator=torch.Generator().manual_seed(19))
        eig = torch.linalg.eig
        shapes = []

        def diagonalize(matrix):
            shapes.append(matrix.shape)
            return eig(matrix)

        monkeypatch.setattr(torch.linalg, 'eig', diagonalize)
        result = function(
            target, cluster_atol=1e-3, rtol=1e-8,
            slices=((0, 0), (1, 1), (2, 2), (3, 3)),
            generator=torch.Generator().manual_seed(81))
        assert shapes == [(4, 4), (4, 4)]
        assert result.rank == [2] * 4
        assert _error(result, clean) < .05

    @pytest.mark.parametrize('options,expected_rank', [
        ({'cutoff': .01}, 3), ({'atol': .001}, 2),
        ({'rtol': 1e-4}, 2), ({'cum_percentage': .9999}, 2),
    ])
    @pytest.mark.parametrize('cluster_atol', [1e-10, 10.])
    def test_slice_discovery_uses_svd_controls_only(
            self, options, expected_rank, cluster_atol, monkeypatch):
        matrix = torch.diag(torch.tensor([4., 3., .02, .001],
                                         dtype=torch.complex128))
        tensor = matrix[:, None, :].expand(-1, 4, -1)

        def diagonalize(numerator, denominator, rank):
            assert rank == expected_rank
            raise RuntimeError('Stop before clustering')

        monkeypatch.setattr(
            'tensorkrowch.decompositions.ring.blostr._ratio_eigen',
            diagonalize)
        with pytest.raises(RuntimeError, match='Stop before clustering'):
            _first_core(tensor, None, ((0,), (1,), (2,), (3,)),
                        'slices', cluster_atol, 50, 10, None,
                        _TruncationSpec(**options))

    def test_cluster_tolerance_does_not_filter_eigenvalues(self):
        weights = torch.tensor([1e-12, 1., 2e-12, 1.],
                                dtype=torch.complex128)
        tensor = torch.ones(4, 4, 4, dtype=torch.complex128)
        tensor = tensor * weights[None, :, None]
        first = _first_core(tensor, [1] * 3, ((0,), (1,), (2,), (3,)),
                            'slices', .1, 50, 10,
                            torch.Generator().manual_seed(81),
                            _TruncationSpec())
        assert first.core.shape == (1, 4, 1)
        assert torch.isfinite(first.core).all()

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    def test_rejects_incompatible_prescribed_rank(self, function):
        tensor = _tensor([2] * 3, [4] * 3)
        with pytest.raises(RuntimeError, match='rank deficient'):
            function(tensor, rank=2, rtol=.999, n_attempts=1,
                     generator=torch.Generator().manual_seed(81))

    @pytest.mark.parametrize('scale,expected_rank', [(1., 3), (100., 4)])
    def test_slice_discovery_uses_absolute_singular_cutoff(
            self, scale, expected_rank, monkeypatch):
        matrix = scale * torch.diag(torch.tensor([4., 3., .02, .001],
                                                dtype=torch.complex128))
        tensor = matrix[:, None, :].expand(-1, 4, -1)

        def diagonalize(numerator, denominator, rank):
            assert rank == expected_rank
            raise RuntimeError('Stop before clustering')

        monkeypatch.setattr(
            'tensorkrowch.decompositions.ring.blostr._ratio_eigen',
            diagonalize)
        with pytest.raises(RuntimeError, match='Stop before clustering'):
            _first_core(tensor, None, ((0,), (1,), (2,), (3,)),
                        'slices', 1e-10, 50, 10, None,
                        _TruncationSpec(cutoff=.01))

    @pytest.mark.parametrize('scale,inner_rank', [(1., 1), (100., 2)])
    def test_tail_truncation_uses_explicit_absolute_error(
            self, scale, inner_rank):
        # The tail has singular values 1 and .05. An absolute tolerance .1
        # drops the second at scale 1, but retains it at scale 100.
        tail = torch.zeros(2, 4, 4, 2, dtype=torch.complex128)
        tail[0, 0, 0, 0] = 1
        tail[0, 1, 1, 0] = .05
        tensor = scale * tail.movedim(-1, 0).reshape(4, 4, 4)
        first = torch.eye(4, dtype=torch.complex128).reshape(4, 2, 2)
        first = first.permute(1, 0, 2)
        engine = tk.decompositions.BLOSTRSVD(tensor)
        cores = engine._recover(first, None, 'slices', None,
                                _TruncationSpec(atol=.1**2), None)
        result = tk.decompositions.TRDecomposition(cores)
        assert result.rank == [2, inner_rank, 2]
        assert (result.contract_dense() - tensor).norm() <= .1

    @pytest.mark.parametrize('function', [tr_blostr, tr_blostr_svd])
    def test_numerically_null_singular_values_fail_safely(self, function):
        matrix = torch.diag(torch.tensor([1., 1e-18, 0., 0.],
                                         dtype=torch.float64))
        target = matrix[:, None, :].expand(-1, 4, -1)
        with pytest.raises(RuntimeError, match='rank deficient'):
            function(target, rank=2, slices=((0,), (1,), (2,), (3,)))

    def test_zero_tensor_has_no_spectral_support(self):
        with pytest.raises(RuntimeError, match='nonzero eigenvalues'):
            tr_blostr_svd(torch.zeros(4, 4, 4), rank=1, n_attempts=1)
