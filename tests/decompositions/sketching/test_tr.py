"""Tests for centered sampled tensor-ring reconstruction."""

from itertools import product

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.sketching import tt as tt_module
from tests.decompositions.als._oracles import (contract_tr_dense,
                                              make_tr_cores)


def _grid(dimensions):
    return torch.tensor(list(product(*(range(dim) for dim in dimensions))))


def _opener(max_sweeps=200):
    return tk.decompositions.ALSLoopOpener({
        'convergence': tk.decompositions.ConvergencePolicy(
            max_sweeps=max_sweeps, rtol=1e-12, keep_best=True)})


def _relative_error(result, target):
    return (result.contract_dense() - target).norm() / target.norm()


def _rank_one_problem(n_sites=5):
    domain = torch.tensor([0., 1.], dtype=torch.float64)
    samples = _grid((2,) * n_sites).to(torch.float64)

    def function(data):
        return (1 + data).prod(dim=1)

    def embedding(values):
        return torch.stack((1 - values, values), dim=-1)

    return function, embedding, samples, domain


class TestTRRSS:  # MARK: TestTRRSS

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('n_sites', [1, 5])
    def test_shared_transform_indices_preserve_rank_one_recovery(
            self, dtype, n_sites):
        samples = _grid((2,) * n_sites).flip(0)
        target = (1 + _grid((2,) * n_sites)).prod(-1).to(dtype)
        target = target.reshape((2,) * n_sites)
        if dtype.is_complex:
            target = target * (1 + .5j)
        transformed = []

        def transform(configuration_map):
            rows, sample_rows = configuration_map.sample_indices()
            assert torch.equal(sample_rows, torch.arange(len(samples)))
            values = configuration_map.values.clone().requires_grad_()
            expected = 2 * target[tuple(samples.unbind(-1))]
            loss = (values[rows] - expected[sample_rows]).abs().square().mean()
            for index in configuration_map.phi_indices:
                phi = values[index]
                loss = loss + .01 * phi.abs().square().mean()
            gradient, = torch.autograd.grad(loss, values)
            values = (values - .1 * gradient).detach()
            transformed.append((configuration_map.configurations, values))
            return values

        result = tk.decompositions.TRRSS(target, transform=transform).fit(
            samples, rank=1, loop_opener=_opener(30),
            generator=torch.Generator().manual_seed(47))
        assert len(transformed) == 1
        configurations, expected = transformed[0]
        actual = result.contract_dense()[tuple(configurations.unbind(-1))]
        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize('rank', [None, 1, 2])
    def test_single_site_ring_has_no_outward_branch(self, rank):
        target = torch.tensor([1., 2., -1.], dtype=torch.complex128)
        if rank is None:
            with pytest.raises(ValueError, match='endpoint'):
                tk.decompositions.TRRSS(target).fit(_grid(target.shape))
            return
        result = tk.decompositions.TRRSS(target).fit(
            _grid(target.shape), rank=rank, loop_opener=_opener(30),
            generator=torch.Generator().manual_seed(47))

        assert result.n_sites == 1
        assert result.metadata['boundary_blocks'] == ((), ())
        torch.testing.assert_close(result.contract_dense(), target)

    def test_zero_target_has_finite_cores_and_zero_error(self):
        target = torch.zeros(2, 3, 2, dtype=torch.float64)
        result = tk.decompositions.TRRSS(target).fit(
            _grid(target.shape), rank=1, loop_opener=_opener(30),
            generator=torch.Generator().manual_seed(47),
            collect_metrics=True)

        assert all(torch.isfinite(core).all() for core in result.cores)
        torch.testing.assert_close(result.contract_dense(), target)
        assert result.metadata['sample_error'] == 0

    @pytest.mark.parametrize('n_sites', [2, 3, 5, 8])
    @pytest.mark.parametrize('center', [0, None, -1])
    def test_rank_one_recovery_from_center_or_end(self, n_sites, center):
        function, embedding, samples, domain = _rank_one_problem(n_sites)
        center = n_sites - 1 if center == -1 else center
        result = tk.decompositions.TRRSS(
            function, embedding, domain=domain).fit(
                samples, rank=1, center=center, loop_opener=_opener(30),
                generator=torch.Generator().manual_seed(47),
                collect_metrics=True)

        target = function(samples).reshape((2,) * n_sites)
        assert isinstance(result, tk.decompositions.TRDecomposition)
        assert result.rank == [1] * n_sites
        assert _relative_error(result, target) < 1e-8
        assert result.metadata['sample_error'] < 1e-8
        assert result.metadata['center'] == (
            n_sites // 2 if center is None else center)
        assert result.metrics.truncations
        assert any(record.kind == 'sketch_samples'
                   for record in result.metrics.errors)

    @pytest.mark.parametrize('rank', [None, (1, 1, 1, 1, 1)])
    def test_rank_discovery_and_explicit_profile(self, rank):
        function, embedding, samples, domain = _rank_one_problem()
        result = tk.decompositions.TRRSS(
            function, embedding, domain=domain).fit(
                samples, rank=rank, loop_opener=_opener(30), rtol=1e-12,
                generator=torch.Generator().manual_seed(3))

        assert result.rank == [1] * 5
        assert _relative_error(
            result, function(samples).reshape((2,) * 5)) < 1e-8
        discovered = result.metadata['rank_discovery'] is not None
        assert discovered == (rank is None)
        assert result.metadata['requested_rank'] == rank

    @pytest.mark.parametrize('n_sites', [2, 3])
    def test_overlapping_automatic_blocks_merge_into_whole_chain(
            self, n_sites):
        target = torch.ones((2,) * n_sites, dtype=torch.complex128)
        result = tk.decompositions.TRRSS(target).fit(
            _grid(target.shape), rank=2, center=1,
            loop_opener=_opener(100),
            generator=torch.Generator().manual_seed(18))

        assert result.metadata['center_block'] == tuple(range(n_sites))
        assert result.metadata['boundary_blocks'] == ((), ())
        assert _relative_error(result, target) < 1e-8

    @pytest.mark.parametrize('block_boundaries', [False, True])
    def test_boundary_block_option_preserves_a_nonconstant_target(
            self, block_boundaries):
        function, embedding, samples, domain = _rank_one_problem(8)
        result = tk.decompositions.TRRSS(
            function, embedding, domain=domain).fit(
                samples, rank=1, block_boundaries=block_boundaries,
                loop_opener=_opener(30),
                generator=torch.Generator().manual_seed(53))

        assert _relative_error(
            result, function(samples).reshape((2,) * 8)) < 1e-8
        if not block_boundaries:
            assert result.metadata['boundary_blocks'] == ((0,), (7,))

    def test_complex_rank_two_recovery_keeps_ambient_sketch_recursion(
            self, monkeypatch):
        generator = torch.Generator().manual_seed(72)
        original = make_tr_cores(
            in_dim=(5,) * 5, rank=(2,) * 5, dtype=torch.complex128,
            generator=generator)
        target = contract_tr_dense(original)
        target = target / target.norm()
        calls = []
        delegate = _opener()

        class RecordingOpener:
            capabilities = delegate.capabilities

            def open(self, target, rank, **kwargs):
                fixed_left = kwargs.get('fixed_left')
                fixed_right = kwargs.get('fixed_right')
                calls.append((tuple(target.shape),
                              None if fixed_left is None else
                              tuple(fixed_left.shape),
                              None if fixed_right is None else
                              tuple(fixed_right.shape)))
                return delegate.open(target, rank, **kwargs)

        planned = []
        original_phi = tt_module.PhiOperator

        def count_phi(*args, **kwargs):
            phi = original_phi(*args, **kwargs)
            planned.append(phi)
            return phi

        monkeypatch.setattr(tt_module, 'PhiOperator', count_phi)
        result = tk.decompositions.TRRSS(target).fit(
            _grid(target.shape), rank=2, center=2,
            loop_opener=RecordingOpener(), batch_size=512,
            generator=torch.Generator().manual_seed(73),
            collect_metrics=True)

        assert result.rank == [2] * 5
        assert _relative_error(result, target) < 1e-6
        assert len(planned) == 5
        assert all(phi.tensor is None for phi in planned)
        assert result.metadata['tt_rank_caps'] == (4,) * 4
        inherited_sizes = [fixed[1] for _, left, right in calls
                           for fixed in (left, right) if fixed is not None]
        assert max(inherited_sizes) > 4
        assert any(record.kind == 'tt_alignment'
                   for record in result.metrics.errors)

    def test_nonuniform_rank_profile_has_one_rank_per_right_bond(self):
        rank = (2, 1, 2, 1, 2)
        original = make_tr_cores(
            in_dim=(5,) * 5, rank=rank, dtype=torch.float64,
            generator=torch.Generator().manual_seed(41))
        target = contract_tr_dense(original)
        target = target / target.norm()
        result = tk.decompositions.TRRSS(target).fit(
            _grid(target.shape), rank=rank, loop_opener=_opener(),
            generator=torch.Generator().manual_seed(42))

        assert result.metadata['requested_rank'] == rank
        assert result.metadata['tt_rank_caps'] == (4, 2, 4, 2)
        assert _relative_error(result, target) < 1e-6
        assert all(actual <= cap for actual, cap in zip(result.rank, rank))

    def test_damped_exact_block_with_qr_balancing_and_refinement(self):
        generator = torch.Generator().manual_seed(701)
        damping = torch.tensor([1., 0.003])
        original = tk.formats.TR([
            torch.randn(2, 4, 2, dtype=torch.complex128,
                        generator=generator) * damping
            for _ in range(6)])
        target = original.contract_dense()
        target = target / target.norm()
        result = tk.decompositions.TRRSS(target).fit(
            _grid(target.shape), rank=2, rtol=1e-20,
            loop_opener=tk.decompositions.QRLoopOpener(_opener(199)),
            collect_metrics=True,
            generator=torch.Generator().manual_seed(702))

        assert len(result.metadata['center_block']) > 1
        assert _relative_error(result, target) < 1e-9
        record = next(error for error in result.metrics.errors
                      if error.kind == 'tt_alignment')
        assert record.relative < 1e-9

    def test_discovery_uses_tt_blocks_before_selecting_central_block(self):
        generator = torch.Generator().manual_seed(93)
        original = tk.formats.TR([
            torch.randn(2, 2, 2, dtype=torch.complex128,
                        generator=generator) for _ in range(11)])
        target = original.contract_dense()
        target = target / target.norm()
        result = tk.decompositions.TRRSS(target).fit(
            _grid(target.shape), rtol=1e-10, loop_opener=_opener(200),
            generator=torch.Generator().manual_seed(94))

        assert result.metadata['rank_discovery']['cyclic_rank'] == 2
        assert len(result.metadata['center_block']) > 1
        assert _relative_error(result, target) < 1e-6

    @pytest.mark.parametrize('block_boundaries', [False, True])
    @pytest.mark.parametrize('count,center,message', [
        (2, 0, 'endpoint'), (7, 0, 'endpoint'), (7, 6, 'endpoint'),
        (7, 1, 'overlaps'), (7, 5, 'overlaps'), (5, 2, 'too small')])
    def test_discovery_rejects_invalid_partitions(
            self, block_boundaries, count, center, message):
        cores = make_tr_cores(
            in_dim=(4,) * count, rank=(2,) * count,
            dtype=torch.complex128,
            generator=torch.Generator().manual_seed(318))
        source = tk.decompositions.FormatSource(tk.formats.TR(cores))
        samples = torch.randint(4, (64, count),
                                generator=torch.Generator().manual_seed(319))
        with pytest.raises(ValueError, match=message):
            tk.decompositions.TRRSS(source).fit(
                samples, center=center, block_boundaries=block_boundaries,
                rtol=1e-12)

    @pytest.mark.parametrize('block_boundaries', [False, True])
    @pytest.mark.parametrize('discover', [False, True])
    @pytest.mark.parametrize('ranks,dimension', [
        ((2,) * 7, 4), ((2, 3, 2, 3, 2, 2, 2), 5)])
    def test_complete_rank_profiles_use_separate_als_cores(
            self, block_boundaries, discover, ranks, dimension, monkeypatch):
        cores = make_tr_cores(
            in_dim=(dimension,) * 7, rank=ranks, dtype=torch.complex128,
            generator=torch.Generator().manual_seed(96))
        ring = tk.formats.TR(cores)
        ring = ring / ring.norm()
        samples = torch.randint(dimension, (128, 7),
                                generator=torch.Generator().manual_seed(319))
        als = _opener(200)
        opener = tk.decompositions.QRLoopOpener(als)
        calls = []
        solve = als.open

        def record(target, rank, **kwargs):
            sites = kwargs['context']['sites']
            calls.append((sites, kwargs['fixed_left'], kwargs['fixed_right']))
            assert target.shape[1:-1] == (dimension,) * len(sites)
            assert tuple(rank) == (ranks[sites[0] - 1],
                                   *(ranks[site] for site in sites), ranks[-1])
            return solve(target, rank, **kwargs)

        monkeypatch.setattr(als, 'open', record)
        result = tk.decompositions.TRRSS(
            tk.decompositions.FormatSource(ring)).fit(
                samples, rank=None if discover else ranks,
                block_boundaries=block_boundaries, rtol=1e-12,
                loop_opener=opener,
                generator=torch.Generator().manual_seed(101))
        assert calls[0][0] == result.metadata['center_block']
        if discover:
            assert all(len(sites) == 1 for sites, _, _ in calls[1:])
        else:
            for boundary in result.metadata['boundary_blocks']:
                if len(boundary) > 1:
                    fixed = next(call for call in calls if call[0] == boundary)
                    assert (fixed[1] is not None) or (fixed[2] is not None)
        assert result.rank == list(ranks)
        assert _relative_error(result, ring.contract_dense()) < 1e-6
        if discover:
            info = result.metadata['rank_discovery']
            sites = info['sites']
            assert info['left_dim'] == ranks[sites[0] - 1] * ranks[-1]
            assert info['right_dim'] == ranks[sites[-1]] * ranks[-1]
            assert info['cyclic_rank'] == ranks[-1]

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('center', [1, 3])
    def test_discovery_keeps_physical_cores_and_recovers_cyclic_one(
            self, dtype, center):
        rank = (3, 3, 3, 3, 1)
        cores = make_tr_cores(
            in_dim=(5,) * 5, rank=rank, dtype=dtype,
            generator=torch.Generator().manual_seed(317))
        target = contract_tr_dense(cores)
        target = target / target.norm()
        result = tk.decompositions.TRRSS(target).fit(
            _grid(target.shape), center=center, rtol=1e-12,
            loop_opener=_opener(30),
            generator=torch.Generator().manual_seed(318))

        assert result.metadata['center_block'] == (
            (1, 2) if center == 1 else (2, 3))
        assert result.metadata['boundary_blocks'] == ((0,), (4,))
        assert result.metadata['rank_discovery']['physical_rank'] == 9
        assert result.metadata['rank_discovery']['cyclic_rank'] == 1
        assert result.rank == list(rank)
        assert _relative_error(result, target) < 1e-10

    def test_function_returns_result_and_repeated_fits_are_independent(self):
        function, embedding, samples, domain = _rank_one_problem()
        engine = tk.decompositions.TRRSS(function, embedding, domain=domain)
        first = engine.fit(samples, rank=1,
                           generator=torch.Generator().manual_seed(10))
        second = engine.fit(samples, rank=1,
                            generator=torch.Generator().manual_seed(11))
        functional = tk.decompositions.tr_rss(
            function, samples, embedding, rank=1, domain=domain,
            generator=torch.Generator().manual_seed(12))

        assert first is not second
        assert first.metrics is not second.metrics
        assert isinstance(functional, tk.decompositions.TRDecomposition)
        torch.testing.assert_close(first.contract_dense(),
                                   functional.contract_dense())
        assert all(left is not right for left, right in
                   zip(first.cores, second.cores))

    def test_global_transform_acts_once_and_keeps_source_queries_cached(self):
        target = torch.arange(1., 33., dtype=torch.float64).reshape((2,) * 5)
        calls = []

        def transform(configuration_map):
            calls.append(configuration_map.configurations.clone())
            raw = configuration_map.evaluate(configuration_map.configurations)
            torch.testing.assert_close(raw, configuration_map.values)
            return torch.ones_like(configuration_map.values)

        result = tk.decompositions.TRRSS(target, transform=transform).fit(
            _grid(target.shape), rank=1,
            generator=torch.Generator().manual_seed(12))

        assert len(calls) == 1
        torch.testing.assert_close(result.contract_dense(),
                                   torch.ones_like(target))

    def test_global_transform_promotes_real_data_and_metrics_to_complex(self):
        target = torch.ones((2,) * 5, dtype=torch.float64)
        result = tk.decompositions.TRRSS(
            target, transform=lambda mapping: 1j * mapping.values).fit(
                _grid(target.shape), rank=1, loop_opener=_opener(30),
                generator=torch.Generator().manual_seed(16),
                collect_metrics=True)

        torch.testing.assert_close(result.contract_dense(), 1j * target)
        assert result.dtype == torch.complex128
        alignment = next(record for record in result.metrics.errors
                         if record.kind == 'tt_alignment')
        assert alignment.relative < 1e-10
        assert result.metrics.errors[-1].kind == 'configuration_map'
        assert result.metrics.errors[-1].relative < 1e-10

    def test_tensor_outputs_preserve_explicit_input_positions(self):
        function, embedding, samples, domain = _rank_one_problem(3)
        left = torch.tensor([1., 2.], dtype=torch.float64)
        right = torch.tensor([1., 3.], dtype=torch.float64)

        def tensor_function(data):
            return (function(data)[:, None, None] * left[None, :, None] *
                    right[None, None, :])

        labels = torch.arange(samples.shape[0]) % 4
        result = tk.decompositions.TRRSS(
            tensor_function, embedding, domain=domain,
            out_position=(0, 4)).fit(
                samples, labels=labels, rank=1, loop_opener=_opener(30),
                generator=torch.Generator().manual_seed(19))

        assert result.in_sites == (1, 2, 3)
        assert result.out_shape == (2, 2)
        torch.testing.assert_close(result.evaluate(samples.long()),
                                   tensor_function(samples))

    def test_matrix_source_returns_paired_matrix_result(self):
        tensor = torch.ones(2, 3, 2, 3, 2, 3, dtype=torch.float64)
        source = tk.decompositions.DenseTensorSource.matrix(
            tensor, ordering='interleaved')
        samples = _grid((2, 2, 2))
        result = tk.decompositions.TRRSS(source).fit(
            samples, rank=1, loop_opener=_opener(30),
            generator=torch.Generator().manual_seed(5))

        assert isinstance(result, tk.decompositions.TRMDecomposition)
        assert result.in_dim == (2, 2, 2)
        assert result.out_dim == (3, 3, 3)
        torch.testing.assert_close(result.contract_dense(), tensor)

    def test_nonseparable_matrix_uses_public_paired_bond_profile(self):
        generator = torch.Generator().manual_seed(28)
        cores = [torch.randn(2, 2, 2, 3, dtype=torch.complex128,
                             generator=generator) for _ in range(3)]
        original = tk.formats.TRM(cores)
        target = original.contract_dense()
        target = target / target.norm()
        source = tk.decompositions.DenseTensorSource.matrix(
            target, ordering='interleaved')
        samples = _grid((2,) * 3).repeat_interleave(27, dim=0)
        labels = torch.arange(27).repeat(8)
        result = tk.decompositions.TRRSS(source).fit(
            samples, labels=labels, rank=(2, 2, 2),
            loop_opener=_opener(200),
            generator=torch.Generator().manual_seed(29))

        assert isinstance(result, tk.decompositions.TRMDecomposition)
        assert result.rank == [2, 2, 2]
        assert result.metadata['requested_rank'] == (2, 2, 2)
        assert _relative_error(result, target) < 1e-6

    @pytest.mark.parametrize('kwargs, error', [
        ({'rank': True}, TypeError),
        ({'rank': (1, 1)}, ValueError),
        ({'rank': 0}, ValueError),
        ({'center': True}, TypeError),
        ({'center': 5}, ValueError),
        ({'rank': 1, 'block_boundaries': 1}, TypeError),
        ({'rank': 1, 'loop_opener': 'unknown'}, ValueError)])
    def test_invalid_fit_options(self, kwargs, error):
        with pytest.raises(error):
            tk.decompositions.TRRSS(torch.ones((2,) * 5)).fit(
                _grid((2,) * 5), **kwargs)


class TestQTRRSS:  # MARK: TestQTRRSS

    @pytest.mark.parametrize('sample_space', ['coordinates', 'digits',
                                              'digit_batch'])
    def test_quantized_wrapper_preserves_layout_and_values(self, sample_space):
        layout = tk.formats.QuantizedLayout(1, base=2, level=4)
        coordinate_map = tk.formats.AffineCoordinateMap(
            torch.tensor([0., 1.]), layout.grid_size,
            grid_offset='endpoints')
        indices = torch.arange(16).reshape(-1, 1)
        coordinates = coordinate_map.from_indices(indices)
        source = tk.decompositions.CallableQuanticsSource(
            lambda values: torch.exp(values[:, 0]), 1, layout=layout,
            coordinate_map=coordinate_map)
        samples = coordinates if sample_space == 'coordinates' else \
            layout.encode_indices(indices)
        if sample_space == 'digit_batch':
            samples = tk.decompositions.ConfigurationBatch(samples)
            sample_space = 'digits'
        result = tk.decompositions.qtr_rss(
            source, samples, sample_space=sample_space, rank=1,
            loop_opener=_opener(30),
            generator=torch.Generator().manual_seed(63))

        assert isinstance(result, tk.decompositions.QTRDecomposition)
        assert result.layout == layout
        torch.testing.assert_close(result.evaluate_indices(indices),
                                   torch.exp(coordinates[:, 0]))

    def test_quantized_wrapper_rejects_wrong_source_or_sample_space(self):
        with pytest.raises(TypeError, match='CallableQuanticsSource'):
            tk.decompositions.qtr_rss(torch.ones(2, 2), torch.zeros(1, 2))
        source = tk.decompositions.CallableQuanticsSource(
            lambda data: data[:, 0], 1, base=2, level=3,
            domain=torch.tensor([0., 1.]))
        with pytest.raises(ValueError, match='sample_space'):
            tk.decompositions.qtr_rss(source, torch.zeros(1, 1),
                                      sample_space='bad')
