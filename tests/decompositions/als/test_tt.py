"""Tests for tensor train ALS engines and the basic functional interface."""

import pytest
import torch

import tensorkrowch as tk
from tests.decompositions.als._oracles import (contract_tt_dense,
                                               make_tt_cores,
                                               solve_local_core)


d = tk.decompositions


class TestTTALS:  # MARK: TestTTALS

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('n_sites', [1, 2, 5])
    @pytest.mark.parametrize('gauge', ['none', 'qr'])
    def test_low_error_recovery_from_perturbed_cores(
        self, dtype, n_sites, gauge
    ):
        dimensions = (3,) * n_sites
        cores = make_tt_cores(dimensions, (2,) * (n_sites - 1), dtype,
                              torch.Generator().manual_seed(18))
        tensor = contract_tt_dense(cores)
        initial = [core + 1e-3 * torch.randn_like(core) for core in cores]
        original = [core.clone() for core in initial]
        result = d.TTALS(tensor, out_device=None).fit(
            initial_cores=initial, gauge=gauge,
            convergence=d.ConvergencePolicy(max_sweeps=40, rtol=1e-9),
            collect_metrics=True)
        relative = (result.contract_dense() - tensor).norm() / tensor.norm()
        assert relative < 1e-8
        errors = [record.abs_error for record in result.metrics.sweeps]
        assert all(new <= old + 1e-11 * tensor.norm()
                   for old, new in zip(errors, errors[1:]))
        assert all(torch.equal(before, after)
                   for before, after in zip(original, initial))
        assert result.metadata['sweep_direction'] == 'alternating'

    @pytest.mark.parametrize('rank', [2, (2, 3, 2), (20, 3, 20)])
    def test_rank_caps_accept_integer_or_per_link_sequence(self, rank):
        tensor = torch.randn(2, 3, 4, 2, dtype=torch.float64)
        result = d.tt_als(tensor, rank=rank, max_sweeps=2)
        expected = [2, 2, 2] if rank == 2 else [2, 3, 2]
        assert result.rank == expected
        assert result.in_dim == tensor.shape

    def test_locally_infeasible_caps_are_clipped(self):
        result = d.tt_als(torch.ones(2, 2, 2, 2), rank=(1, 4, 1))
        assert result.rank == [1, 2, 1]

    @pytest.mark.parametrize('rank', [1, ()])
    def test_one_site_has_no_bonds(self, rank):
        tensor = torch.tensor([1., -2., 0.], dtype=torch.float64)
        result = d.tt_als(tensor, rank=rank, max_sweeps=1,
                          collect_metrics=True)
        assert result.rank == []
        torch.testing.assert_close(result.cores[0], tensor)
        assert result.metrics.sweeps[0].abs_error == 0

    def test_two_sweeps_match_direct_alternating_oracle(self):
        tensor = torch.randn(2, 3, 2, dtype=torch.complex128)
        initial = make_tt_cores(tensor.shape, (2, 2), tensor.dtype)
        expected = [core.clone() for core in initial]
        for order in (range(3), reversed(range(3))):
            for site in order:
                expected[site] = solve_local_core(expected, tensor, site, 'tt')
        result = d.TTALS(tensor).fit(
            initial_cores=initial, gauge='none', renormalize=False,
            solver=d.LeastSquaresSolver(column_scaling=False,
                                        system_scaling=False),
            convergence=d.ConvergencePolicy(max_sweeps=2))
        torch.testing.assert_close(result.contract_dense(),
                                   contract_tt_dense(expected))

    @pytest.mark.parametrize(
        'fixed_sites', [(0,), (2,), (1, 3), (0, 1, 2, 3, 4)]
    )
    def test_frozen_cores_and_blocked_gauge_receivers(self, fixed_sites):
        cores = make_tt_cores((2,) * 5, (2,) * 4)
        tensor = contract_tt_dense(cores)
        fixed = [core if site in fixed_sites else None
                 for site, core in enumerate(cores)]
        result = d.TTALS(tensor).fit(
            initial_cores=cores, fixed_cores=fixed,
            convergence=d.ConvergencePolicy(max_sweeps=3))
        for site in fixed_sites:
            expected = cores[site].squeeze(0) if site == 0 else \
                cores[site].squeeze(-1) if site == 4 else cores[site]
            assert torch.equal(result.cores[site], expected)
        torch.testing.assert_close(result.contract_dense(), tensor)
        if len(fixed_sites) == 5:
            assert result.metadata['stop_reason'] == 'all_cores_fixed'
            assert result.metadata['n_sweeps'] == 0

    @pytest.mark.parametrize('renormalize', [False, True])
    def test_regularized_sweep_matches_original_scale(self, renormalize):
        tensor = torch.randn(2, 3, 2, dtype=torch.float64)
        initial = make_tt_cores(tensor.shape, (2, 2))
        engine = d.TTALS(tensor)
        solver = d.LeastSquaresSolver(l2_reg=1e-3)
        options = dict(initial_cores=initial, solver=solver, gauge='none',
                       convergence=d.ConvergencePolicy(max_sweeps=2))
        direct = engine.fit(renormalize=False, **options)
        actual = engine.fit(renormalize=renormalize, **options)
        torch.testing.assert_close(
            actual.contract_dense(),
            direct.contract_dense(),
            atol=1e-9,
            rtol=1e-9,
        )

    def test_result_wrapper_and_explicit_svd_initialization(self):
        tensor = torch.randn(2, 3, 2, dtype=torch.float64)
        initial = d.tt_svd(tensor, rank=2, out_device=None)
        result = d.tt_als(tensor, initial_cores=initial, max_sweeps=2,
                          collect_metrics=True)
        assert isinstance(result, d.TTDecomposition)
        torch.testing.assert_close(result.contract_dense(), tensor)
        assert result.metrics.local_solves
        assert result.as_info()['metadata']['algorithm'] == 'tt_als'
        assert isinstance(result.to_format(), tk.formats.TT)
        assert tk.models.MPS(tensors=result.cores).boundary == 'obc'

    def test_fast_path_keeps_metrics_empty(self):
        result = d.tt_als(torch.ones(2, 3), rank=1)
        assert result.metrics.local_solves == result.metrics.sweeps == []

    def test_exact_callable_values_are_cached_across_fits(self):
        tensor = torch.arange(12., dtype=torch.float64).reshape(2, 3, 2)
        calls = []

        def function(indices):
            calls.append(indices.clone())
            return tensor[tuple(indices.T)]

        source = d.CallableTensorSource(function, tensor.shape,
                                         out_shape=(), dtype=tensor.dtype)
        engine = d.TTALS(source)
        for rank in (1, 2):
            engine.fit(rank=rank)
        assert len(calls) == 1
        assert engine.source is source
        assert engine.problem.in_dim == engine.in_dim == tensor.shape
        assert engine.out_device == torch.device('cpu')

    def test_all_vector_sources(self, vector_source_case):
        source, tensor = vector_source_case
        initial = d.tt_svd(tensor, rank=2, out_device=None)
        result = d.TTALS(source).fit(
            initial_cores=initial,
            convergence=d.ConvergencePolicy(max_sweeps=2))
        torch.testing.assert_close(result.contract_dense(), tensor,
                                   atol=1e-10, rtol=1e-10)

    def test_all_matrix_sources_and_format_initialization(
        self, matrix_source_case
    ):
        source, tensor = matrix_source_case
        initial = d.tt_svd(source, rank=2, out_device=None)
        result = d.TTALS(source).fit(
            initial_cores=initial,
            convergence=d.ConvergencePolicy(max_sweeps=2))
        assert isinstance(result, d.TTMDecomposition)
        torch.testing.assert_close(result.contract_dense(), tensor,
                                   atol=2e-10, rtol=2e-10)

    @pytest.mark.parametrize('ordering', ['interleaved', 'grouped'])
    def test_quantics_vector_and_matrix_metadata(self, ordering):
        layout = tk.formats.QuantizedLayout(2, 2, 2, ordering=ordering)
        coordinate_map = tk.formats.AffineCoordinateMap(
            torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64),
            layout.grid_size)
        source = d.CallableQuanticsSource(
            lambda x: torch.exp(x.sum(-1)), 2, layout=layout,
            coordinate_map=coordinate_map)
        engine = d.TTALS(source)
        result = engine.fit(rank=1)
        assert isinstance(result, d.QTTDecomposition)
        assert result.layout == source.layout
        torch.testing.assert_close(
            result.contract_dense(), source.contract_dense())
        coordinates = torch.tensor([[0., 0.], [.75, .5]], dtype=torch.float64)
        torch.testing.assert_close(result.evaluate_coordinates(coordinates),
                                   source.function(coordinates))
        other = d.CallableQuanticsSource(
            source.function,
            2,
            layout=tk.formats.QuantizedLayout(
                2,
                2,
                2,
                ordering='grouped'
                if ordering == 'interleaved'
                else 'interleaved',
            ),
            coordinate_map=coordinate_map,
        )
        with pytest.raises(ValueError, match='digit layout'):
            d.TTALS(other).fit(initial_cores=result)
        matrix = d.CallableQuanticsSource.matrix(
            lambda x, y: torch.exp(x.sum(-1) + y.sum(-1)), 1, 1,
            in_base=2, in_level=2, out_base=2, out_level=2,
            in_domain=torch.tensor([0., 1.], dtype=torch.float64),
            out_domain=torch.tensor([0., 1.], dtype=torch.float64))
        result = d.tt_als(matrix, rank=1)
        assert isinstance(result, d.QTTMDecomposition)
        torch.testing.assert_close(
            result.contract_dense(), matrix.contract_dense())

    def test_reordered_inputs_and_output_axes(self):
        tensor = torch.arange(12., dtype=torch.float64).reshape(2, 3, 2)
        source = d.DenseTensorSource(tensor, in_sites=(2, 0))
        result = d.tt_als(source, rank=2)
        assert (result.in_sites == (2, 0)) and (result.out_shape == (3,))
        torch.testing.assert_close(result.contract_dense(), tensor)
        indices = torch.tensor([[0, 1], [1, 0]])
        torch.testing.assert_close(result.evaluate(
            indices), source.evaluate(d.ConfigurationBatch(indices)))

    def test_device_and_dtype(self, device_dtype, assert_close):
        device, dtype = device_dtype
        tensor = torch.arange(1, 4, device=device,
                              dtype=torch.float32).to(dtype)
        if dtype.is_complex:
            tensor = tensor * (1 + 2j)
        result = d.tt_als(tensor, rank=1, out_device=None)
        assert (result.dtype == dtype) and (result.device.type == device)
        assert_close(result.contract_dense(), tensor)


class TestTTALSObjectives:  # MARK: TestTTALSObjectives

    @pytest.mark.parametrize('sparse', [False, True])
    def test_mask_weights_and_observed_zeros(self, sparse):
        tensor = torch.outer(
            torch.tensor([0.0, 2.0, -1.0], dtype=torch.float64),
            torch.tensor([1.0, 2.0, 3.0, -1.0], dtype=torch.float64),
        )
        tensor[2, 3] = 1e5
        mask = torch.ones_like(tensor, dtype=torch.bool)
        mask[2, 3] = False
        weights = torch.linspace(.5, 2., 12).reshape_as(tensor)
        if sparse:
            indices = mask.nonzero()
            mask = d.SparseTensorSource(indices, torch.ones(len(indices),
                                        dtype=torch.bool), tensor.shape)
        engine = d.TTALS(tensor, mask=mask, weights=weights)
        result = engine.fit(rank=1, collect_metrics=True,
                            convergence=d.ConvergencePolicy(max_sweeps=20))
        selected = torch.ones_like(tensor, dtype=torch.bool)
        selected[2, 3] = False
        residual = ((result.contract_dense() - tensor) * weights)[selected]
        expected = residual.norm()
        torch.testing.assert_close(
            result.metrics.sweeps[-1].abs_error, expected)
        assert expected < 1e-8
        if sparse:
            observations = engine.problem.observations
            assert observations.support_values[0] == 0
            assert result.metadata['sampling'] == 'observed'
            assert engine.problem.weights.shape == (11,)
            with pytest.raises(ValueError, match='permanent observations'):
                engine.fit(rank=2, sampler=d.SamplingPolicy(10))

    def test_sparse_mask_does_not_query_unknown_entries(self):
        indices = torch.tensor([[0, 0], [0, 1], [1, 0]])

        def function(query):
            assert not torch.any((query == 1).all(-1))
            return query.sum(-1).double()

        source = d.CallableTensorSource(function, (2, 2),
                                         out_shape=(), dtype=torch.float64)
        mask = d.SparseTensorSource(
            indices, torch.ones(3, dtype=torch.bool), (2, 2))
        result = d.tt_als(source, rank=2, mask=mask, collect_metrics=True)
        assert result.metrics.sweeps[-1].abs_error < 1e-10

    def test_zero_target_has_defined_relative_error(self):
        result = d.tt_als(torch.zeros(2, 3), rank=1, rtol=1e-12,
                          collect_metrics=True)
        assert result.metrics.sweeps[-1].rel_error == 0
        assert result.metadata['stop_reason'] == 'rtol'


class TestTTALSSampling:  # MARK: TestTTALSSampling

    @pytest.mark.parametrize('method', ['uniform', 'leverage'])
    def test_sampled_recovery_and_reproducibility(self, method):
        tensor = torch.einsum(
            'i,j,k->ijk',
            torch.arange(1.0, 4.0).double(),
            torch.arange(1.0, 5.0).double(),
            torch.tensor([1.0, -2.0], dtype=torch.float64),
        )
        options = dict(rank=1, sampler=d.SamplingPolicy(80, method=method),
                       convergence=d.ConvergencePolicy(max_sweeps=20),
                       collect_metrics=True)
        engine = d.TTALS(tensor)
        first = engine.fit(
            generator=torch.Generator().manual_seed(28), **options)
        second = engine.fit(
            generator=torch.Generator().manual_seed(28), **options)
        torch.testing.assert_close(first.contract_dense(), tensor)
        for a, b in zip(first.cores, second.cores):
            torch.testing.assert_close(a, b)
        assert all(record.abs_error is None for record in first.metrics.sweeps)
        assert first.metrics.local_solves
        assert first.metadata['sampler']['method'] == method

    def test_uniform_reuses_values_for_each_generation(self):
        calls = []

        def function(indices):
            calls.append(indices.clone())
            return indices.sum(-1).double()

        source = d.CallableTensorSource(function, (2, 3, 2, 2),
                                         out_shape=(), dtype=torch.float64)
        result = d.TTALS(source).fit(
            rank=2,
            sampler=d.SamplingPolicy(10, reuse_sweeps=2),
            convergence=d.ConvergencePolicy(max_sweeps=3),
            collect_metrics=True,
        )
        assert len(calls) == 2
        assert [r.sample_generation for r in result.metrics.sweeps] == [
            0,
            0,
            1,
        ]

    def test_leverage_redraws_current_site_design(self):
        calls = []

        def function(indices):
            calls.append(indices.clone())
            return indices.sum(-1).double()

        source = d.CallableTensorSource(function, (2, 3, 2),
                                         out_shape=(), dtype=torch.float64)
        result = d.TTALS(source).fit(
            rank=2,
            sampler=d.SamplingPolicy(15, method='leverage', uniform_mix=0.1),
            convergence=d.ConvergencePolicy(max_sweeps=2),
            collect_metrics=True,
        )
        assert len(calls) == 6
        assert all(query.shape == (15, 3) for query in calls)
        assert all(r.sampling_exact for r in result.metrics.local_solves)


class TestTTALSValidation:  # MARK: TestTTALSValidation

    @pytest.mark.parametrize('rank', [True, 0, -1, 1.5, '2', {1, 2},
                                      (1,), (1, 0), (True, 1)])
    def test_invalid_rank(self, rank):
        with pytest.raises((TypeError, ValueError), match='rank'):
            d.TTALS(torch.ones(2, 3, 2)).fit(rank=rank)

    @pytest.mark.parametrize('options', [
        {}, {'rank': 2, 'gauge': 'svd'}, {'rank': 2, 'gauge': None},
        {'rank': 2, 'renormalize': 1}, {'rank': 2, 'sampler': 'uniform'},
        {'rank': 2, 'solver': object()}, {'rank': 2, 'convergence': 10},
        {'rank': 2, 'generator': 1}, {'rank': 2, 'collect_metrics': 1},
        {'rank': 2, 'initial_cores': []},
        {'rank': 2, 'fixed_cores': []},
        {'rank': 2, 'fixed_cores': [None, object(), None]},
    ])
    def test_invalid_fit_arguments(self, options):
        with pytest.raises((TypeError, ValueError)):
            d.TTALS(torch.ones(2, 3, 2)).fit(**options)

    def test_initial_core_types_shapes_topology_and_rank_caps(self):
        tensor = torch.ones(2, 3, 2, dtype=torch.float64)
        cores = make_tt_cores(tensor.shape, (2, 2))
        for invalid in ('cores', [1, 2, 3], [cores[0]],
                        [core.float() for core in cores],
                        [torch.ones(1, 3, 2), *cores[1:]]):
            with pytest.raises((TypeError, ValueError)):
                d.TTALS(tensor).fit(initial_cores=invalid)
        with pytest.raises(ValueError, match='rank.*caps'):
            d.TTALS(tensor).fit(rank=1, initial_cores=cores)
        with pytest.raises(ValueError, match='topology'):
            d.TTALS(tensor).fit(initial_cores=tk.formats.TR(cores))
        with pytest.raises(ValueError, match='shape'):
            d.TTALS(tensor).fit(initial_cores=cores,
                                fixed_cores=[None, torch.ones(1, 3, 1), None])

    def test_invalid_sources_masks_and_sampled_criteria(self):
        with pytest.raises(ValueError, match='discrete'):
            d.TTALS(d.CallableSource(lambda x: x.sum(-1), 2))
        with pytest.raises(ValueError, match='batch'):
            d.TTALS(d.DenseTensorSource(torch.ones(2, 3, 4), n_batches=1))
        with pytest.raises(TypeError, match='floating or complex'):
            d.tt_als(torch.ones(2, 3, dtype=torch.long), rank=1)
        with pytest.raises(TypeError, match='boolean'):
            d.TTALS(torch.ones(2, 3), mask=torch.ones(2, 3))
        with pytest.raises(ValueError, match='dimensions'):
            d.TTALS(torch.ones(2, 3), mask=torch.ones(2, 2, dtype=torch.bool))
        with pytest.raises(ValueError, match='fixed global objective'):
            d.TTALS(torch.ones(2, 3)).fit(
                rank=1, sampler=d.SamplingPolicy(5),
                convergence=d.ConvergencePolicy(rtol=.1))
        with pytest.raises(ValueError, match='gauge'):
            d.TTALS(torch.ones(2, 3)).fit(
                rank=1,
                gauge='none',
                sampler=d.SamplingPolicy(5, method='leverage'),
            )
        cores = make_tt_cores((2, 3), (1,), dtype=torch.float32)
        with pytest.raises(ValueError, match='fixed cores'):
            d.TTALS(torch.ones(2, 3)).fit(
                initial_cores=cores, fixed_cores=[cores[0], None],
                sampler=d.SamplingPolicy(5, method='leverage'))

    @pytest.mark.parametrize('name,value', [
        ('return_info', True), ('return_result', True), ('init', 'svd'),
        ('sweep_direction', 'forward'), ('sampling', 'uniform'),
        ('damping', .5), ('acceptance', 'always'),
    ])
    def test_removed_function_options_are_not_silently_accepted(
        self, name, value
    ):
        with pytest.raises(TypeError):
            d.tt_als(torch.ones(2, 3), rank=1, **{name: value})


@pytest.mark.parametrize('engine', [d.TTALS, d.TRALS])
def test_permuted_sparse_mask_preserves_observation_weights(engine):
    tensor = torch.tensor([[0., 3.], [5., 7.]], dtype=torch.float64)
    weights = torch.tensor([[1., 2.], [4., 8.]], dtype=torch.float64)
    mask = d.SparseTensorSource(torch.tensor([[0, 1], [1, 0]]),
                                 torch.ones(2, dtype=torch.bool),
                                 (2, 2), in_sites=(1, 0))
    problem = engine(tensor, mask=mask, weights=weights).problem
    indices = problem.observations.support.values
    torch.testing.assert_close(problem.weights, weights[tuple(indices.T)])
    approximation = torch.zeros(2, dtype=torch.float64)
    absolute, relative = problem.objective_error(approximation)
    expected = (tensor[tuple(indices.T)] * weights[tuple(indices.T)]).norm()
    torch.testing.assert_close(absolute, expected)
    assert relative == 1


@pytest.mark.parametrize('method', [None, 'leverage'])
def test_initial_ranks_must_allow_directional_qr(method):
    cores = make_tt_cores((2, 2, 2, 2), (2, 4, 1))
    options = {} if method is None else {
        'sampler': d.SamplingPolicy(10, method)}
    with pytest.raises(ValueError, match='algebraically feasible'):
        d.TTALS(torch.ones(2, 2, 2, 2, dtype=torch.float64)).fit(
            initial_cores=cores, **options)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('gauge', ['none', 'qr'])
@pytest.mark.parametrize('n_sites,fixed_sites', [
    (1, ()), (1, (0,)), (5, ()), (5, (3,)), (5, (2, 4)),
    (5, (1,)), (5, (0,)), (5, (0, 3)), (5, (0, 1, 2, 3, 4)),
])
def test_first_sweep_qr_prepares_only_initial_free_segment(
        monkeypatch, dtype, gauge, n_sites, fixed_sites):
    from tensorkrowch.decompositions.als.driver import ALSSweepDriver

    cores = make_tt_cores((3,) * n_sites, (2,) * (n_sites - 1), dtype)
    originals = [core.clone() for core in cores]
    initial_tensor = contract_tt_dense(cores)
    target = 3 * initial_tensor
    captured = []
    original_fit = ALSSweepDriver.fit

    def capture(self, problem, backend, *args, **kwargs):
        captured.extend(core.clone() for core in backend.cores)
        return original_fit(self, problem, backend, *args, **kwargs)

    monkeypatch.setattr(ALSSweepDriver, 'fit', capture)
    d.TTALS(target).fit(
        initial_cores=cores, gauge=gauge,
        fixed_cores=[core if site in fixed_sites else None
                     for site, core in enumerate(cores)],
        convergence=d.ConvergencePolicy(max_sweeps=1))

    stop = min(fixed_sites, default=n_sites) if gauge == 'qr' else 0
    for site in range(n_sites):
        if 1 <= site < stop:
            matrix = captured[site].reshape(captured[site].shape[0], -1)
            identity = torch.eye(matrix.shape[0], dtype=dtype)
            torch.testing.assert_close(matrix @ matrix.mH, identity)
        if (site >= stop) or (stop <= 1):
            assert torch.equal(captured[site], originals[site])
        assert torch.equal(cores[site], originals[site])
    torch.testing.assert_close(contract_tt_dense(captured), initial_tensor)
    torch.testing.assert_close(contract_tt_dense(captured).norm(),
                               initial_tensor.norm())


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('direction', ['forward', 'reverse'])
@pytest.mark.parametrize('n_sites', [1, 5])
def test_shared_qr_preparation_preserves_tensor_and_isometries(
        dtype, direction, n_sites):
    from tensorkrowch.decompositions.als.tt import _prepare_tt_cores

    cores = make_tt_cores((3,) * n_sites, (2,) * (n_sites - 1), dtype)
    originals = [core.clone() for core in cores]
    sites = range(n_sites - 1) if direction == 'forward' else \
        range(n_sites - 1, 0, -1)
    prepared = _prepare_tt_cores(cores, sites, direction)
    torch.testing.assert_close(contract_tt_dense(prepared),
                               contract_tt_dense(originals))
    for site in sites:
        core = prepared[site]
        if direction == 'forward':
            matrix = core.reshape(-1, core.shape[-1])
            gram = matrix.mH @ matrix
        else:
            matrix = core.reshape(core.shape[0], -1)
            gram = matrix @ matrix.mH
        torch.testing.assert_close(gram, torch.eye(len(gram), dtype=dtype))
    assert all(torch.equal(core, old) for core, old in zip(cores, originals))


@pytest.mark.parametrize('method', [None, 'leverage'])
def test_exact_and_leverage_use_shared_qr_preparation(monkeypatch, method):
    from importlib import import_module

    module = import_module('tensorkrowch.decompositions.als.tt')
    original = module._prepare_tt_cores
    calls = []

    def capture(cores, sites, direction):
        calls.append((tuple(sites), direction))
        return original(cores, sites, direction)

    monkeypatch.setattr(module, '_prepare_tt_cores', capture)
    options = {} if method is None else {
        'sampler': d.SamplingPolicy(20, method)}
    result = d.TTALS(torch.ones(2, 3, 2)).fit(
        initial_cores=[torch.ones(1, size, 1) for size in (2, 3, 2)],
        convergence=d.ConvergencePolicy(max_sweeps=2),
        collect_metrics=True, **options)
    assert calls == [((2, 1), 'reverse')]
    torch.testing.assert_close(result.contract_dense(), torch.ones(2, 3, 2))
    if method == 'leverage':
        assert all(record.sampling_exact
                   for record in result.metrics.local_solves)
