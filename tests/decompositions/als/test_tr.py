"""Tests for tensor ring alternating least squares."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.observers import HistoryObserver

from tests.decompositions.als._oracles import (contract_tr_dense,
                                               make_tr_cores,
                                               reference_tr_sweep,
                                               solve_local_core)


def _exact_tr(dtype=torch.float64):
    """Returns a heterogeneous TR and its dense contraction."""
    cores = make_tr_cores(
        in_dim=(2, 3, 2),
        rank=(2, 3, 2),
        dtype=dtype,
        generator=torch.Generator().manual_seed(100))
    return cores, contract_tr_dense(cores)


class TestTRALSExact:  # MARK: TestTRALSExact

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_two_sweeps_match_direct_dense_oracle(self, dtype):
        _, tensor = _exact_tr(dtype)
        initial = make_tr_cores(
            in_dim=tensor.shape, rank=(2, 3, 2), dtype=dtype,
            generator=torch.Generator().manual_seed(101))
        expected = reference_tr_sweep(initial, tensor)
        order = range(tensor.ndim)
        for site in order:
            expected[site] = solve_local_core(
                expected, tensor, site, topology='tr')

        result = tk.decompositions.TRALS(tensor, out_device=None).fit(
            initial_cores=initial, gauge='none', renormalize=False,
            solver=tk.decompositions.LeastSquaresSolver(
                column_scaling=False, system_scaling=False),
            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=2))

        torch.testing.assert_close(
            result.contract_dense(),
            contract_tr_dense(expected),
            rtol=2e-10,
            atol=2e-10,
        )

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_one_sweep_matches_direct_dense_oracle(self, dtype):
        target_cores, tensor = _exact_tr(dtype)
        initial = make_tr_cores(
            in_dim=tensor.shape,
            rank=(2, 3, 2),
            dtype=dtype,
            generator=torch.Generator().manual_seed(101))
        expected = reference_tr_sweep(initial, tensor, qr=False)
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                initial_cores=initial,
                gauge='none',
                solver=tk.decompositions.LeastSquaresSolver(
                    column_scaling=False,
                    system_scaling=False),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
                renormalize=False)

        assert isinstance(result, tk.decompositions.TRDecomposition)
        torch.testing.assert_close(
            result.contract_dense(),
            contract_tr_dense(expected),
            rtol=2e-10,
            atol=2e-10,
        )
        assert result.rank == [2, 3, 2]
        assert target_cores[0].dtype == result.dtype

    @pytest.mark.parametrize('gauge', ['none', 'qr'])
    def test_exact_objective_is_monotone(self, gauge):
        _, tensor = _exact_tr()
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                rank=(2, 2, 2),
                gauge=gauge,
                generator=torch.Generator().manual_seed(102),
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=4),
                collect_metrics=True)

        errors = [record.abs_error for record in result.metrics.sweeps]
        for previous, current in zip(errors, errors[1:]):
            assert current <= previous + 2e-10 * max(1., previous)
        assert errors[-1] <= errors[0] + 2e-10

    def test_explicit_svd_result_can_initialize_als(self):
        _, tensor = _exact_tr()
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                rank=3,
                initial_cores=tk.decompositions.TRSVD(tensor).fit(rank=3),
                gauge='none',
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1))

        assert len(result.rank) == tensor.ndim
        assert all(value <= 3 for value in result.rank)
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)

    def test_fixed_core_is_unchanged_and_blocks_cyclic_absorption(self):
        initial, tensor = _exact_tr()
        fixed = initial[0].clone()
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                initial_cores=initial,
                fixed_cores=(fixed, None, None),
                gauge='qr',
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=2))

        assert torch.equal(result.cores[0], fixed)
        assert result.metadata['fixed_sites'] == [0]
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)

    def test_all_fixed_cores_stop_without_updates(self):
        initial, tensor = _exact_tr()
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                initial_cores=initial,
                fixed_cores=initial)

        assert result.metadata['converged']
        assert result.metadata['stop_reason'] == 'all_cores_fixed'
        assert result.metadata['n_sweeps'] == 0
        assert all(torch.equal(actual, expected)
                   for actual, expected in zip(result.cores, initial))

    def test_fast_path_omits_records_and_objective_reductions(self):
        _, tensor = _exact_tr()
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                rank=2,
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
                collect_metrics=False)

        assert result.metrics.local_solves == []
        assert result.metrics.sweeps == []


class TestTRALSSamplingAndCompletion:  # MARK: TestTRALSSamplingAndCompletion
    def test_uniform_samples_and_values_are_reused_by_generation(
        self, monkeypatch
    ):
        tensor = torch.arange(16., dtype=torch.float64).reshape(2, 2, 2, 2)
        evaluations = []

        def function(indices):
            evaluations.append(indices.clone())
            return tensor[tuple(indices[:, site]
                                for site in range(indices.shape[1]))]

        history = HistoryObserver()
        from importlib import import_module
        monkeypatch.setattr(
            import_module('tensorkrowch.decompositions.als.tr'),
            '_resolve_observer', lambda *args: history)
        result = tk.decompositions.TRALS(
            tk.decompositions.CallableTensorSource(
                function,
                in_dim=tensor.shape,
                dtype=torch.float64,
                out_shape=(),
            ),
            out_device=None,
        ).fit(
            rank=2,
            sampler=tk.decompositions.SamplingPolicy(
                12, method='uniform', reuse_sweeps=2
            ),
            generator=torch.Generator().manual_seed(103),
            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=3),
            collect_metrics=True,
            verbose=1,
        )

        assert len(evaluations) == 2
        assert [event.sweep for event in history.events
                if event.name == 'sample_refresh'] == [0, 2]
        assert [record.sample_generation
                for record in result.metrics.sweeps] == [0, 0, 1]
        assert all(record.abs_error is None
                   for record in result.metrics.sweeps)

    def test_completion_measures_only_permanent_observations(self):
        tensor = torch.arange(8., dtype=torch.float64).reshape(2, 2, 2)
        indices = torch.tensor([
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
        values = tensor[tuple(indices[:, site]
                              for site in range(indices.shape[1]))]
        decomposition = tk.decompositions.TRALS(
            tk.decompositions.SparseTensorSource(
                indices, values, tensor.shape
            ),
            mask=tk.decompositions.SparseTensorSource(
                indices,
                torch.ones(
                    indices.shape[0], dtype=torch.bool, device=indices.device
                ),
                tensor.shape,
            ),
            out_device=None,
        )
        result = decomposition.fit(
            rank=2,
            generator=torch.Generator().manual_seed(104),
            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=3),
            collect_metrics=True)

        approximation = result.evaluate(indices)
        absolute = torch.linalg.vector_norm(approximation - values)
        relative = absolute / torch.linalg.vector_norm(values)
        assert result.metrics.sweeps[-1].abs_error == pytest.approx(
            absolute.item())
        assert result.metrics.sweeps[-1].rel_error == pytest.approx(
            relative.item())
        assert result.metadata['sampling'] == 'observed'

    def test_product_leverage_redraws_current_approximation_per_site(self):
        tensor = torch.arange(8., dtype=torch.float64).reshape(2, 2, 2)
        evaluations = []

        def function(indices):
            evaluations.append(indices.clone())
            return tensor[tuple(indices[:, site]
                                for site in range(indices.shape[1]))]

        result = tk.decompositions.TRALS(
            tk.decompositions.CallableTensorSource(
                function,
                in_dim=tensor.shape,
                dtype=torch.float64,
                out_shape=(),
            ),
            out_device=None,
        ).fit(
            rank=2,
            gauge='none',
            sampler=tk.decompositions.SamplingPolicy(
                10, method='leverage', uniform_mix=0.2
            ),
            generator=torch.Generator().manual_seed(106),
            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=2),
            collect_metrics=True,
        )

        assert len(evaluations) == 2 * tensor.ndim
        for site, indices in enumerate(evaluations[:tensor.ndim]):
            fibers = indices.reshape(10, tensor.shape[site], tensor.ndim)
            assert torch.equal(
                fibers[:, :, site],
                torch.arange(tensor.shape[site]).expand(10, -1))
        assert len(result.metrics.local_solves) == 2 * tensor.ndim
        assert [record.sample_generation
                for record in result.metrics.local_solves] == [0] * 3 + [1] * 3
        assert all(record.sampling_exact is False
                   for record in result.metrics.local_solves)
        assert result.metadata['sampler']['uniform_mix'] == pytest.approx(0.2)
        assert all(record.abs_error is None
                   for record in result.metrics.sweeps)

    def test_product_leverage_requires_sitewise_refresh(self):
        decomposition = tk.decompositions.TRALS(torch.ones(2, 2, 2))

        with pytest.raises(ValueError, match='reuse_sweeps'):
            decomposition.fit(
                rank=2,
                sampler=tk.decompositions.SamplingPolicy(
                    5, method='leverage', reuse_sweeps=2))

    def test_exact_leverage_uses_fibers_and_current_conditional_design(self):
        tensor = torch.arange(16., dtype=torch.float64).reshape(2, 2, 2, 2)
        evaluations = []

        def function(indices):
            evaluations.append(indices.clone())
            return tensor[tuple(indices[:, site]
                                for site in range(indices.shape[1]))]

        result = tk.decompositions.TRALS(
            tk.decompositions.CallableTensorSource(
                function,
                in_dim=tensor.shape,
                dtype=torch.float64,
                out_shape=(),
            ),
            out_device=None,
        ).fit(
            rank=2,
            gauge='none',
            sampler=tk.decompositions.SamplingPolicy(
                6, method='exact_leverage'
            ),
            generator=torch.Generator().manual_seed(108),
            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1),
            collect_metrics=True,
        )

        assert len(evaluations) == tensor.ndim
        for site, indices in enumerate(evaluations):
            fibers = indices.reshape(6, tensor.shape[site], tensor.ndim)
            assert torch.equal(
                fibers[:, :, site],
                torch.arange(tensor.shape[site]).expand(6, -1))
        assert all(record.sampling_exact is True
                   for record in result.metrics.local_solves)
        assert result.metadata['sampler']['method'] == 'exact_leverage'
        assert result.metadata['n_samples'] == 6


class TestTRALSValidationAndWrapper:  # MARK: TestTRALSValidationAndWrapper

    def test_rank_sequence_uses_right_link_semantics(self):
        tensor = torch.randn(2, 3, 4, dtype=torch.float64)
        result = tk.decompositions.TRALS(
            tensor, out_device=None).fit(
                rank=(2, 3, 4),
                gauge='none',
                convergence=tk.decompositions.ConvergencePolicy(
                    max_sweeps=1))

        assert result.rank == [2, 3, 4]
        assert result.cores[0].shape == (4, 2, 2)
        assert result.cores[1].shape == (2, 3, 3)
        assert result.cores[2].shape == (3, 4, 4)

    def test_rank_length_and_cap_are_validated(self):
        tensor = torch.randn(2, 2, 2, dtype=torch.float64)
        initial = make_tr_cores(
            in_dim=tensor.shape,
            rank=(2, 2, 2),
            generator=torch.Generator().manual_seed(105))

        with pytest.raises(ValueError, match='one per link'):
            tk.decompositions.TRALS(tensor).fit(rank=(2, 2))
        with pytest.raises(ValueError, match='rank.*caps'):
            tk.decompositions.TRALS(tensor).fit(
                rank=1, initial_cores=initial)

    def test_wrapper_returns_result_with_optional_metrics(self):
        _, tensor = _exact_tr()
        result = tk.decompositions.tr_als(
            tensor, rank=2, max_sweeps=1, out_device=None, collect_metrics=True
        )
        assert isinstance(result, tk.decompositions.TRDecomposition)
        assert result.metadata['algorithm'] == 'tr_als'
        assert len(result.metrics.sweeps) == 1
        assert result.topology == 'tr'
        assert result.metadata['sampler'] is None


@pytest.mark.parametrize('gauge', ['none', 'qr'])
@pytest.mark.parametrize('quantized', [False, True])
def test_als_formats_devices_and_quantization(gauge,
                                              quantized,
                                              device_dtype,
                                              assert_close):
    engine = tk.decompositions.TRALS
    device, dtype = device_dtype
    data = torch.tensor([[[1., 2.], [2., 4.]], [[3., 6.], [6., 12.]]],
                        dtype=dtype, device=device)
    if dtype.is_complex:
        data = data * (1 + 1j)
    if quantized:
        real_dtype = data.real.dtype
        source = tk.decompositions.CallableQuanticsSource(
            lambda coordinates: torch.exp(coordinates[:, 0]).to(dtype) *
            ((1 + 1j) if dtype.is_complex else 1),
            1, base=2, level=3,
            domain=torch.tensor([0., 1.], dtype=real_dtype, device=device),
            dtype=dtype, device=device)
        problem = engine(source, out_device=None)
        data = source.contract_dense_grid_axes()
    else:
        problem = engine(data, out_device=None)
    options = dict(rank=1, gauge=gauge,
                   convergence=tk.decompositions.ConvergencePolicy(
                       max_sweeps=2),
                   collect_metrics=True)
    if (device == 'mps') and dtype.is_complex and (gauge == 'qr'):
        with pytest.raises(RuntimeError, match='geqrf.*float32'):
            problem.fit(**options)
        return
    result = problem.fit(**options)
    dense = (
        result.contract_dense_grid_axes()
        if quantized
        else result.contract_dense()
    )
    assert_close(dense, data.reshape(8) if quantized else data)
    assert (result.dtype == dtype) and (result.device.type == device)
    assert result.metrics.sweeps
    assert result.metrics.sweeps[-1].abs_error.device.type == 'cpu'
    assert result.metrics.sweeps[-1].abs_error < 5e-5 * data.norm()


@pytest.mark.parametrize('ordering', ['interleaved', 'grouped'])
def test_quantized_als_callable_and_repeated_fits(ordering):
    layout = tk.formats.QuantizedLayout(2, 2, 2, ordering=ordering)
    coordinate_map = tk.formats.AffineCoordinateMap(
        torch.tensor([[0.0, 1.0], [0.0, 1.0]], dtype=torch.float64),
        layout.grid_size,
    )
    source = tk.decompositions.CallableQuanticsSource(
        lambda coordinates: torch.exp(
            coordinates[:, 0] + 2 * coordinates[:, 1]),
        2, layout=layout, coordinate_map=coordinate_map)
    engine = tk.decompositions.TRALS(source, out_device=None)
    for _ in range(2):
        result = engine.fit(
            rank=1,
            convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1),
        )
        torch.testing.assert_close(
            result.contract_dense_grid_axes(),
            source.contract_dense_grid_axes(),
        )
        coordinates = torch.tensor(
            [[0., 0.], [0.75, 0.75]], dtype=torch.float64)
        torch.testing.assert_close(result.evaluate_coordinates(coordinates),
                                   source.function(coordinates))
        assert not result.metrics.sweeps
    other = tk.decompositions.CallableQuanticsSource(
        source.function,
        2,
        layout=tk.formats.QuantizedLayout(
            2,
            2,
            2,
            ordering='grouped' if ordering == 'interleaved' else 'interleaved',
        ),
        coordinate_map=coordinate_map,
    )
    with pytest.raises(ValueError, match='digit layout'):
        tk.decompositions.TRALS(other).fit(initial_cores=result)
    assert tk.decompositions.TRALS(torch.ones(4, 4)).source.is_discrete_tensor


def test_tensor_and_completion_axes_are_not_quantized():
    data = torch.arange(16., dtype=torch.float64).reshape(4, 4)
    engine = tk.decompositions.TRALS(data)
    assert engine.in_dim == (4, 4)
    indices = torch.tensor([[0, 0], [1, 3]])
    values = torch.tensor([1., 2.])
    weights = torch.tensor([2., 3.])
    completion = tk.decompositions.TRALS(
        tk.decompositions.SparseTensorSource(indices, values, (4, 4)),
        mask=tk.decompositions.SparseTensorSource(
            indices,
            torch.ones(
                indices.shape[0], dtype=torch.bool, device=indices.device
            ),
            (4, 4),
        ),
        weights=tk.decompositions.SparseTensorSource(
            indices, weights, (4, 4)
        ).contract_dense(),
        out_device='cpu',
    )
    observed = completion._problem.observations
    assert completion.in_dim == (4, 4)
    torch.testing.assert_close(observed.support.values, indices)
    torch.testing.assert_close(observed.support_values, values)
    torch.testing.assert_close(completion._problem.weights, weights)


def test_masked_ring_uses_only_participating_entries():
    tensor = torch.arange(8., dtype=torch.float64).reshape(2, 2, 2)
    mask = torch.ones(2, 2, 2, dtype=torch.bool)
    mask[1, 1, 1] = False
    engine = tk.decompositions.TRALS(tensor, mask=mask, out_device=None)
    result = engine.fit(
        rank=1,
        collect_metrics=True,
        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=2),
    )
    expected = torch.linalg.vector_norm(
        (result.contract_dense() - tensor)[mask])
    assert result.metrics.sweeps[-1].abs_error == pytest.approx(
        expected.item(), abs=1e-10)


def _ring_initialization(tensor):
    """Uses an independent TT-SVD representation with a unit closing bond."""
    result = tk.decompositions.TTSVD(tensor, out_device=None).fit(rank=2)
    if tensor.ndim == 1:
        cores = [result.cores[0].reshape(1, tensor.shape[0], 1)]
    else:
        cores = [result.cores[0].unsqueeze(0), *result.cores[1:-1],
                 result.cores[-1].unsqueeze(-1)]
    return tk.formats.TR(cores)


def test_tr_als_integrates_vector_source_families(vector_source_case):
    source, tensor = vector_source_case
    initial = _ring_initialization(tensor)
    result = tk.decompositions.TRALS(source, out_device=None).fit(
        initial_cores=initial,
        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=2),
        collect_metrics=True)
    torch.testing.assert_close(result.contract_dense(), tensor,
                               rtol=1e-10, atol=1e-10)
    assert result.metadata['sweep_direction'] == 'forward'
    assert len(result.metrics.local_solves) == 2 * tensor.ndim


def test_tr_als_integrates_matrix_source_families(matrix_source_case):
    source, tensor = matrix_source_case
    fused = tuple(i * o for i, o in zip(source.in_dim, source.out_dim))
    initial = _ring_initialization(tensor.reshape(fused))
    engine = tk.decompositions.TRALS(source, out_device=None)
    result = engine.fit(
        initial_cores=initial,
        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=2))
    result = engine.fit(
        initial_cores=result,
        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
    assert isinstance(result, tk.decompositions.TRMDecomposition)
    assert result.in_dim == source.in_dim
    assert result.out_dim == source.out_dim
    torch.testing.assert_close(result.contract_dense(), tensor,
                               rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('gauge', ['none', 'qr'])
def test_perturbed_nontrivial_ring_initialization_recovers_dense_target(
    dtype, gauge
):
    generator = torch.Generator().manual_seed(148)
    cores = make_tr_cores((3, 3, 3, 3), (2, 2, 2, 2), dtype, generator)
    tensor = contract_tr_dense(cores)
    initial = [core + 0.01 * torch.randn(core.shape, dtype=dtype,
                                       generator=generator)
               for core in cores]
    result = tk.decompositions.TRALS(tensor, out_device=None).fit(
        initial_cores=initial, gauge=gauge,
        convergence=tk.decompositions.ConvergencePolicy(
            max_sweeps=60, rtol=1e-9),
        collect_metrics=True)
    relative = (result.contract_dense() - tensor).norm() / tensor.norm()
    assert relative < 1e-6
    assert result.metrics.sweeps[-1].rel_error == pytest.approx(
        relative.item())


@pytest.mark.parametrize('method', ['uniform', 'leverage', 'exact_leverage'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_sampled_random_initialization_recovers_separable_ring(method, dtype):
    factors = [torch.tensor([1., 2., -1.], dtype=dtype),
               torch.tensor([2., -3.], dtype=dtype),
               torch.tensor([1., 4., 2.], dtype=dtype)]
    if dtype.is_complex:
        factors[1] = factors[1] * (1 + 2j)
    tensor = torch.einsum('i,j,k->ijk', *factors)
    result = tk.decompositions.TRALS(tensor, out_device=None).fit(
        rank=1,
        sampler=tk.decompositions.SamplingPolicy(128, method=method),
        generator=torch.Generator().manual_seed(149),
        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=20))
    relative = (result.contract_dense() - tensor).norm() / tensor.norm()
    assert relative < 1e-8


@pytest.mark.parametrize('fixed_sites', [(1, 3), (0, 2, 4)])
def test_nonadjacent_fixed_ring_cores_remain_unchanged(fixed_sites):
    generator = torch.Generator().manual_seed(150)
    initial = make_tr_cores((2, 3, 2, 3, 2), (2, 2, 2, 2, 2),
                            torch.complex128, generator)
    target = contract_tr_dense(initial)
    fixed = [core.clone() if site in fixed_sites else None
             for site, core in enumerate(initial)]
    result = tk.decompositions.TRALS(target, out_device=None).fit(
        initial_cores=initial, fixed_cores=fixed,
        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=2))
    for site in fixed_sites:
        assert torch.equal(result.cores[site], fixed[site])
    torch.testing.assert_close(result.contract_dense(), target,
                               rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('rank, error', [
    (None, ValueError), (True, TypeError), (0, ValueError),
    ((1, 2), ValueError), ((1, False, 2), TypeError),
    ((1, 0, 2), ValueError), ('2', TypeError),
])
def test_ring_rank_validation(rank, error):
    with pytest.raises(error):
        tk.decompositions.TRALS(torch.ones(2, 3, 2)).fit(rank=rank)


@pytest.mark.parametrize(
    'kwargs, error',
    [
        ({'gauge': 'svd'}, ValueError),
        ({'gauge': True}, TypeError),
        ({'renormalize': 1}, TypeError),
        ({'collect_metrics': 1}, TypeError),
        ({'generator': 1}, TypeError),
        ({'solver': object()}, TypeError),
        ({'convergence': object()}, TypeError),
        ({'sampler': 'uniform'}, TypeError),
        ({'fixed_cores': [None]}, ValueError),
        ({'fixed_cores': [None, object(), None]}, TypeError),
        ({'fixed_cores': [torch.ones(1, 2, 2), None, None]}, ValueError),
        (
            {
                'fixed_cores': [
                    None,
                    torch.ones(2, 3, 2, dtype=torch.float64),
                    None,
                ]
            },
            ValueError,
        ),
    ],
)
def test_ring_fit_argument_validation(kwargs, error):
    with pytest.raises(error):
        tk.decompositions.TRALS(torch.ones(2, 3, 2)).fit(rank=2, **kwargs)


@pytest.mark.parametrize('initial', [
    [], [torch.ones(1, 2, 1)], 'cores',
    [torch.ones(2, 2, 3), torch.ones(2, 3, 2), torch.ones(2, 2, 2)],
    [torch.ones(2, 2), torch.ones(2, 3, 2), torch.ones(2, 2, 2)],
    [torch.ones(2, 3, 2), torch.ones(2, 3, 2), torch.ones(2, 2, 2)],
    [torch.ones(2, 2, 2, dtype=torch.float64),
     torch.ones(2, 3, 2, dtype=torch.float64),
     torch.ones(2, 2, 2, dtype=torch.float64)],
])
def test_ring_initial_core_validation(initial):
    with pytest.raises((TypeError, ValueError)):
        tk.decompositions.TRALS(torch.ones(2, 3, 2)).fit(initial_cores=initial)


def test_ring_rejects_open_topology_initial_format():
    initial = tk.formats.TT([torch.ones(2, 2), torch.ones(2, 3)])
    with pytest.raises(ValueError, match='cyclic'):
        tk.decompositions.TRALS(torch.ones(2, 3)).fit(initial_cores=initial)


def test_observation_objective_rejects_renewable_sampling():
    indices = torch.tensor([[0, 0], [1, 2]])
    source = tk.decompositions.SparseTensorSource(
        indices, torch.ones(2), (2, 3))
    mask = tk.decompositions.SparseTensorSource(
        indices, torch.ones(2, dtype=torch.bool), (2, 3))
    with pytest.raises(ValueError, match='observations'):
        tk.decompositions.TRALS(source, mask=mask).fit(
            rank=1, sampler=tk.decompositions.SamplingPolicy(4))


def test_wrapper_and_engine_agree_with_shared_initialization():
    initial, target = _exact_tr(torch.complex128)
    engine_result = tk.decompositions.TRALS(target, out_device=None).fit(
        initial_cores=initial,
        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=2),
        collect_metrics=True)
    wrapped = tk.decompositions.tr_als(
        target, initial_cores=initial, max_sweeps=2, collect_metrics=True,
        out_device=None)
    torch.testing.assert_close(wrapped.contract_dense(),
                               engine_result.contract_dense())
    assert wrapped.metadata['n_sweeps'] == engine_result.metadata['n_sweeps']
    assert len(wrapped.metrics.local_solves) == len(
        engine_result.metrics.local_solves)


@pytest.mark.parametrize('n_sites', [1, 2])
@pytest.mark.parametrize('criterion', ['atol', 'rtol'])
def test_wrapper_minimal_ring_stops_on_objective_tolerance(n_sites, criterion):
    tensor = torch.arange(1., 4., dtype=torch.float64)
    if n_sites == 2:
        tensor = tensor[:, None] * torch.tensor([2., -1.])
    result = tk.decompositions.tr_als(
        tensor, rank=1, max_sweeps=5, collect_metrics=True,
        generator=torch.Generator().manual_seed(153), **{criterion: 1e-10})
    torch.testing.assert_close(result.contract_dense(), tensor,
                               rtol=1e-10, atol=1e-10)
    assert result.metadata['stop_reason'] == criterion
    assert result.metadata['n_sweeps'] < 5
