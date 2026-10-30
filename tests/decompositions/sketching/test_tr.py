"""Tests for Tensor Ring recursive sketching from samples."""

import math

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.observers import HistoryObserver
from tensorkrowch.decompositions.sketching.tr import (
    _SketchLocalTarget, _SketchLoopOpener)
from tensorkrowch.utils import truncated_svd


def _rank_one_problem(n_sites=4):
    domain = torch.tensor([0., 1.], dtype=torch.float64)
    samples = torch.cartesian_prod(*(domain for _ in range(n_sites)))

    def function(data):
        return (1 + data).prod(dim=1)

    def embedding(values):
        return torch.stack((1 - values, values), dim=-1)

    return function, embedding, samples, domain


class TestSketchLoopOpener:  # MARK: TestSketchLoopOpener

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('fixed_side', [None, 'left', 'right'])
    @pytest.mark.parametrize('physical_shape', [(4,), (4, 3)])
    @pytest.mark.parametrize('absorb_singular_values', [True, False])
    def test_sketch_trimming_objective_and_fixed_gauges(
            self, dtype, fixed_side, physical_shape, absorb_singular_values):
        generator = torch.Generator().manual_seed(73)

        def random(*shape):
            return torch.randn(*shape, dtype=dtype, generator=generator)

        left, _ = torch.linalg.qr(random(6, 3))
        right, _ = torch.linalg.qr(random(7, 3))
        spectrum = torch.tensor([1., 0.1, 0.01], dtype=dtype)
        core = random(3, math.prod(physical_shape), 3)
        core = spectrum[:, None, None] * core * spectrum[None, None, :]
        tensor = torch.einsum('al,lpr,rb->apb', left, core, right.mH)
        tensor = tensor.reshape(6, *physical_shape, 7)
        fixed_left = random(2, 6, 2) if fixed_side == 'left' else None
        fixed_right = random(2, 7, 2) if fixed_side == 'right' else None

        class RecordingOpener:
            capabilities = tk.decompositions.LoopOpenerCapabilities(
                supports_fixed_left=True, supports_fixed_right=True,
                supports_blocks=True)

            def open(self, target, rank, *, fixed_left=None, fixed_right=None,
                     orientation='right', context=None):
                self.target = target
                cores = [0.25 * random(rank[i - 1], size, rank[i])
                         for i, size in enumerate(target.shape)]
                if fixed_left is not None:
                    cores[0] = fixed_left
                if fixed_right is not None:
                    cores[-1] = fixed_right
                self.approximation = tk.formats.TR(cores).contract_dense()
                return tk.decompositions.LoopOpening(
                    left_gauge=cores[0], cores=tuple(cores[1:-1]),
                    right_gauge=cores[-1], rank=tuple(rank),
                    orientation=orientation)

        delegate = RecordingOpener()
        target = _SketchLocalTarget(
            tensor, tuple(range(1, len(physical_shape) + 1)),
            {'rtol': 1e-14}, True)
        result = _SketchLoopOpener(
            delegate, [], absorb_singular_values).open(
            target, [2] * tensor.ndim,
            fixed_left=fixed_left, fixed_right=fixed_right,
            orientation='left' if fixed_side == 'right' else 'right')

        expected_shape = (
            6 if fixed_left is not None else 3, *physical_shape,
            7 if fixed_right is not None else 3)
        assert delegate.target.shape == expected_shape
        compact_error = (delegate.target - delegate.approximation).norm()
        lifted_error = (tensor - result.contract_dense()).norm()
        if absorb_singular_values:
            assert torch.allclose(delegate.target.norm(), tensor.norm(),
                                  rtol=1e-12, atol=1e-12)
            assert torch.allclose(compact_error, lifted_error,
                                  rtol=1e-12, atol=1e-12)
        else:
            if fixed_left is None:
                matrix = delegate.target.reshape(3, -1)
                gram = matrix @ matrix.mH
            else:
                matrix = delegate.target.reshape(-1, 3)
                gram = matrix.mH @ matrix
            torch.testing.assert_close(gram, torch.eye(3, dtype=dtype))
            assert not torch.allclose(compact_error, lifted_error)
        if fixed_left is not None:
            assert torch.equal(result.left_gauge, fixed_left)
        if fixed_right is not None:
            assert torch.equal(result.right_gauge, fixed_right)

    @pytest.mark.parametrize('absorb_singular_values', [True, False])
    def test_sketch_spectra_are_restored_before_returning_the_opening(
            self, absorb_singular_values):
        generator = torch.Generator().manual_seed(17)
        tensor = torch.randn(5, 4, 6, dtype=torch.complex128,
                             generator=generator)
        # A rank-three TT with cyclic rank one supplies an exact local opening.
        tensor = tk.decompositions.TTSVD(tensor).fit(rank=3).contract_dense()

        class ExactOpener:
            capabilities = tk.decompositions.LoopOpenerCapabilities()

            def open(self, target, rank, **kwargs):
                train = tk.decompositions.TTSVD(target).fit(rank=3)
                cores = train._effective_cores()
                return tk.decompositions.LoopOpening(
                    left_gauge=cores[0], cores=cores[1:-1],
                    right_gauge=cores[-1], rank=rank)

        result = _SketchLoopOpener(
            ExactOpener(), [], absorb_singular_values).open(
                _SketchLocalTarget(tensor, (1,), {'rtol': 1e-14}, False),
                (3, 3, 1))
        torch.testing.assert_close(result.contract_dense(), tensor)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('fixed_side', [None, 'left', 'right'])
    @pytest.mark.parametrize('physical_shape', [(4,), (4, 3)])
    @pytest.mark.parametrize('absorb_singular_values', [True, False])
    @pytest.mark.parametrize('project_fixed_gauges', [False, True])
    def test_sketch_refinement_restores_metric_and_preserves_fixed_gauges(
            self, dtype, fixed_side, physical_shape, absorb_singular_values,
            project_fixed_gauges):
        generator = torch.Generator().manual_seed(83)

        def random(*shape):
            return torch.randn(*shape, dtype=dtype, generator=generator)

        left, _ = torch.linalg.qr(random(6, 3))
        right, _ = torch.linalg.qr(random(7, 3))
        spectrum = torch.tensor([1., .1, .01], dtype=dtype)
        center = spectrum[:, None, None] * random(
            3, math.prod(physical_shape), 3) * spectrum[None, None, :]
        tensor = torch.einsum('al,lpr,rb->apb', left, center, right.mH)
        tensor = tensor.reshape(6, *physical_shape, 7)
        fixed_left = random(2, 6, 2) if fixed_side == 'left' else None
        fixed_right = random(2, 7, 2) if fixed_side == 'right' else None
        target = _SketchLocalTarget(tensor, (1,), {'rtol': 1e-14}, True)
        results = []
        for sweeps in (0, 1):
            opener = tk.decompositions.ALSLoopOpener({
                'gauge': 'none',
                'generator': torch.Generator().manual_seed(91),
                'convergence': tk.decompositions.ConvergencePolicy(
                    max_sweeps=2)})
            results.append(_SketchLoopOpener(
                opener, [], absorb_singular_values, sweeps,
                project_fixed_gauges).open(
                target, [2] * tensor.ndim,
                fixed_left=fixed_left, fixed_right=fixed_right,
                orientation='left' if fixed_side == 'right' else 'right'))
        before, refined = results
        initial_error = (tensor - before.contract_dense()).norm()
        initial_error = initial_error / tensor.norm()
        final_error = (tensor - refined.contract_dense()).norm()
        final_error = final_error / tensor.norm()
        assert final_error <= initial_error + 1e-12
        info = refined.diagnostics['sketch_refinement']
        torch.testing.assert_close(
            info['relative_error_before'], initial_error)
        torch.testing.assert_close(info['relative_error_after'], final_error)
        assert info['relative_error_after'] <= info['relative_error_before']
        assert info['metadata']['n_sweeps'] == 1
        assert info['metadata']['sampling'] == 'exact'
        assert info['metadata']['sweep_direction'] == 'forward'
        assert refined.rank == before.rank
        if fixed_left is not None:
            assert torch.equal(refined.left_gauge, fixed_left)
        if fixed_right is not None:
            assert torch.equal(refined.right_gauge, fixed_right)

    def test_sketch_refinement_rejects_a_worse_candidate(self):
        tensor = torch.randn(3, 4, 3, dtype=torch.complex128,
                             generator=torch.Generator().manual_seed(18))
        opener = _SketchLoopOpener(
            tk.decompositions.ALSLoopOpener({
                'gauge': 'none',
                'convergence': tk.decompositions.ConvergencePolicy(
                    max_sweeps=1)}),
            [], False, 1)

        class WorseRefiner:
            def open(self, tensor, rank, *, context, **kwargs):
                self.initial = tk.decompositions.LoopOpening(
                    context['initial_cores'][0],
                    context['initial_cores'][1:-1],
                    context['initial_cores'][-1], rank)
                return tk.decompositions.LoopOpening(
                    self.initial.left_gauge,
                    tuple(100 * core for core in self.initial.cores),
                    self.initial.right_gauge, rank,
                    diagnostics={'metadata': {'n_sweeps': 1}})

        refiner = WorseRefiner()
        opener.refiner = refiner
        result = opener.open(
            _SketchLocalTarget(tensor, (1,), {}, False), (2, 2, 2))
        assert result.diagnostics['sketch_refinement']['accepted'] is False
        torch.testing.assert_close(result.cores[0], refiner.initial.cores[0])
        info = result.diagnostics['sketch_refinement']
        assert info['relative_error_before'] == info['relative_error_after']

    def test_sketch_refinement_handles_a_zero_target(self):
        tensor = torch.zeros(3, 4, 3, dtype=torch.complex128)
        result = _SketchLoopOpener(
            tk.decompositions.ALSLoopOpener(), [], False, 1).open(
                _SketchLocalTarget(tensor, (1,), {'rtol': 1e-14}, False),
                (2, 2, 2))
        torch.testing.assert_close(result.contract_dense(), tensor)
        info = result.diagnostics['sketch_refinement']
        assert info['relative_error_before'] == 0
        assert info['relative_error_after'] == 0

    def test_sketch_refinement_keeps_the_retained_sketch_subspaces(self):
        tensor = torch.randn(6, 4, 7, dtype=torch.complex128,
                             generator=torch.Generator().manual_seed(31))
        u, _, vh = truncated_svd(tensor.reshape(-1, 7), rank=4)
        left, _, _ = truncated_svd(u.reshape(6, -1), rank=4)
        results = []
        for sweeps in (0, 1):
            opener = tk.decompositions.ALSLoopOpener({
                'gauge': 'none',
                'generator': torch.Generator().manual_seed(27),
                'convergence': tk.decompositions.ConvergencePolicy(
                    max_sweeps=1)})
            results.append(_SketchLoopOpener(opener, [], False, sweeps).open(
                _SketchLocalTarget(tensor, (1,), {}, False), (2, 2, 2)))
        before, refined = results
        assert (tensor - refined.contract_dense()).norm() <= \
            (tensor - before.contract_dense()).norm() + 1e-12
        left_gauge = refined.left_gauge.permute(1, 0, 2).reshape(6, -1)
        right_gauge = refined.right_gauge.permute(0, 2, 1).reshape(-1, 7)
        torch.testing.assert_close(left @ left.mH @ left_gauge, left_gauge)
        torch.testing.assert_close(right_gauge @ vh.mH @ vh, right_gauge)


class TestTRRSS:  # MARK: TestTRRSS

    @pytest.mark.parametrize('absorb_singular_values', [True, False])
    @pytest.mark.parametrize('refinement_sweeps', [0, 1])
    def test_spectrum_placement_reaches_public_callers(
            self, absorb_singular_values, refinement_sweeps):
        function, embedding, samples, domain = _rank_one_problem()
        result = tk.decompositions.tr_rss(
            function, embedding, samples, domain=domain, rank=1,
            absorb_singular_values=absorb_singular_values,
            refinement_sweeps=refinement_sweeps,
            return_result=True)
        assert result.metadata['absorb_singular_values'] is \
            absorb_singular_values
        assert result.metadata['refinement_sweeps'] == refinement_sweeps
        torch.testing.assert_close(
            result.contract_dense(), function(samples).reshape(2, 2, 2, 2))

    @pytest.mark.parametrize('invalid', [None, 0, 'outside'])
    def test_invalid_spectrum_placement_is_rejected(self, invalid):
        function, embedding, samples, domain = _rank_one_problem()
        with pytest.raises(TypeError, match='absorb_singular_values'):
            tk.decompositions.TRRSS(
                function, embedding, domain=domain).fit(
                    samples, absorb_singular_values=invalid)

    @pytest.mark.parametrize('project_fixed_gauges', [False, True])
    @pytest.mark.parametrize('refinement_sweeps', [0, 1])
    def test_two_sided_projection_reaches_public_api(
            self, project_fixed_gauges, refinement_sweeps):
        function, embedding, samples, domain = _rank_one_problem(5)
        result = tk.decompositions.tr_rss(
            function, embedding, samples, domain=domain, rank=1,
            absorb_singular_values=False,
            project_fixed_gauges=project_fixed_gauges,
            refinement_sweeps=refinement_sweeps, return_result=True)
        assert result.metadata['project_fixed_gauges'] is project_fixed_gauges
        torch.testing.assert_close(
            result.contract_dense(), function(samples).reshape((2,) * 5))

    def test_invalid_fixed_gauge_projection(self):
        function, embedding, samples, domain = _rank_one_problem()
        with pytest.raises(TypeError, match='project_fixed_gauges'):
            tk.decompositions.TRRSS(function, embedding, domain=domain).fit(
                samples, project_fixed_gauges='both')

    @pytest.mark.parametrize('invalid', [True, None, 1.5, '1', -1])
    def test_invalid_refinement_sweeps_is_rejected(self, invalid):
        function, embedding, samples, domain = _rank_one_problem()
        error = ValueError if invalid == -1 else TypeError
        with pytest.raises(error, match='refinement_sweeps'):
            tk.decompositions.TRRSS(
                function, embedding, domain=domain).fit(
                    samples, refinement_sweeps=invalid)

    def test_refinement_metrics_are_exposed(self):
        function, embedding, samples, domain = _rank_one_problem(n_sites=5)
        result = tk.decompositions.TRRSS(
            function, embedding, domain=domain).fit(
                samples, rank=1, absorb_singular_values=False,
                refinement_sweeps=1, collect_metrics=True)
        refinements = result.metadata['local_refinements']
        # The center and both recursive openings refine before propagation.
        assert len(refinements) == 3
        assert len(result.metrics.truncations) == 4
        for info in refinements:
            assert info['n_sweeps'] == 1
            assert info['relative_error_after'] <= info['relative_error_before']

    def test_rank_one_function_matches_dense_tensor(self):
        function, embedding, samples, domain = _rank_one_problem()
        result = tk.decompositions.TRRSS(embedding=embedding, domain=domain, source=function).fit(
                samples,
                rank=1,
                collect_metrics=True)

        expected = function(samples).reshape(2, 2, 2, 2)
        assert isinstance(result, tk.decompositions.TRDecomposition)
        assert result.rank == [1, 1, 1, 1]
        assert torch.allclose(
            result.contract_dense(), expected, rtol=1e-8, atol=1e-10)
        assert result.metrics.errors[0].kind == 'sketch_samples'
        assert result.metrics.errors[0].relative < 1e-8
        assert len(result.metrics.truncations) == 3

    def test_repeated_fits_are_independent_and_functional_api_is_compatible(
            self):
        function, embedding, samples, domain = _rank_one_problem()
        decomposer = tk.decompositions.TRRSS(embedding=embedding, domain=domain, source=function)
        first = decomposer.fit(samples, rank=1)
        second = decomposer.fit(samples, rank=1)
        cores, info = tk.decompositions.tr_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=1,
            return_info=True)

        assert first is not second
        assert all(left is not right for left, right in zip(
            first.cores, second.cores))
        assert len(cores) == 4
        assert info['topology'] == 'tr'
        assert info['metadata']['requested_rank'] == [1, 1, 1, 1]

    def test_tensor_outputs_use_basis_sites_and_flat_labels(self):
        function, embedding, samples, domain = _rank_one_problem()

        def tensor_function(data):
            value = function(data)
            left = torch.tensor([1., 2.], dtype=data.dtype)
            right = torch.tensor([1., 3.], dtype=data.dtype)
            return value[:, None, None] * left[None, :, None] * \
                right[None, None, :]

        labels = torch.arange(samples.shape[0]).remainder(4)
        result = tk.decompositions.TRRSS(embedding=embedding, domain=domain, out_position=(1, 4), source=tensor_function).fit(
                samples,
                labels=labels,
                rank=1,
                collect_metrics=True)

        assert result.in_dim == (2, 2, 2, 2, 2, 2)
        assert result.metadata['out_position'] == (1, 4)
        assert result.metrics.errors[0].relative < 1e-8

    def test_adaptive_block_discovers_ranks_and_padding_is_opt_in(self):
        function, embedding, samples, domain = _rank_one_problem(n_sites=5)
        decomposer = tk.decompositions.TRRSS(embedding=embedding, domain=domain, source=function)
        adaptive = decomposer.fit(
            samples,
            rank=(2, 2, 2, 2, 2),
            adaptive=True,
            collect_metrics=True)
        padded = decomposer.fit(
            samples,
            rank=2,
            adaptive=True,
            pad_to_rank=True)

        assert adaptive.rank == [1, 1, 1, 1, 1]
        assert adaptive.metadata['center_block'] == (1, 2, 3)
        assert adaptive.metadata['rank_estimate']['limitations'] == ()
        assert adaptive.metrics.errors[0].relative < 1e-8
        assert padded.rank == [2, 2, 2, 2, 2]
        assert torch.allclose(
            adaptive.contract_dense(), padded.contract_dense(),
            rtol=1e-10, atol=1e-12)

    def test_alternating_schedule_solves_odd_ring_in_two_stages(self):
        function, embedding, samples, domain = _rank_one_problem(n_sites=5)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRSS(embedding=embedding, domain=domain, source=function).fit(
                    samples,
                    rank=1,
                    schedule='alternating',
                    collect_metrics=True)

        assert result.metadata['schedule'] == 'alternating'
        assert result.metadata['requested_schedule'] == 'alternating'
        assert result.metadata['center_block'] == (2,)
        assert result.metrics.errors[0].relative < 1e-8

    def test_alternating_schedule_records_even_ring_fallback(self):
        function, embedding, samples, domain = _rank_one_problem(n_sites=4)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRSS(embedding=embedding, domain=domain, source=function).fit(
                    samples,
                    rank=1,
                    schedule='alternating')

        assert result.metadata['schedule'] == 'center_out'
        assert result.metadata['requested_schedule'] == 'alternating'
        assert torch.allclose(
            result.contract_dense(), function(samples).reshape(2, 2, 2, 2),
            rtol=1e-8, atol=1e-10)


__all__ = []


def _physical_grid(layout, coordinate_map, domain):
    variable_indices = torch.cartesian_prod(*(
        torch.arange(size) for size in layout.grid_size))
    if layout.n_coordinates == 1:
        variable_indices = variable_indices.reshape(-1, 1)
    physical = coordinate_map.from_indices(variable_indices)
    return variable_indices, physical


class TestQTRRSS:  # MARK: TestQTRRSS

    @pytest.mark.parametrize('absorb_singular_values', [True, False])
    @pytest.mark.parametrize('refinement_sweeps', [0, 1])
    def test_qtr_functional_and_class_apis_use_ring_driver(
            self, absorb_singular_values, refinement_sweeps):
        layout = tk.formats.QuantizedLayout(1, base=2, level=3)
        coordinate_map = tk.formats.AffineCoordinateMap(
            torch.tensor([0., 1.]), layout.grid_size, grid_offset="endpoints")
        indices, physical = _physical_grid(
            layout, coordinate_map, torch.tensor([0., 1.]))

        source = tk.decompositions.CallableQuanticsSource(
            lambda values: torch.ones_like(values[:, 0]),
            layout.n_coordinates,
            layout=layout,
            coordinate_map=coordinate_map)
        cores, info = tk.decompositions.qtr_rss(
            source,
            physical,
            rank=1,
            absorb_singular_values=absorb_singular_values,
            refinement_sweeps=refinement_sweeps,
            return_info=True)
        result = tk.decompositions.TRDecomposition(cores)

        assert result.rank == [1, 1, 1]
        assert torch.allclose(
            result.evaluate(layout.encode_indices(indices)),
            torch.ones(indices.shape[0]))
        assert info['metadata']['algorithm'] == 'qtr_rss'
        assert info['metadata']['absorb_singular_values'] is \
            absorb_singular_values
        assert info['metadata']['refinement_sweeps'] == refinement_sweeps
        assert info['metadata']['quantization']['n_coordinates'] == 1


def _rank_one_sparse(dtype=torch.float64, in_sites=(0, 1, 2)):
    factors = [
        torch.tensor([1., 2.], dtype=dtype),
        torch.tensor([1., 3.], dtype=dtype),
        torch.tensor([2., 1.], dtype=dtype),
    ]
    dense = torch.einsum('i,j,k->ijk', *factors)
    out_sites = tuple(site for site in range(3) if site not in in_sites)
    indices = torch.cartesian_prod(*(
        torch.arange(dense.shape[site]) for site in in_sites))
    values = dense.permute(*in_sites, *out_sites).reshape(
        indices.shape[0], *(dense.shape[site] for site in out_sites))
    source = tk.decompositions.SparseTensorSource(
        indices, values, dense.shape, in_sites=in_sites)
    return source, dense


class TestTRRS:  # MARK: TestTRRS

    @pytest.mark.parametrize('in_sites', [(0, 1, 2), (2, 0)])
    @pytest.mark.parametrize('recursion', ['adjoint', 'pinv', 'orth'])
    @pytest.mark.parametrize('expanded', [False, True])
    @pytest.mark.parametrize('operator', [
        tk.decompositions.SampledSketch(),
        tk.decompositions.MarginalSketch.markov(),
        tk.decompositions.TTStackSketch(tt_rank=2, stack_size=2),
    ])
    def test_sparse_rank_one_source_is_recovered(
            self, operator, recursion, expanded, in_sites):
        source, dense = _rank_one_sparse(in_sites=in_sites)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRS(
                source,
                sketch_operator=operator,
                out_device=None).fit(
                    rank=1,
                    recursion=recursion,
                    expand_directions=expanded,
                    generator=torch.Generator().manual_seed(311),
                    strict_system=True,
                    collect_metrics=True)

        assert result.rank == [1, 1, 1]
        assert torch.allclose(
            result.contract_dense(), dense, rtol=1e-9, atol=1e-11)
        assert result.metrics.fidelities[0].fidelity > 1 - 1e-10
        assert result.metrics.errors[-1].kind == 'source_support'
        assert result.metrics.errors[-1].relative < 1e-9
        assert result.metadata['algorithm'] == 'tr_rs'
        assert result.metadata['recursion'] == recursion
        assert result.metadata['expand_directions'] == expanded
        assert result.metadata['experimental'] is True

    def test_dataset_functional_api_returns_information(self):
        dataset = torch.tensor([
            [0, 0, 0], [0, 0, 0], [1, 1, 1]])
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            cores, info = tk.decompositions.tr_rs(tk.decompositions.SparseTensorSource.from_data(dataset, in_dim=(2, 2, 2)), rank=1, return_info=True)

        result = tk.decompositions.TRDecomposition(cores)
        assert result.rank == [1, 1, 1]
        assert info['metadata']['source_type'] == 'SparseTensorSource'
        assert info['metrics']['fidelities'][0]['fidelity'] > 1 - 1e-5

    def test_tt_source_uses_structured_contractions(self):
        _, dense = _rank_one_sparse()
        tt = tk.decompositions.TTSVD(
            dense, out_device=None).fit(rank=1)
        source = tk.decompositions.FormatSource(tt)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRS(
                source, out_device=None).fit(rank=1)

        assert torch.allclose(result.contract_dense(), dense)
        assert result.metrics.fidelities[0].fidelity > 1 - 1e-10
        assert source.evaluation_stats.requested_points == 0

    def test_history_observer_receives_one_tr_rs_hierarchy(self):
        source, _ = _rank_one_sparse()
        observer = HistoryObserver()
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = tk.decompositions.TRRS(source).fit(
                rank=1, observer=observer)

        assert observer.metrics is result.metrics
        assert observer.events[0].name == 'start'
        assert all(event.phase == 'TR-RS' for event in observer.events)
        assert sum(event.name == 'site_complete'
                   for event in observer.events) == 3

    def test_rank_and_warm_start_validation(self):
        source, _ = _rank_one_sparse()
        with pytest.raises(TypeError, match='rank'):
            tk.decompositions.TRRS(source).fit(rank=[1, 1, 1])
        initial = tk.decompositions.TRDecomposition([
            torch.ones(1, 2, 1) for _ in range(3)])
        with pytest.raises(NotImplementedError, match='warm-start'):
            tk.decompositions.TRRS(source).fit(
                rank=1, warm_start=initial)


@pytest.mark.parametrize('quantized', [False, True])
def test_rss_function_formats_devices_and_sample_error(quantized,
                                                       device_dtype,
                                                       assert_close):
    engine = tk.decompositions.TRRSS
    device, dtype = device_dtype
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    domain = torch.tensor([0., 1.], dtype=real_dtype, device=device)
    coordinates = torch.cartesian_prod(domain, domain, domain)
    if quantized:
        coordinates = torch.arange(8, dtype=real_dtype, device=device).reshape(-1, 1) / 8
    def function(values):
        result = torch.exp(values.sum(-1)).to(dtype)
        return result * (1 + 1j) if dtype.is_complex else result
    def embedding(values):
        return tk.embeddings.basis(values.to(torch.long), dim=2).to(dtype)
    if quantized:
        layout = tk.formats.QuantizedLayout(1, 2, 3)
        coordinate_map = tk.formats.AffineCoordinateMap(
            torch.tensor([0., 1.], dtype=real_dtype, device=device), layout.grid_size)
        source = tk.decompositions.CallableQuanticsSource(
            function,
            layout.n_coordinates,
            layout=layout,
            coordinate_map=coordinate_map,
            device=device,
            dtype=dtype)
        problem = engine(source=
            source,
            out_device=None)
    else:
        problem = engine(function, embedding, domain=domain, device=device,
                         dtype=dtype, out_device=None)
    result = problem.fit(coordinates, rank=1, collect_metrics=True)
    if quantized:
        actual = result.evaluate_coordinates(coordinates)
        error = result.error(
            function, coordinates, data=layout.encode_indices(
                coordinate_map.to_indices(coordinates)))
    else:
        inputs = embedding(coordinates)
        actual = result.evaluate(inputs)
        error = result.error(function, coordinates, data=inputs)
    assert_close(actual, function(coordinates))
    assert isinstance(error, tk.decompositions.ErrorRecord)
    assert error.relative < 5e-5
    assert result.dtype == dtype and result.device.type == device


@pytest.mark.parametrize('out_shape', [(2,), (2, 2)])
def test_quantized_rss_preserves_tensor_outputs(out_shape):
    layout = tk.formats.QuantizedLayout(1, base=2, level=3)
    physical = torch.arange(8, dtype=torch.float64).reshape(-1, 1) / 7

    def function(values):
        return torch.ones(values.shape[0], *out_shape, dtype=values.dtype,
                          device=values.device)

    source = tk.decompositions.CallableQuanticsSource(
        function,
        layout.n_coordinates,
        layout=layout,
        coordinate_map=tk.formats.AffineCoordinateMap(torch.tensor([0., 1.], dtype=torch.float64), layout.grid_size, grid_offset="endpoints"))
    result = tk.decompositions.TRRSS(source).fit(
        physical.repeat_interleave(math.prod(out_shape), 0),
            labels=torch.arange(math.prod(out_shape)).repeat(physical.shape[0]),
        rank=1, generator=torch.Generator().manual_seed(0))
    assert result.out_shape == out_shape
    torch.testing.assert_close(result.evaluate_coordinates(physical), function(physical))


@pytest.mark.parametrize('singleton_axis', [False, True])
def test_quantized_rss_accepts_scalar_outputs(singleton_axis):
    layout = tk.formats.QuantizedLayout(1, base=2, level=3)
    physical = torch.arange(8, dtype=torch.float64).reshape(-1, 1) / 7

    def function(values):
        result = torch.ones_like(values[:, 0])
        return result.unsqueeze(-1) if singleton_axis else result

    source = tk.decompositions.CallableQuanticsSource(
        function,
        layout.n_coordinates,
        layout=layout,
        coordinate_map=tk.formats.AffineCoordinateMap(torch.tensor([0., 1.], dtype=torch.float64), layout.grid_size, grid_offset="endpoints"))
    result = tk.decompositions.TRRSS(source).fit(
        physical,
        rank=1,)
    assert result.n_sites == layout.n_sites
    assert torch.allclose(result.evaluate_coordinates(physical), torch.ones(8, dtype=torch.float64))


def test_matrix_source_returns_matrix_ring():
    source = tk.decompositions.DenseTensorSource.matrix(
        torch.ones(2, 2, 2, 2, dtype=torch.float64))
    samples = torch.cartesian_prod(torch.arange(2), torch.arange(2)).repeat_interleave(4, 0)
    result = tk.decompositions.TRRSS(source).fit(
        samples, labels=torch.arange(4).repeat(4), rank=1,
        generator=torch.Generator().manual_seed(0))
    assert isinstance(result, tk.decompositions.TRMDecomposition)
    torch.testing.assert_close(result.contract_dense(), source.contract_dense())


def test_matrix_rs_reuses_the_numerical_tensor_view():
    source = tk.decompositions.DenseTensorSource.matrix(torch.ones(2, 2, 2, 2, 2, 2))
    with pytest.warns(tk.decompositions.ExperimentalWarning):
        result = tk.decompositions.TRRS(
            source, sketch_operator=tk.decompositions.SampledSketch()).fit(rank=1)
    assert isinstance(result, tk.decompositions.TRMDecomposition)
    torch.testing.assert_close(result.contract_dense(), source.contract_dense())
