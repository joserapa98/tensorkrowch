"""Tests for local loop-opening contracts, QR preparation and solvers."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.als import tr as tr_module
from tensorkrowch.decompositions.ring.opening import (
    _FixedGaugeCoreOpener,
    resolve_loop_opener,
)
from tests.decompositions.als._oracles import contract_tr_dense, make_tr_cores


def _local_problem(dtype=torch.float64, block=False):
    """Creates an exact local ring with nonuniform ranks."""
    dimensions = (3, 2, 3, 4) if block else (3, 2, 4)
    ranks = (2, 3, 2, 2) if block else (2, 3, 2)
    cores = make_tr_cores(
        in_dim=dimensions, rank=ranks, dtype=dtype,
        generator=torch.Generator().manual_seed(120))
    return cores, contract_tr_dense(cores)


class TestLoopOpeningContracts:

    def test_opening_preserves_order_and_actual_ranks(self):
        cores, tensor = _local_problem(block=True)
        opening = tk.decompositions.LoopOpening(
            left_gauge=cores[0], cores=cores[1:-1], right_gauge=cores[-1],
            rank=(2, 3, 2, 2))
        assert opening.all_cores == tuple(cores)
        torch.testing.assert_close(opening.contract_dense(), tensor)
        with pytest.raises(ValueError, match='actual right-link rank'):
            tk.decompositions.LoopOpening(
                left_gauge=cores[0], cores=cores[1:-1],
                right_gauge=cores[-1], rank=(2, 2, 2, 2))

    @pytest.mark.parametrize('orientation', ['invalid', None, 1])
    def test_invalid_orientation(self, orientation):
        cores, _ = _local_problem()
        with pytest.raises(ValueError, match='orientation'):
            tk.decompositions.LoopOpening(
                left_gauge=cores[0], cores=(cores[1],),
                right_gauge=cores[2], rank=(2, 3, 2),
                orientation=orientation)

    @pytest.mark.parametrize('name', [
        'supports_fixed_left', 'supports_fixed_right',
        'supports_two_fixed_gauges', 'supports_blocks',
    ])
    def test_capability_types(self, name):
        with pytest.raises(TypeError, match=name):
            tk.decompositions.LoopOpenerCapabilities(**{name: 1})

    def test_capabilities_reject_unsupported_constraints(self):
        capabilities = tk.decompositions.LoopOpenerCapabilities(
            supports_fixed_left=True, supports_fixed_right=True)
        with pytest.raises(ValueError, match='two fixed gauges'):
            capabilities.require(
                fixed_left=True, fixed_right=True, block_size=1)
        with pytest.raises(ValueError, match='input blocks'):
            capabilities.require(
                fixed_left=False, fixed_right=False, block_size=2)
        capabilities = tk.decompositions.LoopOpenerCapabilities()
        for left, right in ((True, False), (False, True)):
            with pytest.raises(ValueError, match='fixed'):
                capabilities.require(
                    fixed_left=left, fixed_right=right, block_size=1)


class TestALSLoopOpener:

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('orientation', ['right', 'left'])
    @pytest.mark.parametrize('block', [False, True])
    def test_exact_warm_start(self, dtype, orientation, block):
        cores, tensor = _local_problem(dtype, block)
        opener = tk.decompositions.ALSLoopOpener({
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=1),
        })
        rank = tuple(core.shape[-1] for core in cores)
        opening = opener.open(
            tensor, rank, orientation=orientation,
            context={'initial_cores': cores})
        assert opening.orientation == orientation
        assert opening.rank == rank
        torch.testing.assert_close(
            opening.contract_dense(), tensor, rtol=2e-10, atol=2e-10)
        assert opening.local_records
        assert opening.diagnostics['algorithm'] == 'tr_als'

    @pytest.mark.parametrize('side', ['left', 'right'])
    def test_fixed_gauge_is_preserved(self, side):
        cores, tensor = _local_problem()
        gauge = cores[0] if side == 'left' else cores[-1]
        opener = tk.decompositions.ALSLoopOpener({
            'orthogonalize': True,
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=2),
        })
        opening = opener.open(
            tensor, (2, 3, 2), **{f'fixed_{side}': gauge},
            context={'initial_cores': cores})
        assert torch.equal(getattr(opening, f'{side}_gauge'), gauge)
        torch.testing.assert_close(opening.contract_dense(), tensor)

    @pytest.mark.parametrize('source', ['dense', 'callable'])
    def test_source_and_complex_initializer(self, source):
        cores, tensor = _local_problem()
        initial = [core.to(torch.complex128) for core in cores]
        if source == 'dense':
            target = tk.decompositions.DenseTensorSource(tensor)
        else:
            def target(indices):
                return tensor[tuple(indices.T)]
        opener = tk.decompositions.ALSLoopOpener({
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=1),
        })
        opening = opener.open(
            target, (2, 3, 2), orientation='left', context={
                'in_dim': tensor.shape, 'dtype': tensor.dtype,
                'initial_cores': initial})
        assert all(c.dtype == torch.complex128 for c in opening.all_cores)
        torch.testing.assert_close(
            opening.contract_dense(), tensor.to(torch.complex128))

    def test_defaults_equal_string_preset(self):
        first = tk.decompositions.ALSLoopOpener()
        second = resolve_loop_opener('als')
        assert first.fit_options == second.fit_options
        assert first.fit_options['convergence'].max_sweeps == 100

    @pytest.mark.parametrize('name', [
        'rank', 'fixed_cores', 'initial_cores', 'collect_metrics',
    ])
    def test_reserved_options(self, name):
        with pytest.raises(ValueError, match=name):
            tk.decompositions.ALSLoopOpener({name: None})

    @pytest.mark.parametrize('invalid', [1, [], 'als'])
    def test_invalid_options(self, invalid):
        with pytest.raises(TypeError, match='fit_options'):
            tk.decompositions.ALSLoopOpener(invalid)

    def test_two_fixed_gauges_are_rejected(self):
        cores, tensor = _local_problem()
        with pytest.raises(ValueError, match='two fixed gauges'):
            tk.decompositions.ALSLoopOpener().open(
                tensor, (2, 3, 2),
                fixed_left=cores[0], fixed_right=cores[-1])


class TestQRLoopOpener:

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('orientation', ['left', 'right'])
    @pytest.mark.parametrize('side', [None, 'left', 'right'])
    @pytest.mark.parametrize('method', [
        'uniform', 'leverage', 'product_leverage'])
    @pytest.mark.parametrize('zipup', [False, True])
    def test_sampled_refinement_inherits_policy_and_validation(
            self, dtype, orientation, side, method, zipup, monkeypatch):
        cores, target = _local_problem(dtype)
        sampler = tk.decompositions.SamplingPolicy(32, method=method,
                                                   zipup=zipup)
        policy = tk.decompositions.ConvergencePolicy(
            max_sweeps=3, rtol=1e-12, error_samples=5)
        generator = torch.Generator().manual_seed(321)
        opener = tk.decompositions.QRLoopOpener(
            tk.decompositions.ALSLoopOpener({
                'sampler': sampler, 'convergence': policy,
                'generator': generator}), refinement_sweeps=2)
        assert opener.refiner.fit_options['sampler'] is sampler
        assert opener.refiner.fit_options['convergence'].error_samples == 5
        assert opener.refiner.fit_options['convergence'].rtol == policy.rtol
        assert opener.refiner.fit_options['convergence'].max_sweeps == 2
        initial = []
        refine = opener.refiner.open

        def record(*args, **kwargs):
            initial.append(contract_tr_dense(
                kwargs['context']['initial_cores']))
            return refine(*args, **kwargs)

        monkeypatch.setattr(opener.refiner, 'open', record)
        fixed = {} if side is None else {
            f'fixed_{side}': cores[0 if side == 'left' else -1]}
        result = opener.open(
            target, (2, 3, 2), orientation=orientation, **fixed,
            context={'initial_cores': cores})
        diagnostics = result.diagnostics['qr_refinement']
        info = diagnostics['error_evaluation']
        assert info['estimated']
        assert info['n_samples'] == 5
        assert info['seed'] == 322
        assert diagnostics['sampler']['method'] == method
        assert diagnostics['sampler']['zipup'] is zipup

        rng = torch.Generator().manual_seed(322)
        dimensions = target.shape if orientation == 'right' else \
            tuple(reversed(target.shape))
        indices = torch.stack([torch.randint(dim, (5,), generator=rng)
                               for dim in dimensions], dim=1)
        if orientation == 'left':
            indices = indices.flip(1)
        selection = tuple(indices.unbind(-1))
        denominator = target[selection].norm()
        before = (initial[0][selection] - target[selection]).norm()
        after = (result.contract_dense()[selection] - target[selection]).norm()
        final, = result.diagnostics['errors']
        assert final.kind == 'final'
        assert after <= before + 1e-12 * denominator
        torch.testing.assert_close(final.relative, after / denominator)

    @pytest.mark.parametrize('sampled', [False, True])
    @pytest.mark.parametrize('budget', [None, 100])
    def test_refinement_uses_full_error_without_sampled_validation(
            self, sampled, budget):
        cores, target = _local_problem()
        sampler = tk.decompositions.SamplingPolicy(32) if sampled else None
        opener = tk.decompositions.QRLoopOpener(
            tk.decompositions.ALSLoopOpener({
                'sampler': sampler,
                'convergence': tk.decompositions.ConvergencePolicy(
                    max_sweeps=1, error_samples=budget),
                'generator': torch.Generator().manual_seed(321)}))
        result = opener.open(target, (2, 3, 2),
                             context={'initial_cores': cores})
        record = result.diagnostics['qr_refinement']
        assert not record['error_evaluation']['estimated']
        assert record['error_evaluation']['n_samples'] == target.numel()
        error = (result.contract_dense() - target).norm() / target.norm()
        final, = result.diagnostics['errors']
        torch.testing.assert_close(final.relative, error)

    @pytest.mark.parametrize('orientation', ['left', 'right'])
    def test_sampled_acceptance_uses_validation_without_dense_contraction(
            self, orientation, monkeypatch):
        target = torch.ones(3, 2, 4, dtype=torch.float64)
        cores = [torch.ones(1, dim, 1, dtype=target.dtype)
                 for dim in target.shape]
        cores[1] = 2 * cores[1]
        generator = torch.Generator().manual_seed(123)
        opener = tk.decompositions.QRLoopOpener(
            tk.decompositions.ALSLoopOpener({
                'sampler': tk.decompositions.SamplingPolicy(8),
                'convergence': tk.decompositions.ConvergencePolicy(
                    max_sweeps=1, error_samples=1), 'generator': generator}))
        rng = torch.Generator().manual_seed(124)
        dimensions = target.shape if orientation == 'right' else \
            tuple(reversed(target.shape))
        indices = [int(torch.randint(dim, (1,), generator=rng))
                   for dim in dimensions]
        if orientation == 'left':
            indices.reverse()

        def initial(target, rank, **kwargs):
            initial_cores = kwargs['context']['initial_cores']
            return tk.decompositions.LoopOpening(
                initial_cores[0], initial_cores[1:-1], initial_cores[-1],
                (1, 1, 1),
                orientation=orientation)

        def update(backend, site, sweep, return_record):
            if site == 1:
                candidate = torch.full_like(backend.cores[site], 20)
                candidate[:, indices[1], :] = 1
                backend._update({site: candidate})

        plans = []
        prepare = tk.decompositions.ALSProblem.prepare_evaluation

        def prepare_once(problem, n_samples, generator):
            state = generator.get_state().clone()
            prepare(problem, n_samples, generator)
            assert torch.equal(generator.get_state(), state)
            plans.append(problem.evaluation_info)

        def unexpected_dense(*args):
            pytest.fail('Sampled refinement materialized the complete ring')

        monkeypatch.setattr(opener.opener, 'open', initial)
        monkeypatch.setattr(tr_module._TRSampledALSBackend, 'update_site',
                            update)
        monkeypatch.setattr(tk.decompositions.ALSProblem, 'prepare_evaluation',
                            prepare_once)
        monkeypatch.setattr(tk.decompositions.LoopOpening, 'contract_dense',
                            unexpected_dense)
        result = opener.open(target, 1, orientation=orientation,
                             fixed_left=cores[0],
                             context={'initial_cores': cores})
        assert len(plans) == 1
        record = result.diagnostics['qr_refinement']
        assert record['error_evaluation']['estimated']
        assert result.diagnostics['errors'][0].relative < 1e-12
        # The chosen validation entry improves although the full norm worsens.
        assert (contract_tr_dense(result.all_cores) - target).norm() > \
            target.norm()

    def test_fixed_balancing_is_disabled_by_default(self):
        assert tk.decompositions.QRLoopOpener().balance_fixed is False

    @pytest.mark.parametrize('invalid', [None, 0, 1, 'true'])
    def test_invalid_fixed_balancing(self, invalid):
        with pytest.raises(TypeError, match='balance_fixed'):
            tk.decompositions.QRLoopOpener(balance_fixed=invalid)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('orientation', ['right', 'left'])
    @pytest.mark.parametrize('block', [False, True])
    @pytest.mark.parametrize('refinement_sweeps', [0, 1])
    def test_full_coordinate_round_trip(
            self, dtype, orientation, block, refinement_sweeps):
        cores, tensor = _local_problem(dtype, block)
        opener = tk.decompositions.QRLoopOpener(
            tk.decompositions.ALSLoopOpener({
                'convergence': tk.decompositions.ConvergencePolicy(
                    max_sweeps=1),
            }), refinement_sweeps=refinement_sweeps)
        rank = tuple(c.shape[-1] for c in cores)
        opening = opener.open(
            tensor, rank, orientation=orientation,
            context={'initial_cores': cores})
        assert opening.rank == rank
        assert opening.left_gauge.shape[1] == tensor.shape[0]
        assert opening.right_gauge.shape[1] == tensor.shape[-1]
        torch.testing.assert_close(
            opening.contract_dense(), tensor, rtol=2e-10, atol=2e-10)
        assert opening.diagnostics['qr_left']
        assert opening.diagnostics['qr_right']
        if refinement_sweeps:
            final, = opening.diagnostics['errors']
            assert final.kind == 'final'
            assert final.relative < 1e-10

    @pytest.mark.parametrize('side', ['left', 'right'])
    @pytest.mark.parametrize('balance_fixed', [False, True])
    def test_fixed_axis_is_preserved_with_promotion(self, side, balance_fixed):
        left = torch.tensor([1., 2.], dtype=torch.float64)
        center = torch.tensor([1., -2., 3.], dtype=torch.float64)
        right = torch.tensor([2., 1., -1., 3.], dtype=torch.float64)
        target = torch.einsum('i,j,k->ijk', left, center, right)
        vector = left if side == 'left' else right
        gauge = 1j * vector[None, :, None]
        opening = tk.decompositions.QRLoopOpener(
            balance_fixed=balance_fixed).open(
            target, 1, **{f'fixed_{side}': gauge},
            context={'generator': torch.Generator().manual_seed(121)})
        assert opening.diagnostics[f'qr_{side}'] == balance_fixed
        free_side = 'right' if side == 'left' else 'left'
        assert opening.diagnostics[f'qr_{free_side}']
        assert torch.equal(getattr(opening, f'{side}_gauge'), gauge)
        torch.testing.assert_close(
            opening.contract_dense(), target.to(torch.complex128))

    def test_minimal_rank_one_problem(self):
        target = torch.tensor([[[1.], [2.], [-3.]]], dtype=torch.float64)
        opening = tk.decompositions.QRLoopOpener().open(
            target, 1,
            context={'generator': torch.Generator().manual_seed(123)})
        torch.testing.assert_close(opening.contract_dense(), target)

    @pytest.mark.parametrize('side', [None, 'left', 'right'])
    def test_worsening_refinement_is_rejected(self, side, monkeypatch):
        target = torch.tensor([[[1.], [2.], [-3.]]], dtype=torch.float64)
        cores = [torch.ones(1, 1, 1, dtype=target.dtype),
                 target.clone(), torch.ones(1, 1, 1, dtype=target.dtype)]
        fixed = {} if side is None else {
            f'fixed_{side}': cores[0 if side == 'left' else -1]}

        def initial(**kwargs):
            initial_cores = kwargs['context']['initial_cores']
            return tk.decompositions.LoopOpening(
                initial_cores[0], initial_cores[1:-1], initial_cores[-1],
                (1, 1, 1))

        def worsen(backend, site, sweep, return_record):
            backend.cache.design(site, compact=True)
            replacements = {site: 2 * backend.cores[site]} if site == 1 else {}
            backend._update(replacements)

        capabilities = tk.decompositions.LoopOpenerCapabilities(
            supports_fixed_left=True, supports_fixed_right=True)
        opener = tk.decompositions.QRLoopOpener(
            tk.decompositions.CallableLoopOpener(initial, capabilities))
        monkeypatch.setattr(tr_module._TRALSBackend, 'update_site', worsen)
        result = opener.open(target, 1, **fixed,
                             context={'initial_cores': cores})
        assert result.diagnostics['errors'][0].relative < 1e-12
        torch.testing.assert_close(result.contract_dense(), target)
        for name, gauge in fixed.items():
            assert torch.equal(getattr(result, name.replace('fixed_', '') +
                                       '_gauge'), gauge)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('side', ['left', 'right'])
    def test_fixed_gauge_refinement_improves_original_residual(
            self, dtype, side):
        generator = torch.Generator().manual_seed(324)
        cores = make_tr_cores((9, 3, 7), (1, 1, 1), dtype=dtype,
                              generator=generator)
        target = contract_tr_dense(cores)
        gauge = cores[0 if side == 'left' else -1]
        cores[1] = 2 * cores[1]

        def keep_initial(**kwargs):
            initial = kwargs['context']['initial_cores']
            return tk.decompositions.LoopOpening(
                initial[0], initial[1:-1], initial[-1], (1, 1, 1))

        capabilities = tk.decompositions.LoopOpenerCapabilities(
            supports_fixed_left=True, supports_fixed_right=True)
        opener = tk.decompositions.QRLoopOpener(
            tk.decompositions.CallableLoopOpener(keep_initial, capabilities),
            balance_fixed=True)
        result = opener.open(target, 1, **{f'fixed_{side}': gauge},
                             context={'initial_cores': cores})

        assert result.diagnostics['errors'][0].relative < 1e-12
        assert torch.equal(getattr(result, f'{side}_gauge'), gauge)
        torch.testing.assert_close(result.contract_dense(), target)

    @pytest.mark.parametrize('invalid', [True, None, 1.5, '1', -1])
    def test_invalid_refinement_budget(self, invalid):
        error = ValueError if invalid == -1 else TypeError
        with pytest.raises(error, match='refinement_sweeps'):
            tk.decompositions.QRLoopOpener(refinement_sweeps=invalid)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('orientation', ['left', 'right'])
    @pytest.mark.parametrize('noise', [0., 0.1])
    def test_two_fixed_axes_keep_the_original_least_squares_metric(
            self, dtype, orientation, noise):
        cores, target = _local_problem(dtype)
        generator = torch.Generator().manual_seed(325)
        target = target + noise * torch.randn(
            target.shape, dtype=dtype, generator=generator)
        result = tk.decompositions.QRLoopOpener(
            _FixedGaugeCoreOpener()).open(
                target, (2, 3, 2),
                fixed_left=cores[0], fixed_right=cores[-1],
                orientation=orientation)
        direct = _FixedGaugeCoreOpener().open(
            target, (2, 3, 2), fixed_left=cores[0], fixed_right=cores[-1],
            orientation=orientation)
        assert not result.diagnostics['qr_left']
        assert not result.diagnostics['qr_right']
        assert torch.equal(result.left_gauge, cores[0])
        assert torch.equal(result.right_gauge, cores[-1])
        assert 'qr_refinement' not in result.diagnostics
        torch.testing.assert_close(result.contract_dense(),
                                   direct.contract_dense())
        if not noise:
            torch.testing.assert_close(result.contract_dense(), target)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('external', [1, 3, 9])
    @pytest.mark.parametrize('scale', [1e-150, 1., 1e150])
    def test_fixed_qr_algebra(self, dtype, external, scale):
        generator = torch.Generator().manual_seed(320)
        size = min(external, 6)
        u = torch.linalg.qr(torch.randn(
            external, size, dtype=dtype, generator=generator)).Q
        v = torch.linalg.qr(torch.randn(
            6, size, dtype=dtype, generator=generator)).Q
        singular = torch.logspace(0, -4, size, dtype=torch.float64)
        matrix = ((u * singular) @ v.mH) * scale
        gauge = matrix.reshape(external, 2, 3).permute(1, 0, 2)
        orthogonal, lift, inverse = \
            tk.decompositions.QRLoopOpener._fixed_qr(gauge)
        rows = orthogonal.permute(1, 0, 2).reshape(size, 6)

        torch.testing.assert_close(lift @ rows / scale, matrix / scale)
        torch.testing.assert_close(
            rows @ rows.mH, torch.eye(size, dtype=dtype))
        torch.testing.assert_close(
            inverse @ lift, torch.eye(size, dtype=dtype),
            rtol=1e-9, atol=1e-9)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('orientation', ['left', 'right'])
    @pytest.mark.parametrize('side', ['left', 'right'])
    @pytest.mark.parametrize('block', [False, True])
    @pytest.mark.parametrize('balance_fixed', [False, True])
    def test_fixed_coordinate_round_trip(
            self, dtype, orientation, side, block, balance_fixed):
        generator = torch.Generator().manual_seed(321)
        physical = (3, 2) if block else (5,)
        dimensions = (9, *physical, 4) if side == 'left' else \
            (4, *physical, 9)
        cores = make_tr_cores(
            dimensions, (2,) * len(dimensions), dtype=dtype,
            generator=generator)
        index = 0 if side == 'left' else -1
        external = dimensions[index]
        u = torch.linalg.qr(torch.randn(
            external, 4, dtype=dtype, generator=generator)).Q
        v = torch.linalg.qr(torch.randn(
            4, 4, dtype=dtype, generator=generator)).Q
        matrix = (u * torch.logspace(0, -4, 4)) @ v.mH
        cores[index] = matrix.reshape(external, 2, 2).permute(1, 0, 2)
        gauge = cores[index]
        target = contract_tr_dense(cores)
        originals = [core.clone() for core in cores]
        seen = []

        def inspect(**kwargs):
            initial = kwargs['context']['initial_cores']
            fixed = kwargs[f'fixed_{side}']
            if balance_fixed:
                rows = fixed.permute(1, 0, 2).reshape(4, 4)
                torch.testing.assert_close(
                    rows @ rows.mH, torch.eye(4, dtype=dtype))
            else:
                assert fixed is gauge
            assert initial[index] is fixed
            assert kwargs['target'].shape[index] == (
                4 if balance_fixed else external)
            torch.testing.assert_close(
                contract_tr_dense(initial), kwargs['target'],
                rtol=1e-9, atol=1e-9)
            seen.append(kwargs['target'])
            return tk.decompositions.LoopOpening(
                initial[0], initial[1:-1], initial[-1], kwargs['rank'],
                orientation=kwargs['orientation'])

        opener = tk.decompositions.QRLoopOpener(
            tk.decompositions.CallableLoopOpener(
                inspect, tk.decompositions.LoopOpenerCapabilities(
                    supports_fixed_left=True, supports_fixed_right=True,
                    supports_blocks=True)),
            refinement_sweeps=0, balance_fixed=balance_fixed)
        result = opener.open(
            target, (2,) * len(cores), orientation=orientation,
            **{f'fixed_{side}': gauge},
            context={'initial_cores': cores})

        assert len(seen) == 1
        assert getattr(result, f'{side}_gauge') is gauge
        torch.testing.assert_close(result.contract_dense(), target)
        for core, original in zip(cores, originals):
            assert torch.equal(core, original)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('orientation', ['left', 'right'])
    @pytest.mark.parametrize('side', ['left', 'right'])
    @pytest.mark.parametrize('refinement_sweeps', [0, 1])
    @pytest.mark.parametrize('balance_fixed', [False, True])
    def test_fixed_gauge_als_with_supplied_initialization(
            self, dtype, orientation, side, refinement_sweeps, balance_fixed):
        cores, target = _local_problem(dtype, block=True)
        index = 0 if side == 'left' else -1
        gauge = cores[index]
        opener = tk.decompositions.QRLoopOpener(
            tk.decompositions.ALSLoopOpener({
                'orthogonalize': True,
                'convergence': tk.decompositions.ConvergencePolicy(
                    max_sweeps=2),
            }), refinement_sweeps=refinement_sweeps,
            balance_fixed=balance_fixed)
        result = opener.open(
            target, tuple(core.shape[-1] for core in cores),
            **{f'fixed_{side}': gauge}, orientation=orientation,
            context={'initial_cores': cores})

        assert torch.equal(getattr(result, f'{side}_gauge'), gauge)
        torch.testing.assert_close(result.contract_dense(), target)
        if refinement_sweeps:
            final, = result.diagnostics['errors']
            assert final.kind == 'final'
            assert final.relative < 1e-10

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('side', ['left', 'right'])
    def test_unsupported_target_component_is_measured_after_restoring(
            self, dtype, side):
        generator = torch.Generator().manual_seed(322)
        target = torch.randn(5, 3, 6, dtype=dtype, generator=generator)
        external = target.shape[0 if side == 'left' else -1]
        gauge = torch.randn(1, external, 1, dtype=dtype, generator=generator)
        opener = tk.decompositions.QRLoopOpener(
            tk.decompositions.ALSLoopOpener({
                'convergence': tk.decompositions.ConvergencePolicy(
                    max_sweeps=10),
            }), balance_fixed=True)
        initial = make_tr_cores(
            target.shape, (1, 1, 1), dtype=dtype, generator=generator)
        result = opener.open(target, 1, **{f'fixed_{side}': gauge},
                             context={'initial_cores': initial})
        relative = (result.contract_dense() - target).norm() / target.norm()
        final, = result.diagnostics['errors']

        assert torch.equal(getattr(result, f'{side}_gauge'), gauge)
        assert relative > 0.1
        torch.testing.assert_close(relative, final.relative)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('side', ['left', 'right'])
    def test_zero_fixed_gauge_is_preserved(self, dtype, side):
        target = torch.zeros(5, 3, 6, dtype=dtype)
        external = target.shape[0 if side == 'left' else -1]
        gauge = torch.zeros(2, external, 2, dtype=dtype)
        result = tk.decompositions.QRLoopOpener(balance_fixed=True).open(
            target, 2, **{f'fixed_{side}': gauge},
            context={'generator': torch.Generator().manual_seed(323)})

        assert torch.equal(getattr(result, f'{side}_gauge'), gauge)
        assert all(torch.isfinite(core).all() for core in result.all_cores)
        torch.testing.assert_close(result.contract_dense(), target)

    def test_invalid_target(self):
        opener = tk.decompositions.QRLoopOpener()
        with pytest.raises(TypeError, match='target'):
            opener.open([1, 2, 3], rank=1)
        with pytest.raises(ValueError, match='target'):
            opener.open(torch.ones(2, 3), rank=1)

    def test_invalid_fixed_gauge(self):
        opener = tk.decompositions.QRLoopOpener()
        target = torch.ones(2, 3, 4)
        with pytest.raises(TypeError, match='fixed_left'):
            opener.open(target, 1, fixed_left='invalid')
        with pytest.raises(ValueError, match='fixed_right'):
            opener.open(target, 1, fixed_right=torch.ones(1, 5, 1))


class TestBLOSTROpeners:

    @pytest.mark.parametrize('name', ['blostr', 'blostr_svd'])
    @pytest.mark.parametrize('orientation', ['left', 'right'])
    def test_exact_rank_one_opening(self, name, orientation):
        generator = torch.Generator().manual_seed(88)
        cores = make_tr_cores(
            (3, 3, 3, 3), (1, 1, 1, 1),
            dtype=torch.complex128, generator=generator)
        tensor = contract_tr_dense(cores)
        opener = resolve_loop_opener(name)
        result = opener.open(
            tk.decompositions.DenseTensorSource(tensor),
            rank=1, orientation=orientation,
            context={'generator': torch.Generator().manual_seed(89)})
        assert result.rank == (1, 1, 1, 1)
        torch.testing.assert_close(result.contract_dense(), tensor)
        assert result.diagnostics['algorithm'] == name

    @pytest.mark.parametrize('name', ['blostr', 'blostr_svd'])
    def test_complete_target_decomposition_and_discovery(self, name):
        generator = torch.Generator().manual_seed(303)
        cores = make_tr_cores(
            (4, 4, 4), (2, 2, 2),
            dtype=torch.complex128, generator=generator)
        tensor = contract_tr_dense(cores)
        opener = resolve_loop_opener(name)
        result = opener.decompose(tensor, context={
            'generator': torch.Generator().manual_seed(81)})
        assert isinstance(result, tk.decompositions.TRDecomposition)
        assert result.rank == [2, 2, 2]
        torch.testing.assert_close(
            result.contract_dense(), tensor, rtol=1e-8, atol=1e-8)

    @pytest.mark.parametrize('name', ['blostr', 'blostr_svd'])
    def test_fixed_gauges_are_rejected(self, name):
        with pytest.raises(ValueError, match='fixed left'):
            resolve_loop_opener(name).open(
                torch.ones(3, 3, 3), 1,
                fixed_left=torch.ones(1, 3, 1))

    @pytest.mark.parametrize('name', ['blostr', 'blostr_svd'])
    @pytest.mark.parametrize('invalid', [1, [], 'invalid'])
    def test_invalid_options(self, name, invalid):
        cls = tk.decompositions.BLOSTRLoopOpener if name == 'blostr' \
            else tk.decompositions.BLOSTRSVDLoopOpener
        with pytest.raises(TypeError, match='fit_options'):
            cls(invalid)

    @pytest.mark.parametrize('name', ['blostr', 'blostr_svd'])
    @pytest.mark.parametrize('option', ['source', 'rank', 'out_device',
                                       'collect_metrics', 'verbose'])
    def test_reserved_options(self, name, option):
        cls = tk.decompositions.BLOSTRLoopOpener if name == 'blostr' \
            else tk.decompositions.BLOSTRSVDLoopOpener
        with pytest.raises(ValueError, match='should not override'):
            cls({option: None})

    @pytest.mark.parametrize('orientation', ['right', 'left'])
    @pytest.mark.parametrize('kind', ['tensor', 'dense', 'callable'])
    def test_hybrid_joint_projection_and_nonuniform_ranks(
            self, orientation, kind):
        generator = torch.Generator().manual_seed(515)
        rank = (3, 1, 3, 2)
        cores = make_tr_cores(
            (6, 4, 4, 6), rank, dtype=torch.complex128,
            generator=generator)
        tensor = contract_tr_dense(cores)
        opener = tk.decompositions.BLOSTRSVDLoopOpener({
            'projection': 'gaussian',
            'generator': torch.Generator().manual_seed(516),
        })
        source = tensor
        if kind == 'dense':
            source = tk.decompositions.DenseTensorSource(tensor)
        elif kind == 'callable':
            source = tk.decompositions.CallableTensorSource(
                lambda indices: tensor[tuple(indices.T)],
                in_dim=tensor.shape, dtype=tensor.dtype, out_shape=())
        result = opener.open(source, rank, orientation=orientation)
        torch.testing.assert_close(
            result.contract_dense(), tensor, rtol=1e-8, atol=1e-8)


class TestOpeningAdapters:

    @pytest.mark.parametrize('name, cls', [
        ('als', tk.decompositions.ALSLoopOpener),
        ('blostr', tk.decompositions.BLOSTRLoopOpener),
        ('blostr_svd', tk.decompositions.BLOSTRSVDLoopOpener),
    ])
    def test_resolver(self, name, cls):
        opener = resolve_loop_opener(name)
        assert isinstance(opener, cls)
        assert resolve_loop_opener(opener) is opener

    @pytest.mark.parametrize('invalid', [None, 1, []])
    def test_invalid_resolver_type(self, invalid):
        with pytest.raises(TypeError, match='loop_opener'):
            resolve_loop_opener(invalid)

    def test_invalid_resolver_name(self):
        with pytest.raises(ValueError, match='loop_opener'):
            resolve_loop_opener('blostr+als')

    def test_callable_arguments_and_result(self):
        cores, tensor = _local_problem()
        calls = []

        def function(**kwargs):
            calls.append(kwargs)
            return tk.decompositions.LoopOpening(
                cores[0], cores[1:-1], cores[-1], (2, 3, 2),
                orientation=kwargs['orientation'])

        opener = tk.decompositions.CallableLoopOpener(
            function, tk.decompositions.LoopOpenerCapabilities(
                supports_blocks=True))
        result = opener.open(tensor, (2, 3, 2), orientation='left')
        assert calls[0]['orientation'] == 'left'
        torch.testing.assert_close(result.contract_dense(), tensor)

    def test_callable_accepts_decomposition_result(self):
        result = tk.decompositions.QTRDecomposition(
            [torch.ones(1, 2, 1) for _ in range(3)],
            n_coordinates=1, base=2, level=3, domain=(0., 1.),
            metadata={'fit': 'quantics'})
        opener = tk.decompositions.CallableLoopOpener(
            lambda **kwargs: result,
            tk.decompositions.LoopOpenerCapabilities(supports_blocks=True))
        opening = opener.open(result.contract_dense(), (1, 1, 1))
        assert opening.diagnostics['metadata'] == result.metadata
        torch.testing.assert_close(
            opening.contract_dense(), result.contract_dense())

    def test_callable_checks_actual_block_size_without_context(self):
        def should_not_run(**kwargs):
            raise AssertionError('Unsupported block should be rejected first')

        opener = tk.decompositions.CallableLoopOpener(should_not_run)
        with pytest.raises(ValueError, match='input blocks'):
            opener.open(torch.ones(1, 2, 2, 1), 1)

    def test_invalid_callable_return(self):
        opener = tk.decompositions.CallableLoopOpener(lambda **kwargs: None)
        with pytest.raises(TypeError, match='LoopOpening'):
            opener.open(torch.ones(1, 2, 1), 1)


class TestLegacyFixedGaugeClosure:
    """Checks direct two-fixed-gauge closure for cyclic providers."""

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('orientation', ['right', 'left'])
    def test_direct_core_solve(self, dtype, orientation):
        generator = torch.Generator().manual_seed(121)
        left = torch.randn(1, 4, 2, dtype=dtype, generator=generator)
        core = torch.randn(2, 3, 2, dtype=dtype, generator=generator)
        right = torch.randn(2, 5, 1, dtype=dtype, generator=generator)
        target = contract_tr_dense((left, core, right))
        result = _FixedGaugeCoreOpener().open(
            target, (2, 2, 1), fixed_left=left, fixed_right=right,
            orientation=orientation)
        assert torch.equal(result.left_gauge, left)
        assert torch.equal(result.right_gauge, right)
        torch.testing.assert_close(result.contract_dense(), target)


class TestGaugeCoordinateMaps:

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_rectangular_maps_transform_external_axes_and_preserve_opening(
            self, dtype):
        generator = torch.Generator().manual_seed(707)
        cores = [torch.randn(*shape, dtype=dtype, generator=generator)
                 for shape in ((2, 3, 2), (2, 4, 2), (2, 5, 2))]
        opening = tk.decompositions.LoopOpening(
            cores[0], [cores[1]], cores[2], (2, 2, 2))
        original = opening.contract_dense().clone()
        left = torch.randn(7, 3, dtype=dtype, generator=generator,
                           requires_grad=True)
        right = torch.randn(6, 5, dtype=dtype, generator=generator,
                            requires_grad=True)
        a, b = opening.transform_gauges(left, right)
        mapped = tk.decompositions.LoopOpening(
            a, opening.cores, b, opening.rank).contract_dense()
        expected = torch.einsum('ma,aib,nb->min', left, original, right)
        torch.testing.assert_close(mapped, expected)
        torch.testing.assert_close(opening.contract_dense(), original)
        assert opening.left_gauge is cores[0]
        assert opening.right_gauge is cores[2]
        opening.left_map, opening.right_map = left, right
        stored = opening.transform_gauges()
        torch.testing.assert_close(stored[0], a)
        torch.testing.assert_close(stored[1], b)
        mapped.abs().square().sum().backward()
        assert left.grad is not None and torch.isfinite(left.grad).all()
        assert right.grad is not None and torch.isfinite(right.grad).all()

    def test_complex_maps_promote_real_gauges(self):
        opening = tk.decompositions.LoopOpening(
            torch.ones(1, 2, 1), [torch.ones(1, 3, 1)],
            torch.ones(1, 2, 1), (1, 1, 1))
        left, right = opening.transform_gauges(
            1j * torch.eye(2), torch.eye(2))
        assert left.is_complex()
        torch.testing.assert_close(left, 1j * opening.left_gauge)
        torch.testing.assert_close(right, opening.right_gauge)

    @pytest.mark.parametrize('matrix, error', [
        (torch.ones(2), TypeError), (torch.ones(2, 3), ValueError),
        ('map', TypeError),
    ])
    def test_invalid_maps_leave_original_gauges_unchanged(self, matrix, error):
        opening = tk.decompositions.LoopOpening(
            torch.ones(1, 2, 1), [torch.ones(1, 3, 1)],
            torch.ones(1, 2, 1), (1, 1, 1))
        before = opening.contract_dense().clone()
        with pytest.raises(error):
            opening.transform_gauges(left_map=matrix)
        torch.testing.assert_close(opening.contract_dense(), before)


@pytest.mark.parametrize('orientation', ['left', 'right'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('mode', ['absolute', 'relative'])
def test_fixed_gauge_shared_solve_matches_regularized_expansion(
        orientation, dtype, mode, monkeypatch):
    generator = torch.Generator().manual_seed(325)
    left = torch.randn(2, 4, 2, dtype=dtype, generator=generator)
    right = torch.randn(3, 5, 2, dtype=dtype, generator=generator)
    target = torch.randn(4, 3, 5, dtype=dtype, generator=generator)
    solver = tk.decompositions.LeastSquaresSolver(
        l2_reg=.02, l2_reg_mode=mode, driver='gelsd')
    core = torch.zeros(2, 3, 3, dtype=dtype)
    from tests.decompositions.als._oracles import dense_local_design
    full = dense_local_design([left, core, right], 1, 'tr')
    expected, _ = solver.solve(full, target.flatten())
    calls = []
    lstsq = torch.linalg.lstsq

    def capture(matrix, rhs, **kwargs):
        calls.append((matrix.shape, rhs.shape))
        return lstsq(matrix, rhs, **kwargs)

    monkeypatch.setattr(torch.linalg, 'lstsq', capture)
    result = _FixedGaugeCoreOpener(solver).open(
        target, (2, 3, 2), fixed_left=left, fixed_right=right,
        orientation=orientation)
    torch.testing.assert_close(result.cores[0], expected.reshape(core.shape),
                               atol=1e-10, rtol=1e-10)
    assert len(calls) == 1
    # Twenty data rows plus six Tikhonov rows share three target columns.
    assert calls[0] == ((26, 6), (26, 3))
    assert result.local_records[0].environment_shape == (20, 6)


def test_als_loop_opener_uses_shared_physical_systems():
    opener = tk.decompositions.ALSLoopOpener({
        'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=1)})
    target = torch.randn(4, 3, 5, dtype=torch.complex128,
                         generator=torch.Generator().manual_seed(326))
    result = opener.open(target, (2, 2, 2))
    assert len(result.local_records) == 3
    for site, record in enumerate(result.local_records):
        assert record.environment_shape == (
            target.numel() // target.shape[site], 4)
        assert record.target_shape == (record.environment_shape[0],
                                        target.shape[site])
