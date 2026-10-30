"""Tests for local loop-opening contracts, QR preparation and solvers."""

import pytest
import torch

import tensorkrowch as tk

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
            'gauge': 'qr',
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
            record = opening.diagnostics['qr_refinement']
            assert record['relative_error_after'] <= \
                record['relative_error_before']

    @pytest.mark.parametrize('side', ['left', 'right'])
    def test_fixed_axis_stays_unprojected_and_complex(self, side):
        left = torch.tensor([1., 2.], dtype=torch.float64)
        center = torch.tensor([1., -2., 3.], dtype=torch.float64)
        right = torch.tensor([2., 1., -1., 3.], dtype=torch.float64)
        target = torch.einsum('i,j,k->ijk', left, center, right)
        vector = left if side == 'left' else right
        gauge = 1j * vector[None, :, None]
        opening = tk.decompositions.QRLoopOpener().open(
            target, 1, **{f'fixed_{side}': gauge},
            context={'generator': torch.Generator().manual_seed(121)})
        assert not opening.diagnostics[f'qr_{side}']
        assert torch.equal(getattr(opening, f'{side}_gauge'), gauge)
        torch.testing.assert_close(
            opening.contract_dense(), target.to(torch.complex128))

    def test_minimal_rank_one_problem(self):
        target = torch.tensor([[[1.], [2.], [-3.]]], dtype=torch.float64)
        opening = tk.decompositions.QRLoopOpener().open(
            target, 1,
            context={'generator': torch.Generator().manual_seed(123)})
        torch.testing.assert_close(opening.contract_dense(), target)

    def test_worsening_refinement_is_rejected(self):
        target = torch.tensor([[[1.], [2.], [-3.]]], dtype=torch.float64)
        opener = tk.decompositions.QRLoopOpener()

        def worsen(**kwargs):
            cores = list(kwargs['context']['initial_cores'])
            cores[1] = 2 * cores[1]
            return tk.decompositions.LoopOpening(
                cores[0], cores[1:-1], cores[-1], (1, 1, 1),
                diagnostics={'metadata': {'n_sweeps': 1}})

        opener.refiner = tk.decompositions.CallableLoopOpener(worsen)
        result = opener.open(target, 1)
        assert not result.diagnostics['qr_refinement']['accepted']
        torch.testing.assert_close(result.contract_dense(), target)

    @pytest.mark.parametrize('invalid', [True, None, 1.5, '1', -1])
    def test_invalid_refinement_budget(self, invalid):
        error = ValueError if invalid == -1 else TypeError
        with pytest.raises(error, match='refinement_sweeps'):
            tk.decompositions.QRLoopOpener(refinement_sweeps=invalid)

    def test_two_fixed_axes_need_no_preparation_or_refinement(self):
        cores, target = _local_problem(torch.complex128)
        result = tk.decompositions.QRLoopOpener(
            _FixedGaugeCoreOpener()).open(
                target, (2, 3, 2),
                fixed_left=cores[0], fixed_right=cores[-1])
        assert not result.diagnostics['qr_left']
        assert not result.diagnostics['qr_right']
        assert 'qr_refinement' not in result.diagnostics
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

    def test_hybrid_joint_projection_and_nonuniform_ranks(self):
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
        result = opener.open(tensor, rank)
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
    """Keeps the private solver used by not-yet-reviewed sketching working."""

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
