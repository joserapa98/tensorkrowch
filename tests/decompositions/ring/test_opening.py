"""Tests for local loop-opening contracts and strategies."""

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.ring.opening import resolve_loop_opener

from tests.decompositions.als._oracles import contract_tr_dense, make_tr_cores


def _local_problem(dtype=torch.float64):
    """Creates one exact three-site local TR target."""
    cores = make_tr_cores(
        in_dim=(3, 2, 4),
        rank=(2, 3, 2),
        dtype=dtype,
        generator=torch.Generator().manual_seed(120))
    return cores, contract_tr_dense(cores)


class TestLoopOpeningContracts:  # MARK: TestLoopOpeningContracts

    def test_opening_validates_actual_right_link_ranks(self):
        cores, tensor = _local_problem()
        opening = tk.decompositions.LoopOpening(
            left_gauge=cores[0],
            cores=(cores[1],),
            right_gauge=cores[2],
            rank=(2, 3, 2))

        assert opening.all_cores == tuple(cores)
        assert torch.allclose(opening.contract_dense(), tensor)
        with pytest.raises(ValueError, match='actual right-link rank'):
            tk.decompositions.LoopOpening(
                left_gauge=cores[0],
                cores=(cores[1],),
                right_gauge=cores[2],
                rank=(2, 2, 2))

    def test_capabilities_reject_constraints_before_execution(self):
        capabilities = tk.decompositions.LoopOpenerCapabilities(
            supports_fixed_left=True)

        with pytest.raises(ValueError, match='fixed right'):
            capabilities.require(
                fixed_left=False, fixed_right=True, block_size=1)
        with pytest.raises(ValueError, match='input blocks'):
            capabilities.require(
                fixed_left=False, fixed_right=False, block_size=2)


class TestSVDLoopOpener:  # MARK: TestSVDLoopOpener

    @pytest.mark.parametrize('absorb_singular_values', [False, True])
    @pytest.mark.parametrize('refinement_sweeps', [0, 1])
    def test_exact_rank_one_opening(
            self, absorb_singular_values, refinement_sweeps):
        factors = (
            torch.tensor([1., 2.]),
            torch.tensor([1., -2., 3.]),
            torch.tensor([2., 1., -1., 3.]),
        )
        target = torch.einsum('i,j,k->ijk', *factors).to(torch.complex128)
        result = tk.decompositions.SVDLoopOpener(
            absorb_singular_values=absorb_singular_values,
            refinement_sweeps=refinement_sweeps).open(
                target, rank=1,
                context={'generator': torch.Generator().manual_seed(121)})

        assert result.rank == (1, 1, 1)
        torch.testing.assert_close(result.contract_dense(), target)

    @pytest.mark.parametrize('fixed_side', ['left', 'right'])
    @pytest.mark.parametrize('absorb_singular_values', [False, True])
    def test_fixed_gauge_projection_promotes_complex_dtype(
            self, fixed_side, absorb_singular_values):
        left = torch.tensor([1., 2.], dtype=torch.float64)
        center = torch.tensor([1., -2., 3.], dtype=torch.float64)
        right = torch.tensor([2., 1., -1., 3.], dtype=torch.float64)
        target = torch.einsum('i,j,k->ijk', left, center, right)
        gauge = 1j * (left if fixed_side == 'left' else right)[None, :, None]
        fixed = {f'fixed_{fixed_side}': gauge}
        result = tk.decompositions.SVDLoopOpener(
            absorb_singular_values=absorb_singular_values,
            project_fixed_gauges=True).open(
                target, rank=1, **fixed,
                context={'generator': torch.Generator().manual_seed(121)})

        assert all(core.dtype == torch.complex128 for core in result.all_cores)
        torch.testing.assert_close(
            result.contract_dense(), target.to(gauge.dtype))
        assert torch.equal(getattr(result, f'{fixed_side}_gauge'), gauge)

    def test_invalid_target_is_rejected_before_projection(self):
        opener = tk.decompositions.SVDLoopOpener()
        with pytest.raises(TypeError, match='target'):
            opener.open([1, 2, 3], rank=1)
        with pytest.raises(ValueError, match='target'):
            opener.open(torch.ones(2, 3), rank=1)

    @pytest.mark.parametrize('invalid', [True, None, 1.5, '1', -1])
    def test_invalid_refinement_budget(self, invalid):
        error = ValueError if invalid == -1 else TypeError
        with pytest.raises(error, match='refinement_sweeps'):
            tk.decompositions.SVDLoopOpener(refinement_sweeps=invalid)

    def test_invalid_spectrum_policy(self):
        with pytest.raises(TypeError, match='absorb_singular_values'):
            tk.decompositions.SVDLoopOpener(absorb_singular_values='outside')

    def test_two_fixed_gauges_use_the_direct_solver(self):
        cores, tensor = _local_problem(torch.complex128)
        opener = tk.decompositions.SVDLoopOpener(
            tk.decompositions.FixedGaugeCoreOpener(), refinement_sweeps=1)
        result = opener.open(
            tensor, (2, 3, 2), fixed_left=cores[0], fixed_right=cores[2])
        assert torch.equal(result.left_gauge, cores[0])
        assert torch.equal(result.right_gauge, cores[2])
        assert not result.diagnostics['svd_trimmed_left']
        assert not result.diagnostics['svd_trimmed_right']
        assert 'svd_refinement' not in result.diagnostics
        torch.testing.assert_close(result.contract_dense(), tensor)


class TestALSLoopOpener:  # MARK: TestALSLoopOpener

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('orientation', ['right', 'left'])
    def test_initial_opening_matches_target_in_both_orientations(
            self, dtype, orientation):
        cores, tensor = _local_problem(dtype)
        opener = tk.decompositions.ALSLoopOpener({
            'gauge': 'none',
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=1),
        })
        opening = opener.open(
            tensor,
            rank=(2, 3, 2),
            orientation=orientation,
            context={'initial_cores': cores})

        assert opening.orientation == orientation
        assert opening.rank == (2, 3, 2)
        assert torch.allclose(
            opening.contract_dense(), tensor, rtol=2e-10, atol=2e-10)
        assert opening.local_records
        assert opening.diagnostics['algorithm'] == 'als'

    def test_one_fixed_gauge_remains_bitwise_equal(self):
        cores, tensor = _local_problem()
        opener = tk.decompositions.ALSLoopOpener({
            'gauge': 'qr',
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=2),
        })
        opening = opener.open(
            tensor,
            rank=(2, 3, 2),
            fixed_left=cores[0],
            context={'initial_cores': cores})

        assert torch.equal(opening.left_gauge, cores[0])
        assert torch.allclose(
            opening.contract_dense(), tensor, rtol=2e-10, atol=2e-10)

    def test_left_oriented_callable_uses_shared_source_contract(self):
        cores, tensor = _local_problem()

        def function(indices):
            return tensor[indices[:, 0], indices[:, 1], indices[:, 2]]

        opener = tk.decompositions.ALSLoopOpener({
            'gauge': 'none',
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=1),
        })
        opening = opener.open(
            function,
            rank=(2, 3, 2),
            orientation='left',
            context={
                'in_dim': tensor.shape,
                'dtype': tensor.dtype,
                'initial_cores': cores,
            })

        assert torch.allclose(
            opening.contract_dense(), tensor, rtol=2e-10, atol=2e-10)

    def test_two_fixed_gauges_are_delegated_to_direct_opener(self):
        cores, tensor = _local_problem()
        opener = tk.decompositions.ALSLoopOpener()

        with pytest.raises(ValueError, match='two fixed gauges'):
            opener.open(
                tensor,
                rank=(2, 3, 2),
                fixed_left=cores[0],
                fixed_right=cores[2])


class TestFixedGaugeCoreOpener:  # MARK: TestFixedGaugeCoreOpener

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('orientation', ['right', 'left'])
    def test_direct_core_solve_reconstructs_target(self, dtype, orientation):
        generator = torch.Generator().manual_seed(121)
        left = torch.randn(1, 4, 2, dtype=dtype, generator=generator)
        core = torch.randn(2, 3, 2, dtype=dtype, generator=generator)
        right = torch.randn(2, 5, 1, dtype=dtype, generator=generator)
        target = contract_tr_dense((left, core, right))
        opener = tk.decompositions.FixedGaugeCoreOpener()
        opening = opener.open(
            target,
            rank=(2, 2, 1),
            fixed_left=left,
            fixed_right=right,
            orientation=orientation)

        assert torch.equal(opening.left_gauge, left)
        assert torch.equal(opening.right_gauge, right)
        assert torch.allclose(
            opening.contract_dense(), target, rtol=2e-10, atol=2e-10)
        assert opening.local_records[0].rel_residual < 1e-10

    def test_requires_both_gauges_and_one_physical_site(self):
        left = torch.randn(1, 2, 1)
        right = torch.randn(1, 2, 1)
        opener = tk.decompositions.FixedGaugeCoreOpener()

        with pytest.raises(ValueError, match='Both fixed gauges'):
            opener.open(torch.randn(2, 2, 2), rank=1, fixed_left=left)
        with pytest.raises(ValueError, match='exactly one physical site'):
            opener.open(
                torch.randn(2, 2, 2, 2),
                rank=1,
                fixed_left=left,
                fixed_right=right)


class TestOpeningAdapters:  # MARK: TestOpeningAdapters

    def test_shared_resolver_preserves_advanced_openers_and_als_preset(self):
        advanced = tk.decompositions.ALSLoopOpener()

        assert resolve_loop_opener(advanced) is advanced
        assert isinstance(
            resolve_loop_opener('als'), tk.decompositions.ALSLoopOpener)

    def test_callable_adapter_forwards_normalized_arguments(self):
        cores, tensor = _local_problem()
        calls = []

        def function(**kwargs):
            calls.append(kwargs)
            return tk.decompositions.LoopOpening(
                left_gauge=cores[0],
                cores=(cores[1],),
                right_gauge=cores[2],
                rank=(2, 3, 2),
                orientation=kwargs['orientation'],
                diagnostics={'algorithm': 'callable'})

        opener = tk.decompositions.CallableLoopOpener(
            function,
            tk.decompositions.LoopOpenerCapabilities(
                supports_blocks=True))
        opening = opener.open(
            tensor,
            rank=(2, 3, 2),
            orientation='left')

        assert len(calls) == 1
        assert calls[0]['orientation'] == 'left'
        assert opening.diagnostics['algorithm'] == 'callable'

    def test_composite_uses_initial_opening_for_als_refinement(self):
        cores, tensor = _local_problem()

        def initialize(**kwargs):
            return tk.decompositions.LoopOpening(
                left_gauge=cores[0],
                cores=(cores[1],),
                right_gauge=cores[2],
                rank=(2, 3, 2),
                orientation=kwargs['orientation'],
                diagnostics={'algorithm': 'synthetic_initializer'})

        initializer = tk.decompositions.CallableLoopOpener(
            initialize,
            tk.decompositions.LoopOpenerCapabilities(
                supports_blocks=True))
        refiner = tk.decompositions.ALSLoopOpener({
            'gauge': 'none',
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=1),
        })
        opening = tk.decompositions.CompositeLoopOpener(
            initializer, refiner).open(tensor, rank=(2, 3, 2))

        assert torch.allclose(
            opening.contract_dense(), tensor, rtol=2e-10, atol=2e-10)
        assert opening.diagnostics['initializer']['algorithm'] == \
            'synthetic_initializer'
        assert opening.diagnostics['initializer']['used'] is True

    def test_composite_can_fall_back_after_clean_initializer_failure(self):
        cores, tensor = _local_problem()

        def initialize(**kwargs):
            raise RuntimeError('synthetic spectral failure')

        def refine(**kwargs):
            return tk.decompositions.LoopOpening(
                left_gauge=cores[0],
                cores=(cores[1],),
                right_gauge=cores[2],
                rank=(2, 3, 2),
                orientation=kwargs['orientation'],
                diagnostics={'algorithm': 'synthetic_refiner'})

        capabilities = tk.decompositions.LoopOpenerCapabilities(
            supports_blocks=True)
        initializer = tk.decompositions.CallableLoopOpener(
            initialize, capabilities)
        refiner = tk.decompositions.CallableLoopOpener(refine, capabilities)
        opening = tk.decompositions.CompositeLoopOpener(
            initializer,
            refiner,
            fallback_on_error=True).open(tensor, rank=(2, 3, 2))

        assert torch.allclose(opening.contract_dense(), tensor)
        assert opening.diagnostics['initializer'] == {
            'used': False,
            'error': 'synthetic spectral failure',
        }

    def test_composite_preserves_initializer_errors_by_default(self):
        _, tensor = _local_problem()

        def initialize(**kwargs):
            raise RuntimeError('synthetic spectral failure')

        def refine(**kwargs):
            raise AssertionError('refiner should not run')

        capabilities = tk.decompositions.LoopOpenerCapabilities(
            supports_blocks=True)
        opener = tk.decompositions.CompositeLoopOpener(
            tk.decompositions.CallableLoopOpener(
                initialize, capabilities),
            tk.decompositions.CallableLoopOpener(refine, capabilities))

        with pytest.raises(RuntimeError, match='synthetic spectral failure'):
            opener.open(tensor, rank=(2, 3, 2))

    def test_als_refiner_casts_real_target_to_complex_initializer(self):
        real_cores, tensor = _local_problem()
        complex_cores = tuple(
            core.to(torch.complex128) for core in real_cores)

        def initialize(**kwargs):
            return tk.decompositions.LoopOpening(
                left_gauge=complex_cores[0],
                cores=(complex_cores[1],),
                right_gauge=complex_cores[2],
                rank=(2, 3, 2),
                orientation=kwargs['orientation'])

        initializer = tk.decompositions.CallableLoopOpener(
            initialize,
            tk.decompositions.LoopOpenerCapabilities(
                supports_blocks=True))
        refiner = tk.decompositions.ALSLoopOpener({
            'gauge': 'none',
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=1),
        })
        opening = tk.decompositions.CompositeLoopOpener(
            initializer, refiner).open(tensor, rank=(2, 3, 2))

        assert all(core.dtype == torch.complex128
                   for core in opening.all_cores)
        assert torch.allclose(
            opening.contract_dense(),
            tensor.to(torch.complex128),
            rtol=2e-10,
            atol=2e-10)


def test_callable_loop_opener_accepts_quantics_decomposition_result():
    result = tk.decompositions.QTRDecomposition(
        [torch.ones(1, 2, 1) for _ in range(3)],
        n_coordinates=1, base=2, level=3, domain=(0., 1.),
        metadata={'fit': 'quantics'})
    opener = tk.decompositions.CallableLoopOpener(
        lambda **kwargs: result,
        tk.decompositions.LoopOpenerCapabilities(supports_blocks=True))
    opening = opener.open(result.contract_dense(), rank=(1, 1, 1))
    assert opening.diagnostics['metadata'] == result.metadata
    torch.testing.assert_close(opening.contract_dense(), result.contract_dense())
