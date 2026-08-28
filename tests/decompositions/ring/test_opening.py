"""Tests for local loop-opening contracts and strategies."""

import pytest

import torch
import tensorkrowch as tk

from tests.decompositions.als._oracles import (contract_tr_dense,
                                               make_tr_cores)


def _local_problem(dtype=torch.float64):
    """Creates one exact three-site local TR target."""
    cores = make_tr_cores(
        input_dim=(3, 2, 4),
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
        with pytest.raises(ValueError, match='physical blocks'):
            capabilities.require(
                fixed_left=False, fixed_right=False, block_size=2)


class TestALSLoopOpener:  # MARK: TestALSLoopOpener

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('orientation', ['right', 'left'])
    def test_initial_opening_matches_target_in_both_orientations(
            self, dtype, orientation):
        cores, tensor = _local_problem(dtype)
        opener = tk.decompositions.ALSLoopOpener({
            'gauge': 'none',
            'normalize': False,
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
            'normalize': False,
            'convergence': tk.decompositions.ConvergencePolicy(max_sweeps=1),
        })
        opening = opener.open(
            function,
            rank=(2, 3, 2),
            orientation='left',
            context={
                'input_dim': tensor.shape,
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
        assert opening.local_records[0].residual_relative < 1e-10

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
            'normalize': False,
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
            'normalize': False,
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
