"""Tests for the isolated experimental BLOSTR decomposition."""

import pytest

import torch
import tensorkrowch as tk

from tests.decompositions.als._oracles import contract_tr_dense


def _uniform_tr(dtype=torch.float64, n_sites=3):
    """Creates one generic rank-two TR satisfying BLOSTR dimensions."""
    generator = torch.Generator().manual_seed(300 + n_sites)
    input_dim = (4,) * n_sites
    cores = [
        torch.randn(2, dimension, 2, dtype=dtype, generator=generator)
        for dimension in input_dim
    ]
    return cores, contract_tr_dense(cores)


class TestTRBLOSTR:  # MARK: TestTRBLOSTR

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_explicit_slices_recover_synthetic_tr(self, dtype):
        _, tensor = _uniform_tr(dtype)

        with pytest.warns(tk.decompositions.ExperimentalWarning):
            cores, info = tk.decompositions.tr_blostr(
                tensor,
                rank=2,
                slices=((0,), (1,), (2,), (3,)),
                n_iters=20,
                n_restarts=3,
                output_device=None,
                return_info=True)

        approximation = contract_tr_dense(cores)
        relative = (approximation - tensor).norm() / tensor.norm()
        assert [tuple(core.shape) for core in cores] == [
            (2, 4, 2), (2, 4, 2), (2, 4, 2)]
        assert relative < 1e-10
        assert info['metadata']['algorithm'] == 'blostr'
        assert info['metadata']['experimental'] is True
        assert info['metrics']['errors'][0]['relative'] == pytest.approx(
            relative.item(), rel=1e-10, abs=1e-12)

    def test_generator_reproduces_slice_search(self):
        _, tensor = _uniform_tr()
        outputs = []
        for _ in range(2):
            with pytest.warns(tk.decompositions.ExperimentalWarning):
                outputs.append(tk.decompositions.tr_blostr(
                    tensor,
                    rank=(2, 2, 2),
                    n_attempts=4,
                    n_iters=10,
                    n_restarts=2,
                    generator=torch.Generator().manual_seed(81),
                    output_device=None,
                    return_info=True))

        first_cores, first_info = outputs[0]
        second_cores, second_info = outputs[1]
        assert first_info['metadata']['slices'] == \
            second_info['metadata']['slices']
        assert first_info['metadata']['failed_attempts'] == \
            second_info['metadata']['failed_attempts']
        assert torch.allclose(
            contract_tr_dense(first_cores),
            contract_tr_dense(second_cores),
            rtol=1e-12,
            atol=1e-12)

    def test_full_recovery_supports_more_than_one_tail_cut(self):
        _, tensor = _uniform_tr(n_sites=4)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            cores = tk.decompositions.tr_blostr(
                tensor,
                rank=2,
                slices=((0, 0), (1, 0), (2, 1), (3, 2)),
                n_iters=20,
                n_restarts=3,
                output_device=None)

        assert [tuple(core.shape) for core in cores] == [(2, 4, 2)] * 4
        assert torch.allclose(
            contract_tr_dense(cores),
            tensor.to(cores[0].dtype),
            rtol=1e-9,
            atol=1e-9)

    def test_degenerate_spectrum_fails_cleanly(self):
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            with pytest.raises(RuntimeError, match='spectral attempt failed'):
                tk.decompositions.tr_blostr(
                    torch.zeros(4, 4, 4, dtype=torch.float64),
                    rank=2,
                    slices=((0,), (1,), (2,), (3,)),
                    output_device=None)

    def test_rejects_nonuniform_ranks_and_insufficient_dimensions(self):
        tensor = torch.randn(
            4, 4, 4,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(304))
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            with pytest.raises(ValueError, match='equal TR ranks'):
                tk.decompositions.tr_blostr(tensor, rank=(2, 3, 2))
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            with pytest.raises(RuntimeError, match='first input dimension'):
                tk.decompositions.tr_blostr(
                    tensor[:3], rank=2, n_attempts=1)


class TestBLOSTRLoopOpener:  # MARK: TestBLOSTRLoopOpener

    @pytest.mark.parametrize('orientation', ['right', 'left'])
    def test_opens_unrestricted_block_in_both_orientations(self, orientation):
        _, tensor = _uniform_tr()
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            opener = tk.decompositions.BLOSTRLoopOpener({
                'slices': ((0,), (1,), (2,), (3,)),
                'n_iters': 20,
                'n_restarts': 3,
            })
        opening = opener.open(
            tensor,
            rank=2,
            orientation=orientation)

        assert opening.orientation == orientation
        assert opening.rank == (2, 2, 2)
        assert torch.allclose(
            opening.contract_dense(),
            tensor.to(opening.all_cores[0].dtype),
            rtol=1e-10,
            atol=1e-10)
        assert opening.diagnostics['algorithm'] == 'blostr'

    def test_left_orientation_reverses_multisite_slice_coordinates(self):
        _, tensor = _uniform_tr(n_sites=4)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            opener = tk.decompositions.BLOSTRLoopOpener({
                'slices': ((0, 0), (1, 0), (2, 1), (3, 2)),
                'n_iters': 20,
                'n_restarts': 3,
            })
        opening = opener.open(tensor, rank=2, orientation='left')

        assert torch.allclose(
            opening.contract_dense(),
            tensor.to(opening.all_cores[0].dtype),
            rtol=1e-9,
            atol=1e-9)

    def test_capabilities_reject_fixed_gauges_before_spectral_work(self):
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            opener = tk.decompositions.BLOSTRLoopOpener()
        gauge = torch.randn(2, 4, 2)

        with pytest.raises(ValueError, match='fixed left'):
            opener.open(
                torch.zeros(4, 4, 4),
                rank=2,
                fixed_left=gauge)

    def test_materializes_callable_through_shared_source_contract(self):
        _, tensor = _uniform_tr()

        def function(indices):
            return tensor[
                indices[:, 0], indices[:, 1], indices[:, 2]]

        with pytest.warns(tk.decompositions.ExperimentalWarning):
            opener = tk.decompositions.BLOSTRLoopOpener({
                'slices': ((0,), (1,), (2,), (3,)),
                'n_iters': 20,
                'n_restarts': 3,
            })
        opening = opener.open(
            function,
            rank=2,
            context={
                'input_dim': tensor.shape,
                'dtype': tensor.dtype,
            })

        assert torch.allclose(
            opening.contract_dense(),
            tensor.to(opening.all_cores[0].dtype),
            rtol=1e-10,
            atol=1e-10)
