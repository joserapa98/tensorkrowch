"""Tests for TensorKrowch runtime configuration."""

import os
import subprocess
import sys

import pytest
import torch

import tensorkrowch as tk


class TestSVDMethodConfiguration:  # MARK: TestSVDMethodConfiguration

    def test_refinement_is_independent_and_nested(self):
        previous = tk.get_svd_refinement()
        try:
            tk.set_svd_refinement(False)
            with tk.svd_method('qr_svd', refine=True):
                assert tk.get_svd_refinement()
                with tk.svd_method('svd'):
                    assert tk.get_svd_refinement()
                    assert tk.get_svd_method() == 'svd'
                with tk.svd_method('svd', refine=False):
                    assert not tk.get_svd_refinement()
                assert tk.get_svd_refinement()
            assert not tk.get_svd_refinement()
        finally:
            tk.set_svd_refinement(previous)

    def test_refinement_restores_after_exception(self):
        previous = tk.get_svd_refinement()
        with pytest.raises(RuntimeError, match='context failure'):
            with tk.svd_method('qr_svd', refine=not previous):
                raise RuntimeError('context failure')
        assert tk.get_svd_refinement() == previous

    def test_process_refinement_default(self):
        previous = tk.get_svd_refinement()
        try:
            tk.set_svd_refinement(True)
            assert tk.get_svd_refinement()
            with tk.svd_method('svd', refine=False):
                assert not tk.get_svd_refinement()
            assert tk.get_svd_refinement()
        finally:
            tk.set_svd_refinement(previous)

    @pytest.mark.parametrize('refine', [1, 'yes', []])
    def test_invalid_refinement_does_not_change_context(self, refine):
        previous_method = tk.get_svd_method()
        previous_refinement = tk.get_svd_refinement()
        with pytest.raises(TypeError):
            tk.set_svd_refinement(refine)
        with pytest.raises(TypeError):
            with tk.svd_method('qr_svd', refine=refine):
                pass
        assert tk.get_svd_method() == previous_method
        assert tk.get_svd_refinement() == previous_refinement

    @pytest.mark.parametrize('method', ['svd', 'qr_svd'])
    def test_refinement_reaches_truncation_and_model(self, method):
        tensor = torch.diag(torch.tensor([1., 1e-6, 1e-12],
                                         dtype=torch.float64))
        with tk.svd_method(method, refine=True):
            u, s, vh, info = tk.utils.truncated_svd(
                tensor, rank=2, return_info=True)
            mps = tk.models.MPS(tensors=[tensor[:, :2],
                                          torch.eye(2, dtype=tensor.dtype)],
                                parameterized=False)
            before = mps.tensors
            mps.canonicalize(oc=1, rank=2)
        assert info.svd_method == method
        expected = tensor.clone()
        expected[-1, -1] = 0
        assert torch.allclose((u * s.unsqueeze(-2)) @ vh, expected,
                              atol=1e-20, rtol=1e-12)
        assert torch.allclose(mps.tensors[0] @ mps.tensors[1],
                              before[0] @ before[1], atol=1e-15, rtol=1e-12)

    @pytest.mark.parametrize('method', ['svd', 'qr_svd'])
    @pytest.mark.parametrize('decomposer, shape', [
        (tk.decompositions.TTSVD, (2, 3, 2)),
        (tk.decompositions.TRSVD, (2, 3, 2)),
        (tk.decompositions.TTMSVD, (2, 3, 2, 2)),
        (tk.decompositions.TRMSVD, (2, 3, 2, 2)),
    ])
    def test_refinement_reaches_decompositions(self, method, decomposer, shape):
        generator = torch.Generator().manual_seed(17)
        tensor = torch.randn(shape, dtype=torch.float64, generator=generator)
        with tk.svd_method(method, refine=True):
            result = decomposer(tensor).fit()
        assert torch.allclose(result.contract_dense(), tensor,
                              rtol=1e-10, atol=1e-12)

    def test_set_svd_method(self):
        previous_method = tk.get_svd_method()
        try:
            tk.set_svd_method('qr_svd')
            assert tk.get_svd_method() == 'qr_svd'

            tk.set_svd_method('svd')
            assert tk.get_svd_method() == 'svd'
        finally:
            tk.set_svd_method(previous_method)

    def test_svd_method_context_is_nested_and_temporary(self):
        previous_method = tk.get_svd_method()

        with tk.svd_method('qr_svd'):
            assert tk.get_svd_method() == 'qr_svd'
            with tk.svd_method('svd'):
                assert tk.get_svd_method() == 'svd'
            assert tk.get_svd_method() == 'qr_svd'

        assert tk.get_svd_method() == previous_method

    def test_svd_method_context_restores_after_error(self):
        previous_method = tk.get_svd_method()

        with pytest.raises(RuntimeError, match='context failure'):
            with tk.svd_method('qr_svd'):
                raise RuntimeError('context failure')

        assert tk.get_svd_method() == previous_method

    @pytest.mark.parametrize('method, error_type', [(None, TypeError),
                                                     ('auto', ValueError)])
    def test_invalid_svd_method(self, method, error_type):
        with pytest.raises(error_type):
            tk.set_svd_method(method)
        with pytest.raises(error_type):
            with tk.svd_method(method):
                pass

    def test_truncated_svd_uses_context_backend(self, monkeypatch):
        tensor = torch.randn(8, 4, dtype=torch.float64)
        original_qr = torch.linalg.qr
        qr_calls = []

        def recording_qr(*args, **kwargs):
            qr_calls.append(args[0].shape)
            return original_qr(*args, **kwargs)

        monkeypatch.setattr(torch.linalg, 'qr', recording_qr)
        with tk.svd_method('qr_svd'):
            u, s, vh = tk.utils.truncated_svd(tensor)

        assert qr_calls == [tensor.shape]
        assert torch.allclose(
            (u * s.unsqueeze(-2)) @ vh,
            tensor,
            rtol=1e-10,
            atol=1e-12)

    def test_explicit_kernel_method_overrides_context(self, monkeypatch):
        def unexpected_qr(*args, **kwargs):
            raise AssertionError('QR backend should not be used')

        monkeypatch.setattr(torch.linalg, 'qr', unexpected_qr)
        with tk.svd_method('qr_svd'):
            tk.utils.truncated_svd(
                torch.randn(4, 4),
                svd_method='svd')

    def test_context_reaches_canonicalize(self, monkeypatch):
        original_qr = torch.linalg.qr
        qr_calls = []

        def recording_qr(*args, **kwargs):
            qr_calls.append(args[0].shape)
            return original_qr(*args, **kwargs)

        monkeypatch.setattr(torch.linalg, 'qr', recording_qr)
        mps = tk.models.MPS(n_features=3,
                            phys_dim=2,
                            bond_dim=2,
                            init_method='randn')
        with tk.svd_method('qr_svd'):
            mps.canonicalize(rank=2)

        assert qr_calls

    def test_environment_variable_sets_initial_method(self):
        environment = os.environ.copy()
        environment['TENSORKROWCH_SVD_METHOD'] = 'qr_svd'
        result = subprocess.run(
            [sys.executable,
             '-c',
             'import tensorkrowch as tk; print(tk.get_svd_method())'],
            check=True,
            capture_output=True,
            text=True,
            env=environment)

        assert result.stdout.strip() == 'qr_svd'
