"""Tests for TensorKrowch runtime configuration."""

import os
import subprocess
import sys

import pytest
import torch

import tensorkrowch as tk


class TestSVDMethodConfiguration:  # MARK: TestSVDMethodConfiguration

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
