"""
This script contains tests for utils:

    * TestTruncatedSVD
"""

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.utils import _compact_svd


def _reconstruct_svd(u, s, vh):
    return (u * s.unsqueeze(-2)) @ vh


def _adjoint(tensor):
    return tensor.transpose(-2, -1).conj()


def _svd_tolerances(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 2e-4, 2e-5
    return 1e-10, 1e-12


def _controlled_spectrum_matrix(shape,
                                dtype,
                                singular_values=None,
                                seed=0):
    generator = torch.Generator().manual_seed(seed)
    rows, columns = shape[-2:]
    compact_rank = min(rows, columns)
    u_seed = torch.randn(
        shape[:-2] + (rows, compact_rank),
        dtype=dtype,
        generator=generator)
    v_seed = torch.randn(
        shape[:-2] + (columns, compact_rank),
        dtype=dtype,
        generator=generator)
    u, _ = torch.linalg.qr(u_seed, mode='reduced')
    v, _ = torch.linalg.qr(v_seed, mode='reduced')

    real_dtype = (
        torch.float32
        if dtype in (torch.float32, torch.complex64)
        else torch.float64)
    if singular_values is None:
        singular_values = torch.arange(
            compact_rank + 1,
            1,
            -1,
            dtype=real_dtype)
    else:
        singular_values = torch.tensor(singular_values, dtype=real_dtype)

    return (u * singular_values) @ _adjoint(v)


class TestTruncatedSVD:  # MARK: TestTruncatedSVD

    @pytest.fixture
    def diag_tensor(self):
        # Singular values are exactly [5.0, 3.0, 1.0, 0.1]
        return torch.diag(torch.tensor([5.0, 3.0, 1.0, 0.1]))

    @pytest.mark.parametrize(
        'kwargs, expected_rank',
        [
            ({}, 4),
            ({'rank': 2}, 2),
            ({'cutoff': 1.0}, 2),
            ({'atol': 1.05}, 2),
            ({'rtol': 0.03}, 2),
            ({'cum_percentage': 0.97}, 2),
            ({'rank': 3, 'cutoff': 1.0, 'atol': 1.05}, 2),
            ({'rank': 4, 'rtol': 0.03, 'cum_percentage': 0.97}, 2),
        ],
    )
    @pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
    def test_truncated_svd_rank_selection(self,
                                          diag_tensor,
                                          kwargs,
                                          expected_rank,
                                          svd_method):
        u, s, vh = tk.utils.truncated_svd(
            diag_tensor,
            svd_method=svd_method,
            **kwargs)

        assert u.shape == (4, expected_rank)
        assert s.shape == (expected_rank,)
        assert vh.shape == (expected_rank, 4)
        assert torch.allclose(
            s,
            torch.tensor([5.0, 3.0, 1.0, 0.1])[:expected_rank])

    @pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
    def test_truncated_svd_batched_rank_selection(self, svd_method):
        tensor = torch.stack([
            torch.diag(torch.tensor([5.0, 3.0, 1.0, 0.1])),
            torch.diag(torch.tensor([4.0, 2.0, 0.5, 0.05])),
        ])

        u, s, vh = tk.utils.truncated_svd(
            tensor,
            cutoff=0.5,
            atol=1.0,
            svd_method=svd_method)

        # cutoff gives rank 3 (because one batch has 1.0), atol gives rank 3 -> final rank 3
        assert u.shape == (2, 4, 3)
        assert s.shape == (2, 3)
        assert vh.shape == (2, 3, 4)

    @pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
    def test_truncated_svd_info(self, diag_tensor, svd_method):
        u, s, vh, info = tk.utils.truncated_svd(
            diag_tensor,
            rank=2,
            svd_method=svd_method,
            return_info=True)

        assert u.shape == (4, 2)
        assert s.shape == (2,)
        assert vh.shape == (2, 4)
        assert info.full_rank == 4
        assert info.selected_rank == 2
        assert info.svd_method == svd_method
        assert torch.allclose(info.total_squared_norm,
                              torch.tensor(35.01))
        assert torch.allclose(info.discarded_squared_norm,
                              torch.tensor(1.01))
        assert info.total_squared_norm.device == diag_tensor.device
        assert info.discarded_squared_norm.device == diag_tensor.device
        assert info.total_squared_norm.dtype == s.dtype
        assert info.discarded_squared_norm.dtype == s.dtype
        assert info.total_squared_norm_per_batch.shape == ()
        assert info.discarded_squared_norm_per_batch.shape == ()
        assert 'singular_values' not in info._fields

    @pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
    def test_truncated_svd_batched_info(self, svd_method):
        tensor = torch.stack([
            torch.diag(torch.tensor([5.0, 3.0, 1.0, 0.1])),
            torch.diag(torch.tensor([4.0, 2.0, 0.5, 0.05])),
        ])

        _, _, _, info = tk.utils.truncated_svd(
            tensor,
            cutoff=0.5,
            svd_method=svd_method,
            return_info=True)

        expected_total = torch.tensor([35.01, 20.2525])
        expected_discarded = torch.tensor([0.01, 0.0025])
        assert info.selected_rank == 3
        assert torch.allclose(info.total_squared_norm_per_batch,
                              expected_total)
        assert torch.allclose(info.discarded_squared_norm_per_batch,
                              expected_discarded)
        assert torch.allclose(info.total_squared_norm,
                              expected_total.sum())
        assert torch.allclose(info.discarded_squared_norm,
                              expected_discarded.sum())

    def test_truncated_svd_info_records_active_backend(self, diag_tensor):
        with tk.svd_method('qr_svd'):
            *_, context_info = tk.utils.truncated_svd(
                diag_tensor, return_info=True)
            *_, override_info = tk.utils.truncated_svd(
                diag_tensor,
                svd_method='svd',
                return_info=True)

        assert context_info.svd_method == 'qr_svd'
        assert override_info.svd_method == 'svd'

    def test_truncated_svd_info_uses_one_decomposition(self,
                                                       diag_tensor,
                                                       monkeypatch):
        calls = 0
        compact_svd = tk.utils._compact_svd

        def compact_svd_spy(*args, **kwargs):
            nonlocal calls
            calls += 1
            return compact_svd(*args, **kwargs)

        monkeypatch.setattr(tk.utils, '_compact_svd', compact_svd_spy)

        tk.utils.truncated_svd(diag_tensor, rank=2, return_info=True)

        assert calls == 1

    def test_truncated_svd_default_return_is_unchanged(self, diag_tensor):
        result = tk.utils.truncated_svd(diag_tensor, rank=2)

        assert isinstance(result, tuple)
        assert len(result) == 3

    @pytest.mark.parametrize(
        'kwargs, error_type',
        [
            ({'rank': 0}, ValueError),
            ({'rank': True}, TypeError),
            ({'rank': 1.5}, TypeError),
            ({'cutoff': -1.0}, ValueError),
            ({'cutoff': '1'}, TypeError),
            ({'cutoff': True}, TypeError),
            ({'atol': -0.1}, ValueError),
            ({'atol': float('nan')}, ValueError),
            ({'rtol': -0.1}, ValueError),
            ({'rtol': 1.1}, ValueError),
            ({'rtol': float('inf')}, ValueError),
            ({'cum_percentage': -0.1}, ValueError),
            ({'cum_percentage': 1.1}, ValueError),
            ({'cum_percentage': True}, TypeError),
        ],
    )
    def test_truncated_svd_invalid_arguments(
            self, diag_tensor, kwargs, error_type):
        with pytest.raises(error_type):
            tk.utils.truncated_svd(diag_tensor, **kwargs)

    @pytest.mark.parametrize('criterion', ['rtol', 'cum_percentage'])
    def test_zero_spectrum_keeps_only_minimum_rank(self, criterion):
        kwargs = ({'rtol': 0.1} if criterion == 'rtol'
                  else {'cum_percentage': 0.9})

        _, s, _ = tk.utils.truncated_svd(
            torch.zeros(5, 5), **kwargs)

        assert s.shape == (1,)
        assert s.item() == 0

    def test_zero_batch_does_not_inflate_shared_relative_rank(self):
        tensor = torch.stack([
            torch.zeros(4, 4),
            torch.diag(torch.tensor([3.0, 1.0, 0.0, 0.0])),
        ])

        _, s, _ = tk.utils.truncated_svd(tensor, rtol=0.05)

        assert s.shape == (2, 2)

    @pytest.mark.parametrize(
        'shape',
        [
            (9, 4),
            (4, 9),
            (6, 6),
            (2, 3, 8, 4),
            (2, 3, 4, 8),
        ],
        ids=['tall', 'wide', 'square', 'batched-tall', 'batched-wide'],
    )
    @pytest.mark.parametrize(
        'dtype',
        [torch.float32, torch.float64, torch.complex64, torch.complex128],
    )
    def test_compact_svd_backend_parity(self, shape, dtype):
        tensor = _controlled_spectrum_matrix(shape, dtype)

        u_ref, s_ref, vh_ref = _compact_svd(tensor, svd_method='svd')
        u, s, vh = _compact_svd(tensor, svd_method='qr_svd')

        rtol, atol = _svd_tolerances(dtype)
        assert u.shape == u_ref.shape
        assert s.shape == s_ref.shape
        assert vh.shape == vh_ref.shape
        assert u.dtype == tensor.dtype
        assert s.dtype == s_ref.dtype
        assert vh.dtype == tensor.dtype
        assert u.device == tensor.device
        assert s.device == tensor.device
        assert vh.device == tensor.device
        assert torch.allclose(s, s_ref, rtol=rtol, atol=atol)
        assert torch.allclose(
            _reconstruct_svd(u, s, vh), tensor, rtol=rtol, atol=atol)

        identity = torch.eye(
            s.shape[-1], dtype=tensor.dtype, device=tensor.device)
        assert torch.allclose(
            _adjoint(u) @ u, identity, rtol=rtol, atol=atol)
        assert torch.allclose(
            vh @ _adjoint(vh), identity, rtol=rtol, atol=atol)

        subspace_rank = min(2, s.shape[-1])
        u_projector = u[..., :subspace_rank] @ _adjoint(
            u[..., :subspace_rank])
        u_ref_projector = (
            u_ref[..., :subspace_rank] @
            _adjoint(u_ref[..., :subspace_rank]))
        v = _adjoint(vh[..., :subspace_rank, :])
        v_ref = _adjoint(vh_ref[..., :subspace_rank, :])
        v_projector = v @ _adjoint(v)
        v_ref_projector = v_ref @ _adjoint(v_ref)
        assert torch.allclose(
            u_projector, u_ref_projector, rtol=rtol, atol=atol)
        assert torch.allclose(
            v_projector, v_ref_projector, rtol=rtol, atol=atol)

    @pytest.mark.parametrize('shape', [(7, 4), (4, 7)])
    @pytest.mark.parametrize('svd_method', ['svd', 'qr_svd'])
    def test_compact_svd_zero_and_rank_deficient(self, shape, svd_method):
        zero = torch.zeros(shape, dtype=torch.complex128)
        u, s, vh = _compact_svd(zero, svd_method=svd_method)

        assert torch.equal(s, torch.zeros_like(s))
        assert torch.isfinite(u).all()
        assert torch.isfinite(vh).all()
        assert torch.equal(_reconstruct_svd(u, s, vh), zero)

        diagonal_size = min(shape)
        rank_deficient = _controlled_spectrum_matrix(
            shape,
            torch.float64,
            [4.0, 2.0] + [0.0] * (diagonal_size - 2),
            seed=5)
        u_ref, _, vh_ref = _compact_svd(
            rank_deficient, svd_method='svd')
        u, s, vh = _compact_svd(rank_deficient, svd_method=svd_method)
        assert torch.allclose(
            _reconstruct_svd(u, s, vh), rank_deficient,
            rtol=1e-12, atol=1e-12)
        assert torch.allclose(
            u[..., :2] @ _adjoint(u[..., :2]),
            u_ref[..., :2] @ _adjoint(u_ref[..., :2]),
            rtol=1e-12, atol=1e-12)
        assert torch.allclose(
            _adjoint(vh[..., :2, :]) @ vh[..., :2, :],
            _adjoint(vh_ref[..., :2, :]) @ vh_ref[..., :2, :],
            rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize('shape', [(7, 4), (4, 7)])
    def test_compact_svd_repeated_singular_subspace(self, shape):
        diagonal_size = min(shape)
        tensor = _controlled_spectrum_matrix(
            shape,
            torch.float64,
            [4.0, 4.0, 1.0] + [0.0] * (diagonal_size - 3),
            seed=6)

        u_ref, s_ref, vh_ref = _compact_svd(tensor, svd_method='svd')
        u, s, vh = _compact_svd(tensor, svd_method='qr_svd')

        assert torch.allclose(s, s_ref, rtol=1e-12, atol=1e-12)
        u_projector = u[..., :2] @ _adjoint(u[..., :2])
        u_ref_projector = u_ref[..., :2] @ _adjoint(u_ref[..., :2])
        v_projector = _adjoint(vh[..., :2, :]) @ vh[..., :2, :]
        v_ref_projector = (
            _adjoint(vh_ref[..., :2, :]) @ vh_ref[..., :2, :])
        assert torch.allclose(
            u_projector, u_ref_projector, rtol=1e-12, atol=1e-12)
        assert torch.allclose(
            v_projector, v_ref_projector, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize('transpose', [False, True],
                             ids=['tall', 'wide'])
    def test_compact_svd_noncontiguous(self, transpose):
        generator = torch.Generator().manual_seed(2)
        if transpose:
            tensor = torch.randn(
                8, 9, dtype=torch.float64, generator=generator)[::2, :]
        else:
            tensor = torch.randn(
                9, 8, dtype=torch.float64, generator=generator)[:, ::2]
        assert not tensor.is_contiguous()

        u, s, vh = _compact_svd(tensor, svd_method='qr_svd')

        assert torch.allclose(
            _reconstruct_svd(u, s, vh), tensor, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize('shape', [(7, 4), (4, 7)])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_compact_svd_gradient_parity(self, shape, dtype):
        tensor = _controlled_spectrum_matrix(
            shape, dtype, seed=7).detach().requires_grad_()
        generator = torch.Generator().manual_seed(8)
        probe = torch.randn(shape, dtype=dtype, generator=generator)

        losses = []
        gradients = []
        for svd_method in ('svd', 'qr_svd'):
            u, s, vh = _compact_svd(tensor, svd_method=svd_method)
            approx = _reconstruct_svd(u[..., :2], s[..., :2], vh[..., :2, :])
            loss = (approx.conj() * probe).sum().real
            loss = loss + 0.1 * s[..., :2].square().sum()
            gradient, = torch.autograd.grad(loss, tensor, retain_graph=True)
            losses.append(loss)
            gradients.append(gradient)

        assert torch.allclose(losses[0], losses[1], rtol=1e-9, atol=1e-10)
        assert torch.allclose(
            gradients[0], gradients[1], rtol=1e-7, atol=1e-8)

    def test_compact_svd_gradcheck(self):
        tensor = _controlled_spectrum_matrix(
            (5, 3), torch.float64, seed=9).detach().requires_grad_()

        def truncated_reconstruction(value):
            u, s, vh = _compact_svd(value, svd_method='qr_svd')
            return _reconstruct_svd(u[..., :2], s[..., :2], vh[..., :2, :])

        assert torch.autograd.gradcheck(
            truncated_reconstruction,
            (tensor,),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3)

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason='CUDA is not available')
    @pytest.mark.parametrize('shape', [(64, 16), (16, 64)])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
    def test_compact_svd_cuda(self, shape, dtype):
        tensor = torch.randn(shape, device='cuda', dtype=dtype)

        u, s, vh = _compact_svd(tensor, svd_method='qr_svd')
        *_, info = tk.utils.truncated_svd(
            tensor,
            rank=2,
            svd_method='qr_svd',
            return_info=True)

        assert torch.allclose(
            _reconstruct_svd(u, s, vh), tensor, rtol=2e-4, atol=2e-5)
        assert info.total_squared_norm.device.type == 'cuda'
        assert info.discarded_squared_norm.device.type == 'cuda'

    @pytest.mark.parametrize(
        'svd_method, error_type',
        [(1, TypeError), ('invalid', ValueError)],
    )
    def test_truncated_svd_invalid_method(self,
                                          diag_tensor,
                                          svd_method,
                                          error_type):
        with pytest.raises(error_type):
            tk.utils.truncated_svd(diag_tensor, svd_method=svd_method)

    def test_truncated_svd_invalid_tensor_rank(self):
        with pytest.raises(RuntimeError, match='at least 2 dimensions'):
            tk.utils.truncated_svd(torch.ones(3))

    def test_truncated_svd_invalid_tensor_type(self):
        with pytest.raises(TypeError, match='torch.Tensor type'):
            tk.utils.truncated_svd([[1.0, 0.0], [0.0, 1.0]])

    def test_truncated_svd_invalid_return_info(self, diag_tensor):
        with pytest.raises(TypeError, match='`return_info` should be bool'):
            tk.utils.truncated_svd(diag_tensor, return_info=1)
