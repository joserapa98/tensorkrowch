"""Tests for TRM-SVD classes and functional interfaces."""

import inspect
from math import sqrt

import pytest

import torch
import tensorkrowch as tk

import tensorkrowch.decompositions.svd.tr as tr_module
import tensorkrowch.decompositions.svd.trm as trm_module


SVD_METHODS = ['svd', 'qr_svd']
DEVICE_NAMES = ['cpu', 'cuda', 'mps']


def _device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    if name == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is not available')
    return torch.device(name)


def _interleave(grouped, n_sites):
    axes = tuple(
        axis
        for site in range(n_sites)
        for axis in (site, n_sites + site))
    return grouped.permute(axes)


def _recoverable_rank_two_trm():
    """Builds a structured TRM recoverable with every rank equal to two."""
    fused_cores = [
        torch.zeros(2, 4, 2, dtype=torch.float64)
        for _ in range(4)
    ]
    weights = torch.tensor([[4., 2.], [3., 1.]], dtype=torch.float64)
    for left_rank in range(2):
        fused_cores[0][left_rank, left_rank, left_rank] = 1
        fused_cores[2][left_rank, left_rank, left_rank] = 1
        for right_rank in range(2):
            fused_cores[1][
                left_rank,
                right_rank ^ left_rank,
                right_rank] = 1
            fused_cores[3][left_rank, right_rank, right_rank] = weights[
                left_rank, right_rank]

    cores = [
        core.reshape(2, 2, 2, 2).permute(0, 1, 3, 2)
        for core in fused_cores
    ]
    result = tk.decompositions.TRMDecomposition(cores)
    return result, result.contract_dense()


def _near_exact_tensor():
    """Builds a TRM tensor with one controlled small singular component."""
    fused = torch.zeros(4, 4, 4, 4, dtype=torch.float64)
    fused[0, 0, 0, 0] = 5
    fused[1, 1, 1, 1] = 1
    fused[2, 2, 2, 2] = 1e-4
    return fused.reshape(*(2, 2) * 4)


class TestTRMSVD:  # MARK: TestTRMSVD

    def test_public_signature_uses_shared_rank_and_internal_observer(self):
        fit_parameters = list(inspect.signature(
            tk.decompositions.TRMSVD.fit).parameters)
        function_parameters = inspect.signature(
            tk.decompositions.trm_svd).parameters

        assert fit_parameters.index('center') < fit_parameters.index('rank')
        assert 'observer' not in fit_parameters
        assert {'in_dim', 'out_dim', 'center', 'out_device'} <= set(
            inspect.signature(tk.decompositions.TRMSVD).parameters)
        assert {'in_dim', 'out_dim', 'center', 'out_device'} <= set(
            function_parameters)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('renormalize', [False, True])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_exact_tensorized_dense_oracle(
            self, svd_method, renormalize, dtype):
        generator = torch.Generator().manual_seed(96)
        tensor = torch.randn(
            2, 3, 2, 2, 2, 3,
            dtype=dtype,
            generator=generator) * 1e-2

        with tk.svd_method(svd_method):
            result = tk.decompositions.TRMSVD(
                tensor, out_device=None).fit(renormalize=renormalize)

        assert result.in_dim == (2, 2, 2)
        assert result.out_dim == (3, 2, 3)
        assert result.dtype == dtype
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_recovers_structured_trm_with_original_rank(self, svd_method):
        _, tensor = _recoverable_rank_two_trm()

        with tk.svd_method(svd_method):
            result = tk.decompositions.TRMSVD(
                tensor, out_device=None).fit(rank=2, cutoff=0)

        assert result.rank == [2, 2, 2, 2]
        assert result.in_dim == (2, 2, 2, 2)
        assert result.out_dim == (2, 2, 2, 2)
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_low_rank_error_has_gaussian_noise_scale(self, svd_method):
        _, tensor = _recoverable_rank_two_trm()
        generator = torch.Generator().manual_seed(92)
        noise = 1e-4 * torch.randn(
            tensor.shape, dtype=tensor.dtype, generator=generator)
        noisy_tensor = tensor + noise

        with tk.svd_method(svd_method):
            approximation = tk.decompositions.TRMSVD(
                noisy_tensor, out_device=None).fit(
                    rank=2, cutoff=1e-3).contract_dense()

        noise_norm = torch.linalg.vector_norm(noise)
        residual_norm = torch.linalg.vector_norm(noisy_tensor - approximation)
        clean_error = torch.linalg.vector_norm(tensor - approximation)
        assert residual_norm <= 1.01 * noise_norm
        assert residual_norm >= 0.2 * noise_norm
        assert clean_error <= 1.5 * noise_norm

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('criterion', ['atol', 'rtol'])
    def test_truncation_criteria_bound_dense_error(
            self, svd_method, criterion):
        tensor = _near_exact_tensor()
        small_value = tensor[(1, 0) * 4].item()
        tensor_norm = torch.linalg.vector_norm(tensor).item()

        if criterion == 'atol':
            tolerance = 1.01 * small_value ** 2
            kwargs = {'atol': tolerance}
        else:
            tolerance = 1.01 * small_value ** 2 / tensor_norm ** 2
            kwargs = {'rtol': tolerance}

        with tk.svd_method(svd_method):
            result = tk.decompositions.TRMSVD(
                tensor, center=2, out_device=None).fit(**kwargs)

        if criterion == 'atol':
            initial_error = sqrt(tolerance)
            left_error = sqrt(tolerance)
            right_error = left_error
        else:
            initial_error = sqrt(tolerance) * tensor_norm
            initial_capacity = result.rank[-1] * result.rank[1]
            selected_rank = (
                initial_capacity - result.metadata['initial_padding'])
            left_error = sqrt(tolerance * selected_rank)
            right_error = sqrt(tolerance) * tensor_norm

        error = torch.linalg.vector_norm(
            tensor - result.contract_dense()).item()
        bound = (
            initial_error
            + left_error * tensor_norm
            + (1 + left_error) * right_error)
        assert error > 0
        assert error <= bound * (1 + 1e-8)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('renormalize', [False, True])
    def test_interleaved_grouped_and_matrix_routes_are_equivalent(
            self, svd_method, renormalize):
        generator = torch.Generator().manual_seed(93)
        in_dim = (2, 3, 2)
        out_dim = (3, 2, 2)
        grouped = torch.randn(
            *in_dim, *out_dim,
            dtype=torch.float64,
            generator=generator) * 1e2
        interleaved = _interleave(grouped, len(in_dim))
        matrix = grouped.reshape(12, 12)

        with tk.svd_method(svd_method):
            grouped_result = tk.decompositions.TRMSVD(
                grouped,
                layout='grouped',
                out_device=None).fit(
                    rank=3, renormalize=renormalize)
            interleaved_result = tk.decompositions.TRMSVD(
                interleaved,
                out_device=None).fit(
                    rank=3, renormalize=renormalize)
            matrix_result = tk.decompositions.TRMSVD(
                matrix,
                in_dim=in_dim,
                out_dim=out_dim,
                out_device=None).fit(
                    rank=3, renormalize=renormalize)

        for result in [grouped_result, interleaved_result, matrix_result]:
            assert result.in_dim == in_dim
            assert result.out_dim == out_dim
            assert result.rank == grouped_result.rank
            assert torch.allclose(
                result.contract_dense(),
                grouped_result.contract_dense(),
                rtol=1e-10,
                atol=1e-10)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_matches_tr_svd_on_fused_axes(self, svd_method):
        generator = torch.Generator().manual_seed(94)
        tensor = torch.randn(
            2, 3, 2, 2, 2, 3,
            dtype=torch.float64,
            generator=generator)
        fused = tensor.reshape(6, 4, 6)
        kwargs = {
            'rank': 3,
            'rtol': 0.1,
            'renormalize': True,
            'collect_metrics': True,
        }

        with tk.svd_method(svd_method):
            trm_result = tk.decompositions.TRMSVD(
                tensor, out_device=None).fit(**kwargs)
            tr_result = tk.decompositions.TRSVD(
                fused, out_device=None).fit(**kwargs)

        assert trm_result.rank == tr_result.rank
        assert torch.allclose(
            trm_result.contract_dense().reshape(fused.shape),
            tr_result.contract_dense(),
            rtol=1e-12,
            atol=1e-12)
        assert [record.svd_method
                for record in trm_result.metrics.truncations] == [
                    record.svd_method
                    for record in tr_result.metrics.truncations]

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_fidelity_with_exact_ttm_is_one(self, svd_method):
        generator = torch.Generator().manual_seed(95)
        tensor = torch.randn(
            2, 3, 2, 4, 2, 3,
            dtype=torch.float64,
            generator=generator)

        with tk.svd_method(svd_method):
            ttm_result = tk.decompositions.TTMSVD(
                tensor, out_device=None).fit()
            trm_result = tk.decompositions.TRMSVD(
                tensor, out_device=None).fit()

        assert torch.allclose(
            ttm_result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)
        assert torch.allclose(
            trm_result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)
        assert torch.allclose(
            ttm_result.fidelity(trm_result),
            torch.ones((), dtype=torch.float64),
            rtol=1e-10,
            atol=1e-10)

    def test_repeated_fits_have_independent_results(self):
        tensor = torch.randn(2, 3, 2, 4, 2, 3, dtype=torch.float64)
        decomposer = tk.decompositions.TRMSVD(tensor, out_device=None)

        rank_one = decomposer.fit(rank=1)
        rank_three = decomposer.fit(rank=3)

        assert rank_one.rank == [1, 1, 1]
        assert max(rank_three.rank) <= 3
        assert rank_one.metrics is not rank_three.metrics
        assert torch.linalg.vector_norm(
            tensor - rank_three.contract_dense()) <= torch.linalg.vector_norm(
                tensor - rank_one.contract_dense())

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_zero_tensor_is_finite(self, svd_method, dtype):
        tensor = torch.zeros(2, 3, 2, 4, 2, 3, dtype=dtype)

        with tk.svd_method(svd_method):
            result = tk.decompositions.TRMSVD(
                tensor, out_device=None).fit(
                    rank=1,
                    cutoff=0,
                    renormalize=True,
                    collect_metrics=True)

        assert result.rank == [1, 1, 1]
        assert all(torch.isfinite(core).all() for core in result.cores)
        assert torch.equal(result.contract_dense(), tensor)
        assert result.metrics.errors == []
        assert all(record.local_abs_error == 0
                   for record in result.metrics.truncations)

    def test_reuses_one_cyclic_svd_kernel(self, monkeypatch):
        calls = []
        original_fit = tr_module.TRSVD._fit_validated

        def tracked_fit(self, *args, **kwargs):
            calls.append(tuple(self.tensor.shape))
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(tr_module.TRSVD, '_fit_validated', tracked_fit)
        tk.decompositions.TRMSVD(
            torch.randn(2, 3, 4, 5, 2, 3),
            out_device=None).fit(rank=2)

        assert calls == [(6, 20, 6)]

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('device_name', DEVICE_NAMES)
    @pytest.mark.parametrize('renormalize', [False, True])
    def test_out_device_policy(
            self, device_name, svd_method, renormalize):
        device = _device(device_name)
        tensor = torch.randn(2, 3, 2, 4, 2, 3, device=device)

        with tk.svd_method(svd_method):
            cpu_result = tk.decompositions.TRMSVD(tensor).fit(
                rank=2, renormalize=renormalize)
            active_result = tk.decompositions.TRMSVD(
                tensor, out_device=None).fit(
                    rank=2, renormalize=renormalize)

        assert all(core.device.type == 'cpu' for core in cpu_result.cores)
        assert all(core.device == tensor.device for core in active_result.cores)
        assert torch.allclose(
            active_result.contract_dense(),
            cpu_result.contract_dense().to(device),
            rtol=1e-5,
            atol=1e-6)

    def test_fast_path_skips_diagnostics(self, monkeypatch):
        def unexpected_timer(*args, **kwargs):
            pytest.fail('The fast path should not start synchronized timers')

        def unexpected_event(*args, **kwargs):
            pytest.fail('The fast path should not construct observer events')

        monkeypatch.setattr(
            tr_module._RuntimePolicy, 'timer', unexpected_timer)
        monkeypatch.setattr(trm_module, 'DecompositionEvent', unexpected_event)
        result = tk.decompositions.TRMSVD(
            torch.randn(2, 3, 2, 4),
            out_device=None).fit(rank=2)

        assert result.metrics.timings == []
        assert result.metrics.truncations == []

    def test_console_observer_uses_trm_shapes(self, capsys):
        result = tk.decompositions.TRMSVD(
            torch.randn(2, 3, 2, 4, 2, 3),
            out_device=None).fit(rank=2, verbose=3)

        output = capsys.readouterr().out
        assert 'TRM-SVD\n=======' in output
        assert '\nTR-SVD\n' not in output
        assert 'input dim: (2, 2, 2)' in output
        assert 'output dim: (3, 4, 3)' in output
        assert 'Initial bipartition' in output
        assert tuple(result.cores[0].shape) == (2, 2, 2, 3)
        assert 'shape: (2, 2, 2, 3)' in output

    def test_console_events_are_emitted_during_fit(
            self, capsys, monkeypatch):
        calls = 0
        original_fit = tr_module.TTSVD._fit_validated

        def tracked_fit(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                assert 'Initial bipartition' in capsys.readouterr().out
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(tr_module.TTSVD, '_fit_validated', tracked_fit)
        tk.decompositions.TRMSVD(
            torch.randn(2, 2, 2, 2, 2, 2),
            out_device=None).fit(rank=2, verbose=1)

        assert calls == 2

    @pytest.mark.parametrize(
        'constructor, kwargs, error_type, match',
        [
            (([1, 2],), {}, TypeError,
             '`tensor` should be torch.Tensor type'),
            ((torch.ones(2, 3),), {}, ValueError,
             'requires at least two sites'),
            ((torch.ones(2, 3, 4, 5),), {'layout': 1}, TypeError,
             '`layout` should be str type'),
            ((torch.ones(2, 3, 4, 5),), {'layout': 'other'}, ValueError,
             '`layout` should be either'),
            ((torch.ones(2, 3, 4, 5),), {'in_dim': (2, 4)}, ValueError,
             '`in_dim` and `out_dim` should be provided together'),
            ((torch.ones(6, 19),),
             {'in_dim': (2, 3), 'out_dim': (4, 5)}, ValueError,
             'matrix shape should equal'),
            ((torch.ones(2, 3, 4),), {}, ValueError,
             'positive even number of dimensions'),
        ],
    )
    def test_constructor_errors(self,
                                constructor,
                                kwargs,
                                error_type,
                                match):
        with pytest.raises(error_type, match=match):
            tk.decompositions.TRMSVD(*constructor, **kwargs)

    @pytest.mark.parametrize(
        'kwargs, error_type, match',
        [
            ({'center': 0}, ValueError, '`center` should satisfy'),
            ({'rank': True}, TypeError, '`rank` should be int type'),
            ({'atol': float('nan')}, ValueError,
             '`atol` should be a finite non-negative number'),
            ({'renormalize': 1}, TypeError,
             '`renormalize` should be bool type'),
            ({'collect_metrics': 1}, TypeError,
             '`collect_metrics` should be bool type'),
            ({'verbose': 4}, ValueError,
             '`verbose` should be between 0 and 3'),
        ],
    )
    def test_fit_errors(self, kwargs, error_type, match):
        decomposer = tk.decompositions.TRMSVD(
            torch.ones(2, 3, 4, 5))
        with pytest.raises(error_type, match=match):
            decomposer.fit(**kwargs)

    def test_gradcheck_grouped_layout(self):
        generator = torch.Generator().manual_seed(97)
        tensor = torch.randn(
            2, 2, 2, 2, 2, 2,
            dtype=torch.float64,
            generator=generator,
            requires_grad=True)

        def reconstruct(value):
            return tk.decompositions.TRMSVD(
                value,
                layout='grouped',
                out_device=None).fit().contract_dense()

        with tk.svd_method('qr_svd'):
            assert torch.autograd.gradcheck(
                reconstruct,
                (tensor,),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3)


class TestTRMSVDFunction:  # MARK: TestTRMSVDFunction

    def test_function_returns_cores_and_optional_info(self, monkeypatch):
        tensor = torch.randn(2, 3, 4, 5)
        collect_metrics_calls = []
        original_fit = trm_module.TRMSVD.fit

        def tracked_fit(self, *args, **kwargs):
            collect_metrics_calls.append(kwargs['collect_metrics'])
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(trm_module.TRMSVD, 'fit', tracked_fit)
        cores = tk.decompositions.trm_svd(tensor, rank=2)
        info_cores, info = tk.decompositions.trm_svd(
            tensor, rank=2, return_info=True)

        assert collect_metrics_calls == [False, True]
        assert info['topology'] == 'trm'
        assert info['rank'] == [2, 2]
        assert info['in_dim'] == [2, 4]
        assert info['out_dim'] == [3, 5]
        assert info['metadata']['algorithm'] == 'trm_svd'
        assert info['metrics']['errors'] == []
        assert len(info['metrics']['truncations']) == 1
        assert len(cores) == len(info_cores) == 2

    def test_return_info_validation(self):
        with pytest.raises(TypeError, match='`return_info` should be bool type'):
            tk.decompositions.trm_svd(
                torch.ones(2, 3, 4, 5), return_info=1)
