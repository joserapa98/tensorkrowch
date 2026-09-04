"""Tests for TTM-SVD classes and functional interfaces."""

import pytest

import torch
import tensorkrowch as tk

import tensorkrowch.decompositions.svd.ttm as ttm_module


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


def _exact_ttm():
    """Builds a rank-two TTM and its dense contraction."""
    generator = torch.Generator().manual_seed(80)
    cores = [
        torch.randn(2, 2, 3, dtype=torch.float64, generator=generator),
        torch.randn(2, 3, 2, 2, dtype=torch.float64, generator=generator),
        torch.randn(2, 4, 3, dtype=torch.float64, generator=generator),
    ]
    result = tk.decompositions.TTMDecomposition(cores)
    return result, result.contract_dense()


def _near_exact_ttm():
    """Builds a TTM tensor with one controlled small component."""
    fused = torch.zeros(4, 4, 4, dtype=torch.float64)
    fused[0, 0, 0] = 5
    fused[1, 1, 1] = 1
    fused[2, 2, 2] = 1e-4
    return fused.reshape(2, 2, 2, 2, 2, 2)


class TestTTMSVD:  # MARK: TestTTMSVD

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_recovers_exact_ttm(self, svd_method):
        expected, tensor = _exact_ttm()

        with tk.svd_method(svd_method):
            result = tk.decompositions.TTMSVD(
                tensor, out_device=None).fit(rank=2)

        assert result.rank == expected.rank
        assert result.in_dim == expected.in_dim
        assert result.out_dim == expected.out_dim
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_low_rank_error_has_gaussian_noise_scale(self, svd_method):
        _, tensor = _exact_ttm()
        generator = torch.Generator().manual_seed(81)
        noise = 1e-4 * torch.randn(
            tensor.shape, dtype=tensor.dtype, generator=generator)
        noisy_tensor = tensor + noise

        with tk.svd_method(svd_method):
            approximation = tk.decompositions.TTMSVD(
                noisy_tensor, out_device=None).fit(
                    rank=2).contract_dense()

        noise_norm = torch.linalg.vector_norm(noise)
        residual_norm = torch.linalg.vector_norm(noisy_tensor - approximation)
        clean_error = torch.linalg.vector_norm(tensor - approximation)
        assert residual_norm <= 1.01 * noise_norm
        assert residual_norm >= 0.25 * noise_norm
        assert clean_error <= 1.5 * noise_norm

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('criterion', ['atol', 'rtol'])
    def test_truncation_criteria_bound_global_error(
            self, svd_method, criterion):
        tensor = _near_exact_ttm()
        small_value = tensor[1, 0, 1, 0, 1, 0].item()
        tensor_norm = torch.linalg.vector_norm(tensor).item()
        n_cuts = tensor.ndim // 2 - 1

        if criterion == 'atol':
            tolerance = 1.01 * small_value ** 2
            kwargs = {'atol': tolerance}
            bound = (n_cuts * tolerance) ** 0.5
        else:
            tolerance = 1.01 * small_value ** 2 / tensor_norm ** 2
            kwargs = {'rtol': tolerance}
            bound = tensor_norm * (n_cuts * tolerance) ** 0.5

        with tk.svd_method(svd_method):
            result = tk.decompositions.TTMSVD(
                tensor, out_device=None).fit(**kwargs)

        error = torch.linalg.vector_norm(
            tensor - result.contract_dense()).item()
        assert error > 0
        assert error <= bound * (1 + 1e-10)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('renormalize', [False, True])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_interleaved_and_grouped_layouts_are_equivalent(
            self, svd_method, renormalize, dtype):
        generator = torch.Generator().manual_seed(0)
        in_dim = (2, 3, 2)
        out_dim = (3, 2, 2)
        grouped = torch.randn(
            *in_dim, *out_dim,
            dtype=dtype,
            generator=generator) * 1e2
        interleaved = _interleave(grouped, len(in_dim))

        with tk.svd_method(svd_method):
            grouped_result = tk.decompositions.TTMSVD(
                grouped,
                layout='grouped',
                out_device=None).fit(
                    renormalize=renormalize,
                    collect_metrics=True)
            interleaved_result = tk.decompositions.TTMSVD(
                interleaved,
                layout='interleaved',
                out_device=None).fit(
                    renormalize=renormalize,
                    collect_metrics=True)

        assert grouped_result.in_dim == in_dim
        assert grouped_result.out_dim == out_dim
        assert grouped_result.rank == interleaved_result.rank
        assert torch.allclose(
            grouped_result.contract_dense(),
            interleaved,
            rtol=1e-10,
            atol=1e-10)
        assert torch.allclose(
            interleaved_result.contract_dense(),
            interleaved,
            rtol=1e-10,
            atol=1e-10)
        assert grouped_result.metrics.errors[0].absolute == pytest.approx(
            interleaved_result.metrics.errors[0].absolute,
            rel=1e-12,
            abs=1e-12)

    def test_matrix_route_with_heterogeneous_dimensions(self):
        generator = torch.Generator().manual_seed(1)
        in_dim = (2, 2, 3)
        out_dim = (3, 2, 2)
        matrix = torch.randn(
            12, 12, dtype=torch.float64, generator=generator)
        expected = _interleave(
            matrix.reshape(*in_dim, *out_dim), len(in_dim))

        result = tk.decompositions.TTMSVD(
            matrix,
            in_dim=in_dim,
            out_dim=out_dim,
            out_device=None).fit()

        assert result.in_dim == in_dim
        assert result.out_dim == out_dim
        assert result.metadata['matrix_input'] is True
        assert torch.allclose(
            result.contract_dense(), expected, rtol=1e-10, atol=1e-12)

    def test_explicit_dimensions_validate_tensorized_layout(self):
        in_dim = (2, 3)
        out_dim = (4, 5)
        grouped = torch.randn(
            *in_dim, *out_dim, dtype=torch.float64)
        interleaved = _interleave(grouped, len(in_dim))

        grouped_result = tk.decompositions.TTMSVD(
            grouped,
            in_dim=in_dim,
            out_dim=out_dim,
            layout='grouped').fit()
        interleaved_result = tk.decompositions.TTMSVD(
            interleaved,
            in_dim=in_dim,
            out_dim=out_dim).fit()

        assert grouped_result.in_dim == interleaved_result.in_dim
        assert grouped_result.out_dim == interleaved_result.out_dim
        assert torch.allclose(
            grouped_result.contract_dense(), interleaved,
            rtol=1e-10, atol=1e-12)
        assert torch.allclose(
            interleaved_result.contract_dense(), interleaved,
            rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('renormalize', [False, True])
    def test_one_site_complex_tensor(self, svd_method, renormalize):
        tensor = torch.randn(3, 4, dtype=torch.complex128)

        with tk.svd_method(svd_method):
            result = tk.decompositions.TTMSVD(
                tensor, out_device=None).fit(
                    rank=1,
                    renormalize=renormalize,
                    collect_metrics=True)

        assert result.rank == []
        assert result.in_dim == (3,)
        assert result.out_dim == (4,)
        assert torch.allclose(result.cores[0], tensor)
        assert result.metrics.truncations == []
        assert result.metrics.errors[0].absolute == 0

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('renormalize', [False, True])
    def test_matches_tt_svd_on_fused_axes(self, svd_method, renormalize):
        generator = torch.Generator().manual_seed(2)
        tensor = torch.randn(
            2, 3, 2, 2, 2, 3,
            dtype=torch.float64,
            generator=generator)
        fused = tensor.reshape(6, 4, 6)
        kwargs = {
            'rank': 3,
            'rtol': 0.15,
            'cum_percentage': 0.95,
            'renormalize': renormalize,
            'collect_metrics': True,
        }

        with tk.svd_method(svd_method):
            ttm_result = tk.decompositions.TTMSVD(
                tensor, out_device=None).fit(**kwargs)
            tt_result = tk.decompositions.TTSVD(
                fused, out_device=None).fit(**kwargs)

        assert ttm_result.rank == tt_result.rank
        assert torch.allclose(
            ttm_result.contract_dense().reshape(fused.shape),
            tt_result.contract_dense(),
            rtol=1e-12,
            atol=1e-12)
        ttm_error = ttm_result.metrics.errors[0]
        tt_error = tt_result.metrics.errors[0]
        assert ttm_error.absolute == pytest.approx(
            tt_error.absolute, rel=1e-12, abs=1e-12)
        assert ttm_error.relative == pytest.approx(
            tt_error.relative, rel=1e-12, abs=1e-12)
        assert [record.svd_method
                for record in ttm_result.metrics.truncations] == [
                    svd_method, svd_method]

    def test_repeated_fits_have_independent_results(self):
        tensor = torch.randn(2, 3, 4, 5, dtype=torch.float64)
        decomposer = tk.decompositions.TTMSVD(
            tensor, out_device=None)

        rank_one = decomposer.fit(rank=1)
        rank_four = decomposer.fit(rank=4)

        assert rank_one.rank == [1]
        assert rank_four.rank == [4]
        assert rank_one.metrics is not rank_four.metrics
        assert rank_one.metrics.errors == []
        assert rank_four.metrics.truncations == []
        assert torch.linalg.vector_norm(
            tensor - rank_four.contract_dense()) <= torch.linalg.vector_norm(
                tensor - rank_one.contract_dense())

    @pytest.mark.parametrize('device_name', DEVICE_NAMES)
    def test_out_device_policy(self, device_name):
        device = _device(device_name)
        tensor = torch.randn(2, 3, 4, 5, device=device)

        cpu_result = tk.decompositions.TTMSVD(tensor).fit(rank=2)
        active_result = tk.decompositions.TTMSVD(
            tensor, out_device=None).fit(rank=2)

        assert all(core.device.type == 'cpu' for core in cpu_result.cores)
        assert all(core.device == device for core in active_result.cores)
        assert torch.allclose(
            active_result.contract_dense(),
            cpu_result.contract_dense().to(device),
            rtol=1e-5,
            atol=1e-6)

    def test_console_observer_uses_ttm_shapes(self, capsys):
        result = tk.decompositions.TTMSVD(
            torch.randn(2, 3, 4, 5), out_device=None).fit(
                rank=2,
                verbose=3)

        output = capsys.readouterr().out
        assert 'TTM-SVD\n=======' in output
        assert 'input dim: (2, 4)' in output
        assert 'output dim: (3, 5)' in output
        assert tuple(result.cores[0].shape) == (2, 2, 3)
        assert 'shape: (2, 2, 3)' in output
        assert 'shape: (2, 4, 5)' in output

    @pytest.mark.parametrize(
        'constructor, kwargs, error_type, match',
        [
            (([1, 2],), {}, TypeError,
             '`tensor` should be torch.Tensor type'),
            ((torch.ones(2, 3),), {'layout': 1}, TypeError,
             '`layout` should be str type'),
            ((torch.ones(2, 3),), {'layout': 'other'}, ValueError,
             '`layout` should be either'),
            ((torch.ones(2, 3),), {'in_dim': (2,)}, ValueError,
             '`in_dim` and `out_dim` should be provided together'),
            ((torch.ones(2, 3),),
             {'in_dim': object(), 'out_dim': (3,)}, TypeError,
             '`in_dim` should be int or a sequence of ints'),
            ((torch.ones(2, 3),),
             {'in_dim': (2.0,), 'out_dim': (3,)}, TypeError,
             '`in_dim` should contain only ints'),
            ((torch.ones(2, 3),),
             {'in_dim': (0,), 'out_dim': (3,)}, ValueError,
             '`in_dim` should contain only positive dimensions'),
            ((torch.ones(2, 3),),
             {'in_dim': (2, 1), 'out_dim': (3,)}, ValueError,
             '`in_dim` and `out_dim` should have the same length'),
            ((torch.ones(4, 5),),
             {'in_dim': (2, 2), 'out_dim': (3, 2)}, ValueError,
             'The matrix shape should equal'),
            ((torch.ones(2, 3, 4),), {}, ValueError,
             'positive even number of dimensions'),
            ((torch.ones(2, 3, 4, 5),),
             {'in_dim': (2, 3), 'out_dim': (4, 5)}, ValueError,
             'tensor shape is incompatible'),
        ],
    )
    def test_constructor_errors(self,
                                constructor,
                                kwargs,
                                error_type,
                                match):
        with pytest.raises(error_type, match=match):
            tk.decompositions.TTMSVD(*constructor, **kwargs)

    @pytest.mark.parametrize(
        'kwargs, error_type, match',
        [
            ({'rank': 0}, ValueError, '`rank` should be a positive integer'),
            ({'rtol': 2}, ValueError, '`rtol` should be a number between'),
            ({'renormalize': 1}, TypeError,
             '`renormalize` should be bool type'),
            ({'collect_metrics': 1}, TypeError,
             '`collect_metrics` should be bool type'),
            ({'verbose': 4}, ValueError,
             '`verbose` should be between 0 and 3'),
        ],
    )
    def test_fit_errors(self, kwargs, error_type, match):
        decomposer = tk.decompositions.TTMSVD(torch.ones(2, 3, 4, 5))
        with pytest.raises(error_type, match=match):
            decomposer.fit(**kwargs)

    def test_gradcheck_grouped_layout(self):
        tensor = torch.randn(
            2, 2, 3, 2,
            dtype=torch.float64,
            requires_grad=True)

        def reconstruct(value):
            return tk.decompositions.TTMSVD(
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


class TestTTMSVDFunction:  # MARK: TestTTMSVDFunction

    def test_function_returns_cores_and_optional_info(self, monkeypatch):
        tensor = torch.randn(2, 3, 4, 5)
        collect_metrics_calls = []
        original_fit = ttm_module.TTMSVD.fit

        def tracked_fit(self, *args, **kwargs):
            collect_metrics_calls.append(kwargs['collect_metrics'])
            return original_fit(self, *args, **kwargs)

        monkeypatch.setattr(ttm_module.TTMSVD, 'fit', tracked_fit)
        cores = tk.decompositions.ttm_svd(tensor, rank=2)
        info_cores, info = tk.decompositions.ttm_svd(
            tensor, rank=2, return_info=True)

        assert collect_metrics_calls == [False, True]
        assert info['topology'] == 'ttm'
        assert info['rank'] == [2]
        assert info['in_dim'] == [2, 4]
        assert info['out_dim'] == [3, 5]
        assert info['metadata']['algorithm'] == 'ttm_svd'
        assert len(info['metrics']['truncations']) == 1
        assert len(cores) == len(info_cores) == 2

    def test_return_info_validation(self):
        with pytest.raises(TypeError, match='`return_info` should be bool type'):
            tk.decompositions.ttm_svd(
                torch.ones(2, 3), return_info=1)
