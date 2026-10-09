"""Tests for the TT-SVD class and functional interfaces."""

import math

import pytest
import torch

import tensorkrowch as tk

import tensorkrowch.decompositions.svd.tt as tt_module
from tensorkrowch.decompositions.svd.utils import (_log_tensor_norm,
                                                   _normalize_tensor)


SVD_METHODS = ['svd', 'qr_svd']
DEVICE_NAMES = ['cpu', 'cuda', 'mps']


def _device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    if name == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is not available')
    return torch.device(name)


def _exact_tt():
    """Builds a rank-two TT and its dense contraction."""
    generator = torch.Generator().manual_seed(70)
    cores = [
        torch.randn(3, 2, dtype=torch.float64, generator=generator),
        torch.randn(2, 4, 2, dtype=torch.float64, generator=generator),
        torch.randn(2, 5, dtype=torch.float64, generator=generator),
    ]
    result = tk.decompositions.TTDecomposition(cores)
    return result, result.contract_dense()


def _near_exact_tt():
    """Builds a TT tensor with one controlled small component."""
    tensor = torch.zeros(3, 3, 3, dtype=torch.float64)
    tensor[0, 0, 0] = 5
    tensor[1, 1, 1] = 1
    tensor[2, 2, 2] = 1e-4
    return tensor


class TestTTSVD:  # MARK: TestTTSVD

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_stable_norm_helpers_share_components(self, dtype):
        generator = torch.Generator().manual_seed(69)
        tensor = torch.randn(
            3, 4, dtype=dtype, generator=generator) * 1e200
        tensor[0] = 0

        normalized, normalization_log = _normalize_tensor(tensor, dim=-1)
        norm_log = _log_tensor_norm(tensor, dim=-1)

        assert torch.equal(normalization_log, norm_log)
        assert torch.equal(normalized[0], torch.zeros_like(normalized[0]))
        assert torch.isneginf(norm_log[0])
        assert torch.allclose(
            normalized[1:].norm(dim=-1),
            torch.ones(2, dtype=tensor.real.dtype),
            rtol=1e-12,
            atol=1e-12)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_recovers_exact_tt(self, svd_method):
        expected, tensor = _exact_tt()

        with tk.svd_method(svd_method):
            result = tk.decompositions.TTSVD(
                tensor, out_device=None).fit(rank=2)

        assert result.rank == expected.rank
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('renormalize', [False, True])
    @pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
    @pytest.mark.parametrize('n_batches', [0, 1, 2])
    def test_exact_dense_oracle_with_batches(
            self, svd_method, renormalize, dtype, n_batches):
        generator = torch.Generator().manual_seed(72)
        batch_shape = (2, 3)[:n_batches]
        in_dim = (2, 3, 4)
        tensor = torch.randn(
            *batch_shape,
            *in_dim,
            dtype=dtype,
            generator=generator) * 1e-2

        with tk.svd_method(svd_method):
            result = tk.decompositions.TTSVD(
                tensor,
                n_batches=n_batches,
                out_device=None).fit(renormalize=renormalize)

        assert result.batch_shape == batch_shape
        assert result.in_dim == in_dim
        assert result.dtype == dtype
        assert torch.allclose(
            result.contract_dense(), tensor, rtol=5e-5, atol=1e-7)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_low_rank_error_has_gaussian_noise_scale(self, svd_method):
        _, tensor = _exact_tt()
        generator = torch.Generator().manual_seed(71)
        noise = 1e-4 * torch.randn(
            tensor.shape, dtype=tensor.dtype, generator=generator)
        noisy_tensor = tensor + noise

        with tk.svd_method(svd_method):
            approximation = tk.decompositions.TTSVD(
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
        tensor = _near_exact_tt()
        small_value = tensor[-1, -1, -1].item()
        tensor_norm = torch.linalg.vector_norm(tensor).item()
        n_cuts = tensor.ndim - 1

        if criterion == 'atol':
            tolerance = 1.01 * small_value ** 2
            kwargs = {'atol': tolerance}
            bound = math.sqrt(n_cuts * tolerance)
        else:
            tolerance = 1.01 * small_value ** 2 / tensor_norm ** 2
            kwargs = {'rtol': tolerance}
            bound = tensor_norm * math.sqrt(n_cuts * tolerance)

        with tk.svd_method(svd_method):
            result = tk.decompositions.TTSVD(
                tensor, out_device=None).fit(**kwargs)

        error = torch.linalg.vector_norm(
            tensor - result.contract_dense()).item()
        assert error > 0
        assert error <= bound * (1 + 1e-10)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('renormalize', [False, True])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_fit_reconstructs_and_records_metrics(self,
                                                  svd_method,
                                                  renormalize,
                                                  dtype):
        generator = torch.Generator().manual_seed(0)
        tensor = torch.randn(
            2, 3, 4, dtype=dtype, generator=generator) * 1e3
        decomposer = tk.decompositions.TTSVD(
            tensor, out_device=None)

        with tk.svd_method(svd_method):
            result = decomposer.fit(
                rank=2,
                renormalize=renormalize,
                collect_metrics=True)

        approximation = result.contract_dense()
        error = torch.linalg.vector_norm(tensor - approximation)
        relative = error / torch.linalg.vector_norm(tensor)

        assert isinstance(result, tk.decompositions.TTDecomposition)
        assert result.rank == [2, 2]
        assert result.metadata == {
            'algorithm': 'tt_svd',
            'renormalize': renormalize,
        }
        assert len(result.metrics.truncations) == 2
        assert all(record.svd_method == svd_method
                   for record in result.metrics.truncations)
        assert result.metrics.errors[0].kind == 'truncation'
        assert result.metrics.errors[0].absolute == pytest.approx(
            error.item(), rel=1e-10, abs=1e-10)
        assert result.metrics.errors[0].relative == pytest.approx(
            relative.item(), rel=1e-10, abs=1e-10)
        assert len(result.metrics.timings) == 1
        assert len(result.metrics.timings[0].children) == 2

    @pytest.mark.parametrize('renormalize', [False, True])
    def test_batched_errors_are_stored_per_batch(self, renormalize):
        generator = torch.Generator().manual_seed(1)
        tensor = torch.randn(
            3, 2, 3, 4, dtype=torch.float64, generator=generator)
        tensor[0] = 0

        result = tk.decompositions.TTSVD(
            tensor, n_batches=1, out_device=None).fit(
                rank=2,
                renormalize=renormalize,
                collect_metrics=True)

        difference = (tensor - result.contract_dense()).flatten(1)
        abs_error = difference.norm(dim=-1)
        input_norm = tensor.flatten(1).norm(dim=-1)
        rel_error = torch.where(
            input_norm > 0,
            abs_error / input_norm,
            torch.zeros_like(abs_error))
        record = result.metrics.errors[0]

        assert torch.allclose(record.absolute,
                              abs_error,
                              rtol=1e-10,
                              atol=1e-10)
        assert torch.allclose(record.relative,
                              rel_error,
                              rtol=1e-10,
                              atol=1e-10)
        assert torch.allclose(record.denominator, input_norm)
        assert all(torch.isfinite(core).all() for core in result.cores)

    def test_repeated_fits_have_independent_state(self):
        tensor = torch.randn(3, 4, 5, dtype=torch.float64)
        decomposer = tk.decompositions.TTSVD(
            tensor, out_device=None)

        rank_one = decomposer.fit(rank=1)
        rank_three = decomposer.fit(rank=3)

        assert rank_one.rank == [1, 1]
        assert rank_three.rank == [3, 3]
        assert rank_one.metrics is not rank_three.metrics
        assert rank_one.cores[0] is not rank_three.cores[0]
        assert torch.linalg.vector_norm(
            tensor - rank_three.contract_dense()) <= torch.linalg.vector_norm(
                tensor - rank_one.contract_dense())

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('renormalize', [False, True])
    def test_fast_path_matches_instrumented_fit(self,
                                                monkeypatch,
                                                svd_method,
                                                renormalize):
        tensor = torch.randn(2, 3, 4, dtype=torch.float64)
        decomposer = tk.decompositions.TTSVD(
            tensor, out_device=None)
        return_info_calls = []
        original_truncated_svd = tt_module.truncated_svd

        def tracked_truncated_svd(*args, **kwargs):
            return_info_calls.append(kwargs['return_info'])
            return original_truncated_svd(*args, **kwargs)

        monkeypatch.setattr(
            tt_module, 'truncated_svd', tracked_truncated_svd)
        with tk.svd_method(svd_method):
            fast = decomposer.fit(
                rank=2,
                renormalize=renormalize)
            instrumented = decomposer.fit(
                rank=2,
                renormalize=renormalize,
                collect_metrics=True)

        assert return_info_calls == [False, False, True, True]
        assert fast.metrics.as_info() == {
            'errors': [],
            'truncations': [],
            'timings': [],
            'fidelities': [],
            'warnings': [],
        }
        assert len(instrumented.metrics.errors) == 1
        assert len(instrumented.metrics.truncations) == 2
        assert len(instrumented.metrics.timings) == 1
        for fast_core, instrumented_core in zip(
                fast.cores, instrumented.cores):
            assert torch.allclose(
                fast_core,
                instrumented_core,
                rtol=1e-12,
                atol=1e-12)

    @pytest.mark.parametrize('renormalize', [False, True])
    def test_fast_path_skips_diagnostic_input_norm(self,
                                                   monkeypatch,
                                                   renormalize):
        def unexpected_log_norm(*args, **kwargs):
            pytest.fail('The fast path should not compute diagnostic norms')

        def unexpected_timer(*args, **kwargs):
            pytest.fail('The fast path should not start synchronized timers')

        def unexpected_event(*args, **kwargs):
            pytest.fail('The fast path should not construct observer events')

        monkeypatch.setattr(tt_module, '_log_tensor_norm', unexpected_log_norm)
        monkeypatch.setattr(
            tt_module._RuntimePolicy, 'timer', unexpected_timer)
        monkeypatch.setattr(tt_module, 'DecompositionEvent', unexpected_event)
        result = tk.decompositions.TTSVD(
            torch.randn(2, 3, 4), out_device=None).fit(
                rank=2,
                renormalize=renormalize)

        assert result.rank == [2, 2]
        assert result.metrics.errors == []
        assert result.metrics.truncations == []
        assert result.metrics.timings == []

    @pytest.mark.parametrize('scale', [1e-200, 1e200])
    def test_log_error_accumulation_is_scale_stable(self, scale):
        generator = torch.Generator().manual_seed(4)
        tensor = torch.randn(
            2, 3, 4, dtype=torch.float64, generator=generator)
        reference = tk.decompositions.TTSVD(
            tensor, out_device=None).fit(
                rank=1,
                renormalize=True,
                collect_metrics=True)
        scaled = tk.decompositions.TTSVD(
            tensor * scale, out_device=None).fit(
                rank=1,
                renormalize=True,
                collect_metrics=True)

        error = scaled.metrics.errors[0]
        assert error.absolute == pytest.approx(
            reference.metrics.errors[0].absolute * scale,
            rel=1e-12)
        assert error.relative == pytest.approx(
            reference.metrics.errors[0].relative,
            rel=1e-12)
        assert math.isfinite(error.absolute)
        assert math.isfinite(error.relative)
        assert all(torch.isfinite(record.local_abs_error).all()
                   for record in scaled.metrics.truncations)
        assert all(torch.isfinite(core).all() for core in scaled.cores)

    def test_log_metrics_match_direct_moderate_scale_calculations(self):
        generator = torch.Generator().manual_seed(5)
        tensor = 3 * torch.randn(
            3, 4, 5, dtype=torch.float64, generator=generator)
        result = tk.decompositions.TTSVD(
            tensor, out_device=None).fit(
                rank=2,
                renormalize=True,
                collect_metrics=True)

        residual = tensor
        previous_rank = 1
        local_errors = []
        for site, in_dim in enumerate(tensor.shape[:-1]):
            residual = residual.reshape(
                previous_rank * in_dim, -1)
            _, singular_values, vh = torch.linalg.svd(
                residual, full_matrices=False)
            local_errors.append(singular_values[2:].square().sum().sqrt())
            residual = singular_values[:2].unsqueeze(-1) * vh[:2]
            previous_rank = 2

        direct_absolute = torch.stack(local_errors).square().sum().sqrt()
        direct_relative = direct_absolute / torch.linalg.vector_norm(tensor)
        log_input_norm = _log_tensor_norm(tensor)

        assert log_input_norm.exp() == pytest.approx(
            torch.linalg.vector_norm(tensor).item(), rel=1e-12)
        assert torch.allclose(
            torch.stack([record.local_abs_error
                         for record in result.metrics.truncations]),
            torch.stack(local_errors),
            rtol=1e-12)
        assert result.metrics.errors[0].absolute == pytest.approx(
            direct_absolute.item(), rel=1e-12)
        assert result.metrics.errors[0].relative == pytest.approx(
            direct_relative.item(), rel=1e-12)

    @pytest.mark.parametrize('n_batches', [0, 1])
    @pytest.mark.parametrize(
        'kwargs',
        [
            {'cutoff': 1.0},
            {'atol': 1.05},
            {'cutoff': 1.0, 'atol': 1.05},
        ],
    )
    def test_absolute_criteria_keep_original_scale_when_renormalized(
            self, n_batches, kwargs):
        tensor = torch.diag(
            torch.tensor([5.0, 3.0, 1.0, 0.1], dtype=torch.float64))
        if n_batches:
            tensor = torch.stack([tensor, tensor])

        result = tk.decompositions.TTSVD(
            tensor,
            n_batches=n_batches,
            out_device=None).fit(
                renormalize=True,
                **kwargs)

        assert result.rank == [2]

    def test_one_site_and_zero_tensor(self):
        tensor = torch.zeros(5)
        result = tk.decompositions.TTSVD(
            tensor, out_device=None).fit(
                rank=1,
                renormalize=True,
                collect_metrics=True)

        assert result.rank == []
        assert len(result.cores) == 1
        assert torch.equal(result.cores[0], tensor)
        assert result.metrics.truncations == []
        assert result.metrics.errors[0].absolute == 0
        assert result.metrics.errors[0].relative == 0
        assert result.metrics.timings[0].children == ()

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    @pytest.mark.parametrize('device_name', DEVICE_NAMES)
    @pytest.mark.parametrize('renormalize', [False, True])
    def test_out_device_policy(
            self, device_name, svd_method, renormalize):
        device = _device(device_name)
        tensor = torch.randn(2, 3, 4, device=device)

        with tk.svd_method(svd_method):
            cpu_result = tk.decompositions.TTSVD(tensor).fit(
                rank=2, renormalize=renormalize)
            active_result = tk.decompositions.TTSVD(
                tensor, out_device=None).fit(
                    rank=2, renormalize=renormalize)

        assert all(core.device.type == 'cpu' for core in cpu_result.cores)
        assert all(
            core.device == tensor.device for core in active_result.cores)
        assert torch.allclose(
            active_result.contract_dense(),
            cpu_result.contract_dense().to(device),
            rtol=1e-5,
            atol=1e-6)

    def test_console_observer(self, capsys):
        tensor = torch.randn(2, 3, 4)

        result = tk.decompositions.TTSVD(
            tensor, out_device=None).fit(
                rank=2,
                verbose=2)

        output = capsys.readouterr().out
        assert 'TT-SVD\n======' in output
        assert 'Cut 1-2' in output
        assert 'Cut 2-3' in output
        assert 'selected rank: 2' in output
        assert 'absolute error: 0.00e+00' in output
        assert f'rank: {tuple(result.rank)}' in output
        assert any(
            line.startswith('  elapsed: ') and
            line.endswith(' s') and
            'e' in line
            for line in output.splitlines())
        assert 'Summary\n-------' in output
        assert len(result.metrics.errors) == 1
        assert len(result.metrics.truncations) == 2
        assert len(result.metrics.timings) == 1

    def test_console_observer_aggregates_batch_metrics(self, capsys):
        tensor = torch.arange(48.).reshape(2, 2, 3, 4)

        tk.decompositions.TTSVD(
            tensor, n_batches=1, out_device=None).fit(
                rank=2,
                verbose=2)

        output = capsys.readouterr().out
        assert 'absolute error: tensor(' not in output
        assert 'relative error: tensor(' not in output

    def test_console_events_are_emitted_during_fit(
            self, capsys, monkeypatch):
        truncated_svd = tt_module.truncated_svd
        calls = 0

        def tracked_truncated_svd(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                output = capsys.readouterr().out
                assert 'sites: 3' in output
                assert 'input dim: (2, 3, 4)' in output
            elif calls == 2:
                assert 'Cut 1-2' in capsys.readouterr().out
            return truncated_svd(*args, **kwargs)

        monkeypatch.setattr(
            tt_module, 'truncated_svd', tracked_truncated_svd)
        tk.decompositions.TTSVD(torch.randn(2, 3, 4)).fit(
            rank=2,
            verbose=1)

        assert calls == 2

    @pytest.mark.parametrize(
        'constructor, error_type, match',
        [
            (([1, 2],), TypeError, '`tensor` should be torch.Tensor type'),
            ((torch.ones(2), True), TypeError,
             '`n_batches` should be int type'),
            ((torch.ones(2), 1), ValueError, 'leave at least one'),
            ((torch.ones(2), -1), ValueError, 'leave at least one'),
            ((torch.empty(2, 0),), ValueError,
             'TT input and batch dimensions should be positive'),
        ],
    )
    def test_constructor_errors(self, constructor, error_type, match):
        with pytest.raises(error_type, match=match):
            tk.decompositions.TTSVD(*constructor)

    @pytest.mark.parametrize(
        'kwargs, error_type, match',
        [
            ({'rank': 0}, ValueError, '`rank` should be a positive integer'),
            ({'rank': True}, TypeError, '`rank` should be int type'),
            ({'cutoff': True}, TypeError,
             '`cutoff` should be a real number'),
            ({'atol': float('nan')}, ValueError,
             '`atol` should be a finite non-negative number'),
            ({'rtol': float('inf')}, ValueError,
             '`rtol` should be a finite number between'),
            ({'rtol': 2}, ValueError,
             '`rtol` should be a finite number between'),
            ({'renormalize': 1}, TypeError,
             '`renormalize` should be bool type'),
            ({'collect_metrics': 1}, TypeError,
             '`collect_metrics` should be bool type'),
            ({'verbose': -1}, ValueError,
             '`verbose` should be between 0 and 3'),
            ({'verbose': 4}, ValueError,
             '`verbose` should be between 0 and 3'),
        ],
    )
    def test_fit_errors(self, kwargs, error_type, match):
        decomposer = tk.decompositions.TTSVD(torch.ones(2, 3))
        with pytest.raises(error_type, match=match):
            decomposer.fit(**kwargs)

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_gradcheck(self, svd_method):
        tensor = torch.randn(
            2, 3, 4, dtype=torch.float64, requires_grad=True)

        def reconstruct(value):
            return tk.decompositions.TTSVD(
                value, out_device=None).fit().contract_dense()

        with tk.svd_method(svd_method):
            assert torch.autograd.gradcheck(
                reconstruct,
                (tensor,),
                eps=1e-6,
                atol=1e-4,
                rtol=1e-3)


class TestTTSVDFunction:  # MARK: TestTTSVDFunction

    @pytest.mark.parametrize('svd_method', SVD_METHODS)
    def test_function_and_legacy_wrapper_are_equivalent(self, svd_method):
        tensor = torch.randn(2, 3, 4, dtype=torch.float64)

        with tk.svd_method(svd_method):
            result = tk.decompositions.tt_svd(
                tensor,
                rank=2,
                out_device=None,
                collect_metrics=True)
            cores, info = result.cores, result.as_info()
            with pytest.warns(
                    FutureWarning, match='`vec_to_mps` is deprecated'):
                legacy_cores = tk.decompositions.vec_to_mps(tensor, rank=2)

        assert info['rank'] == [2, 2]
        assert info['metadata']['algorithm'] == 'tt_svd'
        for core, legacy_core in zip(cores, legacy_cores):
            assert torch.allclose(core, legacy_core)

    def test_collect_metrics_validation(self):
        with pytest.raises(TypeError, match='`collect_metrics` should be bool type'):
            tk.decompositions.tt_svd(
                torch.ones(2, 3), collect_metrics=1)

    def test_return_info_selects_diagnostics_path(self, monkeypatch):
        tensor = torch.randn(2, 3, 4)
        return_info_calls = []
        original_truncated_svd = tt_module.truncated_svd

        def tracked_truncated_svd(*args, **kwargs):
            return_info_calls.append(kwargs['return_info'])
            return original_truncated_svd(*args, **kwargs)

        monkeypatch.setattr(
            tt_module, 'truncated_svd', tracked_truncated_svd)
        tk.decompositions.tt_svd(tensor, rank=2)
        result = tk.decompositions.tt_svd(
            tensor,
            rank=2,
            collect_metrics=True)
        _, info = result.cores, result.as_info()

        assert return_info_calls == [False, False, True, True]
        assert len(info['metrics']['errors']) == 1
        assert len(info['metrics']['truncations']) == 2
        assert len(info['metrics']['timings']) == 1








@pytest.mark.parametrize('renormalize', [False, True])
@pytest.mark.parametrize('backend', ['svd', 'qr_svd'])
@pytest.mark.parametrize('refine', [False, True])
def test_svd_formats_across_devices(renormalize,
                                    backend,
                                    refine,
                                    device_dtype,
                                    assert_close):
    device, dtype = device_dtype
    generator = torch.Generator().manual_seed(47)
    data = torch.randn(2, 3, 2, dtype=dtype, generator=generator).to(device)
    engine_type = tk.decompositions.TTSVD
    engine = engine_type(data, out_device=None)
    if device == 'mps' and dtype.is_complex and backend == 'qr_svd':
        with tk.svd_method(backend, refine=refine), pytest.raises(RuntimeError, match='geqrf.*float32'):
            engine.fit(rank=6, renormalize=renormalize)
        return
    with tk.svd_method(backend, refine=refine):
        result = engine.fit(rank=6, renormalize=renormalize, collect_metrics=True)
        wrapped = tk.decompositions.tt_svd(data, rank=6, renormalize=renormalize,
                                                    out_device=None)
    assert isinstance(result, tk.formats.TT)
    assert result.dtype == dtype and result.device.type == device
    assert_close(result.contract_dense(), data)
    assert_close(wrapped.contract_dense(), data)
    for record in result.metrics.truncations:
        assert record.local_abs_error.device.type == 'cpu'
        assert not record.local_abs_error.requires_grad
    for plain in (result.clone(), result.detach(), result.to('cpu')):
        assert type(plain) is type(result)
        assert plain.metrics is result.metrics
    algebra = result + result
    assert not isinstance(algebra, tk.decompositions.TensorDecomposition)
    assert_close(algebra.contract_dense(), 2 * data)


@pytest.mark.parametrize('renormalize', [False, True])
@pytest.mark.parametrize('criteria, rank', [
    ({'rank': 2}, 2), ({'cutoff': 0.5}, 3), ({'atol': 0.02}, 3),
    ({'rtol': 0.001}, 3), ({'cum_percentage': 0.999}, 3),
    ({'rank': 2, 'atol': 0.02}, 2),
])
def test_svd_truncation_known_spectrum_and_error_bound(criteria, rank,
                                                       renormalize, device_dtype,
                                                       assert_close):
    device, dtype = device_dtype
    spectrum = torch.tensor([4., 2., 1., 0.1], device=device)
    data = torch.diag(spectrum).to(dtype)
    if dtype.is_complex:
        data = data * 1j
    result = tk.decompositions.TTSVD(data, out_device=None).fit(
        **criteria, renormalize=renormalize, collect_metrics=True)
    assert result.rank == [rank]
    expected = data.clone()
    expected[rank:, rank:] = 0
    assert_close(result.contract_dense(), expected)
    achieved = (result.contract_dense() - data).norm()
    bound = result.metrics.errors[0].absolute.to(device=device,
                                               dtype=achieved.dtype)
    assert_close(bound, achieved)


@pytest.mark.parametrize('large', [False, True])
def test_renormalized_svd_preserves_extreme_input_scale(large, device_dtype,
                                                      assert_close):
    device, dtype = device_dtype
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    scale = 1e20 if real_dtype == torch.float32 else 1e200
    if not large:
        scale = 1 / scale
    reference = torch.arange(1., 9., device=device).reshape(2, 2, 2).to(dtype)
    if dtype.is_complex:
        reference = reference * (1 + 0.5j)
    data = reference * scale

    result = tk.decompositions.TTSVD(data, out_device=None).fit(
        rank=16, renormalize=True)

    assert all(torch.isfinite(core).all() for core in result.cores)
    # Compare in the original relative scale without overflowing a norm.
    assert_close(result.contract_dense() / scale, reference)


@pytest.mark.parametrize('ordering', ['interleaved', 'grouped'])
def test_quantized_svd_preserves_callable_grid(ordering, device_dtype, assert_close):
    device, dtype = device_dtype
    real_dtype = torch.empty((), dtype=dtype).real.dtype
    layout = tk.formats.QuantizedLayout(2, 2, (2, 2), ordering=ordering)
    coordinate_map = tk.formats.AffineCoordinateMap(
        torch.tensor([[-1., 1.], [2., 3.]], dtype=real_dtype, device=device),
        layout.grid_size)

    def function(coordinates):
        value = (coordinates[:, 0] + 2 * coordinates[:, 1]).to(dtype)
        return value * (1 + 1j) if dtype.is_complex else value

    source = tk.decompositions.QuanticsVectorSource(
        function, 2, layout=layout, coordinate_map=coordinate_map,
        dtype=dtype, device=device)
    engine = tk.decompositions.TTSVD.quantized(source, out_device=None)
    for _ in range(2):
        result = engine.fit(rank=16)
        assert isinstance(result, tk.decompositions.QTTDecomposition)
        assert result.layout == layout
        assert_close(result.to_dense_grid(), source.to_dense_grid())
        indices = torch.tensor([[0, 0], [3, 3], [2, 1]], device=device)
        assert_close(result.evaluate_indices(indices), source.evaluate_indices(indices))
        assert not result.metrics.truncations
        assert not isinstance(result + result, tk.decompositions.TensorDecomposition)
        assert result.dtype == dtype and result.device.type == device
        assert_close(result.coordinate_map.from_indices(indices),
                     coordinate_map.from_indices(indices))


def test_dense_svd_has_fixed_axes_and_no_quantization():
    data = torch.arange(16., dtype=torch.float64).reshape(4, 4)
    result = tk.decompositions.TTSVD(data).fit(rank=4)
    assert result.in_dim == (4, 4)
    assert not isinstance(result, tk.formats.QTT)
    torch.testing.assert_close(result.contract_dense(), data)
    with pytest.raises(TypeError, match='QuanticsVectorSource'):
        tk.decompositions.TTSVD.quantized(data)
    with pytest.raises(TypeError, match='quantization'):
        tk.decompositions.tt_svd(data, quantization=tk.formats.QuantizedLayout(2, 2, 2))
