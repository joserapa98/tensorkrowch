"""Tests for the TT-SVD class and functional interfaces."""

import math

import pytest

import torch
import tensorkrowch as tk

import tensorkrowch.decompositions.svd.tt as tt_module
from tensorkrowch.decompositions.svd.common import _log_vector_norm


SVD_METHODS = ['svd', 'qr_svd']
DEVICE_NAMES = ['cpu', 'cuda', 'mps']


def _device(name):
    if name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    if name == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS is not available')
    return torch.device(name)


class TestTTSVD:  # MARK: TestTTSVD

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
            tensor, output_device=None)

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
    def test_batched_errors_are_per_batch_and_frobenius(self, renormalize):
        generator = torch.Generator().manual_seed(1)
        tensor = torch.randn(
            3, 2, 3, 4, dtype=torch.float64, generator=generator)
        tensor[0] = 0

        result = tk.decompositions.TTSVD(
            tensor, n_batches=1, output_device=None).fit(
                rank=2,
                renormalize=renormalize,
                collect_metrics=True)

        difference = (tensor - result.contract_dense()).flatten(1)
        absolute_per_batch = difference.norm(dim=-1)
        input_norm_per_batch = tensor.flatten(1).norm(dim=-1)
        relative_per_batch = torch.where(
            input_norm_per_batch > 0,
            absolute_per_batch / input_norm_per_batch,
            torch.zeros_like(absolute_per_batch))
        record = result.metrics.errors[0]

        assert torch.allclose(record.absolute_per_batch,
                              absolute_per_batch,
                              rtol=1e-10,
                              atol=1e-10)
        assert torch.allclose(record.relative_per_batch,
                              relative_per_batch,
                              rtol=1e-10,
                              atol=1e-10)
        assert record.absolute == pytest.approx(
            absolute_per_batch.norm().item(), rel=1e-10, abs=1e-10)
        assert record.relative == pytest.approx(
            (absolute_per_batch.norm() /
             input_norm_per_batch.norm()).item(),
            rel=1e-10,
            abs=1e-10)
        assert all(torch.isfinite(core).all() for core in result.cores)

    def test_repeated_fits_have_independent_state(self):
        tensor = torch.randn(3, 4, 5, dtype=torch.float64)
        decomposer = tk.decompositions.TTSVD(
            tensor, output_device=None)

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
            tensor, output_device=None)
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

        monkeypatch.setattr(tt_module, '_log_vector_norm', unexpected_log_norm)
        monkeypatch.setattr(
            tt_module._RuntimePolicy, 'timer', unexpected_timer)
        monkeypatch.setattr(tt_module, 'DecompositionEvent', unexpected_event)
        result = tk.decompositions.TTSVD(
            torch.randn(2, 3, 4), output_device=None).fit(
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
            tensor, output_device=None).fit(
                rank=1,
                renormalize=True,
                collect_metrics=True)
        scaled = tk.decompositions.TTSVD(
            tensor * scale, output_device=None).fit(
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
        assert all(math.isfinite(record.local_absolute_error)
                   for record in scaled.metrics.truncations)
        assert all(torch.isfinite(core).all() for core in scaled.cores)

    def test_log_metrics_match_direct_moderate_scale_calculations(self):
        generator = torch.Generator().manual_seed(5)
        tensor = 3 * torch.randn(
            3, 4, 5, dtype=torch.float64, generator=generator)
        result = tk.decompositions.TTSVD(
            tensor, output_device=None).fit(
                rank=2,
                renormalize=True,
                collect_metrics=True)

        residual = tensor
        previous_rank = 1
        local_errors = []
        for site, input_dim in enumerate(tensor.shape[:-1]):
            residual = residual.reshape(
                previous_rank * input_dim, -1)
            _, singular_values, vh = torch.linalg.svd(
                residual, full_matrices=False)
            local_errors.append(singular_values[2:].square().sum().sqrt())
            residual = singular_values[:2].unsqueeze(-1) * vh[:2]
            previous_rank = 2

        direct_absolute = torch.stack(local_errors).square().sum().sqrt()
        direct_relative = direct_absolute / torch.linalg.vector_norm(tensor)
        log_input_norm = _log_vector_norm(tensor)

        assert log_input_norm.exp() == pytest.approx(
            torch.linalg.vector_norm(tensor).item(), rel=1e-12)
        assert [record.local_absolute_error
                for record in result.metrics.truncations] == pytest.approx(
                    [error.item() for error in local_errors], rel=1e-12)
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
            output_device=None).fit(
                renormalize=True,
                **kwargs)

        assert result.rank == [2]

    def test_one_site_and_zero_tensor(self):
        tensor = torch.zeros(5)
        result = tk.decompositions.TTSVD(
            tensor, output_device=None).fit(
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

    @pytest.mark.parametrize('device_name', DEVICE_NAMES)
    def test_output_device_policy(self, device_name):
        device = _device(device_name)
        tensor = torch.randn(2, 3, 4, device=device)

        cpu_result = tk.decompositions.TTSVD(tensor).fit(rank=2)
        active_result = tk.decompositions.TTSVD(
            tensor, output_device=None).fit(rank=2)

        assert all(core.device.type == 'cpu' for core in cpu_result.cores)
        assert all(core.device == device for core in active_result.cores)
        assert torch.allclose(
            active_result.contract_dense(),
            cpu_result.contract_dense().to(device),
            rtol=1e-5,
            atol=1e-6)

    def test_history_and_console_observers(self, capsys):
        history = tk.decompositions.HistoryObserver()
        tensor = torch.randn(2, 3, 4)

        result = tk.decompositions.TTSVD(
            tensor, output_device=None).fit(
                rank=2,
                verbose=2,
                observer=history)

        output = capsys.readouterr().out
        assert 'TT-SVD\n======' in output
        assert 'Site 1 / 2' in output
        assert 'selected rank: 2' in output
        assert 'Summary\n-------' in output
        assert history.metrics is result.metrics
        assert len(result.metrics.errors) == 1
        assert len(result.metrics.truncations) == 2
        assert len(result.metrics.timings) == 1
        assert [event.name for event in history.events] == [
            'start',
            'site_complete',
            'site_complete',
            'summary',
            'core',
            'core',
            'core',
        ]

    @pytest.mark.parametrize(
        'constructor, error_type, match',
        [
            (([1, 2],), TypeError, '`tensor` should be torch.Tensor type'),
            ((torch.ones(2), 1), ValueError, 'leave at least one'),
            ((torch.ones(2), -1), ValueError, 'leave at least one'),
        ],
    )
    def test_constructor_errors(self, constructor, error_type, match):
        with pytest.raises(error_type, match=match):
            tk.decompositions.TTSVD(*constructor)

    @pytest.mark.parametrize(
        'kwargs, error_type, match',
        [
            ({'rank': 0}, ValueError, '`rank` should be a positive integer'),
            ({'rtol': 2}, ValueError, '`rtol` should be a number between'),
            ({'renormalize': 1}, TypeError,
             '`renormalize` should be bool type'),
            ({'collect_metrics': 1}, TypeError,
             '`collect_metrics` should be bool type'),
            ({'verbose': -1}, ValueError,
             '`verbose` should be between 0 and 3'),
            ({'verbose': 4}, ValueError,
             '`verbose` should be between 0 and 3'),
            ({'observer': object()}, TypeError,
             '`observer` should implement'),
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
                value, output_device=None).fit().contract_dense()

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
            cores, info = tk.decompositions.tt_svd(
                tensor,
                rank=2,
                output_device=None,
                return_info=True)
            legacy_cores = tk.decompositions.vec_to_mps(tensor, rank=2)

        assert info['rank'] == [2, 2]
        assert info['metadata']['algorithm'] == 'tt_svd'
        for core, legacy_core in zip(cores, legacy_cores):
            assert torch.allclose(core, legacy_core)

    def test_return_info_validation(self):
        with pytest.raises(TypeError, match='`return_info` should be bool type'):
            tk.decompositions.tt_svd(
                torch.ones(2, 3), return_info=1)

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
        _, info = tk.decompositions.tt_svd(
            tensor,
            rank=2,
            return_info=True)

        assert return_info_calls == [False, False, True, True]
        assert len(info['metrics']['errors']) == 1
        assert len(info['metrics']['truncations']) == 2
        assert len(info['metrics']['timings']) == 1
