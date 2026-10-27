"""Tests for _runtime."""


import pytest
import torch

from tensorkrowch.decompositions._runtime import _RuntimePolicy


class TestRuntimePolicy:  # MARK: TestRuntimePolicy

    def test_inference_prepare_finalize_and_timer(self):
        tensor = torch.ones(2, dtype=torch.float32)
        runtime = _RuntimePolicy.from_tensor(tensor, dtype=torch.float64)

        prepared = runtime.prepare(tensor)
        finalized = runtime.finalize(prepared)
        with runtime.timer() as timer:
            _ = prepared.square()

        assert runtime.device == tensor.device
        assert runtime.out_device == torch.device('cpu')
        assert runtime.dtype == torch.float64
        assert prepared.dtype == torch.float64
        assert finalized.device.type == 'cpu'
        assert timer.elapsed is not None
        assert timer.elapsed >= 0

    def test_output_device_none_keeps_tensor(self):
        tensor = torch.ones(2)
        runtime = _RuntimePolicy.from_tensor(tensor, out_device=None)

        assert runtime.finalize(tensor) is tensor

    def test_invalid_dtype(self):
        with pytest.raises(TypeError):
            _RuntimePolicy(dtype='float64')

    @pytest.mark.parametrize('operation', ['from_tensor', 'prepare', 'finalize'])
    def test_tensor_operations_reject_other_types(self, operation):
        with pytest.raises(TypeError, match='tensor'):
            getattr(_RuntimePolicy(), operation)([1., 2.])

    @pytest.mark.parametrize('device', ['cuda:0', 'mps'])
    @pytest.mark.parametrize('synchronize', [False, True])
    def test_timer_synchronization_policy(self, device, synchronize, monkeypatch):
        calls = []
        monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: calls.append(args))
        monkeypatch.setattr(torch.mps, 'synchronize', lambda: calls.append(()))
        runtime = _RuntimePolicy(device=device, synchronize_timers=synchronize)
        with runtime.timer() as timer:
            pass
        assert len(calls) == (2 if synchronize else 0)
        assert timer.elapsed >= 0

    def test_timer_keeps_exceptions_and_records_elapsed_time(self):
        with pytest.raises(RuntimeError, match='operation failed'):
            with _RuntimePolicy().timer() as timer:
                raise RuntimeError('operation failed')
        assert timer.elapsed >= 0
