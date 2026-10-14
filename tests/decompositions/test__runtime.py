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
