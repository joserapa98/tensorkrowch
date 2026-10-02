"""Frozen format records retain tensor references and autograd."""

from dataclasses import FrozenInstanceError, is_dataclass

import pytest
import torch
import tensorkrowch as tk

from tensorkrowch.formats.orbits import MinimalCanonicalInfo


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
@pytest.mark.parametrize('kind', ['split', 'rounding', 'minimal'])
def test_records_preserve_tensors_and_gradients(kind, dtype):
    tensor = torch.ones(2, dtype=dtype, requires_grad=True)
    value = tensor.abs().square()
    if kind == 'split':
        record = tk.formats.SplitBlock((tensor,), None, (value,))
        stored, frozen_field = record.spectra[0], 'cores'
        assert record.cores[0] is tensor
    elif kind == 'rounding':
        record = tk.formats.RoundingInfo((2,), (value,), value, None)
        stored, frozen_field = record.error_bound, 'rank'
        assert record.discarded_sq_norm[0] is value
    else:
        record = MinimalCanonicalInfo(0, False, value)
        stored, frozen_field = record.gram_imbalance, 'iterations'

    assert is_dataclass(record)
    assert stored is value
    with pytest.raises(FrozenInstanceError):
        setattr(record, frozen_field, None)
    gradient, = torch.autograd.grad(stored.sum(), tensor)
    assert torch.allclose(gradient, 2 * tensor)
