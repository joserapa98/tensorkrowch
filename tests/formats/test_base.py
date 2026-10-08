"""Shared format records and sample error evaluation."""

from dataclasses import FrozenInstanceError, is_dataclass

import pytest
import torch

import tensorkrowch as tk
from tensorkrowch.formats.orbits import MinimalCanonicalInfo


# Sample errors


@pytest.mark.parametrize('target', [
    torch.tensor([1 + 2 ** -30], dtype=torch.float64),
    torch.tensor([1 + 2j], dtype=torch.complex128),
])
def test_error_preserves_target_precision_and_complex_values(target):
    vector = tk.formats.TT([torch.ones(2, dtype=torch.float32)])
    record = vector.error(lambda samples: target, torch.tensor([[0]]))
    expected = (torch.ones_like(target) - target).norm()
    assert record.absolute.dtype == torch.float64
    assert torch.equal(record.absolute, expected)
    assert torch.equal(record.denominator, target.norm())
    assert torch.equal(record.relative, expected / target.norm())


@pytest.mark.parametrize('container', ['tensor', 'sequence'])
def test_error_accepts_sample_containers(container):
    values = torch.tensor([[[1., 2.], [3., 4.]], [[2., 3.], [4., 5.]]])
    samples = values if container == 'tensor' else tuple(values.unbind(1))
    vector = tk.formats.TT([torch.ones(2, 1), torch.ones(1, 2)])

    def function(argument):
        if isinstance(argument, torch.Tensor):
            return argument.sum(-1).prod(-1)
        return argument[0].sum(-1) * argument[1].sum(-1)

    record = vector.error(function, samples)
    assert record.absolute == 0
    assert record.size == 2


def test_error_accepts_heterogeneous_samples_with_embedding():
    samples = (torch.tensor([1., 2.]), torch.tensor([[3., 4.], [5., 6.]]))
    data = (torch.stack((samples[0], samples[0].square()), dim=-1), samples[1])
    vector = tk.formats.TT([torch.ones(2, 1), torch.ones(1, 2)])

    def function(argument):
        scalar, features = argument
        return (scalar + scalar.square()) * features.sum(-1)

    assert vector.error(function, samples, data=data).absolute == 0
    with pytest.raises(ValueError, match='same batch shape'):
        vector.error(function, (samples[0], samples[1][:1]), data=data)
    with pytest.raises(ValueError, match='matching batch shapes'):
        vector.error(function, samples, data=tuple(item[:1] for item in data))


@pytest.mark.parametrize('batch_shape', [(), (3,), (2, 3)])
@pytest.mark.parametrize('embedded', [False, True])
def test_error_accepts_feature_samples(batch_shape, embedded):
    samples = torch.arange(
        int(torch.Size(batch_shape).numel()) * 4,
        dtype=torch.float64).reshape(*batch_shape, 2, 2) / 10
    data = (torch.cat((samples, torch.ones_like(samples[..., :1])), dim=-1)
            if embedded else None)
    in_dim = 3 if embedded else 2
    vector = tk.formats.TT([
        torch.ones(in_dim, 1, dtype=torch.float64),
        torch.ones(1, in_dim, dtype=torch.float64)])
    expected = (samples.sum(-1) + int(embedded)).prod(-1)

    def function(values):
        assert values.shape == samples.shape
        return 2 * (values.sum(-1) + int(embedded)).prod(-1)

    record = vector.error(function, samples, data=data,
                          n_batches=len(batch_shape))
    assert torch.allclose(record.absolute, expected.norm())
    assert torch.allclose(record.relative, torch.tensor(0.5, dtype=torch.float64))
    assert record.size == int(torch.Size(batch_shape).numel())


# Records


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
