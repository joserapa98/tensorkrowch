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


# Analytic functions and SampleError


@pytest.mark.parametrize('cyclic', [False, True])
@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_exponential_sample_error(exponential_format, cyclic, device_dtype,
                                  assert_close, ordering):
    device, dtype = device_dtype
    format, function = exponential_format(cyclic, dtype, ordering, device=device)
    indices = torch.cartesian_prod(torch.arange(4), torch.arange(9)).to(device)
    coordinates = format.coordinate_map.from_indices(indices)
    digits = format.layout.encode_indices(indices)
    target = function(coordinates)
    assert format.rank == [1] * (format.n_sites if cyclic else format.n_sites - 1)
    assert_close(format.evaluate_coordinates(coordinates), target)
    record = format.error(function, coordinates, data=digits)
    assert isinstance(record, tk.formats.SampleError)
    assert record.kind == 'samples' and record.size == 36
    assert_close(record.absolute, torch.zeros_like(record.absolute))
    assert_close(record.denominator, target.norm())

    changed = format * 1.1
    record = changed.error(function, coordinates, data=digits, scale=1.)
    assert_close(record.absolute, 0.1 * target.norm())
    assert_close(record.relative, target.real.new_tensor(0.1))

    changed.cores[0].requires_grad_()
    changed.error(function, coordinates, data=digits).absolute.backward()
    assert changed.cores[0].grad is not None
    assert torch.isfinite(changed.cores[0].grad).all()


@pytest.mark.parametrize('batch_shape', [(), (5,), (2, 3)])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_basis_embedding_function_and_error(batch_shape, dtype):
    indices = torch.arange(int(torch.Size(batch_shape).numel()) * 2).reshape(
        *batch_shape, 2) % 3
    coefficients = torch.tensor([0.2, -0.3], dtype=dtype)
    if dtype.is_complex:
        coefficients = coefficients + 0.1j
    vectors = [(coefficient * torch.arange(3)).exp() for coefficient in coefficients]
    format = tk.formats.TT([vectors[0][:, None], vectors[1][None, :]])
    data = tk.embeddings.basis(indices, dim=3)
    target = (indices * coefficients).sum(-1).exp()
    assert torch.allclose(format.evaluate(data, n_batches=len(batch_shape)), target)
    record = format.error(lambda samples: (samples * coefficients).sum(-1).exp(),
                          indices, data=data, n_batches=len(batch_shape))
    assert record.absolute < 1e-12
    assert torch.allclose(record.denominator, target.norm())


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_arbitrary_vector_inputs_and_error_gradients(dtype):
    generator = torch.Generator().manual_seed(73)
    first = torch.randn(3, dtype=dtype, generator=generator, requires_grad=True)
    second = torch.randn(4, dtype=dtype, generator=generator, requires_grad=True)
    data = (torch.randn(5, 3, dtype=dtype, generator=generator),
            torch.randn(5, 4, dtype=dtype, generator=generator))
    format = tk.formats.TR([first.reshape(1, 3, 1), second.reshape(1, 4, 1)])
    expected = (data[0] @ first) * (data[1] @ second)
    assert torch.allclose(format.evaluate(data), expected)
    record = format.error(lambda samples: expected.detach() * 0.9, data)
    expected_error = (expected - expected.detach() * 0.9).norm()
    actual_grad = torch.autograd.grad(record.absolute, (first, second), retain_graph=True)
    expected_grad = torch.autograd.grad(expected_error, (first, second))
    assert all(torch.allclose(actual, reference)
               for actual, reference in zip(actual_grad, expected_grad))


@pytest.mark.parametrize('zero_format', [False, True])
def test_sample_error_zero_target(zero_format):
    format = tk.formats.TT([torch.zeros(2) if zero_format else torch.ones(2)])
    samples = torch.tensor([[0], [1]])
    record = format.error(lambda values: torch.zeros(values.shape[0]), samples)
    assert record.denominator == 0
    assert record.relative == (0 if zero_format else torch.inf)


def test_sample_error_structural_batches():
    values = torch.tensor([[1., 2.], [2., 4.]], dtype=torch.float64)
    format = tk.formats.TT([values], n_batches=1)
    samples = torch.tensor([[0], [1]])
    target = values[0]
    record = format.error(lambda data: target, samples)
    assert torch.allclose(record.absolute, target.norm())
    assert torch.allclose(record.denominator, (2 * target.square().sum()).sqrt())
    assert record.size == 2


@pytest.mark.parametrize('kwargs,error', [
    ({'function': 1}, TypeError),
    ({'samples': 'bad'}, TypeError),
    ({'samples': []}, ValueError),
    ({'samples': [1]}, TypeError),
    ({'n_batches': True}, TypeError),
    ({'n_batches': -1}, ValueError),
    ({'function': lambda data: [1., 1.]}, TypeError),
    ({'function': lambda data: torch.ones(3)}, RuntimeError),
])
def test_sample_error_invalid_inputs(kwargs, error):
    arguments = dict(function=lambda data: torch.ones(2),
                     samples=torch.tensor([[0], [1]]))
    arguments.update(kwargs)
    with pytest.raises(error):
        tk.formats.TT([torch.ones(2)]).error(**arguments)


@pytest.mark.parametrize('groups,in_dim,out_dim', [
    ((), (2,), None), ((0, 1), (2,), None),
    ((True,), (2,), None), ((2,), (2,), None),
    ((1,), (2,), (2, 2)),
])
def test_block_layout_invalid(groups, in_dim, out_dim):
    with pytest.raises(ValueError):
        tk.formats.BlockLayout(groups, in_dim, out_dim)


def test_auxiliary_records_are_frozen_and_keep_references():
    value = torch.tensor(2., requires_grad=True)
    record = tk.formats.SampleError('samples', value, value, 3, value)
    assert record.absolute is record.relative is record.denominator is value
    with pytest.raises(FrozenInstanceError):
        record.size = 4
    layout = tk.formats.BlockLayout((1, 2), (2, 3, 4), (3, 2, 2))
    assert layout.groups == (1, 2) and layout.out_dim == (3, 2, 2)
    with pytest.raises(FrozenInstanceError):
        layout.groups = (3,)


@pytest.mark.parametrize('key,value', [
    (0, 4), (slice(0, 2), (4, 5)), (slice(None, None, -1), iter((5, 4))),
])
def test_safe_list_validated_replacements(key, value):
    from tensorkrowch.formats.base import _SafeList

    calls = []
    values = _SafeList([1, 2], lambda: calls.append(tuple(values)))
    values[key] = value
    assert values == [4, 2] if key == 0 else values == [4, 5]
    assert calls == [tuple(values)]


def test_safe_list_restores_values_after_callback_failure():
    from tensorkrowch.formats.base import _SafeList

    def reject():
        raise ValueError('Rejected replacement')

    values = _SafeList([1, 2], reject)
    for key, replacement in [(0, 3), (slice(None), (3, 4))]:
        with pytest.raises(ValueError, match='Rejected replacement'):
            values[key] = replacement
        assert values == [1, 2]
    with pytest.raises(ValueError, match='preserve length'):
        values[:] = [1]
    with pytest.raises(IndexError):
        values[2] = 3


@pytest.mark.parametrize('device', ['cpu', torch.device('cpu'), 'mps'])
def test_cuda_rejects_non_cuda_device(device):
    with pytest.raises(ValueError, match='CUDA device'):
        tk.formats.TT([torch.ones(2)]).cuda(device)


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_mps_rejects_unsupported_precision(dtype):
    if not torch.backends.mps.is_available():
        pytest.skip('MPS is unavailable')
    format = tk.formats.TT([torch.ones(2, dtype=dtype)])
    with pytest.raises(TypeError, match="doesn't support float64"):
        format.mps()


@pytest.mark.parametrize('device', ['cuda', 'mps'])
def test_device_conversion_checks_availability(device):
    available = (torch.cuda.is_available() if device == 'cuda' else
                 torch.backends.mps.is_available())
    if available:
        assert tk.formats.TT([torch.ones(2)]).to(device).device.type == device
        return
    with pytest.raises((AssertionError, RuntimeError), match='CUDA|cuda|MPS|mps'):
        tk.formats.TT([torch.ones(2)]).to(device)
