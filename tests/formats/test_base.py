"""Construction, replacement, storage and parity of compact formats."""

import pytest
import torch
import tensorkrowch as tk


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


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_result_parity(make_format, topology, n_sites, dtype):
    format = make_format(topology, n_sites, dtype=dtype)
    result_class = {'tt': tk.decompositions.TTDecomposition,
                    'tr': tk.decompositions.TRDecomposition,
                    'ttm': tk.decompositions.TTMDecomposition,
                    'trm': tk.decompositions.TRMDecomposition}[topology]
    result = result_class(format.cores)
    dense = result.contract_dense()
    assert torch.allclose(format.contract_dense(), dense)
    assert torch.allclose(format.norm(), dense.norm())
    assert torch.allclose(format.inner(format),
                          dense.abs().square().sum().to(dtype))
    assert torch.allclose(format.fidelity(format), dense.real.new_ones(()))
    inputs = torch.zeros(5, n_sites, dtype=torch.long)
    if format.out_dim is None:
        assert torch.allclose(format.evaluate(inputs), result.evaluate(inputs))
    else:
        outputs = torch.ones_like(inputs)
        assert torch.allclose(format.evaluate(inputs, outputs),
                              result.evaluate(inputs, outputs))
        assert torch.allclose(format.apply(inputs).contract_dense(),
                              result.apply(inputs).contract_dense())


@pytest.mark.parametrize('topology', ['tt', 'tr', 'trm'])
def test_independent_batches(make_format, topology):
    format = make_format(topology, n_batches=2)
    inputs = torch.zeros(2, format.n_sites, dtype=torch.long)
    if format.out_dim is None:
        values = format.evaluate(inputs)
    else:
        values = format.evaluate(inputs, inputs)
    assert values.shape == (2, 2, 2)
    expected = format.contract_dense()[(slice(None), slice(None)) +
                                       (0,) * (format.n_sites *
                                               (2 if format.out_dim else 1))]
    assert torch.allclose(values, expected.unsqueeze(-1).expand_as(values))


def test_core_replacement_and_cache():
    format = tk.formats.TT([torch.ones(2, 3), torch.ones(3, 4)])
    format.cores[:] = [torch.ones(2, 5), torch.ones(5, 4)]
    assert format.rank == [5]
    rank = format.rank
    rank[0] = 1
    assert format.rank == [5]
    assert torch.equal(format.contract_dense(), torch.full((2, 4), 5.))
    format.cores[0] = torch.ones(6, 5)
    assert format.in_dim == (6, 4)
    with pytest.raises(ValueError):
        format.cores = [torch.ones(6, 2), torch.ones(3, 4)]
    assert format.rank == [5]
    with pytest.raises(ValueError, match='Adjacent'):
        format.cores[0] = torch.ones(6, 2)
    assert format.rank == [5]
    assert format.in_dim == (6, 4)
    assert torch.equal(format.contract_dense(), torch.full((6, 4), 5.))
    format.cores[:] = [torch.ones(6, 2), torch.ones(2, 4)]
    assert format.rank == [2]


@pytest.mark.parametrize('operation', [lambda c: c.append(torch.ones(1)),
                                      lambda c: c.reverse(),
                                      lambda c: c.pop(),
                                      lambda c: c.clear()])
def test_structural_mutations_rejected(make_format, operation):
    format = make_format()
    with pytest.raises(TypeError):
        operation(format.cores)
    with pytest.raises(ValueError):
        format.cores[:] = [torch.ones(2)]
    with pytest.raises(TypeError):
        format.cores[0] = 1


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_clone_detach_and_conversion(make_format, topology):
    format = make_format(topology)
    format.cores[0].requires_grad_()
    assert format.to() is format and format.cpu() is format
    copied, detached = format.clone(), format.detach()
    assert type(copied) is type(format)
    assert copied.cores[0].data_ptr() != format.cores[0].data_ptr()
    assert copied.cores[0].requires_grad
    assert detached.cores[0].data_ptr() == format.cores[0].data_ptr()
    assert not detached.cores[0].requires_grad
    assert format.to(dtype=torch.complex128).dtype == torch.complex128
    assert format.detach_() is format
    assert not format.cores[0].requires_grad


def test_invalid_construction_and_data(make_format):
    for cores in [[], [torch.empty(0)], [torch.ones(2), torch.ones(2)],
                  [torch.ones(2, 1), torch.ones(1, 3, dtype=torch.float64)]]:
        with pytest.raises(ValueError):
            tk.formats.TT(cores)
    with pytest.raises(TypeError):
        tk.formats.TT([1])
    with pytest.raises(TypeError):
        tk.formats.TT([torch.ones(2)], n_batches=True)
    with pytest.raises(ValueError):
        make_format('ttm', n_batches=1)
    format = make_format()
    with pytest.raises(ValueError):
        format.evaluate(torch.full((2, 3), -1, dtype=torch.long))
    with pytest.raises(ValueError):
        format.evaluate(torch.ones(2, 3, 2))
    zero = tk.formats.TT([torch.zeros(2)])
    assert zero.norm() == 0
    with pytest.raises(ValueError, match='zero-norm'):
        zero.normalized_overlap(zero)


def test_embedded_inputs_promote_dtype():
    vector = tk.formats.TT([torch.tensor([1., 2.])])
    real_values = vector.evaluate(
        torch.tensor([[[1., 2.]]], dtype=torch.float64))
    assert real_values.dtype == torch.float64
    embedded = torch.tensor([[[1 + 1j, 2 - 1j]]], dtype=torch.complex128)
    values = vector.evaluate(embedded)
    assert values.dtype == torch.complex128
    assert torch.equal(values, torch.tensor([5 - 1j], dtype=values.dtype))

    vector = tk.formats.TT([
        torch.tensor([[1.], [2.]]), torch.tensor([[3., 4.]])])
    first = torch.tensor([[1 + 1j, 2 - 1j]], dtype=torch.complex64)
    second = torch.tensor([[1., 2.]], dtype=torch.float64)
    values = vector.evaluate([first, second])
    assert values.dtype == torch.complex128
    assert torch.equal(values, torch.tensor([55 - 11j], dtype=values.dtype))

    matrix = tk.formats.TTM([torch.eye(2)])
    output = torch.tensor([[[3., 4.]]], dtype=torch.float64)
    values = matrix.evaluate(embedded, output)
    assert values.dtype == torch.complex128
    assert torch.equal(values, torch.tensor([11 - 1j], dtype=values.dtype))
    assert matrix.evaluate(embedded, torch.tensor([[1]])).dtype == \
        torch.complex128
    assert matrix.evaluate(torch.tensor([[1]]), embedded).dtype == \
        torch.complex128

    applied = matrix.apply(embedded)
    assert applied.dtype == torch.complex128
    assert torch.equal(applied.evaluate(torch.tensor([[0], [1]])),
                       embedded.squeeze(0))


@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_scaled_norm_preserves_core_gradients(make_format, dtype):
    format = make_format('tt', 2, dtype=dtype)
    for core in format.cores:
        core.requires_grad_()
    expected = format.contract_dense().norm()
    actual = format.norm()
    expected_grad = torch.autograd.grad(expected, format.cores, retain_graph=True)
    actual_grad = torch.autograd.grad(actual, format.cores)
    assert torch.allclose(actual, expected)
    assert all(torch.allclose(a, b, rtol=1e-9, atol=1e-10)
               for a, b in zip(actual_grad, expected_grad))
