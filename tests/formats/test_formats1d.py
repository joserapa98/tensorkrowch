"""One-dimensional formats, algebra, canonical forms and conversions."""

from itertools import product

import pytest
import torch

import tensorkrowch as tk


# Adapters


@pytest.mark.parametrize('topology', ['tr', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('n_batches', [0, 1])
def test_rotation_and_train_conversion(make_format, topology, n_sites, n_batches):
    format = make_format(topology, n_sites, n_batches, torch.complex128)
    format.bonds = [
        torch.arange(1, rank + 1, dtype=torch.float64) for rank in format.rank]
    dense = format.contract_dense()
    b, width = n_batches, 2 if topology == 'trm' else 1
    for first in range(n_sites):
        rotated = format.rotate(first)
        axes = [*range(b), *range(b + first * width, b + n_sites * width),
                *range(b, b + first * width)]
        assert torch.allclose(rotated.contract_dense(), dense.permute(axes))
        train = rotated.to_tt() if topology == 'tr' else rotated.to_ttm()
        assert train.batch_shape == rotated.batch_shape
        assert torch.allclose(train.contract_dense(), rotated.contract_dense())
        if n_sites > 1:
            closing = rotated.rank[-1]
            assert train.rank == [closing * rank for rank in rotated.rank[:-1]]


def test_invalid_rotation(make_format):
    for first in [-1, 3]:
        with pytest.raises(ValueError):
            make_format('tr').rotate(first)
    with pytest.raises(TypeError):
        make_format('tr').rotate(True)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
def test_model_roundtrip(make_format, topology, n_sites):
    format = make_format(topology, n_sites, dtype=torch.complex128)
    dense = format.contract_dense()
    if topology.endswith('m'):
        model = format.to_mpo()
        restored = model.to_trm() if topology == 'trm' else model.to_ttm()
    else:
        model = format.to_mps()
        restored = model.to_tr() if topology == 'tr' else model.to_tt()
    assert torch.allclose(restored.contract_dense(), dense)


@pytest.mark.parametrize('topology', ['tt', 'tr'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
def test_mps_data_roundtrip(make_format, topology, n_sites):
    format = make_format(topology, n_sites, n_batches=2)
    model = format.to_mps()
    assert isinstance(model, tk.models.MPSData)
    restored = model.to_tt() if topology == 'tt' else model.to_tr()
    assert restored.batch_shape == (2, 2)
    assert torch.allclose(restored.contract_dense(), format.contract_dense())


# Construction and core ownership


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
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
    cores = list(make_format('ttm', n_batches=1).cores)
    cores[-1] = cores[-1][:1]
    with pytest.raises(ValueError, match='same batch shape'):
        tk.formats.TTM(cores, n_batches=1)
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


# Blocking


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('groups', [(1, 1, 1, 1), (2, 2), (1, 3), (4,)])
def test_block_roundtrip(make_format, topology, groups):
    format = make_format(topology, 4, dtype=torch.complex128)
    count = 4 if topology.startswith('tr') else 3
    format.bonds = [
        torch.arange(1, rank + 1, dtype=torch.float64) for rank in format.rank[:count]]
    dense = format.contract_dense()
    in_dim, out_dim = format.in_dim, format.out_dim
    layout = format.block(groups)
    assert format.n_sites == len(groups)
    assert layout.in_dim == in_dim and layout.out_dim == out_dim
    assert format.unblock(layout) is format
    assert torch.allclose(format.contract_dense(), dense)
    assert format.in_dim == in_dim and format.out_dim == out_dim


@pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse', 'left', 'right'])
def test_local_update(make_format, mode):
    format = make_format('tr', 4)
    block = format.contract_block(1, 2)
    replacement = tk.formats.split_block(
        block * 2, format.in_dim[1:3], mode=mode)
    dense = format.contract_dense()
    format.replace_cores(1, replacement.cores, bonds=replacement.bonds)
    assert torch.allclose(format.contract_dense(), 2 * dense)
    format.absorb_bond(1, 'right')
    assert torch.allclose(format.contract_dense(), 2 * dense)


def test_invalid_blocks(make_format):
    format = make_format()
    for groups in [[], [0, 3], [True, 2], [2]]:
        with pytest.raises(ValueError):
            format.block(groups)
    with pytest.raises(ValueError):
        format.unblock(tk.formats.BlockLayout((2,), (2, 3)))
    with pytest.raises(ValueError):
        format.contract_block(2, 0)
    before = format.contract_dense()
    with pytest.raises(ValueError):
        format.replace_cores(1, [torch.ones(7, 3, 9)])
    assert torch.equal(format.contract_dense(), before)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
def test_layout_applies_to_new_solution(make_format, topology, n_sites):
    format = make_format(topology, n_sites, dtype=torch.complex128)
    dense = format.contract_dense()
    layout = format.block([n_sites])
    solution = type(format)([2 * core for core in format.cores])
    assert solution.unblock(layout) is solution
    assert torch.allclose(solution.contract_dense(), 2 * dense)
    assert format.n_sites == 1
    assert format.bonds is None


@pytest.mark.parametrize('topology', ['tt', 'tr', 'trm'])
@pytest.mark.parametrize('n_batches', [1, 2])
def test_batched_blocking(make_format, topology, n_batches):
    format = make_format(topology, 4, n_batches=n_batches)
    dense = format.contract_dense()
    layout = format.block([2, 2])
    assert format.batch_shape == (2,) * n_batches
    format.unblock(layout)
    assert torch.allclose(format.contract_dense(), dense)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_unblocking_truncates_only_internal_bonds(make_format, topology):
    format = make_format(topology, 4)
    layout = format.block([2, 2])
    boundary_rank = format.rank
    format.unblock(layout, rank=1)
    assert format.rank[0] == format.rank[2] == 1
    assert format.rank[1] == boundary_rank[0]
    if topology.startswith('tr'):
        assert format.rank[-1] == boundary_rank[-1]


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('failure', ['dimensions', 'family', 'svd'])
def test_unblocking_failure_preserves_state(make_format, topology, failure):
    format = make_format(topology, 4)
    layout = format.block([2, 2])
    if failure == 'dimensions':
        layout = tk.formats.BlockLayout(layout.groups, (3, 3, 2, 3), layout.out_dim)
    elif failure == 'family':
        outputs = layout.in_dim if format.out_dim is None else None
        layout = tk.formats.BlockLayout(layout.groups, layout.in_dim, outputs)
    cores = format.cores
    before = format.contract_dense()
    dimensions, rank = format.in_dim, format.rank
    with pytest.raises(ValueError):
        format.unblock(layout, mode='invalid' if failure == 'svd' else 'right')
    assert format.cores is cores
    assert format.in_dim == dimensions and format.rank == rank
    assert torch.equal(format.contract_dense(), before)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse', 'left', 'right'])
def test_coupled_core_and_bond_update(make_format, topology, mode, monkeypatch):
    format = make_format(topology, 4, dtype=torch.complex128)
    count = format.n_sites if topology.startswith('tr') else format.n_sites - 1
    format.bonds = [
        torch.arange(1, rank + 1, dtype=torch.float64)
        for rank in format.rank[:count]]
    dense = format.contract_dense()
    factors = format.bonds.factors
    block = format.contract_block(1, 2)
    outputs = None if format.out_dim is None else format.out_dim[1:3]
    replacement = tk.formats.split_block(
        block, format.in_dim[1:3], outputs, rank=1, mode=mode)
    calls = []
    validate = format.validate

    def counted_validate():
        calls.append(True)
        return validate()

    monkeypatch.setattr(format, 'validate', counted_validate)
    assert format.replace_cores(1, replacement.cores, bonds=replacement.bonds) is format
    assert len(calls) == 1
    assert format.rank[1] == 1
    assert format.bonds.factors[0] is factors[0]
    assert format.bonds.factors[2] is factors[2]
    assert format.contract_dense().shape == dense.shape
    with pytest.raises(ValueError, match='factor dimensions'):
        format.bonds.factors[0] = torch.ones(100, dtype=format.dtype)
    assert format.bonds.factors[0] is factors[0]
    left, right = replacement.cores
    factor = replacement.bonds[0]
    if factor is not None:
        left = left * factor[None, None, :]
    local = torch.einsum('aib,bjc->aijc', left, right).reshape(block.shape)
    assert torch.allclose(format.contract_block(1, 2), local)


@pytest.mark.parametrize('failure', ['rank', 'dtype', 'factor', 'batch'])
def test_replacement_failure_preserves_cores_and_bonds(make_format, failure):
    format = make_format('tr', 4)
    format.bonds = [
        torch.ones(rank, dtype=format.dtype) for rank in format.rank]
    block = format.contract_block(1, 2)
    replacement = tk.formats.split_block(block, format.in_dim[1:3], mode='explicit')
    values = list(replacement.cores)
    bonds = replacement.bonds
    if failure == 'rank':
        values[0] = values[0][..., :-1]
    elif failure == 'dtype':
        values[0] = values[0].to(torch.float32)
    elif failure == 'factor':
        bonds = [torch.ones(100)]
    else:
        values[0] = values[0].unsqueeze(0)
    cores, bond_container = format.cores, format.bonds
    dense = format.contract_dense()
    with pytest.raises(ValueError):
        format.replace_cores(1, values, bonds=bonds)
    assert format.cores is cores and format.bonds is bond_container
    with pytest.raises(ValueError, match='factor dimensions'):
        bond_container.factors[0] = torch.ones(100, dtype=format.dtype)
    assert torch.equal(format.contract_dense(), dense)


# Canonical


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('renormalize', [False, True])
def test_qr_gauges(make_format, topology, n_sites, renormalize, monkeypatch):
    format = make_format(topology, n_sites, dtype=torch.complex128)
    dense = format.contract_dense()

    def unexpected_svd(*args, **kwargs):
        raise AssertionError('QR canonicalization should not use SVD')

    monkeypatch.setattr(torch.linalg, 'svd', unexpected_svd)
    for orth_center in range(n_sites):
        result = format.clone().canonicalize(orth_center=orth_center,
                                              renormalize=renormalize)
        assert torch.allclose(result.contract_dense(), dense, rtol=1e-10, atol=1e-12)
        for site, core in enumerate(result._effective_cores()):
            if site < orth_center:
                matrix = core.reshape(-1, core.shape[-1])
                assert torch.allclose(matrix.adjoint() @ matrix,
                                      torch.eye(matrix.shape[-1], dtype=matrix.dtype))
            elif site > orth_center:
                matrix = core.reshape(core.shape[0], -1)
                assert torch.allclose(matrix @ matrix.adjoint(),
                                      torch.eye(matrix.shape[0], dtype=matrix.dtype))


def test_invalid_qr_and_zero(make_format):
    format = make_format()
    for orth_center in [-1, 3]:
        with pytest.raises(ValueError):
            format.canonicalize(orth_center)
    with pytest.raises(TypeError):
        format.canonicalize(True)
    with pytest.raises(TypeError):
        format.canonicalize(renormalize=1)
    format.cores[0] = format.cores[0] * 0
    format.canonicalize(renormalize=True)
    assert format.norm() == 0


# Mutations


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('replacement', ['element', 'slice', 'full'])
def test_invalid_cores_restore_state(make_format, topology, n_sites, replacement):
    format = make_format(topology, n_sites).canonicalize(orth_center=0)
    cores = format.cores
    previous = tuple(cores)
    rank, dimensions = format.rank, format.in_dim
    dense = format.contract_dense()
    invalid = cores[0].to(torch.complex128)
    if n_sites == 1:
        invalid = cores[0].unsqueeze(0)

    with pytest.raises(ValueError):
        if replacement == 'element':
            format.cores[0] = invalid
        elif replacement == 'slice':
            format.cores[:1] = [invalid]
        else:
            format.cores = [invalid, *cores[1:]]

    assert format.cores is cores
    assert all(new is old for new, old in zip(format.cores, previous))
    assert format.rank == rank and format.in_dim == dimensions
    assert format._orth_center == 0
    assert torch.allclose(format.contract_dense(), dense)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_shared_rank_replacement_is_atomic(make_format, topology):
    format = make_format(topology, 2)
    cores = format.cores
    first, last = cores
    right_axis = -2 if format.out_dim is not None else -1
    left_axis = 0
    shape = list(first.shape)
    shape[right_axis] += 1
    new_first = first.new_ones(shape)
    shape = list(last.shape)
    shape[left_axis] += 1
    new_last = last.new_ones(shape)

    with pytest.raises(ValueError, match='ranks'):
        cores[0] = new_first
    assert cores[0] is first and cores[1] is last
    cores[:] = [new_first, new_last]
    assert format.cores is cores
    assert format.rank[0] == new_first.shape[right_axis]
    assert torch.isfinite(format.contract_dense()).all()


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_read_operations_do_not_validate(make_format, topology, monkeypatch):
    format = make_format(topology)
    format.bonds = [None] * len(format.rank)
    dense = format.contract_dense()

    def unexpected_validation(*args, **kwargs):
        raise AssertionError('Read operations should not validate format structure')

    monkeypatch.setattr(format, 'validate', unexpected_validation)
    monkeypatch.setattr(format, 'validate_bonds', unexpected_validation)
    monkeypatch.setattr(format.bonds, 'validate', unexpected_validation)
    for _ in range(2):
        assert format.rank and format.in_dim
        assert format.device == dense.device and format.dtype == dense.dtype
        assert torch.allclose(format.contract_dense(), dense)
        assert torch.allclose(format.norm(), dense.norm())
        assert torch.allclose(format.inner(format), dense.square().sum())
        indices = torch.zeros(2, format.n_sites, dtype=torch.long)
        result = (format.evaluate(indices) if format.out_dim is None else
                  format.evaluate(indices, indices))
        assert result.shape == (2,)


def test_manual_rank_change_discards_vidal_spectrum_use(make_format):
    format = make_format('tt', 2).canonicalize_vidal('implicit')
    format.cores[:] = [torch.ones(2, 5, dtype=format.dtype),
                       torch.ones(5, 3, dtype=format.dtype)]
    assert format.rank == [5] and not format.bonds._valid
    dense = format.contract_dense()
    assert torch.allclose(dense, torch.full((2, 3), 5., dtype=format.dtype))
    assert torch.allclose(format.materialize_bonds().contract_dense(), dense)


def test_core_replacement_checks_existing_bonds(make_format):
    format = make_format('tt', 2)
    original_rank = format.rank
    format.bonds = [
        torch.ones(original_rank[0], dtype=format.dtype)]
    cores, bonds = format.cores, format.bonds
    previous = tuple(cores)
    with pytest.raises(ValueError, match='factor dimensions'):
        cores[:] = [torch.ones(2, 5, dtype=format.dtype),
                    torch.ones(5, 3, dtype=format.dtype)]
    assert format.cores is cores and format.bonds is bonds
    assert all(new is old for new, old in zip(cores, previous))
    assert format.rank == original_rank


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('step', [2, -2])
@pytest.mark.parametrize('container', ['cores', 'bonds'])
def test_extended_slice_generators_preserve_identity(make_format, topology,
                                                     step, container):
    format = make_format(topology, 4).canonicalize(orth_center=0)
    if container == 'bonds':
        format.bonds = [
            torch.ones(rank, dtype=format.dtype) for rank in format.rank]
    format._orth_center = 0
    values = format.cores if container == 'cores' else format.bonds.factors
    previous = tuple(values)
    dense = format.contract_dense()
    key = slice(None, None, step)
    indices = range(len(values))[key]
    values[key] = (2 * value for value in values[key])

    stored = format.cores if container == 'cores' else format.bonds.factors
    assert stored is values
    assert format._orth_center is None
    for index, value in enumerate(values):
        if index in indices:
            assert torch.equal(value, 2 * previous[index])
        else:
            assert value is previous[index]
    assert torch.allclose(format.contract_dense(), 2 ** len(indices) * dense)


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('copy_method', ['clone', 'detach', 'detach_', 'to'])
def test_core_callbacks_are_independent(make_format, topology, copy_method):
    format = make_format(topology).canonicalize(orth_center=0)
    dense = format.contract_dense()
    result = (format.to(dtype=torch.complex128) if copy_method == 'to' else
              getattr(format, copy_method)())
    cores = result.cores
    cores[0] = 2 * cores[0]
    assert result.cores is cores and result._orth_center is None
    assert torch.allclose(result.contract_dense(), (2 * dense).to(result.dtype))
    if result is not format:
        assert format._orth_center == 0
        assert torch.equal(format.contract_dense(), dense)


@pytest.mark.parametrize('replacement', ['element', 'slice', 'full'])
def test_unexpected_core_validation_error_restores_metadata(make_format,
                                                           replacement,
                                                           monkeypatch):
    format = make_format('tt', 2).canonicalize_vidal('explicit')
    format._orth_center = 0
    cores, bonds = format.cores, format.bonds
    previous = tuple(cores)
    dimensions, rank = format.in_dim, format.rank
    same_in_dim = format._same_in_dim
    dense = format.contract_dense()
    validate = format.validate

    def fail_after_validation():
        validate()
        raise RuntimeError('Validation failed after refreshing metadata')

    monkeypatch.setattr(format, 'validate', fail_after_validation)
    value = torch.cat([cores[0], cores[0][:1]], dim=0)
    with pytest.raises(RuntimeError, match='refreshing metadata'):
        if replacement == 'element':
            cores[0] = value
        elif replacement == 'slice':
            cores[:1] = [value]
        else:
            format.cores = [value, *cores[1:]]
    assert format.cores is cores and format.bonds is bonds
    assert all(new is old for new, old in zip(cores, previous))
    assert format.in_dim == dimensions and format.rank == rank
    assert format._same_in_dim == same_in_dim
    assert format._orth_center == 0 and bonds._valid
    assert torch.equal(format.contract_dense(), dense)


# Norm distance


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_orth_center_is_read_only(make_format, topology):
    format = make_format(topology)
    assert format.orth_center is None
    format.canonicalize(orth_center=1)
    assert format.orth_center == 1
    with pytest.raises(AttributeError):
        format.orth_center = 0
    format.cores[0] = format.cores[0].clone()
    assert format.orth_center is None


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_distance_matches_dense_difference(make_format, topology, dtype):
    original = make_format(topology, dtype=dtype)
    changed = original.clone()
    changed.cores[1] = changed.cores[1] * 0.7
    expected = (original.contract_dense() - changed.contract_dense()).norm()
    assert torch.allclose(original.distance(changed), expected,
                          rtol=1e-10, atol=1e-12)
    assert torch.allclose(changed.distance(original), expected,
                          rtol=1e-10, atol=1e-12)
    assert original.distance(original) == 0


def test_distance_resolves_small_residual():
    original = tk.formats.TT([
        torch.eye(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64)])
    changed = original.clone()
    changed.cores[0] = changed.cores[0] + 1e-12 * torch.eye(2,
                                                            dtype=torch.float64)
    expected = (original.contract_dense() - changed.contract_dense()).norm()
    assert torch.allclose(original.distance(changed), expected,
                          rtol=1e-3, atol=1e-14)


def test_distance_resolves_structural_batches(make_format):
    original = make_format('tr', n_batches=2)
    changed = original.clone()
    changed.cores[0] = changed.cores[0] * 0.8
    expected = torch.linalg.vector_norm(
        (original.contract_dense() - changed.contract_dense()).flatten(2),
        dim=-1)
    actual = original.distance(changed)
    assert actual.shape == original.batch_shape
    assert torch.allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('topology,n_batches',
                         [('tt', 0), ('tr', 2), ('ttm', 0), ('trm', 2)])
def test_normalize_preserves_dense_tensor_up_to_scale(make_format, topology,
                                                      n_batches):
    format = make_format(topology, n_batches=n_batches,
                         dtype=torch.complex128)
    dense = format.contract_dense()
    norm = format.norm()
    expected = dense / norm.reshape(*format.batch_shape,
                                    *((1,) * (dense.ndim - n_batches)))
    assert format.normalize() is format
    assert torch.allclose(format.norm(), torch.ones_like(norm),
                          rtol=1e-10, atol=1e-12)
    assert torch.allclose(format.contract_dense(), expected,
                          rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse'])
def test_normalize_preserves_valid_vidal_gauge(make_format, mode):
    format = make_format('tt', 3).canonicalize_vidal(mode=mode)
    dense = format.contract_dense()
    norm = format.norm()
    assert format.bonds._valid
    format.normalize()
    assert format.bonds._valid
    assert torch.allclose(format.contract_dense(), dense / norm,
                          rtol=1e-10, atol=1e-12)
    assert torch.allclose(format.norm(), torch.ones_like(norm),
                          rtol=1e-10, atol=1e-12)
    assert all(torch.allclose(spectrum.norm(), torch.ones_like(norm))
               for spectrum in format.bonds.spectra)
    format.redistribute_vidal(0, mode='left')
    assert format.bonds._valid


def test_normalize_preserves_batched_mixed_vidal_gauge(make_format):
    format = make_format('tt', n_batches=2)
    format.canonicalize_vidal(mode='implicit')
    format.redistribute_vidal(0, mode='left')
    format.redistribute_vidal(1, mode='inverse')
    dense = format.contract_dense()
    norm = format.norm()
    format.normalize()
    expected = dense / norm[..., None, None, None]
    assert format.bonds._valid
    assert torch.allclose(format.contract_dense(), expected,
                          rtol=1e-10, atol=1e-12)
    assert torch.allclose(format.norm(), torch.ones_like(norm),
                          rtol=1e-10, atol=1e-12)
    assert all(torch.allclose(torch.linalg.vector_norm(spectrum, dim=-1),
                              torch.ones_like(norm))
               for spectrum in format.bonds.spectra)


def test_normalize_rejects_zero_without_mutation():
    format = tk.formats.TT([torch.zeros(2, 1), torch.ones(1, 2)])
    cores = tuple(format.cores)
    with pytest.raises(ValueError, match='zero-norm'):
        format.normalize()
    assert all(current is previous for current, previous in zip(format.cores,
                                                                  cores))


def test_normalize_uses_recorded_center():
    format = tk.formats.TT([
        torch.diag(torch.tensor([4., 1.], dtype=torch.float64)),
        torch.eye(2, dtype=torch.float64)])
    format.canonicalize(orth_center=0)
    other_core = format.cores[1]
    format.normalize()
    assert format.orth_center == 0
    assert format.cores[1] is other_core
    assert torch.allclose(format.cores[0].norm(), format.norm())


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_normalize_extreme_scales(scale):
    format = tk.formats.TT([
        torch.ones(2, 1, dtype=torch.float64) * scale,
        torch.ones(1, 2, dtype=torch.float64) * scale])
    format.normalize()
    assert torch.allclose(format.norm(), torch.tensor(1., dtype=torch.float64),
                          rtol=1e-10, atol=1e-12)


def test_normalize_preserves_autograd():
    first = torch.tensor([[2., 0.], [0., 1.]], dtype=torch.float64,
                         requires_grad=True)
    format = tk.formats.TT([first, torch.eye(2, dtype=torch.float64)])
    format.normalize()
    format.cores[0][0, 0].backward()
    assert first.grad is not None
    assert torch.all(torch.isfinite(first.grad))


# Operations


def _matrix(format):
    dense = format.contract_dense()
    n, b = format.n_sites, format.n_batches
    order = [*range(b), *range(b + 1, b + 2 * n, 2),
             *range(b, b + 2 * n, 2)]
    return dense.permute(order).reshape(*format.batch_shape,
                                       int(torch.tensor(format.out_dim).prod()),
                                       int(torch.tensor(format.in_dim).prod()))


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('method', ['stacked', 'block_diagonal'])
def test_exact_algebra(make_format, topology, n_sites, method):
    a = make_format(topology, n_sites, dtype=torch.complex128)
    b = a * (0.2 + 0.3j)
    dense_a, dense_b = a.contract_dense(), b.contract_dense()
    assert torch.allclose(a.add(b, method).contract_dense(), dense_a + dense_b)
    assert torch.allclose(a.sub(b, method).contract_dense(), dense_a - dense_b)
    assert torch.allclose((a * b).contract_dense(), dense_a * dense_b)
    assert torch.allclose((-a).contract_dense(), -dense_a)
    assert torch.equal(a.contract_dense(), dense_a)


@pytest.mark.parametrize('matrix_topology', ['ttm', 'trm'])
@pytest.mark.parametrize('vector_topology', ['tt', 'tr'])
@pytest.mark.parametrize('n_sites', [1, 2, 3])
def test_apply_and_matrix_products(make_format, matrix_topology, vector_topology, n_sites):
    a = make_format(matrix_topology, n_sites, dtype=torch.complex128)
    x = make_format(vector_topology, n_sites, dtype=torch.complex128)
    matrix, vector = _matrix(a), x.contract_dense().flatten()
    y = a @ x
    assert torch.allclose(y.contract_dense().flatten(), matrix @ vector)
    assert torch.allclose(a.apply(x).contract_dense(), y.contract_dense())
    assert y.topology == ('tr' if 'tr' in (matrix_topology, vector_topology) or
                         matrix_topology == 'trm' else 'tt')
    right = y.T @ a
    assert torch.allclose(right.contract_dense().flatten(), (matrix @ vector) @ matrix)
    assert torch.allclose(y.apply(a).contract_dense(), right.T.contract_dense())
    assert torch.allclose((y.H @ a).contract_dense().flatten(),
                          (matrix @ vector).conj() @ matrix)
    assert torch.allclose(_matrix(a @ a.H), matrix @ matrix.adjoint())
    assert torch.allclose(_matrix(a.T), matrix.T)
    assert torch.allclose(_matrix(a.H), matrix.adjoint())
    assert torch.allclose((a @ a.H).trace(), torch.trace(matrix @ matrix.adjoint()))
    assert torch.allclose(((a @ a.H) @ a).contract_dense(),
                          (a @ (a.H @ a)).contract_dense())


@pytest.mark.parametrize('left_topology', ['tt', 'tr'])
@pytest.mark.parametrize('right_topology', ['tt', 'tr'])
@pytest.mark.parametrize('n_sites', [1, 3])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_kets_rows_and_outer_products(make_format, left_topology,
                                     right_topology, n_sites, dtype):
    x = make_format(left_topology, n_sites, dtype=dtype)
    y = make_format(right_topology, n_sites, dtype=dtype) * (1 + 2j)
    x_dense = x.contract_dense().flatten()
    y_dense = y.contract_dense().flatten()
    x_dense = x_dense.to(y_dense.dtype)

    assert torch.allclose(x.T @ y, torch.dot(x_dense, y_dense))
    assert torch.allclose(x.H @ y, torch.vdot(x_dense, y_dense))
    assert type(x.T) is type(x)
    assert type(x.H) is type(x)
    assert torch.equal(x.T.T.contract_dense(), x.contract_dense())
    assert torch.equal(x.H.H.contract_dense(), x.contract_dense())
    assert torch.equal(x.T.contract_dense(), x.contract_dense())
    assert torch.equal(x.H.contract_dense(), x.contract_dense().conj())

    outer = x @ y.H
    assert torch.allclose(_matrix(outer), torch.outer(x_dense, y_dense.conj()))
    assert torch.allclose(_matrix(x @ y.T), torch.outer(x_dense, y_dense))
    cyclic = left_topology == 'tr' or right_topology == 'tr'
    assert isinstance(outer, tk.formats.TRM if cyclic else tk.formats.TTM)


def test_rectangular_outer_product_and_energy():
    x = tk.formats.TT([torch.tensor([1 + 2j, 3 - 1j], dtype=torch.complex128)])
    y = tk.formats.TT([torch.tensor([2 - 1j, 4j, 1], dtype=torch.complex128)])
    outer = x @ y.H
    assert outer.in_dim == (3,)
    assert outer.out_dim == (2,)
    assert torch.allclose(_matrix(outer),
                          torch.outer(x.contract_dense(), y.contract_dense().conj()))
    a = tk.formats.TTM([torch.diag(torch.tensor([2., 5.], dtype=torch.complex128))])
    dense = x.contract_dense()
    energy = (x.H @ a @ x) / (x.H @ x)
    expected = torch.vdot(dense, _matrix(a) @ dense) / torch.vdot(dense, dense)
    assert torch.allclose(energy, expected)

    for left, right in [(x, x), (x, a), (x.T, x.H), (a, x.H)]:
        with pytest.raises(TypeError):
            left @ right
    assert (x.H @ (x * 0)).item() == 0


@pytest.mark.parametrize('topology', ['tt', 'tr'])
def test_vector_rows_keep_format_operations_and_own_containers(make_format, topology):
    x = make_format(topology, dtype=torch.complex128)
    count = x.n_sites if x._cyclic else x.n_sites - 1
    x.bonds = [torch.ones(rank, dtype=torch.float64) for rank in x.rank[:count]]
    dense = x.contract_dense()
    row = x.H
    assert row.cores is not x.cores
    assert row.bonds is not x.bonds
    assert row.cores[0].data_ptr() == x.cores[0].data_ptr()
    assert torch.allclose(row.norm(), dense.norm())

    rows = [row.clone(), row.detach(), row.to(copy=True), row.conj(),
            row * 2, row + row, row * row,
            row.clone().canonicalize(orth_center=1), row.clone().rounding()]
    rows.append(row.to_tt() if x._cyclic else row.clone().canonicalize_minimal())
    for transformed in rows:
        assert torch.allclose(transformed @ x,
                              torch.dot(transformed.contract_dense().flatten(), dense.flatten()))
    with pytest.raises(ValueError, match='orientation'):
        x + row
    with pytest.raises(ValueError, match='orientation'):
        x * row

    with pytest.raises(ValueError):
        row.cores[0] = torch.ones(1, dtype=x.dtype)
    with pytest.raises(ValueError, match='factor dimensions'):
        row.bonds.factors[0] = torch.ones(1)
    assert torch.allclose(row.contract_dense(), dense.conj())

    row.cores[0] = row.cores[0] * 2
    row.bonds.factors[0] = row.bonds.factors[0] * 3
    assert torch.allclose(row.contract_dense(), 6 * dense.conj())
    assert torch.equal(x.contract_dense(), dense)


def test_outer_product_structural_batches(make_format):
    x = make_format('tr', n_batches=1)
    y = make_format('tt')
    outer = x @ y.H
    dense_x, dense_y = x.contract_dense().reshape(2, -1), y.contract_dense().flatten()
    assert outer.n_batches == 1
    assert torch.allclose(_matrix(outer), dense_x.unsqueeze(-1) * dense_y.unsqueeze(0))
    x = make_format('tt', n_batches=1)
    outer = x @ y.T
    dense_x = x.contract_dense().reshape(2, -1)
    assert outer.topology == 'ttm'
    assert outer.n_batches == 1
    assert torch.allclose(_matrix(outer), dense_x.unsqueeze(-1) * dense_y.unsqueeze(0))


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 3])
def test_core_views_and_factored_products(make_format, topology, n_sites):
    format = make_format(topology, n_sites, dtype=torch.complex128)
    count = n_sites if format._cyclic else n_sites - 1
    format.bonds = [torch.full((format.rank[site],), 2 + 1j) for site in range(count)]
    standard, effective, operator = (
        format._standard_cores(), format._effective_cores(), format._operator_cores())
    for site in range(n_sites):
        expected = standard[site] * (2 + 1j) if site < count else standard[site]
        assert torch.equal(effective[site], expected)
        assert operator[site].shape[-4] == standard[site].shape[-3]
        assert operator[site].shape[-2] == standard[site].shape[-1]
    if topology.endswith('m'):
        assert torch.allclose(_matrix(format @ format.H),
                              _matrix(format) @ _matrix(format).adjoint())
    else:
        dense = format.contract_dense().flatten()
        assert torch.allclose(_matrix(format @ format.H), torch.outer(dense, dense.conj()))


def test_mixed_sum_batches_and_errors(make_format):
    a, b = make_format('tt'), make_format('tr', n_batches=1)
    assert torch.allclose((a + b).contract_dense(), a.contract_dense() + b.contract_dense())
    with pytest.raises(ValueError):
        a.add(b, method='unknown')
    with pytest.raises(ValueError):
        a + make_format('ttm')
    with pytest.raises(TypeError):
        a @ a
    with pytest.raises(ValueError):
        a * torch.ones(2)
    with pytest.raises(ValueError):
        make_format('ttm').trace()


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_factored_operations(make_format, topology):
    a = make_format(topology, dtype=torch.complex128)
    count = a.n_sites if topology.startswith('tr') else a.n_sites - 1
    a.bonds = [
        torch.linspace(1, 2, a.rank[site], dtype=torch.float64) * (1 + 0.1j)
        for site in range(count)]
    dense = a.contract_dense()
    assert torch.allclose(a.norm(), dense.norm())
    assert torch.allclose((a * a).contract_dense(), dense.square())
    assert torch.allclose(a.conj().contract_dense(), dense.conj())
    assert torch.allclose(a.clone().materialize_bonds(orth_center=1).contract_dense(), dense)
    copied = a.clone()
    assert copied.bonds.factors[0].data_ptr() != a.bonds.factors[0].data_ptr()
    with pytest.raises(ValueError, match='factor dimensions'):
        a.bonds.factors[0] = torch.ones(1)
    assert torch.allclose(a.contract_dense(), dense)


# Rounding


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('method', ['svd', 'qr_svd'])
def test_exact_rounding(make_format, topology, n_sites, method):
    format = make_format(topology, n_sites, dtype=torch.complex128)
    dense = format.contract_dense()
    with tk.svd_method(method, refine=True):
        result, info = format.rounding(return_info=True)
    assert result is format
    assert torch.allclose(format.contract_dense(), dense, rtol=1e-10, atol=1e-12)
    assert info.rank == tuple(format.rank)
    assert torch.all(info.error_bound == 0)


def test_known_spectrum_and_budget():
    diagonal = torch.diag(torch.tensor([4., 2., 1., 0.1], dtype=torch.float64))
    format = tk.formats.TT([diagonal, torch.eye(4, dtype=diagonal.dtype)])
    expected = diagonal.clone()
    expected[2:, 2:] = 0
    format.rounding(rank=2)
    assert torch.allclose(format.contract_dense(), expected)
    full = tk.formats.TT([diagonal, torch.eye(4, dtype=diagonal.dtype)])
    _, info = full.rounding(rel_error=0.03, return_info=True)
    assert info.bound_satisfied
    assert (full.contract_dense() - diagonal).norm() <= 0.03 * diagonal.norm()
    full = tk.formats.TT([diagonal, torch.eye(4, dtype=diagonal.dtype)])
    with pytest.warns(UserWarning, match='budget'):
        _, info = full.rounding(rank=1, rel_error=1e-5, return_info=True)
    assert not info.bound_satisfied


def test_stacked_cyclic_sum_compresses(make_format):
    format = make_format('tr', 4)
    summed = format + format
    _, info = summed.rounding(rel_error=1e-12, return_info=True)
    assert info.bound_satisfied
    assert torch.allclose(summed.contract_dense(), 2 * format.contract_dense())
    assert all(actual <= original for actual, original in zip(summed.rank, format.rank))
    usual = format.add(format, method='block_diagonal')
    usual.rounding(rel_error=1e-12)
    assert torch.allclose(usual.contract_dense(), 2 * format.contract_dense())
    assert usual.rank[-1] > summed.rank[-1]


@pytest.mark.parametrize('scale', [1e-200, 1e200])
def test_rounding_extreme_scales(scale):
    diagonal = torch.diag(torch.tensor([4., 2., 1., 0.1], dtype=torch.float64)) * scale
    format = tk.formats.TT([diagonal, torch.eye(4, dtype=diagonal.dtype)])
    _, info = format.rounding(rtol=0.001, rel_error=0.03, return_info=True)
    assert format.rank == [3]
    assert info.bound_satisfied
    assert torch.isfinite(info.error_bound)
    assert torch.allclose(format.contract_dense() / scale,
                          torch.diag(torch.tensor([4., 2., 1., 0.], dtype=diagonal.dtype)))


@pytest.mark.parametrize('kwargs', [{'rank': 0}, {'rtol': -1}, {'rel_error': -1},
                                   {'cutoff': float('nan')}, {'return_info': 1}])
def test_invalid_even_one_site(kwargs):
    with pytest.raises((TypeError, ValueError)):
        tk.formats.TT([torch.ones(2)]).rounding(**kwargs)


# Structural batches


@pytest.mark.parametrize('n_sites', [1, 2, 4])
@pytest.mark.parametrize('n_batches', [1, 2])
def test_batched_ttm_operations(make_format, n_sites, n_batches):
    format = make_format('ttm', n_sites, n_batches, torch.complex128)
    dense = format.contract_dense()
    inputs = torch.zeros(3, n_sites, dtype=torch.long)
    outputs = torch.ones_like(inputs)
    values = format.evaluate(inputs, outputs)
    applied = format.apply(inputs).contract_dense()
    norm = format.norm()
    adjoint = format.H.contract_dense()
    doubled = format.add(format).contract_dense()
    squared = format.hadamard(format).contract_dense()
    gram = (format.H @ format).contract_dense()

    for index in product(*[range(size) for size in format.batch_shape]):
        member = tk.formats.TTM([core[index] for core in format.cores])
        assert torch.allclose(dense[index], member.contract_dense())
        assert torch.allclose(values[index], member.evaluate(inputs, outputs))
        assert torch.allclose(applied[index], member.apply(inputs).contract_dense())
        assert torch.allclose(norm[index], member.norm())
        assert torch.allclose(adjoint[index], member.H.contract_dense())
        assert torch.allclose(doubled[index], 2 * member.contract_dense())
        assert torch.allclose(squared[index], member.contract_dense().square())
        assert torch.allclose(gram[index], (member.H @ member).contract_dense())

    canonical = format.clone().canonicalize()
    minimal = format.clone().canonicalize_minimal(method='gradient')
    rounded = format.clone().rounding()
    for result in (canonical, minimal, rounded):
        assert result.batch_shape == format.batch_shape
        assert torch.allclose(result.contract_dense(), dense, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('site', [0, 1, 2])
def test_batched_ttm_rejects_malformed_cores(make_format, site):
    cores = list(make_format('ttm', 3, n_batches=1).cores)
    cores[site] = cores[site].unsqueeze(-1)
    with pytest.raises(ValueError, match='dimensions'):
        tk.formats.TTM(cores, n_batches=1)


def test_batched_ttm_retains_autograd(make_format):
    format = make_format('ttm', 3, n_batches=1)
    original = list(format.cores)
    for core in original:
        core.requires_grad_()
    format.clone().canonicalize().contract_dense().square().sum().backward()
    assert all(core.grad is not None and torch.isfinite(core.grad).all()
               for core in original)
