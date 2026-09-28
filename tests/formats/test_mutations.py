"""Immediate mutation checks, rollback and canonical state ownership."""

import pytest
import torch
import tensorkrowch as tk


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
@pytest.mark.parametrize('invalid', ['rank', 'dtype', 'device', 'type', 'length'])
def test_invalid_bonds_restore_state(make_format, topology, invalid):
    format = make_format(topology, 2)
    format.bonds = tk.formats.BondFactors1D([
        torch.ones(rank, dtype=format.dtype) for rank in format.rank])
    format._orth_center = 0
    bonds = format.bonds
    values = bonds.values
    previous = tuple(values)
    dense = format.contract_dense()
    invalid_value = {
        'rank': torch.ones(format.rank[0] + 1, dtype=format.dtype),
        'dtype': values[0].to(torch.complex128),
        'device': torch.empty(format.rank[0], dtype=format.dtype, device='meta'),
        'type': 1,
        'length': None
    }[invalid]

    with pytest.raises(TypeError if invalid == 'type' else ValueError):
        if invalid == 'length':
            bonds.values = []
        else:
            values[0] = invalid_value

    assert format.bonds is bonds and bonds.values is values
    assert all(new is old for new, old in zip(values, previous))
    assert format._orth_center == 0
    assert torch.allclose(format.contract_dense(), dense)


@pytest.mark.parametrize('topology,n_batches', [('tt', 2), ('tr', 2),
                                               ('ttm', 0), ('trm', 2)])
def test_bond_slices_and_batch_validation(make_format, topology, n_batches):
    format = make_format(topology, 4, n_batches=n_batches)
    format.bonds = tk.formats.BondFactors1D([None] * len(format.rank))
    dense = format.contract_dense()
    values = format.bonds.values
    values[:2] = [torch.full((*format.batch_shape, format.rank[0]), 2.,
                            dtype=format.dtype),
                  torch.full((format.rank[1],), 3., dtype=format.dtype)]
    assert format.bonds.values is values
    assert torch.allclose(format.contract_dense(), 6 * dense)
    with pytest.raises(ValueError):
        values[:2] = [values[0]]
    with pytest.raises(ValueError):
        values[0] = torch.ones(3, format.rank[0], dtype=format.dtype)
    assert torch.allclose(format.contract_dense(), 6 * dense)
    values[::2] = [None] * len(values[::2])
    assert torch.allclose(format.contract_dense(), 3 * dense)
    format.bonds.values = [None] * len(format.rank)
    assert torch.allclose(format.contract_dense(), dense)


@pytest.mark.parametrize('operation', ['append', 'pop', 'clear', 'reverse',
                                      'sort', 'delete', 'extend', 'iadd'])
def test_bond_structure_changes_are_rejected(make_format, operation):
    format = make_format()
    format.bonds = tk.formats.BondFactors1D([None] * len(format.rank))
    values = format.bonds.values
    with pytest.raises(TypeError):
        if operation == 'delete':
            del values[0]
        elif operation == 'append':
            values.append(None)
        elif operation == 'extend':
            values.extend([None])
        elif operation == 'iadd':
            values += [None]
        else:
            getattr(values, operation)()
    assert len(values) == len(format.rank)


@pytest.mark.parametrize('copy_method', ['clone', 'detach', 'detach_', 'to', 'T', 'H'])
def test_bond_callbacks_are_independent(make_format, copy_method):
    format = make_format('ttm', 3).canonicalize_vidal('explicit')
    original = format.contract_dense()
    shared = format.bonds
    peer = make_format('ttm', 3)
    peer.bonds = shared
    assert peer.bonds is not shared
    assert peer.bonds.values[0] is shared.values[0]

    if copy_method == 'to':
        result = format.to(dtype=torch.complex128)
    elif copy_method in ('T', 'H'):
        result = getattr(format, copy_method)
    else:
        result = getattr(format, copy_method)()
    result._orth_center = 0
    peer._orth_center = 0
    if result is not format:
        format._orth_center = 0
    result.bonds.values[0] = None
    assert not result.bonds._valid
    assert result._orth_center is None
    assert peer.bonds._valid
    assert peer._orth_center == 0
    if result is not format:
        assert format.bonds is shared and shared._valid
        assert format._orth_center == 0
        assert torch.allclose(format.contract_dense(), original)


@pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse'])
@pytest.mark.parametrize('replacement', ['core', 'cores', 'values', 'value_slice',
                                        'all_values', 'spectrum', 'power'])
def test_manual_replacement_invalidates_vidal(make_format, mode, replacement):
    format = make_format('tt', 3).canonicalize_vidal(mode)
    if replacement == 'core':
        format.cores[0] = 2 * format.cores[0]
    elif replacement == 'cores':
        format.cores = [2 * format.cores[0], *format.cores[1:]]
    elif replacement == 'values':
        format.bonds.values[0] = None
    elif replacement == 'value_slice':
        format.bonds.values[:1] = [None]
    elif replacement == 'all_values':
        format.bonds.values = [None] * len(format.rank)
    elif replacement == 'spectrum':
        format.bonds.spectra[0] = torch.ones(1)
    else:
        format.bonds.powers[0] = (0, 0)
    assert not format.bonds._valid
    with pytest.raises(ValueError, match='valid stored Vidal'):
        format.redistribute_bond(0)
    dense = format.contract_dense()
    absorbed = format.clone().absorb_bond(0)
    assert torch.allclose(absorbed.contract_dense(), dense)
    assert torch.allclose(format.clone().materialize_bonds().contract_dense(), dense)
    format.canonicalize_vidal('explicit')
    assert format.bonds._valid
    assert torch.allclose(format.contract_dense(), dense)


def test_failed_replacement_preserves_vidal(make_format):
    format = make_format().canonicalize_vidal('explicit')
    bonds = format.bonds
    cores = format.cores
    with pytest.raises(ValueError):
        format.cores = [core.to(torch.complex128) for core in cores[:1]] + cores[1:]
    with pytest.raises(ValueError):
        bonds.values[0] = torch.ones(100)
    assert format.bonds is bonds and bonds._valid
    assert format.cores is cores


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
def test_read_operations_do_not_validate(make_format, topology, monkeypatch):
    format = make_format(topology)
    format.bonds = tk.formats.BondFactors1D([None] * len(format.rank))
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
    format.bonds = tk.formats.BondFactors1D([
        torch.ones(original_rank[0], dtype=format.dtype)])
    cores, bonds = format.cores, format.bonds
    previous = tuple(cores)
    with pytest.raises(ValueError, match='factor dimensions'):
        cores[:] = [torch.ones(2, 5, dtype=format.dtype),
                    torch.ones(5, 3, dtype=format.dtype)]
    assert format.cores is cores and format.bonds is bonds
    assert all(new is old for new, old in zip(cores, previous))
    assert format.rank == original_rank


def test_quantics_replacement_restores_layout_and_metadata():
    layout = tk.formats.QuantizedLayout(1, 2, 2)
    format = tk.formats.QTT([torch.ones(2, 3), torch.ones(3, 2)], layout)
    cores = format.cores
    with pytest.raises(ValueError, match='Digit core dimensions'):
        cores[0] = torch.ones(4, 3)
    with pytest.raises(ValueError, match='Digit core dimensions'):
        format.cores = [torch.ones(2)]
    assert format.cores is cores
    assert format.in_dim == layout.in_dim and format.rank == [3]

    matrix = tk.formats.QTTM([torch.ones(2, 3, 2), torch.ones(3, 2, 2)],
                            layout, layout)
    previous = matrix.cores[0]
    with pytest.raises(ValueError, match='paired digit layouts'):
        matrix.cores[0] = torch.ones(4, 3, 2)
    assert matrix.cores[0] is previous
    assert matrix.in_dim == matrix.out_dim == layout.in_dim


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('step', [2, -2])
@pytest.mark.parametrize('container', ['cores', 'bonds'])
def test_extended_slice_generators_preserve_identity(make_format, topology,
                                                     step, container):
    format = make_format(topology, 4).canonicalize(orth_center=0)
    if container == 'bonds':
        format.bonds = tk.formats.BondFactors1D([
            torch.ones(rank, dtype=format.dtype) for rank in format.rank])
    format._orth_center = 0
    values = format.cores if container == 'cores' else format.bonds.values
    previous = tuple(values)
    dense = format.contract_dense()
    key = slice(None, None, step)
    indices = range(len(values))[key]
    values[key] = (2 * value for value in values[key])

    stored = format.cores if container == 'cores' else format.bonds.values
    assert stored is values
    assert format._orth_center is None
    for index, value in enumerate(values):
        if index in indices:
            assert torch.equal(value, 2 * previous[index])
        else:
            assert value is previous[index]
    assert torch.allclose(format.contract_dense(), 2 ** len(indices) * dense)


@pytest.mark.parametrize('sequence', ['spectra', 'powers'])
def test_vidal_extended_slices_invalidate_only_the_certificate(make_format,
                                                             sequence):
    format = make_format('tt', 4).canonicalize_vidal('explicit')
    format._orth_center = 0
    values = getattr(format.bonds, sequence)
    previous = tuple(values)
    dense = format.contract_dense()
    values[::-2] = (value for value in values[::-2])
    assert getattr(format.bonds, sequence) is values
    assert all(new is old for new, old in zip(values, previous))
    assert not format.bonds._valid and format._orth_center is None
    assert torch.equal(format.contract_dense(), dense)


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


@pytest.mark.parametrize('replacement', ['element', 'slice', 'full'])
def test_unexpected_bond_validation_error_restores_state(make_format,
                                                        replacement,
                                                        monkeypatch):
    format = make_format().canonicalize_vidal('explicit')
    format._orth_center = 0
    bonds = format.bonds
    values = bonds.values
    previous = tuple(values)
    dense = format.contract_dense()
    validate = format.validate_bonds

    def fail_after_validation():
        validate()
        raise RuntimeError('Validation failed after checking factors')

    monkeypatch.setattr(format, 'validate_bonds', fail_after_validation)
    with pytest.raises(RuntimeError, match='checking factors'):
        if replacement == 'element':
            values[0] = 2 * values[0]
        elif replacement == 'slice':
            values[:1] = [2 * values[0]]
        else:
            bonds.values = [2 * values[0], *values[1:]]
    assert format.bonds is bonds and bonds.values is values
    assert all(new is old for new, old in zip(values, previous))
    assert format._orth_center == 0 and bonds._valid
    assert torch.equal(format.contract_dense(), dense)


def test_standalone_bonds_validate_replacements():
    first = torch.ones(2)
    factors = tk.formats.BondFactors1D([first, None])
    values = factors.values
    values[1] = torch.ones(3)
    previous = tuple(values)
    with pytest.raises(TypeError, match='tensors or None'):
        values[::-1] = [None, object()]
    with pytest.raises(TypeError, match='tensors or None'):
        factors.values = [1, None]
    assert factors.values is values
    assert all(new is old for new, old in zip(values, previous))
    values[:] = (value for value in [None, first])
    assert factors.values is values and values[0] is None and values[1] is first
