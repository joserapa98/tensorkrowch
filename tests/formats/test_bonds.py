"""Owned bond factors, Vidal gauges and validated replacements."""

import pytest
import torch

import tensorkrowch as tk


# Bonds


@pytest.mark.parametrize('topology', ['tt', 'ttm'])
@pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse', 'left', 'right'])
def test_vidal_and_mixed_centers(make_format, topology, mode):
    format = make_format(topology, 3, dtype=torch.complex128)
    dense = format.contract_dense()
    format.canonicalize_vidal(mode)
    assert isinstance(format.bonds, tk.formats.VidalGauge)
    assert torch.allclose(format.contract_dense(), dense)
    for orth_center in range(format.n_sites):
        mixed = format.clone().materialize_bonds(orth_center)
        assert mixed.bonds is None
        assert torch.allclose(mixed.contract_dense(), dense)
        for site, core in enumerate(mixed._effective_cores()):
            if site < orth_center:
                matrix = core.reshape(-1, core.shape[-1])
                assert torch.allclose(matrix.adjoint() @ matrix,
                                      torch.eye(matrix.shape[-1], dtype=matrix.dtype),
                                      atol=1e-10, rtol=1e-10)
            elif site > orth_center:
                matrix = core.reshape(core.shape[-3], -1)
                assert torch.allclose(matrix @ matrix.adjoint(),
                                      torch.eye(matrix.shape[-2], dtype=matrix.dtype),
                                      atol=1e-10, rtol=1e-10)
    format.canonicalize_vidal('implicit').canonicalize_vidal('explicit')
    assert torch.allclose(format.contract_dense(), dense)


@pytest.mark.parametrize('remaining_mode,powers', [
    ('explicit', (0, 0)), ('left', (1, 0)), ('right', (0, 1)),
])
def test_mixed_inverse_bonds_with_directional_modes(make_format, remaining_mode, powers):
    format = make_format('tt', 4)
    dense = format.contract_dense()
    format.canonicalize_vidal(inverse_positions=[1], remaining_mode=remaining_mode)
    assert format.bonds.powers == [powers, (1, 1), powers]
    assert torch.allclose(format.materialize_bonds(orth_center=2).contract_dense(), dense)


def test_mixed_inverse_bonds_and_zero(make_format):
    format = make_format('tt', 4)
    dense = format.contract_dense()
    format.canonicalize_vidal(inverse_positions=[1], remaining_mode='explicit')
    assert format.bonds.powers == [(0, 0), (1, 1), (0, 0)]
    assert torch.allclose(format.materialize_bonds(orth_center=2).contract_dense(), dense)
    zero = tk.formats.TT([torch.zeros(2, 2), torch.ones(2, 3)])
    for mode in ['explicit', 'implicit']:
        result = zero.clone().canonicalize_vidal(mode)
        assert torch.isfinite(result.contract_dense()).all()
        assert result.norm() == 0
    with pytest.raises(ValueError, match='Inverse Vidal bond'):
        zero.canonicalize_vidal('inverse')
    assert zero.bonds is None


def test_vidal_validation(make_format):
    assert not hasattr(make_format('tr'), 'canonicalize_vidal')
    for kwargs in [{'mode': 'invalid'}, {'inverse_positions': [0, 0]},
                   {'inverse_positions': [3]}, {'inverse_cutoff': -1}]:
        with pytest.raises(ValueError):
            make_format().canonicalize_vidal(**kwargs)


def test_local_redistribution_retains_other_interfaces(make_format):
    format = make_format().canonicalize_vidal('implicit')
    expected = format.contract_dense()
    second = format.bonds.powers[1]
    for mode, powers in [('inverse', (1, 1)), ('explicit', (0, 0)),
                         ('left', (1, 0)), ('right', (0, 1))]:
        format.redistribute_vidal(0, mode)
        assert format.bonds.powers == [powers, second]
        assert torch.allclose(format.contract_dense(), expected)
    with pytest.raises(ValueError):
        format.redistribute_vidal(0, 'unknown')
    with pytest.raises(ValueError):
        format.redistribute_vidal(0, 'inverse', inverse_cutoff=1e20)


# Mutations


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('invalid', ['rank', 'dtype', 'device', 'type', 'length'])
def test_invalid_bonds_restore_state(make_format, topology, invalid):
    format = make_format(topology, 2)
    format.bonds = [
        torch.ones(rank, dtype=format.dtype) for rank in format.rank]
    format._orth_center = 0
    bonds = format.bonds
    factors = bonds.factors
    previous = tuple(factors)
    dense = format.contract_dense()
    invalid_factor = {
        'rank': torch.ones(format.rank[0] + 1, dtype=format.dtype),
        'dtype': factors[0].to(torch.complex128),
        'device': torch.empty(format.rank[0], dtype=format.dtype, device='meta'),
        'type': 1,
        'length': None
    }[invalid]

    with pytest.raises(TypeError if invalid == 'type' else ValueError):
        if invalid == 'length':
            bonds.factors = []
        else:
            factors[0] = invalid_factor

    assert format.bonds is bonds and bonds.factors is factors
    assert all(new is old for new, old in zip(factors, previous))
    assert format._orth_center == 0
    assert torch.allclose(format.contract_dense(), dense)


@pytest.mark.parametrize('topology,n_batches', [('tt', 2), ('tr', 2),
                                               ('ttm', 0), ('trm', 2)])
def test_bond_slices_and_batch_validation(make_format, topology, n_batches):
    format = make_format(topology, 4, n_batches=n_batches)
    format.bonds = [None] * len(format.rank)
    dense = format.contract_dense()
    factors = format.bonds.factors
    factors[:2] = [torch.full((*format.batch_shape, format.rank[0]), 2.,
                            dtype=format.dtype),
                  torch.full((format.rank[1],), 3., dtype=format.dtype)]
    assert format.bonds.factors is factors
    assert torch.allclose(format.contract_dense(), 6 * dense)
    with pytest.raises(ValueError):
        factors[:2] = [factors[0]]
    with pytest.raises(ValueError):
        factors[0] = torch.ones(3, format.rank[0], dtype=format.dtype)
    assert torch.allclose(format.contract_dense(), 6 * dense)
    factors[::2] = [None] * len(factors[::2])
    assert torch.allclose(format.contract_dense(), 3 * dense)
    format.bonds.factors = [None] * len(format.rank)
    assert torch.allclose(format.contract_dense(), dense)


@pytest.mark.parametrize('operation', ['append', 'pop', 'clear', 'reverse',
                                      'sort', 'delete', 'extend', 'iadd'])
def test_bond_structure_changes_are_rejected(make_format, operation):
    format = make_format()
    format.bonds = [None] * len(format.rank)
    factors = format.bonds.factors
    with pytest.raises(TypeError):
        if operation == 'delete':
            del factors[0]
        elif operation == 'append':
            factors.append(None)
        elif operation == 'extend':
            factors.extend([None])
        elif operation == 'iadd':
            factors += [None]
        else:
            getattr(factors, operation)()
    assert len(factors) == len(format.rank)


@pytest.mark.parametrize('copy_method', ['clone', 'detach', 'detach_', 'to', 'T', 'H'])
def test_bond_callbacks_are_independent(make_format, copy_method):
    format = make_format('ttm', 3).canonicalize_vidal('explicit')
    original = format.contract_dense()
    shared = format.bonds
    peer = make_format('ttm', 3)
    peer.bonds = shared.factors
    assert peer.bonds is not shared
    assert peer.bonds.factors[0] is shared.factors[0]

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
    result.bonds.factors[0] = None
    assert not result.bonds._valid
    assert result._orth_center is None
    assert peer.bonds.factors[0] is shared.factors[0]
    assert peer._orth_center == 0
    if result is not format:
        assert format.bonds is shared and shared._valid
        assert format._orth_center == 0
        assert torch.allclose(format.contract_dense(), original)


@pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse'])
@pytest.mark.parametrize('replacement', ['core', 'cores', 'factor', 'factor_slice',
                                        'all_factors', 'spectrum', 'power'])
def test_manual_replacement_invalidates_vidal(make_format, mode, replacement):
    format = make_format('tt', 3).canonicalize_vidal(mode)
    if replacement == 'core':
        format.cores[0] = 2 * format.cores[0]
    elif replacement == 'cores':
        format.cores = [2 * format.cores[0], *format.cores[1:]]
    elif replacement == 'factor':
        format.bonds.factors[0] = None
    elif replacement == 'factor_slice':
        format.bonds.factors[:1] = [None]
    elif replacement == 'all_factors':
        format.bonds.factors = [None] * len(format.rank)
    elif replacement == 'spectrum':
        format.bonds.spectra[0] = torch.ones(1)
    else:
        format.bonds.powers[0] = (0, 0)
    assert not format.bonds._valid
    with pytest.raises(ValueError, match='valid stored Vidal'):
        format.redistribute_vidal(0)
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
        bonds.factors[0] = torch.ones(100)
    assert format.bonds is bonds and bonds._valid
    assert format.cores is cores


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


@pytest.mark.parametrize('replacement', ['element', 'slice', 'full'])
def test_unexpected_bond_validation_error_restores_state(make_format,
                                                        replacement,
                                                        monkeypatch):
    format = make_format().canonicalize_vidal('explicit')
    format._orth_center = 0
    bonds = format.bonds
    factors = bonds.factors
    previous = tuple(factors)
    dense = format.contract_dense()
    validate = format.validate_bonds

    def fail_after_validation():
        validate()
        raise RuntimeError('Validation failed after checking factors')

    monkeypatch.setattr(format, 'validate_bonds', fail_after_validation)
    with pytest.raises(RuntimeError, match='checking factors'):
        if replacement == 'element':
            factors[0] = 2 * factors[0]
        elif replacement == 'slice':
            factors[:1] = [2 * factors[0]]
        else:
            bonds.factors = [2 * factors[0], *factors[1:]]
    assert format.bonds is bonds and bonds.factors is factors
    assert all(new is old for new, old in zip(factors, previous))
    assert format._orth_center == 0 and bonds._valid
    assert torch.equal(format.contract_dense(), dense)


def test_owned_bonds_validate_replacements():
    first = torch.ones(2)
    format = tk.formats.TT([torch.ones(2, 2), torch.ones(2, 2, 2),
                            torch.ones(2, 2)], bonds=[first, None])
    bonds = format.bonds
    factors = bonds.factors
    factors[1] = torch.ones(2)
    previous = tuple(factors)
    with pytest.raises(TypeError, match='tensors or None'):
        factors[::-1] = [None, object()]
    with pytest.raises(TypeError, match='tensors or None'):
        bonds.factors = [1, None]
    assert bonds.factors is factors
    assert all(new is old for new, old in zip(factors, previous))
    factors[:] = (factor for factor in [None, first])
    assert bonds.factors is factors and factors[0] is None and factors[1] is first


@pytest.mark.parametrize('topology', ['tt', 'tr', 'ttm', 'trm'])
@pytest.mark.parametrize('n_sites', [1, 3])
def test_constructor_owns_bonds(make_format, topology, n_sites):
    source = make_format(topology, n_sites)
    factors = [torch.ones(rank, dtype=source.dtype) for rank in source.rank]
    format = type(source)(source.cores, bonds=factors)
    assert format.bonds.factors is not factors
    assert all(new is old for new, old in zip(format.bonds.factors, factors))
    assert torch.allclose(format.contract_dense(), source.contract_dense())
    with pytest.raises(ValueError, match='one factor per bond'):
        type(source)(source.cores, bonds=[*factors, None])
    with pytest.raises(TypeError):
        type(source)(source.cores, bonds=torch.ones(2))
    if factors:
        previous = format.bonds.factors[0]
        with pytest.raises(ValueError, match='dimensions'):
            format.bonds.factors[0] = torch.ones(100, dtype=source.dtype)
        assert format.bonds.factors[0] is previous


@pytest.mark.parametrize('vidal', [False, True])
def test_complex_conversion_preserves_factors_and_real_spectra(make_format, vidal):
    format = make_format('tt', 3)
    if vidal:
        format.canonicalize_vidal('explicit')
    else:
        format.bonds = [torch.ones(rank, dtype=format.dtype) for rank in format.rank]
    result = format.to(dtype=torch.complex128)
    assert all(factor.dtype == torch.complex128 for factor in result.bonds.factors)
    assert torch.allclose(result.contract_dense(), format.contract_dense().to(torch.complex128))
    if vidal:
        assert all(spectrum.dtype == torch.float64 for spectrum in result.bonds.spectra)
        assert result.bonds._valid
    result.bonds.factors[0] = result.bonds.factors[0] * 1j
    assert torch.allclose(result.contract_dense(), 1j * format.contract_dense())


def test_copy_invalid_vidal_preserves_stored_factors(make_format):
    format = make_format('tt', 3).canonicalize_vidal('inverse')
    format.bonds.factors[0] = 2 * format.bonds.factors[0]
    result = format.clone()
    assert not result.bonds._valid
    assert torch.allclose(result.contract_dense(), format.contract_dense())
    result.materialize_bonds()
    assert torch.allclose(result.contract_dense(), format.contract_dense())
