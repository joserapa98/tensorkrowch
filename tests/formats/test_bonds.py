"""Vidal gauges, inverse support and redistribution to mixed forms."""

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('topology', ['tt', 'ttm'])
@pytest.mark.parametrize('mode', ['explicit', 'implicit', 'inverse'])
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
        for site, core in enumerate(mixed._standard_cores()):
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
    with pytest.raises(ValueError, match='open chain'):
        make_format('tr').canonicalize_vidal()
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
        format.redistribute_bond(0, mode)
        assert format.bonds.powers == [powers, second]
        assert torch.allclose(format.contract_dense(), expected)
    with pytest.raises(ValueError):
        format.redistribute_bond(0, 'unknown')
    with pytest.raises(ValueError):
        format.redistribute_bond(0, 'inverse', inverse_cutoff=1e20)
