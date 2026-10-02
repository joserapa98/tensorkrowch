"""Public center, absolute distance and unit-norm format scaling."""

import pytest
import torch
import tensorkrowch as tk


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
