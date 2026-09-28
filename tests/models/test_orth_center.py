"""Public orthogonality-center naming and deprecated keyword compatibility."""

import warnings

import pytest
import torch
import tensorkrowch as tk


@pytest.mark.parametrize('model_class, options', [
    (tk.models.MPS, {'phys_dim': 2}),
    (tk.models.MPSLayer, {'in_dim': 2, 'out_dim': 2}),
    (tk.models.MPO, {'in_dim': 2, 'out_dim': 2}),
])
@pytest.mark.parametrize('boundary', ['obc', 'pbc'])
@pytest.mark.parametrize('mode', ['svd', 'qr'])
@pytest.mark.parametrize('center', [None, 0, 1, 2])
def test_orth_center_keyword_alias_and_positional(model_class, options,
                                                  boundary, mode, center):
    original = model_class(n_sites=3, bond_dim=2, boundary=boundary,
                           dtype=torch.float64, **options)
    current, legacy, positional = (original.copy() for _ in range(3))
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always', DeprecationWarning)
        current.canonicalize(orth_center=center, mode=mode)
        positional.canonicalize(center, mode)
    assert not any('`oc` is deprecated' in str(item.message) for item in emitted)

    with pytest.warns(DeprecationWarning, match='`oc` is deprecated') as emitted:
        legacy.canonicalize(oc=center, mode=mode)
    assert emitted[0].filename == __file__
    for expected, old, by_position in zip(current.tensors, legacy.tensors,
                                           positional.tensors):
        assert torch.allclose(expected, old)
        assert torch.allclose(expected, by_position)


@pytest.mark.parametrize('model_class, options', [
    (tk.models.MPS, {'phys_dim': 2}),
    (tk.models.MPSLayer, {'in_dim': 2, 'out_dim': 2}),
    (tk.models.MPO, {'in_dim': 2, 'out_dim': 2}),
    (tk.models.UMPS, {'phys_dim': 2}),
    (tk.models.UMPO, {'in_dim': 2, 'out_dim': 2}),
    (tk.models.UMPSLayer, {'in_dim': 2, 'out_dim': 2}),
])
def test_orth_center_conflicting_names(model_class, options):
    model = model_class(n_sites=3, bond_dim=2, **options)
    before = tuple(model.mats_env)
    with pytest.raises(TypeError, match='cannot both be provided'):
        model.canonicalize(orth_center=1, oc=0)
    assert all(old is new for old, new in zip(before, model.mats_env))


@pytest.mark.parametrize('model_class, options', [
    (tk.models.UMPS, {'phys_dim': 2}),
    (tk.models.UMPO, {'in_dim': 2, 'out_dim': 2}),
    (tk.models.UMPSLayer, {'in_dim': 2, 'out_dim': 2}),
])
def test_uniform_orth_center_signatures(model_class, options):
    model = model_class(n_sites=3, bond_dim=2, **options)
    with pytest.raises(NotImplementedError):
        model.canonicalize(orth_center=1)
    with pytest.warns(DeprecationWarning, match='`oc` is deprecated'):
        with pytest.raises(NotImplementedError):
            model.canonicalize(oc=1)
