"""Shared format kernels and result provenance contracts."""

from dataclasses import replace
import torch
import tensorkrowch as tk
from tensorkrowch.decompositions.sketching.quantization import QuantizedLayout
from tensorkrowch.decompositions.sources.quantization import QuantizedSourceAdapter


def test_results_inherit_formats_and_keep_historical_metrics():
    result = tk.decompositions.TTDecomposition(
        [torch.ones(2, 1), torch.ones(1, 3)], metadata={'fit': 'original'})
    assert isinstance(result, tk.formats.TensorTrain)
    replaced = replace(result, cores=[torch.zeros(2, 1), torch.ones(1, 3)])
    assert replaced.norm() == 0
    for transformed in [result.copy(), result.detach(), result.to(dtype=torch.float64)]:
        assert type(transformed) is type(result)
        assert transformed.metrics is result.metrics
        assert transformed.metadata == result.metadata
    assert type(result + result) is tk.formats.TensorTrain
    assert QuantizedLayout is tk.formats.QuantizedLayout
    assert QuantizedSourceAdapter is tk.decompositions.QuantizedSourceAdapter


def test_base_tt_source_absorbs_bonds_without_mutating_format():
    result = tk.formats.TensorTrain([torch.ones(2, 2), torch.ones(2, 3)])
    result.bonds = tk.formats.BondFactors([torch.tensor([2., 3.])])
    source = tk.decompositions.as_tensor_source(result)
    indices = torch.tensor([[0, 0], [1, 2]])
    actual = source.evaluate(tk.decompositions.ConfigurationBatch(indices, kind='indices'))
    assert torch.equal(actual, torch.full((2,), 5.))
    assert result.bonds is not None


import pytest


@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
@pytest.mark.parametrize('method', ['tt', 'tr'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
def test_quantized_svd_preserves_raw_grid_and_returns_generic_algebra(ordering, method, dtype):
    layout = tk.formats.QuantizedLayout(2, 2, (3, 4), ordering=ordering)
    data = torch.arange(128, dtype=torch.float64).reshape(8, 16).to(dtype)
    if data.is_complex():
        data = data + 1j * data.flip(-1)
    result = getattr(tk.decompositions, method + '_svd')(
        data, rank=16, quantization=layout, return_result=True, out_device=None)
    cls = tk.formats.QuanticsTensorTrain if method == 'tt' else tk.formats.QuanticsTensorRing
    assert isinstance(result, cls)
    assert torch.allclose(result.to_dense_grid(), data, atol=1e-9)
    indices = torch.tensor([[0, 0], [7, 15], [3, 8]])
    assert torch.allclose(result.evaluate_indices(indices), data[indices[:, 0], indices[:, 1]], atol=1e-9)
    assert not result.metrics.truncations
    assert not isinstance(result + result, tk.decompositions.TensorDecomposition)


@pytest.mark.parametrize('method', ['ttm', 'trm'])
def test_quantized_matrix_svd_and_adjoint(method):
    a = tk.formats.QuantizedLayout(1, 2, 2)
    b = tk.formats.QuantizedLayout(1, 3, 2)
    data = torch.arange(36, dtype=torch.float64).reshape(4, 9)
    result = getattr(tk.decompositions, method + '_svd')(
        data, in_dim=(4,), out_dim=(9,), quantization=(a, b),
        rank=9, return_result=True, out_device=None)
    assert torch.allclose(result.to_dense_grid(), data, atol=1e-10)
    assert torch.allclose(result.H.to_dense_grid(), data.T, atol=1e-10)


def test_quantized_svd_structural_batches_and_invalid_controls():
    layout = tk.formats.QuantizedLayout(1, 2, 3)
    data = torch.arange(16, dtype=torch.float64).reshape(2, 8)
    result = tk.decompositions.tt_svd(data, n_batches=1, quantization=layout,
                                    return_result=True, rank=4)
    assert torch.allclose(result.to_dense_grid(), data, atol=1e-10)
    with pytest.raises(ValueError):
        tk.decompositions.tt_svd(torch.ones(7), quantization=layout)
    with pytest.raises(ValueError):
        tk.decompositions.tt_svd(data, return_result=True, return_info=True)
    with pytest.raises(TypeError):
        tk.decompositions.tt_svd(data, return_result=1)


@pytest.mark.parametrize('method', ['tt', 'tr'])
@pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
def test_quantized_als_raw_callable_and_repeated_fits(method, ordering):
    layout = tk.formats.QuantizedLayout(2, 2, 2, ordering=ordering)
    data = torch.arange(16, dtype=torch.float64).reshape(4, 4)
    cls = tk.decompositions.TTALS if method == 'tt' else tk.decompositions.TRALS
    engine = cls(data, quantization=layout, out_device=None)
    for _ in range(2):
        result = engine.fit(
            rank=2, init='svd', convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
        assert torch.allclose(result.to_dense_grid(), data, atol=1e-8)
    physical = cls(lambda points: torch.exp(points[:, 0] + 2 * points[:, 1]),
                   quantization=layout, dtype=torch.float64,
                   domain=torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64))
    result = physical.fit(rank=2, init='svd', convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
    points = torch.tensor([[0., 0.], [1., 1.]], dtype=torch.float64)
    assert torch.allclose(result.evaluate_points(points), torch.exp(torch.tensor([0., 3.], dtype=torch.float64)), atol=1e-9)
    assert not result.metrics.sweeps


@pytest.mark.parametrize('cls', [tk.decompositions.TTALS, tk.decompositions.TRALS])
def test_quantized_completion_keeps_values_weights_and_physical_collision_policy(cls):
    layout = tk.formats.QuantizedLayout(2, 2, 2, ordering='interleaved')
    indices = torch.tensor([[3, 1], [0, 0], [1, 2]])
    values = torch.tensor([4., 1., 3.], dtype=torch.float64)
    weights = torch.tensor([2., 3., 4.], dtype=torch.float64)
    engine = cls.completion(indices, values, weights=weights, quantization=layout)
    observed = engine.problem.observations
    decoded = layout.decode_digits(observed.indices)
    for i, row in enumerate(decoded):
        original = torch.nonzero((indices == row).all(-1))[0, 0]
        assert observed.values[i] == values[original]
        assert observed.weights[i] == weights[original]
    result = engine.fit(rank=2, convergence=tk.decompositions.ConvergencePolicy(max_sweeps=1))
    assert result.layout == layout
    with pytest.raises(ValueError, match='Repeated observations'):
        cls.completion(torch.tensor([[0., 0.], [0.01, 0.01]]),
                       torch.tensor([1., 2.]), quantization=layout, sample_space='physical', domain=torch.tensor([[0., 1.], [0., 1.]]))


def test_quantized_als_source_and_initializer_layout_validation():
    layout = tk.formats.QuantizedLayout(2, 2, 2)
    q = tk.decompositions.tt_svd(torch.ones(4, 4), quantization=layout, return_result=True)
    result = tk.decompositions.tt_als(q, quantization=layout, initial_cores=q,
                                    max_sweeps=1, return_result=True)
    assert torch.allclose(result.to_dense_grid(), torch.ones(4, 4))
    other = tk.formats.QuantizedLayout(2, 2, 2, ordering='interleaved')
    with pytest.raises(ValueError, match='fixed digit layout'):
        tk.decompositions.TTALS(torch.ones(4, 4), quantization=other).fit(initial_cores=q)


@pytest.mark.parametrize('method', ['tt', 'tr'])
def test_quantized_svd_retains_unquantized_output_sites(method):
    layout = tk.formats.QuantizedLayout(2, 2, 2, ordering='interleaved')
    data = torch.arange(48, dtype=torch.float64).reshape(4, 3, 4)
    result = getattr(tk.decompositions, method + '_svd')(
        data, quantization=layout, in_features=(0, 2), rank=16,
        return_result=True)
    indices = torch.tensor([[0, 0], [2, 3]])
    assert result.digit_positions == (0, 1, 2, 3)
    assert torch.allclose(result.evaluate_indices(indices),
                          data[indices[:, 0], :, indices[:, 1]], atol=1e-9)
    assert torch.allclose(result.to_dense_grid(), data.permute(0, 2, 1), atol=1e-9)
    replacement = replace(result, cores=result.cores)
    assert replacement.layout == layout
    assert torch.allclose(replacement.to_dense_grid(), result.to_dense_grid())


def test_quantized_rss_result_retains_coordinates_without_source():
    import gc
    import weakref
    layout = tk.formats.QuantizedLayout(2, 2, 2, ordering='interleaved')
    domain = torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64)
    indices = torch.cartesian_prod(torch.arange(4), torch.arange(4))
    points = indices.to(torch.float64) / 3

    def function(values):
        return torch.stack([1 + values[:, 0], 2 + values[:, 1]], dim=-1)

    reference = weakref.ref(function)
    result = tk.decompositions.qtt_rss(
        function, points, layout=layout, domain=domain, rank=4,
        legacy_projection=False, return_result=True)
    assert isinstance(result, tk.decompositions.QTTDecomposition)
    del function
    gc.collect()
    assert reference() is None
    expected = torch.stack([1 + points[:, 0], 2 + points[:, 1]], dim=-1)
    assert torch.allclose(result.evaluate_points(points), expected, atol=1e-9)
    restored = tk.formats.QuanticsTensorTrain.from_mps(
        result.to_mps(), layout=layout, coordinate_map=result.coordinate_map,
        domain=domain, digit_positions=result.digit_positions)
    assert torch.allclose(restored.evaluate_points(points), expected, atol=1e-9)
