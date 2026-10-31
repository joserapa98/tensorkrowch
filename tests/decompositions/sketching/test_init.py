"""Tests for sketching/init."""

import torch

import tensorkrowch as tk


def test_sketching_shared_public_infrastructure():
    assert not hasattr(tk.decompositions.TTRSS, 'quantized')
    assert not hasattr(tk.decompositions.TRRSS, 'quantized')
    layout = tk.formats.QuantizedLayout(1, 2, 3)
    source = tk.decompositions.CallableQuanticsSource(
        lambda coordinates: torch.ones_like(coordinates[:, 0]), 1,
        layout=layout, coordinate_map=tk.formats.AffineCoordinateMap(
            [0., 1.], layout.grid_size))
    indices = layout.encode_indices(torch.tensor([[0], [7]]))
    values = source.evaluate(tk.decompositions.ConfigurationBatch(indices))
    torch.testing.assert_close(values, torch.ones(2))


def test_reviewed_sketching_exports_are_available():
    names = ['TTRSS', 'TRRSS', 'TTRS', 'tt_rss', 'tr_rss', 'tt_rs',
             'qtt_rss', 'qtr_rss', 'ConfigurationMap', 'PhiOperator',
             'InputFitter', 'RangeProjector', 'MarginalSketch',
             'SampledSketch', 'TTStackSketch']
    for name in names:
        assert hasattr(tk.decompositions, name)


def test_removed_experimental_controls_are_not_public():
    names = ['TRRS', 'tr_rs', 'SketchGaugeRecursion', 'EvaluationView',
             'GlobalValueTransform', 'LocalValueTransform',
             'LocalTransformContext', 'IdentityGlobalValueTransform',
             'CallableGlobalValueTransform', 'CompositeGlobalValueTransform']
    for name in names:
        assert not hasattr(tk.decompositions, name)
    assert not hasattr(tk.decompositions.MarginalSketch, 'markov')
