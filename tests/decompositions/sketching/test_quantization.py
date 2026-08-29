"""Tests for multivariable quantized layouts and coordinate maps."""

import pytest

import torch
import tensorkrowch as tk


class TestQuantizedLayout:  # MARK: TestQuantizedLayout

    @pytest.mark.parametrize('ordering', ['grouped', 'interleaved'])
    @pytest.mark.parametrize(
        'digit_order', ['coarse_to_fine', 'fine_to_coarse'])
    def test_multivariable_roundtrip(self, ordering, digit_order):
        layout = tk.decompositions.QuantizedLayout(
            n_variables=3,
            base=(2, 3, 2),
            level=(3, 2, 1),
            ordering=ordering,
            digit_order=digit_order)
        indices = torch.tensor([
            [0, 0, 0], [7, 8, 1], [3, 5, 0], [6, 2, 1]])

        digits = layout.encode_indices(indices)

        assert digits.shape == (4, 6)
        assert torch.equal(layout.decode_digits(digits), indices)
        assert layout.grid_size == (8, 9, 2)
        assert layout.input_dim == tuple(
            layout.base[variable] for variable, _ in layout.sites())

    def test_standard_xyz_schedules_with_unequal_levels(self):
        grouped = tk.decompositions.QuantizedLayout(
            3, level=(3, 2, 1), ordering='grouped')
        interleaved = tk.decompositions.QuantizedLayout(
            3, level=(3, 2, 1), ordering='interleaved')

        assert grouped.sites() == (
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1),
            (2, 0))
        assert interleaved.sites() == (
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1),
            (0, 2))

    def test_fine_to_coarse_reverses_each_variable_schedule(self):
        grouped = tk.decompositions.QuantizedLayout(
            2, level=(3, 2), digit_order='fine_to_coarse')
        interleaved = tk.decompositions.QuantizedLayout(
            2,
            level=(3, 2),
            ordering='interleaved',
            digit_order='fine_to_coarse')

        assert grouped.sites() == (
            (0, 2), (0, 1), (0, 0), (1, 1), (1, 0))
        assert interleaved.sites() == (
            (0, 2), (1, 1), (0, 1), (1, 0), (0, 0))

    def test_custom_permutation_and_reordering_are_reversible(self):
        grouped = tk.decompositions.QuantizedLayout(
            2, base=2, level=(2, 3))
        custom = tk.decompositions.QuantizedLayout(
            2,
            base=2,
            level=(2, 3),
            ordering='custom',
            permutation=((1, 2), (0, 0), (1, 0), (0, 1), (1, 1)))
        indices = torch.tensor([[0, 0], [3, 7], [2, 5]])
        grouped_digits = grouped.encode_indices(indices)

        custom_digits = grouped.reorder_configurations(
            grouped_digits, custom)
        restored = custom.reorder_configurations(
            custom_digits, grouped)

        assert torch.equal(custom.decode_digits(custom_digits), indices)
        assert torch.equal(restored, grouped_digits)

    @pytest.mark.parametrize(
        'kwargs, error, match',
        [
            ({'n_variables': 0}, ValueError, 'positive'),
            ({'n_variables': 2, 'base': (2,)}, ValueError, 'one value'),
            ({'n_variables': 1, 'base': 1}, ValueError, 'at least two'),
            ({'n_variables': 1, 'level': 0}, ValueError, 'positive'),
            ({'n_variables': 1, 'ordering': 'custom'}, ValueError,
             'permutation'),
            ({'n_variables': 1, 'level': 2, 'ordering': 'custom',
              'permutation': ((0, 0), (0, 0))}, ValueError, 'every'),
            ({'n_variables': 1, 'base': 2, 'level': 63}, OverflowError,
             'int64'),
        ])
    def test_invalid_layouts_are_rejected(self, kwargs, error, match):
        with pytest.raises(error, match=match):
            tk.decompositions.QuantizedLayout(**kwargs)

    def test_invalid_shapes_and_bounds_are_rejected(self):
        layout = tk.decompositions.QuantizedLayout(2, level=2)

        with pytest.raises(ValueError, match='n_variables'):
            layout.encode_indices(torch.tensor([[0, 1, 2]]))
        with pytest.raises(ValueError, match='out of bounds'):
            layout.encode_indices(torch.tensor([[4, 0]]))
        with pytest.raises(ValueError, match='layout sites'):
            layout.decode_digits(torch.tensor([[0, 1]]))
        with pytest.raises(ValueError, match='out of bounds'):
            layout.decode_digits(torch.tensor([[0, 1, 2, 0]]))


__all__ = []
