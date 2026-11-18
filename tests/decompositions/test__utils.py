"""Tests for common decomposition validation and index conversions."""

import pytest
import torch

from tensorkrowch.decompositions._utils import (_normalize_dims,
                                                _normalize_positive_int,
                                                _normalize_rank,
                                                _ravel_indices,
                                                _unravel_indices)


class TestDimensionValidation:  # MARK: TestDimensionValidation

    @pytest.mark.parametrize('dimensions', [
        (2, 3), [2, 3], torch.Size([2, 3])])
    def test_positive_dimensions(self, dimensions):
        assert _normalize_dims(dimensions, 'in_dim') == (2, 3)

    @pytest.mark.parametrize('dimensions', [
        2, True, '23', {2, 3}, (2, True), (2, 3.)])
    def test_invalid_dimension_types(self, dimensions):
        with pytest.raises(TypeError, match='in_dim'):
            _normalize_dims(dimensions, 'in_dim')

    @pytest.mark.parametrize('dimensions', [(), (0,), (-1,)])
    def test_invalid_dimension_values(self, dimensions):
        with pytest.raises(ValueError, match='in_dim'):
            _normalize_dims(dimensions, 'in_dim')

    def test_only_scalar_output_shapes_may_be_empty(self):
        assert _normalize_dims((), 'out_shape', allow_empty=True) == ()

    @pytest.mark.parametrize('value, error', [
        (True, TypeError), (2., TypeError), (0, ValueError), (-1, ValueError)])
    def test_positive_integer_contract(self, value, error):
        with pytest.raises(error, match='n_sites'):
            _normalize_positive_int(value, 'n_sites')


class TestRankValidation:  # MARK: TestRankValidation

    def test_shared_and_per_link_ranks(self):
        assert _normalize_rank(2, 3) == (2, 2, 2)
        assert _normalize_rank([1, 2, 3], 3) == (1, 2, 3)

    def test_one_site_tt_has_no_links(self):
        assert _normalize_rank(2, 0) == ()
        assert _normalize_rank((), 0) == ()

    @pytest.mark.parametrize('rank, error', [
        (True, TypeError), ((1, True), TypeError), ((1, 2.), TypeError),
        (0, ValueError), ((1, 0), ValueError), ((1,), ValueError)])
    def test_invalid_ranks(self, rank, error):
        with pytest.raises(error, match='rank'):
            _normalize_rank(rank, 2)

    @pytest.mark.parametrize('rank, error', [
        (True, TypeError), (0, ValueError), (-1, ValueError)])
    def test_scalar_rank_is_checked_without_links(self, rank, error):
        with pytest.raises(error, match='rank'):
            _normalize_rank(rank, 0)


class TestIndexConversions:  # MARK: TestIndexConversions

    def test_row_major_index_order(self):
        indices = torch.tensor([[0, 0, 0], [0, 1, 3], [1, 2, 3]])
        flat_ids = torch.tensor([0, 7, 23])
        assert torch.equal(_ravel_indices(indices, (2, 3, 4)), flat_ids)
        assert torch.equal(_unravel_indices(flat_ids, (2, 3, 4)), indices)

    def test_scalar_shape_and_empty_batches(self):
        indices = torch.empty(3, 0, dtype=torch.long)
        flat_ids = torch.zeros(3, dtype=torch.long)
        assert torch.equal(_ravel_indices(indices, ()), flat_ids)
        assert torch.equal(_unravel_indices(flat_ids, ()), indices)
        empty_ids = torch.empty(0, dtype=torch.long)
        empty_indices = torch.empty(0, 2, dtype=torch.long)
        assert _ravel_indices(empty_indices, (2, 3)).shape == (0,)
        assert _unravel_indices(empty_ids, (2, 3)).shape == (0, 2)


def test_ravel_indices_rejects_unrepresentable_global_space():
    with pytest.raises(ValueError, match='int64 capacity'):
        _ravel_indices(torch.zeros(2, 20, dtype=torch.long), (10,) * 20)


def test_ravel_indices_preserves_int64_boundary():
    indices = torch.ones(1, 63, dtype=torch.long)
    flat = _ravel_indices(indices, (2,) * 63)
    assert flat.item() == torch.iinfo(torch.int64).max
    torch.testing.assert_close(_unravel_indices(flat, (2,) * 63), indices)


class TestCoordinateMatching:

    def test_large_space_reordered_repeated_and_missing_rows(self):
        from tensorkrowch.decompositions._utils import _match_indices
        reference = torch.stack((torch.ones(144, dtype=torch.long),
                                  torch.zeros(144, dtype=torch.long)))
        missing = reference[0].clone()
        missing[0] = 0
        indices = torch.stack((reference[1], reference[0],
                               missing, reference[1]))
        assert _match_indices(indices, reference).tolist() == [1, 0, -1, 1]
        assert _match_indices(indices, reference[:0]).tolist() == [-1] * 4
        assert _match_indices(indices[:0], reference).shape == (0,)
        empty = torch.empty(3, 0, dtype=torch.long)
        assert _match_indices(empty, empty[:1]).tolist() == [0] * 3
