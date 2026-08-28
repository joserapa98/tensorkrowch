"""Tests for regional sample sketches and recursive gather maps."""

import pytest

import torch

from tensorkrowch.decompositions.sketching.regions import (
    RegionSketch,
    SiteRegion,
    SketchRecursion,
    _SamplePool,
)
from tensorkrowch.decompositions.tt_decompositions import create_projector


def _packed_values(sketch):
    """Packs homogeneous scalar/vector region values for legacy oracles."""
    return torch.stack(sketch.values, dim=1)


class TestSiteRegion:  # MARK: TestSiteRegion

    def test_linear_region_keeps_order_separate_from_membership(self):
        region = SiteRegion((2, 0, 1))

        assert tuple(region) == (2, 0, 1)
        assert region.contains(0)
        assert region.contains(SiteRegion((1, 2)))
        assert not region.contains(3)
        assert region.difference(SiteRegion((0,))) == SiteRegion((2, 1))
        assert region.intersection(SiteRegion((1, 3, 2))) == \
            SiteRegion((2, 1))

    def test_union_uses_deterministic_first_occurrence_order(self):
        left = SiteRegion((0, 2))
        right = SiteRegion((2, 1, 3))

        assert left.union(right) == SiteRegion((0, 2, 1, 3))
        assert SiteRegion().union(right) == right

    def test_nd_coordinates_are_hashable_and_dimension_aware(self):
        region = SiteRegion(((0, 1), (2, 3), (1, 1)))

        assert region.contains((2, 3))
        assert region.difference(SiteRegion(((0, 1),))) == \
            SiteRegion(((2, 3), (1, 1)))

    @pytest.mark.parametrize(
        'sites, error, match',
        [
            ((0, 0), ValueError, 'duplicates'),
            ((0, (0, 1)), ValueError, 'mix'),
            (((0, 1), (2, )), ValueError, 'same dimension'),
            ((True,), TypeError, 'integers or tuples'),
            (((),), TypeError, 'integers or tuples'),
            (((0, []),), TypeError, 'hashable'),
        ])
    def test_invalid_site_geometry(self, sites, error, match):
        with pytest.raises(error, match=match):
            SiteRegion(sites)


class TestSamplePool:  # MARK: TestSamplePool

    def test_scalar_restriction_has_vectorized_unique_inverse_and_cache(self):
        samples = torch.tensor([
            [0, 0, 5],
            [0, 1, 5],
            [0, 0, 6],
            [0, 1, 5],
        ])
        pool = _SamplePool(samples, pool_id='scalar')
        region = SiteRegion((0, 1))

        sketch = pool.restrict(region)

        assert isinstance(sketch, RegionSketch)
        assert sketch.pool_id == 'scalar'
        assert sketch.n_rows == 4
        assert sketch.n_unique == 2
        assert torch.equal(sketch.inverse_ids, torch.tensor([0, 1, 0, 1]))
        assert torch.equal(
            sketch.representative_row_ids, torch.tensor([0, 1]))
        assert torch.equal(
            _packed_values(sketch), torch.tensor([[0, 0], [0, 1]]))
        assert pool.restrict(region) is sketch

    def test_vector_input_dimensions_are_preserved_per_site(self):
        samples = torch.tensor([
            [[0., 1.], [2., 3.]],
            [[0., 1.], [4., 5.]],
            [[1., 0.], [2., 3.]],
            [[0., 1.], [2., 3.]],
        ])
        pool = _SamplePool(samples)

        sketch = pool.restrict(SiteRegion((0, 1)))

        assert sketch.n_unique == 3
        assert all(value.shape == (3, 2) for value in sketch.values)
        assert torch.equal(
            sketch.inverse_ids, torch.tensor([0, 1, 2, 0]))

    def test_heterogeneous_mixed_dtype_samples_use_coordinate_sites(self):
        samples = (
            torch.tensor([0., 0., 1., 1.]),
            torch.tensor([[1., 2.], [1., 2.], [3., 4.], [3., 4.]]),
            torch.tensor([2, 2, 3, 3]),
        )
        sites = ((0, 0), (0, 1), (1, 0))
        pool = _SamplePool(samples, sites=sites)

        sketch = pool.restrict(SiteRegion(((0, 1), (1, 0))))

        assert sketch.region.sites == ((0, 1), (1, 0))
        assert sketch.values[0].shape == (2, 2)
        assert sketch.values[1].dtype == torch.int64
        assert torch.equal(sketch.inverse_ids, torch.tensor([0, 0, 1, 1]))

    def test_complex_sample_rows_are_deduplicated_without_losing_phase(self):
        samples = (
            torch.tensor([1 + 2j, 1 + 2j, 1 - 2j]),
            torch.tensor([0., 0., 1.]),
        )

        sketch = _SamplePool(samples).restrict(SiteRegion((0, 1)))

        assert sketch.n_unique == 2
        assert sketch.values[0].is_complex()
        assert torch.equal(sketch.inverse_ids, torch.tensor([1, 1, 0]))

    def test_empty_region_represents_one_shared_boundary_state(self):
        sketch = _SamplePool(torch.arange(12).reshape(4, 3)).restrict(
            SiteRegion())

        assert sketch.n_unique == 1
        assert sketch.values == ()
        assert torch.equal(sketch.inverse_ids, torch.zeros(4, dtype=torch.long))
        assert torch.equal(
            sketch.representative_row_ids, torch.zeros(1, dtype=torch.long))

    @pytest.mark.parametrize(
        'samples, sites, error, match',
        [
            (torch.ones(3), None, ValueError, 'batch and site'),
            ((torch.ones(2), torch.ones(3)), None, ValueError, 'batch size'),
            ((torch.ones(2), torch.tensor([0., float('nan')])),
             None, ValueError, 'site 1'),
            (torch.ones(2, 3), (0, 1), ValueError, 'one identifier'),
        ])
    def test_invalid_pool_samples(self, samples, sites, error, match):
        with pytest.raises(error, match=match):
            _SamplePool(samples, sites=sites)


class TestRegionSketch:  # MARK: TestRegionSketch

    def test_restrict_and_correlated_combine_reuse_original_pool_rows(self):
        samples = torch.tensor([
            [0, 0, 5],
            [0, 0, 6],
            [1, 1, 5],
            [1, 1, 6],
        ])
        pool = _SamplePool(samples)
        left = pool.restrict(SiteRegion((0,)))
        middle = pool.restrict(SiteRegion((1,)))

        combined = left.combine(middle)

        assert left.n_unique == middle.n_unique == 2
        assert combined.n_unique == 2
        assert combined.region == SiteRegion((0, 1))
        assert combined is pool.restrict(SiteRegion((0, 1)))
        assert middle.combine(left) is combined

    def test_combine_rejects_equal_ids_from_distinct_pool_objects(self):
        first = _SamplePool(torch.tensor([[0, 0], [1, 1]]), pool_id='same')
        second = _SamplePool(torch.tensor([[0, 0], [1, 1]]), pool_id='same')

        with pytest.raises(ValueError, match='same sample pool'):
            first.restrict(SiteRegion((0,))).combine(
                second.restrict(SiteRegion((1,))))

    def test_compare_reports_common_and_directional_new_regions(self):
        sites = ((0, 0), (0, 1), (1, 0), (1, 1))
        pool = _SamplePool(torch.randn(5, 4), sites=sites)
        first = pool.restrict(SiteRegion(((0, 0), (0, 1), (1, 0))))
        second = pool.restrict(SiteRegion(((0, 1), (1, 1))))

        common, first_only, second_only = first.compare(second)

        assert common == SiteRegion(((0, 1),))
        assert first_only == SiteRegion(((0, 0), (1, 0)))
        assert second_only == SiteRegion(((1, 1),))

    def test_restrict_rejects_sites_outside_the_sketch(self):
        sketch = _SamplePool(torch.ones(3, 3)).restrict(
            SiteRegion((0, 1)))

        with pytest.raises(ValueError, match='contained'):
            sketch.restrict(SiteRegion((0, 2)))


class TestSketchRecursion:  # MARK: TestSketchRecursion

    @pytest.mark.parametrize('in_dim', [None, 2])
    def test_left_recursion_matches_legacy_create_projector(self, in_dim):
        scalar_samples = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 0],
        ])
        samples = scalar_samples if in_dim is None else torch.stack(
            (scalar_samples, 1 - scalar_samples), dim=2)
        pool = _SamplePool(samples)
        child = pool.restrict(SiteRegion((0,)))
        parent = pool.restrict(SiteRegion((0, 1)))

        recursion = child.recursive_projector(parent)
        legacy_ids, legacy_values = create_projector(
            _packed_values(child), _packed_values(parent))

        assert isinstance(recursion, SketchRecursion)
        assert recursion.new_region == SiteRegion((1,))
        assert torch.equal(recursion.gather, legacy_ids)
        assert torch.equal(recursion.new_values[0], legacy_values.squeeze(1))
        assert not hasattr(recursion, 'matrix')

    def test_right_recursion_uses_the_same_containment_operation(self):
        samples = torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 1],
        ])
        pool = _SamplePool(samples)
        child = pool.restrict(SiteRegion((2,)))
        parent = pool.restrict(SiteRegion((1, 2)))

        recursion = child.recursive_projector(parent)

        expected_rows = parent.representative_row_ids
        assert recursion.new_region == SiteRegion((1,))
        assert torch.equal(
            recursion.gather, child.inverse_ids[expected_rows])
        assert torch.equal(
            recursion.value(1), samples[expected_rows, 1])

    def test_apply_gathers_any_selected_tensor_axis(self):
        pool = _SamplePool(torch.tensor([
            [0, 0], [0, 1], [1, 0], [1, 1],
        ]))
        recursion = pool.restrict(SiteRegion((0,))).recursive_projector(
            pool.restrict(SiteRegion((0, 1))))
        tensor = torch.arange(12).reshape(3, 2, 2)

        result = recursion.apply(tensor, axis=1)

        assert torch.equal(
            result, tensor.index_select(1, recursion.gather))

    def test_composition_matches_direct_recursion_and_sequential_gathers(self):
        samples = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1],
            [1, 1, 1],
        ])
        pool = _SamplePool(samples)
        child = pool.restrict(SiteRegion((0,)))
        middle = pool.restrict(SiteRegion((0, 1)))
        parent = pool.restrict(SiteRegion((0, 1, 2)))
        first = child.recursive_projector(middle)
        second = middle.recursive_projector(parent)

        composed = first.compose(second)
        direct = child.recursive_projector(parent)
        tensor = torch.randn(child.n_unique, 3)

        assert composed.child_region == child.region
        assert composed.parent_region == parent.region
        assert torch.equal(composed.gather, direct.gather)
        assert all(torch.equal(left, right) for left, right in zip(
            composed.new_values, direct.new_values))
        assert torch.equal(
            composed.apply(tensor), second.apply(first.apply(tensor)))

    def test_bijective_recursion_has_a_gather_inverse(self):
        samples = torch.tensor([
            [0, 5], [1, 4], [2, 3], [1, 4],
        ])
        pool = _SamplePool(samples)
        child = pool.restrict(SiteRegion((0,)))
        parent = pool.restrict(SiteRegion((0, 1)))
        recursion = child.recursive_projector(parent)
        inverse = recursion.inverse()
        tensor = torch.randn(child.n_unique, 4)

        assert inverse.removed_region == SiteRegion((1,))
        assert torch.equal(inverse.apply(recursion.apply(tensor)), tensor)

    def test_non_bijective_recursion_cannot_be_inverted(self):
        pool = _SamplePool(torch.tensor([
            [0, 0], [0, 1], [1, 0], [1, 1],
        ]))
        recursion = pool.restrict(SiteRegion((0,))).recursive_projector(
            pool.restrict(SiteRegion((0, 1))))

        with pytest.raises(ValueError, match='bijective'):
            recursion.inverse()

    def test_projector_requires_containment_and_the_same_pool(self):
        first_pool = _SamplePool(torch.tensor([[0, 0], [1, 1]]))
        second_pool = _SamplePool(torch.tensor([[0, 0], [1, 1]]))
        left = first_pool.restrict(SiteRegion((0,)))

        with pytest.raises(ValueError, match='contain'):
            left.recursive_projector(
                first_pool.restrict(SiteRegion((1,))))
        with pytest.raises(ValueError, match='same sample pool'):
            left.recursive_projector(
                second_pool.restrict(SiteRegion((0, 1))))
