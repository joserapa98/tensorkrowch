"""Tests for center-out ring construction and boundary block assembly."""

from dataclasses import dataclass
from math import factorial

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.ring.driver import BidirectionalRingDriver


class SyntheticProvider:
    """Provides rank-one local targets without defining TT or RSS semantics."""

    def __init__(self, in_dim):
        self.in_dim = tuple(in_dim)

    def local_target(self, sites, context):
        return tuple(sites)

    def local_rank(self, sites, rank, context):
        return (1,) * (len(sites) + 2)

    def local_context(self, sites, context):
        return {
            'in_dim': (1, *(self.in_dim[site] for site in sites), 1),
            'sites': tuple(sites),
        }


class SyntheticOpenProvider(SyntheticProvider):
    """Adds provider-specific absorption of both open target boundaries."""

    boundary_mode = 'open'

    def __init__(self, in_dim, sites=None, left=None, right=None):
        super().__init__(in_dim)
        self.sites, self.left, self.right = sites, left, right

    def partition(self, rank, center, block_boundaries, rtol, *,
                  block_center=True):
        from tensorkrowch.decompositions.ring.blocks import _select_block
        sites = self.sites
        if sites is None:
            sites = (_select_block(self, rank, center)['sites']
                     if block_center else (center,))
        left = self.left
        right = self.right
        if left is None:
            left = () if sites[0] == 0 else (0,)
        if right is None:
            right = () if sites[-1] == len(self.in_dim) - 1 else \
                (len(self.in_dim) - 1,)
        ranks = (rank,) * len(self.in_dim) if isinstance(rank, int) else rank
        return tuple(sites), left, right, ranks, None, rtol

    def close_boundary(self, sites, direction, opening, context):
        return tuple(torch.full((1, self.in_dim[site], 1), float(site + 1))
                     for site in sites)


@dataclass
class SyntheticRecursion:
    calls: list  # Recorded recursion directions and source/destination sites

    @staticmethod
    def _step(direction, opening, local_target, recursion_context):
        gauge = opening.right_gauge if direction == 'right' \
            else opening.left_gauge
        gauge_map = tk.decompositions.GaugeMap(
            gauge,
            orientation=direction,
            site=local_target[0]).pinv()
        record = gauge_map.diagnostics()
        recursion_context['calls'].append((
            direction,
            recursion_context['from_sites'],
            recursion_context['to_sites'],
        ))
        return tk.decompositions.GaugeMap(
            gauge, 'left' if direction == 'right' else 'right',
            records=(record,),
            metadata={'target': tuple(local_target)})

    def advance_left(self, opening, local_target, recursion_context):
        recursion_context = dict(recursion_context)
        recursion_context['calls'] = self.calls
        return self._step(
            'left', opening, local_target, recursion_context)

    def advance_right(self, opening, local_target, recursion_context):
        recursion_context = dict(recursion_context)
        recursion_context['calls'] = self.calls
        return self._step(
            'right', opening, local_target, recursion_context)


def _synthetic_opener(calls, supports_two=True, mutate_fixed=False):
    capabilities = tk.decompositions.LoopOpenerCapabilities(
        supports_fixed_left=True,
        supports_fixed_right=True,
        supports_two_fixed_gauges=supports_two,
        supports_blocks=True)

    def open_loop(**kwargs):
        sites = tuple(kwargs['context']['sites'])
        fixed_left = kwargs['fixed_left']
        fixed_right = kwargs['fixed_right']
        left = torch.ones(1, 1, 1) if fixed_left is None else fixed_left
        right = torch.ones(1, 1, 1) if fixed_right is None else fixed_right
        if mutate_fixed and fixed_left is not None:
            left = left + 1
        cores = tuple(
            torch.full((1, kwargs['context']['in_dim'][offset + 1], 1),
                       float(site + 1))
            for offset, site in enumerate(sites))
        all_cores = (left, *cores, right)
        calls.append({
            'sites': sites,
            'orientation': kwargs['orientation'],
            'fixed_left': fixed_left is not None,
            'fixed_right': fixed_right is not None,
        })
        return tk.decompositions.LoopOpening(
            left_gauge=left,
            cores=cores,
            right_gauge=right,
            rank=tuple(core.shape[-1] for core in all_cores),
            orientation=kwargs['orientation'],
            diagnostics={'synthetic': True})

    return tk.decompositions.CallableLoopOpener(open_loop, capabilities)


class TestBidirectionalRingDriver:  # MARK: TestBidirectionalRingDriver

    def test_runs_two_sweeps_and_reconciles_last_boundary_site(self):
        open_calls = []
        recursion_calls = []
        driver = BidirectionalRingDriver()
        result = driver.fit(
            provider=SyntheticProvider((2,) * 7),
            rank=1,
            opener=_synthetic_opener(open_calls),
            recursion=SyntheticRecursion(recursion_calls),
            center=3)

        assert isinstance(result, tk.decompositions.TRDecomposition)
        assert driver.central_block == (3,)
        assert driver.order == (
            (3,), (4,), (2,), (5,), (1,), (6,), (0,))
        assert driver.directions == (
            'center', 'right', 'left', 'right', 'left', 'right', 'boundary')
        assert [(call['sites'], call['fixed_left'], call['fixed_right'])
                for call in open_calls] == [
            ((3,), False, False),
            ((4,), True, False),
            ((2,), False, True),
            ((5,), True, False),
            ((1,), False, True),
            ((6,), True, False),
            ((0,), True, True),
        ]
        assert [call['orientation'] for call in open_calls] == [
            'right', 'right', 'left', 'right', 'left', 'right', 'right']
        assert recursion_calls[-2:] == [
            ('right', (6,), (0,)),
            ('left', (1,), (0,)),
        ]
        assert len(result.metrics.gauges) == 7

    def test_assembles_cores_in_original_site_order(self):
        driver = BidirectionalRingDriver()
        result = driver.fit(
            SyntheticProvider((2,) * 5),
            rank=1,
            opener=_synthetic_opener([]),
            recursion=SyntheticRecursion([]),
            center=2)

        assert [core[0, 0, 0].item() for core in result.cores] == [
            1, 2, 3, 4, 5]
        assert result.rank == [1, 1, 1, 1, 1]
        assert isinstance(result,
                          tk.decompositions.TRDecomposition)

    @pytest.mark.parametrize('center', [0, 6])
    def test_center_can_start_at_either_array_boundary(self, center):
        driver = BidirectionalRingDriver()
        driver.fit(
            SyntheticProvider((2,) * 7),
            rank=1,
            opener=_synthetic_opener([]),
            recursion=SyntheticRecursion([]),
            center=center)

        assert driver.central_block == (center,)
        assert set(site for sites in driver.order for site in sites) == set(
            range(7))
        assert driver.directions[-1] == 'boundary'

    def test_multisite_center_is_stored_as_one_opening(self):
        driver = BidirectionalRingDriver()
        driver.fit(
            SyntheticProvider((2,) * 7),
            rank=2,
            opener=_synthetic_opener([]),
            recursion=SyntheticRecursion([]),
            center=3)

        assert driver.central_block == (2, 3, 4)
        assert driver.order[0] == (2, 3, 4)
        assert len(driver.openings[(2, 3, 4)].cores) == 3

    def test_uses_a_separate_opener_for_the_two_gauge_boundary(self):
        main_calls = []
        boundary_calls = []
        driver = BidirectionalRingDriver()
        driver.fit(
            SyntheticProvider((2,) * 5),
            rank=1,
            opener=_synthetic_opener(main_calls, supports_two=False),
            boundary_opener=_synthetic_opener(boundary_calls),
            recursion=SyntheticRecursion([]),
            center=2)

        assert all(not (call['fixed_left'] and call['fixed_right'])
                   for call in main_calls)
        assert len(boundary_calls) == 1
        assert boundary_calls[0]['fixed_left']
        assert boundary_calls[0]['fixed_right']
        assert driver.directions[-1] == 'boundary'

    def test_open_target_boundaries_use_two_independent_sweeps(self):
        open_calls = []
        driver = BidirectionalRingDriver()
        result = driver.fit(
            SyntheticOpenProvider((2,) * 7),
            rank=1,
            opener=_synthetic_opener(open_calls, supports_two=False),
            recursion=SyntheticRecursion([]),
            center=3)

        assert driver.order == (
            (3,), (4,), (5,), (6,), (2,), (1,), (0,))
        assert driver.directions == (
            'center', 'right', 'right', 'right_boundary',
            'left', 'left', 'left_boundary')
        assert set(driver.boundaries) == {0, 6}
        assert all(not (call['fixed_left'] and call['fixed_right'])
                   for call in open_calls)
        assert [core[0, 0, 0].item() for core in result.cores] == [
            1, 2, 3, 4, 5, 6, 7]

    def test_infeasible_central_selection_has_explicit_diagnostic(self):
        with pytest.raises(ValueError, match='available_sites_exhausted'):
            BidirectionalRingDriver().fit(
                SyntheticProvider((2, 2, 2)),
                rank=10,
                opener=_synthetic_opener([]),
                recursion=SyntheticRecursion([]))

    def test_cyclic_center_leaves_one_reconciliation_site(self):
        with pytest.raises(ValueError, match='leave at least one boundary'):
            BidirectionalRingDriver().fit(
                SyntheticProvider((2, 2, 2)),
                rank=2,
                opener=_synthetic_opener([]),
                recursion=SyntheticRecursion([]),
                center=1)

    def test_rejects_an_opener_that_changes_a_fixed_gauge(self):
        with pytest.raises(ValueError, match='changed the fixed left'):
            BidirectionalRingDriver().fit(
                SyntheticProvider((2,) * 5),
                rank=1,
                opener=_synthetic_opener([], mutate_fixed=True),
                recursion=SyntheticRecursion([]),
                center=2)

    def test_rejects_global_rank_incompatibility_after_assembly(self):
        calls = []
        opener = _synthetic_opener(calls)

        class IncompatibleRecursion(SyntheticRecursion):

            def advance_right(self, opening, local_target, recursion_context):
                step = super().advance_right(
                    opening, local_target, recursion_context)
                if local_target == (4,):
                    return tk.decompositions.GaugeMap(
                        torch.ones(1, 1, 2), 'left')
                return step

        with pytest.raises(ValueError, match='Adjacent TR ranks should match'):
            BidirectionalRingDriver().fit(
                SyntheticProvider((2,) * 5),
                rank=1,
                opener=opener,
                recursion=IncompatibleRecursion([]),
                center=2)


class TestBoundaryAssembly:

    @pytest.mark.parametrize('sites, cores', [
        ((), ()), ((-1,), (torch.ones(1, 2, 1),)),
        ((0, 2), (torch.ones(1, 2, 1),) * 2),
        ((0,), (torch.ones(2, 2),)), ((0,), ()),
    ])
    def test_invalid_closure_does_not_mutate_assembly(self, sites, cores):
        assembled = [None] * 3
        with pytest.raises((TypeError, ValueError)):
            BidirectionalRingDriver._store_boundary(
                sites, 'left', cores, assembled, {}, [], [])
        assert assembled == [None] * 3


class TestOpenBoundaryBlocks:

    def test_prescribed_boundary_blocks_use_constrained_openings(self):
        calls = []
        driver = BidirectionalRingDriver()
        result = driver.fit(
            SyntheticOpenProvider((2,) * 11, sites=(4, 5),
                                  left=(0, 1), right=(8, 9, 10)), rank=1,
            opener=_synthetic_opener(calls, supports_two=False),
            recursion=SyntheticRecursion([]),

            context={'left_boundary': (0, 1),
                     'right_boundary': (8, 9, 10)})
        assert driver.order == ((4, 5), (6,), (7,), (8, 9, 10),
                                (3,), (2,), (0, 1))
        assert len(driver.boundaries[0]) == 2
        assert len(driver.boundaries[8]) == 3
        assert [call['sites'] for call in calls] == list(driver.order)
        assert calls[3]['fixed_left']
        assert calls[-1]['fixed_right']
        assert result.rank == [1] * 11
        torch.testing.assert_close(
            result.contract_dense(),
            torch.full((2,) * 11, float(factorial(11))))

    @pytest.mark.parametrize('center, left, right', [
        ((0, 1, 2), (), (6, 7)),
        ((5, 6, 7), (0, 1), ()),
    ])
    def test_center_merged_with_one_boundary(self, center, left, right):
        driver = BidirectionalRingDriver()
        result = driver.fit(
            SyntheticOpenProvider((2,) * 8, sites=center,
                                  left=left, right=right), rank=1,
            opener=_synthetic_opener([]), recursion=SyntheticRecursion([]),

            context={'left_boundary': left, 'right_boundary': right})
        assert driver.order[0] == center
        assert len(driver.boundaries) == 1
        assert [core[0, 0, 0].item() for core in result.cores] == \
            list(range(1, 9))

    @pytest.mark.parametrize('n_sites', [1, 2, 3, 5])
    def test_all_blocks_merged_requires_only_one_opening(self, n_sites):
        calls = []
        recursion_calls = []
        sites = tuple(range(n_sites))
        driver = BidirectionalRingDriver()
        result = driver.fit(
            SyntheticOpenProvider((2,) * n_sites, sites=sites,
                                  left=(), right=()), rank=1,
            opener=_synthetic_opener(calls),
            recursion=SyntheticRecursion(recursion_calls),

            context={'left_boundary': (), 'right_boundary': ()})
        assert driver.order == (sites,)
        assert driver.directions == ('center',)
        assert len(calls) == 1
        assert not recursion_calls
        assert not driver.boundaries
        assert result.rank == [1] * n_sites

    def test_missing_closure_implementation_is_rejected(self):
        provider = SyntheticProvider((2,) * 5)
        provider.boundary_mode = 'open'
        with pytest.raises(TypeError, match='close_boundary'):
            BidirectionalRingDriver().fit(
                provider, rank=1, opener=_synthetic_opener([]),
                recursion=SyntheticRecursion([]), center=2)
