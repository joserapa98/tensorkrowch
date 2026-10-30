"""Tests for center-out ring construction and boundary block assembly."""

from dataclasses import dataclass
from math import factorial

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.ring.driver import (BidirectionalRingDriver,
                                                     BidirectionalRingResult,
                                                     BoundaryClosure)


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

    def close_boundary(self, sites, direction, opening, context):
        return BoundaryClosure(
            sites=sites,
            direction=direction,
            cores=tuple(torch.full(
                (1, self.in_dim[site], 1), float(site + 1)) for site in sites),
            diagnostics={'synthetic_boundary': True})


@dataclass
class SyntheticRecursion:
    calls: list

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
        return tk.decompositions.GaugeRecursionStep(
            gauge=gauge,
            records=(record,),
            diagnostics={'target': tuple(local_target)})

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
        result = BidirectionalRingDriver().fit(
            provider=SyntheticProvider((2,) * 7),
            rank=1,
            opener=_synthetic_opener(open_calls),
            recursion=SyntheticRecursion(recursion_calls),
            center=3)

        assert isinstance(result, BidirectionalRingResult)
        assert result.central_block.sites == (3,)
        assert result.order == (
            (3,), (4,), (2,), (5,), (1,), (6,), (0,))
        assert result.directions == (
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
        result = BidirectionalRingDriver().fit(
            SyntheticProvider((2,) * 5),
            rank=1,
            opener=_synthetic_opener([]),
            recursion=SyntheticRecursion([]),
            center=2)

        assert [core[0, 0, 0].item() for core in result.cores] == [
            1, 2, 3, 4, 5]
        assert result.rank == (1, 1, 1, 1, 1)
        assert isinstance(result.as_decomposition(),
                          tk.decompositions.TRDecomposition)

    @pytest.mark.parametrize('center', [0, 6])
    def test_center_can_start_at_either_array_boundary(self, center):
        result = BidirectionalRingDriver().fit(
            SyntheticProvider((2,) * 7),
            rank=1,
            opener=_synthetic_opener([]),
            recursion=SyntheticRecursion([]),
            center=center)

        assert result.central_block.sites == (center,)
        assert set(site for sites in result.order for site in sites) == set(
            range(7))
        assert result.directions[-1] == 'boundary'

    def test_multisite_center_is_stored_as_one_opening(self):
        result = BidirectionalRingDriver().fit(
            SyntheticProvider((2,) * 7),
            rank=2,
            opener=_synthetic_opener([]),
            recursion=SyntheticRecursion([]),
            center=3)

        assert result.central_block.sites == (2, 3, 4)
        assert result.order[0] == (2, 3, 4)
        assert len(result.openings[(2, 3, 4)].cores) == 3

    def test_uses_a_separate_opener_for_the_two_gauge_boundary(self):
        main_calls = []
        boundary_calls = []
        result = BidirectionalRingDriver().fit(
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
        assert result.directions[-1] == 'boundary'

    def test_open_target_boundaries_use_two_independent_sweeps(self):
        open_calls = []
        result = BidirectionalRingDriver().fit(
            SyntheticOpenProvider((2,) * 7),
            rank=1,
            opener=_synthetic_opener(open_calls, supports_two=False),
            recursion=SyntheticRecursion([]),
            center=3)

        assert result.order == (
            (3,), (4,), (5,), (6,), (2,), (1,), (0,))
        assert result.directions == (
            'center', 'right', 'right', 'right_boundary',
            'left', 'left', 'left_boundary')
        assert set(result.boundaries) == {0, 6}
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
                    return tk.decompositions.GaugeRecursionStep(
                        gauge=torch.ones(1, 1, 2))
                return step

        with pytest.raises(ValueError, match='Adjacent TR ranks should match'):
            BidirectionalRingDriver().fit(
                SyntheticProvider((2,) * 5),
                rank=1,
                opener=opener,
                recursion=IncompatibleRecursion([]),
                center=2)


class TestBoundaryClosure:

    def test_multisite_closure_keeps_all_cores(self):
        cores = tuple(torch.ones(1, 2, 1) for _ in range(3))
        closure = BoundaryClosure((0, 1, 2), 'left', cores)
        assert closure.sites == (0, 1, 2)
        assert closure.cores == cores

    @pytest.mark.parametrize('sites', [(), (True,), ('0',)])
    def test_invalid_site_types(self, sites):
        with pytest.raises(TypeError, match='sites'):
            BoundaryClosure(sites, 'left', ())

    @pytest.mark.parametrize('sites', [(-1,), (0, 2), (2, 1)])
    def test_invalid_site_interval(self, sites):
        with pytest.raises(ValueError, match='sites'):
            BoundaryClosure(sites, 'left', ())

    @pytest.mark.parametrize('cores', [(), (torch.ones(2, 2),), ('core',)])
    def test_invalid_cores(self, cores):
        with pytest.raises(ValueError, match='cores'):
            BoundaryClosure((0,), 'left', cores)


def _select_sites(sites):
    """Creates an internal selection with a prescribed contiguous interval."""
    from tensorkrowch.decompositions.ring.blocks import (
        _BlockSelection,
        _CentralBlockSelector,
    )

    class Selection(_CentralBlockSelector):
        def select(self, provider, rank, center=None, *, bounds=None):
            return _BlockSelection(
                sites=sites,
                in_dim=tuple(provider.in_dim[site] for site in sites),
                left_rank_cap=1, right_rank_cap=1,
                input_capacity=2**len(sites), required_input_capacity=1,
                feasible=True, reason='prescribed_test_block')

    return Selection()


class TestOpenBoundaryBlocks:

    def test_two_multisite_boundaries_leave_only_single_site_recursion(self):
        calls = []
        result = BidirectionalRingDriver().fit(
            SyntheticOpenProvider((2,) * 11), rank=1,
            opener=_synthetic_opener(calls, supports_two=False),
            recursion=SyntheticRecursion([]),
            block_selector=_select_sites((4, 5)),
            context={'left_boundary': (0, 1),
                     'right_boundary': (8, 9, 10)})
        assert result.order == ((4, 5), (6,), (7,), (8, 9, 10),
                                (3,), (2,), (0, 1))
        assert result.boundaries[0].sites == (0, 1)
        assert result.boundaries[8].sites == (8, 9, 10)
        assert len(result.boundaries[8].cores) == 3
        assert all(len(call['sites']) == 1 for call in calls[1:])
        assert result.rank == (1,) * 11
        torch.testing.assert_close(
            result.contract_dense(),
            torch.full((2,) * 11, float(factorial(11))))

    @pytest.mark.parametrize('center, left, right', [
        ((0, 1, 2), (), (6, 7)),
        ((5, 6, 7), (0, 1), ()),
    ])
    def test_center_merged_with_one_boundary(self, center, left, right):
        result = BidirectionalRingDriver().fit(
            SyntheticOpenProvider((2,) * 8), rank=1,
            opener=_synthetic_opener([]), recursion=SyntheticRecursion([]),
            block_selector=_select_sites(center),
            context={'left_boundary': left, 'right_boundary': right})
        assert result.order[0] == center
        assert len(result.boundaries) == 1
        assert [core[0, 0, 0].item() for core in result.cores] == \
            list(range(1, 9))

    @pytest.mark.parametrize('n_sites', [1, 2, 3, 5])
    def test_all_blocks_merged_requires_only_one_opening(self, n_sites):
        calls = []
        recursion_calls = []
        sites = tuple(range(n_sites))
        result = BidirectionalRingDriver().fit(
            SyntheticOpenProvider((2,) * n_sites), rank=1,
            opener=_synthetic_opener(calls),
            recursion=SyntheticRecursion(recursion_calls),
            block_selector=_select_sites(sites),
            context={'left_boundary': (), 'right_boundary': ()})
        assert result.order == (sites,)
        assert result.directions == ('center',)
        assert len(calls) == 1
        assert not recursion_calls
        assert not result.boundaries
        assert result.rank == (1,) * n_sites

    def test_missing_closure_implementation_is_rejected(self):
        provider = SyntheticProvider((2,) * 5)
        provider.boundary_mode = 'open'
        with pytest.raises(TypeError, match='close_boundary'):
            BidirectionalRingDriver().fit(
                provider, rank=1, opener=_synthetic_opener([]),
                recursion=SyntheticRecursion([]), center=2)
