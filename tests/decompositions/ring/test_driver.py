"""Tests for the isolated bidirectional tensor ring driver."""

from dataclasses import dataclass

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.decompositions.ring.driver import (
    BoundaryClosure,
    BidirectionalRingDriver,
    BidirectionalRingResult,
)
from tensorkrowch.decompositions.ring.schedules import AlternatingRingDriver


class SyntheticProvider:
    """Provides rank-one local targets without defining TT or RSS semantics."""

    def __init__(self, input_dim):
        self.input_dim = tuple(input_dim)

    def local_target(self, sites, context):
        return tuple(sites)

    def local_rank(self, sites, rank, context):
        return (1,) * (len(sites) + 2)

    def local_context(self, sites, context):
        return {
            'input_dim': (1, *(self.input_dim[site] for site in sites), 1),
            'sites': tuple(sites),
        }


class SyntheticOpenProvider(SyntheticProvider):
    """Adds provider-specific absorption of both open target boundaries."""

    boundary_mode = 'open'

    def close_boundary(self, site, direction, opening, context):
        return BoundaryClosure(
            site=site,
            direction=direction,
            core=torch.full((1, self.input_dim[site], 1), float(site + 1)),
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
            site=local_target[0]).inverse_or_pinv('pinv')
        record = gauge_map.require_cancellable()
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
            torch.full((1, kwargs['context']['input_dim'][offset + 1], 1),
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

    def test_central_block_cannot_consume_the_boundary_reconciliation_site(self):
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


class TestAlternatingRingDriver:  # MARK: TestAlternatingRingDriver

    def test_cyclic_even_ring_opens_anchors_then_two_fixed_sites(self):
        calls = []
        opener = _synthetic_opener(calls)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = AlternatingRingDriver().fit(
                SyntheticProvider((2,) * 6),
                rank=1,
                opener=opener,
                fixed_opener=opener,
                recursion=SyntheticRecursion([]))

        assert result.order == (
            (0,), (2,), (4,), (1,), (3,), (5,))
        assert result.directions == (
            'anchor', 'anchor', 'anchor', 'fixed', 'fixed', 'fixed')
        assert result.diagnostics['schedule'] == 'alternating'
        assert result.diagnostics['anchor_blocks'] == ((0,), (2,), (4,))
        assert all(call['fixed_left'] and call['fixed_right']
                   for call in calls[3:])
        assert len(result.diagnostics['gauge_stability']) == 6

    def test_open_odd_ring_closes_edges_from_alternating_anchors(self):
        calls = []
        opener = _synthetic_opener(calls)
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = AlternatingRingDriver().fit(
                SyntheticOpenProvider((2,) * 7),
                rank=1,
                opener=opener,
                fixed_opener=opener,
                recursion=SyntheticRecursion([]))

        assert result.order == (
            (1,), (3,), (5,), (2,), (4,), (6,), (0,))
        assert result.directions == (
            'anchor', 'anchor', 'anchor', 'fixed', 'fixed',
            'right_boundary', 'left_boundary')
        assert set(result.boundaries) == {0, 6}
        assert result.diagnostics['boundary_mode'] == 'open'

    def test_supports_larger_blocks_with_a_capable_fixed_opener(self):
        opener = _synthetic_opener([])
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = AlternatingRingDriver().fit(
                SyntheticProvider((2,) * 8),
                rank=1,
                opener=opener,
                fixed_opener=opener,
                recursion=SyntheticRecursion([]),
                block_size=2)

        assert result.order == ((0, 1), (4, 5), (2, 3), (6, 7))
        assert result.diagnostics['fixed_blocks'] == ((2, 3), (6, 7))

    def test_unsupported_partition_falls_back_explicitly(self):
        with pytest.warns(tk.decompositions.ExperimentalWarning):
            result = AlternatingRingDriver().fit(
                SyntheticProvider((2,) * 5),
                rank=1,
                opener=_synthetic_opener([]),
                recursion=SyntheticRecursion([]))

        assert result.diagnostics['schedule'] == 'center_out'
        assert result.diagnostics['requested_schedule'] == 'alternating'
        assert 'divisible' in result.diagnostics['fallback_reason']

    def test_can_reject_fallback(self):
        with pytest.warns(tk.decompositions.ExperimentalWarning), \
                pytest.raises(ValueError, match='unavailable'):
            AlternatingRingDriver().fit(
                SyntheticProvider((2,) * 5),
                rank=1,
                opener=_synthetic_opener([]),
                recursion=SyntheticRecursion([]),
                fallback=False)
