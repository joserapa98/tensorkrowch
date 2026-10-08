"""Tests for ring/schedules."""

from dataclasses import dataclass

import pytest
import torch

import tensorkrowch as tk

from tensorkrowch.decompositions.ring.driver import BoundaryClosure
from tensorkrowch.decompositions.ring.schedules import AlternatingRingDriver


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

    def close_boundary(self, site, direction, opening, context):
        return BoundaryClosure(
            site=site,
            direction=direction,
            core=torch.full((1, self.in_dim[site], 1), float(site + 1)),
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
