"""Tests for als/init."""

import pytest
import torch

import tensorkrowch as tk


def test_als_public_engines_share_sources_and_results():
    tensor = torch.ones(2, 2, dtype=torch.float64)
    convergence = tk.decompositions.ConvergencePolicy(max_sweeps=1)
    for engine, cls in ((tk.decompositions.TTALS, tk.formats.TT),
                        (tk.decompositions.TRALS, tk.formats.TR)):
        result = engine(tensor, out_device=None).fit(rank=1, init='svd', convergence=convergence)
        assert isinstance(result, cls)
        torch.testing.assert_close(result.contract_dense(), tensor)


@pytest.mark.parametrize('sweep_direction', [None, 'forward', 'alternating'])
@pytest.mark.parametrize('engine, wrapper', [
    (tk.decompositions.TTALS, tk.decompositions.tt_als),
    (tk.decompositions.TRALS, tk.decompositions.tr_als)])
def test_public_als_sweep_direction(engine, wrapper, sweep_direction):
    tensor = torch.ones(2, 2, 2, dtype=torch.float64)
    options = {} if sweep_direction is None else {
        'sweep_direction': sweep_direction}
    direct = engine(tensor, out_device=None).fit(
        rank=1, init='svd', collect_metrics=True,
        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=2),
        **options)
    functional = wrapper(
        tensor, rank=1, init='svd', max_sweeps=2,
        return_info=True, out_device=None, **options)

    direction = sweep_direction or 'forward'
    expected = [0, 1, 2] + (
        [2, 1, 0] if direction == 'alternating' else [0, 1, 2])
    assert direct.metadata['sweep_direction'] == direction
    assert [record.site for record in direct.metrics.local_solves] == expected
    assert functional[1]['metadata']['sweep_direction'] == direction
    assert [record['site'] for record in
            functional[1]['metrics']['local_solves']] == expected
    torch.testing.assert_close(direct.contract_dense(), tensor)
