"""Tests for the public ALS exports and engine/function contracts."""

import inspect

import pytest
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions import als


def test_als_exports_match_the_simplified_api():
    assert 'SamplingPolicy' in als.__all__
    assert tk.decompositions.SamplingPolicy is als.SamplingPolicy
    for removed in ('UpdatePolicy', 'GaugePolicy', 'NoGauge', 'QRGauge',
                    'SVDGauge'):
        assert removed not in als.__all__
        assert not hasattr(als, removed)


@pytest.mark.parametrize('n_sites', [1, 5])
@pytest.mark.parametrize('engine, wrapper, cls, direction', [
    (tk.decompositions.TTALS, tk.decompositions.tt_als,
     tk.formats.TT, 'alternating'),
    (tk.decompositions.TRALS, tk.decompositions.tr_als,
     tk.formats.TR, 'forward')])
def test_engines_and_functions_return_results_with_fixed_sweep_order(
        n_sites, engine, wrapper, cls, direction):
    tensor = torch.ones((2,) * n_sites, dtype=torch.float64)
    direct = engine(tensor, out_device=None).fit(
        rank=1, collect_metrics=True,
        generator=torch.Generator().manual_seed(2),
        convergence=tk.decompositions.ConvergencePolicy(max_sweeps=2))
    functional = wrapper(
        tensor, rank=1, max_sweeps=2, collect_metrics=True,
        generator=torch.Generator().manual_seed(2), out_device=None)

    forward = list(range(n_sites))
    expected = forward + (list(reversed(forward))
                          if direction == 'alternating' else forward)
    for result in (direct, functional):
        assert isinstance(result, cls)
        assert result.metadata['sweep_direction'] == direction
        assert [
            record.site for record in result.metrics.local_solves
        ] == expected
        torch.testing.assert_close(result.contract_dense(), tensor)
    torch.testing.assert_close(
        direct.contract_dense(), functional.contract_dense())


@pytest.mark.parametrize('engine, wrapper', [
    (tk.decompositions.TTALS, tk.decompositions.tt_als),
    (tk.decompositions.TRALS, tk.decompositions.tr_als)])
def test_advanced_settings_belong_to_engine_policies(engine, wrapper):
    fit_parameters = inspect.signature(engine.fit).parameters
    wrapper_parameters = inspect.signature(wrapper).parameters
    assert {'sampler', 'solver', 'convergence'} <= fit_parameters.keys()
    assert not {'sampler', 'solver', 'convergence'} & wrapper_parameters.keys()
    for parameters in (fit_parameters, wrapper_parameters):
        assert (
            not {
                'init',
                'update_policy',
                'sweep_direction',
                'return_info',
                'return_result',
                'sampling',
                'leverage_mode',
            }
            & parameters.keys()
        )


@pytest.mark.parametrize(
    'engine', [tk.decompositions.TTALS, tk.decompositions.TRALS]
)
@pytest.mark.parametrize('fixed', [False, True])
def test_nonfinite_initial_and_fixed_cores_are_rejected(engine, fixed):
    tensor = torch.ones(2, 3, dtype=torch.float64)
    initial = engine(tensor).fit(rank=1).cores
    invalid = [core.clone() for core in initial]
    invalid[0].reshape(-1)[0] = torch.nan
    options = dict(initial_cores=initial, fixed_cores=invalid) if fixed else \
        dict(initial_cores=invalid)
    with pytest.raises(ValueError, match='finite'):
        engine(tensor).fit(**options)


@pytest.mark.parametrize(
    'engine', [tk.decompositions.TTALS, tk.decompositions.TRALS]
)
def test_matrix_initialization_checks_paired_dimensions(engine):
    initial_source = tk.decompositions.DenseTensorSource.matrix(
        torch.ones(2, 3), ordering='interleaved')
    target_source = tk.decompositions.DenseTensorSource.matrix(
        torch.ones(3, 2), ordering='interleaved')
    initial = engine(initial_source).fit(rank=1)
    with pytest.raises(ValueError, match='input/output dimensions'):
        engine(target_source).fit(initial_cores=initial)


@pytest.mark.parametrize(
    'engine', [tk.decompositions.TTALS, tk.decompositions.TRALS]
)
@pytest.mark.parametrize('method', ['uniform', 'leverage'])
def test_sampled_nonfinite_source_values_fail_consistently(engine, method):
    source = tk.decompositions.CallableTensorSource(
        lambda indices: torch.full((len(indices),), torch.inf),
        (2, 3), out_shape=(), dtype=torch.float32)
    sampler = tk.decompositions.SamplingPolicy(5, method)
    with pytest.raises(ValueError, match='finite'):
        engine(source).fit(rank=1, sampler=sampler)
