"""Tests for hierarchical QTT/QTR-Tucker recursive sketching."""

import torch

import tensorkrowch as tk


def _scalar_problem():
    domain = torch.tensor([0., 1.], dtype=torch.float64)
    grid = torch.linspace(0, 1, 4, dtype=torch.float64)
    samples = torch.cartesian_prod(grid, grid)

    def function(values):
        x, y = values.unbind(dim=1)
        return 1 + x + 2 * y + x * y

    return function, samples, domain


class TestQTTTuckerRSS:

    def test_two_level_scalar_function_and_flatten(self):
        function, samples, domain = _scalar_problem()
        result = tk.decompositions.QTTTuckerRSS(
            function,
            n_variables=2,
            base=2,
            level=2,
            domain=domain,
            output_device=None).fit(
                samples,
                rank=2,
                connector_rank=2,
                factor_rank=4,
                batch_size=32,
                legacy_projection=False,
                generator=torch.Generator().manual_seed(41),
                collect_metrics=True)

        assert isinstance(
            result, tk.decompositions.QTTTuckerDecomposition)
        assert len(result.upper.cores) == 2
        assert len(result.factors) == 2
        assert result.variable_positions == (0, 1)
        assert result.input_dim == (2, 2, 2, 2)
        assert torch.allclose(
            result.evaluate(samples), function(samples),
            rtol=1e-9, atol=1e-11)
        assert torch.allclose(
            result.flatten().contract_dense().reshape(4, 4),
            function(samples).reshape(4, 4),
            rtol=1e-9, atol=1e-11)
        assert any(
            record.phase == 'qtt_connector'
            for record in result.metrics.truncations)
        assert any(
            record.phase == 'qtt_factor'
            for record in result.metrics.truncations)

    def test_tensor_output_remains_an_upper_site(self):
        _, samples, domain = _scalar_problem()

        def function(values):
            x, y = values.unbind(dim=1)
            return torch.stack((1 + x + y, x - y), dim=1)

        labels = torch.arange(samples.shape[0]) % 2
        result, info = tk.decompositions.qtt_tucker_rss(
            function,
            samples,
            n_variables=2,
            base=2,
            level=2,
            domain=domain,
            labels=labels,
            out_position=1,
            rank=2,
            connector_rank=2,
            factor_rank=4,
            batch_size=32,
            legacy_projection=False,
            generator=torch.Generator().manual_seed(42),
            output_device=None,
            return_info=True)

        assert result.variable_positions == (0, 2)
        assert result.output_shape == (2,)
        assert result.evaluate(samples).shape == (samples.shape[0], 2)
        assert torch.allclose(result.evaluate(samples), function(samples))
        assert info['topology'] == 'qtt_tucker'
        assert info['output_shape'] == [2]

    def test_heterogeneous_interleaved_layout_evaluates_same_points(self):
        layout = tk.decompositions.QuantizedLayout(
            n_variables=2,
            base=(2, 3),
            level=(2, 1),
            ordering='interleaved',
            digit_order='fine_to_coarse')
        first = torch.linspace(0, 1, 4, dtype=torch.float64)
        second = torch.linspace(-1, 1, 3, dtype=torch.float64)
        samples = torch.cartesian_prod(first, second)

        def function(values):
            return (1 + values[:, 0]) * (2 - values[:, 1])

        result = tk.decompositions.qtt_tucker_rss(
            function,
            samples,
            layout=layout,
            domain=(torch.tensor([0., 1.], dtype=torch.float64),
                    torch.tensor([-1., 1.], dtype=torch.float64)),
            rank=1,
            connector_rank=1,
            factor_rank=2,
            batch_size=16,
            output_device=None)

        assert result.layout.ordering == 'interleaved'
        assert result.input_dim == (2, 2, 3)
        assert torch.allclose(result.evaluate(samples), function(samples))
        digits = layout.encode_indices(torch.cartesian_prod(
            torch.arange(4), torch.arange(3)))
        assert torch.allclose(result.evaluate_digits(digits), function(samples))

    def test_complex_function_preserves_dtype(self):
        function, samples, domain = _scalar_problem()

        def complex_function(values):
            return function(values).to(torch.complex128) * (1 + 0.5j)

        result = tk.decompositions.qtt_tucker_rss(
            complex_function,
            samples,
            n_variables=2,
            base=2,
            level=2,
            domain=domain,
            rank=2,
            connector_rank=2,
            factor_rank=4,
            batch_size=32,
            output_device=None)

        assert result.dtype == torch.complex128
        assert torch.allclose(
            result.evaluate(samples), complex_function(samples))


class TestQTRTuckerRSS:

    def test_rank_one_three_variable_ring(self):
        domain = torch.tensor([0., 1.], dtype=torch.float64)
        grid = torch.linspace(0, 1, 4, dtype=torch.float64)
        samples = torch.cartesian_prod(grid, grid, grid)

        def function(values):
            return (1 + values).prod(dim=1)

        result = tk.decompositions.qtr_tucker_rss(
            function,
            samples,
            n_variables=3,
            base=2,
            level=2,
            domain=domain,
            rank=1,
            connector_rank=1,
            factor_rank=2,
            center=1,
            batch_size=64,
            generator=torch.Generator().manual_seed(43),
            output_device=None)

        assert isinstance(
            result, tk.decompositions.QTRTuckerDecomposition)
        assert result.variable_positions == (0, 1, 2)
        assert result.upper.rank == [1, 1, 1]
        assert torch.allclose(result.evaluate(samples), function(samples))
        assert torch.allclose(
            result.flatten().contract_dense().reshape(4, 4, 4),
            function(samples).reshape(4, 4, 4))
