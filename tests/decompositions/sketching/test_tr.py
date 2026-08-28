"""Tests for Tensor Ring recursive sketching from samples."""

import torch

import tensorkrowch as tk


def _rank_one_problem(n_sites=4):
    domain = torch.tensor([0., 1.], dtype=torch.float64)
    samples = torch.cartesian_prod(*(domain for _ in range(n_sites)))

    def function(data):
        return (1 + data).prod(dim=1)

    def embedding(values):
        return torch.stack((1 - values, values), dim=-1)

    return function, embedding, samples, domain


class TestTRRSS:

    def test_rank_one_function_matches_dense_tensor(self):
        function, embedding, samples, domain = _rank_one_problem()
        result = tk.decompositions.TRRSS(
            function=function,
            embedding=embedding,
            domain=domain).fit(
                samples,
                rank=1,
                collect_metrics=True)

        expected = function(samples).reshape(2, 2, 2, 2)
        assert isinstance(result, tk.decompositions.TRDecomposition)
        assert result.rank == [1, 1, 1, 1]
        assert torch.allclose(
            result.contract_dense(), expected, rtol=1e-8, atol=1e-10)
        assert result.metrics.errors[0].kind == 'sketch_samples'
        assert result.metrics.errors[0].relative < 1e-8
        assert len(result.metrics.truncations) == 3

    def test_repeated_fits_are_independent_and_functional_api_is_compatible(
            self):
        function, embedding, samples, domain = _rank_one_problem()
        decomposer = tk.decompositions.TRRSS(
            function=function,
            embedding=embedding,
            domain=domain)
        first = decomposer.fit(samples, rank=1)
        second = decomposer.fit(samples, rank=1)
        cores, info = tk.decompositions.tr_rss(
            function=function,
            embedding=embedding,
            sketch_samples=samples,
            domain=domain,
            rank=1,
            return_info=True)

        assert first is not second
        assert all(left is not right for left, right in zip(
            first.cores, second.cores))
        assert len(cores) == 4
        assert info['topology'] == 'tr'
        assert info['metadata']['requested_rank'] == [1, 1, 1, 1]

    def test_tensor_outputs_use_basis_sites_and_flat_labels(self):
        function, embedding, samples, domain = _rank_one_problem()

        def tensor_function(data):
            value = function(data)
            left = torch.tensor([1., 2.], dtype=data.dtype)
            right = torch.tensor([1., 3.], dtype=data.dtype)
            return value[:, None, None] * left[None, :, None] * \
                right[None, None, :]

        labels = torch.arange(samples.shape[0]).remainder(4)
        result = tk.decompositions.TRRSS(
            function=tensor_function,
            embedding=embedding,
            domain=domain,
            out_position=(1, 4)).fit(
                samples,
                labels=labels,
                rank=1,
                collect_metrics=True)

        assert result.input_dim == (2, 2, 2, 2, 2, 2)
        assert result.metadata['out_position'] == (1, 4)
        assert result.metrics.errors[0].relative < 1e-8

    def test_adaptive_block_discovers_ranks_and_padding_is_opt_in(self):
        function, embedding, samples, domain = _rank_one_problem(n_sites=5)
        decomposer = tk.decompositions.TRRSS(
            function=function,
            embedding=embedding,
            domain=domain)
        adaptive = decomposer.fit(
            samples,
            rank=(2, 2, 2, 2, 2),
            adaptive=True,
            collect_metrics=True)
        padded = decomposer.fit(
            samples,
            rank=2,
            adaptive=True,
            pad_to_rank=True)

        assert adaptive.rank == [1, 1, 1, 1, 1]
        assert adaptive.metadata['center_block'] == (1, 2, 3)
        assert adaptive.metadata['rank_estimate']['limitations'] == ()
        assert adaptive.metrics.errors[0].relative < 1e-8
        assert padded.rank == [2, 2, 2, 2, 2]
        assert torch.allclose(
            adaptive.contract_dense(), padded.contract_dense(),
            rtol=1e-10, atol=1e-12)


__all__ = []
