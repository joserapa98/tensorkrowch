"""Tests for als/init."""

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
