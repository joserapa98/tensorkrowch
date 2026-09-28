"""Small raw-tensor fixtures and independent dense oracles."""

import pytest
import torch
import tensorkrowch as tk


@pytest.fixture
def make_format():
    def build(topology='tt', n_sites=3, n_batches=0, dtype=torch.float64):
        generator = torch.Generator().manual_seed(31)
        cyclic = topology.startswith('tr')
        matrix = topology.endswith('m')
        batch = (2,) * n_batches
        in_dim = [2 + site % 2 for site in range(n_sites)]
        out_dim = [3 - site % 2 for site in range(n_sites)]
        ranks = [2 + site % 2 for site in range(n_sites + 1)]
        ranks[-1] = ranks[0] if cyclic else 1
        ranks[0] = ranks[0] if cyclic else 1
        cores = []
        for site in range(n_sites):
            shape = (*batch, ranks[site], in_dim[site], ranks[site + 1])
            if matrix:
                shape += (out_dim[site],)
            core = torch.randn(shape, dtype=dtype, generator=generator)
            if not cyclic:
                if site == 0:
                    core = core.squeeze(n_batches)
                if site == n_sites - 1:
                    core = core.squeeze(-2 if matrix else -1)
            cores.append(core)
        classes = {'tt': tk.formats.TensorTrain, 'tr': tk.formats.TensorRing,
                   'ttm': tk.formats.TensorTrainMatrix,
                   'trm': tk.formats.TensorRingMatrix}
        return classes[topology](cores, n_batches=n_batches)
    return build
