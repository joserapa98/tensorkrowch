"""Small dense oracles used to validate ALS implementations."""

from itertools import product
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


def make_tt_cores(
        input_dim: Sequence[int] = (2, 3, 2),
        rank: Sequence[int] = (2, 3),
        dtype: torch.dtype = torch.float64,
        generator: Optional[torch.Generator] = None
        ) -> List[torch.Tensor]:
    """Creates deterministic standard-form TT cores for tests."""
    if len(rank) != (len(input_dim) - 1):
        raise ValueError('TT rank should contain one value per link')
    ranks = (1, *rank, 1)
    return [
        torch.randn(
            ranks[site], site_input_dim, ranks[site + 1],
            dtype=dtype,
            generator=generator)
        for site, site_input_dim in enumerate(input_dim)
    ]


def make_tr_cores(
        input_dim: Sequence[int] = (2, 3, 2, 2),
        rank: Sequence[int] = (2, 3, 2, 2),
        dtype: torch.dtype = torch.float64,
        generator: Optional[torch.Generator] = None
        ) -> List[torch.Tensor]:
    """Creates deterministic heterogeneous-rank TR cores for tests."""
    if len(rank) != len(input_dim):
        raise ValueError('TR rank should contain one value per link')
    return [
        torch.randn(
            rank[site - 1], site_input_dim, rank[site],
            dtype=dtype,
            generator=generator)
        for site, site_input_dim in enumerate(input_dim)
    ]


def contract_tt_dense(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Contracts standard-form TT cores without TensorKrowch models."""
    result = cores[0]
    for core in cores[1:]:
        result = torch.tensordot(result, core, dims=([-1], [0]))
    return result.squeeze(0).squeeze(-1)


def contract_tr_dense(cores: Sequence[torch.Tensor]) -> torch.Tensor:
    """Contracts standard-form TR cores and closes the cyclic trace."""
    result = cores[0]
    for core in cores[1:]:
        result = torch.tensordot(result, core, dims=([-1], [0]))
    return result.diagonal(dim1=0, dim2=-1).sum(-1)


def build_tr_environment(
        cores: Sequence[torch.Tensor],
        site: int) -> torch.Tensor:
    """Contracts every TR core except ``site`` in cyclic order."""
    order = list(range(site + 1, len(cores))) + list(range(site))
    environment = cores[order[0]]
    for other_site in order[1:]:
        environment = torch.tensordot(
            environment, cores[other_site], dims=([-1], [0]))
    return environment


def direct_environment_slices(
        cores: Sequence[torch.Tensor],
        site: int) -> Iterable[Tuple[Tuple[int, ...], torch.Tensor]]:
    """Yields direct matrix products for all environment configurations."""
    order = list(range(site + 1, len(cores))) + list(range(site))
    input_dim = [cores[other_site].shape[1] for other_site in order]
    for configuration in product(*(range(dim) for dim in input_dim)):
        matrices = [
            cores[other_site][:, value, :]
            for other_site, value in zip(order, configuration)
        ]
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result @ matrix
        yield configuration, result


def dense_local_design(
        cores: Sequence[torch.Tensor],
        site: int,
        topology: str) -> torch.Tensor:
    """Builds the exact dense map from one vectorized core to the tensor."""
    if topology == 'tt':
        contract = contract_tt_dense
    elif topology == 'tr':
        contract = contract_tr_dense
    else:
        raise ValueError("`topology` should be 'tt' or 'tr'")

    columns = []
    for column in range(cores[site].numel()):
        basis_core = torch.zeros_like(cores[site])
        basis_core.reshape(-1)[column] = 1
        basis_cores = list(cores)
        basis_cores[site] = basis_core
        columns.append(contract(basis_cores).reshape(-1))
    return torch.stack(columns, dim=1)


def solve_local_core(
        cores: Sequence[torch.Tensor],
        target: torch.Tensor,
        site: int,
        topology: str,
        rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Solves one exact or row-restricted local least-squares problem."""
    design = dense_local_design(cores, site, topology)
    values = target.reshape(-1)
    if rows is not None:
        design = design.index_select(0, rows)
        values = values.index_select(0, rows)
    solution = torch.linalg.lstsq(design, values.unsqueeze(-1)).solution
    return solution.squeeze(-1).reshape_as(cores[site])


def absorb_right_qr(
        cores: Sequence[torch.Tensor],
        site: int,
        fixed_sites: Sequence[int] = ()
        ) -> Tuple[List[torch.Tensor], bool]:
    """Applies a right QR gauge only when its neighbour is trainable."""
    updated = [core.clone() for core in cores]
    next_site = (site + 1) % len(cores)
    if next_site in fixed_sites:
        return updated, False

    core = updated[site]
    matrix = core.reshape(core.shape[0] * core.shape[1], core.shape[2])
    q, r = torch.linalg.qr(matrix, mode='reduced')
    updated[site] = q.reshape(core.shape[0], core.shape[1], q.shape[1])
    updated[next_site] = torch.tensordot(
        r, updated[next_site], dims=([1], [0]))
    return updated, True


def reference_tr_sweep(
        cores: Sequence[torch.Tensor],
        target: torch.Tensor,
        fixed_sites: Sequence[int] = (),
        rows: Optional[torch.Tensor] = None,
        qr: bool = False) -> List[torch.Tensor]:
    """Runs one deliberately dense TR-ALS sweep for characterization."""
    updated = [core.clone() for core in cores]
    for site in range(len(updated)):
        if site in fixed_sites:
            continue
        updated[site] = solve_local_core(
            updated, target, site, topology='tr', rows=rows)
        if qr:
            updated, _ = absorb_right_qr(updated, site, fixed_sites)
    return updated


def sampled_rows(
        n_rows: int,
        n_samples: int,
        generator: torch.Generator) -> torch.Tensor:
    """Draws the legacy uniform row ids with an explicit generator."""
    return torch.randint(
        n_rows, (n_samples,), generator=generator, dtype=torch.long)


def observed_error(
        cores: Sequence[torch.Tensor],
        target: torch.Tensor,
        rows: torch.Tensor) -> torch.Tensor:
    """Returns the L2 error restricted to fixed observed entries."""
    residual = contract_tr_dense(cores).reshape(-1) - target.reshape(-1)
    return residual.index_select(0, rows).norm()
