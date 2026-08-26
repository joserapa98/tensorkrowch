"""Characterization tests and dense baselines for the ALS refactor."""

import pytest

import torch

from tests.decompositions.als._oracles import (
    absorb_right_qr,
    build_tr_environment,
    contract_tr_dense,
    contract_tt_dense,
    dense_local_design,
    direct_environment_slices,
    make_tr_cores,
    make_tt_cores,
    observed_error,
    reference_tr_sweep,
    sampled_rows,
    solve_local_core,
)
from tensorkrowch.decompositions.results import (TRDecomposition,
                                                 TTDecomposition)


class TestDenseALSOracles:  # MARK: TestDenseALSOracles

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_tt_and_tr_contractions_match_results(self, dtype):
        generator = torch.Generator().manual_seed(0)
        tt_cores = make_tt_cores(dtype=dtype, generator=generator)
        tr_cores = make_tr_cores(dtype=dtype, generator=generator)

        tt_result_cores = [tt_cores[0].squeeze(0)]
        tt_result_cores.extend(tt_cores[1:-1])
        tt_result_cores.append(tt_cores[-1].squeeze(-1))

        assert torch.allclose(
            contract_tt_dense(tt_cores),
            TTDecomposition(tt_result_cores).contract_dense())
        assert torch.allclose(
            contract_tr_dense(tr_cores),
            TRDecomposition(tr_cores).contract_dense())

    def test_heterogeneous_tr_environment_matches_direct_products(self):
        cores = make_tr_cores(
            input_dim=(2, 3, 2, 2),
            rank=(2, 3, 2, 4),
            generator=torch.Generator().manual_seed(1))

        for site in range(len(cores)):
            environment = build_tr_environment(cores, site)
            assert environment.shape[0] == cores[site].shape[-1]
            assert environment.shape[-1] == cores[site].shape[0]
            for configuration, expected in direct_environment_slices(
                    cores, site):
                index = (slice(None), *configuration, slice(None))
                assert torch.allclose(environment[index], expected)

    @pytest.mark.parametrize('topology', ['tt', 'tr'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_local_design_reconstructs_original_tensor(self,
                                                       topology,
                                                       dtype):
        generator = torch.Generator().manual_seed(2)
        if topology == 'tt':
            cores = make_tt_cores(dtype=dtype, generator=generator)
            contract = contract_tt_dense
        else:
            cores = make_tr_cores(dtype=dtype, generator=generator)
            contract = contract_tr_dense

        for site, core in enumerate(cores):
            design = dense_local_design(cores, site, topology)
            actual = design @ core.reshape(-1)
            assert torch.allclose(
                actual.reshape_as(contract(cores)),
                contract(cores),
                rtol=1e-12,
                atol=1e-12)


class TestLegacyTRALSCharacterization:  # MARK: TestLegacyTRALSCharacterization

    def test_fixed_cores_are_preserved_bit_for_bit(self):
        generator = torch.Generator().manual_seed(3)
        target_cores = make_tr_cores(generator=generator)
        target = contract_tr_dense(target_cores)
        initial = [core.clone() for core in target_cores]
        initial[1] = torch.randn(
            initial[1].shape, dtype=initial[1].dtype, generator=generator)
        initial[3] = torch.randn(
            initial[3].shape, dtype=initial[3].dtype, generator=generator)
        fixed_sites = (0, 2)

        updated = reference_tr_sweep(
            initial, target, fixed_sites=fixed_sites, qr=True)

        for site in fixed_sites:
            assert torch.equal(updated[site], initial[site])
        assert torch.linalg.vector_norm(
            contract_tr_dense(updated) - target) <= torch.linalg.vector_norm(
                contract_tr_dense(initial) - target)

    def test_qr_is_absorbed_only_into_a_trainable_neighbour(self):
        cores = make_tr_cores(generator=torch.Generator().manual_seed(4))
        target = contract_tr_dense(cores)

        gauged, applied = absorb_right_qr(cores, site=0)
        q_matrix = gauged[0].reshape(-1, gauged[0].shape[-1])

        assert applied
        assert torch.allclose(
            q_matrix.mH @ q_matrix,
            torch.eye(q_matrix.shape[1], dtype=q_matrix.dtype),
            rtol=1e-12,
            atol=1e-12)
        assert torch.allclose(contract_tr_dense(gauged), target)

        skipped, applied = absorb_right_qr(
            cores, site=0, fixed_sites=(1,))
        assert not applied
        assert all(torch.equal(before, after)
                   for before, after in zip(cores, skipped))

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_zero_target_local_solve_is_finite(self, dtype):
        cores = make_tr_cores(
            dtype=dtype, generator=torch.Generator().manual_seed(5))
        target = torch.zeros_like(contract_tr_dense(cores))

        core = solve_local_core(cores, target, site=1, topology='tr')

        assert torch.isfinite(core).all()
        updated = list(cores)
        updated[1] = core
        assert torch.allclose(contract_tr_dense(updated), target)

    def test_rank_deficient_local_system_has_a_finite_solution(self):
        cores = make_tr_cores(generator=torch.Generator().manual_seed(6))
        cores[2] = torch.zeros_like(cores[2])
        target = torch.randn(
            contract_tr_dense(cores).shape,
            dtype=cores[0].dtype,
            generator=torch.Generator().manual_seed(7))
        design = dense_local_design(cores, site=0, topology='tr')

        core = solve_local_core(cores, target, site=0, topology='tr')

        assert torch.linalg.matrix_rank(design) < design.shape[1]
        assert torch.isfinite(core).all()

    def test_sampled_rows_are_deterministic_with_generator(self):
        first = sampled_rows(
            48, 20, generator=torch.Generator().manual_seed(8))
        second = sampled_rows(
            48, 20, generator=torch.Generator().manual_seed(8))
        different = sampled_rows(
            48, 20, generator=torch.Generator().manual_seed(9))

        assert torch.equal(first, second)
        assert not torch.equal(first, different)

    def test_completion_rows_remain_fixed_across_sweeps(self):
        generator = torch.Generator().manual_seed(10)
        target_cores = make_tr_cores(generator=generator)
        target = contract_tr_dense(target_cores)
        initial = make_tr_cores(generator=generator)
        rows = torch.tensor([0, 3, 7, 12, 18, 23])
        original_rows = rows.clone()
        errors = [observed_error(initial, target, rows)]

        updated = initial
        for _ in range(2):
            updated = reference_tr_sweep(updated, target, rows=rows)
            errors.append(observed_error(updated, target, rows))

        assert torch.equal(rows, original_rows)
        assert errors[1] <= errors[0]
        assert errors[2] <= errors[1] + 1e-12
