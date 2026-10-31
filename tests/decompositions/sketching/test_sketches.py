"""Tests for TT-RS sketch operators and core-determining systems."""

import pytest
import torch

import tensorkrowch as tk
from tensorkrowch.decompositions.sketching.sketches import _prepare_phi


def _rank_one_sparse(in_dim=(2, 3, 2), dtype=torch.float64):
    factors = [
        torch.arange(1, dimension + 1, dtype=dtype)
        for dimension in in_dim
    ]
    dense = factors[0]
    for factor in factors[1:]:
        dense = dense.unsqueeze(-1) * factor
    indices = torch.cartesian_prod(*(
        torch.arange(dimension) for dimension in in_dim)).reshape(
            -1, len(in_dim))
    return tk.decompositions.SparseTensorSource(
        indices, dense.reshape(-1), in_dim), dense


class TestPhiPreparation:

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('direction', ['left', 'right'])
    def test_auxiliary_rank_discovery_respects_caps_and_selected_axis(
            self, dtype, direction):
        tensor = torch.ones(4, 2, 4, dtype=dtype)
        left = direction == 'left'
        prepared = _prepare_phi(
            tensor, direction, rank_left=3, rank_right=3,
            numerical=(left, not left))
        prescribed = _prepare_phi(
            tensor, direction, rank_left=3, rank_right=3,
            numerical=(not left, left))
        axis = 0 if left else -1

        assert prepared.tensor.shape[axis] == 1
        assert prescribed.tensor.shape[axis] == 3

        tensor = torch.randn(4, 2, 4, dtype=dtype)
        capped = _prepare_phi(
            tensor, direction, rank_left=2, rank_right=2,
            numerical=(True, True))
        assert capped.tensor.shape[axis] == 2

    @pytest.mark.parametrize('numerical, shape', [
        ((True, False), (1, 2, 3)),
        ((False, True), (2, 2, 1)),
        ((True, True), (1, 2, 1))])
    def test_central_auxiliary_rank_discovery_is_axis_specific(
            self, numerical, shape):
        tensor = torch.ones(4, 2, 4, dtype=torch.float64)
        prepared = _prepare_phi(
            tensor, 'center', rank_left=2, rank_right=3,
            numerical=numerical)

        assert prepared.tensor.shape == shape
        approximation = torch.einsum(
            'ab,bpc,cd->apd', prepared.left, prepared.tensor, prepared.right)
        torch.testing.assert_close(approximation, tensor)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('rtol', [None, 1e-22])
    def test_central_support_selection_preserves_small_singular_values(
            self, dtype, rtol):
        factors = [torch.randn(size, 2, dtype=dtype) for size in (6, 2, 6)]
        tensor = torch.einsum('a,p,b->apb', *(x[:, 0] for x in factors))
        tensor = tensor + 1e-8 * torch.einsum(
            'a,p,b->apb', *(x[:, 1] for x in factors))
        prepared = _prepare_phi(tensor, 'center', rtol=rtol, site=2)

        assert prepared.tensor.shape == (2, 2, 2)
        approximation = torch.einsum(
            'ab,bpc,cd->apd', prepared.left, prepared.tensor, prepared.right)
        torch.testing.assert_close(approximation, tensor,
                                   rtol=1e-12, atol=1e-13)

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_central_factors_restore_full_target(self, dtype):
        tensor = torch.randn(3, 4, 5, dtype=dtype)
        prepared = _prepare_phi(tensor, 'center', site=2,
                                 collect_metrics=True)
        reconstructed = torch.einsum(
            'ab,bpc,cd->apd', prepared.left, prepared.tensor, prepared.right)

        torch.testing.assert_close(reconstructed, tensor)
        torch.testing.assert_close(prepared.tensor.norm(), tensor.norm())
        torch.testing.assert_close(
            prepared.left.mH @ prepared.left,
            torch.eye(prepared.left.shape[-1], dtype=dtype))
        torch.testing.assert_close(
            prepared.right @ prepared.right.mH,
            torch.eye(prepared.right.shape[0], dtype=dtype))
        assert len(prepared.truncations) == 2
        assert [record.site for record in prepared.truncations] == [2, 1]

    @pytest.mark.parametrize('direction', ['left', 'right'])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_outward_tensors_retain_singular_values(self, direction, dtype):
        tensor = torch.randn(3, 4, 5, dtype=dtype)
        prepared = _prepare_phi(tensor, direction, rank_left=2, rank_right=2)
        basis = prepared.tensor
        if direction == 'left':
            basis = basis.reshape(basis.shape[0], -1).mH
            matrix = tensor.reshape(tensor.shape[0], -1).mH
        else:
            basis = basis.reshape(-1, basis.shape[-1])
            matrix = tensor.reshape(-1, tensor.shape[-1])

        spectra = torch.linalg.svdvals(matrix)[:2]
        torch.testing.assert_close(
            basis.mH @ basis, torch.diag(spectra.square()).to(dtype))
        assert prepared.left is None
        assert prepared.right is None

    @pytest.mark.parametrize('direction', ['left', 'right', 'center'])
    def test_truncated_tensors_preserve_retained_scales(self, direction):
        tensor = torch.diag(torch.tensor([4., 2., 0.1])).unsqueeze(1)
        prepared = _prepare_phi(
            tensor, direction, rank_left=2, rank_right=2)
        assert prepared.tensor.norm().item() == pytest.approx(20 ** 0.5)
        if direction == 'center':
            approximation = torch.einsum(
                'ab,bpc,cd->apd', prepared.left, prepared.tensor,
                prepared.right)
            assert (tensor - approximation).norm().item() == \
                pytest.approx(0.1)

    def test_randomized_projection_is_optional_and_owns_its_parameters(self):
        tensor = torch.randn(3, 4, 5, dtype=torch.float64)
        projector = tk.decompositions.RandomizedRangeProjector(
            projection_dim=5, projection_oversampling=1, n_power_iter=1)
        prepared = _prepare_phi(
            tensor, 'center', projector=projector,
            generator=torch.Generator().manual_seed(101),
            collect_metrics=True, site=2)
        reconstructed = torch.einsum(
            'ab,bpc,cd->apd', prepared.left, prepared.tensor, prepared.right)

        torch.testing.assert_close(reconstructed, tensor)
        assert len(prepared.projections) == 2

    def test_raw_boundary_keeps_original_tensor(self):
        tensor = torch.randn(1, 3, 2)
        prepared = _prepare_phi(tensor, None)
        assert prepared.tensor is tensor
        assert prepared.truncations == ()


class TestMarginalSketch:  # MARK: TestMarginalSketch

    @pytest.mark.parametrize('order', [1, 2, 4])
    def test_recursive_blocks_reproduce_both_sketch_directions(self, order):
        in_dim = (2, 3, 2, 4, 2)
        indices = torch.cartesian_prod(*(
            torch.arange(dim) for dim in in_dim))
        operator = tk.decompositions.MarginalSketch(order=order)
        weights = operator._weights(indices, in_dim, torch.float64, None)
        left = torch.ones(indices.shape[0], 1, dtype=torch.float64)
        for cut, block in enumerate(weights.left_blocks):
            selected = block[:, indices[:, cut], :].permute(1, 0, 2)
            left = torch.einsum('nab,nb->na', selected, left)
            torch.testing.assert_close(left, weights.left[cut])
        right = torch.ones(indices.shape[0], 1, dtype=torch.float64)
        for cut in reversed(range(len(in_dim) - 1)):
            block = weights.right_blocks[cut]
            selected = block[:, indices[:, cut + 1], :].permute(1, 0, 2)
            right = torch.einsum('nab,nb->na', selected, right)
            torch.testing.assert_close(right, weights.right[cut])

    @pytest.mark.parametrize('order', [True, 1.5, 0, -1])
    def test_invalid_order(self, order):
        with pytest.raises((TypeError, ValueError), match='order'):
            tk.decompositions.MarginalSketch(order=order)

    def test_invalid_factors(self):
        with pytest.raises(TypeError, match='factor'):
            tk.decompositions.MarginalSketch(factor=[torch.ones(2, 2)])
        source, _ = _rank_one_sparse()
        with pytest.raises(ValueError, match='factor'):
            tk.decompositions.MarginalSketch(
                factor=[torch.ones(2)]).builder(source).build()

    def test_markov_order_one_builds_neighbor_marginals(self):
        source, dense = _rank_one_sparse()
        system = tk.decompositions.MarginalSketch().builder(
            source).build()

        assert system.phi[0].shape == (1, 2, 3)
        assert system.phi[1].shape == (2, 3, 2)
        assert system.phi[2].shape == (3, 2, 1)
        assert torch.allclose(system.phi[0].squeeze(0), dense.sum(dim=2))
        assert torch.allclose(system.phi[1], dense)
        assert torch.allclose(system.phi[2].squeeze(-1), dense.sum(dim=0))
        assert system.diagnostics['marginal_order'] == 1

    def test_factor_changes_the_contracted_measure_explicitly(self):
        source, dense = _rank_one_sparse(in_dim=(2, 2, 2))
        factor = (
            torch.tensor([1., 2.]),
            torch.tensor([1., 3.]),
            torch.tensor([2., 1.]),
        )
        system = tk.decompositions.MarginalSketch(
            order=1, factor=factor).builder(source).build()
        weighted = torch.einsum(
            'ijk,i,j,k->ijk', dense, *factor)

        assert torch.allclose(system.phi[0].squeeze(0), weighted.sum(dim=2))
        assert system.diagnostics['weighted_marginal']


class TestSampledSketch:  # MARK: TestSampledSketch

    @pytest.mark.parametrize('options', [
        {'samples': torch.ones(2, 3)}, {'samples': torch.ones(3)},
        {'samples': torch.empty(0, 3, dtype=torch.long)},
        {'sketch_size': True}, {'sketch_size': 0}])
    def test_invalid_constructor(self, options):
        with pytest.raises((TypeError, ValueError)):
            tk.decompositions.SampledSketch(**options)

    def test_reference_indices_checked_against_source_dimensions(self):
        source, _ = _rank_one_sparse()
        for samples in (torch.tensor([[0, 0]]), torch.tensor([[2, 0, 0]])):
            with pytest.raises(ValueError, match='samples'):
                tk.decompositions.SampledSketch(
                    samples=samples).builder(source).build()

    def test_sparse_route_uses_support_without_source_evaluations(self):
        source, dense = _rank_one_sparse()
        system = tk.decompositions.SampledSketch().builder(source).build(
            batch_size=5)
        result = system.solve(rank=1, strict_system=True)

        assert source.evaluation_stats.source_calls == 0
        assert system.diagnostics['support_size'] == dense.numel()
        assert torch.allclose(result.contract_dense(), dense)

    def test_sample_cap_preserves_recursive_prefix_dimensions(self):
        source, _ = _rank_one_sparse(in_dim=(2, 2, 2, 2))
        system = tk.decompositions.SampledSketch(
            sketch_size=2).builder(source).build(
                generator=torch.Generator().manual_seed(201))

        assert all(dimension <= 2
                   for dimension in system.diagnostics['left_dimensions'])
        assert all(block.shape[0] <= 2 for block in system.left_blocks)


class TestTTStackSketch:  # MARK: TestTTStackSketch

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    def test_recursive_blocks_match_direct_partial_contractions(self, dtype):
        in_dim = (2, 3, 2, 3, 2)
        indices = torch.cartesian_prod(*(
            torch.arange(dim) for dim in in_dim))
        weights = tk.decompositions.TTStackSketch(
            tt_rank=2, stack_size=3)._weights(
                indices, in_dim, dtype, torch.Generator().manual_seed(23))
        left = torch.ones(indices.shape[0], 1, dtype=dtype)
        for cut, block in enumerate(weights.left_blocks):
            selected = block[:, indices[:, cut], :].permute(1, 0, 2)
            left = torch.einsum('nab,nb->na', selected, left)
            torch.testing.assert_close(left, weights.left[cut])
        right = torch.ones(indices.shape[0], 1, dtype=dtype)
        for cut in reversed(range(len(in_dim) - 1)):
            block = weights.right_blocks[cut]
            selected = block[:, indices[:, cut + 1], :].permute(1, 0, 2)
            right = torch.einsum('nab,nb->na', selected, right)
            torch.testing.assert_close(right, weights.right[cut])

    @pytest.mark.parametrize('options', [
        {'tt_rank': True}, {'tt_rank': 0}, {'stack_size': 1.5},
        {'stack_size': -1}, {'orthogonal': 1}])
    def test_invalid_constructor(self, options):
        with pytest.raises((TypeError, ValueError)):
            tk.decompositions.TTStackSketch(**options)

    @pytest.mark.parametrize(
        ('tt_rank', 'stack_size'), [(1, 3), (3, 1), (2, 2)])
    def test_partial_sketch_dimension_is_stack_times_tt_rank(
            self, tt_rank, stack_size):
        source, _ = _rank_one_sparse()
        system = tk.decompositions.TTStackSketch(
            tt_rank=tt_rank,
            stack_size=stack_size).builder(source).build(
                generator=torch.Generator().manual_seed(202))
        expected = tt_rank * stack_size

        assert system.diagnostics['sketch_dimension'] == expected
        assert system.diagnostics['left_dimensions'] == (expected, expected)
        assert system.diagnostics['right_dimensions'] == (expected, expected)
        assert system.diagnostics['core_determining_gate'] == \
            'recursive_left_and_suffix_right'

    def test_gaussian_tt_stack_recovers_rank_one_sparse_tensor(self):
        source, dense = _rank_one_sparse()
        system = tk.decompositions.TTStackSketch(
            tt_rank=2,
            stack_size=2).builder(source).build(
                generator=torch.Generator().manual_seed(203))
        result = system.solve(rank=1, strict_system=True)

        assert torch.allclose(
            result.contract_dense(), dense, rtol=1e-10, atol=1e-12)
        assert result.metadata['coefficient_ranks'] == [1, None, 1]

    def test_orthogonal_variant_is_labelled_separately(self):
        source, _ = _rank_one_sparse(in_dim=(3, 3, 3))
        system = tk.decompositions.TTStackSketch(
            tt_rank=2,
            stack_size=1,
            orthogonal=True).builder(source).build(
                generator=torch.Generator().manual_seed(204))

        assert system.diagnostics['orthogonal']
        assert system.diagnostics['distribution'] == 'orthogonal_variant'


class TestCoreDeterminingSystem:  # MARK: TestCoreDeterminingSystem

    @pytest.mark.parametrize('in_dim', [(3,), (2, 3), (2, 3, 2, 3, 2)])
    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('kind', ['sampled', 'marginal', 'tt_stack'])
    def test_centered_reconstruction_on_minimal_and_general_chains(
            self, in_dim, dtype, kind):
        ranks = (1, *((2,) * (len(in_dim) - 1)), 1)
        cores = [torch.randn(ranks[k], dimension, ranks[k + 1], dtype=dtype)
                 for k, dimension in enumerate(in_dim)]
        dense = cores[0]
        for core in cores[1:]:
            dense = torch.tensordot(dense, core, dims=([-1], [0]))
        dense = dense.reshape(in_dim)
        source = tk.decompositions.DenseTensorSource(dense)
        if kind == 'sampled':
            operator = tk.decompositions.SampledSketch()
        elif kind == 'marginal':
            operator = tk.decompositions.MarginalSketch(order=2)
        else:
            operator = tk.decompositions.TTStackSketch(
                tt_rank=2, stack_size=3)
        system = operator.builder(source).build(
            generator=torch.Generator().manual_seed(32))
        for center in sorted(set((0, len(in_dim) // 2, len(in_dim) - 1))):
            for rank in (2, (2,) * (len(in_dim) - 1), None):
                result = system.solve(rank=rank, center=center)
                torch.testing.assert_close(
                    result.contract_dense(), dense, rtol=1e-8, atol=1e-9)
                assert len(result.cores) == len(in_dim)
                assert result.metadata['center'] == center

    def test_sketch_quality_can_violate_neighbor_rank_inequality(self):
        # Phi0 sums x3 and loses the correction, while Phi1 retains x3.
        h = torch.tensor([-1., 1.], dtype=torch.float64)
        matrix = torch.tensor([[1., 2.], [3., 5.]], dtype=torch.float64)
        dense = torch.ones((2,) * 5, dtype=torch.float64)
        dense = dense + torch.einsum(
            'i,jk,l,m->ijklm', h, matrix, h, torch.ones_like(h))
        source = tk.decompositions.DenseTensorSource(dense)
        system = tk.decompositions.MarginalSketch(order=2).builder(
            source).build()
        result = system.solve(center=0)

        assert result.rank[0] == 1
        assert result.rank[1] == 3
        assert result.rank[1] > dense.shape[1] * result.rank[0]
        assert torch.isfinite(result.contract_dense()).all()

    @pytest.mark.parametrize('center', [0, 2, 4])
    def test_single_svd_decides_each_cut_and_records_remain_in_site_order(
            self, center):
        source, _ = _rank_one_sparse(in_dim=(2, 3, 2, 3, 2))
        system = tk.decompositions.MarginalSketch().builder(source).build()
        events = []
        result = system.solve(rank=1, center=center, collect_metrics=True,
                              _on_core=lambda site, _: events.append(site))

        assert [record.site for record in result.metrics.truncations] == \
            list(range(4))
        assert [record.site for record in result.metrics.local_solves] == \
            [site for site in range(5) if site != center]
        assert events == [center, *range(center + 1, 5),
                          *range(center - 1, -1, -1)]

    def test_transform_can_promote_real_targets_to_complex(self):
        source, dense = _rank_one_sparse()
        system = tk.decompositions.MarginalSketch().builder(source).build()
        calls = []

        def transform(phi):
            calls.append(phi.shape)
            return phi.to(torch.complex128) * (1 + 2j)

        result = system.solve(rank=1, transform=transform)
        assert len(calls) == 3
        assert result.dtype == torch.complex128
        torch.testing.assert_close(result.contract_dense(), dense * (1 + 2j))

    @pytest.mark.parametrize('transform', [
        1, lambda phi: None, lambda phi: phi.flatten(),
        lambda phi: phi * float('nan'), lambda phi: phi.long()])
    def test_invalid_transform(self, transform):
        source, _ = _rank_one_sparse()
        system = tk.decompositions.MarginalSketch().builder(source).build()
        with pytest.raises((TypeError, ValueError), match='transform'):
            system.solve(rank=1, transform=transform)

    @pytest.mark.parametrize('options', [
        {'rank': True}, {'rank': [1]}, {'rank': 0}, {'center': True},
        {'center': -1}, {'center': 3}, {'strict_system': 1},
        {'collect_metrics': 1}, {'rtol': -1}, {'range_projector': 1},
        {'generator': 1}])
    def test_invalid_solve_options(self, options):
        source, _ = _rank_one_sparse()
        system = tk.decompositions.MarginalSketch().builder(source).build()
        with pytest.raises((TypeError, ValueError)):
            system.solve(**options)

    @pytest.mark.parametrize('options', [
        {'phi': [torch.ones(2, 2), torch.ones(1, 2, 1)]},
        {'phi': [torch.ones(1, 2, 1)]},
        {'phi': [torch.ones(2, 2, 1), torch.ones(1, 2, 1)]},
        {'phi': [torch.ones(1, 3, 1), torch.ones(1, 2, 1)]},
        {'left_blocks': []}, {'right_blocks': [torch.ones(2, 2, 1)]},
        {'in_dim': (2, True)},
    ])
    def test_invalid_system_dimensions(self, options):
        defaults = {'phi': [torch.ones(1, 2, 1), torch.ones(1, 2, 1)],
                    'left_blocks': [torch.ones(1, 2, 1)],
                    'right_blocks': [torch.ones(1, 2, 1)],
                    'in_dim': (2, 2), 'operator': 'test'}
        with pytest.raises((TypeError, ValueError)):
            tk.decompositions.CoreDeterminingSystem(**(defaults | options))

    def test_strict_system_rejects_unidentifiable_rank(self):
        system = tk.decompositions.CoreDeterminingSystem(
            phi=(
                torch.ones(1, 2, 1),
                torch.ones(1, 2, 1),
            ),
            left_blocks=(torch.zeros(1, 2, 1),),
            right_blocks=(torch.zeros(1, 2, 1),),
            in_dim=(2, 2),
            operator='degenerate_test')

        with pytest.raises(ValueError, match='rank deficient'):
            system.solve(rank=1, strict_system=True)

    def test_metrics_are_only_collected_when_requested(self):
        source, _ = _rank_one_sparse()
        system = tk.decompositions.MarginalSketch().builder(
            source).build()
        fast = system.solve(rank=1)
        measured = system.solve(rank=1, collect_metrics=True)

        assert not fast.metrics.truncations
        assert not fast.metrics.local_solves
        assert fast.metadata['coefficient_ranks'] is None
        assert len(measured.metrics.truncations) == 2
        assert len(measured.metrics.local_solves) == 2


class TestStructuredTTBackend:  # MARK: TestStructuredTTBackend

    @pytest.mark.parametrize('dtype', [torch.float64, torch.complex128])
    @pytest.mark.parametrize('operator', [
        tk.decompositions.SampledSketch(
            samples=torch.tensor([[0, 0, 0], [1, 1, 1]])),
        tk.decompositions.MarginalSketch(),
        tk.decompositions.TTStackSketch(tt_rank=2, stack_size=2),
    ])
    def test_structured_system_matches_sparse_dense_oracle(
            self, dtype, operator):
        generator = torch.Generator().manual_seed(205)
        cores = [
            torch.randn(1, 2, 2, dtype=dtype, generator=generator),
            torch.randn(2, 2, 2, dtype=dtype, generator=generator),
            torch.randn(2, 2, 1, dtype=dtype, generator=generator),
        ]
        tt_source = tk.decompositions.FormatSource(cores)
        dense = tk.decompositions.TTDecomposition([
            cores[0].squeeze(0), cores[1], cores[2].squeeze(-1)
        ]).contract_dense()
        indices = torch.cartesian_prod(*(
            torch.arange(dimension) for dimension in dense.shape))
        oracle_source = tk.decompositions.SparseTensorSource(
            indices, dense.reshape(-1), dense.shape)

        structured = operator.builder(tt_source).build(
            generator=torch.Generator().manual_seed(206))
        oracle = operator.builder(oracle_source).build(
            generator=torch.Generator().manual_seed(206))

        assert structured.diagnostics['source_path'] == 'structured_tt'
        assert tt_source.evaluation_stats.requested_points == 0
        assert all(torch.allclose(left, right)
                   for left, right in zip(structured.phi, oracle.phi))
        assert all(torch.allclose(left, right)
                   for left, right in zip(
                       structured.left_blocks, oracle.left_blocks))
        assert all(torch.allclose(left, right)
                   for left, right in zip(
                       structured.right_blocks, oracle.right_blocks))


__all__ = []
