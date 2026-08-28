"""Tests for identity and randomized sketch range projections."""

import pytest

import torch
import tensorkrowch as tk

from tensorkrowch.decompositions.sketching import projections as projection_module
from tensorkrowch.decompositions.sketching.projections import ProjectedRange
from tensorkrowch.utils import random_unitary, truncated_svd


def _generator(seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _low_rank_matrix(n_rows=10, n_columns=8, rank=2, dtype=torch.float64):
    generator = _generator(123)
    left = torch.randn(
        n_rows, rank, generator=generator, dtype=dtype)
    right = torch.randn(
        rank, n_columns, generator=generator, dtype=dtype)
    return left @ right


class TestRangeProjectionContracts:  # MARK: TestRangeProjectionContracts

    def test_legacy_square_randu_only_rotates_the_right_basis(self):
        matrix = torch.randn(7, 5, generator=_generator(1), dtype=torch.float64)
        unitary = random_unitary(
            5, dtype=torch.float64, generator=_generator(2))
        rotated = matrix @ unitary

        assert torch.allclose(
            rotated @ rotated.mH, matrix @ matrix.mH, atol=1e-12)
        assert torch.allclose(
            torch.linalg.svdvals(rotated),
            torch.linalg.svdvals(matrix),
            atol=1e-12)

    def test_identity_preserves_the_exact_matrix_and_storage(self):
        matrix = torch.randn(5, 4)
        projector = tk.decompositions.IdentityRangeProjector()

        projected = projector.project(matrix)

        assert isinstance(projector, tk.decompositions.RangeProjector)
        assert isinstance(projected, ProjectedRange)
        assert projected.basis is None
        assert projected.small_matrix is matrix
        assert projected.reconstruct() is matrix
        assert projected.record is None

    def test_identity_diagnostics_are_exact_and_enter_shared_metrics(self):
        matrix = torch.randn(5, 4)

        projected = tk.decompositions.IdentityRangeProjector().project(
            matrix, rank=2, return_info=True)

        assert projected.record.method == 'identity'
        assert projected.record.requested_dim == 2
        assert projected.record.projection_dim == 4
        assert projected.record.error_absolute == 0.
        assert projected.record.error_relative == 0.
        assert projected.record.elapsed >= 0.
        info = tk.decompositions.DecompositionMetrics(
            range_projections=[projected.record]).as_info()
        assert info['range_projections'][0]['method'] == 'identity'

    def test_projection_dimension_defaults_to_rank_and_oversamples(self):
        matrix = torch.randn(12, 9, dtype=torch.float64)
        projector = tk.decompositions.RandomizedRangeProjector(
            projection_oversampling=2)

        projected = projector.project(
            matrix, rank=3, generator=_generator(4), return_info=True)

        assert projected.record.requested_dim == 3
        assert projected.record.projection_dim == 5
        assert projected.record.range_dim == 5
        assert projected.record.oversampling == 2

    def test_rank_none_uses_a_square_non_reducing_random_map(self):
        matrix = torch.randn(7, 5, generator=_generator(5), dtype=torch.float64)

        projected = tk.decompositions.RandomizedRangeProjector().project(
            matrix, generator=_generator(6), return_info=True)

        assert projected.record.requested_dim is None
        assert projected.record.projection_dim == matrix.shape[1]
        assert torch.allclose(projected.reconstruct(), matrix, atol=1e-12)

    def test_explicit_projection_dimension_takes_precedence_over_rank(self):
        matrix = torch.randn(8, 7, dtype=torch.float64)
        projector = tk.decompositions.RandomizedRangeProjector(
            projection_dim=4)

        projected = projector.project(
            matrix, rank=2, generator=_generator(7), return_info=True)

        assert projected.record.requested_dim == 4
        assert projected.record.projection_dim == 4


class TestRandomizedRangeFinder:  # MARK: TestRandomizedRangeFinder

    def test_exact_low_rank_range_and_small_svd_lifting(self):
        matrix = _low_rank_matrix()
        projected = tk.decompositions.RandomizedRangeProjector(
            projection_dim=2).project(matrix, generator=_generator(8))
        u_small, s, vh = truncated_svd(
            projected.small_matrix, rank=2)
        u = projected.lift_left(u_small)
        reconstructed = (u * s.unsqueeze(0)) @ vh

        assert torch.allclose(
            projected.basis.mH @ projected.basis,
            torch.eye(2, dtype=matrix.dtype),
            atol=1e-12)
        assert torch.allclose(reconstructed, matrix, atol=1e-11)

    def test_projection_axis_is_replaced_by_the_lifted_rank(self):
        flat = _low_rank_matrix(n_rows=6, n_columns=5, rank=2)
        tensor = flat.reshape(2, 3, 5).movedim(-1, 1)
        projected = tk.decompositions.RandomizedRangeProjector(
            projection_dim=2).project(
                tensor, axis=1, generator=_generator(9))
        u_small, _, _ = truncated_svd(projected.small_matrix, rank=2)
        restored_u = projected.restore_left(u_small)

        assert tensor.shape == (2, 5, 3)
        assert projected.reconstruct().shape == tensor.shape
        assert restored_u.shape == (2, 2, 3)
        assert torch.allclose(projected.reconstruct(), tensor, atol=1e-11)

    def test_complex_projection_is_deterministic_with_a_generator(self):
        real = _low_rank_matrix(rank=3)
        imag = _low_rank_matrix(rank=3)
        matrix = torch.complex(real, imag)
        projector = tk.decompositions.RandomizedRangeProjector(
            projection_dim=3)

        first = projector.project(matrix, generator=_generator(10))
        second = projector.project(matrix, generator=_generator(10))

        assert first.basis.dtype == torch.complex128
        assert torch.equal(first.basis, second.basis)
        assert torch.equal(first.small_matrix, second.small_matrix)
        assert torch.allclose(first.reconstruct(), matrix, atol=1e-11)

    def test_power_iterations_do_not_worsen_projection_error(self):
        generator = _generator(11)
        left = torch.linalg.qr(torch.randn(
            20, 8, generator=generator, dtype=torch.float64)).Q
        right = torch.linalg.qr(torch.randn(
            12, 8, generator=generator, dtype=torch.float64)).Q
        singular_values = torch.logspace(
            0, -5, 8, dtype=torch.float64)
        matrix = (left * singular_values.unsqueeze(0)) @ right.mH
        plain = tk.decompositions.RandomizedRangeProjector(
            projection_dim=2).project(
                matrix, generator=_generator(12), return_info=True)
        iterated = tk.decompositions.RandomizedRangeProjector(
            projection_dim=2, n_power_iter=2).project(
                matrix, generator=_generator(12), return_info=True)

        assert iterated.record.error_relative <= \
            plain.record.error_relative + 1e-12

    def test_small_svd_reduces_rows_and_tracks_direct_rank_error(self):
        generator = _generator(15)
        left = torch.linalg.qr(torch.randn(
            40, 20, generator=generator, dtype=torch.float64)).Q
        right = torch.linalg.qr(torch.randn(
            30, 20, generator=generator, dtype=torch.float64)).Q
        singular_values = torch.logspace(
            0, -4, 20, dtype=torch.float64)
        matrix = (left * singular_values.unsqueeze(0)) @ right.mH
        exact_u, exact_s, exact_vh = truncated_svd(matrix, rank=5)
        exact = (exact_u * exact_s.unsqueeze(0)) @ exact_vh
        projected = tk.decompositions.RandomizedRangeProjector(
            projection_dim=5,
            projection_oversampling=5,
            n_power_iter=1).project(matrix, generator=_generator(16))
        small_u, small_s, small_vh = truncated_svd(
            projected.small_matrix, rank=5)
        approximate = (projected.lift_left(small_u) *
                       small_s.unsqueeze(0)) @ small_vh
        exact_error = torch.linalg.vector_norm(matrix - exact)
        approximate_error = torch.linalg.vector_norm(matrix - approximate)

        assert projected.small_matrix.shape == (10, 30)
        assert projected.small_matrix.shape[0] < matrix.shape[0]
        assert approximate_error <= 1.1 * exact_error

    def test_no_diagnostics_skips_error_and_synchronized_timer(
            self, monkeypatch):
        matrix = torch.randn(6, 4)

        def fail_error(*args, **kwargs):
            raise AssertionError('Projection error should not be computed')

        monkeypatch.setattr(
            projection_module, '_projection_error', fail_error)

        projected = tk.decompositions.RandomizedRangeProjector(
            projection_dim=2).project(
                matrix, generator=_generator(13), return_info=False)

        assert projected.record is None


_DEVICES = ['cpu']
if torch.cuda.is_available():
    _DEVICES.append('cuda')
if torch.backends.mps.is_available():
    _DEVICES.append('mps')


@pytest.mark.parametrize('device', _DEVICES)
def test_range_projection_runs_on_available_compute_devices(device):
    matrix = _low_rank_matrix(dtype=torch.float32).to(device)
    projected = tk.decompositions.RandomizedRangeProjector(
        projection_dim=2).project(matrix, generator=_generator(14))

    assert projected.small_matrix.device.type == device
    assert projected.basis.device.type == device
    assert torch.allclose(projected.reconstruct(), matrix, atol=1e-4)


@pytest.mark.parametrize(
    'projector, kwargs, error, match',
    [
        (tk.decompositions.RandomizedRangeProjector,
         {'projection_dim': 0}, ValueError, 'positive'),
        (tk.decompositions.RandomizedRangeProjector,
         {'projection_oversampling': -1}, ValueError, 'non-negative'),
        (tk.decompositions.RandomizedRangeProjector,
         {'n_power_iter': 1.5}, TypeError, 'int'),
    ])
def test_invalid_projector_options(projector, kwargs, error, match):
    with pytest.raises(error, match=match):
        projector(**kwargs)


@pytest.mark.parametrize(
    'kwargs, error, match',
    [
        ({'rank': 0}, ValueError, 'positive'),
        ({'axis': 3}, ValueError, 'bounds'),
        ({'generator': 1}, TypeError, 'Generator'),
        ({'return_info': 1}, TypeError, 'bool'),
    ])
def test_invalid_projection_call_options(kwargs, error, match):
    with pytest.raises(error, match=match):
        tk.decompositions.RandomizedRangeProjector().project(
            torch.ones(3, 2), **kwargs)
