"""Tests for svd/_matrix."""

import pytest
import torch

from tensorkrowch.decompositions.svd._matrix import _prepare_matrix_input


@pytest.mark.parametrize('axis_layout', ['grouped', 'interleaved'])
def test_matrix_input_preserves_pairing_and_gradients(axis_layout):
    matrix = torch.arange(24, dtype=torch.float64).reshape(4, 6).requires_grad_()
    grouped = matrix.reshape(2, 2, 2, 3)
    tensor = grouped if axis_layout == 'grouped' else grouped.permute(0, 2, 1, 3)
    result = _prepare_matrix_input(tensor, (2, 2), (2, 3), axis_layout, 'TTM')
    torch.testing.assert_close(result.interleaved, grouped.permute(0, 2, 1, 3))
    result.fused.sum().backward()
    torch.testing.assert_close(matrix.grad, torch.ones_like(matrix))
