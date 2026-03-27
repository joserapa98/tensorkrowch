"""
Tests for embeddings:

    * TestEmbeddings
"""

import pytest

import torch
import tensorkrowch as tk

SIZE_CASES = [1, 10, 20]
BATCH_SIZE_CASES = [20, 100]
DIM_CASES = [1, 2, 3, 5]
DEGREE_CASES = [1, 2, 3, 5]
LEVEL_CASES = [1, 2, 3, 5]
BASE_CASES = [2, 4, 5]


class TestEmbeddings:  # MARK: TestEmbeddings

    @pytest.mark.parametrize('size', SIZE_CASES)
    @pytest.mark.parametrize('dim', DIM_CASES)
    def test_unit(self, size, dim):
        sample = torch.randn(size)
        embedded_sample = tk.embeddings.unit(data=sample, dim=dim)

        assert embedded_sample.shape == (size, dim)

    @pytest.mark.parametrize('batch_size', BATCH_SIZE_CASES)
    @pytest.mark.parametrize('size', SIZE_CASES)
    @pytest.mark.parametrize('dim', DIM_CASES)
    def test_unit_batch(self, batch_size, size, dim):
        sample = torch.randn(batch_size, size)
        embedded_sample = tk.embeddings.unit(data=sample, dim=dim)

        assert embedded_sample.shape == (batch_size, size, dim)

    @pytest.mark.parametrize('size', SIZE_CASES)
    def test_add_ones(self, size):
        sample = torch.randn(size)
        embedded_sample = tk.embeddings.add_ones(data=sample)

        assert embedded_sample.shape == (size, 2)

    @pytest.mark.parametrize('batch_size', BATCH_SIZE_CASES)
    @pytest.mark.parametrize('size', SIZE_CASES)
    def test_add_ones_batch(self, batch_size, size):
        sample = torch.randn(batch_size, size)
        embedded_sample = tk.embeddings.add_ones(data=sample)

        assert embedded_sample.shape == (batch_size, size, 2)

    @pytest.mark.parametrize('size', SIZE_CASES)
    @pytest.mark.parametrize('degree', DEGREE_CASES)
    def test_poly(self, size, degree):
        sample = torch.randn(size)
        embedded_sample = tk.embeddings.poly(data=sample, degree=degree)

        assert embedded_sample.shape == (size, degree + 1)

    @pytest.mark.parametrize('batch_size', BATCH_SIZE_CASES)
    @pytest.mark.parametrize('size', SIZE_CASES)
    @pytest.mark.parametrize('degree', DEGREE_CASES)
    def test_poly_batch(self, batch_size, size, degree):
        sample = torch.randn(batch_size, size)
        embedded_sample = tk.embeddings.poly(data=sample, degree=degree)

        assert embedded_sample.shape == (batch_size, size, degree + 1)

    @pytest.mark.parametrize('size', SIZE_CASES)
    @pytest.mark.parametrize('level', LEVEL_CASES)
    @pytest.mark.parametrize('base', BASE_CASES)
    def test_discretize(self, size, level, base):
        sample = torch.rand(size)
        embedded_sample = tk.embeddings.discretize(data=sample,
                                                   level=level,
                                                   base=base)

        assert embedded_sample.shape == (size, level)

    @pytest.mark.parametrize('batch_size', BATCH_SIZE_CASES)
    @pytest.mark.parametrize('size', SIZE_CASES)
    @pytest.mark.parametrize('level', LEVEL_CASES)
    @pytest.mark.parametrize('base', BASE_CASES)
    def test_discretize_batch(self, batch_size, size, level, base):
        sample = torch.rand(batch_size, size)
        embedded_sample = tk.embeddings.discretize(data=sample,
                                                   level=level,
                                                   base=base)

        assert embedded_sample.shape == (batch_size, size, level)

    @pytest.mark.parametrize('size', SIZE_CASES)
    @pytest.mark.parametrize('dim', DIM_CASES)
    def test_basis(self, size, dim):
        sample = torch.randint(low=0, high=dim, size=(size,))
        embedded_sample = tk.embeddings.basis(data=sample, dim=dim)

        assert embedded_sample.shape == (size, dim)

    @pytest.mark.parametrize('batch_size', BATCH_SIZE_CASES)
    @pytest.mark.parametrize('size', SIZE_CASES)
    @pytest.mark.parametrize('dim', DIM_CASES)
    def test_basis_batch(self, batch_size, size, dim):
        sample = torch.randint(low=0, high=dim, size=(batch_size, size))
        embedded_sample = tk.embeddings.basis(data=sample, dim=dim)

        assert embedded_sample.shape == (batch_size, size, dim)
