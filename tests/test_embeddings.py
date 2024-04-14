"""
Tests for embeddings:

    * TestEmbeddings
"""

import pytest

import torch
import tensorkrowch as tk


class TestEmbeddings:
    
    def test_unit(self):
        for size in [1, 10, 100]:
            for dim in [1, 2, 3, 10]:
                sample = torch.randn(size)
                embedded_sample = tk.embeddings.unit(data=sample,
                                                     dim=dim)
                
                assert embedded_sample.shape == (size, dim)
    
    def test_unit_batch(self):
        for batch_size in [100, 1000]:
            for size in [1, 10, 100]:
                for dim in [1, 2, 3, 10]:
                    sample = torch.randn(batch_size, size)
                    embedded_sample = tk.embeddings.unit(data=sample,
                                                         dim=dim)
                    
                    assert embedded_sample.shape == (batch_size, size, dim)
    
    def test_add_ones(self):
        for size in [1, 10, 100]:
            sample = torch.randn(size)
            embedded_sample = tk.embeddings.add_ones(data=sample)
            
            assert embedded_sample.shape == (size, 2)
    
    def test_add_ones_batch(self):
        for batch_size in [100, 1000]:
            for size in [1, 10, 100]:
                sample = torch.randn(batch_size, size)
                embedded_sample = tk.embeddings.add_ones(data=sample)
                
                assert embedded_sample.shape == (batch_size, size, 2)
    
    def test_poly(self):
        for size in [1, 10, 100]:
            for degree in [1, 2, 3, 10]:
                sample = torch.randn(size)
                embedded_sample = tk.embeddings.poly(data=sample,
                                                     degree=degree)
                
                assert embedded_sample.shape == (size, degree + 1)
    
    def test_poly_batch(self):
        for batch_size in [100, 1000]:
            for size in [1, 10, 100]:
                for degree in [1, 2, 3, 10]:
                    sample = torch.randn(batch_size, size)
                    embedded_sample = tk.embeddings.poly(data=sample,
                                                         degree=degree)
                    
                    assert embedded_sample.shape == (batch_size, size, degree + 1)
    
    def test_discretize(self):
        for size in [1, 10, 100]:
            for level in [1, 2, 3, 10]:
                for base in [2, 4, 10]:
                    sample = torch.rand(size)
                    embedded_sample = tk.embeddings.discretize(data=sample,
                                                               level=level,
                                                               base=base)
                    
                    assert embedded_sample.shape == (size, level)
    
    def test_discretize_batch(self):
        for batch_size in [100, 1000]:
            for size in [1, 10, 100]:
                for level in [1, 2, 3, 10]:
                    for base in [2, 4, 10]:
                        sample = torch.rand(batch_size, size)
                        embedded_sample = tk.embeddings.discretize(data=sample,
                                                                   level=level,
                                                                   base=base)
                        
                        assert embedded_sample.shape == (batch_size, size, level)
    
    def test_basis(self):
        for size in [1, 10, 100]:
            for dim in [1, 2, 3, 10]:
                sample = torch.randint(low=0, high=dim, size=(size,))
                embedded_sample = tk.embeddings.basis(data=sample,
                                                           dim=dim)
                
                assert embedded_sample.shape == (size, dim)
    
    def test_basis_batch(self):
        for batch_size in [100, 1000]:
            for size in [1, 10, 100]:
                for dim in [1, 2, 3, 10]:
                    sample = torch.randint(low=0, high=dim, size=(batch_size,
                                                                  size))
                    embedded_sample = tk.embeddings.basis(data=sample,
                                                               dim=dim)
                    
                    assert embedded_sample.shape == (batch_size, size, dim)
