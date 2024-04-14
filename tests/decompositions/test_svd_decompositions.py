"""
Tests for svd decompositions:

    * TestSVDDecompositions
"""

import pytest

import torch
import tensorkrowch as tk


class TestSVDDecompositions:
    
    def test_vec_to_mps(self):
        for i in range(1, 6):
            dims = torch.randint(low=2, high=10, size=(i,))
            vec = torch.randn(*dims) * 1e-5
            for j in range(i + 1):
                tensors = tk.decompositions.vec_to_mps(vec=vec,
                                                       n_batches=j,
                                                       rank=5)
                
                for k, tensor in enumerate(tensors):
                    assert tensor.shape[:j] == vec.shape[:j]
                    if k == 0:
                        if j < i:
                            assert tensor.shape[j] == vec.shape[j]
                        if j < i - 1:
                            assert tensor.shape[j + 1] <= 5
                    elif k < (i - j - 1):
                        assert tensor.shape[j + 1] == vec.shape[j + k]
                        assert tensor.shape[j + 2] <= 5
                    else:
                        assert tensor.shape[j + 1] == vec.shape[j + k]
                        
                bond_dims = [tensor.shape[-1] for tensor in tensors[:-1]]
                
                if i - j > 0:
                    mps = tk.models.MPSData(tensors=tensors,
                                            n_batches=j)
                    assert mps.phys_dim == dims[j:].tolist()
                    assert mps.bond_dim == bond_dims
                
                if j == 0:
                    mps = tk.models.MPS(tensors=tensors)
                    assert mps.phys_dim == dims.tolist()
                    assert mps.bond_dim == bond_dims
    
    def test_mat_to_mpo(self):
        for i in range(1, 6):
            dims = torch.randint(low=2, high=10, size=(2 * i,))
            mat = torch.randn(*dims) * 1e-5
            
            tensors = tk.decompositions.mat_to_mpo(mat=mat, rank=5)
            
            for k, tensor in enumerate(tensors):
                if k == 0:
                    assert tensor.shape[0] == dims[2 * k]
                    if i > 1:
                        assert tensor.shape[1] <= 5
                    assert tensor.shape[-1] == dims[2 * k + 1]
                elif k < (i - 1):
                    assert tensor.shape[1] == dims[2 * k]
                    assert tensor.shape[2] <= 5
                    assert tensor.shape[3] == dims[2 * k + 1]
                else:
                    assert tensor.shape[-2] == dims[2 * k]
                    assert tensor.shape[-1] == dims[2 * k + 1]
            
            bond_dims = [tensor.shape[-2] for tensor in tensors[:-1]]
                    
            mpo = tk.models.MPO(tensors=tensors)
            assert mpo.in_dim == [dims[2 * j] for j in range(i)]
            assert mpo.out_dim == [dims[2 * j + 1] for j in range(i)]
            assert mpo.bond_dim == bond_dims
          