"""
Tests for svd decompositions:

    * TestSVDDecompositions
"""

import pytest

import torch
import tensorkrowch as tk


class TestSVDDecompositions:
    
    def test_vec_to_mps(self):
        for renormalize in [True, False]:
            for i in range(1, 6):
                dims = torch.randint(low=2, high=10, size=(i,))
                vec = torch.randn(*dims) * 1e-5
                for j in range(i + 1):
                    tensors = tk.decompositions.vec_to_mps(vec=vec,
                                                           n_batches=j,
                                                           rank=5,
                                                           renormalize=renormalize)
                    
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
    
    def test_vec_to_mps_accuracy(self):
        for renormalize in [True, False]:
            for i in range(1, 6):
                dims = torch.randint(low=2, high=10, size=(i,))
                vec = torch.randn(*dims) * 1e-1
                for j in range(i + 1):
                    tensors = tk.decompositions.vec_to_mps(vec=vec,
                                                           n_batches=j,
                                                           cum_percentage=0.9999,
                                                           renormalize=renormalize)
                            
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
                    
                    approx_vec = mps.left_node
                    for node in mps.mats_env + [mps.right_node]:
                        approx_vec @= node
                    
                    approx_vec = approx_vec.tensor
                    diff = vec - approx_vec
                    assert diff.norm() < 1e-1
    
    def test_mat_to_mpo(self):
        for renormalize in [True, False]:
            for i in range(1, 6):
                dims = torch.randint(low=2, high=10, size=(2 * i,))
                mat = torch.randn(*dims) * 1e-5
                
                tensors = tk.decompositions.mat_to_mpo(mat=mat,
                                                       rank=5,
                                                       renormalize=renormalize)
                
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

    def test_mat_to_mpo_accuracy(self):
        for renormalize in [True, False]:
            for i in range(1, 6):
                dims = torch.randint(low=2, high=10, size=(2 * i,))
                mat = torch.randn(*dims) * 1e-1
                
                tensors = tk.decompositions.mat_to_mpo(mat=mat,
                                                       cum_percentage=0.9999,
                                                       renormalize=renormalize)
                bond_dims = [tensor.shape[-2] for tensor in tensors[:-1]]
                        
                mpo = tk.models.MPO(tensors=tensors)
                assert mpo.in_dim == [dims[2 * j] for j in range(i)]
                assert mpo.out_dim == [dims[2 * j + 1] for j in range(i)]
                assert mpo.bond_dim == bond_dims
                
                approx_mat = mpo.left_node
                for node in mpo.mats_env + [mpo.right_node]:
                    approx_mat @= node
                
                approx_mat = approx_mat.tensor
                diff = mat - approx_mat
                assert diff.norm() < 1e-1
    
    def test_mat_to_mpo_permuted_accuracy(self):
        for renormalize in [True, False]:
            dims = [2, 2, 2, 2, 3, 3, 3, 3]  # inputs (2) x outputs (3)
            mat = torch.randn(*dims) * 1e-1
            aux_mat = mat.permute(0, 4, 1, 5, 2, 6, 3, 7)
            assert aux_mat.shape == (2, 3, 2, 3, 2, 3, 2, 3)
            
            tensors = tk.decompositions.mat_to_mpo(mat=aux_mat,
                                                   cum_percentage=0.9999,
                                                   renormalize=renormalize)
            bond_dims = [tensor.shape[-2] for tensor in tensors[:-1]]
                    
            mpo = tk.models.MPO(tensors=tensors)
            assert mpo.in_dim == [2] * 4
            assert mpo.out_dim == [3] * 4
            assert mpo.bond_dim == bond_dims
            
            approx_mat = mpo.left_node
            for node in mpo.mats_env + [mpo.right_node]:
                approx_mat @= node
            
            assert approx_mat.shape == (2, 3, 2, 3, 2, 3, 2, 3)
            
            approx_mat = approx_mat.tensor
            approx_mat = approx_mat.permute(0, 2, 4, 6, 1, 3, 5, 7)
            diff = mat - approx_mat
            assert diff.norm() < 1e-1

    def test_mat_to_mpo_permuted_diff_dimsaccuracy(self):
        for renormalize in [True, False]:
            # inputs (2, 3, 4, 2) x outputs (3, 5, 7, 2)
            dims = [2, 3, 4, 2, 3, 5, 7, 2]
            mat = torch.randn(*dims) * 1e-1
            aux_mat = mat.permute(0, 4, 1, 5, 2, 6, 3, 7)
            assert aux_mat.shape == (2, 3, 3, 5, 4, 7, 2, 2)
            
            tensors = tk.decompositions.mat_to_mpo(mat=aux_mat,
                                                   cum_percentage=0.9999,
                                                   renormalize=renormalize)
            bond_dims = [tensor.shape[-2] for tensor in tensors[:-1]]
                    
            mpo = tk.models.MPO(tensors=tensors)
            assert mpo.in_dim == [2, 3, 4, 2]
            assert mpo.out_dim == [3, 5, 7, 2]
            assert mpo.bond_dim == bond_dims
            
            approx_mat = mpo.left_node
            for node in mpo.mats_env + [mpo.right_node]:
                approx_mat @= node
            
            assert approx_mat.shape == (2, 3, 3, 5, 4, 7, 2, 2)
            
            approx_mat = approx_mat.tensor
            approx_mat = approx_mat.permute(0, 2, 4, 6, 1, 3, 5, 7)
            diff = mat - approx_mat
            assert diff.norm() < 1e-1
