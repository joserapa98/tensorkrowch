"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tk

from typing import Sequence
import time


class TestMPSLayer:
    
    def test_all_algorithms(self):
        example = torch.randn(4, 1, 5)
        data = torch.randn(4, 100, 5)
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                for l_position in range(5):
                    mps = tk.MPSLayer(n_sites=5,
                                      d_phys=5,
                                      n_labels=12,
                                      d_bond=2,
                                      l_position=l_position,
                                      boundary=boundary,
                                      param_bond=param_bond)
                    
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    mps.automemory = automemory
                                    mps.unbind_mode = unbind_mode
                                    
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats)
                                    
                                    assert result.shape == (100, 12)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 5
                                    assert len(mps.data_nodes) == 4
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 4
                                    else:
                                        assert len(mps.virtual_nodes) == 3
                                        
                                    # Canonicalize and continue
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.canonicalize(mode=mode, **sv_cut)
                                            mps.trace(example,
                                                      inline_input=inline_input,
                                                      inline_mats=inline_mats)
                                            result = mps(data,
                                                         inline_input=inline_input,
                                                         inline_mats=inline_mats)
                                            
                                            assert result.shape == (100, 12)
                                            assert len(mps.edges) == 1
                                            assert len(mps.leaf_nodes) == 5
                                            assert len(mps.data_nodes) == 4
                                            if not inline_input and automemory:
                                                assert len(mps.virtual_nodes) == 4
                                            else:
                                                assert len(mps.virtual_nodes) == 3
                                        
    def test_all_algorithms_diff_d_phys(self):
        d_phys = torch.randint(low=2, high=7, size=(4,)).tolist()
        example = [torch.randn(1, d) for d in d_phys]
        data = [torch.randn(100, d) for d in d_phys]
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                for l_position in range(5):
                    mps = tk.MPSLayer(n_sites=5,
                                      d_phys=d_phys,
                                      n_labels=12,
                                      d_bond=2,
                                      l_position=l_position,
                                      boundary=boundary,
                                      param_bond=param_bond)
                    
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    mps.automemory = automemory
                                    mps.unbind_mode = unbind_mode
                                    
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats)
                                    
                                    assert result.shape == (100, 12)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 5
                                    assert len(mps.data_nodes) == 4
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 1
                                    else:
                                        assert len(mps.virtual_nodes) == 0
                                        
                                    # Canonicalize and continue
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.canonicalize(mode=mode, **sv_cut)
                                            mps.trace(example,
                                                      inline_input=inline_input,
                                                      inline_mats=inline_mats)
                                            result = mps(data,
                                                         inline_input=inline_input,
                                                         inline_mats=inline_mats)
                                            
                                            assert result.shape == (100, 12)
                                            assert len(mps.edges) == 1
                                            assert len(mps.leaf_nodes) == 5
                                            assert len(mps.data_nodes) == 4
                                            if not inline_input and automemory:
                                                assert len(mps.virtual_nodes) == 1
                                            else:
                                                assert len(mps.virtual_nodes) == 0
                                        
    def test_all_algorithms_diff_d_bond(self):
        d_bond = torch.randint(low=2, high=7, size=(5,)).tolist()
        example = torch.randn(4, 1, 5)
        data = torch.randn(4, 100, 5)
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                for l_position in range(5):
                    mps = tk.MPSLayer(n_sites=5,
                                      d_phys=5,
                                      n_labels=12,
                                      d_bond=d_bond[:-1] if boundary == 'obc' else d_bond,
                                      l_position=l_position,
                                      boundary=boundary,
                                      param_bond=param_bond)
                    
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    print(boundary, param_bond, l_position,
                                          automemory, unbind_mode, inline_input, inline_mats)
                                    mps.automemory = automemory
                                    mps.unbind_mode = unbind_mode
                                    
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats)
                                    
                                    assert result.shape == (100, 12)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 5
                                    assert len(mps.data_nodes) == 4
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 4
                                    else:
                                        assert len(mps.virtual_nodes) == 3
                                        
                                    # Canonicalize and continue
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.canonicalize(mode=mode, **sv_cut)
                                            mps.trace(example,
                                                      inline_input=inline_input,
                                                      inline_mats=inline_mats)
                                            result = mps(data,
                                                         inline_input=inline_input,
                                                         inline_mats=inline_mats)
                                            
                                            assert result.shape == (100, 12)
                                            assert len(mps.edges) == 1
                                            assert len(mps.leaf_nodes) == 5
                                            assert len(mps.data_nodes) == 4
                                            if not inline_input and automemory:
                                                assert len(mps.virtual_nodes) == 4
                                            else:
                                                assert len(mps.virtual_nodes) == 3
                                        
    def test_all_algorithms_diff_d_phys_d_bond(self):
        d_phys = torch.randint(low=2, high=7, size=(4,)).tolist()
        d_bond = torch.randint(low=2, high=7, size=(5,)).tolist()
        example = [torch.randn(1, d) for d in d_phys]
        data = [torch.randn(100, d) for d in d_phys]
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                for l_position in range(5):
                    mps = tk.MPSLayer(n_sites=5,
                                      d_phys=d_phys,
                                      n_labels=12,
                                      d_bond=d_bond[:-1] if boundary == 'obc' else d_bond,
                                      l_position=l_position,
                                      boundary=boundary,
                                      param_bond=param_bond)
                    
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    print(boundary, param_bond, l_position,
                                          automemory, unbind_mode, inline_input, inline_mats)
                                    mps.automemory = automemory
                                    mps.unbind_mode = unbind_mode
                                    
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats)
                                    
                                    assert result.shape == (100, 12)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 5
                                    assert len(mps.data_nodes) == 4
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 1
                                    else:
                                        assert len(mps.virtual_nodes) == 0
                                        
                                    # Canonicalize and continue
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.canonicalize(mode=mode, **sv_cut)
                                            mps.trace(example,
                                                      inline_input=inline_input,
                                                      inline_mats=inline_mats)
                                            result = mps(data,
                                                         inline_input=inline_input,
                                                         inline_mats=inline_mats)
                                            
                                            assert result.shape == (100, 12)
                                            assert len(mps.edges) == 1
                                            assert len(mps.leaf_nodes) == 5
                                            assert len(mps.data_nodes) == 4
                                            if not inline_input and automemory:
                                                assert len(mps.virtual_nodes) == 1
                                            else:
                                                assert len(mps.virtual_nodes) == 0

    def test_extreme_case_left_output(self):
        # Left node + Outpt node
        mps = tk.MPSLayer(n_sites=2,
                          d_phys=5,
                          n_labels=12,
                          d_bond=2,
                          l_position=1,
                          boundary='obc',
                          param_bond=True)
        example = torch.randn(1, 1, 5)
        data = torch.randn(1, 100, 5)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.automemory = automemory
                        mps.unbind_mode = unbind_mode
                        
                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)
                        
                        assert result.shape == (100, 12)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 3
                        
                        # Canonicalize and continue
                        for mode in ['svd', 'svdr', 'qr']:
                            sv_cut_dicts = [{'rank': 2},
                                            {'cum_percentage': 0.95},
                                            {'cutoff': 1e-5}]
                            for sv_cut in sv_cut_dicts:
                                mps.canonicalize(mode=mode, **sv_cut)
                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)
                                
                                assert result.shape == (100, 12)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == 2
                                assert len(mps.data_nodes) == 1
                                assert len(mps.virtual_nodes) == 3

    def test_extreme_case_output_right(self):
        # Output node + Right node
        mps = tk.MPSLayer(n_sites=2,
                          d_phys=5,
                          n_labels=12,
                          d_bond=2,
                          l_position=0,
                          boundary='obc',
                          param_bond=True)
        example = torch.randn(1, 1, 5)
        data = torch.randn(1, 100, 5)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.automemory = automemory
                        mps.unbind_mode = unbind_mode
                        
                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)
                        
                        assert result.shape == (100, 12)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 3
                        
                        # Canonicalize and continue
                        for mode in ['svd', 'svdr', 'qr']:
                            sv_cut_dicts = [{'rank': 2},
                                            {'cum_percentage': 0.95},
                                            {'cutoff': 1e-5}]
                            for sv_cut in sv_cut_dicts:
                                mps.canonicalize(mode=mode, **sv_cut)
                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)
                                
                                assert result.shape == (100, 12)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == 2
                                assert len(mps.data_nodes) == 1
                                assert len(mps.virtual_nodes) == 3

    def test_extreme_case_output(self):
        # Outpt node
        mps = tk.MPSLayer(n_sites=1,
                          d_phys=5,
                          n_labels=12,
                          d_bond=2,
                          l_position=0,
                          boundary='pbc',
                          param_bond=True)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.automemory = automemory
                        mps.unbind_mode = unbind_mode
                        
                        # We shouldn't pass any data since the MPS only
                        # has the output node
                        mps.trace(inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(inline_input=inline_input,
                                     inline_mats=inline_mats)  # Same as mps.contract()
                        
                        assert result.shape == (12,)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 0
                        assert len(mps.virtual_nodes) == 0
                        
    def test_canonicalize_univocal(self):
        example = torch.randn(4, 1, 2)
        data = torch.randn(4, 100, 2)
        
        mps = tk.MPSLayer(n_sites=5,
                          d_phys=2,
                          d_bond=5,
                          n_labels=2,
                          boundary='obc',
                          param_bond=False)
        
        tensor = torch.randn(mps.output_node.shape) * 1e-9
        aux = torch.eye(tensor.shape[0], tensor.shape[2])
        tensor[:, 0, :] = aux
        mps.output_node.tensor = tensor
        
        # mps.output_node.tensor = torch.randn(mps.output_node.shape)
        
        # Contract with data
        mps.trace(example)
        result = mps(data)
        
        # Contract MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node
        
        mps_tensor = result.tensor
                
        # Canonicalize and continue
        mps.canonicalize_univocal()
        mps.trace(example)
        result = mps(data)
        
        assert result.shape == (100, 2)
        assert len(mps.edges) == 1
        assert len(mps.leaf_nodes) == 5
        assert len(mps.data_nodes) == 4
        assert len(mps.virtual_nodes) == 3
        
        # Contract Canonical MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node
        
        mps_canon_tensor = result.tensor
        
        diff = mps_tensor - mps_canon_tensor
        norm = diff.norm()
        assert norm.item() < 1e-4
        
    def test_canonicalize_univocal_diff_dims(self):
        d_phys = [2, 3, 5, 6]  #torch.arange(2, 6).int().tolist()
        d_bond = torch.arange(2, 6).int().tolist()
        example = [torch.randn(1, d) for d in d_phys]
        data = [torch.randn(100, d) for d in d_phys]
        
        mps = tk.MPSLayer(n_sites=5,
                          d_phys=d_phys,
                          d_bond=d_bond,
                          n_labels=5,
                          boundary='obc',
                          param_bond=False)
        
        tensor = torch.randn(mps.output_node.shape) * 1e-9
        aux = torch.eye(tensor.shape[0], tensor.shape[2])
        tensor[:, 0, :] = aux
        mps.output_node.tensor = tensor
        
        # mps.output_node.tensor = torch.randn(mps.output_node.shape)
        
        # Contract with data
        mps.trace(example)
        result = mps(data)
        
        # Contract MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node
        
        mps_tensor = result.tensor
                
        # Canonicalize and continue
        mps.canonicalize_univocal()
        mps.trace(example)
        result = mps(data)
        
        assert result.shape == (100, 5)
        assert len(mps.edges) == 1
        assert len(mps.leaf_nodes) == 5
        assert len(mps.data_nodes) == 4
        assert len(mps.virtual_nodes) == 0
        
        # Contract Canonical MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node
        
        mps_canon_tensor = result.tensor
        
        diff = mps_tensor - mps_canon_tensor
        norm = diff.norm()
        assert norm.item() < 1e-5
        
    def test_canonicalize_univocal_bond_greater_than_phys(self):
        example = torch.randn(5, 1, 2)
        data = torch.randn(5, 100, 2)
        
        mps = tk.MPSLayer(n_sites=5,
                          d_phys=2,
                          d_bond=20,
                          n_labels=2,
                          boundary='obc',
                          param_bond=False)
        
        tensor = torch.randn(mps.output_node.shape) * 1e-9
        aux = torch.eye(tensor.shape[0], tensor.shape[2])
        tensor[:, 0, :] = aux
        mps.output_node.tensor = tensor
        
        # mps.output_node.tensor = torch.randn(mps.output_node.shape)
        
        # Contract with data
        mps.trace(example)
        result = mps(data)
        
        # Contract MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node
        
        mps_tensor = result.tensor
                
        # Canonicalize and continue
        mps.canonicalize_univocal()
        mps.trace(example)
        result = mps(data)
        
        assert result.shape == (100, 2)
        assert len(mps.edges) == 1
        assert len(mps.leaf_nodes) == 5
        assert len(mps.data_nodes) == 4
        assert len(mps.virtual_nodes) == 3
        
        # Contract Canonical MPS
        result = mps.left_node
        for node in mps.left_env:
            result @= node
        result @= mps.output_node
        for node in mps.right_env:
            result @= node
        result @= mps.right_node
        
        mps_canon_tensor = result.tensor
        
        diff = mps_tensor - mps_canon_tensor
        norm = diff.norm()
        assert norm.item() < 1e-5

    def test_check_grad(self):
        mps = tk.MPSLayer(n_sites=2,
                          d_phys=2,
                          n_labels=2,
                          d_bond=2,
                          l_position=1,
                          boundary='obc').cuda()

        data = torch.randn(1, 1, 2).cuda()
        result = mps.forward(data)
        result[0, 0].backward()

        I = data.squeeze(0)
        A = mps.left_node.tensor
        B = mps.output_node.tensor
        grad_A1 = mps.left_node.grad
        grad_B1 = mps.output_node.grad

        grad_A2 = I.t() @ B[:, 0].view(2, 1).t()
        grad_B2 = (I @ A).t() @ torch.tensor([[1., 0.]]).cuda()

        assert torch.equal(grad_A1, grad_A2)
        assert torch.equal(grad_B1, grad_B2)


class TestUMPSLayer:
    
    def test_all_algorithms(self):
        example = torch.randn(4, 1, 5)
        data = torch.randn(4, 100, 5)
        
        for param_bond in [True, False]:
            for l_position in range(5):
                mps = tk.UMPSLayer(n_sites=5,
                                   d_phys=5,
                                   n_labels=12,
                                   d_bond=2,
                                   l_position=l_position,
                                   param_bond=param_bond)
                
                for automemory in [True, False]:
                    for unbind_mode in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                mps.automemory = automemory
                                mps.unbind_mode = unbind_mode
                                
                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)
                                
                                assert result.shape == (100, 12)
                                assert len(mps.edges) == 1
                                assert len(mps.leaf_nodes) == 5
                                assert len(mps.data_nodes) == 4
                                assert len(mps.virtual_nodes) == 4
                                
    def test_extreme_case_left_output(self):
        # Left node + Output node
        mps = tk.UMPSLayer(n_sites=2,
                           d_phys=5,
                           n_labels=12,
                           d_bond=2,
                           l_position=1,
                           param_bond=True)
        example = torch.randn(1, 1, 5)
        data = torch.randn(1, 100, 5)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.automemory = automemory
                        mps.unbind_mode = unbind_mode
                        
                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)
                        
                        assert result.shape == (100, 12)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 4

    def test_extreme_case_output_right(self):
        # Output node + Right node
        mps = tk.UMPSLayer(n_sites=2,
                           d_phys=5,
                           n_labels=12,
                           d_bond=2,
                           l_position=0,
                           param_bond=True)
        example = torch.randn(1, 1, 5)
        data = torch.randn(1, 100, 5)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.automemory = automemory
                        mps.unbind_mode = unbind_mode
                        
                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)
                        
                        assert result.shape == (100, 12)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 4

    def test_extreme_case_output(self):
        # Outpt node
        mps = tk.UMPSLayer(n_sites=1,
                           d_phys=5,
                           n_labels=12,
                           d_bond=2,
                           l_position=0,
                           param_bond=True)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.automemory = automemory
                        mps.unbind_mode = unbind_mode
                        
                        # We shouldn't pass any data since the MPS only
                        # has the output node
                        mps.trace(inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(inline_input=inline_input,
                                     inline_mats=inline_mats)  # Same as mps.contract()
                        
                        assert result.shape == (12,)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 0
                        assert len(mps.virtual_nodes) == 1


class TestConvMPSLayer:
    
    def test_all_algorithms(self):
        # TODO: change embedding by tk.add_ones
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = tk.add_ones(torch.randn(1, 5, 5), dim=1)
        data = tk.add_ones(torch.randn(100, 5, 5), dim=1)
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                for l_position in range(5):
                    mps = tk.ConvMPSLayer(in_channels=2,
                                          out_channels=5,
                                          d_bond=2,
                                          kernel_size=2,
                                          l_position=l_position,
                                          boundary=boundary,
                                          param_bond=param_bond)
                    
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    for conv_mode in ['flat', 'snake']:
                                        mps.automemory = automemory
                                        mps.unbind_mode = unbind_mode
                                        
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats,
                                                  mode=conv_mode)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats,
                                                     mode=conv_mode)
                                        
                                        assert result.shape == (100, 5, 4, 4)
                                        assert len(mps.edges) == 1
                                        assert len(mps.leaf_nodes) == 5
                                        assert len(mps.data_nodes) == 4
                                        if not inline_input and automemory:
                                            assert len(mps.virtual_nodes) == 4
                                        else:
                                            assert len(mps.virtual_nodes) == 3
                                            
                                        # Canonicalize and continue
                                        for mode in ['svd', 'svdr', 'qr']:
                                            sv_cut_dicts = [{'rank': 2},
                                                            {'cum_percentage': 0.95},
                                                            {'cutoff': 1e-5}]
                                            for sv_cut in sv_cut_dicts:
                                                mps.canonicalize(mode=mode, **sv_cut)
                                                mps.trace(example,
                                                          inline_input=inline_input,
                                                          inline_mats=inline_mats,
                                                          mode=conv_mode)
                                                result = mps(data,
                                                             inline_input=inline_input,
                                                             inline_mats=inline_mats,
                                                             mode=conv_mode)
                                                
                                                assert result.shape == (100, 5, 4, 4)
                                                assert len(mps.edges) == 1
                                                assert len(mps.leaf_nodes) == 5
                                                assert len(mps.data_nodes) == 4
                                                if not inline_input and automemory:
                                                    assert len(mps.virtual_nodes) == 4
                                                else:
                                                    assert len(mps.virtual_nodes) == 3
                                        
    def test_all_algorithms_diff_d_bond(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        d_bond = torch.randint(low=2, high=7, size=(5,)).tolist()
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                for l_position in range(5):
                    mps = tk.ConvMPSLayer(in_channels=2,
                                          out_channels=5,
                                          d_bond=d_bond[:-1] if boundary == 'obc' else d_bond,
                                          kernel_size=2,
                                          l_position=l_position,
                                          boundary=boundary,
                                          param_bond=param_bond)
                    
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for inline_input in [True, False]:
                                for inline_mats in [True, False]:
                                    for conv_mode in ['flat', 'snake']:
                                        mps.automemory = automemory
                                        mps.unbind_mode = unbind_mode
                                        
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats,
                                                  mode=conv_mode)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats,
                                                     mode=conv_mode)
                                        
                                        assert result.shape == (100, 5, 4, 4)
                                        assert len(mps.edges) == 1
                                        assert len(mps.leaf_nodes) == 5
                                        assert len(mps.data_nodes) == 4
                                        if not inline_input and automemory:
                                            assert len(mps.virtual_nodes) == 4
                                        else:
                                            assert len(mps.virtual_nodes) == 3
                                            
                                        # Canonicalize and continue
                                        for mode in ['svd', 'svdr', 'qr']:
                                            sv_cut_dicts = [{'rank': 2},
                                                            {'cum_percentage': 0.95},
                                                            {'cutoff': 1e-5}]
                                            for sv_cut in sv_cut_dicts:
                                                mps.canonicalize(mode=mode, **sv_cut)
                                                mps.trace(example,
                                                          inline_input=inline_input,
                                                          inline_mats=inline_mats,
                                                          mode=conv_mode)
                                                result = mps(data,
                                                             inline_input=inline_input,
                                                             inline_mats=inline_mats,
                                                             mode=conv_mode)
                                                
                                                assert result.shape == (100, 5, 4, 4)
                                                assert len(mps.edges) == 1
                                                assert len(mps.leaf_nodes) == 5
                                                assert len(mps.data_nodes) == 4
                                                if not inline_input and automemory:
                                                    assert len(mps.virtual_nodes) == 4
                                                else:
                                                    assert len(mps.virtual_nodes) == 3

    def test_extreme_case_left_output(self):
        # Left node + Outpt node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
        mps = tk.ConvMPSLayer(in_channels=2,
                              out_channels=5,
                              d_bond=2,
                              kernel_size=1,
                              l_position=1,
                              boundary='obc',
                              param_bond=True)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.automemory = automemory
                            mps.unbind_mode = unbind_mode
                            
                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)
                            
                            assert result.shape == (100, 5, 5, 5)
                            assert len(mps.edges) == 1
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 3
                            
                            # Canonicalize and continue
                            for mode in ['svd', 'svdr', 'qr']:
                                sv_cut_dicts = [{'rank': 2},
                                                {'cum_percentage': 0.95},
                                                {'cutoff': 1e-5}]
                                for sv_cut in sv_cut_dicts:
                                    mps.canonicalize(mode=mode, **sv_cut)
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              mode=conv_mode)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 mode=conv_mode)
                                    
                                    assert result.shape == (100, 5, 5, 5)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 2
                                    assert len(mps.data_nodes) == 1
                                    assert len(mps.virtual_nodes) == 3

    def test_extreme_case_output_right(self):
        # Output node + Right node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
        mps = tk.ConvMPSLayer(in_channels=2,
                              out_channels=5,
                              d_bond=2,
                              kernel_size=1,
                              l_position=0,
                              boundary='obc',
                              param_bond=True)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.automemory = automemory
                            mps.unbind_mode = unbind_mode
                            
                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)
                            
                            assert result.shape == (100, 5, 5, 5)
                            assert len(mps.edges) == 1
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 3
                            
                            # Canonicalize and continue
                            for mode in ['svd', 'svdr', 'qr']:
                                sv_cut_dicts = [{'rank': 2},
                                                {'cum_percentage': 0.95},
                                                {'cutoff': 1e-5}]
                                for sv_cut in sv_cut_dicts:
                                    mps.canonicalize(mode=mode, **sv_cut)
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              mode=conv_mode)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 mode=conv_mode)
                                    
                                    assert result.shape == (100, 5, 5, 5)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 2
                                    assert len(mps.data_nodes) == 1
                                    assert len(mps.virtual_nodes) == 3


class TestConvUMPSLayer:
    
    def test_all_algorithms(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
        for param_bond in [True, False]:
            for l_position in range(5):
                mps = tk.ConvUMPSLayer(in_channels=2,
                                       out_channels=5,
                                       d_bond=2,
                                       kernel_size=2,
                                       l_position=l_position,
                                       param_bond=param_bond)
            
                for automemory in [True, False]:
                    for unbind_mode in [True, False]:
                        for inline_input in [True, False]:
                            for inline_mats in [True, False]:
                                for conv_mode in ['flat', 'snake']:
                                    mps.automemory = automemory
                                    mps.unbind_mode = unbind_mode
                                    
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              mode=conv_mode)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 mode=conv_mode)
                                    
                                    assert result.shape == (100, 5, 4, 4)
                                    assert len(mps.edges) == 1
                                    assert len(mps.leaf_nodes) == 5
                                    assert len(mps.data_nodes) == 4
                                    assert len(mps.virtual_nodes) == 4
                            
    def test_extreme_case_left_output(self):
        # Left node + Output node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
        mps = tk.ConvUMPSLayer(in_channels=2,
                               d_bond=2,
                               out_channels=5,
                               kernel_size=1,
                               l_position=1,
                               param_bond=True)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.automemory = automemory
                            mps.unbind_mode = unbind_mode
                            
                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)
                            
                            assert result.shape == (100, 5, 5, 5)
                            assert len(mps.edges) == 1
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 4

    def test_extreme_case_output_right(self):
        # Output node + Right node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
        mps = tk.ConvUMPSLayer(in_channels=2,
                               d_bond=2,
                               out_channels=5,
                               kernel_size=1,
                               l_position=0,
                               param_bond=True)
        
        for automemory in [True, False]:
            for unbind_mode in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.automemory = automemory
                            mps.unbind_mode = unbind_mode
                            
                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)
                            
                            assert result.shape == (100, 5, 5, 5)
                            assert len(mps.edges) == 1
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 4
