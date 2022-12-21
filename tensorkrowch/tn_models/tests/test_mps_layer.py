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
                                    mps.inline_input = inline_input
                                    mps.inline_mats = inline_mats
                                    
                                    mps.trace(example)
                                    result = mps(data)
                                    
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
                                            mps.delete_non_leaf()
                                            mps.canonicalize(mode=mode, **sv_cut)
                                            mps.trace(example)
                                            result = mps(data)
                                            
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
                                    mps.inline_input = inline_input
                                    mps.inline_mats = inline_mats
                                    
                                    mps.trace(example)
                                    result = mps(data)
                                    
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
                                            mps.delete_non_leaf()
                                            mps.canonicalize(mode=mode, **sv_cut)
                                            mps.trace(example)
                                            result = mps(data)
                                            
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
                                    mps.inline_input = inline_input
                                    mps.inline_mats = inline_mats
                                    
                                    mps.trace(example)
                                    result = mps(data)
                                    
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
                                            mps.delete_non_leaf()
                                            mps.canonicalize(mode=mode, **sv_cut)
                                            mps.trace(example)
                                            result = mps(data)
                                            
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
                                    mps.inline_input = inline_input
                                    mps.inline_mats = inline_mats
                                    
                                    mps.trace(example)
                                    result = mps(data)
                                    
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
                                            mps.delete_non_leaf()
                                            mps.canonicalize(mode=mode, **sv_cut)
                                            mps.trace(example)
                                            result = mps(data)
                                            
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
                        mps.inline_input = inline_input
                        mps.inline_mats = inline_mats
                        
                        mps.trace(example)
                        result = mps(data)
                        
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
                                mps.delete_non_leaf()
                                mps.canonicalize(mode=mode, **sv_cut)
                                mps.trace(example)
                                result = mps(data)
                                
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
                        mps.inline_input = inline_input
                        mps.inline_mats = inline_mats
                        
                        mps.trace(example)
                        result = mps(data)
                        
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
                                mps.delete_non_leaf()
                                mps.canonicalize(mode=mode, **sv_cut)
                                mps.trace(example)
                                result = mps(data)
                                
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
                        mps.inline_input = inline_input
                        mps.inline_mats = inline_mats
                        
                        # We shouldn't pass any data since the MPS only
                        # has the output node
                        mps.trace()
                        result = mps()  # Same as mps.contract()
                        
                        assert result.shape == (12,)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 0
                        assert len(mps.virtual_nodes) == 0

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
                                mps.inline_input = inline_input
                                mps.inline_mats = inline_mats
                                
                                mps.trace(example)
                                result = mps(data)
                                
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
                        mps.inline_input = inline_input
                        mps.inline_mats = inline_mats
                        
                        mps.trace(example)
                        result = mps(data)
                        
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
                        mps.inline_input = inline_input
                        mps.inline_mats = inline_mats
                        
                        mps.trace(example)
                        result = mps(data)
                        
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
                        mps.inline_input = inline_input
                        mps.inline_mats = inline_mats
                        
                        # We shouldn't pass any data since the MPS only
                        # has the output node
                        mps.trace()
                        result = mps()  # Same as mps.contract()
                        
                        assert result.shape == (12,)
                        assert len(mps.edges) == 1
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 0
                        assert len(mps.virtual_nodes) == 1


class TestConvMPSLayer:
    
    def test_all_algorithms(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
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
                                        mps.inline_input = inline_input
                                        mps.inline_mats = inline_mats
                                        
                                        mps.trace(example)
                                        result = mps(data, mode=conv_mode)
                                        
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
                                                mps.delete_non_leaf()
                                                mps.canonicalize(mode=mode, **sv_cut)
                                                mps.trace(example)
                                                result = mps(data, mode=conv_mode)
                                                
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
                                        mps.inline_input = inline_input
                                        mps.inline_mats = inline_mats
                                        
                                        mps.trace(example)
                                        result = mps(data, mode=conv_mode)
                                        
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
                                                mps.delete_non_leaf()
                                                mps.canonicalize(mode=mode, **sv_cut)
                                                mps.trace(example)
                                                result = mps(data, mode=conv_mode)
                                                
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
                            mps.inline_input = inline_input
                            mps.inline_mats = inline_mats
                            
                            mps.trace(example)
                            result = mps(data, mode=conv_mode)
                            
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
                                    mps.delete_non_leaf()
                                    mps.canonicalize(mode=mode, **sv_cut)
                                    mps.trace(example)
                                    result = mps(data, mode=conv_mode)
                                    
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
                            mps.inline_input = inline_input
                            mps.inline_mats = inline_mats
                            
                            mps.trace(example)
                            result = mps(data, mode=conv_mode)
                            
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
                                    mps.delete_non_leaf()
                                    mps.canonicalize(mode=mode, **sv_cut)
                                    mps.trace(example)
                                    result = mps(data, mode=conv_mode)
                                    
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
                                    mps.inline_input = inline_input
                                    mps.inline_mats = inline_mats
                                    
                                    mps.trace(example)
                                    result = mps(data, mode=conv_mode)
                                    
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
                            mps.inline_input = inline_input
                            mps.inline_mats = inline_mats
                            
                            mps.trace(example)
                            result = mps(data, mode=conv_mode)
                            
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
                            mps.inline_input = inline_input
                            mps.inline_mats = inline_mats
                            
                            mps.trace(example)
                            result = mps(data, mode=conv_mode)
                            
                            assert result.shape == (100, 5, 5, 5)
                            assert len(mps.edges) == 1
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 4
