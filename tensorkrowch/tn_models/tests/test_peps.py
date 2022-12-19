"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tk

from typing import Sequence
import time


class TestPEPS:
    
    def test_all_algorithms(self):
        example = torch.randn(3*4, 1, 5)
        data = torch.randn(3*4, 100, 5)
        
        for boundary_0 in ['obc', 'pbc']:
            for boundary_1 in ['pbc']:#['obc', 'pbc']:
                for param_bond in [True, False]:
                    peps = tk.PEPS(n_rows=3,
                                   n_cols=4,
                                   d_phys=5,
                                   d_bond=[2, 3],
                                   boundary=[boundary_0, boundary_1],
                                   param_bond=param_bond)
                
                    for automemory in [True, False]:
                        for unbind_mode in [True, False]:
                            for side in ['left']:#['up', 'down', 'left', 'right']:
                                for inline in [False]:#[True, False]:
                                    print(boundary_0, boundary_1, param_bond,
                                          automemory, unbind_mode, side, inline)
                                    peps.automemory = automemory
                                    peps.unbind_mode = unbind_mode
                                    
                                    peps.trace(example,
                                               from_side=side,
                                               inline=inline)
                                    result = peps(data,
                                                  from_side=side,
                                                  inline=inline)
                                    
                                    assert result.shape == (100,)
                                    assert len(peps.edges) == 0
                                    assert len(peps.leaf_nodes) == 3*4
                                    assert len(peps.data_nodes) == 3*4
                                    if automemory:
                                        if boundary_0 == 'obc':
                                            if boundary_1 == 'obc':
                                                assert len(peps.virtual_nodes) == 8
                                            else:
                                                assert len(peps.virtual_nodes) == 6
                                        else:
                                            if boundary_1 == 'obc':
                                                assert len(peps.virtual_nodes) == 6
                                            else:
                                                assert len(peps.virtual_nodes) == 4
                                    else:
                                        assert len(peps.virtual_nodes) == 3
                                        
    def test_all_algorithms_diff_d_phys(self):
        d_phys = torch.randint(low=2, high=7, size=(10,)).tolist()
        example = [torch.randn(1, d) for d in d_phys]
        data = [torch.randn(100, d) for d in d_phys]
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                mps = tk.MPS(n_sites=10,
                             d_phys=d_phys,
                             d_bond=2,
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
                                
                                assert result.shape == (100,)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == 10
                                assert len(mps.data_nodes) == 10
                                if not inline_input and automemory:
                                    assert len(mps.virtual_nodes) == 1
                                else:
                                    assert len(mps.virtual_nodes) == 0
                                    
                                # Canonicalize and continue
                                for oc in [0, 1, 5, 8, 9]:
                                    for mode in ['svd', 'svdr', 'qr']:
                                        mps.delete_non_leaf()
                                        mps.canonicalize(oc=oc, mode=mode, rank=2)
                                        mps.trace(example)
                                        result = mps(data)
                                        
                                        assert result.shape == (100,)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 10
                                        assert len(mps.data_nodes) == 10
                                        if not inline_input and automemory:
                                            assert len(mps.virtual_nodes) == 1
                                        else:
                                            assert len(mps.virtual_nodes) == 0
                                        
    def test_all_algorithms_diff_d_bond(self):
        d_bond = torch.randint(low=2, high=7, size=(10,)).tolist()
        example = torch.randn(10, 1, 5)
        data = torch.randn(10, 100, 5)
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                mps = tk.MPS(n_sites=10,
                             d_phys=5,
                             d_bond=d_bond[:-1] if boundary == 'obc' else d_bond,
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
                                
                                assert result.shape == (100,)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == 10
                                assert len(mps.data_nodes) == 10
                                if not inline_input and automemory:
                                    assert len(mps.virtual_nodes) == 4
                                else:
                                    assert len(mps.virtual_nodes) == 3
                                    
                                # Canonicalize and continue
                                for oc in [0, 1, 5, 8, 9]:
                                    for mode in ['svd', 'svdr', 'qr']:
                                        mps.delete_non_leaf()
                                        mps.canonicalize(oc=oc, mode=mode, rank=2)
                                        mps.trace(example)
                                        result = mps(data)
                                        
                                        assert result.shape == (100,)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 10
                                        assert len(mps.data_nodes) == 10
                                        if not inline_input and automemory:
                                            assert len(mps.virtual_nodes) == 4
                                        else:
                                            assert len(mps.virtual_nodes) == 3
                                        
    def test_all_algorithms_diff_d_phys_d_bond(self):
        d_phys = torch.randint(low=2, high=7, size=(10,)).tolist()
        d_bond = torch.randint(low=2, high=7, size=(10,)).tolist()
        example = [torch.randn(1, d) for d in d_phys]
        data = [torch.randn(100, d) for d in d_phys]
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                mps = tk.MPS(n_sites=10,
                             d_phys=d_phys,
                             d_bond=d_bond[:-1] if boundary == 'obc' else d_bond,
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
                                
                                assert result.shape == (100,)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == 10
                                assert len(mps.data_nodes) == 10
                                if not inline_input and automemory:
                                    assert len(mps.virtual_nodes) == 1
                                else:
                                    assert len(mps.virtual_nodes) == 0
                                    
                                # Canonicalize and continue
                                for oc in [0, 1, 5, 8, 9]:
                                    for mode in ['svd', 'svdr', 'qr']:
                                        mps.delete_non_leaf()
                                        mps.canonicalize(oc=oc, mode=mode, rank=2)
                                        mps.trace(example)
                                        result = mps(data)
                                        
                                        assert result.shape == (100,)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 10
                                        assert len(mps.data_nodes) == 10
                                        if not inline_input and automemory:
                                            assert len(mps.virtual_nodes) == 1
                                        else:
                                            assert len(mps.virtual_nodes) == 0

    def test_extreme_case_left_right(self):
        # Left node + Right node
        mps = tk.MPS(n_sites=2,
                     d_phys=5,
                     d_bond=2,
                     boundary='obc',
                     param_bond=True)
        example = torch.randn(2, 1, 5)
        data = torch.randn(2, 100, 5)
        
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
                        
                        assert result.shape == (100,)
                        assert len(mps.edges) == 0
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 2
                        assert len(mps.virtual_nodes) == 3
                            
                        # Canonicalize and continue
                        for oc in [0, 1]:
                            for mode in ['svd', 'svdr', 'qr']:
                                mps.delete_non_leaf()
                                mps.canonicalize(oc=oc, mode=mode, rank=2)
                                mps.trace(example)
                                result = mps(data)
                                
                                assert result.shape == (100,)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == 2
                                assert len(mps.data_nodes) == 2
                                assert len(mps.virtual_nodes) == 3

    def test_extreme_case_one_node(self):
        # One node
        mps = tk.MPS(n_sites=1,
                     d_phys=5,
                     d_bond=2,
                     boundary='pbc',
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
                        
                        assert result.shape == (100,)
                        assert len(mps.edges) == 0
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 1
                        if not inline_input and automemory:
                            assert len(mps.virtual_nodes) == 4
                        else:
                            assert len(mps.virtual_nodes) == 3
                            
                        # Canonicalize and continue
                        for mode in ['svd', 'svdr', 'qr']:
                            mps.delete_non_leaf()
                            mps.canonicalize(oc=0, mode=mode, rank=2)
                            mps.trace(example)
                            result = mps(data)
                            
                            assert result.shape == (100,)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 1
                            assert len(mps.data_nodes) == 1
                            if not inline_input and automemory:
                                assert len(mps.virtual_nodes) == 4
                            else:
                                assert len(mps.virtual_nodes) == 3


class TestUMPS:
    
    def test_all_algorithms(self):
        example = torch.randn(10, 1, 5)
        data = torch.randn(10, 100, 5)
        
        for param_bond in [True, False]:
            mps = tk.UMPS(n_sites=10,
                          d_phys=5,
                          d_bond=2,
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
                            
                            assert result.shape == (100,)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 10
                            assert len(mps.data_nodes) == 10
                            assert len(mps.virtual_nodes) == 4
                                
    def test_extreme_case_mats_env_2_nodes(self):
        # Two nodes
        mps = tk.UMPS(n_sites=2,
                      d_phys=5,
                      d_bond=2,
                      param_bond=True)
        example = torch.randn(2, 1, 5)
        data = torch.randn(2, 100, 5)
        
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
                        
                        assert result.shape == (100,)
                        assert len(mps.edges) == 0
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 2
                        assert len(mps.virtual_nodes) == 4

    def test_extreme_case_mats_env_1_node(self):
        # One node
        mps = tk.UMPS(n_sites=1,
                      d_phys=5,
                      d_bond=2,
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
                        
                        assert result.shape == (100,)
                        assert len(mps.edges) == 0
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 4


class TestConvMPS:
    
    def test_all_algorithms(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 14, 14))
        data = embedding(torch.randn(100, 14, 14))
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                mps = tk.ConvMPS(in_channels=2,
                                 d_bond=2,
                                 kernel_size=3,
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
                                    
                                    assert result.shape == (100, 12, 12)
                                    assert len(mps.edges) == 0
                                    assert len(mps.leaf_nodes) == 9
                                    assert len(mps.data_nodes) == 9
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 4
                                    else:
                                        assert len(mps.virtual_nodes) == 3
                                        
                                    # Canonicalize and continue
                                    for mode in ['svd', 'svdr', 'qr']:
                                        mps.delete_non_leaf()
                                        mps.canonicalize(mode=mode, rank=2)
                                        mps.trace(example)
                                        result = mps(data, mode=conv_mode)
                                        
                                        assert result.shape == (100, 12, 12)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 9
                                        assert len(mps.data_nodes) == 9
                                        if not inline_input and automemory:
                                            assert len(mps.virtual_nodes) == 4
                                        else:
                                            assert len(mps.virtual_nodes) == 3
                                        
    def test_all_algorithms_diff_d_bond(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        d_bond = torch.randint(low=2, high=7, size=(9,)).tolist()
        example = embedding(torch.randn(1, 14, 14))
        data = embedding(torch.randn(100, 14, 14))
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                mps = tk.ConvMPS(in_channels=2,
                                 d_bond=d_bond[:-1] if boundary == 'obc' else d_bond,
                                 kernel_size=3,
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
                                    
                                    assert result.shape == (100, 12, 12)
                                    assert len(mps.edges) == 0
                                    assert len(mps.leaf_nodes) == 9
                                    assert len(mps.data_nodes) == 9
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 4
                                    else:
                                        assert len(mps.virtual_nodes) == 3
                                        
                                    # Canonicalize and continue
                                    for mode in ['svd', 'svdr', 'qr']:
                                        mps.delete_non_leaf()
                                        mps.canonicalize(mode=mode, rank=2)
                                        mps.trace(example)
                                        result = mps(data, mode=conv_mode)
                                        
                                        assert result.shape == (100, 12, 12)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 9
                                        assert len(mps.data_nodes) == 9
                                        if not inline_input and automemory:
                                            assert len(mps.virtual_nodes) == 4
                                        else:
                                            assert len(mps.virtual_nodes) == 3

    def test_extreme_case_left_right(self):
        # Left node + Right node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 14, 14))
        data = embedding(torch.randn(100, 14, 14))
        
        mps = tk.ConvMPS(in_channels=2,
                         d_bond=2,
                         kernel_size=(1, 2),
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
                            
                            assert result.shape == (100, 14, 13)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 2
                            assert len(mps.virtual_nodes) == 3
                                
                            # Canonicalize and continue
                            for oc in [0, 1]:
                                for mode in ['svd', 'svdr', 'qr']:
                                    mps.delete_non_leaf()
                                    mps.canonicalize(oc=oc, mode=mode, rank=2)
                                    mps.trace(example)
                                    result = mps(data, mode=conv_mode)
                                    
                                    assert result.shape == (100, 14, 13)
                                    assert len(mps.edges) == 0
                                    assert len(mps.leaf_nodes) == 2
                                    assert len(mps.data_nodes) == 2
                                    assert len(mps.virtual_nodes) == 3

    def test_extreme_case_one_node(self):
        # One node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 14, 14))
        data = embedding(torch.randn(100, 14, 14))
        
        mps = tk.ConvMPS(in_channels=2,
                         d_bond=2,
                         kernel_size=1,
                         boundary='pbc',
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
                            
                            assert result.shape == (100, 14, 14)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 1
                            assert len(mps.data_nodes) == 1
                            if not inline_input and automemory:
                                assert len(mps.virtual_nodes) == 4
                            else:
                                assert len(mps.virtual_nodes) == 3
                                
                            # Canonicalize and continue
                            for mode in ['svd', 'svdr', 'qr']:
                                mps.delete_non_leaf()
                                mps.canonicalize(oc=0, mode=mode, rank=2)
                                mps.trace(example)
                                result = mps(data, mode=conv_mode)
                                
                                assert result.shape == (100, 14, 14)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == 1
                                assert len(mps.data_nodes) == 1
                                if not inline_input and automemory:
                                    assert len(mps.virtual_nodes) == 4
                                else:
                                    assert len(mps.virtual_nodes) == 3


class TestConvUMPS:
    
    def test_all_algorithms(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 14, 14))
        data = embedding(torch.randn(100, 14, 14))
        
        for param_bond in [True, False]:
            mps = tk.ConvUMPS(in_channels=2,
                              d_bond=2,
                              kernel_size=3,
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
                                
                                assert result.shape == (100, 12, 12)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == 9
                                assert len(mps.data_nodes) == 9
                                assert len(mps.virtual_nodes) == 4
                            
    def test_extreme_case_mats_env_2_nodes(self):
        # Two nodes
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 14, 14))
        data = embedding(torch.randn(100, 14, 14))
        
        mps = tk.ConvUMPS(in_channels=2,
                          d_bond=2,
                          kernel_size=(1, 2),
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
                            
                            assert result.shape == (100, 14, 13)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 2
                            assert len(mps.virtual_nodes) == 4

    def test_extreme_case_mats_env_1_node(self):
        # One node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 14, 14))
        data = embedding(torch.randn(100, 14, 14))
        
        mps = tk.ConvUMPS(in_channels=2,
                          d_bond=2,
                          kernel_size=1,
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
                            
                            assert result.shape == (100, 14, 14)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 1
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 4
