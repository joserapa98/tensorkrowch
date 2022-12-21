"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tk

from typing import Sequence
import time


class TestMPS:
    
    def test_all_algorithms(self):
        example = torch.randn(4, 1, 5)
        data = torch.randn(4, 100, 5)
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                mps = tk.MPS(n_sites=4,
                             d_phys=5,
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
                                assert len(mps.leaf_nodes) == 4
                                assert len(mps.data_nodes) == 4
                                if not inline_input and automemory:
                                    assert len(mps.virtual_nodes) == 4
                                else:
                                    assert len(mps.virtual_nodes) == 3
                                    
                                # Canonicalize and continue
                                for oc in range(4):
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.delete_non_leaf()
                                            mps.canonicalize(oc=oc, mode=mode,
                                                             **sv_cut)
                                            mps.trace(example)
                                            result = mps(data)
                                            
                                            assert result.shape == (100,)
                                            assert len(mps.edges) == 0
                                            assert len(mps.leaf_nodes) == 4
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
                mps = tk.MPS(n_sites=4,
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
                                assert len(mps.leaf_nodes) == 4
                                assert len(mps.data_nodes) == 4
                                if not inline_input and automemory:
                                    assert len(mps.virtual_nodes) == 1
                                else:
                                    assert len(mps.virtual_nodes) == 0
                                    
                                # Canonicalize and continue
                                for oc in range(4):
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.delete_non_leaf()
                                            mps.canonicalize(oc=oc, mode=mode,
                                                             **sv_cut)
                                            mps.trace(example)
                                            result = mps(data)
                                            
                                            assert result.shape == (100,)
                                            assert len(mps.edges) == 0
                                            assert len(mps.leaf_nodes) == 4
                                            assert len(mps.data_nodes) == 4
                                            if not inline_input and automemory:
                                                assert len(mps.virtual_nodes) == 1
                                            else:
                                                assert len(mps.virtual_nodes) == 0
                                        
    def test_all_algorithms_diff_d_bond(self):
        d_bond = torch.randint(low=2, high=7, size=(4,)).tolist()
        example = torch.randn(4, 1, 5)
        data = torch.randn(4, 100, 5)
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                mps = tk.MPS(n_sites=4,
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
                                assert len(mps.leaf_nodes) == 4
                                assert len(mps.data_nodes) == 4
                                if not inline_input and automemory:
                                    assert len(mps.virtual_nodes) == 4
                                else:
                                    assert len(mps.virtual_nodes) == 3
                                    
                                # Canonicalize and continue
                                for oc in range(4):
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.delete_non_leaf()
                                            mps.canonicalize(oc=oc, mode=mode,
                                                             **sv_cut)
                                            mps.trace(example)
                                            result = mps(data)
                                            
                                            assert result.shape == (100,)
                                            assert len(mps.edges) == 0
                                            assert len(mps.leaf_nodes) == 4
                                            assert len(mps.data_nodes) == 4
                                            if not inline_input and automemory:
                                                assert len(mps.virtual_nodes) == 4
                                            else:
                                                assert len(mps.virtual_nodes) == 3
                                        
    def test_all_algorithms_diff_d_phys_d_bond(self):
        d_phys = torch.randint(low=2, high=7, size=(4,)).tolist()
        d_bond = torch.randint(low=2, high=7, size=(4,)).tolist()
        example = [torch.randn(1, d) for d in d_phys]
        data = [torch.randn(100, d) for d in d_phys]
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                mps = tk.MPS(n_sites=4,
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
                                assert len(mps.leaf_nodes) == 4
                                assert len(mps.data_nodes) == 4
                                if not inline_input and automemory:
                                    assert len(mps.virtual_nodes) == 1
                                else:
                                    assert len(mps.virtual_nodes) == 0
                                    
                                # Canonicalize and continue
                                for oc in range(4):
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.delete_non_leaf()
                                            mps.canonicalize(oc=oc, mode=mode,
                                                             **sv_cut)
                                            mps.trace(example)
                                            result = mps(data)
                                            
                                            assert result.shape == (100,)
                                            assert len(mps.edges) == 0
                                            assert len(mps.leaf_nodes) == 4
                                            assert len(mps.data_nodes) == 4
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
                        for oc in range(2):
                            for mode in ['svd', 'svdr', 'qr']:
                                sv_cut_dicts = [{'rank': 2},
                                                {'cum_percentage': 0.95},
                                                {'cutoff': 1e-5}]
                                for sv_cut in sv_cut_dicts:
                                    mps.delete_non_leaf()
                                    mps.canonicalize(oc=oc, mode=mode,
                                                        **sv_cut)
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
                            sv_cut_dicts = [{'rank': 2},
                                            {'cum_percentage': 0.95},
                                            {'cutoff': 1e-5}]
                            for sv_cut in sv_cut_dicts:
                                mps.delete_non_leaf()
                                mps.canonicalize(oc=0, mode=mode,
                                                 **sv_cut)
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
        example = torch.randn(4, 1, 5)
        data = torch.randn(4, 100, 5)
        
        for param_bond in [True, False]:
            mps = tk.UMPS(n_sites=4,
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
                            assert len(mps.leaf_nodes) == 4
                            assert len(mps.data_nodes) == 4
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
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                mps = tk.ConvMPS(in_channels=2,
                                 d_bond=2,
                                 kernel_size=2,
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
                                    
                                    assert result.shape == (100, 4, 4)
                                    assert len(mps.edges) == 0
                                    assert len(mps.leaf_nodes) == 4
                                    assert len(mps.data_nodes) == 4
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 4
                                    else:
                                        assert len(mps.virtual_nodes) == 3
                                        
                                    # Canonicalize and continue
                                    for oc in range(4):
                                        for mode in ['svd', 'svdr', 'qr']:
                                            sv_cut_dicts = [{'rank': 2},
                                                            {'cum_percentage': 0.95},
                                                            {'cutoff': 1e-5}]
                                            for sv_cut in sv_cut_dicts:
                                                mps.delete_non_leaf()
                                                mps.canonicalize(oc=oc, mode=mode,
                                                                    **sv_cut)
                                                mps.trace(example)
                                                result = mps(data, mode=conv_mode)
                                                
                                                assert result.shape == (100, 4, 4)
                                                assert len(mps.edges) == 0
                                                assert len(mps.leaf_nodes) == 4
                                                assert len(mps.data_nodes) == 4
                                                if not inline_input and automemory:
                                                    assert len(mps.virtual_nodes) == 4
                                                else:
                                                    assert len(mps.virtual_nodes) == 3
                                        
    def test_all_algorithms_diff_d_bond(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        d_bond = torch.randint(low=2, high=7, size=(4,)).tolist()
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
        for boundary in ['obc', 'pbc']:
            for param_bond in [True, False]:
                mps = tk.ConvMPS(in_channels=2,
                                 d_bond=d_bond[:-1] if boundary == 'obc' else d_bond,
                                 kernel_size=2,
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
                                    
                                    assert result.shape == (100, 4, 4)
                                    assert len(mps.edges) == 0
                                    assert len(mps.leaf_nodes) == 4
                                    assert len(mps.data_nodes) == 4
                                    if not inline_input and automemory:
                                        assert len(mps.virtual_nodes) == 4
                                    else:
                                        assert len(mps.virtual_nodes) == 3
                                        
                                    # Canonicalize and continue
                                    for oc in range(4):
                                        for mode in ['svd', 'svdr', 'qr']:
                                            sv_cut_dicts = [{'rank': 2},
                                                            {'cum_percentage': 0.95},
                                                            {'cutoff': 1e-5}]
                                            for sv_cut in sv_cut_dicts:
                                                mps.delete_non_leaf()
                                                mps.canonicalize(oc=oc, mode=mode,
                                                                    **sv_cut)
                                                mps.trace(example)
                                                result = mps(data, mode=conv_mode)
                                                
                                                assert result.shape == (100, 4, 4)
                                                assert len(mps.edges) == 0
                                                assert len(mps.leaf_nodes) == 4
                                                assert len(mps.data_nodes) == 4
                                                if not inline_input and automemory:
                                                    assert len(mps.virtual_nodes) == 4
                                                else:
                                                    assert len(mps.virtual_nodes) == 3

    def test_extreme_case_left_right(self):
        # Left node + Right node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
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
                            
                            assert result.shape == (100, 5, 4)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 2
                            assert len(mps.virtual_nodes) == 3
                                
                            # Canonicalize and continue
                            for oc in range(2):
                                for mode in ['svd', 'svdr', 'qr']:
                                    sv_cut_dicts = [{'rank': 2},
                                                    {'cum_percentage': 0.95},
                                                    {'cutoff': 1e-5}]
                                    for sv_cut in sv_cut_dicts:
                                        mps.delete_non_leaf()
                                        mps.canonicalize(oc=oc, mode=mode,
                                                         **sv_cut)
                                        mps.trace(example)
                                        result = mps(data, mode=conv_mode)
                                        
                                        assert result.shape == (100, 5, 4)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 2
                                        assert len(mps.data_nodes) == 2
                                        assert len(mps.virtual_nodes) == 3

    def test_extreme_case_one_node(self):
        # One node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
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
                            
                            assert result.shape == (100, 5, 5)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 1
                            assert len(mps.data_nodes) == 1
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
                                    mps.canonicalize(oc=0, mode=mode,
                                                     **sv_cut)
                                    mps.trace(example)
                                    result = mps(data, mode=conv_mode)
                                    
                                    assert result.shape == (100, 5, 5)
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
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
        for param_bond in [True, False]:
            mps = tk.ConvUMPS(in_channels=2,
                              d_bond=2,
                              kernel_size=2,
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
                                
                                assert result.shape == (100, 4, 4)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == 4
                                assert len(mps.data_nodes) == 4
                                assert len(mps.virtual_nodes) == 4
                            
    def test_extreme_case_mats_env_2_nodes(self):
        # Two nodes
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
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
                            
                            assert result.shape == (100, 5, 4)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 2
                            assert len(mps.virtual_nodes) == 4

    def test_extreme_case_mats_env_1_node(self):
        # One node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)
            
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        
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
                            
                            assert result.shape == (100, 5, 5)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 1
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 4
