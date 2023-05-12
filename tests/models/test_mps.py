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
            mps = tk.MPS(n_features=4,
                         in_dim=5,
                         bond_dim=2,
                         boundary=boundary)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats)

                            assert result.shape == (100,)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 4
                            assert len(mps.data_nodes) == 4
                            if not inline_input and auto_stack:
                                assert len(mps.virtual_nodes) == 2
                            else:
                                assert len(mps.virtual_nodes) == 1

                            # Canonicalize and continue
                            for oc in range(4):
                                for mode in ['svd', 'svdr', 'qr']:
                                    sv_cut_dicts = [{'rank': 2},
                                                    {'cum_percentage': 0.95},
                                                    {'cutoff': 1e-5}]
                                    for sv_cut in sv_cut_dicts:
                                        mps.canonicalize(oc=oc, mode=mode,
                                                         **sv_cut)
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats)

                                        assert result.shape == (100,)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 4
                                        assert len(mps.data_nodes) == 4
                                        if not inline_input and auto_stack:
                                            assert len(mps.virtual_nodes) == 2
                                        else:
                                            assert len(mps.virtual_nodes) == 1

    def test_all_algorithms_diff_in_dim(self):
        in_dim = torch.randint(low=2, high=7, size=(4,)).tolist()
        example = [torch.randn(1, d) for d in in_dim]
        data = [torch.randn(100, d) for d in in_dim]

        for boundary in ['obc', 'pbc']:
            mps = tk.MPS(n_features=4,
                         in_dim=in_dim,
                         bond_dim=2,
                         boundary=boundary)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats)

                            assert result.shape == (100,)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 4
                            assert len(mps.data_nodes) == 4
                            if not inline_input and auto_stack:
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
                                        mps.canonicalize(oc=oc, mode=mode,
                                                         **sv_cut)
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats)

                                        assert result.shape == (100,)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 4
                                        assert len(mps.data_nodes) == 4
                                        if not inline_input and auto_stack:
                                            assert len(mps.virtual_nodes) == 1
                                        else:
                                            assert len(mps.virtual_nodes) == 0

    def test_all_algorithms_diff_bond_dim(self):
        bond_dim = torch.randint(low=2, high=7, size=(4,)).tolist()
        example = torch.randn(4, 1, 5)
        data = torch.randn(4, 100, 5)

        for boundary in ['obc', 'pbc']:
            mps = tk.MPS(n_features=4,
                         in_dim=5,
                         bond_dim=bond_dim[:-1] if boundary == 'obc' else bond_dim,
                         boundary=boundary)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats)

                            assert result.shape == (100,)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 4
                            assert len(mps.data_nodes) == 4
                            if not inline_input and auto_stack:
                                assert len(mps.virtual_nodes) == 2
                            else:
                                assert len(mps.virtual_nodes) == 1

                            # Canonicalize and continue
                            for oc in range(4):
                                for mode in ['svd', 'svdr', 'qr']:
                                    sv_cut_dicts = [{'rank': 2},
                                                    {'cum_percentage': 0.95},
                                                    {'cutoff': 1e-5}]
                                    for sv_cut in sv_cut_dicts:
                                        mps.canonicalize(oc=oc, mode=mode,
                                                         **sv_cut)
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats)

                                        assert result.shape == (100,)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 4
                                        assert len(mps.data_nodes) == 4
                                        if not inline_input and auto_stack:
                                            assert len(mps.virtual_nodes) == 2
                                        else:
                                            assert len(mps.virtual_nodes) == 1

    def test_all_algorithms_diff_in_dim_bond_dim(self):
        in_dim = torch.randint(low=2, high=7, size=(4,)).tolist()
        bond_dim = torch.randint(low=2, high=7, size=(4,)).tolist()
        example = [torch.randn(1, d) for d in in_dim]
        data = [torch.randn(100, d) for d in in_dim]

        for boundary in ['obc', 'pbc']:
            mps = tk.MPS(n_features=4,
                         in_dim=in_dim,
                         bond_dim=bond_dim[:-1] if boundary == 'obc' else bond_dim,
                         boundary=boundary)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats)

                            assert result.shape == (100,)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 4
                            assert len(mps.data_nodes) == 4
                            if not inline_input and auto_stack:
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
                                        mps.canonicalize(oc=oc, mode=mode,
                                                         **sv_cut)
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats)

                                        assert result.shape == (100,)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 4
                                        assert len(mps.data_nodes) == 4
                                        if not inline_input and auto_stack:
                                            assert len(mps.virtual_nodes) == 1
                                        else:
                                            assert len(mps.virtual_nodes) == 0

    def test_extreme_case_left_right(self):
        # Left node + Right node
        mps = tk.MPS(n_features=2,
                     in_dim=5,
                     bond_dim=2,
                     boundary='obc')
        example = torch.randn(2, 1, 5)
        data = torch.randn(2, 100, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)

                        assert result.shape == (100,)
                        assert len(mps.edges) == 0
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 2
                        assert len(mps.virtual_nodes) == 1

                        # Canonicalize and continue
                        for oc in range(2):
                            for mode in ['svd', 'svdr', 'qr']:
                                sv_cut_dicts = [{'rank': 2},
                                                {'cum_percentage': 0.95},
                                                {'cutoff': 1e-5}]
                                for sv_cut in sv_cut_dicts:
                                    mps.canonicalize(oc=oc, mode=mode,
                                                     **sv_cut)
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats)

                                    assert result.shape == (100,)
                                    assert len(mps.edges) == 0
                                    assert len(mps.leaf_nodes) == 2
                                    assert len(mps.data_nodes) == 2
                                    assert len(mps.virtual_nodes) == 1

    def test_extreme_case_one_node(self):
        # One node
        mps = tk.MPS(n_features=1,
                     in_dim=5,
                     bond_dim=2,
                     boundary='pbc')
        example = torch.randn(1, 1, 5)
        data = torch.randn(1, 100, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)

                        assert result.shape == (100,)
                        assert len(mps.edges) == 0
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 1
                        if not inline_input and auto_stack:
                            assert len(mps.virtual_nodes) == 2
                        else:
                            assert len(mps.virtual_nodes) == 1

                        # Canonicalize and continue
                        for mode in ['svd', 'svdr', 'qr']:
                            sv_cut_dicts = [{'rank': 2},
                                            {'cum_percentage': 0.95},
                                            {'cutoff': 1e-5}]
                            for sv_cut in sv_cut_dicts:
                                mps.canonicalize(oc=0, mode=mode,
                                                 **sv_cut)
                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats)

                                assert result.shape == (100,)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == 1
                                assert len(mps.data_nodes) == 1
                                if not inline_input and auto_stack:
                                    assert len(mps.virtual_nodes) == 2
                                else:
                                    assert len(mps.virtual_nodes) == 1

    def test_canonicalize_univocal(self):
        example = torch.randn(5, 1, 2)
        data = torch.randn(5, 100, 2)

        mps = tk.MPS(n_features=5,
                     in_dim=2,
                     bond_dim=5,
                     boundary='obc')

        # Contract with data
        mps.trace(example)
        result = mps(data)

        # Contract MPS
        result = mps.left_node
        for node in mps.mats_env:
            result @= node
        result @= mps.right_node

        mps_tensor = result.tensor

        # Canonicalize and continue
        mps.canonicalize_univocal()
        mps.trace(example)
        result = mps(data)

        assert result.shape == (100,)
        assert len(mps.edges) == 0
        assert len(mps.leaf_nodes) == 5
        assert len(mps.data_nodes) == 5
        assert len(mps.virtual_nodes) == 1

        # Contract Canonical MPS
        result = mps.left_node
        for node in mps.mats_env:
            result @= node
        result @= mps.right_node

        mps_canon_tensor = result.tensor

        diff = mps_tensor - mps_canon_tensor
        norm = diff.norm()
        assert norm.item() < 1e-5

    def test_canonicalize_univocal_diff_dims(self):
        in_dim = torch.arange(2, 7).int().tolist()
        bond_dim = torch.arange(2, 6).int().tolist()
        example = [torch.randn(1, d) for d in in_dim]
        data = [torch.randn(100, d) for d in in_dim]

        mps = tk.MPS(n_features=5,
                     in_dim=in_dim,
                     bond_dim=bond_dim,
                     boundary='obc')

        # Contract with data
        mps.trace(example)
        result = mps(data)

        # Contract MPS
        result = mps.left_node
        for node in mps.mats_env:
            result @= node
        result @= mps.right_node

        mps_tensor = result.tensor

        # Canonicalize and continue
        mps.canonicalize_univocal()
        mps.trace(example)
        result = mps(data)

        assert result.shape == (100,)
        assert len(mps.edges) == 0
        assert len(mps.leaf_nodes) == 5
        assert len(mps.data_nodes) == 5
        assert len(mps.virtual_nodes) == 0

        # Contract Canonical MPS
        result = mps.left_node
        for node in mps.mats_env:
            result @= node
        result @= mps.right_node

        mps_canon_tensor = result.tensor

        diff = mps_tensor - mps_canon_tensor
        norm = diff.norm()
        assert norm.item() < 1e-5

    def test_canonicalize_univocal_diff_rand_dims(self):
        # in_dim = [2, 3, 5, 4, 5] #torch.randint(low=2, high=10, size=(5,)).tolist()
        # bond_dim = [2, 3, 4, 5] #torch.randint(low=2, high=10, size=(4,)).tolist()
        # NOTE: just that combination of dims raises error

        in_dim = [2, 3, 5, 3, 2]
        bond_dim = [2, 6, 6, 2]

        example = [torch.randn(1, d) for d in in_dim]
        data = [torch.randn(100, d) for d in in_dim]

        mps = tk.MPS(n_features=5,
                     in_dim=in_dim,
                     bond_dim=bond_dim,
                     boundary='obc')

        # Contract with data
        mps.trace(example)
        result = mps(data)

        # Contract MPS
        result = mps.left_node
        for node in mps.mats_env:
            result @= node
        result @= mps.right_node

        mps_tensor = result.tensor

        # Canonicalize and continue
        mps.canonicalize_univocal()
        mps.trace(example)
        result = mps(data)

        assert result.shape == (100,)
        assert len(mps.edges) == 0
        assert len(mps.leaf_nodes) == 5
        assert len(mps.data_nodes) == 5
        assert len(mps.virtual_nodes) == 0

        # Contract Canonical MPS
        result = mps.left_node
        for node in mps.mats_env:
            result @= node
        result @= mps.right_node

        mps_canon_tensor = result.tensor

        diff = mps_tensor - mps_canon_tensor
        norm = diff.norm()
        assert norm.item() < 1e-4

    def test_canonicalize_univocal_bond_greater_than_phys(self):
        example = torch.randn(5, 1, 2)
        data = torch.randn(5, 100, 2)

        mps = tk.MPS(n_features=5,
                     in_dim=2,
                     bond_dim=20,
                     boundary='obc')

        # Contract with data
        mps.trace(example)
        result = mps(data)

        # Contract MPS
        result = mps.left_node
        for node in mps.mats_env:
            result @= node
        result @= mps.right_node

        mps_tensor = result.tensor

        # Canonicalize and continue
        mps.canonicalize_univocal()
        mps.trace(example)
        result = mps(data)

        assert result.shape == (100,)
        assert len(mps.edges) == 0
        assert len(mps.leaf_nodes) == 5
        assert len(mps.data_nodes) == 5
        assert len(mps.virtual_nodes) == 1

        # Contract Canonical MPS
        result = mps.left_node
        for node in mps.mats_env:
            result @= node
        result @= mps.right_node

        mps_canon_tensor = result.tensor

        diff = mps_tensor - mps_canon_tensor
        norm = diff.norm()
        assert norm.item() < 1e-5


class TestUMPS:

    def test_all_algorithms(self):
        example = torch.randn(4, 1, 5)
        data = torch.randn(4, 100, 5)

        mps = tk.models.UMPS(n_features=4,
                      in_dim=5,
                      bond_dim=2)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)

                        assert result.shape == (100,)
                        assert len(mps.edges) == 0
                        assert len(mps.leaf_nodes) == 4
                        assert len(mps.data_nodes) == 4
                        assert len(mps.virtual_nodes) == 2

    def test_extreme_case_mats_env_2_nodes(self):
        # Two nodes
        mps = tk.models.UMPS(n_features=2,
                      in_dim=5,
                      bond_dim=2)
        example = torch.randn(2, 1, 5)
        data = torch.randn(2, 100, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)

                        assert result.shape == (100,)
                        assert len(mps.edges) == 0
                        assert len(mps.leaf_nodes) == 2
                        assert len(mps.data_nodes) == 2
                        assert len(mps.virtual_nodes) == 2

    def test_extreme_case_mats_env_1_node(self):
        # One node
        mps = tk.models.UMPS(n_features=1,
                      in_dim=5,
                      bond_dim=2)
        example = torch.randn(1, 1, 5)
        data = torch.randn(1, 100, 5)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        mps.auto_stack = auto_stack
                        mps.auto_unbind = auto_unbind

                        mps.trace(example,
                                  inline_input=inline_input,
                                  inline_mats=inline_mats)
                        result = mps(data,
                                     inline_input=inline_input,
                                     inline_mats=inline_mats)

                        assert result.shape == (100,)
                        assert len(mps.edges) == 0
                        assert len(mps.leaf_nodes) == 1
                        assert len(mps.data_nodes) == 1
                        assert len(mps.virtual_nodes) == 2


class TestConvMPS:

    def test_all_algorithms(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))

        for boundary in ['obc', 'pbc']:
            mps = tk.models.ConvMPS(in_channels=2,
                             bond_dim=2,
                             kernel_size=2,
                             boundary=boundary)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            for conv_mode in ['flat', 'snake']:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats,
                                          mode=conv_mode)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats,
                                             mode=conv_mode)

                                assert result.shape == (100, 4, 4)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == 4
                                assert len(mps.data_nodes) == 4
                                if not inline_input and auto_stack:
                                    assert len(mps.virtual_nodes) == 2
                                else:
                                    assert len(mps.virtual_nodes) == 1

                                # Canonicalize and continue
                                for oc in range(4):
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.canonicalize(oc=oc, mode=mode,
                                                             **sv_cut)
                                            mps.trace(example,
                                                      inline_input=inline_input,
                                                      inline_mats=inline_mats,
                                                      mode=conv_mode)
                                            result = mps(data,
                                                         inline_input=inline_input,
                                                         inline_mats=inline_mats,
                                                         mode=conv_mode)

                                            assert result.shape == (100, 4, 4)
                                            assert len(mps.edges) == 0
                                            assert len(mps.leaf_nodes) == 4
                                            assert len(mps.data_nodes) == 4
                                            if not inline_input and auto_stack:
                                                assert len(
                                                    mps.virtual_nodes) == 2
                                            else:
                                                assert len(
                                                    mps.virtual_nodes) == 1

    def test_all_algorithms_diff_bond_dim(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        bond_dim = torch.randint(low=2, high=7, size=(4,)).tolist()
        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))

        for boundary in ['obc', 'pbc']:
            mps = tk.models.ConvMPS(in_channels=2,
                             bond_dim=bond_dim[:-
                                           1] if boundary == 'obc' else bond_dim,
                             kernel_size=2,
                             boundary=boundary)

            for auto_stack in [True, False]:
                for auto_unbind in [True, False]:
                    for inline_input in [True, False]:
                        for inline_mats in [True, False]:
                            for conv_mode in ['flat', 'snake']:
                                mps.auto_stack = auto_stack
                                mps.auto_unbind = auto_unbind

                                mps.trace(example,
                                          inline_input=inline_input,
                                          inline_mats=inline_mats,
                                          mode=conv_mode)
                                result = mps(data,
                                             inline_input=inline_input,
                                             inline_mats=inline_mats,
                                             mode=conv_mode)

                                assert result.shape == (100, 4, 4)
                                assert len(mps.edges) == 0
                                assert len(mps.leaf_nodes) == 4
                                assert len(mps.data_nodes) == 4
                                if not inline_input and auto_stack:
                                    assert len(mps.virtual_nodes) == 2
                                else:
                                    assert len(mps.virtual_nodes) == 1

                                # Canonicalize and continue
                                for oc in range(4):
                                    for mode in ['svd', 'svdr', 'qr']:
                                        sv_cut_dicts = [{'rank': 2},
                                                        {'cum_percentage': 0.95},
                                                        {'cutoff': 1e-5}]
                                        for sv_cut in sv_cut_dicts:
                                            mps.canonicalize(oc=oc, mode=mode,
                                                             **sv_cut)
                                            mps.trace(example,
                                                      inline_input=inline_input,
                                                      inline_mats=inline_mats,
                                                      mode=conv_mode)
                                            result = mps(data,
                                                         inline_input=inline_input,
                                                         inline_mats=inline_mats,
                                                         mode=conv_mode)

                                            assert result.shape == (100, 4, 4)
                                            assert len(mps.edges) == 0
                                            assert len(mps.leaf_nodes) == 4
                                            assert len(mps.data_nodes) == 4
                                            if not inline_input and auto_stack:
                                                assert len(
                                                    mps.virtual_nodes) == 2
                                            else:
                                                assert len(
                                                    mps.virtual_nodes) == 1

    def test_extreme_case_left_right(self):
        # Left node + Right node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))

        mps = tk.models.ConvMPS(in_channels=2,
                         bond_dim=2,
                         kernel_size=(1, 2),
                         boundary='obc')

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)

                            assert result.shape == (100, 5, 4)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 2
                            assert len(mps.data_nodes) == 2
                            assert len(mps.virtual_nodes) == 1

                            # Canonicalize and continue
                            for oc in range(2):
                                for mode in ['svd', 'svdr', 'qr']:
                                    sv_cut_dicts = [{'rank': 2},
                                                    {'cum_percentage': 0.95},
                                                    {'cutoff': 1e-5}]
                                    for sv_cut in sv_cut_dicts:
                                        mps.canonicalize(oc=oc, mode=mode,
                                                         **sv_cut)
                                        mps.trace(example,
                                                  inline_input=inline_input,
                                                  inline_mats=inline_mats,
                                                  mode=conv_mode)
                                        result = mps(data,
                                                     inline_input=inline_input,
                                                     inline_mats=inline_mats,
                                                     mode=conv_mode)

                                        assert result.shape == (100, 5, 4)
                                        assert len(mps.edges) == 0
                                        assert len(mps.leaf_nodes) == 2
                                        assert len(mps.data_nodes) == 2
                                        assert len(mps.virtual_nodes) == 1

    def test_extreme_case_one_node(self):
        # One node
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))

        mps = tk.models.ConvMPS(in_channels=2,
                         bond_dim=2,
                         kernel_size=1,
                         boundary='pbc')

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)

                            assert result.shape == (100, 5, 5)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 1
                            assert len(mps.data_nodes) == 1
                            if not inline_input and auto_stack:
                                assert len(mps.virtual_nodes) == 2
                            else:
                                assert len(mps.virtual_nodes) == 1

                            # Canonicalize and continue
                            for mode in ['svd', 'svdr', 'qr']:
                                sv_cut_dicts = [{'rank': 2},
                                                {'cum_percentage': 0.95},
                                                {'cutoff': 1e-5}]
                                for sv_cut in sv_cut_dicts:
                                    mps.canonicalize(oc=0, mode=mode,
                                                     **sv_cut)
                                    mps.trace(example,
                                              inline_input=inline_input,
                                              inline_mats=inline_mats,
                                              mode=conv_mode)
                                    result = mps(data,
                                                 inline_input=inline_input,
                                                 inline_mats=inline_mats,
                                                 mode=conv_mode)

                                    assert result.shape == (100, 5, 5)
                                    assert len(mps.edges) == 0
                                    assert len(mps.leaf_nodes) == 1
                                    assert len(mps.data_nodes) == 1
                                    if not inline_input and auto_stack:
                                        assert len(mps.virtual_nodes) == 2
                                    else:
                                        assert len(mps.virtual_nodes) == 1


class TestConvUMPS:

    def test_all_algorithms(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))

        mps = tk.models.ConvUMPS(in_channels=2,
                          bond_dim=2,
                          kernel_size=2)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)

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

        mps = tk.models.ConvUMPS(in_channels=2,
                          bond_dim=2,
                          kernel_size=(1, 2))

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)

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

        mps = tk.models.ConvUMPS(in_channels=2,
                          bond_dim=2,
                          kernel_size=1)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline_input in [True, False]:
                    for inline_mats in [True, False]:
                        for conv_mode in ['flat', 'snake']:
                            mps.auto_stack = auto_stack
                            mps.auto_unbind = auto_unbind

                            mps.trace(example,
                                      inline_input=inline_input,
                                      inline_mats=inline_mats,
                                      mode=conv_mode)
                            result = mps(data,
                                         inline_input=inline_input,
                                         inline_mats=inline_mats,
                                         mode=conv_mode)

                            assert result.shape == (100, 5, 5)
                            assert len(mps.edges) == 0
                            assert len(mps.leaf_nodes) == 1
                            assert len(mps.data_nodes) == 1
                            assert len(mps.virtual_nodes) == 4
