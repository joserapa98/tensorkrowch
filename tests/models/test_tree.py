"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tk

from typing import Sequence
import time


class TestTree:

    def test_all_algorithms(self):
        example = torch.randn(12, 1, 5)
        data = torch.randn(12, 100, 5)

        tree = tk.models.Tree(sites_per_layer=[6, 2, 1],
                       bond_dim=[[5, 5, 4], [4, 4, 4, 3], [3, 3, 2]])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, 2)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 9
                    assert len(tree.data_nodes) == 12
                    if auto_stack and not inline:
                        assert len(tree.virtual_nodes) == 6
                    else:
                        assert len(tree.virtual_nodes) == 3

                    # Canonicalize and continue
                    for mode in ['svd', 'svdr', 'qr']:
                        sv_cut_dicts = [{'rank': 2},
                                        {'cum_percentage': 0.95},
                                        {'cutoff': 1e-5}]
                        for sv_cut in sv_cut_dicts:
                            print(mode, sv_cut)
                            tree.canonicalize(mode=mode, **sv_cut)
                            tree.trace(example, inline=inline)
                            result = tree(data, inline=inline)

                            assert result.shape == (100, 2)
                            assert len(tree.edges) == 1
                            assert len(tree.leaf_nodes) == 9
                            assert len(tree.data_nodes) == 12
                            if auto_stack and not inline:
                                assert len(tree.virtual_nodes) == 6
                            else:
                                assert len(tree.virtual_nodes) == 3

    def test_extreme_case_1_node_per_layer(self):
        example = torch.randn(1, 1, 5)
        data = torch.randn(1, 100, 5)

        tree = tk.models.Tree(sites_per_layer=[1, 1, 1],
                       bond_dim=[[5, 4], [4, 3], [3, 2]])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, 2)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 3
                    assert len(tree.data_nodes) == 1
                    if auto_stack and not inline:
                        assert len(tree.virtual_nodes) == 6
                    else:
                        assert len(tree.virtual_nodes) == 3

                    # Canonicalize and continue
                    for mode in ['svd', 'svdr', 'qr']:
                        sv_cut_dicts = [{'rank': 2},
                                        {'cum_percentage': 0.95},
                                        {'cutoff': 1e-5}]
                        for sv_cut in sv_cut_dicts:
                            print(mode, sv_cut)
                            tree.canonicalize(mode=mode, **sv_cut)
                            tree.trace(example, inline=inline)
                            result = tree(data, inline=inline)

                            assert result.shape == (100, 2)
                            assert len(tree.edges) == 1
                            assert len(tree.leaf_nodes) == 3
                            assert len(tree.data_nodes) == 1
                            if auto_stack and not inline:
                                assert len(tree.virtual_nodes) == 6
                            else:
                                assert len(tree.virtual_nodes) == 3

    def test_extreme_case_1_node(self):
        example = torch.randn(3, 1, 5)
        data = torch.randn(3, 100, 5)

        tree = tk.models.Tree(sites_per_layer=[1],
                       bond_dim=[[5, 5, 5, 2]])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, 2)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 1
                    assert len(tree.data_nodes) == 3
                    if auto_stack and not inline:
                        assert len(tree.virtual_nodes) == 4
                    else:
                        assert len(tree.virtual_nodes) == 3

                    # Canonicalize and continue
                    for mode in ['svd', 'svdr', 'qr']:
                        sv_cut_dicts = [{'rank': 2},
                                        {'cum_percentage': 0.95},
                                        {'cutoff': 1e-5}]
                        for sv_cut in sv_cut_dicts:
                            print(mode, sv_cut)
                            tree.canonicalize(mode=mode, **sv_cut)
                            tree.trace(example, inline=inline)
                            result = tree(data, inline=inline)

                            assert result.shape == (100, 2)
                            assert len(tree.edges) == 1
                            assert len(tree.leaf_nodes) == 1
                            assert len(tree.data_nodes) == 3
                            if auto_stack and not inline:
                                assert len(tree.virtual_nodes) == 4
                            else:
                                assert len(tree.virtual_nodes) == 3


class TestUTree:

    def test_all_algorithms(self):
        example = torch.randn(8, 1, 4)
        data = torch.randn(8, 100, 4)

        tree = tk.models.UTree(sites_per_layer=[4, 2, 1],
                        bond_dim=[4, 4, 4])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, 4)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 7
                    assert len(tree.data_nodes) == 8
                    assert len(tree.virtual_nodes) == 4

    def test_extreme_case_1_node_per_layer(self):
        example = torch.randn(1, 1, 4)
        data = torch.randn(1, 100, 4)

        tree = tk.models.UTree(sites_per_layer=[1, 1, 1],
                        bond_dim=[4, 4])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, 4)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 3
                    assert len(tree.data_nodes) == 1
                    assert len(tree.virtual_nodes) == 4

    def test_extreme_case_1_node(self):
        example = torch.randn(3, 1, 5)
        data = torch.randn(3, 100, 5)

        # When there's only 1 node, dimension of output edge can be
        # different from input edges' dimension, otherwise it's not
        # possible
        tree = tk.models.UTree(sites_per_layer=[1],
                        bond_dim=[5, 5, 5, 2])

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, 2)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 1
                    assert len(tree.data_nodes) == 3
                    assert len(tree.virtual_nodes) == 4


class TestConvTree:

    def test_all_algorithms(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        in_channels = 2

        tree = tk.models.ConvTree(sites_per_layer=[4, 2, 1],
                           bond_dim=[[in_channels, 4], [4, 4, 3], [3, 3, 2]],
                           kernel_size=2)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, 2, 4, 4)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 7
                    assert len(tree.data_nodes) == 4
                    if auto_stack and not inline:
                        assert len(tree.virtual_nodes) == 6
                    else:
                        assert len(tree.virtual_nodes) == 3

                    # Canonicalize and continue
                    for mode in ['svd', 'svdr', 'qr']:
                        sv_cut_dicts = [{'rank': 2},
                                        {'cum_percentage': 0.95},
                                        {'cutoff': 1e-5}]
                        for sv_cut in sv_cut_dicts:
                            print(mode, sv_cut)
                            tree.canonicalize(mode=mode, **sv_cut)
                            tree.trace(example, inline=inline)
                            result = tree(data, inline=inline)

                            assert result.shape == (100, 2, 4, 4)
                            assert len(tree.edges) == 1
                            assert len(tree.leaf_nodes) == 7
                            assert len(tree.data_nodes) == 4
                            if auto_stack and not inline:
                                assert len(tree.virtual_nodes) == 6
                            else:
                                assert len(tree.virtual_nodes) == 3

    def test_extreme_case_1_node_per_layer(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        in_channels = 2

        tree = tk.models.ConvTree(sites_per_layer=[1, 1, 1],
                           bond_dim=[[in_channels, 4], [4, 3], [3, 2]],
                           kernel_size=1)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, 2, 5, 5)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 3
                    assert len(tree.data_nodes) == 1
                    if auto_stack and not inline:
                        assert len(tree.virtual_nodes) == 6
                    else:
                        assert len(tree.virtual_nodes) == 3

                    # Canonicalize and continue
                    for mode in ['svd', 'svdr', 'qr']:
                        sv_cut_dicts = [{'rank': 2},
                                        {'cum_percentage': 0.95},
                                        {'cutoff': 1e-5}]
                        for sv_cut in sv_cut_dicts:
                            print(mode, sv_cut)
                            tree.canonicalize(mode=mode, **sv_cut)
                            tree.trace(example, inline=inline)
                            result = tree(data, inline=inline)

                            assert result.shape == (100, 2, 5, 5)
                            assert len(tree.edges) == 1
                            assert len(tree.leaf_nodes) == 3
                            assert len(tree.data_nodes) == 1
                            if auto_stack and not inline:
                                assert len(tree.virtual_nodes) == 6
                            else:
                                assert len(tree.virtual_nodes) == 3

    def test_extreme_case_1_node(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        in_channels = 2

        tree = tk.models.ConvTree(sites_per_layer=[1],
                           bond_dim=[[in_channels, in_channels,
                                    in_channels, in_channels, 2]],
                           kernel_size=2)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, 2, 4, 4)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 1
                    assert len(tree.data_nodes) == 4
                    if auto_stack and not inline:
                        assert len(tree.virtual_nodes) == 4
                    else:
                        assert len(tree.virtual_nodes) == 3

                    # Canonicalize and continue
                    for mode in ['svd', 'svdr', 'qr']:
                        sv_cut_dicts = [{'rank': 2},
                                        {'cum_percentage': 0.95},
                                        {'cutoff': 1e-5}]
                        for sv_cut in sv_cut_dicts:
                            print(mode, sv_cut)
                            tree.canonicalize(mode=mode, **sv_cut)
                            tree.trace(example, inline=inline)
                            result = tree(data, inline=inline)

                            assert result.shape == (100, 2, 4, 4)
                            assert len(tree.edges) == 1
                            assert len(tree.leaf_nodes) == 1
                            assert len(tree.data_nodes) == 4
                            if auto_stack and not inline:
                                assert len(tree.virtual_nodes) == 4
                            else:
                                assert len(tree.virtual_nodes) == 3


class TestConvUTree:

    def test_all_algorithms(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        in_channels = 2

        tree = tk.models.ConvUTree(sites_per_layer=[4, 2, 1],
                            bond_dim=[in_channels, in_channels, in_channels],
                            kernel_size=(2, 4))

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, in_channels, 4, 2)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 7
                    assert len(tree.data_nodes) == 8
                    assert len(tree.virtual_nodes) == 4

    def test_extreme_case_1_node_per_layer(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        in_channels = 2

        tree = tk.models.ConvUTree(sites_per_layer=[1, 1, 1],
                            bond_dim=[in_channels, in_channels],
                            kernel_size=1)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, in_channels, 5, 5)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 3
                    assert len(tree.data_nodes) == 1
                    assert len(tree.virtual_nodes) == 4

    def test_extreme_case_1_node(self):
        def embedding(data: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(data),
                                data], dim=1)

        example = embedding(torch.randn(1, 5, 5))
        data = embedding(torch.randn(100, 5, 5))
        in_channels = 2

        tree = tk.models.ConvUTree(sites_per_layer=[1],
                            bond_dim=[in_channels, in_channels,
                                    in_channels, in_channels, 2],
                            kernel_size=2)

        for auto_stack in [True, False]:
            for auto_unbind in [True, False]:
                for inline in [True, False]:
                    print(auto_stack, auto_unbind, inline)
                    tree.auto_stack = auto_stack
                    tree.auto_unbind = auto_unbind

                    tree.trace(example, inline=inline)
                    result = tree(data, inline=inline)

                    assert result.shape == (100, 2, 4, 4)
                    assert len(tree.edges) == 1
                    assert len(tree.leaf_nodes) == 1
                    assert len(tree.data_nodes) == 4
                    assert len(tree.virtual_nodes) == 4
