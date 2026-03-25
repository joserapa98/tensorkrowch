"""
Tests for tree:

    * TestTree
    * TestUTree
    * TestConvTree
    * TestConvUTree
"""

import pytest

import torch
import tensorkrowch as tk

AUTO_BOOL_CASES = [True, False]
INLINE_CASES = [True, False]
TREE_CASES = [
    ('all_algorithms', [6, 2, 1], [[5, 5, 4], [4, 4, 4, 3], [3, 3, 2]],
     (1, 12, 5), (100, 12, 5), (100, 2), 9, 12, 4),
    ('extreme_case_1_node_per_layer', [1, 1, 1], [[5, 4], [4, 3], [3, 2]],
     (1, 1, 5), (100, 1, 5), (100, 2), 3, 1, 4),
    ('extreme_case_1_node', [1], [[5, 5, 5, 2]],
     (1, 3, 5), (100, 3, 5), (100, 2), 1, 3, 2),
]
UTREE_CASES = [
    ('all_algorithms', [4, 2, 1], [4, 4, 4], (1, 8, 4), (100, 8, 4),
     (100, 4), 7, 8),
    ('extreme_case_1_node_per_layer', [1, 1, 1], [4, 4], (1, 1, 4),
     (100, 1, 4), (100, 4), 3, 1),
    ('extreme_case_1_node', [1], [5, 5, 5, 2], (1, 3, 5), (100, 3, 5),
     (100, 2), 1, 3),
]
CONV_TREE_CASES = [
    ('all_algorithms', [4, 2, 1], [[2, 4], [4, 4, 3], [3, 3, 2]], 2,
     (100, 2, 4, 4), 7, 4, 4),
    ('extreme_case_1_node_per_layer', [1, 1, 1], [[2, 4], [4, 3], [3, 2]], 1,
     (100, 2, 5, 5), 3, 1, 4),
    ('extreme_case_1_node', [1], [[2, 2, 2, 2, 2]], 2,
     (100, 2, 4, 4), 1, 4, 2),
]
CONV_UTREE_CASES = [
    ('all_algorithms', [4, 2, 1], [2, 2, 2], (2, 4), (100, 2, 4, 2), 7, 8),
    ('extreme_case_1_node_per_layer', [1, 1, 1], [2, 2], 1,
     (100, 2, 5, 5), 3, 1),
    ('extreme_case_1_node', [1], [2, 2, 2, 2, 2], 2,
     (100, 2, 4, 4), 1, 4),
]
TREE_CANONICALIZE_CASES = [
    ('svd', {'rank': 2}),
    ('svd', {'cum_percentage': 0.95}),
    ('svd', {'cutoff': 1e-5}),
    ('svdr', {'rank': 2}),
    ('svdr', {'cum_percentage': 0.95}),
    ('svdr', {'cutoff': 1e-5}),
    ('qr', {'rank': 2}),
    ('qr', {'cum_percentage': 0.95}),
    ('qr', {'cutoff': 1e-5}),
]


class _TreeTestMixin:

    @staticmethod
    def _expected_virtual_nodes(auto_stack, inline, stacked_nodes):
        return stacked_nodes if auto_stack and not inline else 1

    @staticmethod
    def _image_inputs():
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)
        return example, data

    def _assert_tree_algorithm(self, tree, example, data, inline,
                               expected_shape, expected_leaf_nodes,
                               expected_data_nodes, expected_virtual_nodes):
        # Every tree variant checks the same forward pass invariants after trace.
        tree.trace(example, inline=inline)
        result = tree(data, inline=inline)

        assert result.shape == expected_shape
        assert len(tree.edges) == 1
        assert len(tree.leaf_nodes) == expected_leaf_nodes
        assert len(tree.data_nodes) == expected_data_nodes
        assert len(tree.virtual_nodes) == expected_virtual_nodes

    def _run_tree_case(self, tree, example, data, auto_stack, auto_unbind,
                       inline, expected_shape, expected_leaf_nodes,
                       expected_data_nodes, expected_virtual_nodes,
                       canonicalize):
        # Reuse the same assertions before and after canonicalization to ensure
        # the normalization step does not change observable behavior.
        tree.auto_stack = auto_stack
        tree.auto_unbind = auto_unbind

        self._assert_tree_algorithm(tree, example, data, inline,
                                    expected_shape, expected_leaf_nodes,
                                    expected_data_nodes, expected_virtual_nodes)

        if canonicalize:
            for mode, sv_cut in TREE_CANONICALIZE_CASES:
                tree.canonicalize(mode=mode, **sv_cut)
                self._assert_tree_algorithm(
                    tree, example, data, inline, expected_shape,
                    expected_leaf_nodes, expected_data_nodes,
                    expected_virtual_nodes
                )


class TestTree(_TreeTestMixin):

    @pytest.mark.parametrize(
        '_,sites_per_layer,bond_dim,example_shape,data_shape,expected_shape,'
        'expected_leaf_nodes,expected_data_nodes,stacked_nodes',
        TREE_CASES,
    )
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline', INLINE_CASES)
    def test_tree_algorithms(self, _, sites_per_layer, bond_dim, example_shape,
                             data_shape, expected_shape, expected_leaf_nodes,
                             expected_data_nodes, stacked_nodes, auto_stack,
                             auto_unbind, inline):
        example = torch.randn(*example_shape)
        data = torch.randn(*data_shape)

        tree = tk.models.Tree(sites_per_layer=sites_per_layer,
                              bond_dim=bond_dim)
        expected_virtual_nodes = self._expected_virtual_nodes(
            auto_stack, inline, stacked_nodes
        )
        self._run_tree_case(tree, example, data, auto_stack, auto_unbind,
                            inline, expected_shape, expected_leaf_nodes,
                            expected_data_nodes, expected_virtual_nodes,
                            canonicalize=True)


class TestUTree:

    @staticmethod
    def _run_utree_case(tree, example, data, auto_stack, auto_unbind, inline,
                        expected_shape, expected_leaf_nodes,
                        expected_data_nodes):
        tree.auto_stack = auto_stack
        tree.auto_unbind = auto_unbind

        tree.trace(example, inline=inline)
        result = tree(data, inline=inline)

        assert result.shape == expected_shape
        assert len(tree.edges) == 1
        assert len(tree.leaf_nodes) == expected_leaf_nodes
        assert len(tree.data_nodes) == expected_data_nodes
        assert len(tree.virtual_nodes) == 2

    @pytest.mark.parametrize(
        '_,sites_per_layer,bond_dim,example_shape,data_shape,expected_shape,'
        'expected_leaf_nodes,expected_data_nodes',
        UTREE_CASES,
    )
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline', INLINE_CASES)
    def test_utree_algorithms(self, _, sites_per_layer, bond_dim, example_shape,
                              data_shape, expected_shape, expected_leaf_nodes,
                              expected_data_nodes, auto_stack, auto_unbind,
                              inline):
        example = torch.randn(*example_shape)
        data = torch.randn(*data_shape)

        tree = tk.models.UTree(sites_per_layer=sites_per_layer,
                               bond_dim=bond_dim)
        self._run_utree_case(tree, example, data, auto_stack, auto_unbind,
                             inline, expected_shape, expected_leaf_nodes,
                             expected_data_nodes)


class TestConvTree(_TreeTestMixin):

    @pytest.mark.parametrize(
        '_,sites_per_layer,bond_dim,kernel_size,expected_shape,'
        'expected_leaf_nodes,expected_data_nodes,stacked_nodes',
        CONV_TREE_CASES,
    )
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline', INLINE_CASES)
    def test_conv_tree_algorithms(self, _, sites_per_layer, bond_dim,
                                  kernel_size, expected_shape,
                                  expected_leaf_nodes, expected_data_nodes,
                                  stacked_nodes, auto_stack, auto_unbind,
                                  inline):
        example, data = self._image_inputs()

        tree = tk.models.ConvTree(sites_per_layer=sites_per_layer,
                                  bond_dim=bond_dim,
                                  kernel_size=kernel_size)
        expected_virtual_nodes = self._expected_virtual_nodes(
            auto_stack, inline, stacked_nodes
        )
        self._run_tree_case(tree, example, data, auto_stack, auto_unbind,
                            inline, expected_shape, expected_leaf_nodes,
                            expected_data_nodes, expected_virtual_nodes,
                            canonicalize=True)


class TestConvUTree:

    @staticmethod
    def _run_conv_utree_case(tree, example, data, auto_stack, auto_unbind,
                             inline, expected_shape, expected_leaf_nodes,
                             expected_data_nodes):
        tree.auto_stack = auto_stack
        tree.auto_unbind = auto_unbind

        tree.trace(example, inline=inline)
        result = tree(data, inline=inline)

        assert result.shape == expected_shape
        assert len(tree.edges) == 1
        assert len(tree.leaf_nodes) == expected_leaf_nodes
        assert len(tree.data_nodes) == expected_data_nodes
        assert len(tree.virtual_nodes) == 2

    @pytest.mark.parametrize(
        '_,sites_per_layer,bond_dim,kernel_size,expected_shape,'
        'expected_leaf_nodes,expected_data_nodes',
        CONV_UTREE_CASES,
    )
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('inline', INLINE_CASES)
    def test_conv_utree_algorithms(self, _, sites_per_layer, bond_dim,
                                   kernel_size, expected_shape,
                                   expected_leaf_nodes, expected_data_nodes,
                                   auto_stack, auto_unbind, inline):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        tree = tk.models.ConvUTree(sites_per_layer=sites_per_layer,
                                   bond_dim=bond_dim,
                                   kernel_size=kernel_size)
        self._run_conv_utree_case(tree, example, data, auto_stack,
                                  auto_unbind, inline, expected_shape,
                                  expected_leaf_nodes, expected_data_nodes)
