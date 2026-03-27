"""
Tests for peps:

    * TestPEPS
    * TestUPEPS
    * TestConvPEPS
    * TestConvUPEPS
"""

import pytest

import torch
import tensorkrowch as tk

AUTO_BOOL_CASES = [True, False]
SIDE_CASES = ['up', 'down', 'left', 'right']
BOUNDARY_CASES = ['obc', 'pbc']
PEPS_EXTREME_CASES = [
    ((2, 2), ['obc', 'obc'], 1),
    ((1, 3), ['pbc', 'obc'], {'stacked': 4, 'default': 1}),
    ((1, 2), ['pbc', 'obc'], {'stacked': 3, 'default': 1}),
    ((3, 1), ['obc', 'pbc'], {'stacked': 4, 'default': 1}),
    ((2, 1), ['obc', 'pbc'], {'stacked': 3, 'default': 1}),
    ((1, 1), ['pbc', 'pbc'], {'stacked': 2, 'default': 1}),
]
UPEPS_EXTREME_CASES = [
    ((2, 2), (100,)),
    ((1, 2), (100,)),
    ((2, 1), (100,)),
    ((1, 1), (100,)),
]
CONV_PEPS_EXTREME_CASES = [
    ((2, 2), ['obc', 'obc'], (100, 4, 4), 1),
    ((1, 3), ['pbc', 'obc'], (100, 5, 3), {'stacked': 4, 'default': 1}),
    ((1, 2), ['pbc', 'obc'], (100, 5, 4), {'stacked': 3, 'default': 1}),
    ((3, 1), ['obc', 'pbc'], (100, 3, 5), {'stacked': 4, 'default': 1}),
    ((2, 1), ['obc', 'pbc'], (100, 4, 5), {'stacked': 3, 'default': 1}),
    ((1, 1), ['pbc', 'pbc'], (100, 5, 5), {'stacked': 2, 'default': 1}),
]
CONV_UPEPS_EXTREME_CASES = [
    ((2, 2), (100, 4, 4)),
    ((1, 2), (100, 5, 4)),
    ((2, 1), (100, 4, 5)),
    ((1, 1), (100, 5, 5)),
]


def _assert_initialized_parameterization(nodes, parameterized, tensor_address=None):
    expected_type = tk.ParamNode if parameterized else tk.Node
    for node in nodes:
        assert isinstance(node, expected_type)
        assert isinstance(node.tensor, torch.nn.Parameter) == parameterized
        if tensor_address is not None:
            assert node.tensor_address() == tensor_address


class _PEPSTestMixin:

    @staticmethod
    def _expected_virtual_nodes(boundary_0, boundary_1, auto_stack):
        # The number of stacked helper nodes depends only on the boundary layout
        # once auto-stacking is enabled.
        if not auto_stack:
            return 1
        if boundary_0 == 'obc' and boundary_1 == 'obc':
            return 6
        if boundary_0 == 'pbc' and boundary_1 == 'pbc':
            return 2
        return 4

    @staticmethod
    def _resolve_virtual_nodes(expected_virtual_nodes, auto_stack):
        # Some extreme geometries have bespoke stacked/default expectations.
        if isinstance(expected_virtual_nodes, dict):
            return expected_virtual_nodes['stacked'] if auto_stack \
                else expected_virtual_nodes['default']
        return expected_virtual_nodes


class TestPEPS(_PEPSTestMixin):  # MARK: TestPEPS

    @pytest.mark.parametrize('parameterized', AUTO_BOOL_CASES)
    def test_initialize_parameterized(self, parameterized):
        peps = tk.models.PEPS(n_rows=2,
                              n_cols=3,
                              in_dim=5,
                              bond_dim=[2, 3],
                              boundary=['obc', 'obc'],
                              parameterized=parameterized)

        nodes = []
        for lst in peps.grid_env:
            nodes.extend(lst)
        _assert_initialized_parameterization(nodes, parameterized)

    @pytest.mark.parametrize('boundary_0', BOUNDARY_CASES)
    @pytest.mark.parametrize('boundary_1', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline', AUTO_BOOL_CASES)
    def test_all_algorithms(self, boundary_0, boundary_1,
                            auto_stack, auto_unbind, side, inline):
        example = torch.randn(1, 12, 5)
        data = torch.randn(100, 12, 5)

        peps = tk.models.PEPS(n_rows=3,
                              n_cols=4,
                              in_dim=5,
                              bond_dim=[2, 3],
                              boundary=[boundary_0, boundary_1])
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, inline=inline)
        result = peps(data, from_side=side, inline=inline)

        assert result.shape == (100,)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == 12
        assert len(peps.data_nodes) == 12
        assert len(peps.virtual_nodes) == self._expected_virtual_nodes(
            boundary_0, boundary_1, auto_stack
        )

    @pytest.mark.parametrize('kernel_size,boundary,expected_virtual_nodes',
                             PEPS_EXTREME_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline', AUTO_BOOL_CASES)
    def test_extreme_cases(self, kernel_size, boundary, expected_virtual_nodes,
                           auto_stack, auto_unbind, side, inline):
        # Keep the original degenerate shapes (single row/column/site) while
        # expressing them as one table of scenarios.
        n_rows, n_cols = kernel_size
        example = torch.randn(1, n_rows * n_cols, 5)
        data = torch.randn(100, n_rows * n_cols, 5)

        peps = tk.models.PEPS(n_rows=n_rows,
                              n_cols=n_cols,
                              in_dim=5,
                              bond_dim=[2, 3],
                              boundary=boundary)
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, inline=inline)
        result = peps(data, from_side=side, inline=inline)

        assert result.shape == (100,)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == n_rows * n_cols
        assert len(peps.data_nodes) == n_rows * n_cols
        assert len(peps.virtual_nodes) == self._resolve_virtual_nodes(
            expected_virtual_nodes, auto_stack
        )


class TestUPEPS:  # MARK: TestUPEPS

    @pytest.mark.parametrize('parameterized', AUTO_BOOL_CASES)
    def test_initialize_parameterized(self, parameterized):
        peps = tk.models.UPEPS(n_rows=2,
                               n_cols=3,
                               in_dim=5,
                               bond_dim=[2, 3],
                               parameterized=parameterized)

        nodes = []
        for lst in peps.grid_env:
            nodes.extend(lst)
        _assert_initialized_parameterization(nodes,
                                             parameterized,
                                             tensor_address='virtual_uniform')
        _assert_initialized_parameterization([peps.uniform_memory], parameterized)

    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline', AUTO_BOOL_CASES)
    def test_all_algorithms(self, auto_stack, auto_unbind, side, inline):
        example = torch.randn(1, 12, 5)
        data = torch.randn(100, 12, 5)

        peps = tk.models.UPEPS(n_rows=3,
                               n_cols=4,
                               in_dim=5,
                               bond_dim=[2, 3])
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, inline=inline)
        result = peps(data, from_side=side, inline=inline)

        assert result.shape == (100,)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == 12
        assert len(peps.data_nodes) == 12
        assert len(peps.virtual_nodes) == 2

    @pytest.mark.parametrize('kernel_size,expected_shape', UPEPS_EXTREME_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline', AUTO_BOOL_CASES)
    def test_extreme_cases(self, kernel_size, expected_shape, auto_stack,
                           auto_unbind, side, inline):
        n_rows, n_cols = kernel_size
        example = torch.randn(1, n_rows * n_cols, 5)
        data = torch.randn(100, n_rows * n_cols, 5)

        peps = tk.models.UPEPS(n_rows=n_rows,
                               n_cols=n_cols,
                               in_dim=5,
                               bond_dim=[2, 3])
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, inline=inline)
        result = peps(data, from_side=side, inline=inline)

        assert result.shape == expected_shape
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == n_rows * n_cols
        assert len(peps.data_nodes) == n_rows * n_cols
        assert len(peps.virtual_nodes) == 2


class TestConvPEPS(_PEPSTestMixin):  # MARK: TestConvPEPS

    @pytest.mark.parametrize('parameterized', AUTO_BOOL_CASES)
    def test_initialize_parameterized(self, parameterized):
        peps = tk.models.ConvPEPS(in_channels=2,
                                  bond_dim=[2, 3],
                                  kernel_size=2,
                                  boundary=['obc', 'obc'],
                                  parameterized=parameterized)

        nodes = []
        for lst in peps.grid_env:
            nodes.extend(lst)
        _assert_initialized_parameterization(nodes, parameterized)

    @pytest.mark.parametrize('boundary_0', BOUNDARY_CASES)
    @pytest.mark.parametrize('boundary_1', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline', AUTO_BOOL_CASES)
    def test_all_algorithms(self, boundary_0, boundary_1,
                            auto_stack, auto_unbind, side, inline):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvPEPS(in_channels=2,
                                  bond_dim=[2, 3],
                                  kernel_size=3,
                                  boundary=[boundary_0, boundary_1])
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, inline=inline, max_bond=8)
        result = peps(data, from_side=side, inline=inline, max_bond=8)

        assert result.shape == (100, 3, 3)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == 9
        assert len(peps.data_nodes) == 9
        assert len(peps.virtual_nodes) == self._expected_virtual_nodes(
            boundary_0, boundary_1, auto_stack
        )

    @pytest.mark.parametrize('kernel_size,boundary,expected_shape,expected_virtual_nodes',
                             CONV_PEPS_EXTREME_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline', AUTO_BOOL_CASES)
    def test_extreme_cases(self, kernel_size, boundary, expected_shape,
                           expected_virtual_nodes, auto_stack, auto_unbind,
                           side, inline):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)
        n_rows, n_cols = kernel_size

        peps = tk.models.ConvPEPS(in_channels=2,
                                  bond_dim=[2, 3],
                                  kernel_size=kernel_size,
                                  boundary=boundary)
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, inline=inline)
        result = peps(data, from_side=side, inline=inline)

        assert result.shape == expected_shape
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == n_rows * n_cols
        assert len(peps.data_nodes) == n_rows * n_cols
        assert len(peps.virtual_nodes) == self._resolve_virtual_nodes(
            expected_virtual_nodes, auto_stack
        )


class TestConvUPEPS:  # MARK: TestConvUPEPS

    @pytest.mark.parametrize('parameterized', AUTO_BOOL_CASES)
    def test_initialize_parameterized(self, parameterized):
        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=2,
                                   parameterized=parameterized)

        nodes = []
        for lst in peps.grid_env:
            nodes.extend(lst)
        _assert_initialized_parameterization(nodes,
                                             parameterized,
                                             tensor_address='virtual_uniform')
        _assert_initialized_parameterization([peps.uniform_memory], parameterized)

    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline', AUTO_BOOL_CASES)
    def test_all_algorithms(self, auto_stack, auto_unbind, side, inline):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=3)
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, inline=inline, max_bond=8)
        result = peps(data, from_side=side, inline=inline, max_bond=8)

        assert result.shape == (100, 3, 3)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == 9
        assert len(peps.data_nodes) == 9
        assert len(peps.virtual_nodes) == 2

    @pytest.mark.parametrize('kernel_size,expected_shape', CONV_UPEPS_EXTREME_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline', AUTO_BOOL_CASES)
    def test_extreme_cases(self, kernel_size, expected_shape, auto_stack,
                           auto_unbind, side, inline):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)
        n_rows, n_cols = kernel_size

        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=kernel_size)
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, inline=inline)
        result = peps(data, from_side=side, inline=inline)

        assert result.shape == expected_shape
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == n_rows * n_cols
        assert len(peps.data_nodes) == n_rows * n_cols
        assert len(peps.virtual_nodes) == 2
