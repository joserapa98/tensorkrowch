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
BOUNDARY_PAIR_CASES = [
    ['obc', 'obc'],
    ['obc', 'pbc'],
    ['pbc', 'obc'],
    ['pbc', 'pbc'],
]
PEPS_EXTREME_CASES = [
    ((2, 2), ['obc', 'obc'], {'stacked': 2, 'default': 1}),
    ((2, 3), ['obc', 'obc'], {'stacked': 2, 'default': 1}),
    ((3, 2), ['obc', 'obc'], {'stacked': 2, 'default': 1}),
    ((3, 3), ['obc', 'obc'], {'stacked': 2, 'default': 1}),
    ((1, 3), ['pbc', 'obc'], {'stacked': 2, 'default': 1}),
    ((1, 2), ['pbc', 'obc'], {'stacked': 2, 'default': 1}),
    ((3, 1), ['obc', 'pbc'], {'stacked': 2, 'default': 1}),
    ((2, 1), ['obc', 'pbc'], {'stacked': 2, 'default': 1}),
    ((1, 1), ['pbc', 'pbc'], {'stacked': 2, 'default': 1}),
]
UPEPS_EXTREME_CASES = [
    ((2, 2), (100,)),
    ((2, 3), (100,)),
    ((3, 2), (100,)),
    ((3, 3), (100,)),
    ((1, 2), (100,)),
    ((2, 1), (100,)),
    ((1, 1), (100,)),
]
CONV_PEPS_EXTREME_CASES = [
    ((2, 2), ['obc', 'obc'], (100, 4, 4), {'stacked': 2, 'default': 1}),
    ((2, 3), ['obc', 'obc'], (100, 4, 3), {'stacked': 2, 'default': 1}),
    ((3, 2), ['obc', 'obc'], (100, 3, 4), {'stacked': 2, 'default': 1}),
    ((3, 3), ['obc', 'obc'], (100, 3, 3), {'stacked': 2, 'default': 1}),
    ((1, 3), ['pbc', 'obc'], (100, 5, 3), {'stacked': 2, 'default': 1}),
    ((1, 2), ['pbc', 'obc'], (100, 5, 4), {'stacked': 2, 'default': 1}),
    ((3, 1), ['obc', 'pbc'], (100, 3, 5), {'stacked': 2, 'default': 1}),
    ((2, 1), ['obc', 'pbc'], (100, 4, 5), {'stacked': 2, 'default': 1}),
    ((1, 1), ['pbc', 'pbc'], (100, 5, 5), {'stacked': 2, 'default': 1}),
]
CONV_UPEPS_EXTREME_CASES = [
    ((2, 2), (100, 4, 4)),
    ((2, 3), (100, 4, 3)),
    ((3, 2), (100, 3, 4)),
    ((3, 3), (100, 3, 3)),
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


def _grid_nodes(peps):
    return [node for row in peps.grid_env for node in row]


def _border_nodes(peps):
    return peps.up_border + peps.down_border + \
        peps.left_border + peps.right_border


def _assert_copied_peps(peps, copied_peps, share_tensors):
    assert peps.n_rows == copied_peps.n_rows
    assert peps.n_cols == copied_peps.n_cols
    assert peps.phys_dim == copied_peps.phys_dim
    assert peps.bond_dim == copied_peps.bond_dim
    assert peps.boundary == copied_peps.boundary
    assert peps.n_batches == copied_peps.n_batches

    for node, copied_node in zip(_grid_nodes(peps), _grid_nodes(copied_peps)):
        assert isinstance(copied_node, tk.ParamNode) == \
            isinstance(node, tk.ParamNode)
        assert isinstance(copied_node.tensor, torch.nn.Parameter) == \
            isinstance(node.tensor, torch.nn.Parameter)
        if share_tensors:
            assert node.tensor is copied_node.tensor
        else:
            assert node.tensor is not copied_node.tensor

    for node, copied_node in zip(_border_nodes(peps), _border_nodes(copied_peps)):
        assert isinstance(copied_node, tk.Node)
        assert not isinstance(copied_node, tk.ParamNode)
        if share_tensors:
            assert node.tensor is copied_node.tensor
        else:
            assert node.tensor is not copied_node.tensor


def _deparameterize_even_grid_nodes(peps):
    expected_param_flags = []
    for i, node in enumerate(_grid_nodes(peps)):
        set_param = (i % 2) == 1
        row = i // peps.n_cols
        col = i % peps.n_cols
        peps._grid_env[row][col] = node.parameterize(set_param=set_param)
        expected_param_flags.append(set_param)
    return expected_param_flags


def _assert_parameterization_pattern(nodes, expected_param_flags):
    assert len(nodes) == len(expected_param_flags)
    for node, is_param in zip(nodes, expected_param_flags):
        assert isinstance(node, tk.ParamNode) == is_param
        assert isinstance(node.tensor, torch.nn.Parameter) == is_param


def _assert_deparameterized_nodes(nodes, tensor_address=None):
    for node in nodes:
        assert isinstance(node, tk.Node)
        assert not isinstance(node.tensor, torch.nn.Parameter)
        if tensor_address is not None:
            assert node.tensor_address() == tensor_address


class _PEPSTestMixin:

    @staticmethod
    def _expected_leaf_nodes(n_rows, n_cols, boundary_0, boundary_1):
        count = n_rows * n_cols
        if boundary_0 == 'obc':
            count += 2 * n_cols
        if boundary_1 == 'obc':
            count += 2 * n_rows
        return count

    @staticmethod
    def _expected_virtual_nodes(boundary_0, boundary_1, auto_stack,
                                inline_input=False):
        # With the homogeneous grid layout, auto-stacking uses the same helper
        # structure regardless of the boundary closure.
        if not auto_stack:
            return 1
        if inline_input:
            return 1
        return 2

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
                              phys_dim=5,
                              bond_dim=[2, 3],
                              boundary=['obc', 'obc'],
                              parameterized=parameterized)

        nodes = []
        for lst in peps.grid_env:
            nodes.extend(lst)
        _assert_initialized_parameterization(nodes, parameterized)

    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy(self, boundary, share_tensors):
        peps = tk.models.PEPS(n_rows=2,
                              n_cols=3,
                              phys_dim=5,
                              bond_dim=[2, 3],
                              boundary=boundary)

        copied_peps = peps.copy(share_tensors=share_tensors)

        assert isinstance(copied_peps, tk.models.PEPS)
        _assert_copied_peps(peps, copied_peps, share_tensors)

    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy_preserves_mixed_parameterization(self, boundary, share_tensors):
        peps = tk.models.PEPS(n_rows=2,
                              n_cols=3,
                              phys_dim=5,
                              bond_dim=[2, 3],
                              boundary=boundary)

        expected_param_flags = _deparameterize_even_grid_nodes(peps)

        copied_peps = peps.copy(share_tensors=share_tensors)

        _assert_parameterization_pattern(_grid_nodes(copied_peps),
                                         expected_param_flags)

    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    @pytest.mark.parametrize('override', AUTO_BOOL_CASES)
    def test_deparameterize(self, boundary, override):
        peps = tk.models.PEPS(n_rows=2,
                              n_cols=3,
                              phys_dim=5,
                              bond_dim=[2, 3],
                              boundary=boundary)

        non_param_peps = peps.parameterize(set_param=False, override=override)

        if override:
            assert non_param_peps is peps
        else:
            assert non_param_peps is not peps

        _assert_deparameterized_nodes(_grid_nodes(non_param_peps))
        _assert_deparameterized_nodes(_border_nodes(non_param_peps))

    @pytest.mark.parametrize('boundary_0', BOUNDARY_CASES)
    @pytest.mark.parametrize('boundary_1', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    def test_all_algorithms(self, boundary_0, boundary_1,
                            auto_stack, auto_unbind, side,
                            inline_input):
        example = torch.randn(1, 12, 5)
        data = torch.randn(100, 12, 5)

        peps = tk.models.PEPS(n_rows=3,
                              n_cols=4,
                              phys_dim=5,
                              bond_dim=[2, 3],
                              boundary=[boundary_0, boundary_1])
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, inline_input=inline_input)
        result = peps(data,
                      from_side=side,
                      inline_input=inline_input)

        assert result.shape == (100,)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == self._expected_leaf_nodes(
            3, 4, boundary_0, boundary_1
        )
        assert len(peps.data_nodes) == 12
        assert len(peps.virtual_nodes) == self._expected_virtual_nodes(
            boundary_0, boundary_1, auto_stack, inline_input=inline_input
        )

    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    def test_3x3_all_boundaries(self, boundary, auto_stack, auto_unbind,
                                side):
        example = torch.randn(1, 9, 5)
        data = torch.randn(100, 9, 5)

        peps = tk.models.PEPS(n_rows=3,
                              n_cols=3,
                              phys_dim=5,
                              bond_dim=[2, 3],
                              boundary=boundary)
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side)
        result = peps(data, from_side=side)

        assert result.shape == (100,)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == self._expected_leaf_nodes(
            3, 3, boundary[0], boundary[1]
        )
        assert len(peps.data_nodes) == 9
        assert len(peps.virtual_nodes) == self._expected_virtual_nodes(
            boundary[0], boundary[1], auto_stack
        )

    @pytest.mark.parametrize('kernel_size,boundary,expected_virtual_nodes',
                             PEPS_EXTREME_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    def test_extreme_cases(self, kernel_size, boundary, expected_virtual_nodes,
                           auto_stack, auto_unbind, side):
        # Keep the original degenerate shapes (single row/column/site) while
        # expressing them as one table of scenarios.
        n_rows, n_cols = kernel_size
        example = torch.randn(1, n_rows * n_cols, 5)
        data = torch.randn(100, n_rows * n_cols, 5)

        peps = tk.models.PEPS(n_rows=n_rows,
                              n_cols=n_cols,
                              phys_dim=5,
                              bond_dim=[2, 3],
                              boundary=boundary)
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side)
        result = peps(data, from_side=side)

        assert result.shape == (100,)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == self._expected_leaf_nodes(
            n_rows, n_cols, boundary[0], boundary[1]
        )
        assert len(peps.data_nodes) == n_rows * n_cols
        assert len(peps.virtual_nodes) == self._resolve_virtual_nodes(
            expected_virtual_nodes, auto_stack
        )


class TestUPEPS(_PEPSTestMixin):  # MARK: TestUPEPS

    @pytest.mark.parametrize('parameterized', AUTO_BOOL_CASES)
    def test_initialize_parameterized(self, parameterized):
        peps = tk.models.UPEPS(n_rows=2,
                               n_cols=3,
                               phys_dim=5,
                               bond_dim=[2, 3],
                               parameterized=parameterized)

        nodes = []
        for lst in peps.grid_env:
            nodes.extend(lst)
        _assert_initialized_parameterization(nodes,
                                             parameterized,
                                             tensor_address='virtual_uniform')
        _assert_initialized_parameterization([peps.uniform_memory], parameterized)

    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy(self, share_tensors):
        peps = tk.models.UPEPS(n_rows=2,
                               n_cols=3,
                               phys_dim=5,
                               bond_dim=[2, 3])

        copied_peps = peps.copy(share_tensors=share_tensors)

        assert isinstance(copied_peps, tk.models.UPEPS)
        _assert_copied_peps(peps, copied_peps, share_tensors)
        if share_tensors:
            assert peps.uniform_memory.tensor is copied_peps.uniform_memory.tensor
        else:
            assert peps.uniform_memory.tensor is not copied_peps.uniform_memory.tensor

    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy_preserves_parameterization(self, share_tensors):
        peps = tk.models.UPEPS(n_rows=2,
                               n_cols=3,
                               phys_dim=5,
                               bond_dim=[2, 3])
        peps = peps.parameterize(set_param=False, override=True)

        copied_peps = peps.copy(share_tensors=share_tensors)

        _assert_initialized_parameterization(_grid_nodes(copied_peps),
                                             parameterized=False,
                                             tensor_address='virtual_uniform')
        _assert_initialized_parameterization([copied_peps.uniform_memory],
                                             parameterized=False)

    @pytest.mark.parametrize('override', AUTO_BOOL_CASES)
    def test_deparameterize(self, override):
        peps = tk.models.UPEPS(n_rows=2,
                               n_cols=3,
                               phys_dim=5,
                               bond_dim=[2, 3])

        non_param_peps = peps.parameterize(set_param=False, override=override)

        if override:
            assert non_param_peps is peps
        else:
            assert non_param_peps is not peps

        _assert_deparameterized_nodes(_grid_nodes(non_param_peps),
                                      tensor_address='virtual_uniform')
        _assert_deparameterized_nodes([non_param_peps.uniform_memory])

    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    def test_all_algorithms(self, auto_stack, auto_unbind, side,
                            inline_input):
        example = torch.randn(1, 12, 5)
        data = torch.randn(100, 12, 5)

        peps = tk.models.UPEPS(n_rows=3,
                               n_cols=4,
                               phys_dim=5,
                               bond_dim=[2, 3])
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, inline_input=inline_input)
        result = peps(data,
                      from_side=side,
                      inline_input=inline_input)

        assert result.shape == (100,)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == 12
        assert len(peps.data_nodes) == 12
        assert len(peps.virtual_nodes) == 2

    @pytest.mark.parametrize('kernel_size,expected_shape', UPEPS_EXTREME_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    def test_extreme_cases(self, kernel_size, expected_shape, auto_stack,
                           auto_unbind, side):
        n_rows, n_cols = kernel_size
        example = torch.randn(1, n_rows * n_cols, 5)
        data = torch.randn(100, n_rows * n_cols, 5)

        peps = tk.models.UPEPS(n_rows=n_rows,
                               n_cols=n_cols,
                               phys_dim=5,
                               bond_dim=[2, 3])
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side)
        result = peps(data, from_side=side)

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

    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy(self, boundary, share_tensors):
        peps = tk.models.ConvPEPS(in_channels=2,
                                  bond_dim=[2, 3],
                                  kernel_size=(2, 3),
                                  boundary=boundary)

        copied_peps = peps.copy(share_tensors=share_tensors)

        assert isinstance(copied_peps, tk.models.ConvPEPS)
        assert peps.in_channels == copied_peps.in_channels
        assert peps.kernel_size == copied_peps.kernel_size
        assert peps.stride == copied_peps.stride
        assert peps.padding == copied_peps.padding
        assert peps.dilation == copied_peps.dilation
        _assert_copied_peps(peps, copied_peps, share_tensors)

    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy_preserves_mixed_parameterization(self, boundary, share_tensors):
        peps = tk.models.ConvPEPS(in_channels=2,
                                  bond_dim=[2, 3],
                                  kernel_size=(2, 3),
                                  boundary=boundary)

        expected_param_flags = _deparameterize_even_grid_nodes(peps)

        copied_peps = peps.copy(share_tensors=share_tensors)

        _assert_parameterization_pattern(_grid_nodes(copied_peps),
                                         expected_param_flags)

    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    @pytest.mark.parametrize('override', AUTO_BOOL_CASES)
    def test_deparameterize(self, boundary, override):
        peps = tk.models.ConvPEPS(in_channels=2,
                                  bond_dim=[2, 3],
                                  kernel_size=(2, 3),
                                  boundary=boundary)

        non_param_peps = peps.parameterize(set_param=False, override=override)

        if override:
            assert non_param_peps is peps
        else:
            assert non_param_peps is not peps

        assert isinstance(non_param_peps, tk.models.ConvPEPS)
        assert peps.in_channels == non_param_peps.in_channels
        assert peps.kernel_size == non_param_peps.kernel_size
        _assert_deparameterized_nodes(_grid_nodes(non_param_peps))
        _assert_deparameterized_nodes(_border_nodes(non_param_peps))

    @pytest.mark.parametrize('boundary_0', BOUNDARY_CASES)
    @pytest.mark.parametrize('boundary_1', BOUNDARY_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    def test_all_algorithms(self, boundary_0, boundary_1,
                            auto_stack, auto_unbind, side,
                            inline_input):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvPEPS(in_channels=2,
                                  bond_dim=[2, 3],
                                  kernel_size=3,
                                  boundary=[boundary_0, boundary_1])
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example,
                   from_side=side,
                   inline_input=inline_input,
                   max_bond=8)
        result = peps(data,
                      from_side=side,
                      inline_input=inline_input,
                      max_bond=8)

        assert result.shape == (100, 3, 3)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == self._expected_leaf_nodes(
            3, 3, boundary_0, boundary_1
        )
        assert len(peps.data_nodes) == 9
        assert len(peps.virtual_nodes) == self._expected_virtual_nodes(
            boundary_0, boundary_1, auto_stack, inline_input=inline_input
        )

    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    def test_3x3_all_boundaries(self, boundary, auto_stack, auto_unbind,
                                side):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvPEPS(in_channels=2,
                                  bond_dim=[2, 3],
                                  kernel_size=3,
                                  boundary=boundary)
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side, max_bond=8)
        result = peps(data, from_side=side, max_bond=8)

        assert result.shape == (100, 3, 3)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == self._expected_leaf_nodes(
            3, 3, boundary[0], boundary[1]
        )
        assert len(peps.data_nodes) == 9
        assert len(peps.virtual_nodes) == self._expected_virtual_nodes(
            boundary[0], boundary[1], auto_stack
        )

    @pytest.mark.parametrize('kernel_size,boundary,expected_shape,expected_virtual_nodes',
                             CONV_PEPS_EXTREME_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    def test_extreme_cases(self, kernel_size, boundary, expected_shape,
                           expected_virtual_nodes, auto_stack, auto_unbind,
                           side):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)
        n_rows, n_cols = kernel_size

        peps = tk.models.ConvPEPS(in_channels=2,
                                  bond_dim=[2, 3],
                                  kernel_size=kernel_size,
                                  boundary=boundary)
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side)
        result = peps(data, from_side=side)

        assert result.shape == expected_shape
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == self._expected_leaf_nodes(
            n_rows, n_cols, boundary[0], boundary[1]
        )
        assert len(peps.data_nodes) == n_rows * n_cols
        assert len(peps.virtual_nodes) == self._resolve_virtual_nodes(
            expected_virtual_nodes, auto_stack
        )


class TestConvUPEPS(_PEPSTestMixin):  # MARK: TestConvUPEPS

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

    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy(self, share_tensors):
        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=(2, 3))

        copied_peps = peps.copy(share_tensors=share_tensors)

        assert isinstance(copied_peps, tk.models.ConvUPEPS)
        assert peps.in_channels == copied_peps.in_channels
        assert peps.kernel_size == copied_peps.kernel_size
        assert peps.stride == copied_peps.stride
        assert peps.padding == copied_peps.padding
        assert peps.dilation == copied_peps.dilation
        _assert_copied_peps(peps, copied_peps, share_tensors)
        if share_tensors:
            assert peps.uniform_memory.tensor is copied_peps.uniform_memory.tensor
        else:
            assert peps.uniform_memory.tensor is not copied_peps.uniform_memory.tensor

    @pytest.mark.parametrize('share_tensors', AUTO_BOOL_CASES)
    def test_copy_preserves_parameterization(self, share_tensors):
        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=(2, 3))
        peps = peps.parameterize(set_param=False, override=True)

        copied_peps = peps.copy(share_tensors=share_tensors)

        _assert_initialized_parameterization(_grid_nodes(copied_peps),
                                             parameterized=False,
                                             tensor_address='virtual_uniform')
        _assert_initialized_parameterization([copied_peps.uniform_memory],
                                             parameterized=False)

    @pytest.mark.parametrize('override', AUTO_BOOL_CASES)
    def test_deparameterize(self, override):
        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=(2, 3))

        non_param_peps = peps.parameterize(set_param=False, override=override)

        if override:
            assert non_param_peps is peps
        else:
            assert non_param_peps is not peps

        assert isinstance(non_param_peps, tk.models.ConvUPEPS)
        assert peps.in_channels == non_param_peps.in_channels
        assert peps.kernel_size == non_param_peps.kernel_size
        _assert_deparameterized_nodes(_grid_nodes(non_param_peps),
                                      tensor_address='virtual_uniform')
        _assert_deparameterized_nodes([non_param_peps.uniform_memory])

    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    @pytest.mark.parametrize('inline_input', AUTO_BOOL_CASES)
    def test_all_algorithms(self, auto_stack, auto_unbind, side,
                            inline_input):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)

        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=3)
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example,
                   from_side=side,
                   inline_input=inline_input,
                   max_bond=8)
        result = peps(data,
                      from_side=side,
                      inline_input=inline_input,
                      max_bond=8)

        assert result.shape == (100, 3, 3)
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == 9
        assert len(peps.data_nodes) == 9
        assert len(peps.virtual_nodes) == 2

    @pytest.mark.parametrize('kernel_size,expected_shape', CONV_UPEPS_EXTREME_CASES)
    @pytest.mark.parametrize('auto_stack', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('auto_unbind', AUTO_BOOL_CASES)
    @pytest.mark.parametrize('side', SIDE_CASES)
    def test_extreme_cases(self, kernel_size, expected_shape, auto_stack,
                           auto_unbind, side):
        example = tk.embeddings.add_ones(torch.randn(1, 5, 5), axis=1)
        data = tk.embeddings.add_ones(torch.randn(100, 5, 5), axis=1)
        n_rows, n_cols = kernel_size

        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=kernel_size)
        peps.auto_stack = auto_stack
        peps.auto_unbind = auto_unbind

        peps.trace(example, from_side=side)
        result = peps(data, from_side=side)

        assert result.shape == expected_shape
        assert len(peps.edges) == 0
        assert len(peps.leaf_nodes) == n_rows * n_cols
        assert len(peps.data_nodes) == n_rows * n_cols
        assert len(peps.virtual_nodes) == 2
