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
GRID_SIZE_CASES = [(n_rows, n_cols)
                   for n_rows in range(1, 4)
                   for n_cols in range(1, 4)]
DEVICE_CASES = ['cpu', 'cuda', 'mps']
DTYPE_CASES = [torch.float32, torch.complex64]


def _expected_bond_dim(grid_size, boundary, bond_dim=(2, 3)):
    n_rows, n_cols = grid_size
    expected = list(bond_dim)
    if (boundary[1] == 'obc') and (n_cols == 1):
        expected[0] = 1
    if (boundary[0] == 'obc') and (n_rows == 1):
        expected[1] = 1
    return expected


GRID_BOUNDARY_CASES = [
    (grid_size, boundary)
    for grid_size in GRID_SIZE_CASES
    for boundary in BOUNDARY_PAIR_CASES
]
VALID_GRID_BOUNDARY_CASES = GRID_BOUNDARY_CASES
PEPS_EXTREME_CASES = [
    (grid_size, boundary, {'stacked': 2, 'default': 1})
    for grid_size, boundary in VALID_GRID_BOUNDARY_CASES
]
UPEPS_EXTREME_CASES = [(grid_size, (100,))
                       for grid_size in GRID_SIZE_CASES]
CONV_PEPS_EXTREME_CASES = [
    (grid_size,
     boundary,
     (100, 6 - grid_size[0], 6 - grid_size[1]),
     {'stacked': 2, 'default': 1})
    for grid_size, boundary in VALID_GRID_BOUNDARY_CASES
]
CONV_UPEPS_EXTREME_CASES = [
    (grid_size, (100, 6 - grid_size[0], 6 - grid_size[1]))
    for grid_size in GRID_SIZE_CASES
]


def _assert_initialized_parameterization(nodes, parameterized, tensor_address=None):
    expected_type = tk.ParamNode if parameterized else tk.Node
    for node in nodes:
        assert isinstance(node, expected_type)
        assert isinstance(node.tensor, torch.nn.Parameter) == parameterized
        if tensor_address is not None:
            assert node.tensor_address() == tensor_address


def _device(device_name):
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA is not available')
        return torch.device('cuda')
    if device_name == 'mps':
        if not getattr(torch.backends, 'mps', None) or \
                not torch.backends.mps.is_available():
            pytest.skip('MPS is not available')
        return torch.device('mps')
    return torch.device('cpu')


def _runtime_kwargs(device_name, dtype):
    return {'device': _device(device_name), 'dtype': dtype}


def _assert_nodes_runtime(nodes, device, dtype):
    for node in nodes:
        assert node.device.type == device.type
        if device.index is not None:
            assert node.device.index == device.index
        assert node.dtype == dtype
        if dtype.is_complex:
            assert node.is_complex()


def _expected_tensor_shape(n_rows,
                           n_cols,
                           boundary,
                           i,
                           j,
                           phys_dim,
                           bond_dim):
    shape = [phys_dim]
    if (boundary[1] == 'pbc') or (j > 0):
        shape.append(bond_dim[0])
    if (boundary[0] == 'pbc') or (i > 0):
        shape.append(bond_dim[1])
    if (boundary[1] == 'pbc') or (j < n_cols - 1):
        shape.append(bond_dim[0])
    if (boundary[0] == 'pbc') or (i < n_rows - 1):
        shape.append(bond_dim[1])
    return tuple(shape)


def _make_peps_tensors(n_rows=2,
                       n_cols=3,
                       phys_dim=5,
                       bond_dim=(2, 3),
                       boundary=None,
                       **kwargs):
    if boundary is None:
        boundary = ['obc', 'obc']
    return [[
        torch.randn(*_expected_tensor_shape(
            n_rows=n_rows,
            n_cols=n_cols,
            boundary=boundary,
            i=i,
            j=j,
            phys_dim=phys_dim,
            bond_dim=bond_dim),
            **kwargs)
        for j in range(n_cols)]
        for i in range(n_rows)]


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

    @pytest.mark.parametrize('grid_size,boundary', VALID_GRID_BOUNDARY_CASES)
    def test_initialize_with_tensors(self, grid_size, boundary):
        n_rows, n_cols = grid_size
        tensors = _make_peps_tensors(n_rows=n_rows,
                                     n_cols=n_cols,
                                     boundary=boundary)

        peps = tk.models.PEPS(tensors=tensors,
                              boundary=boundary)

        assert peps.n_rows == n_rows
        assert peps.n_cols == n_cols
        assert peps.boundary == boundary
        assert peps.phys_dim == 5
        assert peps.bond_dim == _expected_bond_dim(grid_size, boundary)
        for tensor_row, peps_tensor_row in zip(tensors, peps.tensors):
            for tensor, peps_tensor in zip(tensor_row, peps_tensor_row):
                assert torch.equal(tensor, peps_tensor)

    @pytest.mark.parametrize('device_name', DEVICE_CASES)
    @pytest.mark.parametrize('dtype', DTYPE_CASES)
    @pytest.mark.parametrize('grid_size,boundary', VALID_GRID_BOUNDARY_CASES)
    def test_initialize_with_tensors_runtime(self, device_name, dtype,
                                             grid_size, boundary):
        n_rows, n_cols = grid_size
        tensor_kwargs = _runtime_kwargs(device_name, dtype)
        tensors = _make_peps_tensors(n_rows=n_rows,
                                     n_cols=n_cols,
                                     boundary=boundary,
                                     **tensor_kwargs)

        peps = tk.models.PEPS(tensors=tensors,
                              boundary=boundary)

        _assert_nodes_runtime(_grid_nodes(peps),
                              tensor_kwargs['device'],
                              dtype)
        _assert_nodes_runtime(_border_nodes(peps),
                              tensor_kwargs['device'],
                              dtype)
    
    def test_initialize_with_tensors_ignores_runtime_kwargs(self):
        tensors = _make_peps_tensors(boundary=['pbc', 'pbc'])

        peps = tk.models.PEPS(tensors=tensors,
                              parameterized=False,
                              dtype=torch.complex64)

        for tensor_row, node_row in zip(tensors, peps.grid_env):
            for tensor, node in zip(tensor_row, node_row):
                assert node.tensor is tensor
                assert node.dtype == tensor.dtype

    def test_initialize_with_tensors_ignore_rest(self):
        tensors = _make_peps_tensors(boundary=['pbc', 'pbc'])

        peps = tk.models.PEPS(n_rows=8,
                              n_cols=9,
                              phys_dim=10,
                              bond_dim=[11, 12],
                              boundary=['obc', 'obc'],
                              tensors=tensors)

        assert peps.n_rows == 2
        assert peps.n_cols == 3
        assert peps.boundary == ['pbc', 'pbc']
        assert peps.phys_dim == 5
        assert peps.bond_dim == [2, 3]

    def test_initialize_with_tensors_errors(self):
        tensors = _make_peps_tensors()
        tensors[0] = tensors[0][:-1]
        with pytest.raises(ValueError):
            tk.models.PEPS(tensors=tensors)

        tensors = _make_peps_tensors()
        tensors[0][0] = 1
        with pytest.raises(ValueError):
            tk.models.PEPS(tensors=tensors)

        tensors = _make_peps_tensors()
        tensors[0][0] = torch.randn(6, 2, 3)
        with pytest.raises(ValueError):
            tk.models.PEPS(tensors=tensors)

        tensors = _make_peps_tensors()
        tensors[0][1] = torch.randn(5, 4, 2, 3)
        with pytest.raises(ValueError):
            tk.models.PEPS(tensors=tensors)

        tensors = _make_peps_tensors()
        tensors[1][0] = torch.randn(5, 3, 4, 2)
        with pytest.raises(ValueError):
            tk.models.PEPS(tensors=tensors)
        
        tensors = _make_peps_tensors(boundary=['pbc', 'pbc'])
        tensors[0][1] = torch.randn(5, 4, 3, 4, 3)
        with pytest.raises(ValueError):
            tk.models.PEPS(tensors=tensors)
        
        tensors = _make_peps_tensors(boundary=['pbc', 'pbc'])
        tensors[1][0] = torch.randn(5, 2, 4, 2, 4)
        with pytest.raises(ValueError):
            tk.models.PEPS(tensors=tensors)
        
        # Locally compatible horizontal bonds, but not a single global bond dim.
        tensors = [
            [torch.randn(5, 2, 3, 4, 3),
             torch.randn(5, 4, 3, 2, 3)],
            [torch.randn(5, 2, 3, 4, 3),
             torch.randn(5, 4, 3, 2, 3)]
        ]
        with pytest.raises(ValueError):
            tk.models.PEPS(tensors=tensors)
        
        # Locally compatible vertical bonds, but not a single global bond dim.
        tensors = [
            [torch.randn(5, 2, 3, 2, 4),
             torch.randn(5, 2, 3, 2, 4)],
            [torch.randn(5, 2, 4, 2, 3),
             torch.randn(5, 2, 4, 2, 3)]
        ]
        with pytest.raises(ValueError):
            tk.models.PEPS(tensors=tensors)

    @pytest.mark.parametrize('device_name', DEVICE_CASES)
    @pytest.mark.parametrize('dtype', DTYPE_CASES)
    def test_initialize_runtime(self, device_name, dtype):
        model_kwargs = _runtime_kwargs(device_name, dtype)

        peps = tk.models.PEPS(n_rows=2,
                              n_cols=3,
                              phys_dim=5,
                              bond_dim=[2, 3],
                              **model_kwargs)

        _assert_nodes_runtime(_grid_nodes(peps),
                              model_kwargs['device'],
                              dtype)
        _assert_nodes_runtime(_border_nodes(peps),
                              model_kwargs['device'],
                              dtype)

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

    def test_contract_uses_exact_merge_when_max_bond_is_not_saturated(
            self, monkeypatch):
        data = torch.randn(10, 9, 5)
        exact_calls = 0
        original_exact = tk.models.PEPS._exact_line_contraction

        def exact_spy(self, *args, **kwargs):
            nonlocal exact_calls
            exact_calls += 1
            return original_exact(self, *args, **kwargs)

        def split_error(*args, **kwargs):
            raise AssertionError('SVD split should not be used')

        monkeypatch.setattr(tk.models.PEPS,
                            '_exact_line_contraction',
                            exact_spy)
        monkeypatch.setattr(tk.AbstractNode, 'split', split_error)

        peps = tk.models.PEPS(n_rows=3,
                              n_cols=3,
                              phys_dim=5,
                              bond_dim=[2, 3],
                              boundary=['obc', 'obc'])

        result = peps(data, from_side='up', max_bond=32)

        assert result.shape == (10,)
        assert exact_calls > 0

    def test_contract_uses_svd_when_max_bond_is_saturated(self, monkeypatch):
        data = torch.randn(10, 9, 5)
        split_calls = 0
        original_split = tk.AbstractNode.split

        def exact_error(*args, **kwargs):
            raise AssertionError('Exact merge path should not be used')

        def split_spy(self, *args, **kwargs):
            nonlocal split_calls
            split_calls += 1
            return original_split(self, *args, **kwargs)

        monkeypatch.setattr(tk.models.PEPS,
                            '_exact_line_contraction',
                            exact_error)
        monkeypatch.setattr(tk.AbstractNode, 'split', split_spy)

        peps = tk.models.PEPS(n_rows=3,
                              n_cols=3,
                              phys_dim=5,
                              bond_dim=[2, 3],
                              boundary=['obc', 'obc'])

        result = peps(data, from_side='left', max_bond=8)

        assert result.shape == (10,)
        assert split_calls > 0

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

    @pytest.mark.parametrize('device_name', DEVICE_CASES)
    @pytest.mark.parametrize('dtype', DTYPE_CASES)
    def test_initialize_with_tensor_runtime(self, device_name, dtype):
        tensor_kwargs = _runtime_kwargs(device_name, dtype)
        tensor = torch.randn(5, 2, 3, 2, 3, **tensor_kwargs)

        peps = tk.models.UPEPS(n_rows=2,
                               n_cols=3,
                               phys_dim=5,
                               bond_dim=[2, 3],
                               tensor=tensor)

        assert torch.equal(peps.uniform_memory.tensor, tensor)
        _assert_nodes_runtime(_grid_nodes(peps),
                              tensor_kwargs['device'],
                              dtype)
        _assert_nodes_runtime([peps.uniform_memory],
                              tensor_kwargs['device'],
                              dtype)
    
    def test_initialize_with_tensor_infers_shape(self):
        tensor = torch.randn(5, 2, 3, 2, 3)

        peps = tk.models.UPEPS(n_rows=2,
                               n_cols=3,
                               tensor=tensor,
                               parameterized=False)

        assert peps.phys_dim == 5
        assert peps.bond_dim == [2, 3]
        assert peps.uniform_memory.tensor is tensor
    
    @pytest.mark.parametrize('grid_size,boundary', VALID_GRID_BOUNDARY_CASES)
    def test_initialize_boundary(self, grid_size, boundary):
        n_rows, n_cols = grid_size
        peps = tk.models.UPEPS(n_rows=n_rows,
                               n_cols=n_cols,
                               phys_dim=5,
                               bond_dim=[2, 3],
                               boundary=boundary)

        assert peps.n_rows == n_rows
        assert peps.n_cols == n_cols
        assert peps.boundary == boundary
        assert peps.bond_dim == _expected_bond_dim(grid_size, boundary)
    
    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    def test_initialize_boundary_with_tensor(self, boundary):
        tensor = torch.randn(5, 2, 3, 2, 3)
        peps = tk.models.UPEPS(n_rows=2,
                               n_cols=3,
                               boundary=boundary,
                               tensor=tensor,
                               parameterized=False)

        assert peps.boundary == boundary
        assert peps.phys_dim == 5
        assert peps.bond_dim == [2, 3]
        assert peps.uniform_memory.tensor is tensor
    
    def test_initialize_with_tensors_uses_first_tensor(self):
        first_tensor = torch.randn(5, 2, 3, 2, 3)
        other_tensor = torch.randn(5, 2, 3, 2, 3)
        peps = tk.models.UPEPS(n_rows=2,
                               n_cols=2,
                               phys_dim=5,
                               bond_dim=[2, 3],
                               init_method=None,
                               parameterized=False)

        peps.initialize(tensors=[[first_tensor, other_tensor],
                                 [other_tensor, other_tensor]])

        assert peps.uniform_memory.tensor is first_tensor

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
    
    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    def test_copy_preserves_boundary(self, boundary):
        peps = tk.models.UPEPS(n_rows=2,
                               n_cols=3,
                               phys_dim=5,
                               bond_dim=[2, 3],
                               boundary=boundary)

        copied_peps = peps.copy()

        assert copied_peps.boundary == boundary
        assert copied_peps.bond_dim == [2, 3]

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

    @pytest.mark.parametrize('grid_size,boundary', VALID_GRID_BOUNDARY_CASES)
    def test_initialize_with_tensors(self, grid_size, boundary):
        n_rows, n_cols = grid_size
        tensors = _make_peps_tensors(n_rows=n_rows,
                                     n_cols=n_cols,
                                     phys_dim=2,
                                     boundary=boundary)

        peps = tk.models.ConvPEPS(tensors=tensors,
                                  boundary=boundary)

        assert peps.in_channels == 2
        assert peps.kernel_size == grid_size
        assert peps.boundary == boundary
        assert peps.bond_dim == _expected_bond_dim(grid_size, boundary)

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

    @pytest.mark.parametrize('device_name', DEVICE_CASES)
    @pytest.mark.parametrize('dtype', DTYPE_CASES)
    def test_initialize_with_tensor_runtime(self, device_name, dtype):
        tensor_kwargs = _runtime_kwargs(device_name, dtype)
        tensor = torch.randn(2, 2, 3, 2, 3, **tensor_kwargs)

        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=(2, 3),
                                   tensor=tensor)

        assert torch.equal(peps.uniform_memory.tensor, tensor)
        _assert_nodes_runtime(_grid_nodes(peps),
                              tensor_kwargs['device'],
                              dtype)
        _assert_nodes_runtime([peps.uniform_memory],
                              tensor_kwargs['device'],
                              dtype)
    
    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    def test_initialize_boundary(self, boundary):
        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=(2, 3),
                                   boundary=boundary)

        assert peps.boundary == boundary
        assert peps.bond_dim == [2, 3]
    
    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    def test_initialize_boundary_with_tensor(self, boundary):
        tensor = torch.randn(2, 2, 3, 2, 3)
        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=(2, 3),
                                   boundary=boundary,
                                   tensor=tensor,
                                   parameterized=False)

        assert peps.boundary == boundary
        assert peps.uniform_memory.tensor is tensor

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
    
    @pytest.mark.parametrize('boundary', BOUNDARY_PAIR_CASES)
    def test_copy_preserves_boundary(self, boundary):
        peps = tk.models.ConvUPEPS(in_channels=2,
                                   bond_dim=[2, 3],
                                   kernel_size=(2, 3),
                                   boundary=boundary)

        copied_peps = peps.copy()

        assert copied_peps.boundary == boundary
        assert copied_peps.bond_dim == [2, 3]

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
