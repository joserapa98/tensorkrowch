"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tk

import copy

import time
import opt_einsum
import dis


# TODO: nuevos tests
# TODO: shape y dim iguales o no? Veremos para pilas
# TODO: si pasamos lista de edges al hijo, hacemos siempre reattach? Y en copy, permute?
# TODO: no privado, pero solo desde stack (y en general operaciones) es como se optimiza
#  y se lleva registro de hijos y demás


class TestAxis:
    
    def test_same_name(self):
        node = tk.Node(shape=(3, 3),
                       axes_names=('my_axis', 'my_axis'),
                       name='my_node')
        assert node.axes_names == ['my_axis_0', 'my_axis_1']
    
    def test_same_name_empty(self):
        node = tk.Node(shape=(3, 3),
                       axes_names=('', ''),
                       name='my_node')
        assert node.axes_names == ['_0', '_1']
        
    def test_batch_axis(self):
        node = tk.Node(shape=(3, 3),
                       axes_names=('batch', 'axis'),
                       name='my_node')
        assert node.axes_names == ['batch', 'axis']
        assert node['batch'].is_batch()
        assert not node['axis'].is_batch()
        
    def test_change_axis_name(self):
        node = tk.Node(shape=(3, 3),
                       axes_names=('axis', 'axis1'),
                       name='my_node')
        assert node.axes_names == ['axis', 'axis1']
        
        node.axes[1].name = 'axis2'
        assert node.axes_names == ['axis', 'axis2']
        
        node.axes[1].name = 'axis'
        assert node.axes_names == ['axis_0', 'axis_1']
        
    def test_change_name_batch(self):
        node = tk.Node(shape=(3, 3),
                       axes_names=('batch', 'axis'),
                       name='my_node')
        assert node.axes_names == ['batch', 'axis']
        assert node.axes[0].is_batch()
        assert not node.axes[1].is_batch()
            
        # batch attribute depends on the name,
        # only axis with word `batch` or `stack`
        # in the name are batch axis
        node.axes[0].name = 'new_axis'
        assert node.axes_names == ['new_axis', 'axis']
        assert not node.axes[0].is_batch()
        assert not node.axes[1].is_batch()
        
        node.axes[1].name = 'new_batch'
        assert node.axes_names == ['new_axis', 'new_batch']
        assert not node.axes[0].is_batch()
        assert node.axes[1].is_batch()
        
    
class TestInitNode:
    
    def test_init_node_empty(self):
        node = tk.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       name='my_node')

        assert node.name == 'my_node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is None
        
        assert node.tensor is None
        assert node._tensor_info == {'address': 'my_node',
                                     'node_ref': None,
                                     'full': True,
                                     'stack_idx': None,
                                     'index': None}
        
        net = node.network
        assert net.name == 'tensornetwork'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['my_node']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tk.Edge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_node_data(self):
        node = tk.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       network=tk.TensorNetwork('my_net'),
                       data=True)

        assert node.name == 'node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is None
        
        assert node.tensor is None
        assert node._tensor_info == {'address': 'node',
                                     'node_ref': None,
                                     'full': True,
                                     'stack_idx': None,
                                     'index': None}
        
        net = node.network
        assert net.name == 'my_net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 1
        assert list(net._memory_nodes.keys()) == ['node']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tk.Edge)
        
        assert not node.is_leaf()
        assert node.is_data()
        assert node.successors == dict()
        
    def test_init_node_virtual(self):
        node = tk.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       network=tk.TensorNetwork('my_net'),
                       virtual=True)

        assert node.name == 'node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is None
        
        assert node.tensor is None
        assert node._tensor_info == {'address': 'node',
                                     'node_ref': None,
                                     'full': True,
                                     'stack_idx': None,
                                     'index': None}
        
        net = node.network
        assert net.name == 'my_net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.virtual_nodes) == 1
        assert list(net._memory_nodes.keys()) == ['node']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tk.Edge)
        
        assert not node.is_leaf()
        assert node.is_virtual()
        assert node.successors == dict()
        
    def test_init_node_param_edges(self):
        tensor = torch.randn(2, 5, 2)
        node = tk.Node(axes_names=('left', 'input', 'right'),
                       param_edges=True,
                       tensor=tensor)

        assert node.name == 'node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is torch.float32
        
        assert torch.equal(node.tensor, tensor)
        assert node._tensor_info == {'address': 'node',
                                     'node_ref': None,
                                     'full': True,
                                     'stack_idx': None,
                                     'index': None}
        
        net = node.network
        assert net.name == 'tensornetwork'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['node']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tk.ParamEdge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_node_param_edges_no_axis_names(self):
        tensor = torch.randn(2, 5, 2)
        node = tk.Node(param_edges=True,
                       tensor=tensor)

        assert node.name == 'node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['axis_0', 'axis_1', 'axis_2']
        assert node.rank == 3
        assert node.dtype is torch.float32
        
        assert torch.equal(node.tensor, tensor)
        assert node._tensor_info == {'address': 'node',
                                     'node_ref': None,
                                     'full': True,
                                     'stack_idx': None,
                                     'index': None}
        
        net = node.network
        assert net.name == 'tensornetwork'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['node']
        
        assert node['axis_0'] == node.edges[0]
        assert node['axis_0'] == node[0]
        assert isinstance(node.edges[0], tk.ParamEdge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_node_errors(self):
        # A Node cannot be data and virtual at the same time
        with pytest.raises(ValueError):
            node = tk.Node(shape=(2, 5, 2),
                           data=True,
                           virtual=True)
            
        with pytest.raises(ValueError):
            node = tk.Node(shape=(2, 5, 3),
                           tensor=torch.randn(2, 5, 2))
            
            
class TestInitParamNode:
    
    def test_init_paramnode_empty(self):
        node = tk.ParamNode(shape=(2, 5, 2),
                            axes_names=('left', 'input', 'right'),
                            name='my_node')

        assert node.name == 'my_node'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is None
        
        assert node.tensor is None
        assert node._tensor_info == {'address': 'my_node',
                                     'node_ref': None,
                                     'full': True,
                                     'stack_idx': None,
                                     'index': None}
        
        net = node.network
        assert net.name == 'tensornetwork'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['my_node']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tk.Edge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_paramnode_data(self):
        with pytest.raises(ValueError):
            node = tk.ParamNode(shape=(2, 5, 2),
                                axes_names=('left', 'input', 'right'),
                                network=tk.TensorNetwork('my_net'),
                                data=True)
            
    def test_init_paramnode_virtual(self):
        with pytest.raises(ValueError):
            node = tk.ParamNode(shape=(2, 5, 2),
                                axes_names=('left', 'input', 'right'),
                                network=tk.TensorNetwork('my_net'),
                                virtual=True)
        
    def test_init_paramnode_param_edges(self):
        tensor = torch.randn(2, 5, 2)
        node = tk.ParamNode(axes_names=('left', 'input', 'right'),
                            param_edges=True,
                            tensor=tensor)

        assert node.name == 'paramnode'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is torch.float32
        
        assert torch.equal(node.tensor, nn.Parameter(tensor))
        assert node._tensor_info == {'address': 'paramnode',
                                     'node_ref': None,
                                     'full': True,
                                     'stack_idx': None,
                                     'index': None}
        
        net = node.network
        assert net.name == 'tensornetwork'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['paramnode']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tk.ParamEdge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_paramnode_param_edges_no_axis_names(self):
        tensor = torch.randn(2, 5, 2)
        node = tk.ParamNode(param_edges=True,
                            tensor=tensor)

        assert node.name == 'paramnode'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['axis_0', 'axis_1', 'axis_2']
        assert node.rank == 3
        assert node.dtype is torch.float32
        
        assert torch.equal(node.tensor, nn.Parameter(tensor))
        assert node._tensor_info == {'address': 'paramnode',
                                     'node_ref': None,
                                     'full': True,
                                     'stack_idx': None,
                                     'index': None}
        
        net = node.network
        assert net.name == 'tensornetwork'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['paramnode']
        
        assert node['axis_0'] == node.edges[0]
        assert node['axis_0'] == node[0]
        assert isinstance(node.edges[0], tk.ParamEdge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_paramnode_errors(self):
        # Parametric Nodes cannot be data nodes
        with pytest.raises(ValueError):
            node = tk.ParamNode(shape=(2, 5, 2),
                                data=True)
            
        # Parametric Nodes cannot be virtual nodes
        with pytest.raises(ValueError):
            node = tk.ParamNode(shape=(2, 5, 2),
                                virtual=True)
            
        # Parametric Nodes are always leaf nodes
        with pytest.raises(ValueError):
            node = tk.ParamNode(shape=(2, 5, 2),
                                leaf=False)
            
        with pytest.raises(ValueError):
            node = tk.ParamNode(shape=(2, 5, 3),
                                tensor=torch.randn(2, 5, 2))


class TestNodeName:
    
    @pytest.fixture
    def setup(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        network=net)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        network=net)
        
        node1['right'] ^ node2['left']
        net.set_data_nodes([node1['input'], node2['input']], ['batch'])
        
        return net
    
    def test_change_node_name(self, setup):
        net = setup
        
        node1 = net['node1']
        node2 = net['node2']
        assert node1.name == 'node1'
        assert node2.name == 'node2'
        
        node1.name = 'new_node'
        assert node1.name == 'new_node'
        assert node2.name == 'node2'
        
        node1.name = 'node2'
        assert node1.name == 'node2_1'
        assert node2.name == 'node2_0'
        
        node1.name = 'node1'
        assert node1.name == 'node1'
        assert node2.name == 'node2'
        
        node2.name = 'node1'
        assert node1.name == 'node1_0'
        assert node2.name == 'node1_1'
        
        node1.name = 'node'
        assert node1.name == 'node'
        assert node2.name == 'node1'
        
    def test_change_name_to_data(self, setup):
        net = setup
        
        node1 = net['node1']
        node2 = net['node2']
        assert node1.name == 'node1'
        assert node2.name == 'node2'
        
        node1.name = 'data'
        assert node1.name == 'data_2'
        assert node2.name == 'node2'
        
        # What determines whether a node is a data node or
        # not is the attribute `data`, not the name
        data_node = net['data_0']
        assert not node1.is_data()
        assert data_node.is_data()
        
    def test_change_data_name(self, setup):
        net = setup
        
        data_node = net['data_0']
        assert data_node.name == 'data_0'
        
        data_node.name = 'node'
        assert data_node.name == 'node'
        assert data_node.is_data()


class TestSetTensor:
    
    @pytest.fixture
    def setup(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1')
        
        tensor = torch.randn(2, 5, 2)
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=tensor)
        return node1, node2, tensor
    
    def test_set_tensor(self, setup):
        node1, node2, tensor = setup
        
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        node1.tensor = tensor
        assert torch.equal(node1.tensor, node2.tensor)
        assert node1.shape == (2, 5, 2)
        
    def test_change_tensor(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None
        
        # Changing tensor changes node1's and node2's tensor
        node1.tensor = tensor
        tensor[0, 0, 0] = 1000
        assert node1.tensor[0, 0, 0] == 1000
        assert node2.tensor[0, 0, 0] == 1000
        
    def test_unset_tensor(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None
        
        # Using unset_tensor method
        node1.tensor = tensor
        assert torch.equal(node1.tensor, tensor)
        assert node1.shape == (2, 5, 2)
        
        node1.unset_tensor()
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        # Using tensor.setter
        node1.tensor = tensor
        assert torch.equal(node1.tensor, tensor)
        assert node1.shape == (2, 5, 2)
        
        node1.tensor = None
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        
        # Using set_tensor method
        node1.tensor = tensor
        assert torch.equal(node1.tensor, tensor)
        assert node1.shape == (2, 5, 2)
        
        node1.set_tensor(None)
        assert torch.equal(node1.tensor, torch.zeros(node1.shape))
        assert node1.shape == (2, 5, 2)
        
    def test_set_diff_shape(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None
        assert node1.shape == (2, 5, 2)
        assert node1['left'].size() == 2
        assert node1['input'].size() == 5
        assert node1['right'].size() == 2
        
        # If edges are non-dangling, we can set a
        # tensor with different size in those axes
        diff_tensor = torch.randn(2, 5, 5)
        node1.tensor = diff_tensor
        assert node1.shape == (2, 5, 5)
        assert node1['left'].size() == 2
        assert node1['input'].size() == 5
        assert node1['right'].size() == 5
        
    def test_set_init_method(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None
        
        # Initialize tensor of node1
        node1.set_tensor(init_method='randn', mean=1., std=2.)
        assert node1.tensor is not None

        # Set node1's tensor as node2's tensor
        node2.tensor = node1.tensor
        assert torch.equal(node1.tensor, node2.tensor)

        # Changing node1's tensor changes node2's tensor
        node1.tensor[0, 0, 0] = 1000
        assert node2.tensor[0, 0, 0] == 1000
        
    def test_set_parametric(self, setup):
        node1, node2, tensor = setup
        assert node1.tensor is None
        
        # Create parametric node
        node3 = tk.ParamNode(axes_names=('left', 'input', 'right'),
                             tensor=tensor)
        assert isinstance(node3.tensor, nn.Parameter)
        assert torch.equal(node3.tensor.data, tensor)

        # Creating parameter from tensor does not affect tensor's grad
        param = nn.Parameter(tensor)
        param.mean().backward()
        assert node3.grad is None

        # Set nn.Parameter as node3's tensor
        node3.tensor = param
        assert node3.grad is not None
        assert torch.equal(node3.grad, node3.tensor.grad)
        
        node3.tensor = None
        assert node3.tensor is None
        assert node3.grad is None
        
        
class TestConnect:
    
    def test_connect_edges(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')
        
        assert node1.name == 'node'
        assert node2.name == 'node'
        assert node1.network != node2.network
        
        assert isinstance(node1[2], tk.Edge)
        assert isinstance(node2[0], tk.Edge)
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tk.Edge)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
    def test_connect_paramedges(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        param_edges=True)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        param_edges=True)
        
        assert node1.name == 'node'
        assert node2.name == 'node'
        assert node1.network != node2.network
        
        assert isinstance(node1[2], tk.ParamEdge)
        assert isinstance(node2[0], tk.ParamEdge)
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tk.ParamEdge)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
    def test_connect_edge_paramedge(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        param_edges=True)
        
        assert node1.name == 'node'
        assert node2.name == 'node'
        assert node1.network != node2.network
        
        assert isinstance(node1[2], tk.Edge)
        assert isinstance(node2[0], tk.ParamEdge)
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tk.ParamEdge)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
    def test_connect_paramedge_edge(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        param_edges=True)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')
        
        assert node1.name == 'node'
        assert node2.name == 'node'
        assert node1.network != node2.network
        
        assert isinstance(node1[2], tk.ParamEdge)
        assert isinstance(node2[0], tk.Edge)
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tk.ParamEdge)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
    def test_connect_same_network(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net)
        assert node1.name == 'node'
        
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
        assert isinstance(node1[2], tk.Edge)
        assert isinstance(node2[0], tk.Edge)
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tk.Edge)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
    def test_connect_different_sizes(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        param_edges=True)
        
        node2[0].change_size(4)
        assert node2[0].size() == 4
        assert node2[0].dim() == 2
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tk.ParamEdge)
        assert new_edge.size() == 2
        assert new_edge.dim() == 2

        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        param_edges=True)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        param_edges=True)
        node1[2].change_size(3)
        node2[0].change_size(4)
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tk.ParamEdge)
        assert new_edge.size() == 3
        assert new_edge.dim() == 2
        
    def test_connect_with_result(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        init_method='randn')
        node1[2] ^ node2[0]
        node3 = node1 @ node2

        node4 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node4',
                        init_method='randn')
        
        # When we connect an edge from a node that is the result of an operation,
        # be aware that its edges are inherited from leaf nodes, so the connection
        # takes place between the leaf nodes.
        edge = node3[3]
        node3[3] ^ node4[0]
        assert node3[3] == edge
        assert node2[2] == node4[0]
        with pytest.raises(ValueError):
            # Raises error because nodes are not connected,
            # node3 still has the disconnected edge
            node3 @ node4

        # You can do this to overcome this issue, but is not recommended
        # practice. You should better make all the connections before starting
        # with operations
        node4[0] | node4[0]
        node3._reattach_edges(override=True)
        edge = node3[3]
        node3[3] ^ node4[0]
        assert node3[3] != edge
        assert node3[3] == node4[0]
        
    def test_disconnect(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        init_method='randn')
        
        assert node1['right'].is_dangling()
        assert node2['left'].is_dangling()
        assert node1['right'] != node2['left']
        
        edge = node1['right'] ^ node2['left']
        assert not node1['right'].is_dangling()
        assert not node2['left'].is_dangling()
        assert node1['right'] == node2['left']
        
        edge | edge
        assert node1['right'].is_dangling()
        assert node2['left'].is_dangling()
        assert node1['right'] != node2['left']
        
    def test_connect_disconnect_stackedge(self):
        net = tk.TensorNetwork()
        left_nodes = []
        right_nodes = []
        for _ in range(10):
            node = tk.Node(shape=(5,),
                           axes_names=('left',),
                           init_method='randn',
                           network=net)
            left_nodes.append(node)
            
            node = tk.Node(shape=(5,),
                           axes_names=('right',),
                           init_method='randn',
                           network=net)
            right_nodes.append(node)
            
        for i in range(10):
            left_nodes[i][0] ^ right_nodes[i][0]
            
        left_stack = tk.stack(left_nodes)
        right_stack = tk.stack(right_nodes)
        
        assert isinstance(left_stack[1], tk.StackEdge)
        assert left_stack[1].is_dangling()
        assert isinstance(right_stack[1], tk.StackEdge)
        assert right_stack[1].is_dangling()
        
        left_stack[1] ^ right_stack[1]
        assert left_stack[1] == right_stack[1]
        assert isinstance(left_stack[1], tk.StackEdge)
        
        left_stack[1] | left_stack[1]
        assert left_stack[1] != right_stack[1]
        assert isinstance(left_stack[1], tk.StackEdge)
        assert left_stack[1].is_dangling()
        assert isinstance(right_stack[1], tk.StackEdge)
        assert right_stack[1].is_dangling()
        
    def test_connect_disconnected_stacks(self):
        net = tk.TensorNetwork()
        left_nodes = []
        right_nodes = []
        for _ in range(10):
            node = tk.Node(shape=(5,),
                           axes_names=('left',),
                           init_method='randn',
                           network=net)
            left_nodes.append(node)
            
            node = tk.Node(shape=(5,),
                           axes_names=('right',),
                           init_method='randn',
                           network=net)
            right_nodes.append(node)
            
        left_stack = tk.stack(left_nodes)
        right_stack = tk.stack(right_nodes)
        
        with pytest.raises(ValueError):
            left_stack[1] ^ right_stack[1]
    
 
class TestSVD:
    
    @pytest.fixture
    def setup(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(3, 5, 3),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        network=net,
                        init_method='randn')
        node2 = tk.Node(shape=(3, 5, 3),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        network=net,
                        init_method='randn')
        
        edge = node1['right'] ^ node2['left']
        return edge
    
    def test_svd_edge(self, setup):
        edge = setup
        assert isinstance(edge, tk.Edge)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        edge.svd(rank=2)
        assert edge.node1.shape == (3, 5, 2)
        assert edge.node2.shape == (2, 5, 3)
        assert edge.size() == 2
        assert edge.dim() == 2 
        
        edge.svd(cum_percentage=0.9)
        
    def test_svd_paramedge(self, setup):
        edge = setup
        paramedge = edge.parameterize()
        
        assert isinstance(paramedge, tk.ParamEdge)
        assert paramedge.size() == 3
        assert paramedge.dim() == 3
        
        paramedge.svd(rank=2)
        assert paramedge.node1.shape == (3, 5, 2)
        assert paramedge.node2.shape == (2, 5, 3)
        assert paramedge.size() == 2
        assert paramedge.dim() == 2 
        
        paramedge.svd(cum_percentage=0.9)


class TestParameterize:
    
    def test_parameterize_node(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        node1['right'] ^ node2['left']
        
        net = node1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        assert node1['right'] == node2['left']
        
        paramnode1 = node1.parameterize()
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        
        # Now `node2`` and `paramnode1`` share same edges
        assert paramnode1['right'] == node2['left']
        
        # `node1`` still exists and has edges pointing to `node2``,
        # but `node2`` cannot "see" it
        assert node1['right'] != node2['left']
        assert node1['right']._nodes[node1.is_node1('right')] == node2
        del node1
        
        assert isinstance(paramnode1, tk.ParamNode)
        assert paramnode1['left'].node1 == paramnode1
        assert isinstance(paramnode1['left'], tk.Edge)
        
    def test_deparameterize_paramnode(self):
        paramnode1 = tk.ParamNode(axes_names=('left', 'input', 'right'),
                                  name='paramnode1',
                                  tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        paramnode1['right'] ^ node2['left']
        
        net = paramnode1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        assert paramnode1['right'] == node2['left']
        
        node1 = paramnode1.parameterize(False)
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        
        # Now `node2`` and `node1`` share same edges
        assert node1['right'] == node2['left']
        
        # `paramnode1`` still exists and has edges pointing to `node2``,
        # but `node2`` cannot "see" it
        assert paramnode1['right'] != node2['left']
        assert paramnode1['right']._nodes[paramnode1.is_node1('right')] == node2
        del paramnode1
        
        assert isinstance(node1, tk.Node)
        assert node1['left'].node1 == node1
        assert isinstance(node1['left'], tk.Edge)
        
    def test_parameterize_dangling_edge(self):
        node = tk.Node(axes_names=('left', 'input', 'right'),
                       tensor=torch.randn(3, 5, 2))
        net = node.network
        
        prev_edge = node['left']
        assert prev_edge in net.edges

        node['left'].parameterize(set_param=True, size=4)
        assert isinstance(node['left'], tk.ParamEdge)
        assert node.shape == (4, 5, 2)
        assert node.dim() == (3, 5, 2)

        assert prev_edge not in net.edges
        assert node['left'] in net.edges

        node['left'].parameterize(set_param=False)
        assert isinstance(node['left'], tk.Edge)
        assert node.shape == (3, 5, 2)
        assert node.dim() == (3, 5, 2)

        node['left'].parameterize(set_param=True, size=2)
        assert node.shape == (2, 5, 2)
        assert node.dim() == (2, 5, 2)

        node['left'].parameterize(set_param=False)
        assert node.shape == (2, 5, 2)
        assert node.dim() == (2, 5, 2)
        
    def test_parameterize_dangling_edge2(self):
        node = tk.Node(shape=(3, 5, 2),
                       axes_names=('left', 'input', 'right'))
        net = node.network
        
        prev_edge = node['left']
        assert prev_edge in net.edges

        node['left'].parameterize(set_param=True, size=4)
        assert isinstance(node['left'], tk.ParamEdge)
        assert node.shape == (4, 5, 2)
        assert node.dim() == (3, 5, 2)

        assert prev_edge not in net.edges
        assert node['left'] in net.edges

        node['left'].parameterize(set_param=False)
        assert isinstance(node['left'], tk.Edge)
        assert node.shape == (3, 5, 2)
        assert node.dim() == (3, 5, 2)

        node['left'].parameterize(set_param=True, size=2)
        assert node.shape == (2, 5, 2)
        assert node.dim() == (2, 5, 2)

        node['left'].parameterize(set_param=False)
        assert node.shape == (2, 5, 2)
        assert node.dim() == (2, 5, 2)
        
    def test_parameterize_connected_edge(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 3))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(3, 5, 2))
        node1['right'] ^ node2['left']
        
        prev_edge = node2['left']
        assert prev_edge in node2.edges

        node2['left'].parameterize(set_param=True, size=4)
        assert isinstance(node2['left'], tk.ParamEdge)
        assert node2.shape == (4, 5, 2)
        assert node2.dim() == (3, 5, 2)

        assert prev_edge not in node2.edges

        node2['left'].parameterize(set_param=False)
        assert isinstance(node2['left'], tk.Edge)
        assert node2.shape == (3, 5, 2)
        assert node2.dim() == (3, 5, 2)

        node2['left'].parameterize(set_param=True, size=2)
        assert node2.shape == (2, 5, 2)
        assert node2.dim() == (2, 5, 2)

        node2['left'].parameterize(set_param=False)
        assert node2.shape == (2, 5, 2)
        assert node2.dim() == (2, 5, 2) 
        
    def test_change_size_dim_dangling(self):
        node = tk.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       name='node',
                       param_edges=True,
                       init_method='randn')
        
        for i, edge in enumerate(node.edges):
            assert isinstance(edge, tk.ParamEdge)
            assert edge.dim() == node.shape[i]
        
        param_edge = node[0]
        param_edge.change_size(size=4)
        assert param_edge.size() == 4
        assert param_edge.node1.size() == (4, 5, 2)
        assert param_edge.dim() == 2
        assert param_edge.node1.dim() == (2, 5, 2)

        param_edge.change_dim(dim=3)
        assert param_edge.size() == 4
        assert param_edge.node1.size() == (4, 5, 2)
        assert param_edge.dim() == 3
        assert param_edge.node1.dim() == (3, 5, 2)

        param_edge.change_size(size=2)
        assert param_edge.size() == 2
        assert param_edge.node1.size() == (2, 5, 2)
        assert param_edge.dim() == 2
        assert param_edge.node1.dim() == (2, 5, 2)
        
    def test_change_size_dim_connected(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        param_edges=True,
                        init_method='randn')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        param_edges=True,
                        init_method='randn')
        node1['right'] ^ node2['left']
        
        param_edge = node2['left']
        param_edge.change_size(size=4)
        assert param_edge.size() == 4
        assert param_edge.node1.size() == (2, 5, 4)
        assert param_edge.node2.size() == (4, 5, 2)
        assert param_edge.dim() == 2
        assert param_edge.node1.dim() == (2, 5, 2)
        assert param_edge.node2.dim() == (2, 5, 2)

        param_edge.change_dim(dim=3)
        assert param_edge.size() == 4
        assert param_edge.node1.size() == (2, 5, 4)
        assert param_edge.node2.size() == (4, 5, 2)
        assert param_edge.dim() == 3
        assert param_edge.node1.dim() == (2, 5, 3)
        assert param_edge.node2.dim() == (3, 5, 2)

        param_edge.change_size(size=2)
        assert param_edge.size() == 2
        assert param_edge.node1.size() == (2, 5, 2)
        assert param_edge.node2.size() == (2, 5, 2)
        assert param_edge.dim() == 2
        assert param_edge.node1.dim() == (2, 5, 2)
        assert param_edge.node2.dim() == (2, 5, 2)


class TestCopy:
    
    def test_copy_edge(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        edge = node1['right'] ^ node2['left']
        
        copy_edge = edge.copy()
        assert isinstance(edge, tk.Edge)
        assert isinstance(copy_edge, tk.Edge)
        assert copy_edge._nodes == edge._nodes
        assert copy_edge._axes == edge._axes
        assert copy_edge != edge
        
    def test_copy_paramedge(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2),
                        param_edges=True)
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        paramedge = node1['right'] ^ node2['left']
        
        copy_paramedge = paramedge.copy()
        assert isinstance(paramedge, tk.ParamEdge)
        assert isinstance(copy_paramedge, tk.ParamEdge)
        assert copy_paramedge._nodes == paramedge._nodes
        assert copy_paramedge._axes == paramedge._axes
        assert copy_paramedge != paramedge
        
    def test_copy_node_empty(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1')
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2')
        node1['right'] ^ node2['left']
        
        net = node1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        
        copy = node1.copy()
        assert copy.tensor == node1.tensor
        assert len(net.nodes) == 3
        assert len(net.edges) == 6
        
        for i in range(copy.rank):
            edge = node1[i]
            copy_edge = copy[i]
            assert copy_edge._nodes[1 - copy.is_node1(i)] == copy
            assert edge._nodes[1 - copy.is_node1(i)] == node1
            assert copy_edge._nodes[copy.is_node1(i)] == \
                edge._nodes[copy.is_node1(i)]
            
        assert node2['left'].node1 == node1
        
    def test_copy_node(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        node1['right'] ^ node2['left']
        
        net = node1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        
        copy = node1.copy()
        assert torch.equal(copy.tensor, node1.tensor)
        assert len(net.nodes) == 3
        assert len(net.edges) == 6
        
        for i in range(copy.rank):
            edge = node1[i]
            copy_edge = copy[i]
            assert copy_edge._nodes[1 - copy.is_node1(i)] == copy
            assert edge._nodes[1 - copy.is_node1(i)] == node1
            assert copy_edge._nodes[copy.is_node1(i)] == \
                edge._nodes[copy.is_node1(i)]
            
        assert node2['left'].node1 == node1
                
    def test_copy_paramnode_empty(self):
        node1 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node1',
                             param_edges=True)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2')
        node1['right'] ^ node2['left']
        
        net = node1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        
        copy = node1.copy()
        assert copy.tensor == node1.tensor
        assert len(net.nodes) == 3
        assert len(net.edges) == 6
        
        for i in range(copy.rank):
            edge = node1[i]
            copy_edge = copy[i]
            assert copy_edge._nodes[1 - copy.is_node1(i)] == copy
            assert edge._nodes[1 - copy.is_node1(i)] == node1
            assert copy_edge._nodes[copy.is_node1(i)] == \
                edge._nodes[copy.is_node1(i)]
            
        assert node2['left'].node1 == node1
        
    def test_copy_paramnode(self):
        node1 = tk.ParamNode(axes_names=('left', 'input', 'right'),
                             name='node1',
                             tensor=torch.randn(2, 5, 2),
                             param_edges=True)
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        node1['right'] ^ node2['left']
        
        net = node1.network
        assert len(net.nodes) == 2
        assert len(net.edges) == 4
        
        copy = node1.copy()
        assert torch.equal(copy.tensor, node1.tensor)
        assert len(net.nodes) == 3
        assert len(net.edges) == 6
        
        for i in range(copy.rank):
            edge = node1[i]
            copy_edge = copy[i]
            assert copy_edge._nodes[1 - copy.is_node1(i)] == copy
            assert edge._nodes[1 - copy.is_node1(i)] == node1
            assert copy_edge._nodes[copy.is_node1(i)] == \
                edge._nodes[copy.is_node1(i)]
            
        assert node2['left'].node1 == node1
            

class TestTensorNetwork:
    
    def test_change_network(self):
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2))
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2))
        
        net1 = node1.network
        net2 = node2.network
        assert net1 != net2
        assert len(net1.nodes) == 1
        assert len(net1.edges) == 3
        assert len(net2.nodes) == 1
        assert len(net2.edges) == 3
        
        node2.network = net1
        assert node1.network == node2.network
        assert len(net1.nodes) == 2
        assert len(net1.edges) == 6
        assert len(net2.nodes) == 0
        assert len(net2.edges) == 0
        
    def test_move_to_network(self):
        net1 = tk.TensorNetwork(name='net1')
        node1 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2),
                        network=net1)
        node2 = tk.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(2, 5, 2),
                        network=net1)
        
        net2 = tk.TensorNetwork(name='net2')
        node2.network = net2
        assert node1.network == net1
        assert node2.network == net2
        
        node2.network = net1
        assert node1.network == net1
        assert node2.network == net1
        
        node1['right'] ^ node2['left']
        node2.network = net2
        assert node1.network == net2
        assert node2.network == net2
        assert len(net2.nodes) == 2
        assert len(net2.edges) == 4
        assert len(net1.nodes) == 0
        assert len(net1.edges) == 0
        
    def test_add_remove(self):
        net = tk.TensorNetwork()
        for _ in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net)
        assert list(net.nodes.keys()) == [f'node_{i}' for i in range(4)]

        new_node = tk.Node(shape=(2, 5, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node')
        new_node.network = net
        assert new_node.name == 'node_4'

        assert len(net.edges) == 15
        for i in range(4):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.edges) == 7

        net._remove_node(new_node)
        # `new_node` still exists, but not in the network
        assert new_node.name not in net.nodes
        assert net['node_3']['right'].node2 == new_node

        new_node.network = net
        net.delete_node(new_node)
        # When deleting, we remove node AND disconnect,
        # so that we can now delete the object
        assert new_node.name not in net.nodes
        assert net['node_3']['right'].node2 is None
        assert new_node['left'].node2 is None
        del new_node
        
    def test_change_name(self):
        net = tk.TensorNetwork()
        for _ in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net)
        node1 = net['node_0']
        node2 = net['node_1']
        
        node1.name = 'node_1'
        assert list(net.nodes.keys()) == [f'node_{i}' for i in range(4)]
        assert node1.name == 'node_0'
        
        node1.name = 'my_node'
        assert list(net.nodes.keys()) == [f'node_{i}' for i in range(3)] \
                                         + ['my_node']
        assert node1.name == 'my_node'
        
        node2.name = 'my_node'
        assert list(net.nodes.keys()) == ['node_0', 'node_1',
                                          'my_node_0', 'my_node_1']
        assert node1.name == 'my_node_0'
        assert node2.name == 'my_node_1'
            
    def test_consecutive_contractions(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='randn')

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.edges) == 6

        node = net['node_0']
        for i in range(1, 4):
            node @= net[f'node_{i}']
        assert len(node.edges) == 6
        assert node.shape == (2, 5, 5, 5, 5, 2)
        assert len(net.nodes) == 7
        assert len(net.edges) == 6

        new_node = tk.Node(shape=(2, 5, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node',
                           init_method='randn')
        new_node.network = net
        assert new_node.name == 'node_4'
        net['node_3'][2] ^ new_node[0]
        with pytest.raises(ValueError):
            node @= new_node
            
    def test_submodules_empty(self):
        net = tk.TensorNetwork(name='net')
        for i in range(2):
            _ = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             network=net,
                             param_edges=True)
        for i in range(2):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        param_edges=True)

        # Only ParamEdges are submodules, ParamNodes are just
        # parameters of the network
        submodules = [None for _ in net.children()]
        assert len(submodules) == 12
        assert len(net._parameters) == 0

        net['paramnode_0']['right'] ^ net['paramnode_1']['left']
        net['paramnode_1']['right'] ^ net['node_0']['left']
        net['node_0']['right'] ^ net['node_1']['left']
        submodules = [None for _ in net.children()]
        assert len(submodules) == 9
        assert len(net._parameters) == 0
        
    def test_submodules(self):
        net = tk.TensorNetwork(name='net')
        for i in range(2):
            _ = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             network=net,
                             param_edges=True,
                             init_method='randn')
        for i in range(2):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        param_edges=True,
                        init_method='randn')

        # Only ParamEdges are submodules, ParamNodes are just
        # parameters of the network
        submodules = [None for _ in net.children()]
        assert len(submodules) == 12
        assert len(net._parameters) == 2

        net['paramnode_0']['right'] ^ net['paramnode_1']['left']
        net['paramnode_1']['right'] ^ net['node_0']['left']
        net['node_0']['right'] ^ net['node_1']['left']
        submodules = [None for _ in net.children()]
        assert len(submodules) == 9
        assert len(net._parameters) == 2
        
    def test_parameterize_tn_empty(self):
        net = tk.TensorNetwork(name='net')
        for i in range(2):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net)

        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0

        param_net = net.parameterize()
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0
        submodules = [None for _ in param_net.children()]
        assert len(submodules) == 6
        assert len(param_net._parameters) == 0

        param_net = net.parameterize(override=True)
        assert param_net == net
        submodules = [None for _ in net.children()]
        assert len(submodules) == 6
        assert len(net._parameters) == 0

        net.parameterize(set_param=False, override=True)
        assert param_net == net
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0
        
    def test_parameterize_tn(self):
        net = tk.TensorNetwork(name='net')
        for i in range(2):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        init_method='randn')

        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0

        param_net = net.parameterize()
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0
        submodules = [None for _ in param_net.children()]
        assert len(submodules) == 6
        assert len(param_net._parameters) == 2

        param_net = net.parameterize(override=True)
        assert param_net == net
        submodules = [None for _ in net.children()]
        assert len(submodules) == 6
        assert len(net._parameters) == 2

        net.parameterize(set_param=False, override=True)
        assert param_net == net
        submodules = [None for _ in net.children()]
        assert len(submodules) == 0
        assert len(net._parameters) == 0
        
    def test_set_data_nodes_same_shape(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='ones')
        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']

        assert len(net.data_nodes) == 0
        input_edges = []
        for i in range(4):
            input_edges.append(net[f'node_{i}']['input'])
        net.set_data_nodes(input_edges, ['batch'])
        
        # 4 leaf nodes, 4 data nodes, and the
        # stack_data_memory (+2 virtual nodes)
        assert len(net.nodes) == 11
        assert len(net.leaf_nodes) == 4
        assert len(net.data_nodes) == 4
        assert len(net.virtual_nodes) == 3

        input_edges = []
        for i in range(3):
            input_edges.append(net[f'node_{i}']['input'])
        with pytest.raises(ValueError):
            net.set_data_nodes(input_edges, ['batch'])

        net.unset_data_nodes()
        assert len(net.nodes) == 4
        assert len(net.data_nodes) == 0

        input_edges = []
        for i in range(2):
            input_edges.append(net[f'node_{i}']['input'])
        net.set_data_nodes(input_edges, ['batch'])
        assert len(net.nodes) == 9
        assert len(net.leaf_nodes) == 4
        assert len(net.data_nodes) == 2
        assert len(net.virtual_nodes) == 3

    def test_add_data_same_shape(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='ones')
        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']

        input_edges = []
        for i in range(2):
            input_edges.append(net[f'node_{i}']['input'])
        net.set_data_nodes(input_edges, ['batch'])

        # data shape = n_features x batch_dim x feature_dim
        data = torch.randn(2, 10, 5)
        net._add_data(data)
        assert torch.equal(net.data_nodes['data_0'].tensor, data[0, :, :])
        assert torch.equal(net.data_nodes['data_1'].tensor, data[1, :, :])
        assert torch.equal(net.nodes['stack_data_memory'].tensor, data)

        # This causes no error, because the data tensor will be cropped to fit
        # the shape of the stack_data_memory node. It gives a warning
        data = torch.randn(3, 10, 5)
        net._add_data(data)
        
        # This does not give warning, batch size can be changed as we wish
        data = torch.randn(2, 100, 5)
        net._add_data(data)

        # Add data with no data nodes raises error
        net.unset_data_nodes()
        with pytest.raises(ValueError):
            net._add_data(data)
            
    def test_set_data_nodes_diff_shape(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            node = tk.Node(shape=(2, i + 2, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node',
                           network=net,
                           init_method='ones')
            data_node = tk.Node(shape=(i + 2, 1),  # Shape of batch index is irrelevant
                                axes_names=('feature', 'batch'),
                                name='data_node',
                                network=net,
                                init_method='ones',
                                data=True)
            node['input'] ^ data_node['feature']
        
        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']

        assert len(net.nodes) == 8
        assert len(net.leaf_nodes) == 4
        assert len(net.data_nodes) == 4
        assert len(net.virtual_nodes) == 0
        
        net.unset_data_nodes()
        assert len(net.nodes) == 4
        assert len(net.leaf_nodes) == 4
        assert len(net.data_nodes) == 0
        assert len(net.virtual_nodes) == 0
        
    def test_add_data_diff_shape(self):
        net = tk.TensorNetwork(name='net')
        data = []
        for i in range(4):
            node = tk.Node(shape=(2, i + 2, 2),
                           axes_names=('left', 'input', 'right'),
                           name='node',
                           network=net,
                           init_method='ones')
            data_node = tk.Node(shape=(i + 2, 1),  # Shape of batch index is irrelevant
                                axes_names=('feature', 'batch'),
                                name='data',
                                network=net,
                                init_method='ones',
                                data=True)
            node['input'] ^ data_node['feature']
            data.append(torch.randn(i + 2, 100))
        
        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']

        net._add_data(data)
        for i in range(4):
            assert torch.equal(net.data_nodes[f'data_{i}'].tensor, data[i])

        # This does not raise error because indexing data[i] works
        # the same for lists or tensors where the "node" index is the
        # first dimension. Besides, the "feature" dimension has to be
        # greater than any of the "feature" dimensions used in data nodes,
        # since we are cropping
        data = torch.randn(4, 6, 100)
        net._add_data(data)
        
        # If feature dimension is small, it would raise an error
        data = torch.randn(4, 4, 100)
        with pytest.raises(ValueError):
            net._add_data(data)

        # Add data with no data nodes raises error
        net.unset_data_nodes()
        with pytest.raises(ValueError):
            net._add_data(data)
        
    def test_copy_tn(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='ones')
        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']
            
        copy_net = net.copy()
        assert copy_net != net
        assert copy_net.nodes != net.nodes
        assert copy_net.edges != net.edges
        
        assert len(net.nodes) == 4
        assert len(net.edges) == 6
        
        assert len(copy_net.nodes) == 4
        assert len(copy_net.edges) == 6
        
    def test_delete_nodes(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='randn')

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.nodes) == 4
        assert len(net.edges) == 6
        
        net.delete_node(net['node_3'])
        assert len(net.nodes) == 3
        assert len(net.edges) == 5
        
        node = net['node_0']
        for i in range(1, 3):
            node @= net[f'node_{i}']
        assert len(net.nodes) == 5
        assert len(net.leaf_nodes) == 3
        assert len(net.non_leaf_nodes) == 2
        assert len(net.edges) == 5
        
        net.delete_node(node)
        assert len(net.nodes) == 4
        assert len(net.leaf_nodes) == 3
        assert len(net.non_leaf_nodes) == 1
        assert len(net.edges) == 5
        
    def test_delete_non_leaf(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='randn')

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.nodes) == 4
        assert len(net.edges) == 6
        
        stack_node1 = tk.stack([net['node_0'], net['node_2']])
        stack_node2 = tk.stack([net['node_1'], net['node_3']])
        stack_node1['right'] ^ stack_node2['left']
        assert len(net.nodes) == 6
        assert len(net.leaf_nodes) == 4
        assert len(net.non_leaf_nodes) == 2
        assert len(net.edges) == 6
        
        contract_node = stack_node1 @ stack_node2
        assert len(net.nodes) == 7
        assert len(net.leaf_nodes) == 4
        assert len(net.non_leaf_nodes) == 3
        assert len(net.edges) == 6
        
        net.delete_non_leaf()
        assert len(net.nodes) == 4
        assert len(net.leaf_nodes) == 4
        assert len(net.non_leaf_nodes) == 0
        assert len(net.edges) == 6
        
    def test_automemory(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='randn')

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.nodes) == 4
        assert len(net.edges) == 6
        for node in net.nodes.values():
            assert node._tensor_info['address'] is not None
        
        assert net.automemory == False
        assert net.unbind_mode == True
        
        stack = tk.stack(list(net.nodes.values()))
        # All nodes still have their own memory
        for node in net.leaf_nodes.values():
            assert node._tensor_info['address'] is not None
        for node in net.non_leaf_nodes.values():
            assert node._tensor_info['address'] is not None
            
        net.automemory = True
        assert net.automemory == True
        assert net.unbind_mode == True
        
        stack = tk.stack(list(net.nodes.values()))
        # Now leaf nodes have their emory stored in the stack
        for node in net.leaf_nodes.values():
            assert node._tensor_info['address'] is None
        for node in net.non_leaf_nodes.values():
            assert node._tensor_info['address'] is not None
            
    def test_unbind_mode(self):
        net = tk.TensorNetwork(name='net')
        for i in range(4):
            _ = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net,
                        init_method='randn')

        for i in range(3):
            net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
        assert len(net.nodes) == 4
        assert len(net.edges) == 6
        for node in net.nodes.values():
            assert node._tensor_info['address'] is not None
        
        assert net.automemory == False
        assert net.unbind_mode == True
        
        stack1 = tk.stack(list(net.nodes.values()))
        unbinded = tk.unbind(stack1)
        stack2 = tk.stack(unbinded)
        # All nodes still have their own memory
        for node in net.leaf_nodes.values():
            assert node._tensor_info['address'] is not None
            assert node._tensor_info['node_ref'] is None
        for node in net.non_leaf_nodes.values():
            if node.name.startswith('unbind'):
                assert node._tensor_info['node_ref'] is not None
            else:
                assert node._tensor_info['node_ref'] is None
            assert node._tensor_info['address'] is not None
            
        net.unbind_mode = False
        assert net.automemory == False
        assert net.unbind_mode == False
        
        stack1 = tk.stack(list(net.nodes.values()))
        unbinded = tk.unbind(stack1)
        stack2 = tk.stack(unbinded)
        # Now leaf nodes have their emory stored in the stack
        for node in net.leaf_nodes.values():
            assert node._tensor_info['address'] is not None
            assert node._tensor_info['node_ref'] is None
            
        # Unbinded nodes and `stack2` share memory with `stack1`
        assert stack1._tensor_info['address'] is not None
        for node in unbinded:
            assert node._tensor_info['address'] is None
        assert stack2._tensor_info['address'] is None
        
        
        


# def test_is_updated():
#     # TODO: no lo uso
#     node1 = tk.Node(shape=(2, 3),
#                     axes_names=('left', 'right'),
#                     name='node1',
#                     param_edges=True,
#                     init_method='ones')
#     node2 = tk.Node(shape=(3, 4),
#                     axes_names=('left', 'right'),
#                     name='node2',
#                     param_edges=True,
#                     init_method='ones')
#     new_edge = node1['right'] ^ node2['left']
#     prev_matrix = new_edge.matrix
#     optimizer = torch.optim.SGD(params=new_edge.parameters(), lr=0.1)

#     node3 = node1 @ node2
#     mean = node3.mean()
#     mean.backward()
#     optimizer.step()

#     assert not new_edge.is_updated()  # TODO: a lo mejor no sirve pa na

#     new_matrix = new_edge.matrix
#     assert not torch.equal(prev_matrix, new_matrix)


# TODO: Hacer ParamStackNode, al hacer stack, si todos los nodos son leaf y ParamNode
# TODO: Hacer que la matriz de ParamStackEdge se construya apilando las matrices de los edges que contiene