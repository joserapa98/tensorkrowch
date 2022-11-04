"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tentorch as tn

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
        node = tn.Node(shape=(3, 3),
                       axes_names=('my_axis', 'my_axis'),
                       name='my_node')
        assert node.axes_names == ['my_axis_0', 'my_axis_1']
    
    def test_same_name_empty(self):
        node = tn.Node(shape=(3, 3),
                       axes_names=('', ''),
                       name='my_node')
        assert node.axes_names == ['_0', '_1']
        
    def test_batch_axis(self):
        node = tn.Node(shape=(3, 3),
                       axes_names=('batch', 'axis'),
                       name='my_node')
        assert node.axes_names == ['batch', 'axis']
        assert node['batch'].is_batch()
        assert not node['axis'].is_batch()
        
    def test_change_axis_name(self):
        node = tn.Node(shape=(3, 3),
                       axes_names=('axis', 'axis1'),
                       name='my_node')
        assert node.axes_names == ['axis', 'axis1']
        
        node.axes[1].name = 'axis2'
        assert node.axes_names == ['axis', 'axis2']
        
        node.axes[1].name = 'axis'
        assert node.axes_names == ['axis_0', 'axis_1']
        
    
class TestInitNode:
    
    def test_init_node1(self):
        node = tn.Node(shape=(2, 5, 2),
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
        assert net.name == 'net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['my_node']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tn.Edge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_node2(self):
        node = tn.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       network=tn.TensorNetwork('my_net'),
                       leaf=False,
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
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['node']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tn.Edge)
        
        assert not node.is_leaf()
        assert node.is_data()
        assert node.successors == dict()
        
    def test_init_node3(self):
        tensor = torch.randn(2, 5, 2)
        node = tn.Node(axes_names=('left', 'input', 'right'),
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
        assert net.name == 'net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['node']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tn.ParamEdge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_node4(self):
        tensor = torch.randn(2, 5, 2)
        node = tn.Node(param_edges=True,
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
        assert net.name == 'net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['node']
        
        assert node['axis_0'] == node.edges[0]
        assert node['axis_0'] == node[0]
        assert isinstance(node.edges[0], tn.ParamEdge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_node_errors(self):
        with pytest.raises(ValueError):
            node = tn.Node(shape=(2, 5, 2),
                           data=True)
            
        with pytest.raises(ValueError):
            node = tn.Node(shape=(2, 5, 2),
                           tensor=torch.randn(2, 5, 2))
            
            
class TestInitParamNode:
    
    def test_init_paramnode1(self):
        node = tn.ParamNode(shape=(2, 5, 2),
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
        assert net.name == 'net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['my_node']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tn.Edge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_paramnode2(self):
        node = tn.ParamNode(shape=(2, 5, 2),
                            axes_names=('left', 'input', 'right'),
                            network=tn.TensorNetwork('my_net'),
                            leaf=False,
                            data=True)

        assert node.name == 'paramnode'
        assert node.shape == (2, 5, 2)
        assert node.axes_names == ['left', 'input', 'right']
        assert node.rank == 3
        assert node.dtype is None
        
        assert node.tensor is None
        assert node._tensor_info == {'address': 'paramnode',
                                     'node_ref': None,
                                     'full': True,
                                     'stack_idx': None,
                                     'index': None}
        
        net = node.network
        assert net.name == 'my_net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['paramnode']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tn.Edge)
        
        assert not node.is_leaf()
        assert node.is_data()
        assert node.successors == dict()
        
    def test_init_paramnode3(self):
        tensor = torch.randn(2, 5, 2)
        node = tn.ParamNode(axes_names=('left', 'input', 'right'),
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
        assert net.name == 'net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['paramnode']
        
        assert node['left'] == node.edges[0]
        assert node['left'] == node[0]
        assert isinstance(node.edges[0], tn.ParamEdge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_paramnode4(self):
        tensor = torch.randn(2, 5, 2)
        node = tn.ParamNode(param_edges=True,
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
        assert net.name == 'net'
        assert len(net.nodes) == 1
        assert len(net._memory_nodes) == 1
        assert len(net.data_nodes) == 0
        assert list(net._memory_nodes.keys()) == ['paramnode']
        
        assert node['axis_0'] == node.edges[0]
        assert node['axis_0'] == node[0]
        assert isinstance(node.edges[0], tn.ParamEdge)
        
        assert node.is_leaf()
        assert not node.is_data()
        assert node.successors == dict()
        
    def test_init_paramnode_errors(self):
        with pytest.raises(ValueError):
            node = tn.ParamNode(shape=(2, 5, 2),
                                data=True)
            
        with pytest.raises(ValueError):
            node = tn.ParamNode(shape=(2, 5, 2),
                                tensor=torch.randn(2, 5, 2))

    
class TestSetTensor:
    
    @pytest.fixture
    def setup(self):
        node1 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1')
        
        tensor = torch.randn(2, 5, 2)
        node2 = tn.Node(axes_names=('left', 'input', 'right'),
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
        
        # If edges are non-dangling, we can set a
        # tensor with different size in those axes
        diff_tensor = torch.randn(2, 5, 5)
        node1.tensor = diff_tensor
        assert node1.shape == (2, 5, 5)
        
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
        node3 = tn.ParamNode(axes_names=('left', 'input', 'right'),
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
        node1 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')
        node2 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')
        
        assert node1.name == 'node'
        assert node2.name == 'node'
        assert node1.network != node2.network
        
        assert isinstance(node1[2], tn.Edge)
        assert isinstance(node2[0], tn.Edge)
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tn.Edge)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
    def test_connect_paramedges(self):
        node1 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        param_edges=True)
        node2 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        param_edges=True)
        
        assert node1.name == 'node'
        assert node2.name == 'node'
        assert node1.network != node2.network
        
        assert isinstance(node1[2], tn.ParamEdge)
        assert isinstance(node2[0], tn.ParamEdge)
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tn.ParamEdge)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
    def test_connect_edge_paramedge(self):
        node1 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')
        node2 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        param_edges=True)
        
        assert node1.name == 'node'
        assert node2.name == 'node'
        assert node1.network != node2.network
        
        assert isinstance(node1[2], tn.Edge)
        assert isinstance(node2[0], tn.ParamEdge)
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tn.ParamEdge)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
    def test_connect_paramedge_edge(self):
        node1 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        param_edges=True)
        node2 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')
        
        assert node1.name == 'node'
        assert node2.name == 'node'
        assert node1.network != node2.network
        
        assert isinstance(node1[2], tn.ParamEdge)
        assert isinstance(node2[0], tn.Edge)
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tn.ParamEdge)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
    def test_connect_same_network(self):
        net = tn.TensorNetwork()
        node1 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net)
        assert node1.name == 'node'
        
        node2 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        network=net)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
        assert isinstance(node1[2], tn.Edge)
        assert isinstance(node2[0], tn.Edge)
        
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tn.Edge)
        
        assert node1.name == 'node_0'
        assert node2.name == 'node_1'
        assert node1.network == node2.network
        
    def test_connect_different_sizes(self):
        node1 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1')
        node2 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        param_edges=True)
        
        node2[0].change_size(4)
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tn.ParamEdge)
        assert new_edge.size() == 2
        assert new_edge.dim() == 2

        node1 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        param_edges=True)
        node2 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        param_edges=True)
        node1[2].change_size(3)
        node2[0].change_size(4)
        new_edge = node1[2] ^ node2[0]
        assert isinstance(new_edge, tn.ParamEdge)
        assert new_edge.size() == 3
        assert new_edge.dim() == 2
    
    
class TestParameterize:
    
    def test_parameterize_node(self):
        node1 = tn.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 2))
        node2 = tn.Node(axes_names=('left', 'input', 'right'),
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
        
        assert isinstance(paramnode1, tn.ParamNode)
        assert paramnode1['left'].node1 == paramnode1
        assert isinstance(paramnode1['left'], tn.Edge)
        
    def test_deparameterize_paramnode(self):
        paramnode1 = tn.ParamNode(axes_names=('left', 'input', 'right'),
                                  name='paramnode1',
                                  tensor=torch.randn(2, 5, 2))
        node2 = tn.Node(axes_names=('left', 'input', 'right'),
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
        
        assert isinstance(node1, tn.Node)
        assert node1['left'].node1 == node1
        assert isinstance(node1['left'], tn.Edge)
        
    def test_parameterize_dangling_edge(self):
        node = tn.Node(axes_names=('left', 'input', 'right'),
                       tensor=torch.randn(3, 5, 2))
        net = node.network
        
        prev_edge = node['left']
        assert prev_edge in net.edges

        node['left'].parameterize(set_param=True, size=4)
        assert isinstance(node['left'], tn.ParamEdge)
        assert node.shape == (4, 5, 2)
        assert node.dim() == (3, 5, 2)

        assert prev_edge not in net.edges
        assert node['left'] in net.edges

        node['left'].parameterize(set_param=False)
        assert isinstance(node['left'], tn.Edge)
        assert node.shape == (3, 5, 2)
        assert node.dim() == (3, 5, 2)

        node['left'].parameterize(set_param=True, size=2)
        assert node.shape == (2, 5, 2)
        assert node.dim() == (2, 5, 2)

        node['left'].parameterize(set_param=False)
        assert node.shape == (2, 5, 2)
        assert node.dim() == (2, 5, 2)
        
    def test_parameterize_dangling_edge2(self):
        node = tn.Node(shape=(3, 5, 2),
                       axes_names=('left', 'input', 'right'))
        net = node.network
        
        prev_edge = node['left']
        assert prev_edge in net.edges

        node['left'].parameterize(set_param=True, size=4)
        assert isinstance(node['left'], tn.ParamEdge)
        assert node.shape == (4, 5, 2)
        assert node.dim() == (3, 5, 2)

        assert prev_edge not in net.edges
        assert node['left'] in net.edges

        node['left'].parameterize(set_param=False)
        assert isinstance(node['left'], tn.Edge)
        assert node.shape == (3, 5, 2)
        assert node.dim() == (3, 5, 2)

        node['left'].parameterize(set_param=True, size=2)
        assert node.shape == (2, 5, 2)
        assert node.dim() == (2, 5, 2)

        node['left'].parameterize(set_param=False)
        assert node.shape == (2, 5, 2)
        assert node.dim() == (2, 5, 2)
        
    def test_parameterize_connected_edge(self):
        node1 = tn.Node(axes_names=('left', 'input', 'right'),
                        name='node1',
                        tensor=torch.randn(2, 5, 3))
        node2 = tn.Node(axes_names=('left', 'input', 'right'),
                        name='node2',
                        tensor=torch.randn(3, 5, 2))
        node1['right'] ^ node2['left']
        
        prev_edge = node2['left']
        assert prev_edge in node2.edges

        node2['left'].parameterize(set_param=True, size=4)
        assert isinstance(node2['left'], tn.ParamEdge)
        assert node2.shape == (4, 5, 2)
        assert node2.dim() == (3, 5, 2)

        assert prev_edge not in node2.edges

        node2['left'].parameterize(set_param=False)
        assert isinstance(node2['left'], tn.Edge)
        assert node2.shape == (3, 5, 2)
        assert node2.dim() == (3, 5, 2)

        node2['left'].parameterize(set_param=True, size=2)
        assert node2.shape == (2, 5, 2)
        assert node2.dim() == (2, 5, 2)

        node2['left'].parameterize(set_param=False)
        assert node2.shape == (2, 5, 2)
        assert node2.dim() == (2, 5, 2) 
        
    def test_change_size_dim_dangling(self):
        node = tn.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       name='node',
                       param_edges=True,
                       init_method='randn')
        
        for i, edge in enumerate(node.edges):
            assert isinstance(edge, tn.ParamEdge)
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
        node1 = tn.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        param_edges=True,
                        init_method='randn')
        node2 = tn.Node(shape=(2, 5, 2),
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




def test_copy_node():
    node = tn.Node(shape=(2, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node',
                   init_method='randn')
    copy = node.copy()
    assert torch.equal(copy.tensor, node.tensor)
    for i in range(copy.rank):
        if copy.axes[i].is_node1():
            assert copy.edges[i].node1 == copy
            assert node.edges[i].node1 == node
            assert copy.edges[i].node2 == node.edges[i].node2
        else:
            assert copy.edges[i].node2 == copy
            assert node.edges[i].node2 == node
            assert copy.edges[i].node1 == node.edges[i].node1

    node = tn.ParamNode(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        init_method='randn')
    copy = node.copy()
    assert torch.equal(copy.tensor, node.tensor)
    for i in range(copy.rank):
        if copy.axes[i].is_node1():
            assert copy.edges[i].node1 == copy
            assert node.edges[i].node1 == node
            assert copy.edges[i].node2 == node.edges[i].node2
        else:
            assert copy.edges[i].node2 == copy
            assert node.edges[i].node2 == node
            assert copy.edges[i].node1 == node.edges[i].node1
            

def test_is_updated():
    node1 = tn.Node(shape=(2, 3),
                    axes_names=('left', 'right'),
                    name='node1',
                    param_edges=True,
                    init_method='ones')
    node2 = tn.Node(shape=(3, 4),
                    axes_names=('left', 'right'),
                    name='node2',
                    param_edges=True,
                    init_method='ones')
    new_edge = node1['right'] ^ node2['left']
    prev_matrix = new_edge.matrix
    optimizer = torch.optim.SGD(params=new_edge.parameters(), lr=0.1)

    node3 = node1 @ node2
    mean = node3.mean()
    mean.backward()
    optimizer.step()

    assert not new_edge.is_updated()  # TODO: a lo mejor no sirve pa na

    new_matrix = new_edge.matrix
    assert not torch.equal(prev_matrix, new_matrix)


# TODO: Hacer ParamStackNode, al hacer stack, si todos los nodos son leaf y ParamNode
# TODO: Hacer que la matriz de ParamStackEdge se construya apilando las matrices de los edges que contiene
