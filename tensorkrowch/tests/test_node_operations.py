"""
Tests for node_operations
"""

import pytest

import torch
import torch.nn as nn
import tensorkrowch as tk

import time


class TestPermute:
    
    def test_permute_node(self):
        node = tk.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       name='node',
                       init_method='randn')
        permuted_node = node.permute((0, 2, 1))

        assert permuted_node['left']._nodes[permuted_node.is_node1('left')] == \
            node['left']._nodes[node.is_node1('left')]
        assert permuted_node['input']._nodes[permuted_node.is_node1('left')] == \
            node['input']._nodes[node.is_node1('left')]
        assert permuted_node['right']._nodes[permuted_node.is_node1('left')] == \
            node['right']._nodes[node.is_node1('left')]

        assert permuted_node[0]._nodes[permuted_node.is_node1('left')] == \
            node[0]._nodes[node.is_node1('left')]
        assert permuted_node[1]._nodes[permuted_node.is_node1('left')] == \
            node[2]._nodes[node.is_node1('left')]
        assert permuted_node[2]._nodes[permuted_node.is_node1('left')] == \
            node[1]._nodes[node.is_node1('left')]

        assert torch.equal(permuted_node.tensor, node.tensor.permute(0, 2, 1))
        
    def test_permute_paramnode(self):
        node = tk.ParamNode(shape=(2, 5, 2),
                            axes_names=('left', 'input', 'right'),
                            name='node',
                            init_method='randn',
                            param_edges=True)
        permuted_node = node.permute((0, 2, 1))

        assert permuted_node['left']._nodes[permuted_node.is_node1('left')] == \
            node['left']._nodes[node.is_node1('left')]
        assert permuted_node['input']._nodes[permuted_node.is_node1('left')] == \
            node['input']._nodes[node.is_node1('left')]
        assert permuted_node['right']._nodes[permuted_node.is_node1('left')] == \
            node['right']._nodes[node.is_node1('left')]

        assert permuted_node[0]._nodes[permuted_node.is_node1('left')] == \
            node[0]._nodes[node.is_node1('left')]
        assert permuted_node[1]._nodes[permuted_node.is_node1('left')] == \
            node[2]._nodes[node.is_node1('left')]
        assert permuted_node[2]._nodes[permuted_node.is_node1('left')] == \
            node[1]._nodes[node.is_node1('left')]

        assert torch.equal(permuted_node.tensor, node.tensor.permute(0, 2, 1))
        
        # Compute gradients
        permuted_node.sum().backward()
        assert not torch.equal(node.grad, torch.zeros(node.shape))
        node.network.zero_grad()
        assert torch.equal(node.grad, torch.zeros(node.shape))
    
    
class TestBasicOps:
    
    @pytest.fixture
    def setup(self):
        node1 = tk.Node(shape=(2, 3),
                        axes_names=('left', 'right'),
                        name='node1',
                        init_method='randn')
        node2 = tk.Node(shape=(4, 5),
                        axes_names=('left', 'right'),
                        name='node2',
                        init_method='randn')
        return node1, node2
    
    def test_tprod(self, setup):
        node1, node2 = setup
        
        # Tensor product cannot be performed between nodes in different networks
        with pytest.raises(ValueError):
            node3 = node1 % node2

        net = tk.TensorNetwork()
        node1.network = net
        node2.network = net
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        node3 = node1 % node2
        assert node3.shape == (2, 3, 4, 5)
        assert node3.edges == node1.edges + node2.edges
        assert node1.successors != dict()
        assert node1.successors['tprod'][0].child == node3
        assert node2.successors == dict()
        
        result_tensor = node3.tensor

        # Second time
        node3 = node1 % node2
        assert torch.equal(result_tensor, node3.tensor)
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        net.delete_non_leaf()
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        node1[1].change_size(node2[0].size())
        node1[1] ^ node2[0]
        # Tensor product cannot be performed between connected nodes
        with pytest.raises(ValueError):
            node3 = node1 % node2
            
    def test_mul(self, setup):
        node1, node2 = setup
        node1[0].change_size(2)
        node1[1].change_size(2)
        node2[0].change_size(2)
        node2[1].change_size(2)
        
        # Tensor product cannot be performed between nodes in different networks
        with pytest.raises(ValueError):
            node3 = node1 * node2

        net = tk.TensorNetwork()
        node1.network = net
        node2.network = net
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        node3 = node1 * node2
        assert node3.shape == (2, 2)
        assert node3.edges == node1.edges
        assert node1.successors != dict()
        assert node1.successors['mul'][0].child == node3
        assert node2.successors == dict()

        result_tensor = node3.tensor

        # Second time
        node3 = node1 * node2
        assert torch.equal(result_tensor, node3.tensor)
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        net.delete_non_leaf()
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        node1[1] ^ node2[0]
        # Tensor product cannot be performed between connected nodes
        with pytest.raises(ValueError):
            node3 = node1 % node2
            
    def test_add(self, setup):
        node1, node2 = setup
        node1[0].change_size(2)
        node1[1].change_size(2)
        node2[0].change_size(2)
        node2[1].change_size(2)
        
        # Tensor product cannot be performed between nodes in different networks
        with pytest.raises(ValueError):
            node3 = node1 + node2

        net = tk.TensorNetwork()
        node1.network = net
        node2.network = net
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        node3 = node1 + node2
        assert node3.shape == (2, 2)
        assert node3.edges == node1.edges
        assert node1.successors != dict()
        assert node1.successors['add'][0].child == node3
        assert node2.successors == dict()

        result_tensor = node3.tensor

        # Second time
        node3 = node1 + node2
        assert torch.equal(result_tensor, node3.tensor)
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        net.delete_non_leaf()
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        node1[1] ^ node2[0]
        # Tensor product cannot be performed between connected nodes
        with pytest.raises(ValueError):
            node3 = node1 % node2
            
    def test_sub(self, setup):
        node1, node2 = setup
        node1[0].change_size(2)
        node1[1].change_size(2)
        node2[0].change_size(2)
        node2[1].change_size(2)
        
        # Tensor product cannot be performed between nodes in different networks
        with pytest.raises(ValueError):
            node3 = node1 - node2

        net = tk.TensorNetwork()
        node1.network = net
        node2.network = net
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        node3 = node1 - node2
        assert node3.shape == (2, 2)
        assert node3.edges == node1.edges
        assert node1.successors != dict()
        assert node1.successors['sub'][0].child == node3
        assert node2.successors == dict()

        result_tensor = node3.tensor

        # Second time
        node3 = node1 - node2
        assert torch.equal(result_tensor, node3.tensor)
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        net.delete_non_leaf()
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        node1[1] ^ node2[0]
        # Tensor product cannot be performed between connected nodes
        with pytest.raises(ValueError):
            node3 = node1 % node2
            
    @pytest.fixture
    def setup_param(self):
        node1 = tk.ParamNode(shape=(2, 3),
                             axes_names=('left', 'right'),
                             name='node1',
                             init_method='randn',
                             param_edges=True)
        node2 = tk.ParamNode(shape=(4, 5),
                             axes_names=('left', 'right'),
                             name='node2',
                             init_method='randn',
                             param_edges=True)
        return node1, node2
    
    def test_tprod_param(self, setup_param):
        node1, node2 = setup_param
        
        # Tensor product cannot be performed between nodes in different networks
        with pytest.raises(ValueError):
            node3 = node1 % node2

        net = tk.TensorNetwork()
        node1.network = net
        node2.network = net
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        node3 = node1 % node2
        assert node3.shape == (2, 3, 4, 5)
        assert node3.edges == node1.edges + node2.edges
        assert node1.successors != dict()
        assert node1.successors['tprod'][0].child == node3
        assert node2.successors == dict()
        
        result_tensor = node3.tensor

        # Second time
        node3 = node1 % node2
        assert torch.equal(result_tensor, node3.tensor)
        
        # Compute gradients
        node3.sum().backward()
        assert not torch.equal(node1.grad, torch.zeros(node1.shape))
        assert not torch.equal(node2.grad, torch.zeros(node2.shape))
        net.zero_grad()
        assert torch.equal(node1.grad, torch.zeros(node1.shape))
        assert torch.equal(node2.grad, torch.zeros(node2.shape))
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        net.delete_non_leaf()
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        node1[1].change_size(node2[0].size())
        node1[1].change_dim(node2[0].size())
        node1[1] ^ node2[0]
        # Tensor product cannot be performed between connected nodes
        with pytest.raises(ValueError):
            node3 = node1 % node2
            
    def test_mul_param(self, setup_param):
        node1, node2 = setup_param
        node1[0].change_size(2)
        node1[1].change_size(2)
        node2[0].change_size(2)
        node2[1].change_size(2)
        
        # Tensor product cannot be performed between nodes in different networks
        with pytest.raises(ValueError):
            node3 = node1 * node2

        net = tk.TensorNetwork()
        node1.network = net
        node2.network = net
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        node3 = node1 * node2
        assert node3.shape == (2, 2)
        assert node3.edges == node1.edges
        assert node1.successors != dict()
        assert node1.successors['mul'][0].child == node3
        assert node2.successors == dict()

        result_tensor = node3.tensor

        # Second time
        node3 = node1 * node2
        assert torch.equal(result_tensor, node3.tensor)
        
        # Compute gradients
        node3.sum().backward()
        assert not torch.equal(node1.grad, torch.zeros(node1.shape))
        assert not torch.equal(node2.grad, torch.zeros(node2.shape))
        net.zero_grad()
        assert torch.equal(node1.grad, torch.zeros(node1.shape))
        assert torch.equal(node2.grad, torch.zeros(node2.shape))
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        net.delete_non_leaf()
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        node1[1] ^ node2[0]
        # Tensor product cannot be performed between connected nodes
        with pytest.raises(ValueError):
            node3 = node1 % node2
            
    def test_add_param(self, setup_param):
        node1, node2 = setup_param
        node1[0].change_size(2)
        node1[1].change_size(2)
        node2[0].change_size(2)
        node2[1].change_size(2)
        
        # Tensor product cannot be performed between nodes in different networks
        with pytest.raises(ValueError):
            node3 = node1 + node2

        net = tk.TensorNetwork()
        node1.network = net
        node2.network = net
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        node3 = node1 + node2
        assert node3.shape == (2, 2)
        assert node3.edges == node1.edges
        assert node1.successors != dict()
        assert node1.successors['add'][0].child == node3
        assert node2.successors == dict()

        result_tensor = node3.tensor

        # Second time
        node3 = node1 + node2
        assert torch.equal(result_tensor, node3.tensor)
        
        # Compute gradients
        node3.sum().backward()
        assert not torch.equal(node1.grad, torch.zeros(node1.shape))
        assert not torch.equal(node2.grad, torch.zeros(node2.shape))
        net.zero_grad()
        assert torch.equal(node1.grad, torch.zeros(node1.shape))
        assert torch.equal(node2.grad, torch.zeros(node2.shape))
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        net.delete_non_leaf()
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        node1[1] ^ node2[0]
        # Tensor product cannot be performed between connected nodes
        with pytest.raises(ValueError):
            node3 = node1 % node2
            
    def test_sub_param(self, setup_param):
        node1, node2 = setup_param
        node1[0].change_size(2)
        node1[1].change_size(2)
        node2[0].change_size(2)
        node2[1].change_size(2)
        
        # Tensor product cannot be performed between nodes in different networks
        with pytest.raises(ValueError):
            node3 = node1 - node2

        net = tk.TensorNetwork()
        node1.network = net
        node2.network = net
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        node3 = node1 - node2
        assert node3.shape == (2, 2)
        assert node3.edges == node1.edges
        assert node1.successors != dict()
        assert node1.successors['sub'][0].child == node3
        assert node2.successors == dict()

        result_tensor = node3.tensor

        # Second time
        node3 = node1 - node2
        assert torch.equal(result_tensor, node3.tensor)
        
        # Compute gradients
        node3.sum().backward()
        assert not torch.equal(node1.grad, torch.zeros(node1.shape))
        assert not torch.equal(node2.grad, torch.zeros(node2.shape))
        net.zero_grad()
        assert torch.equal(node1.grad, torch.zeros(node1.shape))
        assert torch.equal(node2.grad, torch.zeros(node2.shape))
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        net.delete_non_leaf()
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        node1[1] ^ node2[0]
        # Tensor product cannot be performed between connected nodes
        with pytest.raises(ValueError):
            node3 = node1 % node2
        

class TestContractEdge:
    
    def test_contract_edge(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        init_method='randn',
                        network=net)
        edge = node1[2] ^ node2[0]
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        # Contract edge
        node3 = edge.contract()
        assert node3['left'] == node1['left']
        assert node3['input_0'] == node1['input']
        assert node3['right'] == node2['right']
        assert node3['input_1'] == node2['input']
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node3
        assert node2.successors == dict()

        # Cannot contract edges that are not shared (connected) between two nodes
        with pytest.raises(ValueError):
            tk.contract_edges([node1[0]], node1, node2)
            
    def test_contract_paramedge(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn',
                        network=net,
                        param_edges=True)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        init_method='randn',
                        network=net,
                        param_edges=True)
        edge = node1[2] ^ node2[0]
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        # Contract edge
        node3 = edge.contract()
        assert node3['left'] == node1['left']
        assert node3['input_0'] == node1['input']
        assert node3['right'] == node2['right']
        assert node3['input_1'] == node2['input']
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node3
        assert node2.successors == dict()
        
        # Compute gradient
        node3.sum().backward()
        assert node1[2].grad != (None, None)
        assert node1[2].grad != (torch.zeros(1), torch.zeros(1))
        assert node2[0].grad != (None, None)
        assert node2[0].grad != (torch.zeros(1), torch.zeros(1))
        
        net.zero_grad()
        assert node1[2].grad == (torch.zeros(1), torch.zeros(1))
        assert node2[0].grad == (torch.zeros(1), torch.zeros(1))

        # Cannot contract edges that are not shared (connected) between two nodes
        with pytest.raises(ValueError):
            tk.contract_edges([node1[0]], node1, node2)

    def test_trace_edge(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn')
        edge = node1[2] ^ node1[0]
        
        assert len(node1.network.nodes) == 1
        assert len(node1.network.leaf_nodes) == 1
        assert len(node1.network.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        
        # Contract edge
        node2 = edge.contract()
        assert len(node2.edges) == 1
        assert node2[0].axis1.name == 'input'
        
        assert len(node1.network.nodes) == 2
        assert len(node1.network.leaf_nodes) == 1
        assert len(node1.network.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node2
        
    def test_trace_paramedge(self):
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn',
                        param_edges=True)
        edge = node1[2] ^ node1[0]
        
        assert len(node1.network.nodes) == 1
        assert len(node1.network.leaf_nodes) == 1
        assert len(node1.network.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        
        # Contract edge
        node2 = edge.contract()
        assert len(node2.edges) == 1
        assert node2[0].axis1.name == 'input'
        
        assert len(node1.network.nodes) == 2
        assert len(node1.network.leaf_nodes) == 1
        assert len(node1.network.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node2
        
        # Compute gradient
        node2.sum().backward()
        assert node1[2].grad != (None, None)
        assert node1[2].grad != (torch.zeros(1), torch.zeros(1))
        
        node1.network.zero_grad()
        assert node1[2].grad == (torch.zeros(1), torch.zeros(1))


class TestContractBetween:
    
    def test_contract_nodes(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node2',
                        init_method='randn',
                        network=net)
        node1['left'] ^ node2['left']
        node1['right'] ^ node2['right']
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        # Contract nodes
        node3 = node1 @ node2
        assert node3.shape == (5, 5)
        assert node3.axes_names == ['input_0', 'input_1']
        assert node3.edges == [node1['input'], node2['input']]
        assert node3.network == net
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node3
        assert node2.successors == dict()
        
        result_tensor = node3.tensor

        # Repeat contraction
        node4 = node1 @ node2
        assert node3 == node4
        assert torch.equal(result_tensor, node4.tensor)
        
    def test_contract_paramnodes(self):
        net = tk.TensorNetwork()
        node1 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node1',
                             init_method='randn',
                             network=net,
                             param_edges=True)
        node2 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node1',
                             init_method='randn',
                             network=net,
                             param_edges=True)
        node1['left'] ^ node2['left']
        node1['right'] ^ node2['right']
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        # Contract nodes
        node3 = node1 @ node2
        assert node3.shape == (5, 5)
        assert node3.axes_names == ['input_0', 'input_1']
        assert node3.edges == [node1['input'], node2['input']]
        assert node3.network == net
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node3
        assert node2.successors == dict()
        
        result_tensor = node3.tensor

        # Repeat contraction
        node4 = node1 @ node2
        assert node3 == node4
        assert torch.equal(result_tensor, node4.tensor)
        
        # Compute gradient
        node3.sum().backward()
        assert not torch.equal(node1.grad, torch.zeros(node1.shape))
        assert node1[2].grad != (None, None)
        assert node1[2].grad != (torch.zeros(1), torch.zeros(1))
        assert not torch.equal(node2.grad, torch.zeros(node2.shape))
        assert node2[0].grad != (None, None)
        assert node2[0].grad != (torch.zeros(1), torch.zeros(1))
        
        net.zero_grad()
        assert torch.equal(node1.grad, torch.zeros(node1.shape))
        assert node1[2].grad == (torch.zeros(1), torch.zeros(1))
        assert torch.equal(node2.grad, torch.zeros(node2.shape))
        assert node2[0].grad == (torch.zeros(1), torch.zeros(1))
        
    def test_trace_node(self):
        node1 = tk.Node(shape=(2, 5, 5, 2),
                        axes_names=('left', 'input', 'input', 'right'),
                        name='node1',
                        init_method='randn')
        node1['left'] ^ node1['right']
        node1['input_0'] ^ node1['input_1']
        
        assert len(node1.network.nodes) == 1
        assert len(node1.network.leaf_nodes) == 1
        assert len(node1.network.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        
        # Contract edge
        node2 = node1 @ node1
        assert node2.shape == ()
        assert len(node2.edges) == 0
        
        assert len(node1.network.nodes) == 2
        assert len(node1.network.leaf_nodes) == 1
        assert len(node1.network.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node2
        
        result_tensor = node2.tensor

        # Repeat traces
        node3 = node1 @ node1
        assert node2 == node3
        assert torch.equal(result_tensor, node3.tensor)

    def test_trace_paramnode(self):
        node1 = tk.ParamNode(shape=(2, 5, 5, 2),
                             axes_names=('left', 'input', 'input', 'right'),
                             name='node1',
                             param_edges=True,
                             init_method='randn')
        node1['left'] ^ node1['right']
        node1['input_0'] ^ node1['input_1']
        
        assert len(node1.network.nodes) == 1
        assert len(node1.network.leaf_nodes) == 1
        assert len(node1.network.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        
        # Contract edge
        node2 = node1 @ node1
        assert node2.shape == ()
        assert len(node2.edges) == 0
        
        assert len(node1.network.nodes) == 2
        assert len(node1.network.leaf_nodes) == 1
        assert len(node1.network.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node2
        
        result_tensor = node2.tensor

        # Repeat traces
        node3 = node1 @ node1
        assert node2 == node3
        assert torch.equal(result_tensor, node3.tensor)
        
        # Compute gradient
        node2.sum().backward()
        assert not torch.equal(node1.grad, torch.zeros(node1.shape))
        assert node1[2].grad != (None, None)
        assert node1[2].grad != (torch.zeros(1), torch.zeros(1))
        
        node1.network.zero_grad()
        assert torch.equal(node1.grad, torch.zeros(node1.shape))
        assert node1[2].grad == (torch.zeros(1), torch.zeros(1))

    def test_contract_with_same_batch(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'batch', 'right'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'batch', 'right'),
                        name='node2',
                        init_method='randn',
                        network=net)
        node1['left'] ^ node2['left']
        node1['right'] ^ node2['right']
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        # Contract nodes
        node3 = node1 @ node2
        assert node3.shape == (5,)
        assert node3.axes_names == ['batch']
        assert node3.edges == [node1['batch']]
        assert node3.network == net
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node3
        assert node2.successors == dict()
        
        result_tensor = node3.tensor

        # Repeat contraction
        node4 = node1 @ node2
        assert node3 == node4
        assert torch.equal(result_tensor, node4.tensor)
        
    def test_contract_with_diff_batch(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'batch1', 'right'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'batch2', 'right'),
                        name='node2',
                        init_method='randn',
                        network=net)
        node1['left'] ^ node2['left']
        node1['right'] ^ node2['right']
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        # Contract nodes
        node3 = node1 @ node2
        assert node3.shape == (5, 5)
        assert node3.axes_names == ['batch1', 'batch2']
        assert node3.edges == [node1['batch1'], node2['batch2']]
        assert node3.network == net
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node3
        assert node2.successors == dict()
        
        result_tensor = node3.tensor

        # Repeat contraction
        node4 = node1 @ node2
        assert node3 == node4
        assert torch.equal(result_tensor, node4.tensor)
        
    def test_contract_several_batches(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(2, 10, 15, 5, 6),
                        axes_names=('input', 'batch', 'my_batch', 'stack', 'my_stack'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(2, 10, 20, 5, 8),
                        axes_names=('input', 'batch', 'your_batch', 'stack', 'your_stack'),
                        name='node2',
                        init_method='randn',
                        network=net)
        node1['input'] ^ node2['input']
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        # Contract nodes
        node3 = node1 @ node2
        assert node3.shape == (10, 5, 15, 6, 20, 8)
        assert node3.axes_names == ['batch', 'stack',
                                    'my_batch', 'my_stack',
                                    'your_batch', 'your_stack']
        assert node3.edges == [node1['batch'], node1['stack'],
                               node1['my_batch'], node1['my_stack'],
                               node2['your_batch'], node2['your_stack']]
        assert node3.network == net
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        
        assert node1.successors != dict()
        assert node1.successors['contract_edges'][0].child == node3
        assert node2.successors == dict()
        
        result_tensor = node3.tensor

        # Repeat contraction
        node4 = node1 @ node2
        assert node3 == node4
        assert torch.equal(result_tensor, node4.tensor)


class TestEinsum:
    
    def test_einsum(self):
        net = tk.TensorNetwork(name='net')
        node = tk.Node(shape=(5, 5, 5, 5, 2), axes_names=('input', 'input', 'input', 'input', 'output'), network=net,
                    init_method='randn')
        net.set_data_nodes(node.edges[:-1], ['batch'])
        data = torch.randn(4, 10, 5)
        net._add_data(data)

        out_node = tk.einsum('ijklm,bi,bj,bk,bl->bm', *([node] + list(net.data_nodes.values())))
        assert out_node.shape == (10, 2)
        with pytest.raises(ValueError):
            out_node = tk.einsum('ijklm,bi,bj,bk,bl->bm', *(list(net.data_nodes.values())))
        with pytest.raises(ValueError):
            out_node = tk.einsum('ijklm,bi,bj,bk,bl->b', *([node] + list(net.data_nodes.values())))
        with pytest.raises(ValueError):
            out_node = tk.einsum('ijklm,b,bj,bk,bl->bm', *([node] + list(net.data_nodes.values())))

        net = tk.TensorNetwork(name='net')
        node = tk.ParamNode(shape=(5, 5, 5, 5, 2), axes_names=('input', 'input', 'input', 'input', 'output'), network=net,
                            param_edges=True, init_method='randn')
        net.set_data_nodes(node.edges[:-1], ['batch'])
        data = torch.randn(4, 10, 5)
        net._add_data(data)

        out_node = tk.einsum('ijklm,bi,bj,bk,bl->bm', *([node] + list(net.data_nodes.values())))
        assert out_node.shape == (10, 2)
        with pytest.raises(ValueError):
            out_node = tk.einsum('ijklm,bi,bj,bk,bl->bm', *(list(net.data_nodes.values())))
        with pytest.raises(ValueError):
            out_node = tk.einsum('ijklm,bi,bj,bk,bl->b', *([node] + list(net.data_nodes.values())))
        with pytest.raises(ValueError):
            out_node = tk.einsum('ijklm,b,bj,bk,bl->bm', *([node] + list(net.data_nodes.values())))



class TestStack:
    
    def test_stack(self):
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for _ in range(5):
            node = tk.Node(shape=(3, 3, 2),
                           axes_names=('input', 'input', 'output'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes.append(node)
            input_edges += [node['input_0'], node['input_1']]
        
        net.set_data_nodes(input_edges, ['batch'])
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)

        stack_node = tk.stack(nodes, name='stack_node')
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes],
                                 name='stack_input_0')
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes],
                                 name='stack_input_1')
        stack_node['input_0'] ^ stack_input_0['feature']
        stack_node['input_1'] ^ stack_input_1['feature']

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        unbinded_nodes = tk.unbind(stack_result)
        assert len(unbinded_nodes) == 5
        assert unbinded_nodes[0].shape == (10, 2)

        second_stack = tk.stack(unbinded_nodes[0::2], name='second_stack')
        assert second_stack.shape == (3, 10, 2)

        # Repeat operations second time
        stack_node = tk.stack(nodes, name='stack_node')
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes],
                                name='stack_input_0')
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes],
                                name='stack_input_1')
        stack_node['input_0'] ^ stack_input_0['feature']
        stack_node['input_1'] ^ stack_input_1['feature']

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        nodes = tk.unbind(stack_result)
        assert len(nodes) == 5
        assert nodes[0].shape == (10, 2)

        second_stack = tk.stack(unbinded_nodes[0::2], name='second_stack')
        assert second_stack.shape == (3, 10, 2)

    def test_stack_aux(self):
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for i in range(5):
            node = tk.Node(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
                        init_method='randn')
            nodes.append(node)
            input_edges += [node['input_0'], node['input_1']]
        net.set_data_nodes(input_edges, ['batch'])
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)

        stack_node = tk.stack(nodes, name='stack_node')
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes],
                                name='stack_input_0')
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes],
                                name='stack_input_1')
        stack_node['input_0'] ^ stack_input_0['feature']
        stack_node['input_1'] ^ stack_input_1['feature']

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        unbinded_nodes = tk.unbind(stack_result)
        assert len(unbinded_nodes) == 5
        assert unbinded_nodes[0].shape == (10, 2)

        second_stack = tk.stack(unbinded_nodes[0::2], name='second_stack')
        assert second_stack.shape == (3, 10, 2)

        # Repeat operations second time
        stack_node = tk.stack(nodes, name='stack_node')
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes],
                                name='stack_input_0')
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes],
                                name='stack_input_1')
        stack_node['input_0'] ^ stack_input_0['feature']
        stack_node['input_1'] ^ stack_input_1['feature']

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        nodes = tk.unbind(stack_result)
        assert len(nodes) == 5
        assert nodes[0].shape == (10, 2)

        second_stack = tk.stack(unbinded_nodes[0::2], name='second_stack')
        assert second_stack.shape == (3, 10, 2)
        
        
        
        

        # Param
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for i in range(5):
            node = tk.ParamNode(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
                                param_edges=True, init_method='randn')
            nodes.append(node)
            input_edges += [node['input_0'], node['input_1']]
        net.set_data_nodes(input_edges, ['batch'])
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)

        net._contracting = True
        stack_node = tk.stack(nodes, name='stack_node')
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes],
                                name='stack_input_0')
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes],
                                name='stack_input_1')
        stack_node['input_0'] ^ stack_input_0['feature']  # TODO: estos no son StackEdge, se añadena la Tn si o si
        stack_node['input_1'] ^ stack_input_1['feature']

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        unbinded_nodes = tk.unbind(stack_result)
        assert len(unbinded_nodes) == 5
        assert unbinded_nodes[0].shape == (10, 2)

        second_stack = tk.stack(unbinded_nodes[0::2], name='second_stack')
        assert second_stack.shape == (3, 10, 2)

        # Repeat operations second time
        stack_node = tk.stack(nodes, name='stack_node')
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes],
                                name='stack_input_0')
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes],
                                name='stack_input_1')
        stack_node['input_0'] ^ stack_input_0['feature']
        stack_node['input_1'] ^ stack_input_1['feature']

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        nodes = tk.unbind(stack_result)
        assert len(nodes) == 5
        assert nodes[0].shape == (10, 2)

        second_stack = tk.stack(unbinded_nodes[0::2], name='second_stack')
        assert second_stack.shape == (3, 10, 2)

        # Dimensions
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for i in range(5):
            node = tk.ParamNode(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
                                param_edges=True, init_method='randn')
            nodes.append(node)
            input_edges += [node['input_0'], node['input_1']]
        net.set_data_nodes(input_edges, ['batch'])
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)

        stack_node = tk.stack(nodes, name='stack_node')
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes],
                                name='stack_input_0')
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes],
                                name='stack_input_1')

        # If edges have not the same dimension in an axis
        # that is not connected, it does not matter
        stack_node.edges_dict['output'][0].change_dim(1)
        assert stack_node.edges_dict['output'][0].dim() == 1

        # If edges have not the same dimension in an axis
        # that is going to be connected, it is raised a ValueError
        stack_node.edges_dict['input_0'][0].change_dim(1)
        assert stack_node.edges_dict['input_0'][0].dim() == 1
        with pytest.raises(ValueError):
            stack_node['input_0'] ^ stack_input_0['feature']

        # If edges have not the same parameters in an axis,
        # but they have the same dimension, the parameters
        # are changed to be shared among all edges
        stack_node.edges_dict['input_1'][0].set_parameters(shift=-1., slope=30.)
        assert stack_node.edges_dict['input_1'][0].shift == -1.
        assert stack_node.edges_dict['input_1'][0].slope == 30.
        assert stack_node.edges_dict['input_1'][1].shift == -0.5
        assert stack_node.edges_dict['input_1'][1].slope == 20.

        stack_node['input_1'] ^ stack_input_1['feature']
        assert stack_node.edges_dict['input_1'][0].shift == -1.
        assert stack_node.edges_dict['input_1'][0].slope == 30.
        assert stack_node.edges_dict['input_1'][1].shift == -1.
        assert stack_node.edges_dict['input_1'][1].slope == 30.

        # If stack edges are not connected, but they should,
        # we connect them in einsum
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for i in range(5):
            node = tk.Node(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
                        init_method='randn')
            nodes.append(node)
            input_edges += [node['input_0'], node['input_1']]
        net.set_data_nodes(input_edges, ['batch'])
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)

        stack_node = tk.stack(nodes, name='stack_node')
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes],
                                name='stack_input_0')
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes],
                                name='stack_input_1')

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        nodes = tk.unbind(stack_result)
        assert len(nodes) == 5
        assert nodes[0].shape == (10, 2)

        # Error 1
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for i in range(5):
            node = tk.Node(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
                        init_method='randn')
            nodes.append(node)
            input_edges += [node['input_0'], node['input_1']]
        net.set_data_nodes(input_edges, ['batch'])
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)
        net['data_0'].disconnect()

        stack_node = tk.stack(nodes, name='stack_node')
        stack_input_0 = tk.stack([net['data_0']] + [node.neighbours('input_0') for node in nodes][1:],
                                name='stack_input_0')

        # Try to connect stack edges when some edge of one stack is
        # not connected to the corresponding edge in the other stack
        with pytest.raises(ValueError):
            stack_node['input_0'] ^ stack_input_0['feature']

        # Error 2
        net = tk.TensorNetwork()
        nodes = []
        for i in range(5):
            node = tk.Node(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
                        init_method='randn')
            nodes.append(node)
        net['node_0'].param_edges(True)

        # Try to stack edges of different types
        with pytest.raises(TypeError):
            stack_node = tk.stack(nodes, name='stack_node')


def test_stacked_einsum():
    # MPS
    net = tk.TensorNetwork()
    nodes = []
    input_edges = []
    for i in range(11):
        node = tk.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name=f'node_{i}', network=net,
                       init_method='randn')
        nodes.append(node)
        if i != 5:
            input_edges.append(node['input'])
    for i in range(10):
        net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']
    net.set_data_nodes(input_edges, ['batch'])
    data = torch.randn(10, 10, 5)
    net._add_data(data)
    result_list = tk.stacked_einsum('lir,bi->lbr', nodes[:5] + nodes[6:], list(net.data_nodes.values()))
    result_list = result_list[:5] + [nodes[5]] + result_list[5:]

    node1 = result_list[0]
    for i in range(1, 11):
        node1 @= result_list[i]
    assert node1.shape == (2, 10, 5, 2)

    node2 = result_list[0]
    for i in range(1, 5):
        node2 = tk.einsum('lbr,rbs->lbs', node2, result_list[i])
    node2 = tk.einsum('lbr,ris->lbis', node2, result_list[5])
    for i in range(6, 11):
        node2 = tk.einsum('lbir,rbs->lbis', node2, result_list[i])
    assert node2.shape == (2, 10, 5, 2)

    assert torch.equal(node1.tensor, node2.tensor)

    # Param
    net = tk.TensorNetwork()
    nodes = []
    input_edges = []
    for i in range(11):
        node = tk.ParamNode(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name=f'node_{i}', network=net,
                            param_edges=True, init_method='randn')
        nodes.append(node)
        if i != 5:
            input_edges.append(node['input'])
    for i in range(10):
        net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']
    net.set_data_nodes(input_edges, ['batch'])
    data = torch.randn(10, 10, 5)
    net._add_data(data)
    result_list = tk.stacked_einsum('lir,bi->lbr', nodes[:5] + nodes[6:], list(net.data_nodes.values()))
    result_list = result_list[:5] + [nodes[5]] + result_list[5:]

    node1 = result_list[0]
    for i in range(1, 11):
        node1 @= result_list[i]
    assert node1.shape == (2, 10, 5, 2)

    mean1 = node1.tensor.mean()
    mean1.backward()

    # When stacking, stacked edges get new parameters, shared
    # among all stacked edges
    for i, _ in enumerate(nodes[:-1]):
        if (i != 5) and (i != 4):
            assert nodes[i]['input'].grad == nodes[i + 1]['input'].grad

    node2 = result_list[0]
    for i in range(1, 5):
        node2 = tk.einsum('lbr,rbs->lbs', node2, result_list[i])
    node2 = tk.einsum('lbr,ris->lbis', node2, result_list[5])
    for i in range(6, 11):
        node2 = tk.einsum('lbir,rbs->lbis', node2, result_list[i])
    assert node2.shape == (2, 10, 5, 2)

    print(node1.tensor)
    print(node2.tensor)
    print(torch.equal(node1.tensor, node2.tensor))
    print(torch.eq(node1.tensor, node2.tensor))

    # TODO: parecen iguales pero da False, un poco raro
    #assert torch.equal(node1.tensor, node2.tensor)


# TODO: definir contraccion para traza
# TODO: en stack y unbind, si los datos estaban guardados antes en una pila y se
#  pueden ahorrar algunos calculos, podemos llevar despues de la primera epoca una pista de lo que hay que hacer
#  (e.g. hacemos unbind de una pila y despu'es volvemos a hacer stack de un subconjunto; como siguen guardados
#  originalmente en la pila, en realidad solo hay que indexar la pila)
