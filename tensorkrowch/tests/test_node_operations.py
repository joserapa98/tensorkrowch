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
        

class TestSplit:
    
    def test_split_contracted_node(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(10, 2, 5, 4),
                        axes_names=('batch', 'left', 'input', 'right'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(10, 4, 5, 3),
                        axes_names=('batch', 'left', 'input', 'right'),
                        name='node2',
                        init_method='randn',
                        network=net)
        edge = node1[3] ^ node2[1]
        result = node1.contract(3)
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 1
        assert node1.successors['contract_edges'][0].child == result
        
        # Split result
        new_node1, new_node2 = result.split(node1_axes=['left', 'input_0'],
                                            node2_axes=['input_1', 'right'])
        
        assert new_node1.shape == (10, 2, 5, 10)
        assert new_node1['batch'].size() == 10
        assert new_node1['left'].size() == 2
        assert new_node1['input'].size() == 5
        assert new_node1['splitted'].size() == 10
        
        assert new_node2.shape == (10, 10, 5, 3)
        assert new_node2['batch'].size() == 10
        assert new_node2['splitted'].size() == 10
        assert new_node2['input'].size() == 5
        assert new_node2['right'].size() == 3
        
        assert len(net.nodes) == 5
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 3
        assert result.successors['split'][0].child == [new_node1, new_node2]
        
        assert net.edges == [node1['left'], node1['input'],
                             node2['input'], node2['right']]
        
        # If a splitted node is deleted (or just the splitted edge is
        # disconnected), the neighbour node's splitted edge joins the
        # network's set of edges. This is something people shouldn't do
        net.delete_node(new_node1)
        assert net.edges == [node1['left'], node1['input'],
                             node2['input'], node2['right'],
                             new_node2['splitted']]
        
        net.delete_node(new_node2)
        assert net.edges == [node1['left'], node1['input'],
                             node2['input'], node2['right']]
        
    def test_split_contracted_node_disordered_axes(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(10, 2, 20, 5, 4),
                        axes_names=('batch1', 'left', 'batch2', 'input', 'right'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(20, 4, 5, 3, 10),
                        axes_names=('batch2', 'left', 'input', 'right', 'batch1'),
                        name='node2',
                        init_method='randn',
                        network=net)
        edge = node1['right'] ^ node2['left']
        result = node1.contract('right')
        
        assert result.edges == [node1['batch1'], node1['batch2'],
                                node1['left'], node1['input'],
                                node2['input'], node2['right']]
        
        # Split result
        new_node1, new_node2 = result.split(node1_axes=['input_0', 'left'],
                                            node2_axes=['right', 'batch2', 'input_1'])
        
        assert new_node1.shape == (10, 5, 2, 10)
        assert new_node1.axes_names == ['batch1', 'input', 'left', 'splitted']
        
        assert new_node2.shape == (10, 10, 3, 20, 5)
        assert new_node2.axes_names == ['batch1', 'splitted', 'right', 'batch2', 'input']
        
    def test_split_contracted_node_rank(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(10, 2, 5, 4),
                        axes_names=('batch', 'left', 'input', 'right'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(10, 4, 5, 3),
                        axes_names=('batch', 'left', 'input', 'right'),
                        name='node2',
                        init_method='randn',
                        network=net)
        edge = node1[3] ^ node2[1]
        result = node1.contract(3)
        
        # Split result
        new_node1, new_node2 = result.split(node1_axes=['left', 'input_0'],
                                            node2_axes=['input_1', 'right'],
                                            rank=7)
        
        assert new_node1.shape == (10, 2, 5, 7)
        assert new_node1['batch'].size() == 10
        assert new_node1['left'].size() == 2
        assert new_node1['input'].size() == 5
        assert new_node1['splitted'].size() == 7
        
        assert new_node2.shape == (10, 7, 5, 3)
        assert new_node2['batch'].size() == 10
        assert new_node2['splitted'].size() == 7
        assert new_node2['input'].size() == 5
        assert new_node2['right'].size() == 3
        
        # Repeat operation
        new_node1, new_node2 = result.split(node1_axes=['left', 'input_0'],
                                            node2_axes=['input_1', 'right'],
                                            rank=7)
        
        assert new_node1.shape == (10, 2, 5, 7)
        assert new_node2.shape == (10, 7, 5, 3)
        
    def test_split_contracted_node_cum_percentage(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(10, 2, 5, 4),
                        axes_names=('batch', 'left', 'input', 'right'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(10, 4, 5, 3),
                        axes_names=('batch', 'left', 'input', 'right'),
                        name='node2',
                        init_method='randn',
                        network=net)
        edge = node1[3] ^ node2[1]
        result = node1.contract(3)
        
        high_rank_tensor = torch.eye(10, 15).expand(10, 10, 15).reshape(10, 2, 5, 5, 3)
        result._unrestricted_set_tensor(high_rank_tensor)
        
        # Split result
        new_node1, new_node2 = result.split(node1_axes=['left', 'input_0'],
                                            node2_axes=['input_1', 'right'],
                                            cum_percentage=0.9)
        
        assert new_node1.shape == (10, 2, 5, 9)
        assert new_node1['batch'].size() == 10
        assert new_node1['left'].size() == 2
        assert new_node1['input'].size() == 5
        assert new_node1['splitted'].size() == 9
        
        assert new_node2.shape == (10, 9, 5, 3)
        assert new_node2['batch'].size() == 10
        assert new_node2['splitted'].size() == 9
        assert new_node2['input'].size() == 5
        assert new_node2['right'].size() == 3
        
        # Repeat operation with low rank tensor
        low_rank_tensor = torch.zeros(10, 15)
        low_rank_tensor[0, 0] = 1.
        low_rank_tensor = low_rank_tensor.expand(10, 10, 15).reshape(10, 2, 5, 5, 3)
        result._unrestricted_set_tensor(low_rank_tensor)
        
        new_node1, new_node2 = result.split(node1_axes=['left', 'input_0'],
                                            node2_axes=['input_1', 'right'],
                                            cum_percentage=0.9)
        
        # When using cum_percentage, if the tensor rank changes,
        # the dimension of the splitted edge changes with it
        assert new_node1.shape == (10, 2, 5, 1)
        assert new_node2.shape == (10, 1, 5, 3)
        
    def test_split_contracted_node_paramnodes_cum_percentage(self):
        net = tk.TensorNetwork()
        node1 = tk.ParamNode(shape=(10, 2, 5, 4),
                             axes_names=('batch', 'left', 'input', 'right'),
                             name='node1',
                             init_method='randn',
                             param_edges=True,
                             network=net)
        node2 = tk.ParamNode(shape=(10, 4, 5, 3),
                             axes_names=('batch', 'left', 'input', 'right'),
                             name='node2',
                             init_method='randn',
                             param_edges=True,
                             network=net)
        edge = node1[3] ^ node2[1]
        result = node1.contract(3)
        
        high_rank_tensor = torch.eye(10, 15).expand(10, 10, 15).reshape(10, 2, 5, 5, 3)
        result._unrestricted_set_tensor(high_rank_tensor)
        
        # Split result
        new_node1, new_node2 = result.split(node1_axes=['left', 'input_0'],
                                            node2_axes=['input_1', 'right'],
                                            cum_percentage=0.9)
        
        assert new_node1.shape == (10, 2, 5, 9)
        assert new_node1['batch'].size() == 10
        assert new_node1['left'].size() == 2
        assert new_node1['input'].size() == 5
        assert new_node1['splitted'].size() == 9
        
        assert new_node2.shape == (10, 9, 5, 3)
        assert new_node2['batch'].size() == 10
        assert new_node2['splitted'].size() == 9
        assert new_node2['input'].size() == 5
        assert new_node2['right'].size() == 3
        
        # Repeat operation with low rank tensor
        low_rank_tensor = torch.zeros(10, 15)
        low_rank_tensor[0, 0] = 1.
        low_rank_tensor = low_rank_tensor.expand(10, 10, 15).reshape(10, 2, 5, 5, 3)
        result._unrestricted_set_tensor(low_rank_tensor)
        
        new_node1, new_node2 = result.split(node1_axes=['left', 'input_0'],
                                            node2_axes=['input_1', 'right'],
                                            cum_percentage=0.9)
        
        # When using cum_percentage, if the tensor rank changes,
        # the dimension of the splitted edge changes with it
        assert new_node1.shape == (10, 2, 5, 1)
        assert new_node2.shape == (10, 1, 5, 3)
        
    def test_split_in_place(self):
        net = tk.TensorNetwork()
        node1 = tk.Node(shape=(10, 2, 5, 4),
                        axes_names=('batch', 'left', 'input', 'right'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(10, 4, 5, 3),
                        axes_names=('batch', 'left', 'input', 'right'),
                        name='node2',
                        init_method='randn',
                        network=net)
        node3 = tk.Node(shape=(10, 3, 5, 7),
                        axes_names=('batch', 'left', 'input', 'right'),
                        name='node3',
                        init_method='randn',
                        network=net)
        edge = node1['right'] ^ node2['left']
        node2['right'] ^ node3['left']
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 3
        assert len(net.non_leaf_nodes) == 0
        
        result = node1.contract_('right')
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        assert node1.network != net
        assert node2.network != net
        
        # Split result
        new_node1, new_node2 = result.split_(node1_axes=['left', 'input_0'],
                                             node2_axes=['input_1', 'right'])
        
        assert new_node1.shape == (10, 2, 5, 10)
        assert new_node1['batch'].size() == 10
        assert new_node1['left'].size() == 2
        assert new_node1['input'].size() == 5
        assert new_node1['splitted'].size() == 10
        
        assert new_node2.shape == (10, 10, 5, 3)
        assert new_node2['batch'].size() == 10
        assert new_node2['splitted'].size() == 10
        assert new_node2['input'].size() == 5
        assert new_node2['right'].size() == 3
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 3
        assert len(net.non_leaf_nodes) == 0
        assert result.network != net
        
    def test_split_paramnode(self):
        node = tk.ParamNode(shape=(10, 7, 5, 4),
                            axes_names=('left', 'right', 'left', 'right'),
                            init_method='randn',
                            param_edges=True)
        net = node.network
        
        assert len(net.nodes) == 1
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 0
        
        # Split result
        node1, node2 = node.split(node1_axes=['left_0', 'left_1'],
                                  node2_axes=['right_0', 'right_1'])
        
        assert node1.shape == (10, 5, 28)
        assert node2.shape == (28, 7, 4)
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 2
        
        assert net.edges == node.edges
        
    def test_split_paramnode_in_place(self):
        node = tk.ParamNode(shape=(10, 7, 5, 4),
                            axes_names=('left', 'right', 'left', 'right'),
                            init_method='randn',
                            param_edges=True)
        net = node.network
        
        assert len(net.nodes) == 1
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 0
        
        # Split result
        node1, node2 = node.split_(node1_axes=['left_0', 'left_1'],
                                   node2_axes=['right_0', 'right_1'])
        
        assert node1.shape == (10, 5, 28)
        assert node2.shape == (28, 7, 4)
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert net.edges == node1.edges[:-1] + node2.edges[1:]
        
    def test_split_paramnode_with_trace(self):
        node = tk.ParamNode(shape=(10, 7, 10, 7),
                            axes_names=('left', 'right', 'left', 'right'),
                            init_method='randn',
                            param_edges=True)
        node['left_0'] ^ node['left_1']
        node['right_0'] ^ node['right_1']
        net = node.network
        
        assert len(net.nodes) == 1
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 0
        
        # Split result
        node1, node2 = node.split(node1_axes=['left_0', 'right_0'],
                                  node2_axes=['left_1', 'right_1'])
        
        assert node1.shape == (10, 7, 70)
        assert node2.shape == (70, 10, 7)
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 2
        
        assert net.edges == []
        
        # The splitted ParamEdge is not included
        assert len(net._modules) == 2
        assert list(net._modules.values()) == [node['left_0'],
                                               node['right_0']]
        
    def test_split_paramnode_with_trace_in_node1(self):
        node = tk.ParamNode(shape=(10, 10, 7, 7),
                            axes_names=('left', 'right', 'left', 'right'),
                            init_method='randn',
                            param_edges=True)
        node['left_0'] ^ node['right_0']
        net = node.network
        
        assert len(net.nodes) == 1
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 0
        
        # Split result
        node1, node2 = node.split(node1_axes=['left_0', 'right_0'],
                                  node2_axes=['left_1', 'right_1'])
        
        assert node1.shape == (10, 10, 49)
        assert node2.shape == (49, 7, 7)
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 2
        
        assert net.edges == [node['left_1'], node['right_1']]
        
        # The splitted ParamEdge is not included
        assert len(net._modules) == 3
        assert list(net._modules.values()) == [node['left_1'],
                                               node['right_1'],
                                               node['left_0']]
        
    def test_split_paramnode_with_trace_inplace(self):
        node = tk.ParamNode(shape=(10, 7, 10, 7),
                            axes_names=('left', 'right', 'left', 'right'),
                            init_method='randn',
                            param_edges=True)
        node['left_0'] ^ node['left_1']
        node['right_0'] ^ node['right_1']
        net = node.network
        
        assert len(net.nodes) == 1
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 0
        
        # Split result
        node1, node2 = node.split_(node1_axes=['left_0', 'right_0'],
                                   node2_axes=['left_1', 'right_1'])
        
        assert node1.shape == (10, 7, 70)
        assert node2.shape == (70, 10, 7)
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert net.edges == []
        
        # The 2 original ParamEdge plus the splitted ParamEdge
        assert len(net._modules) == 3
        assert list(net._modules.values()) == [node1['splitted'],
                                               node1['left'],
                                               node1['right']]
        
    def test_split_paramnode_with_trace_in_node1_inplace(self):
        node = tk.ParamNode(shape=(10, 10, 7, 7),
                            axes_names=('left', 'right', 'left', 'right'),
                            init_method='randn',
                            param_edges=True)
        node['left_0'] ^ node['right_0']
        net = node.network
        
        assert len(net.nodes) == 1
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 0
        
        # Split result
        node1, node2 = node.split_(node1_axes=['left_0', 'right_0'],
                                   node2_axes=['left_1', 'right_1'])
        
        assert node1.shape == (10, 10, 49)
        assert node2.shape == (49, 7, 7)
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert net.edges == [node2['left'], node2['right']]
        
        # The splitted ParamEdge is not included
        assert len(net._modules) == 4
        assert list(net._modules.values()) == [node1['left'],
                                               node1['splitted'],
                                               node2['left'],
                                               node2['right']]
            

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
        return net, edge, node1
    
    def test_svd_edge_rank(self, setup):
        net, edge, node1 = setup
        assert isinstance(edge, tk.Edge)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        new_node1, new_node2 = node1.svd(axis='right', rank=2)
        
        assert tk.utils.erase_enum(new_node1.name) == 'svd'
        assert tk.utils.erase_enum(new_node2.name) == 'svd'
        assert new_node1.axes_names == edge.node1.axes_names
        assert new_node1.axes_names == edge.node1.axes_names
        
        assert edge.node1.shape == (3, 5, 3)
        assert edge.node2.shape == (3, 5, 3)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert new_node1.shape == (3, 5, 2)
        assert new_node2.shape == (2, 5, 3)
        assert new_node1['right'].size() == 2
        assert new_node1['right'].dim() == 2
        
        assert len(net.nodes) == 5
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 3
        
        # Repeat operation
        new_node1, new_node2 = node1.svd(axis='right', rank=2)
        
    def test_svd_edge_cum_percentage(self, setup):
        net, edge, node1 = setup
        assert isinstance(edge, tk.Edge)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        tensor1 = torch.eye(15, 3).reshape(3, 5, 3)
        edge.node1.tensor = tensor1
        
        tensor2 = torch.eye(3, 15).reshape(3, 5, 3)
        edge.node2.tensor = tensor2
        
        new_node1, new_node2 = node1.svd(axis='right', cum_percentage=0.5)
        
        assert tk.utils.erase_enum(new_node1.name) == 'svd'
        assert tk.utils.erase_enum(new_node2.name) == 'svd'
        assert new_node1.axes_names == edge.node1.axes_names
        assert new_node1.axes_names == edge.node1.axes_names
        
        assert edge.node1.shape == (3, 5, 3)
        assert edge.node2.shape == (3, 5, 3)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert new_node1.shape == (3, 5, 2)
        assert new_node2.shape == (2, 5, 3)
        assert new_node1['right'].size() == 2
        assert new_node1['right'].dim() == 2
        
        assert len(net.nodes) == 5
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 3
        
        # Repeat operation
        new_node1, new_node2 = node1.svd(axis='right', cum_percentage=0.5)
        
    def test_svd_paramedge_rank(self, setup):
        net, edge, node1 = setup
        edge = edge.parameterize()
        
        assert isinstance(edge, tk.ParamEdge)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        new_node1, new_node2 = node1.svd(axis='right', rank=2)
        
        assert tk.utils.erase_enum(new_node1.name) == 'svd'
        assert tk.utils.erase_enum(new_node2.name) == 'svd'
        assert new_node1.axes_names == edge.node1.axes_names
        assert new_node1.axes_names == edge.node1.axes_names
        
        assert edge.node1.shape == (3, 5, 3)
        assert edge.node2.shape == (3, 5, 3)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert new_node1.shape == (3, 5, 2)
        assert new_node2.shape == (2, 5, 3)
        assert new_node1['right'].size() == 2
        assert new_node1['right'].dim() == 2
        
        assert len(net.nodes) == 5
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 3
        
        # Repeat operation
        new_node1, new_node2 = node1.svd(axis='right', rank=2)
        
    def test_svd_paramedge_cum_percentage(self, setup):
        net, edge, node1 = setup
        edge = edge.parameterize()
        
        assert isinstance(edge, tk.ParamEdge)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        tensor1 = torch.eye(15, 3).reshape(3, 5, 3)
        edge.node1.tensor = tensor1
        
        tensor2 = torch.eye(3, 15).reshape(3, 5, 3)
        edge.node2.tensor = tensor2
        
        new_node1, new_node2 = node1.svd(axis='right', cum_percentage=0.5)
        
        assert tk.utils.erase_enum(new_node1.name) == 'svd'
        assert tk.utils.erase_enum(new_node2.name) == 'svd'
        assert new_node1.axes_names == edge.node1.axes_names
        assert new_node1.axes_names == edge.node1.axes_names
        
        assert edge.node1.shape == (3, 5, 3)
        assert edge.node2.shape == (3, 5, 3)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert new_node1.shape == (3, 5, 2)
        assert new_node2.shape == (2, 5, 3)
        assert new_node1['right'].size() == 2
        assert new_node1['right'].dim() == 2
        
        assert len(net.nodes) == 5
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 3
        
        # Repeat operation
        new_node1, new_node2 = node1.svd(axis='right', cum_percentage=0.5)
        
    def test_svd_edge_rank_inplace(self, setup):
        net, edge, node1 = setup
        assert isinstance(edge, tk.Edge)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        new_node1, new_node2 = node1.svd_(axis='right', rank=2)
        
        assert tk.utils.erase_enum(new_node1.name) == \
            tk.utils.erase_enum(edge.node1.name)
        assert tk.utils.erase_enum(new_node2.name) == \
            tk.utils.erase_enum(edge.node2.name)
        assert new_node1.axes_names == edge.node1.axes_names
        assert new_node1.axes_names == edge.node1.axes_names
        
        assert new_node1.shape == (3, 5, 2)
        assert new_node2.shape == (2, 5, 3)
        assert new_node1['right'].size() == 2
        assert new_node1['right'].dim() == 2
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
    def test_svd_edge_cum_percentage_inplace(self, setup):
        net, edge, node1 = setup
        assert isinstance(edge, tk.Edge)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        tensor1 = torch.eye(15, 3).reshape(3, 5, 3)
        edge.node1.tensor = tensor1
        
        tensor2 = torch.eye(3, 15).reshape(3, 5, 3)
        edge.node2.tensor = tensor2
        
        new_node1, new_node2 = node1.svd_(axis='right', cum_percentage=0.5)
        
        assert tk.utils.erase_enum(new_node1.name) == \
            tk.utils.erase_enum(edge.node1.name)
        assert tk.utils.erase_enum(new_node2.name) == \
            tk.utils.erase_enum(edge.node2.name)
        assert new_node1.axes_names == edge.node1.axes_names
        assert new_node1.axes_names == edge.node1.axes_names
        
        assert new_node1.shape == (3, 5, 2)
        assert new_node2.shape == (2, 5, 3)
        assert new_node1['right'].size() == 2
        assert new_node1['right'].dim() == 2
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
    def test_svd_paramedge_rank_inplace(self, setup):
        net, edge, node1 = setup
        edge = edge.parameterize()
        
        assert isinstance(edge, tk.ParamEdge)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        new_node1, new_node2 = node1.svd_(axis='right', rank=2)
        
        assert tk.utils.erase_enum(new_node1.name) == \
            tk.utils.erase_enum(edge.node1.name)
        assert tk.utils.erase_enum(new_node2.name) == \
            tk.utils.erase_enum(edge.node2.name)
        assert new_node1.axes_names == edge.node1.axes_names
        assert new_node1.axes_names == edge.node1.axes_names
        
        assert new_node1.shape == (3, 5, 2)
        assert new_node2.shape == (2, 5, 3)
        assert new_node1['right'].size() == 2
        assert new_node1['right'].dim() == 2
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
    def test_svd_paramedge_cum_percentage_inplace(self, setup):
        net, edge, node1 = setup
        edge = edge.parameterize()
        
        assert isinstance(edge, tk.ParamEdge)
        assert edge.size() == 3
        assert edge.dim() == 3
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        tensor1 = torch.eye(15, 3).reshape(3, 5, 3)
        edge.node1.tensor = tensor1
        
        tensor2 = torch.eye(3, 15).reshape(3, 5, 3)
        edge.node2.tensor = tensor2
        
        new_node1, new_node2 = node1.svd_(axis='right', cum_percentage=0.5)
        
        assert tk.utils.erase_enum(new_node1.name) == \
            tk.utils.erase_enum(edge.node1.name)
        assert tk.utils.erase_enum(new_node2.name) == \
            tk.utils.erase_enum(edge.node2.name)
        assert new_node1.axes_names == edge.node1.axes_names
        assert new_node1.axes_names == edge.node1.axes_names
        
        assert new_node1.shape == (3, 5, 2)
        assert new_node2.shape == (2, 5, 3)
        assert new_node1['right'].size() == 2
        assert new_node1['right'].dim() == 2
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
    def test_svd_result_from_operation(self, setup):
        net, edge, node1 = setup
        node2 = net['node2']
        
        stacked = tk.stack([node1, node2])
        unbinded = tk.unbind(stacked)
        
        new_node1, new_node2 = unbinded[0].svd('right', rank=2)
        
        assert new_node1.shape == (3, 5, 2)
        assert new_node2.shape == (2, 5, 3)
        assert new_node1['right'].size() == 2
        
        assert unbinded[0].shape == (3, 5, 3)
        assert unbinded[1].shape == (3, 5, 3)
        assert unbinded[0]['right'].size() == 3
        
        assert node1.shape == (3, 5, 3)
        assert node2.shape == (3, 5, 3)
        assert node1['right'].size() == 3
        
        assert node1['right'] == unbinded[0]['right']
        assert node1['right'] != new_node1['right']


class TestContract:
    
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
        node3 = node1.contract(2)  # We could use 2 or 'right'
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
        node3 = node1.contract(2)
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
        node2 = node1.contract(2)
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
        node2 = node1.contract(2)
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
        
    def test_contract_edge_in_place(self):
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
        node3 = node1.contract_(2)
        assert node3['left'] != node1['left']
        assert node3['input_0'] != node1['input']
        assert node3['right'] != node2['right']
        assert node3['input_1'] != node2['input']
        
        assert len(net.nodes) == 1
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
    def test_contract_paramedge_in_place(self):
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
        node3 = node1.contract_(2)
        assert node3['left'] != node1['left']
        assert node3['input_0'] != node1['input']
        assert node3['right'] != node2['right']
        assert node3['input_1'] != node2['input']
        
        assert len(net.nodes) == 1
        assert len(net.leaf_nodes) == 1
        assert len(net.non_leaf_nodes) == 0
        
        assert net.edges == node3.edges
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        # Compute gradient
        node3.sum().backward()
        assert node1[2].grad != (None, None)
        assert node1[2].grad != (torch.zeros(1), torch.zeros(1))
        assert node2[0].grad != (None, None)
        assert node2[0].grad != (torch.zeros(1), torch.zeros(1))
        
        # Now node1 and node2 are not in the network any more
        net.zero_grad()
        assert node1[2].grad != (torch.zeros(1), torch.zeros(1))
        assert node2[0].grad != (torch.zeros(1), torch.zeros(1))


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
        
    def test_contract_with_contract_between(self):
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
        node3 = node1.contract_between(node2)
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
        node4 = node1.contract_between(node2)
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
        node1 = tk.Node(shape=(2, 10, 15, 5),
                        axes_names=('input', 'batch', 'my_batch', 'other_batch'),
                        name='node1',
                        init_method='randn',
                        network=net)
        node2 = tk.Node(shape=(2, 10, 20, 5),
                        axes_names=('input', 'batch', 'your_batch', 'other_batch'),
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
        assert node3.shape == (10, 5, 15, 20)
        assert node3.axes_names == ['batch', 'other_batch',
                                    'my_batch', 'your_batch']
        assert node3.edges == [node1['batch'], node1['other_batch'],
                               node1['my_batch'], node2['your_batch']]
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
        
    def test_contract_nodes_in_place(self):
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
        node3 = tk.Node(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node3',
                        init_method='randn',
                        network=net)
        node1['right'] ^ node2['left']
        node2['right'] ^ node3['left']
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 3
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        # Contract nodes in_place
        node4 = node1.contract_between_(node2)
        assert isinstance(node4, tk.Node)
        assert node4.shape == (2, 5, 5, 2)
        assert node4.axes_names == ['left', 'input_0', 'input_1', 'right']
        
        assert node4['left'] != node1['left']
        assert node4['input_0'] != node1['input']
        assert node4['input_1'] != node2['input']
        assert node4['right'] != node2['right']
        
        assert node4['right'] == node3['left']
        
        assert node4.network == net
        assert node1.network != net
        assert node2.network != net
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()

        # Repeat contraction -> Now node1 and node2 are
        # disconnected and are not in the same network
        with pytest.raises(ValueError):
            node4 = node1.contract_between_(node2)
            
    def test_contract_paramnodes_in_place(self):
        net = tk.TensorNetwork()
        node1 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node1',
                             init_method='randn',
                             param_edges=True,
                             network=net)
        node2 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node2',
                             init_method='randn',
                             param_edges=True,
                             network=net)
        node3 = tk.ParamNode(shape=(2, 5, 2),
                             axes_names=('left', 'input', 'right'),
                             name='node3',
                             init_method='randn',
                             param_edges=True,
                             network=net)
        node1['right'] ^ node2['left']
        node2['right'] ^ node3['left']
        
        assert len(net.nodes) == 3
        assert len(net.leaf_nodes) == 3
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()
        
        # Contract nodes in_place
        node4 = node1.contract_between_(node2)
        
        # Careful! even in in_place mode, it returns a Node,
        # not a ParamNode
        assert isinstance(node4, tk.Node)
        
        # We can parameterize the node later
        node4 = node4.parameterize()
        assert isinstance(node4, tk.ParamNode)
        
        assert node4.shape == (2, 5, 5, 2)
        assert node4.axes_names == ['left', 'input_0', 'input_1', 'right']
        
        assert node4['left'] != node1['left']
        assert node4['input_0'] != node1['input']
        assert node4['input_1'] != node2['input']
        assert node4['right'] != node2['right']
        
        assert node4['right'] == node3['left']
        
        assert net.edges == node3.edges[1:] + node4.edges[:-1]
        
        assert node4.network == net
        assert node1.network != net
        assert node2.network != net
        
        assert len(net.nodes) == 2
        assert len(net.leaf_nodes) == 2
        assert len(net.non_leaf_nodes) == 0
        
        assert node1.successors == dict()
        assert node2.successors == dict()

        # Repeat contraction -> Now node1 and node2 are
        # disconnected and are not in the same network
        with pytest.raises(ValueError):
            node4 = node1.contract_between_(node2)


class TestStackUnbind:
    
    @pytest.fixture
    def setup(self):
        net = tk.TensorNetwork()
        nodes = []
        for _ in range(10):
            node = tk.Node(shape=(2, 5, 2),
                           axes_names=('left', 'input', 'right'),
                           network=net,
                           init_method='randn')
            nodes.append(node)
            
        for i in range(9):
            nodes[i]['right'] ^ nodes[i + 1]['left']
        nodes[9]['right'] ^ nodes[0]['left']
        
        return net, nodes
    
    def test_stack_all_leaf_all_non_param_automemory_unbind(self, setup):
        net, nodes = setup
        
        # Automemory: yes, Unbind mode: yes
        # ---------------------------------
        net.automemory = True
        net.unbind_mode = True
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
    def test_stack_all_leaf_all_non_param_unbind(self, setup):
        net, nodes = setup
        
        # Automemory: no, Unbind mode: yes
        # ---------------------------------
        net.automemory = False
        net.unbind_mode = True
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
    def test_stack_all_leaf_all_non_param_automemory(self, setup):
        net, nodes = setup
        
        # Automemory: yes, Unbind mode: no
        # ---------------------------------
        net.automemory = True
        net.unbind_mode = False
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
    def test_stack_all_leaf_all_non_param(self, setup):
        net, nodes = setup
        
        # Automemory: no, Unbind mode: no
        # ---------------------------------
        net.automemory = False
        net.unbind_mode = False
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
    @pytest.fixture
    def setup_param(self):
        net = tk.TensorNetwork()
        nodes = []
        for _ in range(10):
            node = tk.ParamNode(shape=(2, 5, 2),
                                axes_names=('left', 'input', 'right'),
                                network=net,
                                param_edges=True,
                                init_method='randn')
            nodes.append(node)
            
        for i in range(9):
            nodes[i]['right'] ^ nodes[i + 1]['left']
        nodes[9]['right'] ^ nodes[0]['left']
        
        return net, nodes
            
    def test_stack_all_leaf_all_param_automemory_unbind(self, setup_param):
        net, nodes = setup_param
        
        # Automemory: yes, Unbind mode: yes
        # ---------------------------------
        net.automemory = True
        net.unbind_mode = True
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
    def test_stack_all_leaf_all_param_unbind(self, setup_param):
        net, nodes = setup_param
        
        # Automemory: no, Unbind mode: yes
        # ---------------------------------
        net.automemory = False
        net.unbind_mode = True
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
    def test_stack_all_leaf_all_param_automemory(self, setup_param):
        net, nodes = setup_param
        
        # Automemory: yes, Unbind mode: no
        # ---------------------------------
        net.automemory = True
        net.unbind_mode = False
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
    def test_stack_all_leaf_all_param(self, setup_param):
        net, nodes = setup_param
        
        # Automemory: no, Unbind mode: no
        # ---------------------------------
        net.automemory = False
        net.unbind_mode = False
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for node in nodes:
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
    @pytest.fixture
    def setup_diff_shapes(self):
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for _ in range(10):
            shape = torch.randint(low=2, high=5, size=(3,)).tolist()
            node = tk.ParamNode(shape=shape,
                                axes_names=('left', 'input', 'right'),
                                network=net,
                                param_edges=True,
                                init_method='randn')
            nodes.append(node)
            input_edges.append(node['input'])
            
        shapes = []
        for i in range(9):
            nodes[i]['right'].change_size(nodes[i + 1]['left'].size())
            nodes[i]['right'].change_dim(nodes[i + 1]['left'].dim())
            shapes.append(nodes[i].shape)
            nodes[i]['right'] ^ nodes[i + 1]['left']
            
        nodes[9]['right'].change_size(nodes[0]['left'].size())
        nodes[9]['right'].change_dim(nodes[0]['left'].dim())
        shapes.append(nodes[9].shape)
        nodes[9]['right'] ^ nodes[0]['left']
        
        net.set_data_nodes(input_edges, 1)
        
        return net, nodes, shapes
            
    def test_stack_diff_shapes_all_leaf_all_param_automemory_unbind(self, setup_diff_shapes):
        net, nodes, shapes = setup_diff_shapes
        
        # Automemory: yes, Unbind mode: yes
        # ---------------------------------
        net.automemory = True
        net.unbind_mode = True
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for i, node in enumerate(nodes):
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
            assert node.shape == shapes[i]
            for j in range(node.rank):
                assert shapes[i][j] <= stack.shape[j]
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Re-unbind
        reunbinded = tk.unbind(restack)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for i, node in enumerate(nodes):
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
            assert node.shape == shapes[i]
            for j in range(node.rank):
                assert shapes[i][j] <= stack.shape[j]
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        reunbinded = tk.unbind(restack)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
    def test_stack_diff_shapes_all_leaf_all_param_unbind(self, setup_diff_shapes):
        net, nodes, shapes = setup_diff_shapes
        
        # Automemory: no, Unbind mode: yes
        # ---------------------------------
        net.automemory = False
        net.unbind_mode = True
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for i, node in enumerate(nodes):
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
            assert node.shape == shapes[i]
            for j in range(node.rank):
                assert shapes[i][j] <= stack.shape[j]
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Re-unbind
        reunbinded = tk.unbind(restack)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for i, node in enumerate(nodes):
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
            assert node.shape == shapes[i]
            for j in range(node.rank):
                assert shapes[i][j] <= stack.shape[j]
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] == restack.name
        assert restack._tensor_info['node_ref'] is None
        
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
        reunbinded = tk.unbind(restack)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
    def test_stack_diff_shapes_all_leaf_all_param_automemory(self, setup_diff_shapes):
        net, nodes, shapes = setup_diff_shapes
        
        # Automemory: yes, Unbind mode: no
        # ---------------------------------
        net.automemory = True
        net.unbind_mode = False
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for i, node in enumerate(nodes):
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
            assert node.shape == shapes[i]
            for j in range(node.rank):
                assert shapes[i][j] <= stack.shape[j]
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Re-unbind
        reunbinded = tk.unbind(restack)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for i, node in enumerate(nodes):
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
            assert node.shape == shapes[i]
            for j in range(node.rank):
                assert shapes[i][j] <= stack.shape[j]
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        reunbinded = tk.unbind(restack)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
    def test_stack_diff_shapes_all_leaf_all_param(self, setup_diff_shapes):
        net, nodes, shapes = setup_diff_shapes
        
        # Automemory: no, Unbind mode: no
        # ---------------------------------
        net.automemory = False
        net.unbind_mode = False
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for i, node in enumerate(nodes):
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
            assert node.shape == shapes[i]
            for j in range(node.rank):
                assert shapes[i][j] <= stack.shape[j]
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        # Re-stack
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            # These are non-leaf nodes, so memory is not optimized
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Re-unbind
        reunbinded = tk.unbind(restack)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
        
        for i, node in enumerate(nodes):
            assert node._tensor_info['address'] == node.name
            assert node._tensor_info['node_ref'] is None
            
            assert node.shape == shapes[i]
            for j in range(node.rank):
                assert shapes[i][j] <= stack.shape[j]
        
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        restack = tk.stack(unbinded)
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        reunbinded = tk.unbind(restack)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
    def test_stack_irregular_all_leaf_all_param_automemory(self, setup_diff_shapes):
        net, nodes, shapes = setup_diff_shapes
        
        # Automemory: yes, Unbind mode: no
        # ---------------------------------
        net.automemory = True
        net.unbind_mode = False
        # It only has sense to study the index mode case
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        # Re-stack
        restack = tk.stack(unbinded[::2])
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        assert restack._tensor_info['stack_idx'] == slice(0, 9, 2)
        assert restack._tensor_info['index'] == slice(0, 9, 2)
            
        # Re-unbind
        reunbinded = tk.unbind(restack)
        new_index = range(0, 10, 2)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[new_index[i]]
            assert torch.equal(node.tensor, nodes[new_index[i]].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Re-stack all
        restack_all = tk.stack(unbinded[1::2] + reunbinded)
        new_index = list(range(1, 10, 2)) + list(range(0, 10, 2))
        assert isinstance(restack_all, tk.StackNode)
        assert restack_all.axes_names == ['stack', 'left', 'input', 'right']
        assert restack_all._tensor_info['address'] is None
        assert restack_all._tensor_info['node_ref'] == stack
        assert restack_all._tensor_info['stack_idx'] == new_index
        assert restack_all._tensor_info['index'] == new_index
        
        # Re-unbind all
        reunbinded_all = tk.unbind(restack_all)
        for i, node in enumerate(reunbinded_all):
            assert node.shape == shapes[new_index[i]]
            assert torch.equal(node.tensor, nodes[new_index[i]].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.ParamStackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
            
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        restack = tk.stack(unbinded[::2])
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        assert restack._tensor_info['stack_idx'] == slice(0, 9, 2)
        assert restack._tensor_info['index'] == slice(0, 9, 2)
            
        reunbinded = tk.unbind(restack)
        new_index = range(0, 10, 2)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[new_index[i]]
            assert torch.equal(node.tensor, nodes[new_index[i]].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        restack_all = tk.stack(unbinded[1::2] + reunbinded)
        new_index = list(range(1, 10, 2)) + list(range(0, 10, 2))
        assert isinstance(restack_all, tk.StackNode)
        assert restack_all.axes_names == ['stack', 'left', 'input', 'right']
        assert restack_all._tensor_info['address'] is None
        assert restack_all._tensor_info['node_ref'] == stack
        assert restack_all._tensor_info['stack_idx'] == new_index
        assert restack_all._tensor_info['index'] == new_index
        
        reunbinded_all = tk.unbind(restack_all)
        for i, node in enumerate(reunbinded_all):
            assert node.shape == shapes[new_index[i]]
            assert torch.equal(node.tensor, nodes[new_index[i]].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack      
            
    def test_stack_irregular_all_leaf_all_param(self, setup_diff_shapes):
        net, nodes, shapes = setup_diff_shapes
        
        # Automemory: no, Unbind mode: no
        # ---------------------------------
        net.automemory = False
        net.unbind_mode = False
        # It only has sense to study the index mode case
        
        # Stack
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
            
        # Unbind
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        # Re-stack
        restack = tk.stack(unbinded[::2])
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        assert restack._tensor_info['stack_idx'] == slice(0, 9, 2)
        assert restack._tensor_info['index'] == slice(0, 9, 2)
            
        # Re-unbind
        reunbinded = tk.unbind(restack)
        new_index = range(0, 10, 2)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[new_index[i]]
            assert torch.equal(node.tensor, nodes[new_index[i]].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Re-stack all
        restack_all = tk.stack(unbinded[1::2] + reunbinded)
        new_index = list(range(1, 10, 2)) + list(range(0, 10, 2))
        assert isinstance(restack_all, tk.StackNode)
        assert restack_all.axes_names == ['stack', 'left', 'input', 'right']
        assert restack_all._tensor_info['address'] is None
        assert restack_all._tensor_info['node_ref'] == stack
        assert restack_all._tensor_info['stack_idx'] == new_index
        assert restack_all._tensor_info['index'] == new_index
        
        # Re-unbind all
        reunbinded_all = tk.unbind(restack_all)
        for i, node in enumerate(reunbinded_all):
            assert node.shape == shapes[new_index[i]]
            assert torch.equal(node.tensor, nodes[new_index[i]].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        # Repeat operations
        stack = tk.stack(nodes)
        assert isinstance(stack, tk.StackNode)
        assert stack.axes_names == ['stack', 'left', 'input', 'right']
        assert stack._tensor_info['address'] == stack.name
            
        unbinded = tk.unbind(stack)
        for i, node in enumerate(unbinded):
            assert node.shape == shapes[i]
            assert torch.equal(node.tensor, nodes[i].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
        
        restack = tk.stack(unbinded[::2])
        assert isinstance(restack, tk.StackNode)
        assert restack.axes_names == ['stack', 'left', 'input', 'right']
        assert restack._tensor_info['address'] is None
        assert restack._tensor_info['node_ref'] == stack
        assert restack._tensor_info['stack_idx'] == slice(0, 9, 2)
        assert restack._tensor_info['index'] == slice(0, 9, 2)
            
        reunbinded = tk.unbind(restack)
        new_index = range(0, 10, 2)
        for i, node in enumerate(reunbinded):
            assert node.shape == shapes[new_index[i]]
            assert torch.equal(node.tensor, nodes[new_index[i]].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
        restack_all = tk.stack(unbinded[1::2] + reunbinded)
        new_index = list(range(1, 10, 2)) + list(range(0, 10, 2))
        assert isinstance(restack_all, tk.StackNode)
        assert restack_all.axes_names == ['stack', 'left', 'input', 'right']
        assert restack_all._tensor_info['address'] is None
        assert restack_all._tensor_info['node_ref'] == stack
        assert restack_all._tensor_info['stack_idx'] == new_index
        assert restack_all._tensor_info['index'] == new_index
        
        reunbinded_all = tk.unbind(restack_all)
        for i, node in enumerate(reunbinded_all):
            assert node.shape == shapes[new_index[i]]
            assert torch.equal(node.tensor, nodes[new_index[i]].tensor)
            assert node._tensor_info['address'] is None
            assert node._tensor_info['node_ref'] == stack
            
    def test_error_stack_stacks(self):
        net = tk.TensorNetwork()
        nodes = []
        for _ in range(5):
            node = tk.Node(shape=(3, 2),
                           axes_names=('input', 'output'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes.append(node)
            
        stack_node = tk.stack(nodes)
        
        # Cannot stack stacks
        with pytest.raises(TypeError):
            stack_node2 = tk.stack([stack_node])
            
    def test_error_stack_empty_list(self):
        with pytest.raises(ValueError):
            stack_node = tk.stack([])
            
    def test_connect_stacks(self):
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
        
        net.set_data_nodes(input_edges, 1)
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)
        
        stack_node = tk.stack(nodes)
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes])
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes])
        stackedge1 = stack_node['input_0'] ^ stack_input_0['feature']
        stackedge2 = stack_node['input_1'] ^ stack_input_1['feature']
        
        assert isinstance(stackedge1, tk.StackEdge)
        assert isinstance(stackedge2, tk.StackEdge)
            
    def test_connect_stack_with_no_stack(self):
        net = tk.TensorNetwork()
        nodes = []
        for _ in range(5):
            node = tk.Node(shape=(3, 2),
                           axes_names=('input', 'output'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes.append(node)
        
        stack_node = tk.stack(nodes)
        
        node = tk.Node(shape=(3,),
                       axes_names=('feature',),
                       name='node',
                       network=net,
                       init_method='randn')
        
        # Cannot connect a (Param)StackNode with a (Param)Node
        with pytest.raises(TypeError):
            stack_node['input'] ^ node['feature']
            
    def test_connect_disconnected_stacks(self):
        net = tk.TensorNetwork()
        
        nodes1 = []
        for _ in range(5):
            node = tk.Node(shape=(3, 2),
                           axes_names=('left', 'right'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes1.append(node)
            
        nodes2 = []
        for _ in range(5):
            node = tk.Node(shape=(2, 4),
                           axes_names=('left', 'right'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes2.append(node)
        
        stack_node1 = tk.stack(nodes1)
        stack_node2 = tk.stack(nodes2)
        
        # Cannot connect a (Param)StackEdges that were
        # not previously connected (outside the stacks)
        with pytest.raises(ValueError):
            stack_node1['right'] ^ stack_node2['left']
            
        for i in range(5):
            nodes1[i]['right'] ^ nodes2[i]['left']
        
        # We still cannot connect these stacks because we created
        # StackEdges that don't "know" that the nodes' edges are now connected
        with pytest.raises(ValueError):
            stack_node1['right'] ^ stack_node2['left']
            
        # We have to delete non-leaf nodes, because otherwise stack operations
        # will be repeated, using the disconnected edges
        net.delete_non_leaf()
        
        # Now we can connect them
        stack_node1 = tk.stack(nodes1)
        stack_node2 = tk.stack(nodes2)
        stack_node1['right'] ^ stack_node2['left']
    
    def test_einsum_with_stacks(self):
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
        
        net.set_data_nodes(input_edges, 1)
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)

        stack_node = tk.stack(nodes)
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes])
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes])
        stack_node['input_0'] ^ stack_input_0['feature']
        stack_node['input_1'] ^ stack_input_1['feature']

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        unbinded_nodes = tk.unbind(stack_result)
        assert len(unbinded_nodes) == 5
        assert unbinded_nodes[0].shape == (10, 2)

        second_stack = tk.stack(unbinded_nodes[0::2])
        assert second_stack.shape == (3, 10, 2)

        # Repeat operations second time
        stack_node = tk.stack(nodes)
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes])
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes])
        stack_node['input_0'] ^ stack_input_0['feature']
        stack_node['input_1'] ^ stack_input_1['feature']

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        nodes = tk.unbind(stack_result)
        assert len(nodes) == 5
        assert nodes[0].shape == (10, 2)

        second_stack = tk.stack(unbinded_nodes[0::2])
        assert second_stack.shape == (3, 10, 2)

    def test_einsum_with_paramstacks(self):
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for _ in range(5):
            node = tk.ParamNode(shape=(3, 3, 2),
                                axes_names=('input', 'input', 'output'),
                                name='node',
                                network=net,
                                param_edges=True,
                                init_method='randn')
            nodes.append(node)
            input_edges += [node['input_0'], node['input_1']]
        net.set_data_nodes(input_edges, 1)
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)

        net._contracting = True
        stack_node = tk.stack(nodes)
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes])
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes])
        stack_node['input_0'] ^ stack_input_0['feature']
        stack_node['input_1'] ^ stack_input_1['feature']

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        unbinded_nodes = tk.unbind(stack_result)
        assert len(unbinded_nodes) == 5
        assert unbinded_nodes[0].shape == (10, 2)

        second_stack = tk.stack(unbinded_nodes[0::2])
        assert second_stack.shape == (3, 10, 2)

        # Repeat operations second time
        stack_node = tk.stack(nodes)
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes])
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes])
        stack_node['input_0'] ^ stack_input_0['feature']
        stack_node['input_1'] ^ stack_input_1['feature']

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        nodes = tk.unbind(stack_result)
        assert len(nodes) == 5
        assert nodes[0].shape == (10, 2)

        second_stack = tk.stack(unbinded_nodes[0::2])
        assert second_stack.shape == (3, 10, 2)
        
    def test_einsum_autoconnect_stacks(self):
        # If stack edges are not connected, but they should,
        # we connect them in einsum
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for i in range(5):
            node = tk.Node(shape=(3, 3, 2),
                           axes_names=('input', 'input', 'output'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes.append(node)
            input_edges += [node['input_0'], node['input_1']]
        net.set_data_nodes(input_edges, 1)
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)

        stack_node = tk.stack(nodes)
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes])
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes])
        # No need to reconnect stacks if we are using einsum

        stack_result = tk.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
        assert stack_result.shape == (5, 10, 2)

        nodes = tk.unbind(stack_result)
        assert len(nodes) == 5
        assert nodes[0].shape == (10, 2)
        
        # Error is raised if the nodes' edges were not previously connected
        net = tk.TensorNetwork()
        
        nodes1 = []
        for _ in range(5):
            node = tk.Node(shape=(3, 2),
                           axes_names=('left', 'right'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes1.append(node)
            
        nodes2 = []
        for _ in range(5):
            node = tk.Node(shape=(2, 4),
                           axes_names=('left', 'right'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes2.append(node)
        
        stack_node1 = tk.stack(nodes1)
        stack_node2 = tk.stack(nodes2)
        
        with pytest.raises(ValueError):
            stack_result = tk.einsum('slr,srj->slj', stack_node1, stack_node2)
        
    def test_einsum_stack_and_no_stack(self):
        net = tk.TensorNetwork()
        nodes = []
        for _ in range(5):
            node = tk.Node(shape=(3, 2),
                           axes_names=('input', 'output'),
                           name='node',
                           network=net,
                           init_method='randn')
            nodes.append(node)
        
        stack_node = tk.stack(nodes)
        
        node = tk.Node(shape=(3,),
                       axes_names=('feature',),
                       name='node',
                       network=net,
                       init_method='randn')
        
        # Cannot operate (Param)StackNodes with (Param)Nodes in einsum. In fact,
        # einsum operands have to be connected if they share a subscript, but
        # we cannot connect StackNodes and Nodes, so this is avoided even before
        # taking care of it in einsum
        with pytest.raises(ValueError):
            stack_result = tk.einsum('sio,i->so', stack_node, node)
            
    def test_stack_change_dims(self):
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for _ in range(5):
            node = tk.ParamNode(shape=(3, 3, 2),
                                axes_names=('input', 'input', 'output'),
                                name='node',
                                network=net,
                                param_edges=True,
                                init_method='randn')
            nodes.append(node)
            input_edges += [node['input_0'], node['input_1']]
        net.set_data_nodes(input_edges, 1)
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)

        stack_node = tk.stack(nodes)
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes])
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes])
        
        # We can change dimension of connected edges, since the ParamStackEdge
        # has not its own dimension. ParamStackEdge's matrix is a stack of
        # matrices from all the ParamEdges
        stack_node.edges_dict['input_0'][0].change_dim(1)
        assert stack_node.edges_dict['input_0'][0].dim() == 1
        stack_node['input_0'] ^ stack_input_0['feature']
        
        # We can change dim even after connecting
        stack_node['input_1'] ^ stack_input_1['feature']
        stack_node.edges_dict['input_1'][0].change_dim(1)
        assert stack_node.edges_dict['input_0'][0].dim() == 1
        
    def test_stack_change_params(self):
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for _ in range(5):
            node = tk.ParamNode(shape=(3, 3, 2),
                                axes_names=('input', 'input', 'output'),
                                name='node',
                                network=net,
                                param_edges=True,
                                init_method='randn')
            nodes.append(node)
            input_edges += [node['input_0'], node['input_1']]
        net.set_data_nodes(input_edges, 1)
        data = torch.randn(2 * 5, 10, 3)
        net._add_data(data)

        stack_node = tk.stack(nodes)
        stack_input_0 = tk.stack([node.neighbours('input_0') for node in nodes])
        stack_input_1 = tk.stack([node.neighbours('input_1') for node in nodes])
        
        # If edges have not the same parameters in an axis,
        # but they have the same dimension, the parameters
        # are changed to be shared among all edges
        stack_node.edges_dict['input_1'][0].set_parameters(shift=-1., slope=30.)
        assert stack_node.edges_dict['input_1'][0].shift == -1.
        assert stack_node.edges_dict['input_1'][0].slope == 30.
        assert stack_node.edges_dict['input_1'][1].shift == -0.5
        assert stack_node.edges_dict['input_1'][1].slope == 20.

        # Parameters of each edge remain the same even after connecting
        stack_node['input_1'] ^ stack_input_1['feature']
        assert stack_node.edges_dict['input_1'][0].shift == -1.
        assert stack_node.edges_dict['input_1'][0].slope == 30.
        assert stack_node.edges_dict['input_1'][1].shift == -0.5
        assert stack_node.edges_dict['input_1'][1].slope == 20.
        
    def test_unbind_no_stack(self):
        node = tk.Node(shape=(10, 2),
                       axes_names=('batch', 'output'),
                       init_method='randn')
        
        # Cannot unbind a node that is not a (Param)StackNode
        with pytest.raises(TypeError):
            result = tk.unbind(node)


class TestEinsum:
    
    @pytest.fixture
    def setup(self):
        net = tk.TensorNetwork(name='net')
        node1 = tk.Node(shape=(2, 5, 3),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        init_method='randn')
        node2 = tk.Node(shape=(2, 5, 3),
                        axes_names=('left', 'input', 'right'),
                        network=net,
                        init_method='randn')
        node1['left'] ^ node2['left']
        node1['right'] ^ node2['right']
        
        node3 = tk.Node(shape=(2, 5, 3),
                        axes_names=('left', 'input', 'right'),
                        init_method='randn')
        
        return node1, node2, node3
    
    def test_einsum_error_no_nodes(self):
        # Cannot operate if no nodes are provided
        with pytest.raises(ValueError):
            result = tk.einsum('lir,ljr,abc->ijabc')
            
    def test_einsum_error_diff_networks(self, setup):
        node1, node2, node3 = setup
        
        # Cannot operate with nodes in different networks
        assert node1.network != node3.network
        with pytest.raises(ValueError):
            result = tk.einsum('lir,ljr,abc->ijabc', node1, node2, node3)
            
    def test_einsum_error_string_with_arrow(self, setup):
        node1, node2, _ = setup
        
        # String should have correct format (should have inputs, arrow, outputs)
        with pytest.raises(ValueError):
            result = tk.einsum('lir,ljr', node1, node2)
            
    def test_einsum_error_correct_amount(self, setup):
        node1, node2, _ = setup
        
        # String should have the same amount of input nodes
        # as provided as arguments
        with pytest.raises(ValueError):
            result = tk.einsum('lir,ljr,mn->ij', node1, node2)
            
    def test_einsum_error_output_subscripts_in_input(self, setup):
        node1, node2, _ = setup
        
        # Output string should only contain subscripts
        # present in the input strings
        with pytest.raises(ValueError):
            result = tk.einsum('lir,ljr->ijk', node1, node2)
            
    def test_einsum_error_batch_edges(self, setup):
        node1, node2, _ = setup
        
        # Input edges act as batch edges according to the input string,
        # but they are not batch edges, since their names don't contain
        # the word 'batch'
        assert not node1['input'].is_batch()
        assert not node2['input'].is_batch()
        with pytest.raises(ValueError):
            result = tk.einsum('lir,lir->i', node1, node2)
            
        # All edges have to be batch edges not to raise the error
        node1['input'].axis1.name = 'batch'
        assert node1['batch'].is_batch()
        assert not node2['input'].is_batch()
        with pytest.raises(ValueError):
            result = tk.einsum('lbr,lbr->b', node1, node2)
            
        node2['input'].axis1.name = 'batch'
        assert node1['batch'].is_batch()
        assert node2['batch'].is_batch()
        result = tk.einsum('lbr,lbr->b', node1, node2)
            
    def test_einsum_error_connected_edges(self, setup):
        node1, node2, node3 = setup
        node3.network = node1.network
        
        # Subscripts that do not appear in the output script are understood
        # to represent connected edges that have to be contracted. If these
        # edges are not the same in both nodes, an error is raised
        assert node1['left'] != node3['left']
        assert node1['right'] != node3['right']
        with pytest.raises(ValueError):
            result = tk.einsum('lir,ljr->ij', node1, node3)
            
        # There is no problem if those disconnected edges are (Param)StackEdges
        # whose lists of edges are connected
        stack1 = tk.stack([node1])
        stack2 = tk.stack([node2])
        assert stack1['left'] != stack2['left']
        assert stack1['right'] != stack2['right'] 
        result = tk.einsum('slir,sljr->sij', stack1, stack2)
        
    def test_einsum(self):
        net = tk.TensorNetwork(name='net')
        node = tk.Node(shape=(5, 5, 5, 5, 2),
                       axes_names=('input', 'input', 'input', 'input', 'output'),
                       network=net,
                       init_method='randn')
        net.set_data_nodes(node.edges[:-1], 1)
        data = torch.randn(4, 10, 5)
        net._add_data(data)

        out_node = tk.einsum('ijklm,bi,bj,bk,bl->bm', *([node] + list(net.data_nodes.values())))
        assert out_node.shape == (10, 2)

        net = tk.TensorNetwork(name='net')
        node = tk.ParamNode(shape=(5, 5, 5, 5, 2),
                            axes_names=('input', 'input', 'input', 'input', 'output'),
                            network=net,
                            param_edges=True,
                            init_method='randn')
        net.set_data_nodes(node.edges[:-1], 1)
        data = torch.randn(4, 10, 5)
        net._add_data(data)

        out_node = tk.einsum('ijklm,bi,bj,bk,bl->bm', *([node] + list(net.data_nodes.values())))
        assert out_node.shape == (10, 2)

    def test_stacked_einsum_mps(self):
        # MPS
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for i in range(11):
            node = tk.Node(shape=(2, 5, 2),
                           axes_names=('left', 'input', 'right'),
                           network=net,
                           init_method='randn')
            nodes.append(node)
            if i != 5:
                input_edges.append(node['input'])
            else:
                node.get_axis('input').name = 'output'
                
        for i in range(10):
            net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']
        net['node_10']['right'] ^ net['node_0']['left']
        
        net.set_data_nodes(input_edges, 1)
        data = torch.randn(10, 10, 5)
        net._add_data(data)
        
        result_list = tk.stacked_einsum('lir,bi->lbr', nodes[:5] + nodes[6:],
                                        list(net.data_nodes.values()))
        result_list = result_list[:5] + [nodes[5]] + result_list[5:]

        node1 = result_list[0]
        for i in range(1, 11):
            node1 @= result_list[i]
        assert node1.shape == (10, 5)

        node2 = result_list[0]
        for i in range(1, 5):
            node2 = tk.einsum('lbr,rbs->lbs', node2, result_list[i])
        node2 = tk.einsum('lbr,ros->lbos', node2, result_list[5])
        for i in range(6, 11):
            node2 = tk.einsum('lbor,rbs->lbos', node2, result_list[i])
        node2 = tk.einsum('lbol->bo', node2)
        
        assert node2.shape == (10, 5)
        assert torch.allclose(node1.tensor, node2.tensor)

    def test_stacked_einsum_param_mps(self):
        net = tk.TensorNetwork()
        nodes = []
        input_edges = []
        for i in range(11):
            node = tk.ParamNode(shape=(2, 5, 2),
                                axes_names=('left', 'input', 'right'),
                                network=net,
                                param_edges=True,
                                init_method='randn')
            nodes.append(node)
            if i != 5:
                input_edges.append(node['input'])
            else:
                node.get_axis('input').name = 'output'
                
        for i in range(10):
            net[f'paramnode_{i}']['right'] ^ net[f'paramnode_{i + 1}']['left']
        net['paramnode_10']['right'] ^ net['paramnode_0']['left']
        
        net.set_data_nodes(input_edges, 1)
        data = torch.randn(10, 10, 5)
        net._add_data(data)
        
        result_list = tk.stacked_einsum('lir,bi->lbr', nodes[:5] + nodes[6:],
                                        list(net.data_nodes.values()))
        result_list = result_list[:5] + [nodes[5]] + result_list[5:]

        node1 = result_list[0]
        for i in range(1, 11):
            node1 @= result_list[i]
        assert node1.shape == (10, 5)

        node2 = result_list[0]
        for i in range(1, 5):
            node2 = tk.einsum('lbr,rbs->lbs', node2, result_list[i])
        node2 = tk.einsum('lbr,ros->lbos', node2, result_list[5])
        for i in range(6, 11):
            node2 = tk.einsum('lbor,rbs->lbos', node2, result_list[i])
        node2 = tk.einsum('lbol->bo', node2)
        
        assert node2.shape == (10, 5)
        assert torch.allclose(node1.tensor, node2.tensor, atol=1e-7, rtol=1e-3)
        

class TestTNModels:
    
    def test_mps(self):
        
        class MPS(tk.TensorNetwork):
    
            def __init__(self, image_size):
                super().__init__(name='MPS')
                
                # Create TN
                input_nodes = []
                for _ in range(image_size[0] * image_size[1]):
                    node = tk.ParamNode(shape=(10, 3, 10),
                                        axes_names=('left', 'input', 'right'),
                                        name='input_node',
                                        network=self)
                    input_nodes.append(node)
                    
                for i in range(len(input_nodes) - 1):
                    input_nodes[i]['right'] ^ input_nodes[i + 1]['left']
                    
                output_node = tk.ParamNode(shape=(10, 10, 10),
                                        axes_names=('left', 'output', 'right'),
                                        name='output_node',
                                        network=self)
                output_node['right'] ^ input_nodes[0]['left']
                output_node['left'] ^ input_nodes[-1]['right']
                
                # Initialize nodes
                std = 1e-9
                for node in input_nodes:
                    tensor = torch.randn(node.shape) * std
                    random_eye = torch.randn(tensor.shape[0], tensor.shape[2]) * std
                    random_eye  = random_eye + torch.eye(tensor.shape[0], tensor.shape[2])
                    tensor[:, 0, :] = random_eye
                    
                    node.tensor = tensor
                        
                eye_tensor = torch.eye(output_node.shape[0], output_node.shape[2])\
                    .view([output_node.shape[0], 1, output_node.shape[2]])
                eye_tensor = eye_tensor.expand(output_node.shape)
                tensor = eye_tensor + std * torch.randn(output_node.shape)
                
                output_node.tensor = tensor
                
                self.input_nodes = input_nodes
                self.output_node = output_node
                
            def set_data_nodes(self) -> None:
                input_edges = []
                for node in self.input_nodes:
                    input_edges.append(node['input'])
                        
                super().set_data_nodes(input_edges, 1)
            
            def contract(self):
                stack_input = tk.stack(self.input_nodes)
                stack_data = tk.stack(list(self.data_nodes.values()))
                
                stack_input['input'] ^ stack_data['feature']
                stack_result = stack_input @ stack_data
                
                stack_result = tk.unbind(stack_result)
                
                result = stack_result[0]
                for node in stack_result[1:]:
                    result @= node
                result @= self.output_node
                
                return result
        
        image_size = (10, 10)
        mps = MPS(image_size=image_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mps = mps.to(device)

        # batch_size x height x width
        image = torch.randn(500, image_size[0], image_size[1])

        def embedding(image: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(image),
                                image,
                                1 - image], dim=1)
            
        image = embedding(image)
        image = image.to(device)
        image = image.view(500, 3, image_size[0] * image_size[1]).permute(2, 0, 1)
        
        mps.automemory = True
        mps.unbind_mode = True
        mps.trace(image)
        
        # Forward
        for _ in range(5):
            result = mps(image)
            
    def test_uniform_mps(self):
        
        class UMPS(tk.TensorNetwork):
    
            def __init__(self, image_size):
                super().__init__(name='UMPS')
                
                # Create TN
                input_nodes = []
                for _ in range(image_size[0] * image_size[1]):
                    node = tk.ParamNode(shape=(10, 3, 10),
                                        axes_names=('left', 'input', 'right'),
                                        name='input_node',
                                        network=self)
                    input_nodes.append(node)
                    
                for i in range(len(input_nodes) - 1):
                    input_nodes[i]['right'] ^ input_nodes[i + 1]['left']
                    
                output_node = tk.ParamNode(shape=(10, 10, 10),
                                        axes_names=('left', 'output', 'right'),
                                        name='output_node',
                                        network=self)
                output_node['right'] ^ input_nodes[0]['left']
                output_node['left'] ^ input_nodes[-1]['right']
                
                uniform_memory = tk.ParamNode(shape=(10, 3, 10),
                                              axes_names=('left', 'input', 'right'),
                                              name='virtual_uniform',
                                              network=self,
                                              virtual=True)
                
                # Initialize nodes
                std = 1e-9
                tensor = torch.randn(uniform_memory.shape) * std
                random_eye = torch.randn(tensor.shape[0], tensor.shape[2]) * std
                random_eye  = random_eye + torch.eye(tensor.shape[0], tensor.shape[2])
                tensor[:, 0, :] = random_eye
                
                uniform_memory._unrestricted_set_tensor(tensor)
                
                for node in input_nodes:
                    node._tensor_info['address'] = None
                    node._tensor_info['node_ref'] = uniform_memory
                    node._tensor_info['full'] = True
                    node._tensor_info['stack_idx'] = None
                    node._tensor_info['index'] = None
                        
                eye_tensor = torch.eye(output_node.shape[0], output_node.shape[2])\
                    .view([output_node.shape[0], 1, output_node.shape[2]])
                eye_tensor = eye_tensor.expand(output_node.shape)
                tensor = eye_tensor + std * torch.randn(output_node.shape)
                
                output_node.tensor = tensor
                
                self.input_nodes = input_nodes
                self.output_node = output_node
                self.uniform_memory = uniform_memory
                
            def set_data_nodes(self) -> None:
                input_edges = []
                for node in self.input_nodes:
                    input_edges.append(node['input'])
                        
                super().set_data_nodes(input_edges, 1)
            
            def contract(self):
                stack_input = tk.stack(self.input_nodes)
                stack_data = tk.stack(list(self.data_nodes.values()))
                
                stack_input['input'] ^ stack_data['feature']
                stack_result = stack_input @ stack_data
                
                stack_result = tk.unbind(stack_result)
                
                result = stack_result[0]
                for node in stack_result[1:]:
                    result @= node
                result @= self.output_node
                
                return result
        
        image_size = (10, 10)
        mps = UMPS(image_size=image_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mps = mps.to(device)

        # batch_size x height x width
        image = torch.randn(500, image_size[0], image_size[1])

        def embedding(image: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(image),
                                image,
                                1 - image], dim=1)
            
        image = embedding(image)
        image = image.to(device)
        image = image.view(500, 3, image_size[0] * image_size[1]).permute(2, 0, 1)
        
        mps.automemory = True
        mps.unbind_mode = True
        mps.trace(image)
        
        # Forward
        for _ in range(5):
            result = mps(image)
            
    def test_peps(self):
        
        class PEPS(tk.TensorNetwork):
    
            def __init__(self, image_size):
                super().__init__(name='PEPS')
                
                # Create TN
                input_nodes = []
                for i in range(image_size[0]):
                    aux_lst = []
                    for j in range(image_size[1]):
                        node = tk.ParamNode(shape=(2, 2, 2, 2, 3),
                                            axes_names=('left', 'right', 'up', 'down', 'input'),
                                            name=f'input_node_[{i},{j}]',
                                            network=self)
                        aux_lst.append(node)
                    input_nodes.append(aux_lst)
                    
                for i in range(len(input_nodes)):
                    for j in range(len(input_nodes[i])):
                        if i == (len(input_nodes) - 1):
                            input_nodes[i][j]['down'] ^ input_nodes[0][j]['up']
                        else:
                            input_nodes[i][j]['down'] ^ input_nodes[i + 1][j]['up']
                            
                        if j == (len(input_nodes[i]) - 1):
                            input_nodes[i][j]['right'] ^ input_nodes[i][0]['left']
                        else:
                            input_nodes[i][j]['right'] ^ input_nodes[i][j + 1]['left']
                        
                # j = len(input_nodes[0]) - 1
                # for i in range(len(input_nodes) - 1):
                #     input_nodes[i][j]['right'] ^ input_nodes[i][0]['left']
                        
                # i = len(input_nodes) - 1
                # for j in range(len(input_nodes[i]) - 1):
                #     input_nodes[i][j]['down'] ^ input_nodes[0][j]['up']
                
                # Initialize nodes
                std = 1e-9
                for lst in input_nodes:
                    for node in lst:
                        tensor = torch.randn(node.shape) * std
                        # random_eye = torch.randn(tensor.shape[0], tensor.shape[2]) * std
                        # random_eye  = random_eye + torch.eye(tensor.shape[0], tensor.shape[2])
                        # tensor[:, 0, :] = random_eye
                        
                        node.tensor = tensor
                
                self.input_nodes = input_nodes
                
            def set_data_nodes(self) -> None:
                input_edges = []
                for lst in self.input_nodes:
                    for node in lst:
                        input_edges.append(node['input'])
                        
                super().set_data_nodes(input_edges, 1)
            
            def contract(self):
                all_input_nodes = []
                for lst in self.input_nodes:
                    for node in lst:
                        all_input_nodes.append(node)
                      
                # Contract with data  
                stack_input = tk.stack(all_input_nodes)
                stack_data = tk.stack(list(self.data_nodes.values()))
                
                stack_input['input'] ^ stack_data['feature']
                stack_result = stack_input @ stack_data
                
                result = tk.unbind(stack_result)
                
                # Contract TN
                n_rows = len(self.input_nodes)
                n_cols = len(self.input_nodes[0])
                current_2_lines = [result[:n_cols], result[n_cols:(2 * n_cols)]]
                for i in range(n_rows):
                    if i > 0:
                        for j in range(n_cols):
                            if j == 0:
                                aux_result = current_2_lines[0][j] @ current_2_lines[0][j + 1]
                                new_node1, new_node2 = aux_result.split(node1_axes=['left_0', 'left_1', 'up_0', 'down_0'],
                                                                        node2_axes=['right_0', 'right_1', 'up_1', 'down_1'],
                                                                        rank=2)
                                new_node1.get_axis('splitted').name = 'right'
                                new_node2.get_axis('splitted').name = 'left'
                                current_2_lines[0][j] = new_node1
                                current_2_lines[0][j + 1] = new_node2
                            elif j < (n_cols - 1):
                                aux_result = current_2_lines[0][j] @ current_2_lines[0][j + 1]
                                new_node1, new_node2 = aux_result.split(node1_axes=['left', 'up_0', 'down_0'],
                                                                        node2_axes=['right_0', 'right_1', 'up_1', 'down_1'],
                                                                        rank=2)
                                new_node1.get_axis('splitted').name = 'right'
                                new_node2.get_axis('splitted').name = 'left'
                                current_2_lines[0][j] = new_node1
                                current_2_lines[0][j + 1] = new_node2
                            else:
                                aux_result = current_2_lines[0][j] @ current_2_lines[0][0]
                                new_node1, new_node2 = aux_result.split(node1_axes=['left', 'up_0', 'down_0'],
                                                                        node2_axes=['right', 'up_1', 'down_1'],
                                                                        rank=2)
                                new_node1.get_axis('splitted').name = 'right'
                                new_node2.get_axis('splitted').name = 'left'
                                current_2_lines[0][j] = new_node1
                                current_2_lines[0][0] = new_node2
                    if i < (n_rows - 1):
                        for j in range(n_cols):
                            current_2_lines[1][j] = current_2_lines[0][j] @ current_2_lines[1][j]
                            
                    current_2_lines[0] = current_2_lines[1]
                    current_2_lines[1] = result[(i + 1) * n_cols:((i + 2) * n_cols)]
                            
                result = current_2_lines[0][0]
                for node in current_2_lines[0][1:]:
                    result @= result
                    result @= node
                result @= result
                
                return result
        
        image_size = (3, 3)
        mps = PEPS(image_size=image_size)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mps = mps.to(device)

        # batch_size x height x width
        image = torch.randn(500, image_size[0], image_size[1])

        def embedding(image: torch.Tensor) -> torch.Tensor:
            return torch.stack([torch.ones_like(image),
                                image,
                                1 - image], dim=1)
            
        image = embedding(image)
        image = image.to(device)
        image = image.view(500, 3, image_size[0] * image_size[1]).permute(2, 0, 1)
        
        mps.automemory = True
        mps.unbind_mode = True
        mps.trace(image)
        
        # Forward
        for _ in range(5):
            result = mps(image)
        
        
        
        
    


def test_for_peps():
    node1 = tk.ParamNode(shape=(3, 4),
                         axes_names=('left', 'right'),
                         init_method='randn')
    node2 = tk.ParamNode(shape=(4, 5),
                         axes_names=('left', 'right'),
                         init_method='randn')
    node1['right'] ^ node2['left']
    
    node3 = tk.Node(shape=(3, 4),
                    axes_names=('left', 'right'),
                    network=node1.network,
                    leaf=False,
                    edges=[node1['left'], node1['right']],
                    node1_list=node1.is_node1())
    node4 = tk.Node(shape=(4, 5),
                    axes_names=('left', 'right'),
                    network=node1.network,
                    leaf=False,
                    edges=[node2['left'], node2['right']],
                    node1_list=node2.is_node1())
    node3._unrestricted_set_tensor(torch.randn(3, 2), True)
    node4._unrestricted_set_tensor(torch.randn(2, 5), True)
    # NOTE: I can use this to make non-leaf tensors with smaller
    # dimensions as the original nodes -> Very useful for PEPS
    
    assert node3.shape == (3, 2)
    assert node3['left'].size() == 3
    assert node3['right'].size() == 4
    
    assert node4.shape == (2, 5)
    assert node4['left'].size() == 4
    assert node4['right'].size() == 5
    
    print()
