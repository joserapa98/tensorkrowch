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


def test_init_node():
    node = tn.Node(shape=(2, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node')

    assert node.shape == torch.Size((2, 5, 2))
    assert node.shape == node.tensor.shape
    assert node['left'] == node.edges[0]
    assert node['left'] == node[0]
    assert node.name == 'node'
    assert isinstance(node.edges[0], tn.Edge)


def test_init_param_node():
    node = tn.ParamNode(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')

    assert node.shape == torch.Size((2, 5, 2))
    assert node.shape == node.tensor.shape
    assert node['left'] == node.edges[0]
    assert node['left'] == node[0]
    assert node.name == 'node'
    assert isinstance(node.edges[0], tn.Edge)


def test_set_tensor():
    # Node with empty tensor
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1')

    # Node with tensor
    tensor = torch.randn(2, 5, 2)
    node2 = tn.Node(axes_names=('left', 'input', 'right'),
                    name='node2',
                    tensor=tensor)

    # Set tensor in node1
    node1.set_tensor(tensor=tensor)
    assert torch.equal(node1.tensor, node2.tensor)

    # Changing tensor changes node1's and node2's tensor
    tensor[0, 0, 0] = 1000
    assert node1.tensor[0, 0, 0] == 1000
    assert node2.tensor[0, 0, 0] == 1000

    # Unset tensor in node1
    node1.unset_tensor()
    assert not torch.equal(node1.tensor, tensor)
    assert node1.shape == tensor.shape

    # It's possible to set tensor with different shape if the
    # edges are dangling edges
    diff_tensor = torch.randn(2, 5, 5)
    node1.set_tensor(tensor=diff_tensor)
    assert node1.shape == torch.Size([2, 5, 5])

    # Initialize tensor of node1
    node1.set_tensor(init_method='randn', mean=1., std=2.)

    # Set node1's tensor as node2's tensor
    node2.tensor = node1.tensor
    assert torch.equal(node1.tensor, node2.tensor)

    # Changing node1's tensor changes node2's tensor
    node1.tensor[0, 0, 0] = 1000
    assert node2.tensor[0, 0, 0] == 1000

    # Create parametric node
    node3 = tn.ParamNode(axes_names=('left', 'input', 'right'),
                         name='node3',
                         tensor=tensor)
    assert isinstance(node3.tensor, nn.Parameter)
    assert torch.equal(node3.tensor.data, tensor)

    # Creating parameter from tensor does not affect tensor's grad
    param = nn.Parameter(tensor)
    param.mean().backward()
    assert node3.grad is None

    # Set nn.Parameter as node3's tensor
    node3.set_tensor(param)
    assert node3.grad is not None
    assert torch.equal(node3.grad, node3.tensor.grad)


def test_parameterize():
    net = tn.TensorNetwork(name='net_test')
    node = tn.Node(shape=(3, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node',
                   network=net,
                   init_method='randn')
    node = node.parameterize()
    assert isinstance(node, tn.ParamNode)
    assert node['left'].node1 == node
    assert isinstance(node['left'], tn.Edge)
    assert node.edges == net.edges

    node = node.parameterize(False)
    assert isinstance(node, tn.Node)
    assert node['left'].node1 == node
    assert isinstance(node['left'], tn.Edge)
    assert node.edges == net.edges

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


def test_param_edges():
    node = tn.Node(shape=(2, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node',
                   param_edges=True,
                   init_method='randn')
    for i, edge in enumerate(node.edges):
        assert isinstance(edge, tn.ParamEdge)
        assert edge.dim() == node.shape[i]


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


def test_permute():
    node = tn.Node(shape=(2, 5, 2),
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

    assert permuted_node[0]._nodes[permuted_node.is_node1('left')] == node[0]._nodes[node.is_node1('left')]
    assert permuted_node[1]._nodes[permuted_node.is_node1('left')] == node[2]._nodes[node.is_node1('left')]
    assert permuted_node[2]._nodes[permuted_node.is_node1('left')] == node[1]._nodes[node.is_node1('left')]

    assert torch.equal(permuted_node.tensor, node.tensor.permute(0, 2, 1))

    # Param
    node = tn.ParamNode(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name='node', init_method='randn')
    permuted_node = node.permute((0, 2, 1))

    assert permuted_node['left']._nodes[permuted_node.is_node1('left')] == \
           node['left']._nodes[node.is_node1('left')]
    assert permuted_node['input']._nodes[permuted_node.is_node1('left')] == \
           node['input']._nodes[node.is_node1('left')]
    assert permuted_node['right']._nodes[permuted_node.is_node1('left')] == \
           node['right']._nodes[node.is_node1('left')]

    assert permuted_node[0]._nodes[permuted_node.is_node1('left')] == node[0]._nodes[node.is_node1('left')]
    assert permuted_node[1]._nodes[permuted_node.is_node1('left')] == node[2]._nodes[node.is_node1('left')]
    assert permuted_node[2]._nodes[permuted_node.is_node1('left')] == node[1]._nodes[node.is_node1('left')]

    assert torch.equal(permuted_node.tensor, node.tensor.permute(0, 2, 1))


def test_param_edge():
    node = tn.Node(shape=(2, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node',
                   param_edges=True,
                   init_method='randn')
    param_edge = node[0]
    assert isinstance(param_edge, tn.ParamEdge)

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
