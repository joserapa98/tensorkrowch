"""
Tests for network_components
"""

import pytest

import torch
import torch.nn as nn
import tentorch as tn


def test_init_node():
    node = tn.Node(shape=(2, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node')

    assert node.shape == node.tensor.shape
    assert node['left'] == node.edges[0]
    assert node['left'] == node[0]
    assert node.name == 'node'
    assert isinstance(node.edges[0], tn.Edge)


def test_init_param_node():
    node = tn.ParamNode(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node')

    assert node.shape == node.tensor.shape
    assert node['left'] == node.edges[0]
    assert node['left'] == node[0]
    assert node.name == 'node'
    assert isinstance(node.edges[0], tn.Edge)


def test_set_tensor():
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1')

    tensor = torch.randn(2, 5, 2)
    node2 = tn.Node(axes_names=('left', 'input', 'right'),
                    name='node2',
                    tensor=tensor)

    node1.set_tensor(tensor=tensor)
    assert torch.equal(node1.tensor, node2.tensor)

    tensor[0, 0, 0] = 1000
    assert node1.tensor[0, 0, 0] == 1000

    node1.unset_tensor()
    assert not torch.equal(node1.tensor, torch.empty(node1.shape))

    wrong_tensor = torch.randn(2, 5, 5)
    with pytest.raises(ValueError):
        node1.set_tensor(tensor=wrong_tensor)

    node1.set_tensor(init_method='randn', mean=1., std=2.)
    node2.tensor = node1.tensor
    assert torch.equal(node1.tensor, node2.tensor)

    node3 = tn.ParamNode(axes_names=('left', 'input', 'right'),
                         name='node3',
                         tensor=tensor)
    assert isinstance(node3.tensor, nn.Parameter)
    assert torch.equal(node3.tensor.data, tensor)

    param = nn.Parameter(tensor)
    param.mean().backward()
    assert node3.grad is None

    node3.set_tensor(param)
    assert node3.grad is not None
    assert torch.equal(node3.grad, node3.tensor.grad)


def test_parameterize():
    node = tn.Node(shape=(2, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node',
                   init_method='randn')
    node = node.parameterize()
    assert isinstance(node, tn.ParamNode)
    assert node['left'].node1 == node
    assert isinstance(node['left'], tn.Edge)

    node = node.parameterize(False)
    assert isinstance(node, tn.Node)
    assert node['left'].node1 == node
    assert isinstance(node['left'], tn.Edge)


def test_param_edges():
    node = tn.Node(shape=(2, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node',
                   param_edges=True,
                   init_method='randn')
    assert isinstance(node[0], tn.ParamEdge)
    assert node[0].dim() == node.shape[0]


def test_copy():
    node = tn.Node(shape=(2, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node',
                   init_method='randn')
    copy = node.copy()
    assert torch.equal(copy.tensor, node.tensor)
    for i in range(len(copy.axes)):
        if copy.axes[i].node1:
            assert copy.edges[i].node1 == copy
            assert node.edges[i].node1 == node
            assert copy.edges[i].node2 == node.edges[i].node2
        else:
            assert copy.edges[i].node2 == copy
            assert node.edges[i].node2 == node
            assert copy.edges[i].node1 == node.edges[i].node1


def test_connect():
    net = tn.TensorNetwork(name='net')
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1',
                    network=net)
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2',
                    param_edges=True)
    assert isinstance(node1[2], tn.Edge)
    assert isinstance(node2[0], tn.ParamEdge)
    new_edge = node1[2] ^ node2[0]
    assert isinstance(new_edge, tn.ParamEdge)
    assert node2.network == net

    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1',
                    param_edges=True)
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2',
                    network=net)
    assert isinstance(node1[2], tn.ParamEdge)
    assert isinstance(node2[0], tn.Edge)
    new_edge = node1[2] ^ node2[0]
    assert isinstance(new_edge, tn.ParamEdge)
    assert node1.network == net

    net1 = tn.TensorNetwork(name='net1')
    net2 = tn.TensorNetwork(name='net2')
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1',
                    network=net1)
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2',
                    network=net2)
    with pytest.raises(ValueError):
        node1[2] ^ node2[0]
    tn.connect(node1[2], node2[0], override_network=True)
    assert node1.network == net1
    assert node2.network == net1


def test_contract_between():
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1')
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2')
    node1[2] ^ node2[0]
    node3 = node1 @ node2
    assert node3['left'] == node1['left']
    assert node3['right'] == node2['right']
    if node1.network is not None:
        assert node3.network == node1.network
    elif node2.network is not None:
        assert node3.network == node1.network


def test_connect_different_sizes():
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


def test_connect_reassign():
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1')
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2')
    node1[2] ^ node2[0]
    node3 = node1 @ node2

    node4 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node4')
    edge = node3[3]
    node3[3] ^ node4[0]
    assert node3[3] == edge
    assert node2[2] == node4[0]

    node4[0] | node4[0]
    node3.reassign_edges(override=False)
    edge = node3[3]
    node3[3] ^ node4[0]
    assert node3[3] != edge


def test_tensor_network():
    net = tn.TensorNetwork(name='net')
    for i in range(4):
        _ = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node',
                    network=net)
    assert list(net.nodes.keys()) == [f'node_{i}' for i in range(4)]

    new_node = tn.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       name='node')
    new_node.network = net
    assert new_node.name == 'node_4'

    for i in range(4):
        net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
    assert len(net.edges) == 7

    net.remove_node(new_node)
    assert new_node.name not in net.nodes
    assert net['node_3']['right'].node2 == new_node

    new_node.network = net
    net.delete_node(new_node)
    assert new_node.name not in net.nodes
    assert net['node_3']['right'].node2 is None
    assert new_node['left'].node2 is None

    node = net['node_0']
    node.name = 'node_1'
    assert list(net.nodes.keys()) == [f'node_{i}' for i in range(4)]
    assert node.name == 'node_0'


# TODO: test matrix param edge
# TODO: test submodules tn
# TODO: test tn edges, set_data_nodes
# TODO: test operations node
# TODO: test grad
