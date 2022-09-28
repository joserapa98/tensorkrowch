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


def test_tensor_network():
    net = tn.TensorNetwork(name='net')
    for i in range(4):
        _ = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name='node', network=net)
    assert list(net.nodes.keys()) == [f'node_{i}' for i in range(4)]

    new_node = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name='node')
    new_node.network = net
    assert new_node.name == 'node_4'

    assert len(net.edges) == 15
    for i in range(4):
        net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
    assert len(net.edges) == 7

    net._remove_node(new_node)
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


def test_tn_consecutive_contractions():
    net = tn.TensorNetwork(name='net')
    for i in range(4):
        _ = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name='node', network=net)

    for i in range(3):
        net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
    assert len(net.edges) == 6

    node = net['node_0']
    for i in range(1, 4):
        node @= net[f'node_{i}']
    assert len(node.edges) == 6
    assert node.shape == (2, 5, 5, 5, 5, 2)

    new_node = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name='node')
    new_node.network = net
    assert new_node.name == 'node_4'
    net['node_3'][2] ^ new_node[0]
    with pytest.raises(ValueError):
        node @= new_node


def test_tn_submodules():
    net = tn.TensorNetwork(name='net')
    for i in range(2):
        _ = tn.ParamNode(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), network=net, param_edges=True)
    for i in range(2):
        _ = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), network=net, param_edges=True)

    submodules = [None for _ in net.children()]
    assert len(submodules) == 12

    net['paramnode_0']['right'] ^ net['paramnode_1']['left']
    net['paramnode_1']['right'] ^ net['node_0']['left']
    net['node_0']['right'] ^ net['node_1']['left']
    submodules = [None for _ in net.children()]
    assert len(submodules) == 9


def test_tn_parameterize():
    net = tn.TensorNetwork(name='net')
    for i in range(2):
        _ = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), network=net)

    submodules = [None for _ in net.children()]
    assert len(submodules) == 0

    param_net = net.parameterize()
    submodules = [None for _ in net.children()]
    assert len(submodules) == 0
    submodules = [None for _ in param_net.children()]
    assert len(submodules) == 6

    param_net = net.parameterize(override=True)
    assert param_net == net
    submodules = [None for _ in net.children()]
    assert len(submodules) == 6

    net.parameterize(set_param=False, override=True)
    submodules = [None for _ in net.children()]
    assert len(submodules) == 0


def test_tn_data_nodes():
    net = tn.TensorNetwork(name='net')
    for i in range(4):
        _ = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name='node', network=net,
                    init_method='ones')
    for i in range(3):
        net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']

    assert len(net.data_nodes) == 0
    input_edges = []
    for i in range(4):
        input_edges.append(net[f'node_{i}']['input'])
    net.set_data_nodes(input_edges, [10])
    assert len(net.nodes) == 8
    assert len(net.data_nodes) == 4

    input_edges = []
    for i in range(3):
        input_edges.append(net[f'node_{i}']['input'])
    with pytest.raises(ValueError):
        net.set_data_nodes(input_edges, [10])

    net.unset_data_nodes()
    assert len(net.nodes) == 4
    assert len(net.data_nodes) == 0

    input_edges = []
    for i in range(2):
        input_edges.append(net[f'node_{i}']['input'])
    net.set_data_nodes(input_edges, [10])
    assert len(net.nodes) == 6
    assert len(net.data_nodes) == 2

    data = torch.randn(10, 5, 2)
    net._add_data(data.unbind(2))
    # TODO: revisar
    #assert torch.equal(net.data_nodes['data_0'].tensor, data[:, :, 0])
    #assert torch.equal(net.data_nodes['data_1'].tensor, data[:, :, 1])

    data = torch.randn(10, 5, 3)
    with pytest.raises(IndexError):
        net._add_data(data)


# TODO: funcion de copiar Tn entera
# TODO: test aplicar mismas contracciones varias veces, se reutilizan nodos, pero no se optimiza memoria
# TODO: test definir subclase de TN, y definir y usar forward, se reutilizan nodos y se optimiza memoria
