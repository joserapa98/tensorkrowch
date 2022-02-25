"""
Tests for node_operations
"""

import pytest

import torch
import torch.nn as nn
import tentorch as tn


def test_einsum():
    net = tn.TensorNetwork(name='net')
    node = tn.Node(shape=(5, 5, 5, 5, 2),
                   axes_names=('input', 'input', 'input', 'input', 'output'),
                   network=net,
                   init_method='randn')
    net.set_data_nodes(node.edges[:-1], 10)
    data = torch.randn(10, 5, 4)
    net._add_data(data)

    out_node = tn.einsum('ijklm,bi,bj,bk,bl->bm', *([node] + list(net.data_nodes.values())))
    assert out_node.shape == (10, 2)
    with pytest.raises(ValueError):
        out_node = tn.einsum('ijklm,bi,bj,bk,bl->bm', *(list(net.data_nodes.values())))
    with pytest.raises(ValueError):
        out_node = tn.einsum('ijklm,bi,bj,bk,bl->b', *([node] + list(net.data_nodes.values())))
    with pytest.raises(ValueError):
        out_node = tn.einsum('ijklm,b,bj,bk,bl->bm', *([node] + list(net.data_nodes.values())))

    net = tn.TensorNetwork(name='net')
    node = tn.ParamNode(shape=(5, 5, 5, 5, 2),
                        axes_names=('input', 'input', 'input', 'input', 'output'),
                        network=net,
                        param_edges=True,
                        init_method='randn')
    net.set_data_nodes(node.edges[:-1], 10)
    data = torch.randn(10, 5, 4)
    net._add_data(data)

    out_node = tn.einsum('ijklm,bi,bj,bk,bl->bm', *([node] + list(net.data_nodes.values())))
    assert out_node.shape == (10, 2)
    with pytest.raises(ValueError):
        out_node = tn.einsum('ijklm,bi,bj,bk,bl->bm', *(list(net.data_nodes.values())))
    with pytest.raises(ValueError):
        out_node = tn.einsum('ijklm,bi,bj,bk,bl->b', *([node] + list(net.data_nodes.values())))
    with pytest.raises(ValueError):
        out_node = tn.einsum('ijklm,b,bj,bk,bl->bm', *([node] + list(net.data_nodes.values())))


def test_batched_contract_between():
    node1 = tn.Node(shape=(10, 2, 3),
                    axes_names=('batch', 'left', 'right'),
                    name='node1',
                    init_method='randn')
    node2 = tn.Node(shape=(10, 2, 3),
                    axes_names=('batch', 'left', 'right'),
                    name='node2',
                    init_method='randn')
    node1['left'] ^ node2['left']
    node1['right'] ^ node2['right']
    node3 = tn.batched_contract_between(node1, node2,
                                        node1['batch'],
                                        node2['batch'])
    assert node3.shape == (10,)

    node1 = tn.ParamNode(shape=(10, 2, 3),
                         axes_names=('batch', 'left', 'right'),
                         name='node1',
                         param_edges=True,
                         init_method='randn')
    node2 = tn.ParamNode(shape=(10, 2, 3),
                         axes_names=('batch', 'left', 'right'),
                         name='node2',
                         param_edges=True,
                         init_method='randn')
    node1['left'] ^ node2['left']
    node1['right'] ^ node2['right']
    node3 = tn.batched_contract_between(node1, node2,
                                        node1['batch'],
                                        node2['batch'])
    assert node3.shape == (10,)


def test_stack():
    net = tn.TensorNetwork()
    nodes = []
    input_edges = []
    for i in range(5):
        node = tn.Node(shape=(3, 3, 2),
                       axes_names=('input', 'input', 'output'),
                       name='node',
                       network=net,
                       init_method='randn')
        nodes.append(node)
        input_edges += [node['input_0'], node['input_1']]
    net.set_data_nodes(input_edges=input_edges,
                       batch_size=10)
    data = torch.randn(10, 3, 2*5)
    net._add_data(data)
    
    stack_node = tn.stack(nodes, name='stack_node')
    stack_input_0 = tn.stack([node.neighbours('input_0') for node in nodes],
                             name='stack_input_0')
    stack_input_1 = tn.stack([node.neighbours('input_1') for node in nodes],
                             name='stack_input_1')
    stack_node['input_0'] ^ stack_input_0['feature']
    stack_node['input_1'] ^ stack_input_1['feature']

    stack_result = tn.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
    assert stack_result.shape == (5, 10, 2)

    # Error 1
    net = tn.TensorNetwork()
    nodes = []
    input_edges = []
    for i in range(5):
        node = tn.Node(shape=(3, 3, 2),
                       axes_names=('input', 'input', 'output'),
                       name='node',
                       network=net,
                       init_method='randn')
        nodes.append(node)
        input_edges += [node['input_0'], node['input_1']]
    net.set_data_nodes(input_edges=input_edges,
                       batch_size=10)
    data = torch.randn(10, 3, 2 * 5)
    net._add_data(data)

    stack_node = tn.stack(nodes, name='stack_node')
    stack_input_0 = tn.stack([node.neighbours('input_0') for node in nodes],
                             name='stack_input_0')
    stack_input_1 = tn.stack([node.neighbours('input_1') for node in nodes],
                             name='stack_input_1')

    with pytest.raises(ValueError):
        stack_result = tn.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)

    # Error 2
    net = tn.TensorNetwork()
    nodes = []
    input_edges = []
    for i in range(5):
        node = tn.Node(shape=(3, 3, 2),
                       axes_names=('input', 'input', 'output'),
                       name='node',
                       network=net,
                       init_method='randn')
        nodes.append(node)
        input_edges += [node['input_0'], node['input_1']]
    net.set_data_nodes(input_edges=input_edges,
                       batch_size=10)
    data = torch.randn(10, 3, 2 * 5)
    net._add_data(data)
    net['data_0'].disconnect_edges()

    stack_node = tn.stack(nodes, name='stack_node')
    stack_input_0 = tn.stack([net['data_0']] + [node.neighbours('input_0') for node in nodes][1:],
                             name='stack_input_0')

    with pytest.raises(ValueError):
        stack_node['input_0'] ^ stack_input_0['feature']
