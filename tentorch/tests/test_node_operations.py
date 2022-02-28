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
    net.set_data_nodes(node.edges[:-1], [10])
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
    net.set_data_nodes(node.edges[:-1], [10])
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
                       batch_sizes=[10])
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

    nodes = tn.unbind(stack_result)
    assert len(nodes) == 5
    assert nodes[0].shape == (10, 2)

    # Param
    net = tn.TensorNetwork()
    nodes = []
    input_edges = []
    for i in range(5):
        node = tn.ParamNode(shape=(3, 3, 2),
                            axes_names=('input', 'input', 'output'),
                            name='node',
                            network=net,
                            param_edges=True,
                            init_method='randn')
        nodes.append(node)
        input_edges += [node['input_0'], node['input_1']]
    net.set_data_nodes(input_edges=input_edges,
                       batch_sizes=[10])
    data = torch.randn(10, 3, 2 * 5)
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
                       batch_sizes=[10])
    data = torch.randn(10, 3, 2 * 5)
    net._add_data(data)

    stack_node = tn.stack(nodes, name='stack_node')
    stack_input_0 = tn.stack([node.neighbours('input_0') for node in nodes],
                             name='stack_input_0')
    stack_input_1 = tn.stack([node.neighbours('input_1') for node in nodes],
                             name='stack_input_1')

    with pytest.raises(ValueError):
        tn.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)

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
                       batch_sizes=[10])
    data = torch.randn(10, 3, 2 * 5)
    net._add_data(data)
    net['data_0'].disconnect_edges()

    stack_node = tn.stack(nodes, name='stack_node')
    stack_input_0 = tn.stack([net['data_0']] + [node.neighbours('input_0') for node in nodes][1:],
                             name='stack_input_0')

    with pytest.raises(ValueError):
        stack_node['input_0'] ^ stack_input_0['feature']


def test_mps():
    net = tn.TensorNetwork()
    nodes = []
    input_edges = []
    for i in range(11):
        node = tn.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       name=f'node_{i}',
                       network=net,
                       init_method='randn')
        nodes.append(node)
        if i != 5:
            input_edges.append(node['input'])
    for i in range(10):
        net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']
    net.set_data_nodes(input_edges=input_edges,
                       batch_sizes=[10])
    data = torch.randn(10, 5, 10)
    net._add_data(data)
    result_list = tn.stacked_einsum('lir,bi->lbr', nodes[:5] + nodes[6:], list(net.data_nodes.values()))
    result_list = result_list[:5] + [nodes[5]] + result_list[5:]

    node = result_list[0]
    for i in range(1, 5):
        node = tn.einsum('lbr,rbs->lbs', node, result_list[i])
    node = tn.einsum('lbr,ris->lbis', node, result_list[5])
    for i in range(6, 11):
        node = tn.einsum('lbir,rbs->lbis', node, result_list[i])

    assert node.shape == (2, 10, 5, 2)

    # Param
    net = tn.TensorNetwork()
    nodes = []
    input_edges = []
    shift = nn.Parameter(torch.tensor(-0.5))
    slope = nn.Parameter(torch.tensor(20.))
    for i in range(11):
        node = tn.ParamNode(shape=(2, 5, 2),
                            axes_names=('left', 'input', 'right'),
                            name=f'node_{i}',
                            network=net,
                            param_edges=True,
                            init_method='randn')
        for edge in node.edges:
            edge.set_parameters(shift, slope)
        nodes.append(node)
        if i != 5:
            input_edges.append(node['input'])
    for i in range(10):
        net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']
    net.set_data_nodes(input_edges=input_edges,
                       batch_sizes=[10])
    data = torch.randn(10, 5, 10)
    net._add_data(data)
    result_list = tn.stacked_einsum('lir,bi->lbr', nodes[:5] + nodes[6:], list(net.data_nodes.values()))
    result_list = result_list[:5] + [nodes[5]] + result_list[5:]

    node = result_list[0]
    for i in range(1, 5):
        node = tn.einsum('lbr,rbs->lbs', node, result_list[i])
    node = tn.einsum('lbr,ris->lbis', node, result_list[5])
    for i in range(6, 11):
        node = tn.einsum('lbir,rbs->lbis', node, result_list[i])

    assert node.shape == (2, 10, 5, 2)
    mean = node.tensor.mean()
    mean.backward()
    node



