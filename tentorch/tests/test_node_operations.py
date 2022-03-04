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
    net._add_data(data.unbind(2))

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
    net._add_data(data.unbind(2))

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
    node3 = node1 @ node2
    assert node3.shape == (10,)

    node1 = tn.ParamNode(shape=(10, 2, 3),
                         axes_names=('batch', 'left', 'right'),
                         name='node1',
                         init_method='randn')
    node1['left'].parameterize(True)
    node1['right'].parameterize(True)
    node2 = tn.ParamNode(shape=(10, 2, 3),
                         axes_names=('batch', 'left', 'right'),
                         name='node2',
                         init_method='randn')
    node2['left'].parameterize(True)
    node2['right'].parameterize(True)
    node1['left'] ^ node2['left']
    node1['right'] ^ node2['right']
    node3 = node1 @ node2
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
    net._add_data(data.unbind(2))
    
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
    net._add_data(data.unbind(2))

    stack_node = tn.stack(nodes, name='stack_node')
    stack_input_0 = tn.stack([node.neighbours('input_0') for node in nodes],
                             name='stack_input_0')
    stack_input_1 = tn.stack([node.neighbours('input_1') for node in nodes],
                             name='stack_input_1')
    stack_node['input_0'] ^ stack_input_0['feature']
    stack_node['input_1'] ^ stack_input_1['feature']

    stack_result = tn.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
    assert stack_result.shape == (5, 10, 2)

    # Dimensions
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
    net._add_data(data.unbind(2))

    stack_node = tn.stack(nodes, name='stack_node')
    stack_input_0 = tn.stack([node.neighbours('input_0') for node in nodes],
                             name='stack_input_0')
    stack_input_1 = tn.stack([node.neighbours('input_1') for node in nodes],
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
    net._add_data(data.unbind(2))

    stack_node = tn.stack(nodes, name='stack_node')
    stack_input_0 = tn.stack([node.neighbours('input_0') for node in nodes],
                             name='stack_input_0')
    stack_input_1 = tn.stack([node.neighbours('input_1') for node in nodes],
                             name='stack_input_1')

    stack_result = tn.einsum('sijk,sbi,sbj->sbk', stack_node, stack_input_0, stack_input_1)
    assert stack_result.shape == (5, 10, 2)

    nodes = tn.unbind(stack_result)
    assert len(nodes) == 5
    assert nodes[0].shape == (10, 2)

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
    net._add_data(data.unbind(2))
    net['data_0'].disconnect_edges()

    stack_node = tn.stack(nodes, name='stack_node')
    stack_input_0 = tn.stack([net['data_0']] + [node.neighbours('input_0') for node in nodes][1:],
                             name='stack_input_0')

    # Try to connect stack edges when some edge of one stack is
    # not connected to the corresponding edge in the other stack
    with pytest.raises(ValueError):
        stack_node['input_0'] ^ stack_input_0['feature']

    # Error 2
    net = tn.TensorNetwork()
    nodes = []
    for i in range(5):
        node = tn.Node(shape=(3, 3, 2),
                       axes_names=('input', 'input', 'output'),
                       name='node',
                       network=net,
                       init_method='randn')
        nodes.append(node)
    net['node_0'].param_edges(True)

    # Try to stack edges of different types
    with pytest.raises(TypeError):
        stack_node = tn.stack(nodes, name='stack_node')


def test_stacked_einsum():
    # MPS
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
    net._add_data(data.unbind(2))
    result_list = tn.stacked_einsum('lir,bi->lbr', nodes[:5] + nodes[6:], list(net.data_nodes.values()))
    result_list = result_list[:5] + [nodes[5]] + result_list[5:]

    node1 = result_list[0]
    for i in range(1, 11):
        node1 @= result_list[i]
    assert node1.shape == (2, 10, 5, 2)

    node2 = result_list[0]
    for i in range(1, 5):
        node2 = tn.einsum('lbr,rbs->lbs', node2, result_list[i])
    node2 = tn.einsum('lbr,ris->lbis', node2, result_list[5])
    for i in range(6, 11):
        node2 = tn.einsum('lbir,rbs->lbis', node2, result_list[i])
    assert node2.shape == (2, 10, 5, 2)

    assert torch.equal(node1.tensor, node2.tensor)

    # Param
    net = tn.TensorNetwork()
    nodes = []
    input_edges = []
    for i in range(11):
        node = tn.ParamNode(shape=(2, 5, 2),
                            axes_names=('left', 'input', 'right'),
                            name=f'node_{i}',
                            network=net,
                            param_edges=True,
                            init_method='randn')
        nodes.append(node)
        if i != 5:
            input_edges.append(node['input'])
    for i in range(10):
        net[f'node_{i}']['right'] ^ net[f'node_{i + 1}']['left']
    net.set_data_nodes(input_edges=input_edges,
                       batch_sizes=[10])
    data = torch.randn(10, 5, 10)
    net._add_data(data.unbind(2))
    result_list = tn.stacked_einsum('lir,bi->lbr', nodes[:5] + nodes[6:], list(net.data_nodes.values()))
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
        node2 = tn.einsum('lbr,rbs->lbs', node2, result_list[i])
    node2 = tn.einsum('lbr,ris->lbis', node2, result_list[5])
    for i in range(6, 11):
        node2 = tn.einsum('lbir,rbs->lbis', node2, result_list[i])
    assert node2.shape == (2, 10, 5, 2)

    assert torch.equal(node1.tensor, node2.tensor)
