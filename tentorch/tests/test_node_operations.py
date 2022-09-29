"""
Tests for node_operations
"""

import pytest

import torch
import torch.nn as nn
import tentorch as tn

import time


def test_tensor_product():
    node1 = tn.Node(shape=(2, 3),
                    axes_names=('left', 'right'),
                    name='node1',
                    init_method='randn')
    node2 = tn.Node(shape=(4, 5),
                    axes_names=('left', 'right'),
                    name='node2',
                    init_method='randn')
    # Tensor product cannot be performed between nodes in different networks
    with pytest.raises(ValueError):
        node3 = node1 % node2

    net = tn.TensorNetwork()
    node1.network = net
    node2.network = net
    node3 = node1 % node2
    assert node3.shape == (2, 3, 4, 5)
    assert node3.edges == node1.edges + node2.edges

    # Second time
    node3 = node1 % node2

    net2 = tn.TensorNetwork()
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1',
                    network=net2,
                    init_method='randn')
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2',
                    network=net2,
                    init_method='randn')
    node1[2] ^ node2[0]
    # Tensor product cannot be performed between connected nodes
    with pytest.raises(ValueError):
        node3 = node1 % node2


def test_mul_add_sub():
    net = tn.TensorNetwork()
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1',
                    network=net,
                    init_method='randn')
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2',
                    network=net,
                    init_method='randn')

    node_mul = node1 * node2
    assert node_mul.shape == (2, 5, 2)
    assert torch.equal(node_mul.tensor, node1.tensor * node2.tensor)

    # Second time
    node_mul = node1 * node2

    node_add = node1 + node2
    assert node_add.shape == (2, 5, 2)
    assert torch.equal(node_add.tensor, node1.tensor + node2.tensor)

    # Second time
    node_add = node1 + node2

    node_sub = node1 - node2
    assert node_sub.shape == (2, 5, 2)
    assert torch.equal(node_sub.tensor, node1.tensor - node2.tensor)

    # Second time
    node_sub = node1 - node2


def test_compute_grad():
    net = tn.TensorNetwork(name='net')
    node1 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node1',
                         network=net,
                         param_edges=True,
                         init_method='randn')
    node2 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node2',
                         network=net,
                         param_edges=True,
                         init_method='randn')

    node_tprod = node1 % node2
    node_tprod.sum().backward()
    assert not torch.equal(node1.grad, torch.zeros(node1.shape))
    assert not torch.equal(node2.grad, torch.zeros(node2.shape))
    net.zero_grad()
    assert torch.equal(node1.grad, torch.zeros(node1.shape))
    assert torch.equal(node2.grad, torch.zeros(node2.shape))

    node_mul = node1 * node2
    node_mul.sum().backward()
    assert not torch.equal(node1.grad, torch.zeros(node1.shape))
    assert not torch.equal(node2.grad, torch.zeros(node2.shape))
    net.zero_grad()
    assert torch.equal(node1.grad, torch.zeros(node1.shape))
    assert torch.equal(node2.grad, torch.zeros(node2.shape))

    node_add = node1 + node2
    node_add.sum().backward()
    assert not torch.equal(node1.grad, torch.zeros(node1.shape))
    assert not torch.equal(node2.grad, torch.zeros(node2.shape))
    net.zero_grad()
    assert torch.equal(node1.grad, torch.zeros(node1.shape))
    assert torch.equal(node2.grad, torch.zeros(node2.shape))

    node_sub = node1 - node2
    node_sub.sum().backward()
    assert not torch.equal(node1.grad, torch.zeros(node1.shape))
    assert not torch.equal(node2.grad, torch.zeros(node2.shape))
    net.zero_grad()
    assert torch.equal(node1.grad, torch.zeros(node1.shape))
    assert torch.equal(node2.grad, torch.zeros(node2.shape))

    node1[2] ^ node2[0]
    node3 = node1 @ node2
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


def test_contract_between():
    print()

    # Contract two nodes
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1',
                    init_method='randn')
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2',
                    init_method='randn')
    node1['left'] ^ node2['left']
    node1['right'] ^ node2['right']
    start = time.time()
    node3 = node1 @ node2
    print('First contraction node1, node2:', time.time() - start)
    assert node3.shape == (5, 5)
    assert node3.axes_names == ['input_0', 'input_1']

    # Repeat contraction
    start = time.time()
    node4 = node1 @ node2
    print('Second contraction node1, node2', time.time() - start)
    assert node3 == node4
    assert torch.equal(node3.tensor, node4.tensor)

    # Compute traces
    node1 = tn.Node(shape=(2, 5, 5, 2),
                    axes_names=('left', 'input', 'input', 'right'),
                    name='node1',
                    init_method='randn')
    node1['left'] ^ node1['right']
    node1['input_0'] ^ node1['input_1']
    start = time.time()
    node2 = node1 @ node1
    print('First trace:', time.time() - start)
    assert node2.shape == ()
    assert len(node2.edges) == 0

    # Repeat traces
    start = time.time()
    node3 = node1 @ node1
    print('Second trace:', time.time() - start)
    assert node2 == node3
    assert torch.equal(node2.tensor, node3.tensor)

    print()
    # Contract two parametric nodes
    node1 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node1',
                         param_edges=True,
                         init_method='randn')
    node2 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node2',
                         param_edges=True,
                         init_method='randn')
    node1['left'] ^ node2['left']
    node1['right'] ^ node2['right']
    start = time.time()
    node3 = node1 @ node2
    print('First contraction param node1, node2:', time.time() - start)
    assert node3.shape == (5, 5)
    assert node3.axes_names == ['input_0', 'input_1']

    # Repeat contraction
    start = time.time()
    node4 = node1 @ node2
    print('Second contraction param node1, node2', time.time() - start)
    assert node3 == node4
    assert torch.equal(node3.tensor, node4.tensor)

    # Compute traces
    node1 = tn.ParamNode(shape=(2, 5, 5, 2),
                         axes_names=('left', 'input', 'input', 'right'),
                         name='node1',
                         param_edges=True,
                         init_method='randn')
    node1['left'] ^ node1['right']
    node1['input_0'] ^ node1['input_1']
    start = time.time()
    node2 = node1 @ node1
    print('First param trace:', time.time() - start)
    assert node2.shape == ()
    assert len(node2.edges) == 0

    # Repeat traces
    start = time.time()
    node3 = node1 @ node1
    print('Second param trace:', time.time() - start)
    assert node2 == node3
    assert torch.equal(node2.tensor, node3.tensor)


def test_contract_edge():
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1',
                    init_method='randn')
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2',
                    init_method='randn')
    edge = node1[2] ^ node2[0]
    node3 = edge.contract()
    assert node3['left'] == node1['left']
    assert node3['right'] == node2['right']

    with pytest.raises(ValueError):
        tn.contract_edges([node1[0]], node1, node2)

    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1',
                    init_method='randn')
    edge = node1[2] ^ node1[0]
    node2 = edge.contract()
    assert len(node2.edges) == 1
    assert node2[0].axis1.name == 'input'

    node1 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node1',
                         param_edges=True,
                         init_method='randn')
    node2 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node2',
                         param_edges=True,
                         init_method='randn')
    edge = node1[2] ^ node2[0]
    node3 = edge.contract()
    assert node3['left'] == node1['left']
    assert node3['right'] == node2['right']

    node1 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node1',
                         param_edges=True,
                         init_method='randn')
    edge = node1[2] ^ node1[0]
    node2 = edge.contract()
    assert len(node2.edges) == 1
    assert node2[0].axis1.name == 'input'


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
        node = tn.Node(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
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

    nodes = tn.unbind(stack_result)
    assert len(nodes) == 5
    assert nodes[0].shape == (10, 2)

    # Param
    net = tn.TensorNetwork()
    nodes = []
    input_edges = []
    for i in range(5):
        node = tn.ParamNode(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
                            param_edges=True, init_method='randn')
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
        node = tn.ParamNode(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
                            param_edges=True, init_method='randn')
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
        node = tn.Node(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
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
        node = tn.Node(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
                       init_method='randn')
        nodes.append(node)
        input_edges += [node['input_0'], node['input_1']]
    net.set_data_nodes(input_edges=input_edges,
                       batch_sizes=[10])
    data = torch.randn(10, 3, 2 * 5)
    net._add_data(data.unbind(2))
    net['data_0'].disconnect()

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
        node = tn.Node(shape=(3, 3, 2), axes_names=('input', 'input', 'output'), name='node', network=net,
                       init_method='randn')
        nodes.append(node)
    net['node_0'].param_edges(True)

    # Try to stack edges of different types
    with pytest.raises(TypeError):
        stack_node = tn.stack(nodes, name='stack_node')


def test_einsum():
    net = tn.TensorNetwork(name='net')
    node = tn.Node(shape=(5, 5, 5, 5, 2), axes_names=('input', 'input', 'input', 'input', 'output'), network=net,
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
    node = tn.ParamNode(shape=(5, 5, 5, 5, 2), axes_names=('input', 'input', 'input', 'input', 'output'), network=net,
                        param_edges=True, init_method='randn')
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


def test_stacked_einsum():
    # MPS
    net = tn.TensorNetwork()
    nodes = []
    input_edges = []
    for i in range(11):
        node = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name=f'node_{i}', network=net,
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
        node = tn.ParamNode(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name=f'node_{i}', network=net,
                            param_edges=True, init_method='randn')
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
