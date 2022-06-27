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

    # Try to set tensor with wrong shape
    wrong_tensor = torch.randn(2, 5, 5)
    with pytest.raises(ValueError):
        node1.set_tensor(tensor=wrong_tensor)

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
    net = tn.TensorNetwork()
    node = tn.Node(shape=(3, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node',
                   network=net,
                   init_method='randn')
    node = node.parameterize()
    assert isinstance(node, tn.ParamNode)
    assert node['left'].node1 == node
    assert isinstance(node['left'], tn.Edge)

    node = node.parameterize(False)
    assert isinstance(node, tn.Node)
    assert node['left'].node1 == node
    assert isinstance(node['left'], tn.Edge)

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
    assert isinstance(node[0], tn.ParamEdge)
    assert node[0].dim() == node.shape[0]


def test_copy_node():
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


def test_permute():
    node = tn.Node(shape=(2, 5, 2),
                   axes_names=('left', 'input', 'right'),
                   name='node',
                   init_method='randn')
    permuted_node = node.permute((0, 2, 1))

    assert permuted_node['left'] == node['left']
    assert permuted_node['input'] == node['input']
    assert permuted_node['right'] == node['right']

    assert permuted_node[0] == node[0]
    assert permuted_node[1] == node[2]
    assert permuted_node[2] == node[1]

    assert torch.equal(permuted_node.tensor, node.tensor.permute(0, 2, 1))

    # Param
    node = tn.ParamNode(shape=(2, 5, 2),
                        axes_names=('left', 'input', 'right'),
                        name='node',
                        init_method='randn')
    permuted_node = node.permute((0, 2, 1))

    assert permuted_node['left'] == node['left']
    assert permuted_node['input'] == node['input']
    assert permuted_node['right'] == node['right']

    assert permuted_node[0] == node[0]
    assert permuted_node[1] == node[2]
    assert permuted_node[2] == node[1]

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

    assert not new_edge.is_updated()

    new_matrix = new_edge.matrix
    assert not torch.equal(prev_matrix, new_matrix)


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
    node1['left'] ^ node2['left']
    node1['right'] ^ node2['right']
    node3 = node1 @ node2
    assert node3.shape == (5, 5)
    assert node3.axes_names == ['input_0', 'input_1']

    node1 = tn.Node(shape=(2, 5, 5, 2),
                    axes_names=('left', 'input', 'input', 'right'),
                    name='node1')
    node1['left'] ^ node1['right']
    node1['input_0'] ^ node1['input_1']
    node2 = node1 @ node1
    assert node2.shape == ()
    assert len(node2.edges) == 0

    node1 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node1',
                         param_edges=True)
    node2 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node2',
                         param_edges=True)
    node1['left'] ^ node2['left']
    node1['right'] ^ node2['right']
    node3 = node1 @ node2
    assert node3.shape == (5, 5)
    assert node3.axes_names == ['input_0', 'input_1']

    node1 = tn.ParamNode(shape=(2, 5, 5, 2),
                         axes_names=('left', 'input', 'input', 'right'),
                         name='node1',
                         param_edges=True)
    node1['left'] ^ node1['right']
    node1['input_0'] ^ node1['input_1']
    node2 = node1 @ node1
    assert node2.shape == ()
    assert len(node2.edges) == 0


def test_contract_edge():
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1')
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2')
    edge = node1[2] ^ node2[0]
    node3 = edge.contract()
    assert node3['left'] == node1['left']
    assert node3['right'] == node2['right']

    with pytest.raises(ValueError):
        tn.contract_edges([node1[0]], node1, node2)

    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1')
    edge = node1[2] ^ node1[0]
    node2 = edge.contract()
    assert len(node2.edges) == 1
    assert node2[0].axis1.name == 'input'

    node1 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node1',
                         param_edges=True)
    node2 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node2',
                         param_edges=True)
    edge = node1[2] ^ node2[0]
    node3 = edge.contract()
    assert node3['left'] == node1['left']
    assert node3['right'] == node2['right']

    node1 = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         name='node1',
                         param_edges=True)
    edge = node1[2] ^ node1[0]
    node2 = edge.contract()
    assert len(node2.edges) == 1
    assert node2[0].axis1.name == 'input'


def test_svd_edge():
    node1 = tn.Node(shape=(2, 5, 3),
                    axes_names=('left', 'contract', 'batch1'),
                    name='node1',
                    init_method='randn')
    node2 = tn.Node(shape=(3, 5, 7),
                    axes_names=('batch2', 'contract', 'right'),
                    name='node2',
                    init_method='randn')
    edge = node1['contract'] ^ node2['contract']
    edge.svd(side='left', cum_percentage=0.9)


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
    node3.reattach_edges(override=False)
    edge = node3[3]
    node3[3] ^ node4[0]
    assert node3[3] != edge


def test_tensor_product():
    node1 = tn.Node(shape=(2, 3),
                    axes_names=('left', 'right'),
                    name='node1',
                    init_method='randn')
    node2 = tn.Node(shape=(4, 5),
                    axes_names=('left', 'right'),
                    name='node2',
                    init_method='randn')

    node3 = node1 % node2
    assert node3.shape == (2, 3, 4, 5)
    assert node3.edges == node1.edges + node2.edges

    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1',
                    init_method='randn')
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2',
                    init_method='randn')
    node1[2] ^ node2[0]
    node3 = node1 % node2
    assert node3.shape == (2, 5, 2, 2, 5, 2)
    assert node3.edges == node1.edges + node2.edges


def test_mul_add_sub():
    node1 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node1',
                    init_method='randn')
    node2 = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node2',
                    init_method='randn')

    node_mul = node1 * node2
    assert node_mul.shape == (2, 5, 2)
    assert torch.equal(node_mul.tensor, node1.tensor * node2.tensor)

    node_add = node1 + node2
    assert node_add.shape == (2, 5, 2)
    assert torch.equal(node_add.tensor, node1.tensor + node2.tensor)

    node_sub = node1 - node2
    assert node_sub.shape == (2, 5, 2)
    assert torch.equal(node_sub.tensor, node1.tensor - node2.tensor)


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
    node_tprod.tensor.sum().backward()
    assert not torch.equal(node1.grad, torch.zeros(node1.shape))
    assert not torch.equal(node2.grad, torch.zeros(node2.shape))
    net.zero_grad()
    assert torch.equal(node1.grad, torch.zeros(node1.shape))
    assert torch.equal(node2.grad, torch.zeros(node2.shape))

    node_mul = node1 * node2
    node_mul.tensor.sum().backward()
    assert not torch.equal(node1.grad, torch.zeros(node1.shape))
    assert not torch.equal(node2.grad, torch.zeros(node2.shape))
    net.zero_grad()
    assert torch.equal(node1.grad, torch.zeros(node1.shape))
    assert torch.equal(node2.grad, torch.zeros(node2.shape))

    node_add = node1 + node2
    node_add.tensor.sum().backward()
    assert not torch.equal(node1.grad, torch.zeros(node1.shape))
    assert not torch.equal(node2.grad, torch.zeros(node2.shape))
    net.zero_grad()
    assert torch.equal(node1.grad, torch.zeros(node1.shape))
    assert torch.equal(node2.grad, torch.zeros(node2.shape))

    node_sub = node1 - node2
    node_sub.tensor.sum().backward()
    assert not torch.equal(node1.grad, torch.zeros(node1.shape))
    assert not torch.equal(node2.grad, torch.zeros(node2.shape))
    net.zero_grad()
    assert torch.equal(node1.grad, torch.zeros(node1.shape))
    assert torch.equal(node2.grad, torch.zeros(node2.shape))

    node1[2] ^ node2[0]
    node3 = node1 @ node2
    node3.tensor.sum().backward()
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

    assert len(net.edges) == 15
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


def test_tn_consecutive_contractions():
    net = tn.TensorNetwork(name='net')
    for i in range(4):
        _ = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node',
                    network=net)

    for i in range(3):
        net[f'node_{i}']['right'] ^ net[f'node_{i+1}']['left']
    assert len(net.edges) == 6

    node = net['node_0']
    for i in range(1, 4):
        node @= net[f'node_{i}']
    assert len(node.edges) == 6
    assert node.shape == (2, 5, 5, 5, 5, 2)

    new_node = tn.Node(shape=(2, 5, 2),
                       axes_names=('left', 'input', 'right'),
                       name='node')
    new_node.network = net
    assert new_node.name == 'node_4'
    net['node_3'][2] ^ new_node[0]
    with pytest.raises(ValueError):
        node @= new_node


def test_tn_submodules():
    net = tn.TensorNetwork(name='net')
    for i in range(2):
        _ = tn.ParamNode(shape=(2, 5, 2),
                         axes_names=('left', 'input', 'right'),
                         network=net,
                         param_edges=True)
    for i in range(2):
        _ = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    network=net,
                    param_edges=True)

    submodules = [None for _ in net.children()]
    assert len(submodules) == 14

    net['paramnode_0']['right'] ^ net['paramnode_1']['left']
    net['paramnode_1']['right'] ^ net['node_0']['left']
    net['node_0']['right'] ^ net['node_1']['left']
    submodules = [None for _ in net.children()]
    assert len(submodules) == 11


def test_tn_parameterize():
    net = tn.TensorNetwork(name='net')
    for i in range(2):
        _ = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    network=net)

    submodules = [None for _ in net.children()]
    assert len(submodules) == 0

    param_net = net.parameterize()
    submodules = [None for _ in net.children()]
    assert len(submodules) == 0
    submodules = [None for _ in param_net.children()]
    assert len(submodules) == 8

    param_net = net.parameterize(override=True)
    assert param_net == net
    submodules = [None for _ in net.children()]
    assert len(submodules) == 8

    net.parameterize(set_param=False, override=True)
    submodules = [None for _ in net.children()]
    assert len(submodules) == 0


def test_tn_data_nodes():
    net = tn.TensorNetwork(name='net')
    for i in range(4):
        _ = tn.Node(shape=(2, 5, 2),
                    axes_names=('left', 'input', 'right'),
                    name='node',
                    network=net,
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
    assert torch.equal(net.data_nodes['data_0'].tensor, data[:, :, 0])
    assert torch.equal(net.data_nodes['data_1'].tensor, data[:, :, 1])

    data = torch.randn(10, 5, 3)
    with pytest.raises(IndexError):
        net._add_data(data)
