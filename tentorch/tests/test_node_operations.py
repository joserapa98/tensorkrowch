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
