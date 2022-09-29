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
    assert node1.network == node2.network
    assert node2.network != net

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
    # TODO: Problem! Does not raise error because override is True when using ^
    #with pytest.raises(ValueError):
    #    node1[2] ^ node2[0]
    tn.connect(node1[2], node2[0])
    assert node1.network == net1
    assert node2.network == net1


def test_svd_edge():
    # TODO: complicado, esto pal futuro
    node1 = tn.Node(shape=(2, 5, 3), axes_names=('left', 'contract', 'batch1'), name='node1', init_method='randn')
    node2 = tn.Node(shape=(3, 5, 7), axes_names=('batch2', 'contract', 'right'), name='node2', init_method='randn')
    edge = node1['contract'] ^ node2['contract']
    # edge.svd(side='left', cum_percentage=0.9)


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
    node1 = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name='node1')
    node2 = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name='node2')

    node1[2] ^ node2[0]
    node3 = node1 @ node2

    node4 = tn.Node(shape=(2, 5, 2), axes_names=('left', 'input', 'right'), name='node4')
    edge = node3[3]
    node3[3] ^ node4[0]
    assert node3[3] == edge
    assert node2[2] == node4[0]
    # TODO: Problem! Cuando conectamos edges, el nuevo edge se guarda en los nodos originales
    #  donde los edges estaban conectados, en este caso los edges de node3 están conectados a
    #  node1 y node2, por lo que las nuevas conexiones se verían en esos nodos, pero no en node3
    # TODO: Prohibimos conectar nodos después de hacer operaciones, primero se construye toda la
    #  red, después se definen operaciones, después se optimiza memoria para esas operaciones
    #  y finalmente se realiza toda la contracción
    #node3 @ node4

    node4[0] | node4[0]
    node3._reattach_edges(override=False)
    edge = node3[3]
    #node3[3] ^ node4[0]
    #assert node3[3] != edge
