"""
This script contains:

    Node operations:
        *einsum
        *batched_contract_between

"""
# split, svd, qr, rq, etc.
# contract, contract_between, batched_contract_between, einsum, etc.

from typing import (Union, Optional, Sequence, Text, List)
from tentorch.network_components import Ax, Shape
from abc import ABC, abstractmethod
import warnings
import copy

import torch
import torch.nn as nn
import opt_einsum

from tentorch.network_components import Axis
from tentorch.network_components import AbstractNode, Node, ParamNode
from tentorch.network_components import AbstractEdge, Edge, ParamEdge
from tentorch.network_components import TensorNetwork
from tentorch.network_components import (connect, disconnect, get_shared_edges,
                                         contract_between)


# TODO: comprobar si los edges que se contraen eran de nodos realmente conectados
def einsum(string: Text, *nodes: AbstractNode) -> Node:
    if '->' not in string:
        raise ValueError('Einsum `string` should have an arrow `->` separating '
                         'inputs and output strings')
    input_strings = string.split('->')[0].split(',')
    if len(input_strings) != len(nodes):
        raise ValueError('Number of einsum subscripts must be equal to the number of operands')
    if len(string.split('->')) >= 2:
        output_string = string.split('->')[1]
    else:
        output_string = ''

    matrices = []
    matrices_strings = []
    output_dict = dict(zip(output_string, [0] * len(output_string)))
    contracted_edges = dict()
    axes_names = []
    edges = []
    node1_list = []
    for j, input_string in enumerate(input_strings):
        for k, char in enumerate(input_string):
            if char not in output_dict:
                if char not in contracted_edges:
                    contracted_edges[char] = 1
                else:
                    contracted_edges[char] += 1
                edge = nodes[j][k]
                if isinstance(edge, ParamEdge):
                    in_matrices = False
                    for mat in matrices:
                        if torch.equal(edge.matrix, mat):
                            in_matrices = True
                            break
                    if not in_matrices:
                        matrices_strings.append(2 * char)
                        matrices.append(edge.matrix)
            else:
                if output_dict[char] == 0:
                    axes_names.append(nodes[j].axes[k].name)
                    edges.append(nodes[j][k])
                    node1_list.append(nodes[j].axes[k].node1)
                output_dict[char] += 1

    for char in output_dict:
        if output_dict[char] == 0:
            raise ValueError(f'Output subscript {char} must appear among '
                             f'the input subscripts')
    for char in contracted_edges:
        if contracted_edges[char] == 1:
            raise ValueError(f'Subscript {char} appears only once in the input '
                             f'but it not among the output subscripts')
        elif contracted_edges[char] > 2:
            raise ValueError(f'Subscript {char} appearing more than once in the '
                             f'input should be a batch index, but it does not '
                             f'appear among the output subscripts')

    input_string = ','.join(input_strings + matrices_strings)
    einsum_string = input_string + '->' + output_string
    tensors = list(map(lambda n: n.tensor, nodes))
    new_tensor = opt_einsum.contract(einsum_string, *(tensors + matrices))

    new_node = Node(axes_names=axes_names,
                    name='einsum_node',
                    network=nodes[0].network,
                    param_edges=False,
                    tensor=new_tensor,
                    edges=edges,
                    node1_list=node1_list)
    return new_node


def batched_contract_between(node1: AbstractNode,
                             node2: AbstractNode,
                             batch_edge1: AbstractEdge,
                             batch_edge2: AbstractEdge) -> AbstractNode:
    """
    Contract between that supports one batch edge in each node.

    Uses einsum property: 'bij,bjk->bik'.

    Args:
        node1: First node to contract.
        node2: Second node to contract.
        batch_edge1: The edge of node1 that corresponds to its batch index.
        batch_edge2: The edge of node2 that corresponds to its batch index.

    Returns:
        new_node: Result of the contraction. This node has by default batch_edge1
        as its batch edge. Its edges are in order of the dangling edges of
        node1 followed by the dangling edges of node2.
    """
    # Stop computing if both nodes are the same
    if node1 is node2:
        raise ValueError(f'Cannot perform batched contraction between '
                         f'node {node1} and itself')

    shared_edges = get_shared_edges(node1, node2)
    if not shared_edges:
        raise ValueError(f'No edges found between nodes {node1} and {node2}')

    if batch_edge1 in shared_edges:
        raise ValueError(f'Batch edge {batch_edge1} is shared between the nodes')
    if batch_edge2 in shared_edges:
        raise ValueError(f'Batch edge {batch_edge2} is shared between the nodes')

    n_shared = len(shared_edges)
    shared_subscripts = dict(zip(shared_edges, [opt_einsum.get_symbol(i) for i in range(n_shared)]))

    index = n_shared + 1
    input_strings = []
    used_nodes = []
    output_string = ''
    matrices = []
    matrices_strings = []
    for i, (node, batch_edge) in enumerate(zip([node1, node2],
                                               [batch_edge1, batch_edge2])):
        if (i == 1) and (node1 == node2):
            break
        string = ''
        for edge in node.edges:
            if edge in shared_edges:
                string += shared_subscripts[edge]
                if isinstance(edge, ParamEdge):
                    in_matrices = False
                    for mat in matrices:
                        if torch.equal(edge.matrix, mat):
                            in_matrices = True
                            break
                    if not in_matrices:
                        matrices_strings.append(2 * shared_subscripts[edge])
                        matrices.append(edge.matrix)
            elif edge == batch_edge:
                string += opt_einsum.get_symbol(n_shared)
                if i == 0:
                    output_string += opt_einsum.get_symbol(n_shared)
            else:
                string += opt_einsum.get_symbol(index)
                output_string += opt_einsum.get_symbol(index)
                index += 1
        input_strings.append(string)
        used_nodes.append(node)

    input_string = ','.join(input_strings + matrices_strings)
    einsum_string = input_string + '->' + output_string
    tensors = list(map(lambda n: n.tensor, used_nodes))
    names = '_'.join(map(lambda n: n.name, used_nodes))
    new_tensor = opt_einsum.contract(einsum_string, *(tensors + matrices))
    new_name = f'contract_{names}'

    axes_names = []
    edges = []
    node1_list = []
    i, j, k = 0, 0, 0
    while (i < len(output_string)) and \
            (j < len(input_strings)):
        if output_string[i] == input_strings[j][k]:
            axes_names.append(used_nodes[j].axes[k].name)
            edges.append(used_nodes[j][k])
            node1_list.append(used_nodes[j].axes[k].node1)
            i += 1
        k += 1
        if k == len(input_strings[j]):
            k = 0
            j += 1

    new_node = Node(axes_names=axes_names,
                    name=new_name,
                    network=used_nodes[0].network,
                    param_edges=False,
                    tensor=new_tensor,
                    edges=edges,
                    node1_list=node1_list)
    return new_node


# stack, unbind, stacknode, stackedge, stacked_contraction
# TODO: implement this, ignore this at the moment
class StackNode(Node):
    def __init__(self, nodes: List[AbstractNode]) -> None:
        same_type = True
        same_shape = True
        same_edge_type = True
        for i in range(len(nodes[:-1])):
            same_type &= isinstance(nodes[i], type(nodes[i + 1]))
            same_shape &= nodes[i].shape == nodes[i + 1].shape
            for edge1, edge2 in zip(nodes[i].edges, nodes[i + 1].edges):
                same_edge_type &= isinstance(edge1, type(edge2))
        if not same_type:
            raise TypeError('Cannot stack nodes of different types. Nodes '
                            'must be either all Node or all ParamNode type')
        if not same_shape:
            raise ValueError('Cannot stack nodes with different shapes')
        if not same_edge_type:
            raise TypeError('Cannot stack nodes with edges of different types. '
                            'The edges that are attached to the same axis in '
                            'each node must be either all Edge or all ParamEdge type')

        edges_dict = dict()
        for node in nodes:
            for axis in node.axes:
                if axis in edges_dict:
                    edges_dict[axis] = [node[axis]]
                else:
                    edges_dict[axis] += [node[axis]]
        self._edges_dict = edges_dict

        stacked_tensor = torch.stack([node.tensor for node in nodes])
        super().__init__(axes_names=['stack'] + nodes[0].axes_names,
                         network=nodes[0].network,
                         tensor=stacked_tensor)

    def make_edge(self, axis: Axis):
        if axis.num == 0:
            return Edge(node1=self, axis1=axis)
        if isinstance(self._edges_dict[axis][0], Edge):
            return StackEdge(self._edges_dict[axis], node1=self, axis1=axis)
        elif isinstance(self._edges_dict[axis][0], ParamEdge):
            return ParamStackEdge(self._edges_dict[axis], node1=self, axis1=axis)


class StackEdge(Edge):
    def __init__(self,
                 edges: List[Edge],
                 node1: AbstractNode,
                 axis1: Axis,
                 node2: Optional[AbstractNode] = None,
                 axis2: Optional[Axis] = None) -> None:
        self._edges = edges
        super().__init__(node1=node1, axis1=axis1,
                         node2=node2, axis2=axis2)


class ParamStackEdge(ParamEdge):
    def __init__(self,
                 edges: List[ParamEdge],
                 node1: AbstractNode,
                 axis1: Axis) -> None:
        self._edges = edges
        super().__init__(node1=node1, axis1=axis1,
                         shift=self._edges[0].shift,
                         slope=self._edges[0].slope)


def stack():
    pass


def unbind():
    # if node.axes[0].name == 'stack':
    #   unstack
    pass


def stacked_contract():
    # pasar expresión del tipo einsum y pasando una lista
    # de las secuencias de nodos que irían en einsum
    pass
