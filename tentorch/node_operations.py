# split, svd, qr, rq, etc.
# contract, contract_between, batched_contract_between, einsum, etc.

from typing import (Union, Optional, Sequence, Text, List)
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


def einsum(string: Text, *nodes: AbstractNode) -> Node:
    if '->' not in string:
        raise ValueError('Einsum `string` should have an arrow `->` separating '
                         'inputs and output strings')
    input_strings = string.split('->')[0].split(',')
    if len(input_strings) != len(nodes):
        raise ValueError('Number of einsum subscripts must be equal to the number of operands')
    try:
        output_string = string.split('->')[1]
    except Exception as e:
        raise ValueError('Output subscripts must be specified') from e

    matrices = []
    matrices_strings = []
    output_list = list(output_string)
    axes_names = []
    edges = []
    node1_list = []
    j, k = 0, 0
    while (len(output_list) > 0) and (j < len(input_strings)):
        if input_strings[j][k] not in output_list:
            edge = nodes[j][k]
            if isinstance(edge, ParamEdge):
                in_matrices = False
                for mat in matrices:
                    if torch.equal(edge.matrix, mat):
                        in_matrices = True
                        break
                if not in_matrices:
                    matrices_strings.append(2 * input_strings[j][k])
                    matrices.append(edge.matrix)
        else:
            axes_names.append(nodes[j].axes[k].name)
            edges.append(nodes[j][k])
            node1_list.append(nodes[j].axes[k].node1)
            output_list.remove(input_strings[j][k])
        k += 1
        if k == len(input_strings[j]):
            k = 0
            j += 1

    if len(output_list) > 0:
        raise ValueError('Input and output subscripts should match on all the edges '
                         'that are not going to be contracted')

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
    def __init__(self,
                 nodes_list: List[AbstractNode],
                 dim: int,
                 shape: Optional[Union[int, Sequence[int], torch.Size]] = None,
                 axis_names: Optional[Sequence[Text]] = None,
                 network: Optional['TensorNetwork'] = None,
                 name: Optional[Text] = None,
                 tensor: Optional[torch.Tensor] = None,
                 param_edges: bool = True) -> None:

        tensors_list = list(map(lambda x: x.tensor, nodes_list))
        tensor = torch.stack(tensors_list, dim=dim)

        self.nodes_list = nodes_list
        self.tensor = tensor
        self.stacked_dim = dim

        self.edges_dict = dict()
        j = 0
        for node in nodes_list:
            for i, edge in enumerate(node.edges):
                if i >= self.stacked_dim:
                    j = 1
                if (i + j) in self.edges_dict:
                    self.edges_dict[(i + j)].append(edge)
                else:
                    self.edges_dict[(i + j)] = [edge]

        super().__init__(tensor=self.tensor)

    @staticmethod
    def set_tensor_format(tensor):
        return tensor

    def make_edge(self, axis: Axis):
        if axis == self.stacked_dim:
            return Edge(node1=self, axis1=axis)
        return StackEdge(self.edges_dict[axis], node1=self, axis1=axis)


# TODO: ignore this at the moment
class StackEdge(Edge):
    """
    Edge que es lista de varios Edges. Se usa para un StackNode

    No hacer cambios de tipos, simplemente que este Edge lleve
    parámetros adicionales de los Edges que está aglutinando
    y su vecinos y tal. Pero que no sea en sí mismo una lista
    de Edges, que eso lo lía todo.
    """

    def __init__(self,
                 edges_list: List['AbstractEdge'],
                 node1: 'AbstractNode',
                 axis1: int,
                 name: Optional[str] = None,
                 node2: Optional['AbstractNode'] = None,
                 axis2: Optional[int] = None,
                 shift: Optional[int] = None,
                 slope: Optional[int] = None) -> None:
        # TODO: edges in list must have same size and dimension
        self.edges_list = edges_list
        super().__init__(node1, axis1, name, node2, axis2, shift, slope)

    def create_parameters(self, shift, slope):
        return None, None, None

    def dim(self):
        """
        Si es ParamEdge se mide en función de sus parámetros la dimensión
        """
        return None

    def create_matrix(self, dim):
        """
        Eye for Edge, Parameter for ParamEdge
        """
        return None

    def __xor__(self, other: 'AbstractEdge') -> 'AbstractEdge':
        return None


def stack():
    pass


def unbind():
    pass


def stacked_contract():
    pass
