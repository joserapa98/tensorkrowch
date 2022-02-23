# split, svd, qr, rq, etc.
# contract, contract_between, batched_contract_between, einsum, etc.

from typing import (Union, Optional, Sequence, Text, List)
from abc import ABC, abstractmethod
import warnings
import copy

import torch
import torch.nn as nn
import opt_einsum

from tentorch.network_components import AbstractNode, Node, ParamNode
from tentorch.network_components import AbstractEdge, Edge, ParamEdge
from tentorch.network_components import TensorNetwork
from tentorch.network_components import (connect, disconnect, get_shared_edges,
                                         contract_between)

"""
# TODO: no se usa
def einsum(string: Text,
           *nodes: Sequence[Union[torch.Tensor, AbstractNode]]) -> AbstractNode:
    new_tensor = torch.einsum(string,
                              *tuple(map(lambda x: x.tensor if isinstance(x, AbstractNode) else x, nodes)))
    new_node = Node(tensor=new_tensor)

    out_string = string.split('->')[1]
    strings = string.split('->')[0].split(',')

    i = 0
    for j, string in enumerate(strings):
        if isinstance(nodes[j], AbstractNode):
            for k, char in enumerate(string):
                if i < len(out_string) and out_string[i] == char:
                    new_node.add_edge(nodes[j][k], i, override=True)
                    i += 1
    return new_node
"""


def einsum(string: Text, *nodes: AbstractNode) -> Node:
    input_strings = string.split('->')[0].split(',')
    output_string = string.split('->')[1]

    matrices = []
    matrices_strings = []
    axes_names = []
    edges = []
    node1_list = []
    i, j, k = 0, 0, 0
    while (i < len(output_string)) and \
            (j < len(input_strings)):
        if input_strings[j][k] != output_string[i]:
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
            i += 1
        k += 1
        if k == len(input_strings[j]):
            k = 0
            j += 1

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


def contract_nodes(nodes: List[AbstractNode],
                   shared_edges: List[AbstractEdge]) -> Node:
    for edge in shared_edges:
        is_shared = False
        for i, node1 in enumerate(nodes):
            for node2 in nodes[:i]:
                if edge in get_shared_edges(node1, node2):
                    is_shared = True
                    break
            if is_shared:
                break
        if not is_shared:
            raise ValueError('All edges in `shared_edges` should be non-dangling, '
                             'shared edges between two nodes in `nodes`')

    n_shared = len(shared_edges)
    shared_subscripts = dict(zip(shared_edges, [opt_einsum.get_symbol(i) for i in range(n_shared)]))

    index = n_shared
    input_strings = []
    used_nodes = []
    output_string = ''
    matrices = []
    matrices_strings = []
    for i, node in enumerate(nodes):
        if node in used_nodes:
            continue
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


def contract_edges_with_batch(shared_edges: List[AbstractEdge],
                              node1: AbstractNode,
                              node2: AbstractNode,
                              batch_edges: List[AbstractEdge]) -> Node:
    if node1 is node2:
        raise ValueError(f'Cannot perform batched contraction between '
                         f'node {node1!s} and itself')
    if any([edge not in get_shared_edges(node1, node2) for edge in shared_edges]):
        raise ValueError('All edges in `shared_edges` should be non-dangling, '
                         'shared edges between `node1` and `node2`')
    for edge in batch_edges:
        if edge in shared_edges:
            raise ValueError(f'Batch edge {edge} is shared between the nodes')

    n_shared = len(shared_edges)
    shared_subscripts = dict(zip(shared_edges, [opt_einsum.get_symbol(i) for i in range(n_shared)]))

    index = n_shared
    input_strings = []
    used_nodes = []
    output_string = ''
    matrices = []
    matrices_strings = []
    for i, node in enumerate([node1, node2]):
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


# TODO: deberíamos permitir contraer varios nodos a la vez, como un einsum,
#  para que vaya más optimizado. En un tree tendremos nodos iguales que apilaremos,
#  y cada nodo va conectado a otros 3, 4 nodos (los que sean). Así que hay que
#  contraer esas 3, 4 pilas de input con la pila de los tensores.
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
    shared_subscripts = dict(zip(shared_edges, _VALID_SUBSCRIPTS[:n_shared]))

    index = n_shared + 1
    input_strings = []
    output_string = ''
    matrices = []
    matrices_strings = []
    for node, batch_edge in zip([node1, node2], [batch_edge1, batch_edge2]):
        string = ''
        matrix_string = ''
        for edge in node.edges:
            if edge in shared_edges:
                string += shared_subscripts[edge]
                if isinstance(edge, ParamEdge):
                    matrices.append(edge.matrix)
                    matrix_string = 2 * shared_subscripts[edge]
            elif edge is batch_edge:
                string += _VALID_SUBSCRIPTS[n_shared]
                if node is node1:
                    output_string += _VALID_SUBSCRIPTS[n_shared]
            else:
                string += _VALID_SUBSCRIPTS[index]
                output_string += _VALID_SUBSCRIPTS[index]
                index += 1
        input_strings.append(string)
        matrices_strings.append(matrix_string)

    matrices_string = ','.join(matrices_strings)
    if len(input_strings) == 1:
        input_string = ''.join(input_strings[0])
        if len(matrices) > 0:
            einsum_string = input_string + ',' + matrices_string + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor, *matrices)
        else:
            einsum_string = input_string + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor)
        name = f'{node1.name}'
    else:
        input_string_0 = ''.join(input_strings[0])
        input_string_1 = ''.join(input_strings[1])
        if len(matrices) > 0:
            einsum_string = input_string_0 + ',' + matrices_string + ',' \
                            + input_string_1 + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor, *matrices, node2.tensor)
        else:
            einsum_string = input_string_0 + ',' + input_string_1 + '->' + output_string
            new_tensor = torch.einsum(einsum_string, node1.tensor, node2.tensor)
        name = f'{node1.name}@{node2.name}'

    axes_names = []
    edges = []
    nodes = [node1, node2]
    i = 0
    for j, string in enumerate(input_strings):
        for k, char in enumerate(string):
            if i < len(output_string):
                if output_string[i] == char:
                    axes_names.append(nodes[j].axes[k].name)
                    edges.append(nodes[j][k])
                    i += 1
            else:
                break
        if i >= len(output_string):
            break

    new_node = Node(axes_names=axes_names,
                    name=name,
                    network=nodes[0].network,
                    param_edges=False,
                    tensor=new_tensor)
    new_node.edges = edges
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
