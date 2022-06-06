"""
This script contains:

    Node operations:
        *einsum
        *connect_stack
        *stack
        *unbind
        *stacked_einsum

    Classes for stacks:
        *StackNode
        *AbstractStackEdge:
            +StackEdge
            +ParamStackEdge
"""
# split, svd, qr, rq, etc. -> using einsum-like strings, useful

from typing import Union, Optional, Text, List, Dict
from abc import abstractmethod

import torch
import opt_einsum

from tentorch.network_components import Axis
from tentorch.network_components import AbstractNode, Node
from tentorch.network_components import AbstractEdge, Edge, ParamEdge
from tentorch.network_components import connect

# TODO:
import time


def einsum(string: Text, *nodes: AbstractNode) -> Node:
    """
    Adapt opt_einsum contract function to make it suitable for nodes

    Parameters
    ----------
    string: einsum-like string
    nodes: nodes to be operated

    Returns
    -------
    new_node: node resultant from the einsum operation
    """
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

    # Check string and collect information from involved edges
    matrices = []
    matrices_strings = []
    output_dict = dict(zip(output_string, [0] * len(output_string)))
    # Used for counting appearances of output subscripts in the input strings
    output_char_index = dict(zip(output_string, range(len(output_string))))
    contracted_edges = dict()
    # Used for counting how many times a contracted edge's subscript appears among input strings
    batch_edges = dict()
    # Used for counting how many times a batch edge's subscript appears among input strings
    axes_names = dict(zip(range(len(output_string)),
                          [None] * len(output_string)))
    edges = dict(zip(range(len(output_string)),
                     [None] * len(output_string)))
    node1_list = dict(zip(range(len(output_string)),
                          [None] * len(output_string)))
    for i, input_string in enumerate(input_strings):
        for j, char in enumerate(input_string):
            if char not in output_dict:
                edge = nodes[i][j]
                if char not in contracted_edges:
                    contracted_edges[char] = [edge]
                else:
                    if len(contracted_edges[char]) >= 2:
                        raise ValueError(f'Subscript {char} appearing more than once in the '
                                         f'input should be a batch index, but it does not '
                                         f'appear among the output subscripts')
                    if edge != contracted_edges[char][0]:
                        if isinstance(edge, AbstractStackEdge) and \
                                isinstance(contracted_edges[char][0], AbstractStackEdge):
                            edge = edge ^ contracted_edges[char][0]
                        else:
                            raise ValueError(f'Subscript {char} appears in two nodes that do not '
                                             f'share a connected edge at the specified axis')
                    contracted_edges[char] += [edge]
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
                    edge = nodes[i][j]
                    if edge.is_batch():
                        batch_edges[char] = 0
                    k = output_char_index[char]
                    axes_names[k] = nodes[i].axes[j].name
                    edges[k] = edge
                    node1_list[k] = nodes[i].axes[j].node1
                output_dict[char] += 1
                if char in batch_edges:
                    batch_edges[char] += 1

    for char in output_dict:
        if output_dict[char] == 0:
            raise ValueError(f'Output subscript {char} must appear among '
                             f'the input subscripts')
        if output_dict[char] > 1:
            if char in batch_edges:
                if batch_edges[char] < output_dict[char]:
                    raise ValueError(f'Subscript {char} used as batch, but some '
                                     f'of those edge are not batch edges')
            else:
                raise ValueError(f'Subscript {char} used as batch, but none '
                                 f'of those edges is a batch edge')

    for char in contracted_edges:
        if len(contracted_edges[char]) == 1:
            raise ValueError(f'Subscript {char} appears only once in the input '
                             f'but none among the output subscripts')

    input_string = ','.join(input_strings + matrices_strings)
    einsum_string = input_string + '->' + output_string
    tensors = list(map(lambda n: n.tensor, nodes))
    new_tensor = opt_einsum.contract(einsum_string, *(tensors + matrices))

    # We assume all nodes belong to the same network
    new_node = Node(axes_names=list(axes_names.values()),
                    name='einsum_node',
                    network=nodes[0].network,
                    permanent=False,
                    current_op=True,
                    param_edges=False,
                    tensor=new_tensor,
                    edges=list(edges.values()),
                    node1_list=list(node1_list.values()),
                    parents=set(nodes),
                    operation='einsum')
    return new_node


class StackNode(Node):

    def __new__(cls,
                nodes: List[AbstractNode],
                name: Optional[Text] = None,
                override_node: bool = False) -> AbstractNode:
        # Me obligo a poner primero el nombre para que vaya bien el orden de usar __new__ e __init__
        # También tengo que crear el tensor antes de __init__
        stacked_tensor = torch.stack([node.tensor for node in nodes])
        self = super().__new__(cls,
                               name='stacknode',
                               tensor=stacked_tensor,
                               override_node=override_node, network=nodes[0].network,
                               permanent=False,
                               current_op=True,
                               parents=set(nodes),
                               operation='stack')
        return self

    def __init__(self,
                 nodes: List[AbstractNode],
                 name: Optional[Text] = None,
                 override_node: bool = False) -> None:

        if not self.init:
            for i in range(len(nodes[:-1])):
                if not isinstance(nodes[i], type(nodes[i + 1])):
                    raise TypeError('Cannot stack nodes of different types. Nodes '
                                    'must be either all Node or all ParamNode type')
                if nodes[i].shape != nodes[i + 1].shape:
                    raise ValueError('Cannot stack nodes with different shapes')
                if nodes[i].axes_names != nodes[i + 1].axes_names:
                    raise ValueError('Stacked nodes must have the same name for each axis')
                for edge1, edge2 in zip(nodes[i].edges, nodes[i + 1].edges):
                    if not isinstance(edge1, type(edge2)):
                        raise TypeError('Cannot stack nodes with edges of different types. '
                                        'The edges that are attached to the same axis in '
                                        'each node must be either all Edge or all ParamEdge type')

            edges_dict = dict()
            for node in nodes:
                for axis in node.axes:
                    edge = node[axis]
                    if axis.name not in edges_dict:
                        edges_dict[axis.name] = [edge]
                    else:
                        edges_dict[axis.name] += [edge]
            self._edges_dict = edges_dict

            stacked_tensor = torch.stack([node.tensor for node in nodes])
            super().__init__(axes_names=['stack'] + nodes[0].axes_names,
                             name=name,
                             network=nodes[0].network,
                             permanent=False,
                             current_op=True,
                             tensor=stacked_tensor,
                             override_node=override_node,
                             parents=set(nodes),
                             operation='stack')

    @property
    def edges_dict(self) -> Dict[Text, Union[List[Edge], List[ParamEdge]]]:
        return self._edges_dict

    def make_edge(self, axis: Axis) -> Union['Edge', 'ParamEdge']:
        if axis.num == 0:
            return Edge(node1=self, axis1=axis)
        if isinstance(self.edges_dict[axis.name][0], Edge):
            return StackEdge(self.edges_dict[axis.name], node1=self, axis1=axis)
        elif isinstance(self.edges_dict[axis.name][0], ParamEdge):
            return ParamStackEdge(self.edges_dict[axis.name], node1=self, axis1=axis)


class AbstractStackEdge(AbstractEdge):
    """
    Abstract class for stack edges
    """
    @property
    @abstractmethod
    def edges(self) -> List[AbstractEdge]:
        pass


class StackEdge(AbstractStackEdge, Edge):
    """
    Base class for stacks of non-trainable edges.
    Used for stacked contractions
    """
    def __init__(self,
                 edges: List[Edge],
                 node1: StackNode,
                 axis1: Axis,
                 node2: Optional[StackNode] = None,
                 axis2: Optional[Axis] = None) -> None:

        self._edges = edges
        Edge.__init__(self,
                      node1=node1, axis1=axis1,
                      node2=node2, axis2=axis2)

    @property
    def edges(self) -> List[Edge]:
        return self._edges

    def __xor__(self, other: 'StackEdge') -> Edge:
        return connect_stack(self, other)


class ParamStackEdge(AbstractStackEdge, ParamEdge):
    """
        Base class for stacks of trainable edges.
        Used for stacked contractions
        """
    def __init__(self,
                 edges: List[ParamEdge],
                 node1: StackNode,
                 axis1: Axis,
                 node2: Optional[StackNode] = None,
                 axis2: Optional[Axis] = None) -> None:

        self._edges = edges
        ParamEdge.__init__(self,
                           node1=node1, axis1=axis1,
                           shift=self._edges[0].shift,
                           slope=self._edges[0].slope,
                           node2=node2, axis2=axis2)

    @property
    def edges(self) -> List[ParamEdge]:
        return self._edges

    def __xor__(self, other: 'ParamStackEdge') -> ParamEdge:
        return connect_stack(self, other)


def connect_stack(edge1: AbstractStackEdge,
                  edge2: AbstractStackEdge,
                  override_network: bool = False):
    """
    Connect stack edges only if their lists of edges are the same
    (coming from already connected edges)
    """
    if edge1.edges != edge2.edges:
        raise ValueError('Cannot connect stack edges whose lists of'
                         ' edges are not the same. They will be the '
                         'same when both lists contain edges connecting'
                         ' the nodes that formed the stack nodes.')
    if isinstance(edge1.edges[0], ParamEdge):
        shift = edge1.edges[0].shift
        slope = edge1.edges[0].slope
        params_dict = dict()
        # When connecting stacked edges, parameters have to be
        # shared among all edges in the same ParamStackEdge
        for i, _ in enumerate(edge1.edges[:-1]):
            if edge1.edges[i].dim() != edge1.edges[i + 1].dim():
                raise ValueError('Cannot connect stacked edges with lists of edges '
                                 'of different dimensions')
            edge1.edges[i + 1].set_parameters(shift=shift, slope=slope)

    return connect(edge1=edge1, edge2=edge2,
                   override_network=override_network)


def stack(nodes: List[AbstractNode], name: Optional[Text] = None) -> StackNode:
    """
    Stack nodes into a StackNode. The stack dimension will be the
    first one in the resultant node
    """
    # TODO: override_node = True para solo cambiar el tensor
    self = StackNode(nodes, name=name)
    return self


def unbind(node: AbstractNode) -> List[Node]:
    """
    Unbind stacked node. It is assumed that the stacked dimension
    is the first one
    """
    tensors_list = torch.unbind(node.tensor)
    nodes = []
    start = time.time()
    lst_times = []

    # for i, tensor in enumerate(tensors_list):
    #     start2 = time.time()
    #     new_node = Node(name='unbind_node',
    #                     axes_names=node.axes_names[1:],
    #                     network=node.network,
    #                     permanent=False,
    #                     current_op=True,
    #                     tensor=tensor,
    #                     edges=[edge.edges[i] if isinstance(edge, AbstractStackEdge)
    #                            else edge for edge in node.edges[1:]],
    #                     node1_list=node.node1_list[1:],
    #                     parents={node},
    #                     operation=f'unbind_{i}')
    #     nodes.append(new_node)
    #     lst_times.append(time.time() - start2)
    # print('\t\t\t\t\tCreate 1 node:',
    #       torch.tensor(lst_times).mean(),
    #       torch.tensor(lst_times).min(0),
    #       torch.tensor(lst_times).max(0),
    #       torch.tensor(lst_times).median(0),
    #       len(tensors_list))

    # Invert structure of node.edges
    # TODO: just 1 sec faster per epoch
    is_stack_edge = list(map(lambda e: isinstance(e, AbstractStackEdge), node.edges[1:]))
    edges_to_zip = []
    for i, edge in enumerate(node.edges[1:]):
        if is_stack_edge[i]:
            edges_to_zip.append(edge.edges)
        else:
            edges_to_zip.append([edge] * len(tensors_list))

    lst = list(zip(*([tensors_list, list(zip(*edges_to_zip))])))

    for i, (tensor, edges) in enumerate(lst):
        start2 = time.time()
        new_node = Node(name='unbind_node',
                        axes_names=node.axes_names[1:],
                        network=node.network,
                        permanent=False,
                        current_op=True,
                        tensor=tensor,
                        edges=list(edges),
                        node1_list=node.node1_list[1:],
                        parents={node},
                        operation=f'unbind_{i}')
        nodes.append(new_node)
        lst_times.append(time.time() - start2)
    # print('\t\t\t\t\tCreate 1 node:',
    #       torch.tensor(lst_times).mean(),
    #       torch.tensor(lst_times).min(0),
    #       torch.tensor(lst_times).max(0),
    #       torch.tensor(lst_times).median(0),
    #       len(tensors_list))

    #print('\t\t\t\tCreate nodes:', time.time() - start)
    return nodes


def stacked_einsum(string: Text, *nodes_lists: List[AbstractNode]) -> List[Node]:
    """
    Adapt einsum operation to enable stack contractions. Lists of input nodes
    are stacked and a special character is added before each input/output string

    Parameters
    ----------
    string: einsum-like string built as if only the set formed by the first node
            of each list in `nodes_lists` were involved in the operation
    nodes_lists: lists of nodes involved in the operation. Each element of
                 each list will be operated with the corresponding elements
                 from the other lists. The nodes in each list will be first
                 stacked to perform the einsum operation

    Returns
    -------
    unbind(result): list of nodes resultant from operating the stack nodes and
                    unbind the result
    """
    start = time.time()
    stacks_list = []
    for nodes_list in nodes_lists:
        stacks_list.append(stack(nodes_list))
    #print('\t\t\tStacks:', time.time() - start)

    input_strings = string.split('->')[0].split(',')
    output_string = string.split('->')[1]

    i = 0
    stack_char = opt_einsum.get_symbol(i)
    for input_string in input_strings:
        for input_char in input_string:
            if input_char == stack_char:
                i += 1
                stack_char = opt_einsum.get_symbol(i)
    input_strings = list(map(lambda s: stack_char + s, input_strings))
    input_string = ','.join(input_strings)
    output_string = stack_char + output_string
    string = input_string + '->' + output_string

    start = time.time()
    result = einsum(string, *stacks_list)
    #print('\t\t\tEinsum:', time.time() - start)
    start = time.time()
    unbinded_result = unbind(result)  # <-- Lo más lento
    #print('\t\t\tUnbind:', time.time() - start)
    return unbinded_result
