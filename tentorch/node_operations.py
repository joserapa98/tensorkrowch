"""
This script contains:

    Operations:
        *connect
        *disconnect
        *get_shared_edges
        *get_batch_edges
        *contract_edges
        *contract
        *contract_between

    Node operations:
        *einsum
        *connect_stack
        *stack
        *unbind
        *stacked_einsum
"""
# split, svd, qr, rq, etc. -> using einsum-like strings, useful

from typing import Union, Optional, Text, List, Dict, Tuple, Any, Callable
from abc import abstractmethod

import torch
import opt_einsum

from tentorch.utils import permute_list, inverse_permutation

import tentorch.network_components as nc

import time

AbstractNode, Node, ParamNode = Any, Any, Any
AbstractEdge, Edge, ParamEdge = Any, Any, Any
StackNode = Any
AbstractStackEdge, StackEdge, ParamStackEdge = Any, Any, Any
Successor = Any


################################################
#               EDGE OPERATIONS                #
################################################
def connect(edge1: AbstractEdge, edge2: AbstractEdge) -> Union[Edge, ParamEdge]:
    """
    Connect two dangling, non-batch edges.
    """
    # TODO: no puedo capar el conectar nodos no-leaf, pero no tiene el resultado esperado,
    #  en realidad estás conectando los nodos originales (leaf)
    for edge in [edge1, edge2]:
        if not edge.is_dangling():
            raise ValueError(f'Edge {edge!s} is not a dangling edge. '
                             f'This edge points to nodes: {edge.node1!s} and {edge.node2!s}')
        if edge.is_batch():
            raise ValueError(f'Edge {edge!s} is a batch edge')
    if edge1 == edge2:
        raise ValueError(f'Cannot connect edge {edge1!s} to itself')
    if edge1.dim() != edge2.dim():
        raise ValueError(f'Cannot connect edges of unequal dimension. '
                         f'Dimension of edge {edge1!s}: {edge1.dim()}. '
                         f'Dimension of edge {edge2!s}: {edge2.dim()}')
    if edge1.size() != edge2.size():
        # Keep the minimum size
        if edge1.size() < edge2.size():
            edge2.change_size(edge1.size())
        elif edge1.size() > edge2.size():
            edge1.change_size(edge2.size())

    node1, axis1 = edge1.node1, edge1.axis1
    node2, axis2 = edge2.node1, edge2.axis1
    net1, net2 = node1._network, node2._network

    if net1 != net2:
        node2.move_to_network(net1)
    net1._remove_edge(edge1)
    net1._remove_edge(edge2)
    net = net1

    if isinstance(edge1, nc.ParamEdge) == isinstance(edge2, nc.ParamEdge):
        if isinstance(edge1, nc.ParamEdge):
            shift = edge1.shift
            slope = edge1.slope
            new_edge = nc.ParamEdge(node1=node1, axis1=axis1,
                                    shift=shift, slope=slope,
                                    node2=node2, axis2=axis2)
            net._add_edge(new_edge)
        else:
            new_edge = nc.Edge(node1=node1, axis1=axis1,
                               node2=node2, axis2=axis2)
    else:
        if isinstance(edge1, nc.ParamEdge):
            shift = edge1.shift
            slope = edge1.slope
        else:
            shift = edge2.shift
            slope = edge2.slope
        new_edge = nc.ParamEdge(node1=node1, axis1=axis1,
                                shift=shift, slope=slope,
                                node2=node2, axis2=axis2)
        net._add_edge(new_edge)

    node1._add_edge(new_edge, axis1, True)
    node2._add_edge(new_edge, axis2, False)
    return new_edge


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
    if isinstance(edge1.edges[0], nc.ParamEdge):
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

    return connect(edge1=edge1, edge2=edge2)


def disconnect(edge: Union[Edge, ParamEdge]) -> Tuple[Union[Edge, ParamEdge],
                                                      Union[Edge, ParamEdge]]:
    """
    Disconnect an edge, returning a couple of dangling edges
    """
    if edge.is_dangling():
        raise ValueError('Cannot disconnect a dangling edge')

    node1, node2 = edge.node1, edge.node2
    axis1, axis2 = edge.axis1, edge.axis2
    if isinstance(edge, nc.Edge):
        new_edge1 = nc.Edge(node1=node1, axis1=axis1)
        new_edge2 = nc.Edge(node1=node2, axis1=axis2)
        net = edge.node1._network
        net._add_edge(new_edge1)
        net._add_edge(new_edge2)
    else:
        assert isinstance(edge, nc.ParamEdge)
        shift = edge.shift
        slope = edge.slope
        new_edge1 = nc.ParamEdge(node1=node1, axis1=axis1,
                                 shift=shift, slope=slope)
        new_edge2 = nc.ParamEdge(node1=node2, axis1=axis2,
                                 shift=shift, slope=slope)
        net = edge.node1._network
        net._remove_edge(edge)
        net._add_edge(new_edge1)
        net._add_edge(new_edge2)

    node1._add_edge(new_edge1, axis1, True)
    node2._add_edge(new_edge2, axis2, True)
    return new_edge1, new_edge2


################################################
#               NODE OPERATIONS                #
################################################
# TODO: otra opcion: successors tuplas (kwargs, operation), si los nodos padres
#  coinciden en kwargs (ya sucedio la operacion), operation guarda el objeto
#  operacion optimizada para tensores
class Operation:

    def __init__(self, check_first, func1, func2):
        assert isinstance(check_first, Callable)
        assert isinstance(func1, Callable)
        assert isinstance(func2, Callable)
        self.func1 = func1
        self.func2 = func2
        self.check_first = check_first

    def __call__(self, *args, **kwargs):
        successor = self.check_first(*args, **kwargs)
        if successor is None:
            return self.func1(*args, **kwargs)
        else:
            return self.func2(successor=successor, *args, **kwargs)


#################   CONTRACT   #################
def _check_first_contract_edges(edges: List[AbstractEdge],
                                node1: AbstractNode,
                                node2: AbstractNode) -> Optional[Successor]:
    kwargs = {'edges': edges,
              'node1': node1,
              'node2': node2}
    if 'contract_edges' in node1._network._successors:
        for succ in node1.network._successors['contract_edges']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _contract_edges_first(edges: List[AbstractEdge],
                          node1: AbstractNode,
                          node2: AbstractNode) -> Node:
    """
    Contract edges between two nodes for the first time.

    Parameters
    ----------
    edges: list of edges that are to be contracted. They must be edges
        shared between `node1` and `node2`. Batch contraction is automatically
        computed if each node has one batch edge that share the same name
    node1: first node of the contraction
    node2: second node of the contraction

    Returns
    -------
    new_node: Node resultant from the contraction
    """

    shared_edges = get_shared_edges(node1, node2)
    for edge in edges:
        if edge not in shared_edges:
            raise ValueError('Edges selected to be contracted must be shared '
                             'edges between `node1` and `node2`')

    if node1 == node2:
        result = node1.tensor
        axes_nums = dict(zip(range(node1.rank), range(node1.rank)))
        for edge in edges:
            result = torch.diagonal(result,
                                    offset=0,
                                    dim1=axes_nums[edge.axis1.num],
                                    dim2=axes_nums[edge.axis2.num]).sum(-1)
            min_axis = min(edge.axis1.num, edge.axis2.num)
            max_axis = max(edge.axis1.num, edge.axis2.num)
            for num in axes_nums:
                if num < min_axis:
                    continue
                elif num == min_axis:
                    axes_nums[num] = -1
                elif (num > min_axis) and (num < max_axis):
                    axes_nums[num] -= 1
                elif num == max_axis:
                    axes_nums[num] = -1
                elif num > max_axis:
                    axes_nums[num] -= 2

        new_axes_names = []
        new_edges = []
        new_node1_list = []
        for num in axes_nums:
            if axes_nums[num] >= 0:
                new_axes_names.append(node1._axes[num]._name)
                new_edges.append(node1._edges[num])
                new_node1_list.append(node1.is_node1(num))

        hints = None

    else:
        # TODO: si son StackEdge, ver que todos los correspondientes edges están conectados


        nodes = [node1, node2]
        tensors = [node1.tensor, node2.tensor]
        non_contract_edges = [dict(), dict()]
        batch_edges = dict()
        contract_edges = dict()

        for i in [0, 1]:
            for j, edge in enumerate(nodes[i]._edges):
                if edge in edges:
                    if i == 0:
                        if isinstance(edge, nc.ParamEdge):
                            # Obtain permutations
                            permutation_dims = [k if k < j else k + 1
                                                for k in range(nodes[i].rank - 1)] + [j]
                            inv_permutation_dims = inverse_permutation(permutation_dims)

                            # Send multiplication dimension to the end, multiply, recover original shape
                            tensors[i] = tensors[i].permute(permutation_dims)
                            tensors[i] = tensors[i] @ edge.matrix
                            tensors[i] = tensors[i].permute(inv_permutation_dims)

                        contract_edges[edge] = [tensors[i].shape[j]]

                    contract_edges[edge].append(j)

                elif edge.is_batch():
                    if i == 0:
                        batch_in_node2 = False
                        for aux_edge in nodes[1]._edges:
                            if aux_edge.is_batch() and (edge.axis1._name == aux_edge.axis1._name):
                                batch_edges[edge.axis1._name] = [tensors[0].shape[j], j]
                                batch_in_node2 = True
                                break

                        if not batch_in_node2:
                            non_contract_edges[i][edge] = [tensors[i].shape[j], j]

                    else:
                        if edge.axis1._name in batch_edges:
                            batch_edges[edge.axis1._name].append(j)
                        else:
                            non_contract_edges[i][edge] = [tensors[i].shape[j], j]

                else:
                    non_contract_edges[i][edge] = [tensors[i].shape[j], j]

        # TODO: esto seguro que se puede hacer mejor
        permutation_dims = [None, None]
        permutation_dims[0] = list(map(lambda l: l[1], batch_edges.values())) + \
                              list(map(lambda l: l[1], non_contract_edges[0].values())) + \
                              list(map(lambda l: l[1], contract_edges.values()))
        permutation_dims[1] = list(map(lambda l: l[2], batch_edges.values())) + \
                              list(map(lambda l: l[2], contract_edges.values())) + \
                              list(map(lambda l: l[1], non_contract_edges[1].values()))

        aux_permutation = inverse_permutation(list(map(lambda l: l[1], batch_edges.values())) +
                                              list(map(lambda l: l[1], non_contract_edges[0].values())))
        aux_permutation2 = inverse_permutation(list(map(lambda l: l[1], non_contract_edges[1].values())))
        inv_permutation_dims = aux_permutation + list(map(lambda x: x + len(aux_permutation), aux_permutation2))

        aux_shape = [None, None]
        aux_shape[0] = (torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
                        torch.tensor(list(map(lambda l: l[0], non_contract_edges[0].values()))).prod().long().item(),
                        torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item())

        aux_shape[1] = (torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
                        torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item(),
                        torch.tensor(list(map(lambda l: l[0], non_contract_edges[1].values()))).prod().long().item())

        new_shape = list(map(lambda l: l[0], batch_edges.values())) + \
                    list(map(lambda l: l[0], non_contract_edges[0].values())) + \
                    list(map(lambda l: l[0], non_contract_edges[1].values()))

        for i in [0, 1]:
            tensors[i] = tensors[i].permute(permutation_dims[i])
            tensors[i] = tensors[i].reshape(aux_shape[i])

        result = tensors[0] @ tensors[1]
        result = result.view(new_shape).permute(inv_permutation_dims)

        indices = [None, None]
        indices[0] = permute_list(list(map(lambda l: l[1], batch_edges.values())) +
                                  list(map(lambda l: l[1], non_contract_edges[0].values())),
                                  aux_permutation)
        indices[1] = list(map(lambda l: l[1], non_contract_edges[1].values()))

        new_axes_names = []
        new_edges = []
        new_node1_list = []
        for i in [0, 1]:
            for idx in indices[i]:
                new_axes_names.append(nodes[i].axes_names[idx])
                new_edges.append(nodes[i][idx])
                new_node1_list.append(nodes[i].axes[idx].is_node1())

        hints = {'permutation_dims': permutation_dims,
                 'inv_permutation_dims': inv_permutation_dims,
                 'aux_shape': aux_shape,
                 'new_shape': new_shape}

    new_node = nc.Node(axes_names=new_axes_names,
                       name=f'contract_{node1._name}_{node2._name}',
                       network=node1._network,
                       leaf=False,
                       param_edges=False,
                       tensor=result,
                       edges=new_edges,
                       node1_list=new_node1_list)

    net = node1._network
    successor = nc.Successor(kwargs={'edges': edges,
                                     'node1': node1,
                                     'node2': node2},
                             child=new_node,
                             hints=hints)
    if 'contract_edges' in net._successors:
        net._successors['contract_edges'].append(successor)
    else:
        net._successors['contract_edges'] = [successor]

    return new_node


def _contract_edges_next(successor: Successor,
                         edges: List[AbstractEdge],
                         node1: AbstractNode,
                         node2: AbstractNode) -> Node:
    """
    Contract edges between two nodes.
    """

    if node1 == node2:
        result = node1.tensor
        axes_nums = dict(zip(range(node1.rank), range(node1.rank)))
        for edge in edges:
            result = torch.diagonal(result,
                                    offset=0,
                                    dim1=axes_nums[edge.axis1.num],
                                    dim2=axes_nums[edge.axis2.num]).sum(-1)
            min_axis = min(edge.axis1.num, edge.axis2.num)
            max_axis = max(edge.axis1.num, edge.axis2.num)
            for num in axes_nums:
                if num < min_axis:
                    continue
                elif num == min_axis:
                    axes_nums[num] = -1
                elif (num > min_axis) and (num < max_axis):
                    axes_nums[num] -= 1
                elif num == max_axis:
                    axes_nums[num] = -1
                elif num > max_axis:
                    axes_nums[num] -= 2

    else:
        # TODO: si son StackEdge, ver que todos los correspondientes edges están conectados

        nodes = [node1, node2]
        tensors = [node1.tensor, node2.tensor]

        for i, edge in enumerate(edges):
            if isinstance(edge, nc.ParamEdge):
                # Obtain permutations
                permutation_dims = [k if k < i else k + 1
                                    for k in range(nodes[0].rank - 1)] + [i]
                inv_permutation_dims = inverse_permutation(permutation_dims)

                # Send multiplication dimension to the end, multiply, recover original shape
                tensors[0] = tensors[0].permute(permutation_dims)
                tensors[0] = tensors[0] @ edge.matrix
                tensors[0] = tensors[0].permute(inv_permutation_dims)

        hints = successor.hints
        for i in [0, 1]:
            tensors[i] = tensors[i].permute(hints['permutation_dims'][i])
            tensors[i] = tensors[i].reshape(hints['aux_shape'][i])

        result = tensors[0] @ tensors[1]
        result = result.view(hints['new_shape']).permute(hints['inv_permutation_dims'])

    child = successor.child
    child.tensor = result

    return child


contract_edges = Operation(_check_first_contract_edges,
                           _contract_edges_first,
                           _contract_edges_next)


def contract(edge: AbstractEdge) -> Node:
    """
    Contract only one edge
    """
    return contract_edges([edge], edge.node1, edge.node2)


def get_shared_edges(node1: AbstractNode, node2: AbstractNode) -> List[AbstractEdge]:
    """
    Obtain list of edges shared between two nodes
    """
    edges = set()
    for i1, edge1 in enumerate(node1._edges):
        for i2, edge2 in enumerate(node2._edges):
            if (edge1 == edge2) and not edge1.is_dangling():
                if node1.is_node1(i1) != node2.is_node1(i2):
                    edges.add(edge1)
    return list(edges)


def contract_between(node1: AbstractNode, node2: AbstractNode) -> Node:
    """
    Contract all shared edges between two nodes, also performing batch contraction
    between batch edges that share name in both nodes
    """
    edges = get_shared_edges(node1, node2)
    if not edges:
        raise ValueError(f'No batch edges neither shared edges between '
                         f'nodes {node1!s} and {node2!s} found')
    return contract_edges(edges, node1, node2)


###################   STACK   ##################
def stack_unequal_tensors(lst_tensors: List[torch.Tensor]) -> torch.Tensor:
    if lst_tensors:
        same_dims_all = True
        max_shape = list(lst_tensors[0].shape)
        for tensor in lst_tensors[1:]:
            same_dims = True
            for idx, dim in enumerate(tensor.shape):
                if dim > max_shape[idx]:
                    max_shape[idx] = dim
                    same_dims = False
                elif dim < max_shape[idx]:
                    same_dims = False
            if not same_dims:
                same_dims_all = False

        if not same_dims_all:
            for idx, tensor in enumerate(lst_tensors):
                if tensor.shape != max_shape:
                    aux_zeros = torch.zeros(max_shape, device=tensor.device)  # TODO: replace with pad
                    replace_slice = []
                    for dim in tensor.shape:
                        replace_slice.append(slice(0, dim))
                    replace_slice = tuple(replace_slice)
                    aux_zeros[replace_slice] = tensor
                    lst_tensors[idx] = aux_zeros
        return torch.stack(lst_tensors)


def _check_first_stack(nodes: List[AbstractNode], name: Optional[Text] = None) -> bool:
    kwargs = {'nodes': set(nodes)}
    if 'stack' in nodes[0].network._successors:
        for t in nodes[0].network._successors['stack']:
            if t[0] == kwargs:
                return False
    return True


def _stack_first(nodes: List[AbstractNode], name: Optional[Text] = None) -> StackNode:
    """
        Stack nodes into a StackNode. The stack dimension will be the
        first one in the resultant node
        """
    all_leaf = True
    for node in nodes:
        if not node.is_leaf():
            all_leaf = False
            break

    stack_node = nc.StackNode(nodes, name=name)
    # TODO: Crear ParamStackNode, para cuando todos son leaf y
    #  guardamos los parámetros en la pila de la que leemos cada tensor

    if all_leaf:
        net = nodes[0].network
        for i, node in enumerate(nodes):
            shape = node.shape
            if node._tensor_info['address'] is not None:
                del net._memory_nodes[node._tensor_info['address']]
            node._tensor_info['address'] = None
            node._tensor_info['node_ref'] = stack_node
            node._tensor_info['full'] = False
            node._tensor_info['stack_idx'] = i
            index = [i]
            for s in shape:
                index.append(slice(0, s))
            node._tensor_info['index'] = index

            if 'stack' not in node.network._successors:
                node.network._successors['stack'] = [({'nodes': nodes}, stack_node)]
            else:
                node.network._successors['stack'].append(({'nodes': nodes}, stack_node))

    return stack_node


def _stack_next(nodes: List[AbstractNode], name: Optional[Text] = None) -> StackNode:
    all_leaf = True
    for node in nodes:
        if not node.is_leaf():
            all_leaf = False
            break

    kwargs = {'nodes': set(nodes)}
    for t in nodes[0]._network._successors['stack']:
        if t[0] == kwargs:
            child = t[1]
            break

    if all_leaf:
        return child

    child.tensor = torch.stack([node.tensor for node in nodes])
    return child


stack = Operation(_check_first_stack, _stack_first, _stack_next)


##################   UNBIND   ##################
def _check_first_unbind(node: AbstractNode) -> bool:
    kwargs = {'node': node}
    if 'unbind' in node._network._successors:
        for t in node._network._successors['unbind']:
            if t[0] == kwargs:
                return False
    return True


def _unbind_first(node: AbstractNode) -> List[Node]:
    """
    Unbind stacked node. It is assumed that the stacked dimension
    is the first one
    """
    tensors_list = torch.unbind(node.tensor)
    nodes = []
    start = time.time()
    lst_times = []

    # Invert structure of node.edges
    # TODO: just 1 sec faster per epoch
    is_stack_edge = list(map(lambda e: isinstance(e, nc.AbstractStackEdge), node.edges[1:]))
    edges_to_zip = []
    for i, edge in enumerate(node.edges[1:]):
        if is_stack_edge[i]:
            edges_to_zip.append(edge.edges)
        else:
            edges_to_zip.append([edge] * len(tensors_list))

    lst = list(zip(*([tensors_list, list(zip(*edges_to_zip))])))

    for i, (tensor, edges) in enumerate(lst):
        start2 = time.time()
        new_node = nc.Node(axes_names=node.axes_names[1:], name='unbind_node', network=node.network, leaf=False,
                        tensor=tensor, edges=list(edges), node1_list=node.is_node1()[1:])
        # TODO: arreglar node1_list, deber'iamos lllevarlo guardado tambi'en en el StackNode
        nodes.append(new_node)
        lst_times.append(time.time() - start2)

    for i, new_node in enumerate(nodes):
        shape = new_node.shape
        if new_node._tensor_info['address'] is not None:
            del new_node.network._memory_nodes[new_node._tensor_info['address']]
        new_node._tensor_info['address'] = None
        new_node._tensor_info['node_ref'] = node
        new_node._tensor_info['full'] = False
        new_node._tensor_info['stack_idx'] = i
        index = [i]
        for s in shape:
            index.append(slice(0, s))
        new_node._tensor_info['index'] = index

    if 'unbind' not in node.network._successors:
        node.network._successors['unbind'] = [({'nodes': nodes}, nodes)]
    else:
        node.network._successors['unbind'].append(({'nodes': nodes}, nodes))

    return nodes


def _unbind_next(node: AbstractNode) -> List[Node]:
    kwargs = {'node': node}
    for t in node.network._successors['unbind']:
        if t[0] == kwargs:
            return t[1]  # TODO: No tenemos que hacer nada si ya hicimos unbind antes


unbind = Operation(_check_first_unbind, _unbind_first, _unbind_next)


##################   OTHERS   ##################
# TODO: más adelante, no prioritario

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
                        if isinstance(edge, nc.AbstractStackEdge) and \
                                isinstance(contracted_edges[char][0], nc.AbstractStackEdge):
                            edge = edge ^ contracted_edges[char][0]
                        else:
                            raise ValueError(f'Subscript {char} appears in two nodes that do not '
                                             f'share a connected edge at the specified axis')
                    contracted_edges[char] += [edge]
                if isinstance(edge, nc.ParamEdge):
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
                    node1_list[k] = nodes[i].axes[j].is_node1()
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
                                     f'of those edges are not batch edges')
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
    new_node = nc.Node(axes_names=list(axes_names.values()), name='einsum_node', network=nodes[0].network, leaf=False,
                    param_edges=False, tensor=new_tensor, edges=list(edges.values()),
                    node1_list=list(node1_list.values()))
    return new_node


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
