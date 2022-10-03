"""
This script contains:

    Edge operations:
        *connect
        *connect_stack
        *disconnect

    Node operations:
        Class for node operations:

            *Operation:

                (Basic operations)
                +permute
                +tprod
                +mul
                +add
                +sub

                (Contract)
                +contract_edges
            *contract
            *get_shared_edges
            *contract_between

                (Stack)
                +stack

                (Unbind)
                +unbind

    Other operations:
        *einsum
        *stacked_einsum
"""
# split, svd, qr, rq, etc. -> using einsum-like strings, useful

from typing import Union, Optional, Text, List, Dict, Tuple, Any, Callable, Sequence
from abc import abstractmethod

import torch
import torch.nn as nn
import opt_einsum

from tentorch.utils import is_permutation, permute_list, inverse_permutation

import tentorch.network_components as nc

import time

Axis = Any
Ax = Union[int, Text, Axis]

AbstractNode, Node, ParamNode = Any, Any, Any
AbstractEdge, Edge, ParamEdge = Any, Any, Any
StackNode = Any
AbstractStackEdge, StackEdge, ParamStackEdge = Any, Any, Any
Successor = Any
# TODO: hacer import Tensor, Parameter?

PRINT_MODE = False
CHECK_TIMES = []


################################################
#               EDGE OPERATIONS                #
################################################
def connect(edge1: AbstractEdge, edge2: AbstractEdge) -> Union[Edge, ParamEdge]:
    """
    Connect two dangling, non-batch edges.
    """
    # TODO: no puedo capar el conectar nodos no-leaf, pero no tiene el resultado esperado,
    #  en realidad estás conectando los nodos originales (leaf)
    if edge1 == edge2:
        return edge1

    for edge in [edge1, edge2]:
        if not edge.is_dangling():
            raise ValueError(f'Edge {edge!s} is not a dangling edge. '
                             f'This edge points to nodes: {edge.node1!s} and {edge.node2!s}')
        if edge.is_batch():
            raise ValueError(f'Edge {edge!s} is a batch edge')
    # if edge1 == edge2:
    #     raise ValueError(f'Cannot connect edge {edge1!s} to itself')
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


def connect_stack(edge1: AbstractStackEdge, edge2: AbstractStackEdge):
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
        # When connecting stacked edges, parameters have to be
        # shared among all edges in the same ParamStackEdge
        for i, _ in enumerate(edge1.edges[:-1]):
            if edge1.edges[i].dim() != edge1.edges[i + 1].dim():
                raise ValueError('Cannot connect stacked edges with lists of edges '
                                 'of different dimensions')
            edge1.edges[i + 1].set_parameters(shift=shift, slope=slope)  # TODO: sure? We want this? Share parameters?

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
class Operation:

    def __init__(self, check_first, func1, func2):
        assert isinstance(check_first, Callable)
        assert isinstance(func1, Callable)
        assert isinstance(func2, Callable)
        self.func1 = func1
        self.func2 = func2
        self.check_first = check_first

    def __call__(self, *args, **kwargs):
        start = time.time()
        successor = self.check_first(*args, **kwargs)
        # if PRINT_MODE:
        #     diff = time.time() - start
        #     print('Check:', diff)
        #     global CHECK_TIMES
        #     CHECK_TIMES.append(diff)

        if successor is None:
            return self.func1(*args, **kwargs)
        else:
            args = [successor] + list(args)
            return self.func2(*args, **kwargs)


#################   BASIC OP   #################
def _check_first_permute(node: AbstractNode, axes: Sequence[Ax]) -> Optional[Successor]:
    kwargs = {'node': node,
              'axes': axes}
    if 'permute' in node._network._successors:
        for succ in node.network._successors['permute']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _permute_first(node: AbstractNode, axes: Sequence[Ax]) -> Node:
    axes_nums = []
    for axis in axes:
        axes_nums.append(node.get_axis_number(axis))

    if not is_permutation(list(range(len(axes_nums))), axes_nums):
        raise ValueError('The provided list of axis is not a permutation of the'
                         ' axes of the node')
    else:
        new_node = nc.Node(axes_names=permute_list(node.axes_names, axes_nums),
                           name='permute_' + node._name,
                           network=node._network,
                           param_edges=node.param_edges(),
                           tensor=node.tensor.permute(axes_nums),
                           edges=permute_list(node._edges, axes_nums),
                           node1_list=permute_list(node.is_node1(), axes_nums))

    net = node._network
    successor = nc.Successor(kwargs={'node': node,
                                     'axes': axes},
                             child=new_node,
                             hints=axes_nums)
    if 'permute' in net._successors:
        net._successors['permute'].append(successor)
    else:
        net._successors['permute'] = [successor]

    net._list_ops.append(('permute', len(net._successors['permute']) - 1))

    return new_node


def _permute_next(successor: Successor, node: AbstractNode, axes: Sequence[Ax]) -> Node:
    new_tensor = node.tensor.permute(successor.hints)
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


def _check_first_tprod(node1: AbstractNode, node2: AbstractNode) -> Optional[Successor]:
    kwargs = {'node1': node1,
              'node2': node2}
    if 'tprod' in node1._network._successors:
        for succ in node1.network._successors['tprod']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _tprod_first(node1: AbstractNode, node2: AbstractNode) -> Node:
    if node1._network != node2._network:
        raise ValueError('Nodes must be in the same network')
    if node2 in node1.neighbours():
        raise ValueError('Tensor product cannot be performed between connected nodes')

    new_tensor = torch.outer(node1.tensor.flatten(),
                             node2.tensor.flatten()).view(*(list(node1.shape) +
                                                            list(node2.shape)))
    new_node = nc.Node(axes_names=node1.axes_names + node2.axes_names,
                       name=f'tprod_{node1._name}_{node2._name}',
                       network=node1._network,
                       leaf=False,
                       tensor=new_tensor,
                       edges=node1._edges + node2._edges,
                       node1_list=node1.is_node1() + node2.is_node1())

    net = node1._network
    successor = nc.Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'tprod' in net._successors:
        net._successors['tprod'].append(successor)
    else:
        net._successors['tprod'] = [successor]

    net._list_ops.append(('tprod', len(net._successors['tprod']) - 1))

    return new_node


def _tprod_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = torch.outer(node1.tensor.flatten(),
                             node2.tensor.flatten()).view(*(list(node1.shape) +
                                                            list(node2.shape)))
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


def _check_first_mul(node1: AbstractNode, node2: AbstractNode) -> Optional[Successor]:
    kwargs = {'node1': node1,
              'node2': node2}
    if 'mul' in node1._network._successors:
        for succ in node1.network._successors['mul']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _mul_first(node1: AbstractNode, node2: AbstractNode) -> Node:
    if node1._network != node2._network:
        raise ValueError('Nodes must be in the same network')

    new_tensor = node1.tensor * node2.tensor
    new_node = nc.Node(axes_names=node1.axes_names,
                       name=f'mul_{node1._name}_{node2._name}',
                       network=node1._network,
                       leaf=False,
                       tensor=new_tensor)

    net = node1._network
    successor = nc.Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'mul' in net._successors:
        net._successors['mul'].append(successor)
    else:
        net._successors['mul'] = [successor]

    net._list_ops.append(('mul', len(net._successors['mul']) - 1))

    return new_node


def _mul_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = node1.tensor * node2.tensor
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


def _check_first_add(node1: AbstractNode, node2: AbstractNode) -> Optional[Successor]:
    kwargs = {'node1': node1,
              'node2': node2}
    if 'add' in node1._network._successors:
        for succ in node1.network._successors['add']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _add_first(node1: AbstractNode, node2: AbstractNode) -> Node:
    if node1._network != node2._network:
        raise ValueError('Nodes must be in the same network')

    new_tensor = node1.tensor + node2.tensor
    new_node = nc.Node(axes_names=node1.axes_names,
                       name=f'add_{node1._name}_{node2._name}',
                       network=node1._network,
                       leaf=False,
                       tensor=new_tensor)

    net = node1._network
    successor = nc.Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'add' in net._successors:
        net._successors['add'].append(successor)
    else:
        net._successors['add'] = [successor]

    net._list_ops.append(('add', len(net._successors['add']) - 1))

    return new_node


def _add_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = node1.tensor + node2.tensor
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


def _check_first_sub(node1: AbstractNode, node2: AbstractNode) -> Optional[Successor]:
    kwargs = {'node1': node1,
              'node2': node2}
    if 'sub' in node1._network._successors:
        for succ in node1.network._successors['sub']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _sub_first(node1: AbstractNode, node2: AbstractNode) -> Node:
    if node1._network != node2._network:
        raise ValueError('Nodes must be in the same network')

    new_tensor = node1.tensor - node2.tensor
    new_node = nc.Node(axes_names=node1.axes_names,
                       name=f'sub_{node1._name}_{node2._name}',
                       network=node1._network,
                       leaf=False,
                       tensor=new_tensor)

    net = node1._network
    successor = nc.Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'sub' in net._successors:
        net._successors['sub'].append(successor)
    else:
        net._successors['sub'] = [successor]

    net._list_ops.append(('sub', len(net._successors['sub']) - 1))

    return new_node


def _sub_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = node1.tensor * node2.tensor
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


permute = Operation(_check_first_permute, _permute_first, _permute_next)
tprod = Operation(_check_first_tprod, _tprod_first, _tprod_next)
mul = Operation(_check_first_mul, _mul_first, _mul_next)
add = Operation(_check_first_add, _add_first, _add_next)
sub = Operation(_check_first_sub, _sub_first, _sub_next)


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
        for j, edge in enumerate(node1._edges):
            if edge in edges:
                if isinstance(edge, nc.ParamEdge):
                    # Obtain permutations
                    permutation_dims = [k if k < j else k + 1
                                        for k in range(node1.rank - 1)] + [j]
                    inv_permutation_dims = inverse_permutation(permutation_dims)

                    # Send multiplication dimension to the end, multiply, recover original shape
                    result = result.permute(permutation_dims)
                    result = result @ edge.matrix
                    result = result.permute(inv_permutation_dims)

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
                                if 'batch' in edge.axis1._name:  # TODO: restringir a solo un edge de batch!!
                                    batch_edges[edge.axis1._name] = [-1, j]
                                else:
                                    batch_edges[edge.axis1._name] = [tensors[0].shape[j], j]
                                batch_in_node2 = True
                                break

                        if not batch_in_node2:
                            if 'batch' in edge.axis1._name:  # TODO: restringir a solo un edge de batch!!
                                non_contract_edges[i][edge] = [-1, j]
                            else:
                                non_contract_edges[i][edge] = [tensors[i].shape[j], j]

                    else:
                        if edge.axis1._name in batch_edges:
                            batch_edges[edge.axis1._name].append(j)
                        else:
                            if 'batch' in edge.axis1._name:  # TODO: restringir a solo un edge de batch!!
                                non_contract_edges[i][edge] = [-1, j]
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
        aux_shape[0] = [torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
                        torch.tensor(list(map(lambda l: l[0], non_contract_edges[0].values()))).prod().long().item(),
                        torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item()]

        aux_shape[1] = [torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
                        torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item(),
                        torch.tensor(list(map(lambda l: l[0], non_contract_edges[1].values()))).prod().long().item()]
        for i in [0, 1]:
            for j in [0, 1, 2]:
                if aux_shape[i][j] < 0:
                    aux_shape[i][j] = -1  # Por si hemos hecho producto de -1 por otra dim

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

    net._list_ops.append(('contract_edges', len(net._successors['contract_edges']) - 1))

    return new_node


def _contract_edges_next(successor: Successor,
                         edges: List[AbstractEdge],
                         node1: AbstractNode,
                         node2: AbstractNode) -> Node:
    """
    Contract edges between two nodes.
    """
    total_time = time.time()

    if node1 == node2:
        result = node1.tensor
        for j, edge in enumerate(node1._edges):
            if edge in edges:
                if isinstance(edge, nc.ParamEdge):
                    # Obtain permutations
                    permutation_dims = [k if k < j else k + 1
                                        for k in range(node1.rank - 1)] + [j]
                    inv_permutation_dims = inverse_permutation(permutation_dims)

                    # Send multiplication dimension to the end, multiply, recover original shape
                    result = result.permute(permutation_dims)
                    result = result @ edge.matrix
                    result = result.permute(inv_permutation_dims)

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
        if PRINT_MODE: print('\t\t\t\tCheckpoint 1:', time.time() - total_time)
        # TODO: si son StackEdge, ver que todos los correspondientes edges están conectados

        # TODO: Bien, pero cuidad con la shape del batch al meterla en hints, hay que dejar libertad en ese hueco
        nodes = [node1, node2]
        tensors = [node1.tensor, node2.tensor]

        if PRINT_MODE:
            diff = time.time() - total_time
            print('\t\t\t\tCheckpoint 2:', diff)
            if diff >= 0.003:
                print('------------------HERE-------------------')
                pass

        start = time.time()

        for j, edge in enumerate(nodes[0]._edges):
            if edge in edges:
                if isinstance(edge, nc.ParamEdge):
                    # Obtain permutations
                    permutation_dims = [k if k < j else k + 1
                                        for k in range(nodes[0].rank - 1)] + [j]
                    inv_permutation_dims = inverse_permutation(permutation_dims)

                    # Send multiplication dimension to the end, multiply, recover original shape
                    tensors[0] = tensors[0].permute(permutation_dims)
                    tensors[0] = tensors[0] @ edge.matrix
                    tensors[0] = tensors[0].permute(inv_permutation_dims)

        if PRINT_MODE: print('\t\t\t\tCheck ParamEdges:', time.time() - start)
        if PRINT_MODE: print('\t\t\t\tCheckpoint 3:', time.time() - total_time)

        hints = successor.hints
        # batch_idx = hints['batch_idx']
        # batch_shapes = []
        # for idx in batch_idx:
        #     batch_shapes.append(tensors[0].shape[idx])
        #
        # new_shape = hints['new_shape']
        # new_shape[:len(batch_shapes)] = batch_shapes
        #
        # aux_shape = hints['aux_shape']
        # aux_shape[0][0] = torch.tensor(batch_shapes).prod().long().item()
        # aux_shape[1][0] = torch.tensor(batch_shapes).prod().long().item()
        if PRINT_MODE: print('\t\t\t\tCheckpoint 4:', time.time() - total_time)

        start = time.time()
        for i in [0, 1]:
            tensors[i] = tensors[i].permute(hints['permutation_dims'][i])
            tensors[i] = tensors[i].reshape(hints['aux_shape'][i])

        result = tensors[0] @ tensors[1]
        result = result.view(hints['new_shape']).permute(hints['inv_permutation_dims'])
        if PRINT_MODE: print('\t\t\t\tCompute contraction:', time.time() - start)
        if PRINT_MODE: print('\t\t\t\tCheckpoint 5:', time.time() - total_time)

    child = successor.child
    if PRINT_MODE: print('\t\t\t\tCheckpoint 6:', time.time() - total_time)
    start = time.time()
    child._unrestricted_set_tensor(result)
    if PRINT_MODE: print('\t\t\t\tSave in child:', time.time() - start)
    if PRINT_MODE: print('\t\t\t\tCheckpoint 7:', time.time() - total_time)

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
        same_dims = True
        max_shape = list(lst_tensors[0].shape)
        for tensor in lst_tensors[1:]:
            for idx, dim in enumerate(tensor.shape):
                if (dim != max_shape[idx]) and same_dims:
                    same_dims = False
                if dim > max_shape[idx]:
                    max_shape[idx] = dim

        if not same_dims:
            for idx, tensor in enumerate(lst_tensors):
                if tensor.shape != max_shape:
                    pad = []
                    for max_dim, dim in zip(max_shape, tensor.shape):
                        pad += [0, max_dim - dim]
                    pad.reverse()
                    lst_tensors[idx] = nn.functional.pad(tensor, pad)
        return torch.stack(lst_tensors)


def _check_first_stack(nodes: List[AbstractNode], name: Optional[Text] = None) -> Optional[Successor]:
    kwargs = {'nodes': nodes}  # TODO: mejor si es set(nodes) por si acaso, o llevarlo controlado
    if 'stack' in nodes[0].network._successors:
        for succ in nodes[0].network._successors['stack']:
            if succ.kwargs == kwargs:
                return succ
    return None


# TODO: hacer optimizacion: si todos los nodos tienen memoria que hace referencia a un nodo
#  (sus memorias estaban guardadas en la misma pila), entonces no hay que crear nueva stack,
#  solo indexar en la previa
def _stack_first(nodes: List[AbstractNode], name: Optional[Text] = None) -> StackNode:
    """
    Stack nodes into a StackNode or ParamStackNode. The stack dimension will be the
    first one in the resultant node.
    """
    # Check if all the nodes have the same type, and/or are leaf nodes
    all_leaf = True       # Check if all the nodes are leaf
    all_non_param = True  # Check if all the nodes are non-parametric
    all_param = True      # Check if all the nodes are parametric
    all_same_ref = True   # Check if all the nodes' memory is stored in the same reference node's memory
    node_ref = None       # In the case above, the reference node
    stack_indices = []    # In the case above, stack indices of each node in the reference node's memory
    stack_indices_slice = [None, None, None]  # TODO: intentar convertir lista de indices a slice
    indices = []          # In the case above, indices of each node in the reference node's memory
    for node in nodes:
        if not node._leaf:
            all_leaf = False

        if isinstance(node, nc.ParamNode):
            all_non_param = False
        else:
            all_param = False

        if node._tensor_info['address'] is None:
            if node_ref is None:
                node_ref = node._tensor_info['node_ref']
            else:
                if node._tensor_info['node_ref'] != node_ref:
                    all_same_ref = False
            stack_indices.append(node._tensor_info['stack_idx'])
            if stack_indices_slice[0] is None:
                stack_indices_slice[0] = node._tensor_info['stack_idx']
            elif stack_indices_slice[1] == None:
                stack_indices_slice[1] = node._tensor_info['stack_idx']
                stack_indices_slice[2] = stack_indices_slice[1] - stack_indices_slice[0]
                # TODO: cuidado con ir al revés, step < 0
            else:
                if stack_indices_slice[2] is not None:
                    if abs(stack_indices_slice[1] - node._tensor_info['stack_idx']) == stack_indices_slice[2]:
                        stack_indices_slice[1] = node._tensor_info['stack_idx']
                    else:
                        stack_indices_slice[2] = None
            indices.append(node._tensor_info['index'])
        else:
            all_same_ref = False

    if stack_indices_slice[2] is not None:
        stack_indices_slice[1] += 1
        stack_indices = slice(stack_indices_slice[0],
                              stack_indices_slice[1],
                              stack_indices_slice[2])

    if all_param:
        stack_node = nc.ParamStackNode(nodes, name=name)
    else:
        stack_node = nc.StackNode(nodes, name=name)

    net = nodes[0]._network
    if all_same_ref:
        # This memory management can happen always, even not in contracting mode
        del net._memory_nodes[stack_node._tensor_info['address']]
        stack_node._tensor_info['address'] = None
        stack_node._tensor_info['node_ref'] = node_ref
        stack_node._tensor_info['full'] = False
        stack_node._tensor_info['stack_idx'] = stack_indices
        stack_node._tensor_info['index'] = stack_indices  #list(zip(*indices))

    else:
        # TODO: quitamos todos non-param por lo de la stack de data nodes, hay que controlar eso
        if all_leaf and (all_param or all_non_param) and net._contracting:
        # if all_leaf and all_param and net._contracting:
            # This memory management can only happen for leaf nodes,
            # all having the same type, in contracting mode
            for i, node in enumerate(nodes):
                shape = node.shape
                if node._tensor_info['address'] is not None:
                    del net._memory_nodes[node._tensor_info['address']]
                node._tensor_info['address'] = None
                node._tensor_info['node_ref'] = stack_node
                node._tensor_info['full'] = False
                node._tensor_info['stack_idx'] = i
                index = [i]
                for max_dim, dim in zip(stack_node.shape[1:], shape):
                    index.append(slice(max_dim - dim, max_dim))
                node._tensor_info['index'] = index

                if all_param:
                    delattr(net, 'param_' + node._name)

    successor = nc.Successor(kwargs={'nodes': nodes},
                             child=stack_node,
                             contracting=net._contracting,
                             hints={'all_leaf': all_leaf and (all_param or all_non_param),
                                    'all_same_ref': all_same_ref})
    if 'stack' in net._successors:
        net._successors['stack'].append(successor)
    else:
        net._successors['stack'] = [successor]

    net._list_ops.append(('stack', len(net._successors['stack']) - 1))

    return stack_node


def _stack_next(successor: Successor,
                nodes: List[AbstractNode],
                name: Optional[Text] = None) -> StackNode:
    child = successor.child
    if successor.hints['all_same_ref'] or (successor.hints['all_leaf'] and successor.contracting):
        return child

    # stack_tensor = stack_unequal_tensors([node.tensor for node in nodes])  # TODO:
    stack_tensor = torch.stack([node.tensor for node in nodes])
    child._unrestricted_set_tensor(stack_tensor)

    # If contracting turns True, but stack operation had been already performed
    net = nodes[0]._network
    if successor.hints['all_leaf'] and net._contracting:
        for i, node in enumerate(nodes):
            shape = node.shape
            if node._tensor_info['address'] is not None:
                del net._memory_nodes[node._tensor_info['address']]
            node._tensor_info['address'] = None
            node._tensor_info['node_ref'] = child
            node._tensor_info['full'] = False
            node._tensor_info['stack_idx'] = i
            index = [i]
            for max_dim, dim in zip(stack_tensor.shape[1:], shape):
                index.append(slice(max_dim - dim, max_dim))
            node._tensor_info['index'] = index

        successor.contracting = True

    return child


stack = Operation(_check_first_stack, _stack_first, _stack_next)


##################   UNBIND   ##################
def _check_first_unbind(node: AbstractNode) -> Optional[Successor]:
    kwargs = {'node': node}
    if 'unbind' in node._network._successors:
        for succ in node._network._successors['unbind']:
            if succ.kwargs == kwargs:
                return succ
    return None


# TODO: se puede optimizar, hace falta hacer el torch.unbind realmente?
def _unbind_first(node: AbstractNode) -> List[Node]:
    """
    Unbind stacked node. It is assumed that the stacked dimension
    is the first one.
    """
    tensors = torch.unbind(node.tensor)
    nodes = []

    # Invert structure of node.edges_lists
    # is_stack_edge = list(map(lambda e: isinstance(e, nc.AbstractStackEdge), node.edges_lists[1:]))
    edges_lists = []
    node1_lists = []
    batch_idx = None
    for i, edge in enumerate(node._edges[1:]):
        # if is_stack_edge[i]:
        if isinstance(edge, nc.AbstractStackEdge):
            edges_lists.append(edge._edges)
            node1_lists.append(edge._node1_lists)
            if edge._edges[0].is_batch() and 'batch' in edge._edges[0].axis1._name:
                batch_idx = i  # TODO: caso batch edge que puede cambiar
        else:
            # TODO: caso edges_lists que designan el indice de pila?? node1 siempre True? para que este caso?
            edges_lists.append([edge] * len(tensors))
            node1_lists.append([True] * len(tensors))
    lst = list(zip(tensors, list(zip(*edges_lists)), list(zip(*node1_lists))))

    net = node._network
    for i, (tensor, edges, node1_list) in enumerate(lst):
        new_node = nc.Node(axes_names=node.axes_names[1:],
                           name='unbind_node',
                           network=net,
                           leaf=False,
                           tensor=tensor,
                           edges=list(edges),
                           node1_list=list(node1_list))
        nodes.append(new_node)

    # This memory management can happen always, even not in contracting mode
    for i, new_node in enumerate(nodes):
        shape = new_node.shape
        if new_node._tensor_info['address'] is not None:
            del new_node.network._memory_nodes[new_node._tensor_info['address']]
        new_node._tensor_info['address'] = None
        new_node._tensor_info['node_ref'] = node
        new_node._tensor_info['full'] = False
        new_node._tensor_info['stack_idx'] = i
        index = [i]
        for max_dim, dim in zip(node.shape[1:], shape):  # TODO: max_dim == dim siempre creo
            index.append(slice(max_dim - dim, max_dim))
        new_node._tensor_info['index'] = index

    successor = nc.Successor(kwargs={'node': node},
                             child=nodes,
                             hints=batch_idx)
    if 'unbind' in net._successors:
        net._successors['unbind'].append(successor)
    else:
        net._successors['unbind'] = [successor]

    net._list_ops.append(('unbind', len(net._successors['unbind']) - 1))

    return nodes


def _unbind_next(successor: Successor, node: AbstractNode) -> List[Node]:
    # torch.unbind gives a reference to the stacked tensor, it doesn't create a new tensor
    # Thus if we have already created the unbinded nodes with their reference to where their
    # memory is stored, the next times we don't have to compute anything

    batch_idx = successor.hints
    children = successor.child
    new_dim = node.shape[batch_idx + 1]
    child_dim = children[0].shape[batch_idx]

    if new_dim == child_dim:
        return children

    for i, child in enumerate(children):
        child._tensor_info['index'][batch_idx + 1] = slice(0, new_dim)

    return successor.child  # TODO: cambia el tamaño del batch


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
    # TODO: no adaptado a'un, creamos nodos nuevos cada vez, y a su vez con unbind
    result = einsum(string, *stacks_list)
    #print('\t\t\tEinsum:', time.time() - start)
    start = time.time()
    unbinded_result = unbind(result)  # <-- Lo más lento
    #print('\t\t\tUnbind:', time.time() - start)
    return unbinded_result
