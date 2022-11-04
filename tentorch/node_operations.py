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

from tentorch.network_components import *

import time

# Axis = Any
Ax = Union[int, Text, Axis]

# AbstractNode, Node, ParamNode = Any, Any, Any
# AbstractEdge, Edge, ParamEdge = Any, Any, Any
# StackNode = Any
# AbstractStackEdge, StackEdge, ParamStackEdge = Any, Any, Any
# Successor = Any
# TODO: hacer import Tensor, Parameter?

PRINT_MODE = False
CHECK_TIMES = []


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
        if PRINT_MODE:
            diff = time.time() - start
            print('Check:', diff)
            global CHECK_TIMES
            CHECK_TIMES.append(diff)

        if successor is None:
            return self.func1(*args, **kwargs)
        else:
            args = [successor] + list(args)
            return self.func2(*args, **kwargs)


#################   BASIC OP   #################
def _check_first_permute(node: AbstractNode, axes: Sequence[Ax]) -> Optional[Successor]:
    kwargs = {'node': node,
              'axes': axes}
    if 'permute' in node._successors:
        for succ in node._successors['permute']:
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
        new_node = Node(axes_names=permute_list(node.axes_names, axes_nums),
                           name='permute',
                           network=node._network,
                           leaf=False,
                           param_edges=node.param_edges(),
                           tensor=node.tensor.permute(axes_nums),
                           edges=permute_list(node._edges, axes_nums),
                           node1_list=permute_list(node.is_node1(), axes_nums))

    net = node._network
    successor = Successor(kwargs={'node': node,
                                     'axes': axes},
                             child=new_node,
                             hints=axes_nums)
    if 'permute' in node._successors:
        node._successors['permute'].append(successor)
    else:
        node._successors['permute'] = [successor]

    net._list_ops.append((node, 'permute', len(node._successors['permute']) - 1))

    return new_node


def _permute_next(successor: Successor, node: AbstractNode, axes: Sequence[Ax]) -> Node:
    new_tensor = node.tensor.permute(successor.hints)
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


def _check_first_tprod(node1: AbstractNode, node2: AbstractNode) -> Optional[Successor]:
    kwargs = {'node1': node1,
              'node2': node2}
    if 'tprod' in node1._successors:
        for succ in node1._successors['tprod']:
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
    new_node = Node(axes_names=node1.axes_names + node2.axes_names,
                       name=f'tprod',
                       network=node1._network,
                       leaf=False,
                       tensor=new_tensor,
                       edges=node1._edges + node2._edges,
                       node1_list=node1.is_node1() + node2.is_node1())

    net = node1._network
    successor = Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'tprod' in node1._successors:
        node1._successors['tprod'].append(successor)
    else:
        node1._successors['tprod'] = [successor]

    net._list_ops.append((node1, 'tprod', len(node1._successors['tprod']) - 1))

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
    if 'mul' in node1._successors:
        for succ in node1._successors['mul']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _mul_first(node1: AbstractNode, node2: AbstractNode) -> Node:
    if node1._network != node2._network:
        raise ValueError('Nodes must be in the same network')

    new_tensor = node1.tensor * node2.tensor
    new_node = Node(axes_names=node1.axes_names,
                       name=f'mul',
                       network=node1._network,
                       leaf=False,
                       tensor=new_tensor)

    net = node1._network
    successor = Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'mul' in node1._successors:
        node1._successors['mul'].append(successor)
    else:
        node1._successors['mul'] = [successor]

    net._list_ops.append((node1, 'mul', len(node1._successors['mul']) - 1))

    return new_node


def _mul_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = node1.tensor * node2.tensor
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


def _check_first_add(node1: AbstractNode, node2: AbstractNode) -> Optional[Successor]:
    kwargs = {'node1': node1,
              'node2': node2}
    if 'add' in node1._successors:
        for succ in node1._successors['add']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _add_first(node1: AbstractNode, node2: AbstractNode) -> Node:
    if node1._network != node2._network:
        raise ValueError('Nodes must be in the same network')

    new_tensor = node1.tensor + node2.tensor
    new_node = Node(axes_names=node1.axes_names,
                       name=f'add',
                       network=node1._network,
                       leaf=False,
                       tensor=new_tensor)

    net = node1._network
    successor = Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'add' in node1._successors:
        node1._successors['add'].append(successor)
    else:
        node1._successors['add'] = [successor]

    net._list_ops.append((node1, 'add', len(node1._successors['add']) - 1))

    return new_node


def _add_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = node1.tensor + node2.tensor
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


def _check_first_sub(node1: AbstractNode, node2: AbstractNode) -> Optional[Successor]:
    kwargs = {'node1': node1,
              'node2': node2}
    if 'sub' in node1._successors:
        for succ in node1._successors['sub']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _sub_first(node1: AbstractNode, node2: AbstractNode) -> Node:
    if node1._network != node2._network:
        raise ValueError('Nodes must be in the same network')

    new_tensor = node1.tensor - node2.tensor
    new_node = Node(axes_names=node1.axes_names,
                       name=f'sub',
                       network=node1._network,
                       leaf=False,
                       tensor=new_tensor)

    net = node1._network
    successor = Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'sub' in node1._successors:
        node1._successors['sub'].append(successor)
    else:
        node1._successors['sub'] = [successor]

    net._list_ops.append((node1, 'sub', len(node1._successors['sub']) - 1))

    return new_node


def _sub_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = node1.tensor * node2.tensor
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


permute = Operation(_check_first_permute, _permute_first, _permute_next)
def permute_node(node, axes): return permute(node, axes)
AbstractNode.permute = permute_node

tprod = Operation(_check_first_tprod, _tprod_first, _tprod_next)
def tprod_node(node1, node2): return tprod(node1, node2)
AbstractNode.__mod__ = tprod_node

mul = Operation(_check_first_mul, _mul_first, _mul_next)
def mul_node(node1, node2): return mul(node1, node2)
AbstractNode.__mul__ = mul_node

add = Operation(_check_first_add, _add_first, _add_next)
def add_node(node1, node2): return add(node1, node2)
AbstractNode.__add__ = add_node

sub = Operation(_check_first_sub, _sub_first, _sub_next)
def sub_node(node1, node2): return sub(node1, node2)
AbstractNode.__sub__ = sub_node


#################   CONTRACT   #################
def _check_first_contract_edges(edges: List[AbstractEdge],
                                node1: AbstractNode,
                                node2: AbstractNode) -> Optional[Successor]:
    kwargs = {'edges': edges,
              'node1': node1,
              'node2': node2}
    if 'contract_edges' in node1._successors:
        for succ in node1._successors['contract_edges']:
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
                if isinstance(edge, ParamEdge):
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
                        if isinstance(edge, ParamEdge):
                            # Obtain permutations
                            permutation_dims = [k if k < j else k + 1
                                                for k in range(nodes[i].rank - 1)] + [j]
                            inv_permutation_dims = inverse_permutation(permutation_dims) # TODO: dont permute if not necessary

                            # Send multiplication dimension to the end, multiply, recover original shape
                            tensors[i] = tensors[i].permute(permutation_dims)
                            if isinstance(edge, ParamStackEdge):
                                mat = edge.matrix
                                tensors[i] = tensors[i] @ mat.view(mat.shape[0],
                                                                  *[1]*(len(tensors[i].shape) - 3),
                                                                  *mat.shape[1:])  # First dim is stack, last 2 dims are
                            else:
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
                                    # Cuando es una stack
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
        
        for i in [0, 1]:
            if permutation_dims[i] == list(range(len(permutation_dims[i]))):
                permutation_dims[i] = []

        aux_permutation = inverse_permutation(list(map(lambda l: l[1], batch_edges.values())) +
                                              list(map(lambda l: l[1], non_contract_edges[0].values())))
        aux_permutation2 = inverse_permutation(list(map(lambda l: l[1], non_contract_edges[1].values())))
        inv_permutation_dims = aux_permutation + list(map(lambda x: x + len(aux_permutation), aux_permutation2))
        
        if inv_permutation_dims == list(range(len(inv_permutation_dims))):
            inv_permutation_dims = []

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
                    
        for i in [0, 1]:
            if tensors[i].reshape(aux_shape[i]).shape == tensors[i].shape:
                aux_shape[i] = []

        new_shape = list(map(lambda l: l[0], batch_edges.values())) + \
                    list(map(lambda l: l[0], non_contract_edges[0].values())) + \
                    list(map(lambda l: l[0], non_contract_edges[1].values()))
                
        if (aux_shape[0] == []) and (aux_shape[1] == []):
            new_shape = []

        for i in [0, 1]:
            if permutation_dims[i]:
                tensors[i] = tensors[i].permute(permutation_dims[i])
            if aux_shape[i]:
                tensors[i] = tensors[i].reshape(aux_shape[i])

        result = tensors[0] @ tensors[1]
        if new_shape:
            result = result.view(new_shape)
        if inv_permutation_dims:
            result = result.permute(inv_permutation_dims)

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

        # TODO: Save time if no transformation occurs
        # TODO: Como hacemos esto despu'es de haber cambiado la forma de los tensores,
        # est'a mal, estamos midiendo shapes erroneas. Deber'iamos hacerlo antes, al crear
        # cada lista. Y tambi'en en funci'on de esto cambiar los inv_permutation y demas
        # if permutation_dims[0] == list(range(len(tensors[0].shape))):
        #     permutation_dims[0] = []
        # if permutation_dims[1] == list(range(len(tensors[1].shape))):
        #     permutation_dims[1] = []
        # if aux_shape[0] == tensors[0].shape:
        #     aux_shape[0] = []
        # if aux_shape[1] == tensors[1].shape:
        #     aux_shape[1] = []

        hints = {'permutation_dims': permutation_dims,
                 'inv_permutation_dims': inv_permutation_dims,
                 'aux_shape': aux_shape,
                 'new_shape': new_shape}

    new_node = Node(axes_names=new_axes_names,
                       name=f'contract',
                       network=node1._network,
                       leaf=False,
                       param_edges=False,
                       tensor=result,
                       edges=new_edges,
                       node1_list=new_node1_list)

    net = node1._network
    successor = Successor(kwargs={'edges': edges,
                                     'node1': node1,
                                     'node2': node2},
                             child=new_node,
                             hints=hints)
    if 'contract_edges' in node1._successors:
        node1._successors['contract_edges'].append(successor)
    else:
        node1._successors['contract_edges'] = [successor]

    net._list_ops.append((node1, 'contract_edges', len(node1._successors['contract_edges']) - 1))

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
                if isinstance(edge, ParamEdge):
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
        total_time = time.time()
        # TODO: si son StackEdge, ver que todos los correspondientes edges están conectados

        # TODO: Bien, pero cuidad con la shape del batch al meterla en hints, hay que dejar libertad en ese hueco
        nodes = [node1, node2]
        tensors = [node1.tensor, node2.tensor]

        # torch.cuda.synchronize()

        if PRINT_MODE:
            diff = time.time() - total_time
            print('\t\t\t\tCheckpoint 2:', diff)
            if diff >= 0.003:
                print('------------------HERE-------------------')
                pass
        total_time = time.time()

        for j, edge in enumerate(nodes[0]._edges):
            if edge in edges:
                if isinstance(edge, ParamEdge):
                    # Obtain permutations
                    permutation_dims = [k if k < j else k + 1
                                        for k in range(nodes[0].rank - 1)] + [j]
                    inv_permutation_dims = inverse_permutation(permutation_dims)

                    # Send multiplication dimension to the end, multiply, recover original shape
                    tensors[0] = tensors[0].permute(permutation_dims)
                    if isinstance(edge, ParamStackEdge):
                        mat = edge.matrix
                        tensors[0] = tensors[0] @ mat.view(mat.shape[0],
                                                           *[1]*(len(tensors[0].shape) - 3),
                                                           *mat.shape[1:])  # First dim is stack, last 2 dims are
                    else:
                        tensors[0] = tensors[0] @ edge.matrix
                    tensors[0] = tensors[0].permute(inv_permutation_dims)
                    
        if PRINT_MODE: print('\t\t\t\tCheckpoint 3:', time.time() - total_time)
        total_time = time.time()

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
        total_time = time.time()

        for i in [0, 1]:
            # TODO: save time if transformations don't occur
            if hints['permutation_dims'][i]:
                tensors[i] = tensors[i].permute(hints['permutation_dims'][i])
            if hints['aux_shape'][i]:
                tensors[i] = tensors[i].reshape(hints['aux_shape'][i])
        # torch.cuda.synchronize()
        if PRINT_MODE: print('\t\t\t\tCheckpoint 4.1:', time.time() - total_time)
        total_time = time.time()

        result = tensors[0] @ tensors[1]
        # torch.cuda.synchronize()
        if PRINT_MODE: print('\t\t\t\tCheckpoint 4.2:', time.time() - total_time)
        total_time = time.time()

        # result = torch.randn(tensors[0].shape) @ torch.randn(tensors[1].shape)
        # torch.cuda.synchronize()
        # if PRINT_MODE: print('\t\t\t\tCheckpoint 4.2:', time.time() - total_time)

        # TODO: save time if transformations don't occur
        if hints['new_shape']:
            result = result.view(hints['new_shape'])
        if hints['inv_permutation_dims']:
            result = result.permute(hints['inv_permutation_dims'])
        # result = result.view(hints['new_shape']).permute(hints['inv_permutation_dims'])
        if PRINT_MODE: print('\t\t\t\tCheckpoint 5:', time.time() - total_time)
        total_time = time.time()

    child = successor.child
    if PRINT_MODE: print('\t\t\t\tCheckpoint 6:', time.time() - total_time)
    total_time = time.time()
    
    child._unrestricted_set_tensor(result)
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

AbstractEdge.contract = contract


# NOTE: más rápido -> es una estuidez, al llamar a contract_edges el input
# NOTE: ya estaba guardado siempre en kwargs del successor
# def _check_first_get_shared_edges(node1: AbstractNode, node2: AbstractNode) -> Optional[Successor]:
#     kwargs = {'node1': node1,
#               'node2': node2}
#     if 'get_shared_edges' in node1._successors:
#         for succ in node1._successors['get_shared_edges']:
#             if succ.kwargs == kwargs:
#                 return succ
#     return None


# def _get_shared_edges_first(node1: AbstractNode, node2: AbstractNode) -> List[AbstractEdge]:
#     """
#     Obtain list of edges shared between two nodes
#     """
#     edges = set()
#     for i1, edge1 in enumerate(node1._edges):
#         for i2, edge2 in enumerate(node2._edges):
#             if (edge1 == edge2) and not edge1.is_dangling():
#                 if node1.is_node1(i1) != node2.is_node1(i2):
#                     edges.add(edge1)
#     edges = list(edges)
                    
#     successor = nc.Successor(kwargs={'node1': node1, 'node2': node2},
#                              child=edges)
    
#     net = node1._network
#     if 'get_shared_edges' in node1._successors:
#         node1._successors['get_shared_edges'].append(successor)
#     else:
#         node1._successors['get_shared_edges'] = [successor]

#     net._list_ops.append((node1, 'get_shared_edges', len(node1._successors['get_shared_edges']) - 1))
    
#     return edges


# def _get_shared_edges_next(successor: Successor,
#                            node1: AbstractNode,
#                            node2: AbstractNode) -> List[AbstractEdge]:
#     """
#     Obtain list of edges shared between two nodes
#     """
#     return successor.child


# get_shared_edges = Operation(_check_first_get_shared_edges,
#                              _get_shared_edges_first,
#                              _get_shared_edges_next)
# NOTE: hasta aquí


# NOTE: modo no Operation
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
# NOTE: modo no Operation


def contract_between(node1: AbstractNode, node2: AbstractNode) -> Node:
    """
    Contract all shared edges between two nodes, also performing batch contraction
    between batch edges that share name in both nodes
    """
    start = time.time()
    edges = get_shared_edges(node1, node2)
    if PRINT_MODE: print('Get edges:', time.time() - start)
    if not edges:
        raise ValueError(f'No batch edges neither shared edges between '
                         f'nodes {node1!s} and {node2!s} found')
    return contract_edges(edges, node1, node2)

AbstractNode.__matmul__ = contract_between


###################   STACK   ##################
def _check_first_stack(nodes: List[AbstractNode], name: Optional[Text] = None) -> Optional[Successor]:
    kwargs = {'nodes': nodes}  # TODO: mejor si es set(nodes) por si acaso, o llevarlo controlado
    if 'stack' in nodes[0]._successors:
        for succ in nodes[0]._successors['stack']:
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
    use_slice = True
    stack_indices_slice = [None, None, None]  # TODO: intentar convertir lista de indices a slice
    # indices = []          # In the case above, indices of each node in the reference node's memory
    for node in nodes:
        if not node._leaf:
            all_leaf = False

        if isinstance(node, ParamNode):
            all_non_param = False
        else:
            all_param = False

        # NOTE: index mode / mix index mode
        if node._tensor_info['address'] is None or node.name.startswith('unbind'):
        # NOTE: mix index mode
        
        # NOTE: unbind mode
        # if node._tensor_info['address'] is None:
        # NOTE: unbind mode
            if node_ref is None:
                node_ref = node._tensor_info['node_ref']
            else:
                if node._tensor_info['node_ref'] != node_ref:
                    all_same_ref = False

            stack_indices.append(node._tensor_info['stack_idx'])

            if use_slice:
                if stack_indices_slice[0] is None:
                    stack_indices_slice[0] = node._tensor_info['stack_idx']
                    stack_indices_slice[1] = node._tensor_info['stack_idx']
                elif stack_indices_slice[2] == None:
                    stack_indices_slice[1] = node._tensor_info['stack_idx']
                    stack_indices_slice[2] = stack_indices_slice[1] - stack_indices_slice[0]
                    # TODO: cuidado con ir al revés, step < 0
                else:
                    if (node._tensor_info['stack_idx'] - stack_indices_slice[1]) == stack_indices_slice[2]:
                        stack_indices_slice[1] = node._tensor_info['stack_idx']
                    else:
                        use_slice = False

            # indices.append(node._tensor_info['index'])
        else:
            all_same_ref = False

    if stack_indices_slice[0] is not None and use_slice:
        stack_indices_slice[1] += 1
        stack_indices = slice(stack_indices_slice[0],
                              stack_indices_slice[1],
                              stack_indices_slice[2])

    if all_param:
        stack_node = ParamStackNode(nodes, name=name)
    else:
        stack_node = StackNode(nodes, name=name)

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
        # TODO: quitamos todos non-param por lo de la stack de data nodes, hay que
        #  controlar eso -> ya esta, los data nodes no son leaf
        if all_leaf and (all_param or all_non_param) and net._automemory:
        # if all_leaf and all_param and net._automemory:
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

    successor = Successor(kwargs={'nodes': nodes},
                                    child=stack_node,
                                    hints={'all_leaf': all_leaf and (all_param or all_non_param),
                                           'all_same_ref': all_same_ref,
                                           'automemory': net._automemory})
    if 'stack' in nodes[0]._successors:
        nodes[0]._successors['stack'].append(successor)
    else:
        nodes[0]._successors['stack'] = [successor]

    net._list_ops.append((nodes[0], 'stack', len(nodes[0]._successors['stack']) - 1))

    return stack_node


def _stack_next(successor: Successor,
                nodes: List[AbstractNode],
                name: Optional[Text] = None) -> StackNode:
    child = successor.child
    if successor.hints['all_same_ref'] or \
        (successor.hints['all_leaf'] and successor.hints['automemory']):
        return child

    stack_tensor = stack_unequal_tensors([node.tensor for node in nodes])  # TODO:
    # stack_tensor = torch.stack([node.tensor for node in nodes])
    child._unrestricted_set_tensor(stack_tensor)

    # If contracting turns True, but stack operation had been already performed
    net = nodes[0]._network
    if successor.hints['all_leaf'] and net._automemory:
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

        successor.hints['automemory'] = True

    return child


stack = Operation(_check_first_stack, _stack_first, _stack_next)


##################   UNBIND   ##################
def _check_first_unbind(node: AbstractNode) -> Optional[Successor]:
    kwargs = {'node': node}
    if 'unbind' in node._successors:
        for succ in node._successors['unbind']:
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
        if isinstance(edge, AbstractStackEdge):
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
        new_node = Node(axes_names=node.axes_names[1:],
                           name='unbind',
                           network=net,
                           leaf=False,
                           tensor=tensor,
                           edges=list(edges),
                           node1_list=list(node1_list))
        nodes.append(new_node)

    # TODO: originalmente borramos informacion y solo hacemos referencia a la pila
    # This memory management can happen always, even not in contracting mode
    # NOTE: index mode
    # for i, new_node in enumerate(nodes):
    #     shape = new_node.shape
    #     if new_node._tensor_info['address'] is not None:
    #         del new_node.network._memory_nodes[new_node._tensor_info['address']]
    #     new_node._tensor_info['address'] = None
    #     new_node._tensor_info['node_ref'] = node
    #     new_node._tensor_info['full'] = False
    #     new_node._tensor_info['stack_idx'] = i
    #     index = [i]
    #     for max_dim, dim in zip(node.shape[1:], shape):  # TODO: max_dim == dim siempre creo
    #         index.append(slice(max_dim - dim, max_dim))
    #     new_node._tensor_info['index'] = index
    # NOTE: index mode

    # This memory management can happen always, even not in contracting mode
    # NOTE: unbind mode / mix index mode
    for i, new_node in enumerate(nodes):
        shape = new_node.shape
        # if new_node._tensor_info['address'] is not None:
        #     del new_node.network._memory_nodes[new_node._tensor_info['address']]
        # new_node._tensor_info['address'] = None
        new_node._tensor_info['node_ref'] = node
        new_node._tensor_info['full'] = False
        new_node._tensor_info['stack_idx'] = i
        index = [i]
        for max_dim, dim in zip(node.shape[1:], shape):  # TODO: max_dim == dim siempre creo
            index.append(slice(max_dim - dim, max_dim))
        new_node._tensor_info['index'] = index
    # NOTE: unbind mode / mix index mode

    successor = Successor(kwargs={'node': node},
                             child=nodes,
                             hints=batch_idx)
    if 'unbind' in node._successors:
        node._successors['unbind'].append(successor)
    else:
        node._successors['unbind'] = [successor]

    net._list_ops.append((node, 'unbind', len(node._successors['unbind']) - 1))

    return nodes[:]


def _unbind_next(successor: Successor, node: AbstractNode) -> List[Node]:
    # torch.unbind gives a reference to the stacked tensor, it doesn't create a new tensor
    # Thus if we have already created the unbinded nodes with their reference to where their
    # memory is stored, the next times we don't have to compute anything

    # NOTE: unbind mode / mix index mode
    tensors = torch.unbind(node.tensor)
    children = successor.child
    for tensor, child in zip(tensors, children):
        child._unrestricted_set_tensor(tensor)
    return children[:]
    # NOTE: unbind mode / mix index mode

    # NOTE: index mode
    # batch_idx = successor.hints
    # children = successor.child
    # new_dim = node.shape[batch_idx + 1]
    # child_dim = children[0].shape[batch_idx]
    #
    # if new_dim == child_dim:
    #     return children[:]  # TODO: añadimos [:] para no poder modificar la lista de hijos desde fuera
    #
    # for i, child in enumerate(children):
    #     child._tensor_info['index'][batch_idx + 1] = slice(0, new_dim)
    #
    # return successor.child[:]  # TODO: cambia el tamaño del batch
    # NOTE: index mode


unbind = Operation(_check_first_unbind, _unbind_first, _unbind_next)


##################   OTHERS   ##################
def _check_first_einsum(string: Text, *nodes: AbstractNode) -> Optional[Successor]:
    kwargs = {'string': string,
              'nodes': nodes}
    if 'einsum' in nodes[0]._successors:
        for succ in nodes[0]._successors['einsum']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _einsum_first(string: Text, *nodes: AbstractNode) -> Node:
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
    path, _ = opt_einsum.contract_path(einsum_string, *(tensors + matrices))
    new_tensor = opt_einsum.contract(einsum_string, *(tensors + matrices),
                                     optimize=path)

    # We assume all nodes belong to the same network
    new_node = Node(axes_names=list(axes_names.values()),
                    name='einsum',
                    network=nodes[0].network,
                    leaf=False,
                    param_edges=False,
                    tensor=new_tensor,
                    edges=list(edges.values()),
                    node1_list=list(node1_list.values()))
    
    successor = Successor(kwargs = {'string': string,
                                    'nodes': nodes},
                          child=new_node,
                          hints=path)
    if 'einsum' in nodes[0]._successors:
        nodes[0]._successors['einsum'].append(successor)
    else:
        nodes[0]._successors['einsum'] = [successor]

    net = nodes[0].network
    net._list_ops.append((nodes[0], 'einsum', len(nodes[0]._successors['einsum']) - 1))
    
    return new_node


# TODO: pensar esto un poco mejor, ahorrar c'odigo y c'alculos
def _einsum_next(successor: Successor, string: Text, *nodes: AbstractNode) -> Node:
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
    new_tensor = opt_einsum.contract(einsum_string, *(tensors + matrices),
                                     optimize=successor.hints)
    
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


einsum = Operation(_check_first_einsum, _einsum_first, _einsum_next)



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
