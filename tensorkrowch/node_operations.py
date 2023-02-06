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

from typing import Union, Optional, Text, List, Dict, Tuple, Any, Callable, Sequence
from abc import abstractmethod
import time

import torch
import torch.nn as nn
import opt_einsum

from tensorkrowch.utils import (is_permutation, permute_list,
                                inverse_permutation, list2slice)

from tensorkrowch.network_components import *


Ax = Union[int, Text, Axis]


PRINT_MODE = False
CHECK_TIMES = []


################################################
#               NODE OPERATIONS                #
################################################
class Operation:

    def __init__(self, name, check_first, func1, func2):
        assert isinstance(check_first, Callable)
        assert isinstance(func1, Callable)
        assert isinstance(func2, Callable)
        self.func1 = func1
        self.func2 = func2
        self.check_first = check_first
        
        TensorNetwork.operations[name] = self

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


#################   PERMUTE    #################
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
        axes_nums.append(node.get_axis_num(axis))

    if not is_permutation(list(range(len(axes_nums))), axes_nums):
        raise ValueError('The provided list of axis is not a permutation of the'
                         ' axes of the node')
    else:
        # TODO: allow node.tenso be None?? -> nope, operations must be
        # performed to a non empty node
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
    
    node._record_in_inverse_memory()

    return new_node


def _permute_next(successor: Successor, node: AbstractNode, axes: Sequence[Ax]) -> Node:
    new_tensor = node.tensor.permute(successor.hints)
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    
    node._record_in_inverse_memory()
    
    return child


permute_op = Operation('permute', _check_first_permute, _permute_first, _permute_next)

def permute(node, axes):
    """
    Parameters
    ----------
    node: Node
        node to which we apply permute
    axes: List[Axis]
        list of axes to permute
    """
    return permute_op(node, axes)

def permute_node(node, axes):
    """
    Parameters
    ----------
    axes: List[Axis]
        list of axes to permute
    """
    return permute_op(node, axes)

AbstractNode.permute = permute_node


def permute_(node: AbstractNode, axes: Sequence[Ax]) -> Node:
    """Permute in place"""
    axes_nums = []
    for axis in axes:
        axes_nums.append(node.get_axis_num(axis))

    if not is_permutation(list(range(len(axes_nums))), axes_nums):
        raise ValueError('The provided list of axis is not a permutation of the'
                         ' axes of the node')
    else:
        # TODO: allow node.tenso be None?? -> nope, operations must be
        # performed to a non empty node
        new_node = Node(axes_names=permute_list(node.axes_names, axes_nums),
                        name=node._name,
                        override_node=True,
                        network=node._network,
                        param_edges=node.param_edges(),
                        override_edges=True,
                        tensor=node.tensor.permute(axes_nums),
                        edges=permute_list(node._edges, axes_nums),
                        node1_list=permute_list(node.is_node1(), axes_nums))
        
    return new_node

AbstractNode.permute_ = permute_


#################   BASIC OPS  #################
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
    
    node1._record_in_inverse_memory()
    node2._record_in_inverse_memory()

    return new_node


def _tprod_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = torch.outer(node1.tensor.flatten(),
                             node2.tensor.flatten()).view(*(list(node1.shape) +
                                                            list(node2.shape)))
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    
    node1._record_in_inverse_memory()
    node2._record_in_inverse_memory()
    
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
                       tensor=new_tensor,
                       edges=node1._edges,
                       node1_list=node1.is_node1())  # NOTE: edges resultant from operation always inherit edges

    net = node1._network
    successor = Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'mul' in node1._successors:
        node1._successors['mul'].append(successor)
    else:
        node1._successors['mul'] = [successor]

    net._list_ops.append((node1, 'mul', len(node1._successors['mul']) - 1))
    
    node1._record_in_inverse_memory()
    node2._record_in_inverse_memory()

    return new_node


def _mul_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = node1.tensor * node2.tensor
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    
    node1._record_in_inverse_memory()
    node2._record_in_inverse_memory()
    
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
                       tensor=new_tensor,
                       edges=node1._edges,
                       node1_list=node1.is_node1())

    net = node1._network
    successor = Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'add' in node1._successors:
        node1._successors['add'].append(successor)
    else:
        node1._successors['add'] = [successor]

    net._list_ops.append((node1, 'add', len(node1._successors['add']) - 1))
    
    node1._record_in_inverse_memory()
    node2._record_in_inverse_memory()

    return new_node


def _add_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = node1.tensor + node2.tensor
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    
    node1._record_in_inverse_memory()
    node2._record_in_inverse_memory()
    
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
                       tensor=new_tensor,
                       edges=node1._edges,
                       node1_list=node1.is_node1())

    net = node1._network
    successor = Successor(kwargs={'node1': node1,
                                     'node2': node2},
                             child=new_node)
    if 'sub' in node1._successors:
        node1._successors['sub'].append(successor)
    else:
        node1._successors['sub'] = [successor]

    net._list_ops.append((node1, 'sub', len(node1._successors['sub']) - 1))
    
    node1._record_in_inverse_memory()
    node2._record_in_inverse_memory()

    return new_node


def _sub_next(successor: Successor, node1: AbstractNode, node2: AbstractNode) -> Node:
    new_tensor = node1.tensor - node2.tensor
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    
    node1._record_in_inverse_memory()
    node2._record_in_inverse_memory()
    
    return child


tprod = Operation('tprod', _check_first_tprod, _tprod_first, _tprod_next)
def tprod_node(node1, node2): return tprod(node1, node2)
AbstractNode.__mod__ = tprod_node


mul = Operation('mul', _check_first_mul, _mul_first, _mul_next)
def mul_node(node1, node2): return mul(node1, node2)
AbstractNode.__mul__ = mul_node


add = Operation('add', _check_first_add, _add_first, _add_next)
def add_node(node1, node2): return add(node1, node2)
AbstractNode.__add__ = add_node


sub = Operation('sub', _check_first_sub, _sub_first, _sub_next)
def sub_node(node1, node2): return sub(node1, node2)
AbstractNode.__sub__ = sub_node


#################     SPLIT    #################
def _check_first_split(node: AbstractNode,
                       node1_axes: Optional[Sequence[Ax]] = None,
                       node2_axes: Optional[Sequence[Ax]] = None,
                       #  node1: Optional[AbstractNode] = None,
                       #  node2: Optional[AbstractNode] = None,
                       mode = 'svd',
                       side = 'left',
                       rank: Optional[int] = None,
                       cum_percentage: Optional[float] = None,
                       cutoff: Optional[float] = None) -> Optional[Successor]:
    kwargs={'node': node,
            'node1_axes': node1_axes,
            'node2_axes': node2_axes,
            'mode': mode,
            'side': side,
            'rank': rank,
            'cum_percentage': cum_percentage,
            'cutoff': cutoff}
    if 'split' in node._successors:
        for succ in node._successors['split']:
            if succ.kwargs == kwargs:
                return succ
    return None


def _split_first(node: AbstractNode,
                 node1_axes: Optional[Sequence[Ax]] = None,
                 node2_axes: Optional[Sequence[Ax]] = None,
                 mode = 'svd',
                 side = 'left',
                 rank: Optional[int] = None,
                 cum_percentage: Optional[float] = None,
                 cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    """
    Split one node in two via SVD. The set of edges has to be split in two sets,
    corresponding to the edges of the first and second resultant nodes. Batch
    edges that don't appear in any of the lists will be repeated in both nodes

    Args:
        node: node we are splitting
        node1_axes: sequence of axes from `node` whose attached edges will
            go to `node1`
        node2_axes: sequence of axes from `node` whose attached edges will
            go to `node2`
        mode: available modes are `qr` (QR decomposition), `rq`, `svd` (SVD
            decomposition cutting off singular values according to `rank`
            or `cum_percentage`) or `svdr` (like `svd` but multiplying with
            random diagonal matrix of 1's and -1's)
        side: when performing SVD, the singular values matrix has to be absorbed
            by either the "left" (U) or "right" (Vh) matrix 
        rank: number of singular values to keep
        cum_percentage: if rank is None, number of singular values to keep will
            be the amount of values whose sum with respect to the total sum of
            singular values is greater than `cum_percentage`
        cutoff: if rank and cum_percentage are None, only singular values greater
            than cutoff will remain
            
    Returns:
        node1, node2: nodes that store both parts of the splitted tensor
    """
    if node1_axes is not None:
        if node2_axes is None:
            raise ValueError('If `node1_edges` is provided `node2_edges` '
                             'should also be provided')
    else:
        if node2_axes is not None:
            raise ValueError('If `node2_edges` is provided `node1_edges` '
                             'should also be provided')
            
    if (rank is not None) and (cum_percentage is not None):
        raise ValueError('Only one of `rank` and `cum_percentage` should '
                         'be provided')
    
    if node1_axes is not None:
        if not isinstance(node1_axes, (list, tuple)):
            raise TypeError('`node1_edges` should be list or tuple type')
        if not isinstance(node2_axes, (list, tuple)):
            raise TypeError('`node2_edges` should be list or tuple type')
        
        kwargs = {'node': node,
                  'node1_axes': node1_axes,
                  'node2_axes': node2_axes,
                  'mode': mode,
                  'side': side,
                  'rank': rank,
                  'cum_percentage': cum_percentage,
                  'cutoff': cutoff}
        
        node1_axes = [node.get_axis_num(axis) for axis in node1_axes]
        node2_axes = [node.get_axis_num(axis) for axis in node2_axes]
        
        batch_axes = []
        all_axes = node1_axes + node2_axes
        all_axes.sort()
        
        if all_axes:
            j = 0
            k = all_axes[0]
        else:
            k = node.rank
        for i in range(node.rank):
            if i < k:
                if not node._edges[i].is_batch():
                    raise ValueError(f'Edge {node._edges[i]} is not a batch '
                                     'edge but it\'s not included in `node1_axes` '
                                     'neither in `node2_axes`')
                else:
                    batch_axes.append(i)
            else:
                if (j + 1) == len(all_axes):
                    k = node.rank
                else:
                    j += 1
                    k = all_axes[j]
        
        batch_shape = torch.tensor(node.shape)[batch_axes].tolist()
        node1_shape = torch.tensor(node.shape)[node1_axes]
        node2_shape = torch.tensor(node.shape)[node2_axes]
        
        permutation_dims = batch_axes + node1_axes + node2_axes
        if permutation_dims == list(range(node.rank)):
            permutation_dims = []
            
        if permutation_dims:
            node_tensor = node.tensor\
                .permute(*(batch_axes + node1_axes + node2_axes))\
                .reshape(*(batch_shape +
                        [node1_shape.prod().item()] +
                        [node2_shape.prod().item()]))
        else:
            node_tensor = node.tensor\
                .reshape(*(batch_shape +
                        [node1_shape.prod().item()] +
                        [node2_shape.prod().item()]))
    
    else: # TODO: not used, just case 1
        lst_permute_all = []
        lst_permute1 = []
        lst_permute2 = [] # TODO: inverse_permute
        
        lst_batches = []
        lst_batches_names = []
        
        for i, edge in enumerate(node._edges):
            in_node1 = False
            in_node2 = False
            
            for j, edge1 in enumerate(node1._edges):
                if edge == edge1:
                    in_node1 = True
                    if edge.is_batch() and (node1._axes[j]._name in node2.axes_names):
                        lst_permute_all = [i] + lst_permute_all
                        lst_permute1 = [j] + lst_permute1
                        lst_batches = [edge.size()] + lst_batches
                        lst_batches_names = [edge.axis1.name] + lst_batches_names
                    else:
                        lst_permute_all.append(i)
                        lst_permute1.append(j)
                    break
                        
            for j, edge2 in enumerate(node2._edges):
                if edge == edge2:
                    in_node2 = True
                    lst_permute_all.append(i)
                    lst_permute2.append(j)
                    break
                        
            if not in_node1 and not in_node2:
                raise ValueError('The node has an edge that is not present in '
                                 'either `node1` nor `node2`')
            
        # NOTE: there should only be one contracted_edge, since we only use
        # `node1` and `node2` from svd, after contracting a single edge
        contracted_edge = None
        for edge in node1._edges:
            if edge not in node._edges:
                if edge in node2._edges:
                    contracted_edge = edge
                    break
                
        reshape_edges1 = torch.tensor(node.shape)[lst_permute1]
        reshape_edges2 = torch.tensor(node.shape)[lst_permute2]

        node_tensor = node.tensor.permute(*lst_permute_all).reshape(
            *(lst_batches +
            [reshape_edges1.prod().item()] +
            [reshape_edges2.prod().item()]))
        
    if (mode == 'svd') or (mode == 'svdr'):
        u, s, vh = torch.linalg.svd(node_tensor, full_matrices=False)

        if cum_percentage is not None:
            if (rank is not None) or (cutoff is not None):
                raise ValueError('Only one of `rank`, `cum_percentage` and '
                                 '`cutoff` should be provided')
            percentages = s.cumsum(-1) / s.sum(-1).view(*s.shape[:-1], 1).expand(s.shape)
            cum_percentage_tensor = torch.tensor(cum_percentage).repeat(percentages.shape[:-1])
            rank = 0
            for i in range(percentages.shape[-1]):
                p = percentages[..., i]
                rank += 1
                # NOTE: cortamos cuando en todos los batches nos pasamos del cum_percentage
                if torch.ge(p, cum_percentage_tensor).all():
                    break
                
        elif cutoff is not None:
            if rank is not None:
                raise ValueError('Only one of `rank`, `cum_percentage` and '
                                 '`cutoff` should be provided')
            cutoff_tensor = torch.tensor(cutoff).repeat(s.shape[:-1])
            rank = 0
            for i in range(s.shape[-1]):
                # NOTE: cortamos cuando en todos los batches nos pasamos del cutoff
                if torch.le(s[..., i], cutoff_tensor).all():
                    break
                rank += 1
            if rank == 0:
                rank = 1

        if rank is None:
            # raise ValueError('One of `rank` and `cum_percentage` should be provided')
            rank = s.shape[-1]
        else:
            if rank < s.shape[-1]:
                u = u[..., :rank]
                s = s[..., :rank]
                vh = vh[..., :rank, :]
            else:
                rank = s.shape[-1]
                
        if mode == 'svdr':
            phase = torch.sign(torch.randn(s.shape))
            phase = torch.diag_embed(phase)
            u = u @ phase
            vh = phase @ vh

        if side == 'left':
            u = u @ torch.diag_embed(s)
        elif side == 'right':
            vh = torch.diag_embed(s) @ vh
        else:
            # TODO: could be changed to bool or "node1"/"node2"
            raise ValueError('`side` can only be "left" or "right"')
        
        node1_tensor = u
        node2_tensor = vh
        
    elif mode == 'qr':
        q, r = torch.linalg.qr(node_tensor)
        rank = q.shape[-1]
        
        node1_tensor = q
        node2_tensor = r
        
    elif mode == 'rq':
        q, r = torch.linalg.qr(node_tensor.transpose(-1, -2))
        q = q.transpose(-1, -2)
        r = r.transpose(-1, -2)
        rank = r.shape[-1]
        
        node1_tensor = r
        node2_tensor = q
    
    else:
        raise ValueError('`mode` can only be \'svd\', \'svdr\', \'qr\' or \'rq\'')
    
    node1_tensor = node1_tensor.reshape(*(batch_shape + node1_shape.tolist() + [rank]))
    node2_tensor = node2_tensor.reshape(*(batch_shape + [rank] + node2_shape.tolist()))
    
    net = node._network
    
    node1_axes_names = permute_list(node.axes_names,
                                    batch_axes + node1_axes) + \
                       ['splitted']
    node1 = Node(axes_names=node1_axes_names,
                 name='split',
                 network=net,
                 leaf=False,
                 param_edges=node.param_edges(),
                 tensor=node1_tensor)
    
    node2_axes_names = permute_list(node.axes_names, batch_axes) + \
                       ['splitted'] + \
                       permute_list(node.axes_names, node2_axes)
    node2 = Node(axes_names=node2_axes_names,
                 name='split',
                 network=net,
                 leaf=False,
                 param_edges=node.param_edges(),
                 tensor=node2_tensor)
    
    n_batches = len(batch_axes)
    for edge in node1._edges[n_batches:-1]:
        net._remove_edge(edge)
    for edge in node2._edges[(n_batches + 1):]:
        net._remove_edge(edge)
    
    trace_node2_axes = []
    for i, axis1 in enumerate(node1_axes):
        edge1 = node._edges[axis1]
        
        in_node2 = False
        for j, axis2 in enumerate(node2_axes):
            edge2 = node._edges[axis2]
            if edge1 == edge2:
                in_node2 = True
                trace_node2_axes.append(axis2)
                
                node1_is_node1 = node.is_node1(axis1)
                if node1_is_node1:
                    new_edge = edge1.__class__(node1=node1, axis1=n_batches + i,
                                               node2=node2, axis2=n_batches + j + 1)
                else:
                    new_edge = edge1.__class__(node1=node2, axis1=n_batches + j + 1,
                                               node2=node1, axis2=n_batches + i)
                    
                node1._add_edge(edge=new_edge,
                                axis=n_batches + i,
                                node1=node1_is_node1)
                node2._add_edge(edge=new_edge,
                                axis=n_batches + j + 1,
                                node1=not node1_is_node1)
        
        if not in_node2:
            node1._add_edge(edge=edge1,
                            axis=n_batches + i,
                            node1=node.is_node1(axis1))
            
    for j, axis2 in enumerate(node2_axes):
        if axis2 not in trace_node2_axes:
            node2._add_edge(edge=node._edges[axis2],
                            axis=n_batches + j + 1,
                            node1=node.is_node1(axis2))
        
    splitted_edge = node1['splitted'] ^ node2['splitted']
    net._remove_edge(splitted_edge)
    
    successor = Successor(kwargs=kwargs,
                          child=[node1, node2],
                          hints={'batch_axes': batch_axes,
                                 'node1_axes': node1_axes,
                                 'node2_axes': node2_axes,
                                 'permutation_dims': permutation_dims,
                                 'splitted_edge': splitted_edge})
    if 'split' in node._successors:
        node._successors['split'].append(successor)
    else:
        node._successors['split'] = [successor]

    net._list_ops.append((node, 'split', len(node._successors['split']) - 1))
    
    node._record_in_inverse_memory()

    return node1, node2


def _split_next(successor: Successor,
                node: AbstractNode,
                node1_axes: Optional[Sequence[Ax]] = None,
                node2_axes: Optional[Sequence[Ax]] = None,
                mode = 'svd',
                side = 'left',
                rank: Optional[int] = None,
                cum_percentage: Optional[float] = None,
                cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    
    batch_axes = successor.hints['batch_axes']
    node1_axes = successor.hints['node1_axes']
    node2_axes = successor.hints['node2_axes']
    permutation_dims = successor.hints['permutation_dims']
    splitted_edge = successor.hints['splitted_edge']
    
    batch_shape = torch.tensor(node.shape)[batch_axes].tolist()
    node1_shape = torch.tensor(node.shape)[node1_axes]
    node2_shape = torch.tensor(node.shape)[node2_axes]
        
    if permutation_dims:
        node_tensor = node.tensor\
            .permute(*(batch_axes + node1_axes + node2_axes))\
            .reshape(*(batch_shape +
                    [node1_shape.prod().item()] +
                    [node2_shape.prod().item()]))
    else:
        node_tensor = node.tensor\
            .reshape(*(batch_shape +
                    [node1_shape.prod().item()] +
                    [node2_shape.prod().item()]))
            
    if (mode == 'svd') or (mode == 'svdr'):
        u, s, vh = torch.linalg.svd(node_tensor, full_matrices=False)

        if cum_percentage is not None:
            if (rank is not None) or (cutoff is not None):
                raise ValueError('Only one of `rank`, `cum_percentage` and '
                                 '`cutoff` should be provided')
            percentages = s.cumsum(-1) / s.sum(-1).view(*s.shape[:-1], 1).expand(s.shape)
            cum_percentage_tensor = torch.tensor(cum_percentage).repeat(percentages.shape[:-1])
            rank = 0
            for i in range(percentages.shape[-1]):
                p = percentages[..., i]
                rank += 1
                # NOTE: cortamos cuando en todos los batches nos pasamos del cum_percentage
                if torch.ge(p, cum_percentage_tensor).all():
                    break
                
        elif cutoff is not None:
            if rank is not None:
                raise ValueError('Only one of `rank`, `cum_percentage` and '
                                 '`cutoff` should be provided')
            cutoff_tensor = torch.tensor(cutoff).repeat(s.shape[:-1])
            rank = 0
            for i in range(s.shape[-1]):
                # NOTE: cortamos cuando en todos los batches nos pasamos del cutoff
                if torch.le(s[..., i], cutoff_tensor).all():
                    break
                rank += 1
            if rank == 0:
                rank = 1

        if rank is None:
            # raise ValueError('One of `rank` and `cum_percentage` should be provided')
            rank = s.shape[-1]
        else:
            if rank < s.shape[-1]:
                u = u[..., :rank]
                s = s[..., :rank]
                vh = vh[..., :rank, :]
            else:
                rank = s.shape[-1]
                
        if mode == 'svdr':
            phase = torch.sign(torch.randn(s.shape))
            phase = torch.diag_embed(phase)
            u = u @ phase
            vh = phase @ vh

        if side == 'left':
            u = u @ torch.diag_embed(s)
        elif side == 'right':
            vh = torch.diag_embed(s) @ vh
        else:
            # TODO: could be changed to bool or "node1"/"node2"
            raise ValueError('`side` can only be "left" or "right"')
        
        node1_tensor = u
        node2_tensor = vh
    
        splitted_edge._size = rank
        
    elif mode == 'qr':
        q, r = torch.linalg.qr(node_tensor)
        rank = q.shape[-1]
        
        node1_tensor = q
        node2_tensor = r
    
    elif mode == 'rq':
        q, r = torch.linalg.qr(node_tensor.transpose(-1, -2))
        q = q.transpose(-1, -2)
        r = r.transpose(-1, -2)
        rank = r.shape[-1]
        
        node1_tensor = r
        node2_tensor = q
    
    else:
        raise ValueError('`mode` can only be \'svd\', \'svdr\', \'qr\' or \'rq\'')
    
    node1_tensor = node1_tensor.reshape(*(batch_shape + node1_shape.tolist() + [rank]))
    node2_tensor = node2_tensor.reshape(*(batch_shape + [rank] + node2_shape.tolist()))
    
    children = successor.child
    children[0]._unrestricted_set_tensor(node1_tensor)
    children[1]._unrestricted_set_tensor(node2_tensor)
    
    node._record_in_inverse_memory()
    
    return children[0], children[1]


split = Operation('split', _check_first_split, _split_first, _split_next)

def split_node(node: AbstractNode,
                 node1_axes: Optional[Sequence[Ax]] = None,
                 node2_axes: Optional[Sequence[Ax]] = None,
                mode = 'svd',
                side = 'left',
                rank: Optional[int] = None,
                cum_percentage: Optional[float] = None,
                cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    return split(node, node1_axes, node2_axes,
                 mode, side, rank, cum_percentage, cutoff)

AbstractNode.split = split_node


def split_(node: AbstractNode,
           node1_axes: Optional[Sequence[Ax]] = None,
           node2_axes: Optional[Sequence[Ax]] = None,
           mode = 'svd',
           side = 'left',
           rank: Optional[int] = None,
           cum_percentage: Optional[float] = None,
           cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    """
    Contract only one edge
    """
    node1, node2 = split(node, node1_axes, node2_axes,
                         mode, side, rank, cum_percentage, cutoff)
    node1.reattach_edges(True)
    node2.reattach_edges(True)
    
    # Delete node (and its edges) from the TN
    net = node.network
    net.delete_node(node)
    
    # Add edges of result to the TN
    for res_edge in node1._edges + node2._edges:
        net._add_edge(res_edge)
    
    net._change_node_type(node1, 'leaf')
    net._change_node_type(node2, 'leaf')
    
    node._successors = dict()
    net._list_ops = []
    
    # Remove non-leaf names
    node1.name = 'split_ip'
    node2.name = 'split_ip'
    
    return node1, node2

AbstractNode.split_ = split_


def svd_(edge,
         side = 'left',
         rank: Optional[int] = None,
         cum_percentage: Optional[float] = None,
         cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    """
    Edge SVD inplace.

    Parameters
    ----------
    edge : _type_
        _description_
    side : str, optional
        _description_, by default 'left'
    rank : Optional[int], optional
        _description_, by default None
    cum_percentage : Optional[float], optional
        _description_, by default None
    cutoff : Optional[float], optional
        _description_, by default None

    Returns
    -------
    Tuple[Node, Node]
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    
    node1, node2 = edge.node1, edge.node2
    node1_name, node2_name = node1.name, node2.name
    axis1, axis2 = edge.axis1, edge.axis2
    
    batch_axes = []
    for axis in node1._axes:
        if axis.is_batch() and (axis.name in node2.axes_names):
            batch_axes.append(axis)
    
    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1
    
    contracted = edge.contract_()
    new_node1, new_node2 = split_(node=contracted,
                                  node1_axes=list(range(n_batches,
                                                        n_batches + n_axes1)),
                                  node2_axes=list(range(n_batches + n_axes1,
                                                        n_batches + n_axes1 + n_axes2)),
                                  mode='svd',
                                  side=side,
                                  rank=rank,
                                  cum_percentage=cum_percentage,
                                  cutoff=cutoff)
    
    # new_node1
    prev_nums = [ax.num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1.num):
            prev_nums.append(i)
    prev_nums += [axis1.num]
    
    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute_(permutation)
        
    # new_node2 
    prev_nums = [node2.in_which_axis(node1[ax]).num for ax in batch_axes] + [axis2.num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)
            
    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute_(permutation)
    
    new_node1.name = node1_name
    new_node1.get_axis(axis1.num).name = axis1.name
    
    new_node2.name = node2_name
    new_node2.get_axis(axis2.num).name = axis2.name
    
    return new_node1, new_node2
    
AbstractEdge.svd_ = svd_


def svdr_(edge,
         side = 'left',
         rank: Optional[int] = None,
         cum_percentage: Optional[float] = None,
         cutoff: Optional[float] = None) -> Tuple[Node, Node]:
    
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    
    node1, node2 = edge.node1, edge.node2
    node1_name, node2_name = node1.name, node2.name
    axis1, axis2 = edge.axis1, edge.axis2
    
    batch_axes = []
    for axis in node1._axes:
        if axis.is_batch() and (axis.name in node2.axes_names):
            batch_axes.append(axis)
    
    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1
    
    contracted = edge.contract_()
    new_node1, new_node2 = split_(node=contracted,
                                  node1_axes=list(range(n_batches,
                                                        n_batches + n_axes1)),
                                  node2_axes=list(range(n_batches + n_axes1,
                                                        n_batches + n_axes1 + n_axes2)),
                                  mode='svdr',
                                  side=side,
                                  rank=rank,
                                  cum_percentage=cum_percentage,
                                  cutoff=cutoff)
    
    # new_node1
    prev_nums = [ax.num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1.num):
            prev_nums.append(i)
    prev_nums += [axis1.num]
    
    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute_(permutation)
        
    # new_node2 
    prev_nums = [node2.in_which_axis(node1[ax]).num for ax in batch_axes] + [axis2.num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)
            
    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute_(permutation)
    
    new_node1.name = node1_name
    new_node1.get_axis(axis1.num).name = axis1.name
    
    new_node2.name = node2_name
    new_node2.get_axis(axis2.num).name = axis2.name
    
    return new_node1, new_node2
    
AbstractEdge.svdr_ = svdr_


def qr_(edge) -> Tuple[Node, Node]:
    
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    
    node1, node2 = edge.node1, edge.node2
    node1_name, node2_name = node1.name, node2.name
    axis1, axis2 = edge.axis1, edge.axis2
    
    batch_axes = []
    for axis in node1._axes:
        if axis.is_batch() and (axis.name in node2.axes_names):
            batch_axes.append(axis)
    
    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1
    
    contracted = edge.contract_()
    new_node1, new_node2 = split_(node=contracted,
                                  node1_axes=list(range(n_batches,
                                                        n_batches + n_axes1)),
                                  node2_axes=list(range(n_batches + n_axes1,
                                                        n_batches + n_axes1 + n_axes2)),
                                  mode='qr')
    
    # new_node1
    prev_nums = [ax.num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1.num):
            prev_nums.append(i)
    prev_nums += [axis1.num]
    
    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute_(permutation)
        
    # new_node2 
    prev_nums = [node2.in_which_axis(node1[ax]).num for ax in batch_axes] + [axis2.num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)
            
    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute_(permutation)
    
    new_node1.name = node1_name
    new_node1.get_axis(axis1.num).name = axis1.name
    
    new_node2.name = node2_name
    new_node2.get_axis(axis2.num).name = axis2.name
    
    return new_node1, new_node2
    
AbstractEdge.qr_ = qr_


def rq_(edge) -> Tuple[Node, Node]:
    
    if edge.is_dangling():
        raise ValueError('Edge should be connected to perform SVD')
    
    node1, node2 = edge.node1, edge.node2
    node1_name, node2_name = node1.name, node2.name
    axis1, axis2 = edge.axis1, edge.axis2
    
    batch_axes = []
    for axis in node1._axes:
        if axis.is_batch() and (axis.name in node2.axes_names):
            batch_axes.append(axis)
    
    n_batches = len(batch_axes)
    n_axes1 = len(node1._axes) - n_batches - 1
    n_axes2 = len(node2._axes) - n_batches - 1
    
    contracted = edge.contract_()
    new_node1, new_node2 = split_(node=contracted,
                                  node1_axes=list(range(n_batches,
                                                        n_batches + n_axes1)),
                                  node2_axes=list(range(n_batches + n_axes1,
                                                        n_batches + n_axes1 + n_axes2)),
                                  mode='rq')
    
    # new_node1
    prev_nums = [ax.num for ax in batch_axes]
    for i in range(new_node1.rank):
        if (i not in prev_nums) and (i != axis1.num):
            prev_nums.append(i)
    prev_nums += [axis1.num]
    
    if prev_nums != list(range(new_node1.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node1 = new_node1.permute_(permutation)
        
    # new_node2 
    prev_nums = [node2.in_which_axis(node1[ax]).num for ax in batch_axes] + [axis2.num]
    for i in range(new_node2.rank):
        if i not in prev_nums:
            prev_nums.append(i)
            
    if prev_nums != list(range(new_node2.rank)):
        permutation = inverse_permutation(prev_nums)
        new_node2 = new_node2.permute_(permutation)
    
    new_node1.name = node1_name
    new_node1.get_axis(axis1.num).name = axis1.name
    
    new_node2.name = node2_name
    new_node2.get_axis(axis2.num).name = axis2.name
    
    return new_node1, new_node2
    
AbstractEdge.rq_ = rq_


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
                    if isinstance(edge, ParamStackEdge):
                        mat = edge.matrix
                        # TODO: comprobar si en cualquier situacion el orden del stack edge es asi
                        result = result @ mat.view(mat.shape[0],
                                                   *[1]*(len(result.shape) - 3),
                                                   *mat.shape[1:])  # First dim is stack, last 2 dims are
                    else:
                        result = result @ edge.matrix
                    result = result.permute(inv_permutation_dims)

        axes_nums = dict(zip(range(node1.rank), range(node1.rank)))
        for edge in edges:
            axes = node1.in_which_axis(edge)
            result = torch.diagonal(result,
                                    offset=0,
                                    dim1=axes_nums[axes[0].num],
                                    dim2=axes_nums[axes[1].num])
            result = result.sum(-1)
            min_axis = min(axes[0].num, axes[1].num)
            max_axis = max(axes[0].num, axes[1].num)
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
        
        node1._record_in_inverse_memory()

    else:
        # TODO: si son StackEdge, ver que todos los correspondientes edges están conectados


        nodes = [node1, node2]
        tensors = [node1.tensor, node2.tensor]
        non_contract_edges = [dict(), dict()]
        batch_edges = dict()
        contract_edges = dict()
        
        # TODO: Guardar mejor lo de batch _edges y demás, se puede hacer más simple.
        # Usamos new_shape_hint y en next leemos dimensiones para hacer tanto
        # new_shape como aux_shape

        for i in [0, 1]:
            for j, axis in enumerate(nodes[i]._axes):
                edge = nodes[i]._edges[j]
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
                                # TODO: comprobar si en cualquier situacion el orden del stack edge es asi
                                tensors[i] = tensors[i] @ mat.view(mat.shape[0],
                                                                  *[1]*(len(tensors[i].shape) - 3),
                                                                  *mat.shape[1:])  # First dim is stack, last 2 dims are
                            else:
                                tensors[i] = tensors[i] @ edge.matrix
                            tensors[i] = tensors[i].permute(inv_permutation_dims)

                        contract_edges[edge] = []

                    contract_edges[edge].append(j)

                elif axis.is_batch():
                    if i == 0:
                        batch_in_node2 = False
                        for aux_axis in nodes[1]._axes:
                            if aux_axis.is_batch() and (axis._name == aux_axis._name):
                                batch_edges[axis._name] = [j]
                                batch_in_node2 = True
                                break

                        if not batch_in_node2:
                            non_contract_edges[i][axis._name] = j

                    else:
                        if axis._name in batch_edges:
                            batch_edges[axis._name].append(j)
                        else:
                            non_contract_edges[i][axis._name] = j

                else:
                    non_contract_edges[i][axis._name] = j

        # TODO: esto seguro que se puede hacer mejor
        permutation_dims = [None, None]
        permutation_dims[0] = list(map(lambda l: l[0], batch_edges.values())) + \
                              list(non_contract_edges[0].values()) + \
                              list(map(lambda l: l[0], contract_edges.values()))
        permutation_dims[1] = list(map(lambda l: l[1], batch_edges.values())) + \
                              list(map(lambda l: l[1], contract_edges.values())) + \
                              list(non_contract_edges[1].values())
        
        for i in [0, 1]:
            if permutation_dims[i] == list(range(len(permutation_dims[i]))):
                permutation_dims[i] = []
                
        for i in [0, 1]:
            if permutation_dims[i]:
                tensors[i] = tensors[i].permute(permutation_dims[i])
                
        shape_limits = {'batch': len(batch_edges),
                        'non_contract': len(non_contract_edges[0]),
                        'contract': len(contract_edges)}
        
        aux_shape = [None, None]
        aux_shape[0] = [torch.tensor(tensors[0].shape[:shape_limits['batch']])] +\
            [torch.tensor(tensors[0].shape[shape_limits['batch']:
                (shape_limits['batch'] + shape_limits['non_contract'])])] +\
            [torch.tensor(tensors[0].shape[(shape_limits['batch'] + shape_limits['non_contract']):])]
        aux_shape[1] = [torch.tensor(tensors[1].shape[:shape_limits['batch']])] +\
            [torch.tensor(tensors[1].shape[shape_limits['batch']:
                (shape_limits['batch'] + shape_limits['contract'])])] +\
            [torch.tensor(tensors[1].shape[(shape_limits['batch'] + shape_limits['contract']):])]
        
        new_shape = aux_shape[0][0].tolist() + aux_shape[0][1].tolist() + aux_shape[1][2].tolist()
        
        for i in [0, 1]:
            for j in range(len(aux_shape[i])):
                aux_shape[i][j] = aux_shape[i][j].prod().long().item()
                
        for i in [0, 1]:
            if aux_shape[i]:
                tensors[i] = tensors[i].reshape(aux_shape[i])

        result = tensors[0] @ tensors[1]
        if new_shape:
            result = result.view(new_shape)
        
        # NOTE: Dejamos los batches al principio
        indices = [None, None]
        indices[0] = list(map(lambda l: l[0], batch_edges.values())) + \
            list(non_contract_edges[0].values())
        indices[1] = list(non_contract_edges[1].values())
        # NOTE: Dejamos los batches al principio

        new_axes_names = []
        new_edges = []
        new_node1_list = []
        for i in [0, 1]:
            for idx in indices[i]:
                new_axes_names.append(nodes[i].axes_names[idx])
                new_edges.append(nodes[i][idx])
                new_node1_list.append(nodes[i].axes[idx].is_node1())

        hints = {'permutation_dims': permutation_dims,
                 'aux_shape': aux_shape,
                 'shape_limits': shape_limits}
        
        node1._record_in_inverse_memory()
        node2._record_in_inverse_memory()
        
    node1_is_stack = isinstance(node1, (StackNode, ParamStackNode))
    node2_is_stack = isinstance(node2, (StackNode, ParamStackNode))
    if node1_is_stack and node2_is_stack:
            new_node = StackNode(axes_names=new_axes_names,
                                 name=f'contract_edges',
                                 network=node1._network,
                                 tensor=result,
                                 edges=new_edges,
                                 node1_list=new_node1_list)
    elif node1_is_stack or node2_is_stack:
        raise TypeError('Can only contract (Param)StackNode with other (Param)StackNode')
    else:
        new_node = Node(axes_names=new_axes_names,
                        name=f'contract_edges',
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
        # node1._record_in_inverse_memory()
        for j, edge in enumerate(node1._edges):
            if edge in edges:
                if isinstance(edge, ParamEdge):
                    # Obtain permutations
                    permutation_dims = [k if k < j else k + 1
                                        for k in range(node1.rank - 1)] + [j]
                    inv_permutation_dims = inverse_permutation(permutation_dims)

                    # Send multiplication dimension to the end, multiply, recover original shape
                    result = result.permute(permutation_dims)
                    if isinstance(edge, ParamStackEdge):
                        mat = edge.matrix
                        # TODO: comprobar si en cualquier situacion el orden del stack edge es asi
                        result = result @ mat.view(mat.shape[0],
                                                   *[1]*(len(result.shape) - 3),
                                                   *mat.shape[1:])  # First dim is stack, last 2 dims are
                    else:
                        result = result @ edge.matrix
                    result = result.permute(inv_permutation_dims)

        axes_nums = dict(zip(range(node1.rank), range(node1.rank)))
        for edge in edges:
            axes = node1.in_which_axis(edge)
            result = torch.diagonal(result,
                                    offset=0,
                                    dim1=axes_nums[axes[0].num],
                                    dim2=axes_nums[axes[1].num])
            result = result.sum(-1)
            min_axis = min(axes[0].num, axes[1].num)
            max_axis = max(axes[0].num, axes[1].num)
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
                    
        node1._record_in_inverse_memory()

    else:
        if PRINT_MODE: print('\t\t\t\tCheckpoint 1:', time.time() - total_time)
        total_time = time.time()
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
        if PRINT_MODE: print('\t\t\t\tCheckpoint 4:', time.time() - total_time)
        total_time = time.time()

        for i in [0, 1]:
            # TODO: save time if transformations don't occur
            if hints['permutation_dims'][i]:
                tensors[i] = tensors[i].permute(hints['permutation_dims'][i])
            
        # NOTE: new way of computing shapes
        shape_limits = hints['shape_limits']
            
        aux_shape = [None, None]
        aux_shape[0] = [torch.tensor(tensors[0].shape[:shape_limits['batch']])] +\
            [torch.tensor(tensors[0].shape[shape_limits['batch']:
                (shape_limits['batch'] + shape_limits['non_contract'])])] +\
            [torch.tensor(tensors[0].shape[(shape_limits['batch'] + shape_limits['non_contract']):])]
        aux_shape[1] = [torch.tensor(tensors[1].shape[:shape_limits['batch']])] +\
            [torch.tensor(tensors[1].shape[shape_limits['batch']:
                (shape_limits['batch'] + shape_limits['contract'])])] +\
            [torch.tensor(tensors[1].shape[(shape_limits['batch'] + shape_limits['contract']):])]
        
        new_shape = aux_shape[0][0].tolist() + aux_shape[0][1].tolist() + aux_shape[1][2].tolist()
        
        for i in [0, 1]:
            for j in range(len(aux_shape[i])):
                aux_shape[i][j] = aux_shape[i][j].prod().long().item()
            
        for i in [0, 1]:
            if aux_shape[i]:
                tensors[i] = tensors[i].reshape(aux_shape[i])
        # torch.cuda.synchronize()
        if PRINT_MODE: print('\t\t\t\tCheckpoint 4.1:', time.time() - total_time)
        total_time = time.time()
        
        result = tensors[0] @ tensors[1]
        # torch.cuda.synchronize()
        if PRINT_MODE: print('\t\t\t\tCheckpoint 4.2:', time.time() - total_time)
        total_time = time.time()

        if new_shape:
            result = result.view(new_shape)
        if PRINT_MODE: print('\t\t\t\tCheckpoint 5:', time.time() - total_time)
        total_time = time.time()
        
        node1._record_in_inverse_memory()
        node2._record_in_inverse_memory()

    child = successor.child
    if PRINT_MODE: print('\t\t\t\tCheckpoint 6:', time.time() - total_time)
    total_time = time.time()
    
    child._unrestricted_set_tensor(result)
    if PRINT_MODE: print('\t\t\t\tCheckpoint 7:', time.time() - total_time)

    return child


contract_edges = Operation('contract_edges',
                           _check_first_contract_edges,
                           _contract_edges_first,
                           _contract_edges_next)


def contract_(edge: AbstractEdge) -> Node:
    """
    Contract only one edge
    """
    result = contract_edges([edge], edge.node1, edge.node2)
    result.reattach_edges(True)
    
    # Delete nodes (and their edges) from the TN
    net = result.network
    net.delete_node(edge.node1)
    net.delete_node(edge.node2)
    
    # Add edges of result to the TN
    for res_edge in result._edges:
        net._add_edge(res_edge)
    
    net._change_node_type(result, 'leaf')
    
    edge.node1._successors = dict()
    edge.node2._successors = dict()
    net._list_ops = []
    
    # Remove non-leaf name
    result.name = 'contract_edges_ip'
    
    return result

AbstractEdge.contract_ = contract_


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


def contract_between(node1: AbstractNode,
                     node2: AbstractNode,
                     axes: Optional[Sequence[Ax]] = None) -> Node:
    """
    Contract all shared edges between two nodes, also performing batch contraction
    between batch edges that share name in both nodes
    """
    start = time.time()
    if axes is None:
        edges = get_shared_edges(node1, node2)
    else:
        edges = [node1.get_edge(ax) for ax in axes]
    if PRINT_MODE: print('Get edges:', time.time() - start)
    if not edges:
        raise ValueError(f'No batch edges neither shared edges between '
                         f'nodes {node1!s} and {node2!s} found')
    return contract_edges(edges, node1, node2)

AbstractNode.__matmul__ = contract_between
AbstractNode.contract_between = contract_between


def contract_between_(node1: AbstractNode,
                      node2: AbstractNode,
                      axes: Optional[Sequence[Ax]] = None) -> Node:
    """
    Contract all shared edges between two nodes, also performing batch contraction
    between batch edges that share name in both nodes
    """
    result = contract_between(node1, node2, axes)
    result.reattach_edges(True)
    
    # Delete nodes (and their edges) from the TN
    net = result.network
    net.delete_node(node1)
    net.delete_node(node2)
    
    # Add edges of result to the TN
    for res_edge in result._edges:
        net._add_edge(res_edge)
    
    net._change_node_type(result, 'leaf')
    
    node1._successors = dict()
    node2._successors = dict()
    net._list_ops = []
    
    # Remove non-leaf name
    result.name = 'contract_edges_ip'
    
    return result

AbstractNode.contract_between_ = contract_between_


###################   STACK   ##################
def _check_first_stack(nodes: List[AbstractNode], name: Optional[Text] = None) -> Optional[Successor]:
    kwargs = {'nodes': nodes}  # TODO: mejor si es set(nodes) por si acaso, o llevarlo controlado
    
    if not nodes:
        raise ValueError('`nodes` should be a non-empty sequence of nodes')
    
    if 'stack' in nodes[0]._successors:
        for succ in nodes[0]._successors['stack']:
            if succ.kwargs == kwargs:
                return succ
    return None


# TODO: hacer optimizacion: si todos los nodos tienen memoria que hace referencia a un nodo
#  (sus memorias estaban guardadas en la misma pila), entonces no hay que crear nueva stack,
#  solo indexar en la previa
def _stack_first(nodes: Sequence[AbstractNode]) -> StackNode:
    """
    Stack nodes into a StackNode or ParamStackNode. The stack dimension will be the
    first one in the resultant node.
    """
    # Check if all the nodes have the same type, and/or are leaf nodes
    all_leaf = True       # Check if all the nodes are leaf
    all_non_param = True  # Check if all the nodes are non-parametric
    all_param = True      # Check if all the nodes are parametric
    all_same_ref = True   # Check if all the nodes' memory is stored in the same reference node's memory
    stack_node_ref = True # Chech if the shared reference node is a stack
    node_ref = None       # In the case above, the reference node
    stack_indices = []    # In the case above, stack indices of each node in the reference node's memory
    use_slice = True
    stack_indices_slice = [None, None, None]  # TODO: intentar convertir lista de indices a slice
    # indices = []          # In the case above, indices of each node in the reference node's memory
    
    if not isinstance(nodes, (list, tuple)):
        raise TypeError('`nodes` should be a list or tuple of nodes')
    
    net = nodes[0]._network
    for node in nodes:
        if not node._leaf:
            all_leaf = False

        if isinstance(node, ParamNode):
            all_non_param = False
        else:
            all_param = False
        
        if all_same_ref:
            if node._tensor_info['address'] is None:
                # TODO: recursion en node_ref
                # TODO: hacer stack_slice despues de haber leiod todos, por si vienen en otro orden
                aux_node_ref = node
                address = node._tensor_info['address']
                while address is None:
                    aux_node_ref = aux_node_ref._tensor_info['node_ref']
                    address = aux_node_ref._tensor_info['address']
                    
                if node_ref is None:
                    node_ref = aux_node_ref
                else:
                    if aux_node_ref != node_ref:
                        all_same_ref = False
                        continue
                        
                if not isinstance(aux_node_ref, (StackNode, ParamStackNode)):
                    all_same_ref = False
                    stack_node_ref = False
                    continue

                stack_indices.append(node._tensor_info['index'][0])
                
            else:
                all_same_ref = False

    if all_param and stack_node_ref and net._automemory:
        stack_node = ParamStackNode(nodes=nodes, name='virtual_stack', virtual=True)
    else:
        stack_node = StackNode(nodes=nodes, name='stack')

    if all_same_ref: # NOTE: entra aqui en index mode, no entra en if ni else en unbind mode
        # TODO: make distinction here between unbind or index mode -> solo se entra aqui en index mode
        # This memory management can happen always, even not in contracting mode
        
        stack_indices = list2slice(stack_indices)
        
        del net._memory_nodes[stack_node._tensor_info['address']]
        stack_node._tensor_info['address'] = None
        stack_node._tensor_info['node_ref'] = node_ref
        stack_node._tensor_info['full'] = False
        
        index = [stack_indices]
        if node_ref.shape[1:] != stack_node.shape[1:]:
            for i, (max_dim, dim) in enumerate(zip(node_ref.shape[1:], stack_node.shape[1:])):
                if stack_node._axes[i + 1].is_batch():
                    index.append(slice(0, None))
                else:
                    index.append(slice(max_dim - dim, max_dim))
        stack_node._tensor_info['index'] = index

    else:
        # TODO: quitamos todos non-param por lo de la stack de data nodes, hay que
        #  controlar eso -> ya esta, los data nodes no son leaf
        if all_leaf and (all_param or all_non_param) \
            and stack_node_ref and net._automemory:
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
                index = [i]
                for j, (max_dim, dim) in enumerate(zip(stack_node.shape[1:], shape)):
                    if node._axes[j].is_batch():
                        index.append(slice(0, None))
                    else:
                        index.append(slice(max_dim - dim, max_dim))
                node._tensor_info['index'] = index

                if all_param:
                    delattr(net, 'param_' + node._name)
                    
        for node in nodes:
            node._record_in_inverse_memory()

    successor = Successor(kwargs={'nodes': nodes},
                                    child=stack_node,
                                    hints={'all_leaf': all_leaf and (all_param or all_non_param) and stack_node_ref,
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
    child._unrestricted_set_tensor(stack_tensor)

    # TODO: no permitido ya cambiar automemory mode en mitad de ejecucion, asi que esto no pasa
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
            index = [i]
            for j, (max_dim, dim) in enumerate(zip(stack_tensor.shape[1:], shape)):
                    if node._axes[j].is_batch():
                        index.append(slice(0, None))
                    else:
                        index.append(slice(max_dim - dim, max_dim))
            node._tensor_info['index'] = index

        successor.hints['automemory'] = True
        
    else:
        for node in nodes:
            node._record_in_inverse_memory()

    return child


stack = Operation('stack', _check_first_stack, _stack_first, _stack_next)


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
    if not isinstance(node, (StackNode, ParamStackNode)):
        raise TypeError('Cannot unbind node if it is not a (Param)StackNode')
    
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
        
    if not net.unbind_mode: # TODO: organizar mas simple
        # TODO: originalmente borramos informacion y solo hacemos referencia a la pila
        # This memory management can happen always, even not in contracting mode
        # NOTE: index mode
        aux_node_ref = node
        address = node._tensor_info['address']
        while address is None:
            aux_node_ref = aux_node_ref._tensor_info['node_ref']
            address = aux_node_ref._tensor_info['address']
                
        for i, new_node in enumerate(nodes):
            
            shape = new_node.shape
            if new_node._tensor_info['address'] is not None:
                del new_node.network._memory_nodes[new_node._tensor_info['address']]
            new_node._tensor_info['address'] = None
            
            new_node._tensor_info['node_ref'] = aux_node_ref
            new_node._tensor_info['full'] = False
            
            if aux_node_ref == node:
                index = [i]
                for j, (max_dim, dim) in enumerate(zip(node.shape[1:], shape)):  # TODO: max_dim == dim siempre creo
                    if new_node._axes[j].is_batch():
                        index.append(slice(0, None))
                    else:
                        index.append(slice(max_dim - dim, max_dim))
                new_node._tensor_info['index'] = index
                
            else:
                node_index = node._tensor_info['index']
                aux_slice = node_index[0]
                if isinstance(aux_slice, list):
                    index = [aux_slice[i]]
                else:
                    index = [range(aux_slice.start, aux_slice.stop, aux_slice.step)[i]]
                    
                # If node is indexing from the original stack
                if node_index[1:]:
                    for j, (aux_slice, dim) in enumerate(zip(node_index[1:], shape)):
                        if new_node._axes[j].is_batch():
                            index.append(slice(0, None))
                        else:
                            index.append(slice(aux_slice.stop - dim, aux_slice.stop))
                # If node has the same shape as the original stack
                else:
                    for j, (max_dim, dim) in enumerate(zip(node.shape[1:], shape)):  # TODO: max_dim == dim siempre creo
                        if new_node._axes[j].is_batch():
                            index.append(slice(0, None))
                        else:
                            index.append(slice(max_dim - dim, max_dim))
                new_node._tensor_info['index'] = index
        
    else:
        node._record_in_inverse_memory()

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

    net = node._network
    if net.unbind_mode:
        # NOTE: unbind mode / mix index mode
        tensors = torch.unbind(node.tensor)
        children = successor.child
        for tensor, child in zip(tensors, children):
            child._unrestricted_set_tensor(tensor)
            
        node._record_in_inverse_memory()
        return children[:]
        # NOTE: unbind mode / mix index mode
        
    else:
        # NOTE: index mode
        # TODO: creo que esto del batch ya no me sirve de nada,
        # puedo poner tensores con distintas dims en el nodo
        batch_idx = successor.hints
        children = successor.child
        
        if batch_idx is None:
            # node._record_in_inverse_memory()
            return children[:]
        
        new_dim = node.shape[batch_idx + 1]
        child_dim = children[0].shape[batch_idx]
        
        if new_dim == child_dim:
            # node._record_in_inverse_memory()
            return children[:]  # TODO: añadimos [:] para no poder modificar la lista de hijos desde fuera
        
        return successor.child[:]  # TODO: cambia el tamaño del batch
        # NOTE: index mode


unbind = Operation('unbind', _check_first_unbind, _unbind_first, _unbind_next)


##################   OTHERS   ##################
def _check_first_einsum(string: Text, *nodes: AbstractNode) -> Optional[Successor]:
    kwargs = {'string': string,
              'nodes': nodes}
    
    if not nodes:
        raise ValueError('No nodes were provided')
    
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
    
    for i in range(len(nodes[:-1])):
        if nodes[i].network != nodes[i + 1].network:
            raise ValueError('All `nodes` must be in the same network')
    
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
    which_matrices = []
    # matrices = []
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
        if isinstance(nodes[i], (StackNode, ParamStackNode)):
            stack_char = input_string[0]  # TODO: nodes can have ParamStackEdge if they are result of operation with stacks
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
                    contracted_edges[char].append(edge)
                if isinstance(edge, ParamStackEdge):
                    if edge not in which_matrices:
                        matrices_strings.append(stack_char + (2 * char))
                        which_matrices.append(edge)
                        
                elif isinstance(edge, ParamEdge):
                    if edge not in which_matrices:
                        matrices_strings.append(2 * char)
                        which_matrices.append(edge)
                        
            else:
                edge = nodes[i][j]
                if output_dict[char] == 0:
                    if edge.is_batch():
                        batch_edges[char] = 0
                    k = output_char_index[char]
                    axes_names[k] = nodes[i].axes[j].name
                    edges[k] = edge
                    node1_list[k] = nodes[i].axes[j].is_node1()
                output_dict[char] += 1
                if (char in batch_edges) and edge.is_batch():
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
    tensors = [node.tensor for node in nodes]
    matrices = [edge.matrix for edge in which_matrices]
    path, _ = opt_einsum.contract_path(einsum_string, *(tensors + matrices))
    new_tensor = opt_einsum.contract(einsum_string, *(tensors + matrices),
                                     optimize=path)

    all_stack = True
    all_non_stack = True
    for node in nodes:
        if isinstance(node, (StackNode, ParamStackNode)):
            all_stack &= True
            all_non_stack &= False
        else:
            all_stack &= False
            all_non_stack &= True
            
    if all_stack and all_non_stack:
        raise TypeError('Cannot operate (Param)StackNode\'s with '
                        'other (non-stack) nodes')
        
    if all_stack:
        new_node = StackNode(axes_names=list(axes_names.values()),
                             name='einsum',
                             network=nodes[0].network,
                             tensor=new_tensor,
                             edges=list(edges.values()),
                             node1_list=list(node1_list.values()))
    else:
        # TODO: We assume all nodes belong to the same network
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
                          hints={'einsum_string': einsum_string,
                                 'which_matrices': which_matrices,
                                 'path': path})
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
    hints = successor.hints
    
    tensors = [node.tensor for node in nodes]
    matrices = [edge.matrix for edge in hints['which_matrices']]
    new_tensor = opt_einsum.contract(hints['einsum_string'], *(tensors + matrices),
                                     optimize=hints['path'])
    
    child = successor.child
    child._unrestricted_set_tensor(new_tensor)
    return child


einsum = Operation('einsum', _check_first_einsum, _einsum_first, _einsum_next)


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
    unbinded_result = unbind(result)  # <-- Lo más lento
    return unbinded_result


###########################################
#    Save operations in TensorNetwork     #
###########################################

# TensorNetwork.operations = {'permute': permute,
#                             'tprod': tprod,
#                             'mul': mul,
#                             'add': add,
#                             'sub': sub,
#                             'split': split,
#                             'contract_edges': contract_edges,
#                             'stack': stack,
#                             'unbind': unbind}
# TODO: add einsum, stacked_einsum
