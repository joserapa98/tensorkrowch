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
"""

import tentorch.network_components as nc
import torch

from typing import Callable, List, Any
from tentorch.utils import permute_list, inverse_permutation

AbstractNode = Any
Node = Any
AbstractEdge = Any
ParamEdge = Any

TN_MODE = True


class tn_mode:

    def __init__(self):
        global TN_MODE
        TN_MODE = False

    def __enter__(self):
        pass

    def __exit__(self, *args, **kws):
        global TN_MODE
        TN_MODE = True


class Foo:

    def __init__(self, func1, func2):
        print('Creating operation foo')
        self.func1 = func1
        self.func2 = func2
        return

    def __call__(self, data):
        global TN_MODE
        if TN_MODE:
            return self.func1(data)
        else:
            return self.func2(data)

    # def op(self, node1: Node, node2: Node):
    #     print('Operating node1 and node2')
    #     return


def _func1(data):
    print('Computing func1')
    a = nc.Node(tensor=torch.randn(2, 3))
    print(type(a))


def _func2(data):
    a = torch.randn(2, 3)
    print('Computing func2')
    print(type(a))


foo = Foo(_func1, _func2)


# def _func1(data):
#     print('Computing func1')
#
#
# def _func2(data):
#     print('Computing func2')
#
#
# foo = Foo(_func1, _func2)
#
#
# import torch
# import tentorch as tn
# import pytest
#
#
# def test_foo():
#     a = tn.Node(tensor=torch.randn(2, 3))
#     a.foo(0)
#     with tn_mode():
#         a.foo(0)


# from abc import ABC
#
# # TODO: usar esto para copy y permute
# class Foo2(ABC):
#     def foo2(self):
#         cls = self.__class__
#         new = cls()
#         return new
#
# class Foo2_1(Foo2):
#     pass
#
# class Foo2_2(Foo2):
#     pass


class Operation:

    def __init__(self, check_first, func1, func2):
        assert isinstance(check_first, Callable)
        assert isinstance(func1, Callable)
        assert isinstance(func2, Callable)
        self.func1 = func1
        self.func2 = func2
        self.check_first = check_first

    def __call__(self, *args, **kwargs):
        if self.check_first(*args, **kwargs):
            return self.func1(*args, **kwargs)
        else:
            return self.func2(*args, **kwargs)


def get_shared_edges(node1: AbstractNode, node2: AbstractNode) -> List[AbstractEdge]:
    """
    Obtain list of edges shared between two nodes
    """
    edges = []
    for edge in node1.edges:
        if (edge in node2.edges):  # and (not edge.is_dangling()):  # TODO: why I had this?
            edges.append(edge)
    return edges


# TODO: method of nodes
def get_batch_edges(node: AbstractNode) -> List[AbstractEdge]:
    """
    Obtain list of batch edges shared between two nodes
    """
    edges = []
    for edge in node.edges:
        if edge.is_batch():
            edges.append(edge)
    return edges


def _check_first_contract_edges(edges: List[AbstractEdge],
                                node1: AbstractNode,
                                node2: AbstractNode) -> bool:
    kwargs = {'edges': edges,
              'node1': node1,
              'node2': node2}
    if 'contract_edges' in node1.successors:
        for t in node1.successors['contract_edges']:
            if t[0] == kwargs:
                return False
    return True


def _contract_edges_first(edges: List[AbstractEdge],
                          node1: AbstractNode,
                          node2: AbstractNode) -> Node:
    """
    Contract edges between two nodes.

    Parameters
    ----------
    edges: list of edges that are to be contracted. They can be edges shared
        between `node1` and `node2`, or batch edges that are in both nodes
    node1: first node of the contraction
    node2: second node of the contraction

    Returns
    -------
    new_node: Node resultant from the contraction
    """

    if node1 == node2:
        # TODO: hacer esto
        raise ValueError('Trace not implemented')

    nodes = [node1, node2]
    tensors = [node1.tensor, node2.tensor]
    non_contract_edges = [dict(), dict()]
    batch_edges = dict()
    contract_edges = dict()

    for i in range(2):
        for j, edge in enumerate(nodes[i].edges):
            if edge in edges:
                if (edge in node2.edges) and (not edge.is_dangling()):
                    if i == 0:
                        if isinstance(edge, nc.ParamEdge):
                            # Obtain permutations
                            permutation_dims = [k if k < j else k + 1
                                                for k in range(len(tensors[i].shape) - 1)] + [j]
                            inv_permutation_dims = inverse_permutation(permutation_dims)

                            # Send multiplication dimension to the end, multiply, recover original shape
                            tensors[i] = tensors[i].permute(permutation_dims)
                            tensors[i] = tensors[i] @ edge.matrix
                            tensors[i] = tensors[i].permute(inv_permutation_dims)

                        contract_edges[edge] = [nodes[i].shape[j]]

                    contract_edges[edge].append(j)

                else:
                    raise ValueError('All edges in `edges` should be non-dangling, '
                                     'shared edges between `node1` and `node2`, or batch edges')

            elif edge.is_batch():
                if i == 0:
                    batch_in_node2 = False
                    for aux_edge in node2.edges:
                        if aux_edge.is_batch() and (edge.axis1.name == aux_edge.axis1.name):
                            batch_edges[edge.axis1.name] = [node1.shape[j], j]
                            batch_in_node2 = True
                            break

                    if not batch_in_node2:
                        non_contract_edges[i][edge] = [nodes[i].shape[j], j]

                else:
                    if edge.axis1.name in batch_edges:
                        batch_edges[edge.axis1.name].append(j)
                    else:
                        non_contract_edges[i][edge] = [nodes[i].shape[j], j]

            else:
                non_contract_edges[i][edge] = [nodes[i].shape[j], j]

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
    final_inv_permutation_dims = aux_permutation + list(map(lambda x: x+len(aux_permutation), aux_permutation2))

    new_shape = [None, None]
    new_shape[0] = (torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
                    torch.tensor(list(map(lambda l: l[0], non_contract_edges[0].values()))).prod().long().item(),
                    torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item())

    new_shape[1] = (torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
                    torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item(),
                    torch.tensor(list(map(lambda l: l[0], non_contract_edges[1].values()))).prod().long().item())

    final_shape = list(map(lambda l: l[0], batch_edges.values())) + \
                  list(map(lambda l: l[0], non_contract_edges[0].values())) + \
                  list(map(lambda l: l[0], non_contract_edges[1].values()))

    for i in range(2):
        tensors[i] = tensors[i].permute(permutation_dims[i])
        tensors[i] = tensors[i].reshape(new_shape[i])

    result = tensors[0] @ tensors[1]
    result = result.view(final_shape).permute(final_inv_permutation_dims)

    indices_node1 = permute_list(list(map(lambda l: l[1], batch_edges.values())) +
                                 list(map(lambda l: l[1], non_contract_edges[0].values())),
                                 aux_permutation)
    indices_node2 = list(map(lambda l: l[1], non_contract_edges[1].values()))
    indices = [indices_node1, indices_node2]
    final_edges = []
    final_axes = []
    final_node1 = []
    for i in range(2):
        for idx in indices[i]:
            final_edges.append(nodes[i][idx])
            final_axes.append(nodes[i].axes_names[idx])
            final_node1.append(nodes[i].axes[idx].is_node1())

    new_node = nc.Node(axes_names=final_axes, name=f'contract_{node1.name}_{node2.name}', network=nodes[0].network,
                       leaf=False, param_edges=False, tensor=result, edges=final_edges, node1_list=final_node1)

    for node in nodes:
        if 'contract_edges' in node._successors:
            node._successors['contract_edges'].append(({'edges': edges,
                                                        'node1': node1,
                                                        'node2': node2},
                                                       new_node))
        else:
            node._successors['contract_edges'] = [({'edges': edges,
                                                    'node1': node1,
                                                    'node2': node2},
                                                   new_node)]

    return new_node


def _contract_edges_next(edges: List[AbstractEdge],
                         node1: AbstractNode,
                         node2: AbstractNode) -> Node:
    """
    Contract edges between two nodes.

    Parameters
    ----------
    edges: list of edges that are to be contracted. They can be edges shared
        between `node1` and `node2`, or batch edges that are in both nodes
    node1: first node of the contraction
    node2: second node of the contraction

    Returns
    -------
    new_node: Node resultant from the contraction
    """

    if node1 == node2:
        # TODO: hacer esto
        raise ValueError('Trace not implemented')

    nodes = [node1, node2]
    tensors = [node1.tensor, node2.tensor]
    non_contract_edges = [dict(), dict()]
    batch_edges = dict()
    contract_edges = dict()

    for i in range(2):
        for j, edge in enumerate(nodes[i].edges):
            if edge in edges:
                if (edge in node2.edges) and (not edge.is_dangling()):
                    if i == 0:
                        if isinstance(edge, nc.ParamEdge):
                            # Obtain permutations
                            permutation_dims = [k if k < j else k + 1
                                                for k in range(len(tensors[i].shape) - 1)] + [j]
                            inv_permutation_dims = inverse_permutation(permutation_dims)

                            # Send multiplication dimension to the end, multiply, recover original shape
                            tensors[i] = tensors[i].permute(permutation_dims)
                            tensors[i] = tensors[i] @ edge.matrix
                            tensors[i] = tensors[i].permute(inv_permutation_dims)

                        contract_edges[edge] = [nodes[i].shape[j]]

                    contract_edges[edge].append(j)

                else:
                    raise ValueError('All edges in `edges` should be non-dangling, '
                                     'shared edges between `node1` and `node2`, or batch edges')

            elif edge.is_batch():
                if i == 0:
                    batch_in_node2 = False
                    for aux_edge in node2.edges:
                        if aux_edge.is_batch() and (edge.name == aux_edge.name):
                            batch_edges[edge] = [node1.shape[j], j]
                            batch_in_node2 = True
                            break

                    if not batch_in_node2:
                        non_contract_edges[i][edge] = [nodes[i].shape[j], j]

                else:
                    if edge in batch_edges:
                        batch_edges[edge].append(j)
                    else:
                        non_contract_edges[i][edge] = [nodes[i].shape[j], j]

            else:
                non_contract_edges[i][edge] = [nodes[i].shape[j], j]

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
    final_inv_permutation_dims = aux_permutation + list(map(lambda x: x + len(aux_permutation), aux_permutation2))

    new_shape = [None, None]
    new_shape[0] = (torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
                    torch.tensor(list(map(lambda l: l[0], non_contract_edges[0].values()))).prod().long().item(),
                    torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item())

    new_shape[1] = (torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
                    torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item(),
                    torch.tensor(list(map(lambda l: l[0], non_contract_edges[1].values()))).prod().long().item())

    final_shape = list(map(lambda l: l[0], batch_edges.values())) + \
                  list(map(lambda l: l[0], non_contract_edges[0].values())) + \
                  list(map(lambda l: l[0], non_contract_edges[1].values()))

    for i in range(2):
        tensors[i] = tensors[i].permute(permutation_dims[i])
        tensors[i] = tensors[i].reshape(new_shape[i])

    result = tensors[0] @ tensors[1]
    result = result.view(final_shape).permute(final_inv_permutation_dims)

    kwargs = {'edges': edges,
              'node1': node1,
              'node2': node2}
    for t in node1._successors['contract_edges']:
        if t[0] == kwargs:
            child = t[1]
            break
    child.tensor = result

    return child


contract_edges = Operation(_check_first_contract_edges,
                           _contract_edges_first,
                           _contract_edges_next)
