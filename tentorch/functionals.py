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


import opt_einsum
from typing import List, Optional, Text


def get_shared_edges(node1, node2):
    """
    Obtain list of edges shared between two nodes
    """
    edges = []
    for edge in node1.edges:
        if (edge in node2.edges):  # and (not edge.is_dangling()):  # TODO: why I had this?
            edges.append(edge)
    return edges


# TODO: method of nodes
def get_batch_edges(node):
    """
    Obtain list of batch edges shared between two nodes
    """
    edges = []
    for edge in node.edges:
        if edge.is_batch():
            edges.append(edge)
    return edges


def contract_edges(edges,
                   node1,
                   node2,
                   operation = None):
    """
    Contract edges between two nodes.

    Parameters
    ----------
    edges: list of edges that are to be contracted. They can be edges shared
        between `node1` and `node2`, or batch edges that are in both nodes
    node1: first node of the contraction
    node2: second node of the contraction
    operation: operation string referencing the operation form which
        `contract_between` is called

    Returns
    -------
    new_node: Node resultant from the contraction
    """
    all_shared_edges = get_shared_edges(node1, node2)
    shared_edges = []
    batch_edges = dict()
    for edge in edges:
        if edge in all_shared_edges:
            shared_edges.append(edge)
        elif edge.is_batch():
            if edge.axis1.name in batch_edges:
                batch_edges[edge.axis1.name] += 1
            else:
                batch_edges[edge.axis1.name] = 1
        else:
            raise ValueError('All edges in `edges` should be non-dangling, '
                             'shared edges between `node1` and `node2`, or batch edges')

    n_shared = len(shared_edges)
    n_batch = len(batch_edges)
    shared_subscripts = dict(zip(shared_edges,
                                 [opt_einsum.get_symbol(i) for i in range(n_shared)]))
    batch_subscripts = dict(zip(batch_edges,
                                [opt_einsum.get_symbol(i)
                                 for i in range(n_shared, n_shared + n_batch)]))

    index = n_shared + n_batch
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
                if isinstance(edge, nc.ParamEdge):
                    in_matrices = False
                    for mat in matrices:
                        if torch.equal(edge.matrix, mat):
                            in_matrices = True
                            break
                    if not in_matrices:
                        matrices_strings.append(2 * shared_subscripts[edge])
                        matrices.append(edge.matrix)
            elif edge.is_batch():
                if batch_edges[edge.axis1.name] == 2:
                    # Only perform batch contraction if the batch edge appears
                    # with the same name in both nodes
                    string += batch_subscripts[edge.axis1.name]
                    if i == 0:
                        output_string += batch_subscripts[edge.axis1.name]
                else:
                    string += opt_einsum.get_symbol(index)
                    output_string += opt_einsum.get_symbol(index)
                    index += 1
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
            node1_list.append(used_nodes[j].axes[k].is_node1())
            i += 1
        k += 1
        if k == len(input_strings[j]):
            k = 0
            j += 1

    # If nodes were connected, we can assume that both are in the same network
    if operation is None:
        operation = f'contract_edge_{edges}'
    new_node = nc.Node(axes_names=axes_names, name=new_name, network=used_nodes[0].network, param_edges=False,
                       tensor=new_tensor, edges=edges, node1_list=node1_list, parents={node1, node2}, operation=operation,
                       leaf=False)
    return new_node
