"""
Tensor Networks Class
"""

import warnings
from typing import (overload, Union, Optional,
                    Sequence, Text, List, Tuple)

import torch
import torch.nn as nn

from tentorch.network_components import (AbstractNode, Node, ParamNode,
                                         AbstractEdge, Edge, ParamEdge)

# TODO: parameterize and deparameterize network, return a different network so that
#  we can retrieve the original parameterized nodes/edges with their initial sizes


# TODO: names of (at least) ParamNodes and ParamEdges must be unique in a tensor network
class TensorNetwork(nn.Module):
    """
    Al contraer una red se crea una Network auxiliar formada por Nodes en lugar
    de ParamNodes y Edges en lugar de ParamEdges. Ahí se van guardando todos los
    nodos auxiliares, resultados de operaciones intermedias, se termina de contraer
    la red y se devuelve el resultado

    Formado opr AbstractNodes y AbstractEdges

        -nodes (add modules, del modules)
        -edges
        -forward (create nodes for input data and generate the final
            network to be contracted)
        -contraction of network (identify similar nodes to stack them
            and optimize)
        -to_tensornetwork (returns a dict or list with all the nodes
            with edges connected as the original network)
    """

    def __init__(self):
        super().__init__()
        self._nodes = list()

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        edges_list = list()
        for node in self.nodes:
            for edge in node.edges:
                edges_list.append(edge)
        return edges_list

    def _add_param(self, param: Union[ParamNode, ParamEdge]) -> None:
        if not hasattr(self, param.name):
            self.add_module(param.name, param)
        else:
            raise ValueError(f'Network already has attribute named {param.name}')

    def _remove_param(self, param: Union[ParamNode, ParamEdge]) -> None:
        if hasattr(self, param.name):
            delattr(self, param.name)
        else:
            warnings.warn('Cannot remove a parameter that is not in the network')

    def add_node(self, node: AbstractNode) -> None:
        # TODO: check this
        if node.network != self:
            node._network = self
        if isinstance(node, ParamNode):
            self.add_module(node.name, node)
            for edge in node.edges:
                if isinstance(edge, ParamEdge):
                    self.add_module(edge.name, edge)
        self._nodes.append(node)

    def add_nodes_from(self, nodes_list: Sequence[AbstractNode]):
        for name, node in nodes_list:
            self.add_node(node)

    """
    def connect_nodes(nodes_list, axis_list):
        if len(nodes_list) == 2 and len(axis_list) == 2:
            nodes_list[0][axis_list[0]] ^ nodes_list[1][axis_list[1]]
        else:
            raise ValueError('Both nodes_list and axis_list must have length 2')

    def initialize(self, *args):
        for child in self.children():
            if isinstance(child, 'AbstractNode'):
                child.initialize(*args)
            # Los Edges se inicializan solos

    # def remove_node(self, name) -> None:
    #    delattr(self, name)
    #    self._nodes.remove(name)

    # def add_data(self, data):
    #    pass

    # @abstractmethod
    def contract_network(self):
        pass

    # def forward(self, data):
    #    aux_net = self.add_data(data)
    #    result = aux_net.contract_network()
    #    self.clear_op_network()
    #    return result
    """


class MPS(TensorNetwork):
    """
        optimized operations
    """

    def __init__(self):
        super().__init__()

        # node = Node(net=self)
        # ...

    # En cada función restringir el tipo de Nodos que se pueden usar,
    # para ser siempre MPS


class Tree(TensorNetwork):
    """
        optimized operations
    """
    # Lo mismo que en MPS
    pass