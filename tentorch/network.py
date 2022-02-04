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

# TODO: poder acceder a los edges libres de la network
# nombrar los moduels de nodos y edges por orden: node0, node1, ..., edge0, edge1, ...
