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
from tentorch.network_components import TensorNetwork

# TODO: MPS -> contraemos resultados y luego hacemos delete_node(node) y
#  del node para eliminar los nodos intermedios de la red y borrar las
#  referencias a ellos para poder liberar memoria
# TODO: poner nombre "especial" a los nodos resultantes para deletearlos fácil
class MPS(TensorNetwork):
    """
        optimized operations
    """

    def __init__(self):
        super().__init__()

        # node = Node(net=self)
        # ...
        # Aquí creamos los nodos y los añadimos a la red

    # En cada función restringir el tipo de Nodos que se pueden usar,
    # para ser siempre MPS


class Tree(TensorNetwork):
    """
        optimized operations
    """
    # Lo mismo que en MPS
    pass

# nombrar los moduels de nodos y edges por orden: node0, node1, ..., edge0, edge1, ...
