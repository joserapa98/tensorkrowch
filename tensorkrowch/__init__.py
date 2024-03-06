"""
TensorKrowch
============

Tensor Networks with PyTorch
"""

# Version
__version__ = '1.0.1'

# Network components
from tensorkrowch.components import Axis
from tensorkrowch.components import AbstractNode, Node, ParamNode
from tensorkrowch.components import StackNode, ParamStackNode
from tensorkrowch.components import Edge, StackEdge
from tensorkrowch.components import Successor, TensorNetwork

# Initializers
from tensorkrowch.initializers import empty, zeros, ones, copy, rand, randn

# Embeddings
import tensorkrowch.embeddings as embeddings

# Edge operations
from tensorkrowch.components import connect, connect_stack, disconnect
from tensorkrowch.operations import svd_, svdr_, qr_, rq_
from tensorkrowch.operations import contract_

# Node operations
from tensorkrowch.operations import Operation
from tensorkrowch.operations import get_shared_edges  # Not in docs

from tensorkrowch.operations import permute, permute_, tprod, mul, add, sub
from tensorkrowch.operations import (split, split_, contract_edges, contract_,
                                     contract_between, contract_between_,
                                     stack, unbind, einsum, stacked_einsum)

# Models
import tensorkrowch.models as models
