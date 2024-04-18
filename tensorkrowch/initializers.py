"""
Alternative functions to initialize nodes
"""

from typing import Optional, Sequence, Text

import torch

from tensorkrowch.components import Shape
from tensorkrowch.components import AbstractNode, Node, ParamNode
from tensorkrowch.components import TensorNetwork


def _initializer(init_method: Text,
                 shape: Shape,
                 param_node: bool = False,
                 *args,
                 **kwargs) -> AbstractNode:
    if not param_node:
        return Node(shape=shape, init_method=init_method, *args, **kwargs)
    else:
        return ParamNode(shape=shape, init_method=init_method, *args, **kwargs)
   

def empty(shape: Shape,
          param_node: bool = False,
          *args,
          **kwargs) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` without tensor.
 
     Parameters
     ----------
     shape : list[int], tuple[int] or torch.Size
          Node's shape, that is, the shape of its tensor.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).
     args :
          Arguments to initialize an :class:`~tensorkrowch.AbstractNode`.
     kwargs :
          Keyword arguments to initialize an :class:`~tensorkrowch.AbstractNode`.
 
     Returns
     -------
     Node or ParamNode
     """
     return _initializer(None,
                         shape=shape,
                         param_node=param_node,
                         *args,
                         **kwargs)


def zeros(shape: Shape,
          param_node: bool = False,
          *args,
          **kwargs) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` filled with zeros.

     Parameters
     ----------
     shape : list[int], tuple[int] or torch.Size
          Node's shape, that is, the shape of its tensor.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).
     args :
          Arguments to initialize an :class:`~tensorkrowch.AbstractNode`.
     kwargs :
          Keyword arguments to initialize an :class:`~tensorkrowch.AbstractNode`.

     Returns
     -------
     Node or ParamNode
     """
     return _initializer('zeros',
                         shape=shape,
                         param_node=param_node,
                         *args,
                         **kwargs)


def ones(shape: Shape,
         param_node: bool = False,
         *args,
         **kwargs) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` filled with ones.

     Parameters
     ----------
     shape : list[int], tuple[int] or torch.Size
          Node's shape, that is, the shape of its tensor.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).
     args :
          Arguments to initialize an :class:`~tensorkrowch.AbstractNode`.
     kwargs :
          Keyword arguments to initialize an :class:`~tensorkrowch.AbstractNode`.

     Returns
     -------
     Node or ParamNode
     """
     return _initializer('ones',
                         shape=shape,
                         param_node=param_node,
                         *args,
                         **kwargs)


def copy(shape: Shape,
         param_node: bool = False,
         *args,
         **kwargs) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` with a copy tensor, that is,
     a tensor filled with zeros except in the diagonal (elements
     :math:`T_{i_1 \ldots i_n}` with :math:`i_1 = \ldots = i_n`), which is
     filled with ones.

     Parameters
     ----------
     shape : list[int], tuple[int] or torch.Size
          Node's shape, that is, the shape of its tensor.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).
     args :
          Arguments to initialize an :class:`~tensorkrowch.AbstractNode`.
     kwargs :
          Keyword arguments to initialize an :class:`~tensorkrowch.AbstractNode`.

     Returns
     -------
     Node or ParamNode
     """
     return _initializer('copy',
                         shape=shape,
                         param_node=param_node,
                         *args,
                         **kwargs)


def rand(shape: Shape,
         param_node: bool = False,
         *args,
         **kwargs) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` filled with elements drawn from
     a uniform distribution :math:`U(low, high)`.

     Parameters
     ----------
     shape : list[int], tuple[int] or torch.Size
          Node's shape, that is, the shape of its tensor.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).
     args :
          Arguments to initialize an :class:`~tensorkrowch.AbstractNode`.
     kwargs :
          Keyword arguments to initialize an :class:`~tensorkrowch.AbstractNode`.

     Returns
     -------
     Node or ParamNode
     """
     return _initializer('rand',
                         shape=shape,
                         param_node=param_node,
                         *args,
                         **kwargs)


def randn(shape: Shape,
          param_node: bool = False,
          *args,
          **kwargs) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` filled with elements drawn from
     a normal distribution :math:`N(mean, std)`.

     Parameters
     ----------
     shape : list[int], tuple[int] or torch.Size
          Node's shape, that is, the shape of its tensor.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).
     args :
          Arguments to initialize an :class:`~tensorkrowch.AbstractNode`.
     kwargs :
          Keyword arguments to initialize an :class:`~tensorkrowch.AbstractNode`.

     Returns
     -------
     Node or ParamNode
     """
     return _initializer('randn',
                         shape=shape,
                         param_node=param_node,
                         *args,
                         **kwargs)
