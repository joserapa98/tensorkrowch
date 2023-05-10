"""
Alternative functions to initialize nodes
"""

from typing import Optional, Sequence, Text

from tensorkrowch.components import Shape
from tensorkrowch.components import AbstractNode, Node, ParamNode
from tensorkrowch.components import TensorNetwork


def _initializer(init_method,
                 shape: Shape,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional[TensorNetwork] = None,
                 param_node: bool = False,
                 **kwargs: float) -> AbstractNode:
    if not param_node:
        return Node(shape=shape,
                    axes_names=axes_names,
                    name=name,
                    network=network,
                    init_method=init_method,
                    **kwargs)
    else:
        return ParamNode(shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         init_method=init_method,
                         **kwargs)


def empty(shape: Shape,
          axes_names: Optional[Sequence[Text]] = None,
          name: Optional[Text] = None,
          network: Optional[TensorNetwork] = None,
          param_node: bool = False) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` without tensor.

     Parameters
     ----------
     shape : list[int], tuple[int], torch.Size
          Node's shape, that is, the shape of its tensor.
     axes_names : list[str], tuple[str], optional
          Sequence of names for each of the node's axes. Names are used to access
          the edge that is attached to the node in a certain axis. Hence they
          should be all distinct.
     name : str, optional
          Node's name, used to access the node from de :class:`TensorNetwork`
          where it belongs. It cannot contain blank spaces.
     network : TensorNetwork, optional
          Tensor network where the node should belong. If None, a new tensor
          network, will be created to contain the node.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).

     Returns
     -------
     Node or ParamNode
     """
     return _initializer(None,
                         shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         param_node=param_node)


def zeros(shape: Shape,
          axes_names: Optional[Sequence[Text]] = None,
          name: Optional[Text] = None,
          network: Optional[TensorNetwork] = None,
          param_node: bool = False) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` filled with zeros.

     Parameters
     ----------
     shape : list[int], tuple[int], torch.Size
          Node's shape, that is, the shape of its tensor.
     axes_names : list[str], tuple[str], optional
          Sequence of names for each of the node's axes. Names are used to access
          the edge that is attached to the node in a certain axis. Hence they
          should be all distinct.
     name : str, optional
          Node's name, used to access the node from de :class:`TensorNetwork`
          where it belongs. It cannot contain blank spaces.
     network : TensorNetwork, optional
          Tensor network where the node should belong. If None, a new tensor
          network, will be created to contain the node.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).

     Returns
     -------
     Node or ParamNode
     """
     return _initializer('zeros',
                         shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         param_node=param_node)


def ones(shape: Optional[Shape] = None,
         axes_names: Optional[Sequence[Text]] = None,
         name: Optional[Text] = None,
         network: Optional[TensorNetwork] = None,
         param_node: bool = False) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` filled with ones.

     Parameters
     ----------
     shape : list[int], tuple[int], torch.Size
          Node's shape, that is, the shape of its tensor.
     axes_names : list[str], tuple[str], optional
          Sequence of names for each of the node's axes. Names are used to access
          the edge that is attached to the node in a certain axis. Hence they
          should be all distinct.
     name : str, optional
          Node's name, used to access the node from de :class:`TensorNetwork`
          where it belongs. It cannot contain blank spaces.
     network : TensorNetwork, optional
          Tensor network where the node should belong. If None, a new tensor
          network, will be created to contain the node.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).

     Returns
     -------
     Node or ParamNode
     """
     return _initializer('ones',
                         shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         param_node=param_node)


def copy(shape: Optional[Shape] = None,
         axes_names: Optional[Sequence[Text]] = None,
         name: Optional[Text] = None,
         network: Optional[TensorNetwork] = None,
         param_node: bool = False) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` with a copy tensor, that is, 
     a tensor filled with zeros except in the diagonal (elements
     :math:`T_{i_1 \ldots i_n}` with :math:`i_1 = \ldots = i_n`), which is
     filled with ones.

     Parameters
     ----------
     shape : list[int], tuple[int], torch.Size
          Node's shape, that is, the shape of its tensor.
     axes_names : list[str], tuple[str], optional
          Sequence of names for each of the node's axes. Names are used to access
          the edge that is attached to the node in a certain axis. Hence they
          should be all distinct.
     name : str, optional
          Node's name, used to access the node from de :class:`TensorNetwork`
          where it belongs. It cannot contain blank spaces.
     network : TensorNetwork, optional
          Tensor network where the node should belong. If None, a new tensor
          network, will be created to contain the node.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).

     Returns
     -------
     Node or ParamNode
     """
     return _initializer('copy',
                         shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         param_node=param_node)


def rand(shape: Optional[Shape] = None,
         axes_names: Optional[Sequence[Text]] = None,
         name: Optional[Text] = None,
         network: Optional[TensorNetwork] = None,
         param_node: bool = False,
         low: float = 0.,
         high: float = 1.,) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` filled with elements drawn from
     a uniform distribution :math:`U(low, high)`.

     Parameters
     ----------
     shape : list[int], tuple[int], torch.Size
          Node's shape, that is, the shape of its tensor.
     axes_names : list[str], tuple[str], optional
          Sequence of names for each of the node's axes. Names are used to access
          the edge that is attached to the node in a certain axis. Hence they
          should be all distinct.
     name : str, optional
          Node's name, used to access the node from de :class:`TensorNetwork`
          where it belongs. It cannot contain blank spaces.
     network : TensorNetwork, optional
          Tensor network where the node should belong. If None, a new tensor
          network, will be created to contain the node.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).
     low : float
          Lower limit of the uniform distribution.
     high : float
          Upper limit of the uniform distribution.

     Returns
     -------
     Node or ParamNode
     """
     return _initializer('rand',
                         shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         param_node=param_node,
                         low=low,
                         high=high)


def randn(shape: Optional[Shape] = None,
          axes_names: Optional[Sequence[Text]] = None,
          name: Optional[Text] = None,
          network: Optional[TensorNetwork] = None,
          param_node: bool = False,
          mean: float = 0.,
          std: float = 1.,) -> AbstractNode:
     """
     Returns :class:`Node` or :class:`ParamNode` filled with elements drawn from
     a normal distribution :math:`N(mean, std)`.

     Parameters
     ----------
     shape : list[int], tuple[int], torch.Size
          Node's shape, that is, the shape of its tensor.
     axes_names : list[str], tuple[str], optional
          Sequence of names for each of the node's axes. Names are used to access
          the edge that is attached to the node in a certain axis. Hence they
          should be all distinct.
     name : str, optional
          Node's name, used to access the node from de :class:`TensorNetwork`
          where it belongs. It cannot contain blank spaces.
     network : TensorNetwork, optional
          Tensor network where the node should belong. If None, a new tensor
          network, will be created to contain the node.
     param_node : bool
          Boolean indicating whether the node should be a :class:`ParamNode`
          (``True``) or a :class:`Node` (``False``).
     mean : float
          Mean of the normal distribution.
     std : float
          Standard deviation of the normal distribution.

     Returns
     -------
     Node or ParamNode
     """
     return _initializer('randn',
                         shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         param_node=param_node,
                         mean=mean,
                         std=std)
