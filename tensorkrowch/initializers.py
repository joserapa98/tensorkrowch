"""
Alternative functions to initialize nodes
"""

from typing import Optional, Text, Sequence
from tensorkrowch.network_components import Shape

from tensorkrowch.network_components import AbstractNode, Node, ParamNode
from tensorkrowch.network_components import TensorNetwork


def _initializer(init_method,
                 shape: Optional[Shape] = None,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional[TensorNetwork] = None,
                 override_node: bool = False,
                 param_node: bool = False,
                 param_edges: bool = False,
                 **kwargs: float) -> AbstractNode:
    if not param_node:
        return Node(shape=shape,
                    axes_names=axes_names,
                    name=name,
                    network=network,
                    override_node=override_node,
                    param_edges=param_edges,
                    init_method=init_method,
                    **kwargs)
    else:
        return ParamNode(shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         override_node=override_node,
                         param_edges=param_edges,
                         init_method=init_method,
                         **kwargs)


def zeros(shape: Optional[Shape] = None,
          axes_names: Optional[Sequence[Text]] = None,
          name: Optional[Text] = None,
          network: Optional[TensorNetwork] = None,
          override_node: bool = False,
          param_node: bool = False,
          param_edges: bool = False,
          **kwargs: float) -> AbstractNode:
    return _initializer('zeros',
                        shape=shape,
                        axes_names=axes_names,
                        name=name,
                        network=network,
                        override_node=override_node,
                        param_node=param_node,
                        param_edges=param_edges,
                        **kwargs)


def ones(shape: Optional[Shape] = None,
         axes_names: Optional[Sequence[Text]] = None,
         name: Optional[Text] = None,
         network: Optional[TensorNetwork] = None,
         override_node: bool = False,
         param_node: bool = False,
         param_edges: bool = False,
         **kwargs: float) -> AbstractNode:
    return _initializer('ones',
                        shape=shape,
                        axes_names=axes_names,
                        name=name,
                        network=network,
                        override_node=override_node,
                        param_node=param_node,
                        param_edges=param_edges,
                        **kwargs)


def copy(shape: Optional[Shape] = None,
         axes_names: Optional[Sequence[Text]] = None,
         name: Optional[Text] = None,
         network: Optional[TensorNetwork] = None,
         override_node: bool = False,
         param_node: bool = False,
         param_edges: bool = False,
         **kwargs: float) -> AbstractNode:
    return _initializer('copy',
                        shape=shape,
                        axes_names=axes_names,
                        name=name,
                        network=network,
                        override_node=override_node,
                        param_node=param_node,
                        param_edges=param_edges,
                        **kwargs)


def rand(shape: Optional[Shape] = None,
         axes_names: Optional[Sequence[Text]] = None,
         name: Optional[Text] = None,
         network: Optional[TensorNetwork] = None,
         override_node: bool = False,
         param_node: bool = False,
         param_edges: bool = False,
         **kwargs: float) -> AbstractNode:
    return _initializer('rand',
                        shape=shape,
                        axes_names=axes_names,
                        name=name,
                        network=network,
                        override_node=override_node,
                        param_node=param_node,
                        param_edges=param_edges,
                        **kwargs)


def randn(shape: Optional[Shape] = None,
          axes_names: Optional[Sequence[Text]] = None,
          name: Optional[Text] = None,
          network: Optional[TensorNetwork] = None,
          override_node: bool = False,
          param_node: bool = False,
          param_edges: bool = False,
          **kwargs: float) -> AbstractNode:
    return _initializer('randn',
                        shape=shape,
                        axes_names=axes_names,
                        name=name,
                        network=network,
                        override_node=override_node,
                        param_node=param_node,
                        param_edges=param_edges,
                        **kwargs)
