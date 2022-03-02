"""
This script contains:

    Classes for Nodes and Edges:
        *Axis
        *AbstractNode:
            +Node
            +ParamNode
        *AbstractEdge:
            +Edge
            +ParamEdge

    Class for Tensor Networks:
        *TensorNetwork

    Operations:
        *connect
        *disconnect
        *get_shared_edges
        *get_batch_edges
        *contract_edges
        *contract
        *contract_between
"""

from typing import (overload, Union, Optional, Dict,
                    Sequence, Text, List, Tuple)
from abc import ABC, abstractmethod
import warnings
import copy

import torch
import torch.nn as nn
import opt_einsum

from tentorch.utils import (tab_string, check_name_style,
                            erase_enum, enum_repeated_names,
                            permute_list, is_permutation)


################################################
#                    AXIS                      #
################################################
class Axis:
    """
    Class for axes. An axis can be denoted by a number or a name.
    """

    def __init__(self,
                 num: int,
                 name: Text,
                 node: Optional['AbstractNode'] = None,
                 node1: bool = True,
                 batch: bool = False) -> None:
        """
        Create an axis for a node.

        Parameters
        ----------
        num: index in the node's axes list
        name: axis name
        node: node to which the axis belongs
        node1: boolean indicating whether `node1` of the edge
               attached to this axis is the node that contains
               the axis. If False, node is `node2` of the edge
        batch: boolean indicating whether the axis is used for
               a batch index

        Raises
        ------
        TypeError
        """

        if not isinstance(num, int):
            raise TypeError('`num` should be int type')
        self._num = num

        if not isinstance(name, str):
            raise TypeError('`name` should be str type')
        if not check_name_style(name):
            raise ValueError('Names can only contain letters, numbers and underscores')
        self._name = name

        if node is not None:
            if not isinstance(node, AbstractNode):
                raise TypeError('`node` should be AbstractNode type')
        self._node = node

        if not isinstance(node1, bool):
            raise TypeError('`node1` should be bool type')
        self._node1 = node1

        if not isinstance(batch, bool):
            raise TypeError('`batch` should be bool type')
        if ('batch' in name) or ('stack' in name) or batch:
            self._batch = True
        else:
            self._batch = False

    # properties
    @property
    def num(self) -> int:
        return self._num

    @property
    def name(self) -> Text:
        return self._name

    @name.setter
    def name(self, name: Text) -> None:
        """
        Set axis name. Should not contain blank spaces
        if it is intended to be used as index of submodules.
        """
        if not check_name_style(name):
            raise ValueError('Names can only contain letters, numbers and underscores')
        elif self.node is not None:
            self.node._change_axis_name(self, name)
        else:
            self._name = name

    @property
    def node(self) -> 'AbstractNode':
        return self._node

    @property
    def node1(self) -> bool:
        return self._node1

    @property
    def batch(self) -> bool:
        return self._batch

    @batch.setter
    def batch(self, batch: bool) -> None:
        if batch != self.batch:
            if self.node[self].is_dangling():
                if self.node is not None:
                    if self.node.network is not None:
                        if batch:
                            self.node.network._edges.remove(self.node[self])
                        else:
                            self.node.network._edges += [self.node[self]]
                self._batch = batch
            else:
                raise ValueError('Cannot change `batch` attribute of non-dangling edges')

    # methods
    def __int__(self) -> int:
        return self.num

    def __str__(self) -> Text:
        return self.name

    def __repr__(self) -> Text:
        return f'{self.__class__.__name__}( {self.name} ({self.num}) )'


################################################
#                   NODES                      #
################################################
Ax = Union[int, Text, Axis]
Shape = Union[int, Sequence[int], torch.Size]


class AbstractNode(ABC):
    """
    Abstract class for nodes. Should be subclassed.

    A node is the minimum element in a tensor network. It is
    made up of a tensor and edges that can be connected to
    other nodes.
    """

    def __init__(self,
                 shape: Shape,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None) -> None:
        """
        Create a node. Should be subclassed before usage and
        a limited number of abstract methods overridden.

        Parameters
        ----------
        shape: node shape (the shape of its tensor)
        axes_names: list of names for each of the node's axes
        name: node name

        Raises
        ------
        TypeError
        ValueError
        """

        super().__init__()

        # shape
        if shape is not None:
            if not isinstance(shape, (int, tuple, list, torch.Size)):
                raise TypeError('`shape` should be int, tuple[int, ...], list[int, ...] or torch.Size type')
            if isinstance(shape, (tuple, list)):
                for i in shape:
                    if not isinstance(i, int):
                        raise TypeError('`shape` elements should be int type')

        # axes_names
        if axes_names is None:
            axes = [Axis(num=i, name=f'axis_{i}', node=self)
                    for i, _ in enumerate(shape)]
        else:
            if not isinstance(axes_names, (tuple, list)):
                raise TypeError('`axes_names` should be tuple[str, ...] or list[str, ...] type')
            if len(axes_names) != len(shape):
                raise ValueError('`axes_names` length should match `shape` length')
            else:
                axes_names = enum_repeated_names(axes_names)
                axes = [Axis(num=i, name=name, node=self)
                        for i, name in enumerate(axes_names)]
        self._axes = axes
        self._tensor = torch.empty(shape)
        self._param_edges = False
        self._edges = []

        # name
        if name is None:
            name = self.__class__.__name__.lower()
        elif not isinstance(name, str):
            raise TypeError('`name` should be str type')
        elif not check_name_style(name):
            raise ValueError('Names can only contain letters, numbers and underscores')
        self._name = name

        # network
        self._network = None

    # ----------
    # Properties
    # ----------
    @property
    def tensor(self) -> Union[torch.Tensor, nn.Parameter]:
        return self._tensor

    @tensor.setter
    def tensor(self, tensor: torch.Tensor) -> None:
        self.set_tensor(tensor)

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def axes(self) -> List[Axis]:
        return self._axes

    @property
    def axes_names(self) -> List[Text]:
        return list(map(lambda axis: axis.name, self.axes))

    @property
    def node1_list(self) -> List[bool]:
        return list(map(lambda axis: axis.node1, self.axes))

    @property
    def edges(self) -> List['AbstractEdge']:
        return self._edges

    @property
    def name(self) -> Text:
        return self._name

    @name.setter
    def name(self, name: Text) -> None:
        if not check_name_style(name):
            raise ValueError('Names can only contain letters, numbers and underscores')
        elif self.network is not None:
            self.network._change_node_name(self, name)
        else:
            self._name = name

    @property
    def network(self) -> Optional['TensorNetwork']:
        return self._network

    @network.setter
    def network(self, network: 'TensorNetwork') -> None:
        self.move_to_network(network)

    # ----------------
    # Abstract methods
    # ----------------
    @staticmethod
    @abstractmethod
    def set_tensor_format(tensor: torch.Tensor) -> Union[torch.Tensor, nn.Parameter]:
        """
        Set the tensor format for each type of node. For normal nodes the format
        is just a torch.Tensor, but for parameterized nodes it might be a nn.Parameter
        """
        pass

    @abstractmethod
    def parameterize(self, set_param: bool) -> 'AbstractNode':
        """
        Make a normal node a parametric node and viceversa, replacing the node in
        the network
        """
        pass

    @abstractmethod
    def copy(self) -> 'AbstractNode':
        """
        Copy the node, creating a new one with new edges that are reattached to it
        """
        pass

    @abstractmethod
    def permute(self, axes: Sequence[Ax]) -> 'AbstractNode':
        """
        Extend the permute function of tensors
        """
        pass

    # -------
    # Methods
    # -------
    def size(self, axis: Optional[Ax] = None) -> Union[torch.Size, int]:
        if axis is None:
            return self.shape
        axis_num = self.get_axis_number(axis)
        return self.shape[axis_num]

    def dim(self, axis: Optional[Ax] = None) -> Union[torch.Size, int]:
        """
        Similar to `size`, but if a ParamEdge is attached to an axis,
        it is returned its dimension (number of 1's in the diagonal of
        the matrix) rather than its total size (number of 1's and 0's
        in the diagonal of the matrix)

        """
        if axis is None:
            return torch.Size(list(map(lambda edge: edge.dim(), self.edges)))
        axis_num = self.get_axis_number(axis)
        return self.edges[axis_num].dim()

    def neighbours(self, axis: Optional[Ax] = None) -> Union[Optional['AbstractNode'],
                                                             List['AbstractNode']]:
        """
        Return nodes to which self is connected
        """
        node1_list = self.node1_list
        if axis is not None:
            node2 = self[axis]._nodes[node1_list[self.get_axis_number(axis)]]
            return node2
        neighbours = set()
        for i, edge in enumerate(self.edges):
            node2 = edge._nodes[node1_list[i]]
            if node2 is not None:
                neighbours.add(node2)
        return list(neighbours)

    def _change_axis_name(self, axis: Axis, name: Text) -> None:
        """
        Used to change the name of an axis. If an axis belongs to a node,
        we have to take care of repeated names. If the name that is going
        to be assigned to the axis is already set for another axis, we change
        those names by an enumerated version of them
        """
        if axis.node != self:
            raise ValueError('Cannot change the name of an axis that does '
                             'not belong to the node')
        if name != axis.name:
            axes_names = self.axes_names[:]
            for i, axis_name in enumerate(axes_names):
                if axis_name == axis.name:
                    axes_names[i] = name
                    break
            new_axes_names = enum_repeated_names(axes_names)
            for i, axis in enumerate(self.axes):
                axis._name = new_axes_names[i]

    def _change_axis_size(self,
                          axis: Ax,
                          size: int,
                          padding_method: Text = 'zeros',
                          **kwargs: float) -> None:
        """
        Change axis size, that is, change size of node's tensor and corresponding edges
        at a certain axis.

        Parameters
        ----------
        axis: axis where the size is changed
        size: new size to set
        padding_method: if new size is greater than the older one, the method used to
                        pad the new positions of the node's tensor. Available methods
                        are the same used in `make_tensor`
        kwargs: keyword arguments used in `make_tensor`
        """
        if size <= 0:
            raise ValueError('new `size` should be greater than zero')
        axis_num = self.get_axis_number(axis)
        index = list()
        for i, dim in enumerate(self.shape):
            if i == axis_num:
                if size > dim:
                    index.append(slice(size - dim, size))
                else:
                    index.append(slice(dim - size, dim))
            else:
                index.append(slice(0, dim))

        if size < self.shape[axis_num]:
            self._tensor = self.set_tensor_format(self.tensor[index])
        elif size > self.shape[axis_num]:
            new_shape = list(self.shape)
            new_shape[axis_num] = size
            new_tensor = self.make_tensor(new_shape, padding_method, **kwargs)
            new_tensor[index] = self.tensor
            self._tensor = self.set_tensor_format(new_tensor)

    def get_axis_number(self, axis: Ax) -> int:
        if isinstance(axis, int):
            for ax in self.axes:
                if axis == ax.num:
                    return ax.num
            IndexError(f'Node {self!s} has no axis with index {axis}')
        elif isinstance(axis, str):
            for ax in self.axes:
                if axis == ax.name:
                    return ax.num
            IndexError(f'Node {self!s} has no axis with name {axis}')
        elif isinstance(axis, Axis):
            for ax in self.axes:
                if axis == ax:
                    return ax.num
            IndexError(f'Node {self!s} has no axis {axis!r}')
        else:
            TypeError('`axis` should be int, str or Axis type')

    def get_edge(self, axis: Ax) -> 'AbstractEdge':
        axis_num = self.get_axis_number(axis)
        return self.edges[axis_num]

    def add_edge(self,
                 edge: 'AbstractEdge',
                 axis: Ax,
                 override: bool = False,
                 node1: Optional[bool] = None) -> None:
        """
        Add an edge to a given axis of the node.

        Parameters
        ----------
        edge: edge that is to be attached
        axis: axis to which the edge will be attached
        override: boolean indicating whether `edge` should override
                  an existing non-dangling edge at `axis`
        node1: boolean indicating if `self` is the node1 or node2 of `edge`
        """
        if edge.size() != self.size(axis):
            raise ValueError(f'Edge size should match node size at axis {axis!r}')
        if edge.dim() != self.dim(axis):
            raise ValueError(f'Edge dimension should match node dimension at axis {axis!r}')
        if node1 is None:
            if edge.node1 == self:
                node1 = True
            elif (edge.node1 != self) and (edge.node2 == self):
                node1 = False
            else:
                raise ValueError(f'If neither node1 nor node2 of `edge` is equal to {self!s}, '
                                 f'`node1` should be provided. Otherwise `edge` cannot be attached')
        axis_num = self.get_axis_number(axis)
        if (not self.edges[axis_num].is_dangling()) and (not override):
            raise ValueError(f'Node {self!s} already has a non-dangling edge for axis {axis!r}')
        self._axes[axis_num]._node1 = node1
        self._edges[axis_num] = edge

    def param_edges(self,
                    set_param: Optional[bool] = None,
                    sizes: Optional[Sequence[int]] = None) -> Optional[bool]:
        """
        Return param_edges attribute or change it if set_param is provided.

        Parameters
        ----------
        set_param: boolean indicating whether edges have to be parameterized
                   (True) or de-parameterized (False)
        sizes: if edges are parameterized, their dimensions will match the current
               shape, but a sequence of `sizes` can also be given to expand that
               shape (in that case, sizes and dimensions will be different)
        """
        if set_param is None:
            return self._param_edges
        else:
            if set_param:
                if not sizes:
                    sizes = self.shape
                elif len(sizes) != len(self.edges):
                    raise ValueError('`sizes` length should match the number of node\'s axes')
                for i, edge in enumerate(self.edges):
                    edge.parameterize(True, size=sizes[i])
            else:
                for param_edge in self.edges:
                    param_edge.parameterize(False)
            self._param_edges = set_param

    def reattach_edges(self,
                       axis: Optional[Ax] = None,
                       override: bool = False) -> None:
        """
        When a node has edges that are a reference to other previously created edges,
        those edges might make no reference to this node. With `reattach_edges`,
        node1 or node2 of all/one of the edges is redirected to the node, according
        to each axis node1 attribute.

        Parameters
        ----------
        axis: which edge is to be reattached. If None, all edges are reattached
        override: if True, node1/node2 is changed in the original edge,
                  otherwise the edge will be copied and reattached
        """
        if axis is None:
            edges = enumerate(self.edges)
        else:
            axis_num = self.get_axis_number(axis)
            edges = [(axis_num, self.edges[axis_num])]
        if not override:
            self._edges = []
            for i, edge in edges:
                new_edge = edge.copy()
                self._edges.append(new_edge)
                if self.axes[i].node1:
                    new_edge._nodes[0] = self
                    new_edge._axes[0] = self.axes[i]
                else:
                    new_edge._nodes[1] = self
                    new_edge._axes[1] = self.axes[i]
        else:
            for i, edge in enumerate(self.edges):
                if self.axes[i].node1:
                    edge._nodes[0] = self
                    edge._axes[0] = self.axes[i]
                else:
                    edge._nodes[1] = self
                    edge._axes[1] = self.axes[i]

    def disconnect_edges(self, axis: Optional[Ax] = None) -> None:
        """
        Disconnect specified edges of the node if they were connected to other nodes

        Parameters
        ----------
        axis: which edge is to be disconnected. If None, all edges are disconnected
        """
        if axis is not None:
            edges = [self[axis]]
        else:
            edges = self.edges
        for edge in edges:
            if edge.is_attached_to(self):
                if not edge.is_dangling():
                    edge | edge

    @staticmethod
    def _make_copy_tensor(shape: Shape) -> torch.Tensor:
        for i in shape[1:]:
            if i != shape[0]:
                raise ValueError(f'`shape` has unequal dimensions. Copy tensors '
                                 f'have the same dimension in all their axes.')
        copy_tensor = torch.zeros(shape)
        rank = len(shape)
        i = torch.arange(shape[0])
        copy_tensor[(i,) * rank] = 1.
        return copy_tensor

    @staticmethod
    def _make_rand_tensor(shape: Shape,
                          low: float = 0.,
                          high: float = 1.) -> torch.Tensor:
        if not isinstance(low, float):
            raise TypeError('`low` should be float type')
        if not isinstance(high, float):
            raise TypeError('`high` should be float type')
        if low >= high:
            raise ValueError('`low` should be strictly smaller than `high`')
        return torch.rand(shape) * (high - low) + low

    @staticmethod
    def _make_randn_tensor(shape: Shape,
                           mean: float = 0.,
                           std: float = 1.) -> torch.Tensor:
        if not isinstance(mean, float):
            raise TypeError('`mean` should be float type')
        if not isinstance(std, float):
            raise TypeError('`std` should be float type')
        if std <= 0:
            raise ValueError('`std` should be positive')
        return torch.randn(shape) * std + mean

    def make_tensor(self,
                    shape: Optional[Shape] = None,
                    init_method: Text = 'zeros',
                    **kwargs: float) -> torch.Tensor:
        if shape is None:
            shape = self.shape
        if init_method == 'zeros':
            return torch.zeros(shape)
        elif init_method == 'ones':
            return torch.ones(shape)
        elif init_method == 'copy':
            return self._make_copy_tensor(shape)
        elif init_method == 'rand':
            return self._make_rand_tensor(shape, **kwargs)
        elif init_method == 'randn':
            return self._make_randn_tensor(shape, **kwargs)
        else:
            raise ValueError('Choose a valid `init_method`: "zeros", '
                             '"ones", "copy", "rand", "randn"')

    def set_tensor(self,
                   tensor: Optional[torch.Tensor] = None,
                   init_method: Optional[Text] = 'zeros',
                   **kwargs: float) -> None:
        """
        Set a new node's tensor or create one with `make_tensor` and set it.
        To set the tensor it is also used `set_tensor_format`, which depends
        on the type of node.
        """
        if tensor is not None:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError('`tensor` should be torch.Tensor type')
            if tensor.shape != self.shape:
                raise ValueError('`tensor` shape should match node shape')
            self._tensor = self.set_tensor_format(tensor)
        elif init_method is not None:
            tensor = self.make_tensor(init_method=init_method, **kwargs)
            self._tensor = self.set_tensor_format(tensor)
        else:
            raise ValueError('One of `tensor` or `init_method` must be provided')

    def unset_tensor(self) -> None:
        """
        Change node's tensor by an empty tensor.
        """
        self._tensor = torch.empty(self.shape)

    def move_to_network(self,
                        network: 'TensorNetwork',
                        visited: Optional[List] = None) -> None:
        """
        Move node to another network. All other nodes connected to it, or
        to a node connected to it, etc. are also moved to the new network.

        Parameters
        ----------
        network: new network to which the nodes will be moved
        visited: list indicating the nodes that are already moved to the
                 network, used by this DFS-like algorithm
        """
        if network != self.network:
            if visited is None:
                visited = []
            if self not in visited:
                if self.network is not None:
                    self.network.remove_node(self)
                network._add_node(self)
                visited.append(self)
                for neighbour in self.neighbours():
                    neighbour.move_to_network(network=network, visited=visited)

    @overload
    def __getitem__(self, key: slice) -> List['AbstractEdge']:
        pass

    @overload
    def __getitem__(self, key: Ax) -> 'AbstractEdge':
        pass

    def __getitem__(self, key: Union[slice, Ax]) -> Union[List['AbstractEdge'], 'AbstractEdge']:
        if isinstance(key, slice):
            return self.edges[key]
        return self.get_edge(key)

    # -----------------
    # Tensor operations
    # -----------------
    def sum(self, axis: Optional[Sequence[Ax]] = None) -> torch.tensor:
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_number(ax))
        return self.tensor.sum(dim=axis_num)

    def mean(self, axis: Optional[Sequence[Ax]] = None) -> torch.tensor:
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_number(ax))
        return self.tensor.mean(dim=axis_num)

    def std(self, axis: Optional[Sequence[Ax]] = None) -> torch.tensor:
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_number(ax))
        return self.tensor.std(dim=axis_num)

    def norm(self, p=2, axis: Optional[Sequence[Ax]] = None) -> torch.tensor:
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_number(ax))
        return self.tensor.norm(p=p, dim=axis_num)

    # ---------------
    # Node operations
    # ---------------
    """
    All operations return a Node, since the nodes resulting from
    tensor network operations should not be parameterized
    """
    # Contraction of all edges connecting two nodes
    def __matmul__(self, other: 'AbstractNode') -> 'Node':
        return contract_between(self, other)

    # Tensor product of two nodes
    def __mod__(self, other: 'AbstractNode') -> 'Node':
        i = 0
        self_string = ''
        for _ in self.axes:
            self_string += opt_einsum.get_symbol(i)
            i += 1
        other_string = ''
        for _ in other.axes:
            other_string += opt_einsum.get_symbol(i)
            i += 1
        einsum_string = self_string + ',' + other_string + '->' + self_string + other_string
        new_tensor = opt_einsum.contract(einsum_string, self.tensor, other.tensor)
        new_node = Node(axes_names=self.axes_names + other.axes_names,
                        name=f'product_{self.name}_{other.name}',
                        network=self.network,
                        tensor=new_tensor,
                        edges=self.edges + other.edges,
                        node1_list=self.node1_list + other.node1_list)
        return new_node

    # For element-wise operations (not tensor-network-like operations),
    # a new Node with new edges is created
    def __mul__(self, other: 'AbstractNode') -> 'Node':
        new_node = Node(axes_names=self.axes_names,
                        name=f'mul_{self.name}_{other.name}',
                        network=self.network,
                        tensor=self.tensor * other.tensor)
        return new_node

    def __add__(self, other: 'AbstractNode') -> 'Node':
        new_node = Node(axes_names=self.axes_names,
                        name=f'add_{self.name}_{other.name}',
                        network=self.network,
                        tensor=self.tensor + other.tensor)
        return new_node

    def __sub__(self, other: 'AbstractNode') -> 'Node':
        new_node = Node(axes_names=self.axes_names,
                        name=f'sub_{self.name}_{other.name}',
                        network=self.network,
                        tensor=self.tensor - other.tensor)
        return new_node

    def __str__(self) -> Text:
        return self.name

    def __repr__(self) -> Text:
        return f'{self.__class__.__name__}(\n ' \
               f'\tname: {self.name}\n' \
               f'\ttensor:\n{tab_string(repr(self.tensor.data), 2)}\n' \
               f'\taxes: {self.axes_names}\n' \
               f'\tedges:\n{tab_string(repr(self.edges), 2)})'


class Node(AbstractNode):
    """
    Base class for non-trainable nodes. Should be subclassed by
    any new class of non-trainable nodes.

    Used for fixed nodes of the network or intermediate,
    derived nodes resulting from operations between other nodes.
    """

    def __init__(self,
                 shape: Optional[Shape] = None,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 override_node: bool = False,
                 param_edges: bool = False,
                 tensor: Optional[torch.Tensor] = None,
                 edges: Optional[List['AbstractEdge']] = None,
                 node1_list: Optional[List[bool]] = None,
                 init_method: Optional[Text] = None,
                 **kwargs: float) -> None:
        """
        Parameters
        ----------

        network: tensor network to which the node belongs
        override_node: boolean used if network is not None. If node name
                       overrides an existing node name in the network, and
                       override_node is set to True, the existing node is
                       substituted by the new one
        param_edges: boolean indicating whether node's edges
                     are parameterized (trainable) or not
        tensor: tensor "contained" in the node
        edges: list of edges to attach to the node
        node1_list: list of node1 boolean values to attach to each axis
        init_method: method to use to initialize the
                     node's tensor when it is not provided
        kwargs: keyword arguments for the init_method
        """

        # shape and tensor
        if (shape is None) == (tensor is None):
            if shape is None:
                raise ValueError('One of `shape` or `tensor` must be provided')
            else:
                raise ValueError('Only one of `shape` or `tensor` should be provided')
        elif shape is not None:
            super().__init__(shape=shape, axes_names=axes_names, name=name)
            if init_method is not None:
                self.set_tensor(init_method=init_method, **kwargs)
        else:
            super().__init__(shape=tensor.shape, axes_names=axes_names, name=name)
            self.set_tensor(tensor=tensor)

        # edges
        self._param_edges = param_edges
        if edges is None:
            edges = [self.make_edge(ax)
                     for ax in self.axes]
        else:
            if node1_list is None:
                raise ValueError('If `edges` are provided, `node1_list` should also be provided')
            for i, axis in enumerate(self.axes):
                if not isinstance(node1_list[i], bool):
                    raise TypeError('`node1_list` should be List[bool] type')
                axis._node1 = node1_list[i]
        self._edges = edges

        # network
        if network is not None:
            if not isinstance(network, TensorNetwork):
                raise TypeError('`network` should be TensorNetwork type')
            network._add_node(self, override=override_node)

    # -------
    # Methods
    # -------
    @staticmethod
    def set_tensor_format(tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(tensor, nn.Parameter):
            return tensor.data
        return tensor

    def parameterize(self, set_param: bool = True) -> Union['Node', 'ParamNode']:
        if set_param:
            new_node = ParamNode(axes_names=self.axes_names,
                                 name=self.name,
                                 network=self.network,
                                 override_node=True,
                                 param_edges=self.param_edges(),
                                 tensor=self.tensor,
                                 edges=self.edges,
                                 node1_list=self.node1_list)
            new_node.reattach_edges(override=True)
            return new_node
        else:
            return self

    def copy(self) -> 'Node':
        new_node = Node(axes_names=self.axes_names,
                        name='copy_' + self.name,
                        network=self.network,
                        param_edges=self.param_edges(),
                        tensor=self.tensor,
                        edges=self.edges,
                        node1_list=self.node1_list)
        new_node.reattach_edges(override=False)
        return new_node

    def permute(self, axes: Sequence[Ax]) -> 'Node':
        """
        Extend the permute function of tensors
        """
        axes_nums = []
        for axis in axes:
            axes_nums.append(self.get_axis_number(axis))
        if not is_permutation(list(range(len(axes_nums))), axes_nums):
            raise ValueError('The provided list of axis is not a permutation of the'
                             ' axes of the node')
        else:
            new_node = Node(axes_names=permute_list(self.axes_names, axes_nums),
                            name='permute_' + self.name,
                            network=self.network,
                            param_edges=self.param_edges(),
                            tensor=self.tensor.permute(axes_nums),
                            edges=permute_list(self.edges, axes_nums),
                            node1_list=permute_list(self.node1_list, axes_nums))
            return new_node

    def make_edge(self, axis: Axis) -> Union['Edge', 'ParamEdge']:
        if self.param_edges():
            return ParamEdge(node1=self, axis1=axis)
        return Edge(node1=self, axis1=axis)


class ParamNode(AbstractNode, nn.Module):
    """
    Class for trainable nodes. Subclass of PyTorch nn.Module.
    Should be subclassed by any new class of trainable nodes.

    Used as initial nodes of a tensor network that is to be trained.
    """

    def __init__(self,
                 shape: Optional[Shape] = None,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 override_node: bool = False,
                 param_edges: bool = False,
                 tensor: Optional[torch.Tensor] = None,
                 edges: Optional[List['AbstractEdge']] = None,
                 node1_list: Optional[List[bool]] = None,
                 init_method: Optional[Text] = None,
                 **kwargs: float) -> None:
        """
        Parameters
        ----------

        network: tensor network to which the node belongs
        override_node: boolean used if network is not None. If node name
                       overrides an existing node name in the network, and
                       override_node is set to True, the existing node is
                       substituted by the new one
        param_edges: boolean indicating whether node's edges
                     are parameterized (trainable) or not
        tensor: tensor "contained" in the node
        edges: list of edges to attach to the node
        node1_list: list of node1 boolean values to attach to each axis
        init_method: method to use to initialize the
                     node's tensor when it is not provided
        kwargs: keyword arguments for the init_method
        """

        nn.Module.__init__(self)

        # shape and tensor
        if (shape is None) == (tensor is None):
            if shape is None:
                raise ValueError('One of `shape` or `tensor` must be provided')
            else:
                raise ValueError('Only one of `shape` or `tensor` should be provided')
        elif shape is not None:
            AbstractNode.__init__(self, shape=shape, axes_names=axes_names, name=name)
            if init_method is not None:
                self.set_tensor(init_method=init_method, **kwargs)
        else:
            AbstractNode.__init__(self, shape=tensor.shape, axes_names=axes_names, name=name)
            self.set_tensor(tensor=tensor)

        # edges
        self._param_edges = param_edges
        if edges is None:
            edges = [self.make_edge(ax)
                     for ax in self.axes]
        else:
            if node1_list is None:
                raise ValueError('If `edges` are provided, `node1_list` should also be provided')
            for i, axis in enumerate(self.axes):
                if not isinstance(node1_list[i], bool):
                    raise TypeError('`node1_list` should be List[bool] type')
                axis._node1 = node1_list[i]
        self._edges = edges

        # network
        if network is not None:
            if not isinstance(network, TensorNetwork):
                raise TypeError('`network` should be TensorNetwork type')
            network._add_node(self, override=override_node)

    # ----------
    # Properties
    # ----------
    @property
    def grad(self) -> Optional[torch.Tensor]:
        return self.tensor.grad

    # -------
    # Methods
    # -------
    @staticmethod
    def set_tensor_format(tensor: torch.Tensor) -> nn.Parameter:
        """
        If a nn.Parameter is provided, the ParamNode will use such parameter
        instead of creating a new nn.Parameter object, thus creating a dependence
        """
        if isinstance(tensor, nn.Parameter):
            return tensor
        return nn.Parameter(tensor)

    def parameterize(self, set_param: bool = True) -> Union['Node', 'ParamNode']:
        if not set_param:
            new_node = Node(axes_names=self.axes_names,
                            name=self.name,
                            network=self.network,
                            override_node=True,
                            param_edges=self.param_edges(),
                            tensor=self.tensor,
                            edges=self.edges,
                            node1_list=self.node1_list)
            new_node.reattach_edges(override=True)
            return new_node
        else:
            return self

    def copy(self) -> 'ParamNode':
        new_node = ParamNode(axes_names=self.axes_names,
                             name='copy_' + self.name,
                             network=self.network,
                             param_edges=self.param_edges(),
                             tensor=self.tensor,
                             edges=self.edges,
                             node1_list=self.node1_list)
        new_node.reattach_edges(override=False)
        return new_node

    def permute(self, axes: Sequence[Ax]) -> 'ParamNode':
        """
        Extend the permute function of tensors
        """
        axes_nums = []
        for axis in axes:
            axes_nums.append(self.get_axis_number(axis))
        if not is_permutation(list(range(len(axes_nums))), axes_nums):
            raise ValueError('The provided list of axis is not a permutation of the'
                             ' axes of the node')
        else:
            new_node = ParamNode(axes_names=permute_list(self.axes_names, axes_nums),
                                 name='permute_' + self.name,
                                 network=self.network,
                                 param_edges=self.param_edges(),
                                 tensor=self.tensor.permute(axes_nums),
                                 edges=permute_list(self.edges, axes_nums),
                                 node1_list=permute_list(self.node1_list, axes_nums))
            return new_node

    def make_edge(self, axis: Axis) -> Union['ParamEdge', 'Edge']:
        if self.param_edges():
            return ParamEdge(node1=self, axis1=axis)
        return Edge(node1=self, axis1=axis)


################################################
#                   EDGES                      #
################################################
EdgeParameter = Union[int, float, nn.Parameter]
_DEFAULT_SHIFT = -0.5
_DEFAULT_SLOPE = 20.


class AbstractEdge(ABC):
    """
    Abstract class for edges. Should be subclassed.

    An edge is just a wrap up of references to the nodes it connects.
    """

    def __init__(self,
                 node1: AbstractNode,
                 axis1: Axis,
                 node2: Optional[AbstractNode] = None,
                 axis2: Optional[Axis] = None) -> None:
        """
        Create an edge. Should be subclassed before usage and
        a limited number of abstract methods overridden.

        Parameters
        ----------
        node1: first node to which the edge is connected
        axis1: axis of `node1` where the edge is attached
        node2: second, optional, node to which the edge is connected
        axis2: axis of `node2` where the edge is attached

        Raises
        ------
        ValueError
        TypeError
        """

        super().__init__()

        # node1 and axis1
        if not isinstance(node1, AbstractNode):
            raise TypeError('`node1` should be AbstractNode type')
        if not isinstance(axis1, Axis):
            raise TypeError('`axis1` should be Axis type')

        # node2 and axis2
        if (node2 is None) != (axis2 is None):
            raise ValueError('`node2` and `axis2` must either be both None or both not be None')
        if node2 is not None:
            if node1.shape[axis1.num] != node2.shape[axis2.num]:
                raise ValueError('Shapes of `axis1` and `axis2` should match')
            if (node2 == node1) and (axis2 == axis1):
                raise ValueError('Cannot connect the same axis of the same node to itself')

        self._nodes = [node1, node2]
        self._axes = [axis1, axis2]

    # ----------
    # Properties
    # ----------
    @property
    def node1(self) -> AbstractNode:
        return self._nodes[0]

    @property
    def node2(self) -> AbstractNode:
        return self._nodes[1]

    @property
    def axis1(self) -> Axis:
        return self._axes[0]

    @property
    def axis2(self) -> Axis:
        return self._axes[1]

    @property
    def name(self) -> Text:
        if self.is_dangling():
            return f'{self.node1.name}[{self.axis1.name}] <-> None'
        return f'{self.node1.name}[{self.axis1.name}] <-> ' \
               f'{self.node2.name}[{self.axis2.name}]'

    # ----------------
    # Abstract methods
    # ----------------
    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def change_size(self, size: int, padding_method: Text = 'zeros', **kwargs) -> None:
        """
        Change size of edge, thus changing sizes of adjacent nodes (node1 and node2)
        at axis1 and axis2, respectively
        """
        pass

    @abstractmethod
    def parameterize(self,
                     set_param: bool,
                     size: Optional[int] = None) -> 'AbstractEdge':
        """
        Substitute current edge by a (de-)parameterized version of it
        """
        pass

    @abstractmethod
    def copy(self) -> 'AbstractEdge':
        """
        Create a new edge referencing the same nodes at the same axis
        """
        pass

    @abstractmethod
    def __xor__(self, other: 'AbstractEdge') -> 'AbstractEdge':
        """
        Connect two edges
        """
        pass

    @abstractmethod
    def __or__(self, other: 'AbstractEdge') -> List['AbstractEdge']:
        """
        Disconnect one edge (from itself)
        """
        pass

    # -------
    # Methods
    # -------
    def is_dangling(self) -> bool:
        return self.node2 is None

    def is_batch(self) -> bool:
        if self.is_dangling():
            return self.axis1.batch
        return False

    def is_attached_to(self, node: AbstractNode) -> bool:
        return (self.node1 == node) or (self.node2 == node)

    def size(self) -> int:
        return self.node1.size(self.axis1)

    def contract(self) -> Node:
        return contract(self)

    def __str__(self) -> Text:
        return self.name

    def __repr__(self) -> Text:
        if self.is_dangling():
            return f'{self.__class__.__name__}( {self.name} )  (Dangling Edge)'
        return f'{self.__class__.__name__}( {self.name} )'


class Edge(AbstractEdge):
    """
    Base class for non-trainable edges. Should be subclassed
    by any new class of non-trainable edges.

    Used by default to create a non-trainable node.
    """

    # -------
    # Methods
    # -------
    def dim(self) -> int:
        return self.size()

    def change_size(self, size: int, padding_method: Text = 'zeros', **kwargs) -> None:
        if not isinstance(size, int):
            TypeError('`size` should be int type')
        if not self.is_dangling():
            self.node2._change_axis_size(self.axis2, size, padding_method, **kwargs)
        self.node1._change_axis_size(self.axis1, size, padding_method, **kwargs)

    def parameterize(self,
                     set_param: bool = True,
                     size: Optional[int] = None) -> Union['Edge', 'ParamEdge']:
        if set_param:
            dim = self.dim()
            if size is not None:
                if size > dim:
                    self.change_size(size)
                elif size < dim:
                    raise ValueError(f'`size` should be greater than current '
                                     f'dimension: {dim}, or NoneType')
            new_edge = ParamEdge(node1=self.node1, axis1=self.axis1, dim=dim,
                                 node2=self.node2, axis2=self.axis2)
            if not self.is_dangling():
                self.node2.add_edge(new_edge, self.axis2, override=True)
            self.node1.add_edge(new_edge, self.axis1, override=True)
            if self.node1.network is not None:
                self.node1.network._add_param(new_edge)
            return new_edge
        else:
            return self

    def copy(self) -> 'Edge':
        new_edge = Edge(node1=self.node1, axis1=self.axis1,
                        node2=self.node2, axis2=self.axis2)
        return new_edge

    @overload
    def __xor__(self, other: 'Edge') -> 'Edge':
        pass

    @overload
    def __xor__(self, other: 'ParamEdge') -> 'ParamEdge':
        pass

    def __xor__(self, other: Union['Edge', 'ParamEdge']) -> Union['Edge', 'ParamEdge']:
        return connect(self, other)

    def __or__(self, other: 'Edge') -> Tuple['Edge', 'Edge']:
        if other == self:
            return disconnect(self)
        else:
            raise ValueError('Cannot disconnect one edge from another, different one. '
                             'Edge should be disconnected from itself')


class ParamEdge(AbstractEdge, nn.Module):
    """
    Class for trainable edges. Subclass of PyTorch nn.Module.
    Should be subclassed by any new class of trainable edges.
    """

    def __init__(self,
                 node1: AbstractNode,
                 axis1: Axis,
                 dim: Optional[int] = None,
                 shift: Optional[EdgeParameter] = None,
                 slope: Optional[EdgeParameter] = None,
                 node2: Optional[AbstractNode] = None,
                 axis2: Optional[Axis] = None) -> None:

        nn.Module.__init__(self)
        AbstractEdge.__init__(self, node1, axis1, node2, axis2)

        # batch
        if axis1.batch:
            warnings.warn('`axis1` is for a batch index. Batch edges should '
                          'not be parameterized. De-parameterize it before'
                          ' usage')
        if axis2 is not None:
            if axis2.batch:
                warnings.warn('`axis2` is for a batch index. Batch edges should '
                              'not be parameterized. De-parameterize it before'
                              ' usage')

        # shift and slope
        if dim is not None:
            if (shift is not None) or (slope is not None):
                warnings.warn('`shift` and/or `slope` might have been ignored '
                              'when initializing the edge')
            shift, slope = self.compute_parameters(node1.size(axis1), dim)
        else:
            if shift is None:
                shift = _DEFAULT_SHIFT
            if slope is None:
                slope = _DEFAULT_SLOPE

        self._sigmoid = nn.Sigmoid()
        self._matrix = None
        self._dim = None

        self._shift = None
        self._slope = None
        self.set_parameters(shift, slope)

    # ----------
    # Properties
    # ----------
    @property
    def shift(self) -> nn.Parameter:
        return self._shift

    @shift.setter
    def shift(self, shift: EdgeParameter) -> None:
        self.set_parameters(shift=shift)

    @property
    def slope(self) -> nn.Parameter:
        return self._slope

    @slope.setter
    def slope(self, slope: EdgeParameter) -> None:
        self.set_parameters(slope=slope)

    @property
    def matrix(self) -> torch.Tensor:
        if self.is_updated():
            return self._matrix
        self.set_matrix()
        return self._matrix

    @property
    def grad(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.shift.grad, self.slope.grad

    @property
    def module_name(self) -> Text:
        """
        Create adapted name to be used when calling it as a submodule of
        a tensor network
        """
        if self.is_dangling():
            return f'edge_{self.node1.name}_{self.axis1.name}'
        return f'edge_{self.node1.name}_{self.axis1.name}_' \
               f'{self.node2.name}_{self.axis2.name}'

    # -------
    # Methods
    # -------
    @staticmethod
    def compute_parameters(size: int, dim: int) -> Tuple[float, float]:
        """
        Compute shift and slope parameters given a certain size and dimension
        """
        if not isinstance(size, int):
            raise TypeError('`size` should be int type')
        if not isinstance(dim, int):
            raise TypeError('`dim` should be int type')
        if dim > size:
            raise ValueError('`dim` should be smaller or equal than `size`')
        shift = (size - dim) - 0.5
        slope = _DEFAULT_SLOPE
        return shift, slope

    def set_parameters(self,
                       shift: Optional[EdgeParameter] = None,
                       slope: Optional[EdgeParameter] = None):
        """
        Set both parameters, update them and set the new matrix
        """
        if shift is not None:
            if isinstance(shift, int):
                shift = float(shift)
                self._shift = nn.Parameter(torch.tensor(shift))
                self._prev_shift = shift
            elif isinstance(shift, float):
                self._shift = nn.Parameter(torch.tensor(shift))
                self._prev_shift = shift
            elif isinstance(shift, nn.Parameter):
                self._shift = shift
                self._prev_shift = shift.item()
            else:
                raise TypeError('`shift` should be int, float or nn.Parameter type')
        if slope is not None:
            if isinstance(slope, int):
                slope = float(slope)
                self._slope = nn.Parameter(torch.tensor(slope))
                self._prev_slope = slope
            elif isinstance(slope, float):
                self._slope = nn.Parameter(torch.tensor(slope))
                self._prev_slope = slope
            elif isinstance(slope, nn.Parameter):
                self._slope = slope
                self._prev_slope = slope.item()
            else:
                raise TypeError('`slope` should be int, float or nn.Parameter type')
        self.set_matrix()

    def is_updated(self) -> bool:
        """
        Track if shift and slope have changed during training, in order
        to set the new corresponding matrix
        """
        if (self._prev_shift == self._shift.item()) and \
                (self._prev_slope == self._slope.item()):
            return True
        return False

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return self._sigmoid(x)

    def make_matrix(self) -> torch.Tensor:
        """
        Create the matrix depending on shift and slope. The matrix is
        near the identity, although it might have some zeros in the first
        positions of the diagonal (dimension is equal to number of 1's, while
        size is equal to the matrix size)
        """
        matrix = torch.zeros((self.size(), self.size()))
        i = torch.arange(self.size())
        matrix[(i, i)] = self.sigmoid(self.slope * (i - self.shift))
        return matrix

    def set_matrix(self) -> None:
        """
        Create the matrix and set it, also updating the dimension
        """
        self._matrix = self.make_matrix()
        signs = torch.sign(self._matrix.diagonal() - 0.5)
        dim = int(torch.where(signs == 1,
                              signs, torch.zeros_like(signs)).sum())
        if dim <= 0:
            warnings.warn(f'Dimension of edge {self!r} is not greater than zero')
        self._dim = dim

    def dim(self) -> int:
        return self._dim

    def change_dim(self, dim: Optional[int] = None) -> None:
        if dim != self.dim():
            shift, slope = self.compute_parameters(self.size(), dim)
            self.set_parameters(shift, slope)

    def change_size(self, size: int, padding_method: Text = 'zeros', **kwargs) -> None:
        if not isinstance(size, int):
            TypeError('`size` should be int type')
        if not self.is_dangling():
            self.node2._change_axis_size(self.axis2, size, padding_method, **kwargs)
        self.node1._change_axis_size(self.axis1, size, padding_method, **kwargs)

        shift, slope = self.compute_parameters(size, min(size, self.dim()))
        self.set_parameters(shift, slope)

    def parameterize(self,
                     set_param: bool = True,
                     size: Optional[int] = None) -> Union['Edge', 'ParamEdge']:
        if not set_param:
            self.change_size(self.dim())
            new_edge = Edge(node1=self.node1, axis1=self.axis1,
                            node2=self.node2, axis2=self.axis2)
            if not self.is_dangling():
                self.node2.add_edge(new_edge, self.axis2, override=True)
            self.node1.add_edge(new_edge, self.axis1, override=True)
            if self.node1.network is not None:
                self.node1.network._remove_param(self)
            return new_edge
        else:
            return self

    def copy(self) -> 'ParamEdge':
        new_edge = ParamEdge(node1=self.node1, axis1=self.axis1,
                             shift=self.shift.item(), slope=self.slope.item(),
                             node2=self.node2, axis2=self.axis2)
        return new_edge

    @overload
    def __xor__(self, other: Edge) -> 'ParamEdge':
        pass

    @overload
    def __xor__(self, other: 'ParamEdge') -> 'ParamEdge':
        pass

    def __xor__(self, other: Union['Edge', 'ParamEdge']) -> 'ParamEdge':
        return connect(self, other)

    def __or__(self, other: 'ParamEdge') -> Tuple['ParamEdge', 'ParamEdge']:
        if other == self:
            return disconnect(self)
        else:
            raise ValueError('Cannot disconnect one edge from another, different one. '
                             'Edge should be disconnected from itself')


################################################
#                TENSOR NETWORK                #
################################################
class TensorNetwork(nn.Module):
    """
    General class for Tensor Networks. Subclass of PyTorch nn.Module.
    Should be subclassed to implement custom initialization and contraction
    methods that suit the particular topology of each type of Tensor
    Network.

    TensorNetwork can be instantiated to build network structures of nodes,
    and perform site-wise contractions, even though network contraction
    methods are not implemented. Useful for experimentation.
    """

    def __init__(self, name: Optional[Text] = None):
        super().__init__()
        if name is None:
            name = 'net'
        self.name = name
        self._nodes = dict()
        self._data_nodes = dict()
        self._edges = []

    @property
    def nodes(self) -> Dict[Text, AbstractNode]:
        """
        All the nodes belonging to the network (including data nodes)
        """
        return self._nodes

    @property
    def nodes_names(self) -> List[Text]:
        return list(self._nodes.keys())

    @property
    def data_nodes(self) -> Dict[Text, AbstractNode]:
        """
        Data nodes created to feed the tensor network with input data
        """
        return self._data_nodes

    @property
    def edges(self) -> List[AbstractEdge]:
        """
        List of dangling, non-batch edges of the network
        """
        return self._edges

    def _add_node(self, node: AbstractNode, override: bool = False) -> None:
        """
        Add node to the network, adding its parameters (parametric tensor anr/or edges)
        to the network parameters.

        Parameters
        ----------
        node: node to be added
        override: if the node that is to be added has the same name that other node
                  that already belongs to the network, override indicates if the first
                  node have to override the second one. If not, the names are changed
                  to avoid conflicts
        """
        if node.network == self:
            warnings.warn('`node` is already in the network')
        else:
            if override:
                # when overriding nodes, we do not take care of its edges
                # we suppose they have already been handled
                if node.name not in self.nodes_names:
                    raise ValueError('Cannot override with a node whose name is not in the network')
                prev_node = self.nodes[node.name]
                if isinstance(prev_node, ParamNode):
                    self._remove_param(prev_node)
                self._nodes[node.name] = node
                if isinstance(node, ParamNode):
                    self._add_param(node)
            else:
                if erase_enum(node.name) in map(erase_enum, self.nodes_names):
                    nodes_names = self.nodes_names + [node.name]
                    new_nodes_names = enum_repeated_names(nodes_names)
                    self._rename_nodes(nodes_names[:-1], new_nodes_names[:-1])
                    node._name = new_nodes_names[-1]
                self._nodes[node.name] = node
                if isinstance(node, ParamNode):
                    self._add_param(node)
                for edge in node.edges:
                    if isinstance(edge, ParamEdge):
                        self._add_param(edge)
            node._network = self
            self._edges += [edge for edge in node.edges if (edge.is_dangling() and not edge.is_batch())]

    def add_nodes_from(self, nodes_list: Sequence[AbstractNode]):
        for name, node in nodes_list:
            self._add_node(node)

    def remove_node(self, node: AbstractNode) -> None:
        """
        This function only removes the reference to the node, and the reference
        to the TN that is kept by the node. To completely get rid of the node,
        it should be disconnected from any other node of the TN and removed from
        the TN
        """
        del self.nodes[node.name]
        node._network = None
        if erase_enum(node.name) != node.name:
            nodes_names = self.nodes_names
            new_nodes_names = enum_repeated_names(nodes_names)
            self._rename_nodes(nodes_names, new_nodes_names)
        for edge in node.edges:
            if edge.is_dangling() and not edge.is_batch():
                self._edges.remove(edge)

    def delete_node(self, node: AbstractNode) -> None:
        """
        This function disconnects the node from its neighbours and
        removes it from the TN
        """
        node.disconnect_edges()
        self.remove_node(node)

    def _add_param(self, param: Union[ParamNode, ParamEdge]) -> None:
        """
        Add parameters of ParamNode or ParamEdge to the TN
        """
        if isinstance(param, ParamNode):
            if not hasattr(self, param.name):
                self.add_module(param.name, param)
            else:
                # Nodes names are never repeated, so it is likely that this case will never occur
                raise ValueError(f'Network already has attribute named {param.name}')
        elif isinstance(param, ParamEdge):
            if not hasattr(self, param.module_name):
                self.add_module(param.module_name, param)
            # If ParamEdge is already a submodule, it is the case in which we are
            # adding a node that "inherits" edges from previous nodes

    def _remove_param(self, param: Union[ParamNode, ParamEdge]) -> None:
        """
        Remove parameters of ParamNode or ParamEdge from the TN
        """
        if isinstance(param, ParamNode):
            if hasattr(self, param.name):
                delattr(self, param.name)
            else:
                warnings.warn('Cannot remove a parameter that is not in the network')
        elif isinstance(param, ParamEdge):
            if hasattr(self, param.module_name):
                delattr(self, param.module_name)
            else:
                warnings.warn('Cannot remove a parameter that is not in the network')

    def _rename_nodes(self, prev_names: List[Text], new_names: List[Text]) -> None:
        """
        Rename nodes in the network given the old and new lists of names
        """
        if len(prev_names) != len(new_names):
            raise ValueError('Both lists of names should have the same length')
        for prev_name, new_name in zip(prev_names, new_names):
            if prev_name != new_name:
                prev_node = self.nodes[prev_name]
                if isinstance(prev_node, ParamNode):
                    self._remove_param(prev_node)
                for edge in prev_node.edges:
                    if isinstance(edge, ParamEdge):
                        self._remove_param(edge)
                self._nodes[new_name] = self._nodes.pop(prev_name)
                prev_node._name = new_name
                if isinstance(prev_node, ParamNode):
                    self._add_param(prev_node)
                for edge in prev_node.edges:
                    if isinstance(edge, ParamEdge):
                        self._add_param(edge)

    def _change_node_name(self, node: AbstractNode, name: Text) -> None:
        """
        Used to change the name of a node. If a node belongs to a network,
        we have to take care of repeated names in the network
        """
        if node.network != self:
            raise ValueError('Cannot change the name of a node that does '
                             'not belong to the network')
        if name != node.name:
            nodes_names = self.nodes_names[:]
            for i, node_name in enumerate(nodes_names):
                if node_name == node.name:
                    nodes_names[i] = name
                    break
            new_nodes_names = enum_repeated_names(nodes_names)
            self._rename_nodes(self.nodes_names, new_nodes_names)

    def parameterize(self,
                     set_param: bool = True,
                     override: bool = False) -> 'TensorNetwork':
        """
        Parameterize all nodes and edges of the network.

        Parameters
        ----------
        set_param: boolean indicating whether the Tn has to be
                   parameterized (True) or de-parameterized (False)
        override: boolean indicating if the TN must be copied before
                  parameterized (False) or not (True)
        """
        if not override:
            new_net = copy.deepcopy(self)
            for node in new_net.nodes.values():
                node.parameterize(set_param)
                node.param_edges(set_param)
            return new_net
        else:
            for node in self.nodes.values():
                node.parameterize(set_param)
                node.param_edges(set_param)
            return self

    def initialize(self) -> None:
        """
        Initialize all nodes' tensors in the network
        """
        # Initialization methods depend on the topology of the network. Number of nodes,
        # edges and its dimensions might be relevant when specifying the initial distribution
        # (e.g. mean, std) of each node
        raise NotImplementedError('Initialization methods not implemented for generic TensorNetwork class')

    def set_data_nodes(self,
                       input_edges: Union[List[int], List[AbstractEdge]],
                       batch_sizes: Sequence[int]) -> None:
        """
        Create data nodes and connect them to the list of specified edges of the TN.
        `set_data_nodes` should be executed after instantiating a TN, before
        computing forward.

        Parameters
        ----------
        input_edges: list of edges in the same order as they are expected to be
                     contracted with each feature node of the input data_nodes
        batch_sizes: Sequence[int], list of sizes of data nodes' tensors dimensions
                     associated to batch indices. Each data node will have as many
                     batch edges as elements in `batch_sizes`
        """
        if self.data_nodes:
            raise ValueError('Tensor network data nodes should be unset in order to set new ones')
        for i, edge in enumerate(input_edges):
            if isinstance(edge, int):
                edge = self[edge]
            elif isinstance(edge, AbstractEdge):
                if edge not in self.edges:
                    raise ValueError(f'Edge {edge!r} should be a dangling edge of the Tensor Network')
            else:
                raise TypeError('`input_edges` should be List[int] or List[AbstractEdge] type')
            node = Node(shape=(*batch_sizes, edge.size()),
                        axes_names=(*[f'batch_{j}' for j in range(len(batch_sizes))],
                                    'feature'),
                        name=f'data_{i}',
                        network=self)
            node['feature'] ^ edge
            self._data_nodes[node.name] = node

    def unset_data_nodes(self) -> None:
        if self.data_nodes:
            for node in self.data_nodes.values():
                self.delete_node(node)
            self._data_nodes = dict()

    def _add_data(self, data: Sequence[torch.Tensor]) -> None:
        """
                Add data to data nodes, that is, change its tensor by a new one given a new data set.
                If each `data_node` has a different feature size, a sequence of data tensors of shape
                batch_size x feature_size_{i} is provided, one for each data node. If all feature sizes
                are equal, the tensors can be passed as a sequence or as a unique tensor with shape
                batch_size x feature_size x n_features.
                """
        """
        Add data to data nodes, that is, change their tensors by new data tensors given a new data set.
        
        Parameters
        ----------
        data: sequence of tensors, each having the same shape as the corresponding data node,
              batch_size_{0} x ... x batch_size_{n} x feature_size_{i}
        """
        if len(data) != len(self.data_nodes):
            raise IndexError(f'Number of data nodes does not match number of features '
                             f'for input data with {len(data)} features')
        for i, node in enumerate(self.data_nodes.values()):
            if data[i].shape != node.shape:
                raise ValueError(f'Input data tensor with shape {data[i].shape} does '
                                 f'not match data node shape {node.shape}')
            node.tensor = data[i]

    def contract(self) -> AbstractNode:
        """
        Contract tensor network
        """
        # Custom, optimized contraction methods should be defined for each new subclass of TensorNetwork
        raise NotImplementedError('Contraction methods not implemented for generic TensorNetwork class')

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Contract Tensor Network with input data with shape batch x n_features x feature.
        """
        raise NotImplementedError('Forward method not implemented for generic TensorNetwork class')

    def __getitem__(self, key: Union[int, Text]) -> Union[AbstractEdge, AbstractNode]:
        if isinstance(key, int):
            return self._edges[key]
        elif isinstance(key, Text):
            try:
                return self.nodes[key]
            except Exception:
                raise KeyError(f'Tensor network {self!s} does not have any node with name {key}')
        else:
            raise TypeError('`key` should be int or str type')

    def __str__(self) -> Text:
        return self.name

    def __repr__(self) -> Text:
        return f'{self.__class__.__name__}(\n ' \
               f'\tnodes: \n{tab_string(repr(list(self.nodes.keys())), 2)}\n' \
               f'\tedges:\n{tab_string(repr(self.edges), 2)})'


################################################
#                  OPERATIONS                  #
################################################
def connect(edge1: AbstractEdge,
            edge2: AbstractEdge,
            override_network: bool = False) -> Union[Edge, ParamEdge]:
    """
    Connect two dangling, non-batch edges.

    Parameters
    ----------
    edge1: first edge to be connected
    edge2: second edge to be connected
    override_network: boolean indicating whether network of node2 should
                      be overridden with network of node1, in case both
                      nodes are already in a network. If only one node
                      is in a network, the other is moved to that network
    """
    for edge in [edge1, edge2]:
        if not edge.is_dangling():
            raise ValueError(f'Edge {edge!s} is not a dangling edge. '
                             f'This edge points to nodes: {edge.node1!s} and {edge.node2!s}')
        if edge.is_batch():
            raise ValueError(f'Edge {edge!s} is a batch edge')
    if edge1 == edge2:
        raise ValueError(f'Cannot connect edge {edge1!s} to itself')
    if edge1.dim() != edge2.dim():
        raise ValueError(f'Cannot connect edges of unequal dimension. '
                         f'Dimension of edge {edge1!s}: {edge1.dim()}. '
                         f'Dimension of edge {edge2!s}: {edge2.dim()}')
    if edge1.size() != edge2.size():
        # Keep the minimum size
        if edge1.size() < edge2.size():
            edge2.change_size(edge1.size())
        elif edge1.size() > edge2.size():
            edge1.change_size(edge2.size())

    node1, axis1 = edge1.node1, edge1.axis1
    node2, axis2 = edge2.node1, edge2.axis1
    net1, net2 = node1.network, node2.network

    if net1 is not None:
        if net1 != net2:
            if (net2 is not None) and not override_network:
                raise ValueError(f'Cannot connect edges from nodes in different networks. '
                                 f'Set `override` to True if you want to override {net2!s} '
                                 f'with {net1!s} in {node1!s} and its neighbours.')
            node2.move_to_network(net1)
        net1._edges.remove(edge1)
        net1._edges.remove(edge2)
        net = net1
    else:
        if net2 is not None:
            node1.move_to_network(net2)
            net2._edges.remove(edge1)
            net2._edges.remove(edge2)
        net = net2

    if isinstance(edge1, ParamEdge) == isinstance(edge2, ParamEdge):
        if isinstance(edge1, ParamEdge):
            shift = edge1.shift
            slope = edge1.slope
            new_edge = ParamEdge(node1=node1, axis1=axis1,
                                 shift=shift, slope=slope,
                                 node2=node2, axis2=axis2)
            if net is not None:
                net._remove_param(edge1)
                net._remove_param(edge2)
                net._add_param(new_edge)
        else:
            new_edge = Edge(node1=node1, axis1=axis1,
                            node2=node2, axis2=axis2)
    else:
        if isinstance(edge1, ParamEdge):
            shift = edge1.shift
            slope = edge1.slope
            if net is not None:
                net._remove_param(edge1)
        else:
            shift = edge2.shift
            slope = edge2.slope
            if net is not None:
                net._remove_param(edge2)
        new_edge = ParamEdge(node1=node1, axis1=axis1,
                             shift=shift, slope=slope,
                             node2=node2, axis2=axis2)
        if net is not None:
            net._add_param(new_edge)

    node1.add_edge(new_edge, axis1)
    node2.add_edge(new_edge, axis2)
    return new_edge


def disconnect(edge: Union[Edge, ParamEdge]) -> Tuple[Union[Edge, ParamEdge],
                                                      Union[Edge, ParamEdge]]:
    """
    Disconnect an edge, returning a couple of dangling edges
    """
    if edge.is_dangling():
        raise ValueError('Cannot disconnect a dangling edge')

    node1, node2 = edge.node1, edge.node2
    axis1, axis2 = edge.axis1, edge.axis2
    if isinstance(edge, Edge):
        new_edge1 = Edge(node1=node1, axis1=axis1)
        new_edge2 = Edge(node1=node2, axis1=axis2)
        net = edge.node1.network
        if net is not None:
            net._edges += [new_edge1, new_edge2]
    else:
        assert isinstance(edge, ParamEdge)
        shift = edge.shift
        slope = edge.slope
        new_edge1 = ParamEdge(node1=node1, axis1=axis1,
                              shift=shift, slope=slope)
        new_edge2 = ParamEdge(node1=node2, axis1=axis2,
                              shift=shift, slope=slope)
        net = edge.node1.network
        if net is not None:
            net._remove_param(edge)
            net._add_param(new_edge1)
            net._add_param(new_edge2)
            net._edges += [new_edge1, new_edge2]

    node1.add_edge(new_edge1, axis1, override=True)
    node2.add_edge(new_edge2, axis2, override=True)
    return new_edge1, new_edge2


def get_shared_edges(node1: AbstractNode, node2: AbstractNode) -> List[AbstractEdge]:
    """
    Obtain list of edges shared between two nodes
    """
    edges = []
    for edge in node1.edges:
        if (edge in node2.edges) and (not edge.is_dangling()):
            edges.append(edge)
    return edges


def get_batch_edges(node: AbstractNode) -> List[AbstractEdge]:
    """
    Obtain list of batch edges shared between two nodes
    """
    edges = []
    for edge in node.edges:
        if edge.is_batch():
            edges.append(edge)
    return edges


def contract_edges(edges: List[AbstractEdge],
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
                if isinstance(edge, ParamEdge):
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
            node1_list.append(used_nodes[j].axes[k].node1)
            i += 1
        k += 1
        if k == len(input_strings[j]):
            k = 0
            j += 1

    # If nodes were connected, we can assume that both are in the same network
    new_node = Node(axes_names=axes_names,
                    name=new_name,
                    network=used_nodes[0].network,
                    param_edges=False,
                    tensor=new_tensor,
                    edges=edges,
                    node1_list=node1_list)
    return new_node


def contract(edge: AbstractEdge) -> Node:
    """
    Contract only one edge
    """
    return contract_edges([edge], edge.node1, edge.node2)


def contract_between(node1: AbstractNode, node2: AbstractNode) -> Node:
    """
    Contract all shared edges between two nodes, also performing batch contraction
    between batch edges that share name in both nodes
    """
    edges = get_shared_edges(node1, node2) + get_batch_edges(node1) + get_batch_edges(node2)
    if not edges:
        raise ValueError(f'No batch edges neither shared edges between '
                         f'nodes {node1!s} and {node2!s} found')
    return contract_edges(edges, node1, node2)
