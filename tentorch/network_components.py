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

    Classes for stacks:
        *StackNode
        *AbstractStackEdge:
            +StackEdge
            +ParamStackEdge

    Class for Tensor Networks:
        *TensorNetwork
"""

from typing import (overload, Union, Optional, Dict,
                    Sequence, Text, List, Tuple, Any)
from abc import ABC, abstractmethod
import warnings
import copy

import torch
from torch import Tensor, Size
import torch.nn as nn
from torch.nn import Parameter

import opt_einsum

from tentorch.utils import (tab_string, check_name_style,
                            erase_enum, enum_repeated_names,
                            permute_list, is_permutation)
import tentorch.node_operations as nop

import time


# Tensor = torch.Tensor
# Parameter = nn.Parameter
# Size = torch.Size


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
                 batch: Optional[bool] = None) -> None:
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

        # Check types
        if not isinstance(num, int):
            raise TypeError('`num` should be int type')

        if not isinstance(name, str):
            raise TypeError('`name` should be str type')
        if not check_name_style(name):
            raise ValueError('Names can only contain letters, numbers and underscores')

        if node is not None:
            if not isinstance(node, AbstractNode):
                raise TypeError('`node` should be AbstractNode type')

        if not isinstance(node1, bool):
            raise TypeError('`node1` should be bool type')

        if batch is not None:
            if not isinstance(batch, bool):
                raise TypeError('`batch` should be bool type')

        # Set attributes
        self._num = num
        self._name = name
        self._node = node
        self._node1 = node1
        if batch is None:
            if ('batch' in name) or ('stack' in name):
                self._batch = True
            else:
                self._batch = False
        else:
            self._batch = batch

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
        since it is intended to be used as name of submodules.
        """
        if not isinstance(name, str):
            raise TypeError('`name` should be str type')
        if not check_name_style(name):
            raise ValueError('Names can only contain letters, numbers and underscores')
        elif self._node is not None:
            self._node._change_axis_name(self, name)
        else:
            self._name = name

    @property
    def node(self) -> 'AbstractNode':
        return self._node

    # methods
    def is_node1(self) -> bool:
        return self._node1

    def is_batch(self, batch: Optional[bool] = None) -> bool:
        if batch is None:
            return self._batch

        if batch != self._batch:
            if self._node is not None:
                edge = self._node[self]
                if edge.is_dangling():
                    if batch:
                        self._node._network._remove_edge(edge)
                    else:
                        self._node._network._add_edge(edge)
                else:
                    raise ValueError('Cannot change `batch` attribute of non-dangling edges')
            self._batch = batch

    def __int__(self) -> int:
        return self._num

    def __str__(self) -> Text:
        return self._name

    def __repr__(self) -> Text:
        return f'{self.__class__.__name__}( {self._name} ({self._num}) )'


################################################
#                   NODES                      #
################################################
Ax = Union[int, Text, Axis]
Shape = Union[int, Sequence[int], Size]


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
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 leaf: bool = True) -> None:
        """
        Create a node. Should be subclassed before usage and
        a limited number of abstract methods overridden.

        Parameters
        ----------
        shape: node shape (the shape of its tensor, it is always provided)
        axes_names: list of names for each of the node's axes
        name: node's name
        network: tensor network to which the node belongs
        leaf: indicates if the node is a leaf node in the network

        Raises
        ------
        TypeError
        ValueError
        """

        super().__init__()

        # check shape
        if shape is not None:
            if not isinstance(shape, (int, tuple, list, Size)):
                raise TypeError('`shape` should be int, tuple[int, ...], list[int, ...] or Size type')
            if isinstance(shape, (tuple, list)):
                for i in shape:
                    if not isinstance(i, int):
                        raise TypeError('`shape` elements should be int type')

        # check axes_names
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

        # check name
        if name is None:
            name = self.__class__.__name__.lower()
        elif not isinstance(name, str):
            raise TypeError('`name` should be str type')
        elif not check_name_style(name):
            raise ValueError('Names can only contain letters, numbers and underscores')

        # check network
        if network is not None:
            if not isinstance(network, TensorNetwork):
                raise TypeError('`network` should be TensorNetwork type')
        else:
            network = TensorNetwork()

        # Set attributes
        self._tensor_info = None
        self._temp_tensor = torch.empty(shape)
        self._axes = axes
        self._edges = []
        self._name = name
        self._network = network
        self._leaf = leaf
        
        # NOTE: introducing use of successors in node instead of in TN
        self._successors = dict()

    # ----------
    # Properties
    # ----------
    # @property
    # def tensor(self) -> Union[torch.Tensor, Parameter]:
    #     if self._tensor_info is None:
    #         return self._temp_tensor
    #
    #     if self._tensor_info['address'] is None:
    #         node = self._tensor_info['node_ref']
    #     else:
    #         node = self
    #
    #     if self._tensor_info['full']:
    #         return self._network._memory_nodes[node._tensor_info['address']]
    #     return self._network._memory_nodes[node._tensor_info['address']][self._tensor_info['index']]

    @property
    def tensor(self) -> Union[torch.Tensor, Parameter]:
        total_time = time.time()
        if self._tensor_info is None:
            result = self._temp_tensor
            # print('\t\t\t\t\tTensor info None:', time.time() - total_time)
            return result

        if self._tensor_info['address'] is None:
            node = self._tensor_info['node_ref']
        else:
            node = self

        # NOTE: index mode
        # if self._tensor_info['full']:
        # NOTE: index mode
        
        # NOTE: unbind mode
        if self._tensor_info['full'] or self.name.startswith('unbind'):
        # NOTE: unbind mode
        
            # TODO: truquito para que los unbind lean su tensor directamente
            result = self._network._memory_nodes[node._tensor_info['address']]
            # print('\t\t\t\t\tFull True:', time.time() - total_time)
            return result
        network = self._network
        # print('\t\t\t\t\t\tNetwork:', time.time() - total_time)
        memory = network._memory_nodes
        # print('\t\t\t\t\t\tMemory:', time.time() - total_time)
        address = node._tensor_info['address']
        # print('\t\t\t\t\t\tAddress:', time.time() - total_time)
        tensor = memory[address]
        # print('\t\t\t\t\t\tTensor:', time.time() - total_time)
        index = self._tensor_info['stack_idx']  # TODO: 'index'
        result = tensor[index]
        # if isinstance(index, list):
        #     if len(index) == 1:
        #         result = tensor[index[0]].view(1, *tensor.shape[1:])
        #     else:
        #         result = tensor[index]
        # else:
        #     result = tensor[index]
        # print('\t\t\t\t\t\tIndex tensor:', time.time() - total_time)
        # print('\t\t\t\t\tFull False:', time.time() - total_time)
        return result

    @tensor.setter
    def tensor(self, tensor: torch.Tensor) -> None:
        self.set_tensor(tensor)

    @property
    def shape(self) -> Size:
        if hasattr(self, '_shape'):
            return self._shape
        return self.tensor.shape

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
        return list(map(lambda axis: axis._name, self._axes))

    @property
    def edges(self) -> List['AbstractEdge']:
        return self._edges

    @property
    def name(self) -> Text:
        return self._name

    @name.setter
    def name(self, name: Text) -> None:
        if not isinstance(name, str):
            raise TypeError('`name` should be str type')
        elif not check_name_style(name):
            raise ValueError('Names can only contain letters, numbers and underscores')
        self._network._change_node_name(self, name)

    @property
    def network(self) -> 'TensorNetwork':
        return self._network

    @network.setter
    def network(self, network: 'TensorNetwork') -> None:
        self.move_to_network(network)
        
    @property
    def successors(self) -> Dict[Text, 'Successor']:
        """
        Dictionary with operations' names as keys, and list of successors as values
        """
        return self._successors

    # ----------------
    # Abstract methods
    # ----------------
    @staticmethod
    @abstractmethod
    def _set_tensor_format(tensor: torch.Tensor) -> Union[torch.Tensor, Parameter]:
        """
        Set the tensor format for each type of node. For normal nodes the format
        is just a Tensor, but for parameterized nodes it should be a Parameter
        """
        pass

    @abstractmethod
    def parameterize(self, set_param: bool) -> 'AbstractNode':
        """
        Turn a normal node into a parametric node and vice versa, replacing the node in
        the network
        """
        pass

    @abstractmethod
    def copy(self) -> 'AbstractNode':
        """
        Copy the node, creating a new one with new, copied edges that are reattached to it
        """
        pass

    #@abstractmethod
    def permute(self, axes: Sequence[Ax]) -> 'AbstractNode':
        """
        Extend the permute function of tensors
        """
        #pass
        return nop.permute(self, axes)

    # -------
    # Methods
    # -------
    def is_leaf(self) -> bool:
        return self._leaf

    def size(self, axis: Optional[Ax] = None) -> Union[Size, int]:
        if axis is None:
            return self.shape
        axis_num = self.get_axis_number(axis)
        return self.shape[axis_num]

    def dim(self, axis: Optional[Ax] = None) -> Union[Size, int]:
        """
        Similar to `size`, but if a ParamEdge is attached to an axis,
        it is returned its dimension (number of 1's in the diagonal of
        the matrix) rather than its total size (number of 1's and 0's
        in the diagonal of the matrix)
        """
        if axis is None:
            return Size(map(lambda edge: edge.dim(), self.edges))
        axis_num = self.get_axis_number(axis)
        return self.edges[axis_num].dim()

    def _compatible_dims(self, tensor: Tensor) -> bool:
        """
        Check if a tensor has a shape that is compatible with the dimensions
        of the current node in order to set it as the new tensor
        """
        if len(tensor.shape) == self.rank:
            for i, dim in enumerate(tensor.shape):
                edge = self.get_edge(i)
                if not edge.is_dangling() and dim != edge.dim():
                    return False
            return True
        return False

    def is_node1(self, axis: Optional[Ax] = None) -> Union[bool, List[bool]]:
        if axis is None:
            return list(map(lambda ax: ax._node1, self._axes))
        axis_num = self.get_axis_number(axis)
        return self.axes[axis_num]._node1

    def neighbours(self, axis: Optional[Ax] = None) -> Union[Optional['AbstractNode'],
                                                             List['AbstractNode']]:
        """
        Return nodes to which self is connected
        """
        node1_list = self.is_node1()
        if axis is not None:
            node2 = self[axis]._nodes[node1_list[self.get_axis_number(axis)]]
            return node2
        neighbours = set()
        for i, edge in enumerate(self._edges):
            if not edge.is_dangling():
                node2 = edge._nodes[node1_list[i]]
                neighbours.add(node2)
        return list(neighbours)

    def _change_axis_name(self, axis: Axis, name: Text) -> None:
        """
        Used to change the name of an axis. If an axis belongs to a node,
        we have to take care of repeated names. If the name that is going
        to be assigned to the axis is already set for another axis, we change
        those names by an enumerated version of them
        """
        if axis._node != self:
            raise ValueError('Cannot change the name of an axis that does '
                             'not belong to the node')
        if name != axis._name:
            axes_names = self.axes_names[:]
            for i, axis_name in enumerate(axes_names):
                if axis_name == axis._name:
                    axes_names[i] = name
                    break
            new_axes_names = enum_repeated_names(axes_names)
            for axis, axis_name in zip(self._axes, new_axes_names):
                axis._name = axis_name

    def _change_axis_size(self, axis: Ax, size: int) -> None:
        """
        Change axis size, that is, change size of node's tensor and corresponding edges
        at a certain axis.
        """
        if size <= 0:
            raise ValueError('new `size` should be greater than zero')
        axis_num = self.get_axis_number(axis)

        if size < self.shape[axis_num]:
            index = []
            for i, dim in enumerate(self.shape):
                if i == axis_num:
                    if size > dim:
                        index.append(slice(size - dim, size)) # TODO: aqui no se entra
                    else:
                        index.append(slice(dim - size, dim))
                else:
                    index.append(slice(0, dim))
            self.tensor = self.tensor[index]

        elif size > self.shape[axis_num]:
            pad = []
            for i, dim in enumerate(self.shape):
                if i == axis_num:
                    if size > dim:
                        pad += [0, size - dim]
                    else:
                        pad += [0, 0]  # TODO: aqui no se entra
                else:
                    pad += [0, 0]
            pad.reverse()
            self.tensor = nn.functional.pad(self.tensor, pad)

    def get_axis_number(self, axis: Ax) -> int:
        if isinstance(axis, int):
            for ax in self._axes:
                if axis == ax._num:
                    return ax._num
            IndexError(f'Node {self!s} has no axis with index {axis}')
        elif isinstance(axis, str):
            for ax in self._axes:
                if axis == ax._name:
                    return ax._num
            IndexError(f'Node {self!s} has no axis with name {axis}')
        elif isinstance(axis, Axis):
            for ax in self._axes:
                if axis == ax:
                    return ax._num
            IndexError(f'Node {self!s} has no axis {axis!r}')
        else:
            TypeError('`axis` should be int, str or Axis type')

    def get_edge(self, axis: Ax) -> 'AbstractEdge':
        axis_num = self.get_axis_number(axis)
        return self._edges[axis_num]

    def _add_edge(self,
                  edge: 'AbstractEdge',
                  axis: Ax,
                  node1: bool = True) -> None:
        """
        Add an edge to a given axis of the node.

        Parameters
        ----------
        edge: edge that is to be attached
        axis: axis to which the edge will be attached
        node1: boolean indicating if `self` is the node1 or node2 of `edge`
        """
        axis_num = self.get_axis_number(axis)
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
            shape, but a sequence of `sizes` can also be given to expand that shape
            (in that case, sizes and dimensions will be different)

        Returns
        -------
        Returns True if all edges are parametric edges, False if all edges are
        non-parametric edges, and None if there are some edges of each type
        """
        if set_param is None:
            all_edges = True
            all_param_edges = True
            for edge in self._edges:
                if isinstance(edge, ParamEdge):
                    all_edges = False
                elif isinstance(edge, Edge):
                    all_param_edges = False

            if all_edges:
                return False
            elif all_param_edges:
                return True
            else:
                return None

        else:
            if set_param:
                if not sizes:
                    sizes = self.shape
                elif len(sizes) != len(self._edges):
                    raise ValueError('`sizes` length should match the number of node\'s axes')
                for i, edge in enumerate(self._edges):
                    edge.parameterize(True, size=sizes[i])
            else:
                for param_edge in self._edges:
                    param_edge.parameterize(False)

    def _reattach_edges(self, override: bool = False) -> None:
        """
        When a node has edges that are a reference to other previously created edges,
        those edges might have no reference to this node. With `reattach_edges`,
        `node1` or `node2` of all the edges is redirected to the node, according
        to each axis `node1` attribute.

        Parameters
        ----------
        override: if True, node1/node2 is changed in the original edge, otherwise
            the edge will be copied and reattached
        """
        pass # TODO: uncomment when is fixed problem in __init__ Nodes
        # for i, (edge, node1) in enumerate(zip(self._edges, self.is_node1())):
        #     if not override:
        #         edge = edge.copy()
        #         self._edges[i] = edge
        #     edge._nodes[1 - node1] = self
        #     edge._axes[1 - node1] = self._axes[i]

    def disconnect(self, axis: Optional[Ax] = None) -> None:
        """
        Disconnect specified edges of the node if they were connected to other nodes

        Parameters
        ----------
        axis: which edge is to be disconnected. If None, all edges are disconnected
        """
        if axis is not None:
            edges = [self[axis]]
        else:
            edges = self._edges

        for edge in edges:
            if edge.is_attached_to(self):
                if not edge.is_dangling():
                    edge | edge

    @staticmethod
    def _make_copy_tensor(shape: Shape, device: torch.device = torch.device('cpu')) -> Tensor:
        copy_tensor = torch.zeros(shape, device=device)
        rank = len(shape)
        i = torch.arange(min(shape), device=device)
        copy_tensor[(i,) * rank] = 1.
        return copy_tensor

    @staticmethod
    def _make_rand_tensor(shape: Shape,
                          low: float = 0.,
                          high: float = 1.,
                          device: torch.device = torch.device('cpu')) -> Tensor:
        if not isinstance(low, float):
            raise TypeError('`low` should be float type')
        if not isinstance(high, float):
            raise TypeError('`high` should be float type')
        if low >= high:
            raise ValueError('`low` should be strictly smaller than `high`')
        return torch.rand(shape, device=device) * (high - low) + low

    @staticmethod
    def _make_randn_tensor(shape: Shape,
                           mean: float = 0.,
                           std: float = 1.,
                           device: torch.device = torch.device('cpu')) -> Tensor:
        if not isinstance(mean, float):
            raise TypeError('`mean` should be float type')
        if not isinstance(std, float):
            raise TypeError('`std` should be float type')
        if std <= 0:
            raise ValueError('`std` should be positive')
        return torch.randn(shape, device=device) * std + mean

    def make_tensor(self,
                    shape: Optional[Shape] = None,
                    init_method: Text = 'zeros',
                    device: torch.device = torch.device('cpu'),
                    **kwargs: float) -> Tensor:
        if shape is None:
            shape = self.shape
        if init_method == 'zeros':
            return torch.zeros(shape, device=device)
        elif init_method == 'ones':
            return torch.ones(shape, device=device)
        elif init_method == 'copy':
            return self._make_copy_tensor(shape, device=device)
        elif init_method == 'rand':
            return self._make_rand_tensor(shape, device=device, **kwargs)
        elif init_method == 'randn':
            return self._make_randn_tensor(shape, device=device, **kwargs)
        else:
            raise ValueError('Choose a valid `init_method`: "zeros", '
                             '"ones", "copy", "rand", "randn"')

    def _unrestricted_set_tensor(self,
                                 tensor: Optional[Tensor] = None,
                                 init_method: Optional[Text] = 'zeros',
                                 device: Optional[torch.device] = None,
                                 **kwargs: float) -> None:
        """
        Set a new node's tensor or create one with `make_tensor` and set it.
        To set the tensor it is also used `set_tensor_format`, which depends
        on the type of node. This can be used in any node, even in non-leaf nodes.

        Parameters
        ----------
        tensor: new tensor to be set in the node
        init_method: if `tensor` is not provided, a new tensor is initialized
            according to `init_method`
        device: if `tensor` is not provided, device in which the new tensor
            should be initialized
        kwargs: keyword arguments for the initialization method
        """
        if tensor is not None:
            # start = time.time()
            if not isinstance(tensor, Tensor):
                raise ValueError('`tensor` should be Tensor type')
            elif not self._compatible_dims(tensor):  # False
                
                # NOTE
                if len(tensor.shape) == self.rank:
                    index = []
                    for i, dim in enumerate(tensor.shape):
                        edge = self.get_edge(i)
                        
                        if dim > edge.size():
                            index.append(slice(dim - edge.size(), dim))
                        elif dim == edge.size():
                            index.append(slice(0, dim))
                        else:
                            raise ValueError('Cannot crop tensor if its dimensions'
                                             'are less than previous dimensions')
                            
                    tensor = tensor[index]
                    
                else:
                    raise ValueError('`tensor` should have the same number of'
                                     'dimensions as previous tensor')
                # NOTE
                
                
                # TODO: digamos que podemos siempre, cuidado, dim del edge es la del node1
                # raise ValueError('`tensor` dimensions should match the '
                #                  'dimensions of non-dangling edges')
            elif device is not None:
                warnings.warn('`device` was specified but is being ignored. Provide '
                              'a tensor that is already in the required device')
            # print('Conditionals:', time.time() - start)
            # start = time.time()
            
            correct_format_tensor = self._set_tensor_format(tensor)
            # print('Set format:', time.time() - start)
            # start = time.time()

        elif init_method is not None:
            if device is None:
                device = self.tensor.device
            tensor = self.make_tensor(init_method=init_method, device=device, **kwargs)
            correct_format_tensor = self._set_tensor_format(tensor)

        else:
            raise ValueError('One of `tensor` or `init_method` must be provided')

        self._save_in_network(correct_format_tensor)
        # print('Save in network:', time.time() - start)
        
        self._shape = tensor.shape  # NOTE: new! to save shape instead of having to access the tensor each time

    def set_tensor(self,
                   tensor: Optional[Tensor] = None,
                   init_method: Optional[Text] = 'zeros',
                   device: Optional[torch.device] = None,
                   **kwargs: float) -> None:
        """
        Set a new node's tensor for leaf nodes.
        """
        if self._leaf and not self._network._contracting:
            self._unrestricted_set_tensor(tensor=tensor, init_method=init_method, device=device, **kwargs)
        else:
            raise ValueError('Node\'s tensor can only be changed if it is a leaf tensor '
                             'and the network is not in contracting mode')

    def unset_tensor(self, device: torch.device = torch.device('cpu')) -> None:
        """
        Change node's tensor by an empty tensor.
        """
        if self._leaf and not self._network._contracting:
            self.tensor = torch.empty(self.shape, device=device)

    def _assign_memory(self,
                       address: Optional[Text] = None,
                       node_ref: Optional['AbstractNode'] = None,
                       full: Optional[bool] = None,
                       stack_idx: Optional[Tuple[slice, ...]] = None,
                       index: Optional[Tuple[slice, ...]] = None) -> None:
        """
        Change information about tensor storage when we are changing memory management.
        """
        if address is not None:
            self._tensor_info['address'] = address
        if node_ref is not None:
            self._tensor_info['node_ref'] = node_ref
        if full is not None:
            self._tensor_info['full'] = full
        if stack_idx is not None:
            self._tensor_info['stack_idx'] = stack_idx
        if index is not None:
            self._tensor_info['index'] = index

    def _save_in_network(self, tensor: Union[Tensor, Parameter]) -> None:
        """
        Save new node's tensor in the network storage
        """
        self._network._memory_nodes[self._tensor_info['address']] = tensor
        if isinstance(tensor, Parameter):
            if not hasattr(self, 'param_' + self._tensor_info['address']):
                self._network.register_parameter('param_' + self._tensor_info['address'], tensor)
            else:
                raise ValueError(f'Network already has attribute named {self._tensor_info["address"]}')

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
        if network != self._network:
            if visited is None:
                visited = []
            if self not in visited:
                if self._network is not None:
                    self._network._remove_node(self)
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
            return self._edges[key]
        return self.get_edge(key)

    # -----------------
    # Tensor operations
    # -----------------
    def sum(self, axis: Optional[Sequence[Ax]] = None) -> Tensor:
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_number(ax))
        return self.tensor.sum(dim=axis_num)

    def mean(self, axis: Optional[Sequence[Ax]] = None) -> Tensor:
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_number(ax))
        return self.tensor.mean(dim=axis_num)

    def std(self, axis: Optional[Sequence[Ax]] = None) -> Tensor:
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_number(ax))
        return self.tensor.std(dim=axis_num)

    def norm(self, p=2, axis: Optional[Sequence[Ax]] = None) -> Tensor:
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
        return nop.contract_between(self, other)

    # Tensor product of two nodes
    def __mod__(self, other: 'AbstractNode') -> 'Node':
        return nop.tprod(self, other)

    # For element-wise operations (not tensor-network-like operations),
    # a new Node with new edges is created
    def __mul__(self, other: 'AbstractNode') -> 'Node':
        return nop.mul(self, other)

    def __add__(self, other: 'AbstractNode') -> 'Node':
        return nop.add(self, other)

    def __sub__(self, other: 'AbstractNode') -> 'Node':
        return nop.sub(self, other)

    def __str__(self) -> Text:
        return self.name

    def __repr__(self) -> Text:
        return f'{self.__class__.__name__}(\n ' \
               f'\tname: {self._name}\n' \
               f'\ttensor:\n{tab_string(repr(self.tensor.data), 2)}\n' \
               f'\taxes: {self.axes_names}\n' \
               f'\tedges:\n{tab_string(repr(self._edges), 2)})'


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
                 leaf: bool = True,
                 override_node: bool = False,
                 param_edges: bool = False,
                 tensor: Optional[Tensor] = None,
                 edges: Optional[List['AbstractEdge']] = None,
                 node1_list: Optional[List[bool]] = None,
                 init_method: Optional[Text] = None,
                 **kwargs: float) -> None:
        """
        Parameters
        ----------
        override_node: boolean indicating whether the node should override
            a node in the network with the same name (e.g. if we parameterize
            a node, we want to replace it in the network)
        param_edges: boolean indicating whether node's edges are parameterized
            (trainable) or not
        tensor: tensor "contained" in the node
        edges: list of edges to be attached to the node
        node1_list: list of node1 boolean values corresponding to each axis
        init_method: method to use to initialize the node's tensor when it
            is not provided
        kwargs: keyword arguments for the init_method
        """

        # shape and tensor
        if (shape is None) == (tensor is None):
            if shape is None:
                raise ValueError('One of `shape` or `tensor` must be provided')
            else:
                raise ValueError('Only one of `shape` or `tensor` should be provided')
        elif shape is not None:
            super().__init__(shape=shape,
                             axes_names=axes_names,
                             name=name,
                             network=network,
                             leaf=leaf)
        else:
            super().__init__(shape=tensor.shape,
                             axes_names=axes_names,
                             name=name,
                             network=network,
                             leaf=leaf)

        # edges
        if edges is None:
            self._edges = [self.make_edge(ax, param_edges) for ax in self.axes]
        else:
            if node1_list is None:
                raise ValueError('If `edges` are provided, `node1_list` should also be provided')
            for i, axis in enumerate(self.axes):
                if not isinstance(node1_list[i], bool):
                    raise TypeError('`node1_list` should be List[bool] type')
                axis._node1 = node1_list[i]
            self._edges = edges[:]
            if self._leaf and not self._network._contracting:
                # TODO: parameterize, permute, copy, etc.
                self._reattach_edges(override=False) 
                # TODO: no se para que puse eso, no es bueno,
                # cuando hago permute en MPs contract, acabo aqu'i, y
                # creo nuevos edges malos en lugar de los que quer'ia usar

        # network
        self._network._add_node(self, override=override_node)

        if shape is not None:
            if init_method is not None:
                self._unrestricted_set_tensor(init_method=init_method, **kwargs)
        else:
            self._unrestricted_set_tensor(tensor=tensor)

    # -------
    # Methods
    # -------
    @staticmethod
    def _set_tensor_format(tensor: Tensor) -> Tensor:
        if isinstance(tensor, Parameter):
            return tensor.detach()
        return tensor

    def parameterize(self, set_param: bool = True) -> Union['Node', 'ParamNode']:
        if set_param:
            new_node = ParamNode(axes_names=self.axes_names,
                                 name=self._name,
                                 network=self._network,
                                 override_node=True,
                                 param_edges=self.param_edges(),
                                 tensor=self.tensor,
                                 edges=self._edges,
                                 node1_list=self.is_node1())
            return new_node
        else:
            return self

    def copy(self) -> 'Node':
        new_node = Node(axes_names=self.axes_names,
                        name='copy_' + self._name,
                        network=self._network,
                        param_edges=self.param_edges(),
                        tensor=self.tensor,
                        edges=self._edges,
                        node1_list=self.is_node1())
        return new_node

    # def permute(self, axes: Sequence[Ax]) -> 'Node':
    #     """
    #     Extend the permute function of tensors
    #     """
    #     axes_nums = []
    #     for axis in axes:
    #         axes_nums.append(self.get_axis_number(axis))
    #
    #     if not is_permutation(list(range(len(axes_nums))), axes_nums):
    #         raise ValueError('The provided list of axis is not a permutation of the'
    #                          ' axes of the node')
    #     else:
    #         new_node = Node(axes_names=permute_list(self.axes_names, axes_nums),
    #                         name='permute_' + self._name,
    #                         network=self._network,
    #                         param_edges=self.param_edges(),
    #                         tensor=self.tensor.permute(axes_nums),
    #                         edges=permute_list(self._edges, axes_nums),
    #                         node1_list=permute_list(self.is_node1(), axes_nums))
    #         return new_node

    def make_edge(self, axis: Axis, param_edges: bool) -> Union['Edge', 'ParamEdge']:
        if param_edges:
            return ParamEdge(node1=self, axis1=axis)
        return Edge(node1=self, axis1=axis)


class ParamNode(AbstractNode):
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
                 leaf: bool = True,
                 override_node: bool = False,
                 param_edges: bool = False,
                 tensor: Optional[Tensor] = None,
                 edges: Optional[List['AbstractEdge']] = None,
                 node1_list: Optional[List[bool]] = None,
                 init_method: Optional[Text] = None,
                 **kwargs: float) -> None:
        """
        Parameters
        ----------
        override_node: boolean indicating whether the node should override
            a node in the network with the same name (e.g. if we parameterize
            a node, we want to replace it in the network)
        param_edges: boolean indicating whether node's edges are parameterized
            (trainable) or not
        tensor: tensor "contained" in the node
        edges: list of edges to be attached to the node
        node1_list: list of node1 boolean values corresponding to each axis
        init_method: method to use to initialize the node's tensor when it
            is not provided
        kwargs: keyword arguments for the init_method
        """

        # shape and tensor
        if (shape is None) == (tensor is None):
            if shape is None:
                raise ValueError('One of `shape` or `tensor` must be provided')
            else:
                raise ValueError('Only one of `shape` or `tensor` should be provided')
        elif shape is not None:
            AbstractNode.__init__(self,
                                  shape=shape,
                                  axes_names=axes_names,
                                  name=name,
                                  network=network,
                                  leaf=leaf)
        else:
            AbstractNode.__init__(self,
                                  shape=tensor.shape,
                                  axes_names=axes_names,
                                  name=name,
                                  network=network,
                                  leaf=leaf)

        # edges
        if edges is None:
            self._edges = [self.make_edge(ax, param_edges) for ax in self.axes]
        else:
            if node1_list is None:
                raise ValueError('If `edges` are provided, `node1_list` should also be provided')
            for i, axis in enumerate(self.axes):
                if not isinstance(node1_list[i], bool):
                    raise TypeError('`node1_list` should be List[bool] type')
                axis._node1 = node1_list[i]
            self._edges = edges[:]
            if self._leaf and not self._network._contracting:
                self._reattach_edges(override=False)

        # network
        self._network._add_node(self, override=override_node)

        if shape is not None:
            if init_method is not None:
                self._unrestricted_set_tensor(init_method=init_method, **kwargs)
        else:
            self._unrestricted_set_tensor(tensor=tensor)

    # ----------
    # Properties
    # ----------
    @property
    def grad(self) -> Optional[Tensor]:
        if self._tensor_info['address'] is None:
            aux_node = self._tensor_info['node_ref']
            aux_grad = aux_node._network._memory_nodes[aux_node._tensor_info['address']].grad
        else:
            aux_grad = self._network._memory_nodes[self._tensor_info['address']].grad

        if aux_grad is None:
            return aux_grad
        else:
            if self._tensor_info['full']:
                return aux_grad
            return aux_grad[self._tensor_info['index']]

    # -------
    # Methods
    # -------
    @staticmethod
    def _set_tensor_format(tensor: Tensor) -> Parameter:
        """
        If a Parameter is provided, the ParamNode will use such parameter
        instead of creating a new Parameter object, thus creating a dependence
        """
        if isinstance(tensor, Parameter):
            return tensor
        return Parameter(tensor)

    def parameterize(self, set_param: bool = True) -> Union['Node', 'ParamNode']:
        if not set_param:
            new_node = Node(axes_names=self.axes_names,
                            name=self._name,
                            network=self._network,
                            override_node=True,
                            param_edges=self.param_edges(),
                            tensor=self.tensor,
                            edges=self._edges,
                            node1_list=self.is_node1())
            return new_node
        else:
            return self

    def copy(self) -> 'ParamNode':
        new_node = ParamNode(axes_names=self.axes_names,
                             name='copy_' + self._name,
                             network=self._network,
                             param_edges=self.param_edges(),
                             tensor=self.tensor,
                             edges=self._edges,
                             node1_list=self.is_node1())
        return new_node

    # def permute(self, axes: Sequence[Ax]) -> 'ParamNode':
    #     """
    #     Extend the permute function of tensors
    #     """
    #     axes_nums = []
    #     for axis in axes:
    #         axes_nums.append(self.get_axis_number(axis))
    #
    #     if not is_permutation(list(range(len(axes_nums))), axes_nums):
    #         raise ValueError('The provided list of axis is not a permutation of the'
    #                          ' axes of the node')
    #     else:
    #         new_node = ParamNode(axes_names=permute_list(self.axes_names, axes_nums),
    #                              name='permute_' + self._name,
    #                              network=self._network,
    #                              param_edges=self.param_edges(),
    #                              tensor=self.tensor.permute(axes_nums),
    #                              edges=permute_list(self._edges, axes_nums),
    #                              node1_list=permute_list(self.is_node1(), axes_nums))
    #         return new_node

    def make_edge(self, axis: Axis, param_edges: bool) -> Union['ParamEdge', 'Edge']:
        if param_edges:
            return ParamEdge(node1=self, axis1=axis)
        return Edge(node1=self, axis1=axis)


################################################
#                   EDGES                      #
################################################
EdgeParameter = Union[int, float, Parameter]
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

        # check node1 and axis1
        if not isinstance(node1, AbstractNode):
            raise TypeError('`node1` should be AbstractNode type')
        if not isinstance(axis1, Axis):
            raise TypeError('`axis1` should be Axis type')

        # check node2 and axis2
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
            return f'{self.node1._name}[{self.axis1._name}] <-> None'
        return f'{self.node1._name}[{self.axis1._name}] <-> ' \
               f'{self.node2._name}[{self.axis2._name}]'

    # ----------------
    # Abstract methods
    # ----------------
    @abstractmethod
    def dim(self) -> int:
        pass

    @abstractmethod
    def change_size(self, size: int) -> None:
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

    def is_batch(self, batch: Optional[bool] = None) -> bool:
        return self.axis1.is_batch(batch)  # TODO: Gestionar al conectar el edge y demás

    def is_attached_to(self, node: AbstractNode) -> bool:
        return (self.node1 == node) or (self.node2 == node)

    def size(self) -> int:
        return self.node1.size(self.axis1)

    def contract(self) -> Node:
        return nop.contract(self)

    def svd(self,
            side='left',
            rank: Optional[int] = None,
            cum_percentage: Optional[float] = None) -> None:
        # TODO: problema del futuro xd
        contracted_node = self.contract()

        lst_permute_all = []
        lst_batches = []
        lst_batches_names = []
        lst_reshape_edges1 = []
        idx = 0
        for edge in self.node1.edges:
            if edge in contracted_node.edges:
                if edge.is_batch() and edge.axis1.name in self.node2.axes_names:
                    lst_permute_all = [idx] + lst_permute_all
                    lst_batches = [edge.size()] + lst_batches
                    lst_batches_names = [edge.axis1.name] + lst_batches_names
                else:
                    lst_permute_all.append(idx)
                    lst_reshape_edges1.append(edge.size())
                idx += 1

        lst_reshape_edges2 = []
        for edge in self.node2.edges:
            if edge in contracted_node.edges:
                lst_permute_all.append(idx)
                lst_reshape_edges2.append(edge.size())
                idx += 1

        contracted_tensor = contracted_node.tensor. \
            permute(*lst_permute_all). \
            reshape(*(lst_batches +
                      [torch.tensor(lst_reshape_edges1).prod().item()] +
                      [torch.tensor(lst_reshape_edges2).prod().item()]))
        u, s, vh = torch.linalg.svd(contracted_tensor, full_matrices=False)

        if cum_percentage is not None:
            if rank is not None:
                raise ValueError('Only one of `rank` and `cum_percentage` should be provided')
            percentages = s.cumsum(-1) / s.sum(-1).view(*s.shape[:-1], 1).expand(s.shape)
            cum_percentage_tensor = torch.tensor(cum_percentage).repeat(percentages.shape[:-1])
            rank = 0
            for i in range(percentages.shape[-1]):
                p = percentages[..., i]
                rank += 1
                if torch.ge(p, cum_percentage_tensor).all():
                    break

        if rank is None:
            raise ValueError('One of `rank` and `cum_percentage` should be provided')
        if rank <= len(s):
            u = u[..., :rank]
            s = s[..., :rank]
            vh = vh[..., :rank, :]
        else:
            rank = len(s)

        if side == 'left':
            u = u @ torch.diag_embed(s)
        elif side == 'right':
            vh = torch.diag_embed(s) @ vh
        else:
            # TODO: could be changed to bool or "node1"/"node2"
            raise ValueError('`side` can only be "left" or "right"')

        u = u.reshape(*(lst_batches + lst_reshape_edges1 + [rank]))
        vh = vh.reshape(*(lst_batches + [rank] + lst_reshape_edges2))

        n_batches = len(lst_batches)
        lst_permute1 = []
        idx = 0
        idx_batch = 0
        for edge in self.node1.edges:
            if edge == self:
                lst_permute1.append(len(u.shape) - 1)
            else:
                if edge.is_batch() and edge.axis1.name in lst_batches_names:
                    lst_permute1.append(idx_batch)
                    idx_batch += 1
                else:
                    lst_permute1.append(n_batches + idx)
                    idx += 1

        lst_permute2 = []
        idx = 0
        for edge in self.node2.edges:
            if edge == self:
                lst_permute2.append(n_batches)
            else:
                if edge.is_batch():
                    found = False
                    for idx_name, name in enumerate(lst_batches_names):
                        if edge.axis1.name == name:
                            found = True
                            lst_permute2.append(idx_name)
                    if not found:
                        lst_permute2.append(n_batches + 1 + idx)
                        idx += 1
                else:
                    lst_permute2.append(n_batches + 1 + idx)
                    idx += 1

        u = u.permute(*lst_permute1)
        vh = vh.permute(*lst_permute2)

        self.change_size(rank)
        self.node1.tensor = u
        self.node2.tensor = vh

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

    def change_size(self, size: int) -> None:
        if not isinstance(size, int):
            TypeError('`size` should be int type')
        if not self.is_dangling():
            self.node2._change_axis_size(self.axis2, size)
        self.node1._change_axis_size(self.axis1, size)

    def parameterize(self,
                     set_param: bool = True,
                     size: Optional[int] = None) -> Union['Edge', 'ParamEdge']:
        if set_param:
            dim = self.dim()
            if size is not None:
                self.change_size(size)
            new_edge = ParamEdge(node1=self.node1, axis1=self.axis1,
                                 dim=min(dim, self.size()),
                                 node2=self.node2, axis2=self.axis2)
            if not self.is_dangling():
                self.node2._add_edge(new_edge, self.axis2, False)
            self.node1._add_edge(new_edge, self.axis1, True)

            self.node1._network._remove_edge(self)
            self.node1._network._add_edge(new_edge)
            return new_edge
        else:
            return self

    def copy(self) -> 'Edge':
        # TODO: cuando copiams edge tenemos que añadirlo a la TN?
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
        return nop.connect(self, other)

    def __or__(self, other: 'Edge') -> Tuple['Edge', 'Edge']:
        if other == self:
            return nop.disconnect(self)
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

        # check batch
        if axis1._batch:
            warnings.warn('`axis1` is for a batch index. Batch edges should '
                          'not be parameterized. De-parameterize it before'
                          ' usage')
        if axis2 is not None:
            if axis2._batch:
                warnings.warn('`axis2` is for a batch index. Batch edges should '
                              'not be parameterized. De-parameterize it before'
                              ' usage')

        # shift and slope
        if dim is not None:
            if (shift is not None) or (slope is not None):
                warnings.warn('`shift` and/or `slope` might have been ignored '
                              'when initializing the edge, since dim was provided')
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
    def shift(self) -> Parameter:
        return self._shift

    @shift.setter
    def shift(self, shift: EdgeParameter) -> None:
        self.set_parameters(shift=shift)

    @property
    def slope(self) -> Parameter:
        return self._slope

    @slope.setter
    def slope(self, slope: EdgeParameter) -> None:
        self.set_parameters(slope=slope)

    @property
    def matrix(self) -> Tensor:
        self.set_matrix()
        return self._matrix

    @property
    def grad(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        return self.shift.grad, self.slope.grad

    @property
    def module_name(self) -> Text:
        """
        Create adapted name to be used when calling it as a submodule of
        a tensor network
        """
        if self.is_dangling():
            return f'edge_{self.node1._name}_{self.axis1._name}'
        return f'edge_{self.node1._name}_{self.axis1._name}_' \
               f'{self.node2._name}_{self.axis2._name}'

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
                self._shift = Parameter(torch.tensor(shift))
                self._prev_shift = shift
            elif isinstance(shift, float):
                self._shift = Parameter(torch.tensor(shift))
                self._prev_shift = shift
            elif isinstance(shift, Parameter):
                self._shift = shift
                self._prev_shift = shift.item()
            else:
                raise TypeError('`shift` should be int, float or Parameter type')
        if slope is not None:
            if isinstance(slope, int):
                slope = float(slope)
                self._slope = Parameter(torch.tensor(slope))
                self._prev_slope = slope
            elif isinstance(slope, float):
                self._slope = Parameter(torch.tensor(slope))
                self._prev_slope = slope
            elif isinstance(slope, Parameter):
                # TODO: eligible device
                self._slope = slope
                self._prev_slope = slope.item()
            else:
                raise TypeError('`slope` should be int, float or Parameter type')
        self.set_matrix()

    def is_updated(self) -> bool:  # TODO: creo que esto no sirve para nada, lo podemos borrar
        """
        Track if shift and slope have changed during training, in order
        to set the new corresponding matrix
        """
        if (self._prev_shift == self._shift.item()) and \
                (self._prev_slope == self._slope.item()):
            return True
        return False

    def sigmoid(self, x: Tensor) -> Tensor:
        return self._sigmoid(x)

    def make_matrix(self) -> Tensor:
        """
        Create the matrix depending on shift and slope. The matrix is
        near the identity, although it might have some zeros in the first
        positions of the diagonal (dimension is equal to number of 1's, while
        size is equal to the matrix size)
        """
        matrix = torch.zeros((self.size(), self.size()), device=self.shift.device)
        i = torch.arange(self.size(), device=self.shift.device)
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

    def change_size(self, size: int) -> None:
        if not isinstance(size, int):
            TypeError('`size` should be int type')
        if not self.is_dangling():
            self.node2._change_axis_size(self.axis2, size)
        self.node1._change_axis_size(self.axis1, size)

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
                self.node2._add_edge(new_edge, self.axis2, False)
            self.node1._add_edge(new_edge, self.axis1, True)

            self.node1._network._remove_edge(self)
            self.node1._network._add_edge(new_edge)
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
        return nop.connect(self, other)

    def __or__(self, other: 'ParamEdge') -> Tuple['ParamEdge', 'ParamEdge']:
        if other == self:
            return nop.disconnect(self)
        else:
            raise ValueError('Cannot disconnect one edge from another, different one. '
                             'Edge should be disconnected from itself')


################################################
#                    STACKS                    #
################################################
# TODO: hacer privados
# TODO: queda comprobar stacks
# TODO: ver si se puede reestructurar, igual un AbstractStackNode que aglutine
#  ambas clases y luego hacer subclases de Node y Paramnode
# TODO: añadir unbind como metodo interno
class StackNode(Node):
    """
    Class for stacked nodes. This is a node that stores the information
    of a list of nodes that are stacked in order to perform some operation
    """

    def __init__(self,
                 nodes: List[AbstractNode],
                 name: Optional[Text] = None,
                 tensor: Optional[Tensor] = None,
                 override_node: bool = False) -> None:

        # TODO: Y en la misma TN todos
        for i in range(len(nodes[:-1])):
            if not isinstance(nodes[i], type(nodes[i + 1])):
                raise TypeError('Cannot stack nodes of different types. Nodes '
                                'must be either all Node or all ParamNode type')
            # if nodes[i].shape != nodes[i + 1].shape:
            #     raise ValueError('Cannot stack nodes with different shapes')
            if nodes[i].axes_names != nodes[i + 1].axes_names:
                raise ValueError('Stacked nodes must have the same name for each axis')
            for edge1, edge2 in zip(nodes[i].edges, nodes[i + 1].edges):
                if not isinstance(edge1, type(edge2)):
                    raise TypeError('Cannot stack nodes with edges of different types. '
                                    'The edges that are attached to the same axis in '
                                    'each node must be either all Edge or all ParamEdge type')

        edges_dict = dict()
        node1_lists_dict = dict()
        for node in nodes:
            for axis in node._axes:
                edge = node[axis]
                if axis.name not in edges_dict:
                    edges_dict[axis._name] = [edge]
                    node1_lists_dict[axis._name] = [axis._node1]
                else:
                    edges_dict[axis._name].append(edge)
                    node1_lists_dict[axis._name].append(axis._node1)

        self._edges_dict = edges_dict
        self._node1_lists_dict = node1_lists_dict
        self.nodes = nodes

        # stacked_tensor = torch.stack([node.tensor for node in nodes])
        if tensor is None:
            tensor = nop.stack_unequal_tensors([node.tensor for node in nodes])  # TODO: not sure if this is necessary
        super().__init__(axes_names=['stack'] + nodes[0].axes_names,
                         name=name,
                         network=nodes[0]._network,
                         leaf=False,
                         override_node=override_node,
                         tensor=tensor)

    @property
    def edges_dict(self) -> Dict[Text, List[AbstractEdge]]:
        return self._edges_dict

    @property
    def node1_lists_dict(self) -> Dict[Text, List[bool]]:
        return self._node1_lists_dict

    def make_edge(self, axis: Axis, param_edges: bool) -> Union['Edge', 'ParamEdge']:
        # TODO: param_edges not used here
        if axis.num == 0:
            # Stack axis
            return Edge(node1=self, axis1=axis)
        elif isinstance(self._edges_dict[axis._name][0], Edge):
            return StackEdge(self._edges_dict[axis._name],
                             self._node1_lists_dict[axis._name],
                             node1=self, axis1=axis)
        elif isinstance(self._edges_dict[axis._name][0], ParamEdge):
            return ParamStackEdge(self._edges_dict[axis._name],
                                  self._node1_lists_dict[axis._name],
                                  node1=self,
                                  axis1=axis)

    def _assign_memory(self,
                       address: Optional[Text] = None,
                       node_ref: Optional[AbstractNode] = None,
                       full: Optional[bool] = None,
                       stack_idx: Optional[Tuple[slice, ...]] = None,
                       index: Optional[Tuple[slice, ...]] = None) -> None:
        """
        Change information about tensor storage when we are changing memory management.
        """
        # TODO: creo que ya no necesito esto
        # for node in self.nodes:
        #     # TODO: Para cuando cambiamos de nombre la stack
        #     node._assign_memory(address=address)
        # self._tensor_info = {'address': address,
        #                      'full': full,
        #                      'stack_idx': stack_idx,
        #                      'index': index}
        if address is not None:
            self._tensor_info['address'] = address
        if node_ref is not None:
            self._tensor_info['node_ref'] = node_ref
        if full is not None:
            self._tensor_info['full'] = full
        if stack_idx is not None:
            self._tensor_info['stack_idx'] = stack_idx
        if index is not None:
            self._tensor_info['index'] = index


class ParamStackNode(ParamNode):
    """
    Class for parametric stacked nodes. This is a node that stores the information
    of a list of parametric nodes that are stacked in order to perform some operation
    """

    def __init__(self,
                 nodes: List[AbstractNode],
                 name: Optional[Text] = None,
                 tensor: Optional[Tensor] = None,
                 override_node: bool = False) -> None:

        # TODO: Y en la misma TN todos
        for i in range(len(nodes[:-1])):
            if not isinstance(nodes[i], type(nodes[i + 1])):
                raise TypeError('Cannot stack nodes of different types. Nodes '
                                'must be either all Node or all ParamNode type')
            # if nodes[i].shape != nodes[i + 1].shape:
            #     raise ValueError('Cannot stack nodes with different shapes')
            if nodes[i].axes_names != nodes[i + 1].axes_names:
                raise ValueError('Stacked nodes must have the same name for each axis')
            for edge1, edge2 in zip(nodes[i].edges, nodes[i + 1].edges):
                if not isinstance(edge1, type(edge2)):
                    raise TypeError('Cannot stack nodes with edges of different types. '
                                    'The edges that are attached to the same axis in '
                                    'each node must be either all Edge or all ParamEdge type')

        edges_dict = dict()
        node1_lists_dict = dict()
        for node in nodes:
            for axis in node._axes:
                edge = node[axis]
                if axis.name not in edges_dict:
                    edges_dict[axis._name] = [edge]
                    node1_lists_dict[axis._name] = [axis._node1]
                else:
                    edges_dict[axis._name].append(edge)
                    node1_lists_dict[axis._name].append(axis._node1)

        self._edges_dict = edges_dict
        self._node1_lists_dict = node1_lists_dict
        self.nodes = nodes

        # stacked_tensor = torch.stack([node.tensor for node in nodes])
        if tensor is None:
            tensor = nop.stack_unequal_tensors([node.tensor for node in nodes])
        super().__init__(axes_names=['stack'] + nodes[0].axes_names,
                         name=name,
                         network=nodes[0]._network,
                         leaf=False,
                         override_node=override_node,
                         tensor=tensor)

    @property
    def edges_dict(self) -> Dict[Text, List[AbstractEdge]]:
        return self._edges_dict

    @property
    def node1_lists_dict(self) -> Dict[Text, List[bool]]:
        return self._node1_lists_dict

    def make_edge(self, axis: Axis, param_edges: bool) -> Union['Edge', 'ParamEdge']:
        # TODO: param_edges not used here
        if axis.num == 0:
            # Stack axis
            return Edge(node1=self, axis1=axis)
        elif isinstance(self._edges_dict[axis._name][0], Edge):
            return StackEdge(self._edges_dict[axis._name],
                             self._node1_lists_dict[axis._name],
                             node1=self, axis1=axis)
        elif isinstance(self._edges_dict[axis._name][0], ParamEdge):
            return ParamStackEdge(self._edges_dict[axis._name],
                                  self._node1_lists_dict[axis._name],
                                  node1=self,
                                  axis1=axis)


class AbstractStackEdge(AbstractEdge):
    """
    Abstract class for stack edges
    """

    @property
    @abstractmethod
    def edges(self) -> List[AbstractEdge]:
        pass


class StackEdge(AbstractStackEdge, Edge):
    """
    Base class for stacks of non-trainable edges.
    Used for stacked contractions
    """

    def __init__(self,
                 edges: List[Edge],
                 node1_lists: List[bool],
                 node1: Union[StackNode, ParamStackNode],
                 axis1: Axis,
                 node2: Optional[Union[StackNode, ParamStackNode]] = None,
                 axis2: Optional[Axis] = None) -> None:
        self._edges = edges
        self._node1_lists = node1_lists
        Edge.__init__(self,
                      node1=node1, axis1=axis1,
                      node2=node2, axis2=axis2)

    @property
    def edges(self) -> List[Edge]:
        return self._edges

    @property
    def node1_lists(self) -> List[bool]:
        return self._node1_lists

    def __xor__(self, other: 'StackEdge') -> Edge:
        return nop.connect_stack(self, other)


class ParamStackEdge(AbstractStackEdge, ParamEdge):
    """
    Base class for stacks of trainable edges.
    Used for stacked contractions
    """

    def __init__(self,
                 edges: List[ParamEdge],
                 node1_lists: List[bool],
                 node1: Union[StackNode, ParamStackNode],
                 axis1: Axis,
                 node2: Optional[Union[StackNode, ParamStackNode]] = None,
                 axis2: Optional[Axis] = None) -> None:
        self._edges = edges
        self._node1_lists = node1_lists
        ParamEdge.__init__(self,
                           node1=node1, axis1=axis1,
                        #    shift=self._edges[0].shift,
                        #    slope=self._edges[0].slope,
                           node2=node2, axis2=axis2)

    @property
    def edges(self) -> List[ParamEdge]:
        return self._edges

    @property
    def node1_lists(self) -> List[bool]:
        return self._node1_lists
    
    @property
    def matrix(self) -> Tensor:
        mats = []
        for edge in self.edges:
            mats.append(edge.matrix)
        return nop.stack_unequal_tensors(mats)

    def __xor__(self, other: 'ParamStackEdge') -> ParamEdge:
        return nop.connect_stack(self, other)

    # TODO: crear matrix en función de los parámetros de sus edges apilados (saber dimension maxima)


################################################
#                TENSOR NETWORK                #
################################################
class Successor:
    """
    Class for successors. Object that stores information about
    the already computed operations in the network, in order to
    compute them faster next time.
    """

    def __init__(self,
                 kwargs: Dict[Text, Any],
                 child: Union[AbstractNode, List[AbstractNode]],
                 contracting: Optional[bool] = None,
                 hints: Optional[Any] = None) -> None:
        """
        Parameters
        ----------
        kwargs: keyword arguments used in the operation
        child: node resultant from the operation
        contracting: boolean indicating whether the first time
            the operation was computed was in contracting mode
            (i.e. optimizing memory management) or not
        hints: hints created the first time the computation was
            performed, so that next times we can avoid calculating
            auxiliary information needed for the computation
        """

        self.kwargs = kwargs
        self.child = child
        self.contracting = contracting
        self.hints = hints


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
        self._memory_nodes = dict()
        self._repeated_nodes_names = dict()

        self._data_nodes = dict()
        self._memory_data_nodes = None

        self._edges = []

        self._contracting = False  # Flag to indicate whether the TN has optimized memory to perform contraction

        self._list_ops = []

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
        Add node to the network, adding its parameters (parametric tensor and/or edges)
        to the network parameters.

        Parameters
        ----------
        node: node to be added
        override: if the node that is to be added has the same name that other node
            that already belongs to the network, override indicates if the first node
            have to override the second one. If not, the names are changed to avoid
            conflicts
        """
        if override:
            prev_node = self._nodes[node.name]
            self._remove_node(prev_node)

        name = node.name # TODO: esto para evitar con nombres larguisimos de contract
        if node.name.startswith('contract_'):
            name = 'node'
        self._assign_node_name(node, name, True)
        # TODO: estoy borrando numeraci'on de nodos que ya doy numeracion,
        #  como cuando opero dos nodos y heredo un subindice. En ese caso
        #  deber'ia dejar el subindice

    def add_nodes_from(self, nodes_list: Sequence[AbstractNode]):
        for name, node in nodes_list:
            self._add_node(node)

    def _add_edge(self, edge: AbstractEdge) -> None:
        if not isinstance(edge, AbstractStackEdge):  # TODO: evitar añadir los edges de los stacks a la TN
            if isinstance(edge, ParamEdge):
                if not hasattr(self, edge.module_name):
                    # If ParamEdge is already a submodule, it is the case in which we are
                    # adding a node that "inherits" edges from previous nodes
                    self.add_module(edge.module_name, edge)
            if edge.is_dangling() and not edge.is_batch() and (edge not in self.edges):
                self._edges.append(edge)

    def _remove_edge(self, edge: AbstractEdge) -> None:
        if not isinstance(edge, AbstractStackEdge):  # TODO: evitar añadir los edges de los stacks a la TN
            if isinstance(edge, ParamEdge):
                delattr(self, edge.module_name)
            if edge in self.edges:
                self._edges.remove(edge)

    def _remove_node(self, node: AbstractNode) -> None:
        """
        This function only removes the reference to the node, and the reference
        to the TN that is kept by the node. To completely get rid of the node,
        it should be disconnected from any other node of the TN and removed from
        the TN
        """
        node._temp_tensor = node.tensor
        node._tensor_info = None
        node._network = None

        self._unassign_node_name(node)

        if node._name in self._nodes:
            if self._nodes[node._name] == node:
                del self._nodes[node._name]
                
                if node._name in self._memory_nodes:  # NOTE: puede que no est'e si usaba memory de otro nodo
                    del self._memory_nodes[node._name]

    def delete_node(self, node: AbstractNode) -> None:
        """
        This function disconnects the node from its neighbours and
        removes it from the TN
        """
        node.disconnect()
        self._remove_node(node)

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

    # def _remove_param(self, param: Union[ParamNode, ParamEdge]) -> None:
    #     """
    #     Remove parameters of ParamNode or ParamEdge from the TN
    #     """
    #     if isinstance(param, ParamNode):
    #         if hasattr(self, param.name):
    #             delattr(self, param.name)
    #         else:
    #             warnings.warn('Cannot remove a parameter that is not in the network')
    #     elif isinstance(param, ParamEdge):
    #         if hasattr(self, param.module_name):
    #             delattr(self, param.module_name)
    #         else:
    #             warnings.warn('Cannot remove a parameter that is not in the network')

    def _update_node_info(self, node: AbstractNode, new_name: Text) -> None:
        prev_name = node._name

        if new_name in self._nodes:
            aux_node = self._nodes[new_name]
            aux_node._temp_tensor = aux_node.tensor

        if self._nodes[prev_name] == node:
            self._nodes[new_name] = self._nodes.pop(prev_name)
            # TODO: A lo mejor esto solo si address is not None
            if node._tensor_info['address'] is not None:  # TODO: caso se est'a usando la memoria de otro nodo
                self._memory_nodes[new_name] = self._memory_nodes.pop(prev_name)
                node._assign_memory(address=new_name)
        else:
            self._nodes[new_name] = node
            self._memory_nodes[new_name] = node._temp_tensor
            node._temp_tensor = None
            node._assign_memory(address=new_name)
            # node._tensor_info['address'] = new_name

    def _update_node_name(self, node: AbstractNode, new_name: Text) -> None:
        if isinstance(node.tensor, Parameter):
            delattr(self, 'param_' + node._name)
        for edge in node.edges:
            self._remove_edge(edge)

        self._update_node_info(node, new_name)
        node._name = new_name

        if isinstance(node.tensor, Parameter):
            if not hasattr(self, 'param_' + node.name):
                self.register_parameter('param_' + node._name, self._memory_nodes[node._name])
            else:
                # Nodes names are never repeated, so it is likely that this case will never occur
                raise ValueError(f'Network already has attribute named {node._name}')
        for edge in node.edges:
            self._add_edge(edge)

    def _assign_node_name(self, node: AbstractNode, name: Text, first_time: bool = False) -> None:
        """
        Used to assign a new name to a node in the network
        """
        non_enum_prev_name = erase_enum(name)
        if non_enum_prev_name in self._repeated_nodes_names:
            count = self._repeated_nodes_names[non_enum_prev_name]
            if count == 1:
                aux_node = self.nodes[non_enum_prev_name]
                aux_new_name = non_enum_prev_name + '_0'
                self._update_node_name(aux_node, aux_new_name)
            new_name = non_enum_prev_name + '_' + str(count)
        else:
            new_name = non_enum_prev_name
            self._repeated_nodes_names[non_enum_prev_name] = 0
        self._repeated_nodes_names[non_enum_prev_name] += 1

        if first_time:
            self._nodes[new_name] = node
            self._memory_nodes[new_name] = node._temp_tensor
            node._tensor_info = {'address': new_name,
                                 'node_ref': None,
                                 'full': True,
                                 'stack_idx': None,
                                 'index': None}
            node._temp_tensor = None
            node._network = self
            node._name = new_name
        else:
            self._update_node_info(node, new_name)
            node._name = new_name

        if isinstance(node.tensor, Parameter):
            if not hasattr(self, 'param_' + node.name):
                self.register_parameter('param_' + node._name, self._memory_nodes[node._name])
            else:
                # TODO: Nodes names are never repeated, so it is likely that this case will never occur
                raise ValueError(f'Network already has attribute named {node._name}')
        for edge in node.edges:
            self._add_edge(edge)

    def _unassign_node_name(self, node: AbstractNode):
        """
        Modify remaining nodes names when we remove one node
        """
        if isinstance(node.tensor, Parameter):
            delattr(self, 'param_' + node._name)
        for edge in node.edges:
            self._remove_edge(edge)

        non_enum_prev_name = erase_enum(node.name)
        count = self._repeated_nodes_names[non_enum_prev_name]
        if count > 1:
            enum = int(node.name.split('_')[-1])
            for i in range(enum + 1, count):
                aux_prev_name = non_enum_prev_name + '_' + str(i)
                aux_new_name = non_enum_prev_name + '_' + str(i - 1)
                aux_node = self.nodes[aux_prev_name]
                self._update_node_name(aux_node, aux_new_name)

        self._repeated_nodes_names[non_enum_prev_name] -= 1
        count -= 1
        if count == 0:
            del self._repeated_nodes_names[non_enum_prev_name]
        elif count == 1:
            aux_prev_name = non_enum_prev_name + '_0'
            aux_new_name = non_enum_prev_name
            aux_node = self.nodes[aux_prev_name]
            self._update_node_name(aux_node, aux_new_name)

    def _change_node_name(self, node: AbstractNode, name: Text) -> None:
        """
        Used to change the name of a node. If a node belongs to a network,
        we have to take care of repeated names in the network. This entails
        assigning a new name to the node, and removing the previous name
        (with subsequent changes)
        """
        # TODO: Esto no pasa, est'a protegida, solo la llamo cuando quiero
        if node.network != self:
            raise ValueError('Cannot change the name of a node that does '
                             'not belong to the network')

        if erase_enum(name) != erase_enum(node.name):
            self._unassign_node_name(node)
            self._assign_node_name(node, name)

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
                param_node = node.parameterize(set_param)
                param_node.param_edges(set_param)
            return new_net
        else:
            nodes = list(self.nodes.values())
            for node in nodes:
                param_node = node.parameterize(set_param)
                param_node.param_edges(set_param)
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
                     batch edges as elements in `batch_sizes`  # TODO: en realidad deberíamos permitir solo un batch
        """
        if self.data_nodes:
            raise ValueError('Tensor network data nodes should be unset in order to set new ones')

        # TODO: Stack data node donde se guardan los datos, se supone que todas las features tienen la misma dim
        stack_node = Node(shape=(*batch_sizes, len(input_edges), input_edges[0].size()),  # TODO: supongo edge es AbstractEdge
                          axes_names=('n_features',
                                      *[f'batch_{j}' for j in range(len(batch_sizes))],
                                      'feature'),
                          name=f'stack_data_memory',  # TODO: guardo aqui la memory, no uso memory_data_nodes
                          network=self)

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
                        network=self,
                        leaf=False)
            node['feature'] ^ edge
            self._data_nodes[node._name] = node  # TODO: se guarda mal, el nombre del primer data cambia de 'data' a 'data_0'

        # TODO: igual mejor como antes en un solo bucle, pero cuidado con los nombres
        #  (cuando modificamos la memoria del primer nodo su nombre es 'data', pero
        #  luego pasa a ser 'data_0' y no lo cambiamos bien)
        for i, node in enumerate(self._data_nodes.values()):
            del self._memory_nodes[node._tensor_info['address']]
            node._tensor_info['address'] = None
            node._tensor_info['node_ref'] = stack_node
            node._tensor_info['full'] = False
            node._tensor_info['stack_idx'] = i
            node._tensor_info['index'] = i

    def unset_data_nodes(self) -> None:
        if self.data_nodes:
            for node in self.data_nodes.values():
                self.delete_node(node)
            self._data_nodes = dict()

    def _add_data(self, data: Sequence[Tensor]) -> None:  # TODO: data should be stored as one stacked tensor
        """
        Add data to data nodes, that is, change their tensors by new data tensors given a new data set.
        
        Parameters
        ----------
        data: sequence of tensors, each having the same shape as the corresponding data node,
              batch_size_{0} x ... x batch_size_{n} x feature_size_{i}  # TODO: n_features x batch_size x feature_size
        """
        # if len(data) != len(self.data_nodes):
        #     raise IndexError(f'Number of data nodes does not match number of features '
        #                      f'for input data with {len(data)} features')
        # for i, node in enumerate(self.data_nodes.values()):
        #     # TODO: mientras coincidan los edges conectados, bien. Los de batch pueden ser distintos
        #     # if data[i].shape != node.shape:
        #     #     raise ValueError(f'Input data tensor with shape {data[i].shape} does '
        #     #                      f'not match data node shape {node.shape}')
        #     node._unrestricted_set_tensor(data[i])

        if data.shape[0] != len(self.data_nodes):
            raise IndexError(f'Number of data nodes does not match number of features '
                             f'for input data with {len(data)} features')

        stack_node = self['stack_data_memory']
        stack_node._unrestricted_set_tensor(data)

    def is_contracting(self, contracting: Optional[bool] = None) -> Optional[bool]:
        # TODO:
        if contracting is None:
            return self._contracting

        if self._contracting and not contracting:
            pass
            # TODO: separar las memorias, una para cada nodo de nuevo
        elif not self._contracting and contracting:
            pass
            # TODO: esto no se puede cambiar aq'i, se cambia cuando se
            #  contrae la red por primera vez. raise ValueError
        self._contracting = contracting

    def contract(self) -> Tensor:
        """
        Contract tensor network
        """
        # Custom, optimized contraction methods should be defined for each new subclass of TensorNetwork
        raise NotImplementedError('Contraction methods not implemented for generic TensorNetwork class')

    def forward(self, data: Tensor) -> Tensor:
        """
        Contract Tensor Network with input data with shape batch x n_features x feature.
        """
        # TODO: algo as'i, en la primera epoca se meten datos con batch 1, solo
        #  para ir creando todos los nodos intermedios necesarios r'apidamente,
        #  luego ya se contrae la red haciendo operaciones de tensores
        if not self.is_contracting():
            # First contraction
            aux_data = torch.zeros([1] * (len(data.shape) - 1) + [data.shape[-1]])
            self._add_data(aux_data)
            self.is_contracting(True)
            self.contract()

        self._add_data(data)
        self.contract()
        # raise NotImplementedError('Forward method not implemented for generic TensorNetwork class')

    # TODO: add_data, wrap(contract), where we only define the way in which data is fed to the TN and TN
    #  is contracted; `wrap` is used to manage memory and creation of nodes in the first epoch, feeding
    #  data (zeros only batch_size=1) with torch.no_grad()

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
               f'\tname: {self.name}' \
               f'\tnodes: \n{tab_string(repr(list(self.nodes.keys())), 2)}\n' \
               f'\tedges:\n{tab_string(repr(self.edges), 2)})'

    # TODO: Function to build instructions and reallocate memory, optimized for a function
    #  (se deben reasignar los par'ametros)
    # TODO: Function to allocate one memory tensor for each node, like old mode

# ################################################
# #                  OPERATIONS                  #
# ################################################
# def connect(edge1: AbstractEdge,
#             edge2: AbstractEdge,
#             override_network: bool = False) -> Union[Edge, ParamEdge]:
#     """
#     Connect two dangling, non-batch edges.
#
#     Parameters
#     ----------
#     edge1: first edge to be connected
#     edge2: second edge to be connected
#     override_network: boolean indicating whether network of node2 should
#                       be overridden with network of node1, in case both
#                       nodes are already in a network. If only one node
#                       is in a network, the other is moved to that network
#                       # TODO: siempre sobreviven los datos de node1, self, nodo izquierdo
#     """
#     # TODO: no puedo capar el conectar nodos no-leaf, pero no tiene el resultado esperado,
#     #  en realidad estás conectando los nodos originales (leaf)
#     for edge in [edge1, edge2]:
#         if not edge.is_dangling():
#             raise ValueError(f'Edge {edge!s} is not a dangling edge. '
#                              f'This edge points to nodes: {edge.node1!s} and {edge.node2!s}')
#         if edge.is_batch():
#             raise ValueError(f'Edge {edge!s} is a batch edge')
#     if edge1 == edge2:
#         raise ValueError(f'Cannot connect edge {edge1!s} to itself')
#     if edge1.dim() != edge2.dim():
#         raise ValueError(f'Cannot connect edges of unequal dimension. '
#                          f'Dimension of edge {edge1!s}: {edge1.dim()}. '
#                          f'Dimension of edge {edge2!s}: {edge2.dim()}')
#     if edge1.size() != edge2.size():
#         # Keep the minimum size
#         if edge1.size() < edge2.size():
#             edge2.change_size(edge1.size())
#         elif edge1.size() > edge2.size():
#             edge1.change_size(edge2.size())
#
#     node1, axis1 = edge1.node1, edge1.axis1
#     node2, axis2 = edge2.node1, edge2.axis1
#     net1, net2 = node1.network, node2.network
#
#     # TODO: siempre sobreescribir con la net1
#     if net1 is not None:
#         if net1 != net2:
#             if (net2 is not None) and not override_network:
#                 raise ValueError(f'Cannot connect edges from nodes in different networks. '
#                                  f'Set `override` to True if you want to override {net2!s} '
#                                  f'with {net1!s} in {node1!s} and its neighbours.')
#             node2.move_to_network(net1)
#         net1._remove_edge(edge1)
#         net1._remove_edge(edge2)
#         net = net1
#     else:
#         if net2 is not None:
#             node1.move_to_network(net2)
#             net2._remove_edge(edge1)
#             net2._remove_edge(edge2)
#         net = net2
#
#     if isinstance(edge1, ParamEdge) == isinstance(edge2, ParamEdge):
#         if isinstance(edge1, ParamEdge):
#             shift = edge1.shift
#             slope = edge1.slope
#             new_edge = ParamEdge(node1=node1, axis1=axis1,
#                                  shift=shift, slope=slope,
#                                  node2=node2, axis2=axis2)
#             net._add_edge(new_edge)
#         else:
#             new_edge = Edge(node1=node1, axis1=axis1,
#                             node2=node2, axis2=axis2)
#     else:
#         if isinstance(edge1, ParamEdge):
#             shift = edge1.shift
#             slope = edge1.slope
#         else:
#             shift = edge2.shift
#             slope = edge2.slope
#         new_edge = ParamEdge(node1=node1, axis1=axis1,
#                              shift=shift, slope=slope,
#                              node2=node2, axis2=axis2)
#         net._add_edge(new_edge)
#
#     node1._add_edge(new_edge, axis1)
#     node2._add_edge(new_edge, axis2)
#     return new_edge
#
#
# def disconnect(edge: Union[Edge, ParamEdge]) -> Tuple[Union[Edge, ParamEdge],
#                                                       Union[Edge, ParamEdge]]:
#     """
#     Disconnect an edge, returning a couple of dangling edges
#     """
#     if edge.is_dangling():
#         raise ValueError('Cannot disconnect a dangling edge')
#
#     node1, node2 = edge.node1, edge.node2
#     axis1, axis2 = edge.axis1, edge.axis2
#     if isinstance(edge, Edge):
#         new_edge1 = Edge(node1=node1, axis1=axis1)
#         new_edge2 = Edge(node1=node2, axis1=axis2)
#         net = edge.node1.network
#         if net is not None:
#             net._edges += [new_edge1, new_edge2]
#     else:
#         assert isinstance(edge, ParamEdge)
#         shift = edge.shift
#         slope = edge.slope
#         new_edge1 = ParamEdge(node1=node1, axis1=axis1,
#                               shift=shift, slope=slope)
#         new_edge2 = ParamEdge(node1=node2, axis1=axis2,
#                               shift=shift, slope=slope)
#         net = edge.node1.network
#         if net is not None:
#             net._remove_param(edge)
#             net._add_param(new_edge1)
#             net._add_param(new_edge2)
#             net._edges += [new_edge1, new_edge2]
#
#     node1._add_edge(new_edge1, axis1, override=True)
#     node2._add_edge(new_edge2, axis2, override=True)
#     return new_edge1, new_edge2
#
#
# # TODO: otra opcion: successors tuplas (kwargs, operation), si los nodos padres
# #  coinciden en kwargs (ya sucedio la operacion), operation guarda el objeto
# #  operacion optimizada para tensores
# class Operation:
#
#     def __init__(self, check_first, func1, func2):
#         assert isinstance(check_first, Callable)
#         assert isinstance(func1, Callable)
#         assert isinstance(func2, Callable)
#         self.func1 = func1
#         self.func2 = func2
#         self.check_first = check_first
#
#     def __call__(self, *args, **kwargs):
#         if self.check_first(*args, **kwargs):
#             return self.func1(*args, **kwargs)
#         else:
#             return self.func2(*args, **kwargs)
#
#
# def get_shared_edges(node1: AbstractNode, node2: AbstractNode) -> List[AbstractEdge]:
#     """
#     Obtain list of edges shared between two nodes
#     """
#     edges = []
#     for edge in node1.edges:
#         if (edge in node2.edges):  # and (not edge.is_dangling()):  # TODO: why I had this?
#             edges.append(edge)
#     return edges
#
#
# # TODO: method of nodes
# def get_batch_edges(node: AbstractNode) -> List[AbstractEdge]:
#     """
#     Obtain list of batch edges shared between two nodes
#     """
#     edges = []
#     for edge in node.edges:
#         if edge.is_batch():
#             edges.append(edge)
#     return edges
#
#
# def _check_first_contract_edges(edges: List[AbstractEdge],
#                                 node1: AbstractNode,
#                                 node2: AbstractNode) -> bool:
#     kwargs = {'edges': edges,
#               'node1': node1,
#               'node2': node2}
#     if 'contract_edges' in node1.successors:
#         for t in node1.successors['contract_edges']:
#             if t[0] == kwargs:
#                 return False
#     return True
#
#
# def _contract_edges_first(edges: List[AbstractEdge],
#                           node1: AbstractNode,
#                           node2: AbstractNode) -> Node:
#     """
#     Contract edges between two nodes.
#
#     Parameters
#     ----------
#     edges: list of edges that are to be contracted. They can be edges shared
#         between `node1` and `node2`, or batch edges that are in both nodes
#     node1: first node of the contraction
#     node2: second node of the contraction
#
#     Returns
#     -------
#     new_node: Node resultant from the contraction
#     """
#
#     if node1 == node2:
#         # TODO: hacer esto
#         raise ValueError('Trace not implemented')
#
#     # TODO: si son StackEdge, ver que todos los correspondientes edges están conectados
#
#     nodes = [node1, node2]
#     tensors = [node1.tensor, node2.tensor]
#     non_contract_edges = [dict(), dict()]
#     batch_edges = dict()
#     contract_edges = dict()
#
#     for i in range(2):
#         for j, edge in enumerate(nodes[i].edges):
#             if edge in edges:
#                 if (edge in nodes[1-i].edges) and (not edge.is_dangling()):
#                     if i == 0:
#                         if isinstance(edge, ParamEdge):
#                             # Obtain permutations
#                             permutation_dims = [k if k < j else k + 1
#                                                 for k in range(len(tensors[i].shape) - 1)] + [j]
#                             inv_permutation_dims = inverse_permutation(permutation_dims)
#
#                             # Send multiplication dimension to the end, multiply, recover original shape
#                             tensors[i] = tensors[i].permute(permutation_dims)
#                             tensors[i] = tensors[i] @ edge.matrix
#                             tensors[i] = tensors[i].permute(inv_permutation_dims)
#
#                         contract_edges[edge] = [nodes[i].shape[j]]
#
#                     contract_edges[edge].append(j)
#
#                 else:
#                     raise ValueError('All edges in `edges` should be non-dangling, '
#                                      'shared edges between `node1` and `node2`, or batch edges')
#
#             elif edge.is_batch():
#                 if i == 0:
#                     batch_in_node2 = False
#                     for aux_edge in node2.edges:
#                         if aux_edge.is_batch() and (edge.axis1.name == aux_edge.axis1.name):
#                             batch_edges[edge.axis1.name] = [node1.shape[j], j]
#                             batch_in_node2 = True
#                             break
#
#                     if not batch_in_node2:
#                         non_contract_edges[i][edge] = [nodes[i].shape[j], j]
#
#                 else:
#                     if edge.axis1.name in batch_edges:
#                         batch_edges[edge.axis1.name].append(j)
#                     else:
#                         non_contract_edges[i][edge] = [nodes[i].shape[j], j]
#
#             else:
#                 non_contract_edges[i][edge] = [nodes[i].shape[j], j]
#
#     # TODO: esto seguro que se puede hacer mejor
#     permutation_dims = [None, None]
#     permutation_dims[0] = list(map(lambda l: l[1], batch_edges.values())) + \
#                           list(map(lambda l: l[1], non_contract_edges[0].values())) + \
#                           list(map(lambda l: l[1], contract_edges.values()))
#     permutation_dims[1] = list(map(lambda l: l[2], batch_edges.values())) + \
#                           list(map(lambda l: l[2], contract_edges.values())) + \
#                           list(map(lambda l: l[1], non_contract_edges[1].values()))
#
#     aux_permutation = inverse_permutation(list(map(lambda l: l[1], batch_edges.values())) +
#                                           list(map(lambda l: l[1], non_contract_edges[0].values())))
#     aux_permutation2 = inverse_permutation(list(map(lambda l: l[1], non_contract_edges[1].values())))
#     final_inv_permutation_dims = aux_permutation + list(map(lambda x: x+len(aux_permutation), aux_permutation2))
#
#     new_shape = [None, None]
#     new_shape[0] = (torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
#                     torch.tensor(list(map(lambda l: l[0], non_contract_edges[0].values()))).prod().long().item(),
#                     torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item())
#
#     new_shape[1] = (torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
#                     torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item(),
#                     torch.tensor(list(map(lambda l: l[0], non_contract_edges[1].values()))).prod().long().item())
#
#     final_shape = list(map(lambda l: l[0], batch_edges.values())) + \
#                   list(map(lambda l: l[0], non_contract_edges[0].values())) + \
#                   list(map(lambda l: l[0], non_contract_edges[1].values()))
#
#     for i in range(2):
#         tensors[i] = tensors[i].permute(permutation_dims[i])
#         tensors[i] = tensors[i].reshape(new_shape[i])
#
#     result = tensors[0] @ tensors[1]
#     result = result.view(final_shape).permute(final_inv_permutation_dims)
#
#     indices_node1 = permute_list(list(map(lambda l: l[1], batch_edges.values())) +
#                                  list(map(lambda l: l[1], non_contract_edges[0].values())),
#                                  aux_permutation)
#     indices_node2 = list(map(lambda l: l[1], non_contract_edges[1].values()))
#     indices = [indices_node1, indices_node2]
#     final_edges = []
#     final_axes = []
#     final_node1 = []
#     for i in range(2):
#         for idx in indices[i]:
#             final_edges.append(nodes[i][idx])
#             final_axes.append(nodes[i].axes_names[idx])
#             final_node1.append(nodes[i].axes[idx].is_node1())
#
#     new_node = Node(axes_names=final_axes, name=f'contract_{node1.name}_{node2.name}', network=nodes[0].network,
#                     leaf=False, param_edges=False, tensor=result, edges=final_edges, node1_list=final_node1)
#
#     for node in nodes:
#         if 'contract_edges' in node._successors:
#             node._successors['contract_edges'].append(({'edges': edges,
#                                                         'node1': node1,
#                                                         'node2': node2},
#                                                        new_node))
#         else:
#             node._successors['contract_edges'] = [({'edges': edges,
#                                                     'node1': node1,
#                                                     'node2': node2},
#                                                    new_node)]
#
#     return new_node
#
#
# def _contract_edges_next(edges: List[AbstractEdge],
#                          node1: AbstractNode,
#                          node2: AbstractNode) -> Node:
#     """
#     Contract edges between two nodes.
#
#     Parameters
#     ----------
#     edges: list of edges that are to be contracted. They can be edges shared
#         between `node1` and `node2`, or batch edges that are in both nodes
#     node1: first node of the contraction
#     node2: second node of the contraction
#
#     Returns
#     -------
#     new_node: Node resultant from the contraction
#     """
#
#     if node1 == node2:
#         # TODO: hacer esto
#         raise ValueError('Trace not implemented')
#
#     nodes = [node1, node2]
#     tensors = [node1.tensor, node2.tensor]
#     non_contract_edges = [dict(), dict()]
#     batch_edges = dict()
#     contract_edges = dict()
#
#     for i in range(2):
#         for j, edge in enumerate(nodes[i].edges):
#             if edge in edges:
#                 if (edge in node2.edges) and (not edge.is_dangling()):
#                     if i == 0:
#                         if isinstance(edge, ParamEdge):
#                             # Obtain permutations
#                             permutation_dims = [k if k < j else k + 1
#                                                 for k in range(len(tensors[i].shape) - 1)] + [j]
#                             inv_permutation_dims = inverse_permutation(permutation_dims)
#
#                             # Send multiplication dimension to the end, multiply, recover original shape
#                             tensors[i] = tensors[i].permute(permutation_dims)
#                             tensors[i] = tensors[i] @ edge.matrix
#                             tensors[i] = tensors[i].permute(inv_permutation_dims)
#
#                         contract_edges[edge] = [nodes[i].shape[j]]
#
#                     contract_edges[edge].append(j)
#
#                 else:
#                     raise ValueError('All edges in `edges` should be non-dangling, '
#                                      'shared edges between `node1` and `node2`, or batch edges')
#
#             elif edge.is_batch():
#                 if i == 0:
#                     batch_in_node2 = False
#                     for aux_edge in node2.edges:
#                         if aux_edge.is_batch() and (edge.name == aux_edge.name):
#                             batch_edges[edge] = [node1.shape[j], j]
#                             batch_in_node2 = True
#                             break
#
#                     if not batch_in_node2:
#                         non_contract_edges[i][edge] = [nodes[i].shape[j], j]
#
#                 else:
#                     if edge in batch_edges:
#                         batch_edges[edge].append(j)
#                     else:
#                         non_contract_edges[i][edge] = [nodes[i].shape[j], j]
#
#             else:
#                 non_contract_edges[i][edge] = [nodes[i].shape[j], j]
#
#     # TODO: esto seguro que se puede hacer mejor
#     permutation_dims = [None, None]
#     permutation_dims[0] = list(map(lambda l: l[1], batch_edges.values())) + \
#                           list(map(lambda l: l[1], non_contract_edges[0].values())) + \
#                           list(map(lambda l: l[1], contract_edges.values()))
#     permutation_dims[1] = list(map(lambda l: l[2], batch_edges.values())) + \
#                           list(map(lambda l: l[2], contract_edges.values())) + \
#                           list(map(lambda l: l[1], non_contract_edges[1].values()))
#
#     aux_permutation = inverse_permutation(list(map(lambda l: l[1], batch_edges.values())) +
#                                           list(map(lambda l: l[1], non_contract_edges[0].values())))
#     aux_permutation2 = inverse_permutation(list(map(lambda l: l[1], non_contract_edges[1].values())))
#     final_inv_permutation_dims = aux_permutation + list(map(lambda x: x + len(aux_permutation), aux_permutation2))
#
#     new_shape = [None, None]
#     new_shape[0] = (torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
#                     torch.tensor(list(map(lambda l: l[0], non_contract_edges[0].values()))).prod().long().item(),
#                     torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item())
#
#     new_shape[1] = (torch.tensor(list(map(lambda l: l[0], batch_edges.values()))).prod().long().item(),
#                     torch.tensor(list(map(lambda l: l[0], contract_edges.values()))).prod().long().item(),
#                     torch.tensor(list(map(lambda l: l[0], non_contract_edges[1].values()))).prod().long().item())
#
#     final_shape = list(map(lambda l: l[0], batch_edges.values())) + \
#                   list(map(lambda l: l[0], non_contract_edges[0].values())) + \
#                   list(map(lambda l: l[0], non_contract_edges[1].values()))
#
#     for i in range(2):
#         tensors[i] = tensors[i].permute(permutation_dims[i])
#         tensors[i] = tensors[i].reshape(new_shape[i])
#
#     result = tensors[0] @ tensors[1]
#     result = result.view(final_shape).permute(final_inv_permutation_dims)
#
#     kwargs = {'edges': edges,
#               'node1': node1,
#               'node2': node2}
#     for t in node1._successors['contract_edges']:
#         if t[0] == kwargs:
#             child = t[1]
#             break
#     child.tensor = result
#
#     return child
#
#
# contract_edges = Operation(_check_first_contract_edges,
#                            _contract_edges_first,
#                            _contract_edges_next)
#
#
# # def contract_edges(edges: List[AbstractEdge],
# #                    node1: AbstractNode,
# #                    node2: AbstractNode,
# #                    operation: Optional[Text] = None) -> Node:
# #     """
# #     Contract edges between two nodes.
# #
# #     Parameters
# #     ----------
# #     edges: list of edges that are to be contracted. They can be edges shared
# #         between `node1` and `node2`, or batch edges that are in both nodes
# #     node1: first node of the contraction
# #     node2: second node of the contraction
# #     operation: operation string referencing the operation form which
# #         `contract_between` is called
# #
# #     Returns
# #     -------
# #     new_node: Node resultant from the contraction
# #     """
# #     all_shared_edges = get_shared_edges(node1, node2)
# #     shared_edges = []
# #     batch_edges = dict()
# #     for edge in edges:
# #         if edge in all_shared_edges:
# #             shared_edges.append(edge)
# #         elif edge.is_batch():
# #             if edge.axis1.name in batch_edges:
# #                 batch_edges[edge.axis1.name] += 1
# #             else:
# #                 batch_edges[edge.axis1.name] = 1
# #         else:
# #             raise ValueError('All edges in `edges` should be non-dangling, '
# #                              'shared edges between `node1` and `node2`, or batch edges')
# #
# #     n_shared = len(shared_edges)
# #     n_batch = len(batch_edges)
# #     shared_subscripts = dict(zip(shared_edges,
# #                                  [opt_einsum.get_symbol(i) for i in range(n_shared)]))
# #     batch_subscripts = dict(zip(batch_edges,
# #                                 [opt_einsum.get_symbol(i)
# #                                  for i in range(n_shared, n_shared + n_batch)]))
# #
# #     index = n_shared + n_batch
# #     input_strings = []
# #     used_nodes = []
# #     output_string = ''
# #     matrices = []
# #     matrices_strings = []
# #     for i, node in enumerate([node1, node2]):
# #         if (i == 1) and (node1 == node2):
# #             break
# #         string = ''
# #         for edge in node.edges:
# #             if edge in shared_edges:
# #                 string += shared_subscripts[edge]
# #                 if isinstance(edge, ParamEdge):
# #                     in_matrices = False
# #                     for mat in matrices:
# #                         if torch.equal(edge.matrix, mat):
# #                             in_matrices = True
# #                             break
# #                     if not in_matrices:
# #                         matrices_strings.append(2 * shared_subscripts[edge])
# #                         matrices.append(edge.matrix)
# #             elif edge.is_batch():
# #                 if batch_edges[edge.axis1.name] == 2:
# #                     # Only perform batch contraction if the batch edge appears
# #                     # with the same name in both nodes
# #                     string += batch_subscripts[edge.axis1.name]
# #                     if i == 0:
# #                         output_string += batch_subscripts[edge.axis1.name]
# #                 else:
# #                     string += opt_einsum.get_symbol(index)
# #                     output_string += opt_einsum.get_symbol(index)
# #                     index += 1
# #             else:
# #                 string += opt_einsum.get_symbol(index)
# #                 output_string += opt_einsum.get_symbol(index)
# #                 index += 1
# #         input_strings.append(string)
# #         used_nodes.append(node)
# #
# #     input_string = ','.join(input_strings + matrices_strings)
# #     einsum_string = input_string + '->' + output_string
# #     tensors = list(map(lambda n: n.tensor, used_nodes))
# #     names = '_'.join(map(lambda n: n.name, used_nodes))
# #     new_tensor = opt_einsum.contract(einsum_string, *(tensors + matrices))
# #     new_name = f'contract_{names}'
# #
# #     axes_names = []
# #     edges = []
# #     node1_list = []
# #     i, j, k = 0, 0, 0
# #     while (i < len(output_string)) and \
# #             (j < len(input_strings)):
# #         if output_string[i] == input_strings[j][k]:
# #             axes_names.append(used_nodes[j].axes[k].name)
# #             edges.append(used_nodes[j][k])
# #             node1_list.append(used_nodes[j].axes[k].is_node1())
# #             i += 1
# #         k += 1
# #         if k == len(input_strings[j]):
# #             k = 0
# #             j += 1
# #
# #     # If nodes were connected, we can assume that both are in the same network
# #     if operation is None:
# #         operation = f'contract_edge_{edges}'
# #     new_node = Node(axes_names=axes_names, name=new_name, network=used_nodes[0].network, param_edges=False,
# #                     tensor=new_tensor, edges=edges, node1_list=node1_list, parents={node1, node2}, operation=operation,
# #                     leaf=False)
# #     return new_node
#
#
# def contract(edge: AbstractEdge) -> Node:
#     """
#     Contract only one edge
#     """
#     return contract_edges([edge] + get_batch_edges(edge.node1) + get_batch_edges(edge.node2), edge.node1, edge.node2)
#
#
# def contract_between(node1: AbstractNode, node2: AbstractNode) -> Node:
#     """
#     Contract all shared edges between two nodes, also performing batch contraction
#     between batch edges that share name in both nodes
#     """
#     edges = get_shared_edges(node1, node2) #+ get_batch_edges(node1) + get_batch_edges(node2)
#     if not edges:
#         raise ValueError(f'No batch edges neither shared edges between '
#                          f'nodes {node1!s} and {node2!s} found')
#     return contract_edges(edges, node1, node2)

# class mod_user:
#
#     def __init__(self):
#         global MODE
#         self._old_mode = MODE
#         MODE = "user"
#
#     def __enter__(self):
#         pass
#
#     def __exit__(self, *args, **kws):
#         global MODE
#         MODE = self._old_mode
#
# MODE = "sudo"
#
# with mod_user():
#     print MODE  # print : user.
#
# print MODE  # print: sudo.
#
# mod_user()
# print MODE   # print: user.

# def _func1(data):
#     print('Computing func1')
#
#
# def _func2(data):
#     print('Computing func2')
#
#
# foo = Foo(_func1, _func2)

# Para leer las funciones que se ejecutan en el forward
# https://stackoverflow.com/questions/51901676/get-the-lists-of-functions-used-called-within-a-function-in-python
# https://stackoverflow.com/questions/3061/calling-a-function-of-a-module-by-using-its-name-a-string

# https://pytorch.org/docs/stable/generated/torch.index_select.html

# Cross imports
# https://stackoverflow.com/questions/17226016/simple-cross-import-in-python
