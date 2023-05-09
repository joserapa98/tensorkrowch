"""
This script contains:

    Classes for Nodes and Edges:
        * Axis
        * AbstractNode:
            + Node:
                - StackNode
            + ParamNode:
                - ParamStackNode
        * Edge:
            + StackEdge
            
    Edge operations:
        * connect
        * connect_stack
        * disconnect
    
    Class for successors:        
        * Successor

    Class for Tensor Networks:
        * TensorNetwork
"""

from abc import abstractmethod, ABC
import copy
from typing import (overload,
                    Any, Dict, List, Optional,
                    Sequence, Text, Tuple, Union)
import warnings

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.nn import Parameter

from tensorkrowch.utils import (check_name_style, enum_repeated_names, erase_enum,
                                print_list, stack_unequal_tensors, tab_string)


###############################################################################
#                                     AXIS                                    #
###############################################################################
class Axis:
    """
    The axes are the objects that stick edges to nodes. Every :class:`node
    <AbstractNode>` has a list of :math:`N` axes, each corresponding to one
    edge; and every axis stores information that helps accessing that edge,
    such as its :attr:`name` and :attr:`num` (index). Also, the axis keeps
    track of the :meth:`batch <is_batch>` and :meth:`node1 <is_node1>`
    attributes:

    * **batch**: If axis name containes the word "`batch`", the edge attached
      to this axis will be a batch edge, that is, that edge will not be able to
      be connected to other nodes, but rather specify a dimension with which we
      can perform batch operations (e.g. batch contraction). If the name of the
      axis is changed and no longer contains the word "`batch`", the corresponding
      edge will not be a batch edge any more. Also, :class:`StackNode` and
      :class:`ParamStackNode` instances always have an axis with name "`stack`"
      whose edge is a batch edge.

    * **node1**: When two dangling edges are connected the result is a new
      edge linking two nodes, say ``nodeA`` and ``nodeB``. If the
      connection is performed in the following order::

        new_edge = nodeA[edgeA] ^ nodeB[edgeB]

      Then ``nodeA`` will be the `node1` of ``new_edge`` and ``nodeB``, the `node2`.
      Hence, to access one of the nodes from ``new_edge`` one needs to know if
      it is `node1` or `node2`.

    Even though we can create Axis instances, that will not be usually the case,
    since axes are automatically created when instantiating a new :class:`node
    <AbstractNode>`.

    Parameters
    ----------
    num : int
        Index in the node's axes list.
    name : str
        Axis name, should not contain blank spaces or special characters since
        it is intended to be used as name of submodules.
    node : AbstractNode, optional
        Node to which the axis belongs
    node1 : bool
        Boolean indicating whether `node1` of the edge attached to this axis is
        the node that contains the axis (``True``). Otherwise, the node is `node2`
        of the edge (``False``).

    Examples
    --------
    Although Axis will not be usually explicitly instantiated, it can be done
    like so:

    >>> axis = tk.Axis(0, 'left')
    >>> axis
    Axis( left (0) )

    >>> axis.is_node1()
    True

    >>> axis.is_batch()
    False

    Since "`batch`" is not contained in "`left`", ``axis`` does not correspond
    to a batch edge, but that can be changed:

    >>> axis.name = 'mybatch'
    >>> axis.is_batch()
    True

    Also, as explained before, knowing if a node is the `node1` or `node2` of an
    edge enables users to access that node from the edge:

    >>> nodeA = tk.Node(shape=(2, 3), axes_names=['left', 'right'])
    >>> nodeB = tk.Node(shape=(3, 4), axes_names=['left', 'right'])
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    >>> nodeA == new_edge.nodes[1 - nodeA.get_axis('right').is_node1()]
    True

    >>> nodeB == new_edge.nodes[nodeA.get_axis('right').is_node1()]
    True
    """

    def __init__(self,
                 num: int,
                 name: Text,
                 node: Optional['AbstractNode'] = None,
                 node1: bool = True) -> None:

        # Check types
        if not isinstance(num, int):
            raise TypeError('`num` should be int type')

        if not isinstance(name, str):
            raise TypeError('`name` should be str type')

        if node is not None:
            if not isinstance(node, AbstractNode):
                raise TypeError('`node` should be AbstractNode type')

        if not isinstance(node1, bool):
            raise TypeError('`node1` should be bool type')

        # Check name
        if 'stack' in name:
            if not isinstance(node, (StackNode, ParamStackNode)):
                raise ValueError('Axis cannot be named `stack` if the node is '
                                 'not a StackNode or ParamStackNode')
            if num != 0:
                raise ValueError('Axis `stack` in node should have index 0')

        # Set attributes
        self._num = num
        self._name = name
        self._node = node
        self._node1 = node1
        if ('batch' in name) or ('stack' in name):
            self._batch = True
        else:
            self._batch = False

    # ----------
    # Properties
    # ----------
    @property
    def num(self) -> int:
        """Index in the node's axes list."""
        return self._num

    @property
    def name(self) -> Text:
        """
        Axis name, used to access edges by name of the axis. It cannot contain
        blank spaces or special characters.
        """
        return self._name

    @name.setter
    def name(self, name: Text) -> None:
        """
        Set axis name. The name should not contain blank spaces or special
        characters since it is intended to be used as name of submodule.
        """
        if not isinstance(name, str):
            raise TypeError('`name` should be str type')

        if self._name == 'stack':
            raise ValueError('Name "stack" of stack edge cannot be changed')
        if 'stack' in name:
            raise ValueError(
                'Name "stack" is reserved for stack edges of (Param)StackNodes')

        if self._batch and not ('batch' in name or 'stack' in name):
            self._batch = False
        elif not self._batch and ('batch' in name or 'stack' in name):
            self._batch = True

        if self._node is not None:
            self._node._change_axis_name(self, name)
        else:
            self._name = name

    @property
    def node(self) -> 'AbstractNode':
        """Node to which the axis belongs."""
        return self._node

    # -------
    # Methods
    # -------
    def is_node1(self) -> bool:
        """
        Returns boolean indicating whether `node1` of the edge attached to this
        axis is the node that contains the axis. Otherwise, the node is `node2`
        of the edge.
        """
        return self._node1

    def is_batch(self) -> bool:
        """
        Returns boolean indicating whether the edge in this axis is used as a
        batch edge.
        """
        return self._batch

    def __int__(self) -> int:
        return self._num

    def __str__(self) -> Text:
        return self._name

    def __repr__(self) -> Text:
        return f'{self.__class__.__name__}( {self._name} ({self._num}) )'


###############################################################################
#                                    NODES                                    #
###############################################################################
Ax = Union[int, Text, Axis]
Shape = Union[Sequence[int], Size]


class AbstractNode(ABC):
    """
    Abstract class for all types of nodes. Defines what a node is and most of its
    properties and methods. Since it is an abstract class, cannot be instantiated.

    A node is the minimum element in a :class:`TensorNetwork`. At its most basic
    level, it is just a container for a tensor that stores information about its
    neighbours (with what other nodes it is connected), edges (names to access
    each of them, whether they are batch edges or not, etc.) or successors (nodes
    that result from operating the node). Besides, and what is more important,
    this information is useful to:

    * Perform tensor network :class:`Operations <Operation>` such as :func:`contraction
      <contract_between>` of two neighbouring nodes without having to worry about
      tensor's shapes, order of axes, etc.

    * Perform more advanced operations such as :func:`stack` or :func:`unbind`
      saving memory and time.

    * Keep track of operations in which a node has taken place, so that several
      steps can be skipped in further training iterations. See :meth:`TensorNetwork.trace`.

    Refer to the subclasses of ``AbstractNode`` to see how to instantiate nodes:

    * :class:`Node`

    * :class:`ParamNode`

    * :class:`StackNode`

    * :class:`ParamStackNode`
    """

    def __init__(self,
                 shape: Optional[Shape] = None,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 data: bool = False,
                 virtual: bool = False,
                 override_node: bool = False,
                 tensor: Optional[Tensor] = None,
                 edges: Optional[List['Edge']] = None,
                 override_edges: bool = False,
                 node1_list: Optional[List[bool]] = None,
                 init_method: Optional[Text] = None,
                 device: Optional[torch.device] = None,
                 **kwargs: float) -> None:

        super().__init__()

        # Check shape and tensor.shape
        if (shape is None) == (tensor is None):
            if shape is None:
                raise ValueError('One of `shape` or `tensor` must be provided')
            elif tensor.shape == shape:
                shape = None
            else:
                raise ValueError('If both `shape` and `tensor` are given, '
                                 '`tensor`\'s shape should be equal to `shape`')

        # Check shape type
        if shape is not None:
            if not isinstance(shape, (tuple, list, Size)):
                raise TypeError(
                    '`shape` should be tuple[int], list[int] or torch.Size type')
            if isinstance(shape, (tuple, list)):
                for i in shape:
                    if not isinstance(i, int):
                        raise TypeError('`shape` elements should be int type')
            aux_shape = Size(shape)
        else:
            aux_shape = tensor.shape
            
        # Check tensor type
        if tensor is not None:
            if not isinstance(tensor, Tensor):
                raise TypeError('`tensor` should be torch.Tensor type')

        # Check axes_names
        if axes_names is None:
            axes = [Axis(num=i, name=f'axis_{i}', node=self)
                    for i, _ in enumerate(aux_shape)]
        else:
            if not isinstance(axes_names, (tuple, list)):
                raise TypeError(
                    '`axes_names` should be tuple[str, ...] or list[str, ...] type')
            if len(axes_names) != len(aux_shape):
                raise ValueError(
                    '`axes_names` length should match `shape` length')
            else:
                axes_names = enum_repeated_names(axes_names)
                axes = [Axis(num=i, name=name, node=self)
                        for i, name in enumerate(axes_names)]

        # Check name
        if name is None:
            name = self.__class__.__name__.lower()
        elif not isinstance(name, str):
            raise TypeError('`name` should be str type')
        elif not check_name_style(name, 'node'):
            raise ValueError('Names cannot contain blank spaces')

        # Check network
        if network is not None:
            if not isinstance(network, TensorNetwork):
                raise TypeError('`network` should be TensorNetwork type')
        else:
            network = TensorNetwork()

        # Set attributes
        self._tensor_info = None
        self._temp_tensor = None
        self._shape = aux_shape
        self._axes = axes
        self._edges = []
        self._name = name
        self._network = network
        self._successors = dict()

        # Set node type
        if not hasattr(self, '_leaf'):
            self._leaf = not (data or virtual)
            # else, it is False (check _create_resultant)
        self._data = data
        self._virtual = virtual

        if (self._leaf + self._data + self._virtual) > 1:
            raise ValueError('The node can only be one of `leaf`, `data`, `virtual`'
                             ' and `resultant`')

        # Add edges
        if edges is None:
            self._edges = [self._make_edge(ax) for ax in self._axes]
        else:
            if node1_list is None:
                raise ValueError(
                    'If `edges` are provided, `node1_list` should also be provided')
            for i, axis in enumerate(self._axes):
                if not isinstance(node1_list[i], bool):
                    raise TypeError('`node1_list` should be list[bool] type')
                axis._node1 = node1_list[i]
            self._edges = edges[:]
            # If node stores its own tensor
            if not self.is_resultant():
                self.reattach_edges(override=override_edges)

        # Add to network
        self._network._add_node(self, override=override_node)

        # Remove from the network edges from virtual nodes
        if self._virtual:
            for edge in self._edges:
                self._network._remove_edge(edge)

        # Set tensor
        if shape is not None:
            if init_method is not None:
                self._unrestricted_set_tensor(
                    init_method=init_method, device=device, **kwargs)
        else:
            self._unrestricted_set_tensor(tensor=tensor)

    @classmethod
    def _create_resultant(cls, *args, **kwargs) -> None:
        obj = super().__new__(cls, *args, **kwargs)

        # Only way to set _leaf to False
        obj._leaf = False
        cls.__init__(obj, *args, **kwargs)

        return obj

    # ----------
    # Properties
    # ----------
    @property
    def tensor(self) -> Optional[Union[Tensor, Parameter]]:
        """
        Node's tensor. It can be a ``torch.Tensor``, ``torch.nn.Parameter`` or
        ``None`` if the node is empty.
        """
        if (self._temp_tensor is not None) or (self._tensor_info is None):
            result = self._temp_tensor
            return result

        address = self._tensor_info['address']
        node_ref = self._tensor_info['node_ref']
        full = self._tensor_info['full']
        index = self._tensor_info['index']

        if address is None:
            address = node_ref._tensor_info['address']
        result = self._network._memory_nodes[address]

        return_result = full or (result is None)
        if self._network._unbind_mode:
            return_result = return_result or self._name.startswith('unbind')

        if return_result:
            return result
        return result[index]

    @tensor.setter
    def tensor(self, tensor: torch.Tensor) -> None:
        if tensor is None:
            self.unset_tensor()
        else:
            self.set_tensor(tensor)

    @property
    def shape(self) -> Size:
        """Shape of node's :attr:`tensor`. It is a ``torch.Size``."""
        return self._shape

    @property
    def rank(self) -> int:
        """Length of node's :attr:`shape`, that is, number of edges of the node."""
        return len(self._shape)

    @property
    def dtype(self):
        """``torch.dtype`` of node's :attr:`tensor`."""
        tensor = self.tensor
        if tensor is None:
            return
        return tensor.dtype

    @property
    def axes(self) -> List[Axis]:
        """List of nodes's :class:`axes <Axis>`."""
        return self._axes

    @property
    def axes_names(self) -> List[Text]:
        """List of names of node's axes."""
        return list(map(lambda axis: axis._name, self._axes))

    @property
    def edges(self) -> List['Edge']:
        """List of node's :class:`edges <Edge>`."""
        return self._edges

    @property
    def network(self) -> 'TensorNetwork':
        """
        :class:`TensorNetwork` where the node belongs. If the node is moved to
        another :class:`TensorNetwork`, the entire connected component of the
        graph where the node is will be moved.
        """
        return self._network

    @network.setter
    def network(self, network: 'TensorNetwork') -> None:
        self.move_to_network(network)

    @property
    def successors(self) -> Dict[Text, 'Successor']:
        """
        Dictionary with :class:`Operations <Operation>`' names as keys, and the
        list of :class:`Successors <Successor>` of the node as values.
        """
        return self._successors

    @property
    def name(self) -> Text:
        """
        Node's name, used to access the node from the :attr:`tensor network <network>`
        where it belongs. It cannot contain blank spaces.
        """
        return self._name

    @name.setter
    def name(self, name: Text) -> None:
        if not isinstance(name, str):
            raise TypeError('`name` should be str type')
        elif not check_name_style(name, 'node'):
            raise ValueError('`name` cannot contain blank spaces')
        self._network._change_node_name(self, name)

    # ----------------
    # Abstract methods
    # ----------------
    @abstractmethod
    def _make_edge(self, axis: Axis) -> 'Edge':
        pass

    @staticmethod
    @abstractmethod
    def _set_tensor_format(tensor: Tensor) -> Union[Tensor, Parameter]:
        pass

    @abstractmethod
    def parameterize(self, set_param: bool) -> 'AbstractNode':
        pass

    @abstractmethod
    def copy(self, share_tensor: bool = False) -> 'AbstractNode':
        pass

    # -------
    # Methods
    # -------
    def is_leaf(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``leaf`` node. These are
        the nodes that form the :class:`TensorNetwork`. Usually, these will be
        the `trainable` nodes. These nodes can hold their own tensors or use
        other node's tensor.
        """
        return self._leaf

    def is_data(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``data`` node. These are
        the nodes where input data tensors will be put. Essentially these are
        also leaf nodes, but they only store temporary data tensors that will
        be replaced in each training epoch.
        """
        return self._data

    def is_virtual(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``virtual`` node. These
        are a sort of `hidden` nodes that can be used, for instance, to store
        the information of other ``leaf`` or ``data`` nodes more efficiently
        (e.g. :class:`Uniform MPS <UMPS>` uses a unique ``virtual`` node to
        store the tensor used by all the nodes in the network).

        There is a special case of virtual nodes one can create: the ones
        used as memory for uniform (traslationally invariant) tensor networks.
        In this case, it is recommended to use the string "virtual_uniform" in
        the node's name (e.g. "virtual_uniform_mps").
        """
        return self._virtual

    def is_resultant(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``resultant`` node. These
        are the nodes that result from an operation on any type of nodes.
        """
        return not (self._leaf or self._data or self._virtual)

    def size(self, axis: Optional[Ax] = None) -> Union[Size, int]:
        """
        Returns the size of the node's tensor. If ``axis`` is specified, returns
        the size of that axis; otherwise returns the shape of the node (same as
        :attr:`shape`).

        Parameters
        ----------
        axis : int, str or Axis, optional
            Axis for which to retrieve the size.

        Returns
        -------
        int or torch.Size
        """
        if axis is None:
            return self._shape
        axis_num = self.get_axis_num(axis)
        return self._shape[axis_num]

    def is_node1(self, axis: Optional[Ax] = None) -> Union[bool, List[bool]]:
        """
        Returns :meth:`node <Axis.is_node1>` attribute of axes of the node. If
        ``axis`` is specified, returns only the ``node`` of that axis; otherwise
        returns the ``node1`` of all axes of the node.

        Parameters
        ----------
        axis : int, str or Axis, optional
            Axis for which to retrieve the ``node1``.

        Returns
        -------
        bool or list[bool]
        """
        if axis is None:
            return list(map(lambda ax: ax._node1, self._axes))
        axis_num = self.get_axis_num(axis)
        return self._axes[axis_num]._node1

    def neighbours(self, axis: Optional[Ax] = None) -> Union[Optional['AbstractNode'],
                                                             List['AbstractNode']]:
        """
        Returns the neighbours of the node, the nodes to which it is connected.

        Parameters
        ----------
        axis : int, str or Axis, optional
            Axis for which to retrieve the neighbour.

        Returns
        -------
        AbstractNode or list[AbstractNode]
        """
        node1_list = self.is_node1()
        if axis is not None:
            node2 = self[axis]._nodes[node1_list[self.get_axis_num(axis)]]
            return node2

        neighbours = set()
        for i, edge in enumerate(self._edges):
            if not edge.is_dangling():
                node2 = edge._nodes[node1_list[i]]
                neighbours.add(node2)
        return list(neighbours)

    def _change_axis_name(self, axis: Axis, name: Text) -> None:
        """
        Changes the name of an axis. If an axis belongs to a node, we have to
        take care of repeated names. If the name that is going to be assigned
        to the axis is already set for another axis, we change  those names by
        an enumerated version of them.

        Parameters
        ----------
        axis : Axis
            Axis whose name is going to be changed.
        name : str
            New name.

        Raises
        ------
        ValueError
            If ``axis`` does not belong to the node.
        """
        if axis._node != self:
            raise ValueError('Cannot change the name of an axis that does '
                             'not belong to the node')
        if name != axis._name:
            axes_names = self.axes_names[:]
            for i, axis_name in enumerate(axes_names):
                if axis_name == axis._name:  # Axes names are unique
                    axes_names[i] = name
                    break
            new_axes_names = enum_repeated_names(axes_names)
            for axis, axis_name in zip(self._axes, new_axes_names):
                axis._name = axis_name

    def _change_axis_size(self, axis: Ax, size: int) -> None:
        """
        Changes axis size, that is, changes size of node's tensor and corresponding
        edges at a certain axis.

        Parameters
        ----------
        axis : int, str or Axis
            Axis where size is going to be changed.
        size : int
            New size.

        Raises
        ------
        ValueError
            If new size is not positive.
        """
        if size <= 0:
            raise ValueError('New `size` should be greater than zero')
        axis_num = self.get_axis_num(axis)

        tensor = self.tensor
        if tensor is None:
            aux_shape = list(self._shape)
            aux_shape[axis_num] = size
            self._shape = Size(aux_shape)

        else:
            if size < self._shape[axis_num]:
                # If new size is smaller than current, tensor is cropped
                # starting from the "left", "top", "front", etc. in each dimension
                index = []
                for i, dim in enumerate(self._shape):
                    if i == axis_num:
                        index.append(slice(dim - size, dim))
                    else:
                        index.append(slice(0, dim))
                aux_shape = list(self._shape)
                aux_shape[axis_num] = size
                self._shape = Size(aux_shape)
                self._direct_set_tensor(tensor[index])

            elif size > self._shape[axis_num]:
                # If new size is greater than current, tensor is expanded with
                # zeros in the "left", "top", "front", etc. dimension
                pad = []
                for i, dim in enumerate(self._shape):
                    if i == axis_num:
                        pad += [0, size - dim]
                    else:
                        pad += [0, 0]
                pad.reverse()
                aux_shape = list(self._shape)
                aux_shape[axis_num] = size
                self._shape = Size(aux_shape)
                self._direct_set_tensor(nn.functional.pad(tensor, pad))

    def get_axis(self, axis: Ax) -> 'Edge':
        """Returns :class:`Axis` given its ``name`` or ``num``."""
        axis_num = self.get_axis_num(axis)
        return self._axes[axis_num]

    def get_axis_num(self, axis: Ax) -> int:
        """Returns axis' ``num`` given the :class:`Axis` or its ``name``."""
        if isinstance(axis, int):
            if axis < 0:
                axis = axis % self.rank  # When indexing with -1, -2, ...
            for ax in self._axes:
                if axis == ax._num:
                    return ax._num
            raise IndexError(f'Node {self!s} has no axis with index {axis}')
        elif isinstance(axis, str):
            for ax in self._axes:
                if axis == ax._name:
                    return ax._num
            raise IndexError(f'Node {self!s} has no axis with name {axis}')
        elif isinstance(axis, Axis):
            for ax in self._axes:
                if axis == ax:
                    return ax._num
            raise IndexError(f'Node {self!s} has no axis {axis!r}')
        else:
            raise TypeError('`axis` should be int, str or Axis type')

    def _add_edge(self,
                  edge: 'Edge',
                  axis: Ax,
                  node1: bool = True) -> None:
        """
        Adds an edge to the specified axis of the node.

        Parameters
        ----------
        edge : Edge
            Edge that will be added.
        axis : int, str or Axis
            Axes where the edge will be attached.
        node1 : bool, optional
            Boolean indicating whether the node is the `node1` (``True``) or
            `node2` (``False``) of the edge.
        """
        axis_num = self.get_axis_num(axis)
        self._axes[axis_num]._node1 = node1
        self._edges[axis_num] = edge

    def get_edge(self, axis: Ax) -> 'Edge':
        """
        Returns :class:`Edge` given the :class:`Axis` (or its ``name``
        or ``num``) where it is attached to the node.
        """
        axis_num = self.get_axis_num(axis)
        return self._edges[axis_num]

    def in_which_axis(self, edge: 'Edge') -> Axis:
        """
        Returns :class:`Axis` given the :class:`Edge` that is attached
        to the node through it.
        """
        lst = []
        for ax, ed in zip(self._axes, self._edges):
            if ed == edge:
                lst.append(ax)

        if len(lst) == 0:
            raise ValueError(f'Edge {edge} not in node {self}')
        elif len(lst) == 1:
            return lst[0]
        else:
            # Case of trace edges (attached to the node in two axes)
            return lst

    def reattach_edges(self, override: bool = False) -> None:
        """
        Substitutes current edges by copies of them that are attached to the node.
        It can happen that an edge is not attached to the node if it is the result
        of an :class:`Operation` and, hence, it inherits edges from the operands.
        In that case, the new copied edges will be attached to the resultant node,
        replacing each previous `node1` or `node2` with it (according to the
        ``node1`` attribute of each axis).

        Used for in-place operations like :func:`permute_` or :func:`split_` and
        to (de)parameterize nodes.

        Parameters
        ----------
        override: bool
            Boolean indicating if the new, reattached edges should also replace
            the corresponding edges in the node's neighbours (``True``). Otherwise,
            the neighbours' edges will be pointing to the original nodes from which
            the current node inherits its edges (``False``).
        """
        for i, (edge, node1) in enumerate(zip(self._edges, self.is_node1())):
            node = edge._nodes[1 - node1]
            if node != self:
                # New edges are always a copy, so that the original
                # nodes have different edges from the current node
                new_edge = edge.copy()
                self._edges[i] = new_edge

                new_edge._nodes[1 - node1] = self
                new_edge._axes[1 - node1] = self._axes[i]

                # Case of trace edges (attached to the node in two axes)
                neighbour = new_edge._nodes[node1]
                if neighbour == node:
                    for j, other_edge in enumerate(self._edges):
                        if (other_edge == edge) and (i != j):
                            self._edges[j] = new_edge
                            new_edge._nodes[node1] = self
                            new_edge._axes[node1] = self._axes[j]

                if override:
                    if not new_edge.is_dangling() and (neighbour != node):
                        neighbour._add_edge(
                            new_edge, new_edge._axes[node1], not node1)

    def disconnect(self, axis: Optional[Ax] = None) -> None:
        """
        Disconnects all edges of the node if they were connected to other nodes.
        If ``axis`` is sepcified, only the corresponding edge is disconnected.

        Parameters
        ----------
        axis : int, str or Axis, optional
            Axis whose edge will be disconnected.
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
    def _make_copy_tensor(shape: Shape,
                          device: torch.device = torch.device('cpu')) -> Tensor:
        """Returns copy tensor (ones in the "diagonal", zeros elsewhere)."""
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
        """Returns tensor whose entries are drawn from the uniform distribution."""
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
        """Returns tensor whose entries are drawn from the normal distribution."""
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
        """
        Returns a tensor that can be put in the node, and is initialized according
        to ``init_method``. By default, it has the same shape as the node.

        Parameters
        ----------
        shape : list[int], tuple[int] or torch.Size, optional
            Shape of the tensor. If None, node's shape will be used.
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensor.
        kwargs : float
            Keyword arguments for the different initialization methods:

            * ``low``, ``high`` for uniform initialization. See
              `torch.rand() <https://pytorch.org/docs/stable/generated/torch.rand.html>`_

            * ``mean``, ``std`` for normal initialization. See
              `torch.randn() <https://pytorch.org/docs/stable/generated/torch.randn.html>`_

        Returns
        -------
        torch.Tensor

        Raises
        ------
        ValueError
            If ``init_method`` is not one of "zeros", "ones", "copy", "rand", "randn".
        """
        if shape is None:
            shape = self._shape
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

    def _compatible_shape(self, tensor: Tensor) -> bool:
        """
        Checks if tensor's shape is "compatible" with the node's shape, meaning
        that the sizes in all axes must match except for the batch axes, where
        sizes can be different.
        """
        if len(tensor.shape) == self.rank:
            for i, dim in enumerate(tensor.shape):
                edge = self.get_edge(i)
                if not edge.is_batch() and (dim != edge.size()):
                    return False
            return True
        return False

    def _crop_tensor(self, tensor: Tensor) -> Tensor:
        """
        Crops the tensor in case its shape is not compatible with the node's shape.
        That is, if the tensor has a size that is smaller than the corresponding
        size of the node for a certain axis, the tensor is cropped in that axis
        (provided that the axis is not a batch axis). If that size is greater in
        the tensor that in the node, raises a ``ValueError``.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to be cropped.

        Returns
        -------
        torch.Tensor
        """
        if len(tensor.shape) == self.rank:
            index = []
            for i, dim in enumerate(tensor.shape):
                edge = self.get_edge(i)

                if edge.is_batch() or (dim == edge.size()):
                    index.append(slice(0, dim))
                elif dim > edge.size():
                    index.append(slice(dim - edge.size(), dim))
                else:
                    raise ValueError(f'Cannot crop tensor if its size at axis {i}'
                                     ' is smaller than node\'s size')
            return tensor[index]

        else:
            raise ValueError('`tensor` should have the same number of'
                             ' dimensions as node\'s tensor (same rank)')
            
    def _direct_set_tensor(self,
                        tensor: Optional[Tensor],
                        check_shape: bool = False) -> None:
        if check_shape and not self._compatible_shape(tensor):
            tensor = self._crop_tensor(tensor)
        correct_format_tensor = self._set_tensor_format(tensor)

        self._save_in_network(correct_format_tensor)
        self._shape = tensor.shape

    def _unrestricted_set_tensor(self,
                                 tensor: Optional[Tensor] = None,
                                 init_method: Optional[Text] = 'zeros',
                                 device: Optional[torch.device] = None,
                                 **kwargs: float) -> None:
        """
        Sets a new node's tensor or creates one with :meth:`make_tensor` and sets
        it. Before setting it, it is casted to the correct type, so that a
        ``torch.Tensor`` can be turned into a ``nn.Parameter`` when setting it
        in :class:`ParamNodes <ParamNode`. This can be used in any node, even in
        ``resultant`` nodes.

        Parameters
        ----------
        tensor : torch.Tensor, optional
            Tensor to be set in the node. If None, and `init_method` is provided,
            the tensor is created with :meth:`make_tensor`. Otherwise, a None is
            set as node's tensor.
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensor.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`make_tensor`.
        """
        if tensor is not None:
            if not isinstance(tensor, Tensor):
                raise TypeError('`tensor` should be torch.Tensor type')
            elif device is not None:
                warnings.warn('`device` was specified but is being ignored. Provide '
                              'a tensor that is already in the required device')

            if not self._compatible_shape(tensor):
                tensor = self._crop_tensor(tensor)
            correct_format_tensor = self._set_tensor_format(tensor)

        elif init_method is not None:
            node_tensor = self.tensor
            if (device is None) and (node_tensor is not None):
                device = node_tensor.device
            tensor = self.make_tensor(
                init_method=init_method, device=device, **kwargs)
            correct_format_tensor = self._set_tensor_format(tensor)

        else:
            correct_format_tensor = None

        self._save_in_network(correct_format_tensor)
        self._shape = tensor.shape

    def set_tensor(self,
                   tensor: Optional[Tensor] = None,
                   init_method: Optional[Text] = 'zeros',
                   device: Optional[torch.device] = None,
                   **kwargs: float) -> None:
        """
        Sets a new node's tensor or creates one with :meth:`make_tensor` and sets
        it. Before setting it, it is casted to the correct type, so that a
        ``torch.Tensor`` can be turned into a ``nn.Parameter`` when setting it
        in :class:`ParamNodes <ParamNode>`.

        This way of setting tensors is only applicable to ``leaf`` nodes. For
        ``resultant`` nodes, their tensors come from the result of operations on
        ``leaf`` tensors; hence they should not be modified. For ``data`` nodes,
        tensors are set into nodes when calling the :meth:`TensorNetwork.forward`
        method of :class:`tensor networks <TensorNetwork>` with a data tensor or
        a sequence of tensors.

        Besides, this can only be used if the :class:`TensorNetwork` is not in
        :attr:`~TensorNetwork.automemory` mode.

        Parameters
        ----------
        tensor : torch.Tensor, optional
            Tensor to be set in the node. If None, and `init_method` is provided,
            the tensor is created with :meth:`make_tensor`. Otherwise, a ``None``
            is set as node's tensor.
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensor.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`make_tensor`.

        Raises
        ------
        ValueError
            If the node is not a ``leaf`` node or the tensor network is in
            ``automemory`` mode.
        """
        # If node stores its own tensor
        if not self.is_resultant() and (self._tensor_info['address'] is not None):
            self._unrestricted_set_tensor(tensor=tensor,
                                          init_method=init_method,
                                          device=device,
                                          **kwargs)
        else:
            raise ValueError('Node\'s tensor can only be changed if it is not'
                             'resultant and if it stores its own tensor')
            
    def set_tensor_from(self, other: 'AbstractNode') -> None:
        """Sets node's tensor as the tensor used by other node."""
        del self._network._memory_nodes[self._tensor_info['address']]
        
        if other._tensor_info['address'] is not None:
            self._tensor_info['address'] = None
            self._tensor_info['node_ref'] = other
            self._tensor_info['full'] = True
            self._tensor_info['index'] = None
        else:
            self._tensor_info = other._tensor_info

    def unset_tensor(self) -> None:
        """Replaces node's tensor with None."""
        # If node stores its own tensor
        if not self.is_resultant() and (self._tensor_info['address'] is not None):
            self._save_in_network(None)

    def _save_in_network(self, tensor: Union[Tensor, Parameter]) -> None:
        """Saves new node's tensor in the network's memory."""
        self._network._memory_nodes[self._tensor_info['address']] = tensor
        if isinstance(tensor, Parameter):
            self._network.register_parameter(
                'param_' + self._tensor_info['address'], tensor)

    def _record_in_inverse_memory(self):
        """
        Records information of the node in network's ``inverse memory``. This
        memory is a dictionary that, for each node used in an :class:`Operation`,
        keeps track of:

        * The total amount of times that the node's tensor is accessed to compute
          operations (calculated when contracting the network for the first time,
          in ``tracing`` mode).

        * The number of accesses to the node's tensor in the current contraction.

        * Whether this node's tensor can be erased after using it for all the
          operations in which it is involved.

        When contracting the :class:`TensorNetwork`, if the node's tensor has been
        accessed the total amount of times it has to be accessed, and it can be
        erased, then its tensor is indeed replaced by None.
        """
        net = self._network
        address = self._tensor_info['address']
        if address is None:
            node_ref = self._tensor_info['node_ref']
            address = node_ref._tensor_info['address']
            check_nodes = [self, node_ref]
        else:
            check_nodes = [self]

        # When tracing network, node is recorded in inverse memory
        if net._tracing:
            if address in net._inverse_memory:
                if net._inverse_memory[address]['erase']:
                    net._inverse_memory[address]['accessed'] += 1
            else:
                # Node can only be erased if both itself and the node from which
                # it is taking the tensor information (node_ref) are resultant or
                # data nodes (including virtual node that stores stack data tensor)
                erase = True
                for node in check_nodes:
                    erase &= node.is_resultant() or node._data or \
                        (node._virtual and node._name == 'stack_data_memory')

                net._inverse_memory[address] = {
                    'accessed': 1,
                    're-accessed': 0,
                    'erase': erase}

        # When contracting network, we keep track of the number of accesses
        # to "erasable" nodes
        else:
            if address in net._inverse_memory:
                net._inverse_memory[address]['re-accessed'] += 1
                aux_dict = net._inverse_memory[address]

                if aux_dict['accessed'] == aux_dict['re-accessed']:
                    if aux_dict['erase']:
                        self._network._memory_nodes[address] = None
                    net._inverse_memory[address]['re-accessed'] = 0
                    
    def tensor_address(self) -> Text:
        """Returns address of the node's tensor in the network's memory."""
        address = self._tensor_info['address']
        if address is None:
            node_ref = self._tensor_info['node_ref']
            address = node_ref._tensor_info['address']
        return address

    def move_to_network(self,
                        network: 'TensorNetwork',
                        visited: Optional[List['AbstractNode']] = None) -> None:
        """
        Moves node to another network. All other nodes connected to it, or
        to a node connected to it, etc. are also moved to the new network.

        Parameters
        ----------
        network : TensorNetwork
            Tensor Network to which the nodes will be moved.
        visited : list[AbstractNode], optional
            List indicating the nodes that have been already moved to the new
            network, used by this DFS-like algorithm.
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
    def __getitem__(self, key: slice) -> List['Edge']:
        pass

    @overload
    def __getitem__(self, key: Ax) -> 'Edge':
        pass

    def __getitem__(self, key: Union[slice, Ax]) -> Union[List['Edge'],
                                                          'Edge']:
        if isinstance(key, slice):
            return self._edges[key]
        return self.get_edge(key)

    # -----------------
    # Tensor operations
    # -----------------
    def sum(self, axis: Optional[Union[Ax, Sequence[Ax]]] = None) -> Tensor:
        """
        Returns the sum of all elements in the node's tensor. If an ``axis`` is
        specified, the sum is over that axis. If ``axis`` is a sequence of axes,
        reduce over all of them.

        This is not a node :class:`Operation`, hence it returns a ``torch.Tensor``
        instead of a :class:`Node`.

        See also `torch.sum() <https://pytorch.org/docs/stable/generated/torch.sum.html>`_.

        Parameters
        ----------
        axis : int, str, Axis or list[int, str or Axis], optional
            Axis or sequence of axes over which to reduce.

        Returns
        -------
        torch.Tensor
        """
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_num(ax))
        return self.tensor.sum(dim=axis_num)

    def mean(self, axis: Optional[Union[Ax, Sequence[Ax]]] = None) -> Tensor:
        """
        Returns the mean of all elements in the node's tensor. If an ``axis`` is
        specified, the mean is over that axis. If ``axis`` is a sequence of axes,
        reduce over all of them.

        This is not a node :class:`Operation`, hence it returns a ``torch.Tensor``
        instead of a :class:`Node`.

        See also `torch.mean() <https://pytorch.org/docs/stable/generated/torch.mean.html>`_.

        Parameters
        ----------
        axis : int, str, Axis or list[int, str or Axis], optional
            Axis or sequence of axes over which to reduce.

        Returns
        -------
        torch.Tensor
        """
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_num(ax))
        return self.tensor.mean(dim=axis_num)

    def std(self, axis: Optional[Union[Ax, Sequence[Ax]]] = None) -> Tensor:
        """
        Returns the std of all elements in the node's tensor. If an ``axis`` is
        specified, the std is over that axis. If ``axis`` is a sequence of axes,
        reduce over all of them.

        This is not a node :class:`Operation`, hence it returns a ``torch.Tensor``
        instead of a :class:`Node`.

        See also `torch.std() <https://pytorch.org/docs/stable/generated/torch.std.html>`_.

        Parameters
        ----------
        axis : int, str, Axis or list[int, str or Axis], optional
            Axis or sequence of axes over which to reduce.

        Returns
        -------
        torch.Tensor
        """
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_num(ax))
        return self.tensor.std(dim=axis_num)

    def norm(self, p=2, axis: Optional[Sequence[Ax]] = None) -> Tensor:
        """
        Returns the norm of all elements in the node's tensor. If an ``axis`` is
        specified, the norm is over that axis. If ``axis`` is a sequence of axes,
        reduce over all of them.

        This is not a node :class:`Operation`, hence it returns a ``torch.Tensor``
        instead of a :class:`Node`.

        See also `torch.norm() <https://pytorch.org/docs/stable/generated/torch.norm.html>`_.

        Parameters
        ----------
        p : int, float
            The order of the norm.
        axis : int, str, Axis or list[int, str or Axis], optional
            Axis or sequence of axes over which to reduce.

        Returns
        -------
        torch.Tensor
        """
        axis_num = []
        if axis is not None:
            for ax in axis:
                axis_num.append(self.get_axis_num(ax))
        return self.tensor.norm(p=p, dim=axis_num)

    def __str__(self) -> Text:
        return self._name

    def __repr__(self) -> Text:
        return f'{self.__class__.__name__}(\n ' \
               f'\tname: {self._name}\n' \
               f'\ttensor:\n{tab_string(repr(self.tensor), 2)}\n' \
               f'\taxes:\n{tab_string(print_list(self.axes_names), 2)}\n' \
               f'\tedges:\n{tab_string(print_list(self._edges), 2)})'


class Node(AbstractNode):
    """
    Base class for non-trainable nodes. Should be subclassed by any class of nodes
    that are not intended to be trained (e.g. :class:`StackNode`).

    Can be used for fixed nodes of the :class:`TensorNetwork`, or intermediate
    nodes that are resultant from an :class:`Operation` between nodes.

    For a complete list of properties and methods, see also :class:`AbstractNode`.

    Parameters
    ----------
    shape : list[int], tuple[int], torch.Size, optional
        Node's shape, that is, the shape of its tensor. If ``shape`` and
        ``init_method`` are provided, a tensor will be made for the node. Otherwise,
        ``tensor`` would be required.
    axes_names : list[str], tuple[str], optional
        Sequence of names for each of the node's axes. Names are used to access
        the edge that is attached to the node in a certain axis. Hence they should
        be all distinct.
    name : str, optional
        Node's name, used to access the node from de :class:`TensorNetwork` where
        it belongs. It cannot contain blank spaces.
    network : TensorNetwork, optional
        Tensor network where the node should belong. If None, a new tensor network,
        will be created to contain the node.
    data : bool
        Boolean indicating if the node is a ``data`` node.
    virtual : bool
        Boolean indicating if the node is a ``virtual`` node.
    override_node : bool
        Boolean indicating whether the node should override (``True``) another
        node in the network that has the same name (e.g. if a node is parameterized,
        it would be required that a new :class:`ParamNode` replaces the non-parameterized
        node in the network).
    tensor : torch.Tensor, optional
        Tensor that is to be stored in the node. If None, ``shape`` and ``init_method``
        will be required.
    edges : list[Edge], optional
        List of edges that are to be attached to the node. This can be used in
        case the node inherits the edges from other node(s), like in :class:`Operations
        <Operation>`.
    override_edges : bool
        Boolean indicating whether the provided ``edges`` should be overriden
        (``True``) when reattached (e.g. if a node is parameterized, it would
        be required that the new :class:`ParamNode`'s edges are indeed connected
        to it, instead of to the original non-parameterized node).
    node1_list : list[bool], optional
        If ``edges`` are provided, the list of ``node1`` attributes of each edge
        should also be provided.
    init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
        Initialization method.
    device : torch.device, optional
        Device where to initialize the tensor.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`AbstractNode.make_tensor`.
    """

    # -------
    # Methods
    # -------
    def _make_edge(self, axis: Axis) -> 'Edge':
        """Makes ``Edges`` that will be attached to each axis."""
        return Edge(node1=self, axis1=axis)

    @staticmethod
    def _set_tensor_format(tensor: Tensor) -> Tensor:
        """Returns a torch.Tensor if input tensor is given as nn.Parameter."""
        if isinstance(tensor, Parameter):
            return tensor.detach()
        return tensor

    def parameterize(self, set_param: bool = True) -> Union['Node', 'ParamNode']:
        """
        Replaces the node with a parameterized version of it, that is, turns a
        fixed :class:`Node` into a trainable :class:`ParamNode`.

        Since the node is `replaced`, it will be completely removed from the network,
        and its neighbours will point to the new parameterized node.

        Parameters
        ----------
        set_param : bool
            Boolean indicating whether the node should be parameterized (``True``).
            Otherwise (``False``), the non-parameterized node itself will be
            returned.

        Returns
        -------
        Node or ParamNode
            The original node or a parameterized version of it.
        """
        if set_param:
            new_node = ParamNode(shape=self.shape,
                                 axes_names=self.axes_names,
                                 name=self._name,
                                 network=self._network,
                                 override_node=True,
                                 tensor=self.tensor,
                                 edges=self._edges,
                                 override_edges=True,
                                 node1_list=self.is_node1())
            return new_node
        else:
            return self

    def copy(self, share_tensor: bool = False) -> 'Node':
        """
        Returns a copy of the node. That is, returns a node whose tensor is a copy
        of the original, whose edges are directly inherited (these are not copies,
        but the exact same edges) and whose name is extended with the prefix
        ``"copy_"``.

        Returns
        -------
        Node
        """
        if share_tensor:
            new_node = Node(shape=self._shape,
                        axes_names=self.axes_names,
                        name=self._name + '_copy',
                        network=self._network,
                        edges=self._edges,
                        node1_list=self.is_node1())
            new_node.set_tensor_from(self)
        else:  
            new_node = Node(shape=self._shape,
                            axes_names=self.axes_names,
                            name=self._name + '_copy',
                            network=self._network,
                            tensor=self.tensor,
                            edges=self._edges,
                            node1_list=self.is_node1())
        return new_node


class ParamNode(AbstractNode):
    """
    Class for trainable nodes. Should be subclassed by any class of nodes that
    are intended to be trained (e.g. :class:`ParamStackNode`).

    Should be used as the initial nodes conforming the :class:`TensorNetwork`,
    if it is going to be trained. When operating these initial nodes, the resultant
    nodes will be non-parameterized (e.g. :class:`Node`, :class:`StackNode`).

    The main difference with :class:`Nodes <Node>` is that ``ParamNodes`` have
    ``nn.Parameter`` tensors instead of ``torch.Tensor``. Therefore, a ``ParamNode``
    is a sort of `parameter` that is attached to the :class:`TensorNetwork` (which
    is itself a ``nn.Module``). That is, the list of parameters of the tensor
    network module contains the tensors of all ``ParamNodes``. 

    To see how to initialize ``ParamNodes``, see :class:`Node`.

    For a complete list of properties and methods, see also :class:`AbstractNode`.
    """

    def __init__(self,
                 shape: Optional[Shape] = None,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 data: bool = False,
                 virtual: bool = False,
                 override_node: bool = False,
                 tensor: Optional[Tensor] = None,
                 edges: Optional[List['Edge']] = None,
                 override_edges: bool = False,
                 node1_list: Optional[List[bool]] = None,
                 init_method: Optional[Text] = None,
                 device: Optional[torch.device] = None,
                 **kwargs: float) -> None:

        # Check data
        if data:
            raise ValueError('ParamNode cannot be a data node')

        # Check leaf
        if hasattr(self, '_leaf'):
            # Case _create_resultant, _leaf = False
            raise ValueError('ParamNode cannot be a resultant node')

        super().__init__(shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         data=data,
                         virtual=virtual,
                         override_node=override_node,
                         tensor=tensor,
                         edges=edges,
                         override_edges=override_edges,
                         node1_list=node1_list,
                         init_method=init_method,
                         device=device,
                         **kwargs)

    # ----------
    # Properties
    # ----------
    @property
    def grad(self) -> Optional[Tensor]:
        """
        Returns gradient of the param-node's tensor.

        See also `torch.Tensor.grad()
        <https://pytorch.org/docs/stable/generated/torch.Tensor.grad.html>`_

        Returns
        -------
        torch.Tensor or None
        """
        if self._tensor_info['address'] is None:
            aux_node = self._tensor_info['node_ref']
            tensor = aux_node._network._memory_nodes[aux_node._tensor_info['address']]
        else:
            tensor = self._network._memory_nodes[self._tensor_info['address']]

        if tensor is None:
            return

        aux_grad = tensor.grad
        if aux_grad is None:
            return aux_grad
        else:
            if self._tensor_info['full']:
                return aux_grad
            return aux_grad[self._tensor_info['index']]

    # -------
    # Methods
    # -------
    def _make_edge(self, axis: Axis) -> 'Edge':
        """Makes ``Edges`` that will be attached to each axis."""
        return Edge(node1=self, axis1=axis)

    @staticmethod
    def _set_tensor_format(tensor: Tensor) -> Parameter:
        """Returns a nn.Parameter if input tensor is just torch.Tensor."""
        if isinstance(tensor, Parameter):
            return tensor
        return Parameter(tensor)

    def parameterize(self, set_param: bool = True) -> Union['Node', 'ParamNode']:
        """
        Replaces the param-node with a de-parameterized version of it, that is,
        turns a :class:`ParamNode` into a non-trainable, fixed :class:`Node`.

        Since the param-node is `replaced`, it will be completely removed from
        the network, and its neighbours will point to the new node.

        Parameters
        ----------
        set_param : bool
            Boolean indicating whether the node should stay parameterized
            (``True``), thus returning the param-node itself. Otherwise (``False``),
            the param-node will be de-parameterized.

        Returns
        -------
        ParamNode or Node
            The original node or a de-parameterized version of it.
        """
        if not set_param:
            new_node = Node(shape=self.shape,
                            axes_names=self.axes_names,
                            name=self._name,
                            network=self._network,
                            override_node=True,
                            tensor=self.tensor,
                            edges=self._edges,
                            override_edges=True,
                            node1_list=self.is_node1())
            return new_node
        else:
            return self

    def copy(self, share_tensor: bool = False) -> 'ParamNode':
        """
        Returns a copy of the param-node. That is, returns a param-node whose
        tensor is a copy of the original, whose edges are directly inherited
        (these are not copies, but the exact same edges) and whose name is
        extended with the prefix ``"copy_"``.

        Returns
        -------
        ParamNode
        """
        if share_tensor:
            new_node = ParamNode(shape=self._shape,
                        axes_names=self.axes_names,
                        name=self._name + '_copy',
                        network=self._network,
                        edges=self._edges,
                        node1_list=self.is_node1())
            new_node.set_tensor_from(self)
        else:  
            new_node = ParamNode(shape=self._shape,
                            axes_names=self.axes_names,
                            name=self._name + '_copy',
                            network=self._network,
                            tensor=self.tensor,
                            edges=self._edges,
                            node1_list=self.is_node1())
        return new_node


###############################################################################
#                                 STACK NODES                                 #
###############################################################################
class StackNode(Node):
    """
    Class for stacked nodes. ``StackNodes`` are nodes that store the information
    of a list of nodes that are stacked via :func:`stack`, although they can also
    be instantiated directly. To do so, there are two options:

    * Provide a sequence of nodes: if ``nodes`` are provided, their tensors will
      be stacked and stored in the ``StackNode``. It is necessary that all nodes
      are of the same type (:class:`Node` or :class:`ParamNode`), have the same
      rank (although dimension of each leg can be different for different nodes;
      in which case smaller tensors are extended with 0's to match the dimensions
      of the largest tensor in the stack), same axes names (to ensure only the
      `same kind` of nodes are stacked), belong to the same network and have edges
      with the same type in each axis (:class:`Edge` or :class:`ParamEdge`).

    * Provide a stacked tensor: if the stacked ``tensor`` is provided, it is also
      necessary to specify the ``axes_names``, ``network``, ``edges``, ``node1_list``.

    ``StackNodes`` have an additional axis for the new `stack` dimension, which
    is a batch edge. This way, some contractions can be computed in parallel by
    first stacking two sequences of nodes (connected pair-wise), performing the
    batch contraction and finally unbinding the ``StackNodes`` to retrieve just
    one sequence of nodes.

    For the rest of the axes, a list of the edges corresponding to all nodes in
    the stack is stored, so that, when :func:`unbinding <unbind>` the stack, it
    can be inferred to which nodes the unbinded nodes have to be connected.

    Parameters
    ----------
    nodes : list[AbstractNode] or tuple[AbstractNode], optional
        Sequence of nodes that are to be stacked.
    axes_names : list[str], tuple[str], optional
        Sequence of names for each of the node's axes. Names are used to access
        the edge that is attached to the node in a certain axis. Hence they should
        be all distinct. Necessary if ``nodes`` are not provided.
    name : str, optional
        Node's name, used to access the node from de :class:`TensorNetwork` where
        it belongs. It cannot contain blank spaces.
    network : TensorNetwork, optional
        Tensor network where the node should belong. Necessary if ``nodes`` are
        not provided.
    override_node : bool, optional
        Boolean indicating whether the node should override (``True``) another
        node in the network that has the same name (e.g. if a node is parameterized,
        it would be required that a new :class:`ParamNode` replaces the non-parameterized
        node in the network).
    tensor : torch.Tensor, optional
        Tensor that is to be stored in the node. Necessary if ``nodes`` are not
        provided.
    edges : list[Edge], optional
        List of edges that are to be attached to the node. Necessary if ``nodes``
        are not provided.
    node1_list : list[bool], optional
        If ``edges`` are provided, the list of ``node1`` attributes of each edge
        should also be provided. Necessary if ``nodes`` are not provided.

    Example
    -------
    >>> net = tk.TensorNetwork()
    >>> nodes = [tk.randn(shape=(2, 4, 2),
    ...                   axes_names=('left', 'input', 'right'),
    ...                   network=net)
    ...          for _ in range(10)]
    >>> data = [tk.randn(shape=(4,),
    ...                  axes_names=('feature',),
    ...                  network=net)
    ...         for _ in range(10)]
    ...
    >>> for i in range(10):
    ...     _ = nodes[i]['input'] ^ data[i]['feature']
    ...
    >>> stack_nodes = tk.stack(nodes)
    >>> stack_data = tk.stack(data)
    ...
    >>> # It is necessary to re-connect stacks
    >>> _ = stack_nodes['input'] ^ stack_data['feature']
    >>> result = tk.unbind(stack_nodes @ stack_data)
    >>> print(result[0].name)
    unbind_0

    >>> print(result[0].axes)
    [Axis( left (0) ), Axis( right (1) )]

    >>> print(result[0].shape)
    torch.Size([2, 2])
    """

    def __init__(self,
                 nodes: Optional[Sequence[AbstractNode]] = None,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 override_node: bool = False,
                 tensor: Optional[Tensor] = None,
                 edges: Optional[List['Edge']] = None,
                 node1_list: Optional[List[bool]] = None) -> None:

        if nodes is not None:
            if not isinstance(nodes, (list, tuple)):
                raise TypeError('`nodes` should be a list or tuple of nodes')
            for node in nodes:
                if isinstance(node, (StackNode, ParamStackNode)):
                    raise TypeError(
                        'Cannot create a stack using (Param)StackNode\'s')
            if tensor is not None:
                raise ValueError(
                    'If `nodes` are provided, `tensor` must not be given')

            # Check all nodes share properties
            for i in range(len(nodes[:-1])):
                if not isinstance(nodes[i], type(nodes[i + 1])):
                    raise TypeError('Cannot stack nodes of different types. Nodes '
                                    'must be either all Node or all ParamNode type')
                if nodes[i].rank != nodes[i + 1].rank:
                    raise ValueError(
                        'Cannot stack nodes with different number of edges')
                if nodes[i].axes_names != nodes[i + 1].axes_names:
                    raise ValueError(
                        'Stacked nodes must have the same name for each axis')
                if nodes[i]._network != nodes[i + 1]._network:
                    raise ValueError(
                        'Stacked nodes must all be in the same network')

            edges_dict = dict()  # Each axis has a list of edges
            node1_lists_dict = dict()
            for node in nodes:
                for axis in node._axes:
                    edge = node[axis]
                    if axis._name not in edges_dict:
                        edges_dict[axis._name] = [edge]
                        node1_lists_dict[axis._name] = [axis._node1]
                    else:
                        edges_dict[axis._name].append(edge)
                        node1_lists_dict[axis._name].append(axis._node1)

            self._edges_dict = edges_dict
            self._node1_lists_dict = node1_lists_dict

            tensor = stack_unequal_tensors([node.tensor for node in nodes])
            super().__init__(axes_names=['stack'] + nodes[0].axes_names,
                             name=name,
                             network=nodes[0]._network,
                             override_node=override_node,
                             tensor=tensor)

        else:
            # Case stacked tensor is provided, and there is no need of having
            # to stack the nodes' tensors
            if axes_names is None:
                raise ValueError(
                    'If `nodes` are not provided, `axes_names` must be given')
            if network is None:
                raise ValueError(
                    'If `nodes` are not provided, `network` must be given')
            if tensor is None:
                raise ValueError(
                    'If `nodes` are not provided, `tensor` must be given')
            if edges is None:
                raise ValueError(
                    'If `nodes` are not provided, `edges` must be given')
            if node1_list is None:
                raise ValueError(
                    'If `nodes` are not provided, `node1_list` must be given')

            edges_dict = dict()
            node1_lists_dict = dict()
            for axis_name, edge in zip(axes_names[1:], edges[1:]):
                edges_dict[axis_name] = edge.edges
                node1_lists_dict[axis_name] = edge.node1_lists

            self._edges_dict = edges_dict
            self._node1_lists_dict = node1_lists_dict

            super().__init__(axes_names=axes_names,
                             name=name,
                             network=network,
                             override_node=override_node,
                             tensor=tensor,
                             edges=edges,
                             node1_list=node1_list)

    # ----------
    # Properties
    # ----------
    @property
    def edges_dict(self) -> Dict[Text, List['Edge']]:
        """Returns dictionary with list of edges of each axis."""
        return self._edges_dict

    @property
    def node1_lists_dict(self) -> Dict[Text, List[bool]]:
        """Returns a dictionary with list of ``node1_list`` attribute of each axis."""
        return self._node1_lists_dict

    # -------
    # Methods
    # -------
    def _make_edge(self, axis: Axis) -> 'Edge':
        """
        Makes ``StackEdges``that will be attached to each axis. Also makes an
        ``Edge`` for the stack dimension.
        """
        if axis._num == 0:
            # Stack axis
            return Edge(node1=self, axis1=axis)
        else:
            return StackEdge(edges=self._edges_dict[axis._name],
                             node1_lists=self._node1_lists_dict[axis._name],
                             node1=self, axis1=axis)


class ParamStackNode(ParamNode):
    """
    Class for parametric stacked nodes. They are essentially the same as
    :class:`StackNodes <StackNode>` but they are also :class:`ParamNodes <ParamNode>`.
    They are used to optimize memory usage and save some time when the first
    operation that occurs to param-nodes in a contraction (that might be
    computed several times during training) is :func:`stack`. If this is the case,
    the param-nodes no longer store their own tensors, but rather they make
    reference to a slide of a greater ``ParamStackNode`` (if ``automemory`` attribute
    of the :class:`TensorNetwork` is set to ``True``). Hence, that first :func:`stack`
    is never computed.

    ``ParamStackNodes`` can only be instantiated by providing a sequence of nodes.

    Parameters
    ----------
    nodes : list[AbstractNode] or tuple[AbstractNode]
        Sequence of nodes that are to be stacked.
    name : str, optional
        Node's name, used to access the node from de :class:`TensorNetwork` where
        it belongs. It cannot contain blank spaces.
    virtual : bool, optional
        Boolean indicating if the node is a ``virtual`` node. Since it will be
        used mainly for the case described :class:`here <ParamStackNode>`, the
        node will be virtual, since it will not be an `effective` part of the
        tensor network, but rather a `virtual` node used just to store tensors.
    override_node : bool, optional
        Boolean indicating whether the node should override (``True``) another
        node in the network that has the same name (e.g. if a node is parameterized,
        it would be required that a new :class:`ParamNode` replaces the non-parameterized
        node in the network).

    Example
    -------
    >>> net = tk.TensorNetwork()
    >>> net.automemory = True
    >>> nodes = [tk.randn(shape=(2, 4, 2),
    ...                   axes_names=('left', 'input', 'right'),
    ...                   network=net,
    ...                   param_node=True)
    ...          for _ in range(10)]
    >>> data = [tk.randn(shape=(4,),
    ...                  axes_names=('feature',),
    ...                  network=net)
    ...         for _ in range(10)]
    ...
    >>> for i in range(10):
    ...     _ = nodes[i]['input'] ^ data[i]['feature']
    ...
    >>> stack_nodes = tk.stack(nodes)
    >>> stack_nodes.name = 'my_stack'
    >>> # ._tensor_info has info regarding where the node's tensor is stored
    >>> print(nodes[0]._tensor_info['node_ref'].name)
    my_stack

    >>> stack_data = tk.stack(data)
    ...
    >>> # It is necessary to re-connect stacks
    >>> _ = stack_nodes['input'] ^ stack_data['feature']
    >>> result = tk.unbind(stack_nodes @ stack_data)
    >>> print(result[0].name)
    unbind_0

    >>> print(result[0].axes)
    [Axis( left (0) ), Axis( right (1) )]

    >>> print(result[0].shape)
    torch.Size([2, 2])
    """

    def __init__(self,
                 nodes: Sequence[AbstractNode],
                 name: Optional[Text] = None,
                 virtual: bool = False,
                 override_node: bool = False) -> None:

        if not isinstance(nodes, (list, tuple)):
            raise TypeError('`nodes` should be a list or tuple of nodes')

        for node in nodes:
            if isinstance(node, (StackNode, ParamStackNode)):
                raise TypeError(
                    'Cannot create a stack using (Param)StackNode\'s')

        for i in range(len(nodes[:-1])):
            if not isinstance(nodes[i], type(nodes[i + 1])):
                raise TypeError('Cannot stack nodes of different types. Nodes '
                                'must be either all Node or all ParamNode type')
            if nodes[i].rank != nodes[i + 1].rank:
                raise ValueError(
                    'Cannot stack nodes with different number of edges')
            if nodes[i].axes_names != nodes[i + 1].axes_names:
                raise ValueError(
                    'Stacked nodes must have the same name for each axis')
            if nodes[i]._network != nodes[i + 1]._network:
                raise ValueError(
                    'Stacked nodes must all be in the same network')

        edges_dict = dict()
        node1_lists_dict = dict()
        for node in nodes:
            for axis in node._axes:
                edge = node[axis]
                if axis._name not in edges_dict:
                    edges_dict[axis._name] = [edge]
                    node1_lists_dict[axis._name] = [axis._node1]
                else:
                    edges_dict[axis._name].append(edge)
                    node1_lists_dict[axis._name].append(axis._node1)

        self._edges_dict = edges_dict
        self._node1_lists_dict = node1_lists_dict
        self.nodes = nodes

        tensor = stack_unequal_tensors([node.tensor for node in nodes])
        super().__init__(axes_names=['stack'] + nodes[0].axes_names,
                         name=name,
                         network=nodes[0]._network,
                         virtual=virtual,
                         override_node=override_node,
                         tensor=tensor)

    # ----------
    # Properties
    # ----------
    @property
    def edges_dict(self) -> Dict[Text, List['Edge']]:
        """Returns dictionary with list of edges of each axis."""
        return self._edges_dict

    @property
    def node1_lists_dict(self) -> Dict[Text, List[bool]]:
        """Returns a dictionary with list of ``node1_list`` attribute of each axis."""
        return self._node1_lists_dict

    # -------
    # Methods
    # -------
    def _make_edge(self, axis: Axis) -> 'Edge':
        """
        Makes ``StackEdges``that will be attached to each axis. Also makes an
        ``Edge`` for the stack dimension.
        """
        if axis.num == 0:
            # Stack axis
            return Edge(node1=self, axis1=axis)
        else:
            return StackEdge(edges=self._edges_dict[axis._name],
                             node1_lists=self._node1_lists_dict[axis._name],
                             node1=self, axis1=axis)


###############################################################################
#                                    EDGES                                    #
###############################################################################
class Edge:
    """
    Base class for edges. Should be subclassed by any new class of edges.

    An edge is nothing more than an object that wraps references to the nodes it
    connects. Thus it stores information like the nodes it connects, the corresponding
    nodes' axes it is attached to, whether it is dangling or batch, its size, etc.

    Above all, its importance lies in that edges enable to connect nodes, forming
    any possible graph, and to perform easily :class:`Operations <Operation>` like
    contracting and splitting nodes.

    Furthermore, edges have specific operations like :meth:`contract_` or :meth:`svd_`
    (and its variations) that allow in-place modification of the :class:`TensorNetwork`.

    Parameters
    ----------
    node1 : AbstractNode
        First node to which the edge is connected.
    axis1: int, str or Axis
        Axis of ``node1`` where the edge is attached.
    node2 : AbstractNode, optional
        Second node to which the edge is connected. If None, the edge will be
        dangling.
    axis2 : int, str, Axis, optional
        Axis of ``node2`` where the edge is attached.
    """

    def __init__(self,
                 node1: AbstractNode,
                 axis1: Ax,
                 node2: Optional[AbstractNode] = None,
                 axis2: Optional[Ax] = None) -> None:

        super().__init__()

        # check node1 and axis1
        if not isinstance(node1, AbstractNode):
            raise TypeError('`node1` should be AbstractNode type')
        if not isinstance(axis1, (int, str, Axis)):
            raise TypeError('`axis1` should be int, str or Axis type')
        if not isinstance(axis1, Axis):
            axis1 = node1.get_axis(axis1)

        # check node2 and axis2
        if (node2 is None) != (axis2 is None):
            raise ValueError(
                '`node2` and `axis2` must both be None or both not be None')
        if node2 is not None:
            if not isinstance(node2, AbstractNode):
                raise TypeError('`node2` should be AbstractNode type')
            if not isinstance(axis2, (int, str, Axis)):
                raise TypeError('`axis2` should be int, str or Axis type')
            if not isinstance(axis2, Axis):
                axis2 = node2.get_axis(axis2)

            if node1._shape[axis1._num] != node2._shape[axis2._num]:
                raise ValueError('Sizes of `axis1` and `axis2` should match')
            if (node2 == node1) and (axis2 == axis1):
                raise ValueError(
                    'Cannot connect the same axis of the same node to itself')

        self._nodes = [node1, node2]
        self._axes = [axis1, axis2]

    # ----------
    # Properties
    # ----------
    @property
    def node1(self) -> AbstractNode:
        """Returns `node1` of the edge."""
        return self._nodes[0]

    @property
    def node2(self) -> AbstractNode:
        """Returns `node2` of the edge. If the edge is dangling, it is None."""
        return self._nodes[1]

    @property
    def nodes(self) -> List[AbstractNode]:
        """Returns a list with `node1` and `node2`."""
        return self._nodes

    @property
    def axis1(self) -> Axis:
        """Returns axis where the edge is attached to `node1`."""
        return self._axes[0]

    @property
    def axis2(self) -> Axis:
        """
        Returns axis where the edge is attached to `node2`. If the edge is dangling,
        it is None.
        """
        return self._axes[1]

    @property
    def axes(self) -> List[Axis]:
        """
        Returns a list of axes where the edge is attached to `node1` and `node2`,
        respectively.
        """
        return self._axes

    @property
    def name(self) -> Text:
        """
        Returns edge's name. It is formed with the corresponding nodes' and axes'
        names.

        Example
        -------
        >>> nodeA = tk.Node(shape=(2, 3), name='nodeA', axes_names=['left', 'right'])
        >>> edge = nodeA['right']
        >>> print(edge.name)
        nodeA[right] <-> None

        >>> nodeB = tk.Node(shape=(3, 4), name='nodeB', axes_names=['left', 'right'])
        >>> new_edge = nodeA['right'] ^ nodeB['left']
        >>> print(new_edge.name)
        nodeA[right] <-> nodeB[left]
        """
        if self.is_dangling():
            return f'{self.node1._name}[{self.axis1._name}] <-> None'
        return f'{self.node1._name}[{self.axis1._name}] <-> ' \
               f'{self.node2._name}[{self.axis2._name}]'

    # -------
    # Methods
    # -------
    def is_dangling(self) -> bool:
        """Returns boolean indicating whether the edge is a dangling edge."""
        return self.node2 is None

    def is_batch(self) -> bool:
        """Returns boolean indicating whether the edge is a batch edge."""
        return self.axis1.is_batch()

    def is_attached_to(self, node: AbstractNode) -> bool:
        """Returns boolean indicating whether the edge is attached to ``node``."""
        return (self.node1 == node) or (self.node2 == node)

    def size(self) -> int:
        """Returns edge's size."""
        return self._nodes[0]._shape[self._axes[0]._num]

    def change_size(self, size: int) -> None:
        """
        Changes size of the edge, thus changing the size of tensors of `node1`
        and `node2` at the corresponding axes. If new size is smaller, the tensor
        will be cropped; if larger, the tensor will be expanded with zeros. In
        both cases, the process (cropping/expanding) occurs at the "left", "top",
        "front", etc. of each dimension.
        """
        if not isinstance(size, int):
            TypeError('`size` should be int type')
        if not self.is_dangling():
            self.node2._change_axis_size(self.axis2, size)
        self.node1._change_axis_size(self.axis1, size)

    def copy(self) -> 'Edge':
        """
        Returns a copy of the edge, that is, a new edge referencing the same
        nodes at the same axes.
        """
        new_edge = Edge(node1=self.node1, axis1=self.axis1,
                        node2=self.node2, axis2=self.axis2)
        return new_edge

    def connect(self, other: 'Edge') -> 'Edge':
        """
        Connects dangling edge to another dangling edge.

        It is necessary that both edges have the same dimension so that contractions
        along that edge can be computed.

        Parameters
        ----------
        other : Edge
            The other edge to which current edge will be connected.

        Returns
        -------
        Edge

        Example
        -------
        To connect two edges, the overloaded operator ``^`` can also be used.

        >>> nodeA = tk.Node(shape=(2, 3), name='nodeA', axes_names=['left', 'right'])
        >>> nodeB = tk.Node(shape=(3, 4), name='nodeB', axes_names=['left', 'right'])
        >>> new_edge = nodeA['right'] ^ nodeB['left']  # Same as .connect()
        >>> print(new_edge.name)
        nodeA[right] <-> nodeB[left]
        """
        return connect(self, other)

    def disconnect(self) -> Tuple['Edge', 'Edge']:
        """
        Disconnects connected edge, that is, the connected edge is splitted into
        two dangling edges, one for each node.

        Returns
        -------
        tuple[Edge, Edge]

        Example
        -------
        To disconnect an edge, the overloaded operator ``|`` can also be used.

        >>> nodeA = tk.Node(shape=(2, 3), name='nodeA', axes_names=['left', 'right'])
        >>> nodeB = tk.Node(shape=(3, 4), name='nodeB', axes_names=['left', 'right'])
        >>> new_edge = nodeA['right'] ^ nodeB['left']
        >>> new_edgeA, new_edgeB = new_edge | new_edge  # Same as .disconnect()
        >>> print(new_edgeA.name)
        nodeA[right] <-> None

        >>> print(new_edgeB.name)
        nodeB[left] <-> None
        """
        return disconnect(self)

    def __xor__(self, other: 'Edge') -> 'Edge':
        return self.connect(other)

    def __or__(self, other: 'Edge') -> List['Edge']:
        if other == self:
            return self.disconnect()
        else:
            raise ValueError(
                'Cannot disconnect one edge from another, different one. '
                'Edge should be disconnected from itself')

    def __str__(self) -> Text:
        return self.name

    def __repr__(self) -> Text:
        if self.is_batch():
            return f'{self.__class__.__name__}( {self.name} )  (Batch Edge)'
        if self.is_dangling():
            return f'{self.__class__.__name__}( {self.name} )  (Dangling Edge)'
        return f'{self.__class__.__name__}( {self.name} )'


###############################################################################
#                                 STACK EDGES                                 #
###############################################################################
AbstractStackNode = Union[StackNode, ParamStackNode]


class StackEdge(Edge):
    """
    Class for stacked edges. They are just like :class:`Edges <Edge>` but used
    when stacking a collection of nodes into a :class:`StackNode`. When doing
    this, all edges of the stacked nodes must be kept, since they have the
    information regarding the nodes' neighbours, which will be used when :func:
    `unbinding <unbind>` the stack. Thus, ``StackEdges`` have two additional
    properties, ``edges`` and ``node1_lists``, that is, the edges of all stacked
    nodes corresponding to a certain axis, and their ``node1_list``'s.

    Parameters
    ----------
    edges : list[Edge]
        List of non-trainable edges that will be stacked.
    node1_lists : list[bool]
        List of ``node1_list``'s corresponding to each edge in ``edges``.
    node1 : StackNode or ParamStackNode
        First node to which the edge is connected.
    axis1: int, str or Axis
        Axis of ``node1`` where the edge is attached.
    node2 : StackNode or ParamStackNode, optional
        Second node to which the edge is connected. If None, the edge will be
        dangling.
    axis2 : int, str, Axis, optional
        Axis of ``node2`` where the edge is attached.
    """

    def __init__(self,
                 edges: List[Edge],
                 node1_lists: List[bool],
                 node1: AbstractStackNode,
                 axis1: Axis,
                 node2: Optional[AbstractStackNode] = None,
                 axis2: Optional[Axis] = None) -> None:
        self._edges = edges
        self._node1_lists = node1_lists
        super().__init__(node1=node1, axis1=axis1,
                         node2=node2, axis2=axis2)

    @property
    def edges(self) -> List[Edge]:
        """Returns list of stacked edges corresponding to this axis."""
        return self._edges

    @property
    def node1_lists(self) -> List[bool]:
        """Returns list of ``node1_list``'s corresponding to this axis."""
        return self._node1_lists

    def connect(self, other: 'StackEdge') -> 'StackEdge':
        """
        Same as :meth:`~Edge.connect` but it is verified that all stacked edges
        corresponding to both ``StackEdges`` are the same. That is, this is a
        redundant operation to re-connect a list of edges that should be already
        connected. However, this is mandatory, since when stacking two sequences
        of nodes independently it cannot be inferred that the resultant
        ``StackNodes`` had to be connected.

        Parameters
        ----------
        other : StackEdge
            The other edge to which current edge will be connected.

        Returns
        -------
        StackEdge

        Example
        -------
        To connect two stack-edges, the overloaded operator ``^`` can also be used.

        >>> net = tk.TensorNetwork()
        >>> nodes = [tk.randn(shape=(2, 4, 2),
        ...                   axes_names=('left', 'input', 'right'),
        ...                   network=net)
        ...          for _ in range(10)]
        >>> data = [tk.randn(shape=(4,),
        ...                  axes_names=('feature',),
        ...                  network=net)
        ...         for _ in range(10)]
        ...
        >>> for i in range(10):
        ...     _ = nodes[i]['input'] ^ data[i]['feature']
        ...
        >>> stack_nodes = tk.stack(nodes)
        >>> stack_data = tk.stack(data)
        ...
        >>> # It is necessary to re-connect stacks to be able to contract
        >>> _ = stack_nodes['input'] ^ stack_data['feature']
        """
        return connect_stack(self, other)


###############################################################################
#                               EDGE OPERATIONS                               #
###############################################################################
def connect(edge1: Edge, edge2: Edge) -> Edge:
    # TODO: change docstring
    """
    Connects two dangling edges. If both are :class:`Edges <Edge>`, the result
    will be also an ``Edge``. Otherwise, the result will be a :class:`ParamEdge`.

    It is necessary that both edges have the same dimension so that contractions
    along that edge can be computed. Note that for ``ParamEdges`` the sizes could
    be different. In this case the resultant param-edge will have the minimum
    size between them.

    This operation is the same as :meth:`Edge.connect` and :meth:`ParamEdge.connect`.

    Parameters
    ----------
    edge1 : Edge
        The first edge that will be connected. Its node will become the ``node1``
        of the resultant edge.
    edge2 : Edge
        The second edge that will be connected. Its node will become the ``node2``
        of the resultant edge.

    Returns
    -------
    Edge
    """
    # Case edge is already connected
    if edge1 == edge2:
        return edge1

    for edge in [edge1, edge2]:
        if not edge.is_dangling():
            raise ValueError(f'Edge {edge!s} is not a dangling edge. '
                             f'This edge points to nodes: {edge.node1!s} and '
                             f'{edge.node2!s}')
        if edge.is_batch():
            raise ValueError(f'Edge {edge!s} is a batch edge. Batch edges '
                             'cannot be connected')

    if edge1.size() != edge2.size():
        raise ValueError(f'Cannot connect edges of unequal size. '
                         f'Size of edge {edge1!s}: {edge1.size()}. '
                         f'Size of edge {edge2!s}: {edge2.size()}')

    node1, axis1 = edge1.node1, edge1.axis1
    node2, axis2 = edge2.node1, edge2.axis1
    net1, net2 = node1._network, node2._network

    if net1 != net2:
        node2.move_to_network(net1)
    net1._remove_edge(edge1)
    net1._remove_edge(edge2)
    net = net1

    if isinstance(edge1, StackEdge):
        new_edge = StackEdge(edges=edge1._edges,
                             node1_lists=edge1._node1_lists,
                             node1=node1, axis1=axis1,
                             node2=node2, axis2=axis2)
    else:
        new_edge = Edge(node1=node1, axis1=axis1,
                        node2=node2, axis2=axis2)

    node1._add_edge(new_edge, axis1, True)
    node2._add_edge(new_edge, axis2, False)
    return new_edge


def connect_stack(edge1: StackEdge, edge2: StackEdge) -> StackEdge:
    # TODO: change docstring
    """
    Same as :func:`connect` but it is verified that all stacked edges corresponding
    to both ``(Param)StackEdges`` are the same. That is, this is a redundant
    operation to re-connect a list of edges that should be already connected.
    However, this is mandatory, since when stacking two sequences of nodes
    independently it cannot be inferred that the resultant ``(Param)StackNodes``
    had to be connected.

    This operation is the same as :meth:`StackEdge.connect` and
    :meth:`ParamStackEdge.connect`.

    Parameters
    ----------
    edge1 : StackEdge or ParamStackEdge
        The first edge that will be connected. Its node will become the ``node1``
        of the resultant edge.
    edge2 : StackEdge or ParamStackEdge
        The second edge that will be connected. Its node will become the ``node2``
        of the resultant edge.
    """
    if not isinstance(edge1, StackEdge) or not isinstance(edge2, StackEdge):
        raise TypeError('Both edges should be StackEdge\'s')

    if edge1._edges != edge2._edges:
        raise ValueError('Cannot connect stack edges whose lists of edges are '
                         'not the same. They will be the same when both lists '
                         'contain edges connecting the nodes that formed the '
                         'stack nodes.')
    return connect(edge1=edge1, edge2=edge2)


def disconnect(edge: Edge) -> Tuple[Edge, Edge]:
    """
    Disconnects connected edge, that is, the connected edge is splitted into
    two dangling edges, one for each node.

    This operation is the same as :meth:`Edge.disconnect`.

    Parameters
    ----------
    edge : Edge
        Edge that is going to be disconnected (splitted in two).

    Returns
    -------
    tuple[Edge, Edge]
    """
    if edge.is_dangling():
        raise ValueError('Cannot disconnect a dangling edge')

    # This is to avoid disconnecting an edge when we are trying to disconnect a
    # copy of that edge, in which case `copy_edge` might connect `node1` and
    # `node2`, but none of them "sees" `copy_edge`, since they have `edge`
    # connecting them
    nodes = []
    axes = []
    for axis, node in zip(edge._axes, edge._nodes):
        if edge in node._edges:
            nodes.append(node)
            axes.append(axis)

    new_edges = []
    for axis, node in zip(axes, nodes):
        if isinstance(edge, StackEdge):
            new_edge = StackEdge(edges=edge._edges,
                                 node1_lists=edge._node1_lists,
                                 node1=node,
                                 axis1=axis)
            new_edges.append(new_edge)

        else:
            new_edge = Edge(node1=node, axis1=axis)
            new_edges.append(new_edge)

            net = node._network
            net._add_edge(new_edge)

    for axis, node, new_edge in zip(axes, nodes, new_edges):
        node._add_edge(new_edge, axis, True)

    return tuple(new_edges)


###############################################################################
#                                   SUCCESSOR                                 #
###############################################################################
class Successor:
    """
    Class for successors. This is a sort of cache memory for :class:`operations
    <Operation>` that have been already computed.

    For instance, when contracting two nodes, the result gives a new node that
    stores the tensor resultant from contracting both nodes's tensors. However,
    when training a :class:`TensorNetwork`, the tensors inside the nodes will
    change every epoch, but there is actually no need to create a new resultant
    node every time. Instead, it is more efficient to keep track of which node
    arose as the result of an operation, and simply change its tensor.

    Hence, a ``Successor`` is instantiated providing the arguments of the operation
    that gave rise to a resultant node, a reference to the resultant node itself,
    and some hints that might help accelerating the computations the next time
    the operation is performed.

    These three properties can be accessed via ``successor.kwargs``, ``successor.child``
    and ``successor.hints``.

    Parameters
    ----------
    kwargs : dict[str, any]
        Dictionary with keyword arguments used to call an operation.
    child : AbstractNode or list[AbstractNode]
        The node or list of nodes that result from an operation.
    hints : dict[str, any], optional
        A dictionary of hints created the first time an operation is computed in
        order to save some computation in the next calls of the operation.

    Example
    -------
    When contracting two nodes, a ``Successor`` is created and added to the list
    of successors of the first node (left operand).

    >>> nodeA = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
    >>> nodeB = tk.randn(shape=(3, 4), axes_names=('left', 'right'))
    ...
    >>> # Connect nodes
    >>> _ = nodeA['right'] ^ nodeB['left']
    ...
    >>> # Contract nodes
    >>> result = nodeA @ nodeB
    >>> result.name = 'my_result'
    ...
    >>> print(result.name, nodeA.successors['contract_edges'][0].child.name)
    my_result my_result
    """

    def __init__(self,
                 kwargs: Dict[Text, Any],
                 child: Union[AbstractNode, List[AbstractNode]],
                 hints: Optional[Dict[Text, Any]] = None) -> None:
        self.kwargs = kwargs
        self.child = child
        self.hints = hints


###############################################################################
#                                TENSOR NETWORK                               #
###############################################################################
class TensorNetwork(nn.Module):
    """
    Class for arbitrary Tensor Networks. Subclass of **PyTorch** ``nn.Module``.

    Tensor Networks are the central objects of **TensorKrowch**. Basically,
    a tensor network is a graph where vertices are :class:`Nodes <AbstractNode>`
    and edges are, pun intended, :class:`Edges <Edge>`. In these models,
    nodes' tensors will be trained so that the contraction of the whole network
    approximates a certain function. Hence, Tensor Networks are the `trainable
    objects` of **TensorKrowch**, very much like ``nn.Module``'s are the
    `trainable objects` of **PyTorch**.

    Recall that the common way of defining models out of ``nn.Module`` is by
    defining a subclass where the ``__init__`` and ``forward`` methods are
    overriden:

    * ``__init__``: Defines the model itself (its layers, attributes, etc.).
    * ``forward``: Defines the way the model operates, that is, how the different
      parts of the model migh combine to get an output from a particular input.

    With ``TensorNetwork``, the workflow is similar, though there are other
    methods that should be overriden:

    * ``__init__``: Defines the graph of the tensor network and initializes the
      tensors of the nodes.
    * ``set_data_nodes``: Creates the data nodes where the data tensor(s) will
      be placed. Usually, it will just select the edges to which the data nodes
      should be connected, and call the parent method ``set_data_nodes``.
    * ``contract``: Defines the contraction algorithm of the whole tensor network,
      thus returning a single node. Very much like ``forward`` this is the main
      method that describes how the components of the network are combined.
      Hence, in ``TensorNetwork`` the ``forward`` method shall not be overriden,
      since it will just call ``contract``.

    Although one can define how the network is going to be contracted, there a
    couple of modes that can change how this contraction behaves at a lower level:

    * **automemory** (``False`` by default): This mode indicates whether node
      :class:`Operations <Operation>` have the ability to take control of the
      memory management of the network. For instance, if ``automemory`` is set
      to ``True`` and a collection of :class:`ParamNodes <ParamNode>` are
      :func:`stacked <stack>` (as the first operation in the contraction),
      then those nodes will no longer store their own tensors, but rather a
      ``virtual`` :class:`ParamStackNode` will store the stacked tensor, avoiding
      the computation of the first :func:`stack` in every contraction. This
      behaviour is not possible if ``automemory`` is set to ``False``, in which
      case all nodes will always store their own tensors.

    * **unbind_mode** (``False`` by default): This mode indicates whether the
      operation :func:`unbind` has to actually `unbind` the stacked tensor or
      just generate a collection of references. That is, if ``unbind_mode`` is
      set to ``True``, :func:`unbind` creates a collection of nodes, each of them
      storing the corresponding slice of the stacked tensor. If ``unbind_mode``
      is set to ``False`` (called ``index_mode``), :func:`unbind` just creates
      the nodes and gives each of them an index for the stacked tensor, so that
      each node's tensor would be retrieved by indexing the stack. This avoids
      performing the operation, since these indices will be the same in consecutive
      iterations. Hence, in a similar way to ``automemory``, this mode entails
      a certain control of the memory management of the network.

    Once the training algorithm starts, these modes should not be changed (very
    often at least), since changing them entails first resetting the whole
    network (see :meth:`reset`), which is a costly method. To understand what
    reset means, check the different types of nodes a network might have:

    * **leaf**: These are the nodes that make up the graph of the Tensor Network,
      except for the nodes containing data. These can be either type :class:`Node`
      or :class:`ParamNode` (trainable nodes).
    * **data**: These are :class:`Nodes <Node>` which store the tensors coming
      from input data. These are set via :meth:`set_data_nodes`.
    * **virtual**: These nodes (:class:`Node` or :class:`ParamNode`) are a sort
      of ancillay, hidden nodes that accomplish some useful task (e.g. in uniform
      tensor networks a virtual node has the shared tensor, while all the other
      nodes in the network just have a reference to it).
    * **resultant**: These are :class:`Nodes <Node>` that result from an
      :class:`Operation`. They are intermediate nodes that (almost always)
      inherit edges from ``leaf``  and ``data`` nodes, the ones that really
      establish the network's graph. Thus ``resultant`` nodes can be thought of
      like permutations or combinations of ``leaf`` nodes.

    This way, when the Tensor Network is defined, it has a bunch of ``leaf``,
    ``data`` and ``virtual`` nodes that make up the network structure, each of
    them storing its own tensor. However, when the network is contracted, several
    ``resultant`` nodes become new members of the network, even modifying its
    memory (depending on the ``automemory`` and ``unbind_mode`` modes). Therefore,
    if one wants to `reset` the network to its initial state after performing
    some operations, all the ``resultant`` nodes should be deleted, and all the
    tensors should return to its nodes. This is exactly what :meth:`reset` does.
    Besides, since ``automemory`` and ``unbind_mode`` can change how the tensors
    are stored, if one wants to change these modes, the network should be first
    reset (this is already done automatically when changing the modes).

    Example
    -------
    This is how one may define an **MPS**.

    ::

        class MPS(tk.TensorNetwork):

            def __init__(self, image_size, uniform=False):
                super().__init__(name='MPS')

                # Create TN
                input_nodes = []
                for _ in range(image_size[0] * image_size[1]):
                    node = tk.ParamNode(shape=(10, 3, 10),
                                        axes_names=('left',
                                                    'input',
                                                    'right'),
                                        name='input_node',
                                        network=self)
                    input_nodes.append(node)

                for i in range(len(input_nodes) - 1):
                    input_nodes[i]['right'] ^ input_nodes[i + 1]['left']

                # Periodic boundary conditions
                output_node = tk.ParamNode(shape=(10, 10, 10),
                                           axes_names=('left',
                                                       'output',
                                                       'right'),
                                           name='output_node',
                                           network=self)
                output_node['right'] ^ input_nodes[0]['left']
                output_node['left'] ^ input_nodes[-1]['right']

                self.input_nodes = input_nodes
                self.output_node = output_node

                if uniform:
                    uniform_memory = tk.ParamNode(shape=(10, 3, 10),
                                                  axes_names=('left',
                                                              'input',
                                                              'right'),
                                                  name='virtual_uniform',
                                                  network=self,
                                                  virtual=True)
                    self.uniform_memory = uniform_memory

                # Initialize nodes
                if uniform:
                    std = 1e-9
                    tensor = torch.randn(uniform_memory.shape) * std
                    random_eye = torch.randn(tensor.shape[0],
                                             tensor.shape[2]) * std
                    random_eye  = random_eye + torch.eye(tensor.shape[0],
                                                         tensor.shape[2])
                    tensor[:, 0, :] = random_eye

                    uniform_memory._unrestricted_set_tensor(tensor)

                    # Memory of each node is just a reference
                    # to the uniform_memory tensor
                    for node in input_nodes:
                        del self._memory_nodes[node._tensor_info['address']]
                        node._tensor_info['address'] = None
                        node._tensor_info['node_ref'] = uniform_memory
                        node._tensor_info['full'] = True
                        node._tensor_info['index'] = None

                else:
                    std = 1e-9
                    for node in input_nodes:
                        tensor = torch.randn(node.shape) * std
                        random_eye = torch.randn(tensor.shape[0],
                                                 tensor.shape[2]) * std
                        random_eye  = random_eye + torch.eye(tensor.shape[0],
                                                             tensor.shape[2])
                        tensor[:, 0, :] = random_eye

                        node.tensor = tensor

                eye_tensor = torch.eye(
                    output_node.shape[0],
                    output_node.shape[2]).view([output_node.shape[0],
                                                1,
                                                output_node.shape[2]])
                eye_tensor = eye_tensor.expand(output_node.shape)
                tensor = eye_tensor + std * torch.randn(output_node.shape)

                output_node.tensor = tensor

                self.input_nodes = input_nodes
                self.output_node = output_node

            def set_data_nodes(self) -> None:
                input_edges = []
                for node in self.input_nodes:
                    input_edges.append(node['input'])

                super().set_data_nodes(input_edges, 1)

            def contract(self):
                stack_input = tk.stack(self.input_nodes)
                stack_data = tk.stack(list(self.data_nodes.values()))

                stack_input['input'] ^ stack_data['feature']
                stack_result = stack_input @ stack_data

                stack_result = tk.unbind(stack_result)

                result = stack_result[0]
                for node in stack_result[1:]:
                    result @= node
                result @= self.output_node

                return result
    """

    operations = dict()  # References to the Operations defined for nodes

    def __init__(self, name: Optional[Text] = None):
        super().__init__()

        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name

        # Types of nodes of the TN
        self._leaf_nodes = dict()
        self._data_nodes = dict()
        self._virtual_nodes = dict()
        self._resultant_nodes = dict()

        # Repeated nodes to keep track of enumeration
        self._repeated_nodes_names = dict()

        # Edges
        self._edges = []

        # Memories
        self._memory_nodes = dict()   # address -> memory
        self._inverse_memory = dict()  # address -> nodes using that memory

        # TN modes
        # Auto-management of memory mode (train -> True)
        self._automemory = False
        self._unbind_mode = False  # Unbind/index mode (train -> True)
        self._tracing = False     # Tracing mode (True while calling .trace())

        # Lis of operations used to contract the TN
        self._seq_ops = []

    # ----------
    # Properties
    # ----------
    @property
    def nodes(self) -> Dict[Text, AbstractNode]:
        """
        Returns dictionary with all the nodes belonging to the network (``leaf``,
        ``data``, ``virtual`` and ``resultant``).
        """
        all_nodes = dict()
        all_nodes.update(self._leaf_nodes)
        all_nodes.update(self._data_nodes)
        all_nodes.update(self._virtual_nodes)
        all_nodes.update(self._resultant_nodes)
        return all_nodes

    @property
    def nodes_names(self) -> List[Text]:
        """
        Returns list of names of all the nodes belonging to the network (``leaf``,
        ``data``, ``virtual`` and ``resultant``).
        """
        all_nodes_names = []
        al_nodes_names += list(self._leaf_nodes.keys())
        al_nodes_names += list(self._data_nodes.keys())
        al_nodes_names += list(self._virtual_nodes.keys())
        al_nodes_names += list(self._resultant_nodes.keys())
        return all_nodes_names

    @property
    def leaf_nodes(self) -> Dict[Text, AbstractNode]:
        """Returns ``leaf`` nodes of the network."""
        return self._leaf_nodes

    @property
    def data_nodes(self) -> Dict[Text, AbstractNode]:
        """Returns ``data`` nodes of the network."""
        return self._data_nodes

    @property
    def virtual_nodes(self) -> Dict[Text, AbstractNode]:
        """Returns ``virtual`` nodes of the network."""
        return self._virtual_nodes

    @property
    def resultant_nodes(self) -> Dict[Text, AbstractNode]:
        """Returns ``resultant`` nodes of the network."""
        return self._resultant_nodes

    @property
    def edges(self) -> List[Edge]:
        """Returns list of dangling, non-batch edges of the network."""
        return self._edges

    @property
    def automemory(self) -> bool:
        """Returns boolean indicating whether ``automemory`` is on/off."""
        return self._automemory

    @automemory.setter
    def automemory(self, automem: bool) -> None:
        self.reset()
        self._automemory = automem

    @property
    def unbind_mode(self) -> bool:
        """Returns boolean indicating whether ``unbind_mode`` is on/off."""
        return self._unbind_mode

    @unbind_mode.setter
    def unbind_mode(self, unbind: bool) -> None:
        self.reset()
        self._unbind_mode = unbind

    # -------
    # Methods
    # -------
    def _which_dict(self, node: AbstractNode) -> Optional[Dict[Text, AbstractNode]]:
        """Returns the corresponding dict, depending on the type of the node."""
        if node._leaf:
            return self._leaf_nodes
        elif node._data:
            return self._data_nodes
        elif node._virtual:
            return self._virtual_nodes
        else:
            return self._resultant_nodes

    def _add_edge(self, edge: Edge) -> None:
        """
        Adds an edge to the network. If it is a :class:`ParamEdge` that is not
        already a sub-module of the network, it is added as such. If it is a
        ``(Param)StackEdge``, it is not added, since those edges are a sort of
        `virtual` edges.

        Parameters
        ----------
        edge : Edge
            Edge to be added.
        """
        if not isinstance(edge, StackEdge):
            if edge.is_dangling() and not edge.is_batch() and \
                    (edge not in self._edges):
                self._edges.append(edge)

    def _remove_edge(self, edge: Edge) -> None:
        """
        Removes an edge from the network. If it is a :class:`ParamEdge`, it is
        removed as sub-module.

        Parameters
        ----------
        edge : Edge
            Edge to be removed.
        """
        if not isinstance(edge, StackEdge):
            if edge.is_dangling() and not edge.is_batch() and \
                    (edge in self._edges):
                self._edges.remove(edge)

    def _add_node(self, node: AbstractNode, override: bool = False) -> None:
        """
        Adds a node to the network. If it is a :class:`ParamEdge`, its tensor
        (``nn.Parameter``) becomes a parameter of the network. If it has
        :class:`ParamEdges <ParamEdge>`, they become sub-modules of the network.

        Parameters
        ----------
        node : Node or ParamNode
            Node to be added to the network.
        override : bool
            Boolean indicating whether ``node`` should override an existing node
            of the network that has the same name (e.g. when parameterizing a 
            node, the param-node overrides the original node).
        """
        if override:
            prev_node = self.nodes[node._name]
            self._remove_node(prev_node)
        self._assign_node_name(node, node._name, True)

    def _remove_node(self, node: AbstractNode, move_names=True) -> None:
        """
        Removes a node from the network. It just removes the reference to the
        node from the network, as well as the reference to the network that is
        kept by the node. To completely get rid of the node, it should be first
        disconnected from all its neighbours (that belong to the network) and
        then removed. This what :meth:`delete_node` does.

        Parameters
        ----------
        node : AbstractNode
            Node to be removed.
        move_names : bool
            Boolean indicating whether names' enumerations should be decreased
            when removing a node (``True``) or kept as they are (``False``).
            This is useful when several nodes are being modified at once, and
            each resultant node has the same enumeration as the corresponding
            original node.
        """
        node._temp_tensor = node.tensor
        node._tensor_info = None
        node._network = None

        self._unassign_node_name(node, move_names)

        nodes_dict = self._which_dict(node)
        if node._name in nodes_dict:
            if nodes_dict[node._name] == node:
                # If we remove "node_0" from ["node_0", "node_1", "node_2"],
                # "node_1" will become "node_0" when unassigning "node_0" name.
                # Hence, it can happen that nodes_dict[node._name] != node
                del nodes_dict[node._name]

                if node._name in self._memory_nodes:
                    # It can happen that name is not in memory_nodes, when the
                    # node is using the memory of another node
                    del self._memory_nodes[node._name]

    def delete_node(self, node: AbstractNode, move_names=True) -> None:
        """
        Disconnects node from all its neighbours and removes it from the network.

        Parameters
        ----------
        node : Node or ParamNode
            Node to be deleted.
        move_names : bool
            Boolean indicating whether names' enumerations should be decreased
            when removing a node (``True``) or kept as they are (``False``).
            This is useful when several nodes are being modified at once, and
            each resultant node has the same enumeration as the corresponding
            original node.
        """
        node.disconnect()
        self._remove_node(node, move_names)
        del node

    def _update_node_info(self, node: AbstractNode, new_name: Text) -> None:
        """
        Updates a single node's ``tensor_info`` "address" and its corresponding
        address in ``memory_nodes`` and ``inverse_memory``.
        """
        prev_name = node._name
        nodes_dict = self._which_dict(node)

        # For instance, when changing the name of node "node_0", its name
        # will be first unassigned, thus moving the node "node_1" to "node_0"
        # (which would be ``new_name``, and is already in ``nodes_dict``)
        if new_name in nodes_dict:
            aux_node = nodes_dict[new_name]
            aux_node._temp_tensor = aux_node.tensor

        if nodes_dict.get(prev_name) == node:
            nodes_dict[new_name] = nodes_dict.pop(prev_name)
            if node._tensor_info['address'] is not None:
                self._memory_nodes[new_name] = self._memory_nodes.pop(
                    prev_name)
                node._tensor_info['address'] = new_name

                if self._tracing and (prev_name in self._inverse_memory):
                    self._inverse_memory[new_name] = self._inverse_memory.pop(
                        prev_name)

        # This would be the case in the example above, where the original
        # "node_0" will be replaced by the original "node_1", and hence the
        # "node_0" in ``nodes_dict`` will be the original "node_1"
        else:
            nodes_dict[new_name] = node
            self._memory_nodes[new_name] = node._temp_tensor
            node._temp_tensor = None
            node._tensor_info['address'] = new_name

    def _update_node_name(self, node: AbstractNode, new_name: Text) -> None:
        """Updates a single node's name, without taking care of the other names."""
        # Node is ParamNode and tensor is not None
        if isinstance(node.tensor, Parameter):
            delattr(self, 'param_' + node._name)
        for edge in node._edges:
            if edge.is_attached_to(node):
                self._remove_edge(edge)

        self._update_node_info(node, new_name)
        node._name = new_name

        if isinstance(node.tensor, Parameter):
            if not hasattr(self, 'param_' + node.name):
                self.register_parameter(
                    'param_' + node._name, self._memory_nodes[node._name])
            else:
                # Nodes names are never repeated, so it is likely that
                # this case will never occur
                raise ValueError(
                    f'Network already has attribute named {node._name}')

        for edge in node._edges:
            self._add_edge(edge)

    def _assign_node_name(self, node: AbstractNode,
                          name: Text,
                          first_time: bool = False) -> None:
        """
        Assigns a new name to a node in the network. If the node was not previously
        in the network, a new ``tensor_info`` dict is created and the node's tensor
        is stored in the network's memory (``memory_nodes``). If the node was
        in the network (case its name is being changed), the ``tensor_info`` dict
        gets updated.

        In case there are already nodes with the same name in the network, an
        enumeration is added to the new node's name (and to the network's node
        in case there was only one node with the same name).

        * **New node's name**: "my_node"
          **Network's nodes' names**: ["my_node_0", "my_node_1"]
          **Result**: ["my_node_0", "my_node_1", "my_node_2"(new node)]

        * **New node's name**: "my_node"
          **Network's nodes' names**: ["my_node"]
          **Result**: ["my_node_0", "my_node_1"(new node)]

        Also, if the new node's name already had an enumeration, it will be removed.

        * **New node's name**: "my_node_0" -> "my_node" (then apply other rules)
          **Network's nodes' names**: ["my_node"]
          **Result**: ["my_node_0", "my_node_1"(new node)]

        To add a custom enumeration to keep track of the nodes of the network
        in a used-defined way, one may use brackets or parenthesis.

        * **New node's name**: "my_node_(0)"
          **Network's nodes' names**: ["my_node"]
          **Result**: ["my_node", "my_node_(0)"(new node)]

        Parameters
        ----------
        node : Node or ParamNode
            Node whose name will be assigned in the network.
        name : str
            Node's name. If it coincides with a name that already exists in the
            network, the rules explained before will modify the name and assign
            this version as the node's name.
        first_time : bool
            Boolean indicating whether it is the first time a name is assigned
            to ``node``. In this case (``True``), a new ``tensor_info`` dict is
            created for the node.
        """
        non_enum_prev_name = erase_enum(name)

        if not node.is_resultant() and (non_enum_prev_name in self.operations):
            raise ValueError(f'Node\'s name cannot be an operation name: '
                             f'{list(self.operations.keys())}')

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

        # Since node name might change, edges should be removed and
        # added later, so that their names as sub-modules are correct
        for edge in node._edges:
            if edge.is_attached_to(node):
                self._remove_edge(edge)

        if first_time:
            nodes_dict = self._which_dict(node)
            nodes_dict[new_name] = node
            self._memory_nodes[new_name] = node._temp_tensor
            node._tensor_info = {'address': new_name,
                                 'node_ref': None,
                                 'full': True,
                                 'index': None}
            node._temp_tensor = None
            node._network = self
            node._name = new_name
        else:
            self._update_node_info(node, new_name)
            node._name = new_name

        # Node is ParamNode and tensor is not None
        if isinstance(node.tensor, Parameter):
            if not hasattr(self, 'param_' + node._name):
                self.register_parameter(
                    'param_' + node._name, self._memory_nodes[node._name])
            else:
                # Nodes names are never repeated, so it is likely that
                # this case will never occur
                raise ValueError(
                    f'Network already has attribute named {node._name}')

        for edge in node._edges:
            self._add_edge(edge)

    def _unassign_node_name(self, node: AbstractNode, move_names=True):
        """
        Unassigns node's name from the network, thus deleting its tensor as
        a parameter of the network (if node is a ``ParamNode``), and removing
        its edges. Also, if ``move_names`` is set to ``True``, the enumeration
        of all the nodes in the network that have the same name as the given
        node will be decreased by one. If only one node remains, the enumeration
        will be removed.

        * **Node's name**: "my_node_1"
          **Network's nodes' names**: ["my_node_0", "my_node_1"(node), "my_node_2"]
          **Result**: ["my_node_0", "my_node_1"]

        * **Node's name**: "my_node_1"
          **Network's nodes' names**: ["my_node_0", "my_node_1"(node)]
          **Result**: ["my_node"]

        Parameters
        ----------
        node : Node or ParamNode
            Node whose name will be unassigned in the network.
        move_names : bool
            Boolean indicating whether names' enumerations should be decreased
            when removing a node (``True``) or kept as they are (``False``).
            This is useful when several nodes are being modified at once, and
            each resultant node has the same enumeration as the corresponding
            original node.
        """
        # Node is ParamNode and tensor is not None
        if isinstance(node.tensor, Parameter):
            delattr(self, 'param_' + node._name)
        for edge in node._edges:
            if edge.is_attached_to(node):
                self._remove_edge(edge)

        non_enum_prev_name = erase_enum(node.name)
        count = self._repeated_nodes_names[non_enum_prev_name]
        if move_names:
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
            if move_names:
                aux_prev_name = non_enum_prev_name + '_0'
                aux_new_name = non_enum_prev_name
                aux_node = self.nodes[aux_prev_name]
                self._update_node_name(aux_node, aux_new_name)

    def _change_node_name(self, node: AbstractNode, name: Text) -> None:
        """
        Changes the name of a node in the network. To take care of all the other
        names this change might affect, it calls :meth:`TensorNetwork._unassign_node_name`
        and :meth:`TensorNetwork._assign_node_name`.
        """
        if node._network != self:
            raise ValueError('Cannot change the name of a node that does '
                             'not belong to the network')

        if erase_enum(name) != erase_enum(node._name):
            self._unassign_node_name(node)
            self._assign_node_name(node, name)

    def copy(self) -> 'TensorNetwork':
        """Copies the tensor network (via ``copy.deepcopy``)."""
        return copy.deepcopy(self)

    def parameterize(self,
                     set_param: bool = True,
                     override: bool = False) -> 'TensorNetwork':
        """
        Parameterizes all nodes of the network.

        Parameters
        ----------
        set_param : bool
            Boolean indicating whether the tensor network has to be parameterized
            (``True``) or de-parameterized (``False``).
        override : bool
            Boolean indicating whether the tensor network should be parameterized
            in-place (``True``) or copied and then parameterized (``False``).
        """
        if override:
            net = self
        else:
            net = self.copy()

        # TODO: reset always? care with what does parameterize, case uniform?
        if self._resultant_nodes:
            warnings.warn('Resultant nodes will be removed before parameterizing'
                          ' the TN')
            self.reset()

        for node in list(net._leaf_nodes.values()):
            node.parameterize(set_param)

        return net

    def trace(self, example: Optional[Tensor] = None, *args, **kwargs) -> None:
        """
        Traces the tensor network contraction algorithm with two purposes:

        * Create all the intermediate ``resultant`` nodes that result from
          :class:`Operations <Operation>` so that in the next contractions only
          the tensor-like operations have to be computed, thus saving a lot of time.

        * Keep track of the tensors that are used to compute operations, so that
          intermediate results that are not useful any more can be deleted, thus
          saving a lot of memory. This is achieved by constructing an ``inverse_memory``
          that, given a memory address, stores the nodes that use the tensor located
          in that address.

        To trace a tensor network, it is necessary to provide the same arguments
        that would be required in the forward call. In case the tensor network
        is contracted with some input data, an example tensor with batch dimension
        1 and filled with zeros would be enough to trace the contraction.

        Parameters
        ----------
        example : torch.Tensor, optional
            Example tensor used to trace the contraction of the tensor network.
            In case the tensor network is contracted with some input data, an
            example tensor with batch dimension 1 and filled with zeros would
            be enough to trace the contraction.
        args :
            Arguments that might be used in :meth:`contract`.
        kwargs :
            Keyword arguments that might be used in :meth:`contract`.
        """
        self.reset()

        with torch.no_grad():
            self._tracing = True
            self(example, *args, **kwargs)
            self._tracing = False
            self(example, *args, **kwargs)

    def reset(self):
        """
        Resets the tensor network as it was before tracing, contracting or, in
        general, performing any non-in-place :class:`Operation`. Hence, it deletes
        all ``resultant`` and ``virtual`` nodes that are created when performing
        an operation, and resets the ``memory_nodes`` of the network, so that
        each node stores its corresponding tensor. Also, the lists of successors
        of all ``leaf`` and ``data`` nodes are emptied.
        """
        self._seq_ops = []
        self._inverse_memory = dict()

        if self._resultant_nodes or self._virtual_nodes:
            aux_dict = dict()
            aux_dict.update(self._leaf_nodes)
            aux_dict.update(self._resultant_nodes)
            aux_dict.update(self._virtual_nodes)
            for node in aux_dict.values():
                if node._virtual and ('virtual_stack' not in node._name):
                    # Virtual nodes named "virtual_stack" are ParamStackNodes
                    # that result from stacking a collection of ParamNodes
                    # This condition is satisfied by the rest of virtual nodes
                    # (e.g. "virtual_feature", "virtual_n_features")
                    continue

                node._successors = dict()

                node_ref = node._tensor_info['node_ref']
                if node_ref is not None:
                    if node_ref._virtual and ('virtual_uniform' in node_ref._name):
                        # Virtual nodes named "virtual_uniform" are ParamNodes
                        # whose tensor is shared accross all the nodes in a
                        # uniform tensor network
                        continue

                # Store tensor as temporary
                node._temp_tensor = node.tensor
                node._tensor_info['address'] = node._name
                node._tensor_info['node_ref'] = None
                node._tensor_info['full'] = True
                node._tensor_info['index'] = None

                if isinstance(node._temp_tensor, Parameter):
                    if hasattr(self, 'param_' + node._name):
                        delattr(self, 'param_' + node._name)

                if node._name not in self._memory_nodes:
                    self._memory_nodes[node._name] = None

                # Set tensor and save it in ``memory_nodes``
                if node._temp_tensor is not None:
                    node._unrestricted_set_tensor(node._temp_tensor)
                    node._temp_tensor = None

            for node in list(self._data_nodes.values()):
                node._successors = dict()

            aux_dict = dict()
            aux_dict.update(self._resultant_nodes)
            aux_dict.update(self._virtual_nodes)
            for node in list(aux_dict.values()):
                if node._virtual and ('virtual_stack' not in node._name):
                    # This condition is satisfied by the rest of virtual nodes
                    # (e.g. "virtual_feature", "virtual_n_features")
                    continue
                self.delete_node(node, False)

    def set_data_nodes(self,
                       input_edges: Union[List[int], List[Edge]],
                       num_batch_edges: int) -> None:
        """
        Creates ``data`` nodes with as many batch edges as ``num_batch_edges``
        and one feature edge, and connects each of these feature edges to an
        edge from the list ``input_edges`` (following the provided order).

        If all the data nodes have the same shape, a ``virtual`` node will
        contain all the tensors stacked in one, what will save some memory
        and time in computations.

        This method can be overriden in subclasses so that it is specified in
        its implementation to which edges of the network the data nodes should
        be connected. In this case, there is no need to call ``set_data_nodes``
        explicitly during training, since it will be done in the :meth:`forward`
        call. Otherwise, it should be called before starting training.

        Parameters
        ----------
        input_edges : list[int] or list[Edge]
            List of edges (or indices of :meth:`edges` if given as ``int``) to
            which the ``data`` nodes' feature edges will be connected.
        num_batch_edges : int
            Number of batch edges in the ``data`` nodes.
        """
        if input_edges == []:
            raise ValueError(
                '`input_edges` is empty.'
                ' Cannot set data nodes if no edges are provided')
        if self._data_nodes:
            raise ValueError(
                'Tensor network already has data nodes. These should be unset '
                'in order to set new ones')

        # "stack_data_memory" is only created if all the input edges
        # have the same dimension
        same_dim = True
        for i in range(len(input_edges) - 1):
            if input_edges[i].size() != input_edges[i + 1].size():
                same_dim = False
                break

        if same_dim:
            if 'stack_data_memory' not in self._virtual_nodes:
                # If ``same_dim``, all input_edges have the same feature dimension
                stack_node = Node(shape=(len(input_edges),
                                         *([1]*num_batch_edges),
                                         input_edges[0].size()),
                                  axes_names=('n_features',
                                              *(['batch']*num_batch_edges),
                                              'feature'),
                                  name='stack_data_memory',
                                  network=self,
                                  virtual=True)
            else:
                stack_node = self._virtual_nodes['stack_data_memory']

        data_nodes = []
        for i, edge in enumerate(input_edges):
            if isinstance(edge, int):
                edge = self[edge]
            elif isinstance(edge, Edge):
                if edge not in self._edges:
                    raise ValueError(
                        f'Edge {edge!r} should be a dangling edge of the '
                        'Tensor Network')
            else:
                raise TypeError(
                    '`input_edges` should be list[int] or list[Edge] type')

            node = Node(shape=(*([1]*num_batch_edges), edge.size()),
                        axes_names=(*(['batch']*num_batch_edges), 'feature'),
                        name='data',
                        network=self,
                        data=True)
            node['feature'] ^ edge
            data_nodes.append(node)

        if same_dim:
            for i, node in enumerate(data_nodes):
                del self._memory_nodes[node._tensor_info['address']]
                node._tensor_info['address'] = None
                node._tensor_info['node_ref'] = stack_node
                node._tensor_info['full'] = False
                node._tensor_info['index'] = i

    def unset_data_nodes(self) -> None:
        """
        Deletes all ``data`` nodes (and ``virtual`` ancillary nodes in case it
        is necessary).
        """
        if self._data_nodes:
            for node in list(self._data_nodes.values()):
                self.delete_node(node)
            self._data_nodes = dict()

            if 'stack_data_memory' in self._virtual_nodes:
                self.delete_node(self._virtual_nodes['stack_data_memory'])

    def add_data(self, data: Union[Tensor, Sequence[Tensor]]) -> None:
        """
        Adds data tensor(s) to ``data`` nodes, that is, changes their tensors
        by new data tensors when a new batch is provided.

        Parameters
        ----------
        data : torch.Tensor or list[torch.Tensor]
            If all data nodes have the same shape, thus having its tensor stored
            in "stack_data_memory", ``data`` should be a tensor of shape
            n_features x batch_size_{0} x ... x batch_size_{n} x feature_size.
            Otherwise, it should be a list with n_features elements, each of them
            being a tensor with shape batch_size_{0} x ... x batch_size_{n} x feature_size.
        """

        stack_node = self._virtual_nodes.get('stack_data_memory')

        if stack_node is not None:
            stack_node.tensor = data
        elif self._data_nodes:
            for i, data_node in enumerate(list(self._data_nodes.values())):
                data_node.tensor = data[i]
        else:
            raise ValueError('Cannot add data if no data nodes are set')

    def contract(self) -> AbstractNode:
        """
        Contracts the whole tensor network returning a single ``Node``. This
        method is not implemented and should be overriden in subclasses of
        :class:`TensorNetwork`.
        """
        # Custom, optimized contraction methods should be defined for each new
        # subclass of TensorNetwork
        raise NotImplementedError(
            'Contraction methods not implemented for generic TensorNetwork class')

    def forward(self,
                data: Optional[Union[Tensor, Sequence[Tensor]]] = None,
                *args, **kwargs) -> Tensor:
        r"""
        Contract Tensor Network with input data with shape batch x n_features x feature.

        Overrides the ``forward`` method of **PyTorch** ``nn.Module``. Sets data
        nodes automatically whenever ``set_data_nodes`` is overriden, adds data
        tensor(s) to these nodes, and contracts the whole network according to
        :meth:`contract`, returning a single ``torch.Tensor``.

        It can be called using the ``__call__`` operator ``()``.

        Parameters
        ----------
        data : torch.Tensor or list[torch.Tensor], optional
            If all data nodes have the same shape, thus having its tensor stored
            in "stack_data_memory", ``data`` should be a tensor of shape

            .. math::
                n_{features} \times batch\_size_{0} \times ... \times
                batch\_size_{n} \times feature\_size.

            Otherwise, it should be a list with :math:`n_{features}` elements,
            each of them being a tensor with shape

            .. math::
                batch\_size_{0} \times ... \times batch\_size_{n} \times
                feature\_size.

            Also, it is not necessary that the network has ``data`` nodes, thus
            ``None`` is also valid.
        args :
            Arguments that might be used in :meth:`contract`.
        kwargs :
            Keyword arguments that might be used in :meth:`contract`.
        """
        if data is not None:
            if not self._data_nodes:
                self.set_data_nodes()
            self.add_data(data=data)

        if not self._resultant_nodes:
            output = self.contract(*args, **kwargs)
            return output.tensor

        else:
            for op in self._seq_ops:
                output = self.operations[op[0]](**op[1])

            if not isinstance(output, Node):
                if (op[0] == 'unbind') and (len(output) == 1):
                    output = output[0]
                else:
                    raise ValueError('The last operation should be the one '
                                     'returning a single resultant node')

            return output.tensor

    def __getitem__(self, key: Union[int, Text]) -> Union[Edge, AbstractNode]:
        if isinstance(key, int):
            return self._edges[key]
        elif isinstance(key, Text):
            try:
                return self.nodes[key]
            except Exception:
                raise KeyError(
                    f'Tensor network {self!s} does not have any node with '
                    f'name {key}')
        else:
            raise TypeError('`key` should be int or str type')

    def __str__(self) -> Text:
        return self.name

    def __repr__(self) -> Text:
        return f'{self.__class__.__name__}(\n ' \
               f'\tname: {self.name}\n' \
               f'\tnodes: \n{tab_string(print_list(list(self.nodes.keys())), 2)}\n' \
               f'\tedges:\n{tab_string(print_list(self._edges), 2)})'
