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
        *ParamStackNode
        *AbstractStackEdge:
            +StackEdge
            +ParamStackEdge
    
    Class for successors:        
        *Successor

    Class for Tensor Networks:
        *TensorNetwork
        
    Edge operations:
        *connect
        *connect_stack
        *disconnect
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


################################################
#                    AXIS                      #
################################################
class Axis:
    """
    The axes are the objects that stick edges to nodes. Every :class:`node <AbstractNode>`
    has a list of :math:`N` axes, each corresponding to one edge; and every axis
    stores information that helps accessing that edge, such as its :attr:`name`
    and :attr:`num` (index). Also, the axis keeps track of the :meth:`batch <is_batch>`
    and :meth:`node1 <is_node1>` attributes:
    
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
      connection is performed in the following order:
      ::
        new_edge = nodeA[edgeA] ^ nodeB[edgeB]
        
      Then ``nodeA`` will be the `node1` of ``new_edge`` and ``nodeB``, the `node2`.
      Hence, to access one of the nodes from ``new_edge`` one needs to know if it is
      `node1` or `node2`.
      
    Even though we can create Axis instances, that will not be usually the case,
    since axes are automatically created when instantiating a new :class:`node <AbstractNode>`.

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
        if not check_name_style(name, 'axis'):
            raise ValueError(
                '`name` cannot contain blank spaces or special characters '
                'since it is intended to be used as name of submodules')

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

    # properties
    @property
    def num(self) -> int:
        """Index in the node's axes list."""
        return self._num

    @property
    def name(self) -> Text:
        """
        Axis name, used to access edges by name of the axis. It cannot contain
        blank spaces or special characters since it is intended to be used as
        name of submodules.
        """
        return self._name

    @name.setter
    def name(self, name: Text) -> None:
        """
        Set axis name. Should not contain blank spaces or special characters
        since it is intended to be used as name of submodules.
        """
        if not isinstance(name, str):
            raise TypeError('`name` should be str type')
        if not check_name_style(name, 'axis'):
            raise ValueError(
                '`name` cannot contain blank spaces or special characters '
                'since it is intended to be used as name of submodules')
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

    # methods
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


################################################
#                   NODES                      #
################################################
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
                 shape: Shape,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 leaf: bool = True,
                 data: bool = False,
                 virtual: bool = False) -> None:

        super().__init__()

        # check shape
        if shape is not None:
            if not isinstance(shape, (tuple, list, Size)):
                raise TypeError(
                    '`shape` should be tuple[int, ...], list[int, ...] or '
                    'torch.Size type')
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
                raise TypeError(
                    '`axes_names` should be tuple[str, ...] or list[str, ...] type')
            if len(axes_names) != len(shape):
                raise ValueError(
                    '`axes_names` length should match `shape` length')
            else:
                axes_names = enum_repeated_names(axes_names)
                axes = [Axis(num=i, name=name, node=self)
                        for i, name in enumerate(axes_names)]

        # check name
        if name is None:
            name = self.__class__.__name__.lower()
        elif not isinstance(name, str):
            raise TypeError('`name` should be str type')
        elif not check_name_style(name, 'node'):
            raise ValueError('Names cannot contain blank spaces')

        # check network
        if network is not None:
            if not isinstance(network, TensorNetwork):
                raise TypeError('`network` should be TensorNetwork type')
        else:
            network = TensorNetwork()

        # check leaf and data
        if data and virtual:
            raise ValueError(
                '`data` and `virtual` arguments cannot be both True')
        elif data or virtual:
            leaf = False

        # Set attributes
        self._tensor_info = None
        self._temp_tensor = None
        self._shape = shape
        self._axes = axes
        self._edges = []
        self._name = name
        self._network = network
        self._leaf = leaf
        self._data = data
        self._virtual = virtual
        self._successors = dict()

    # ----------
    # Properties
    # ----------
    @property
    def tensor(self) -> Optional[Union[Tensor, Parameter]]:
        """
        Node's tensor. It can be a ``torch.Tensor``, ``torch.nn.Parameter`` or
        None if the node is empty.
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
    def edges(self) -> List['AbstractEdge']:
        """List of node's :class:`edges <AbstractEdge>`."""
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
    @staticmethod
    @abstractmethod
    def _set_tensor_format(tensor: Tensor) -> Union[Tensor, Parameter]:
        pass

    @abstractmethod
    def parameterize(self, set_param: bool) -> 'AbstractNode':
        pass

    @abstractmethod
    def copy(self) -> 'AbstractNode':
        pass

    # -------
    # Methods
    # -------
    def is_leaf(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``leaf`` node. These are
        the nodes that form the :class:`TensorNetwork`. Usually, these will be
        the `trainable` nodes.
        """
        return self._leaf

    def is_data(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``data`` node. These are
        the nodes where input data tensors will be put.
        """
        return self._data

    def is_virtual(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``virtual`` node. These
        are a sort of `hidden` nodes that can be used, for instance, to store
        the information of other ``leaf`` or ``data`` nodes more efficiently
        (e.g. :class:`Uniform MPS <UMPS>` uses a unique ``virtual`` node to
        store the tensor used by all the nodes in the network).
        """
        return self._virtual

    def is_non_leaf(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``non_leaf`` node. These
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

    def dim(self, axis: Optional[Ax] = None) -> Union[Size, int]:
        """
        Returns the dimensions of the node's tensor. If ``axis`` is specified,
        returns the dimension of that edge; otherwise returns the dimensions of
        all edges.
        
        See also :meth:`Edge.dim` and :meth:`ParamEdge.dim`.

        Parameters
        ----------
        axis : int, str or Axis, optional
            Axis for which to retrieve the dimension.

        Returns
        -------
        int or torch.Size
        """
        if axis is None:
            return Size(map(lambda edge: edge.dim(), self._edges))
        axis_num = self.get_axis_num(axis)
        return self._edges[axis_num].dim()

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
            self._shape = tuple(aux_shape)

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
                self._shape = tuple(aux_shape)
                self.tensor = tensor[index]

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
                self._shape = tuple(aux_shape)
                self.tensor = nn.functional.pad(tensor, pad)

    def get_axis(self, axis: Ax) -> 'AbstractEdge':
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
                  edge: 'AbstractEdge',
                  axis: Ax,
                  node1: bool = True) -> None:
        """
        Adds an edge to the specified axis of the node.

        Parameters
        ----------
        edge : AbstractEdge
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
        
    def _reattach_edges(self, override: bool = False) -> None:
        """
        Substitutes current edges by copies of them that are attached to the node.
        It can happen that an edge is not attached to the node if it is the result
        of an :class:`Operation` and, hence, it inherits edges from the operands.
        In that case, the new copied edges will be attached to the resultant node,
        replacing each previous `node1` or `node2` with it (according to the `node1`
        attribute of each axis).
        
        Used for inplace operations like :func:`permute_` or :func`split_` and
        to :meth:`Node.parameterize`.

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
        
    def get_edge(self, axis: Ax) -> 'AbstractEdge':
        """
        Returns :class:`AbstractEdge` given the :class:`Axis` (or its ``name``
        or ``num``) where it is attached to the node.
        """
        axis_num = self.get_axis_num(axis)
        return self._edges[axis_num]
    
    def in_which_axis(self, edge: 'AbstractEdge') -> Axis:
        """
        Returns :class:`Axis` given the :class:`AbstractEdge` that is attached
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

    def param_edges(self,
                    set_param: Optional[bool] = None,
                    sizes: Optional[Sequence[int]] = None) -> Optional[bool]:
        r"""
        Returns ``param_edges`` attribute or changes it if ``set_param`` is provided,
        by parameterizing or de-parameterizing all edges (see also :meth:`Edge.parameterize`
        or :meth:`ParamEdge.parameterize`).

        Parameters
        ----------
        set_param : bool, optional
            Boolean indicating whether edges have to be parameterized (``True``)
            or de-parameterized (``False``).
        sizes : list[int] or tuple[int], optional
            Sizes used to expand or shrink the node's shape if desired. If edges
            are parameterized, their dimensions will be :math:`\text{dim} =
            \min(\text{prev_size}, \text{new_size})`

        Returns
        -------
        bool or None
            If ``set_param`` is not provided, returns ``True`` if all edges are
            :class:`ParamEdges <ParamEdge>` or ``False`` if all edges are
            :class:`Edges <Edge>`. If there are some of each type, or ``set_param``
            is provided, returns None.

        Raises
        ------
        ValueError
            If ``sizes`` are provided but do not match the number of edges of
            the node.
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
                    sizes = self._shape
                elif len(sizes) != len(self._axes):
                    raise ValueError(
                        '`sizes` length should match the number of node\'s axes')
                for i, edge in enumerate(self._edges):
                    edge.parameterize(True, size=sizes[i])
            else:
                for param_edge in self._edges:
                    param_edge.parameterize(False)

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

    def _crop_tensor(self, tensor: Tensor, allow_diff_shape: bool = False) -> Tensor:
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
        allow_diff_shape : bool
            Boolean indicating whether different sizes are allowed for all axes
            (``True``), rather than just for batch axes (``Falsee``). If ``True``,
            any tensor could be set in the node, and the node's shape would change
            accordingly.
            
        Returns
        -------
        torch.Tensor
        """
        if len(tensor.shape) == self.rank:
            index = []
            for i, dim in enumerate(tensor.shape):
                edge = self.get_edge(i)

                if edge.is_batch() or (dim == edge.size()) or allow_diff_shape:
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

    def _unrestricted_set_tensor(self,
                                 tensor: Optional[Tensor] = None,
                                 allow_diff_shape: bool = False,
                                 init_method: Optional[Text] = 'zeros',
                                 device: Optional[torch.device] = None,
                                 **kwargs: float) -> None:
        """
        Sets a new node's tensor or creates one with :meth:`make_tensor` and sets
        it. Before setting it, it is casted to the correct type, so that a
        ``torch.Tensor`` can be turned into a ``nn.Parameter`` when setting it
        in :class:`ParamNodes <ParamNode`. This can be used in any node, even in
        non-leaf nodes.

        Parameters
        ----------
        tensor : torch.Tensor, optional
            Tensor to be set in the node. If None, and `init_method` is provided,
            the tensor is created with :meth:`make_tensor`. Otherwise, a None is
            set as node's tensor.
        allow_diff_shape : bool, optional
            Boolean indicating whether different sizes are allowed for all axes
            (``True``), rather than just for batch axes (``Falsee``). If ``True``,
            any tensor could be set in the node, and the node's shape would change
            accordingly.
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
                tensor = self._crop_tensor(tensor, allow_diff_shape)
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
        ``non-leaf`` nodes, their tensors come from the result of operations on
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
            the tensor is created with :meth:`make_tensor`. Otherwise, a None is
            set as node's tensor.
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
        if self._leaf and not self._network._automemory:
            self._unrestricted_set_tensor(
                tensor=tensor, init_method=init_method, device=device, **kwargs)

            for edge, size in zip(self._edges, self._shape):
                edge._size = size
        else:
            raise ValueError('Node\'s tensor can only be changed if it is a leaf'
                             ' tensor and the network is not in automemory mode')

    def unset_tensor(self) -> None:
        """Replaces node's tensor with None."""
        if self._leaf and not self._network._automemory:
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
                # it is taking the tensor information (node_ref) are non-leaf or
                # data nodes (including virtual node that stores stack data tensor)
                erase = True
                for node in check_nodes:
                    erase &= node.is_non_leaf() or \
                        node.is_data() or \
                        (node.is_virtual() and
                         node._name == 'stack_data_memory')

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
    def __getitem__(self, key: slice) -> List['AbstractEdge']:
        pass

    @overload
    def __getitem__(self, key: Ax) -> 'AbstractEdge':
        pass

    def __getitem__(self, key: Union[slice, Ax]) -> Union[List['AbstractEdge'],
                                                          'AbstractEdge']:
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
    leaf : bool, optional
        Boolean indicating if the node is a ``leaf`` node. If a node is neither
        ``leaf`` nor ``data`` nor ``virtual``, it is ``non-leaf``.
    data : bool, optional
        Boolean indicating if the node is a ``data`` node.
    virtual : bool, optional
        Boolean indicating if the node is a ``virtual`` node.
    override_node : bool, optional
        Boolean indicating whether the node should override (``True``) another
        node in the network that has the same name (e.g. if a node is parameterized,
        it would be required that a new :class:`ParamNode` replaces the non-parameterized
        node in the network).
    param_edges : bool, optional
        Boolean indicating whether all node's edges should be :class:`ParamEdges
        <ParamEdge>` (``True``) or not (``False``).
    tensor : torch.Tensor, optional
        Tensor that is to be stored in the node. If None, ``shape`` and ``init_method``
        will be required.
    edges : list[AbstractEdge], optional
        List of edges that are to be attached to the node. This can be used in
        case the node inherits the edges from other node(s), like in :class:`Operations
        <Operation>`.
    override_edges : bool, optional
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

    def __init__(self,
                 shape: Optional[Shape] = None,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 leaf: bool = True,
                 data: bool = False,
                 virtual: bool = False,
                 override_node: bool = False,
                 param_edges: bool = False,
                 tensor: Optional[Tensor] = None,
                 edges: Optional[List['AbstractEdge']] = None,
                 override_edges: bool = False,
                 node1_list: Optional[List[bool]] = None,
                 init_method: Optional[Text] = None,
                 device: Optional[torch.device] = None,
                 **kwargs: float) -> None:

        # shape and tensor
        if (shape is None) == (tensor is None):
            if shape is None:
                raise ValueError('One of `shape` or `tensor` must be provided')
            elif tensor.shape == shape:
                shape = None
            else:
                raise ValueError('If both `shape` or `tensor` are given, '
                                 '`tensor`\'s shape should be equal to `shape`')
        if shape is not None:
            super().__init__(shape=shape,
                             axes_names=axes_names,
                             name=name,
                             network=network,
                             leaf=leaf,
                             data=data,
                             virtual=virtual)
        else:
            super().__init__(shape=tensor.shape,
                             axes_names=axes_names,
                             name=name,
                             network=network,
                             leaf=leaf,
                             data=data,
                             virtual=virtual)

        # edges
        if edges is None:
            self._edges = [self._make_edge(ax, param_edges) for ax in self._axes]
        else:
            if node1_list is None:
                raise ValueError(
                    'If `edges` are provided, `node1_list` should also be provided')
            for i, axis in enumerate(self._axes):
                if not isinstance(node1_list[i], bool):
                    raise TypeError('`node1_list` should be list[bool] type')
                axis._node1 = node1_list[i]
            self._edges = edges[:]
            if self._leaf and not self._network._automemory:
                self._reattach_edges(override=override_edges)

        # network
        self._network._add_node(self, override=override_node)

        if shape is not None:
            if init_method is not None:
                self._unrestricted_set_tensor(
                    init_method=init_method, device=device, **kwargs)
        else:
            self._unrestricted_set_tensor(tensor=tensor)

    # -------
    # Methods
    # -------
    @staticmethod
    def _set_tensor_format(tensor: Tensor) -> Tensor:
        """Returns a torch.Tensor if input tensor is given as nn.Parameter."""
        if isinstance(tensor, Parameter):
            return tensor.detach()
        return tensor
    
    def _make_edge(self, axis: Axis, param_edges: bool) -> Union['Edge', 'ParamEdge']:
        """Makes Edges or ParamEdges depending on the ``param_edges`` attribute"""
        if param_edges:
            return ParamEdge(node1=self, axis1=axis)
        return Edge(node1=self, axis1=axis)

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
                                 param_edges=self.param_edges(),
                                 tensor=self.tensor,
                                 edges=self._edges,
                                 override_edges=True,
                                 node1_list=self.is_node1())
            return new_node
        else:
            return self

    def copy(self) -> 'Node':
        """
        Returns a copy of the node. That is, returns a node whose tensor is a copy
        of the original, whose edges are directly inherited (these are not copies,
        but the exact same edges) and whose name is extended with the prefix ``"copy_"``.

        Returns
        -------
        Node
        """
        new_node = Node(shape=self._shape,
                        axes_names=self.axes_names,
                        name='copy_' + self._name,
                        network=self._network,
                        param_edges=self.param_edges(),
                        tensor=self.tensor,
                        edges=self._edges,
                        node1_list=self.is_node1())
        return new_node


class ParamNode(Node):
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
    
    To see how to initialize `` ParamNodes``, see :class:`Node`.
    
    For a complete list of properties and methods, see also :class:`AbstractNode`.
    """

    def __init__(self,
                 shape: Optional[Shape] = None,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 leaf: bool = True,
                 data: bool = False,
                 virtual: bool = False,
                 override_node: bool = False,
                 param_edges: bool = False,
                 tensor: Optional[Tensor] = None,
                 edges: Optional[List['AbstractEdge']] = None,
                 override_edges: bool = False,
                 node1_list: Optional[List[bool]] = None,
                 init_method: Optional[Text] = None,
                 device: Optional[torch.device] = None,
                 **kwargs: float) -> None:

        # data
        if data:
            raise ValueError('ParamNode cannot be a data node')

        # leaf
        if not leaf:
            raise ValueError(
                'ParamNode is always a leaf node. Cannot set leaf to False')

        super().__init__(shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         leaf=leaf,
                         data=data,
                         virtual=virtual,
                         override_node=override_node,
                         param_edges=param_edges,
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
                            param_edges=self.param_edges(),
                            tensor=self.tensor,
                            edges=self._edges,
                            override_edges=True,
                            node1_list=self.is_node1())
            return new_node
        else:
            return self

    def copy(self) -> 'ParamNode':
        """
        Returns a copy of the param-node. That is, returns a param-node whose
        tensor is a copy of the original, whose edges are directly inherited
        (these are not copies, but the exact same edges) and whose name is
        extended with the prefix ``"copy_"``.

        Returns
        -------
        ParamNode
        """
        new_node = ParamNode(shape=self.shape,
                             axes_names=self.axes_names,
                             name='copy_' + self._name,
                             network=self._network,
                             param_edges=self.param_edges(),
                             tensor=self.tensor,
                             edges=self._edges,
                             node1_list=self.is_node1())
        return new_node


################################################
#                   EDGES                      #
################################################

class AbstractEdge(ABC):
    """
    Abstract class for all types of edges. Defines what an edge is and some of its
    properties and methods. Since it is an abstract class, cannot be instantiated.

    An edge is nothing more than an object that wraps references to the nodes it
    connects. Thus it stores information like the nodes it connects, the corresponding
    nodes' axes it is attached to, whether it is dangling or batch, its size and
    dim, etc.
    
    Above all, its importance lies in that edges enable to connect nodes, forming
    any possible graph, and perform easily :class:`Operations <Operation>` like
    contracting and splitting nodes.
    
    Furthermore, edges have specific operations like :meth:`contract_` or :meth:`svd_`
    (and its variations) that allow inplace modification of the :class:`TensorNetwork`.
    
    Refer to the subclasses of ``AbstractEdge`` to see how to instantiate edges:
    
    * :class:`Edge`
    
    * :class:`ParamEdge`
    
    * :class:`StackEdge`
    
    * :class:`ParamStackEdge`
    

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
        self._size = node1._shape[axis1._num]

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
        return self._nodes

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
    def connect(self, other: 'AbstractEdge') -> 'AbstractEdge':
        """
        Connect two edges
        """
        pass

    @abstractmethod
    def disconnect(self) -> List['AbstractEdge']:
        """
        Disconnect one edge (from itself)
        """
        pass

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
        return self._size

    def __xor__(self, other: 'AbstractEdge') -> 'AbstractEdge':
        return self.connect(other)

    def __or__(self, other: 'AbstractEdge') -> List['AbstractEdge']:
        if other == self:
            return self.disconnect()
        else:
            raise ValueError('Cannot disconnect one edge from another, different one. '
                             'Edge should be disconnected from itself')

    def __str__(self) -> Text:
        return self.name

    def __repr__(self) -> Text:
        if self.is_batch():
            return f'{self.__class__.__name__}( {self.name} )  (Batch Edge)'
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
        self._size = size
        if not self.is_dangling():
            self.node2._change_axis_size(self.axis2, size)
        self.node1._change_axis_size(self.axis1, size)

    def parameterize(self,
                     set_param: bool = True,
                     size: Optional[int] = None) -> Union['Edge', 'ParamEdge']:
        """
        Returns ``param_edges`` attribute or changes it if ``set_param`` is provided.

        Returns
        -------
        Returns True if all edges are parametric edges, False if all edges are
        non-parametric edges, and None if there are some edges of each type
        

        Parameters
        ----------
        set_param : bool, optional
            Boolean indicating whether edges have to be parameterized (``True``)
            or de-parameterized (``False``).
        sizes : list[int] or tuple[int], optional
            Sizes used to expand or shrink the node's shape if desired. By default,
            if an :class:`Edge` is parameterized, its corresponding ``size`` will
            match its ``dim``. However, if ``sizes`` are provided, 

        Returns
        -------
        _type_
            _description_
        """
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
    def connect(self, other: 'Edge') -> 'Edge':
        pass

    @overload
    def connect(self, other: 'ParamEdge') -> 'ParamEdge':
        pass

    def connect(self, other: Union['Edge', 'ParamEdge']) -> Union['Edge', 'ParamEdge']:
        return connect(self, other)

    def disconnect(self) -> Tuple['Edge', 'Edge']:
        return disconnect(self)


EdgeParameter = Union[int, float, Parameter]
_DEFAULT_SHIFT = -0.5
_DEFAULT_SLOPE = 1.


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

        axis1, axis2 = self._axes[0], self._axes[1]

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
            shift, slope = self.compute_parameters(node1.size(axis1), node1.size(axis1))
            
            # if shift is None:
            #     shift = _DEFAULT_SHIFT
            # if slope is None:
            #     slope = _DEFAULT_SLOPE

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
        # shift = (size - dim) - 0.5
        shift = dim - 0.5
        slope = _DEFAULT_SLOPE
        return shift, slope

    def set_parameters(self,
                       shift: Optional[EdgeParameter] = None,
                       slope: Optional[EdgeParameter] = None):
        """
        Set both parameters, update them and set the new matrix
        """
        device = None
        if self.node1.tensor is not None:
            device = self.node1.tensor.device
        
        if shift is not None:
            if isinstance(shift, (int, float)):
                shift = float(shift)
                if device is None:
                    self._shift = Parameter(torch.tensor(shift))
                else:
                    self._shift = Parameter(torch.tensor(shift).to(device))
                self._prev_shift = _DEFAULT_SHIFT
            elif isinstance(shift, Parameter):
                self._shift = shift
                self._prev_shift = shift.item()
            else:
                raise TypeError(
                    '`shift` should be int, float or Parameter type')
        if slope is not None:
            if isinstance(slope, (int, float)):
                slope = float(slope)
                if device is None:
                    self._slope = Parameter(torch.tensor(slope))
                else:
                    self._slope = Parameter(torch.tensor(slope).to(device))
                self._prev_slope = slope
            elif isinstance(slope, Parameter):
                # TODO: eligible device
                self._slope = slope
                self._prev_slope = slope.item()
            else:
                raise TypeError(
                    '`slope` should be int, float or Parameter type')
        
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
        matrix = torch.zeros((self.size(), self.size()),
                             device=self.shift.device)
        i = torch.arange(self.size(), device=self.shift.device)
        # matrix[(i, i)] = self.sigmoid(self.slope * (i - self.shift))
        matrix[(i, i)] = self.sigmoid(self.slope * (self.shift - i))
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
            warnings.warn(
                f'Dimension of edge {self!r} is not greater than zero')
        self._dim = dim

    def dim(self) -> int:
        """
        Here, `dimension` is not the same as `size`. The ``dim`` and ``size`` in
        a certain axis can be equal if the edge attached to that axis is not parametric
        (e.g. :class:`Edge`). In the case of parametric edges (e.g. :class:`ParamEdge`),
        `bond dimensions` can be learned, meaning that, although the tensor's shape
        can be fixed through the whole training process, what is learned is an
        `effective` dimension
        
        Similar to `size`, but if a ParamEdge is attached to an axis,
        it is returned its dimension (number of 1's in the diagonal of
        the matrix) rather than its total size (number of 1's and 0's
        in the diagonal of the matrix)

        Returns
        -------
        int
            _description_
        """
        return self._dim

    def change_dim(self, dim: Optional[int] = None) -> None:
        if dim != self.dim():
            shift, slope = self.compute_parameters(self.size(), dim)
            self.set_parameters(shift, slope)

    def change_size(self, size: int) -> None:
        if not isinstance(size, int):
            TypeError('`size` should be int type')
        self._size = size
        shift, slope = self.compute_parameters(size, min(size, self.dim()))
        device = self._shift.device

        self.set_parameters(shift, slope)
        self.to(device)

        if not self.is_dangling():
            self.node2._change_axis_size(self.axis2, size)
        self.node1._change_axis_size(self.axis1, size)

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
                             shift=self.shift, slope=self.slope,
                             node2=self.node2, axis2=self.axis2)
        return new_edge

    @overload
    def connect(self, other: Edge) -> 'ParamEdge':
        pass

    @overload
    def connect(self, other: 'ParamEdge') -> 'ParamEdge':
        pass

    def connect(self, other: Union['Edge', 'ParamEdge']) -> 'ParamEdge':
        return connect(self, other)

    def disconnect(self) -> Tuple['ParamEdge', 'ParamEdge']:
        return disconnect(self)


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
                 nodes: Optional[Sequence[AbstractNode]] = None,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 override_node: bool = False,
                 tensor: Optional[Tensor] = None,
                 edges: Optional[List['AbstractEdge']] = None,
                 node1_list: Optional[List[bool]] = None) -> None:

        if nodes is not None:

            if not isinstance(nodes, (list, tuple)):
                raise TypeError('`nodes` should be a list or tuple of nodes')

            for node in nodes:
                if isinstance(node, (StackNode, ParamStackNode)):
                    raise TypeError(
                        'Cannot create a stack using (Param)StackNode\'s')

            # TODO: Y en la misma TN todos
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
                if nodes[i].network != nodes[i + 1].network:
                    raise ValueError(
                        'Stacked nodes must all be in the same network')
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
                    if axis._name not in edges_dict:
                        edges_dict[axis._name] = [edge]
                        node1_lists_dict[axis._name] = [axis._node1]
                    else:
                        edges_dict[axis._name].append(edge)
                        node1_lists_dict[axis._name].append(axis._node1)

            self._edges_dict = edges_dict
            self._node1_lists_dict = node1_lists_dict
            # self.nodes = nodes

            # stacked_tensor = torch.stack([node.tensor for node in nodes])
            if tensor is None:
                # TODO: not sure if this is necessary
                tensor = stack_unequal_tensors([node.tensor for node in nodes])
            super().__init__(axes_names=['stack'] + nodes[0].axes_names,
                             name=name,
                             network=nodes[0]._network,
                             leaf=False,
                             override_node=override_node,
                             tensor=tensor)

        else:
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
            # self.nodes = nodes

            super().__init__(axes_names=axes_names,
                             name=name,
                             network=network,
                             leaf=False,
                             override_node=override_node,
                             tensor=tensor,
                             edges=edges,
                             node1_list=node1_list)

    @property
    def edges_dict(self) -> Dict[Text, List[AbstractEdge]]:
        return self._edges_dict

    @property
    def node1_lists_dict(self) -> Dict[Text, List[bool]]:
        return self._node1_lists_dict

    def _make_edge(self, axis: Axis, param_edges: bool) -> Union['Edge', 'ParamEdge']:
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


class ParamStackNode(ParamNode):
    """
    Class for parametric stacked nodes. This is a node that stores the information
    of a list of parametric nodes that are stacked in order to perform some operation
    """

    def __init__(self,
                 nodes: Sequence[AbstractNode],
                 name: Optional[Text] = None,
                 virtual: bool = False,
                 override_node: bool = False,
                 tensor: Optional[Tensor] = None) -> None:

        if not isinstance(nodes, (list, tuple)):
            raise TypeError('`nodes` should be a list or tuple of nodes')

        for node in nodes:
            if isinstance(node, (StackNode, ParamStackNode)):
                raise TypeError(
                    'Cannot create a stack using (Param)StackNode\'s')

        # TODO: Y en la misma TN todos
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
            if nodes[i].network != nodes[i + 1].network:
                raise ValueError(
                    'Stacked nodes must all be in the same network')
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
                if axis._name not in edges_dict:
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
            tensor = stack_unequal_tensors([node.tensor for node in nodes])
        super().__init__(axes_names=['stack'] + nodes[0].axes_names,
                         name=name,
                         network=nodes[0]._network,
                         virtual=virtual,
                         #  leaf=False,
                         override_node=override_node,
                         tensor=tensor)

    @property
    def edges_dict(self) -> Dict[Text, List[AbstractEdge]]:
        return self._edges_dict

    @property
    def node1_lists_dict(self) -> Dict[Text, List[bool]]:
        return self._node1_lists_dict

    def _make_edge(self, axis: Axis, param_edges: bool) -> Union['Edge', 'ParamEdge']:
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


AbstractStackNode = Union[StackNode, ParamStackNode]


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
                 node1: AbstractStackNode,
                 axis1: Axis,
                 node2: Optional[AbstractStackNode] = None,
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
        return connect_stack(self, other)


class ParamStackEdge(AbstractStackEdge, ParamEdge):
    """
    Base class for stacks of trainable edges.
    Used for stacked contractions
    """

    def __init__(self,
                 edges: List[ParamEdge],
                 node1_lists: List[bool],
                 node1: AbstractStackNode,
                 axis1: Axis,
                 node2: Optional[AbstractStackNode] = None,
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
        stacked_mats = stack_unequal_tensors(mats)

        # When stacking nodes that were previously stacked, and the memory of
        # the current stack makes reference to the previous one with, possibly,
        # a different size, the stacked_mats could have a size that is smaller
        # from the current stack
        if stacked_mats.shape[-2:] != (self._size, self._size):
            pad = [self._size - stacked_mats.shape[-1], 0,
                   self._size - stacked_mats.shape[-2], 0]
            stacked_mats = nn.functional.pad(stacked_mats, pad)

        return stacked_mats

    def __xor__(self, other: 'ParamStackEdge') -> ParamEdge:
        return connect_stack(self, other)

    # TODO: Cual es la dimension de este edge si apilo las matrices??


################################################
#                TENSOR NETWORK                #
################################################
class Successor:
    """
    Class for successors. Object that stores information about
    the already computed operations in the network, in order to
    compute them faster next time.

    Parameters
    ----------
    kwargs: keyword arguments used in the operation
    child: node resultant from the operation
    hints: hints created the first time the computation was
        performed, so that next times we can avoid calculating
        auxiliary information needed for the computation
    """

    def __init__(self,
                 kwargs: Dict[Text, Any],
                 child: Union[AbstractNode, List[AbstractNode]],
                 contracting: Optional[bool] = None,
                 hints: Optional[Any] = None) -> None:
        self.kwargs = kwargs
        self.child = child
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
    
    operations = dict()

    def __init__(self, name: Optional[Text] = None):
        super().__init__()
        if name is None:
            name = self.__class__.__name__.lower()
        self.name = name

        # self._nodes = dict()
        self._leaf_nodes = dict()
        self._data_nodes = dict()
        self._virtual_nodes = dict()
        self._non_leaf_nodes = dict()

        self._repeated_nodes_names = dict()

        self._memory_nodes = dict()   # address -> memory
        self._inverse_memory = dict()  # address -> nodes using that memory

        self._data_nodes = dict()
        # self._memory_data_nodes = None

        self._edges = []

        # TODO: poder pasar esto como parametros
        # Flag to indicate whether the TN has optimized memory to perform contraction
        self._automemory = False
        self._unbind_mode = True  # True if training, False if not training
        self._tracing = False

        self._list_ops = []

    @property
    def nodes(self) -> Dict[Text, AbstractNode]:
        """
        All the nodes belonging to the network (including data nodes)
        """
        all_nodes = dict()
        all_nodes.update(self._leaf_nodes)
        all_nodes.update(self._data_nodes)
        all_nodes.update(self._virtual_nodes)
        all_nodes.update(self._non_leaf_nodes)
        return all_nodes

    @property
    def nodes_names(self) -> List[Text]:
        all_nodes_names = []
        al_nodes_names += list(self._leaf_nodes.keys())
        al_nodes_names += list(self._data_nodes.keys())
        al_nodes_names += list(self._virtual_nodes.keys())
        al_nodes_names += list(self._non_leaf_nodes.keys())
        return all_nodes_names

    @property
    def leaf_nodes(self) -> Dict[Text, AbstractNode]:
        # TODO: cuanto sentido tiene proteger listas, dicts, etc.
        # O devuelvo copias para protegerlos de verdad o no lo protejo
        """
        Data nodes created to feed the tensor network with input data
        """
        return self._leaf_nodes

    @property
    def data_nodes(self) -> Dict[Text, AbstractNode]:
        """
        Data nodes created to feed the tensor network with input data
        """
        return self._data_nodes

    @property
    def virtual_nodes(self) -> Dict[Text, AbstractNode]:
        """
        Data nodes created to feed the tensor network with input data
        """
        return self._virtual_nodes

    @property
    def non_leaf_nodes(self) -> Dict[Text, AbstractNode]:
        """
        Data nodes created to feed the tensor network with input data
        """
        return self._non_leaf_nodes

    @property
    def edges(self) -> List[AbstractEdge]:
        """
        List of dangling, non-batch edges of the network
        """
        return self._edges

    @property
    def automemory(self) -> bool:
        return self._automemory

    @automemory.setter
    def automemory(self, automem: bool) -> None:
        # TODO: necesito rehacer todo?
        self.clear()
        self._automemory = automem

    @property
    def unbind_mode(self) -> bool:
        return self._unbind_mode

    @unbind_mode.setter
    def unbind_mode(self, unbind: bool) -> None:
        # TODO: necesito rehacer todo?
        self.clear()
        self._unbind_mode = unbind

    def trace(self, example: Optional[Tensor] = None, *args, **kwargs) -> None:
        """Trace TensorNetwork contraction."""
        with torch.no_grad():
            self._tracing = True
            self(example, *args, **kwargs)
            self._tracing = False
            self(example, *args, **kwargs)
        # self._tracing = False

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
            prev_node = self.nodes[node.name]
            self._remove_node(prev_node)
        self._assign_node_name(node, node.name, True)

    # TODO: not used
    def add_nodes_from(self, nodes_list: Sequence[AbstractNode]):
        for name, node in nodes_list:
            self._add_node(node)

    def _add_edge(self, edge: AbstractEdge) -> None:
        # TODO: evitar añadir los edges de los stacks a la TN
        if not isinstance(edge, AbstractStackEdge):
            if isinstance(edge, ParamEdge):
                if not hasattr(self, edge.module_name):
                    # If ParamEdge is already a submodule, it is the case in which we are
                    # adding a node that "inherits" edges from previous nodes
                    self.add_module(edge.module_name, edge)
            if edge.is_dangling() and not edge.is_batch() and (edge not in self._edges):
                self._edges.append(edge)

    def _remove_edge(self, edge: AbstractEdge) -> None:
        # TODO: evitar añadir los edges de los stacks a la TN
        if not isinstance(edge, AbstractStackEdge):
            if isinstance(edge, ParamEdge):
                if hasattr(self, edge.module_name):
                    delattr(self, edge.module_name)
            if edge in self._edges:
                self._edges.remove(edge)

    def _which_dict(self, node: AbstractNode) -> Optional[Dict[Text, AbstractNode]]:
        if node._leaf:
            return self._leaf_nodes
        elif node._data:
            return self._data_nodes
        elif node._virtual:
            return self._virtual_nodes
        else:
            return self._non_leaf_nodes

    def _remove_node(self, node: AbstractNode, move_names=True) -> None:
        """
        This function only removes the reference to the node, and the reference
        to the TN that is kept by the node. To completely get rid of the node,
        it should be disconnected from any other node of the TN and removed from
        the TN.

        Args
        ----
        move_nodes: indicates whether the rest of the names should be
            changed to maintain a correct enumeration. Used when we want
            to delete many nodes quickly (and we know there will be no
            problems with remaining names)
        """
        node._temp_tensor = node.tensor
        node._tensor_info = None
        node._network = None

        self._unassign_node_name(node, move_names)

        nodes_dict = self._which_dict(node)
        if node._name in nodes_dict:
            if nodes_dict[node._name] == node:
                del nodes_dict[node._name]

                if node._name in self._memory_nodes:  # NOTE: puede que no est'e si usaba memory de otro nodo
                    del self._memory_nodes[node._name]

    def delete_node(self, node: AbstractNode, move_names=True) -> None:
        """
        This function disconnects the node from its neighbours and
        removes it from the TN
        """
        node.disconnect()
        self._remove_node(node, move_names)
        # TODO: del node

    def clear(self):
        """
        Elimina todos los non leaf y vuelve a poner a cada nodo leaf su propia memoria
        """
        # TODO: tarda mogoll'on, tengo que arreglarlo
        self._list_ops = []
        self._inverse_memory = dict()

        if self._non_leaf_nodes or self._virtual_nodes:
            # TODO: pensar esto, igual no hace falta siempre cambiar los leaf nodes
            # TODO: solo poner memoria a sí mismos si su memoria estaba en un nodo non_leaf
            # (node_ref era nodo non_leaf), as'i podemos hacer Uniform TN guardando siempre
            # tensor en nodos virtuales
            aux_dict = dict()
            aux_dict.update(self._leaf_nodes)
            aux_dict.update(self._non_leaf_nodes)
            aux_dict.update(self._virtual_nodes)
            for node in aux_dict.values():
                if node.is_virtual() and ('virtual_stack' not in node.name):
                    continue

                node._successors = dict()

                node_ref = node._tensor_info['node_ref']
                if node_ref is not None:
                    if node_ref.is_virtual() and ('virtual_uniform' in node_ref.name):
                        continue

                node._temp_tensor = node.tensor
                node._tensor_info['address'] = node.name
                node._tensor_info['node_ref'] = None
                node._tensor_info['full'] = True
                node._tensor_info['index'] = None

                if isinstance(node._temp_tensor, Parameter):
                    if hasattr(self, 'param_' + node.name):
                        delattr(self, 'param_' + node.name)

                if node.name not in self._memory_nodes:
                    self._memory_nodes[node.name] = None

                if node._temp_tensor is not None:
                    # TODO: why i need this?
                    node._unrestricted_set_tensor(node._temp_tensor)
                    node._temp_tensor = None

            for node in list(self._data_nodes.values()):
                node._successors = dict()

            aux_dict = dict()
            aux_dict.update(self._non_leaf_nodes)
            aux_dict.update(self._virtual_nodes)
            for node in list(aux_dict.values()):
                if node.is_virtual() and ('virtual_stack' not in node.name):
                    continue
                self.delete_node(node, False)

    def _add_param(self, param: Union[ParamNode, ParamEdge]) -> None:
        """
        Add parameters of ParamNode or ParamEdge to the TN
        """
        if isinstance(param, ParamNode):
            if not hasattr(self, param.name):
                self.add_module(param.name, param)
            else:
                # Nodes names are never repeated, so it is likely that this case will never occur
                raise ValueError(
                    f'Network already has attribute named {param.name}')
        elif isinstance(param, ParamEdge):
            if not hasattr(self, param.module_name):
                self.add_module(param.module_name, param)
            # If ParamEdge is already a submodule, it is the case in which we are
            # adding a node that "inherits" edges from previous nodes

    def _update_node_info(self, node: AbstractNode, new_name: Text) -> None:
        prev_name = node._name
        nodes_dict = self._which_dict(node)

        if new_name in nodes_dict:
            aux_node = nodes_dict[new_name]
            aux_node._temp_tensor = aux_node.tensor

        if nodes_dict.get(prev_name) == node:
            nodes_dict[new_name] = nodes_dict.pop(prev_name)
            # TODO: A lo mejor esto solo si address is not None
            # TODO: caso se est'a usando la memoria de otro nodo
            if node._tensor_info['address'] is not None:
                self._memory_nodes[new_name] = self._memory_nodes.pop(
                    prev_name)
                node._tensor_info['address'] = new_name

                if self._tracing and (prev_name in self._inverse_memory):
                    self._inverse_memory[new_name] = self._inverse_memory.pop(
                        prev_name)

        else:  # NOTE: Case change node name
            nodes_dict[new_name] = node
            self._memory_nodes[new_name] = node._temp_tensor
            node._temp_tensor = None
            node._tensor_info['address'] = new_name

            # TODO: in tracing mode i do not change names, this does not happen
            # if self._tracing and (prev_name in self._inverse_memory):
            #     self._inverse_memory[new_name] = self._inverse_memory[prev_name]

    def _update_node_name(self, node: AbstractNode, new_name: Text) -> None:
        if isinstance(node.tensor, Parameter):
            delattr(self, 'param_' + node._name)
        for edge in node.edges:
            if edge.is_attached_to(node):
                self._remove_edge(edge)

        self._update_node_info(node, new_name)
        node._name = new_name

        if isinstance(node.tensor, Parameter):
            if not hasattr(self, 'param_' + node.name):
                self.register_parameter(
                    'param_' + node._name, self._memory_nodes[node._name])
            else:
                # Nodes names are never repeated, so it is likely that this case will never occur
                raise ValueError(
                    f'Network already has attribute named {node._name}')
        for edge in node.edges:
            self._add_edge(edge)

    def _assign_node_name(self, node: AbstractNode, name: Text, first_time: bool = False) -> None:
        """
        Used to assign a new name to a node in the network
        """
        non_enum_prev_name = erase_enum(name)

        if not node.is_non_leaf() and (non_enum_prev_name in self.operations):
            raise ValueError(f'Node\'s name cannot be an operation name '
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

        for edge in node.edges:
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

        if isinstance(node.tensor, Parameter):
            if not hasattr(self, 'param_' + node.name):
                self.register_parameter(
                    'param_' + node._name, self._memory_nodes[node._name])
            else:
                # TODO: Nodes names are never repeated, so it is likely that this case will never occur
                raise ValueError(
                    f'Network already has attribute named {node._name}')
        for edge in node.edges:
            self._add_edge(edge)

    def _unassign_node_name(self, node: AbstractNode, move_names=True):
        """
        Modify remaining nodes names when we remove one node
        """
        if isinstance(node.tensor, Parameter):
            delattr(self, 'param_' + node._name)
        for edge in node.edges:
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
        Used to change the name of a node. If a node belongs to a network,
        we have to take care of repeated names in the network. This entails
        assigning a new name to the node, and removing the previous name
        (with subsequent changes)
        """
        # TODO: Esto no pasa, est'a protegida, solo la llamo cuando quiero
        # TODO: a lo mejor no deberiamos dejar llamar a nodos como data_...
        # si no son data nodes
        if node.network != self:
            raise ValueError('Cannot change the name of a node that does '
                             'not belong to the network')

        if erase_enum(name) != erase_enum(node.name):
            self._unassign_node_name(node)
            self._assign_node_name(node, name)

    def _change_node_type(self, node: AbstractNode, type: Text) -> None:
        """
        Used to change node from leaf, non_leaf, data or virtual
        types to another
        """
        if type not in ['leaf', 'non_leaf', 'data', 'virtual']:
            raise ValueError('`type` can only be \'leaf\', \'non_leaf\', '
                             '\'data\' or \'virtual\'')

        prev_dict = self._which_dict(node)

        if type == 'leaf':
            new_dict = self._leaf_nodes
            node._leaf = True
            node._data = False
            node._virtual = False
        elif type == 'data':
            new_dict = self._data_nodes
            node._leaf = False
            node._data = True
            node._virtual = False
        elif type == 'virtual':
            new_dict = self._virtual_nodes
            node._leaf = False
            node._data = False
            node._virtual = True
        elif type == 'non_leaf':
            new_dict = self._non_leaf_nodes
            node._leaf = False
            node._data = False
            node._virtual = False

        del prev_dict[node.name]
        new_dict[node.name] = node

    def copy(self) -> 'TensorNetwork':
        return copy.deepcopy(self)

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
        if override:
            net = self
        else:
            net = self.copy()

        if self.non_leaf_nodes:
            warnings.warn('Non-leaf nodes will be removed before parameterizing '
                          'the TN')
            self.clear()

        for node in list(net.leaf_nodes.values()):
            param_node = node.parameterize(set_param)
            param_node.param_edges(set_param)

        return net

    def initialize(self) -> None:
        """
        Initialize all nodes' tensors in the network
        """
        # Initialization methods depend on the topology of the network. Number of nodes,
        # edges and its dimensions might be relevant when specifying the initial distribution
        # (e.g. mean, std) of each node
        raise NotImplementedError(
            'Initialization methods not implemented for generic TensorNetwork class')

    def set_data_nodes(self,
                       input_edges: Union[List[int], List[AbstractEdge]],
                       num_batch_edges: int,
                       names_batch_edges: Optional[Sequence[Text]] = None) -> None:
        """
        Create data nodes and connect them to the list of specified edges of the TN.
        `set_data_nodes` should be executed after instantiating a TN, before
        computing forward.

        Parameters
        ----------
        input_edges: list of edges in the same order as they are expected to be
            contracted with each feature node of the input data_nodes
        num_batch_edges: number of batch edges in the input data
        names_batch_edges: sequence of names for the batch edges
        """
        if input_edges == []:
            raise ValueError(
                '`input_edges` is empty. Cannot set data nodes if no edges are provided')
        if self.data_nodes:
            raise ValueError(
                'Tensor network data nodes should be unset in order to set new ones')

        # Only make stack_data_memory if all the input edges have the same dimension
        same_dim = True
        for i in range(len(input_edges) - 1):
            if input_edges[i].size() != input_edges[i + 1].size():
                same_dim = False
                break

        # num_batch_edges = len(names_batch_edges)

        if same_dim:
            if 'stack_data_memory' not in self._virtual_nodes:
                # TODO: Stack data node donde se guardan los datos, se supone que todas las features tienen la misma dim
                stack_node = Node(shape=(len(input_edges), *([1]*num_batch_edges), input_edges[0].size()),  # TODO: supongo edge es AbstractEdge
                                  axes_names=('n_features',
                                              *(['batch']*num_batch_edges),
                                              'feature'),
                                  name=f'stack_data_memory',  # TODO: guardo aqui la memory, no uso memory_data_nodes
                                  network=self,
                                  virtual=True)
                n_features_node = Node(shape=(stack_node.shape[0],),
                                       axes_names=('n_features',),
                                       name='virtual_n_features',
                                       network=self,
                                       virtual=True)
                feature_node = Node(shape=(stack_node.shape[-1],),
                                    axes_names=('feature',),
                                    name='virtual_feature',
                                    network=self,
                                    virtual=True)
                stack_node['n_features'] ^ n_features_node['n_features']
                stack_node['feature'] ^ feature_node['feature']
            else:
                stack_node = self._virtual_nodes['stack_data_memory']

        # if names_batch_edges is not None:
        #     if len(names_batch_edges) != num_batch_edges:
        #         raise ValueError(f'`names_batch_edges` should have exactly '
        #                          f'{num_batch_edges} names')
        # else:
        #     names_batch_edges = [f'batch_{j}' for j in range(num_batch_edges)]

        data_nodes = []
        for i, edge in enumerate(input_edges):
            if isinstance(edge, int):
                edge = self[edge]
            elif isinstance(edge, AbstractEdge):
                if edge not in self.edges:
                    raise ValueError(
                        f'Edge {edge!r} should be a dangling edge of the Tensor Network')
            else:
                raise TypeError(
                    '`input_edges` should be List[int] or List[AbstractEdge] type')
            node = Node(shape=(*([1]*num_batch_edges), edge.size()),
                        axes_names=(*(['batch']*num_batch_edges),
                                    'feature'),
                        name=f'data_{i}',
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
        if self.data_nodes:
            for node in list(self.data_nodes.values()):
                self.delete_node(node)
            self._data_nodes = dict()

            if 'stack_data_memory' in self.virtual_nodes:
                self.delete_node(self.virtual_nodes['stack_data_memory'])
                self.delete_node(self.virtual_nodes['virtual_n_features'])
                self.delete_node(self.virtual_nodes['virtual_feature'])

    def _add_data(self, data: Union[Tensor, Sequence[Tensor]]) -> None:
        """
        Add data to data nodes, that is, change their tensors by new data tensors given a new data set.

        Parameters
        ----------
        data: data tensor, of dimensions
            n_features x batch_size_{0} x ... x batch_size_{n} x feature_size
        """

        stack_node = self.virtual_nodes.get('stack_data_memory')

        if stack_node is not None:
            stack_node._unrestricted_set_tensor(data)
        elif self.data_nodes:
            for i, data_node in enumerate(list(self.data_nodes.values())):
                data_node._unrestricted_set_tensor(data[i])
        else:
            raise ValueError('Cannot add data if no data nodes are set')

    def contract(self) -> Tensor:
        """
        Contract tensor network
        """
        # Custom, optimized contraction methods should be defined for each new subclass of TensorNetwork
        raise NotImplementedError(
            'Contraction methods not implemented for generic TensorNetwork class')

    def forward(self,
                data: Optional[Union[Tensor, Sequence[Tensor]]] = None,
                *args, **kwargs) -> Tensor:
        """
        Contract Tensor Network with input data with shape batch x n_features x feature.
        """
        # NOTE: solo hay que definir de antemano set_data_nodes y contract
        if data is not None:
            if not self.data_nodes:
                self.set_data_nodes()
            self._add_data(data=data)

        if not self.non_leaf_nodes:
            output = self.contract(*args, **kwargs)

            self._seq_ops = []
            for op in self._list_ops:
                self._seq_ops.append(
                    (op[1], op[0]._successors[op[1]][op[2]].kwargs))

            return output.tensor

        else:
            # output = self.contract(*args, **kwargs)

            # total = time.time()
            for op in self._seq_ops:
                # start = time.time()
                output = self.operations[op[0]](**op[1])
                # print(f'Time {op[0]}: {time.time() - start:.4f}')
            # print(f'Total time: {time.time() - total:.4f}')

            if not isinstance(output, Node):
                if (op[0] == 'unbind') and (len(output) == 1):
                    output = output[0]
                else:
                    raise ValueError('The last operation should be the one '
                                     'returning a single resulting node')

            return output.tensor

        # TODO: algo as'i, en la primera epoca se meten datos con batch 1, solo
        #  para ir creando todos los nodos intermedios necesarios r'apidamente,
        #  luego ya se contrae la red haciendo operaciones de tensores
        # if not self.is_contracting():
        #     # First contraction
        #     aux_data = torch.zeros([1] * (len(data.shape) - 1) + [data.shape[-1]])
        #     self._add_data(aux_data)
        #     self.is_contracting(True)
        #     self.contract()

        # self._add_data(data)
        # self.contract()
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
                raise KeyError(
                    f'Tensor network {self!s} does not have any node with name {key}')
        else:
            raise TypeError('`key` should be int or str type')

    def __str__(self) -> Text:
        return self.name

    def __repr__(self) -> Text:
        return f'{self.__class__.__name__}(\n ' \
               f'\tname: {self.name}\n' \
               f'\tnodes: \n{tab_string(print_list(list(self.nodes.keys())), 2)}\n' \
               f'\tedges:\n{tab_string(print_list(self.edges), 2)})'

    # TODO: Function to build instructions and reallocate memory, optimized for a function
    #  (se deben reasignar los par'ametros)
    # TODO: Function to allocate one memory tensor for each node, like old mode


################################################
#               EDGE OPERATIONS                #
################################################
def connect(edge1: AbstractEdge, edge2: AbstractEdge) -> Union[Edge, ParamEdge]:
    """
    Connect two dangling, non-batch edges.
    """
    # TODO: no puedo capar el conectar nodos no-leaf, pero no tiene el resultado esperado,
    #  en realidad estás conectando los nodos originales (leaf)
    if edge1 == edge2:
        return edge1

    for edge in [edge1, edge2]:
        if not edge.is_dangling():
            raise ValueError(f'Edge {edge!s} is not a dangling edge. '
                             f'This edge points to nodes: {edge.node1!s} and {edge.node2!s}')
        if edge.is_batch():
            raise ValueError(f'Edge {edge!s} is a batch edge')
    # if edge1 == edge2:
    #     raise ValueError(f'Cannot connect edge {edge1!s} to itself')
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
    net1, net2 = node1._network, node2._network

    if net1 != net2:
        node2.move_to_network(net1)
    net1._remove_edge(edge1)
    net1._remove_edge(edge2)
    net = net1

    if isinstance(edge1, ParamEdge) == isinstance(edge2, ParamEdge):
        if isinstance(edge1, ParamEdge):
            if isinstance(edge1, ParamStackEdge):
                new_edge = ParamStackEdge(edges=edge1.edges, node1_lists=edge1.node1_lists,
                                          node1=node1, axis1=axis1,
                                          node2=node2, axis2=axis2)
                # net._add_edge(new_edge)
            else:
                shift = edge1.shift
                slope = edge1.slope
                new_edge = ParamEdge(node1=node1, axis1=axis1,
                                     shift=shift, slope=slope,
                                     node2=node2, axis2=axis2)
                net._add_edge(new_edge)
        else:
            if isinstance(edge1, StackEdge):
                new_edge = StackEdge(edges=edge1.edges, node1_lists=edge1.node1_lists,
                                     node1=node1, axis1=axis1,
                                     node2=node2, axis2=axis2)
            else:
                new_edge = Edge(node1=node1, axis1=axis1,
                                node2=node2, axis2=axis2)
    else:
        if isinstance(edge1, ParamEdge):
            shift = edge1.shift
            slope = edge1.slope
        else:
            shift = edge2.shift
            slope = edge2.slope
        new_edge = ParamEdge(node1=node1, axis1=axis1,
                             shift=shift, slope=slope,
                             node2=node2, axis2=axis2)
        net._add_edge(new_edge)

    node1._add_edge(new_edge, axis1, True)
    node2._add_edge(new_edge, axis2, False)
    return new_edge


def connect_stack(edge1: AbstractStackEdge, edge2: AbstractStackEdge):
    """
    Connect stack edges only if their lists of edges are the same
    (coming from already connected edges)
    """
    if not isinstance(edge1, AbstractStackEdge) or \
            not isinstance(edge2, AbstractStackEdge):
        raise TypeError('Both edges should be (Param)StackEdge\'s')

    if edge1.edges != edge2.edges:
        raise ValueError('Cannot connect stack edges whose lists of'
                         ' edges are not the same. They will be the '
                         'same when both lists contain edges connecting'
                         ' the nodes that formed the stack nodes.')
    return connect(edge1=edge1, edge2=edge2)


def disconnect(edge: Union[Edge, ParamEdge]) -> Tuple[Union[Edge, ParamEdge],
                                                      Union[Edge, ParamEdge]]:
    """
    Disconnect an edge, returning a couple of dangling edges
    """
    if edge.is_dangling():
        raise ValueError('Cannot disconnect a dangling edge')

    nodes = []
    axes = []
    for axis, node in zip(edge._axes, edge._nodes):
        if edge in node._edges:
            nodes.append(node)
            axes.append(axis)

    new_edges = []
    first = True
    for axis, node in zip(axes, nodes):
        if isinstance(edge, Edge):
            if isinstance(edge, StackEdge):
                new_edge = StackEdge(edges=edge.edges,
                                     node1_lists=edge.node1_lists,
                                     node1=node,
                                     axis1=axis)
                new_edges.append(new_edge)

            else:
                new_edge = Edge(node1=node, axis1=axis)
                new_edges.append(new_edge)

                net = node._network
                net._add_edge(new_edge)
        else:
            if isinstance(edge, ParamStackEdge):
                new_edge = ParamStackEdge(edges=edge.edges,
                                          node1_lists=edge.node1_lists,
                                          node1=node,
                                          axis1=axis)
                new_edges.append(new_edge)

            else:
                shift = edge.shift
                slope = edge.slope
                new_edge = ParamEdge(node1=node, axis1=axis,
                                     shift=shift, slope=slope)
                new_edges.append(new_edge)

                net = node._network
                if first:
                    net._remove_edge(edge)
                    first = False
                net._add_edge(new_edge)

    for axis, node, new_edge in zip(axes, nodes, new_edges):
        node._add_edge(new_edge, axis, True)

    return tuple(new_edges)
