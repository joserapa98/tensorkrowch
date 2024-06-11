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

import copy
import warnings
from abc import abstractmethod, ABC
from typing import (overload,
                    Any, Dict, List, Optional,
                    Sequence, Text, Tuple, Union)

import torch
import torch.nn as nn
from torch import Size, Tensor
from torch.nn import Parameter

from tensorkrowch.utils import (check_name_style, enum_repeated_names, erase_enum,
                                print_list, stack_unequal_tensors, tab_string)


###############################################################################
#                                     AXIS                                    #
###############################################################################
class Axis:  # MARK: Axis
    """
    Axes are the objects that stick edges to nodes. Each instance of the
    :class:`AbstractNode` class has a list of :math:`N` axes, each corresponding
    to one edge. Each axis stores information that facilitates accessing that
    edge, such as its :attr:`name` and :attr:`num` (index). Additionally, an axis
    keeps track of its :meth:`batch <is_batch>` and :meth:`node1 <is_node1>`
    attributes.

    * **batch**: If the axis name contains the word "`batch`", the edge will be
      a batch edge, which means that it cannot be connected to other nodes.
      Instead, it specifies a dimension that allows for batch operations (e.g.,
      batch contraction). If the name of the axis is changed and no longer contains
      the word "`batch`", the corresponding edge will no longer be a batch edge.
      Furthermore, instances of the :class:`StackNode` and :class:`ParamStackNode`
      classes always have an axis with name "`stack`" whose edge is a batch edge.

    * **node1**: When two dangling edges are connected the result is a new
      edge linking two nodes, say ``nodeA`` and ``nodeB``. If the
      connection is performed in the following order::

        new_edge = nodeA[edgeA] ^ nodeB[edgeB]

      Then ``nodeA`` will be the ``node1`` of ``new_edge`` and ``nodeB``, the
      ``node2``. Hence, to access one of the nodes from ``new_edge`` one needs
      to know if it is ``node1`` or ``node2``.
      
    |

    Even though we can create ``Axis`` instances, that will not be usually the
    case, since axes are automatically created when instantiating a new
    :class:`node <AbstractNode>`.
    
    |
    
    Other thing one must take into account is the naming of ``Axes``. Since
    the name of an ``Axis`` is used to access it from the ``Node``, the
    same name cannot be used by more than one ``Axis``. In that case, repeated
    names get an automatic enumeration of the form ``"name_{number}"``
    (underscore followed by number).
    
    To add a custom enumeration in a user-defined way, one may use brackets or
    parenthesis: ``"name_({number})"``.
    
    |

    Parameters
    ----------
    num : int
        Index of the axis in the node's axes list.
    name : str
        Axis name, should not contain blank spaces or special characters. If it
        contains the word "`batch`", the axis will correspond to a batch edge.
        The word "`stack`" cannot be used in the name, since it is reserved for
        stacks.
    node : AbstractNode, optional
        Node to which the axis belongs.
    node1 : bool
        Boolean indicating whether ``node1`` of the edge attached to this axis
        is the node that contains the axis (``True``). Otherwise, the node is
        ``node2`` of the edge (``False``).

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

    Also, as explained before, knowing if a node is the ``node1`` or ``node2``
    of an edge enables users to access that node from the edge:

    >>> nodeA = tk.Node(shape=(2, 3), axes_names=('left', 'right'))
    >>> nodeB = tk.Node(shape=(3, 4), axes_names=('left', 'right'))
    >>> new_edge = nodeA['right'] ^ nodeB['left']
    ...
    >>> # nodeA is node1 and nodeB is node2 of new_edge
    >>> nodeA == new_edge.nodes[1 - nodeA.get_axis('right').is_node1()]
    True

    >>> nodeB == new_edge.nodes[nodeA.get_axis('right').is_node1()]
    True
    
    The ``node1`` attribute is extended to ``resultant`` nodes that inherit
    edges.
    
    >>> nodeA = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
    >>> nodeB = tk.randn(shape=(3, 4), axes_names=('left', 'right'))
    >>> nodeC = tk.randn(shape=(4, 5), axes_names=('left', 'right'))
    >>> edge1 = nodeA['right'] ^ nodeB['left']
    >>> edge2 = nodeB['right'] ^ nodeC['left']
    >>> result = nodeA @ nodeB
    ...
    >>> # result inherits the edges nodeA['left'] and edge2
    >>> result['left'] == nodeA['left']
    True
    
    >>> result['right'] == edge2
    True
    
    >>> # result is still node1 of edge2, since nodeA was
    >>> result.is_node1('right')
    True
    
    
    |
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
                raise ValueError('Axis cannot be named "stack" if the node is '
                                 'not a StackNode or ParamStackNode')
            if num != 0:
                raise ValueError('Axis with name "stack" should have index 0')

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
        blank spaces or special characters. If it contains the word "`batch`",
        the axis will correspond to a batch edge. The word "`stack`" cannot be
        used in the name, since it is reserved for stacks.
        """
        return self._name

    @name.setter
    def name(self, name: Text) -> None:
        """
        Sets axis name. The name should not contain blank spaces or special
        characters.
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
        Returns boolean indicating whether ``node1`` of the edge attached to this
        axis is the node that contains the axis. Otherwise, the node is ``node2``
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


class AbstractNode(ABC):  # MARK: AbstractNode
    """
    Abstract class for all types of nodes. Defines what a node is and most of its
    properties and methods. Since it is an abstract class, cannot be instantiated.
      
    Nodes are the elements that make up a :class:`TensorNetwork`. At its most
    basic level, a node is a container for a ``torch.Tensor`` that stores other
    relevant information which enables to build any network and operate nodes
    to contract it (and train it!). Some of the information that is carried by
    the nodes includes:
    
    * **Shape**: Every node needs a shape to know if connections with other
      nodes are possible. Even if the tensor is not specified, an empty node
      needs a shape.
      
    * **Tensor**: The key ingredient of the node. Although the node acts as a
      `container` for the tensor, the node does not `contain` it. Actually,
      for efficiency purposes, the tensors are stored in a sort of memory that
      is shared by all the nodes of the :class:`TensorNetwork`. Therefore, all
      that nodes `contain` is a memory address. Furthermore, some nodes can share
      the same (or a part of the same) tensor, thus containing the same address.
      Sometimes, to maintain consistency, when two nodes share a tensor, one
      stores its memory address, and the other one stores a reference to the
      former.
    
    * **Axes**: A list of :class:`Axes <Axis>` that make it easy to access edges
      just using a name or an index.
      
    * **Edges**: A list of :class:`Edges <Edge>`, one for each dimension of the
      node. Each edge is attached to the node via an :class:`Axis`. Edges are
      useful to connect several nodes, creating a :class:`TensorNetwork`.
      
    * **Network**: The :class:`TensorNetwork` to which the node belongs. If
      the network is not specified when creating the node, a new ``TensorNetwork``
      is created to contain the node. Although the network can be thought of
      as a graph, it is a ``torch.nn.Module``, so it is much more than that.
      Actually, the ``TensorNetwork`` can contain different types of nodes,
      not all of them being part of the graph, but being used for different
      purposes.
      
    * **Successors**: A dictionary with information about the nodes that result
      from :class:`Operations <Operation>` in which the current node was involved.
      See :class:`Successor`.
      
      
    Carrying this information with the node is what makes it easy to:
    
    * Perform tensor network :class:`Operations <Operation>` such as :func:`contraction
      <contract_between>` of two neighbouring nodes, without having to worry about
      tensor's shapes, order of axes, etc.

    * Perform more advanced operations such as :func:`stack` or :func:`unbind`
      saving memory and time.

    * Keep track of operations in which a node has taken place, so that several
      steps can be skipped in further training iterations.
      See :meth:`TensorNetwork.trace`.
      
    |
      
    Also, there are **4 excluding types** of nodes that will have different
    roles in the :class:`TensorNetwork`:
    
    * **leaf**: These are the nodes that form the :class:`TensorNetwork`
      (together with the ``data`` nodes). Usually, these will be the `trainable`
      nodes. These nodes can store their own tensors or use other node's tensor.
      
    * **data**: These are similar to ``leaf`` nodes, but they are never `trainable`,
      and are used to store the temporary tensors coming from input data. These
      nodes can store their own tensors or use other node's tensor.
      
    * **virtual**: These nodes are a sort of ancillary, `hidden` nodes that
      accomplish some useful task (e.g. in uniform tensor networks a virtual
      node can store the shared tensor, while all the other nodes in the
      network just have a reference to it). These nodes always store their own
      tensors.
      
    * **resultant**: These are nodes that result from an :class:`Operation`.
      They are intermediate nodes that (almost always) inherit edges from ``leaf``
      and ``data`` nodes, the ones that really form the network. These nodes can
      store their own tensors or use other node's tensor. The names of the
      ``resultant`` nodes are the name of the :class:`Operation` that originated
      it.
      
    See :class:`TensorNetwork` and :meth:`~TensorNetwork.reset` to learn more
    about the importance of these 4 types of nodes.
    
    |
    
    Other thing one should take into account are **reserved nodes' names**:
    
    * **"stack_data_memory"**: Name of the ``virtual`` :class:`StackNode` that
      is created in :meth:`~TensorNetwork.set_data_nodes` to store the whole
      data tensor from which each ``data`` node might take just one `slice`.
      There should be at most one ``"stack_data_memory"`` in the network.
      To learn more about this, see :meth:`~TensorNetwork.set_data_nodes` and
      :meth:`~TensorNetwork.add_data`.
    
    * **"virtual_result"**: Name of ``virtual`` nodes that are not explicitly
      part of the network, but are required for some situations during
      contraction. For instance, the :class:`ParamStackNode` that
      results from stacking :class:`ParamNodes <ParamNode>` as the first
      operation in the network contraction, if ``auto_stack`` mode is set to
      ``True``. To learn more about this, see :class:`ParamStackNode`.
    
    * **"virtual_uniform"**: Name of the ``virtual`` :class:`Node` or
      :class:`ParamNode` that is used in uniform (translationally invariant)
      tensor networks to store the tensor that will be shared by all ``leaf``
      nodes. There might be as much ``"virtual_uniform"`` nodes as shared
      memories are used for the ``leaf`` nodes in the network (usually just one).
    
    For ``"virtual_result"`` and ``"virtual_uniform"``, these special
    behaviours are not restricted to nodes having those names, but also nodes
    whose names contain those strings.
      
    Although these names can in principle be used for other nodes, this can lead
    to undesired behaviour.
      
    See :meth:`~TensorNetwork.reset` to learn more about the importance of these
    reserved nodes' names.
    
    |
    
    Other thing one must take into account is the naming of ``Nodes``. Since
    the name of a ``Node`` is used to access it from the ``TensorNetwork``, the
    same name cannot be used by more than one ``Node``. In that case, repeated
    names get an automatic enumeration of the form ``"name_{number}"`` (underscore
    followed by number).
    
    To add a custom enumeration to keep track of the nodes of the network in a
    user-defined way, one may use brackets or parenthesis: ``"name_({number})"``.
    
    The same automatic enumeration of names occurs for :class:`Axes <Axis>`'
    names in a ``Node``.
    
    |
    
    Refer to the subclasses of ``AbstractNode`` to see how to instantiate nodes:

    * :class:`Node`

    * :class:`ParamNode`

    * :class:`StackNode`

    * :class:`ParamStackNode`
    
    |
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
                 dtype: Optional[torch.dtype] = None,
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
            if isinstance(shape, Sequence):
                if any([not isinstance(i, int) for i in shape]):
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
            if not isinstance(axes_names, Sequence):
                raise TypeError(
                    '`axes_names` should be tuple[str] or list[str] type')
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
            raise ValueError('The node can only be one of `leaf`, `data`, '
                             '`virtual` and `resultant`')

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
                    init_method=init_method,
                    device=device,
                    dtype=dtype,
                    **kwargs)
        else:
            self._unrestricted_set_tensor(tensor=tensor)

    @classmethod
    def _create_resultant(cls, *args, **kwargs) -> 'AbstractNode':
        """
        Private constructor to create resultant nodes. Called from
        :class:`Operations <Operation>`.
        """
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
        
        address = self.tensor_address()
        index = self._tensor_info['index']
        
        result = self._network._memory_nodes[address]

        return_result = (index is None) or (result is None)
        if not self._network._auto_unbind:
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
        """Shape of node's :attr:`tensor`. It is of type ``torch.Size``."""
        return self._shape

    @property
    def rank(self) -> int:
        """Length of node's :attr:`shape`, that is, number of edges of the node."""
        return len(self._shape)

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """``torch.dtype`` of node's :attr:`tensor`."""
        tensor = self.tensor
        if tensor is None:
            return
        return tensor.dtype

    @property
    def device(self) -> Optional[torch.device]:
        """``torch.device`` of node's :attr:`tensor`."""
        tensor = self.tensor
        if tensor is None:
            return
        return tensor.device

    @property
    def axes(self) -> List[Axis]:
        """List of nodes' :class:`axes <Axis>`."""
        return self._axes

    @property
    def axes_names(self) -> List[Text]:
        """List of names of node's :class:`axes <Axis>`."""
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
        Dictionary with :class:`Operations <Operation>`' names as keys, and
        dictionaries of :class:`Successors <Successor>` of the node as values.
        The inner dictionaries use as keys the arguments used when the
        operation was called.
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
    def _save_in_network(self,
                         tensor: Optional[Union[Tensor, Parameter]]) -> None:
        pass

    @abstractmethod
    def parameterize(self, set_param: bool) -> 'AbstractNode':
        pass

    @abstractmethod
    def copy(self, share_tensor: bool = False) -> 'AbstractNode':
        pass
    
    @abstractmethod
    def change_type(self,
                    leaf: bool = False,
                    data: bool = False,
                    virtual: bool = False,) -> None:
        pass

    # -------
    # Methods
    # -------
    def is_leaf(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``leaf`` node. These are
        the nodes that form the :class:`TensorNetwork` (together with the
        ``data`` nodes). Usually, these will be the `trainable` nodes. These
        nodes can store their own tensors or use other node's tensor.
        """
        return self._leaf

    def is_data(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``data`` node. These nodes
        are similar to ``leaf`` nodes, but they are never `trainable`, and are
        used to store the temporary tensors coming from input data. These nodes
        can store their own tensors or use other node's tensor.
        """
        return self._data

    def is_virtual(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``virtual`` node. These
        nodes are a sort of ancillary, `hidden` nodes that accomplish some useful
        task (e.g. in uniform tensor networks a virtual node can store the shared
        tensor, while all the other nodes in the network just have a reference
        to it). These nodes always store their own tensors.
        
        If a ``virtual`` node is used as the node storing the shared tensor in
        a uniform (translationally invariant) :class:`TensorNetwork`, it is
        recommended to use the string **"virtual_uniform"** in the node's name
        (e.g. "virtual_uniform_mps").
        """
        return self._virtual

    def is_resultant(self) -> bool:
        """
        Returns a boolean indicating if the node is a ``resultant`` node. These
        are nodes that result from an :class:`Operation`. They are intermediate
        nodes that (almost always) inherit edges from ``leaf`` and ``data``
        nodes, the ones that really form the network. These nodes can store
        their own tensors or use other node's tensor.
        """
        return not (self._leaf or self._data or self._virtual)
    
    def is_conj(self) -> bool:
        """
        Equivalent to `torch.is_conj()
        <https://pytorch.org/docs/stable/generated/torch.is_conj.html>`_.
        """
        tensor = self.tensor
        if tensor is None:
            return
        return tensor.is_conj()
    
    def is_complex(self) -> bool:
        """
        Equivalent to `torch.is_complex()
        <https://pytorch.org/docs/stable/generated/torch.is_complex.html>`_.
        """
        tensor = self.tensor
        if tensor is None:
            return
        return tensor.is_complex()
    
    def is_floating_point(self) -> bool:
        """
        Equivalent to `torch.is_floating_point()
        <https://pytorch.org/docs/stable/generated/torch.is_floating_point.html>`_.
        """
        tensor = self.tensor
        if tensor is None:
            return
        return tensor.is_floating_point()

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
        Returns :meth:`node1 <Axis.is_node1>` attribute of axes of the node. If
        ``axis`` is specified, returns only the ``node1`` of that axis; otherwise
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
        
        If ``self`` is a ``resultant`` node, this will return the neighbours of
        the ``leaf`` nodes from which ``self`` inherits the edges. Therefore,
        one cannot check if two ``resultant`` nodes are connected by looking
        into their neighbours lists. To do that, use :meth:`is_connected_to`.

        Parameters
        ----------
        axis : int, str or Axis, optional
            Axis for which to retrieve the neighbour.

        Returns
        -------
        AbstractNode or list[AbstractNode]
        
        Examples
        --------
        >>> nodeA = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
        >>> nodeB = tk.randn(shape=(3, 4), axes_names=('left', 'right'))
        >>> nodeC = tk.randn(shape=(4, 5), axes_names=('left', 'right'))
        >>> _ = nodeA['right'] ^ nodeB['left']
        >>> _ = nodeB['right'] ^ nodeC['left']
        >>> set(nodeB.neighbours()) == {nodeA, nodeC}
        True
        
        >>> nodeB.neighbours('right') == nodeC
        True
        
        Nodes ``resultant`` from operations are still connected to original
        neighbours.
        
        >>> result = nodeA @ nodeB
        >>> result.neighbours('right') == nodeC
        True
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
    
    def is_connected_to(self, other: 'AbstractNode') -> List[Tuple[Axis]]:
        """Returns list of tuples of axes where the node is connected to ``other``"""
        connected_axes = []
        for i1, edge1 in enumerate(self._edges):
            for i2, edge2 in enumerate(other._edges):
                if (edge1 == edge2) and not edge1.is_dangling():
                    if self.is_node1(i1) != other.is_node1(i2):
                        connected_axes.append((self._axes[i1],
                                               other._axes[i2]))
        return connected_axes

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
        Changes axis size, that is, changes size of node's tensor at a certain
        axis.

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
                # starting from the "right", "bottom", "back", etc. in each dimension
                index = []
                for i, dim in enumerate(self._shape):
                    if i == axis_num:
                        index.append(slice(0, size))
                    else:
                        index.append(slice(0, dim))
                aux_shape = list(self._shape)
                aux_shape[axis_num] = size
                self._shape = Size(aux_shape)
                correct_format_tensor = self._set_tensor_format(tensor[index])
                self._direct_set_tensor(correct_format_tensor)

            elif size > self._shape[axis_num]:
                # If new size is greater than current, tensor is expanded with
                # zeros in the "right", "bottom", "back", etc. dimension
                pad = []
                for i, dim in enumerate(self._shape):
                    if i == axis_num:
                        pad += [size - dim, 0]
                    else:
                        pad += [0, 0]
                pad.reverse()
                aux_shape = list(self._shape)
                aux_shape[axis_num] = size
                self._shape = Size(aux_shape)
                correct_format_tensor = self._set_tensor_format(
                    nn.functional.pad(tensor, pad))
                self._direct_set_tensor(correct_format_tensor)

    def get_axis(self, axis: Ax) -> Axis:
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
            raise IndexError(f'Node "{self!s}" has no axis with index {axis}')
        elif isinstance(axis, str):
            for ax in self._axes:
                if axis == ax._name:
                    return ax._num
            raise IndexError(f'Node "{self!s}" has no axis with name "{axis}"')
        elif isinstance(axis, Axis):
            for ax in self._axes:
                if axis == ax:
                    return ax._num
            raise IndexError(f'Node "{self!s}" has no axis "{axis!r}"')
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
            Boolean indicating whether the node is the ``node1`` (``True``) or
            ``node2`` (``False``) of the edge.
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

    def in_which_axis(self, edge: 'Edge') -> Union[Axis, Tuple[Axis]]:
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
            return tuple(lst)

    def reattach_edges(self,
                       axes: Optional[Sequence[Ax]] = None,
                       override: bool = False) -> None:
        """
        Substitutes current edges by copies of them that are attached to the node.
        It can happen that an edge is not attached to the node if it is the result
        of an :class:`Operation` and, hence, it inherits edges from the operands.
        In that case, the new copied edges will be attached to the resultant node,
        replacing each previous ``node1`` or ``node2`` with it (according to the
        ``node1`` attribute of each axis).

        Used for in-place operations like :func:`permute_` or :func:`split_` and
        to (de)parameterize nodes.

        Parameters
        ----------
        axis : list[int, str or Axis] or tuple[int, str or Axis], optional
            The edge attached to these axes will be reattached. If ``None``,
            all edges will be reattached.
        override : bool
            Boolean indicating if the new, reattached edges should also replace
            the corresponding edges in the node's neighbours (``True``). Otherwise,
            the neighbours' edges will be pointing to the original nodes from which
            the current node inherits its edges (``False``).
            
        Examples
        --------
        >>> nodeA = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
        >>> nodeB = tk.randn(shape=(3, 4), axes_names=('left', 'right'))
        >>> nodeC = tk.randn(shape=(4, 5), axes_names=('left', 'right'))
        >>> _ = nodeA['right'] ^ nodeB['left']
        >>> _ = nodeB['right'] ^ nodeC['left']
        >>> result = nodeA @ nodeB
        
        Node ``result`` inherits its ``right`` edge from ``nodeB``.
        
        >>> result['right'] == nodeB['right']
        True
        
        However, ``nodeB['right']`` still connects ``nodeB`` and ``nodeC``.
        There is no reference to ``result``.
        
        >>> result in result['right'].nodes
        False
        
        One can reattach its edges so that ``result``'s edges do have references
        to it.
        
        >>> result.reattach_edges()
        >>> result in result['right'].nodes
        True
        
        If ``override`` is ``True``, ``nodeB['right']`` would be replaced by the
        new ``result['right']``.
        """
        if axes is None:
            edges = list(enumerate(self._edges))
        else:
            edges = []
            for axis in axes:
                axis_num = self.get_axis_num(axis)
                edges.append((axis_num, self._edges[axis_num]))
        
        skip_edges = []
        for i, edge in edges:
            if i in skip_edges:
                continue
            
            node1 = self._axes[i]._node1
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
                if not new_edge.is_dangling():
                    if neighbour != self:
                        for j, other_edge in edges[(i + 1):]:
                            if other_edge == edge:
                                new_edge._nodes[node1] = self
                                new_edge._axes[node1] = self._axes[j]
                                self._edges[j] = new_edge
                                skip_edges.append(j)

                if override:
                    if not new_edge.is_dangling():
                        if new_edge._nodes[0] != new_edge._nodes[1]:
                            new_edge._nodes[node1]._add_edge(
                                new_edge, new_edge._axes[node1], not node1)

    def disconnect(self, axis: Optional[Ax] = None) -> None:
        """
        Disconnects all edges of the node if they were connected to other nodes.
        If ``axis`` is sepcified, only the corresponding edge is disconnected.

        Parameters
        ----------
        axis : int, str or Axis, optional
            Axis whose edge will be disconnected.
            
        Examples
        --------
        >>> nodeA = tk.Node(shape=(2, 3), axes_names=('left', 'right'))
        >>> nodeB = tk.Node(shape=(3, 4), axes_names=('left', 'right'))
        >>> nodeC = tk.Node(shape=(4, 5), axes_names=('left', 'right'))
        >>> _ = nodeA['right'] ^ nodeB['left']
        >>> _ = nodeB['right'] ^ nodeC['left']
        >>> set(nodeB.neighbours()) == {nodeA, nodeC}
        True
        
        >>> nodeB.disconnect()
        >>> nodeB.neighbours() == []
        True
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
                          device: Optional[torch.device] = None,
                          dtype: Optional[torch.dtype] = None) -> Tensor:
        """Returns copy tensor (ones in the "diagonal", zeros elsewhere)."""
        copy_tensor = torch.zeros(shape, device=device, dtype=dtype)
        rank = len(shape)
        if rank <= 1:
            i = 0
        else:
            i = torch.arange(min(shape), device=device)
        copy_tensor[(i,) * rank] = 1.
        return copy_tensor

    @staticmethod
    def _make_rand_tensor(shape: Shape,
                          low: float = 0.,
                          high: float = 1.,
                          device: Optional[torch.device] = None,
                          dtype: Optional[torch.dtype] = None) -> Tensor:
        """Returns tensor whose entries are drawn from the uniform distribution."""
        if not isinstance(low, float):
            raise TypeError('`low` should be float type')
        if not isinstance(high, float):
            raise TypeError('`high` should be float type')
        if low >= high:
            raise ValueError('`low` should be strictly smaller than `high`')
        return torch.rand(shape, device=device, dtype=dtype) * (high - low) + low

    @staticmethod
    def _make_randn_tensor(shape: Shape,
                           mean: float = 0.,
                           std: float = 1.,
                           device: Optional[torch.device] = None,
                           dtype: Optional[torch.dtype] = None) -> Tensor:
        """Returns tensor whose entries are drawn from the normal distribution."""
        if not isinstance(mean, float):
            raise TypeError('`mean` should be float type')
        if not isinstance(std, float):
            raise TypeError('`std` should be float type')
        if std <= 0:
            raise ValueError('`std` should be positive')
        return torch.randn(shape, device=device, dtype=dtype) * std + mean

    def make_tensor(self,
                    shape: Optional[Shape] = None,
                    init_method: Text = 'zeros',
                    device: Optional[torch.device] = None,
                    dtype: Optional[torch.dtype] = None,
                    **kwargs: float) -> Tensor:
        """
        Returns a tensor that can be put in the node, and is initialized according
        to ``init_method``. By default, it has the same shape as the node.

        Parameters
        ----------
        shape : list[int], tuple[int] or torch.Size, optional
            Shape of the tensor. If ``None``, node's shape will be used.
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensor.
        dtype : torch.dtype, optional
            Dtype of the tensor.
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
            return torch.zeros(shape, device=device, dtype=dtype)
        elif init_method == 'ones':
            return torch.ones(shape, device=device, dtype=dtype)
        elif init_method == 'copy':
            return self._make_copy_tensor(shape, device=device, dtype=dtype)
        elif init_method == 'rand':
            return self._make_rand_tensor(shape, device=device, dtype=dtype,
                                          **kwargs)
        elif init_method == 'randn':
            return self._make_randn_tensor(shape, device=device, dtype=dtype,
                                           **kwargs)
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
        That is, if the tensor has a size that is greater than the corresponding
        size of the node for a certain axis, the tensor is cropped in that axis
        (provided that the axis is not a batch axis). If that size is smaller in
        the tensor than in the node, raises a ``ValueError``.

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

    def _direct_get_tensor(self, node_ref, index) -> Optional[Union[Tensor, Parameter]]:
        """
        Return node's tensor without checking extra conditions. This direct
        access to the tensor is used from ``Operations``, so ``self`` should
        (and will) be a non-empty node.
        """
        if index is None:
            return self._network._memory_nodes[node_ref._tensor_info['address']]
        return self._network._memory_nodes[node_ref._tensor_info['address']][index]

    def _direct_set_tensor(self, tensor: Optional[Tensor]) -> None:
        """Sets a new node's tensor without checking extra conditions."""
        self._save_in_network(tensor)
        self._shape = tensor.shape

    def _unrestricted_set_tensor(self,
                                 tensor: Optional[Tensor] = None,
                                 init_method: Optional[Text] = 'zeros',
                                 device: Optional[torch.device] = None,
                                 dtype: Optional[torch.dtype] = None,
                                 **kwargs: float) -> None:
        """
        Sets a new node's tensor or creates one with :meth:`make_tensor` and sets
        it. Before setting it, it is cast to the correct type, so that a
        ``torch.Tensor`` can be turned into a ``torch.nn.Parameter`` when setting
        it in :class:`ParamNodes <ParamNode>`. This is not restricted, can be
        used in any node, even in ``resultant`` nodes.

        Parameters
        ----------
        tensor : torch.Tensor, optional
            Tensor to be set in the node. If ``None``, and ``init_method`` is
            provided, the tensor is created with :meth:`make_tensor`. Otherwise,
            a ``None`` is set as node's tensor.
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensor.
        dtype : torch.dtype, optional
            Dtype of the tensor.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`make_tensor`.
        """
        if tensor is not None:
            if not isinstance(tensor, Tensor):
                raise TypeError('`tensor` should be torch.Tensor type')
            if device is not None:
                warnings.warn('`device` was specified but is being ignored. '
                              'Provide a tensor that is already in the required'
                              ' device')
            if dtype is not None:
                warnings.warn('`dtype` was specified but is being ignored. '
                              'Provide a tensor that already has the required'
                              ' dtype')

            if not self._compatible_shape(tensor):
                tensor = self._crop_tensor(tensor)
            correct_format_tensor = self._set_tensor_format(tensor)

        elif init_method is not None:
            node_tensor = self.tensor
            if node_tensor is not None:
                if device is None:
                    device = node_tensor.device
                if dtype is None:
                    dtype = node_tensor.dtype
            tensor = self.make_tensor(
                init_method=init_method, device=device, dtype=dtype, **kwargs)
            correct_format_tensor = self._set_tensor_format(tensor)

        else:
            correct_format_tensor = None

        self._save_in_network(correct_format_tensor)
        self._shape = tensor.shape

    def set_tensor(self,
                   tensor: Optional[Tensor] = None,
                   init_method: Optional[Text] = 'zeros',
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None,
                   **kwargs: float) -> None:
        """
        Sets new node's tensor or creates one with :meth:`make_tensor` and sets
        it. Before setting it, it is cast to the correct type: ``torch.Tensor``
        for :class:`Node` and ``torch.nn.Parameter`` for :class:`ParamNode`.
        
        When a tensor is **set** in the node, it means the node stores it, that
        is, the node has its own memory address for its tensor, rather than a
        reference to other node's tensor. Because of this, ``set_tensor`` cannot
        be applied for nodes that have a reference to other node's tensor, since
        that tensor would be changed also in the referenced node. To overcome
        this issue, see :meth:`reset_tensor_address`.
        
        This can only be used for **non** ``resultant``nodes that store their
        own tensors. For ``resultant`` nodes, tensors are set automatically when
        computing :class:`Operations <Operation>`.
        
        Although this can also be used for ``data`` nodes, input data will be
        usually automatically set into nodes when calling the :meth:`TensorNetwork.forward`
        method of :class:`TensorNetwork` with a data tensor or a sequence of
        tensors. This method calls :meth:`TensorNetwork.add_data`, which can
        also be used to set data tensors into the ``data`` nodes.

        Parameters
        ----------
        tensor : torch.Tensor, optional
            Tensor to be set in the node. If ``None``, and ``init_method`` is
            provided, the tensor is created with :meth:`make_tensor`. Otherwise,
            a ``None`` is set as node's tensor.
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensor.
        dtype : torch.dtype, optional
            Dtype of the tensor.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`make_tensor`.

        Raises
        ------
        ValueError
            If the node is a ``resultant`` node or if it does not store its own
            tensor.
            
        Examples
        --------
        >>> node = tk.Node(shape=(2, 3), axes_names=('left', 'right'))
        ...
        >>> # Call set_tensor without arguments uses the
        >>> # default init_method ("zeros")
        >>> node.set_tensor()
        >>> torch.equal(node.tensor, torch.zeros(node.shape))
        True
        
        >>> node.set_tensor(init_method='randn', mean=1., std=2., device='cuda')
        >>> torch.equal(node.tensor, torch.zeros(node.shape, device='cuda'))
        False
        
        >>> node.device
        device(type='cuda', index=0)
        
        >>> tensor = torch.randn(2, 3)
        >>> node.set_tensor(tensor)
        >>> torch.equal(node.tensor, tensor)
        True
        """
        # If node stores its own tensor
        if not self.is_resultant() and (self._tensor_info['address'] is not None):
            if (tensor is not None) and not self._compatible_shape(tensor):
                warnings.warn(f'`tensor` is being cropped to fit the shape of '
                              f'node "{self!s}" at non-batch edges')
            self._unrestricted_set_tensor(tensor=tensor,
                                          init_method=init_method,
                                          device=device,
                                          dtype=dtype,
                                          **kwargs)
        else:
            raise ValueError('Node\'s tensor can only be changed if it is not'
                             ' resultant and stores its own tensor')

    def unset_tensor(self) -> None:
        """
        Replaces node's tensor with ``None``. This can only be used for **non**
        ``resultant`` nodes that store their own tensors.
        
        Examples
        --------
        >>> node = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
        >>> node.tensor is None
        False
        
        >>> node.unset_tensor()
        >>> node.tensor is None
        True
        """
        # If node stores its own tensor
        if not self.is_resultant() and (self._tensor_info['address'] is not None):
            self._save_in_network(None)
        else:
            raise ValueError('Node\'s tensor can only be changed if it is not'
                             ' resultant and stores its own tensor')

    def set_tensor_from(self, other: 'AbstractNode') -> None:
        """
        Sets node's tensor as the tensor used by ``other`` node. That is, when
        setting the tensor this way, the current node will store a reference to
        the ``other`` node's tensor, instead of having its own tensor.
        
        The node and ``other`` should be both the same type (:class:`Node` or
        :class:`ParamNode`). Also, they should be in the same :class:`TensorNetwork`.

        Parameters
        ----------
        other : Node or ParamNode
            Node whose tensor is to be set in current node.

        Raises
        ------
        TypeError
            If ``other`` is a different type than the current node, or if it is
            in a different network.
            
        Examples
        --------
        >>> nodeA = tk.randn(shape=(2, 3),
        ...                  name='nodeA',
        ...                  axes_names=('left', 'right'))
        >>> nodeB = tk.empty(shape=(2, 3),
        ...                  name='nodeB',
        ...                  axes_names=('left', 'right'),
        ...                  network=nodeA.network)
        >>> nodeB.set_tensor_from(nodeA)
        >>> print(nodeB.tensor_address())
        nodeA
        
        Since ``nodeB`` has a reference to ``nodeA``'s tensor, if this one is
        changed, ``nodeB`` will reproduce all the changes.
        
        >>> nodeA.tensor = torch.randn(nodeA.shape)
        >>> torch.equal(nodeA.tensor, nodeB.tensor)
        True
        """
        if not isinstance(self, type(other)):
            raise TypeError('Both nodes should be the same type')
        elif self._network is not other._network:
            raise ValueError('Both nodes should be in the same network')

        del self._network._memory_nodes[self._tensor_info['address']]

        if other._tensor_info['address'] is not None:
            self._tensor_info['address'] = None
            self._tensor_info['node_ref'] = other
            self._tensor_info['index'] = None
        else:
            self._tensor_info = other._tensor_info.copy()

    def tensor_address(self) -> Text:
        """Returns address of the node's tensor in the network's memory."""
        address = self._tensor_info['address']
        if address is None:
            node_ref = self._tensor_info['node_ref']
            address = node_ref._tensor_info['address']
        return address
    
    def node_ref(self) -> 'AbstractNode':
        """Returns the node that stores current node's tensor."""
        if self._tensor_info['address'] is None:
            return self._tensor_info['node_ref']
        return self

    def reset_tensor_address(self):
        """
        Resets memory address of node's tensor to reference the node itself.
        Thus, the node will store its own tensor, instead of having a reference
        to other node's tensor.
        
        Examples
        --------
        >>> nodeA = tk.randn(shape=(2, 3),
        ...                  name='nodeA',
        ...                  axes_names=('left', 'right'))
        >>> nodeB = tk.empty(shape=(2, 3),
        ...                  name='nodeB',
        ...                  axes_names=('left', 'right'),
        ...                  network=nodeA.network)
        >>> nodeB.set_tensor_from(nodeA)
        >>> print(nodeB.tensor_address())
        nodeA
        
        Now one cannot set in ``nodeB`` a different tensor from the one in
        ``nodeA``, unless tensor address is reset in ``nodeB``.
        
        >>> nodeB.reset_tensor_address()
        >>> nodeB.tensor = torch.randn(nodeB.shape)
        >>> torch.equal(nodeA.tensor, nodeB.tensor)
        False
        """
        if self._tensor_info['address'] is None:
            self._temp_tensor = self.tensor
            self._tensor_info['address'] = self._name
            self._tensor_info['node_ref'] = None
            self._tensor_info['index'] = None

            if isinstance(self._temp_tensor, Parameter):
                if hasattr(self._network, 'param_' + self._name):
                    delattr(self._network, 'param_' + self._name)

            if self._name not in self._network._memory_nodes:
                self._network._memory_nodes[self._name] = None

            # Set tensor and save it in ``memory_nodes``
            if self._temp_tensor is not None:
                self._unrestricted_set_tensor(self._temp_tensor)
                self._temp_tensor = None
    
    def _check_inverse_memory(self, node_ref):
        """
        Checks how many times a node's tensor is accessed during contraction.
        If that tensor can be erased when ``"re-accessed"`` reaches ``"accessed"``,
        it is replaced by ``None``.
        
        This is used in subsequent calls to an operation, so it is assumed that
        all addresses accessed are already in the ``inverse_memory``.
        """
        net = self._network
        address = node_ref._tensor_info['address']

        # When contracting network, if it is traced, we keep track of the number
        # of accesses to "erasable" nodes
        aux_dict = net._inverse_memory[address]
        aux_dict['re-accessed'] += 1
        if aux_dict['accessed'] == aux_dict['re-accessed']:
            if aux_dict['erase']:
                net._memory_nodes[address] = None
            aux_dict['re-accessed'] = 0

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
        erased, then its tensor is indeed replaced by ``None``.
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

    def move_to_network(self,
                        network: 'TensorNetwork',
                        visited: Optional[List['AbstractNode']] = None) -> None:
        """
        Moves node to another network. All other nodes connected to it, or
        to a node connected to it, etc. are also moved to the new network.
        
        If a node does not store its own tensor, and is moved to other network,
        it will recover the "ownership" of its tensor.

        Parameters
        ----------
        network : TensorNetwork
            Tensor Network to which the nodes will be moved.
        visited : list[AbstractNode], optional
            List indicating the nodes that have been already moved to the new
            network, used by this DFS-like algorithm.
            
        Examples
        --------
        >>> net = tk.TensorNetwork()
        >>> nodeA = tk.Node(shape=(2, 3),
        ...                 axes_names=('left', 'right'),
        ...                 network=net)
        >>> nodeB = tk.Node(shape=(3, 4),
        ...                 axes_names=('left', 'right'),
        ...                 network=net)
        >>> nodeC = tk.Node(shape=(5, 5),
        ...                 axes_names=('left', 'right'),
        ...                 network=net)
        >>> _ = nodeA['right'] ^ nodeB['left']
        
        If ``nodeA`` is moved to other network, ``nodeB`` will also move, but
        ``nodeC`` will not.
        
        >>> net2 = tk.TensorNetwork()
        >>> nodeA.network = net2
        >>> nodeA.network == nodeB.network
        True
        
        >>> nodeA.network != nodeC.network
        True
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

    def __getitem__(self, key: Union[slice, Ax]) -> Union[List['Edge'], 'Edge']:
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
        
        Examples
        --------
        >>> node = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
        >>> node.tensor
        tensor([[-0.2799, -0.4383, -0.8387],
                [ 1.6225, -0.3370, -1.2316]])
            
        >>> node.sum()
        tensor(-1.5029)
        
        >>> node.sum('left')
        tensor([ 1.3427, -0.7752, -2.0704])
        """
        axis_num = []
        if axis is not None:
            if isinstance(axis, (list, tuple)):
                for ax in axis:
                    axis_num.append(self.get_axis_num(ax))
            else:
                axis_num.append(self.get_axis_num(axis))
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
        
        Examples
        --------
        >>> node = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
        >>> node.tensor
        tensor([[ 1.4005, -0.0521, -1.2091],
                [ 1.9844,  0.3513, -0.5920]])
            
        >>> node.mean()
        tensor(0.3139)
        
        >>> node.mean('left')
        tensor([ 1.6925,  0.1496, -0.9006])
        """
        axis_num = []
        if axis is not None:
            if isinstance(axis, (list, tuple)):
                for ax in axis:
                    axis_num.append(self.get_axis_num(ax))
            else:
                axis_num.append(self.get_axis_num(axis))
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
        
        Examples
        --------
        >>> node = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
        >>> node.tensor
        tensor([[ 0.2111, -0.9551, -0.7812],
                [ 0.2254,  0.3381, -0.2461]])
            
        >>> node.std()
        tensor(0.5567)
        
        >>> node.std('left')
        tensor([0.0101, 0.9145, 0.3784])
        """
        axis_num = []
        if axis is not None:
            if isinstance(axis, (list, tuple)):
                for ax in axis:
                    axis_num.append(self.get_axis_num(ax))
            else:
                axis_num.append(self.get_axis_num(axis))
        return self.tensor.std(dim=axis_num)

    def norm(self,
             p: Union[int, float] = 2,
             axis: Optional[Union[Ax, Sequence[Ax]]] = None,
             keepdim: bool = False) -> Tensor:
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
        keepdim : bool
            Boolean indicating whether the output tensor have dimensions
            retained or not.

        Returns
        -------
        torch.Tensor
        
        Examples
        --------
        >>> node = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
        >>> node.tensor
        tensor([[ 1.5570,  1.8441, -0.0743],
                [ 0.4572,  0.7592,  0.6356]])
            
        >>> node.norm()
        tensor(2.6495)
        
        >>> node.norm(axis='left')
        tensor([1.6227, 1.9942, 0.6399])
        """
        axis_num = []
        if axis is not None:
            if isinstance(axis, (list, tuple)):
                for ax in axis:
                    axis_num.append(self.get_axis_num(ax))
            else:
                axis_num.append(self.get_axis_num(axis))
        return self.tensor.norm(p=p, dim=axis_num, keepdim=keepdim)

    def numel(self) -> Tensor:
        """
        Returns the total number of elements in the node's tensor.

        See also `torch.numel() <https://pytorch.org/docs/stable/generated/torch.numel.html>`_.

        Returns
        -------
        int
        
        Examples
        --------
        >>> node = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
        >>> node.numel()
        6
        """
        return self.tensor.numel()

    def __str__(self) -> Text:
        return self._name

    def __repr__(self) -> Text:
        return f'{self.__class__.__name__}(\n ' \
               f'\tname: {self._name}\n' \
               f'\ttensor:\n{tab_string(repr(self.tensor), 2)}\n' \
               f'\taxes:\n{tab_string(print_list(self.axes_names), 2)}\n' \
               f'\tedges:\n{tab_string(print_list(self._edges), 2)})'


class Node(AbstractNode):  # MARK: Node
    """
    Base class for non-trainable nodes. Should be subclassed by any class of nodes
    that are not intended to be trained (e.g. :class:`StackNode`).

    Can be used for fixed nodes of the :class:`TensorNetwork`, or intermediate
    nodes that are resultant from an :class:`Operation` between nodes.
    
    |
    
    All **4 types of nodes** (``leaf``, ``data``, ``virtual`` and ``resultant``)
    can be ``Node``. In fact, ``data`` and ``resultant`` nodes can **only** be
    of class ``Node``, since they are not intended to be trainable. To learn
    more about these **4 types of nodes**, see :class:`AbstractNode`.
    
    |

    For a complete list of properties and methods, see also :class:`AbstractNode`.
    
    |

    Parameters
    ----------
    shape : list[int], tuple[int] or torch.Size, optional
        Node's shape, that is, the shape of its tensor. If ``shape`` and
        ``init_method`` are provided, a tensor will be made for the node. Otherwise,
        ``tensor`` would be required.
    axes_names : list[str] or tuple[str], optional
        Sequence of names for each of the node's axes. Names are used to access
        the edge that is attached to the node in a certain axis. Hence, they should
        be all distinct. They cannot contain blank spaces or special characters.
        By default, axes names will be ``"axis_0"``, ..., ``"axis_n"``, being
        ``n`` the nummber of axes. If an axis' name contains the word ``"batch"``,
        it will define a batch edge. The word ``"stack"`` cannot be used, since
        it is reserved for the stack edge of :class:`StackNode`.
    name : str, optional
        Node's name, used to access the node from de :class:`TensorNetwork` where
        it belongs. It cannot contain blank spaces. By default, it is the name
        of the class (e.g. ``"node"``, ``"paramnode"``).
    network : TensorNetwork, optional
        Tensor network where the node should belong. If ``None``, a new tensor
        network will be created to contain the node.
    data : bool
        Boolean indicating if the node is a ``data`` node.
    virtual : bool
        Boolean indicating if the node is a ``virtual`` node.
    override_node : bool
        Boolean indicating whether the node should override (``True``) another
        node in the network that has the same name (e.g. if a node is parameterized,
        it would be required that a new :class:`ParamNode` replaces the
        non-parameterized node in the network).
    tensor : torch.Tensor, optional
        Tensor that is to be stored in the node. If ``None``, ``shape`` and
        ``init_method`` will be required.
    edges : list[Edge], optional
        List of edges that are to be attached to the node. This can be used in
        case the node inherits the edges from other node(s), like results from
        :class:`Operations <Operation>`.
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
        Device where to initialize the tensor if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`AbstractNode.make_tensor`.
        
    Examples
    --------
    >>> node = tk.Node(shape=(2, 5, 2),
    ...                axes_names=('left', 'input', 'right'),
    ...                name='my_node',
    ...                init_method='randn',
    ...                mean=0.,
    ...                std=1.)
    >>> node
    Node(
     	name: my_node
    	tensor:
                tensor([[[-1.2517, -1.8147],
                         [-0.7997, -0.0440],
                         [-0.2808,  0.3508],
                         [-1.2380,  0.8859],
                         [-0.3585,  0.8815]],
                        [[-0.2898, -2.2775],
                         [ 1.2856, -0.3222],
                         [-0.8911, -0.4216],
                         [ 0.0086,  0.2449],
                         [-2.1998, -1.6295]]])
    	axes:
                [left
                 input
                 right]
    	edges:
                [my_node[left] <-> None
                 my_node[input] <-> None
                 my_node[right] <-> None])
    
    Also, one can use one of the :ref:`Initializers` to simplify:
    
    >>> node = tk.randn((2, 5, 2))
    >>> node
    Node(
     	name: node
    	tensor:
                tensor([[[ 0.6545, -0.0445],
                         [-0.9265, -0.2730],
                         [-0.5069, -0.6524],
                         [-0.8227, -1.1211],
                         [ 0.2390,  0.9432]],
                        [[ 0.8633,  0.4402],
                         [-0.6982,  0.4461],
                         [-0.0633, -0.9320],
                         [ 1.6023,  0.5406],
                         [ 0.3489, -0.3088]]])
    	axes:
                [axis_0
                 axis_1
                 axis_2]
    	edges:
                [node[axis_0] <-> None
                 node[axis_1] <-> None
                 node[axis_2] <-> None])
    
    
    |
    """

    # -------
    # Methods
    # -------
    def _make_edge(self, axis: Axis) -> 'Edge':
        """Makes :class:`Edges <Edge>` that will be attached to each axis."""
        return Edge(node1=self, axis1=axis)

    @staticmethod
    def _set_tensor_format(tensor: Tensor) -> Tensor:
        """
        Returns a ``torch.Tensor`` if input tensor is given as ``torch.nn.Parameter``.
        """
        if isinstance(tensor, Parameter):
            return tensor.detach()
        return tensor
    
    def _save_in_network(self,
                         tensor: Optional[Union[Tensor, Parameter]]) -> None:
        """Saves new node's tensor in the network's memory."""
        self._network._memory_nodes[self._tensor_info['address']] = tensor

    def parameterize(self, set_param: bool = True) -> Union['Node', 'ParamNode']:
        """
        Replaces the node with a parameterized version of it, that is, turns a
        fixed :class:`Node` into a trainable :class:`ParamNode`.

        Since the node is **replaced**, it will be completely removed from the
        network, and its neighbours will point to the new parameterized node.

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
            
        Examples
        --------
        >>> nodeA = tk.randn((2, 3))
        >>> nodeB = tk.randn((3, 4))
        >>> _ = nodeA[1] ^ nodeB[0]
        >>> paramnodeA = nodeA.parameterize()
        >>> nodeB.neighbours() == [paramnodeA]
        True
        
        >>> isinstance(paramnodeA.tensor, torch.nn.Parameter)
        True
        
        ``nodeA`` still exists and has an edge pointing to ``nodeB``, but the
        latter does not "see" the former. It should be deleted.
        
        >>> del nodeA
        
        To overcome this issue, one should override ``nodeA``:
        
        >>> nodeA = nodeA.parameterize()
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
        but the exact same edges) and whose name is extended with the suffix
        ``"_copy"``.
        
        To create a copy that has its own (non-inherited) edges, one can use
        :meth:`~AbstractNode.reattach_edges` afterwards.
        
        Parameters
        ----------
        share_tensor : bool
            Boolean indicating whether the copied node should store its own
            copy of the tensor (``False``) or share it with the original node
            (``True``) storing a reference to it.

        Returns
        -------
        Node
            
        Examples
        --------
        >>> node = tk.randn(shape=(2, 3), name='node')
        >>> copy = node.copy()
        >>> node.tensor_address() != copy.tensor_address()
        True
        
        >>> torch.equal(node.tensor, copy.tensor)
        True
        
        If tensor is shared:
        
        >>> copy = node.copy(True)
        >>> node.tensor_address() == copy.tensor_address()
        True
        
        >>> torch.equal(node.tensor, copy.tensor)
        True
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
    
    def change_type(self,
                    leaf: bool = False,
                    data: bool = False,
                    virtual: bool = False,) -> None:
        """
        Changes node type, only if node is not a resultant node.
        
        Parameters
        ----------
        leaf : bool
            Boolean indicating if the new node type is ``leaf``.
        data : bool
            Boolean indicating if the new node type is ``data``.
        virtual : bool
            Boolean indicating if the new node type is ``virtual``.
        """
        if self.is_resultant():
            raise ValueError('Only non-resultant nodes\' types can be changed')
        
        if (leaf + data + virtual) != 1:
            raise ValueError('One, and only one, of `leaf`, `data` and `virtual`'
                             ' can be set to True')
        
        # Unset current type
        if self._leaf and not leaf:
            node_dict = self._network._leaf_nodes
            self._leaf = False
            del node_dict[self._name]
        elif self._data and not data:
            node_dict = self._network._data_nodes
            self._data = False
            del node_dict[self._name]
        elif self._virtual and not virtual:
            node_dict = self._network._virtual_nodes
            self._virtual = False
            del node_dict[self._name]
        
        # Set new type
        if leaf:
            self._leaf = True
            self._network._leaf_nodes[self._name] = self
        elif data:
            self._data = True
            self._network._data_nodes[self._name] = self
        elif virtual:
            self._virtual = True
            self._network._virtual_nodes[self._name] = self


class ParamNode(AbstractNode):  # MARK: ParamNode
    """
    Class for trainable nodes. Should be subclassed by any class of nodes that
    are intended to be trained (e.g. :class:`ParamStackNode`).

    Should be used as the initial nodes conforming the :class:`TensorNetwork`,
    if it is going to be trained. When operating these initial nodes, the resultant
    nodes will be non-parameterized (e.g. :class:`Node`, :class:`StackNode`).

    The main difference with :class:`Nodes <Node>` is that ``ParamNodes`` have
    ``torch.nn.Parameter`` tensors instead of ``torch.Tensor``. Therefore, a
    ``ParamNode`` is a sort of `parameter` that is attached to the
    :class:`TensorNetwork` (which is itself a ``torch.nn.Module``). That is,
    the **list of parameters of the tensor network** module contains the tensors
    of all ``ParamNodes``.
    
    |
    
    ``ParamNodes`` can only be ``leaf`` and ``virtual`` (e.g. a ``virtual`` node
    used in a uniform :class:`TensorNetwork` to store the tensor that is shared
    by all the trainable nodes must also be a ``ParamNode``, since it stores
    a ``torch.nn.Parameter``).
    
    |

    For a complete list of properties and methods, see also :class:`AbstractNode`.
    
    |
    
    Parameters
    ----------
    shape : list[int], tuple[int] or torch.Size, optional
        Node's shape, that is, the shape of its tensor. If ``shape`` and
        ``init_method`` are provided, a tensor will be made for the node. Otherwise,
        ``tensor`` would be required.
    axes_names : list[str] or tuple[str], optional
        Sequence of names for each of the node's axes. Names are used to access
        the edge that is attached to the node in a certain axis. Hence, they should
        be all distinct. They cannot contain blank spaces or special characters.
        By default, axes names will be ``"axis_0"``, ..., ``"axis_n"``, being
        ``n`` the nummber of axes. If an axis' name contains the word ``"batch"``,
        it will define a batch edge. The word ``"stack"`` cannot be used, since
        it is reserved for the stack edge of :class:`StackNode`.
    name : str, optional
        Node's name, used to access the node from de :class:`TensorNetwork` where
        it belongs. It cannot contain blank spaces. By default, it is the name
        of the class (e.g. ``"node"``, ``"paramnode"``).
    network : TensorNetwork, optional
        Tensor network where the node should belong. If ``None``, a new tensor
        network will be created to contain the node.
    virtual : bool
        Boolean indicating if the node is a ``virtual`` node.
    override_node : bool
        Boolean indicating whether the node should override (``True``) another
        node in the network that has the same name (e.g. if a node is parameterized,
        it would be required that a new :class:`ParamNode` replaces the
        non-parameterized node in the network).
    tensor : torch.Tensor, optional
        Tensor that is to be stored in the node. If ``None``, ``shape`` and
        ``init_method`` will be required.
    edges : list[Edge], optional
        List of edges that are to be attached to the node. This can be used in
        case the node inherits the edges from other node(s), like results from
        :class:`Operations <Operation>`.
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
        Device where to initialize the tensor if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`AbstractNode.make_tensor`.
        
    Examples
    --------
    >>> node = tk.ParamNode(shape=(2, 5, 2),
    ...                     axes_names=('left', 'input', 'right'),
    ...                     name='my_paramnode',
    ...                     init_method='randn',
    ...                     mean=0.,
    ...                     std=1.)
    >>> node
    ParamNode(
     	name: my_paramnode
    	tensor:
                Parameter containing:
                tensor([[[ 1.8090, -0.1371],
                         [-0.0501, -1.0371],
                         [ 1.4588, -0.8361],
                         [-0.4974, -1.9957],
                         [ 0.3760, -1.0412]],
                        [[ 0.3393, -0.2503],
                         [ 1.7752, -0.0188],
                         [-0.9561, -0.0806],
                         [-1.0465, -0.5731],
                         [ 1.5021,  0.4181]]], requires_grad=True)
    	axes:
                [left
                 input
                 right]
    	edges:
                [my_paramnode[left] <-> None
                 my_paramnode[input] <-> None
                 my_paramnode[right] <-> None])
    
    Also, one can use one of the :ref:`Initializers` to simplify:
    
    >>> node = tk.randn((2, 5, 2),
    ...                 param_node=True)
    >>> node
    ParamNode(
     	name: paramnode
    	tensor:
                Parameter containing:
                tensor([[[-0.8442,  1.4184],
                         [ 0.4431, -1.4385],
                         [-0.5161, -0.6492],
                         [ 0.2095,  0.5760],
                         [-0.9925, -1.5797]],
                        [[-0.8649, -0.5401],
                         [-0.1091,  1.1654],
                         [-0.3821, -0.2477],
                         [-0.7688, -2.4731],
                         [-0.0234,  0.9618]]], requires_grad=True)
    	axes:
                [axis_0
                 axis_1
                 axis_2]
    	edges:
                [paramnode[axis_0] <-> None
                 paramnode[axis_1] <-> None
                 paramnode[axis_2] <-> None])
       
    
    |
    """

    def __init__(self,
                 shape: Optional[Shape] = None,
                 axes_names: Optional[Sequence[Text]] = None,
                 name: Optional[Text] = None,
                 network: Optional['TensorNetwork'] = None,
                 virtual: bool = False,
                 override_node: bool = False,
                 tensor: Optional[Tensor] = None,
                 edges: Optional[List['Edge']] = None,
                 override_edges: bool = False,
                 node1_list: Optional[List[bool]] = None,
                 init_method: Optional[Text] = None,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs: float) -> None:

        super().__init__(shape=shape,
                         axes_names=axes_names,
                         name=name,
                         network=network,
                         data=False,
                         virtual=virtual,
                         override_node=override_node,
                         tensor=tensor,
                         edges=edges,
                         override_edges=override_edges,
                         node1_list=node1_list,
                         init_method=init_method,
                         device=device,
                         dtype=dtype,
                         **kwargs)

    # ----------
    # Properties
    # ----------
    @property
    def grad(self) -> Optional[Tensor]:
        """
        Returns gradient of the param-node's tensor.

        See also `torch.Tensor.grad
        <https://pytorch.org/docs/stable/generated/torch.Tensor.grad.html>`_

        Returns
        -------
        torch.Tensor or None
        
        Examples
        --------
        >>> paramnode = tk.randn((2, 3), param_node=True)
        >>> paramnode.tensor
        Parameter containing:
        tensor([[-0.3340,  0.6811, -0.2866],
                [ 1.3371,  1.4761,  0.6551]], requires_grad=True)
            
        >>> paramnode.sum().backward()
        >>> paramnode.grad
        tensor([[1., 1., 1.],
                [1., 1., 1.]])
        """
        if self._tensor_info['address'] is None:
            aux_node = self._tensor_info['node_ref']
            tensor = aux_node._network._memory_nodes[
                aux_node._tensor_info['address']]
        else:
            tensor = self._network._memory_nodes[
                self._tensor_info['address']]

        if tensor is None:
            return

        aux_grad = tensor.grad
        if aux_grad is None:
            return aux_grad
        else:
            if self._tensor_info['index'] is None:
                return aux_grad
            return aux_grad[self._tensor_info['index']]

    # -------
    # Methods
    # -------
    @classmethod
    def _create_resultant(cls, *args, **kwargs) -> None:
        """
        Private constructor to create resultant nodes. Called from
        :class:`Operations <Operation>`.
        """
        raise NotImplementedError('ParamNodes can not be resultant nodes')

    def _make_edge(self, axis: Axis) -> 'Edge':
        """Makes ``Edges`` that will be attached to each axis."""
        return Edge(node1=self, axis1=axis)

    @staticmethod
    def _set_tensor_format(tensor: Tensor) -> Parameter:
        """Returns a nn.Parameter if input tensor is just torch.Tensor."""
        if isinstance(tensor, Parameter):
            return tensor
        return Parameter(tensor)
    
    def _save_in_network(self,
                         tensor: Optional[Union[Tensor, Parameter]]) -> None:
        """Saves new node's tensor in the network's memory, and registers parameter."""
        self._network._memory_nodes[self._tensor_info['address']] = tensor
        self._network.register_parameter(
            'param_' + self._tensor_info['address'], tensor)

    def parameterize(self, set_param: bool = True) -> Union['Node', 'ParamNode']:
        """
        Replaces the param-node with a de-parameterized version of it, that is,
        turns a :class:`ParamNode` into a non-trainable, fixed :class:`Node`.

        Since the param-node is **replaced**, it will be completely removed from
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
            
        Examples
        --------
        >>> paramnodeA = tk.randn((2, 3), param_node=True)
        >>> paramnodeB = tk.randn((3, 4), param_node=True)
        >>> _ = paramnodeA[1] ^ paramnodeB[0]
        >>> nodeA = paramnodeA.parameterize(False)
        >>> paramnodeB.neighbours() == [nodeA]
        True
        
        >>> isinstance(nodeA.tensor, torch.nn.Parameter)
        False
        
        ``paramnodeA`` still exists and has an edge pointing to ``paramnodeB``,
        but the latter does not "see" the former. It should be deleted.
        
        >>> del paramnodeA
        
        To overcome this issue, one should override ``paramnodeA``:
        
        >>> paramnodeA = paramnodeA.parameterize()
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
        extended with the suffix ``"_copy"``.
        
        To create a copy that has its own (non-inherited) edges, one can use
        :meth:`~AbstractNode.reattach_edges` afterwards.
        
        Parameters
        ----------
        share_tensor : bool
            Boolean indicating whether the copied param-node should store its
            own copy of the tensor (``False``) or share it with the original
            param-node (``True``) storing a reference to it.

        Returns
        -------
        ParamNode
        
        Examples
        --------
        >>> paramnode = tk.randn(shape=(2, 3), name='node', param_node=True)
        >>> copy = paramnode.copy()
        >>> paramnode.tensor_address() != copy.tensor_address()
        True
        
        >>> torch.equal(paramnode.tensor, copy.tensor)
        True
        
        If tensor is shared:
        
        >>> copy = paramnode.copy(True)
        >>> paramnode.tensor_address() == copy.tensor_address()
        True
        
        >>> torch.equal(paramnode.tensor, copy.tensor)
        True
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
    
    def change_type(self,
                    leaf: bool = False,
                    virtual: bool = False,) -> None:
        """
        Changes node type, only if node is not a resultant node.
        
        Parameters
        ----------
        leaf : bool
            Boolean indicating if the new node type is ``leaf``.
        virtual : bool
            Boolean indicating if the new node type is ``virtual``.
        """
        if (leaf + virtual) != 1:
            raise ValueError('One, and only one, of `leaf`, and `virtual`'
                             ' can be set to True')
        
        # Unset current type
        if self._leaf and not leaf:
            node_dict = self._network._leaf_nodes
            self._leaf = False
            del node_dict[self._name]
        elif self._virtual and not virtual:
            node_dict = self._network._virtual_nodes
            self._virtual = False
            del node_dict[self._name]
        
        # Set new type
        if leaf:
            self._leaf = True
            self._network._leaf_nodes[self._name] = self
        elif virtual:
            self._virtual = True
            self._network._virtual_nodes[self._name] = self


###############################################################################
#                                 STACK NODES                                 #
###############################################################################
class StackNode(Node):  # MARK: StackNode
    """
    Class for stacked nodes. ``StackNodes`` are nodes that store the information
    of a list of nodes that are stacked via :func:`stack`, although they can also
    be instantiated directly. To do so, there are two options:

    * Provide a sequence of nodes: if ``nodes`` are provided, their tensors will
      be stacked and stored in the ``StackNode``. It is necessary that all nodes
      are of the same class (:class:`Node` or :class:`ParamNode`), have the same
      rank (although dimension of each leg can be different for different nodes;
      in which case smaller tensors are extended with 0's to match the dimensions
      of the largest tensor in the stack), same axes names (to ensure only the
      "same kind" of nodes are stacked), belong to the same network and have edges
      with the same type in each axis (:class:`Edge` or :class:`ParamEdge`).

    * Provide a stacked tensor: if the stacked ``tensor`` is provided, it is also
      necessary to specify the ``axes_names``, ``network``, ``edges`` and
      ``node1_list``.
      
    |

    ``StackNodes`` have an additional axis for the new `stack` dimension, which
    is a batch edge. This way, some contractions can be computed in parallel by
    first stacking two sequences of nodes (connected pair-wise), performing the
    batch contraction and finally unbinding the ``StackNodes`` to retrieve just
    one sequence of nodes.

    For the rest of the axes, a list of the edges corresponding to all nodes in
    the stack is stored, so that, when :func:`unbinding <unbind>` the stack, it
    can be inferred to which nodes the unbound nodes have to be connected.
    
    |

    Parameters
    ----------
    nodes : list[AbstractNode] or tuple[AbstractNode], optional
        Sequence of nodes that are to be stacked. They should all be of the same
        class (:class:`Node` or :class:`ParamNode`), have the same rank, same
        axes names and belong to the same network. They do not need to have equal
        shapes.
    axes_names : list[str], tuple[str], optional
        Sequence of names for each of the node's axes. Names are used to access
        the edge that is attached to the node in a certain axis. Hence, they should
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
        it would be required that a new :class:`ParamNode` replaces the
        non-parameterized node in the network).
    tensor : torch.Tensor, optional
        Tensor that is to be stored in the node. Necessary if ``nodes`` are not
        provided.
    edges : list[Edge], optional
        List of edges that are to be attached to the node. Necessary if ``nodes``
        are not provided.
    node1_list : list[bool], optional
        If ``edges`` are provided, the list of ``node1`` attributes of each edge
        should also be provided. Necessary if ``nodes`` are not provided.

    Examples
    --------
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

    >>> result[0].axes
    [Axis( left (0) ), Axis( right (1) )]

    >>> result[0].shape
    torch.Size([2, 2])
    
    
    |
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
            if not isinstance(nodes, Sequence):
                raise TypeError('`nodes` should be a list or tuple of nodes')
            if any([isinstance(node, (StackNode, ParamStackNode)) for node in nodes]):
                raise TypeError('Cannot create a stack using (Param)StackNode\'s')
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

                # NOTE: If this condition is removed, ensure batch attr is equal
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
                edges_dict[axis_name] = edge._edges
                node1_lists_dict[axis_name] = edge._node1_list

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
        """
        Returns dictionary where the keys are the axes. For each axis, the value
        is the list of all the edges (one from each node) that correspond to
        that axis.
        """
        return self._edges_dict

    @property
    def node1_lists_dict(self) -> Dict[Text, List[bool]]:
        """
        Returns dictionary where the keys are the axes. For each axis, the value
        is the list with the ``node1`` attribute of that axis for all nodes.
        """
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
                             node1_list=self._node1_lists_dict[axis._name],
                             node1=self, axis1=axis)
    
    def reconnect(self, other: Union['StackNode', 'ParamStackNode']) -> None:
        """
        Re-connects the ``StackNode`` to another ``(Param)StackNode``, in the
        axes where the original stacked nodes were already connected.
        """
        for axis1 in self._edges_dict:
            for axis2 in other._edges_dict:
                if self._edges_dict[axis1][0] == other._edges_dict[axis2][0]:
                    connect_stack(self.get_edge(axis1), other.get_edge(axis2))
    
    def __xor__(self, other: Union['StackNode', 'ParamStackNode']) -> None:
        self.reconnect(other)


class ParamStackNode(ParamNode):  # MARK: ParamStackNode
    """
    Class for parametric stacked nodes. They are essentially the same as
    :class:`StackNodes <StackNode>` but they are :class:`ParamNodes <ParamNode>`.
    
    They are used to optimize memory usage and save some time when the first
    operation that occurs to param-nodes in a contraction (that might be
    computed several times during training) is :func:`stack`. If this is the case,
    the param-nodes no longer store their own tensors, but rather they make
    reference to a slide of a greater ``ParamStackNode`` (if ``auto_stack``
    attribute of the :class:`TensorNetwork` is set to ``True``). Hence, that
    first :func:`stack` is never actually computed.
    
    The ``ParamStackNode`` that results from this process has the name
    ``"virtual_result_stack"``, which contains the reserved name
    ``"virtual_result"``, as explained :class:`here <AbstractNode>`. This node
    stores the tensor from which all the stacked :class:`ParamNodes <ParamNode>`
    just take one `slice`.
    
    This behaviour occurs when stacking param-nodes via :func:`stack`, not when
    instantiating ``ParamStackNode`` manually.
    
    |

    ``ParamStackNodes`` can only be instantiated by providing a sequence of nodes.
    
    |

    Parameters
    ----------
    nodes : list[AbstractNode] or tuple[AbstractNode]
        Sequence of nodes that are to be stacked. They should all be of the same
        class (:class:`Node` or :class:`ParamNode`), have the same rank, same
        axes names and belong to the same network. They do not need to have equal
        shapes.
    name : str, optional
        Node's name, used to access the node from de :class:`TensorNetwork` where
        it belongs. It cannot contain blank spaces.
    virtual : bool, optional
        Boolean indicating if the node is a ``virtual`` node. Since it will be
        used mainly for the case described :class:`here <ParamStackNode>`, the
        node will be ``virtual``, it will not be an `effective` part of the
        tensor network.
    override_node : bool, optional
        Boolean indicating whether the node should override (``True``) another
        node in the network that has the same name (e.g. if a node is parameterized,
        it would be required that a new :class:`ParamNode` replaces the
        non-parameterized node in the network).

    Examples
    --------
    >>> net = tk.TensorNetwork()
    >>> net.auto_stack = True
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
    >>> print(nodes[0].tensor_address())
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
    
    
    |
    """

    def __init__(self,
                 nodes: Sequence[AbstractNode],
                 name: Optional[Text] = None,
                 virtual: bool = False,
                 override_node: bool = False) -> None:

        if not isinstance(nodes, Sequence):
            raise TypeError('`nodes` should be a list or tuple of nodes')
        if any([isinstance(node, (StackNode, ParamStackNode)) for node in nodes]):
                raise TypeError('Cannot create a stack using (Param)StackNode\'s')

        for i in range(len(nodes[:-1])):
            if not isinstance(nodes[i], type(nodes[i + 1])):
                raise TypeError('Cannot stack nodes of different types. Nodes '
                                'must be either all Node or all ParamNode type')
            if nodes[i].rank != nodes[i + 1].rank:
                raise ValueError(
                    'Cannot stack nodes with different number of edges')

            # NOTE: If this condition is removed, ensure batch attr is equal
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
        """
        Returns dictionary where the keys are the axes. For each axis, the value
        is the list of all the edges (one from each node) that correspond to
        that axis.
        """
        return self._edges_dict

    @property
    def node1_lists_dict(self) -> Dict[Text, List[bool]]:
        """
        Returns dictionary where the keys are the axes. For each axis, the value
        is the list with the ``node1`` attribute of that axis for all nodes.
        """
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
                             node1_list=self._node1_lists_dict[axis._name],
                             node1=self, axis1=axis)
    
    def reconnect(self, other: Union['StackNode', 'ParamStackNode']) -> None:
        """
        Re-connects the ``StackNode`` to another ``(Param)StackNode``, in the
        axes where the original stacked nodes were already connected.
        """
        for axis1 in self._edges_dict:
            for axis2 in other._edges_dict:
                if self._edges_dict[axis1][0] == other._edges_dict[axis2][0]:
                    connect_stack(self.get_edge(axis1), other.get_edge(axis2))
    
    def __xor__(self, other: Union['StackNode', 'ParamStackNode']) -> None:
        self.reconnect(other)


###############################################################################
#                                    EDGES                                    #
###############################################################################
class Edge:  # MARK: Edge
    """
    Base class for edges. Should be subclassed by any new class of edges.

    An edge is nothing more than an object that wraps references to the nodes it
    connects. Thus, it stores information like the nodes it connects, the
    corresponding nodes' axes it is attached to, whether it is dangling or
    batch, its size, etc.

    Above all, its importance lies in that edges enable to connect nodes, forming
    any possible graph, and to perform easily :class:`Operations <Operation>` like
    contracting and splitting nodes.
    
    |

    Furthermore, edges have specific operations like :meth:`contract` or
    :meth:`svd` (and its variations), as well as in-place versions of them 
    (:meth:`contract_`, :meth:`svd_`, etc.) that allow in-place modification
    of the :class:`TensorNetwork`.
    
    |

    Parameters
    ----------
    node1 : AbstractNode
        First node to which the edge is connected.
    axis1: int, str or Axis
        Axis of ``node1`` where the edge is attached.
    node2 : AbstractNode, optional
        Second node to which the edge is connected. If ``None,`` the edge will
        be dangling.
    axis2 : int, str, Axis, optional
        Axis of ``node2`` where the edge is attached.
    
    Examples
    --------
    >>> nodeA = tk.randn((2, 3))
    >>> nodeB = tk.randn((3, 4))
    >>> _ = nodeA[1] ^ nodeB[0]
    >>> nodeA[0]
    Edge( node_0[axis_0] <-> None )  (Dangling Edge)
    
    >>> nodeA[1]
    Edge( node_0[axis_1] <-> node_1[axis_0] )
    
    >>> nodeB[1]
    Edge( node_1[axis_1] <-> None )  (Dangling Edge)
    
    
    |
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
        """Returns ``node1`` of the edge."""
        return self._nodes[0]

    @property
    def node2(self) -> AbstractNode:
        """Returns ``node2`` of the edge. If the edge is dangling, it is ``None``."""
        return self._nodes[1]

    @property
    def nodes(self) -> List[AbstractNode]:
        """Returns a list with ``node1`` and ``node2``."""
        return self._nodes

    @property
    def axis1(self) -> Axis:
        """Returns axis where the edge is attached to ``node1``."""
        return self._axes[0]

    @property
    def axis2(self) -> Axis:
        """
        Returns axis where the edge is attached to ``node2``. If the edge is
        dangling, it is ``None``.
        """
        return self._axes[1]

    @property
    def axes(self) -> List[Axis]:
        """
        Returns a list of axes where the edge is attached to ``node1`` and
        ``node2``, respectively.
        """
        return self._axes

    @property
    def name(self) -> Text:
        """
        Returns edge's name. It is formed with the corresponding nodes' and axes'
        names.

        Examples
        --------
        >>> nodeA = tk.Node(shape=(2, 3),
        ...                 name='nodeA',
        ...                 axes_names=['left', 'right'])
        >>> edge = nodeA['right']
        >>> print(edge.name)
        nodeA[right] <-> None

        >>> nodeB = tk.Node(shape=(3, 4),
        ...                 name='nodeB',
        ...                 axes_names=['left', 'right'])
        >>> _ = new_edge = nodeA['right'] ^ nodeB['left']
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
        return self._axes[0]._batch

    def is_attached_to(self, node: AbstractNode) -> bool:
        """Returns boolean indicating whether the edge is attached to ``node``."""
        return (self.node1 == node) or (self.node2 == node)

    def size(self) -> int:
        """Returns edge's size. Equivalent to node's shape in that axis."""
        return self._nodes[0]._shape[self._axes[0]._num]

    def change_size(self, size: int) -> None:
        """
        Changes size of the edge, thus changing the size of tensors of ``node1``
        and ``node2`` at the corresponding axes. If new size is smaller, the
        tensor will be cropped; if larger, the tensor will be expanded with zeros.
        In both cases, the process (cropping/expanding) occurs at the "right",
        "bottom", "back", etc. of each dimension.
        
        Parameters
        ----------
        size : int
            New size of the edge.
            
        Examples
        --------
        >>> nodeA = tk.ones((2, 3))
        >>> nodeB = tk.ones((3, 4))
        >>> _ = edge = nodeA[1] ^ nodeB[0]
        >>> edge.size()
        3
        
        >>> edge.change_size(4)
        >>> nodeA.tensor
        tensor([[1., 1., 1., 0.],
                [1., 1., 1., 0.]])
        
        >>> nodeB.tensor
        tensor([[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [0., 0., 0., 0.]])
                
        >>> edge.size()
        4
        
        >>> edge.change_size(2)
        >>> nodeA.tensor
        tensor([[1., 1.],
                [1., 1.]])
        
        >>> nodeB.tensor
        tensor([[1., 1., 1., 1.],
                [1., 1., 1., 1.]])
                
        >>> edge.size()
        2
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
        
        Examples
        --------
        >>> nodeA = tk.randn((2, 3))
        >>> nodeB = tk.randn((3, 4))
        >>> _ = edge = nodeA[1] ^ nodeB[0]
        >>> copy = edge.copy()
        >>> copy != edge
        True
        
        >>> copy.is_attached_to(nodeA)
        True
        
        >>> copy.is_attached_to(nodeB)
        True
        """
        new_edge = Edge(node1=self.node1, axis1=self.axis1,
                        node2=self.node2, axis2=self.axis2)
        return new_edge

    def connect(self, other: 'Edge') -> 'Edge':
        """
        Connects dangling edge to another dangling edge. It is necessary that
        both edges have the same size so that contractions along that edge can
        be computed.
        
        Note that this connectes edges from ``leaf`` (or ``data``, ``virtual``)
        nodes, but never from ``resultant`` nodes. If one tries to connect
        one of the inherited edges of a ``resultant`` node, the new connected
        edge will be attached to the original ``leaf`` nodes from which the
        ``resultant`` node inherited its edges. Hence, the ``resultant`` node
        will not "see" the connection until the :class:`TensorNetwork` is
        :meth:`~TensorNetwork.reset`.
        
        If the nodes that are being connected come from different networks, the
        ``node2`` (and its connected component) will be moved to ``node1``'s
        network. See also :meth:`~AbstractNode.move_to_network`.

        Parameters
        ----------
        other : Edge
            The other edge to which current edge will be connected.

        Returns
        -------
        Edge

        Examples
        --------
        To connect two edges, the overloaded operator ``^`` can also be used.

        >>> nodeA = tk.Node(shape=(2, 3),
        ...                 name='nodeA',
        ...                 axes_names=('left', 'right'))
        >>> nodeB = tk.Node(shape=(3, 4),
        ...                 name='nodeB',
        ...                 axes_names=('left', 'right'))
        >>> _ = new_edge = nodeA['right'] ^ nodeB['left']  # Same as .connect()
        >>> print(new_edge.name)
        nodeA[right] <-> nodeB[left]
        """
        return connect(self, other)

    def disconnect(self) -> Tuple['Edge', 'Edge']:
        """
        Disconnects connected edge, that is, the connected edge is split into
        two dangling edges, one for each node.

        Returns
        -------
        tuple[Edge, Edge]

        Examples
        --------
        To disconnect an edge, the overloaded operator ``|`` can also be used.

        >>> nodeA = tk.Node(shape=(2, 3),
        ...                 name='nodeA',
        ...                 axes_names=('left', 'right'))
        >>> nodeB = tk.Node(shape=(3, 4),
        ...                 name='nodeB',
        ...                 axes_names=('left', 'right'))
        >>> _ = new_edge = nodeA['right'] ^ nodeB['left']
        >>> new_edgeA, new_edgeB = new_edge | new_edge  # Same as .disconnect()
        >>> print(new_edgeA.name)
        nodeA[right] <-> None

        >>> print(new_edgeB.name)
        nodeB[left] <-> None
        """
        return disconnect(self)

    def __xor__(self, other: 'Edge') -> 'Edge':
        return self.connect(other)

    def __or__(self, other: 'Edge') -> Tuple['Edge', 'Edge']:
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


class StackEdge(Edge):  # MARK: StackEdge
    """
    Class for stacked edges. They are just like :class:`Edges <Edge>` but used
    when stacking a collection of nodes into a :class:`StackNode`. When doing
    this, all edges of the stacked nodes must be kept, since they have the
    information regarding the nodes' neighbours, which will be used when :func:
    `unbinding <unbind>` the stack.
    
    |

    Parameters
    ----------
    edges : list[Edge]
        List of edges (one from each node that is being stacked) that are
        attached to the equivalent of ``axis1`` in each node.
    node1_list : list[bool]
        List of ``axis1`` attributes (one from each node that is being stacked)
        of the equivalent of ``axis1`` in each node.
    node1 : StackNode or ParamStackNode
        First node to which the edge is connected.
    axis1: int, str or Axis
        Axis of ``node1`` where the edge is attached.
    node2 : StackNode or ParamStackNode, optional
        Second node to which the edge is connected. If ``None``, the edge will
        be dangling.
    axis2 : int, str, Axis, optional
        Axis of ``node2`` where the edge is attached.
    
    
    |
    """

    def __init__(self,
                 edges: List[Edge],
                 node1_list: List[bool],
                 node1: AbstractStackNode,
                 axis1: Axis,
                 node2: Optional[AbstractStackNode] = None,
                 axis2: Optional[Axis] = None) -> None:
        self._edges = edges
        self._node1_list = node1_list
        super().__init__(node1=node1, axis1=axis1,
                         node2=node2, axis2=axis2)

    @property
    def edges(self) -> List[Edge]:
        """Returns list of stacked edges corresponding to this axis."""
        return self._edges

    @property
    def node1_list(self) -> List[bool]:
        """Returns list of ``node1``'s corresponding to this axis."""
        return self._node1_list

    def connect(self, other: 'StackEdge') -> 'StackEdge':
        """
        Same as :meth:`~Edge.connect` but it is first verified that all stacked
        :meth:`edges` corresponding to both ``StackEdges`` are the same.
        
        That is, this is a redundant operation to **re-connect** a list of edges
        that should be already connected. However, this is mandatory, since when
        stacking two sequences of nodes independently it cannot be inferred that
        the resultant ``StackNodes`` had to be connected.

        Parameters
        ----------
        other : StackEdge
            The other edge to which current edge will be connected.

        Returns
        -------
        StackEdge

        Examples
        --------
        To connect two stack-edges, the overloaded operator ``^`` can also be
        used.

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
        >>> _ = new_edge = stack_nodes['input'] ^ stack_data['feature']
        >>> print(new_edge.name)
        stack_0[input] <-> stack_1[feature]
        """
        return connect_stack(self, other)


###############################################################################
#                               EDGE OPERATIONS                               #
###############################################################################
def connect(edge1: Edge, edge2: Edge) -> Edge:
    """
    Connects two dangling edges. It is necessary that both edges have the same
    size so that contractions along that edge can be computed.
    
    Note that this connectes edges from ``leaf`` (or ``data``, ``virtual``)
    nodes, but never from ``resultant`` nodes. If one tries to connect one of
    the inherited edges of a ``resultant`` node, the new connected edge will be
    attached to the original ``leaf`` nodes from which the ``resultant`` node
    inherited its edges. Hence, the ``resultant`` node will not "see" the
    connection until the :class:`TensorNetwork` is :meth:`~TensorNetwork.reset`.
    
    If the nodes that are being connected come from different networks, the
    ``node2`` (and its connected component) will be movec to ``node1``'s network.
    See also :meth:`~AbstractNode.move_to_network`.

    This operation is the same as :meth:`Edge.connect`.

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
    
    Examples
    --------
    >>> nodeA = tk.Node(shape=(2, 3),
    ...                 name='nodeA',
    ...                 axes_names=('left', 'right'))
    >>> nodeB = tk.Node(shape=(3, 4),
    ...                 name='nodeB',
    ...                 axes_names=('left', 'right'))
    >>> new_edge = tk.connect(nodeA['right'], nodeB['left'])
    >>> print(new_edge.name)
    nodeA[right] <-> nodeB[left]
    """
    if isinstance(edge1, StackEdge) or isinstance(edge2, StackEdge):
        raise TypeError('No edge should be StackEdge type. '
                        'Use connect_stack in that case')
    if not isinstance(edge1, Edge) or not isinstance(edge2, Edge):
        raise TypeError('Both edges should be Edge type')

    # Case edge is already connected
    if edge1 == edge2:
        return edge1

    for edge in [edge1, edge2]:
        if not edge.is_dangling():
            raise ValueError(f'Edge "{edge!s}" is not a dangling edge. '
                             f'This edge points to nodes: "{edge.node1!s}" and '
                             f'"{edge.node2!s}"')
        if edge.is_batch():
            raise ValueError(f'Edge "{edge!s}" is a batch edge. Batch edges '
                             'cannot be connected')

    if edge1.size() != edge2.size():
        raise ValueError(f'Cannot connect edges of unequal size. '
                         f'Size of edge "{edge1!s}": {edge1.size()}. '
                         f'Size of edge "{edge2!s}": {edge2.size()}')

    node1, axis1 = edge1.node1, edge1.axis1
    node2, axis2 = edge2.node1, edge2.axis1
    net1, net2 = node1._network, node2._network

    if net1 != net2:
        node2.move_to_network(net1)
    net1._remove_edge(edge1)
    net1._remove_edge(edge2)

    new_edge = Edge(node1=node1, axis1=axis1,
                    node2=node2, axis2=axis2)

    node1._add_edge(new_edge, axis1, True)
    node2._add_edge(new_edge, axis2, False)
    return new_edge


def connect_stack(edge1: StackEdge, edge2: StackEdge) -> StackEdge:
    """
    Same as :func:`connect` but it is first verified that all stacked edges
    corresponding to both ``StackEdges`` are the same. That is, this is a
    redundant operation to **re-connect** a list of edges that should be already
    connected. However, this is mandatory, since when stacking two sequences of
    nodes independently it cannot be inferred that the resultant ``StackNodes``
    had to be connected.

    This operation is the same as :meth:`StackEdge.connect`.

    Parameters
    ----------
    edge1 : StackEdge
        The first edge that will be connected. Its node will become the ``node1``
        of the resultant edge.
    edge2 : StackEdge
        The second edge that will be connected. Its node will become the ``node2``
        of the resultant edge.
    """
    if not isinstance(edge1, StackEdge) or not isinstance(edge2, StackEdge):
        raise TypeError('Both edges should be StackEdge type')

    if edge1._edges != edge2._edges:
        raise ValueError('Cannot connect stack edges whose lists of edges are '
                         'not the same. They will be the same when both lists '
                         'contain edges connecting the nodes that formed the '
                         'stack nodes.')
    # Case edge is already connected
    if edge1 == edge2:
        return edge1

    for edge in [edge1, edge2]:
        if not edge.is_dangling():
            raise ValueError(f'Edge "{edge!s}" is not a dangling edge. '
                             f'This edge points to nodes: "{edge.node1!s}" and '
                             f'"{edge.node2!s}"')

    node1, axis1 = edge1.node1, edge1.axis1
    node2, axis2 = edge2.node1, edge2.axis1
    net1, net2 = node1._network, node2._network

    net1._remove_edge(edge1)
    net1._remove_edge(edge2)

    new_edge = StackEdge(edges=edge1._edges,
                         node1_list=edge1._node1_list,
                         node1=node1, axis1=axis1,
                         node2=node2, axis2=axis2)

    node1._add_edge(new_edge, axis1, True)
    node2._add_edge(new_edge, axis2, False)
    return new_edge


def disconnect(edge: Union[Edge, StackEdge]) -> Union[Tuple[Edge, Edge],
                                                      Tuple[StackEdge, StackEdge]]:
    """
    Disconnects connected edge, that is, the connected edge is split into
    two dangling edges, one for each node.

    This operation is the same as :meth:`Edge.disconnect`.

    Parameters
    ----------
    edge : Edge or StackEdge
        Edge that is going to be disconnected (split in two).

    Returns
    -------
    tuple[Edge, Edge] or tuple[StackEdge, StackEdge]
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
    if isinstance(edge, StackEdge):
        for axis, node in zip(axes, nodes):
            new_edge = StackEdge(edges=edge._edges,
                                 node1_list=edge._node1_list,
                                 node1=node,
                                 axis1=axis)
            new_edges.append(new_edge)
    else:
        for axis, node in zip(axes, nodes):
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
class Successor:  # MARK: Successor
    """
    Class for successors. This is a sort of cache memory for :class:`Operations
    <Operation>` that have been already computed.

    For instance, when contracting two nodes, the result gives a new node that
    stores the tensor resultant from contracting both nodes' tensors. However,
    when training a :class:`TensorNetwork`, the tensors inside the nodes will
    change every epoch, but there is actually no need to create a new resultant
    node every time. Instead, it is more efficient to keep track of which node
    arose as the result of an operation, and simply change its tensor.

    Hence, a ``Successor`` is instantiated providing details to get the operand
    nodes' tensors, as well as a reference to the resultant node, and some hints
    that might help accelerating the computations the next time the operation
    is performed.
    
    |

    These properties can be accessed via ``successor.node_ref``,
    ``successor.index``, ``successor.child`` and ``successor.hints``.
    
    |
    
    See the different :class:`operations <Operation>` to learn which resultant
    node keeps the ``Successor`` information.

    Parameters
    ----------
    node_ref : Node, ParamNode, or list[Node, ParamNode]
        For the nodes that are involved in an operation, this are the
        corresponding nodes that store their tensors.
    index : list[int, slice] or list[list[int, slice]], optional
        For the nodes that are involved in an operation, this are the
        corresponding indices used to access their tensors.
    child : Node or list[Node]
        The node or list of nodes that result from an operation.
    hints : any, optional
        A dictionary of hints created the first time an operation is computed in
        order to save some computation in the next calls of the operation.

    Examples
    --------
    When contracting two nodes, a ``Successor`` is created and added to the list
    of successors of the first node (left operand).

    >>> nodeA = tk.randn(shape=(2, 3), axes_names=('left', 'right'))
    >>> nodeB = tk.randn(shape=(3, 4), axes_names=('left', 'right'))
    >>> _ = nodeA['right'] ^ nodeB['left']
    ...
    >>> # Contract nodes
    >>> result = nodeA @ nodeB
    >>> print(result.name)
    contract_edges
    
    >>> # To get a successor, the name of the operation and the arguments have
    >>> # to be provided as keys of the successors dictionary
    >>> nodeA.successors['contract_edges'][(None, nodeA, nodeB)].child == result
    True
    
    
    |
    """

    def __init__(self,
                 node_ref: Union[AbstractNode, List[AbstractNode]],
                 index: Union[Optional[List[Union[int, slice]]],
                              List[Optional[List[Union[int, slice]]]]],
                 child: Union[Node, List[Node]],
                 hints: Optional[Any] = None) -> None:
        self.node_ref = node_ref
        self.index = index
        self.child = child
        self.hints = hints


###############################################################################
#                                TENSOR NETWORK                               #
###############################################################################
class TensorNetwork(nn.Module):  # MARK: TensorNetwork
    """
    Class for arbitrary Tensor Networks. Subclass of **PyTorch**
    ``torch.nn.Module``.

    Tensor Networks are the central objects of **TensorKrowch**. Basically,
    a tensor network is a graph with vertices (:class:`Nodes <AbstractNode>`)
    connected by :class:`Edges <Edge>`. In these models, nodes' tensors will be
    trained so that the contraction of the whole network approximates a certain
    function. Hence, ``TensorNetwork``'s are the **trainable objects** of
    **TensorKrowch**, very much like ``torch.nn.Module``'s are the **trainable
    objects** of **PyTorch**.
    
    |

    Recall that the common way of defining models out of ``torch.nn.Module`` is
    by defining a subclass where the ``__init__`` and ``forward`` methods are
    overriden:

    * **__init__**: Defines the model itself (its layers, attributes, etc.).
    
    * **forward**: Defines the way the model operates, that is, how the different
      parts of the model might combine to get an output from a particular input.


    With ``TensorNetwork``, the workflow is similar, though there are other
    methods that should be overriden:
    
    * **__init__**: Defines the graph of the tensor network and initializes the
      tensors of the nodes. See :class:`AbstractNode` and :class:`Edge` to learn
      how to create nodes and connect them.
      
    * **set_data_nodes** (optional): Creates the data nodes where the data
      tensor(s) will be placed. Usually, it will just select the edges to which
      the ``data`` nodes should be connected, and call the parent method. See
      :meth:`set_data_nodes` to learn good practices to override it. See also
      :meth:`add_data`.
      
    * **add_data** (optional): Adds new data tensors that will be stored in
      ``data`` nodes. Usually it will not be necessary to override this method,
      but if one wants to customize how data is set into the ``data`` nodes,
      :meth:`add_data` can be overriden.
      
    * **contract**: Defines the contraction algorithm of the whole tensor network,
      thus returning a single node. Very much like ``forward`` this is the main
      method that describes how the components of the network are combined.
      Hence, in ``TensorNetwork`` the :meth:`forward` method shall not be
      overriden, since it will just call :meth:`set_data_nodes`, if needed,
      :meth:`add_data` and :meth:`contract` and then it will return the tensor
      corresponding to the last ``resultant`` node. Hence, the order in which
      ``Operations`` are called from ``contract`` is important. The last
      operation must be the one returning the final node.
      
    |

    Although one can define how the network is going to be contracted, there
    are a couple of modes that can change how this contraction behaves at a
    lower level:

    * **auto_stack** (``False`` by default): This mode indicates whether the
      operation :func:`stack` can take control of the memory management of the
      network to skip some steps in future computations. If ``auto_stack`` is
      set to ``True`` and a collection of :class:`ParamNodes <ParamNode>` are
      :func:`stacked <stack>` (as the first operation in which these nodes are
      involved), then those nodes will no longer store their own tensors,
      but rather a ``virtual`` :class:`ParamStackNode` will store the stacked
      tensor, avoiding the computation of that first :func:`stack` in every
      contraction. This behaviour is not possible if ``auto_stack`` is set to
      ``False``, in which case all nodes will always store their own tensors.
      
      Setting ``auto_stack`` to ``True`` will be faster for both **inference**
      and **training**. However, while experimenting with ``TensorNetwork``'s
      one might want that all nodes store their own tensors to avoid problems.

    * **auto_unbind** (``False`` by default): This mode indicates whether the
      operation :func:`unbind` has to actually `unbind` the stacked tensor or
      just generate a collection of references. That is, if ``auto_unbind`` is
      set to ``False``, :func:`unbind` creates a collection of nodes, each of
      them storing the corresponding slice of the stacked tensor. If
      ``auto_unbind`` is set to ``True``, :func:`unbind` just creates the nodes
      and gives each of them an index to reference the stacked tensor, so that
      each node's tensor would be retrieved by indexing the stack. This avoids
      performing the operation, since these indices will be the same in
      subsequent iterations.
      
      Setting ``auto_unbind`` to ``True`` will be faster for **inference**, but
      slower for **training**.

    Once the training algorithm starts, these modes should not be changed (very
    often at least), since changing them entails first :meth:`resetting <reset>`
    the whole network, which is a costly method.
    
    |

    When the ``TensorNetwork`` is defined, it has a bunch of ``leaf``, ``data``
    and ``virtual`` nodes that make up the network structure, each of them
    storing its own tensor. However, when the network is contracted, several
    ``resultant`` nodes become new members of the network, even modifying its
    memory (depending on the ``auto_stack`` and ``auto_unbind`` modes).
    
    Therefore, if one wants to :meth:`reset` the network to its initial state
    after performing some operations, all the ``resultant`` nodes should be
    deleted, and all the tensors should return to its nodes (each node stores
    its own tensor). This is exactly what :meth:`reset` does. Besides, since
    ``auto_stack`` and ``auto_unbind`` can change how the tensors are stored,
    if one wants to change these modes, the network should be first reset (this
    is already done automatically when changing the modes).
    
    See :class:`AbstractNode` to learn about the **4 excluding types** of nodes,
    and :meth:`reset` to learn about how these nodes are treated differently.
    
    |
    
    There are also some special nodes that one should take into account. These
    are specified by name. See :class:`AbstractNode` to learn about
    **reserved nodes' names**, and :meth:`reset` to learn about how these
    nodes are treated differently.
    
    |
    
    Other thing one must take into account is the naming of ``Nodes``. Since
    the name of a ``Node`` is used to access it from the ``TensorNetwork``, the
    same name cannot be used by more than one ``Node``. In that case, repeated
    names get an automatic enumeration of the form ``"name_{number}"`` (underscore
    followed by number).
    
    To add a custom enumeration to keep track of the nodes of the network in a
    user-defined way, one may use brackets or parenthesis: ``"name_({number})"``.
    
    |
    
    For an example, check this :ref:`tutorial <tutorial_5>`.
    
    |
    
    Parameters
    ----------
    name : str, optional
        Network's name. By default, it is the name of the class
        (e.g. ``"tensornetwork"``).
    
    
    |
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
        self._memory_nodes = dict()    # address -> memory
        self._inverse_memory = dict()  # address -> nodes using that memory

        # TN modes
        # Auto-management of memory mode
        self._auto_stack = True   # train -> True / eval -> True
        self._auto_unbind = False  # train -> False / eval -> True
        self._tracing = False      # Tracing mode (True while calling .trace())
        self._traced = False       # True if .trace() is called, False if reset()

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
        all_nodes_names += list(self._leaf_nodes.keys())
        all_nodes_names += list(self._data_nodes.keys())
        all_nodes_names += list(self._virtual_nodes.keys())
        all_nodes_names += list(self._resultant_nodes.keys())
        return all_nodes_names

    @property
    def leaf_nodes(self) -> Dict[Text, AbstractNode]:
        """Returns dictionary of ``leaf`` nodes of the network."""
        return self._leaf_nodes

    @property
    def data_nodes(self) -> Dict[Text, AbstractNode]:
        """Returns dictionary of ``data`` nodes of the network."""
        return self._data_nodes

    @property
    def virtual_nodes(self) -> Dict[Text, AbstractNode]:
        """Returns dictionary of ``virtual`` nodes of the network."""
        return self._virtual_nodes

    @property
    def resultant_nodes(self) -> Dict[Text, AbstractNode]:
        """Returns dictionary of ``resultant`` nodes of the network."""
        return self._resultant_nodes

    @property
    def edges(self) -> List[Edge]:
        """
        Returns list of dangling, non-batch edges of the network. Dangling
        edges from ``virtual`` nodes are not included.
        """
        return self._edges

    @property
    def auto_stack(self) -> bool:
        """
        Returns boolean indicating whether ``auto_stack`` mode is active. By
        default, it is ``True``.
        
        This mode indicates whether the operation :func:`stack` can take control
        of the memory management of the network to skip some steps in future
        computations. If ``auto_stack`` is set to ``True`` and a collection of
        :class:`ParamNodes <ParamNode>` are :func:`stacked <stack>` (as the
        first operation in which these nodes are involved), then those nodes
        will no longer store their own tensors, but rather a ``virtual``
        :class:`ParamStackNode` will store the stacked tensor, avoiding the
        computation of that first :func:`stack` in every contraction. This
        behaviour is not possible if ``auto_stack`` is set to ``False``, in
        which case all nodes will always store their own tensors.
        
        Setting ``auto_stack`` to ``True`` will be faster for both **inference**
        and **training**. However, while experimenting with ``TensorNetwork``'s
        one might want that all nodes store their own tensors to avoid problems.
        
        Be aware that changing ``auto_stack`` mode entails :meth:`resetting <reset>`
        the network, which will modify its nodes. This has to be done manually
        in order to avoid undesired behaviour.
        """
        return self._auto_stack

    @auto_stack.setter
    def auto_stack(self, set_mode: bool) -> None:
        if set_mode != self._auto_stack:
            self.reset()
            self._auto_stack = set_mode

    @property
    def auto_unbind(self) -> bool:
        """
        Returns boolean indicating whether ``auto_unbind`` mode is active. By
        default, it is ``False``.
        
        This mode indicates whether the operation :func:`unbind` has to actually
        `unbind` the stacked tensor or just generate a collection of references.
        That is, if ``auto_unbind`` is set to ``False``, :func:`unbind` creates
        a collection of nodes, each of them storing the corresponding slice of
        the stacked tensor. If ``auto_unbind`` is set to ``True``, :func:`unbind`
        just creates the nodes and gives each of them an index to reference the
        stacked tensor, so that each node's tensor would be retrieved by indexing
        the stack. This avoids performing the operation, since these indices
        will be the same in subsequent iterations.
      
        Setting ``auto_unbind`` to ``True`` will be faster for **inference**, but
        slower for **training**.
        
        Be aware that changing ``auto_unbind`` mode entails :meth:`resetting
        <reset>` the network, which will modify its nodes. Thus, this mode has
        to be changed manually in order to avoid undesired behaviour.
        """
        return self._auto_unbind

    @auto_unbind.setter
    def auto_unbind(self, set_mode: bool) -> None:
        if set_mode != self._auto_unbind:
            self.reset()
            self._auto_unbind = set_mode

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
        Adds an edge to the network. If it is a ``StackEdge``, it is not added,
        since those edges are a sort of `virtual` edges used for ``resultant``
        ``StackNodes``.

        Parameters
        ----------
        edge : Edge
            Edge to be added.
        """
        # StackEdges are ignores since these are edges for ``resultant`` nodes
        if not isinstance(edge, StackEdge):
            if edge.is_dangling() and not edge.is_batch() and \
                    (edge not in self._edges):
                self._edges.append(edge)

    def _remove_edge(self, edge: Edge) -> None:
        """
        Removes an edge from the network.

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
        Adds a node to the network. If it is a :class:`ParamNode`, its tensor
        (``torch.nn.Parameter``) becomes a parameter of the network.

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
        then removed. This is what :meth:`delete_node` does.

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
        To completely get rid of the node, do not forget to delete it:
        
        >>> del node
            
        or override it:
        
        >>> node = node.copy()  # .copy() calls to .delete_node()

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
            
        Examples
        --------
        >>> nodeA = tk.randn((2, 3))
        >>> nodeB = tk.randn((3, 4))
        >>> _ = nodeA[1] ^ nodeB[0]
        >>> print(nodeA.name, nodeB.name)
        node_0 node_1
        
        >>> nodeB.network.delete_node(nodeB)
        >>> nodeA.neighbours() == []
        True
        
        >>> print(nodeA.name)
        node
        
        If ``move_names`` is set to ``False``, enumeration is not removed.
        Useful to avoid managing enumeration of a list of nodes that are all
        going to be deleted.
        
        >>> nodeA = tk.randn((2, 3))
        >>> nodeB = tk.randn((3, 4))
        >>> _ = nodeA[1] ^ nodeB[0]
        >>> nodeB.network.delete_node(nodeB, False)
        >>> print(nodeA.name)
        node_0
        """
        node.disconnect()
        self._remove_node(node, move_names)
        node._temp_tensor = None
    
    def delete(self) -> None:
        for node in self.nodes.values():
            self.delete_node(node)

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
            if hasattr(self, 'param_' + node._name):
                delattr(self, 'param_' + node._name)
        for edge in node._edges:
            if edge.is_attached_to(node):
                self._remove_edge(edge)

        self._update_node_info(node, new_name)
        node._name = new_name

        if isinstance(node.tensor, Parameter):
            if node._tensor_info['address'] is not None:
                if not hasattr(self, 'param_' + node._name):
                    self.register_parameter('param_' + node._name,
                                            self._memory_nodes[node._name])
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
        in a user-defined way, one may use brackets or parenthesis.

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
            self._assign_node_name(node, node._name)
            raise ValueError(f'Node\'s name cannot be an operation name: '
                             f'{list(self.operations.keys())}')

        if non_enum_prev_name in self._repeated_nodes_names:
            count = self._repeated_nodes_names[non_enum_prev_name]
            if count == 1:
                aux_node = self.nodes[non_enum_prev_name]
                aux_new_name = '_'.join([non_enum_prev_name, '0'])
                self._update_node_name(aux_node, aux_new_name)
            new_name = '_'.join([non_enum_prev_name, str(count)])
        else:
            new_name = non_enum_prev_name
            self._repeated_nodes_names[non_enum_prev_name] = 0
        self._repeated_nodes_names[non_enum_prev_name] += 1

        # Since node name might change, edges should be removed and
        # added later, so that their names as submodules are correct
        for edge in node._edges:
            if edge.is_attached_to(node):
                self._remove_edge(edge)

        if first_time:
            nodes_dict = self._which_dict(node)
            nodes_dict[new_name] = node
            self._memory_nodes[new_name] = node._temp_tensor
            node._tensor_info = {'address': new_name,
                                 'node_ref': None,
                                 'index': None}
            node._temp_tensor = None
            node._network = self
            node._name = new_name
        else:
            self._update_node_info(node, new_name)
            node._name = new_name

        # Node is ParamNode and tensor is not None
        if isinstance(node.tensor, Parameter):
            if not hasattr(self, 'param_' + node._name,):
                self.register_parameter( 'param_' + node._name,
                                        self._memory_nodes[node._name])
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
            if hasattr(self, 'param_' + node._name):
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
                    aux_prev_name = '_'.join([non_enum_prev_name, str(i)])
                    aux_new_name = '_'.join([non_enum_prev_name, str(i - 1)])
                    aux_node = self.nodes[aux_prev_name]
                    self._update_node_name(aux_node, aux_new_name)

        self._repeated_nodes_names[non_enum_prev_name] -= 1
        count -= 1
        if count == 0:
            del self._repeated_nodes_names[non_enum_prev_name]
        elif count == 1:
            if move_names:
                aux_prev_name = '_'.join([non_enum_prev_name, '0'])
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
        Parameterizes all ``leaf`` nodes of the network. If there are
        ``resultant`` nodes in the :class:`TensorNetwork`, it will be first
        :meth:`reset`.

        Parameters
        ----------
        set_param : bool
            Boolean indicating whether the tensor network has to be parameterized
            (``True``) or de-parameterized (``False``).
        override : bool
            Boolean indicating whether the tensor network should be parameterized
            in-place (``True``) or copied and then parameterized (``False``).
        """
        if self._resultant_nodes:
            warnings.warn(
                'Resultant nodes will be removed before parameterizing the TN')
            self.reset()

        if override:
            net = self
        else:
            net = self.copy()

        for node in list(net._leaf_nodes.values()):
            node.parameterize(set_param)

        return net

    def set_data_nodes(self,
                       input_edges: List[Edge],
                       num_batch_edges: int) -> None:
        """
        Creates ``data`` nodes with as many batch edges as ``num_batch_edges``
        and one feature edge. Then it connects each of these nodes' feature
        edges to an edge from the list ``input_edges`` (following the provided
        order). Thus, edges in ``input_edges`` need to be dangling. Also, if
        there are already ``data`` nodes (or the ``"stack_data_memory"``) in
        the network, they should be :meth:`unset` first.

        If all the ``data`` nodes have the same shape, a ``virtual`` node will
        contain all the tensors stacked in one, what will save some memory
        and time in computations. This node is ``"stack_data_memory"``. See
        :class:`AbstractNode` to learn more about this node.
        
        If this method is overriden in subclasses, it can be done in two
        flavours:
        
        ::
        
            def set_data_nodes(self):
                # Collect input edges
                input_edges = [node_1[i], ..., node_n[j]]
                
                # Define number of batches
                num_batch_edges = m
                
                # Call parent method
                super().set_data_nodes(input_edges, num_batch_edges)
                
        ::
        
            def set_data_nodes(self):
                # Create data nodes directly
                data_nodes = [
                    tk.Node(shape=(batch_1, ..., batch_m, feature_dim),
                                   axes_names=('batch_1', ..., 'batch_m', 'feature')
                                   network=self,
                                   data=True)
                    for _ in range(n)]
                    
                # Connect them with the leaf nodes
                for i, data_node in enumerate(data_nodes):
                    data_node['feature'] ^ self.my_nodes[i]['input']
                    
        If this method is overriden, there is no need to call it explicitly
        during training, since it will be done in the :meth:`forward` call.
        
        On the other hand, if one does not override ``set_data_nodes``, it
        should be called before starting training.

        Parameters
        ----------
        input_edges : list[Edge]
            List of edges to which the ``data`` nodes' feature edges will be
            connected.
        num_batch_edges : int
            Number of batch edges in the ``data`` nodes.
            
        Examples
        --------
        >>> nodeA = tk.Node(shape=(2, 5, 2),
        ...                 axes_names=('left', 'input', 'right'),
        ...                 name='nodeA',
        ...                 init_method='randn')
        >>> nodeB = tk.Node(shape=(2, 5, 2),
        ...                 axes_names=('left', 'input', 'right'),
        ...                 name='nodeB',
        ...                 init_method='randn')
        >>> _ = nodeA['right'] ^ nodeB['left']
        ...
        >>> net = nodeA.network
        >>> input_edges = [nodeA['input'], nodeB['input']]
        >>> net.set_data_nodes(input_edges, 1)
        >>> list(net.data_nodes.keys())
        ['data_0', 'data_1']
        
        >>> net['data_0']
        Node(
            name: data_0
            tensor:
                    None
            axes:
                    [batch
                     feature]
            edges:
                    [data_0[batch] <-> None
                     data_0[feature] <-> nodeA[input]])
        """
        if not input_edges:
            raise ValueError(
                '`input_edges` is empty. '
                'Cannot set data nodes if no edges are provided')
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
                # If `same_dim`, all input_edges have the same feature dimension

                # 'n_features' is still in the first position because it is
                # easier to index just the first position of the tensor,
                # provided there might be several batch edges
                stack_node = Node(shape=(len(input_edges),
                                         *([1] * num_batch_edges),
                                         input_edges[0].size()),
                                  axes_names=('n_features',
                                              *(['batch'] * num_batch_edges),
                                              'feature'),
                                  name='stack_data_memory',
                                  network=self,
                                  virtual=True)
            else:
                raise ValueError(
                    'Tensor network already has "stack_data_memory" node. '
                    'Data nodes should be unset in order to set new ones')

        data_nodes = []
        for i, edge in enumerate(input_edges):
            if isinstance(edge, int):
                edge = self[edge]
            elif isinstance(edge, Edge):
                if edge not in self._edges:
                    raise ValueError(
                        f'Edge "{edge!r}" should be a dangling edge of the '
                        'Tensor Network')
            else:
                raise TypeError(
                    '`input_edges` should be list[int] or list[Edge] type')

            node = Node(shape=(*([1] * num_batch_edges), edge.size()),
                        axes_names=(*(['batch'] * num_batch_edges), 'feature'),
                        name='data',
                        network=self,
                        data=True)
            node['feature'] ^ edge
            data_nodes.append(node)

        if same_dim:
            # Use "stack_data_memory" tensor for all data nodes
            for i, node in enumerate(data_nodes):
                del self._memory_nodes[node._tensor_info['address']]
                node._tensor_info['address'] = None
                node._tensor_info['node_ref'] = stack_node
                node._tensor_info['index'] = i

    def unset_data_nodes(self) -> None:
        """
        Deletes all ``data`` nodes (including the ``"stack_data_memory"`` when
        this node exists).
        """
        if self._data_nodes:
            for node in list(self._data_nodes.values()):
                self.delete_node(node, False)
            self._data_nodes = dict()

            if 'stack_data_memory' in self._virtual_nodes:
                self.delete_node(self._virtual_nodes['stack_data_memory'])

    def add_data(self, data: Union[Tensor, Sequence[Tensor]]) -> None:
        r"""
        Adds data tensor(s) to ``data`` nodes, that is, changes their tensors
        by new data tensors when a new batch is provided.
        
        If all data nodes have the same shape, thus having its tensor stored in
        ``"stack_data_memory"``, the whole data tensor will be stored by this
        node. The ``data`` nodes will just store a reference to a slice of that
        tensor.
        
        Otherwise, each tensor in the list (``data``) will be stored by each
        ``data`` node in the network, in the order they appear in
        :meth:`data_nodes`.
        
        If one wants to customize how data is set into the ``data`` nodes, this
        method can be overriden.

        Parameters
        ----------
        data : torch.Tensor or list[torch.Tensor]
            If all data nodes have the same shape, thus having its tensor stored
            in ``"stack_data_memory"``, ``data`` should be a tensor of shape
            
            .. math::
                batch\_size_{0} \times ... \times batch\_size_{n} \times
                n_{features} \times feature\_dim
                
            Otherwise, it should be a list with :math:`n_{features}` elements,
            each of them being a tensor with shape
            
            .. math::
                batch\_size_{0} \times ... \times batch\_size_{n} \times
                feature\_dim
                
        Examples
        --------
        >>> nodeA = tk.Node(shape=(3, 5, 3),
        ...                 axes_names=('left', 'input', 'right'),
        ...                 name='nodeA',
        ...                 init_method='randn')
        >>> nodeB = tk.Node(shape=(3, 5, 3),
        ...                 axes_names=('left', 'input', 'right'),
        ...                 name='nodeB',
        ...                 init_method='randn')
        >>> _ = nodeA['right'] ^ nodeB['left']
        ...
        >>> net = nodeA.network
        >>> input_edges = [nodeA['input'], nodeB['input']]
        >>> net.set_data_nodes(input_edges, 1)
        ...
        >>> net.add_data(torch.randn(100, 2, 5))
        >>> net['data_0'].shape
        torch.Size([100, 5])
        """

        stack_node = self._virtual_nodes.get('stack_data_memory')

        if stack_node is not None:
            if isinstance(data, Tensor):
                data = data.movedim(-2, 0)
            else:
                data = torch.stack(data, dim=0)
            stack_node.tensor = data
            for i, data_node in enumerate(list(self._data_nodes.values())):
                data_node._shape = data[i].shape
        elif self._data_nodes:
            for i, data_node in enumerate(list(self._data_nodes.values())):
                data_node.tensor = data[i]
        else:
            raise ValueError('Cannot add data if no data nodes are set')

    def reset(self):
        """
        Resets the :class:`TensorNetwork` to its initial state, before computing
        any non-in-place :class:`Operation`. Different actions apply to different
        types of nodes:
        
        * ``leaf``: These nodes retrieve their tensors in case they were just
          referencing a slice of the tensor in the :class:`ParamStackNode` that
          is created when :func:`stacking <stack>` :class:`ParamNodes <ParamNode>`
          (if ``auto_stack`` mode is active). If there is a ``"virtual_uniform"``
          node in the network from which all ``leaf`` nodes take their tensor,
          this is not modified.
          
        * ``virtual``: Only virtual nodes created in :class:`operations
          <Operation>` are :meth:`deleted <delete_node>`. This only includes
          nodes using the reserved name ``"virtual_result"``.
          
        * ``resultant``: These nodes are :meth:`deleted <delete_node>` from the
          network.
        
        Also, the dictionaries of :class:`Successors <Successor>` of all ``leaf``
        and ``data`` nodes are emptied.
        
        The :class:`TensorNetwork` is automatically ``reset`` when
        :meth:`parameterizing <parameterize>` it, changing :meth:`auto_stack`
        or :meth:`auto_unbind` modes, or :meth:`tracing <trace>`.
        
        See :class:`AbstractNode` to learn more about the **4 types** of nodes
        and the **reserved names**.
        
        For an example, check this :ref:`tutorial <tutorial_5>`.
        """
        self._traced = False
        self._seq_ops = []
        self._inverse_memory = dict()

        if self._resultant_nodes or self._virtual_nodes:
            aux_dict = dict()
            aux_dict.update(self._leaf_nodes)
            aux_dict.update(self._resultant_nodes)
            aux_dict.update(self._virtual_nodes)
            for node in aux_dict.values():
                if node._virtual and ('virtual_result' not in node._name):
                    # Virtual nodes named "virtual_result" are nodes that are
                    # required in some situations during contraction, like
                    # ParamStackNodes
                    # This condition is satisfied by the rest of virtual nodes
                    continue

                node._successors = dict()

                node_ref = node._tensor_info['node_ref']
                if node_ref is not None:
                    if node_ref._virtual and ('virtual_uniform' in node_ref._name):
                        # Virtual nodes named "virtual_uniform" are ParamNodes
                        # whose tensor is shared across all the nodes in a
                        # uniform tensor network
                        continue

                # Store tensor as temporary
                node._temp_tensor = node.tensor
                node._tensor_info['address'] = node._name
                node._tensor_info['node_ref'] = None
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
                if node._virtual and ('virtual_result' not in node._name):
                    # This condition is satisfied by the rest of virtual nodes
                    continue
                self.delete_node(node, False)

    @torch.no_grad()
    def trace(self, example: Optional[Tensor] = None, *args, **kwargs) -> None:
        """
        Traces the tensor network contraction algorithm with two purposes:

        * Create all the intermediate ``resultant`` nodes that result from
          :class:`Operations <Operation>` so that in the next contractions only
          the tensor operations have to be computed, thus saving a lot of time.

        * Keep track of the tensors that are used to compute operations, so that
          intermediate results that are not useful anymore can be deleted, thus
          saving a lot of memory. This is achieved by constructing an
          ``inverse_memory`` that, given a memory address, stores the nodes that
          use the tensor located in that address of the network's memory.

        To trace a tensor network, it is necessary to provide the same arguments
        that would be required in the forward call. In case the tensor network
        is contracted with some input data, an example tensor with batch dimension
        1 and filled with zeros would be enough to trace the contraction.
        
        For an example, check this :ref:`tutorial <tutorial_5>`.

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
        
        self._tracing = True
        self(example, *args, **kwargs)
        self._tracing = False
        self(example, *args, **kwargs)
        self._traced = True

    def contract(self) -> Node:
        """
        Contracts the whole tensor network returning a single :class:`Node`.
        This method is not implemented and subclasses of :class:`TensorNetwork`
        should override it to define the contraction algorithm of the network.
        """
        # Custom, optimized contraction methods should be defined for each new
        # subclass of TensorNetwork
        raise NotImplementedError(
            'Contraction methods not implemented for generic TensorNetwork class')

    def forward(self,
                data: Optional[Union[Tensor, Sequence[Tensor]]] = None,
                *args,
                **kwargs) -> Tensor:
        r"""
        Contracts :class:`TensorNetwork` with input data. It can be called using
        the ``__call__`` operator ``()``.

        Overrides the ``forward`` method of **PyTorch**'s ``torch.nn.Module``.
        Sets data nodes automatically whenever :meth:`set_data_nodes` is
        overriden, :meth:`adds data <add_data>` tensor(s) to these nodes, and
        contracts the whole network according to :meth:`contract`, returning a
        single ``torch.Tensor``.
        
        Furthermore, to optimize the contraction algorithm during training,
        once the :class:`TensorNetwork` is :meth:`traced <trace>`, all that
        ``forward`` does is calling the different :class:`Operations <Operation>`
        used in :meth:`contract` in the same order they appeared in the code.
        Hence, the **last operation** in :meth:`contract` should be the one that
        **returns the single output** :class:`Node`.
        
        For an example, check this :ref:`tutorial <tutorial_5>`.

        Parameters
        ----------
        data : torch.Tensor or list[torch.Tensor], optional
            If all data nodes have the same shape, thus having its tensor stored
            in ``"stack_data_memory"``, ``data`` should be a tensor of shape
            
            .. math::
                batch\_size_{0} \times ... \times batch\_size_{n} \times
                n_{features} \times feature\_dim
                
            Otherwise, it should be a list with :math:`n_{features}` elements,
            each of them being a tensor with shape
            
            .. math::
                batch\_size_{0} \times ... \times batch\_size_{n} \times
                feature\_dim

            Also, it is not necessary that the network has ``data`` nodes, thus
            ``None`` is also valid.
        args :
            Arguments that might be used in :meth:`contract`.
        kwargs :
            Keyword arguments that might be used in :meth:`contract`.
        """
        if data is not None:
            if not self._data_nodes:
                try:
                    self.set_data_nodes()
                except TypeError:
                    raise TypeError(
                        'set_data_nodes missing 2 required positional arguments:'
                        ' `input_edges` and `num_batch_edges`. Override method'
                        ' with no arguments in subclasses of TensorNetwork or '
                        'call set_data_nodes explicitly before forward')
            self.add_data(data=data)

        if not self._resultant_nodes:
            output = self.contract(*args, **kwargs)
            return output.tensor

        else:
            output = list(map(lambda op: self.operations[op[0]](*op[1]),
                              self._seq_ops))[-1]

            if not isinstance(output, Node):
                if (self._seq_ops[-1][0] == 'unbind') and (len(output) == 1):
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
                    f'Tensor network "{self!s}" does not have any node with '
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
