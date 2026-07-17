"""
This script contains:
    * MPS:
        + UMPS
        + MPSLayer
        + UMPSLayer
    * AbstractConvClass:
        + ConvMPS
        + ConvUMPS
        + ConvMPSLayer
        + ConvUMPSLayer
"""

import warnings
from abc import abstractmethod, ABC
from typing import (Callable, List, Optional, Sequence,
                    Text, Tuple, Union)

from math import sqrt

import torch
import torch.nn as nn

import tensorkrowch.operations as op
from tensorkrowch.components import AbstractNode, Node, ParamNode
from tensorkrowch.components import TensorNetwork
from tensorkrowch.models import MPO, UMPO
from tensorkrowch.embeddings import basis
from tensorkrowch.utils import split_sequence_into_regions, random_unitary


class MPS(TensorNetwork):  # MARK: MPS
    """
    Class for Matrix Product States. This is the base class from which
    :class:`UMPS`, :class:`MPSLayer` and :class:`UMPSLayer` inherit.
    
    Matrix Product States are formed by:
    
    * ``mats_env``: Environment of `matrix` nodes with axes
      ``("left", "input", "right")``.
    
    * ``left_node``, ``right_node``: `Vector` nodes with axes ``("right",)``
      and ``("left",)``, respectively. These are used to close the boundary
      in the case ``boundary`` is ``"obc"``. Otherwise, both are ``None``.
    
    The base ``MPS`` class enables setting various nodes as either input or
    output nodes. This feature proves useful when computing marginal or
    conditional distributions. The assignment of roles can be altered
    dynamically, allowing input nodes to transition to output nodes, and vice
    versa.
    
    Input nodes will be connected to data nodes at their ``"input"`` edges, and
    contracted against them when calling :meth:`contract`. Output nodes, on the
    other hand, will remain disconnected. If ``marginalize_output = True`` in
    :meth:`contract`, the open indices of the output nodes can be marginalized
    so that the output is a single scalar (or a vector with only batch
    dimensions). If ``marginalize_output = False`` the result will be a tensor
    with as many dimensions as output nodes where in the MPS, plus the
    corresponding batch dimensions.
    
    If all input nodes have the same physical dimensions, the input data tensor
    can be passed as a single tensor. Otherwise, it would have to be passed as
    a list of tensors with different sizes.

    Parameters
    ----------
    n_features : int, optional
        Number of nodes that will be in ``mats_env``. That is, number of nodes
        without taking into account ``left_node`` and ``right_node``.
    phys_dim : int, list[int] or tuple[int], optional
        Physical dimension(s). If given as a sequence, its length should be
        equal to ``n_features``.
    bond_dim : int, list[int] or tuple[int], optional
        Bond dimension(s). If given as a sequence, its length should be equal
        to ``n_features`` (if ``boundary = "pbc"``) or ``n_features - 1`` (if
        ``boundary = "obc"``). The i-th bond dimension is always the dimension
        of the right edge of the i-th node.
    boundary : {"obc", "pbc"}
        String indicating whether periodic or open boundary conditions should
        be used.
    tensors: list[torch.Tensor] or tuple[torch.Tensor], optional
        Instead of providing ``n_features``, ``phys_dim``, ``bond_dim`` and
        ``boundary``, a list of MPS tensors can be provided. In such case, all
        mentioned attributes will be inferred from the given tensors. All
        tensors should be rank-3 tensors, with shape ``(bond_dim, phys_dim,
        bond_dim)``. If the first and last elements are rank-2 tensors, with
        shapes ``(phys_dim, bond_dim)``, ``(bond_dim, phys_dim)``, respectively,
        the inferred boundary conditions will be "obc". Also, if ``tensors``
        contains a single element, it can be rank-1 ("obc") or rank-3 ("pbc").
    in_features: list[int] or tuple[int], optional
        List of indices indicating the positions of the MPS nodes that will be
        considered as input nodes. These nodes will have a neighbouring data
        node connected to its ``"input"`` edge when the :meth:`set_data_nodes`
        method is called. ``in_features`` is the complementary set of
        ``out_features``, so it is only required to specify one of them.
    out_features: list[int] or tuple[int], optional
        List of indices indicating the positions of the MPS nodes that will be
        considered as output nodes. These nodes will be left with their ``"input"``
        edges open when contrating the network. If ``marginalize_output`` is
        set to ``True`` in :meth:`contract`, the network will be connected to
        itself at these nodes, and contracted. ``out_features`` is the
        complementary set of ``in_features``, so it is only required to specify
        one of them.
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
        Initialization method. Check :meth:`initialize` for a more detailed
        explanation of the different initialization methods.
    parameterized : bool, optional
        Boolean indicating whether MPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
    Examples
    --------
    ``MPS`` with the same physical dimensions:
    
    >>> mps = tk.models.MPS(n_features=5,
    ...                     phys_dim=2,
    ...                     bond_dim=5)
    >>> data = torch.ones(20, 5, 2) # batch_size x n_features x feature_size
    >>> result = mps(data)
    >>> result.shape
    torch.Size([20])
    
    ``MPS`` with different physical dimensions:
    
    >>> mps = tk.models.MPS(n_features=5,
    ...                     phys_dim=list(range(2, 7)),
    ...                     bond_dim=5)
    >>> data = [torch.ones(20, i)
    ...         for i in range(2, 7)] # n_features * [batch_size x feature_size]
    >>> result = mps(data)
    >>> result.shape
    torch.Size([20])
    
    ``MPS`` can also be initialized from a list of tensors:
    
    >>> tensors = [torch.randn(5, 2, 5) for _ in range(10)]
    >>> mps = tk.models.MPS(tensors=tensors)
    
    If ``in_features``/``out_features`` are specified, data will only be
    connected to the input nodes, leaving output nodes open:
    
    >>> mps = tk.models.MPS(tensors=tensors,
    ...                     out_features=[0, 3, 9])
    >>> data = torch.ones(20, 7, 2) # batch_size x n_features x feature_size
    >>> result = mps(data)
    >>> result.shape
    torch.Size([20, 2, 2, 2])
    
    >>> mps.reset()
    >>> result = mps(data, marginalize_output=True)
    >>> result.shape
    torch.Size([20, 20])
    """

    def __init__(self,
                 n_features: Optional[int] = None,
                 phys_dim: Optional[Union[int, Sequence[int]]] = None,
                 bond_dim: Optional[Union[int, Sequence[int]]] = None,
                 boundary: Text = 'obc',
                 tensors: Optional[Sequence[torch.Tensor]] = None,
                 in_features: Optional[Sequence[int]] = None,
                 out_features: Optional[Sequence[int]] = None,
                 n_batches: int = 1,
                 init_method: Text = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs) -> None:

        super().__init__(name='mps')
        
        if tensors is None:
            # boundary
            if boundary not in ['obc', 'pbc']:
                raise ValueError('`boundary` should be one of "obc" or "pbc"')
            self._boundary = boundary

            # n_features
            if not isinstance(n_features, int):
                raise TypeError('`n_features` should be int type')
            elif n_features < 1:
                raise ValueError('`n_features` should be at least 1')
            self._n_features = n_features

            # phys_dim
            if isinstance(phys_dim, Sequence):
                if len(phys_dim) != n_features:
                    raise ValueError('If `phys_dim` is given as a sequence of int, '
                                     'its length should be equal to `n_features`')
                self._phys_dim = list(phys_dim)
            elif isinstance(phys_dim, int):
                self._phys_dim = [phys_dim] * n_features
            else:
                raise TypeError('`phys_dim` should be int, tuple[int] or list[int] '
                                'type')

            # bond_dim
            if isinstance(bond_dim, Sequence):
                if boundary == 'obc':
                    if len(bond_dim) != n_features - 1:
                        raise ValueError(
                            'If `bond_dim` is given as a sequence of int, and '
                            '`boundary` is "obc", its length should be equal '
                            'to `n_features` - 1')
                elif boundary == 'pbc':
                    if len(bond_dim) != n_features:
                        raise ValueError(
                            'If `bond_dim` is given as a sequence of int, and '
                            '`boundary` is "pbc", its length should be equal '
                            'to `n_features`')
                self._bond_dim = list(bond_dim)
            elif isinstance(bond_dim, int):
                if boundary == 'obc':
                    self._bond_dim = [bond_dim] * (n_features - 1)
                elif boundary == 'pbc':
                    self._bond_dim = [bond_dim] * n_features
            else:
                raise TypeError('`bond_dim` should be int, tuple[int] or list[int]'
                                ' type')
        
        else:
            self._n_features, self._phys_dim, self._bond_dim, self._boundary = \
                self._infer_shape_from_tensors(tensors)
        
        # in_features and out_features
        if in_features is None:
            if out_features is None:
                # By default, all nodes are input nodes
                self._in_features = list(range(self._n_features))
                self._out_features = []
            else:
                if isinstance(out_features, (list, tuple)):
                    for out_f in out_features:
                        if not isinstance(out_f, int):
                            raise TypeError('`out_features` should be tuple[int]'
                                            ' or list[int] type')
                        if (out_f < 0) or (out_f >= self._n_features):
                            raise ValueError('Elements of `out_features` should'
                                             ' be between 0 and (`n_features` - 1)')
                    out_features = set(out_features)
                    in_features = set(range(self._n_features)).difference(out_features)
                    
                    self._in_features = list(in_features)
                    self._out_features = list(out_features)
                    
                    self._in_features.sort()
                    self._out_features.sort()
                else:
                    raise TypeError('`out_features` should be tuple[int]'
                                    ' or list[int] type')
        else:
            if isinstance(in_features, (list, tuple)):
                for in_f in in_features:
                    if not isinstance(in_f, int):
                        raise TypeError('`in_features` should be tuple[int]'
                                        ' or list[int] type')
                    if (in_f < 0) or (in_f >= self._n_features):
                        raise ValueError('Elements in `in_features` should'
                                         'be between 0 and (`n_features` - 1)')
                in_features = set(in_features)
            else:
                raise TypeError('`in_features` should be tuple[int]'
                                ' or list[int] type')
                    
            if out_features is None:
                out_features = set(range(self._n_features)).difference(in_features)
                
                self._in_features = list(in_features)
                self._out_features = list(out_features)
                
                self._in_features.sort()
                self._out_features.sort()
            else:
                out_features = set(out_features)
                union = in_features.union(out_features)
                inter = in_features.intersection(out_features)
                
                if (union == set(range(self._n_features))) and (inter == set([])):
                    self._in_features = list(in_features)
                    self._out_features = list(out_features)
                    
                    self._in_features.sort()
                    self._out_features.sort()
                else:
                    raise ValueError(
                        'If both `in_features` and `out_features` are provided,'
                        ' they should be complementary. That is, the union should'
                        ' be the total range 0, ..., (`n_features` - 1), and '
                        'the intersection should be empty')
        
        # n_batches
        if not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        self._n_batches = n_batches

        if not isinstance(parameterized, bool):
            raise TypeError('`parameterized` should be bool type')
        
        # Properties
        self._left_node = None
        self._right_node = None
        self._mats_env = []

        # Create Tensor Network
        self._make_nodes(parameterized)
        self.initialize(tensors=tensors,
                        init_method=init_method,
                        device=device,
                        dtype=dtype,
                        **kwargs)
    
    # ----------
    # Properties
    # ----------
    @property
    def n_features(self) -> int:
        """Returns number of nodes."""
        return self._n_features

    @property
    def phys_dim(self) -> List[int]:
        """Returns physical dimensions."""
        return self._phys_dim

    @property
    def bond_dim(self) -> List[int]:
        """Returns bond dimensions."""
        return self._bond_dim
    
    @property
    def boundary(self) -> Text:
        """Returns boundary condition ("obc" or "pbc")."""
        return self._boundary

    @property
    def n_batches(self) -> int:
        """
        Returns number of batch edges of the ``data`` nodes. To change this
        attribute, first call :meth:`~tensorkrowch.TensorNetwork.unset_data_nodes`
        if there are already data nodes in the network.
        """
        return self._n_batches

    @n_batches.setter
    def n_batches(self, n_batches: int) -> None:
        if n_batches != self._n_batches:
            if self._data_nodes:
                raise ValueError(
                    '`n_batches` cannot be changed if the MPS has data nodes. '
                    'Use unset_data_nodes first')
            elif not isinstance(n_batches, int):
                raise TypeError('`n_batches` should be int type')
            self._n_batches = n_batches
    
    @property
    def in_features(self) -> List[int]:
        """
        Returns list of positions of the input nodes. To change this
        attribute, first call :meth:`~tensorkrowch.TensorNetwork.unset_data_nodes`
        if there are already data nodes in the network. When changing it,
        :attr:`out_features` will change accordingly to be the complementary.
        """
        return self._in_features
    
    @in_features.setter
    def in_features(self, in_features) -> None:
        if self._data_nodes:
            raise ValueError(
                '`in_features` cannot be changed if the MPS has data nodes. '
                'Use unset_data_nodes first')
                
        if isinstance(in_features, (list, tuple)):
            for in_f in in_features:
                if not isinstance(in_f, int):
                    raise TypeError('`in_features` should be tuple[int]'
                                    ' or list[int] type')
                if (in_f < 0) or (in_f >= self._n_features):
                    raise ValueError('Elements in `in_features` should'
                                     'be between 0 and (`n_features` - 1)')
            in_features = set(in_features)
            out_features = set(range(self._n_features)).difference(in_features)
                
            self._in_features = list(in_features)
            self._out_features = list(out_features)
            
            self._in_features.sort()
            self._out_features.sort()
        else:
            raise TypeError(
                '`in_features` should be tuple[int] or list[int] type')
    
    @property
    def out_features(self) -> List[int]:
        """
        Returns list of positions of the output nodes. To change this
        attribute, first call :meth:`~tensorkrowch.TensorNetwork.unset_data_nodes`
        if there are already data nodes in the network. When changing it,
        :attr:`in_features` will change accordingly to be the complementary.
        """
        return self._out_features
    
    @out_features.setter
    def out_features(self, out_features) -> None:
        if self._data_nodes:
                raise ValueError(
                    '`out_features` cannot be changed if the MPS has data nodes. '
                    'Use unset_data_nodes first')
                
        if isinstance(out_features, (list, tuple)):
            for out_f in out_features:
                if not isinstance(out_f, int):
                    raise TypeError('`out_features` should be tuple[int]'
                                    ' or list[int] type')
                if (out_f < 0) or (out_f >= self._n_features):
                    raise ValueError('Elements in `out_features` should'
                                        'be between 0 and (`n_features` - 1)')
            out_features = set(out_features)
            in_features = set(range(self._n_features)).difference(out_features)
                
            self._in_features = list(in_features)
            self._out_features = list(out_features)
            
            self._in_features.sort()
            self._out_features.sort()
        else:
            raise TypeError(
                '`out_features` should be tuple[int] or list[int] type')
            
    @property
    def in_regions(self) -> List[List[int]]:
        """ Returns a list of lists of consecutive input positions."""
        return split_sequence_into_regions(self._in_features)
    
    @property
    def out_regions(self) -> List[List[int]]:
        """ Returns a list of lists of consecutive output positions."""
        return split_sequence_into_regions(self._out_features)
    
    @property
    def left_node(self) -> Optional[AbstractNode]:
        """Returns the ``left_node``."""
        return self._left_node
    
    @property
    def right_node(self) -> Optional[AbstractNode]:
        """Returns the ``right_node``."""
        return self._right_node
    
    @property
    def mats_env(self) -> List[AbstractNode]:
        """Returns the list of nodes in ``mats_env``."""
        return self._mats_env
    
    @property
    def in_env(self) -> List[AbstractNode]:
        """Returns the list of input nodes."""
        return [self._mats_env[i] for i in self._in_features]
    
    @property
    def out_env(self) -> List[AbstractNode]:
        """Returns the list of output nodes."""
        return [self._mats_env[i] for i in self._out_features]

    @property
    def tensors(self) -> List[torch.Tensor]:
        """Returns the list of MPS tensors."""
        mps_tensors = [node.tensor for node in self._mats_env]
        if self._boundary == 'obc':
            mps_tensors[0] = torch.einsum('l,lir->ir',
                                          self._left_node.tensor,
                                          mps_tensors[0])
            mps_tensors[-1] = torch.einsum('lir,r->li',
                                           mps_tensors[-1],
                                           self._right_node.tensor)
        return mps_tensors

    # -------
    # Methods
    # -------
    @staticmethod
    def _infer_shape_from_tensors(
            tensors: Sequence[torch.Tensor]
            ) -> Tuple[int, List[int], List[int], Text]:
        """Infers MPS metadata from a sequence of tensors."""
        if not isinstance(tensors, Sequence):
            raise TypeError('`tensors` should be a tuple[torch.Tensor] or '
                            'list[torch.Tensor] type')

        n_features = len(tensors)
        phys_dim = []
        bond_dim = []
        boundary = None

        for i, t in enumerate(tensors):
            if not isinstance(t, torch.Tensor):
                raise TypeError('`tensors` should be a tuple[torch.Tensor]'
                                ' or list[torch.Tensor] type')

            if i == 0:
                if t.ndim not in [1, 2, 3]:
                    raise ValueError(
                        'The first and last elements in `tensors` '
                        'should be both rank-2 or rank-3 tensors. If'
                        ' the first element is also the last one,'
                        ' it should be a rank-1 tensor')
                if t.ndim == 1:
                    boundary = 'obc'
                    phys_dim.append(t.shape[0])
                elif t.ndim == 2:
                    boundary = 'obc'
                    phys_dim.append(t.shape[0])
                    bond_dim.append(t.shape[1])
                else:
                    boundary = 'pbc'
                    phys_dim.append(t.shape[1])
                    bond_dim.append(t.shape[2])
            elif i == (n_features - 1):
                if t.ndim != tensors[0].ndim:
                    raise ValueError(
                        'The first and last elements in `tensors` '
                        'should have the same rank. Both should be '
                        'rank-2 or rank-3 tensors. If the first '
                        'element is also the last one, it should '
                        'be a rank-1 tensor')
                if t.ndim == 2:
                    phys_dim.append(t.shape[1])
                else:
                    if t.shape[-1] != tensors[0].shape[0]:
                        raise ValueError(
                            'If the first and last elements in `tensors`'
                            ' are rank-3 tensors, the first dimension '
                            'of the first element should coincide with'
                            ' the last dimension of the last element')
                    phys_dim.append(t.shape[1])
                    bond_dim.append(t.shape[2])
            else:
                if t.ndim != 3:
                    raise ValueError(
                        'The elements of `tensors` should be rank-3 '
                        'tensors, except the first and lest elements'
                        ' if boundary is "obc"')
                phys_dim.append(t.shape[1])
                bond_dim.append(t.shape[2])

        return n_features, phys_dim, bond_dim, boundary
    
    def _make_nodes(self, parameterized: bool = True) -> None:
        """Creates all the nodes of the MPS."""
        if self._leaf_nodes:
            raise ValueError('Cannot create MPS nodes if the MPS already has '
                             'nodes')
        
        aux_bond_dim = self._bond_dim
        
        if self._boundary == 'obc':
            if not aux_bond_dim:
                aux_bond_dim = [1]
                
            self._left_node = Node(shape=(aux_bond_dim[0],),
                                   axes_names=('right',),
                                   name='left_node',
                                   network=self)
            self._right_node = Node(shape=(aux_bond_dim[-1],),
                                    axes_names=('left',),
                                    name='right_node',
                                    network=self)
            
            aux_bond_dim = aux_bond_dim + [aux_bond_dim[-1]] + [aux_bond_dim[0]]
        
        node_cls = ParamNode if parameterized else Node

        for i in range(self._n_features):
            node = node_cls(shape=(aux_bond_dim[i - 1],
                                   self._phys_dim[i],
                                   aux_bond_dim[i]),
                            axes_names=('left', 'input', 'right'),
                            name=f'mats_env_node_({i})',
                            network=self)
            self._mats_env.append(node)

            if i != 0:
                self._mats_env[-2]['right'] ^ self._mats_env[-1]['left']

            if self._boundary == 'pbc':
                if i == 0:
                    periodic_edge = self._mats_env[-1]['left']
                if i == self._n_features - 1:
                    self._mats_env[-1]['right'] ^ periodic_edge
            else:
                if i == 0:
                    self._left_node['right'] ^ self._mats_env[-1]['left']
                if i == self._n_features - 1:
                    self._mats_env[-1]['right'] ^ self._right_node['left']
    
    def _make_canonical(self,
                        device: Optional[torch.device] = None,
                        dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
        """
        Creates random unitaries to initialize the MPS in canonical form with
        orthogonality center at the rightmost node. Unitaries in nodes are
        scaled so that the total norm squared of the initial MPS is the product
        of all the physical dimensions.
        """
        tensors = []
        for i, node in enumerate(self._mats_env):
            if self._boundary == 'obc':
                if i == 0:
                    node_shape = node.shape[1:]
                    aux_shape = node_shape
                    phys_dim = node_shape[0]
                elif i == (self._n_features - 1):
                    node_shape = node.shape[:2]
                    aux_shape = node_shape
                    phys_dim = node_shape[1]
                else:
                    node_shape = node.shape
                    aux_shape = (node.shape[:2].numel(), node.shape[2])
                    phys_dim = node_shape[1]
            else:
                node_shape = node.shape
                aux_shape = (node.shape[:2].numel(), node.shape[2])
                phys_dim = node_shape[1]
            size = max(aux_shape[0], aux_shape[1])
            
            tensor = random_unitary(size, device=device, dtype=dtype)
            tensor = tensor[:min(aux_shape[0], size), :min(aux_shape[1], size)]
            tensor = tensor.reshape(*node_shape)
            
            if i == (self._n_features - 1):
                if (self._boundary == 'obc') and (i == 0):
                    tensor = tensor[:, 0]
                tensor = tensor / tensor.norm()
            tensor = tensor * sqrt(phys_dim)
            
            tensors.append(tensor)
        return tensors
    
    def _make_unitaries(self,
                        device: Optional[torch.device] = None,
                        dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
        """
        Creates random unitaries to initialize the MPS nodes as stacks of
        unitaries.
        """
        tensors = []
        for i, node in enumerate(self._mats_env):
            units = []
            size = max(node.shape[0], node.shape[2])
            if self._boundary == 'obc':
                if i == 0:
                    size_1 = 1
                    size_2 = min(node.shape[2], size)
                elif i == (self._n_features - 1):
                    size_1 = min(node.shape[0], size)
                    size_2 = 1
                else:
                    size_1 = min(node.shape[0], size)
                    size_2 = min(node.shape[2], size)
            else:
                size_1 = min(node.shape[0], size)
                size_2 = min(node.shape[2], size)
            
            for _ in range(node.shape[1]):
                tensor = random_unitary(size, device=device, dtype=dtype)
                tensor = tensor[:size_1, :size_2]
                units.append(tensor)
            
            units = torch.stack(units, dim=1)
            
            if self._boundary == 'obc':
                if i == 0:
                    units = units.squeeze(0)
                elif i == (self._n_features - 1):
                    units = units.squeeze(-1)
            tensors.append(units)
        
        return tensors

    def initialize(self,
                   tensors: Optional[Sequence[torch.Tensor]] = None,
                   init_method: Optional[Text] = 'randn',
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None,
                   **kwargs: float) -> None:
        """
        Initializes all the nodes of the :class:`MPS`. It can be called when
        instantiating the model, or to override the existing nodes' tensors.
        
        There are different methods to initialize the nodes:
        
        * ``{"zeros", "ones", "copy", "rand", "randn"}``: Each node is
          initialized calling :meth:`~tensorkrowch.AbstractNode.set_tensor` with
          the given method, ``device``, ``dtype`` and ``kwargs``.
        
        * ``"randn_eye"``: Nodes are initialized as in this
          `paper <https://arxiv.org/abs/1605.03795>`_, adding identities at the
          top of random gaussian tensors. In this case, ``std`` should be
          specified with a low value, e.g., ``std = 1e-9``.
        
        * ``"unit"``: Nodes are initialized as stacks of random unitaries. This,
          combined (at least) with an embedding of the inputs as elements of
          the computational basis (:func:`~tensorkrowch.embeddings.discretize`
          combined with :func:`~tensorkrowch.embeddings.basis`)
        
        * ``"canonical"```: MPS is initialized in canonical form with a squared
          norm `close` to the product of all the physical dimensions (if bond
          dimensions are bigger than the powers of the physical dimensions,
          the norm could vary). Th orthogonality center is at the rightmost
          node.
        
        Parameters
        ----------
        tensors : list[torch.Tensor] or tuple[torch.Tensor], optional
            Sequence of tensors to set in each of the MPS nodes. If ``boundary``
            is ``"obc"``, all tensors should be rank-3, except the first and
            last ones, which can be rank-2, or rank-1 (if the first and last are
            the same). If ``boundary`` is ``"pbc"``, all tensors should be
            rank-3.
        init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensors if ``init_method`` is provided.
        dtype : torch.dtype, optional
            Dtype of the tensor if ``init_method`` is provided.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        """
        if init_method == 'unit':
            tensors = self._make_unitaries(device=device, dtype=dtype)
        elif init_method == 'canonical':
            tensors = self._make_canonical(device=device, dtype=dtype)

        if tensors is not None:
            if len(tensors) != self._n_features:
                raise ValueError(
                    '`tensors` should be a sequence of `n_features` elements')
            
            if self._boundary == 'obc':
                tensors = tensors[:]
                
                if device is None:
                    device = tensors[0].device
                if dtype is None:
                    dtype = tensors[0].dtype
                
                if len(tensors) == 1:
                    tensors[0] = tensors[0].reshape(1, -1, 1)
                else:
                    # Left node
                    aux_tensor = torch.zeros(*self._mats_env[0].shape,
                                             device=device,
                                             dtype=dtype)
                    aux_tensor[0] = tensors[0]
                    tensors[0] = aux_tensor
                    
                    # Right node
                    aux_tensor = torch.zeros(*self._mats_env[-1].shape,
                                             device=device,
                                             dtype=dtype)
                    aux_tensor[..., 0] = tensors[-1]
                    tensors[-1] = aux_tensor
                
            for tensor, node in zip(tensors, self._mats_env):
                node.tensor = tensor
                
        elif init_method is not None:
            add_eye = False
            if init_method == 'randn_eye':
                init_method = 'randn'
                add_eye = True
                
            for i, node in enumerate(self._mats_env):
                node.set_tensor(init_method=init_method,
                                device=device,
                                dtype=dtype,
                                **kwargs)
                if add_eye:
                    aux_tensor = node.tensor.detach()
                    aux_tensor[:, 0, :] += torch.eye(node.shape[0],
                                                     node.shape[2],
                                                     device=device,
                                                     dtype=dtype)
                    node.tensor = aux_tensor
                
                if self._boundary == 'obc':
                    aux_tensor = torch.zeros(*node.shape,
                                             device=device,
                                             dtype=dtype)
                    if i == 0:
                        # Left node
                        aux_tensor[0] = node.tensor[0]
                        node.tensor = aux_tensor
                    elif i == (self._n_features - 1):
                        # Right node
                        aux_tensor[..., 0] = node.tensor[..., 0]
                        node.tensor = aux_tensor
        
        if self._boundary == 'obc':
            self._left_node.set_tensor(init_method='copy',
                                       device=device,
                                       dtype=dtype)
            self._right_node.set_tensor(init_method='copy',
                                        device=device,
                                        dtype=dtype)

    def set_data_nodes(self) -> None:
        """
        Creates ``data`` nodes and connects each of them to the ``"input"``
        edge of each input node.
        """      
        input_edges = [node['input'] for node in self.in_env]
        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self._n_batches)
    
    def copy(self, share_tensors: bool = False) -> 'MPS':
        """
        Creates a copy of the :class:`MPS`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether tensors in the copied MPS should be
            set as the tensors in the current MPS (``True``), or cloned
            (``False``). In the former case, tensors in both MPS's will be
            the same, which might be useful if one needs more than one copy
            of an MPS, but wants to compute all the gradients with respect
            to the same, unique, tensors.

        Returns
        -------
        MPS
        """
        new_mps = MPS(n_features=self._n_features,
                      phys_dim=self._phys_dim,
                      bond_dim=self._bond_dim,
                      boundary=self._boundary,
                      tensors=None,
                      in_features=self._in_features,
                      out_features=self._out_features,
                      n_batches=self._n_batches,
                      init_method=None,
                      device=None,
                      dtype=None)
        new_mps.name = self.name + '_copy'
        
        for i in range(self._n_features):
            new_mps._mats_env[i] = new_mps._mats_env[i].parameterize(
                set_param=isinstance(self._mats_env[i], ParamNode))
        
        if share_tensors:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor
            if self._boundary == 'obc':
                new_mps._left_node.tensor = self._left_node.tensor
                new_mps._right_node.tensor = self.right_node.tensor
        else:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor.clone()
            if self._boundary == 'obc':
                new_mps._left_node.tensor = self._left_node.tensor.clone()
                new_mps.right_node.tensor = self.right_node.tensor.clone()
        return new_mps
    
    def parameterize(self,
                     set_param: bool = True,
                     override: bool = False) -> 'TensorNetwork':
        """
        Parameterizes all nodes of the MPS. If there are ``resultant`` nodes
        in the MPS, it will be first :meth:`~tensorkrowch.TensorNetwork.reset`.

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
            net = self.copy(share_tensors=False)
        
        for i in range(self._n_features):
            net._mats_env[i] = net._mats_env[i].parameterize(set_param=set_param)
            
        return net
    
    def update_bond_dim(self) -> None:
        """
        Updates the :attr:`bond_dim` attribute of the ``MPS``, in case it is
        outdated.
        
        If bond dimensions are changed, usually due to decompositions like
        :func:`~tensorkrowch.svd`, ``update_bond_dim`` should be
        called. This might modify some elements of the model, so it is
        recommended to do this before saving the ``state_dict`` of the model.
        Besides, if one wants to continue training, the ``parameters`` of the
        model that are passed to the optimizer should be updated also.
        Otherwise, the optimizer could be tracking outdated parameters that are
        not members of the model any more.
        """
        if self._boundary == 'obc':
            self._bond_dim = [node._shape[-1] for node in self._mats_env[:-1]]
            
            if self._bond_dim:
                left_size = self._bond_dim[0]
                if left_size != self._mats_env[0]._shape[0]:
                    self._mats_env[0]['left'].change_size(left_size)
                
                right_size = self._bond_dim[-1]
                if right_size != self._mats_env[-1]._shape[-1]:
                    self._mats_env[-1]['right'].change_size(right_size)
        else:
            self._bond_dim = [node._shape[-1] for node in self._mats_env]

    def _input_contraction(self,
                           nodes_env: List[AbstractNode],
                           input_nodes: List[AbstractNode],
                           inline_input: bool = False) -> Tuple[
                                                       Optional[List[Node]],
                                                       Optional[List[Node]]]:
        """Contracts input data nodes with MPS input nodes."""
        if inline_input:
            mats_result = [
                in_node @ node
                for node, in_node in zip(nodes_env, input_nodes)
                ]
            return mats_result

        else:
            if nodes_env:
                stack = op.stack(nodes_env)
                stack_data = op.stack(input_nodes)

                stack ^ stack_data

                result = stack_data @ stack
                mats_result = op.unbind(result)
                return mats_result
            else:
                return []

    @staticmethod
    def _inline_contraction(mats_env: List[AbstractNode],
                            renormalize: bool = False,
                            from_left: bool = True) -> Node:
        """Contracts sequence of MPS nodes (matrices) inline."""
        if from_left:
            result_node = mats_env[0]
            for node in mats_env[1:]:
                result_node @= node
                
                if renormalize:
                    right_axes = []
                    for ax_name in result_node.axes_names:
                        if 'right' in ax_name:
                            right_axes.append(ax_name)
                    if right_axes:
                        result_node = result_node.renormalize(axis=right_axes)
            
            return result_node
        
        else:
            result_node = mats_env[-1]
            for node in mats_env[-2::-1]:
                result_node = node @ result_node
                
                if renormalize:
                    left_axes = []
                    for ax_name in result_node.axes_names:
                        if 'left' in ax_name:
                            left_axes.append(ax_name)
                    if left_axes:
                        result_node = result_node.renormalize(axis=left_axes)
            
            return result_node

    def _contract_envs_inline(self,
                              mats_env: List[AbstractNode],
                              renormalize: bool = False) -> Node:
        """Contracts nodes environments inline."""
        from_left = True
        if self._boundary == 'obc':
            if mats_env[0].neighbours('left') is self._left_node:
                mats_env = [self._left_node] + mats_env
            if mats_env[-1].neighbours('right') is self._right_node:
                mats_env = mats_env + [self._right_node]
                from_left = False
        return self._inline_contraction(mats_env=mats_env,
                                        renormalize=renormalize,
                                        from_left=from_left)

    def _aux_pairwise(self,
                      mats_env: List[AbstractNode],
                      renormalize: bool = False) -> Tuple[List[Node],
    List[Node]]:
        """Contracts a sequence of MPS nodes (matrices) pairwise."""
        length = len(mats_env)
        aux_nodes = mats_env
        if length > 1:
            half_length = length // 2
            nice_length = 2 * half_length

            even_nodes = aux_nodes[0:nice_length:2]
            odd_nodes = aux_nodes[1:nice_length:2]
            leftover = aux_nodes[nice_length:]

            stack1 = op.stack(even_nodes)
            stack2 = op.stack(odd_nodes)

            stack1['right'] ^ stack2['left']

            aux_nodes = stack1 @ stack2
            
            if renormalize:
                axes = []
                for ax_name in aux_nodes.axes_names:
                    if ('left' in ax_name) or ('right' in ax_name):
                        axes.append(ax_name)
                if axes:
                    aux_nodes = aux_nodes.renormalize(axis=axes)
            
            aux_nodes = op.unbind(aux_nodes)

            return aux_nodes, leftover
        return mats_env, []

    def _pairwise_contraction(self,
                              mats_env: List[AbstractNode],
                              renormalize: bool = False) -> Node:
        """Contracts nodes environments pairwise."""
        length = len(mats_env)
        aux_nodes = mats_env
        if length > 1:
            leftovers = []
            while length > 1:
                aux1, aux2 = self._aux_pairwise(mats_env=aux_nodes,
                                                renormalize=renormalize)
                aux_nodes = aux1
                leftovers = aux2 + leftovers
                length = len(aux1)

            aux_nodes = aux_nodes + leftovers
            return self._pairwise_contraction(mats_env=aux_nodes,
                                              renormalize=renormalize)

        return self._contract_envs_inline(mats_env=aux_nodes,
                                          renormalize=renormalize)

    def _absorb_in_results_in_out_regions(self,
                                          in_results: List[Node]
                                          ) -> List[AbstractNode]:
        """Absorbs contracted input regions into the output regions."""
        nodes_out_env = []
        out_first = self.out_regions[0][0] == 0
        out_last = self.out_regions[-1][-1] == (self._n_features - 1)
        
        for i, region in enumerate(self.out_regions):
            aux_out_env = [self._mats_env[j] for j in region]
            
            if (i == 0) and out_first:
                if self._boundary == 'obc':
                    aux_out_env[0] = self._left_node @ aux_out_env[0]
            else:
                aux_out_env[0] = in_results[i - out_first] @ aux_out_env[0]
            nodes_out_env += aux_out_env

        if out_last:
            if self._boundary == 'obc':
                nodes_out_env[-1] = nodes_out_env[-1] @ self._right_node
        else:
            nodes_out_env[-1] = nodes_out_env[-1] @ in_results[-1]
        
        return nodes_out_env

    def _zipup_contraction(self,
                           nodes_envs: List[List[AbstractNode]],
                           renormalize: bool = False) -> Node:
        """Contracts two MPS or MPS-MPO-MPS via the zip-up method."""
        for i, node_tuple in enumerate(zip(*nodes_envs)):
            if i == 0:
                result_node = node_tuple[0]
                for node in node_tuple[1:]:
                    result_node @= node
            else:
                for node in node_tuple:
                    result_node @= node
                    
            if renormalize:
                right_axes = []
                for ax_name in result_node.axes_names:
                    if 'right' in ax_name:
                        right_axes.append(ax_name)
                if right_axes:
                    result_node = result_node.renormalize(axis=right_axes)
        
        return result_node

    def contract(self,
                 inline_input: bool = False,
                 inline_mats: bool = False,
                 renormalize: bool = False,
                 marginalize_output: bool = False,
                 embedding_matrices: Optional[
                                        Union[torch.Tensor,
                                              Sequence[torch.Tensor]]] = None,
                 mpo: Optional[MPO] = None
                 ) -> Node:
        """
        Contracts the whole MPS.
        
        If the MPS has input nodes, these are contracted against input ``data``
        nodes.
        
        If the MPS has output nodes, these can be left with their ``"input"``
        edges open, or can be marginalized, contracting the remaining output
        nodes with themselves, if the argument ``"marginalize_output"`` is set
        to ``True``.
        
        In the latter case, one can add additional nodes in between the MPS-MPS
        contraction:
        
        * ``embedding_matrices``: A list of matrices with appropiate physical
          dimensions can be passed, one for each output node. These matrices
          will connect the two ``"input"`` edges of the corresponding nodes.
        
        * ``mpo``: If an :class:`MPO` is passed, when calling
          ``mps(marginalize_output=True, mpo=mpo)``, this will perform the
          MPS-MPO-MPS contraction at the output nodes of the MPS. Therefore,
          the MPO should have as many nodes as output nodes are in the MPS.
          
          After contraction, the MPS will still be connected to the MPO nodes
          until these are manually disconnected.
          
          The provided MPO can also be already connected to the MPS before
          contraction. In this case, it is assumed that the output nodes of the
          MPS are connected to the ``"output"`` edges of the MPO nodes, and
          that the MPO nodes have been moved to the MPS, so that all nodes
          belong to the MPS network. In this case, each MPO node will connect
          the two ``"input"`` edges of the corresponding MPS nodes.
          
          If the MPO nodes are not trainable, they can be de-parameterized
          by doing ``mpo = mpo.parameterize(set_param=False, override=True)``.
          This should be done before the contraction, or before connecting
          the MPO nodes to the MPS, since the de-parameterized nodes are not
          the same nodes as the original ``ParamNodes`` of the MPO.
        
        When ``marginalize_output = True``, the contracted input nodes are
        duplicated using different batch dimensions. That is, if the MPS
        is contracted with input data with ``batch_size = 100``, and some
        other (output) nodes are marginalized, the result will be a tensor
        with shape ``(100, 100)`` rather than just ``(100,)``.
        
        Parameters
        ----------
        inline_input : bool
            Boolean indicating whether input ``data`` nodes should be contracted
            with the ``MPS`` input nodes inline (one contraction at a time) or
            in a single stacked contraction.
        inline_mats : bool
            Boolean indicating whether the sequence of matrices (resultant
            after contracting the input ``data`` nodes) should be contracted
            inline or as a sequence of pairwise stacked contrations.
        renormalize : bool
            Indicates whether nodes should be renormalized after contraction.
            If not, it may happen that the norm explodes or vanishes, as it
            is being accumulated from all nodes. Renormalization aims to avoid
            this undesired behavior by extracting the norm of each node on a
            logarithmic scale. The renormalization only occurs when multiplying
            sequences of matrices, once the `input` contractions have been
            already performed, including contracting against embedding matrices
            or MPOs when ``marginalize_output = True``.
        marginalize_output : bool
            Boolean indicating whether output nodes should be marginalized. If
            ``True``, after contracting all the input nodes with their
            neighbouring data nodes, this resultant network is contracted with
            itself connecting output nodes to itselves at ``"input"`` edges. If
            ``False``, output nodes are left with their ``"input"`` edges
            disconnected.
        embedding_matrices : torch.Tensor, list[torch.Tensor] or tuple[torch.Tensor], optional
            If ``marginalize_output = True``, a matrix can be introduced
            between each output node and its copy, connecting the ``"input"``
            edges. This can be useful when data vectors are not represented
            as qubits in the computational basis, but are transformed via
            some :ref:`Embeddings` function.
        mpo : MPO, optional
            MPO that is to be contracted with the MPS at the output nodes, if
            ``marginalize_output = True``. In this case, the ``"output"`` edges
            of the MPO nodes will be connected to the ``"input"`` edges of the
            MPS output nodes. If there are no input nodes, the MPS-MPO-MPS
            is performed by calling ``mps(marginalize_output=True, mpo=mpo)``,
            without passing extra data tensors.

        Returns
        -------
        Node
        """
        if embedding_matrices is not None:
            if isinstance(embedding_matrices, Sequence):
                if len(embedding_matrices) != len(self._out_features):
                    raise ValueError(
                        '`embedding_matrices` should have the same amount of '
                        'elements as output nodes are in the MPS')
            else:
                embedding_matrices = [embedding_matrices] * len(self._out_features)
                
            for i, (mat, node) in enumerate(zip(embedding_matrices,
                                                self.out_env)):
                if not isinstance(mat, torch.Tensor):
                    raise TypeError(
                        '`embedding_matrices` should be torch.Tensor type')
                if mat.ndim != 2:
                    raise ValueError(
                        '`embedding_matrices should ne rank-2 tensors')
                if mat.shape[0] != mat.shape[1]:
                    raise ValueError(
                        '`embedding_matrices` should have equal dimensions')
                if node['input'].size() != mat.shape[0]:
                    raise ValueError(
                        '`embedding_matrices` dimensions should be equal '
                        'to the input dimensions of the corresponding MPS '
                        'output nodes')
        elif mpo is not None:
            if not isinstance(mpo, MPO):
                raise TypeError('`mpo` should be MPO type')
            if mpo._n_features != len(self._out_features):
                raise ValueError(
                    '`mpo` should have as many features as output nodes are '
                    'in the MPS')
            
        in_regions = self.in_regions
        out_regions = self.out_regions
        
        mats_in_env = self._input_contraction(
            nodes_env=self.in_env,
            input_nodes=[node.neighbours('input') for node in self.in_env],
            inline_input=inline_input)
        
        in_results = []
        for region in in_regions:
            if inline_mats:
                result = self._contract_envs_inline(
                    mats_env=mats_in_env[:len(region)],
                    renormalize=renormalize)
            else:
                result = self._pairwise_contraction(
                    mats_env=mats_in_env[:len(region)],
                    renormalize=renormalize)
            
            mats_in_env = mats_in_env[len(region):]
            in_results.append(result)
        
        if not out_regions:
            # If there is only input region, in_results has only 1 node
            result = in_results[0]
        
        else:
            nodes_out_env = self._absorb_in_results_in_out_regions(in_results)
            
            if not marginalize_output:
                # Contract all output nodes sequentially
                result = self._inline_contraction(mats_env=nodes_out_env,
                                                  renormalize=renormalize)
            
            else:
                # Copy output nodes sharing tensors
                copied_nodes = []
                for node in nodes_out_env:
                    copied_node = node.__class__(shape=node._shape,
                                                 axes_names=node.axes_names,
                                                 name='virtual_result_copy',
                                                 network=self,
                                                 virtual=True)
                    copied_node.set_tensor_from(node)
                    copied_nodes.append(copied_node)
                    
                    # Change batch names so that they not coincide with
                    # original batches, which gives duplicate output batches
                    for ax in copied_node.axes:
                        if ax._batch:
                            ax.name = ax.name + '_copy'
                
                # Connect copied nodes with neighbours
                for i in range(len(copied_nodes)):
                    if (i == 0) and (self._boundary == 'pbc'):
                        if nodes_out_env[i - 1].is_connected_to(nodes_out_env[i]):
                            copied_nodes[i - 1]['right'] ^ copied_nodes[i]['left']
                    elif i > 0:
                        copied_nodes[i - 1]['right'] ^ copied_nodes[i]['left']
                
                nodes_envs = [nodes_out_env]
                
                # Contract with embedding matrices
                if embedding_matrices is not None:
                    mats_nodes = []
                    for i, node in enumerate(nodes_out_env):
                        # Reattach input edges
                        node.reattach_edges(axes=['input'])
                        
                        # Create matrices
                        mat_node = Node(tensor=embedding_matrices[i],
                                        axes_names=('input', 'output'),
                                        name='virtual_result_mat',
                                        network=self,
                                        virtual=True)
                        
                        # Connect matrices to output nodes
                        mat_node['output'] ^ node['input']
                        mats_nodes.append(mat_node)
                    
                    # Connect matrices to copies
                    for mat_node, copied_node in zip(mats_nodes, copied_nodes):
                        copied_node['input'] ^ mat_node['input']
                    
                    nodes_envs.append(mats_nodes)
                
                # Contract with mpo
                elif mpo is not None:
                    # Move all the connected component to the MPS network
                    mpo._mats_env[0].move_to_network(self)
                    
                    # Move uniform memory
                    if isinstance(mpo, UMPO):
                        mpo.uniform_memory.move_to_network(self)
                        for node in mpo._mats_env:
                            node.set_tensor_from(mpo.uniform_memory)
                    
                    # Connect MPO to MPS
                    for mps_node, mpo_node in zip(nodes_out_env, mpo._mats_env):
                        # Reattach input edges
                        mps_node.reattach_edges(axes=['input'])
                        mpo_node['output'] ^ mps_node['input']
                    
                    # Connect MPO to copies
                    for copied_node, mpo_node in zip(copied_nodes, mpo._mats_env):
                        copied_node['input'] ^ mpo_node['input']
                    
                    mpo_nodes = mpo._mats_env[:]
                    
                    # Contract MPO left and right nodes
                    if mpo._boundary == 'obc':
                        mpo_nodes[0] = mpo._left_node @ mpo_nodes[0]
                        mpo_nodes[-1] = mpo_nodes[-1] @ mpo._right_node
                    
                    nodes_envs.append(mpo_nodes)
                
                else:
                    # Reattach input edges of resultant output nodes and connect
                    # with copied nodes
                    for node, copied_node in zip(nodes_out_env, copied_nodes):
                        # Reattach input edges
                        node.reattach_edges(axes=['input'])
                        
                        # Connect copies directly to output nodes
                        copied_node['input'] ^ node['input']
                
                # If MPS nodes are complex, copied nodes are their conjugates
                is_complex = copied_nodes[0].is_complex()
                if is_complex:
                    for i, node in enumerate(copied_nodes):
                        copied_nodes[i] = node.conj()
                
                nodes_envs.append(copied_nodes)
                
                # Contract nodes (MPS-MPS, MPS-mats-MPS, or MPS-MPO-MPS) via zip-up
                result = self._zipup_contraction(nodes_envs=nodes_envs,
                                                 renormalize=renormalize)
            
        # Contract periodic edge
        if result.is_connected_to(result):
            result @= result
        
        # Put batch edges in first positions
        batch_edges = []
        other_edges = []
        for i, edge in enumerate(result.edges):
            if edge.is_batch():
                batch_edges.append(i)
            else:
                other_edges.append(i)
        
        all_edges = batch_edges + other_edges
        if all_edges != list(range(len(all_edges))):
            result = result.permute(tuple(all_edges))
        
        return result
    
    def norm(self,
             log_scale: bool = False) -> torch.Tensor:
        """
        Computes the norm of the MPS.
        
        This method internally removes all data nodes in the MPS, if any, and
        contracts the nodes with themselves. Therefore, this may alter the
        usual behaviour of :meth:`contract` if the MPS is not
        :meth:`~tensorkrowch.TensorNetwork.reset` afterwards. Also, if the MPS
        was contracted before with other arguments, it should be ``reset``
        before calling ``norm`` to avoid undesired behaviour.
        
        Since the norm is computed by contracting the MPS, it means one can
        take gradients of it with respect to the MPS tensors, if it is needed.
        
        Parameters
        ----------
        log_scale : bool
            Boolean indicating whether the resulting norm should be given in
            logarithmoc scale. Useful for cases where the norm explodes or
            vanishes.
        """
        if self._data_nodes:
            self.unset_data_nodes()
        
        # All nodes belong to the output region
        all_nodes = self.mats_env[:]
        
        if self._boundary == 'obc':
            all_nodes[0] = self._left_node @ all_nodes[0]
            all_nodes[-1] = all_nodes[-1] @ self._right_node
        
        # Check if nodes are already connected to copied nodes
        create_copies = []
        for node in all_nodes:
            neighbour = node.neighbours('input')
            if neighbour is None:
                create_copies.append(True)
            else:
                if 'virtual_result_copy' not in neighbour.name:
                    raise ValueError(
                        f'Node {node} is already connected to another node '
                        'at axis "input". Disconnect the node or reset the '
                        'network before calling `norm`')
                else:
                    create_copies.append(False)
        
        if any(create_copies) and not all(create_copies):
            raise ValueError(
                'There are some nodes connected and some disconnected at axis '
                '"input". Disconnect all of them before calling `norm`')
        
        create_copies = any(create_copies)
        
        # Copy output nodes sharing tensors
        if create_copies:
            copied_nodes = []
            for node in all_nodes:
                copied_node = node.__class__(shape=node._shape,
                                             axes_names=node.axes_names,
                                             name='virtual_result_copy',
                                             network=self,
                                             virtual=True)
                copied_node.set_tensor_from(node)
                copied_nodes.append(copied_node)
                
                # Change batch names so that they not coincide with
                # original batches, which gives duplicate output batches
                for ax in copied_node.axes:
                    if ax._batch:
                        ax.name = ax.name + '_copy'
            
            # Connect copied nodes with neighbours
            for i in range(len(copied_nodes)):
                if (i == 0) and (self._boundary == 'pbc'):
                    if all_nodes[i - 1].is_connected_to(all_nodes[i]):
                        copied_nodes[i - 1]['right'] ^ copied_nodes[i]['left']
                elif i > 0:
                    copied_nodes[i - 1]['right'] ^ copied_nodes[i]['left']
            
            # Reattach input edges of resultant output nodes and connect
            # with copied nodes
            for node, copied_node in zip(all_nodes, copied_nodes):
                # Reattach input edges
                node.reattach_edges(axes=['input'])
                
                # Connect copies directly to output nodes
                copied_node['input'] ^ node['input']
        else:
            copied_nodes = []
            for node in all_nodes:
                copied_nodes.append(node.neighbours('input'))
        
        # If MPS nodes are complex, copied nodes are their conjugates
        is_complex = copied_nodes[0].is_complex()
        if is_complex:
            for i, node in enumerate(copied_nodes):
                copied_nodes[i] = node.conj()
        
        # Contract nodes with copies via zip-up
        log_norm = 0
        nodes_envs = [all_nodes, copied_nodes]
        for i, (node, copied_node) in enumerate(zip(*nodes_envs)):
            if i == 0:
                result_node = node @ copied_node
            else:
                result_node @= node
                result_node @= copied_node
            
            if log_scale:
                log_norm += result_node.norm().log()
                result_node = result_node.renormalize()
        
        # Contract periodic edge
        if result_node.is_connected_to(result_node):
            result_node @= result_node
            
            if log_scale:
                log_norm += result_node.norm().log()
                result_node = result_node.renormalize()
        
        if log_scale:
            return log_norm / 2
        
        result = result_node.tensor.sqrt()
        
        if is_complex:
            result = result.abs()  # result is already real
        
        return result

    def reduced_density(self,
                        trace_sites: Sequence[int] = [],
                        renormalize: bool = True) -> torch.Tensor:
        r"""
        Returns de partial density matrix, tracing out the sites specified
        by ``trace_sites``: :math:`\rho_A`.
        
        This method internally sets ``out_features = trace_sites``, and calls
        the :meth:`~tensorkrowch.TensorNetwork.forward` method with
        ``marginalize_output = True``. Therefore, it may alter the behaviour
        of the MPS if it is not :meth:`~tensorkrowch.TensorNetwork.reset`
        afterwards. Also, if the MPS was contracted before with other arguments,
        it should be ``reset`` before calling ``reduced_density`` to avoid
        undesired behaviour.
        
        Since the density matrix is computed by contracting the MPS, it means
        one can take gradients of it with respect to the MPS tensors, if it
        is needed.
        
        This method may also alter the attribute :attr:`n_batches` of the
        :class:`MPS`.
        
        Parameters
        ----------
        trace_sites : list[int] or tuple[int]
            Sequence of nodes' indices in the MPS. These indices specify the
            nodes that should be traced to compute the density matrix. If
            it is empty ``[]``, the total density matrix will be returned,
            though this may be costly if :attr:`n_features` is big.
        renormalize : bool
            Indicates whether nodes should be renormalized after contraction.
            If not, it may happen that the norm explodes or vanishes, as it
            is being accumulated from all nodes. Renormalization aims to avoid
            this undesired behavior by extracting the norm of each node on a
            logarithmic scale. The renormalization only occurs when multiplying
            sequences of matrices, once the `input` contractions have been
            already performed.
        
        Examples
        --------
        >>> mps = tk.models.MPS(n_features=4,
        ...                     phys_dim=[2, 3, 4, 5],
        ...                     bond_dim=5)
        >>> density = mps.reduced_density(trace_sites=[0, 2])
        >>> density.shape
        torch.Size([3, 5, 3, 5])
        """
        if not isinstance(trace_sites, Sequence):
            raise TypeError(
                '`trace_sites` should be list[int] or tuple[int] type')
            
        for site in trace_sites:
            if not isinstance(site, int):
                raise TypeError(
                    'elements of `trace_sites` should be int type')
            if (site < 0) or (site >= self._n_features):
                raise ValueError(
                    'Elements of `trace_sites` should be between 0 and '
                    '(`n_features` - 1)')
        
        if set(trace_sites) != set(self.out_features):
            if self._data_nodes:
                self.unset_data_nodes()
            self.out_features = trace_sites
        
        # Create dataset with all possible combinations for the input nodes
        # so that they are kept sort of "open"
        dims = torch.tensor([self._phys_dim[i] for i in self._in_features])
        
        data = []
        for i in range(len(self._in_features)):
            aux = torch.arange(dims[i]).view(-1, 1)
            aux = aux.repeat(1, dims[(i + 1):].prod()).flatten().view(-1, 1)
            aux = aux.repeat(dims[:i].prod(), 1)
            
            data.append(aux.reshape(*dims, 1))
        
        n_dims = len(set(dims))
        if n_dims >= 1:
            if n_dims == 1:
                data = torch.cat(data, dim=-1)
                data = basis(data, dim=dims[0])\
                    .to(self.in_env[0].dtype)\
                    .to(self.in_env[0].device)
            elif n_dims > 1:
                data = [
                    basis(dat, dim=dim).squeeze(-2)\
                        .to(self.in_env[0].dtype)\
                        .to(self.in_env[0].device)
                    for dat, dim in zip(data, dims)
                    ]
            
            self.n_batches = len(dims)
            result = self.forward(data,
                                  renormalize=renormalize,
                                  marginalize_output=True)
            
        else:
            result = self.forward(renormalize=renormalize,
                                  marginalize_output=True)
        
        if self._n_features == 1:
            size = result.shape[0]
            result = result.outer(result).view(size, size)
        
        return result
    
    @torch.no_grad()
    def entropy(self,
                middle_site: int,
                renormalize: bool = False) -> Union[float, Tuple[float]]:
        r"""
        Computes the von Neumann entropy of the reduced density matrix
        :math:`\rho_A` (entanglement entropy) between subsystems :math:`A` and
        :math:`B`, where :math:`A` goes from site 0 to ``middle_site``, and
        :math:`B` goes from ``middle_site + 1`` to ``n_features - 1``.
        
        To compute the entanglement entropy, the MPS is put into canonical form
        with orthogonality center at ``middle_site``. Bond dimensions are not
        changed if possible. Only when the bond dimension is bigger than the
        physical dimension multiplied by the other bond dimension of the node,
        it will be cropped to that size.
        
        If the MPS is not normalized, it may happen that the computation of the
        entanglement entropy fails due to errors in the Singular Value
        Decompositions. To avoid this, it is recommended to set
        ``renormalize = True``. In this case, the norm of each node after the
        SVD is extracted in logarithmic form, and accumulated. As a result,
        the function will return the tuple ``(entropy, log_norm)``, which is a
        scaled entanglement entropy. This is, indeed, the entanglement entropy
        of a distribution, since the schmidt values are normalized to sum up
        to 1.
        
        The actual entanglement entropy, without rescaling, could be obtained as:
        
        .. math::
        
            \exp(\texttt{log_norm})^2 \cdot S(\rho_A) - 
            \exp(\texttt{log_norm})^2 \cdot 2 \cdot \texttt{log_norm}
        
        This method internally calls :meth:`~tensorkrowch.TensorNetwork.reset`,
        as :meth:`canonicalize` may change the form of the tensors.
        
        Parameters
        ----------
        middle_site : int
            Position that separates regios :math:`A` and :math:`B`. It should
            be between 0 and ``n_features - 2``.
        renormalize : bool
            Indicates whether nodes should be renormalized after SVD/QR
            decompositions. If not, it may happen that the norm explodes as it
            is being accumulated from all nodes. Renormalization aims to avoid
            this undesired behavior by extracting the norm of each node on a
            logarithmic scale after SVD/QR decompositions are computed. Finally,
            the normalization factor is evenly distributed among all nodes of
            the MPS.
        
        Returns
        -------
        float or tuple[float, float]
        """
        self.reset()

        prev_auto_stack = self._auto_stack
        self.auto_stack = False
        
        if (middle_site < 0) or (middle_site > (self._n_features - 2)):
            raise ValueError(
                '`middle_site` should be between 0 and `n_features` - 2')
        
        log_norm = 0
        
        nodes = self._mats_env[:]
        if self._boundary == 'obc':
            nodes[0].tensor[1:] = torch.zeros_like(
                nodes[0].tensor[1:])
            nodes[-1].tensor[..., 1:] = torch.zeros_like(
                nodes[-1].tensor[..., 1:])
        
        # Keep track of which nodes are parameterized
        set_params = [isinstance(node, ParamNode) for node in nodes]
        
        for i in range(middle_site):
            result1, result2 = nodes[i]['right'].svd_(
                side='right',
                rank=nodes[i]['right'].size())
            
            if renormalize:
                aux_norm = result2.norm()
                if not aux_norm.isinf() and (aux_norm > 0):
                    result2.tensor = result2.tensor / aux_norm
                    log_norm += aux_norm.log()
            
            nodes[i] = result1.parameterize(set_param=set_params[i])
            nodes[i + 1] = result2

        for i in range(len(nodes) - 1, middle_site, -1):
            result1, result2 = nodes[i]['left'].svd_(
                side='left',
                rank=nodes[i]['left'].size())
            
            if renormalize:
                aux_norm = result1.norm()
                if not aux_norm.isinf() and (aux_norm > 0):
                    result1.tensor = result1.tensor / aux_norm
                    log_norm += aux_norm.log()

            nodes[i] = result2.parameterize(set_param=set_params[i])
            nodes[i - 1] = result1
        
        nodes[middle_site] = nodes[middle_site].parameterize(
            set_param=set_params[middle_site])
        
        # Compute entanglement entropy
        middle_tensor = nodes[middle_site].tensor.clone()
        _, s, _ = torch.linalg.svd(
            middle_tensor.reshape(middle_tensor.shape[:-1].numel(), # left x input
                                  middle_tensor.shape[-1]),         # right
            full_matrices=False)
        
        s /= s.norm()
        s2 = s[s > 0].pow(2)
        entropy = -(s2 * s2.log()).sum()
        
        # Rescale
        if renormalize and (log_norm != 0):
            rescale = (log_norm / len(nodes)).exp()
            for node in nodes:
                node.tensor = node.tensor * rescale
        
        # Update variables
        self._mats_env = nodes
        self.update_bond_dim()

        self.auto_stack = prev_auto_stack
        
        if renormalize:
            return entropy, log_norm
        else:
            return entropy
    
    @torch.no_grad()
    def condition(self,
                  data: Union[torch.Tensor, Sequence[torch.Tensor]]) -> 'MPS':
        """
        Conditions output nodes on embedded data and returns a new MPS.

        Each conditioned node is contracted with its data vector. Consecutive
        conditioned nodes are then absorbed into the closest input node.
        
        If there are ``resultant`` nodes in the MPS, it will be first
        :meth:`~tensorkrowch.TensorNetwork.reset`.

        Embedded data can be passed as a single tensor with one of the
        following layouts:

        * ``(phys_dim,)`` when there is only one output node.
        * ``(len(out_features), phys_dim)`` with no batch dimension.
        * ``(n_features, phys_dim)`` with no batch dimension. Only the entries
          corresponding to ``out_features`` are used.
        * Either of the two previous layouts with an initial batch dimension
          of size 1.

        It can also be passed as a list or tuple containing either
        ``len(out_features)`` or ``n_features`` tensors. Each tensor should
        have shape ``(phys_dim,)`` or ``(1, phys_dim)``. In the latter case,
        the initial dimension is a batch dimension of size 1. For a sequence
        with ``n_features`` elements, only those corresponding to
        ``out_features`` are used.

        Thus, all accepted layouts represent a single configuration; batches
        with more than one element are not accepted.

        Parameters
        ----------
        data : torch.Tensor, list[torch.Tensor] or tuple[torch.Tensor]
            Embedded data used to condition the output nodes, with one of the
            layouts described above.

        Returns
        -------
        MPS

        Examples
        --------
        >>> mps = tk.models.MPSLayer(n_features=4,
        ...                          in_dim=2,
        ...                          out_dim=3,
        ...                          bond_dim=5)
        >>> label = torch.tensor(1)
        >>> embedded_label = tk.embeddings.basis(label, dim=3).float()
        >>> embedded_label.shape
        torch.Size([3])
        >>> cond_mps = mps.condition(embedded_label)
        >>> cond_mps.n_features
        3

        >>> mps = tk.models.MPS(n_features=4, phys_dim=2, bond_dim=5,
        ...                     out_features=[1, 3])
        >>> all_data = tk.embeddings.basis(
        ...     torch.tensor([[0, 1, 0, 1]]), dim=2).float()
        >>> cond_mps = mps.condition(all_data)
        >>> samples = cond_mps.sample(n_samples=10)
        """
        
        if self._resultant_nodes:
            warnings.warn(
                'Resultant nodes will be removed before conditioning the TN')
            self.reset()
        if self._data_nodes:
            self.unset_data_nodes()

        if not self._out_features:
            raise ValueError('Cannot condition an MPS with no output nodes')
        if not self._in_features:
            raise ValueError('Conditioning all nodes would not return an MPS')

        # Remove the optional batch dimension and select output data.
        if isinstance(data, torch.Tensor):
            if data.ndim == 3:
                if data.shape[0] != 1:
                    raise ValueError('`data` should have batch size 1')
                data = data.squeeze(0)
            elif data.ndim == 1:
                data = data.unsqueeze(0)
            elif data.ndim != 2:
                raise ValueError(
                    '`data` should be provided without batch dimension or '
                    'with batch size 1')

            if data.shape[-2] == self._n_features:
                data = data[self._out_features]
        else:
            if len(data) == self._n_features:
                data = [data[site] for site in self._out_features]

            data = list(data)
            for i, tensor in enumerate(data):
                if tensor.ndim == 2:
                    if tensor.shape[0] != 1:
                        raise ValueError('`data` should have batch size 1')
                    data[i] = tensor.squeeze(0)
                elif tensor.ndim != 1:
                    raise ValueError(
                        '`data` should be provided without batch dimension or '
                        'with batch size 1')

        # Temporarily turn conditioned output nodes into inputs with data.
        self.in_features = self.out_features
        super().set_data_nodes(input_edges=[node['input'] for node in self.in_env],
                               num_batch_edges=0)
        self.add_data(data)

        mats_in_env = self._input_contraction(
            nodes_env=self.in_env,
            input_nodes=[node.neighbours('input') for node in self.in_env],
            inline_input=True)
        
        in_results = []
        for region in self.in_regions:
            result = self._contract_envs_inline(mats_env=mats_in_env[:len(region)])
            mats_in_env = mats_in_env[len(region):]
            in_results.append(result)

        nodes_out_env = self._absorb_in_results_in_out_regions(in_results)
        
        cond_mps = MPS(tensors=[node.tensor for node in nodes_out_env])
        
        self.reset()
        self.unset_data_nodes()
        self.in_features = self.out_features
        
        return cond_mps
    
    ############################
    # SAMPLE: work in progress #
    ############################
    def _copy_sample_mps(self) -> Tuple[List[AbstractNode],
                                        Optional[AbstractNode],
                                        Optional[AbstractNode]]:
        """Moves a tensor-sharing virtual MPS copy to the current network."""
        copied_mps = self.copy(share_tensors=True)
        copied_nodes = copied_mps.mats_env[:]
        copied_boundaries = []
        if self._boundary == 'obc':
            copied_boundaries = [copied_mps.left_node,
                                 copied_mps.right_node]

        # Mark the copied MPS as temporary before moving the whole component.
        for node in copied_nodes + copied_boundaries:
            node.name = 'virtual_result_sample_copy'
            node.change_type(virtual=True)
        copied_nodes[0].move_to_network(self)

        if self._boundary == 'obc':
            copied_left = copied_boundaries[0]
            copied_right = copied_boundaries[1]
        else:
            copied_left = None
            copied_right = None

        return copied_nodes, copied_left, copied_right

    @staticmethod
    def _contract_sample_site(node: AbstractNode,
                              copied_node: AbstractNode,
                              env: Optional[AbstractNode],
                              matrix_node: Optional[AbstractNode] = None,
                              data_nodes: Optional[Tuple[AbstractNode,
                                                         AbstractNode]] = None,
                              from_left: bool = False,
                              renormalize: bool = False) -> Node:
        """Absorbs one double-layer MPS site into a sweep environment."""
        if data_nodes is not None:
            data_node, copied_data_node = data_nodes
            if env is None:
                pair_node = Node(
                    tensor=torch.ones(1,
                                      device=data_node.device,
                                      dtype=data_node.dtype),
                    axes_names=('pair',),
                    name='virtual_result_sample_pair',
                    network=data_node.network,
                    virtual=True)
                copied_pair_node = Node(
                    tensor=torch.ones(1,
                                      device=copied_data_node.device,
                                      dtype=copied_data_node.dtype),
                    axes_names=('pair',),
                    name='virtual_result_sample_pair_copy',
                    network=data_node.network,
                    virtual=True)
                pair_node['pair'] ^ copied_pair_node['pair']
                data_node = data_node % pair_node
                copied_data_node = copied_data_node % copied_pair_node

            node = data_node @ node
            copied_node = copied_data_node @ copied_node

        if env is None:
            if matrix_node is not None:
                result = (node @ matrix_node) @ copied_node
            elif node.is_connected_to(copied_node):
                result = node @ copied_node
            else:
                raise ValueError(
                    'The two sample layers should be connected before '
                    'contracting a site')
        elif from_left:
            result = env @ node
            if matrix_node is not None:
                result = result @ matrix_node
            result = result @ copied_node
        else:
            result = node @ env
            if matrix_node is not None:
                result = result @ matrix_node
            result = result @ copied_node

        if renormalize:
            axes = [axis.name for axis in result.axes
                    if not axis.is_batch()]
            if axes:
                result = result.renormalize(axis=axes)

        return result

    @staticmethod
    def _trapezoidal_weights(domain: torch.Tensor) -> torch.Tensor:
        """Returns the point weights induced by the trapezoidal rule."""
        if (domain.ndim != 1) or (domain.numel() <= 1) or \
                (not torch.is_floating_point(domain)):
            return torch.ones(domain.shape[0],
                              device=domain.device,
                              dtype=torch.get_default_dtype())

        grid_basis = torch.eye(domain.numel(),
                               device=domain.device,
                               dtype=domain.dtype)
        return torch.trapezoid(grid_basis, x=domain, dim=0)

    @staticmethod
    def _probabilities_from_amplitudes(probs: torch.Tensor) -> torch.Tensor:
        """Cleans and row-normalizes Born probabilities."""
        if torch.is_complex(probs):
            scale = probs.abs().max().detach()
            imag = probs.imag.abs().max().detach()
            if scale > 0:
                if imag > (1e-5 * scale):
                    warnings.warn(
                        'Probabilities have a non-negligible imaginary part. '
                        'Only their real part will be used')
            probs = probs.real

        if not torch.is_floating_point(probs):
            probs = probs.float()

        if not torch.isfinite(probs).all():
            raise ValueError('Probabilities contain non-finite values')

        finfo = torch.finfo(probs.dtype)
        scale = probs.abs().max().detach().clamp_min(1)
        tol = 1000 * finfo.eps * scale
        if (probs < -tol).any():
            raise ValueError(
                'Some probabilities are negative beyond numerical tolerance')

        probs = probs.clamp_min(0)
        norm = probs.sum(dim=1, keepdim=True)
        if (norm <= 0).any() or (not torch.isfinite(norm).all()):
            raise ValueError(
                'Cannot sample from a zero or non-finite probability row')

        return probs / norm

    def _sample_apply_embedding(self,
                                domain: torch.Tensor,
                                embedding: Optional[Callable],
                                phys_dim: int,
                                reference: torch.Tensor) -> torch.Tensor:
        """Embeds one sampling domain."""
        if embedding is None:
            if torch.is_floating_point(domain):
                rounded = domain.round()
                if not torch.equal(domain, rounded):
                    raise ValueError(
                        'Default basis embedding requires integer domain values')
                domain = rounded.long()
            embedded = basis(domain.long(), dim=phys_dim).to(
                dtype=reference.dtype)
        else:
            embedded = embedding(domain)

        if not isinstance(embedded, torch.Tensor):
            raise TypeError('`embedding` should return torch.Tensor')

        if embedded.ndim >= 3 and embedded.shape[-2] == 1:
            embedded = embedded.squeeze(-2)

        if (embedded.ndim != 2) or (embedded.shape[-1] != phys_dim):
            raise ValueError(
                'Embedded domains should have shape (domain_size, phys_dim)')

        if not torch.is_floating_point(embedded) and \
                not torch.is_complex(embedded):
            embedded = embedded.to(dtype=reference.dtype)

        return embedded

    def _prepare_sample_domains(self,
                                domain,
                                embedding,
                                embedding_matrices,
                                build_matrices: bool
                                ) -> Tuple[List[torch.Tensor],
                                           List[torch.Tensor],
                                           List[torch.Tensor],
                                           List[torch.Tensor]]:
        """Prepares domains, embedded domains and physical metrics."""
        n_inputs = len(self._in_features)

        # Normalize site-wise arguments in input-feature order.
        if domain is None:
            first_site = self._in_features[0]
            same_default_domain = all(
                (self._phys_dim[site] == self._phys_dim[first_site]) and
                (self._mats_env[site].device ==
                 self._mats_env[first_site].device)
                for site in self._in_features)
            if same_default_domain:
                shared_domain = torch.arange(
                    self._phys_dim[first_site],
                    device=self._mats_env[first_site].device)
                domains = [shared_domain] * n_inputs
            else:
                domains = [None] * n_inputs
        elif isinstance(domain, torch.Tensor):
            domains = [domain] * n_inputs
        elif isinstance(domain, Sequence):
            if len(domain) == self._n_features:
                domains = [domain[site] for site in self._in_features]
            elif len(domain) == n_inputs:
                domains = list(domain)
            else:
                raise ValueError(
                    '`domain` should have either `n_features` elements or as '
                    'many elements as input nodes are sampled')
        else:
            raise TypeError('`domain` should be torch.Tensor type')

        if embedding is None or callable(embedding):
            embeddings_arg = [embedding] * n_inputs
        elif isinstance(embedding, Sequence):
            if len(embedding) == self._n_features:
                embeddings_arg = [embedding[site]
                                  for site in self._in_features]
            elif len(embedding) == n_inputs:
                embeddings_arg = list(embedding)
            else:
                raise ValueError(
                    '`embedding` should have either `n_features` elements or '
                    'as many elements as input nodes are sampled')
        else:
            raise TypeError('`embedding` should be callable type')

        if embedding_matrices is None or \
                isinstance(embedding_matrices, torch.Tensor):
            matrices_arg = [embedding_matrices] * n_inputs
        elif isinstance(embedding_matrices, Sequence):
            if len(embedding_matrices) == self._n_features:
                matrices_arg = [embedding_matrices[site]
                                for site in self._in_features]
            elif len(embedding_matrices) == n_inputs:
                matrices_arg = list(embedding_matrices)
            else:
                raise ValueError(
                    '`embedding_matrices` should have either `n_features` '
                    'elements or as many elements as input nodes are sampled')
        else:
            raise TypeError(
                '`embedding_matrices` should be torch.Tensor type')

        for i, site in enumerate(self._in_features):
            if domains[i] is None:
                domains[i] = torch.arange(
                    self._phys_dim[site], device=self._mats_env[site].device)
            elif not isinstance(domains[i], torch.Tensor):
                raise TypeError('`domain` should be torch.Tensor type')
            if domains[i].ndim != 1:
                raise ValueError(
                    'Each element of `domain` should be a rank-1 tensor')
            if domains[i].numel() < 1:
                raise ValueError('`domain` tensors cannot be empty')

        shared_embedding = all(aux_domain is domains[0]
                               for aux_domain in domains) and \
            all(aux_embedding is embeddings_arg[0]
                for aux_embedding in embeddings_arg) and \
            all(self._phys_dim[site] == self._phys_dim[self._in_features[0]]
                for site in self._in_features)

        if shared_embedding:
            embedded = self._sample_apply_embedding(
                domain=domains[0],
                embedding=embeddings_arg[0],
                phys_dim=self._phys_dim[self._in_features[0]],
                reference=self._mats_env[self._in_features[0]].tensor)
            embeddings = [embedded] * n_inputs
        else:
            embeddings = [
                self._sample_apply_embedding(
                    domain=aux_domain,
                    embedding=aux_embedding,
                    phys_dim=self._phys_dim[site],
                    reference=self._mats_env[site].tensor)
                for site, aux_domain, aux_embedding in zip(
                    self._in_features, domains, embeddings_arg)
            ]

        matrices = []
        weights = []
        shared_metric = shared_embedding and \
            all(aux_matrix is matrices_arg[0]
                for aux_matrix in matrices_arg)
        for i, (site, aux_domain, aux_embedding, aux_matrix) in enumerate(zip(
                self._in_features, domains, embeddings, matrices_arg)):
            if shared_metric and i > 0:
                matrices.append(matrices[0])
                weights.append(weights[0])
                continue

            phys_dim = self._phys_dim[site]
            if aux_matrix is not None:
                if not isinstance(aux_matrix, torch.Tensor):
                    raise TypeError(
                        '`embedding_matrices` should be torch.Tensor type')
                if aux_matrix.shape != (phys_dim, phys_dim):
                    raise ValueError(
                        '`embedding_matrices` should have shape '
                        '(phys_dim, phys_dim)')
                aux_weights = torch.ones(aux_domain.shape[0],
                                         device=aux_domain.device,
                                         dtype=torch.get_default_dtype())
            elif build_matrices:
                if torch.is_floating_point(aux_domain) and \
                        (aux_domain.numel() > 1):
                    if not (aux_domain[1:] > aux_domain[:-1]).all():
                        raise ValueError(
                            'Continuous `domain` tensors should be strictly '
                            'increasing')
                    integrand = torch.einsum('ds,dt->dst',
                                              aux_embedding,
                                              aux_embedding.conj())
                    aux_matrix = torch.trapezoid(integrand,
                                                  x=aux_domain,
                                                  dim=0)
                    aux_weights = self._trapezoidal_weights(aux_domain)
                else:
                    aux_matrix = torch.einsum('ds,dt->st',
                                              aux_embedding,
                                              aux_embedding.conj())
                    aux_weights = torch.ones(
                        aux_domain.shape[0],
                        device=aux_domain.device,
                        dtype=torch.get_default_dtype())
            else:
                reference = self._mats_env[site]
                aux_matrix = torch.eye(phys_dim,
                                       device=reference.device,
                                       dtype=reference.dtype)
                aux_weights = torch.ones(
                    aux_domain.shape[0],
                    device=aux_domain.device,
                    dtype=reference.tensor.real.dtype)

            matrices.append(aux_matrix)
            weights.append(aux_weights)

        return domains, embeddings, matrices, weights

    def _prepare_in_condition(self,
                              in_condition,
                              n_samples: Optional[int]
                              ) -> Tuple[Optional[Union[torch.Tensor,
                                                        List[torch.Tensor]]], int]:
        """Normalizes embedded input conditions in input-feature order."""
        if in_condition is None:
            return None, 1 if n_samples is None else n_samples

        if isinstance(in_condition, torch.Tensor):
            if in_condition.ndim != (self._n_batches + 2):
                raise ValueError(
                    '`in_condition` should have `n_batches` batch dimensions '
                    'followed by an input-feature and a physical dimension')
            if in_condition.shape[-2] != len(self._in_features):
                raise ValueError(
                    'The penultimate dimension of `in_condition` should be '
                    'the number of input nodes')
            if any(in_condition.shape[-1] != self._phys_dim[site]
                   for site in self._in_features):
                raise ValueError(
                    'The last dimension of `in_condition` should match the '
                    'physical dimensions of all input nodes')

            batch_size = in_condition.shape[:-2].numel()
            in_condition = in_condition.reshape(
                batch_size, len(self._in_features), in_condition.shape[-1])

        elif isinstance(in_condition, Sequence):
            if len(in_condition) != len(self._in_features):
                raise ValueError(
                    '`in_condition` should have as many elements as input nodes')
            conditions = list(in_condition)

            for condition in conditions:
                if not isinstance(condition, torch.Tensor):
                    raise TypeError(
                        'Elements of `in_condition` should be torch.Tensor type')
                if condition.ndim != (self._n_batches + 1):
                    raise ValueError(
                        'Elements of `in_condition` should have `n_batches` '
                        'batch dimensions followed by a physical dimension')

            batch_shape = None
            for site, condition in zip(self._in_features, conditions):
                if condition.shape[-1] != self._phys_dim[site]:
                    raise ValueError(
                        'The last dimension of each `in_condition` tensor '
                        'should match the physical dimension of its input node')
                if batch_shape is None:
                    batch_shape = condition.shape[:-1]
                elif condition.shape[:-1] != batch_shape:
                    raise ValueError(
                        'All `in_condition` tensors should have the same '
                        'batch shape')

            batch_size = torch.Size(batch_shape).numel()
            in_condition = [
                condition.reshape(batch_size, condition.shape[-1])
                for condition in conditions]
        else:
            raise TypeError(
                '`in_condition` should be torch.Tensor, tuple[torch.Tensor] '
                'or list[torch.Tensor] type')

        if n_samples is None:
            n_samples = batch_size
        elif batch_size not in [1, n_samples]:
            raise ValueError(
                '`n_samples` should match the batch size of `in_condition`')

        if batch_size == 1 and n_samples > 1:
            if isinstance(in_condition, torch.Tensor):
                in_condition = in_condition.expand(n_samples, -1, -1)
            else:
                in_condition = [condition.expand(n_samples, -1)
                                for condition in in_condition]

        return in_condition, n_samples

    def _build_sample_right_envs(self,
                                 copied_nodes: List[Node],
                                 matrix_nodes: List[Optional[Node]],
                                 condition_nodes: List[Optional[
                                     Tuple[Node, Node]]],
                                 right_env: Optional[Node],
                                 renormalize: bool) -> List[Node]:
        """Builds reusable right environments with a right-to-left zip-up."""
        right_envs = [None] * self._n_features
        right_envs[-1] = right_env

        for site in range(self._n_features - 1, 0, -1):
            right_env = self._contract_sample_site(
                node=self._mats_env[site],
                copied_node=copied_nodes[site],
                env=right_env,
                matrix_node=matrix_nodes[site],
                data_nodes=condition_nodes[site],
                from_left=False,
                renormalize=renormalize)
            right_envs[site - 1] = right_env

        return right_envs

    @torch.no_grad()
    def sample(self,
               n_samples: Optional[int] = None,
               domain: Optional[Union[torch.Tensor,
                                      Sequence[torch.Tensor]]] = None,
               embedding: Optional[Union[Callable,
                                         Sequence[Callable]]] = None,
               embedding_matrices: Optional[Union[torch.Tensor,
                                                  Sequence[torch.Tensor]]] = None,
               build_matrices: bool = False,
               in_condition: Optional[Union[torch.Tensor,
                                            Sequence[torch.Tensor]]] = None,
               canonical: bool = False,
               renormalize: bool = False,
               return_indices: bool = False,
               generator: Optional[torch.Generator] = None
               ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Samples input configurations from the Born distribution of the MPS.

        The probability of a configuration is proportional to the squared
        modulus of the MPS amplitude. Output nodes are always marginalized.
        If ``in_condition`` is provided, its embedded values are used as a
        right context while all input sites are resampled from left to right.
        Thus, it does not fix any feature. To sample the remaining features
        conditional on fixed values, use :meth:`condition` first and then call
        :meth:`sample` on the returned MPS.

        When ``build_matrices`` is ``True`` and a floating-point domain is
        given, numerical marginalization uses the trapezoidal rule over that
        domain. The same quadrature weights are included in the categorical
        probabilities used to sample each domain point.

        This method internally calls
        :meth:`~tensorkrowch.TensorNetwork.reset`, as sampling constructs a
        temporary double-layer tensor network and contraction environments.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to draw. If ``None`` and ``in_condition`` has a
            batch dimension, that batch size is used. Otherwise the default is
            1.
        domain : torch.Tensor, list[torch.Tensor] or tuple[torch.Tensor], optional
            Values from which each input node is sampled. A single tensor is
            shared by all input nodes. A sequence can be ordered either by the
            sampled input nodes, with ``len(in_features)`` elements, or by all
            MPS sites, with ``n_features`` elements. In the latter case,
            entries at output sites are ignored. If ``None``, each input node
            uses ``torch.arange(phys_dim)``.
        embedding : callable, list[callable] or tuple[callable], optional
            Embedding applied to the domains before contraction. The callable
            should return tensors with shape ``(domain_size, phys_dim)``. If
            ``None``, domains are embedded with
            :func:`~tensorkrowch.embeddings.basis`.
        embedding_matrices : torch.Tensor, list[torch.Tensor] or tuple[torch.Tensor], optional
            Physical metric matrices used to marginalize input nodes. If not
            provided, identity matrices are used unless ``build_matrices`` is
            ``True``.
        build_matrices : bool
            Boolean indicating whether metric matrices should be approximated
            numerically from ``domain`` and ``embedding``. Floating-point
            domains use :func:`torch.trapezoid`; non-floating domains are
            treated as discrete sums.
        in_condition : torch.Tensor, list[torch.Tensor] or tuple[torch.Tensor], optional
            Embedded values for the input nodes used as a right context during
            the sampling sweep. At site ``i``, only values of input nodes to
            its right affect its distribution; every input node is eventually
            sampled again. Output nodes are always marginalized, never
            conditioned. Like data passed to
            :meth:`~tensorkrowch.TensorNetwork.forward`, it can be a tensor
            with shape ``(*batch, len(in_features), phys_dim)`` or a sequence
            with one ``(*batch, phys_dim_i)`` tensor per input node.
        canonical : bool
            If ``True``, the MPS is assumed to be in right-canonical form with
            the orthogonality center at the leftmost site, and right
            environments can be replaced by identities. For PBC, an
            ``in_condition`` context or non-identity physical metrics, this
            option is ignored with a warning and right environments are
            explicitly constructed.
        renormalize : bool
            Boolean indicating whether intermediate environments should be
            normalized during the sweep.
        return_indices : bool
            If ``True``, returns both sampled domain values and sampled domain
            indices.
        generator : torch.Generator, optional
            Random generator passed to :func:`torch.multinomial`.

        Returns
        -------
        torch.Tensor or tuple[torch.Tensor, torch.Tensor]
            If ``return_indices`` is ``False``, returns a tensor with shape
            ``(n_samples, len(in_features))`` containing sampled domain values.
            Otherwise returns ``(samples, indices)``.

        Examples
        --------
        >>> mps = tk.models.MPS(n_features=4,
        ...                     phys_dim=2,
        ...                     bond_dim=5)
        >>> samples = mps.sample(n_samples=10)
        >>> samples.shape
        torch.Size([10, 4])

        >>> domain = torch.linspace(0, 1, 32)
        >>> samples = mps.sample(n_samples=5,
        ...                      domain=domain,
        ...                      embedding=tk.embeddings.unit,
        ...                      build_matrices=True)
        >>> samples.shape
        torch.Size([5, 4])

        >>> mps.out_features = [1]
        >>> fixed_data = tk.embeddings.basis(torch.tensor([0]), dim=2).float()
        >>> cond_mps = mps.condition(fixed_data)
        >>> samples = cond_mps.sample(n_samples=5)
        >>> samples.shape
        torch.Size([5, 3])
        """
        if not isinstance(build_matrices, bool):
            raise TypeError('`build_matrices` should be bool type')
        if not isinstance(canonical, bool):
            raise TypeError('`canonical` should be bool type')
        if not isinstance(renormalize, bool):
            raise TypeError('`renormalize` should be bool type')
        if not isinstance(return_indices, bool):
            raise TypeError('`return_indices` should be bool type')
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError('`generator` should be torch.Generator type')
        if n_samples is not None:
            if not isinstance(n_samples, int):
                raise TypeError('`n_samples` should be int type')
            if n_samples <= 0:
                raise ValueError('`n_samples` should be greater than 0')

        if not self._in_features:
            raise ValueError('Cannot sample an MPS with no input nodes')

        if canonical:
            incompatible = (self._boundary != 'obc') or \
                (in_condition is not None) or \
                (embedding_matrices is not None) or build_matrices
            if incompatible:
                warnings.warn(
                    '`canonical` will be ignored and right environments will '
                    'be explicitly contracted because its assumptions are '
                    'not satisfied')
                canonical = False

        in_condition, n_samples = self._prepare_in_condition(
            in_condition=in_condition,
            n_samples=n_samples)

        domains, embeddings, matrices, quadrature_weights = \
            self._prepare_sample_domains(
            domain=domain,
            embedding=embedding,
            embedding_matrices=embedding_matrices,
            build_matrices=build_matrices)

        # Sampling builds a temporary double layer in the current network.
        if self._resultant_nodes:
            warnings.warn(
                'Resultant nodes will be removed before sampling the MPS')
            self.reset()
        if self._data_nodes:
            self.unset_data_nodes()

        copied_nodes, copied_left, copied_right = self._copy_sample_mps()

        # For PBC, insert an identity on each periodic edge. The combined
        # identity is the initial left environment and keeps the ring connected.
        if self._boundary == 'pbc':
            self._mats_env[-1]['right'].disconnect()
            copied_nodes[-1]['right'].disconnect()

            closure = Node(
                tensor=torch.eye(self._mats_env[-1]['right'].size(),
                                 device=self._mats_env[-1].device,
                                 dtype=self._mats_env[-1].dtype),
                axes_names=('right', 'left'),
                name='virtual_result_sample_closure',
                network=self,
                virtual=True)
            copied_closure = Node(
                shape=closure.shape,
                axes_names=closure.axes_names,
                name='virtual_result_sample_closure_copy',
                network=self,
                virtual=True)
            copied_closure.set_tensor_from(closure)

            self._mats_env[-1]['right'] ^ closure['right']
            closure['left'] ^ self._mats_env[0]['left']
            copied_nodes[-1]['right'] ^ copied_closure['right']
            copied_closure['left'] ^ copied_nodes[0]['left']

            copied_nodes = [node.conj() for node in copied_nodes]
            copied_closure = copied_closure.conj()
            left_env = closure % copied_closure
            right_env = None
        else:
            if canonical:
                for i in range(self._n_features - 1):
                    self._mats_env[i]['right'].disconnect()
                    copied_nodes[i]['right'].disconnect()
                self._mats_env[-1]['right'].disconnect()
                copied_nodes[-1]['right'].disconnect()

            copied_nodes = [node.conj() for node in copied_nodes]
            copied_left = copied_left.conj()
            copied_right = copied_right.conj()
            left_env = self._left_node % copied_left
            right_env = None if canonical \
                else self._right_node % copied_right

        for copied_node in copied_nodes:
            copied_node.reattach_edges(axes=['input'])

        # Connect physical metrics or right-context data to both layers.
        matrix_nodes = [None] * self._n_features
        condition_nodes = [None] * self._n_features
        if in_condition is not None:
            super().set_data_nodes(
                input_edges=[node['input'] for node in self.in_env],
                num_batch_edges=1)
            self.add_data(in_condition)
            data_nodes = list(self.data_nodes.values())

            for site, data_node in zip(self._in_features, data_nodes):
                copied_data = Node(
                    tensor=data_node.tensor,
                    axes_names=data_node.axes_names,
                    name='sample_data_copy',
                    network=self,
                    data=True)
                copied_data = copied_data.conj()
                copied_data.reattach_edges(axes=['feature'])
                copied_data['feature'] ^ copied_nodes[site]['input']
                condition_nodes[site] = (data_node, copied_data)

        input_idx = 0
        for site, (node, copied_node) in enumerate(
                zip(self._mats_env, copied_nodes)):
            if site in self._in_features:
                if condition_nodes[site] is None:
                    matrix_node = Node(
                        tensor=matrices[input_idx],
                        axes_names=('input', 'input_copy'),
                        name='virtual_result_sample_mat',
                        network=self,
                        virtual=True)
                    node['input'] ^ matrix_node['input']
                    matrix_node['input_copy'] ^ copied_node['input']
                    matrix_nodes[site] = matrix_node
                input_idx += 1
            else:
                node['input'] ^ copied_node['input']

        if canonical:
            right_envs = [None] * self._n_features
        else:
            right_envs = self._build_sample_right_envs(
                copied_nodes=copied_nodes,
                matrix_nodes=matrix_nodes,
                condition_nodes=condition_nodes,
                right_env=right_env,
                renormalize=renormalize)

        # Sweep from left to right, sampling one input node at a time.
        samples = []
        sample_indices = []
        input_idx = 0

        for site in range(self._n_features):
            if site in self._in_features:
                node = self._mats_env[site]
                copied_node = copied_nodes[site]
                right_env = right_envs[site]

                if canonical:
                    copied_node.reattach_edges(axes=['right'])
                    right_env = Node(
                        tensor=torch.eye(node['right'].size(),
                                         device=node.device,
                                         dtype=node.dtype),
                        axes_names=('right', 'right_copy'),
                        name='virtual_result_sample_right_env',
                        network=self,
                        virtual=True)
                    node['right'] ^ right_env['right']
                    copied_node['right'] ^ right_env['right_copy']

                # Replace only the physical edges to evaluate all candidates;
                # bond edges remain inherited from the stored environments.
                candidate = node.permute(tuple(range(node.ndim)))
                copied_candidate = copied_node.permute(
                    tuple(range(copied_node.ndim)))
                candidate.reattach_edges(axes=['input'])
                copied_candidate.reattach_edges(axes=['input'])
                candidate.disconnect('input')
                copied_candidate.disconnect('input')

                candidate_data = Node(
                    tensor=embeddings[input_idx],
                    axes_names=('domain_batch', 'feature'),
                    name='sample_candidate_data',
                    network=self,
                    data=True)
                copied_candidate_data = Node(
                    tensor=candidate_data.tensor,
                    axes_names=candidate_data.axes_names,
                    name='sample_candidate_data_copy',
                    network=self,
                    data=True)
                copied_candidate_data = copied_candidate_data.conj()
                copied_candidate_data.reattach_edges(axes=['feature'])
                candidate_data['feature'] ^ candidate['input']
                copied_candidate_data['feature'] ^ \
                    copied_candidate['input']

                candidate = candidate_data @ candidate
                copied_candidate = copied_candidate_data @ copied_candidate
                probs_node = left_env @ candidate
                if right_env is not None:
                    probs_node = probs_node @ right_env
                probs_node = probs_node @ copied_candidate
                if probs_node.is_connected_to(probs_node):
                    probs_node @= probs_node

                domain_axis = probs_node.get_axis_num('domain_batch')
                probs = probs_node.tensor.movedim(domain_axis, -1)
                probs = probs.reshape(-1, embeddings[input_idx].shape[0])
                if probs.shape[0] == 1 and n_samples > 1:
                    probs = probs.expand(n_samples, -1)

                probs = probs * quadrature_weights[input_idx].reshape(1, -1)
                probs = self._probabilities_from_amplitudes(probs)
                ids = torch.multinomial(probs,
                                        num_samples=1,
                                        replacement=True,
                                        generator=generator).squeeze(1)

                sample_indices.append(ids)
                samples.append(domains[input_idx][ids])

                # Absorb the selected values into the reusable left environment.
                selected = node.permute(node.axes_names)
                copied_selected = copied_node.permute(copied_node.axes_names)
                selected.reattach_edges(axes=['input'])
                copied_selected.reattach_edges(axes=['input'])
                selected.disconnect('input')
                copied_selected.disconnect('input')

                selected_data = Node(
                    tensor=embeddings[input_idx][ids],
                    axes_names=('batch', 'feature'),
                    name='sample_selected_data',
                    network=self,
                    data=True)
                copied_selected_data = Node(
                    tensor=selected_data.tensor,
                    axes_names=selected_data.axes_names,
                    name='sample_selected_data_copy',
                    network=self,
                    data=True)
                copied_selected_data = copied_selected_data.conj()
                copied_selected_data.reattach_edges(axes=['feature'])
                selected_data['feature'] ^ selected['input']
                copied_selected_data['feature'] ^ copied_selected['input']

                left_env = self._contract_sample_site(
                    node=selected,
                    copied_node=copied_selected,
                    env=left_env,
                    data_nodes=(selected_data, copied_selected_data),
                    from_left=True,
                    renormalize=renormalize)
                input_idx += 1

            else:
                left_env = self._contract_sample_site(
                    node=self._mats_env[site],
                    copied_node=copied_nodes[site],
                    env=left_env,
                    from_left=True,
                    renormalize=renormalize)

            if canonical and site < (self._n_features - 1):
                right_axes = [axis.name for axis in left_env.axes
                              if ('right' in axis.name) and
                              (not axis.is_batch())]
                left_env.reattach_edges(axes=right_axes)
                for axis in right_axes:
                    left_env.disconnect(axis)

                copied_nodes[site + 1].reattach_edges(axes=['left'])
                left_env[right_axes[0]] ^ self._mats_env[site + 1]['left']
                left_env[right_axes[1]] ^ copied_nodes[site + 1]['left']

        # Collect values in the order of ``in_features``.
        samples = torch.stack(samples, dim=1)
        sample_indices = torch.stack(sample_indices, dim=1)

        self.reset()
        self.unset_data_nodes()
        if self._boundary == 'pbc':
            self._mats_env[-1]['right'] ^ self._mats_env[0]['left']
        elif canonical:
            for i in range(self._n_features - 1):
                self._mats_env[i]['right'] ^ self._mats_env[i + 1]['left']
            self._mats_env[-1]['right'] ^ self._right_node['left']

        if return_indices:
            return samples, sample_indices
        return samples
    ############################
    # SAMPLE: work in progress #
    ############################
    
    @torch.no_grad()
    def canonicalize(self,
                     oc: Optional[int] = None,
                     mode: Text = 'svd',
                     rank: Optional[int] = None,
                     cutoff: Optional[float] = None,
                     atol: Optional[float] = None,
                     rtol: Optional[float] = None,
                     cum_percentage: Optional[float] = None,
                     renormalize: bool = False) -> None:
        r"""
        Turns MPS into canonical form via local SVD/QR decompositions.
        
        To specify the new bond dimensions, the arguments ``rank``,
        ``cum_percentage`` or ``cutoff`` can be specified. These will be used
        equally for all SVD computations.
        
        If none of them are specified, the bond dimensions won't be modified
        if possible. Only when the bond dimension is bigger than the physical
        dimension multiplied by the other bond dimension of the node, it will
        be cropped to that size.
        
        If rank is not specified, the current bond dimensions will be used as
        the rank. That is, the current bond dimensions will be the upper bound
        for the possibly new bond dimensions given by the truncation criterions.
        
        This method internally calls :meth:`~tensorkrowch.TensorNetwork.reset`,
        as canonicalization may change the form of the tensors.
        
        Note
        ----
        Canonicalization relies on repeated SVD/QR decompositions of
        intermediate tensors. If the MPS becomes numerically unstable, for
        instance because tensor norms explode during the sweep, those
        intermediate tensors may contain non-finite values such as ``NaN`` or
        ``Inf``. In that case, the underlying SVD routine may raise
        :class:`torch.linalg.LinAlgError`.

        In practice, using ``renormalize=True`` is often enough to mitigate
        these instabilities. When working close to numerical limits, it is also
        advisable to save the current tensors before calling
        :meth:`canonicalize`. If a decomposition fails, a robust recovery
        strategy is to instantiate a new MPS from the original tensors rather
        than continuing from the partially updated state.
        
        Parameters
        ----------
        oc : int
            Position of the orthogonality center. It should be between 0 and 
            ``n_features - 1``.
        mode : {"svd", "svdr", "qr"}
            Indicates which decomposition should be used to split a node after
            contracting it. See more at :func:`~tensorkrowch.svd_`,
            :func:`~tensorkrowch.svdr_`, :func:`~tensorkrowch.qr_`.
            If mode is "qr", operation :func:`~tensorkrowch.qr_` will be
            performed on nodes at the left of the output node, whilst operation
            :func:`~tensorkrowch.rq_` will be used for nodes at the right.
        rank : int, optional
            Number of singular values to keep.
        cutoff : float, optional
            Minimum singular value to keep. It must be non-negative. Singular
            values ``<= cutoff`` are removed.
        atol : float, optional
            Absolute tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded
            while the accumulated sum of squares is ``<= atol``. It must be
            non-negative.
        rtol : float, optional
            Relative tolerance over the tail sum of squared singular values.
            Starting from the smallest singular value, values are discarded
            while the tail sum of squares divided by the total sum of squares
            is ``<= rtol``. It must be in ``[0, 1]``.
        cum_percentage : float, optional
            Minimum fraction of squared singular-value mass to keep. Equivalent
            to setting ``rtol = 1 - cum_percentage``. It must be in ``[0, 1]``.

            .. math::

                \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
                cum\_percentage
        renormalize : bool
            Indicates whether nodes should be renormalized after SVD/QR
            decompositions. If not, it may happen that the norm explodes as it
            is being accumulated from all nodes. Renormalization aims to avoid
            this undesired behavior by extracting the norm of each node on a
            logarithmic scale after SVD/QR decompositions are computed. Finally,
            the normalization factor is evenly distributed among all nodes of
            the MPS.
            
        Examples
        --------
        >>> mps = tk.models.MPS(n_features=4,
        ...                     phys_dim=2,
        ...                     bond_dim=5)
        >>> mps.canonicalize(rank=3)
        >>> mps.bond_dim
        [3, 3, 3]
        """
        self.reset()

        prev_auto_stack = self._auto_stack
        self.auto_stack = False

        if oc is None:
            oc = self._n_features - 1
        elif (oc < 0) or (oc >= self._n_features):
            raise ValueError('Orthogonality center position `oc` should be '
                             'between 0 and `n_features` - 1')
        
        log_norm = 0
        
        nodes = self._mats_env[:]
        if self._boundary == 'obc':
            nodes[0].tensor[1:] = torch.zeros_like(
                nodes[0].tensor[1:])
            nodes[-1].tensor[..., 1:] = torch.zeros_like(
                nodes[-1].tensor[..., 1:])
        
        # Keep track of which nodes are parameterized
        set_params = [isinstance(node, ParamNode) for node in nodes]
        
        # If mode is svd or svdr and none of the args is provided, the ranks are
        # kept as they were originally
        keep_rank = False
        if rank is None:
            keep_rank = True
        
        for i in range(oc):
            if mode == 'svd':
                result1, result2 = nodes[i]['right'].svd_(
                    side='right',
                    rank=nodes[i]['right'].size() if keep_rank else rank,
                    cutoff=cutoff,
                    atol=atol,
                    rtol=rtol,
                    cum_percentage=cum_percentage)
            elif mode == 'svdr':
                result1, result2 = nodes[i]['right'].svdr_(
                    side='right',
                    rank=nodes[i]['right'].size() if keep_rank else rank,
                    cutoff=cutoff,
                    atol=atol,
                    rtol=rtol,
                    cum_percentage=cum_percentage)
            elif mode == 'qr':
                result1, result2 = nodes[i]['right'].qr_()
            else:
                raise ValueError('`mode` can only be "svd", "svdr" or "qr"')
            
            if renormalize:
                aux_norm = result2.norm()
                if not aux_norm.isinf() and (aux_norm > 0):
                    result2.tensor = result2.tensor / aux_norm
                    log_norm += aux_norm.log()

            nodes[i] = result1.parameterize(set_param=set_params[i])
            nodes[i + 1] = result2

        for i in range(len(nodes) - 1, oc, -1):
            if mode == 'svd':
                result1, result2 = nodes[i]['left'].svd_(
                    side='left',
                    rank=nodes[i]['left'].size() if keep_rank else rank,
                    cutoff=cutoff,
                    atol=atol,
                    rtol=rtol,
                    cum_percentage=cum_percentage)
            elif mode == 'svdr':
                result1, result2 = nodes[i]['left'].svdr_(
                    side='left',
                    rank=nodes[i]['left'].size() if keep_rank else rank,
                    cutoff=cutoff,
                    atol=atol,
                    rtol=rtol,
                    cum_percentage=cum_percentage)
            elif mode == 'qr':
                result1, result2 = nodes[i]['left'].rq_()
            else:
                raise ValueError('`mode` can only be "svd", "svdr" or "qr"')
            
            if renormalize:
                aux_norm = result1.norm()
                if not aux_norm.isinf() and (aux_norm > 0):
                    result1.tensor = result1.tensor / aux_norm
                    log_norm += aux_norm.log()

            nodes[i] = result2.parameterize(set_param=set_params[i])
            nodes[i - 1] = result1

        nodes[oc] = nodes[oc].parameterize(set_param=set_params[oc])
        
        # Rescale
        if renormalize and (log_norm != 0):
            rescale = (log_norm / len(nodes)).exp()
            for node in nodes:
                node.tensor = node.tensor * rescale
        
        # Update variables
        self._mats_env = nodes
        self.update_bond_dim()

        self.auto_stack = prev_auto_stack

    def _project_to_bond_dim(self,
                             nodes: List[AbstractNode],
                             bond_dim: int,
                             side: Text = 'right'):
        """Projects all nodes into a space of dimension ``bond_dim``."""
        device = nodes[0].tensor.device
        dtype = nodes[0].tensor.dtype

        if side == 'left':
            nodes.reverse()
        elif side != 'right':
            raise ValueError('`side` can only be \'left\' or \'right\'')

        for node in nodes:
            if not node['input'].is_dangling():
                self.delete_node(node.neighbours('input'))

        line_mat_nodes = []
        phys_dim_lst = []
        proj_mat_node = None
        for j in range(len(nodes)):
            phys_dim_lst.append(nodes[j]['input'].size())
            if bond_dim <= torch.tensor(phys_dim_lst).prod().item():
                proj_mat_node = Node(shape=(*phys_dim_lst, bond_dim),
                                     axes_names=(*(['input'] * len(phys_dim_lst)),
                                                 'bond_dim'),
                                     name=f'proj_mat_node_{side}',
                                     network=self)

                proj_mat_node.tensor = torch.eye(
                    torch.tensor(phys_dim_lst).prod().int().item(),
                    bond_dim).view(*phys_dim_lst, -1).to(dtype).to(device)
                for k in range(j + 1):
                    nodes[k]['input'] ^ proj_mat_node[k]

                aux_result = proj_mat_node
                for k in range(j + 1):
                    aux_result @= nodes[k]
                line_mat_nodes.append(aux_result)  # bond_dim x left x right
                break

        if proj_mat_node is None:
            bond_dim = torch.tensor(phys_dim_lst).prod().int().item()
            proj_mat_node = Node(shape=(*phys_dim_lst, bond_dim),
                                 axes_names=(*(['input'] * len(phys_dim_lst)),
                                             'bond_dim'),
                                 name=f'proj_mat_node_{side}',
                                 network=self)

            proj_mat_node.tensor = torch.eye(
                torch.tensor(phys_dim_lst).prod().int().item(),
                bond_dim).view(*phys_dim_lst, -1).to(dtype).to(device)
            for k in range(j + 1):
                nodes[k]['input'] ^ proj_mat_node[k]

            aux_result = proj_mat_node
            for k in range(j + 1):
                aux_result @= nodes[k]
            line_mat_nodes.append(aux_result)

        k = j + 1
        while k < len(nodes):
            phys_dim = nodes[k]['input'].size()
            proj_vec_node = Node(shape=(phys_dim,),
                                 axes_names=('input',),
                                 name=f'proj_vec_node_{side}_({k})',
                                 network=self)

            proj_vec_node.tensor = torch.eye(phys_dim, 1).squeeze()\
                .to(dtype).to(device)
            nodes[k]['input'] ^ proj_vec_node['input']
            line_mat_nodes.append(proj_vec_node @ nodes[k])

            k += 1

        line_mat_nodes.reverse()
        result = line_mat_nodes[0]
        for node in line_mat_nodes[1:]:
            result @= node

        return result  # bond_dim x left/right

    def _aux_canonicalize_univocal(self,
                                   nodes: List[AbstractNode],
                                   idx: int,
                                   left_nodeL: AbstractNode):
        """Returns canonicalized version of the tensor at site ``idx``."""
        L = nodes[idx]  # left x input x right
        left_nodeC = None

        if idx > 0:
            # bond_dim[-1] x input  x right  /  bond_dim[-1] x input
            L = left_nodeL @ L

        L = L.tensor

        if idx < (self._n_features - 1):
            bond_dim = self._bond_dim[idx]

            prod_phys_left = 1
            for i in range(idx + 1):
                prod_phys_left *= self.phys_dim[i]
            bond_dim = min(bond_dim, prod_phys_left)

            prod_phys_right = 1
            for i in range(idx + 1, self._n_features):
                prod_phys_right *= self.phys_dim[i]
            bond_dim = min(bond_dim, prod_phys_right)

            if bond_dim < self._bond_dim[idx]:
                self._bond_dim[idx] = bond_dim

            left_nodeC = self._project_to_bond_dim(nodes=nodes[:idx + 1],
                                                   bond_dim=bond_dim,
                                                   side='left')  # bond_dim x right
            right_node = self._project_to_bond_dim(nodes=nodes[idx + 1:],
                                                   bond_dim=bond_dim,
                                                   side='right')  # bond_dim x left

            C = left_nodeC @ right_node  # bond_dim x bond_dim
            C = torch.linalg.inv(C.tensor)

            if idx == 0:
                L @= right_node.tensor.t()  # input x bond_dim
                L @= C
            else:
                shape_L = L.shape
                # (bond_dim[-1] * input) x bond_dim
                L = (L.view(-1, L.shape[-1]) @ right_node.tensor.t())
                L @= C
                L = L.view(*shape_L[:-1], right_node.shape[0])

        return L, left_nodeC

    @torch.no_grad()
    def canonicalize_univocal(self):
        """
        Turns MPS into the univocal canonical form defined `here
        <https://arxiv.org/abs/2202.12319>`_.

        This method internally calls :meth:`~tensorkrowch.TensorNetwork.reset`,
        as canonicalization may change the form of the tensors.
        """
        if self._boundary != 'obc':
            raise ValueError('`canonicalize_univocal` can only be used if '
                             'boundary is `obc`')

        self.reset()

        prev_auto_stack = self._auto_stack
        self.auto_stack = False

        nodes = self._mats_env[:]
        for node in nodes:
            if not node['input'].is_dangling():
                node['input'].disconnect()
        
        # Boundary is 'obc'
        nodes[0] = self._left_node @ nodes[0]
        nodes[0].reattach_edges(axes=['input'])
        
        nodes[-1] = nodes[-1] @ self._right_node
        nodes[-1].reattach_edges(axes=['input'])

        new_tensors = []
        left_nodeC = None
        for i in range(self._n_features):
            tensor, left_nodeC = self._aux_canonicalize_univocal(
                nodes=nodes,
                idx=i,
                left_nodeL=left_nodeC)
            new_tensors.append(tensor)
        
        for i, node in enumerate(nodes):
            if i < (self._n_features - 1):
                if self._bond_dim[i] < node['right'].size():
                    node['right'].change_size(self._bond_dim[i])

            if not node['input'].is_dangling():
                self.delete_node(node.neighbours('input'))
        
        self.reset()
        self.initialize(tensors=new_tensors)
        self.update_bond_dim()

        for node, data_node in zip(self.in_env, self._data_nodes.values()):
            node['input'] ^ data_node['feature']

        self.auto_stack = prev_auto_stack


class UMPS(MPS):  # MARK: UMPS
    """
    Class for Uniform (translationally invariant) Matrix Product States. It is
    the uniform version of :class:`MPS`, that is, all nodes share the same
    tensor. Thus this class cannot have different physical or bond dimensions
    for each node, and boundary conditions are always periodic (``"pbc"``).
    
    |
    
    For a more detailed list of inherited properties and methods,
    check :class:`MPS`.

    Parameters
    ----------
    n_features : int
        Number of nodes that will be in ``mats_env``.
    phys_dim : int, optional
        Physical dimension.
    bond_dim : int, optional
        Bond dimension.
    boundary : {"obc", "pbc"}
        String indicating whether periodic or open boundary conditions should
        be used.
    tensor: torch.Tensor, optional
        Instead of providing ``phys_dim`` and ``bond_dim``, a single tensor
        can be provided. ``n_features`` is still needed to specify how many
        times the tensor should be used to form a finite MPS. The tensor
        should be rank-3, with its first and last dimensions being equal.
    in_features: list[int] or tuple[int], optional
        List of indices indicating the positions of the MPS nodes that will be
        considered as input nodes. These nodes will have a neighbouring data
        node connected to its ``"input"`` edge when the :meth:`set_data_nodes`
        method is called. ``in_features`` is the complementary set of
        ``out_features``, so it is only required to specify one of them.
    out_features: list[int] or tuple[int], optional
        List of indices indicating the positions of the MPS nodes that will be
        considered as output nodes. These nodes will be left with their ``"input"``
        edges open when contrating the network. If ``marginalize_output`` is
        set to ``True`` in :meth:`contract`, the network will be connected to
        itself at these nodes, and contracted. ``out_features`` is the
        complementary set of ``in_features``, so it is only required to specify
        one of them.
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit"}, optional
        Initialization method. Check :meth:`initialize` for a more detailed
        explanation of the different initialization methods.
    parameterized : bool, optional
        Boolean indicating whether UMPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
    Examples
    --------
    >>> mps = tk.models.UMPS(n_features=4,
    ...                      phys_dim=2,
    ...                      bond_dim=5)
    >>> for node in mps.mats_env:
    ...     assert node.tensor_address() == 'virtual_uniform'
    ...
    >>> data = torch.ones(20, 4, 2) # batch_size x n_features x feature_size
    >>> result = mps(data)
    >>> result.shape
    torch.Size([20])
    """

    def __init__(self,
                 n_features: int,
                 phys_dim: Optional[int] = None,
                 bond_dim: Optional[int] = None,
                 boundary: Text = 'pbc',
                 tensor: Optional[torch.Tensor] = None,
                 in_features: Optional[Sequence[int]] = None,
                 out_features: Optional[Sequence[int]] = None,
                 n_batches: int = 1,
                 init_method: Text = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs) -> None:
        
        tensors = None
        
        # n_features
        if not isinstance(n_features, int):
            raise TypeError('`n_features` should be int type')
        elif n_features < 1:
            raise ValueError('`n_features` should be at least 1')
        
        if tensor is not None:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError('`tensor` should be torch.Tensor type')
            if tensor.ndim != 3:
                raise ValueError('`tensor` should be a rank-3 tensor')
            if tensor.shape[0] != tensor.shape[2]:
                raise ValueError('`tensor` first and last dimensions should'
                                 ' be equal')
            if phys_dim is None:
                phys_dim = tensor.shape[1]
            if bond_dim is None:
                bond_dim = tensor.shape[0]
            if boundary == 'pbc':
                tensors = [tensor] * n_features
        
        super().__init__(n_features=n_features,
                         phys_dim=phys_dim,
                         bond_dim=bond_dim,
                         boundary=boundary,
                         tensors=tensors,
                         in_features=in_features,
                         out_features=out_features,
                         n_batches=n_batches,
                         init_method=init_method,
                         parameterized=parameterized,
                         device=device,
                         dtype=dtype,
                         **kwargs)
        self.name = 'umps'
        if (tensor is not None) and (boundary == 'obc'):
            self.initialize(tensors=[tensor],
                            init_method=None)

    def _make_nodes(self, parameterized: bool = True) -> None:
        """Creates all the nodes of the MPS."""
        super()._make_nodes(parameterized)
        
        # Virtual node
        node_cls = ParamNode if parameterized else Node
        bond_dim = self._bond_dim[0] if self._bond_dim else 1
        uniform_memory = node_cls(shape=(bond_dim,
                                         self._phys_dim[0],
                                         bond_dim),
                                  axes_names=('left', 'input', 'right'),
                                  name='virtual_uniform',
                                  network=self,
                                  virtual=True)
        self.uniform_memory = uniform_memory
        
        for node in self._mats_env:
            node.set_tensor_from(uniform_memory)
    
    def _make_canonical(self,
                        device: Optional[torch.device] = None,
                        dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
        """
        Creates random unitaries to initialize the MPS in canonical form with
        orthogonality center at the rightmost node. Unitaries in nodes are
        scaled so that the total norm squared of the initial MPS is the product
        of all the physical dimensions.
        """
        node = self.uniform_memory
        node_shape = node.shape
        aux_shape = (node.shape[:2].numel(), node.shape[2])
        
        size = max(aux_shape[0], aux_shape[1])
        phys_dim = node_shape[1]
        
        tensor = random_unitary(size, device=device, dtype=dtype)
        tensor = tensor[:min(aux_shape[0], size), :min(aux_shape[1], size)]
        tensor = tensor.reshape(*node_shape)
        tensor = tensor * sqrt(phys_dim)
        return tensor
    
    def _make_unitaries(self,
                        device: Optional[torch.device] = None,
                        dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
        """
        Creates random unitaries to initialize the MPS nodes as stacks of
        unitaries.
        """
        node = self.uniform_memory
        node_shape = node.shape
        
        units = []
        for _ in range(node_shape[1]):
            tensor = random_unitary(node_shape[0], device=device, dtype=dtype)
            units.append(tensor)
        tensor = torch.stack(units, dim=1)
        return tensor

    def initialize(self,
                   tensors: Optional[Sequence[torch.Tensor]] = None,
                   init_method: Optional[Text] = 'randn',
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None,
                   **kwargs: float) -> None:
        """
        Initializes the common tensor of the :class:`UMPS`. It can be called
        when instantiating the model, or to override the existing nodes' tensors.
        
        There are different methods to initialize the nodes:
        
        * ``{"zeros", "ones", "copy", "rand", "randn"}``: The tensor is
          initialized calling :meth:`~tensorkrowch.AbstractNode.set_tensor` with
          the given method, ``device``, ``dtype`` and ``kwargs``.
        
        * ``"randn_eye"``: Tensor is initialized as in this
          `paper <https://arxiv.org/abs/1605.03795>`_, adding identities at the
          top of a random gaussian tensor. In this case, ``std`` should be
          specified with a low value, e.g., ``std = 1e-9``.
        
        * ``"unit"``: Tensor is initialized as a stack of random unitaries. This,
          combined (at least) with an embedding of the inputs as elements of
          the computational basis (:func:`~tensorkrowch.embeddings.discretize`
          combined with :func:`~tensorkrowch.embeddings.basis`)
        
        * ``"canonical"```: MPS is initialized in canonical form with a squared
          norm `close` to the product of all the physical dimensions (if bond
          dimensions are bigger than the powers of the physical dimensions,
          the norm could vary).
        
        Parameters
        ----------
        tensors : list[torch.Tensor] or tuple[torch.Tensor], optional
            Sequence of a single tensor to set in each of the MPS nodes. The
            tensor should be rank-3, with its first and last dimensions being
            equal.
        init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensors if ``init_method`` is provided.
        dtype : torch.dtype, optional
            Dtype of the tensor if ``init_method`` is provided.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        """
        node = self.uniform_memory
        
        if init_method == 'unit':
            tensors = [self._make_unitaries(device=device, dtype=dtype)]
        elif init_method == 'canonical':
            tensors = [self._make_canonical(device=device, dtype=dtype)]
        
        if tensors is not None:
            node.tensor = tensors[0]
            device = tensors[0].device
            dtype = tensors[0].dtype
        
        elif init_method is not None:
            add_eye = False
            if init_method == 'randn_eye':
                init_method = 'randn'
                add_eye = True
            
            node.set_tensor(init_method=init_method,
                            device=device,
                            dtype=dtype,
                            **kwargs)
            if add_eye:
                aux_tensor = node.tensor.detach()
                aux_tensor[:, 0, :] += torch.eye(node.shape[0],
                                                 node.shape[2],
                                                 device=device,
                                                 dtype=dtype)
                node.tensor = aux_tensor
        
        if self._boundary == 'obc':
            self._left_node.set_tensor(init_method='copy',
                                       device=device,
                                       dtype=dtype)
            self._right_node.set_tensor(init_method='copy',
                                        device=device,
                                        dtype=dtype)
    
    def copy(self, share_tensors: bool = False) -> 'UMPS':
        """
        Creates a copy of the :class:`UMPS`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether the common tensor in the copied UMPS
            should be set as the tensor in the current UMPS (``True``), or
            cloned (``False``). In the former case, the tensor in both UMPS's
            will be the same, which might be useful if one needs more than one
            copy of a UMPS, but wants to compute all the gradients with respect
            to the same, unique, tensor.

        Returns
        -------
        UMPS
        """
        new_mps = UMPS(n_features=self._n_features,
                       phys_dim=self._phys_dim[0],
                       bond_dim=self._bond_dim[0] if self._bond_dim else 1,
                       boundary=self._boundary,
                       tensor=None,
                       in_features=self._in_features,
                       out_features=self._out_features,
                       n_batches=self._n_batches,
                       init_method=None,
                       parameterized=isinstance(self.uniform_memory, ParamNode),
                       device=None,
                       dtype=None)
        new_mps.name = self.name + '_copy'
        if share_tensors:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor
            if self._boundary == 'obc':
                new_mps._left_node.tensor = self._left_node.tensor
                new_mps.right_node.tensor = self.right_node.tensor
        else:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor.clone()
            if self._boundary == 'obc':
                new_mps._left_node.tensor = self._left_node.tensor.clone()
                new_mps.right_node.tensor = self.right_node.tensor.clone()
        return new_mps
    
    def parameterize(self,
                     set_param: bool = True,
                     override: bool = False) -> 'TensorNetwork':
        """
        Parameterizes all nodes of the MPS. If there are ``resultant`` nodes
        in the MPS, it will be first :meth:`~tensorkrowch.TensorNetwork.reset`.

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
            net = self.copy(share_tensors=False)
        
        for i in range(self._n_features):
            net._mats_env[i] = net._mats_env[i].parameterize(set_param=set_param)
        
        # It is important that uniform_memory is parameterized after the rest
        # of the nodes
        net.uniform_memory = net.uniform_memory.parameterize(set_param=set_param)
        
        # Tensor addresses have to be reassigned to reference
        # the uniform memory
        for node in net._mats_env:
            node.set_tensor_from(net.uniform_memory)
            
        return net
    
    def canonicalize(self,
                     oc: Optional[int] = None,
                     mode: Text = 'svd',
                     rank: Optional[int] = None,
                     cutoff: Optional[float] = None,
                     atol: Optional[float] = None,
                     rtol: Optional[float] = None,
                     cum_percentage: Optional[float] = None,
                     renormalize: bool = False) -> None:
        """:meta private:"""
        raise NotImplementedError(
            '`canonicalize` not implemented for UMPS')
    
    def canonicalize_univocal(self):
        """:meta private:"""
        raise NotImplementedError(
            '`canonicalize_univocal` not implemented for UMPS')


class MPSLayer(MPS):  # MARK: MPSLayer
    """
    Class for Matrix Product States with a single output node. That is, this
    MPS has :math:`n` nodes, being :math:`n-1` input nodes connected to ``data``
    nodes (nodes that will contain the data tensors), and one output node,
    whose physical dimension (``out_dim``) is used as the label (for
    classification tasks).
    
    Besides, since this class has an output edge, when contracting the whole
    tensor network (with input data), the result will be a vector that can be
    plugged into the next layer (being this other tensor network or a neural
    network layer).
    
    If the physical dimensions of all the input nodes (``in_dim``) are equal,
    the input data tensor can be passed as a single tensor. Otherwise, it would
    have to be passed as a list of tensors with different sizes.
    
    |
    
    That is, ``MPSLayer`` is equivalent to :class:`MPS` with
    ``out_features = [out_position]``. However, ``in_features`` and
    ``out_features`` are still free to be changed if necessary, even though
    this may change the expected behaviour of the ``MPSLayer``. The expected
    behaviour can be recovered by setting ``out_features = [out_position]``
    again.
    
    |
    
    For a more detailed list of inherited properties and methods,
    check :class:`MPS`.

    Parameters
    ----------
    n_features : int, optional
        Number of nodes that will be in ``mats_env``. That is, number of nodes
        without taking into account ``left_node`` and ``right_node``. This also
        includes the output node, so if one wants to instantiate an ``MPSLayer``
        for a dataset with ``n`` features, it should be ``n_features = n + 1``,
        to account for the output node.
    in_dim : int, list[int] or tuple[int], optional
        Input dimension(s). Equivalent to the physical dimension(s) but only
        for input nodes. If given as a sequence, its length should be equal to
        ``n_features - 1``, since these are the input dimensions of the input
        nodes.
    out_dim : int, optional
        Output dimension (labels) for the output node. Equivalent to the
        physical dimension of the output node.
    bond_dim : int, list[int] or tuple[int], optional
        Bond dimension(s). If given as a sequence, its length should be equal
        to ``n_features`` (if ``boundary = "pbc"``) or ``n_features - 1`` (if
        ``boundary = "obc"``). The i-th bond dimension is always the dimension
        of the right edge of the i-th node (including output node).
    out_position : int, optional
        Position of the output node (label). Should be between 0 and
        ``n_features - 1``. If ``None``, the output node will be located at the
        middle of the MPS.
    boundary : {"obc", "pbc"}
        String indicating whether periodic or open boundary conditions should
        be used.
    tensors: list[torch.Tensor] or tuple[torch.Tensor], optional
        Instead of providing ``n_features``, ``in_dim``, ``out_dim``,
        ``bond_dim`` and ``boundary``, a list of MPS tensors can be provided.
        In such case, all mentioned attributes will be inferred from the given
        tensors. All tensors should be rank-3 tensors, with shape ``(bond_dim,
        phys_dim, bond_dim)``. If the first and last elements are rank-2 tensors,
        with shapes ``(phys_dim, bond_dim)``, ``(bond_dim, phys_dim)``,
        respectively, the inferred boundary conditions will be "obc". Also, if
        ``tensors`` contains a single element, it can be rank-1 ("obc") or
        rank-3 ("pbc").
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (e.g. one edge for data batched, other edge for
        image patches in convolutional layers).
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
        Initialization method. Check :meth:`initialize` for a more detailed
        explanation of the different initialization methods.
    parameterized : bool, optional
        Boolean indicating whether MPSLayer nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
    Examples
    --------
    ``MPSLayer`` with same input dimensions:
    
    >>> mps_layer = tk.models.MPSLayer(n_features=4,
    ...                                in_dim=2,
    ...                                out_dim=10,
    ...                                bond_dim=5)
    >>> data = torch.ones(20, 3, 2) # batch_size x (n_features - 1) x feature_size
    >>> result = mps_layer(data)
    >>> result.shape
    torch.Size([20, 10])
    
    ``MPSLayer`` with different input dimensions:
    
    >>> mps_layer = tk.models.MPSLayer(n_features=4,
    ...                                in_dim=list(range(2, 5)),
    ...                                out_dim=10,
    ...                                bond_dim=5)
    >>> data = [torch.ones(20, i)
    ...         for i in range(2, 5)] # (n_features - 1) * [batch_size x feature_size]
    >>> result = mps_layer(data)
    >>> result.shape
    torch.Size([20, 10])
    """

    def __init__(self,
                 n_features: Optional[int] = None,
                 in_dim: Optional[Union[int, Sequence[int]]] = None,
                 out_dim: Optional[int] = None,
                 bond_dim: Optional[Union[int, Sequence[int]]] = None,
                 out_position: Optional[int] = None,
                 boundary: Text = 'obc',
                 tensors: Optional[Sequence[torch.Tensor]] = None,
                 n_batches: int = 1,
                 init_method: Text = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs) -> None:
        
        phys_dim = None
        
        if tensors is not None:
            if not isinstance(tensors, (list, tuple)):
                raise TypeError('`tensors` should be a tuple[torch.Tensor] or '
                                'list[torch.Tensor] type')
            n_features = len(tensors)
        else:
            if not isinstance(n_features, int):
                raise TypeError('`n_features` should be int type')
            
        # out_position
        if out_position is None:
            out_position = n_features // 2
        if (out_position < 0) or (out_position > n_features):
            raise ValueError(
                f'`out_position` should be between 0 and {n_features}')
        self._out_position = out_position
        
        if tensors is None:
            # in_dim
            if isinstance(in_dim, (list, tuple)):
                if len(in_dim) != (n_features - 1):
                    raise ValueError(
                        'If `in_dim` is given as a sequence of int, its '
                        'length should be equal to `n_features` - 1')
                else:
                    for dim in in_dim:
                        if not isinstance(dim, int):
                            raise TypeError(
                                '`in_dim` should be int, tuple[int] or '
                                'list[int] type')
                in_dim = list(in_dim)
            elif isinstance(in_dim, int):
                in_dim = [in_dim] * (n_features - 1)
            else:
                if n_features == 1:
                    in_dim = []
                else:
                    raise TypeError(
                        '`in_dim` should be int, tuple[int] or list[int] type')

            # out_dim
            if not isinstance(out_dim, int):
                raise TypeError('`out_dim` should be int type')
            
            # phys_dim
            phys_dim = in_dim[:out_position] + [out_dim] + in_dim[out_position:]
            
        super().__init__(n_features=n_features,
                         phys_dim=phys_dim,
                         bond_dim=bond_dim,
                         boundary=boundary,
                         tensors=tensors,
                         in_features=None,
                         out_features=[out_position],
                         n_batches=n_batches,
                         init_method=init_method,
                         parameterized=parameterized,
                         device=device,
                         dtype=dtype,
                         **kwargs)
        self.name = 'mpslayer'
        self._in_dim = self._phys_dim[:out_position] + \
            self._phys_dim[(out_position + 1):]
        self._out_dim = self._phys_dim[out_position]
    
    @property
    def in_dim(self) -> List[int]:
        """Returns input dimensions."""
        return self._in_dim

    @property
    def out_dim(self) -> int:
        """
        Returns the output dimension, that is, the number of labels in the
        output node. Same as ``in_dim`` for input nodes.
        """
        return self._out_dim

    @property
    def out_position(self) -> int:
        """Returns position of the output node (label)."""
        return self._out_position
    
    @property
    def out_node(self) -> ParamNode:
        """Returns the output node."""
        return self._mats_env[self._out_position]
    
    def _make_canonical(self,
                        device: Optional[torch.device] = None,
                        dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
        """
        Creates random unitaries to initialize the MPS in canonical form with
        orthogonality center at the rightmost node. Unitaries in nodes are
        scaled so that the total norm squared of the initial MPS is the product
        of all the physical dimensions.
        """
        # Left nodes
        left_tensors = []
        for i, node in enumerate(self._mats_env[:self._out_position]):
            if self._boundary == 'obc':
                if i == 0:
                    node_shape = node.shape[1:]
                    aux_shape = node_shape
                    phys_dim = node_shape[0]
                else:
                    node_shape = node.shape
                    aux_shape = (node.shape[:2].numel(), node.shape[2])
                    phys_dim = node_shape[1]
            else:
                node_shape = node.shape
                aux_shape = (node.shape[:2].numel(), node.shape[2])
                phys_dim = node_shape[1]
            size = max(aux_shape[0], aux_shape[1])
            
            tensor = random_unitary(size, device=device, dtype=dtype)
            tensor = tensor[:min(aux_shape[0], size), :min(aux_shape[1], size)]
            tensor = tensor.reshape(*node_shape)
            
            left_tensors.append(tensor * sqrt(phys_dim))
        
        # Output node
        out_tensor = torch.randn(self.out_node.shape,
                                 device=device,
                                 dtype=dtype)
        phys_dim = out_tensor.shape[1]
        if self._boundary == 'obc':
            if self._out_position == 0:
                out_tensor = out_tensor[0]
            if self._out_position == (self._n_features - 1):
                out_tensor = out_tensor[..., 0]
        out_tensor = out_tensor / out_tensor.norm() * sqrt(phys_dim)
        
        # Right nodes
        right_tensors = []
        for i, node in enumerate(self._mats_env[-1:self._out_position:-1]):
            if self._boundary == 'obc':
                if i == 0:
                    node_shape = node.shape[:2]
                    aux_shape = node_shape
                    phys_dim = node_shape[1]
                else:
                    node_shape = node.shape
                    aux_shape = (node.shape[0], node.shape[1:].numel())
                    phys_dim = node_shape[1]
            else:
                node_shape = node.shape
                aux_shape = (node.shape[0], node.shape[1:].numel())
                phys_dim = node_shape[1]
            size = max(aux_shape[0], aux_shape[1])
            
            tensor = random_unitary(size, device=device, dtype=dtype)
            tensor = tensor[:min(aux_shape[0], size), :min(aux_shape[1], size)]
            tensor = tensor.reshape(*node_shape)
            
            right_tensors.append(tensor * sqrt(phys_dim))
        right_tensors.reverse()
        
        # All tensors
        tensors = left_tensors + [out_tensor] + right_tensors
        return tensors
    
    def _make_unitaries(self,
                        device: Optional[torch.device] = None,
                        dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
        """
        Creates random unitaries to initialize the MPS nodes as stacks of
        unitaries.
        """
        # Left_nodes
        left_tensors = []
        for i, node in enumerate(self._mats_env[:self._out_position]):
            units = []
            size = max(node.shape[0], node.shape[2])
            if self._boundary == 'obc':
                if i == 0:
                    size_1 = 1
                    size_2 = min(node.shape[2], size)
                else:
                    size_1 = min(node.shape[0], size)
                    size_2 = min(node.shape[2], size)
            else:
                size_1 = min(node.shape[0], size)
                size_2 = min(node.shape[2], size)
            
            for _ in range(node.shape[1]):
                tensor = random_unitary(size, device=device, dtype=dtype)
                tensor = tensor[:size_1, :size_2]
                units.append(tensor)
            
            units = torch.stack(units, dim=1)
            
            if self._boundary == 'obc':
                if i == 0:
                    left_tensors.append(units.squeeze(0))
                else:
                    left_tensors.append(units)
            else:    
                left_tensors.append(units)
        
        # Output node
        out_tensor = torch.randn(self.out_node.shape,
                                 device=device,
                                 dtype=dtype)
        if self._boundary == 'obc':
            if self._out_position == 0:
                out_tensor = out_tensor[0]
            if self._out_position == (self._n_features - 1):
                out_tensor = out_tensor[..., 0]
        
        # Right nodes
        right_tensors = []
        for i, node in enumerate(self._mats_env[-1:self._out_position:-1]):
            units = []
            size = max(node.shape[0], node.shape[2])
            if self._boundary == 'obc':
                if i == 0:
                    size_1 = min(node.shape[0], size)
                    size_2 = 1
                else:
                    size_1 = min(node.shape[0], size)
                    size_2 = min(node.shape[2], size)
            else:
                size_1 = min(node.shape[0], size)
                size_2 = min(node.shape[2], size)
            
            for _ in range(node.shape[1]):
                tensor = random_unitary(size, device=device, dtype=dtype).H
                tensor = tensor[:size_1, :size_2]
                units.append(tensor)
            
            units = torch.stack(units, dim=1)
            
            if self._boundary == 'obc':
                if i == 0:
                    right_tensors.append(units.squeeze(-1))
                else:
                    right_tensors.append(units)
            else:    
                right_tensors.append(units)
        right_tensors.reverse()
        
        # All tensors
        tensors = left_tensors + [out_tensor] + right_tensors
        return tensors
    
    def initialize(self,
                   tensors: Optional[Sequence[torch.Tensor]] = None,
                   init_method: Optional[Text] = 'randn',
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None,
                   **kwargs: float) -> None:
        """
        Initializes all the nodes of the :class:`MPSLayer`. It can be called when
        instantiating the model, or to override the existing nodes' tensors.
        
        There are different methods to initialize the nodes:
        
        * ``{"zeros", "ones", "copy", "rand", "randn"}``: Each node is
          initialized calling :meth:`~tensorkrowch.AbstractNode.set_tensor` with
          the given method, ``device``, ``dtype`` and ``kwargs``.
        
        * ``"randn_eye"``: Nodes are initialized as in this
          `paper <https://arxiv.org/abs/1605.03795>`_, adding identities at the
          top of random gaussian tensors. In this case, ``std`` should be
          specified with a low value, e.g., ``std = 1e-9``.
        
        * ``"unit"``: Nodes are initialized as stacks of random unitaries. This,
          combined (at least) with an embedding of the inputs as elements of
          the computational basis (:func:`~tensorkrowch.embeddings.discretize`
          combined with :func:`~tensorkrowch.embeddings.basis`)
        
        * ``"canonical"```: MPS is initialized in canonical form with a squared
          norm `close` to the product of all the physical dimensions (if bond
          dimensions are bigger than the powers of the physical dimensions,
          the norm could vary). Th orthogonality center is at the output node.
        
        Parameters
        ----------
        tensors : list[torch.Tensor] or tuple[torch.Tensor], optional
            Sequence of tensors to set in each of the MPS nodes. If ``boundary``
            is ``"obc"``, all tensors should be rank-3, except the first and
            last ones, which can be rank-2, or rank-1 (if the first and last are
            the same). If ``boundary`` is ``"pbc"``, all tensors should be
            rank-3.
        init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensors if ``init_method`` is provided.
        dtype : torch.dtype, optional
            Dtype of the tensor if ``init_method`` is provided.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        """
        if init_method == 'unit':
            tensors = self._make_unitaries(device=device, dtype=dtype)
        elif init_method == 'canonical':
            tensors = self._make_canonical(device=device, dtype=dtype)

        if tensors is not None:
            if len(tensors) != self._n_features:
                raise ValueError('`tensors` should be a sequence of `n_features`'
                                 ' elements')
            
            if self._boundary == 'obc':
                tensors = tensors[:]
                
                if device is None:
                    device = tensors[0].device
                if dtype is None:
                    dtype = tensors[0].dtype
                
                if len(tensors) == 1:
                    tensors[0] = tensors[0].reshape(1, -1, 1)
                else:
                    # Left node
                    aux_tensor = torch.zeros(*self._mats_env[0].shape,
                                             device=device,
                                             dtype=dtype)
                    aux_tensor[0] = tensors[0]
                    tensors[0] = aux_tensor
                    
                    # Right node
                    aux_tensor = torch.zeros(*self._mats_env[-1].shape,
                                             device=device,
                                             dtype=dtype)
                    aux_tensor[..., 0] = tensors[-1]
                    tensors[-1] = aux_tensor
                
            for tensor, node in zip(tensors, self._mats_env):
                node.tensor = tensor
                
        elif init_method is not None:
            add_eye = False
            if init_method == 'randn_eye':
                init_method = 'randn'
                add_eye = True
                
            for i, node in enumerate(self._mats_env):
                node.set_tensor(init_method=init_method,
                                device=device,
                                dtype=dtype,
                                **kwargs)
                if add_eye:
                    aux_tensor = node.tensor.detach()
                    eye_tensor = torch.eye(node.shape[0],
                                           node.shape[2],
                                           device=device,
                                           dtype=dtype)
                    if i == self._out_position:
                        eye_tensor = eye_tensor.unsqueeze(1)
                        eye_tensor = eye_tensor.expand(node.shape)
                        aux_tensor += eye_tensor
                    else:
                        aux_tensor[:, 0, :] += eye_tensor
                    node.tensor = aux_tensor
                
                if self._boundary == 'obc':
                    aux_tensor = torch.zeros(*node.shape,
                                             device=device,
                                             dtype=dtype)
                    if i == 0:
                        # Left node
                        aux_tensor[0] = node.tensor[0]
                        node.tensor = aux_tensor
                    elif i == (self._n_features - 1):
                        # Right node
                        aux_tensor[..., 0] = node.tensor[..., 0]
                        node.tensor = aux_tensor
        
        if self._boundary == 'obc':
            self._left_node.set_tensor(init_method='copy',
                                       device=device,
                                       dtype=dtype)
            self._right_node.set_tensor(init_method='copy',
                                        device=device,
                                        dtype=dtype)

    def copy(self, share_tensors: bool = False) -> 'MPSLayer':
        """
        Creates a copy of the :class:`MPSLayer`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether tensors in the copied MPSLayer should be
            set as the tensors in the current MPSLayer (``True``), or cloned
            (``False``). In the former case, tensors in both MPSLayer's will be
            the same, which might be useful if one needs more than one copy
            of an MPSLayer, but wants to compute all the gradients with respect
            to the same, unique, tensors.
        
        Returns
        -------
        MPSLayer
        """
        new_mps = MPSLayer(n_features=self._n_features,
                           in_dim=self._in_dim,
                           out_dim=self._out_dim,
                           bond_dim=self._bond_dim,
                           out_position=self._out_position,
                           boundary=self._boundary,
                           tensors=None,
                           n_batches=self._n_batches,
                           init_method=None,
                           device=None,
                           dtype=None)
        new_mps.name = self.name + '_copy'

        for i in range(self._n_features):
            new_mps._mats_env[i] = new_mps._mats_env[i].parameterize(
                set_param=isinstance(self._mats_env[i], ParamNode))

        if share_tensors:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor
            if self._boundary == 'obc':
                new_mps._left_node.tensor = self._left_node.tensor
                new_mps.right_node.tensor = self.right_node.tensor
        else:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor.clone()
            if self._boundary == 'obc':
                new_mps._left_node.tensor = self._left_node.tensor.clone()
                new_mps.right_node.tensor = self.right_node.tensor.clone()
        
        return new_mps


class UMPSLayer(MPS):  # MARK: UMPSLayer
    """
    Class for Uniform (translationally invariant) Matrix Product States with an
    output node. It is the uniform version of :class:`MPSLayer`, with all input
    nodes sharing the same tensor, but with a different node for the output node.
    Thus this class cannot have different input or bond dimensions for each node,
    and boundary conditions are always periodic (``"pbc"``).
    
    |
    
    For a more detailed list of inherited properties and methods,
    check :class:`MPS`.

    Parameters
    ----------
    n_features : int
        Number of nodes that will be in ``mats_env``. This also includes the
        output node, so if one wants to instantiate a ``UMPSLayer`` for a
        dataset with ``n`` features, it should be ``n_features = n + 1``, to
        account for the output node.
    in_dim : int, optional
        Input dimension. Equivalent to the physical dimension but only for
        input nodes.
    out_dim : int, optional
        Output dimension (labels) for the output node.
    bond_dim : int, optional
        Bond dimension.
    out_position : int, optional
        Position of the output node (label). Should be between 0 and
        ``n_features - 1``. If ``None``, the output node will be located at the
        middle of the MPS.
    tensors: list[torch.Tensor] or tuple[torch.Tensor], optional
        Instead of providing ``in_dim``, ``out_dim`` and ``bond_dim``, a
        sequence of 2 tensors can be provided, the first one will be the uniform
        tensor, and the second one will be the output node's tensor.
        ``n_features`` is still needed to specify how many times the uniform
        tensor should be used to form a finite MPS. In this case, since the
        output node will have a different tensor, the uniform tensor will be
        used in the remaining ``n_features - 1`` input nodes. Both tensors
        should be rank-3, with all their first and last dimensions being equal.
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
        Initialization method. Check :meth:`initialize` for a more detailed
        explanation of the different initialization methods.
    parameterized : bool, optional
        Boolean indicating whether UMPSLayer nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
    Examples
    --------
    >>> mps_layer = tk.models.UMPSLayer(n_features=4,
    ...                                 in_dim=2,
    ...                                 out_dim=10,
    ...                                 bond_dim=5)
    >>> for i, node in enumerate(mps_layer.mats_env):
    ...     if i != mps_layer.out_position: 
    ...         assert node.tensor_address() == 'virtual_uniform'
    ...
    >>> data = torch.ones(20, 3, 2) # batch_size x (n_features - 1) x feature_size
    >>> result = mps_layer(data)
    >>> result.shape
    torch.Size([20, 10])
    """
        
    def __init__(self,
                 n_features: int,
                 in_dim: Optional[int] = None,
                 out_dim: Optional[int] = None,
                 bond_dim: Optional[int] = None,
                 out_position: Optional[int] = None,
                 tensors: Optional[Sequence[torch.Tensor]] = None,
                 n_batches: int = 1,
                 init_method: Text = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs) -> None:
        
        phys_dim = None
        
        # n_features
        if not isinstance(n_features, int):
            raise TypeError('`n_features` should be int type')
        elif n_features < 1:
            raise ValueError('`n_features` should be at least 1')
            
        # out_position
        if out_position is None:
            out_position = n_features // 2
        if (out_position < 0) or (out_position > n_features):
            raise ValueError(
                f'`out_position` should be between 0 and {n_features}')
        self._out_position = out_position
        
        if tensors is None:
            # in_dim
            if isinstance(in_dim, int):
                in_dim = [in_dim] * (n_features - 1)
            else:
                if n_features == 1:
                    in_dim = []
                else:
                    raise TypeError(
                        '`in_dim` should be int, tuple[int] or list[int] type')

            # out_dim
            if not isinstance(out_dim, int):
                raise TypeError('`out_dim` should be int type')
            
            # phys_dim
            phys_dim = in_dim[:out_position] + [out_dim] + in_dim[out_position:]
        
        else:
            if not isinstance(tensors, Sequence):
                raise TypeError('`tensors` should be a tuple[torch.Tensor] or '
                                'list[torch.Tensor] type')
            if len(tensors) != 2:
                raise ValueError('`tensors` should have 2 elements, the first'
                                 ' corresponding to the common input tensor, '
                                 'and another one for the output node')
            for t in tensors:
                if not isinstance(t, torch.Tensor):
                    raise TypeError(
                        'Elements of `tensors` should be torch.Tensor type')
                if t.ndim != 3:
                    raise ValueError(
                        'Elements of `tensors` should be a rank-3 tensor')
                if t.shape[0] != t.shape[2]:
                    raise ValueError(
                        'Elements of `tensors` should have equal first and last'
                        ' dimensions so that the MPS can have periodic boundary'
                        ' conditions')
            
            if n_features == 1:
                # Only output node is used, uniform memory will
                # take that tensor too
                tensors = [tensors[1]]
            else:
                tensors = [tensors[0]] * out_position + [tensors[1]] + \
                    [tensors[0]] * (n_features - 1 - out_position)
        
        super().__init__(n_features=n_features,
                         phys_dim=phys_dim,
                         bond_dim=bond_dim,
                         boundary='pbc',
                         tensors=tensors,
                         in_features=None,
                         out_features=[out_position],
                         n_batches=n_batches,
                         init_method=init_method,
                         parameterized=parameterized,
                         device=device,
                         dtype=dtype,
                         **kwargs)
        self.name = 'umpslayer'
        self._in_dim = self._phys_dim[:out_position] + \
            self._phys_dim[(out_position + 1):]
        self._out_dim = self._phys_dim[out_position]
    
    @property
    def in_dim(self) -> List[int]:
        """Returns input dimensions."""
        return self._in_dim

    @property
    def out_dim(self) -> int:
        """
        Returns the output dimension, that is, the number of labels in the
        output node. Same as ``in_dim`` for input nodes.
        """
        return self._out_dim

    @property
    def out_position(self) -> int:
        """Returns position of the output node (label)."""
        return self._out_position
    
    @property
    def out_node(self) -> ParamNode:
        """Returns the output node."""
        return self._mats_env[self._out_position]

    def _make_nodes(self, parameterized: bool = True) -> None:
        """Creates all the nodes of the MPS."""
        super()._make_nodes(parameterized)
        
        # Virtual node
        node_cls = ParamNode if parameterized else Node
        uniform_memory = node_cls(shape=(self._bond_dim[0],
                                         self._phys_dim[0],
                                         self._bond_dim[0]),
                                  axes_names=('left', 'input', 'right'),
                                  name='virtual_uniform',
                                  network=self,
                                  virtual=True)
        self.uniform_memory = uniform_memory
        
        in_nodes = self._mats_env[:self._out_position] + \
            self._mats_env[(self._out_position + 1):]
        for node in in_nodes:
            node.set_tensor_from(uniform_memory)
    
    def _make_canonical(self,
                        device: Optional[torch.device] = None,
                        dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
        """
        Creates random unitaries to initialize the MPS in canonical form with
        orthogonality center at the rightmost node. Unitaries in nodes are
        scaled so that the total norm squared of the initial MPS is the product
        of all the physical dimensions.
        """
        # Uniform node
        node = self.uniform_memory
        node_shape = node.shape
        aux_shape = (node.shape[:2].numel(), node.shape[2])
        
        size = max(aux_shape[0], aux_shape[1])
        phys_dim = node_shape[1]
        
        uni_tensor = random_unitary(size, device=device, dtype=dtype)
        uni_tensor = uni_tensor[:min(aux_shape[0], size), :min(aux_shape[1], size)]
        uni_tensor = uni_tensor.reshape(*node_shape)
        uni_tensor = uni_tensor * sqrt(phys_dim)
        
        # Output node
        out_tensor = torch.randn(self.out_node.shape,
                                 device=device,
                                 dtype=dtype)
        out_tensor = out_tensor / out_tensor.norm() * sqrt(out_tensor.shape[1])
        
        return [uni_tensor, out_tensor]
    
    def _make_unitaries(self,
                        device: Optional[torch.device] = None,
                        dtype: Optional[torch.dtype] = None) -> List[torch.Tensor]:
        """
        Creates random unitaries to initialize the MPS nodes as stacks of
        unitaries.
        """
        tensors = []
        for node in [self.uniform_memory, self.out_node]:
            node_shape = node.shape
            
            units = []
            for _ in range(node_shape[1]):
                tensor = random_unitary(node_shape[0],
                                        device=device,
                                        dtype=dtype)
                units.append(tensor)
            
            tensors.append(torch.stack(units, dim=1))
        
        return tensors

    def initialize(self,
                   tensors: Optional[Sequence[torch.Tensor]] = None,
                   init_method: Optional[Text] = 'randn',
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None,
                   **kwargs: float) -> None:
        """
        Initializes the common tensor of the :class:`UMPSLayer`. It can be called
        when instantiating the model, or to override the existing nodes' tensors.
        
        There are different methods to initialize the nodes:
        
        * ``{"zeros", "ones", "copy", "rand", "randn"}``: The tensor is
          initialized calling :meth:`~tensorkrowch.AbstractNode.set_tensor` with
          the given method, ``device``, ``dtype`` and ``kwargs``.
        
        * ``"randn_eye"``: Tensor is initialized as in this
          `paper <https://arxiv.org/abs/1605.03795>`_, adding identities at the
          top of a random gaussian tensor. In this case, ``std`` should be
          specified with a low value, e.g., ``std = 1e-9``.
          
        * ``"unit"``: Tensor is initialized as a stack of random unitaries. This,
          combined (at least) with an embedding of the inputs as elements of
          the computational basis (:func:`~tensorkrowch.embeddings.discretize`
          combined with :func:`~tensorkrowch.embeddings.basis`)
        
        * ``"canonical"```: MPS is initialized in canonical form with a squared
          norm `close` to the product of all the physical dimensions (if bond
          dimensions are bigger than the powers of the physical dimensions,
          the norm could vary).
        
        Parameters
        ----------
        tensors : list[torch.Tensor] or tuple[torch.Tensor], optional
            Sequence of a 2 tensors, the first one will be the uniform tensor
            that will be set in all input nodes, and the second one will be the
            output node's tensor. Both tensors should be rank-3, with all their
            first and last dimensions being equal.
        init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensors if ``init_method`` is provided.
        dtype : torch.dtype, optional
            Dtype of the tensor if ``init_method`` is provided.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        """
        if init_method == 'unit':
            tensors = self._make_unitaries(device=device, dtype=dtype)
        elif init_method == 'canonical':
            tensors = self._make_canonical(device=device, dtype=dtype)
        
        if tensors is not None:
            self.uniform_memory.tensor = tensors[0]
            self.out_node.tensor = tensors[-1]
        
        elif init_method is not None:
            for i, node in enumerate([self.uniform_memory, self.out_node]):
                add_eye = False
                if init_method == 'randn_eye':
                    init_method = 'randn'
                    add_eye = True
                
                node.set_tensor(init_method=init_method,
                                device=device,
                                dtype=dtype,
                                **kwargs)
                if add_eye:
                    aux_tensor = node.tensor.detach()
                    eye_tensor = torch.eye(node.shape[0],
                                           node.shape[2],
                                           device=device,
                                           dtype=dtype)
                    if i == 0:
                        aux_tensor[:, 0, :] += eye_tensor
                    else:
                        eye_tensor = eye_tensor.unsqueeze(1)
                        eye_tensor = eye_tensor.expand(node.shape)
                        aux_tensor += eye_tensor
                    node.tensor = aux_tensor
    
    def copy(self, share_tensors: bool = False) -> 'UMPSLayer':
        """
        Creates a copy of the :class:`UMPSLayer`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether tensors in the copied UMPSLayer should be
            set as the tensors in the current UMPSLayer (``True``), or cloned
            (``False``). In the former case, tensors in both UMPSLayer's will be
            the same, which might be useful if one needs more than one copy
            of an UMPSLayer, but wants to compute all the gradients with respect
            to the same, unique, tensors.
        
        Returns
        -------
        UMPSLayer
        """
        new_mps = UMPSLayer(n_features=self._n_features,
                            in_dim=self._in_dim[0] if self._in_dim else None,
                            out_dim=self._out_dim,
                            bond_dim=self._bond_dim,
                            out_position=self._out_position,
                            tensor=None,
                            n_batches=self._n_batches,
                            init_method=None,
                            parameterized=isinstance(self.uniform_memory, ParamNode),
                            device=None,
                            dtype=None)
        new_mps.name = self.name + '_copy'
        
        new_mps._mats_env[self._out_position] = \
            new_mps._mats_env[self._out_position].parameterize(
                set_param=isinstance(self.out_node, ParamNode))
        
        if share_tensors:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor
            new_mps.out_node.tensor = self.out_node.tensor
        else:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor.clone()
            new_mps.out_node.tensor = self.out_node.tensor.clone()
        return new_mps
    
    def parameterize(self,
                     set_param: bool = True,
                     override: bool = False) -> 'TensorNetwork':
        """
        Parameterizes all nodes of the MPS. If there are ``resultant`` nodes
        in the MPS, it will be first :meth:`~tensorkrowch.TensorNetwork.reset`.

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
            net = self.copy(share_tensors=False)
        
        for i in range(self._n_features):
            net._mats_env[i] = net._mats_env[i].parameterize(set_param=set_param)
        
        # It is important that uniform_memory is parameterized after the rest
        # of the nodes
        net.uniform_memory = net.uniform_memory.parameterize(set_param=set_param)
        
        # Tensor addresses have to be reassigned to reference
        # the uniform memory
        for node in net._mats_env:
            node.set_tensor_from(net.uniform_memory)
        
        return net
    
    def canonicalize(self,
                     oc: Optional[int] = None,
                     mode: Text = 'svd',
                     rank: Optional[int] = None,
                     cutoff: Optional[float] = None,
                     atol: Optional[float] = None,
                     rtol: Optional[float] = None,
                     cum_percentage: Optional[float] = None,
                     renormalize: bool = False) -> None:
        """:meta private:"""
        raise NotImplementedError(
            '`canonicalize` not implemented for UMPSLayer')
    
    def canonicalize_univocal(self):
        """:meta private:"""
        raise NotImplementedError(
            '`canonicalize_univocal` not implemented for UMPSLayer')


###############################################################################
#                                 CONV MODELS                                 #
###############################################################################
class AbstractConvClass(ABC):  # MARK: AbstractConvClass
    
    @abstractmethod
    def __init__(self):
        pass
    
    def _set_attributes(self,
                        in_channels: int,
                        kernel_size: Union[int, Sequence[int]],
                        stride: int,
                        padding: int,
                        dilation: int) -> nn.Module:

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size, Sequence):
            raise TypeError('`kernel_size` must be int, list[int] or tuple[int]')

        if isinstance(stride, int):
            stride = (stride, stride)
        elif not isinstance(stride, Sequence):
            raise TypeError('`stride` must be int, list[int] or tuple[int]')

        if isinstance(padding, int):
            padding = (padding, padding)
        elif not isinstance(padding, Sequence):
            raise TypeError('`padding` must be int, list[int] or tuple[int]')

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        elif not isinstance(dilation, Sequence):
            raise TypeError('`dilation` must be int , list[int] or tuple[int]')

        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        
        unfold = nn.Unfold(kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           dilation=dilation)
        return unfold

    def forward(self, image, mode='flat', *args, **kwargs):
        r"""
        Overrides :meth:`~tensorkrowch.TensorNetwork.forward` to compute a
        convolution on the input image.
        
        Parameters
        ----------
        image : torch.Tensor
            Input batch of images with shape
            
            .. math::
            
                batch\_size \times in\_channels \times height \times width
        mode : {"flat", "snake"}
            Indicates the order in which MPS should take the pixels in the image.
            When ``"flat"``, the image is flattened putting one row of the image
            after the other. When ``"snake"``, its row is put in the opposite
            orientation as the previous row (like a snake running through the
            image).
        args :
            Arguments that might be used in :meth:`~MPS.contract`.
        kwargs :
            Keyword arguments that might be used in :meth:`~MPS.contract`,
            like ``inline_input`` or ``inline_mats``.
        """
        # Input image shape: batch_size x in_channels x height x width

        patches = self.unfold(image).transpose(1, 2)
        # batch_size x nb_windows x (in_channels * nb_pixels)

        patches = patches.view(*patches.shape[:-1], self.in_channels, -1)
        # batch_size x nb_windows x in_channels x nb_pixels

        patches = patches.transpose(2, 3)
        # batch_size x nb_windows x nb_pixels x in_channels

        if mode == 'snake':
            new_patches = patches[..., :self._kernel_size[1], :]
            for i in range(1, self._kernel_size[0]):
                if i % 2 == 0:
                    aux = patches[..., (i * self._kernel_size[1]):
                                    ((i + 1) * self._kernel_size[1]), :]
                else:
                    aux = patches[...,
                        (i * self._kernel_size[1]):
                        ((i + 1) * self._kernel_size[1]), :].flip(dims=[0])
                new_patches = torch.cat([new_patches, aux], dim=2)

            patches = new_patches

        elif mode != 'flat':
            raise ValueError('`mode` can only be "flat" or "snake"')
        
        h_in = image.shape[2]
        w_in = image.shape[3]

        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] *
                    (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] *
                    (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows (x out_channels ...)
        
        if result.ndim == 3:
            result = result.movedim(1, -1)
            # batch_size (x out_channels ...) x nb_windows

        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size (x out_channels ...) x height_out x width_out

        return result


class ConvMPS(AbstractConvClass, MPS):  # MARK: ConvMPS
    r"""
    Convolutional version of :class:`MPS`, where the input data is assumed to
    be a batch of images.
    
    Input data as well as initialization parameters are described in `torch.nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    in_channels : int
        Input channels. Same as ``phys_dim`` in :class:`MPS`.
    bond_dim : int, list[int] or tuple[int]
        Bond dimension(s). If given as a sequence, its length should be equal
        to :math:`kernel\_size_0 \cdot kernel\_size_1` (if ``boundary = "pbc"``)
        or :math:`kernel\_size_0 \cdot kernel\_size_1 - 1` (if
        ``boundary = "obc"``). The i-th bond dimension is always the dimension
        of the right edge of the i-th node.
    kernel_size : int, list[int] or tuple[int]
        Kernel size used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    stride : int
        Stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
    padding : int
        Padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    dilation : int
        Dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    boundary : {"obc", "pbc"}
        String indicating whether periodic or open boundary conditions should
        be used.
    tensors: list[torch.Tensor] or tuple[torch.Tensor], optional
        To initialize MPS nodes, a list of MPS tensors can be provided. All
        tensors should be rank-3 tensors, with shape ``(bond_dim, in_channels,
        bond_dim)``. If the first and last elements are rank-2 tensors, with
        shapes ``(in_channels, bond_dim)``, ``(bond_dim, in_channels)``,
        respectively, the inferred boundary conditions will be "obc". Also, if
        ``tensors`` contains a single element, it can be rank-1 ("obc") or
        rank-3 ("pbc").
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
        Initialization method. Check :meth:`~MPS.initialize` for a more detailed
        explanation of the different initialization methods.
    parameterized : bool, optional
        Boolean indicating whether ConvMPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
    Examples
    --------
    >>> conv_mps = tk.models.ConvMPS(in_channels=2,
    ...                              bond_dim=5,
    ...                              kernel_size=2)
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_mps(data)
    >>> result.shape
    torch.Size([20, 1, 1])
    """

    def __init__(self,
                 in_channels: int,
                 bond_dim: Union[int, Sequence[int]],
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 boundary: Text = 'obc',
                 tensors: Optional[Sequence[torch.Tensor]] = None,
                 init_method: Text = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs):
        
        unfold = self._set_attributes(in_channels=in_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation)

        MPS.__init__(self,
                     n_features=self._kernel_size[0] * self._kernel_size[1],
                     phys_dim=in_channels,
                     bond_dim=bond_dim,
                     boundary=boundary,
                     tensors=tensors,
                     n_batches=2,
                     init_method=init_method,
                     parameterized=parameterized,
                     device=device,
                     dtype=dtype,
                     **kwargs)
        
        self.unfold = unfold
    
    @property
    def in_channels(self) -> int:
        """Returns ``in_channels``. Same as ``phys_dim`` in :class:`MPS`."""
        return self._in_channels

    @property
    def kernel_size(self) -> Tuple[int, int]:
        r"""
        Returns ``kernel_size``. Number of nodes is given by
        :math:`kernel\_size_0 \cdot kernel\_size_1`.
        """
        return self._kernel_size

    @property
    def stride(self) -> Tuple[int, int]:
        """
        Returns stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._stride

    @property
    def padding(self) -> Tuple[int, int]:
        """
        Returns padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._padding

    @property
    def dilation(self) -> Tuple[int, int]:
        """
        Returns dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._dilation
    
    def copy(self, share_tensors: bool = False) -> 'ConvMPS':
        """
        Creates a copy of the :class:`ConvMPS`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether tensors in the copied ConvMPS should be
            set as the tensors in the current ConvMPS (``True``), or cloned
            (``False``). In the former case, tensors in both ConvMPS's will be
            the same, which might be useful if one needs more than one copy
            of a ConvMPS, but wants to compute all the gradients with respect
            to the same, unique, tensors.

        Returns
        -------
        ConvMPS
        """
        new_mps = ConvMPS(in_channels=self._in_channels,
                          bond_dim=self._bond_dim,
                          kernel_size=self._kernel_size,
                          stride=self._stride,
                          padding=self._padding,
                          dilation=self.dilation,
                          boundary=self._boundary,
                          tensors=None,
                          init_method=None,
                          device=None,
                          dtype=None)
        new_mps.name = self.name + '_copy'
        
        for i in range(self._n_features):
            new_mps._mats_env[i] = new_mps._mats_env[i].parameterize(
                set_param=isinstance(self._mats_env[i], ParamNode))
        
        if share_tensors:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor
            if self._boundary == 'obc':
                new_mps._left_node.tensor = self._left_node.tensor
                new_mps.right_node.tensor = self.right_node.tensor
        else:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor.clone()
            if self._boundary == 'obc':
                new_mps._left_node.tensor = self._left_node.tensor.clone()
                new_mps.right_node.tensor = self.right_node.tensor.clone()
        
        return new_mps


class ConvUMPS(AbstractConvClass, UMPS):  # MARK: ConvUMPS
    """
    Convolutional version of :class:`UMPS`, where the input data is assumed to
    be a batch of images.
    
    Input data as well as initialization parameters are described in `torch.nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    in_channels : int
        Input channels. Same as ``phys_dim`` in :class:`UMPS`.
    bond_dim : int
        Bond dimension.
    kernel_size : int, list[int] or tuple[int]
        Kernel size used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    stride : int
        Stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
    padding : int
        Padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    dilation : int
        Dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    tensor: torch.Tensor, optional
        To initialize MPS nodes, a MPS tensor can be provided. The tensor
        should be rank-3, with its first and last dimensions being equal.
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
        Initialization method. Check :meth:`~UMPS.initialize` for a more detailed
        explanation of the different initialization methods.
    parameterized : bool, optional
        Boolean indicating whether ConvUMPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
    
    
    Examples
    --------
    >>> conv_mps = tk.models.ConvUMPS(in_channels=2,
    ...                               bond_dim=5,
    ...                               kernel_size=2)
    >>> for node in conv_mps.mats_env:
    ...     assert node.tensor_address() == 'virtual_uniform'
    ...
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_mps(data)
    >>> result.shape
    torch.Size([20, 1, 1])
    """

    def __init__(self,
                 in_channels: int,
                 bond_dim: int,
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 boundary: Text = 'pbc',
                 tensor: Optional[torch.Tensor] = None,
                 init_method: Text = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs):

        unfold = self._set_attributes(in_channels=in_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation)

        UMPS.__init__(self,
                      n_features=self._kernel_size[0] * self._kernel_size[1],
                      phys_dim=in_channels,
                      bond_dim=bond_dim,
                      boundary=boundary,
                      tensor=tensor,
                      n_batches=2,
                      init_method=init_method,
                      parameterized=parameterized,
                      device=device,
                      dtype=dtype,
                      **kwargs)
        
        self.unfold = unfold
    
    @property
    def in_channels(self) -> int:
        """Returns ``in_channels``. Same as ``phys_dim`` in :class:`MPS`."""
        return self._in_channels

    @property
    def kernel_size(self) -> Tuple[int, int]:
        r"""
        Returns ``kernel_size``. Number of nodes is given by
        :math:`kernel\_size_0 \cdot kernel\_size_1`.
        """
        return self._kernel_size

    @property
    def stride(self) -> Tuple[int, int]:
        """
        Returns stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._stride

    @property
    def padding(self) -> Tuple[int, int]:
        """
        Returns padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._padding

    @property
    def dilation(self) -> Tuple[int, int]:
        """
        Returns dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._dilation
    
    def copy(self, share_tensors: bool = False) -> 'ConvUMPS':
        """
        Creates a copy of the :class:`ConvUMPS`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether the common tensor in the copied ConvUMPS
            should be set as the tensor in the current ConvUMPS (``True``), or
            cloned (``False``). In the former case, the tensor in both ConvUMPS's
            will be the same, which might be useful if one needs more than one
            copy of a ConvUMPS, but wants to compute all the gradients with respect
            to the same, unique, tensor.

        Returns
        -------
        ConvUMPS
        """
        new_mps = ConvUMPS(in_channels=self._in_channels,
                           bond_dim=self._bond_dim[0] if self._bond_dim else 1,
                           kernel_size=self._kernel_size,
                           stride=self._stride,
                           padding=self._padding,
                           dilation=self.dilation,
                           boundary=self._boundary,
                           tensor=None,
                           init_method=None,
                           parameterized=isinstance(self.uniform_memory, ParamNode),
                           device=None,
                           dtype=None)
        new_mps.name = self.name + '_copy'
        if share_tensors:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor
        else:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor.clone()
        
        return new_mps


class ConvMPSLayer(AbstractConvClass, MPSLayer):  # MARK: ConvMPSLayer
    r"""
    Convolutional version of :class:`MPSLayer`, where the input data is assumed to
    be a batch of images.
    
    Input data as well as initialization parameters are described in `torch.nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    in_channels : int
        Input channels. Same as ``in_dim`` in :class:`MPSLayer`.
    out_channels : int
        Output channels. Same as ``out_dim`` in :class:`MPSLayer`.
    bond_dim : int, list[int] or tuple[int]
        Bond dimension(s). If given as a sequence, its length should be equal
        to :math:`kernel\_size_0 \cdot kernel\_size_1 + 1`
        (if ``boundary = "pbc"``) or :math:`kernel\_size_0 \cdot kernel\_size_1`
        (if ``boundary = "obc"``). The i-th bond dimension is always the dimension
        of the right edge of the i-th node (including output node).
    kernel_size : int, list[int] or tuple[int]
        Kernel size used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    stride : int
        Stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
    padding : int
        Padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    dilation : int
        Dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    out_position : int, optional
        Position of the output node (label). Should be between 0 and
        :math:`kernel\_size_0 \cdot kernel\_size_1`. If ``None``, the output node
        will be located at the middle of the MPS.
    boundary : {"obc", "pbc"}
        String indicating whether periodic or open boundary conditions should
        be used.
    tensors: list[torch.Tensor] or tuple[torch.Tensor], optional
        To initialize MPS nodes, a list of MPS tensors can be provided. All
        tensors should be rank-3 tensors, with shape ``(bond_dim, in_channels,
        bond_dim)``. If the first and last elements are rank-2 tensors, with
        shapes ``(in_channels, bond_dim)``, ``(bond_dim, in_channels)``,
        respectively, the inferred boundary conditions will be "obc". Also, if
        ``tensors`` contains a single element, it can be rank-1 ("obc") or
        rank-3 ("pbc").
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
        Initialization method. Check :meth:`~MPSLayer.initialize` for a more detailed
        explanation of the different initialization methods.
    parameterized : bool, optional
        Boolean indicating whether ConvMPSLayer nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
    Examples
    --------
    >>> conv_mps_layer = tk.models.ConvMPSLayer(in_channels=2,
    ...                                         out_channels=10,
    ...                                         bond_dim=5,
    ...                                         kernel_size=2)
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_mps_layer(data)
    >>> result.shape
    torch.Size([20, 10, 1, 1])
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bond_dim: Union[int, Sequence[int]],
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 out_position: Optional[int] = None,
                 boundary: Text = 'obc',
                 tensors: Optional[Sequence[torch.Tensor]] = None,
                 init_method: Text = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs):

        unfold = self._set_attributes(in_channels=in_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation)

        MPSLayer.__init__(self,
                          n_features=self._kernel_size[0] * \
                              self._kernel_size[1] + 1,
                          in_dim=in_channels,
                          out_dim=out_channels,
                          bond_dim=bond_dim,
                          out_position=out_position,
                          boundary=boundary,
                          tensors=tensors,
                          n_batches=2,
                          init_method=init_method,
                          parameterized=parameterized,
                          device=device,
                          dtype=dtype,
                          **kwargs)
        
        self._out_channels = out_channels
        self.unfold = unfold
    
    @property
    def in_channels(self) -> int:
        """Returns ``in_channels``. Same as ``in_dim`` in :class:`MPSLayer`."""
        return self._in_channels

    @property
    def out_channels(self) -> int:
        """Returns ``out_channels``. Same as ``out_dim`` in :class:`MPSLayer`."""
        return self._out_channels

    @property
    def kernel_size(self) -> Tuple[int, int]:
        r"""
        Returns ``kernel_size``. Number of nodes is given by
        :math:`kernel\_size_0 \cdot kernel\_size_1 + 1`.
        """
        return self._kernel_size

    @property
    def stride(self) -> Tuple[int, int]:
        """
        Returns stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._stride

    @property
    def padding(self) -> Tuple[int, int]:
        """
        Returns padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._padding

    @property
    def dilation(self) -> Tuple[int, int]:
        """
        Returns dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._dilation
    
    def copy(self, share_tensors: bool = False) -> 'ConvMPSLayer':
        """
        Creates a copy of the :class:`ConvMPSLayer`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether tensors in the copied ConvMPSLayer should
            be set as the tensors in the current ConvMPSLayer (``True``), or
            cloned (``False``). In the former case, tensors in both ConvMPSLayer's
            will be the same, which might be useful if one needs more than one
            copy of an ConvMPSLayer, but wants to compute all the gradients with
            respect to the same, unique, tensors.
        
        Returns
        -------
        ConvMPSLayer
        """
        new_mps = ConvMPSLayer(in_channels=self._in_channels,
                               out_channels=self._out_channels,
                               bond_dim=self._bond_dim,
                               kernel_size=self._kernel_size,
                               stride=self._stride,
                               padding=self._padding,
                               dilation=self.dilation,
                               boundary=self._boundary,
                               tensors=None,
                               init_method=None,
                               device=None,
                               dtype=None)
        new_mps.name = self.name + '_copy'

        for i in range(self._n_features):
            new_mps._mats_env[i] = new_mps._mats_env[i].parameterize(
                set_param=isinstance(self._mats_env[i], ParamNode))
        
        if share_tensors:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor
            if self._boundary == 'obc':
                new_mps._left_node.tensor = self._left_node.tensor
                new_mps.right_node.tensor = self.right_node.tensor
        else:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor.clone()
            if self._boundary == 'obc':
                new_mps._left_node.tensor = self._left_node.tensor.clone()
                new_mps.right_node.tensor = self.right_node.tensor.clone()
        
        return new_mps


class ConvUMPSLayer(AbstractConvClass, UMPSLayer):  # MARK: ConvUMPSLayer
    r"""
    Convolutional version of :class:`UMPSLayer`, where the input data is assumed to
    be a batch of images.
    
    Input data as well as initialization parameters are described in `torch.nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    in_channels : int
        Input channels. Same as ``in_dim`` in :class:`UMPSLayer`.
    out_channels : int
        Output channels. Same as ``out_dim`` in :class:`UMPSLayer`.
    bond_dim : int
        Bond dimension.
    kernel_size : int, list[int] or tuple[int]
        Kernel size used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    stride : int
        Stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
    padding : int
        Padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    dilation : int
        Dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    out_position : int, optional
        Position of the output node (label). Should be between 0 and
        :math:`kernel\_size_0 \cdot kernel\_size_1`. If ``None``, the output node
        will be located at the middle of the MPS.
        
    tensors: list[torch.Tensor] or tuple[torch.Tensor], optional
        To initialize MPS nodes, a sequence of 2 tensors can be provided, the
        first one will be the uniform tensor, and the second one will be the
        output node's tensor. Both tensors should be rank-3, with all their
        first and last dimensions being equal.
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit", "canonical"}, optional
        Initialization method. Check :meth:`~UMPSLayer.initialize` for a more
        detailed explanation of the different initialization methods.
    parameterized : bool, optional
        Boolean indicating whether ConvUMPSLayer nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
    Examples
    --------
    >>> conv_mps_layer = tk.models.ConvUMPSLayer(in_channels=2,
    ...                                          out_channels=10,
    ...                                          bond_dim=5,
    ...                                          kernel_size=2)
    >>> for i, node in enumerate(conv_mps_layer.mats_env):
    ...     if i != conv_mps_layer.out_position:
    ...         assert node.tensor_address() == 'virtual_uniform'
    ...
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_mps_layer(data)
    >>> result.shape
    torch.Size([20, 10, 1, 1])
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bond_dim: Union[int, Sequence[int]],
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 out_position: Optional[int] = None,
                 tensors: Optional[Sequence[torch.Tensor]] = None,
                 init_method: Text = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs):

        unfold = self._set_attributes(in_channels=in_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation)

        UMPSLayer.__init__(self,
                           n_features=self._kernel_size[0] * \
                               self._kernel_size[1] + 1,
                           in_dim=in_channels,
                           out_dim=out_channels,
                           bond_dim=bond_dim,
                           out_position=out_position,
                           tensors=tensors,
                           n_batches=2,
                           init_method=init_method,
                           parameterized=parameterized,
                           device=device,
                           dtype=dtype,
                           **kwargs)
        
        self._out_channels = out_channels
        self.unfold = unfold
    
    @property
    def in_channels(self) -> int:
        """Returns ``in_channels``. Same as ``in_dim`` in :class:`UMPSLayer`."""
        return self._in_channels

    @property
    def out_channels(self) -> int:
        """Returns ``out_channels``. Same as ``out_dim`` in :class:`UMPSLayer`."""
        return self._out_channels

    @property
    def kernel_size(self) -> Tuple[int, int]:
        r"""
        Returns ``kernel_size``. Number of nodes is given by
        :math:`kernel\_size_0 \cdot kernel\_size_1 + 1`.
        """
        return self._kernel_size

    @property
    def stride(self) -> Tuple[int, int]:
        """
        Returns stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._stride

    @property
    def padding(self) -> Tuple[int, int]:
        """
        Returns padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._padding

    @property
    def dilation(self) -> Tuple[int, int]:
        """
        Returns dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._dilation
    
    @property
    def out_channels(self) -> int:
        """Returns ``out_channels``. Same as ``phys_dim`` in :class:`MPS`."""
        return self._out_channels

    def copy(self, share_tensors: bool = False) -> 'ConvUMPSLayer':
        """
        Creates a copy of the :class:`ConvUMPSLayer`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether tensors in the copied ConvUMPSLayer should
            be set as the tensors in the current ConvUMPSLayer (``True``), or
            cloned (``False``). In the former case, tensors in both ConvUMPSLayer's
            will be the same, which might be useful if one needs more than one
            copy of an ConvUMPSLayer, but wants to compute all the gradients with
            respect to the same, unique, tensors.
        
        Returns
        -------
        ConvUMPSLayer
        """
        new_mps = ConvUMPSLayer(in_channels=self._in_channels,
                                out_channels=self._out_channels,
                                bond_dim=self._bond_dim[0],
                                kernel_size=self._kernel_size,
                                stride=self._stride,
                                padding=self._padding,
                                dilation=self.dilation,
                                tensor=None,
                                init_method=None,
                                parameterized=isinstance(self.uniform_memory, ParamNode),
                                device=None,
                                dtype=None)
        new_mps.name = self.name + '_copy'
        
        new_mps._mats_env[self._out_position] = \
            new_mps._mats_env[self._out_position].parameterize(
                set_param=isinstance(self.out_node, ParamNode))
        
        if share_tensors:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor
            new_mps.out_node.tensor = self.out_node.tensor
        else:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor.clone()
            new_mps.out_node.tensor = self.out_node.tensor.clone()
        
        return new_mps
