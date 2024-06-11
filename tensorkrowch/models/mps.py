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
from typing import (List, Optional, Sequence,
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
      in the case ``boudary`` is ``"obc"``. Otherwise, both are ``None``.
    
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
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit"}, optional
        Initialization method. Check :meth:`initialize` for a more detailed
        explanation of the different initialization methods.
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
            if not isinstance(tensors, Sequence):
                raise TypeError('`tensors` should be a tuple[torch.Tensor] or '
                                'list[torch.Tensor] type')
            else:
                self._n_features = len(tensors)
                self._phys_dim = []
                self._bond_dim = []
                for i, t in enumerate(tensors):
                    if not isinstance(t, torch.Tensor):
                        raise TypeError('`tensors` should be a tuple[torch.Tensor]'
                                        ' or list[torch.Tensor] type')
                    
                    if i == 0:
                        if len(t.shape) not in [1, 2, 3]:
                            raise ValueError(
                                'The first and last elements in `tensors` '
                                'should be both rank-2 or rank-3 tensors. If'
                                ' the first element is also the last one,'
                                ' it should be a rank-1 tensor')
                        if len(t.shape) == 1:
                            self._boundary = 'obc'
                            self._phys_dim.append(t.shape[0])
                        elif len(t.shape) == 2:
                            self._boundary = 'obc'
                            self._phys_dim.append(t.shape[0])
                            self._bond_dim.append(t.shape[1])
                        else:
                            self._boundary = 'pbc'
                            self._phys_dim.append(t.shape[1])
                            self._bond_dim.append(t.shape[2])
                    elif i == (self._n_features - 1):
                        if len(t.shape) != len(tensors[0].shape):
                            raise ValueError(
                                'The first and last elements in `tensors` '
                                'should have the same rank. Both should be '
                                'rank-2 or rank-3 tensors. If the first '
                                'element is also the last one, it should '
                                'be a rank-1 tensor')
                        if len(t.shape) == 2:
                            self._phys_dim.append(t.shape[1])
                        else:
                            if t.shape[-1] != tensors[0].shape[0]:
                                raise ValueError(
                                    'If the first and last elements in `tensors`'
                                    ' are rank-3 tensors, the first dimension '
                                    'of the first element should coincide with'
                                    ' the last dimension of the last element')
                            self._phys_dim.append(t.shape[1])
                            self._bond_dim.append(t.shape[2])
                    else:
                        if len(t.shape) != 3:
                            raise ValueError(
                                'The elements of `tensors` should be rank-3 '
                                'tensors, except the first and lest elements'
                                ' if boundary is "obc"')
                        self._phys_dim.append(t.shape[1])
                        self._bond_dim.append(t.shape[2])
        
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
        
        # Properties
        self._left_node = None
        self._right_node = None
        self._mats_env = []

        # Create Tensor Network
        self._make_nodes()
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
                                          self.left_node.tensor,
                                          mps_tensors[0])
            mps_tensors[-1] = torch.einsum('lir,r->li',
                                           mps_tensors[-1],
                                           self.right_node.tensor)
        return mps_tensors
    
    # -------
    # Methods
    # -------
    def _make_nodes(self) -> None:
        """Creates all the nodes of the MPS."""
        if self._leaf_nodes:
            raise ValueError('Cannot create MPS nodes if the MPS already has '
                             'nodes')
        
        aux_bond_dim = self._bond_dim
        
        if self._boundary == 'obc':
            if not aux_bond_dim:
                aux_bond_dim = [1]
                
            self._left_node = ParamNode(shape=(aux_bond_dim[0],),
                                        axes_names=('right',),
                                        name='left_node',
                                        network=self)
            self._right_node = ParamNode(shape=(aux_bond_dim[-1],),
                                         axes_names=('left',),
                                         name='right_node',
                                         network=self)
            
            aux_bond_dim = aux_bond_dim + [aux_bond_dim[-1]] + [aux_bond_dim[0]]
        
        for i in range(self._n_features):
            node = ParamNode(shape=(aux_bond_dim[i - 1],
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
        if share_tensors:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor
        else:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor.clone()
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
            net._mats_env[i] = net._mats_env[i].parameterize(set_param)
        
        if net._boundary == 'obc':
            net._left_node = net._left_node.parameterize(set_param)
            net._right_node = net._right_node.parameterize(set_param)
            
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
                if len(mat.shape) != 2:
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
        
        # NOTE: to leave the input edges open and marginalize output
        # data_nodes = []
        # for node in self.in_env:
        #     data_node = node.neighbours('input')
        #     if data_node:
        #         data_nodes.append(data_node)
        
        # if data_nodes:
        #     mats_in_env = self._input_contraction(
        #         nodes_env=self.in_env,
        #         input_nodes=data_nodes,
        #         inline_input=inline_input)
        # else:
        #     mats_in_env = self.in_env
        
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
            # Contract each in_result with the next output node
            nodes_out_env = []
            out_first = out_regions[0][0] == 0
            out_last = out_regions[-1][-1] == (self._n_features - 1)
                
            for i in range(len(out_regions)):
                aux_out_env = [self._mats_env[j] for j in out_regions[i]]
                
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
                    # original batches, which gives dupliicate output batches
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
                    
                    # Contract output nodes with matrices
                    nodes_out_env = self._input_contraction(
                        nodes_env=nodes_out_env,
                        input_nodes=mats_nodes,
                        inline_input=True)
                
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

                    # Contract MPO with MPS
                    nodes_out_env = self._input_contraction(
                        nodes_env=nodes_out_env,
                        input_nodes=mpo._mats_env,
                        inline_input=True)
                    
                    # Contract MPO left and right nodes
                    if mpo._boundary == 'obc':
                        nodes_out_env[0] = mpo._left_node @ nodes_out_env[0]
                        nodes_out_env[-1] = nodes_out_env[-1] @ mpo._right_node
                
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
                
                # Contract output nodes with copies
                mats_out_env = self._input_contraction(
                    nodes_env=nodes_out_env,
                    input_nodes=copied_nodes,
                    inline_input=True)
                
                # Contract resultant matrices
                result = self._inline_contraction(mats_env=mats_out_env,
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
            result = op.permute(result, tuple(all_edges))
        
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
                # original batches, which gives dupliicate output batches
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
            
        # Contract output nodes with copies
        mats_out_env = self._input_contraction(
            nodes_env=all_nodes,
            input_nodes=copied_nodes,
            inline_input=True)
        
        # Contract resultant matrices
        log_norm = 0
        result_node = mats_out_env[0]
        if log_scale:
            log_norm += result_node.norm().log()
            result_node = result_node.renormalize()
                
        for node in mats_out_env[1:]:
            result_node @= node
            
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
        Computes the reduced von Neumann Entropy between subsystems :math:`A`
        and :math:`B`, :math:`S(\rho_A)`, where :math:`A` goes from site
        0 to ``middle_site``, and :math:`B` goes from ``middle_site + 1`` to
        ``n_features - 1``.
        
        To compute the reduced entropy, the MPS is put into canonical form
        with orthogonality center at ``middle_site``. Bond dimensions are not
        changed if possible. Only when the bond dimension is bigger than the
        physical dimension multiplied by the other bond dimension of the node,
        it will be cropped to that size.
        
        If the MPS is not normalized, it may happen that the computation of the
        reduced entropy fails due to errors in the Singular Value
        Decompositions. To avoid this, it is recommended to set
        ``renormalize = True``. In this case, the norm of each node after the
        SVD is extracted in logarithmic form, and accumulated. As a result,
        the function will return the tuple ``(entropy, log_norm)``, which is a
        sort of `scaled` reduced entropy. This is, indeed, the reduced entropy
        of a distribution, since the schmidt values are normalized to sum up
        to 1.
        
        The actual reduced entropy, without rescaling, could be obtained as:
        
        .. math::
        
            \exp(\texttt{log_norm})^2 \cdot S(\rho_A) - 
            \exp(\texttt{log_norm})^2 \cdot 2 \cdot \texttt{log_norm}
        
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
        
        for i in range(middle_site):
            result1, result2 = nodes[i]['right'].svd_(
                side='right',
                rank=nodes[i]['right'].size())
            
            if renormalize:
                aux_norm = result2.norm()
                if not aux_norm.isinf() and (aux_norm > 0):
                    result2.tensor = result2.tensor / aux_norm
                    log_norm += aux_norm.log()

            result1 = result1.parameterize()
            nodes[i] = result1
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

            result2 = result2.parameterize()
            nodes[i] = result2
            nodes[i - 1] = result1
        
        nodes[middle_site] = nodes[middle_site].parameterize()
        
        # Compute mutual information
        middle_tensor = nodes[middle_site].tensor.clone()
        _, s, _ = torch.linalg.svd(
            middle_tensor.reshape(middle_tensor.shape[:-1].numel(), # left x input
                                  middle_tensor.shape[-1]),         # right
            full_matrices=False)
        
        s = s[s.pow(2) > 0]
        entropy = -(s.pow(2) * s.pow(2).log()).sum()
        
        # Rescale
        if log_norm != 0:
            rescale = (log_norm / len(nodes)).exp()
        
        if renormalize and (log_norm != 0):
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
    def canonicalize(self,
                     oc: Optional[int] = None,
                     mode: Text = 'svd',
                     rank: Optional[int] = None,
                     cum_percentage: Optional[float] = None,
                     cutoff: Optional[float] = None,
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
        for the possibly new bond dimensions given by the arguments
        ``cum_percentage`` and/or ``cutoff``.
        
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
        cum_percentage : float, optional
            Proportion that should be satisfied between the sum of all singular
            values kept and the total sum of all singular values.
            
            .. math::
            
                \frac{\sum_{i \in \{kept\}}{s_i}}{\sum_{i \in \{all\}}{s_i}} \ge
                cum\_percentage
        cutoff : float, optional
            Quantity that lower bounds singular values in order to be kept.
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
        
        # If mode is svd or svr and none of the args is provided, the ranks are
        # kept as they were originally
        keep_rank = False
        if rank is None:
            keep_rank = True
        
        for i in range(oc):
            if mode == 'svd':
                result1, result2 = nodes[i]['right'].svd_(
                    side='right',
                    rank=nodes[i]['right'].size() if keep_rank else rank,
                    cum_percentage=cum_percentage,
                    cutoff=cutoff)
            elif mode == 'svdr':
                result1, result2 = nodes[i]['right'].svdr_(
                    side='right',
                    rank=nodes[i]['right'].size() if keep_rank else rank,
                    cum_percentage=cum_percentage,
                    cutoff=cutoff)
            elif mode == 'qr':
                result1, result2 = nodes[i]['right'].qr_()
            else:
                raise ValueError('`mode` can only be "svd", "svdr" or "qr"')
            
            if renormalize:
                aux_norm = result2.norm()
                if not aux_norm.isinf() and (aux_norm > 0):
                    result2.tensor = result2.tensor / aux_norm
                    log_norm += aux_norm.log()

            result1 = result1.parameterize()
            nodes[i] = result1
            nodes[i + 1] = result2

        for i in range(len(nodes) - 1, oc, -1):
            if mode == 'svd':
                result1, result2 = nodes[i]['left'].svd_(
                    side='left',
                    rank=nodes[i]['left'].size() if keep_rank else rank,
                    cum_percentage=cum_percentage,
                    cutoff=cutoff)
            elif mode == 'svdr':
                result1, result2 = nodes[i]['left'].svdr_(
                    side='left',
                    rank=nodes[i]['left'].size() if keep_rank else rank,
                    cum_percentage=cum_percentage,
                    cutoff=cutoff)
            elif mode == 'qr':
                result1, result2 = nodes[i]['left'].rq_()
            else:
                raise ValueError('`mode` can only be "svd", "svdr" or "qr"')
            
            if renormalize:
                aux_norm = result1.norm()
                if not aux_norm.isinf() and (aux_norm > 0):
                    result1.tensor = result1.tensor / aux_norm
                    log_norm += aux_norm.log()

            result2 = result2.parameterize()
            nodes[i] = result2
            nodes[i - 1] = result1

        nodes[oc] = nodes[oc].parameterize()
        
        # Rescale
        if log_norm != 0:
            rescale = (log_norm / len(nodes)).exp()
        
        if renormalize and (log_norm != 0):
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

        for node, data_node in zip(self._mats_env, self._data_nodes.values()):
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
                 tensor: Optional[torch.Tensor] = None,
                 in_features: Optional[Sequence[int]] = None,
                 out_features: Optional[Sequence[int]] = None,
                 n_batches: int = 1,
                 init_method: Text = 'randn',
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs) -> None:
        
        tensors = None
        
        # n_features
        if not isinstance(n_features, int):
            raise TypeError('`n_features` should be int type')
        elif n_features < 1:
            raise ValueError('`n_features` should be at least 1')
        
        if tensor is None:
            # phys_dim
            if not isinstance(phys_dim, int):
                raise TypeError('`phys_dim` should be int type')

            # bond_dim
            if not isinstance(bond_dim, int):
                raise TypeError('`bond_dim` should be int type')
            
        else:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError('`tensor` should be torch.Tensor type')
            if len(tensor.shape) != 3:
                raise ValueError('`tensor` should be a rank-3 tensor')
            if tensor.shape[0] != tensor.shape[2]:
                raise ValueError('`tensor` first and last dimensions should'
                                 ' be equal so that the MPS can have '
                                 'periodic boundary conditions')
            
            tensors = [tensor] * n_features
        
        super().__init__(n_features=n_features,
                         phys_dim=phys_dim,
                         bond_dim=bond_dim,
                         boundary='pbc',
                         tensors=tensors,
                         in_features=in_features,
                         out_features=out_features,
                         n_batches=n_batches,
                         init_method=init_method,
                         device=device,
                         dtype=dtype,
                         **kwargs)
        self.name = 'umps'

    def _make_nodes(self) -> None:
        """Creates all the nodes of the MPS."""
        super()._make_nodes()
        
        # Virtual node
        uniform_memory = ParamNode(shape=(self._bond_dim[0],
                                          self._phys_dim[0],
                                          self._bond_dim[0]),
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
                       bond_dim=self._bond_dim[0],
                       tensor=None,
                       in_features=self._in_features,
                       out_features=self._out_features,
                       n_batches=self._n_batches,
                       init_method=None,
                       device=None,
                       dtype=None)
        new_mps.name = self.name + '_copy'
        if share_tensors:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor
        else:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor.clone()
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
            net._mats_env[i] = net._mats_env[i].parameterize(set_param)
        
        # It is important that uniform_memory is parameterized after the rest
        # of the nodes
        net.uniform_memory = net.uniform_memory.parameterize(set_param)
        
        # Tensor addresses have to be reassigned to reference
        # the uniform memory
        for node in net._mats_env:
            node.set_tensor_from(net.uniform_memory)
            
        return net
    
    def canonicalize(self,
                     oc: Optional[int] = None,
                     mode: Text = 'svd',
                     rank: Optional[int] = None,
                     cum_percentage: Optional[float] = None,
                     cutoff: Optional[float] = None,
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
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit"}, optional
        Initialization method. Check :meth:`initialize` for a more detailed
        explanation of the different initialization methods.
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
        if share_tensors:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor
        else:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor.clone()
        
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
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit"}, optional
        Initialization method. Check :meth:`initialize` for a more detailed
        explanation of the different initialization methods.
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
                if len(t.shape) != 3:
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

    def _make_nodes(self) -> None:
        """Creates all the nodes of the MPS."""
        super()._make_nodes()
        
        # Virtual node
        uniform_memory = ParamNode(shape=(self._bond_dim[0],
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
                            device=None,
                            dtype=None)
        new_mps.name = self.name + '_copy'
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
            net._mats_env[i] = net._mats_env[i].parameterize(set_param)
        
        # It is important that uniform_memory is parameterized after the rest
        # of the nodes
        net.uniform_memory = net.uniform_memory.parameterize(set_param)
        
        # Tensor addresses have to be reassigned to reference
        # the uniform memory
        for node in net._mats_env:
            node.set_tensor_from(net.uniform_memory)
        
        return net
    
    def canonicalize(self,
                     oc: Optional[int] = None,
                     mode: Text = 'svd',
                     rank: Optional[int] = None,
                     cum_percentage: Optional[float] = None,
                     cutoff: Optional[float] = None,
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
        
        if len(result.shape) == 3:
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
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit"}, optional
        Initialization method. Check :meth:`~MPS.initialize` for a more detailed
        explanation of the different initialization methods.
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
        if share_tensors:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor
        else:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor.clone()
        
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
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit"}, optional
        Initialization method. Check :meth:`~UMPS.initialize` for a more detailed
        explanation of the different initialization methods.
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
                 tensor: Optional[torch.Tensor] = None,
                 init_method: Text = 'randn',
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
                      tensor=tensor,
                      n_batches=2,
                      init_method=init_method,
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
                           bond_dim=self._bond_dim[0],
                           kernel_size=self._kernel_size,
                           stride=self._stride,
                           padding=self._padding,
                           dilation=self.dilation,
                           tensor=None,
                           init_method=None,
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
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit"}, optional
        Initialization method. Check :meth:`~MPSLayer.initialize` for a more detailed
        explanation of the different initialization methods.
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
        if share_tensors:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor
        else:
            for new_node, node in zip(new_mps._mats_env, self._mats_env):
                new_node.tensor = node.tensor.clone()
        
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
    init_method : {"zeros", "ones", "copy", "rand", "randn", "randn_eye", "unit"}, optional
        Initialization method. Check :meth:`~UMPSLayer.initialize` for a more
        detailed explanation of the different initialization methods.
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
                                device=None,
                                dtype=None)
        new_mps.name = self.name + '_copy'
        if share_tensors:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor
            new_mps.out_node.tensor = self.out_node.tensor
        else:
            new_mps.uniform_memory.tensor = self.uniform_memory.tensor.clone()
            new_mps.out_node.tensor = self.out_node.tensor.clone()
        
        return new_mps
