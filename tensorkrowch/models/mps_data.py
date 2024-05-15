"""
This script contains:
    * MPSData
"""

from abc import abstractmethod, ABC
from typing import (List, Optional, Sequence,
                    Text, Tuple, Union)

from math import sqrt

import torch
import torch.nn as nn

import tensorkrowch.operations as op
from tensorkrowch.components import AbstractNode, Node, ParamNode
from tensorkrowch.components import TensorNetwork

from tensorkrowch.utils import split_sequence_into_regions, random_unitary


class MPSData(TensorNetwork):  # MARK: MPSData
    """
    Class for data vectors in the form of Matrix Product States. That is, this
    is a class similar to :class:`MPS`, but where all nodes can have additional
    batch edges.
    
    Besides, since this class is intended to store data vectors, all 
    :class:`nodes <tensorkrowch.Node>` are non-parametric, and are ``data``
    nodes. Also, this class does not have an inherited ``contract`` method,
    since it is not intended to be contracted with input data, but rather
    act itself as input data of another tensor network model.
    
    Similar to :class:`MPS`, ``MPSData`` is formed by:
    
    * ``mats_env``: Environment of `matrix` nodes with axes
      ``("batch_0", ..., "batch_n", "left", "feature", "right")``.
    
    * ``left_node``, ``right_node``: `Vector` nodes with axes ``("right",)``
      and ``("left",)``, respectively. These are used to close the boundary
      in the case ``boudary`` is ``"obc"``. Otherwise, both are ``None``.
    
    Since ``MPSData`` is designed to store input data vectors, this can be
    accomplished by calling the custom :meth:`add_data` method with a given
    list of tensors. This is in contrast to the usual way of setting nodes'
    tensors in :class:`MPS` and its derived classes via :meth:`MPS.initialize`.

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
    n_batches : int
        Number of batch edges of the MPS nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    tensors: list[torch.Tensor] or tuple[torch.Tensor], optional
        Instead of providing ``n_features``, ``phys_dim``, ``bond_dim`` and
        ``boundary``, a list of MPS tensors can be provided. In such case, all
        mentioned attributes will be inferred from the given tensors. All
        tensors should be rank-(n+3) tensors, with shape
        ``(batch_1, ..., batch_n, bond_dim, phys_dim, bond_dim)``. If the first
        and last elements are rank-(n+2) tensors, with shapes
        ``(batch_1, ..., batch_n, phys_dim, bond_dim)``,
        ``(batch_1, ..., batch_n, bond_dim, phys_dim)``, respectively,
        the inferred boundary conditions will be "obc". Also, if ``tensors``
        contains a single element, it can be rank-(n+1) ("obc") or rank-(n+3)
        ("pbc").
    init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
        Initialization method. Check :meth:`initialize` for a more detailed
        explanation of the different initialization methods. By default it is
        ``None``, since ``MPSData`` is intended to store input data vectors in
        MPS form, rather than initializing its own random tensors. Check
        :meth:`add_data` to see how to initialize MPS nodes with data tensors.
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
            Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
    Examples
    --------
    >>> mps = tk.models.MPSData(n_features=5,
    ...                         phys_dim=2,
    ...                         bond_dim=5,
    ...                         boundary="pbc")
    >>> # n_features * (batch_size x bond_dim x feature_size x bond_dim)
    >>> data = [torch.ones(20, 5, 2, 5) for _ in range(5)]
    >>> mps.add_data(data)
    >>> for node in mps.mats_env:
    ...     assert node.shape == (20, 5, 2, 5)
    """

    def __init__(self,
                 n_features: Optional[int] = None,
                 phys_dim: Optional[Union[int, Sequence[int]]] = None,
                 bond_dim: Optional[Union[int, Sequence[int]]] = None,
                 boundary: Text = 'obc',
                 n_batches: int = 1,
                 tensors: Optional[Sequence[torch.Tensor]] = None,
                 init_method: Optional[Text] = None,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs) -> None:

        super().__init__(name='mps_data')
        
        # n_batches
        if not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        self._n_batches = n_batches
        
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
            if isinstance(phys_dim, (list, tuple)):
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
            if isinstance(bond_dim, (list, tuple)):
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
                        if len(t.shape) not in [n_batches + 1,
                                                n_batches + 2,
                                                n_batches + 3]:
                            raise ValueError(
                                'The first and last elements in `tensors` '
                                'should be both rank-(n+2) or rank-(n+3) tensors.'
                                ' If the first element is also the last one,'
                                ' it should be a rank-(n+1) tensor')
                        if len(t.shape) == n_batches + 1:
                            self._boundary = 'obc'
                            self._phys_dim.append(t.shape[-1])
                        elif len(t.shape) == n_batches + 2:
                            self._boundary = 'obc'
                            self._phys_dim.append(t.shape[-2])
                            self._bond_dim.append(t.shape[-1])
                        else:
                            self._boundary = 'pbc'
                            self._phys_dim.append(t.shape[-2])
                            self._bond_dim.append(t.shape[-1])
                    elif i == (self._n_features - 1):
                        if len(t.shape) != len(tensors[0].shape):
                            raise ValueError(
                                'The first and last elements in `tensors` '
                                'should have the same rank. Both should be '
                                'rank-(n+2) or rank-(n+3) tensors. If the first'
                                ' element is also the last one, it should '
                                'be a rank-(n+1) tensor')
                        if len(t.shape) == n_batches + 2:
                            self._phys_dim.append(t.shape[-1])
                        else:
                            if t.shape[-1] != tensors[0].shape[-3]:
                                raise ValueError(
                                    'If the first and last elements in `tensors`'
                                    ' are rank-(n+3) tensors, the first dimension'
                                    ' of the first element should coincide with'
                                    ' the last dimension of the last element '
                                    '(ignoring batch dimensions)')
                            self._phys_dim.append(t.shape[-2])
                            self._bond_dim.append(t.shape[-1])
                    else:
                        if len(t.shape) != n_batches + 3:
                            raise ValueError(
                                'The elements of `tensors` should be rank-(n+3) '
                                'tensors, except the first and lest elements'
                                ' if boundary is "obc"')
                        self._phys_dim.append(t.shape[-2])
                        self._bond_dim.append(t.shape[-1])
        
        # Properties
        self._left_node = None
        self._right_node = None
        self._mats_env = []

        # Create Tensor Network
        self._make_nodes()
        self.initialize(init_method=init_method,
                            device=device,
                            dtype=dtype,
                            **kwargs)
        
        if tensors is not None:
            self.add_data(data=tensors)
    
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
        """Returns number of batch edges of the MPS nodes."""
        return self._n_batches
    
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
    def tensors(self) -> List[torch.Tensor]:
        """Returns the list of MPS tensors."""
        mps_tensors = [node.tensor for node in self._mats_env]
        if self._boundary == 'obc':
            mps_tensors[0] = mps_tensors[0][0, :, :]
            mps_tensors[-1] = mps_tensors[-1][:, :, 0]
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
            node = Node(shape=(*([1] * self._n_batches),
                               aux_bond_dim[i - 1],
                               self._phys_dim[i],
                               aux_bond_dim[i]),
                        axes_names=(*(['batch'] * self._n_batches),
                                    'left',
                                    'feature',
                                    'right'),
                        name=f'mats_env_data_node_({i})',
                        data=True,
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
    
    def initialize(self,
                   init_method: Optional[Text] = 'randn',
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None,
                   **kwargs: float) -> None:
        """
        Initializes all the nodes of the :class:`MPSData`. It can be called when
        instantiating the model, or to override the existing nodes' tensors.
        
        There are different methods to initialize the nodes:
        
        * ``{"zeros", "ones", "copy", "rand", "randn"}``: Each node is
          initialized calling :meth:`~tensorkrowch.AbstractNode.set_tensor` with
          the given method, ``device``, ``dtype`` and ``kwargs``.
        
        Parameters
        ----------
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method. Check :meth:`add_data` to see how to
            initialize MPS nodes with data tensors.
        device : torch.device, optional
            Device where to initialize the tensors if ``init_method`` is provided.
        dtype : torch.dtype, optional
            Dtype of the tensor if ``init_method`` is provided.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        """
        if self._boundary == 'obc':
            self._left_node.set_tensor(init_method='copy',
                                       device=device,
                                       dtype=dtype)
            self._right_node.set_tensor(init_method='copy',
                                        device=device,
                                        dtype=dtype)
                
        if init_method is not None:
            for i, node in enumerate(self._mats_env):
                node.set_tensor(init_method=init_method,
                                device=device,
                                dtype=dtype,
                                **kwargs)
                
                if self._boundary == 'obc':
                    aux_tensor = torch.zeros(*node.shape,
                                             device=device,
                                             dtype=dtype)
                    if i == 0:
                        # Left node
                        aux_tensor[..., 0, :, :] = node.tensor[..., 0, :, :]
                        node.tensor = aux_tensor
                    elif i == (self._n_features - 1):
                        # Right node
                        aux_tensor[..., 0] = node.tensor[..., 0]
                        node.tensor = aux_tensor
    
    def add_data(self, data: Sequence[torch.Tensor]) -> None:
        """
        Adds data to MPS data nodes. Input is a list of mps tensors.
        
        The physical dimensions of the given data tensors should coincide with
        the physical dimensions of the MPS. The bond dimensions can be different.
        
        Parameters
        ----------
        data : list[torch.Tensor] or tuple[torch.Tensor]
            A sequence of tensors, one for each of the MPS nodes. If ``boundary``
            is ``"pbc"``, all tensors should have the same rank, with shapes
            ``(batch_0, ..., batch_n, bond_dim, phys_dim, bond_dim)``. If
            ``boundary`` is ``"obc"``, the first and last tensors should have
            shapes ``(batch_0, ..., batch_n, phys_dim, bond_dim)`` and
            ``(batch_0, ..., batch_n, bond_dim, phys_dim)``, respectively.
        """
        if not isinstance(data, Sequence):
            raise TypeError(
                '`data` should be list[torch.Tensor] or tuple[torch.Tensor] type')
        if len(data) != self._n_features:
            raise ValueError('`data` should be a sequence of tensors of length'
                             ' equal to `n_features`')
        if any([not isinstance(x, torch.Tensor) for x in data]):
            raise TypeError(
                '`data` should be list[torch.Tensor] or tuple[torch.Tensor] type')
        
        # Check physical dimensions coincide
        for i, (data_tensor, node) in enumerate(zip(data, self._mats_env)):
            if (self._boundary == 'obc') and (i == (self._n_features - 1)):
                if data_tensor.shape[-1] != node.shape[-2]:
                    raise ValueError(
                        f'Physical dimension {data_tensor.shape[-1]} of '
                        f'data tensor at position {i} does not coincide '
                        f'with the corresponding physical dimension '
                        f'{node.shape[-2]} of the MPS')
            else:
                if data_tensor.shape[-2] != node.shape[-2]:
                    raise ValueError(
                        f'Physical dimension {data_tensor.shape[-2]} of '
                        f'data tensor at position {i} does not coincide '
                        f'with the corresponding physical dimension '
                        f'{node.shape[-2]} of the MPS')
        
        data = data[:]
        device = data[0].device
        dtype = data[0].dtype
        for i, node in enumerate(self._mats_env):
            if self._boundary == 'obc':
                if (i == 0) and (i == (self._n_features - 1)):
                    aux_tensor = torch.zeros(*data[i].shape[:-1],
                                             *node.shape[-3:],
                                             device=device,
                                             dtype=dtype)
                    aux_tensor[..., 0, :, 0] = data[i]
                    data[i] = aux_tensor
                elif i == 0:
                    aux_tensor = torch.zeros(*data[i].shape[:-2],
                                             *node.shape[-3:-1],
                                             data[i].shape[-1],
                                             device=device,
                                             dtype=dtype)
                    aux_tensor[..., 0, :, :] = data[i]
                    data[i] = aux_tensor
                elif i == (self._n_features - 1):
                    aux_tensor = torch.zeros(*data[i].shape[:-1],
                                             *node.shape[-2:],
                                             device=device,
                                             dtype=dtype)
                    aux_tensor[..., 0] = data[i]
                    data[i] = aux_tensor
                    
            node._direct_set_tensor(data[i])
        
        # Send left and right nodes to correct device and dtype
        if self._boundary == 'obc':
            if self._left_node.device != device:
                self._left_node.tensor = self._left_node.tensor.to(device)
            if self._left_node.dtype != dtype:
                self._left_node.tensor = self._left_node.tensor.to(dtype)
            
            if self._right_node.device != device:
                self._right_node.tensor = self._right_node.tensor.to(device)
            if self._right_node.dtype != dtype:
                self._right_node.tensor = self._right_node.tensor.to(dtype)
        
        # Update bond dim
        if self._boundary == 'obc':
            self._bond_dim = [node['right'].size() for node in self._mats_env[:-1]]
        else:
            self._bond_dim = [node['right'].size() for node in self._mats_env]
