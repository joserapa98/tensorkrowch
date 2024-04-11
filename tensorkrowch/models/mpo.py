"""
This script contains:
    * MPO:
        + UMPO
"""

from typing import (List, Optional, Sequence,
                    Text, Tuple, Union)

import torch

import tensorkrowch.operations as op
from tensorkrowch.components import AbstractNode, Node, ParamNode
from tensorkrowch.components import TensorNetwork


class MPO(TensorNetwork):  # MARK: MPO
    """
    Class for Matrix Product Operators. This is the base class from which
    :class:`UMPO` inherits.
    
    Matrix Product Operators are formed by:
    
    * ``mats_env``: Environment of `matrix` nodes with axes
      ``("left", "input", "right", "output")``.
    
    * ``left_node``, ``right_node``: `Vector` nodes with axes ``("right",)``
      and ``("left",)``, respectively. These are used to close the boundary
      in the case ``boudary`` is ``"obc"``. Otherwise, both are ``None``.
    
    In contrast with :class:`MPS`, in ``MPO`` all nodes act both as input and
    output, with corresponding edges dedicated to that. Thus, data nodes will
    be connected to the ``"input"`` edge of all nodes. Upon contraction of the
    whole network, a resultant tensor will be formed, with as many dimensions
    as nodes were in the MPO.
    
    If all nodes have the same input dimensions, the input data tensor can be
    passed as a single tensor. Otherwise, it would have to be passed as a list
    of tensors with different sizes.

    Parameters
    ----------
    n_features : int, optional
        Number of nodes that will be in ``mats_env``. That is, number of nodes
        without taking into account ``left_node`` and ``right_node``.
    in_dim : int, list[int] or tuple[int], optional
        Input dimension(s). If given as a sequence, its length should be equal
        to ``n_features``.
    out_dim : int, list[int] or tuple[int], optional
        Output dimension(s). If given as a sequence, its length should be equal
        to ``n_features``.
    bond_dim : int, list[int] or tuple[int], optional
        Bond dimension(s). If given as a sequence, its length should be equal
        to ``n_features`` (if ``boundary = "pbc"``) or ``n_features - 1`` (if
        ``boundary = "obc"``). The i-th bond dimension is always the dimension
        of the right edge of the i-th node.
    boundary : {"obc", "pbc"}
        String indicating whether periodic or open boundary conditions should
        be used.
    tensors: list[torch.Tensor] or tuple[torch.Tensor], optional
        Instead of providing ``n_features``, ``in_dim``, ``in_dim``, ``bond_dim``
        and ``boundary``, a list of MPO tensors can be provided. In such case,
        all mentioned attributes will be inferred from the given tensors. All
        tensors should be rank-4 tensors, with shape ``(bond_dim, in_dim,
        bond_dim, out_dim)``. If the first and last elements are rank-3 tensors,
        with shapes ``(in_dim, bond_dim, out_dim)``, ``(bond_dim, in_dim, out_dim)``,
        respectively, the inferred boundary conditions will be "obc". Also, if
        ``tensors`` contains a single element, it can be rank-2 ("obc") or
        rank-4 ("pbc").
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
        Initialization method. Check :meth:`initialize` for a more detailed
        explanation of the different initialization methods.
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
    Examples
    --------
    ``MPO`` with same input/output dimensions:
    
    >>> mpo = tk.models.MPO(n_features=5,
    ...                     in_dim=2,
    ...                     out_dim=2,
    ...                     bond_dim=5)
    >>> data = torch.ones(20, 5, 2) # batch_size x n_features x feature_size
    >>> result = mpo(data)
    >>> result.shape
    torch.Size([20, 2, 2, 2, 2, 2])
    
    ``MPO`` with different input/physical dimensions:
    
    >>> mpo = tk.models.MPO(n_features=5,
    ...                     in_dim=list(range(2, 7)),
    ...                     out_dim=list(range(7, 2, -1)),
    ...                     bond_dim=5)
    >>> data = [torch.ones(20, i)
    ...         for i in range(2, 7)] # n_features * [batch_size x feature_size]
    >>> result = mpo(data)
    >>> result.shape
    torch.Size([20, 7, 6, 5, 4, 3])
    """

    def __init__(self,
                 n_features: Optional[int] = None,
                 in_dim: Optional[Union[int, Sequence[int]]] = None,
                 out_dim: Optional[Union[int, Sequence[int]]] = None,
                 bond_dim: Optional[Union[int, Sequence[int]]] = None,
                 boundary: Text = 'obc',
                 tensors: Optional[Sequence[torch.Tensor]] = None,
                 n_batches: int = 1,
                 init_method: Text = 'randn',
                 device: Optional[torch.device] = None,
                 **kwargs) -> None:

        super().__init__(name='mpo')
        
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

            # in_dim
            if isinstance(in_dim, (list, tuple)):
                if len(in_dim) != n_features:
                    raise ValueError('If `in_dim` is given as a sequence of int, '
                                     'its length should be equal to `n_features`')
                self._in_dim = list(in_dim)
            elif isinstance(in_dim, int):
                self._in_dim = [in_dim] * n_features
            else:
                raise TypeError('`in_dim` should be int, tuple[int] or list[int] '
                                'type')
            
            # out_dim
            if isinstance(out_dim, (list, tuple)):
                if len(out_dim) != n_features:
                    raise ValueError('If `out_dim` is given as a sequence of int, '
                                     'its length should be equal to `n_features`')
                self._out_dim = list(out_dim)
            elif isinstance(out_dim, int):
                self._out_dim = [out_dim] * n_features
            else:
                raise TypeError('`out_dim` should be int, tuple[int] or list[int] '
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
            if not isinstance(tensors, (list, tuple)):
                raise TypeError('`tensors` should be a tuple[torch.Tensor] or '
                                'list[torch.Tensor] type')
            else:
                self._n_features = len(tensors)
                self._in_dim = []
                self._out_dim = []
                self._bond_dim = []
                for i, t in enumerate(tensors):
                    if not isinstance(t, torch.Tensor):
                        raise TypeError('`tensors` should be a tuple[torch.Tensor]'
                                        ' or list[torch.Tensor] type')
                    
                    if i == 0:
                        if len(t.shape) not in [2, 3, 4]:
                            raise ValueError(
                                'The first and last elements in `tensors` '
                                'should be both rank-3 or rank-4 tensors. If'
                                ' the first element is also the last one,'
                                ' it should be a rank-2 tensor')
                        if len(t.shape) == 2:
                            self._boundary = 'obc'
                            self._in_dim.append(t.shape[0])
                            self._out_dim.append(t.shape[1])
                        elif len(t.shape) == 3:
                            self._boundary = 'obc'
                            self._in_dim.append(t.shape[0])
                            self._bond_dim.append(t.shape[1])
                            self._out_dim.append(t.shape[2])
                        else:
                            self._boundary = 'pbc'
                            self._in_dim.append(t.shape[1])
                            self._bond_dim.append(t.shape[2])
                            self._out_dim.append(t.shape[3])
                    elif i == (self._n_features - 1):
                        if len(t.shape) != len(tensors[0].shape):
                            raise ValueError(
                                'The first and last elements in `tensors` '
                                'should have the same rank. Both should be '
                                'rank-3 or rank-4 tensors. If the first '
                                'element is also the last one, it should '
                                'be a rank-2 tensor')
                        if len(t.shape) == 3:
                            self._in_dim.append(t.shape[1])
                            self._out_dim.append(t.shape[2])
                        else:
                            if t.shape[2] != tensors[0].shape[0]:
                                raise ValueError(
                                    'If the first and last elements in `tensors`'
                                    ' are rank-4 tensors, the first dimension '
                                    'of the first element should coincide with'
                                    ' the third dimension of the last element')
                            self._in_dim.append(t.shape[1])
                            self._bond_dim.append(t.shape[2])
                            self._out_dim.append(t.shape[3])
                    else:
                        if len(t.shape) != 4:
                            raise ValueError(
                                'The elements of `tensors` should be rank-4 '
                                'tensors, except the first and lest elements'
                                ' if boundary is "obc"')
                        self._in_dim.append(t.shape[1])
                        self._bond_dim.append(t.shape[2])
                        self._out_dim.append(t.shape[3])
        
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
                        **kwargs)
    
    # ----------
    # Properties
    # ----------
    @property
    def n_features(self) -> int:
        """Returns number of nodes."""
        return self._n_features

    @property
    def in_dim(self) -> List[int]:
        """Returns input dimensions."""
        return self._in_dim
    
    @property
    def out_dim(self) -> List[int]:
        """Returns output dimensions."""
        return self._out_dim

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
    
    # -------
    # Methods
    # -------
    def _make_nodes(self) -> None:
        """Creates all the nodes of the MPO."""
        if self._leaf_nodes:
            raise ValueError('Cannot create MPO nodes if the MPO already has '
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
                                    self._in_dim[i],
                                    aux_bond_dim[i],
                                    self._out_dim[i]),
                             axes_names=('left', 'input', 'right', 'output'),
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
    
    def initialize(self,
                   tensors: Optional[Sequence[torch.Tensor]] = None,
                   init_method: Optional[Text] = 'randn',
                   device: Optional[torch.device] = None,
                   **kwargs: float) -> None:
        """
        Initializes all the nodes of the :class:`MPO`. It can be called when
        instantiating the model, or to override the existing nodes' tensors.
        
        There are different methods to initialize the nodes:
        
        * ``{"zeros", "ones", "copy", "rand", "randn"}``: Each node is
          initialized calling :meth:`~tensorkrowch.AbstractNode.set_tensor` with
          the given method, ``device`` and ``kwargs``.
        
        Parameters
        ----------
        tensors : list[torch.Tensor] or tuple[torch.Tensor], optional
            Sequence of tensors to set in each of the MPO nodes. If ``boundary``
            is ``"obc"``, all tensors should be rank-4, except the first and
            last ones, which can be rank-3, or rank-2 (if the first and last are
            the same). If ``boundary`` is ``"pbc"``, all tensors should be
            rank-4.
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensors if ``init_method`` is provided.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        """
        if self._boundary == 'obc':
            self._left_node.set_tensor(init_method='copy', device=device)
            self._right_node.set_tensor(init_method='copy', device=device)

        if tensors is not None:
            if len(tensors) != self._n_features:
                raise ValueError('`tensors` should be a sequence of `n_features`'
                                 ' elements')
            
            if self._boundary == 'obc':
                if len(tensors) == 1:
                    tensors[0] = tensors[0].reshape(1,
                                                    tensors[0].shape[0],
                                                    1,
                                                    tensors[0].shape[1])
                    
                else:
                    # Left node
                    aux_tensor = torch.zeros(*self._mats_env[0].shape,
                                             device=tensors[0].device)
                    aux_tensor[0] = tensors[0]
                    tensors[0] = aux_tensor
                    
                    # Right node
                    aux_tensor = torch.zeros(*self._mats_env[-1].shape,
                                             device=tensors[-1].device)
                    aux_tensor[..., 0, :] = tensors[-1]
                    tensors[-1] = aux_tensor
                
            for tensor, node in zip(tensors, self._mats_env):
                node.tensor = tensor
                
        elif init_method is not None:
                
            for i, node in enumerate(self._mats_env):
                node.set_tensor(init_method=init_method,
                                device=device,
                                **kwargs)
                
                if self._boundary == 'obc':
                    aux_tensor = torch.zeros(*node.shape, device=device)
                    if i == 0:
                        # Left node
                        aux_tensor[0] = node.tensor[0]
                    elif i == (self._n_features - 1):
                        # Right node
                        aux_tensor[..., 0, :] = node.tensor[..., 0, :]
                    node.tensor = aux_tensor
    
    def set_data_nodes(self) -> None:
        """
        Creates ``data`` nodes and connects each of them to the ``"input"``
        edge of each node.
        """      
        input_edges = [node['input'] for node in self._mats_env]
        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self._n_batches)
    
    def copy(self, share_tensors: bool = False) -> 'MPO':
        """
        Creates a copy of the :class:`MPO`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether tensors in the copied MPO should be
            set as the tensors in the current MPO (``True``), or cloned
            (``False``). In the former case, tensors in both MPO's will be
            the same, which might be useful if one needs more than one copy
            of an MPO, but wants to compute all the gradients with respect
            to the same, unique, tensors.

        Returns
        -------
        MPO
        """
        new_mpo = MPO(n_features=self._n_features,
                      in_dim=self._in_dim,
                      out_dim=self._out_dim,
                      bond_dim=self._bond_dim,
                      boundary=self._boundary,
                      tensors=None,
                      n_batches=self._n_batches,
                      init_method=None,
                      device=None)
        new_mpo.name = self.name + '_copy'
        if share_tensors:
            for new_node, node in zip(new_mpo._mats_env, self._mats_env):
                new_node.tensor = node.tensor
        else:
            for new_node, node in zip(new_mpo._mats_env, self._mats_env):
                new_node.tensor = node.tensor.clone()
        
        return new_mpo
    
    def _input_contraction(self,
                           nodes_env: List[Node],
                           inline_input: bool = False) -> Tuple[
                                                       Optional[List[Node]],
                                                       Optional[List[Node]]]:
        """Contracts input data nodes with MPO nodes."""
        if inline_input:
            mats_result = [node['input'].contract() for node in nodes_env]
            return mats_result

        else:
            if nodes_env:
                stack = op.stack(nodes_env)
                stack_data = op.stack(
                    [node.neighbours('input') for node in nodes_env])

                stack ^ stack_data

                result = stack_data @ stack
                mats_result = op.unbind(result)
                return mats_result
            else:
                return []

    @staticmethod
    def _inline_contraction(nodes: List[Node], from_left: bool = True) -> Node:
        """Contracts sequence of MPO nodes (matrices) inline."""
        if from_left:
            result_node = nodes[0]
            for node in nodes[1:]:
                result_node @= node
            return result_node
        else:
            result_node = nodes[-1]
            for node in nodes[-2::-1]:
                result_node = node @ result_node
            return result_node

    def _contract_envs_inline(self, mats_env: List[Node]) -> Node:
        """Contracts nodes environments inline."""
        from_left = True
        if self._boundary == 'obc':
            if mats_env[0].neighbours('left') is self._left_node:
                mats_env = [self._left_node] + mats_env
            if mats_env[-1].neighbours('right') is self._right_node:
                mats_env = mats_env + [self._right_node]
                from_left = False
        return self._inline_contraction(mats_env, from_left)

    def _aux_pairwise(self, nodes: List[Node]) -> Tuple[List[Node],
    List[Node]]:
        """Contracts a sequence of MPO nodes (matrices) pairwise."""
        length = len(nodes)
        aux_nodes = nodes
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
            aux_nodes = op.unbind(aux_nodes)

            return aux_nodes, leftover
        return nodes, []

    def _pairwise_contraction(self, mats_nodes: List[Node]) -> Node:
        """Contracts nodes environments pairwise."""
        length = len(mats_nodes)
        aux_nodes = mats_nodes
        if length > 1:
            leftovers = []
            while length > 1:
                aux1, aux2 = self._aux_pairwise(aux_nodes)
                aux_nodes = aux1
                leftovers = aux2 + leftovers
                length = len(aux1)

            aux_nodes = aux_nodes + leftovers
            return self._pairwise_contraction(aux_nodes)

        return self._contract_envs_inline(aux_nodes)
    
    def contract(self,
                 inline_input: bool = False,
                 inline_mats: bool = False) -> Node:
        """
        Contracts the whole MPO.
        
        Parameters
        ----------
        inline_input : bool
            Boolean indicating whether input ``data`` nodes should be contracted
            with the ``MPO`` nodes inline (one contraction at a time) or in a
            single stacked contraction.
        inline_mats : bool
            Boolean indicating whether the sequence of matrices (resultant
            after contracting the input ``data`` nodes) should be contracted
            inline or as a sequence of pairwise stacked contrations.

        Returns
        -------
        Node
        """
        mats_env = self._input_contraction(self._mats_env, inline_input)
        
        if inline_mats:
            result = self._contract_envs_inline(mats_env)
        else:
            result = self._pairwise_contraction(mats_env)
            
        # Contract periodic edge
        if self._boundary == 'pbc':
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


class UMPO(MPO):  # MARK: UMPO
    """
    Class for Uniform (translationally invariant) Matrix Product Operators. It is
    the uniform version of :class:`MPO`, that is, all nodes share the same
    tensor. Thus this class cannot have different input/output or bond dimensions
    for each node, and boundary conditions are always periodic (``"pbc"``).
    
    |
    
    For a more detailed list of inherited properties and methods,
    check :class:`MPSO`.

    Parameters
    ----------
    n_features : int
        Number of nodes that will be in ``mats_env``.
    in_dim : int, optional
        Input dimension.
    out_dim : int, optional
        Output dimension.
    bond_dim : int, optional
        Bond dimension.
    tensor: torch.Tensor, optional
        Instead of providing ``in_dim``, ``out_dim`` and ``bond_dim``, a single
        tensor can be provided. ``n_features`` is still needed to specify how
        many times the tensor should be used to form a finite MPO. The tensor
        should be rank-4, with its first and third dimensions being equal.
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
        Initialization method. Check :meth:`initialize` for a more detailed
        explanation of the different initialization methods.
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
    Examples
    --------
    >>> mpo = tk.models.UMPO(n_features=4,
    ...                      in_dim=2,
    ...                      out_dim=2,
    ...                      bond_dim=5)
    >>> for node in mpo.mats_env:
    ...     assert node.tensor_address() == 'virtual_uniform'
    ...
    >>> data = torch.ones(20, 4, 2) # batch_size x n_features x feature_size
    >>> result = mpo(data)
    >>> result.shape
    torch.Size([20, 2, 2, 2, 2])
    """

    def __init__(self,
                 n_features: int = None,
                 in_dim: Optional[int] = None,
                 out_dim: Optional[int] = None,
                 bond_dim: Optional[int] = None,
                 tensor: Optional[torch.Tensor] = None,
                 n_batches: int = 1,
                 init_method: Text = 'randn',
                 device: Optional[torch.device] = None,
                 **kwargs) -> None:

        tensors = None
        
        # n_features
        if not isinstance(n_features, int):
            raise TypeError('`n_features` should be int type')
        elif n_features < 1:
            raise ValueError('`n_features` should be at least 1')
        
        if tensor is None:
            # in_dim
            if not isinstance(in_dim, int):
                raise TypeError('`in_dim` should be int type')
            
            # out_dim
            if not isinstance(out_dim, int):
                raise TypeError('`out_dim` should be int type')

            # bond_dim
            if not isinstance(bond_dim, int):
                raise TypeError('`bond_dim` should be int type')
            
        else:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError('`tensor` should be torch.Tensor type')
            if len(tensor.shape) != 4:
                raise ValueError('`tensor` should be a rank-4 tensor')
            if tensor.shape[0] != tensor.shape[2]:
                raise ValueError('`tensor` first and last dimensions should'
                                 ' be equal so that the MPS can have '
                                 'periodic boundary conditions')
            
            tensors = [tensor] * n_features
        
        super().__init__(n_features=n_features,
                         in_dim=in_dim,
                         out_dim=out_dim,
                         bond_dim=bond_dim,
                         boundary='pbc',
                         tensors=tensors,
                         n_batches=n_batches,
                         init_method=init_method,
                         device=device,
                         **kwargs)
        self.name = 'umpo'
    
    def _make_nodes(self) -> None:
        """Creates all the nodes of the MPO."""
        super()._make_nodes()
        
        # Virtual node
        uniform_memory = ParamNode(shape=(self._bond_dim[0],
                                          self._in_dim[0],
                                          self._bond_dim[0],
                                          self._out_dim[0]),
                                   axes_names=('left', 'input', 'right', 'output'),
                                   name='virtual_uniform',
                                   network=self,
                                   virtual=True)
        self.uniform_memory = uniform_memory
        
        for node in self._mats_env:
            node.set_tensor_from(uniform_memory)
    
    def initialize(self,
                   tensors: Optional[Sequence[torch.Tensor]] = None,
                   init_method: Optional[Text] = 'randn',
                   device: Optional[torch.device] = None,
                   **kwargs: float) -> None:
        """
        Initializes the common tensor of the :class:`UMPO`. It can be called
        when instantiating the model, or to override the existing nodes' tensors.
        
        There are different methods to initialize the nodes:
        
        * ``{"zeros", "ones", "copy", "rand", "randn"}``: The tensor is
          initialized calling :meth:`~tensorkrowch.AbstractNode.set_tensor` with
          the given method, ``device`` and ``kwargs``.
        
        Parameters
        ----------
        tensors : list[torch.Tensor] or tuple[torch.Tensor], optional
            Sequence of a single tensor to set in each of the MPO nodes. The
            tensor should be rank-4, with its first and third dimensions being
            equal.
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensors if ``init_method`` is provided.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        """
        node = self.uniform_memory
        
        if tensors is not None:
            self.uniform_memory.tensor = tensors[0]
        
        elif init_method is not None:
            self.uniform_memory.set_tensor(init_method=init_method,
                                           device=device,
                                           **kwargs)
    
    def copy(self, share_tensors: bool = False) -> 'UMPO':
        """
        Creates a copy of the :class:`UMPO`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether the common tensor in the copied UMPO
            should be set as the tensor in the current UMPO (``True``), or
            cloned (``False``). In the former case, the tensor in both UMPO's
            will be the same, which might be useful if one needs more than one
            copy of a UMPO, but wants to compute all the gradients with respect
            to the same, unique, tensor.

        Returns
        -------
        UMPO
        """
        new_mpo = UMPO(n_features=self._n_features,
                       in_dim=self._in_dim[0],
                       out_dim=self._out_dim[0],
                       bond_dim=self._bond_dim[0],
                       tensor=None,
                       n_batches=self._n_batches,
                       init_method=None,
                       device=None)
        new_mpo.name = self.name + '_copy'
        if share_tensors:
            new_mpo.uniform_memory.tensor = self.uniform_memory.tensor
        else:
            new_mpo.uniform_memory.tensor = self.uniform_memory.tensor.clone()
        return new_mpo