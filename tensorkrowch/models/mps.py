"""
This script contains:
    * MPS
    * UMPS
    * ConvMPS
    * ConvUMPS
"""

from typing import (List, Optional, Sequence,
                    Text, Tuple, Union)

import torch
import torch.nn as nn

import tensorkrowch.operations as op
from tensorkrowch.components import AbstractNode, Node, ParamNode
from tensorkrowch.components import TensorNetwork


class MPS(TensorNetwork):
    """
    Class for Matrix Product States, where all nodes are input nodes, that is,
    they are all connected to ``data`` nodes that will store the input data
    tensor(s). When contracting the MPS with new input data, the result will
    be a just a number.
    
    If the input dimensions of all the input nodes are equal, the input data
    tensor can be passed as a single tensor. Otherwise, it would have to be
    passed as a list of tensors with different sizes.
    
    An ``MPS`` is formed by the following nodes:
    
    * ``left_node``, ``right_node``: `Vector` nodes with axes ``("input", "right")``
      and ``("left", "input")``, respectively. These are the nodes at the
      extremes of the ``MPS``. If ``boundary`` is ``"pbc""``, both are ``None``.
      
    * ``mats_env``: Environment of `matrix` nodes that. These nodes have axes
      ``("left", "input", "right")``.

    Parameters
    ----------
    n_features : int
        Number of input nodes.
    in_dim : int, list[int] or tuple[int]
        Input dimension(s). Equivalent to the physical dimension. If given as a
        sequence, its length should be equal to ``n_features``.
    bond_dim : int, list[int] or tuple[int]
        Bond dimension(s). If given as a sequence, its length should be equal
        to ``n_features`` (if ``boundary = "pbc"``) or ``n_features - 1`` (if
        ``boundary = "obc"``). The i-th bond dimension is always the dimension
        of the right edge of the i-th node.
    boundary : {"obc", "pbc"}
        String indicating whether periodic or open boundary conditions should
        be used.
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
        
    Examples
    --------
    ``MPS`` with same input/physical dimensions:
    
    >>> mps = tk.models.MPS(n_features=5,
    ...                     in_dim=2,
    ...                     bond_dim=5)
    >>> data = torch.ones(20, 5, 2) # batch_size x n_features x feature_size
    >>> result = mps(data)
    >>> result.shape
    torch.Size([20])
    
    ``MPS`` with different input/physical dimensions:
    
    >>> mps = tk.models.MPS(n_features=5,
    ...                     in_dim=list(range(2, 7)),
    ...                     bond_dim=5)
    >>> data = [torch.ones(20, i)
    ...         for i in range(2, 7)] # n_features * [batch_size x feature_size]
    >>> result = mps(data)
    >>> result.shape
    torch.Size([20])
    """

    def __init__(self,
                 n_features: int,
                 in_dim: Union[int, Sequence[int]],
                 bond_dim: Union[int, Sequence[int]],
                 boundary: Text = 'obc',
                 n_batches: int = 1) -> None:

        super().__init__(name='mps')

        # boundary
        if boundary not in ['obc', 'pbc']:
            raise ValueError('`boundary` should be one of "obc" or "pbc"')
        self._boundary = boundary

        # n_features
        if boundary == 'obc':
            if n_features < 2:
                raise ValueError('If `boundary` is "obc", at least '
                                 'there has to be 2 nodes')
        elif boundary == 'pbc':
            if n_features < 1:
                raise ValueError('If `boundary` is "pbc", at least '
                                 'there has to be one node')
        else:
            raise ValueError('`boundary` should be one of "obc" or "pbc"')
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

        # bond_dim
        if isinstance(bond_dim, (list, tuple)):
            if boundary == 'obc':
                if len(bond_dim) != n_features - 1:
                    raise ValueError('If `bond_dim` is given as a sequence of int,'
                                     ' and `boundary` is "obc", its length '
                                     'should be equal to `n_features` - 1')
            elif boundary == 'pbc':
                if len(bond_dim) != n_features:
                    raise ValueError('If `bond_dim` is given as a sequence of int,'
                                     ' and `boundary` is "pbc", its length '
                                     'should be equal to `n_features`')
            self._bond_dim = list(bond_dim)
        elif isinstance(bond_dim, int):
            if boundary == 'obc':
                self._bond_dim = [bond_dim] * (n_features - 1)
            elif boundary == 'pbc':
                self._bond_dim = [bond_dim] * n_features
        else:
            raise TypeError('`in_dim` should be int, tuple[int] or list[int] '
                            'type')

        # n_batches
        if not isinstance(n_batches, int):
            raise TypeError('`n_batches should be int type')
        self._n_batches = n_batches

        # Create Tensor Network
        self._make_nodes()
        self.initialize()

    @property
    def n_features(self) -> int:
        """Returns number of nodes."""
        return self._n_features

    @property
    def boundary(self) -> Text:
        """Returns boundary condition ("obc" or "pbc")."""
        return self._boundary

    @property
    def in_dim(self) -> List[int]:
        """Returns input/physical dimension."""
        return self._in_dim

    @property
    def bond_dim(self) -> List[int]:
        """Returns bond dimension."""
        return self._bond_dim

    @property
    def n_batches(self) -> int:
        """Returns number of batch edges of the ``data`` nodes."""
        return self._n_batches

    def _make_nodes(self) -> None:
        """Creates all the nodes of the MPS."""
        if self.leaf_nodes:
            raise ValueError('Cannot create MPS nodes if the MPS already has '
                             'nodes')

        self.left_node = None
        self.right_node = None
        self.mats_env = []

        if self.boundary == 'obc':
            self.left_node = ParamNode(shape=(self.in_dim[0], self.bond_dim[0]),
                                       axes_names=('input', 'right'),
                                       name='left_node',
                                       network=self)

            for i in range(self._n_features - 2):
                node = ParamNode(shape=(self.bond_dim[i],
                                        self.in_dim[i + 1],
                                        self.bond_dim[i + 1]),
                                 axes_names=('left', 'input', 'right'),
                                 name=f'mats_env_node_({i})',
                                 network=self)
                self.mats_env.append(node)

                if i == 0:
                    self.left_node['right'] ^ self.mats_env[-1]['left']
                else:
                    self.mats_env[-2]['right'] ^ self.mats_env[-1]['left']

            self.right_node = ParamNode(shape=(self.bond_dim[-1],
                                               self.in_dim[-1]),
                                        axes_names=('left', 'input'),
                                        name='right_node',
                                        network=self)

            if self._n_features > 2:
                self.mats_env[-1]['right'] ^ self.right_node['left']
            else:
                self.left_node['right'] ^ self.right_node['left']

        else:
            for i in range(self._n_features):
                node = ParamNode(shape=(self.bond_dim[i - 1],
                                        self.in_dim[i],
                                        self.bond_dim[i]),
                                 axes_names=('left', 'input', 'right'),
                                 name=f'mats_env_node_({i})',
                                 network=self)
                self.mats_env.append(node)

                if i == 0:
                    periodic_edge = self.mats_env[-1]['left']
                else:
                    self.mats_env[-2]['right'] ^ self.mats_env[-1]['left']

                if i == self._n_features - 1:
                    self.mats_env[-1]['right'] ^ periodic_edge

    def initialize(self, std: float = 1e-9) -> None:
        """
        Initializes all the nodes as explained `here <https://arxiv.org/abs/1605.03795>`_.
        It can be overriden for custom initializations.
        """
        # Left node
        if self.left_node is not None:
            tensor = torch.randn(self.left_node.shape) * std
            aux = torch.zeros(tensor.shape[1]) * std
            aux[0] = 1.
            tensor[0, :] = aux
            self.left_node.tensor = tensor

        # Right node
        if self.right_node is not None:
            tensor = torch.randn(self.right_node.shape) * std
            aux = torch.zeros(tensor.shape[0]) * std
            aux[0] = 1.
            tensor[:, 0] = aux
            self.right_node.tensor = tensor

        # Mats env
        for node in self.mats_env:
            tensor = torch.randn(node.shape) * std
            aux = torch.eye(tensor.shape[0], tensor.shape[2])
            tensor[:, 0, :] = aux
            node.tensor = tensor

    def set_data_nodes(self) -> None:
        """
        Creates ``data`` nodes and connects each of them to the input/physical
        edge of each input node.
        """
        input_edges = []
        if self.left_node is not None:
            input_edges.append(self.left_node['input'])
        input_edges += list(map(lambda node: node['input'], self.mats_env))
        if self.right_node is not None:
            input_edges.append(self.right_node['input'])

        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self.n_batches)

        if self.mats_env:
            self.mats_env_data = list(map(lambda node: node.neighbours('input'),
                                          self.mats_env))

    def _input_contraction(self,
                           inline_input: bool = False) -> Tuple[
        Optional[List[Node]],
        Optional[List[Node]]]:
        """Contracts input data nodes with MPS nodes."""
        if inline_input:
            mats_result = list(map(lambda node: node @ node.neighbours('input'),
                                   self.mats_env))
            
            return mats_result

        else:
            if self.mats_env:
                stack = op.stack(self.mats_env)
                stack_data = op.stack(self.mats_env_data)

                stack['input'] ^ stack_data['feature']

                result = stack_data @ stack
                mats_result = op.unbind(result)
                return mats_result
            else:
                return []

    @staticmethod
    def _inline_contraction(nodes: List[Node]) -> Node:
        """Contracts sequence of MPS nodes (matrices) inline."""
        result_node = nodes[0]
        for node in nodes[1:]:
            result_node @= node
        return result_node

    def _contract_envs_inline(self, mats_env: List[Node]) -> Node:
        """Contracts the left and right environments inline."""
        if self.boundary == 'obc':
            left_node = self.left_node @ self.left_node.neighbours('input')
            right_node = self.right_node @ self.right_node.neighbours('input')
            contract_lst = [left_node] + mats_env + [right_node]
        elif len(mats_env) > 1:
            contract_lst = mats_env
        else:
            return mats_env[0] @ mats_env[0]

        return self._inline_contraction(contract_lst)

    def _aux_pairwise(self, nodes: List[Node]) -> Tuple[List[Node],
    List[Node]]:
        """Contracts a sequence of MPS nodes (matrices) pairwise."""
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
        """Contracts the left and right environments pairwise."""
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
        Contracts the whole MPS.
        
        Parameters
        ----------
        inline_input : bool
            Boolean indicating whether input ``data`` nodes should be contracted
            with the ``MPS`` nodes inline (one contraction at a time) or in a
            single stacked contraction.
        inline_mats : bool
            Boolean indicating whether the sequence of matrices (resultant
            after contracting the input ``data`` nodes) should be contracted
            inline or as a sequence of pairwise stacked contrations.

        Returns
        -------
        Node
        """
        mats_env = self._input_contraction(inline_input)

        if inline_mats:
            result = self._contract_envs_inline(mats_env)
        else:
            result = self._pairwise_contraction(mats_env)

        return result

    def canonicalize(self,
                     oc: Optional[int] = None,
                     mode: Text = 'svd',
                     rank: Optional[int] = None,
                     cum_percentage: Optional[float] = None,
                     cutoff: Optional[float] = None) -> None:
        r"""
        Turns MPS into canonical form via local SVD/QR decompositions.
        
        Parameters
        ----------
        oc : int
            Position of the orthogonality center. It should be between 0 and 
            ``n_features -1``.
        mode : {"svd", "svdr", "qr"}
            Indicates which decomposition should be used to split a node after
            contracting it. See more at :func:`svd_`, :func:`svdr_`, :func:`qr_`.
            If mode is "qr", operation :func:`qr_` will be performed on nodes at
            the left of the output node, whilst operation :func:`rq_` will be
            used for nodes at the right.
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
            
        Examples
        --------
        >>> mps = tk.models.MPS(n_features=4,
        ...                     in_dim=2,
        ...                     bond_dim=5)
        >>> mps.canonicalize(rank=3)
        >>> mps.bond_dim
        [2, 3, 2]
        """
        self.reset()

        prev_auto_stack = self._auto_stack
        self.auto_stack = False

        if oc is None:
            oc = self._n_features - 1
        elif oc >= self._n_features:
            raise ValueError(f'Orthogonality center position `oc` should be '
                             f'between 0 and {self._n_features - 1}')

        nodes = self.mats_env
        if self.boundary == 'obc':
            nodes = [self.left_node] + nodes + [self.right_node]

        for i in range(oc):
            if mode == 'svd':
                result1, result2 = nodes[i]['right'].svd_(
                    side='right',
                    rank=rank,
                    cum_percentage=cum_percentage,
                    cutoff=cutoff)
            elif mode == 'svdr':
                result1, result2 = nodes[i]['right'].svdr_(
                    side='right',
                    rank=rank,
                    cum_percentage=cum_percentage,
                    cutoff=cutoff)
            elif mode == 'qr':
                result1, result2 = nodes[i]['right'].qr_()
            else:
                raise ValueError('`mode` can only be "svd", "svdr" or "qr"')

            result1 = result1.parameterize()
            nodes[i] = result1
            nodes[i + 1] = result2

        for i in range(len(nodes) - 1, oc, -1):
            if mode == 'svd':
                result1, result2 = nodes[i]['left'].svd_(
                    side='left',
                    rank=rank,
                    cum_percentage=cum_percentage,
                    cutoff=cutoff)
            elif mode == 'svdr':
                result1, result2 = nodes[i]['left'].svdr_(
                    side='left',
                    rank=rank,
                    cum_percentage=cum_percentage,
                    cutoff=cutoff)
            elif mode == 'qr':
                result1, result2 = nodes[i]['left'].rq_()
            else:
                raise ValueError('`mode` can only be "svd", "svdr" or "qr"')

            result2 = result2.parameterize()
            nodes[i] = result2
            nodes[i - 1] = result1

        nodes[oc] = nodes[oc].parameterize()

        if self.boundary == 'obc':
            self.left_node = nodes[0]
            self.mats_env = nodes[1:-1]
            self.right_node = nodes[-1]
        else:
            self.mats_env = nodes

        bond_dim = []
        for node in nodes:
            if 'right' in node.axes_names:
                bond_dim.append(node['right'].size())
        self._bond_dim = bond_dim

        self.auto_stack = prev_auto_stack

    def _project_to_bond_dim(self,
                             nodes: List[AbstractNode],
                             bond_dim: int,
                             side: Text = 'right'):
        """Projects all nodes into a space of dimension ``bond_dim``."""
        device = nodes[0].tensor.device

        if side == 'left':
            nodes.reverse()
        elif side != 'right':
            raise ValueError('`side` can only be \'left\' or \'right\'')

        for node in nodes:
            if not node['input'].is_dangling():
                self.delete_node(node.neighbours('input'))

        line_mat_nodes = []
        in_dim_lst = []
        proj_mat_node = None
        for j in range(len(nodes)):
            in_dim_lst.append(nodes[j]['input'].size())
            if bond_dim <= torch.tensor(in_dim_lst).prod().item():
                proj_mat_node = Node(shape=(*in_dim_lst, bond_dim),
                                     axes_names=(*(['input'] * len(in_dim_lst)),
                                                 'bond_dim'),
                                     name=f'proj_mat_node_{side}',
                                     network=self)

                proj_mat_node.tensor = torch.eye(
                    torch.tensor(in_dim_lst).prod().int().item(),
                    bond_dim).view(*in_dim_lst, -1).to(device)
                for k in range(j + 1):
                    nodes[k]['input'] ^ proj_mat_node[k]

                aux_result = proj_mat_node
                for k in range(j + 1):
                    aux_result @= nodes[k]
                line_mat_nodes.append(aux_result)  # bond_dim x left x right
                break

        if proj_mat_node is None:
            bond_dim = torch.tensor(in_dim_lst).prod().int().item()
            proj_mat_node = Node(shape=(*in_dim_lst, bond_dim),
                                 axes_names=(*(['input'] * len(in_dim_lst)),
                                             'bond_dim'),
                                 name=f'proj_mat_node_{side}',
                                 network=self)

            proj_mat_node.tensor = torch.eye(
                torch.tensor(in_dim_lst).prod().int().item(),
                bond_dim).view(*in_dim_lst, -1).to(device)
            for k in range(j + 1):
                nodes[k]['input'] ^ proj_mat_node[k]

            aux_result = proj_mat_node
            for k in range(j + 1):
                aux_result @= nodes[k]
            line_mat_nodes.append(aux_result)

        k = j + 1
        while k < len(nodes):
            in_dim = nodes[k]['input'].size()
            proj_vec_node = Node(shape=(in_dim,),
                                 axes_names=('input',),
                                 name=f'proj_vec_node_{side}_({k})',
                                 network=self)

            proj_vec_node.tensor = torch.eye(in_dim, 1).squeeze().to(device)
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
        """Returns canonicalize version of the tensor at site ``idx``."""
        L = nodes[idx]  # left x input x right
        left_nodeC = None

        if idx > 0:
            # bond_dim[-1] x input  x right  /  bond_dim[-1] x input
            L = left_nodeL @ L

        L = L.tensor

        if idx < self._n_features - 1:
            bond_dim = self._bond_dim[idx]

            prod_phys_left = 1
            for i in range(idx + 1):
                prod_phys_left *= self.in_dim[i]
            bond_dim = min(bond_dim, prod_phys_left)

            prod_phys_right = 1
            for i in range(idx + 1, self._n_features):
                prod_phys_right *= self.in_dim[i]
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

        nodes = [self.left_node] + self.mats_env + [self.right_node]
        for node in nodes:
            if not node['input'].is_dangling():
                node['input'].disconnect()

        new_tensors = []
        left_nodeC = None
        for i in range(self._n_features):
            tensor, left_nodeC = self._aux_canonicalize_univocal(
                nodes=nodes,
                idx=i,
                left_nodeL=left_nodeC)
            new_tensors.append(tensor)

        for i, (node, tensor) in enumerate(zip(nodes, new_tensors)):
            if i < self._n_features - 1:
                if self._bond_dim[i] < node['right'].size():
                    node['right'].change_size(self._bond_dim[i])
            node.tensor = tensor

            if not node['input'].is_dangling():
                self.delete_node(node.neighbours('input'))
        self.reset()

        for node, data_node in zip(nodes, self._data_nodes.values()):
            node['input'] ^ data_node['feature']

        self.auto_stack = prev_auto_stack


class UMPS(TensorNetwork):
    """
    Class for Uniform (translationally invariant) Matrix Product States where
    all nodes are input nodes. It is the uniform version of :class:`MPS`, that
    is, all nodes share the same tensor. Thus this class cannot have different
    input or bond dimensions for each node, and boundary conditions are
    always periodic (``"pbc"``).
    
    A ``UMPS`` is formed by the following nodes:
      
    * ``mats_env``: Environment of `matrix` nodes that. These nodes have axes
      ``("left", "input", "right")``.

    Parameters
    ----------
    n_features : int
        Number of input nodes.
    in_dim : int
        Input dimension. Equivalent to the physical dimension.
    bond_dim : int
        Bond dimension.
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
        
    Examples
    --------
    >>> mps = tk.models.UMPS(n_features=4,
    ...                      in_dim=2,
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
                 in_dim: int,
                 bond_dim: int,
                 n_batches: int = 1) -> None:

        super().__init__(name='mps')

        # n_features
        if n_features < 1:
            raise ValueError('`n_features` cannot be lower than 1')
        self._n_features = n_features

        # in_dim
        if isinstance(in_dim, int):
            self._in_dim = in_dim
        else:
            raise TypeError('`in_dim` should be int type')

        # bond_dim
        if isinstance(bond_dim, int):
            self._bond_dim = bond_dim
        else:
            raise TypeError('`bond_dim` should be int type')

        # n_batches
        if not isinstance(n_batches, int):
            raise TypeError('`n_batches should be int type')
        self._n_batches = n_batches

        # Create Tensor Network
        self._make_nodes()
        self.initialize()

    @property
    def n_features(self) -> int:
        """Returns number of nodes."""
        return self._n_features

    @property
    def in_dim(self) -> int:
        """Returns input/physical dimension."""
        return self._in_dim

    @property
    def bond_dim(self) -> int:
        """Returns bond dimension."""
        return self._bond_dim

    @property
    def n_batches(self) -> int:
        """Returns number of batch edges of the ``data`` nodes."""
        return self._n_batches

    def _make_nodes(self) -> None:
        """Creates all the nodes of the MPS."""
        if self._leaf_nodes:
            raise ValueError('Cannot create MPS nodes if the MPS already has '
                             'nodes')

        self.left_node = None
        self.right_node = None
        self.mats_env = []

        for i in range(self._n_features):
            node = ParamNode(shape=(self.bond_dim, self.in_dim, self.bond_dim),
                             axes_names=('left', 'input', 'right'),
                             name=f'mats_env_node_({i})',
                             network=self)
            self.mats_env.append(node)

            if i == 0:
                periodic_edge = self.mats_env[-1]['left']
            else:
                self.mats_env[-2]['right'] ^ self.mats_env[-1]['left']

            if i == self._n_features - 1:
                self.mats_env[-1]['right'] ^ periodic_edge

        # Virtual node
        uniform_memory = ParamNode(shape=(self.bond_dim, self.in_dim, self.bond_dim),
                                   axes_names=('left', 'input', 'right'),
                                   name='virtual_uniform',
                                   network=self,
                                   virtual=True)
        self.uniform_memory = uniform_memory

    def initialize(self, std: float = 1e-9) -> None:
        """
        Initializes output and uniform nodes as explained `here
        <https://arxiv.org/abs/1605.03795>`_.
        It can be overriden for custom initializations.
        """
        # Virtual node
        tensor = torch.randn(self.uniform_memory.shape) * std
        random_eye = torch.randn(tensor.shape[0], tensor.shape[2]) * std
        random_eye = random_eye + torch.eye(tensor.shape[0], tensor.shape[2])
        tensor[:, 0, :] = random_eye

        self.uniform_memory.tensor = tensor

        for node in self.mats_env:
            node.set_tensor_from(self.uniform_memory)

    def set_data_nodes(self) -> None:
        """
        Creates ``data`` nodes and connects each of them to the input/physical
        edge of each input node.
        """
        input_edges = []
        if self.left_node is not None:
            input_edges.append(self.left_node['input'])
        input_edges += list(map(lambda node: node['input'], self.mats_env))
        if self.right_node is not None:
            input_edges.append(self.right_node['input'])

        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self._n_batches)

        if self.mats_env:
            self.mats_env_data = list(map(lambda node: node.neighbours('input'),
                                          self.mats_env))

    def _input_contraction(self,
                           inline_input: bool = False) -> Tuple[
        Optional[List[Node]],
        Optional[List[Node]]]:
        """Contracts input data nodes with MPS nodes."""
        if inline_input:
            mats_result = []
            for node in self.mats_env:
                mats_result.append(node @ node.neighbours('input'))
            return mats_result

        else:
            if self.mats_env:
                stack = op.stack(self.mats_env)
                stack_data = op.stack(self.mats_env_data)

                stack['input'] ^ stack_data['feature']

                result = stack_data @ stack
                mats_result = op.unbind(result)
                return mats_result
            else:
                return []

    @staticmethod
    def _inline_contraction(nodes: List[Node]) -> Node:
        """Contracts sequence of MPS nodes (matrices) inline."""
        result_node = nodes[0]
        for node in nodes[1:]:
            result_node @= node
        return result_node

    def _contract_envs_inline(self, mats_env: List[Node]) -> Node:
        """Contracts the left and right environments inline."""
        if len(mats_env) > 1:
            contract_lst = mats_env
        else:
            return mats_env[0] @ mats_env[0]

        return self._inline_contraction(contract_lst)

    def _aux_pairwise(self, nodes: List[Node]) -> Tuple[List[Node],
    List[Node]]:
        """Contracts a sequence of MPS nodes (matrices) pairwise."""
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
        """Contracts the left and right environments pairwise."""
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
        Contracts the whole MPS.
        
        Parameters
        ----------
        inline_input : bool
            Boolean indicating whether input ``data`` nodes should be contracted
            with the ``MPS`` nodes inline (one contraction at a time) or in a
            single stacked contraction.
        inline_mats : bool
            Boolean indicating whether the sequence of matrices (resultant
            after contracting the input ``data`` nodes) should be contracted
            inline or as a sequence of pairwise stacked contrations.

        Returns
        -------
        Node
        """
        mats_env = self._input_contraction(inline_input)

        if inline_mats:
            result = self._contract_envs_inline(mats_env)
        else:
            result = self._pairwise_contraction(mats_env)

        return result


class ConvMPS(MPS):
    """
    Class for Matrix Product States, where all nodes are input nodes, and where
    the input data is a batch of images. It is the convolutional version of
    :class:`MPS`.
    
    Input data as well as initialization parameters are described in `torch.nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    in_channels : int
        Input channels. Same as ``in_dim`` in :class:`MPS`.
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
        
    Examples
    --------
    >>> conv_mps = tk.models.ConvMPS(in_channels=2,
    ...                              bond_dim=5,
    ...                              kernel_size=2)
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_mps(data)
    >>> print(result.shape)
    torch.Size([20, 1, 1])
    """

    def __init__(self,
                 in_channels: int,
                 bond_dim: Union[int, Sequence[int]],
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 boundary: Text = 'obc'):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size, Sequence):
            raise TypeError('`kernel_size` must be int or Sequence')

        if isinstance(stride, int):
            stride = (stride, stride)
        elif not isinstance(stride, Sequence):
            raise TypeError('`stride` must be int or Sequence')

        if isinstance(padding, int):
            padding = (padding, padding)
        elif not isinstance(padding, Sequence):
            raise TypeError('`padding` must be int or Sequence')

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        elif not isinstance(dilation, Sequence):
            raise TypeError('`dilation` must be int or Sequence')

        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation

        super().__init__(n_features=kernel_size[0] * kernel_size[1],
                         in_dim=in_channels,
                         bond_dim=bond_dim,
                         boundary=boundary,
                         n_batches=2)

        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)

    @property
    def in_channels(self) -> int:
        """Returns ``in_channels``. Same as ``in_dim`` in :class:`MPS`."""
        return self._in_channels

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """
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

    def forward(self, image, mode='flat', *args, **kwargs):
        r"""
        Overrides ``torch.nn.Module``'s forward to compute a convolution on the
        input image.
        
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

        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows

        h_in = image.shape[2]
        w_in = image.shape[3]

        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] *
                     (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] *
                     (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x height_out x width_out

        return result


class ConvUMPS(UMPS):
    """
    Class for Uniform Matrix Product States, where all nodes are input nodes,
    and where the input data is a batch of images. It is the convolutional
    version of :class:`UMPS`. This class cannot have different bond dimensions
    for each site and boundary conditions are always periodic.
    
    Input data as well as initialization parameters are described in `torch.nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    in_channels : int
        Input channels. Same as ``in_dim`` in :class:`UMPS`.
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
    
    Examples
    --------
    >>> conv_mps = tk.models.ConvMPS(in_channels=2,
    ...                              bond_dim=5,
    ...                              kernel_size=2)
    >>> for node in mps.mats_env:
    ...     assert node.tensor_address() == 'virtual_uniform'
    ...
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_mps(data)
    >>> print(result.shape)
    torch.Size([20, 1, 1])
    """

    def __init__(self,
                 in_channels: int,
                 bond_dim: int,
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1):

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif not isinstance(kernel_size, Sequence):
            raise TypeError('`kernel_size` must be int or Sequence')

        if isinstance(stride, int):
            stride = (stride, stride)
        elif not isinstance(stride, Sequence):
            raise TypeError('`stride` must be int or Sequence')

        if isinstance(padding, int):
            padding = (padding, padding)
        elif not isinstance(padding, Sequence):
            raise TypeError('`padding` must be int or Sequence')

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        elif not isinstance(dilation, Sequence):
            raise TypeError('`dilation` must be int or Sequence')

        self._in_channels = in_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation

        super().__init__(n_features=kernel_size[0] * kernel_size[1],
                         in_dim=in_channels,
                         bond_dim=bond_dim,
                         n_batches=2)

        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)

    @property
    def in_channels(self) -> int:
        """Returns ``in_channels``. Same as ``in_dim`` in :class:`UMPS`."""
        return self._in_channels

    @property
    def kernel_size(self) -> Tuple[int, int]:
        """
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

    def forward(self, image, mode='flat', *args, **kwargs):
        r"""
        Overrides ``torch.nn.Module``'s forward to compute a convolution on the
        input image.
        
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
            Arguments that might be used in :meth:`~UMPS.contract`.
        kwargs :
            Keyword arguments that might be used in :meth:`~UMPS.contract`,
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

        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows

        h_in = image.shape[2]
        w_in = image.shape[3]

        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] *
                     (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] *
                     (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x height_out x width_out

        return result
