"""
This script contains:
    *MPS
    *UMPS
    *ConvMPS
    *ConvUMPS
"""

from typing import (List, Optional, Sequence,
                    Text, Tuple, Union)

import torch
import torch.nn as nn

from tensorkrowch.components import AbstractNode, Node, ParamNode
from tensorkrowch.components import TensorNetwork
import tensorkrowch.operations as op


class MPS(TensorNetwork):
    """
    Class for Matrix Product States, where all nodes are input nodes, that is,
    they are all connected to ``data`` nodes that will store the input data
    tensor(s). When contracting the MPS with new input data, the result will
    be a just a number.
    
    If the physical dimensions of all the input nodes are equal, the input data
    tensor can be passed as a single tensor. Otherwise, it would have to be
    passed as a list of tensors with different sizes.

    Parameters
    ----------
    n_sites : int
        Number of sites.
    d_phys : int, list[int] or tuple[int]
        Physical dimension(s). If given as a sequence, its length should be
        equal to ``n_sites``.
    d_bond : int, list[int] or tuple[int]
        Bond dimension(s). If given as a sequence, its length should be equal
        to ``n_sites`` (if ``boundary = "pbc"``) or ``n_sites - 1`` (if
        ``boundary = "obc"``). The i-th bond dimension is always the dimension
        of the right edge of the i-th node.
    boundary : {'obc', 'pbc'}
        String indicating whether periodic or open boundary conditions should
        be used.
    param_bond : bool
        Boolean indicating whether bond edges should be :class:`ParamEdge`.
    num_batches : int
        Number of batch edges of input data nodes. Usually ``num_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``num_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
        
    Examples
    --------
    ``MPS`` with same physical dimensions:
    
    >>> mps = tk.MPS(n_sites=5,
    ...              d_phys=2,
    ...              d_bond=5)
    >>> data = torch.ones(5, 20, 2) # n_features x batch_size x feature_size
    >>> result = mps(data)
    >>> print(result.shape)
    torch.Size([20])
    
    ``MPS`` with different physical dimensions:
    
    >>> mps = tk.MPS(n_sites=5,
    ...              d_phys=list(range(2, 7)),
    ...              d_bond=5)
    >>> data = [torch.ones(20, i)
    ...         for i in range(2, 7)] # n_features * [batch_size x feature_size]
    >>> result = mps(data)
    >>> print(result.shape)
    torch.Size([20])
    """

    def __init__(self,
                 n_sites: int,
                 d_phys: Union[int, Sequence[int]],
                 d_bond: Union[int, Sequence[int]],
                 boundary: Text = 'obc',
                 param_bond: bool = False,
                 num_batches: int = 1) -> None:

        super().__init__(name='mps')

        # boundary
        if boundary == 'obc':
            if n_sites < 2:
                raise ValueError('If `boundary` is "obc", at least '
                                 'there has to be 2 sites')
        elif boundary == 'pbc':
            if n_sites < 1:
                raise ValueError('If `boundary` is "pbc", at least '
                                 'there has to be one site')
        else:
            raise ValueError('`boundary` should be one of "obc" or "pbc"')
        self._n_sites = n_sites
        self._boundary = boundary

        # d_phys
        if isinstance(d_phys, (list, tuple)):
            if len(d_phys) != n_sites:
                raise ValueError('If `d_phys` is given as a sequence of int, '
                                 'its length should be equal to `n_sites`')
            self._d_phys = list(d_phys)
        elif isinstance(d_phys, int):
            self._d_phys = [d_phys] * n_sites
        else:
            raise TypeError('`d_phys` should be `int` type or a list/tuple '
                            'of ints')

        # d_bond
        if isinstance(d_bond, (list, tuple)):
            if boundary == 'obc':
                if len(d_bond) != n_sites - 1:
                    raise ValueError('If `d_bond` is given as a sequence of int,'
                                     ' and `boundary` is "obc", its length '
                                     'should be equal to `n_sites` - 1')
            elif boundary == 'pbc':
                if len(d_bond) != n_sites:
                    raise ValueError('If `d_bond` is given as a sequence of int,'
                                     ' and `boundary` is "pbc", its length '
                                     'should be equal to `n_sites`')
            self._d_bond = list(d_bond)
        elif isinstance(d_bond, int):
            if boundary == 'obc':
                self._d_bond = [d_bond] * (n_sites - 1)
            elif boundary == 'pbc':
                self._d_bond = [d_bond] * n_sites
        else:
            raise TypeError('`d_bond` should be `int` type or a list/tuple '
                            'of ints')

        # param_bond
        self._param_bond = param_bond

        self._make_nodes()
        self.param_bond(set_param=param_bond)
        self.initialize()
        
        self._num_batches = num_batches

    @property
    def n_sites(self) -> int:
        """Returns number of nodes."""
        return self._n_sites

    @property
    def boundary(self) -> Text:
        """Returns boundary condition ("obc" or "pbc")."""
        return self._boundary

    @property
    def d_phys(self) -> List[int]:
        """Returns physical dimension."""
        return self._d_phys

    @property
    def d_bond(self) -> List[int]:
        """Returns bond dimension."""
        return self._d_bond

    def param_bond(self, set_param: Optional[bool] = None) -> Optional[bool]:
        """
        Returns ``param_bond`` attribute or changes it if ``set_param`` is
        provided.

        Parameters
        ----------
        set_param : bool, optional
            Boolean indicating whether edges have to be parameterized (``True``)
            or de-parameterized (``False``).
        """
        if set_param is None:
            return self._param_bond
        else:
            for node in self.leaf_nodes.values():
                if 'left' in node.axes_names:
                    node['left'].parameterize(set_param=set_param)
                if 'right' in node.axes_names:
                    node['right'].parameterize(set_param=set_param)
            self._param_bond = set_param

    def _make_nodes(self) -> None:
        """Creates all the nodes of the MPS."""
        if self._leaf_nodes:
            raise ValueError('Cannot create MPS nodes if the MPS already has '
                             'nodes')

        self.left_node = None
        self.right_node = None
        self.mats_env = []
        
        if self.boundary == 'obc':
            self.left_node = ParamNode(shape=(self.d_phys[0], self.d_bond[0]),
                                       axes_names=('input', 'right'),
                                       name='left_node',
                                       network=self)
            
            for i in range(self._n_sites - 2):
                node = ParamNode(shape=(self.d_bond[i],
                                        self.d_phys[i + 1],
                                        self.d_bond[i + 1]),
                                 axes_names=('left', 'input', 'right'),
                                 name=f'mats_env_node_({i})',
                                 network=self)
                self.mats_env.append(node)
                
                if i == 0:
                    self.left_node['right'] ^ self.mats_env[-1]['left']
                else:
                    self.mats_env[-2]['right'] ^ self.mats_env[-1]['left']
                    
                
            self.right_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[-1]),
                                        axes_names=('left', 'input'),
                                        name='right_node',
                                        network=self)
            
            if self._n_sites > 2:
                self.mats_env[-1]['right'] ^ self.right_node['left']
            else:
                self.left_node['right'] ^ self.right_node['left']
                
        else:
            for i in range(self._n_sites):
                node = ParamNode(shape=(self.d_bond[i - 1],
                                        self.d_phys[i],
                                        self.d_bond[i]),
                                 axes_names=('left', 'input', 'right'),
                                 name=f'mats_env_node_({i})',
                                 network=self)
                self.mats_env.append(node)
                
                if i == 0:
                    periodic_edge = self.mats_env[-1]['left']
                else:
                    self.mats_env[-2]['right'] ^ self.mats_env[-1]['left']
                
                if i == self._n_sites - 1:
                    self.mats_env[-1]['right'] ^ periodic_edge

    def initialize(self, std: float = 1e-9) -> None:
        """Initializes all the nodes."""
        # Left node
        if self.left_node is not None:
            tensor = torch.randn(self.left_node.shape) * std
            if self.boundary == 'obc':
                aux = torch.zeros(tensor.shape[1]) * std
                aux[0] = 1.
                tensor[0, :] = aux
            else:
                aux = torch.eye(self.left_node.shape[0],
                                self.left_node.shape[2])
                tensor[:, 0, :] = aux
            self.left_node.tensor = tensor
        
        # Right node
        if self.right_node is not None:
            tensor = torch.randn(self.right_node.shape) * std
            if self.boundary == 'obc':
                aux = torch.zeros(tensor.shape[0]) * std
                aux[0] = 1.
                tensor[:, 0] = aux
            else:
                aux = torch.eye(self.right_node.shape[0],
                                self.right_node.shape[2])
                tensor[:, 0, :] = aux
            self.right_node.tensor = tensor
        
        # Mats env
        for node in self.mats_env:
            tensor = torch.randn(node.shape) * std
            aux = torch.eye(tensor.shape[0], tensor.shape[2])
            tensor[:, 0, :] = aux
            node.tensor = tensor

    def set_data_nodes(self) -> None:
        """
        Creates data nodes and connects each of them to the physical edge of
        an input node.
        """
        input_edges = []
        if self.left_node is not None:
            input_edges.append(self.left_node['input'])
        input_edges += list(map(lambda node: node['input'], self.mats_env))
        if self.right_node is not None:
            input_edges.append(self.right_node['input'])
            
        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self._num_batches)
        
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
            Boolean indicating whether input data nodes should be contracted
            inline (one contraction at a time) or in a single stacked
            contraction.
        inline_mats : bool
            Boolean indicating whether the sequence of matrices (resultant
            after contracting the input data nodes) should be contracted inline
            or as a sequence of pairwise stacked contrations.

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
            ``n_sites -1``.
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
        """
        prev_automemory = self._automemory
        self.automemory = False
        
        if oc is None:
            oc = self._n_sites - 1
        elif oc >= self._n_sites:
            raise ValueError(f'Orthogonality center position `oc` should be '
                             f'between 0 and {self._n_sites - 1}')
        
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
            
        self.param_bond(set_param=self._param_bond)
        
        d_bond = []
        for node in nodes:
            if 'right' in node.axes_names:
                d_bond.append(node['right'].size())  # TODO: _size?
        self._d_bond = d_bond
        
        self.automemory = prev_automemory
        
    def _project_to_d_bond(self,
                           nodes: List[AbstractNode],
                           d_bond: int,
                           side: Text = 'right'):
        """Projects all nodes into a space of dimension ``d_bond``."""
        device = nodes[0].tensor.device
        
        if side == 'left':
            nodes.reverse()
        elif side != 'right':
            raise ValueError('`side` can only be \'left\' or \'right\'')
        
        for node in nodes:
            if not node['input'].is_dangling():
                self.delete_node(node.neighbours('input'))
        
        line_mat_nodes = []
        d_phys_lst = []
        proj_mat_node = None
        for j in range(len(nodes)):
            d_phys_lst.append(nodes[j]['input'].size())
            if d_bond <= torch.tensor(d_phys_lst).prod().item():
                proj_mat_node = Node(shape=(*d_phys_lst,d_bond),
                                     axes_names=(*(['input'] * len(d_phys_lst)),
                                                 'd_bond'),
                                     name=f'proj_mat_node_{side}',
                                     network=self)
                
                proj_mat_node.tensor = torch.eye(
                    torch.tensor(d_phys_lst).prod().int().item(),
                    d_bond).view(*d_phys_lst, -1).to(device)
                for k in range(j + 1):
                    nodes[k]['input'] ^ proj_mat_node[k]
                    
                aux_result = proj_mat_node
                for k in range(j + 1):
                    aux_result @= nodes[k]
                line_mat_nodes.append(aux_result)  # d_bond x left x right
                break
            
        if proj_mat_node is None:
            d_bond = torch.tensor(d_phys_lst).prod().int().item()
            proj_mat_node = Node(shape=(*d_phys_lst, d_bond),
                                 axes_names=(*(['input'] * len(d_phys_lst)),
                                             'd_bond'),
                                 name=f'proj_mat_node_{side}',
                                 network=self)
            
            proj_mat_node.tensor = torch.eye(
                torch.tensor(d_phys_lst).prod().int().item(),
                d_bond).view(*d_phys_lst, -1).to(device)
            for k in range(j + 1):
                nodes[k]['input'] ^ proj_mat_node[k]
                
            aux_result = proj_mat_node
            for k in range(j + 1):
                aux_result @= nodes[k]
            line_mat_nodes.append(aux_result)
            
        k = j + 1
        while k < len(nodes):
            d_phys = nodes[k]['input'].size()
            proj_vec_node = Node(shape=(d_phys,),
                                 axes_names=('input',),
                                 name=f'proj_vec_node_{side}_({k})',
                                 network=self)
            
            proj_vec_node.tensor = torch.eye(d_phys, 1).squeeze().to(device)
            nodes[k]['input'] ^ proj_vec_node['input']
            line_mat_nodes.append(proj_vec_node @ nodes[k])
            
            k += 1
        
        line_mat_nodes.reverse()
        result = line_mat_nodes[0]
        for node in line_mat_nodes[1:]:
            result @= node
            
        return result  # d_bond x left/right
    
    def _aux_canonicalize_univocal(self,
                                   nodes: List[AbstractNode],
                                   idx: int,
                                   left_nodeL: AbstractNode):
        """Returns canonicalize version of the tensor at site ``idx``."""
        L = nodes[idx]  # left x input x right
        left_nodeC = None
        
        if idx > 0:
            # d_bond[-1] x input  x right  /  d_bond[-1] x input
            L = left_nodeL @ L
        
        L = L.tensor
        
        if idx < self._n_sites - 1:
            d_bond = self._d_bond[idx]
            
            prod_phys_left = 1
            for i in range(idx + 1):
                prod_phys_left *= self._d_phys[i]
            d_bond = min(d_bond, prod_phys_left)
            
            prod_phys_right = 1
            for i in range(idx + 1, self._n_sites):
                prod_phys_right *= self._d_phys[i]
            d_bond = min(d_bond, prod_phys_right)
            
            if d_bond < self._d_bond[idx]:
                self._d_bond[idx] = d_bond
            
            left_nodeC = self._project_to_d_bond(nodes=nodes[:idx + 1],
                                                 d_bond=d_bond,
                                                 side='left')  # d_bond x right
            right_node = self._project_to_d_bond(nodes=nodes[idx + 1:],
                                                 d_bond=d_bond,
                                                 side='right')  # d_bond x left
            
            C = left_nodeC @ right_node  # d_bond x d_bond
            C = torch.linalg.inv(C.tensor)
            
            if idx == 0:
                L @= right_node.tensor.t()  # input x d_bond
                L @= C
            else:
                shape_L = L.shape
                # (d_bond[-1] * input) x d_bond
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
            
        prev_automemory = self._automemory
        self.automemory = False
        
        nodes = [self.left_node] + self.mats_env + [self.right_node]
        for node in nodes:
            if not node['input'].is_dangling():
                node['input'].disconnect()
        
        new_tensors = []
        left_nodeC = None
        for i in range(self._n_sites):
            tensor, left_nodeC = self._aux_canonicalize_univocal(
                nodes=nodes,
                idx=i,
                left_nodeL=left_nodeC)
            new_tensors.append(tensor)
        
        for i, (node, tensor) in enumerate(zip(nodes, new_tensors)):
            if i < self._n_sites - 1:
                if self._d_bond[i] < node['right'].size():
                    node['right'].change_size(self._d_bond[i])
            node.tensor = tensor
            
            if not node['input'].is_dangling():
                self.delete_node(node.neighbours('input'))
        self.reset()
            
        for node, data_node in zip(nodes, self._data_nodes.values()):
            node['input'] ^ data_node['feature']
            
        self.automemory = prev_automemory


class UMPS(TensorNetwork):
    """
    Class for Uniform (translationally invariant) Matrix Product States where
    all nodes are input nodes. It is the uniform version of :class:`MPS`, that
    is, all nodes share the same tensor. Thus this class cannot have different
    physical or bond dimensions for each site, and boundary conditions are
    always periodic.

    Parameters
    ----------
    n_sites : int
        Number of sites.
    d_phys : int
        Physical dimension.
    d_bond : int
        Bond dimension.
    param_bond : bool
        Boolean indicating whether bond edges should be :class:`ParamEdge`.
    num_batches : int
        Number of batch edges of input data nodes. Usually ``num_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``num_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    """

    def __init__(self,
                 n_sites: int,
                 d_phys: int,
                 d_bond: int,
                 param_bond: bool = False,
                 num_batches: int = 1) -> None:

        super().__init__(name='mps')

        # boundary
        if n_sites < 1:
            raise ValueError('If `boundary` is "pbc", at least '
                             'there has to be one site')
        self._n_sites = n_sites

        # d_phys
        if isinstance(d_phys, int):
            self._d_phys = d_phys
        else:
            raise TypeError('`d_phys` should be `int` type')

        # d_bond
        if isinstance(d_bond, int):
            self._d_bond = d_bond
        else:
            raise TypeError('`d_bond` should be `int` type')

        # param_bond
        self._param_bond = param_bond

        self._make_nodes()
        self.param_bond(set_param=param_bond)
        self.initialize()
        
        self._num_batches = num_batches

    @property
    def n_sites(self) -> int:
        """Returns number of nodes."""
        return self._n_sites

    @property
    def d_phys(self) -> int:
        """Returns physical dimension."""
        return self._d_phys

    @property
    def d_bond(self) -> int:
        """Returns bond dimension."""
        return self._d_bond

    def param_bond(self, set_param: Optional[bool] = None) -> Optional[bool]:
        """
        Returns ``param_bond`` attribute or changes it if ``set_param`` is
        provided.

        Parameters
        ----------
        set_param : bool, optional
            Boolean indicating whether edges have to be parameterized (``True``)
            or de-parameterized (``False``).
        """
        if set_param is None:
            return self._param_bond
        else:
            for node in self.leaf_nodes.values():
                if 'left' in node.axes_names:
                    node['left'].parameterize(set_param=set_param)
                if 'right' in node.axes_names:
                    node['right'].parameterize(set_param=set_param)
            self._param_bond = set_param

    def _make_nodes(self) -> None:
        """Creates all the nodes of the MPS."""
        if self._leaf_nodes:
            raise ValueError('Cannot create MPS nodes if the MPS already has '
                             'nodes')
        
        self.left_node = None
        self.right_node = None
        self.mats_env = []
        
        for i in range(self._n_sites):
            node = ParamNode(shape=(self.d_bond, self.d_phys, self.d_bond),
                             axes_names=('left', 'input', 'right'),
                             name=f'mats_env_node_({i})',
                             network=self)
            self.mats_env.append(node)
            
            if i == 0:
                periodic_edge = self.mats_env[-1]['left']
            else:
                self.mats_env[-2]['right'] ^ self.mats_env[-1]['left']
            
            if i == self._n_sites - 1:
                self.mats_env[-1]['right'] ^ periodic_edge
                        
        # Virtual node
        uniform_memory = ParamNode(shape=(self.d_bond, self.d_phys, self.d_bond),
                                   axes_names=('left', 'input', 'right'),
                                   name='virtual_uniform',
                                   network=self,
                                   virtual=True)
        self.uniform_memory = uniform_memory
        
        for edge in uniform_memory._edges:
            self._remove_edge(edge)

    def initialize(self, std: float = 1e-9) -> None:
        """Initializes output and uniform nodes."""
        # Virtual node
        tensor = torch.randn(self.uniform_memory.shape) * std
        random_eye = torch.randn(tensor.shape[0], tensor.shape[2]) * std
        random_eye  = random_eye + torch.eye(tensor.shape[0], tensor.shape[2])
        tensor[:, 0, :] = random_eye
        
        self.uniform_memory._unrestricted_set_tensor(tensor)
    
        for node in self.mats_env:
            del self._memory_nodes[node._tensor_info['address']]
            node._tensor_info['address'] = None
            node._tensor_info['node_ref'] = self.uniform_memory
            node._tensor_info['full'] = True
            node._tensor_info['stack_idx'] = None
            node._tensor_info['index'] = None

    def set_data_nodes(self) -> None:
        """
        Creates data nodes and connects each of them to the physical edge of
        an input node.
        """
        input_edges = []
        if self.left_node is not None:
            input_edges.append(self.left_node['input'])
        input_edges += list(map(lambda node: node['input'], self.mats_env))
        if self.right_node is not None:
            input_edges.append(self.right_node['input'])
            
        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self._num_batches)
        
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
            Boolean indicating whether input data nodes should be contracted
            inline (one contraction at a time) or in a single stacked
            contraction.
        inline_mats : bool
            Boolean indicating whether the sequence of matrices (resultant
            after contracting the input data nodes) should be contracted inline
            or as a sequence of pairwise stacked contrations.

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
    
    Input data as well as initialization parameters are described in `nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    in_channels : int
        Input channels. Same as ``d_phys`` in :class:`MPS`.
    d_bond : int, list[int] or tuple[int]
        Bond dimension(s). If given as a sequence, its length should be equal
        to :math:`kernel\_size_0 \cdot kernel\_size_1` (if ``boundary = "pbc"``)
        or :math:`kernel\_size_0 \cdot kernel\_size_1 - 1` (if
        ``boundary = "obc"``). The i-th bond dimension is always the dimension
        of the right edge of the i-th node.
    kernel_size : int, list[int] or tuple[int]
        Kernel size used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    stride : int
        Stride used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
    padding : int
        Padding used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    dilation : int
        Dilation used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    boundary : {'obc', 'pbc'}
        String indicating whether periodic or open boundary conditions should
        be used.
    param_bond : bool
        Boolean indicating whether bond edges should be :class:`ParamEdge`.
        
    Examples
    --------
    >>> conv_mps = tk.ConvMPS(in_channels=2,
    ...                       d_bond=5,
    ...                       kernel_size=2)
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_mps(data)
    >>> print(result.shape)
    torch.Size([20, 1, 1])
    """
    
    def __init__(self,
                 in_channels: int,
                 d_bond: Union[int, Sequence[int]],
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 boundary: Text = 'obc',
                 param_bond: bool = False):
        
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
        
        super().__init__(n_sites=kernel_size[0] * kernel_size[1],
                         d_phys=in_channels,
                         d_bond=d_bond,
                         boundary=boundary,
                         param_bond=param_bond,
                         num_batches=2)
        
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)
        
    @property
    def in_channels(self) -> int:
        """Returns ``in_channels``. Same as ``d_phys`` in :class:`MPS`."""
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
        Returns stride used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._stride
    
    @property
    def padding(self) -> Tuple[int, int]:
        """
        Returns padding used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._padding
    
    @property
    def dilation(self) -> Tuple[int, int]:
        """
        Returns dilation used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._dilation
    
    def forward(self, image, mode='flat', *args, **kwargs):
        r"""
        Overrides ``nn.Module``'s forward to compute a convolution on the input
        image.
        
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
        
        patches = patches.permute(3, 0, 1, 2)
        # nb_pixels x batch_size x nb_windows x in_channels
        
        if mode == 'snake':
            new_patches = patches[:self._kernel_size[1]]
            for i in range(1, self._kernel_size[0]):
                if i % 2 == 0:
                    aux = patches[(i * self._kernel_size[1]):
                                  ((i + 1) * self._kernel_size[1])]
                else:
                    aux = patches[
                        (i * self._kernel_size[1]):
                        ((i + 1) * self._kernel_size[1])].flip(dims=[0])
                new_patches = torch.cat([new_patches, aux], dim=0)
                
            patches = new_patches
            
        elif mode != 'flat':
            raise ValueError('`mode` can only be "flat" or "snake"')
        
        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows
        
        h_in = image.shape[2]
        w_in = image.shape[3]
        
        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] * \
            (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] * \
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
    
    Input data as well as initialization parameters are described in `nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    in_channels : int
        Input channels. Same as ``d_phys`` in :class:`UMPS`.
    d_bond : int
        Bond dimension.
    kernel_size : int, list[int] or tuple[int]
        Kernel size used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    stride : int
        Stride used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
    padding : int
        Padding used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    dilation : int
        Dilation used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    param_bond : bool
        Boolean indicating whether bond edges should be :class:`ParamEdge`.
    """
    
    def __init__(self,
                    in_channels: int,
                    d_bond: int,
                    kernel_size: Union[int, Sequence],
                    stride: int = 1,
                    padding: int = 0,
                    dilation: int = 1,
                    param_bond: bool = False):
        
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
        
        super().__init__(n_sites=kernel_size[0] * kernel_size[1],
                         d_phys=in_channels,
                         d_bond=d_bond,
                         param_bond=param_bond,
                         num_batches=2)
        
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)
        
    @property
    def in_channels(self) -> int:
        """Returns ``in_channels``. Same as ``d_phys`` in :class:`UMPS`."""
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
        Returns stride used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._stride
    
    @property
    def padding(self) -> Tuple[int, int]:
        """
        Returns padding used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._padding
    
    @property
    def dilation(self) -> Tuple[int, int]:
        """
        Returns dilation used in `nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._dilation
    
    def forward(self, image, mode='flat', *args, **kwargs):
        r"""
        Overrides ``nn.Module``'s forward to compute a convolution on the input
        image.
        
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
        
        patches = patches.permute(3, 0, 1, 2)
        # nb_pixels x batch_size x nb_windows x in_channels
        
        if mode == 'snake':
            new_patches = patches[:self._kernel_size[1]]
            for i in range(1, self._kernel_size[0]):
                if i % 2 == 0:
                    aux = patches[(i * self._kernel_size[1]):
                                  ((i + 1) * self._kernel_size[1])]
                else:
                    aux = patches[
                        (i * self._kernel_size[1]):
                        ((i + 1) * self._kernel_size[1])].flip(dims=[0])
                new_patches = torch.cat([new_patches, aux], dim=0)
                
            patches = new_patches
            
        elif mode != 'flat':
            raise ValueError('`mode` can only be "flat" or "snake"')
        
        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows
        
        h_in = image.shape[2]
        w_in = image.shape[3]
        
        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] * \
            (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] * \
                    (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        
        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x height_out x width_out
        
        return result
