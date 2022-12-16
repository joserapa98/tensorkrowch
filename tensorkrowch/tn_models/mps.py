"""
MPSLayer + UMPSLayer classes
"""

from typing import (Union, Optional, Sequence,
                    Text, List, Tuple)

import torch
from torch.nn.functional import pad
import torch.nn as nn

from tensorkrowch.network_components import (AbstractNode, Node, ParamNode,
                                         AbstractEdge)
from tensorkrowch.network_components import TensorNetwork

from tensorkrowch.node_operations import einsum, stacked_einsum

import tensorkrowch as tk

import opt_einsum
import math

import time
from torchviz import make_dot

PRINT_MODE = False


class MPS(TensorNetwork):

    def __init__(self,
                 n_sites: int,
                 d_phys: Union[int, Sequence[int]],
                 d_bond: Union[int, Sequence[int]],
                 boundary: Text = 'obc',
                 param_bond: bool = False,
                 num_batches: int = 1,
                 inline_input: bool = False,
                 inline_mats: bool = False) -> None:
        """
        Create an MPS module.

        Parameters
        ----------
        n_sites: number of sites, including the input and output_node sites
        d_phys: physic dimension
        d_bond: bond dimension. If given as a sequence, the i-th bond
            dimension is always the dimension of the right edge of th i-th node
        boundary: string indicating whether we are using periodic or open
            boundary conditions
        param_bond: boolean indicating whether bond edges should be parametric
        num_batches: number of batch edges of input data
        inline_input: boolean indicating whether input should be contracted
            inline or in a single stacked contraction
        inline_mats: boolean indicating whether sequence of matrices
            should be contracted inline or as a sequence of pairwise stacked
            contrations
        """

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
            raise TypeError('`d_phys` should be `int` type or a list/tuple of ints')

        # d_bond
        if isinstance(d_bond, (list, tuple)):
            if boundary == 'obc':
                if len(d_bond) != n_sites - 1:
                    raise ValueError('If `d_bond` is given as a sequence of int, '
                                     'and `boundary` is "obc", its length should be'
                                     ' equal to `n_sites` - 1')
            elif boundary == 'pbc':
                if len(d_bond) != n_sites:
                    raise ValueError('If `d_bond` is given as a sequence of int, '
                                     'and `boundary` is "pbc", its length should be'
                                     ' equal to `n_sites`')
            self._d_bond = list(d_bond)
        elif isinstance(d_bond, int):
            if boundary == 'obc':
                self._d_bond = [d_bond] * (n_sites - 1)
            elif boundary == 'pbc':
                self._d_bond = [d_bond] * n_sites
        else:
            raise TypeError('`d_bond` should be `int` type or a list/tuple of ints')

        # param_bond
        self._param_bond = param_bond

        self._make_nodes()
        self.param_bond(set_param=param_bond)
        self.initialize()
        
        self._num_batches = num_batches
        
        # Contraction algorithm
        self.inline_input = inline_input
        self.inline_mats = inline_mats

    @property
    def l_position(self) -> int:
        return self._l_position

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def boundary(self) -> Text:
        return self._boundary

    @property
    def d_phys(self) -> List[int]:
        return self._d_phys

    @property
    def d_bond(self) -> List[int]:
        return self._d_bond

    def param_bond(self, set_param: Optional[bool] = None) -> Optional[bool]:
        """
        Return param_bond attribute or change it if set_param is provided.

        Parameters
        ----------
        set_param: boolean indicating whether edges have to be parameterized
                   (True) or de-parameterized (False)
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
        if self.nodes:
            raise ValueError('Cannot create MPS nodes if the MPS already has nodes')

        self.left_node = None
        self.right_node = None
        self.mats_env = []
        
        if self._n_sites == 1:
            node = ParamNode(shape=(self.d_bond[0], self.d_phys[0], self.d_bond[0]),
                             axes_names=('left', 'input', 'right'),
                             name=f'mats_env_node',
                             network=self)
            node['right'] ^ node['left']
            self.mats_env.append(node)
            
        else:
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
        # Left node
        if self.left_node is not None:
            tensor = torch.randn(self.left_node.shape) * std
            if self.boundary == 'obc':
                aux = torch.zeros(tensor.shape[1]) * std  # NOTE: Add randn to eye?
                aux[0] = 1.
                tensor[0, :] = aux
            else:
                aux = torch.eye(self.left_node.shape[0], self.left_node.shape[2])
                tensor[:, 0, :] = aux
            self.left_node.tensor = tensor
        
        # Right node
        if self.right_node is not None:
            tensor = torch.randn(self.right_node.shape) * std
            if self.boundary == 'obc':
                aux = torch.zeros(tensor.shape[0]) * std  # NOTE: Add randn to eye?
                aux[0] = 1.
                tensor[:, 0] = aux
            else:
                aux = torch.eye(self.right_node.shape[0], self.right_node.shape[2])
                tensor[:, 0, :] = aux
            self.right_node.tensor = tensor
        
        # Mats env
        for node in self.mats_env:
            tensor = torch.randn(node.shape) * std
            aux = torch.eye(tensor.shape[0], tensor.shape[2])  # NOTE: Add randn to eye?
            tensor[:, 0, :] = aux
            node.tensor = tensor

    def set_data_nodes(self) -> None:
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

    def _input_contraction(self) -> Tuple[Optional[List[Node]],
                                          Optional[List[Node]]]:
        if self.inline_input:
            mats_result = []
            for node in self.mats_env:
                mats_result.append(node @ node.neighbours('input'))
            return mats_result

        else:
            # start = time.time()
            if self.mats_env:
                # print('\t\tFind data:', time.time() - start)
                start = time.time()
                stack = tk.stack(self.mats_env)
                stack_data = tk.stack(self.mats_env_data)
                stack['input'] ^ stack_data['feature']
                result = stack @ stack_data
                mats_result = tk.unbind(result)
                if PRINT_MODE: print('\t\tResult:', time.time() - start)
                return mats_result
            else:
                return []

    @staticmethod
    def _inline_contraction(nodes: List[Node]) -> Node:
        """
        Contract a sequence of MPS tensors that have
        parameterized bond dimensions.
        
        Parameters
        ----------
        nodes: list of nodes (cannot be empty)
        """
        result_node = nodes[0]
        for node in nodes[1:]:
            start = time.time()
            result_node @= node
            if PRINT_MODE: print('\t\t\tMatrix contraction:', time.time() - start)
        return result_node
        
    def _contract_envs_inline(self, mats_env: List[Node]) -> Node:
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
        length = len(nodes)
        aux_nodes = nodes
        if length > 1:
            start_total = time.time()

            half_length = length // 2
            nice_length = 2 * half_length

            start = time.time()
            even_nodes = aux_nodes[0:nice_length:2]
            odd_nodes = aux_nodes[1:nice_length:2]
            leftover = aux_nodes[nice_length:]
            if PRINT_MODE: print('\t\t\tSelect nodes:', time.time() - start)

            start = time.time()
            stack1 = tk.stack(even_nodes)
            if PRINT_MODE: print('\t\t\tStack even nodes:', time.time() - start)
            start = time.time()
            stack2 = tk.stack(odd_nodes)
            if PRINT_MODE: print('\t\t\tStack odd nodes:', time.time() - start)
            start = time.time()
            stack1['right'] ^ stack2['left']
            if PRINT_MODE: print('\t\t\tConnect stacks:', time.time() - start)
            start = time.time()
            aux_nodes = stack1 @ stack2
            if PRINT_MODE: print('\t\t\tContract stacks:', time.time() - start)
            start = time.time()
            aux_nodes = tk.unbind(aux_nodes)
            if PRINT_MODE: print('\t\t\tUnbind stacks:', time.time() - start)

            if PRINT_MODE: print('\t\tPairwise contraction:', time.time() - start_total)

            return aux_nodes, leftover
        return nodes, []

    def _pairwise_contraction(self, mats_nodes: List[Node]) -> Node:
        """
        Contract a sequence of MPS tensors that do not have
        parameterized bond dimensions, making the operation
        more efficient (from jemisjoki/TorchMPS)
        """
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

    def contract(self) -> Node:
        start = time.time()
        mats_env = self._input_contraction()
        if PRINT_MODE: print('\tInput:', time.time() - start)

        start = time.time()
        if self.inline_mats:
            result = self._contract_envs_inline(mats_env)
        else:
            result = self._pairwise_contraction(mats_env)
                    
        if PRINT_MODE: print('\tMatrices contraction:', time.time() - start)
        
        return result
    
    def canonicalize(self,
                     oc: Optional[int] = None,
                     mode: Text = 'svd',
                     rank: Optional[int] = None,
                     cum_percentage: Optional[float] = None) -> None:
        """
        Turns the MPS into canonical form
        
        Parameters
        ----------
        oc: orthogonality center position
        mode: can be either 'svd', 'svdr' or 'qr'
        """
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
                result1, result2 = nodes[i]['right'].svd_(side='right',
                                                          rank=rank,
                                                          cum_percentage=cum_percentage)
            elif mode == 'svdr':
                result1, result2 = nodes[i]['right'].svdr_(side='right',
                                                           rank=rank,
                                                           cum_percentage=cum_percentage)
            elif mode == 'qr':
                result1, result2 = nodes[i]['right'].qr_()
            else:
                raise ValueError('`mode` can only be \'svd\', \'svdr\' or \'qr\'')
            
            result1 = result1.parameterize()
            nodes[i] = result1
            nodes[i + 1] = result2
            
        for i in range(len(nodes) - 1, oc, -1):
            if mode == 'svd':
                result1, result2 = nodes[i]['left'].svd_(side='left',
                                                         rank=rank,
                                                         cum_percentage=cum_percentage)
            elif mode == 'svdr':
                result1, result2 = nodes[i]['left'].svdr_(side='left',
                                                          rank=rank,
                                                          cum_percentage=cum_percentage)
            elif mode == 'qr':
                result1, result2 = nodes[i]['left'].rq_()
            else:
                raise ValueError('`mode` can only be \'svd\', \'svdr\' or \'qr\'')
            
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


class UMPS(TensorNetwork):

    def __init__(self,
                 n_sites: int,
                 d_phys: int,
                 d_bond: int,
                 param_bond: bool = False,
                 num_batches: int = 1,
                 inline_input: bool = False,
                 inline_mats: bool = False) -> None:
        """
        Create an MPS module.

        Parameters
        ----------
        n_sites: number of sites, including the input and output_node sites
        d_phys: physic dimension
        d_bond: bond dimension. If given as a sequence, the i-th bond
            dimension is always the dimension of the right edge of th i-th node
        param_bond: boolean indicating whether bond edges should be parametric
        num_batches: number of batch edges of input data
        inline_input: boolean indicating whether input should be contracted
            inline or in a single stacked contraction
        inline_mats: boolean indicating whether sequence of matrices
            should be contracted inline or as a sequence of pairwise stacked
            contrations
        """

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
        
        # Contraction algorithm
        self.inline_input = inline_input
        self.inline_mats = inline_mats

    @property
    def l_position(self) -> int:
        return self._l_position

    @property
    def n_sites(self) -> int:
        return self._n_sites

    @property
    def boundary(self) -> Text:
        return self._boundary

    @property
    def d_phys(self) -> List[int]:
        return self._d_phys

    @property
    def d_bond(self) -> List[int]:
        return self._d_bond

    def param_bond(self, set_param: Optional[bool] = None) -> Optional[bool]:
        """
        Return param_bond attribute or change it if set_param is provided.

        Parameters
        ----------
        set_param: boolean indicating whether edges have to be parameterized
                   (True) or de-parameterized (False)
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
        if self.nodes:
            raise ValueError('Cannot create MPS nodes if the MPS already has nodes')
        
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
        uniform_memory = tk.ParamNode(shape=(self.d_bond, self.d_phys, self.d_bond),
                                      axes_names=('left', 'input', 'right'),
                                      name='virtual_uniform',
                                      network=self,
                                      virtual=True)
        self.uniform_memory = uniform_memory
        
        for edge in uniform_memory._edges:
            self._remove_edge(edge)

    def initialize(self, std: float = 1e-9) -> None:
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

    def _input_contraction(self) -> Tuple[Optional[List[Node]],
                                          Optional[List[Node]]]:
        if self.inline_input:
            mats_result = []
            for node in self.mats_env:
                mats_result.append(node @ node.neighbours('input'))
            return mats_result

        else:
            # start = time.time()
            if self.mats_env:
                # print('\t\tFind data:', time.time() - start)
                start = time.time()
                stack = tk.stack(self.mats_env)
                stack_data = tk.stack(self.mats_env_data)
                stack['input'] ^ stack_data['feature']
                result = stack @ stack_data
                mats_result = tk.unbind(result)
                if PRINT_MODE: print('\t\tResult:', time.time() - start)
                return mats_result
            else:
                return []

    @staticmethod
    def _inline_contraction(nodes: List[Node]) -> Node:
        """
        Contract a sequence of MPS tensors that have
        parameterized bond dimensions.
        
        Parameters
        ----------
        nodes: list of nodes (cannot be empty)
        """
        result_node = nodes[0]
        for node in nodes[1:]:
            start = time.time()
            result_node @= node
            if PRINT_MODE: print('\t\t\tMatrix contraction:', time.time() - start)
        return result_node
        
    def _contract_envs_inline(self, mats_env: List[Node]) -> Node:
        if len(mats_env) > 1:
            contract_lst = mats_env
        else:
            return mats_env[0] @ mats_env[0]
        
        return self._inline_contraction(contract_lst)

    def _aux_pairwise(self, nodes: List[Node]) -> Tuple[List[Node],
                                                        List[Node]]:
        length = len(nodes)
        aux_nodes = nodes
        if length > 1:
            start_total = time.time()

            half_length = length // 2
            nice_length = 2 * half_length

            start = time.time()
            even_nodes = aux_nodes[0:nice_length:2]
            odd_nodes = aux_nodes[1:nice_length:2]
            leftover = aux_nodes[nice_length:]
            if PRINT_MODE: print('\t\t\tSelect nodes:', time.time() - start)

            start = time.time()
            stack1 = tk.stack(even_nodes)
            if PRINT_MODE: print('\t\t\tStack even nodes:', time.time() - start)
            start = time.time()
            stack2 = tk.stack(odd_nodes)
            if PRINT_MODE: print('\t\t\tStack odd nodes:', time.time() - start)
            start = time.time()
            stack1['right'] ^ stack2['left']
            if PRINT_MODE: print('\t\t\tConnect stacks:', time.time() - start)
            start = time.time()
            aux_nodes = stack1 @ stack2
            if PRINT_MODE: print('\t\t\tContract stacks:', time.time() - start)
            start = time.time()
            aux_nodes = tk.unbind(aux_nodes)
            if PRINT_MODE: print('\t\t\tUnbind stacks:', time.time() - start)

            if PRINT_MODE: print('\t\tPairwise contraction:', time.time() - start_total)

            return aux_nodes, leftover
        return nodes, []

    def _pairwise_contraction(self, mats_nodes: List[Node]) -> Node:
        """
        Contract a sequence of MPS tensors that do not have
        parameterized bond dimensions, making the operation
        more efficient (from jemisjoki/TorchMPS)
        """
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

    def contract(self) -> Node:
        start = time.time()
        mats_env = self._input_contraction()
        if PRINT_MODE: print('\tInput:', time.time() - start)

        start = time.time()
        if self.inline_mats:
            result = self._contract_envs_inline(mats_env)
        else:
            result = self._pairwise_contraction(mats_env)
                    
        if PRINT_MODE: print('\tMatrices contraction:', time.time() - start)
        
        return result
    
    # def canonicalize(self,
    #                  mode: Text = 'svd',
    #                  rank: Optional[int] = None,
    #                  cum_percentage: Optional[float] = None) -> None:
    #     # Left
    #     left_nodes = []
    #     if self.left_node is not None:
    #         left_nodes.append(self.left_node)
    #     left_nodes += self.left_env
        
    #     if left_nodes:
    #         new_left_nodes = []
    #         node = left_nodes[0]
    #         for _ in range(len(left_nodes)):
    #             if mode == 'svd':
    #                 result1, result2 = node['right'].svd_(side='right',
    #                                                       rank=rank,
    #                                                       cum_percentage=cum_percentage)
    #             elif mode == 'svdr':
    #                 result1, result2 = node['right'].svdr_(side='right',
    #                                                        rank=rank,
    #                                                        cum_percentage=cum_percentage)
    #             elif mode == 'qr':
    #                 result1, result2 = node['right'].qr_()
    #             else:
    #                 raise ValueError('`mode` can only be \'svd\', \'svdr\' or \'qr\'')
                
    #             node = result2
    #             result1 = result1.parameterize()
    #             new_left_nodes.append(result1)
                
    #         output_node = node
                
    #         if new_left_nodes:
    #             self.left_node = new_left_nodes[0]
    #         self.left_env = new_left_nodes[1:]

    #     # Right
    #     right_nodes = self.right_env[:]
    #     right_nodes.reverse()
    #     if self.right_node is not None:
    #         right_nodes = [self.right_node] + right_nodes
        
    #     if right_nodes:
    #         new_right_nodes = []
    #         node = right_nodes[0]
    #         for _ in range(len(right_nodes)):
    #             if mode == 'svd':
    #                 result1, result2 = node['left'].svd_(side='left',
    #                                                      rank=rank,
    #                                                      cum_percentage=cum_percentage)
    #             elif mode == 'svdr':
    #                 result1, result2 = node['left'].svdr_(side='left',
    #                                                       rank=rank,
    #                                                       cum_percentage=cum_percentage)
    #             elif mode == 'qr':
    #                 result1, result2 = node['left'].qr_()
    #             else:
    #                 raise ValueError('`mode` can only be \'svd\', \'svdr\' or \'qr\'')
                
    #             node = result1
    #             result2 = result2.parameterize()
    #             new_right_nodes = [result2] + new_right_nodes
                
    #         output_node = node
                
    #         if new_right_nodes:
    #             self.right_node = new_right_nodes[-1]
    #         self.right_env = new_right_nodes[:-1]
            
    #     self.output_node = output_node.parameterize()


class ConvMPS(MPS):
    
    def __init__(self,
                 in_channels: int,
                 d_bond: Union[int, Sequence[int]],
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 boundary: Text = 'obc',
                 param_bond: bool = False,
                 inline_input: bool = False,
                 inline_mats: bool = False):
        
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
                         num_batches=2,
                         inline_input=inline_input,
                         inline_mats=inline_mats)
        
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)
        
    @property
    def in_channels(self) -> int:
        return self._in_channels
    
    @property
    def kernel_size(self) -> Tuple[int, int]:
        return self._kernel_size
    
    @property
    def stride(self) -> Tuple[int, int]:
        return self._stride
    
    @property
    def padding(self) -> Tuple[int, int]:
        return self._padding
    
    @property
    def dilation(self) -> Tuple[int, int]:
        return self._dilation
    
    def forward(self, image, mode='flat'):
        """
        Parameters
        ----------
        image: input image with shape batch_size x in_channels x height x width
        mode: can be either 'flat' or 'snake', indicates the ordering of
            the pixels in the MPS
        """
        # Input image shape: batch_size x in_channels x height x width
        
        patches = self.unfold(image).transpose(1, 2)
        # batch_size x nb_windows x (in_channels * nb_pixels)
        
        patches = patches.view(*patches.shape[:-1], self.in_channels, -1)
        # batch_size x nb_windows x in_channels x nb_pixels
        
        patches = patches.permute(3, 0, 1, 2)
        # nb_pixels x batch_size x nb_windows x in_channels
        
        result = super().forward(patches)
        # batch_size x nb_windows
        
        h_in = image.shape[2]
        w_in = image.shape[3]
        
        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] * \
            (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] * \
                    (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        
        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x out_channels x height_out x width_out
        
        return result
    
    
class ConvUMPS(UMPS):
    
    def __init__(self,
                    in_channels: int,
                    d_bond: int,
                    kernel_size: Union[int, Sequence],
                    stride: int = 1,
                    padding: int = 0,
                    dilation: int = 1,
                    param_bond: bool = False,
                    inline_input: bool = False,
                    inline_mats: bool = False):
        
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
                         num_batches=2,
                         inline_input=inline_input,
                         inline_mats=inline_mats)
        
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)
        
    @property
    def in_channels(self) -> int:
        return self._in_channels
    
    @property
    def kernel_size(self) -> Tuple[int, int]:
        return self._kernel_size
    
    @property
    def stride(self) -> Tuple[int, int]:
        return self._stride
    
    @property
    def padding(self) -> Tuple[int, int]:
        return self._padding
    
    @property
    def dilation(self) -> Tuple[int, int]:
        return self._dilation
    
    def forward(self, image, mode='flat'):
        """
        Parameters
        ----------
        image: input image with shape batch_size x in_channels x height x width
        mode: can be either 'flat' or 'snake', indicates the ordering of
            the pixels in the MPS
        """
        # Input image shape: batch_size x in_channels x height x width
        
        patches = self.unfold(image).transpose(1, 2)
        # batch_size x nb_windows x (in_channels * nb_pixels)
        
        patches = patches.view(*patches.shape[:-1], self.in_channels, -1)
        # batch_size x nb_windows x in_channels x nb_pixels
        
        patches = patches.permute(3, 0, 1, 2)
        # nb_pixels x batch_size x nb_windows x in_channels
        
        result = super().forward(patches)
        # batch_size x nb_windows
        
        h_in = image.shape[2]
        w_in = image.shape[3]
        
        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] * \
            (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] * \
                    (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        
        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x out_channels x height_out x width_out
        
        return result
