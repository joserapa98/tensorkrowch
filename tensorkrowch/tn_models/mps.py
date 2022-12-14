"""
MPS class
"""

from typing import (Union, Optional, Sequence,
                    Text, List, Tuple)

import torch
from torch.nn.functional import pad

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
                 n_labels: int,
                 d_bond: Union[int, Sequence[int]],
                 l_position: Optional[int] = None,
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
        n_labels: output_node label dimension
        d_bond: bond dimension. If given as a sequence, the i-th bond
            dimension is always the dimension of the right edge of th i-th node
        l_position: position of output_node site
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

        # l_position
        if l_position is None:
            l_position = n_sites // 2
        self._l_position = l_position

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
            if len(d_phys) != n_sites - 1:
                raise ValueError('If `d_phys` is given as a sequence of int, '
                                 'its length should be equal to `n_sites` - 1')
            self._d_phys = list(d_phys[:l_position]) + [n_labels] + \
                           list(d_phys[l_position:])
        elif isinstance(d_phys, int):
            self._d_phys = [d_phys] * l_position + [n_labels] + \
                           [d_phys] * (n_sites - l_position - 1)
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
            raise TypeError('`d_phys` should be `int` type or a list/tuple of ints')

        # param_bond
        self._param_bond = param_bond

        self._make_nodes()
        self.param_bond(set_param=param_bond)
        self.initialize()
        
        self.num_batches = num_batches
        
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
            for node in self.nodes.values():
                if 'left' in node.axes_names:
                    node['left'].parameterize(set_param=set_param)
                if 'right' in node.axes_names:
                    node['right'].parameterize(set_param=set_param)
            self._param_bond = set_param

    def _make_nodes(self) -> None:
        if self.nodes:
            raise ValueError('Cannot create MPS nodes if the MPS already has nodes')

        self.left_env = []
        self.right_env = []

        # Left
        if self.l_position > 0:
            # Left node
            if self.boundary == 'obc':
                self.left_node = ParamNode(shape=(self.d_phys[0], self.d_bond[0]),
                                           axes_names=('input', 'right'),
                                           name='left_node',
                                           network=self)
                                           #param_edges=self.param_bond())
            else:
                self.left_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[0], self.d_bond[0]),
                                           axes_names=('left', 'input', 'right'),
                                           name='left_node',
                                           network=self)
                                           #param_edges=self.param_bond())
                periodic_edge = self.left_node['left']
        else:
            self.left_node = None

        if self.l_position > 1:
            # Left environment
            for i in range(1, self.l_position):
                node = ParamNode(shape=(self.d_bond[i - 1], self.d_phys[i], self.d_bond[i]),
                                 axes_names=('left', 'input', 'right'),
                                 name=f'left_env_node_({i})',
                                 network=self)
                                 #param_edges=self.param_bond())
                self.left_env.append(node)
                if i == 1:
                    self.left_node['right'] ^ self.left_env[-1]['left']
                else:
                    self.left_env[-2]['right'] ^ self.left_env[-1]['left']

        # Output
        if self.l_position == 0:
            if self.boundary == 'obc':
                self.output_node = ParamNode(shape=(self.d_phys[0], self.d_bond[0]),
                                             axes_names=('output', 'right'),
                                             name='output_node',
                                             network=self)
                                             #param_edges=self.param_bond())
            else:
                self.output_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[0], self.d_bond[0]),
                                             axes_names=('left', 'output', 'right'),
                                             name='output_node',
                                             network=self)
                                             #param_edges=self.param_bond())
                periodic_edge = self.output_node['left']

        if self.l_position == self.n_sites - 1:
            # if self.n_sites - 1 != 0:
            if self.boundary == 'obc':
                if self.n_sites - 1 != 0:
                    self.output_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[-1]),
                                                    axes_names=('left', 'output'),
                                                    name='output_node',
                                                    network=self)
                                                    #param_edges=self.param_bond())
            else:
                if self.n_sites - 1 != 0:
                    self.output_node = ParamNode(shape=(self.d_bond[-2], self.d_phys[-1], self.d_bond[-1]),
                                                    axes_names=('left', 'output', 'right'),
                                                    name='output_node',
                                                    network=self)
                                                    #param_edges=self.param_bond())
                self.output_node['right'] ^ periodic_edge
                    
            if self.left_env:
                self.left_env[-1]['right'] ^ self.output_node['left']
            else:
                if self.left_node:
                    self.left_node['right'] ^ self.output_node['left']

        if (self.l_position > 0) and (self.l_position < self.n_sites - 1):
            self.output_node = ParamNode(shape=(self.d_bond[self.l_position - 1],
                                                self.d_phys[self.l_position],
                                                self.d_bond[self.l_position]),
                                         axes_names=('left', 'output', 'right'),
                                         name='output_node',
                                         network=self)
                                         #param_edges=self.param_bond())
            if self.left_env:
                self.left_env[-1]['right'] ^ self.output_node['left']
            else:
                self.left_node['right'] ^ self.output_node['left']

        # Right
        if self.l_position < self.n_sites - 2:
            # Right environment
            for i in range(self.l_position + 1, self.n_sites - 1):
                node = ParamNode(shape=(self.d_bond[i - 1], self.d_phys[i], self.d_bond[i]),
                                 axes_names=('left', 'input', 'right'),
                                 name=f'right_env_node_({i})',
                                 network=self)
                                 #param_edges=self.param_bond())
                self.right_env.append(node)
                if i == self.l_position + 1:
                    self.output_node['right'] ^ self.right_env[-1]['left']
                else:
                    self.right_env[-2]['right'] ^ self.right_env[-1]['left']

        if self.l_position < self.n_sites - 1:
            # Right node
            if self.boundary == 'obc':
                self.right_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[-1]),
                                            axes_names=('left', 'input'),
                                            name='right_node',
                                            network=self)
                                            #param_edges=self.param_bond())
            else:
                self.right_node = ParamNode(shape=(self.d_bond[-2], self.d_phys[-1], self.d_bond[-1]),
                                            axes_names=('left', 'input', 'right'),
                                            name='right_node',
                                            network=self)
                                            #param_edges=self.param_bond())
                self.right_node['right'] ^ periodic_edge
            if self.right_env:
                self.right_env[-1]['right'] ^ self.right_node['left']
            else:
                self.output_node['right'] ^ self.right_node['left']
        else:
            self.right_node = None

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
        
        # Left env + Right env
        for node in self.left_env + self.right_env:
            tensor = torch.randn(node.shape) * std
            aux = torch.eye(tensor.shape[0], tensor.shape[2])  # NOTE: Add randn to eye?
            tensor[:, 0, :] = aux
            node.tensor = tensor
            
        # Output node
        if self.l_position == 0:
            if self.boundary == 'obc':
                eye_tensor = torch.eye(self.output_node.shape[1])[0, :]
                eye_tensor = eye_tensor.view([1, self.output_node.shape[1]])
                eye_tensor = eye_tensor.expand(self.output_node.shape)
            else:
                eye_tensor = torch.eye(self.output_node.shape[0], self.output_node.shape[2])
                eye_tensor = eye_tensor.view([self.output_node.shape[0], 1, self.output_node.shape[2]])
                eye_tensor = eye_tensor.expand(self.output_node.shape)

        elif self.l_position == self.n_sites - 1:
            if self.boundary == 'obc':
                eye_tensor = torch.eye(self.output_node.shape[0])[0, :]
                eye_tensor = eye_tensor.view([self.output_node.shape[0], 1])
                eye_tensor = eye_tensor.expand(self.output_node.shape)
            else:
                eye_tensor = torch.eye(self.output_node.shape[0], self.output_node.shape[2])
                eye_tensor = eye_tensor.view([self.output_node.shape[0], 1, self.output_node.shape[2]])
                eye_tensor = eye_tensor.expand(self.output_node.shape)
        else:
            # NOTE: trying oher initializations
            eye_tensor = torch.eye(self.output_node.shape[0], self.output_node.shape[2])
            eye_tensor = eye_tensor.view([self.output_node.shape[0], 1, self.output_node.shape[2]])
            eye_tensor = eye_tensor.expand(self.output_node.shape)
            
            # eye_tensor = torch.zeros(node.shape)
            # eye_tensor[0, 0, 0] += 1.

        # Add on a bit of random noise
        tensor = eye_tensor + std * torch.randn(self.output_node.shape)
        self.output_node.tensor = tensor

    def set_data_nodes(self) -> None:
        input_edges = []
        if self.left_node is not None:
            input_edges.append(self.left_node['input'])
        input_edges += list(map(lambda node: node['input'],
                                self.left_env + self.right_env))
        if self.right_node is not None:
            input_edges.append(self.right_node['input'])
            
        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self.num_batches)
        
        if self.left_env + self.right_env:
            self.lr_env_data = list(map(lambda node: node.neighbours('input'), self.left_env + self.right_env))

    def _input_contraction(self) -> Tuple[Optional[List[Node]],
                                          Optional[List[Node]]]:
        if not self.inline_input:
            # start = time.time()
            if self.left_env + self.right_env:
                # print('\t\tFind data:', time.time() - start)
                start = time.time()
                stack = tk.stack(self.left_env + self.right_env)
                stack_data = tk.stack(self.lr_env_data)
                stack['input'] ^ stack_data['feature']
                result = stack @ stack_data
                result = tk.unbind(result)
                if PRINT_MODE: print('\t\tResult:', time.time() - start)
                start = time.time()
                left_result = result[:len(self.left_env)]
                right_result = result[len(self.left_env):]
                if PRINT_MODE: print('\t\tLists:', time.time() - start)
                return left_result, right_result
            else:
                return [], []

        else:
            left_result = []
            for node in self.left_env:
                left_result.append(node @ node.neighbours('input'))
            right_result = []
            for node in self.right_env:
                right_result.append(node @ node.neighbours('input'))
            return left_result, right_result

    @staticmethod
    def _inline_contraction(nodes: List[Node], left) -> Node:
        """
        Contract a sequence of MPS tensors that have
        parameterized bond dimensions
        """
        if left:
            result_node = nodes[0]
            for node in nodes[1:]:
                start = time.time()
                result_node @= node
                if PRINT_MODE: print('\t\t\tMatrix contraction (left):', time.time() - start)
            return result_node
        else:
            result_node = nodes[0]
            for node in nodes[1:]:
                start = time.time()
                result_node = node @ result_node
                if PRINT_MODE: print('\t\t\tMatrix contraction (right):', time.time() - start)
            return result_node
        
    def _contract_envs_inline(self,
                              left_env: List[Node],
                              right_env: List[Node]) -> Tuple[List[Node],
                                                              List[Node]]:
        if left_env:
            left_node = (self.left_node @ self.left_node.neighbours('input'))#.permute((1, 0))
            left_env = [self._inline_contraction([left_node] + left_env, True)]
        elif self.left_node is not None:
            left_env = [self.left_node @ self.left_node.neighbours('input')]

        if right_env:
            right_node = self.right_node @ self.right_node.neighbours('input')
            lst = right_env + [right_node]
            lst.reverse()
            right_env = [self._inline_contraction(lst, False)]
        elif self.right_node is not None:
            right_env = [self.right_node @ self.right_node.neighbours('input')]
        else:
            right_env = []
            
        return left_env, right_env

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

    def _pairwise_contraction(self,
                              left_nodes: List[Node],
                              right_nodes: List[Node]) -> Tuple[List[Node],
                                                                List[Node]]:
        """
        Contract a sequence of MPS tensors that do not have
        parameterized bond dimensions, making the operation
        more efficient (from jemisjoki/TorchMPS)
        """
        left_length = len(left_nodes)
        left_aux_nodes = left_nodes
        right_length = len(right_nodes)
        right_aux_nodes = right_nodes
        if left_length > 1 or right_length > 1:
            left_leftovers = []
            right_leftovers = []
            while left_length > 1 or right_length > 1:
                aux1, aux2 = self._aux_pairwise(left_aux_nodes)
                left_aux_nodes = aux1
                left_leftovers = aux2 + left_leftovers
                left_length = len(aux1)

                aux1, aux2 = self._aux_pairwise(right_aux_nodes)
                right_aux_nodes = aux1
                right_leftovers = aux2 + right_leftovers
                right_length = len(aux1)

            left_aux_nodes = left_aux_nodes + left_leftovers
            right_aux_nodes = right_aux_nodes + right_leftovers
            return self._pairwise_contraction(left_aux_nodes, right_aux_nodes)

        return self._contract_envs_inline(left_aux_nodes, right_aux_nodes)

    def contract(self) -> Node:
        start = time.time()
        left_env, right_env = self._input_contraction()
        if PRINT_MODE: print('\tInput:', time.time() - start)

        start = time.time()
        if not self.inline_mats:
            left_env_contracted, right_env_contracted = self._pairwise_contraction(left_env, right_env)
        else:
            left_env_contracted, right_env_contracted = self._contract_envs_inline(left_env, right_env)
                    
        if PRINT_MODE: print('\tMatrices contraction:', time.time() - start)
        
        result = self.output_node
        if left_env_contracted and right_env_contracted:
            result = left_env_contracted[0] @ result @ right_env_contracted[0]
        elif left_env_contracted:
            result = left_env_contracted[0] @ result
        elif right_env_contracted:
            result = right_env_contracted[0] @ result
        else:
            result @= result
        return result
