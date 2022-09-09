"""
MPS class
"""

from typing import (Union, Optional, Sequence,
                    Text, List, Tuple)

import torch
from torch.nn.functional import pad

from tentorch.network_components import (AbstractNode, Node, ParamNode,
                                         AbstractEdge)
from tentorch.network_components import TensorNetwork

from tentorch.node_operations import einsum, stacked_einsum

import opt_einsum

import time


# TODO: move_l_position -> needs svd and qr to contract and split nodes
# TODO: change l_position int by just 'start', 'end', 'medium',
#  and n_sites -> n_features (n_sites - 1)
class MPS(TensorNetwork):

    def __init__(self,
                 n_sites: int,
                 d_phys: Union[int, Sequence[int]],
                 n_labels: int,
                 d_bond: Union[int, Sequence[int]],
                 l_position: Optional[int] = None,
                 boundary: Text = 'obc',
                 param_bond: bool = False) -> None:
        """
        Create an MPS module.

        Parameters
        ----------
        n_sites: number of sites, including the input and output_node sites
        d_phys: physic dimension
        n_labels: output_node dimension
        d_bond: bond dimension. If given as a sequence, the i-th bond
                dimension is always the dimension of the right edge of
                th i-th node
        l_position: position of output_node site
        param_bond: boolean indicating whether bond edges should be parametric
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

        # d_phys and d_phys_l
        if isinstance(d_phys, (list, tuple)):
            if len(d_phys) != n_sites - 1:
                raise ValueError('If `d_phys` is given as a sequence of int, '
                                 'its length should be equal to `n_sites` - 1')
            self._d_phys = list(d_phys[:l_position]) + [n_labels] + \
                           list(d_phys[l_position:])
        elif isinstance(d_phys, int):
            self._d_phys = [d_phys] * l_position + [n_labels] + \
                           [d_phys] * (n_sites - l_position - 1)

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

        # param_bond
        self._param_bond = param_bond

        self._make_nodes()
        self.param_bond(set_param=param_bond)
        self.initialize()

        # Save references to permanent nodes
        # TODO: This should be in all TN
        self._permanent_nodes = []

        permanent_nodes = []
        if self.left_node is not None:
            permanent_nodes.append(self.left_node)
        permanent_nodes += self.left_env + [self.output_node] + self.right_env
        if self.right_node is not None:
            permanent_nodes.append(self.right_node)
        self._permanent_nodes = permanent_nodes

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

    @property
    def permanent_nodes(self) -> List[AbstractNode]:
        return self._permanent_nodes

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

    def same_d_phys(self) -> bool:
        """
        Check if all physic dimensions(sizes) are equal,
        in order to perform a pairwise contraction
        """
        input_edges = []
        if self.left_node is not None:
            input_edges.append(self.left_node['input'])
        input_edges += list(map(lambda node: node['input'],
                                self.left_env + self.right_env))
        if self.right_node is not None:
            input_edges.append(self.right_node['input'])

        for i, _ in enumerate(input_edges[:-1]):
            if input_edges[i].size() != input_edges[i + 1].size():
                return False
        return True

    def same_d_bond(self) -> bool:
        """
        Check if bond edges sizes are all equal, in
        order to perform a pairwise contraction
        """
        env_nodes = self.left_env + self.right_env
        for i, _ in enumerate(env_nodes[:-1]):
            if env_nodes[i].size() != env_nodes[i + 1].size():
                return False
        return True

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
                                           network=self,)
                                           #param_edges=self.param_bond())
            else:
                self.left_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[0], self.d_bond[0]),
                                           axes_names=('left', 'input', 'right'),
                                           name='left_node',
                                           network=self,)
                                           #param_edges=self.param_bond())
                periodic_edge = self.left_node['left']
        else:
            self.left_node = None

        if self.l_position > 1:
            # Left environment
            for i in range(1, self.l_position):
                node = ParamNode(shape=(self.d_bond[i - 1], self.d_phys[i], self.d_bond[i]),
                                 axes_names=('left', 'input', 'right'),
                                 name='left_env_node',
                                 network=self,)
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
                                             network=self,)
                                             #param_edges=self.param_bond())
            else:
                self.output_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[0], self.d_bond[0]),
                                             axes_names=('left', 'output', 'right'),
                                             name='output_node',
                                             network=self,)
                                             #param_edges=self.param_bond())
                periodic_edge = self.output_node['left']

        if self.l_position == self.n_sites - 1:
            if self.n_sites - 1 != 0:
                if self.boundary == 'obc':
                    self.output_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[-1]),
                                                 axes_names=('left', 'output'),
                                                 name='output_node',
                                                 network=self,)
                                                 #param_edges=self.param_bond())
                else:
                    self.output_node = ParamNode(shape=(self.d_bond[-2], self.d_phys[-1], self.d_bond[-1]),
                                                 axes_names=('left', 'output', 'right'),
                                                 name='output_node',
                                                 network=self,)
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
                                         network=self,)
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
                                 name='right_env_node',
                                 network=self,)
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
                                            network=self,)
                                            #param_edges=self.param_bond())
            else:
                self.right_node = ParamNode(shape=(self.d_bond[-2], self.d_phys[-1], self.d_bond[-1]),
                                            axes_names=('left', 'input', 'right'),
                                            name='right_node',
                                            network=self,)
                                            #param_edges=self.param_bond())
                self.right_node['right'] ^ periodic_edge
            if self.right_env:
                self.right_env[-1]['right'] ^ self.right_node['left']
            else:
                self.output_node['right'] ^ self.right_node['left']
        else:
            self.right_node = None

    def initialize(self, eps: float = 10e-10) -> None:
        """
        # OBC
        if self.boundary == 'obc':
            # Left node
            if self.left_node is not None:
                self.left_node.set_tensor(init_method='randn',
                                          std=self.left_node['input'].dim() ** (-1/2) + eps)

            # Left environment
            for node in self.left_env:
                node.set_tensor(init_method='randn',
                                std=(node['input'].dim() * node['left'].dim()) ** (-1/2) + eps)

            # Output
            bonds = self.output_node.axes_names[:]
            bonds.remove('output')
            bonds_product = 1
            for name in bonds:
                bonds_product *= self.output_node[name].dim()
            self.output_node.set_tensor(init_method='randn', std=bonds_product ** (-1 / 2) + eps)

            # Right environment
            for node in self.right_env:
                node.set_tensor(init_method='randn',
                                std=(node['input'].dim() * node['right'].dim()) ** (-1/2) + eps)

            # Right node
            if self.right_node is not None:
                self.right_node.set_tensor(init_method='randn',
                                           std=self.right_node['input'].dim() ** (-1/2) + eps)
        else:
            # Left node
            if self.left_node is not None:
                self.left_node.set_tensor(init_method='randn',
                                          std=(self.left_node['input'].dim() *
                                               self.left_node['left'].dim()) ** (-1/2) + eps)

            # Left environment
            for node in self.left_env:
                node.set_tensor(init_method='randn',
                                std=(node['input'].dim() *
                                     node['left'].dim()) ** (-1/2) + eps)

            # Output
            self.output_node.set_tensor(init_method='randn',
                                        std=(self.output_node['output'].dim() *
                                             self.output_node['left'].dim()) ** (-1 / 2) + eps)

            # Right environment
            for node in self.right_env:
                node.set_tensor(init_method='randn',
                                std=(node['input'].dim() *
                                     node['left'].dim()) ** (-1/2) + eps)

            # Right node
            if self.right_node is not None:
                self.right_node.set_tensor(init_method='randn',
                                           std=(self.right_node['input'].dim() *
                                                self.right_node['left'].dim()) ** (-1/2) + eps)

        # Same distribution for all nodes -> it seems to be worse
        bond_product = 1
        for d in self.d_bond:
            bond_product *= d
        for node in self.nodes.values():
            node.set_tensor(init_method='randn', std=bond_product ** (-1 / (2 * self.n_sites)))
        """
        # initialize like torchMPS
        std = 1e-9
        for node in self.nodes.values():
            if node.name != 'output_node':
                #node.set_tensor(init_method='randn', std=std)
                tensor = torch.randn(node.shape) * std
                # TODO: can be also simplified
                if node.name == 'left_node':
                    if self.boundary == 'obc':
                        aux = torch.randn(tensor.shape[1]) * std
                        aux[0] = 1.
                        tensor[0, :] = aux
                    else:
                        aux = torch.eye(node.shape[0], node.shape[2])
                        tensor[:, 0, :] = aux
                elif node.name == 'right_node':
                    if self.boundary == 'obc':
                        aux = torch.randn(tensor.shape[0]) * std
                        aux[0] = 1.
                        tensor[:, 0] = aux
                    else:
                        aux = torch.eye(node.shape[0], node.shape[2])
                        tensor[:, 0, :] = aux
                else:
                    aux = torch.randn(tensor.shape[0], tensor.shape[2]) * std +\
                          torch.eye(tensor.shape[0], tensor.shape[2])
                    tensor[:, 0, :] = aux
                node.set_tensor(tensor=tensor)
            else:
                # TODO: can be simplified
                if self.l_position == 0:
                    if self.boundary == 'obc':
                        eye_tensor = torch.eye(node.shape[1])[0, :].view([1, node.shape[1]])
                        eye_tensor = eye_tensor.expand(node.shape)
                    else:
                        eye_tensor = torch.eye(node.shape[0], node.shape[2]).view([node.shape[0], 1, node.shape[2]])
                        eye_tensor = eye_tensor.expand(node.shape)

                elif self.l_position == self.n_sites - 1:
                    if self.boundary == 'obc':
                        eye_tensor = torch.eye(node.shape[0])[0, :].view([node.shape[0], 1])
                        eye_tensor = eye_tensor.expand(node.shape)
                    else:
                        eye_tensor = torch.eye(node.shape[0], node.shape[2]).view([node.shape[0], 1, node.shape[2]])
                        eye_tensor = eye_tensor.expand(node.shape)
                else:
                    eye_tensor = torch.eye(node.shape[0], node.shape[2]).view([node.shape[0], 1, node.shape[2]])
                    eye_tensor = eye_tensor.expand(node.shape)

                # Add on a bit of random noise
                tensor = eye_tensor + std * torch.randn(node.shape)
                node.set_tensor(tensor=tensor)

    def initialize2(self) -> None:
        # TODO: device should be eligible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # OBC
        if self.boundary == 'obc':
            # Left node
            if self.left_node is not None:
                data = self.left_node.neighbours('input').tensor
                mean_squared = data.pow(2).mean(0)
                target_std = (mean_squared *
                              self.left_node['input'].dim()).pow(-1 / 2)

                tensor = torch.empty_like(self.left_node.tensor).to(device)
                for i in range(tensor.shape[1]):
                    tensor[:, i] = torch.randn(tensor.shape[0]).to(device) * target_std
                self.left_node.set_tensor(tensor=tensor)

            # Left environment
            for node in self.left_env:
                data = node.neighbours('input').tensor
                mean_squared = data.pow(2).mean(0)
                target_std = (mean_squared *
                              node['input'].dim() *
                              node['left'].dim()).pow(-1 / 2)

                tensor = torch.empty_like(node.tensor).to(device)
                for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[2]):
                        tensor[i, :, j] = torch.randn(tensor.shape[1]).to(device) * target_std
                node.set_tensor(tensor=tensor)

            # Output
            bonds = self.output_node.axes_names[:]
            bonds.remove('output')
            bonds_product = 1
            for name in bonds:
                bonds_product *= self.output_node[name].dim()
            self.output_node.set_tensor(device=device,
                                        init_method='randn',
                                        std=bonds_product ** (-1 / 2))

            # Right environment
            for node in self.right_env:
                data = node.neighbours('input').tensor
                mean_squared = data.pow(2).mean(0)
                target_std = (mean_squared *
                              node['input'].dim() *
                              node['right'].dim()).pow(-1 / 2)

                tensor = torch.empty_like(node.tensor).to(device)
                for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[2]):
                        tensor[i, :, j] = torch.randn(tensor.shape[1]).to(device) * target_std
                node.set_tensor(tensor=tensor)

            # Right node
            if self.right_node is not None:
                data = self.right_node.neighbours('input').tensor
                mean_squared = data.pow(2).mean(0)
                target_std = (mean_squared *
                              self.right_node['input'].dim()).pow(-1 / 2)

                tensor = torch.empty_like(self.right_node.tensor).to(device)
                for i in range(tensor.shape[0]):
                    tensor[i, :] = torch.randn(tensor.shape[1]).to(device) * target_std
                self.right_node.set_tensor(tensor=tensor)

        # PBC
        else:
            # Left node
            if self.left_node is not None:
                data = self.left_node.neighbours('input').tensor
                mean_squared = data.pow(2).mean(0)
                target_std = (mean_squared *
                              self.left_node['input'].dim() *
                              self.left_node['left'].dim()).pow(-1 / 2)

                tensor = torch.empty_like(self.left_node.tensor).to(device)
                for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[2]):
                        tensor[i, :, j] = torch.randn(tensor.shape[1]).to(device) * target_std
                self.left_node.set_tensor(tensor=tensor)

            # Left environment
            for node in self.left_env:
                data = node.neighbours('input').tensor
                mean_squared = data.pow(2).mean(0)
                target_std = (mean_squared *
                              node['input'].dim() *
                              node['left'].dim()).pow(-1 / 2)

                tensor = torch.empty_like(node.tensor).to(device)
                for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[2]):
                        tensor[i, :, j] = torch.randn(tensor.shape[1]).to(device) * target_std
                node.set_tensor(tensor=tensor)

            # Output
            self.output_node.set_tensor(device=device,
                                        init_method='randn',
                                        std=self.output_node['left'].dim() ** (-1 / 2))

            # Right environment
            for node in self.right_env:
                data = node.neighbours('input').tensor
                mean_squared = data.pow(2).mean(0)
                target_std = (mean_squared *
                              node['input'].dim() *
                              node['left'].dim()).pow(-1 / 2)

                tensor = torch.empty_like(node.tensor).to(device)
                for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[2]):
                        tensor[i, :, j] = torch.randn(tensor.shape[1]).to(device) * target_std
                node.set_tensor(tensor=tensor)

            # Right node
            if self.right_node is not None:
                data = self.right_node.neighbours('input').tensor
                mean_squared = data.pow(2).mean(0)
                target_std = (mean_squared *
                              self.right_node['input'].dim() *
                              self.right_node['left'].dim()).pow(-1 / 2)

                tensor = torch.empty_like(self.right_node.tensor).to(device)
                for i in range(tensor.shape[0]):
                    for j in range(tensor.shape[2]):
                        tensor[i, :, j] = torch.randn(tensor.shape[1]).to(device) * target_std
                self.right_node.set_tensor(tensor=tensor)

    def set_data_nodes(self,
                       batch_sizes: Sequence[int],
                       input_edges: Optional[Union[List[int],
                                                   List[AbstractEdge]]] = None) -> None:
        if input_edges is None:
            input_edges = []
            if self.left_node is not None:
                input_edges.append(self.left_node['input'])
            input_edges += list(map(lambda node: node['input'],
                                    self.left_env + self.right_env))
            if self.right_node is not None:
                input_edges.append(self.right_node['input'])
        super().set_data_nodes(input_edges=input_edges,
                               batch_sizes=batch_sizes)
        self._permanent_nodes += list(self.data_nodes.values())

    def _input_contraction(self) -> Tuple[Optional[List[Node]],
                                          Optional[List[Node]]]:
        # left_result = None
        # right_result = None
        # if self.left_env:
        #     left_env_data = list(map(lambda node: node.neighbours('input'), self.left_env))
        #     left_result = stacked_einsum('lir,bi->lbr', self.left_env, left_env_data)
        # if self.right_env:
        #     right_env_data = list(map(lambda node: node.neighbours('input'), self.right_env))
        #     right_result = stacked_einsum('lir,bi->lbr', self.right_env, right_env_data)
        # return left_result, right_result

        if self.same_d_bond():
            #start = time.time()
            if self.left_env + self.right_env:
                env_data = list(map(lambda node: node.neighbours('input'), self.left_env + self.right_env))
                #print('\t\tFind data:', time.time() - start)
                #start = time.time()
                result = stacked_einsum('lir,bi->lbr', self.left_env + self.right_env, env_data)
                #print('\t\tResult:', time.time() - start)
                left_result = result[:len(self.left_env)]
                right_result = result[len(self.left_env):]
                return left_result, right_result
            else:
                return [], []

        else:
            left_result = []
            for node in self.left_env:
                left_result.append(node['input'].contract())
            right_result = []
            for node in self.right_env:
                right_result.append(node['input'].contract())
            return left_result, right_result

        # start = time.time()
        # env_data = list(map(lambda node: node.neighbours('input'), self.left_env + self.right_env))
        # print('Find data:', time.time() - start)
        # start = time.time()
        # result = stacked_einsum('lir,bi->lbr', self.left_env + self.right_env, env_data)
        # print('result:', time.time() - start)
        # left_result = result[:len(self.left_env)]
        # right_result = result[len(self.left_env):]
        # return left_result, right_result

    @staticmethod
    def _inline_contraction(nodes: List[Node]) -> Node:
        """
        Contract a sequence of MPS tensors that have
        parameterized bond dimensions
        """
        result_node = nodes[0]
        for node in nodes[1:]:
            result_node @= node
        return result_node

    @staticmethod
    def _pairwise_contraction(left_nodes: List[Node], right_nodes: List[Node]) -> Tuple[Node, Node]:
        """
        Contract a sequence of MPS tensors that do not have
        parameterized bond dimensions, making the operation
        more efficient (from jemisjoki/TorchMPS)
        """
        # TODO: rewrite this (simplify)
        length_left = len(left_nodes)
        length_right = len(right_nodes)
        while (length_left > 1) or (length_right > 1):
            if (length_left > 1) and (length_right > 1):
                odd_length_left = length_left % 2 == 1
                half_length_left = length_left // 2
                nice_length_left = 2 * half_length_left

                left_even_nodes = left_nodes[0:nice_length_left:2]
                left_odd_nodes = left_nodes[1:nice_length_left:2]
                left_leftover = left_nodes[nice_length_left:]

                odd_length_right = length_right % 2 == 1
                half_length_right = length_right // 2
                nice_length_right = 2 * half_length_right

                right_even_nodes = right_nodes[0:nice_length_right:2]
                right_odd_nodes = right_nodes[1:nice_length_right:2]
                right_leftover = right_nodes[nice_length_right:]

                nodes = stacked_einsum('ibj,jbk->ibk',
                                       left_even_nodes + right_even_nodes,
                                       left_odd_nodes + right_odd_nodes)

                left_nodes = nodes[:half_length_left]
                left_nodes += left_leftover
                length_left = half_length_left + int(odd_length_left)

                right_nodes = nodes[half_length_left:]
                right_nodes += right_leftover
                length_right = half_length_right + int(odd_length_right)

            elif length_left > 1:
                odd_length_left = length_left % 2 == 1
                half_length_left = length_left // 2
                nice_length_left = 2 * half_length_left

                left_even_nodes = left_nodes[0:nice_length_left:2]
                left_odd_nodes = left_nodes[1:nice_length_left:2]
                left_leftover = left_nodes[nice_length_left:]

                nodes = stacked_einsum('ibj,jbk->ibk',
                                       left_even_nodes,
                                       left_odd_nodes)

                left_nodes = nodes
                left_nodes += left_leftover
                length_left = half_length_left + int(odd_length_left)

            else:
                odd_length_right = length_right % 2 == 1
                half_length_right = length_right // 2
                nice_length_right = 2 * half_length_right

                right_even_nodes = right_nodes[0:nice_length_right:2]
                right_odd_nodes = right_nodes[1:nice_length_right:2]
                right_leftover = right_nodes[nice_length_right:]

                nodes = stacked_einsum('ibj,jbk->ibk',
                                       right_even_nodes,
                                       right_odd_nodes)

                right_nodes = nodes
                right_nodes += right_leftover
                length_right = half_length_right + int(odd_length_right)

        if left_nodes and right_nodes:
            return left_nodes[0], right_nodes[0]
        elif left_nodes:
            return left_nodes[0], None
        elif right_nodes:
            return None, right_nodes[0]
        else:
            return None, None

    def contract(self) -> Node:
        start = time.time()
        # TODO: case different bond dimensions
        # TODO: should be better to use the same bond dimension across "stackable"  operations,
        #  and add function to "simplify" dimensions later
        left_env, right_env = self._input_contraction()
        #print('\tInput:', time.time() - start)


        if not self.param_bond() and self.same_d_phys() and self.same_d_bond():
            left_env_contracted, right_env_contracted = self._pairwise_contraction(left_env, right_env)
        else:
            left_env_contracted = None
            right_env_contracted = None
            if left_env:
                left_env_contracted = self._inline_contraction(left_env)
            if right_env:
                right_env_contracted = self._inline_contraction(right_env)

        
        # Operations of left environment
        left_list = []
        if self.left_node is not None:
            start = time.time()
            if self.boundary == 'obc':
                left_node = einsum('ir,bi->br', self.left_node, self.left_node.neighbours('input'))
            else:
                left_node = einsum('lir,bi->lbr', self.left_node, self.left_node.neighbours('input'))
            left_list.append(left_node)
        if left_env_contracted is not None:
            left_list.append(left_env_contracted)  # new
            #print('\tLeft node:', time.time() - start)
        # if left_env:
        #     start = time.time()
        #     if not self.param_bond() and self.same_d_phys() and self.same_d_bond():
        #         left_env_contracted = self._pairwise_contraction(left_env)
        #     else:
        #         left_env_contracted = self._inline_contraction(left_env)
        #     left_list.append(left_env_contracted)
        #     #print('\tLeft env:', time.time() - start)

        # Operations of right environment
        right_list = []
        if right_env_contracted is not None:
            right_list.append(right_env_contracted)  # new
        # if right_env:
        #     start = time.time()
        #     if not self.param_bond() and self.same_d_phys() and self.same_d_bond():
        #         right_env_contracted = self._pairwise_contraction(right_env)
        #     else:
        #         right_env_contracted = self._inline_contraction(right_env)
        #     right_list.append(right_env_contracted)
        #     #print('\tRight env:', time.time() - start)
        if self.right_node is not None:
            start = time.time()
            if self.boundary == 'obc':
                right_node = einsum('li,bi->lb', self.right_node, self.right_node.neighbours('input'))
            else:
                right_node = einsum('lir,bi->lbr', self.right_node, self.right_node.neighbours('input'))
            right_list.append(right_node)
            #print('\tRight node:', time.time() - start)

        start = time.time()
        result_list = left_list + [self.output_node] + right_list
        result = result_list[0]
        for node in result_list[1:]:
            result @= node
        #print('\tFinal contraction:', time.time() - start)

        # Clean intermediate nodes
        #mps_nodes = list(self.nodes.values())
        #for node in mps_nodes:
        #    if node not in self.permanent_nodes:
        #        self.delete_node(node)

        return result

    def stack(self, lst_tensors: List[torch.Tensor]) -> torch.Tensor:
        if lst_tensors:
            same_dims_all = True
            max_shape = list(lst_tensors[0].shape)
            for tensor in lst_tensors[1:]:
                same_dims = True
                for idx, dim in enumerate(tensor.shape):
                    if dim > max_shape[idx]:
                        max_shape[idx] = dim
                        same_dims = False
                    elif dim < max_shape[idx]:
                        same_dims = False
                if not same_dims:
                    same_dims_all = False

            if not same_dims_all:
                for idx, tensor in enumerate(lst_tensors):
                    if tensor.shape != max_shape:
                        aux_zeros = torch.zeros(max_shape, device=tensor.device)  # TODO: replace with pad
                        replace_slice = []
                        for dim in tensor.shape:
                            replace_slice.append(slice(0, dim))
                        replace_slice = tuple(replace_slice)
                        aux_zeros[replace_slice] = tensor
                        lst_tensors[idx] = aux_zeros
            return torch.stack(lst_tensors)

    def contract2(self) -> Node:
        start = time.time()
        # TODO: innecesario crear esta lista cada vez, deberíamos llevar guardados esos data_nodes
        list_data_tensors = list(map(lambda node: node.neighbours('input').tensor, self.left_env + self.right_env))
        #print('Search for data tensors:', time.time() - start)
        start = time.time()
        data_tensor = torch.stack(list_data_tensors)
        #print('Stack data tensors:', time.time() - start)

        start = time.time()
        # TODO: lo mismo esto
        list_mps_tensors = list(map(lambda node: node.tensor, self.left_env + self.right_env))
        #print('Search for node tensors:', time.time() - start)
        start = time.time()
        # TODO: también innecesario hacer bucle para buscar max_shape cada vez
        mps_tensor = self.stack(list_mps_tensors)
        #print('Stack node tensors:', time.time() - start)

        start = time.time()
        # Similar in time
        # mps_tensor = mps_tensor.permute(0, 1, 3, 2)
        # data_tensor = data_tensor.permute(0, 2, 1)
        # print('\tEinsum stacks - permutes:', time.time() - start)
        # envs_result = mps_tensor.reshape(mps_tensor.shape[0], -1, mps_tensor.shape[-1]) @ data_tensor
        # print('\tEinsum stacks - envs_result:', time.time() - start)
        # envs_result = envs_result.view(*(list(mps_tensor.shape[:-1]) + [data_tensor.shape[-1]])).permute(0, 1, 3, 2)
        # print('\tEinsum stacks - envs_result permute:', time.time() - start)
        envs_result = opt_einsum.contract('slir,sbi->slbr', mps_tensor, data_tensor)
        #print('Einsum stacks:', time.time() - start)

        start = time.time()
        left_tensors = envs_result[:len(self.left_env)].permute(2, 0, 1, 3)
        right_tensors = envs_result[len(self.left_env):].permute(2, 0, 1, 3)

        n_mats = left_tensors.shape[1]
        while n_mats > 1:
            half_n = n_mats // 2
            floor_n = half_n * 2

            # Split matrices up into even and odd numbers (maybe w/ leftover)
            even_mats = left_tensors[:, 0:floor_n:2, :, :]
            odd_mats = left_tensors[:, 1:floor_n:2, :, :]
            leftover = left_tensors[:, floor_n:, :, :]

            # Batch multiply everything, append remainder
            left_tensors = even_mats @ odd_mats
            left_tensors = torch.cat((left_tensors, leftover), dim=1)
            n_mats = left_tensors.shape[1]
        left_tensors = left_tensors.squeeze(1)

        n_mats = right_tensors.shape[1]
        while n_mats > 1:
            half_n = n_mats // 2
            floor_n = half_n * 2

            # Split matrices up into even and odd numbers (maybe w/ leftover)
            even_mats = right_tensors[:, 0:floor_n:2, :, :]
            odd_mats = right_tensors[:, 1:floor_n:2, :, :]
            leftover = right_tensors[:, floor_n:, :, :]

            # Batch multiply everything, append remainder
            right_tensors = even_mats @ odd_mats
            right_tensors = torch.cat((right_tensors, leftover), dim=1)
            n_mats = right_tensors.shape[1]
        right_tensors = right_tensors.squeeze(1)
        #print('Pairwise contraction:', time.time() - start)

        start = time.time()
        left_node_tensor = pad(self.left_node.tensor,
                               pad=(0, left_tensors.shape[1] - self.left_node.tensor.shape[1]))
        #print('Pad left node tensor:', time.time() - start)
        start = time.time()
        left_result = opt_einsum.contract('bi,il,blr->br', *[self.left_node.neighbours('input').tensor,
                                                             left_node_tensor,
                                                             left_tensors])
        #print('Contract left node:', time.time() - start)
        right_node_tensor = pad(self.right_node.tensor,
                                pad=(0, 0, 0, right_tensors.shape[2] - self.right_node.tensor.shape[0]))
        right_result = opt_einsum.contract('blr,ri,bi->bl', *[right_tensors,
                                                              right_node_tensor,
                                                              self.right_node.neighbours('input').tensor])
        start = time.time()
        output_tensor = pad(self.output_node.tensor,
                            pad=(0, right_result.shape[1] - self.output_node.tensor.shape[2],
                                 0, 0,
                                 0, left_result.shape[1] - self.output_node.tensor.shape[0]))
        #print('Pad output node tensor:', time.time() - start)
        start = time.time()
        result = opt_einsum.contract('bl,lor,br->bo', *[left_result, output_tensor, right_result])
        #print('Final contraction:', time.time() - start)

        return result

    def _update_current_op_nodes(self) -> None:
        for node in self.nodes.values():
            if not node.permanent and node.current_op:
                node.current_op = False

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data: tensor with shape batch x feature x n_features
        """
        if not self.data_nodes:
            #start = time.time()
            # All data tensors have the same batch size
            self.set_data_nodes(batch_sizes=[data.shape[0]])
            self._add_data(data=data.unbind(2))
            self._permanent_nodes = list(self.nodes.values())  # TODO: this does nothing
            #self.initialize2()
            #print('Add data:', time.time() - start)
        else:
            #start = time.time()
            self._add_data(data=data.unbind(2))
            end = time.time()
            #print('Add data:', end - start)
        #start = time.time()
        output = self.contract().tensor  #self.contract2()
        #print('Contract:', time.time() - start)
        #print()
        self._update_current_op_nodes()
        #self.num_current_op_nodes = []
        return output
