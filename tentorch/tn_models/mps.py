"""
MPS class
"""

from typing import (Union, Optional, Sequence,
                    Text, List, Tuple)

import torch

from tentorch.network_components import (AbstractNode, Node, ParamNode,
                                         AbstractEdge)
from tentorch.network_components import TensorNetwork

from tentorch.node_operations import einsum, stacked_einsum

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
                                           network=self,
                                           param_edges=self.param_bond())
            else:
                self.left_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[0], self.d_bond[0]),
                                           axes_names=('left', 'input', 'right'),
                                           name='left_node',
                                           network=self,
                                           param_edges=self.param_bond())
                periodic_edge = self.left_node['left']
        else:
            self.left_node = None

        if self.l_position > 1:
            # Left environment
            for i in range(1, self.l_position):
                node = ParamNode(shape=(self.d_bond[i - 1], self.d_phys[i], self.d_bond[i]),
                                 axes_names=('left', 'input', 'right'),
                                 name='left_env_node',
                                 network=self,
                                 param_edges=self.param_bond())
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
                                             network=self,
                                             param_edges=self.param_bond())
            else:
                self.output_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[0], self.d_bond[0]),
                                             axes_names=('left', 'output', 'right'),
                                             name='output_node',
                                             network=self,
                                             param_edges=self.param_bond())
                periodic_edge = self.output_node['left']

        if self.l_position == self.n_sites - 1:
            if self.n_sites - 1 != 0:
                if self.boundary == 'obc':
                    self.output_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[-1]),
                                                 axes_names=('left', 'output'),
                                                 name='output_node',
                                                 network=self,
                                                 param_edges=self.param_bond())
                else:
                    self.output_node = ParamNode(shape=(self.d_bond[-2], self.d_phys[-1], self.d_bond[-1]),
                                                 axes_names=('left', 'output', 'right'),
                                                 name='output_node',
                                                 network=self,
                                                 param_edges=self.param_bond())
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
                                         network=self,
                                         param_edges=self.param_bond())
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
                                 network=self,
                                 param_edges=self.param_bond())
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
                                            network=self,
                                            param_edges=self.param_bond())
            else:
                self.right_node = ParamNode(shape=(self.d_bond[-2], self.d_phys[-1], self.d_bond[-1]),
                                            axes_names=('left', 'input', 'right'),
                                            name='right_node',
                                            network=self,
                                            param_edges=self.param_bond())
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
                    aux = torch.randn(tensor.shape[0], tensor.shape[2]) * std + torch.eye(tensor.shape[0])
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

    def _input_contraction(self) -> Tuple[List[Node], List[Node]]:
        left_result = None
        right_result = None
        if self.left_env:
            left_env_data = list(map(lambda node: node.neighbours('input'), self.left_env))
            left_result = stacked_einsum('lir,bi->lbr', self.left_env, left_env_data)
        if self.right_env:
            right_env_data = list(map(lambda node: node.neighbours('input'), self.right_env))
            right_result = stacked_einsum('lir,bi->lbr', self.right_env, right_env_data)
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
    def _pairwise_contraction(nodes: List[Node]) -> Node:
        """
        Contract a sequence of MPS tensors that do not have
        parameterized bond dimensions, making the operation
        more efficient (from jemisjoki/TorchMPS)
        """
        length = len(nodes)
        while length > 1:
            odd_length = length % 2 == 1
            half_length = length // 2
            nice_length = 2 * half_length

            even_nodes = nodes[0:nice_length:2]
            odd_nodes = nodes[1:nice_length:2]
            leftover = nodes[nice_length:]

            nodes = stacked_einsum('ibj,jbk->ibk', even_nodes, odd_nodes)
            nodes += leftover
            length = half_length + int(odd_length)

        return nodes[0]

    def contract(self) -> Node:
        start = time.time()
        left_env, right_env = self._input_contraction()
        print('Input:', time.time() - start)
        
        # Operations of left environment
        left_list = []
        if self.left_node is not None:
            start = time.time()
            if self.boundary == 'obc':
                left_node = einsum('ir,bi->br', self.left_node, self.left_node.neighbours('input'))
            else:
                left_node = einsum('lir,bi->lbr', self.left_node, self.left_node.neighbours('input'))
            left_list.append(left_node)
            print('Left node:', time.time() - start)
        if left_env is not None:
            start = time.time()
            if not self.param_bond() and self.same_d_phys() and self.same_d_bond():
                left_env_contracted = self._pairwise_contraction(left_env)
            else:
                left_env_contracted = self._inline_contraction(left_env)
            left_list.append(left_env_contracted)
            print('Left env:', time.time() - start)

        # Operations of right environment
        right_list = []
        if right_env is not None:
            start = time.time()
            if not self.param_bond() and self.same_d_phys() and self.same_d_bond():
                right_env_contracted = self._pairwise_contraction(right_env)
            else:
                right_env_contracted = self._inline_contraction(right_env)
            right_list.append(right_env_contracted)
            print('Right env:', time.time() - start)
        if self.right_node is not None:
            start = time.time()
            if self.boundary == 'obc':
                right_node = einsum('li,bi->lb', self.right_node, self.right_node.neighbours('input'))
            else:
                right_node = einsum('lir,bi->lbr', self.right_node, self.right_node.neighbours('input'))
            right_list.append(right_node)
            print('Right node:', time.time() - start)

        start = time.time()
        result_list = left_list + [self.output_node] + right_list
        result = result_list[0]
        for node in result_list[1:]:
            result @= node
        print('Final contraction:', time.time() - start)

        # Clean intermediate nodes
        #mps_nodes = list(self.nodes.values())
        #for node in mps_nodes:
        #    if node not in self.permanent_nodes:
        #        self.delete_node(node)

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
            start = time.time()
            # All data tensors have the same batch size
            self.set_data_nodes(batch_sizes=[data.shape[0]])
            self._add_data(data=data.unbind(2))
            self._permanent_nodes = list(self.nodes.values())
            #self.initialize2()
            print('Add data:', time.time() - start)
        else:
            start = time.time()
            self._add_data(data=data.unbind(2))
            end = time.time()
            print('Add data:', end - start)
        start = time.time()
        output = self.contract().tensor
        print('Contract:', time.time() - start)
        print()
        self._update_current_op_nodes()
        #self.num_current_op_nodes = []
        return output
