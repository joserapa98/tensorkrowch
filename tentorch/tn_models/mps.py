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

import tentorch as tn

import opt_einsum
import math

import time
from torchviz import make_dot

PRINT_MODE = False


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

        # Save references to _leaf nodes
        # TODO: This should be in all TN
        self._permanent_nodes = []

        permanent_nodes = []
        if self.left_node is not None:
            permanent_nodes.append(self.left_node)
        permanent_nodes += self.left_env + [self.output_node] + self.right_env
        if self.right_node is not None:
            permanent_nodes.append(self.right_node)
        self._permanent_nodes = permanent_nodes

        # TODO: esto deber'ia crearse y llevarse como variable,
        #  y luego ya si eso actualizarse
        self._same_d_bond = self.same_d_bond()
        self._same_d_phys = self.same_d_phys()

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
        start = time.time()
        env_nodes = self.left_env + self.right_env
        for i, _ in enumerate(env_nodes[:-1]):
            if env_nodes[i].size() != env_nodes[i + 1].size():
                if PRINT_MODE: print('\t\tSame d bond:', time.time() - start)
                return False
        if PRINT_MODE: print('\t\tSame d bond:', time.time() - start)
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
                self.left_node = ParamNode(shape=(self.d_phys[0], self.d_bond[0]), axes_names=('input', 'right'),
                                           name='left_node', network=self)
                                           #param_edges=self.param_bond())
            else:
                self.left_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[0], self.d_bond[0]),
                                           axes_names=('left', 'input', 'right'), name='left_node', network=self)
                                           #param_edges=self.param_bond())
                periodic_edge = self.left_node['left']
        else:
            self.left_node = None

        if self.l_position > 1:
            # Left environment
            for i in range(1, self.l_position):
                node = ParamNode(shape=(self.d_bond[i - 1], self.d_phys[i], self.d_bond[i]),
                                 axes_names=('left', 'input', 'right'), name='left_env_node', network=self)
                                 #param_edges=self.param_bond())
                self.left_env.append(node)
                if i == 1:
                    self.left_node['right'] ^ self.left_env[-1]['left']
                else:
                    self.left_env[-2]['right'] ^ self.left_env[-1]['left']

        # Output
        if self.l_position == 0:
            if self.boundary == 'obc':
                self.output_node = ParamNode(shape=(self.d_phys[0], self.d_bond[0]), axes_names=('output', 'right'),
                                             name='output_node', network=self)
                                             #param_edges=self.param_bond())
            else:
                self.output_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[0], self.d_bond[0]),
                                             axes_names=('left', 'output', 'right'), name='output_node', network=self)
                                             #param_edges=self.param_bond())
                periodic_edge = self.output_node['left']

        if self.l_position == self.n_sites - 1:
            if self.n_sites - 1 != 0:
                if self.boundary == 'obc':
                    self.output_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[-1]),
                                                 axes_names=('left', 'output'), name='output_node', network=self)
                                                 #param_edges=self.param_bond())
                else:
                    self.output_node = ParamNode(shape=(self.d_bond[-2], self.d_phys[-1], self.d_bond[-1]),
                                                 axes_names=('left', 'output', 'right'), name='output_node',
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
                                                self.d_bond[self.l_position]), axes_names=('left', 'output', 'right'),
                                         name='output_node', network=self)
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
                                 axes_names=('left', 'input', 'right'), name='right_env_node', network=self)
                                 #param_edges=self.param_bond())
                self.right_env.append(node)
                if i == self.l_position + 1:
                    self.output_node['right'] ^ self.right_env[-1]['left']
                else:
                    self.right_env[-2]['right'] ^ self.right_env[-1]['left']

        if self.l_position < self.n_sites - 1:
            # Right node
            if self.boundary == 'obc':
                self.right_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[-1]), axes_names=('left', 'input'),
                                            name='right_node', network=self)
                                            #param_edges=self.param_bond())
            else:
                self.right_node = ParamNode(shape=(self.d_bond[-2], self.d_phys[-1], self.d_bond[-1]),
                                            axes_names=('left', 'input', 'right'), name='right_node', network=self)
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

        if self._same_d_bond:  # TODO: cuidado, era self.same_d_bond()
            # start = time.time()
            if self.left_env + self.right_env:
                # env_data = list(map(lambda node: node.neighbours('input'), self.left_env + self.right_env))
                # print('\t\tFind data:', time.time() - start)
                start = time.time()
                stack = tn.stack(self.left_env + self.right_env)
                stack_data = tn.stack(self.env_data)
                stack['input'] ^ stack_data['feature']
                result = stack @ stack_data
                result = result.permute((0, 1, 3, 2))
                result = tn.unbind(result)
                # left_result = []
                # right_result = []
                # if self.left_env:
                #     left_stack = tn.stack(self.left_env)
                #     left_stack_data = tn.stack(self.left_env_data)
                #     left_stack['input'] ^ left_stack_data['feature']
                #     left_result = left_stack @ left_stack_data
                #     left_result = left_result.permute((0, 1, 3, 2))
                #     left_result = tn.unbind(left_result)
                # if self.right_env:
                #     right_stack = tn.stack(self.right_env)
                #     right_stack_data = tn.stack(self.right_env_data)
                #     right_stack['input'] ^ right_stack_data['feature']
                #     right_result = right_stack @ right_stack_data
                #     right_result = right_result.permute((0, 1, 3, 2))
                #     right_result = tn.unbind(right_result)
                #result = stacked_einsum('lir,bi->lbr', self.left_env + self.right_env, env_data)
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
    def _inline_contraction(nodes: List[Node], left) -> Node:
        """
        Contract a sequence of MPS tensors that have
        parameterized bond dimensions
        """
        # if left:
        #     vec = nodes[0].tensor.unsqueeze(1)
        #     for mat in nodes[1:]:
        #         vec = vec @ mat.tensor.permute(1, 0, 2)
        #     return vec
        # else:
        #     vec = nodes[0].tensor.permute(1, 0).unsqueeze(2)
        #     for mat in nodes[1:]:
        #         vec = mat.tensor.permute(1, 0, 2) @ vec
        #     return vec

        result_node = nodes[0]
        for node in nodes[1:]:
            start = time.time()
            result_node @= node
            if PRINT_MODE: print('\t\t\tMatrix contraction:', time.time() - start)
        return result_node

    @staticmethod
    def _aux_pairwise2(nodes: List[Node]) -> List[Node]:
        length = len(nodes)
        start_total = time.time()
        while length > 1:
            odd_length = length % 2 == 1
            half_length = length // 2
            nice_length = 2 * half_length

            start = time.time()
            even_nodes = nodes[0:nice_length:2]
            odd_nodes = nodes[1:nice_length:2]
            leftover = nodes[nice_length:]
            if PRINT_MODE: print('\t\t\tSelect nodes:', time.time() - start)

            start = time.time()
            stack1 = tn.stack(even_nodes)
            if PRINT_MODE: print('\t\t\tStack even nodes:', time.time() - start)
            start = time.time()
            stack2 = tn.stack(odd_nodes)
            if PRINT_MODE: print('\t\t\tStack odd nodes:', time.time() - start)
            start = time.time()
            stack1['right'] ^ stack2['left']
            if PRINT_MODE: print('\t\t\tConnect stacks:', time.time() - start)
            start = time.time()
            nodes = stack1 @ stack2
            if PRINT_MODE: print('\t\t\tContract stacks:', time.time() - start)
            start = time.time()
            nodes = tn.unbind(nodes)
            if PRINT_MODE: print('\t\t\tUnbind stacks:', time.time() - start)

            nodes += leftover
            length = half_length + int(odd_length)
        if PRINT_MODE: print('\t\tPairwise contraction:', time.time() - start_total)

        return nodes

    @staticmethod
    def _aux_pairwise(nodes: List[Node]) -> List[Node]:
        length = len(nodes)
        if length > 1:
            lst_lengths = [0]
            while length > 0:
                exp = math.floor(math.log(length) / math.log(2))
                lst_lengths.append(lst_lengths[-1] + 2**exp)
                length -= 2**exp

            aux_nodes = []
            for i in range(len(lst_lengths) - 1):
                aux_nodes.append(MPS._aux_pairwise2(nodes[lst_lengths[i]:lst_lengths[i+1]])[0])

            return MPS._aux_pairwise(aux_nodes)

        return nodes

    def _aux_pairwise3(self, nodes: List[Node], left) -> List[Node]:
        length = len(nodes)
        aux_nodes = nodes
        if length > 1:
            lst_lengths = [0]
            while length > 0:
                exp = math.floor(math.log(length) / math.log(2))
                lst_lengths.append(lst_lengths[-1] + 2 ** exp)
                length -= 2 ** exp

            aux_nodes = []
            for i in range(len(lst_lengths) - 1):
                aux_nodes.append(MPS._aux_pairwise2(nodes[lst_lengths[i]:lst_lengths[i + 1]])[0])

        if left:
            left_node = (self.left_node @ self.left_node.neighbours('input')).permute((1, 0))
            result = [self._inline_contraction([left_node] + aux_nodes, left)]
        else:
            right_node = self.right_node @ self.right_node.neighbours('input')
            lst = aux_nodes + [right_node]
            lst.reverse()
            result = [self._inline_contraction(lst, left)]

        return result

    def _aux_pairwise4(self, nodes: List[Node], left) -> List[Node]:
        length = len(nodes)
        aux_nodes = nodes
        if length > 1:
            leftovers = []
            start_total = time.time()
            while length > 1:
                half_length = length // 2
                nice_length = 2 * half_length

                start = time.time()
                even_nodes = aux_nodes[0:nice_length:2]
                odd_nodes = aux_nodes[1:nice_length:2]
                leftovers = aux_nodes[nice_length:] + leftovers
                if PRINT_MODE: print('\t\t\tSelect nodes:', time.time() - start)

                start = time.time()
                stack1 = tn.stack(even_nodes)
                if PRINT_MODE: print('\t\t\tStack even nodes:', time.time() - start)
                start = time.time()
                stack2 = tn.stack(odd_nodes)
                if PRINT_MODE: print('\t\t\tStack odd nodes:', time.time() - start)
                start = time.time()
                stack1['right'] ^ stack2['left']
                if PRINT_MODE: print('\t\t\tConnect stacks:', time.time() - start)
                start = time.time()
                aux_nodes = stack1 @ stack2
                if PRINT_MODE: print('\t\t\tContract stacks:', time.time() - start)
                start = time.time()
                aux_nodes = tn.unbind(aux_nodes)
                if PRINT_MODE: print('\t\t\tUnbind stacks:', time.time() - start)

                length = half_length
            if PRINT_MODE: print('\t\tPairwise contraction:', time.time() - start_total)

            aux_nodes = aux_nodes + leftovers

        if left:
            left_node = (self.left_node @ self.left_node.neighbours('input')).permute((1, 0))
            result = [self._inline_contraction([left_node] + aux_nodes, left)]
        else:
            right_node = self.right_node @ self.right_node.neighbours('input')
            lst = aux_nodes + [right_node]
            lst.reverse()
            result = [self._inline_contraction(lst, left)]

        return result

    def _aux_pairwise5(self, nodes: List[Node], left) -> List[Node]:
        length = len(nodes)
        aux_nodes = nodes
        if length > 1:
            leftovers = []
            start_total = time.time()
            while length > 1:
                half_length = length // 2
                nice_length = 2 * half_length

                start = time.time()
                even_nodes = aux_nodes[0:nice_length:2]
                odd_nodes = aux_nodes[1:nice_length:2]
                leftovers = aux_nodes[nice_length:] + leftovers
                if PRINT_MODE: print('\t\t\tSelect nodes:', time.time() - start)

                start = time.time()
                stack1 = tn.stack(even_nodes)
                if PRINT_MODE: print('\t\t\tStack even nodes:', time.time() - start)
                start = time.time()
                stack2 = tn.stack(odd_nodes)
                if PRINT_MODE: print('\t\t\tStack odd nodes:', time.time() - start)
                start = time.time()
                stack1['right'] ^ stack2['left']
                if PRINT_MODE: print('\t\t\tConnect stacks:', time.time() - start)
                start = time.time()
                aux_nodes = stack1 @ stack2
                if PRINT_MODE: print('\t\t\tContract stacks:', time.time() - start)
                start = time.time()
                aux_nodes = tn.unbind(aux_nodes)
                if PRINT_MODE: print('\t\t\tUnbind stacks:', time.time() - start)

                length = half_length
            if PRINT_MODE: print('\t\tPairwise contraction:', time.time() - start_total)

            aux_nodes = aux_nodes + leftovers
            return self._aux_pairwise5(aux_nodes, left)

        if left:
            left_node = (self.left_node @ self.left_node.neighbours('input')).permute((1, 0))
            result = [self._inline_contraction([left_node] + aux_nodes, left)]
        else:
            right_node = self.right_node @ self.right_node.neighbours('input')
            lst = aux_nodes + [right_node]
            lst.reverse()
            result = [self._inline_contraction(lst, left)]

        return result

    def _aux_aux_pairwise6(self, nodes: List[Node]) -> Tuple[List[Node], List[Node]]:
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
            stack1 = tn.stack(even_nodes)
            if PRINT_MODE: print('\t\t\tStack even nodes:', time.time() - start)
            start = time.time()
            stack2 = tn.stack(odd_nodes)
            if PRINT_MODE: print('\t\t\tStack odd nodes:', time.time() - start)
            start = time.time()
            stack1['right'] ^ stack2['left']
            if PRINT_MODE: print('\t\t\tConnect stacks:', time.time() - start)
            start = time.time()
            aux_nodes = stack1 @ stack2
            if PRINT_MODE: print('\t\t\tContract stacks:', time.time() - start)
            start = time.time()
            aux_nodes = tn.unbind(aux_nodes)
            if PRINT_MODE: print('\t\t\tUnbind stacks:', time.time() - start)

            if PRINT_MODE: print('\t\tPairwise contraction:', time.time() - start_total)

            return aux_nodes, leftover
        return nodes, []

    def _aux_pairwise6(self, left_nodes: List[Node], right_nodes: List[Node]) -> Tuple[List[Node], List[Node]]:
        left_length = len(left_nodes)
        left_aux_nodes = left_nodes
        right_length = len(right_nodes)
        right_aux_nodes = right_nodes
        if left_length > 1 or right_length > 1:
            left_leftovers = []
            right_leftovers = []
            while left_length > 1 or right_length > 1:
                aux1, aux2 = self._aux_aux_pairwise6(left_aux_nodes)
                left_aux_nodes = aux1
                left_leftovers = aux2 + left_leftovers
                left_length = len(aux1)

                aux1, aux2 = self._aux_aux_pairwise6(right_aux_nodes)
                right_aux_nodes = aux1
                right_leftovers = aux2 + right_leftovers
                right_length = len(aux1)

            left_aux_nodes = left_aux_nodes + left_leftovers
            right_aux_nodes = right_aux_nodes + right_leftovers
            return self._aux_pairwise6(left_aux_nodes, right_aux_nodes)

        # left_node = (self.left_node @ self.left_node.neighbours('input'))
        # left_env = left_node @ left_aux_nodes[0]
        #
        # right_node = self.right_node @ self.right_node.neighbours('input')
        # right_env = (right_aux_nodes[0] @ right_node).permute((1, 0))

        left_node = (self.left_node @ self.left_node.neighbours('input')).permute((1, 0))
        left_env = [self._inline_contraction([left_node] + left_aux_nodes, True)]

        right_node = self.right_node @ self.right_node.neighbours('input')
        lst = right_aux_nodes + [right_node]
        lst.reverse()
        right_env = [self._inline_contraction(lst, False)]

        return left_env, right_env

    # @staticmethod
    def _pairwise_contraction(self, left_nodes: List[Node], right_nodes: List[Node]) -> Tuple[Node, Node]:
        """
        Contract a sequence of MPS tensors that do not have
        parameterized bond dimensions, making the operation
        more efficient (from jemisjoki/TorchMPS)
        """
        # TODO: rewrite this (simplify)
        # TODO: hacer algoritmo que encuentre subconjuntos de longitud potencias de 2 donde se vaya
        #  haciendo las operaciones reutilizando siempre la misma pila, y al final con los que queden
        #  solo 1 o 2 operaciones que no reutilizan pila (pensar con formula si esto seria ventaja)
        # left_nodes = MPS._aux_pairwise2(left_nodes)
        # right_nodes = MPS._aux_pairwise2(right_nodes)
        # left_nodes = self._aux_pairwise5(left_nodes, True)
        # right_nodes = self._aux_pairwise5(right_nodes, False)
        left_nodes, right_nodes = self._aux_pairwise6(left_nodes, right_nodes)

        # length_left = len(left_nodes)
        # length_right = len(right_nodes)
        # while (length_left > 1) or (length_right > 1):
        #     if (length_left > 1) and (length_right > 1):
        #         odd_length_left = length_left % 2 == 1
        #         half_length_left = length_left // 2
        #         nice_length_left = 2 * half_length_left
        #
        #         left_even_nodes = left_nodes[0:nice_length_left:2]
        #         left_odd_nodes = left_nodes[1:nice_length_left:2]
        #         left_leftover = left_nodes[nice_length_left:]
        #
        #         odd_length_right = length_right % 2 == 1
        #         half_length_right = length_right // 2
        #         nice_length_right = 2 * half_length_right
        #
        #         right_even_nodes = right_nodes[0:nice_length_right:2]
        #         right_odd_nodes = right_nodes[1:nice_length_right:2]
        #         right_leftover = right_nodes[nice_length_right:]
        #
        #         stack1 = tn.stack(left_even_nodes + right_even_nodes)
        #         stack2 = tn.stack(left_odd_nodes + right_odd_nodes)
        #         stack1['right'] ^ stack2['left']
        #         nodes = stack1 @ stack2
        #         nodes = tn.unbind(nodes)
        #
        #         # nodes = stacked_einsum('ibj,jbk->ibk',
        #         #                        left_even_nodes + right_even_nodes,
        #         #                        left_odd_nodes + right_odd_nodes)
        #
        #         left_nodes = nodes[:half_length_left]
        #         left_nodes += left_leftover
        #         length_left = half_length_left + int(odd_length_left)
        #
        #         right_nodes = nodes[half_length_left:]
        #         right_nodes += right_leftover
        #         length_right = half_length_right + int(odd_length_right)
        #
        #     elif length_left > 1:
        #     if length_left > 1:
        #         odd_length_left = length_left % 2 == 1
        #         half_length_left = length_left // 2
        #         nice_length_left = 2 * half_length_left
        #
        #         left_even_nodes = left_nodes[0:nice_length_left:2]
        #         left_odd_nodes = left_nodes[1:nice_length_left:2]
        #         left_leftover = left_nodes[nice_length_left:]
        #
        #         stack1 = tn.stack(left_even_nodes)
        #         stack2 = tn.stack(left_odd_nodes)
        #         stack1['right'] ^ stack2['left']
        #         nodes = stack1 @ stack2
        #         nodes = tn.unbind(nodes)
        #
        #         # nodes = stacked_einsum('ibj,jbk->ibk',
        #         #                        left_even_nodes,
        #         #                        left_odd_nodes)
        #
        #         left_nodes = nodes
        #         left_nodes += left_leftover
        #         length_left = half_length_left + int(odd_length_left)
        #
        #     else:
        #     if length_right > 1:
        #         odd_length_right = length_right % 2 == 1
        #         half_length_right = length_right // 2
        #         nice_length_right = 2 * half_length_right
        #
        #         right_even_nodes = right_nodes[0:nice_length_right:2]
        #         right_odd_nodes = right_nodes[1:nice_length_right:2]
        #         right_leftover = right_nodes[nice_length_right:]
        #
        #         stack1 = tn.stack(right_even_nodes)
        #         stack2 = tn.stack(right_odd_nodes)
        #         stack1['right'] ^ stack2['left']
        #         nodes = stack1 @ stack2
        #         nodes = tn.unbind(nodes)
        #
        #         # nodes = stacked_einsum('ibj,jbk->ibk',
        #         #                        right_even_nodes,
        #         #                        right_odd_nodes)
        #
        #         right_nodes = nodes
        #         right_nodes += right_leftover
        #         length_right = half_length_right + int(odd_length_right)
        #
        # start_total = time.time()
        # while length_left > 1:
        #     odd_length_left = length_left % 2 == 1
        #     half_length_left = length_left // 2
        #     nice_length_left = 2 * half_length_left
        #
        #     start = time.time()
        #     left_even_nodes = left_nodes[0:nice_length_left:2]
        #     left_odd_nodes = left_nodes[1:nice_length_left:2]
        #     left_leftover = left_nodes[nice_length_left:]
        #     if PRINT_MODE: print('\t\t\tSelect nodes:', time.time() - start)
        #
        #     start = time.time()
        #     stack1 = tn.stack(left_even_nodes)
        #     if PRINT_MODE: print('\t\t\tStack even nodes:', time.time() - start)
        #     start = time.time()
        #     stack2 = tn.stack(left_odd_nodes)
        #     if PRINT_MODE: print('\t\t\tStack odd nodes:', time.time() - start)
        #     start = time.time()
        #     stack1['right'] ^ stack2['left']
        #     if PRINT_MODE: print('\t\t\tConnect stacks:', time.time() - start)
        #     start = time.time()
        #     nodes = stack1 @ stack2
        #     if PRINT_MODE: print('\t\t\tContract stacks:', time.time() - start)
        #     start = time.time()
        #     nodes = tn.unbind(nodes)
        #     if PRINT_MODE: print('\t\t\tUnbind stacks:', time.time() - start)
        #
        #     left_nodes = nodes
        #     left_nodes += left_leftover
        #     length_left = half_length_left + int(odd_length_left)
        # if PRINT_MODE: print('\t\tLeft pairwise contraction:', time.time() - start_total)
        #
        # start = time.time()
        # while length_right > 1:
        #     odd_length_right = length_right % 2 == 1
        #     half_length_right = length_right // 2
        #     nice_length_right = 2 * half_length_right
        #
        #     right_even_nodes = right_nodes[0:nice_length_right:2]
        #     right_odd_nodes = right_nodes[1:nice_length_right:2]
        #     right_leftover = right_nodes[nice_length_right:]
        #
        #     stack1 = tn.stack(right_even_nodes)
        #     stack2 = tn.stack(right_odd_nodes)
        #     stack1['right'] ^ stack2['left']
        #     nodes = stack1 @ stack2
        #     nodes = tn.unbind(nodes)
        #
        #     right_nodes = nodes
        #     right_nodes += right_leftover
        #     length_right = half_length_right + int(odd_length_right)
        # if PRINT_MODE: print('\t\tRight pairwise contraction:', time.time() - start)

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
        if PRINT_MODE: print('\tInput:', time.time() - start)

        start = time.time()
        # TODO: cuidado, era self.same_d_phys() y self.same_d_bond()
        if not self.param_bond() and self._same_d_phys and self._same_d_bond:
            left_env_contracted, right_env_contracted = self._pairwise_contraction(left_env, right_env)
        else:
            # TODO: if left_node/right_node is not None
            left_env_contracted = None
            right_env_contracted = None
            if left_env:
                left_node = (self.left_node @ self.left_node.neighbours('input')).permute((1, 0))
                left_env_contracted = self._inline_contraction([left_node] + left_env, True)
                # aux = self.nodes['permute_node_0'].tensor[:len(left_env)].unbind()
                # left_env_contracted = self._inline_contraction([left_node.tensor] + list(aux), True)
            if right_env:
                right_node = self.right_node @ self.right_node.neighbours('input')
                lst = right_env + [right_node]
                # lst = list(self.nodes['permute_node_0'].tensor[len(left_env):].unbind()) + [right_node.tensor]
                lst.reverse()
                right_env_contracted = self._inline_contraction(lst, False)
        if PRINT_MODE: print('\tMatrices contraction:', time.time() - start)

        # result = opt_einsum.contract('bxl,lor,bry->bxoy',
        #                              left_env_contracted,
        #                              self.output_node.tensor,
        #                              right_env_contracted).squeeze(1).squeeze(2)
        # return result  # TODO: inline solo tensor

        result = opt_einsum.contract('bl,lor,br->bo',
                                     left_env_contracted.tensor,
                                     self.output_node.tensor,
                                     right_env_contracted.tensor)
        return result
        
        # Operations of left environment
        left_list = []
        if self.left_node is not None:
            start = time.time()
            if self.boundary == 'obc':
                left_node = (self.left_node @ self.left_node.neighbours('input')).permute((1, 0))
                # left_node = einsum('ir,bi->br', self.left_node, self.left_node.neighbours('input'))
            else:
                left_node = (self.left_node @ self.left_node.neighbours('input')).permute((0, 2, 1))
                # left_node = einsum('lir,bi->lbr', self.left_node, self.left_node.neighbours('input'))
            left_list.append(left_node)
            if PRINT_MODE: print('\tLeft node:', time.time() - start)
        if left_env_contracted is not None:
            left_list.append(left_env_contracted)  # new
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
                right_node = self.right_node @ self.right_node.neighbours('input')
                # right_node = einsum('li,bi->lb', self.right_node, self.right_node.neighbours('input'))
            else:
                right_node = (self.right_node @ self.right_node.neighbours('input')).permute((0, 2, 1))
                # right_node = einsum('lir,bi->lbr', self.right_node, self.right_node.neighbours('input'))
            right_list.append(right_node)
            if PRINT_MODE: print('\tRight node:', time.time() - start)

        start = time.time()
        result_list = left_list + [self.output_node] + right_list
        result = result_list[0]
        for node in result_list[1:]:
            result @= node
        if PRINT_MODE: print('\tFinal contraction:', time.time() - start)

        # Clean intermediate nodes
        #mps_nodes = list(self.nodes.values())
        #for node in mps_nodes:
        #    if node not in self.permanent_nodes:
        #        self.delete_node(node)

        return result.tensor

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
            if not node._leaf and node.current_op:
                node.current_op = False

    # def forward(self, data: torch.Tensor) -> torch.Tensor:
    #     """
    #     Parameters
    #     ----------
    #     data: tensor with shape batch x feature x n_features
    #     """
    #     if not self.data_nodes:
    #         start = time.time()
    #         # All data tensors have the same batch size
    #         self.set_data_nodes(batch_sizes=[data.shape[0]])
    #         if self.left_env + self.right_env:
    #             self.env_data = list(map(lambda node: node.neighbours('input'), self.left_env + self.right_env))
    #         # self._add_data(data=data.unbind(2))
    #         self._add_data(data=data.permute(2, 0, 1))
    #         self._permanent_nodes = list(self.nodes.values())  # TODO: this does nothing
    #         #self.initialize2()
    #         # print('Add data:', time.time() - start)
    #     else:
    #         start = time.time()
    #         # self._add_data(data=data.unbind(2))
    #         self._add_data(data=data.permute(2, 0, 1))
    #         end = time.time()
    #         # print('Add data:', end - start)
    #     start = time.time()
    #     output = self.contract().tensor  #self.contract2()
    #     # print('Contract:', time.time() - start)
    #     # print()
    #     #self._update_current_op_nodes()
    #     #self.num_current_op_nodes = []
    #     return output

    def embedding(self, image: torch.Tensor) -> torch.Tensor:
        # return torch.stack([image, 1 - image], dim=1).squeeze(0)
        return torch.stack([torch.ones_like(image), image, 1 - image], dim=1)#.squeeze(0)

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
            if self.left_env + self.right_env:
                self.env_data = list(map(lambda node: node.neighbours('input'), self.left_env + self.right_env))
            # TODO: usar 2 lados left y right en input contraction tarda lo mismo que solo 1 lado
            # if self.left_env:
            #     self.left_env_data = list(map(lambda node: node.neighbours('input'), self.left_env))
            # if self.right_env:
            #     self.right_env_data = list(map(lambda node: node.neighbours('input'), self.right_env))
            self._add_data(data=self.embedding(data).permute(2, 0, 1))
            if PRINT_MODE: print('Add data:', time.time() - start)
            start = time.time()
            output = self.contract()#.tensor  # self.contract2()
            # TODO: esto solo si output a la izda del todo
            # output = output.permute((1, 0))  # TODO: cuidado donde acaba el batch, tiene que acabar al principio
            if PRINT_MODE: print('Contract:', time.time() - start)
            if PRINT_MODE: print()

            self._seq_ops = []
            for op in self._list_ops:
                self._seq_ops.append((op[0], self._successors[op[0]][op[1]].kwargs))

            return output
        else:
            start = time.time()
            self._add_data(data=self.embedding(data).permute(2, 0, 1))
            end = time.time()
            if PRINT_MODE: print('Add data:', end - start)

            start = time.time()
            output = self.contract()  # self.contract2()
            if PRINT_MODE: print('Contract:', time.time() - start)
            if PRINT_MODE: print()

            # TODO: esta puede ser la forma gen'erica del forward, y solo hay que definir
            #  add_data y contract (para la primera vez)
            start_contract = time.time()
            # operations = self._list_ops
            # for i, op in enumerate(operations):
            #     if op[0] == 'permute':
            #         output = tn.permute(**self._successors['permute'][op[1]].kwargs)
            #     elif op[0] == 'tprod':
            #         output = tn.tprod(**self._successors['tprod'][op[1]].kwargs)
            #     elif op[0] == 'mul':
            #         output = tn.mul(**self._successors['mul'][op[1]].kwargs)
            #     elif op[0] == 'add':
            #         output = tn.add(**self._successors['add'][op[1]].kwargs)
            #     elif op[0] == 'sub':
            #         output = tn.sub(**self._successors['sub'][op[1]].kwargs)
            #     elif op[0] == 'contract_edges':
            #         output = tn.contract_edges(**self._successors['contract_edges'][op[1]].kwargs)
            #     elif op[0] == 'stack':
            #         output = tn.stack(**self._successors['stack'][op[1]].kwargs)
            #     elif op[0] == 'unbind':
            #         output = tn.unbind(**self._successors['unbind'][op[1]].kwargs)

            # stack_times = []
            # unbind_times = []
            # contract_edges_times = []
            #
            # operations = self._seq_ops
            # for i, op in enumerate(operations):
            #     if op[0] == 'permute':
            #         start = time.time()
            #         output = tn.permute(**op[1])
            #         if PRINT_MODE: print('permute:', time.time() - start)
            #
            #     elif op[0] == 'tprod':
            #         start = time.time()
            #         output = tn.tprod(**op[1])
            #         if PRINT_MODE: print('tprod:', time.time() - start)
            #
            #     elif op[0] == 'mul':
            #         start = time.time()
            #         output = tn.mul(**op[1])
            #         if PRINT_MODE: print('mul:', time.time() - start)
            #
            #     elif op[0] == 'add':
            #         start = time.time()
            #         output = tn.add(**op[1])
            #         if PRINT_MODE: print('add:', time.time() - start)
            #
            #     elif op[0] == 'sub':
            #         start = time.time()
            #         output = tn.sub(**op[1])
            #         if PRINT_MODE: print('sub:', time.time() - start)
            #
            #     elif op[0] == 'contract_edges':
            #         start = time.time()
            #         output = tn.contract_edges(**op[1])
            #         if PRINT_MODE:
            #             diff = time.time() - start
            #             print('contract_edges:', diff)
            #             contract_edges_times.append(diff)
            #
            #     elif op[0] == 'stack':
            #         start = time.time()
            #         output = tn.stack(**op[1])
            #         if PRINT_MODE:
            #             diff = time.time() - start
            #             print('stack:', diff)
            #             stack_times.append(diff)
            #
            #     elif op[0] == 'unbind':
            #         start = time.time()
            #         output = tn.unbind(**op[1])
            #         if PRINT_MODE:
            #             diff = time.time() - start
            #             print('unbind:', diff)
            #             unbind_times.append(diff)
            #
            # # TODO: Se tarda igual con _list_ops y _seq_ops
            #
            # if PRINT_MODE:
            #     print('Contract:', time.time() - start_contract)
            #     print('Check times sum:', torch.tensor(tn.CHECK_TIMES)[-len(operations):].sum())
            #     print('Stack times sum:', torch.tensor(stack_times).sum())
            #     print('Unbind times sum:', torch.tensor(unbind_times).sum())
            #     print('Contract edges times sum:', torch.tensor(contract_edges_times).sum())
            #     print()

            output = output#.tensor

            # TODO: esto solo si output a la izda del todo
            # output = output.permute((1, 0))  # TODO: cuidado donde acaba el batch, tiene que acabar al principio

            return output
