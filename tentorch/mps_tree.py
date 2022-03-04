"""
MPS class
"""

import warnings
from typing import (overload, Union, Optional,
                    Sequence, Text, List, Tuple)

import torch
import torch.nn as nn

from tentorch.network_components import (AbstractNode, Node, ParamNode,
                                         AbstractEdge, Edge, ParamEdge)
from tentorch.network_components import TensorNetwork

from tentorch.node_operations import einsum, stacked_einsum


# TODO: MPS -> contraemos resultados y luego hacemos delete_node(node) y
#  del node para eliminar los nodos intermedios de la red y borrar las
#  referencias a ellos para poder liberar memoria
# TODO: poner nombre "especial" a los nodos resultantes para deletearlos fácil
# TODO: move_l_position -> needs svd and qr to contract and split nodes

class MPS(TensorNetwork):

    def __init__(self,
                 n_sites: int,
                 d_phys: Union[int, Sequence[int]],
                 d_phys_l: int,
                 d_bond: Union[int, Sequence[int]],
                 l_position: Optional[int] = None,
                 boundary: Text = 'obc',
                 param_bond: bool = False) -> None:
        """
        Create an MPS module.

        Parameters
        ----------
        n_sites: number of sites, including the input and output sites
        d_phys: physic dimension
        d_phys_l: output dimension
        d_bond: bond dimension. If given as a sequence, the i-th bond
                dimension is always the dimension of the right edge of
                th i-th node
        l_position: position of output site
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
            self._d_phys = list(d_phys[:l_position]) + [d_phys_l] + \
                           list(d_phys[l_position:])
        elif isinstance(d_phys, int):
            self._d_phys = [d_phys] * l_position + [d_phys_l] + \
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

        # nodes
        self.left_node = None
        self.left_env = []
        self.output_node = None
        self.right_env = []
        self.right_node = None

        self._make_nodes()
        self.param_bond(set_param=param_bond)
        self.initialize()

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

        # Left
        if self.l_position > 0:
            # Left node
            if self.boundary == 'obc':
                self.left_node = ParamNode(shape=(self.d_phys[0], self.d_bond[0]),
                                           axes_names=('input', 'right'),
                                           name='mps_left_node',
                                           network=self,
                                           param_edges=self.param_bond())
            else:
                self.left_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[0], self.d_bond[0]),
                                           axes_names=('left', 'input', 'right'),
                                           name='mps_left_node',
                                           network=self,
                                           param_edges=self.param_bond())
                periodic_edge = self.left_node['left']

        if self.l_position > 1:
            # Left environment
            for i in range(1, self.l_position):
                node = ParamNode(shape=(self.d_bond[i - 1], self.d_phys[i], self.d_bond[i]),
                                 axes_names=('left', 'input', 'right'),
                                 name='mps_left_env',
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
                self.output = ParamNode(shape=(self.d_phys[0], self.d_bond[0]),
                                        axes_names=('output', 'right'),
                                        name='mps_output',
                                        network=self,
                                        param_edges=self.param_bond())
            else:
                self.output = ParamNode(shape=(self.d_bond[-1], self.d_phys[0], self.d_bond[0]),
                                        axes_names=('left', 'output', 'right'),
                                        name='mps_output',
                                        network=self,
                                        param_edges=self.param_bond())
                periodic_edge = self.output['left']

        if self.l_position == self.n_sites - 1:
            if self.n_sites - 1 != 0:
                if self.boundary == 'obc':
                    self.output = ParamNode(shape=(self.d_bond[-1], self.d_phys[-1]),
                                            axes_names=('left', 'output'),
                                            name='mps_output',
                                            network=self,
                                            param_edges=self.param_bond())
                else:
                    self.output = ParamNode(shape=(self.d_bond[-2], self.d_phys[-1], self.d_bond[-1]),
                                            axes_names=('left', 'output', 'right'),
                                            name='mps_output',
                                            network=self,
                                            param_edges=self.param_bond())
                    self.output['right'] ^ periodic_edge
            if self.left_env:
                self.left_env[-1]['right'] ^ self.output['left']

        if (self.l_position > 0) and (self.l_position < self.n_sites - 1):
            self.output = ParamNode(shape=(self.d_bond[self.l_position - 1],
                                           self.d_phys[self.l_position],
                                           self.d_bond[self.l_position]),
                                    axes_names=('left', 'output', 'right'),
                                    name='mps_output',
                                    network=self,
                                    param_edges=self.param_bond())
            if self.left_env:
                self.left_env[-1]['right'] ^ self.output['left']
            else:
                self.left_node['right'] ^ self.output['left']

        # Right
        if self.l_position < self.n_sites - 2:
            # Right environment
            for i in range(self.l_position + 1, self.n_sites - 1):
                node = ParamNode(shape=(self.d_bond[i - 1], self.d_phys[i], self.d_bond[i]),
                                 axes_names=('left', 'input', 'right'),
                                 name='mps_right_env',
                                 network=self,
                                 param_edges=self.param_bond())
                self.right_env.append(node)
                if i == self.l_position + 1:
                    self.output['right'] ^ self.right_env[-1]['left']
                else:
                    self.right_env[-2]['right'] ^ self.right_env[-1]['left']

        if self.l_position < self.n_sites - 1:
            # Right node
            if self.boundary == 'obc':
                self.right_node = ParamNode(shape=(self.d_bond[-1], self.d_phys[-1]),
                                            axes_names=('left', 'input'),
                                            name='mps_right_node',
                                            network=self,
                                            param_edges=self.param_bond())
            else:
                self.right_node = ParamNode(shape=(self.d_bond[-2], self.d_phys[-1], self.d_bond[-1]),
                                            axes_names=('left', 'input', 'right'),
                                            name='mps_right_node',
                                            network=self,
                                            param_edges=self.param_bond())
                self.right_node['right'] ^ periodic_edge
            if self.right_env:
                self.right_env[-1]['right'] ^ self.right_node['left']
            else:
                self.output['right'] ^ self.right_node['left']

    def initialize(self) -> None:
        for node in self.nodes.values():
            node.set_tensor(init_method='randn')

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

    # TODO: remove intermediate nodes
    def contract(self) -> Node:
        left_env, right_env = self._input_contraction()
        
        # Operations of left environment
        left_list = []
        if self.left_node is not None:
            if self.boundary == 'obc':
                left_node = einsum('ir,bi->br', self.left_node, self.left_node.neighbours('input'))
            else:
                left_node = einsum('lir,bi->lbr', self.left_node, self.left_node.neighbours('input'))
            left_list.append(left_node)
        if left_env is not None:
            if not self.param_bond() and self.same_d_phys() and self.same_d_bond():
                left_env_contracted = self._pairwise_contraction(left_env)
            else:
                left_env_contracted = self._inline_contraction(left_env)
            left_list.append(left_env_contracted)

        # Operations of right environment
        right_list = []
        if self.right_node is not None:
            if self.boundary == 'obc':
                right_node = einsum('li,bi->lb', self.right_node, self.right_node.neighbours('input'))
            else:
                right_node = einsum('lir,bi->lbr', self.right_node, self.right_node.neighbours('input'))
            right_list.append(right_node)
        if right_env is not None:
            if not self.param_bond() and self.same_d_phys() and self.same_d_bond():
                right_env_contracted = self._pairwise_contraction(right_env)
            else:
                right_env_contracted = self._inline_contraction(right_env)
            right_list.append(right_env_contracted)
            
        result_list = left_list + [self.output] + right_list
        result = result_list[0]
        for node in result_list[1:]:
            result @= node

        # Clean intermediate nodes
        permanent_nodes = []
        if self.left_node is not None:
            permanent_nodes.append(self.left_node)
        permanent_nodes += self.left_env + [self.output] + self.right_env
        if self.right_node is not None:
            permanent_nodes.append(self.right_node)
        permanent_nodes += list(self.data_nodes.values())

        mps_nodes = list(self.nodes.values())
        for node in mps_nodes:
            if node not in permanent_nodes:
                self.delete_node(node)

        return result

    def forward(self, data: Sequence[torch.Tensor]) -> torch.Tensor:
        self._add_data(data=data)
        return self.contract().tensor


class Tree(TensorNetwork):

    def __init__(self,
                 n_sites: int,
                 d_phys: int,
                 d_bond: int,
                 l_position: Optional[int] = None):
        super().__init__()
        pass

    def initialize(self) -> None:
        pass

    def contract(self) -> torch.Tensor:
        pass

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass
