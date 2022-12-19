"""
PEPS + UPEPS
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


class PEPS(TensorNetwork):

    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 d_phys: int,
                 d_bond: Sequence[int],
                 boundary: Sequence[Text] = ['obc', 'obc'],
                 param_bond: bool = False,
                 num_batches: int = 1) -> None:  # TODO: poner n_batches
        """
        Create an MPS module.

        Parameters
        ----------
        n_rows: number of rows in the grid
        n_cols: number of columns in the grid
        d_phys: physical dimension
        d_bond: bond dimension, it should only have 2 elements, the d_bond of
            horizontal edges and the d_bond of vertical edges
        boundary: list of strings indicating whether we are using periodic or open
            boundary conditions in the horizontal and vertical boundaries
        param_bond: boolean indicating whether bond edges should be parametric
        num_batches: number of batch edges of input data
        """

        super().__init__(name='mps')

        # boundary
        if not isinstance(boundary, Sequence):
            raise TypeError('`boundary` should be a sequence of two elements')
        elif len(boundary) != 2:
            raise ValueError('`boundary` should be a sequence of two elements')
        
        if boundary[0] == 'obc':
            if n_rows < 2:
                raise ValueError('If `boundary` of rows is "obc", at least '
                                 'there has to be 2 rows')
        elif boundary[0] == 'pbc':
            if n_rows < 1:
                raise ValueError('If `boundary` of rows is "pbc", at least '
                                 'there has to be one site')
        else:
            raise ValueError('`boundary` elements should be one of "obc" or "pbc"')
        
        if boundary[1] == 'obc':
            if n_rows < 2:
                raise ValueError('If `boundary` of columns is "obc", at least '
                                 'there has to be 2 rows')
        elif boundary[1] == 'pbc':
            if n_rows < 1:
                raise ValueError('If `boundary` of columns is "pbc", at least '
                                 'there has to be one site')
        else:
            raise ValueError('`boundary` elements should be one of "obc" or "pbc"')
        
        self._n_rows = n_rows
        self._n_cols = n_cols
        self._boundary = boundary

        # d_phys
        if not isinstance(d_phys, int):
            raise ValueError('`d_phys` should be int type')
        self._d_phys = d_phys
            
        # d_bond
        if isinstance(d_bond, (list, tuple)):
            if len(d_bond) != 2:
                raise ValueError('`d_bond` should be a pair of ints')
            self._d_bond = list(d_bond)
        else:
            raise TypeError('`d_bond` should be a pair of ints')

        # param_bond
        self._param_bond = param_bond

        self._make_nodes()
        self.param_bond(set_param=param_bond)
        self.initialize()
        
        self._num_batches = num_batches
        
    @property
    def n_rows(self) -> int:
        return self._n_rows
    
    @property
    def n_cols(self) -> int:
        return self._n_cols

    @property
    def boundary(self) -> Text:
        return self._boundary

    @property
    def d_phys(self) -> int:
        return self._d_phys

    @property
    def d_bond(self) -> Sequence[int]:
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
                if 'up' in node.axes_names:
                    node['up'].parameterize(set_param=set_param)
                if 'down' in node.axes_names:
                    node['down'].parameterize(set_param=set_param)
            self._param_bond = set_param
            
    def _make_nodes(self) -> None:
        if self.leaf_nodes:
            raise ValueError('Cannot create PEPS nodes if the PEPS already has nodes')

        self.left_border = []
        self.right_border = []
        self.up_border = []
        self.down_border = []
        
        self.left_up_corner = None
        self.left_down_corner = None
        self.right_up_corner = None
        self.right_down_corner = None
        
        self.grid_env = []
        
        d_phys = self._d_phys
        d_bond = self._d_bond
        
        if self._boundary == ['obc', 'obc']:
            # Left up corner
            node = ParamNode(shape=(d_phys, d_bond[0], d_bond[1]),
                             axes_names=('input', 'right', 'down'),
                             name=f'left_up_corner_node',
                             network=self)
            self.left_up_corner = node
            
            # Up border
            for j in range(self._n_cols - 2):
                node = ParamNode(shape=(d_phys, d_bond[0], d_bond[0], d_bond[1]),
                                 axes_names=('input', 'left', 'right', 'down'),
                                 name=f'up_border_node_({j})',
                                 network=self)
                
                if j == 0:
                    self.left_up_corner['right'] ^ node['left']
                else:
                    self.up_border[-1]['right'] ^ node['left']
                self.up_border.append(node)
                    
            # Right up corner
            node = ParamNode(shape=(d_phys, d_bond[0], d_bond[1]),
                             axes_names=('input', 'left', 'down'),
                             name=f'right_up_corner_node',
                             network=self)
            
            if self._n_cols > 2:
                self.up_border[-1]['right'] ^ node['left']
            else:
                self.left_up_corner['right'] ^ node['left']
            self.right_up_corner = node
                
            # Left border
            for i in range(self._n_rows - 2):
                node = ParamNode(shape=(d_phys, d_bond[0], d_bond[1], d_bond[1]),
                                 axes_names=('input', 'right', 'up', 'down'),
                                 name=f'left_border_node_({i})',
                                 network=self)
                
                if i == 0:
                    self.left_up_corner['down'] ^ node['up']
                else:
                    self.left_border[-1]['down'] ^ node['up']
                self.left_border.append(node)
                    
            # Right border
            for i in range(self._n_rows - 2):
                node = ParamNode(shape=(d_phys, d_bond[0], d_bond[1], d_bond[1]),
                                 axes_names=('input', 'left', 'up', 'down'),
                                 name=f'right_border_node_({i})',
                                 network=self)
                
                if i == 0:
                    self.right_up_corner['down'] ^ node['up']
                else:
                    self.right_border[-1]['down'] ^ node['up']
                self.right_border.append(node)
                    
            # Left down corner
            node = ParamNode(shape=(d_phys, d_bond[0], d_bond[1]),
                             axes_names=('input', 'right', 'up'),
                             name=f'left_down_corner_node',
                             network=self)
            
            if self._n_rows > 2:
                self.left_border[-1]['down'] ^ node['up']
            else:
                self.left_up_corner['down'] ^ node['up']
            self.left_down_corner = node
                
            # Down border
            for j in range(self._n_cols - 2):
                node = ParamNode(shape=(d_phys, d_bond[0], d_bond[0], d_bond[1]),
                                 axes_names=('input', 'left', 'right', 'up'),
                                 name=f'down_border_node_({j})',
                                 network=self)
                
                if j == 0:
                    self.left_down_corner['right'] ^ node['left']
                else:
                    self.down_border[-1]['right'] ^ node['left']
                self.down_border.append(node)
                    
            # Right down corner
            node = ParamNode(shape=(d_phys, d_bond[0], d_bond[1]),
                             axes_names=('input', 'left', 'up'),
                             name=f'right_down_corner_node',
                             network=self)
            
            if self._n_rows > 2:
                self.right_border[-1]['down'] ^ node['up']
            else:
                self.right_up_corner['down'] ^ node['up']
                
            if self._n_cols > 2:
                self.down_border[-1]['right'] ^ node['left']
            else:
                self.left_down_corner['right'] ^ node['left']
            self.right_down_corner = node
            
            # Grid env
            for i in range(self._n_rows - 2):
                self.grid_env.append([])
                for j in range(self._n_cols - 2):
                    node = ParamNode(shape=(d_phys, d_bond[0], d_bond[0], d_bond[1], d_bond[1]),
                                     axes_names=('input', 'left', 'right', 'up', 'down'),
                                     name=f'grid_env_node_({i},{j})',
                                     network=self)
                    
                    if i == 0:
                        self.up_border[j]['down'] ^ node['up']
                    else:
                        self.grid_env[i - 1][j]['down'] ^ node['up']
                    if i == self._n_rows - 3:
                        node['down'] ^ self.down_border[j]['up']
                        
                    if j == 0:
                        self.left_border[i]['right'] ^ node['left']
                    else:
                        self.grid_env[i][j - 1]['right'] ^ node['left']
                    if j == self._n_cols - 3:
                        node['right'] ^ self.right_border[i]['left']
                        
                    self.grid_env[-1].append(node)
                    
        elif self._boundary == ['obc', 'pbc']:
            # Up border
            for j in range(self._n_cols):
                node = ParamNode(shape=(d_phys, d_bond[0], d_bond[0], d_bond[1]),
                                 axes_names=('input', 'left', 'right', 'down'),
                                 name=f'up_border_node_({j})',
                                 network=self)
                
                if j > 0:
                    self.up_border[-1]['right'] ^ node['left']
                if j == self._n_cols - 1:
                    node['right'] ^ self.up_border[0]['left']
                self.up_border.append(node)
                
            # Down border
            for j in range(self._n_cols):
                node = ParamNode(shape=(d_phys, d_bond[0], d_bond[0], d_bond[1]),
                                 axes_names=('input', 'left', 'right', 'up'),
                                 name=f'down_border_node_({j})',
                                 network=self)
                
                if j > 0:
                    self.down_border[-1]['right'] ^ node['left']
                if j == self._n_cols - 1:
                    node['right'] ^ self.down_border[0]['left']
                self.down_border.append(node)
            
            # Grid env
            for i in range(self._n_rows - 2):
                self.grid_env.append([])
                for j in range(self._n_cols):
                    node = ParamNode(shape=(d_phys, d_bond[0], d_bond[0], d_bond[1], d_bond[1]),
                                     axes_names=('input', 'left', 'right', 'up', 'down'),
                                     name=f'grid_env_node_({i},{j})',
                                     network=self)
                    
                    if i == 0:
                        self.up_border[j]['down'] ^ node['up']
                    else:
                        self.grid_env[i - 1][j]['down'] ^ node['up']
                    if i == self._n_rows - 3:
                        node['down'] ^ self.down_border[j]['up']
                        
                    if j > 0:
                        self.grid_env[i][j - 1]['right'] ^ node['left']
                    if j == self._n_cols - 1:
                        node['right'] ^ self.grid_env[i][0]['left']
                        
                    self.grid_env[-1].append(node)
                    
        elif self._boundary == ['pbc', 'obc']:
            # Left border
            for i in range(self._n_rows):
                node = ParamNode(shape=(d_phys, d_bond[0], d_bond[1], d_bond[1]),
                                 axes_names=('input', 'right', 'up', 'down'),
                                 name=f'left_border_node_({i})',
                                 network=self)
                
                if i > 0:
                    self.left_border[-1]['down'] ^ node['up']
                if i == self._n_rows - 1:
                    node['down'] ^ self.left_border[0]['up']
                self.left_border.append(node)
                    
            # Right border
            for i in range(self._n_rows):
                node = ParamNode(shape=(d_phys, d_bond[0], d_bond[1], d_bond[1]),
                                 axes_names=('input', 'left', 'up', 'down'),
                                 name=f'right_border_node_({i})',
                                 network=self)
                
                if i > 0:
                    self.right_border[-1]['down'] ^ node['up']
                if i == self._n_rows - 1:
                    node['down'] ^ self.right_border[0]['up']
                self.right_border.append(node)
            
            # Grid env
            for i in range(self._n_rows):
                self.grid_env.append([])
                for j in range(self._n_cols - 2):
                    node = ParamNode(shape=(d_phys, d_bond[0], d_bond[0], d_bond[1], d_bond[1]),
                                     axes_names=('input', 'left', 'right', 'up', 'down'),
                                     name=f'grid_env_node_({i},{j})',
                                     network=self)
                    
                    if i > 0:
                        self.grid_env[i - 1][j]['down'] ^ node['up']
                    if i == self._n_rows - 1:
                        node['down'] ^ self.grid_env[0][j]['up']
                        
                    if j == 0:
                        self.left_border[i]['right'] ^ node['left']
                    else:
                        self.grid_env[i][j - 1]['right'] ^ node['left']
                    if j == self._n_cols - 3:
                        node['right'] ^ self.right_border[i]['left']
                        
                    self.grid_env[-1].append(node)
                    
        else:
            # Grid env
            for i in range(self._n_rows):
                self.grid_env.append([])
                for j in range(self._n_cols):
                    node = ParamNode(shape=(d_phys, d_bond[0], d_bond[0], d_bond[1], d_bond[1]),
                                     axes_names=('input', 'left', 'right', 'up', 'down'),
                                     name=f'grid_env_node_({i},{j})',
                                     network=self)
                    
                    if i > 0:
                        self.grid_env[i - 1][j]['down'] ^ node['up']
                    if i == self._n_rows - 1:
                        node['down'] ^ self.grid_env[0][j]['up']
                        
                    if j > 0:
                        self.grid_env[i][j - 1]['right'] ^ node['left']
                    if j == self._n_cols - 1:
                        node['right'] ^  self.grid_env[i][0]['left']
                        
                    self.grid_env[-1].append(node)
                    
    def initialize(self, std: float = 1e-9) -> None:
        for node in self.leaf_nodes.values():
            node.tensor = torch.randn(node.shape) * std
    
    def set_data_nodes(self) -> None:
        input_edges = []
        
        for i in range(self._n_rows):
            for j in range(self._n_cols):
                if self._boundary == ['obc', 'obc']:
                    if i == 0:
                        if j == 0:
                            node = self.left_up_corner
                        elif j < self._n_cols - 1:
                            node = self.up_border[j - 1]
                        else:
                            node = self.right_up_corner
                    elif i < self._n_rows - 1:
                        if j == 0:
                            node = self.left_border[i - 1]
                        elif j < self._n_cols - 1:
                            node = self.grid_env[i - 1][j - 1]
                        else:
                            node = self.right_border[i - 1]
                    else:
                        if j == 0:
                            node = self.left_down_corner
                        elif j < self._n_cols - 1:
                            node = self.down_border[j - 1]
                        else:
                            node = self.right_down_corner
                elif self._boundary == ['obc', 'pbc']:
                    if i == 0:
                        node = self.up_border[j]
                    elif i < self._n_rows - 1:
                        node = self.grid_env[i - 1][j]
                    else:
                        node = self.down_border[j]
                elif self._boundary == ['pbc', 'obc']:
                    if j == 0:
                        node = self.left_border[i]
                    elif j < self._n_cols - 1:
                        node = self.grid_env[i][j - 1]
                    else:
                        node = self.right_border[i]
                else:
                    node = self.grid_env[i][j]
                    
                input_edges.append(node['input'])
            
        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self._num_batches)
        
    def _input_contraction(self):
        full_grid = []
        for _ in range(self._n_rows):
            full_grid.append([])
            for _ in range(self._n_cols):
                full_grid[-1].append(None)
        
        # Contract with data
        if self._boundary == ['obc', 'obc']:
            # Corners
            full_grid[0][0] = self.left_up_corner.neighbours('input') @ \
                self.left_up_corner
            full_grid[0][-1] = self.right_up_corner.neighbours('input') @ \
                self.right_up_corner
            full_grid[-1][0] = self.left_down_corner.neighbours('input') @ \
                self.left_down_corner
            full_grid[-1][-1] = self.right_down_corner.neighbours('input') @ \
                self.right_down_corner
            
        if self._boundary[0] == 'obc':
            # Up border
            stack_up_border = tk.stack(self.up_border)
            stack_up_border_data = tk.stack(list(map(lambda x: x.neighbours('input'),
                                                     self.up_border)))
            stack_up_border_data['feature'] ^ stack_up_border['input']
            result_up_border = stack_up_border_data @ stack_up_border
            result_up_border = tk.unbind(result_up_border)
            
            # Down border
            stack_down_border = tk.stack(self.down_border)
            stack_down_border_data = tk.stack(list(map(lambda x: x.neighbours('input'),
                                                       self.down_border)))
            stack_down_border_data['feature'] ^ stack_down_border['input']
            result_down_border = stack_down_border_data @ stack_down_border
            result_down_border = tk.unbind(result_down_border)
            
            if self._boundary[1] == 'obc':
                full_grid[0][1:-1] = result_up_border
                full_grid[-1][1:-1] = result_down_border
            else:
                full_grid[0] = result_up_border
                full_grid[-1] = result_down_border
            
        if self._boundary[1] == 'obc':
            # Left border
            stack_left_border = tk.stack(self.left_border)
            stack_left_border_data = tk.stack(list(map(lambda x: x.neighbours('input'),
                                                       self.left_border)))
            stack_left_border_data['feature'] ^ stack_left_border['input']
            result_left_border = stack_left_border_data @ stack_left_border
            result_left_border = tk.unbind(result_left_border)
            
            # Right border
            stack_right_border = tk.stack(self.right_border)
            stack_right_border_data = tk.stack(list(map(lambda x: x.neighbours('input'),
                                                        self.right_border)))
            stack_right_border_data['feature'] ^ stack_right_border['input']
            result_right_border = stack_right_border_data @ stack_right_border
            result_right_border = tk.unbind(result_right_border)
            
            if self._boundary[0] == 'obc':
                for i in range(self._n_rows - 2):
                    full_grid[i + 1][0] = result_left_border[i]
                    full_grid[i + 1][-1] = result_right_border[i]
            else:
                for i in range(self._n_rows):
                    full_grid[i][0] = result_left_border[i]
                    full_grid[i][-1] = result_right_border[i]
            
        # Grid env
        list_grid_env = []
        for lst in self.grid_env:
            list_grid_env += lst
        stack_grid_env = tk.stack(list_grid_env)
        stack_grid_env_data = tk.stack(list(map(lambda x: x.neighbours('input'),
                                                list_grid_env)))
        stack_grid_env_data['feature'] ^ stack_grid_env['input']
        result_grid_env = stack_grid_env_data @ stack_grid_env
        result_grid_env = tk.unbind(result_grid_env)
        
        if self._boundary == ['obc', 'obc']:
            for i in range(self._n_rows - 2):
                for j in range(self._n_cols - 2):
                    full_grid[i + 1][j + 1] = result_grid_env[i * (self._n_cols - 2) + j]
        elif self._boundary == ['obc', 'pbc']:
            for i in range(self._n_rows - 2):
                for j in range(self._n_cols):
                    full_grid[i + 1][j] = result_grid_env[i * self._n_cols + j]
        elif self._boundary == ['pbc', 'obc']:
            for i in range(self._n_rows):
                for j in range(self._n_cols - 2):
                    full_grid[i][j + 1] = result_grid_env[i * (self._n_cols - 2) + j]
        else:
            for i in range(self._n_rows):
                for j in range(self._n_cols):
                    full_grid[i][j] = result_grid_env[i * self._n_cols + j]
                    
        return full_grid
    
    def _contract_2_lines(self,
                          line1: List[Node],
                          line2: List[Node],
                          from_side: Text = 'up',
                          inline: bool = True) -> List[Node]:
        result_list = []
        
        if inline:
            for (node1, node2) in zip(line1, line2):
                result_list.append(node1 @ node2)
                
        else:
            if self._boundary == ['obc', 'obc']:
                result_list.append(line1[0] @ line2[0])
                
                stack1 = tk.stack(line1[1:-1])
                stack2 = tk.stack(line2[1:-1])
                
                node1 = line1[1]
                node2 = line2[1]
                axis1, axis2 = None, None
                for i1, edge1 in enumerate(node1._edges):
                    for edge2 in node2._edges:
                        if edge1 == edge2:
                            axis1 = edge1._axes[1 - node1.is_node1(i1)]
                            axis2 = edge1._axes[node1.is_node1(i1)]
                            break
                    if axis1 != None:
                        break
                
                stack1[axis1.name] ^ stack2[axis2.name]
                
                # if from_side == 'left':
                #     stack1['right'] ^ stack2['left']
                # elif from_side == 'right':
                #     stack1['left'] ^ stack2['right']
                # elif from_side == 'up':
                #     stack1['down'] ^ stack2['up']
                # else:
                #     stack1['up'] ^ stack2['down']
                
                stack_result = stack1 @ stack2
                result_list += tk.unbind(stack_result)
                
                result_list.append(line1[-1] @ line2[-1])
            
            else: # TODO: aqui hay que diferenciar pbc y obc en cada orientacion
                stack1 = tk.stack(line1)
                stack2 = tk.stack(line2)
                
                node1 = line1[0]
                node2 = line2[0]
                axis1, axis2 = None, None
                for i1, edge1 in enumerate(node1._edges):
                    for edge2 in node2._edges:
                        if edge1 == edge2:
                            axis1 = edge1._axes[1 - node1.is_node1(i1)]
                            axis2 = edge1._axes[node1.is_node1(i1)]
                            break
                    if axis1 != None:
                        break
                
                stack1[axis1.name] ^ stack2[axis2.name]
                
                # if from_side == 'left':
                #     stack1['right'] ^ stack2['left']
                # elif from_side == 'right':
                #     stack1['left'] ^ stack2['right']
                # elif from_side == 'up':
                #     stack1['down'] ^ stack2['up']
                # else:
                #     stack1['up'] ^ stack2['down']
                
                stack_result = stack1 @ stack2
                result_list = tk.unbind(stack_result)
                
        return result_list
    
    def _split_line(self,
                    line: List[Node],
                    pbc: bool = False,
                    from_side: Text = 'up',
                    max_bond: int = 32) -> List[Node]:
        nb = self._num_batches
        for i in range(len(line) - 1):
            contracted = line[i] @ line[i + 1]
            splitted1, splitted2 = contracted.split(
                node1_axes=contracted._axes[nb:(line[i].rank - 2)],
                node2_axes=contracted._axes[(line[i].rank - 2):],
                rank=max_bond)
            if (from_side == 'up') or (from_side == 'down'):
                splitted1.get_axis('splitted').name = 'right'
                splitted2.get_axis('splitted').name = 'left'
            else:
                splitted1.get_axis('splitted').name = 'down'
                splitted2.get_axis('splitted').name = 'up'
            line[i] = splitted1
            line[i + 1] = splitted2
            
        if pbc:
            contracted = line[-1] @ line[0]
            splitted1, splitted2 = contracted.split(
                node1_axes=contracted._axes[nb:(line[-1].rank - 2)],
                node2_axes=contracted._axes[(line[-1].rank - 2):],
                rank=max_bond)
            if (from_side == 'up') or (from_side == 'down'):
                splitted1.get_axis('splitted').name = 'right'
                splitted2.get_axis('splitted').name = 'left'
            else:
                splitted1.get_axis('splitted').name = 'down'
                splitted2.get_axis('splitted').name = 'up'
            line[-1] = splitted1
            line[0] = splitted2
            
        return line
                    
        # if self._boundary == ['obc', 'obc']:
        #     dict_sides = {'up': ['left', 'right', 'up', 'down'],    # contraemos hacia derecha
        #                   'down': ['left', 'right', 'down', 'up'],  # contraemos hacia derecha
        #                   'left': ['up', 'down', 'left', 'right'],
        #                   'right': [None, None, None, None]}
        #     if from_side == 'up':
        #         for i in range(len(line) - 1):
        #             # if i == 0:
        #             contracted = line[i] @ line[i + 1]
        #             splitted1, splitted2 = contracted.split(
        #                 node1_axes=contracted._axes[self._num_batches:
        #                     (line[i].rank - 1 + self._num_batches)],
        #                 node2_axes=contracted._axes[
        #                     (line[i].rank - 1 + self._num_batches):],
        #                 rank=max_bond)
        #             splitted1.get_axis('splitted').name = 'right'
        #             splitted2.get_axis('splitted').name = 'left'
        #             line[i] = splitted1
        #             line[i + 1] = splitted2
        #                 
                    # elif i < len(line) - 2:
                    #     contracted = line[i] @ line[i + 1]
                    #     splitted1, splitted2 = contracted.split(node1_axes=['left', 'down_0'],
                    #                                             node2_axes=['right_0', 'right_1', 'down_1'],
                    #                                             rank=max_bond)
                    #     splitted1.get_axis('splitted').name = 'right'
                    #     splitted2.get_axis('splitted').name = 'left'
                    #     line[i] = splitted1
                    #     line[i + 1] = splitted2
                    #     
                    # else:
                    #     contracted = line[i] @ line[i + 1]
                    #     splitted1, splitted2 = contracted.split(node1_axes=['left', 'down_0'],
                    #                                             node2_axes=['down_1'],
                    #                                             rank=max_bond)
                    #     splitted1.get_axis('splitted').name = 'right'
                    #     splitted2.get_axis('splitted').name = 'left'
                    #     line[i] = splitted1
                    #     line[i + 1] = splitted2
                                    
        
    def contract(self, from_side='up', max_bond=32, inline=True):
        full_grid = self._input_contraction()
        
        pbc = False
        if from_side == 'up':
            if self._boundary[1] == 'pbc':
                pbc = True
                
        elif from_side == 'down':
            full_grid.reverse()
            
            if self._boundary[1] == 'pbc':
                pbc = True
                
        elif from_side == 'left':
            new_grid = []
            for j in range(len(full_grid[0])):
                row = []
                for i in range(len(full_grid)):
                    row.append(full_grid[i][j])
                new_grid.append(row)
            full_grid = new_grid
            
            if self._boundary[0] == 'pbc':
                pbc = True
                
        elif from_side == 'right':
            new_grid = []
            for j in range(len(full_grid[0])):
                row = []
                for i in range(len(full_grid)):
                    row.append(full_grid[i][j])
                new_grid.append(row)
            full_grid = new_grid
            full_grid.reverse()
            
            if self._boundary[0] == 'pbc':
                pbc = True
            
        for i in range(len(full_grid) - 1):
            line1 = full_grid[i]
            line2 = full_grid[i + 1]
            contracted_line = self._contract_2_lines(line1, line2,
                                                     from_side=from_side,
                                                     inline=inline)
            splitted_line = self._split_line(contracted_line,
                                             pbc=pbc,
                                             from_side=from_side,
                                             max_bond=max_bond)
            
            full_grid[i + 1] = splitted_line
            
        result = full_grid[-1][0]
        for node in full_grid[-1][1:]:
            result @= node
        
        return result
    