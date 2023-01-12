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


class Tree(TensorNetwork):
    """
    Create an Tree module.

    Parameters
    ----------
    sites_per_layer: number of sites in each layer of the tree
    d_bond: bond dimensions of nodes in each layer. In each layer, all
        nodes have the same shape, formed by various input edges and a
        single output edge. d_bond should be a sequence of sequences,
        one for each layer. Each sequence is formed by the bond dimensions
        of the input edges and the bond dimension of the output (last
        element in the sequence)
    param_bond: boolean indicating whether bond edges should be parametric
    num_batches: number of batch edges of input data
    """

    def __init__(self,
                 sites_per_layer: Sequence[int],
                 d_bond: Sequence[Sequence[int]],
                 param_bond: bool = False,
                 num_batches: int = 1) -> None:

        super().__init__(name='tree')

        # sites_per_layer
        if isinstance(sites_per_layer, (list, tuple)):
            if not sites_per_layer:
                raise ValueError('`sites_per_layer` cannot be empty, at least '
                                 'one node is required')
            for el in sites_per_layer:
                if not isinstance(el, int):
                    raise TypeError('`sites_per_layer` should be a sequence of ints')
                if el < 1:
                    raise ValueError('Elements of `sites_per_layer` should be '
                                     'ints greater than 0')
            if sites_per_layer[-1] > 1:
                raise ValueError('The last element of `sites_per_layer` should '
                                 'be always 1')
        else:
            raise TypeError('`sites_per_layer` should be a sequence '
                            '(list or tuple type)')
        self._sites_per_layer = sites_per_layer

        # d_bond
        if isinstance(d_bond, (list, tuple)):
            aux_d_bond = []
            for lst in d_bond:
                if isinstance(lst, (list, tuple)):
                    if len(lst) < 2:
                        raise ValueError('`d_bond` sequences should have at '
                                         'least two elements, one for input '
                                         'and one for output')
                    for el in lst:
                        if not isinstance(el, int):
                            raise TypeError('`d_bond` should be a sequence of '
                                            'sequences of ints')
                else:
                    raise TypeError('`d_bond` should be a sequence of '
                                    'sequences of ints')
                aux_d_bond.append(list(lst))
        else:
            raise TypeError('`d_bond` should be a sequence of sequences of ints')
        self._d_bond = aux_d_bond
        
        if len(sites_per_layer) != len(d_bond):
            raise ValueError('`sites_per_layer` and `d_bond` should have the '
                             'same number of elements')

        # param_bond
        self._param_bond = param_bond

        self._make_nodes()
        self.param_bond(set_param=param_bond)
        self.initialize()
        
        self._num_batches = num_batches

    @property
    def sites_per_layer(self) -> int:
        return self._sites_per_layer

    @property
    def d_bond(self) -> Sequence[Sequence[int]]:
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
            for layer in self.layers[1:]:
                for node in layer:
                    for edge in node._edges[:-1]:
                        edge.parameterize(set_param=set_param)
            self._param_bond = set_param

    def _make_nodes(self) -> None:
        if self.leaf_nodes:
            raise ValueError('Cannot create Tree nodes if the Tree already has nodes')

        self.layers = []
        
        for i, n_sites in enumerate(self._sites_per_layer):
            layer_lst = []
            for j in range(n_sites):
                node = ParamNode(shape=(*self._d_bond[i],),
                                 axes_names=(*(['input'] * (len(self._d_bond[i]) - 1)),
                                             'output'),
                                 name=f'tree_node_({i},{j})',
                                 network=self)
                layer_lst.append(node)
                
            if i > 0:
                idx_last_layer = 0
                for node in layer_lst:
                    for edge in node._edges[:-1]:
                        if idx_last_layer == len(self.layers[-1]):
                            raise ValueError(f'There are more input edges in '
                                             f'layer {i} than output edges in '
                                             f'layer {i - 1}')
                            
                        self.layers[-1][idx_last_layer]['output'] ^ edge
                        idx_last_layer += 1
                    
                if idx_last_layer < len(self.layers[-1]):
                    raise ValueError(f'There are more output edges in '
                                     f'layer {i - 1} than input edges in '
                                     f'layer {i}')
            
            self.layers.append(layer_lst)

    def initialize(self, std: float = 1e-9) -> None:
        for layer in self.layers:
            for node in layer:
                tensor = torch.randn(node.shape) * std
                tensor[(0,) * node.rank] = 1.
                node.tensor = tensor

    def set_data_nodes(self) -> None:
        input_edges = []
        for node in self.layers[0]:
            input_edges += node._edges[:-1]
            
        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self._num_batches)

    def _input_contraction(self,
                           layer1: List[Node],
                           layer2: List[Node],
                           inline: bool) -> Tuple[Optional[List[Node]],
                                                  Optional[List[Node]]]:
        if inline:
            result_lst = []
            i = 0
            for node in layer2:
                for _ in range(node.rank - 1):
                    node = layer1[i] @ node
                    i += 1
                result_lst.append(node)
                
            return result_lst

        else:
            n_input = layer2[0].rank - 1
            stack2 = tk.stack(layer2)
            
            layer1_stacks = []
            for i in range(n_input):
                stack_lst = []
                for j in range(i, len(layer1), n_input):
                    stack_lst.append(layer1[j])
                layer1_stacks.append(tk.stack(stack_lst))
                
            for i in range(n_input):
                stack2[i + 1] ^ layer1_stacks[i][-1]
                
            result = stack2
            for i in range(n_input):
                result = layer1_stacks[i] @ result
                
            result_lst = tk.unbind(result)
            return result_lst

    def contract(self, inline=True) -> Node:
        layers = [list(self.data_nodes.values())] + self.layers
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i + 1]
            result_lst = self._input_contraction(layer1, layer2, inline=inline)
            layers[i + 1] = result_lst

        return result_lst[0]
    
    def _canonicalize_layer(self,
                            layer1: List[Node],
                            layer2: List[Node],
                            mode: Text = 'svd',
                            rank: Optional[int] = None,
                            cum_percentage: Optional[float] = None,
                            cutoff: Optional[float] = None) -> None:
        new_layer1 = []
        new_layer2 = []
        i = 0
        for node in layer2:
            for _ in range(node.rank - 1):
                if mode == 'svd':
                    result1, node = layer1[i]['output'].svd_(side='right',
                                                             rank=rank,
                                                             cum_percentage=cum_percentage,
                                                             cutoff=cutoff)
                elif mode == 'svdr':
                    result1, node = layer1[i]['output'].svdr_(side='right',
                                                              rank=rank,
                                                              cum_percentage=cum_percentage,
                                                              cutoff=cutoff)
                elif mode == 'qr':
                    result1, node = layer1[i]['output'].qr_()
                else:
                    raise ValueError('`mode` can only be \'svd\', \'svdr\' or \'qr\'')
                
                new_layer1.append(result1.parameterize())
                i += 1
                
            new_layer2.append(node.parameterize())
            
        return new_layer1, new_layer2
    
    def canonicalize(self,
                     mode: Text = 'svd',
                     rank: Optional[int] = None,
                     cum_percentage: Optional[float] = None,
                     cutoff: Optional[float] = None) -> None:
        """
        Turns the MPS into canonical form
        
        Parameters
        ----------
        mode: can be either 'svd', 'svdr' or 'qr'
        """
        if len(self.layers) > 1:
            for i in range(len(self.layers) - 1):
                layer1 = self.layers[i]
                layer2 = self.layers[i + 1]
                layer1, layer2 = self._canonicalize_layer(layer1, layer2,
                                                          mode=mode,
                                                          rank=rank,
                                                          cum_percentage=cum_percentage,
                                                          cutoff=cutoff)
                self.layers[i] = layer1
                self.layers[i + 1] = layer2
            
        self.param_bond(set_param=self._param_bond)


class UTree(TensorNetwork):
    """
    Create an UTree module.

    Parameters
    ----------
    sites_per_layer: number of sites in each layer of the tree
    d_bond: bond dimensions of nodes in each layer. In each layer, all
        nodes have the same shape, formed by various input edges and a
        single output edge. d_bond should be a sequence of sequences,
        one for each layer. Each sequence is formed by the bond dimensions
        of the input edges and the bond dimension of the output (last
        element in the sequence)
    param_bond: boolean indicating whether bond edges should be parametric
    num_batches: number of batch edges of input data
    """

    def __init__(self,
                 sites_per_layer: Sequence[int],
                 d_bond: Sequence[int],
                 param_bond: bool = False,
                 num_batches: int = 1) -> None:

        super().__init__(name='tree')

        # sites_per_layer
        if isinstance(sites_per_layer, (list, tuple)):
            if not sites_per_layer:
                raise ValueError('`sites_per_layer` cannot be empty, at least '
                                 'one node is required')
            for el in sites_per_layer:
                if not isinstance(el, int):
                    raise TypeError('`sites_per_layer` should be a sequence of ints')
                if el < 1:
                    raise ValueError('Elements of `sites_per_layer` should be '
                                     'ints greater than 0')
            if sites_per_layer[-1] > 1:
                raise ValueError('The last element of `sites_per_layer` should '
                                 'be always 1')
        else:
            raise TypeError('`sites_per_layer` should be a sequence '
                            '(list or tuple type)')
        self._sites_per_layer = sites_per_layer

        # d_bond
        if isinstance(d_bond, (list, tuple)):
            if len(d_bond) < 2:
                raise ValueError('`d_bond` should have at least two elements, '
                                 'one for input and one for output')
            for el in d_bond:
                if not isinstance(el, int):
                    raise TypeError('`d_bond` should be a sequence of ints')
        else:
            raise TypeError('`d_bond` should be a sequence of ints')
        self._d_bond = list(d_bond)

        # param_bond
        self._param_bond = param_bond

        self._make_nodes()
        self.param_bond(set_param=param_bond)
        self.initialize()
        
        self._num_batches = num_batches

    @property
    def sites_per_layer(self) -> int:
        return self._sites_per_layer

    @property
    def d_bond(self) -> Sequence[Sequence[int]]:
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
            for layer in self.layers[1:]:
                for node in layer:
                    for edge in node._edges[:-1]:
                        edge.parameterize(set_param=set_param)
            self._param_bond = set_param

    def _make_nodes(self) -> None:
        if self.leaf_nodes:
            raise ValueError('Cannot create Tree nodes if the Tree already has nodes')

        self.layers = []
        
        for i, n_sites in enumerate(self._sites_per_layer):
            layer_lst = []
            for j in range(n_sites):
                node = ParamNode(shape=(*self._d_bond,),
                                 axes_names=(*(['input'] * (len(self._d_bond) - 1)),
                                             'output'),
                                 name=f'tree_node_({i},{j})',
                                 network=self)
                layer_lst.append(node)
                
            if i > 0:
                idx_last_layer = 0
                for node in layer_lst:
                    for edge in node._edges[:-1]:
                        if idx_last_layer == len(self.layers[-1]):
                            raise ValueError(f'There are more input edges in '
                                             f'layer {i} than output edges in '
                                             f'layer {i - 1}')
                            
                        edge ^ self.layers[-1][idx_last_layer]['output']
                        idx_last_layer += 1
                    
                if idx_last_layer < len(self.layers[-1]):
                    raise ValueError(f'There are more output edges in '
                                     f'layer {i - 1} than input edges in '
                                     f'layer {i}')
            
            self.layers.append(layer_lst)
            
        # Virtual node
        uniform_memory = node = ParamNode(shape=(*self._d_bond,),
                                          axes_names=(*(['input'] * (len(self._d_bond) - 1)),
                                                      'output'),
                                          name='virtual_uniform',
                                          network=self,
                                          virtual=True)
        self.uniform_memory = uniform_memory
        
        for edge in uniform_memory._edges:
            self._remove_edge(edge)

    def initialize(self, std: float = 1e-9) -> None:
        # Virtual node
        tensor = torch.randn(self.uniform_memory.shape) * std
        tensor[(0,) * len(tensor.shape)] = 1.
        self.uniform_memory._unrestricted_set_tensor(tensor)
        
        for layer in self.layers:
            for node in layer:
                del self._memory_nodes[node._tensor_info['address']]
                node._tensor_info['address'] = None
                node._tensor_info['node_ref'] = self.uniform_memory
                node._tensor_info['full'] = True
                node._tensor_info['stack_idx'] = None
                node._tensor_info['index'] = None

    def set_data_nodes(self) -> None:
        input_edges = []
        for node in self.layers[0]:
            input_edges += node._edges[:-1]
            
        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self._num_batches)

    def _input_contraction(self,
                           layer1: List[Node],
                           layer2: List[Node],
                           inline: bool) -> Tuple[Optional[List[Node]],
                                                  Optional[List[Node]]]:
        if inline:
            result_lst = []
            i = 0
            for node in layer2:
                for _ in range(node.rank - 1):
                    node = layer1[i] @ node
                    i += 1
                result_lst.append(node)
                
            return result_lst

        else:
            n_input = layer2[0].rank - 1
            stack2 = tk.stack(layer2)
            
            layer1_stacks = []
            for i in range(n_input):
                stack_lst = []
                for j in range(i, len(layer1), n_input):
                    stack_lst.append(layer1[j])
                layer1_stacks.append(tk.stack(stack_lst))
                
            for i in range(n_input):
                stack2[i + 1] ^ layer1_stacks[i][-1]
                
            result = stack2
            for i in range(n_input):
                result = layer1_stacks[i] @ result
                
            result_lst = tk.unbind(result)
            return result_lst

    def contract(self, inline=True) -> Node:
        layers = [list(self.data_nodes.values())] + self.layers
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i + 1]
            result_lst = self._input_contraction(layer1, layer2, inline=inline)
            layers[i + 1] = result_lst

        return result_lst[0]


class ConvTree(Tree):
    
    def __init__(self,
                 sites_per_layer: Sequence[int],
                 d_bond: Sequence[Sequence[int]],
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 param_bond: bool = False,):
        
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
        
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        
        super().__init__(sites_per_layer=sites_per_layer,
                         d_bond=d_bond,
                         param_bond=param_bond,
                         num_batches=2)
        self._in_channels = d_bond[0][0]
        
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
    
    def forward(self, image, *args, **kwargs):
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
        
        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows x out_channels
        
        result = result.transpose(1, 2)
        # batch_size x out_channels x nb_windows
        
        h_in = image.shape[2]
        w_in = image.shape[3]
        
        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] * \
            (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] * \
                    (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        
        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x out_channels x height_out x width_out
        
        return result
    

class ConvUTree(UTree):
    
    def __init__(self,
                 sites_per_layer: Sequence[int],
                 d_bond: Sequence[int],
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 param_bond: bool = False,):
        
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
        
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        
        super().__init__(sites_per_layer=sites_per_layer,
                         d_bond=d_bond,
                         param_bond=param_bond,
                         num_batches=2)
        self._in_channels = d_bond[0]
        
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
    
    def forward(self, image, *args, **kwargs):
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
        
        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows x out_channels
        
        result = result.transpose(1, 2)
        # batch_size x out_channels x nb_windows
        
        h_in = image.shape[2]
        w_in = image.shape[3]
        
        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] * \
            (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] * \
                    (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        
        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x out_channels x height_out x width_out
        
        return result
