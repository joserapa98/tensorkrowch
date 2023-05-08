"""
This script contains:
    *Tree
    *UTree
    *ConvTree
    *ConvUTree
"""

from typing import (List, Optional, Sequence,
                    Text, Tuple, Union)

import torch
import torch.nn as nn

from tensorkrowch.components import Node, ParamNode
from tensorkrowch.components import TensorNetwork
import tensorkrowch.operations as op


class Tree(TensorNetwork):
    """
    Class for Tree States. These states form a tree structure where the ``data``
    nodes are in the base. All nodes have a sequence of input edges and an
    output edge. Thus the contraction of the Tree returns a vector.

    Parameters
    ----------
    sites_per_layer : list[int] or tuple[int]
        Number of sites in each layer of the tree. All nodes in the same layer
        have the same shape. Number of nodes in each layer times the number of
        input edges these have should match the number ot output edges in the
        previous layer.
    d_bond : list[list[int]] or tuple[tuple[int]]
        Bond dimensions of nodes in each layer. Each sequence corresponds to the
        shape of the nodes in each layer (some input edges and an output edge in
        the last position).
    num_batches : int
        Number of batch edges of input data nodes. Usually ``num_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``num_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    
    Examples
    --------
    >>> tree = tk.Tree(sites_per_layer=[4, 2, 1],
    ...                d_bond=[[3, 3, 4], [4, 4, 2], [2, 2, 2]])
    >>> data = torch.ones(8, 20, 3) # n_features x batch_size x feature_size
    >>> result = tree(data)
    >>> print(result.shape)
    torch.Size([20, 2])
    """

    def __init__(self,
                 sites_per_layer: Sequence[int],
                 d_bond: Sequence[Sequence[int]],
                 num_batches: int = 1) -> None:

        super().__init__(name='tree')

        # sites_per_layer
        if isinstance(sites_per_layer, (list, tuple)):
            if not sites_per_layer:
                raise ValueError('`sites_per_layer` cannot be empty, at least '
                                 'one node is required')
            for el in sites_per_layer:
                if not isinstance(el, int):
                    raise TypeError('`sites_per_layer` should be a sequence of '
                                    'ints')
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
            raise TypeError('`d_bond` should be a sequence of sequences of '
                            'ints')
        self._d_bond = aux_d_bond
        
        if len(sites_per_layer) != len(d_bond):
            raise ValueError('`sites_per_layer` and `d_bond` should have the '
                             'same number of elements')

        self._make_nodes()
        self.initialize()
        
        self._num_batches = num_batches

    @property
    def sites_per_layer(self) -> Sequence[int]:
        """Returns number of sites in each layer of the tree."""
        return self._sites_per_layer

    @property
    def d_bond(self) -> Sequence[Sequence[int]]:
        """
        Returns bond dimensions of nodes in each layer. Each sequence
        corresponds to the shape of the nodes in each layer (some input edges
        and an output edge in the last position).
        """
        return self._d_bond

    def _make_nodes(self) -> None:
        """Creates all the nodes of the Tree."""
        if self._leaf_nodes:
            raise ValueError('Cannot create Tree nodes if the Tree already has'
                             ' nodes')

        self.layers = []
        
        for i, n_sites in enumerate(self._sites_per_layer):
            layer_lst = []
            for j in range(n_sites):
                node = ParamNode(shape=(*self._d_bond[i],),
                                 axes_names=(*(['input'] * (
                                     len(self._d_bond[i]) - 1)),
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
        """Initializes all the nodes."""
        for layer in self.layers:
            for node in layer:
                tensor = torch.randn(node._shape) * std
                tensor[(0,) * node.rank] = 1.
                node.tensor = tensor

    def set_data_nodes(self) -> None:
        """
        Creates data nodes and connects each of them to the physical edge of
        an input node.
        """
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
        """Contracts two consecutive layers of the tree."""
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
            stack2 = op.stack(layer2)
            
            layer1_stacks = []
            for i in range(n_input):
                stack_lst = []
                for j in range(i, len(layer1), n_input):
                    stack_lst.append(layer1[j])
                layer1_stacks.append(op.stack(stack_lst))
                
            for i in range(n_input):
                stack2[i + 1] ^ layer1_stacks[i][-1]
                
            result = stack2
            for i in range(n_input):
                result = layer1_stacks[i] @ result
                
            result_lst = op.unbind(result)
            return result_lst

    def contract(self, inline: bool = False) -> Node:
        """
        Contracts the whole Tree Tensor Network.

        Parameters
        ----------
        inline : bool
            Boolean indicating whether consecutive layers should be contracted
            inline or in parallel (using a single stacked contraction).

        Returns
        -------
        Node
        """
        layers = [list(self.data_nodes.values())] + self.layers
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i + 1]
            result_lst = self._input_contraction(layer1,
                                                 layer2,
                                                 inline=inline)
            layers[i + 1] = result_lst

        return result_lst[0]
    
    def _canonicalize_layer(self,
                            layer1: List[Node],
                            layer2: List[Node],
                            mode: Text = 'svd',
                            rank: Optional[int] = None,
                            cum_percentage: Optional[float] = None,
                            cutoff: Optional[float] = None) -> None:
        """
        Turns each layer into canonical form, moving singular values matrices
        or non-isometries to the upper layer.
        """
        new_layer1 = []
        new_layer2 = []
        i = 0
        for node in layer2:
            for _ in range(node.rank - 1):
                if mode == 'svd':
                    result1, node = layer1[i]['output'].svd_(
                        side='right',
                        rank=rank,
                        cum_percentage=cum_percentage,
                        cutoff=cutoff)
                elif mode == 'svdr':
                    result1, node = layer1[i]['output'].svdr_(
                        side='right',
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
        r"""
        Turns Tree into canonical form via local SVD/QR decompositions, moving
        singular values matrices or non-isometries to the upper layers.
        
        Parameters
        ----------
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
        if len(self.layers) > 1:
            
            prev_automemory = self._automemory
            self.automemory = False
            
            for i in range(len(self.layers) - 1):
                layer1 = self.layers[i]
                layer2 = self.layers[i + 1]
                layer1, layer2 = self._canonicalize_layer(
                    layer1, layer2,
                    mode=mode,
                    rank=rank,
                    cum_percentage=cum_percentage,
                    cutoff=cutoff)
                self.layers[i] = layer1
                self.layers[i + 1] = layer2
                
            self.automemory = prev_automemory


class UTree(TensorNetwork):
    """
    Class for Uniform Tree States where all nodes have the same shape. It is
    the uniform version of :class:`Tree`, that is, all nodes share the same
    tensor.

    Parameters
    ----------
    sites_per_layer : list[int] or tuple[int]
        Number of sites in each layer of the tree. All nodes have the same
        shape. Number of nodes in each layer times the number of input edges
        these have should match the number ot output edges in the previous
        layer.
    d_bond : list[int] or tuple[int]
        Bond dimensions of nodes in each layer. Since all nodes have the same
        shape, it is enough to pass a single sequence of dimensions (some input
        edges and an output edge in the last position).
    num_batches : int
        Number of batch edges of input data nodes. Usually ``num_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``num_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    """

    def __init__(self,
                 sites_per_layer: Sequence[int],
                 d_bond: Sequence[int],
                 num_batches: int = 1) -> None:

        super().__init__(name='tree')

        # sites_per_layer
        if isinstance(sites_per_layer, (list, tuple)):
            if not sites_per_layer:
                raise ValueError('`sites_per_layer` cannot be empty, at least '
                                 'one node is required')
            for el in sites_per_layer:
                if not isinstance(el, int):
                    raise TypeError('`sites_per_layer` should be a sequence of'
                                    ' ints')
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

        self._make_nodes()
        self.initialize()
        
        self._num_batches = num_batches

    @property
    def sites_per_layer(self) -> Sequence[int]:
        """Returns number of sites in each layer of the tree."""
        return self._sites_per_layer

    @property
    def d_bond(self) -> Sequence[int]:
        """Returns bond dimensions of nodes in each layer. Since all nodes have
        the same shape, it is a single sequence of dimensions (some input edges
        and an output edge in the last position).
        """
        return self._d_bond

    def _make_nodes(self) -> None:
        """Creates all the nodes of the Tree."""
        if self._leaf_nodes:
            raise ValueError('Cannot create Tree nodes if the Tree already has'
                             ' nodes')

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
                                          axes_names=(*(['input'] * (
                                              len(self._d_bond) - 1)),
                                                      'output'),
                                          name='virtual_uniform',
                                          network=self,
                                          virtual=True)
        self.uniform_memory = uniform_memory

    def initialize(self, std: float = 1e-9) -> None:
        """Initializes all the nodes."""
        # Virtual node
        tensor = torch.randn(self.uniform_memory._shape) * std
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
        """
        Creates data nodes and connects each of them to the physical edge of
        an input node.
        """
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
        """Contracts two consecutive layers of the tree."""
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
            stack2 = op.stack(layer2)
            
            layer1_stacks = []
            for i in range(n_input):
                stack_lst = []
                for j in range(i, len(layer1), n_input):
                    stack_lst.append(layer1[j])
                layer1_stacks.append(op.stack(stack_lst))
                
            for i in range(n_input):
                stack2[i + 1] ^ layer1_stacks[i][-1]
                
            result = stack2
            for i in range(n_input):
                result = layer1_stacks[i] @ result
                
            result_lst = op.unbind(result)
            return result_lst

    def contract(self, inline: bool = False) -> Node:
        """
        Contracts the whole Tree Tensor Network.

        Parameters
        ----------
        inline : bool
            Boolean indicating whether consecutive layers should be contracted
            inline or in parallel (using a single stacked contraction).

        Returns
        -------
        Node
        """
        layers = [list(self.data_nodes.values())] + self.layers
        for i in range(len(layers) - 1):
            layer1 = layers[i]
            layer2 = layers[i + 1]
            result_lst = self._input_contraction(layer1,
                                                 layer2,
                                                 inline=inline)
            layers[i + 1] = result_lst

        return result_lst[0]


class ConvTree(Tree):
    """
    Class for Tree States where the input data is a batch of images. It is the
    convolutional version of :class:`Tree`.
    
    Input data as well as initialization parameters are described in `nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    sites_per_layer : list[int] or tuple[int]
        Number of sites in each layer of the tree. All nodes in the same layer
        have the same shape. Number of nodes in each layer times the number of
        input edges these have should match the number ot output edges in the
        previous layer.
    d_bond : list[list[int]] or tuple[tuple[int]]
        Bond dimensions of nodes in each layer. Each sequence corresponds to the
        shape of the nodes in each layer (some input edges and an output edge in
        the last position).
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
        
    Examples
    --------
    >>> conv_tree = tk.ConvTree(sites_per_layer=[2, 1],
    ...                         d_bond=[[2, 2, 3], [3, 3, 5]],
    ...                         kernel_size=2)
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_tree(data)
    >>> print(result.shape)
    torch.Size([20, 5, 1, 1])
    """
    
    def __init__(self,
                 sites_per_layer: Sequence[int],
                 d_bond: Sequence[Sequence[int]],
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
        
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        
        super().__init__(sites_per_layer=sites_per_layer,
                         d_bond=d_bond,
                         num_batches=2)
        self._in_channels = d_bond[0][0]
        
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)
    
    @property
    def in_channels(self) -> int:
        """
        Returns ``in_channels``. Same as the first elements in ``d_bond``
        from :class:`Tree`, corresponding to dimensions of the input.
        """
        return self._in_channels
    
    @property
    def kernel_size(self) -> Tuple[int, int]:
        """
        Returns ``kernel_size``, corresponding to number of ``data`` nodes.
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
    
    def forward(self, image, *args, **kwargs):
        r"""
        Overrides ``nn.Module``'s forward to compute a convolution on the input
        image.
        
        Parameters
        ----------
        image : torch.Tensor
            Input batch of images with shape
            
            .. math::
            
                batch\_size \times in\_channels \times height \times width
        args :
            Arguments that might be used in :meth:`~Tree.contract`.
        kwargs :
            Keyword arguments that might be used in :meth:`~Tree.contract`,
            like ``inline``.
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
    """
    Class for Uniform Tree States where the input data is a batch of images. It
    is the convolutional version of :class:`UTree`.
    
    Input data as well as initialization parameters are described in `nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    sites_per_layer : list[int] or tuple[int]
        Number of sites in each layer of the tree. All nodes have the same
        shape. Number of nodes in each layer times the number of input edges
        these have should match the number ot output edges in the previous
        layer.
    d_bond : list[int] or tuple[int]
        Bond dimensions of nodes in each layer. Since all nodes have the same
        shape, it is enough to pass a single sequence of dimensions (some input
        edges and an output edge in the last position).
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
    """
    
    def __init__(self,
                 sites_per_layer: Sequence[int],
                 d_bond: Sequence[int],
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
        
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        
        super().__init__(sites_per_layer=sites_per_layer,
                         d_bond=d_bond,
                         num_batches=2)
        self._in_channels = d_bond[0]
        
        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)
    
    @property
    def in_channels(self) -> int:
        """
        Returns ``in_channels``. Same as the first elements in ``d_bond``
        from :class:`UTree`, corresponding to dimensions of the input.
        """
        return self._in_channels
    
    @property
    def kernel_size(self) -> Tuple[int, int]:
        """
        Returns ``kernel_size``, corresponding to number of ``data`` nodes.
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
    
    def forward(self, image, *args, **kwargs):
        r"""
        Overrides ``nn.Module``'s forward to compute a convolution on the input
        image.
        
        Parameters
        ----------
        image : torch.Tensor
            Input batch of images with shape
            
            .. math::
            
                batch\_size \times in\_channels \times height \times width
        args :
            Arguments that might be used in :meth:`~UTree.contract`.
        kwargs :
            Keyword arguments that might be used in :meth:`~UTree.contract`,
            like ``inline``.
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
