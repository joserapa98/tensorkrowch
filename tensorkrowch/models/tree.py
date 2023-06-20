"""
This script contains:
    * Tree
    * UTree
    * ConvTree
    * ConvUTree
"""

from typing import (List, Optional, Sequence,
                    Text, Tuple, Union)

import torch
import torch.nn as nn

import tensorkrowch.operations as op
from tensorkrowch.components import Node, ParamNode
from tensorkrowch.components import TensorNetwork


class Tree(TensorNetwork):
    """
    Class for Tree States. These states form a tree structure where the ``data``
    nodes are in the base. All nodes have a sequence of input edges and an
    output edge. Thus the contraction of the Tree returns a `vector` node.
    
    All nodes in the network are in ``self.layers``, a list containing the lists
    of nodes in each layer (starting from the bottom).

    Parameters
    ----------
    sites_per_layer : list[int] or tuple[int]
        Number of sites in each layer of the tree. All nodes in the same layer
        have the same shape. Number of nodes in each layer times the number of
        input edges these have should match the number ot output edges in the
        previous layer. The last element of ``sites_per_layer`` should be always
        1, which corresponds to the output node.
    bond_dim : list[list[int]] or tuple[tuple[int]]
        Bond dimensions of nodes in each layer. Each sequence corresponds to the
        shape of the nodes in each layer, starting from the bottom (some input
        edges and an output edge in the last position).
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    
    Examples
    --------
    >>> tree = tk.models.Tree(sites_per_layer=[4, 2, 1],
    ...                       bond_dim=[[3, 3, 4], [4, 4, 2], [2, 2, 2]])
    >>> data = torch.ones(20, 8, 3) # batch_size x n_features x feature_size
    >>> result = tree(data)
    >>> result.shape
    torch.Size([20, 2])
    """

    def __init__(self,
                 sites_per_layer: Sequence[int],
                 bond_dim: Sequence[Sequence[int]],
                 n_batches: int = 1) -> None:

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

        # bond_dim
        if isinstance(bond_dim, (list, tuple)):
            aux_bond_dim = []
            for lst in bond_dim:
                if isinstance(lst, (list, tuple)):
                    if len(lst) < 2:
                        raise ValueError('`bond_dim` sequences should have at '
                                         'least two elements, one for input '
                                         'and one for output')
                    for el in lst:
                        if not isinstance(el, int):
                            raise TypeError('`bond_dim` should be a sequence of '
                                            'sequences of ints')
                else:
                    raise TypeError('`bond_dim` should be a sequence of '
                                    'sequences of ints')
                aux_bond_dim.append(list(lst))
        else:
            raise TypeError('`bond_dim` should be a sequence of sequences of '
                            'ints')
        self._bond_dim = aux_bond_dim

        if len(sites_per_layer) != len(bond_dim):
            raise ValueError('`sites_per_layer` and `bond_dim` should have the '
                             'same number of elements')

        # n_batches
        if not isinstance(n_batches, int):
            raise TypeError('`n_batches should be int type')
        self._n_batches = n_batches

        # Create Tensor Network
        self._make_nodes()
        self.initialize()

    @property
    def sites_per_layer(self) -> Sequence[int]:
        """Returns number of sites in each layer of the tree."""
        return self._sites_per_layer

    @property
    def bond_dim(self) -> Union[Sequence[Sequence[int]],
                                Sequence[Sequence[Sequence[int]]]]:
        """
        Returns bond dimensions of nodes in each layer. Each sequence
        corresponds to the shape of the nodes in each layer (some input edges
        and an output edge in the last position).
        
        It can have two forms:
        
        1) ``[shape_all_nodes_layer_1, ..., shape_all_nodes_layer_N]``
        2) ``[[shape_node_1_layer_1, ..., shape_node_i1_layer_1], ...,
           [shape_node_1_layer_N, ..., shape_node_iN_layer_N]]``
        """
        return self._bond_dim

    @property
    def n_batches(self) -> int:
        """Returns number of batch edges of the ``data`` nodes."""
        return self._n_batches

    def _make_nodes(self) -> None:
        """Creates all the nodes of the Tree."""
        if self._leaf_nodes:
            raise ValueError('Cannot create Tree nodes if the Tree already has'
                             ' nodes')

        self.layers = []

        for i, n_sites in enumerate(self.sites_per_layer):
            layer_lst = []
            for j in range(n_sites):
                node = ParamNode(shape=(*self.bond_dim[i],),
                                 axes_names=(*(['input'] * (
                                         len(self.bond_dim[i]) - 1)),
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
                tensor = torch.randn(node.shape) * std
                tensor[(0,) * node.rank] = 1.
                node.tensor = tensor

    def set_data_nodes(self) -> None:
        """
        Creates data nodes and connects each of them to the physical edge of
        an input node.
        """
        input_edges = []
        for node in self.layers[0]:
            input_edges += node.edges[:-1]

        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self.n_batches)

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
                            layer1: List[ParamNode],
                            layer2: List[ParamNode],
                            mode: Text = 'svd',
                            rank: Optional[int] = None,
                            cum_percentage: Optional[float] = None,
                            cutoff: Optional[float] = None) -> Tuple[List[ParamNode],
                                                                     List[ParamNode]]:
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
                    raise ValueError('`mode` can only be "svd", "svdr" or "qr"')

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
            
        Examples
        --------
        >>> tree = tk.models.Tree(sites_per_layer=[4, 2, 1],
        ...                       bond_dim=[[3, 3, 4], [4, 4, 2], [2, 2, 2]])
        >>> tree.canonicalize(rank=2)
        >>> tree.bond_dim
        [[[3, 3, 2], [3, 3, 2], [3, 3, 2], [3, 3, 2]], [[2, 2, 2], [2, 2, 2]], [[2, 2, 2]]]
        """
        if len(self.layers) > 1:
            self.reset()

            prev_auto_stack = self._auto_stack
            self.auto_stack = False

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

            bond_dim = []
            for layer in self.layers:
                layer_bond_dim = []
                for node in layer:
                    layer_bond_dim.append(list(node.shape))
                bond_dim.append(layer_bond_dim)
            self._bond_dim = bond_dim

            self.auto_stack = prev_auto_stack


class UTree(TensorNetwork):
    """
    Class for Uniform Tree States where all nodes have the same shape. It is
    the uniform version of :class:`Tree`, that is, all nodes share the same
    tensor.
    
    All nodes in the network are in ``self.layers``, a list containing the lists
    of nodes in each layer (starting from the bottom).

    Parameters
    ----------
    sites_per_layer : list[int] or tuple[int]
        Number of sites in each layer of the tree. All nodes have the same
        shape. Number of nodes in each layer times the number of input edges
        these have should match the number ot output edges in the previous
        layer. The last element of ``sites_per_layer`` should be always 1,
        which corresponds to the output node.
    bond_dim : list[int] or tuple[int]
        Bond dimensions of nodes in each layer. Since all nodes have the same
        shape, it is enough to pass a single sequence of dimensions (some input
        edges and an output edge in the last position).
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
        
    Examples
    --------
    >>> tree = tk.models.UTree(sites_per_layer=[4, 2, 1],
    ...                        bond_dim=[3, 3, 3])
    >>> for layer in tree.layers:
    ...     for node in layer:
    ...         assert node.tensor_address() == 'virtual_uniform'
    ...
    >>> data = torch.ones(20, 8, 3) # batch_size x n_features x feature_size
    >>> result = tree(data)
    >>> result.shape
    torch.Size([20, 3])
    """

    def __init__(self,
                 sites_per_layer: Sequence[int],
                 bond_dim: Sequence[int],
                 n_batches: int = 1) -> None:

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

        # bond_dim
        if isinstance(bond_dim, (list, tuple)):
            if len(bond_dim) < 2:
                raise ValueError('`bond_dim` should have at least two elements, '
                                 'one for input and one for output')
            for el in bond_dim:
                if not isinstance(el, int):
                    raise TypeError('`bond_dim` should be a sequence of ints')
        else:
            raise TypeError('`bond_dim` should be a sequence of ints')
        self._bond_dim = list(bond_dim)

        # n_batches
        if not isinstance(n_batches, int):
            raise TypeError('`n_batches should be int type')
        self._n_batches = n_batches

        # Create Tensor Network
        self._make_nodes()
        self.initialize()

    @property
    def sites_per_layer(self) -> Sequence[int]:
        """Returns number of sites in each layer of the tree."""
        return self._sites_per_layer

    @property
    def bond_dim(self) -> Sequence[int]:
        """Returns bond dimensions of nodes in each layer. Since all nodes have
        the same shape, it is a single sequence of dimensions (some input edges
        and an output edge in the last position).
        """
        return self._bond_dim

    @property
    def n_batches(self) -> int:
        """Returns number of batch edges of the ``data`` nodes."""
        return self._n_batches

    def _make_nodes(self) -> None:
        """Creates all the nodes of the Tree."""
        if self._leaf_nodes:
            raise ValueError('Cannot create Tree nodes if the Tree already has'
                             ' nodes')

        self.layers = []

        for i, n_sites in enumerate(self._sites_per_layer):
            layer_lst = []
            for j in range(n_sites):
                node = ParamNode(shape=(*self.bond_dim,),
                                 axes_names=(*(['input'] * (len(self.bond_dim) - 1)),
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
        uniform_memory = node = ParamNode(shape=(*self.bond_dim,),
                                          axes_names=(*(['input'] * (
                                                  len(self.bond_dim) - 1)),
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
        self.uniform_memory.tensor = tensor

        for layer in self.layers:
            for node in layer:
                node.set_tensor_from(self.uniform_memory)

    def set_data_nodes(self) -> None:
        """
        Creates data nodes and connects each of them to the physical edge of
        an input node.
        """
        input_edges = []
        for node in self.layers[0]:
            input_edges += node._edges[:-1]

        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self.n_batches)

    def _input_contraction(self,
                           layer1: List[Node],
                           layer2: List[ParamNode],
                           inline: bool) -> List[Node]:
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
    
    Input data as well as initialization parameters are described in `torch.nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    sites_per_layer : list[int] or tuple[int]
        Number of sites in each layer of the tree. All nodes in the same layer
        have the same shape. Number of nodes in each layer times the number of
        input edges these have should match the number ot output edges in the
        previous layer.
    bond_dim : list[list[int]] or tuple[tuple[int]]
        Bond dimensions of nodes in each layer. Each sequence corresponds to the
        shape of the nodes in each layer (some input edges and an output edge in
        the last position).
    kernel_size : int, list[int] or tuple[int]
        Kernel size used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    stride : int
        Stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
    padding : int
        Padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    dilation : int
        Dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
        
    Examples
    --------
    >>> conv_tree = tk.models.ConvTree(sites_per_layer=[2, 1],
    ...                                bond_dim=[[2, 2, 3], [3, 3, 5]],
    ...                                kernel_size=2)
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_tree(data)
    >>> print(result.shape)
    torch.Size([20, 5, 1, 1])
    """

    def __init__(self,
                 sites_per_layer: Sequence[int],
                 bond_dim: Sequence[Sequence[int]],
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
                         bond_dim=bond_dim,
                         n_batches=2)
        self._in_channels = bond_dim[0][0]

        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)

    @property
    def in_channels(self) -> int:
        """
        Returns ``in_channels``. Same as the first elements in ``bond_dim``
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
        Returns stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._stride

    @property
    def padding(self) -> Tuple[int, int]:
        """
        Returns padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._padding

    @property
    def dilation(self) -> Tuple[int, int]:
        """
        Returns dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._dilation

    def forward(self, image, *args, **kwargs):
        r"""
        Overrides ``torch.nn.Module``'s forward to compute a convolution on the
        input image.
        
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

        patches = patches.transpose(2, 3)
        # batch_size x nb_windows x nb_pixels x in_channels

        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows x out_channels

        result = result.transpose(1, 2)
        # batch_size x out_channels x nb_windows

        h_in = image.shape[2]
        w_in = image.shape[3]

        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] *
                     (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] *
                     (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x out_channels x height_out x width_out

        return result


class ConvUTree(UTree):
    """
    Class for Uniform Tree States where the input data is a batch of images. It
    is the convolutional version of :class:`UTree`.
    
    Input data as well as initialization parameters are described in `torch.nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    sites_per_layer : list[int] or tuple[int]
        Number of sites in each layer of the tree. All nodes have the same
        shape. Number of nodes in each layer times the number of input edges
        these have should match the number ot output edges in the previous
        layer.
    bond_dim : list[int] or tuple[int]
        Bond dimensions of nodes in each layer. Since all nodes have the same
        shape, it is enough to pass a single sequence of dimensions (some input
        edges and an output edge in the last position).
    kernel_size : int, list[int] or tuple[int]
        Kernel size used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    stride : int
        Stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
    padding : int
        Padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
    dilation : int
        Dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        If given as an ``int``, the actual kernel size will be
        ``(kernel_size, kernel_size)``.
        
    Examples
    --------
    >>> conv_tree = tk.models.ConvUTree(sites_per_layer=[2, 1],
    ...                                 bond_dim=[2, 2, 2],
    ...                                 kernel_size=2)
    >>> for layer in conv_tree.layers:
    ...     for node in layer:
    ...         assert node.tensor_address() == 'virtual_uniform'
    ...
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_tree(data)
    >>> print(result.shape)
    torch.Size([20, 2, 1, 1])
    """

    def __init__(self,
                 sites_per_layer: Sequence[int],
                 bond_dim: Sequence[int],
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
                         bond_dim=bond_dim,
                         n_batches=2)
        self._in_channels = bond_dim[0]

        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)

    @property
    def in_channels(self) -> int:
        """
        Returns ``in_channels``. Same as the first elements in ``bond_dim``
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
        Returns stride used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._stride

    @property
    def padding(self) -> Tuple[int, int]:
        """
        Returns padding used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._padding

    @property
    def dilation(self) -> Tuple[int, int]:
        """
        Returns dilation used in `torch.nn.Unfold
        <https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold>`_.
        """
        return self._dilation

    def forward(self, image, *args, **kwargs):
        r"""
        Overrides ``torch.nn.Module``'s forward to compute a convolution on the
        input image.
        
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

        patches = patches.transpose(2, 3)
        # batch_size x nb_windows x nb_pixels x in_channels

        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows x out_channels

        result = result.transpose(1, 2)
        # batch_size x out_channels x nb_windows

        h_in = image.shape[2]
        w_in = image.shape[3]

        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] *
                     (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] *
                     (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x out_channels x height_out x width_out

        return result
