"""
This script contains:
    * PEPS
    * UPEPS
    * ConvPEPS
    * ConvUPEPS
"""

from typing import (List, Sequence,
                    Text, Tuple, Union)

import torch
import torch.nn as nn

import tensorkrowch.operations as op
from tensorkrowch.components import Node, ParamNode
from tensorkrowch.components import TensorNetwork


class PEPS(TensorNetwork):
    """
    Class for Projected Entangled Pair States, where all nodes are input nodes,
    that is, they are all connected to ``data`` nodes that will store the input
    data tensor(s). When contracting the PEPS with new input data, the result
    will be just a number.
    
    A ``PEPS`` is formed by the following nodes:

    * ``grid_env``: Grid environment of nodes with 5 edges, ("input", "left",
      "up", "right", "down"). Is is a list of lists of nodes.

    * ``left_border``, ``right_border``, ``up_border``, ``down_border``:
      Border nodes with a single virtual edge, connected to the corresponding
      side of the grid when open boundary conditions are used.

    Parameters
    ----------
    n_rows : int
        Number of rows of the 2D grid.
    n_cols : int
        Number of columns of the 2D grid.
    phys_dim : int
        Physical dimension.
    bond_dim : list[int] or tuple[int]
        Bond dimensions for horizontal and vertical edges (in that order). Thus
        it should contain 2 elements.
    boundary : list[{"obc", "pbc"}]
        List of strings indicating whether periodic or open boundary conditions
        should be used in the horizontal (up and down) and vertical (left and
        right) boundaries.
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    parameterized : bool, optional
        Boolean indicating whether PEPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    
    Examples
    --------
    >>> peps = tk.models.PEPS(n_rows=2,
    ...                       n_cols=2,
    ...                       phys_dim=3,
    ...                       bond_dim=[5, 5])
    >>> data = torch.ones(20, 4, 3) # batch_size x n_features x feature_size
    >>> result = peps(data)
    >>> result.shape
    torch.Size([20])
    """

    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 phys_dim: int,
                 bond_dim: Sequence[int],
                 boundary: Sequence[Text] = ['obc', 'obc'],
                 n_batches: int = 1,
                 parameterized: bool = True) -> None:

        super().__init__(name='peps')

        # n_rows
        if not isinstance(n_rows, int):
            raise TypeError('`n_rows` should be int type')
        elif n_rows < 1:
            raise ValueError('`n_rows` should be at least 1')

        # n_cols
        if not isinstance(n_cols, int):
            raise TypeError('`n_cols` should be int type')
        elif n_cols < 1:
            raise ValueError('`n_cols` should be at least 1')

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
                                 'there has to be one row')
        else:
            raise ValueError('`boundary` elements should be one of "obc" or '
                             '"pbc"')

        if boundary[1] == 'obc':
            if n_cols < 2:
                raise ValueError('If `boundary` of columns is "obc", at least '
                                 'there has to be 2 columns')
        elif boundary[1] == 'pbc':
            if n_cols < 1:
                raise ValueError('If `boundary` of columns is "pbc", at least '
                                 'there has to be one column')
        else:
            raise ValueError('`boundary` elements should be one of "obc" or '
                             '"pbc"')

        self._n_rows = n_rows
        self._n_cols = n_cols
        self._boundary = boundary

        # phys_dim
        if not isinstance(phys_dim, int):
            raise TypeError('`phys_dim` should be int type')
        self._phys_dim = phys_dim

        # bond_dim
        if isinstance(bond_dim, (list, tuple)):
            if len(bond_dim) != 2:
                raise ValueError('`bond_dim` should be a pair of ints')
            self._bond_dim = list(bond_dim)
        else:
            raise TypeError('`bond_dim` should be a pair of ints')

        # n_batches
        if not isinstance(n_batches, int):
            raise TypeError('`n_batches` should be int type')
        self._n_batches = n_batches

        # Properties
        self._left_border = []
        self._right_border = []
        self._up_border = []
        self._down_border = []
        self._grid_env = []

        if not isinstance(parameterized, bool):
            raise TypeError('`parameterized` should be bool type')

        # Create Tensor Network
        self._make_nodes(parameterized)
        self.initialize()

    @property
    def n_rows(self) -> int:
        """Returns number of rows of the 2D grid."""
        return self._n_rows

    @property
    def n_cols(self) -> int:
        """Returns number of columns of the 2D grid."""
        return self._n_cols

    @property
    def boundary(self) -> List[Text]:
        """
        Returns boundary conditions in the horizontal (up and down) and
        vertical (left and right) boundaries.
        """
        return self._boundary

    @property
    def phys_dim(self) -> int:
        """Returns physical dimension."""
        return self._phys_dim

    @property
    def bond_dim(self) -> List[int]:
        """Returns bond dimensions for horizontal and vertical edges."""
        return self._bond_dim

    @property
    def n_batches(self) -> int:
        """Returns number of batch edges of the ``data`` nodes."""
        return self._n_batches

    @property
    def left_border(self) -> List[Node]:
        """Returns the nodes connected to the left side of the grid."""
        return self._left_border

    @property
    def right_border(self) -> List[Node]:
        """Returns the nodes connected to the right side of the grid."""
        return self._right_border

    @property
    def up_border(self) -> List[Node]:
        """Returns the nodes connected to the upper side of the grid."""
        return self._up_border

    @property
    def down_border(self) -> List[Node]:
        """Returns the nodes connected to the lower side of the grid."""
        return self._down_border

    @property
    def grid_env(self) -> List[List[Node]]:
        """Returns the grid environment of PEPS nodes."""
        return self._grid_env

    def _make_nodes(self, parameterized: bool = True) -> None:
        """Creates all the nodes of the PEPS."""
        if self.leaf_nodes:
            raise ValueError('Cannot create PEPS nodes if the PEPS already has'
                             ' nodes')

        self._left_border = []
        self._right_border = []
        self._up_border = []
        self._down_border = []

        self._grid_env = []
        node_cls = ParamNode if parameterized else Node

        phys_dim = self._phys_dim
        bond_dim = self._bond_dim

        for i in range(self._n_rows):
            self._grid_env.append([])
            for j in range(self._n_cols):
                node = node_cls(shape=(phys_dim, bond_dim[0], bond_dim[1],
                                       bond_dim[0], bond_dim[1]),
                                axes_names=('input', 'left', 'up',
                                            'right', 'down'),
                                name=f'grid_env_node_({i},{j})',
                                network=self)
                self._grid_env[-1].append(node)

                if i > 0:
                    self._grid_env[i - 1][j]['down'] ^ node['up']
                if j > 0:
                    self._grid_env[i][j - 1]['right'] ^ node['left']

        if self._boundary[0] == 'pbc':
            for j in range(self._n_cols):
                self._grid_env[-1][j]['down'] ^ self._grid_env[0][j]['up']
        else:
            for j in range(self._n_cols):
                node = Node(shape=(bond_dim[1],),
                            axes_names=('down',),
                            name=f'up_border_node_({j})',
                            network=self)
                self._up_border.append(node)
                node['down'] ^ self._grid_env[0][j]['up']

                node = Node(shape=(bond_dim[1],),
                            axes_names=('up',),
                            name=f'down_border_node_({j})',
                            network=self)
                self._down_border.append(node)
                self._grid_env[-1][j]['down'] ^ node['up']

        if self._boundary[1] == 'pbc':
            for i in range(self._n_rows):
                self._grid_env[i][-1]['right'] ^ self._grid_env[i][0]['left']
        else:
            for i in range(self._n_rows):
                node = Node(shape=(bond_dim[0],),
                            axes_names=('right',),
                            name=f'left_border_node_({i})',
                            network=self)
                self._left_border.append(node)
                node['right'] ^ self._grid_env[i][0]['left']

                node = Node(shape=(bond_dim[0],),
                            axes_names=('left',),
                            name=f'right_border_node_({i})',
                            network=self)
                self._right_border.append(node)
                self._grid_env[i][-1]['right'] ^ node['left']

    def initialize(self, std: float = 1e-9) -> None:
        """Initializes all the nodes."""
        for node in self.leaf_nodes.values():
            node.tensor = torch.randn(node.shape) * std

    def set_data_nodes(self) -> None:
        """
        Creates data nodes and connects each of them to the physical edge of
        an input node.
        """
        input_edges = [node['input']
                       for row in self._grid_env
                       for node in row]

        super().set_data_nodes(input_edges=input_edges,
                               num_batch_edges=self._n_batches)

    def _input_contraction(self,
                           inline_input: bool = False) -> List[List[Node]]:
        """Contracts input data nodes with PEPS nodes."""
        full_grid = []

        if inline_input:
            for row in self._grid_env:
                result_row = []
                for node in row:
                    result_row.append(node.neighbours('input') @ node)
                full_grid.append(result_row)
        else:
            list_grid_env = []
            for lst in self._grid_env:
                list_grid_env += lst

            if list_grid_env:
                stack_grid_env = op.stack(list_grid_env)
                stack_grid_env_data = op.stack(
                    list(map(lambda x: x.neighbours('input'),
                             list_grid_env)))
                stack_grid_env_data['feature'] ^ stack_grid_env['input']
                result_grid_env = op.unbind(stack_grid_env_data @ stack_grid_env)

                for i in range(self._n_rows):
                    row = []
                    for j in range(self._n_cols):
                        row.append(result_grid_env[i * self._n_cols + j])
                    full_grid.append(row)
        
        if self._boundary[0] == 'obc':
            for j, border in enumerate(self._up_border):
                full_grid[0][j] = border @ full_grid[0][j]
            for j, border in enumerate(self._down_border):
                full_grid[-1][j] = full_grid[-1][j] @ border
        
        if self._boundary[1] == 'obc':
            for i, border in enumerate(self._left_border):
                full_grid[i][0] = border @ full_grid[i][0]
            for i, border in enumerate(self._right_border):
                full_grid[i][-1] = full_grid[i][-1] @ border

        return full_grid

    def _zipup_contraction(self,
                           line1: List[Node],
                           line2: List[Node],
                           from_side: Text = 'up',
                           max_bond: int = 32) -> List[Node]:
        """Contracts two consecutive lines of the PEPS via boundary-MPS zip-up."""
        if from_side in ['up', 'down']:
            in_axis = 'left'
            out_axis = 'right'
            pbc = self._boundary[1] == 'pbc'
        else:
            in_axis = 'up'
            out_axis = 'down'
            pbc = self._boundary[0] == 'pbc'

        result_line = []
        carry = None

        for i, (node1, node2) in enumerate(zip(line1, line2)):
            if carry is None:
                local = node1 @ node2
            else:
                local = carry @ node1
                local @= node2

            if i < len(line1) - 1:
                next_edges = op.get_shared_edges(local, line1[i + 1]) + \
                    op.get_shared_edges(local, line2[i + 1])
            elif pbc and (len(line1) > 1):
                next_edges = op.get_shared_edges(local, result_line[0])
            else:
                next_edges = []

            if not next_edges:
                result_line.append(local)
                carry = None
                continue

            node2_axes = [axis
                          for axis, edge in zip(local.axes, local.edges)
                          if edge in next_edges]
            node1_axes = [axis
                          for axis, edge in zip(local.axes, local.edges)
                          if (not edge.is_batch()) and (edge not in next_edges)]

            split1, split2 = local.split(node1_axes=node1_axes,
                                         node2_axes=node2_axes,
                                         rank=max_bond)
            split1.get_axis('split').name = out_axis
            split2.get_axis('split').name = in_axis

            result_line.append(split1)
            carry = split2

        if pbc and (len(result_line) > 1) and (carry is not None):
            result_line[0] = carry @ result_line[0]

        return result_line

    def contract(self,
                 from_side: Text = 'up',
                 max_bond: int = 32,
                 inline_input: bool = False):
        """
        Contracts the whole PEPS.
        
        Parameters
        ----------
        from_side : {"up", "down", "left", "right"}
            Indicates from which side of the 2D grid the contraction algorithm
            should start.
        max_bond : int
            The maximum allowed bond dimension. If, when contracting consecutive
            lines (rows or columns) of the PEPS this bond dimension is exceeded,
            the bond dimension is reduced using singular value decomposition
            (see :func:`split`).
        inline_input : bool
            Boolean indicating whether input data nodes should be contracted
            with the grid nodes one by one or in parallel using a single
            stacked contraction.

        Returns
        -------
        Node
        """
        full_grid = self._input_contraction(inline_input=inline_input)

        if from_side == 'up':
            pass

        elif from_side == 'down':
            full_grid.reverse()

        elif from_side == 'left':
            new_grid = []
            for j in range(len(full_grid[0])):
                row = []
                for i in range(len(full_grid)):
                    row.append(full_grid[i][j])
                new_grid.append(row)
            full_grid = new_grid

        elif from_side == 'right':
            new_grid = []
            for j in range(len(full_grid[0])):
                row = []
                for i in range(len(full_grid)):
                    row.append(full_grid[i][j])
                new_grid.append(row)
            full_grid = new_grid
            full_grid.reverse()

        for i in range(len(full_grid) - 1):
            line1 = full_grid[i]
            line2 = full_grid[i + 1]
            full_grid[i + 1] = self._zipup_contraction(line1=line1,
                                                       line2=line2,
                                                       from_side=from_side,
                                                       max_bond=max_bond)

        result = full_grid[-1][0]
        for node in full_grid[-1][1:]:
            result @= node

        for edge in result.edges:
            if not edge.is_batch():
                result @= result
                break

        return result


class UPEPS(PEPS):
    """
    Class for Uniform (translationally invariant) Projected Entangled Pair
    States, where all nodes are input nodes. It is the uniform version of
    :class:`PEPS`, that is, all nodes share the same tensor. Thus boundary
    conditions are always periodic.
    
    A ``UPEPS`` is formed by the following nodes:
    
    * ``grid_env``: Grid environment of nodes with 5 edges, ("input", "left",
      "up", "right", "down"). Is is a list of lists of nodes.

    Parameters
    ----------
    n_rows : int
        Number of rows of the 2D grid.
    n_cols : int
        Number of columns of the 2D grid
    phys_dim : int
        Physical dimension.
    bond_dim : list[int] or tuple[int]
        Bond dimensions for horizontal and vertical edges (in that order). Thus
        it should also contain 2 elements
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``nu_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    parameterized : bool, optional
        Boolean indicating whether UPEPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
        
    Examples
    --------
    >>> peps = tk.models.PEPS(n_rows=2,
    ...                       n_cols=2,
    ...                       phys_dim=3,
    ...                       bond_dim=[5, 5])
    >>> for node in peps.grid_env:
    ...     assert node.tensor_address() == 'virtual_uniform'
    ...
    >>> data = torch.ones(20, 4, 3) # batch_size x n_features x feature_size
    >>> result = peps(data)
    >>> result.shape
    torch.Size([20])
    """

    def __init__(self,
                 n_rows: int,
                 n_cols: int,
                 phys_dim: int,
                 bond_dim: Sequence[int],
                 n_batches: int = 1,
                 parameterized: bool = True) -> None:
        super().__init__(n_rows=n_rows,
                         n_cols=n_cols,
                         phys_dim=phys_dim,
                         bond_dim=bond_dim,
                         boundary=['pbc', 'pbc'],
                         n_batches=n_batches,
                         parameterized=parameterized)
        self.name = 'upeps'

    def _make_nodes(self, parameterized: bool = True) -> None:
        """Creates all the nodes of the PEPS."""
        super()._make_nodes(parameterized)

        # Virtual node
        node_cls = ParamNode if parameterized else Node
        uniform_memory = node_cls(shape=(self._phys_dim,
                                         self._bond_dim[0],
                                         self._bond_dim[1],
                                         self._bond_dim[0],
                                         self._bond_dim[1]),
                                  axes_names=('input', 'left', 'up',
                                              'right', 'down'),
                                  name='virtual_uniform',
                                  network=self,
                                  virtual=True)
        self.uniform_memory = uniform_memory

        for lst in self._grid_env:
            for node in lst:
                node.set_tensor_from(uniform_memory)

    def initialize(self, std: float = 1e-9) -> None:
        """Initializes all the nodes."""
        # Virtual node
        tensor = torch.randn(self.uniform_memory.shape) * std
        self.uniform_memory.tensor = tensor

class ConvPEPS(PEPS):
    """
    Class for Projected Entangled Pair States, where all nodes are input nodes,
    and where the input data is a batch of images. It is the convolutional
    version of :class:`PEPS`.
    
    Input data as well as initialization parameters are described in `torch.nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    in_channels : int
        Input channels. Same as ``phys_dim`` in :class:`PEPS`.
    bond_dim : list[int] or tuple[int]
        Bond dimensions for horizontal and vertical edges (in that order). Thus
        it should also contain 2 elements
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
    boundary : list[{"obc", "pbc"}]
        List of strings indicating whether periodic or open boundary conditions
        should be used in the horizontal (up and down) and vertical (left and
        right) boundaries.
    parameterized : bool, optional
        Boolean indicating whether ConvPEPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
        
    Examples
    --------
    >>> conv_peps = tk.models.ConvPEPS(in_channels=2,
    ...                                bond_dim=[5, 5],
    ...                                kernel_size=2)
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_peps(data)
    >>> result.shape
    torch.Size([20, 1, 1])
    """

    def __init__(self,
                 in_channels: int,
                 bond_dim: Sequence[int],
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 boundary: Sequence[Text] = ['obc', 'obc'],
                 parameterized: bool = True) -> None:

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

        super().__init__(n_rows=kernel_size[0],
                         n_cols=kernel_size[1],
                         phys_dim=in_channels,
                         bond_dim=bond_dim,
                         boundary=boundary,
                         n_batches=2,
                         parameterized=parameterized)

        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)

    @property
    def in_channels(self) -> int:
        """Returns ``in_channels``. Same as ``phys_dim`` in :class:`PEPS`."""
        return self._in_channels

    @property
    def kernel_size(self) -> Tuple[int, int]:
        r"""
        Returns ``kernel_size``. Number of rows and columns in the 2D grid is
        given by :math:`kernel\_size_0` and :math:`kernel\_size_1`, respectively.
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
        Overrides ``torch.nn.Module``'s forward to compute a convolution on the input
        image.
        
        Parameters
        ----------
        image : torch.Tensor
            Input batch of images with shape
            
            .. math::
            
                batch\_size \times in\_channels \times height \times width
        args :
            Arguments that might be used in :meth:`~PEPS.contract`.
        kwargs :
            Keyword arguments that might be used in :meth:`~PEPS.contract`,
            like ``from_size``, ``max_bond`` or ``inline_input``.
        """
        # Input image shape: batch_size x in_channels x height x width

        patches = self.unfold(image).transpose(1, 2)
        # batch_size x nb_windows x (in_channels * nb_pixels)

        patches = patches.view(*patches.shape[:-1], self.in_channels, -1)
        # batch_size x nb_windows x in_channels x nb_pixels

        patches = patches.transpose(2, 3)
        # batch_size x nb_windows x nb_pixels x in_channels

        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows

        h_in = image.shape[2]
        w_in = image.shape[3]

        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] *
                     (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] *
                     (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x height_out x width_out

        return result


class ConvUPEPS(UPEPS):
    """
    Class for Uniform Projected Entangled Pair States, where all nodes are input
    nodes, and where the input data is a batch of images. It is the convolutional
    version of :class:`UPEPS`.
    
    Input data as well as initialization parameters are described in `torch.nn.Conv2d
    <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_.

    Parameters
    ----------
    in_channels : int
        Input channels. Same as ``phys_dim`` in :class:`UPEPS`.
    bond_dim : list[int] or tuple[int]
        Bond dimensions for horizontal and vertical edges (in that order). Thus
        it should also contain 2 elements
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
    parameterized : bool, optional
        Boolean indicating whether ConvUPEPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
        
    Examples
    --------
    >>> conv_peps = tk.models.ConvPEPS(in_channels=2,
    ...                                bond_dim=[5, 5],
    ...                                kernel_size=2)
    >>> for node in conv_peps.grid_env:
    ...     assert node.tensor_address() == 'virtual_uniform'
    ...
    >>> data = torch.ones(20, 2, 2, 2) # batch_size x in_channels x height x width
    >>> result = conv_peps(data)
    >>> result.shape
    torch.Size([20, 1, 1])
    """

    def __init__(self,
                 in_channels: int,
                 bond_dim: Union[int, Sequence[int]],
                 kernel_size: Union[int, Sequence],
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 parameterized: bool = True) -> None:

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

        super().__init__(n_rows=kernel_size[0],
                         n_cols=kernel_size[1],
                         phys_dim=in_channels,
                         bond_dim=bond_dim,
                         n_batches=2,
                         parameterized=parameterized)

        self.unfold = nn.Unfold(kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation)

    @property
    def in_channels(self) -> int:
        """Returns ``in_channels``. Same as ``phys_dim`` in :class:`UPEPS`."""
        return self._in_channels

    @property
    def kernel_size(self) -> Tuple[int, int]:
        r"""
        Returns ``kernel_size``. Number of rows and columns in the 2D grid is
        given by :math:`kernel\_size_0` and :math:`kernel\_size_1`, respectively.
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
        Overrides ``torch.nn.Module``'s forward to compute a convolution on the input
        image.
        
        Parameters
        ----------
        image : torch.Tensor
            Input batch of images with shape
            
            .. math::
            
                batch\_size \times in\_channels \times height \times width
        args :
            Arguments that might be used in :meth:`~PEPS.contract`.
        kwargs :
            Keyword arguments that might be used in :meth:`~PEPS.contract`,
            like ``from_size``, ``max_bond`` or ``inline_input``.
        """
        # Input image shape: batch_size x in_channels x height x width

        patches = self.unfold(image).transpose(1, 2)
        # batch_size x nb_windows x (in_channels * nb_pixels)

        patches = patches.view(*patches.shape[:-1], self.in_channels, -1)
        # batch_size x nb_windows x in_channels x nb_pixels

        patches = patches.transpose(2, 3)
        # batch_size x nb_windows x nb_pixels x in_channels

        result = super().forward(patches, *args, **kwargs)
        # batch_size x nb_windows

        h_in = image.shape[2]
        w_in = image.shape[3]

        h_out = int((h_in + 2 * self.padding[0] - self.dilation[0] *
                     (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w_in + 2 * self.padding[1] - self.dilation[1] *
                     (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        result = result.view(*result.shape[:-1], h_out, w_out)
        # batch_size x height_out x width_out

        return result
