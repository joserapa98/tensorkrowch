"""
This script contains:
    * PEPS
    * UPEPS
    * ConvPEPS
    * ConvUPEPS
"""

from typing import (List, Optional, Sequence,
                    Text, Tuple, Union)

import warnings

import torch
import torch.nn as nn

import tensorkrowch.operations as op
from tensorkrowch.components import Node, ParamNode
from tensorkrowch.components import TensorNetwork


class PEPS(TensorNetwork):  # MARK: PEPS
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
    tensors : list[list[torch.Tensor]] or tuple[tuple[torch.Tensor]], optional
        Instead of providing ``n_rows``, ``n_cols``, ``phys_dim``, ``bond_dim``
        and ``boundary``, a list of lists of PEPS tensors can be provided. In
        such case, all mentioned attributes will be inferred from the given
        tensors. Tensors should follow axis order ``("input", "left", "up",
        "right", "down")``. If a boundary is open, tensors at that side should
        omit the corresponding axis. Hence, open-boundary corner tensors are
        rank-3, open-boundary side tensors are rank-4, and inner or periodic
        tensors are rank-5.
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``n_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
        Initialization method.
    parameterized : bool, optional
        Boolean indicating whether PEPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
    
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

    ``PEPS`` can also be initialized from a list of lists of tensors:

    >>> tensors = [[torch.randn(3, 5, 5, 5, 5) for _ in range(3)]
    ...            for _ in range(2)]
    >>> peps = tk.models.PEPS(tensors=tensors)
    """

    def __init__(self,
                 n_rows: Optional[int] = None,
                 n_cols: Optional[int] = None,
                 phys_dim: Optional[int] = None,
                 bond_dim: Optional[Sequence[int]] = None,
                 boundary: Sequence[Text] = ['obc', 'obc'],
                 tensors: Optional[Sequence[Sequence[torch.Tensor]]] = None,
                 n_batches: int = 1,
                 init_method: Optional[Text] = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs: float) -> None:

        super().__init__(name='peps')

        if tensors is None:
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

            if boundary[0] not in ['obc', 'pbc']:
                raise ValueError('`boundary` elements should be one of "obc" or '
                                 '"pbc"')

            if boundary[1] not in ['obc', 'pbc']:
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
                if (boundary[1] == 'obc') and (n_cols == 1):
                    self._bond_dim[0] = 1
                if (boundary[0] == 'obc') and (n_rows == 1):
                    self._bond_dim[1] = 1
            else:
                raise TypeError('`bond_dim` should be a pair of ints')

        else:
            n_rows, n_cols, phys_dim, bond_dim, boundary = \
                self._infer_shape_from_tensors(tensors, boundary=boundary)
            self._n_rows = n_rows
            self._n_cols = n_cols
            self._boundary = boundary
            self._phys_dim = phys_dim
            self._bond_dim = bond_dim

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
        self.initialize(tensors=tensors,
                        init_method=init_method,
                        device=device,
                        dtype=dtype,
                        **kwargs)

    # ----------
    # Properties
    # ----------
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

    @property
    def tensors(self) -> List[List[torch.Tensor]]:
        """Returns the list of lists of PEPS tensors."""
        tensors = []
        for i, row in enumerate(self._grid_env):
            tensor_row = []
            for j, node in enumerate(row):
                tensor_row.append(self._reduced_tensor(node.tensor, i, j))
            tensors.append(tensor_row)
        return tensors
    
    # -------
    # Methods
    # -------
    def _candidate_from_tensors(self,
                                tensors: Sequence[Sequence[torch.Tensor]],
                                boundary: Sequence[Text]):
        """Checks whether tensors match a candidate boundary convention."""
        n_rows = len(tensors)
        n_cols = len(tensors[0])
        phys_dim = None
        horizontal_bond_dim = None
        vertical_bond_dim = None

        for i, row in enumerate(tensors):
            for j, tensor in enumerate(row):
                if not isinstance(tensor, torch.Tensor):
                    return None

                shape = list(tensor.shape)
                if len(shape) < 1:
                    return None

                if phys_dim is None:
                    phys_dim = shape[0]
                elif shape[0] != phys_dim:
                    return None

                axis = 1
                if (boundary[1] == 'pbc') or (j > 0):
                    if axis >= len(shape):
                        return None
                    if horizontal_bond_dim is None:
                        horizontal_bond_dim = shape[axis]
                    elif shape[axis] != horizontal_bond_dim:
                        return None
                    axis += 1

                if (boundary[0] == 'pbc') or (i > 0):
                    if axis >= len(shape):
                        return None
                    if vertical_bond_dim is None:
                        vertical_bond_dim = shape[axis]
                    elif shape[axis] != vertical_bond_dim:
                        return None
                    axis += 1

                if (boundary[1] == 'pbc') or (j < n_cols - 1):
                    if axis >= len(shape):
                        return None
                    if horizontal_bond_dim is None:
                        horizontal_bond_dim = shape[axis]
                    elif shape[axis] != horizontal_bond_dim:
                        return None
                    axis += 1

                if (boundary[0] == 'pbc') or (i < n_rows - 1):
                    if axis >= len(shape):
                        return None
                    if vertical_bond_dim is None:
                        vertical_bond_dim = shape[axis]
                    elif shape[axis] != vertical_bond_dim:
                        return None
                    axis += 1

                if axis != len(shape):
                    return None

        if horizontal_bond_dim is None:
            horizontal_bond_dim = 1
        if vertical_bond_dim is None:
            vertical_bond_dim = 1

        return phys_dim, [horizontal_bond_dim, vertical_bond_dim]

    def _infer_shape_from_tensors(self,
                                  tensors: Sequence[Sequence[torch.Tensor]],
                                  boundary: Optional[Sequence[Text]] = None):
        """Infers PEPS metadata from a rectangular grid of tensors."""
        if not isinstance(tensors, Sequence):
            raise TypeError('`tensors` should be a sequence of sequences of '
                            'torch.Tensor')
        if len(tensors) == 0:
            raise ValueError('`tensors` should contain at least one row')
        if not isinstance(tensors[0], Sequence) or len(tensors[0]) == 0:
            raise ValueError('`tensors` rows should be non-empty sequences')

        n_rows = len(tensors)
        n_cols = len(tensors[0])
        for row in tensors:
            if not isinstance(row, Sequence):
                raise TypeError('`tensors` should be a sequence of sequences '
                                'of torch.Tensor')
            if len(row) != n_cols:
                raise ValueError('All rows in `tensors` should have the same '
                                 'number of elements')

        boundary_candidates = []
        for candidate_boundary in (['obc', 'obc'], ['obc', 'pbc'],
                                   ['pbc', 'obc'], ['pbc', 'pbc']):
            candidate = self._candidate_from_tensors(tensors,
                                                     candidate_boundary)
            if candidate is not None:
                phys_dim, bond_dim = candidate
                boundary_candidates.append((phys_dim, bond_dim,
                                            candidate_boundary))

        if not boundary_candidates:
            raise ValueError('Could not infer a valid PEPS layout from '
                             '`tensors`')
        if len(boundary_candidates) > 1:
            if boundary is not None:
                for candidate in boundary_candidates:
                    if list(boundary) == candidate[2]:
                        return (n_rows, n_cols, candidate[0], candidate[1],
                                candidate[2])
            raise ValueError('Ambiguous PEPS boundary conditions inferred from '
                             '`tensors`')

        phys_dim, bond_dim, boundary = boundary_candidates[0]
        return n_rows, n_cols, phys_dim, bond_dim, boundary
    
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
                node = node_cls(shape=(phys_dim,
                                       bond_dim[0], bond_dim[1],
                                       bond_dim[0], bond_dim[1]),
                                axes_names=('input',
                                            'left', 'up',
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
    
    def _expected_tensor_shape(self,
                               n_rows: int,
                               n_cols: int,
                               boundary: Sequence[Text],
                               i: int,
                               j: int,
                               phys_dim: int,
                               bond_dim: Sequence[int]) -> Tuple[int, ...]:
        """Computes the reduced tensor shape expected at a grid position."""
        shape = [phys_dim]
        if (boundary[1] == 'pbc') or (j > 0):
            shape.append(bond_dim[0])
        if (boundary[0] == 'pbc') or (i > 0):
            shape.append(bond_dim[1])
        if (boundary[1] == 'pbc') or (j < n_cols - 1):
            shape.append(bond_dim[0])
        if (boundary[0] == 'pbc') or (i < n_rows - 1):
            shape.append(bond_dim[1])
        return tuple(shape)
    
    def _embed_obc_tensor(self,
                          tensor: torch.Tensor,
                          i: int,
                          j: int,
                          device: Optional[torch.device] = None,
                          dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Embeds a reduced OBC tensor into the full internal grid-node shape."""
        is_left_border = (self._boundary[1] == 'obc') and (j == 0)
        is_up_border = (self._boundary[0] == 'obc') and (i == 0)
        is_right_border = (self._boundary[1] == 'obc') and \
            (j == self._n_cols - 1)
        is_down_border = (self._boundary[0] == 'obc') and \
            (i == self._n_rows - 1)
        
        if not (is_left_border or is_up_border or
                is_right_border or is_down_border):
            return tensor
        
        if device is None:
            device = tensor.device
        if dtype is None:
            dtype = tensor.dtype
        
        aux_tensor = torch.zeros(*self._grid_env[i][j].shape,
                                 device=device,
                                 dtype=dtype)
        
        selection = [slice(None)]
        if is_left_border:
            selection.append(0)
        else:
            selection.append(slice(None))
        if is_up_border:
            selection.append(0)
        else:
            selection.append(slice(None))
        if is_right_border:
            selection.append(0)
        else:
            selection.append(slice(None))
        if is_down_border:
            selection.append(0)
        else:
            selection.append(slice(None))
        
        aux_tensor[tuple(selection)] = tensor
        return aux_tensor

    def _reduced_tensor(self,
                        tensor: torch.Tensor,
                        i: int,
                        j: int) -> torch.Tensor:
        """Extracts the public reduced tensor from the full internal tensor."""
        selection = [slice(None)]
        if (self._boundary[1] == 'obc') and (j == 0):
            selection.append(0)
        else:
            selection.append(slice(None))
        if (self._boundary[0] == 'obc') and (i == 0):
            selection.append(0)
        else:
            selection.append(slice(None))
        if (self._boundary[1] == 'obc') and (j == self._n_cols - 1):
            selection.append(0)
        else:
            selection.append(slice(None))
        if (self._boundary[0] == 'obc') and (i == self._n_rows - 1):
            selection.append(0)
        else:
            selection.append(slice(None))
        
        return tensor[tuple(selection)]
    
    def initialize(self,
                   tensors: Optional[Sequence[Sequence[torch.Tensor]]] = None,
                   init_method: Optional[Text] = 'randn',
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None,
                   **kwargs: float) -> None:
        """
        Initializes all the nodes.

        Parameters
        ----------
        tensors : list[list[torch.Tensor]] or tuple[tuple[torch.Tensor]], optional
            Sequence of sequences of tensors to set in each of the PEPS nodes.
            Tensor axes follow the order ``("input", "left", "up", "right",
            "down")``. Axes corresponding to open-boundary sides should be
            omitted.
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensors if ``init_method`` is provided.
        dtype : torch.dtype, optional
            Dtype of the tensor if ``init_method`` is provided.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        """
        if tensors is not None:
            if len(tensors) != self._n_rows:
                raise ValueError('`tensors` should have `n_rows` rows')
            
            device = tensors[0][0].device
            dtype = tensors[0][0].dtype
            for i, row in enumerate(tensors):
                if len(row) != self._n_cols:
                    raise ValueError('Each row in `tensors` should have '
                                     '`n_cols` elements')
                for j, tensor in enumerate(row):
                    expected_shape = self._expected_tensor_shape(
                        n_rows=self._n_rows,
                        n_cols=self._n_cols,
                        boundary=self._boundary,
                        i=i,
                        j=j,
                        phys_dim=self._phys_dim,
                        bond_dim=self._bond_dim)
                    if tuple(tensor.shape) != expected_shape:
                        raise ValueError('`tensors` elements have incorrect '
                                         'shapes for the PEPS layout')
                    self._grid_env[i][j].tensor = self._embed_obc_tensor(
                        tensor=tensor,
                        i=i,
                        j=j,
                        device=device,
                        dtype=dtype)
        elif init_method is not None:
            for i, row in enumerate(self._grid_env):
                for j, node in enumerate(row):
                    node.set_tensor(init_method=init_method,
                                    device=device,
                                    dtype=dtype,
                                    **kwargs)
                    if 'obc' in self._boundary:
                        node.tensor = self._embed_obc_tensor(
                            tensor=self._reduced_tensor(node.tensor, i, j),
                            i=i,
                            j=j,
                            device=device,
                            dtype=dtype)

        for node in self._up_border + self._down_border + \
                self._left_border + self._right_border:
            node.set_tensor(init_method='copy', device=device, dtype=dtype)

    def copy(self, share_tensors: bool = False) -> 'PEPS':
        """
        Creates a copy of the :class:`PEPS`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether tensors in the copied PEPS should be
            set as the tensors in the current PEPS (``True``), or cloned
            (``False``). In the former case, tensors in both PEPS's will be
            the same, which might be useful if one needs more than one copy
            of an PEPS, but wants to compute all the gradients with respect
            to the same, unique, tensors.

        Returns
        -------
        PEPS
        """
        new_peps = PEPS(n_rows=self._n_rows,
                        n_cols=self._n_cols,
                        phys_dim=self._phys_dim,
                        bond_dim=self._bond_dim,
                        boundary=self._boundary,
                        tensors=None,
                        n_batches=self._n_batches,
                        init_method=None,
                        device=None,
                        dtype=None)
        new_peps.name = self.name + '_copy'

        for i in range(self._n_rows):
            for j in range(self._n_cols):
                new_peps._grid_env[i][j] = new_peps._grid_env[i][j].parameterize(
                    set_param=isinstance(self._grid_env[i][j], ParamNode))

        if share_tensors:
            for new_row, row in zip(new_peps._grid_env, self._grid_env):
                for new_node, node in zip(new_row, row):
                    new_node.tensor = node.tensor
            for new_node, node in zip(new_peps._up_border, self._up_border):
                new_node.tensor = node.tensor
            for new_node, node in zip(new_peps._down_border, self._down_border):
                new_node.tensor = node.tensor
            for new_node, node in zip(new_peps._left_border, self._left_border):
                new_node.tensor = node.tensor
            for new_node, node in zip(new_peps._right_border, self._right_border):
                new_node.tensor = node.tensor
        else:
            for new_row, row in zip(new_peps._grid_env, self._grid_env):
                for new_node, node in zip(new_row, row):
                    new_node.tensor = node.tensor.clone()
            for new_node, node in zip(new_peps._up_border, self._up_border):
                new_node.tensor = node.tensor.clone()
            for new_node, node in zip(new_peps._down_border, self._down_border):
                new_node.tensor = node.tensor.clone()
            for new_node, node in zip(new_peps._left_border, self._left_border):
                new_node.tensor = node.tensor.clone()
            for new_node, node in zip(new_peps._right_border, self._right_border):
                new_node.tensor = node.tensor.clone()

        return new_peps

    def parameterize(self,
                     set_param: bool = True,
                     override: bool = False) -> 'TensorNetwork':
        """
        Parameterizes all grid nodes of the PEPS. If there are ``resultant``
        nodes in the PEPS, it will be first :meth:`~tensorkrowch.TensorNetwork.reset`.

        Parameters
        ----------
        set_param : bool
            Boolean indicating whether the tensor network has to be parameterized
            (``True``) or de-parameterized (``False``).
        override : bool
            Boolean indicating whether the tensor network should be parameterized
            in-place (``True``) or copied and then parameterized (``False``).
        """
        if self._resultant_nodes:
            warnings.warn(
                'Resultant nodes will be removed before parameterizing the TN')
            self.reset()

        if override:
            net = self
        else:
            net = self.copy(share_tensors=False)

        for i in range(net._n_rows):
            for j in range(net._n_cols):
                net._grid_env[i][j] = net._grid_env[i][j].parameterize(
                    set_param=set_param)

        return net

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


class UPEPS(PEPS):  # MARK: UPEPS
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
    phys_dim : int, optional
        Physical dimension. If ``tensor`` is provided, it is inferred from it.
    bond_dim : list[int] or tuple[int], optional
        Bond dimensions for horizontal and vertical edges (in that order). Thus
        it should also contain 2 elements. If ``tensor`` is provided, it is
        inferred from it.
    tensor : torch.Tensor, optional
        Tensor to set in the UPEPS ``uniform_memory`` node. It should be rank-5
        with shape ``(phys_dim, bond_dim[0], bond_dim[1], bond_dim[0],
        bond_dim[1])``.
    n_batches : int
        Number of batch edges of input ``data`` nodes. Usually ``n_batches = 1``
        (where the batch edge is used for the data batched) but it could also
        be ``nu_batches = 2`` (one edge for data batched, other edge for image
        patches in convolutional layers).
    init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
        Initialization method.
    parameterized : bool, optional
        Boolean indicating whether UPEPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensor if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
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
                 phys_dim: Optional[int] = None,
                 bond_dim: Optional[Sequence[int]] = None,
                 tensor: Optional[torch.Tensor] = None,
                 n_batches: int = 1,
                 init_method: Optional[Text] = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs: float) -> None:
        tensors = None

        if tensor is not None:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError('`tensor` should be torch.Tensor type')
            if len(tensor.shape) != 5:
                raise ValueError('`tensor` should be a rank-5 tensor')
            if tensor.shape[1] != tensor.shape[3]:
                raise ValueError('`tensor` left and right dimensions should '
                                 'be equal so that the PEPS can have '
                                 'periodic boundary conditions')
            if tensor.shape[2] != tensor.shape[4]:
                raise ValueError('`tensor` up and down dimensions should '
                                 'be equal so that the PEPS can have '
                                 'periodic boundary conditions')
            tensors = [[tensor for _ in range(n_cols)]
                       for _ in range(n_rows)]

        super().__init__(n_rows=n_rows,
                         n_cols=n_cols,
                         phys_dim=phys_dim,
                         bond_dim=bond_dim,
                         boundary=['pbc', 'pbc'],
                         tensors=tensors,
                         n_batches=n_batches,
                         init_method=init_method,
                         parameterized=parameterized,
                         device=device,
                         dtype=dtype,
                         **kwargs)
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

    def initialize(self,
                   tensors: Optional[Sequence[Sequence[torch.Tensor]]] = None,
                   init_method: Optional[Text] = 'randn',
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None,
                   **kwargs: float) -> None:
        """
        Initializes the ``uniform_memory`` node.

        Parameters
        ----------
        tensors : list[list[torch.Tensor]] or tuple[tuple[torch.Tensor]], optional
            Sequence containing the shared tensor used to initialize the UPEPS.
            If more tensors are provided, only the first one is used.
        init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
            Initialization method.
        device : torch.device, optional
            Device where to initialize the tensor if ``init_method`` is provided.
        dtype : torch.dtype, optional
            Dtype of the tensor if ``init_method`` is provided.
        kwargs : float
            Keyword arguments for the different initialization methods. See
            :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        """
        if tensors is not None:
            if len(tensors) == 0:
                raise ValueError('`tensors` should contain at least one row')
            if not isinstance(tensors[0], Sequence) or len(tensors[0]) == 0:
                raise ValueError('`tensors` rows should be non-empty sequences')
            tensor = tensors[0][0]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError('`tensors` should contain torch.Tensor objects')
            if tuple(tensor.shape) != tuple(self.uniform_memory.shape):
                raise ValueError('`tensor` has incorrect shape for UPEPS')
            self.uniform_memory.tensor = tensor
        elif init_method is not None:
            self.uniform_memory.set_tensor(init_method=init_method,
                                           device=device,
                                           dtype=dtype,
                                           **kwargs)

    def copy(self, share_tensors: bool = False) -> 'UPEPS':
        """
        Creates a copy of the :class:`UPEPS`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether the common tensor in the copied UPEPS
            should be set as the tensor in the current UPEPS (``True``), or
            cloned (``False``). In the former case, the tensor in both UPEPS's
            will be the same, which might be useful if one needs more than one
            copy of a UPEPS, but wants to compute all the gradients with respect
            to the same, unique, tensor.

        Returns
        -------
        UPEPS
        """
        new_peps = UPEPS(n_rows=self._n_rows,
                         n_cols=self._n_cols,
                         phys_dim=self._phys_dim,
                         bond_dim=self._bond_dim,
                         tensor=None,
                         n_batches=self._n_batches,
                         init_method=None,
                         parameterized=isinstance(self.uniform_memory,
                                                  ParamNode),
                         device=None,
                         dtype=None)
        new_peps.name = self.name + '_copy'

        if share_tensors:
            new_peps.uniform_memory.tensor = self.uniform_memory.tensor
        else:
            new_peps.uniform_memory.tensor = self.uniform_memory.tensor.clone()

        return new_peps

    def parameterize(self,
                     set_param: bool = True,
                     override: bool = False) -> 'TensorNetwork':
        """
        Parameterizes all nodes of the UPEPS. If there are ``resultant`` nodes
        in the UPEPS, it will be first :meth:`~tensorkrowch.TensorNetwork.reset`.

        Parameters
        ----------
        set_param : bool
            Boolean indicating whether the tensor network has to be parameterized
            (``True``) or de-parameterized (``False``).
        override : bool
            Boolean indicating whether the tensor network should be parameterized
            in-place (``True``) or copied and then parameterized (``False``).
        """
        if self._resultant_nodes:
            warnings.warn(
                'Resultant nodes will be removed before parameterizing the TN')
            self.reset()

        if override:
            net = self
        else:
            net = self.copy(share_tensors=False)

        for i in range(net._n_rows):
            for j in range(net._n_cols):
                net._grid_env[i][j] = net._grid_env[i][j].parameterize(
                    set_param=set_param)

        net.uniform_memory = net.uniform_memory.parameterize(
            set_param=set_param)

        for row in net._grid_env:
            for node in row:
                node.set_tensor_from(net.uniform_memory)

        return net

class ConvPEPS(PEPS):  # MARK: ConvPEPS
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
    tensors : list[list[torch.Tensor]] or tuple[tuple[torch.Tensor]], optional
        Sequence of PEPS tensors from which ``kernel_size``, ``in_channels``,
        ``bond_dim`` and ``boundary`` can be inferred.
    init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
        Initialization method.
    parameterized : bool, optional
        Boolean indicating whether ConvPEPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensors if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
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
                 in_channels: Optional[int] = None,
                 bond_dim: Optional[Sequence[int]] = None,
                 kernel_size: Optional[Union[int, Sequence]] = None,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 boundary: Sequence[Text] = ['obc', 'obc'],
                 tensors: Optional[Sequence[Sequence[torch.Tensor]]] = None,
                 init_method: Optional[Text] = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs: float) -> None:

        if tensors is not None:
            n_rows, n_cols, phys_dim, inferred_bond_dim, inferred_boundary = \
                self._infer_shape_from_tensors(tensors, boundary=boundary)
            if kernel_size is None:
                kernel_size = (n_rows, n_cols)
            if in_channels is None:
                in_channels = phys_dim
            if bond_dim is None:
                bond_dim = inferred_bond_dim
            boundary = inferred_boundary

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
                         tensors=tensors,
                         n_batches=2,
                         init_method=init_method,
                         parameterized=parameterized,
                         device=device,
                         dtype=dtype,
                         **kwargs)

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

    def copy(self, share_tensors: bool = False) -> 'ConvPEPS':
        """
        Creates a copy of the :class:`ConvPEPS`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether tensors in the copied ConvPEPS should be
            set as the tensors in the current ConvPEPS (``True``), or cloned
            (``False``). In the former case, tensors in both ConvPEPS's will be
            the same, which might be useful if one needs more than one copy
            of a ConvPEPS, but wants to compute all the gradients with respect
            to the same, unique, tensors.

        Returns
        -------
        ConvPEPS
        """
        new_peps = ConvPEPS(in_channels=self._in_channels,
                            bond_dim=self._bond_dim,
                            kernel_size=self._kernel_size,
                            stride=self._stride,
                            padding=self._padding,
                            dilation=self._dilation,
                            boundary=self._boundary,
                            tensors=None,
                            init_method=None,
                            device=None,
                            dtype=None)
        new_peps.name = self.name + '_copy'

        for i in range(self._n_rows):
            for j in range(self._n_cols):
                new_peps._grid_env[i][j] = new_peps._grid_env[i][j].parameterize(
                    set_param=isinstance(self._grid_env[i][j], ParamNode))

        if share_tensors:
            for new_row, row in zip(new_peps._grid_env, self._grid_env):
                for new_node, node in zip(new_row, row):
                    new_node.tensor = node.tensor
            for new_node, node in zip(new_peps._up_border, self._up_border):
                new_node.tensor = node.tensor
            for new_node, node in zip(new_peps._down_border, self._down_border):
                new_node.tensor = node.tensor
            for new_node, node in zip(new_peps._left_border, self._left_border):
                new_node.tensor = node.tensor
            for new_node, node in zip(new_peps._right_border, self._right_border):
                new_node.tensor = node.tensor
        else:
            for new_row, row in zip(new_peps._grid_env, self._grid_env):
                for new_node, node in zip(new_row, row):
                    new_node.tensor = node.tensor.clone()
            for new_node, node in zip(new_peps._up_border, self._up_border):
                new_node.tensor = node.tensor.clone()
            for new_node, node in zip(new_peps._down_border, self._down_border):
                new_node.tensor = node.tensor.clone()
            for new_node, node in zip(new_peps._left_border, self._left_border):
                new_node.tensor = node.tensor.clone()
            for new_node, node in zip(new_peps._right_border, self._right_border):
                new_node.tensor = node.tensor.clone()

        return new_peps

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


class ConvUPEPS(UPEPS):  # MARK: ConvUPEPS
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
    tensor : torch.Tensor, optional
        Tensor to set in the ConvUPEPS ``uniform_memory`` node.
    init_method : {"zeros", "ones", "copy", "rand", "randn"}, optional
        Initialization method.
    parameterized : bool, optional
        Boolean indicating whether ConvUPEPS nodes should be created as
        :class:`ParamNode` (``True``) or as :class:`Node` (``False``).
    device : torch.device, optional
        Device where to initialize the tensor if ``init_method`` is provided.
    dtype : torch.dtype, optional
        Dtype of the tensor if ``init_method`` is provided.
    kwargs : float
        Keyword arguments for the different initialization methods. See
        :meth:`~tensorkrowch.AbstractNode.make_tensor`.
        
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
                 tensor: Optional[torch.Tensor] = None,
                 init_method: Optional[Text] = 'randn',
                 parameterized: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None,
                 **kwargs: float) -> None:

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
                         tensor=tensor,
                         n_batches=2,
                         init_method=init_method,
                         parameterized=parameterized,
                         device=device,
                         dtype=dtype,
                         **kwargs)

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

    def copy(self, share_tensors: bool = False) -> 'ConvUPEPS':
        """
        Creates a copy of the :class:`ConvUPEPS`.

        Parameters
        ----------
        share_tensor : bool, optional
            Boolean indicating whether the common tensor in the copied ConvUPEPS
            should be set as the tensor in the current ConvUPEPS (``True``), or
            cloned (``False``). In the former case, tensors in both ConvUPEPS's will be
            the same, which might be useful if one needs more than one copy
            of a ConvUPEPS, but wants to compute all the gradients with respect
            to the same, unique, tensors.

        Returns
        -------
        ConvUPEPS
        """
        new_peps = ConvUPEPS(in_channels=self._in_channels,
                             bond_dim=self._bond_dim,
                             kernel_size=self._kernel_size,
                             stride=self._stride,
                             padding=self._padding,
                             dilation=self._dilation,
                             tensor=None,
                             init_method=None,
                             parameterized=isinstance(self.uniform_memory,
                                                      ParamNode),
                             device=None,
                             dtype=None)
        new_peps.name = self.name + '_copy'

        if share_tensors:
            new_peps.uniform_memory.tensor = self.uniform_memory.tensor
        else:
            new_peps.uniform_memory.tensor = self.uniform_memory.tensor.clone()

        return new_peps

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
