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
                 d_phys: Union[int, Sequence[int]],
                 d_bond: Union[int, Sequence[int]],
                 boundary: Text = 'obc',
                 param_bond: bool = False,
                 num_batches: int = 1,
                 inline_input: bool = False,
                 inline_mats: bool = False) -> None:
        """
        Create an MPS module.

        Parameters
        ----------
        n_sites: number of sites, including the input and output_node sites
        d_phys: physic dimension
        d_bond: bond dimension. If given as a sequence, the i-th bond
            dimension is always the dimension of the right edge of th i-th node
        boundary: string indicating whether we are using periodic or open
            boundary conditions
        param_bond: boolean indicating whether bond edges should be parametric
        num_batches: number of batch edges of input data
        inline_input: boolean indicating whether input should be contracted
            inline or in a single stacked contraction
        inline_mats: boolean indicating whether sequence of matrices
            should be contracted inline or as a sequence of pairwise stacked
            contrations
        """

        super().__init__(name='mps')

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

        # d_phys
        if isinstance(d_phys, (list, tuple)):
            if len(d_phys) != n_sites:
                raise ValueError('If `d_phys` is given as a sequence of int, '
                                 'its length should be equal to `n_sites`')
            self._d_phys = list(d_phys)
        elif isinstance(d_phys, int):
            self._d_phys = [d_phys] * n_sites
        else:
            raise TypeError('`d_phys` should be `int` type or a list/tuple of ints')

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
        else:
            raise TypeError('`d_bond` should be `int` type or a list/tuple of ints')

        # param_bond
        self._param_bond = param_bond

        self._make_nodes()
        self.param_bond(set_param=param_bond)
        self.initialize()
        
        self._num_batches = num_batches
        
        # Contraction algorithm
        self.inline_input = inline_input
        self.inline_mats = inline_mats