"""
This script contains:

    * extend_with_output
    * sketching
    * trimming
    * create_projector
    * val_error
    * tt_rss
"""

import time
import warnings
from typing import Optional, Union, Callable, Tuple, List, Sequence, Text
from math import sqrt

import torch
from torch.utils.data import TensorDataset, DataLoader

from tensorkrowch.embeddings import basis
from tensorkrowch.utils import random_unitary
import tensorkrowch.models as models


def create_projector_cols(Sc_kc_1, Sc_kc):
    merged_inv = torch.stack([Sc_kc_1[1], Sc_kc[1]], dim=1)
    merged_inv = merged_inv.unique(sorted=True, dim=0)

    s_k_0 = merged_inv[:, 0]
    s_k_1 = Sc_kc[0][:, :, -1:]
    s_k = (s_k_0, s_k_1)
    return s_k  


def peps_rss(function: Callable,
             embedding: Callable,
             sketch_samples: torch.Tensor,
             domain: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
             domain_multiplier: int = 1,
             rank: Optional[int] = None,
             cum_percentage: Optional[float] = None,
             batch_size: int = 64,
             device: Optional[torch.device] = None,
             verbose: bool = True,
             return_info: bool = False) -> Union[List[torch.Tensor],
                                                 Tuple[List[torch.Tensor], dict]]:
    r"""
    PEPS via Recursive Sketching from Samples
    
    Parameters
    ----------
    function : Callable
        Function that is going to be decomposed. It needs to have a single
        input argument, the data, which is a tensor of shape
        ``batch_size x n_rows x n_cols`` or ``batch_size x n_rows x n_cols x in_dim``.
        It must return a tensor of shape ``batch_size``.
    embedding : Callable
        Embedding function that maps the data tensor to a higher dimensional
        space. It needs to have a single argument. It is a function that
        transforms the given data tensor of shape ``batch_size x n_rows x n_cols``
        or ``batch_size x n_rows x n_cols x in_dim`` and returns an embedded
        tensor of shape ``batch_size x n_rows x n_cols x embed_dim``.
    sketch_samples : torch.Tensor
        Samples that will be used as sketches to decompose the function. It has
        to be a tensor of shape ``batch_size x n_rows x n_cols`` or
        ``batch_size x n_rows x n_cols x in_dim``. The ``batch_size`` should be
        a multiple of ``rank``.
    domain : torch.Tensor or list[torch.Tensor], optional
        Domain of the input variables. It should be given as a finite set of
        possible values that can take each variable. If all variables live in
        the same domain, it should be given as a tensor with shape ``n_values``
        or ``n_values x in_dim``, where the possible ``n_values`` should be at
        least as large as the desired input dimension of the PEPS cores, which
        is the ``embed_dim`` of the ``embedding``. The more values are given,
        the more accurate will be the tensorization but more costly will be to
        do it. If ``domain`` is given as a list, it should have the same
        number of elements as input variables, so that each variable can live
        in a different domain. If ``domain`` is not given, it will be obtained
        from the values each variable takes in the ``sketch_samples``.
    domain_multiplier : int
        Upper bound for how many values are used for the input variable domain
        if ``domain`` is not provided. If ``domain`` is not provided, the
        domain of the input variables will be inferred from the unique values
        each variable takes in the ``sketch_samples``. In this case, only
        ``domain_multiplier * embed_dim`` values will be taken randomly.
    rank : int, optional
        Upper bound for the bond dimension of all cores.
    cum_percentage : float, optional
        When getting the proper bond dimension of each core via truncated SVD,
        this is the proportion that should be satisfied between the sum of all
        singular values kept and the total sum of all singular values. Therefore,
        it specifies the rank of each core independently, allowing for
        varying bond dimensions.
    batch_size : int
        Batch size used to process ``sketch_samples`` with ``DataLoaders``
        during the decomposition.
    device : torch.device, optional
        Device to which ``sketch_samples`` will be sent to compute sketches. It
        should coincide with the device the ``function`` is in, in the case the
        function is a call to a ``nn.Module`` or uses tensors that are in a 
        specific device. This also applies to the ``embedding`` function.
    verbose : bool
        Default is ``True``.
    return_info : bool
        Boolean indicating if an additional dictionary with total time and
        validation error should be returned.

    Returns
    -------
    list[torch.Tensor]
        List of tensor cores of the MPS.
    dictionary
        If ``return_info`` is ``True``.
    """
    if not isinstance(function, Callable):
        raise TypeError('`function` should be callable')
    
    if not isinstance(embedding, Callable):
        raise TypeError('`embedding` should be callable')
    
    # Number of input features
    if not isinstance(sketch_samples, torch.Tensor):
        raise TypeError('`sketch_samples` should be torch.Tensor type')
    if len(sketch_samples.shape) not in [3, 4]:
        # batch_size x n_rows x n_cols or batch_size x n_rows x n_cols x in_dim
        raise ValueError(
            '`sketch_samples` should be a tensor with shape (batch_size, '
            'n_rows, n_cols) or (batch_size, n_rows, n_cols, in_dim)')
    n_rows = sketch_samples.size(1)
    n_cols = sketch_samples.size(2)
    if n_rows * n_cols == 0:
        raise ValueError('`sketch_samples` cannot be 0 dimensional')
    
    # Embedding dimension
    try:
        aux_embed = embedding(sketch_samples[:1, :1, :1].to(device))
    except:
        raise ValueError(
            '`embedding` should take as argument a single tensor with shape '
            '(batch_size, n_rows, n_cols) or (batch_size, n_rows, n_cols, in_dim)')
        
    if len(aux_embed.shape) != 4:
        raise ValueError('`embedding` should return a tensor of shape '
                         '(batch_size, n_rows, n_cols, embed_dim)')
    embed_dim = aux_embed.size(3)
    if embed_dim == 0:
        raise ValueError('Embedding dimension cannot be 0')
    
    # Input domain
    if domain is not None:
        if not isinstance(domain, torch.Tensor):
            if not isinstance(domain, Sequence):
                raise TypeError(
                    '`domain` should be torch.Tensor or list[list[torch.Tensor]] type')
            elif len(domain) != n_rows:
                raise ValueError(
                    'If `domain` is given as a sequence of sequences of '
                    'tensors, it should have as many sequences as number of rows')
            else:
                for row_domain in domain:
                    if not isinstance(row_domain, Sequence):
                        raise TypeError(
                            '`domain` should be torch.Tensor or '
                            'list[list[torch.Tensor]] type')
                    elif len(row_domain) != n_cols:
                        raise ValueError(
                        'If `domain` is given as a sequence of sequences of '
                        'tensors, each sequence should have as many elements as'
                        ' number of columns')
                    else:
                        for t in row_domain:
                            if not isinstance(t, torch.Tensor):
                                raise TypeError(
                                    '`domain` should be torch.Tensor or '
                                    'list[list[torch.Tensor]] type')
        else:
            if len(domain.shape) != (len(sketch_samples.shape) - 2):
                raise ValueError(
                    'If `domain` is given as a torch.Tensor, it should have '
                    'shape (n_values,) or (n_values, in_dim), and it should '
                    'only include `in_dim` if it also appears in the shape of '
                    '`sketch_samples`')
            if len(domain.shape) == 2:
                if domain.shape[1] == 1:
                    raise ValueError()
    
    # Rank
    if rank is not None:
        if not isinstance(rank, int):
            raise TypeError('`rank` should be int type')
        if rank < 1:
            raise ValueError('`rank` should be greater or equal than 1')
    
    # Cum. percentage
    if cum_percentage is not None:
        if not isinstance(cum_percentage, float):
            raise TypeError('`cum_percentage` should be float type')
        if (cum_percentage <= 0) or (cum_percentage > 1):
            raise ValueError('`cum_percentage` should be in the range (0, 1]')
    
    if (rank is None) and (cum_percentage is None):
        raise ValueError(
            'At least one of `rank` and `cum_percentage` should be given')
    
    # Batch size
    if not isinstance(batch_size, int):
        raise TypeError('`batch_size` should be int type')
        
    def aux_embedding(data):
        """
        For the cases where ``n_rows * n_cols = 1``, it returns an embedded
        tensor with shape ``batch_size x embed_dim``.
        """
        return embedding(data).squeeze(1)
    
    # Create projectors by columns
    # These are shared for all rows
    Sc = []
    sc = []
    Tc = []
    for kc in range(n_cols):
        if kc == 0:
            Sc_aux, Sc_inv = \
                sketch_samples[:, :, :(kc + 1)].unique(sorted=True,
                                                       return_inverse=True,
                                                       dim=0)
            Sc.append((Sc_aux, Sc_inv))
            sc.append(Sc[-1][0])
            Tc.append(None)
        elif kc == (n_cols - 1):
            Sc.append(None)
            sc.append(None)
            Tc.append(sketch_samples[:, :, kc:].unique(sorted=True, dim=0))
        else:
            Sc_aux, Sc_inv = \
                sketch_samples[:, :, :(kc + 1)].unique(sorted=True,
                                                       return_inverse=True,
                                                       dim=0)
            Sc.append((Sc_aux, Sc_inv))
            sc.append(create_projector_cols(Sc[-2], Sc[-1]))
            Tc.append(sketch_samples[:, :, kc:].unique(sorted=True,
                                                        dim=0))
    
    for kr in range(n_rows):
        
        for kc in range(n_cols):
            
            
    
    pass
