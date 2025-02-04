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
from math import sqrt, ceil
import itertools
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tensorkrowch.embeddings import basis
from tensorkrowch.utils import random_unitary
import tensorkrowch.models as models


def expand_tensors(tensors_list):
    # Expand all tensors so that each one has shape d1 x ... x dm x ri x ci
    aux_list = []
    for i in range(len(tensors_list)):
        view_shape = []
        expand_shape = []
        for j in range(len(tensors_list)):
            if j == i:
                view_shape.append(tensors_list[i].size(0))
                expand_shape.append(-1)
            else:
                view_shape.append(1)
                expand_shape.append(tensors_list[j].size(0))
        view_shape.append(tensors_list[i].size(1))
        view_shape.append(tensors_list[i].size(2))
        expand_shape.append(-1)
        expand_shape.append(-1)
        
        if len(tensors_list[i].shape) == 4:
            # If shape is di x ri x ci x in_dim, add in_dim to all tensors
            view_shape.append(tensors_list[i].size(3))
            expand_shape.append(-1)
        else:
            # If shape is di x ri x ci, add extra aux dimension of 1
            view_shape.append(1)
            expand_shape.append(-1)
        
        aux_tensor = tensors_list[i].view(*view_shape).expand(*expand_shape)
        aux_list.append(aux_tensor)
    
    return aux_list


def sketching(function, tensors_cols, tensors_rows, kc, kr, batch_size, device):
    n_tensors = len(tensors_cols) + len(tensors_rows)
    aux_batch = int(batch_size ** (1/n_tensors))
    rows_pos = int(kc > 0)
    
    final_shape = [t.size(0) for t in tensors_cols[:rows_pos]] + \
        [t.size(0) for t in tensors_rows] + \
        [t.size(0) for t in tensors_cols[rows_pos:]]
    
    # batches_cols = [
    #     DataLoader(
    #         TensorDataset(t),
    #         batch_size=aux_batch,
    #         shuffle=False,
    #         num_workers=0
    #         )
    #     for t in tensors_cols
    #     ]
    
    # batches_rows = [
    #     DataLoader(
    #         TensorDataset(t),
    #         batch_size=aux_batch,
    #         shuffle=False,
    #         num_workers=0
    #         )
    #     for t in tensors_rows
    #     ]
    
    # Phi_tilde_kr_kc = []
    # for cols_comb in itertools.product(*batches_cols):
    #     cols_comb = [t[0] for t in cols_comb]
    #     # cols_comb -> [] each is batch x rows(all) x cols (x in_dim)
    #     for rows_comb in itertools.product(*batches_rows):
    #         # rows_comb -> [] each is batch x rows x cols(1) (x in_dim)
    #         rows_comb = [t[0] for t in rows_comb]
    #         aux_rows_comb = expand_tensors(rows_comb)
    #         rows_tensor = torch.cat(aux_rows_comb, dim=len(rows_comb))
            
    #         aux_rows_tensor = rows_tensor.view(-1,
    #                                            rows_tensor.size(-3),
    #                                            rows_tensor.size(-2),
    #                                            rows_tensor.size(-1))
            
    #         aux_cols_comb = cols_comb[:rows_pos] + \
    #                         [aux_rows_tensor] + \
    #                         cols_comb[rows_pos:]
            
    #         aux_cols_comb = expand_tensors(aux_cols_comb)
    #         batch_tensor = torch.cat(aux_cols_comb, dim=len(aux_cols_comb) + 1)
    #         # new_shape = list(batch_tensor.shape[:-3])
            
    #         if batch_tensor.shape[-1] == 1:
    #             batch_tensor = batch_tensor.view(-1,
    #                                              batch_tensor.size(-3),
    #                                              batch_tensor.size(-2))
    #         else:
    #             batch_tensor = batch_tensor.view(-1,
    #                                              batch_tensor.size(-3),
    #                                              batch_tensor.size(-2),
    #                                              batch_tensor.size(-1))
            
    #         aux_result = function(batch_tensor.to(device)).cpu()
    #         Phi_tilde_kr_kc.append(aux_result)
            
    # Phi_tilde_kr_kc = torch.cat(Phi_tilde_kr_kc, dim=0)
    
    
    all_tensors = [t for t in tensors_cols[:rows_pos]] + \
        [t for t in tensors_rows] + \
        [t for t in tensors_cols[rows_pos:]]
    
    all_tensors = [(t, torch.arange(t.size(0)).view(-1, 1, 1))
                   for t in all_tensors]
    
    all_batches = [
        DataLoader(
            TensorDataset(*t),
            batch_size=aux_batch,
            shuffle=False,
            num_workers=0
            )
        for t in all_tensors
        ]
    
    Phi_tilde_kr_kc = []
    Phi_labels = []
    for all_comb in itertools.product(*all_batches):
        all_labels = [t[1] for t in all_comb]
        # print(all_labels)
        
        all_comb = [t[0] for t in all_comb]
        
        rows_comb = all_comb[rows_pos:(rows_pos + len(tensors_rows))]
        rows_labels_comb = all_labels[rows_pos:(rows_pos + len(tensors_rows))]
        
        aux_rows_comb = expand_tensors(rows_comb)
        rows_tensor = torch.cat(aux_rows_comb, dim=len(rows_comb))
        
        aux_rows_labels_comb = expand_tensors(rows_labels_comb)
        rows_labels_tensor = torch.cat(aux_rows_labels_comb,
                                       dim=-2)
        
        aux_rows_tensor = rows_tensor.view(-1,
                                        rows_tensor.size(-3),
                                        rows_tensor.size(-2),
                                        rows_tensor.size(-1))
        
        aux_cols_comb = all_comb[:rows_pos] + \
                        [aux_rows_tensor] + \
                        all_comb[(rows_pos + len(tensors_rows)):]
        
        
        aux_rows_labels_tensor = rows_labels_tensor.view(-1,
                                        rows_labels_tensor.size(-3),
                                        rows_labels_tensor.size(-2),
                                        rows_labels_tensor.size(-1))
        
        aux_cols_labels_comb = all_labels[:rows_pos] + \
                        [aux_rows_labels_tensor] + \
                        all_labels[(rows_pos + len(tensors_rows)):]
        
        
        aux_cols_comb = expand_tensors(aux_cols_comb)
        batch_tensor = torch.cat(aux_cols_comb, dim=len(aux_cols_comb) + 1)
        # new_shape = list(batch_tensor.shape[:-3])
        
        aux_cols_labels_comb = expand_tensors(aux_cols_labels_comb)
        batch_labels_tensor = torch.cat(aux_cols_labels_comb,
                                        dim=-2)
        
        if batch_tensor.shape[-1] == 1:
            batch_tensor = batch_tensor.view(-1,
                                            batch_tensor.size(-3),
                                            batch_tensor.size(-2))
        else:
            batch_tensor = batch_tensor.view(-1,
                                            batch_tensor.size(-3),
                                            batch_tensor.size(-2),
                                            batch_tensor.size(-1))
        
        aux_result = function(batch_tensor.to(device)).cpu()
        aux_result = aux_result.view(*[t.size(0) for t in all_comb])
        Phi_tilde_kr_kc.append(aux_result)
        
        batch_labels_tensor = batch_labels_tensor.view(
            *[t.size(0) for t in all_comb], len(all_comb))
        Phi_labels.append(batch_labels_tensor)
        
    
    
    aux2_Phi_tilde_kr_kc = torch.cat([t.view(-1) for t in Phi_tilde_kr_kc],
                                     dim=0)
    aux2_Phi_tilde_kr_kc = aux2_Phi_tilde_kr_kc.reshape(*final_shape)
    
    
    
    all_n_batches = [ceil(t[0].size(0) / aux_batch) for t in all_tensors]
    
    j = 0
    while all_n_batches:
        group_size = all_n_batches.pop(-1)
        j += 1
        
        aux_Phi = []
        aux_Phi_labels = []
        for i in range(0, len(Phi_tilde_kr_kc), group_size):
            aux_Phi.append(
                torch.cat(Phi_tilde_kr_kc[i:(i + group_size)],
                          dim=len(all_tensors) - j))
            
            aux_Phi_labels.append(
                torch.cat(Phi_labels[i:(i + group_size)],
                          dim=len(all_tensors) - j))
            
        Phi_tilde_kr_kc = aux_Phi
        Phi_labels = aux_Phi_labels
    Phi_tilde_kr_kc = Phi_tilde_kr_kc[0]
    Phi_labels = Phi_labels[0]
    
    # assert torch.equal(aux2_Phi_tilde_kr_kc, Phi_tilde_kr_kc)
    
    
    # Phi_tilde_kr_kc = torch.cat(Phi_tilde_kr_kc, dim=0)
    
    
    
    # batches_cols = [
    #     DataLoader(
    #         TensorDataset(t),
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=0
    #         )
    #     for t in tensors_cols
    #     ]
    
    # batches_rows = [
    #     DataLoader(
    #         TensorDataset(t),
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=0
    #         )
    #     for t in tensors_rows
    #     ]
    
    # aux_Phi_tilde_kr_kc = []
    # for cols_comb in itertools.product(*batches_cols):
    #     cols_comb = [t[0] for t in cols_comb]
    #     # cols_comb -> [] each is batch x rows(all) x cols (x in_dim)
    #     for rows_comb in itertools.product(*batches_rows):
    #         # rows_comb -> [] each is batch x rows x cols(1) (x in_dim)
    #         rows_comb = [t[0] for t in rows_comb]
    #         aux_rows_comb = expand_tensors(rows_comb)
    #         rows_tensor = torch.cat(aux_rows_comb, dim=len(rows_comb))
            
    #         aux_rows_tensor = rows_tensor.view(-1,
    #                                            rows_tensor.size(-3),
    #                                            rows_tensor.size(-2),
    #                                            rows_tensor.size(-1))
            
    #         aux_cols_comb = cols_comb[:rows_pos] + \
    #                         [aux_rows_tensor] + \
    #                         cols_comb[rows_pos:]
            
    #         aux_cols_comb = expand_tensors(aux_cols_comb)
    #         batch_tensor = torch.cat(aux_cols_comb, dim=len(aux_cols_comb) + 1)
    #         # new_shape = list(batch_tensor.shape[:-3])
            
    #         if batch_tensor.shape[-1] == 1:
    #             batch_tensor = batch_tensor.view(-1,
    #                                              batch_tensor.size(-3),
    #                                              batch_tensor.size(-2))
    #         else:
    #             batch_tensor = batch_tensor.view(-1,
    #                                              batch_tensor.size(-3),
    #                                              batch_tensor.size(-2),
    #                                              batch_tensor.size(-1))
            
    #         aux_result = function(batch_tensor.to(device)).cpu()
    #         aux_Phi_tilde_kr_kc.append(aux_result)
    
    
    
    all_labels = [t[1] for t in all_tensors]
    all_comb = [t[0] for t in all_tensors]
        
    # all_comb = all_tensors
    rows_comb = all_comb[rows_pos:(rows_pos + len(tensors_rows))]
    rows_labels_comb = all_labels[rows_pos:(rows_pos + len(tensors_rows))]
    
    aux_rows_comb = expand_tensors(rows_comb)
    rows_tensor = torch.cat(aux_rows_comb, dim=len(rows_comb))
    
    aux_rows_labels_comb = expand_tensors(rows_labels_comb)
    rows_labels_tensor = torch.cat(aux_rows_labels_comb, dim=-2)
    
    aux_rows_tensor = rows_tensor.view(-1,
                                        rows_tensor.size(-3),
                                        rows_tensor.size(-2),
                                        rows_tensor.size(-1))
    
    aux_rows_labels_tensor = rows_labels_tensor.view(-1,
                                        rows_labels_tensor.size(-3),
                                        rows_labels_tensor.size(-2),
                                        rows_labels_tensor.size(-1))
    
    aux_cols_comb = all_comb[:rows_pos] + \
                    [aux_rows_tensor] + \
                    all_comb[(rows_pos + len(tensors_rows)):]
                    
    aux_cols_labels_comb = all_labels[:rows_pos] + \
                    [aux_rows_labels_tensor] + \
                    all_labels[(rows_pos + len(tensors_rows)):]
    
    aux_cols_comb = expand_tensors(aux_cols_comb)
    batch_tensor = torch.cat(aux_cols_comb, dim=len(aux_cols_comb) + 1)
    # new_shape = list(batch_tensor.shape[:-3])
    
    aux_cols_labels_comb = expand_tensors(aux_cols_labels_comb)
    batch_labels_tensor = torch.cat(aux_cols_labels_comb, dim=-2)
    batch_labels_tensor = batch_labels_tensor.view(
            *[t.size(0) for t in all_comb], len(all_comb))
    
    if batch_tensor.shape[-1] == 1:
        batch_tensor = batch_tensor.view(-1,
                                            batch_tensor.size(-3),
                                            batch_tensor.size(-2))
    else:
        batch_tensor = batch_tensor.view(-1,
                                            batch_tensor.size(-3),
                                            batch_tensor.size(-2),
                                            batch_tensor.size(-1))
    
    aux_Phi_tilde_kr_kc = function(batch_tensor.to(device)).cpu()
    aux_Phi_tilde_kr_kc = aux_Phi_tilde_kr_kc.reshape(*final_shape)
    
    
    assert torch.equal(batch_labels_tensor, Phi_labels)
    
    # aux_Phi_tilde_kr_kc = torch.cat(aux_Phi_tilde_kr_kc, dim=0)
    
    # assert torch.equal(aux_Phi_tilde_kr_kc, Phi_tilde_kr_kc)
    
    
    
    # new_shape = new_shape[:rows_pos] + \
    #             list(rows_tensor.shape[:-3]) + \
    #             new_shape[rows_pos:]
    
    # Phi_tilde_kr_kc = Phi_tilde_kr_kc.view(*final_shape)
    # aux_Phi_tilde_kr_kc = aux_Phi_tilde_kr_kc.reshape(*final_shape)
    
    # assert torch.equal(aux_Phi_tilde_kr_kc, Phi_tilde_kr_kc)
    
    Phi_tilde_kr_kc = aux_Phi_tilde_kr_kc
    
    # Current shape: (left, up, input, down, right) -> 0, 1, 2, 3, 4
    # Permute to shape: (input, up, right, down, left) -> 2, 1, 4, 3, 0
    # Check how many indices are not in tensor
    absent_ids = []
    if len(tensors_cols) == 1:
        if kc == 0:
            absent_ids.append(0)
        else:
            absent_ids.append(4)
    if len(tensors_rows) == 2:
        if kr == 0:
            absent_ids.append(1)
        else:
            absent_ids.append(3)
    absent_ids.sort()
    
    perm_ids = [2, 1, 4, 3, 0]
    if absent_ids:
        aux_perm_ids = []
        for i in range(5):
            aux_id = perm_ids[i]
            append_id = True
            for abs_id in absent_ids:
                if abs_id < perm_ids[i]:
                    aux_id -= 1
                elif abs_id == perm_ids[i]:
                    append_id = False
            if append_id:
                aux_perm_ids.append(aux_id)
        perm_ids = aux_perm_ids
    
    Phi_tilde_kr_kc = Phi_tilde_kr_kc.permute(*perm_ids)
    
    return Phi_tilde_kr_kc


def trimming(tensor, dim, rank, cum_percentage):
    """
    Given a tensor, forms a matrix with edges split as (rest, dim), and
    returns the U from the SVD and an appropiate rank.
    """
    n_dims = len(tensor.shape)
    perm_ids = list(range(dim)) + list(range(dim + 1, n_dims)) + [dim]
    inv_perm_ids = list(range(dim)) + [n_dims - 1] + list(range(dim, n_dims - 1))
    
    aux_tensor = tensor.permute(perm_ids)
    mat = aux_tensor.reshape(-1, aux_tensor.size(-1))
    
    u, s, vh = torch.linalg.svd(mat, full_matrices=False)
    
    # if rank is None:
    #     rank = len(s)

    percentages = s.cumsum(0) / (s.sum().expand(s.shape) + 1e-10)
    cum_percentage_tensor = torch.tensor(cum_percentage)
    
    aux_rank = 0
    for p in percentages:
        if p == 0:
            if aux_rank == 0:
                aux_rank = 1
            break
        aux_rank += 1
        
        # Cut when ``cum_percentage`` is exceeded
        if p >= cum_percentage_tensor:
            break
        elif aux_rank >= rank:
            break
    
    u = u.view(*aux_tensor.shape[:(n_dims - 1)], -1)
    # u = u[..., :aux_rank]
    u = u[..., :aux_rank] @ torch.diag_embed(s[:aux_rank])  # TODO: check if this causes problems
    u = u.permute(inv_perm_ids)
    
    vh = vh[..., :aux_rank, :]
    
    # TODO: deal with error
    error = s[aux_rank:].norm()
        
    return u, vh  #u, s, vh


def solving(A, B):
    # Split B ids as: (left, (input, up, right, down))
    # A has shape: (left, aux_left)
    # We find G with shape: (aux_left, (input, up, right, down))
    # Return G to shape: (input, up, right, down, aux_left)
    n_dims = len(B.shape)
    perm_ids = [n_dims - 1] + list(range(n_dims - 1))
    inv_perm_ids = list(range(1, n_dims)) + [0]
    
    aux_B = B.permute(perm_ids)
    mat_B = aux_B.reshape(aux_B.size(0), -1)
    
    G = torch.linalg.lstsq(A, mat_B).solution
    
    G = G.view(A.size(1), *aux_B.shape[1:])
    G = G.permute(inv_perm_ids)
    
    return G


def match_multiples(prev_size, rank, size):
    new_prev_size, new_size = prev_size, size
    while new_prev_size * rank != new_size:
        if new_prev_size * rank < new_size:
            raise ValueError('Problem with dimensions')
            new_prev_size += 1
        else:
            new_size += 1
    
    return new_prev_size, new_size


def solving_mpo(prev_mpo_core, mpo_core, rank, kr, kc, n_rows, n_cols):
    # mpo_core shape: (input, up, right, down, left)
    #              -> (up, (input, right, down, left))
    #              -> ((up, right_prev, left_prev), (input, right_rank, down, left_rank))
    # prev_mpo_core shape: (up, right_prev, aux_up, left_prev)
    #                   -> ((up, right_prev, left_prev), aux_up)
    
    # TODO: bad
    # prev_mpo_core shape: (aux_up, up, right_prev, left_prev)
    #                   -> ((up, right_prev, left_prev), aux_up)
    
    # We find G with shape: (aux_up, (input, right, down, left))
    # Return G to shape: (input, aux_up, right, down, left)
    n_dims = len(mpo_core.shape)
    perm_ids = [1, 0] + list(range(2, n_dims))
    mpo_core = mpo_core.permute(perm_ids)
    
    prev_n_dims = len(prev_mpo_core.shape)
    aux_up_pos = prev_n_dims - 1 - int(kc > 0)
    prev_perm_ids = list(range(aux_up_pos)) + \
        list(range(aux_up_pos + 1, prev_n_dims)) + [aux_up_pos] 
    prev_mpo_core = prev_mpo_core.permute(prev_perm_ids)
    
    # TODO: check if padding might cause error and may be better not to use uniques
    # TODO: padding with 0's at the end of left and right dimensions shouldn't
    # be a problem because we already trimmed that legs via SVD, so it would be
    # like putting 0's in place of the smallest singular values that we cut
    
    # Check left/right dims match left_prev/right_prev * rank
    if kc > 0:
        _, new_left_size = match_multiples(
            prev_mpo_core.size(-2), rank, mpo_core.size(-1))
        
        mpo_core = nn.functional.pad(mpo_core,
                                     (0, new_left_size - mpo_core.size(-1)))
    
    if kc < (n_cols - 1):
        _, new_right_size = match_multiples(
            prev_mpo_core.size(int(kr > 0)), rank, mpo_core.size(2))
        
        reverse_right_pos = int(kc > 0) + int(kr < (n_rows - 1))
        pad = reverse_right_pos * [0, 0] + \
            [0, new_right_size - mpo_core.size(2)]
        mpo_core = nn.functional.pad(mpo_core, pad)
    
    # Split left/right dims in left_prev/right_prev and left_rank/right_rank
    #    (up, (input, right, down, left))
    # -> ((up, right_prev), (input, right_rank, down, left))
    if kc < (n_cols - 1):
        aux_shape = list(mpo_core.shape)
        aux_shape = aux_shape[:2] + [prev_mpo_core.size(int(kr > 0)), rank] + aux_shape[3:]
        mpo_core = mpo_core.reshape(aux_shape)
        
        aux_perm_ids = [0, 2, 1] + list(range(3, len(mpo_core.shape)))
        mpo_core = mpo_core.permute(aux_perm_ids)
        
    #    ((up, right_prev), (input, right_rank, down, left))
    # -> ((up, right_prev, left_prev), (input, right_rank, down, left_rank))
    if kc > 0:
        aux_shape = list(mpo_core.shape)
        aux_shape = aux_shape[:-1] + [prev_mpo_core.size(-2), rank]
        mpo_core = mpo_core.reshape(aux_shape)
        
        if kc < (n_cols - 1):
            aux_perm_ids = [0, 1, -2] + \
                list(range(2, len(mpo_core.shape) - 2)) + \
                [len(mpo_core.shape) - 1]
        else:
            aux_perm_ids = [0, -2] + \
                list(range(1, len(mpo_core.shape) - 2)) + \
                [len(mpo_core.shape) - 1]
        mpo_core = mpo_core.permute(aux_perm_ids)
    
    # Merge dimensions
    split_pos = 1 + int(kc > 0) + int(kc < (n_cols - 1))
    aux_shape = [reduce(mul, mpo_core.shape[:split_pos])] + \
        [reduce(mul, mpo_core.shape[split_pos:])]
    aux_mpo_core = mpo_core.reshape(aux_shape)
    # ((up, right_prev, left_prev), (input, right_rank, down, left_rank))
    
    aux_shape = [reduce(mul, prev_mpo_core.shape[:-1])] + \
        [prev_mpo_core.shape[-1]]
    aux_prev_mpo_core = prev_mpo_core.reshape(aux_shape)
    # ((up, right_prev, left_prev), aux_up)
    
    peps_core = torch.linalg.lstsq(aux_prev_mpo_core, aux_mpo_core).solution
    # (aux_up, (input, right_rank, down, left_rank))
    
    print('PEPS core error:',
          (aux_prev_mpo_core @ peps_core - aux_mpo_core).norm())
    
    peps_core = peps_core.reshape(peps_core.size(0), *mpo_core.shape[split_pos:])
    
    perm_ids = [1, 0] + list(range(2, len(peps_core.shape)))
    peps_core = peps_core.permute(perm_ids)
    
    return peps_core


def create_projector(batch_tensors, input_tensor):
    # TODO: check this returns batch in the same order as if we have
    # all the tensors together and apply unique
    merged_inv = torch.stack(
        [bt[1] for bt in batch_tensors] + [input_tensor[1]], dim=1)
    merged_inv = merged_inv.unique(sorted=True, dim=0)
    
    ordered_batch_tensors = [merged_inv[:, i] for i in range(len(batch_tensors))]
    ordered_input_tensor = input_tensor[0][merged_inv[:, -1]]
    
    return ordered_batch_tensors, ordered_input_tensor


# def create_projector_cols(Sc_kc_minus_1, Sc_kc):
#     merged_inv = torch.stack([Sc_kc_minus_1[1], Sc_kc[1]], dim=1)
#     merged_inv = merged_inv.unique(sorted=True, dim=0)

#     s_k_0 = merged_inv[:, 0]
#     s_k_1 = Sc_kc[0][:, :, -1:]
#     s_k = (s_k_0, s_k_1)
#     return s_k


def peps_rss(function: Callable,
             embedding: Callable,
             sketch_samples: torch.Tensor,
             domain: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]] = None,
             domain_multiplier: int = 1,
             rank: int = 10,
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
    rank : int
        Upper bound for the bond dimension of all cores. It has to be greater
        or equal than ``embed_dim``.
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
    if not isinstance(rank, int):
        raise TypeError('`rank` should be int type')
    # TODO: we removed this, needed?
    # if rank < embed_dim:
    #     raise ValueError(
    #         '`rank` should be greater or equal than the embedding dimension')
    
    # Sketch size multiple os rank
    sketch_size = sketch_samples.size(0)
    sksize_rank_factor = sketch_size / rank
    # if int(sksize_rank_factor) != sksize_rank_factor:
    #     sketch_size = int(sksize_rank_factor) * sketch_size
    #     sketch_samples = sketch_samples[:sketch_size]
    #     warnings.warn(
    #         '`sketch_size` is being modified to make it a multiple of `rank`')
    # if int(sksize_rank_factor) != sksize_rank_factor:
    #     raise ValueError(
    #         'The size of the sketches (number of samples) should be a '
    #         'multiple of the rank')
    # TODO: MMMMM AAAAH nope, there's still a problem, since when doing unique
    # we might change the sketch size
    # TODO: Then maybe just remove the uniques? and keep sketch size at all points?
    
    
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
        return embedding(data).squeeze(1, 2)
    
    # Create projectors by columns
    # These are shared for all rows
    Sc = []
    # sc = []
    Tc = []
    for kc in range(n_cols):
        if kc == 0:
            Sc_aux, Sc_inv = \
                sketch_samples[:, :, :(kc + 1)].unique(sorted=True,
                                                       return_inverse=True,
                                                       dim=0)
            Sc.append((Sc_aux, Sc_inv))
            # sc.append(Sc[-1][0])
            Tc.append(None)
        elif kc < (n_cols - 1):
            Sc_aux, Sc_inv = \
                sketch_samples[:, :, :(kc + 1)].unique(sorted=True,
                                                       return_inverse=True,
                                                       dim=0)
            Sc.append((Sc_aux, Sc_inv))
            # sc.append(create_projector_cols(Sc[-2], Sc[-1]))
            Tc.append(sketch_samples[:, :, kc:].unique(sorted=True,
                                                        dim=0))
        else:
            Sc.append(None)
            # sc.append(None)
            Tc.append(sketch_samples[:, :, kc:].unique(sorted=True, dim=0))
    
    start_time = time.time()
    einsum_ids = 'ABCDE'
    peps = []
    for kr in range(n_rows):
        peps_row = []
        # mpo_kr = []
        aux_mpo_kr_minus_1 = []
        # Sr_kr = []
        
        for kc in range(n_cols):
            
            if verbose:
                site_count = \
                    f'|| Site: ({kr + 1} / {n_rows}, {kc + 1} / {n_cols}) ||'
                site_count = ['=' * len(site_count), site_count]
                print('\n\n' + site_count[0] + '\n' + site_count[1] + \
                      '\n' + site_count[0])
            
            if verbose:
                print('\t* Preparing sketches...', end=' ')
            
            # Prepare x_kr_kc
            if domain is not None:
                if isinstance(domain, torch.Tensor):
                    x_kr_kc = domain.view(-1, 1, 1)
                else:
                    x_kr_kc = domain[kr][kc].view(-1, 1, 1)
            else:
                x_kr_kc = \
                    sketch_samples[:, kr:(kr + 1), kc:(kc + 1)].unique(sorted=True,
                                                                       dim=0)
                
                if x_kr_kc.size(0) >= (domain_multiplier * embed_dim):
                    perm = torch.randperm(x_kr_kc.size(0))
                    idx = perm[:(domain_multiplier * embed_dim)]
                    x_kr_kc = x_kr_kc[idx]
            
            aux_x_kr_kc = \
                sketch_samples[:, kr:(kr + 1), kc:(kc + 1)].unique(sorted=True,
                                                                   return_inverse=True,
                                                                   dim=0)
            
            # Prepare Sr_kr_minus_1_kc
            if kr > 0:
                Sr_kr_minus_1_kc = \
                    sketch_samples[:, :kr, kc:(kc + 1)].unique(sorted=True,
                                                               return_inverse=True,
                                                               dim=0)
                    
            # Prepare Tr_kr_plus_1_kc
            if kr < (n_rows - 1):
                Tr_kr_plus_1_kc = \
                    sketch_samples[:, (kr + 1):, kc:(kc + 1)].unique(sorted=True,
                                                                     return_inverse=True,
                                                                     dim=0)
            
            if verbose:
                torch.cuda.synchronize(device=device)
                aux_time = time.time() - start_time
                print(f'Done! ({aux_time:.2f}s)')
            
            # Collect projectors
            tensors_rows = [x_kr_kc]
            if kr < (n_rows - 1):
                tensors_rows.append(Tr_kr_plus_1_kc[0])
            if kr > 0:
                tensors_rows = [Sr_kr_minus_1_kc[0]] + tensors_rows
            
            tensors_cols = []
            if kc < (n_cols - 1):
                tensors_cols.append(Tc[kc + 1])
            if kc > 0:
                tensors_cols = [Sc[kc - 1][0]] + tensors_cols
            
            # Position right idx: (input, up, right, down, left)
            # right_pos = int(kr > 0) + int(kc < (n_cols - 1))
            right_pos = int(kr > 0) + 1
            
            # Position down idx: (input, up, right, down, left)
            # down_pos = n_dims_mpo_core - 1 - int(kc > 0)
            down_pos = int(kr > 0) + int(kc < (n_cols - 1)) + 1
            
            
            # Sketching
            if verbose:
                print('\t* Sketching...', end=' ')
            
            Phi_tilde_kr_kc = sketching(
                function=function,
                tensors_cols=tensors_cols,
                tensors_rows=tensors_rows,
                kc=kc,
                kr=kr,
                batch_size=batch_size,
                device=device)
            
            # aux_s = torch.linalg.svd(Phi_tilde_kr_kc.reshape(-1, 9),
            #                          full_matrices=False)[1]
            
            if verbose:
                torch.cuda.synchronize(device=device)
                aux_time = time.time() - start_time
                print(f'Done! ({aux_time:.2f}s)')
            
            
            # Random projection for Tc
            if kc < (n_cols - 1):
                # Sample random unitary
                randu_t = random_unitary(Phi_tilde_kr_kc.size(right_pos))
                randu_t = randu_t.to(Phi_tilde_kr_kc.dtype)
                
                # Prepare einsum string
                randu_ids = 'r' + einsum_ids[right_pos]
                aux_ein_ids = einsum_ids[:right_pos] + 'r' + \
                    einsum_ids[(right_pos + 1):len(Phi_tilde_kr_kc.shape)]
                einsum_str = aux_ein_ids + ',' + randu_ids + '->' + \
                    einsum_ids[:len(Phi_tilde_kr_kc.shape)]
                Phi_tilde_kr_kc = torch.einsum(einsum_str,
                                               Phi_tilde_kr_kc, randu_t)
            
            # Remove embedding from Phi
            aux_Phi_tilde_kr_kc = torch.linalg.lstsq(
                aux_embedding(x_kr_kc.to(device)).cpu(),
                Phi_tilde_kr_kc.reshape(x_kr_kc.size(0), -1)).solution
            
            new_shape = list(Phi_tilde_kr_kc.shape)
            new_shape[0] = embed_dim
            Phi_tilde_kr_kc = aux_Phi_tilde_kr_kc.reshape(*new_shape)
            
            
            # Trimming right
            if verbose:
                print('\t* Trimming right Phi_tilde_kr_kc...', end=' ')
            
            # right_rank = list(Phi_tilde_kr_kc.shape)
            # if kc > 0:
            #     right_rank[-1] = A_kr_kc_minus_1.shape[-1]
            # if kc < (n_cols - 1):
            #     right_rank[right_pos] = 1
            # if kr < (n_rows - 1):
            #     right_rank[down_pos] = rank
            # right_rank = reduce(mul, right_rank)
            
            if kr == 0:
                right_rank = sketch_size
            else:
                right_rank = mpo_kr_minus_1[kc].shape[right_pos - 1] * rank
            
            if kc < (n_cols - 1):
                B_kr_kc, _ = trimming(tensor=Phi_tilde_kr_kc,
                                      dim=right_pos,
                                      rank=right_rank,
                                      cum_percentage=cum_percentage)
            else:
                B_kr_kc = Phi_tilde_kr_kc
            
            if verbose:
                torch.cuda.synchronize(device=device)
                aux_time = time.time() - start_time
                print(f'Done! ({aux_time:.2f}s)')
            
            
            # Solving
            if verbose:
                print('\t* Solving for mpo_core...', end=' ')
            
            if kc == 0:
                G_kr_kc = B_kr_kc
            else:
                G_kr_kc = solving(A_kr_kc_minus_1, B_kr_kc)
            # mpo_kr.append(G_kr_kc)
            
            if verbose:
                torch.cuda.synchronize(device=device)
                aux_time = time.time() - start_time
                print(f'Done! ({aux_time:.2f}s)')
            
            
            # Create A_kr_kc
            if verbose:
                print('\t* Create A_kr_kr...', end=' ')
                
            if kc < (n_cols - 1):
                batch_tensors = []
                pos_ids = []
                if kr < (n_rows - 1):
                    batch_tensors.append(Tr_kr_plus_1_kc)
                    pos_ids = [2]
                if kr > 0:
                    batch_tensors = [Sr_kr_minus_1_kc] + batch_tensors
                    if kr < (n_rows - 1):
                        pos_ids = [1, 3]
                    else:
                        pos_ids = [1]
                if kc > 0:
                    batch_tensors.append(Sc[kc - 1])
                    pos_ids.append(len(B_kr_kc.shape) - 1)
                
                sc_kr_kc = create_projector(batch_tensors, aux_x_kr_kc)
                
                # Fix ordering
                # TODO: this is auxiliary solution (?)
                aux_batch_tensors = []  # up, down, left
                for i in range(len(pos_ids)):
                    aux_batch_tensors.append(
                        batch_tensors[i][0].index_select(dim=0,
                                                         index=sc_kr_kc[0][i]))
                
                aux_col = aux_batch_tensors[:(kr > 0)] + [sc_kr_kc[1]] + \
                    aux_batch_tensors[(kr > 0):]
                if kc > 0:
                    aux_col = aux_col[:-1]
                aux = torch.cat(aux_col, dim=1)
                
                if kc > 0:
                    aux = torch.cat([aux_batch_tensors[-1], aux], dim=2)
                
                aux, inv_ids = aux.unique(dim=0,
                                          sorted=True,
                                          return_inverse=True)
                
                for i in range(len(pos_ids)):
                    sc_kr_kc[0][i] = sc_kr_kc[0][i][inv_ids]
                sc_kr_kc = (sc_kr_kc[0], sc_kr_kc[1][inv_ids])
                
                
                # # TODO: check
                # aux_batch_tensors = []  # up, down, left
                # for i in range(len(pos_ids)):
                #     aux_batch_tensors.append(
                #         batch_tensors[i][0].index_select(dim=0,
                #                                          index=sc_kr_kc[0][i]))
                
                # aux_col = aux_batch_tensors[:(kr > 0)] + [sc_kr_kc[1]] + \
                #     aux_batch_tensors[(kr > 0):]
                # if kc > 0:
                #     aux_col = aux_col[:-1]
                # aux = torch.cat(aux_col, dim=1)
                
                # if kc > 0:
                #     aux = torch.cat([aux_batch_tensors[-1], aux], dim=2)
                
                # aux, inv_ids = aux.unique(dim=0,
                #                           sorted=True,
                #                           return_inverse=True)
                
                # assert torch.equal(aux, Sc[kc][0])
                
                
                A_kr_kc = B_kr_kc
                for i in range(len(pos_ids)):
                    A_kr_kc = A_kr_kc.index_select(dim=pos_ids[i],
                                                   index=sc_kr_kc[0][i])
                
                aux_sc = aux_embedding(sc_kr_kc[1].to(device)).cpu()
                
                einsum_str = 'i'
                i, j = 1, 0
                while i < len(B_kr_kc.shape):
                    if j < len(pos_ids):
                        if i < pos_ids[j]:
                            einsum_str += 'r'
                            i += 1
                        elif i == pos_ids[j]:
                            einsum_str += 'b'
                            i += 1
                            j += 1
                        else:
                            raise ValueError('Something strange')
                    else:
                        einsum_str += 'r'
                        i += 1
                einsum_str += ',bi->br'
                
                A_kr_kc = torch.einsum(einsum_str, A_kr_kc, aux_sc)
                
                # Set variables for next iteration
                A_kr_kc_minus_1 = A_kr_kc
            
            if verbose:
                torch.cuda.synchronize(device=device)
                aux_time = time.time() - start_time
                print(f'Done! ({aux_time:.2f}s)')
                        
                
                # if kc == 0:
                #     if kr == 0:
                #         sc_kr_kc = create_projector_cols([Tr_kr_plus_1_kc],
                #                                          aux_x_kr_kc)
                #         # input x right x down
                #         A_kr_kc = B_kr_kc[:, :, sc_kr_kc[0][0]]
                        
                #         aux_sc = aux_embedding(sc_kr_kc[1].to(device)).cpu()
                #         A_kr_kc = torch.einsum('irb,bi->br', A_kr_kc, aux_sc)
                #     elif kr < (n_rows - 1):
                #         sc_kr_kc = create_projector_cols([Sr_kr_minus_1[kc],
                #                                           Tr_kr_plus_1_kc],
                #                                          aux_x_kr_kc)
                #         # input x up x right x down
                #         A_kr_kc = B_kr_kc[:, sc_kr_kc[0][0], :, :]
                #         A_kr_kc = A_kr_kc[:, :, :, sc_kr_kc[0][1]]
                        
                #         aux_sc = aux_embedding(sc_kr_kc[1].to(device)).cpu()
                #         A_kr_kc = torch.einsum('ibrb,bi->br', A_kr_kc, aux_sc)
                #     else:
                #         sc_kr_kc = create_projector_cols([Sr_kr_minus_1[kc]],
                #                                          aux_x_kr_kc)
                #         # input x up x right
                #         A_kr_kc = B_kr_kc[:, sc_kr_kc[0][0], :]
                        
                #         aux_sc = aux_embedding(sc_kr_kc[1].to(device)).cpu()
                #         A_kr_kc = torch.einsum('ibr,bi->br', A_kr_kc, aux_sc)
                # else:
                #     if kr == 0:
                #         sc_kr_kc = create_projector_cols([Tr_kr_plus_1_kc,
                #                                           Sc[kc - 1]],
                #                                          aux_x_kr_kc)
                #         # input x right x down x left
                #         A_kr_kc = B_kr_kc[:, :, sc_kr_kc[0][0], :]
                #         A_kr_kc = A_kr_kc[:, :, :, sc_kr_kc[0][1]]
                        
                #         aux_sc = aux_embedding(sc_kr_kc[1].to(device)).cpu()
                #         A_kr_kc = torch.einsum('irbb,bi->dr', A_kr_kc, aux_sc)
                #     elif kr < (n_rows - 1):
                #         sc_kr_kc = create_projector_cols([Sr_kr_minus_1[kc],
                #                                           Tr_kr_plus_1_kc,
                #                                           Sc[kc - 1]],
                #                                          aux_x_kr_kc)
                #         # input x up x right x down x left
                #         A_kr_kc = B_kr_kc[:, sc_kr_kc[0][0], :, :, :]
                #         A_kr_kc = A_kr_kc[:, :, :, sc_kr_kc[0][1], :]
                #         A_kr_kc = A_kr_kc[:, :, :, :, sc_kr_kc[0][2]]
                        
                #         aux_sc = aux_embedding(sc_kr_kc[1].to(device)).cpu()
                #         A_kr_kc = torch.einsum('ibrbb,bi->br', A_kr_kc, aux_sc)
                #     else:
                #         sc_kr_kc = create_projector_cols([Sr_kr_minus_1[kc],
                #                                           Sc[kc - 1]],
                #                                          aux_x_kr_kc)
                #         # input x up x right x left
                #         A_kr_kc = B_kr_kc[:, sc_kr_kc[0][0], :, :]
                #         A_kr_kc = A_kr_kc[:, :, :, sc_kr_kc[0][1]]
                        
                #         aux_sc = aux_embedding(sc_kr_kc[1].to(device)).cpu()
                #         A_kr_kc = torch.einsum('ibrb,bi->br', A_kr_kc, aux_sc)
                
                
                
                # # Set variables for next iteration
                # A_kr_kc_minus_1 = A_kr_kc
            
            # # Create Sr_kr
            # if kr < (n_rows - 1):
            #     Sr_kr_aux, Sr_kr_inv = \
            #         sketch_samples[:, :(kr + 1), kc:(kc + 1)].unique(
            #             sorted=True,
            #             return_inverse=True,
            #             dim=0
            #         )
            #     Sr_kr.append((Sr_kr_aux, Sr_kr_inv))
            
            
            # Process current MPO
            mpo_core = G_kr_kc
            n_dims_mpo_core = len(mpo_core.shape)
            # down_pos = n_dims_mpo_core - 1 - int(kc > 0)
            
            # Random projection for Tr
            if kr < (n_rows - 1):
                # Sample random unitary
                randu_t = random_unitary(mpo_core.size(down_pos))
                randu_t = randu_t.to(mpo_core.dtype)
                
                # Prepare einsum string
                randu_ids = 'd' + einsum_ids[down_pos]
                aux_ein_ids = einsum_ids[:down_pos] + 'd' + \
                    einsum_ids[(down_pos + 1):n_dims_mpo_core]
                einsum_str = aux_ein_ids + ',' + randu_ids + '->' + \
                    einsum_ids[:n_dims_mpo_core]
                mpo_core = torch.einsum(einsum_str, mpo_core, randu_t)
            
            
            # Trimming down
            if verbose:
                print('\t* Trimming down mpo_core...', end=' ')
             
            if kr < (n_rows - 1):
                mpo_core, _ = trimming(tensor=mpo_core,
                                       dim=down_pos,
                                       rank=rank,
                                       cum_percentage=cum_percentage)
            
            if verbose:
                torch.cuda.synchronize(device=device)
                aux_time = time.time() - start_time
                print(f'Done! ({aux_time:.2f}s)')
            
            
            # Solving for MPOs
            if verbose:
                print('\t* Solving for peps_core...', end=' ')
            
            if kr == 0:
                peps_core = mpo_core
            else:
                peps_core = solving_mpo(mpo_kr_minus_1[kc], mpo_core,
                                        rank, kr, kc, n_rows, n_cols)
            
            if verbose:
                torch.cuda.synchronize(device=device)
                aux_time = time.time() - start_time
                print(f'Done! ({aux_time:.2f}s)')
            
            
            # Trimming right peps core
            if verbose:
                print('\t* Trimming right peps_core...', end=' ')
            
            if kr > 0:
                if kc > 0:
                    aux_peps_core = peps_core.reshape(-1, peps_core.size(-1))
                    aux_peps_core = aux_peps_core @ prev_vh_peps.T
                    peps_core = aux_peps_core.view(*peps_core.shape[:-1],
                                                prev_vh_peps.size(0))
                if kc < (n_cols - 1):
                    peps_core, prev_vh_peps = trimming(tensor=peps_core,
                                                    dim=right_pos,
                                                    rank=rank,
                                                    cum_percentage=cum_percentage)
            
            if verbose:
                torch.cuda.synchronize(device=device)
                aux_time = time.time() - start_time
                print(f'Done! ({aux_time:.2f}s)')
            
            if verbose:
                print(f'\t*Core shape: {peps_core.shape}')
            
            peps_row.append(peps_core)
            
            
            # Create mpo_kr_minus_1
            if verbose:
                print('\t* Create mpo_kr_minus_1...', end=' ')
            
            if kr < (n_rows - 1):
                if kr == 0:
                    sr_kr_kc = aux_x_kr_kc
                    aux_sr_kr_kc = aux_embedding(sr_kr_kc[0].to(device)).cpu()
                    
                    aux_ein_ids = einsum_ids[:(n_dims_mpo_core - 1)]
                    einsum_str = 'i' + aux_ein_ids + ',' + 'bi->' + \
                        'b' + aux_ein_ids
                    mpo_core = torch.einsum(einsum_str,
                                            mpo_core,
                                            aux_sr_kr_kc)
                    
                else:
                    sr_kr_kc = create_projector([Sr_kr_minus_1_kc], aux_x_kr_kc)
                    
                    # # Fix ordering
                    # # TODO: this is auxiliary solution (?)
                    # aux_Sr_kr_minus_1_kc = Sr_kr_minus_1_kc[0].index_select(
                    #     dim=0,
                    #     index=sr_kr_kc[0][0])
                    
                    # aux_col = [aux_Sr_kr_minus_1_kc, sr_kr_kc[1]]
                    # aux = torch.cat(aux_col, dim=1)
                    
                    # aux_Sr_kr_kc = sketch_samples[:, :(kr + 1), kc:(kc + 1)].unique(
                    #     sorted=True,
                    #     dim=0)
                    
                    # assert torch.equal(aux_Sr_kr_kc, aux)
                    
                    # aux, inv_ids = aux.unique(dim=0,
                    #                           sorted=True,
                    #                           return_inverse=True)
                    
                    # sr_kr_kc = ([sr_kr_kc[0][0][inv_ids]], sr_kr_kc[1][inv_ids])
                    
                    mpo_core = mpo_core[:, sr_kr_kc[0][0]]
                    aux_sr_kr_kc = aux_embedding(sr_kr_kc[1].to(device)).cpu()
                    
                    aux_ein_ids = einsum_ids[:(n_dims_mpo_core - 2)]
                    einsum_str = 'ib' + aux_ein_ids + ',' + 'bi->' + \
                        'b' + aux_ein_ids
                    mpo_core = torch.einsum(einsum_str,
                                            mpo_core,
                                            aux_sr_kr_kc)
                    
                    right_pos -= 1
                
                # Trimming right
                if kc > 0:
                    aux_mpo_core = mpo_core.reshape(-1, mpo_core.size(-1))
                    aux_mpo_core = aux_mpo_core @ prev_vh_mpo.T
                    mpo_core = aux_mpo_core.view(*mpo_core.shape[:-1],
                                                 prev_vh_mpo.size(0))
                if kc < (n_cols - 1):
                    mpo_core, prev_vh_mpo = trimming(tensor=mpo_core,
                                                     dim=right_pos,
                                                     rank=int(right_rank/rank),
                                                     cum_percentage=1.)
                
                aux_mpo_kr_minus_1.append(mpo_core)
            
            if verbose:
                torch.cuda.synchronize(device=device)
                aux_time = time.time() - start_time
                print(f'Done! ({aux_time:.2f}s)') 
        
        
        peps.append(peps_row)
        
        # Set variables for next iteration
        if kr < (n_rows - 1):
            mpo_kr_minus_1 = aux_mpo_kr_minus_1
    
    if verbose:
        torch.cuda.synchronize(device=device)
        total_time = time.time() - start_time
        print(f'\n\n Tensorization finished in {total_time:.2f}s\n')
        
    if return_info:
        total_time = time.time() - start_time
        info = {'total_time': total_time}
        
        return peps, info
            
    return peps
