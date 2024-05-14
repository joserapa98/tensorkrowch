"""
This script contains:

    * vec_to_mps
    * mat_to_mpo
"""

from typing import (List, Optional)

import torch


def vec_to_mps(vec: torch.Tensor,
               n_batches: int = 0,
               rank: Optional[int] = None,
               cum_percentage: Optional[float] = None,
               cutoff: Optional[float] = None,
               renormalize: bool = False) -> List[torch.Tensor]:
    r"""
    Splits a vector into a sequence of MPS tensors via consecutive SVD
    decompositions. The resultant tensors can be used to instantiate a
    :class:`~tensorkrowch.models.MPS` with ``boundary = "obc"``.
    
    The number of resultant tensors and their respective physical dimensions
    depend on the shape of the input vector. That is, if one expects to recover
    a MPS with physical dimensions
    
    .. math::
    
        d_1 \times \cdots \times d_n
    
    the input vector will have to be provided with that shape. This can be done
    with `reshape <https://pytorch.org/docs/stable/generated/torch.reshape.html>`_.
    
    If the input vector has batch dimensions, having as shape
    
    .. math::
    
        b_1 \times \cdots \times b_m \times d_1 \times \cdots \times d_n
    
    the number of batch dimensions :math:`m` can be specified in ``n_batches``.
    In this case, the resultant tensors will all have the extra batch dimensions.
    These tensors can be used to instantiate a :class:`~tensorkrowch.models.MPSData`
    with ``boundary = "obc"``.
    
    To specify the bond dimension of each cut done via SVD, one can use the
    arguments ``rank``, ``cum_percentage`` and ``cutoff``. If more than
    one is specified, the resulting rank will be the one that satisfies all
    conditions.

    Parameters
    ----------
    vec : torch.Tensor
        Input vector to decompose.
    n_batches : int
        Number of batch dimensions of the input vector. Each resultant tensor
        will have also the corresponding batch dimensions. It should be between
        0 and the rank of ``vec``.
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
    renormalize : bool
            Indicates whether nodes should be renormalized after SVD/QR
            decompositions. If not, it may happen that the norm explodes as it
            is being accumulated from all nodes. Renormalization aims to avoid
            this undesired behavior by extracting the norm of each node on a
            logarithmic scale after SVD/QR decompositions are computed. Finally,
            the normalization factor is evenly distributed among all nodes of
            the MPS.

    Returns
    -------
    List[torch.Tensor]
    """
    if not isinstance(vec, torch.Tensor):
        raise TypeError('`vec` should be torch.Tensor type')
    
    if n_batches > len(vec.shape):
        raise ValueError(
            '`n_batches` should be between 0 and the rank of `vec`')
    
    batches_shape = vec.shape[:n_batches]
    phys_dims = torch.tensor(vec.shape[n_batches:])
    
    log_norm = 0
    prev_bond = 1
    tensors = []
    for i in range(len(phys_dims) - 1):
        vec = vec.reshape(*batches_shape,
                          prev_bond * phys_dims[i],
                          phys_dims[(i + 1):].prod())
        
        u, s, vh = torch.linalg.svd(vec, full_matrices=False)
        
        lst_ranks = []
        
        if rank is None:
            aux_rank = s.shape[-1]
            lst_ranks.append(aux_rank)
        else:
            lst_ranks.append(min(max(1, int(rank)), s.shape[-1]))
            
        if cum_percentage is not None:
            s_percentages = s.cumsum(-1) / \
                (s.sum(-1, keepdim=True).expand(s.shape) + 1e-10) # To avoid having all 0's
            cum_percentage_tensor = cum_percentage * torch.ones_like(s)
            cp_rank = torch.lt(
                s_percentages,
                cum_percentage_tensor
                ).view(-1, s.shape[-1]).any(dim=0).sum()
            lst_ranks.append(max(1, cp_rank.item() + 1))
            
        if cutoff is not None:
            cutoff_tensor = cutoff * torch.ones_like(s)
            co_rank = torch.ge(
                s,
                cutoff_tensor
                ).view(-1, s.shape[-1]).any(dim=0).sum()
            lst_ranks.append(max(1, co_rank.item()))
        
        # Select rank from specified restrictions
        aux_rank = min(lst_ranks)
        
        u = u[..., :aux_rank]
        if i > 0:
            u = u.reshape(*batches_shape, prev_bond, phys_dims[i], aux_rank)
            
        s = s[..., :aux_rank]
        vh = vh[..., :aux_rank, :]
        
        if renormalize:
            aux_norm = s.norm(dim=-1, keepdim=True)
            if not aux_norm.isinf().any() and (aux_norm > 0).any():
                s = s / aux_norm
                log_norm += aux_norm.log()
        
        tensors.append(u)
        prev_bond = aux_rank
        
        if vh.is_complex():
            s = s.to(vh.dtype)
        vec = torch.diag_embed(s) @ vh
        
    tensors.append(vec)
    
    if log_norm is not 0:
        rescale = (log_norm / len(tensors)).exp()
        for vec in tensors:
            vec *= rescale.view(*vec.shape[:n_batches],
                                *([1] * len(vec.shape[n_batches:])))
    
    return tensors


def mat_to_mpo(mat: torch.Tensor,
               rank: Optional[int] = None,
               cum_percentage: Optional[float] = None,
               cutoff: Optional[float] = None,
               renormalize: bool = False) -> List[torch.Tensor]:
    r"""
    Splits a matrix into a sequence of MPO tensors via consecutive SVD
    decompositions. The resultant tensors can be used to instantiate a
    :class:`~tensorkrowch.models.MPO` with ``boundary = "obc"``.
    
    The number of resultant tensors and their respective input/output dimensions
    depend on the shape of the input matrix. That is, if one expects to recover
    a MPO with input/output dimensions
    
    .. math::
    
        in_1 \times out_1 \times \cdots \times in_n \times out_n
    
    the input matrix will have to be provided with that shape. Thus it must
    have an even number of dimensions. To accomplish this, it may happen that
    some input/output dimensions are 1. This can be done with
    `reshape <https://pytorch.org/docs/stable/generated/torch.reshape.html>`_.
    
    To specify the bond dimension of each cut done via SVD, one can use the
    arguments ``rank``, ``cum_percentage`` and ``cutoff``. If more than
    one is specified, the resulting rank will be the one that satisfies all
    conditions.

    Parameters
    ----------
    mat : torch.Tensor
        Input matrix to decompose. It must have an even number of dimensions.
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
    renormalize : bool
            Indicates whether nodes should be renormalized after SVD/QR
            decompositions. If not, it may happen that the norm explodes as it
            is being accumulated from all nodes. Renormalization aims to avoid
            this undesired behavior by extracting the norm of each node on a
            logarithmic scale after SVD/QR decompositions are computed. Finally,
            the normalization factor is evenly distributed among all nodes of
            the MPS.

    Returns
    -------
    List[torch.Tensor]
    """
    if not isinstance(mat, torch.Tensor):
        raise TypeError('`mat` should be torch.Tensor type')
    if not len(mat.shape) % 2 == 0:
        raise ValueError('`mat` have an even number of dimensions')
    
    in_out_dims = torch.tensor(mat.shape)
    if len(in_out_dims) == 2:
        return [mat]
    
    log_norm = 0
    prev_bond = 1
    tensors = []
    for i in range(0, len(in_out_dims) - 2, 2):
        mat = mat.reshape(prev_bond * in_out_dims[i] * in_out_dims[i + 1],
                          in_out_dims[(i + 2):].prod())
        
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        
        lst_ranks = []
        
        if rank is None:
            aux_rank = s.shape[-1]
            lst_ranks.append(aux_rank)
        else:
            lst_ranks.append(min(max(1, int(rank)), s.shape[-1]))
            
        if cum_percentage is not None:
            s_percentages = s.cumsum(-1) / \
                (s.sum(-1, keepdim=True).expand(s.shape) + 1e-10) # To avoid having all 0's
            cum_percentage_tensor = cum_percentage * torch.ones_like(s)
            cp_rank = torch.lt(
                s_percentages,
                cum_percentage_tensor
                ).view(-1, s.shape[-1]).any(dim=0).sum()
            lst_ranks.append(max(1, cp_rank.item() + 1))
            
        if cutoff is not None:
            cutoff_tensor = cutoff * torch.ones_like(s)
            co_rank = torch.ge(
                s,
                cutoff_tensor
                ).view(-1, s.shape[-1]).any(dim=0).sum()
            lst_ranks.append(max(1, co_rank.item()))
        
        # Select rank from specified restrictions
        aux_rank = min(lst_ranks)
        
        u = u[..., :aux_rank]
        if i == 0:
            u = u.reshape(in_out_dims[i], in_out_dims[i + 1], aux_rank)
            u = u.permute(0, 2, 1) # input x right x output
        else:
            u = u.reshape(prev_bond, in_out_dims[i], in_out_dims[i + 1], aux_rank)
            u = u.permute(0, 1, 3, 2) # left x input x right x output
            
        s = s[..., :aux_rank]
        vh = vh[..., :aux_rank, :]
        
        if renormalize:
            aux_norm = s.norm(dim=-1)
            if not aux_norm.isinf() and (aux_norm > 0):
                s = s / aux_norm
                log_norm += aux_norm.log()
        
        tensors.append(u)
        prev_bond = aux_rank
        
        if vh.is_complex():
            s = s.to(vh.dtype)
        mat = torch.diag_embed(s) @ vh
    
    mat = mat.reshape(aux_rank, in_out_dims[-2], in_out_dims[-1])
    tensors.append(mat)
    
    if renormalize:
        rescale = (log_norm / len(tensors)).exp()
        for mat in tensors:
            mat *= rescale
    
    return tensors
