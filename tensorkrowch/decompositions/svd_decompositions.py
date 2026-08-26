"""
This script contains:

    * vec_to_mps
    * mat_to_mpo
"""

from typing import (List, Optional)
import torch

from tensorkrowch.decompositions.svd.tt import TTSVD
from tensorkrowch.utils import truncated_svd


def vec_to_mps(vec: torch.Tensor,
               n_batches: int = 0,
               rank: Optional[int] = None,
               cutoff: Optional[float] = None,
               atol: Optional[float] = None,
               rtol: Optional[float] = None,
               cum_percentage: Optional[float] = None,
               renormalize: bool = False) -> List[torch.Tensor]:
    r"""
    Splits a vector into a sequence of :class:`~tensorkrowch.models.MPS`
    tensors via consecutive SVD decompositions. The resultant tensors can be
    used to instantiate a
    :class:`~tensorkrowch.models.MPS` with ``boundary = "obc"``.
    
    The number of resultant tensors and their respective physical dimensions
    depend on the shape of the input vector. That is, if one expects to recover
    a :class:`~tensorkrowch.models.MPS` with physical dimensions
    
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
    truncation criterions. If more than one criterion is specified, the final
    rank is the minimum one, i.e. the one imposed by the most restrictive
    criterion.

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
    cutoff : float, optional
        Minimum singular value to keep. It must be non-negative. Singular
        values ``<= cutoff`` are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the accumulated sum of squares is ``<= atol``. It must be non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the tail sum of squares divided by the total sum of squares is
        ``<= rtol``. It must be in ``[0, 1]``.
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent to
        setting ``rtol = 1 - cum_percentage``. It must be in ``[0, 1]``.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
            cum\_percentage

    renormalize : bool
            Indicates whether nodes should be renormalized after SVD/QR
            decompositions. If not, it may happen that the norm explodes as it
            is being accumulated from all nodes. Renormalization aims to avoid
            this undesired behavior by extracting the norm of each node on a
            logarithmic scale after SVD/QR decompositions are computed. Finally,
            the normalization factor is evenly distributed among all nodes of
            the :class:`~tensorkrowch.models.MPS`.

    Returns
    -------
    List[torch.Tensor]

    Examples
    --------
    >>> vec = torch.arange(16.).reshape(2, 2, 2, 2)
    >>> tensors = tk.decompositions.vec_to_mps(vec, rank=2)
    >>> [tuple(t.shape) for t in tensors]
    [(2, 2), (2, 2, 2), (2, 2, 2), (2, 2)]
    """
    if not isinstance(vec, torch.Tensor):
        raise TypeError('`vec` should be torch.Tensor type')

    if n_batches > vec.ndim:
        raise ValueError(
            '`n_batches` should be between 0 and the rank of `vec`')

    return TTSVD(
        tensor=vec,
        n_batches=n_batches,
        output_device=None).fit(
            rank=rank,
            cutoff=cutoff,
            atol=atol,
            rtol=rtol,
            cum_percentage=cum_percentage,
            renormalize=renormalize).cores


def mat_to_mpo(mat: torch.Tensor,
               rank: Optional[int] = None,
               cutoff: Optional[float] = None,
               atol: Optional[float] = None,
               rtol: Optional[float] = None,
               cum_percentage: Optional[float] = None,
               renormalize: bool = False) -> List[torch.Tensor]:
    r"""
    Splits a matrix into a sequence of :class:`~tensorkrowch.models.MPO`
    tensors via consecutive SVD decompositions. The resultant tensors can be
    used to instantiate a
    :class:`~tensorkrowch.models.MPO` with ``boundary = "obc"``.
    
    The dimensions of ``mat`` must be interleaved by site, with each input
    dimension immediately followed by its corresponding output dimension. The
    number of resultant tensors and their respective input/output dimensions
    depend on this shape. That is, if one expects to recover a
    :class:`~tensorkrowch.models.MPO` with
    input/output dimensions
    
    .. math::
    
        in_1 \times out_1 \times \cdots \times in_n \times out_n
    
    the input matrix must have shape
    ``(in_1, out_1, ..., in_n, out_n)``. A tensor whose axes are grouped as
    ``(in_1, ..., in_n, out_1, ..., out_n)`` has to be permuted first. Thus the
    input must have an even number of dimensions. To accomplish this, it may
    happen that some input/output dimensions are 1. This can be done with
    `reshape <https://pytorch.org/docs/stable/generated/torch.reshape.html>`_.
    
    To specify the bond dimension of each cut done via SVD, one can use the
    truncation criterions. If more than one criterion is specified, the final
    rank is the minimum one, i.e. the one imposed by the most restrictive
    criterion.

    Parameters
    ----------
    mat : torch.Tensor
        Input matrix to decompose. It must have an even number of dimensions.
    rank : int, optional
        Number of singular values to keep.
    cutoff : float, optional
        Minimum singular value to keep. It must be non-negative. Singular
        values ``<= cutoff`` are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the accumulated sum of squares is ``<= atol``. It must be non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the tail sum of squares divided by the total sum of squares is
        ``<= rtol``. It must be in ``[0, 1]``.
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent to
        setting ``rtol = 1 - cum_percentage``. It must be in ``[0, 1]``.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
            cum\_percentage

    renormalize : bool
            Indicates whether nodes should be renormalized after SVD/QR
            decompositions. If not, it may happen that the norm explodes as it
            is being accumulated from all nodes. Renormalization aims to avoid
            this undesired behavior by extracting the norm of each node on a
            logarithmic scale after SVD/QR decompositions are computed. Finally,
            the normalization factor is evenly distributed among all nodes of
            the :class:`~tensorkrowch.models.MPO`.

    Returns
    -------
    List[torch.Tensor]

    Examples
    --------
    >>> mat = torch.arange(64.).reshape(2, 2, 2, 2, 2, 2)
    >>> tensors = tk.decompositions.mat_to_mpo(mat, rank=2)
    >>> [tuple(t.shape) for t in tensors]
    [(2, 2, 2), (2, 2, 2, 2), (2, 2, 2)]
    """
    if not isinstance(mat, torch.Tensor):
        raise TypeError('`mat` should be torch.Tensor type')
    if not mat.ndim % 2 == 0:
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
        
        u, s, vh = truncated_svd(tensor=mat,
                                 rank=rank,
                                 cutoff=cutoff,
                                 atol=atol,
                                 rtol=rtol,
                                 cum_percentage=cum_percentage)
        aux_rank = s.shape[-1]
        
        if i == 0:
            u = u.reshape(in_out_dims[i], in_out_dims[i + 1], aux_rank)
            u = u.permute(0, 2, 1) # input x right x output
        else:
            u = u.reshape(prev_bond, in_out_dims[i], in_out_dims[i + 1], aux_rank)
            u = u.permute(0, 1, 3, 2) # left x input x right x output
        
        if renormalize:
            aux_norm = s.norm(dim=-1)
            if not aux_norm.isinf() and (aux_norm > 0):
                s = s / aux_norm
                log_norm += aux_norm.log()
        
        # If u is not cloned, it leads to errors in backward computation
        tensors.append(u.clone())
        prev_bond = aux_rank
        
        if vh.is_complex():
            s = s.to(vh.dtype)
        mat = torch.diag_embed(s) @ vh
    
    mat = mat.reshape(aux_rank, in_out_dims[-2], in_out_dims[-1])
    tensors.append(mat)
    
    if renormalize and (log_norm != 0):
        rescale = (log_norm / len(tensors)).exp()
        for mat in tensors:
            mat *= rescale
    
    return tensors
