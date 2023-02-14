"""
Embedding methods
"""

from math import pi, sqrt
import torch

from tensorkrowch.utils import binomial_coeffs


def unit(data: torch.Tensor, size: int) -> torch.Tensor:
    r"""
    Embedds the data tensor using the local feature map defined in the original
    `paper <https://arxiv.org/abs/1605.05775>`_ by E. Miles Stoudenmire and David
    J. Schwab.
    
    Parameters
    ----------
    data : torch.Tensor
        Data tensor with shape
        
        .. math::
        
            n_{features} \times batch\_size \times feature\_size
            
        where :math:`feature\_size = 1`.
    size : int
        New feature size.
            
    Returns
    -------
    torch.Tensor
        New data tensor with shape
        
        .. math::
        
            n_{features} \times batch\_size \times size
    """
    lst_tensors = []
    for i in range(1, size + 1):
        aux = sqrt(binomial_coeffs(size - 1, i - 1)) * \
                (pi / 2 * data).cos().pow(size - i) * \
                (pi / 2 * data).sin().pow(i - 1)
        lst_tensors.append(aux)
    return torch.stack(lst_tensors, dim=-1)


def add_ones(data: torch.Tensor, dim: int = -1) -> torch.Tensor:
    r"""
    Embedds the data tensor adding 1's as the first component of each vector.
    
    .. math::

        \hat{x}_i = \begin{bmatrix}
                        1 \\
                        x_i
                    \end{bmatrix}
    
    Parameters
    ----------
    data : torch.Tensor
        Data tensor with shape
        
        .. math::
        
            n_{features} \times batch\_size
    dim : int
        New dimension (axis) to insert. Should be between 0 and the rank of
        ``data``.
            
    Returns
    -------
    torch.Tensor
        New data tensor with shape
        
        .. math::
        
            n_{features} \times batch\_size \times 2
    """
    return torch.stack([torch.ones_like(data), data], dim=dim)
