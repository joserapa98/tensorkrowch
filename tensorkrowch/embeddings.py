"""
Embedding methods
"""

from math import pi, sqrt

import torch

from tensorkrowch.utils import binomial_coeffs


def unit(data: torch.Tensor, dim: int = 2) -> torch.Tensor:
    r"""
    Embedds the data tensor using the local feature map defined in the original
    `paper <https://arxiv.org/abs/1605.05775>`_ by E. Miles Stoudenmire and David
    J. Schwab.
    
    Given a vector :math:`x` with :math:`N` components, the embedding
    will be
    
    .. math::
    
        \Phi^{s_1...s_N}(x) = \phi^{s_1}(x_1)\otimes\cdots\otimes\phi^{s_N}(x_N)
        
    where each :math:`\phi^{s_j}(x_j)` is
    
    .. math::
    
        \phi^{s_j}(x_j) = \sqrt{\binom{d-1}{s_j-1}}
            (\cos{\frac{\pi}{2}x_j})^{d-s_j}(\sin{\frac{\pi}{2}x_j})^{s_j-1}
            
    being :math:`d` the desired output dimension (``dim``).
    
    
    Parameters
    ----------
    data : torch.Tensor
        Data tensor with shape 
        
        .. math::
        
            batch\_size \times n_{features}
        
        That is, ``data`` is a (batch) vector with :math:`n_{features}`
        components. The :math:`batch\_size` is optional.
    dim : int
        New feature dimension.
            
    Returns
    -------
    torch.Tensor
        New data tensor with shape
        
        .. math::
        
            batch\_size \times n_{features} \times dim
            
    Examples
    --------
    >>> a = torch.ones(5)
    >>> a
    tensor([1., 1., 1., 1., 1.])
    
    >>> emb_a = tk.embeddings.unit(a)
    >>> emb_a
    tensor([[-4.3711e-08,  1.0000e+00],
            [-4.3711e-08,  1.0000e+00],
            [-4.3711e-08,  1.0000e+00],
            [-4.3711e-08,  1.0000e+00],
            [-4.3711e-08,  1.0000e+00]])
    
    >>> b = torch.randn(100, 5)
    >>> emb_b = tk.embeddings.unit(b)
    >>> emb_b.shape
    torch.Size([100, 5, 2])
    """
    lst_tensors = []
    for i in range(1, dim + 1):
        aux = sqrt(binomial_coeffs(dim - 1, i - 1)) * \
              (pi / 2 * data).cos().pow(dim - i) * \
              (pi / 2 * data).sin().pow(i - 1)
        lst_tensors.append(aux)
    return torch.stack(lst_tensors, dim=-1)


def add_ones(data: torch.Tensor, axis: int = -1) -> torch.Tensor:
    r"""
    Embedds the data tensor adding 1's as the first component of each vector.
    That is, given a vector
    
    .. math::

        x = \begin{bmatrix}
                x_1\\
                \vdots\\
                x_N
            \end{bmatrix}
            
    returns a matrix
    
    .. math::

        \hat{x} = \begin{bmatrix}
                        1 & x_1\\
                        \vdots & \vdots\\
                        1 & x_N
                  \end{bmatrix}
    
    Parameters
    ----------
    data : torch.Tensor
        Data tensor with shape  
        
        .. math::
        
            batch\_size \times n_{features}
            
        That is, ``data`` is a (batch) vector with :math:`n_{features}`
        components. The :math:`batch\_size` is optional.
    axis : int
        Axis where the ``data`` tensor is 'expanded' with the 1's. Should be
        between 0 and the rank of ``data``. By default, it is -1, which returns
        a tensor with shape  
        
        .. math::
        
            batch\_size \times n_{features} \times 2
            
    Returns
    -------
    torch.Tensor
    
    Examples
    --------
    >>> a = 2 * torch.ones(5)
    >>> a
    tensor([2., 2., 2., 2., 2.])
    
    >>> emb_a = tk.embeddings.add_ones(a)
    >>> emb_a
    tensor([[1., 2.],
            [1., 2.],
            [1., 2.],
            [1., 2.],
            [1., 2.]])
    
    >>> b = torch.randn(100, 5)
    >>> emb_b = tk.embeddings.add_ones(b)
    >>> emb_b.shape
    torch.Size([100, 5, 2])
    """
    return torch.stack([torch.ones_like(data), data], dim=axis)


def poly(data: torch.Tensor, degree: int = 2, axis: int = -1) -> torch.Tensor:
    r"""
    Embedds the data tensor stacking powers of it. That is, given the vector
    
    .. math::

        x = \begin{bmatrix}
                x_1\\
                \vdots\\
                x_N
            \end{bmatrix}
            
    returns a matrix
    
    .. math::

        \hat{x} = \begin{bmatrix}
                        1 & x_1 & x_1^2 & \cdots & x_1^n\\
                        \vdots & \vdots & \vdots & \ddots & \vdots\\
                        1 & x_N & x_N^2 & \cdots & x_N^n
                  \end{bmatrix}
        
    being :math:`n` the ``degree`` of the monomials.
    
    If ``degree = 1``, it is equivalent to :func:`add_ones`.
    
    Parameters
    ----------
    data : torch.Tensor
        Data tensor with shape 
        
        .. math::
        
            batch\_size \times n_{features}
            
        That is, ``data`` is a (batch) vector with :math:`n_{features}`
        components. The :math:`batch\_size` is optional.
    degree : int
        Maximum degree of the monomials.
    axis : int
        Axis where the ``data`` tensor is 'expanded' with monomials. Should be
        between 0 and the rank of ``data``. By default, it is -1, which returns
        a tensor with shape 
        
        .. math::
        
            batch\_size \times n_{features} \times (degree + 1)
            
    Returns
    -------
    torch.Tensor
    
    Examples
    --------
    >>> a = 2 * torch.ones(5)
    >>> a
    tensor([2., 2., 2., 2., 2.])
    
    >>> emb_a = tk.embeddings.poly(a)
    >>> emb_a
    tensor([[1., 2., 4.],
            [1., 2., 4.],
            [1., 2., 4.],
            [1., 2., 4.],
            [1., 2., 4.]])
    
    >>> b = torch.randn(100, 5)
    >>> emb_b = tk.embeddings.poly(b)
    >>> emb_b.shape
    torch.Size([100, 5, 3])
    """
    lst_powers = []
    for i in range(degree + 1):
        lst_powers.append(data.pow(i))
    return torch.stack(lst_powers, dim=axis)
