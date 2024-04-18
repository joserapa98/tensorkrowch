"""
Embedding methods
"""

from math import pi, sqrt

import torch

from tensorkrowch.utils import binomial_coeffs


def unit(data: torch.Tensor, dim: int = 2, axis: int = -1) -> torch.Tensor:
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
        
            (batch_0 \times \cdots \times batch_n \times) n_{features}
        
        That is, ``data`` is a (batch) vector with :math:`n_{features}`
        components. The :math:`batch` sizes are optional.
    dim : int
        New feature dimension.
    axis : int
        Axis where the ``data`` tensor is 'expanded'. Should be between 0 and
        the rank of ``data``. By default, it is -1, which returns a tensor with
        shape  
        
        .. math::
        
            (batch_0 \times \cdots \times batch_n \times) n_{features}
            \times dim
            
    Returns
    -------
    torch.Tensor
            
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
    >>> emb_b = tk.embeddings.unit(b, dim=6)
    >>> emb_b.shape
    torch.Size([100, 5, 6])
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError('`data` should be torch.Tensor type')
    
    lst_tensors = []
    for i in range(1, dim + 1):
        aux = sqrt(binomial_coeffs(dim - 1, i - 1)) * \
              (pi / 2 * data).cos().pow(dim - i) * \
              (pi / 2 * data).sin().pow(i - 1)
        lst_tensors.append(aux)
    return torch.stack(lst_tensors, dim=axis)


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
        
            (batch_0 \times \cdots \times batch_n \times) n_{features}
        
        That is, ``data`` is a (batch) vector with :math:`n_{features}`
        components. The :math:`batch` sizes are optional.
    axis : int
        Axis where the ``data`` tensor is 'expanded'. Should be between 0 and
        the rank of ``data``. By default, it is -1, which returns a tensor with
        shape  
        
        .. math::
        
            (batch_0 \times \cdots \times batch_n \times) n_{features}
            \times 2
            
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
    if not isinstance(data, torch.Tensor):
        raise TypeError('`data` should be torch.Tensor type')
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
        
            (batch_0 \times \cdots \times batch_n \times) n_{features}
        
        That is, ``data`` is a (batch) vector with :math:`n_{features}`
        components. The :math:`batch` sizes are optional.
    degree : int
        Maximum degree of the monomials. The feature dimension will be
        ``degree + 1``.
    axis : int
        Axis where the ``data`` tensor is 'expanded'. Should be between 0 and
        the rank of ``data``. By default, it is -1, which returns a tensor with
        shape  
        
        .. math::
        
            (batch_0 \times \cdots \times batch_n \times) n_{features}
            \times (degree + 1)
            
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
    >>> emb_b = tk.embeddings.poly(b, degree=3)
    >>> emb_b.shape
    torch.Size([100, 5, 4])
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError('`data` should be torch.Tensor type')
    
    lst_powers = []
    for i in range(degree + 1):
        lst_powers.append(data.pow(i))
    return torch.stack(lst_powers, dim=axis)


def discretize(data: torch.Tensor,
               level: int,
               base: int = 2,
               axis: int = -1) -> torch.Tensor:
    r"""
    Embedds the data tensor discretizing each variable in a certain ``basis``
    and with a certain ``level`` of precision, assuming the values to discretize
    are all between 0 and 1. That is, given a vector
    
    .. math::

        x = \begin{bmatrix}
                x_1\\
                \vdots\\
                x_N
            \end{bmatrix}
            
    returns a matrix
    
    .. math::

        \hat{x} = \begin{bmatrix}
                        \lfloor x_1 b^1 \rfloor \mod b & \cdots &
                            \lfloor x_1 b^{l} \rfloor \mod b\\
                        \vdots & \ddots & \vdots\\
                        \lfloor x_N b^1 \rfloor \mod b & \cdots &
                            \lfloor x_N b^{l} \rfloor \mod b
                  \end{bmatrix}
    
    where :math:`b` stands for ``base``, and :math:`l` for ``level``.
    
    Parameters
    ----------
    data : torch.Tensor
        Data tensor with shape 
        
        .. math::
        
            (batch_0 \times \cdots \times batch_n \times) n_{features}
        
        That is, ``data`` is a (batch) vector with :math:`n_{features}`
        components. The :math:`batch` sizes are optional. The ``data`` tensor
        is assumed to have elements between 0 and 1.
    level : int
        Level of precision of the discretization. This will be the new feature
        dimension.
    base : int
        The base of the discretization.
    axis : int
        Axis where the ``data`` tensor is 'expanded'. Should be between 0 and
        the rank of ``data``. By default, it is -1, which returns a tensor with
        shape  
        
        .. math::
        
            (batch_0 \times \cdots \times batch_n \times) n_{features}
            \times level
            
    Returns
    -------
    torch.Tensor
            
    Examples
    --------
    >>> a = torch.tensor([0, 0.5, 0.75, 1])
    >>> a
    tensor([0.0000, 0.5000, 0.7500, 1.0000])
    
    >>> emb_a = tk.embeddings.discretize(a, level=3)
    >>> emb_a
    tensor([[0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [1., 1., 1.]])
    
    >>> b = torch.rand(100, 5)
    >>> emb_b = tk.embeddings.discretize(b, level=3)
    >>> emb_b.shape
    torch.Size([100, 5, 3])
    
    To embed a data tensor with elements between 0 and 1 as basis vectors, one
    can concatenate :func:`discretize` with :func:`basis`.
    
    >>> a = torch.rand(100, 10)
    >>> emb_a = tk.embeddings.discretize(a, level=1, base=5)
    >>> emb_a.shape
    torch.Size([100, 10, 1])
    
    >>> emb_a = tk.embeddings.basis(emb_a.squeeze(2).int(), dim=5)
    >>> emb_a.shape
    torch.Size([100, 10, 5])
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError('`data` should be torch.Tensor type')
    if not torch.ge(data, torch.zeros_like(data)).all():
        raise ValueError('Elements of `data` should be between 0 and 1')
    if not torch.le(data, torch.ones_like(data)).all():
        raise ValueError('Elements of `data` should be between 0 and 1')

    max_discr_value = (base - 1) * sum([base ** -i for i in range(1, level + 1)])
    data = torch.where(data > max_discr_value, max_discr_value, data)
    
    base = torch.tensor(base, device=data.device)
    ids = [torch.remainder((data * base.pow(i)).floor(), base)
           for i in range(1, level + 1)]
    ids = torch.stack(ids, dim=axis)
    return ids


def basis(data: torch.Tensor, dim: int = 2, axis: int = -1) -> torch.Tensor:
    r"""
    Embedds the data tensor transforming each value, assumed to be an integer
    between 0 and ``dim - 1``, into the corresponding vector of the
    computational basis. That is, given a vector
    
    .. math::

        x = \begin{bmatrix}
                x_1\\
                \vdots\\
                x_N
            \end{bmatrix}
            
    returns a matrix
    
    .. math::

        \hat{x} = \begin{bmatrix}
                      \lvert x_1 \rangle\\
                      \vdots\\
                      \lvert x_N \rangle
                  \end{bmatrix}
    
    Parameters
    ----------
    data : torch.Tensor
        Data tensor with shape 
        
        .. math::
        
            (batch_0 \times \cdots \times batch_n \times) n_{features}
        
        That is, ``data`` is a (batch) vector with :math:`n_{features}`
        components. The :math:`batch` sizes are optional. The ``data`` tensor
        is assumed to have integer elements between 0 and ``dim - 1``.
    dim : int
        The dimension of the computational basis. This will be the new feature
        dimension.
    axis : int
        Axis where the ``data`` tensor is 'expanded'. Should be between 0 and
        the rank of ``data``. By default, it is -1, which returns a tensor with
        shape  
        
        .. math::
        
            (batch_0 \times \cdots \times batch_n \times) n_{features}
            \times dim
            
    Returns
    -------
    torch.Tensor
            
    Examples
    --------
    >>> a = torch.arange(5)
    >>> a
    tensor([0, 1, 2, 3, 4])
    
    >>> emb_a = tk.embeddings.basis(a, dim=5)
    >>> emb_a
    tensor([[1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]])
    
    >>> b = torch.randint(low=0, high=10, size=(100, 5))
    >>> emb_b = tk.embeddings.basis(b, dim=10)
    >>> emb_b.shape
    torch.Size([100, 5, 10])
    
    To embed a data tensor with elements between 0 and 1 as basis vectors, one
    can concatenate :func:`discretize` with :func:`basis`.
    
    >>> a = torch.rand(100, 10)
    >>> emb_a = tk.embeddings.discretize(a, level=1, base=5)
    >>> emb_a.shape
    torch.Size([100, 10, 1])
    
    >>> emb_a = tk.embeddings.basis(emb_a.squeeze(2).int(), dim=5)
    >>> emb_a.shape
    torch.Size([100, 10, 5])
    """
    if not isinstance(data, torch.Tensor):
        raise TypeError('`data` should be torch.Tensor type')
    if torch.is_floating_point(data):
        raise ValueError('`data` should be a tensor of integers')
    if not torch.ge(data, torch.zeros_like(data)).all():
        raise ValueError('Elements of `data` should be between 0 and (dim - 1)')
    if not torch.le(data, torch.ones_like(data) * (dim - 1)).all():
        raise ValueError('Elements of `data` should be between 0 and (dim - 1)')
    
    ids = torch.arange(dim, device=data.device).repeat(*data.shape, 1)
    ids = torch.where(ids == data.unsqueeze(-1), 1, 0)
    ids = ids.movedim(-1, axis)
    return ids
