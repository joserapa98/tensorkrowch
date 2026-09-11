"""
This script contains:

    * print_list
    * tab_string
    * check_name_style
    * erase_enum
    * enum_repeated_names
    * permute_list
    * is_permutation
    * inverse_permutation
    * fact
    * binomial_coeffs
    * stack_unequal_tensors
    * list2slice
    * split_sequence_into_regions
    * random_unitary
    * truncated_svd
"""

from math import isfinite
from typing import NamedTuple, Optional, Tuple, List, Sequence, Text, Union

import torch
import torch.nn as nn
from torch import Tensor

from tensorkrowch.config import _validate_svd_method, get_svd_method


def print_list(lst: List) -> Text:
    return '[' + '\n '.join(f'{item}' for item in lst) + ']'


def tab_string(string: Text, num_tabs: int = 1) -> Text:
    """
    Introduces '\t' a certain amount of times before each line.

    Parameters
    ----------
    string : str
        Text to be displaced.
    num_tabs : int
        Number of '\t' introduced.
        
    Returns
    -------
    str
    """
    string_lst = string.split('\n')
    string_lst = list(map(lambda x: num_tabs * '\t' + x, string_lst))
    displaced_string = '\n'.join(string_lst)
    return displaced_string


def check_name_style(name: Text, type: Text = 'axis') -> bool:
    """
    Axis' names can only contain letters, numbers and underscores. Nodes' names
    cannot contain blank spaces.
    """
    for char in name:
        if type == 'axis':
            if (not char.isalpha()) and (not char.isnumeric()) and (char != '_'):
                return False
        elif type == 'node':
            if char == ' ':
                return False
        else:
            raise ValueError('`type` can only be "axis" or "node"')
    return True


def erase_enum(name: Text) -> Text:
    """
    Given a name, returns the same name without any enumeration suffix with
    format ``_{digit}``.
    """
    name_list = name.split('_')
    i = len(name_list) - 1
    while i >= 0:
        if name_list[i].isdigit():
            i -= 1
        else:
            break
    new_name = '_'.join(name_list[:i + 1])
    return new_name


def enum_repeated_names(names_list: List[Text]) -> List[Text]:
    """
    Given a list of (axes or nodes) names, returns the same list but adding
    an enumeration for the names that appear more than once in the list.
    """
    counts = dict()
    aux_list = []
    for name in names_list:
        name = erase_enum(name)
        aux_list.append(name)
        if name in counts:
            counts[name] += 1
        else:
            counts[name] = 0

    for name in counts:
        if counts[name] == 0:
            counts[name] = -1

    aux_list.reverse()
    for i, name in enumerate(aux_list):
        if counts[name] >= 0:
            aux_list[i] = f'{name}_{counts[name]}'
            counts[name] -= 1
    aux_list.reverse()
    return aux_list


def permute_list(lst: List, dims: Sequence[int]) -> List:
    """
    Permutes elements of list based on a permutation of indices. It is not
    required that ``lst`` and ``dims`` have the same length; in such case the
    returned list will only have as many elements as indices specified in
    ``dims``, in the corresponding order.

    Parameters
    ----------
    lst : list
        List to be permuted.
    dims : list[int]
        List of dimensions (indices) in the new order.
    """
    new_lst = []
    for i in dims:
        if i >= len(lst):
            raise IndexError(f'Index out of bounds. `dims` given to permute '
                             f'`lst` according to contains index {i}, which '
                             'exceeds length of `lst`')
        new_lst.append(lst[i])
    return new_lst


def is_permutation(lst: List, permuted_lst: List) -> bool:
    """
    Indicates if ``permuted_lst`` is a permutation of the elements of ``lst``.
    """
    if len(lst) != len(permuted_lst):
        return False
    aux_lst = lst[:]
    for el in permuted_lst:
        if el not in aux_lst:
            return False
        aux_lst.remove(el)
    return True


def inverse_permutation(dims: Sequence[int]):
    """
    Given a permutation of indices (to permute the elements of a list, tensor,
    etc.), returns the inverse permutation of indices needed to recover the
    original object (in the original order).

    Parameters
    ----------
    dims: list[int]
        Permutation of indices. It can be complete if all numbers in
        range(len(dims)) appear (e.g. (2, 0, 1) -> (1, 2, 0)), or incomplete
        if after permutation some elements were removed (e.g. (3, 0, 2) ->
        (1, 2, 0), removed element in position 1).
    """
    if dims:
        inverse_dims = [-1] * (max(dims) + 1)
        for i, j in enumerate(dims):
            inverse_dims[j] = i
        return list(filter(lambda x: x != -1, inverse_dims))
    return []


def fact(n: int) -> int:
    """Returns factorial of ``n``."""
    if n < 0:
        raise ValueError('Argument should be greater than zero')
    if n == 0:
        return 1
    return n * (fact(n - 1))


def binomial_coeffs(n: int, k: int) -> int:
    """Returns binomiaal coefficients (``n`` choose ``k``)."""
    return fact(n) // (fact(k) * fact(n - k))


def stack_unequal_tensors(lst_tensors: List[torch.Tensor]) -> torch.Tensor:
    """
    Stacks a list of tensors. These tensors need not have equal sizes in each
    dimension, but they must have the same rank.
    
    The smallest tensors are extended with zeros to match the shape of the
    biggest ones.

    Parameters
    ----------
    lst_tensors : list[torch.Tensor]
        List of tensors to be stacked

    Returns
    -------
    torch.Tensor
    """
    # To protect the original list
    lst_tensors = lst_tensors[:]
    if lst_tensors:
        same_dims = True
        max_shape = list(lst_tensors[0].shape)
        for tensor in lst_tensors[1:]:
            for idx, dim in enumerate(tensor.shape):
                if same_dims and (dim != max_shape[idx]):
                    same_dims = False
                if dim > max_shape[idx]:
                    max_shape[idx] = dim

        if not same_dims:
            for idx, tensor in enumerate(lst_tensors):
                if tensor.shape != max_shape:
                    pad = []
                    for max_dim, dim in zip(max_shape, tensor.shape):
                        pad += [0, max_dim - dim]
                    pad.reverse()
                    lst_tensors[idx] = nn.functional.pad(tensor, pad)
                    # NOTE: nn.functional.pad induces non-deterministic
                    # behaviour in its backward pass on CUDA
        return torch.stack(lst_tensors)


def list2slice(lst: List) -> Union[List, slice]:
    """
    Given a list (of indices) returns, if possible, an object ``slice``
    containing the same indices.
    """
    aux_slice = [None, None, None]
    use_slice = False

    if len(lst) >= 1:
        use_slice = True

        for el in lst:
            if aux_slice[0] is None:
                aux_slice[0] = el
                aux_slice[1] = el
            elif aux_slice[2] is None:
                aux_slice[1] = el
                aux_slice[2] = aux_slice[1] - aux_slice[0]
            else:
                if (el - aux_slice[1]) == aux_slice[2]:
                    aux_slice[1] = el
                else:
                    use_slice = False
                    break

    if use_slice:
        aux_slice[1] += 1
        return slice(*aux_slice)
    return lst


def split_sequence_into_regions(lst: Sequence[int]) -> List[List[int]]:
    """
    Splits a sequence of integers into regions where each region contains
    consecutive integers.

    Parameters
    ----------
    lst : list[int] or tuple[int]
        List of integers in ascending order.

    Returns
    -------
    list[list[int]]

    Raises
    ------
    TypeError
        If the input is not a sequence of integers.
    ValueError
        If the input sequence is not ordered.

    Example
    -------
    >>> sequence = [1, 2, 3, 5, 6, 7, 10, 11, 13]
    >>> split_sequence_into_regions(sequence)
    [[1, 2, 3], [5, 6, 7], [10, 11], [13]]
    """
    if not isinstance(lst, Sequence) or not all(isinstance(x, int) for x in lst):   #TODO: use this in my code
        raise TypeError('Input must be a sequence of integers')
    
    if len(lst) != len(set(lst)):
        raise ValueError('Input sequence cannot contain repeated elements')

    if any(lst[i + 1] < lst[i] for i in range(len(lst) - 1)):
        raise ValueError('Input sequence must be in ascending order')

    if not lst:
        return []

    regions = []
    current_region = [lst[0]]

    for i in range(1, len(lst)):
        if lst[i] == lst[i - 1] + 1:
            current_region.append(lst[i])
        else:
            regions.append(current_region)
            current_region = [lst[i]]

    regions.append(current_region)
    return regions


def random_unitary(n,
                   device: Optional[torch.device] = None,
                   dtype: Optional[torch.dtype] = None,
                   generator: Optional[torch.Generator] = None):
    """
    Returns random unitary matrix from the Haar measure of size n x n.
    
    Unitary matrix is created as described in this `paper
    <https://arxiv.org/abs/math-ph/0609050v2>`_.

    ``generator`` controls the Gaussian matrix without modifying the global
    random state and should belong to the requested ``device``.
    """
    mat = torch.randn(
        n, n, device=device, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(mat)
    d = torch.diagonal(r)
    ph = d / d.abs()
    q = q @ torch.diag(ph)
    return q


class _TruncatedSVDInfo(NamedTuple):
    """Numerical diagnostics from one call to :func:`truncated_svd`."""

    full_rank: int
    selected_rank: int
    total_squared_norm: Tensor
    discarded_squared_norm: Tensor
    total_squared_norm_per_batch: Tensor
    discarded_squared_norm_per_batch: Tensor
    svd_method: Text


def _validate_truncation(rank: Optional[int] = None,
                         cutoff: Optional[float] = None,
                         atol: Optional[float] = None,
                         rtol: Optional[float] = None,
                         cum_percentage: Optional[float] = None) -> None:
    """Validates the truncation contract shared by SVD-based methods."""
    if rank is not None:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError('`rank` should be int type')
        if rank < 1:
            raise ValueError('`rank` should be a positive integer')

    for name, value in (('cutoff', cutoff), ('atol', atol)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'`{name}` should be a real number')
        if (value < 0) or not isfinite(value):
            raise ValueError(
                f'`{name}` should be a finite non-negative number')

    for name, value in (('rtol', rtol),
                        ('cum_percentage', cum_percentage)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'`{name}` should be a real number')
        if (value < 0) or (value > 1) or not isfinite(value):
            raise ValueError(
                f'`{name}` should be a finite number between 0 and 1')


def _compact_svd(tensor: Tensor,
                 svd_method: Text) -> Tuple[Tensor, Tensor, Tensor]:
    """Computes an exact economy-size SVD with an already-resolved backend."""
    if not isinstance(tensor, Tensor):
        raise TypeError('`tensor` should be torch.Tensor type')
    if tensor.ndim < 2:
        # Preserve the exception type and message of the historical direct
        # SVD path instead of defining a second dimensionality contract here.
        return torch.linalg.svd(tensor, full_matrices=False)

    if svd_method == 'svd':
        return torch.linalg.svd(tensor, full_matrices=False)

    if tensor.shape[-2] >= tensor.shape[-1]:
        q, r = torch.linalg.qr(tensor, mode='reduced')
        u_r, s, vh = torch.linalg.svd(r, full_matrices=False)
        u = q @ u_r
        return u, s, vh

    tensor_h = tensor.transpose(-2, -1).conj()
    q, r = torch.linalg.qr(tensor_h, mode='reduced')
    r_h = r.transpose(-2, -1).conj()
    u, s, vh_r = torch.linalg.svd(r_h, full_matrices=False)
    q_h = q.transpose(-2, -1).conj()
    vh = vh_r @ q_h
    return u, s, vh


def truncated_svd(tensor: Tensor,
                  rank: Optional[int] = None,
                  cutoff: Optional[float] = None,
                  atol: Optional[float] = None,
                  rtol: Optional[float] = None,
                  cum_percentage: Optional[float] = None,
                  svd_method: Optional[Text] = None,
                  return_info: bool = False) -> Union[
                      Tuple[Tensor, Tensor, Tensor],
                      Tuple[Tensor, Tensor, Tensor, _TruncatedSVDInfo]]:
    r"""
    Computes a truncated SVD. If no truncation criterion is specified, it
    returns the full SVD. If more than one criterion is specified, the final
    rank is the minimum one, i.e. the one imposed by the most restrictive
    criterion.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be decomposed, with shape (*, m, n) where * is zero or more
        batch dimensions.
    rank : int, optional
        Maximum number of singular values to keep.
    cutoff : float, optional
        Minimum singular value to keep. It must be finite and non-negative.
        Singular values ``<= cutoff`` are removed.
    atol : float, optional
        Absolute tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the accumulated sum of squares is ``<= atol``. It must be finite and
        non-negative.
    rtol : float, optional
        Relative tolerance over the tail sum of squared singular values.
        Starting from the smallest singular value, values are discarded while
        the tail sum of squares divided by the total sum of squares is
        ``<= rtol``. It must be finite and in ``[0, 1]``.
    cum_percentage : float, optional
        Minimum fraction of squared singular-value mass to keep. Equivalent to
        setting ``rtol = 1 - cum_percentage``. It must be finite and in
        ``[0, 1]``.

        .. math::

            \frac{\sum_{i \in \{kept\}}{s_i^2}}{\sum_{i \in \{all\}}{s_i^2}} \ge
            cum\_percentage

    svd_method : {"svd", "qr_svd"}, optional
        Exact backend used to compute the economy-size SVD. ``"svd"`` calls
        :func:`torch.linalg.svd` directly. ``"qr_svd"`` first reduces the
        larger matrix dimension with QR. If omitted, it uses the active value
        from :func:`~tensorkrowch.get_svd_method`, whose initial default is
        ``"svd"``. The active backend can be changed globally with
        :func:`~tensorkrowch.set_svd_method`, or temporarily with the
        :func:`~tensorkrowch.svd_method` context manager. These configuration
        mechanisms also select the backend for higher-level methods that call
        :func:`truncated_svd` internally.

        .. note::

            Backward through ``"qr_svd"`` follows the differentiability
            requirements of :func:`torch.linalg.qr`: the input needs full
            column rank in the tall case and full row rank in the wide case.

    return_info : bool
        If ``True``, also returns ``_TruncatedSVDInfo`` with the full and
        selected ranks, the total and discarded squared norms, their per-batch
        values and the effective SVD backend. The complete singular value
        spectrum is not retained in this record.

    Returns
    -------
    tuple
        By default, returns ``(u, s, vh)`` from an exact economy-size SVD after
        truncation. ``u`` has shape ``(*, m, r)``, ``s`` has shape ``(*, r)``,
        and ``vh`` has shape ``(*, r, n)``, where ``r`` is the selected final
        rank. If ``return_info=True``, returns ``(u, s, vh, info)``.

    Raises
    ------
    TypeError
        If ``tensor`` is not a :class:`torch.Tensor` or ``svd_method`` is not
        a string, if ``rank`` is not an integer, if a tolerance is not a real
        number, or if ``return_info`` is not boolean.
    ValueError
        If ``rank`` is not positive, a tolerance is not finite or lies outside
        its accepted interval, or ``svd_method`` is not accepted.
    RuntimeError
        If ``tensor`` has fewer than two dimensions, as raised by
        :func:`torch.linalg.svd`.
    
    Examples
    --------
    >>> tensor = torch.randn(4, 4)
    >>> u, s, vh = truncated_svd(tensor, rank=2)
    >>> len(s)
    2
    """
    if not isinstance(return_info, bool):
        raise TypeError('`return_info` should be bool type')
    _validate_truncation(
        rank=rank,
        cutoff=cutoff,
        atol=atol,
        rtol=rtol,
        cum_percentage=cum_percentage)
    
    if cum_percentage is not None:
        if rtol is None:
            rtol = 1 - cum_percentage
        else:
            rtol = max(rtol, 1 - cum_percentage)

    if svd_method is None:
        effective_svd_method = get_svd_method()
    else:
        effective_svd_method = _validate_svd_method(svd_method)

    u, s, vh = _compact_svd(
        tensor=tensor,
        svd_method=effective_svd_method)
    final_rank = s.shape[-1]
    
    if rank is not None:
        final_rank = min(final_rank, rank)
    
    if cutoff is not None:
        co_rank = (s > cutoff).reshape(-1, s.shape[-1]).any(dim=0).sum()
        final_rank = min(final_rank, max(1, co_rank.item()))

    squared_s = None
    tail_squared_norm = None
    if (atol is not None) or (rtol is not None) or return_info:
        squared_s = s.square()
    if (atol is not None) or (rtol is not None):
        tail_squared_norm = squared_s.flip(dims=[-1]).cumsum(-1)

    if atol is not None:
        atol_rank = (tail_squared_norm > atol).reshape(
            -1, s.shape[-1]).any(dim=0).sum()
        final_rank = min(final_rank, max(1, atol_rank.item()))

    if rtol is not None:
        total_squared_norm = squared_s.sum(-1, keepdim=True)
        positive_norm = total_squared_norm > 0
        safe_squared_norm = torch.where(
            positive_norm,
            total_squared_norm,
            torch.ones_like(total_squared_norm))
        tail_ratios = tail_squared_norm / safe_squared_norm
        rtol_rank = (tail_ratios > rtol).reshape(
            -1, s.shape[-1]).any(dim=0).sum()
        final_rank = min(final_rank, max(1, rtol_rank.item()))
    
    if return_info:
        total_squared_norm_per_batch = squared_s.sum(dim=-1)
        discarded_squared_norm_per_batch = squared_s[..., final_rank:].sum(
            dim=-1)
        info = _TruncatedSVDInfo(
            full_rank=s.shape[-1],
            selected_rank=final_rank,
            total_squared_norm=total_squared_norm_per_batch.sum(),
            discarded_squared_norm=discarded_squared_norm_per_batch.sum(),
            total_squared_norm_per_batch=total_squared_norm_per_batch,
            discarded_squared_norm_per_batch=(
                discarded_squared_norm_per_batch),
            svd_method=effective_svd_method)

    u = u[..., :final_rank]
    s = s[..., :final_rank]
    vh = vh[..., :final_rank, :]
    
    if return_info:
        return u, s, vh, info
    return u, s, vh
