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
"""

from typing import Optional, List, Sequence, Text, Union

import torch
import torch.nn as nn


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
            elif aux_slice[2] == None:
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
                   dtype: Optional[torch.dtype] = None):
    """
    Returns random unitary matrix from the Haar measure of size n x n.
    
    Unitary matrix is created as described in this `paper
    <https://arxiv.org/abs/math-ph/0609050v2>`_.
    """
    mat = torch.randn(n, n, device=device, dtype=dtype)
    q, r = torch.linalg.qr(mat)
    d = torch.diagonal(r)
    ph = d / d.abs()
    q = q @ torch.diag(ph)
    return q
