
from typing import List, Sequence, Text


def tab_string(string: Text, num_tabs: int = 1) -> Text:
    """
    Introduce '\t' a certain amount of times before each line.

    Parameters
    ----------
    string: text to be displaced
    num_tabs: number of '\t' introduced
    """
    string_lst = string.split('\n')
    string_lst = list(map(lambda x: num_tabs * '\t' + x, string_lst))
    displaced_string = '\n'.join(string_lst)
    return displaced_string


def check_name_style(name: Text) -> bool:
    """
    Names can only contain letters, numbers and underscores.
    """
    for char in name:
        if (not char.isalpha()) and (not char.isnumeric()) and (char != '_'):
            return False
    return True


def erase_enum(name: Text) -> Text:
    """
    Given a name, returns the same name without any
    enumeration suffix with format `_{digit}`.
    """
    name_list = name.split('_')
    i = len(name_list) - 1
    while i >= 0:
        if name_list[i].isdigit():
            i -= 1
        else:
            break
    new_name = '_'.join(name_list[:i+1])
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
    Permute elements of list based on a permutation of indices.

    Parameters
    ----------
    lst: list to be permuted
    dims: list of dimensions (indices) in the new order
    """
    if len(dims) != len(lst):
        raise ValueError('Number of `dims` must match number of elements in `lst`')
    new_lst = []
    for i in dims:
        new_lst.append(lst[i])
    return new_lst


def is_permutation(lst: List, permuted_lst: List) -> bool:
    """
    Decide whether `permuted_lst` is a permutation of the elements of `lst`
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
    Given a permutation of indices (to permute the elements of a list, tensor, etc.),
    returns the inverse permutation of indices needed to recover the original object
    (in the original order).

    Parameters
    ----------
    dims: permutation of indices. It can be complete if all numbers in range(len(dims))
        appear in dims (e.g. (2, 0, 1) -> (1, 2, 0)), or incomplete if after permutation
        some elements were removed (e.g. (3, 0, 2) -> (1, 2, 0), removed element in position 1).
    """
    inverse_dims = [-1] * (max(dims) + 1)
    for i, j in enumerate(dims):
        inverse_dims[j] = i
    return list(filter(lambda x: x != -1, inverse_dims))


def fact(n: int) -> int:
    if n < 0:
        raise ValueError('Argument should be greater than zero')
    if n == 0:
        return 1
    return n * (fact(n - 1))


def comb_num(n: int, k: int) -> int:
    return fact(n) // (fact(k) * fact(n - k))
