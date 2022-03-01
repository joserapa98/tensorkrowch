
from typing import List, Text


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
    aux_name = ''.join(name.split('_'))
    for char in aux_name:
        if not (char.isalpha() or char.isnumeric()):
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
