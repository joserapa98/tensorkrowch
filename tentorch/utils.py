# TODO: functions to transform between tensornetwork, tntorch and tentorch
# TODO: functions to find contraction paths using tensornetwork
from typing import List, Text


def tab_string(string: Text, num_tabs: int = 1) -> Text:
    string_lst = string.split('\n')
    string_lst = list(map(lambda x: num_tabs * '\t' + x, string_lst))
    return '\n'.join(string_lst)


def erase_enum(name: Text) -> Text:
    """
    Given a name, returns the same name without any
    enumeration suffix (like `_{digit}`)
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
    Given a list of (axes, nodes) names, returns the same list but adding
    an enumeration for the names that appear more than once in the list
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
