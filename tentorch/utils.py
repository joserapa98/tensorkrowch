# TODO: functions to transform between tensornetwork, tntorch and tentorch
# TODO: functions to find contraction paths using tensornetwork
from typing import List, Text


def tab_string(string: Text, num_tabs: int = 1) -> Text:
    string_lst = string.split('\n')
    string_lst = list(map(lambda x: num_tabs*'\t' + x, string_lst))
    return '\n'.join(string_lst)


def enum_repeated_names(names_list: List[Text]) -> List[Text]:
    """
    Given a list of (axes, nodes) names, returns the same list but adding
    a enumerations for the names that appear more than once in the list
    """
    counts = dict()
    for name in names_list:
        if name in counts:
            counts[name] += 1
        else:
            counts[name] = 0

    for name in counts:
        if counts[name] == 0:
            counts[name] = -1

    names_list.reverse()
    for i, name in enumerate(names_list):
        if counts[name] >= 0:
            names_list[i] = f'{name}_{counts[name]}'
            counts[name] -= 1
    names_list.reverse()
    return names_list
