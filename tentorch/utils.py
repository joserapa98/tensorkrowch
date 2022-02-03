# TODO: functions to transform between tensornetwork and tentorch
# TODO: functions to find contraction paths using tensornetwork
from typing import Text


def tab_string(string: Text, num_tabs: int = 1) -> Text:
    string_lst = string.split('\n')
    string_lst = list(map(lambda x: num_tabs*'\t' + x, string_lst))
    return '\n'.join(string_lst)
