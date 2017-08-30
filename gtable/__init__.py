from .table import Table
from .column import Column
from .version import __version__
from .joins import inner_join, full_outer_join
from .lib import merge_table as _merge_table


def merge(table_left, table_right, key):
    """
    Merge two tables by key.

    :param table_left:
    :param table_right:
    :param key:
    :return:
    """
    t = Table()
    t.data, t.keys, t.index = _merge_table(table_right, table_left, key)

    return t
