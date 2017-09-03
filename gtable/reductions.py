import numpy as np
from functools import partial
from .fast import reduce_funcs
from .table import Table


def reduce_by_key(table, column_name, func, check_sorted=True):
    """
    Func is a string

    :param table:
    :param column_name:
    :param func:
    :param check_sorted:
    :return:
    """
    key_data = table[column_name]
    key_index = table._index_column(column_name)

    if func not in reduce_funcs:
        raise ValueError('Reduction not available')

    if check_sorted:
        if not np.all(key_data == np.sort(key_data)):
            raise ValueError('You can only reduce from a sorted column')

    unique_keys = np.unique(key_data)
    other_cols = set(table.keys) - {column_name}
    new_data = [unique_keys]
    new_index = [np.ones(len(unique_keys), dtype=np.uint8)]
    new_keys = [column_name]

    # Check if key data is a string, and substitute with an equivalent integer
    if key_data.dtype.kind in {'S', 'U'}:
        key_data = np.searchsorted(key_data, key_data)

    for col in other_cols:
        col_data = table[col]
        col_index = table._index_column(col)

        reduced_col_data, reduced_col_index = reduce_funcs[func](
            key_data, key_index, col_data, col_index, len(unique_keys)
        )
        new_data.append(reduced_col_data)
        new_index.append(reduced_col_index)
        new_keys.append(col)

    t = Table()
    t.data = new_data
    t.index = np.vstack(new_index)
    t.keys = new_keys

    return t


class ReductorByKey:
    def __init__(self, table, column, check_sorted=False):
        for reduction_f in reduce_funcs:
            self.__dict__[reduction_f] = partial(
                reduce_by_key, table, column, reduction_f, check_sorted)
