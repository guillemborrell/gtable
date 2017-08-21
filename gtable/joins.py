from gtable import Table
import numpy as np
from itertools import chain
from gtable.fast import sieve_column_other, crop_column_other, sieve_column_own


def inner_join(table_left, table_right, column):
    """
    Inner join. If columns are repeated, the left table has preference.

    :param table_left:
    :param table_right:
    :param column:
    :return:
    """
    if column not in table_left.keys:
        raise ValueError('{} not in left table'.format(column))

    if column not in table_right.keys:
        raise ValueError('{} not in right table'.format(column))

    all_columns = set(chain(table_left.keys, table_right.keys))
    joined_columns = all_columns - set(column)

    common_left = table_left.get(column)
    common_right = table_right.get(column)

    intersection = np.sort(np.intersect1d(common_left.values,
                                          common_right.values))
    data_filter_left = np.in1d(common_left.values, intersection)
    data_filter_right = np.in1d(common_right.values, intersection)
    order_left = np.searchsorted(intersection,
                                 common_left.values[data_filter_left])
    order_right = np.searchsorted(intersection,
                                  common_right.values[data_filter_right])

    data = list()
    index = list()
    keys = list()

    _, index_left_sieved = sieve_column_own(common_left.values,
                                            common_left.index,
                                            data_filter_left)

    _, index_right_sieved = sieve_column_own(common_right.values,
                                             common_right.index,
                                             data_filter_right)

    data.append(intersection)
    index.append(np.ones(len(intersection), dtype=np.uint8))
    keys.append(column)

    for i_column in joined_columns:
        if i_column in table_left.keys:
            c = table_left.get(i_column)
            c.values, c.index = sieve_column_other(c.values,
                                                   c.index,
                                                   common_left.index,
                                                   data_filter_left)
            c.values, c.index = crop_column_other(c.values,
                                                  c.index,
                                                  index_left_sieved)
            c.reorder(order_left)
            data.append(c.values)
            index.append(c.index)

        elif column in table_right.keys:
            c = table_right.get(i_column)
            c.values, c.index = sieve_column_other(c.values,
                                                   c.index,
                                                   common_right.index,
                                                   data_filter_right)
            c.values, c.index = crop_column_other(c.values,
                                                  c.index,
                                                  index_right_sieved)
            c.reorder(order_right)
            data.append(c.values)
            index.append(c.index)

        keys.append(i_column)

    res = Table()
    res.data = data
    res.index = np.vstack(index)
    res.keys = keys

    return res
