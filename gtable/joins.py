from gtable import Table
import numpy as np
from itertools import chain
from gtable.fast import join_low_level


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

    if not np.all(common_left.values == np.sort(common_left.values)):
        raise ValueError('Trying to join with a non sorted column')

    if not np.all(common_right.values == np.sort(common_right.values)):
        raise ValueError('Trying to join with a non sorted column')

    intersection = np.intersect1d(common_left.values, common_right.values)
    data_filter_left = np.in1d(common_left.values, intersection)
    data_filter_right = np.in1d(common_right.values, intersection)

    common_left = common_left.mask(data_filter_left)
    common_right = common_right.mask(data_filter_right)

    data_joined, global_left, global_right = join_low_level(
        common_left.values, common_left.index,
        common_right.values, common_right.index)

    data = list()
    index = list()
    keys = list()

    data.append(data_joined)
    index.append(np.ones(len(data_joined), dtype=np.uint8))
    keys.append(column)

    for i_column in joined_columns:
        if i_column in table_left:
            c = table_left.get(i_column)
            c = c.reindex(global_left)
            keys.append(i_column)
            data.append(c.values)
            index.append(c.index)

        elif i_column in table_right:
            c = table_right.get(i_column)
            c = c.reindex(global_right)
            keys.append(i_column)
            data.append(c.values)
            index.append(c.index)

    res = Table()
    res.data = data
    res.index = np.vstack(index)
    res.keys = keys

    return res
