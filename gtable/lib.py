import numpy as np
import pandas as pd
from .fast import gen_column_filter


def fillna_column(values, index, reverse=False, fillvalue=None):
    """
    Fills the non available value sequentially with the previous
    available position.
    """
    # Create the new values array
    new_index = np.ones(index.shape, dtype=np.uint8)

    # Compute the indices where values has to be fetched
    if not reverse:
        indices = np.cumsum(index.astype(np.int8)) - np.array(1)
        if fillvalue is not None:
            new_values = np.empty(indices.shape, values.dtype)
            new_values[np.where(indices >= 0)] = values[
                indices[np.where(indices >= 0)]]
            new_values[np.where(indices < 0)] = fillvalue

        else:
            new_values = values[indices[np.where(indices >= 0)]]

    else:
        indices = np.cumsum(index.astype(np.int8)[::-1]) - np.array(1)
        if fillvalue is not None:
            new_values = np.empty(indices.shape, values.dtype)
            new_values[np.where(indices >= 0)] = values[::-1][
                indices[np.where(indices >= 0)]]
            new_values[np.where(indices < 0)] = fillvalue

        else:
            new_values = values[::-1][indices[np.where(indices >= 0)]]

        new_values = new_values[::-1]

    if fillvalue is None:
        new_index[np.where(indices < 0)] = 0

    if reverse:
        new_index = new_index[::-1]

    return new_values, new_index


def merge_table(table_left, table_right, column):
    """
    Merge two tables using a column as index. The order of the resulting
    table is predictable if the column in both tables is sorted.
    """
    if column not in table_left.keys:
        raise ValueError('{} not in left table'.format(column))

    if column not in table_right.keys:
        raise ValueError('{} not in right table'.format(column))

    left_key = table_left.keys.index(column)
    right_key = table_right.keys.index(column)
    left_data = table_left[column]
    right_data = table_right[column]
    left_index = table_left.index[left_key, :]
    right_index = table_right.index[right_key, :]

    sorter = np.argsort(left_data)

    left_length = len(left_index)
    right_length = len(right_index)

    if left_length != len(left_data):
        raise ValueError('Merge only with a dense column')
    if right_length != len(right_data):
        raise ValueError('Merge only with a dense column')

    insertions = np.searchsorted(left_data, right_data, sorter=sorter)
    all_columns = set(table_left.keys) | set(table_right.keys)

    width = len(all_columns)
    length = left_length + right_length

    new_index = np.empty((width, length), dtype=np.uint8)
    new_data = list()
    new_keys = list()

    for i, column in enumerate(all_columns):
        if column in table_left.keys and column in table_right.keys:
            new_index[i, :] = np.ones(length, dtype=np.uint8)

            # If the column is present in both tables, data must be merged
            left_key = table_left.keys.index(column)
            right_key = table_right.keys.index(column)
            left_orig_data = table_left.data[left_key]
            right_orig_data = table_right.data[right_key]
            merged = np.insert(left_orig_data, insertions, right_orig_data)

            new_data.append(merged)
            new_keys.append(column)

        elif column in table_left.keys:
            left_key = table_left.keys.index(column)
            new_index[i, :] = np.insert(
                table_left.index[left_key, :], insertions,
                np.zeros(len(insertions), dtype=np.uint8))
            new_data.append(table_left.data[left_key])
            new_keys.append(column)

        else:
            right_key = table_right.keys.index(column)
            new_index[i, :] = np.insert(
                np.zeros(left_length, dtype=np.uint8), insertions,
                np.ones(len(insertions), dtype=np.uint8))
            new_data.append(table_right.data[right_key])
            new_keys.append(column)

    return new_data, new_keys, new_index


def sort_table(table, column):
    """
    Sort the table inplace according to the elements of a column.
    If the column does not span the entire table, the rows
    not affected are left untouched.
    """
    if column not in table.keys:
        raise ValueError('{} not in table'.format(column))

    length = table.index.shape[1]
    table_index = np.arange(length)
    column_pos = table.keys.index(column)
    column_index = table.index[column_pos, :]
    filtered_index = table_index[column_index.astype(np.bool)]
    sorted_subindex = np.argsort(table.data[column_pos])
    table_index[filtered_index] = sorted_subindex

    # Now, with the complete table index, sort all the columns.
    for idx in range(len(table.keys)):
        column_index = table.index[idx, :]
        column_data = table.data[idx]
        if len(column_data) == len(table_index):
            table.data[idx] = column_data[table_index]
        else:
            # Argsort twice gives the sorted index for a subset of
            # a column.
            new_indexer = np.argsort(
                np.argsort(table_index[column_index.astype(np.bool)])
            )
            table.data[idx] = column_data[new_indexer]
        table.index[idx, :] = table.index[idx][table_index]


def records(table, fill=False):
    """
    Generator. Returns a dictionary for every row of the table.
    
    :param table: a Table.
    :param fill: True if empty values have to be replaced with NaN
    :return: Generator with each record as a dictionary
    """
    counters = np.zeros((table.index.shape[0]), dtype=np.int)
    keys = np.array(table.keys)

    for record in table.index.T:
        selected_keys = keys[np.where(record)]
        selected_counters = counters[np.where(record)]
        if fill:
            record_data = {k: table.data[table.keys.index(k)][c]
                           for k, c in zip(selected_keys, selected_counters)}
            record_data.update({k: np.nan for k in keys
                                if k not in selected_keys})
        else:
            record_data = {k: table.data[table.keys.index(k)][c]
                           for k, c in zip(selected_keys, selected_counters)}
        counters[np.where(record)] += 1

        yield record_data


def first_record(table, fill=False):
    """
    Return the first record of the table

    :param table: a Table.
    :param fill: True if empty values have to be replaced with NaN
    :return: Generator with each record as a dictionary
    """
    keys = np.array(table.keys)
    record = table.index[:, 0]

    selected_keys = keys[np.where(record)]

    if fill:
        record_data = {
            k: table.data[table.keys.index(k)][0] for k in selected_keys}
        record_data.update(
            {k: np.nan for k in keys if k not in selected_keys})
    else:
        record_data = {
            k: table.data[table.keys.index(k)][0] for k in selected_keys}

    return record_data


def last_record(table, fill=False):
    """
    Return the last record of the table

    :param table: a Table.
    :param fill: True if empty values have to be replaced with NaN
    :return: Generator with each record as a dictionary
    """
    keys = np.array(table.keys)
    record = table.index[:, -1]

    selected_keys = keys[np.where(record)]

    if fill:
        record_data = {
            k: table.data[table.keys.index(k)][0] for k in selected_keys}
        record_data.update(
            {k: np.nan for k in keys if k not in selected_keys})
    else:
        record_data = {
            k: table.data[table.keys.index(k)][0] for k in selected_keys}

    return record_data
        

def stack_table_inplace(left_table, right_table):
    """
    Stack a the right table to the bottom of the left table. Modifies
    the left_table inplace.
    
    :param left_table: 
    :param right_table: 
    :return: 
    """
    # First step is to rearrange the bitmap index if needed
    joined_columns = set(left_table.keys) | set(right_table.keys)
    hspill = len(joined_columns) - left_table.index.shape[0]
    before_growth = left_table.index.shape

    tindex = right_table.index.copy()

    # Add the horizontal spill (more columns)
    if joined_columns != set(left_table.keys):
        left_table.index = np.concatenate(
            [left_table.index, np.zeros(
                (hspill, left_table.index.shape[1]), dtype=np.uint8
            )])

    # Add the vertical spill (the data)
    left_table.index = np.concatenate(
        [left_table.index, np.zeros(
            (left_table.index.shape[0], tindex.shape[1]), dtype=np.uint8
        )], axis=1)

    # Include the keys present in both tables with this light nested loop.
    for old_key in left_table.keys:
        for new_key in right_table.keys:
            if new_key == old_key:
                right_table_index = right_table.keys.index(new_key)
                left_table_index = left_table.keys.index(old_key)
                left_table.index[left_table.keys.index(old_key),
                before_growth[1]:] = tindex[right_table_index, :]
                left_table.data[left_table.keys.index(old_key)] = np.concatenate(
                    [left_table.data[left_table_index],
                     right_table.data[right_table_index]]
                )

    # Include keys that are not added in the previous table
    new_cols_added = 0
    for new_key in right_table.keys:
        if new_key not in left_table.keys:
            new_index = right_table.keys.index(new_key)
            left_table.index[before_growth[0] + new_cols_added,
                             before_growth[1]:] = tindex[new_index, :]
            left_table.data.append(right_table.data[new_index])
            left_table.keys.append(new_key)
            new_cols_added += 1


def stack_table(left_table, right_table):
    """
    Stack a the right table to the bottom of the left table. Modifies
    the left_table inplace.

    :param left_table:
    :param right_table:
    :return:
    """
    # First step is to rearrange the bitmap index if needed
    joined_columns = set(left_table.keys) | set(right_table.keys)
    hspill = len(joined_columns) - left_table.index.shape[0]
    before_growth = left_table.index.shape

    tindex = right_table.index.copy()

    # Add the horizontal spill (more columns)
    if joined_columns != set(left_table.keys):
        left_table.index = np.concatenate(
            [left_table.index, np.zeros(
                (hspill, left_table.index.shape[1]), dtype=np.uint8
            )])

    # Add the vertical spill (the data)
    left_table.index = np.concatenate(
        [left_table.index, np.zeros(
            (left_table.index.shape[0], tindex.shape[1]), dtype=np.uint8
        )], axis=1)

    # Include the keys present in both tables with this light nested loop.
    for old_key in left_table.keys:
        for new_key in right_table.keys:
            if new_key == old_key:
                right_table_index = right_table.keys.index(new_key)
                left_table_index = left_table.keys.index(old_key)
                left_table.index[left_table.keys.index(old_key),
                before_growth[1]:] = tindex[right_table_index, :]
                left_table.data[
                    left_table.keys.index(old_key)] = np.concatenate(
                    [left_table.data[left_table_index],
                     right_table.data[right_table_index]]
                )

    # Include keys that are not added in the previous table
    new_cols_added = 0
    for new_key in right_table.keys:
        if new_key not in left_table.keys:
            new_index = right_table.keys.index(new_key)
            left_table.index[before_growth[0] + new_cols_added,
            before_growth[1]:] = tindex[new_index, :]
            left_table.data.append(right_table.data[new_index])
            left_table.keys.append(new_key)
            new_cols_added += 1


def from_chunks(tables):
    """
    Create a table from chunks

    :param tables: Iterable of tables.
    :return:
    """
    for i, table in enumerate(tables):
        if i == 0:
            # Some things to do with the first table
            result = table.copy()

        else:
            result.index = np.hstack([result.index, table.index])
            for rk, tk in zip(result.keys, table.keys):
                result[rk] = np.hstack([result[rk], table[tk]])

    return result


def add_column(table, k, v, dtype=None, index=None, align='top'):
    """
    Adds a column to a table inplace
    
    :param table: 
    :param k: 
    :param v:
    :param dtype:
    :param index:
    :param align:
    :return: 
    """
    if k in table.keys:
        raise KeyError("Key {} already present".format(k))

    if type(v) == list:
        # TODO: Remove ASAP
        # You may get a list of Timestamps. Specific to NFQ
        if len(v) > 0 and type(v[0]) == pd.Timestamp:
            table.data.append(pd.DatetimeIndex(v).values)
        else:
            if dtype is None:
                # Infer the dtype of the list
                table.data.append(np.array(v))
            else:
                table.data.append(np.array(v, dtype=dtype))
        table.keys.append(k)

    elif type(v) == np.ndarray:
        if not len(v.shape) == 1:
            raise ValueError("Only 1D arrays supported")
        table.data.append(v)
        table.keys.append(k)

    elif type(v) == pd.DatetimeIndex:
        table.data.append(np.array(v))
        table.keys.append(k)

    else:
        raise ValueError("Column type not supported")

    if index is None:
        if len(v) > table.index.shape[1]:
            table.index = np.concatenate(
                [table.index,
                 np.zeros(
                     (table.index.shape[0],
                      len(v) - table.index.shape[1]),
                     dtype=np.uint8)],
                axis=1
            )

        # Concatenate the shape of the array to the bitmap
        index_stride = np.zeros((1, table.index.shape[1]), dtype=np.uint8)
        if align == 'top':
            index_stride[0, :len(v)] = 1
        elif align == 'bottom':
            index_stride[0, -len(v):] = 1
        else:
            raise ValueError('Alignment can be either "top" or "bottom"')

        table.index = np.concatenate([table.index, index_stride])

    else:
        # Handle the fact that the new column my be longer, so extend bitmap
        if index.shape[0] > table.index.shape[1]:
            table.index = np.concatenate(
                [table.index,
                 np.zeros(
                     (table.index.shape[0],
                      index.shape[0] - table.index.shape[0]),
                     dtype=np.uint8)],
                axis=1
            )

        # Concatenate the new column to the bitmap.
        table.index = np.concatenate([table.index, np.atleast_2d(index)])


def filter_table(table, predicate):
    new_keys = table.keys
    new_data = list()
    new_index = list()

    # Now for the values
    for column, index in zip(table.data, table.index):
        data_filter, new_col_index = gen_column_filter(
            predicate.values, predicate.index, index
        )
        new_index.append(new_col_index)
        new_data.append(column[data_filter])

    return new_data, new_keys, np.vstack(new_index)


def dropnan_table(table):
    for column in table.keys:
        isnan = np.isnan(table[column].astype(np.float))
        # Drop in data
        table[column] = table[column][~isnan]
        # Get index column
        index = table.index[table.keys.index(column), :]
        # Drop from index
        enumerator = (np.cumsum(index) - np.array(1))[np.where(isnan)]
        index[enumerator] = 0


def required_column(table, key, dtype):
    """Adds a column inplace with the given name and type"""
    table.keys.append(key)
    table.data.append(np.array([], dtype=dtype))
    table.index = np.vstack((table.index, np.zeros((1, table.index.shape[1]), dtype=np.uint8)))


def required_columns(table, *args):
    """Adds the required columns inplace"""
    new_cols = set(args) - set(table.keys)
    length = table.index.shape[1]

    for col in new_cols:
        table.keys.append(col)
        table.data.append(np.array([]))
        table.index = np.vstack(
            (table.index, np.zeros((len(new_cols), length), dtype=np.uint8))
        )

