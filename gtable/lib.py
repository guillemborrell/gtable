from itertools import chain
import numpy as np
import pandas as pd


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

    new_index[np.where(indices < 0)] = 0

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

    column_left = table_left[column]
    left_key = table_left.keys.index(column)
    column_left_index = table_left.index[left_key, :]
    column_right = table_right[column]
    right_key = table_right.keys.index(column)
    column_right_index = table_right.index[right_key, :]
    sorter = np.argsort(column_left)

    if len(column_left_index) != len(table_left[column]):
        raise ValueError('Merge only with a dense column')

    if len(column_right_index) != len(table_right[column]):
        raise ValueError('Merge only with a dense column')

    insertions = np.searchsorted(column_left, column_right, sorter=sorter)
    all_columns = set(chain(table_left.keys, table_right.keys))
    left_length = table_left.index.shape[1]
    width = len(all_columns)
    length = table_left.index.shape[1] + table_right.index.shape[1]
    new_index = np.empty((width, length), dtype=np.uint8)
    new_data = list()
    new_keys = list()

    for i, column in enumerate(all_columns):
        if column in table_left.keys and column in table_right.keys:
            left_index = table_left.index[table_left.keys.index(column), :]
            right_index = table_right.index[table_right.keys.index(column), :]
            # Merge indices
            merged_index = np.insert(left_index, insertions, right_index)
            new_index[i, :] = merged_index

            # If the column is present in both tables, data must be merged
            left_key = table_left.keys.index(column)
            right_key = table_right.keys.index(column)
            left_orig_data = table_left.data[left_key]
            right_orig_data = table_right.data[right_key]
            if len(insertions) == len(right_orig_data):
                merged = np.insert(left_orig_data, insertions, right_orig_data)
            else:
                merged = np.insert(
                    left_orig_data,
                    np.argsort(np.argsort(
                        insertions[right_index.astype(np.bool)])),
                    right_orig_data)

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


def records(table):
    """
    Generator. Returns a dictionary for every row of the table.
    
    :param table: a Table.
    :return: Generator with each record as a dictionary
    """
    counters = np.zeros((table.index.shape[0]), dtype=np.int)
    keys = np.array(table.keys)

    for record in table.index.T:
        selected_keys = keys[np.where(record)]
        selected_counters = counters[np.where(record)]
        selected_values = list()

        for k, c in zip(selected_keys, selected_counters):
            selected_values.append(table.data[table.keys.index(k)][c])
        counters[np.where(record)] += 1

        yield {k: v for k, v in zip(selected_keys, selected_values)}
        

def stitch_table(left_table, right_table):
    """
    Stitch a the right table to the bottom of the left table. Modifies
    the left_table inplace.
    
    :param left_table: 
    :param right_table: 
    :return: 
    """
    # First step is to rearrange the bitmap index if needed
    joined_columns = set(chain(left_table.keys, right_table.keys))
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
            
            
def add_column(table, k, v, index=None, align='top'):
    """
    Adds a column to a table inplace
    
    :param table: 
    :param k: 
    :param v: 
    :param index:
    :param align:
    :return: 
    """
    if k in table.keys:
        raise KeyError("Key {} already present".format(k))

    if type(v) == list:
        table.data.append(np.array(v))
        table.keys.append(k)

    elif type(v) == np.ndarray:
        if not len(v.shape) == 1:
            raise ValueError("Only 1D arrays supported")
        table.data.append(v)
        table.keys.append(k)

    elif type(v) == pd.DatetimeIndex:
        table.data.append(v)
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
            raise ValueError('Wrong alignment code')

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

    # First step is to compute the new index by filtering twice:
    new_index = table.index[:, predicate.index]
    new_index = table.index[:, predicate.values]

    # Now for the values
    for column, index in zip(table.data, table.index):
        enumerator = index.cumsum() - np.array(1)
        new_data.append(
            column[enumerator[index.astype(np.bool)][
                predicate.values.astype(np.bool)]]
        )

    return new_data, new_keys, new_index
