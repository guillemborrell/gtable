from itertools import chain
import numpy as np


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
    column_left_index = table_left.index[table_left.keys.index(column), :]
    column_right = table_right[column]
    sorter = np.argsort(column_left)

    insertions = np.searchsorted(column_left, column_right, sorter=sorter)
    # This is a little tricky. This indices are for the insertions within
    # each column of the index. Lots of magic and pencil and paper.
    insertions_index = np.searchsorted(column_left_index.cumsum(), insertions+1)
    all_columns = set(chain(table_left.keys, table_right.keys))
    left_length = table_left.index.shape[1]
    width = len(all_columns)
    length = table_left.index.shape[1] + len(insertions_index)
    new_index = np.empty((width, length), dtype=np.uint8)
    new_data = list()

    new_keys = list()
    for i, column in enumerate(all_columns):
        if column in table_left.keys and column in table_right.keys:
            left_index = table_left.index[table_left.keys.index(column), :]
            # Merge indices
            merged_index = np.insert(left_index, insertions_index,
                                     np.ones(len(insertions_index),
                                             dtype=np.uint8)
                                     )
            new_index[i, :] = merged_index

            # If the column is present in both tables, data must be merged
            left_key = table_left.keys.index(column)
            right_key = table_right.keys.index(column)
            left_orig_data = table_left.data[left_key]
            right_orig_data = table_right.data[right_key]
            left_dtype = left_orig_data.dtype
            left_data = np.empty(left_length, dtype=left_dtype)
            left_data[left_index.astype(np.bool)] = left_orig_data
            merged = np.insert(left_data, insertions, right_orig_data)
            new_data.append(merged[merged_index.astype(np.bool)])
            new_keys.append(column)

        elif column in table_left.keys:
            left_key = table_left.keys.index(column)
            new_index[i, :] = np.insert(
                table_left.index[left_key, :],
                insertions_index,
                np.zeros(len(insertions_index), dtype=np.uint8))
            new_data.append(table_left.data[table_left.keys.index(column)])
            new_keys.append(column)

        else:
            new_index[i, :] = np.insert(
                np.zeros(left_length, dtype=np.uint8),
                insertions_index,
                np.ones(len(insertions_index), dtype=np.uint8))
            new_data.append(table_right.data[table_right.keys.index(column)])
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
        column_subindex = np.cumsum(column_index) - np.array(1)
        new_indexer = column_subindex[table_index][column_index.astype(np.bool)]
        table.data[idx] = table.data[idx][new_indexer]
