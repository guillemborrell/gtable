import numpy as np
from gtable import Table
from itertools import chain


def merge(table_left, table_right, column):
    """
    Merge two tables using a column as index
    """
    if column not in table_left._keys:
        raise ValueError('{} not in left table'.format(column))

    if column not in table_right._keys:
        raise ValueError('{} not in right table'.format(column))
    
    column_left = table_left[column]
    column_right = table_right[column]
    sorter = np.argsort(column_left)

    insertions = np.searchsorted(column_left, column_right, sorter=sorter)
    all_columns = list(set(chain(table_left._keys, table_right._keys)))
    left_width = table_left._index.shape[0]
    left_length = table_left._index.shape[1]
    right_width = table_right._index.shape[0]
    right_length = table_right._index.shape[1]
    width = len(all_columns)
    length = table_left._index.shape[1] + table_right._index.shape[1]
    new_index = np.empty((width, length), dtype=np.uint8)
    new_data = list()

    existing_columns = 0
    new_columns = 0
    for i, column in enumerate(all_columns):
        if column in table_left._keys and column in table_right._keys:
            left_index = table_left._index[table_left._keys.index(column),:]
            right_index = table_right._index[table_right._keys.index(column),:]
            new_index[existing_columns,:] = np.insert(
                left_index, insertions, right_index
            )
            existing_columns += 1

            # If the column is present in both tables, data must be merged
            left_data = np.NAN * np.empty((left_length), dtype=np.uint8)
            right_data = np.NAN * np.empty((right_length), dtype=np.uint8)
            left_key = table_left._keys.index(column)
            right_key = table_right._keys.index(column)
            left_data[left_index.astype(np.bool)] = table_left._data[left_key]
            right_data[right_index.astype(np.bool)] = table_right._data[right_key]
            print(table_left._data[left_key], table_right._data[right_key])
            print(left_index, right_index)
            print(left_data, right_data)

            merged = np.insert(left_data, insertions, right_data)
            new_data.append(merged[~np.isnan(merged)])
            
        elif column in table_left._keys:
            new_index[existing_columns,:] = np.insert(
                table_left._index[table_left._keys.index(column),:],
                insertions,
                np.zeros((right_length), dtype=np.uint8))
            existing_columns += 1
            new_data.append(table_left._data[table_left._keys.index(column)])

        else:
            new_index[left_width + new_columns,:] = np.insert(
                np.zeros((left_length), dype=np.uint8),
                insertions,
                table_right._index[table_right._keys.index(column),:])
            new_columns += 1
            new_data.append(table_right._data[table_right._keys.index(column)])

    new_table = Table()
    new_table._data = new_data
    new_table._keys = all_columns
    new_table._index = new_index

    return new_table
    
