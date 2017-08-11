import numba as nb
import numpy as np

@nb.jit('double[:](double[:], double[:])', nopython=True)
def concatenate_float(column_left, column_right):
    order_left = column_left.argsort()
    order_right = column_right.argsort()
    indices = np.searchsorted(column_left, column_right)
    return np.empty_like(column_left)


# @nb.jit('uint8[:,:](uint8[:,:], uint8[:,:], int64[:], int64[:], int64)',
#         nopython=True)
def merge_indices(left_index, right_index, insertions, new_columns, spill):
    left_index_0 = left_index.shape[0]
    left_index_1 = left_index.shape[1]
    ninsertions = insertions.shape[0]
    new_index_0 = left_index_0 + spill
    new_index_1 = left_index.shape[1] + right_index.shape[1]
    new_index = np.empty((new_index_0, new_index_1), dtype=np.uint8)
    new_columns_0 = new_columns.shape[0]
    
    ninserted = 0
    cursor_new = 0
    toinsert = insertions[ninserted]
    
    for i in range(new_columns_0):
        if new_columns[i] < 0:
            new_columns[i] = i
    
    for i in range(left_index_1):
        if i != toinsert:
            for j in range(new_index_0):
                if j < left_index_0:
                    new_index[j, cursor_new] = left_index[j, i]
                else:
                    new_index[j, cursor_new] = 0

            cursor_new += 1
        else:
            for j in range(new_index_0):
                if j < left_index_0:
                    new_index[j, cursor_new] = left_index[j, i]
                else:
                    new_index[j, cursor_new] = 0

            cursor_new += 1

            for nins in range(ninsertions - ninserted):
                l = insertions[ninserted + nins]
                if l > toinsert:
                    break

                for r in range(new_index_0):
                    new_index[r, cursor_new] = 0
                for k in range(new_columns_0):
                    j = new_columns[k]
                    new_index[j, cursor_new] = right_index[k, nins]

                cursor_new += 1

            ninserted += nins
            toinsert = l

    # Append the rest
    for nins in range(ninsertions - ninserted):
        l = insertions[ninserted + nins]
        for r in range(new_index_0):
            new_index[r, cursor_new] = 0
            
        for k in range(new_columns_0):
            j = new_columns[k]
            new_index[j, cursor_new] = right_index[k, nins]

        cursor_new += 1
            
    return new_index
