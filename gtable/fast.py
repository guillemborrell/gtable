"""
File with all the numba-accelerated functions. Some of this functions are
hard, and tremendously non-pythonic due to JIT optimization. They are not
that hard, but you must take it with patience.
"""

import numpy as np
from numba import jit, generated_jit, types


@generated_jit(nopython=True, nogil=True, cache=True)
def apply_fast_add(value_left, value_right, index_left, index_right):
    if value_left.dtype == types.int64 and value_right.dtype == types.int64:
        result_dtype = np.int64
    else:
        result_dtype = np.float64

    def f(value_left, value_right, index_left, index_right):
        index = index_left * index_right
        if len(value_left) > len(value_right):
            result = np.empty(len(value_left), dtype=result_dtype)
        else:
            result = np.empty(len(value_right), dtype=result_dtype)
        cursor_result = 0
        cursor_left = 0
        cursor_right = 0
        for il, ir in zip(index_left, index_right):
            if il & ir:
                result[cursor_result] = value_left[cursor_left] +\
                                        value_right[cursor_right]
                cursor_result += 1
                cursor_left += 1
                cursor_right += 1

            elif il:
                cursor_left += 1

            elif ir:
                cursor_right += 1

        return result[:cursor_result], index

    return f


@generated_jit(nopython=True, nogil=True, cache=True)
def apply_fast_sub(value_left, value_right, index_left, index_right):
    if value_left.dtype == types.int64 and value_right.dtype == types.int64:
        result_dtype = np.int64
    else:
        result_dtype = np.float64

    def f(value_left, value_right, index_left, index_right):
        index = index_left * index_right
        if len(value_left) > len(value_right):
            result = np.empty(len(value_left), dtype=result_dtype)
        else:
            result = np.empty(len(value_right), dtype=result_dtype)

        cursor_result = 0
        cursor_left = 0
        cursor_right = 0
        for il, ir in zip(index_left, index_right):
            if il & ir:
                result[cursor_result] = value_left[cursor_left] -\
                                        value_right[cursor_right]
                cursor_result += 1
                cursor_left += 1
                cursor_right += 1

            elif il:
                cursor_left += 1

            elif ir:
                cursor_right += 1

        return result[:cursor_result], index

    return f


@generated_jit(nopython=True, nogil=True, cache=True)
def apply_fast_mul(value_left, value_right, index_left, index_right):
    if value_left.dtype == types.int64 and value_right.dtype == types.int64:
        result_dtype = np.int64
    else:
        result_dtype = np.float64

    def f(value_left, value_right, index_left, index_right):
        index = index_left * index_right
        if len(value_left) > len(value_right):
            result = np.empty(len(value_left), dtype=result_dtype)
        else:
            result = np.empty(len(value_right), dtype=result_dtype)

        cursor_result = 0
        cursor_left = 0
        cursor_right = 0
        for il, ir in zip(index_left, index_right):
            if il & ir:
                result[cursor_result] = value_left[cursor_left] *\
                                        value_right[cursor_right]
                cursor_result += 1
                cursor_left += 1
                cursor_right += 1

            elif il:
                cursor_left += 1

            elif ir:
                cursor_right += 1

        return result[:cursor_result], index

    return f


@generated_jit(nopython=True, nogil=True, cache=True)
def apply_fast_truediv(value_left, value_right, index_left, index_right):
    if value_left.dtype == types.int64 and value_right.dtype == types.int64:
        result_dtype = np.int64
    else:
        result_dtype = np.float64

    def f(value_left, value_right, index_left, index_right):
        index = index_left * index_right
        if len(value_left) > len(value_right):
            result = np.empty(len(value_left), dtype=result_dtype)
        else:
            result = np.empty(len(value_right), dtype=result_dtype)

        cursor_result = 0
        cursor_left = 0
        cursor_right = 0
        for il, ir in zip(index_left, index_right):
            if il & ir:
                result[cursor_result] = value_left[cursor_left] /\
                                        value_right[cursor_right]
                cursor_result += 1
                cursor_left += 1
                cursor_right += 1

            elif il:
                cursor_left += 1

            elif ir:
                cursor_right += 1

        return result[:cursor_result], index

    return f


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_floordiv(value_left, value_right, index_left, index_right):
    index = index_left * index_right
    if len(value_left) > len(value_right):
        result = np.empty(len(value_left), dtype=np.int64)
    else:
        result = np.empty(len(value_right), dtype=np.int64)

    cursor_result = 0
    cursor_left = 0
    cursor_right = 0
    for il, ir in zip(index_left, index_right):
        if il & ir:
            result[cursor_result] = value_left[cursor_left] //\
                                    value_right[cursor_right]
            cursor_result += 1
            cursor_left += 1
            cursor_right += 1

        elif il:
            cursor_left += 1

        elif ir:
            cursor_right += 1

    return result[:cursor_result], index


@generated_jit(nopython=True, nogil=True, cache=True)
def apply_fast_pow(value_left, value_right, index_left, index_right):
    if value_left.dtype == types.int64 and value_right.dtype == types.int64:
        result_dtype = np.int64
    else:
        result_dtype = np.float64

    def f(value_left, value_right, index_left, index_right):
        index = index_left * index_right
        if len(value_left) > len(value_right):
            result = np.empty(len(value_left), dtype=result_dtype)
        else:
            result = np.empty(len(value_right), dtype=result_dtype)

        cursor_result = 0
        cursor_left = 0
        cursor_right = 0
        for il, ir in zip(index_left, index_right):
            if il & ir:
                result[cursor_result] = value_left[cursor_left] **\
                                        value_right[cursor_right]
                cursor_result += 1
                cursor_left += 1
                cursor_right += 1

            elif il:
                cursor_left += 1

            elif ir:
                cursor_right += 1

        return result[:cursor_result], index

    return f


@generated_jit(nopython=True, nogil=True, cache=True)
def apply_fast_mod(value_left, value_right, index_left, index_right):
    if value_left.dtype == types.int64 and value_right.dtype == types.int64:
        result_dtype = np.int64
    else:
        result_dtype = np.float64

    def f(value_left, value_right, index_left, index_right):
        index = index_left * index_right
        if len(value_left) > len(value_right):
            result = np.empty(len(value_left), dtype=result_dtype)
        else:
            result = np.empty(len(value_right), dtype=result_dtype)

        cursor_result = 0
        cursor_left = 0
        cursor_right = 0
        for il, ir in zip(index_left, index_right):
            if il & ir:
                result[cursor_result] = value_left[cursor_left] %\
                                        value_right[cursor_right]
                cursor_result += 1
                cursor_left += 1
                cursor_right += 1

            elif il:
                cursor_left += 1

            elif ir:
                cursor_right += 1

        return result[:cursor_result], index

    return f


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_gt(value_left, value_right, index_left, index_right):
    index = index_left * index_right
    if len(value_left) > len(value_right):
        result = np.empty(len(value_left), dtype=np.bool_)
    else:
        result = np.empty(len(value_right), dtype=np.bool_)

    cursor_result = 0
    cursor_left = 0
    cursor_right = 0
    for il, ir in zip(index_left, index_right):
        if il & ir:
            result[cursor_result] = value_left[cursor_left] >\
                                    value_right[cursor_right]
            cursor_result += 1
            cursor_left += 1
            cursor_right += 1

        elif il:
            cursor_left += 1

        elif ir:
            cursor_right += 1

    return result[:cursor_result], index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_ge(value_left, value_right, index_left, index_right):
    index = index_left * index_right
    if len(value_left) > len(value_right):
        result = np.empty(len(value_left), dtype=np.bool_)
    else:
        result = np.empty(len(value_left), dtype=np.bool_)

    cursor_result = 0
    cursor_left = 0
    cursor_right = 0
    for il, ir in zip(index_left, index_right):
        if il & ir:
            result[cursor_result] = value_left[cursor_left] >=\
                                    value_right[cursor_right]
            cursor_result += 1
            cursor_left += 1
            cursor_right += 1

        elif il:
            cursor_left += 1

        elif ir:
            cursor_right += 1

    return result[:cursor_result], index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_lt(value_left, value_right, index_left, index_right):
    index = index_left * index_right
    if len(value_left) > len(value_right):
        result = np.empty(len(value_left), dtype=np.bool_)
    else:
        result = np.empty(len(value_right), dtype=np.bool_)

    cursor_result = 0
    cursor_left = 0
    cursor_right = 0
    for il, ir in zip(index_left, index_right):
        if il & ir:
            result[cursor_result] = value_left[cursor_left] <\
                                    value_right[cursor_right]
            cursor_result += 1
            cursor_left += 1
            cursor_right += 1

        elif il:
            cursor_left += 1

        elif ir:
            cursor_right += 1

    return result[:cursor_result], index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_le(value_left, value_right, index_left, index_right):
    index = index_left * index_right
    if len(value_left) > len(value_right):
        result = np.empty(len(value_left), dtype=np.bool_)
    else:
        result = np.empty(len(value_right), dtype=np.bool_)

    cursor_result = 0
    cursor_left = 0
    cursor_right = 0
    for il, ir in zip(index_left, index_right):
        if il & ir:
            result[cursor_result] = value_left[cursor_left] <=\
                                    value_right[cursor_right]
            cursor_result += 1
            cursor_left += 1
            cursor_right += 1

        elif il:
            cursor_left += 1

        elif ir:
            cursor_right += 1

    return result[:cursor_result], index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_and(value_left, value_right, index_left, index_right):
    index = index_left * index_right
    if len(value_left) > len(value_right):
        result = np.empty(len(value_left), dtype=np.bool_)
    else:
        result = np.empty(len(value_right), dtype=np.bool_)

    cursor_left = 0
    cursor_right = 0
    cursor_result = 0

    for il, ir in zip(index_left, index_right):
        if il & ir:
            if value_left[cursor_left] and value_right[cursor_right]:
                result[cursor_result] = 1
            else:
                result[cursor_result] = 0
            cursor_result += 1
            cursor_left += 1
            cursor_right += 1

        elif il:
            cursor_left += 1

        elif ir:
            cursor_right += 1

    return result[:cursor_result], index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_or(value_left, value_right, index_left, index_right):
    index = index_left * index_right
    if len(value_left) > len(value_right):
        result = np.empty(len(value_left), dtype=np.bool_)
    else:
        result = np.empty(len(value_right), dtype=np.bool_)

    cursor_left = 0
    cursor_right = 0
    cursor_result = 0

    for il, ir in zip(index_left, index_right):
        if il & ir:
            if value_left[cursor_left] or value_right[cursor_right]:
                result[cursor_result] = 1
            else:
                result[cursor_result] = 0
            cursor_result += 1
            cursor_left += 1
            cursor_right += 1

        elif il:
            cursor_left += 1

        elif ir:
            cursor_right += 1

    return result[:cursor_result], index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_xor(value_left, value_right, index_left, index_right):
    index = index_left * index_right
    if len(value_left) > len(value_right):
        result = np.empty(len(value_left), dtype=np.bool_)
    else:
        result = np.empty(len(value_right), dtype=np.bool_)

    cursor_left = 0
    cursor_right = 0
    cursor_result = 0

    for il, ir in zip(index_left, index_right):
        if il & ir:
            if value_left[cursor_left]:
                if value_right[cursor_right]:
                    result[cursor_result] = 0
                else:
                    result[cursor_result] = 1
            else:
                if value_right[cursor_right]:
                    result[cursor_result] = 1
                else:
                    result[cursor_result] = 0
            cursor_result += 1
            cursor_left += 1
            cursor_right += 1

        elif il:
            cursor_left += 1

        elif ir:
            cursor_right += 1

    return result[:cursor_result], index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_eq(value_left, value_right, index_left, index_right):
    index = index_left * index_right
    if len(value_left) > len(value_right):
        result = np.empty(len(value_left), dtype=np.bool_)
    else:
        result = np.empty(len(value_right), dtype=np.bool_)

    cursor_result = 0
    cursor_left = 0
    cursor_right = 0
    for il, ir in zip(index_left, index_right):
        if il & ir:
            if value_left[cursor_left] == value_right[cursor_right]:
                result[cursor_result] = 1
            else:
                result[cursor_result] = 0
            cursor_result += 1
            cursor_left += 1
            cursor_right += 1

        elif il:
            cursor_left += 1

        elif ir:
            cursor_right += 1

    return result[:cursor_result], index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_ne(value_left, value_right, index_left, index_right):
    index = index_left * index_right
    if len(value_left) > len(value_right):
        result = np.empty(len(value_left), dtype=np.bool_)
    else:
        result = np.empty(len(value_right), dtype=np.bool_)

    cursor_result = 0
    cursor_left = 0
    cursor_right = 0
    for il, ir in zip(index_left, index_right):
        if il & ir:
            if value_left[cursor_left] != value_right[cursor_right]:
                result[cursor_result] = 1
            else:
                result[cursor_result] = 0
            cursor_result += 1
            cursor_left += 1
            cursor_right += 1

        elif il:
            cursor_left += 1

        elif ir:
            cursor_right += 1

    return result[:cursor_result], index


@jit(nopython=True, nogil=True, cache=True)
def apply_mask_column(data, index, mask):
    """
    Sieve picks a mask over data and returns the filtered data array and index
    """
    new_data = data[mask]
    new_index = np.empty_like(index)

    data_cursor = 0
    for i, idx in enumerate(index):
        if idx:
            if mask[data_cursor]:
                new_index[i] = 1
            else:
                new_index[i] = 0
            data_cursor += 1
        else:
            new_index[i] = 0

    return new_data, new_index


@jit(nopython=True, nogil=True, cache=True)
def isin_sorted(base, test):
    result = np.empty(base.shape, dtype=np.bool_)
    cursor_result = 0
    cursor_test = 0
    for elem in base:
        result[cursor_result] = False
        for i in range(len(test)):
            if elem < test[cursor_test]:
                break
            elif elem == test[cursor_test]:
                result[cursor_result] = True
                break
            else:
                # array exhausted
                if cursor_test == len(test) - 1:
                    break
                # Advance test array
                else:
                    cursor_test += 1

        cursor_result += 1

    return result


@jit(nopython=True, nogil=True, cache=True)
def intersection_sorted(base, test):
    result = np.empty(base.shape, dtype=base.dtype)
    cursor_result = 0
    cursor_test = 0
    for elem in base:
        for i in range(len(test)):
            if elem < test[cursor_test]:
                break
            elif elem == test[cursor_test]:
                result[cursor_result] = elem
                cursor_result += 1
                break
            else:
                # array exhausted
                if cursor_test == len(test) - 1:
                    break
                # Advance test array
                else:
                    cursor_test += 1

    return result[:cursor_result]


@jit(nopython=True, nogil=True, cache=True)
def union_sorted(base, test):
    result = np.empty(base.shape[0] + test.shape[0], dtype=base.dtype)
    cursor_result = 0
    cursor_base = 0
    cursor_test = 0
    started = False

    while cursor_base < len(base) and cursor_test < len(test):
        if base[cursor_base] < test[cursor_test]:
            if started:
                if base[cursor_base] == result[cursor_result - 1]:
                    continue
                else:
                    result[cursor_result] = base[cursor_base]
                    cursor_result += 1
                    cursor_base += 1
            else:
                result[cursor_result] = base[cursor_base]
                cursor_result += 1
                cursor_base += 1
                started = True

        elif test[cursor_test] < base[cursor_base]:
            if started:
                if test[cursor_test] == result[cursor_result - 1]:
                    continue
                else:
                    result[cursor_result] = test[cursor_test]
                    cursor_result += 1
                    cursor_test += 1
            else:
                result[cursor_result] = test[cursor_test]
                cursor_result += 1
                cursor_test += 1
                started = True

        else:
            if started:
                if test[cursor_test] == result[cursor_result - 1]:
                    continue
                else:
                    result[cursor_result] = test[cursor_test]
                    cursor_result += 1
                    cursor_base += 1
                    cursor_test += 1

            else:
                result[cursor_result] = test[cursor_test]
                cursor_result += 1
                cursor_base += 1
                cursor_test += 1
                started = True

    while cursor_base < len(base):
        if base[cursor_base] == result[cursor_result - 1]:
            continue
        else:
            result[cursor_result] = base[cursor_base]
            cursor_result += 1
            cursor_base += 1

    while cursor_test < len(test):
        if test[cursor_test] == result[cursor_result - 1]:
            continue
        else:
            result[cursor_result] = test[cursor_test]
            cursor_result += 1
            cursor_test += 1

    return result[:cursor_result]


@jit(nopython=True, nogil=True, cache=True)
def join_low_level(data_left, index_left,
                   data_right, index_right,
                   common_rec):
    """This is probably the hardest function of all. It returns the joined array
    plus the indices for the left and right data that produce the join. The most
    important detail is that, for the left and right index, a negative value
    means that the value must not be present in the joined column. There are
    two passes over the data, the first one is get the size of the arrays
    that have to be allocated, and the second is the actual computation of
    the join. This is only used for full outer joins, but it should work for
    left and right joins too. Due to numba limitations, you cannot use
    a column of strings to join two tables."""
    data_filter_left = isin_sorted(data_left, common_rec)
    data_filter_right = isin_sorted(data_right, common_rec)

    filtered_data_left, index_left = apply_mask_column(
        data_left, index_left, data_filter_left)
    filtered_data_right, index_right = apply_mask_column(
        data_right, index_right, data_filter_right)

    length = 0
    left_len = len(filtered_data_left)
    right_len = len(filtered_data_right)
    cur_left = 0
    cur_right = 0
    stop_left = False
    stop_right = False

    # Go through the whole data just to get the length of the result.
    for i in range(left_len + right_len):  # Upper limit to prevent Inf loop
        if filtered_data_left[cur_left] < filtered_data_right[cur_right]:
            if stop_left:
                if data_filter_right[cur_right]:
                    length += 1
                if cur_right == right_len - 1:
                    stop_right = True
                else:
                    cur_right += 1
            else:
                if data_filter_left[cur_left]:
                    length += 1
                if cur_left == left_len - 1:
                    stop_left = True
                else:
                    cur_left += 1

        elif filtered_data_left[cur_left] == filtered_data_right[cur_right]:
            # Both filters always true
            if cur_left == left_len - 1:
                stop_left = True
            else:
                cur_left += 1
            if cur_right == right_len - 1:
                stop_right = True
            else:
                cur_right += 1

            length += 1

        else:
            if stop_right:
                if data_filter_left[cur_left]:
                    length += 1
                if cur_left == left_len - 1:
                    stop_left = True
                else:
                    cur_left += 1
            else:
                if data_filter_right[cur_right]:
                    length += 1
                if cur_right == right_len - 1:
                    stop_right = True
                else:
                    cur_right += 1

        if stop_left and stop_right:
            break

    data_joined = np.empty(length, dtype=filtered_data_left.dtype)
    order_left = np.empty(length, dtype=np.int64)
    order_right = np.empty(length, dtype=np.int64)

    stop_left = False
    stop_right = False
    cur_left = 0
    cur_right = 0
    added = 0

    while added < length:  # Upper limit to prevent Inf loop
        if filtered_data_left[cur_left] < filtered_data_right[cur_right]:
            if stop_left:
                if data_filter_right[cur_right]:
                    data_joined[added] = filtered_data_right[cur_right]
                    order_left[added] = cur_left
                    order_right[added] = cur_right
                    added += 1
                if cur_right == right_len - 1:
                    stop_right = True
                else:
                    cur_right += 1
            else:
                if data_filter_left[cur_left]:
                    data_joined[added] = filtered_data_left[cur_left]
                    order_left[added] = cur_left
                    order_right[added] = cur_right
                    added += 1
                if cur_left == left_len - 1:
                    stop_left = True
                else:
                    cur_left += 1

        elif filtered_data_left[cur_left] == filtered_data_right[cur_right]:
            # Both filters always true
            data_joined[added] = filtered_data_left[cur_left]
            order_left[added] = cur_left
            order_right[added] = cur_right

            if cur_left == left_len - 1:
                stop_left = True
            else:
                cur_left += 1
            if cur_right == right_len - 1:
                stop_right = True
            else:
                cur_right += 1

            added += 1

        else:
            if stop_right:
                if data_filter_left[cur_left]:
                    data_joined[added] = filtered_data_left[cur_left]
                    order_left[added] = cur_left
                    order_right[added] = cur_right
                    added += 1
                if cur_left == left_len - 1:
                    stop_left = True
                else:
                    cur_left += 1
            else:
                if data_filter_right[cur_right]:
                    data_joined[added] = filtered_data_right[cur_right]
                    order_left[added] = cur_left
                    order_right[added] = cur_right
                    added += 1
                if cur_right == right_len - 1:
                    stop_right = True
                else:
                    cur_right += 1

        if stop_left and stop_right:
            break

    index_mapping_left = np.arange(len(index_left))[index_left == 1]
    index_mapping_right = np.arange(len(index_right))[index_right == 1]

    global_left = index_mapping_left[order_left]
    global_right = index_mapping_right[order_right]

    mask_left = isin_sorted(data_joined, data_left)
    mask_right = isin_sorted(data_joined, data_right)

    # Clean data with the mask
    global_left[np.where(~mask_left)] = -1
    global_right[np.where(~mask_right)] = -1

    return data_joined, global_left, global_right


@jit(nopython=True, nogil=True, cache=True)
def reindex(index, global_index):
    """Reindex a column data using a global (table-wise) index. A global
    index is the index for the full column, and this operation exists because
    data columns can be shorter."""
    data_len = 0

    for idx in global_index:
        # Negative index means empty
        if index[idx] == 1 and idx >= 0:
            data_len += 1

    new_data_index = np.empty(data_len, dtype=np.int64)
    new_index = np.empty(len(global_index), dtype=np.uint8)
    data_index = index.cumsum() - np.array(1)

    data_cursor = 0
    cursor = 0

    for idx in global_index:
        if idx < 0:
            new_index[cursor] = 0
        elif index[idx]:
            new_data_index[data_cursor] = int(data_index[idx])
            data_cursor += 1
            new_index[cursor] = 1
        else:
            new_index[cursor] = 0

        cursor += 1

    return new_data_index, new_index


# This is an important wrapper, since numba does not know how to deal with
# unicode strings.
def reindex_column(data, index, global_index):
    new_data_index, new_index = reindex(index, global_index)
    return data[new_data_index], new_index


@jit(nopython=True, nogil=True, cache=True)
def reindex_join(left_index, right_index,
                 left_global_index, right_global_index):
    """Applies a global index"""
    # TODO: Clean unused variables.
    len_left = 0
    len_right = 0
    idx_left_cur = 0
    idx_right_cur = 0

    for left_gidx, right_gidx in zip(left_global_index, right_global_index):
        if left_gidx >= 0:
            if left_index[idx_left_cur] > 0:
                len_left += 1

            else:
                if right_gidx >= 0:
                    if right_index[idx_right_cur] > 0:
                        len_right += 1
                    idx_right_cur += 1

            idx_left_cur += 1
                    
        elif right_gidx >= 0:
            if right_index[idx_right_cur] > 0:
                len_right += 1

            else:
                if left_gidx >= 0:
                    if left_index[idx_left_cur] > 0:
                        len_left += 1
                    idx_left_cur += 1

            idx_right_cur += 1

    lid = np.empty(len_left + len_right, dtype=np.bool_)
    rid = np.empty(len_left + len_right, dtype=np.bool_)
    left_global = np.empty(len_left, dtype=np.int64)
    right_global = np.empty(len_right, dtype=np.int64)
    index_global = np.zeros(len(left_global_index), dtype=np.uint8)

    idx_data = 0
    idx_left = 0
    idx_right = 0
    idx_left_cur = 0
    idx_right_cur = 0
    idx_index = 0
    
    for left_gidx, right_gidx in zip(left_global_index, right_global_index):
        if left_gidx >= 0:
            if left_index[idx_left_cur] > 0:
                left_global[idx_left] = left_gidx
                index_global[idx_index] = 1
                idx_left += 1
                lid[idx_data] = True
                rid[idx_data] = False
                idx_data += 1
                if right_gidx >= 0:
                    idx_right_cur += 1

            else:
                if right_gidx >= 0:
                    if right_index[idx_right_cur] > 0:
                        right_global[idx_right] = right_gidx
                        index_global[idx_index] = 1
                        idx_right += 1
                        lid[idx_data] = False
                        rid[idx_data] = True
                        idx_data += 1
                    idx_right_cur += 1

            idx_left_cur += 1
                    
        elif right_gidx >= 0:
            if right_index[idx_right_cur] > 0:
                right_global[idx_right] = right_gidx
                index_global[idx_index] = 1
                idx_right += 1
                lid[idx_data] = False
                rid[idx_data] = True
                idx_data += 1
                if left_gidx >= 0:
                    idx_left_cur += 1

            else:
                if left_gidx >= 0:
                    if left_index[idx_left_cur] > 0:
                        left_global[idx_left] = left_gidx
                        index_global[idx_index] = 1
                        idx_left += 1
                        lid[idx_data] = True
                        rid[idx_data] = False
                        idx_data += 1
                    idx_left_cur += 1

            idx_right_cur += 1
        idx_index += 1

    return index_global, left_global, right_global, lid, rid, idx_data


# Same wrapper for column join
def reindex_join_columns(left_column, right_column,
                         left_global_index, right_global_index):

    new_index, left_global, right_global, lid, rid, size = reindex_join(
        left_column.index, right_column.index,
        left_global_index, right_global_index)

    data = np.empty(size, dtype=left_column.values.dtype)
    data[lid] = left_column.values[left_global]
    data[rid] = right_column.values[right_global]

    return data, new_index


@generated_jit(nopython=True, nogil=True, cache=True)
def reduce_sum(key_data, key_index, col_data, col_index, size):
    result_dtype = col_data.dtype

    def f(key_data, key_index, col_data, col_index, size):
        prev_key_data = key_data[0]

        reduction_started = False
        new_index = np.zeros(size, dtype=np.uint8)
        reduced = np.empty(size, dtype=result_dtype)
        key_data_cursor = -1
        col_data_cursor = -1
        new_index_cursor = -1
        red_index = 0

        for idxk, idxd in zip(key_index, col_index):
            if idxk == 1:
                key_data_cursor += 1
                if prev_key_data != key_data[key_data_cursor] or \
                                key_data_cursor == 0:
                    new_index_cursor += 1

            if idxd == 1:
                col_data_cursor += 1

            if idxk == idxd == 1:
                if reduction_started:
                    if prev_key_data == key_data[key_data_cursor]:
                        # Here is the reduction of repeated keys
                        reduced[red_index] += col_data[col_data_cursor]
                    else:
                        red_index += 1
                        # Here is the addition of the first key
                        reduced[red_index] = col_data[col_data_cursor]
                        new_index[new_index_cursor] = 1
                else:
                    # Here is the addition of the absolutely first key
                    reduced[red_index] = col_data[col_data_cursor]
                    new_index[new_index_cursor] = 1
                    prev_key_data = key_data[key_data_cursor]
                    reduction_started = True

            if idxk == 1:
                prev_key_data = key_data[key_data_cursor]

        return reduced[:red_index+1][:], new_index

    return f


@generated_jit(nopython=True, nogil=True, cache=True)
def reduce_prod(key_data, key_index, col_data, col_index, size):
    result_dtype = col_data.dtype

    def f(key_data, key_index, col_data, col_index, size):
        prev_key_data = key_data[0]

        reduction_started = False
        new_index = np.zeros(size, dtype=np.uint8)
        reduced = np.empty(size, dtype=result_dtype)
        key_data_cursor = -1
        col_data_cursor = -1
        new_index_cursor = -1
        red_index = 0

        for idxk, idxd in zip(key_index, col_index):
            if idxk == 1:
                key_data_cursor += 1
                if prev_key_data != key_data[key_data_cursor] or \
                                key_data_cursor == 0:
                    new_index_cursor += 1

            if idxd == 1:
                col_data_cursor += 1

            if idxk == idxd == 1:
                if reduction_started:
                    if prev_key_data == key_data[key_data_cursor]:
                        # Here is the reduction of repeated keys
                        reduced[red_index] *= col_data[col_data_cursor]
                    else:
                        red_index += 1
                        # Here is the addition of the first key
                        reduced[red_index] = col_data[col_data_cursor]
                        new_index[new_index_cursor] = 1
                else:
                    # Here is the addition of the absolutely first key
                    reduced[red_index] = col_data[col_data_cursor]
                    new_index[new_index_cursor] = 1
                    prev_key_data = key_data[key_data_cursor]
                    reduction_started = True

            if idxk == 1:
                prev_key_data = key_data[key_data_cursor]

        return reduced[:red_index+1][:], new_index

    return f


@jit(nopython=True, nogil=True, cache=True)
def gen_column_filter(p_values, p_index, c_index):
    data_filter = np.empty(c_index.sum(), dtype=np.bool_)
    new_index = np.zeros(p_values.sum(), dtype=np.uint8)

    p_values_cursor = 0
    d_values_cursor = 0
    d_index_cursor = 0

    for p_i, c_i in zip(p_index, c_index):
        if p_i and c_i:
            if p_values[p_values_cursor]:
                new_index[d_index_cursor] = 1
                data_filter[d_values_cursor] = True
                d_index_cursor += 1
            else:
                data_filter[d_values_cursor] = False

            p_values_cursor += 1
            d_values_cursor += 1

        elif p_i:
            if p_values[p_values_cursor]:
                new_index[d_index_cursor] = 0
                d_index_cursor += 1
            p_values_cursor += 1

        elif c_i:
            data_filter[d_values_cursor] = False
            d_values_cursor += 1

    return data_filter, new_index
