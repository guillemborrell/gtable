from numba import jit
import numpy as np
import operator


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_add(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.add(value_left[cursor_left],
                                          value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_sub(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.sub(value_left[cursor_left],
                                          value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_mul(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.mul(value_left[cursor_left],
                                          value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_truediv(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.truediv(value_left[cursor_left],
                                              value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_floordiv(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.floordiv(value_left[cursor_left],
                                               value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_pow(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.pow(value_left[cursor_left],
                                          value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_mod(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.mod(value_left[cursor_left],
                                          value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_gt(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.gt(value_left[cursor_left],
                                         value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_ge(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.ge(value_left[cursor_left],
                                         value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_lt(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.lt(value_left[cursor_left],
                                         value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_le(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.float64)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            result[cursor] = operator.le(value_left[cursor_left],
                                         value_right[cursor_right])
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_and(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.uint8)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            if value_left[cursor_left] and value_right[cursor_right]:
                result[cursor] = 1
            else:
                result[cursor] = 0
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_or(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.uint8)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            if value_left[cursor_left] or value_right[cursor_right]:
                result[cursor] = 1
            else:
                result[cursor] = 0
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_xor(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.bool)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            if value_left[cursor_left]:
                if value_right[cursor_right]:
                    result[cursor] = 0
                else:
                    result[cursor] = 1
            else:
                if value_right[cursor_right]:
                    result[cursor] = 1
                else:
                    result[cursor] = 0
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_eq(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.uint8)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            if operator.eq(value_left[cursor_left], value_right[cursor_right]):
                result[cursor] = 1
            else:
                result[cursor] = 0
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_fast_ne(value_left, value_right, index_left, index_right):
    index = np.bitwise_and(index_left, index_right)
    result = np.empty(index.sum(), dtype=np.uint8)

    cursor = 0
    cursor_left = 0
    cursor_right = 0
    for i in range(len(index_left)):
        if index_left[i] & index_right[i] & index_left[i] == index_right[i]:
            if operator.ne(value_left[cursor_left], value_right[cursor_right]):
                result[cursor] = 1
            else:
                result[cursor] = 0
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index


@jit(nopython=True, nogil=True, cache=True)
def apply_mask_column(data, index, mask):
    """
    Sieve picks a mask over data and returns the filtered data array and index
    """
    new_data = data[mask]
    new_index = np.empty_like(index)

    data_cursor = 0
    for i in range(len(index)):
        if index[i]:
            if mask[data_cursor]:
                new_index[i] = 1
            else:
                new_index[i] = 0
            data_cursor += 1
        else:
            new_index[i] = 0

    return new_data, new_index


@jit(nopython=True, nogil=True, cache=True)
def join_low_level(filtered_data_left, index_left,
                   filtered_data_right, index_right):
    length = 0
    left_len = len(filtered_data_left)
    right_len = len(filtered_data_right)
    cur_left = 0
    cur_right = 0
    stop_left = False
    stop_right = False

    while not (stop_left and stop_right):
        if filtered_data_left[cur_left] < filtered_data_right[cur_right]:
            length += 1
            if cur_left == left_len - 1:
                stop_left = True
            else:
                cur_left += 1

        elif filtered_data_left[cur_left] == filtered_data_right[cur_right]:
            length += 1
            if cur_left == left_len - 1:
                stop_left = True
            else:
                cur_left += 1
            if cur_right == right_len - 1:
                stop_right = True
            else:
                cur_right += 1

        else:
            length += 1
            if cur_right == right_len - 1:
                stop_right = True
            else:
                cur_right += 1

    data_joined = np.empty(length, dtype=filtered_data_left.dtype)
    order_left = np.empty(length, dtype=np.int64)
    order_right = np.empty(length, dtype=np.int64)

    cur_left = 0
    cur_right = 0
    added = 0

    while added < length:
        if filtered_data_left[cur_left] < filtered_data_right[cur_right]:
            data_joined[added] = filtered_data_left[cur_left]
            order_left[added] = cur_left
            order_right[added] = cur_right
            added += 1
            if cur_left < left_len - 1:
                cur_left += 1

        elif filtered_data_left[cur_left] == filtered_data_right[cur_right]:
            data_joined[added] = filtered_data_left[cur_left]
            order_left[added] = cur_left
            order_right[added] = cur_right
            added += 1
            if cur_left < left_len - 1:
                cur_left += 1
            if cur_right < right_len - 1:
                cur_right += 1

        else:
            data_joined[added] = filtered_data_right[cur_right]
            order_left[added] = cur_left
            order_right[added] = cur_right
            added += 1
            if cur_right < right_len - 1:
                cur_right += 1

    index_mapping_left = np.arange(len(index_left))[index_left == 1]
    index_mapping_right = np.arange(len(index_right))[index_right == 1]

    global_left = index_mapping_left[order_left]
    global_right = index_mapping_right[order_right]

    return data_joined, global_left, global_right


@jit(nopython=True, nogil=True, cache=True)
def reindex_column(data, index, global_index):
    data_len = 0

    for idx in global_index:
        if index[idx] == 1:
            data_len += 1

    new_data = np.empty(data_len, dtype=data.dtype)
    new_index = np.empty(len(global_index), dtype=np.uint8)
    data_index = index.cumsum() - np.array(1)

    data_cursor = 0
    cursor = 0

    for idx in global_index:
        if index[idx]:
            new_data[data_cursor] = data[int(data_index[idx])]
            data_cursor += 1
            new_index[cursor] = 1
        else:
            new_index[cursor] = 0

        cursor += 1

    return new_data, new_index
