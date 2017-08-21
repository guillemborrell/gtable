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
def sieve_column_own(data, mask, index):
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
def sieve_column_other(data, index, rhs_index, mask):
    """
    Sieve data and index from the mask of other data array
    """
    new_index = np.empty_like(index)
    filtered_rhs_index = rhs_index.copy()
    rhs_index_cursor = 0
    for i in range(len(filtered_rhs_index)):
        if rhs_index[i]:
            if not mask[rhs_index_cursor]:
                filtered_rhs_index[i] = np.uint8(0)

            rhs_index_cursor += 1

    new_data_size = (index * filtered_rhs_index).sum()
    new_data = np.empty(new_data_size, dtype=data.dtype)

    rhs_data_cursor = 0
    new_data_cursor = 0

    for i in range(len(rhs_index)):
        if rhs_index[i]:
            if index[i]:
                if mask[rhs_data_cursor]:
                    new_data[new_data_cursor] = data[rhs_data_cursor]
                    new_data_cursor += 1
                    new_index[i] = np.uint8(1)

                else:
                    new_index[i] = np.uint8(0)
            else:
                new_index[i] = np.uint8(0)

            rhs_data_cursor += 1
        else:
            new_index[i] = np.uint8(0)

    return new_data, new_index


@jit(nopython=True, nogil=True, cache=True)
def crop_column_own(data, index):
    new_data = np.empty(index.sum(), dtype=data.dtype)
    new_index = np.ones(index.sum(), dtype=np.uint8)

    data_cursor = 0
    for i in range(len(index)):
        if index[i]:
            new_data[data_cursor] = data[data_cursor]
            data_cursor += 1

    return new_data, new_index


@jit(nopython=True, nogil=True, cache=True)
def crop_column_other(data, index, rhs_index):
    new_index = np.empty(rhs_index.sum(), dtype=np.uint8)
    new_data_size = (index * rhs_index).sum()
    new_data = np.empty(new_data_size, dtype=data.dtype)

    data_cursor = 0
    new_data_cursor = 0

    for i in range(len(rhs_index)):
        if rhs_index[i]:
            if index[i]:
                new_data[new_data_cursor] = data[data_cursor]
                new_index[data_cursor] = np.uint8(1)
                new_data_cursor += 1
            else:
                new_index[data_cursor] = np.uint8(0)

            data_cursor += 1

    return new_data, new_index
