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
def sieve_own(data, mask, index):
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
        
    
@jit(nopyton=True, nogil=True, cache=True)
def sieve_other(data, datafilter, mask, index, indexfilter):
    """
    Sieve data and index from the mask of other data array
    """
    new_index = np.empty_like(index)
    new_data = list()

    data_cursor = 0

    for i in range(len(index)):
        if indexfilter[i]:
            if index[i]:
            if mask[data_cursor]:
                new_index[i] = 1
            else:
                new_index[i] = 0
            data_cursor += 1
        else:
            new_index[i] = 0
