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
            result[cursor] = operator.xor(
                value_left[cursor_left].astype(np.bool),
                value_right[cursor_right].astype(np.bool))
            cursor += 1

        if index_left[i]:
            cursor_left += 1

        if index_right[i]:
            cursor_right += 1

    return result, index
