import numpy as np
import operator
from gtable.lib import fillna_column
from gtable.fast import apply_fast_add, apply_fast_mul, apply_fast_truediv, \
    apply_fast_sub


class Column:
    """
    Indexed column view of the table
    """
    def __init__(self, values, index):
        self.values = values
        self.index = index
        self._enumerator = None

    def __repr__(self):
        return "<Column[ {} ] object at {}>".format(self.values.dtype,
                                                    hex(id(self)))

    def __add__(self, y):
        return apply_add(self, y)

    def __radd__(self, y):
        return apply_add(self, y)

    def __sub__(self, y):
        return apply_truediv(self, y)

    def __rsub__(self, y):
        return apply_truediv(self, y)

    def __mul__(self, y):
        return apply_mul(self, y)

    def __rmul__(self, y):
        return apply_mul(self, y)

    def __truediv__(self, y):
        return apply_truediv(self, y)

    def __rtruediv__(self, y):
        return apply_truediv(self, y)
    
    def __floordiv__(self, y):
        return apply_operator(self, y, operator.floordiv)

    def __rfloordiv__(self, y):
        return apply_operator(self, y, operator.floordiv)

    def __pow__(self, y):
        return apply_operator(self, y, operator.pow)

    def __mod__(self, y):
        return apply_operator(self, y, operator.mod)

    def __lt__(self, y):
        return apply_operator(self, y, operator.lt)

    def __le__(self, y):
        return apply_operator(self, y, operator.le)

    def __gt__(self, y):
        return apply_operator(self, y, operator.gt)

    def __ge__(self, y):
        return apply_operator(self, y, operator.ge)

    def __eq__(self, y):
        return apply_operator_str(self, y, operator.eq)

    def __ne__(self, y):
        return apply_operator_str(self, y, operator.ne)

    def __and__(self, y):
        return apply_operator_bool(self, y, operator.and_)

    def __or__(self, y):
        return apply_operator_bool(self, y, operator.or_)

    def __xor__(self, y):
        return apply_operator_bool(self, y, operator.xor)

    def __neg__(self):
        return Column(-self.values, self.index)

    def __getitem__(self, i):
        if self.index[i]:
            if self._enumerator is None:
                self._enumerator = np.cumsum(self.index) - np.array(1)
            return self.values[self._enumerator[i]]
        else:
            return None

    def fillna(self, reverse=False, fillvalue=None):
        """
        Fills the non available value sequentially with the previous
        available position.
        """
        self.values, self.index = fillna_column(self.values,
                                                self.index,
                                                reverse,
                                                fillvalue)


def apply_add(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.add(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_add(left.values, right.values,
                                       left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.add(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.add(left.values, right), left.index)
    else:
        raise ValueError('type not supported')


def apply_sub(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.sub(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_sub(left.values, right.values,
                                       left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.sub(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.sub(left.values, right), left.index)
    else:
        raise ValueError('type not supported')


def apply_mul(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.mul(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_mul(left.values, right.values,
                                       left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.mul(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.mul(left.values, right), left.index)
    else:
        raise ValueError('type not supported')


def apply_truediv(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.truediv(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_truediv(left.values, right.values,
                                           left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.truediv(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.truediv(left.values, right), left.index)
    else:
        raise ValueError('type not supported')


def apply_operator(left: Column, right: Column, op):
    if type(right) == np.ndarray:
        return Column(op(left.values, right), left.index)
    elif type(right) == Column:
        if np.all(right.index == left.index):
            return Column(op(left.values, right.values), left.index)
        else:
            index = np.bitwise_and(left.index, right.index)
            masked_left = left.values[
                index.astype(np.bool)[
                    left.index.astype(np.bool)]]
            masked_right = right.values[
                index.astype(np.bool)[
                    right.index.astype(np.bool)]]
            result = op(masked_left, masked_right)
            return Column(result, index)

    elif type(right) == int:
        return Column(op(left.values, right), left.index)
    elif type(right) == float:
        return Column(op(left.values, right), left.index)
    else:
        raise ValueError('type not supported')


def apply_operator_str(left: Column, right: Column, op):
    if type(right) == np.ndarray:
        return Column(op(left.values, right), left.index)
    elif type(right) == Column:
        if np.all(right.index == left.index):
            return Column(op(left.values, right.values), left.index)
        else:
            index = np.bitwise_and(left.index, right.index)
            masked_left = left.values[
                index.astype(np.bool)[
                    left.index.astype(np.bool)]]
            masked_right = right.values[
                index.astype(np.bool)[
                    right.index.astype(np.bool)]]
            result = op(masked_left, masked_right)
            return Column(result, index)

    elif type(right) == str:
        return Column(op(left.values, right), left.index)
    elif type(right) == int:
        return Column(op(left.values, right), left.index)
    elif type(right) == float:
        return Column(op(left.values, right), left.index)
    else:
        raise ValueError('type not supported')


def apply_operator_bool(left: Column, right: Column, op):
    if type(right) == np.ndarray:
        return Column(op(left.values.astype(np.bool), right), left.index)
    elif type(right) == Column:
        if np.all(right.index == left.index):
            return Column(op(left.values.astype(np.bool),
                             right.values.astype(np.bool)),
                          left.index)
        else:
            index = np.bitwise_and(left.index, right.index)
            masked_left = left.values[
                index.astype(np.bool)[
                    left.index.astype(np.bool)]]
            masked_right = right.values[
                index.astype(np.bool)[
                    right.index.astype(np.bool)]]
            result = op(masked_left.astype(np.bool),
                        masked_right.astype(np.bool))
            return Column(result, index)

    elif type(right) == str:
        return Column(op(left.values.astype(np.bool), right), left.index)
    elif type(right) == int:
        return Column(op(left.values.astype(np.bool), right), left.index)
    elif type(right) == float:
        return Column(op(left.values.astype(np.bool), right), left.index)
    else:
        raise ValueError('type not supported')
