import numpy as np
import operator
from gtable.lib import fillna_column
from gtable.fast import apply_fast_add, apply_fast_mul, apply_fast_truediv, \
    apply_fast_sub, apply_fast_floordiv, apply_fast_and, apply_fast_or, \
    apply_fast_xor, apply_fast_pow, apply_fast_mod, apply_fast_ge, \
    apply_fast_gt, apply_fast_le, apply_fast_lt


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
        return apply_floordiv(self, y)

    def __rfloordiv__(self, y):
        return apply_floordiv(self, y)

    def __pow__(self, y):
        return apply_pow(self, y)

    def __mod__(self, y):
        return apply_mod(self, y)

    def __lt__(self, y):
        return apply_lt(self, y)

    def __le__(self, y):
        return apply_le(self, y)

    def __gt__(self, y):
        return apply_gt(self, y)

    def __ge__(self, y):
        return apply_ge(self, y)

    def __eq__(self, y):
        return apply_eq(self, y)

    def __ne__(self, y):
        return apply_ne(self, y)

    def __and__(self, y):
        return apply_and(self, y)

    def __or__(self, y):
        return apply_or(self, y)

    def __xor__(self, y):
        return apply_xor(self, y)

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
    
    
def apply_floordiv(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.floordiv(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_floordiv(left.values, right.values,
                                            left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.floordiv(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.floordiv(left.values, right), left.index)
    else:
        raise ValueError('type not supported')
    

def apply_pow(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.pow(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_pow(left.values, right.values,
                                       left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.pow(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.pow(left.values, right), left.index)
    else:
        raise ValueError('type not supported')


def apply_mod(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.mod(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_mod(left.values, right.values,
                                       left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.mod(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.mod(left.values, right), left.index)
    else:
        raise ValueError('type not supported')
    

def apply_gt(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.gt(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_gt(left.values, right.values,
                                       left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.gt(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.gt(left.values, right), left.index)
    else:
        raise ValueError('type not supported')


def apply_ge(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.ge(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_ge(left.values, right.values,
                                      left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.ge(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.ge(left.values, right), left.index)
    else:
        raise ValueError('type not supported')    


def apply_lt(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.lt(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_lt(left.values, right.values,
                                       left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.lt(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.lt(left.values, right), left.index)
    else:
        raise ValueError('type not supported')    


def apply_le(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.le(left.values, right), left.index)
    elif type(right) == Column:
        result, index = apply_fast_le(left.values, right.values,
                                       left.index, right.index)
        return Column(result, index)

    elif type(right) == int:
        return Column(operator.le(left.values, right), left.index)
    elif type(right) == float:
        return Column(operator.le(left.values, right), left.index)
    else:
        raise ValueError('type not supported')


def apply_and(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.and_(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == Column:
        result, index = apply_fast_and(left.values, right.values,
                                       left.index, right.index)
        return Column(result.astype(np.bool), index)

    elif type(right) == str:
        return Column(operator.and_(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == int:
        return Column(operator.and_(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == float:
        return Column(operator.and_(left.values.astype(np.bool), right),
                      left.index)
    else:
        raise ValueError('type not supported')


def apply_or(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.or_(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == Column:
        result, index = apply_fast_or(left.values, right.values,
                                      left.index, right.index)
        return Column(result.astype(np.bool), index)

    elif type(right) == str:
        return Column(operator.or_(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == int:
        return Column(operator.or_(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == float:
        return Column(operator.or_(left.values.astype(np.bool), right),
                      left.index)
    else:
        raise ValueError('type not supported')


def apply_xor(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.xor(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == Column:
        result, index = apply_fast_xor(left.values, right.values,
                                       left.index, right.index)
        return Column(result.astype(np.bool), index)

    elif type(right) == str:
        return Column(operator.xor(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == int:
        return Column(operator.xor(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == float:
        return Column(operator.xor(left.values.astype(np.bool), right),
                      left.index)
    else:
        raise ValueError('type not supported')

    
def apply_eq(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.eq(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == Column:
        result, index = apply_fast_eq(left.values, right.values,
                                      left.index, right.index)
        return Column(result.astype(np.bool), index)

    elif type(right) == str:
        return Column(operator.eq(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == int:
        return Column(operator.eq(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == float:
        return Column(operator.eq(left.values.astype(np.bool), right),
                      left.index)
    else:
        raise ValueError('type not supported')

    
def apply_ne(left: Column, right: Column):
    if type(right) == np.ndarray:
        return Column(operator.ne(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == Column:
        result, index = apply_fast_ne(left.values, right.values,
                                      left.index, right.index)
        return Column(result.astype(np.bool), index)

    elif type(right) == str:
        return Column(operator.ne(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == int:
        return Column(operator.ne(left.values.astype(np.bool), right),
                      left.index)
    elif type(right) == float:
        return Column(operator.ne(left.values.astype(np.bool), right),
                      left.index)
    else:
        raise ValueError('type not supported')

    
