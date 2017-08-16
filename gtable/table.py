import numpy as np
import pandas as pd
from gtable.lib import fillna_column, records, stitch_table, add_column,\
    merge_table, sort_table, filter_table


class Column:
    """
    Indexed column view of the table
    """
    def __init__(self, values, index):
        self.values = values
        self.index = index

    def __repr__(self):
        return "<Column[ {} ] object at {}>".format(self.values.dtype, hex(id(self)))

    def __add__(self, y):
        if type(y) == np.ndarray:
            return Column(self.values + y, self.index)
        elif type(y) == Column:
            return Column(self.values + y.values, self.index)
        elif type(y) == int:
            return Column(self.values + y, self.index)
        elif type(y) == float:
            return Column(self.values + y, self.index)
        else:
            raise ValueError('Type not supported')

    def __sub__(self, y):
        if type(y) == np.ndarray:
            return Column(self.values - y, self.index)
        elif type(y) == Column:
            return Column(self.values - y.values, self.index)
        elif type(y) == int:
            return Column(self.values - y, self.index)
        elif type(y) == float:
            return Column(self.values - y, self.index)
        else:
            raise ValueError('Type not supported')
    
    def __mul__(self, y):
        if type(y) == int:
            return Column(self.values * y, self.index)
        elif type(y) == float:
            return Column(self.values * y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values * y, self.index)
        elif type(y) == Column:
            return Column(self.values * y.values, self.index)

    def __truediv__(self, y):
        if type(y) == int:
            return Column(self.values / y, self.index)
        elif type(y) == float:
            return Column(self.values / y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values / y, self.index)
        elif type(y) == Column:
            return Column(self.values / y.values, self.index)

    def __pow__(self, y):
        if type(y) == int:
            return Column(self.values ** y, self.index)
        elif type(y) == float:
            return Column(self.values ** y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values ** y, self.index)
        elif type(y) == Column:
            return Column(self.values ** y.values, self.index)

    def __and__(self, y):
        if type(y) == int:
            return Column(self.values & y, self.index)
        elif type(y) == float:
            return Column(self.values & y, self.index)
        elif type(y) == np.ndarray:
            return Column(np.bitwise_and(self.values, y), self.index)
        elif type(y) == Column:
            return Column(np.bitwise_and(self.values, y.values), self.index)

    def __or__(self, y):
        if type(y) == int:
            return Column(self.values | y, self.index)
        elif type(y) == float:
            return Column(self.values | y, self.index)
        elif type(y) == np.ndarray:
            return Column(np.bitwise_or(self.values, y), self.index)
        elif type(y) == Column:
            return Column(np.bitwise_or(self.values, y.values), self.index)

    def __xor__(self, y):
        if type(y) == int:
            return Column(self.values ^ y, self.index)
        elif type(y) == float:
            return Column(self.values ^ y, self.index)
        elif type(y) == np.ndarray:
            return Column(np.bitwise_xor(self.values, y), self.index)
        elif type(y) == Column:
            return Column(np.bitwise_xor(self.values, y.values), self.index)

    def __lt__(self, y):
        if type(y) == int:
            return Column(self.values < y, self.index)
        elif type(y) == float:
            return Column(self.values < y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values < y, self.index)
        elif type(y) == Column:
            return Column(self.values < y.values, self.index)

    def __le__(self, y):
        if type(y) == int:
            return Column(self.values <= y, self.index)
        elif type(y) == float:
            return Column(self.values <= y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values <= y, self.index)
        elif type(y) == Column:
            return Column(self.values <= y.values, self.index)

    def __gt__(self, y):
        if type(y) == int:
            return Column(self.values > y, self.index)
        elif type(y) == float:
            return Column(self.values > y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values > y, self.index)
        elif type(y) == Column:
            return Column(self.values > y.values, self.index)

    def __ge__(self, y):
        if type(y) == int:
            return Column(self.values >= y, self.index)
        elif type(y) == float:
            return Column(self.values >= y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values >= y, self.index)
        elif type(y) == Column:
            return Column(self.values >= y.values, self.index)

    def __eq__(self, y):
        if type(y) == str:
            return Column(self.values == y, self.index)
        if type(y) == int:
            return Column(self.values == y, self.index)
        elif type(y) == float:
            return Column(self.values == y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values == y, self.index)
        elif type(y) == Column:
            return Column(self.values == y.values, self.index)

    def __ne__(self, y):
        if type(y) == int:
            return Column(self.values != y, self.index)
        elif type(y) == float:
            return Column(self.values != y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values != y, self.index)
        elif type(y) == Column:
            return Column(self.values != y.values, self.index)

    def fillna(self, reverse=False, fillvalue=None):
        """
        Fills the non available value sequentially with the previous
        available position.
        """
        self.values, self.index = fillna_column(self.values,
                                                self.index,
                                                reverse,
                                                fillvalue)


def _check_length(i, k, this_length, length_last):
    if i == 0:
        length_last = this_length
    else:
        if this_length != length_last:
            raise ValueError("Column {} length mismatch".format(k))
        else:
            length_last = this_length

    return length_last


class Table:
    """
    Table is a class for fast columnar storage using a bitmap index for
    sparse storage
    """
    def __init__(self, data={}):
        # This list stores the keys
        self.keys = []
        # This list stores the columns
        self.data = []
        # This is the index bitmap
        self.index = None
        length_last = 0

        # Creating the table only supports assigning a single index
        for i, (k, v) in enumerate(data.items()):
            # If the column is a list, cast it to a numpy array
            if type(v) == list:
                self.data.append(np.array(v))
                self.keys.append(k)
                length_last = _check_length(i, k, len(v), length_last)
                    
            elif type(v) == np.ndarray:
                if not len(v.shape) == 1:
                    raise ValueError("Only 1D arrays supported")
                self.data.append(v)
                self.keys.append(k)
                length_last = _check_length(i, k, v.shape[0], length_last)

            # Pandas DatetimeIndex is supported for convenience.
            elif type(v) == pd.DatetimeIndex:
                self.data.append(v.values)
                self.keys.append(k)
                length_last = _check_length(i, k, v.shape[0], length_last)
                
            else:
                raise ValueError("Column type not supported")

        # Create the index and the ordered arrays
        self.index = np.ones((len(data), length_last), dtype=np.uint8)

    def _repr_html_(self):
        return "<i>xxx</i>"

    def _index_column(self, key):
        return self.index[self.keys.index(key), :]

    def copy(self):
        t = Table()
        t.data = [d.copy() for d in self.data]
        t.keys = self.keys[:]
        t.index = self.index.copy()

        return t

    def add_column(self, k, v, index=None, align='top'):
        """
        Column concatenation.
        """
        add_column(self, k, v, index, align=align)

    def stitch(self, table):
        """Vertical (Table) concatenation."""
        stitch_table(self, table)

    def merge(self, table, column):
        self.data, self.keys, self.index = merge_table(table, self, column)

    def records(self):
        """Generator that returns a dictionary for each row of the table"""
        yield from records(self)

    def sort_by(self, column):
        """Sorts by values of a column"""
        sort_table(self, column)

    def filter(self, predicate):
        t = Table()
        t.data, t.keys, t.index = filter_table(self, predicate)
        return t

    def to_pandas(self):
        return pd.DataFrame.from_records(self.records())

    def to_dict(self):
        return {k: v for k, v in zip(self.keys, self.data)}

    def __repr__(self):
        column_info = list()
        for k, v in zip(self.keys, self.data):
            if type(v) == np.ndarray:
                column_type = v.dtype
            else:
                column_type = 'object'
            count = np.count_nonzero(self._index_column(k))
            column_info.append('{}[{}] <{}>'.format(k, count, column_type))

        return "<Table[ {} ] object at {}>".format(', '.join(column_info),
                                                   hex(id(self)))

    @staticmethod
    def __dir__():
        return ['copy', 'add_column', 'stitch', 'merge', 'records', 'to_pandas',
                'to_dict']
    
    def __getattr__(self, key):
        return Column(self.data[self.keys.index(key)], self._index_column(key))

    def __getitem__(self, key):
        return self.data[self.keys.index(key)]

    def __setitem__(self, key, value):
        self.data[self.keys.index(key)] = value

    def __setattr__(self, key, value):
        if key in ['data', 'keys', 'index']:
            self.__dict__[key] = value
        else:
            if type(value) == Column:
                if key in self.keys:
                    self.data[self.keys.index(key)] = value.values
                    self.index[self.keys.index(key), :] = value.index
                else:
                    self.add_column(key, value.values, value.index)

            elif type(value) == np.ndarray:
                if key in self.keys:
                    self.data[self.keys.index(key)] = value
                else:
                    self.add_column(key, value)

            elif type(value) == pd.DatetimeIndex:
                if key in self.keys:
                    self.data[self.keys.index(key)] = value.values
                else:
                    self.add_column(key, value)

