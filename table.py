import numpy as np
import pandas as pd
import sys
from collections import defaultdict


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
        if type(y) == int:
            return Column(self.values + y, self.index)
        elif type(y) == float:
            return Column(self.values + y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values + y, self.index)
        elif type(y) == Column:
            return Column(self.values + y.values, self.index)

    def __sub__(self, y):
        if type(y) == int:
            return Column(self.values - y, self.index)
        elif type(y) == float:
            return Column(self.values - y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values - y, self.index)
        elif type(y) == Column:
            return Column(self.values - y.values, self.index)
    
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


class Table:
    """
    Table is a class for fast columnar storage
    """
    @staticmethod
    def _check_length(i, k, this_length, length_last):
        if i == 0:
            length_last = this_length
        else:
            if this_length != length_last:
                raise ValueError("Column {} length mismatch".format(k))
            else:
                length_last = this_length

        return length_last

    def _index(self, k):
        return self._index_store[self._indexes[k]]
    
    def __init__(self, data={}):
        # This dictionary stores the columns
        self._data = dict()
        # This dictionary maps the column name with its index
        self._indexes = {}
        # This list stores the indexes
        self._index_store = []

        length_last = 0

        # Creating the table only supports assigning a single index
        for i, (k, v) in enumerate(data.items()):
            # If the column is a list, cast it to a numpy array
            if type(v) == list:
                self._data[k] = np.array(v)
                length_last = self._check_length(i, k, len(v), length_last)
                    
            elif type(v) == np.ndarray:
                if not len(v.shape) == 1:
                    raise ValueError("Only 1D arrays supported")
                self._data[k] = v
                length_last = self._check_length(i, k, v.shape[0], length_last)

            # Pandas DatetimeIndex is supported for convenience.
            elif type(v) == pd.DatetimeIndex:
                self._data[k] = v
                length_last = self._check_length(i, k, v.shape[0], length_last)
                
            else:
                raise ValueError("Column type not supported")

            self._indexes[k] = 0

        # The first index is also the primary index
        self._index_store.append(np.arange(length_last))
            
    def __repr__(self):
        return "<Table[ {} ] object at {}>".format(
            ', '.join([
                '{}[{}] <{}>'.format(
                    k, len(self._index(k)), v.dtype
                ) if type(v) == np.ndarray else '{}[{}] <object>'.format(
                    k, len(self._index(k))) for k, v in self._data.items()]),
            hex(id(self)))
    
    def __getattr__(self, key):
        return Column(self._data[key], self._index(key))

    def hcat(self, k, v, index=None):
        """
        Column concatenation.
        """
        if type(v) == list:
            self._data[k] = np.array(v)
                    
        elif type(v) == np.ndarray:
            if not len(v.shape) == 1:
                raise ValueError("Only 1D arrays supported")
            self._data[k] = v

        elif type(v) == pd.DatetimeIndex:
            self._data[k] = v
                
        else:
            raise ValueError("Column type not supported")

        if index is None:
            # If the length of the value is the same as the first index,
            # just use the first index
            if len(v) == len(self._index_store[0]):
                self._indexes[k] = 0
            
            # The length of the index does not match the old index, so a
            # new one is introduced
            else:
                new_index = len(self._index_store)
                self._indexes[k] = new_index
                self._index_store.append(np.arange(len(v)))
        else:
            new_index = len(self._index_store)
            self._indexes[k] = new_index
            self._index_store.append(index)

    def vcat(self, table):
        """Vertical (Table) concatenation"""
        if len(table._index_store) > 1:
            raise ValueError("vcat only supported with single-index tables")

        # Get the maximum length of all indexes within the table.
        inserted_index = False
        cursor = max(len(v) for v in self._index_store)
        cat_index = table._index_store[0] + cursor

        for k, v in table._data.items():
            if k in self._data:
                self._data[k] = np.concatenate([self._data[k], v])
                new_index = len(self._index_store)
                if len(self._index(k)) < len(self._data[k]):
                    self._index_store.append(
                        np.concatenate([self._index(k),
                                        cat_index])
                        )
                    self._indexes[k] = new_index
                    
            elif not inserted_index:
                self._data[k] = v
                new_index = len(self._index_store)
                self._indexes[k] = new_index
                self._index_store.append(cat_index)
                inserted_index = True

            else:
                self._data[k] = v
                
    def records(self):
        """Generator that returns a dictionary for each row of the table"""
        # Infinite counter. SLOOOOOOW. This is columnar storage.
        # Maybe reimplement in Cython for speed.
        cursors = {k: 0 for k in self._data}
        values = {k: self._index(k)[0] for k in self._data}
        lengths = {k: len(self._index(k)) for k in self._data}

        global_counter = 0
        while sum(cursors.values()) < sum(lengths.values()):
            keys_to_advance = [k for k, v in values.items() if global_counter == v]
            yield {k: self._data[k][cursors[k]] for k in keys_to_advance}
            
            for k in keys_to_advance:
                if cursors[k] < lengths[k] - 1:
                    cursors[k] += 1
                    values[k] = self._index(k)[cursors[k]]
                else:
                    cursors[k] += 1
                    
            global_counter += 1

    def __setattr__(self, key, value):
        if key.startswith('_'):
            self.__dict__[key] = value
        else:
            if type(value) == Column:
                if key in self._data:
                    self._data[key] = value.values
                    self._index_store[self._indexes[key]] = value.index
                else:
                    self._data[key] = value.values
                    new_index = len(self._index_store)
                    self._indexes[key] = new_index
                    self._index_store.append(value.index)

            elif type(value) == np.ndarray:
                if key in self._data:
                    raise ValueError('Column assignment only with Column type')
                else:
                    self._data[key] = value
                    new_index = len(self._index_store)
                    self._indexes[key] = new_index
                    self._index_store.append(np.arange(len(value)))

    @property
    def data(self):
        return self._data
            
if __name__ == '__main__':
    t = Table({'a': [1,2,3], 'b': np.array([4,5,6])})
    print(t)
    print(t.a)
    t.hcat('c', [7,8,9])
    print(t)
    
    t1 = Table({'a': [1,2,3], 'd': np.array([4,5,6])})
    t.vcat(t1)

    for r in t.records():
        print(r)
    
    t.e = t.a + t.a / 2

    for r in t.records():
        print(r)

    print(t)

    t2 = Table()
    print(t2)
    t2.a = np.arange(10)
    print(t2)
