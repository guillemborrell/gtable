import numpy as np
import pandas as pd
import sys
from collections import OrderedDict
from itertools import chain


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

    def __pow__(self, y):
        if type(y) == int:
            return Column(self.values ** y, self.index)
        elif type(y) == float:
            return Column(self.values ** y, self.index)
        elif type(y) == np.ndarray:
            return Column(self.values ** y, self.index)
        elif type(y) == Column:
            return Column(self.values ** y.values, self.index)


class Table:
    """
    Table is a class for fast columnar storage using a bitmap index for
    sparse storage
    """
    def copy(self):
        t = Table()
        t._data = OrderedDict([(k, v.copy()) for k, v in self._data.items()])
        t._index = self._index.copy()
        if self._order is not None:
            t._order = self._order.copy()

        return t
    
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

    def _index_column(self, key):
        # This only works with OrderedDicts
        for i, k in enumerate(self._data):
            if key == k:
                return self._index[i, :]

    def _index_index(self, key):
        # This only works with OrderedDicts
        for i, k in enumerate(self._data):
            if key == k:
                return i

    def __init__(self, data={}):
        # This dictionary stores the columns
        self._data = OrderedDict()
        # This is the index bitmap
        self._index = None
        # This is the order of the table
        self._order = None
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

        # Create the index and the ordered arrays
        self._index = np.ones((len(data), length_last), dtype=np.uint8)
            
    def __repr__(self):
        column_info = list()
        for k, v in self._data.items():
            if type(v) == np.ndarray:
                column_type = v.dtype
            else:
                column_type = 'object'        
            count = np.count_nonzero(self._index_column(k))
            column_info.append('{}[{}] <{}>'.format(k, count, column_type))
            
        return "<Table[ {} ] object at {}>".format(', '.join(column_info),hex(id(self)))
    
    def __getattr__(self, key):
        return Column(self._data[key], self._index_column(key))
    
    def hcat(self, k, v, index=None):
        """
        Column concatenation.
        """
        if k in self._data:
            raise KeyError("Key {} already present".format(k))

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
            if len(v) > self._index.shape[1]:
                self._index = np.concatenate(
                    [self._index,
                     np.zeros(
                         (self._index.shape[0],
                          len(v) - self._index.shape[1]),
                         dtype=np.uint8)],
                    axis=1
                )
                
            # Concatenate the shape of the array to the bitmap
            index_stride = np.zeros((1, self._index.shape[1]), dtype=np.uint8)
            index_stride[0, :len(v)] = 1
            self._index = np.concatenate([self._index, index_stride])
            
        else:
            # Handle the fact that the new column my be longer, so extend bitmap
            if index.shape[0] > self._index.shape[1]:
                self._index = np.concatenate(
                    [self._index,
                     np.zeros(
                         (self._index.shape[0],
                        index.shape[0] - self._index.shape[0]),
                         dtype=np.uint8)],
                    axis=1
                )

            # Concatenate the new column to the bitmap.
            self._index = np.concatenate([self._index, np.atleast_2d(index)])


    def vcat(self, table):
        """Vertical (Table) concatenation."""
        # First step is to rearrange the bitmap index if needed
        joined_columns = set(chain(self._data, table._data))
        hspill = len(joined_columns) - self._index.shape[0]
        before_growth = self._index.shape

        tindex = table._index.copy()
        
        # Add the horizontal spill (more columns)
        if joined_columns != set(self._data):
            self._index = np.concatenate(
                [self._index, np.zeros(
                    (hspill, self._index.shape[1]), dtype=np.uint8
                )])

        # Add the vertical spill (the data)
        self._index = np.concatenate(
            [self._index, np.zeros(
                (self._index.shape[0], tindex.shape[1]), dtype=np.uint8
            )], axis=1)

        # Include the keys present in both tables with this light nested loop.
        for i, old_key in enumerate(self._data):
            for j, new_key in enumerate(table._data):
                if new_key == old_key:
                    self._index[i, before_growth[1]:] = tindex[j, :]
                    self._data[old_key] = np.concatenate(
                        [self._data[old_key], table._data[new_key]]
                        )

        # Include keys that are not added in the previous table
        new_cols_added = 0
        for i, new_key in enumerate(table._data):
            if new_key not in self._data:
                self._index[before_growth[0] + new_cols_added,
                            before_growth[1]:] = tindex[i, :]
                self._data[new_key] = table._data[new_key]
                new_cols_added += 1

                
    def records(self, fill=False):
        """Generator that returns a dictionary for each row of the table"""
        # Infinite counter. SLOOOOOOW. This is columnar storage.
        counters = np.zeros((self._index.shape[0]), dtype=np.int)
        keys = np.array([k for k in self._data])
        
        if self._order is not None:
            for record in self._index.T[self._order]:
                selected_keys = keys[np.where(record)]
                selected_counters = counters[np.where(record)]
                selected_values = list()
                for k, c in zip(selected_keys, selected_counters):
                    selected_values.append(self._data[k][c])
                counters[np.where(record)] += 1
            
                if fill:
                    rec = {k: v for k, v in zip(selected_keys, selected_values)}
                    remaining = set(keys) - set(selected_keys)
                    rec.update({k: np.nan for k in remaining})
                    yield rec
                else:
                    yield {k: v for k, v in zip(selected_keys, selected_values)}
        else:
            for record in self._index.T:
                selected_keys = keys[np.where(record)]
                selected_counters = counters[np.where(record)]
                selected_values = list()
                for k, c in zip(selected_keys, selected_counters):
                    selected_values.append(self._data[k][c])
                counters[np.where(record)] += 1
            
                if fill:
                    rec = {k: v for k, v in zip(selected_keys, selected_values)}
                    remaining = set(keys) - set(selected_keys)
                    rec.update({k: np.nan for k in remaining})
                    yield rec
                else:
                    yield {k: v for k, v in zip(selected_keys, selected_values)}


    def __setattr__(self, key, value):
        if key.startswith('_'):
            self.__dict__[key] = value
        else:
            if type(value) == Column:
                if key in self._data:
                    self._data[key] = value.values
                    self._index[self._index_index(key), :] = value.index
                else:
                    self.hcat(key, value.values, value.index)

            elif type(value) == np.ndarray:
                if key in self._data:
                    self._data[key] = value
                else:
                    self.hcat(key, value)

    def to_pandas(self):
        return pd.DataFrame.from_records(self.records())
    
    @property
    def data(self):
        return self._data

    @property
    def index(self):
        return self._index
