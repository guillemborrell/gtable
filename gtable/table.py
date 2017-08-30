import numpy as np
import pandas as pd

from gtable.column import Column
from gtable.lib import records, stack_table_inplace, add_column, \
    merge_table, sort_table, filter_table, dropnan_table, first_record, \
    last_record, fillna_column


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
                # TODO: Remove ASAP
                # You may get a list of Timestamps. Specific to NFQ
                if type(v[0]) == pd.Timestamp:
                    self.data.append(pd.DatetimeIndex(v).values)
                else:
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
                self.data.append(np.array(v))
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
        """
        Returns a copy of the table
        """
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

    def del_column(self, k):
        """
        Column deletion
        """
        del self[k]
        idx = self.keys.index(k)
        self.keys.pop(idx)
        self.index = np.delete(self.index, idx, axis=0)

    def stack(self, table):
        """Vertical (Table) concatenation."""
        stack_table_inplace(self, table)

    def merge(self, table, column):
        """Merge two tables using two dense and sorted columns"""
        self.data, self.keys, self.index = merge_table(table, self, column)

    def records(self, fill=False):
        """Generator that returns a dictionary for each row of the table"""
        yield from records(self, fill)

    def sort_by(self, column):
        """Sorts by values of a column"""
        sort_table(self, column)

    def filter(self, predicate):
        """Filter table using a column specification or predicate"""
        t = Table()
        t.data, t.keys, t.index = filter_table(self, predicate)
        return t

    def first_record(self, fill=False):
        """Returns the first record of the table"""
        return first_record(self, fill)

    def last_record(self, fill=False):
        """Returns the last record of the table"""
        return last_record(self, fill)

    def to_pandas(self, fill=False):
        """Translate the table to a pandas dataframe"""
        return pd.DataFrame.from_records(self.records(fill))

    def to_dict(self):
        """Translate the table to a dict {key -> array_of_values}"""
        return {k: v for k, v in zip(self.keys, self.data)}

    def dropnan(self, clip=False):
        """Drop the NaNs and leave missing values instead"""
        dropnan_table(self)

    def get(self, key):
        """Gets a column"""
        return Column(self.data[self.keys.index(key)], self._index_column(key))

    def fillna_column(self, key, reverse=False, fillvalue=None):
        """
        Fillna on a column inplace

        :param key:
        :param reverse:
        :param fillvalue:
        :return:
        """
        self[key], self.index[self.keys.index(key), :] = fillna_column(
            self[key], self._index_column(key), reverse, fillvalue)


    @classmethod
    def from_pandas(cls, dataframe):
        """Create a table from a pandas dataframe"""
        table = {'idx': dataframe.index.values}
        table.update({k: dataframe[k].values for k in dataframe})
        return cls(table)

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

    def __contains__(self, item):
        return item in self.keys

    @staticmethod
    def __dir__():
        return ['copy', 'add_column', 'stitch', 'merge', 'records', 'to_pandas',
                'to_dict', 'filter', 'dropnan']
    
    def __getattr__(self, key):
        return Column(self.data[self.keys.index(key)], self._index_column(key))

    def __getitem__(self, key):
        return self.data[self.keys.index(key)]

    def __setitem__(self, key, value):
        self.data[self.keys.index(key)] = value

    def __delitem__(self, key):
        del self.data[self.keys.index(key)]

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

