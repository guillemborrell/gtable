import numpy as np
import pandas as pd
from functools import partial

from gtable.column import Column
from gtable.lib import records, stack_table_inplace, add_column, \
    merge_table, sort_table, filter_table, dropnan_table, first_record, \
    last_record, fillna_column, from_chunks, required_columns, required_column


def _check_length(i, k, this_length, length_last):
    if i == 0:
        length_last = this_length
    else:
        if this_length != length_last:
            raise ValueError("Column {} length mismatch".format(k))
        else:
            length_last = this_length

    return length_last


def get_reductor(out_check_sorted):
    from gtable.reductions import reduce_funcs, reduce_by_key

    class ReductorByKey:
        @staticmethod
        def __dir__():
            return [f for f in reduce_funcs]

        def __init__(self, table, column, check_sorted=out_check_sorted):
            for reduction_f in reduce_funcs:
                self.__dict__[reduction_f] = partial(
                    reduce_by_key, table, column, reduction_f, check_sorted)

    return ReductorByKey


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
                if len(v) > 0 and type(v[0]) == pd.Timestamp:
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

    def add_column(self, k, v, dtype=None, index=None, align='top'):
        """
        Column concatenation.
        """
        add_column(self, k, v, dtype, index, align=align)

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

    def sieve(self, idx):
        """Filter table using a one-dimensional array of boolean values"""
        t = Table()
        # This could be improved, but added as syntactic sugar ATM.
        t.data, t.keys, t.index = filter_table(self, Column(idx.astype(np.int8), np.ones_like(idx)))
        return t

    def crop(self, key):
        """Purge the records where the column key is empty"""
        t = Table()
        col = self.get(key)
        predicate = (col == col)
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

    def get(self, key, copy=False):
        """Gets a column or a table with columns"""
        if type(key) == str:
            return Column(self[key], self._index_column(key))

        elif type(key) == list or type(key) == tuple:
            t = Table()
            indices = [self.keys.index(k) for k in key]
            if copy:
                t.data = [self.data[idx].copy() for idx in indices]
                t.index = self.index[indices, :][:, :]
            else:
                t.data = [self.data[idx].copy() for idx in indices]
                t.index = self.index[indices, :]
            t.keys = key

            return t

    def fillna_column(self, key, reverse=False, fillvalue=None):
        """
        Fillna on a column inplace

        :param key: string or list
        :param reverse:
        :param fillvalue:
        :return:
        """
        if (type(key) == list) or (type(key) == tuple):
            for k in key:
                self[k], self.index[self.keys.index(k), :] = fillna_column(
                    self[k], self._index_column(k), reverse, fillvalue)

        else:
            self[key], self.index[self.keys.index(key), :] = fillna_column(
                self[key], self._index_column(key), reverse, fillvalue)

    def fill_column(self, key, fillvalue):
        """
        Fill N/A elements in the given columns with fillvalue

        :param key: String, list or tuple with the column names to be filled.
        :param fillvalue: Scalar to fill the N/A elements
        :return:
        """
        if (type(key) == list) or (type(key) == tuple):
            for k in key:
                col = getattr(self, k)
                col.fill(fillvalue)
                setattr(self, k, col)
        else:
                col = getattr(self, key)
                col.fill(fillvalue)
                setattr(self, key, col)

    def reduce_by_key(self, column, check_sorted=False):
        """
        Reduce by key

        :param column:
        :param check_sorted:
        :return:
        """
        return get_reductor(check_sorted)(self, column)

    def required_column(self, key, dtype):
        """
        Enforce the required column with a dtype

        :param key:
        :param dtype:
        :return:
        """
        required_column(self, key, dtype)

    def required_columns(self, *args):
        """
        Enforce the required columns. Create empty columns if necessary.

        :param args:
        :return:
        """
        required_columns(self, *args)

    def rename_column(self, old_name, new_name):
        """
        Rename a column of the table

        :param old_name:
        :param new_name:
        :return:
        """
        idx = self.keys.index(old_name)
        if new_name not in self.keys:
            self.keys[idx] = new_name
        else:
            raise ValueError('Column names must be unique')

    @classmethod
    def from_pandas(cls, dataframe):
        """Create a table from a pandas dataframe"""
        table = cls()

        if np.all(np.isfinite(dataframe.index.values)):
            table.add_column('idx', dataframe.index.values)
        else:
            raise ValueError('Dataframe index must not contain NaNs')

        for k in dataframe:
            if dataframe[k].values.dtype == np.dtype('O'):
                table.add_column(k, np.array(list(dataframe[k].values)))
            elif dataframe[k].values.dtype == np.dtype('datetime64[ns]'):
                nidx = dataframe[k].values == np.datetime64('NaT')
                table.add_column(k, dataframe[k].values[~nidx], dtype=dataframe[k].values.dtype, index=~nidx)
            else:
                nidx = np.isnan(dataframe[k].values)
                table.add_column(k, dataframe[k].values[~nidx], dtype=dataframe[k].values.dtype, index=~nidx)

        return table

    @staticmethod
    def from_chunks(chunks):
        """
        Create a table from table chunks

        :param chunks:
        :return:
        """
        return from_chunks(chunks)

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
        return []
    
    def __getattr__(self, key):
        return Column(self.data[self.keys.index(key)], self._index_column(key))

    def __getitem__(self, key):
        return self.data[self.keys.index(key)]

    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray):
            self.data[self.keys.index(key)] = value
        else:
            raise ValueError('Direct assignment only valid with Numpy arrays')

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

    def __getstate__(self):

        index = self.index.copy()
        data = [d.copy() for d in self.data]
        keys = self.keys[:]

        return index, data, keys

    def __setstate__(self, state):

        index, data, keys = state
        self.index = index
        self.data = data
        self.keys = keys

    def __len__(self):
        return self.index.shape[1]
