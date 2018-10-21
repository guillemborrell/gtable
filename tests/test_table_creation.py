from gtable import Table
import numpy as np
import pandas as pd


def test_empty_table():
    t = Table()
    assert t.data == []

    
def test_simple_table():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})

    assert t.to_dict()['a'][2] == 3
    assert np.all(t.index == np.ones((2, 3), dtype=np.uint8))

    
def test_hcat_table():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t.add_column('c', [7, 8, 9])

    assert np.all(t.index == np.ones((3, 3), dtype=np.uint8))
    assert np.all(t.c.values == np.array([7, 8, 9]))

    t.add_column('d', [0, 1, 2, 3, 4])
    assert np.all(t.index == np.array([[1, 1, 1, 0, 0],
                                       [1, 1, 1, 0, 0],
                                       [1, 1, 1, 0, 0],
                                       [1, 1, 1, 1, 1]], dtype=np.uint8))


def test_vcat_table():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t1 = Table({'a': [1, 2, 3], 'd': np.array([4, 5, 6])})
    t.stack(t1)

    assert np.all(t.index == np.array([[1, 1, 1, 1, 1, 1],
                                       [1, 1, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 1]], dtype=np.uint8))


def test_compute_column():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t1 = Table({'a': [1, 2, 3], 'd': np.array([4, 5, 6])})
    t.stack(t1)

    t.c = t.a + t.a/2
    records = [r for r in t.records()]
    assert records == [
        {'a': 1, 'b': 4, 'c': 1.5},
        {'a': 2, 'b': 5, 'c': 3.0},
        {'a': 3, 'b': 6, 'c': 4.5},
        {'a': 1, 'd': 4, 'c': 1.5},
        {'a': 2, 'd': 5, 'c': 3.0},
        {'a': 3, 'd': 6, 'c': 4.5}]


def test_compute_wrong_size():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t1 = Table({'a': [1, 2, 3], 'd': np.array([4, 5, 6])})
    t.stack(t1)

    t.c = t.a + t.b/2
    assert np.all(t.c.values == np.array([3, 4.5, 6]))
    assert np.all(t.c.index == np.array([1, 1, 1, 0, 0, 0]))


def test_add_array():
    t = Table()
    t.a = np.arange(10)
    assert t.__repr__()[:13] == "<Table[ a[10]"


def test_add_one():
    tb = Table({'a': pd.date_range('2000-01-01', freq='M', periods=10),
                'b': np.random.randn(10)})
    tb.add_column('schedule', np.array(['first']))
    assert np.all(tb.index == np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))


def test_vcat_heterogeneous():
    tb = Table({'a': pd.date_range('2000-01-01', freq='M', periods=3),
                'b': np.random.randn(3)})
    tb.add_column('schedule', np.array(['first']))
    tb1 = tb.copy()
    tb1.schedule.values[0] = 'second'
    tb.stack(tb1)
    assert np.all(tb.index == np.array([[1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1],
                                        [1, 0, 0, 1, 0, 0]], dtype=np.uint8))
    assert np.all(tb.schedule.values == np.array(['first', 'secon']))


def test_from_pandas():
    df = pd.DataFrame(
        {'a': [1, 2, 3, 4],
         'b': np.arange(4, dtype=np.float64),
         'c': pd.date_range('2002-01-01', periods=4, freq='M')}
    )
    t = Table.from_pandas(df)
    assert np.all(t.a.values == df.a.values)
    assert np.all(t.b.values == df.b.values)
    assert np.all(t.c.values == df.c.values)
    assert np.all(t.idx.values == df.index.values)


def test_from_pandas_sparse():
    df = pd.DataFrame(
        {'a': [1, 2, 3, np.nan],
         'b': np.arange(4, dtype=np.float64),
         'c': pd.date_range('2002-01-01', periods=4, freq='M')}
    )
    t = Table.from_pandas(df)

    assert np.all(t.index == np.array(
            [[1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1]], dtype=np.uint8))

    assert np.all(t.a.values == np.array([1,2,3], dtype=np.float64))


def test_simple_rename():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t.rename_column('a', 'c')

    assert t.keys == ['c', 'b']
