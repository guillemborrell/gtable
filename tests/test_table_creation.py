from gtable import Table
import numpy as np
import pytest
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
    t = Table({'a': [1, 2, 3], 'b': np.array([4,5,6])})
    t1 = Table({'a': [1, 2, 3], 'd': np.array([4,5,6])})
    t.stitch(t1)

    assert np.all(t.index == np.array([[1, 1, 1, 1, 1, 1],
                                       [1, 1, 1, 0, 0, 0],
                                       [0, 0, 0, 1, 1, 1]], dtype=np.uint8))


def test_records():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t1 = Table({'a': [1, 2, 3], 'd': np.array([4, 5, 6])})
    t.stitch(t1)

    records = [r for r in t.records()]

    assert records == [
        {'a': 1, 'b': 4},
        {'a': 2, 'b': 5},
        {'a': 3, 'b': 6},
        {'a': 1, 'd': 4},
        {'a': 2, 'd': 5},
        {'a': 3, 'd': 6}]

    
def test_compute_column():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t1 = Table({'a': [1, 2, 3], 'd': np.array([4, 5, 6])})
    t.stitch(t1)

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
    t.stitch(t1)

    with pytest.raises(ValueError) as excinfo:
        t.c = t.a + t.b/2
    assert 'broadcast together' in str(excinfo.value)


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
    tb.stitch(tb1)
    assert np.all(tb.index == np.array([[1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1],
                                        [1, 0, 0, 1, 0, 0]], dtype=np.uint8))
    assert np.all(tb.schedule.values == np.array(['first', 'secon']))
