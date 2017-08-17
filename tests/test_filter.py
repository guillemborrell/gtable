from gtable import Table
import numpy as np
import pandas as pd


def test_filter_1():
    a = np.arange(10, dtype=np.double)
    table_a = Table({'a': a, 'b': a})
    assert np.all(
        table_a.filter(table_a.a > 5).a.values == np.array([6, 7, 8, 9]))


def test_filter_2():
    a = np.arange(10, dtype=np.double)
    table_a = Table({'a': a, 'b': a})
    assert np.all(
        table_a.filter(
            (table_a.a > 5) & (table_a.b <= 6)).a.values == np.array([6])
    )


def test_filter_3():
    t = Table()
    t.a = np.random.rand(10)
    t.b = pd.date_range('2000-01-01', freq='M', periods=10)
    t.c = np.array([1, 2])
    t.add_column('d', np.array([1, 2]), align='bottom')
    t1 = t.filter(t.c > 0)

    assert t1.c.values[0] == 1
    assert np.all(t1.d.values == np.array([]))
