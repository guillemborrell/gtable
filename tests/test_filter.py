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


def test_filter_contains():
    t = Table()
    t.add_column('a', np.arange(10))
    t.add_column('b', np.arange(5))

    filt = t.a.contains(t.b)
    assert np.all(t.filter(filt).a.values == t.b.values)

    filt = t.a.contains(t.b.values)
    assert np.all(t.filter(filt).a.values == t.b.values)

    assert len(t.filter(filt)) == 5

    filt = t.a.contains(np.array([11]))
    assert t.filter(filt).a.is_empty()


def test_filter_date():
    t = Table()
    t.a = np.random.rand(10)
    t.b = pd.date_range('2000-01-01', freq='D', periods=10)
    t.c = np.array([1, 2])
    t.add_column('d', np.array([1, 2]), align='bottom')

    thres1 = np.array(['2000-01-03'], dtype=np.datetime64)
    thres2 = np.array(['2000-01-05'], dtype=np.datetime64)
    t1 = t.filter(t.b >= thres1)
    assert np.all(t1.c.values == np.array([]))
    assert np.all(t1.d.values == np.array([1, 2]))
    assert np.all(t1.a.values == t.a.values[2:])

    t1 = t.filter((t.b >= thres1) & (t.b <= thres2))
    assert np.all(t1.c.values == np.array([]))
    assert np.all(t1.d.values == np.array([]))
    assert np.all(t1.a.values == t.a.values[2:5])

    t1 = t.filter(t.b.date_range(fr=thres1, to=thres2))
    assert np.all(t1.c.values == np.array([]))
    assert np.all(t1.d.values == np.array([]))
    assert np.all(t1.a.values == t.a.values[2:5])


