from gtable import Table
import numpy as np


def test_add():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    assert np.all((t.a + t.a).values == np.array([2, 4, 6]))


def test_mul():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    assert np.all((t.a * t.a).values == np.array([1, 4, 9]))


def test_add_align():
    t = Table()
    t.add_column('a', [1, 2, 3, 4, 5, 6], dtype=np.int64)
    t.add_column('b', [1, 2, 3], align='bottom', dtype=np.int64)

    c = t.a + t.b

    assert np.all(c.values == np.array([5, 7, 9]))
    assert t.a.values.dtype == np.int64
    assert c.values.dtype == np.int64


def test_column_assign():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t['a'][0] = 0

    assert np.all(t.a.values == np.array([0, 2, 3]))


def test_date_range():
    t = Table(
        {'a': np.array([0, 1, 2, 3, 4, 5, 6]).astype('datetime64[D]'),
         'b': np.array([1, 2, 3, 4, 5, 6, 7])}
    )

    assert np.count_nonzero(
        t.a.date_range(to='1970-01-05').values) == 5
    assert np.count_nonzero(
        t.a.date_range(to='1970-01-05', include_to=False).values) == 4
    assert np.count_nonzero(
        t.a.date_range(fr='1970-01-05').values) == 3
    assert np.count_nonzero(
        t.a.date_range(fr='1970-01-05', include_fr=False).values) == 2
