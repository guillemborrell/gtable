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
    t.add_column('a', [1, 2, 3, 4, 5, 6])
    t.add_column('b', [1, 2, 3], align='bottom')

    c = t.a + t.b

    assert np.all(c.values == np.array([5, 7, 9]))
    assert c.values.dtype == np.int64
