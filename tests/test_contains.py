from gtable import Table
import numpy as np


def test_contains_scalar():
    t = Table()
    t.add_column('a', np.arange(4))
    t.add_column('b', np.arange(2))

    filt = t.a.contains(np.array([11]))
    assert np.all(filt.values == np.array([False, False, False, False]))
    assert np.all(filt.index == np.array([1, 1, 1, 1]))


def test_contains_column():
    t = Table()
    t.add_column('a', np.arange(4))
    t.add_column('b', np.arange(2))

    filt = t.a.contains(t.b)
    assert np.all(filt.values == np.array([True, True, False, False]))
    assert np.all(filt.index == np.array([1, 1, 1, 1]))
