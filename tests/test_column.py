from gtable import Table
import numpy as np


def test_add():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    assert np.all((t.a + t.a).values == np.array([2, 4, 6]))


def test_mul():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    assert np.all((t.a * t.a).values == np.array([1, 4, 9]))
