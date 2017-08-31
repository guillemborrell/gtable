import gtable as gt
import numpy as np


def test_get_column():
    t = gt.Table()
    t.add_column('a', [1, 2, 3, 4, 5])
    t.add_column('b', [1, 2, 3])
    t.add_column('c', [3, 4, 5], align='bottom')

    c = t.get('a')

    assert np.all(c.values == np.array([1, 2, 3, 4, 5]))


def test_get_table():
    t = gt.Table()
    t.add_column('a', [1, 2, 3, 4, 5])
    t.add_column('b', [1, 2, 3])
    t.add_column('c', [3, 4, 5], align='bottom')

    t1 = t.get(['a', 'b'])
    print(t1.index)

    assert np.all(t1.a.values == np.array([1, 2, 3, 4, 5]))
    assert np.all(t1.a.index == np.array([1, 1, 1, 1, 1]))
