from gtable import Table
from gtable.joins import inner_join
import numpy as np


def test_inner_join_1():
    t1 = Table()
    t1.add_column('a', [1, 2, 3, 4, 5, 6])
    t1.add_column('b', [1, 2, 3])

    t2 = Table()
    t2.add_column('a', [2, 3, 4])
    t2.add_column('b', [1, 1, 1])
    t2.add_column('c', [5, 6, 7])

    t3 = inner_join(t1, t2, 'a')

    assert np.all(t3.a.values == np.array([2, 3, 4]))
    assert np.all(t3.b.values == np.array([2, 3]))
    assert np.all(t3.c.values == np.array([5, 6, 7]))


def test_inner_join_2():
    t1 = Table()
    t1.add_column('a', [1, 2, 2, 3, 3, 4, 5, 6])
    t1.add_column('b', [1, 2, 3])

    t2 = Table()
    t2.add_column('a', [2, 3, 4])
    t2.add_column('b', [1, 1, 1])
    t2.add_column('c', [5, 6, 7])

    t3 = inner_join(t1, t2, 'b')

    assert np.all(t3.a.values == np.array([1, 1, 1]))
    assert np.all(t3.b.values == np.array([1, 1, 1]))
    assert np.all(t3.c.values == np.array([5, 6, 7]))


def test_inner_join_3():
    t1 = Table()
    t1.add_column('a', [1, 2, 2, 3, 4, 5, 6])
    t1.add_column('b', [1, 2, 3, 4])

    t2 = Table()
    t2.add_column('a', [0, 2, 3, 3, 4])
    t2.add_column('b', [1, 1, 1, 1, 1])
    t2.add_column('c', [4, 5, 6, 7, 8])

    t3 = inner_join(t1, t2, 'a')

    assert np.all(t3.a.values == np.array([2, 2, 3, 3, 4]))
    assert np.all(t3.b.values == np.array([2, 3, 4]))
    assert np.all(t3.c.values == np.array([5, 6, 6, 7, 8]))
