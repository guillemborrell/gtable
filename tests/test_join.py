from gtable import Table
from gtable.joins import inner_join, full_outer_join
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


def test_outer_join_1():
    t1 = Table()
    t1.add_column('a', [1, 2, 3, 4, 5, 6])
    t1.add_column('b', [1, 2, 3])

    t2 = Table()
    t2.add_column('a', [2, 3, 4])
    t2.add_column('b', [1, 1, 1])
    t2.add_column('c', [5, 6, 7])

    t3 = full_outer_join(t1, t2, 'a')

    assert np.all(t3.a.values == np.array([1, 2, 3, 4, 5, 6]))
    assert np.all(t3.b.values == np.array([1, 2, 3, 1]))
    assert np.all(t3.c.values == np.array([5, 6, 7]))


def test_outer_join_2():
    t1 = Table({'A': ['A0', 'A1', 'A2', 'A3'],
                'B': ['B0', 'B1', 'B2', 'B3'],
                'C': ['C0', 'C1', 'C2', 'C3'],
                'D': ['D0', 'D1', 'D2', 'D3'],
                'idx': [1, 2, 3, 4]})

    t2 = Table({'A': ['A4', 'A5', 'A6', 'A7'],
                'B': ['B4', 'B5', 'B6', 'B7'],
                'C': ['C4', 'C5', 'C6', 'C7'],
                'D': ['D4', 'D5', 'D6', 'D7'],
                'idx': [5, 6, 7, 8]})

    t3 = full_outer_join(t1, t2, 'idx')

    assert np.all(t3.A.values == np.array(['A0', 'A1', 'A2', 'A3',
                                           'A4', 'A5', 'A6', 'A7']))


def test_outer_join_3():
    t1 = Table({'A': np.random.rand(100),
                'B': np.random.rand(100),
                'C': np.random.rand(100),
                'D': np.random.rand(100),
                'E': np.random.rand(100),
                'F': np.random.rand(100),
                'G': np.random.rand(100),
                'idx': np.arange(100)})

    t2 = Table({'A': np.random.rand(100),
                'G': np.random.rand(100),
                'idx': np.arange(100, 200)})

    print(t1)
    print(t2)

    t3 = full_outer_join(t1, t2, 'idx', check_sorted=False)

    assert np.all(t3.G.values == np.hstack([t1.G.values, t2.G.values]))

