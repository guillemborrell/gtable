from gtable import Table, merge
import numpy as np


def test_merge_1():
    a = np.arange(10, dtype=np.double)
    b = np.arange(5, 15, dtype=np.double)
    table_a = Table({'a': a, 'b': a})
    table_b = Table({'b': b})
    
    table_a.merge(table_b, 'b')
    assert np.all(
        table_a.b.values == np.array(
            [0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11, 12, 13, 14]
            ))
    assert np.all(
        table_a.b.index == np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ))

    
def test_merge_2():
    a = [0, 1, 1, 5, 3, 2]
    b = [-4, -1, 0, 4, 10, 20]
    table_a = Table({'a': a, 'b': a})
    table_b = Table({'b': b, 'c': a})
    
    table_a.merge(table_b, 'b')

    assert np.all(
        table_a.b.values == np.array(
            [-4, -1, 0, 0, 1, 1, 3, 2, 4, 5, 10, 20]
            ))


def test_merge_3():
    table_a = Table()
    table_b = Table()

    table_a.keys = ['a', 'b']
    table_a.data = [
        np.array([1, 2, 3]),
        np.array([1, 2, 3])
    ]
    table_a.index = np.array(
        np.array([
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=np.uint8)
    )
    table_b.keys = ['a', 'c']
    table_b.data = [
        np.array([1, 2, 3]),
        np.array([1, 2, 3])
    ]
    table_b.index = np.array(
        np.array([
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=np.uint8)
    )
    table_a.merge(table_b, 'a')

    assert np.all(table_a.a.index == np.array([1, 1, 1, 1, 1, 1]))
    assert np.all(table_a.b.index == np.array([1, 0, 1, 0, 1, 0]))
    assert np.all(table_a.c.index == np.array([0, 1, 0, 1, 0, 1]))


def test_merge_4():
    table_a = Table()
    table_b = Table()

    table_a.add_column('a', [1])
    table_a.add_column('b', [1])

    table_b.add_column('a', [1, 1, 1])
    table_b.add_column('c', [1, 2, 3])

    c = merge(table_a, table_b, 'a')
    assert np.all(c.a.index == np.array([1, 1, 1, 1]))
    assert np.all(c.b.index == np.array([1, 0, 0, 0]))
    assert np.all(c.c.index == np.array([0, 1, 1, 1]))


def test_merge_empty_1():
    table_a = Table()
    table_b = Table()

    table_a.add_column('a', [])
    table_a.add_column('b', [])

    table_b.add_column('a', [1, 1, 1])
    table_b.add_column('c', [1, 2, 3])

    c = merge(table_a, table_b, 'a')
    assert np.all(c.a.index == np.array([1, 1, 1]))
    assert np.all(c.b.index == np.array([0, 0, 0]))
    assert np.all(c.c.index == np.array([1, 1, 1]))



