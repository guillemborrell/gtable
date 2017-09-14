import gtable as gt
import gtable.reductions
import numpy as np


def test_reduce_sum_1():
    t = gt.Table()
    t.add_column('a', [1, 2, 2, 3, 4, 5])
    t.add_column('b', [1, 2, 3])
    t.add_column('c', [3, 4, 5], align='bottom')

    t1 = gtable.reduce_by_key(t, 'a', 'sum')
    assert np.all(t1.b.values == np.array([1, 5]))
    assert np.all(t1.b.index == np.array([1, 1, 0, 0, 0]))
    assert np.all(t1.c.values == np.array([3, 4, 5]))
    assert np.all(t1.c.index == np.array([0, 0, 1, 1, 1]))


def test_reduce_sum_2():
    t = gt.Table()
    t.add_column('a', ['a', 'b', 'b', 'c', 'c', 'd'])
    t.add_column('b', [1, 2, 3])
    t.add_column('c', [3, 4, 5], align='bottom')

    t1 = gtable.reduce_by_key(t, 'a', 'sum')

    assert np.all(t1.a.values == np.array(list('abcd')))
    assert np.all(t1.a.index == np.array([1, 1, 1, 1]))
    assert np.all(t1.b.values == np.array([1, 5]))
    assert np.all(t1.b.index == np.array([1, 1, 0, 0]))
    assert np.all(t1.c.values == np.array([7, 5]))
    assert np.all(t1.c.index == np.array([0, 0, 1, 1]))


def test_reduce_prod_1():
    t = gt.Table()
    t.add_column('a', ['a', 'b', 'b', 'c', 'c', 'd'])
    t.add_column('b', [1, 2, 3])
    t.add_column('c', [3, 4, 5], align='bottom')

    t1 = gtable.reduce_by_key(t, 'a', 'prod')

    assert np.all(t1.a.values == np.array(list('abcd')))
    assert np.all(t1.a.index == np.array([1, 1, 1, 1]))
    assert np.all(t1.b.values == np.array([1, 6]))
    assert np.all(t1.b.index == np.array([1, 1, 0, 0]))
    assert np.all(t1.c.values == np.array([12, 5]))
    assert np.all(t1.c.index == np.array([0, 0, 1, 1]))


def test_reduce_method():
    t = gt.Table()
    t.add_column('a', ['a', 'b', 'b', 'c', 'c', 'd'])
    t.add_column('b', [1, 2, 3])
    t.add_column('c', [3, 4, 5], align='bottom')

    t1 = t.reduce_by_key('a').sum()

    assert np.all(t1.a.values == np.array(list('abcd')))
    assert np.all(t1.a.index == np.array([1, 1, 1, 1]))
    assert np.all(t1.b.values == np.array([1, 5]))
    assert np.all(t1.b.index == np.array([1, 1, 0, 0]))
    assert np.all(t1.c.values == np.array([7, 5]))
    assert np.all(t1.c.index == np.array([0, 0, 1, 1]))


def test_reduce_mean():
    t = gt.Table()
    t.add_column('a', ['a', 'b', 'b', 'c', 'c', 'd'])
    t.add_column('b', [1, 2, 3])
    t.add_column('c', [3, 4, 5], align='bottom')

    t1 = t.reduce_by_key('a').mean()

    assert np.all(t1.a.values == np.array(list('abcd')))
    assert np.all(t1.a.index == np.array([1, 1, 1, 1]))
    assert np.all(t1.b.values == np.array([1, 2.5]))
    assert np.all(t1.b.index == np.array([1, 1, 0, 0]))
    assert np.all(t1.c.values == np.array([3.5, 5]))
    assert np.all(t1.c.index == np.array([0, 0, 1, 1]))


def test_reduce_std():
    t = gt.Table()
    t.add_column('a', ['a', 'b', 'b', 'c', 'c', 'd'])
    t.add_column('b', [1, 2, 3])
    t.add_column('c', [3, 4, 5], align='bottom')

    t1 = t.reduce_by_key('a').std()

    assert np.all(t1.a.values == np.array(list('abcd')))
    assert np.all(t1.a.index == np.array([1, 1, 1, 1]))
    assert np.all(t1.b.values == np.array([0, 0.5]))
    assert np.all(t1.b.index == np.array([1, 1, 0, 0]))
    assert np.all(t1.c.values == np.array([0.5, 0]))
    assert np.all(t1.c.index == np.array([0, 0, 1, 1]))
