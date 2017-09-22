from gtable import Table
import numpy as np


def test_fillna():
    t = Table()
    t.keys = ['a', 'b']
    t.data = [
        np.arange(10),
        np.array([1, 2])]
    t.index = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)

    b = t.b
    b.fillna()
    
    assert np.all(b.values == np.array([1, 1, 1, 1, 1, 2, 2, 2]))
    assert np.all(b.index == np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))


def test_fillna_fillvalue():
    t = Table()
    t.keys = ['a', 'b']
    t.data = [
        np.arange(10),
        np.array([1, 2])]
    t.index = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)

    b = t.b
    b.fillna(fillvalue=-1)
    
    assert np.all(b.values == np.array([-1, -1, 1, 1, 1, 1, 1, 2, 2, 2]))
    assert np.all(b.index == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))


def test_fillna_reverse():
    t = Table()
    t.keys = ['a', 'b']
    t.data = [
        np.arange(10),
        np.array([1, 2])]
    t.index = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)

    b = t.b
    b.fillna(reverse=True)
    
    assert np.all(b.values == np.array([1, 1, 1, 2, 2, 2, 2, 2]))
    assert np.all(b.index == np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]))

    
def test_fillna_reverse_fillvalue():
    t = Table()
    t.keys = ['a', 'b']
    t.data = [
        np.arange(10),
        np.array([1, 2])]
    t.index = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)

    b = t.b
    b.fillna(reverse=True, fillvalue=-1)
    
    assert np.all(b.values == np.array([1, 1, 1, 2, 2, 2, 2, 2, -1, -1]))
    assert np.all(b.index == np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))


def test_fillna_table():
    t = Table()
    t.keys = ['a', 'b']
    t.data = [
        np.arange(10),
        np.array([1, 2])]
    t.index = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)

    t.fillna_column('b')

    assert np.all(t.b.values == np.array([1, 1, 1, 1, 1, 2, 2, 2]))
    assert np.all(t.b.index == np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))


def test_fill_column():
    t = Table()
    t.keys = ['a', 'b']
    t.data = [
        np.arange(10),
        np.array([1, 2])]
    t.index = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)

    b = t.b
    b.fill(fillvalue=3)

    assert np.all(b.values == np.array([3, 3, 1, 3, 3, 3, 3, 2, 3, 3]))


def test_fill_table():
    t = Table()
    t.keys = ['a', 'b']
    t.data = [
        np.arange(10),
        np.array([1, 2])]
    t.index = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0]], dtype=np.uint8)

    t.fill_column('b', fillvalue=3)

    assert np.all(t.b.values == np.array([3, 3, 1, 3, 3, 3, 3, 2, 3, 3]))
