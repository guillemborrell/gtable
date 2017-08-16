from gtable import Table
import numpy as np


def test_filter_1():
    a = np.arange(10, dtype=np.double)
    table_a = Table({'a': a, 'b': a})
    assert np.all(
        table_a.filter(table_a.a > 5).a.values == np.array([6, 7, 8, 9]))


def test_filter_2():
    a = np.arange(10, dtype=np.double)
    table_a = Table({'a': a, 'b': a})
    assert np.all(
        table_a.filter(
            (table_a.a > 5) & (table_a.b <= 6)).a.values == np.array([6])
    )