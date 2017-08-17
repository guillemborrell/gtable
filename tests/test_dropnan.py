from gtable import Table
import numpy as np


def test_dropnan():
    a = np.arange(10, dtype=np.double)
    a[5] = np.nan
    table_a = Table({'a': a, 'b': a})
    table_a.dropnan()

    assert np.all(table_a.a.values == np.array(
        [0, 1, 2, 3, 4, 6, 7, 8, 9], dtype=np.float64))
    assert np.all(table_a.a.index == np.array(
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]))
