from gtable import Table
import numpy as np


def test_sort_0():
    t1 = Table()
    t1.add_column('a', [1, 2, 3, 4, 5])
    t1.add_column('b', [1, 2, 3])

    t2 = t1.copy()
    t3 = t1.copy()

    t4 = Table.from_chunks([t1, t2, t3])

    assert np.all(t4.b.index == np.array([1, 1, 1, 0, 0,
                                          1, 1, 1, 0, 0,
                                          1, 1, 1, 0, 0]))
