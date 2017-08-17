from gtable import Table
import numpy as np


def test_delete_column():
    a = np.arange(10, dtype=np.double)
    table_a = Table({'a': a, 'b': a})

    table_a.del_column('b')
    print(table_a.keys)
    assert table_a.keys == ['a']
    assert table_a.index.shape == (1, 10)
    assert np.all(table_a.a.values == np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64))
