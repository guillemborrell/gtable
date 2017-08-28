from gtable.lib import sort_table
from gtable import Table
import numpy as np


def test_sort():
    a = [0, 1, 1, 5, 3, 2]
    b = [0, 1, 2, 3, 4, 5]
    table = Table({'a': a, 'b': b})

    # Concatenate another table to have one longer column
    table.stack(Table({'b': [1, 2, 3, 4]}))

    sort_table(table, 'a')

    assert np.all(table.a.values == np.array([0, 1, 1, 2, 3, 5]))
    assert np.all(table.b.values == np.array(
        [0, 1, 2, 5, 4, 3, 1,  2, 3, 4]
    ))

