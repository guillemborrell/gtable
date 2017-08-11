from gtable.methods import sort
from gtable import Table
import numpy as np

def test_sort():
    a = [0,1,1,5,3,2]
    b = [0,1,2,3,4,5]
    table = Table({'a': a, 'b': b})

    # Concatenate another table to have one longer column
    table.vcat(Table({'b': [1,2,3,4]}))

    sort(table, 'a')
    print(table.a.values)
    print(table.b.values)

    assert np.all(table.a.values == np.array([0,1,1,2,3,5]))
    assert np.all(table.b.values == np.array(
        [0,1,2,5,4,3,1,2,3,4]
    ))

