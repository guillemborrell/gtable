from gtable.transformations import merge
from gtable import Table
import numpy as np


def test_merge_1():
    a = np.arange(10, dtype=np.double)
    b = np.arange(5, 15, dtype=np.double)
    table_a = Table({'a': a, 'b': a})
    table_b = Table({'b': b})
    
    table_c = merge(table_a, table_b, 'b')
    assert np.all(
        table_c.b.values == np.array(
            [0, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 11, 12, 13, 14]
            ))
    assert np.all(
        table_c.b.index == np.array(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ))

    
def test_merge_2():
    a = [0, 1, 1, 5, 3, 2]
    b = [-4, -1, 0, 4, 10, 20]
    table_a = Table({'a': a, 'b': a})
    table_b = Table({'b': b, 'c': a})
    
    table_c = merge(table_a, table_b, 'b')

    print(table_c.b.values)
    assert np.all(
        table_c.b.values == np.array(
            [-4, -1,  0,  0,  1,  1,  5,  3,  4,  2, 10, 20]
            ))
