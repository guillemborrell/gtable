import gtable as gt
import numpy as np


def test_gh6():
    ta = gt.Table({'A': [1, 2, 3, 1, 2, 3, 6], 'B': [0.2, 0.4, 0.4548, 0.34465, 0.4, 0.315456, 0.5454545]})
    tb = gt.Table({'A': [1, 2, 5, 7, 8, 3], 'C': [564, 4545, 8989, 45, 23, 78]})
    merged1 = gt.merge(ta, tb, 'A')
    print('fffff')
    tc = gt.Table({'A': [1, 7], 'D': [np.datetime64('2017-02-15'), np.datetime64('2017-03-28')]})
    merged2 = gt.merge(merged1, tc, 'A')
    assert merged2.index.sum(axis=1).tolist() == [len(l) for l in merged2.data]