from gtable.fast import intersection_sorted, union_sorted
import numpy as np


def test_intersection():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 4])

    assert np.all(np.intersect1d(a, b) == intersection_sorted(a, b))

def test_union():
    a = np.array([1, 2, 3, 6, 7])
    b = np.array([1, 4, 5, 6, 7])

    assert np.all(np.union1d(a, b) == union_sorted(a, b))