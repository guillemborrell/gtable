from gtable import Table
import numpy as np


def test_records():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t1 = Table({'a': [1, 2, 3], 'd': np.array([4, 5, 6])})
    t.stack(t1)

    records = [r for r in t.records()]
    assert records == [
        {'a': 1, 'b': 4},
        {'a': 2, 'b': 5},
        {'a': 3, 'b': 6},
        {'a': 1, 'd': 4},
        {'a': 2, 'd': 5},
        {'a': 3, 'd': 6}]

    records = [r for r in t.records(fill=True)]
    assert records == [
        {'a': 1, 'b': 4, 'd': np.nan},
        {'a': 2, 'b': 5, 'd': np.nan},
        {'a': 3, 'b': 6, 'd': np.nan},
        {'a': 1, 'b': np.nan, 'd': 4},
        {'a': 2, 'b': np.nan, 'd': 5},
        {'a': 3, 'b': np.nan, 'd': 6}]


def test_first_record():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t1 = Table({'a': [1, 2, 3], 'd': np.array([4, 5, 6])})
    t.stack(t1)

    assert t.first_record() == {'a': 1, 'b': 4}
    assert t.first_record(fill=True) == {'a': 1, 'b': 4, 'd': np.nan}


def test_last_record():
    t = Table({'a': [1, 2, 3], 'b': np.array([4, 5, 6])})
    t1 = Table({'a': [1, 2, 3], 'd': np.array([4, 5, 6])})
    t.stack(t1)

    assert t.last_record() == {'a': 1, 'd': 4}
    assert t.last_record(fill=True) == {'a': 1, 'd': 4, 'b': np.nan}

