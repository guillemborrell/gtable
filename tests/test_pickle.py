import pickle
import numpy as np
from gtable import Table


def test_pickle():
    """This test refers to PR #4 which adds pickle support."""

    original_table = Table({'a': [1, 2, 3], 'b': [4, 5, 6]})
    serialized_table = pickle.dumps(original_table)
    deserialized_table = pickle.loads(serialized_table)

    assert original_table.keys == deserialized_table.keys
    assert np.array_equal(original_table.data, deserialized_table.data)
    assert np.array_equal(original_table.index, deserialized_table.index)
