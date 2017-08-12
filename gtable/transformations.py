import numpy as np


def fillna(values, index, reverse=False, fillvalue=None):
    """
    Fills the non available value sequentially with the previous
    available position.
    """
    # Create the new values array
    new_index = np.ones(index.shape, dtype=np.uint8)

    # Compute the indices where values has to be fetched
    if not reverse:
        indices = np.cumsum(index.astype(np.int8)) - 1
        if fillvalue is not None:
            new_values = np.empty(indices.shape, values.dtype)
            new_values[np.where(indices>=0)] = values[
                indices[np.where(indices>=0)]]
            new_values[np.where(indices<0)] = fillvalue

        else:
            new_values = np.empty((indices>=0).sum(), values.dtype)
            new_values = values[indices[np.where(indices>=0)]]

    else:
        indices = np.cumsum(index.astype(np.int8)[::-1]) - 1
        if fillvalue is not None:
            new_values = np.empty(indices.shape, values.dtype)
            new_values[np.where(indices>=0)] = values[::-1][
                indices[np.where(indices>=0)]]
            new_values[np.where(indices<0)] = fillvalue

        else:
            new_values = np.empty((indices>=0).sum(), values.dtype)
            new_values = values[::-1][indices[np.where(indices>=0)]]

        new_values = new_values[::-1]
        
    new_index[np.where(indices<0)] = 0
    
    return new_values, new_index
