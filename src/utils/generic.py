try:
    import numpy as np
except ImportError as err:
    exit("{}: {}".format(__file__, err))


def randomize_arrays(arrs):
    """
    Randomize a list of arrays with the same random state.

    :param arrs: a list of arrays
    :return: a list of randomized arrays
    """
    # Get the numpy random state
    state = np.random.get_state()
    for i, arr in enumerate(arrs):
        if i > 0:
            np.random.set_state(state)
        # Randomize
        np.random.shuffle(arr)
        # Replace this array
        arrs[i] = arr
    return arrs
