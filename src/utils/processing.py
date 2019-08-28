"""
Processing file.
"""
try:
    import sys
except ImportError as err:
    exit("{}: {}".format(__file__, err))


def make_trainable(m, val):
    """
    TODO

    :param m: Keras model
    :param val: value (True or False)
    :return:
    """
    if val not in [True, False]:
        raise ValueError("Bad value for `val`: {}".format(val))

    for l in m.layers:
        l.trainable = val
