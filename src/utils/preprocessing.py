try:
    import numpy as np
except ImportError as err:
    exit(err)


def discriminator_shape(n, d_out_shape):
    """
    TODO
    :param n:
    :param d_out_shape:
    :return:
    """
    if len(d_out_shape) == 1:  # image gan
        return n, d_out_shape[0]
    elif len(d_out_shape) == 3:  # pixel, patch gan
        return n, d_out_shape[0], d_out_shape[1], d_out_shape[2]
    return None


def input2discriminator(real_img_patches, real_seg_patches, fake_seg_patches, d_out_shape):
    """
    TODO
    :param real_img_patches:
    :param real_seg_patches:
    :param fake_seg_patches:
    :param d_out_shape:
    :return:
    """
    # First of all, get the numpy random state
    rnd_state = np.random.get_state()
    real = np.concatenate((real_img_patches, real_seg_patches), axis=-1)
    fake = np.concatenate((real_img_patches, fake_seg_patches), axis=-1)

    d_x_batch = np.concatenate((real, fake), axis=0)
    # Shuffle this array
    np.random.shuffle(d_x_batch)

    # Reset the random state to the previous one
    np.random.set_state(rnd_state)

    # real: 1, fake: 0
    d_y_batch = np.ones(discriminator_shape(d_x_batch.shape[0], d_out_shape))
    d_y_batch[real.shape[0]:, ...] = 0
    # Shuffle this array
    np.random.shuffle(d_y_batch)

    return d_x_batch, d_y_batch


def input2gan(real_img_patches, real_seg_patches, d_out_shape):
    """
    TODO
    :param real_img_patches:
    :param real_seg_patches:
    :param d_out_shape:
    :return:
    """
    g_x_batch = [real_img_patches, real_seg_patches]
    # Set 1 to all labels (real: 1, fake: 0)
    g_y_batch = np.ones(discriminator_shape(real_seg_patches.shape[0], d_out_shape))
    return g_x_batch, g_y_batch
