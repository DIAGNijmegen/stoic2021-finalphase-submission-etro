
from algorithm.preprocessing.definitions import *


def normalize_intensity(img_hu):
    """
    Normalize the HU intensity.
    The output intensity values are in the range [-1, 1].
    :param img_hu: input numpy array image
    :return: normalized image
    """
    # CLip HU intensities
    img_hu[img_hu < MIN_HU] = MIN_HU
    img_hu[img_hu > MAX_HU] = MAX_HU
    # Rescale intensities to [0, 1]
    img_hu = (img_hu - MIN_HU) / (MAX_HU - MIN_HU)
    # rescale to [-1, 1]
    img_hu = 2 * (img_hu - 0.5)
    return img_hu
