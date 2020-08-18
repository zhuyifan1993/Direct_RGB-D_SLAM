import numpy as np


def downscale(i, d, k):
    """

    :param i: rgb color image
    :param d: depth image
    :param k: camera calibration matrix
    :return: downscaled id, dd, kd
    """

    kd = np.array([[k[0, 0] / 2, 0, (k[0, 2] + 0.5) / 2 - 0.5],
                   [0, k[1, 1] / 2, (k[1, 2] + 0.5) / 2 - 0.5],
                   [0, 0, 1]])

    id = (i[0::2, 0::2] + i[0::2, 1::2] + i[1::2, 0::2] + i[1::2, 1::2]) * 0.25
    ddcountvalid = np.sign(d[0::2, 0::2]) + np.sign(d[0::2, 1::2]) + np.sign(d[1::2, 0::2]) + np.sign(d[1::2, 1::2])
    dd = (d[0::2, 0::2] + d[0::2, 1::2] + d[1::2, 0::2] + d[1::2, 1::2]) / ddcountvalid
    dd[np.isnan(dd)] = 0
    return id, dd, kd
