import numpy as np
from utils import se3
from scipy.spatial.transform import Rotation as R


def trans_to_quater(trans):
    r = R.from_dcm(trans[0:3, 0:3])
    quater = r.as_quat()

    q_pose = np.round(np.array([trans[0, 3], trans[1, 3], trans[2, 3], quater[0], quater[1], quater[2], quater[3]]),
                      4)
    return q_pose


def quater_to_trans(q):
    tx, ty, tz = q[0], q[1], q[2]
    qx, qy, qz, qw = q[3], q[4], q[5], q[6]
    r = R.from_quat([qx, qy, qz, qw])
    rotation_m = r.as_matrix()
    trans_m = np.array([[rotation_m[0, 0], rotation_m[0, 1], rotation_m[0, 2], tx],
                        [rotation_m[1, 0], rotation_m[1, 1], rotation_m[1, 2], ty],
                        [rotation_m[2, 0], rotation_m[2, 1], rotation_m[2, 2], tz],
                        [0, 0, 0, 1]], dtype='float64')

    return trans_m
