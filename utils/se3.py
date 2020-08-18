import numpy as np
from scipy.linalg import expm, logm


def se3Exp(twist):
    m = np.array([[0, -twist[5], twist[4], twist[0]],
                  [twist[5], 0, -twist[3], twist[1]],
                  [-twist[4], twist[3], 0, twist[2]],
                  [0, 0, 0, 0]], dtype='float64')

    omega_hat = m[0:3, 0:3]
    omega_norm = np.linalg.norm(twist[3:6])
    v = twist[0:3]

    if omega_norm:
        exp_omega_hat = np.identity(3) + np.sin(omega_norm) / omega_norm * omega_hat + (
                1 - np.cos(omega_norm)) / omega_norm ** 2 * omega_hat @ omega_hat

        A = np.identity(3) + (1 - np.cos(omega_norm)) / omega_norm ** 2 * omega_hat + (
                omega_norm - np.sin(omega_norm)) / omega_norm ** 3 * omega_hat @ omega_hat
    else:
        exp_omega_hat = np.identity(3)
        A = np.identity(3)
    t_matrix = np.identity(4)
    t_matrix[0:3, 0:3] = exp_omega_hat
    t_matrix[0:3, 3] = (A @ v).ravel()

    # t_matrix_appro = expm(m)
    return t_matrix


def se3Log(t_matrix):
    # lg = logm(t_matrix)
    R = t_matrix[0:3, 0:3]
    t = t_matrix[0:3, 3]
    omega_norm = np.arccos((np.trace(R) - 1) / 2)
    if omega_norm:
        log_R = omega_norm / (2 * np.sin(omega_norm)) * (R - R.T)
        A_inv = np.identity(3) - log_R / 2 + 1 / omega_norm ** 2 * (
                1 - np.sin(omega_norm) * omega_norm / (2 * (1 - np.cos(omega_norm)))) * log_R @ log_R
    else:
        log_R = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]) * omega_norm
        A_inv = np.identity(3)

    lg = np.zeros([4, 4])
    lg[0:3, 0:3] = log_R
    lg[0:3, 3] = A_inv @ t

    twist = np.array([lg[0, 3], lg[1, 3], lg[2, 3], lg[2, 1], lg[0, 2], lg[1, 0]], dtype='float64')
    return twist
