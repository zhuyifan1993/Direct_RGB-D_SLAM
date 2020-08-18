from utils import read_pair as rp
import numpy as np
from scipy.linalg import cho_solve, cholesky
from utils import downscale as ds
from utils import lukas_kanade as lk
from utils import se3
import warnings

warnings.filterwarnings('ignore')


def pgo(pair_path=None, absolute_poses=None, kf_index=None, loop_closure=False, loop_pairs=None, level=4,
        r_p_list=[], s_m_list=[], lambd=0.1):
    pose_num = len(absolute_poses)
    if loop_closure:
        loop_num = len(loop_pairs)
    else:
        loop_num = 0
    loop_ind = 0

    b_k = np.zeros(6 * pose_num)  # b_k vector
    h_k = np.zeros([6 * pose_num, 6 * pose_num])  # H_k matrix

    residual = np.zeros([pose_num + loop_num, 6])  # residual vector

    relative_pose_list = r_p_list
    sigma_matrix_list = s_m_list

    for ind in np.arange(pose_num - 1 + loop_num):

        if ind >= pose_num - 1:
            # add loop closure manually at the end of the trajectory by observing overlap
            loop_pair = loop_pairs[loop_ind]
            loop_ind += 1
            first_pose_ind = loop_pair[0]
            second_pose_ind = loop_pair[1]
            ref_frame = kf_index[first_pose_ind]
            cur_frame = kf_index[second_pose_ind]
            first_pose = absolute_poses[first_pose_ind]
            second_pose = absolute_poses[second_pose_ind]

        else:
            first_pose_ind = ind
            second_pose_ind = ind + 1
            ref_frame = kf_index[first_pose_ind]
            cur_frame = kf_index[second_pose_ind]
            first_pose = absolute_poses[first_pose_ind]
            second_pose = absolute_poses[second_pose_ind]

        # calculate the pose difference between keyframes pair
        pose_diff = np.linalg.inv(second_pose) @ first_pose

        # create relative pose constraint between keyframes pair through direct image alignment in the first pgo iter
        if len(relative_pose_list) < ind + 1:
            image1, depth1, _ = rp.read_pair(pair_path, ref_frame)
            image2, depth2, _ = rp.read_pair(pair_path, cur_frame)
            relative_pose, sigma_matrix = create_relative_pose_constraint(image1, depth1, image2, depth2, level,
                                                                          se3.se3Log(pose_diff))

            relative_pose_list.append(relative_pose)
            sigma_matrix_list.append(sigma_matrix)
        else:
            relative_pose = relative_pose_list[ind]
            sigma_matrix = sigma_matrix_list[ind]

        # convert twist coordinate to 4*4 transformation matrix
        relative_pose = se3.se3Exp(relative_pose)

        # calculate relative pose residual between this key-frame pair
        resid = se3.se3Log(np.linalg.inv(relative_pose) @ pose_diff)
        # print(resid)

        # stack residual vector
        residual[ind + 1] = resid

        # calculate jacobian matrix
        jacobian = calculatejacobian(pose_num, first_pose, second_pose, relative_pose, first_pose_ind, second_pose_ind,
                                     resid)

        # accumulate b_k and h_k for gauss newton step
        b_k += jacobian.T @ sigma_matrix @ resid
        h_k += jacobian.T @ sigma_matrix @ jacobian
        # print('keyframes pair:{}'.format(ind + 1))

    # add another term for first pose in b_k and h_k
    resid_0 = se3.se3Log(absolute_poses[0])
    residual[0] = resid_0
    jaco_0 = calculate_jaco_single_frame(pose_num, absolute_poses[0], resid_0)
    sigma_matrix = np.identity(6) * 1e6
    b_k += jaco_0.T @ sigma_matrix @ resid_0
    h_k += jaco_0.T @ sigma_matrix @ jaco_0

    residual = residual.reshape(-1)

    # update all key frame poses with Levenberg-Marquardt method
    # upd = - np.linalg.inv(h_k + lambd * np.diag(np.diag(h_k))) @ b_k

    # with gauss newton method
    # upd = - np.linalg.inv(h_k) @ b_k

    # use cholesky factorization to solve the linear system -h_k * upd = b_k
    c = cholesky(h_k)
    upd = - cho_solve((c, False), b_k)

    upd = upd.reshape(-1, 6)
    for jj in range(pose_num):
        absolute_poses[jj] = se3.se3Exp(upd[jj]) @ absolute_poses[jj]

    residual_after_update = calculate_residual(absolute_poses, relative_pose_list, loop_pairs, loop_num)

    return absolute_poses, residual, relative_pose_list, sigma_matrix_list, residual_after_update


def calculate_residual(absolute_poses, relative_poses, loop_pairs, loop_num):
    pose_num = len(absolute_poses)
    loop_ind = 0

    residual = np.zeros([pose_num + loop_num, 6])
    for ind in range(pose_num - 1 + loop_num):
        if ind >= pose_num - 1:
            # add loop closure manually at the end of the trajectory by observing overlap
            loop_pair = loop_pairs[loop_ind]
            loop_ind += 1
            first_pose_ind = loop_pair[0]
            second_pose_ind = loop_pair[1]
            first_pose = absolute_poses[first_pose_ind]
            second_pose = absolute_poses[second_pose_ind]

        else:
            first_pose_ind = ind
            second_pose_ind = ind + 1
            first_pose = absolute_poses[first_pose_ind]
            second_pose = absolute_poses[second_pose_ind]

        # calculate the pose difference between keyframes pair
        pose_diff = np.linalg.inv(second_pose) @ first_pose

        relative_pose = se3.se3Exp(relative_poses[ind])
        residual[ind + 1] = se3.se3Log(np.linalg.inv(relative_pose) @ pose_diff)

    residual[0] = se3.se3Log(absolute_poses[0])
    residual = residual.reshape(-1)
    return residual


def calculate_jaco_single_frame(pose_num, absolute_pose, resid):
    jacobian = np.zeros([6, 6 * pose_num])

    for j in range(6):
        epsVec = np.zeros(6)
        eps = 1e-6
        epsVec[j] = eps

        # multiply pose increment from left onto the absolute poses
        first_pose_eps = se3.se3Exp(epsVec) @ absolute_pose
        resid_eps = se3.se3Log(first_pose_eps)

        # calculate jacobian at first and second pose position
        jacobian[:, j] = (resid_eps - resid) / eps

    return jacobian


def pgo_with_auto_loop_closure(pair_path=None, absolute_poses=None, kf_index=None, loop_pairs=None, level=4,
                               r_p_list=[], s_m_list=[], lambd=0.1):
    pose_num = len(absolute_poses)
    pair_num = len(loop_pairs)

    b_k = np.zeros(6 * pose_num)  # b_k vector
    h_k = np.zeros([6 * pose_num, 6 * pose_num])  # H_k matrix

    residual = np.zeros([pair_num + 1, 6])  # residual vector

    relative_pose_list = r_p_list
    sigma_matrix_list = s_m_list

    for i in np.arange(pair_num):
        ref_frame = loop_pairs[i][0]
        cur_frame = loop_pairs[i][1]
        first_pose_ind = np.where(kf_index == ref_frame)[0][0]
        second_pose_ind = np.where(kf_index == cur_frame)[0][0]
        first_pose = absolute_poses[first_pose_ind]
        second_pose = absolute_poses[second_pose_ind]
        image1, depth1, _ = rp.read_pair(pair_path, ref_frame)
        image2, depth2, _ = rp.read_pair(pair_path, cur_frame)

        # calculate the pose difference between keyframes pair
        pose_diff = np.linalg.inv(second_pose) @ first_pose

        # create relative pose constraint between keyframes pair through direct image alignment in the first pgo iter
        if len(relative_pose_list) < i + 1:
            image1, depth1, _ = rp.read_pair(pair_path, ref_frame)
            image2, depth2, _ = rp.read_pair(pair_path, cur_frame)
            relative_pose, sigma_matrix = create_relative_pose_constraint(image1, depth1, image2, depth2, level,
                                                                          se3.se3Log(pose_diff))
            relative_pose_list.append(relative_pose)
            sigma_matrix_list.append(sigma_matrix)
        else:
            relative_pose = relative_pose_list[i]
            sigma_matrix = sigma_matrix_list[i]

        # convert twist coordinate to 4*4 transformation matrix
        relative_pose = se3.se3Exp(relative_pose)

        # calculate relative pose residual between this key-frame pair
        resid = se3.se3Log(np.linalg.inv(relative_pose) @ pose_diff)

        # stack residual vector
        residual[i + 1] = resid

        # calculate jacobian matrix
        jacobian = calculatejacobian(pose_num, first_pose, second_pose, relative_pose, first_pose_ind, second_pose_ind,
                                     resid)

        # accumulate b_k and h_k for gauss newton step
        b_k += jacobian.T @ sigma_matrix @ resid
        h_k += jacobian.T @ sigma_matrix @ jacobian
        # print('keyframes pair:{}'.format(i + 1))

    # add another term for first pose in b_k and h_k
    resid_0 = se3.se3Log(absolute_poses[0])
    residual[0] = resid_0
    jaco_0 = calculate_jaco_single_frame(pose_num, absolute_poses[0], resid_0)
    sigma_matrix = np.identity(6) * 1e6
    b_k += jaco_0.T @ sigma_matrix @ resid_0
    h_k += jaco_0.T @ sigma_matrix @ jaco_0

    # update all key frame poses
    # upd = - np.linalg.inv(h_k) @ b_k # use gauss-newton

    # use cholesky factorization to solve the linear system H x = b
    c = cholesky(h_k + lambd * np.diag(np.diag(h_k)))  # use levenberg marquardt
    upd = - cho_solve((c, False), b_k)

    upd = upd.reshape(-1, 6)
    for jj in range(pose_num):
        absolute_poses[jj] = se3.se3Exp(upd[jj]) @ absolute_poses[jj]

    residual = residual.reshape(-1)
    residual_after_update = calculate_residual_with_auto_lc(absolute_poses, relative_pose_list, loop_pairs, kf_index)

    return absolute_poses, residual, relative_pose_list, sigma_matrix_list, residual_after_update


def calculate_residual_with_auto_lc(absolute_poses, relative_poses, loop_pairs, kf_index):
    pair_num = len(loop_pairs)

    residual = np.zeros([pair_num + 1, 6])  # residual vector

    for i in np.arange(pair_num):
        ref_frame = loop_pairs[i][0]
        cur_frame = loop_pairs[i][1]
        first_pose_ind = np.where(kf_index == ref_frame)[0][0]
        second_pose_ind = np.where(kf_index == cur_frame)[0][0]
        first_pose = absolute_poses[first_pose_ind]
        second_pose = absolute_poses[second_pose_ind]
        # calculate the pose difference between keyframes pair
        pose_diff = np.linalg.inv(second_pose) @ first_pose

        relative_pose = se3.se3Exp(relative_poses[i])
        residual[i + 1] = se3.se3Log(np.linalg.inv(relative_pose) @ pose_diff)

    residual[0] = se3.se3Log(absolute_poses[0])
    residual = residual.reshape(-1)

    return residual


def create_relative_pose_constraint(image1, depth1, image2, depth2, level, initial_rela_pose):
    # camera calibration matrix
    fx = 520.9  # focal length x
    fy = 521.0  # focal length y
    cx = 325.1  # optical center x
    cy = 249.7  # optical center y
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    use_hubernorm = 1
    norm_param = 0.2

    # perform level times down_sampling and display the downscaled image
    Id = image1
    Dd = depth1
    Kd = K
    for i in range(level):
        Id, Dd, Kd = ds.downscale(Id, Dd, Kd)

    Iref = Id
    Dref = Dd

    Id = image2
    Dd = depth2
    Kd = K

    for i in range(level):
        Id, Dd, Kd = ds.downscale(Id, Dd, Kd)

    # do direct image alignment between this key-frame pair
    xi = initial_rela_pose
    relative_pose, sigma_matrix = lk.do_alignment(Iref, Dref, Kd, Id, Dd, xi, norm_param,
                                                  use_hubernorm)
    return relative_pose, sigma_matrix


def calculatejacobian(pose_num, first_pose, second_pose, relative_pose, first_pose_ind, second_pose_ind, resid):
    jacobian = np.zeros([6, 6 * pose_num])

    for j in range(6):
        epsVec = np.zeros(6)
        eps = 1e-6
        epsVec[j] = eps

        # multiply pose increment from left onto the absolute poses
        first_pose_eps = se3.se3Exp(epsVec) @ first_pose
        second_pose_eps = se3.se3Exp(epsVec) @ second_pose

        # calculate new pose difference after eps pose increment and new relative pose residual
        # first key frame, fix second_pose
        pose_diff_eps = np.linalg.inv(second_pose) @ first_pose_eps
        resid_first_eps = se3.se3Log(np.linalg.inv(relative_pose) @ pose_diff_eps)

        # second key frame, fix first pose
        pose_diff_eps = np.linalg.inv(second_pose_eps) @ first_pose
        resid_second_eps = se3.se3Log(np.linalg.inv(relative_pose) @ pose_diff_eps)

        # calculate jacobian at first and second pose position
        jacobian[:, 6 * first_pose_ind + j] = (resid_first_eps - resid) / eps
        jacobian[:, 6 * second_pose_ind + j] = (resid_second_eps - resid) / eps
    return jacobian


def optimise_pose_graph(pair_path=None, pose_path=None, index_path=None, loop_closure=False, level=4):
    x = rp.read_absolute_poses(pose_path)
    kf_index = rp.read_pose_index(index_path)
    # errLast = 1e10
    loop_pairs = [[172, 0]]  # manually chosen loop closure keyframe pair on level 4

    relativ_pose_list = []
    sigma_matrix_list = []
    lambd = 1e-4
    new_x = x.copy()

    for i in range(20):
        x, residual, relativ_pose_list, sigma_matrix_list, residual_after_update = pgo(pair_path=pair_path,
                                                                                       absolute_poses=new_x.copy(),
                                                                                       kf_index=kf_index,
                                                                                       loop_closure=loop_closure,
                                                                                       loop_pairs=loop_pairs,
                                                                                       level=level,
                                                                                       r_p_list=relativ_pose_list,
                                                                                       s_m_list=sigma_matrix_list,
                                                                                       lambd=lambd)

        errLast = np.mean(residual ** 2)
        err = np.mean(residual_after_update ** 2)
        if err < errLast:
            # accept the update and set lambda = lambda/2
            print('iter:', i, 'errLast', errLast, 'err:', err, "accepted", 'lambda', lambd)
            new_x = x.copy()
            lambd = lambd / 2

        elif err > errLast:
            # reject the update and set lambda = lambda*2
            print('iter:', i, 'errLast', errLast, 'err:', err, "rejected", 'lambda', lambd)
            lambd = lambd * 2

        else:
            print('error converges')
            break

        np.save('traj_{}.npy'.format(i), new_x)


def optimise_pose_graph_with_auto_loop_closure(pair_path=None, pose_path=None, index_path=None, loop_pairs_path=None,
                                               level=4):
    x = rp.read_absolute_poses(pose_path)
    kf_index = rp.read_pose_index(index_path)
    loop_pairs = np.load(loop_pairs_path)
    # errLast = 1e10
    relativ_pose_list = []
    sigma_matrix_list = []
    lambd = 1e-4
    new_x = x.copy()

    for i in range(20):
        x, residual, relativ_pose_list, sigma_matrix_list, residual_after_update = pgo_with_auto_loop_closure(
            pair_path=pair_path, absolute_poses=new_x.copy(), kf_index=kf_index, loop_pairs=loop_pairs, level=level,
            r_p_list=relativ_pose_list, s_m_list=sigma_matrix_list, lambd=lambd)

        errLast = np.mean(residual ** 2)
        err = np.mean(residual_after_update ** 2)
        if err < errLast:
            # accept the update and set lambda = lambda/2
            print('iter:', i, 'errLast', errLast, 'err:', err, "accepted", 'lambda', lambd)
            new_x = x.copy()
            lambd = lambd / 2

        elif err > errLast:
            # reject the update and set lambda = lambda*2
            print('iter:', i, 'errLast', errLast, 'err:', err, "rejected", 'lambda', lambd)
            lambd = lambd * 2

        else:
            print('error converges')
            break

        np.save('traj_{}.npy'.format(i), new_x)
