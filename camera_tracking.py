import cv2
import numpy as np
import math
from utils import downscale as ds
from utils import read_pair as rp
from utils import lukas_kanade as lk
from utils import se3
from scipy.spatial.transform import Rotation as R
import warnings

warnings.filterwarnings('ignore')


def camera_tracking_with_rot_and_trans_thresh(pair_file_path=None, level=4, rot_thresh=0.1, trans_thresh=0.1):
    # read first frame as original frame
    image, depth, timestamp = rp.read_pair(pair_file_path, 0)

    # save initial pose as the first (key)frame pose into '.txt'
    pose = np.array([0, 0, 0, 0, 0, 0, 1])
    with open('kf_pose_level{}.txt'.format(level), 'a') as file_handle:
        file_handle.write(timestamp)
        for ii in range(len(pose)):
            file_handle.write(' ' + str(pose[ii]))
        file_handle.write('\n')
    with open('kf_index_level{}.txt'.format(level), 'a') as file_handle:
        file_handle.write(timestamp)
        file_handle.write(' ' + str(0))
        file_handle.write('\n')
    with open('poses_rot_trans_level{}.txt'.format(level), 'a') as file_handle:
        file_handle.write(timestamp)
        for ii in range(len(pose)):
            file_handle.write(' ' + str(pose[ii]))
        file_handle.write('\n')

    # camera calibration matrix
    fx = 520.9  # focal length x
    fy = 521.0  # focal length y
    cx = 325.1  # optical center x
    cy = 249.7  # optical center y
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    cv2.imshow('original_image', image)
    cv2.imshow('original_depth', depth)

    # perform level times down_sampling and display the downscaled image
    Id = image
    Dd = depth
    Kd = K
    for i in range(level):
        Id, Dd, Kd = ds.downscale(Id, Dd, Kd)
    cv2.imshow('Id', Id)
    cv2.imshow('Dd', Dd)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # set the first key frame
    Iref = Id
    Dref = Dd

    use_hubernorm = 1
    norm_param = 0.2
    frames_length = len(open(pair_file_path).readlines())

    # camera tracking with key-frame based method
    keyfr_to_wfr = np.identity(4)  # the transform from the latest key frame to the original(world) frame
    xi = np.zeros(6)  # the transform from the latest key frame to the current frame

    for i in np.arange(1, frames_length):

        Id, Dd, timestamp = rp.read_pair(pair_file_path, i)
        Kd = K
        # downscale the original image
        for j in range(level):
            Id, Dd, Kd = ds.downscale(Id, Dd, Kd)

        # do direct image alignment between the current frame and the latest key frame
        xi, _ = lk.do_alignment(Iref, Dref, Kd, Id, Dd, xi, norm_param, use_hubernorm)

        # choose new key frame wrt. rotational and translational threshold
        add_new_keyframe = False
        if np.linalg.norm(xi[:3]) > trans_thresh or np.linalg.norm(xi[3:]) > rot_thresh:
            # set the current frame as new key frame
            Iref = Id
            Dref = Dd

            # update the transform from the new key frame to the world frame
            keyfr_to_wfr = keyfr_to_wfr @ se3.se3Exp(-xi)

            # reset the transform from the latest key frame to the current frame to identity
            xi = np.zeros(6)
            print("add new key frame")
            add_new_keyframe = True

        trans = keyfr_to_wfr @ se3.se3Exp(-xi)

        # convert rotation matrix to unit quaternion [qx,qy,qz,qw]
        r = R.from_dcm(trans[0:3, 0:3])
        quater = r.as_quat()
        # create the output pose [tx,ty,tz,qx,qy,qz,qw]
        pose = np.round(np.array([trans[0, 3], trans[1, 3], trans[2, 3], quater[0], quater[1], quater[2], quater[3]]),
                        4)

        # save key frames trajectory for pose graph optimization
        if add_new_keyframe:
            with open('kf_pose_level{}.txt'.format(level), 'a') as file_handle:
                file_handle.write(timestamp)
                for ii in range(len(pose)):
                    file_handle.write(' ' + str(pose[ii]))
                file_handle.write('\n')
            with open('kf_index_level{}.txt'.format(level), 'a') as file_handle:
                file_handle.write(timestamp)
                file_handle.write(' ' + str(i))
                file_handle.write('\n')

        # write the current pose into '.txt' to get the trajectory
        with open('poses_rot_trans_level{}.txt'.format(level), 'a') as file_handle:
            file_handle.write(timestamp)
            for ii in range(len(pose)):
                file_handle.write(' ' + str(pose[ii]))
            file_handle.write('\n')

        np.set_printoptions(suppress=True)
        print("timestamp:", i, pose)


def camera_tracking_with_entropy(pair_file_path=None, level=4, entropy_ratio_threshold=0.95):
    # Implement the relative entropy measure from lecture 9 to decide when to create new keyframes.

    # read first frame as original frame
    image, depth, _ = rp.read_pair(pair_file_path, 0)

    # camera calibration matrix
    fx = 520.9  # focal length x
    fy = 521.0  # focal length y
    cx = 325.1  # optical center x
    cy = 249.7  # optical center y
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # perform level times down_sampling and display the downscaled image
    Id = image
    Dd = depth
    Kd = K
    for i in range(level):
        Id, Dd, Kd = ds.downscale(Id, Dd, Kd)

    # set the first key frame
    Iref = Id
    Dref = Dd
    Kref = Kd

    use_hubernorm = 1
    norm_param = 0.2
    frames_length = len(open(pair_file_path).readlines())

    def entropy(relative_pose_corvariance):
        entropy = 0.5 * math.log(np.linalg.det(2 * np.pi * np.e * relative_pose_corvariance))
        return entropy

    entropy_next_to_keyframe = np.identity(6)
    add_new_kf = True
    keyfr_to_wfr = np.identity(4)
    xi = np.zeros(6)

    for i in np.arange(1, frames_length):

        Id, Dd, timestamp = rp.read_pair(pair_file_path, i)
        Kd = K
        # downscale the original image
        for j in range(level):
            Id, Dd, Kd = ds.downscale(Id, Dd, Kd)

        # do direct image alignment between the current frame and the last key frame
        xi, hessian_m = lk.do_alignment(Iref, Dref, Kd, Id, Dd, xi, norm_param, use_hubernorm)
        relative_pose_corvariance_i = np.linalg.inv(hessian_m)

        if add_new_kf:
            entropy_next_to_keyframe = entropy(relative_pose_corvariance_i)
            add_new_kf = False
        else:
            entropy_cur_frame = entropy(relative_pose_corvariance_i)
            ratio = entropy_cur_frame / entropy_next_to_keyframe
            if ratio < entropy_ratio_threshold:
                Iref = Id
                Dref = Dd
                add_new_kf = True
                print("add new key frame")
                keyfr_to_wfr = keyfr_to_wfr @ se3.se3Exp(-xi)
                xi = np.zeros(6)

        trans = keyfr_to_wfr @ se3.se3Exp(-xi)

        # convert rotation matrix to unit quaternion [qx,qy,qz,qw]
        r = R.from_dcm(trans[0:3, 0:3])
        quater = r.as_quat()
        # create the output pose [tx,ty,tz,qx,qy,qz,qw]
        pose = np.round(np.array([trans[0, 3], trans[1, 3], trans[2, 3], quater[0], quater[1], quater[2], quater[3]]),
                        4)

        # write the current pose into 'poses.txt' to get the trajectory
        with open('poses_entropy_level{}.txt'.format(level), 'a') as file_handle:
            file_handle.write(timestamp)
            for ii in range(len(pose)):
                file_handle.write(' ' + str(pose[ii]))
            file_handle.write('\n')

        np.set_printoptions(suppress=True)
        print("timestamp:", i, pose)
