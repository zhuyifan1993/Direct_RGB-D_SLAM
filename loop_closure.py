import math
import numpy as np
from utils import read_pair as rp
from pose_graph_optimization import create_relative_pose_constraint
from utils import se3


def entropy(relative_pose_corvariance):
    entropy = 0.5 * math.log(np.linalg.det(2 * np.pi * np.e * relative_pose_corvariance))
    return entropy


def automatic_loop_closure_detection(pair_path, index_path, spatial_range, relative_entropy_threshold, level,
                                     pose_path):
    kf_index = rp.read_pose_index(index_path)
    key_frame_num = len(kf_index)
    loop_closure = []
    absolute_pose = rp.read_absolute_poses(pose_path)

    # for every keyframe_ki candidate search the potential matches to old keyframes k
    for i in np.arange(spatial_range, key_frame_num):
        for j in np.arange(i - 1, i - 1 - spatial_range, -1):
            keyframe_candidate = kf_index[i]
            keyframe_old = kf_index[j]
            potential_match, relative_entropy = validation_loop_closure(pair_path, keyframe_old, keyframe_candidate,
                                                                        relative_entropy_threshold, level,
                                                                        absolute_pose)
            if potential_match:
                loop_closure.append([keyframe_old, keyframe_candidate])
                print('loop closure pair:', keyframe_old, keyframe_candidate)
            else:
                print(relative_entropy)

    np.save('loop_closure_pair.npy', loop_closure)


def automatic_loop_closure_detection_with_trans_dis(index_path, pose_path, pair_path):
    kf_pose = rp.read_absolute_poses(pose_path)
    kf_index = rp.read_pose_index(index_path)
    key_frame_num = len(kf_index)
    loop_closure = []
    # generate relative constraints between close-by keyframes
    for i in np.arange(1, key_frame_num):
        for j in range(i):
            first_pose = kf_pose[j]
            second_pose = kf_pose[i]
            ref_frame = kf_index[j]
            cur_frame = kf_index[i]
            trans_dis = np.linalg.norm(first_pose[0:3, 3] - second_pose[0:3, 3])
            if trans_dis < 0.4:
                # print('keyframe', ref_frame, cur_frame, trans_dis)
                loop_closure.append([ref_frame, cur_frame])

    # generate loop closure at the end of trajectory to early visited keyframes
    for i in np.arange(160, key_frame_num):
        for j in range(30):
            first_pose = kf_pose[j]
            second_pose = kf_pose[i]
            ref_frame = kf_index[j]
            cur_frame = kf_index[i]
            trans_dis = np.linalg.norm(first_pose[0:3, 3] - second_pose[0:3, 3])
            if trans_dis < 3:
                # print('keyframe', ref_frame, cur_frame, trans_dis)
                loop_closure.append([ref_frame, cur_frame])

    return loop_closure


def validation_loop_closure(pair_path, keyframe_old, keyframe_candidate, relative_entropy_threshold, level,
                            absolute_pose):
    mean_entropy = []
    potential_match = False
    for i in np.arange(keyframe_candidate, keyframe_old, -1):
        ref_frame = keyframe_old
        cur_frame = i
        image1, depth1, _ = rp.read_pair(pair_path, ref_frame)
        image2, depth2, _ = rp.read_pair(pair_path, cur_frame)
        first_pose = absolute_pose[ref_frame]
        second_pose = absolute_pose[cur_frame]
        pose_diff = np.linalg.inv(second_pose) @ first_pose

        _, hessian_m = create_relative_pose_constraint(image1, depth1, image2, depth2, level, se3.se3Log(pose_diff))
        relative_pose_corvariance = np.linalg.inv(hessian_m)
        mean_entropy.append(entropy(relative_pose_corvariance))

    mean_entropy = sum(mean_entropy) / len(mean_entropy)

    # calculate pose entropy between keyframe_k and keyframe_ki
    ref_frame = keyframe_old
    cur_frame = keyframe_candidate
    image1, depth1, _ = rp.read_pair(pair_path, ref_frame)
    image2, depth2, _ = rp.read_pair(pair_path, cur_frame)
    first_pose = absolute_pose[ref_frame]
    second_pose = absolute_pose[cur_frame]
    pose_diff = np.linalg.inv(second_pose) @ first_pose
    _, hessian_m = create_relative_pose_constraint(image1, depth1, image2, depth2, level, se3.se3Log(pose_diff))
    relative_pose_corvariance = np.linalg.inv(hessian_m)
    entropy_k_ki = entropy(relative_pose_corvariance)

    relative_entropy = entropy_k_ki / mean_entropy
    if relative_entropy > relative_entropy_threshold:
        potential_match = True

    return potential_match, relative_entropy


if __name__ == '__main__':
    pair_file_path = 'pairs_list.txt'  # RGB-D frame pairs
    absolute_pose_path = 'kf_pose_level4.txt'  # keyframe absolute pose
    keyframe_index_path = 'kf_index_level4.txt'  # keyframe index number
    es_pose_path = 'poses_rot_trans_level4.txt'  # all frame absolute pose

    # use relative entropy measurement as criteria
    # automatic_loop_closure_detection(pair_file_path, keyframe_index_path, spatial_range=3,
    #                                  relative_entropy_threshold=0.95, level=4, pose_path=es_pose_path)

    # use translational distance as criteria, (without DIA)much faster than using entropy criteria
    lc = automatic_loop_closure_detection_with_trans_dis(keyframe_index_path, absolute_pose_path, pair_file_path)
    print(len(lc))
    np.save('loop_closure_pair.npy', lc)
