import cv2
import numpy as np
from utils import conversions as conv


def read_pair(pair_file_path, line_index=0):
    f = open(pair_file_path)
    line = f.readlines()[line_index].split()
    rgb_path = 'rgbd_dataset_freiburg2_desk/' + line[1]
    depth_path = 'rgbd_dataset_freiburg2_desk/' + line[3]
    timestamp = line[0]
    image = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE).astype('float64') / 255
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype('float64') / 5000
    return image, depth, timestamp


def read_absolute_poses(pose_path):
    f = open(pose_path)
    line = f.readlines()
    pose_num = len(line)
    absolute_pose = []
    for i in range(pose_num):
        pose = line[i].split()[1:]
        trans = conv.quater_to_trans(np.asarray(pose, dtype='double'))
        absolute_pose.append(trans)

    return absolute_pose


def read_pose_index(pose_ind_path):
    f = open(pose_ind_path)
    line = f.readlines()
    num = len(line)
    pose_index = np.zeros(num, dtype=int)
    for i in range(num):
        index = line[i].split()[1]
        pose_index[i] = index

    return pose_index
