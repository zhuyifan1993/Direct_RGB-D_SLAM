import numpy as np
import utils.conversions as conv

traj = np.load('traj_0.npy')
num = len(traj)

absolute_pose_path = 'kf_index_level4.txt'
f = open(absolute_pose_path)
line = f.readlines()

for i in range(num):
    trans = traj[i]
    quater = conv.trans_to_quater(trans)
    timestamp = line[i].split()[0]
    with open('pose_op_level4_with_auto_lc.txt', 'a') as file_handle:
        file_handle.write(timestamp)
        for ii in range(len(quater)):
            file_handle.write(' ' + str(quater[ii]))
        file_handle.write('\n')
