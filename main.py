import camera_tracking as ct
import pose_graph_optimization as pgo
import argparse

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(
        description='RGB-D SLAM project'
    )
    parser.add_argument('--tracking', type=int, default=0, help='run keyframe-based camera tracking algorithm')
    parser.add_argument('--keyframe_selection', type=int, default=0, help='choose keyframe selection strategy')
    parser.add_argument('--pgo', type=int, default=0,
                        help='run pose-graph optimization algorithm for the keyframe poses')
    parser.add_argument('--loop_closure', type=int, default=0, help='adding loop closure manually or automatically')
    args = parser.parse_args()

    pair_file_path = 'pairs_list.txt'  # RGB-D frame pairs
    absolute_pose_path = 'kf_pose_level4.txt'  # keyframe absolute pose
    keyframe_index_path = 'kf_index_level4.txt'  # keyframe index number
    loop_pair_file_path = 'loop_closure_pair.npy'  # loop closure pairs index

    # run keyframe-based camera tracking algorithm ######################
    if args.tracking:
        if args.keyframe_selection == 0:
            # key-frame selection strategy: thresholds on the rotational and translational distance
            ct.camera_tracking_with_rot_and_trans_thresh(pair_file_path=pair_file_path, level=4, rot_thresh=0.1,
                                                         trans_thresh=0.1)

        elif args.keyframe_selection == 1:
            # key-frame selection strategy: relative entropy measure
            ct.camera_tracking_with_entropy(pair_file_path=pair_file_path, level=4, entropy_ratio_threshold=0.95)

    # run pose-graph optimization algorithm for the keyframe poses #######
    if args.pgo:
        if args.loop_closure == 0:
            pgo.optimise_pose_graph(pair_path=pair_file_path, pose_path=absolute_pose_path,
                                    index_path=keyframe_index_path,
                                    loop_closure=True, level=4)

        elif args.loop_closure == 1:

            # run pose graph optimization with automatic loop closure
            pgo.optimise_pose_graph_with_auto_loop_closure(pair_path=pair_file_path, pose_path=absolute_pose_path,
                                                           index_path=keyframe_index_path,
                                                           loop_pairs_path=loop_pair_file_path,
                                                           level=4)
