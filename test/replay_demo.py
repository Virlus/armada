import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hardware.my_device.robot import FlexivRobot, FlexivGripper
from robot_env import RobotEnv
from hardware.my_device.macros import CAM_SERIAL
from diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer

from scipy.spatial.transform import Rotation as R
from PIL import Image
import argparse
import numpy as np
import time

HUMAN = 0
ROBOT = 1
PRE_INTV = 2
INTV = 3

def main(args):
    # Initialize robot environment
    img_shape = [3, 224, 224]
    fps = 10
    # robot_env = RobotEnv(camera_serial=CAM_SERIAL, img_shape=img_shape, fps=fps)
    replay_buffer = ReplayBuffer.copy_from_path(args.demo_path, keys=None)

    # If there exists visual reference for initial state, load it before rollout
    reference_path = args.reference
    if os.path.exists(reference_path):
        refer = True
    else:
        refer = False

    ## =================================================== Calculate human intervention ratio and success rate ======================================================= ###

    import pdb; pdb.set_trace()
    zarr_path = os.path.join(args.output, 'replay_buffer.zarr')
    save_buffer = ReplayBuffer.create_from_path(zarr_path, mode='a')

    curr_round_start = replay_buffer.episode_ends[args.start_index-1] if args.start_index > 0 else 0
    curr_round_end = replay_buffer.episode_ends[-1]

    print("Human intervention ratio: ", np.sum(replay_buffer['action_mode'][curr_round_start:curr_round_end] == INTV) / \
                                                     (curr_round_end - curr_round_start) * 100.0)
    
    import pdb; pdb.set_trace()

    success_count = 0

    for i in range(args.start_index, replay_buffer.n_episodes):
        curr_action_mode = replay_buffer['action_mode'][replay_buffer.episode_ends[i-1]:replay_buffer.episode_ends[i]]
        if np.sum(curr_action_mode == INTV) == 0:
            success_count += 1

    success_rate = success_count / (replay_buffer.n_episodes - args.start_index)
    
    print(f"Success rate in this round: {success_rate * 100:.2f}%")
    import pdb; pdb.set_trace()

    # ============================================================================================================================================================ ###

    ## ==================================================== Calculate TPR and TNR for failure-detection-embodied rollouts on episode level========================================= ###

    TPR_buffer = [] # True indicates failed trajectories
    TNR_buffer = []

    failure_detection_activated = False

    for i in range(args.start_index, replay_buffer.n_episodes):
        curr_action_mode = replay_buffer['action_mode'][replay_buffer.episode_ends[i-1]:replay_buffer.episode_ends[i]]
        curr_failure_indices = replay_buffer['failure_indices'][replay_buffer.episode_ends[i-1]:replay_buffer.episode_ends[i]]
        if np.sum(curr_action_mode == INTV) == 0:
            failure_detection_activated = True
            if np.sum(curr_failure_indices) == 0:
                TNR_buffer.append(1.0)
            else:
                TNR_buffer.append(0.0)
        else:
            if np.sum(curr_failure_indices) > 0:
                TPR_buffer.append(1.0)
            else:
                if failure_detection_activated:
                    TPR_buffer.append(0.0)

    print(f"Episode-level TPR: {np.mean(TPR_buffer) * 100:.2f}%, Episode-level TNR: {np.mean(TNR_buffer) * 100:.2f}%")
    print(f"Episode-level Accuracy: {np.mean(TPR_buffer) * 50 + np.mean(TNR_buffer) * 50:.2f}%")
    print(f"Episode-level Weighted Accuracy: {np.mean(TPR_buffer) * success_rate * 100 + np.mean(TNR_buffer) * (1 - success_rate) * 100:.2f}%")

    import pdb; pdb.set_trace()

    ### ============================================================================================================================================================ ###

# ============================================================================================================================================================ ###

    ## ==================================================== Calculate TPR and TNR for failure-detection-embodied rollouts on sample level========================================= ###

    TNR_buffer = []
    total_sample_count = 0

    for i in range(args.start_index, replay_buffer.n_episodes):
        curr_action_mode = replay_buffer['action_mode'][replay_buffer.episode_ends[i-1]:replay_buffer.episode_ends[i]]
        curr_failure_indices = replay_buffer['failure_indices'][replay_buffer.episode_ends[i-1]:replay_buffer.episode_ends[i]]
        if np.sum(curr_action_mode == INTV) == 0:
            if np.sum(curr_failure_indices) == 0:
                TNR_buffer.append(curr_action_mode.shape[0])
            else:
                TNR_buffer.append(curr_action_mode.shape[0] - np.sum(curr_failure_indices))
            total_sample_count += curr_action_mode.shape[0]
        else:
            first_failure_index = np.where(curr_action_mode == INTV)[0][0]
            total_sample_count += first_failure_index
            TNR_buffer.append(first_failure_index - np.sum(curr_failure_indices[:first_failure_index-8]))

    print(f"Sample-level TNR: {np.sum(TNR_buffer) / total_sample_count * 100:.2f}%")

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--demo_path', type=str, required=True)
    parser.add_argument('-s', '--start_index', type=int, required=True)
    parser.add_argument('-ref', '--reference', type=str, default='')
    parser.add_argument('-o', '--output', type=str, default='./debug')
    args = parser.parse_args()
    main(args)