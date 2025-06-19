import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hardware.my_device.robot import FlexivRobot, FlexivGripper
from robot_env import RobotEnv
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
    camera_serial = ["135122075425", "135122070361"]
    img_shape = [3, 224, 224]
    fps = 10
    robot_env = RobotEnv(camera_serial, img_shape, fps)
    replay_buffer = ReplayBuffer.copy_from_path(args.demo_path, keys=None)

    # If there exists visual reference for initial state, load it before rollout
    reference_path = args.reference
    if os.path.exists(reference_path):
        refer = True
    else:
        refer = False

    # import pdb; pdb.set_trace()
    # # zarr_path = os.path.join(args.output, 'replay_buffer.zarr')
    # # save_buffer = ReplayBuffer.create_from_path(zarr_path, mode='a')

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

    TPR_buffer = [] # True indicates failed trajectories
    TNR_buffer = []

    for i in range(args.start_index, replay_buffer.n_episodes):
        curr_action_mode = replay_buffer['action_mode'][replay_buffer.episode_ends[i-1]:replay_buffer.episode_ends[i]]
        curr_failure_indices = replay_buffer['failure_indices'][replay_buffer.episode_ends[i-1]:replay_buffer.episode_ends[i]]
        if np.sum(curr_action_mode == INTV) == 0:
            if np.sum(curr_failure_indices) == 0:
                TNR_buffer.append(1.0)
            else:
                TNR_buffer.append(0.0)
        else:
            if np.sum(curr_failure_indices) > 0:
                TPR_buffer.append(1.0)
            else:
                TPR_buffer.append(0.0)

    print(f"TPR: {np.mean(TPR_buffer) * 100:.2f}%, TNR: {np.mean(TNR_buffer) * 100:.2f}%")
    print(f"Accuracy: {np.mean(TPR_buffer) * 50 + np.mean(TNR_buffer) * 50:.2f}%")
    print(f"Weighted Accuracy: {np.mean(TPR_buffer) * success_rate * 100 + np.mean(TNR_buffer) * (1 - success_rate) * 100:.2f}%")

    import pdb; pdb.set_trace()

    image_save_path = os.path.join(os.path.dirname(args.demo_path), 'images')
    os.makedirs(image_save_path, exist_ok=True)

    for i in range(args.start_index, replay_buffer.n_episodes):
        curr_init_pose = replay_buffer['tcp_pose'][replay_buffer.episode_ends[i-1]] if i > 0 else replay_buffer['tcp_pose'][0]
        robot_env.deploy_action(curr_init_pose[:7], robot_env.gripper.max_width)
        time.sleep(2)
        
        if refer:
            robot_env.align_scene(reference_path, i - args.start_index)

        last_p = curr_init_pose[:3]
        last_r = R.from_quat(curr_init_pose[3:7], scalar_first=True)
        for j in range(0 if i == 0 else replay_buffer.episode_ends[i-1], replay_buffer.episode_ends[i]):
            start_time = time.time()
            curr_p_action = replay_buffer['action'][j, :3]
            curr_r_action = R.from_quat(replay_buffer['action'][j, 3:7], scalar_first=True)
            curr_p = last_p + curr_p_action
            curr_r = last_r * curr_r_action
            last_p = curr_p
            last_r = curr_r
            robot_env.deploy_action(np.concatenate((curr_p, curr_r.as_quat(scalar_first=True)), 0), replay_buffer['action'][j, 7])
            time.sleep(max(1 / 10 - (time.time() - start_time), 0))

        for j in range(0 if i == 0 else replay_buffer.episode_ends[i-1], replay_buffer.episode_ends[i]):
            episode_path = os.path.join(image_save_path, f'episode_{i}')
            os.makedirs(episode_path, exist_ok=True)
            side_img = replay_buffer['side_cam'][j]
            wrist_img = replay_buffer['wrist_cam'][j]
            Image.fromarray(side_img).save(os.path.join(episode_path, f'side_img_{j}.png'))
            Image.fromarray(wrist_img).save(os.path.join(episode_path, f'wrist_img_{j}.png'))

        while True:
            key = input('Press "s" to save to filtered data, "c" to continue, or "q" to quit: ')
            if key == 'q':
                exit(0)
            elif key == 'c':
                break
            elif key == 's':
                # save_buffer.add_episode(replay_buffer.get_episode(i, copy=True), compressors='disk')
                break
            else:
                continue

        robot_env.reset_robot()
        robot_env.keyboard.ctn = False
        robot_env.keyboard.start = False
        robot_env.keyboard.quit = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--demo_path', type=str, required=True)
    parser.add_argument('-s', '--start_index', type=int, required=True)
    parser.add_argument('-ref', '--reference', type=str, default='')
    parser.add_argument('-o', '--output', type=str, default='/mnt/workspace/DP/debug_0619')
    args = parser.parse_args()
    main(args)