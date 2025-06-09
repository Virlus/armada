from hardware.my_device.robot import FlexivRobot, FlexivGripper
from third_party.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer

from scipy.spatial.transform import Rotation as R
from PIL import Image
import argparse
import numpy as np
import time
import os

HUMAN = 0
ROBOT = 1
PRE_INTV = 2
INTV = 3

def main(args):
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    replay_buffer = ReplayBuffer.copy_from_path(args.demo_path, keys=['wrist_cam', 'side_cam', 'tcp_pose', \
                                                                      'joint_pos', 'action', 'action_mode'])
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
    
    print(f"Success rate in this round: {success_count / (replay_buffer.n_episodes - args.start_index) * 100:.2f}%")
    import pdb; pdb.set_trace()

    image_save_path = os.path.join(os.path.dirname(args.demo_path), 'images')
    os.makedirs(image_save_path, exist_ok=True)

    for i in range(args.start_index, replay_buffer.n_episodes):
        robot.send_tcp_pose(robot.init_pose)
        gripper.move(gripper.max_width)
        time.sleep(2)
        # last_p = robot.init_pose[:3]
        # last_r = R.from_quat(robot.init_pose[3:7], scalar_first=True)
        # for j in range(0 if i == 0 else replay_buffer.episode_ends[i-1], replay_buffer.episode_ends[i]):
        #     start_time = time.time()
        #     curr_p_action = replay_buffer['action'][j, :3]
        #     curr_r_action = R.from_quat(replay_buffer['action'][j, 3:7], scalar_first=True)
        #     curr_p = last_p + curr_p_action
        #     curr_r = last_r * curr_r_action
        #     last_p = curr_p
        #     last_r = curr_r
        #     robot.send_tcp_pose(np.concatenate((curr_p, curr_r.as_quat(scalar_first=True)), 0))
        #     gripper.move(replay_buffer['action'][j, 7])
        #     time.sleep(max(1 / 10 - (time.time() - start_time), 0))

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
                save_buffer.add_episode(replay_buffer.get_episode(i, copy=True), compressors='disk')
                break
            else:
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--demo_path', type=str, default='/mnt/workspace/DP/0518_pour_cluttered_50_bsf_failure_detection_round2_1_expert/replay_buffer.zarr')
    parser.add_argument('-s', '--start_index', type=int, default=0)
    parser.add_argument('-o', '--output', type=str, default='/mnt/workspace/DP/debug')
    args = parser.parse_args()
    main(args)