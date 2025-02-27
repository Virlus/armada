from hardware.my_device.robot import FlexivRobot, FlexivGripper
from third_party.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer

from scipy.spatial.transform import Rotation as R
from PIL import Image
import argparse
import numpy as np
import time
import os

def main(args):
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)
    replay_buffer = ReplayBuffer.copy_from_path(args.demo_path, keys=['wrist_cam', 'side_cam', 'joint_pos', 'action'])
    image_save_path = os.path.join(os.path.dirname(args.demo_path), 'images')
    os.makedirs(image_save_path, exist_ok=True)

    for i in range(replay_buffer.n_episodes):
        robot.send_tcp_pose(robot.init_pose)
        gripper.move(gripper.max_width)
        time.sleep(2)
        for j in range(0 if i == 0 else replay_buffer.episode_ends[i-1], replay_buffer.episode_ends[i]):
            start_time = time.time()
            curr_p = replay_buffer['action'][j, :3]
            curr_r = R.from_quat(replay_buffer['action'][j, [4,5,6,3]])
            robot.send_tcp_pose(np.concatenate((curr_p, curr_r.as_quat()[[3,0,1,2]]), 0))
            if replay_buffer['action'][j, 7] == 0:
                gripper.move(gripper.max_width)
            else:
                gripper.move(0)
            time.sleep(max(1 / 10 - (time.time() - start_time), 0))

        for j in range(0 if i == 0 else replay_buffer.episode_ends[i-1], replay_buffer.episode_ends[i]):
            episode_path = os.path.join(image_save_path, f'episode_{i}')
            os.makedirs(episode_path, exist_ok=True)
            side_img = replay_buffer['side_cam'][j]
            Image.fromarray(side_img).save(os.path.join(episode_path, f'side_img_{j}.png'))

        while True:
            key = input('Press "c" to continue, or "q" to quit: ')
            if key == 'q':
                exit(0)
            elif key == 'c':
                break
            else:
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--demo_path', type=str, default='/mnt/workspace/DP/0225_abs_PnP_deprecated/replay_buffer.zarr')
    args = parser.parse_args()
    main(args)