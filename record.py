import time
import os
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from hardware.robot_env import RobotEnv
from hardware.my_device.macros import CAM_SERIAL
from armada.diffusion_policy.diffusion_policy.common.replay_buffer import ReplayBuffer

def record(replay_buffer:ReplayBuffer, robot_env:RobotEnv):
    tcp_pose = []
    joint_pos = []
    action = []
    wrist_cam = []
    side_cam = []

    robot_env.keyboard.start = False
    robot_env.keyboard.discard = False
    robot_env.keyboard.finish = False
    cnt = 0

    robot_env.reset_robot()
    last_p = robot_env.robot.init_pose[:3]
    last_r = R.from_quat(robot_env.robot.init_pose[3:7], scalar_first=True)

    seed = int(time.time())
    np.random.seed(seed)

    while not robot_env.keyboard.quit and not robot_env.keyboard.discard and not robot_env.keyboard.finish:
        transition_data, last_p, last_r = robot_env.human_teleop_step(last_p, last_r)
        if not robot_env.keyboard.start or transition_data is None:
            continue

        # Initialize at the beginning of the episode
        if cnt == 0:
            random_init_pose = robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
            robot_env.reset_robot(random_init=True, random_init_pose=random_init_pose)
            last_p = random_init_pose[:3]
            last_r = R.from_quat(random_init_pose[3:7], scalar_first=True)
            cnt += 1
            print("Episode start!")
            continue
        
        wrist_cam.append(transition_data['demo_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        side_cam.append(transition_data['demo_side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        tcp_pose.append(transition_data['tcp_pose'])
        joint_pos.append(transition_data['joint_pos'])
        action.append(transition_data['action'])

    if not robot_env.keyboard.start or robot_env.keyboard.quit or robot_env.keyboard.discard:
        print('WARNING: discard the demo!')
        robot_env.gripper.move(robot_env.gripper.max_width)
        time.sleep(0.5)
        return
    
    episode = dict()
    episode['wrist_cam'] = np.stack(wrist_cam, axis=0)
    episode['side_cam'] = np.stack(side_cam, axis=0)
    episode['tcp_pose'] = np.stack(tcp_pose, axis=0)
    episode['joint_pos'] = np.stack(joint_pos, axis=0)
    episode['action'] = np.stack(action, axis=0)
    replay_buffer.add_episode(episode, compressors='disk')
    episode_id = replay_buffer.n_episodes - 1
    print('Saved episode ', episode_id)

    robot_env.gripper.move(robot_env.gripper.max_width)
    time.sleep(0.5)


def main(args):
    robot_env = RobotEnv(camera_serial=CAM_SERIAL, img_shape=[3]+args.resolution, fps=args.fps)
    zarr_path = os.path.join(args.output, 'replay_buffer.zarr')
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='a')
    while not robot_env.keyboard.quit:
        print("start recording...")
        record(replay_buffer, robot_env)
        if not robot_env.keyboard.quit:
            print("reset the environment...")
            time.sleep(10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-res', '--resolution', nargs='+', type=int)
    parser.add_argument('--fps', type=float, default=10.0)
    args = parser.parse_args()
    main(args)